"""E-CH4-14: closed-by-default, exact-walk Softplus-JS 36-cell runner.

Ordinary CLI is dry-run. Research fits require a separate reviewed workflow,
Issue #90 authorization, and an exclusive budget ledger. Never reads Test.
"""
from __future__ import annotations

import argparse
import csv
from hashlib import sha256
import io
import json
import os
from pathlib import Path
import platform
import sys
import time

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT/'code/experiments/benchmark_v1'))
sys.path.insert(0, str(ROOT/'code/data_generator'))
from engine import DT, ExactMaskedWalk, balanced, jsd, LN2
from train_missing_objectives import train_loss
from data import SUBSETS, NoiseSpec, load_mother, derive_all, fingerprint, spec_sha, file_sha

FAMILIES = ('shared', 'class')
OBJECTIVE = 'softplus_JS_tau005'
UPDATES = 30
LR = 0.12
BETA, LAMBDA, H, K = 0.4, 0.8, 12, 3
JOB_GRID = tuple((subset, family) for subset, _ in SUBSETS for family in FAMILIES)
EXEC_WORKFLOW = 'E-CH4-14 authorized 36-fit run'  # No such workflow is installed in this mission.
BRANCH = 'experiment/ch4-14-noise-sensitivity-01'


def grid():
    if (len(JOB_GRID) != 36 or len(set(JOB_GRID)) != 36
            or set(f for _, f in JOB_GRID) != set(FAMILIES)):
        raise AssertionError('Preregistered job grid changed')
    return JOB_GRID


def shape_for(d, family):
    if not 3 <= d <= 8 or family not in FAMILIES:
        raise ValueError('Only d=3..8 and shared/class allowed')
    return (d + 1, d) if family == 'shared' else (K, d + 1, d)


def initial_logits(d, family):
    return torch.nn.Parameter(torch.zeros(shape_for(d, family), dtype=DT))


def prediction_counts(scores, y, *, k=K):
    """Labels used exclusively for diagnostics, never for graph query/Adam."""
    pred = scores.argmax(dim=1)
    correct = (pred == y).to(DT)
    ba = balanced(correct, y, k)
    return float(ba), float(correct.mean())


def validation_scores(graph, z, family):
    """Candidate-hypothesis query scoring; does not consume Validation labels."""
    if graph.qx is None:
        raise ValueError('Held-out Validation query graph missing')
    train = graph.train_outputs(z, family)
    centers = train.centers
    scores = []
    if family == 'shared':
        rep = graph.represent_query(z, family)
        scores = [1 - jsd(rep, centers[c]) / LN2 for c in range(K)]
    elif family == 'class':
        scores = [1 - jsd(graph.represent_query(z, family, c), centers[c]) / LN2
                  for c in range(K)]
    else:
        raise ValueError('Only shared and class may predict query labels')
    return torch.stack(scores, dim=1)


def fit_trace(graph, validation_labels, family, *, steps=UPDATES, fixture=False):
    """Exactly steps Adam updates, observed at 0..steps, with no held-out gradient.

    In production this function is only reached from execute(), after all gates.
    Direct calls require fixture=True and a tiny graph; 300-row fits cannot be
    run through the test interface.
    """
    if not fixture:
        raise PermissionError('Research fit may only run via E14 authorized execute()')
    if graph.n > 12 or len(graph.qx) > 8 or graph.d > 8:
        raise PermissionError('Fixture path cannot train on E14 research data')
    if steps not in range(1, UPDATES + 1):
        raise ValueError('Out-of-bounds update count')
    return _fit_trace(graph, validation_labels, family, steps=steps)


def _fit_trace(graph, validation_labels, family, *, steps, update_callback=None):
    if steps != UPDATES and (graph.n > 12 or len(graph.qx) > 8):
        raise ValueError('Real research job must use exactly 30 updates')
    if family not in FAMILIES or graph.k != K:
        raise ValueError('Invalid training family/class count')
    vy = torch.as_tensor(validation_labels, dtype=torch.long)
    if graph.qx is None or vy.shape != (len(graph.qx),):
        raise ValueError('Validation labels shape mismatch')
    if not bool(((vy >= 0) & (vy < K)).all()):
        raise ValueError('Validation class index invalid')
    z = initial_logits(graph.d, family)
    optimizer = torch.optim.Adam([z], lr=LR)
    trace = []
    previous_loss = None
    update_l2 = None
    started = time.perf_counter()
    for step in range(steps + 1):
        optimizer.zero_grad(set_to_none=True)
        loss = train_loss(graph, z, family, OBJECTIVE)  # Canonical, not reimplemented.
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError(f'Nonfinite loss at step {step}')
        loss.backward()
        if z.grad is None or not bool(torch.isfinite(z.grad).all()):
            raise FloatingPointError(f'Nonfinite gradient at step {step}')
        with torch.no_grad():
            train = graph.train_outputs(z, family)
            train_ba, train_acc = prediction_counts(train.scores, graph.labels)
            val_scores = validation_scores(graph, z, family)
            val_ba, val_acc = prediction_counts(val_scores, vy)
            value = float(loss.detach())
            row = {
                'step': step,
                'update_completed': step,
                'train_softplus_loss': value,
                'train_BA': train_ba, 'train_Accuracy': train_acc,
                'validation_BA': val_ba, 'validation_Accuracy': val_acc,
                'gradient_l2': float(torch.linalg.vector_norm(z.grad)),
                'gradient_linf': float(z.grad.abs().max()),
                'logits_l2': float(torch.linalg.vector_norm(z)),
                'adam_update_l2': update_l2,
                'loss_delta': None if previous_loss is None else value - previous_loss,
                'elapsed_s': time.perf_counter() - started,
                'numeric_status': 'finite',
            }
        if not all(np.isfinite(v) for v in row.values() if isinstance(v, float)):
            raise FloatingPointError('Nonfinite trace scalar')
        trace.append(row)
        previous_loss = value
        if step < steps:
            old = z.detach().clone()
            optimizer.step()
            if not bool(torch.isfinite(z).all()):
                raise FloatingPointError(f'Nonfinite Adam update {step+1}')
            update_l2 = float(torch.linalg.vector_norm(z.detach() - old))
            if update_callback is not None:
                update_callback(step+1)  # durable budget ledger after each completed update
    if len(trace) != steps + 1 or trace[0]['adam_update_l2'] is not None:
        raise AssertionError('Trace step alignment error')
    return z, optimizer, trace


def trace_as_csv(trace):
    if not trace:
        raise ValueError('No trace rows')
    b = io.StringIO()
    writer = csv.DictWriter(b, fieldnames=list(trace[0]), lineterminator='\n')
    writer.writeheader()
    writer.writerows(trace)
    return b.getvalue()


def new_file(path, text):
    path = Path(path)
    if not path.parent.is_dir():
        raise FileNotFoundError(path.parent)
    with path.open('x', encoding='utf8') as f:
        f.write(text)
    return file_sha(path)


def auth_gate(auth, *, source_sha, run_token, spec_digest):
    """Fail closed. The existing no-fit CI can never satisfy this gate."""
    if (os.environ.get('GITHUB_ACTIONS') != 'true'
            or os.environ.get('GITHUB_REF_NAME') != BRANCH
            or os.environ.get('GITHUB_WORKFLOW') != EXEC_WORKFLOW
            or os.environ.get('E_CH4_14_FIT_AUTHORIZED') != 'true'
            or os.environ.get('GITHUB_SHA') != source_sha):
        raise PermissionError('No approved independent E14 execution workflow')
    required = {'issue':90, 'authorized_fits':36, 'updates_per_fit':30,
                'test_accessed':False, 'source_sha':source_sha,
                'data_spec_sha256':spec_digest,
                'workflow_run_id':os.environ.get('GITHUB_RUN_ID')}
    if any(auth.get(k) != v for k, v in required.items()):
        raise PermissionError('Execution approval does not match this source/data/run')
    for key in ('owner_approval_url', 'independent_qa_url'):
        value = auth.get(key)
        if not isinstance(value,str) or not value.startswith('https://github.com/mahhammad/My-Thesis/issues/90#issuecomment-'):
            raise PermissionError('Missing owner and independent QA evidence')
    if not run_token or sha256(run_token.encode()).hexdigest() != auth.get('one_time_token_sha256'):
        raise PermissionError('Missing one-time execution token')
    if not auth.get('budget_reservation') or not auth.get('owner_approved_data_sha256'):
        raise PermissionError('No 36-fit budget/data reservation')
    return True


def plan():
    return {'schema':'e-ch4-14-grid-v1','status':'DRY_RUN_NO_FIT',
            'mother_seed':20261102, 'n_train':300, 'n_validation':90,
            'subset_count':18, 'fits':len(grid()),'updates_per_fit':UPDATES,
            'total_update_budget':len(grid())*UPDATES,
            'families':list(FAMILIES),'objective':OBJECTIVE,
            'beta':BETA,'lambda':LAMBDA,'H':H,'tau':.005,'Adam_lr':LR,
            'job_ids':[f'{s}__{f}' for s,f in grid()], 'test_accessed':False}


def read_locked_assets(spec, auth):
    """Read-only; data-generation stages must have already supplied immutable files."""
    from data import HERE as EXP
    rawdir, maskdir = EXP/'inputs/noise_raw', EXP/'inputs/noise_masks'
    paths = [(rawdir/'new_noise_seed20261102.npz', rawdir/'manifest.json'),
             (maskdir/'new_noise_masks.npz', maskdir/'manifest.json')]
    buffers = []
    for path, manpath in paths:
        manifest = json.loads(manpath.read_text(encoding='utf8'))
        if (manifest.get('spec_sha256') != spec_sha(spec)
                or manifest.get('test_accessed') is not False
                or manifest.get('source_mother_sha256') != auth['owner_approved_data_sha256']
                or manifest.get('file_sha256') != file_sha(path)):
            raise ValueError('Locked data/approval mismatch')
        with np.load(path, allow_pickle=False) as f:
            if set(f.files) != set(manifest['array_sha256']):
                raise ValueError('Unexpected NPZ keys')
            arrays = {k:f[k].copy() for k in f.files}
        if any(fingerprint(a)!=manifest['array_sha256'][k] for k,a in arrays.items()):
            raise ValueError('Locked asset array SHA mismatch')
        buffers.append(arrays)
    raw, masks = buffers
    if set(raw)!= {'train_noise','validation_noise'} or set(masks)!= {'train_mask','validation_mask'}:
        raise ValueError('Unexpected asset array names')
    return raw['train_noise'],raw['validation_noise'],masks['train_mask'],masks['validation_mask']


def execute(authorization, source_sha, run_token):
    """Explicit batch entry. No PR, no Test; reservation stops any repeat/retry."""
    if len(source_sha) != 40 or any(c not in '0123456789abcdef' for c in source_sha):
        raise ValueError('Full Git source commit required')
    config = json.loads(Path(authorization).read_text(encoding='utf8'))
    spec = NoiseSpec(**config['noise_spec']).validate()
    auth_gate(config,source_sha=source_sha,run_token=run_token,spec_digest=spec_sha(spec))
    # Never generate data in this function. All assets must be locked in prior missions.
    mother = load_mother()
    if mother.source_sha != config['owner_approved_data_sha256']:
        raise ValueError('Unexpected mother source hash')
    raws = read_locked_assets(spec,config)
    old_gamma = json.loads((ROOT/'code/data_generator/config.json').read_text())['kernel_gamma']
    datasets=derive_all(mother,*raws,old_gamma=old_gamma,spec=spec)
    by_id={d.subset_id:d for d in datasets}
    if len(by_id)!=18:
        raise ValueError('Missing derived subset')
    # Explicit reservation is created before the first fit and is NEVER reused.
    ledger_dir=HERE/'results/metrics'
    ledger_path=ledger_dir/'e14_spent_budget.jsonl'
    if ledger_path.exists() or any((HERE/'results/traces'/f'{s}__{f}').exists() for s,f in grid()):
        raise FileExistsError('E14 results/budget already exist; no implicit retry')
    if not ledger_dir.is_dir():
        raise FileNotFoundError(ledger_dir)
    reports = {}
    with ledger_path.open('x',encoding='utf8') as ledger:
        ledger.write(json.dumps({'event':'BUDGET_RESERVED','fits':36,'updates':1080,
                                 'source_sha':source_sha,'test_accessed':False})+'\n')
        ledger.flush(); os.fsync(ledger.fileno())
        for subset, family in grid():
            run_id=f'{subset}__{family}'
            d=by_id[subset]
            try:
                job_started=time.perf_counter()
                graph=ExactMaskedWalk(d.train_x,d.train_y,d.gamma,
                                      beta=BETA,lam=LAMBDA,horizon=H,
                                      query_x=d.validation_x)
                trace_dir=HERE/'results/traces'/run_id
                checkpoint_dir=HERE/'results/checkpoints'/run_id
                trace_dir.mkdir(exist_ok=False)
                checkpoint_dir.mkdir(exist_ok=False)
                ledger.write(json.dumps({'event':'START','run_id':run_id})+'\n');ledger.flush(); os.fsync(ledger.fileno())
                def record_update(completed):
                    ledger.write(json.dumps({'event':'UPDATE','run_id':run_id,
                                             'updates_completed':completed})+'\n')
                    ledger.flush(); os.fsync(ledger.fileno())
                z,opt,trace=_fit_trace(graph,d.validation_y,family,
                                      steps=UPDATES,update_callback=record_update)
                from checkpoints import save_checkpoint
                from e14_metrics import report_graph
                meta={'subset_id':subset,'family':family,'data_sha256':d.hashes['train_x_sha256'],
                      'validation_sha256':d.hashes['validation_x_sha256'],
                      'feature_mapping':list(d.feature_mapping), 'gamma':d.gamma.tolist(),
                      'objective':OBJECTIVE,'tau':.005,'beta':BETA,'lambda':LAMBDA,
                      'H':H,'lr':LR,'source_commit':source_sha,
                      'torch':str(torch.__version__),'numpy':str(np.__version__),
                      'trace_sha256':None,'test_accessed':False}
                trace_record={'schema':'e-ch4-14-trace-v1','run_id':run_id,'trace':trace,
                              'test_accessed':False}
                trace_text=json.dumps(trace_record,indent=2,allow_nan=False)+'\n'
                trace_sha=new_file(trace_dir/'trace.json',trace_text)
                new_file(trace_dir/'trace.csv',trace_as_csv(trace))
                meta['trace_sha256']=trace_sha
                save_checkpoint(checkpoint_dir/'step_30.pt',z,opt,UPDATES,meta)
                report=report_graph(graph,z,family,d.train_y,d.validation_y,
                                    data_hashes=d.hashes,subset_id=subset)
                import psutil  # only the approved research workflow installs RSS monitoring
                report['resource_usage'] = {
                    'job_compute_elapsed_s': float(time.perf_counter()-job_started),
                    'process_rss_mib_at_report': float(
                        psutil.Process().memory_info().rss/(1024*1024)),
                    'policy_parameter_count': int(z.numel()),
                    'rss_measurement': 'process_current_not_peak',
                }
                reports[run_id] = report
                new_file(ledger_dir/f'{run_id}.json',json.dumps(report,indent=2,allow_nan=False)+'\n')
                ledger.write(json.dumps({'event':'SUCCESS','run_id':run_id,
                                         'trace_sha256':trace_sha})+'\n');ledger.flush(); os.fsync(ledger.fileno())
            except Exception as e:
                ledger.write(json.dumps({'event':'FAIL_STOP','run_id':run_id,
                                         'error_type':type(e).__name__,
                                         'message':str(e)})+'\n');ledger.flush(); os.fsync(ledger.fileno())
                raise
    if len(reports) != 36:
        raise AssertionError('Batch incomplete; no final summary')
    from analysis import summarize_reports
    descriptive = summarize_reports(reports)  # Pure postprocessing of saved reports.
    new_file(ledger_dir/'metrics_full.json',
             json.dumps({'schema':'e-ch4-14-metrics-batch-v1',
                         'test_accessed':False,'runs':reports},allow_nan=False,indent=2)+'\n')
    new_file(ledger_dir/'coverage.json',
             json.dumps({'schema':'e-ch4-14-coverage-v1',
                         'runs':{key:report['coverage'] for key,report in reports.items()}},
                        allow_nan=False,indent=2)+'\n')
    new_file(ledger_dir/'metrics_summary.csv', trace_as_csv(descriptive['individual']))
    new_file(ledger_dir/'metrics_grouped.csv', trace_as_csv(descriptive['grouped']))
    new_file(ledger_dir/'metrics_paired.csv', trace_as_csv(descriptive['paired']))
    new_file(ledger_dir/'metrics_paired_by_k.csv', trace_as_csv(descriptive['paired_by_k']))
    new_file(ledger_dir/'metrics_analysis.json',
             json.dumps(descriptive, indent=2, allow_nan=False) + '\n')
    new_file(ledger_dir/'provenance.json',
             json.dumps({'schema':'e-ch4-14-provenance-v1','source_sha':source_sha,
                         'mother_sha256':mother.source_sha,'data_spec_sha256':spec_sha(spec),
                         'n_fits':36,'updates_per_fit':30,'test_accessed':False},indent=2)+'\n')
    return ledger_path


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--authorization',type=Path)
    parser.add_argument('--source-sha',default='')
    args=parser.parse_args(argv)
    if not args.execute:
        print(json.dumps(plan(),indent=2,ensure_ascii=False))
        return 0
    if not args.authorization:
        raise PermissionError('--execute requires separate Issue #90 authorization')
    execute(args.authorization,args.source_sha,
            os.environ.get('E_CH4_14_EXECUTION_TOKEN',''))
    return 0

if __name__=='__main__':
    raise SystemExit(main())
