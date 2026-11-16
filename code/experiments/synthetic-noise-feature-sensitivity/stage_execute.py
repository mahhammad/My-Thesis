"""E-CH4-14 staged artifacts: verify stored inputs before each authorized phase.

Research writes are only reached AFTER the online execution_gate reservation.
The CLI 'preflight' performs no writes and never opens a Test split.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
from data import (NoiseSpec, SUBSETS, derived_manifest, derive_all,
                  file_sha, fingerprint, generate_noise_masks,
                  generate_raw_noise, load_mother, save_npz_once, spec_sha)

RAWDIR = HERE/'inputs/noise_raw'
MASKDIR = HERE/'inputs/noise_masks'
DERIVED = HERE/'inputs/derived'
RAWFILE = RAWDIR/'new_noise_seed20261102.npz'
MASKFILE = MASKDIR/'new_noise_masks.npz'
RAWKEYS = ('train_noise','validation_noise')
MASKKEYS = ('train_mask','validation_mask')
DERIVEDKEYS = ('train_x','train_y','validation_x','validation_y')


def load_context(path, stage):
    path=Path(path).resolve()
    runner_temp=os.environ.get('RUNNER_TEMP')
    if not runner_temp or path.parent != Path(runner_temp).resolve():
        raise PermissionError('Context must be in trusted runner temp')
    ctx=json.loads(path.read_text('utf8'))
    if (ctx.get('stage') != stage or ctx.get('test_accessed') is not False
            or ctx.get('issue') != 90
            or ctx.get('source_sha') != os.environ.get('GITHUB_SHA')
            or ctx.get('budget_reservation') !=
                    'refs/tags/e-ch4-14-'+stage.replace('_','-')+'-budget-spent'):
        raise PermissionError('Unverified stage context')
    if not ctx.get('owner_approval_url') or not ctx.get('independent_qa_url'):
        raise PermissionError('No independently checked evidence')
    spec=NoiseSpec(**ctx['noise_spec']).validate()
    if spec_sha(spec) != ctx.get('approved_spec_sha256'):
        raise PermissionError('Drifted owner-approved NoiseSpec')
    return ctx,spec


def ensure_new(path):
    if Path(path).exists() or Path(path).is_symlink():
        raise FileExistsError('Immutable artifact already present: '+str(path))


def new_manifest(path, record):
    ensure_new(path)
    if not path.parent.is_dir():
        raise FileNotFoundError(path.parent)
    with path.open('x',encoding='utf8') as f:
        json.dump(record,f,indent=2,sort_keys=True,allow_nan=False)
        f.write('\n')


def asset_read(path, manifest_path, keys, ctx, *, stage):
    manifest=json.loads(Path(manifest_path).read_text('utf8'))
    if (manifest.get('schema') != 'e-ch4-14-'+stage+'-manifest-v1'
            or manifest.get('test_accessed') is not False
            or manifest.get('spec_sha256') != ctx['approved_spec_sha256']
            or manifest.get('source_mother_sha256') != ctx['owner_approved_data_sha256']
            or manifest.get('file_sha256') != file_sha(path)
            or set(manifest.get('array_sha256',{})) != set(keys)):
        raise ValueError('Immutable '+stage+' artifact/manifest provenance mismatch')
    with np.load(path,allow_pickle=False) as payload:
        if set(payload.files) != set(keys):
            raise ValueError('Unexpected '+stage+' NPZ keys: Test forbidden')
        arr={key:payload[key].copy() for key in keys}
    for key in keys:
        expected=(300,3) if key.startswith('train') else (90,3)
        dt='bool' if stage == 'noise_masks' else 'float64'
        if arr[key].shape != expected or arr[key].dtype != np.dtype(dt):
            raise ValueError('Unexpected staged '+key+' dtype/shape')
        if fingerprint(arr[key]) != manifest['array_sha256'][key]:
            raise ValueError('Staged array hash mismatch '+key)
    if stage == 'raw_noise' and any(not np.isfinite(v).all() for v in arr.values()):
        raise ValueError('Raw noise contains missing/nonfinite values')
    return arr,manifest


def originals(ctx):
    m=load_mother()
    if m.source_sha != ctx['owner_approved_data_sha256']:
        raise ValueError('Mother frozen SHA drift')
    return m


def old_gamma():
    config=json.loads((ROOT/'code/data_generator/config.json').read_text('utf8'))
    v=config['kernel_gamma']
    if len(v) != 5:
        raise ValueError('Mother gamma dimensionality changed')
    return v


def _derived_valid(d, ctx):
    if d.subset_id not in {sid for sid,_ in SUBSETS}:
        raise AssertionError('Unregistered subset')
    if d.train_x.shape!=(300,len(d.feature_mapping)) or d.validation_x.shape!=(90,len(d.feature_mapping)):
        raise ValueError('Derived dimensions not locked')
    if d.hashes.get('source_mother_sha256')!=ctx['owner_approved_data_sha256']:
        raise ValueError('Wrong derived parent')
    if d.hashes.get('spec_sha256')!=ctx['approved_spec_sha256']:
        raise ValueError('Wrong derived NoiseSpec')
    if d.subset_id=='N2-01' and list(d.feature_mapping)!=[0,1,2,3,4]:
        raise ValueError('N2-01 mapping must reproduce original 5 features')


def verify_derived_files(ctx, spec, *, mother=None, raw=None, masks=None):
    if mother is None:
        mother=originals(ctx)
    if raw is None:
        raw,_=asset_read(RAWFILE,RAWDIR/'manifest.json',RAWKEYS,ctx,stage='raw_noise')
    if masks is None:
        masks,_=asset_read(MASKFILE,MASKDIR/'manifest.json',MASKKEYS,ctx,stage='noise_masks')
    predicted=derive_all(mother,raw['train_noise'],raw['validation_noise'],
                          masks['train_mask'],masks['validation_mask'],
                          old_gamma=old_gamma(),spec=spec)
    manifest_path=DERIVED/'manifest.json'
    manifest=json.loads(manifest_path.read_text('utf8'))
    if (manifest.get('schema') != 'e-ch4-14-derived-v1'
            or manifest.get('test_accessed') is not False
            or manifest.get('n_train')!=300 or manifest.get('n_validation')!=90
            or len(manifest.get('datasets',[]))!=18):
        raise ValueError('Derived manifest missing or inconsistent')
    if (manifest.get('spec_sha256')!=ctx['approved_spec_sha256']
            or manifest.get('source_mother_sha256')!=mother.source_sha
            or manifest.get('raw_file_sha256')!=file_sha(RAWFILE)
            or manifest.get('mask_file_sha256')!=file_sha(MASKFILE)):
        raise ValueError('Derived manifest broken input linkage')
    for d,entry in zip(predicted,manifest['datasets']):
        _derived_valid(d,ctx)
        if (entry['subset_id']!=d.subset_id or
                entry['feature_mapping']!=list(d.feature_mapping) or
                entry['hashes']!=d.hashes):
            raise ValueError('Derived mapping/array fingerprint metadata drift')
        path=DERIVED/(d.subset_id+'.npz')
        if file_sha(path)!=entry['file_sha256']:
            raise ValueError('Derived NPZ file checksum mismatch: '+d.subset_id)
        with np.load(path,allow_pickle=False) as stored:
            if set(stored.files)!=set(DERIVEDKEYS):
                raise ValueError('Extra derived NPZ keys: Test forbidden')
            for field in DERIVEDKEYS:
                got=stored[field]
                original=getattr(d,field)
                if (got.dtype!=original.dtype or got.shape!=original.shape or
                        got.tobytes(order='C')!=original.tobytes(order='C')):
                    raise ValueError('Saved derived array differs from deterministic projection')
    five=next(d for d in predicted if d.subset_id=='N2-01')
    if (five.train_x.tobytes()!=mother.train_x.tobytes() or
            five.validation_x.tobytes()!=mother.validation_x.tobytes()):
        raise ValueError('Bit-level N2=mother validation failed')
    return predicted


def preflight(stage,ctx,spec):
    """Never consumes reservation/budget or opens a Test split."""
    mother=originals(ctx)
    if stage=='raw_noise':
        ensure_new(RAWFILE);ensure_new(RAWDIR/'manifest.json')
    elif stage=='noise_masks':
        asset_read(RAWFILE,RAWDIR/'manifest.json',RAWKEYS,ctx,stage='raw_noise')
        ensure_new(MASKFILE);ensure_new(MASKDIR/'manifest.json')
    elif stage=='derived':
        raw,_=asset_read(RAWFILE,RAWDIR/'manifest.json',RAWKEYS,ctx,stage='raw_noise')
        masks,_=asset_read(MASKFILE,MASKDIR/'manifest.json',MASKKEYS,ctx,stage='noise_masks')
        if len(raw)!=2 or len(masks)!=2:
            raise ValueError('Incomplete stage inputs')
        ensure_new(DERIVED/'manifest.json')
        for subset_id,_ in SUBSETS:
            ensure_new(DERIVED/(subset_id+'.npz'))
    elif stage=='fit':
        verify_derived_files(ctx,spec,mother=mother)
        results=HERE/'results/metrics/e14_spent_budget.jsonl'
        ensure_new(results)
        for subset_id,_ in SUBSETS:
            for family in ('shared','class'):
                ensure_new(HERE/'results/traces'/f'{subset_id}__{family}')
                ensure_new(HERE/'results/checkpoints'/f'{subset_id}__{family}')
    else:
        raise PermissionError('Unsupported E14 stage')
    print('E14_STAGE_PREFLIGHT_PASS',stage,'Train=300','Validation=90','Test=false')


def _manifest(ctx,stage,arrays,path):
    return dict(schema='e-ch4-14-'+stage+'-manifest-v1',
                test_accessed=False,source_mother_sha256=ctx['owner_approved_data_sha256'],
                spec_sha256=ctx['approved_spec_sha256'],
                approval_source_sha=ctx['source_sha'],
                owner_approval_url=ctx['owner_approval_url'],
                independent_qa_url=ctx['independent_qa_url'],
                file=path.name,file_sha256=file_sha(path),
                array_sha256={k:fingerprint(v) for k,v in arrays.items()},
                n_train=300,n_validation=90,
                created_utc=datetime.now(timezone.utc).isoformat(),
                numpy=np.__version__,python=platform.python_version())


def run_data_stage(stage,ctx,spec):
    """Stage has already been checked AND globally reserved on GitHub."""
    if stage=='raw_noise':
        m=originals(ctx)
        train,val=generate_raw_noise(spec,300,90,approval=ctx)
        arrays={'train_noise':train,'validation_noise':val}
        save_npz_once(RAWFILE,arrays,stage=stage,schema='e-ch4-14-raw-noise-v1',
                      approval=ctx,spec=spec)
        rec=_manifest(ctx,stage,arrays,RAWFILE)
        rec.update(distribution=spec.distribution,raw_seed=spec.raw_seed,
                   std=list(spec.std),gamma=list(spec.gamma))
        new_manifest(RAWDIR/'manifest.json',rec)
    elif stage=='noise_masks':
        before=file_sha(RAWFILE)
        asset_read(RAWFILE,RAWDIR/'manifest.json',RAWKEYS,ctx,stage='raw_noise')
        train,val=generate_noise_masks(spec,300,90,approval=ctx)
        arrays={'train_mask':train,'validation_mask':val}
        save_npz_once(MASKFILE,arrays,stage=stage,schema='e-ch4-14-noise-masks-v1',
                      approval=ctx,spec=spec)
        rec=_manifest(ctx,stage,arrays,MASKFILE)
        rec.update(mask_seed=spec.mask_seed,missing_rate=list(spec.missing_rate),
                   raw_file_sha256=before,contract_true_means='observed')
        new_manifest(MASKDIR/'manifest.json',rec)
        if file_sha(RAWFILE)!=before:
            raise ValueError('Frozen raw noise modified during mask stage')
    elif stage=='derived':
        mother=originals(ctx)
        raw,_=asset_read(RAWFILE,RAWDIR/'manifest.json',RAWKEYS,ctx,stage='raw_noise')
        masks,_=asset_read(MASKFILE,MASKDIR/'manifest.json',MASKKEYS,ctx,stage='noise_masks')
        datasets=derive_all(mother,raw['train_noise'],raw['validation_noise'],
                            masks['train_mask'],masks['validation_mask'],
                            old_gamma=old_gamma(),spec=spec)
        manifest=derived_manifest(datasets)
        manifest.update(spec_sha256=ctx['approved_spec_sha256'],
                        source_mother_sha256=mother.source_sha,
                        raw_file_sha256=file_sha(RAWFILE),
                        mask_file_sha256=file_sha(MASKFILE))
        for d,entry in zip(datasets,manifest['datasets']):
            _derived_valid(d,ctx)
            arrays={key:getattr(d,key) for key in DERIVEDKEYS}
            path=DERIVED/(d.subset_id+'.npz')
            save_npz_once(path,arrays,stage=stage,
                          schema='e-ch4-14-derived-data-v1',
                          approval=ctx,spec=spec)
            entry['file_sha256']=file_sha(path)
        new_manifest(DERIVED/'manifest.json',manifest)
        verify_derived_files(ctx,spec,mother=mother,raw=raw,masks=masks)
    else:
        raise ValueError('This command only creates staged data assets')
    print('E14_STAGE_CREATED',stage,'Test=false','immutable=True')


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',choices=('raw_noise','noise_masks','derived','fit'),required=True)
    p.add_argument('--context',type=Path,required=True)
    p.add_argument('--mode',choices=('preflight','execute'),required=True)
    a=p.parse_args(argv)
    ctx,spec=load_context(a.context,a.stage)
    if a.mode=='preflight':
        preflight(a.stage,ctx,spec)
        return 0
    if a.stage=='fit':
        raise ValueError('Training is only entered through separately authorized runner.py')
    # Local stage preflight repeated immediately before writing, after online reservation.
    preflight(a.stage,ctx,spec)
    run_data_stage(a.stage,ctx,spec)
    return 0


if __name__=='__main__':
    raise SystemExit(main())
