"""Exact center inverse for four policy families on a fixed 300-sample graph.

Target is a KNOWN-REACHABLE center obtained from a shared Z, not an arbitrary
Bayes target. Runs zero-init unconstrained inverse and a teacher-replication
witness separately. No labels from Validation/Test enter inverse fitting.
"""
import argparse,csv,hashlib,json,pathlib,sys,time,zipfile
import numpy as np,torch
from engine import DT, ExactMaskedWalk, jsd
from metrics import summarize
from train_margin import prepare,validate
import inspect
if 'tau' not in inspect.signature(validate).parameters:
    from runner_extended import validate
from generate import load_config
HERE=pathlib.Path(__file__).resolve().parent
ROOT=HERE.parents[2] if (HERE.parents[2]/'code/data_generator/config.json').exists() else HERE
FAMILIES=['shared','class','node','node_class']

def teacher_z(core_zip):
    with zipfile.ZipFile(core_zip) as f:
        names=[n for n in f.namelist() if n.endswith('_n300_shared_margin_JSD_tau0.005_s50.json')]
        if len(names)!=1:raise AssertionError(names)
        data=json.load(f.open(names[0]))
    return torch.tensor(data['raw_logits'],dtype=DT)

def broadcast_teacher(z,g,family):
    return {'shared':lambda:z,'class':lambda:z.expand(g.k,-1,-1).clone(),
            'node':lambda:z.expand(g.n,-1,-1).clone(),
            'node_class':lambda:z.expand(g.k,g.n,-1,-1).clone()}[family]()

def fit(g,vy,target,family,steps=50,lr=.12):
    initial=torch.zeros_like(broadcast_teacher(torch.zeros((g.d+1,g.d),dtype=DT),g,family))
    zz=torch.nn.Parameter(initial)
    opt=torch.optim.Adam([zz],lr=lr)
    hist=[];start=time.perf_counter()
    for step in range(steps+1):
        out=g.train_outputs(zz,family)
        C=out.centers
        loss=jsd(C,target).mean()
        if step in (0,steps) or step%10==0:
            with torch.no_grad():
                m=summarize(out,g.labels)
                val=validate(g,zz,family,C.detach(),vy,.005)
                hist.append({'step':step,'center_mean_JSD':float(loss.detach()),
                    'center_max_L2':float(torch.linalg.vector_norm(C-target,dim=1).max().detach()),
                    'train_BA':m['balanced_accuracy'],
                    'validation_BA':val.get('balanced_accuracy'),
                    'I':m['I'],'R_sqrt_JS':m['R_sqrt_JS'],'R_JS':m['R_JS'],'R_KL':m['R_KL'],
                    'walltime_s':time.perf_counter()-start})
        if step==steps:break
        opt.zero_grad(set_to_none=True);loss.backward()
        if not bool(torch.isfinite(zz.grad).all()):raise RuntimeError('nonfinite inverse gradient')
        opt.step()
    return {'family':family,'inverse_objective':'mean_JS_centers_to_teacher',
        'init':'all_zero','steps':steps,'lr':lr,'hist':hist,'final':hist[-1],
        'raw_logits':zz.detach().tolist(),'elapsed_s':time.perf_counter()-start}

def main():
    p=argparse.ArgumentParser();p.add_argument('--steps',type=int,default=50)
    p.add_argument('--core-zip',default=str(ROOT/'code/experiments/standard-policy-comparison/results/core_50_16runs.zip') if ROOT!=HERE else '/mnt/data/bv1_for_commit/core_50_16runs.zip')
    p.add_argument('--out-dir',default='code/experiments/benchmark_v1/results/inverse');args=p.parse_args()
    torch.set_num_threads(1)
    spec=load_config(ROOT/'code/data_generator/config.json')
    x,y,vx,vy,ids=prepare(spec,300,20261107)
    g=ExactMaskedWalk(x,y,spec['gamma'],query_x=vx)
    teacher=teacher_z(args.core_zip)
    center=g.train_outputs(teacher,'shared').centers.detach()
    outpath=pathlib.Path(args.out_dir);outpath.mkdir(exist_ok=True)
    result={'kind':'benchmark-v1-known-reachable-center-inversion',
            'target_from':'margin_JSD_tau0.005_shared_50steps','target_centers':center.tolist(),
            'reference_ids':ids.tolist(),'data_seed':spec['seed'],'test_used':False,'runs':[]}
    for family in FAMILIES:
        witness=broadcast_teacher(teacher,g,family)
        with torch.no_grad():
            witness_out=g.train_outputs(witness,family)
            witness_error=float(jsd(witness_out.centers,center).mean())
        run=fit(g,vy,center,family,args.steps)
        run['teacher_broadcast_center_loss']=witness_error
        run['teacher_broadcast_is_valid']=witness_error<1e-12
        print(family,'inverse JS',run['final']['center_mean_JSD'],'trainBA',run['final']['train_BA'],'VAL',run['final']['validation_BA'],flush=True)
        file=outpath/f'{family}_inverse.json';file.write_text(json.dumps(run,ensure_ascii=False,indent=2)+'\n')
        result['runs'].append({k:v for k,v in run.items() if k!='raw_logits' and k!='hist'})
    (outpath/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    with (outpath/'summary.csv').open('w',newline='') as f:
        ks=['family','steps','center_mean_JSD','center_max_L2','train_BA','validation_BA','R_sqrt_JS','R_JS','R_KL','I','witness_error']
        w=csv.DictWriter(f,fieldnames=ks);w.writeheader()
        for run in result['runs']:
            x=run['final'];w.writerow({'family':run['family'],'steps':run['steps'],
                'witness_error':run['teacher_broadcast_center_loss'],**{k:x.get(k) for k in ks if k in x}})
if __name__=='__main__':main()
