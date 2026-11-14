"""Canonical-metric re-evaluation of historical 300-node constrained runs.
This DOES NOT claim to have reoptimized the constrained trajectories: policies
are reconstructed from original 2026-11-08 ZIP snapshots, with center residual
verified against original margin-teacher centers. No Test access.
"""
import argparse,csv,json,pathlib,sys,zipfile
import numpy as np,torch
from engine import DT,ExactMaskedWalk,jsd,LN2
from metrics import summarize
from train_margin import prepare, validate
import inspect
if 'tau' not in inspect.signature(validate).parameters:
    from runner_extended import validate
from generate import load_config
HERE=pathlib.Path(__file__).resolve().parent
ROOT=HERE.parents[2] if (HERE.parents[2]/'code/data_generator/config.json').exists() else HERE
ZFILE=str(ROOT/'code/experiments/archive/historical_archive/archives/margin_fixed_centers_complete.zip') if ROOT!=HERE else '/mnt/data/margin_fixed_centers_complete.zip'

def evaluate(z,kind,g,vy,target_centers):
    z=torch.as_tensor(z,dtype=DT)
    with torch.no_grad():
        out=g.train_outputs(z,kind)
        tr=summarize(out,g.labels,tau=.005)
        val=validate(g,z,kind,out.centers,vy,tau=.005)
        distance=torch.linalg.vector_norm(out.centers-target_centers,dim=1)
        return {'train':tr,'validation':val,
            'max_center_l2_from_teacher':float(distance.max()),
            'mean_center_l2_from_teacher':float(distance.mean()),
            'center_JSD_from_teacher':jsd(out.centers,target_centers).tolist()}

def run(archive=ZFILE):
    torch.set_num_threads(1)
    spec=load_config(ROOT/'code/data_generator/config.json')
    x,y,vx,vy,ids=prepare(spec,300,20261107)
    g=ExactMaskedWalk(x,y,spec['gamma'],query_x=vx)
    with zipfile.ZipFile(archive) as f:
        inverse=json.load(f.open('inverse_class_conditional_results.json'))
        teacher_snapshot=json.load(f.open('code/experiments/margin_training/results/full300_tau0.005.json'))
        compact=json.load(f.open('results18.json'))
        finals={n:json.load(f.open(f'margin_fixed_center_results/{n}.json')) for n in ['teacher','compact_teacher','compact_inverse']}
    teacher=teacher_snapshot['Z']
    target=g.train_outputs(torch.as_tensor(teacher,dtype=DT),'shared').centers.detach()
    inverse_class=[x['z_recovered'] for x in inverse['results']]
    cases=[('teacher_shared','shared',teacher,'historical_teacher'),
           ('inverse_class','class',inverse_class,'historical_inverse'),
           ('compacted_teacher','class',[x['Z'] for x in compact['groups']['teacher']['policies']],'historical_compaction_18'),
           ('compacted_inverse','class',[x['Z'] for x in compact['groups']['inverse']['policies']],'historical_compaction_18')]
    cases.extend((f'fixed_margin_{n}','class',finals[n]['policies'],'historical_constrained_margin_12_cycles') for n in finals)
    report={'kind':'benchmark-v1-re-evaluation-historical-snapshots',
            'source_archive':pathlib.Path(archive).name,
            'dataset_seed':spec['seed'],'n_reference':300,'n_validation':len(vy),'test_accessed':False,
            'source_note':'checkpoints only; does not rerun constrained optimizer',
            'center_target':target.tolist(),'runs':{}}
    for name,family,z,source in cases:
        d=evaluate(z,family,g,vy,target)
        d['source']=source;d['family']=family
        report['runs'][name]=d
        print(name,family,'train',round(d['train']['balanced_accuracy'],6),'val',round(d['validation']['balanced_accuracy'],6),
              'Rjs',round(d['train']['R_JS'],9),'max center',d['max_center_l2_from_teacher'],flush=True)
    return report

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--out',default='/mnt/data/bv1_fixed_center_backfill.json');args=a.parse_args()
    result=run()
    p=pathlib.Path(args.out);p.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    table=[{'run':name,'family':d['family'],'source':d['source'],
            'train_BA':d['train']['balanced_accuracy'],
            'val_BA':d['validation']['balanced_accuracy'],
            'I':d['train']['I'],'R_sqrt_JS':d['train']['R_sqrt_JS'],
            'R_JS':d['train']['R_JS'],'R_KL':d['train']['R_KL'],
            'max_center_error_L2':d['max_center_l2_from_teacher']}
           for name,d in result['runs'].items()]
    with p.with_suffix('.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(table[0]));w.writeheader();w.writerows(table)
