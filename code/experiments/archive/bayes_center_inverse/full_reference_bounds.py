#!/usr/bin/env python3
"""Fast rigorous coordinate bounds for 300-reference graph without return exclusion.

Relaxing the forbidden-origin constraint makes the reachable set larger,
so a target coordinate outside the relaxed bound is certainly unattainable
under the actual no-origin-return process. Runs in O(K H E), O(E) space.
"""
import argparse
import json
import tempfile
import time
from pathlib import Path
import numpy as np
from run_bayes import ROOT, load_config, generate, bayes_class_centers, coefficients
from feasibility_bound import origin_return_relaxed_bounds


def measure(seed=20261107, sizes=(72, 120, 300), beta=.4, lam=.8, H=12):
    spec=load_config(ROOT / 'code/data_generator/config.json')
    with tempfile.TemporaryDirectory() as temp:
        splits=generate(spec,temp,finite_q=False)
    x,y=splits['train'][0],splits['train'][2]
    target,_=bayes_class_centers(spec,x,y)
    outcome={'meta':{'data_seed':spec['seed'],'reference_selection_seed':seed,'beta':beta,'lambda':lam,'H':H,
                     'relaxation':'return-to-origin allowed; feature choice can be source/time dependent'},
             'targets':target.tolist(),'by_reference_size':{}}
    alpha=coefficients(beta,lam,H)
    for size in sizes:
        if size==len(x):
            idx=np.arange(len(x))
        else:
            if size%len(spec['pi']):
                raise ValueError('subset size must be multiple of K')
            rng=np.random.default_rng(seed)
            idx=np.concatenate([rng.choice(np.flatnonzero(y==c),size//len(spec['pi']),replace=False)
                                for c in range(len(spec['pi']))])
            rng.shuffle(idx)
        start=time.perf_counter()
        bounds=origin_return_relaxed_bounds(x[idx],y[idx],spec['gamma'],alpha)
        elapsed=time.perf_counter()-start
        low,high=np.asarray(bounds['lower']),np.asarray(bounds['upper'])
        outside=(target < low-1e-12)|(target > high+1e-12)
        outcome['by_reference_size'][str(size)]={'bounds':bounds,
             'time_s':elapsed,'reference_indices':idx.tolist(),
             'diagonal_target':np.diag(target).tolist(),
             'diagonal_upper':np.diag(high).tolist(),
             'proved_infeasible_rows':[int(c) for c in np.flatnonzero(outside.any(axis=1))],
             'diagonal_gap_to_upper':(np.diag(target)-np.diag(high)).tolist()}
    return outcome


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--out',type=Path,default=Path(__file__).parent/'results/full_reference_bounds.json')
    args=p.parse_args()
    v=measure()
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(v,indent=2,ensure_ascii=False)+'\n')
    for n,x in v['by_reference_size'].items():
        print(n,'diag upper',np.round(x['diagonal_upper'],5),
              'proven unattainable classes',x['proved_infeasible_rows'])
    print('saved',args.out)

if __name__=='__main__':main()
