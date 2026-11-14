"""Compare Bayes-oracle empirical accuracy and trained policy accuracy on the
EXACT three 72-reference training subsets used in margin-only experiments.

Run: python code/experiments/margin_training/audit_bayes_72.py
No model training, no validation/test selection, no policy modifications.
"""
from pathlib import Path
import csv
import json
import sys
import tempfile

import numpy as np

ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'code/data_generator'))
from generate import load_config,generate,bayes_posterior

def audit():
    spec=load_config(ROOT/'code/data_generator/config.json')
    with tempfile.TemporaryDirectory() as temp:
        train,_,train_y=generate(spec,temp,finite_q=False)['train']
    rows=list(csv.DictReader((HERE/'results/summary.csv').open(newline='',encoding='utf-8')))
    seen_seeds=sorted({int(r['seed']) for r in rows if int(r['nref'])==72})
    out=[]
    for seed in seen_seeds:
        rng=np.random.default_rng(seed)
        idx=np.concatenate([
            rng.choice(np.flatnonzero(train_y==c),24,replace=False)
            for c in range(len(spec['pi']))])
        x,y=train[idx],train_y[idx]
        p=bayes_posterior(spec,x)
        pred=p.argmax(axis=1)
        bal=float(np.mean([(pred[y==c]==c).mean() for c in range(len(spec['pi']))]))
        check=[r for r in rows if int(r['nref'])==72 and int(r['seed'])==seed]
        vals={r['tau']:{
            'train_BA_before':float(r['train_BA_0']),
            'train_BA_after':float(r['train_BA_50'])
        } for r in check}
        out.append({
            'reference_seed':seed,'nref':len(y),
            'class_counts':np.bincount(y,minlength=len(spec['pi'])).tolist(),
            'bayes_empirical_balanced_accuracy':bal,
            'bayes_empirical_ordinary_accuracy':float(np.mean(pred==y)),
            'mean_max_bayes_posterior':float(p.max(axis=1).mean()),
            'policy_accuracy':vals
        })
    return {
        'data_seed':int(spec['seed']),
        'note':'All 72-reference subsets are drawn from ONE fixed train split. Accuracy of models here is IN-SAMPLE and not an out-of-fold performance estimate. Bayes classifier is population-optimal in expectation, not necessarily on this specific labeled training subset.',
        'subsets':out,
        'means':{
            'bayes_empirical_balanced_accuracy':float(np.mean([r['bayes_empirical_balanced_accuracy'] for r in out])),
            'mean_max_bayes_posterior':float(np.mean([r['mean_max_bayes_posterior'] for r in out])),
            'policy_balanced_accuracy_before':float(np.mean([next(iter(r['policy_accuracy'].values()))['train_BA_before'] for r in out])),
            'policy_balanced_accuracy_after_by_tau':{
                tau:float(np.mean([r['policy_accuracy'][tau]['train_BA_after'] for r in out]))
                for tau in sorted(out[0]['policy_accuracy'])
            }
        }
    }

if __name__=='__main__':
    print(json.dumps(audit(),ensure_ascii=False,indent=2))
