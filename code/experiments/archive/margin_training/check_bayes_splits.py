"""Diagnose whether the fixed synthetic validation split is unusually easy.
Uses true data-generating parameters: diagnostic oracle ONLY, never policy training.
Run from repo root: python code/experiments/margin_training/check_bayes_splits.py
"""
from pathlib import Path
import json
import sys
import tempfile
import numpy as np

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'code/data_generator'))
from generate import load_config, generate, bayes_posterior

def main():
    spec=load_config(ROOT/'code/data_generator/config.json')
    with tempfile.TemporaryDirectory() as temp:
        splits=generate(spec,temp,finite_q=False)
    results={}
    for name,(x,mask,y) in splits.items():
        posterior=bayes_posterior(spec,x)
        pred=posterior.argmax(axis=1)
        balanced=float(np.mean([(pred[y==c]==c).mean() for c in range(len(spec['pi']))]))
        results[name]={
            'n_samples':int(len(y)),
            'class_counts':np.bincount(y,minlength=len(spec['pi'])).tolist(),
            'bayes_empirical_balanced_accuracy':balanced,
            'bayes_empirical_ordinary_accuracy':float(np.mean(pred==y))
        }
    print(json.dumps({'data_seed':spec['seed'],'splits':results,
        'note':'Uses true generative Bayes classifier only for split difficulty diagnostic; not an unbiased generalization gap.'},
        ensure_ascii=False,indent=2))
if __name__=='__main__':
    main()
