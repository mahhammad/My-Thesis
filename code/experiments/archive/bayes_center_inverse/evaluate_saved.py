#!/usr/bin/env python3
"""Recompute exact/truncated propagation from policies stored in a raw run JSON."""
import argparse
import json
import tempfile
from pathlib import Path
import numpy as np
import torch
from run_bayes import ROOT, DT, ExactGraph, generate, load_config
from matrix_walk import approximation_diagnostic


def evaluate(path):
    path = Path(path)
    saved = json.loads(path.read_text())
    spec = load_config(ROOT / 'code/data_generator/config.json')
    with tempfile.TemporaryDirectory() as tmp:
        data = generate(spec, tmp, finite_q=False)
    chosen = np.array(saved['meta']['nref_indices_in_original_training'], dtype=int)
    x, y = data['train'][0][chosen], data['train'][2][chosen]
    g = ExactGraph(x, y, spec['gamma'], np.arange(len(y)), data['validation'][0],
                   beta=saved['meta']['beta'], lam=saved['meta']['lambda'],
                   horizon=saved['meta']['horizon'])
    result = {'source_run': path.name,
              'alpha': g.coeff.numpy().tolist(),
              'approx': {}}
    for method in saved['methods']:
        if method == 'uniform':
            zs = [torch.zeros((g.d + 1, g.d), dtype=DT)] * g.k
        elif method == 'shared_exact_adam':
            # stored in comparison only as shared final centers; reconstruct from
            # its optimizer separately if desired, not guessed here.
            continue
        else:
            zs = [torch.tensor(saved['per_class'][c][method]['Z'], dtype=DT) for c in range(g.k)]
        result['approx'][method] = approximation_diagnostic(g, zs, y)
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument('saved_run', type=Path)
    p.add_argument('--out', type=Path)
    a=p.parse_args()
    out=evaluate(a.saved_run)
    target=a.out or a.saved_run.with_name('approx_' + a.saved_run.name)
    target.write_text(json.dumps(out, indent=2, ensure_ascii=False) + '\n')
    print('Saved',target)
    for method, cls in out['approx'].items():
        print(method,'tail bound h=3',cls['0']['3']['tail_mass_TV_bound'],
              'realized mean TV h=3',round(np.mean([c['3']['empirical_mean_TV'] for c in cls.values()]),8))

if __name__=='__main__':
    main()
