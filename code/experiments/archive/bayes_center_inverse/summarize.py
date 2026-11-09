#!/usr/bin/env python3
"""Aggregate only comparable seed/split regimes; label mixed GN budgets."""
import json
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
FILES=['bayes_seed_20261107_gn10.json',
       'bayes_seed_20261108.json',   # GN30, not pooled for timing as equal budget
       'bayes_seed_20261109_gn10.json']

def aggregate(files=FILES):
    runs=[json.loads((HERE/'results'/f).read_text()) for f in files]
    names=list(runs[0]['methods'])
    metrics=['mean_target_JSD','train_within_KL_R','train_within_JSD',
             'train_JSD_to_individual_Bayes_posteriors']
    result={'source_files':files,
            'caution':'The 20261108 run uses GN budget 30, other two use 10. All share Q45, Exact90, same seed-dependent subset size 72 and identical all-300 Bayes targets. GN timing/precision are not a matched-budget comparison.',
            'target_centers':runs[0]['Bayes']['centers'],
            'bayes_target_geometry':runs[0]['Bayes']['center_geometry'],
            'by_seed':[], 'means':{}}
    for run in runs:
        x={'seed':run['meta']['seed'],'nref':run['meta']['nref'],
           'GN_budget':run['meta']['bootstrap_steps'],
           'proved_infeasible_rows':run['relaxed_feasibility_bounds']['proved_infeasible_center_rows'],
           'method_data':{}}
        for m in names:
            r=run['methods'][m]
            x['method_data'][m]={**{t:r[t] for t in metrics},
                                 'I':r['center_geometry']['I'],
                                 'mean_pairwise_JSD':r['center_geometry']['mean_pairwise_JSD'],
                                 'min_pairwise_sqrt_JSD':r['center_geometry']['min_pairwise_sqrt_JSD'],
                                 'training_elapsed_s':r['timing']['elapsed_s_total'] if r['timing'] else None}
        result['by_seed'].append(x)
    for m in names:
        result['means'][m]={t:float(np.mean([q['method_data'][m][t] for q in result['by_seed']]))
                             for t in metrics+['I','mean_pairwise_JSD','min_pairwise_sqrt_JSD']}
        if m!='uniform':
            result['means'][m]['training_elapsed_s_mixed_gn_budget']=float(np.mean(
                [q['method_data'][m]['training_elapsed_s'] for q in result['by_seed']]))
    return result

if __name__=='__main__':
    x=aggregate()
    p=HERE/'results'/'summary.json'
    p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
    print('Mean metrics across 3 seed-dependent subsets:')
    for m, r in x['means'].items():
        print(m,'error',round(r['mean_target_JSD'],6), 'I',round(r['I'],5),
              'R_JS',round(r['train_within_JSD'],5))
    print('saved',p)
