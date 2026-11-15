"""Auditable three-checkpoint metrics from saved per-sample walk representations.

Run from repository root:
    python code/experiments/center_matching_comparison/detailed_metrics.py

The metrics.py functions remain the source of truth for KL, JSD, sqrt-JSD,
JSD-similarity, weighted NLL, center criteria and J-margin.  This script only
adds experiment-specific class-hypothesis scoring and likelihood diagnostics.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / 'code' / 'policy_evaluator'))
from metrics import (class_centers, center_metrics, entropy, jsd, sqrt_jsd,
                     js_similarity, kl_divergence, sample_weights_vector,
                     normalized_nll, soft_margin_loss, report, margin_criterion)

CASES = {
    'uniform': HERE / 'results' / 'representations_uniform.json',
    'guided_round0': HERE / 'results' / 'representations_guided_round0.json',
    'guided_round4': HERE / 'results' / 'representations_guided_round4.json',
}


def distribution_geometry(v):
    """All center-only criteria, including three equivalent JS representations."""
    v = np.asarray(v, float)
    geom = center_metrics(v)
    k = len(v)
    pair_j = jsd(v[:, None, :], v[None, :, :])
    pair_d = sqrt_jsd(v[:, None, :], v[None, :, :])
    pair_s = js_similarity(v[:, None, :], v[None, :, :])
    off = np.triu_indices(k, 1)
    return {
        'T': geom['T'], 'BT': geom['BT'],
        'TNL': geom['TNL'], 'BTNL': geom['BTNL'], 'I': geom['I'],
        'center_entropy_mean': float(np.mean(entropy(v))),
        'min_center_sqrt_JSD': float(np.min(pair_d[off])),
        'mean_center_sqrt_JSD': float(np.mean(pair_d[off])),
        'mean_center_JSD': float(np.mean(pair_j[off])),
        'mean_center_similarity_JS': float(np.mean(pair_s[off])),
        'max_center_similarity_JS': float(np.max(pair_s[off])),
        'pairwise_JSD': pair_j.tolist(),
        'pairwise_sqrt_JSD': pair_d.tolist(),
        'pairwise_similarity_JS': pair_s.tolist(),
    }


def per_sample_center_metrics(p, y, v):
    """Chapter-3 class-balanced within metrics; likelihood and R are distinct.

    This measures the true-center candidate on validation, which is known only
    for evaluating the metrics, NEVER to construct validation trajectories.
    """
    p = np.asarray(p, float)
    y = np.asarray(y, int)
    v = np.asarray(v, float)
    k = len(v)
    if p.shape != (len(y), k):
        raise ValueError('P must be N by K')
    w = sample_weights_vector(y, k)
    q = v[y]
    kl = kl_divergence(p, q)
    j = jsd(p, q)
    d = sqrt_jsd(p, q)
    s = js_similarity(p, q)
    lp = np.sum(np.where(p > 0, p * np.log(q), 0.), axis=1)
    out = {
        'R_KL': float(w @ kl),
        'JSD_within': float(w @ j),
        'sqrt_JSD_within': float(w @ d),
        'similarity_JS_within': float(w @ s),
        'mean_center_log_likelihood': float(w @ lp),
        'mean_center_cross_entropy': float(-w @ lp),
        'mean_sample_entropy': float(w @ entropy(p)),
        'per_class': [],
    }
    for c in range(k):
        local = p[y == c]
        vc = v[c]
        if len(local) == 0:
            raise ValueError('each class must have training or evaluation samples')
        ll = np.sum(np.where(local > 0, local * np.log(vc), 0.), axis=1)
        row = {
            'class': c,
            'H_center': float(entropy(vc)),
            'mean_sample_entropy': float(np.mean(entropy(local))),
            'mean_log_likelihood': float(np.mean(ll)),
            'R_KL': float(np.mean(kl[y == c])),
            'JSD_within': float(np.mean(j[y == c])),
            'sqrt_JSD_within': float(np.mean(d[y == c])),
            'similarity_JS_within': float(np.mean(s[y == c])),
        }
        out['per_class'].append(row)
    return out


def candidate_classification(p, y, v):
    """Class-hypothesis scoring P[i,c,:] vs V[c,:], without label leakage."""
    p = np.asarray(p, float)
    v = np.asarray(v, float)
    y = np.asarray(y, int)
    n, k, k2 = p.shape
    if k != k2 or v.shape != (k, k) or y.shape != (n,):
        raise ValueError('expected P=(N,K,K), V=(K,K) and y=(N,)')
    jj = np.column_stack([jsd(p[:, c, :], v[c]) for c in range(k)])
    kk = np.column_stack([kl_divergence(p[:, c, :], v[c]) for c in range(k)])
    sim = 1. - jj / np.log(2.)
    pred = np.argmin(jj, axis=1)
    w = sample_weights_vector(y, k)
    rows = np.arange(n)
    true = jj[rows, y]
    rival = np.where(np.eye(k, dtype=bool)[y], np.inf, jj).min(axis=1)
    with np.errstate(over='ignore', invalid='ignore'):
        nll = np.logaddexp.reduce(-kk, axis=1) + kk[rows, y]
    losses = {
        'NLL_KL': float(w @ nll),
        'NLL_normalized_KL': normalized_nll(kk, y, w),
        'NLL_similarity_JS': float(w @ (np.logaddexp.reduce(sim, axis=1) - sim[rows, y])),
        'softplus_similarity_tau_003': soft_margin_loss(sim, y, .03, w),
        'softplus_similarity_tau_008': soft_margin_loss(sim, y, .08, w),
        'softplus_similarity_tau_015': soft_margin_loss(sim, y, .15, w),
    }
    return {
        'accuracy': float(np.mean(pred == y)),
        'balanced_accuracy': float(w @ (pred == y)),
        'balanced_error': float(w @ (pred != y)),
        'balanced_true_margin_JSD': float(w @ (rival - true)),
        'balanced_nonpositive_margin_fraction': float(w @ (rival <= true)),
        'mean_winning_gap_JSD': float(np.mean(np.sort(jj, axis=1)[:, 1] - np.sort(jj, axis=1)[:, 0])),
        'per_class_recall': [float(np.mean(pred[y == c] == c)) for c in range(k)],
        'confusion': np.histogram2d(y, pred, bins=[np.arange(k + 1) - .5, np.arange(k + 1) - .5])[0].astype(int).tolist(),
        **losses,
    }


def analyze_case(case_id, filename, history):
    data = json.loads(Path(filename).read_text(encoding='utf-8'))
    train, validation = data['training'], data['validation']
    ptrain = np.asarray(train['P'], float)
    ytrain = np.asarray(train['y'], int)
    centers = np.asarray(data['centers'], float)
    np.testing.assert_allclose(class_centers(ptrain, ytrain), centers, atol=1e-12)
    expected = history['baseline']['centers'] if case_id == 'uniform' else history['history'][int(case_id[-1])]['realized_centers']
    np.testing.assert_allclose(centers, expected, atol=1e-12)
    geom = distribution_geometry(centers)
    train_comp = per_sample_center_metrics(ptrain, ytrain, centers)
    for row in train_comp['per_class']:
        # The two likelihood identities are valid for fitted TRAINING centers.
        np.testing.assert_allclose(row['R_KL'], row['H_center'] - row['mean_sample_entropy'], atol=1e-12)
        np.testing.assert_allclose(row['mean_log_likelihood'], -row['H_center'], atol=1e-12)
    train_pred, _ = report(ptrain, centers, ytrain, metric='js_similarity')
    train_loss = {name: train_pred[name] for name in ('R', 'NLL', 'NLL_norm', 'J_margin',
                  'margin_softplus_0_03', 'margin_softplus_0_08', 'margin_softplus_0_15')}
    vy = np.asarray(validation['y'], int)
    if case_id == 'uniform':
        pv = np.asarray(validation['P'], float)
        vr, _ = report(pv, centers, vy, metric='js_similarity')
        val_pred = {
            'accuracy': vr['accuracy'], 'balanced_accuracy': vr['balanced_accuracy'],
            'NLL_KL': vr['NLL'], 'NLL_normalized_KL': vr['NLL_norm'],
            'NLL_similarity_JS': float('nan'),  # recomputed explicitly below
            'J_margin_KL': vr['J_margin'],
            'softplus_similarity_tau_003': vr['margin_softplus_0_03'],
            'softplus_similarity_tau_008': vr['margin_softplus_0_08'],
            'softplus_similarity_tau_015': vr['margin_softplus_0_15'],
        }
        pp = js_similarity(pv[:, None, :], centers[None, :, :])
        w = sample_weights_vector(vy, len(centers))
        val_pred['NLL_similarity_JS'] = float(w @ (np.logaddexp.reduce(pp, axis=1) - pp[np.arange(len(vy)), vy]))
        val_within = per_sample_center_metrics(pv, vy, centers)
    else:
        pv = np.asarray(validation['candidate_P'], float)
        val_pred = candidate_classification(pv, vy, centers)
        val_within = per_sample_center_metrics(pv[np.arange(len(vy)), vy], vy, centers)
    official_val = (history['baseline']['validation'] if case_id == 'uniform' else
                    history['history'][int(case_id[-1])]['validation_hypothesis'])
    np.testing.assert_allclose(val_pred['balanced_accuracy'], official_val['balanced_accuracy'], atol=1e-12)
    return {
        'case': case_id,
        'inference_protocol': ('class_independent' if case_id == 'uniform' else 'candidate_class_conditional'),
        'centers': centers.tolist(),
        'target_centers': None if case_id == 'uniform' else data['target'],
        'geometry': geom,
        'training_within': train_comp,
        'training_prediction': {'accuracy': train_pred['accuracy'], 'balanced_accuracy': train_pred['balanced_accuracy'], **train_loss},
        'validation_within_true_class_hypothesis': val_within,
        'validation_prediction': val_pred,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=HERE / 'results' / 'detailed_three_cases.json')
    args = parser.parse_args()
    history = json.loads((HERE / 'results' / 'history_00_48.json').read_text(encoding='utf-8'))
    cases = [analyze_case(name, filename, history) for name, filename in CASES.items()]
    result = {
        'methodology': 'Saved exact-seeded replay P; training centers fitted only on train. All class weights uniform.',
        'source_history': 'results/history_00_48.json',
        'note': 'Guided validation uses a different representation for each candidate class. Guided training uses origin class and is oracle-conditioned.',
        'likelihood_note': 'Training E_P log(V) = -H(V) for fitted arithmetic centers; R=H(V)-mean H(P) measures information lost by compression.',
        'sqrt_JSD_note': 'mean sqrt(JSD) is calculated samplewise; it is NOT sqrt(mean JSD).',
        'cases': cases,
    }
    args.out.parent.mkdir(exist_ok=True, parents=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    fields = ['case', 'validation_balanced_accuracy', 'I', 'T', 'TNL', 'BT', 'BTNL',
              'mean_center_JSD', 'min_center_sqrt_JSD', 'mean_center_sqrt_JSD', 'mean_center_similarity_JS',
              'train_R_KL', 'train_log_likelihood', 'train_JSD_within', 'train_sqrt_JSD_within',
              'train_similarity_JS_within', 'train_NLL_KL', 'train_NLL_normalized', 'train_J_margin',
              'validation_R_KL', 'validation_JSD_within', 'validation_sqrt_JSD_within',
              'validation_similarity_JS_within', 'validation_NLL_KL', 'validation_NLL_normalized',
              'validation_NLL_similarity_JS', 'validation_margin']
    with args.out.with_suffix('.csv').open('w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for row in cases:
            g, tw, vp, vw, tp = (row['geometry'], row['training_within'],
                                 row['validation_prediction'], row['validation_within_true_class_hypothesis'],
                                 row['training_prediction'])
            writer.writerow({
                'case': row['case'], 'validation_balanced_accuracy': vp['balanced_accuracy'],
                **{field: g[field] for field in ('I', 'T', 'TNL', 'BT', 'BTNL', 'mean_center_JSD', 'min_center_sqrt_JSD', 'mean_center_sqrt_JSD', 'mean_center_similarity_JS')},
                'train_R_KL': tw['R_KL'], 'train_log_likelihood': tw['mean_center_log_likelihood'],
                'train_JSD_within': tw['JSD_within'], 'train_sqrt_JSD_within': tw['sqrt_JSD_within'],
                'train_similarity_JS_within': tw['similarity_JS_within'],
                'train_NLL_KL': tp['NLL'], 'train_NLL_normalized': tp['NLL_norm'],
                'train_J_margin': tp['J_margin'],
                'validation_R_KL': vw['R_KL'], 'validation_JSD_within': vw['JSD_within'],
                'validation_sqrt_JSD_within': vw['sqrt_JSD_within'], 'validation_similarity_JS_within': vw['similarity_JS_within'],
                'validation_NLL_KL': vp['NLL_KL'], 'validation_NLL_normalized': vp['NLL_normalized_KL'],
                'validation_NLL_similarity_JS': vp['NLL_similarity_JS'],
                'validation_margin': vp.get('balanced_true_margin_JSD'),
            })
    # Center-only diagnostics are available for ALL rounds; per-sample metrics
    # need saved or replayed per-sample representations and are never guessed.
    trajectory = []
    for record in history['history']:
        geo = distribution_geometry(np.asarray(record['realized_centers'], float))
        trajectory.append({
            'round': record['round'], 'geometry': geo,
            'targets': record['targets'], 'realized_centers': record['realized_centers'],
            'mean_target_realized_JSD': record.get('mean_target_realized_jsd'),
            'validation_balanced_accuracy': ((record.get('validation_hypothesis') or {}).get('balanced_accuracy')),
        })
    (args.out.parent / 'center_geometry_49.json').write_text(
        json.dumps(trajectory, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    with (args.out.parent / 'center_geometry_49.csv').open('w', newline='', encoding='utf-8') as f:
        geo_keys = ('I', 'T', 'TNL', 'BT', 'BTNL', 'min_center_sqrt_JSD',
                    'mean_center_sqrt_JSD', 'mean_center_JSD',
                    'mean_center_similarity_JS', 'max_center_similarity_JS',
                    'center_entropy_mean')
        writer = csv.DictWriter(f, fieldnames=['round', *geo_keys,
            'mean_target_realized_JSD', 'validation_balanced_accuracy'])
        writer.writeheader()
        for rec in trajectory:
            writer.writerow({'round': rec['round'], **{key: rec['geometry'][key] for key in geo_keys},
                'mean_target_realized_JSD': rec['mean_target_realized_JSD'],
                'validation_balanced_accuracy': rec['validation_balanced_accuracy']})
    print('Three cases checked against saved centers and validation accuracies.')
    for c in cases:
        print(c['case'], 'BA', round(c['validation_prediction']['balanced_accuracy'], 6),
              'I', round(c['geometry']['I'], 6), 'R', round(c['training_within']['R_KL'], 6),
              'E sqrt-JSD', round(c['training_within']['sqrt_JSD_within'], 6))

if __name__ == '__main__':
    main()
