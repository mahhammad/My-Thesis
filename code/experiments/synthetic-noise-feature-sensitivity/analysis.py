"""Pure post-fit E-CH4-14 summaries from 36 immutable canonical reports.

No graph, RNG, optimizer, file I/O or inferential confidence intervals are used.
Subset spread describes selected (dependent) feature combinations, not seeds.
"""
from __future__ import annotations

from math import isfinite, sqrt

from data import SUBSETS

FAMILIES = ('shared', 'class')
METRICS = ('train_BA', 'validation_BA', 'train_Accuracy', 'validation_Accuracy',
           'train_R_JS', 'validation_R_JS', 'train_I')


def _values(report):
    tr, val = report['train'], report['validation']
    return dict(train_BA=tr['balanced_accuracy'],
                validation_BA=val['balanced_accuracy'],
                train_Accuracy=tr['ordinary_accuracy'],
                validation_Accuracy=val['ordinary_accuracy'],
                train_R_JS=tr['R_JS'], validation_R_JS=val['R_JS'],
                train_I=tr['I'])


def _predictions(report, split):
    section = report[split]
    per = section['per_sample']
    preds, labels = per['predicted'], per['labels']
    if (not isinstance(preds, list) or not isinstance(labels, list)
            or not preds or len(preds) != len(labels)
            or any(type(x) is not int or x not in (0, 1, 2) for x in preds + labels)):
        raise ValueError('Prediction/label record missing or malformed')
    if len(preds) != section['rows']:
        raise ValueError('Prediction row count drift')
    return preds, labels


def _stats(rows, field):
    values = [r[field] for r in rows]
    if not values or any(type(v) not in (int, float) or not isfinite(v) for v in values):
        raise ValueError(f'Metric {field} missing/nonfinite')
    n = len(values)
    avg = sum(values) / n
    # Descriptive sample SD across the selected dependent subsets, not seeds.
    sd = sqrt(sum((x - avg)**2 for x in values) / (n-1)) if n > 1 else None
    return {f'{field}_mean': avg, f'{field}_subset_sd': sd}


def summarize_reports(reports):
    """Validate exact paired grid and return individual, grouped and paired rows.

    Requires the saved canonical *per_sample* predictions and row-aligned labels;
    flips are prediction changes vs N0 within the same family and split, not
    accuracy changes. Validation labels never enter model training here.
    """
    expected = {f'{sid}__{family}' for sid, _ in SUBSETS for family in FAMILIES}
    if not isinstance(reports, dict) or set(reports) != expected:
        raise ValueError('Exactly 36 immutable subset/family reports are required')
    baseline = {family: reports[f'N0-01__{family}'] for family in FAMILIES}
    rows = []
    by_key = {}
    for subset_id, noise in SUBSETS:
        for family in FAMILIES:
            run_id = f'{subset_id}__{family}'
            report = reports[run_id]
            if (report.get('subset_id') != subset_id or report.get('family') != family
                    or report.get('test_accessed') is not False):
                raise ValueError('Report subset/family/Test provenance mismatch')
            metrics = _values(report)
            if any(type(v) not in (int, float) or not isfinite(v) for v in metrics.values()):
                raise ValueError('Nonfinite or missing metric')
            row = {'run_id': run_id, 'subset_id': subset_id, 'family': family,
                   'k': len(noise), **metrics}
            for split in ('train', 'validation'):
                pred, labels = _predictions(report, split)
                orig_pred, orig_labels = _predictions(baseline[family], split)
                if len(pred) != len(orig_pred) or labels != orig_labels:
                    raise ValueError('Rows/labels must match N0 in the same family')
                row[f'{split}_prediction_flips_vs_N0'] = sum(
                    p != q for p, q in zip(pred, orig_pred))
            rows.append(row)
            by_key[(subset_id, family)] = row

    paired, groups, paired_groups = [], [], []
    for subset_id, noise in SUBSETS:
        shared, class_ = by_key[(subset_id, 'shared')], by_key[(subset_id, 'class')]
        row = {'subset_id': subset_id, 'k': len(noise)}
        for metric in METRICS:
            row[f'{metric}_class_minus_shared'] = class_[metric] - shared[metric]
        paired.append(row)

    for k in range(6):
        subset_ids = [sid for sid, noise in SUBSETS if len(noise) == k]
        for family in FAMILIES:
            members = [by_key[(sid, family)] for sid in subset_ids]
            row = {'k': k, 'family': family, 'n_subsets': len(members),
                   'subset_ids': ','.join(subset_ids),
                   'subset_spread_status': 'descriptive' if len(members) > 1 else 'not_applicable'}
            for metric in METRICS:
                row.update(_stats(members, metric))
            groups.append(row)
        paired_rows = [p for p in paired if p['k'] == k]
        row = {'k': k, 'n_pairs': len(paired_rows), 'subset_ids': ','.join(subset_ids),
               'subset_spread_status': 'descriptive' if len(paired_rows) > 1 else 'not_applicable'}
        for metric in METRICS:
            key = f'{metric}_class_minus_shared'
            row.update(_stats(paired_rows, key))
        paired_groups.append(row)
    return {'schema': 'e-ch4-14-paired-descriptive-v1',
            'design': 'one_mother_seed__dependent_subsets__no_inferential_CI',
            'test_accessed': False, 'individual': rows, 'grouped': groups,
            'paired': paired, 'paired_by_k': paired_groups}
