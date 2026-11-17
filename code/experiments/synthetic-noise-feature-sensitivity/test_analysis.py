"""Fixture-only verification of descriptive aggregation; no optimizer or data writes."""
from math import isclose, sqrt
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analysis import summarize_reports
from data import SUBSETS


def reports_fixture():
    reports = {}
    for subset_id, noise in SUBSETS:
        ordinal = [s for s, n in SUBSETS if len(n) == len(noise)].index(subset_id)
        for family in ('shared', 'class'):
            shift = .02 if family == 'class' else 0.
            val = len(noise) / 10 + ordinal / 100 + shift
            predicted_tr = [0, 1, 2, 0, 1, 2]
            predicted_va = [0, 1, 2]
            if subset_id == 'N1-01' and family == 'shared':
                predicted_tr = [1, 1, 2, 0, 1, 2]
                predicted_va = [1, 1, 2]
            def split(rows, labels, preds):
                return {'rows': rows, 'balanced_accuracy': val,
                        'ordinary_accuracy': val+.01, 'R_JS':val+.03,
                        'per_sample': {'labels':labels, 'predicted':preds}}
            tr = split(6,[0,1,2,0,1,2],predicted_tr)
            tr['I'] = val + .04
            va = split(3,[0,1,2],predicted_va)
            reports[f'{subset_id}__{family}'] = dict(
                subset_id=subset_id, family=family, test_accessed=False,
                train=tr, validation=va)
    return reports


class AnalysisTests(unittest.TestCase):
    def test_exact_groups_and_paired_differences(self):
        result = summarize_reports(reports_fixture())
        self.assertEqual(len(result['individual']),36)
        self.assertEqual(len(result['grouped']),12)
        self.assertEqual(len(result['paired']),18)
        self.assertEqual(len(result['paired_by_k']),6)
        grouped = next(r for r in result['grouped'] if r['k']==1 and r['family']=='shared')
        self.assertEqual(grouped['n_subsets'],4)
        self.assertTrue(isclose(grouped['validation_BA_mean'],.115))
        self.assertTrue(isclose(grouped['validation_BA_subset_sd'],sqrt(.0005/3)))
        self.assertEqual(grouped['subset_spread_status'],'descriptive')
        for k in (0,5):
            for g in (r for r in result['grouped'] if r['k']==k):
                self.assertIsNone(g['train_BA_subset_sd'])
                self.assertEqual(g['subset_spread_status'],'not_applicable')
            paired = next(r for r in result['paired_by_k'] if r['k']==k)
            self.assertIsNone(paired['validation_BA_class_minus_shared_subset_sd'])
        for pair in result['paired']:
            self.assertTrue(isclose(pair['train_BA_class_minus_shared'],.02))
        self.assertNotIn('confidence_interval',str(result).lower())

    def test_exact_prediction_flips_per_split_and_family(self):
        result = summarize_reports(reports_fixture())
        rows = {(r['subset_id'],r['family']):r for r in result['individual']}
        self.assertEqual(rows['N1-01','shared']['train_prediction_flips_vs_N0'],1)
        self.assertEqual(rows['N1-01','shared']['validation_prediction_flips_vs_N0'],1)
        self.assertEqual(rows['N1-01','class']['validation_prediction_flips_vs_N0'],0)
        for family in ('shared','class'):
            self.assertEqual(rows['N0-01',family]['train_prediction_flips_vs_N0'],0)
            self.assertEqual(rows['N0-01',family]['validation_prediction_flips_vs_N0'],0)

    def test_reject_missing_job_or_misaligned_rows(self):
        reports=reports_fixture()
        reports.pop('N3-01__shared')
        with self.assertRaisesRegex(ValueError,'Exactly 36'):
            summarize_reports(reports)
        reports=reports_fixture()
        reports['N2-01__shared']['validation']['per_sample']['labels']=[1,0,2]
        with self.assertRaisesRegex(ValueError,'Rows/labels'):
            summarize_reports(reports)

    def test_reject_modified_provenance_and_nonfinite_metric(self):
        reports=reports_fixture()
        reports['N4-01__class']['test_accessed']=True
        with self.assertRaisesRegex(ValueError,'provenance'):
            summarize_reports(reports)
        reports=reports_fixture()
        reports['N5-01__shared']['train']['I']=float('nan')
        with self.assertRaisesRegex(ValueError,'Nonfinite'):
            summarize_reports(reports)


if __name__=='__main__': unittest.main()
