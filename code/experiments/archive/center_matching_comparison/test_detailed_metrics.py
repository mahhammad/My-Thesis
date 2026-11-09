"""Regression tests for class-conditional diagnostics and JS transforms."""
from __future__ import annotations
import json
import unittest
import numpy as np
from pathlib import Path
from detailed_metrics import (CASES, HERE, analyze_case, distribution_geometry,
                              per_sample_center_metrics, candidate_classification)
from metrics import jsd, sqrt_jsd, js_similarity, class_centers


class DetailedMetricsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.history = json.loads((HERE / 'results' / 'history_00_48.json').read_text())
        cls.cases = [analyze_case(name, path, cls.history) for name, path in CASES.items()]

    def test_training_centers_and_js_identity(self):
        for key, path in CASES.items():
            data = json.loads(Path(path).read_text())
            p = np.asarray(data['training']['P'])
            y = np.asarray(data['training']['y'])
            v = np.asarray(data['centers'])
            np.testing.assert_allclose(class_centers(p, y), v, atol=1e-12)
            d = per_sample_center_metrics(p, y, v)
            self.assertAlmostEqual(d['similarity_JS_within'],
                                   1-d['JSD_within']/np.log(2), places=12)
            self.assertGreaterEqual(np.sqrt(d['JSD_within']) + 1e-14,
                                    d['sqrt_JSD_within'])
            for row in d['per_class']:
                self.assertAlmostEqual(row['R_KL'],
                                       row['H_center']-row['mean_sample_entropy'], places=12)
                self.assertAlmostEqual(row['mean_log_likelihood'],
                                       -row['H_center'], places=12)

    def test_center_pairwise(self):
        for case in self.cases:
            v = np.asarray(case['centers'])
            m = distribution_geometry(v)
            js = jsd(v[:, None, :], v[None, :, :])
            np.testing.assert_allclose(m['pairwise_JSD'], js)
            np.testing.assert_allclose(m['pairwise_sqrt_JSD'], np.sqrt(js))
            np.testing.assert_allclose(m['pairwise_similarity_JS'], 1-js/np.log(2))
            self.assertAlmostEqual(m['mean_center_similarity_JS'],
                                   1-m['mean_center_JSD']/np.log(2), places=12)
            self.assertLessEqual(m['min_center_sqrt_JSD'], m['mean_center_sqrt_JSD'])

    def test_grounded_accuracy(self):
        expected = [0.848639455782313, 0.7260596546310832, 0.8383045525902669]
        for case, target in zip(self.cases, expected):
            self.assertAlmostEqual(case['validation_prediction']['balanced_accuracy'],target,places=12)

    def test_nll_accuracy_misalignment(self):
        first, best = self.cases[1:]
        self.assertLess(first['validation_prediction']['balanced_accuracy'],
                        best['validation_prediction']['balanced_accuracy'])
        self.assertLess(first['validation_prediction']['NLL_KL'],
                        best['validation_prediction']['NLL_KL'])

    def test_reference_49_rounds_no_interpolation(self):
        traj=self.history['history']
        self.assertEqual(len(traj),49)
        measured=[r for r in traj if r.get('validation_hypothesis') is not None]
        self.assertEqual(len(measured),25)
        self.assertEqual(max(measured,key=lambda r:r['validation_hypothesis']['balanced_accuracy'])['round'],4)


if __name__=='__main__': unittest.main()
