"""Cheap E09 controls of real capped-horizon coefficients and stored evidence."""
from __future__ import annotations
import json
import unittest
import numpy as np

from analyze import (
    BETA, LAM, HORIZONS, POLICIES, HERE, OUT,
    coefficients, independent_small_h, by_hypothesis, collapse,
    source_inventory, class_centers, max_abs
)


class TrueHorizonCoefficientTests(unittest.TestCase):
    def test_h1_first_step(self):
        p,alpha=coefficients(1)
        np.testing.assert_array_equal(p,[1.])
        np.testing.assert_array_equal(alpha,[1.])

    def test_h2_closed_form(self):
        p,a=coefficients(2)
        np.testing.assert_allclose(p,[.4,.6],rtol=0,atol=1e-15)
        np.testing.assert_allclose(a,[11/15,4/15],rtol=0,atol=1e-15)
        np.testing.assert_allclose(a,independent_small_h(2),rtol=0,atol=1e-15)

    def test_h3_independent_length_sum(self):
        p,a=coefficients(3)
        np.testing.assert_allclose(p,[.4,.24,.36],rtol=0,atol=1e-15)
        np.testing.assert_allclose(a,independent_small_h(3),rtol=0,atol=1e-15)

    def test_mass_and_nonnegative_for_all_h(self):
        for h in HORIZONS:
            p,a=coefficients(h)
            self.assertEqual(len(a),h)
            self.assertTrue(np.all(a>=0))
            self.assertAlmostEqual(sum(a),1,places=12)
            self.assertAlmostEqual(sum(p),1,places=12)
            expected_length=(1-(1-BETA)**h)/BETA
            self.assertAlmostEqual(float(p@np.arange(1,h+1)),expected_length,places=12)

    def test_true_cap_not_fixed_h12_truncation(self):
        a12=coefficients(12)[1]
        for h in (2,3,4,8):
            true=coefficients(h)[1]
            truncated=a12[:h]/a12[:h].sum()
            self.assertGreater(np.abs(true-truncated).sum(),1e-9)


class PersistedE09EvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not (OUT/"metrics_full.json").exists():
            raise unittest.SkipTest("No E09 results yet; run --execute after source preflight")
        cls.results=json.loads((OUT/"metrics_full.json").read_text(encoding="utf8"))
        cls.comparisons=json.loads((OUT/"comparisons.json").read_text(encoding="utf8"))

    def test_12_cases_24_splits_and_classes(self):
        data=self.results
        self.assertEqual(len(data["cases"]),12)
        self.assertEqual({(c["policy"],c["H"]) for c in data["cases"]},
                         {(p,h) for p in POLICIES for h in HORIZONS})
        for c in data["cases"]:
            for split,n,counts in (("train",300,[72,147,81]),
                                    ("validation",90,[13,49,28])):
                x=c[split]
                self.assertEqual(x["rows"],n)
                self.assertEqual(x["counts"],counts)
                self.assertEqual(len(x["per_sample"]["predicted"]),n)
                confusion=np.asarray(x["confusion_true_rows_pred_cols"])
                np.testing.assert_array_equal(confusion.sum(axis=1),counts)
                self.assertAlmostEqual(float(np.mean(np.diag(confusion)/counts)),
                                       x["balanced_accuracy"],places=12)
                self.assertIsNotNone(x["R_JS"])
                self.assertIsNotNone(x["R_sqrt_JS"])
                self.assertIn("center_geometry",x)
                self.assertIn("NLL_KL_unavailable_reason",x)
                self.assertIsNone(c["walk_engine_execution_seconds"])

    def test_E04_and_E08_checks_present(self):
        a=self.comparisons
        self.assertEqual(set(a["E04_uniform_true_h_max_abs_diff"]),
                         {"1","2","3","4","12"})
        for v in a["E04_uniform_true_h_max_abs_diff"].values():
            self.assertLessEqual(max(v.values()),2e-10)
        self.assertEqual(set(a["E08_H12_center_max_abs_diff"]),set(POLICIES))
        self.assertTrue(all(v<2e-10 for v in a["E08_H12_center_max_abs_diff"].values()))

    def test_h1_predicted_from_first_stored_step_without_new_walk(self):
        inv=source_inventory()
        for p in POLICIES:
            step=inv["steps"][f"{p}_train_steps"][0]
            tr=collapse(inv["steps"][f"{p}_train_steps"],coefficients(1)[1])
            np.testing.assert_allclose(step,tr,rtol=0,atol=1e-14)

    def test_no_test_or_training_marker(self):
        c=json.loads((OUT/"coverage.json").read_text(encoding="utf8"))
        for k in ("historical_test_accessed","data_generated","trained_policy_updated",
                  "monte_carlo_run","original_E08_engine_rerun"):
            self.assertFalse(c[k])
        self.assertEqual(c["common_metric_blocks"],24)


if __name__=="__main__":
    unittest.main()
