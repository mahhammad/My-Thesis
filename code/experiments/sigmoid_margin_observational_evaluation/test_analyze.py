"""Offline, lightweight tests for historical E-CH4-07 artifacts; no walks or training."""
import csv
import json
import math
import unittest
from pathlib import Path
from analyze import rows, sigmoid_loss, normalized_sensitivity, compare_rounds

HERE = Path(__file__).resolve().parent

class SigmoidObservationalTests(unittest.TestCase):
    def test_sigmoid_properties(self):
        for tau in (0.002, 0.005, 0.02):
            self.assertAlmostEqual(sigmoid_loss(0,tau),.5)
            self.assertAlmostEqual(normalized_sensitivity(0,tau),1)
            self.assertAlmostEqual(sigmoid_loss(-.03,tau)+sigmoid_loss(.03,tau),1)
            self.assertLess(normalized_sensitivity(.03,tau),1)
        with self.assertRaises(ValueError):
            sigmoid_loss(0,0)

    def test_archived_rows_and_direction_reversal(self):
        d=rows()
        self.assertEqual(len(d),18)
        self.assertEqual(len({(x["case"],x["tau"]) for x in d}),18)
        by=compare_rounds(d)
        self.assertTrue(by[.005]["guided_round4_lower_loss"])
        self.assertFalse(by[.01]["guided_round4_lower_loss"])

    def test_common_report_and_coverage(self):
        j=json.loads((HERE/"results/metrics_full.json").read_text(encoding="utf-8"))
        coverage=json.loads((HERE/"results/coverage.json").read_text(encoding="utf-8"))
        self.assertFalse(j["historical_test_accessed"])
        self.assertEqual(len(j["cases"]),3)
        self.assertEqual(len(j["training_runs"]),12)
        self.assertEqual(coverage["measurement_counts"]["benchmark_full_v2_verified_runs"],36)
        self.assertTrue(coverage["archive_only_or_resolved_coverage"])
        for c in j["cases"]:
            for split in ("train","validation"):
                x=c[split]
                for key in ("R_KL","R_JS","R_sqrt_JS","balanced_accuracy","ordinary_accuracy","margin_quantiles"):
                    self.assertIn(key,x)
                self.assertEqual(len(x["counts"]),3)
            for key in ("T","BT","TNL","BTNL","I","pairwise_JS","pairwise_sqrt_JS"):
                self.assertIn(key,c["center_geometry"])

    def test_common_and_diagnostic_tau005_agree(self):
        j=json.loads((HERE/"results/metrics_full.json").read_text(encoding="utf-8"))
        d=rows()
        for c in j["cases"]:
            diag=next(x for x in d if x["case"]==c["case_id"] and float(x["tau"])==.005)
            self.assertAlmostEqual(float(diag["balanced_accuracy"]),c["validation"]["balanced_accuracy"],places=9)
            self.assertAlmostEqual(float(diag["balanced_sigmoid_loss"]),c["validation"]["balanced_sigmoid_margin_loss"],places=9)

if __name__=="__main__":
    unittest.main()
