"""Regression tests for archived sigmoid-margin diagnostics (no random walks)."""
import json
import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze


class TestSigmoidDiagnostics(unittest.TestCase):
    def test_loss_bounded_and_symmetric(self):
        for tau in analyze.TAUS:
            self.assertEqual(analyze.sigmoid_loss(0, tau), 0.5)
            for m in (0.002, 0.02, 0.1):
                a, b = analyze.sigmoid_loss(m, tau), analyze.sigmoid_loss(-m, tau)
                self.assertGreaterEqual(a, 0)
                self.assertLessEqual(b, 1)
                self.assertAlmostEqual(a + b, 1, places=12)

    def test_jsd_known_cases(self):
        self.assertAlmostEqual(analyze.jsd([1, 0], [1, 0]), 0, places=12)
        self.assertAlmostEqual(analyze.jsd([1, 0], [0, 1]), math.log(2), places=12)

    def test_reproduce_committed_summary(self):
        fresh = analyze.analyze()
        saved = json.loads((analyze.OUTPUT / "sigmoid_validation_diagnostics.json")
                           .read_text(encoding="utf-8"))
        self.assertEqual([x["case"] for x in fresh["cases"]],
                         [x["case"] for x in saved["cases"]])
        self.assertEqual(len(fresh["cases"]), 3)
        for current, archived in zip(fresh["cases"], saved["cases"]):
            self.assertEqual(len(current["per_sample"]), 90)
            self.assertAlmostEqual(current["balanced_accuracy"],
                                   archived["balanced_accuracy"], places=10)
            self.assertEqual(len(current["metrics_by_tau"]), 6)
            for x, y in zip(current["metrics_by_tau"], archived["metrics_by_tau"]):
                for metric in ("balanced_sigmoid_loss", "mean_relative_sensitivity",
                               "wrong_sensitivity_share", "mass_abs_margin_le_tau"):
                    self.assertAlmostEqual(x[metric], y[metric], places=10)


if __name__ == "__main__":
    unittest.main()
