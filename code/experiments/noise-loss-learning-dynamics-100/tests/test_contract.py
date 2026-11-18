"""No-fit CI checks for the new 100-step E-CH4-14 design and trace gate."""
from __future__ import annotations
from collections import Counter
from pathlib import Path
import json
import sys
import unittest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

from protocol import (PRIMARY, NOISE_POOL, OBJECTIVES, FAMILIES,
                      choose_six, make_design, UPDATES)
from report import VALIDATION_KEYS, TRAIN_KEYS, check_record


class ExperimentalDesignTests(unittest.TestCase):
    def test_exact_grid(self):
        d = make_design()
        self.assertEqual(len(d["datasets"]), 38)
        self.assertEqual(len(d["runs"]), 152)
        self.assertEqual(sum(r["comparison"] for r in d["runs"]), 144)
        self.assertEqual(OBJECTIVES, ("sigmoid_JS_margin_tau005", "softplus_JS_tau005"))
        self.assertEqual(FAMILIES, ("shared", "class"))
        self.assertEqual(UPDATES, 100)
        self.assertEqual(d["test_accessed"], False)

    def test_six_distinct_subsets_per_level(self):
        d = make_design()
        self.assertEqual(len(PRIMARY), 3)
        self.assertEqual(len(NOISE_POOL), 7)
        for size in range(8):
            datasets = [x for x in d["datasets"] if x["noise_count"] == size]
            self.assertEqual(len(datasets), 1 if size in (0, 7) else 6)
            self.assertEqual(len({tuple(x["noise_indices"]) for x in datasets}), len(datasets))
            for x in datasets:
                self.assertEqual(len(x["noise_indices"]), size)
                self.assertTrue(set(x["noise_indices"]).issubset(NOISE_POOL))
                self.assertEqual(len(x["feature_indices"]), 3 + size)
                self.assertEqual(x["comparison"], 1 <= size <= 6)

    def test_determinism_and_pairing(self):
        self.assertEqual(make_design(), make_design())
        d = make_design()
        pairs = Counter((r["dataset_id"], r["objective"], r["family"])
                        for r in d["runs"])
        self.assertTrue(all(n == 1 for n in pairs.values()))
        for ds in d["datasets"]:
            self.assertEqual(sum(r["dataset_id"] == ds["dataset_id"] for r in d["runs"]), 4)
        for count in range(1, 7):
            self.assertEqual(choose_six(count), choose_six(count))

    def test_validation_scope(self):
        self.assertEqual(set(VALIDATION_KEYS),
                         {"accuracy", "balanced_accuracy",
                          "precision_macro", "recall_macro"})
        self.assertNotIn("density_r_js", VALIDATION_KEYS)
        self.assertNotIn("separation_i", VALIDATION_KEYS)
        self.assertIn("density_r_js", TRAIN_KEYS)
        self.assertIn("separation_i", TRAIN_KEYS)

    def test_rejects_wrong_trace(self):
        r = make_design()["runs"][4]
        self.assertRaises(Exception, check_record, {"step": 0, "run_id": r["run_id"]}, r)


if __name__ == "__main__":
    unittest.main()
