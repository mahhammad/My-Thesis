"""Lightweight tests for policy -> raw snapshot -> independent metric reporter.

No optimization, no historical Test split, no 76-run replay.
Run from repo root:
  python -m unittest -v code/experiments/benchmark_v1/test_snapshot_pipeline.py
"""
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "code/data_generator"))

from generate import load_config
from backfill_saved_scores import dataset_train_val
from engine import DT, ExactMaskedWalk
from policy_interface import create_policy, evaluate_policy
from snapshot_pipeline import load_snapshot, save_snapshot
from report_snapshot import report_snapshot, save_report, standard_metrics
from experiment_center_target import save_center_target_report


class SnapshotChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.config = ROOT / "code/data_generator/config.json"
        spec = load_config(cls.config)
        splits = dataset_train_val(spec)  # Explicitly Train and Validation ONLY.
        tx, ty = splits["train"]
        vx, vy = splits["validation"]
        cls.ids = np.concatenate([np.flatnonzero(ty == c)[:8]
                                  for c in range(len(spec["pi"]))])
        cls.y, cls.vy = ty[cls.ids], vy
        cls.x, cls.vx = tx[cls.ids], vx
        cls.spec = spec
        cls.g = ExactMaskedWalk(cls.x, cls.y, spec["gamma"], horizon=4,
                                query_x=cls.vx)
        cls.z = torch.zeros((cls.g.d + 1, cls.g.d), dtype=DT)

    def snapshot(self, root, *, family="shared", logits=None, target=False):
        if logits is None:
            logits = self.z if family == "shared" else self.z.repeat(self.g.n, 1, 1)
        centers = self.g.train_outputs(logits, family).centers.detach().numpy()
        return save_snapshot(
            root, self.g, logits, family, reference_ids=self.ids,
            config_path=self.config,
            config_reference="code/data_generator/config.json",
            data_seed=self.spec["seed"], reference_seed=20261107,
            objective="uniform", tau=.005, beta=.4, lam=.8, horizon=4,
            validation_x=self.vx,
            target_centers=centers if target else None,
            target_source="generated_by_known_policy" if target else None
        )

    def test_common_report_matches_existing_metrics_without_training(self):
        with tempfile.TemporaryDirectory() as td:
            snapshot = self.snapshot(td)
            self.assertEqual(snapshot["prediction"]["validation_status"], "available")
            obj, arrays = load_snapshot(td)
            self.assertEqual(arrays["train_by_hypothesis"].shape,
                             (self.g.n, self.g.k, self.g.k))
            self.assertEqual(arrays["validation_by_hypothesis"].shape,
                             (len(self.vx), self.g.k, self.g.k))
            report = report_snapshot(td)
            adjusted = report_snapshot(td, tau_override=.020, per_sample=False)
            self.assertEqual(adjusted["source"]["metric_tau_source"], "report_override")
            self.assertAlmostEqual(adjusted["source"]["metric_tau"], .020)
            old = self.g.metrics(self.g.train_outputs(self.z, "shared"))
            for key in ("balanced_accuracy", "ordinary_accuracy", "I",
                        "R_JS", "R_sqrt_JS", "R_KL",
                        "balanced_sigmoid_margin_loss",
                        "balanced_JS_similarity_NLL"):
                self.assertAlmostEqual(report["train"][key], old[key], places=9,
                                       msg=key)
            self.assertIsNone(report["validation"]["R_KL_entropy_identity"])
            save_report(td, report)
            self.assertTrue((Path(td)/"report.md").exists())
            self.assertTrue((Path(td)/"report.json").exists())
            self.assertTrue((Path(td)/"model.json").exists())
            self.assertTrue((Path(td)/"distributions.npz").exists())

    def test_node_has_no_out_of_graph_validation(self):
        z = torch.zeros((self.g.n, self.g.d + 1, self.g.d), dtype=DT)
        with tempfile.TemporaryDirectory() as td:
            self.snapshot(td, family="node", logits=z)
            report = report_snapshot(td)
            self.assertEqual(report["validation"]["status"], "not_applicable")
            self.assertIsNone(report["validation"]["balanced_accuracy"])

    def test_generic_policy_supports_four_families_and_class_tied(self):
        g = self.g
        data = {"ref_x": self.x, "ref_y": self.y, "query_x": self.vx}
        shapes = {
            "shared": (g.d + 1, g.d),
            "class": (g.k, g.d + 1, g.d),
            "node": (g.n, g.d + 1, g.d),
            "node_class": (g.k, g.n, g.d + 1, g.d)
        }
        for family, shape in shapes.items():
            with self.subTest(family=family):
                obj = evaluate_policy(data, family, torch.zeros(shape, dtype=DT),
                                      gamma=self.spec["gamma"], horizon=4)
                self.assertEqual(obj["train_by_hypothesis"].shape,
                                 (g.n, g.k, g.k))
                self.assertEqual(obj["validation_status"] == "available",
                                 family in ("shared", "class"))
        hybrid = {
            "start_by_class": np.zeros((g.k, g.d)),
            "intermediate_by_node_class": np.zeros((g.k, g.n, g.d, g.d))
        }
        policy = create_policy("class_tied_node", hybrid, n=g.n, k=g.k,
                               d=g.d, n_query=len(self.vx))
        self.assertEqual(tuple(policy.logits.shape),
                         (g.k, g.n, g.d + 1, g.d))
        out = evaluate_policy(data, "class_tied_node", hybrid,
                              gamma=self.spec["gamma"], horizon=4)
        ref = evaluate_policy(data, "shared", self.z, gamma=self.spec["gamma"],
                              horizon=4)
        self.assertEqual(out["validation_status"], "available")
        np.testing.assert_allclose(out["train_by_hypothesis"],
                                   ref["train_by_hypothesis"], atol=1e-10)
        np.testing.assert_allclose(out["validation_by_hypothesis"],
                                   ref["validation_by_hypothesis"], atol=1e-10)

    def test_special_center_report_is_not_part_of_common_report(self):
        with tempfile.TemporaryDirectory() as td:
            self.snapshot(td, target=True)
            generic = report_snapshot(td, per_sample=False)
            self.assertNotIn("specialized", generic)
            specialized = save_center_target_report(td)
            self.assertAlmostEqual(specialized["result"]["mean_center_JS"],
                                   0., places=12)
            self.assertTrue((Path(td)/"experiment_report.json").is_file())

    def test_tampering_and_immutable_snapshot_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            self.snapshot(td)
            with self.assertRaises(FileExistsError):
                self.snapshot(td)
            path = Path(td)/"distributions.npz"
            path.write_bytes(path.read_bytes()+b"tampered")
            with self.assertRaisesRegex(ValueError, "SHA256"):
                load_snapshot(td)

    def test_tie_is_not_fabricated_as_a_correct_margin(self):
        reps = np.full((2,2,2), .5)
        centers = np.full((2,2), .5)
        result = standard_metrics(reps, centers, [0,1], .005, split="train")
        self.assertEqual(result["tie_count"], 2)
        self.assertAlmostEqual(result["balanced_accuracy"], .5)
        self.assertAlmostEqual(result["hard_margin_error_conservative"], 1.0)


if __name__ == "__main__":
    unittest.main()
