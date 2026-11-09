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
from dataset_adapter import dataset_train_val
from backfill_saved_scores import dataset_train_val as legacy_dataset_train_val
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

    def test_new_dataset_adapter_matches_historical_generator(self):
        current = dataset_train_val(self.spec)
        legacy = legacy_dataset_train_val(self.spec)
        for split in ("train", "validation"):
            x, y = current[split]
            old_x, old_y = legacy[split]
            np.testing.assert_array_equal(y, old_y)
            np.testing.assert_allclose(x, old_x, equal_nan=True, atol=0, rtol=0)

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

    def test_validation_start_roundtrip_from_saved_dataset_and_policy(self):
        """Re-evaluate heldout queries from saved parameters, not saved predictions."""
        from snapshot_pipeline import array_fingerprint
        k, n, d, m = self.g.k, self.g.n, self.g.d, len(self.vx)
        base = np.arange(d, dtype=np.float64) * .07
        cases = [
            ("shared", np.zeros((d + 1, d)), None, None, None),
            ("class", np.zeros((k, d + 1, d)), None, None, None),
            ("node", np.zeros((n, d + 1, d)),
             np.broadcast_to(base, (m, d)).copy(), "test_external_node", None),
            ("node_class", np.zeros((k, n, d + 1, d)),
             np.broadcast_to(base, (k, m, d)).copy(), "test_external_node_class", None),
        ]
        tied = np.arange(k * d, dtype=np.float64).reshape(k, d) * .03
        tied_mid = np.zeros((k, n, d, d), dtype=np.float64)
        hybrid = create_policy(
            "class_tied_node",
            {"start_by_class": tied, "intermediate_by_node_class": tied_mid},
            n=n, k=k, d=d, n_query=m)
        cases.append(("node_class", hybrid.logits.numpy(),
                      hybrid.query_start_logits.numpy(),
                      hybrid.query_start_source, "class_tied_node"))
        for family, raw, starts, source, variant in cases:
            with self.subTest(family=family, variant=variant):
                with tempfile.TemporaryDirectory() as td:
                    save_snapshot(
                        td, self.g, torch.as_tensor(raw, dtype=DT), family,
                        reference_ids=self.ids, config_path=self.config,
                        config_reference="code/data_generator/config.json",
                        data_seed=self.spec["seed"], reference_seed=20261107,
                        objective="roundtrip", tau=.005, beta=.4, lam=.8,
                        horizon=4, validation_x=self.vx,
                        query_starts=(None if starts is None else
                                      torch.as_tensor(starts, dtype=DT)),
                        query_start_source=source, policy_variant=variant)
                    model, arrays = load_snapshot(td)
                    # Independent dataset reconstruction and original reference order.
                    spec = load_config(self.config)
                    spec["seed"] = model["data"]["data_seed"]
                    splits = dataset_train_val(spec)
                    x_all, y_all = splits["train"]
                    query_x, _ = splits["validation"]
                    ids = np.asarray(model["data"]["reference_ids"], dtype=int)
                    self.assertEqual(array_fingerprint(query_x),
                                     model["data"]["validation_x_sha256"])
                    self.assertEqual(array_fingerprint(x_all[ids]),
                                     model["data"]["train_reference_x_sha256"])
                    saved_start = model["parameters"]["query_start"]
                    reconstruction = {
                        "logits": model["parameters"]["logits"]
                    }
                    if saved_start is not None:
                        reconstruction["query_start_logits"] = saved_start["logits"]
                        reconstruction["query_start_source"] = saved_start["source"]
                    reevaluated = evaluate_policy(
                        {"ref_x": x_all[ids], "ref_y": y_all[ids],
                         "query_x": query_x},
                        family, reconstruction, gamma=model["parameters"]["gamma"],
                        beta=model["parameters"]["beta"],
                        lam=model["parameters"]["lambda"],
                        horizon=model["parameters"]["H"])
                    self.assertEqual(reevaluated["validation_status"], "available")
                    np.testing.assert_allclose(
                        reevaluated["validation_by_hypothesis"],
                        arrays["validation_by_hypothesis"], atol=1e-12, rtol=0)
                    if variant == "class_tied_node":
                        build = model["parameters"]["policy_construction"]
                        np.testing.assert_array_equal(build["start_by_class"], tied)
                        self.assertIn("masked_softmax_over_observed_features",
                                      build["rule"])

    def test_query_start_metadata_tampering_is_rejected(self):
        import json
        n, d, m = self.g.n, self.g.d, len(self.vx)
        z = torch.zeros((n, d + 1, d), dtype=DT)
        starts = torch.arange(d, dtype=DT)[None, :].expand(m, d).clone()
        with tempfile.TemporaryDirectory() as td:
            save_snapshot(
                td, self.g, z, "node", reference_ids=self.ids,
                config_path=self.config,
                data_seed=self.spec["seed"], reference_seed=20261107,
                objective="test", tau=.005, beta=.4, lam=.8, horizon=4,
                validation_x=self.vx, query_starts=starts,
                query_start_source="explicit_test")
            p = Path(td) / "model.json"
            original = json.loads(p.read_text(encoding="utf8"))
            for field, replacement in (
                ("sha256", "0" * 64),
                ("shape", [m + 1, d]),
                ("source", "tampered"),
            ):
                bad = json.loads(json.dumps(original))
                bad["parameters"]["query_start"][field] = replacement
                p.write_text(json.dumps(bad), encoding="utf8")
                with self.assertRaises(ValueError):
                    load_snapshot(td)
            p.write_text(json.dumps(original), encoding="utf8")
            load_snapshot(td)

    def test_class_tied_query_start_mismatch_is_rejected(self):
        import json
        k, n, d, m = self.g.k, self.g.n, self.g.d, len(self.vx)
        a = np.arange(k*d, dtype=np.float64).reshape(k, d) * .1
        policy = create_policy(
            "class_tied_node",
            {"start_by_class": a,
             "intermediate_by_node_class": np.zeros((k, n, d, d))},
            n=n, k=k, d=d, n_query=m)
        with tempfile.TemporaryDirectory() as td:
            save_snapshot(
                td, self.g, policy.logits, policy.family, reference_ids=self.ids,
                config_path=self.config,
                data_seed=self.spec["seed"], reference_seed=20261107,
                objective="test", tau=.005, beta=.4, lam=.8, horizon=4,
                validation_x=self.vx, query_starts=policy.query_start_logits,
                query_start_source=policy.query_start_source,
                policy_variant=policy.variant)
            p = Path(td) / "model.json"
            model = json.loads(p.read_text(encoding="utf8"))
            model["parameters"]["policy_construction"]["start_by_class"][0][0] += .2
            p.write_text(json.dumps(model), encoding="utf8")
            with self.assertRaises(ValueError):
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
