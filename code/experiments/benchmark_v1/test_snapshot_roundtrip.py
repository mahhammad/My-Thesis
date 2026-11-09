"""Small validation round-trip tests. Regenerate Train/Validation only; never Test.

Run: python -m unittest -v code/experiments/benchmark_v1/test_snapshot_roundtrip.py
"""
from pathlib import Path
import json
import sys
import tempfile
import unittest

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "code" / "data_generator"))

from generate import load_config
from dataset_adapter import dataset_train_val
from engine import DT, ExactMaskedWalk
from policy_interface import create_policy, evaluate_policy
from snapshot_pipeline import array_fingerprint, load_snapshot, save_snapshot


class ValidationRoundTrip(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.config = ROOT / "code/data_generator/config.json"
        cls.spec = load_config(cls.config)
        splits = dataset_train_val(cls.spec)
        tx, ty = splits["train"]
        vx, _ = splits["validation"]
        cls.ids = np.concatenate([
            np.flatnonzero(ty == c)[:5] for c in range(len(cls.spec["pi"]))
        ])
        cls.graph = ExactMaskedWalk(tx[cls.ids], ty[cls.ids],
                                    cls.spec["gamma"], horizon=3, query_x=vx)

    def save(self, folder, adapter):
        graph = self.graph
        return save_snapshot(
            folder, graph, adapter.logits, adapter.family,
            reference_ids=self.ids, config_path=self.config,
            config_reference="code/data_generator/config.json",
            data_seed=self.spec["seed"], reference_seed=20261107,
            objective="roundtrip_test", tau=.005,
            beta=.4, lam=.8, horizon=3,
            validation_x=graph.qx, query_starts=adapter.query_start_logits,
            query_start_source=adapter.query_start_source,
            policy_variant=adapter.variant, start_by_class=adapter.start_by_class
        )

    def reload_dataset(self, model, arrays):
        """Rebuild from saved seed and ordered reference IDs, not in-memory graph."""
        config = load_config(self.config)
        config["seed"] = model["data"]["data_seed"]
        splits = dataset_train_val(config)  # only Train/Validation
        tx, ty = splits["train"]
        vx, _ = splits["validation"]
        ids = arrays["reference_ids"]
        self.assertEqual(array_fingerprint(tx[ids]),
                         model["data"]["train_reference_x_sha256"])
        self.assertEqual(array_fingerprint(vx),
                         model["data"]["validation_x_sha256"])
        return {"ref_x": tx[ids], "ref_y": ty[ids], "query_x": vx}

    def replay(self, model, arrays, variant=None):
        dataset = self.reload_dataset(model, arrays)
        params = model["parameters"]
        if variant == "class_tied_node":
            components = model["policy_components"]
            policy = {
                "start_by_class": components["start_by_class"]["logits"],
                "intermediate_by_node_class": np.asarray(
                    params["logits"], dtype=np.float64
                )[:, :, :self.graph.d, :].tolist()
            }
            family = "class_tied_node"
        else:
            start = model["prediction"]["query_start"]
            policy = {
                "logits": params["logits"],
                "query_start_logits": None if start is None else start["logits"],
                "query_start_source": model["prediction"]["query_start_source"]
            }
            family = model["family"]
        regenerated = evaluate_policy(
            dataset, family, policy, gamma=params["gamma"],
            beta=params["beta"], lam=params["lambda"], horizon=params["H"]
        )
        self.assertEqual(regenerated["validation_status"], "available")
        np.testing.assert_allclose(
            regenerated["validation_by_hypothesis"],
            arrays["validation_by_hypothesis"], rtol=0, atol=1e-12
        )

    def test_free_node_and_node_class_full_round_trip(self):
        g = self.graph
        for family in ("node", "node_class"):
            with self.subTest(family=family), tempfile.TemporaryDirectory() as td:
                shape = ((g.n, g.d + 1, g.d) if family == "node" else
                         (g.k, g.n, g.d + 1, g.d))
                logits = (torch.arange(np.prod(shape), dtype=DT)
                          .reshape(shape) * .002)
                start_shape = ((len(g.qx), g.d) if family == "node" else
                               (g.k, len(g.qx), g.d))
                start = (np.arange(np.prod(start_shape), dtype=np.float64)
                         .reshape(start_shape) * .003)
                source = "roundtrip_fixed_starts_no_query_labels"
                adapter = create_policy(
                    family, {"logits": logits, "query_start_logits": start,
                             "query_start_source": source},
                    n=g.n, k=g.k, d=g.d, n_query=len(g.qx)
                )
                self.save(td, adapter)
                model, arrays = load_snapshot(td)
                self.assertEqual(model["prediction"]["query_start"]["shape"],
                                 list(start_shape))
                self.assertEqual(model["prediction"]["query_start_source"], source)
                self.assertEqual(
                    model["prediction"]["query_start"]["sha256"],
                    array_fingerprint(start)
                )
                self.replay(model, arrays)
                path = Path(td) / "model.json"
                tampered = json.loads(path.read_text(encoding="utf8"))
                if family == "node":
                    tampered["prediction"]["query_start"]["logits"][0][0] = 9.0
                else:
                    tampered["prediction"]["query_start"]["logits"][0][0][0] = 9.0
                path.write_text(json.dumps(tampered), encoding="utf8")
                with self.assertRaisesRegex(ValueError, "SHA256"):
                    load_snapshot(td)

    def test_hybrid_start_components_rule_and_round_trip(self):
        g = self.graph
        start = (torch.arange(g.k * g.d, dtype=DT)
                 .reshape(g.k, g.d) * .07)
        middle = (torch.arange(g.k * g.n * g.d * g.d, dtype=DT)
                  .reshape(g.k, g.n, g.d, g.d) * .001)
        adapter = create_policy(
            "class_tied_node",
            {"start_by_class": start, "intermediate_by_node_class": middle},
            n=g.n, k=g.k, d=g.d, n_query=len(g.qx)
        )
        with tempfile.TemporaryDirectory() as td:
            self.save(td, adapter)
            model, arrays = load_snapshot(td)
            self.assertEqual(model["policy_variant"], "class_tied_node")
            independent = model["policy_components"]["start_by_class"]
            self.assertEqual(independent["shape"], [g.k, g.d])
            self.assertEqual(independent["sha256"], array_fingerprint(start.numpy()))
            self.assertEqual(
                model["policy_components"]["composition_rule"],
                "intermediate_plus_tied_class_start_row"
            )
            self.replay(model, arrays, variant="class_tied_node")
            self.replay(model, arrays)  # same output from effective node_class tensor
            path = Path(td) / "model.json"
            changed = json.loads(path.read_text(encoding="utf8"))
            # Even a valid independent-start checksum cannot conceal disagreement
            # with the effective policy's tied rows.
            changed["policy_components"]["start_by_class"]["logits"][0][0] += .5
            independent_new = np.asarray(
                changed["policy_components"]["start_by_class"]["logits"],
                dtype=np.float64
            )
            changed["policy_components"]["start_by_class"]["sha256"] = (
                array_fingerprint(independent_new)
            )
            path.write_text(json.dumps(changed), encoding="utf8")
            with self.assertRaisesRegex(ValueError, "does not reconstruct"):
                load_snapshot(td)

    def test_node_validation_without_start_remains_undefined(self):
        g = self.graph
        zeros = torch.zeros((g.n, g.d + 1, g.d), dtype=DT)
        adapter = create_policy("node", zeros, n=g.n, k=g.k, d=g.d,
                                n_query=len(g.qx))
        with tempfile.TemporaryDirectory() as td:
            self.save(td, adapter)
            model, arrays = load_snapshot(td)
            self.assertEqual(model["prediction"]["validation_status"],
                             "not_defined_without_query_start")
            self.assertIsNone(model["prediction"]["query_start"])
            self.assertEqual(arrays["validation_by_hypothesis"].shape, (0,g.k,g.k))


if __name__ == "__main__":
    unittest.main()
