"""Regression tests for exact probability-space mixture vs old logits engine."""
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "code/data_generator"))
sys.path.insert(0, str(HERE.parent / "benchmark_v1"))
from generate import load_config
from dataset_adapter import dataset_train_val
from engine import DT, ExactMaskedWalk
from probability_walk import ProbabilityMixtureWalk


class ProbabilityMixtureWalkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = load_config(ROOT / "code/data_generator/config.json")
        data = dataset_train_val(spec)
        cls.x, cls.y = data["train"]
        cls.q, cls.qy = data["validation"]
        torch.set_num_threads(1)
        cls.graph = ProbabilityMixtureWalk(
            cls.x, cls.y, spec["gamma"], beta=.4,
            lam=.8, horizon=12, query_x=cls.q
        )

    def test_alpha_zero_exact_matches_original_engine_train_and_validation(self):
        graph = self.graph
        z = torch.zeros((6, 5), dtype=DT)
        with torch.no_grad():
            old_train = graph.represent_train(z, "shared")
            old_val = graph.represent_query(z, "shared")
            new_train = graph.represent_train_mixture(0)
            new_val = graph.represent_validation_mixture(0)
        torch.testing.assert_close(new_train, old_train, atol=5e-13, rtol=0)
        torch.testing.assert_close(new_val, old_val, atol=5e-13, rtol=0)

    def test_alpha_one_guided_has_exact_zero_noise_for_all_valid_masks(self):
        graph = self.graph
        masks = (graph.mass > 1e-12) & graph.available
        for alpha in [0., .25, .5, .75, 1.]:
            p = graph._mixture(masks, alpha)
            self.assertTrue(torch.all(p >= 0))
            torch.testing.assert_close(
                p.sum(1), torch.ones(len(masks), dtype=DT),
                atol=1e-14, rtol=0
            )
            if alpha == 1.:
                self.assertTrue(torch.all(p[:, [2, 4]] == 0))
                # Informative valid features have equal probability.
                infos = (masks & graph.informative_mask).to(DT)
                n = infos.sum(1, keepdim=True)
                torch.testing.assert_close(
                    p, infos/n, atol=1e-14, rtol=0
                )

    def test_every_alpha_train_validation_stays_on_probability_simplex(self):
        for alpha in (0., .25, .5, .75, 1.):
            with torch.no_grad():
                t = self.graph.represent_train_mixture(alpha)
                v = self.graph.represent_validation_mixture(alpha)
            for r in (t, v):
                self.assertTrue(torch.all(r >= -1e-13))
                torch.testing.assert_close(
                    r.sum(1), torch.ones(len(r), dtype=DT),
                    atol=1e-8, rtol=0
                )


if __name__ == "__main__":
    unittest.main()
