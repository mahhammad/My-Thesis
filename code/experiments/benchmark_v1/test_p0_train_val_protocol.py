"""P0 Train/Validation-only checks; intentionally never generate or open Test."""
from pathlib import Path
import sys
import unittest

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "code" / "data_generator"))

from generate import load_config, bayes_posterior
from dataset_adapter import dataset_train_val


class SyntheticP0WithoutTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spec = load_config(ROOT / "code/data_generator/config.json")

    def test_train_validation_reproducible_and_counts(self):
        a = dataset_train_val(self.spec)
        b = dataset_train_val(self.spec)
        self.assertEqual(set(a), {"train", "validation"})
        self.assertEqual(set(b), {"train", "validation"})
        for name, n, counts in [
            ("train", 300, [72, 147, 81]),
            ("validation", 90, [13, 49, 28]),
        ]:
            x, y = a[name]
            xb, yb = b[name]
            self.assertEqual(x.shape, (n, 5))
            np.testing.assert_array_equal(y, yb)
            np.testing.assert_array_equal(x, xb)
            np.testing.assert_array_equal(np.bincount(y, minlength=3), counts)
            self.assertTrue(np.isfinite(x[:, 0]).all())  # Feature 0 has zero MCAR loss
            self.assertFalse(np.isinf(x).any())
            self.assertTrue(np.isnan(x[:, 2]).any())  # Feature-specific missingness
            self.assertTrue(np.isnan(x[:, 4]).any())

    def test_oracle_uses_only_observed_features(self):
        for x, _ in dataset_train_val(self.spec).values():
            p = bayes_posterior(self.spec, x)
            self.assertEqual(p.shape, (len(x), 3))
            self.assertTrue(np.isfinite(p).all())
            np.testing.assert_allclose(p.sum(-1), 1.0, rtol=0, atol=1e-12)

    def test_noise_features_and_mcar_rates_are_configured(self):
        self.assertEqual(self.spec["noise"], [2, 4])
        np.testing.assert_array_equal(self.spec["missing"], [0., .05, .12, .05, .20])
        np.testing.assert_allclose(self.spec["gamma"], [.8, 1.2, .5, 1., .6])
        np.testing.assert_allclose(self.spec["pi"], [.25, .45, .30])


if __name__ == "__main__":
    unittest.main()
