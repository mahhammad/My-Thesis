"""No-dataset/no-Train/no-Test unit tests for the exact 4.5 Guided mixture."""
import importlib.util
import json
import math
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("guided_probability_policy", ROOT / "policy.py")
policy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(policy)


class GuidedInformativeUniformMixtureTests(unittest.TestCase):
    def test_design_matrix_and_generator_feature_roles(self):
        meta = json.loads((ROOT / "guided_policy.json").read_text("utf8"))
        gen = json.loads((ROOT.parents[2] / "code/data_generator/config.json").read_text("utf8"))
        self.assertEqual(meta["informative_features"], [0, 1, 3])
        self.assertEqual(meta["noise_features"], [2, 4])
        self.assertEqual(meta["start_row_index"], 5)
        self.assertEqual(meta["transition_weights"], [[1,1,0,1,0]]*6)
        # Both noise dimensions have identical class-conditional means and
        # standard deviations, with no correlation to informative features.
        for j in meta["noise_features"]:
            self.assertEqual(len({row[j] for row in gen["means"]}), 1)
            self.assertEqual(len({row[j] for row in gen["stds"]}), 1)
            for k in meta["informative_features"]:
                self.assertEqual(gen["correlation"][j][k], 0)
        self.assertEqual(gen["missing_percent"][0], 0)

    def test_all_available_equal_support_and_zero_noise(self):
        g = policy.guided_probabilities((1,1,1,1,1))
        self.assertEqual(g, (1/3,1/3,0,1/3,0))
        self.assertEqual(sum(g), 1)
        self.assertEqual(len(policy.transition_matrix((1,)*5, 1.)), 6)
        for row in policy.transition_matrix((1,)*5, 1.):
            self.assertEqual(row, g)

    def test_missing_observations_are_excluded_before_normalization(self):
        valid = (1,0,1,1,0)  # Feature 1 missing, feature 2 noise present
        self.assertEqual(policy.guided_probabilities(valid), (.5,0,0,.5,0))
        self.assertEqual(policy.uniform_probabilities(valid), (1/3,0,1/3,1/3,0))
        self.assertEqual(policy.guided_probabilities((1,0,0,0,1)), (1,0,0,0,0))

    def test_probability_mixture_not_softmax_temperature(self):
        valid = (1,1,1,1,1)
        self.assertEqual(policy.mixed_probabilities(valid, 0), (.2,)*5)
        self.assertEqual(policy.mixed_probabilities(valid, 1), (1/3,1/3,0,1/3,0))
        p = policy.mixed_probabilities(valid, .5)
        for j in (0,1,3):
            self.assertAlmostEqual(p[j], .5*.2 + .5/3, places=14)
        for j in (2,4):
            self.assertAlmostEqual(p[j], .1, places=14)
        self.assertAlmostEqual(sum(p), 1, places=14)

    def test_unreachable_noise_support_and_alpha_edges(self):
        self.assertEqual(policy.mixed_probabilities((0,0,1,0,1), 0),
                         (0,0,.5,0,.5))
        for alpha in (.01,.5,1):
            with self.assertRaises(ValueError):
                policy.mixed_probabilities((0,0,1,0,1), alpha)
        with self.assertRaises(ValueError):
            policy.mixed_probabilities((0,0,0,0,0), 0)
        for alpha in (-.1,1.01,math.nan):
            with self.assertRaises(ValueError):
                policy.mixed_probabilities((1,)*5,alpha)


if __name__ == "__main__":
    unittest.main()
