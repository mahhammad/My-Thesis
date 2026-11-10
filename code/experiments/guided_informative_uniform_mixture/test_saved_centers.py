"""Data-only regression checks for five E-CH4-05 class centers and R_JS.

No reruns, no model fitting, no Test: verifies persisted original artifact
center means, pairwise sqrt-JSD and per-class dispersion.
"""
from pathlib import Path
import json
import math
import unittest

HERE = Path(__file__).resolve().parent


def entropy(p):
    return -sum(v*math.log(v) for v in p if v > 0)


def sqrt_jsd(p,q):
    m=[(a+b)/2 for a,b in zip(p,q)]
    return math.sqrt(max(0.,entropy(m)-(entropy(p)+entropy(q))/2))


class SavedCenterEvidence(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.a=json.loads((HERE/"results/centers_comparison.json").read_text(encoding="utf8"))["results"]
        cls.b=json.loads((HERE/"results/summary.json").read_text(encoding="utf8"))["results"]

    def test_centers_are_probabilities_and_pairwise_distances_match(self):
        self.assertEqual(len(self.a),5)
        for item in self.a:
            self.assertEqual(len(item["centers"]),3)
            for v in item["centers"]:
                self.assertEqual(len(v),3)
                self.assertTrue(all(0<=w<=1 for w in v))
                self.assertAlmostEqual(sum(v),1.,delta=1e-12)
            for actual,(i,j) in zip(item["pairwise_sqrt_JS"],[(0,1),(0,2),(1,2)]):
                self.assertAlmostEqual(actual,sqrt_jsd(item["centers"][i],item["centers"][j]),delta=1e-12)

    def test_rjs_is_class_balanced_and_consistent_with_primary_summary(self):
        for row,ref in zip(self.a,self.b):
            self.assertEqual(row["alpha"],ref["alpha"])
            self.assertAlmostEqual(row["I_train"],ref["I_train"],delta=1e-12)
            self.assertAlmostEqual(row["R_JS_train"],ref["R_JS_train"],delta=1e-12)
            self.assertAlmostEqual(sum(row["R_JS_classwise"])/3,ref["R_JS_train"],delta=1e-12)

    def test_every_pairwise_center_distance_and_class_dispersion_increases(self):
        for i in range(3):
            distances=[row["pairwise_sqrt_JS"][i] for row in self.a]
            spreads=[row["R_JS_classwise"][i] for row in self.a]
            self.assertTrue(all(u<v for u,v in zip(distances,distances[1:])))
            self.assertTrue(all(u<v for u,v in zip(spreads,spreads[1:])))


if __name__=="__main__":
    unittest.main()
