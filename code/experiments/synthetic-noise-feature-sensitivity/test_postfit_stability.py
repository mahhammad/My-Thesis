"""Read-only regression tests for frozen E14 postfit stability outputs."""
from __future__ import annotations
import unittest
import numpy as np
import geometry_stability_analysis as geom
import feature_stability_analysis as pref

class PostfitStabilityTests(unittest.TestCase):
    def test_jsd_identity_and_symmetry(self):
        p=np.array([.2,.5,.3])
        q=np.array([.7,.2,.1])
        self.assertAlmostEqual(float(geom.jsd(p,p)),0,places=12)
        self.assertAlmostEqual(float(geom.jsd(p,q)),float(geom.jsd(q,p)),places=12)
        self.assertGreater(float(geom.jsd(p,q)),0)

    def test_canonical_geometry_and_sample_alignment(self):
        rows,compact,groups=geom.build()
        self.assertEqual((len(rows),len(compact),len(groups)),(36,36,12))
        for row in rows:
            self.assertEqual(row["train"]["n"],300)
            self.assertEqual(row["validation"]["n"],90)
            for split in ("train","validation"):
                t=row[split]["outcome_transitions"]
                self.assertEqual(sum(t.values()),row[split]["n"])
            self.assertTrue(np.isfinite(row["center_geometry"]["mean_center_drift"]))
            if row["noise_count"]==0:
                self.assertLess(row["center_geometry"]["mean_center_drift"],1e-12)
                self.assertEqual(row["validation"]["prediction_flips"],0)

    def test_feature_preference_contract(self):
        rows,groups=pref.build()
        self.assertEqual((len(rows),len(groups)),(108,36))
        for row in rows:
            self.assertAlmostEqual(row["main_share"]+row["noise_share"],1,places=10)
            self.assertTrue(0<=row["main_conditional_TV"]<=1)
            self.assertEqual(row["rank_reversal_count"]+
                             row["rank_tie_pair_count"]+
                             row["rank_preserved_pairs"],3)
            self.assertAlmostEqual(sum(row[f"feature_{a}_conditional_share"]
                                       for a in pref.GOOD),1,places=10)
            if row["noise_count"]==0:
                self.assertAlmostEqual(row["main_conditional_TV"],0,places=12)

if __name__=="__main__":
    unittest.main()
