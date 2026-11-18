"""Independent invariants for percent-relative E14 postfit reporting."""
import unittest

from relative_stability_analysis import derive, percent


class RelativeStabilityTests(unittest.TestCase):
    def test_relative_definition(self):
        self.assertAlmostEqual(percent(.8,1.),-20.)
        self.assertAlmostEqual(percent(1.25,1.),25.)
        with self.assertRaises(AssertionError):
            percent(.1,0.)

    def test_full_noise_levels_and_baselines(self):
        runs, groups, allocation = derive()
        self.assertEqual(len(runs),36)
        self.assertEqual(len(groups),12)
        self.assertEqual(len(allocation),12)
        for group in groups:
            self.assertEqual(group["n_dependent_subsets"],
                             1 if group["noise_count"] in (0,5) else 4)
            if group["noise_count"]==0:
                self.assertAlmostEqual(group["I_pct_change_mean"],0.,places=10)
                self.assertAlmostEqual(group["mean_pairwise_pct_change_mean"],0.,places=10)
                self.assertAlmostEqual(group["BA_TV_0_60_pct_change_mean"],0.,places=10)
        for row in allocation:
            k=row["noise_count"]
            self.assertAlmostEqual(
                row["noise_share_h3_pct"]+row["informative_share_h3_pct"],100,places=9)
            self.assertAlmostEqual(row["uniform_noise_share_pct"],
                                    100*k/(3+k),places=9)
            self.assertAlmostEqual(
                row["noise_vs_uniform_pp"],
                row["noise_share_h3_pct"]-row["uniform_noise_share_pct"],places=9)
            for h in (1,2,3):
                self.assertGreaterEqual(row[f"informative_share_h{h}_pct"],-1e-8)
                self.assertLessEqual(row[f"informative_share_h{h}_pct"],100+1e-8)

if __name__=="__main__":
    unittest.main()
