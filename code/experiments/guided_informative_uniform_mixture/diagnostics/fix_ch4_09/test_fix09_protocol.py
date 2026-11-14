"""FIX-CH4-09: light saved-evidence/protocol regression; no Monte Carlo reruns."""
import json
from pathlib import Path
import unittest
import sys
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[4]
BASE=HERE.parents[1]
LEGACY=ROOT/"code"/"policy_evaluator"
sys.path.insert(0,str(LEGACY))
from metrics import classify

class TestFix09Regression(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.a=json.loads((HERE/"results"/"controlled_uniform_replay.json").read_text(encoding="utf8"))
        cls.b=json.loads((HERE/"results"/"mc_convergence.json").read_text(encoding="utf8"))

    def test_legacy_rounding_and_exact_e05_still_match(self):
        a=self.a
        self.assertTrue(a["replayed_MC_BA_equals_historical_to_two_decimals"])
        self.assertAlmostEqual(a["cases"]["mc_val_mc_centers"]["balanced_accuracy"],.8825222396650968,places=14)
        self.assertAlmostEqual(a["cases"]["exact_val_exact_centers"]["balanced_accuracy"],.9676870748299321,places=14)
        self.assertTrue(a["reference"]["E05_exact_predictions_match_per_sample"])
        self.assertLess(a["reference"]["E05_exact_center_max_abs_error"],1e-12)

    def test_four_crossed_decisions_and_no_tie(self):
        a=self.a
        for result in a["cases"].values():
            c=np.asarray(result["confusion_true_rows_pred_cols"])
            np.testing.assert_array_equal(c.sum(axis=1),[13,49,28])
            self.assertEqual(result["ties"],0)
            self.assertEqual(len(result["predictions"]),90)
            self.assertAlmostEqual(float(np.mean(np.diag(c)/c.sum(axis=1))),
                                   result["balanced_accuracy"],places=14)
        self.assertAlmostEqual(a["cases"]["mc_val_mc_centers"]["balanced_accuracy"],
                               a["cases"]["mc_val_exact_centers"]["balanced_accuracy"],places=14)
        self.assertEqual(a["disagreement_counts"]["MC_vs_exact"],11)
        self.assertEqual(a["disagreement_counts"]["MC_query_only_vs_exact"],11)
        self.assertEqual(a["disagreement_counts"]["MC_centers_only_vs_exact"],1)

    def test_fixed_data_provenance_and_no_test_leakage(self):
        a,b=self.a,self.b
        self.assertEqual(a["data"]["data_seed"],20261102)
        self.assertEqual(b["MC_seed"],20261103)
        self.assertEqual(a["policy"]["MC_walks_per_origin"],300)
        self.assertTrue(a["data"]["E08_fingerprint_match_by_split"]["train_labels"])
        self.assertFalse(a["data"]["E08_fingerprint_match_by_split"]["train_features"])
        self.assertFalse(a["data"]["E08_fingerprint_match_by_split"]["validation_features"])
        self.assertFalse(a["data"]["original_historical_generated_data_byte_identity_verifiable"])
        self.assertFalse(a["historical_test_loaded"])
        self.assertFalse(a["retrain"])
        self.assertFalse(b["historical_test_accessed"])
        self.assertFalse(b["research_new_training"])

    def test_budget_result_same_confusion_but_not_identical_predictions(self):
        b=self.b
        rows=b["rows"]
        self.assertEqual([x["walks"] for x in rows],[300,1200,4800,9600])
        self.assertEqual([x["validation_disagreement_count_vs_exact"] for x in rows],
                         [11,4,2,2])
        e=self.a["cases"]["exact_val_exact_centers"]
        self.assertEqual(rows[-1]["validation_confusion"],
                         e["confusion_true_rows_pred_cols"])
        self.assertNotEqual(rows[-1]["validation_predictions"],
                            e["predictions"])
        self.assertAlmostEqual(rows[-1]["validation_BA"],
                               e["balanced_accuracy"],places=14)
        self.assertTrue(all(i["validation_TV_mean"]>j["validation_TV_mean"]
                            for i,j in zip(rows,rows[1:])))
        self.assertFalse(b["budgets_are_nested"])

    def test_historical_tables_unchanged_and_nll_is_not_decision(self):
        h=(BASE/"tables"/"historical_fixed_policy.tex").read_text(encoding="utf8")
        mc=(BASE/"tables"/"score_mc_fixed_policy.tex").read_text(encoding="utf8")
        exact=(BASE/"tables"/"exact_fixed_policy.tex").read_text(encoding="utf8")
        self.assertIn("88.25",h)
        self.assertIn("88.25",mc)
        self.assertIn("96.77",exact)
        self.assertIn("1.05791",h)
        self.assertEqual(self.a["policy"]["policy_family"],"shared, logits all zeros")

    def test_lowest_class_on_identical_js_scores(self):
        centers=np.eye(3)
        uniform=np.array([[1/3,1/3,1/3]])
        pred=classify(uniform,centers,metric="js_similarity")["prediction"]
        self.assertEqual(pred.tolist(),[0])

    def test_thesis_only_replaces_unresolved_caveat(self):
        s=(ROOT/"tex"/"chapter4.tex").read_text(encoding="utf8")
        self.assertNotIn("سهم خطای نمونه‌گیری و تفاوت‌های احتمالی پیاده‌سازی",s)
        self.assertIn("خطای نمونه‌گیری بازنمایی",s)
        self.assertIn("$9600$ گشت",s)
        self.assertIn("برابری بایت‌به‌بایت",s)
        self.assertEqual(s.count(r"\label{sec:ch4-exact-matrix-evaluation}"),1)

if __name__=="__main__":
    unittest.main()
