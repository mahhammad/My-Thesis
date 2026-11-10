"""Regression checks for the Chapter-4 unified metric registry.

No dataset generation or experiment training, simple probability fixtures only.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
import unittest
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT/"code"/"policy_evaluator"))
from report_snapshot import standard_metrics
from metrics import center_metrics, jsd, sqrt_jsd


class SharedReportContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.contract = json.loads((HERE.parent/"COMMON_METRICS_CONTRACT.json").read_text("utf8"))
        cls.centers = np.array([
            [.75,.20,.05], [.15,.75,.10], [.08,.22,.70]
        ],float)
        cls.labels=np.array([0,1,2,0,1,2],int)
        p=np.array([
            [.70,.23,.07], [.18,.72,.10], [.09,.21,.70],
            [.73,.22,.05], [.17,.69,.14], [.06,.27,.67]
        ],float)
        cls.by=np.broadcast_to(p[:,None,:],(len(p),3,3))

    def test_required_groups_and_valid_missingness_codes(self):
        ids={x["group"]:x for x in self.contract["metric_groups"]}
        self.assertEqual(set(ids),{
            "density","center_geometry","classification",
            "classification_surrogates","experiment_specific"
        })
        self.assertEqual(self.contract["calculation"]["logs"],"natural")
        self.assertIn("not_measured",self.contract["calculation"]["unavailable_states"])

    def test_report_has_all_three_R_and_full_center_geometry(self):
        result=standard_metrics(self.by,self.centers,self.labels,.005,
                                split="train",per_sample=True)
        self.assertEqual(result["R"],result["R_KL"])
        required=(
            "R_JS","R_sqrt_JS","R_KL","R","I",
            "T","BT","TNL","BTNL","center_geometry",
            "nearest_center_sqrt_JS_per_class","pairwise_center_sqrt_JS",
            "min_pairwise_sqrt_JS","mean_pairwise_sqrt_JS"
        )
        for key in required:
            self.assertIn(key,result)
            self.assertIsNotNone(result[key],key)
        self.assertEqual(len(result["per_class"]),3)
        for row in result["per_class"]:
            for key in ("R_JS","R_sqrt_JS","R_KL"):
                self.assertIn(key,row)

    def test_center_geometry_reuses_chapter3_metric_engine(self):
        r=standard_metrics(self.by,self.centers,self.labels,.005,
                           split="train",per_sample=False)
        g=center_metrics(self.centers)
        for key in ("T","BT","TNL","BTNL","I"):
            self.assertAlmostEqual(r[key],g[key],places=12)
        p=np.asarray(r["pairwise_center_sqrt_JS"])
        self.assertEqual(p.shape,(3,3))
        expected=sqrt_jsd(self.centers[:,None,:],self.centers[None,:,:])
        np.testing.assert_allclose(p,expected,atol=1e-12)
        self.assertAlmostEqual(r["min_pairwise_sqrt_JS"],
                               min(p[0,1],p[0,2],p[1,2]))
        self.assertAlmostEqual(r["mean_pairwise_sqrt_JS"],
                               (p[0,1]+p[0,2]+p[1,2])/3)
        self.assertEqual(len(r["nearest_center_sqrt_JS_per_class"]),3)

    def test_chapter3_references_exist_and_chapter4_registry_is_4_2_subsection(self):
        c3=(ROOT/"tex/chapter3.tex").read_text("utf8")
        c4=(ROOT/"tex/chapter4.tex").read_text("utf8")
        required=[
            "eq:ch3-R-average","eq:ch3-R-JS-forms",
            "subsec:ch3-center-separation-criteria",
            "eq:ch3-js-center-separation",
            "eq:ch3-js-exact-balanced-accuracy",
            "eq:ch3-js-sigmoid-balanced-objective",
        ]
        for label in required:
            self.assertIn("\\label{"+label+"}",c3)
            self.assertIn("\\ref{"+label+"}",c4) if "\\ref{"+label+"}" in c4 else None
        self.assertIn("\\label{subsec:ch4-common-evaluation-contract}",c4)
        self.assertLess(
            c4.index("\\label{subsec:ch4-common-evaluation-contract}"),
            c4.index("\\section{اعتبارسنجی همگرایی توزیع انتقال یک‌گامی")
        )


if __name__=="__main__":
    unittest.main()
