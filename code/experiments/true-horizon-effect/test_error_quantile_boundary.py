"""E09 first quantile after empirical misclassification rate, no new graph walk."""
import json
import unittest
from pathlib import Path

HERE=Path(__file__).resolve().parent


class ErrorBoundaryQuantileQA(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        p=HERE/"results"/"error_quantile_boundary.json"
        if not p.exists():
            raise unittest.SkipTest("Quantile output generated once after source tests")
        cls.b=json.loads(p.read_text(encoding="utf8"))
        cls.margin=json.loads((HERE/"results"/"margin_analysis.json").read_text(encoding="utf8"))
        cls.quant=json.loads((HERE/"results"/"margin_summary.json").read_text(encoding="utf8"))

    def test_full_policy_h_depth_split_coverage(self):
        rows=self.b["rows"]
        self.assertEqual(len(rows),24)
        expected={(p,h,s) for p in ("uniform","trained_shared")
                  for h in (1,2,3,4,8,12) for s in ("train","validation")}
        self.assertEqual({(r["policy"],r["H"],r["split"]) for r in rows},expected)
        self.assertFalse(self.b["historical_test_accessed"])
        self.assertFalse(self.b["walker_invoked"])
        self.assertFalse(self.b["training_performed"])

    def test_strict_first_quantile_against_original_grid(self):
        marg={(r["policy"],r["H"],r["split"]):r for r in self.margin["rows"]}
        summ={(r["policy"],r["H"],r["split"]):r for r in self.quant["rows"]}
        levels=[(0.,"min"),(.1,"p10"),(.25,"p25"),(.5,"median"),
                (.75,"p75"),(.9,"p90"),(1.,"max")]
        for r in self.b["rows"]:
            key=(r["policy"],r["H"],r["split"])
            count=marg[key]["error_count"]
            n=marg[key]["n"]
            self.assertEqual(r["errors"],count)
            self.assertEqual(r["n"],n)
            self.assertAlmostEqual(r["observed_unweighted_error_fraction"],count/n)
            choices=[(x,k) for x,k in levels if x>count/n]
            selected=choices[0] if choices else None
            self.assertEqual(r["first_reported_percentile_above_error_fraction"],
                             selected[0] if selected else None)
            if selected:
                self.assertAlmostEqual(
                    r["margin_at_first_reported_percentile_above_error_fraction"],
                    summ[key]["quantiles"][selected[1]],places=13)
            if count:
                self.assertLess(r["last_misclassified_margin"],0)
            if count<n:
                self.assertGreater(r["first_correct_margin"],0)

    def test_actual_first_positive_and_reported_quantile_are_distinct(self):
        a=next(x for x in self.b["rows"] if
               x["policy"]=="uniform" and x["H"]==1 and x["split"]=="train")
        self.assertEqual(a["errors"],56)
        self.assertEqual(a["first_reported_percentile_above_error_fraction"],.25)
        self.assertAlmostEqual(a["margin_at_first_reported_percentile_above_error_fraction"],
                               .0029723061637359494,places=13)
        self.assertAlmostEqual(a["first_correct_margin"],
                               .00004607562154623768,places=13)
        self.assertNotAlmostEqual(a["first_correct_margin"],
                                  a["margin_at_first_reported_percentile_above_error_fraction"],
                                  places=8)

    def test_tex_and_manuscript_refer_to_new_table(self):
        tex=(HERE/"tables"/"error_quantile_boundary.tex").read_text(encoding="utf8")
        self.assertEqual(tex.count(r"\label{tab:ch4-e09-first-quantile-after-error}"),1)
        chapter=(HERE.parents[2]/"tex"/"chapter4.tex").read_text(encoding="utf8")
        a=chapter.index(r"\section{اثر سقف واقعی طول گشت")
        b=chapter.index(r"\section{اعتبارسنجی محاسبه ماتریسی",a)
        self.assertIn("tables/error_quantile_boundary.tex",chapter[a:b])


if __name__=="__main__":
    unittest.main()
