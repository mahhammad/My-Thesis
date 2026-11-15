"""E09 saved signed-margin report QA. No walk or training is executed."""
from __future__ import annotations
import csv
import json
import unittest
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
POLICIES=("uniform","trained_shared")
HS=(1,2,3,4,8,12)


class MarginPresentationQA(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw=json.loads((HERE/"results"/"metrics_full.json").read_text(encoding="utf8"))
        cls.margin=json.loads((HERE/"results"/"margin_summary.json").read_text(encoding="utf8"))
        cls.cases={(c["policy"],c["H"]):c for c in cls.raw["cases"]}
        cls.rows={(r["policy"],r["H"],r["split"]):r for r in cls.margin["rows"]}

    def test_all_policy_depth_and_split_cases_covered(self):
        expected={(p,h,s) for p in POLICIES for h in HS for s in ("train","validation")}
        self.assertEqual(set(self.rows),expected)
        self.assertEqual(len(self.margin["rows"]),24)
        self.assertAlmostEqual(self.margin["tau"],.005,places=14)
        for key in ("Test_accessed","trained_policy_updated","walker_executed"):
            self.assertFalse(self.margin[key])

    def test_every_margin_quantile_is_from_the_common_reporter(self):
        for (p,h,s),row in self.rows.items():
            old=self.cases[p,h][s]
            values=np.asarray(old["per_sample"]["margins"])
            self.assertEqual(len(values),300 if s=="train" else 90)
            for name,q in (("min",0.),("p10",.1),("p25",.25),("median",.5),
                           ("p75",.75),("p90",.9),("max",1.)):
                self.assertAlmostEqual(row["quantiles"][name],
                                       old["margin_quantiles"][str(q)],places=12)
                self.assertAlmostEqual(row["quantiles"][name],
                                       float(np.quantile(values,q)),places=12)

    def test_balanced_losses_and_nonpositive_counts(self):
        for (p,h,s),row in self.rows.items():
            src=self.cases[p,h][s]
            labels=np.asarray(src["per_sample"]["labels"])
            margins=np.asarray(src["per_sample"]["margins"])
            self.assertAlmostEqual(row["hard_margin_error_balanced"],
                                   src["hard_margin_error_conservative"],places=13)
            self.assertAlmostEqual(row["sigmoid_margin_loss_balanced"],
                                   src["balanced_sigmoid_margin_loss"],places=13)
            self.assertEqual(row["nonpositive_count"],int(np.sum(margins<=0)))
            self.assertAlmostEqual(row["class_balanced_mean_margin"],
                float(np.mean([margins[labels==c].mean() for c in range(3)])),places=13)
            self.assertEqual(row["ties"],src["tie_count"])

    def test_csv_has_all_values_without_changing_source(self):
        with (HERE/"results"/"margin_summary.csv").open(encoding="utf8",newline="") as f:
            cells=list(csv.DictReader(f))
        self.assertEqual(len(cells),24)
        for r in cells:
            x=self.rows[r["policy"],int(r["H"]),r["split"]]
            self.assertAlmostEqual(float(r["median"]),x["quantiles"]["median"],places=13)
            self.assertAlmostEqual(float(r["sigmoid_margin_loss_balanced"]),
                                   x["sigmoid_margin_loss_balanced"],places=13)

    def test_tex_margin_table_and_both_figures_trace_to_saved_numbers(self):
        table=(HERE/"tables"/"margin_by_H.tex").read_text(encoding="utf8")
        self.assertEqual(table.count(r"\label{tab:ch4-e09-margin-by-h}"),1)
        self.assertEqual(table.count(r"\begin{table}[htbp]"),1)
        for p in POLICIES:
            for h in HS:
                train=self.rows[p,h,"train"]
                val=self.rows[p,h,"validation"]
                self.assertIn(f"{train['quantiles']['median']:.5f}",table)
                self.assertIn(f"{val['quantiles']['median']:.5f}",table)
        for split in ("train","validation"):
            path=HERE/"figures"/f"{split}_margin_median_vs_H.tex"
            src=path.read_text(encoding="utf8")
            self.assertEqual(src.count(r"\label{fig:ch4-e09-"+split+"-margin}"),1)
            self.assertEqual(src.count(r"\begin{tikzpicture}"),1)
            self.assertEqual(src.count(r"\begin{tikzpicture}["),1)
            self.assertEqual(src.count(r"\end{tikzpicture}"),1)

    def test_chapter_and_report_include_train_and_validation(self):
        manuscript=(ROOT/"tex"/"chapter4.tex").read_text(encoding="utf8")
        a=manuscript.index(r"\section{اثر سقف واقعی طول گشت")
        b=manuscript.index(r"\section{اعتبارسنجی محاسبه ماتریسی",a)
        sect=manuscript[a:b]
        for name in ("tables/margin_by_H.tex","figures/train_margin_median_vs_H.tex",
                     "figures/validation_margin_median_vs_H.tex"):
            self.assertIn(name,sect)
        report=(HERE/"README_FA.md").read_text(encoding="utf8")
        self.assertIn("## حاشیه تصمیم: Train و Validation",report)
        self.assertIn("زیان سیگموئیدی متوازن",report)


if __name__=="__main__":
    unittest.main()
