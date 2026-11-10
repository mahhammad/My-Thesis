"""E09 LaTeX section and generated print assets must be grounded in stored results."""
import json
import unittest
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]


class E09Presentation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data=json.loads((HERE/"results"/"metrics_full.json").read_text(encoding="utf8"))
        cls.comp=json.loads((HERE/"results"/"comparisons.json").read_text(encoding="utf8"))
        cls.chapter=(ROOT/"tex"/"chapter4.tex").read_text(encoding="utf8")

    def test_section_order_and_unmodified_E08_identifier(self):
        text=self.chapter
        new=r"\section{اثر سقف واقعی طول گشت بر بازنمایی و طبقه‌بندی}"
        old=r"\section{اعتبارسنجی محاسبه ماتریسی بازنمایی چندگامی}"
        self.assertEqual(text.count(new),1)
        self.assertEqual(text.count(old),1)
        self.assertLess(text.index(new),text.index(old))
        self.assertEqual(text.count(r"\label{sec:ch4-true-horizon-effect}"),1)
        self.assertEqual(text.count(r"\label{sec:ch4-exact-matrix-evaluation}"),1)
        section=text[text.index(new):text.index(old)]
        for name in ("main_depth_effect.tex","true_H_vs_E08_truncation.tex"):
            self.assertIn(name,section)
        for name in ("validation_ba_vs_H.tex","dispersion_vs_H.tex"):
            self.assertIn(name,section)
        self.assertNotIn(r"\section{",section[10:])

    def test_two_tables_grounded_in_saved_cases(self):
        main=(HERE/"tables"/"main_depth_effect.tex").read_text(encoding="utf8")
        other=(HERE/"tables"/"true_H_vs_E08_truncation.tex").read_text(encoding="utf8")
        self.assertEqual(main.count(r"\begin{table}[htbp]"),1)
        self.assertEqual(other.count(r"\begin{table}[htbp]"),1)
        for case in self.data["cases"]:
            n=f"{case['train']['I']:.5f}"
            ba=f"{100*case['validation']['balanced_accuracy']:.2f}"
            self.assertIn(n,main)
            self.assertIn(ba,main)
            row=self.comp["true_H_vs_E08_fixed_H12_truncation"][
                f"{case['policy']}_H{case['H']}"]
            self.assertIn(f"{row['coefficient_l1']:.5f}",other)
            self.assertIn(f"{row['validation_mean_TV']:.5f}",other)

    def test_three_unique_figure_labels_and_no_engine_runtime_claim(self):
        for name,tag in (
            ("validation_ba_vs_H.tex","fig:ch4-e09-validation-ba"),
            ("center_information_vs_H.tex","fig:ch4-e09-I"),
            ("dispersion_vs_H.tex","fig:ch4-e09-RJS")):
            s=(HERE/"figures"/name).read_text(encoding="utf8")
            self.assertEqual(s.count("\\label{"+tag+"}"),1)
            self.assertEqual(s.count("\\begin{tikzpicture}["),1)
            self.assertEqual(s.count("\\end{tikzpicture}"),1)
        report=(HERE/"REPORT_FA.md").read_text(encoding="utf8")
        self.assertIn("زمان اجرای موتور گشت",report)
        self.assertIn("H=1,2,3,4,8,12",report)
        self.assertIn("از توزیع‌های گام‌به‌گام ذخیره‌شده E08",report)

    def test_research_journal_pre_reg_preserved(self):
        s=(ROOT/"research_journal"/"2026-11-10-E-CH4-09-depth-effect.md").read_text(encoding="utf8")
        self.assertIn("## پیش‌ثبت پیش از اجرای بازتحلیل",s)
        self.assertIn("## نتایج بازتحلیل پس از گذر کنترل‌ها",s)
        self.assertIn("بدون Merge",s)
        self.assertIn("ممیزی مستقل",s)


if __name__=="__main__":
    unittest.main()
