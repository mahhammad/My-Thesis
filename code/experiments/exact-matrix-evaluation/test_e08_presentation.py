"""Lightweight E08 presentation and provenance checks; never executes the walker."""
import json
import re
from pathlib import Path
import unittest

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]

class TestE08Presentation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not (HERE/"results"/"experiment_report.json").is_file():
            raise unittest.SkipTest("E08 raw evidence not yet produced")
        cls.d=json.loads((HERE/"results"/"experiment_report.json").read_text(encoding="utf8"))
        cls.m=json.loads((HERE/"results"/"metrics_full.json").read_text(encoding="utf8"))
        cls.p=json.loads((HERE/"results"/"provenance.json").read_text(encoding="utf8"))

    def test_tex_tables_are_sourced_from_one_record(self):
        a=(HERE/"tables"/"truncation.tex").read_text(encoding="utf8")
        b=(HERE/"tables"/"monte_carlo.tex").read_text(encoding="utf8")
        c=(HERE/"tables"/"runtime.tex").read_text(encoding="utf8")
        for name in ("uniform","trained_shared"):
            for row in self.d["policy_results"][name]["truncation"]:
                self.assertIn(f"{row['epsilon_tail']:.5f}",a)
                self.assertIn(f"{row['train_mean_balanced_tv']:.5f}",a)
                self.assertIn(f"{row['validation_mean_balanced_tv']:.5f}",a)
            for row in self.d["monte_carlo"]["summary"]:
                if row["policy"]==name:
                    self.assertIn(f"{row['mean_tv']:.5f}",b)
            for row in self.d["timing"]:
                if row["policy"]==name:
                    self.assertIn(f"{row['median_seconds']:.5f}",c)
        for label in ("tab:ch4-matrix-truncation","tab:ch4-matrix-mc-check",
                      "tab:ch4-matrix-runtime"):
            self.assertEqual(sum(label in t for t in (a,b,c)),1)

    def test_bound_figure_has_three_nonflat_data_series(self):
        fig=(HERE/"figures"/"bound_and_max_tv.tex").read_text(encoding="utf8")
        paths=re.findall(r"plot coordinates \{([^}]+)\};",fig)
        self.assertEqual(len(paths),3)
        for line in paths:
            pts=re.findall(r"\(([^,]+),([^)]+)\)",line)
            self.assertEqual(len(pts),6)
            self.assertGreater(len({y for x,y in pts}),2)

    def test_only_section_48_uses_new_sourced_assets(self):
        chapter=(ROOT/"tex"/"chapter4.tex").read_text(encoding="utf8")
        first=chapter.index(r"\section{اعتبارسنجی محاسبه ماتریسی بازنمایی چندگامی}")
        last=chapter.index(r"\section{بازیابی معکوس سیاست از مراکز دست‌یافتنی}",first)
        segment=chapter[first:last]
        for path in (
            "tables/truncation.tex","tables/monte_carlo.tex",
            "tables/runtime.tex","figures/bound_and_max_tv.tex"
        ):
            self.assertIn(
                r"\input{code/experiments/exact-matrix-evaluation/"+path+"}",segment)
        self.assertNotIn("مرجع $72$",segment)
        self.assertNotIn("approx_bayes_seed",segment)
        self.assertEqual(chapter.count(r"\label{sec:ch4-exact-matrix-evaluation}"),1)

    def test_horizontal_self_transition_count_and_raw_data_persisted(self):
        g=self.d["graph_size"]
        self.assertEqual(g["m_sample_feature_nodes"],1366)
        self.assertEqual(g["directed_vertical_possible"],374024)
        self.assertEqual(g["horizontal_possible"],6334)
        self.assertTrue(g["horizontal_feature_self_choices_included"])
        for p in ("uniform","trained_shared"):
            for r in (2,3,4,6,8,12):
                self.assertIn(str(r),self.m["policies"][p])
        self.assertTrue((HERE/"results"/"raw_step_distributions.npz").exists())

    def test_original_checkpoint_and_no_test_evidence(self):
        self.assertEqual(self.p["archive_sha256"],
                         "9609eb96c207316fde26934cbef4668ba287a275d4485175d1da7e798a297040")
        self.assertTrue(self.p["original_archive_verified"])
        self.assertEqual(sorted(self.p["source_reference_ids"]),list(range(300)))
        self.assertFalse(self.p["historical_test_accessed"])
        self.assertFalse(self.d["training_performed"])
        self.assertEqual(len(self.d["monte_carlo"]["raw"]),72)

if __name__=="__main__":
    unittest.main()
