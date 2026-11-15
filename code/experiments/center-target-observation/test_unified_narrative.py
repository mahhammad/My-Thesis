"""N-CH4-01 evidence-only numeric and LaTeX presentation checks.
Archival provenance tests in test_replay_presentation.py remain intact.
No Monte Carlo, training, historical Test access, or heavy calculations.
"""
from __future__ import annotations
import json
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CHAPTER = ROOT.parents[2] / "tex/chapter4.tex"

class UnifiedTrajectoryNarrative(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        j=json.loads((ROOT/"results/metrics_full.json").read_text("utf-8"))
        cls.cases=j["replayed_fixed_targets"]["cases"]
        cls.geometry=j["guided_rounds"]

    def test_49_verified_rounds_and_best(self):
        self.assertEqual([r["round"] for r in self.cases],list(range(49)))
        self.assertTrue(all(isinstance(r["validation"]["balanced_accuracy"],(float,int))
                            for r in self.cases))
        best=max(self.cases,key=lambda r:r["validation"]["balanced_accuracy"])
        self.assertEqual(best["round"],4)
        self.assertAlmostEqual(100*best["validation"]["balanced_accuracy"],83.83,delta=.005)

    def test_all_49_table_numbers_from_digital_source(self):
        src=(ROOT/"tables/validation_ba_49_unified.tex").read_text("utf-8")
        pairs=re.findall(r"\$(\d+)\$\s*&\s*\$([0-9]+\.[0-9]{2})\$",src)
        self.assertEqual(len(pairs),49)
        self.assertEqual({int(r) for r,v in pairs},set(range(49)))
        for r,v in pairs:
            self.assertEqual(v,f"{100*self.cases[int(r)]['validation']['balanced_accuracy']:.2f}")
        self.assertNotRegex(src,r"تاریخی|بازپخش|بازیابی")

    def test_plot_contains_exactly_49_validated_values(self):
        src=(ROOT/"figures/validation_ba_49_unified.tex").read_text("utf-8")
        pts=re.findall(r"\\fill\[[^\]]+\]\s+\((\d+),([0-9]+\.[0-9]{4})\)\s+circle",src)
        self.assertEqual(len(pts),49)
        self.assertEqual({int(r) for r,v in pts},set(range(49)))
        for r,v in pts:
            self.assertEqual(v,f"{100*self.cases[int(r)]['validation']['balanced_accuracy']:.4f}")
        self.assertIn(r"\label{fig:ch4-target-center-trajectory-validation}",src)
        self.assertNotRegex(src,r"۲۵|۲۴|تاریخی|بازپخش|دایره توخالی")

    def test_selected_table_preserves_geometry_and_all_selected_BA(self):
        src=(ROOT/"tables/selected_rounds_unified.tex").read_text("utf-8")
        indices=[0,1,2,3,4,5,10,20,30,40,48]
        vals=re.findall(r"^\$(\d+)\$\s*&\s*\$([.0-9]+)\$\s*&\s*\$([.0-9]+)\$\s*&\s*\$([.0-9]+)\$\s*&\s*\$([.0-9]+)\$",src,re.M)
        self.assertEqual(len(vals),len(indices))
        for (r,I,R,G,BA),i in zip(vals,indices):
            self.assertEqual(int(r),i)
            self.assertEqual(I,f"{self.cases[i]['train']['I']:.5f}")
            self.assertEqual(R,f"{self.cases[i]['train']['R_JS']:.5f}")
            self.assertEqual(G,f"{self.geometry[i]['target_realized_mean_JS']:.6f}")
            self.assertEqual(BA,f"{100*self.cases[i]['validation']['balanced_accuracy']:.2f}")

    def test_chapter_imports_unified_and_keeps_archive_intact(self):
        manuscript=CHAPTER.read_text("utf-8")
        start=manuscript.index(r"\section{هدایت گشت به مراکز هدف")
        end=manuscript.index(r"\section{ارزیابی تجربی زیان حاشیه‌ای",start)
        section=manuscript[start:end]
        for asset in ("tables/selected_rounds_unified.tex",
                      "tables/validation_ba_49_unified.tex",
                      "figures/trajectory_geometry_balanced.tex",
                      "figures/validation_ba_49_unified.tex"):
            self.assertEqual(section.count("center-target-observation/"+asset),1)
        for old in ("tables/selected_rounds_balanced.tex","figures/validation_ba_replayed_49.tex"):
            self.assertNotIn("center-target-observation/"+old,section)
            self.assertTrue((ROOT/old).is_file())
        self.assertNotRegex(section,r"تاریخچه|تاریخی|بازپخش|۲۵ مقدار|۲۴ مقدار")
        self.assertIn("پروتکل",section)

if __name__=="__main__":
    unittest.main()
