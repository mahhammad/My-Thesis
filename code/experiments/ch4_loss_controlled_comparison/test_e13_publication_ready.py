"""Presentation-only review checks for E-CH4-13 (NO FIT / NO TEST)."""
from pathlib import Path
import re
import unittest
import json
import sys
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sys.path.insert(0,str(HERE))
import phase2_figures_1_8_v3 as v3
import phase2_figure11_y70_100 as trace

class E13Presentation(unittest.TestCase):
    def test_eight_overlays_are_reproducible(self):
        for rel,blob in v3.produce().items():
            self.assertEqual((HERE/rel).read_bytes(),blob,rel)

    def test_corrected_train_ba_axis_is_reproducible(self):
        blob=trace.make()
        self.assertEqual(trace.OUT.read_bytes(),blob)
        svg=blob.decode()
        self.assertIn(">70</text>",svg)
        self.assertIn(">100</text>",svg)
        self.assertNotIn(">102.3</text>",svg)

    def test_all_tex_references_resolve_to_vector_pdfs(self):
        chapter=(ROOT/"tex/chapter4.tex").read_text()
        fragment=chapter[chapter.index(r"\subsection{مقایسه کنترل‌شده زیان حاشیه‌ای"):chapter.index(r"\subsection{NLL مبتنی بر KL و JS")]
        inputs=re.findall(r"\\input\{([^}]+)\}",fragment)
        self.assertEqual(len(inputs),5) # 2 tables and 3 graphics wrappers
        paths=[]
        for name in inputs:
            p=ROOT/name
            self.assertTrue(p.is_file(),name)
            paths.append(p)
        figures=[]
        for p in paths:
            figures+=re.findall(r"\\includegraphics(?:\[[^\]]+\])?\{([^}]+)\}",p.read_text())
        self.assertEqual(len(figures),11)
        self.assertEqual(len(set(figures)),11)
        for name in figures:
            p=ROOT/name
            self.assertTrue(p.exists(),name)
            self.assertTrue(p.read_bytes().startswith(b"%PDF"),name)

    def test_unique_figure_table_labels(self):
        chapter=(ROOT/"tex/chapter4.tex").read_text()
        labels=re.findall(r"\\label\{([^}]+)\}",chapter)
        # Chapter headings are pre-existing; this checks new labels and their referenced wrappers
        wrappers=(HERE/"figures/phase2_comparison_figures.tex").read_text()+(
            HERE/"figures/phase2_density_figures.tex").read_text()+(
            HERE/"figures/phase2_dynamics_figures.tex").read_text()+(
            HERE/"tables/phase2_validation_pairs.tex").read_text()+(
            HERE/"tables/phase2_validation_summary.tex").read_text()
        inserted=re.findall(r"\\label\{([^}]+)\}",wrappers)
        self.assertEqual(len(inserted),13) # 11 figures, 2 tables
        self.assertEqual(len(set(inserted)),13)
        self.assertTrue(all(x.startswith(("fig:ch4-e13-","tab:ch4-e13-")) for x in inserted))
        self.assertEqual(len(labels),len(set(labels)))
        for l in inserted:self.assertNotIn(l,labels) # wrappers only

    def test_table_values_equal_frozen_raw_and_matched_bayes(self):
        from phase2_derived_reproduce_v2 import inputs,pair_data
        from validation_bayes_summary import main as bayes_summary
        _,D=inputs()
        rows=pair_data(D)
        table=(HERE/"tables/phase2_validation_pairs.tex").read_text()
        for row in rows:
            expected=(f"{row['seed']}",f"{100*row['validation_BA_sigmoid']:.4f}",
                      f"{100*row['validation_BA_softplus']:.4f}",
                      f"{row['delta_BA_pp']:+.4f}" if abs(row['delta_BA_pp'])>1e-10 else "0.0000")
            for value in expected:self.assertIn(value,table,(row['seed'],row['family'],value))
        summary=(HERE/"tables/phase2_validation_summary.tex").read_text()
        for row in bayes_summary()["rows"]:
            for key in ("mean_percent","sample_variance_pp_squared",
                        "sample_standard_deviation_pp"):
                self.assertIn(f"{row[key]:.4f}",summary,(row["family"],row["objective"],key))
        # Raw comparisons and Bayes reference use the same 3 dataset seeds.

    def test_no_test_and_no_raw_rewrite(self):
        m=json.loads((HERE/"results/phase2_300_25/manifest.json").read_text())
        self.assertIs(m["test_accessed"],False)
        self.assertEqual(m["fit_count"],12)
        self.assertEqual(m["updates_per_fit"],25)

if __name__=="__main__":unittest.main()
