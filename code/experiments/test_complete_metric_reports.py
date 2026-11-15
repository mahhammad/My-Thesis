"""Fast regression of unified Chapter 4 metric reports (no walk/data generation)."""
import csv
import json
from pathlib import Path
import unittest

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
CH=ROOT/"tex/chapter4.tex"


def read(path):
    return json.loads(path.read_text(encoding="utf8"))


class Chapter4CommonMetricArtifactTests(unittest.TestCase):
    def test_section_numbering_and_chapter3_reference_registry(self):
        c3=(ROOT/"tex/chapter3.tex").read_text(encoding="utf8")
        c4=CH.read_text(encoding="utf8")
        self.assertIn("\\label{subsec:ch4-common-evaluation-contract}",c4)
        start=c4.index("\\label{subsec:ch4-common-evaluation-contract}")
        end=c4.index("\\section{اعتبارسنجی همگرایی توزیع انتقال یک‌گامی")
        self.assertLess(start,end)
        for ref in ("subsec:ch3-within-class-compactness",
                    "subsec:ch3-center-separation-criteria",
                    "subsec:ch3-compactness-separation",
                    "subsec:ch3-sigmoid-balanced-error"):
            self.assertIn("\\label{"+ref+"}",c3)
            self.assertIn("\\ref{"+ref+"}",c4)

    def test_all_exact_uniform_and_mixture_runs_have_three_R_and_geometry(self):
        for folder,count in (
            ("uniform_policy_hyperparameter_sensitivity",9),
            ("guided_informative_uniform_mixture",5),
        ):
            base=HERE/folder
            data=read(base/"results/metrics_full.json")
            cover=read(base/"results/coverage.json")
            self.assertEqual(len(data["runs"]),count)
            self.assertEqual(cover["configurations"],count)
            self.assertEqual(cover["all_required_center_representation_metrics"],"measured")
            self.assertFalse(data["has_test_data"])
            for r in data["runs"]:
                center=r["center_geometry"]
                self.assertEqual(len(r["centers"]),3)
                self.assertEqual(len(center["pairwise_sqrt_JS"]),3)
                self.assertEqual(len(center["nearest_center_sqrt_JS_per_class"]),3)
                self.assertGreaterEqual(center["mean_pairwise_sqrt_JS"],center["min_pairwise_sqrt_JS"])
                self.assertLessEqual(center["mean_pairwise_sqrt_JS"],1.0)
                for key in ("T","BT","TNL","BTNL","I"):
                    self.assertIsNotNone(center[key])
                for split in ("train","validation"):
                    m=r[split]
                    self.assertEqual(m["R"],m["R_KL"])
                    self.assertEqual(len(m["per_class"]),3)
                    for key in ("R","R_KL","R_JS","R_sqrt_JS",
                                "balanced_accuracy","ordinary_accuracy"):
                        self.assertIsNotNone(m[key],f"{folder}/{r['run_id']}/{split}/{key}")
                    self.assertAlmostEqual(
                        sum(v["R_JS"] for v in m["per_class"])/3,
                        m["R_JS"],places=12)
                    self.assertAlmostEqual(
                        sum(v["R_KL"] for v in m["per_class"])/3,
                        m["R_KL"],places=12)
                    self.assertAlmostEqual(
                        sum(v["R_sqrt_JS"] for v in m["per_class"])/3,
                        m["R_sqrt_JS"],places=12)
            with (base/"results/metrics_summary.csv").open(encoding="utf8",newline="") as f:
                rows=list(csv.DictReader(f))
            self.assertEqual(len(rows),count)
            for rec,r in zip(rows,data["runs"]):
                self.assertEqual(rec["run_id"],r["run_id"])
                self.assertAlmostEqual(float(rec["train_R"]),r["train"]["R"],places=12)
                self.assertAlmostEqual(float(rec["train_R_sqrt_JS"]),r["train"]["R_sqrt_JS"],places=12)

    def test_guided_compact_table_displays_three_R_plus_center_distance(self):
        base=HERE/"guided_informative_uniform_mixture"
        data=read(base/"results/metrics_full.json")
        table=(base/"tables/complete_compactness.tex").read_text(encoding="utf8")
        for rec in data["runs"]:
            for val in (rec["train"]["R"],rec["train"]["R_JS"],
                        rec["train"]["R_sqrt_JS"],rec["train"]["I"],
                        rec["center_geometry"]["min_pairwise_sqrt_JS"]):
                self.assertIn(f"{val:.5f}",table)
        chapter=CH.read_text(encoding="utf8")
        self.assertEqual(chapter.count(
            "\\input{./code/experiments/guided_informative_uniform_mixture/tables/complete_compactness.tex}"),1)

    def test_q_only_n_a_and_historical_49_25_missingness_are_explicit(self):
        q=read(HERE/"q-validation/results/coverage.json")
        self.assertTrue(any("R_KL" in r["ids"] for r in q["not_applicable"]))
        t=read(HERE/"target_center_guidance_trajectory/results/metrics_full.json")
        cov=read(HERE/"target_center_guidance_trajectory/results/coverage.json")
        self.assertEqual(len(t["guided_rounds"]),49)
        self.assertEqual(len(t["selected_cases"]),3)
        self.assertEqual(sum(x["validation"]["balanced_accuracy"] is not None for x in t["guided_rounds"]),25)
        self.assertEqual(sum(x["train"]["R_KL"] is None for x in t["guided_rounds"]),47)
        self.assertEqual(sum(x["train"]["R_sqrt_JS"] is None for x in t["guided_rounds"]),47)
        self.assertEqual(cov["measurement_counts"]["missing_validation_BA"],24)
        self.assertFalse(t["historical_test_accessed"])
        for x in t["guided_rounds"]:
            self.assertEqual(len(x["center_geometry"]["pairwise_sqrt_JS"]),3)
            self.assertIsNotNone(x["train"]["R_JS"])
        for case in t["selected_cases"]:
            for sp in ("train","validation"):
                for key in ("R_KL","R_JS","R_sqrt_JS"):
                    self.assertIsNotNone(case[sp][key])


if __name__=="__main__":
    unittest.main()
