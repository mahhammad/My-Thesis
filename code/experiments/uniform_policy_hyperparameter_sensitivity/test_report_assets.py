"""Fast, data-only verification of the Chapter 4 Uniform sensitivity report.

No training, torch, dataset generation, snapshot replay, or historical Test.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
import unittest
from xml.etree import ElementTree

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
SOURCE = (REPO / "code/experiments/benchmark_v1/results/pipeline_v1"
          / "fixed_baselines_summary_20261109.csv")
TEX = REPO / "tex/chapter4.tex"


def parse_csv(path):
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def as_num(d, k):
    return float(d[k])


def blob_sha(path):
    content = path.read_bytes()
    return hashlib.sha1(
        b"blob " + str(len(content)).encode("ascii") + b"\0" + content
    ).hexdigest()


def coefficients(beta, lam, H):
    alpha = [0.0] * H
    for L in range(1, H + 1):
        p = beta * (1 - beta) ** (L-1) if L < H else (1-beta) ** (H-1)
        den = sum(lam**j for j in range(L))
        for t in range(1, L + 1):
            alpha[t-1] += p * lam**(t-1) / den
    return alpha


class UniformExperimentReportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original = parse_csv(SOURCE)
        cls.runs = parse_csv(HERE / "results/uniform_runs.csv")
        cls.plot = parse_csv(HERE / "results/plot_groups.csv")
        cls.meta = json.loads((HERE / "results/provenance.json").read_text("utf8"))
        cls.base = next(r for r in cls.runs
                        if float(r["beta"]) == .4 and float(r["lam"]) == .8
                        and int(r["H"]) == 12)

    def test_raw_data_is_exact_unchanged_subset_of_verified_13(self):
        source = SOURCE.read_text(encoding="utf8").splitlines(keepends=True)
        extracted = "".join(source[:1] + [
            l for l in source[1:] if l.split(",")[2] == "uniform"
        ])
        self.assertEqual(extracted, (HERE / "results/uniform_runs.csv").read_text("utf8"))
        self.assertEqual(len(self.original), 13)
        self.assertEqual(len(self.runs), 9)
        self.assertEqual(len(set(r["id"] for r in self.runs)), 9)
        self.assertEqual({r["kind"] for r in self.runs}, {"uniform"})
        self.assertEqual({float(r["alpha"]) for r in self.runs}, {0.0})

    def test_git_blob_provenance_matches_original_and_config(self):
        self.assertEqual(self.meta["id"], "E-CH4-04")
        self.assertEqual(blob_sha(SOURCE), self.meta["source"]["git_blob_sha1"])
        self.assertEqual(
            blob_sha(REPO / self.meta["generator"]["config"]),
            self.meta["generator"]["config_git_blob_sha1"],
        )
        self.assertEqual(self.meta["source"]["uniform_records"], 9)
        self.assertEqual(self.meta["generator"]["data_seed"], 20261102)
        self.assertFalse(self.meta["generator"]["historical_test_accessed"])
        self.assertFalse(self.meta["policy"]["learned"])
        self.assertEqual(set(self.meta["run_ids"]), {r["id"] for r in self.runs})

    def test_one_factor_at_a_time_and_default_not_retrained(self):
        self.assertEqual(sum(r["id"] == self.base["id"] for r in self.runs), 1)
        self.assertEqual(sum(float(r["lam"]) == .8 and int(r["H"]) == 12
                             for r in self.runs), 3)
        self.assertEqual(sum(float(r["beta"]) == .4 and int(r["H"]) == 12
                             for r in self.runs), 3)
        self.assertEqual(sum(float(r["beta"]) == .4 and float(r["lam"]) == .8
                             for r in self.runs), 5)
        for r in self.runs:
            differences = (float(r["beta"]) != .4) + (
                float(r["lam"]) != .8) + (int(r["H"]) != 12)
            self.assertLessEqual(differences, 1)

    def test_length_and_depth_recomputed_from_geometric_weights(self):
        for r in self.runs:
            beta, lam, H = float(r["beta"]), float(r["lam"]), int(r["H"])
            expected = (1-(1-beta)**H)/beta
            alpha = coefficients(beta, lam, H)
            self.assertAlmostEqual(sum(alpha), 1.0, delta=3e-14)
            effective = sum((j+1)*w for j, w in enumerate(alpha))
            self.assertAlmostEqual(expected, as_num(r, "expected_length"), delta=1e-12)
            self.assertAlmostEqual(effective, as_num(r, "effective_depth"), delta=1e-12)

    def test_plot_group_is_exact_reuse_not_more_independent_runs(self):
        self.assertEqual(len(self.plot), 11)  # 3+3+5; default drawn three times
        self.assertEqual({r["group"] for r in self.plot}, {"beta", "lambda", "H"})
        byid = {r["id"]: r for r in self.runs}
        self.assertEqual(sum(r["run_id"] == self.base["id"] for r in self.plot), 3)
        for p in self.plot:
            r = byid[p["run_id"]]
            for source_key, plot_key in (
                ("beta", "beta"), ("lam", "lambda"), ("H", "H"),
                ("effective_depth", "effective_depth"), ("I", "I"),
                ("R_JS", "R_JS"), ("R_KL", "R_KL"),
                ("validation_median_margin", "median_margin_val"),
                ("train_BA", "BA_train"), ("validation_BA", "BA_val"),
            ):
                self.assertAlmostEqual(float(p[plot_key]), float(r[source_key]),
                                       delta=1e-15)

    def test_accuracy_and_geometry_ranges_are_not_guessed(self):
        original = as_num(self.base, "validation_BA")
        self.assertAlmostEqual(original, .9676870748299321, delta=1e-15)
        self.assertTrue(all(as_num(r, "validation_BA") == original for r in self.runs))
        self.assertTrue(all(as_num(r, "validation_accuracy") ==
                            as_num(self.base, "validation_accuracy") for r in self.runs))
        self.assertAlmostEqual(min(as_num(r, "train_BA") for r in self.runs),
                               .8341941714957589, delta=1e-15)
        self.assertAlmostEqual(max(as_num(r, "train_BA") for r in self.runs),
                               .836461745191904, delta=1e-15)
        self.assertLess(min(as_num(r, "I") for r in self.runs),
                        max(as_num(r, "I") for r in self.runs))
        self.assertLess(min(as_num(r, "R_JS") for r in self.runs),
                        max(as_num(r, "R_JS") for r in self.runs))

    def test_tables_contain_all_nine_numeric_rows_and_unique_labels(self):
        a = (HERE / "tables/main_results.tex").read_text("utf8")
        b = (HERE / "tables/diagnostics.tex").read_text("utf8")
        self.assertEqual(a.count(r"\begin{table}"), 1)
        self.assertEqual(a.count(r"\end{table}"), 1)
        for r in self.runs:
            main = ("$%.4f$ & $%.4f$ & $%.5f$ & $%.5f$ & $%.2f$" %
                    (as_num(r, "expected_length"), as_num(r, "effective_depth"),
                     as_num(r, "I"), as_num(r, "R_JS"),
                     as_num(r, "validation_BA")*100))
            self.assertIn(main, a)
            diag = ("$%.5f$ & $%.5f$ & $%.5f$ & $%.2f$ & $%.2f$ & $%.5f$" %
                    (as_num(r, "R_KL"), as_num(r, "R_sqrt_JS"),
                     as_num(r, "validation_median_margin"),
                     as_num(r, "train_BA")*100,
                     as_num(r, "validation_accuracy")*100,
                     as_num(r, "min_pairwise_sqrt_JS")))
            self.assertIn(diag, b)
        self.assertEqual(a.count(r"\label{tab:ch4-uniform-hyperparams-exact}"), 1)
        self.assertEqual(b.count(r"\label{tab:ch4-uniform-sensitivity-diagnostics}"), 1)

    def test_svg_charts_are_safe_well_formed_and_match_all_metric_points(self):
        for n in ("depth_sensitivity", "geometry_vs_depth", "balanced_accuracy"):
            text = (HERE / "figures" / (n+".svg")).read_text("utf8")
            svg = ElementTree.fromstring(text)
            self.assertEqual(svg.tag, "{http://www.w3.org/2000/svg}svg")
            self.assertNotIn("<script", text.lower())
            self.assertNotIn("http://", text.lower().replace("http://www.w3.org/2000/svg", ""))
        # Curve drawing coordinates are fully sourced from the nine CSV rows.
        g = (HERE / "figures/geometry_vs_depth.svg").read_text("utf8")
        for r in self.runs:
            x = float(r["effective_depth"])
            for i, (col, lim) in enumerate((("I", .036), ("R_JS", .0043))):
                px = ([110,640][i]+(x-.95)*260)
                py = 330-float(r[col])/lim*220
                self.assertIn(f'cx="{px:.2f}" cy="{py:.2f}"', g)
        ba = (HERE / "figures/balanced_accuracy.svg").read_text("utf8")
        ordered = self.ordered_rows()
        for i, r in enumerate(ordered):
            for col in ("train_BA", "validation_BA"):
                self.assertIn(
                    f'cx="{130+108*i}" cy="{350-255*float(r[col]):.2f}"', ba
                )
        depth = (HERE / "figures/depth_sensitivity.svg").read_text("utf8")
        self.assertIn("not a separate run", depth)

    @classmethod
    def ordered_rows(cls):
        d = cls.runs
        return (
            sorted((r for r in d if float(r["lam"]) == .8 and int(r["H"]) == 12),
                   key=lambda r: float(r["beta"])) +
            sorted((r for r in d if float(r["beta"]) == .4 and
                    int(r["H"]) == 12 and float(r["lam"]) != .8),
                   key=lambda r: float(r["lam"])) +
            sorted((r for r in d if float(r["beta"]) == .4 and
                    float(r["lam"]) == .8 and int(r["H"]) != 12),
                   key=lambda r: int(r["H"]))
        )

    def test_tikz_figures_contain_the_original_all_nine_data_points(self):
        depth = (HERE / "figures/depth_sensitivity.tex").read_text("utf8")
        geom = (HERE / "figures/geometry_vs_depth.tex").read_text("utf8")
        acc = (HERE / "figures/balanced_accuracy.tex").read_text("utf8")
        labels = (
            "fig:ch4-uniform-depth-sensitivity",
            "fig:ch4-uniform-geometry-vs-depth",
            "fig:ch4-uniform-balanced-accuracy",
        )
        for src, lab in zip((depth, geom, acc), labels):
            self.assertEqual(src.count(r"\begin{tikzpicture}"), 1)
            self.assertEqual(src.count(r"\end{tikzpicture}"), 1)
            self.assertEqual(src.count(r"\label{" + lab + "}"), 1)
        for r in self.runs:
            self.assertIn(f"{.8+1.28*float(r['effective_depth']):.3f}", depth)
            self.assertIn(f"{.9+float(r['I'])/.036*3.1:.3f}", geom)
            self.assertIn(f"{.9+float(r['R_JS'])/.0043*3.1:.3f}", geom)
        for r in self.ordered_rows():
            self.assertIn(f"{.85+3.55*float(r['train_BA']):.3f}", acc)
            self.assertIn(f"{.85+3.55*float(r['validation_BA']):.3f}", acc)

    def test_latex_inputs_are_unique_and_the_guided_table_is_not_in_this_experiment(self):
        chapter = TEX.read_text(encoding="utf8")
        self.assertIn(
            r"\section{ارزیابی سیاست یکنواخت و تحلیل حساسیت ابرپارامترهای گشت}",
            chapter,
        )
        self.assertEqual(chapter.count(r"\label{sec:ch4-uniform-hyperparameter-sensitivity}"), 1)
        self.assertIn(r"\section{ارزیابی سیاست ثابت هدایت‌شده و مقایسه معیارها}", chapter)
        self.assertEqual(chapter.count(r"\label{sec:ch4-guided-policy-comparison}"), 1)
        # Every figure/table is sourced once from this experiment, not duplicated.
        inputs = (
            "tables/main_results.tex", "tables/diagnostics.tex",
            "figures/depth_sensitivity.tex",
            "figures/geometry_vs_depth.tex",
            "figures/balanced_accuracy.tex",
        )
        paths = [
            "./code/experiments/uniform_policy_hyperparameter_sensitivity/" + x
            for x in inputs
        ]
        for x in paths:
            self.assertEqual(chapter.count("\\input{" + x + "}"), 1)
        for label in (
            "tab:ch4-uniform-hyperparams-exact",
            "tab:ch4-uniform-sensitivity-diagnostics",
            "fig:ch4-uniform-depth-sensitivity",
            "fig:ch4-uniform-geometry-vs-depth",
            "fig:ch4-uniform-balanced-accuracy",
        ):
            self.assertNotIn(r"\label{" + label + "}", chapter)
            self.assertIn(r"\ref{" + label + "}", chapter)
        # Existing cross-references into the older fixed-policy section remain valid.
        self.assertEqual(chapter.count(r"\label{sec:ch4-fixed-policy-evaluation}"), 1)
        self.assertNotIn("guided", "\n".join(
            r["kind"] for r in self.runs
        ))


if __name__ == "__main__":
    unittest.main()
