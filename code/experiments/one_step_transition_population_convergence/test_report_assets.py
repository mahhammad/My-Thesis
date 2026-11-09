"""Revalidate archived 48 observations, derived tables and vector plots.

No calls to analyze(): no Monte Carlo reruns, no training, no Test access.
"""
import csv
import json
from math import isclose, sqrt
from pathlib import Path
import re
import unittest

ROOT = Path(__file__).resolve().parent


class ReportAssetsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = json.loads((ROOT / "results/metrics.json").read_text(encoding="utf8"))
        with (ROOT / "results/replicates.csv").open(encoding="utf8") as fh:
            cls.raw = list(csv.DictReader(fh))
        cls.comparison = cls.data["comparison"]

    def test_protocol_and_exact_replications(self):
        d = self.data
        self.assertEqual(d["reference_sizes"], [300, 1000, 3000, 10000])
        self.assertEqual(d["replicates_per_size"], 12)
        self.assertEqual(d["n_fixed_validation_queries"], 90)
        self.assertEqual(d["fixed_query_seed"], 20261102)
        self.assertEqual(d["exact_default_300_reference"]["n_observed_feature_queries"], 416)
        self.assertEqual(len(self.raw), 48)
        self.assertEqual(len(d["replicate_data"]), 48)
        self.assertEqual({int(r["trial"]) for r in self.raw}, set(range(12)))
        for r, source in zip(self.raw, d["replicate_data"]):
            self.assertEqual(int(r["n_reference"]), source["n_reference"])
            self.assertEqual(int(r["trial"]), source["trial"])
            for k in ("mean_TV", "mean_sqrt_JSD", "p95_TV"):
                self.assertAlmostEqual(float(r[k]), source[k], delta=6e-13)

    def test_group_aggregation_sd_is_sample_sd_of_replicate_means(self):
        for group in self.comparison:
            vals = [r for r in self.data["replicate_data"]
                    if r["n_reference"] == group["n_reference"]]
            self.assertEqual(len(vals), 12)
            means = [r["mean_TV"] for r in vals]
            mean = sum(means) / 12
            sd = sqrt(sum((v - mean)**2 for v in means) / 11)
            self.assertAlmostEqual(mean, group["avg_mean_TV"], delta=1e-12)
            self.assertAlmostEqual(sd, group["sd_mean_TV"], delta=1e-12)
            self.assertAlmostEqual(sum(r["mean_sqrt_JSD"] for r in vals) / 12,
                                   group["avg_mean_sqrt_JSD"], delta=1e-12)
            self.assertAlmostEqual(sum(r["p95_TV"] for r in vals) / 12,
                                   group["avg_p95_TV"], delta=1e-12)
            self.assertGreaterEqual(group["avg_mean_TV"], 0)
            self.assertLessEqual(group["avg_mean_TV"], 1)

    def test_derived_csvs_match_json(self):
        with (ROOT / "results/summary.csv").open(encoding="utf8") as fh:
            summ = list(csv.DictReader(fh))
        with (ROOT / "results/per_feature.csv").open(encoding="utf8") as fh:
            features = list(csv.DictReader(fh))
        self.assertEqual(len(summ), len(self.comparison))
        self.assertEqual(len(features), len(self.comparison))
        for r, s, f in zip(self.comparison, summ, features):
            self.assertEqual(int(s["n_reference"]), r["n_reference"])
            self.assertEqual(int(f["n_reference"]), r["n_reference"])
            self.assertEqual(int(s["replicates"]), r["replicates"])
            for col, key in [("mean_TV", "avg_mean_TV"),
                             ("sd_replicate_mean_TV", "sd_mean_TV"),
                             ("mean_sqrt_JSD", "avg_mean_sqrt_JSD"),
                             ("mean_p95_TV", "avg_p95_TV")]:
                self.assertAlmostEqual(float(s[col]), r[key], delta=1e-15)
            for a in range(5):
                self.assertAlmostEqual(float(f[f"feature_{a}"]),
                                       r["avg_mean_TV_per_feature"][a], delta=1e-15)

    def test_latex_table_rows_reflect_source(self):
        a = (ROOT / "tables/main_results.tex").read_text(encoding="utf8")
        b = (ROOT / "tables/per_feature_tv.tex").read_text(encoding="utf8")
        for r in self.comparison:
            x = ("$%d$ & $%.5f$ & $%.5f$ & $%.5f$ & $%.5f$" %
                 (r["n_reference"], r["avg_mean_TV"], r["sd_mean_TV"],
                  r["avg_mean_sqrt_JSD"], r["avg_p95_TV"])) + "\\\\"
            self.assertIn(x, a)
            y = "$%d$ & " % r["n_reference"]
            y += " & ".join("$%.5f$" % v for v in r["avg_mean_TV_per_feature"])
            self.assertIn(y + "\\\\", b)
        self.assertIn(r"\label{tab:ch4-q-population-convergence}", a)
        self.assertIn(r"\label{tab:ch4-q-feature-tv}", b)

    def test_github_svg_previews_match_48_observation_source(self):
        from xml.etree import ElementTree as ET
        a = (ROOT / "figures/error_convergence.svg").read_text(encoding="utf8")
        b = (ROOT / "figures/per_feature_tv.svg").read_text(encoding="utf8")
        for src in (a, b):
            root = ET.fromstring(src)
            self.assertEqual(root.tag, "{http://www.w3.org/2000/svg}svg")
            self.assertNotIn("<script", src.lower())
        xs = [180, 400, 620, 840]
        y = lambda v: f"{450 - 350*v/.07:.2f}"
        tv = " ".join(f"{x},{y(r['avg_mean_TV'])}" for x, r
                      in zip(xs, self.comparison))
        js = " ".join(f"{x},{y(r['avg_mean_sqrt_JSD'])}" for x, r
                      in zip(xs, self.comparison))
        self.assertIn(f'id="series-tv" points="{tv}"', a)
        self.assertIn(f'id="series-sqrt-jsd" points="{js}"', a)
        for i in range(5):
            pts = " ".join(f"{x},{y(r['avg_mean_TV_per_feature'][i])}"
                           for x, r in zip(xs, self.comparison))
            self.assertIn(f'id="feature-{i}" points="{pts}"', b)

    def test_thesis_section_43_uses_canonical_title_and_sources(self):
        chapter = (ROOT.parents[2] / "tex/chapter4.tex").read_text(encoding="utf8")
        self.assertIn(
            r"\section{اعتبارسنجی همگرایی توزیع انتقال یک‌گامی به مرجع جامعه}",
            chapter,
        )
        self.assertEqual(chapter.count(r"\label{sec:ch4-q-validation}"), 1)
        for relative in (
            "tables/main_results.tex",
            "tables/per_feature_tv.tex",
            "figures/error_convergence.tex",
            "figures/per_feature_tv.tex",
        ):
            self.assertTrue((ROOT / relative).is_file())
            expected = (r"\input{./code/experiments/"
                        r"one_step_transition_population_convergence/"
                        + relative + "}")
            self.assertEqual(chapter.count(expected), 1)
        labels = re.findall(r"\\label\{([^}]+)\}", chapter)
        for label in (
            "tab:ch4-q-population-convergence", "tab:ch4-q-feature-tv",
            "fig:ch4-q-error-convergence", "fig:ch4-q-error-by-feature",
        ):
            self.assertIn(r"\ref{" + label + "}", chapter)
            self.assertNotIn(r"\label{" + label + "}", chapter)
            self.assertEqual(
                sum((ROOT / rel).read_text(encoding="utf8").count(
                    r"\label{" + label + "}"
                ) for rel in (
                    "tables/main_results.tex", "tables/per_feature_tv.tex",
                    "figures/error_convergence.tex", "figures/per_feature_tv.tex"
                )),
                1,
            )

    def test_tikz_plot_coordinates_and_sd_whiskers_reflect_json(self):
        a = (ROOT / "figures/error_convergence.tex").read_text(encoding="utf8")
        b = (ROOT / "figures/per_feature_tv.tex").read_text(encoding="utf8")
        xs = (1.45, 3.45, 5.45, 7.45)
        y = lambda v: 0.75 + 60 * v
        tv = " ".join("(%.3f,%.3f)" % (x, y(r["avg_mean_TV"]))
                      for x, r in zip(xs, self.comparison))
        js = " ".join("(%.3f,%.3f)" % (x, y(r["avg_mean_sqrt_JSD"]))
                      for x, r in zip(xs, self.comparison))
        self.assertIn("plot coordinates {" + tv + "}", a)
        self.assertIn("plot coordinates {" + js + "}", a)
        for x, r in zip(xs, self.comparison):
            self.assertIn("(%.3f,%.3f)--(%.3f,%.3f)" % (
                x, y(r["avg_mean_TV"]-r["sd_mean_TV"]), x,
                y(r["avg_mean_TV"]+r["sd_mean_TV"])), a)
        for feat in range(5):
            p = " ".join("(%.3f,%.3f)" % (x, y(r["avg_mean_TV_per_feature"][feat]))
                         for x, r in zip(xs, self.comparison))
            self.assertIn("plot coordinates {" + p + "}", b)
        for path in ("figures/error_convergence.tex", "figures/per_feature_tv.tex"):
            text = (ROOT / path).read_text(encoding="utf8")
            self.assertIn(r"\begin{tikzpicture}", text)
            self.assertIn(r"\end{tikzpicture}", text)
            self.assertNotIn(r"\usepackage", text)


if __name__ == "__main__":
    unittest.main()
