"""Chapter-4 evidence-contract regression for fully saved experiment reports.

Stdlib only, no original walk replay, no model fitting, no Test access.
Complete means every APPLICABLE metric is present or carries an explicit,
scientifically justified not_measured/undefined status.
"""
from __future__ import annotations

import json
from pathlib import Path
import unittest

HERE = Path(__file__).resolve().parent
CONTRACT = json.loads((HERE/"COMMON_METRICS_CONTRACT.json").read_text("utf8"))


def load(slug, name="metrics_full.json"):
    return json.loads((HERE/slug/"results"/name).read_text("utf8"))


def ensure_metrics(case, caseid):
    train, validation, geometry = case["train"],case["validation"],case["center_geometry"]
    for split, obj in [("train",train),("validation",validation)]:
        for name in ("R_KL","R_JS","R_sqrt_JS","R","balanced_accuracy",
                     "ordinary_accuracy","per_class","margin_quantiles",
                     "balanced_sigmoid_margin_loss","balanced_JS_similarity_NLL",
                     "NLL_KL_simple"):
            if name not in obj:
                raise AssertionError(f"{caseid} {split} missing {name}")
        if obj["R_KL"] is not None:
            if abs(obj["R"]-obj["R_KL"])>1e-12:
                raise AssertionError(f"{caseid}: R is not R_KL")
        if len(obj["per_class"])!=3:
            raise AssertionError(f"{caseid}: only {len(obj['per_class'])} per-class entries")
        for row in obj["per_class"]:
            for name in ("R_JS","R_sqrt_JS","R_KL"):
                if name not in row:
                    raise AssertionError(f"{caseid}: missing per-class {name}")
    for key in ("T","BT","TNL","BTNL","I","pairwise_JS",
                "pairwise_sqrt_JS","nearest_center_sqrt_JS_per_class",
                "min_pairwise_sqrt_JS","mean_pairwise_sqrt_JS"):
        if key not in geometry:
            raise AssertionError(f"{caseid} geometry missing {key}")
    if len(geometry["pairwise_JS"])!=3:
        raise AssertionError(f"{caseid}: invalid pairwise matrix")


class CompleteExperimentEvidence(unittest.TestCase):
    def test_machine_readable_contract_is_versioned(self):
        self.assertEqual(CONTRACT["schema"],"chapter4-common-evaluation-contract-v2")
        self.assertIn("center_geometry",[
            x["group"] for x in CONTRACT["metric_groups"]])

    def test_Q_only_diagnostic_has_full_measured_48_and_declared_na(self):
        slug="q-validation"
        m,c=load(slug),load(slug,"coverage.json")
        self.assertEqual(m["experiment_id"],"E-CH4-03")
        self.assertTrue(m["representative_Q_only"])
        self.assertEqual(len(m["all_replicate_observations"]),48)
        self.assertEqual(len(m["reference_sizes"]),4)
        self.assertEqual(m["independent_replicates_per_size"],12)
        self.assertTrue(c["not_applicable"])
        self.assertTrue(any(
            "R_KL" in str(item) and "BA" in str(item)
            for item in c["not_applicable"]))
        self.assertFalse(c["historical_test_accessed"])

    def test_nine_uniform_runs_have_all_applicable_metrics(self):
        slug="uniform-hyperparameter-sensitivity"
        m,c=load(slug),load(slug,"coverage.json")
        self.assertEqual(len(m["runs"]),9)
        self.assertEqual(c["configurations"],9)
        for x in m["runs"]:
            ensure_metrics(x,x["run_id"])
        self.assertEqual(c["not_measured"],[])
        self.assertTrue((HERE/slug/"tables/complete_metrics.tex").is_file())

    def test_five_guided_runs_have_all_applicable_metrics(self):
        slug="fixed-policy-evaluation"
        m,c=load(slug),load(slug,"coverage.json")
        self.assertEqual(len(m["runs"]),5)
        self.assertEqual(c["configurations"],5)
        self.assertEqual([x["setting"]["alpha"] for x in m["runs"]],
                         [0,.25,.5,.75,1])
        for x in m["runs"]:
            ensure_metrics(x,x["run_id"])
        self.assertEqual(c["not_measured"],[])
        self.assertTrue((HERE/slug/"tables/complete_metrics.tex").is_file())

    def test_center_trajectory_has_49_geometries_and_3_complete_checkpoint_reports(self):
        slug="center-target-observation"
        m,c=load(slug),load(slug,"coverage.json")
        self.assertEqual(len(m["guided_rounds"]),49)
        self.assertEqual(len(m["selected_cases"]),3)
        self.assertEqual(sum(
            r["validation"]["balanced_accuracy"] is not None
            for r in m["guided_rounds"]),25)
        for case in m["selected_cases"]:
            st=case["standard_v2"]
            ensure_metrics({"train":st["train"],
                            "validation":st["validation"],
                            "center_geometry":case["center_geometry"]},case["case_id"])
        for row in m["guided_rounds"]:
            for key in ("T","BT","TNL","BTNL","I",
                        "pairwise_JS","pairwise_sqrt_JS"):
                self.assertIn(key,row["center_geometry"])
            self.assertIn("R_JS",row["train"])
        self.assertEqual(c["measurement_counts"]["guided_rounds"],49)
        self.assertEqual(c["measurement_counts"]["measured_validation_BA"],25)
        self.assertEqual(c["measurement_counts"]["fully_diagnosed_cases"],3)
        self.assertEqual(c["measurement_counts"]["measured_guided_train_R_KL"],2)
        self.assertTrue(any(x.get("count")==47 for x in c["not_measured"]))
        self.assertTrue((HERE/slug/"tables/complete_metrics.tex").is_file())
        self.assertTrue((HERE/slug/"results/metrics_summary.csv").is_file())

    def test_exactly_eleven_official_studies_plus_shared_benchmark(self):
        study_dirs = ["q-validation","uniform-hyperparameter-sensitivity","fixed-policy-evaluation","center-target-observation","sigmoid_margin_observational_evaluation","walk_depth_effect","exact_matrix_walk_validation","inverse_attainable_standard","bayes_center_reachability_standard","standard_policy_comparison","ch4_loss_controlled_comparison"]
        actual = {p.name for p in HERE.iterdir() if p.is_dir() and not p.name.startswith((".", "__"))}
        self.assertEqual(actual, set(study_dirs) | {"benchmark_v1"})
        for slug in study_dirs:
            self.assertTrue((HERE / slug / "README_FA.md").is_file(), slug)
        self.assertTrue((HERE.parents[1] / "research_journal/ch4-14-synthetic-data-structure-sensitivity-plan.md").is_file())

    def test_every_experiment_has_single_technical_readme(self):
        for folder in HERE.iterdir():
            if not folder.is_dir() or folder.name.startswith((".", "__")):
                continue
            markdown_files=sorted(p.relative_to(folder).as_posix()
                                  for p in folder.rglob("*.md"))
            self.assertEqual(markdown_files, ["README_FA.md"], folder.name)
            self.assertIn("# ", (folder/"README_FA.md").read_text("utf8"))

    def test_all_four_experiment_reports_link_to_full_metric_records(self):
        for slug in (
            "q-validation",
            "uniform-hyperparameter-sensitivity",
            "fixed-policy-evaluation",
            "center-target-observation",
        ):
            report=(HERE/slug/"README_FA.md").read_text("utf8")
            self.assertEqual([p.name for p in (HERE/slug).rglob("*.md")],["README_FA.md"])
            self.assertIn("results/metrics_full.json",report,slug)
            self.assertIn("results/coverage.json",report,slug)
            self.assertTrue((HERE/slug/"results/metrics_full.json").is_file())


if __name__=="__main__":
    unittest.main()
