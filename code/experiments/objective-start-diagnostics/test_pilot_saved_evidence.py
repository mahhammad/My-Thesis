"""Read-only post-pilot evidence integrity tests. Never invoke fit or any optimizer."""
import csv
import hashlib
import json
from pathlib import Path
import sys
import unittest
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sys.path.insert(0,str(ROOT/"code/experiments/benchmark_v1"))
sys.path.insert(0,str(ROOT/"code/data_generator"))
from snapshot_pipeline import load_snapshot, array_fingerprint
from report_snapshot import report_snapshot

BASE=HERE/"results/pilot_seed20261102_shared_two_fits"
EXPECTED=("sigmoid_JS_margin_tau005","softplus_JS_tau005")

class SavedPilotIntegrity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not (BASE/"manifest.json").exists():
            raise unittest.SkipTest("No pilot artifact on this branch")
        cls.manifest=json.loads((BASE/"manifest.json").read_text("utf8"))
        cls.coverage=json.loads((BASE/"coverage.json").read_text("utf8"))

    def test_exact_two_fits_and_paired_references(self):
        m=self.manifest
        self.assertEqual(m["n_fits"],2)
        self.assertEqual(m["steps_per_fit"],10)
        self.assertEqual(m["recovery_training_updates"],0)
        self.assertFalse(m["test_accessed"])
        self.assertEqual([r["objective"] for r in m["runs"]],list(EXPECTED))
        self.assertEqual(m["n_train"],50)
        self.assertEqual(m["n_validation"],90)
        self.assertEqual(m["family"],"shared")
        self.assertEqual(m["source_commit"],"66a007928e129ad6dd5acffeaa93fc2485ee9775")
        self.assertEqual(len(set(m["reference_ids"])),50)

    def test_sha256_snapshot_roundtrip_and_report(self):
        for run in self.manifest["runs"]:
            with self.subTest(run=run["objective"]):
                d=BASE/run["objective"]
                for fn in ("model.json","distributions.npz","trace.json","metrics_full.json"):
                    actual=hashlib.sha256((d/fn).read_bytes()).hexdigest()
                    self.assertEqual(actual,run[fn+"_sha256"])
                model,arrays=load_snapshot(d)
                self.assertEqual(model["schema"],"imc-snapshot-v2")
                np.testing.assert_array_equal(arrays["reference_ids"],self.manifest["reference_ids"])
                self.assertEqual(array_fingerprint(arrays["reference_ids"]),
                                 array_fingerprint(np.asarray(self.manifest["reference_ids"],dtype=np.int64)))
                metrics=json.loads((d/"metrics_full.json").read_text("utf8"))
                fresh=report_snapshot(d)
                for split in ("train","validation"):
                    self.assertAlmostEqual(metrics[split]["balanced_accuracy"],fresh[split]["balanced_accuracy"],places=12)
                    self.assertEqual(metrics[split]["confusion_true_rows_pred_cols"],
                                     fresh[split]["confusion_true_rows_pred_cols"])
                self.assertAlmostEqual(run["validation_BA"],metrics["validation"]["balanced_accuracy"],places=12)

    def test_all_traces_ten_updates_and_finite_gradients(self):
        for run in self.manifest["runs"]:
            trace=json.loads((BASE/run["objective"]/"trace.json").read_text("utf8"))
            self.assertEqual([t["update_completed"] for t in trace],list(range(11)))
            self.assertTrue(all(np.isfinite(t["gradient_l2"]) for t in trace))
            self.assertAlmostEqual(trace[0]["initial_gradient_l2"] if "initial_gradient_l2" in trace[0]
                                   else trace[0]["gradient_l2"],
                                   run["initial_gradient_l2"],places=12)
            self.assertAlmostEqual(trace[-1]["train_objective"],run["final_objective"],places=12)

    def test_coverage_has_no_false_zero_and_summary_rebuild(self):
        self.assertEqual(self.coverage["test"]["status"],"not_applicable")
        self.assertEqual(self.coverage["runtime_memory_peak_bytes"]["status"],"not_measured")
        for name in EXPECTED:
            fields=self.coverage["objectives"][name]
            self.assertEqual(len(fields),43)
            self.assertTrue(all(field["status"]=="measured" for field in fields.values()))
        with (BASE/"metrics_summary.csv").open(encoding="utf8") as f:
            rows=list(csv.DictReader(f))
        self.assertEqual(len(rows),2)
        for row,run in zip(rows,self.manifest["runs"]):
            self.assertEqual(row["objective"],run["objective"])
            self.assertAlmostEqual(float(row["validation_BA"]),run["validation_BA"],places=12)
        svg=(BASE/"validation_ba.svg").read_text("utf8")
        for x,value in ((145,self.manifest["runs"][0]["validation_BA"]),
                        (420,self.manifest["runs"][1]["validation_BA"])):
            self.assertIn(f'cx="{x}" cy="{250-200*value:.3f}"',svg)

    def test_publication_table_and_figure_are_raw_derived(self):
        table=(HERE/"tables/pilot_summary.tex").read_text("utf8")
        result_svg=(BASE/"validation_ba.svg").read_bytes()
        figure_svg=(HERE/"figures/validation_ba.svg").read_bytes()
        self.assertEqual(result_svg,figure_svg)
        for i,run in enumerate(self.manifest["runs"]):
            name="Sigmoid-JS" if i==0 else "Softplus-JS"
            row=(f"{name} & {run['train_BA']*100:.4f} & {run['validation_BA']*100:.4f} "
                 f"& {run['I']:.6f} & {run['R_JS']:.6f} "
                 f"& {run['R_KL']:.6f} & {run['R_sqrt_JS']:.6f}")
            self.assertIn(row,table)

if __name__=="__main__":
    unittest.main()
