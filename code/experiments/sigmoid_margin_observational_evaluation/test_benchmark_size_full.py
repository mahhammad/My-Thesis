"""Verify all 36 standardized size-study checkpoints have complete common reports."""
import csv
import json
import unittest
from pathlib import Path

ROOT=Path(__file__).resolve().parent

class FullSizeBenchmarkEvidence(unittest.TestCase):
    def test_full_train_val_v2_metric_coverage(self):
        j=json.loads((ROOT/"results/benchmark_size_metrics_full.json").read_text("utf8"))
        c=json.loads((ROOT/"results/coverage.json").read_text("utf8"))
        self.assertEqual(len(j["runs"]),36)
        self.assertEqual(c["measurement_counts"]["benchmark_full_v2_verified_runs"],36)
        self.assertFalse(j["retraining"])
        self.assertFalse(j["test_accessed"])
        self.assertEqual({(r["n"],r["family"],r["data_seed"]) for r in j["runs"]}.__len__(),36)
        self.assertEqual({r["n"] for r in j["runs"]},{50,100,150,200,250,300})
        self.assertEqual({r["family"] for r in j["runs"]},{"shared","class"})
        self.assertEqual({r["data_seed"] for r in j["runs"]},{20261102,20261103,20261104})
        with (ROOT/"results/benchmark_size_3seeds.csv").open(newline="",encoding="utf8") as f:
            rows=list(csv.DictReader(f))
        source={(int(r["n"]),r["family"],int(r["data_seed"])):r for r in rows}
        for rec in j["runs"]:
            original=source[(rec["n"],rec["family"],rec["data_seed"])]
            self.assertEqual(rec["source_sha256"],original["source_sha256"])
            self.assertLess(rec["max_diff_vs_benchmark"],2e-8)
            for split in ("train","validation"):
                x=rec[split]
                self.assertEqual(x["status"],"ok")
                for key in ("R_KL","R_JS","R_sqrt_JS","T","BT","TNL","BTNL","I",
                            "balanced_accuracy","ordinary_accuracy","margin_quantiles",
                            "NLL_KL_simple","confusion_true_rows_pred_cols"):
                    self.assertIn(key,x)

if __name__=="__main__":
    unittest.main()
