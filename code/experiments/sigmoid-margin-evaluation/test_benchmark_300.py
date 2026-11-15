"""Lightweight independent evidence link between E07 full300 and core Benchmark."""
import csv
import json
import unittest
from pathlib import Path

HERE=Path(__file__).resolve().parent
BENCHMARK=HERE.parent/"standard_policy_comparison/results/master_runs_76.csv"

class BenchmarkFull300Evidence(unittest.TestCase):
    def test_all_three_final_policies_match_benchmark_50_steps(self):
        report=json.loads((HERE/"results/metrics_full.json").read_text("utf-8"))
        replay=report["full300_replay"]
        self.assertEqual(replay["status"],"replayed_verified_saved_policy_no_training")
        self.assertTrue(replay["no_historical_Test"])
        # no_historical_Test is a flag: True means no historical Test was read
        with BENCHMARK.open(encoding="utf-8",newline="") as f:
            source=list(csv.DictReader(f))
        main=[r for r in source if r["study"]=="I_and_tau_50steps"
              and r["objective"]=="margin_JSD"
              and r["family"]=="shared" and r["n"]=="300"
              and r["steps"]=="50" and r["data_seed"]=="20261102"]
        self.assertEqual(len(main),3)
        self.assertEqual(len(replay["trained"]),3)
        for result in replay["trained"]:
            b=next(r for r in main if abs(float(r["tau"])-result["tau"])<1e-12)
            tr,val=result["train"],result["validation"]
            for v,expected in ((tr["balanced_accuracy"],b["train_BA"]),
                               (val["balanced_accuracy"],b["val_BA"]),
                               (tr["I"],b["I"]),
                               (tr["R_JS"],b["R_JS"]),
                               (tr["R_sqrt_JS"],b["R_sqrt_JS"]),
                               (tr["R_KL"],b["R_KL"]),
                               (tr["balanced_sigmoid_margin_loss"],b["objective_loss"])):
                self.assertAlmostEqual(v,float(expected),places=8)

if __name__=="__main__":
    unittest.main()
