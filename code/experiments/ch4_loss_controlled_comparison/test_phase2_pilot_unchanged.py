"""Pilot immutable raw byte regression; no legacy BLAS-dependent regeneration."""
from hashlib import sha256
import json
from pathlib import Path
import unittest
HERE=Path(__file__).resolve().parent
BASE=HERE/"results/pilot_seed20261102_shared_two_fits"
class PilotRawPreservation(unittest.TestCase):
    def test_byte_exact_all_eight_raw_files(self):
        m=json.loads((BASE/"manifest.json").read_text("utf8"))
        self.assertEqual(m["source_commit"],"66a007928e129ad6dd5acffeaa93fc2485ee9775")
        self.assertEqual((m["n_fits"],m["steps_per_fit"],m["recovery_training_updates"]),(2,10,0))
        self.assertFalse(m["test_accessed"])
        for run in m["runs"]:
            for file in ("model.json","distributions.npz","trace.json","metrics_full.json"):
                with self.subTest(run=run["objective"],file=file):
                    payload=(BASE/run["objective"]/file).read_bytes()
                    self.assertEqual(sha256(payload).hexdigest(),run[file+"_sha256"])
    def test_original_traces_and_coverage_are_untouched(self):
        m=json.loads((BASE/"manifest.json").read_text("utf8"))
        cv=json.loads((BASE/"coverage.json").read_text("utf8"))
        self.assertEqual(cv["test"]["status"],"not_applicable")
        for run in m["runs"]:
            trace=json.loads((BASE/run["objective"]/"trace.json").read_text("utf8"))
            self.assertEqual([x["update_completed"] for x in trace],list(range(11)))
            self.assertEqual(len(cv["objectives"][run["objective"]]),43)
if __name__=="__main__":unittest.main()
