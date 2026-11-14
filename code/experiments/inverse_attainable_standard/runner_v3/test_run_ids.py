"""Pure 90-run plan crosscheck: no graph, optimizer or data regeneration."""
from __future__ import annotations
import json,sys,unittest
from pathlib import Path

ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(Path(__file__).resolve().parent))
from run_one import planned_run_id
PLAN=ROOT/"code/experiments/inverse_attainable_standard/standard_plan/plan_108.json"

class IdMappingTests(unittest.TestCase):
    def test_every_b02_to_b06_plan_row(self):
        p=json.loads(PLAN.read_text())
        rows=[r for r in p["runs"] if r["block_id"].split("-")[1] in
              ("B02","B03","B04","B05","B06")]
        self.assertEqual(len(rows),90)
        for row in rows:
            short=row["block_id"].split("-")[1]
            self.assertEqual(planned_run_id(short,row["class_index"],
                row["initialization"],row["solver"]),row["run_id"])
    def test_b02_seed_distinct_from_b06(self):
        self.assertEqual(planned_run_id("B02",0,"zero","Q-only"),
            "INV01-B02-n300-data20261102-C0-Z-Q")
        self.assertEqual(planned_run_id("B06",0,"zero","Q-only"),
            "INV01-B06-n300-data20261104-C0-Z-Q")
    def test_wrong_scope_is_denied(self):
        from data_contract import Deny
        for b in ("B01","B07",None):
            with self.assertRaises(Deny):
                planned_run_id(b,0,"zero","Q-only")
        with self.assertRaises(Deny):
            planned_run_id("B02",3,"zero","Q-only")

if __name__=="__main__":
    unittest.main()
