"""Read-only immutable three-seed Train/Validation fixture and tamper checks."""
from pathlib import Path
import json
import shutil
import sys
import tempfile
import unittest
import numpy as np
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sys.path.insert(0,str(ROOT/"code/experiments/benchmark_v1"))
from snapshot_pipeline import array_fingerprint
from phase2_frozen_data import load,INPUTS,sha
from phase2_protocol import SEEDS

class FrozenInputsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not (INPUTS/"manifest.json").exists():
            raise unittest.SkipTest("Inputs have not been frozen; no Test or model fit")

    def test_all_three_input_seeds_are_bytes_verified(self):
        m=json.loads((INPUTS/"manifest.json").read_text("utf8"))
        self.assertEqual(m["seed_set"],list(SEEDS))
        self.assertFalse(m["test_accessed"])
        self.assertTrue(m["train_validation_only"])
        for seed in SEEDS:
            with self.subTest(seed=seed):
                (tx,ty),(vx,vy)=load(seed)
                self.assertEqual(tx.shape,(300,5))
                self.assertEqual(vx.shape,(90,5))
                rec=m["records"][str(seed)]
                self.assertEqual(sha(INPUTS/rec["file"]),rec["file_sha256"])
                self.assertEqual(array_fingerprint(tx),rec["train_x_sha256"])
                self.assertEqual(array_fingerprint(vx),rec["validation_x_sha256"])
                self.assertEqual(sum(rec["train_counts"]),300)
                self.assertEqual(sum(rec["validation_counts"]),90)

    def test_corruption_is_rejected_without_touching_original_inputs(self):
        m=json.loads((INPUTS/"manifest.json").read_text("utf8"))
        seed=SEEDS[0]
        file=m["records"][str(seed)]["file"]
        with tempfile.TemporaryDirectory() as tmp:
            tmp=Path(tmp)
            shutil.copy2(INPUTS/"manifest.json",tmp/"manifest.json")
            shutil.copy2(INPUTS/file,tmp/file)
            self.assertEqual(load(seed,out=tmp)[0][0].shape,(300,5))
            with (tmp/file).open("ab") as f:f.write(b"corruption")
            with self.assertRaisesRegex(ValueError,"byte corruption"):load(seed,out=tmp)

    def test_refuse_missing_or_unapproved_seed(self):
        with self.assertRaises(ValueError):load(20261105)
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):load(SEEDS[0],out=tmp)

if __name__=="__main__":unittest.main()
