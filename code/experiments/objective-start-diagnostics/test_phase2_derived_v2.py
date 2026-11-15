"""No-training, no-Test regression checks for post-audit derived coverage and 11 SVGs."""
from __future__ import annotations
import json
import shutil
import tempfile
import unittest
from pathlib import Path

import phase2_derived_reproduce_v2 as derived

HERE=Path(__file__).resolve().parent

class DerivedScientificQAFixesV2(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original_raw=derived.RAW
        cls.manifest,cls.data=derived.inputs()

    def test_all_12_coverage_aliases_match_true_confusions(self):
        files=derived.corrected(self.data)
        self.assertEqual(len(files),13)
        for item in self.manifest["runs"]:
            key="derived/phase2_qa_fixes_v2/coverage/"+item["run_key"]+".json"
            corrected=json.loads(files[key])
            raw=json.loads((derived.RAW/item["run_key"]/"coverage.json").read_text())
            for split in ("train","validation"):
                self.assertEqual(raw[split+".per_class_recall"]["status"],"not_measured")
                self.assertEqual(corrected[split+".per_class_recall"]["status"],"measured")
            # No other canonical metric changed.
            original_others={k:v for k,v in raw.items() if not k.endswith(".per_class_recall")}
            fixed_others={k:v for k,v in corrected.items() if not k.endswith(".per_class_recall")}
            self.assertEqual(original_others,fixed_others)

    def test_11_plot_output_names_and_value_sources(self):
        files=derived.produce()
        imgs=[key for key in files if key.startswith("figures/phase2_repro_v2/")]
        self.assertEqual(len(imgs),11)
        self.assertEqual(len(files),27)
        for file in imgs:
            self.assertTrue(files[file].startswith(b'<svg '))
            self.assertTrue(files[file].rstrip().endswith(b'</svg>'))
        p=derived.pair_data(self.data)
        self.assertEqual(len(p),6)
        for family in derived.FAMILIES:
            for seed in derived.SEEDS:
                pair=next(x for x in p if x["family"]==family and x["seed"]==seed)
                self.assertTrue(all(pair[split+"_"+metric+"_sigmoid"]>pair[split+"_"+metric+"_softplus"]
                                    for split in ("train","validation") for metric in derived.RS))
                self.assertGreater(pair["delta_I"],0)

    def test_normalization_guard_and_grid(self):
        trace=derived.trajectories(self.data)
        self.assertEqual(len(trace),104)
        for f in derived.FAMILIES:
            for loss in derived.LOSSES:
                row=next(r for r in trace if r["family"]==f and r["loss"]==loss and r["update_completed"]==0)
                self.assertAlmostEqual(row["mean_loss_divided_by_initial"],1,places=12)
                self.assertAlmostEqual(row["mean_gradient_divided_by_initial"],1,places=12)

    def test_all_generated_files_byte_reproducible_in_repo(self):
        outputs=derived.produce()
        for rel,contents in outputs.items():
            target=HERE/rel
            self.assertTrue(target.is_file(),rel)
            self.assertEqual(contents,target.read_bytes(),rel)

    def test_tampered_raw_is_rejected_without_touching_original(self):
        with tempfile.TemporaryDirectory() as tmp:
            counterfeit=Path(tmp)/"raw"
            shutil.copytree(derived.RAW,counterfeit)
            file=counterfeit/self.manifest["runs"][0]["run_key"]/"coverage.json"
            file.write_bytes(file.read_bytes()+b'\n ')
            try:
                derived.RAW=counterfeit
                with self.assertRaisesRegex(ValueError,"Original byte SHA mismatch"):
                    derived.inputs()
            finally:
                derived.RAW=self.original_raw

if __name__=="__main__":
    unittest.main()
