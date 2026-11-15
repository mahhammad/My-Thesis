"""Read-only linkage checks for Chapter 4 benchmark archive cleanup.

No optimizer, graph replay, Test-set access, or data modification.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import unittest
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / "code" / "experiments"
OFFICIAL = EXP / "standard_policy_comparison" / "results"
BENCH = EXP / "benchmark_v1"
CURATED = EXP / "archive" / "benchmark_v1_legacy_evidence"

ZIP_SHA256 = {
    "core_50_16runs.zip": "9609eb96c207316fde26934cbef4668ba287a275d4485175d1da7e798a297040",
    "extra_25_12runs.zip": "061a00923ae69ac2f8b923947bd9cdea9d4f3dba1d1771063bcb3ecffc0201e3",
    "size_25_24runs_seed20261102.zip": "d570fb3f1dcab66476f027e1ade17b4e87c7ca16ff7fa6c76fda87b260c84e9f",
    "size_25_24runs_seeds20261103_20261104.zip": "fdb862462e237e5094204ff17a5bc1a79a0f1d17f3a9b9d90ce967dba8b11fdd",
}

CONSUMERS = (
    "exact_matrix_walk_validation/run.py",
    "sigmoid_margin_observational_evaluation/replay_benchmark_size.py",
    "sigmoid_margin_observational_evaluation/test_benchmark_300.py",
    "walk_depth_effect/analyze.py",
    "test_chapter4_presentation.py",
    "benchmark_v1/backfill_saved_scores.py",
    "benchmark_v1/report_backfill_v2/center_only_rebuild.py",
    "benchmark_v1/inverse_centers_v1.py",
    "benchmark_v1/train_missing_objectives.py",
    "benchmark_v1/train_class_tied_node_intermediate.py",
    "benchmark_v1/node_class_validation_starts.py",
)

SHARED_MODULES = (
    "engine.py", "dataset_adapter.py", "metrics.py", "policy_interface.py",
    "policy_evaluator.py", "snapshot_pipeline.py", "report_snapshot.py",
    "test_engine.py", "test_snapshot_roundtrip.py", "test_snapshot_pipeline.py",
)


class BenchmarkArchiveLinkage(unittest.TestCase):
    def test_four_canonical_archives_are_identical_to_registered_sha256(self):
        for filename, expected in ZIP_SHA256.items():
            with self.subTest(filename=filename):
                path = OFFICIAL / filename
                self.assertTrue(path.is_file(), path)
                self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), expected)
                with ZipFile(path) as archive:
                    self.assertIsNone(archive.testzip())
                self.assertFalse((BENCH / "results" / filename).exists())

    def test_76_archived_runs_and_center_only_inventory(self):
        with (OFFICIAL / "master_runs_76.csv").open(encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 76)
        self.assertEqual(len({r["file"] for r in rows}), 76)
        with (OFFICIAL / "center_only_metrics.csv").open(encoding="utf-8", newline="") as stream:
            center_rows = list(csv.DictReader(stream))
        self.assertEqual(len(center_rows), 76)
        coverage = json.loads((OFFICIAL / "coverage_manifest_76.json").read_text("utf-8"))
        self.assertEqual(len(coverage["runs"]), 76)

    def test_curated_legacy_evidence_matches_original_git_blobs(self):
        manifest = json.loads((CURATED / "MOVE_MANIFEST.json").read_text("utf-8"))
        moves = manifest["moves"]
        self.assertEqual(len(moves), 20)
        self.assertEqual(len({item["curated_path"] for item in moves}), 20)
        for item in moves:
            with self.subTest(source=item["source_path"]):
                file = ROOT / item["curated_path"]
                self.assertTrue(file.is_file(), file)
                data = file.read_bytes()
                self.assertEqual(len(data), item["size_bytes"])
                git_blob = b"blob " + str(len(data)).encode("ascii") + b"\0" + data
                self.assertEqual(hashlib.sha1(git_blob).hexdigest(), item["git_blob_sha1"])
                self.assertFalse((ROOT / item["source_path"]).exists())

    def test_archive_consumers_and_shared_runtime_are_preserved(self):
        for name in CONSUMERS:
            with self.subTest(consumer=name):
                self.assertIn("standard_policy_comparison", (EXP / name).read_text("utf-8"))
        for name in SHARED_MODULES:
            self.assertTrue((BENCH / name).is_file(), name)

    def test_e13_frozen_inputs_and_figure_manifest_remain(self):
        e13 = EXP / "ch4_loss_controlled_comparison"
        for name in (
            "phase2_runner.py", "phase2_report.py", "phase2_frozen_data.py",
            "results/phase2_300_25/manifest.json",
            "derived/phase2_persian_figures_manifest.json",
        ):
            self.assertTrue((e13 / name).is_file(), name)


if __name__ == "__main__":
    unittest.main()
