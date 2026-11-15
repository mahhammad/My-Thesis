"""Read-only linkage checks for Chapter 4 benchmark archive cleanup.

No optimizer, graph replay, Test-set access, or data modification.
"""
from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path
import unittest
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / "code" / "experiments"
OFFICIAL = EXP / "standard_policy_comparison" / "results"
BENCH = EXP / "benchmark_v1"
RECOVERY = ROOT / "research_journal" / "archive_recovery"

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

    def test_137_retired_archive_files_recover_from_frozen_tag(self):
        self.assertFalse((EXP / "archive").exists(), "Retired archive must not remain in active checkout")
        man = json.loads((RECOVERY / "ARCHIVE_RECOVERY_2026-11-15.json").read_text("utf-8"))
        self.assertEqual(man["recovery_commit"], "a8c9027460e538fa04b8aab2f762eb4ba4c31e3d")
        self.assertEqual(man["expected_files"], 137)
        self.assertEqual(man["expected_bytes"], 3491507)
        p = subprocess.run(
            [sys.executable, str(RECOVERY / "restore_archive.py")],
            cwd=ROOT, text=True, capture_output=True, check=False,
        )
        self.assertEqual(p.returncode, 0, p.stdout + "\n" + p.stderr)
        report = json.loads(p.stdout)
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(report["verified_files"], 137)
        self.assertEqual(report["verified_bytes"], 3491507)
        self.assertFalse(report["restored"])

    def test_archive_consumers_and_shared_runtime_are_preserved(self):
        for name in CONSUMERS:
            with self.subTest(consumer=name):
                self.assertIn("standard_policy_comparison", (EXP / name).read_text("utf-8"))
        for name in SHARED_MODULES:
            self.assertTrue((BENCH / name).is_file(), name)
        for path in (
            "multistep_policy_target_match/run.py",
            "multistep_policy_target_match/bootstrap_inverse.py",
            "diag_max_full300/stream_graph.py",
        ):
            self.assertTrue((EXP / path).is_file(), path)


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
