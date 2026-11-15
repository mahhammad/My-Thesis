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
    "sigmoid-margin-evaluation/replay_benchmark_size.py",
    "sigmoid-margin-evaluation/test_benchmark_300.py",
    "true-horizon-effect/analyze.py",
    "test_chapter4_presentation.py",
    "benchmark_v1/backfill_saved_scores.py",
    "standard_policy_comparison/r03_center_recovery/center_only_rebuild.py",
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
        for name, expected_sha in {"run.py":"3485b1c9c1c2d97ecf3fa8f91fe5bd5b1ed656e9","bootstrap_inverse.py":"bac81cc95e37c0b196aac8080c317dec577e5f69","stream_graph.py":"aaa268314bc6049e1fb75a8105cafbce000b2713","observe.py":"53a30050a8b18ba6949f3707fea02691779cb2a1","projection.py":"91ba43f58486735fd2141d3b095df959a666bbda"}.items():
            path = BENCH / name
            self.assertTrue(path.is_file(), path)
            payload = path.read_bytes()
            gitblob = b"blob " + str(len(payload)).encode() + b"\x00" + payload
            self.assertEqual(hashlib.sha1(gitblob).hexdigest(), expected_sha)
        for retired in ("diag_max_full300", "multistep_policy_target_match", "center_matching_observation", "synthetic_data_structure_sensitivity"):
            self.assertFalse((EXP / retired).exists(), retired)


    def test_runtime_sources_refer_to_shared_benchmark_paths(self):
        forbidden = ("code/experiments/diag_max_full300",
                     "code/experiments/multistep_policy_target_match",
                     "code/experiments/center_matching_observation")
        historical_pins = {BENCH / "run.py", BENCH / "stream_graph.py",
                           EXP / "inverse_attainable_standard/b02_gate_v3/pure_gate.py",
                           EXP / "inverse_attainable_standard/b02_gate_candidate/pure_gate.py"}
        stale = []
        for file in EXP.rglob("*.py"):
            if file in historical_pins or file == Path(__file__):
                continue
            data = file.read_text(encoding="utf-8")
            for marker in forbidden:
                if marker in data:
                    stale.append((str(file.relative_to(EXP)), marker))
        self.assertEqual(stale, [])

    def test_no_active_python_code_reads_retired_archive_paths(self):
        unresolved = []
        trust_contract = EXP / "inverse_attainable_standard/runner_v3/data_contract.py"
        pinned_blob = "4984961c6a8fd8057a7c1ff3d292d30ad6c9afcb"
        for source in sorted(EXP.rglob("*.py")):
            if source.resolve() == Path(__file__).resolve():
                continue  # This audit's search expression is not an archive dependency.
            if source == trust_contract:
                # Historical owner-authenticated roots pin this file EXACTLY.
                # Its original archive paths are intentionally frozen, and it
                # must only be run in the historical tagged worktree.
                raw = source.read_bytes()
                gitblob = b"blob " + str(len(raw)).encode() + b"\0" + raw
                self.assertEqual(hashlib.sha1(gitblob).hexdigest(), pinned_blob)
                continue
            if "code/experiments/archive/" in source.read_text("utf-8"):
                unresolved.append(str(source.relative_to(ROOT)))
        self.assertEqual(unresolved, [], "Retired archive path still in live Python code")

    def test_frozen_e06_reference_code_is_present(self):
        for name in ("observe.py", "projection.py"):
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
