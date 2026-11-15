#!/usr/bin/env python3
"""RESEARCHER-03 center-only historical backfill: NO TRAINING, NO WALK REPLAY, NO TEST.

Runs only when explicitly passed --execute. Uses the historical ZIP archive members
and the *existing Chapter-3 canonical* center_metrics routine. Historical archives
and other branches are never written. This regenerates center-only metrics after
strict archive/member SHA-256 checks; it does not recompute per-sample statistics.

Example (checkout of experiment/ch4-metrics-backfill-v2):
  python code/experiments/benchmark_v1/report_backfill_v2/center_only_rebuild.py
  python code/experiments/benchmark_v1/report_backfill_v2/center_only_rebuild.py --execute --out-dir /tmp/ch4-r03-centers

Do not confuse this regeneration capability with a claim that all 76 were
recomputed during the bounded interactive research session.
"""
from __future__ import annotations

import argparse
import csv
from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import sys
from zipfile import ZipFile

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
BASE = ROOT / "code" / "experiments" / "standard_policy_comparison" / "results"
CODE = ROOT / "code" / "policy_evaluator" / "metrics.py"
METRIC_CONTRACT = "chapter4-common-evaluation-contract-v2"

EXPECTED_ARCHIVES = {
    "core_50_16runs.zip": (
        "9609eb96c207316fde26934cbef4668ba287a275d4485175d1da7e798a297040", 16
    ),
    "extra_25_12runs.zip": (
        "061a00923ae69ac2f8b923947bd9cdea9d4f3dba1d1771063bcb3ecffc0201e3", 12
    ),
    "size_25_24runs_seed20261102.zip": (
        "d570fb3f1dcab66476f027e1ade17b4e87c7ca16ff7fa6c76fda87b260c84e9f", 24
    ),
    "size_25_24runs_seeds20261103_20261104.zip": (
        "fdb862462e237e5094204ff17a5bc1a79a0f1d17f3a9b9d90ce967dba8b11fdd", 24
    ),
}


def reference_functions():
    spec = importlib.util.spec_from_file_location("r03_chapter3_center", CODE)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.center_metrics, module.jsd


def sha(raw: bytes) -> str:
    return sha256(raw).hexdigest()


def source_archive(row: dict) -> str:
    if int(row["steps"]) == 50:
        return "core_50_16runs.zip"
    if row["objective"] in ("margin_sqrtJSD", "margin_KL", "combined_I_R_margin"):
        return "extra_25_12runs.zip"
    if int(row["data_seed"]) == 20261102:
        return "size_25_24runs_seed20261102.zip"
    return "size_25_24runs_seeds20261103_20261104.zip"


def read_index():
    with (BASE / "master_runs_76.csv").open(encoding="utf8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    if len(rows) != 76 or len({r["file"] for r in rows}) != 76:
        raise ValueError("Master run IDs must be 76 unique filenames")
    counts = {fam: sum(r["family"] == fam for r in rows)
              for fam in ("shared", "class", "node", "node_class")}
    if counts != {"shared": 25, "class": 25, "node": 13, "node_class": 13}:
        raise ValueError("Family counts do not match source: " + str(counts))
    val = sum(bool(r["val_BA"]) for r in rows)
    if val != 50:
        raise ValueError(f"Unexpected numeric Validation count {val}")
    return {r["file"]: r for r in rows}


def archive_records(archive: str, master: dict):
    path = BASE / archive
    expected_sha, expected_n = EXPECTED_ARCHIVES[archive]
    actual_sha = sha(path.read_bytes())
    if actual_sha != expected_sha:
        raise ValueError(f"Archive checksum mismatch: {archive} {actual_sha}")
    with ZipFile(path) as z:
        manifests = {}
        for name in z.namelist():
            if name == "manifest.json" or name.endswith("/manifest.json"):
                m = json.loads(z.read(name))
                prefix = name[: -len("manifest.json")]
                for entry in m["runs"]:
                    member = prefix + entry["file"]
                    if member in manifests:
                        raise ValueError("Duplicate manifest entry " + member)
                    manifests[member] = entry["sha256"]
        parsed = []
        for member in z.namelist():
            if not member.endswith(".json") or member.endswith("manifest.json"):
                continue
            name = Path(member).name
            if name not in master:
                raise ValueError(f"Unknown archive member: {member}")
            row = master[name]
            if source_archive(row) != archive:
                raise ValueError("Wrong source archive: " + member)
            raw = z.read(member)
            digest = sha(raw)
            if digest != row["sha256"] or digest != manifests.get(member):
                raise ValueError("Member SHA disagrees with master/manifest " + member)
            original = json.loads(raw)
            if original["family"] != row["family"] or original["objective"] != row["objective"]:
                raise ValueError("Checkpoint identity mismatch: " + member)
            if int(original["steps"]) != int(row["steps"]):
                raise ValueError("Step budget mismatch: " + member)
            if original["dataset"]["data_seed"] != int(row["data_seed"]):
                raise ValueError("Data seed mismatch: " + member)
            if original["dataset"].get("test_used") is not False:
                raise ValueError("Test-use evidence is not FALSE: " + member)
            parsed.append((row, original, member, digest, actual_sha))
    if len(parsed) != expected_n:
        raise ValueError(f"{archive}: expected {expected_n}, found {len(parsed)}")
    return parsed


def center_report(center, center_metrics, jsd):
    c = np.asarray(center, dtype=np.float64)
    if c.ndim != 2 or c.shape[0] != c.shape[1]:
        raise ValueError("Center matrix must be K by K")
    if not np.isfinite(c).all() or (c < 0).any():
        raise ValueError("Nonfinite/negative center")
    if not np.allclose(c.sum(axis=1), 1.0, atol=1e-10, rtol=0):
        raise ValueError("Center rows must sum to one")
    result = center_metrics(c)
    pair = np.asarray(result["center_jsd"], dtype=np.float64)
    k = len(c)
    diag_mask = np.eye(k, dtype=bool)
    pair_sqrt = np.sqrt(pair)
    upper = np.triu_indices(k, k=1)
    nearest = np.where(diag_mask, np.inf, pair_sqrt).min(axis=1)
    metrics = {
        "T": float(result["T"]),
        "BT": float(result["BT"]),
        "TNL": float(result["TNL"]),
        "BTNL": float(result["BTNL"]),
        "I": float(result["I"]),
        "min_pairwise_sqrt_JS": float(np.min(nearest)),
        "mean_pairwise_sqrt_JS": float(np.mean(pair_sqrt[upper])),
        "mean_pairwise_JS": float(np.mean(pair[upper])),
        "nearest_center_sqrt_JS_per_class": nearest.tolist(),
        "pairwise_JS": pair.tolist(),
        "pairwise_sqrt_JS": pair_sqrt.tolist(),
        "pairwise_JS_similarity": (1 - pair / np.log(2)).tolist()
    }
    if not np.allclose(pair, jsd(c[:, None], c[None, :]), atol=1e-12, rtol=0):
        raise ValueError("Canonical JS geometry mismatch")
    return metrics


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--execute", action="store_true", help="Read ZIPs and calculate center-only metrics")
    p.add_argument("--out-dir", type=Path, default=HERE,
                   help="Derived output folder; do not overwrite raw archives")
    args = p.parse_args()
    if not args.execute:
        print(json.dumps({
            "status": "DRY_RUN", "action": "center-only, no inference/no training",
            "required_archives": list(EXPECTED_ARCHIVES),
            "sources": str(BASE), "output_dir": str(args.out_dir),
            "test_accessed": False,
        }, indent=2))
        return

    center_metrics, jsd = reference_functions()
    master = read_index()
    fields = [
        "run_id", "source_archive", "source_member", "member_sha256",
        "data_seed", "n_ref", "family", "objective", "tau", "steps",
        "T", "BT", "TNL", "BTNL", "I", "legacy_I",
        "min_pairwise_sqrt_JS", "legacy_min_pairwise_sqrt_JS",
        "mean_pairwise_sqrt_JS", "mean_pairwise_JS", "legacy_mean_pairwise_JS",
        "nearest_center_sqrt_JS_per_class", "pairwise_JS",
        "pairwise_sqrt_JS", "pairwise_JS_similarity"
    ]
    results, mismatches = [], []
    for archive in EXPECTED_ARCHIVES:
        for row, source, member, digest, archive_sha in archive_records(archive, master):
            centers = source["final"]["train"]["centers"]
            metrics = center_report(centers, center_metrics, jsd)
            prior = source["final"]["train"]
            checks = {
                "I": ("I", 1e-10), "min_pairwise_sqrt_JS": ("min_pairwise_sqrt_JS", 1e-10),
                "mean_pairwise_JS": ("mean_pairwise_JS", 1e-10)
            }
            for key, (legacy_key, tolerance) in checks.items():
                delta = abs(float(metrics[key]) - float(prior[legacy_key]))
                if delta > tolerance:
                    mismatches.append({"run_id": row["file"], "metric": key,
                                       "delta": delta, "tolerance": tolerance})
            out = {
                "run_id": row["file"], "source_archive": archive,
                "source_member": member, "member_sha256": digest,
                "data_seed": int(row["data_seed"]), "n_ref": int(row["n"]),
                "family": row["family"], "objective": row["objective"],
                "tau": float(row["tau"]), "steps": int(row["steps"]),
                "legacy_I": float(prior["I"]),
                "legacy_min_pairwise_sqrt_JS": float(prior["min_pairwise_sqrt_JS"]),
                "legacy_mean_pairwise_JS": float(prior["mean_pairwise_JS"]),
                **metrics,
            }
            results.append(out)
    if mismatches:
        raise ValueError(f"Center consistency regression: {mismatches[:3]}")
    if len(results) != 76 or len({row["run_id"] for row in results}) != 76:
        raise ValueError("Expected exactly 76 unique verified centers")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    dest = args.out_dir / "center_only_metrics.csv"
    with dest.open("w", encoding="utf8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in sorted(results, key=lambda x: x["run_id"]):
            writer.writerow({k: json.dumps(v, separators=(",", ":")) if isinstance(v, list)
                             else v for k, v in row.items()})
    diag = {
        "schema": "researcher03-center-only-full-checkout-v2",
        "source": "all four historical ZIPs on exact local repository checkout",
        "source_commit": "de02fdda91e23db177de4429b3e1a9693c046d83",
        "contract": METRIC_CONTRACT,
        "processed": len(results), "archive_sha_verified": True,
        "member_sha_verified": True, "max_tolerance": 1e-10,
        "test_accessed": False, "optimizer_steps_executed": 0,
        "walk_inference_replays": 0, "center_regressions": mismatches
    }
    (args.out_dir / "center_only_rebuild_audit.json").write_text(
        json.dumps(diag, ensure_ascii=False, indent=2) + "\n", encoding="utf8")
    print(json.dumps(diag, ensure_ascii=False))


if __name__ == "__main__":
    main()
