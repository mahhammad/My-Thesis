"""Independently regenerate Train/Validation outputs from each stored v2 snapshot.

No training and no historical Test. Do not use saved distribution arrays as
inputs; only compare against them after re-running the stored effective policy.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "code/data_generator"))

from generate import load_config
from dataset_adapter import dataset_train_val
from policy_interface import evaluate_policy
from snapshot_pipeline import array_fingerprint, config_fingerprint, load_snapshot


def verify(run_dir, *, config):
    run_dir = Path(run_dir)
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf8"))
    if manifest["schema"] != "ch4-fixed-baselines-p1p2-v1":
        raise ValueError("Wrong registered experiment schema")
    if manifest["config_sha256"] != config_fingerprint(config):
        raise ValueError("Generator config SHA256 mismatch")
    spec = load_config(config)
    spec["seed"] = int(manifest["data_seed"])
    splits = dataset_train_val(spec)  # NEVER Test
    all_x, all_y = splits["train"]
    qx, _ = splits["validation"]
    if len(manifest["runs"]) != manifest["run_count"] or len(set(manifest["run_ids"])) != manifest["run_count"]:
        raise ValueError("Malformed manifest")
    for run in manifest["runs"]:
        model, arrays = load_snapshot(run_dir / run["run_dir"])
        params = model["parameters"]
        ids = np.asarray(model["data"]["reference_ids"], dtype=np.int64)
        assert model["schema"] == "imc-snapshot-v2"
        if model["data"]["test_accessed"] or model["data"]["data_seed"] != spec["seed"]:
            raise ValueError("Forbidden data exposure or seed mismatch")
        if array_fingerprint(all_x[ids]) != model["data"]["train_reference_x_sha256"]:
            raise ValueError("Train X fingerprint mismatch")
        if array_fingerprint(qx) != model["data"]["validation_x_sha256"]:
            raise ValueError("Validation X fingerprint mismatch")
        effective = evaluate_policy(
            {"ref_x": all_x[ids], "ref_y": all_y[ids], "query_x": qx},
            model["family"], {"logits": params["logits"]},
            gamma=params["gamma"], beta=params["beta"],
            lam=params["lambda"], horizon=params["H"]
        )
        if effective["validation_status"] != "available":
            raise ValueError("Undefined Validation in a registered baseline")
        for key in ("train_by_hypothesis", "validation_by_hypothesis", "centers"):
            np.testing.assert_allclose(effective[key], arrays[key], atol=1e-12, rtol=0,
                                       err_msg=f"Round-trip failed for {run['id']}:{key}")
        print("ROUNDTRIP_OK", run["id"], flush=True)
    print("VERIFIED", manifest["run_count"], "Train/Validation snapshots without Test", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True, type=Path)
    p.add_argument("--config", type=Path, default=ROOT / "code/data_generator/config.json")
    args = p.parse_args()
    verify(args.run_dir, config=args.config)


if __name__ == "__main__":
    main()
