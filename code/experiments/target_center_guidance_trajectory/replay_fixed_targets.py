"""E-CH4-06 fixed-target replay: fill metrics without rerunning center updates.

The 49 historical target matrices are preserved. For each target, regenerate
the original class-conditioned MC train and held-out walks. Abort if recreated
centers or any historical BA disagree. Never read/generate historical Test.
Do not use newly computed Validation values to reselect a favored round.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
for source in (
    ROOT / "code/data_generator",
    ROOT / "code/experiments/benchmark_v1",
    ROOT / "code/experiments/center_matching_observation",
    ROOT / "code/policy_evaluator",
):
    sys.path.insert(0, str(source))
from generate import load_config
from dataset_adapter import dataset_train_val
from walks import ReferenceGraph
from observe import CenterPolicyWalker
from report_snapshot import standard_metrics, jsd


def main(max_rounds=49, commit_results=True):
    data = json.loads((HERE / "results/history_00_48.json").read_text(encoding="utf-8"))
    config = data["config"]
    history = data["history"]
    if len(history) != 49 or (config["seed"], config["walks_per_sample"],
                             config["optimizer_steps"]) != (20261104,100,45):
        raise ValueError("Archive configuration mismatch")
    spec = load_config(ROOT / "code/data_generator/config.json")
    splits = dataset_train_val(spec)
    x, y = splits["train"]
    vx, vy = splits["validation"]
    if len(y) != 300 or len(vy) != 90:
        raise ValueError("Expected only historical Train and Validation")
    saved = json.loads((HERE / "results/representations_guided_round0.json")
                       .read_text(encoding="utf8"))
    if list(y) != saved["training"]["y"] or list(vy) != saved["validation"]["y"]:
        raise AssertionError("Generated data labels do not match original records")
    graph = ReferenceGraph(x, y, spec["gamma"])
    engine = CenterPolicyWalker(graph, beta=config["beta"], lam=config["lambda"],
                                horizon=config["max_steps"], nwalks=100,
                                optimizer_steps=45)
    cases = []
    original_observed = 0
    for row in history[:max_rounds]:
        i = int(row["round"])
        targets = np.asarray(row["targets"], dtype=float)
        orig_center = np.asarray(row["realized_centers"], dtype=float)
        p, diagnostic = engine.train_representations(targets, seed=config["seed"])
        centers = np.stack([p[y == c].mean(axis=0) for c in range(graph.nclasses)])
        err = float(np.max(np.abs(centers - orig_center)))
        if err > 1e-8:
            raise AssertionError(f"Round {i}: center mismatch {err}; cannot certify replay")
        v, vdiag = engine.heldout_hypotheses(vx, targets, seed=config["seed"]+1)
        tr = standard_metrics(np.broadcast_to(p[:,None,:],(len(y),3,3)),
                              orig_center, y, .005, split="train", per_sample=False)
        va = standard_metrics(v, orig_center, vy, .005,
                              split="validation", per_sample=False)
        historic_r = row.get("train_within_class_jsd")
        if historic_r is None:
            historic_r = row.get("train_oracle_target", {}).get("within_class_jsd")
        # The post-resume legacy summary may use a different aggregation.
        # Do not force it to agree with balanced Chapter-3 R_JS.
        historic_r_diagnostic = None
        if historic_r is not None:
            ordinary_r = float(jsd(p, orig_center[y]).mean())
            historic_r_diagnostic = {
                "historical_value": float(historic_r),
                "balanced_v2": tr["R_JS"],
                "ordinary_unweighted": ordinary_r,
                "matches_balanced": abs(tr["R_JS"] - float(historic_r)) < 1e-8,
                "matches_unweighted": abs(ordinary_r - float(historic_r)) < 1e-8
            }
            if i < 5 and not historic_r_diagnostic["matches_balanced"]:
                raise AssertionError(f"Round {i}: balanced historical R_JS mismatch")
            if not historic_r_diagnostic["matches_balanced"]:
                print(f"R_JS_AGGREGATION_DIFFERENCE round={i:02d} "
                      f"historical={historic_r} balanced={tr['R_JS']:.9f} "
                      f"unweighted={ordinary_r:.9f}", flush=True)
        historical_validation = row.get("validation_hypothesis")
        if historical_validation is not None:
            original_observed += 1
            if abs(va["balanced_accuracy"]-historical_validation["balanced_accuracy"])>1e-10:
                raise AssertionError(f"Round {i}: recorded Validation BA mismatch")
            if abs(va["ordinary_accuracy"]-historical_validation["accuracy"])>1e-10:
                raise AssertionError(f"Round {i}: recorded Validation Acc mismatch")
        if i in (0,4):
            stored = json.loads((HERE / f"results/representations_guided_round{i}.json")
                                .read_text(encoding="utf8"))
            if not np.allclose(p, stored["training"]["P"], atol=1e-8, rtol=0):
                raise AssertionError(f"Round {i}: archived per-sample train distributions mismatch")
            if not np.allclose(v, stored["validation"]["candidate_P"], atol=1e-8, rtol=0):
                raise AssertionError(f"Round {i}: archived per-sample validation distributions mismatch")
        cases.append({
            "round": i, "source": "fixed historical target MC replay",
            "match_historical_centers_max_abs": err,
            "historical_validation_observed": historical_validation is not None,
            "historical_R_JS_aggregation_diagnostic": historic_r_diagnostic,
            "train_oracle_origin_class": True,
            "validation_inference": "each candidate class simulated without actual query label",
            "train": tr, "validation": va
        })
        print(f"REPLAY_46 round={i:02d} center_err={err:.2e} "
              f"train_R_JS={tr['R_JS']:.9f} validation_BA={va['balanced_accuracy']:.9f}",
              flush=True)
    if not commit_results:
        print(f"SMOKE_OK {len(cases)} cases", flush=True)
        return
    if len(cases)!=49 or original_observed!=25:
        raise AssertionError("Expected full 49 rounds and 25 historical Validation records")
    path = HERE / "results/metrics_full.json"
    full = json.loads(path.read_text(encoding="utf-8"))
    full["replayed_fixed_targets"] = {
        "status": "verified_against_archived_centers_and_observed_BA_with_R_JS_aggregation_audit",
        "no_historical_Test_accessed": True,
        "historical_49_centers_unchanged": True,
        "historical_BA_25_unchanged": True,
        "historical_R_JS_aggregation": "saved separately; some post-resume values differ from balanced v2",
        "new_validation_BA_24": "replayed_only; not in original selected 25",
        "inference_note": "Guided train is oracle-origin-conditioned and not deployable BA",
        "sampling": "same 100 MC walks/origin, 45 FW steps, fixed targets from historical record",
        "cases": cases
    }
    path.write_text(json.dumps(full,indent=2,ensure_ascii=False,allow_nan=False)+"\n",
                    encoding="utf-8")
    path = HERE / "results/coverage.json"
    cov = json.loads(path.read_text(encoding="utf-8"))
    cov["replay_addendum"] = {
        "status": "49/49 fixed-target replay verified",
        "train_and_validation_common_metrics": "49 rounds, all metric groups",
        "historical_25_BA_values": "unchanged and verified",
        "additional_24_validation_BA_values": "new replay; not historical measurements",
        "independent_data_seeds": 1,
        "test_accessed": False,
        "historical_R_JS_aggregation": "replay separately flags any changed aggregation; not silently overwritten",
        "source": "results/metrics_full.json#replayed_fixed_targets"
    }
    for r in cov.get("not_measured", []):
        if r.get("scope", "").startswith("guided rounds") or r.get("scope", "").startswith("24 of 49"):
            r["scope"] += " (original archived trajectory only; recovered in replay addendum)"
            r["reason"] += " New fixed-target replay is reported separately; not counted as originally observed."
    path.write_text(json.dumps(cov,indent=2,ensure_ascii=False,allow_nan=False)+"\n",
                    encoding="utf-8")
    print("E-CH4-06 FULL 49 REPLAY VERIFIED without Test",flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke",action="store_true")
    arg = parser.parse_args()
    main(max_rounds=1 if arg.smoke else 49,commit_results=not arg.smoke)
