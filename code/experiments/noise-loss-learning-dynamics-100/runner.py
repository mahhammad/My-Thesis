"""Exact E-CH4-14 dual-objective 100-update trainer (fail closed by default).

Uses canonical thesis ExactMaskedWalk, canonical Sigmoid-JS (§4.13), canonical
Softplus-JS (§4.13), and canonical Train-only center geometry. Validation
labels are used only for accuracy, BA, precision, recall. No Test access.
Normal CLI prints the job grid and never trains.
"""
from __future__ import annotations
from hashlib import sha256
import argparse
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "code/experiments/benchmark_v1"))
from protocol import make_design, OBJECTIVES, FAMILIES, UPDATES, ADAM_LR, TAU, BETA, LAMBDA, H, sha
from engine import DT, LN2, ExactMaskedWalk, jsd
from replay_v1 import objective_loss
from train_missing_objectives import train_loss
from report_snapshot import standard_metrics

BRANCH = "experiment/ch4-14-noise-sensitivity-01"
WORKFLOW = "E-CH4-14 100-step dual-objective authorized experiment"
DATA = HERE / "inputs/derived"
RESULTS = HERE / "results"


def canonical_objective(graph, logits, family, objective):
    if objective == "sigmoid_JS_margin_tau005":
        return objective_loss(graph, graph.train_outputs(logits, family),
                              "margin_JSD", TAU)
    if objective == "softplus_JS_tau005":
        return train_loss(graph, logits, family, "softplus_JS_tau005")
    raise ValueError(f"Objective not in Chapter 4.13: {objective}")


def classification(scores, labels, n_classes=3):
    """Only class predictions and true labels; undefined precision set to 0."""
    actual = np.asarray(labels, dtype=np.int64)
    pred = np.asarray(scores).argmax(axis=1)
    confusion = np.zeros((n_classes, n_classes), dtype=np.int64)
    np.add.at(confusion, (actual, pred), 1)
    true_positive = np.diag(confusion).astype(float)
    recall = true_positive / np.maximum(1, confusion.sum(axis=1))
    precision = true_positive / np.maximum(1, confusion.sum(axis=0))
    return {"accuracy": float(np.mean(actual == pred)),
            "balanced_accuracy": float(np.mean(recall)),
            "precision_macro": float(np.mean(precision)),
            "recall_macro": float(np.mean(recall)),
            "precision_by_class": precision.tolist(),
            "recall_by_class": recall.tolist()}


def representations(graph, logits, family, *, query):
    fn = graph.represent_query if query else graph.represent_train
    if family == "shared":
        rep = fn(logits, family).detach().cpu().numpy()
        return np.repeat(rep[:, None, :], graph.k, axis=1)
    return np.stack([fn(logits, family, c).detach().cpu().numpy()
                     for c in range(graph.k)], axis=1)


def train_geometry_and_validation(graph, logits, family, validation_y):
    """Canonical R/I measured on Train; Validation solely classification."""
    with torch.no_grad():
        tr = representations(graph, logits, family, query=False)
        tr_y = graph.labels.detach().cpu().numpy()
        centers = np.stack([tr[tr_y == c, c, :].mean(axis=0)
                            for c in range(graph.k)])
        train_full = standard_metrics(tr, centers, tr_y, TAU,
                                      split="train", per_sample=False)
        tr_scores = np.stack([
            1 - jsd(torch.from_numpy(tr[:, c, :]),
                    torch.from_numpy(centers[c])).numpy() / LN2
            for c in range(graph.k)], axis=1)
        train = classification(tr_scores, tr_y)
        train.update({
            "density_r_kl": train_full["R_KL"],
            "density_r_js": train_full["R_JS"],
            "density_r_sqrt_js": train_full["R_sqrt_JS"],
            "separation_i": train_full["center_geometry"]["I"],
            "separation_t": train_full["center_geometry"]["T"],
            "separation_bt": train_full["center_geometry"]["BT"],
            "separation_tnl": train_full["center_geometry"]["TNL"],
            "separation_btnl": train_full["center_geometry"]["BTNL"],
            "margin_mean": float(np.mean([
                c["mean_margin"] * c["count"] for c in train_full["per_class"]
            ]) / len(tr_y)),
            "r_by_class": [
                {"R_KL": c["R_KL"], "R_JS": c["R_JS"],
                 "R_sqrt_JS": c["R_sqrt_JS"]} for c in train_full["per_class"]
            ],
        })
        va = representations(graph, logits, family, query=True)
        score = np.stack([
            1 - jsd(torch.from_numpy(va[:, c, :]),
                    torch.from_numpy(centers[c])).numpy() / LN2
            for c in range(graph.k)], axis=1)
        validation = classification(score, validation_y)
    return train, validation, centers


def exposure_from_logits(logits, feature_indices):
    """Unmasked auxiliary policy of Chapter 3.8, per hypothesis."""
    arr = logits.detach().cpu().numpy()
    matrices = [arr] if arr.ndim == 2 else list(arr)
    noise = [i for i, old in enumerate(feature_indices) if old not in (0, 1, 3)]
    pure = [i for i, old in enumerate(feature_indices) if old in (0, 1, 3)]
    out = []
    for matrix in matrices:
        a = matrix - matrix.max(axis=1, keepdims=True)
        p = np.exp(a) / np.exp(a).sum(axis=1, keepdims=True)
        p0, P = p[-1], p[:-1]
        dist = p0.copy()
        hit_no_noise = float(p0[pure] @ P[np.ix_(pure, pure)]
                             @ P[np.ix_(pure, pure)] @ np.ones(len(pure)))
        expected = 0.0
        q = []
        for _ in range(3):
            q.append(float(dist[noise].sum()))
            expected += q[-1]
            dist = dist @ P
        hit = 1.0 - hit_no_noise
        ref = 1.0 - (len(pure) / len(p0))**3
        out.append({"p0": p0.tolist(), "P": P.tolist(),
                    "q": q, "hit_at_least_once": hit,
                    "expected_noise_visits": expected,
                    "relative_hit_to_uniform": hit / ref if ref > 0 else None})
    return out


def _safe_number(x):
    if x is None: return None
    f = float(x)
    return f if math.isfinite(f) else None


def prepare_row(graph, z, optimizer, run, step, previous_update_l2):
    optimizer.zero_grad(set_to_none=True)
    loss = canonical_objective(graph, z, run["family"], run["objective"])
    if not bool(torch.isfinite(loss)):
        raise FloatingPointError(f"Nonfinite loss at {step}")
    loss.backward()
    if z.grad is None or not bool(torch.isfinite(z.grad).all()):
        raise FloatingPointError(f"Nonfinite gradient at {step}")
    train, validation, centers = train_geometry_and_validation(
        graph, z, run["family"], graph.validation_y)
    state = optimizer.state[z]
    m = state.get("exp_avg", torch.zeros_like(z))
    v = state.get("exp_avg_sq", torch.zeros_like(z))
    with torch.no_grad():
        train["loss"] = float(loss.detach())
        train["gradient_l2"] = float(torch.linalg.vector_norm(z.grad))
        record = {
            "step": step, "run_id": run["run_id"],
            "train": train, "validation": validation,
            "policy": {"centers": centers.tolist(),
                       "unmasked": exposure_from_logits(z, run["feature_indices"])},
            "optimizer": {
                "shape": list(z.shape), "logits": z.detach().flatten().tolist(),
                "gradients": z.grad.detach().flatten().tolist(),
                "adam_m": m.detach().flatten().tolist(),
                "adam_v": v.detach().flatten().tolist(),
                "previous_update_l2": previous_update_l2,
                "adam_update_to_next": None
            }
        }
    return record


def load_graph(run, data_dir):
    ds = run["dataset_id"]
    manifest_file = data_dir / "manifest.json"
    manifest = json.loads(manifest_file.read_text("utf8"))
    if manifest.get("test_accessed") is not False:
        raise ValueError("Unexpected Test provenance")
    record = next(x for x in manifest["datasets"] if x["dataset_id"] == ds)
    if record["feature_indices"] != run["feature_indices"]:
        raise ValueError("Feature mapping drift")
    source = data_dir / (ds + ".npz")
    if sha(source) != record["sha256"]:
        raise ValueError("Frozen subset hash mismatch")
    with np.load(source, allow_pickle=False) as f:
        if set(f.files) != {"train_x", "train_y", "validation_x", "validation_y", "gamma"}:
            raise ValueError("Test or other unapproved key in derived data")
        tx, ty, vx, vy, gamma = (f[k].copy() for k in
            ("train_x", "train_y", "validation_x", "validation_y", "gamma"))
    d = len(run["feature_indices"])
    if (tx.shape != (300, d) or vx.shape != (90, d) or
            ty.shape != (300,) or vy.shape != (90,) or gamma.shape != (d,)):
        raise ValueError("Unexpected derived dataset dimensions")
    if (not np.array_equal(np.unique(ty), np.arange(3))
            or not np.array_equal(np.unique(vy), np.arange(3))):
        raise ValueError("All Train and Validation classes required")
    graph = ExactMaskedWalk(tx, ty, gamma, beta=BETA, lam=LAMBDA,
                            horizon=H, query_x=vx)
    graph.validation_y = vy
    return graph


def authorization_gate(approval_path, source_sha):
    """A preliminary gate, NOT independent online proof of owner authorization."""
    if (os.environ.get("GITHUB_ACTIONS") != "true"
            or os.environ.get("GITHUB_REF_NAME") != BRANCH
            or os.environ.get("GITHUB_WORKFLOW") != WORKFLOW
            or os.environ.get("E14_100_AUTHORIZED") != "true"):
        raise PermissionError("Only separately authorized manual GitHub Actions fits allowed")
    if len(source_sha) != 40 or any(c not in "0123456789abcdef" for c in source_sha):
        raise ValueError("Commit SHA must be complete")
    if os.environ.get("GITHUB_SHA") != source_sha:
        raise PermissionError("Authorization must refer to the executed commit")
    if approval_path is None:
        raise PermissionError("Missing separately verified authorization")
    approval = json.loads(Path(approval_path).read_text("utf8"))
    if (approval.get("source_sha") != source_sha
            or approval.get("planned_fits") != 152
            or approval.get("updates_per_fit") != 100
            or approval.get("test_accessed") is not False
            or not str(approval.get("manager_approval_url", "")).startswith(
                "https://github.com/mahhammad/My-Thesis/issues/")):
        raise PermissionError("Research execution budget requires separate review")
    return approval


def execute_run(run, output_dir, data_dir):
    torch.set_num_threads(1)
    graph = load_graph(run, data_dir)
    d = graph.d
    shape = (d + 1, d) if run["family"] == "shared" else (3, d + 1, d)
    z = torch.nn.Parameter(torch.zeros(shape, dtype=DT))
    opt = torch.optim.Adam([z], lr=ADAM_LR)
    dest = output_dir / run["run_id"]
    dest.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    previous = None
    trace = dest / "trace.jsonl"
    try:
        with trace.open("x", encoding="utf8") as log:
            for step in range(UPDATES + 1):
                row = prepare_row(graph, z, opt, run, step, previous)
                row["elapsed_s"] = time.perf_counter() - started
                if step < UPDATES:
                    old = z.detach().clone()
                    opt.step()
                    if not bool(torch.isfinite(z).all()):
                        raise FloatingPointError("Nonfinite Adam update")
                    update = (z.detach() - old)
                    row["optimizer"]["adam_update_to_next"] = update.flatten().tolist()
                    previous = float(torch.linalg.vector_norm(update))
                log.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                log.flush()
                os.fsync(log.fileno())
        (dest / "complete.json").write_text(json.dumps({
            "schema": "E-CH4-14-100-run-complete-v1", "run_id": run["run_id"],
            "status": "COMPLETE_UNAUDITED", "updates": UPDATES,
            "trace_sha256": sha(trace), "test_accessed": False}, indent=2) + "\n",
            encoding="utf8")
    except BaseException:
        # Partial trace is preserved for forensic review, never auto-resumed.
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--execute", action="store_true")
    p.add_argument("--run-id", default=None)
    p.add_argument("--authorization", type=Path, default=None)
    p.add_argument("--source-sha", default="")
    p.add_argument("--data-dir", type=Path, default=DATA)
    p.add_argument("--out", type=Path, default=RESULTS)
    args = p.parse_args()
    design = make_design()
    runs = design["runs"]
    if args.run_id is not None:
        runs = [r for r in runs if r["run_id"] == args.run_id]
        if len(runs) != 1:
            raise ValueError("Unknown run ID")
    if not args.execute:
        print(json.dumps({"status": "DRY_RUN_NO_FIT", "jobs": len(runs),
            "steps": UPDATES, "test_accessed": False}, ensure_ascii=False))
        return
    authorization_gate(args.authorization, args.source_sha)
    for j, run in enumerate(runs, 1):
        print(f"[{j}/{len(runs)}] {run['run_id']}", flush=True)
        execute_run(run, args.out, args.data_dir)


if __name__ == "__main__":
    main()
