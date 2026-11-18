"""NumPy-only E-CH4-14 exact-walk dual-objective 100-step learner.

No torch or automatic fitting. Both losses match thesis §4.13. Every update
records all logits, analytic gradients, Adam moments/updates and Train I/R;
Validation is classification-only, no checkpoint selection or Test access.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
from protocol import (HERE, make_design, OBJECTIVES, FAMILIES, UPDATES,
                      ADAM_LR, TAU, BETA, LAMBDA, H, sha)
from numpy_engine import ExactMaskedWalkNP
from numpy_objectives import (train_loss_and_grad, train_diagnostics,
                              validation_diagnostics, classification)

BRANCH = "experiment/ch4-14-noise-sensitivity-01"
WORKFLOW = "E-CH4-14 100-step NumPy-only authorized experiment"
DATA = HERE / "inputs/derived"
RESULTS = HERE / "results"


class AdamNP:
    """PyTorch-default Adam convention implemented without PyTorch."""
    def __init__(self, shape, lr=ADAM_LR, beta1=.9, beta2=.999, eps=1e-8):
        self.m = np.zeros(shape, dtype=np.float64)
        self.v = np.zeros(shape, dtype=np.float64)
        self.step_count = 0
        self.lr, self.beta1, self.beta2, self.eps = lr, beta1, beta2, eps

    def update(self, z, gradient):
        if z.shape != self.m.shape or gradient.shape != z.shape:
            raise ValueError("Adam parameter/gradient shape mismatch")
        self.step_count += 1
        self.m *= self.beta1
        self.m += (1 - self.beta1) * gradient
        self.v *= self.beta2
        self.v += (1 - self.beta2) * gradient**2
        mhat = self.m / (1 - self.beta1**self.step_count)
        vhat = self.v / (1 - self.beta2**self.step_count)
        delta = -self.lr * mhat / (np.sqrt(vhat) + self.eps)
        z += delta
        if not np.isfinite(z).all():
            raise FloatingPointError("Nonfinite NumPy Adam update")
        return delta


def exposure_from_logits(logits, feature_indices):
    """Unmasked Chapter 3.8 feature-policy process; not the masked walk."""
    matrices = [logits] if logits.ndim == 2 else list(logits)
    noise = [i for i, old in enumerate(feature_indices) if old not in (0, 1, 3)]
    informative = [i for i, old in enumerate(feature_indices) if old in (0, 1, 3)]
    out = []
    for a in matrices:
        e = np.exp(a - a.max(axis=1, keepdims=True))
        p = e / e.sum(axis=1, keepdims=True)
        p0, transition = p[-1], p[:-1]
        dist = p0.copy()
        q = []
        for _ in range(3):
            q.append(float(dist[noise].sum()))
            dist = dist @ transition
        no_noise = (p0[informative]
                    @ transition[np.ix_(informative, informative)]
                    @ transition[np.ix_(informative, informative)]
                    @ np.ones(len(informative)))
        hit = float(1 - no_noise)
        uniform = 1 - (len(informative) / len(p0))**3
        out.append({
            "p0": p0.tolist(), "P": transition.tolist(),
            "q": q, "hit_at_least_once": hit,
            "expected_noise_visits": float(sum(q)),
            "relative_hit_to_uniform": hit / uniform if uniform > 0 else None
        })
    return out


def load_graph(run, data_dir):
    """Use only SHA-checked derived Train/Validation data."""
    data_dir = Path(data_dir)
    manifest = json.loads((data_dir / "manifest.json").read_text("utf8"))
    if (manifest.get("schema") != "E-CH4-14-100-derived-standard-seed-v1"
            or manifest.get("standard_data_seed") != 20261102
            or manifest.get("test_accessed") is not False):
        raise ValueError("Missing approved standard-seed provenance")
    ds = run["dataset_id"]
    record = next(r for r in manifest["datasets"] if r["dataset_id"] == ds)
    if record["feature_indices"] != run["feature_indices"]:
        raise ValueError("Derived subset mapping differs from frozen grid")
    source = data_dir / (ds + ".npz")
    if sha(source) != record["sha256"]:
        raise ValueError("Dataset content SHA differs from audited manifest")
    with np.load(source, allow_pickle=False) as f:
        if set(f.files) != {"train_x", "train_y", "validation_x",
                            "validation_y", "gamma"}:
            raise ValueError("An unapproved or Test field exists")
        tx, ty, vx, vy, gamma = [f[k].copy() for k in
             ("train_x", "train_y", "validation_x", "validation_y", "gamma")]
    d = len(run["feature_indices"])
    if (tx.shape != (300, d) or vx.shape != (90, d)
            or ty.shape != (300,) or vy.shape != (90,) or gamma.shape != (d,)):
        raise ValueError("Derived dataset shape mismatch")
    if (not np.array_equal(np.unique(ty), np.arange(3))
            or not np.array_equal(np.unique(vy), np.arange(3))):
        raise ValueError("Expected three Train/Validation classes")
    return ExactMaskedWalkNP(tx, ty, gamma, beta=BETA, lam=LAMBDA,
                             horizon=H, query_x=vx), vy


def prepare_row(graph, z, adam, run, step, vy, previous_update_l2=None):
    """Every metric, upstream gradient and optimizer state at z_step."""
    loss, gradient, reps, centers, scores = train_loss_and_grad(
        graph, z, run["family"], run["objective"])
    train = train_diagnostics(graph, reps, centers, scores)
    train["loss"] = float(loss)
    train["gradient_l2"] = float(np.linalg.norm(gradient.ravel()))
    validation = validation_diagnostics(graph, z, run["family"], centers, vy)
    return {
        "step": step, "run_id": run["run_id"],
        "train": train, "validation": validation,
        "policy": {
            "centers": centers.tolist(),
            "unmasked": exposure_from_logits(z, run["feature_indices"])
        },
        "optimizer": {
            "shape": list(z.shape),
            "logits": z.ravel().tolist(),
            "gradients": gradient.ravel().tolist(),
            "adam_m": adam.m.ravel().tolist(),
            "adam_v": adam.v.ravel().tolist(),
            "previous_update_l2": previous_update_l2,
            "adam_update_to_next": None,
            "adam_step_completed": adam.step_count
        }
    }, gradient


def authorization_gate(path, source_sha):
    """Requires manual authorization captured in separately audited workflow.

    This local preflight is *not* a substitute for an online manager approval
    and branch/SHA-locked CI and single-use batch budget.
    """
    if (os.environ.get("GITHUB_ACTIONS") != "true"
            or os.environ.get("GITHUB_REF_NAME") != BRANCH
            or os.environ.get("GITHUB_WORKFLOW") != WORKFLOW
            or os.environ.get("E14_100_AUTHORIZED") != "true"
            or os.environ.get("GITHUB_ACTOR") != "mahhammad"
            or os.environ.get("GITHUB_RUN_ATTEMPT") != "1"):
        raise PermissionError("No approved 152-fit GitHub Actions execution")
    if (len(source_sha) != 40 or
            any(x not in "0123456789abcdef" for x in source_sha)
            or os.environ.get("GITHUB_SHA") != source_sha):
        raise PermissionError("Execution SHA is not pinned")
    if path is None:
        raise PermissionError("Signed off manager authorization missing")
    record = json.loads(Path(path).read_text("utf8"))
    if (record.get("source_sha") != source_sha
            or record.get("planned_fits") != 152
            or record.get("updates_per_fit") != 100
            or record.get("test_accessed") is not False
            or not str(record.get("manager_approval_url", "")).startswith(
                "https://github.com/mahhammad/My-Thesis/issues/")):
        raise PermissionError("A separate owner-approved budget is necessary")
    return record


def execute_run(run, results_root, data_root, *, steps=UPDATES):
    """One irreversible run; no reuse of a partially written run directory."""
    graph, vy = load_graph(run, data_root)
    z = np.zeros(graph.shape(run["family"]), dtype=np.float64)
    adam = AdamNP(z.shape)
    output = Path(results_root) / run["run_id"]
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    previous = None
    trace = output / "trace.jsonl"
    with trace.open("x", encoding="utf8") as log:
        for step in range(steps + 1):
            row, gradient = prepare_row(graph, z, adam, run, step, vy, previous)
            row["elapsed_s"] = float(time.perf_counter() - started)
            if step < steps:
                change = adam.update(z, gradient)
                previous = float(np.linalg.norm(change.ravel()))
                row["optimizer"]["adam_update_to_next"] = change.ravel().tolist()
            log.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            log.flush()
            os.fsync(log.fileno())
    (output / "complete.json").write_text(json.dumps({
        "schema": "E-CH4-14-100-run-complete-v1", "run_id": run["run_id"],
        "status": "COMPLETE_UNAUDITED",
        "updates": steps, "trace_sha256": sha(trace),
        "test_accessed": False}, indent=2) + "\n", encoding="utf8")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--run-id")
    parser.add_argument("--family", choices=FAMILIES)
    parser.add_argument("--objective", choices=OBJECTIVES)
    parser.add_argument("--noise-count", type=int, choices=tuple(range(8)))
    parser.add_argument("--authorization", type=Path)
    parser.add_argument("--source-sha", default="")
    parser.add_argument("--data-dir", type=Path, default=DATA)
    parser.add_argument("--out", type=Path, default=RESULTS)
    args = parser.parse_args()
    design = make_design()
    runs = design["runs"]
    if args.run_id:
        if any(v is not None for v in (args.family, args.objective, args.noise_count)):
            raise ValueError("Use either --run-id or a family/objective/noise group")
        runs = [r for r in runs if r["run_id"] == args.run_id]
        if len(runs) != 1:
            raise ValueError("Unknown run id")
    else:
        if args.family is not None:
            runs = [r for r in runs if r["family"] == args.family]
        if args.objective is not None:
            runs = [r for r in runs if r["objective"] == args.objective]
        if args.noise_count is not None:
            runs = [r for r in runs if r["noise_count"] == args.noise_count]
    if not runs:
        raise ValueError("No run matches requested group")
    if not args.execute:
        print(json.dumps({"status": "DRY_RUN_NO_FIT",
            "backend": "NUMPY_ONLY_ANALYTIC_GRADIENT", "jobs": len(runs),
            "updates": UPDATES, "test_accessed": False}))
        return
    authorization_gate(args.authorization, args.source_sha)
    for i, run in enumerate(runs, 1):
        print(f"[{i}/{len(runs)}] {run['run_id']}", flush=True)
        execute_run(run, args.out, args.data_dir)


if __name__ == "__main__":
    main()
