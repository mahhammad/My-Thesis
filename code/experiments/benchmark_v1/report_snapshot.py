"""Independent COMMON-metric reporter for the two-file IMC snapshot.

No training, no experiment-specific objectives, no historical Test split.
Experiments own and save their specialized diagnostics separately.
Train/Validation labels are regenerated and checked against fingerprints.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "code" / "data_generator"))
from generate import load_config
from backfill_saved_scores import dataset_train_val
from snapshot_pipeline import (array_fingerprint, config_fingerprint,
                               load_snapshot, _atomic_bytes)

LN2 = float(np.log(2.0))
REPORT_SCHEMA = "imc-report-v1"


def entropy(p):
    p = np.asarray(p, float)
    return -np.sum(np.where(p > 0, p * np.log(np.maximum(p, 1e-300)), 0.0), axis=-1)


def jsd(p, q):
    m = (p + q) / 2.0
    return np.maximum(0.0, entropy(m) - (entropy(p) + entropy(q)) / 2.0)


def kl_forward(p, q):
    """Exact forward KL: +inf if p>0 and q=0."""
    p, q = np.broadcast_arrays(np.asarray(p, float), np.asarray(q, float))
    bad = np.any((p > 0) & (q <= 0), axis=-1)
    terms = np.where(p > 0, p * (
        np.log(np.maximum(p, 1e-300)) - np.log(np.maximum(q, 1e-300))), 0.0)
    return np.where(bad, np.inf, terms.sum(axis=-1))


def logsumexp(a):
    maxval = np.max(a, axis=-1, keepdims=True)
    return maxval[..., 0] + np.log(np.exp(a - maxval).sum(axis=-1))


def class_mean(values, y, k):
    return float(np.mean([np.mean(values[y == c]) for c in range(k)]))


def maybe_number(value):
    return None if not np.isfinite(value) else float(value)


def check_labels(labels, n, k):
    y = np.asarray(labels, dtype=np.int64)
    if y.shape != (n,) or not np.all((y >= 0) & (y < k)):
        raise ValueError("Labels must be 0-based and agree with rows/classes")
    if not np.array_equal(np.unique(y), np.arange(k)):
        raise ValueError("Every class must have at least one labeled row")
    return y


def standard_metrics(by_hypothesis, centers, labels, tau, *, split, per_sample=True):
    """Calculate common metrics using only stored distributions and external labels."""
    if tau <= 0:
        raise ValueError("tau must be positive")
    v = np.asarray(by_hypothesis, dtype=np.float64)
    centers = np.asarray(centers, dtype=np.float64)
    n, k, k2 = v.shape
    if k != k2 or centers.shape != (k, k):
        raise ValueError("Prediction and center dimensions differ")
    y = check_labels(labels, n, k)
    rows = np.arange(n)
    # Score c uses the representation under candidate hypothesis c.
    scores = 1.0 - jsd(v, centers[None, :, :]) / LN2
    pred = scores.argmax(axis=1)
    true_score = scores[rows, y]
    rival = scores.copy()
    rival[rows, y] = -np.inf
    margin = true_score - rival.max(axis=1)
    tie = (np.isclose(scores, scores.max(axis=1, keepdims=True),
                     rtol=0, atol=1e-12).sum(axis=1) > 1)
    correct = (pred == y).astype(float)
    own = v[rows, y, :]
    true_centers = centers[y]
    r_js = jsd(own, true_centers)
    r_sqrt = np.sqrt(r_js)
    r_kl = kl_forward(own, true_centers)
    confusion = np.zeros((k, k), dtype=int)
    np.add.at(confusion, (y, pred), 1)
    pair_js = np.asarray([jsd(centers[c], centers[r])
                          for c in range(k) for r in range(c + 1, k)])
    nll_js = logsumexp(scores) - true_score
    sigmoid = np.exp(-np.logaddexp(0.0, margin / tau))
    softplus = tau * logsumexp((scores - true_score[:, None]) / tau)
    dkl = kl_forward(v, centers[None, :, :])
    if np.isfinite(dkl).all():
        norm = -(dkl - dkl.mean(axis=1, keepdims=True)) / np.maximum(
            dkl.std(axis=1, keepdims=True), 1e-14)
        nll_kl_norm = class_mean(logsumexp(norm) - norm[rows, y], y, k)
        nll_kl = class_mean(logsumexp(-dkl) + dkl[rows, y], y, k)
        unavailable_reason = None
    else:
        nll_kl_norm = nll_kl = None
        unavailable_reason = "Candidate KL infinite; no numerical smoothing used"
    per_class = []
    counts = np.bincount(y, minlength=k).astype(int)
    for c in range(k):
        hit = y == c
        per_class.append({
            "class": int(c), "count": int(counts[c]),
            "accuracy": float(correct[hit].mean()),
            "R_JS": float(r_js[hit].mean()),
            "R_sqrt_JS": float(r_sqrt[hit].mean()),
            "R_KL": maybe_number(r_kl[hit].mean()),
            "mean_margin": float(margin[hit].mean())
        })
    report = {
        "status": "ok", "split": split, "rows": int(n), "counts": counts.tolist(),
        "balanced_accuracy": class_mean(correct, y, k),
        "ordinary_accuracy": float(correct.mean()),
        "confusion_true_rows_pred_cols": confusion.tolist(),
        "tie_count": int(tie.sum()),
        "hard_margin_error_conservative": class_mean((margin <= 0).astype(float), y, k),
        "balanced_sigmoid_margin_loss": class_mean(sigmoid, y, k),
        "balanced_JS_similarity_NLL": class_mean(nll_js, y, k),
        "balanced_softplus_all_rivals": class_mean(softplus, y, k),
        "NLL_KL_simple": nll_kl,
        "NLL_KL_normalized_historical": nll_kl_norm,
        "NLL_KL_unavailable_reason": unavailable_reason,
        "margin_quantiles": {str(q): float(np.quantile(margin, q))
                             for q in (0., .1, .25, .5, .75, .9, 1.)},
        "R_JS": class_mean(r_js, y, k),
        "R_sqrt_JS": class_mean(r_sqrt, y, k),
        "R_KL": maybe_number(class_mean(r_kl, y, k)),
        "R_KL_infinite_count": int(np.isinf(r_kl).sum()),
        "R_KL_entropy_identity": (
            float(np.mean([entropy(centers[c]) - entropy(own[y == c]).mean()
                           for c in range(k)])) if split == "train" else None),
        "R_KL_entropy_identity_applicable": split == "train",
        "mean_true_class_similarity": 1.0 - class_mean(r_js, y, k) / LN2,
        "I": float(entropy(centers.mean(axis=0)) - entropy(centers).mean()),
        "min_pairwise_sqrt_JS": float(np.sqrt(pair_js.min())),
        "mean_pairwise_JS": float(pair_js.mean()),
        "per_class": per_class
    }
    if per_sample:
        report["per_sample"] = {
            "labels": y.tolist(), "predicted": pred.tolist(),
            "scores_all_candidates": scores.tolist(),
            "margins": margin.tolist(), "ties": tie.tolist(),
            "R_JS": r_js.tolist(), "R_sqrt_JS": r_sqrt.tolist(),
            "R_KL": [maybe_number(z) for z in r_kl]
        }
    return report


def load_evaluation_labels(model, arrays, *, config_override=None):
    """Reconstruct only Train and Validation, verify hashes before reporting."""
    info = model["data"]
    config_path = (Path(config_override) if config_override else
                   ROOT / info["config_path"])
    if config_fingerprint(config_path) != info["config_sha256"]:
        raise ValueError("Generator config differs from snapshot")
    spec = load_config(config_path)
    spec["seed"] = info["data_seed"]
    splits = dataset_train_val(spec)
    x, y = splits["train"]
    vx, vy = splits["validation"]
    ids = arrays["reference_ids"]
    if np.any(ids < 0) or np.any(ids >= len(x)):
        raise ValueError("Reference IDs out of bounds")
    if array_fingerprint(x[ids]) != info["train_reference_x_sha256"]:
        raise ValueError("Train reference features/order mismatch")
    if array_fingerprint(np.asarray(y[ids], dtype=np.int64)) != (
        info["train_reference_y_sha256"]
    ):
        raise ValueError("Train label fingerprint mismatch")
    if info["validation_x_sha256"] is not None and (
        array_fingerprint(vx) != info["validation_x_sha256"]
    ):
        raise ValueError("Validation feature fingerprint mismatch")
    if len(vx) != info["validation_count"]:
        raise ValueError("Validation row count mismatch")
    return np.asarray(y[ids], dtype=int), np.asarray(vy, dtype=int)


def report_snapshot(run_dir, *, config_override=None, per_sample=True):
    model, arrays = load_snapshot(run_dir)
    train_y, val_y = load_evaluation_labels(
        model, arrays, config_override=config_override
    )
    centers = arrays["centers"]
    # The stored centers must be EXACT Train class means, not arbitrary or
    # re-estimated from Validation (important for the KL entropy identity).
    by = arrays["train_by_hypothesis"]
    reference_means = np.stack([
        by[train_y == c, c, :].mean(axis=0)
        for c in range(len(centers))
    ])
    if not np.allclose(reference_means, centers, atol=1e-10, rtol=0):
        raise ValueError("Stored centers disagree with Train class means")
    tau = model["objective"]["tau"]
    train = standard_metrics(arrays["train_by_hypothesis"], centers, train_y,
                             tau, split="train", per_sample=per_sample)
    if model["prediction"]["validation_status"] == "available":
        val = standard_metrics(
            arrays["validation_by_hypothesis"], centers, val_y,
            tau, split="validation", per_sample=per_sample)
    else:
        val = {
            "status": "not_applicable",
            "reason": model["prediction"]["validation_status"],
            "R_JS": None, "R_sqrt_JS": None, "R_KL": None,
            "balanced_accuracy": None
        }
    return {
        "schema": REPORT_SCHEMA,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "source": {
            "schema": model["schema"], "family": model["family"],
            "objective": model["objective"],
            "data_seed": model["data"]["data_seed"],
            "reference_seed": model["data"]["reference_seed"],
            "model_json_sha256": sha256(
                (Path(run_dir) / "model.json").read_bytes()).hexdigest(),
            "distributions_sha256": model["distributions_sha256"],
            "historical_test_accessed": False
        },
        "centers": centers.tolist(), "train": train, "validation": val
    }


def report_markdown(report):
    def fmt(x):
        return "—" if x is None else f"{x:.6f}"
    t, v = report["train"], report["validation"]
    fields = [
        ("Balanced accuracy", "balanced_accuracy"),
        ("Accuracy", "ordinary_accuracy"), ("I", "I"),
        ("R_JS", "R_JS"), ("R_sqrt_JS", "R_sqrt_JS"), ("R_KL", "R_KL"),
        ("Sigmoid margin loss", "balanced_sigmoid_margin_loss"),
        ("JS-similarity NLL", "balanced_JS_similarity_NLL"),
        ("Multiclass Softplus", "balanced_softplus_all_rivals")
    ]
    lines = [
        "# گزارش استاندارد IMC", "",
        "خانواده سیاست: " + report["source"]["family"],
        "تابع هدف: " + report["source"]["objective"]["name"],
        "داده Test تاریخی خوانده نشده است.", "",
        "| معیار | Train | Validation |", "|---|---:|---:|"
    ]
    for label, key in fields:
        lines.append(f"| {label} | {fmt(t.get(key))} | "
                     f"{fmt(None if key == 'I' else v.get(key))} |")
    lines.extend(["", "## محدودیت‌ها", "",
                  "وضعیت Validation: " + v["status"],
                  "خطای حاشیه در تساوی محافظه‌کارانه است؛ argmax اندیس اول را انتخاب می‌کند.",
                  "NLL نرمال KL تاریخی با صورت پایدار eta/epsilon فصل سوم یکسان نیست.", ""])
    return "\n".join(lines)


def save_report(run_dir, report):
    """Derived report files; raw artifacts are never overwritten."""
    path = Path(run_dir)
    b = (json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False)
         + "\n").encode("utf8")
    _atomic_bytes(path / "report.json", b)
    _atomic_bytes(path / "report.md", report_markdown(report).encode("utf8"))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", required=True, type=Path)
    p.add_argument("--config-override", type=Path)
    p.add_argument("--no-per-sample", action="store_true")
    args = p.parse_args()
    report = report_snapshot(
        args.run_dir, config_override=args.config_override,
        per_sample=not args.no_per_sample)
    save_report(args.run_dir, report)
    print("Saved:", args.run_dir / "report.json", args.run_dir / "report.md")
    print("Train BA:", report["train"]["balanced_accuracy"],
          "Validation BA:", report["validation"].get("balanced_accuracy"))


if __name__ == "__main__":
    main()
