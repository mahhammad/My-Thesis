#!/usr/bin/env python3
"""Read-only analysis: accidental class separation by each candidate noise column.

Reconstructs Train/Validation columns from the frozen mother and protocol seeds.
No IMC fitting, no Test access, no use of k=7 as an IMC comparison level.
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from protocol import (
    load_frozen, PRIMARY, NOISE_POOL, NEW_NOISE_COUNT, NEW_NOISE_STD,
    NEW_NOISE_MISSING_RATE, SEED_NEW_VALUES, SEED_NEW_MASK
)

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "analysis" / "chance-separability"
N_PERM = 2000
PERM_SEED = 2026111917


def reconstruct():
    tx, ty, vx, vy, _, src_sha = load_frozen()
    value_rng = np.random.default_rng(SEED_NEW_VALUES)
    nt = value_rng.standard_normal((len(tx), NEW_NOISE_COUNT)) * NEW_NOISE_STD
    nv = value_rng.standard_normal((len(vx), NEW_NOISE_COUNT)) * NEW_NOISE_STD
    mask_rng = np.random.default_rng(SEED_NEW_MASK)
    nt[mask_rng.random(nt.shape) < NEW_NOISE_MISSING_RATE] = np.nan
    nv[mask_rng.random(nv.shape) < NEW_NOISE_MISSING_RATE] = np.nan
    tr, va = np.column_stack((tx, nt)), np.column_stack((vx, nv))
    assert tr.shape == (300, 10) and va.shape == (90, 10)
    assert np.array_equal(tr[:, :5], tx, equal_nan=True)
    assert np.array_equal(va[:, :5], vx, equal_nan=True)
    assert set(np.unique(ty)) == set(np.unique(vy)) == {0, 1, 2}
    return tr, ty, va, vy, src_sha


def eta_squared(values, labels):
    observed = np.isfinite(values)
    x, y = values[observed], labels[observed]
    if len(x) < 5 or len(np.unique(y)) < 2:
        return 0.0
    mean = x.mean()
    ss_total = np.square(x - mean).sum()
    if ss_total <= 1e-20:
        return 0.0
    ss_between = sum(np.count_nonzero(y == c) * (x[y == c].mean() - mean)**2
                     for c in np.unique(y))
    return float(np.clip(ss_between / ss_total, 0.0, 1.0))


def cohens_d(values, labels, c):
    obs = np.isfinite(values)
    a, b = values[obs & (labels == c)], values[obs & (labels != c)]
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    var = ((len(a)-1)*a.var(ddof=1)+(len(b)-1)*b.var(ddof=1)) / (len(a)+len(b)-2)
    return float((a.mean()-b.mean())/np.sqrt(var)) if var > 0 else 0.0


def auc(values, labels, c, sign):
    obs = np.isfinite(values)
    pos = sign*values[obs & (labels == c)]
    neg = sign*values[obs & (labels != c)]
    if len(pos) == 0 or len(neg) == 0:
        return None
    comparison = pos[:, None] - neg[None, :]
    return float((np.count_nonzero(comparison > 0) +
                  0.5*np.count_nonzero(comparison == 0))/comparison.size)


def correlation(a, b):
    valid = np.isfinite(a) & np.isfinite(b)
    if np.count_nonzero(valid) < 5:
        return None
    x, y = a[valid], b[valid]
    if x.std() <= 1e-12 or y.std() <= 1e-12:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def permutation_test(x, y, repeats, seed):
    """Shared label shuffle preserves inter-feature correlations and masks."""
    cols = [x[:, col] for col in NOISE_POOL]
    masks = [(~np.isfinite(c)).astype(float) for c in cols]
    observed = np.array([eta_squared(c, y) for c in cols])
    observed_mask = np.array([eta_squared(c, y) for c in masks])
    raw = np.zeros(len(cols), dtype=int)
    fwer = np.zeros(len(cols), dtype=int)
    mask_fwer = np.zeros(len(cols), dtype=int)
    rng = np.random.default_rng(seed)
    for _ in range(repeats):
        yy = rng.permutation(y)
        e = np.array([eta_squared(c, yy) for c in cols])
        m = np.array([eta_squared(c, yy) for c in masks])
        raw += e >= observed
        fwer += np.max(e) >= observed
        mask_fwer += np.max(m) >= observed_mask
    return {int(col): {
        "eta2": float(observed[j]),
        "p_raw": float((raw[j] + 1)/(repeats + 1)),
        "p_fwer7": float((fwer[j] + 1)/(repeats + 1)),
        "mask_eta2": float(observed_mask[j]),
        "mask_p_fwer7": float((mask_fwer[j] + 1)/(repeats + 1))
    } for j, col in enumerate(NOISE_POOL)}


def rank_diagnostic(x):
    complete = x[np.isfinite(x).all(axis=1)]
    if len(complete) < x.shape[1] + 2:
        return {"n_complete": int(len(complete)), "rank": None,
                "condition_number": None}
    scaled = (complete - complete.mean(axis=0)) / complete.std(axis=0)
    svals = np.linalg.svd(scaled, compute_uv=False)
    return {"n_complete": int(len(complete)),
            "rank": int(np.linalg.matrix_rank(scaled)),
            "condition_number": float(svals[0] / svals[-1])}


def analyse(repeats=N_PERM):
    tr, ty, va, vy, mother_sha = reconstruct()
    pt = permutation_test(tr, ty, repeats, PERM_SEED)
    pv = permutation_test(va, vy, repeats, PERM_SEED + 1)
    rows = []
    for col in NOISE_POOL:
        d = [cohens_d(tr[:, col], ty, c) for c in range(3)]
        target = int(np.nanargmax(np.abs(d)))
        sign = 1 if d[target] >= 0 else -1
        train_corr = [correlation(tr[:, col], tr[:, c]) for c in PRIMARY]
        val_corr = [correlation(va[:, col], va[:, c]) for c in PRIMARY]
        rows.append({
          "feature": int(col),
          "source": "original" if col in (2, 4) else "new",
          "n_train_observed": int(np.isfinite(tr[:, col]).sum()),
          "n_validation_observed": int(np.isfinite(va[:, col]).sum()),
          "eta2_train": pt[col]["eta2"],
          "p_train": pt[col]["p_raw"],
          "p_train_fwer7": pt[col]["p_fwer7"],
          "eta2_validation": pv[col]["eta2"],
          "p_validation": pv[col]["p_raw"],
          "p_validation_fwer7": pv[col]["p_fwer7"],
          "mask_eta2_train": pt[col]["mask_eta2"],
          "mask_p_train_fwer7": pt[col]["mask_p_fwer7"],
          "mask_eta2_validation": pv[col]["mask_eta2"],
          "mask_p_validation_fwer7": pv[col]["mask_p_fwer7"],
          "selected_class_train": target,
          "signed_d_train": float(d[target]),
          "auc_train": auc(tr[:, col], ty, target, sign),
          "auc_validation_same_class": auc(va[:, col], vy, target, sign),
          "max_abs_corr_informative_train":
              max(abs(v) for v in train_corr if v is not None),
          "max_abs_corr_informative_validation":
              max(abs(v) for v in val_corr if v is not None)
        })
    return {"schema": "e14-chance-separability-v1", "mother_sha256": mother_sha,
            "repetitions": repeats, "n_train": len(tr), "n_validation": len(va),
            "scope": "individual noise columns; no k=7 policy comparison",
            "no_fit": True, "test_accessed": False,
            "train_complete_case_rank": rank_diagnostic(tr),
            "noise2_noise4_corr_train": correlation(tr[:,2], tr[:,4]),
            "noise2_noise4_corr_validation": correlation(va[:,2], va[:,4]),
            "rows": rows}


def report(result):
    lines = [
      "# تفکیک تصادفی کلاس‌ها به کمک ستون نویزی منفرد",
      "",
      "تحلیل بدون آموزش IMC، فقط روی Train و Validation منجمد.",
      "هفت ستون نامزد نویز جداگانه بررسی شده‌اند؛ هیچ مقایسه‌ای",
      "برای پیکربندی هفت نویز در این گزارش صورت نمی‌گیرد.",
      "",
      "مقیاس eta2 پراکندگی بین‌کلاسی مقادیر مشاهده‌شده به کل",
      "پراکندگی است. p_fwer7 از بیشینهٔ آماره روی هفت ستون",
      "با جایگشت مشترک برچسب‌ها محاسبه شده و وابستگی بین",
      "نویزها و الگوی ماسک را ثابت می‌گذارد. آزمون ماسک جداست.",
      "بهترین کلاس one-vs-rest و جهت آن فقط روی Train انتخاب",
      "و AUC همان کلاس و همان جهت روی Validation ارزیابی شده است.",
      "این یک تحلیل اکتشافی روی تنها یک تحقق داده است.",
      "",
      "| نویز | eta2 Train | p_FWER7 Train | eta2 Val | p_FWER7 Val | کلاس | AUC Train | AUC Val |",
      "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in result["rows"]:
        lines.append(
            "| {feature} | {eta2_train:.4f} | {p_train_fwer7:.4f} |"
            " {eta2_validation:.4f} | {p_validation_fwer7:.4f} |"
            " {selected_class_train} | {auc_train:.4f} |"
            " {auc_validation_same_class:.4f} |".format(**r))
    lines += [
      "",
      "کنترل مکمل: ارتباط ماسک فقدان با برچسب و بیشینهٔ",
      "هم‌بستگی هر نویز با سه ستون اطلاع‌رسان در by_feature.csv ثبت شده است.",
      "",
      "هم‌بستگی نمونه‌ای، با وابستگی خطی جبری و با اطلاعات",
      "ذاتی کلاس در مدل مولد تفاوت دارد. اگر جداسازی ظاهری",
      "روی Train رخ دهد ولی روی Validationِ مستقل همان",
      "جهت حفظ نشود، تفسیر سازگار با نوسان نمونه‌گیری است؛",
      "حفظ اتفاقی در یک Validation کوچک هم استقلال مولدی",
      "نویز از برچسب را نقض نمی‌کند.",
      "",
      "توجه: روش تک‌ستونی به تنهایی تعامل‌های چندمتغیره را",
      "نمی‌سنجد و این گزارش علت ثابت‌ماندن BA را اثبات نمی‌کند.",
      ""
    ]
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--reps", type=int, default=N_PERM)
    p.add_argument("--out", type=Path, default=OUTPUT)
    a = p.parse_args()
    if a.reps < 100:
        raise ValueError("At least 100 permutations are required")
    result = analyse(a.reps)
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out/"results.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+"\n",
        encoding="utf-8")
    with (a.out/"by_feature.csv").open("w", newline="", encoding="utf-8") as stream:
        wr = csv.DictWriter(stream, fieldnames=list(result["rows"][0]))
        wr.writeheader()
        wr.writerows(result["rows"])
    (a.out/"README_FA.md").write_text(report(result), encoding="utf-8")
    print(json.dumps({
        "status": "NO_FIT_ANALYSIS_COMPLETE",
        "top_train_eta": sorted(
          ((r["feature"], r["eta2_train"]) for r in result["rows"]),
          key=lambda r:r[1], reverse=True)[:3],
        "fwer_significant_train": [r["feature"] for r in result["rows"]
                                  if r["p_train_fwer7"] < 0.05],
        "test_accessed": False}, ensure_ascii=False))


if __name__ == "__main__":
    main()
