"""Lightweight, specialized sigmoid diagnostics. Does NOT compute common metrics.

Run from the repository root. Reads archived CSV only; never loads Test.
"""
import csv
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
ARCHIVE = ROOT / "code/experiments/archive/sigmoid_margin_diagnostics/results/sigmoid_validation_summary.csv"

def rows():
    with ARCHIVE.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))

def sigmoid_loss(margin, tau):
    if tau <= 0:
        raise ValueError("tau must be positive")
    z = margin / tau
    if z >= 0:
        a = math.exp(-z)
        return a / (1+a)
    return 1 / (1+math.exp(z))

def normalized_sensitivity(margin, tau):
    p = sigmoid_loss(margin, tau)
    return 4*p*(1-p)

def compare_rounds(rows_):
    out = {}
    for tau in sorted({float(r["tau"]) for r in rows_}):
        r0 = next(r for r in rows_ if r["case"]=="guided_round0" and float(r["tau"])==tau)
        r4 = next(r for r in rows_ if r["case"]=="guided_round4" and float(r["tau"])==tau)
        out[tau] = {"guided_round4_lower_loss":float(r4["balanced_sigmoid_loss"])<float(r0["balanced_sigmoid_loss"]),
                    "delta_round4_minus_round0":float(r4["balanced_sigmoid_loss"])-float(r0["balanced_sigmoid_loss"])}
    return out

if __name__ == "__main__":
    for tau, item in compare_rounds(rows()).items():
        print(f"tau={tau:.3f}: delta round4-round0={item['delta_round4_minus_round0']:+.8f}")
