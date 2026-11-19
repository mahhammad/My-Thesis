#!/usr/bin/env python3
"""Publication-only end-of-training sweeps over |A_N|=1,...,6.

All values come from frozen, audited E-CH4-14 101-step aggregate CSVs.
Four curves per panel (two policy families x two losses); each point is the
mean of six *dependent* noise subsets. Shaded ribbons show descriptive
minimum and maximum, not confidence intervals. No fitting or Test access.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import arabic_reshaper
from bidi.algorithm import get_display

plt.rcParams.update({"font.family": "DejaVu Sans", "axes.unicode_minus": False,
                     "mathtext.fontset": "dejavusans"})

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "figures" / "final-sweep"
FAMILIES = ("shared", "class")
LOSSES = ("sigmoid_JS_margin_tau005", "softplus_JS_tau005")
NOISE_LEVELS = tuple(range(1, 7))

# (metric, panel label, split). Use the thesis' canonical names.
FIGURES = {
    "decision_t100.png": (
        "Classification at update 100", "classification",
        (
            ("accuracy", "Train Accuracy", "train"),
            ("balanced_accuracy", "Train BA (= Macro Recall)", "train"),
            ("precision_macro", "Train Macro Precision", "train"),
            ("accuracy", "Validation Accuracy", "validation"),
            ("balanced_accuracy", "Validation BA (= Macro Recall)", "validation"),
            ("precision_macro", "Validation Macro Precision", "validation"),
        )),
    "geometry_primary_t100.png": (
        "Train geometry at update 100", "geometry",
        (
            ("separation_i", "I", "train"),
            ("density_r_kl", "R_KL", "train"),
            ("density_r_js", "R_JS", "train"),
            ("density_r_sqrt_js", "R_sqrt(JS)", "train"),
            ("margin_mean", "Margin", "train"),
        )),
    "geometry_other_t100.png": (
        "Train separation metrics at update 100", "geometry",
        (
            ("separation_t", "T", "train"),
            ("separation_bt", "BT", "train"),
            ("separation_tnl", "TNL", "train"),
            ("separation_btnl", "BTNL", "train"),
        )),
    "policy_exposure_t100.png": (
        "سیاست کمکی بدون ماسک: انتخاب و برخورد با نویز در گام ۱۰۰", "policy",
        (
            ("q1", ("احتمال انتخاب نویز در نوبت اول", r"$q_1$"), ""),
            ("q2", ("احتمال انتخاب نویز در نوبت دوم", r"$q_2$"), ""),
            ("q3", ("احتمال انتخاب نویز در نوبت سوم", r"$q_3$"), ""),
            ("hit2", ("حداقل یک برخورد در دو نوبت", r"$p_{\mathrm{hit},N}^{(2)}$"), ""),
            ("hit3", ("حداقل یک برخورد در سه نوبت", r"$p_{\mathrm{hit},N}^{(3)}$"), ""),
            ("nbar3", ("امید تعداد انتخاب نویز در سه نوبت", r"$\bar n_N^{(3)}$"), ""),
        )),
    "policy_normalized_t100.png": (
        "سیاست کمکی بدون ماسک: سهم نویز و نسبت به مرجع یکنواخت در گام ۱۰۰", "policy",
        (
            ("s2", ("سهم نویز در دو نوبت", r"$s_N^{(2)}$"), ""),
            ("s3", ("سهم نویز در سه نوبت", r"$s_N^{(3)}$"), ""),
            ("sp3", ("سهم ویژگی‌های اطلاع‌رسان", r"$s_P^{(3)}$"), ""),
            ("NFI3", ("نسبت احتمال برخورد به یکنواخت", r"$\mathrm{NFI}_N^{(3)}$"), ""),
            ("NSR2", ("نسبت سهم نویز در دو نوبت", r"$\mathrm{NSR}_N^{(2)}$"), ""),
            ("NSR3", ("نسبت سهم نویز در سه نوبت", r"$\mathrm{NSR}_N^{(3)}$"), ""),
        )),

}

STYLES = {
    ("shared", LOSSES[0]): ("سیاست مشترک، زیان سیگمویدی", "#2563eb", "-", "o"),
    ("shared", LOSSES[1]): ("سیاست مشترک، زیان سافت‌پلاس", "#d97706", "--", "s"),
    ("class", LOSSES[0]): ("سیاست وابسته به کلاس، زیان سیگمویدی", "#0f766e", "-.", "^"),
    ("class", LOSSES[1]): ("سیاست وابسته به کلاس، زیان سافت‌پلاس", "#a21caf", ":", "D"),
}


def fa(text):
    """Persian shaping for Matplotlib; mathtext fragments stay untouched."""
    return get_display(arabic_reshaper.reshape(str(text)))


def uniform_noise_reference(metric, noise):
    """Exact unmasked same-dimensional Uniform baseline; §3.8."""
    assert 1 <= noise <= 6
    d = 3 + noise
    p = noise / d
    if metric in ("q1", "q2", "q3", "s2", "s3"):
        return p
    if metric == "sp3":
        return 3 / d
    if metric == "hit2":
        return 1 - (3 / d)**2
    if metric == "hit3":
        return 1 - (3 / d)**3
    if metric == "nbar3":
        return 3 * p
    if metric in ("NFI3", "NSR2", "NSR3"):
        return 1.0
    raise KeyError(metric)



def read_summary(path: Path, kind: str):
    index = {}
    with path.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            assert row["family"] in FAMILIES and row["objective"] in LOSSES
            assert 1 <= int(row["noise_count"]) <= 6
            assert int(row["subset_count"]) == 6
            assert row.get("split", "") != "test"
            if int(row["step"]) != 100:
                continue
            split = row.get("split", "") if kind == "classification" else (
                row.get("split", "") if kind == "geometry" else "")
            key = (row["family"], row["objective"], int(row["noise_count"]),
                   split, row["metric"])
            assert key not in index, ("Duplicate t100 summary", key)
            low = float(row["minimum"] if kind != "policy" else row["min"])
            high = float(row["maximum"] if kind != "policy" else row["max"])
            mean = float(row["mean"])
            sd = float(row["descriptive_sd"])
            assert np.isfinite([low, mean, high, sd]).all()
            assert low - 1e-10 <= mean <= high + 1e-10
            index[key] = (mean, low, high, sd)
    return index


def plot_one(name, title, kind, panels, metrics, policies):
    source = policies if kind == "policy" else metrics
    columns = 2  # A4-readable panels; never squeeze six metrics into three columns.
    rows = (len(panels) + columns - 1) // columns
    fig, axes = plt.subplots(rows, columns, figsize=(13.0, 4.4 * rows),
                             squeeze=False)
    used = []
    for ax, (metric, title_panel, split) in zip(axes.flat, panels):
        all_values = []
        for family in FAMILIES:
            for loss in LOSSES:
                label, color, line, marker = STYLES[(family, loss)]
                vals = [source[family, loss, noise, split, metric]
                        for noise in NOISE_LEVELS]
                factor = 100.0 if kind == "classification" else 1.0
                y = np.array([v[0] for v in vals]) * factor
                lo = np.array([v[1] for v in vals]) * factor
                hi = np.array([v[2] for v in vals]) * factor
                all_values.extend(lo.tolist() + hi.tolist())
                artist, = ax.plot(NOISE_LEVELS, y, color=color,
                                  ls=line, marker=marker, ms=7, lw=2.6,
                                  label=label)
                ax.fill_between(NOISE_LEVELS, lo, hi, color=color, alpha=0.09,
                                linewidth=0)
                if ax is axes.flat[0]:
                    used.append(artist)
        if kind == "policy":
            uniform_values = np.array(
                [uniform_noise_reference(metric, n) for n in NOISE_LEVELS],
                dtype=float)
            assert np.isfinite(uniform_values).all()
            uniform_artist, = ax.plot(
                NOISE_LEVELS, uniform_values, color="#253349",
                linestyle=(0, (6, 3)), linewidth=2.2,
                label=fa("سیاست یکنواخت"), zorder=4)
            if ax is axes.flat[0]:
                used.append(uniform_artist)
            all_values.extend(uniform_values.tolist())
            description, formula = title_panel
            ax.set_title(fa(description) + "\n" + formula, fontsize=15, pad=10)
        else:
            ax.set_title(title_panel, fontsize=16, pad=10)
        ax.set_xlabel(r"$|A_N|$", fontsize=15)
        ax.set_xticks(NOISE_LEVELS)
        ax.tick_params(axis='both', labelsize=13)
        ax.grid(alpha=0.22, linewidth=0.65)
        if kind == "classification":
            ax.set_ylabel("%", fontsize=14)
            # Keep the agreed category-wide BA scale, not one scale per panel.
            ax.set_ylim(70, 100)
        else:
            low, high = min(all_values), max(all_values)
            span = max(high - low, max(abs(low), abs(high), 0.01) * 0.07)
            floor = max(0.0, low - 0.12 * span) if low >= 0 else low - 0.12 * span
            ax.set_ylim(floor, high + 0.14 * span)
    for ax in list(axes.flat)[len(panels):]:
        ax.axis("off")
    fig.suptitle(fa(title) if kind == "policy" else title,
                 fontsize=19, weight="semibold")
    labels = [fa(STYLES[k][0]) for k in STYLES]
    if kind == "policy":
        labels.append(fa("مرجع یکنواخت"))
    fig.legend(used, labels, loc="lower center",
               ncol=3 if kind == "policy" else 4,
               bbox_to_anchor=(0.5, 0.01), fontsize=12, frameon=False)
    fig.tight_layout(rect=(0, 0.115 if kind == "policy" else 0.085, 1, 0.94),
                     h_pad=2.4, w_pad=1.5)
    OUT.mkdir(parents=True, exist_ok=True)
    target = OUT / name
    fig.savefig(target, dpi=160, facecolor="white",
                metadata={"Software": "E-CH4-14 frozen t100 summaries"})
    plt.close(fig)
    assert target.stat().st_size > 10000, target
    return target


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true",
                        help="Write the five reproducible PNG charts")
    args = parser.parse_args()
    metrics = read_summary(ROOT / "all_metrics_101_steps.csv", "classification")
    policies = read_summary(ROOT / "noise-policy/noise_policy_101_steps.csv",
                            "policy")
    for name, (_title, kind, panels) in FIGURES.items():
        source = policies if kind == "policy" else metrics
        for metric, _title_panel, split in panels:
            for family in FAMILIES:
                for loss in LOSSES:
                    for noise in NOISE_LEVELS:
                        assert (family, loss, noise, split, metric) in source, (
                            "Missing final summary", kind, metric,
                            family, loss, noise)
    # Algebraic cross-checks of the macro means (without masked-sample walks).
    for family in FAMILIES:
        for loss in LOSSES:
            for noise in NOISE_LEVELS:
                def mean(metric):
                    return policies[family, loss, noise, "", metric][0]
                assert abs(3 * mean("s3") - mean("nbar3")) < 1e-8
                assert abs(mean("q1") + mean("q2") + mean("q3")
                           - mean("nbar3")) < 1e-8
                assert abs(mean("s3") + mean("sp3") - 1) < 1e-8
                assert abs(mean("NSR3") -
                           mean("s3") / (noise / (noise + 3))) < 1e-8
    for noise in NOISE_LEVELS:
        p = noise / (3 + noise)
        assert abs(uniform_noise_reference("nbar3", noise) - 3*p) < 1e-12
        assert abs(uniform_noise_reference("s3", noise) - p) < 1e-12
        assert abs(uniform_noise_reference("hit3", noise) -
                   (1 - (3 / (3 + noise))**3)) < 1e-12
        assert uniform_noise_reference("NFI3", noise) == 1.0
        assert uniform_noise_reference("NSR3", noise) == 1.0
    if args.write:
        generated = [str(plot_one(name, title, kind, panels, metrics, policies))
                     for name, (title, kind, panels) in FIGURES.items()]
        (OUT / "manifest.json").write_text(
            json.dumps({"source": "frozen E-CH4-14 t100 aggregates",
                        "no_fit": True, "test_accessed": False,
                        "noise_levels": list(NOISE_LEVELS),
                        "series_per_panel": 4,
                        "dependent_subsets_per_point": 6,
                        "figures": list(FIGURES),
                        "generated": generated}, indent=2),
            encoding="utf-8")
        print("E14_FINAL_NOISE_SWEEPS_PASS", len(generated), "FIGURES")
    else:
        print("E14_FINAL_NOISE_SWEEPS_DATA_PASS", len(FIGURES), "FIGURES")


if __name__ == "__main__":
    main()
