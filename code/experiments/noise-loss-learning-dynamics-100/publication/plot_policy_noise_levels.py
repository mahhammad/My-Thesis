#!/usr/bin/env python3
"""Deterministic, NO-FIT t=100 policy diagnostic panels for E-CH4-14.

Reads frozen aggregate data (six dependent subsets per setting).
Writes TikZ/PGFPlots source; x-axis is |A_N|=1,...,6.
"""
from __future__ import annotations
import argparse
import csv
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "noise-policy" / "noise_policy_101_steps.csv"
DEST = ROOT / "latex" / "noise-level-plots"
OBJ = ("sigmoid_JS_margin_tau005", "softplus_JS_tau005")
SERIES = (
    ("shared", OBJ[0], "blue!75!black", "solid", "*", "shared / Sigmoid--JS"),
    ("shared", OBJ[1], "orange!85!black", "solid", "diamond*", "shared / Softplus--JS"),
    ("class", OBJ[0], "blue!75!black", "dashed", "square*", "class / Sigmoid--JS"),
    ("class", OBJ[1], "orange!85!black", "dashed", "triangle*", "class / Softplus--JS"),
)
CHARTS = (
    ("policy_selection_t100.tex", (
        ("q1", r"$q_1$"), ("q2", r"$q_2$"),
        ("q3", r"$q_3$"), ("s3", r"$s_N^{(3)}$"))),
    ("policy_exposure_t100.tex", (
        ("hit2", r"$p_{\mathrm{hit},N}^{(2)}$"),
        ("hit3", r"$p_{\mathrm{hit},N}^{(3)}$"),
        ("nbar3", r"$\bar n_N^{(3)}$"),
        ("s2", r"$s_N^{(2)}$"))),
    ("policy_normalized_t100.tex", (
        ("sp3", r"$s_P^{(3)}$"),
        ("NFI3", r"$\mathrm{NFI}_N^{(3)}$"),
        ("NSR2", r"$\mathrm{NSR}_N^{(2)}$"),
        ("NSR3", r"$\mathrm{NSR}_N^{(3)}$"))),
)

def load_final():
    metrics = {m for _, fields in CHARTS for m, _ in fields}
    data = {}
    with SOURCE.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if int(row["step"]) != 100 or row["metric"] not in metrics:
                continue
            key = (row["family"], row["objective"], int(row["noise_count"]), row["metric"])
            assert key not in data and int(row["subset_count"]) == 6, key
            value = float(row["mean"])
            assert math.isfinite(value), key
            data[key] = value
    assert len(data) == len(metrics) * len(SERIES) * 6, (len(data), len(metrics))
    return data

def render(data, filename, items):
    lines = [
        "% Generated from noise_policy_101_steps.csv at step=100; NO FIT; NO TEST.",
        "% Each curve is a mean of six DEPENDENT subsets per |A_N|.",
        r"\begin{tikzpicture}",
        r"\begin{groupplot}[group style={group size=2 by 2, horizontal sep=1.0cm, vertical sep=1.1cm},",
        r"width=0.43\linewidth, height=0.245\textheight,",
        r"xmin=1, xmax=6, xtick={1,2,3,4,5,6},",
        r"xlabel={$|A_N|$}, tick label style={font=\scriptsize},",
        r"label style={font=\scriptsize}, title style={font=\small},",
        r"grid=major, grid style={gray!18},",
        r"legend style={draw=none, font=\tiny, fill opacity=0.9, text opacity=1},",
        r"legend cell align=left]",
    ]
    for j, (metric, title) in enumerate(items):
        lines.append(r"\nextgroupplot[title={" + title + r"}]")
        if metric in ("NFI3", "NSR2", "NSR3"):
            lines.append(r"\addplot[gray!60, densely dotted, thin, forget plot] coordinates {(1,1) (6,1)};")
        for family, objective, color, style, marker, label in SERIES:
            coords = " ".join(f"({k},{data[(family, objective, k, metric)]:.6f})" for k in range(1, 7))
            lines.append(r"\addplot+[color=" + color + ", " + style + ", mark=" + marker
                         + r", mark size=1.3pt, thick] coordinates {" + coords + "};")
            if j == 0:
                lines.append(r"\addlegendentry{\lr{" + label + "}}")
    lines += [r"\end{groupplot}", r"\end{tikzpicture}", ""]
    return "\n".join(lines)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    data = load_final()
    for filename, items in CHARTS:
        rendered = render(data, filename, items)
        path = DEST / filename
        if args.write:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(rendered, encoding="utf-8")
            print("WROTE", path, len(rendered))
        else:
            assert path.is_file() and path.read_text(encoding="utf-8") == rendered, (
                "Stale t100 noise-level chart: " + str(path))
    print("E14_POLICY_LEVEL_CHARTS_PASS: 12 metrics, 6 levels, 4 curves each")

if __name__ == "__main__":
    main()
