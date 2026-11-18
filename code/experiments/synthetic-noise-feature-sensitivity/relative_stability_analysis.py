"""E-CH4-14 relative geometry and feature allocation, frozen step-60 only.

Source of geometry: tables/geometry_stability_by_run_60.json, itself built
from tables/final_60_canonical_reports.json; NEVER results/metrics (step 30).
BA total variation uses final_60_by_run.csv (0..60), NOT the older
stability_by_noise_level.csv (historical 0..30).
No training, Test, model inference or alteration of frozen inputs.
"""
from __future__ import annotations
import argparse
import csv
import io
import json
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
TABLE = HERE / "tables"
FAMILIES = ("shared", "class")
KEYS = ("mean_pairwise_pct_change", "mean_abs_pairwise_shift_pct_ref",
        "center_movement_pct_ref_spacing", "I_pct_change",
        "train_R_JS_pct_change", "validation_R_JS_pct_change",
        "validation_BA_delta_pp", "validation_prediction_flip_pct",
        "BA_TV_0_60_pct_change")


def loadcsv(filename):
    with (TABLE / filename).open(encoding="utf8", newline="") as f:
        return list(csv.DictReader(f))


def csvbytes(rows):
    s = io.StringIO(newline="")
    writer = csv.DictWriter(s, list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return s.getvalue().encode("utf8")


def avg(a):
    a = list(a)
    assert len(a) > 0
    return sum(a) / len(a)


def percent(a, reference):
    assert reference > 1e-12, "Undefined relative percentage; baseline is zero"
    return 100 * (a - reference) / reference


def derive():
    geometry = json.loads((TABLE / "geometry_stability_by_run_60.json").read_text("utf8"))
    assert geometry["n_runs"] == 36
    geo = {r["run_id"]: r for r in geometry["runs"]}
    final = {r["run_id"]: r for r in loadcsv("final_60_by_run.csv")}
    assert len(geo) == len(final) == 36
    rows = []
    for run in sorted(geo):
        g = geo[run]
        f = final[run]
        family = g["family"]
        k = g["noise_count"]
        assert family in FAMILIES and int(f["k"]) == k
        assert f["family"] == family
        baseid = "N0-01__" + family
        baseg = geo[baseid]
        basef = final[baseid]
        base_spacing = avg(baseg["center_geometry"]["center_pairwise_sqrt_JSD"])
        now_spacing = avg(g["center_geometry"]["center_pairwise_sqrt_JSD"])
        assert base_spacing > 0 and float(basef["train_I"]) > 0
        assert float(basef["train_R_JS"]) > 0 and float(basef["validation_R_JS"]) > 0
        assert float(basef["ba_tv_0_60_pp"]) > 0
        assert abs((float(f["train_I"]) - float(basef["train_I"])) -
                   float(g["center_geometry"]["delta_I"])) < 1e-9
        assert abs((float(f["validation_R_JS"]) - float(basef["validation_R_JS"])) -
                   float(g["validation"]["R_JS_delta"])) < 1e-9
        rows.append({
            "run_id": run, "family": family, "noise_count": k,
            "reference_run_id": baseid,
            "reference_mean_pairwise_sqrt_JSD": base_spacing,
            "mean_pairwise_pct_change": percent(now_spacing, base_spacing),
            "mean_abs_pairwise_shift_pct_ref":
                100 * g["center_geometry"]["mean_absolute_pairwise_distance_shift"] / base_spacing,
            # This is NOT a percent change in center coordinates: no natural origin.
            "center_movement_pct_ref_spacing":
                100 * g["center_geometry"]["mean_center_drift"] / base_spacing,
            "I_pct_change": percent(float(f["train_I"]), float(basef["train_I"])),
            "train_R_JS_pct_change":
                percent(float(f["train_R_JS"]), float(basef["train_R_JS"])),
            "validation_R_JS_pct_change":
                percent(float(f["validation_R_JS"]), float(basef["validation_R_JS"])),
            "validation_BA_delta_pp":
                100 * (float(f["validation_BA"]) - float(basef["validation_BA"])),
            "validation_prediction_flip_pct":
                100 * float(g["validation"]["prediction_flip_rate"]),
            "BA_TV_0_60_pct_change":
                percent(float(f["ba_tv_0_60_pp"]), float(basef["ba_tv_0_60_pp"]))
        })
    assert len(rows) == 36
    grouped = []
    for family in FAMILIES:
        for k in range(6):
            sel = [r for r in rows if r["family"] == family and r["noise_count"] == k]
            assert len(sel) == (1 if k in (0, 5) else 4)
            out = {"family": family, "noise_count": k, "n_dependent_subsets": len(sel)}
            for name in KEYS:
                values = [float(r[name]) for r in sel]
                out[name + "_mean"] = avg(values)
                out[name + "_min"] = min(values)
                out[name + "_max"] = max(values)
            grouped.append(out)
    assert all(abs(r[key]) < 1e-9 for r in grouped if r["noise_count"] == 0
               for key in KEYS)

    prefs = loadcsv("feature_preference_by_noise_level_60.csv")
    buffer = loadcsv("softmax_buffer_by_noise_level_60.csv")
    assert len(prefs) == 36 and len(buffer) == 12
    pref_index = {(r["family"], int(r["noise_count"]), int(r["h"])): r for r in prefs}
    buffer_index = {(r["family"], int(r["noise_count"])): r for r in buffer}
    byfamily = {}
    for family in FAMILIES:
        ref = buffer_index[(family, 0)]
        byfamily[family] = float(ref["gradient_rms_30_60_mean"])
        assert byfamily[family] > 0
    allocation = []
    for family in FAMILIES:
        for k in range(6):
            pref = [pref_index[(family,k,h)] for h in (1,2,3)]
            buff = buffer_index[(family,k)]
            hshares = [100*float(p["main_share_mean"]) for p in pref]
            noisy = 100 - hshares[-1]
            noise_uniform = 100*k/(k+3)
            assert all(-1e-8 <= x <= 100+1e-8 for x in hshares)
            assert abs(float(buff["main_share_h3_60_mean"])*100 - hshares[2]) < 1e-8
            allocation.append({
                "family": family, "noise_count": k,
                "n_dependent_subsets": 1 if k in (0,5) else 4,
                "informative_share_h1_pct": hshares[0],
                "informative_share_h2_pct": hshares[1],
                "informative_share_h3_pct": hshares[2],
                "noise_share_h3_pct": noisy,
                "uniform_noise_share_pct": noise_uniform,
                "noise_vs_uniform_pp": noisy - noise_uniform,
                "informative_conditional_TV_h3_pct":
                    100 * float(pref[2]["main_conditional_TV_mean"]),
                "mean_rank_reversals_h3":
                    float(pref[2]["rank_reversal_count_mean"]),
                "gradient_RMS_30_60_pct_change":
                    percent(float(buff["gradient_rms_30_60_mean"]), byfamily[family]),
            })
    assert len(allocation) == 12
    assert all(abs(r["informative_share_h3_pct"]-100) < 1e-9
               for r in allocation if r["noise_count"] == 0)
    return rows, grouped, allocation


def artifacts():
    runs, groups, allocation = derive()
    return {
        TABLE / "relative_geometry_by_run_60.csv": csvbytes(runs),
        TABLE / "relative_geometry_by_noise_level_60.csv": csvbytes(groups),
        TABLE / "feature_allocation_percent_by_level_60.csv": csvbytes(allocation)
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("write","check"), default="check")
    arg = parser.parse_args()
    for path, data in artifacts().items():
        if arg.mode == "write":
            path.write_bytes(data)
        else:
            assert path.is_file() and path.read_bytes()==data, f"Stale or absent: {path}"
    print("E14_RELATIVE_STABILITY_PASS")


if __name__ == "__main__":
    main()
