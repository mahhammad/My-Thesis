"""E-CH4-14 controlled 100-update comparison: fixed design and paired data.

No training is performed here. Only frozen Train/Validation inputs are read.
The standard five-column seed-20261102 mother is SHA-verified. The prior
E14 three added noise columns are kept byte-identical. Two more noise
columns bring the total to five newly added and seven noise columns overall.
"""
from __future__ import annotations
from collections import Counter
from hashlib import sha256
import itertools
import json
from pathlib import Path
import random

PRIMARY = (0, 1, 3)
NOISE_POOL = (2, 4, 5, 6, 7, 8, 9)
OBJECTIVES = ("sigmoid_JS_margin_tau005", "softplus_JS_tau005")
FAMILIES = ("shared", "class")
SEED_DESIGN = 20261119
SEED_NEW_VALUES = 2026111901
SEED_NEW_MASK = 2026111902
UPDATES = 100
TAU = 0.005
ADAM_LR = 0.12
BETA, LAMBDA, H = 0.4, 0.8, 12
ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
PREVIOUS = ROOT / "code/experiments/synthetic-noise-feature-sensitivity"
STANDARD = ROOT / "code/experiments/objective-start-diagnostics/inputs/phase2_train_validation"
STANDARD_FILE = STANDARD / "ds20261102_train_val.npz"
STANDARD_MANIFEST = STANDARD / "manifest.json"
STANDARD_SEED = 20261102
PREVIOUS_FILE = PREVIOUS / "inputs/derived/N5-01.npz"
PREVIOUS_MANIFEST = PREVIOUS / "inputs/derived/manifest.json"
CONFIG = ROOT / "code/data_generator/config.json"


def sha(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def choose_six(noise_count):
    """Deterministic, distinct subsets with approximately balanced column use.

    This is a design heuristic, not an independent-seed sampling procedure.
    """
    if not 1 <= noise_count <= 6:
        raise ValueError("Six subsets required only for levels 1..6")
    rng = random.Random(SEED_DESIGN + noise_count)
    candidates = list(itertools.combinations(NOISE_POOL, noise_count))
    rng.shuffle(candidates)
    selected = []
    exposure = Counter()
    pair_exposure = Counter()
    for _ in range(6):
        target = (_ + 1) * noise_count / len(NOISE_POOL)
        def quality(combo):
            counts = [exposure[col] + (col in combo) for col in NOISE_POOL]
            balance = sum((c-target)**2 for c in counts)
            pairs = tuple(itertools.combinations(combo, 2))
            pair_cost = sum(pair_exposure[p] for p in pairs)
            overlap = sum(len(set(combo) & set(prev))**2 for prev in selected)
            return (balance, pair_cost, overlap)
        choice = min((c for c in candidates if c not in selected), key=quality)
        selected.append(choice)
        exposure.update(choice)
        pair_exposure.update(itertools.combinations(choice, 2))
    return sorted(selected)


def make_design():
    datasets = []
    for noise_count in range(8):
        subsets = ([()] if noise_count == 0
                   else [NOISE_POOL] if noise_count == 7
                   else choose_six(noise_count))
        for j, noise in enumerate(subsets, 1):
            datasets.append({
                "dataset_id": f"N{noise_count}-{j:02d}",
                "noise_count": noise_count,
                "noise_indices": list(noise),
                "feature_indices": sorted(PRIMARY + tuple(noise)),
                "subset_index": j,
                "subset_count": len(subsets),
                "comparison": 1 <= noise_count <= 6,
            })
    runs = []
    for ds in datasets:
        for objective in OBJECTIVES:
            for family in FAMILIES:
                runs.append({**ds, "objective": objective, "family": family,
                    "run_id": f"{ds['dataset_id']}__{objective}__{family}"})
    if (len(datasets), len(runs), sum(r["comparison"] for r in runs)) != (38, 152, 144):
        raise AssertionError("Frozen grid counts drifted")
    if len({r["run_id"] for r in runs}) != 152:
        raise AssertionError("Nonunique run")
    return {
        "schema": "E-CH4-14-100-dual-loss-design-v1",
        "primary_indices": list(PRIMARY), "noise_pool_indices": list(NOISE_POOL),
        "objectives": list(OBJECTIVES), "families": list(FAMILIES),
        "steps": UPDATES, "learning_rate": ADAM_LR,
        "beta": BETA, "lambda": LAMBDA, "H": H, "tau": TAU,
        "seeds": {"design": SEED_DESIGN, "new_values": SEED_NEW_VALUES,
                  "new_mask": SEED_NEW_MASK},
        "test_accessed": False, "independent_data_seed_count": 1,
        "subsets_are_dependent": True,
        "datasets": datasets, "runs": runs,
    }


def fingerprint(array):
    """Identical to the official frozen Train/Validation fingerprint."""
    import numpy as np
    a = np.ascontiguousarray(np.asarray(array))
    head = json.dumps({"dtype": a.dtype.str, "shape": list(a.shape)},
                      sort_keys=True).encode("utf8")
    return sha256(head + b"\0" + a.tobytes()).hexdigest()


def load_frozen():
    """Use standard seed-20261102 mother plus three authenticated previous noises.

    Return an eight-column base that is *proved* to be the standard five
    columns followed by exactly the prior three added columns. Neither the
    mother nor the prior three noises is regenerated.
    """
    import numpy as np
    m = json.loads(STANDARD_MANIFEST.read_text("utf8"))
    if (m.get("schema") != "e-ch4-13-phase2-frozen-train-validation-v1"
            or m.get("train_validation_only") is not True
            or m.get("test_accessed") is not False
            or STANDARD_SEED not in m.get("seed_set", [])):
        raise ValueError("Unexpected standard mother manifest")
    if sha(CONFIG) != m["config_sha256"]:
        raise ValueError("Frozen generator config SHA mismatch")
    rec = m["records"][str(STANDARD_SEED)]
    if (rec.get("file") != STANDARD_FILE.name
            or rec.get("test_accessed") is not False
            or sha(STANDARD_FILE) != rec["file_sha256"]):
        raise ValueError("Standard mother file SHA mismatch")
    with np.load(STANDARD_FILE, allow_pickle=False) as f:
        if set(f.files) != {"train_x", "train_y", "validation_x", "validation_y"}:
            raise ValueError("Unexpected keys in standard mother")
        tx, ty, vx, vy = (f[k].copy() for k in
                          ("train_x", "train_y", "validation_x", "validation_y"))
    if (tx.shape != (300, 5) or vx.shape != (90, 5)
            or ty.shape != (300,) or vy.shape != (90,)
            or tx.dtype != np.float64 or vx.dtype != np.float64
            or ty.dtype != np.int64 or vy.dtype != np.int64):
        raise ValueError("Standard mother shape/dtype mismatch")
    checks = {"train_x_sha256": fingerprint(tx),
              "train_y_sha256": fingerprint(ty),
              "validation_x_sha256": fingerprint(vx),
              "validation_y_sha256": fingerprint(vy),
              "mask_train_sha256": fingerprint(np.isfinite(tx)),
              "mask_validation_sha256": fingerprint(np.isfinite(vx))}
    if any(checks[k] != rec[k] for k in checks):
        raise ValueError("Standard mother array/mask SHA mismatch")
    config_gamma = np.asarray(
        json.loads(CONFIG.read_text("utf8"))["kernel_gamma"], dtype=np.float64)
    if config_gamma.shape != (5,) or not np.isfinite(config_gamma).all():
        raise ValueError("Invalid standard gamma")

    prev = json.loads(PREVIOUS_MANIFEST.read_text("utf8"))
    if prev.get("test_accessed") is not False:
        raise ValueError("Previous experiment Test evidence invalid")
    old = next(d for d in prev["datasets"] if d["subset_id"] == "N5-01")
    if (old["feature_mapping"] != list(range(8))
            or old["hashes"]["source_mother_sha256"] != rec["file_sha256"]
            or sha(PREVIOUS_FILE) != old["file_sha256"]):
        raise ValueError("Prior three added noises are not authenticated")
    with np.load(PREVIOUS_FILE, allow_pickle=False) as f:
        if set(f.files) != {"train_x", "train_y", "validation_x", "validation_y"}:
            raise ValueError("Unexpected old experiment input keys")
        tx8, ty8, vx8, vy8 = (f[k].copy() for k in
                              ("train_x", "train_y", "validation_x", "validation_y"))
    if (tx8.shape != (300, 8) or vx8.shape != (90, 8)
            or not np.array_equal(tx, tx8[:, :5], equal_nan=True)
            or not np.array_equal(vx, vx8[:, :5], equal_nan=True)
            or not np.array_equal(ty, ty8)
            or not np.array_equal(vy, vy8)):
        raise ValueError("The old eight-column data do not preserve the standard mother")
    old_gamma = np.asarray(old["gamma"], dtype=np.float64)
    if (old_gamma.shape != (8,)
            or not np.array_equal(config_gamma, old_gamma[:5])
            or not np.array_equal(old_gamma[5:],
                                  np.full(3, 0.6, dtype=np.float64))):
        raise ValueError("Old kernels do not match the standard and three reused noises")
    return (tx8, ty, vx8, vy, old_gamma,
            {"standard_mother_sha256": rec["file_sha256"],
             "prior_e14_eight_column_sha256": old["file_sha256"],
             "reused_old_noise_raw_seed": old["hashes"]["raw_seed"],
             "reused_old_noise_mask_seed": old["hashes"]["mask_seed"]})


def generate_datasets(output_dir):
    """Generate five *additional* noises in total: reuse three, create two.

    No Test access, no model fit. The old standardized five columns and
    previously generated three noise columns remain unchanged.
    """
    import numpy as np
    output_dir = Path(output_dir)
    design = make_design()
    tx8, ty, vx8, vy, old_gamma, lineage = load_frozen()
    rng = np.random.default_rng(SEED_NEW_VALUES)
    new_train = rng.standard_normal((len(tx8), 2)) * 1.1
    new_val = rng.standard_normal((len(vx8), 2)) * 1.1
    mask_rng = np.random.default_rng(SEED_NEW_MASK)
    new_train[mask_rng.random(new_train.shape) < .20] = np.nan
    new_val[mask_rng.random(new_val.shape) < .20] = np.nan
    full_train = np.column_stack((tx8, new_train))
    full_val = np.column_stack((vx8, new_val))
    gamma = np.concatenate((old_gamma, np.array([.6, .6], dtype=np.float64)))
    if (full_train.shape != (300, 10) or full_val.shape != (90, 10)
            or len(gamma) != 10):
        raise AssertionError("Expected three informative plus seven noise columns")
    if output_dir.exists():
        raise FileExistsError("Refusing to overwrite existing derived input directory")
    output_dir.mkdir(parents=True, exist_ok=False)
    manifest = {"schema": "E-CH4-14-100-derived-v2",
                "test_accessed": False, "standard_seed": STANDARD_SEED,
                **lineage,
                "new_values_seed": SEED_NEW_VALUES, "new_mask_seed": SEED_NEW_MASK,
                "noise_columns_in_standard_mother": [2, 4],
                "reused_prior_added_noise_columns": [5, 6, 7],
                "newly_generated_noise_columns": [8, 9],
                "n_train": 300, "n_validation": 90, "datasets": []}
    for ds in design["datasets"]:
        selected = ds["feature_indices"]
        dest = output_dir / (ds["dataset_id"] + ".npz")
        with dest.open("xb") as f:
            np.savez_compressed(f, train_x=full_train[:, selected],
                                train_y=ty, validation_x=full_val[:, selected],
                                validation_y=vy, gamma=gamma[selected])
        manifest["datasets"].append({**ds, "sha256": sha(dest)})
    with (output_dir / "manifest.json").open("x", encoding="utf8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
        f.write("\n")
    return manifest


def main():
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--write-data", action="store_true")
    p.add_argument("--input-dir", type=Path, default=HERE / "inputs/derived")
    p.add_argument("--out-design", type=Path, default=HERE / "inputs/design.json")
    a = p.parse_args()
    design = make_design()
    if not a.write_data:
        print(json.dumps({"status": "DRY_RUN", "datasets": len(design["datasets"]),
            "runs": len(design["runs"]), "training_started": False}, ensure_ascii=False))
        return
    if a.out_design.exists():
        raise FileExistsError("Refusing to replace frozen design")
    a.out_design.parent.mkdir(parents=True, exist_ok=True)
    generate_datasets(a.input_dir)
    a.out_design.write_text(json.dumps(design, indent=2) + "\n", encoding="utf8")
    print("Prepared 38 paired Train/Validation datasets, no training and no Test")


if __name__ == "__main__":
    main()
