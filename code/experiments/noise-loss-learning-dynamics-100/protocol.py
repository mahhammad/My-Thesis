"""E-CH4-14 controlled 100-update comparison: fixed design and paired data.

No training is performed here. Only frozen Train/Validation inputs are read.
The standard five-column seed-20261102 mother is SHA-verified.
Five new noise columns are generated independently here; no historical E14
noise assets or results are reused or compared.
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
STANDARD = ROOT / "code/experiments/objective-start-diagnostics/inputs/phase2_train_validation"
STANDARD_FILE = STANDARD / "ds20261102_train_val.npz"
STANDARD_MANIFEST = STANDARD / "manifest.json"
STANDARD_SEED = 20261102
NEW_NOISE_COUNT = 5
NEW_NOISE_STD = 1.1
NEW_NOISE_MISSING_RATE = 0.20
NEW_NOISE_GAMMA = 0.6
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


def fingerprint(array):
    """Use the standard frozen source's array-fingerprint contract."""
    import numpy as np
    a = np.ascontiguousarray(np.asarray(array))
    header = json.dumps({"dtype": a.dtype.str, "shape": list(a.shape)},
                        sort_keys=True).encode("utf8")
    return sha256(header + b"\0" + a.tobytes()).hexdigest()


def load_frozen():
    """SHA-verified standard five-column data; no historical E14 assets."""
    import numpy as np
    manifest = json.loads(STANDARD_MANIFEST.read_text("utf8"))
    if (manifest.get("schema") != "e-ch4-13-phase2-frozen-train-validation-v1"
            or manifest.get("train_validation_only") is not True
            or manifest.get("test_accessed") is not False
            or STANDARD_SEED not in manifest.get("seed_set", [])):
        raise ValueError("Unexpected standard mother manifest")
    if sha(CONFIG) != manifest["config_sha256"]:
        raise ValueError("Standard frozen generator configuration changed")
    rec = manifest["records"][str(STANDARD_SEED)]
    if (rec.get("file") != STANDARD_FILE.name
            or rec.get("test_accessed") is not False
            or sha(STANDARD_FILE) != rec["file_sha256"]):
        raise ValueError("Standard mother file SHA mismatch")
    with np.load(STANDARD_FILE, allow_pickle=False) as f:
        if set(f.files) != {"train_x", "train_y", "validation_x", "validation_y"}:
            raise ValueError("Only Train/Validation source fields are allowed")
        tx, ty, vx, vy = (f[key].copy() for key in
                          ("train_x", "train_y", "validation_x", "validation_y"))
    if (tx.shape != (300, 5) or vx.shape != (90, 5)
            or ty.shape != (300,) or vy.shape != (90,)
            or tx.dtype != np.float64 or vx.dtype != np.float64
            or ty.dtype != np.int64 or vy.dtype != np.int64):
        raise ValueError("Standard mother shape/dtype changed")
    checks = {
        "train_x_sha256": fingerprint(tx),
        "train_y_sha256": fingerprint(ty),
        "validation_x_sha256": fingerprint(vx),
        "validation_y_sha256": fingerprint(vy),
        "mask_train_sha256": fingerprint(np.isfinite(tx)),
        "mask_validation_sha256": fingerprint(np.isfinite(vx)),
    }
    if any(checks[key] != rec[key] for key in checks):
        raise ValueError("Standard mother source or mask fingerprint mismatch")
    if (np.isinf(tx).any() or np.isinf(vx).any()
            or not np.array_equal(np.unique(ty), np.arange(3))
            or not np.array_equal(np.unique(vy), np.arange(3))):
        raise ValueError("Invalid source observations/classes")
    gammas = np.asarray(json.loads(CONFIG.read_text("utf8"))["kernel_gamma"],
                        dtype=np.float64)
    if gammas.shape != (5,) or not np.isfinite(gammas).all():
        raise ValueError("Invalid standard kernel gammas")
    return tx, ty, vx, vy, gammas, rec["file_sha256"]


def generate_datasets(output_dir):
    """Derive 38 paired subsets from the same standard mother plus five noises.

    The two original standard noise columns are preserved. Every additional
    noise column is generated independent of class labels and other columns.
    The five added columns are not taken from any previous E14 result.
    """
    import numpy as np
    output_dir = Path(output_dir)
    design = make_design()
    tx, ty, vx, vy, gamma_standard, mother_sha = load_frozen()
    rng = np.random.default_rng(SEED_NEW_VALUES)
    new_train = rng.standard_normal((len(tx), NEW_NOISE_COUNT)) * NEW_NOISE_STD
    new_val = rng.standard_normal((len(vx), NEW_NOISE_COUNT)) * NEW_NOISE_STD
    mask_rng = np.random.default_rng(SEED_NEW_MASK)
    new_train[mask_rng.random(new_train.shape) < NEW_NOISE_MISSING_RATE] = np.nan
    new_val[mask_rng.random(new_val.shape) < NEW_NOISE_MISSING_RATE] = np.nan
    full_train = np.column_stack((tx, new_train))
    full_val = np.column_stack((vx, new_val))
    gammas = np.concatenate((gamma_standard,
                             np.full(NEW_NOISE_COUNT, NEW_NOISE_GAMMA)))
    if (full_train.shape != (300, 10) or full_val.shape != (90, 10)
            or gammas.shape != (10,)
            or not np.array_equal(full_train[:, :5], tx, equal_nan=True)
            or not np.array_equal(full_val[:, :5], vx, equal_nan=True)):
        raise AssertionError("Standard columns changed or total noise pool size wrong")
    if output_dir.exists():
        raise FileExistsError("Never replace previously generated subset evidence")
    output_dir.mkdir(parents=True, exist_ok=False)
    manifest = {
        "schema": "E-CH4-14-100-derived-standard-seed-v1",
        "standard_data_seed": STANDARD_SEED,
        "standard_mother_sha256": mother_sha,
        "new_value_seed": SEED_NEW_VALUES,
        "new_mask_seed": SEED_NEW_MASK,
        "new_noise_std": NEW_NOISE_STD,
        "new_noise_missing_rate": NEW_NOISE_MISSING_RATE,
        "new_noise_gamma": NEW_NOISE_GAMMA,
        "informative_columns": list(PRIMARY),
        "original_standard_noise_columns": [2, 4],
        "newly_generated_noise_columns": [5, 6, 7, 8, 9],
        "n_train": 300, "n_validation": 90,
        "test_accessed": False, "datasets": []
    }
    for ds in design["datasets"]:
        selected = ds["feature_indices"]
        dest = output_dir / (ds["dataset_id"] + ".npz")
        with dest.open("xb") as stream:
            np.savez_compressed(stream,
                train_x=full_train[:, selected],
                train_y=ty,
                validation_x=full_val[:, selected],
                validation_y=vy,
                gamma=gammas[selected])
        manifest["datasets"].append({**ds, "sha256": sha(dest)})
    with (output_dir / "manifest.json").open("x", encoding="utf8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
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
