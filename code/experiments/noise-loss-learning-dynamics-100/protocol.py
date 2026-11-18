"""E-CH4-14 controlled 100-update comparison: fixed design and paired data.

No training is performed here. Only frozen Train/Validation inputs are read.
Old eight columns are verified against the immutable E14 N5-01 evidence;
two additional independent Gaussian noise columns complete A_N^max of size 7.
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
OLD = ROOT / "code/experiments/synthetic-noise-feature-sensitivity"
FROZEN = OLD / "inputs/derived/N5-01.npz"
FROZEN_MANIFEST = OLD / "inputs/derived/manifest.json"


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


def load_frozen():
    """Verify the complete prior E14 input before adding columns."""
    import numpy as np
    manifest = json.loads(FROZEN_MANIFEST.read_text("utf8"))
    if manifest.get("test_accessed") is not False:
        raise ValueError("Old source has invalid Test provenance")
    record = next(d for d in manifest["datasets"] if d["subset_id"] == "N5-01")
    if record["feature_mapping"] != list(range(8)) or sha(FROZEN) != record["file_sha256"]:
        raise ValueError("Prior eight-column data hash / mapping mismatch")
    with np.load(FROZEN, allow_pickle=False) as file:
        if set(file.files) != {"train_x", "train_y", "validation_x", "validation_y"}:
            raise ValueError("Only frozen Train/Validation keys are allowed")
        train_x = file["train_x"].copy()
        train_y = file["train_y"].copy()
        val_x = file["validation_x"].copy()
        val_y = file["validation_y"].copy()
    if (train_x.shape != (300, 8) or val_x.shape != (90, 8)
            or train_y.shape != (300,) or val_y.shape != (90,)):
        raise ValueError("Old E14 shapes drifted")
    if len(record["gamma"]) != 8:
        raise ValueError("Missing original gamma values")
    return train_x, train_y, val_x, val_y, np.asarray(record["gamma"], dtype=float), record


def generate_datasets(output_dir, *, overwrite=False):
    """Create paired datasets; no test generation, fitting or mutation of old files."""
    import numpy as np
    output_dir = Path(output_dir)
    plan = make_design()
    tx, ty, vx, vy, old_gamma, source = load_frozen()
    rng = np.random.default_rng(SEED_NEW_VALUES)
    new_train = rng.standard_normal((len(tx), 2)) * 1.1
    new_val = rng.standard_normal((len(vx), 2)) * 1.1
    mask_rng = np.random.default_rng(SEED_NEW_MASK)
    new_train[mask_rng.random(new_train.shape) < 0.20] = np.nan
    new_val[mask_rng.random(new_val.shape) < 0.20] = np.nan
    full_train = np.column_stack((tx, new_train))
    full_val = np.column_stack((vx, new_val))
    gamma = np.concatenate((old_gamma, np.array([0.6, 0.6])))
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"schema": "E-CH4-14-100-derived-v1",
                "test_accessed": False, "source_old_sha256": source["file_sha256"],
                "new_values_seed": SEED_NEW_VALUES, "new_mask_seed": SEED_NEW_MASK,
                "n_train": 300, "n_validation": 90, "datasets": []}
    for ds in plan["datasets"]:
        selected = ds["feature_indices"]
        dest = output_dir / (ds["dataset_id"] + ".npz")
        if dest.exists():
            if not overwrite:
                raise FileExistsError(f"Refusing to overwrite derived data: {dest}")
            raise PermissionError("Frozen dataset overwrite is not supported")
        with dest.open("xb") as stream:
            np.savez_compressed(stream, train_x=full_train[:, selected],
                train_y=ty, validation_x=full_val[:, selected],
                validation_y=vy, gamma=gamma[selected])
        manifest["datasets"].append({**ds, "sha256": sha(dest)})
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf8")
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
