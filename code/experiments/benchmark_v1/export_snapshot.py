"""Export an IMC model snapshot without fitting or reporting.

Default is DRY RUN. Set --execute to compute and save two raw model artifacts.
Supports uniform policy, a saved JSON checkpoint, or an archived ZIP member.

Examples (no historical Test loaded):
  python code/experiments/benchmark_v1/export_snapshot.py --uniform \
    --run-dir code/experiments/benchmark_v1/results/pipeline_v1/uniform
  python code/experiments/benchmark_v1/export_snapshot.py --uniform --execute \
    --run-dir code/experiments/benchmark_v1/results/pipeline_v1/uniform
  python code/experiments/benchmark_v1/export_snapshot.py --checkpoint-json run.json \
    --run-dir PATH --execute
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys
import zipfile

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "code" / "data_generator"))
from generate import load_config
from backfill_saved_scores import dataset_train_val
from engine import ExactMaskedWalk, DT
from snapshot_pipeline import save_snapshot
from policy_interface import create_policy


def choose_proportional(y, count, seed):
    """Class-proportional subset as in benchmark replay; no Test data."""
    y = np.asarray(y, dtype=int)
    if count < 1 or count > len(y):
        raise ValueError("nref must be between 1 and total Train")
    if count == len(y):
        return np.arange(len(y), dtype=int)
    k = int(y.max()) + 1
    population = np.bincount(y, minlength=k)
    amounts = population * (count / len(y))
    per = np.floor(amounts).astype(int)
    rem = count - int(per.sum())
    order = np.argsort(-(amounts - per), kind="stable")
    per[order[:rem]] += 1
    if np.any(per == 0) or np.any(per > population):
        raise ValueError("Each class needs a nonempty subset")
    rng = np.random.default_rng(seed)
    return np.concatenate([
        rng.permutation(np.flatnonzero(y == c))[:per[c]] for c in range(k)
    ])


def read_checkpoint(args):
    if args.uniform:
        return None, "uniform: zeros"
    if args.checkpoint_json:
        path = Path(args.checkpoint_json)
        raw = path.read_bytes()
        return json.loads(raw), f"checkpoint_json:{path}:sha256:{sha256(raw).hexdigest()}"
    if not (args.archive and args.member):
        raise ValueError("--archive and --member must be supplied together")
    path = Path(args.archive)
    with zipfile.ZipFile(path) as archive:
        members = archive.namelist()
        if args.member not in members:
            raise ValueError("Requested member not in archive")
        # Require the member's digest in the corresponding saved manifest.
        manifests = [n for n in members if n == "manifest.json"
                     or n.endswith("/manifest.json")]
        if not manifests:
            raise ValueError("Archive lacks its original manifest")
        raw = archive.read(args.member)
        digest = sha256(raw).hexdigest()
        verified = False
        for mf in manifests:
            prefix = mf[:-len("manifest.json")]
            manifest = json.loads(archive.read(mf))
            for item in manifest.get("runs", []):
                if prefix + item.get("file", "") == args.member:
                    if item.get("sha256") != digest:
                        raise ValueError("Checkpoint digest mismatch")
                    verified = True
        if not verified:
            raise ValueError("Requested member not declared by manifest")
    return json.loads(raw), (f"archive:{path}:member:{args.member}"
                             f":archive_sha256:{sha256(path.read_bytes()).hexdigest()}"
                             f":member_sha256:{digest}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--uniform", action="store_true")
    g.add_argument("--checkpoint-json", type=Path)
    g.add_argument("--archive", type=Path)
    p.add_argument("--member", help="Archive member named in its manifest")
    p.add_argument("--run-dir", required=True, type=Path)
    p.add_argument("--config", type=Path, default=ROOT / "code/data_generator/config.json")
    p.add_argument("--family", choices=("shared", "class", "node", "node_class"),
                   default="shared", help="Only for --uniform")
    p.add_argument("--nref", type=int, default=300, help="Only for --uniform")
    p.add_argument("--data-seed", type=int, default=20261102)
    p.add_argument("--reference-seed", type=int, default=20261107)
    p.add_argument("--tau", type=float, help="Defaults to checkpoint tau or .005")
    p.add_argument("--target-centers-json", type=Path, help="Optional objective-specific analysis")
    p.add_argument("--target-source", help="Required with target centers")
    p.add_argument("--code-commit", help="For reproducibility; optional for exploration")
    p.add_argument("--execute", action="store_true", help="Compute and write snapshot")
    args = p.parse_args()

    if args.archive and not args.member:
        p.error("--archive requires --member")
    if args.member and not args.archive:
        p.error("--member requires --archive")
    if args.target_centers_json and not args.target_source:
        p.error("--target-centers-json requires --target-source")
    if not args.execute:
        print(json.dumps({
            "status": "DRY_RUN", "run_dir": str(args.run_dir),
            "source": "uniform" if args.uniform else "saved_checkpoint",
            "raw_outputs": ["model.json", "distributions.npz"],
            "training": False, "historical_test_accessed": False
        }, ensure_ascii=False, indent=2))
        return

    config = args.config.resolve()
    try:
        portable_config = config.relative_to(ROOT)
    except ValueError as exc:
        raise ValueError("Generator config must be inside repository for portable snapshot") from exc
    ckpt, origin = read_checkpoint(args)
    spec = load_config(config)
    if ckpt is None:
        seed = args.data_seed
        reference_seed = args.reference_seed
    else:
        dat = ckpt.get("dataset", {})
        seed = int(dat.get("data_seed", dat.get("seed", args.data_seed)))
        reference_seed = int(dat.get("reference_seed",
                                    dat.get("ref_seed", args.reference_seed)))
    spec["seed"] = seed
    ds = dataset_train_val(spec)
    x, y = ds["train"]
    val_x, _ = ds["validation"]

    if ckpt is None:
        ids = choose_proportional(y, args.nref, reference_seed)
        family = args.family
        objective = "uniform"
    else:
        dat = ckpt.get("dataset", {})
        if "reference_ids" not in dat:
            raise ValueError("Checkpoint lacks reference_ids; cannot verify reference order")
        ids = np.asarray(dat["reference_ids"], dtype=int)
        if len(ids) == 0 or np.any(ids < 0) or np.any(ids >= len(y)):
            raise ValueError("Invalid checkpoint reference_ids")
        family = ckpt["family"]
        if family == "class_tied_start_node_class_middle":
            family = "class_tied_node"
        objective = ckpt.get("objective", "historical_unknown")

    graph = ExactMaskedWalk(x[ids], y[ids], spec["gamma"],
                            beta=.4, lam=.8, horizon=12, query_x=val_x)
    if ckpt is None:
        shapes = {"shared": (graph.d + 1, graph.d),
                  "class": (graph.k, graph.d + 1, graph.d),
                  "node": (graph.n, graph.d + 1, graph.d),
                  "node_class": (graph.k, graph.n, graph.d + 1, graph.d)}
        parameters = torch.zeros(shapes[family], dtype=DT)
    else:
        parameters = ckpt.get("raw_logits", ckpt.get("logits"))
        if parameters is None:
            raise ValueError("Checkpoint lacks raw_logits/logits")
    adapter = create_policy(
        family, parameters, n=graph.n, k=graph.k, d=graph.d, n_query=len(val_x)
    )
    z = adapter.logits
    graph.check_logits(z, adapter.family)

    tau = args.tau if args.tau is not None else (
        ckpt.get("tau", ckpt.get("optimizer", {}).get("tau", .005))
        if ckpt else .005)
    target = None
    if args.target_centers_json:
        raw = json.loads(args.target_centers_json.read_text(encoding="utf8"))
        target = raw.get("target_centers", raw.get("centers")) if isinstance(raw, dict) else raw
    manifest = save_snapshot(
        args.run_dir, graph, z, adapter.family, reference_ids=ids,
        config_path=config, config_reference=portable_config.as_posix(),
        data_seed=seed, reference_seed=reference_seed,
        objective=objective, tau=float(tau), beta=.4, lam=.8, horizon=12,
        validation_x=val_x, query_starts=adapter.query_start_logits,
        query_start_source=adapter.query_start_source,
        policy_variant=adapter.variant, target_centers=target,
        target_source=args.target_source,
        code_commit=args.code_commit, notes=origin)
    print("Saved:", args.run_dir / "model.json",
          args.run_dir / "distributions.npz")
    print("Family:", manifest["family"], "Validation:", manifest["prediction"]["validation_status"])
    print("No optimizer was run. Historical Test was not accessed.")


if __name__ == "__main__":
    main()
