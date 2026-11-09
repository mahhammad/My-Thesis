"""Immutable model-output contract for Benchmark v1 (no training, no scoring).

A run exports exactly two raw artifacts:
  model.json         policy logits, graph/data/optimization parameters, provenance
  distributions.npz  Y_i under EVERY class hypothesis, class centers and row IDs

Evaluation labels are deliberately excluded; the independent reporter reloads
them from the configured dataset and verifies fingerprints. No Test data read.
"""
from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
from io import BytesIO
from pathlib import Path
import json
import os
import tempfile

import numpy as np
import torch

from engine import DT, coefficients

SCHEMA = "imc-snapshot-v1"
ARRAYS = ("train_by_hypothesis", "validation_by_hypothesis", "centers", "reference_ids")


def array_fingerprint(array):
    """Hash dtype, shape and canonical C-order bytes (includes labels only as hash)."""
    a = np.ascontiguousarray(np.asarray(array))
    header = json.dumps({"dtype": a.dtype.str, "shape": list(a.shape)},
                        sort_keys=True).encode("utf8")
    return sha256(header + b"\0" + a.tobytes()).hexdigest()


def config_fingerprint(config_path):
    return sha256(Path(config_path).read_bytes()).hexdigest()


def _check_simplex(name, a, expected_shape):
    a = np.asarray(a, dtype=np.float64)
    if a.shape != tuple(expected_shape):
        raise ValueError(f"{name} has shape {a.shape}, expected {expected_shape}")
    if not np.all(np.isfinite(a)) or np.min(a) < -1e-9:
        raise ValueError(f"{name} has invalid probabilities")
    if not np.allclose(a.sum(axis=-1), 1, atol=1e-8, rtol=0):
        raise ValueError(f"{name} has rows not summing to one")
    return a


def _atomic_bytes(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=path.name + ".", dir=str(path.parent))
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(content)
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def collect_representations(graph, logits, family, query_starts=None):
    """Create (N, K hypotheses, K destination labels) without choosing true class."""
    graph.check_logits(logits, family)
    k, n = graph.k, graph.n
    with torch.no_grad():
        if family in ("shared", "node"):
            common = graph.represent_train(logits, family).detach().cpu().numpy()
            train = np.repeat(common[:, None, :], k, axis=1)
        else:
            train = np.stack([
                graph.represent_train(logits, family, c).detach().cpu().numpy()
                for c in range(k)
            ], axis=1)
        train = _check_simplex("train_by_hypothesis", train, (n, k, k))
        labels = graph.labels.detach().cpu().numpy()
        if not np.array_equal(np.unique(labels), np.arange(k)):
            raise ValueError("Every reference class must be nonempty and 0-based")
        centers = np.stack([train[labels == c, c, :].mean(axis=0)
                            for c in range(k)])
        centers = _check_simplex("centers", centers, (k, k))

        query_count = 0 if graph.qx is None else len(graph.qx)
        if query_count and (family in ("shared", "class") or query_starts is not None):
            if family in ("shared", "node"):
                v = graph.represent_query(
                    logits, family, query_start_logits=query_starts
                ).detach().cpu().numpy()
                val = np.repeat(v[:, None, :], k, axis=1)
            else:
                val = np.stack([
                    graph.represent_query(
                        logits, family, c, query_start_logits=query_starts
                    ).detach().cpu().numpy()
                    for c in range(k)
                ], axis=1)
            val = _check_simplex("validation_by_hypothesis", val,
                                 (query_count, k, k))
            status = "available"
        else:
            val = np.empty((0, k, k), dtype=np.float64)
            status = "not_defined_without_query_start" if query_count else "no_queries"
        return train, centers, val, status


def save_snapshot(
    run_dir, graph, logits, family, *, reference_ids, config_path,
    data_seed, reference_seed, objective, tau, beta, lam, horizon,
    validation_x=None, query_starts=None, query_start_source=None,
    target_centers=None, target_source=None, code_commit=None, notes=None,
    policy_variant=None, config_reference=None, experiment_metadata=None
):
    """Write model.json + distributions.npz. Does not calculate accuracy or R/I.

    The reporter independently regenerates train/validation labels. The caller
    must provide the exact order of reference_ids relative to generator Train.
    """
    run_dir = Path(run_dir)
    if (run_dir / "model.json").exists() or (run_dir / "distributions.npz").exists():
        raise FileExistsError("Snapshot is immutable; choose a fresh run directory")
    if float(tau) <= 0:
        raise ValueError("tau must be positive")
    expected_alpha = coefficients(beta, lam, horizon)
    if not torch.allclose(graph.alpha, expected_alpha, atol=1e-14, rtol=0):
        raise ValueError("Saved walk parameters do not match graph coefficients")
    if horizon != graph.horizon:
        raise ValueError("horizon mismatch")
    if graph.qx is None and validation_x is not None:
        raise ValueError("graph was not configured with validation queries")
    if graph.qx is not None and validation_x is None:
        raise ValueError("validation_x is required to fingerprint query order")
    if validation_x is not None and not np.array_equal(
        graph.qx, np.asarray(validation_x), equal_nan=True
    ):
        raise ValueError("validation_x differs from graph queries")
    if query_starts is not None and not query_start_source:
        raise ValueError("explicit query start logits require provenance")
    if target_centers is not None and not target_source:
        raise ValueError("target centers require provenance")
    ids = np.asarray(reference_ids, dtype=np.int64)
    if ids.shape != (graph.n,) or np.min(ids) < 0 or len(set(ids.tolist())) != graph.n:
        raise ValueError("Invalid reference_ids")
    if not bool(torch.isfinite(logits).all()):
        raise ValueError("Policy logits must be finite")
    train, centers, val, status = collect_representations(
        graph, logits, family, query_starts=query_starts
    )
    target = None if target_centers is None else _check_simplex(
        "target_centers", target_centers, centers.shape
    ).tolist()
    buff = BytesIO()
    np.savez_compressed(buff, train_by_hypothesis=train,
                        validation_by_hypothesis=val, centers=centers,
                        reference_ids=ids)
    payload = buff.getvalue()
    config_path = Path(config_path)
    manifest = {
        "schema": SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "family": family, "policy_variant": policy_variant or family,
        "objective": {"name": str(objective), "tau": float(tau)},
        "parameters": {
            "logits": logits.detach().cpu().numpy().tolist(),
            "shape": list(logits.shape),
            "gamma": np.asarray(graph.gamma, dtype=float).tolist(),
            "beta": float(beta), "lambda": float(lam), "H": int(horizon),
            "dtype": "float64", "walk": "exact-capped-geometric"
        },
        "data": {
            "config_path": (config_reference if config_reference is not None
                            else config_path.as_posix()),
            "config_sha256": config_fingerprint(config_path),
            "data_seed": int(data_seed), "reference_seed": int(reference_seed),
            "reference_ids": ids.tolist(),
            "train_reference_x_sha256": array_fingerprint(graph.x),
            "train_reference_y_sha256": array_fingerprint(
                graph.labels.detach().cpu().numpy().astype(np.int64)),
            "validation_x_sha256": (
                array_fingerprint(validation_x) if validation_x is not None else None),
            "validation_count": 0 if validation_x is None else len(validation_x),
            "test_accessed": False
        },
        "prediction": {
            "axis_order": ["sample", "candidate_class", "destination_label"],
            "validation_status": status,
            "query_start_source": query_start_source
        },
        "target": {"centers": target, "source": target_source},
        "provenance": {"code_commit": code_commit, "notes": notes},
        "experiment": experiment_metadata or {},
        "distributions_file": "distributions.npz",
        "distributions_sha256": sha256(payload).hexdigest()
    }
    encoded = (json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False)
               + "\n").encode("utf8")
    _atomic_bytes(run_dir / "distributions.npz", payload)
    _atomic_bytes(run_dir / "model.json", encoded)
    return manifest


def load_snapshot(run_dir):
    """Validate and load the two raw files, never execute untrusted pickle."""
    run_dir = Path(run_dir)
    model = json.loads((run_dir / "model.json").read_text(encoding="utf8"))
    if model.get("schema") != SCHEMA:
        raise ValueError("Unknown snapshot schema")
    raw = (run_dir / "distributions.npz").read_bytes()
    if sha256(raw).hexdigest() != model["distributions_sha256"]:
        raise ValueError("distributions.npz SHA256 mismatch")
    with np.load(BytesIO(raw), allow_pickle=False) as data:
        if set(data.files) != set(ARRAYS):
            raise ValueError("Unexpected array keys")
        arrays = {key: data[key].copy() for key in ARRAYS}
    k = len(arrays["centers"])
    ids = arrays["reference_ids"]
    n = len(ids)
    _check_simplex("centers", arrays["centers"], (k, k))
    _check_simplex("train_by_hypothesis", arrays["train_by_hypothesis"], (n, k, k))
    v = arrays["validation_by_hypothesis"]
    if v.shape != (0, k, k):
        _check_simplex("validation_by_hypothesis", v, (len(v), k, k))
    if ids.tolist() != model["data"]["reference_ids"]:
        raise ValueError("Reference ID mismatch between files")
    if len(v) != (model["data"]["validation_count"]
                 if model["prediction"]["validation_status"] == "available" else 0):
        raise ValueError("Validation row count mismatch")
    family = model["family"]
    d = len(model["parameters"]["gamma"])
    shape = {
        "shared": (d + 1, d), "class": (k, d + 1, d),
        "node": (n, d + 1, d), "node_class": (k, n, d + 1, d)
    }.get(family)
    if shape is None or tuple(model["parameters"]["shape"]) != shape or (
        np.shape(model["parameters"]["logits"]) != shape
    ):
        raise ValueError("Policy family/logits shape mismatch")
    return model, arrays
