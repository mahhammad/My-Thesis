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

SCHEMA = "imc-snapshot-v2"
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


def _saved_query_start(graph, logits, family, query_starts, source, variant):
    """Capture effective Validation starts, independently of training logits."""
    explicit = query_starts is not None
    if explicit and family not in ("node", "node_class"):
        raise ValueError("Explicit query starts only apply to node families")
    if explicit:
        starts = np.asarray(query_starts.detach().cpu() if isinstance(query_starts, torch.Tensor)
                            else query_starts, dtype=np.float64)
        expected = ((len(graph.qx), graph.d) if family == "node" else
                    (graph.k, len(graph.qx), graph.d))
        if graph.qx is None or starts.shape != expected:
            raise ValueError("Query start logits have incorrect shape")
        if not np.isfinite(starts).all() or not source:
            raise ValueError("Query start logits require finite values and provenance")
        saved = {"logits": starts.tolist(), "shape": list(starts.shape),
                 "sha256": array_fingerprint(starts), "source": source}
    else:
        saved = None

    construction = None
    if variant == "class_tied_node":
        if family != "node_class" or not explicit:
            raise ValueError("class_tied_node requires node_class logits and query starts")
        z = logits.detach().cpu().numpy()
        a = np.ascontiguousarray(z[:, 0, graph.d, :], dtype=np.float64)
        if not np.array_equal(z[:, :, graph.d, :],
                              np.broadcast_to(a[:, None, :], (graph.k, graph.n, graph.d))):
            raise ValueError("class_tied_node origin starts must be tied across nodes")
        if not np.array_equal(starts,
                              np.broadcast_to(a[:, None, :],
                                              (graph.k, len(graph.qx), graph.d))):
            raise ValueError("class_tied_node Validation starts must match tied class starts")
        construction = {
            "variant": "class_tied_node",
            "start_by_class": a.tolist(), "start_shape": [graph.k, graph.d],
            "start_sha256": array_fingerprint(a),
            "rule": "effective_logits[c,i,d,:]=start_by_class[c,:]; "
                    "effective_logits[c,i,:d,:]=intermediate_by_node_class[c,i,:,:]; "
                    "query_start_logits[c,q,:]=start_by_class[c,:]; "
                    "masked_softmax_over_observed_features"
        }
    return saved, construction


def collect_representations(graph, logits, family, query_starts=None):
    """Shared evaluator is the sole source for model distributions."""
    from policy_evaluator import evaluate_graph
    policy = {"family": family, "logits": logits,
              "query_start_logits": query_starts}
    out = evaluate_graph(graph, policy)
    return (out.train_by_hypothesis, out.centers,
            out.validation_by_hypothesis, out.validation_status)


def save_snapshot(
    run_dir, graph, logits, family, *, reference_ids, config_path,
    data_seed, reference_seed, objective, tau, beta, lam, horizon,
    validation_x=None, query_starts=None, query_start_source=None,
    target_centers=None, target_source=None, code_commit=None, notes=None,
    policy_variant=None, config_reference=None, experiment_metadata=None,
    start_by_class=None
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
    if query_starts is None and query_start_source is not None:
        raise ValueError("query_start_source requires explicit query starts")
    n_query = 0 if graph.qx is None else len(graph.qx)
    start_record = None
    if query_starts is not None:
        if family not in ("node", "node_class") or n_query == 0:
            raise ValueError("Explicit query starts require a free node family and queries")
        starts = torch.as_tensor(query_starts, dtype=DT).detach().cpu().numpy()
        expected_shape = ((n_query, graph.d) if family == "node" else
                          (graph.k, n_query, graph.d))
        if starts.shape != expected_shape or not np.isfinite(starts).all():
            raise ValueError("Invalid query start logits shape/values")
        start_record = {
            "logits": starts.tolist(), "shape": list(starts.shape),
            "dtype": "float64", "sha256": array_fingerprint(starts)
        }
    variant = policy_variant or family
    components = None
    if variant == "class_tied_node":
        if family != "node_class" or start_by_class is None:
            raise ValueError("class_tied_node requires independent start_by_class")
        independent = torch.as_tensor(start_by_class, dtype=DT).detach().cpu().numpy()
        if independent.shape != (graph.k, graph.d) or not np.isfinite(independent).all():
            raise ValueError("Invalid independent class-tied starts")
        effective = logits.detach().cpu().numpy()
        expected_train = np.broadcast_to(independent[:, None, :],
                                         (graph.k, graph.n, graph.d))
        if not np.array_equal(effective[:, :, graph.d, :], expected_train):
            raise ValueError("Effective policy disagrees with class-tied start logits")
        if n_query and (start_record is None or not np.array_equal(
            starts, np.broadcast_to(independent[:, None, :],
                                    (graph.k, n_query, graph.d))
        )):
            raise ValueError("Query starts disagree with class-tied start logits")
        components = {
            "start_by_class": {
                "logits": independent.tolist(), "shape": list(independent.shape),
                "dtype": "float64", "sha256": array_fingerprint(independent)
            },
            "intermediate_source": "parameters.logits[:,:,:d,:]",
            "composition_rule": "intermediate_plus_tied_class_start_row",
            "mask_rule": "masked_softmax_over_observed_query_features"
        }
    elif start_by_class is not None:
        raise ValueError("start_by_class is only valid for class_tied_node")
    if target_centers is not None and not target_source:
        raise ValueError("target centers require provenance")
    ids = np.asarray(reference_ids, dtype=np.int64)
    if ids.shape != (graph.n,) or np.min(ids) < 0 or len(set(ids.tolist())) != graph.n:
        raise ValueError("Invalid reference_ids")
    if not bool(torch.isfinite(logits).all()):
        raise ValueError("Policy logits must be finite")
    start_record, construction = _saved_query_start(
        graph, logits, family, query_starts, query_start_source, policy_variant or family
    )
    train, centers, val, status = collect_representations(
        graph, logits, family, query_starts=query_starts
    )
    if status == "available" and family in ("node", "node_class") and start_record is None:
        raise ValueError("Available node Validation must retain its query starts")
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
        "family": family, "policy_variant": variant,
        "policy_components": components,
        "objective": {"name": str(objective), "tau": float(tau)},
        "parameters": {
            "logits": logits.detach().cpu().numpy().tolist(),
            "shape": list(logits.shape),
            "gamma": np.asarray(graph.gamma, dtype=float).tolist(),
            "beta": float(beta), "lambda": float(lam), "H": int(horizon),
            "dtype": "float64", "walk": "exact-capped-geometric",
            "query_start": start_record,
            "policy_construction": construction
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
            "query_start_source": query_start_source,
            "query_start": start_record
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
    # Validate the complete out-of-graph starting policy before accepting a snapshot.
    prediction = model["prediction"]
    status = prediction["validation_status"]
    n_query = model["data"]["validation_count"]
    start_meta = prediction.get("query_start")
    restored_starts = None
    if start_meta is not None:
        if family not in ("node", "node_class") or status != "available":
            raise ValueError("Unexpected stored query starts")
        if not prediction.get("query_start_source"):
            raise ValueError("Stored query starts require provenance")
        restored_starts = np.asarray(start_meta["logits"], dtype=np.float64)
        expected_start_shape = ((n_query, d) if family == "node" else
                                (k, n_query, d))
        if (start_meta.get("dtype") != "float64" or
                start_meta.get("shape") != list(expected_start_shape) or
                restored_starts.shape != expected_start_shape or
                not np.isfinite(restored_starts).all()):
            raise ValueError("Stored query start logits shape/dtype/values mismatch")
        if array_fingerprint(restored_starts) != start_meta.get("sha256"):
            raise ValueError("Query start logits SHA256 mismatch")
    elif family in ("node", "node_class") and status == "available":
        raise ValueError("Available node Validation lacks stored query starts")
    elif prediction.get("query_start_source") is not None:
        raise ValueError("Orphaned query start provenance")

    components = model.get("policy_components")
    if model.get("policy_variant") == "class_tied_node":
        if family != "node_class" or not isinstance(components, dict):
            raise ValueError("Missing class-tied policy components")
        part = components["start_by_class"]
        independent = np.asarray(part["logits"], dtype=np.float64)
        if (part.get("dtype") != "float64" or
                part.get("shape") != [k, d] or
                independent.shape != (k, d) or not np.isfinite(independent).all()):
            raise ValueError("Invalid independent class-tied start parameters")
        if array_fingerprint(independent) != part.get("sha256"):
            raise ValueError("Class-tied start SHA256 mismatch")
        if (components.get("intermediate_source") != "parameters.logits[:,:,:d,:]" or
                components.get("composition_rule") != "intermediate_plus_tied_class_start_row" or
                components.get("mask_rule") != "masked_softmax_over_observed_query_features"):
            raise ValueError("Unknown class-tied policy construction rule")
        effective = np.asarray(model["parameters"]["logits"], dtype=np.float64)
        if not np.array_equal(effective[:, :, d, :],
                              np.broadcast_to(independent[:, None, :], (k, n, d))):
            raise ValueError("Class-tied start does not reconstruct effective policy")
        if restored_starts is not None and not np.array_equal(
            restored_starts, np.broadcast_to(independent[:, None, :],
                                             (k, n_query, d))
        ):
            raise ValueError("Class-tied start does not reconstruct query starts")
    elif components is not None:
        raise ValueError("Unexpected policy components for non-hybrid family")
    starts = model["parameters"].get("query_start")
    construction = model["parameters"].get("policy_construction")
    variant = model.get("policy_variant", family)
    if starts is not None:
        if family not in ("node", "node_class"):
            raise ValueError("Query starts are not valid for this family")
        m = model["data"]["validation_count"]
        expected = (m, d) if family == "node" else (k, m, d)
        if tuple(starts["shape"]) != expected:
            raise ValueError("Query start shape mismatch")
        a = np.asarray(starts["logits"], dtype=np.float64)
        if a.shape != expected or not np.isfinite(a).all():
            raise ValueError("Invalid query start logits")
        if array_fingerprint(a) != starts["sha256"]:
            raise ValueError("Query start SHA256 mismatch")
        if not starts.get("source") or starts["source"] != model["prediction"]["query_start_source"]:
            raise ValueError("Query start provenance mismatch")
    if family in ("node", "node_class") and model["prediction"]["validation_status"] == "available" and starts is None:
        raise ValueError("Available node Validation must include explicit query starts")
    if variant == "class_tied_node":
        if family != "node_class" or starts is None or construction is None:
            raise ValueError("Missing class-tied construction")
        if construction.get("variant") != variant or "masked_softmax_over_observed_features" not in construction.get("rule", ""):
            raise ValueError("Invalid class-tied construction rule")
        a = np.asarray(construction["start_by_class"], dtype=np.float64)
        if a.shape != (k, d) or tuple(construction["start_shape"]) != (k, d):
            raise ValueError("Invalid class-tied start shape")
        if not np.isfinite(a).all() or array_fingerprint(a) != construction["start_sha256"]:
            raise ValueError("Class-tied start SHA256 mismatch")
        z = np.asarray(model["parameters"]["logits"], dtype=np.float64)
        if not np.array_equal(z[:, :, d, :],
                              np.broadcast_to(a[:, None, :], (k, n, d))):
            raise ValueError("Training starts do not match class-tied start")
        if not np.array_equal(np.asarray(starts["logits"]),
                              np.broadcast_to(a[:, None, :], (k, model["data"]["validation_count"], d))):
            raise ValueError("Validation starts do not match class-tied start")
    elif construction is not None:
        raise ValueError("Unexpected policy construction")
    return model, arrays
