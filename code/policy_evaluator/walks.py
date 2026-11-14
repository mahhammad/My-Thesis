"""Fixed-policy Monte Carlo label-neighborhood representation; no training.

Each walk produces ONE normalized label distribution, with geometric length
capped at max_steps. Origin and current reference nodes cannot be destinations.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


def load_policy(path: str | Path, d: int) -> dict:
    cfg = json.loads(Path(path).read_text(encoding='utf-8'))
    logits = np.asarray(cfg['logits'], dtype=float)
    gamma = np.asarray(cfg['kernel_gamma'], dtype=float)
    if logits.shape != (d + 1, d) or gamma.shape != (d,):
        raise ValueError(f'policy logits must be ({d+1},{d}) and kernel_gamma ({d},)')
    if not np.isfinite(logits).all() or not np.isfinite(gamma).all() or (gamma <= 0).any():
        raise ValueError('policy logits must be finite; gamma must be finite and positive')
    beta, lam = float(cfg['beta']), float(cfg['lambda'])
    if not (0 < beta <= 1 and 0 < lam <= 1):
        raise ValueError('beta and lambda must be in (0,1]')
    walks, maximum = cfg['walks_per_sample'], cfg['max_steps']
    if type(walks) is not int or walks < 1 or (maximum is not None and (type(maximum) is not int or maximum < 1)):
        raise ValueError('walks_per_sample and max_steps must be positive integers')
    if type(cfg['seed']) is not int or cfg['seed'] < 0:
        raise ValueError('seed must be a nonnegative integer')
    return {'logits': logits, 'kernel_gamma': gamma, 'beta': beta,
            'lambda': lam, 'walks_per_sample': walks, 'max_steps': maximum,
            'seed': cfg['seed'], 'name': cfg.get('name', 'fixed')}


def policy_fingerprint(policy: dict) -> str:
    content = {k: (v.tolist() if isinstance(v, np.ndarray) else v)
               for k, v in policy.items() if k != 'name'}
    return hashlib.sha256(json.dumps(content, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def reference_fingerprint(x: np.ndarray, y: np.ndarray) -> str:
    h = hashlib.sha256()
    for a in (np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.int64)):
        h.update(str(a.shape).encode())
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


def load_data(path: str | Path, labels_required: bool = True):
    with np.load(path, allow_pickle=False) as z:
        x = np.asarray(z['x'], dtype=float)
        y = np.asarray(z['y'], dtype=int) if 'y' in z.files else None
        if 'mask' in z.files and not np.array_equal(np.asarray(z['mask'], bool), np.isfinite(x)):
            raise ValueError('mask must coincide with finite observed values')
    if x.ndim != 2 or np.isinf(x).any() or (labels_required and y is None):
        raise ValueError('x must be 2D with NaN missing values; reference requires labels')
    if y is not None and (y.shape != (len(x),) or (y < 0).any()):
        raise ValueError('invalid labels')
    return x, y


def _weights(x, refs, gamma, self_indices=None):
    """Vectorized feature-kernel weights, chunked by queries for bounded memory."""
    x, refs, gamma = np.asarray(x,float), np.asarray(refs,float), np.asarray(gamma,float)
    nq, d = x.shape
    nr = len(refs)
    if refs.ndim != 2 or refs.shape[1] != d or gamma.shape != (d,):
        raise ValueError('reference/feature dimensions mismatch')
    if self_indices is not None:
        self_indices = np.asarray(self_indices, int)
        if self_indices.shape != (nq,):
            raise ValueError('self_indices dimension mismatch')
    out = np.zeros((nq, d, nr), dtype=np.float64)
    for a in range(d):
        rows = np.flatnonzero(np.isfinite(x[:, a]))
        cols = np.flatnonzero(np.isfinite(refs[:, a]))
        if not len(cols):
            continue
        for start in range(0, len(rows), 64):
            selected = rows[start:start + 64]
            logw = -gamma[a] * (x[selected, a, None] - refs[None, cols, a])**2
            if self_indices is not None:
                logw = np.where(cols[None, :] == self_indices[selected, None], -np.inf, logw)
            peak = logw.max(axis=1)
            good = np.isfinite(peak)
            if not good.any():
                continue
            expw = np.exp(logw[good] - peak[good, None])
            out[selected[good, None], a, cols[None, :]] = expw
    return out

@dataclass
class ReferenceGraph:
    x: np.ndarray
    y: np.ndarray
    gamma: np.ndarray

    def __post_init__(self):
        self.x = np.asarray(self.x, float)
        self.y = np.asarray(self.y, int)
        if self.x.ndim != 2 or self.y.shape != (len(self.x),):
            raise ValueError('reference shapes mismatch')
        self.nclasses = int(self.y.max()) + 1
        if np.unique(self.y).size != self.nclasses:
            raise ValueError('all classes 0..K-1 must occur in training reference')
        self.weights = _weights(self.x, self.x, self.gamma,
                                np.arange(len(self.x)))
        self.totals = self.weights.sum(axis=2)
        self.cumulatives = np.cumsum(self.weights, axis=2)
        self.prior = np.bincount(self.y, minlength=self.nclasses) / len(self.y)

    def representation(self, queries: np.ndarray, policy: dict,
                       train_indices: np.ndarray | None = None,
                       *, seed: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Label distributions and no-walk flags. No query labels are used.

        For train samples pass train_indices=np.arange(n_train). For held-out
        queries, pass None: all transition destinations remain train nodes.
        """
        queries = np.asarray(queries, float)
        nq, d = queries.shape
        if d != self.x.shape[1] or not np.array_equal(policy['kernel_gamma'], self.gamma):
            raise ValueError('feature dimension or kernel gamma differs from reference')
        if train_indices is None:
            ids = np.full(nq, -1, dtype=int)
        else:
            ids = np.asarray(train_indices, dtype=int)
            if ids.shape != (nq,) or ((ids < 0) | (ids >= len(self.x))).any():
                raise ValueError('train_indices must map each query to its reference index')
            if not np.array_equal(np.nan_to_num(queries, nan=1e300),
                                  np.nan_to_num(self.x[ids], nan=1e300)):
                raise ValueError('train queries must match their reference samples')
        seeds = np.random.SeedSequence(policy['seed'] if seed is None else seed).spawn(nq)
        qweights = _weights(queries, self.x, self.gamma, ids)
        neighborhoods = np.zeros((nq, self.nclasses))
        no_walk = np.zeros(nq, dtype=bool)
        for i in range(nq):
            neighborhoods[i], no_walk[i] = self._one(qweights[i], ids[i], policy,
                                                       np.random.default_rng(seeds[i]))
        return neighborhoods, no_walk

    def _one(self, start_weights: np.ndarray, origin: int, policy: dict,
             rng: np.random.Generator) -> tuple[np.ndarray, bool]:
        nwalks = policy['walks_per_sample']
        horizon = policy['max_steps']
        length = rng.geometric(policy['beta'], size=nwalks)
        if horizon is not None:
            length = np.minimum(length, horizon)
        # Entire walk has unit mass, regardless of its random length.
        lam = policy['lambda']
        z = -np.expm1(length * np.log(lam)) / (1 - lam) if lam != 1.0 else length.astype(float)
        result = np.zeros((nwalks, self.nclasses), dtype=float)
        current = np.full(nwalks, -1, dtype=int)
        prev = np.full(nwalks, self.x.shape[1], dtype=int)  # last row = START
        for t in range(1, int(length.max()) + 1):
            selected = np.flatnonzero(length >= t)
            if not len(selected):
                break
            curr = current[selected]
            first = curr < 0
            allowed_total = np.empty((len(selected), self.x.shape[1]), dtype=float)
            if first.any():
                allowed_total[first] = start_weights.sum(axis=1)
            if (~first).any():
                allowed_total[~first] = self.totals[curr[~first]]
                if origin >= 0:
                    allowed_total[~first] -= self.weights[curr[~first], :, origin]
            good = allowed_total > 1e-13
            if not np.all(good.any(axis=1)):
                raise ValueError('dead-end walk: no allowed feature/destination; adjust dataset/mask')
            logits = policy['logits'][prev[selected]]
            zlog = np.where(good, logits, -np.inf)
            w = np.exp(zlog - np.max(zlog, axis=1, keepdims=True))
            w /= w.sum(axis=1, keepdims=True)
            u = rng.random(len(selected))
            feat = (np.cumsum(w, axis=1) < u[:, None]).sum(axis=1)
            cum = np.empty((len(selected), len(self.x)), dtype=float)
            full_total = np.empty(len(selected), dtype=float)
            banned_weight = np.zeros(len(selected), dtype=float)
            if first.any():
                cum[first] = np.cumsum(start_weights[feat[first]], axis=1)
                full_total[first] = cum[first, -1]
            if (~first).any():
                cum[~first] = self.cumulatives[curr[~first], feat[~first], :]
                full_total[~first] = self.totals[curr[~first], feat[~first]]
                if origin >= 0:
                    banned_weight[~first] = self.weights[curr[~first], feat[~first], origin]
            u2 = rng.random(len(selected))
            thresh = np.maximum(np.finfo(float).tiny,
                                u2 * (full_total - banned_weight))
            if origin >= 0:
                prefix = cum[:, origin] - banned_weight
                thresh += (thresh >= prefix) * banned_weight
            dest = (cum < thresh[:, None]).sum(axis=1)
            if (dest >= len(self.x)).any() or (dest == curr).any() or (origin >= 0 and (dest == origin).any()):
                raise RuntimeError('invalid graph transition: destination is forbidden')
            weight_at_step = lam ** (t - 1) / z[selected]
            np.add.at(result, (selected, self.y[dest]), weight_at_step)
            current[selected] = dest
            prev[selected] = feat
        return result.mean(axis=0), False


    def one_step_q(self, queries: np.ndarray | None = None,
                   train_indices: np.ndarray | None = None,
                   origin_indices: np.ndarray | None = None,
                   batch_size: int = 64) -> np.ndarray:
        """Exact one-step next-label distributions (N,D,K), no walk sampling."""
        from one_step import one_step_q
        return one_step_q(self, queries=queries, train_indices=train_indices,
                          origin_indices=origin_indices, batch_size=batch_size)
