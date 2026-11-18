"""NumPy-only exact masked sample-feature walk and analytic reverse-mode VJP.

Port of benchmark_v1/engine.py::ExactMaskedWalk for shared/class policies.
This is not a Monte Carlo estimator and does not import torch. The origin is
excluded after each train hop; query hops use the original reference kernel.
"""
from __future__ import annotations
import numpy as np

LOG2 = float(np.log(2.0))


def coefficients(beta=.4, lam=.8, horizon=12):
    a = np.zeros(horizon, dtype=np.float64)
    for length in range(1, horizon + 1):
        probability = (beta * (1 - beta)**(length - 1)
                       if length < horizon else (1 - beta)**(horizon - 1))
        denom = sum(lam**j for j in range(length))
        for t in range(length):
            a[t] += probability * lam**t / denom
    if not np.isclose(a.sum(), 1, atol=1e-12):
        raise AssertionError("Exact-walk coefficients lost mass")
    return a


def feature_kernel(x, gamma):
    n, d = x.shape
    out = np.zeros((d, n, n), dtype=np.float64)
    for b in range(d):
        observed = np.isfinite(x[:, b])
        values = np.where(observed, x[:, b], 0.)
        w = np.exp(-gamma[b] * (values[:, None] - values[None, :])**2)
        w *= observed[:, None] & observed[None, :]
        np.fill_diagonal(w, 0.)
        den = w.sum(axis=1, keepdims=True)
        np.divide(w, den, out=w, where=den > 0)
        out[b] = w
    return out


def query_kernel(q, ref, gamma):
    m, d = q.shape
    n = len(ref)
    out = np.zeros((m, d, n), dtype=np.float64)
    for b in range(d):
        qo, ro = np.isfinite(q[:, b]), np.isfinite(ref[:, b])
        qv = np.where(qo, q[:, b], 0.)
        rv = np.where(ro, ref[:, b], 0.)
        w = np.exp(-gamma[b] * (qv[:, None] - rv[None, :])**2)
        w *= qo[:, None] & ro[None, :]
        den = w.sum(axis=1, keepdims=True)
        np.divide(w, den, out=w, where=den > 0)
        out[:, b, :] = w
    return out


def masked_softmax(logits, mask):
    """Match the thesis masked-softmax -1e25 convention, including empty rows."""
    a = np.where(mask, logits, -1e25)
    e = np.exp(a - np.max(a, axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def entropy(p):
    return -np.sum(np.where(p > 0, p * np.log(np.maximum(p, 1e-300)), 0), axis=-1)


def jsd(p, q):
    m = .5 * (p + q)
    return np.maximum(0., entropy(m) - .5 * (entropy(p) + entropy(q)))


def js_score_and_partials(p, q):
    """JS similarity and both partial derivatives at fixed probability support."""
    p, q = np.broadcast_arrays(np.asarray(p), np.asarray(q))
    score = 1. - jsd(p, q) / LOG2
    tot = p + q
    # Fixed graph/mask support: coordinates that are identically zero have
    # identically zero derivatives. Do not use a zero-filled synthetic gradient.
    pa = np.zeros_like(p)
    qa = np.zeros_like(q)
    np.log(np.maximum(2 * p / np.maximum(tot, 1e-300), 1e-300), out=pa)
    np.log(np.maximum(2 * q / np.maximum(tot, 1e-300), 1e-300), out=qa)
    pa *= -.5 / LOG2
    qa *= -.5 / LOG2
    pa = np.where(p > 0, pa, 0.)
    qa = np.where(q > 0, qa, 0.)
    return score, pa, qa


class ExactMaskedWalkNP:
    def __init__(self, train_x, train_y, gamma, *, beta=.4, lam=.8,
                 horizon=12, query_x=None):
        self.x = np.asarray(train_x, dtype=np.float64)
        self.labels = np.asarray(train_y, dtype=np.int64)
        self.n, self.d = self.x.shape
        self.k = int(self.labels.max()) + 1
        self.gamma = np.asarray(gamma, dtype=np.float64)
        if self.gamma.shape != (self.d,) or np.any(self.gamma <= 0):
            raise ValueError("Invalid kernel widths")
        self.horizon = horizon
        self.alpha = coefficients(beta, lam, horizon)
        self.base = feature_kernel(self.x, self.gamma)
        self.available = np.isfinite(self.x)
        self.mass = self.base.sum(axis=2).T.copy()  # node, feature
        self.den = self.mass[None, :, :] - self.base.transpose(2, 1, 0)
        self.valid = (self.den > 1e-12) & self.available[None, :, :]
        self.yhot = np.eye(self.k)[self.labels]
        self.qx = None if query_x is None else np.asarray(query_x, dtype=np.float64)
        if self.qx is not None:
            self.qweights = query_kernel(self.qx, self.x, self.gamma)
            self.qavailable = np.isfinite(self.qx)

    def shape(self, family):
        if family == "shared":
            return self.d + 1, self.d
        if family == "class":
            return self.k, self.d + 1, self.d
        raise ValueError("Only shared/class hypotheses supported")

    def _policy(self, z, family, c):
        z = np.asarray(z, dtype=np.float64)
        if z.shape != self.shape(family):
            raise ValueError(f"Policy shape mismatch: {z.shape}")
        return z if family == "shared" else z[c]

    def _train_matrices(self, policy):
        n, d = self.n, self.d
        p0 = masked_softmax(np.broadcast_to(policy[-1], (n, d)),
                            self.available)
        mid = np.broadcast_to(policy[:d], (n, n, d, d))
        masks = self.valid[:, :, None, :]
        pol = masked_softmax(mid, masks)
        return p0, pol

    def forward_train(self, z, family, c=0, *, record=False):
        policy = self._policy(z, family, c)
        p0, pol = self._train_matrices(policy)
        state = self.base.transpose(1, 2, 0) * p0[:, None, :]
        states = [] if record else None
        out = np.zeros((self.n, self.k), dtype=np.float64)
        for t, coeff in enumerate(self.alpha):
            if record:
                states.append(state)
            out += coeff * (state.sum(axis=2) @ self.yhot)
            if t == self.horizon - 1:
                break
            choice = np.einsum("sia,siab->sib", state, pol, optimize=True)
            moved = []
            for b in range(self.d):
                mass = np.divide(choice[:, :, b], self.den[:, :, b],
                                 out=np.zeros((self.n, self.n)),
                                 where=self.den[:, :, b] > 1e-12)
                dest = mass @ self.base[b]
                dest[np.arange(self.n), np.arange(self.n)] = 0.
                moved.append(dest)
            state = np.stack(moved, axis=2)
        if not np.allclose(out.sum(axis=1), 1., rtol=0, atol=1e-8):
            raise AssertionError("Train exact walk lost probability mass")
        return out, ((states, p0, pol) if record else None)

    def backward_train(self, context, upstream):
        """d(sum_s,u upstream[s,u]*Y[s,u])/d(one hypothesis logits).

        Computes the exact adjoint of every choice and masked transition.
        """
        states, p0, pol = context
        upstream = np.asarray(upstream, dtype=np.float64)
        n, d = self.n, self.d
        if upstream.shape != (n, self.k):
            raise ValueError("Wrong upstream representation gradient")
        incoming = upstream @ self.yhot.T
        gm = np.zeros((d, d), dtype=np.float64)
        bar_next = None
        for t in range(self.horizon - 1, -1, -1):
            bar = np.broadcast_to(self.alpha[t] * incoming[:, :, None],
                                  (n, n, d)).copy()
            if bar_next is not None:
                dchoice = np.empty((n, n, d), dtype=np.float64)
                for b in range(d):
                    adj_dest = bar_next[:, :, b].copy()
                    adj_dest[np.arange(n), np.arange(n)] = 0.
                    numerator = adj_dest @ self.base[b].T
                    dchoice[:, :, b] = np.divide(
                        numerator, self.den[:, :, b],
                        out=np.zeros_like(numerator),
                        where=self.den[:, :, b] > 1e-12)
                bar = bar + np.einsum("sib,siab->sia", dchoice, pol,
                                      optimize=True)
                mean = np.einsum("siab,sib->sia", pol, dchoice, optimize=True)
                sensitivity = pol * (dchoice[:, :, None, :] - mean[:, :, :, None])
                gm += np.einsum("sia,siab->ab", states[t], sensitivity,
                                optimize=True)
            bar_next = bar
        dp0 = np.einsum("sjb,bsj->sb", bar_next, self.base, optimize=True)
        g0 = (p0 * (dp0 - np.sum(p0 * dp0, axis=1, keepdims=True))).sum(axis=0)
        return np.concatenate((gm, g0[None, :]), axis=0)

    def forward_query(self, z, family, c=0):
        if self.qx is None:
            raise ValueError("No Validation query supplied")
        policy = self._policy(z, family, c)
        m, d = self.qx.shape
        p0 = masked_softmax(np.broadcast_to(policy[-1], (m, d)),
                            self.qavailable)
        state = self.qweights.transpose(0, 2, 1) * p0[:, None, :]
        val = (self.mass > 1e-12) & self.available
        pol = masked_softmax(np.broadcast_to(policy[:d], (self.n, d, d)),
                             val[:, None, :])
        out = np.zeros((m, self.k), dtype=np.float64)
        for t, coeff in enumerate(self.alpha):
            out += coeff * (state.sum(axis=2) @ self.yhot)
            if t == self.horizon - 1:
                break
            choice = np.einsum("qia,iab->qib", state, pol, optimize=True)
            state = np.stack(
                [choice[:, :, b] @ self.base[b] for b in range(d)], axis=2)
        if not np.allclose(out.sum(axis=1), 1., rtol=0, atol=1e-8):
            raise AssertionError("Validation exact walk lost probability mass")
        return out
