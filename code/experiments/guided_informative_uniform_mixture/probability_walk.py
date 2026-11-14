"""Exact random-walk evaluation using a probability-space mixture, not logits.

E-CH4-05 uses the SAME graph weights, no-origin-return constraint, geometric
cutoff and step weights as benchmark_v1.engine.ExactMaskedWalk. Since the
Guided policy has six identical rows, destination-feature probability does
not depend on the previous feature, reducing the transition tensor size.

Do not silently change benchmark_v1.engine (used by historical snapshots).
"""
from __future__ import annotations

from pathlib import Path
import sys

import torch

HERE = Path(__file__).resolve().parent
BENCH = HERE.parent / "benchmark_v1"
sys.path.insert(0, str(BENCH))
from engine import DT, ExactMaskedWalk


class ProbabilityMixtureWalk(ExactMaskedWalk):
    """Exact masked walk with pi=(1-alpha)Uniform+alpha*Uniform(informative)."""

    def __init__(self, *args, informative=(0, 1, 3), **kwargs):
        super().__init__(*args, **kwargs)
        informative = tuple(int(x) for x in informative)
        if self.d != 5 or set(informative) != {0, 1, 3}:
            raise ValueError("Only registered 5-feature scenario {0,1,3} is supported")
        self.informative_mask = torch.as_tensor(
            [i in informative for i in range(self.d)], dtype=torch.bool
        )

    def _mixture(self, valid, strength):
        """Operate on final *traversable* feature mask [..., d].

        All-false rows can occur in inaccessible graph states and get zero
        probability; a reachable row cannot be empty. If positive-probability
        rows exclude every informative feature, reject alpha>0 rather than
        silently route to a noise feature.
        """
        strength = float(strength)
        if not 0 <= strength <= 1:
            raise ValueError("alpha must lie in [0,1]")
        if not isinstance(valid, torch.Tensor) or valid.dtype != torch.bool:
            raise TypeError("valid mask must be boolean torch.Tensor")
        if valid.shape[-1] != self.d:
            raise ValueError("last mask dimension does not match features")
        m = valid.to(DT)
        count = m.sum(dim=-1, keepdim=True)
        uniform = m / count.clamp_min(1)
        if strength == 0:
            return uniform
        info = valid & self.informative_mask
        informed = info.to(DT)
        n_info = informed.sum(dim=-1, keepdim=True)
        if torch.any((count > 0) & (n_info == 0)):
            raise ValueError("a reachable mask has no informative feature")
        guided = informed / n_info.clamp_min(1)
        if strength == 1:
            return guided
        return (1-strength)*uniform + strength*guided

    def represent_train_mixture(self, strength):
        """Train walk: exclude each origin after every hop, exactly as engine."""
        ids = torch.arange(self.n, dtype=torch.long)
        den = self.mass.unsqueeze(0) - self.base[:, :, ids].permute(2, 1, 0)
        valid = (den > 1e-12) & self.available.unsqueeze(0)
        # No dependence on previous feature: one masked row per current node.
        p_next = self._mixture(valid, strength)
        p_start = self._mixture(self.available[ids], strength)
        state = self.base[:, ids, :].permute(1, 2, 0) * p_start[:, None, :]
        out = torch.zeros((self.n, self.k), dtype=DT)
        for t in range(self.horizon):
            out += self.alpha[t] * (state.sum(2) @ self.yhot)
            if t == self.horizon-1:
                break
            choose = state.sum(dim=2).unsqueeze(-1) * p_next
            moved = []
            for b in range(self.d):
                mass = torch.where(
                    den[:, :, b] > 1e-12,
                    choose[:, :, b] / den[:, :, b].clamp_min(1e-12),
                    torch.zeros_like(choose[:, :, b]),
                )
                dest = mass @ self.base[b]
                dest = dest.scatter(1, ids[:, None],
                                    torch.zeros((self.n, 1), dtype=DT))
                moved.append(dest)
            state = torch.stack(moved, dim=2)
        if not torch.allclose(out.sum(1), torch.ones(self.n, dtype=DT),
                              atol=1e-8, rtol=0):
            raise AssertionError("Train walk lost probability mass")
        return out

    def represent_validation_mixture(self, strength):
        """Out-of-reference validation: only training labels are destinations."""
        if self.qx is None:
            raise ValueError("query_x must be supplied at construction")
        B = len(self.qx)
        p_start = self._mixture(self.qavailable, strength)
        valid = (self.mass > 1e-12) & self.available
        p_next = self._mixture(valid, strength)
        state = self.qweights.permute(0, 2, 1) * p_start[:, None, :]
        out = torch.zeros((B, self.k), dtype=DT)
        for t in range(self.horizon):
            out += self.alpha[t] * (state.sum(2) @ self.yhot)
            if t == self.horizon-1:
                break
            choose = state.sum(dim=2).unsqueeze(-1) * p_next[None, :, :]
            state = torch.stack(
                [choose[:, :, b] @ self.base[b] for b in range(self.d)], dim=2
            )
        if not torch.allclose(out.sum(1), torch.ones(B, dtype=DT),
                              atol=1e-8, rtol=0):
            raise AssertionError("Validation walk lost probability mass")
        return out
