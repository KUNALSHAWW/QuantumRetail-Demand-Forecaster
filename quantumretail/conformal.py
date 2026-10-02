"""Conformal calibration of quantile forecasts.

Raw quantile-regression outputs are usually mis-calibrated: an "80% interval" might
cover 70% of outcomes. Split conformal prediction fixes that using a held-out
calibration set, with a finite-sample guarantee that does not depend on the model.

Two tools are provided, both fitted separately for each forecast horizon:

* :meth:`QuantileConformalizer.interval`: conformalized quantile regression
  (CQR, Romano et al. 2019). Widens or narrows an interval by the right amount so
  that marginal coverage is at least ``1 - alpha``.
* :meth:`QuantileConformalizer.quantiles`: one-sided recalibration of every
  quantile level, used for inventory decisions that need the whole distribution.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def _conformal_level(n: int, q: float) -> float:
    """Finite-sample corrected quantile level: ceil((n+1) q) / n, capped at 1."""
    return float(min(1.0, np.ceil((n + 1) * q) / max(n, 1)))


@dataclass
class QuantileConformalizer:
    taus: tuple[float, ...]
    # per horizon -> one-sided shift for each tau (K,)
    shifts: dict[int, np.ndarray] = field(default_factory=dict)
    # per (horizon, alpha) -> CQR score quantile
    cqr: dict[tuple[int, float], float] = field(default_factory=dict)
    alphas: tuple[float, ...] = (0.10, 0.20)

    def to_dict(self) -> dict:
        return {
            "taus": list(self.taus),
            "alphas": list(self.alphas),
            "shifts": {str(h): v.tolist() for h, v in self.shifts.items()},
            "cqr": [[h, a, q] for (h, a), q in self.cqr.items()],
        }

    @classmethod
    def from_dict(cls, d: dict) -> QuantileConformalizer:
        obj = cls(tuple(d["taus"]), alphas=tuple(d["alphas"]))
        obj.shifts = {int(h): np.asarray(v) for h, v in d["shifts"].items()}
        obj.cqr = {(int(h), float(a)): float(q) for h, a, q in d["cqr"]}
        return obj

    def _col(self, tau: float) -> int:
        return int(np.argmin(np.abs(np.asarray(self.taus) - tau)))

    def fit(self, pred: np.ndarray, y: np.ndarray, horizon: np.ndarray) -> QuantileConformalizer:
        """Fit on a calibration set. ``pred`` is ``(n, K)`` aligned with ``self.taus``."""
        taus = np.asarray(self.taus)
        for h in np.unique(horizon):
            m = horizon == h
            n = int(m.sum())
            resid = y[m, None] - pred[m]                        # (n, K) signed residuals
            self.shifts[int(h)] = np.array(
                [np.quantile(resid[:, k], _conformal_level(n, taus[k])) for k in range(len(taus))]
            )
            for a in self.alphas:
                lo, hi = self.taus[self._col(a / 2)], self.taus[self._col(1 - a / 2)]
                lo_p, hi_p = pred[m, self._col(lo)], pred[m, self._col(hi)]
                score = np.maximum(lo_p - y[m], y[m] - hi_p)
                self.cqr[(int(h), a)] = float(np.quantile(score, _conformal_level(n, 1 - a)))
        return self

    def quantiles(self, pred: np.ndarray, horizon: np.ndarray) -> np.ndarray:
        """One-sided recalibrated quantiles, kept non-negative and non-crossing."""
        out = pred.copy()
        for h, shift in self.shifts.items():
            m = horizon == h
            out[m] = pred[m] + shift[None, :]
        return np.sort(np.maximum(out, 0.0), axis=1)

    def interval(
        self, pred: np.ndarray, horizon: np.ndarray, alpha: float = 0.20
    ) -> tuple[np.ndarray, np.ndarray]:
        """CQR interval with marginal coverage of at least ``1 - alpha``."""
        lo_c = self._col(alpha / 2)
        hi_c = self._col(1 - alpha / 2)
        lo, hi = pred[:, lo_c].copy(), pred[:, hi_c].copy()
        for (h, a), q in self.cqr.items():
            if a != alpha:
                continue
            m = horizon == h
            lo[m] -= q
            hi[m] += q
        return np.maximum(lo, 0.0), np.maximum(hi, lo)


def pinball_loss(y: np.ndarray, q: np.ndarray, tau: float) -> float:
    d = y - q
    return float(np.mean(np.maximum(tau * d, (tau - 1) * d)))


def coverage(y: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> float:
    return float(np.mean((y >= lo) & (y <= hi)))
