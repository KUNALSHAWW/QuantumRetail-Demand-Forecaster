"""From forecasts to decisions: how much should the store stock?

For a perishable product the order quantity trades two costs: lost margin when
the shelf empties (underage cost ``cu``) and waste when stock is left over
(overage cost ``co``). The classical newsvendor result says the cost-minimising
order is the demand quantile at the *critical fractile* ``cu / (cu + co)``.
Having calibrated quantiles of demand makes that decision directly computable.
"""
from __future__ import annotations

import numpy as np


def quantile_from_knots(taus: np.ndarray, values: np.ndarray, tau: float) -> np.ndarray:
    """Evaluate a quantile function at ``tau`` by linear interpolation.

    Args:
        taus: ``(K,)`` increasing quantile levels the model was trained at.
        values: ``(n, K)`` quantile forecasts (non-decreasing along axis 1).
        tau: requested level; clamped to the covered range.
    """
    taus = np.asarray(taus, dtype=float)
    tau = float(np.clip(tau, taus[0], taus[-1]))
    j = int(np.searchsorted(taus, tau))
    if taus[j] == tau or j == 0:
        return values[:, j].copy()
    w = (tau - taus[j - 1]) / (taus[j] - taus[j - 1])
    return (1 - w) * values[:, j - 1] + w * values[:, j]


def critical_fractile(underage_cost: float, overage_cost: float) -> float:
    return underage_cost / (underage_cost + overage_cost)


def newsvendor_order(
    taus: np.ndarray, values: np.ndarray, underage_cost: float, overage_cost: float
) -> np.ndarray:
    """Cost-optimal order quantity at the critical fractile."""
    return quantile_from_knots(taus, values, critical_fractile(underage_cost, overage_cost))


def baseline_order(mean: np.ndarray, sigma: np.ndarray, z: float) -> np.ndarray:
    """The textbook baseline: forecast mean plus ``z`` standard deviations of safety stock."""
    return np.maximum(mean + z * sigma, 0.0)


def policy_outcome(order: np.ndarray, demand: np.ndarray) -> dict:
    """Score an ordering policy against realised demand.

    * fill rate  : share of demand that was served
    * waste rate : share of stock that went unsold
    * stockout rate: share of days on which demand exceeded stock
    """
    sold = np.minimum(order, demand)
    total_d = float(demand.sum())
    total_o = float(order.sum())
    return {
        "fill_rate": float(sold.sum() / total_d),
        "waste_rate": float((order - sold).sum() / total_o) if total_o > 0 else 0.0,
        "stockout_rate": float(np.mean(order < demand)),
        "avg_stock_per_unit_demand": total_o / total_d,
    }


def expected_cost(order: np.ndarray, demand: np.ndarray, underage_cost: float, overage_cost: float) -> float:
    """Mean realised newsvendor cost per day."""
    short = np.maximum(demand - order, 0.0)
    over = np.maximum(order - demand, 0.0)
    return float(np.mean(underage_cost * short + overage_cost * over))
