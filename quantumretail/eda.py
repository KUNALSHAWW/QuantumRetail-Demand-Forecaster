"""Reproducible exploratory estimates (replacing hand-quoted percentages).

Every number is a *within-series* contrast, so differences between stores and
products (a big store always sells more) do not masquerade as effects. Demand is
the recovered latent demand, because raw sales are censored on stockout days and
censoring is correlated with promotions and weekends (busy days sell out), which
would bias the raw estimates.
"""
from __future__ import annotations

import numpy as np

from .data import Panel


def _ratio_effect(demand: np.ndarray, flag: np.ndarray) -> dict:
    """Mean demand when ``flag`` is on vs off, using series that have both."""
    on = flag > 0.5
    off = ~on
    n_on, n_off = on.sum(1), off.sum(1)
    ok = (n_on >= 3) & (n_off >= 3)
    m_on = (demand * on).sum(1)[ok] / n_on[ok]
    m_off = (demand * off).sum(1)[ok] / n_off[ok]
    return {
        "lift": float(m_on.sum() / m_off.sum() - 1),
        "median_series_lift": float(np.median(m_on / np.maximum(m_off, 1e-6)) - 1),
        "series_used": int(ok.sum()),
    }


def _within_slope(demand: np.ndarray, x: np.ndarray) -> float:
    """Series fixed-effects OLS slope of log-demand on x (percent change per unit)."""
    ly = np.log(demand + 0.05)
    ly = ly - ly.mean(1, keepdims=True)
    xx = x - x.mean(1, keepdims=True)
    return float((xx * ly).sum() / (xx**2).sum())


def effects(panel: Panel, demand: np.ndarray, raw: np.ndarray | None = None, days: int = 90) -> dict:
    d = {k: v[:, :days] for k, v in panel.daily.items()}
    y = demand[:, :days]
    dow = ((panel.dates[:days].astype("datetime64[D]").astype(np.int64) + 3) % 7)
    weekend = np.broadcast_to((dow >= 5)[None, :], y.shape).astype(np.float32)

    out = {
        "weekend_vs_weekday": _ratio_effect(y, weekend),
        "promotion_active_vs_not": _ratio_effect(y, d["activity_flag"]),
        "holiday_vs_not": _ratio_effect(y, d["holiday_flag"]),
        "discount_elasticity": {
            "pct_change_in_demand_per_1pct_price_cut": float(
                -_within_slope(y, np.log(np.maximum(d["discount"], 0.05)))
            ),
            "note": "series fixed-effects slope of log demand on log price factor, sign flipped so positive means demand rises when price falls",
        },
        "temperature_per_degree_c": {"pct_change_in_demand": float(np.expm1(_within_slope(y, d["avg_temperature"])))},
        "precipitation_per_unit": {"pct_change_in_demand": float(np.expm1(_within_slope(y, d["precpt"])))},
    }
    if raw is not None:
        r = raw[:, :days]
        out["promotion_active_vs_not_raw_sales"] = _ratio_effect(r, d["activity_flag"])
        out["weekend_vs_weekday_raw_sales"] = _ratio_effect(r, weekend)
    return out
