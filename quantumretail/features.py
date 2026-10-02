"""Feature construction for the global multi-horizon forecaster.

Setup (direct multi-horizon): at forecast *origin* ``t`` (the last day whose
sales are known) we predict day ``u = t + h`` for ``h = 1..7``. Features come
from three places and never look past ``t`` except for covariates the retailer
knows in advance (price discount, promotion flag, holiday, weather forecast):

* history up to ``t``: lags, rolling statistics, stockout intensity
* the target day ``u``: calendar and known-future covariates
* static attributes: store, product, category hierarchy
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .data import Panel

HORIZONS = tuple(range(1, 8))
MIN_ORIGIN = 27  # need 28 days of history for the longest window

CATEGORICAL = [
    "store_id",
    "product_id",
    "city_id",
    "management_group_id",
    "first_category_id",
    "second_category_id",
    "third_category_id",
]


@dataclass
class RowSet:
    """Feature matrix plus everything needed to score it."""

    X: np.ndarray                 # (n, k) float32
    columns: list[str]
    series: np.ndarray            # (n,) series index into the panel
    origin: np.ndarray            # (n,)
    horizon: np.ndarray           # (n,)
    y: np.ndarray                 # (n,) training target (recovered or raw demand)
    y_obs: np.ndarray             # (n,) observed sales on the target day
    stockout_hours: np.ndarray    # (n,) stockout hours on the target day
    naive7: np.ndarray            # (n,) observed sales 7 days before target day
    ma7: np.ndarray               # (n,) mean observed sales over the 7 days up to origin

    def __len__(self) -> int:
        return len(self.y)


def _dow(dates: np.ndarray) -> np.ndarray:
    days = dates.astype("datetime64[D]").astype(np.int64)
    return ((days + 3) % 7).astype(np.int8)  # Monday = 0


def _roll(a: np.ndarray, t: int, k: int) -> np.ndarray:
    return a[:, t - k + 1 : t + 1]


def make_rows(
    panel: Panel,
    y_feat: np.ndarray,
    y_target: np.ndarray,
    origin: int,
    horizons: tuple[int, ...] = HORIZONS,
    series_idx: np.ndarray | None = None,
) -> RowSet:
    """Build one block of rows (every series x every horizon) for a given origin.

    Args:
        panel: dense panel; must contain days up to ``origin + max(horizons)``.
        y_feat: ``(S, T)`` demand series used to build history features
            (recovered demand, or raw sales for the ablation).
        y_target: ``(S, T)`` series used as the training target.
        origin: index of the last known day.
        series_idx: optional subset of series to build rows for.
    """
    t = origin
    if t < MIN_ORIGIN:
        raise ValueError(f"origin must be >= {MIN_ORIGIN}")
    if t + max(horizons) >= panel.n_days:
        raise ValueError("panel does not extend far enough for the requested horizons")

    sel = np.arange(panel.n_series) if series_idx is None else np.asarray(series_idx)
    y = y_feat[sel]
    obs = panel.sales[sel]
    so = panel.stockout_hours[sel].astype(np.float32)
    d = {k: v[sel] for k, v in panel.daily.items()}
    dow = _dow(panel.dates)

    # ---- origin-level (history) features -------------------------------- #
    with np.errstate(all="ignore"):
        base = {
            "y_t": y[:, t],
            "y_t1": y[:, t - 1],
            "y_t2": y[:, t - 2],
            "mean3": _roll(y, t, 3).mean(1),
            "mean7": _roll(y, t, 7).mean(1),
            "mean14": _roll(y, t, 14).mean(1),
            "mean28": _roll(y, t, 28).mean(1),
            "std7": _roll(y, t, 7).std(1),
            "std14": _roll(y, t, 14).std(1),
            "max7": _roll(y, t, 7).max(1),
            "zero_frac14": (_roll(y, t, 14) <= 0).mean(1),
            "obs_mean7": _roll(obs, t, 7).mean(1),
            "so_t": so[:, t],
            "so_mean7": _roll(so, t, 7).mean(1),
            "so_mean28": _roll(so, t, 28).mean(1),
            "so_days7": (_roll(so, t, 7) > 0).mean(1),
            "act_t": d["activity_flag"][:, t],
            "act_mean7": _roll(d["activity_flag"], t, 7).mean(1),
            "disc_mean7": _roll(d["discount"], t, 7).mean(1),
        }
        base["trend"] = base["mean7"] / np.maximum(base["mean28"], 1e-3)
        base["understatement7"] = (
            base["mean7"] - base["obs_mean7"]
        ) / np.maximum(base["mean7"], 1e-3)

    cols: list[str] = []
    blocks: list[np.ndarray] = []
    meta = {k: [] for k in ("series", "origin", "horizon", "y", "y_obs", "so", "naive7", "ma7")}

    ma7 = _roll(obs, t, 7).mean(1)
    for h in horizons:
        u = t + h
        feats = dict(base)
        feats["h"] = np.full(len(sel), h, dtype=np.float32)
        feats["dow"] = np.full(len(sel), dow[u], dtype=np.float32)
        for lag in (7, 14, 21, 28):
            feats[f"y_lag{lag}"] = y[:, u - lag] if u - lag >= 0 else np.full(len(sel), np.nan, np.float32)
        feats["same_dow_mean"] = np.nanmean(
            np.stack([feats[f"y_lag{k}"] for k in (7, 14, 21, 28)]), axis=0
        )
        feats["discount"] = d["discount"][:, u]
        feats["activity"] = d["activity_flag"][:, u]
        feats["holiday"] = d["holiday_flag"][:, u]
        feats["precpt"] = d["precpt"][:, u]
        feats["temp"] = d["avg_temperature"][:, u]
        feats["humidity"] = d["avg_humidity"][:, u]
        feats["wind"] = d["avg_wind_level"][:, u]
        feats["discount_rel"] = d["discount"][:, u] / np.maximum(base["disc_mean7"], 1e-3)
        for name in CATEGORICAL:
            feats[name] = panel.static[name][sel].astype(np.float32)

        if not cols:
            cols = list(feats.keys())
        blocks.append(np.stack([np.asarray(feats[c], dtype=np.float32) for c in cols], axis=1))
        meta["series"].append(sel)
        meta["origin"].append(np.full(len(sel), t, dtype=np.int16))
        meta["horizon"].append(np.full(len(sel), h, dtype=np.int8))
        meta["y"].append(y_target[sel, u])
        meta["y_obs"].append(panel.sales[sel, u])
        meta["so"].append(panel.stockout_hours[sel, u])
        meta["naive7"].append(panel.sales[sel, u - 7])
        meta["ma7"].append(ma7)

    cat = {k: np.concatenate(v) for k, v in meta.items()}
    return RowSet(
        X=np.concatenate(blocks, axis=0),
        columns=cols,
        series=cat["series"],
        origin=cat["origin"],
        horizon=cat["horizon"],
        y=cat["y"].astype(np.float32),
        y_obs=cat["y_obs"].astype(np.float32),
        stockout_hours=cat["so"],
        naive7=cat["naive7"].astype(np.float32),
        ma7=cat["ma7"].astype(np.float32),
    )


def stack_rows(parts: list[RowSet]) -> RowSet:
    """Concatenate row blocks built for different origins."""
    cols = parts[0].columns
    return RowSet(
        X=np.concatenate([p.X for p in parts]),
        columns=cols,
        series=np.concatenate([p.series for p in parts]),
        origin=np.concatenate([p.origin for p in parts]),
        horizon=np.concatenate([p.horizon for p in parts]),
        y=np.concatenate([p.y for p in parts]),
        y_obs=np.concatenate([p.y_obs for p in parts]),
        stockout_hours=np.concatenate([p.stockout_hours for p in parts]),
        naive7=np.concatenate([p.naive7 for p in parts]),
        ma7=np.concatenate([p.ma7 for p in parts]),
    )
