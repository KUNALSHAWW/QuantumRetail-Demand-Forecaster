"""Latent demand recovery for stockout-censored sales.

When a product sells out, observed sales understate demand. FreshRetailNet-50K
records the stockout status of every hour, so we know *which* hours are censored
and can reconstruct what would probably have sold.

Three estimators are provided and compared:

* ``legacy``   : the original project formula, scale observed sales by
                 ``16 / (16 - stockout_hours)``. It assumes demand is spread
                 uniformly over the 16 selling hours, which is wrong because
                 demand peaks in the morning and late afternoon.
* ``profile``  : a ratio estimator. Estimate each product's typical hourly share
                 of daily demand from days that never stocked out, then divide the
                 sales observed in in-stock hours by the share of demand those
                 hours normally carry.
* ``none``     : raw observed sales (the do-nothing baseline).

Because true demand is unobservable on stockout days, the estimators are
validated with a controlled simulation (the protocol used in the FreshRetailNet
paper): take days that never stocked out, where observed sales equal true demand,
censor them with realistic stockout patterns, and measure how well each method
reconstructs the known total. See :func:`validate_recovery`.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .data import SELLING_WINDOW, Panel

OPEN, CLOSE = SELLING_WINDOW
WINDOW_HOURS = CLOSE - OPEN  # 16


@dataclass
class Profile:
    """Hourly share of daily demand, per product, with a global fallback."""

    product_ids: np.ndarray   # (P,) sorted unique product ids
    shares: np.ndarray        # (P, 24), each row sums to 1
    global_shares: np.ndarray  # (24,)
    support_days: np.ndarray  # (P,) number of fully stocked days behind each row

    def rows_for(self, product_ids: np.ndarray) -> np.ndarray:
        """Return the (N, 24) profile for each requested product id."""
        pos = np.searchsorted(self.product_ids, product_ids)
        pos = np.clip(pos, 0, len(self.product_ids) - 1)
        known = self.product_ids[pos] == product_ids
        out = np.tile(self.global_shares, (len(product_ids), 1))
        out[known] = self.shares[pos[known]]
        return out


def fit_profile(panel: Panel, days: slice | None = None, prior_days: float = 20.0) -> Profile:
    """Estimate hourly demand shares from days that never stocked out.

    A product's share vector is shrunk towards the global profile in proportion
    to how few fully stocked days support it.
    """
    days = days or slice(0, panel.n_days)
    hours = panel.hours[:, days]
    sales = panel.sales[:, days]
    full = (panel.stockout_hours[:, days] == 0) & (sales > 0)

    pid = np.broadcast_to(panel.static["product_id"][:, None], sales.shape)[full]
    h = hours[full].astype(np.float64)
    share = h / h.sum(axis=1, keepdims=True)

    uniq, inv = np.unique(pid, return_inverse=True)
    sums = np.zeros((len(uniq), 24))
    np.add.at(sums, inv, share)
    n = np.bincount(inv, minlength=len(uniq)).astype(np.float64)
    glob = share.mean(axis=0)
    glob = glob / glob.sum()
    mean = sums / np.maximum(n[:, None], 1)
    w = (n / (n + prior_days))[:, None]
    blended = w * mean + (1 - w) * glob
    blended /= blended.sum(axis=1, keepdims=True)
    return Profile(uniq, blended, glob, n.astype(np.int32))


def recover_none(sales: np.ndarray) -> np.ndarray:
    return sales.astype(np.float32)


def recover_legacy(sales: np.ndarray, stockout_hours: np.ndarray) -> np.ndarray:
    """The original formula: ``sales * 16 / (16 - stockout_hours)``.

    Days that were out of stock for the whole window keep their observed value.
    """
    k = stockout_hours.astype(np.float32)
    den = np.maximum(WINDOW_HOURS - k, 1.0)
    out = sales * (WINDOW_HOURS / den)
    out = np.where(k >= WINDOW_HOURS, sales, out)
    return out.astype(np.float32)


def recover_profile(
    hours: np.ndarray,
    stock: np.ndarray,
    shares: np.ndarray,
    min_share: float = 0.10,
) -> tuple[np.ndarray, np.ndarray]:
    """Ratio estimator of daily demand.

    Args:
        hours: ``(..., 24)`` observed hourly sales.
        stock: ``(..., 24)`` stockout flags (1 = out of stock).
        shares: ``(..., 24)`` hourly demand shares for each row.
        min_share: if the in-stock hours carry less than this share of normal
            daily demand there is too little signal and the value is returned
            as NaN so the caller can impute it from neighbouring days.

    Returns:
        ``(demand, unreliable)`` where ``unreliable`` marks NaN rows.
    """
    in_stock = 1.0 - stock.astype(np.float32)
    covered = (in_stock * shares).sum(axis=-1)
    observed_in_stock = (in_stock * hours).sum(axis=-1)
    observed_total = hours.sum(axis=-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        est = observed_in_stock / covered
    est = np.maximum(est, observed_total)          # demand is never below what sold
    unreliable = covered < min_share
    est = np.where(unreliable, np.nan, est)
    return est.astype(np.float32), unreliable


def impute_unreliable(demand: np.ndarray, window: int = 7) -> np.ndarray:
    """Fill NaN days with the median of valid neighbours on the same series."""
    s, t = demand.shape
    pad = np.pad(demand, ((0, 0), (window, window)), constant_values=np.nan)
    win = np.lib.stride_tricks.sliding_window_view(pad, 2 * window + 1, axis=1)
    with np.errstate(all="ignore"):
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            med = np.nanmedian(win, axis=-1)
            series_med = np.nanmedian(demand, axis=1, keepdims=True)
    med = np.where(np.isnan(med), series_med, med)
    med = np.where(np.isnan(med), 0.0, med)
    return np.where(np.isnan(demand), med, demand).astype(np.float32)


def recover_panel(
    panel: Panel,
    profile: Profile,
    min_share: float = 0.10,
) -> tuple[np.ndarray, np.ndarray]:
    """Recover latent daily demand for a whole panel.

    Returns ``(demand, imputed_mask)`` of shape ``(S, T)``.
    """
    s, t = panel.sales.shape
    prow = profile.rows_for(panel.static["product_id"])          # (S, 24)
    shares = np.broadcast_to(prow[:, None, :], (s, t, 24))
    est, unreliable = recover_profile(panel.hours, panel.stock, shares, min_share)
    return impute_unreliable(est), unreliable


# --------------------------------------------------------------------------- #
# Validation by controlled censoring
# --------------------------------------------------------------------------- #

def _first_stockout_hours(panel: Panel) -> np.ndarray:
    """Empirical distribution of the hour at which stock first runs out."""
    st = panel.stock[:, :, OPEN:CLOSE]
    any_out = st.any(axis=-1)
    first = st.argmax(axis=-1) + OPEN
    return first[any_out]


def wape(est: np.ndarray, truth: np.ndarray) -> float:
    return float(np.abs(est - truth).sum() / truth.sum())


def wpe(est: np.ndarray, truth: np.ndarray) -> float:
    """Weighted percentage error (signed bias): negative means under-estimation."""
    return float((est - truth).sum() / truth.sum())


def validate_recovery(
    panel: Panel,
    fit_days: slice = slice(0, 60),
    eval_days: slice = slice(60, None),
    n_samples: int = 300_000,
    min_shares: tuple[float, ...] = (0.05, 0.10, 0.20),
    seed: int = 0,
) -> dict:
    """Controlled-censoring validation of the recovery estimators.

    Fully stocked days (observed sales == true demand) from ``eval_days`` are
    censored with stockout times drawn from the empirical distribution. The
    hourly profile is fitted on ``fit_days`` only, so no day is used both to
    fit and to test.
    """
    rng = np.random.default_rng(seed)
    profile = fit_profile(panel, fit_days)

    hours = panel.hours[:, eval_days]
    sales = panel.sales[:, eval_days]
    full = (panel.stockout_hours[:, eval_days] == 0) & (sales > 0)
    si, ti = np.nonzero(full)
    if len(si) > n_samples:
        pick = rng.choice(len(si), n_samples, replace=False)
        si, ti = si[pick], ti[pick]

    true_hours = hours[si, ti].astype(np.float64)               # (N, 24)
    truth = true_hours.sum(axis=1)

    first = _first_stockout_hours(panel)
    first = first[(first >= OPEN + 1) & (first < CLOSE)]        # partial-day stockouts
    cut = rng.choice(first, size=len(si))

    hour_idx = np.arange(24)[None, :]
    censored_mask = hour_idx >= cut[:, None]
    obs_hours = np.where(censored_mask, 0.0, true_hours)
    stock = censored_mask.astype(np.uint8)
    obs_total = obs_hours.sum(axis=1)
    stock_hours = np.clip(CLOSE - cut, 0, WINDOW_HOURS)

    shares = profile.rows_for(panel.static["product_id"][si])

    results = {
        "n_samples": int(len(si)),
        "fit_days": [fit_days.start or 0, fit_days.stop],
        "mean_stockout_hours_simulated": float(stock_hours.mean()),
        "none": {"wape": wape(obs_total, truth), "wpe": wpe(obs_total, truth)},
    }
    legacy = recover_legacy(obs_total.astype(np.float32), stock_hours)
    results["legacy"] = {"wape": wape(legacy, truth), "wpe": wpe(legacy, truth)}

    for ms in min_shares:
        est, bad = recover_profile(obs_hours.astype(np.float32), stock, shares, ms)
        # unreliable rows fall back to the observed value in this isolated test
        est = np.where(bad, obs_total, est)
        results[f"profile_min_share_{ms:.2f}"] = {
            "wape": wape(est, truth),
            "wpe": wpe(est, truth),
            "share_unreliable": float(bad.mean()),
        }

    # breakdown by how early the product sold out
    bins = [(7, 10), (10, 14), (14, 18), (18, 22)]
    best_key = min(
        (k for k in results if k.startswith("profile_")), key=lambda k: results[k]["wape"]
    )
    ms_best = float(best_key.rsplit("_", 1)[1])
    est_best, bad = recover_profile(obs_hours.astype(np.float32), stock, shares, ms_best)
    est_best = np.where(bad, obs_total, est_best)
    by_bin = {}
    for lo, hi in bins:
        m = (cut >= lo) & (cut < hi)
        if m.sum() == 0:
            continue
        by_bin[f"stockout_from_{lo:02d}:00_to_{hi:02d}:00"] = {
            "n": int(m.sum()),
            "none_wape": wape(obs_total[m], truth[m]),
            "legacy_wape": wape(legacy[m], truth[m]),
            "profile_wape": wape(est_best[m], truth[m]),
            "legacy_wpe": wpe(legacy[m], truth[m]),
            "profile_wpe": wpe(est_best[m], truth[m]),
        }
    results["best_profile_setting"] = best_key
    results["by_stockout_onset"] = by_bin
    return results


def demand_understatement(panel: Panel, demand: np.ndarray) -> dict:
    """How much real demand do raw sales miss on days that stocked out?"""
    so = panel.stockout_hours > 0
    obs = panel.sales[so].sum()
    rec = demand[so].sum()
    full_day = panel.stockout_hours >= WINDOW_HOURS
    return {
        "share_of_days_with_stockout": float(so.mean()),
        "share_of_days_out_all_day": float(full_day.mean()),
        "unobserved_share_of_demand_on_stockout_days": float(1 - obs / rec),
        "unobserved_share_of_total_demand": float(1 - panel.sales.sum() / demand.sum()),
    }
