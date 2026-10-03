"""Evaluation protocol and benchmark runner.

Protocol (all splits are disjoint in time; series are disjoint across roles):

    days   0 ................ 68 | 75 | 82 | 89 | 90 ..... 96
           training origins        valid  calib  last    test targets
           (targets <= 75)         origin origin known   (official eval split)
                                   (76-82) (83-89) day

* ``train`` series  : fit the models (origins 27, 29, ... 67; targets up to day 75)
* ``valid`` origin  : day 75, early stopping on the same series
* ``calib`` series  : day-82 origin, targets 83-89, used for conformal calibration
* ``test``  series  : never seen in training or calibration; the 7 days after
                      day 89 come from the dataset's official ``eval`` split

Forecast targets are *demand*, but demand is only observable on days that did not
stock out. Metrics are therefore reported two ways: against observed sales on all
days (what happened) and against observed sales on stockout-free days only (where
observed sales equal true demand).
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import recovery as rec
from .conformal import QuantileConformalizer, coverage, pinball_loss
from .data import Panel, build_panel, fetch
from .features import MIN_ORIGIN, RowSet, make_rows, stack_rows
from .inventory import (
    baseline_order,
    expected_cost,
    newsvendor_order,
    policy_outcome,
    quantile_from_knots,
)
from .model import ForecasterConfig, GlobalForecaster

LAST_KNOWN_DAY = 89
CALIB_ORIGIN = 82
VALID_ORIGIN = 75
TRAIN_ORIGINS = tuple(range(MIN_ORIGIN, 68, 2))


@dataclass
class BenchmarkConfig:
    n_train_series: int = 10_000
    n_calib_series: int = 5_000
    n_test_series: int | None = None      # None = all remaining series
    per_series_baseline_n: int = 300
    seed: int = 0
    min_share: float = 0.05
    model: ForecasterConfig | None = None


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #

def point_metrics(y: np.ndarray, pred: np.ndarray) -> dict:
    err = pred - y
    return {
        "wape": float(np.abs(err).sum() / y.sum()),
        "wpe": float(err.sum() / y.sum()),
        "mae": float(np.abs(err).mean()),
        "rmse": float(np.sqrt((err**2).mean())),
        "n": int(len(y)),
    }


def split_series(n: int, cfg: BenchmarkConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(cfg.seed)
    perm = rng.permutation(n)
    a = cfg.n_train_series
    b = a + cfg.n_calib_series
    train, calib, test = np.sort(perm[:a]), np.sort(perm[a:b]), np.sort(perm[b:])
    if cfg.n_test_series:
        test = test[: cfg.n_test_series]
    return train, calib, test


# --------------------------------------------------------------------------- #
# per-series baseline (the approach used by v1 of this project)
# --------------------------------------------------------------------------- #

def _series_features(panel: Panel, s: int, u: np.ndarray) -> np.ndarray:
    """Features for target days ``u`` of series ``s``: lags 7 and 14, known covariates, weekday."""
    y, d = panel.sales[s], panel.daily
    dow = (panel.dates[u].astype("datetime64[D]").astype(np.int64) + 3) % 7
    return np.stack(
        [
            y[u - 7],
            y[u - 14],
            d["discount"][s, u],
            d["activity_flag"][s, u],
            d["holiday_flag"][s, u],
            d["avg_temperature"][s, u],
            d["precpt"][s, u],
            dow,
        ],
        axis=1,
    )


def per_series_lightgbm(panel: Panel, series: np.ndarray, rows: RowSet) -> np.ndarray:
    """One small LightGBM model per series, trained on that series' own 90 days only."""
    import lightgbm as lgb

    preds = np.full(len(rows), np.nan, dtype=np.float32)
    days = np.arange(14, LAST_KNOWN_DAY + 1)
    for s in series:
        m = lgb.LGBMRegressor(
            n_estimators=100, num_leaves=8, learning_rate=0.05, min_child_samples=5, verbose=-1, n_jobs=1
        )
        m.fit(_series_features(panel, s, days), panel.sales[s][days])
        sel = rows.series == s
        u = LAST_KNOWN_DAY + rows.horizon[sel].astype(int)
        preds[sel] = np.maximum(m.predict(_series_features(panel, s, u)), 0)
    return preds


# --------------------------------------------------------------------------- #
# the benchmark
# --------------------------------------------------------------------------- #

def build_training_rows(
    panel: Panel, y: np.ndarray, series: np.ndarray, origins: tuple[int, ...]
) -> RowSet:
    return stack_rows([make_rows(panel, y, y, t, series_idx=series) for t in origins])


def run_benchmark(
    train_path: str | Path,
    eval_path: str | Path,
    out_dir: str | Path = "benchmarks/results",
    cfg: BenchmarkConfig | None = None,
    save_models_to: str | Path | None = None,
) -> dict:
    cfg = cfg or BenchmarkConfig()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results: dict = {"config": {k: v for k, v in cfg.__dict__.items() if k != "model"}}

    _log("loading panel")
    panel = build_panel(train_path, eval_path)
    train_view = panel.first_days(90)

    _log("validating latent demand recovery by simulated censoring")
    results["recovery_validation"] = rec.validate_recovery(train_view)

    _log("fitting hourly profile on days 0-60 and recovering demand")
    profile = rec.fit_profile(panel, slice(0, 61))
    demand, imputed = rec.recover_panel(panel, profile, cfg.min_share)
    results["demand_understatement"] = rec.demand_understatement(
        train_view, demand[:, :90]
    )
    results["share_days_imputed"] = float(imputed[:, :90].mean())

    tr, ca, te = split_series(panel.n_series, cfg)
    results["split"] = {"train": len(tr), "calib": len(ca), "test": len(te)}
    raw = panel.sales

    models: dict[str, GlobalForecaster] = {}
    for name, y in (("recovered", demand), ("raw", raw)):
        _log(f"building training rows ({name})")
        train_rows = build_training_rows(panel, y, tr, TRAIN_ORIGINS)
        valid_rows = make_rows(panel, y, y, VALID_ORIGIN, series_idx=tr)
        _log(f"fitting global forecaster ({name}): {len(train_rows):,} rows")
        t0 = time.time()
        models[name] = GlobalForecaster(cfg.model or ForecasterConfig()).fit(train_rows, valid_rows)
        _log(f"  done in {time.time() - t0:.0f}s, best iteration {models[name].point.best_iteration}")
    results["feature_importance_top10"] = dict(list(models["recovered"].feature_importance().items())[:10])

    # ---- conformal calibration (recovered-demand model) ------------------ #
    taus = np.asarray(models["recovered"].config.quantiles)
    calib_rows = make_rows(panel, demand, panel.sales, CALIB_ORIGIN, series_idx=ca)
    free_c = calib_rows.stockout_hours == 0
    qc = models["recovered"].predict_quantiles(calib_rows.X)
    conf = QuantileConformalizer(tuple(taus.tolist())).fit(
        qc[free_c], calib_rows.y_obs[free_c], calib_rows.horizon[free_c]
    )

    # ---- test ------------------------------------------------------------ #
    _log("scoring test series")
    test_rows = make_rows(panel, demand, panel.sales, LAST_KNOWN_DAY, series_idx=te)
    raw_rows = make_rows(panel, raw, raw, LAST_KNOWN_DAY, series_idx=te)
    y_obs = test_rows.y_obs
    free = test_rows.stockout_hours == 0
    h = test_rows.horizon

    preds = {
        "seasonal_naive_7d": test_rows.naive7,
        "moving_average_7d": test_rows.ma7,
        "global_lgbm_raw_sales": models["raw"].predict(raw_rows.X),
        "global_lgbm_recovered_demand": models["recovered"].predict(test_rows.X),
    }
    q_rec_raw = models["recovered"].predict_quantiles(test_rows.X)
    q_rec_cal = conf.quantiles(q_rec_raw, h)
    preds["global_lgbm_recovered_demand_q50"] = q_rec_cal[:, list(taus).index(0.5)]

    sample = np.random.default_rng(cfg.seed).choice(
        te, size=min(cfg.per_series_baseline_n, len(te)), replace=False
    )
    in_sample = np.isin(test_rows.series, sample)
    ps = per_series_lightgbm(panel, sample, test_rows)
    results["per_series_lightgbm_v1_style"] = {
        "n_series": int(len(sample)),
        "all_days": point_metrics(y_obs[in_sample], ps[in_sample]),
        "stockout_free_days": point_metrics(
            y_obs[in_sample & free], ps[in_sample & free]
        ),
        "global_lgbm_recovered_demand_same_series": {
            "all_days": point_metrics(y_obs[in_sample], preds["global_lgbm_recovered_demand"][in_sample]),
            "stockout_free_days": point_metrics(
                y_obs[in_sample & free], preds["global_lgbm_recovered_demand"][in_sample & free]
            ),
        },
    }

    table = {}
    for name, p in preds.items():
        table[name] = {
            "all_days_vs_observed_sales": point_metrics(y_obs, p),
            "stockout_free_days_vs_true_demand": point_metrics(y_obs[free], p[free]),
        }
    results["point_forecasts"] = table
    results["point_forecasts_by_horizon"] = {
        str(k): {
            n: point_metrics(y_obs[free & (h == k)], p[free & (h == k)])["wape"]
            for n, p in preds.items()
        }
        for k in range(1, 8)
    }

    # ---- probabilistic: calibration before / after ----------------------- #
    prob = {}
    for alpha in (0.20, 0.10):
        lo_c, hi_c = taus[list(taus).index(round(alpha / 2, 2))], taus[list(taus).index(round(1 - alpha / 2, 2))]
        lo0, hi0 = q_rec_raw[:, list(taus).index(lo_c)], q_rec_raw[:, list(taus).index(hi_c)]
        lo1, hi1 = conf.interval(q_rec_raw, h, alpha)
        prob[f"{int((1 - alpha) * 100)}pct_interval"] = {
            "nominal": 1 - alpha,
            "raw_quantile_coverage": coverage(y_obs[free], lo0[free], hi0[free]),
            "conformal_coverage": coverage(y_obs[free], lo1[free], hi1[free]),
            "raw_mean_width": float((hi0 - lo0)[free].mean()),
            "conformal_mean_width": float((hi1 - lo1)[free].mean()),
        }
    prob["pinball"] = {
        f"tau_{t:.2f}": {
            "raw": pinball_loss(y_obs[free], q_rec_raw[free, k], t),
            "recalibrated": pinball_loss(y_obs[free], q_rec_cal[free, k], t),
            "empirical_coverage_raw": float((y_obs[free] <= q_rec_raw[free, k]).mean()),
            "empirical_coverage_recalibrated": float((y_obs[free] <= q_rec_cal[free, k]).mean()),
        }
        for k, t in enumerate(taus)
    }
    results["probabilistic_forecasts"] = prob

    # ---- inventory policy simulation ------------------------------------- #
    _log("simulating inventory policies")
    demand_true = y_obs[free]
    q_free = q_rec_cal[free]
    sigma = np.maximum(test_rows.X[free][:, test_rows.columns.index("std7")], 0.05)
    mean = test_rows.ma7[free]
    frontier_q = []
    for sl in (0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95):
        order = quantile_from_knots(taus, q_free, sl)
        frontier_q.append({"service_level_target": sl, **policy_outcome(order, demand_true)})
    frontier_b = []
    for z in (-0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0, 1.28, 1.65, 2.0, 2.5):
        order = baseline_order(mean, sigma, z)
        frontier_b.append({"z": z, **policy_outcome(order, demand_true)})
    q_raw_free = models["recovered"].predict_quantiles(test_rows.X[free])
    frontier_u = []
    for sl in (0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95):
        order = quantile_from_knots(taus, q_raw_free, sl)
        frontier_u.append({"service_level_target": sl, **policy_outcome(order, demand_true)})

    def fill_at(frontier: list[dict], waste: float) -> float | None:
        pts = sorted((f["waste_rate"], f["fill_rate"]) for f in frontier)
        w, fr = zip(*pts)
        if waste < w[0] or waste > w[-1]:
            return None
        return float(np.interp(waste, w, fr))

    matched = {}
    for w in (0.10, 0.15, 0.20, 0.25, 0.30):
        matched[f"waste_{int(w * 100)}pct"] = {
            "conformal_quantile_policy": fill_at(frontier_q, w),
            "uncalibrated_quantile_policy": fill_at(frontier_u, w),
            "mean_plus_z_sigma_baseline": fill_at(frontier_b, w),
        }
    costs = {}
    for cu, co in ((1.0, 1.0), (3.0, 1.0), (9.0, 1.0)):
        order_n = newsvendor_order(taus, q_free, cu, co)
        costs[f"cu{cu:g}_co{co:g}"] = {
            "newsvendor_conformal": expected_cost(order_n, demand_true, cu, co),
            "best_baseline_z": min(
                (expected_cost(baseline_order(mean, sigma, z), demand_true, cu, co), z)
                for z in (-0.5, 0.0, 0.5, 1.0, 1.28, 1.65, 2.0, 2.5)
            ),
            "moving_average_only": expected_cost(baseline_order(mean, sigma, 0.0), demand_true, cu, co),
        }
    results["inventory"] = {
        "evaluated_on": "stockout-free test days (observed sales equal true demand)",
        "n_points": int(free.sum()),
        "frontier_conformal_quantile": frontier_q,
        "frontier_uncalibrated_quantile": frontier_u,
        "frontier_mean_plus_z_sigma": frontier_b,
        "fill_rate_at_matched_waste": matched,
        "newsvendor_cost_per_day": costs,
    }

    out = out_dir / "forecast_benchmark.json"
    out.write_text(json.dumps(results, indent=1, default=float))
    _log(f"wrote {out}")

    if save_models_to:
        _save_bundle(save_models_to, models["recovered"], conf, profile)
    return results


def _save_bundle(path: str | Path, model: GlobalForecaster, conf: QuantileConformalizer, profile: rec.Profile) -> None:
    from .bundle import save_models

    save_models(path, model, conf, profile)


if __name__ == "__main__":  # pragma: no cover
    tp, ep = fetch()
    run_benchmark(tp, ep)
