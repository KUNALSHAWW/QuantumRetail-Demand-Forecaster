"""Build the compact artefacts that let the app run without downloading the dataset.

The demo model is deliberately small (a few MB). The 400 demo series are drawn
from the *test* pool, so the app only ever shows out-of-sample forecasts.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from . import recovery as rec
from .backtest import (
    CALIB_ORIGIN,
    TRAIN_ORIGINS,
    VALID_ORIGIN,
    BenchmarkConfig,
    build_training_rows,
    split_series,
)
from .bundle import save_models, save_panel
from .conformal import QuantileConformalizer
from .data import build_panel
from .features import make_rows
from .model import ForecasterConfig, GlobalForecaster


def build_demo_bundle(
    train_path: str | Path,
    eval_path: str | Path,
    models_dir: str | Path = "models/qr_v2",
    panel_path: str | Path = "data/demo/demo_panel.npz",
    n_demo: int = 400,
    n_train: int = 6000,
    n_calib: int = 4000,
    seed: int = 0,
) -> None:
    panel = build_panel(train_path, eval_path)
    cfg = BenchmarkConfig(n_train_series=n_train, n_calib_series=n_calib, seed=seed)
    tr, ca, te = split_series(panel.n_series, cfg)

    profile = rec.fit_profile(panel, slice(0, 61))
    demand, _ = rec.recover_panel(panel, profile, 0.05)

    train_rows = build_training_rows(panel, demand, tr, TRAIN_ORIGINS)
    valid_rows = make_rows(panel, demand, demand, VALID_ORIGIN, series_idx=tr)
    mcfg = ForecasterConfig(
        n_estimators_point=300, n_estimators_quantile=200, num_leaves=31, min_data_in_leaf=100
    )
    model = GlobalForecaster(mcfg).fit(train_rows, valid_rows)

    calib_rows = make_rows(panel, demand, panel.sales, CALIB_ORIGIN, series_idx=ca)
    free = calib_rows.stockout_hours == 0
    q = model.predict_quantiles(calib_rows.X)
    conf = QuantileConformalizer(tuple(mcfg.quantiles)).fit(
        q[free], calib_rows.y_obs[free], calib_rows.horizon[free]
    )

    save_models(models_dir, model, conf, profile)
    rng = np.random.default_rng(seed + 1)
    demo_idx = np.sort(rng.choice(te, size=min(n_demo, len(te)), replace=False))
    save_panel(panel_path, panel.subset(demo_idx))
