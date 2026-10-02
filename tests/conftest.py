"""Shared synthetic data: a small panel with a known hourly demand profile."""
import numpy as np
import pytest

from quantumretail.data import DAILY_COLS, STATIC_COLS, Panel

PROFILE = np.array(
    [0, 0, 0, 0, 0, 0, 0.02, 0.06, 0.12, 0.10, 0.06, 0.05, 0.05, 0.05, 0.06, 0.08,
     0.12, 0.10, 0.06, 0.04, 0.02, 0.01, 0.0, 0.0]
)
PROFILE = PROFILE / PROFILE.sum()


def make_panel(n_series: int = 80, n_days: int = 97, seed: int = 0, stockout_rate: float = 0.35) -> Panel:
    rng = np.random.default_rng(seed)
    dates = (np.datetime64("2024-03-28") + np.arange(n_days)).astype("datetime64[D]")
    dow = (dates.astype(np.int64) + 3) % 7

    level = rng.uniform(1.0, 8.0, size=n_series)[:, None]
    weekly = np.where(dow >= 5, 1.3, 1.0)[None, :]
    activity = (rng.random((n_series, n_days)) < 0.3).astype(np.float32)
    demand = level * weekly * (1 + 0.4 * activity) * rng.uniform(0.85, 1.15, (n_series, n_days))

    hours = demand[..., None] * PROFILE[None, None, :]
    stock = np.zeros((n_series, n_days, 24), dtype=np.uint8)
    hit = rng.random((n_series, n_days)) < stockout_rate
    cut = rng.integers(8, 21, size=(n_series, n_days))
    mask = (np.arange(24)[None, None, :] >= cut[..., None]) & hit[..., None]
    stock[mask] = 1
    obs_hours = np.where(mask, 0.0, hours).astype(np.float32)

    static = {c: rng.integers(0, 5, n_series) for c in STATIC_COLS}
    static["store_id"] = np.arange(n_series)
    static["product_id"] = rng.integers(0, 6, n_series)
    daily = {
        "discount": np.ones((n_series, n_days), np.float32) - 0.1 * activity,
        "holiday_flag": (rng.random((n_series, n_days)) < 0.05).astype(np.float32),
        "activity_flag": activity,
        "precpt": rng.random((n_series, n_days)).astype(np.float32),
        "avg_temperature": (20 + 5 * rng.standard_normal((n_series, n_days))).astype(np.float32),
        "avg_humidity": rng.uniform(40, 90, (n_series, n_days)).astype(np.float32),
        "avg_wind_level": rng.uniform(0, 3, (n_series, n_days)).astype(np.float32),
    }
    assert set(daily) == set(DAILY_COLS)
    return Panel(
        static=static,
        daily=daily,
        sales=obs_hours.sum(-1),
        hours=obs_hours,
        stock=stock,
        stockout_hours=stock[:, :, 6:22].sum(-1).astype(np.int16),
        dates=dates,
    )


@pytest.fixture(scope="session")
def panel() -> Panel:
    return make_panel()
