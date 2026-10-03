import numpy as np

from quantumretail import recovery as rec
from quantumretail.data import SELLING_WINDOW

from .conftest import PROFILE, make_panel


def test_profile_estimator_is_exact_when_profile_is_known():
    demand = np.array([4.0, 10.0, 2.5])
    hours = demand[:, None] * PROFILE[None, :]
    stock = np.zeros_like(hours, dtype=np.uint8)
    stock[:, 14:] = 1  # sold out from 14:00
    obs = np.where(stock == 1, 0.0, hours).astype(np.float32)
    est, bad = rec.recover_profile(obs, stock, np.tile(PROFILE, (3, 1)), min_share=0.05)
    assert not bad.any()
    np.testing.assert_allclose(est, demand, rtol=1e-4)


def test_no_stockout_returns_observed_sales():
    demand = np.array([3.0])
    hours = demand[:, None] * PROFILE[None, :]
    est, _ = rec.recover_profile(
        hours.astype(np.float32), np.zeros_like(hours, np.uint8), PROFILE[None], 0.05
    )
    np.testing.assert_allclose(est, demand, rtol=1e-5)


def test_estimate_never_below_observed():
    hours = np.full((1, 24), 0.5, dtype=np.float32)
    stock = np.zeros((1, 24), dtype=np.uint8)
    stock[0, 22:] = 1
    est, _ = rec.recover_profile(hours, stock, PROFILE[None], 0.01)
    assert est[0] >= hours.sum()


def test_too_little_signal_is_flagged_and_imputed():
    hours = np.zeros((1, 24), dtype=np.float32)
    stock = np.ones((1, 24), dtype=np.uint8)  # out of stock all day
    est, bad = rec.recover_profile(hours, stock, PROFILE[None], 0.10)
    assert bad[0] and np.isnan(est[0])
    series = np.array([[2.0, 2.0, np.nan, 2.0, 2.0]], dtype=np.float32)
    assert rec.impute_unreliable(series, window=2)[0, 2] == 2.0


def test_legacy_formula():
    sales = np.array([5.0, 5.0], dtype=np.float32)
    k = np.array([8, 16])
    out = rec.recover_legacy(sales, k)
    assert out[0] == 10.0  # 16 / (16 - 8) = 2x
    assert out[1] == 5.0  # out all day: nothing to scale


def test_fitted_profile_recovers_true_shape():
    panel = make_panel(seed=3)
    prof = rec.fit_profile(panel)
    assert np.allclose(prof.global_shares, PROFILE, atol=0.01)
    assert np.allclose(prof.shares.sum(axis=1), 1.0)


def test_unknown_product_falls_back_to_global_profile():
    panel = make_panel(seed=1)
    prof = rec.fit_profile(panel)
    rows = prof.rows_for(np.array([10_000]))
    np.testing.assert_allclose(rows[0], prof.global_shares)


def test_profile_beats_legacy_under_controlled_censoring():
    panel = make_panel(n_series=300, stockout_rate=0.15, seed=5)
    res = rec.validate_recovery(panel, slice(0, 50), slice(50, None), n_samples=20_000)
    best = res[res["best_profile_setting"]]
    assert best["wape"] < res["legacy"]["wape"] < res["none"]["wape"]
    assert abs(best["wpe"]) < 0.05
    assert res["none"]["wpe"] < -0.1  # doing nothing under-estimates demand


def test_validation_never_fits_on_evaluation_days():
    panel = make_panel(seed=2)
    res = rec.validate_recovery(panel, slice(0, 40), slice(40, None), n_samples=2_000)
    assert res["fit_days"] == [0, 40]


def test_recover_panel_never_goes_below_observed_sales():
    panel = make_panel(seed=4)
    prof = rec.fit_profile(panel, slice(0, 60))
    demand, _ = rec.recover_panel(panel, prof)
    assert demand.shape == panel.sales.shape
    assert (demand >= panel.sales - 1e-4).all()


def test_selling_window_constants():
    assert SELLING_WINDOW == (6, 22)
