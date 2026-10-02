import numpy as np
import pytest

from quantumretail import recovery as rec
from quantumretail.backtest import build_training_rows
from quantumretail.bundle import load_models, load_panel, save_models, save_panel
from quantumretail.conformal import QuantileConformalizer
from quantumretail.features import make_rows
from quantumretail.model import ForecasterConfig, GlobalForecaster
from quantumretail.service import DemandService

from .conftest import make_panel

CFG = ForecasterConfig(n_estimators_point=40, n_estimators_quantile=30, num_leaves=7, min_data_in_leaf=20)


@pytest.fixture(scope="module")
def trained():
    panel = make_panel(n_series=120, seed=11)
    profile = rec.fit_profile(panel, slice(0, 61))
    demand, _ = rec.recover_panel(panel, profile, 0.05)
    tr = np.arange(0, 80)
    train = build_training_rows(panel, demand, tr, tuple(range(27, 68, 4)))
    valid = make_rows(panel, demand, demand, 75, series_idx=tr)
    model = GlobalForecaster(CFG).fit(train, valid)
    ca = np.arange(80, 120)
    calib = make_rows(panel, demand, panel.sales, 82, series_idx=ca)
    free = calib.stockout_hours == 0
    conf = QuantileConformalizer(tuple(CFG.quantiles)).fit(
        model.predict_quantiles(calib.X)[free], calib.y_obs[free], calib.horizon[free]
    )
    return panel, profile, model, conf


def test_quantiles_sorted_and_nonnegative(trained):
    panel, _, model, _ = trained
    rows = make_rows(panel, panel.sales, panel.sales, 70, series_idx=np.arange(10))
    q = model.predict_quantiles(rows.X)
    assert q.shape == (len(rows), len(CFG.quantiles))
    assert (np.diff(q, axis=1) >= 0).all() and (q >= 0).all()
    assert (model.predict(rows.X) >= 0).all()


def test_model_beats_a_constant_forecast(trained):
    panel, profile, model, _ = trained
    demand, _ = rec.recover_panel(panel, profile, 0.05)
    rows = make_rows(panel, demand, demand, 82, series_idx=np.arange(80, 120))
    assert np.abs(model.predict(rows.X) - rows.y).mean() < np.abs(rows.y.mean() - rows.y).mean()


def test_contributions_sum_to_the_raw_prediction(trained):
    panel, _, model, _ = trained
    rows = make_rows(panel, panel.sales, panel.sales, 70, series_idx=np.arange(5))
    contribs, base = model.contributions(rows.X)
    raw = model.point.predict(rows.X, num_iteration=model.point.best_iteration)
    np.testing.assert_allclose(contribs.sum(axis=1) + base, raw, rtol=1e-4, atol=1e-4)


def test_save_load_roundtrip(trained, tmp_path):
    panel, profile, model, conf = trained
    save_models(tmp_path / "m", model, conf, profile)
    m2, c2, p2 = load_models(tmp_path / "m")
    rows = make_rows(panel, panel.sales, panel.sales, 70, series_idx=np.arange(10))
    np.testing.assert_allclose(model.predict(rows.X), m2.predict(rows.X), rtol=1e-6)
    np.testing.assert_allclose(model.predict_quantiles(rows.X), m2.predict_quantiles(rows.X), rtol=1e-6)
    np.testing.assert_allclose(profile.shares, p2.shares, rtol=1e-5)
    h = rows.horizon.astype(int)
    np.testing.assert_allclose(
        conf.quantiles(model.predict_quantiles(rows.X), h), c2.quantiles(m2.predict_quantiles(rows.X), h)
    )


def test_panel_roundtrip(trained, tmp_path):
    panel = trained[0]
    save_panel(tmp_path / "p.npz", panel.subset(np.arange(5)))
    p2 = load_panel(tmp_path / "p.npz")
    assert p2.sales.shape == (5, panel.n_days)
    np.testing.assert_allclose(p2.sales, panel.sales[:5])
    assert (p2.dates == panel.dates).all()


def test_service_forecast_plan_explain_and_whatif(trained, tmp_path):
    panel, profile, model, conf = trained
    save_models(tmp_path / "m", model, conf, profile)
    svc = DemandService(tmp_path / "m", panel.subset(np.arange(80, 120)))
    fc = svc.forecast(0)
    assert len(fc) == 7 and (fc["lo80"] <= fc["hi80"]).all() and (fc["lo90"] <= fc["lo80"] + 1e-9).all()
    plan = svc.order_plan(fc, 3.0, 1.0)
    assert (plan.frame["order_qty"] >= 0).all()
    assert abs(plan.critical_fractile - 0.75) < 1e-9
    hi = svc.order_plan(fc, 9.0, 1.0).frame["order_qty"].to_numpy()
    lo = svc.order_plan(fc, 1.0, 1.0).frame["order_qty"].to_numpy()
    assert (hi >= lo - 1e-9).all()
    exp = svc.explain(0, horizon=1)
    assert len(exp) == 8 and "base_value" in exp.attrs

    promo = svc.forecast(0, overrides={"activity_flag": [1] * 7, "discount": [0.7] * 7})
    assert not np.allclose(promo["point"], fc["point"])
    with pytest.raises(ValueError):
        svc.forecast(0, origin=5)
