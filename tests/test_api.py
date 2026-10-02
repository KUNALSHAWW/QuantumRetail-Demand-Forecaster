import numpy as np
import pytest
from fastapi.testclient import TestClient

from quantumretail import recovery as rec
from quantumretail.backtest import build_training_rows
from quantumretail.bundle import save_models, save_panel
from quantumretail.conformal import QuantileConformalizer
from quantumretail.features import make_rows
from quantumretail.model import ForecasterConfig, GlobalForecaster

from .conftest import make_panel

CFG = ForecasterConfig(n_estimators_point=30, n_estimators_quantile=20, num_leaves=7, min_data_in_leaf=20)


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    d = tmp_path_factory.mktemp("bundle")
    panel = make_panel(n_series=100, seed=21)
    profile = rec.fit_profile(panel, slice(0, 61))
    demand, _ = rec.recover_panel(panel, profile, 0.05)
    tr = np.arange(0, 70)
    model = GlobalForecaster(CFG).fit(
        build_training_rows(panel, demand, tr, tuple(range(27, 68, 4))),
        make_rows(panel, demand, demand, 75, series_idx=tr),
    )
    calib = make_rows(panel, demand, panel.sales, 82, series_idx=np.arange(70, 100))
    free = calib.stockout_hours == 0
    conf = QuantileConformalizer(tuple(CFG.quantiles)).fit(
        model.predict_quantiles(calib.X)[free], calib.y_obs[free], calib.horizon[free]
    )
    save_models(d / "models", model, conf, profile)
    save_panel(d / "panel.npz", panel.subset(np.arange(70, 100)))

    import os

    os.environ["QR_MODELS"] = str(d / "models")
    os.environ["QR_PANEL"] = str(d / "panel.npz")
    from api.main import app, service

    service.cache_clear()
    return TestClient(app), panel


def test_health_and_series(client):
    c, _ = client
    r = c.get("/health").json()
    assert r["status"] == "ok" and r["series"] == 30
    assert len(c.get("/series", params={"limit": 5}).json()) == 5


def test_forecast_endpoint(client):
    c, panel = client
    sid = c.get("/series").json()[0]
    r = c.get("/forecast", params=sid)
    assert r.status_code == 200
    rows = r.json()["forecast"]
    assert len(rows) == 7 and all(x["lo80"] <= x["hi80"] for x in rows)


def test_unknown_series_is_404_and_bad_origin_is_422(client):
    c, _ = client
    assert c.get("/forecast", params={"store": -1, "product": -1}).status_code == 404
    sid = c.get("/series").json()[0]
    assert c.get("/forecast", params={**sid, "origin": 3}).status_code == 422


def test_order_plan_respects_costs(client):
    c, _ = client
    sid = c.get("/series").json()[0]
    low = c.post("/order-plan", json={**sid, "underage_cost": 1, "overage_cost": 1}).json()
    high = c.post("/order-plan", json={**sid, "underage_cost": 9, "overage_cost": 1}).json()
    assert high["critical_fractile"] > low["critical_fractile"]
    assert sum(r["order_qty"] for r in high["plan"]) >= sum(r["order_qty"] for r in low["plan"])


def test_order_plan_validates_input(client):
    c, _ = client
    sid = c.get("/series").json()[0]
    assert c.post("/order-plan", json={**sid, "underage_cost": 0, "overage_cost": 1}).status_code == 422


def test_what_if_changes_the_forecast(client):
    c, _ = client
    sid = c.get("/series").json()[0]
    r = c.post("/what-if", json={**sid, "discount": [0.7] * 7, "activity_flag": [1] * 7}).json()
    assert r["baseline"] != r["scenario"]


def test_explain_returns_top_contributions(client):
    c, _ = client
    sid = c.get("/series").json()[0]
    r = c.get("/explain", params={**sid, "horizon": 2}).json()
    assert len(r["contributions"]) == 8 and r["horizon"] == 2
