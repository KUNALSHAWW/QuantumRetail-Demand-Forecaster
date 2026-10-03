"""REST API: ``uvicorn api.main:app``.

Environment variables:
    QR_MODELS : directory with the trained bundle (default ``models/qr_v2``)
    QR_PANEL  : npz panel with the series to serve (default ``data/demo/demo_panel.npz``)
"""
from __future__ import annotations

import os
from functools import lru_cache

import numpy as np
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from quantumretail.bundle import load_panel
from quantumretail.service import DemandService

app = FastAPI(
    title="QuantumRetail API",
    version="2.0.0",
    description="Stockout-aware probabilistic demand forecasts and inventory plans.",
)


@lru_cache(maxsize=1)
def service() -> DemandService:
    models = os.environ.get("QR_MODELS", "models/qr_v2")
    panel = os.environ.get("QR_PANEL", "data/demo/demo_panel.npz")
    return DemandService(models, load_panel(panel))


def _locate(store: int, product: int) -> int:
    s = service().panel.static
    hit = np.where((s["store_id"] == store) & (s["product_id"] == product))[0]
    if len(hit) == 0:
        raise HTTPException(404, f"series store={store} product={product} is not available")
    return int(hit[0])


def _records(df) -> list[dict]:
    out = df.copy()
    for c in out.columns:
        if str(out[c].dtype).startswith("datetime"):
            out[c] = out[c].dt.strftime("%Y-%m-%d")
    return out.round(4).to_dict(orient="records")


class PlanRequest(BaseModel):
    store: int
    product: int
    underage_cost: float = Field(3.0, gt=0, description="Cost of one unit of unmet demand")
    overage_cost: float = Field(1.0, gt=0, description="Cost of one unit left unsold (waste)")
    origin: int | None = None


class WhatIfRequest(BaseModel):
    store: int
    product: int
    discount: list[float] | None = Field(None, min_length=1, max_length=7, description="price factor per day, 1.0 = no discount")
    activity_flag: list[int] | None = Field(None, min_length=1, max_length=7)
    origin: int | None = None


@app.get("/health")
def health() -> dict:
    svc = service()
    return {"status": "ok", "series": svc.panel.n_series, "quantiles": list(map(float, svc.taus))}


@app.get("/series")
def series(limit: int = Query(50, ge=1, le=1000)) -> list[dict]:
    s = service().panel.static
    return [
        {"store": int(s["store_id"][i]), "product": int(s["product_id"][i])}
        for i in range(min(limit, service().panel.n_series))
    ]


@app.get("/forecast")
def forecast(store: int, product: int, origin: int | None = None) -> dict:
    svc = service()
    i = _locate(store, product)
    try:
        fc = svc.forecast(i, origin)
    except ValueError as e:
        raise HTTPException(422, str(e)) from e
    return {"series": svc.label(i), "forecast": _records(fc)}


@app.post("/order-plan")
def order_plan(req: PlanRequest) -> dict:
    svc = service()
    i = _locate(req.store, req.product)
    try:
        fc = svc.forecast(i, req.origin)
    except ValueError as e:
        raise HTTPException(422, str(e)) from e
    plan = svc.order_plan(fc, req.underage_cost, req.overage_cost)
    return {
        "series": svc.label(i),
        "critical_fractile": round(plan.critical_fractile, 4),
        "plan": _records(plan.frame),
    }


@app.post("/what-if")
def what_if(req: WhatIfRequest) -> dict:
    svc = service()
    i = _locate(req.store, req.product)
    overrides = {}
    if req.discount:
        overrides["discount"] = req.discount
    if req.activity_flag:
        overrides["activity_flag"] = req.activity_flag
    base = svc.forecast(i, req.origin)
    alt = svc.forecast(i, req.origin, overrides or None)
    n = min(len(req.discount or []) or 7, 7)
    return {
        "series": svc.label(i),
        "baseline": _records(base[["date", "point", "lo80", "hi80"]]),
        "scenario": _records(alt[["date", "point", "lo80", "hi80"]]),
        "uplift_units_first_days": round(float((alt["point"] - base["point"]).iloc[:n].sum()), 4),
    }


@app.get("/explain")
def explain(store: int, product: int, horizon: int = Query(1, ge=1, le=7), origin: int | None = None) -> dict:
    svc = service()
    i = _locate(store, product)
    exp = svc.explain(i, horizon, origin)
    return {
        "series": svc.label(i),
        "horizon": horizon,
        "base_value": round(exp.attrs["base_value"], 4),
        "contributions": _records(exp),
    }
