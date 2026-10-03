"""QuantumRetail dashboard: probabilistic demand forecasts and inventory decisions.

Run from the repository root:  streamlit run streamlit-app/app.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from quantumretail import ui_theme as ui  # noqa: E402
from quantumretail.bundle import load_panel  # noqa: E402
from quantumretail.inventory import quantile_from_knots  # noqa: E402
from quantumretail.service import DemandService  # noqa: E402

MODELS = ROOT / "models" / "qr_v2"
PANEL = ROOT / "data" / "demo" / "demo_panel.npz"
BENCH = ROOT / "benchmarks" / "results" / "forecast_benchmark.json"

ACCENT = "#38BDF8"
ui.apply(ACCENT, "QuantumRetail", "◆")


@st.cache_resource(show_spinner="Loading model and demo data...")
def get_service() -> DemandService:
    return DemandService(MODELS, load_panel(PANEL))


if not (MODELS / "meta.json").exists() or not PANEL.exists():
    st.error("Demo artefacts not found. Run `python -m quantumretail build-demo` first.")
    st.stop()

svc = get_service()
panel = svc.panel

# ------------------------------------------------------------------ sidebar
st.sidebar.markdown("### QuantumRetail")
st.sidebar.caption("Stockout-aware probabilistic demand forecasting")
ids = [svc.label(i) for i in range(panel.n_series)]
choice = st.sidebar.selectbox("Store / product", range(panel.n_series), format_func=lambda i: ids[i])

origins = list(range(svc.max_origin, svc.max_origin - 8, -1))
origin = st.sidebar.selectbox(
    "Forecast made at the end of",
    origins,
    format_func=lambda o: str(panel.dates[o]) + ("  (latest)" if o == svc.max_origin else "  (backtest)"),
)

st.sidebar.subheader("Inventory economics")
underage = st.sidebar.slider("Cost of a lost sale (per unit)", 0.5, 10.0, 3.0, 0.5)
overage = st.sidebar.slider("Cost of waste (per unit)", 0.5, 10.0, 1.0, 0.5)

st.sidebar.subheader("What-if promotion")
promo = st.sidebar.checkbox("Run a promotion over the next 7 days")
price_factor = st.sidebar.slider("Price factor (1.0 = no discount)", 0.60, 1.10, 0.85, 0.01, disabled=not promo)

overrides = {"discount": [price_factor] * 7, "activity_flag": [1] * 7} if promo else None
base_fc = svc.forecast(choice, origin)
fc = svc.forecast(choice, origin, overrides) if promo else base_fc
plan = svc.order_plan(fc, underage, overage)
hist = svc.history(choice)

ui.hero(
    "Demand intelligence",
    f"Demand outlook: {svc.label(choice)}",
    "Probabilistic 7-day forecasts trained on demand recovered from stockout hours, with calibrated intervals "
    "and a cost-optimal order plan.",
    [("Conformal intervals", "accent"), ("Newsvendor orders", "ok"), ("Exact SHAP", "neutral"),
     ("Promotion what-if", "warn") if promo else ("Stockout-aware", "neutral")],
)
c1, c2, c3, c4 = st.columns(4)
c1.metric(
    "7-day forecast (units)",
    f"{fc['point'].sum():.1f}",
    f"{fc['point'].sum() - base_fc['point'].sum():+.1f} vs no promo" if promo else None,
)
c2.metric("80% range", f"{fc['lo80'].sum():.1f} to {fc['hi80'].sum():.1f}", help="Sum of daily interval ends; conservative.")
c3.metric(
    "Order plan (units)",
    f"{plan.frame['order_qty'].sum():.1f}",
    help=f"Newsvendor order at the {plan.critical_fractile:.0%} critical fractile",
)
c4.metric("Stockout days (28d)", int((hist["stockout_hours"].iloc[: origin + 1].tail(28) > 0).sum()))

tab_f, tab_i, tab_w, tab_s, tab_m = st.tabs(
    ["Forecast", "Inventory plan", "Why this forecast", "Stockouts and recovery", "Model card"]
)

# ------------------------------------------------------------------ forecast
with tab_f:
    h = hist.iloc[: origin + 1].tail(42)
    fig = go.Figure()
    fig.add_bar(x=h["date"], y=h["recovered_demand"], name="Recovered demand", marker_color="rgba(56,189,248,0.30)")
    fig.add_scatter(
        x=h["date"], y=h["observed_sales"], name="Observed sales", mode="lines+markers", line=dict(color="#EDEEF0", width=1.6), marker=dict(size=5)
    )
    for lo, hi, name, alpha in (("lo90", "hi90", "90% interval", 0.12), ("lo80", "hi80", "80% interval", 0.25)):
        fig.add_scatter(
            x=pd.concat([fc["date"], fc["date"][::-1]]),
            y=pd.concat([fc[hi], fc[lo][::-1]]),
            fill="toself",
            mode="lines",
            fillcolor=f"rgba(167,139,250,{alpha})",
            line=dict(width=0),
            name=name,
        )
    fig.add_scatter(x=fc["date"], y=fc["point"], name="Forecast", line=dict(color="#A78BFA", width=3))
    actual = fc[fc["observed_sales"].notna()]
    fig.add_scatter(
        x=actual["date"], y=actual["observed_sales"], mode="markers", name="Actual sales",
        marker=dict(color="#F5B93E", size=9, symbol="diamond", line=dict(color="#08090A", width=1)),
    )
    ui.style_fig(fig, ACCENT, 430).update_layout(yaxis_title="units per day")
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "Bars show demand reconstructed for hours when the shelf was empty; the line shows what was actually "
        "sold. Intervals are conformally calibrated, so an 80% band is meant to contain about 80% of outcomes."
    )
    show = fc[["date", "point", "lo80", "hi80", "observed_sales", "stockout_hours"]].copy()
    show["date"] = show["date"].dt.strftime("%a %d %b")
    st.dataframe(show.round(2).rename(columns={"point": "forecast", "observed_sales": "actual sales"}), hide_index=True, use_container_width=True)

# ------------------------------------------------------------------ inventory
with tab_i:
    st.subheader("How much should the store stock each day?")
    st.write(
        f"With a lost sale costing {underage:g} and a wasted unit costing {overage:g}, the cost-minimising "
        f"order is the **{plan.critical_fractile:.0%} quantile** of demand (the newsvendor critical fractile)."
    )
    pf = plan.frame.copy()
    pf["date"] = pf["date"].dt.strftime("%a %d %b")
    st.dataframe(
        pf.round(2).rename(
            columns={
                "order_qty": "order",
                "expected_demand": "expected demand",
                "expected_waste_units": "expected waste",
                "expected_unmet_units": "expected unmet demand",
                "expected_fill_rate": "expected fill rate",
                "expected_cost": "expected cost",
            }
        ),
        hide_index=True,
        use_container_width=True,
    )
    taus = svc.taus
    knots = fc[[f"q{int(round(t * 100)):02d}" for t in taus]].to_numpy()
    grid = np.stack([quantile_from_knots(taus, knots, g) for g in np.linspace(taus[0], taus[-1], 61)], axis=1)
    rows = []
    for lv in np.linspace(0.5, 0.95, 10):
        q = quantile_from_knots(taus, knots, lv)
        short = np.maximum(grid - q[:, None], 0).mean(1).sum()
        over = np.maximum(q[:, None] - grid, 0).mean(1).sum()
        rows.append({"service level": lv, "expected waste": over, "expected fill rate": 1 - short / grid.mean(1).sum()})
    curve = pd.DataFrame(rows)
    f2 = go.Figure()
    f2.add_scatter(
        x=curve["expected waste"], y=curve["expected fill rate"], mode="lines+markers",
        text=[f"{v:.0%}" for v in curve["service level"]], name="order policy",
    )
    ui.style_fig(f2, ACCENT, 320).update_layout(xaxis_title="expected waste (units, 7 days)", yaxis_title="expected fill rate")
    st.plotly_chart(f2, use_container_width=True)
    st.caption("Each point is one service-level target. Moving right buys fill rate with extra waste.")

# ------------------------------------------------------------------ explain
with tab_w:
    day = st.slider("Explain forecast for day", 1, 7, 1)
    exp = svc.explain(choice, day, origin, top=10)
    base = exp.attrs["base_value"]
    colors = ["#3DD68C" if v > 0 else "#F2555A" for v in exp["contribution"]]
    f3 = go.Figure(go.Bar(x=exp["contribution"][::-1], y=exp["feature"][::-1], orientation="h", marker_color=colors[::-1]))
    ui.style_fig(f3, ACCENT, 380).update_layout(xaxis_title="effect on forecast (units)")
    st.plotly_chart(f3, use_container_width=True)
    st.caption(f"Exact tree-SHAP contributions to the point forecast, relative to a baseline of {base:.2f} units.")
    st.dataframe(exp.round(3), hide_index=True, use_container_width=True)

# ------------------------------------------------------------------ recovery
with tab_s:
    st.subheader("Stockout-aware demand recovery")
    days = list(range(max(0, origin - 13), origin + 1))
    heat = panel.stock[choice, days].astype(float)
    f4 = go.Figure(
        go.Heatmap(
            z=heat, x=list(range(24)), y=[str(panel.dates[d]) for d in days],
            colorscale=[[0, "#14161B"], [1, "#38BDF8"]], showscale=False, xgap=2, ygap=2,
        )
    )
    ui.style_fig(f4, ACCENT, 380).update_layout(xaxis_title="hour of day", title="Bright cells = out of stock")
    st.plotly_chart(f4, use_container_width=True)
    understate = 1 - hist["observed_sales"].sum() / max(hist["recovered_demand"].sum(), 1e-9)
    st.write(
        f"For this series, raw sales miss an estimated **{understate:.0%}** of demand because of stockouts. "
        "Training on raw sales would teach a model to under-forecast exactly the products that sell out."
    )
    prof = svc.profile.rows_for(panel.static["product_id"][[choice]])[0]
    f5 = go.Figure(go.Bar(x=list(range(24)), y=prof, marker_color="#A78BFA"))
    ui.style_fig(f5, ACCENT, 260).update_layout(title="Typical share of daily demand by hour", xaxis_title="hour of day")
    st.plotly_chart(f5, use_container_width=True)

# ------------------------------------------------------------------ model card
with tab_m:
    st.subheader("Evidence")
    if BENCH.exists():
        res = json.loads(BENCH.read_text())
        st.caption(f"Series split: {res['split']}. Test series were never seen in training or calibration.")
        rows = []
        for name, m in res["point_forecasts"].items():
            f = m["stockout_free_days_vs_true_demand"]
            rows.append({"model": name, "WAPE": f["wape"], "bias (WPE)": f["wpe"], "MAE": f["mae"]})
        st.markdown("**Point forecast accuracy on stockout-free test days** (observed sales equal true demand)")
        st.dataframe(pd.DataFrame(rows).round(4), hide_index=True, use_container_width=True)
        st.markdown("**Interval calibration**")
        st.json(res["probabilistic_forecasts"]["80pct_interval"])
        rv = res["recovery_validation"]
        st.markdown("**Demand-recovery validation** (controlled censoring of fully stocked days)")
        st.dataframe(
            pd.DataFrame({k: rv[k] for k in ("none", "legacy", rv["best_profile_setting"])}).T.round(4),
            use_container_width=True,
        )
    else:
        st.info("Run `python -m quantumretail benchmark` to generate benchmarks/results/forecast_benchmark.json.")
    st.caption("Dataset: FreshRetailNet-50K (Dingdong-Inc), 50,000 store-product series, hourly sales and stockout flags.")
