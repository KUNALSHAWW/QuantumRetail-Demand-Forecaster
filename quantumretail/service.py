"""High-level inference API used by the Streamlit app, the REST API and the CLI."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from . import recovery as rec
from .bundle import load_models
from .data import Panel
from .features import HORIZONS, MIN_ORIGIN, make_rows
from .inventory import critical_fractile, quantile_from_knots


@dataclass
class OrderPlan:
    """Cost-optimal order for each forecast day."""

    frame: pd.DataFrame
    critical_fractile: float


class DemandService:
    """Forecast, explain and plan inventory for series in a panel."""

    def __init__(self, model_dir: str | Path, panel: Panel, min_share: float = 0.05):
        self.model, self.conformal, self.profile = load_models(model_dir)
        self.panel = panel
        self.taus = np.asarray(self.model.config.quantiles)
        self.demand, self.imputed = rec.recover_panel(panel, self.profile, min_share)

    # ------------------------------------------------------------------ #
    @property
    def max_origin(self) -> int:
        return self.panel.n_days - 1 - max(HORIZONS)

    def label(self, i: int) -> str:
        s = self.panel.static
        return f"store {int(s['store_id'][i])} / product {int(s['product_id'][i])}"

    def _rows(self, i: int, origin: int, overrides: dict | None):
        panel = self.panel.subset([i])
        demand = self.demand[[i]]
        if overrides:
            daily = {k: v.copy() for k, v in panel.daily.items()}
            for key, values in overrides.items():
                vals = np.asarray(values, dtype=np.float32)
                span = slice(origin + 1, origin + 1 + len(vals))
                daily[key][0, span] = vals
            panel = Panel(**{**panel.__dict__, "daily": daily})
        return panel, make_rows(panel, demand, demand, origin, series_idx=np.array([0]))

    def forecast(self, i: int, origin: int | None = None, overrides: dict | None = None) -> pd.DataFrame:
        """Seven-day forecast with calibrated quantiles and intervals.

        ``overrides`` replaces known-future covariates for the horizon, for example
        ``{"discount": [0.8] * 7, "activity_flag": [1] * 7}`` for a what-if promotion.
        """
        origin = self.max_origin if origin is None else origin
        if not MIN_ORIGIN <= origin <= self.max_origin:
            raise ValueError(f"origin must be between {MIN_ORIGIN} and {self.max_origin}")
        _, rows = self._rows(i, origin, overrides)
        raw_q = self.model.predict_quantiles(rows.X)
        h = rows.horizon.astype(int)
        cal = self.conformal.quantiles(raw_q, h)
        lo80, hi80 = self.conformal.interval(raw_q, h, 0.20)
        lo90, hi90 = self.conformal.interval(raw_q, h, 0.10)
        dates = self.panel.dates[origin + h]
        df = pd.DataFrame(
            {
                "date": pd.to_datetime(dates),
                "horizon": h,
                "point": self.model.predict(rows.X),
                "lo80": lo80,
                "hi80": hi80,
                "lo90": lo90,
                "hi90": hi90,
                "observed_sales": rows.y_obs,
                "stockout_hours": rows.stockout_hours,
            }
        )
        for k, tau in enumerate(self.taus):
            df[f"q{int(round(tau * 100)):02d}"] = cal[:, k]
        return df

    # ------------------------------------------------------------------ #
    def order_plan(self, forecast: pd.DataFrame, underage_cost: float = 3.0, overage_cost: float = 1.0) -> OrderPlan:
        """Newsvendor order quantities and expected outcomes for each day."""
        cols = [f"q{int(round(t * 100)):02d}" for t in self.taus]
        knots = forecast[cols].to_numpy()
        frac = critical_fractile(underage_cost, overage_cost)
        order = quantile_from_knots(self.taus, knots, frac)
        grid = np.linspace(self.taus[0], self.taus[-1], 61)
        samples = np.stack([quantile_from_knots(self.taus, knots, g) for g in grid], axis=1)
        exp_over = np.maximum(order[:, None] - samples, 0).mean(1)
        exp_short = np.maximum(samples - order[:, None], 0).mean(1)
        exp_demand = samples.mean(1)
        out = forecast[["date", "horizon"]].copy()
        out["order_qty"] = order
        out["expected_demand"] = exp_demand
        out["expected_waste_units"] = exp_over
        out["expected_unmet_units"] = exp_short
        out["expected_fill_rate"] = 1 - exp_short / np.maximum(exp_demand, 1e-9)
        out["expected_cost"] = underage_cost * exp_short + overage_cost * exp_over
        return OrderPlan(out, frac)

    def explain(self, i: int, horizon: int, origin: int | None = None, top: int = 8) -> pd.DataFrame:
        """Top feature contributions to the point forecast for one horizon (exact)."""
        origin = self.max_origin if origin is None else origin
        _, rows = self._rows(i, origin, None)
        k = int(np.where(rows.horizon == horizon)[0][0])
        contribs, base = self.model.contributions(rows.X[k : k + 1])
        out = pd.DataFrame(
            {"feature": rows.columns, "value": rows.X[k], "contribution": contribs[0]}
        )
        out["abs"] = out["contribution"].abs()
        out = out.sort_values("abs", ascending=False).head(top).drop(columns="abs")
        out.attrs["base_value"] = base
        return out.reset_index(drop=True)

    def history(self, i: int) -> pd.DataFrame:
        """Observed vs recovered demand for the whole panel of one series."""
        return pd.DataFrame(
            {
                "date": pd.to_datetime(self.panel.dates),
                "observed_sales": self.panel.sales[i],
                "recovered_demand": self.demand[i],
                "stockout_hours": self.panel.stockout_hours[i],
            }
        )
