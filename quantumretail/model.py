"""Global LightGBM forecaster: one point model plus a family of quantile models.

A single model is trained across every store-product series (a "global" model).
With only 90 days per series, a per-series model has under 80 training rows;
pooling series lets the model learn shared structure (weekly seasonality, promo
and weather response, how stockouts distort history) and generalise to series it
has never seen.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import lightgbm as lgb
import numpy as np

from .features import CATEGORICAL, RowSet

QUANTILES = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)


@dataclass
class ForecasterConfig:
    objective: str = "regression_l1"  # median regression: optimal for the WAPE metric
    n_estimators_point: int = 800
    n_estimators_quantile: int = 400
    learning_rate: float = 0.05
    num_leaves: int = 63
    min_data_in_leaf: int = 200
    feature_fraction: float = 0.8
    bagging_fraction: float = 0.8
    bagging_freq: int = 1
    lambda_l2: float = 1.0
    early_stopping_rounds: int = 50
    seed: int = 7
    n_jobs: int = 0
    quantiles: tuple[float, ...] = QUANTILES


@dataclass
class GlobalForecaster:
    config: ForecasterConfig = field(default_factory=ForecasterConfig)
    columns: list[str] = field(default_factory=list)
    point: lgb.Booster | None = None
    quantile_models: dict[float, lgb.Booster] = field(default_factory=dict)

    # -- training --------------------------------------------------------- #
    def _params(self, objective: str, alpha: float | None = None) -> dict:
        c = self.config
        p = {
            "objective": objective,
            "learning_rate": c.learning_rate,
            "num_leaves": c.num_leaves,
            "min_data_in_leaf": c.min_data_in_leaf,
            "feature_fraction": c.feature_fraction,
            "bagging_fraction": c.bagging_fraction,
            "bagging_freq": c.bagging_freq,
            "lambda_l2": c.lambda_l2,
            "seed": c.seed,
            "num_threads": c.n_jobs,
            "verbosity": -1,
            "cat_smooth": 20,
            "max_cat_to_onehot": 4,
        }
        if alpha is not None:
            p["alpha"] = alpha
        return p

    def _dataset(self, rows: RowSet) -> lgb.Dataset:
        cats = [c for c in CATEGORICAL if c in self.columns]
        return lgb.Dataset(
            rows.X,
            label=rows.y,
            feature_name=self.columns,
            categorical_feature=cats,
            free_raw_data=False,
        )

    def fit(self, train: RowSet, valid: RowSet, verbose: bool = False) -> GlobalForecaster:
        self.columns = list(train.columns)
        c = self.config
        dtrain = self._dataset(train)
        dvalid = lgb.Dataset(valid.X, label=valid.y, reference=dtrain, free_raw_data=False)
        cb = [lgb.early_stopping(c.early_stopping_rounds, verbose=False)]
        if verbose:
            cb.append(lgb.log_evaluation(100))

        self.point = lgb.train(
            self._params(c.objective), dtrain, c.n_estimators_point, valid_sets=[dvalid], callbacks=cb
        )
        self.quantile_models = {}
        for q in c.quantiles:
            self.quantile_models[q] = lgb.train(
                self._params("quantile", q),
                dtrain,
                c.n_estimators_quantile,
                valid_sets=[dvalid],
                callbacks=[lgb.early_stopping(c.early_stopping_rounds, verbose=False)],
            )
        return self

    # -- inference -------------------------------------------------------- #
    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.maximum(self.point.predict(X, num_iteration=self.point.best_iteration), 0.0)

    def predict_quantiles(self, X: np.ndarray) -> np.ndarray:
        """Return ``(n, K)`` quantile forecasts, sorted to remove quantile crossing."""
        cols = [
            self.quantile_models[q].predict(X, num_iteration=self.quantile_models[q].best_iteration)
            for q in self.config.quantiles
        ]
        out = np.maximum(np.stack(cols, axis=1), 0.0)
        return np.sort(out, axis=1)

    def contributions(self, X: np.ndarray) -> tuple[np.ndarray, float]:
        """Per-feature SHAP-style contributions of the point forecast (exact, via LightGBM).

        Returns ``(contribs, base_value)``; ``contribs`` has one column per feature
        and ``contribs.sum(axis=1) + base_value`` equals the raw model output.
        """
        raw = self.point.predict(X, pred_contrib=True, num_iteration=self.point.best_iteration)
        return raw[:, :-1], float(raw[0, -1])

    def feature_importance(self, kind: str = "gain") -> dict[str, float]:
        imp = self.point.feature_importance(importance_type=kind)
        total = imp.sum() or 1.0
        return {c: float(v / total) for c, v in sorted(zip(self.columns, imp), key=lambda t: -t[1])}

    # -- persistence ------------------------------------------------------ #
    def save(self, directory: str | Path) -> None:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        self.point.save_model(str(d / "point.txt"), num_iteration=self.point.best_iteration)
        for q, m in self.quantile_models.items():
            m.save_model(str(d / f"q{int(round(q * 100)):02d}.txt"), num_iteration=m.best_iteration)
        meta = {"columns": self.columns, "config": asdict(self.config)}
        (d / "meta.json").write_text(json.dumps(meta, indent=1))

    @classmethod
    def load(cls, directory: str | Path) -> GlobalForecaster:
        d = Path(directory)
        meta = json.loads((d / "meta.json").read_text())
        cfg = ForecasterConfig(**{**meta["config"], "quantiles": tuple(meta["config"]["quantiles"])})
        obj = cls(config=cfg, columns=meta["columns"])
        obj.point = lgb.Booster(model_file=str(d / "point.txt"))
        obj.quantile_models = {
            q: lgb.Booster(model_file=str(d / f"q{int(round(q * 100)):02d}.txt")) for q in cfg.quantiles
        }
        return obj
