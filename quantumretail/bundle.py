"""Persistence for trained artefacts and the small demo dataset used by the app."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .conformal import QuantileConformalizer
from .data import DAILY_COLS, STATIC_COLS, Panel
from .model import GlobalForecaster
from .recovery import Profile


def save_models(path: str | Path, model: GlobalForecaster, conf: QuantileConformalizer, profile: Profile) -> None:
    """Write model, conformal calibration and hourly profile to a directory."""
    p = Path(path)
    model.save(p)
    (p / "conformal.json").write_text(json.dumps(conf.to_dict()))
    np.savez_compressed(
        p / "profile.npz",
        product_ids=profile.product_ids,
        shares=profile.shares.astype(np.float32),
        global_shares=profile.global_shares,
        support_days=profile.support_days,
    )


def load_models(path: str | Path) -> tuple[GlobalForecaster, QuantileConformalizer, Profile]:
    p = Path(path)
    model = GlobalForecaster.load(p)
    conf = QuantileConformalizer.from_dict(json.loads((p / "conformal.json").read_text()))
    z = np.load(p / "profile.npz")
    profile = Profile(
        product_ids=z["product_ids"],
        shares=z["shares"].astype(np.float64),
        global_shares=z["global_shares"],
        support_days=z["support_days"],
    )
    return model, conf, profile


def save_panel(path: str | Path, panel: Panel) -> None:
    """Persist a (small) panel as a compressed npz."""
    arrays = {f"static__{k}": v for k, v in panel.static.items()}
    arrays.update({f"daily__{k}": v for k, v in panel.daily.items()})
    arrays.update(
        sales=panel.sales,
        hours=panel.hours.astype(np.float16),
        stock=panel.stock,
        stockout_hours=panel.stockout_hours,
        dates=panel.dates.astype("datetime64[D]").astype(np.int32),
        n_train_days=np.int32(panel.n_train_days),
    )
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def load_panel(path: str | Path) -> Panel:
    z = np.load(path)
    return Panel(
        static={k: z[f"static__{k}"] for k in STATIC_COLS},
        daily={k: z[f"daily__{k}"] for k in DAILY_COLS},
        sales=z["sales"],
        hours=z["hours"].astype(np.float32),
        stock=z["stock"],
        stockout_hours=z["stockout_hours"],
        dates=z["dates"].astype("datetime64[D]"),
        n_train_days=int(z["n_train_days"]),
    )
