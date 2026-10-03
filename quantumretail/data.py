"""Data layer: download FreshRetailNet-50K and reshape it into a dense panel.

FreshRetailNet-50K stores one row per (store, product, day). Each row carries the
24 hourly sales values and 24 hourly stockout flags (1 = out of stock), plus
daily covariates. Every series has exactly 90 training days
(2024-03-28 .. 2024-06-25) and 7 held-out evaluation days (2024-06-26 .. 2024-07-02).

Working with a dense ``(series, day, hour)`` tensor makes every later step
(recovery, features, backtests) a vectorised numpy operation.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

HF_REPO = "Dingdong-Inc/FreshRetailNet-50K"
TRAIN_DAYS = 90
TEST_DAYS = 7
HOURS = 24
SELLING_WINDOW = (6, 22)  # restocking happens at 06:00; the shop closes at 22:00

STATIC_COLS = [
    "city_id",
    "store_id",
    "management_group_id",
    "first_category_id",
    "second_category_id",
    "third_category_id",
    "product_id",
]
DAILY_COLS = [
    "discount",
    "holiday_flag",
    "activity_flag",
    "precpt",
    "avg_temperature",
    "avg_humidity",
    "avg_wind_level",
]


@dataclass
class Panel:
    """Dense representation of the dataset.

    Shapes: ``S`` series, ``T`` days (train days, optionally followed by the test days).
    """

    static: dict[str, np.ndarray]            # name -> (S,)
    daily: dict[str, np.ndarray]             # name -> (S, T) float32
    sales: np.ndarray                        # (S, T) observed daily sales
    hours: np.ndarray                        # (S, T, 24) observed hourly sales
    stock: np.ndarray                        # (S, T, 24) uint8, 1 = stockout
    stockout_hours: np.ndarray               # (S, T) stockout hours inside 06:00-22:00
    dates: np.ndarray                        # (T,) datetime64[D]
    n_train_days: int = TRAIN_DAYS
    meta: dict = field(default_factory=dict)

    @property
    def n_series(self) -> int:
        return self.sales.shape[0]

    @property
    def n_days(self) -> int:
        return self.sales.shape[1]

    def subset(self, idx: np.ndarray) -> Panel:
        """Return a panel restricted to the given series indices."""
        idx = np.asarray(idx)
        return Panel(
            static={k: v[idx] for k, v in self.static.items()},
            daily={k: v[idx] for k, v in self.daily.items()},
            sales=self.sales[idx],
            hours=self.hours[idx],
            stock=self.stock[idx],
            stockout_hours=self.stockout_hours[idx],
            dates=self.dates,
            n_train_days=self.n_train_days,
            meta=dict(self.meta),
        )

    def first_days(self, n: int) -> Panel:
        """Return a panel truncated to its first ``n`` days."""
        return Panel(
            static=self.static,
            daily={k: v[:, :n] for k, v in self.daily.items()},
            sales=self.sales[:, :n],
            hours=self.hours[:, :n],
            stock=self.stock[:, :n],
            stockout_hours=self.stockout_hours[:, :n],
            dates=self.dates[:n],
            n_train_days=min(self.n_train_days, n),
            meta=dict(self.meta),
        )


def fetch(cache_dir: str | Path = "data/raw") -> tuple[Path, Path]:
    """Download the train and eval parquet files from the Hugging Face Hub."""
    from huggingface_hub import hf_hub_download

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for name in ("train", "eval"):
        p = hf_hub_download(
            repo_id=HF_REPO,
            filename=f"data/{name}.parquet",
            repo_type="dataset",
            local_dir=str(cache_dir),
        )
        paths.append(Path(p))
    return paths[0], paths[1]


def _read_table(path: Path):
    import pyarrow.parquet as pq

    return pq.read_table(path)


def _list_column(table, name: str) -> np.ndarray:
    """Read a fixed-length list column (24 values per row) as an (N, 24) array."""
    arr = table[name].combine_chunks()
    flat = arr.flatten().to_numpy(zero_copy_only=False)
    return flat.reshape(-1, HOURS)


def _frame_to_arrays(table) -> dict[str, np.ndarray]:
    import pyarrow as pa
    import pyarrow.compute as pc

    out: dict[str, np.ndarray] = {}
    for c in STATIC_COLS + DAILY_COLS + ["sale_amount", "stock_hour6_22_cnt"]:
        out[c] = table[c].to_numpy()
    dt = pc.cast(table["dt"], pa.date32())
    out["dt"] = dt.to_numpy(zero_copy_only=False).astype("datetime64[D]")
    out["hours_sale"] = _list_column(table, "hours_sale").astype(np.float32)
    out["hours_stock_status"] = _list_column(table, "hours_stock_status").astype(np.uint8)
    return out


def build_panel(train_path: str | Path, eval_path: str | Path | None = None) -> Panel:
    """Load parquet files and return a dense :class:`Panel`.

    When ``eval_path`` is given, the 7 evaluation days are appended after the
    90 training days so covariates for the forecast horizon are available.
    """
    parts = [_frame_to_arrays(_read_table(Path(train_path)))]
    if eval_path is not None:
        parts.append(_frame_to_arrays(_read_table(Path(eval_path))))
    cols = parts[0].keys()
    data = {c: np.concatenate([p[c] for p in parts]) for c in cols}

    order = np.lexsort((data["dt"], data["product_id"], data["store_id"]))
    data = {c: v[order] for c, v in data.items()}

    n_rows = len(order)
    n_days = TRAIN_DAYS + (TEST_DAYS if eval_path is not None else 0)
    if n_rows % n_days:
        raise ValueError(f"{n_rows} rows is not a multiple of {n_days} days per series")
    n_series = n_rows // n_days

    def to_sd(a: np.ndarray) -> np.ndarray:
        return a.reshape(n_series, n_days, *a.shape[1:])

    dates = to_sd(data["dt"])[0]
    static = {c: to_sd(data[c])[:, 0].copy() for c in STATIC_COLS}
    daily = {c: to_sd(data[c]).astype(np.float32) for c in DAILY_COLS}
    return Panel(
        static=static,
        daily=daily,
        sales=to_sd(data["sale_amount"]).astype(np.float32),
        hours=to_sd(data["hours_sale"]),
        stock=to_sd(data["hours_stock_status"]),
        stockout_hours=to_sd(data["stock_hour6_22_cnt"]).astype(np.int16),
        dates=dates,
        meta={"source": HF_REPO},
    )


def sample_series(panel: Panel, n: int, seed: int = 0) -> np.ndarray:
    """Deterministic random sample of series indices."""
    rng = np.random.default_rng(seed)
    n = min(n, panel.n_series)
    return np.sort(rng.choice(panel.n_series, size=n, replace=False))
