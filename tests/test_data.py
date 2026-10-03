import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from quantumretail.data import DAILY_COLS, STATIC_COLS, build_panel, sample_series


def _write(path, n_series, days, start):
    rows = []
    rng = np.random.default_rng(0)
    for s in range(n_series):
        for d in range(days):
            hs = rng.random(24).round(2).tolist()
            st = (rng.random(24) < 0.2).astype(int).tolist()
            rows.append(
                {
                    **{c: s % 3 for c in STATIC_COLS},
                    "store_id": s,
                    "product_id": s,
                    "dt": str(np.datetime64(start) + d),
                    "sale_amount": float(sum(hs)),
                    "hours_sale": hs,
                    "stock_hour6_22_cnt": int(sum(st[6:22])),
                    "hours_stock_status": st,
                    **{c: float(rng.random()) for c in DAILY_COLS},
                }
            )
    cols = {k: [r[k] for r in rows] for k in rows[0]}
    pq.write_table(pa.table(cols), path)


def test_build_panel_reshapes_and_orders(tmp_path):
    _write(tmp_path / "train.parquet", 4, 90, "2024-03-28")
    _write(tmp_path / "eval.parquet", 4, 7, "2024-06-26")
    p = build_panel(tmp_path / "train.parquet", tmp_path / "eval.parquet")
    assert p.sales.shape == (4, 97)
    assert p.hours.shape == (4, 97, 24) and p.stock.shape == (4, 97, 24)
    assert (np.diff(p.dates.astype("int64")) == 1).all()  # contiguous and sorted
    np.testing.assert_allclose(p.hours.sum(-1), p.sales, rtol=1e-4)  # daily sales == sum of hours
    assert list(p.static["store_id"]) == [0, 1, 2, 3]


def test_train_only_panel_has_90_days(tmp_path):
    _write(tmp_path / "train.parquet", 3, 90, "2024-03-28")
    assert build_panel(tmp_path / "train.parquet").n_days == 90


def test_sample_series_is_deterministic(panel):
    a = sample_series(panel, 10, seed=1)
    b = sample_series(panel, 10, seed=1)
    assert (a == b).all() and len(set(a)) == 10
