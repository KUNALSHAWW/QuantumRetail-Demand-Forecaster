import numpy as np
import pytest

from quantumretail.features import MIN_ORIGIN, make_rows, stack_rows

FUTURE_DEPENDENT = ("y_lag7", "y_lag14", "y_lag21", "y_lag28", "same_dow_mean")


def test_shapes_and_horizons(panel):
    rows = make_rows(panel, panel.sales, panel.sales, 60)
    assert len(rows) == panel.n_series * 7
    assert rows.X.shape == (len(rows), len(rows.columns))
    assert set(np.unique(rows.horizon)) == set(range(1, 8))


def test_no_leakage_from_future_sales(panel):
    """Rewriting everything after the origin must not change history features."""
    t = 60
    base = make_rows(panel, panel.sales, panel.sales, t)
    mutated = panel.sales.copy()
    mutated[:, t + 1 :] = 999.0
    mut = make_rows(panel, mutated, mutated, t)
    keep = [i for i, c in enumerate(base.columns) if c not in FUTURE_DEPENDENT]
    np.testing.assert_array_equal(base.X[:, keep], mut.X[:, keep])
    assert not np.array_equal(base.y, mut.y)  # the target itself did change


def test_same_dow_lags_never_look_past_the_origin(panel):
    t = 60
    mutated = panel.sales.copy()
    mutated[:, t + 1 :] = 1e6
    rows = make_rows(panel, mutated, mutated, t)
    lag_cols = [rows.columns.index(f"y_lag{k}") for k in (7, 14, 21, 28)]
    assert (rows.X[:, lag_cols] < 1e5).all()


def test_known_future_covariates_do_flow_into_features(panel):
    t = 60
    a = make_rows(panel, panel.sales, panel.sales, t)
    daily = {k: v.copy() for k, v in panel.daily.items()}
    daily["discount"][:, t + 1 : t + 8] = 0.5
    changed = type(panel)(**{**panel.__dict__, "daily": daily})
    b = make_rows(changed, changed.sales, changed.sales, t)
    c = a.columns.index("discount")
    assert not np.array_equal(a.X[:, c], b.X[:, c])


def test_origin_bounds(panel):
    with pytest.raises(ValueError):
        make_rows(panel, panel.sales, panel.sales, MIN_ORIGIN - 1)
    with pytest.raises(ValueError):
        make_rows(panel, panel.sales, panel.sales, panel.n_days - 3)


def test_stack_rows_concatenates(panel):
    r1 = make_rows(panel, panel.sales, panel.sales, 40)
    r2 = make_rows(panel, panel.sales, panel.sales, 50)
    s = stack_rows([r1, r2])
    assert len(s) == len(r1) + len(r2)
    assert s.X.shape[1] == r1.X.shape[1]


def test_series_subset(panel):
    rows = make_rows(panel, panel.sales, panel.sales, 60, series_idx=np.array([3, 7]))
    assert set(np.unique(rows.series)) == {3, 7}
    assert len(rows) == 14
