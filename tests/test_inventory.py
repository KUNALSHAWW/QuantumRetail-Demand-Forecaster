import numpy as np

from quantumretail import inventory as inv

TAUS = np.array([0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95])
KNOTS = np.array([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]])


def test_interpolation_between_knots():
    assert inv.quantile_from_knots(TAUS, KNOTS, 0.50)[0] == 4.0
    assert abs(inv.quantile_from_knots(TAUS, KNOTS, 0.625)[0] - 4.5) < 1e-9
    assert inv.quantile_from_knots(TAUS, KNOTS, 0.99)[0] == 7.0  # clamped to the covered range


def test_newsvendor_symmetric_costs_gives_median():
    assert inv.newsvendor_order(TAUS, KNOTS, 1.0, 1.0)[0] == 4.0


def test_higher_underage_cost_orders_more():
    low = inv.newsvendor_order(TAUS, KNOTS, 1.0, 1.0)[0]
    high = inv.newsvendor_order(TAUS, KNOTS, 9.0, 1.0)[0]
    assert high > low
    assert inv.critical_fractile(9, 1) == 0.9


def test_policy_outcome_arithmetic():
    order = np.array([5.0, 5.0, 5.0, 5.0])
    demand = np.array([3.0, 5.0, 7.0, 1.0])
    out = inv.policy_outcome(order, demand)
    assert out["fill_rate"] == (3 + 5 + 5 + 1) / 16
    assert out["waste_rate"] == (2 + 0 + 0 + 4) / 20
    assert out["stockout_rate"] == 0.25


def test_expected_cost_is_asymmetric():
    assert inv.expected_cost(np.array([2.0]), np.array([5.0]), 3.0, 1.0) == 9.0
    assert inv.expected_cost(np.array([5.0]), np.array([2.0]), 3.0, 1.0) == 3.0


def test_newsvendor_beats_a_mean_policy_on_skewed_demand():
    rng = np.random.default_rng(0)
    demand = rng.gamma(3.0, 1.0, size=50_000)
    knots = np.tile(np.quantile(demand, TAUS), (len(demand), 1))
    cu, co = 4.0, 1.0
    opt = inv.expected_cost(inv.newsvendor_order(TAUS, knots, cu, co), demand, cu, co)
    mean_policy = inv.expected_cost(np.full_like(demand, demand.mean()), demand, cu, co)
    assert opt < mean_policy
