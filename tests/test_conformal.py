from statistics import NormalDist

import numpy as np

from quantumretail.conformal import QuantileConformalizer, coverage, pinball_loss

TAUS = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)
Z = np.array([NormalDist().inv_cdf(t) for t in TAUS])


def _draw(rng, n, scale):
    h = rng.integers(1, 8, n)
    mu = rng.uniform(8, 12, n)  # far from zero: demand cannot be negative
    sd = scale * (1 + 0.1 * h)
    y = mu + sd * rng.standard_normal(n)
    return h, mu, y


def _narrow_quantiles(mu, sd_assumed):
    return mu[:, None] + sd_assumed * Z[None, :]


def test_cqr_restores_coverage_of_overconfident_quantiles():
    rng = np.random.default_rng(0)
    h, mu, y = _draw(rng, 40_000, 1.0)
    conf = QuantileConformalizer(TAUS).fit(_narrow_quantiles(mu, 0.4), y, h)

    h2, mu2, y2 = _draw(rng, 40_000, 1.0)
    q2 = _narrow_quantiles(mu2, 0.4)
    assert coverage(y2, q2[:, 1], q2[:, 5]) < 0.55  # raw 80% interval is badly off
    lo, hi = conf.interval(q2, h2, alpha=0.20)
    assert abs(coverage(y2, lo, hi) - 0.80) < 0.02  # conformalised: on target


def test_one_sided_recalibration_fixes_every_quantile():
    rng = np.random.default_rng(1)
    h, mu, y = _draw(rng, 40_000, 1.0)
    conf = QuantileConformalizer(TAUS).fit(_narrow_quantiles(mu, 0.5), y, h)
    h2, mu2, y2 = _draw(rng, 40_000, 1.0)
    q2 = conf.quantiles(_narrow_quantiles(mu2, 0.5), h2)
    for k, tau in enumerate(TAUS):
        assert abs((y2 <= q2[:, k]).mean() - tau) < 0.02


def test_recalibrated_quantiles_do_not_cross_and_stay_non_negative():
    rng = np.random.default_rng(2)
    h, mu, y = _draw(rng, 5_000, 1.0)
    q = _narrow_quantiles(mu, 0.3)
    out = QuantileConformalizer(TAUS).fit(q, y, h).quantiles(q, h)
    assert (np.diff(out, axis=1) >= 0).all()
    assert (out >= 0).all()


def test_serialisation_roundtrip():
    rng = np.random.default_rng(3)
    h, mu, y = _draw(rng, 3_000, 1.0)
    q = _narrow_quantiles(mu, 0.5)
    conf = QuantileConformalizer(TAUS).fit(q, y, h)
    again = QuantileConformalizer.from_dict(conf.to_dict())
    np.testing.assert_allclose(again.quantiles(q, h), conf.quantiles(q, h))
    a, b = again.interval(q, h, 0.10), conf.interval(q, h, 0.10)
    np.testing.assert_allclose(a[0], b[0])
    np.testing.assert_allclose(a[1], b[1])


def test_pinball_loss_matches_definition():
    y = np.array([1.0, 2.0, 3.0])
    q = np.array([2.0, 2.0, 2.0])
    # tau = 0.9: errors are -1, 0, +1 -> losses 0.1, 0, 0.9
    assert abs(pinball_loss(y, q, 0.9) - (0.1 + 0 + 0.9) / 3) < 1e-9
