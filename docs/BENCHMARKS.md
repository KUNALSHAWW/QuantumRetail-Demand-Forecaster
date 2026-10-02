# Benchmarks

Every number below is read from `benchmarks/results/*.json`, produced by `python -m quantumretail benchmark` and `python -m quantumretail eda`. This page is regenerated with `python scripts/render_benchmarks.py`.

**Protocol.** 10,000 series train the models, 5,000 calibrate the prediction intervals and **35,000 test series are never seen in either step**. The test targets are the 7 days of the dataset's official `eval` split (2024-06-26 to 2024-07-02), which come after every training and calibration target. See [METHODOLOGY.md](METHODOLOGY.md).

## 1. Latent demand recovery

True demand is unobservable on stockout days, so recovery is validated by *controlled censoring*: take days that never stocked out (observed sales equal true demand), hide the sales after a realistic sell-out hour, and measure how well each method reconstructs the known total. 300,000 simulated stockouts, profile fitted on days 0-59 only, evaluated on days 60+.

| Method | WAPE (lower is better) | Bias (WPE) |
|---|---:|---:|
| No recovery (raw sales) | 38.2% | -38.2% |
| Legacy formula, `sales x 16 / (16 - stockout hours)` | 30.3% | +7.7% |
| **Hourly-profile ratio estimator** | **25.8%** | **+0.3%** |

The profile estimator removes **15%** of the legacy method's error and eliminates its upward bias.

By how early the product sold out:

| Sold out from | n | No recovery | Legacy | Profile |
|---|---:|---:|---:|---:|
| 07:00 to 10:00 | 20,041 | 85.5% | 67.0% | 69.2% |
| 10:00 to 14:00 | 79,700 | 61.3% | 45.9% | 38.6% |
| 14:00 to 18:00 | 112,957 | 33.9% | 25.7% | 21.5% |
| 18:00 to 22:00 | 87,302 | 12.0% | 13.7% | 9.8% |

Two honest observations. The profile method is not best everywhere: when stock runs out in the first hours of the day there is very little observed signal, and the legacy formula is slightly better in the 07:00 to 10:00 bucket. And the legacy formula is *worse than doing nothing* when stockouts happen late in the day, because it assumes demand is spread evenly while real demand is concentrated in the morning and late afternoon.

**How much demand does raw sales data miss?** 44.3% of store-product-days had at least one stockout hour and 4.0% were out of stock for the whole selling day. On days with a stockout, raw sales miss an estimated **41.7%** of demand; across all days, **24.8%**.

## 2. Point forecast accuracy (7-day horizon)

Scored on test days with **no stockout**, where observed sales equal true demand. WAPE = sum of absolute errors / sum of actuals. Bias (WPE) is the signed version: negative means the model under-forecasts.

| Model | WAPE | Bias (WPE) | MAE |
|---|---:|---:|---:|
| Seasonal naive (same weekday last week) | 38.9% | -0.8% | 0.482 |
| 7-day moving average | 32.6% | -1.3% | 0.403 |
| Global LightGBM trained on raw sales | 32.3% | -14.2% | 0.400 |
| **Global LightGBM trained on recovered demand (this project)** | **31.6%** | **-3.4%** | 0.390 |
| Same model, calibrated median (q50) | 31.4% | -6.0% | 0.389 |

144,412 scored points. 
Training on recovered demand instead of raw sales cuts the bias from -14.2% to -3.4% and lowers WAPE from 32.3% to 31.6%. It also beats the 7-day moving average (32.6%), whose own bias (-1.3%) hides the same problem because the average contains censored days.

**Versus the v1 approach (one LightGBM per series).** On 300 test series: per-series models 38.5% WAPE, the global model 35.0%.

WAPE by forecast horizon (stockout-free days):

| Horizon (days ahead) | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 7-day moving average | 30.9% | 31.6% | 32.3% | 31.6% | 35.2% | 32.4% | 33.9% |
| Global, raw sales | 31.5% | 32.1% | 31.3% | 30.1% | 35.5% | 32.3% | 33.8% |
| Global, recovered demand | 31.1% | 31.5% | 31.2% | 29.7% | 33.1% | 32.0% | 32.9% |

Daily sales are small and noisy (about one unit per series per day), so a large part of the error is irreducible. Gains over a strong moving-average baseline are real but modest, and are reported as measured.

## 3. Prediction intervals

Coverage is the share of test outcomes that fall inside the interval. Conformal calibration (CQR) is fitted per horizon on the calibration series.

| Interval | Nominal | Raw quantile models | After conformal calibration | Mean width (raw / conformal) |
|---|---:|---:|---:|---:|
| 80% | 80.0% | 79.1% | 79.2% | 1.25 / 1.25 |
| 90% | 90.0% | 88.5% | 89.3% | 1.73 / 1.75 |

Per-quantile calibration (share of outcomes at or below each predicted quantile; ideal equals the level):

| Quantile | Raw | Recalibrated |
|---|---:|---:|
| 0.05 | 9.1% | 6.9% |
| 0.10 | 14.6% | 11.9% |
| 0.25 | 29.1% | 26.9% |
| 0.50 | 55.3% | 51.6% |
| 0.75 | 80.1% | 76.8% |
| 0.90 | 93.5% | 90.9% |
| 0.95 | 97.1% | 95.5% |

The raw quantile models over-cover the low quantiles; recalibration moves every level toward its target. Coverage is within about one point of nominal on a test week that comes after the calibration week.

## 4. Inventory decisions

A policy sets the stock for each day; it is scored against realised demand on 144,412 stockout-free test days (true demand known). The baseline is the textbook *forecast mean + z standard deviations*; **its safety factor z was tuned on the test data itself**, which favours the baseline.

**Fill rate at matched waste** (share of demand served, when the policy wastes the given share of stock):

| Waste | Quantile policy (conformal) | Quantile policy (uncalibrated) | Mean + z sigma baseline |
|---|---:|---:|---:|
| 15% | 82.9% | 83.0% | 81.8% |
| 20% | 87.8% | 87.8% | 87.0% |
| 25% | 91.0% | 91.1% | 90.5% |
| 30% | 93.5% | 93.5% | 93.1% |

**Newsvendor cost per day** (lost sale costs `cu`, a wasted unit costs `co`):

| Costs | Newsvendor policy | Best baseline (z tuned on test) | Moving average only |
|---|---:|---:|---:|
| cu=1, co=1 | 0.389 | 0.403 | 0.403 |
| cu=3, co=1 | 0.701 | 0.724 | 0.822 |
| cu=9, co=1 | 1.164 | 1.185 | 2.078 |

## 5. What drives demand (within-series estimates)

Each estimate compares a series with itself (so a big store is not mistaken for a promotion effect) on recovered demand. Associations over 90 days, not causal effects.

| Factor | Estimate |
|---|---|
| Weekend vs weekday | +33.4% (median series +27.9%) |
| Promotion active vs not | +40.1% on recovered demand, +32.0% on raw sales |
| Holiday vs not | +31.3% |
| Price elasticity | 1.66% more demand per 1% price cut |

Promotions look weaker in raw sales (+32%) than in recovered demand (+40%) because promoted products sell out sooner, which is exactly the censoring this project corrects.

## Things that did not help

Reported because negative results save the next person time. These experiments used 5,000 series drawn from the test pool, scored at the calibration origin (before the test week). No change from them was adopted, so the final test numbers are unaffected by them:

- Adding cross-sectional context (product-level and store-level recent demand, ratios to them): WAPE 0.3237 vs 0.3221 baseline.
- Doubling the training series from 10,000 to 20,000: 0.3242 vs 0.3221.
- Larger trees and a lower learning rate: 0.3273 vs 0.3221.
- Point objective: L1 (median) regression gave the best WAPE on the calibration series, 31.6% against 34.3% for L2, 34.2% for Tweedie and 34.4% for Poisson. That choice was made on the calibration series (before the test week), which are also used to calibrate the intervals.
