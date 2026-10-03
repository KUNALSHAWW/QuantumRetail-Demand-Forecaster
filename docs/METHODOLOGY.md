# Methodology

This document explains the problem, the method and, most importantly, why the evaluation can be trusted.

## 1. The problem: sales are not demand

Fresh retail sells perishable goods in limited quantities. When a product sells out at 14:00, the zero sales for the rest of the day record that the shelf was empty, not that nobody wanted the product. Training a forecaster on those numbers teaches it to under-forecast exactly the products that sell out, which makes the next stockout more likely. This is called *censored demand*.

The dataset, [FreshRetailNet-50K](https://huggingface.co/datasets/Dingdong-Inc/FreshRetailNet-50K) (Wang et al., 2025), is unusual because it records **hourly stockout flags** next to hourly sales, so the censored hours are known rather than guessed.

| Property | Value |
|---|---|
| Series (store x product) | 50,000 (898 stores, 18 cities, 863 perishable SKUs) |
| Training history | 90 days, 2024-03-28 to 2024-06-25 |
| Official hold-out | 7 days, 2024-06-26 to 2024-07-02 (`eval` split, same series) |
| Per row | daily sales, 24 hourly sales, 24 hourly stockout flags, discount, promotion, holiday, weather |

Facts verified directly from the data and relied upon: `sale_amount` equals the sum of the 24 hourly values exactly; `hours_stock_status == 1` means *out of stock*; `stock_hour6_22_cnt` counts stockout hours in the 06:00 to 22:00 window; 44.3% of store-product-days contain a stockout and 4.0% are out of stock all day.

## 2. Latent demand recovery

For a day with sales `s_h` in each hour `h` and stockout flags `z_h`, we want the daily demand `D` that would have been observed with unlimited stock.

**Legacy method (v1).** Scale by the fraction of the 16-hour window that was in stock: `D = sales x 16 / (16 - stockout hours)`. This assumes demand is uniform through the day.

**Hourly-profile ratio estimator (v2).** Demand is not uniform; it peaks in the morning and late afternoon. Estimate each product's typical share of daily demand in each hour, `pi_h` (summing to 1), from days that **never stocked out**, shrunk towards the global profile when a product has few such days. Then

```
D = (sales observed in in-stock hours) / (sum of pi_h over in-stock hours)
```

The denominator is the share of normal daily demand that the in-stock hours account for, so the estimator scales up exactly as much as the missing hours warrant. The estimate is never allowed below observed sales. When the in-stock hours carry under 5% of normal demand (for example out of stock from opening), there is not enough signal and the day is imputed from the median of neighbouring valid days on the same series.

### Validation by controlled censoring

True demand on a stockout day is unknowable, so a method cannot be scored on real stockout days. Following the protocol of the FreshRetailNet paper, we validate on *simulated* stockouts instead:

1. take days that never stocked out, where observed sales equal true demand,
2. choose a sell-out hour drawn from the **empirical distribution** of real sell-out hours, and hide all sales after it,
3. apply each method to the censored day and compare with the known total.

The hourly profile is fitted on days 0 to 59 and the simulation runs on days 60 onward, so no day is used both to fit and to test. Results are in [BENCHMARKS.md](BENCHMARKS.md).

*Limitation.* Fully stocked days are, by selection, days where demand was low enough not to sell out, so they may differ from stockout days. The simulation measures the estimator under realistic censoring patterns; it cannot prove the estimate is unbiased on real stockout days.

## 3. Forecasting

A **global** LightGBM model is trained across series rather than one model per series. With 90 days per series a per-series model has under 80 training rows. Pooling lets the model learn shared structure (weekly seasonality, promotion and weather response, how stockouts distort history).

**Direct multi-horizon.** At forecast origin `t` (the last known day) the model predicts day `t + h` for `h = 1..7`, with `h` as a feature.

**Features.**
- History up to `t`: lags, 3/7/14/28-day means, volatility, zero-sales fraction, observed-vs-recovered gap, stockout intensity.
- Target day `u`: weekday, and covariates the retailer knows in advance (price discount, promotion flag, holiday, weather forecast), plus the same-weekday lags at `u-7 ... u-28`.
- Static: store, product, city, category hierarchy.

**Targets and history are recovered demand.** The same recovery is applied to the history features at inference, so training and serving agree.

**Objective.** L1 (median) regression, the natural fit for the WAPE metric (chosen on the calibration series, see [BENCHMARKS.md](BENCHMARKS.md)).

### No leakage

`tests/test_features.py` mutates everything after the forecast origin and asserts that no history feature changes, and that the same-weekday lags never reach past the origin. The hourly profile is fitted on days 0 to 60 only, before every training, calibration and test target.

## 4. Evaluation protocol

```
days   0 ................ 68 | 75 | 82 | 89 | 90 ..... 96
       training origins        valid  calib  last    official eval split
       (targets <= 75)         origin origin known   (test targets)
```

| Role | Series | Origins | Targets |
|---|---|---|---|
| Train | 10,000 | 27, 29, ..., 67 | up to day 75 |
| Early stopping | same 10,000 | 75 | days 76-82 |
| Conformal calibration | 5,000 different series | 82 | days 83-89 |
| **Test** | **35,000 series never seen above** | 89 | **days 90-96 (official eval split)** |

Series are disjoint across roles and time is strictly ordered, so a good test score cannot come from memorising a series or peeking at the future.

**What is scored against what.** Forecasts target demand. Demand is observable only on days without a stockout, so the primary comparison uses **test days with no stockout** (observed sales equal true demand). A secondary comparison against observed sales on all days is also stored in the result JSON.

**Metrics.** WAPE (sum of absolute errors over sum of actuals) and WPE (its signed version, the bias), as in the FreshRetailNet paper; MAE; pinball loss and empirical coverage for quantiles.

**Baselines.** Seasonal naive, 7-day moving average, the same model trained on raw sales (the ablation that isolates the effect of recovery), and the v1 style per-series LightGBM.

## 5. Prediction intervals

Eight quantile models (5%, 10%, 25%, 50%, 75%, 90%, 95%) give a predictive distribution. Raw quantile regression is typically mis-calibrated, so two **split conformal** corrections are fitted per horizon on the calibration series:

- **CQR** (Romano et al., 2019) for symmetric intervals: widen or narrow `[q_lo, q_hi]` by the finite-sample-corrected quantile of the conformity score `max(q_lo - y, y - q_hi)`, giving marginal coverage of at least `1 - alpha` under exchangeability.
- **One-sided recalibration** of each quantile level, used by the inventory layer, which needs the whole distribution.

Calibration uses stockout-free calibration days so the target is true demand. The guarantee needs calibration and test points to be exchangeable; here the test week follows the calibration week, so a small shift is expected and the benchmark reports the coverage actually achieved.

## 6. From forecast to decision

For a perishable product the order quantity trades a lost sale (underage cost `cu`) against a wasted unit (overage cost `co`). The newsvendor result says the cost-minimising stock is the demand quantile at the **critical fractile** `cu / (cu + co)`. With calibrated quantiles that is a direct lookup (linear interpolation between quantile knots).

Policies are scored on realised demand by **fill rate** (share of demand served), **waste rate** (share of stock unsold) and **cost**. They are compared at matched waste, and the mean-plus-`z`-sigma baseline is given its best `z` on the test data, which favours the baseline.

## 7. Explanations

`GlobalForecaster.contributions` returns exact tree-SHAP contributions (LightGBM's native `pred_contrib`), so the app can show which features pushed a particular forecast up or down. A test asserts the contributions plus the base value reproduce the raw model output.

## 8. Known limitations

- Three months of history: no yearly seasonality, and weather estimates are confounded with the calendar.
- Daily sales are close to one unit per series, so a large share of error is irreducible noise.
- Demand recovery is validated by simulation, not on real stockout days.
- Conformal calibration assumes exchangeability between the calibration and test weeks.
- Inventory is evaluated on stockout-free days only (the only days with known demand), which excludes the hardest days.
- A single data source (one retailer, one region, one quarter).

## References

- Wang et al. *FreshRetailNet-50K: A Stockout-Annotated Censored Demand Dataset for Latent Demand Recovery and Forecasting in Fresh Retail.* arXiv:2505.16319 (2025).
- Romano, Patterson, Candes. *Conformalized Quantile Regression.* NeurIPS (2019).
- Ke et al. *LightGBM: A Highly Efficient Gradient Boosting Decision Tree.* NeurIPS (2017).
- Arrow, Harris, Marschak. *Optimal Inventory Policy.* Econometrica (1951), the newsvendor model.
