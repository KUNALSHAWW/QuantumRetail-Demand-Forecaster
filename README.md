<div align="center">

# QuantumRetail

### Forecast what customers wanted, not just what sold

Stockout-aware probabilistic demand forecasting with calibrated prediction intervals and inventory decisions, built on the FreshRetailNet-50K dataset

[![CI](https://github.com/KUNALSHAWW/QuantumRetail-Demand-Forecaster/actions/workflows/ci.yml/badge.svg)](https://github.com/KUNALSHAWW/QuantumRetail-Demand-Forecaster/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![LightGBM](https://img.shields.io/badge/LightGBM-quantile_regression-02569B)](https://lightgbm.readthedocs.io/)
[![Conformal](https://img.shields.io/badge/uncertainty-conformal_prediction-6f42c1)](docs/METHODOLOGY.md#5-prediction-intervals)
[![Streamlit](https://img.shields.io/badge/app-Streamlit-FF4B4B?logo=streamlit&logoColor=white)](streamlit-app/app.py)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

[The problem](#the-problem) | [What is different](#what-is-different) | [Results](#results) | [Quick start](#quick-start) | [How it works](#how-it-works) | [Limitations](#limitations)

</div>

---

## The problem

A fresh-food retailer sells out of strawberries at 2 pm. The sales record for the rest of the day says **zero**. A model trained on that history concludes that strawberries are unpopular on that day, forecasts less, and the shelf is empty again tomorrow. Forecasting from sales instead of demand is a self-reinforcing mistake, and in this dataset it is not a corner case: **44% of store-product-days contain a stockout, and raw sales miss an estimated 42% of demand on those days.**

QuantumRetail treats that gap as the central problem: reconstruct the demand that went unobserved, forecast it with honest uncertainty, and turn the forecast into a stocking decision.

## What is different

| Typical demand-forecasting project | QuantumRetail |
|---|---|
| Trains on sales as if sales were demand | **Recovers latent demand** from hourly stockout flags, and *validates the recovery* with simulated stockouts |
| One model per series (about 80 training rows each) | **One global model** across 10,000 series that generalises to series it has never seen |
| A single number, or a "confidence band" of plus or minus one standard deviation | **Conformally calibrated intervals** with measured coverage |
| Stops at the forecast | **Newsvendor inventory decisions** scored on fill rate, waste and cost |
| Accuracy quoted on data the model has seen | **35,000 held-out series** on the dataset's official evaluation week, and negative results reported |
| Black box | Exact **SHAP explanations** and a **what-if promotion** simulator |

## Results

All results come from `python -m quantumretail benchmark` and are reproduced in [docs/BENCHMARKS.md](docs/BENCHMARKS.md). The 35,000 test series were never used for training or calibration, and the test week comes after everything the models saw.

**1. Demand recovery.** Validated by hiding sales after a realistic sell-out hour on days that never stocked out (300,000 simulated stockouts):

| Method | Error (WAPE) | Bias |
|---|---:|---:|
| No recovery (raw sales) | 38.2% | -38.2% |
| Legacy formula `sales x 16 / (16 - stockout hours)` | 30.3% | +7.7% |
| **Hourly-profile estimator (this project)** | **25.8%** | **+0.3%** |

The new estimator removes 15% of the legacy method's error and its bias. It is not best everywhere (the legacy formula wins slightly when stock runs out in the first hours of the day) and the legacy formula is worse than doing nothing when stockouts happen late in the day. Both are documented.

**2. Forecast accuracy** (7-day horizon, scored on days with no stockout, where observed sales equal true demand):

| Model | WAPE | Bias |
|---|---:|---:|
| Seasonal naive | 38.9% | -0.8% |
| 7-day moving average | 32.6% | -1.3% |
| Global LightGBM on **raw sales** | 32.3% | **-14.2%** |
| **Global LightGBM on recovered demand** | **31.6%** | **-3.4%** |

Training on recovered demand instead of sales cuts the forecast bias from -14.2% to -3.4% and also lowers the error. On a like-for-like sample, one-model-per-series (the v1 approach) scores 38.5% against 35.0% for the global model. Daily sales are small and noisy (about one unit per series per day), so the gains over a strong moving-average baseline are real but modest, and are reported exactly as measured.

**3. Prediction intervals.** The 80% interval covers **79.2%** of outcomes and the 90% interval covers **89.3%**, on a week that comes after the calibration week.

**4. Inventory.** Ordering at the newsvendor critical fractile of the calibrated forecast costs **2 to 3% less per day** than the best-tuned "mean plus z standard deviations" rule and serves 0.5 to 1.1 more points of demand at the same waste. The baseline's safety factor was tuned on the test data, which favours the baseline.

**5. What drives demand** (within-series, recovered demand): weekends **+33%**, promotions **+40%** (they look like +32% in raw sales because promoted items sell out sooner), holidays +31%, price elasticity about 1.7.

## Quick start

```bash
git clone https://github.com/KUNALSHAWW/QuantumRetail-Demand-Forecaster.git
cd QuantumRetail-Demand-Forecaster
pip install -r requirements-dev.txt && pip install -e .

streamlit run streamlit-app/app.py      # dashboard; runs on the bundled demo model and 400 held-out series
python -m quantumretail forecast --store 2 --product 774       # CLI forecast and order plan
uvicorn api.main:app                    # REST API, docs at /docs
make test                               # 47 tests, about 15 seconds, no data download
```

To reproduce the benchmark from scratch:

```bash
make data         # downloads FreshRetailNet-50K (about 115 MB) from Hugging Face
make benchmark    # about 30 minutes on 12 CPU cores
make eda
make demo         # rebuilds the compact demo model
```

## The dashboard

- **Forecast**: history with observed vs recovered demand, a 7-day forecast with 80% and 90% calibrated bands, and the actuals when you back-test.
- **Inventory plan**: set the cost of a lost sale and of waste; get the cost-optimal order per day and the expected fill rate, waste and cost.
- **Why this forecast**: exact SHAP contributions for any forecast day.
- **What-if**: switch on a promotion and a price factor and watch the forecast and the order plan change.
- **Stockouts and recovery**: an hour-by-hour stockout heatmap and the product's demand profile.
- **Model card**: the benchmark evidence, from the JSON files.

## How it works

```
 hourly sales + hourly stockout flags
            |
            v
 hourly-profile recovery  --->  recovered daily demand        (validated by simulated censoring)
            |
            v
 leak-free features at origin t  (history, calendar, known-future promo / price / weather)
            |
            v
 global LightGBM: point (median) + 7 quantile models, horizons 1..7
            |
            v
 split-conformal calibration per horizon  --->  intervals with measured coverage
            |
            v
 newsvendor critical fractile  --->  order quantity, expected fill rate / waste / cost
```

[docs/METHODOLOGY.md](docs/METHODOLOGY.md) explains each step, the evaluation protocol and why it can be trusted.

## REST API

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/forecast?store=&product=` | 7-day forecast with calibrated quantiles and intervals |
| `POST` | `/order-plan` | newsvendor order quantities for given lost-sale and waste costs |
| `POST` | `/what-if` | forecast under a discount and promotion scenario |
| `GET` | `/explain?store=&product=&horizon=` | SHAP contributions |
| `GET` | `/series`, `/health` | available series, service status |

## Project structure

```
quantumretail/      data layer, recovery, features, model, conformal, inventory, backtest, service, CLI
streamlit-app/      the dashboard
api/                FastAPI service
tests/              47 tests: recovery, leakage, conformal coverage, inventory maths, API, app smoke test
benchmarks/results/ the benchmark JSON that every published number is read from
docs/               METHODOLOGY.md and generated BENCHMARKS.md
models/qr_v2/       compact demo model (about 8 MB)
data/demo/          400 held-out series for the demo (about 0.5 MB)
notebook/           the original exploratory analysis
```

## Tests

`make test` runs the suite on small synthetic data. Besides ordinary unit tests it checks properties that matter for correctness:

- the profile estimator recovers demand **exactly** when the profile is known, and beats the legacy formula under controlled censoring
- **no feature leaks the future**: everything after the forecast origin is overwritten and history features must not change
- conformal intervals restore nominal coverage on deliberately over-confident quantiles
- SHAP contributions sum to the model output
- the REST API and the Streamlit app run end to end

## Limitations

- Three months of data: no yearly seasonality, and weather effects are confounded with the calendar.
- Recovery is validated by simulation on fully stocked days, which are low-demand by selection; it cannot prove the estimate is unbiased on real stockout days.
- Forecast and inventory results are scored on stockout-free days, the only days where demand is known. The hardest days are excluded.
- Conformal guarantees assume the calibration and test weeks are exchangeable; the benchmark reports the coverage actually achieved.
- A single retailer, region and quarter. Transfer to other data is untested.

## Cite

The dataset: Wang et al., *FreshRetailNet-50K: A Stockout-Annotated Censored Demand Dataset for Latent Demand Recovery and Forecasting in Fresh Retail*, arXiv:2505.16319 (2025). Intervals use Conformalized Quantile Regression (Romano, Patterson and Candes, NeurIPS 2019).

## Author

**Kunal Kumar Shaw**: [GitHub](https://github.com/KUNALSHAWW) | [Portfolio](https://kunalshaw.vercel.app/)

Released under the MIT License.
