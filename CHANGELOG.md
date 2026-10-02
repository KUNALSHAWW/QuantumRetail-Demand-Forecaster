# Changelog

All notable changes are documented here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [2.0.0] - 2026-10-02

A rebuild around one question: how do you forecast demand when you can only see what sold, not what customers wanted?

### Added
- `quantumretail` package with a dense `(series, day, hour)` data layer for FreshRetailNet-50K.
- **Hourly-profile latent demand recovery** with a controlled-censoring validation (simulated stockouts on fully stocked days). On 300,000 simulated stockouts it reduces error from 30.3% (legacy formula) to 25.8% and removes the +7.7% bias.
- **Global multi-horizon LightGBM forecaster** with leak-free features, trained across series.
- **Conformal prediction intervals** (CQR) and per-quantile recalibration, with measured coverage.
- **Inventory decision layer**: newsvendor order quantities from calibrated quantiles, scored on fill rate, waste and cost.
- Exact SHAP explanations and a **what-if promotion** simulator.
- Rebuilt Streamlit dashboard, a FastAPI service and a CLI (`python -m quantumretail`).
- Reproducible benchmark with series-disjoint, time-ordered splits; `docs/METHODOLOGY.md` and generated `docs/BENCHMARKS.md`.
- Automated tests, GitHub Actions CI, `pyproject.toml`, Makefile.

### Changed
- Forecasts are now probabilistic. The old "confidence band" (plus or minus one residual standard deviation, never calibrated) is gone.
- Models are trained on recovered demand instead of raw sales. On stockout-free test days this cuts forecast bias from -14.2% to -3.4% and lowers WAPE from 32.3% to 31.6%.

### Removed
- The v1 per-series pipeline (`src/`), cached pickled models and the 67 MB processed parquet that was committed to git.
- Unsupported performance claims from the previous README (an "average across 100 pairs" table and several quoted percentages). They are replaced by results reproduced by `make benchmark`.

## [1.0.0] - 2025-12-06

Initial release: per-series LightGBM and XGBoost with a Streamlit dashboard.
