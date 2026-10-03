# Contributing

Thanks for taking an interest. Bug reports, ideas and pull requests are welcome.

## Setup

```bash
git clone https://github.com/KUNALSHAWW/QuantumRetail-Demand-Forecaster.git
cd QuantumRetail-Demand-Forecaster
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
make install                                           # dev dependencies + editable install
make test                                              # about 15 seconds, no data download needed
make lint
```

The tests use small synthetic data, so they need neither the dataset nor a GPU. To reproduce the published results:

```bash
make data         # downloads FreshRetailNet-50K (about 115 MB)
make benchmark    # about 30 minutes on 12 CPU cores
make demo         # rebuilds the compact model used by the app
```

## Project layout

| Path | Purpose |
|---|---|
| `quantumretail/recovery.py` | Latent demand recovery and its controlled-censoring validation |
| `quantumretail/features.py` | Leak-free multi-horizon feature construction |
| `quantumretail/model.py`, `conformal.py` | Global LightGBM quantile models and conformal calibration |
| `quantumretail/inventory.py` | Newsvendor decisions and policy scoring |
| `quantumretail/backtest.py` | The evaluation protocol and benchmark runner |
| `quantumretail/service.py` | Inference API used by the app and the REST API |
| `streamlit-app/`, `api/` | Dashboard and REST service |
| `docs/` | Methodology and generated benchmark tables |

## Ground rules

- **Claims need evidence.** A number that appears in the README or docs must come from a file in `benchmarks/results/`. Regenerate `docs/BENCHMARKS.md` with `python scripts/render_benchmarks.py` rather than editing it by hand.
- **Protect the evaluation.** Do not tune anything on the test series. If you add a model or feature, compare it on the calibration series and report negative results too.
- **No leakage.** New features must pass `tests/test_features.py`. History features may use only days up to the forecast origin; only covariates known in advance may look ahead.
- Keep changes small and tested: `make test` and `make lint` must pass.
- Commit messages: a short imperative summary line, then an explanation of *why* if it is not obvious.

## Ideas that would be valuable

- Evaluate on a second dataset (for example M5) to test how well the method transfers.
- Hierarchical reconciliation across store, city and category.
- Learn an uncertainty-aware order policy directly instead of reading a quantile.
- Validate demand recovery against any real data with ground-truth unconstrained demand.
