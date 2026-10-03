.PHONY: install test lint data recovery benchmark eda demo app api clean

PY ?= python

install:
	$(PY) -m pip install -r requirements-dev.txt && $(PY) -m pip install -e .

test:
	$(PY) -m pytest

lint:
	ruff check .

data:                     ## download FreshRetailNet-50K (about 115 MB)
	$(PY) -m quantumretail fetch

recovery:                 ## controlled-censoring validation of demand recovery
	$(PY) -m quantumretail validate-recovery

benchmark:                ## full forecasting + inventory benchmark (about 30 min on 12 CPU cores)
	$(PY) -m quantumretail benchmark

eda:                      ## within-series demand effects
	$(PY) -m quantumretail eda

demo:                     ## train the compact demo model and write the demo panel
	$(PY) -m quantumretail build-demo

app:
	streamlit run streamlit-app/app.py

api:
	uvicorn api.main:app --reload

clean:
	rm -rf .pytest_cache .ruff_cache build *.egg-info
