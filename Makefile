# DWTS reproduction — development commands.
#
# All commands run inside the repository-local virtual environment. `make install` sets it
# up with every dependency needed to run the full analysis AND the dev toolchain
# (`pip install -e ".[analysis,dev]"`); the core package itself is numpy/pandas only, but the
# full pipeline needs scipy/matplotlib/scikit-learn/statsmodels/xgboost too. Torch is NOT
# required (the softmin fit uses a hand-written Adam; see pyproject.toml). `make phase0-accept`
# runs every gate without modifying any file.

VENV := .venv
PY := $(VENV)/bin/python
PIP := $(VENV)/bin/pip

.PHONY: venv install format lint type test verify-inputs smoke check-scope phase0-accept

venv:
	python3 -m venv $(VENV)
	$(PIP) install --upgrade pip

install: venv
	$(PIP) install -e ".[analysis,dev]"

format:
	$(PY) -m ruff format .

lint:
	$(PY) -m ruff check .

type:
	$(PY) -m mypy src/dwts_reproduction

test:
	$(PY) -m pytest -q

verify-inputs:
	$(PY) scripts/hash_inputs.py --validate

smoke:
	$(PY) scripts/smoke_test.py

check-scope:
	$(PY) scripts/check_scope.py

phase0-accept:
	$(PY) -m ruff format --check .
	$(PY) -m ruff check .
	$(PY) -m mypy src/dwts_reproduction
	$(PY) -m pytest -q
	$(PY) scripts/hash_inputs.py --validate
	$(PY) scripts/smoke_test.py
	$(PY) scripts/check_scope.py
