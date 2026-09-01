# DWTS reproduction — Phase 0 development commands.
#
# All commands run inside the repository-local virtual environment. `make install` sets it
# up. `make phase0-accept` runs every Phase 0 gate without modifying any file.

VENV := .venv
PY := $(VENV)/bin/python
PIP := $(VENV)/bin/pip

.PHONY: venv install format lint type test verify-inputs smoke check-scope phase0-accept

venv:
	python3 -m venv $(VENV)
	$(PIP) install --upgrade pip

install: venv
	$(PIP) install -e ".[dev]"

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
