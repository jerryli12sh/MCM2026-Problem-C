# Common development and release commands.

VENV := .venv
PY := $(VENV)/bin/python
PIP := $(VENV)/bin/pip

HERMETIC_TESTS := \
	tests/test_config.py \
	tests/test_hashing.py \
	tests/test_mechanism_phase.py \
	tests/test_problem2_rules.py \
	tests/test_release_compare.py \
	tests/test_run_manifest.py \
	tests/test_inventory_completeness.py \
	tests/test_sensitivity.py \
	tests/test_smoke.py

DATA_DESELECTS := \
	--deselect tests/test_mechanism_phase.py::test_real_phase_metrics_structural_p \
	--deselect tests/test_inventory_completeness.py::test_paper_figures_covered \
	--deselect tests/test_inventory_completeness.py::test_paper_tables_covered \
	--deselect tests/test_sensitivity.py::test_panel_with_variant_preserves_shape_on_real_data \
	--deselect tests/test_smoke.py::test_raw_shape \
	--deselect tests/test_smoke.py::test_run_smoke_checks_pass

.PHONY: venv install format format-check lint type compile test test-hermetic check \
	verify-inputs smoke release-verify verify-data release clean

venv:
	python3 -m venv $(VENV)
	$(PIP) install --upgrade pip

install: venv
	$(PIP) install -e ".[analysis,dev]"

format:
	$(PY) -m ruff format .

format-check:
	$(PY) -m ruff format --check .

lint:
	$(PY) -m ruff check .

type:
	$(PY) -m mypy src/dwts_reproduction

compile:
	$(PY) -m compileall -q src scripts

test:
	$(PY) -m pytest -q

test-hermetic:
	$(PY) -m pytest -q $(HERMETIC_TESTS) $(DATA_DESELECTS)

# Runs in a public clone without the external contest source bundle.
check: format-check lint type compile test-hermetic

verify-inputs:
	$(PY) scripts/hash_inputs.py --validate

smoke:
	$(PY) scripts/smoke_test.py

release-verify:
	$(PY) scripts/run_release.py --verify-only

# Requires the external read-only source bundle. Output-dependent tests skip until a release exists.
verify-data: format-check lint type compile test verify-inputs smoke

# Requires the source bundle; regenerates the full release under outputs/.
release:
	$(PY) scripts/run_release.py

# Only ignored, reproducible artifacts are removed.
clean:
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	rm -rf .pytest_cache .mypy_cache .ruff_cache src/*.egg-info
	find outputs -mindepth 1 ! -name .gitkeep -delete
