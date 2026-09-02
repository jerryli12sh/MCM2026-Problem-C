# CI: what runs where, and why

This repository has **two distinct test gates**, and the boundary between them is not an
accident of configuration — it follows from what this repository is allowed to *contain*.

## The two-tier gate

| Gate | What it runs | Where it runs |
|---|---|---|
| **Hermetic gate** | Formatting (`ruff format --check`), lint (`ruff check`), typing (`mypy`), byte-compile of every module, and an 81-test selection that needs only committed repo files. | **Public CI** (`.github/workflows/ci.yml`) on every push/PR, on a source-less clone. |
| **Data-bound gate** | The full **229-test** suite, the 20-row baseline comparison (`run_release.py --verify-only` / full), and `hash_inputs.py --validate` (174 input hashes). | **Owner's machine** (and any trusted machine holding the read-only source bundle) via `make phase0-accept`. |

## Why there are two gates

The full suite and the release pipeline read the **external read-only source bundle** —
the contest data (`2026_MCM_Problem_C_Data.csv`, `data_3.csv`), the submitted paper
(`2107542.tex`), the consolidated review note, the legacy implementation, and the legacy
outcomes: **174 hashed files** listed in `manifests/legacy_inventory.csv` and pinned by
`manifests/input_manifest.sha256`. Those inputs live *outside* this repository
(`configs/paths.yaml`, `source_root: ..`) and are **not redistributed** with it — the
contest dataset's terms are the owner's to confirm, and this artifact ships a code
reproduction, not the contest materials.

A standalone public clone therefore cannot hold those files, and GitHub Actions cannot
fetch them. A CI job that ran the whole suite on a source-less checkout would not pass —
it would fail on `FileNotFoundError` for the very files it is meant to verify. The honest
design is to run everything that *can* run without the bundle, document exactly what is
excluded and why, and keep the authoritative full gate on the machine that holds the
bundle. Nothing is silently skipped: every deselected test is named below.

## What the hermetic 81 actually covers

`tests/test_config.py` (5) · `tests/test_hashing.py` (4) · `tests/test_mechanism_phase.py`
(9) · `tests/test_problem2_rules.py` (14) · `tests/test_release_compare.py` (22) ·
`tests/test_run_manifest.py` (7) · `tests/test_scope.py` (3) ·
`tests/test_inventory_completeness.py` (2) · `tests/test_sensitivity.py` (14) ·
`tests/test_smoke.py` (1).

That selection exercises the mechanism **rule functions** (rank/percentage ×
direct/bottom-2+save), the **release-comparison engine** (all 20 baseline-row checks),
the **run-manifest schema**, the **path/scope guardrails**, the **hashing primitives**,
and the pure-logic portions of the mechanism-phase, sensitivity, smoke, and inventory
code — everything whose inputs are committed.

The six tests deselected inside those modules, and the modules not collected at all
(`test_preprocess.py`, all `test_problem1_*.py`, `test_problem2_replay.py`,
`test_problem3.py`, `test_problem4.py`), read the external bundle or outputs generated
from it:

- `test_mechanism_phase.py::test_real_phase_metrics_structural_p` — reads fitted phase
  outputs.
- `test_sensitivity.py::test_panel_with_variant_preserves_shape_on_real_data` — builds a
  panel on real data.
- `test_inventory_completeness.py::{test_paper_figures_covered, test_paper_tables_covered}`
  — cross-check the 174-file inventory against the files on disk.
- `test_smoke.py::{test_raw_shape, test_run_smoke_checks_pass}` — read the raw data file
  and run the smoke comparison against it.

## Determinism

`ruff` and `mypy` are **pinned to exact versions** (`ruff==0.16.5`, `mypy==2.3.1`) in the
workflow because both evolve between releases: an unpinned `ruff format --check` can go
red from a formatting rule change, not a defect. The broader third-party environment that
produced the recorded release is frozen in
[`docs/python313-release-freeze-20260903.txt`](python313-release-freeze-20260903.txt).

CI is a **consistency gate**, not the release evidence. The authoritative numbers are the
recorded release run and its manifests (see [`docs/STATUS.md`](STATUS.md) and
[`docs/PHASE7_ACCEPTANCE.md`](PHASE7_ACCEPTANCE.md)), which CI does not recreate.

## The data-bound gate (owner)

With the source bundle in place (a directory whose `data/`, `src/`, `review/`,
`paper_Latex/` sit next to each other — pointed at via `$DWTS_SOURCE_ROOT` or by placing
the bundle as this repository's parent), the full gate is:

```bash
.venv/bin/python scripts/hash_inputs.py --validate  # 174 input hashes
.venv/bin/python -m pytest -q                        # 229 tests
.venv/bin/python scripts/run_release.py --verify-only # 20/20 baselines (~min)
.venv/bin/python scripts/run_release.py               # full 19-stage release (~23 min)
# or, for the whole correctness gate at once:
make phase0-accept
```

See [`docs/ENVIRONMENT.md`](ENVIRONMENT.md) for the install and snapshot-verification
recipe.

## Maintenance rules

- A test that reads the source bundle **belongs in a data-bound module** (or, if it lives
  in a hermetic module, must be added to the `--deselect` list in
  `.github/workflows/ci.yml` and documented here). If it is neither, CI will error on it —
  loudly, never silently.
- A new hermetic test may be added to any module listed above; CI picks it up
  automatically.
- If `ruff`/`mypy` versions are upgraded in `pyproject.toml`, update the pins in the
  workflow to the versions that actually produced the recorded release, and re-run
  `make phase0-accept` locally before relying on CI.
