# Phase 0 acceptance packet

Phase 0 (baseline & provenance) is implemented. This packet is evidence, not a narrative
claim. The owner reviews it and the independent audit (`../prompts/PHASE0_AUDIT.md`) verifies
it before Phase 1 begins.

## Scope completed

Locked the read-only inputs, inventoried the paper/review/legacy assets, registered the
paper-output baseline, and stood up the package/test/config skeleton plus the run-manifest
discipline. No preprocessing, modeling, or figure code was written.

## Gate result

`make phase0-accept` passes end to end:

```
ruff format --check .   -> 35 files already formatted
ruff check .            -> All checks passed!
mypy src/dwts_reproduction -> Success (5 files)
pytest -q               -> 24 passed
hash_inputs.py --validate -> OK (manifest matches sources)
smoke_test.py           -> SMOKE TEST: PASS (raw shape 421x53; legacy hash unchanged)
check_scope.py          -> OK (0 staged paths outside repo/)
```

## Files created / modified (all under `repo/`)

- Skeleton: `pyproject.toml`, `Makefile`, `configs/paths.yaml`, `configs/phase0.yaml`.
- Package: `src/dwts_reproduction/{__init__,config,hashing,run_manifest,smoke}.py`.
- Scripts: `scripts/{hash_inputs,smoke_test,check_scope,inventory_sources,build_traceability,build_conflict_matrix,build_baseline}.py`.
- Tests: `tests/{test_config,test_hashing,test_run_manifest,test_smoke,test_inventory_completeness,test_scope}.py`.
- Docs: `DATA_DICTIONARY.md`, `LEGACY_INVENTORY.md` (generated), `TRACEABILITY_PAPER.md`,
  `TRACEABILITY_REVIEW.md`, `TRACEABILITY_COVERAGE.md`, `CONFLICT_MATRIX.md`,
  `BASELINE_PAPER_OUTPUTS.md`, `RUN_MANIFEST.md`, `PHASE0_ACCEPTANCE.md`; modified `DECISIONS.md`.
- Manifests: `manifests/input_manifest.sha256` (174 entries), `traceability_paper.csv`,
  `traceability_review.csv`, `conflict_matrix.csv`, `baseline.csv`, `legacy_inventory.csv`.

## Inventory counts (programmatically generated)

| artifact | count |
|---|---|
| Input manifest entries (hashed) | 174 |
| Excluded (caches/build artifacts) | 13 |
| `raw_input` / `paper_spec` / `review_spec` | 5 / 68 / 7 |
| `legacy_impl` / `legacy_output` / `reference` | 34 / 58 / 2 |
| `src/` `.py` / `.ipynb` / `.pyc` | 22 / 9 / 8 |
| Paper traceability rows | 96 |
| Review traceability rows | 35 |
| Conflicts | 6 |
| Baseline rows | 17 |

## Provenance & scope safety

- Input hashes recorded and re-validated (`hash_inputs.py --validate`).
- No legacy file modified: `git status --porcelain -- data/ src/ paper_Latex/ figure/` is empty.
- Repository-scope: every staged/created path is under `repo/`; the parent repo's untracked
  `review/`, `Latex/`, `*.zip`, `*.docx`, and the pre-existing `paper.pdf` deletion are untouched.
- No secret, virtual environment, cache, or generated output is tracked (`.venv/`, `*.egg-info/`,
  `outputs/**`, `__pycache__/` are gitignored).

## Decisions & known gaps

Six conflicts recorded in `docs/CONFLICT_MATRIX.md` with decision IDs `D-20260901-01..06`
(see `docs/DECISIONS.md`). The era-mapping direction (`D-20260901-01`) is **suspected**, not
established — the legacy era-assignment code is inspected in Phase 1. Tolerances in
`docs/BASELINE_PAPER_OUTPUTS.md` are proposed, pending owner approval. `track_P_module` /
`track_R_module` / `acceptance_test` fields are `planned` targets, not existing code.

## Rerun command

```bash
cd repo && make install && make phase0-accept
```

## Independent audit command

Start a fresh Claude session from `repo/`, read `../prompts/PHASE0_AUDIT.md`, and audit Phase 0
against the paper (`../paper_Latex/2107542.tex`), the review (`../review/notes/review_all.md` and
the refactor prompts), the complete git diff, the tests, the manifests, and this packet.
