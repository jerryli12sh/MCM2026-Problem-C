# Continuous integration and data boundary

The repository has two verification levels because the public tree deliberately excludes the
official contest dataset, submitted paper source, and legacy workspace.

## Public source-free gate

GitHub Actions runs on every push and pull request:

- Ruff formatting and lint checks;
- mypy type checking for `src/dwts_reproduction`;
- byte-compilation of all modules and scripts;
- tests whose inputs are entirely committed to the repository.

The test selection covers configuration, hashing primitives, hand-worked mechanism rules, release
comparison logic, run-manifest validation, sensitivity utilities, and smoke/inventory logic. Six
individual cases are explicitly deselected because they open external data or generated release
outputs:

- `test_mechanism_phase.py::test_real_phase_metrics_structural_p`;
- `test_inventory_completeness.py::test_paper_figures_covered`;
- `test_inventory_completeness.py::test_paper_tables_covered`;
- `test_sensitivity.py::test_panel_with_variant_preserves_shape_on_real_data`;
- `test_smoke.py::test_raw_shape`;
- `test_smoke.py::test_run_smoke_checks_pass`.

Run the same gate locally with:

```bash
make check
```

## Owner/data-bound gate

With an authorized source bundle available through `DWTS_SOURCE_ROOT` (or as the repository's parent
directory), run:

```bash
make verify-data
```

That command collects the complete 226-test suite, validates all 174 input hashes, and runs raw-data
smoke checks. On a cleaned tree, 27 tests that require generated release files skip. `make release`
regenerates the full 19-stage analysis and figure set and performs the 20-row baseline comparison;
`make release-verify` rechecks an existing release.

The earlier recorded release had 229 tests. Three of those asserted that this directory was nested
inside the owner's original monorepo. They were intentionally removed during publication cleanup:
such tests fail by design after this directory becomes the public Git root and say nothing about the
analysis.

## Why the public gate does not download data

Automatically fetching an unofficial copy would weaken provenance and could violate redistribution
terms. A fake miniature dataset would test only plumbing, not the registered numerical claims. The
project therefore makes the boundary explicit: pure logic is publicly checked; data-bound claims are
checked against a locally authorized, SHA-256-pinned source bundle.

## Dependency stability

CI uses Python 3.13 and pins Ruff/mypy to the versions used during release verification. The broader
environment is recorded in `requirements-lock.txt`. Numerical equality is judged by
registered tolerances; PNG byte identity is not assumed across operating systems.
