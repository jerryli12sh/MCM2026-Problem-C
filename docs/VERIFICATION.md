# Release verification

This page records the checks for the publication-cleaned repository. Historical model evidence is
reconciled in [`STATUS.md`](STATUS.md); this page concerns the current tree's integrity.

## Current gates

| Gate | Expected result | Scope |
|---|---|---|
| `make check` | format, lint, typing, compilation, and source-free tests pass | any public clone |
| `pytest -q` | 226 tests collected; on the cleaned tree, 199 passed and 27 output-dependent cases skipped | owner/data-bound environment |
| `scripts/hash_inputs.py --validate` | 174 external inputs match their registered SHA-256 | owner/data-bound environment |
| `scripts/run_release.py --verify-only` | 20/20 registered baselines pass | owner/data-bound environment |
| `scripts/run_release.py` | 19/19 stages complete | owner/data-bound environment |

Publication check on 2026-09-04: the tree was copied into a standalone directory without the outer
MCM workspace or source bundle. Formatting, lint, typing, and compilation passed; the public test
selection reported **78 passed, 6 explicitly deselected**.

The recorded numerical release used Python 3.13.3 on macOS arm64 and took 1375.8 seconds. It
produced the Track P/Track R values in `STATUS.md`. The complete dependency snapshot is
[`requirements-lock.txt`](../requirements-lock.txt).

After `make release` regenerates `outputs/`, the skipped output-dependent cases become runnable. The
recorded pre-cleanup suite had 229/229 passing tests; removing three monorepo-layout checks leaves 226
analysis tests in the publication tree.

## Publication hygiene

- raw contest data, submitted-paper source, and legacy code are not part of the public subtree;
- generated `outputs/`, virtual environments, caches, bytecode, and editable-install metadata are
  ignored;
- tracked content contains no machine-specific user-home path or credential pattern;
- the public Git root is expected to be this directory after the owner publishes the `repo/` subtree;
- the `review_all.md` mirror is pinned as binary-like (`-text`) and retains SHA-256
  `a0e265acb9c36bb5d3acd5bde0b3ec0a6798b2e93e75e5bc5996e950e3070ea5`.

## Reproducibility boundary

The source-free gate proves code quality and pure-logic behavior. It cannot independently recreate
the contest results because the public repository intentionally omits the input bundle. Numerical
reproduction therefore requires an authorized local copy of that bundle and should be judged
against registered tolerances, not cross-platform byte identity of PNG files.
