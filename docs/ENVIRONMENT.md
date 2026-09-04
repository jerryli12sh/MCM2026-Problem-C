# Environment & install guide

This page records the *one* environment the release evidence was produced in, so an unfamiliar
third party can recreate it and trust that the recorded numbers come from a stated interpreter and
dependency set.

## Runtime used for the release evidence

| Item | Value |
|---|---|
| OS / platform | macOS (darwin), arm64 — wheel build matters for pin fidelity |
| Python interpreter | 3.13.3 (CPython) |
| Install mode | `pip install -e ".[analysis,dev]"` into a repo-local `.venv` |
| Packages installed | 37 third-party pins + `dwts-reproduction` (local editable) — see frozen snapshot |
| Release-evidence commit | `569994b` in the original workspace history |
| Frozen snapshot | [`requirements-lock.txt`](../requirements-lock.txt) |

> The snapshot is a *point-in-time* record of the exact environment that ran the recorded release
> and produced `outputs/release_manifest.json`. It is evidence of what was tested,
> not a promise that pip will resolve identically forever; see
> [Reproducibility limits](#reproducibility-limits).

## Supported Python versions

- `requires-python = ">=3.11"` — the language floor.
- 3.11-compatible *syntax* is enforced by `ruff` with `target-version = "py311"`.
- Static type-checking (`mypy`) and the documented release runs use **Python 3.13**. mypy targets
  3.13 (not the 3.11 floor) deliberately: installed numpy 2.x inline stubs use 3.12-only `type`
  statements, so a 3.11 mypy target fails inside numpy before it ever checks this package. The
  comment block in `pyproject.toml` records the full rationale.

Install with the interpreter you intend to run (3.13 recommended; 3.11–3.13 should work, but only
3.13 is gated/verified here).

## One-time install

```bash
# From this repository root (repo/). Requires python3 >= 3.11 on PATH.
python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -e ".[analysis,dev]"
```

or simply:

```bash
make install
```

`.[analysis,dev]` installs the runtime *and* analysis dependencies
(scipy, matplotlib, scikit-learn, statsmodels, xgboost) plus the dev toolchain (pytest, ruff, mypy).

### Torch is optional — and not needed for the release

The submitted paper's figures were produced with a PyTorch reference, but the reproduction models
are implemented in numpy/scipy with a hand-written Adam optimizer, so **no torch package is
required** to run any test, figure, or release stage. `pyproject.toml` still exposes a `torch`
extra for anyone who wants to cross-check against the reference line; it is intentionally absent
from the recorded snapshot.

## Verifying the install matches the recorded snapshot

```bash
.venv/bin/python --version   # expect 3.13.3
.venv/bin/pip freeze | grep -v -E "^-e |#egg=dwts|subdirectory=repo" \
  | diff - <(sed -e '/^#/d' -e '/^$/d' requirements-lock.txt) \
  && echo "identical to recorded snapshot"
```

The snapshot stores only *third-party* pins. The project's own editable self-line is excluded: pip
freeze renders an editable install by introspecting the enclosing git worktree (remote + HEAD +
subdirectory), so that line is machine- and commit-specific and would change for anyone who checks
out a different location or commit. The grep filters the same self-line out of a fresh `pip freeze`,
and the `sed` strips the snapshot's comment header, leaving the two pin lists to compare directly.

## Reproducibility limits (honest)

- The lock records the full resolved third-party set from 2026-09-03. A future install still depends
  on those exact releases and compatible wheels remaining available for the target platform.
- pandas/numpy/scipy wheels are platform-specific; an equivalent Linux wheel set will resolve to
  different build hashes. The recorded numbers should be reproduced *within registered tolerances*,
  not bit-for-bit, across platforms.
- The file pins versions, not wheel hashes. For supply-chain-grade reproducibility, add
  platform-specific hashes with a dedicated lockfile tool.

## Verification commands

```bash
.venv/bin/python -m ruff format --check .
.venv/bin/python -m ruff check .
.venv/bin/python -m mypy src/dwts_reproduction
.venv/bin/python -m pytest -q            # 226 tests in the public-ready tree
.venv/bin/python scripts/hash_inputs.py --validate
.venv/bin/python scripts/smoke_test.py
.venv/bin/python scripts/run_release.py --verify-only   # fast re-check of the full release
.venv/bin/python scripts/run_release.py                 # full release, ≈24 min
```

Use `make check` for the source-free public gate, `make verify-data` for the complete local gate,
and `make release` to regenerate every result. See [`CI.md`](CI.md) and
[`VERIFICATION.md`](VERIFICATION.md) for the scope and recorded evidence.
