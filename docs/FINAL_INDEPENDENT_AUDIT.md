# Final independent audit

**Date:** 2026-09-03 · **Scope:** the publishable `repo/` subtree on `reproduction/main` (recorded
release `569994b`), audited as the release candidate for public publication.

This document records the independent pre-release audit and the disposition of every finding. Its
conclusion feeds the owner-acceptance decision; nothing here is a substitute for the owner's
sign-off (see [`docs/STATUS.md`](STATUS.md) → *Process status*).

**Verdict:** the repository was found **publishable with three minor documentation/provenance
findings**, all of which are resolved in this close-out commit (see *Findings & resolutions*).
No finding affected analysis code, a formula, a sample definition, a target metric, or a research
conclusion; no secret, private reference, or dishonest framing was found anywhere in the tracked
tree.

---

## Method — two independent passes

The audit was run as two passes that derived their conclusions **independently from the tree**, so a
shared error or a shared blind spot is less likely to survive:

1. **Repository-lead self-audit** (this session) — automated and manual checks across the release
   checklist: review-mirror byte integrity, secret/PII scan of every tracked file, manifest ground
   truth, document reconciliation, release-drift re-check, tracked-tree hygiene, and full-suite and
   public-topology test runs.
2. **Independent auditor agent** — a fresh-context agent, pointed only at the repository and the
   publication constraints, that re-derived everything itself: it built a faithful **split-out public
   clone** (git root = the repository itself, no monorepo parent, no source bundle), ran the exact
   `.github/workflows/ci.yml` hermetic selection, scanned for secrets and private references, checked
   internal consistency across every document and manifest, reviewed the honesty of every headline
   claim, byte-verified the review mirror, and checked git hygiene.

## Pass 1 — repository-lead audit (selected results)

| Check | Result |
|---|---|
| Review mirror byte-integrity across history | SHA-256 `a0e265acb9c36bb5d3acd5bde0b3ec0a6798b2e93e75e5bc5996e950e3070ea5`, `cmp`-identical to the source of record, byte-stable since it was added (`.gitattributes` pins `review/notes/review_all.md -text`) |
| Secret / PII / machine-path scan of tracked files | Clean — no credentials, API keys, private keys, owner identity, or `/Users/…` absolute paths in tracked content |
| Manifest ground truth (CSV-parsed) | paper 96 (all `implemented`) · review 40 (all `implemented`) · baseline 20 · conflict 7 · legacy inventory 174 · input sha 174 · decisions 24 (D-20260901-01..24) — every count matches the claims |
| STATUS ↔ README reconciliation | Headline numbers identical (P 0.9495/0.6043/0.7785/3.117 · R 0.8349/0.5342/0.6331/3.378); no metric without a track label |
| Release drift re-check | `run_release.py --verify-only` → `checked=20 pass=20 fail=0 release_ok=True` at `bae7671` |
| Tracked-tree hygiene | Lean (≈2.8 MB); no caches, venvs, `.pyc`, or source-bundle files tracked; `outputs/` ships only `.gitkeep` |
| Full suite / public topology | 229 tests pass at `bae7671`; hermetic 78 pass in the true split-out public topology (see Pass 2 / A) |

## Pass 2 — independent auditor (verbatim conclusions)

**A. Hermetic gate (ran in `/tmp/dwts_public`):** PASS. Exact ci.yml selection → `78/84 tests
collected (6 deselected)`, **78 passed**, 0 failed. `ruff check .` → All checks passed.
`ruff format --check .` → 103 files already formatted. `mypy src/dwts_reproduction` → Success
(39 files). `compileall src scripts` → OK. All 6 `--deselect` IDs exist in their modules (not stale
no-ops); CI.md's per-module counts sum to 78 and match actual collection.

**B. Secrets / private refs:** No `/Users/`, `/home/`, owner name, credentials, API keys, tokens, or
private keys anywhere in tracked files. All "jerry" hits are contestant **Jerry Rice** (data
content). No `plan0.txt`/`data.zip`/`src.zip`/`*.docx`/`paper.pdf`; no source-bundle data/tex/csv
files tracked; `outputs/` physically contains only `.gitkeep`.

**C. Internal consistency:** PASS. README headline numbers equal STATUS.md exactly; 229 tests,
ruff/mypy clean, git 569994b, 1375.8 s, 20/20 `release_ok=True` all agree; CI.md's module list,
counts, 6 deselects, and `test_scope.py` exclusion rationale match ci.yml exactly; manifest rows
match STATUS.md; README repo-map paths all exist; **0 broken markdown links**; public-clone tracked
tree is byte-identical to the live repo (only gitignored `outputs/` and untracked `*.egg-info`
differ).

**D. Overclaims/dishonesty:** None found. Every headline metric carries a track label or explicit
Track-P context; provisional items (Track R numbers, negative `β_j`, B-01 not-reproduced,
R² = 0.2704 not > 0.6, R-040 unsupported, P-057 untestable, B-12/B-13 direction-confirmed, `beta3`
directional) are flagged with decision IDs; "not yet licensed", "owner acceptance not yet", and
"independent audit pending" are stated plainly.

**E. Review mirror:** PASS. `shasum -a 256` = `a0e265ac…` (both trees); `cmp` byte-identical to the
source of record; `.gitattributes` pins the file `-text`.

**F. Git hygiene:** PASS. Working tree == HEAD; `.gitignore`/`.gitattributes` are sane.

## Findings & resolutions

The auditor's verdict was **PUBLISHABLE WITH MINOR FIXES**; the three minor findings and one
informational note are resolved or dispositioned as follows (all changes are documentation /
provenance only — no analysis output, formula, sample definition, or metric changed):

- **F1 (minor) — incomplete post-release commit enumeration.** `docs/STATUS.md` said only two
  non-analysis commits had landed since the recorded run `569994b`, but the graph shows eight
  (the substantive claim — none changes analysis output — held). **Resolved:** STATUS.md now
  enumerates all eight commits since the recorded run, and records that `run_release.py
  --verify-only` passes 20/20 (`release_ok=True`) at `bae7671`, confirming no science drift.
- **F2 (minor) — ephemeral scratch path in shipped content.** The owner's machine-local
  `/tmp/p1e_legacy/xgb_by_week_legacy.csv` was cited in `docs/CONFLICT_MATRIX.md` (C-07),
  `manifests/conflict_matrix.csv` (C-07), `docs/DECISIONS.md` (D-20260901-11), and
  `scripts/build_conflict_matrix.py`. Not private, but a third party cannot "see" that file.
  **Resolved:** the generator source now describes the evidence as a live legacy re-run whose
  per-week scratch output is held outside the repository — keeping the evidential numbers
  (week-mean 0.821101 / season-mean 0.817496) and the "repo port is bit-for-bit identical" claim —
  and the CSV + markdown were regenerated from it (only the C-07 cell changed, confirming generator
  idempotence). The two hand-authored DECISIONS.md references were updated in place.
- **F3 (minor) — provenance claim not verifiable in a clone.** `evidence/README.md` said the two
  phase-diagram PNGs carry sidecars pinning source-table hash, git commit, and run-manifest match,
  but those JSONs existed only in gitignored `outputs/`. **Resolved:** the two sidecars
  (`problem2_phase_diagram_{P,R}.json`, ≈430 B each, pinning source-table SHA-256 + git `569994b` +
  run-manifest match) are now committed in `evidence/figures/` and the README states they are
  committed.
- **F4 (informational — no change).** Release-evidence hashes (`569994b`, …) belong to the owner's
  monorepo; a public clone's history is the rewritten subtree and `outputs/release_manifest.json` /
  `outputs/release_comparison.json` are gitignored. This is disclosed by design (STATUS "recorded
  release"; `docs/ENVIRONMENT.md` verification recipe) and the comparison is reproducible on any
  machine holding the source bundle via `run_release.py --verify-only`. No action.

## Post-fix verification

After the three fixes, the repository was re-verified end to end at this close-out commit:

- Full **229-test** suite (`pytest -q`): **229 passed** (`release_ok` unaffected).
- Static gates: `ruff format --check` (104 files) / `ruff check` clean, `mypy
  src/dwts_reproduction` clean (39 files), `compileall src scripts` OK.
- Hermetic gate (exact ci.yml 78-test selection, run locally on this tree): **78 passed**, 0 failed.
- Release-drift re-check `run_release.py --verify-only`: **`checked=20 pass=20 fail=0 info=0
  release_ok=True`** — the conflict-matrix text edit (F2) does not move any baseline.
- Public-topology confirmation (belt-and-suspenders): a refreshed **split-out public clone** of this
  commit re-passes the hermetic gate and byte-verifies the review mirror (reported with the release
  notes).

## Residual risks & open items (not audit failures)

- **Owner acceptance and licensing remain the owner's call.** Nothing here is formally accepted, the
  repository ships unlicensed (all-rights-reserved), and the contest-data / paper / review-note
  redistribution terms are the owner's to confirm — the owner's release notes record these open
  decisions. This audit clears the technical/consistency gates, not the owner's sign-off.
- **Cross-platform figure bytes.** The committed `evidence/figures/*.png` record what the recorded
  release produced on macOS/darwin; matplotlib PNG bytes are not guaranteed bit-identical across
  platforms. Recorded *numbers* reproduce within registered tolerances (see `docs/ENVIRONMENT.md`).
- **Track R and negative `β_j` are provisional** pending owner acceptance of the honest-limitations
  framing (D-20260901-02 / D-20260901-24), not pending any further technical work.

## How to reproduce this audit

- Build the public topology: `git subtree split -P repo` → `git init && git fetch <split> &&
  git checkout FETCH_HEAD` (the git root must be the repository itself).
- Run the hermetic gate exactly as `.github/workflows/ci.yml` does (78 tests; see `docs/CI.md`).
- Byte-verify the mirror: `shasum -a 256 review/notes/review_all.md` → `a0e265ac…`.
- Reconcile documents against `docs/STATUS.md`; re-check the release comparison with
  `.venv/bin/python scripts/run_release.py --verify-only`.
