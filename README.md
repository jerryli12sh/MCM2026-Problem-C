# DWTS MCM 2026 Problem C — reproduction and review-corrected re-analysis

This repository is a **clean, testable, dependency-pinned re-implementation** of a submitted MCM 2026
Problem C paper about *Dancing with the Stars* (DWTS) contestant elimination. It does two things,
always labeled separately:

- **Track P** — a *faithful* reproduction of the submitted paper: fit a latent fan-support model
  `q` from elimination outcomes, then condition each week's viewer-vote share `p` on the observed
  elimination. Track P metrics are **internal/explanatory** because the same outcome is used twice
  (to fit `q` *and* to condition `p`). This reproduces the paper on its own terms.
- **Track R** — a *review-corrected* re-analysis that fixes that double-use by integrating the weekly
  vote share out of the elimination likelihood,
  `P(Y|β,u) = ∫ P(Y|p,J) · Dirichlet(p | κ·q) dp`, so each outcome is used once.

When the paper and its review disagree, **both** are implemented behind explicit configuration, each
is tested, and the results are compared — never silently merged. No metric is reported without a
track label.

> This is a **reproduction/verification** artifact, not the contest paper and not a claim that the
> paper's numbers are wrong. Everything here was rebuilt from the submitted paper
> (`2107542.tex`), the contest data, the legacy implementation, and the review note — see
> [Data & provenance](#data--provenance). Deviations are registered openly in
> [`docs/DECISIONS.md`](docs/DECISIONS.md).

---

## Status at a glance

| State | Value |
|---|---|
| Release evidence | 19/19 pipeline stages exit 0 · 20/20 registered baselines **PASS** (`release_ok=True`) |
| Full test suite | **229 passed** · `ruff` clean · `mypy` clean |
| Recorded release | git `569994b`, Python 3.13.3, 1375.8 s (≈23 min) |
| Track P headline | top-1 **0.9495** · PCP 0.6043 · S-bar 0.7785 · CI width 3.117 *(internal/explanatory)* |
| Track R headline | top-1 **0.8349** · PCP 0.5342 · S-bar 0.6331 · CI width 3.378 *(single-use; provisional)* |
| Owner acceptance | **not yet** — automated gates pass; final independent audit and owner sign-off pending |

The authoritative reconciliation of every number and document in this repository is
[`docs/STATUS.md`](docs/STATUS.md). **Start there** if you want the ground truth.

---

## Headline findings (honest)

1. **The paper's own (Track P) pipeline reproduces.** The torch reference's in-season accuracy
   (0.952092) and the review-rebuild targets (top-1 0.9495, PCP 0.6043, S-bar 0.7785) reproduce to
   registered tolerances.
2. **The paper's XGBoost baseline target (0.806554) is *not* reproducible** from the current legacy
   code/data — an exhaustive sweep tops out at week 0.8211 / season 0.8175, and the repo port is
   bit-for-bit identical to a live legacy run (B-01, D-20260901-11 / C-07).
3. **The paper's ranking-gap claim (`R² > 0.6`) is *not* reproducible.** The exact cell-56 port gives
   `R² = 0.2704`, n=421 (D-20260901-12).
4. **The review's favored "Perc + Bottom-2" mechanism claim (R-040) is *not supported*** on the
   reproduced data — the top mechanism by every reported quantity is `rank_bottom2` on both tracks
   (D-20260901-10). Paper cell P-057 is not testable (its subset is empty).
5. **Track R is structurally more honest but not stronger on the headline metrics** — single-use
   elimination likelihood yields lower raw fit (top-1 0.835 vs 0.950). That gap is the expected cost
   of removing double-use, and the Track R optimum's negative judge-weight `β_j` is **provisional**
   pending the final independent audit.
6. **Never read the fan-vote estimates as ground truth.** They are posterior estimates constrained by
   observed eliminations (D-20260901-06).

Full limitations and their decisions: [`docs/STATUS.md`](docs/STATUS.md) § limitations and
[`docs/DECISIONS.md`](docs/DECISIONS.md).

---

## What the problems map to

The contest analysis and this codebase are organized around four analysis problems plus sensitivity:

| Code module | Analysis | Paper output reproduced |
|---|---|---|
| `problem1_run.py` (+`extras`) | Weekly viewer-vote share inference, elimination prediction, uncertainty, cumulative consistency | top-1 / PCP / S-bar / CI-width; in-season torch & XGBoost baselines; ranking-gap |
| `problem2_run.py` | Merge rule & Bottom-2+judge-save mechanisms, rank/percentage eras, named case studies | Table 1 `\|d\|`/Flip; phase diagram; claim checks |
| `problem3_run.py` | Survival determinants — age, industry, partner, surprise/growth | age/actor/partner/surprise coefficients (honest status) |
| `problem4_run.py` | Counterfactual new-rule simulations (V1 legacy params, V2 design) | V1 parity cells, V2 claims |
| `sensitivity_run.py` | κ grid, λ, judge-weight, leave-one-season | stability surfaces (A1–A4) |

---

## Repository map

```text
docs/                      specification, decisions, acceptance, and evidence documents
  STATUS.md                ← authoritative state + reconciliation hub (read first)
  ENVIRONMENT.md           install & snapshot-verification guide
  METHOD_SPEC.md           method specification incl. shared-notation table (P-010)
  DECISIONS.md             24 registered decisions (D-20260901-01..24)
  BASELINE_PAPER_OUTPUTS.md, CONFLICT_MATRIX.md, DATA_DICTIONARY.md, …
configs/paths.yaml         read-only source-root declaration (override: $DWTS_SOURCE_ROOT)
src/dwts_reproduction/     typed, reusable Python package (config, model, rules, release)
scripts/                   thin CLI entry points (one per run/plot/build task)
tests/                     229 unit/invariant/integration tests
manifests/                 machine-readable ground truth (baseline 20, paper 96,
                           review 40, conflicts 7, legacy inventory 174, input sha256)
review/notes/review_all.md byte-for-byte mirror of the external review note (Track R spec)
outputs/                   generated tables/figures/manifests (gitignored; see Artifacts)
prompts/, .claude/, CLAUDE.md   maintainer agent-workflow notes (not part of the science)
Makefile                   install / format / lint / type / test / verify-inputs / smoke
pyproject.toml             package + tool config (ruff, mypy, pytest, extras)
```

---

## Install, verify, reproduce

Requires **Python 3.13** (3.11–3.13 supported; 3.13 is the gated/verified interpreter — see
[`docs/ENVIRONMENT.md`](docs/ENVIRONMENT.md) for the exact reasoning and the pinned snapshot).

```bash
# 1. Install runtime + analysis + dev dependencies into a local venv
python3 -m venv .venv
.venv/bin/pip install -e ".[analysis,dev]"     # or: make install
```

Fast correctness gates (no heavy compute):

```bash
.venv/bin/python -m ruff format --check .
.venv/bin/python -m ruff check .
.venv/bin/python -m mypy src/dwts_reproduction
.venv/bin/python -m pytest -q                       # 229 tests
.venv/bin/python scripts/hash_inputs.py --validate  # 174 input hashes
.venv/bin/python scripts/smoke_test.py
.venv/bin/python scripts/check_scope.py
# or the whole gate at once:
make phase0-accept
```

Reproduce the release evidence:

```bash
.venv/bin/python scripts/run_release.py --verify-only   # fast: re-check 20/20 baselines (~min)
.venv/bin/python scripts/run_release.py                 # full: all 19 stages (~23 min)
```

The full run writes `outputs/release_manifest.json` (per-stage timing/exit/commit) and
`outputs/release_comparison.json` (20/20 PASS). Verify your environment matches the recorded
snapshot with the recipe in [`docs/ENVIRONMENT.md`](docs/ENVIRONMENT.md) § Verify.

> **Third-party verifiers need the read-only source bundle.** The raw data, the submitted paper, the
> legacy implementation, and the review note live *outside* this repository and are not redistributed
> here. Point the code at a local copy of that bundle (a directory whose `data/`, `paper_Latex/`,
> `review/`, `src/` sit next to each other) by setting `DWTS_SOURCE_ROOT` to it, or by placing the
> bundle as this repository's parent (the default `source_root: ..`).

---

## Data & provenance

- **Raw data:** `2026_MCM_Problem_C_Data.csv` — the official contest dataset for MCM 2026 Problem C
  (34 DWTS seasons). 174 inputs/evidence files are **hashed and read-only**; the sha manifest is
  validated by `scripts/hash_inputs.py --validate`.
- **Paper:** the submitted paper source `2107542.tex` (Team control number 2623768) defines **Track P**.
- **Review:** the consolidated review note defines **Track R**.
- **Legacy implementation** (`src/` and review legacy code) is evidence for *how* the paper was
  computed; it may clarify but never silently override the written paper.
- All of these are outside this repository and never modified by this codebase. Paths resolve through
  `configs/paths.yaml` / `$DWTS_SOURCE_ROOT` — no machine-specific absolute path appears in code.
- This repo carries a **verbatim byte-for-byte mirror** of the review note at
  [`review/notes/review_all.md`](review/notes/review_all.md) (SHA-256
  `a0e265acb9c36bb5d3acd5bde0b3ec0a6798b2e93e75e5bc5996e950e3070ea5`, pinned `-text` in
  `.gitattributes` so git never alters it). The mirror exists so a standalone clone documents the
  Track R spec; the source of record lives outside the repo.

## Artifact policy

- `outputs/` is **gitignored and regenerable**: every table, figure, and run manifest is produced by
  a documented script, and every figure is registered in a figure manifest that pins its input-table
  hash. Reproducibility evidence is the *recipe + hashes*, not the bytes.
- Committed evidence lives in `docs/` (acceptance packets, baseline/traceability/conflict tables) and
  `manifests/` (machine-readable rows). The review mirror is the one committed *byte artifact*.
- Nothing in `outputs/` is evidence unless its run manifest records command, seeds, input hashes, git
  commit, and environment (see `docs/RUN_MANIFEST.md`).

---

## Documentation index

| Document | What it is |
|---|---|
| [`docs/STATUS.md`](docs/STATUS.md) | **Authoritative status + reconciliation hub** |
| [`docs/ENVIRONMENT.md`](docs/ENVIRONMENT.md) | Environment, install, snapshot verification |
| [`docs/RUNBOOK.md`](docs/RUNBOOK.md) | Owner operating procedure (phase loop, stop rules) |
| [`docs/METHOD_SPEC.md`](docs/METHOD_SPEC.md) | Method spec + shared-notation table |
| [`docs/DECISIONS.md`](docs/DECISIONS.md) | Every ambiguity resolved (D-20260901-01..24) |
| [`docs/PHASE0_ACCEPTANCE.md`](docs/PHASE0_ACCEPTANCE.md) · [`docs/PHASE7_ACCEPTANCE.md`](docs/PHASE7_ACCEPTANCE.md) | Acceptance packets |
| [`docs/BASELINE_PAPER_OUTPUTS.md`](docs/BASELINE_PAPER_OUTPUTS.md) · [`docs/CONFLICT_MATRIX.md`](docs/CONFLICT_MATRIX.md) · [`docs/DATA_DICTIONARY.md`](docs/DATA_DICTIONARY.md) | Baselines, conflicts, data dictionary |
| [`docs/TRACEABILITY_PAPER.md`](docs/TRACEABILITY_PAPER.md) · [`docs/TRACEABILITY_REVIEW.md`](docs/TRACEABILITY_REVIEW.md) | Every paper (96) / review (40) requirement → implementation |
| `PLAN.md` | Phase-by-phase implementation plan with evidence notes |

---

## Honest limitations (short form)

- XGBoost paper target and the ranking-gap `R² > 0.6` claim are **not reproducible** from the
  available code/data (B-01, D-20260901-11/12).
- R-040 (review's Perc+Bottom-2 claim) is **not supported**; P-057 is **not testable**
  (D-20260901-10).
- Problem-3 actor/partner coefficients are **direction-confirmed only**, and `beta3` is directional
  only (D-20260901-17).
- Track R headline numbers and its negative `β_j` are **provisional** until the final independent
  audit (D-20260901-02/24).
- Fan votes are never ground truth (D-20260901-06).

## Citation & license

- **Status:** this repository is *not yet licensed* and has *no public citation*. Ownership of the
  contest paper, data, and analysis is with the author (Team 2623768) and the contest organizers.
  The licensing decision is the owner's to make; see the repository owner's release notes before
  redistributing. If you use this reproduction, cite the repository and the submitted paper rather
  than this artifact alone.
- The submitted-paper analysis is reproduced here under academic/verification intent; the contest
  dataset's redistribution terms are the owner's to confirm.
