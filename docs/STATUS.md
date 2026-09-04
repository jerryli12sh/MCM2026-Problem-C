# Repository status and result reconciliation

This is the authoritative summary of what the repository implements and what the recorded analysis
found. Track P and Track R are always labeled separately.

## Scope

- **Track P (paper-faithful):** reproduces the submitted two-stage method. It first fits pooled
  support from elimination outcomes, then conditions weekly fan shares on those outcomes. Its
  reconstruction metrics are internal/explanatory because the outcome is used twice.
- **Track R (review-corrected):** implements the integrated marginal likelihood proposed in
  `review/notes/review_all.md`, using each elimination once.
- **Problems 2–4:** replay historical rules, analyze contestant/partner pathways, and simulate a new
  mechanism.
- **Sensitivity:** varies concentration, temperature, regularization, judge representation, and
  held-out seasons.

Legacy scripts, the submitted paper, and raw data are external read-only evidence. All material
deviations are registered in [`DECISIONS.md`](DECISIONS.md).

## Recorded release

| Item | Value |
|---|---|
| Release command | `.venv/bin/python scripts/run_release.py` |
| Pipeline | 19/19 stages completed |
| Baseline comparison | 20/20 passed; `release_ok=True` |
| Runtime | 1375.8 seconds (about 23 minutes) |
| Recorded commit | `569994b` in the original workspace history |
| Environment | CPython 3.13.3, macOS arm64 |
| Full data-bound suite at release | 229 tests passed |
| Current cleaned-tree suite | 226 collected: 199 passed, 27 output-dependent tests skipped |

The publication cleanup changed documentation, repository topology checks, and ignored local
artifacts only. It did not change formulas, samples, model code, or numerical evidence. Current
verification is recorded in [`VERIFICATION.md`](VERIFICATION.md).

## Headline comparison

| Quantity | Track P | Track R |
|---|---:|---:|
| Top-1 elimination reconstruction | 0.9495 | 0.8349 |
| Mean weighted PCP | 0.6043 | 0.5342 |
| Mean relative credible-interval width | 3.117 | 3.378 |
| Season-path consistency, `S-bar` | 0.7785 | 0.6331 |

Track P's stronger historical reconstruction is expected after conditioning twice on the outcome;
it is not a held-out predictive advantage. Track R is structurally more defensible but its
negative fitted judge coefficient and lower headline values should be treated as model findings
with sensitivity, not universal truths.

## Registered evidence

| Artifact | Count | Meaning |
|---|---:|---|
| `manifests/traceability_paper.csv` | 96 rows | paper requirement to implementation/test mapping |
| `manifests/traceability_review.csv` | 40 rows | review requirement to implementation/test mapping |
| `manifests/baseline.csv` | 20 rows | expected outputs and tolerances |
| `manifests/conflict_matrix.csv` | 7 rows | paper/review/legacy conflicts |
| `manifests/legacy_inventory.csv` | 174 rows | external source inventory |
| `manifests/input_manifest.sha256` | 174 rows | external input fingerprints |
| `docs/DECISIONS.md` | 24 decisions | resolved modeling and reproduction ambiguities |
| `evidence/figures/` | 10 PNGs | representative release figures |

## Findings that did not reproduce cleanly

1. The paper's XGBoost target `0.806554` was not reproducible from the available legacy code/data.
   The faithful live legacy line was 0.821101 by week and 0.817496 by season
   (D-20260901-11).
2. The paper's ranking-gap claim `R² > 0.6` reproduced as `R² = 0.2704`, `n = 421`
   (D-20260901-12).
3. Review claim R-040—percentage plus Bottom-2 having the best reported mechanism position—was not
   supported. `rank_bottom2` led the reproduced quantities on both tracks (D-20260901-10).
4. Paper claim P-057 could not be tested because its `x >= 0.3` subset was empty
   (D-20260901-10).
5. Actor and partner targets were direction-confirmed only; the surprise-by-experience interaction
   was directional but not significant (`0.0104`, `p = 0.51`) (D-20260901-17).
6. Fan shares remain latent posterior estimates constrained by eliminations, never observed vote
   totals (D-20260901-06).

## The review-note mirror

`review/notes/review_all.md` is the repository's conceptual core and a byte-for-byte mirror of the
external source note:

- SHA-256:
  `a0e265acb9c36bb5d3acd5bde0b3ec0a6798b2e93e75e5bc5996e950e3070ea5`;
- `.gitattributes` marks it `-text`, preventing line-ending normalization;
- it defines the corrected statistical direction, rule comparison, explanatory models, and new-rule
  design that the code makes executable.

## Publication status

The technical tree is ready to publish as a standalone repository after running the documented
source-free gate. The official dataset, submitted paper source, and legacy workspace are deliberately
excluded. No open-source license is granted; any broader reuse or redistribution remains an owner
decision.

## Where to go next

| Need | Document |
|---|---|
| Understand the whole repository in Chinese | [`REPOSITORY_GUIDE.zh-CN.md`](REPOSITORY_GUIDE.zh-CN.md) |
| Follow the modeling argument | [`../review/notes/review_all.md`](../review/notes/review_all.md) |
| Check notation and algorithms | [`METHOD_SPEC.md`](METHOD_SPEC.md) |
| See why choices were made | [`DECISIONS.md`](DECISIONS.md) |
| Inspect data/table meanings | [`DATA_DICTIONARY.md`](DATA_DICTIONARY.md) |
| Review the engineering journey | [`DEVELOPMENT.md`](DEVELOPMENT.md) |
| Reproduce the environment | [`ENVIRONMENT.md`](ENVIRONMENT.md) |
| Understand public vs. data-bound CI | [`CI.md`](CI.md) |
