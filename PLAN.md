# Reproduction plan

Status values: `pending`, `active`, `blocked`, `accepted`.

## Phase 0 — Baseline and provenance (`active`)

- Inventory raw inputs, legacy scripts/notebooks, review requirements, and existing outputs.
- Build separate paper and review requirement inventories and a conflict matrix.
- Hash immutable inputs and create a data dictionary.
- Establish environment, package skeleton, lint/type/test commands, and run-manifest schema.
- Create a legacy baseline table: metric/figure, legacy producer, expected value, tolerance, status.

Gate: a fresh environment can validate input hashes and run a smoke test without modifying legacy
files. Every material paper and review requirement maps to an implementation module and an acceptance
test; every conflict is classified as Track P versus Track R rather than silently resolved.

**Phase 0 implementation complete (2026-09-01):** `make phase0-accept` passes (24 tests; 174 hashed
inputs; 96 paper / 35 review traceability rows; 6 conflicts; 17 baseline rows). Evidence in
`docs/PHASE0_ACCEPTANCE.md`. Awaiting independent audit + owner acceptance; do not enter Phase 1.

## Phase 1 — Canonical preprocessing (`pending`)

- Parse result states, season lengths, active horizons, structural zeroes, and missing values.
- Build contestant roster, contestant-week table, long judge table, and elimination-event table.
- Explicitly distinguish regular single elimination, multiple elimination, withdrawal, and final end.

Gate: schema and invariant tests pass; row counts and key summaries reconcile against both raw data
and the reviewed preprocessing artifacts. All differences are documented.

## Phase 2 — Latent fan-support models (`active`)

- Implement alive-set-normalized judge signals and feature construction.
- Implement pooled support center `q = softmax(X beta + u)`.
- Track P: reproduce the paper's penalized softmin fit for `q`, followed by Dirichlet-softmin weekly
  posterior updating via importance sampling.
- Track R: fit the review's integrated model by marginalizing `p ~ Dirichlet(kappa q)` in the
  elimination likelihood, avoiding double use of the same outcome.
- Produce posterior means, intervals, effective sample size diagnostics, and deterministic samples.

**Track P implementation complete (2026-09-01):** pooled softmin fit (hand-written numpy + Adam,
float32, matching the torch reference) + Dirichlet importance-sampling posteriors reproduce the
review-rebuild targets — `top1 = 0.9495412844036697` (bit-for-bit), `mean_pcp_weighted = 0.6043173`
(rel 1.4e-5), `mean_ess_ratio = 0.9625174` (rel 3e-6), `mean_ci_rel_width = 3.1171359` (rel 6.4e-5),
`S_bar = 0.7785` (abs 0.0015 vs 0.78); panel 4199 rows, 218 train weeks, 292 elimination events.
`scripts/problem1_run.py` writes 11 track-tagged artifacts + a run manifest (input sha
`7485ffa4…f44b`). 73 tests pass; ruff/mypy clean. Decisions: D-20260901-01 (era mapping), D-20260901-04
(numpy-vs-torch gap), D-20260901-07 (posterior reweighting mode). Track R still pending.

Gate: simplex/numerical/gradient tests pass for both tracks; synthetic recovery succeeds; convergence
and sampling diagnostics meet thresholds; paper-number reproduction, Track P limitations, Track R
differences, leakage-safe held-out evaluation, and in-sample reconstruction are reported separately.

## Phase 3 — Evaluation and uncertainty (`pending`)

- Top-1 elimination accuracy, rank-sensitive scores, season-path score, PCP, NLL/Brier where valid.
- XGBoost or simpler observable-feature baseline with season-grouped splits.
- Credible intervals, relative width, crowded-field analysis, calibration and stability checks.

Gate: metrics are reproducible from one command and every chart is backed by a saved table.

## Phase 4 — Historical and counterfactual mechanisms (`pending`)

- Implement rank, percentage, and bottom-2 judges-save rules as separately tested pure functions.
- Replay posterior draws, quantify override/reversal, and reproduce named controversy case studies.
- Keep historical replay distinct from counterfactual simulation assumptions.

Gate: hand-worked fixtures pass; ties and special seasons have explicit policies; posterior uncertainty
propagates into reported comparisons.

## Phase 5 — New mechanism (`pending`)

- Implement fan compression, judge amplification, momentum bonus, and bottom-2 save.
- Pre-register parameter grid and fairness/excitement metrics before comparing outcomes.
- Run sensitivity, Pareto, and robustness analyses rather than selecting from one best run.

Gate: recommendation is stable over a documented parameter region, or instability is reported.

## Phase 6 — Mechanism explanation (`pending`)

- Build comparable judge/fan z-signals and parallel celebrity/partner pathway models.
- Test age, industry, partner, phase/era, surprise, nonlinear growth, and interaction claims.
- Use uncertainty-aware estimates and avoid causal language unsupported by design.

Gate: coefficient tables include uncertainty, reference groups, sample definitions, and robustness
checks; narrative claims map to table cells.

## Phase 7 — Release reproduction (`pending`)

- Produce final figures, tables, report, environment lock, data/metric dictionaries, and run manifest.
- Produce a paper-faithful release, a review-corrected release, and a paper-vs-review comparison.
- Execute from a clean checkout and compare all controlled outputs with registered tolerances.

Gate: one documented command recreates the release; another runs the full test suite; all deviations
from the review or legacy results appear in `docs/DECISIONS.md`.
