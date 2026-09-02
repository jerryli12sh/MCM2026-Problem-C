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

**P-010 notations table complete (2026-09-01):** the last non-implemented traceability row is now
`implemented` — the shared notations table lives in `docs/METHOD_SPEC.md#shared-notation` (core
symbols `s,t,i,A,J,p,q,kappa,S,tau,i*,PCP,e_hat,Bottom2`, definitions pinned to the code).
All 96 paper traceability rows are now `status=implemented` (verified programmatically).

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
`7485ffa4…f44b`). Decisions: D-20260901-01 (era mapping), D-20260901-04 (numpy-vs-torch gap),
D-20260901-07 (posterior reweighting mode).

**Track R implementation complete (2026-09-01):** `fit_integrated_marginal` fits the review's
integrated marginal likelihood `P(Y|β,u)=∫P(Y|p,J)Dirichlet(p|κq)dp` with a score-function /
self-normalized-importance-weights Monte Carlo gradient (B=1200 fresh Dirichlet draws per choice
set per step, NumpyAdam as Track P, single `tau_like=0.15`, `alpha_floor=0.1`, `era_mode='official'`).
Validity pinned by tests: n=2 quadrature identity (MC logL and gradient agree to ~0.1%), Dirichlet
score vs FD, in-family synthetic recovery (signal β_j>0 recovered, zero-signal control |β_j|<0.2
with u structure found), real-data end-to-end (model_type `integrated_marginal_mc`, official era,
MC rel_se≈0.0045, ESS≈1159). Measured Track R vs Track P (labeled separately): top1 `0.8349` vs
`0.9495`, PCP `0.5342` vs `0.6043`, CI width `3.378` vs `3.117`, `S_bar` `0.6331` vs `0.7785`; the
gap is structural (Track P double-uses the outcome; its metrics are internal/explanatory) and the
marginal-likelihood optimum provisionally has β_j<0 (full-batch-converged; treated as provisional until the final independent audit, D-20260901-02). `scripts/problem1_run.py
--track R` writes 11 `_R`-tagged artifacts + manifest; sensitivity spans top1 0.803–0.844 across
seeds/B. 81 tests pass; ruff/mypy clean. Decision: D-20260901-08 (MC estimator + optimizer choice).

Gate: simplex/numerical/gradient tests pass for both tracks; synthetic recovery succeeds; convergence
and sampling diagnostics meet thresholds; paper-number reproduction, Track P limitations, Track R
differences, leakage-safe held-out evaluation, and in-sample reconstruction are reported separately.

## Phase 3 — Evaluation and uncertainty (`active`)

- Top-1 elimination accuracy, rank-sensitive scores, season-path score, PCP, NLL/Brier where valid.
- XGBoost or simpler observable-feature baseline with season-grouped splits.
- Credible intervals, relative width, crowded-field analysis, calibration and stability checks.

**Phase 3 Problem-1 evaluation extras complete (2026-09-01):** the in-season accuracy
baselines (P-027/P-029) and the uncertainty/structural figures (P-025/P-033/P-035/P-037) are
implemented and tested (`scripts/problem1_extras_run.py` writes 8 saved tables + run manifest;
`scripts/plot_problem1_figures.py` renders 6 charts from those tables with a figure manifest
pinning input-table hashes). The torch in-season line reproduces the paper exactly
(season-mean 0.952092, wins 33/33 seasons, 218 training weeks) and proves the paper's
aggregation is the mean of per-season means (D-20260901-13). The XGBoost line's paper target
0.806554 is **not reproducible** from the current legacy code/data — an exhaustive sweep
(features, kappa 1–30, three seed schemes) tops out at 0.821101 week-mean / 0.817496
season-mean, and the repo port is bit-for-bit identical to a live legacy run (C-07,
D-20260901-11); B-01 is updated to preserve the paper target while reporting the honest legacy
line. The paper's ranking-gap `R^2 > 0.6` claim is not reproducible either: the exact cell-56
port gives `R^2 = 0.2704`, n=421 (D-20260901-12; jitter is plot-only, D-20260901-15). PCP is
reported in both variants (paper uniform-weight formula vs importance-weighted reading,
D-20260901-14), and the S8/S21 uncertainty heatmaps adapt exit-week and p_mean/CI width from the
saved posterior summary (D-20260901-16). Traceability P-025..P-038 are `implemented`. Decisions:
D-20260901-11..16.

Gate (recheck): `scripts/problem1_extras_run.py && scripts/plot_problem1_figures.py` reproduce
every metric and chart from one command; each chart is backed by a saved table and the figure
manifest pins the input hash. 142 tests pass (`pytest -q`), `ruff check`/`mypy` clean.

## Phase 4 — Historical and counterfactual mechanisms (`active`)

- Implement rank, percentage, and bottom-2 judges-save rules as separately tested pure functions.
- Replay posterior draws, quantify override/reversal, and reproduce named controversy case studies.
- Keep historical replay distinct from counterfactual simulation assumptions.

Gate: hand-worked fixtures pass; ties and special seasons have explicit policies; posterior uncertainty
propagates into reported comparisons.

**Problem 2 figure rendering complete (2026-09-01):** the ten never-produced paper figure rows
(P-042/P-043/P-045/P-046/P-049/P-050/P-051/P-053/P-054/P-055) are now rendered by
`scripts/plot_problem2_figures.py` from the saved track-tagged CSVs (figure→data mapping verified in
legacy notebook cells 5/10/13/16/23/27/34/39) into `outputs/figures_{P,R}/` (paper-exact filenames),
25 PNGs per track. Every PNG is recorded in `outputs/problem2_fig_manifest_{P,R}.json` with its
traceability id + sha256; no figure exists without a manifest record. Key rendering decisions are
documented in D-20260901-21 (per-figure source, P-045 deterministic bootstrap seed 42, fixed era
palette, producer attribution, P-054 Solid=Save/Dashed=Direct) and D-20260901-22 (deterministic
Bottom-2/judges-save labels persisted in `problem2_case_rank_traces_*.csv` for the P-055 ring
annotations). `problem2_case_weekly_probs_*.csv` now carries `celebrity_name` so season-27's two
cases stay unambiguous (tested by `test_case_weekly_probs_carries_case_identity`). Traceability
P-042..P-055 all `implemented`; figure acceptance evidenced by the manifests.

**Phase 4 implementation complete (2026-09-01):** the four mechanisms (rank/percentage ×
direct/Bottom-2+save) are implemented as pure, tested rule functions (`problem2/rules.py`) and a
counterfactual trajectory replay over fitted posterior draws (`problem2/replay.py`) with explicit
tie policies (D-20260901-09) and carry-forward alive-set snapshots. Named case studies reproduce
Table 1's `|d|`/`Flip` values and the reference `metrics_b2_save.csv`. The mechanism phase diagram
(paper Fig. 5, P-056/P-057) and the review's axis definition (R-039) are implemented behind the track
label (D-20260901-10) with `scripts/problem2_run.py` writing `problem2_*_{P,R}.csv` tables + a run
manifest and `scripts/plot_phase_diagram.py` rendering `outputs/figures/problem2_phase_diagram_{P,R}.png`
from the saved tables (sidecar pins source-table hash + manifest match). Posterior uncertainty is
propagated as 10–90% CI whiskers and claim-check deltas; the review's Perc+Bottom2-highest claim
(R-040) is checked and honestly reported as *not supported* on the reproduced data (top mechanism by
y, y_review, and x is `rank_bottom2` on both tracks; the paper's `x >= 0.3` fan-influence subset is
empty, so P-057 is reported "not testable"). 136 phase rows, 12 b2 rows, 5-6 claim-check rows per
track. Decisions: D-20260901-09, D-20260901-10. Traceability P-039..P-057 and R-036..R-040 now
`implemented`; baseline B-18 registers the b2/phase tables.

Gate (recheck): hand-worked fixtures pass; ties and special seasons have explicit policies
(D-20260901-09); posterior uncertainty propagates into reported comparisons (CI whiskers + claim-check
deltas).

## Phase 5 — New mechanism (`pending`)

- Implement fan compression, judge amplification, momentum bonus, and bottom-2 save.
- Pre-register parameter grid and fairness/excitement metrics before comparing outcomes.
- Run sensitivity, Pareto, and robustness analyses rather than selecting from one best run.

Gate: recommendation is stable over a documented parameter region, or instability is reported.

## Phase 6 — Mechanism explanation (`active`)

- Build comparable judge/fan z-signals and parallel celebrity/partner pathway models.
- Test age, industry, partner, phase/era, surprise, nonlinear growth, and interaction claims.
- Use uncertainty-aware estimates and avoid causal language unsupported by design.

Gate: coefficient tables include uncertainty, reference groups, sample definitions, and robustness
checks; narrative claims map to table cells.

**Phase 6 mechanism-explanation complete (2026-09-01):** Problem 3 (survival determinants,
P-058..P-071) reproduces the paper's three sub-analyses on the registered `data/data_3.csv`
(392 rows, sha256 `72ca124e3890…`): (1) **demographic divergence** — a faithful port of the legacy
`dwts_pro_celeb_regression.py` OLS pipeline (7/7 R² parity within 1e-4; base + season-FE specs,
incremental R², forward CV) plus the paper's exact Eq. (demo_model) with `Other` as the industry
reference; (2) **professional-partner effects** — leakage-safe H_abil/H_exp, partner-FE model,
trait correlations, per-partner FE (HC1 SEs, D-20260901-17); (3) **surprise/growth dynamics** —
S/G construction at t=W6 (n=173) and t=final (n=105), linear + quadratic fits, claim checks.
`scripts/problem3_run.py` writes 19 saved tables/JSONs + a P3 run manifest;
`scripts/plot_problem3_figures.py` renders 5 figures from the saved tables with a figure manifest
pinning the run-manifest hash. Honest claim status (D-20260901-17): B-11 age
(-0.0301/-0.0329/-0.0359) and B-14 surprise beta1 (0.3419, p<0.001) **reproduced**; B-12 actor
(+0.254 / -1.0221) and B-13 partner r (0.134) are **direction-confirmed only** (paper targets
0.16/-0.87 and 0.23 not within tolerance); beta2 (S²) > 0 confirmed (0.1819, p<0.001, Matthew);
beta3 (S×H_exp) > 0 **directional only** (0.0104, p=0.51) — reported honestly, not overstated.
The legacy `.eq()`/`.le()` pro-history artifact is preserved for R² parity and pinned by test.
Traceability P-058..P-071 (and P-007/P-008/P-015) now `implemented`; baseline B-11..B-14 updated
with honest status. Decisions: D-20260901-17.

Gate (recheck): `scripts/problem3_run.py && scripts/plot_problem3_figures.py` reproduce every
metric and chart from one command; coefficient tables include robust SEs, the `Other` industry
reference group, sample definitions, and claim-check booleans; non-reproduced values are reported
with honest status in D-20260901-17. 155 tests pass (`pytest -q`), `ruff check`/`mypy` clean.

## Phase 7 — Release reproduction (`active`)

- Produce final figures, tables, report, environment lock, data/metric dictionaries, and run manifest.
- Produce a paper-faithful release, a review-corrected release, and a paper-vs-review comparison.
- Execute from a clean checkout and compare all controlled outputs with registered tolerances.

Gate: one documented command recreates the release; another runs the full test suite; all deviations
from the review or legacy results appear in `docs/DECISIONS.md`.

**Release-comparison framework complete (2026-09-01):** `scripts/run_release.py` (19-stage driver:
problem1 P/R + extras, problem2 P/R, problem3, problem4, sensitivity, all plot scripts, baseline/
traceability/conflict-matrix builders) writes `outputs/release_manifest.json` (per-stage timings,
exit codes, stdout/stderr tails, git commit, python/platform) and `outputs/release_comparison.json`.
The comparison checks all 20 registered baseline rows (B-01..B-20) against the produced artifacts
via `src/dwts_reproduction/release/compare.py` (13→18 hermetic tests). Two false failures were
resolved (D-20260901-23): B-16 now scopes to the registered R-001..R-019 preprocessing targets, and
B-17 counts figures across all three manifest schema variants (`figures`/`outputs`/`files`);
review traceability is now 40/40 `implemented` (R_STATUS mapping). `--verify-only` reports
**20/20 PASS, release_ok=True**. A hostile self-review then hardened B-08 from a structural
(names+count only) check to a numeric contract that asserts each registered |d|/Flip value within
abs 1e-2, and reframed the Track R β_j<0 explanation as provisional awaiting the final independent
audit (D-20260901-24); 22 release-comparison tests pass.

**Phase 7 release reproduction complete (2026-09-01):** the full end-to-end default run
(`scripts/run_release.py`, 1375.8 s ≈ 23 min) completed with **all 19 stages exit 0** and
**20/20 PASS, release_ok=True** (`outputs/release_manifest.json` pins git 569994b, python 3.13.3,
per-stage durations, stdout/stderr tails; `outputs/release_comparison.json` pins the comparison
sha256). Every registered baseline row B-01..B-20 is verified against the freshly produced
artifacts, including the strengthened B-08 numeric contract. Full test suite: **229 tests pass**;
ruff format/check and mypy clean. Honest limitations are registered (B-01 XGBoost not
reproducible, ranking-gap R² claim not reproducible, R-040 not supported, P-057 not testable,
B-12/B-13 direction-confirmed only, Track R performance/β_j provisional, fan votes never ground
truth). Acceptance packet: `docs/PHASE7_ACCEPTANCE.md`. Remaining: owner acceptance and the final
independent audit.
