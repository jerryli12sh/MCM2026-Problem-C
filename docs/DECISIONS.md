# Decision log

Record decisions as `D-YYYYMMDD-NN` with status, context, options, choice, rationale, consequences,
review references, legacy references, and user approval when required.

## Open decisions

- Historical rule/season mapping, especially judge-save seasons and exceptional events.
- Feature set and identifiability constraints for contestant effects `u`.
- Definition of fan-influence/excitement and technical-alignment metrics used for the
  *Problem 4* recommendation (the phase-diagram axes are resolved in D-20260901-10).
- Tolerances for numerical equivalence versus deliberate correction of legacy results
  (resolved for Problem 1 metrics in D-20260901-04; other phases still open).

## Phase 0 — conflict decisions

These map one-to-one to `docs/CONFLICT_MATRIX.md`. Status is `open` until the owning phase
inspects the relevant legacy code or the owner approves a tolerance.

### D-20260901-01 — Era mapping direction (established)

- **Status:** established (2026-09-01).
- **Context:** The official mapping is seasons 1–2 rank, 3–27 percent, 28–34 rank (Bottom-2
  ~28+). Legacy `../src/model.py` uses `season>=28 -> "percent"` — the opposite.
- **Options:** (a) official only; (b) legacy only; (c) both behind an explicit `era_mode`.
- **Choice:** (c) — expose `era_mode='legacy'|'official'`. Track P registers `legacy`; Track R
  registers `official`. The legacy direction was confirmed line-by-line in
  `../src/model.py` and `../src/model_new.py`, and the Track P pipeline reproduces the review
  rebuild's outputs under it (top-1 accuracy bit-for-bit, see D-20260901-04).
- **Rationale:** reproducing the paper numbers requires the legacy mapping; the review requires
  the official one. Both must be reproducible and comparable; never merged silently.
- **Consequences:** `era_mode` config; Track P metrics are only comparable to the paper/reference
  under the legacy mapping. Track R must not reuse Track P's era-dependent `j_metric` without
  recomputing the panel (implemented behind `era_mode`).
- **Refs:** `../review/00_codex_preprocessing_refactor_prompt.md`,
  `../review/01_codex_problem1_refactor_prompt.md`, `../src/model.py`, `../src/model_new.py`.

### D-20260901-02 — Double-use of the elimination outcome

- **Status:** open (resolved by design: Track P vs Track R).
- **Context:** Track P fits `q` from eliminations then conditions weekly `p` on the same
  elimination; Track R uses the integrated marginal `P(Y|β,u)=∫P(Y|p,J)Dirichlet(p|κq)dp`.
- **Choice:** implement both tracks; label Track P metrics in-sample/explanatory.
- **Rationale:** paper-faithful reproduction must keep the two-stage procedure; the review's
  correction is the Track R deliverable. Do not merge them.
- **Consequences:** Track P accuracy claims are internal validity, never prediction.
- **Refs:** `../paper_Latex/2107542.tex` (Problem 1), `../review/notes/review_all.md`.

### D-20260901-03 — Single vs dual temperature

- **Status:** open.
- **Context:** The paper writes one `τ`; legacy uses `tau_train=0.05` (fit) and `tau_like=0.15`
  (posterior reweight).
- **Choice:** preserve both as a named variant; Track R uses a single explicit temperature.
- **Rationale:** uncertainty magnitude depends on which is used; the paper/legacy discrepancy must
  not be silently resolved.
- **Consequences:** CI widths differ across settings; must be reported side by side.
- **Refs:** `../data/超参数.md`, `../review/problem1_rebuild/outputs/problem1_summary.json`.

### D-20260901-04 — Top-1 accuracy gap (measured: numpy rebuild reproduces the reference)

- **Status:** resolved (2026-09-01, within registered tolerance).
- **Context:** paper reports `0.952092`; review rebuild reports `0.949541` (~0.003 absolute gap).
- **Choice:** the numpy rebuild targets the review-rebuild numbers, not the paper's. Measured
  numpy-vs-torch gap on the full Track P run (`outputs/problem1_summary_P.json`):
  `overall_top1_accuracy = 0.9495412844036697` (**bit-for-bit equal** to the reference);
  `mean_pcp_weighted = 0.6043173` (rel 1.4e-5); `mean_ess_ratio = 0.9625174` (rel 3e-6);
  `mean_ci_rel_width = 3.1171359` (rel 6.4e-5). Pooled-fit final loss `1.2615689` vs reference
  `1.2615778` (rel ~7e-6), attributable to float32 accumulation order in the hand-written numpy
  Adam vs torch. All within the registered rel 1e-3 / abs 0.02 tolerances.
- **Rationale:** the ~0.003 gap is a paper-vs-rebuild discrepancy (the rebuild itself does not
  reproduce the paper's `0.952092`), not a numpy-vs-torch implementation gap. The numpy rebuild is
  faithful to the reference rebuild within 1e-5.
- **Consequences:** Track P reproduction target stays on the review-rebuild numbers. The
  paper-vs-rebuild `0.003` remains an open paper discrepancy to document in Phase 7.
- **Refs:** `../paper_Latex/2107542.tex` (Problem 1),
  `../review/problem1_rebuild/outputs/problem1_summary.json`, `outputs/problem1_summary_P.json`,
  `outputs/problem1_fit_meta_P.json`.

### D-20260901-05 — η specification drift

- **Status:** open.
- **Context:** paper `η = xᵀβ + u` with `x = {J, age}`; legacy adds intercept `b` and an
  `era_is_percent` feature.
- **Choice:** preserve the paper's stated features and the legacy feature set as a named
  paper-implementation variant.
- **Rationale:** affects identifiability and coefficient interpretation; both must be reproducible.
- **Consequences:** Track P may require the legacy feature set to match paper numbers.
- **Refs:** `../paper_Latex/2107542.tex` (eq:latent_score),
  `../review/problem1_rebuild/outputs/problem1_fit_metadata.json`.

### D-20260901-07 — Posterior reweighting mode (rebuild vs legacy)

- **Status:** established (2026-09-01).
- **Context:** the reference rebuild restricts softmin reweighting to single-elimination,
  non-final weeks with a full finite judge signal (``has_posterior_mode='rebuild'``) for the
  posterior-summary targets, but the legacy pipeline that produced the paper's ``S_bar = 0.78``
  reweighted every single-elimination event including finales (``'legacy'``). The two families of
  targets require different modes.
- **Choice:** expose ``has_posterior_mode='rebuild'|'legacy'``. ``posterior_summary`` / reference
  summary targets (top-1, PCP, ESS, CI width) use ``'rebuild'`` (default); the cumulative
  consistency event tables reproduce ``S_bar`` under ``'legacy'``.
- **Rationale:** neither mode is a bug; they answer different questions. Mixing them would make one
  family of reference numbers unreproducible.
- **Consequences:** measured ``S_bar = 0.7785311681432541`` under legacy mode (within the
  registered abs 0.02 of ``0.78``); top-1/PCP/ESS/CI under rebuild mode (see D-20260901-04). The
  choice is per-call, not a global config, so both families stay reproducible from one fit.
- **Refs:** `../review/problem1_rebuild/problem1_fan_support.py`,
  `../src/model_new.py`, `../src/build_event_table.py`,
  `../review/problem1_rebuild/outputs/problem1_summary.json`.

### D-20260901-06 — Fan votes are not ground truth

- **Status:** closed (labeling rule).
- **Context:** inferred `p̂` must never be described as the true official fan vote.
- **Choice:** label as posterior estimates constrained by observed outcomes, in all tracks.
- **Rationale:** fan vote totals are unobserved; only eliminations constrain the posterior.
- **Consequences:** no claim of ground-truth fan votes anywhere in the repository.
- **Refs:** `../review/problem1_rebuild/outputs/problem1_readme.md`, `CLAUDE.md`.

### D-20260901-08 — Approximation and optimization method for the integrated marginal likelihood

- **Status:** established (2026-09-01).
- **Context:** Track R requires `P(Y|β,u)=∫P(Y|p,J)Dirichlet(p|κq)dp`, which has no closed form.
  The review specifies the formulation but not the estimator. The open decision item
  "Approximation and optimization method for the integrated marginal likelihood" is resolved here.
- **Options:** (a) Gauss–Hermite / Laplace quadratic approximations; (b) nested Monte Carlo with a
  reparameterized (pathwise) gradient; (c) score-function (REINFORCE) gradient with self-normalized
  importance weights; (d) quadrature over a grid of `p`.
- **Choice:** (c) — `B` fresh Dirichlet draws per choice set per minibatch step, softmin
  log-likelihood as the unnormalized weight, Dirichlet score `d/dα log Dir(p|α)` as the influence,
  chained through `α=κq` by the softmax Jacobian. Fit with the same hand-written NumpyAdam as Track
  P (lr 0.02, betas (0.9, 0.999), eps 1e-8). Single explicit softmin temperature
  `tau_like=0.15` (D-20260901-03), `era_mode='official'` (D-20260901-01), `B=1200`,
  `alpha_floor=0.1` (below ~0.1 the gamma sampler underflows to exact zeros whose `log` corrupts
  the score; 0.1 also bounds the score variance `trigamma(α_i) ≈ 1/α_i`).
- **Rationale:** the score estimator is unbiased and needs only unnormalized softmin likelihoods
  (the Dirichlet normalizer cancels in the normalized weights). Three estimator bugs were found and
  fixed during calibration: self-normalized weights must be renormalized to sum to 1
  (`exp(log f − log L)` is `f/mean(f)`, off by a factor `B`), the alpha clip must floor at 0.1, and
  the fit accumulates the negative of `d log L/dη` (Track P's `−1/τ` chain factor already encodes
  the NLL sign, Track R's positive log-likelihood gradient does not). Validity is pinned by tests:
  an exact n=2 quadrature identity (sigmoid softmin × Beta) agrees with the MC log-likelihood and
  gradient to ~0.1%, an FD test of the Dirichlet score, and an in-family synthetic recovery (signal
  `β_j>0` recovered, zero-signal control stays near 0 with its `u` structure found).
- **Consequences:** measured Track R (`outputs/problem1_summary_R.json`) vs Track P: overall
  top-1 `0.8349` vs `0.9495`, mean PCP `0.5342` vs `0.6043`, mean CI width `3.378` vs `3.117`,
  `S_bar` `0.6331` vs `0.7785`. The top-1 gap is structural: Track P reconditions the weekly
  posterior on the *same* observed elimination used to fit `q` (internal/explanatory), Track R uses
  each outcome once. The marginal-likelihood optimum genuinely has `β_j < 0` (full-batch
  convergence and the gradient at the fitted point both confirm it), so Track R's different
  coefficients are not a bug. MC error at the fitted point: `mc_se_relative ≈ 0.0045`, mean
  importance ESS ≈ 1159, and sensitivity across seeds/B spans top-1 `0.803–0.844`. Track P and
  Track R metrics must always be reported with their track label (D-20260901-02).
- **Refs:** `../review/notes/review_all.md` (integrated marginal-likelihood proposal),
  `src/dwts_reproduction/problem1/track_r.py`,
  `tests/test_problem1_track_r.py`, `outputs/problem1_summary_R.json`.

### D-20260901-09 — Tie handling and b2-save semantics for Problem 2

- **Status:** established (2026-09-01).
- **Context:** the paper's rank / percentage / Bottom-2 + judges'-save formulas
  (`../paper_Latex/2107542.tex`, Problem 2) use ``argmax``/``argmin`` with no stated tie rule.
  The legacy notebook ``src/2_rank_vs_pct_cross_season.ipynb`` (cells 20/29/34) and the reference
  b2-save metrics producer ``src/b2_save_metrics.py`` (the producer of
  ``../data/metrics_b2_save.csv``) each apply their own deterministic lexsort tie-breaking, and the
  two differ.
- **Options:** (a) first-index ties everywhere; (b) replicate each legacy producer's lexsort
  exactly where a reference output exists; (c) a single global tie policy across all functions.
- **Choice:** (c) rejected; (b) with a named policy per function family. ``simulate_week`` and the
  paper-formula point helpers use the legacy notebook's lexsort ``(name_key, p, j, score)`` with
  ``score`` primary; ``risk_and_bottom2``/``b2_case_metrics`` port ``src/b2_save_metrics.py``
  exactly (primary ``-risk``, then ``judge_pct``, then ``p_draw``, then name). First-index ties on
  the name-sorted ordering are used only where the paper formula is evaluated on its own.
- **Rationale:** the b2 reference CSV and the Table 1 ``|d|``/``Flip`` case table were produced
  under these exact orderings; deviating would make the registered values unreproducible. The three
  policies coincide except on exact ties, which are rare on real data.
- **Consequences:** Track P reproduces the paper Table 1 ``|d|``/``Flip`` and the reference
  ``metrics_b2_save.csv`` within the registered tolerances (see ``outputs/problem2_summary_P.json``);
  the phase-diagram replay uses first-index ties on name-sorted rosters. The distinction is pinned by
  `tests/test_problem2_rules.py` and the replay tests. Invariant scope for
  ``risk_and_bottom2``: the *save* eliminee is mode-consistent (both ``'rank'`` and ``'pct'``
  eliminate the worse judge of the bottom two and agree whenever the bottom-two pair coincides), but
  the *direct* eliminee ``elim_base`` is **not** expected to match across modes — the rank risk
  ``wJ*jr + wF*fr`` and the pct risk ``wJ*(1-J) + wF*(1-p)`` are different objective functions that
  legitimately pick different worst contestants, which *is* the paper's rank-vs-percentage premise
  (``DR``/``Flip``). An earlier draft test asserted ``r_base == p_base``; that invariant was invalid
  and was corrected to pin the save rule and a deterministic seed-7 divergence check instead.
- **Refs:** `../src/2_rank_vs_pct_cross_season.ipynb`, `../src/b2_save_metrics.py`,
  `../data/metrics_b2_save.csv`, `src/dwts_reproduction/problem2/rules.py`.

### D-20260901-10 — Mechanism phase diagram axis definitions (paper Fig 5 vs review R-040)

- **Status:** established (2026-09-01).
- **Context:** the paper's phase diagram (`../paper_Latex/2107542.tex`, lines 738-749) embeds each
  season by fan influence ``x = mu(|Ds|)`` with ``Ds = p - J`` (fan share minus judge share) and
  judge consistency ``y = 1 - mu(|Dr|)`` with ``Dr`` the *raw within-week* rank differences
  ``r_Final - r_J`` (ranks span ``1..n``). The review (`../review/notes/review_all.md`, R-040)
  defines ``x = mu(|p_i - J_i|)`` and ``y = 1 - mu(|r_Final - r_J|)`` with ``r_Final`` the
  descending rank of survival-week counts and ``r_J`` the descending rank of mean ``judge_percent``
  over alive weeks. No legacy code producer exists in ``src/``; the paper's ``y`` uses raw rank
  differences so it is **not bounded below** (typically negative), and the paper only makes
  *comparative* claims about it.
- **Options:** (a) paper axis definitions only; (b) review definitions only; (c) both behind the
  track label.
- **Choice:** (c) — Track P reports the paper's ``y = 1 - mu(|Dr|)`` (``y_posterior_mean``), Track R
  the review's survival-week-vs-judge-ranking ``y_review_posterior_mean``; ``x`` is shared. The
  replay is a counterfactual trajectory over the fitted posterior draws with carry-forward
  alive-set snapshots (the counterfactual alive set is *not* a subset of the observed alive set once
  a mechanism pre-eliminates or keeps someone the observed data did not).
- **Rationale:** each track should be checked against the axis definition its source document
  states, and the two can disagree on the recommended mechanism (R-040 predicts Perc+Bottom2 highest
  overall on both axes; the reproduced tables put ``rank_bottom2`` highest on both tracks, see the
  claim checks in ``outputs/problem2_*_phase_claim_checks_{P,R}.csv``). Reporting only one would
  silently drop the conflict the governing rules require us to surface.
- **Consequences:** ``y``/``y_review`` are not bounded below and must not be normalized away (alive
  sets differ across mechanisms under divergence, so a per-week ``(n-1)`` normalization is not
  applied); claim checks compare posterior-mean *deltas* rather than absolute ``y``. The paper's
  high fan-influence subset (``x >= 0.3``) is empty on the reproduced data, so P-057's conditional
  claim is reported as "not testable" rather than silently omitted.
- **Refs:** `../paper_Latex/2107542.tex` (Fig 5, P-056/P-057), `../review/notes/review_all.md`
  (R-040), `src/dwts_reproduction/problem2/mechanism_phase.py`,
  `tests/test_mechanism_phase.py`.

### D-20260901-11 — XGBoost baseline accuracy 0.806554 not reproducible from legacy (established)

- **Status:** established (2026-09-01).
- **Context:** the paper reports the XGBoost in-season baseline ``A = 0.806554``
  ("same features" as torch; ``../paper_Latex/2107542.tex`` line ~396). The repo port
  ``evaluate_inseason_accuracy(model_kind='xgb')`` reproduces the legacy pipeline
  ``src/xgb_baseline.py`` + ``src/compare_models_cv.py`` **bit-for-bit**: a live legacy run today
  gives xgb week-mean ``0.821101`` / season-mean ``0.817496`` (evidence
  ``/tmp/p1e_legacy/xgb_by_week_legacy.csv``), identical to the repo's week-mean ``0.821101``.
- **Options:** (a) fabricate/force the paper number; (b) keep the paper number as the registered
  target and report the legacy-reproduced line honestly; (c) drop the target silently.
- **Choice:** (b) — B-01 stays registered in ``docs/BASELINE_PAPER_OUTPUTS.md`` at the proposed
  rel-1e-3 tolerance, the paper number is explicitly marked **not reproducible from the current
  legacy code/data**, and the reproduced xgb line is reported with both week-mean (``0.821101``)
  and season-mean (``0.817496``) labelled P1E/Track P. C-07 added to the conflict matrix.
- **Rationale:** the discrepancy was diagnosed exhaustively before concluding: exact-legacy port
  (0.821101), age-filled "same features as torch" variant (0.821101), seed schemes ``s*100`` /
  ``s*1000`` / ``(s-1)*100`` (0.821101 / 0.825688 / 0.811927), feature-set variants (no-age
  ``0.899083``, no-era ``0.811927``), and a kappa sweep 1–30 (week-mean ``0.729–0.959``). None
  reaches 0.806554. Meanwhile the torch season-mean ``0.952092`` reproduces the paper exactly,
  proving (a) the paper's ``A`` is the mean of per-season means and (b) the torch pipeline is the
  current legacy one. The xgb gap is therefore a real paper-vs-legacy conflict, not a port bug.
- **Consequences:** the xgb line is reported as-is with the paper target recorded but unmet; the
  paper's "wins in every season" comparison still holds (torch season-mean 0.952092 > xgb
  season-mean 0.817496). The limitation is preserved, not hidden.
- **Refs:** ``../paper_Latex/2107542.tex``, ``../src/xgb_baseline.py``,
  ``../src/compare_models_cv.py``, ``/tmp/p1e_legacy/xgb_driver.py``,
  ``src/dwts_reproduction/problem1/baselines.py``, ``docs/BASELINE_PAPER_OUTPUTS.md`` (B-01),
  ``docs/CONFLICT_MATRIX.md`` (C-07).

### D-20260901-12 — Ranking-gap "R² > 0.6" paper claim not reproducible (established)

- **Status:** established (2026-09-01).
- **Context:** the paper claims the ranking-gap quadratic fit has ``R² > 0.6``
  (``../paper_Latex/2107542.tex`` line ~454). The repo's ``ranking_gap_frame`` is an exact port of
  ``src/week_evolution.ipynb`` cell 56 (placement first-row, judge_avg mean, audience_mean mean,
  result_minus_judge, audience_rank groupby rank, polyfit order 2). On the saved Track P posterior
  summary (kappa=10, B=1200) the fit gives ``n=421``, ``R² = 0.2704``, coeffs
  ``[-0.0474, 1.0274, 6.5607]``.
- **Options:** (a) report 0.2704 honestly and mark the paper claim unverifiable; (b) tune
  kappa/B/subset until R² exceeds 0.6 and claim success; (c) drop the claim.
- **Choice:** (a) — the figure (P-035) is produced from the saved table with the honest R²; the
  paper's ``>0.6`` claim is recorded as **not reproducible from the saved posterior data** in
  ``docs/BASELINE_PAPER_OUTPUTS.md`` and the traceability doc.
- **Rationale:** the paper's figure apparently used posterior data not recoverable from the saved
  tables (or a different model variant); manufacturing an R² above 0.6 would overstate the result
  and violate the honesty rules. Reproducing the pipeline faithfully is the deliverable.
- **Consequences:** P-035's summary/fit JSON carry ``ranking_gap_claim_r2_gt_0_6: false`` and the
  reproduced R² is reported with its n. The ranking-gap *table* (n=421) is a faithful pipeline
  reproduction.
- **Refs:** ``../paper_Latex/2107542.tex``, ``../src/week_evolution.ipynb`` cell 56,
  ``src/dwts_reproduction/problem1/structural.py``,
  ``outputs/problem1_extras_summary_P1E.json``.

### D-20260901-13 — P-029 accuracy line is a visual item; torch aggregation = mean of per-season means (established)

- **Status:** established (2026-09-01).
- **Context:** the paper Fig. 1 accuracy line (P-029) compares per-season torch vs xgb accuracy.
  The torch overall number 0.952092 is reproduced **exactly** by the repo's per-season
  ``fit_pooled_softmin`` + ``posterior_draws_for_week`` port when aggregated as the mean of
  per-season means (season-mean 0.952092); the week-mean is 0.954128. This proves the paper's
  ``A`` is ``mean_s(A_s)``, not the mean over weeks.
- **Options:** (a) register a numeric target for the torch line; (b) treat it as a visual item with
  the aggregation convention documented.
- **Choice:** (b) — the line is registered as a visual item (P-029) with no numeric target in
  ``docs/BASELINE_PAPER_OUTPUTS.md``; the aggregation convention ``mean of per-season means`` is
  documented here and in ``accuracy_by_season``. D-20260901-04 already registers the torch top-1
  accuracy target.
- **Rationale:** the line's purpose is the per-season comparison; the exact overall reproduction is
  a bonus proof of the aggregation convention, not a separate target.
- **Consequences:** ``accuracy_by_season`` documents "mean of per-season means"; the by-week table
  is kept as the source of truth so either aggregation is reproducible.
- **Refs:** ``../src/plot_cv_accuracy_line.py``, ``../src/compare_models_cv.py``,
  ``src/dwts_reproduction/problem1/baselines.py``, ``docs/BASELINE_PAPER_OUTPUTS.md``.

### D-20260901-14 — P-033 PCP weighted vs unweighted variants (established)

- **Status:** established (2026-09-01).
- **Context:** the paper's PCP formula (``../paper_Latex/2107542.tex`` line ~435) averages with
  uniform weights ``1/B`` over posterior draws — i.e. **unweighted** over the draw population. The
  Track P posterior summary also carries softmin *importance weights* (the same likelihood used to
  reweight draws); weighting by them is the conditional-expectation estimate of the same quantity.
- **Options:** (a) paper formula only (uniform); (b) importance-weighted only; (c) both.
- **Choice:** (c) — the table reports ``pcp_weighted`` and ``pcp_unweighted`` side by side.
  ``pcp_unweighted`` is the paper formula (uniform ``1/B``); ``pcp_weighted`` is the importance-
  weighted variant. Both are computed from the same saved posterior draws so the parameter
  discrepancy is visible, never silently merged.
- **Rationale:** the review's corrected posterior (Track R) is exactly the importance-weighted
  reading; keeping both makes the paper-vs-corrected difference explicit.
- **Consequences:** the crowded-field figure (P-033) renders both variants from the saved table;
  the paper headline number, when reported, is ``pcp_unweighted``.
- **Refs:** ``../paper_Latex/2107542.tex``, ``src/dwts_reproduction/problem1/structural.py``,
  ``outputs/problem1_extras_crowded_field_P1E.csv``.

### D-20260901-15 — Ranking-gap frame is an exact cell-56 port; jitter is plot-only (established)

- **Status:** established (2026-09-01).
- **Context:** the paper's ranking-gap scatter (P-035) originates in ``src/week_evolution.ipynb``
  cell 56, which applies a small normal jitter before plotting. The repo's ``ranking_gap_frame``
  produces the **un-jittered** table; jitter is applied only inside ``plot_ranking_gap`` (seeded
  ``default_rng(42)``, ``scale=0.15``) and never to ``x``/``y`` before the fit, so the quadratic
  fit and R² are deterministic functions of the saved table.
- **Options:** (a) jitter in the table (as the notebook did); (b) jitter only in the plot.
- **Choice:** (b) — fit on un-jittered data; jitter in the rendering only.
- **Rationale:** the fit must be reproducible from the saved CSV alone (CLAUDE.md: figures only from
  saved source tables); jitter in the data would make R² depend on the RNG.
- **Consequences:** the figure's scatter is jittered for legibility; the fit band and R² (0.2704,
  see D-20260901-12) are recomputed from the un-jittered table in
  ``scripts/plot_problem1_figures.py``.
- **Refs:** ``../src/week_evolution.ipynb`` cell 56,
  ``src/dwts_reproduction/problem1/structural.py``, ``src/dwts_reproduction/problem1/figures.py``.

### D-20260901-16 — Season 8 / Season 21 heatmaps adapt exit-week and p_mean (established)

- **Status:** established (2026-09-01).
- **Context:** the paper's heatmaps (P-025 Season 8, P-037 Season 21) visualize per-week posterior
  fan support and credible-interval width across a season's alive rosters. The paper's exact grid
  layout (who is alive in which week) is not recoverable from the saved posterior summary alone, so
  the heatmaps use the posterior summary's per-week ``p_mean`` (S8) and ``ci_rel_width`` (S21) over
  the alive roster each week, ordered by the reproduction's stable contestant ordering.
- **Options:** (a) reconstruct the paper grid exactly; (b) render from the saved posterior summary
  with the exit-week/``p_mean`` adaptation and document it.
- **Choice:** (b) — the source tables carry ``season``/``week``/``celebrity_name``/``p_mean``/
  ``ci_rel_width``/exit-week; the figure is a faithful rendering of the saved posterior, with the
  adaptation recorded here.
- **Rationale:** the paper's exact grid would require re-running the posterior with the paper's
  (unstated) ordering; the saved-summary rendering is reproducible and does not overstate fidelity.
- **Consequences:** P-025/P-037 are registered as renderings of the saved posterior tables; the
  adaptation is visible in the traceability doc.
- **Refs:** ``../paper_Latex/2107542.tex``, ``outputs/problem1_extras_s8_heatmap_P1E.csv``,
  ``outputs/problem1_extras_s21_heatmap_P1E.csv``, ``src/dwts_reproduction/problem1/figures.py``.

### D-20260901-17 — Problem 3 reproduction scope, honesty status, and HC1 FE SEs (established)

- **Status:** established (2026-09-01).
- **Context:** Problem 3 (P-058..P-071, survival determinants) has three sub-analyses:
  demographic divergence (OLS port of ``../src/dwts_pro_celeb_regression.py`` + the paper's exact
  Eq. (demo_model) with ``Other`` as the industry reference), professional-partner effects, and
  surprise/growth dynamics. A full-source search found **no legacy producer** for the partner or
  surprise analyses — only ``dwts_pro_celeb_regression.py`` (demographic) exists and the 5 figure
  PNGs appear only in ``../paper_Latex/img/`` — so those two analyses are reproduced from the
  paper's formulas on the registered ``data/data_3.csv`` (sha256 ``72ca124e3890…``). The paper's
  exact target values are reproduced where possible and reported honestly where not.
- **Options:**
  - (a) assume legacy producers exist for partner/surprise and re-derive from them;
  - (b) reproduce from the paper formulas on the registered input, recording the lack of a producer.
- **Choice:** (b). All partner/surprise artifacts are labelled as formula-level reproductions on
  ``data_3.csv``, and every claim check stores its own within-tolerance boolean.
- **Reproduction honesty status (Track P, all on ``data_3.csv``, n=392):**
  - B-11 age: judge coefs ``-0.0301/-0.0329/-0.0359`` (paper ~ ``-0.04``), each within abs 0.02. ✓
  - B-12 actor: judge W1 ``+0.254`` (p=0.20), fan W6 ``-1.0221`` (p=0.0002). The paper's
    ``0.16 / -0.87`` is **NOT** reproduced within abs 0.1 (``paper_P060_within_abs_0_1=False``);
    the sign pattern (positive with judges early, strongly negative with fans mid-season) is
    confirmed. Reported as direction-confirmed, not magnitude-confirmed.
  - B-13 partner tenure: ``r(H_exp, judge_w1) = 0.134`` — positive, but the paper's ``0.23`` is
    **NOT** reproduced within abs 0.05 (``paper_P064_within_abs_0_05=False``). Reported honestly.
  - B-14 surprise: ``beta1 = 0.3419`` (paper 0.34, within abs 0.05, p=1.8e-6). ✓
  - Matthew effect: ``beta2 (S^2) = 0.1819 > 0``, p<0.001. ✓
  - Veteran leverage: ``beta3 (S x H_exp) = 0.0104 > 0`` but **directional only** (p=0.51) — the
    paper's "beta3 > 0" is not significant on the reproduced data; the claim check passes
    ``beta3_gt_0`` and the summary reports the non-significance explicitly. Do not overstate.
- **HC1 choice for partner-FE SEs:** the partner-FE fits (Eq. fe_model) under HC3 emit
  ``inf`` SEs because singleton partners have leverage h=1 (``(1-h)^-2`` degenerates), seen as
  statsmodels' benign "divide by zero" RuntimeWarning. Coefficients are identical under any robust
  convention; only the SEs differ. The module therefore uses **HC1** (classic finite White
  estimator) for the FE fits and documents why, keeping HC3 for the demographic OLS pipeline
  (which has no such degeneracy). The warning is suppressed in the run script only.
- **Legacy `.eq()`/`.le()` artifact in pro-history features:** the port of
  ``dwts_pro_celeb_regression.py`` reproduces ``(s.shift(1).eq(1)).expanding().mean()`` and
  ``(s.shift(1).le(3)).expanding().mean()`` exactly: the leading ``shift`` NaN is converted to
  ``False`` by ``eq``/``le``, so a dancer's first prior appearance never contributes to
  ``win_rate``/``top3_rate``. This is a genuine legacy artifact that **must not be "fixed"** in
  the Track P port — R² parity (7/7 within 1e-4) depends on it. It is pinned by
  ``test_pro_history_leakage_safe`` and recorded here.
- **Consequences:** ``docs/BASELINE_PAPER_OUTPUTS.md`` and ``manifests/baseline.csv`` record B-11
  as reproduced, B-12/B-13 as direction-confirmed only, and B-14 as reproduced; the Problem 3 run
  manifest records every within-tolerance boolean. P-058..P-071 (and P-007/P-008/P-015) are
  ``implemented`` in the traceability inventory.
- **Refs:** ``../src/dwts_pro_celeb_regression.py`` (lines 113-122),
  ``../paper_Latex/2107542.tex``, ``data/data_3.csv``, ``src/dwts_reproduction/problem3/*``,
  ``scripts/problem3_run.py``, ``outputs/problem3_summary_P3.json``,
  ``tests/test_problem3.py``.
