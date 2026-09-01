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
