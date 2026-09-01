# Decision log

Record decisions as `D-YYYYMMDD-NN` with status, context, options, choice, rationale, consequences,
review references, legacy references, and user approval when required.

## Open decisions

- Exact tie handling under each historical mechanism and season.
- Historical rule/season mapping, especially judge-save seasons and exceptional events.
- Feature set and identifiability constraints for contestant effects `u`.
- Approximation and optimization method for the integrated marginal likelihood.
- Definition of fan-influence/excitement and technical-alignment metrics used for recommendation.
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
