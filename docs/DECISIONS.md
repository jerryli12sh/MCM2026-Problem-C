# Decision log

Record decisions as `D-YYYYMMDD-NN` with status, context, options, choice, rationale, consequences,
review references, legacy references, and user approval when required.

## Open decisions

- Exact tie handling under each historical mechanism and season.
- Historical rule/season mapping, especially judge-save seasons and exceptional events.
- Feature set and identifiability constraints for contestant effects `u`.
- Approximation and optimization method for the integrated marginal likelihood.
- Definition of fan-influence/excitement and technical-alignment metrics used for recommendation.
- Tolerances for numerical equivalence versus deliberate correction of legacy results.

## Phase 0 — conflict decisions

These map one-to-one to `docs/CONFLICT_MATRIX.md`. Status is `open` until the owning phase
inspects the relevant legacy code or the owner approves a tolerance.

### D-20260901-01 — Era mapping direction (suspected)

- **Status:** open (suspected, not established).
- **Context:** The official mapping is seasons 1–2 rank, 3–27 percent, 28–34 rank (Bottom-2
  ~28+). Legacy `../src/model.py` reportedly uses `season>=28 -> "percent"` — the opposite.
- **Options:** (a) official only; (b) legacy only; (c) both behind an explicit `era_mode`.
- **Choice:** (c) — expose `era_mode='legacy'|'official'`; leave the Track P default uncommitted
  until the legacy era-assignment code is inspected line-by-line in Phase 1.
- **Rationale:** reproducing the paper numbers may require the legacy mapping; the review
  requires the official one. A variable named "percent" is not by itself proof of reversal.
- **Consequences:** `era_mode` config; Track P may not match paper numbers if the legacy mapping
  is confirmed reversed and the paper in fact used the official mapping.
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

### D-20260901-04 — Top-1 accuracy gap

- **Status:** open (pending owner tolerance).
- **Context:** paper reports `0.952092`; review rebuild reports `0.949541` (~0.003 absolute gap).
- **Choice:** register both; propose a tolerance and flag for owner approval before Phase 1 sets a
  reproduction target.
- **Rationale:** the gap may be seed/optimizer noise or a real pipeline difference; Phase 0 does not
  decide which.
- **Consequences:** Track P reproduction target is provisional until the gap is classified.
- **Refs:** `../paper_Latex/2107542.tex` (Problem 1),
  `../review/problem1_rebuild/outputs/problem1_summary.json`.

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

### D-20260901-06 — Fan votes are not ground truth

- **Status:** closed (labeling rule).
- **Context:** inferred `p̂` must never be described as the true official fan vote.
- **Choice:** label as posterior estimates constrained by observed outcomes, in all tracks.
- **Rationale:** fan vote totals are unobserved; only eliminations constrain the posterior.
- **Consequences:** no claim of ground-truth fan votes anywhere in the repository.
- **Refs:** `../review/problem1_rebuild/outputs/problem1_readme.md`, `CLAUDE.md`.
