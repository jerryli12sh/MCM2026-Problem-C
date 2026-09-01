# Agent instructions

## Mission

Reproduce the DWTS MCM analysis as a clean, testable Python repository. The primary deliverable is a
faithful reproduction of `../paper_Latex/2107542.tex` (**Track P**). The secondary deliverable is the
review-corrected approach in `../review/notes/review_all.md` (**Track R**). Use legacy scripts to
recover operational details. Never silently combine conflicting methods.

## Non-negotiable rules

- Treat `../data/`, `../src/`, `../review/`, `../paper_Latex/`, and all PDFs as read-only.
- Work only inside this `repo/` directory.
- Preserve the paper's formulas, sample definitions, hyperparameters, and rule implementations in
  Track P even when the review later criticizes them.
- When paper and review conflict, implement both Track P and Track R behind explicit configuration,
  add tests for both, and produce a comparison. Do not choose one silently.
- Do not reinterpret or “improve” either method without recording a decision request in
  `docs/DECISIONS.md`. Continue unrelated work; stop only the affected decision path.
- Never claim that latent fan votes are ground truth. Report them as posterior estimates constrained
  by observed outcomes.
- Track P intentionally reproduces the paper's two-stage procedure: fit `q` from elimination outcomes,
  then condition weekly `p` on the observed elimination. Label resulting reconstruction metrics as
  internal/explanatory and document the double-use limitation.
- Track R implements the integrated marginal-likelihood formulation proposed in the review to avoid
  that double use. Never report Track P and Track R metrics without their track label.
- Fit preprocessing parameters and statistical models on training folds only. Any historical
  reconstruction metric using the observed outcome must be labeled in-sample/explanatory.
- Use deterministic seeds and stable contestant-season-week identifiers.
- Do not copy notebook cells wholesale. Extract intent, then implement small typed functions with
  docstrings, tests, and explicit inputs/outputs.
- No absolute machine-specific paths in production code. Resolve paths through configuration.
- Keep generated files, virtual environments, caches, and secrets out of Git.

## Work protocol

1. Read `docs/METHOD_SPEC.md`, `docs/ACCEPTANCE.md`, `docs/RUNBOOK.md`, and the active phase in `PLAN.md`.
2. Inspect only the legacy files relevant to that phase and record them in the commit message or
   run manifest.
3. Before implementation, add or update tests for the phase's invariants.
4. Implement the smallest complete vertical slice.
5. Run formatting, static checks, unit tests, integration tests, and the phase acceptance command.
6. Update `docs/DECISIONS.md` for every ambiguity and `PLAN.md` with evidence of completion.
7. Show the user the diff, test summary, metric comparison, cost/usage report, and unresolved risks.
8. You may make local checkpoint commits after all automated gates pass. Never push, merge, rebase,
   rewrite history, or alter a formula/sample definition/conclusion without explicit user approval.

## Stop conditions

Stop the affected path and record a decision when the paper, review, and legacy behavior conflict; an
equation is ambiguous; an expected paper result cannot be reproduced within tolerance; data must be
edited; or a choice could materially change a conclusion. Continue independent work where safe.

## Coding standard

Target Python 3.11+. Prefer pathlib, pandas/numpy/scipy, scikit-learn/statsmodels, and PyTorch only
where gradients or sampling justify it. Separate pure transformations from file I/O. Functions
should be short enough to test directly. Public APIs require type hints and concise docstrings.

## Model usage discipline

Use plan mode for phase design and normal mode for one accepted phase at a time. Do not issue a
single prompt to rebuild the entire project. Keep each session scoped to a deliverable that can be
verified in under 30 minutes. At the end of every session, request a compact handoff containing:
changed files, commands run, tests/metrics, assumptions, remaining risks, and the next exact task.
