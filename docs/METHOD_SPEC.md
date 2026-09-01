# Method specification

This repository has two explicit analytical tracks.

- **Track P — Paper-faithful (primary):** reproduce `../../paper_Latex/2107542.tex` and the actual
  computations evidenced by `../../src/`.
- **Track R — Review-corrected (secondary):** implement `../../review/notes/review_all.md`, especially
  corrections that differ from the submitted paper.

Shared components must be reused where mathematically identical. Conflicting components must remain
separate behind explicit configuration. Every output filename, table, figure, metric, and manifest
must contain its track.

## Shared data contract

The canonical unit is contestant-season-week `(contestant_id, season, week)`. Define season length
as the latest week with any positive judge total. Define a contestant horizon from parsed result:
elimination week for regular eliminations, season length for placements, and last positive-score week
for withdrawals. A row is eligible iff its week is no later than both horizons. Structural zeroes
after the contestant horizon become missing; original missingness remains distinguishable in audit
fields. Fan shares and judge normalization operate only within the eligible set.

An elimination event is an eligible-to-ineligible transition. Finale endings, withdrawals, multiple
eliminations, and judge-save events are separate event types and cannot silently enter the regular
single-elimination likelihood.

Percentage judge signal is the contestant's judge total divided by the alive-set total. Historical
rank replay uses the actual summed judge ranks. A transformed rank signal may be used as a comparable
model feature, but never as a substitute for the historical rule.

## Track P: submitted-paper model

Implement the paper's two-stage procedure exactly:

1. Compute `q = softmax(X beta + u)` within each alive set.
2. Fit `beta`, contestant-season effects `u`, and temperature `tau` using the penalized softmin
   likelihood based on `C = J + q` and observed regular single-elimination outcomes.
3. Set the weekly prior `p ~ Dirichlet(kappa q)`.
4. Condition that week's `p` on the observed eliminatee using the Dirichlet-softmin likelihood and
   approximate posterior expectations and intervals by importance sampling.

This track intentionally reproduces the paper even though the same elimination outcome informs both
the fitted trend and weekly posterior. All corresponding accuracy/PCP claims must be described as
internal validity, reconstruction, or explanatory consistency—not independent prediction.

Paper-stated hyperparameters, features, rules, and reported metrics are reproduction targets. When
the text and legacy code disagree, record both in the conflict matrix and preserve the version that
actually generated the paper output as a named paper-implementation variant if evidence supports it.

## Track R: review-corrected model

Implement the review's integrated formulation:

`P(Y | beta, u) = integral P(Y | p, J) Dirichlet(p | kappa q(beta, u)) dp`.

Fit global parameters against this marginal likelihood, then compute weekly posteriors from the same
generative model. This avoids the Track P double use. The approximation must report Monte Carlo error,
effective sample size, convergence, and sensitivity to seed/sample count.

## Evaluation contract

Historical reconstruction that conditions on the observed weekly outcome is explanatory/in-sample,
not predictive. Prediction claims require held-out seasons and no access to their outcome during
inference. Report top-1, rank-sensitive and probabilistic metrics; compare to an observable-feature
baseline under identical splits.

Report Track P and Track R side by side, including why they differ.

## Rule comparison and downstream analyses

For each track, implement rank sum, percentage sum, and optional bottom-2 judges-save. Quantify fan override,
save reversal, judge/fan discrepancy, technical alignment, and posterior uncertainty. Named case
studies include Jerry Rice, Billy Ray Cyrus, Bristol Palin, Bobby Bones, Tinashe, and Vinny Guadagnino.

Track P reproduces the paper's V1/V2 mechanism definitions and stated parameter values. Track R uses
the final review formulation. Where they differ, run both rather than creating a hybrid. For the
celebrity/partner analysis, reproduce the paper first, then add review-requested corrections and
clearly separate associational statements from causal claims.
