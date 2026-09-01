"""Problem 2: elimination-rule comparison and named controversy case studies.

Implements the paper's rank, percentage, and Bottom-2 + judges'-save rules as
pure functions (``rules.py``) and a historical replay of the fitted weekly
fan-support posteriors (``replay.py``) that reproduces the paper Table 1
case-study inputs (``|d|``, ``Flip``) and the reference bottom-2 save metrics
(``../data/metrics_b2_save.csv``) for either track.

Historical replay conditions on the observed elimination through the fitted
posterior and is therefore explanatory, not predictive (D-20260901-02,
D-20260901-06, D-20260901-09).  Counterfactual simulation lives in Problem 4.
"""

from dwts_reproduction.problem2.mechanism_phase import (
    HIGH_FAN_INFLUENCE_X,
    mechanism_phase_metrics,
    phase_claim_checks,
    season_phase_metrics,
)
from dwts_reproduction.problem2.replay import (
    B_DIVERGENCE,
    B_MECHANISM,
    TABLE1_CASES,
    TABLE1_REFERENCE,
    DrawCache,
    b2_case_metrics,
    build_replay_inputs,
    case_divergence,
    case_weekly_probs,
    config_from_fit,
    eligible_weeks,
    load_pooled_fit,
    p_hat_unweighted,
    season_rule_metrics,
    week_judge_vector,
)
from dwts_reproduction.problem2.rules import (
    MECHANISMS,
    ascending_rank,
    descending_rank,
    elim_pct_idx,
    elim_rank_idx,
    fan_worst_idx,
    judge_vectors_from_shares,
    judge_worst_idx,
    risk_and_bottom2,
    simulate_week,
)

__all__ = [
    "HIGH_FAN_INFLUENCE_X",
    "B_DIVERGENCE",
    "B_MECHANISM",
    "MECHANISMS",
    "TABLE1_CASES",
    "TABLE1_REFERENCE",
    "DrawCache",
    "ascending_rank",
    "b2_case_metrics",
    "build_replay_inputs",
    "case_divergence",
    "case_weekly_probs",
    "config_from_fit",
    "descending_rank",
    "eligible_weeks",
    "elim_pct_idx",
    "elim_rank_idx",
    "fan_worst_idx",
    "judge_vectors_from_shares",
    "judge_worst_idx",
    "load_pooled_fit",
    "mechanism_phase_metrics",
    "p_hat_unweighted",
    "phase_claim_checks",
    "risk_and_bottom2",
    "season_phase_metrics",
    "season_rule_metrics",
    "simulate_week",
    "week_judge_vector",
]
