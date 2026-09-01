"""Problem 1: latent fan-support reconstruction (Tracks P and R).

Track P reproduces the paper's two-stage procedure (pooled softmin popularity
prior ``q``, then weekly ``p ~ Dirichlet(kappa q)`` posterior conditioned on the
observed elimination).  Track R adds the review's integrated marginal-likelihood
fit.  The reference outputs and metric targets are in
``docs/BASELINE_PAPER_OUTPUTS.md``.
"""

from dwts_reproduction.problem1.config import Problem1Config, Problem1Paths, load_problem1_config
from dwts_reproduction.problem1.evaluate import (
    build_event_tables,
    compute_cumulative_consistency,
    evaluate_top1_accuracy,
    s_bar,
    summarize_posterior,
)
from dwts_reproduction.problem1.panel import (
    build_problem1_panel,
    build_train_weeks,
    validate_panel,
)
from dwts_reproduction.problem1.track_p import (
    PooledFit,
    fit_pooled_softmin,
    infer_all_weekly_fan_support,
    pooled_q_for_week,
    posterior_draws_for_week,
)
from dwts_reproduction.problem1.track_r import (
    fit_integrated_marginal,
    fit_sensitivity,
    integrated_week_terms,
    marginal_likelihood_diagnostics,
)

__all__ = [
    "PooledFit",
    "Problem1Config",
    "Problem1Paths",
    "build_event_tables",
    "build_problem1_panel",
    "build_train_weeks",
    "compute_cumulative_consistency",
    "evaluate_top1_accuracy",
    "fit_integrated_marginal",
    "fit_pooled_softmin",
    "fit_sensitivity",
    "infer_all_weekly_fan_support",
    "integrated_week_terms",
    "load_problem1_config",
    "marginal_likelihood_diagnostics",
    "posterior_draws_for_week",
    "pooled_q_for_week",
    "s_bar",
    "summarize_posterior",
    "validate_panel",
]
