"""Sensitivity analysis for the fan-vote inference (paper A1-A4, Figure 10).

Ports the legacy ``../src/sensitivity_analysis_a.py`` and
``../src/sensitivity_viz_a.py`` onto the repo's Problem 1 pipeline so the four
perturbation families (A1 tau/kappa grid, A2 lambda ratio, A3 judge transforms,
A4 leave-one-season-out) and the paper's Figure 10 panels (stability scatter,
PCP tornado, A1 line) are reproducible from saved tables and run manifests.
Track P (paper-faithful): ``era_mode="legacy"``.
"""

from dwts_reproduction.sensitivity.analysis import (
    add_stability_metrics,
    build_grid_values,
    build_judge_rank_share_variant,
    build_panel_with_variant,
    compute_metrics,
    compute_u_var_share,
    infer_week_posterior,
    js_distance,
    run_a1_grid,
    run_a2_lambda_scan,
    run_a3_judge_transform,
    run_a4_leave_one_season_out,
    select_nearby_grid,
    spearman_corr,
)
from dwts_reproduction.sensitivity.claims import (
    check_all,
    check_p091,
    check_p092,
    check_p093,
    effect_sizes,
)

__all__ = [
    "add_stability_metrics",
    "build_grid_values",
    "build_judge_rank_share_variant",
    "build_panel_with_variant",
    "check_all",
    "check_p091",
    "check_p092",
    "check_p093",
    "compute_metrics",
    "compute_u_var_share",
    "effect_sizes",
    "infer_week_posterior",
    "js_distance",
    "run_a1_grid",
    "run_a2_lambda_scan",
    "run_a3_judge_transform",
    "run_a4_leave_one_season_out",
    "select_nearby_grid",
    "spearman_corr",
]
