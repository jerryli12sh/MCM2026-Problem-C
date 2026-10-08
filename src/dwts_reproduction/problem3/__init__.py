"""Contestant attributes, professional-partner effects and support-growth models."""

from .figures import (  # noqa: F401
    plot_partner_correlation_heatmap,
    plot_partner_heterogeneity,
    plot_success_factors_heatmap,
    plot_surprise_linear,
    plot_surprise_nonlinear,
)
from .partner import (  # noqa: F401
    judge_fan_supporting,
    partner_fe_params,
    partner_fe_regressions,
    partner_trait_correlations,
)
from .regression import (  # noqa: F401
    FAN_OUTCOMES,
    JUDGE_OUTCOMES,
    OUTCOMES,
    add_pro_history_features,
    cv_table,
    engineer_features,
    extract_key_coefs,
    fit_all_ols,
    group_rare_categories,
    incremental_r2_table,
    load_data,
    paper_demo_model,
    season_forward_cv,
)
from .surprise import (  # noqa: F401
    LATE_TFINAL,
    PRIMARY_TW6,
    fit_growth_linear,
    fit_growth_quadratic,
    predict_quadratic,
    surprise_claim_checks,
    surprise_growth_frame,
)
