"""Checks for problem3."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from dwts_reproduction.config import load_paths
from dwts_reproduction.problem3 import (
    FAN_OUTCOMES,
    JUDGE_OUTCOMES,
    LATE_TFINAL,
    PRIMARY_TW6,
    add_pro_history_features,
    engineer_features,
    fit_all_ols,
    fit_growth_linear,
    fit_growth_quadratic,
    group_rare_categories,
    load_data,
    paper_demo_model,
    surprise_claim_checks,
    surprise_growth_frame,
)
from dwts_reproduction.problem3.partner import (
    partner_fe_params,
    partner_fe_regressions,
    partner_trait_correlations,
)

# The partner-FE model contains singleton partners whose HC3 leverage is 1
# (statsmodels warns in het_scale); coefficients are unaffected and the SEs of
# those rows are not used by any claim.
warnings.filterwarnings("ignore", message="divide by zero encountered in divide")


# --------------------------------------------------------------------------- #
# Unit invariants (synthetic fixtures)
# --------------------------------------------------------------------------- #
def _partner_frame() -> pd.DataFrame:
    """Three seasons for one partner; placement_z is deterministic."""
    return pd.DataFrame(
        {
            "season": [1, 2, 3, 1, 2, 3],
            "celebrity_name": ["c1", "c2", "c3", "d1", "d2", "d3"],
            "ballroom_partner": ["p1"] * 3 + ["p2"] * 3,
            "placement": [1, 3, 2, 5, 1, 4],
            "placement_z": [1.5, -0.5, 0.0, -1.0, 2.0, -2.0],
            "celebrity_age_during_season": [30, 31, 32, 40, 41, 42],
        }
    )


def test_pro_history_leakage_safe():
    """Prior-season features use only actual earlier results."""
    feat = add_pro_history_features(_partner_frame())
    p1 = feat[feat["ballroom_partner"].eq("p1")].sort_values("season")
    # Season 1: no prior history -> zero-filled.
    assert p1["pro_hist_n_prev"].iloc[0] == 0
    assert p1["pro_hist_mean_placez"].iloc[0] == 0.0
    assert p1["pro_hist_win_rate"].iloc[0] == 0.0
    assert p1["pro_hist_top3_rate"].iloc[0] == 0.0
    # Season 2: exactly season-1 placement_z (placement 1 -> a "win").
    assert p1["pro_hist_n_prev"].iloc[1] == 1
    assert p1["pro_hist_mean_placez"].iloc[1] == pytest.approx(1.5)
    assert p1["pro_hist_win_rate"].iloc[1] == pytest.approx(1.0)
    assert p1["pro_hist_top3_rate"].iloc[1] == pytest.approx(1.0)
    # Season 3: mean of seasons 1-2.  Priors were placements [1, 3]: one win,
    # both top-3 -> win_rate 1/2, top3_rate 1.
    assert p1["pro_hist_n_prev"].iloc[2] == 2
    assert p1["pro_hist_mean_placez"].iloc[2] == pytest.approx((1.5 - 0.5) / 2)
    assert p1["pro_hist_win_rate"].iloc[2] == pytest.approx(1 / 2)
    assert p1["pro_hist_top3_rate"].iloc[2] == pytest.approx(1.0)


def test_group_rare_categories_min_count():
    s = pd.Series(["A"] * 6 + ["B"] * 3 + [np.nan] * 2)
    grouped = group_rare_categories(s, min_count=6)
    assert (grouped == "A").sum() == 6
    # B (3) and NaN (2 -> "Unknown", 2 < 6) both collapse to Other.
    assert (grouped == "Other").sum() == 5


def test_surprise_growth_construction():
    """S = judge_t - fan_prev; G = fan_t - fan_prev (P-067/P-068)."""
    df = pd.DataFrame(
        {
            "season": [1, 1],
            "celebrity_name": ["a", "b"],
            "industry_grp": ["Athlete", "Singer/Rapper"],
            "ballroom_partner": ["p1", "p2"],
            "pro_hist_n_prev": [0, 2],
            "week6_judge_score_placement_z": [1.0, -0.5],
            "week6_p_score_placement_z": [0.5, 0.25],
            "week1_p_score_placement_z": [0.1, -0.1],
        }
    )
    frame = surprise_growth_frame(df, **PRIMARY_TW6)
    assert list(frame["S"]) == pytest.approx([1.0 - 0.1, -0.5 - (-0.1)])
    assert list(frame["G"]) == pytest.approx([0.5 - 0.1, 0.25 - (-0.1)])
    assert list(frame["H_exp"]) == [0, 2]


# --------------------------------------------------------------------------- #
# Real-data regression targets (read-only data_3.csv)
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def eng() -> pd.DataFrame:
    paths = load_paths()
    df = load_data(paths.data3_csv)
    assert len(df) == 392  # registered row count
    return engineer_features(df)


def test_regression_outcomes_match_official_placements(eng):
    raw = pd.read_csv(load_paths().raw_data_csv)
    raw["celebrity_name"] = raw["celebrity_name"].str.strip()
    checked = eng.merge(
        raw[["season", "celebrity_name", "placement"]],
        on=["season", "celebrity_name"],
        suffixes=("_feature", "_official"),
        validate="one_to_one",
    )
    assert len(checked) == len(eng) == 392
    assert checked.placement_feature.eq(checked.placement_official).all()
    # Within every season, a better official placement has a higher outcome score.
    for _, group in eng.groupby("season"):
        assert group.sort_values("placement").placement_z.is_monotonic_decreasing


def test_base_and_season_models_have_valid_fit(eng):
    summary, _ = fit_all_ols(eng)
    assert len(summary) == 14
    assert summary.R2.between(0, 1).all()
    assert set(summary.spec) == {"base", "seasonFE"}


def test_paper_demo_age(eng):
    """Paper Eq. (demo_model) age coef ~ -0.04 on judge outcomes (P-059)."""
    demo = paper_demo_model(eng)
    age = demo[demo["term"].eq("celebrity_age_during_season")]
    got = {
        o: float(age[age["outcome"].eq(o)]["coef"].iloc[0])
        for o in ("judge_w1", "judge_w6", "judge_w11")
    }
    assert got == pytest.approx(
        {"judge_w1": -0.0301, "judge_w6": -0.0329, "judge_w11": -0.0359}, abs=1e-3
    )
    assert all(abs(v - (-0.04)) <= 0.02 for v in got.values())  # B-11 tolerance


def test_paper_demo_actor_direction(eng):
    """Industry coefficients retain their signs and reference values."""
    demo = paper_demo_model(eng)
    actor = demo[demo["term"].str.contains("Actor/Actress", na=False)]
    assert not actor.empty  # Actor/Actress must not be the reference category
    j1 = float(actor[actor["outcome"].eq("judge_w1")]["coef"].iloc[0])
    f6 = float(actor[actor["outcome"].eq("fan_w6")]["coef"].iloc[0])
    assert j1 > 0 and f6 < 0
    assert j1 == pytest.approx(0.254, abs=0.05)
    assert f6 == pytest.approx(-1.022, abs=0.05)


def test_partner_tenure_correlation(eng):
    """Partner experience has the expected positive reference correlation."""
    corr = partner_trait_correlations(eng)
    row = corr[corr["trait"].eq("H_exp") & corr["outcome"].eq("judge_w1")]
    assert not row.empty
    r = float(row["r"].iloc[0])
    assert r > 0
    assert r == pytest.approx(0.134, abs=0.01)


def test_partner_fe_shape(eng):
    """Partner-FE model returns both traits per outcome (P-062/P-063)."""
    fe = partner_fe_regressions(eng, JUDGE_OUTCOMES + FAN_OUTCOMES)
    assert len(fe) == 12  # 6 outcomes x 2 traits (judge + fan)
    assert set(fe["trait"]) == {"H_abil", "H_exp"}
    assert (fe["p"].between(0, 1)).all()
    params = partner_fe_params(eng, "judge_w1")
    assert len(params) >= 50  # >50 partners
    assert {"alpha_p", "H_abil_mean", "outcome_mean"}.issubset(params.columns)


def test_surprise_linear_reproduces_paper(eng):
    """P-069: beta1 ~ 0.34 reproduces the paper's claim within B-14."""
    frame = surprise_growth_frame(eng, **PRIMARY_TW6)
    assert len(frame) == 173
    fit = fit_growth_linear(frame)
    assert fit.coefs["S"] == pytest.approx(0.3419, abs=0.01)
    assert fit.coefs["S"] == pytest.approx(0.34, abs=0.05)  # B-14 tolerance
    assert fit.pvalues["S"] < 0.001


def test_surprise_quadratic_matthew(eng):
    """P-070: beta2 (S^2) > 0, significant; P-071: beta3 > 0 directional only."""
    frame = surprise_growth_frame(eng, **PRIMARY_TW6)
    fit = fit_growth_quadratic(frame)
    assert fit.coefs["I(S ** 2)"] == pytest.approx(0.1819, abs=0.01)
    assert fit.coefs["I(S ** 2)"] > 0
    assert fit.pvalues["I(S ** 2)"] < 0.001
    b3 = fit.coefs["S:H_exp"]
    assert b3 > 0
    # The interaction interval includes zero.
    assert fit.pvalues["S:H_exp"] > 0.05


def test_surprise_claim_checks_shape(eng):
    claims = surprise_claim_checks(surprise_growth_frame(eng, **PRIMARY_TW6))
    assert len(claims) == 3
    assert set(claims["claim"]) == {
        "beta1 (S) ~ 0.34, p<0.001",
        "beta2 (S^2) > 0 (Matthew effect)",
        "beta3 (S x H_exp) > 0 (veteran amplifies)",
    }


def test_late_stage_surprise(eng):
    """t=final variant uses fan_final as the late fan signal (documented)."""
    frame = surprise_growth_frame(eng, **LATE_TFINAL)
    assert len(frame) == 105
    assert {"S", "G", "H_exp", "industry_grp"}.issubset(frame.columns)
    fit = fit_growth_quadratic(frame)
    assert fit.n == 105
    assert set(fit.coefs) == {"Intercept", "S", "I(S ** 2)", "S:H_exp"}
