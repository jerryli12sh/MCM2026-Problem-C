"""Structural Problem-1 analyses: PCP-vs-alive-size (P-033) and ranking gap (P-035).

``crowded_field_from_posterior`` collapses the saved posterior summary to one row
per season-week and keeps both PCP variants (paper formula = uniform ``1/B``;
weighted variant = importance-weighted) — see D-20260901-14.  ``ranking_gap_frame``
is an exact port of ``week_evolution.ipynb`` cell 56; the quadratic fit is computed
on un-jittered data (D-20260901-15), and the paper's ``R^2 > 0.6`` claim is NOT
reproduced (reproduced ``R^2 = 0.2704``, n=421 — see D-20260901-12).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dwts_reproduction.config import load_paths
from dwts_reproduction.preprocess import build_all_tables
from dwts_reproduction.problem1 import (
    build_problem1_panel,
    build_train_weeks,
    crowded_field_from_posterior,
    fit_pooled_softmin,
    infer_all_weekly_fan_support,
    quadratic_fit_with_ci,
    ranking_gap_frame,
)
from dwts_reproduction.problem1.config import Problem1Config

RANKING_GAP_N = 421  # reproducible from the saved posterior summary
RANKING_GAP_R2 = 0.2704
CROWDED_FIELD_N = 335  # one row per season-week in the saved crowded-field CSV


# --------------------------------------------------------------------------- #
# Synthetic checks
# --------------------------------------------------------------------------- #
def _posterior_summary(season_weeks: list[tuple[int, int]], n_cont: int = 3) -> pd.DataFrame:
    rows = []
    for s, w in season_weeks:
        for i in range(n_cont):
            rows.append(
                {
                    "season": s,
                    "week": w,
                    "celebrity_name": f"c{i}",
                    "era": "percent",
                    "alive_n": n_cont,
                    "pcp_weighted": 0.5 + 0.01 * i,
                    "pcp_unweighted": 0.4 + 0.02 * i,
                    "has_posterior": True,
                    "p_mean": 1.0 / n_cont + 0.01 * i,
                }
            )
    return pd.DataFrame(rows)


def test_crowded_field_one_row_per_season_week():
    post = _posterior_summary([(1, 1), (1, 2), (2, 1)])
    cf = crowded_field_from_posterior(post)
    assert len(cf) == 3
    assert cf["season"].tolist() == [1, 1, 2]
    assert cf["week"].tolist() == [1, 2, 1]
    assert set(cf.columns) >= {
        "season",
        "week",
        "era",
        "alive_n",
        "pcp_weighted",
        "pcp_unweighted",
        "has_posterior",
    }


def test_crowded_field_takes_first_row_per_group():
    post = _posterior_summary([(1, 1)], n_cont=3)
    cf = crowded_field_from_posterior(post)
    assert len(cf) == 1
    assert cf["alive_n"].iloc[0] == 3
    # first row (c0) values are kept
    assert cf["pcp_weighted"].iloc[0] == pytest.approx(0.5)


def _weekly() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "season": [1, 1, 1, 1, 1, 1, 2, 2, 2, 2],
            "week": [1, 2, 3, 1, 2, 3, 1, 2, 3, 4],
            "celebrity_name": ["a", "a", "a", "b", "b", "b", "a", "a", "a", "a"],
            "placement": [np.nan, np.nan, 2.0, np.nan, np.nan, 1.0, np.nan, np.nan, np.nan, 1.0],
            "judge_rank": [1.0, 2.0, 1.0, 1.0, 2.0, 2.0, 1.0, 2.0, 1.0, np.nan],
        }
    )


def _ranking_gap_post() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "season": [1, 1, 1, 1, 1, 1, 2, 2, 2],
            "week": [1, 2, 3, 1, 2, 3, 1, 2, 3],
            "celebrity_name": ["a", "a", "a", "b", "b", "b", "a", "a", "a"],
            "p_mean": [0.6, 0.5, 0.4, 0.8, 0.8, 0.8, 0.8, 0.7, 0.6],
        }
    )


def test_ranking_gap_semantics_match_cell_56():
    frame = ranking_gap_frame(_weekly(), _ranking_gap_post())
    # placement is the FIRST non-null row (sorted by season, name, week) -> 2.0 for (1,a)
    a1 = frame[(frame["season"].eq(1)) & (frame["celebrity_name"].eq("a"))]
    assert len(a1) == 1
    assert a1["placement"].iloc[0] == 2.0
    # judge_avg = mean over non-null judge_rank in season 1 for 'a' -> (1+2+1)/3
    assert a1["judge_avg_rank"].iloc[0] == pytest.approx((1 + 2 + 1) / 3)
    # audience_mean = mean p_mean over the season -> (0.6+0.5+0.4)/3
    assert a1["audience_mean"].iloc[0] == pytest.approx((0.6 + 0.5 + 0.4) / 3)
    # result_minus_judge = placement - judge_avg
    assert a1["result_minus_judge"].iloc[0] == pytest.approx(2.0 - (1 + 2 + 1) / 3)
    # audience_rank is descending within season (rank 1 = largest audience_mean)
    b1 = frame[(frame["season"].eq(1)) & (frame["celebrity_name"].eq("b"))]
    assert b1["audience_rank"].iloc[0] == 1  # season-1 b mean 0.8 > season-1 a mean 0.5
    assert a1["audience_rank"].iloc[0] == 2
    a2 = frame[(frame["season"].eq(2)) & (frame["celebrity_name"].eq("a"))]
    assert a2["audience_rank"].iloc[0] == 1


def test_quadratic_fit_recovers_perfect_quadratic():
    x = np.linspace(-2, 2, 31)
    y = 3 * x**2 - 2 * x + 1
    fit = quadratic_fit_with_ci(x, y, order=2)
    assert fit.r_squared == pytest.approx(1.0, abs=1e-12)
    assert fit.n == 31
    assert fit.coeffs.shape == (3,)
    # CI band must straddle the fitted curve on the grid
    assert (fit.ci_lo <= fit.y_fit + 1e-9).all()
    assert (fit.y_fit - 1e-9 <= fit.ci_hi).all()


def test_quadratic_fit_rejects_too_few_points():
    with pytest.raises(ValueError):
        quadratic_fit_with_ci(np.array([1.0, 2.0]), np.array([1.0, 2.0]), order=2)


def test_quadratic_fit_drops_nan_entries():
    x = np.array([1.0, 2.0, np.nan, 3.0, 4.0])
    y = np.array([1.0, 4.0, 0.0, 9.0, 16.0])
    fit = quadratic_fit_with_ci(x, y, order=2)
    assert fit.n == 4
    assert fit.r_squared == pytest.approx(1.0, abs=1e-9)


# --------------------------------------------------------------------------- #
# Regression on real data
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def real():
    config = Problem1Config.for_track("P")
    paths = load_paths()
    tables = build_all_tables(paths.raw_data_csv)
    panel = build_problem1_panel(tables, config.era_mode, [])
    train = build_train_weeks(panel)
    fit = fit_pooled_softmin(panel, train, config)
    posterior = infer_all_weekly_fan_support(panel, fit, config)
    return tables, posterior


def test_ranking_gap_reproduced_n_and_r2(real):
    tables, posterior = real
    gap = ranking_gap_frame(tables.weekly, posterior)
    fit = quadratic_fit_with_ci(
        gap["result_minus_judge"].to_numpy(), gap["audience_rank"].to_numpy()
    )
    assert fit.n == RANKING_GAP_N
    assert fit.r_squared == pytest.approx(RANKING_GAP_R2, abs=0.05)


def test_crowded_field_real_has_expected_shape_and_finite_pcp(real):
    _, posterior = real
    cf = crowded_field_from_posterior(posterior)
    assert len(cf) == CROWDED_FIELD_N  # matches saved problem1_extras_crowded_field CSV
    assert cf["season"].nunique() == 34
    assert cf["alive_n"].min() >= 2
    assert cf["era"].nunique() >= 1
    # the 218 reweighted weeks keep a finite unweighted PCP in [0, 1]
    has = cf[cf["has_posterior"]]
    assert len(has) == 218
    assert ((has["pcp_unweighted"] >= 0.0) & (has["pcp_unweighted"] <= 1.0)).all()
    assert has["pcp_unweighted"].notna().all()
