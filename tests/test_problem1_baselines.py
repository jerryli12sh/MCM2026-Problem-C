"""In-season accuracy baselines: XGBoost vs torch lines (P-027 / P-029).

The XGBoost line is an exact port of the legacy ``src/xgb_baseline.py`` +
``src/compare_models_cv.py`` loop; the reproduced week-mean ``0.821101`` matches a
live legacy run bit-for-bit (the paper's registered ``0.806554`` is NOT
reproducible from the current legacy code/data — see D-20260901-11 / C-07). The
torch line's season-mean ``0.952092`` reproduces the paper exactly, which also
proves the paper's aggregation is the mean of per-season means (D-20260901-13).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dwts_reproduction.config import load_paths
from dwts_reproduction.preprocess import build_all_tables
from dwts_reproduction.problem1 import (
    XgbPooledFit,
    accuracy_by_season,
    build_problem1_panel,
    build_xgb_features,
    build_xgb_features_for_rows,
    evaluate_inseason_accuracy,
    week_accuracy_from_posterior,
)
from dwts_reproduction.problem1.config import Problem1Config

# Legacy-reproduced / paper numbers (see D-20260901-11 and D-20260901-13).
XGB_WEEK_MEAN_LEGACY = 0.821101  # what legacy xgb_baseline produces today
XGB_SEASON_MEAN_LEGACY = 0.817496
TORCH_SEASON_MEAN_PAPER = 0.952092  # mean of per-season means, reproduced exactly
N_TRAIN_WEEKS = 218
N_SEASONS = 33


# --------------------------------------------------------------------------- #
# Unit: week accuracy from a posterior frame
# --------------------------------------------------------------------------- #
def _post(eliminated_name: str | None, n: int, elim_col, p_mean) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "celebrity_name": [f"c{i}" for i in range(n)],
            "j_metric": [0.3, 0.5, 0.7][:n],
            "p_mean": p_mean,
            "elim_this_week_end": elim_col,
        }
    )


def test_week_accuracy_returns_1_when_argmin_is_eliminee():
    # 3 contestants; c1 eliminated; p_mean makes C_hat = J + p_minimal for c1.
    post = _post(
        "c1",
        3,
        [1, 0, 0],
        [0.0, 0.9, 0.9],
    )
    assert week_accuracy_from_posterior(post) == 1


def test_week_accuracy_returns_0_when_argmin_is_not_eliminee():
    post = _post(
        "c0",
        3,
        [1, 0, 0],
        [0.9, 0.0, 0.9],
    )
    assert week_accuracy_from_posterior(post) == 0


def test_week_accuracy_returns_none_without_single_elimination():
    no_elim = _post(None, 3, [0, 0, 0], [0.3, 0.3, 0.3])
    double_elim = _post(None, 3, [1, 1, 0], [0.3, 0.3, 0.3])
    assert week_accuracy_from_posterior(None) is None
    assert week_accuracy_from_posterior(no_elim) is None
    assert week_accuracy_from_posterior(double_elim) is None


# --------------------------------------------------------------------------- #
# Unit: accuracy_by_season aggregation = mean of per-season means
# --------------------------------------------------------------------------- #
def test_accuracy_by_season_is_mean_of_season_means():
    by_week = pd.DataFrame(
        {
            "model": ["x"] * 6 + ["t"] * 6,
            "season": [1, 1, 1, 2, 2, 2] * 2,
            "week": [1, 2, 3, 1, 2, 3] * 2,
            "accuracy": [1, 0, 1, 0, 0, 1] + [1, 1, 1, 0, 1, 1],
        }
    )
    agg = accuracy_by_season(by_week)
    assert len(agg) == 4
    for model in ("x", "t"):
        sub = agg[agg["model"].eq(model)]
        for _, row in sub.iterrows():
            s = int(row["season"])
            assert row["accuracy"] == pytest.approx(
                by_week[(by_week["model"].eq(model)) & (by_week["season"].eq(s))]["accuracy"].mean()
            )


def test_accuracy_by_season_sorts_by_model_then_season():
    by_week = pd.DataFrame(
        {
            "model": ["t", "x", "x", "t", "x", "t"],
            "season": [1, 1, 1, 1, 2, 2],
            "week": [1, 1, 2, 2, 1, 1],
            "accuracy": [1, 0, 1, 0, 1, 0],
        }
    )
    agg = accuracy_by_season(by_week)
    # one row per (model, season); model then season ascending ('t' < 'x')
    assert list(agg["model"]) == ["t", "t", "x", "x"]
    assert list(agg["season"]) == [1, 2, 1, 2]


# --------------------------------------------------------------------------- #
# Unit: XGBoost feature builder age handling (D-20260901-11: NaN preserved)
# --------------------------------------------------------------------------- #
def _train_rows(n: int = 3, with_nan_age: bool = True) -> pd.DataFrame:
    rows = pd.DataFrame(
        {
            "season": [1] * n,
            "week": [1] * n,
            "celebrity_name": [f"c{i}" for i in range(n)],
            "j_metric": [0.2, 0.4, 0.6],
            "age": [30.0, 40.0, np.nan if with_nan_age else 50.0],
            "era": ["percent"] * n,
            "elim_this_week_end": [0, 1, 0],
            "alive": [True] * n,
        }
    )
    return rows


def test_build_xgb_features_keeps_missing_age_as_nan():
    panel = pd.DataFrame()
    feat, meta = build_xgb_features(panel, _train_rows(with_nan_age=True))
    assert meta["use_age"] is True
    assert feat["age_z"].iloc[2] != feat["age_z"].iloc[2]  # NaN (not 0-filled)
    assert np.isfinite(feat["age_z"].iloc[:2]).all()


def test_build_xgb_features_zeros_age_only_when_no_age_data():
    rows = _train_rows(with_nan_age=True)
    rows["age"] = np.nan  # no age data at all
    feat, meta = build_xgb_features(pd.DataFrame(), rows)
    assert meta["use_age"] is False
    assert (feat["age_z"] == 0.0).all()


def test_build_xgb_features_for_rows_keeps_individual_nan_age():
    # A fit trained with any non-null age keeps NaN on individual inference rows
    # (XGBoost native missing handling) rather than filling with the fit mean.
    fit = XgbPooledFit(
        model=None,  # type: ignore[arg-type]
        X_cols=["j_metric_z", "age_z", "era_is_percent"],
        jm_mean=0.5,
        jm_std=0.5,
        use_age=True,
        age_mean=40.0,
        age_std=10.0,
        seed=7,
        kappa=10.0,
    )
    out = build_xgb_features_for_rows(_train_rows(with_nan_age=True), fit)
    assert out["age_z"].iloc[2] != out["age_z"].iloc[2]  # individual NaN stays NaN
    assert out["age_z"].iloc[0] == pytest.approx((30.0 - 40.0) / 10.0)


# --------------------------------------------------------------------------- #
# Regression: full in-season lines on the real data
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def lines():
    config = Problem1Config.for_track("P")
    paths = load_paths()
    tables = build_all_tables(paths.raw_data_csv)
    panel = build_problem1_panel(tables, config.era_mode, [])
    xgb_week = evaluate_inseason_accuracy(
        panel, "xgb", seed=42, kappa=config.kappa, tau_like=config.tau_like, B=config.B
    )
    torch_week = evaluate_inseason_accuracy(panel, "torch", config=config)
    return xgb_week, torch_week


def test_lines_cover_218_training_weeks(lines):
    xgb_week, torch_week = lines
    assert len(xgb_week) == N_TRAIN_WEEKS
    assert len(torch_week) == N_TRAIN_WEEKS


def test_xgb_week_mean_matches_legacy_reproduction(lines):
    """The repo xgb line equals the legacy run's 0.821101 (D-20260901-11)."""
    xgb_week, _ = lines
    value = float(xgb_week["accuracy"].mean())
    assert value == pytest.approx(XGB_WEEK_MEAN_LEGACY, abs=2e-3)


def test_torch_season_mean_reproduces_paper(lines):
    """Mean of per-season means equals the paper's 0.952092 (D-20260901-13)."""
    _, torch_week = lines
    season_means = torch_week.groupby("season")["accuracy"].mean()
    assert float(season_means.mean()) == pytest.approx(TORCH_SEASON_MEAN_PAPER, abs=2e-3)


def test_torch_wins_every_season(lines):
    xgb_week, torch_week = lines
    xgb_s = xgb_week.groupby("season")["accuracy"].mean()
    torch_s = torch_week.groupby("season")["accuracy"].mean()
    assert set(xgb_s.index) == set(torch_s.index)
    assert len(xgb_s) == N_SEASONS
    worst = (torch_s - xgb_s).min()
    assert worst > -1e-12, f"torch must win every season, worst margin {worst:.4f}"


def test_season_aggregation_matches_mean_of_season_means(lines):
    xgb_week, torch_week = lines
    by_week = pd.concat([xgb_week, torch_week], ignore_index=True)
    agg = accuracy_by_season(by_week)
    for model in ("xgboost_baseline", "torch_model"):
        for _, row in agg[agg["model"].eq(model)].iterrows():
            s = int(row["season"])
            assert row["accuracy"] == pytest.approx(
                by_week[(by_week["model"].eq(model)) & (by_week["season"].eq(s))]["accuracy"].mean()
            )
