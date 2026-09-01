"""Problem 1 evaluation tests: event tables and cumulative consistency S_s.

The paper's season-path consistency (``S_bar ~ 0.78``, B-04) is reproduced from
the legacy event-table semantics — every single-elimination event including
finales is softmin-reweighted — over all 292 elimination events.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dwts_reproduction.config import load_paths
from dwts_reproduction.preprocess import build_all_tables
from dwts_reproduction.problem1 import (
    build_event_tables,
    build_problem1_panel,
    build_train_weeks,
    compute_cumulative_consistency,
    evaluate_top1_accuracy,
    fit_pooled_softmin,
    s_bar,
)
from dwts_reproduction.problem1.config import Problem1Config

REF_S_BAR = 0.78


@pytest.fixture(scope="module")
def event_outputs():
    config = Problem1Config.for_track("P")
    paths = load_paths()
    tables = build_all_tables(paths.raw_data_csv)
    panel = build_problem1_panel(tables, config.era_mode)
    train = build_train_weeks(panel)
    fit = fit_pooled_softmin(panel, train, config)
    events = build_event_tables(panel, fit, tables.elim_events, config)
    cum = compute_cumulative_consistency(events["event_long"], events["event_table"])
    return events, cum


def test_event_table_has_292_events(event_outputs):
    events, _ = event_outputs
    assert len(events["event_table"]) == 292, "reference event count is 292 eliminations"


def test_event_table_columns(event_outputs):
    events, _ = event_outputs
    et = events["event_table"]
    for col in (
        "season",
        "week",
        "alive_list",
        "elim_obs_list",
        "m_elim",
        "alive_n",
        "c_hat_list",
        "pi_hat_list",
        "p_mean_list",
        "has_posterior",
        "ess",
        "B",
    ):
        assert col in et.columns


def test_event_long_rows_match_alive_counts(event_outputs):
    events, _ = event_outputs
    long = events["event_long"]
    per_event = long.groupby(["season", "week"]).size()
    et = events["event_table"].set_index(["season", "week"])
    for (s, w), n in per_event.items():
        assert n == et.loc[(s, w), "alive_n"]


def test_event_long_is_elim_marker_consistent(event_outputs):
    events, _ = event_outputs
    et = events["event_table"]
    long = events["event_long"]
    for _, e in et.iterrows():
        elim = set(str(x) for x in e["elim_obs_list"].split("|"))
        sub = long[(long["season"] == e["season"]) & (long["week"] == e["week"])]
        assert sub["is_elim_obs"].sum() == len(elim)


def test_pi_hat_sums_to_one_per_event(event_outputs):
    """pi_hat is a softmin over the alive set, so each event sums to 1 — except
    the 3 events whose alive set carries a missing judge signal, where pi_hat is
    NaN exactly as the legacy ``_softmin_prob`` (``np.max`` propagates NaN)."""
    events, _ = event_outputs
    long = events["event_long"]
    missing_j_events = long.groupby(["season", "week"])["j_metric"].apply(
        lambda s: not s.notna().all()
    )
    assert int(missing_j_events.sum()) == 3, "exactly 3 events lack a full judge signal"
    has_j = long.groupby(["season", "week"])["j_metric"].transform(lambda s: s.notna().all())
    full_sums = long[has_j].groupby(["season", "week"])["pi_hat"].sum()
    assert (full_sums - 1.0).abs().max() < 1e-6, "softmin pi_hat must sum to 1 per event"


def test_s_bar_reproduces_reference(event_outputs):
    events, cum = event_outputs
    value = s_bar(cum)
    assert value == pytest.approx(REF_S_BAR, abs=0.02), f"S_bar = {value:.4f}, expected ~0.78"


def test_s_bar_with_require_posterior_is_also_stable(event_outputs):
    """Requiring a posterior drops the 74 multi-elimination events (legacy mode
    reweights every single-elim week); the remaining subset still gives a valid
    consistency score close to the full-sample value."""
    events, _ = event_outputs
    cum_strict = compute_cumulative_consistency(
        events["event_long"], events["event_table"], require_posterior=True
    )
    value = s_bar(cum_strict)
    assert 0.5 <= value <= 1.0, f"strict-mode S_bar = {value:.4f} is outside a plausible band"


def test_cumulative_consistency_season_scopes(event_outputs):
    _, cum = event_outputs
    assert len(cum) == cum["season"].nunique()
    assert (cum["K_s"] >= 1).all()
    assert cum["S_s"].between(0.0, 1.0).all()


def test_summarize_posterior_aggregates_full_summary():
    from dwts_reproduction.problem1 import summarize_posterior

    posterior = pd.DataFrame(
        {
            "pcp_weighted": [0.5, np.nan, 0.7],
            "ess_ratio": [0.9, 0.8, 0.7],
            "ci_rel_width": [2.0, 3.0, 4.0],
        }
    )
    summary = summarize_posterior(posterior, {"overall_top1_accuracy": 0.95})
    assert summary["overall_top1_accuracy"] == 0.95
    assert summary["mean_pcp_weighted"] == pytest.approx(0.6)
    assert summary["mean_ess_ratio"] == pytest.approx(0.8)
    assert summary["mean_ci_rel_width"] == pytest.approx(3.0)


def test_top1_accuracy_structure():
    # minimal synthetic check of the by-week/by-season aggregation
    posterior = pd.DataFrame(
        {
            "season": [1, 1, 1, 1, 2, 2],
            "week": [1, 1, 2, 2, 1, 1],
            "celebrity_name": ["A", "B", "A", "B", "A", "B"],
            "j_metric": [0.9, 0.1, 0.8, 0.1, 0.9, 0.1],
            "p_mean": [0.5, 0.05, 0.5, 0.05, 0.5, 0.05],
            "pcp_unweighted": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            "pcp_weighted": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            "ess_ratio": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            "alive_n": [2, 2, 2, 2, 2, 2],
        }
    )
    panel = posterior.copy()
    train = pd.DataFrame(
        {
            "season": [1, 1, 2],
            "week": [1, 2, 1],
            "true_eliminatee": ["B", "B", "B"],
            "elim_cnt": [1, 1, 1],
            "alive_n": [2, 2, 2],
            "max_week": [3, 3, 3],
        }
    )
    by_week, by_season, summary = evaluate_top1_accuracy(panel, posterior, train)
    assert len(by_week) == 3
    assert by_week["correct"].all()
    assert summary["overall_top1_accuracy"] == 1.0
    assert by_season["top1_accuracy"].eq(1.0).all()
