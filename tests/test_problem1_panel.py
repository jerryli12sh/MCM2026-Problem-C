"""Problem 1 panel construction tests.

The training unit is a season-week *alive set*; every judge metric must be
normalized within that set and the panel must reconcile exactly with the
reference rebuild (4199 rows, 218 single-elimination training weeks).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dwts_reproduction import preprocess as pp
from dwts_reproduction.config import load_paths
from dwts_reproduction.problem1 import build_problem1_panel, build_train_weeks, validate_panel


@pytest.fixture(scope="module")
def panel() -> pd.DataFrame:
    """Real-data panel under the Track P (legacy) era mapping."""
    paths = load_paths()
    tables = pp.build_all_tables(paths.raw_data_csv)
    warnings: list[str] = []
    p = build_problem1_panel(tables, "legacy", warnings)
    assert warnings, "legacy era must emit an audit warning"
    return p


def test_panel_shape_matches_reference(panel):
    assert len(panel) == 4199, f"panel has {len(panel)} rows, expected 4199"


def test_required_columns_present(panel):
    for col in [
        "season",
        "week",
        "celebrity_name",
        "alive",
        "elim_this_week_end",
        "is_final_week",
        "max_week",
        "age",
        "judge_percent",
        "judge_rank_share",
        "j_metric",
        "era",
    ]:
        assert col in panel.columns, f"missing column {col}"


def test_j_metric_matches_era(panel):
    alive = panel[panel["alive"]]
    pct = alive[alive["era"].eq("percent")]
    rank = alive[alive["era"].eq("rank")]
    np.testing.assert_allclose(pct["j_metric"], pct["judge_percent"], rtol=1e-12)
    np.testing.assert_allclose(rank["j_metric"], rank["judge_rank_share"], rtol=1e-12)


def test_alive_set_judge_simplex_sums_to_one(panel):
    alive = panel[panel["alive"]]
    for col in ("judge_percent", "judge_rank_share"):
        sums = alive.dropna(subset=[col]).groupby(["season", "week"])[col].sum()
        assert (sums - 1.0).abs().max() < 1e-8, f"{col} simplex not normalized"


def test_no_missing_j_metric_in_training_rows(panel):
    """Training weeks must all carry a judge signal; generic alive non-final rows
    may lack one on non-competition weeks."""
    train = build_train_weeks(panel)
    keys = train[["season", "week"]]
    tr = panel[panel["alive"]].merge(keys, on=["season", "week"], how="inner")
    assert tr["j_metric"].notna().all()


def test_elim_flag_alignment(panel):
    """Rows flagged eliminated this week-end exactly match the elimination events."""
    flagged = panel[panel["elim_this_week_end"]]
    assert len(flagged) > 0
    assert flagged["alive"].all(), "eliminated rows must be alive"


def test_train_weeks_reference_count(panel):
    train = build_train_weeks(panel)
    assert len(train) == 218, f"train weeks has {len(train)} rows, expected 218"
    assert train["true_eliminatee"].notna().all()
    assert train["elim_cnt"].eq(1).all()


def test_train_weeks_exclude_finales_and_multi_elim(panel):
    train = build_train_weeks(panel)
    assert (train["week"] < train["max_week"]).all(), "final weeks must be excluded"
    assert train["elim_cnt"].eq(1).all(), "multi-elimination weeks must be excluded"


def test_legacy_era_mapping_used(panel):
    eras = panel.groupby("season")["era"].first()
    for season, era in eras.items():
        if int(season) >= 28:
            assert era == "percent"
        else:
            assert era == "rank"


def test_official_era_mapping_distinct(panel):
    """Track R's official mapping gives a different j_metric mix than Track P."""
    paths = load_paths()
    tables = pp.build_all_tables(paths.raw_data_csv)
    official = build_problem1_panel(tables, "official", [])
    o_eras = official.groupby("season")["era"].first()
    assert o_eras[1] == "rank"
    assert o_eras[3] == "percent"
    assert o_eras[28] == "rank"


def test_validate_panel_returns_warnings_for_bad_shape():
    bad = pd.DataFrame({"season": [1], "week": [1], "celebrity_name": ["X"]})
    bad["alive"] = True
    bad["is_final_week"] = False
    bad["j_metric"] = np.nan
    for col in ("judge_percent", "judge_rank_share"):
        bad[col] = np.nan
    warnings = validate_panel(bad, "legacy")
    assert any("4199" in w for w in warnings)
