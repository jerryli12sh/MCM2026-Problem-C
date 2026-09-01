"""Preprocessing unit, invariant, and regression tests.

The pure transformations (result parsing, season-length / horizon inference, structural
zero cleaning) are tested on hand-built fixtures.  The full pipeline is regression-tested
against the review rebuild's validation targets (R-01..R-19) using the read-only raw data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dwts_reproduction import preprocess as pp


# --------------------------------------------------------------------------- #
# Unit fixtures
# --------------------------------------------------------------------------- #
def _raw_frame() -> pd.DataFrame:
    """Two seasons, three weeks each: one single elim, one multi elim, one finale."""
    score_cols = [f"week{w}_judge{j}_score" for w in range(1, 4) for j in range(1, 3)]
    cols = [
        "celebrity_name",
        "ballroom_partner",
        "celebrity_industry",
        "celebrity_homestate",
        "celebrity_homecountry/region",
        "celebrity_age_during_season",
        "season",
        "results",
        "placement",
    ] + score_cols
    rows = [
        # Season 1
        ["A", "p1", "Actor", "CA", "United States", 40, 1, "1st Place", 1, 8, 8, 9, 9, 9, 9],
        [
            "B",
            "p2",
            "Athlete",
            "TX",
            "United States",
            30,
            1,
            "Eliminated Week 1",
            4,
            7,
            7,
            0,
            0,
            0,
            0,
        ],
        [
            "C",
            "p3",
            "Singer",
            "NY",
            "United States",
            35,
            1,
            "Eliminated Week 2",
            3,
            8,
            8,
            7,
            7,
            0,
            0,
        ],
        ["D", "p4", "Comedian", "OH", "United States", 45, 1, "Withdrew", 2, 6, 6, 6, 6, 0, 0],
        # Season 2
        [
            "E",
            "p5",
            "Actor",
            "FL",
            "United States",
            50,
            2,
            "Eliminated Week 1",
            1,
            7,
            7,
            0,
            0,
            0,
            0,
        ],
        ["F", "p6", "Athlete", "GA", "United States", 55, 2, "2nd Place", 2, 8, 8, 8, 8, 8, 8],
    ]
    return pd.DataFrame(rows, columns=cols)


def test_identify_score_columns_sorted():
    df = _raw_frame()
    cols = pp.identify_score_columns(df)
    assert cols == [f"week{w}_judge{j}_score" for w in range(1, 4) for j in range(1, 3)]


def test_standardize_columns_renames_and_strips():
    df = _raw_frame()
    out = pp.standardize_columns(df)
    assert "celebrity_homecountry_region" in out.columns
    assert "celebrity_homecountry/region" not in out.columns


def test_parse_results_flags():
    df = pp.parse_results(pp.standardize_columns(_raw_frame()))
    elim = df["elim_week_result"].astype(float).tolist()
    expected_elim = [np.nan, 1.0, 2.0, np.nan, 1.0, np.nan]
    assert [e for e in elim if not pd.isna(e)] == [e for e in expected_elim if not pd.isna(e)]
    assert pd.isna(elim[0]) and pd.isna(elim[3]) and pd.isna(elim[5])
    assert list(df["is_withdrew"].astype(bool)) == [False, False, False, True, False, False]
    assert list(df["is_place"].astype(bool)) == [True, False, False, False, False, True]


def test_season_lengths_infer_from_positive_totals():
    df = pp.parse_results(pp.standardize_columns(_raw_frame()))
    cols = pp.identify_score_columns(df)
    totals = pp.compute_week_totals(df, cols)
    lengths = pp.infer_season_lengths(df, totals)
    # Season 1 has A positive through week 3 -> length 3.  Season 2 has F through week 3 -> 3.
    assert int(lengths.loc[1]) == 3
    assert int(lengths.loc[2]) == 3


def test_activity_windows_horizon_rules():
    df = pp.parse_results(pp.standardize_columns(_raw_frame()))
    cols = pp.identify_score_columns(df)
    totals = pp.compute_week_totals(df, cols)
    win = pp.infer_activity_windows(df, totals)
    # A placement -> season length (3); B elim w1 -> 1; C elim w2 -> 2; D withdrew -> last positive (2).
    assert win.loc[0, "active_until"] == 3.0
    assert win.loc[1, "active_until"] == 1.0
    assert win.loc[2, "active_until"] == 2.0
    assert win.loc[3, "active_until"] == 2.0


def test_structural_zero_cleaning_post_elimination():
    df = pp.parse_results(pp.standardize_columns(_raw_frame()))
    cols = pp.identify_score_columns(df)
    totals = pp.compute_week_totals(df, cols)
    win = pp.infer_activity_windows(df, totals)
    clean, _ = pp.clean_structural_zeros(win, cols)
    # B eliminated week 1 -> week2/week3 structural zeros become NaN.
    assert np.isnan(clean.loc[1, "week2_judge1_score"])
    assert np.isnan(clean.loc[1, "week3_judge1_score"])
    # A (placement) keeps its week3 scores.
    assert clean.loc[0, "week3_judge1_score"] == 9


def test_elimination_events_final_multi_distinct():
    df = pp.parse_results(pp.standardize_columns(_raw_frame()))
    cols = pp.identify_score_columns(df)
    totals = pp.compute_week_totals(df, cols)
    win = pp.infer_activity_windows(df, totals)
    clean, _ = pp.clean_structural_zeros(win, cols)
    long_judge = pp.build_long_judge_table(clean, cols)
    weekly = pp.build_weekly_table(clean, long_judge)
    roster = pp.build_roster_table(weekly)
    season_max = clean.drop_duplicates("season").set_index("season")["season_max_week"].astype(int)
    events = pp.build_elimination_events(roster, season_max)
    s1 = events[events["season"] == 1].reset_index(drop=True)
    # week1 single elim (B), week2 multi elim (C+D), week3 finale (A).
    assert s1["m_elim"].tolist() == [1, 2, 1]
    assert s1["is_final_week_end"].tolist() == [False, False, True]


def test_era_mapping_official_vs_legacy():
    assert pp.assign_official_rule_method(1) == "rank"
    assert pp.assign_official_rule_method(3) == "percent"
    assert pp.assign_official_rule_method(27) == "percent"
    assert pp.assign_official_rule_method(28) == "rank"
    assert pp.legacy_code_era_mapping(27) == "rank"
    assert pp.legacy_code_era_mapping(28) == "percent"


# --------------------------------------------------------------------------- #
# Full-pipeline regression against the review rebuild targets
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def tables():
    from dwts_reproduction.config import load_paths

    paths = load_paths()
    return pp.build_all_tables(paths.raw_data_csv)


def test_validation_report_all_required_pass(tables):
    report = tables.validation
    required = report[report["status"] != "warn"]
    assert set(required["status"]) == {"pass"}
    warns = report[report["status"] == "warn"]
    assert list(warns["check"]) == ["era_mapping_audit"]


def test_shapes_match_reference(tables):
    assert tables.clean.shape == (421, 59)
    assert tables.long_judge.shape == (18524, 20)
    assert tables.weekly.shape == (4199, 23)
    assert tables.roster.shape == (4199, 10)
    assert tables.elim_events.shape == (292, 5)


def test_keys_unique_at_grain(tables):
    assert tables.clean.duplicated(["season", "celebrity_name"]).sum() == 0
    assert tables.weekly.duplicated(["season", "celebrity_name", "week"]).sum() == 0
    assert tables.long_judge.duplicated(["season", "celebrity_name", "week", "judge"]).sum() == 0


def test_judge_percent_sums_to_one(tables):
    valid = tables.weekly[tables.weekly["performed"] & tables.weekly["eligible"]]
    sums = valid.groupby(["season", "week"])["judge_percent"].sum()
    assert float((sums - 1.0).abs().max()) <= 1e-10


def test_no_eliminated_contestant_in_later_alive_set(tables):
    leaving = tables.roster[
        tables.roster["eligible"] & ~tables.roster["eligible_next"].astype(bool)
    ]
    bad = 0
    for _, row in leaving.iterrows():
        later = tables.roster[
            (tables.roster["season"] == row["season"])
            & (tables.roster["celebrity_name"] == row["celebrity_name"])
            & (tables.roster["week"] > row["week"])
            & (tables.roster["eligible"])
        ]
        bad += len(later)
    assert bad == 0


def test_event_types_distinct(tables):
    assert tables.elim_events["is_final_week_end"].sum() == 34
    non_final = tables.elim_events[~tables.elim_events["is_final_week_end"]]
    assert non_final["m_elim"].value_counts().sort_index().to_dict() == {1: 218, 2: 37, 3: 3}


def test_repro_exact_match_to_review_rebuild(tables):
    """The five tables match the review's independent rebuild column-for-column."""
    from dwts_reproduction.config import load_paths

    paths = load_paths()
    rev = paths.source_root / "review" / "srcs_0"

    for fname, mine in [
        ("df_clean.csv", tables.clean),
        ("df_weekly.csv", tables.weekly),
        ("df_roster.csv", tables.roster),
        ("df_long_judge.csv", tables.long_judge),
    ]:
        ref = pd.read_csv(rev / fname)
        cols = [c for c in ref.columns if c in mine.columns]
        a = mine[cols].reset_index(drop=True)
        b = ref[cols].reset_index(drop=True)
        for c in cols:
            x, y = a[c], b[c]
            if pd.api.types.is_float_dtype(x) or pd.api.types.is_float_dtype(y):
                assert np.allclose(x.astype(float), y.astype(float), equal_nan=True), (fname, c)
            else:
                assert (x.fillna("<NA>").astype(str) == y.fillna("<NA>").astype(str)).all(), (
                    fname,
                    c,
                )
