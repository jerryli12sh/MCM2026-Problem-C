"""Canonical DWTS preprocessing (shared by Track P and Track R).

This module reconstructs the preprocessing layer used by the original project and the
independent review rebuild.  It stops at table construction: no fan-vote inference,
model training, or rule simulation is performed here.

It is a faithful, typed reimplementation of ``review/srcs_0/0_dwts_preprocess.py``,
preserving the exact result-state semantics, season-length and activity-horizon
inference, and structural-zero cleaning described in ``docs/METHOD_SPEC.md`` and
``docs/DATA_DICTIONARY.md``.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

SCORE_RE = re.compile(r"^week(\d+)_judge(\d+)_score$")

# Canonical stable identifier columns shared across tables.
_STATIC_COLUMNS = (
    "season",
    "celebrity_name",
    "ballroom_partner",
    "celebrity_industry",
    "celebrity_homestate",
    "celebrity_homecountry_region",
    "celebrity_age_during_season",
    "results",
    "placement",
    "elim_week_result",
    "is_withdrew",
    "is_place",
    "season_max_week",
    "active_until",
    "last_week_positive",
)


@dataclass
class PreprocessTables:
    """Container for every table produced by preprocessing."""

    raw: pd.DataFrame
    clean: pd.DataFrame
    long_judge: pd.DataFrame
    weekly: pd.DataFrame
    roster: pd.DataFrame
    elim_events: pd.DataFrame
    zero_audit: pd.DataFrame
    validation: pd.DataFrame


# --------------------------------------------------------------------------- #
# Raw I/O
# --------------------------------------------------------------------------- #
def load_raw_data(path: str | Path) -> pd.DataFrame:
    """Read the original contestant-season wide CSV (BOM-safe) without changing values."""
    return pd.read_csv(path, encoding="utf-8-sig")


def _score_col_week_judge(col: str) -> tuple[int, int]:
    """Return the ``(week, judge)`` parsed from a ``weekX_judgeY_score`` column name."""
    m = SCORE_RE.match(col)
    assert m is not None, f"not a score column: {col!r}"
    return int(m.group(1)), int(m.group(2))


def identify_score_columns(df: pd.DataFrame) -> list[str]:
    """Return ``weekX_judgeY_score`` columns sorted by (week, judge)."""
    cols = [c for c in df.columns if SCORE_RE.match(c)]
    return sorted(cols, key=_score_col_week_judge)


def _max_week_from_columns(score_cols: Sequence[str]) -> int:
    if not score_cols:
        raise ValueError("No score columns matching weekX_judgeY_score were found.")
    return max(_score_col_week_judge(c)[0] for c in score_cols)


def _score_columns_for_week(score_cols: Sequence[str], week: int) -> list[str]:
    return [c for c in score_cols if _score_col_week_judge(c)[0] == week]


# --------------------------------------------------------------------------- #
# Result-state parsing and season/horizon inference
# --------------------------------------------------------------------------- #
def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename the awkward ``homecountry/region`` column and strip text for stable joins."""
    out = df.rename(columns={"celebrity_homecountry/region": "celebrity_homecountry_region"})
    for col in out.columns:
        if pd.api.types.is_object_dtype(out[col]) or pd.api.types.is_string_dtype(out[col]):
            out[col] = out[col].astype("string").str.strip()
    if "season" in out.columns:
        out["season"] = pd.to_numeric(out["season"], errors="raise").astype(int)
    return out


def parse_results(df: pd.DataFrame) -> pd.DataFrame:
    """Parse semi-structured result text into competition-state flags."""
    out = df.copy()
    if "results" not in out.columns:
        raise KeyError("Expected a `results` column in the raw DWTS data.")
    result_text = out["results"].astype("string")
    elim_week = result_text.str.extract(r"Eliminated Week\s+(\d+)", expand=False)
    out["elim_week_result"] = pd.to_numeric(elim_week, errors="coerce")
    out["is_withdrew"] = result_text.str.contains(r"Withdrew", case=False, na=False)
    out["is_place"] = result_text.str.contains(r"\bPlace\b", case=False, na=False)
    return out


def compute_week_totals(df: pd.DataFrame, score_cols: Sequence[str]) -> pd.DataFrame:
    """Contestant weekly judge totals; all-missing weeks stay missing (not zero)."""
    totals: dict[int, pd.Series] = {}
    max_week = _max_week_from_columns(score_cols)
    for week in range(1, max_week + 1):
        week_cols = _score_columns_for_week(score_cols, week)
        week_scores = df[week_cols]
        total = week_scores.sum(axis=1, skipna=True)
        total = total.mask(week_scores.isna().all(axis=1), np.nan)
        totals[week] = total
    return pd.DataFrame(totals, index=df.index)


def infer_season_lengths(df: pd.DataFrame, week_totals: pd.DataFrame) -> pd.Series:
    """Infer each season's true final show week from positive judge totals.

    ``T_s = max{w : exists i, sum_j X_{i,s,w,j} > 0}``.
    """
    positive_by_week = week_totals.gt(0).groupby(df["season"].to_numpy()).any()

    def last_positive_week(row: pd.Series) -> int | None:
        positive_weeks = [int(w) for w, present in row.items() if bool(present)]
        return max(positive_weeks) if positive_weeks else None

    return positive_by_week.apply(last_positive_week, axis=1).astype("Int64")


def infer_activity_windows(df: pd.DataFrame, week_totals: pd.DataFrame) -> pd.DataFrame:
    """Define the last week each contestant belongs to the weekly competition set.

    ``H_{i,s}`` is the elimination week for regular eliminations, the season length for
    placements, and the last positive-score week for withdrawals.
    """
    out = df.copy()
    season_max_week = infer_season_lengths(out, week_totals)
    out["season_max_week"] = out["season"].map(season_max_week).astype("Int64")

    def last_row_positive(row: pd.Series) -> float:
        positive = row[row > 0]
        return float(positive.index.max()) if len(positive) else np.nan

    out["last_week_positive"] = week_totals.apply(last_row_positive, axis=1)
    out["active_until"] = out["elim_week_result"]

    place_mask = out["active_until"].isna() & out["is_place"]
    out.loc[place_mask, "active_until"] = out.loc[place_mask, "season_max_week"].astype(float)

    withdrew_mask = out["is_withdrew"]
    out.loc[withdrew_mask, "active_until"] = out.loc[withdrew_mask, "last_week_positive"]

    out["active_until"] = out["active_until"].fillna(out["season_max_week"]).astype(float)
    return out


def clean_structural_zeros(
    df: pd.DataFrame, score_cols: Sequence[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Convert structural score zeros to missing values; audit impossible active zeros.

    Post-elimination zeros are placeholders for "not competing", not real judge scores.
    Active all-zero weeks are also invalid under the DWTS scoring scale.
    """
    out = df.copy()
    audit_rows: list[dict[str, object]] = []
    max_week = _max_week_from_columns(score_cols)

    for week in range(1, max_week + 1):
        week_cols = _score_columns_for_week(score_cols, week)
        week_scores = out[week_cols]
        in_show = week <= out["season_max_week"].astype(float)
        active = (week <= out["active_until"].astype(float)) & in_show
        has_any_recorded = week_scores.notna().any(axis=1)
        all_recorded_zero_or_missing = week_scores.fillna(0).eq(0).all(axis=1)
        impossible_active_zero = active & has_any_recorded & all_recorded_zero_or_missing

        for idx in out.index[impossible_active_zero]:
            audit_rows.append(
                {
                    "season": int(out.at[idx, "season"]),
                    "celebrity_name": out.at[idx, "celebrity_name"],
                    "week": week,
                    "reason": "active_week_all_recorded_scores_zero",
                    "active_until": out.at[idx, "active_until"],
                    "season_max_week": out.at[idx, "season_max_week"],
                    "zero_cells": int(out.loc[idx, week_cols].eq(0).sum()),
                }
            )
        out.loc[impossible_active_zero, week_cols] = np.nan

    for week in range(1, max_week + 1):
        week_cols = _score_columns_for_week(score_cols, week)
        post_active = (week > out["active_until"].astype(float)) & (
            week <= out["season_max_week"].astype(float)
        )
        out.loc[post_active, week_cols] = out.loc[post_active, week_cols].replace(0, np.nan)

    zero_audit = pd.DataFrame(audit_rows)
    return out, zero_audit


def _static_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in _STATIC_COLUMNS if c in df.columns]


# --------------------------------------------------------------------------- #
# Table construction
# --------------------------------------------------------------------------- #
def build_long_judge_table(df_clean: pd.DataFrame, score_cols: Sequence[str]) -> pd.DataFrame:
    """Reshape contestant-season rows to one row per (season, contestant, week, judge)."""
    id_cols = _static_columns(df_clean)
    long = df_clean.melt(
        id_vars=id_cols,
        value_vars=list(score_cols),
        var_name="week_judge",
        value_name="judge_score",
    )
    parsed = long["week_judge"].str.extract(r"week(\d+)_judge(\d+)_score")
    long["week"] = parsed[0].astype(int)
    long["judge"] = parsed[1].astype(int)
    long = long.drop(columns=["week_judge"])
    long["is_show_week"] = long["week"] <= long["season_max_week"].astype(int)
    long["eligible"] = (long["week"] <= long["active_until"].astype(float)) & long["is_show_week"]
    return long.sort_values(["season", "celebrity_name", "week", "judge"]).reset_index(drop=True)


def build_weekly_table(df_clean: pd.DataFrame, df_long_judge: pd.DataFrame) -> pd.DataFrame:
    """Aggregate judge-level rows to contestant-week rows used by later models."""
    show_rows = df_long_judge[df_long_judge["is_show_week"]].copy()
    weekly = show_rows.groupby(["season", "celebrity_name", "week"], as_index=False).agg(
        total_judge_score=("judge_score", "sum"),
        mean_judge_score=("judge_score", "mean"),
        n_judges_scored=("judge_score", lambda x: int(x.notna().sum())),
    )
    weekly.loc[weekly["n_judges_scored"].eq(0), "total_judge_score"] = np.nan
    weekly["performed"] = weekly["total_judge_score"].fillna(0) > 0

    static = df_clean[_static_columns(df_clean)].drop_duplicates(["season", "celebrity_name"])
    weekly = weekly.merge(static, on=["season", "celebrity_name"], how="left")
    weekly["eligible"] = (weekly["week"] <= weekly["active_until"].astype(float)) & (
        weekly["week"] <= weekly["season_max_week"].astype(int)
    )

    valid = weekly["performed"] & weekly["eligible"]
    weekly.loc[valid, "judge_rank"] = (
        weekly.loc[valid]
        .groupby(["season", "week"])["total_judge_score"]
        .rank(method="average", ascending=False)
    )
    denominators = (
        weekly.loc[valid].groupby(["season", "week"])["total_judge_score"].transform("sum")
    )
    weekly.loc[valid, "judge_percent"] = weekly.loc[valid, "total_judge_score"] / denominators

    return weekly.sort_values(["season", "celebrity_name", "week"]).reset_index(drop=True)


def build_roster_table(df_weekly: pd.DataFrame) -> pd.DataFrame:
    """Build the weekly alive-set table ``A_{s,t}`` from eligibility flags."""
    keep_cols = [
        c
        for c in (
            "season",
            "celebrity_name",
            "week",
            "eligible",
            "season_max_week",
            "celebrity_age_during_season",
            "celebrity_industry",
            "celebrity_homestate",
            "celebrity_homecountry_region",
        )
        if c in df_weekly.columns
    ]
    roster = df_weekly[keep_cols].copy().sort_values(["season", "celebrity_name", "week"])
    roster["eligible_next"] = (
        roster.groupby(["season", "celebrity_name"])["eligible"].shift(-1).fillna(False)
    )
    return roster.reset_index(drop=True)


def build_elimination_events(df_roster: pd.DataFrame, season_max_week: pd.Series) -> pd.DataFrame:
    """Group all contestants leaving the alive set at the end of each week."""
    leaving = df_roster[df_roster["eligible"] & ~df_roster["eligible_next"].astype(bool)].copy()
    leaving["elim_at_end_of_week"] = leaving["week"]
    events = leaving.groupby(["season", "elim_at_end_of_week"], as_index=False).agg(
        eliminated=("celebrity_name", lambda values: list(values))
    )
    events["is_final_week_end"] = events["elim_at_end_of_week"].eq(
        events["season"].map(season_max_week).astype(int)
    )
    events["m_elim"] = events["eliminated"].apply(len).astype(int)
    return events.sort_values(["season", "elim_at_end_of_week"]).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Era mapping
# --------------------------------------------------------------------------- #
def assign_official_rule_method(season: int) -> str:
    """Problem-statement era mapping: 1-2 rank, 3-27 percent, 28-34 rank."""
    s = int(season)
    if s <= 2:
        return "rank"
    if 3 <= s <= 27:
        return "percent"
    return "rank"


def legacy_code_era_mapping(season: int) -> str:
    """Mirror the old ``model.py`` direction for audit-only comparison."""
    return "percent" if int(season) >= 28 else "rank"


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
def _validation_row(
    check: str, value: object, expected: object, status: str, notes: str = ""
) -> dict[str, object]:
    return {"check": check, "value": value, "expected": expected, "status": status, "notes": notes}


def validate_preprocessing(raw: pd.DataFrame, tables: PreprocessTables) -> pd.DataFrame:
    """Return a tidy validation report for shapes, semantics, and audit warnings."""
    rows: list[dict[str, object]] = []
    score_cols_raw = identify_score_columns(standardize_columns(raw))
    score_cols_clean = identify_score_columns(tables.clean)

    rows.append(
        _validation_row(
            "raw_shape", tuple(raw.shape), "(421, 53)", "pass" if raw.shape == (421, 53) else "warn"
        )
    )
    rows.append(
        _validation_row(
            "score_columns",
            len(score_cols_raw),
            44,
            "pass" if len(score_cols_raw) == 44 else "fail",
        )
    )
    rows.append(
        _validation_row(
            "max_week_in_columns",
            _max_week_from_columns(score_cols_raw),
            11,
            "pass" if _max_week_from_columns(score_cols_raw) == 11 else "fail",
        )
    )
    rows.append(
        _validation_row(
            "season_count_range",
            f"{tables.clean['season'].nunique()}, {tables.clean['season'].min()}-{tables.clean['season'].max()}",
            "34, 1-34",
            "pass" if tables.clean["season"].nunique() == 34 else "fail",
        )
    )

    dupes = int(tables.clean.duplicated(["season", "celebrity_name"]).sum())
    rows.append(
        _validation_row("duplicate_season_celebrity", dupes, 0, "pass" if dupes == 0 else "fail")
    )

    raw_zeros = int((standardize_columns(raw)[score_cols_raw] == 0).sum().sum())
    raw_missing = int(standardize_columns(raw)[score_cols_raw].isna().sum().sum())
    clean_zeros = int((tables.clean[score_cols_clean] == 0).sum().sum())
    clean_missing = int(tables.clean[score_cols_clean].isna().sum().sum())
    rows.extend(
        [
            _validation_row(
                "raw_literal_zeros", raw_zeros, 4671, "pass" if raw_zeros == 4671 else "warn"
            ),
            _validation_row(
                "raw_missing_scores", raw_missing, 4741, "pass" if raw_missing == 4741 else "warn"
            ),
            _validation_row(
                "clean_literal_zeros", clean_zeros, 0, "pass" if clean_zeros == 0 else "fail"
            ),
            _validation_row(
                "clean_missing_scores",
                clean_missing,
                9412,
                "pass" if clean_missing == 9412 else "warn",
            ),
        ]
    )

    shape_expectations = {
        "clean_shape_rows": (len(tables.clean), 421),
        "long_judge_shape_rows": (len(tables.long_judge), 18524),
        "weekly_shape_rows": (len(tables.weekly), 4199),
        "roster_shape_rows": (len(tables.roster), 4199),
        "elim_events_shape_rows": (len(tables.elim_events), 292),
    }
    for check, (value, expected) in shape_expectations.items():
        rows.append(
            _validation_row(check, value, expected, "pass" if value == expected else "warn")
        )

    withdrew = int(tables.clean["is_withdrew"].sum())
    place_rows = int(tables.clean["is_place"].sum())
    performed = int(tables.weekly["performed"].sum())
    eligible = int(tables.weekly["eligible"].sum())
    final_events = int(tables.elim_events["is_final_week_end"].sum())
    non_final = tables.elim_events[~tables.elim_events["is_final_week_end"]]
    m_dist = non_final["m_elim"].value_counts().sort_index().to_dict()
    rows.extend(
        [
            _validation_row(
                "withdrawn_contestants", withdrew, 10, "pass" if withdrew == 10 else "warn"
            ),
            _validation_row(
                "final_placement_rows", place_rows, 113, "pass" if place_rows == 113 else "warn"
            ),
            _validation_row(
                "performed_weekly_rows", performed, 2777, "pass" if performed == 2777 else "warn"
            ),
            _validation_row(
                "eligible_weekly_rows", eligible, 2780, "pass" if eligible == 2780 else "warn"
            ),
            _validation_row(
                "final_events", final_events, 34, "pass" if final_events == 34 else "fail"
            ),
            _validation_row(
                "non_final_events", len(non_final), 258, "pass" if len(non_final) == 258 else "warn"
            ),
            _validation_row(
                "non_final_m_elim_distribution",
                m_dist,
                "{1: 218, 2: 37, 3: 3}",
                "pass" if m_dist == {1: 218, 2: 37, 3: 3} else "warn",
            ),
        ]
    )

    valid_percent = tables.weekly[tables.weekly["performed"] & tables.weekly["eligible"]].copy()
    percent_sums = valid_percent.groupby(["season", "week"])["judge_percent"].sum()
    max_percent_error = float((percent_sums - 1).abs().max()) if len(percent_sums) else np.nan
    rows.append(
        _validation_row(
            "judge_percent_group_sum_max_abs_error",
            max_percent_error,
            "<= 1e-10",
            "pass" if pd.notna(max_percent_error) and max_percent_error <= 1e-10 else "fail",
        )
    )

    missing_rank = int(tables.weekly.loc[tables.weekly["performed"], "judge_rank"].isna().sum())
    rows.append(
        _validation_row(
            "judge_rank_missing_when_performed",
            missing_rank,
            0,
            "pass" if missing_rank == 0 else "fail",
        )
    )

    final_by_season = (
        tables.elim_events[tables.elim_events["is_final_week_end"]].groupby("season").size()
    )
    final_once = bool(
        (final_by_season == 1).all() and len(final_by_season) == tables.clean["season"].nunique()
    )
    rows.append(
        _validation_row(
            "one_final_event_per_season", bool(final_once), True, "pass" if final_once else "fail"
        )
    )

    multi_non_final = int((non_final["m_elim"] > 1).sum())
    rows.append(
        _validation_row(
            "multi_elimination_events_preserved",
            multi_non_final,
            40,
            "pass" if multi_non_final == 40 else "warn",
        )
    )

    rows.append(
        _validation_row(
            "era_mapping_audit",
            "official: 1-2 rank, 3-27 percent, 28-34 rank",
            "old model.py uses season >= 28 -> percent",
            "warn",
        )
    )

    return pd.DataFrame(rows, columns=["check", "value", "expected", "status", "notes"])


def build_all_tables(raw_csv_path: str | Path) -> PreprocessTables:
    """Run the complete preprocessing pipeline and return every table."""
    raw = load_raw_data(raw_csv_path)
    standardized = standardize_columns(raw)
    parsed = parse_results(standardized)
    score_cols = identify_score_columns(parsed)
    week_totals = compute_week_totals(parsed, score_cols)
    windowed = infer_activity_windows(parsed, week_totals)
    clean, zero_audit = clean_structural_zeros(windowed, score_cols)
    long_judge = build_long_judge_table(clean, score_cols)
    weekly = build_weekly_table(clean, long_judge)
    roster = build_roster_table(weekly)
    season_max_week = (
        clean.drop_duplicates("season").set_index("season")["season_max_week"].astype(int)
    )
    elim_events = build_elimination_events(roster, season_max_week)

    placeholder = PreprocessTables(
        raw=raw,
        clean=clean,
        long_judge=long_judge,
        weekly=weekly,
        roster=roster,
        elim_events=elim_events,
        zero_audit=zero_audit,
        validation=pd.DataFrame(),
    )
    validation = validate_preprocessing(raw, placeholder)
    return PreprocessTables(
        raw=raw,
        clean=clean,
        long_judge=long_judge,
        weekly=weekly,
        roster=roster,
        elim_events=elim_events,
        zero_audit=zero_audit,
        validation=validation,
    )
