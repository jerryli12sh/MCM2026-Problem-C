"""Problem 1 panel construction.

The training unit is a season-week *alive set*: elimination is a comparison among
the contestants alive that week, so every judge signal is normalized within that
set.  This module assembles the contestant-week panel, the single-elimination
training weeks, and the standardized row features used by both tracks.

The construction mirrors ``review/problem1_rebuild/problem1_fan_support.py``
line for line so the rebuilt panel reconciles exactly with the reference outputs.
"""

from __future__ import annotations

import ast
from collections.abc import Iterable
from typing import Any, cast

import numpy as np
import pandas as pd

from dwts_reproduction.preprocess import (
    PreprocessTables,
    assign_official_rule_method,
    legacy_code_era_mapping,
)

REQUIRED_PANEL_COLUMNS = [
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
]


# --------------------------------------------------------------------------- #
# Era mapping
# --------------------------------------------------------------------------- #
def assign_era(season: int, era_mode: str) -> str:
    """Return the judge-aggregation era for a season under an audited mapping."""
    if era_mode == "legacy":
        # Historical downstream code used this mapping; kept only to reproduce old
        # outputs (see D-20260901-01).  Pipeline metadata records a warning.
        return legacy_code_era_mapping(season)
    if era_mode == "official":
        return assign_official_rule_method(season)
    raise ValueError("era_mode must be 'legacy' or 'official'.")


def _softmax_np(x: np.ndarray) -> np.ndarray:
    z = np.asarray(x, dtype=float)
    z = z - np.nanmax(z)
    e = np.exp(z)
    return np.asarray(e / e.sum())


# --------------------------------------------------------------------------- #
# Judge signals
# --------------------------------------------------------------------------- #
def _parse_eliminated_list(value: object) -> list[Any]:
    """Normalize an ``eliminated`` cell (already a list, or a literal-eval string)."""
    if isinstance(value, str):
        parsed: list[Any] = ast.literal_eval(value)
        return parsed
    return list(cast(Iterable[Any], value))


def build_elim_long(df_elim_events: pd.DataFrame) -> pd.DataFrame:
    """Explode elimination events into one row per eliminated contestant."""
    tmp = df_elim_events.copy()
    tmp["eliminated_list"] = tmp["eliminated"].apply(_parse_eliminated_list)
    out = (
        tmp.rename(columns={"elim_at_end_of_week": "week"})
        .explode("eliminated_list")
        .rename(columns={"eliminated_list": "celebrity_name"})[
            ["season", "week", "celebrity_name", "is_final_week_end"]
        ]
    )
    out["elim_this_week_end"] = True
    return out


def build_judge_percent(df_weekly: pd.DataFrame, base: pd.DataFrame) -> pd.DataFrame:
    """Percentage-era judge shares on the alive-set simplex.

    ``df_weekly.judge_percent`` is already the alive-set normalization from
    preprocessing; when it is absent the total judge score is re-normalized here.
    """
    if "judge_percent" in df_weekly.columns and df_weekly["judge_percent"].notna().any():
        return df_weekly[["season", "week", "celebrity_name", "judge_percent"]].copy()

    score_col = "total_judge_score" if "total_judge_score" in df_weekly.columns else "judge_total"
    if score_col not in df_weekly.columns:
        raise KeyError("df_weekly must contain judge_percent or a judge total column.")
    w = df_weekly.merge(
        base[["season", "week", "celebrity_name", "alive"]],
        on=["season", "week", "celebrity_name"],
        how="left",
    )
    w = w[(w["alive"]) & (w[score_col].notna())].copy()
    denom = w.groupby(["season", "week"])[score_col].transform("sum")
    w["judge_percent"] = np.where(denom > 0, w[score_col] / denom, np.nan)
    return w[["season", "week", "celebrity_name", "judge_percent"]].copy()


def build_judge_rank_share(df_long_judge: pd.DataFrame, base: pd.DataFrame) -> pd.DataFrame:
    """Rank-era judge shares: softmax of negative summed judge ranks per alive set."""
    dj = df_long_judge.copy()
    if "eligible" in dj.columns:
        dj = dj[dj["eligible"].astype(bool)].copy()
    if "is_show_week" in dj.columns:
        dj = dj[dj["is_show_week"].astype(bool)].copy()
    dj = dj[dj["judge_score"].notna()].copy()

    dj["judge_rank"] = dj.groupby(["season", "week", "judge"])["judge_score"].rank(
        ascending=False, method="average"
    )
    rank_sum = (
        dj.groupby(["season", "week", "celebrity_name"], as_index=False)
        .agg(rank_sum=("judge_rank", "sum"), n_judges=("judge_rank", "count"))
        .merge(
            base[["season", "week", "celebrity_name", "alive"]],
            on=["season", "week", "celebrity_name"],
            how="left",
        )
    )
    rank_sum = rank_sum[rank_sum["alive"].astype(bool)].copy()
    rank_sum["rank_score"] = -rank_sum["rank_sum"].astype(float)
    rank_sum["judge_rank_share"] = rank_sum.groupby(["season", "week"])["rank_score"].transform(
        lambda s: _softmax_np(s.to_numpy())
    )
    return rank_sum[["season", "week", "celebrity_name", "judge_rank_share"]].copy()


# --------------------------------------------------------------------------- #
# Panel assembly
# --------------------------------------------------------------------------- #
def build_problem1_panel(
    tables: PreprocessTables,
    era_mode: str,
    warnings: list[str] | None = None,
) -> pd.DataFrame:
    """Construct the contestant-week panel used by Problem 1.

    Args:
        tables: The :class:`PreprocessTables` instance from preprocessing.
        era_mode: ``"legacy"`` or ``"official"`` era mapping.
        warnings: Optional list to which audit warnings are appended.

    Returns:
        The contestant-week panel; every judge metric is normalized within its
        season-week alive set.
    """
    warnings = [] if warnings is None else warnings
    if era_mode == "legacy":
        warnings.append(
            "era_mode='legacy' uses old downstream behavior: seasons >= 28 are treated as "
            "percent and earlier seasons as rank. This conflicts with the official "
            "problem-statement mapping."
        )

    base = tables.roster.copy()
    elim_long = build_elim_long(tables.elim_events)
    base = base.merge(elim_long, on=["season", "week", "celebrity_name"], how="left")
    base["elim_this_week_end"] = base["elim_this_week_end"].fillna(False).astype(bool)
    base["alive"] = base["eligible"].astype(bool)
    max_week = base.loc[base["alive"]].groupby("season")["week"].max()
    base["max_week"] = base["season"].map(max_week).astype(int)
    base["is_final_week"] = base["week"].eq(base["max_week"])
    base["age"] = pd.to_numeric(base["celebrity_age_during_season"], errors="coerce")

    judge_percent = build_judge_percent(tables.weekly, base)
    judge_rank_share = build_judge_rank_share(tables.long_judge, base)
    panel = (
        base.merge(judge_percent, on=["season", "week", "celebrity_name"], how="left")
        .merge(judge_rank_share, on=["season", "week", "celebrity_name"], how="left")
        .copy()
    )
    panel["era"] = panel["season"].map(lambda s: assign_era(int(s), era_mode))
    panel["j_metric"] = np.where(
        panel["era"].eq("percent"), panel["judge_percent"], panel["judge_rank_share"]
    )

    panel = panel[
        REQUIRED_PANEL_COLUMNS + [c for c in panel.columns if c not in REQUIRED_PANEL_COLUMNS]
    ].copy()
    validate_panel(panel, era_mode, warnings)
    return panel


def validate_panel(
    panel: pd.DataFrame, era_mode: str, warnings: list[str] | None = None
) -> list[str]:
    """Return (and optionally append) audit warnings about the assembled panel."""
    warnings = [] if warnings is None else warnings
    alive = panel[panel["alive"]].copy()
    if not (4000 <= len(panel) <= 4400):
        warnings.append(f"Panel has {len(panel)} rows; expected around 4199 from preprocessing.")

    train_like = alive[~alive["is_final_week"]]
    missing_j = int(train_like["j_metric"].isna().sum())
    if missing_j:
        warnings.append(f"{missing_j} alive non-final rows have missing j_metric.")

    pct_sum = (
        alive.dropna(subset=["judge_percent"]).groupby(["season", "week"])["judge_percent"].sum()
    )
    if not pct_sum.empty:
        max_err = float((pct_sum - 1.0).abs().max())
        if max_err > 1e-8:
            warnings.append(f"judge_percent alive-set sums deviate from 1 by up to {max_err:.3g}.")

    rank_sum = (
        alive.dropna(subset=["judge_rank_share"])
        .groupby(["season", "week"])["judge_rank_share"]
        .sum()
    )
    if not rank_sum.empty:
        max_err = float((rank_sum - 1.0).abs().max())
        if max_err > 1e-8:
            warnings.append(
                f"judge_rank_share alive-set sums deviate from 1 by up to {max_err:.3g}."
            )
    return warnings


# --------------------------------------------------------------------------- #
# Training weeks and features
# --------------------------------------------------------------------------- #
def build_train_weeks(panel: pd.DataFrame) -> pd.DataFrame:
    """Return non-final alive sets with exactly one end-of-week eliminatee."""
    alive = panel[panel["alive"]].copy()
    grouped = alive.groupby(["season", "week"])
    train_weeks = (
        grouped.agg(
            elim_cnt=("elim_this_week_end", "sum"),
            alive_n=("celebrity_name", "size"),
            max_week=("max_week", "first"),
        )
        .reset_index()
        .query("elim_cnt == 1 and week < max_week")
        .copy()
    )
    eliminatees = (
        alive[alive["elim_this_week_end"]]
        .groupby(["season", "week"])["celebrity_name"]
        .first()
        .rename("true_eliminatee")
        .reset_index()
    )
    return train_weeks.merge(eliminatees, on=["season", "week"], how="left").sort_values(
        ["season", "week"]
    )


def build_feature_frame(
    panel: pd.DataFrame, train_weeks: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build standardized row features and contestant-season indices for training.

    Returns ``(df_feat, meta)`` where ``meta`` carries the standardization moments,
    the ``(season, celebrity)`` index map, and the feature column names needed by
    inference and serialization.
    """
    keys = train_weeks[["season", "week"]]
    df_feat = panel[panel["alive"]].merge(keys, on=["season", "week"], how="inner").copy()
    df_feat["j_metric"] = pd.to_numeric(df_feat["j_metric"], errors="coerce")
    if df_feat["j_metric"].isna().any():
        bad = df_feat.loc[df_feat["j_metric"].isna(), ["season", "week", "celebrity_name"]]
        raise ValueError(
            f"j_metric has NaN in training rows. Examples: {bad.head().to_dict('records')}"
        )

    jm_mean = float(df_feat["j_metric"].mean())
    jm_std = float(df_feat["j_metric"].std(ddof=0) + 1e-12)
    df_feat["j_metric_z"] = (df_feat["j_metric"] - jm_mean) / jm_std

    df_feat["age"] = pd.to_numeric(df_feat["age"], errors="coerce")
    use_age = bool(df_feat["age"].notna().any())
    if use_age:
        age_mean = float(df_feat["age"].mean())
        age_std = float(df_feat["age"].std(ddof=0) + 1e-12)
        df_feat["age_z"] = (df_feat["age"].fillna(age_mean) - age_mean) / age_std
    else:
        age_mean = None
        age_std = None
        df_feat["age_z"] = 0.0

    df_feat["era_is_percent"] = df_feat["era"].eq("percent").astype(float)

    all_alive = panel[panel["alive"]].copy()
    all_alive["_cs_key"] = list(
        zip(
            all_alive["season"].astype(int),
            all_alive["celebrity_name"].astype(str),
            strict=True,
        )
    )
    cs_levels = sorted(all_alive["_cs_key"].unique().tolist())
    cs2idx = {key: i for i, key in enumerate(cs_levels)}
    df_feat["_cs_key"] = list(
        zip(df_feat["season"].astype(int), df_feat["celebrity_name"].astype(str), strict=True)
    )
    df_feat["cs_idx"] = df_feat["_cs_key"].map(cs2idx).astype(int)

    X_cols = ["j_metric_z", "age_z", "era_is_percent"]
    meta: dict[str, Any] = {
        "jm_mean": jm_mean,
        "jm_std": jm_std,
        "age_mean": age_mean,
        "age_std": age_std,
        "use_age": use_age,
        "X_cols": X_cols,
        "cs2idx": cs2idx,
        "n_cs": len(cs2idx),
    }
    return df_feat, meta
