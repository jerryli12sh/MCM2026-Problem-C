"""Named controversy case studies for both mechanism simulators.

Exact ports of the two legacy case scripts ``../src/sim_rank_trend_cases.py``
(V1, schemes S1/S2/S3, paper Table-1 |d|/Flip context) and
``../src/sim_rank_trend_cases_2.py`` (V2, schemes V4/V5, paper P-085
"Bobby Bones champion -> 6th" / "Tinashe week-7 -> week-8" claims).

The two scripts read different sim_detail columns and produce different weekly
summaries:

- V1: ``combined_rank`` -> ``mean_rank/p10/p90`` (ranks), ``alive_rate``;
  plots 2-panel rank + survival; files ``fig_sim_trend_<season>_<name>.png``.
- V2: ``score_S`` -> per-(scheme,sim,season,week) rank ``rank_S``,
  ``bottom2_rate`` (Pr(rank_S<=2)), ``elim_rate``; single-panel plot with
  gray bottom-2 shading and elimination markers; files
  ``fig_sim2_trend_<season>_<name>.png``.

Plotting imports matplotlib lazily so the data transforms stay import-safe for
testing; both functions are pure renderers over a saved detail frame.  The case
set (``CONTROVERSY_CASES``) is shared with :mod:`.features`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .features import CONTROVERSY_CASES

# V2 plot: weeks shaded when the max-across-schemes bottom-2 rate crosses this.
BOTTOM2_THRESHOLD = 0.5
# V2 plot: first week with elim_rate at/above this splits solid/dashed lines.
ELIM_THRESHOLD = 0.5

V1_COLORS = {"S1": "#1f77b4", "S2": "#ff7f0e", "S3": "#2ca02c"}
V2_COLORS = {"V4": "#d62728", "V5": "#2ca02c"}


def sanitize_filename(s: str) -> str:
    """Port of ``_sanitize_filename`` (space/separators/quote -> safe chars)."""
    return (
        s.replace(" ", "_").replace("/", "_").replace("\\", "_").replace("'", "").replace('"', "")
    )


def load_sim_detail(path: str | Path, score_column: str) -> pd.DataFrame:
    """Read a sim_detail frame, requiring the case script's columns.

    ``score_column`` is ``"combined_rank"`` for the V1 script and ``"score_S"``
    for the V2 script; both scripts additionally need scheme/sim/season/week/
    celebrity_name/eliminated_this_week.
    """
    df = pd.read_csv(path)
    required = {
        "scheme",
        "sim",
        "season",
        "week",
        "celebrity_name",
        score_column,
        "eliminated_this_week",
    }
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"sim_detail.csv missing columns: {sorted(missing)}")
    return df


def n_sims_by_scheme(df: pd.DataFrame) -> dict[str, int]:
    """Port of ``_get_n_sims``: distinct sim ids per scheme."""
    return {scheme: int(g["sim"].nunique()) for scheme, g in df.groupby("scheme")}


def filter_cases(
    df: pd.DataFrame,
    cases: tuple[tuple[int, str], ...] = CONTROVERSY_CASES,
) -> list[tuple[int, str, pd.DataFrame]]:
    """Split the detail frame per named case (season, celebrity_name)."""
    out = []
    for season, name in cases:
        df_case = df[(df["season"] == season) & (df["celebrity_name"] == name)].copy()
        if df_case.empty:
            continue
        out.append((season, name, df_case))
    return out


# --- V1 case pipeline (combined_rank based) --------------------------------


def summarize_case_rank(df_case: pd.DataFrame, n_sims: int) -> pd.DataFrame:
    """Port of V1 ``_summarize_case``: weekly rank trend + alive rate."""
    rows = []
    for scheme, g in df_case.groupby("scheme"):
        by_week = g.groupby("week", as_index=False).agg(
            mean_rank=("combined_rank", "mean"),
            p10=("combined_rank", lambda x: np.quantile(x, 0.1)),
            p90=("combined_rank", lambda x: np.quantile(x, 0.9)),
            alive_cnt=("sim", "count"),
        )
        by_week["alive_rate"] = by_week["alive_cnt"] / max(n_sims, 1)
        by_week["scheme"] = scheme
        rows.append(by_week)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def case_summary_table_rank(df_case: pd.DataFrame, n_sims: int) -> pd.DataFrame:
    """Port of V1 ``_case_summary_table``: one row per scheme."""
    rows = []
    max_week = int(df_case["week"].max())
    for scheme, g in df_case.groupby("scheme"):
        by_week = g.groupby("week", as_index=False).agg(
            mean_rank=("combined_rank", "mean"),
            alive_cnt=("sim", "count"),
        )
        by_week["alive_rate"] = by_week["alive_cnt"] / max(n_sims, 1)
        final_alive = (
            by_week.loc[by_week["week"] == max_week, "alive_rate"].iloc[0]
            if (by_week["week"] == max_week).any()
            else 0.0
        )
        rows.append(
            {
                "scheme": scheme,
                "mean_rank": float(by_week["mean_rank"].mean()),
                "mean_alive_rate": float(by_week["alive_rate"].mean()),
                "final_alive_rate": float(final_alive),
            }
        )
    return pd.DataFrame(rows)


# --- V2 case pipeline (score_S / rank_S based) -----------------------------


def add_week_ranks(df_case: pd.DataFrame) -> pd.DataFrame:
    """Port of V2 ``_add_week_ranks``: rank of score_S within (scheme,sim,season,week)."""
    df_case = df_case.copy()
    df_case["rank_S"] = df_case.groupby(["scheme", "sim", "season", "week"])["score_S"].rank(
        ascending=False, method="average"
    )
    return df_case


def summarize_case_score(df_case: pd.DataFrame, n_sims: int) -> pd.DataFrame:
    """Port of V2 ``_summarize_case``: rank_S trend, bottom-2 and elim rates."""
    rows = []
    for scheme, g in df_case.groupby("scheme"):
        by_week = g.groupby("week", as_index=False).agg(
            mean_rank=("rank_S", "mean"),
            p10=("rank_S", lambda x: np.quantile(x, 0.1)),
            p90=("rank_S", lambda x: np.quantile(x, 0.9)),
            bottom2_rate=("rank_S", lambda x: float(np.mean(x <= 2))),
            elim_rate=("eliminated_this_week", "mean"),
            alive_cnt=("sim", "count"),
        )
        by_week["alive_rate"] = by_week["alive_cnt"] / max(n_sims, 1)
        by_week["scheme"] = scheme
        rows.append(by_week)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def case_summary_table_score(df_case: pd.DataFrame, n_sims: int) -> pd.DataFrame:
    """Port of V2 ``_case_summary_table``: rank_S summary + bottom2/elim means."""
    rows = []
    max_week = int(df_case["week"].max())
    for scheme, g in df_case.groupby("scheme"):
        by_week = g.groupby("week", as_index=False).agg(
            mean_rank=("rank_S", "mean"),
            alive_cnt=("sim", "count"),
        )
        by_week["alive_rate"] = by_week["alive_cnt"] / max(n_sims, 1)
        final_alive = (
            by_week.loc[by_week["week"] == max_week, "alive_rate"].iloc[0]
            if (by_week["week"] == max_week).any()
            else 0.0
        )
        rows.append(
            {
                "scheme": scheme,
                "mean_rank": float(by_week["mean_rank"].mean()),
                "mean_alive_rate": float(by_week["alive_rate"].mean()),
                "final_alive_rate": float(final_alive),
                "mean_bottom2_rate": float(np.mean(g["rank_S"] <= 2)),
                "mean_elim_rate": float(np.mean(g["eliminated_this_week"])),
            }
        )
    return pd.DataFrame(rows)


# --- Plotting (matplotlib imported lazily) ----------------------------------


def build_case_summary(
    df: pd.DataFrame,
    detail_kind: str,
    cases: tuple[tuple[int, str], ...] = CONTROVERSY_CASES,
) -> pd.DataFrame:
    """Build the per-case ``sim_case_summary`` table for either simulator.

    ``detail_kind`` is ``"V1"`` (requires ``combined_rank``) or ``"V2"``
    (requires ``score_S``, adds ``rank_S``).  Returns one row per
    (season, celebrity_name, scheme); inserts season/celebrity_name at front
    exactly like the legacy scripts.
    """
    if detail_kind == "V2":
        # Legacy ``sim_rank_trend_cases_2.main`` ranks the *full* detail frame
        # once (within scheme/sim/season/week) and only then filters to the
        # case; ranking the per-case slice instead would collapse ``rank_S``
        # to 1 for every single-contestant group.
        df = add_week_ranks(df)
    n_sims_map = n_sims_by_scheme(df)
    rows = []
    for season, name, df_case in filter_cases(df, cases):
        n_sims = n_sims_map.get(df_case["scheme"].iloc[0], 1)
        if detail_kind == "V1":
            tbl = case_summary_table_rank(df_case, n_sims)
        else:
            tbl = case_summary_table_score(df_case, n_sims)
        tbl.insert(0, "celebrity_name", name)
        tbl.insert(0, "season", season)
        rows.append(tbl)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def plot_case_rank_weekly(
    summary: pd.DataFrame,
    season: int,
    name: str,
    out_path: str | Path,
) -> None:
    """Port of V1 ``_plot_case`` from a weekly summary table: rank + survival."""
    import matplotlib.pyplot as plt

    if summary.empty:
        return
    schemes = sorted(summary["scheme"].unique())

    fig, axes = plt.subplots(2, 1, figsize=(7.2, 6.2), sharex=True)
    ax0 = axes[0]
    for scheme in schemes:
        g = summary[summary["scheme"] == scheme].sort_values("week")
        ax0.plot(g["week"], g["mean_rank"], label=scheme, color=V1_COLORS.get(scheme))
        ax0.fill_between(g["week"], g["p10"], g["p90"], color=V1_COLORS.get(scheme), alpha=0.15)
    ax0.set_title(f"{name} (Season {season}) - Simulated Rank Trend")
    ax0.set_ylabel("Combined Rank (lower is better)")
    ax0.invert_yaxis()
    ax0.legend(frameon=False, ncol=len(schemes))
    ax0.grid(alpha=0.2)

    ax1 = axes[1]
    for scheme in schemes:
        g = summary[summary["scheme"] == scheme].sort_values("week")
        ax1.plot(g["week"], g["alive_rate"], label=scheme, color=V1_COLORS.get(scheme))
    ax1.set_ylabel("Alive Rate")
    ax1.set_xlabel("Week")
    ax1.set_ylim(-0.02, 1.02)
    ax1.grid(alpha=0.2)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_case_rank(
    df_case: pd.DataFrame,
    n_sims: int,
    season: int,
    name: str,
    out_path: str | Path,
) -> None:
    """Port of V1 ``_plot_case`` from a detail frame (delegates to the weekly plot)."""
    plot_case_rank_weekly(summarize_case_rank(df_case, n_sims), season, name, out_path)


def plot_case_score_weekly(
    summary: pd.DataFrame,
    season: int,
    name: str,
    out_path: str | Path,
) -> None:
    """Port of V2 ``_plot_case`` from a weekly summary table: rank + elim marks."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    if summary.empty:
        return
    schemes = sorted(summary["scheme"].unique())

    fig, ax0 = plt.subplots(1, 1, figsize=(7.6, 4.8))
    week_min = int(summary["week"].min())
    week_max = int(summary["week"].max())
    b2_weeks = summary.groupby("week")["bottom2_rate"].max().reset_index()
    for _, row in b2_weeks.iterrows():
        if row["bottom2_rate"] >= BOTTOM2_THRESHOLD:
            w = int(row["week"])
            ax0.axvspan(w - 0.5, w + 0.5, color="#d9d9d9", alpha=0.25, zorder=0)

    for scheme in schemes:
        g = summary[summary["scheme"] == scheme].sort_values("week").copy()
        elim_weeks = g.loc[g["elim_rate"] >= ELIM_THRESHOLD, "week"]
        if not elim_weeks.empty:
            elim_week = int(elim_weeks.iloc[0])
            g_pre = g[g["week"] <= elim_week]
            g_post = g[g["week"] >= elim_week]
        else:
            g_pre = g
            g_post = g.iloc[0:0]

        ax0.plot(
            g_pre["week"],
            g_pre["mean_rank"],
            label=scheme,
            color=V2_COLORS.get(scheme),
            linewidth=2.9,
            marker="o",
            markersize=3.5,
            markerfacecolor=V2_COLORS.get(scheme),
            markeredgecolor=V2_COLORS.get(scheme),
        )
        if not g_post.empty:
            ax0.plot(
                g_post["week"],
                g_post["mean_rank"],
                color=V2_COLORS.get(scheme),
                linestyle="--",
                alpha=0.5,
                linewidth=2.0,
            )
        ax0.fill_between(g["week"], g["p10"], g["p90"], color=V2_COLORS.get(scheme), alpha=0.15)
        b2 = g[g["bottom2_rate"] >= BOTTOM2_THRESHOLD]
        if not b2.empty:
            ax0.scatter(
                b2["week"],
                b2["mean_rank"],
                s=90,
                facecolors="none",
                edgecolors=V2_COLORS.get(scheme),
                linewidths=2,
                zorder=5,
            )
        if not elim_weeks.empty:
            row = g[g["week"] == elim_week].iloc[0]
            ax0.scatter(
                [row["week"]], [row["mean_rank"]], s=28, color=V2_COLORS.get(scheme), zorder=6
            )
            ax0.scatter(
                [row["week"]],
                [row["mean_rank"]],
                s=130,
                facecolors="none",
                edgecolors="black",
                linewidths=2,
                zorder=7,
            )
            ax0.annotate(
                "eliminate",
                xy=(row["week"], row["mean_rank"]),
                xytext=(6, -8),
                textcoords="offset points",
                fontsize=8,
                color="black",
            )

    ax0.set_title(f"{name} (Season {season}) - Simulated Rank Trend")
    ax0.set_ylabel("Rank (lower is better)")
    ax0.invert_yaxis()
    ax0.set_xlim(week_min - 0.2, week_max + 0.2)
    legend_handles = [
        Line2D([0], [0], color=V2_COLORS.get("V4"), lw=2, label="V4"),
        Line2D([0], [0], color=V2_COLORS.get("V5"), lw=2, label="V5"),
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor="none",
            markeredgecolor="gray",
            markersize=8,
            label="Bottom2 (>=50%)",
        ),
    ]
    ax0.legend(handles=legend_handles, frameon=False, loc="lower right")
    ax0.grid(alpha=0.2)
    ax0.set_xlabel("Week")

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_case_score(
    df_case: pd.DataFrame,
    n_sims: int,
    season: int,
    name: str,
    out_path: str | Path,
) -> None:
    """Port of V2 ``_plot_case`` from a detail frame (delegates to the weekly plot)."""
    plot_case_score_weekly(summarize_case_score(df_case, n_sims), season, name, out_path)


def build_case_weekly(
    df: pd.DataFrame,
    detail_kind: str,
    cases: tuple[tuple[int, str], ...] = CONTROVERSY_CASES,
) -> pd.DataFrame:
    """Long per-case weekly summary table (one row per season, name, scheme, week).

    Saves the small tables the trend figures render from, so the plot script
    never needs the multi-hundred-MB detail frame.  V1 rows carry
    ``mean_rank/p10/p90/alive_rate``; V2 rows additionally carry
    ``bottom2_rate/elim_rate``.
    """
    if detail_kind == "V2":
        # Same full-frame-before-filter ranking as the legacy V2 case script
        # (see :func:`build_case_summary`).
        df = add_week_ranks(df)
    n_sims_map = n_sims_by_scheme(df)
    rows = []
    for season, name, df_case in filter_cases(df, cases):
        n_sims = n_sims_map.get(df_case["scheme"].iloc[0], 1)
        if detail_kind == "V1":
            tbl = summarize_case_rank(df_case, n_sims)
        else:
            tbl = summarize_case_score(df_case, n_sims)
        tbl.insert(0, "celebrity_name", name)
        tbl.insert(0, "season", season)
        rows.append(tbl)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def render_case_figures(
    df: pd.DataFrame,
    detail_kind: str,
    out_dir: str | Path,
    *,
    prefix: str = "",
) -> list[str]:
    """Render all named-case figures from a detail frame; return written names.

    ``detail_kind`` selects the V1 (``fig_sim_trend_*``) or V2
    (``fig_sim2_trend_*``) recipe.  ``prefix`` is prepended to every file name —
    the paper embeds the V2 trends as ``8_fig_sim2_trend_*``, so the plot
    script passes ``prefix="8_"`` for V2 to match the ``\\includegraphics``
    references (D-20260901-18).  Matches the legacy scripts' naming otherwise
    (``_sanitize_filename`` applied to the celebrity name).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    n_sims_map = n_sims_by_scheme(df)
    written: list[str] = []
    for season, name, df_case in filter_cases(df):
        n_sims = n_sims_map.get(df_case["scheme"].iloc[0], 1)
        base = (
            f"fig_sim_trend_{season}_{sanitize_filename(name)}.png"
            if detail_kind == "V1"
            else f"fig_sim2_trend_{season}_{sanitize_filename(name)}.png"
        )
        fname = f"{prefix}{base}"
        if detail_kind == "V1":
            plot_case_rank(df_case, n_sims, season, name, out_dir / fname)
        else:
            plot_case_score(df_case, n_sims, season, name, out_dir / fname)
        written.append(fname)
    return written
