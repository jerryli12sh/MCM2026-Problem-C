import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


CASES: List[Tuple[int, str]] = [
    (2, "Jerry Rice"),
    (4, "Billy Ray Cyrus"),
    (11, "Bristol Palin"),
    (27, "Bobby Bones"),
    (27, "Tinashe"),
    (31, "Vinny Guadagnino"),
]


BOTTOM2_THRESHOLD = 0.5
ELIM_THRESHOLD = 0.5


def _sanitize_filename(s: str) -> str:
    return (
        s.replace(" ", "_")
        .replace("/", "_")
        .replace("\\", "_")
        .replace("'", "")
        .replace('"', "")
    )


def _load_sim_detail(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {
        "scheme",
        "sim",
        "season",
        "week",
        "celebrity_name",
        "score_S",
        "eliminated_this_week",
    }
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"sim_detail.csv missing columns: {sorted(missing)}")
    return df


def _get_n_sims(df: pd.DataFrame) -> Dict[str, int]:
    out = {}
    for scheme, g in df.groupby("scheme"):
        out[scheme] = int(g["sim"].nunique())
    return out


def _add_week_ranks(df_case: pd.DataFrame) -> pd.DataFrame:
    df_case = df_case.copy()
    df_case["rank_S"] = (
        df_case.groupby(["scheme", "sim", "season", "week"])["score_S"]
        .rank(ascending=False, method="average")
    )
    return df_case


def _summarize_case(
    df_case: pd.DataFrame, n_sims_by_scheme: Dict[str, int]
) -> pd.DataFrame:
    rows = []
    for scheme, g in df_case.groupby("scheme"):
        n_sims = n_sims_by_scheme.get(scheme, 1)
        by_week = (
            g.groupby("week", as_index=False)
            .agg(
                mean_rank=("rank_S", "mean"),
                p10=("rank_S", lambda x: np.quantile(x, 0.1)),
                p90=("rank_S", lambda x: np.quantile(x, 0.9)),
                bottom2_rate=("rank_S", lambda x: float(np.mean(x <= 2))),
                elim_rate=("eliminated_this_week", "mean"),
                alive_cnt=("sim", "count"),
            )
        )
        by_week["alive_rate"] = by_week["alive_cnt"] / max(n_sims, 1)
        by_week["scheme"] = scheme
        rows.append(by_week)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def _case_summary_table(df_case: pd.DataFrame, n_sims_by_scheme: Dict[str, int]) -> pd.DataFrame:
    rows = []
    max_week = int(df_case["week"].max())
    for scheme, g in df_case.groupby("scheme"):
        n_sims = n_sims_by_scheme.get(scheme, 1)
        by_week = (
            g.groupby("week", as_index=False)
            .agg(
                mean_rank=("rank_S", "mean"),
                alive_cnt=("sim", "count"),
            )
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


def _plot_case(
    df_case: pd.DataFrame,
    n_sims_by_scheme: Dict[str, int],
    season: int,
    name: str,
    out_dir: Path,
):
    summary = _summarize_case(df_case, n_sims_by_scheme)
    if summary.empty:
        return

    schemes = sorted(summary["scheme"].unique())
    colors = {
        "V4": "#d62728",
        "V5": "#2ca02c",
    }

    fig, ax0 = plt.subplots(1, 1, figsize=(7.6, 4.8))
    week_min = int(summary["week"].min())
    week_max = int(summary["week"].max())
    b2_weeks = (
        summary.groupby("week")["bottom2_rate"].max().reset_index()
    )
    for _, row in b2_weeks.iterrows():
        if row["bottom2_rate"] >= BOTTOM2_THRESHOLD:
            w = int(row["week"])
            ax0.axvspan(
                w - 0.5,
                w + 0.5,
                color="#d9d9d9",
                alpha=0.25,
                zorder=0,
            )

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
            color=colors.get(scheme),
            linewidth=2.9,
            marker="o",
            markersize=3.5,
            markerfacecolor=colors.get(scheme),
            markeredgecolor=colors.get(scheme),
        )
        if not g_post.empty:
            ax0.plot(
                g_post["week"],
                g_post["mean_rank"],
                color=colors.get(scheme),
                linestyle="--",
                alpha=0.5,
                linewidth=2.0,
            )
        ax0.fill_between(
            g["week"], g["p10"], g["p90"], color=colors.get(scheme), alpha=0.15
        )
        b2 = g[g["bottom2_rate"] >= BOTTOM2_THRESHOLD]
        if not b2.empty:
            ax0.scatter(
                b2["week"],
                b2["mean_rank"],
                s=90,
                facecolors="none",
                edgecolors=colors.get(scheme),
                linewidths=2,
                zorder=5,
            )
        if not elim_weeks.empty:
            row = g[g["week"] == elim_week].iloc[0]
            ax0.scatter(
                [row["week"]],
                [row["mean_rank"]],
                s=28,
                color=colors.get(scheme),
                zorder=6,
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
        Line2D([0], [0], color=colors.get("V4"), lw=2, label="V4"),
        Line2D([0], [0], color=colors.get("V5"), lw=2, label="V5"),
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

    fname = f"fig_sim2_trend_{season}_{_sanitize_filename(name)}.png"
    fig.tight_layout()
    fig.savefig(out_dir / fname, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Plot simulated score trends for controversy cases (V4/V5)."
    )
    parser.add_argument(
        "--sim-detail",
        default="sim_results_v5/sim_detail.csv",
        help="Path to sim_detail.csv from season_simulator2.py",
    )
    parser.add_argument(
        "--out-dir", default="sim_results_v5", help="Directory to save figures"
    )
    args = parser.parse_args()

    sim_path = Path(args.sim_detail)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = _load_sim_detail(sim_path)
    df = _add_week_ranks(df)
    n_sims_by_scheme = _get_n_sims(df)

    summary_rows = []
    for season, name in CASES:
        df_case = df[(df["season"] == season) & (df["celebrity_name"] == name)].copy()
        if df_case.empty:
            continue
        _plot_case(df_case, n_sims_by_scheme, season, name, out_dir)
        tbl = _case_summary_table(df_case, n_sims_by_scheme)
        tbl.insert(0, "celebrity_name", name)
        tbl.insert(0, "season", season)
        summary_rows.append(tbl)

    if summary_rows:
        summary = pd.concat(summary_rows, ignore_index=True)
        summary.to_csv(out_dir / "sim_case_summary.csv", index=False)
        print(f"Wrote {len(summary)} rows to {out_dir / 'sim_case_summary.csv'}")
    else:
        print("No case rows found in sim_detail.csv.")


if __name__ == "__main__":
    main()
