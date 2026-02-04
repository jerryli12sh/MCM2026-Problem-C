import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


CASES: List[Tuple[int, str]] = [
    (2, "Jerry Rice"),
    (4, "Billy Ray Cyrus"),
    (11, "Bristol Palin"),
    (27, "Bobby Bones"),
    (27, "Tinashe"),
    (31, "Vinny Guadagnino"),
]


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
        "combined_rank",
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


def _summarize_case(
    df_case: pd.DataFrame, n_sims_by_scheme: Dict[str, int]
) -> pd.DataFrame:
    rows = []
    for scheme, g in df_case.groupby("scheme"):
        n_sims = n_sims_by_scheme.get(scheme, 1)
        by_week = (
            g.groupby("week", as_index=False)
            .agg(
                mean_rank=("combined_rank", "mean"),
                p10=("combined_rank", lambda x: np.quantile(x, 0.1)),
                p90=("combined_rank", lambda x: np.quantile(x, 0.9)),
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
                mean_rank=("combined_rank", "mean"),
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
        "S1": "#1f77b4",
        "S2": "#ff7f0e",
        "S3": "#2ca02c",
    }

    fig, axes = plt.subplots(2, 1, figsize=(7.2, 6.2), sharex=True)

    ax0 = axes[0]
    for scheme in schemes:
        g = summary[summary["scheme"] == scheme].sort_values("week")
        ax0.plot(g["week"], g["mean_rank"], label=scheme, color=colors.get(scheme))
        ax0.fill_between(
            g["week"], g["p10"], g["p90"], color=colors.get(scheme), alpha=0.15
        )
    ax0.set_title(f"{name} (Season {season}) - Simulated Rank Trend")
    ax0.set_ylabel("Combined Rank (lower is better)")
    ax0.invert_yaxis()
    ax0.legend(frameon=False, ncol=len(schemes))
    ax0.grid(alpha=0.2)

    ax1 = axes[1]
    for scheme in schemes:
        g = summary[summary["scheme"] == scheme].sort_values("week")
        ax1.plot(g["week"], g["alive_rate"], label=scheme, color=colors.get(scheme))
    ax1.set_ylabel("Alive Rate")
    ax1.set_xlabel("Week")
    ax1.set_ylim(-0.02, 1.02)
    ax1.grid(alpha=0.2)

    fname = f"fig_sim_trend_{season}_{_sanitize_filename(name)}.png"
    fig.tight_layout()
    fig.savefig(out_dir / fname, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Plot simulated rank trends for controversy cases."
    )
    parser.add_argument(
        "--sim-detail",
        default="sim_results/sim_detail.csv",
        help="Path to sim_detail.csv",
    )
    parser.add_argument(
        "--out-dir", default="sim_results", help="Directory to save figures"
    )
    args = parser.parse_args()

    sim_path = Path(args.sim_detail)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = _load_sim_detail(sim_path)
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
