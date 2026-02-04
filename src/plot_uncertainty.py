import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def order_contestants(df_season: pd.DataFrame) -> list:
    g = df_season.groupby("celebrity_name", as_index=False).agg(
        mean_p_hat=("p_hat", "mean"),
        exit_week=("exit_week", "max"),
    )
    g = g.sort_values(["exit_week", "mean_p_hat"], ascending=[False, False])
    return g["celebrity_name"].tolist()


def plot_heatmap(df: pd.DataFrame, season: int, metric: str, output: str) -> None:
    df_season = df[df["season"] == season].copy()
    if df_season.empty:
        raise ValueError(f"No data for season {season}")

    order = order_contestants(df_season)

    pivot = (df_season.pivot(index="celebrity_name", columns="week", values=metric)
             .reindex(index=order))

    fig, ax = plt.subplots(figsize=(12, max(4, 0.35 * len(order))))
    im = ax.imshow(pivot.values, aspect="auto", interpolation="nearest")

    ax.set_xticks(np.arange(pivot.shape[1]))
    ax.set_xticklabels(pivot.columns.astype(int))
    ax.set_yticks(np.arange(pivot.shape[0]))
    ax.set_yticklabels(pivot.index)

    ax.set_xlabel("Week")
    ax.set_ylabel("Contestant (alive weeks only)")
    title_metric = "Relative CI Width" if metric == "ci_rel_width" else "CI Width"
    ax.set_title(f"Season {season}: {title_metric} Heatmap")

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(title_metric)

    fig.tight_layout()
    fig.savefig(output, dpi=300)
    print(f"Saved {output}")


def plot_ess(df: pd.DataFrame, season: int, metric: str, output: str) -> None:
    df_season = df[df["season"] == season].copy()
    if df_season.empty:
        raise ValueError(f"No data for season {season}")

    ess_week = (df_season[["season", "week", "ess", "ess_ratio"]]
                .drop_duplicates()
                .sort_values(["season", "week"]))

    y = ess_week[metric].to_numpy()
    x = ess_week["week"].to_numpy()

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(x, y, marker="o", linewidth=1.5)
    ax.set_xlabel("Week")
    ax.set_ylabel(metric)
    ax.set_title(f"Season {season}: {metric} by Week")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output, dpi=300)
    print(f"Saved {output}")


def main():
    parser = argparse.ArgumentParser(description="Plot posterior uncertainty figures.")
    parser.add_argument("--input", type=str, default="posterior_uncertainty.csv")
    parser.add_argument("--season", type=int, default=None)
    parser.add_argument("--metric", type=str, choices=["ci_rel_width", "ci_width"], default="ci_rel_width")
    parser.add_argument("--ess-metric", type=str, choices=["ess", "ess_ratio"], default="ess_ratio")
    parser.add_argument("--heatmap-out", type=str, default="fig2_ci_heatmap.png")
    parser.add_argument("--ess-out", type=str, default="figD1_ess_by_week.png")
    args = parser.parse_args()

    df = pd.read_csv(args.input)

    if args.season is not None:
        plot_heatmap(df, args.season, args.metric, args.heatmap_out)
        plot_ess(df, args.season, args.ess_metric, args.ess_out)
        return

    seasons = sorted(df["season"].dropna().unique().tolist())
    for s in seasons:
        heatmap_out = args.heatmap_out.replace(".png", f"_S{s}.png")
        ess_out = args.ess_out.replace(".png", f"_S{s}.png")
        plot_heatmap(df, int(s), args.metric, heatmap_out)
        plot_ess(df, int(s), args.ess_metric, ess_out)


if __name__ == "__main__":
    main()
