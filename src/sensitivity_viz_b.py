import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    import seaborn as sns
    _HAS_SEABORN = True
except Exception:
    _HAS_SEABORN = False


def load_csv(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    return pd.read_csv(path)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def plot_reversal_by_week(b1_rev: pd.DataFrame, outdir: Path) -> None:
    if b1_rev is None or b1_rev.empty:
        return
    fig, ax = plt.subplots(figsize=(7, 5))
    for cutoff, g in b1_rev.groupby("era_cutoff"):
        tmp = g.groupby("week")["reversal_rate"].mean().reset_index()
        ax.plot(tmp["week"], tmp["reversal_rate"], marker="o", label=f"cutoff={cutoff}")
    ax.set_xlabel("week")
    ax.set_ylabel("reversal_rate")
    ax.set_title("B1: Reversal rate by week (avg across seasons)")
    ax.legend(frameon=False)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(outdir / "B1_reversal_by_week.png", dpi=200)
    plt.close(fig)


def plot_cutoff_fairness(b1_sum: pd.DataFrame, outdir: Path) -> None:
    if b1_sum is None or b1_sum.empty:
        return
    fig, ax = plt.subplots(figsize=(7, 5))
    for scheme, g in b1_sum.groupby("scheme"):
        tmp = g.groupby("era_cutoff")["fairness_score"].mean().reset_index()
        ax.plot(tmp["era_cutoff"], tmp["fairness_score"], marker="o", label=scheme)
    ax.set_xlabel("era_cutoff")
    ax.set_ylabel("fairness_score")
    ax.set_title("B1: Fairness score vs era cutoff")
    ax.legend(frameon=False)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(outdir / "B1_fairness_by_cutoff.png", dpi=200)
    plt.close(fig)


def plot_controversy_bar(df: pd.DataFrame, outdir: Path, name: str) -> None:
    if df is None or df.empty:
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    if "era_cutoff" in df.columns:
        pivot = df.pivot_table(index="celebrity_name", columns="era_cutoff", values="flip_rate", aggfunc="mean")
        if _HAS_SEABORN:
            sns.heatmap(pivot, ax=ax, cmap="viridis", cbar=True)
        else:
            im = ax.imshow(pivot.values, origin="lower", aspect="auto", cmap="viridis")
            fig.colorbar(im, ax=ax)
            ax.set_xticks(range(len(pivot.columns)))
            ax.set_xticklabels([str(x) for x in pivot.columns])
            ax.set_yticks(range(len(pivot.index)))
            ax.set_yticklabels([str(y) for y in pivot.index])
        ax.set_xlabel("era_cutoff")
        ax.set_ylabel("celebrity_name")
        ax.set_title("B1: controversy flip rate")
    else:
        pivot = df.pivot_table(index="celebrity_name", columns="judge_save_rule", values="flip_rate", aggfunc="mean")
        if _HAS_SEABORN:
            sns.heatmap(pivot, ax=ax, cmap="viridis", cbar=True)
        else:
            im = ax.imshow(pivot.values, origin="lower", aspect="auto", cmap="viridis")
            fig.colorbar(im, ax=ax)
            ax.set_xticks(range(len(pivot.columns)))
            ax.set_xticklabels([str(x) for x in pivot.columns])
            ax.set_yticks(range(len(pivot.index)))
            ax.set_yticklabels([str(y) for y in pivot.index])
        ax.set_xlabel("judge_save_rule")
        ax.set_ylabel("celebrity_name")
        ax.set_title("B3: controversy flip rate")
    fig.tight_layout()
    fig.savefig(outdir / name, dpi=200)
    plt.close(fig)


def plot_b2_heatmap(b2_sum: pd.DataFrame, outdir: Path) -> None:
    if b2_sum is None or b2_sum.empty:
        return
    df = b2_sum.copy()
    if not {"wJ", "fan_temp", "fairness_score", "scheme"}.issubset(df.columns):
        return
    best = (
        df.sort_values("fairness_score", ascending=False)
        .groupby(["wJ", "fan_temp"])
        .first()
        .reset_index()
    )
    pivot = best.pivot_table(index="fan_temp", columns="wJ", values="scheme", aggfunc="first")
    fig, ax = plt.subplots(figsize=(6, 5))
    if _HAS_SEABORN:
        sns.heatmap(pivot, ax=ax, cmap="tab20", cbar=False)
    else:
        ax.imshow(pivot.values, origin="lower", aspect="auto")
    ax.set_xlabel("wJ")
    ax.set_ylabel("fan_temp")
    ax.set_title("B2: Robustness map (best scheme)")
    fig.tight_layout()
    fig.savefig(outdir / "B2_robustness_map.png", dpi=200)
    plt.close(fig)


def plot_b2_surface(b2_sum: pd.DataFrame, outdir: Path) -> None:
    if b2_sum is None or b2_sum.empty:
        return
    df = b2_sum.copy()
    fig, ax = plt.subplots(figsize=(7, 5))
    for scheme, g in df.groupby("scheme"):
        tmp = g.groupby(["wJ", "fan_temp"])["fairness_score"].mean().reset_index()
        sc = ax.scatter(tmp["wJ"], tmp["fan_temp"], c=tmp["fairness_score"], label=scheme, cmap="viridis")
    ax.set_xlabel("wJ")
    ax.set_ylabel("fan_temp")
    ax.set_title("B2: fairness score by grid (color)")
    fig.colorbar(sc, ax=ax, label="fairness_score")
    fig.tight_layout()
    fig.savefig(outdir / "B2_fairness_scatter.png", dpi=200)
    plt.close(fig)


def plot_b3_bottom_summary(b3_sum: pd.DataFrame, outdir: Path) -> None:
    if b3_sum is None or b3_sum.empty:
        return
    fig, ax = plt.subplots(figsize=(7, 5))
    for scheme, g in b3_sum.groupby("scheme"):
        tmp = g.groupby(["bottom_k", "judge_save_rule"])["fairness_score"].mean().reset_index()
        tmp["label"] = tmp["bottom_k"].astype(str) + ":" + tmp["judge_save_rule"].astype(str)
        ax.plot(tmp["label"], tmp["fairness_score"], marker="o", label=scheme)
    ax.set_xlabel("bottom_k:judge_rule")
    ax.set_ylabel("fairness_score")
    ax.set_title("B3: fairness score by bottom-k and judge-save rule")
    ax.legend(frameon=False)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(outdir / "B3_fairness_by_bottom.png", dpi=200)
    plt.close(fig)


def tornado_plot(summary: pd.DataFrame, outdir: Path) -> None:
    if summary is None or summary.empty:
        return
    if "fairness_score" not in summary.columns:
        return
    groups = []
    if "era_cutoff" in summary.columns:
        groups.append(("era_cutoff", summary.groupby("era_cutoff")["fairness_score"].mean()))
    if "wJ" in summary.columns and "fan_temp" in summary.columns:
        groups.append(("wJ", summary.groupby("wJ")["fairness_score"].mean()))
        groups.append(("fan_temp", summary.groupby("fan_temp")["fairness_score"].mean()))
    if "bottom_k" in summary.columns:
        groups.append(("bottom_k", summary.groupby("bottom_k")["fairness_score"].mean()))
    if "judge_save_rule" in summary.columns:
        groups.append(("judge_save_rule", summary.groupby("judge_save_rule")["fairness_score"].mean()))

    effects = []
    for name, series in groups:
        if series.empty:
            continue
        effects.append((name, float(series.max() - series.min())))
    if not effects:
        return
    effects = sorted(effects, key=lambda x: x[1])
    labels = [e[0] for e in effects]
    vals = [e[1] for e in effects]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(labels, vals, color="#4C78A8")
    ax.set_xlabel("Range of fairness_score")
    ax.set_title("Tornado: sensitivity of fairness_score")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(outdir / "tornado_fairness.png", dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualizations for sensitivity_analysis_b outputs.")
    parser.add_argument("--outdir", type=str, default="sensitivity_b_outputs")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    ensure_dir(outdir)

    b1_sum = load_csv(outdir / "B1_cutoff_summary.csv")
    b1_rev = load_csv(outdir / "B1_reversal_week.csv")
    b1_con = load_csv(outdir / "B1_controversy.csv")

    b2_sum = load_csv(outdir / "B2_grid_summary.csv")

    b3_sum = load_csv(outdir / "B3_bottom_summary.csv")
    b3_con = load_csv(outdir / "B3_controversy.csv")

    plot_reversal_by_week(b1_rev, outdir)
    plot_cutoff_fairness(b1_sum, outdir)
    plot_controversy_bar(b1_con, outdir, "B1_controversy_heatmap.png")

    plot_b2_heatmap(b2_sum, outdir)
    plot_b2_surface(b2_sum, outdir)

    plot_b3_bottom_summary(b3_sum, outdir)
    plot_controversy_bar(b3_con, outdir, "B3_controversy_heatmap.png")

    # Tornado (combined)
    all_summary = None
    for df in [b1_sum, b2_sum, b3_sum]:
        if df is None or df.empty:
            continue
        all_summary = df if all_summary is None else pd.concat([all_summary, df], ignore_index=True)
    tornado_plot(all_summary, outdir)


if __name__ == "__main__":
    main()
