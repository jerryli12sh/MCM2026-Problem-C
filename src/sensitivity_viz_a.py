import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

try:
    import seaborn as sns
    _HAS_SEABORN = True
except Exception:
    _HAS_SEABORN = False

import model


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_csv(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    return pd.read_csv(path)


def build_season_era_map(era_cutoff: int = 28) -> Dict[int, str]:
    df_elim_events, df_roster, df_weekly, df_long_judge, df_clean = model.load_tables()
    elim_long = model.build_elim_long(df_elim_events)
    base = model.build_base(df_roster, elim_long, df_clean)
    judge_percent = model.build_judge_percent(df_weekly, base)
    judge_rank_share = model.build_judge_rank_share(df_long_judge, base)
    panel = model.build_panel(base, judge_percent, judge_rank_share, era_cutoff=era_cutoff)
    season_era = panel.drop_duplicates(subset=["season"])[["season", "era"]]
    return {int(r.season): str(r.era) for r in season_era.itertuples(index=False)}


def heatmap_by_era(
    a1_week: pd.DataFrame,
    season_era: Dict[int, str],
    outdir: Path,
    metric: str = "pcp",
) -> None:
    if a1_week is None or a1_week.empty:
        return
    df = a1_week.copy()
    df["era"] = df["season"].map(season_era)
    if "tau" not in df.columns or "kappa" not in df.columns:
        return

    eras = sorted(df["era"].dropna().unique().tolist())
    if not eras:
        return

    ncols = len(eras)
    fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols, 4), squeeze=False)
    for i, era in enumerate(eras):
        ax = axes[0, i]
        dfe = df[df["era"] == era]
        pivot = dfe.pivot_table(index="kappa", columns="tau", values=metric, aggfunc="mean")
        if _HAS_SEABORN:
            sns.heatmap(pivot, ax=ax, cmap="viridis", cbar=True)
        else:
            im = ax.imshow(pivot.values, origin="lower", aspect="auto", cmap="viridis")
            fig.colorbar(im, ax=ax)
            ax.set_xticks(range(len(pivot.columns)))
            ax.set_xticklabels([f"{x:g}" for x in pivot.columns])
            ax.set_yticks(range(len(pivot.index)))
            ax.set_yticklabels([f"{y:g}" for y in pivot.index])
        ax.set_title(f"{metric.upper()} heatmap ({era})")
        ax.set_xlabel("tau")
        ax.set_ylabel("kappa")
    fig.tight_layout()
    fig.savefig(outdir / f"A1_heatmap_{metric}_by_era.png", dpi=200)
    plt.close(fig)


def line_plot_tau_kappa(
    a1_summary: pd.DataFrame,
    outdir: Path,
    metric: str = "pcp_mean",
) -> None:
    if a1_summary is None or a1_summary.empty:
        return
    df = a1_summary.copy()
    if "tau" not in df.columns or "kappa" not in df.columns:
        return

    fig, ax = plt.subplots(figsize=(7, 5))
    for kappa, g in df.groupby("kappa"):
        g = g.sort_values("tau")
        ax.plot(g["tau"], g[metric], marker="o", label=f"kappa={kappa:g}")
        if "pcp_p25" in g.columns and "pcp_p75" in g.columns and metric == "pcp_mean":
            ax.fill_between(g["tau"], g["pcp_p25"], g["pcp_p75"], alpha=0.15)
    ax.set_xlabel("tau")
    ax.set_ylabel(metric)
    ax.set_title(f"A1: {metric} vs tau (lines by kappa)")
    ax.legend(frameon=False)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(outdir / f"A1_line_{metric}_by_kappa.png", dpi=200)
    plt.close(fig)


def tornado_plot(
    summary_all: pd.DataFrame,
    baseline_summary: pd.DataFrame,
    outdir: Path,
    metric: str = "pcp_mean",
) -> None:
    if summary_all is None or summary_all.empty or baseline_summary is None or baseline_summary.empty:
        return
    base_val = float(baseline_summary.iloc[0].get(metric, np.nan))
    if not np.isfinite(base_val):
        return

    effects = []

    a1 = summary_all[summary_all["scenario"] == "A1_grid"].copy()
    if not a1.empty:
        tau_means = a1.groupby("tau")[metric].mean()
        kappa_means = a1.groupby("kappa")[metric].mean()
        effects.append(("tau", float((tau_means.max() - tau_means.min()) / max(1e-12, base_val))))
        effects.append(("kappa", float((kappa_means.max() - kappa_means.min()) / max(1e-12, base_val))))

    a2 = summary_all[summary_all["scenario"] == "A2_lambda_ratio"].copy()
    if not a2.empty:
        effects.append((
            "lambda_ratio",
            float((a2[metric].max() - a2[metric].min()) / max(1e-12, base_val)),
        ))

    a3 = summary_all[summary_all["scenario"] == "A3_judge_transform"].copy()
    if not a3.empty:
        effects.append((
            "judge_transform",
            float((a3[metric].max() - a3[metric].min()) / max(1e-12, base_val)),
        ))

    a4 = summary_all[summary_all["scenario"] == "A4_leave_one_season_out"].copy()
    if not a4.empty:
        effects.append((
            "leave_one_season_out",
            float((a4[metric].max() - a4[metric].min()) / max(1e-12, base_val)),
        ))

    if not effects:
        return

    effects = sorted(effects, key=lambda x: x[1])
    labels = [e[0] for e in effects]
    vals = [e[1] for e in effects]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(labels, vals, color="#4C78A8")
    ax.set_xlabel(f"Std. range (max-min) / baseline {metric}")
    ax.set_title(f"Tornado: sensitivity of {metric}")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(outdir / f"tornado_{metric}.png", dpi=200)
    plt.close(fig)


def pick_scenarios_for_scatter(summary_all: pd.DataFrame, k: int = 4) -> List[str]:
    if summary_all is None or summary_all.empty:
        return []
    cand = summary_all.dropna(subset=["spearman_p"]).copy()
    if cand.empty:
        return []
    cand = cand.sort_values("spearman_p")
    return cand["scenario_id"].head(k).tolist()


def stability_scatter(
    baseline_post: pd.DataFrame,
    all_post_files: List[Path],
    summary_all: pd.DataFrame,
    outdir: Path,
    k: int = 4,
    max_points: int = 4000,
) -> None:
    if baseline_post is None or baseline_post.empty:
        return
    scenario_ids = pick_scenarios_for_scatter(summary_all, k=k)
    if not scenario_ids:
        return

    scenario_post = []
    for path in all_post_files:
        df = load_csv(path)
        if df is None or df.empty:
            continue
        scenario_post.append(df)
    if not scenario_post:
        return
    post_all = pd.concat(scenario_post, ignore_index=True)

    fig, axes = plt.subplots(1, len(scenario_ids), figsize=(5 * len(scenario_ids), 4), squeeze=False)
    for i, sid in enumerate(scenario_ids):
        ax = axes[0, i]
        df = post_all[post_all["scenario_id"] == sid]
        merged = df.merge(
            baseline_post[["season", "week", "celebrity_name", "p_mean"]],
            on=["season", "week", "celebrity_name"],
            how="inner",
            suffixes=("", "_base"),
        )
        if merged.empty:
            continue
        if len(merged) > max_points:
            merged = merged.sample(max_points, random_state=42)
        ax.scatter(merged["p_mean_base"], merged["p_mean"], s=6, alpha=0.4)
        mn = min(merged["p_mean_base"].min(), merged["p_mean"].min())
        mx = max(merged["p_mean_base"].max(), merged["p_mean"].max())
        ax.plot([mn, mx], [mn, mx], color="red", lw=1)
        spearman = summary_all.loc[summary_all["scenario_id"] == sid, "spearman_p"].iloc[0]
        ax.set_title(f"{sid}\nSpearman={spearman:.3f}")
        ax.set_xlabel("baseline p_mean")
        ax.set_ylabel("scenario p_mean")
        ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(outdir / "stability_scatter.png", dpi=200)
    plt.close(fig)


def violin_for_contestants(
    baseline_post: pd.DataFrame,
    all_post_files: List[Path],
    summary_all: pd.DataFrame,
    contestants: List[str],
    outdir: Path,
    k: int = 3,
) -> None:
    if not contestants:
        return
    scenario_ids = pick_scenarios_for_scatter(summary_all, k=k)
    if not scenario_ids:
        return

    scenario_post = []
    for path in all_post_files:
        df = load_csv(path)
        if df is None or df.empty:
            continue
        scenario_post.append(df)
    if not scenario_post:
        return
    post_all = pd.concat(scenario_post, ignore_index=True)

    rows = []
    for name in contestants:
        base_vals = baseline_post[baseline_post["celebrity_name"] == name]["p_mean"].to_numpy()
        for v in base_vals:
            rows.append({"celebrity_name": name, "scenario": "baseline", "p_mean": v})
        for sid in scenario_ids:
            vals = post_all[
                (post_all["scenario_id"] == sid) & (post_all["celebrity_name"] == name)
            ]["p_mean"].to_numpy()
            for v in vals:
                rows.append({"celebrity_name": name, "scenario": sid, "p_mean": v})

    df = pd.DataFrame(rows)
    if df.empty:
        return

    fig, ax = plt.subplots(figsize=(8, 4 + 0.6 * len(contestants)))
    if _HAS_SEABORN:
        sns.violinplot(data=df, x="p_mean", y="celebrity_name", hue="scenario", ax=ax, cut=0)
    else:
        groups = [df[df["scenario"] == s]["p_mean"].to_numpy() for s in ["baseline"] + scenario_ids]
        ax.boxplot(groups, vert=False)
    ax.set_title("p_mean distribution for selected contestants")
    ax.set_xlabel("p_mean")
    fig.tight_layout()
    fig.savefig(outdir / "contestant_pmean_violin.png", dpi=200)
    plt.close(fig)


def robustness_map(outdir: Path) -> None:
    path = outdir / "robustness_map.csv"
    df = load_csv(path)
    if df is None or df.empty:
        return
    if not {"w_j", "fan_temp", "winner"}.issubset(df.columns):
        return
    pivot = df.pivot_table(index="fan_temp", columns="w_j", values="winner", aggfunc="first")
    fig, ax = plt.subplots(figsize=(6, 5))
    if _HAS_SEABORN:
        sns.heatmap(pivot, ax=ax, cmap="tab20", cbar=False)
    else:
        ax.imshow(pivot.values, origin="lower", aspect="auto")
    ax.set_title("Robustness map (winner by grid)")
    ax.set_xlabel("w_j")
    ax.set_ylabel("fan_temp")
    fig.tight_layout()
    fig.savefig(outdir / "robustness_map.png", dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualizations for sensitivity_analysis_a outputs.")
    parser.add_argument("--outdir", type=str, default="sensitivity_outputs")
    parser.add_argument("--metric", type=str, default="pcp_mean")
    parser.add_argument("--era-cutoff", type=int, default=28)
    parser.add_argument("--contestants", type=str, default="")
    parser.add_argument("--max-points", type=int, default=4000)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    ensure_dir(outdir)

    summary_all = load_csv(outdir / "summary_all.csv")
    baseline_summary = load_csv(outdir / "baseline_summary.csv")
    baseline_post = load_csv(outdir / "baseline_post.csv")
    a1_summary = load_csv(outdir / "A1_grid_summary.csv")
    a1_week = load_csv(outdir / "A1_grid_week.csv")

    season_era = build_season_era_map(era_cutoff=args.era_cutoff)

    heatmap_by_era(a1_week, season_era, outdir, metric="pcp")
    line_plot_tau_kappa(a1_summary, outdir, metric=args.metric)
    tornado_plot(summary_all, baseline_summary, outdir, metric=args.metric)

    post_files = [
        outdir / "A1_grid_post.csv",
        outdir / "A2_lambda_post.csv",
        outdir / "A3_judge_post.csv",
        outdir / "A4_leave_one_season_post.csv",
    ]
    stability_scatter(
        baseline_post, post_files, summary_all, outdir, k=4, max_points=args.max_points
    )

    contestants = [c.strip() for c in args.contestants.split(",") if c.strip()]
    if contestants:
        violin_for_contestants(baseline_post, post_files, summary_all, contestants, outdir, k=3)

    robustness_map(outdir)


if __name__ == "__main__":
    main()
