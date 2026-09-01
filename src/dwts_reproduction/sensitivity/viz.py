"""Figure 10 panels for the sensitivity analysis (matplotlib-only).

A dependency-free port of ``../src/sensitivity_viz_a.py`` producing exactly the
three panels the paper labels 10(a)-(c):

- ``10_stability_scatter.png`` (P-091): scenario vs baseline ``p_mean`` for the
  four lowest-Spearman scenarios, with the diagonal.
- ``10_tornado_pcp_mean.png`` (P-092): relative range of ``pcp_mean`` per
  perturbation family, reusing :func:`~.claims.effect_sizes` so the figure and
  the P-092 claim share one computation.
- ``10_A1_line_pcp_mean_by_kappa.png`` (P-093): ``pcp_mean`` vs ``tau``, one line
  per kappa, with the p25/p75 band.

Figures are generated only from saved source tables (the summary/post CSVs and
their run manifest), never re-run inference.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from dwts_reproduction.sensitivity.claims import effect_sizes


def load_csv(path: Path) -> pd.DataFrame | None:
    """Read a saved CSV, or ``None`` when absent/empty."""
    if not path.exists():
        return None
    df = pd.read_csv(path)
    return df if not df.empty else None


def line_plot_tau_kappa(
    a1_summary: pd.DataFrame | None,
    outdir: Path,
    metric: str = "pcp_mean",
    fname: str = "10_A1_line_pcp_mean_by_kappa.png",
) -> Path | None:
    """A1 lines: ``metric`` vs ``tau``, one line per kappa, p25/p75 band."""
    if a1_summary is None or a1_summary.empty:
        return None
    df = a1_summary.copy()
    if "tau" not in df.columns or "kappa" not in df.columns:
        return None
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
    path = outdir / fname
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def tornado_plot(
    summary_all: pd.DataFrame | None,
    baseline_summary: pd.DataFrame | None,
    outdir: Path,
    metric: str = "pcp_mean",
    fname: str = "10_tornado_pcp_mean.png",
) -> Path | None:
    """Tornado bar chart of ``metric`` relative ranges per perturbation family."""
    if (
        summary_all is None
        or summary_all.empty
        or baseline_summary is None
        or baseline_summary.empty
    ):
        return None
    effects = effect_sizes(summary_all, baseline_summary, metric)
    if not effects:
        return None
    labels = sorted(effects, key=lambda x: effects[x])
    vals = [effects[k] for k in labels]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(labels, vals, color="#4C78A8")
    ax.set_xlabel(f"Std. range (max-min) / baseline {metric}")
    ax.set_title(f"Tornado: sensitivity of {metric}")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    path = outdir / fname
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def pick_scenarios_for_scatter(summary_all: pd.DataFrame | None, k: int = 4) -> list[str]:
    """The ``k`` scenarios with the lowest Spearman (most informative scatter)."""
    if summary_all is None or summary_all.empty:
        return []
    cand = summary_all.dropna(subset=["spearman_p"]).copy()
    if cand.empty:
        return []
    cand = cand.sort_values("spearman_p")
    return list(cand["scenario_id"].head(k))


def stability_scatter(
    baseline_post: pd.DataFrame | None,
    all_post_files: list[Path],
    summary_all: pd.DataFrame | None,
    outdir: Path,
    k: int = 4,
    max_points: int = 4000,
    fname: str = "10_stability_scatter.png",
) -> Path | None:
    """Scatter of scenario vs baseline ``p_mean`` for the least stable scenarios."""
    if baseline_post is None or baseline_post.empty:
        return None
    scenario_ids = pick_scenarios_for_scatter(summary_all, k=k)
    if not scenario_ids or summary_all is None:
        return None
    post_all = pd.concat(
        [df for p in all_post_files if (df := load_csv(p)) is not None], ignore_index=True
    )
    if post_all.empty:
        return None

    fig, axes = plt.subplots(
        1, len(scenario_ids), figsize=(5 * len(scenario_ids), 4), squeeze=False
    )
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
    path = outdir / fname
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path
