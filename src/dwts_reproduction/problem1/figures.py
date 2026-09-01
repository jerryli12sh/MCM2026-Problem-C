"""Problem 1 figure producers (P-029, P-033, P-035, P-025, P-037).

Every function takes a *saved source table* and an output path — the figures are
pure functions of persisted CSVs so the run manifest can back each chart with a
registered input (CLAUDE.md: figures only from saved source tables).

matplotlib is imported lazily inside each function so the core package stays
importable in environments without a plotting backend.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# seaborn's ``talk`` style; matplotlib >= 3.6 ships it under the ``seaborn-v0_8-``
# prefix.  Try the modern name, then the bare name (older matplotlib), then
# matplotlib's default so figures still render without seaborn installed.
_TALK_STYLE_CANDIDATES = ("seaborn-v0_8-talk", "talk", "default")


def _apply_talk_style(plt: Any) -> None:
    for style in _TALK_STYLE_CANDIDATES:
        if style in plt.style.available:
            plt.style.use(style)
            return
    plt.style.use("default")


PALETTE_ACCURACY: dict[str, str] = {
    "torch_model": "#2B6CB0",
    "xgboost_baseline": "#D97706",
}
PALETTE_ERA: dict[str, str] = {
    "rank": "#4C72B0",
    "percent": "#DD8452",
}
# Colour of the overall crowded-field trend line (seaborn regplot default for a
# hue-less fit in the legacy notebook).
TREND_COLOR = "#4C72B0"


def plot_accuracy_line(by_season: pd.DataFrame, output_path: str | Path) -> Path:
    """Reproduce the paper Fig. 1 in-season accuracy line (P-029).

    Matplotlib-only port of ``src/plot_cv_accuracy_line.py`` (seaborn is not a
    dependency here): two per-season lines (torch model, XGBoost baseline) over
    ``ylim (-0.02, 1.02)`` with the legacy palette, markers, and title.  The torch
    line is the repository numpy rebuild fitted per season; the XGBoost line is
    the exact ``src/xgb_baseline.py`` port (see D-20260901-13).
    """
    import matplotlib
    import matplotlib.pyplot as plt

    matplotlib.use("Agg")
    _apply_talk_style(plt)
    fig, ax = plt.subplots(figsize=(10, 5.2))
    for model in ["torch_model", "xgboost_baseline"]:
        g = by_season[by_season["model"].eq(model)].sort_values("season")
        if g.empty:
            continue
        ax.plot(
            g["season"],
            g["accuracy"],
            marker="o",
            linewidth=2.6,
            markersize=6.5,
            color=PALETTE_ACCURACY[model],
            label=model,
        )
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("In-Season Accuracy by Season", pad=12, weight="semibold")
    ax.set_xlabel("Season")
    ax.set_ylabel("Accuracy")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle="-", alpha=0.35)
    ax.margins(x=0.02, y=0.02)
    ax.legend(loc="lower right", frameon=True)
    fig.tight_layout()
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=220)
    plt.close(fig)
    return out


def plot_crowded_field(
    crowded: pd.DataFrame, output_path: str | Path, *, pcp_col: str = "pcp_weighted"
) -> Path:
    """Reproduce the PCP-vs-alive-set-size scatter (P-033, notebook cell 54).

    Scatter ``y=pcp`` against ``x=|A_{s,t}|`` coloured by era, with an overall
    linear trend (``regplot(scatter=False)`` equivalent, drawn over all points).
    ``pcp_col`` selects the plotted PCP variant (``pcp_weighted`` or
    ``pcp_unweighted``) and defaults to the weighted one; the parameter
    discrepancy against the legacy notebook (``kappa=30``/``B=2500`` weighted IS)
    is recorded in D-20260901-14.  Only weeks with a finite PCP are plotted.
    """
    import matplotlib
    import matplotlib.pyplot as plt

    matplotlib.use("Agg")
    data = crowded[crowded[pcp_col].notna() & crowded["alive_n"].notna()].copy()
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    for era, g in data.groupby("era"):
        ax.scatter(
            g["alive_n"],
            g[pcp_col],
            color=PALETTE_ERA.get(str(era), "#444444"),
            alpha=0.7,
            s=40,
            label=str(era),
        )
    # Overall linear trend across all weeks (legacy regplot, scatter=False).
    x = data["alive_n"].to_numpy(dtype=float)
    y = data[pcp_col].to_numpy(dtype=float)
    if len(x) >= 2:
        slope, intercept = np.polyfit(x, y, 1)
        xs = np.linspace(x.min(), x.max(), 100)
        ax.plot(xs, slope * xs + intercept, color=TREND_COLOR, linewidth=2.0)
    ax.set_title("PCP vs alive-set size", pad=12, weight="semibold")
    ax.set_xlabel(r"Alive-set size $|A_{s,t}|$")
    ax.set_ylabel("PCP")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=True, title="era")
    fig.tight_layout()
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_ranking_gap(
    frame: pd.DataFrame,
    fit: Any,
    output_path: str | Path,
    *,
    jitter: float = 0.15,
    seed: int = 42,
) -> Path:
    """Reproduce the ranking-gap scatter with a quadratic fit and 95% band (P-035).

    The fit (``fit`` is the :class:`QuadFit` from ``structural.quadratic_fit_with_ci``)
    is computed on un-jittered data; only the scatter points receive a uniform
    ``±jitter`` noise on the x-axis, matching ``seaborn.regplot(x_jitter=..., seed=42)``
    in notebook cell 56 (D-20260901-15).
    """
    import matplotlib
    import matplotlib.pyplot as plt

    matplotlib.use("Agg")
    rng = np.random.default_rng(seed)
    x_orig = frame["result_minus_judge"].to_numpy(dtype=float)
    y = frame["audience_rank"].to_numpy(dtype=float)
    mask = np.isfinite(x_orig) & np.isfinite(y)
    x_scatter = x_orig[mask] + rng.uniform(-jitter, jitter, size=int(mask.sum()))

    fig, ax = plt.subplots(figsize=(7.4, 6.2))
    ax.scatter(x_scatter, y[mask], s=30, alpha=0.8, color="#2B6CB0")
    ax.fill_between(
        fit.x_grid,
        fit.ci_lo,
        fit.ci_hi,
        color=TREND_COLOR,
        alpha=0.18,
        label="95% CI",
    )
    ax.plot(
        fit.x_grid,
        fit.y_fit,
        color=TREND_COLOR,
        linewidth=2.2,
        label=f"Quadratic fit (R²={fit.r_squared:.3f})",
    )
    ax.set_title(
        "Inferred Fan-Vote Ranking vs. Final Placement–Judge Ranking Gap", pad=12, weight="semibold"
    )
    ax.set_xlabel("Final Placement minus Average Judge Ranking")
    ax.set_ylabel("Inferred Fan-Vote Ranking (from Model Mean p)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=True, loc="lower right")
    fig.tight_layout()
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


# --------------------------------------------------------------------------- #
# Uncertainty heatmaps (P-025, P-037) — legacy src/plot_uncertainty.py
# --------------------------------------------------------------------------- #
def order_contestants(df_season: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Order contestants by (exit week desc, mean metric desc) like the legacy tool.

    The legacy heatmap sorted by ``exit_week`` (max alive week, from the legacy
    per-draw posterior) then ``mean_p_hat``.  The repo's saved
    ``posterior_summary`` has no explicit ``exit_week`` — it only contains alive
    rows, so ``exit_week = max(week)`` per contestant reproduces the legacy
    quantity (D-20260901-16).
    """
    g = (
        df_season.groupby("celebrity_name", as_index=False)
        .agg(
            mean_metric=(metric, "mean"),
            exit_week=("week", "max"),
        )
        .sort_values(["exit_week", "mean_metric"], ascending=[False, False])
    )
    return g


def plot_heatmap(
    df_season: pd.DataFrame,
    season: int,
    metric: str,
    output_path: str | Path,
    *,
    title_metric: str | None = None,
) -> Path:
    """Plot the per-week contestant heatmap for one season (P-025 / P-037).

    ``metric`` is a column of the saved posterior summary (``p_mean`` for the
    P-025 adaptation, ``ci_rel_width`` for P-037).  Contestants are ordered by
    ``order_contestants``; only alive weeks appear (the legacy tool pivoted alive
    weeks only, so the heatmap is a ragged grid with NaN for non-alive weeks).
    """
    import matplotlib
    import matplotlib.pyplot as plt

    matplotlib.use("Agg")
    order = order_contestants(df_season, metric)
    pivot = df_season.pivot_table(
        index="celebrity_name", columns="week", values=metric, aggfunc="first"
    )
    pivot = pivot.reindex(order["celebrity_name"])

    height = max(4, 0.35 * len(order))
    fig, ax = plt.subplots(figsize=(12, height))
    im = ax.imshow(
        pivot.to_numpy(dtype=float),
        aspect="auto",
        interpolation="nearest",
        cmap="viridis",
    )
    ax.set_xticks(np.arange(pivot.shape[1]))
    ax.set_xticklabels(pivot.columns.astype(int).tolist())
    ax.set_yticks(np.arange(pivot.shape[0]))
    ax.set_yticklabels(pivot.index.tolist())
    ax.set_xlabel("Week")
    ax.set_ylabel("Contestant (alive weeks only)")
    ax.set_title(
        f"Season {season}: {(title_metric or metric).replace('_', ' ').title()} Heatmap",
        pad=12,
        weight="semibold",
    )
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300)
    plt.close(fig)
    return out
