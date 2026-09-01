"""Problem-4 composite figures (paper Figure 8 family).

The paper embeds four chart families with an ``8_`` filename prefix
(``\\includegraphics`` references in ``2107542.tex``).  This module reproduces
the three *composite* charts as pure functions of the saved simulator summary
tables; the named-case trend charts live in :mod:`.cases`
(``8_fig_sim2_trend_<season>_<name>.png``).

- ``8_V1_plot1.png`` — week-by-archetype heatmap of
  ``Delta(-rank_bar) = (-rank_bar_S3) - (-rank_bar_S1)``; positive = improved
  average placement.  Black lines overlay each archetype's weekly mean effect.
- ``8_fig_diff_contour_avg_rank_S3_minus_S1.png`` — the ``4_plot_1.ipynb``
  cell-4 diff-contour recipe (rank_score = -avg_rank).
- ``8_fig_ribbon_survival_by_archetype.png`` — the cell-1 survival ribbon
  ``S(t)`` per archetype for the V2 mechanisms (``V4`` baseline vs ``V5``
  proposed).  The paper's ``fig:survival_by_archetype`` caption describes the
  V0-vs-V2 comparison and the legacy V2 notebook produces the ribbon, so the
  V2 summary is the source of record (D-20260901-18).

The legacy notebooks styled with seaborn; seaborn is not a repo dependency, so
``_apply_style`` reproduces the visual intent with explicit matplotlib
rcParams.  Figures are only ever rendered from saved source tables — they never
touch simulator internals.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ARCHETYPE_ORDER = ["balanced", "relative_popular", "relative_technical"]
_TALK_STYLE_CANDIDATES = ("seaborn-v0_8-talk", "talk", "default")


def _apply_style(plt: Any) -> None:
    for style in _TALK_STYLE_CANDIDATES:
        if style in plt.style.available:
            plt.style.use(style)
            return
    plt.style.use("default")


def _delta_rank_frame(summary_v1: pd.DataFrame, base: str = "S1", new: str = "S3") -> pd.DataFrame:
    """Long frame of ``Delta(-avg_rank) = (-avg_rank_new) - (-avg_rank_base)``."""
    if summary_v1.empty:
        return summary_v1
    wide = summary_v1.pivot_table(index="week", columns=["scheme", "archetype"], values="avg_rank")
    rank_score = -wide
    diff = rank_score[(new,)] - rank_score[(base,)]
    return (
        diff.stack()
        .rename("delta_rank_score")
        .reset_index()
        .rename(columns={"level_1": "archetype"})
    )


def _pivot_diff_2d(
    summary_v1: pd.DataFrame, base: str = "S1", new: str = "S3"
) -> tuple[pd.DataFrame, float]:
    """(archetype x week) diff pivot + a symmetric vmax for diverging maps."""
    long_df = _delta_rank_frame(summary_v1, base, new)
    piv = long_df.pivot(index="archetype", columns="week", values="delta_rank_score")
    piv = piv.reindex([a for a in ARCHETYPE_ORDER if a in piv.index])
    vmax = float(np.nanmax(np.abs(piv.to_numpy()))) if piv.notna().any().any() else 1.0
    return piv, vmax


def heatmap_delta_rank(summary_v1: pd.DataFrame, out_path: str | Path) -> Path:
    """Render ``8_V1_plot1.png``: week-by-archetype heatmap of Delta(-rank_bar).

    Positive values (red) = improved average placement under S3 vs S1.  Black
    lines trace each archetype's weekly mean effect across the heatmap (paper
    caption "Black lines show the weekly mean effect by archetype").
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    _apply_style(plt)
    piv, vmax = _pivot_diff_2d(summary_v1)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9.5, 4.2))
    norm = TwoSlopeNorm(vcenter=0.0, vmin=-vmax, vmax=vmax)
    im = ax.imshow(piv.to_numpy(), aspect="auto", cmap="RdBu_r", norm=norm)
    ax.set_yticks(range(len(piv.index)))
    ax.set_yticklabels(piv.index)
    ax.set_xticks(range(len(piv.columns)))
    ax.set_xticklabels(piv.columns)
    ax.set_xlabel("Week")
    ax.set_ylabel("Archetype")
    ax.set_title(r"$\Delta(-\overline{rank}) = (-\overline{rank}_{S3}) - (-\overline{rank}_{S1})$")

    # Black lines: each archetype's weekly mean effect, drawn over its row.
    x = np.arange(len(piv.columns))
    for _, values in piv.iterrows():
        y = values.to_numpy(dtype=float)
        ax.plot(x, y, color="black", lw=1.4, alpha=0.85)

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("mean rank improvement")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def diff_contour_rank(summary_v1: pd.DataFrame, out_path: str | Path) -> Path:
    """Render ``8_fig_diff_contour_avg_rank_S3_minus_S1.png`` (4_plot_1 cell 4).

    contourf at 15 levels on a symmetric diverging scale, 10 thin contour
    levels, the zero contour emphasized (lw 2.2), and archetype trajectories
    overlaid; the colorbar carries the notebook's exact label.
    """
    import matplotlib.pyplot as plt

    _apply_style(plt)
    piv, vmax = _pivot_diff_2d(summary_v1)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    weeks = piv.columns.to_numpy(dtype=float)
    archetypes = piv.index
    z = piv.to_numpy(dtype=float)
    if vmax <= 0:
        vmax = 1.0
    levels = np.linspace(-vmax, vmax, 15)

    fig, ax = plt.subplots(figsize=(9.5, 5.4))
    cf = ax.contourf(weeks, np.arange(len(archetypes)), z, levels=levels, cmap="RdBu_r")
    ax.contour(weeks, np.arange(len(archetypes)), z, levels=10, colors="black", linewidths=0.5)
    ax.contour(
        weeks,
        np.arange(len(archetypes)),
        z,
        levels=[0.0],
        colors="black",
        linewidths=2.2,
    )
    for i, _arc in enumerate(archetypes):
        ax.plot(weeks, z[i, :], color="#444444", lw=1.4, alpha=0.9)
    ax.set_yticks(np.arange(len(archetypes)))
    ax.set_yticklabels(archetypes)
    ax.set_xlabel("Week")
    ax.set_ylabel("Archetype")
    ax.set_title("S3 vs S1: change in mean rank score (rank_score = -avg_rank)")
    cbar = fig.colorbar(cf, ax=ax)
    cbar.set_label(
        r"$\Delta$ rank_score = $(-\overline{avg\_rank}_{S3}) - (-\overline{avg\_rank}_{S1})$"
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _survival_frame(summary: pd.DataFrame) -> pd.DataFrame:
    """Cumulative survival per (scheme, archetype) with the week-0 baseline row."""
    if summary.empty:
        return summary
    s = summary[["scheme", "week", "archetype", "alive_rate"]].copy()
    s = s.sort_values(["scheme", "archetype", "week"])
    s["cum_alive_rate"] = s.groupby(["scheme", "archetype"])["alive_rate"].cumprod()
    baseline = (
        s.groupby(["scheme", "archetype"], as_index=False)
        .first()
        .assign(week=0, alive_rate=1.0, cum_alive_rate=1.0)
    )
    return pd.concat([baseline, s], ignore_index=True)


def ribbon_survival(summary_v2: pd.DataFrame, out_path: str | Path) -> Path:
    """Render ``8_fig_ribbon_survival_by_archetype.png`` (4_plot_2 cell 1).

    One panel per archetype, ``S(t)`` (cumulative product of weekly alive_rate)
    for the V2 baseline (``V4``) and proposed (``V5``) schemes, lines with
    point markers and a translucent fill down to 0, mirroring the notebook
    recipe (week-0 baseline at 1.0 included).
    """
    import matplotlib.pyplot as plt

    _apply_style(plt)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    surv = _survival_frame(summary_v2)
    archetypes = [a for a in ARCHETYPE_ORDER if a in surv["archetype"].unique()]
    colors = {"V4": "#d62728", "V5": "#2ca02c"}

    ncols = 3
    nrows = -(-len(archetypes) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.4 * ncols, 4.6 * nrows), squeeze=False)
    for idx, arc in enumerate(archetypes):
        ax = axes[idx // ncols][idx % ncols]
        for scheme, g in surv[surv["archetype"] == arc].groupby("scheme"):
            g = g.sort_values("week")
            color = colors.get(scheme, "#888888")
            ax.plot(
                g["week"],
                g["cum_alive_rate"],
                color=color,
                marker="o",
                lw=2.6,
                ms=4,
                label=scheme,
            )
            ax.fill_between(g["week"], g["cum_alive_rate"], 0, color=color, alpha=0.12)
        ax.set_title(f"{arc}")
        ax.set_ylim(-0.02, 1.02)
        ax.set_xlabel("Week")
        ax.set_ylabel("Survival S(t)")
        ax.grid(alpha=0.2)
        ax.legend(frameon=False, loc="lower left")
    for j in range(len(archetypes), nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")
    fig.suptitle("Simulated survival by archetype — V4 (baseline) vs V5 (proposed)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path
