"""Problem 3 figure producers (P-061, P-065, P-066, P-070, P-071).

Every function takes *saved source tables* (plus a fitted-summary dict) and an
output path, so each chart is a pure function of registered CSVs/JSON
(CLAUDE.md: figures only from saved source tables and run manifests).

The paper's Figure 6 PNGs exist in ``paper_Latex/img/`` but no legacy producer
created them; the tables here reproduce the same *information* (coefficients,
correlations, fixed effects, surprise fits) with honest uncertainty markers.

matplotlib is imported lazily so the core package stays importable in
environments without a plotting backend.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_TALK_STYLE_CANDIDATES = ("seaborn-v0_8-talk", "talk", "default")
TERM_RE = r"C\(industry_grp, Treatment\('Other'\)\)\[T\.(.*)\]$"


def _apply_talk_style(plt: Any) -> None:
    for style in _TALK_STYLE_CANDIDATES:
        if style in plt.style.available:
            plt.style.use(style)
            return
    plt.style.use("default")


def _sig(p: float) -> str:
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def _clean_term(term: str) -> str:
    import re

    if term == "Intercept":
        return "Intercept"
    if term == "celebrity_age_during_season":
        return "Age"
    m = re.match(TERM_RE, term)
    if m:
        return m.group(1)
    return term


def plot_success_factors_heatmap(
    coefs: pd.DataFrame, output_path: str | Path, *, metric: str = "coef"
) -> Path:
    """P-061: term x outcome coefficient heatmap with significance stars.

    ``coefs`` is the saved ``paper_demo_model`` table (term, outcome, coef, p).
    Rows are Age then industry deltas (Other reference); columns are the seven
    outcomes (judge W1/W6/W11, fan W1/W6/final, placement).  A blue-red
    diverging scale marks positive/negative effects.
    """
    import matplotlib
    import matplotlib.pyplot as plt

    matplotlib.use("Agg")
    _apply_talk_style(plt)

    df = coefs.copy()
    df["term_clean"] = df["term"].map(_clean_term)
    df = df[df["term_clean"] != "Intercept"]

    order = ["Age"] + sorted(df[df["term_clean"].ne("Age")]["term_clean"].unique())
    pivot = df.pivot(index="term_clean", columns="outcome", values=metric).reindex(order)
    pv = df.pivot(index="term_clean", columns="outcome", values="p").reindex(order)

    vmax = max(abs(pivot.min().min()), abs(pivot.max().max()), 0.05)
    fig, ax = plt.subplots(figsize=(9.6, max(4.2, 0.45 * len(order) + 1.5)))
    im = ax.imshow(pivot.to_numpy(dtype=float), cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_xticks(np.arange(pivot.shape[1]))
    ax.set_xticklabels(pivot.columns, rotation=40, ha="right")
    ax.set_yticks(np.arange(pivot.shape[0]))
    ax.set_yticklabels(pivot.index)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.iat[i, j]
            if np.isnan(val):
                continue
            star = _sig(float(pv.iat[i, j]))
            ax.text(
                j,
                i,
                f"{val:+.2f}{star}",
                ha="center",
                va="center",
                fontsize=8,
                color="#111111",
            )
    ax.set_title(
        "Success factors: standardized judge/fan outcomes on age + industry (HC3; * p<0.05)",
        pad=12,
        weight="semibold",
    )
    fig.colorbar(im, ax=ax, label="coef (z units)")
    fig.tight_layout()
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_partner_correlation_heatmap(corr: pd.DataFrame, output_path: str | Path) -> Path:
    """P-065: trait x outcome Pearson-r heatmap with significance stars."""
    import matplotlib
    import matplotlib.pyplot as plt

    matplotlib.use("Agg")
    _apply_talk_style(plt)

    pivot = corr.pivot(index="trait", columns="outcome", values="r").reindex(["H_exp", "H_abil"])
    pv = corr.pivot(index="trait", columns="outcome", values="p").reindex(["H_exp", "H_abil"])

    fig, ax = plt.subplots(figsize=(9.0, 2.6))
    im = ax.imshow(pivot.to_numpy(dtype=float), cmap="RdBu_r", vmin=-0.5, vmax=0.5)
    ax.set_xticks(np.arange(pivot.shape[1]))
    ax.set_xticklabels(pivot.columns, rotation=40, ha="right")
    ax.set_yticks(np.arange(pivot.shape[0]))
    ax.set_yticklabels(pivot.index)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.iat[i, j]
            if np.isnan(val):
                continue
            star = _sig(float(pv.iat[i, j]))
            ax.text(
                j,
                i,
                f"{val:+.2f}{star}",
                ha="center",
                va="center",
                fontsize=10,
                color="#111111",
            )
    ax.set_title(
        "Partner traits vs standardized outcomes (Pearson r; * p<0.05)", pad=12, weight="semibold"
    )
    fig.colorbar(im, ax=ax, label="r")
    fig.tight_layout()
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_partner_heterogeneity(
    fe_params: pd.DataFrame, output_path: str | Path, *, outcome_label: str = "judge W1"
) -> Path:
    """P-066: raw ability vs partner fixed effects ('Kingmaker' panel).

    Left: per-partner raw ability (H_abil mean) vs raw mean judge outcome with an
    OLS trend (the noisy raw correlation).  Right: per-partner fixed effects
    ``alpha_p`` from the judge-outcome FE fit, top 8 / bottom 4 labelled —
    veteran partners whose FE clears their raw ability are the 'Kingmakers'.
    """
    import matplotlib
    import matplotlib.pyplot as plt

    matplotlib.use("Agg")
    _apply_talk_style(plt)

    df = fe_params.sort_values("alpha_p", ascending=False).reset_index(drop=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 5.0))

    x = df["H_abil_mean"].to_numpy(dtype=float)
    y = df["outcome_mean"].to_numpy(dtype=float)
    ax1.scatter(x, y, s=34, color="#4C72B0", alpha=0.8)
    if len(x) >= 3:
        slope, intercept = np.polyfit(x, y, 1)
        xs = np.linspace(x.min(), x.max(), 80)
        ax1.plot(xs, slope * xs + intercept, color="#DD8452", linewidth=2)
    ax1.set_xlabel(r"Raw ability $H_{abil}$ (mean prior placement $z$)")
    ax1.set_ylabel("Mean judge outcome (z)")
    ax1.set_title(f"Raw ability vs outcome ({outcome_label})", pad=10, weight="semibold")
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    top = df.head(8)
    bottom = df.tail(4)
    show = pd.concat([top, bottom]).reset_index(drop=True)
    show["label"] = show["ballroom_partner"].map(
        lambda s: str(s).split()[-1] if len(str(s).split()) > 1 else str(s)
    )
    bar_colors = [
        "#2B6CB0" if str(row["ballroom_partner"]) else "#BBBBBB" for _, row in show.iterrows()
    ]
    order = show.sort_values("alpha_p", ascending=True)
    ax2.barh(range(len(order)), order["alpha_p"], color=bar_colors, alpha=0.85)
    ax2.set_yticks(range(len(order)))
    ax2.set_yticklabels(order["label"])
    ax2.axvline(0, color="#888888", linewidth=1)
    ax2.set_xlabel(r"Partner fixed effect $\hat{\alpha}_p$ (z units)")
    ax2.set_title("Partner FE (judge outcome)", pad=10, weight="semibold")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    fig.suptitle("Professional partner heterogeneity (P-066)", y=1.02, weight="semibold")
    fig.tight_layout()
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_surprise_linear(
    frame: pd.DataFrame,
    fit: dict[str, Any],
    output_path: str | Path,
) -> Path:
    """P-070: surprise-vs-growth scatter with the linear fit.

    ``frame`` is the saved S/G table (with ``industry_grp``); ``fit`` is the
    saved linear fit JSON (coefs/pvalues).  Athletes are highlighted.
    """
    import matplotlib
    import matplotlib.pyplot as plt

    matplotlib.use("Agg")
    _apply_talk_style(plt)

    fig, ax = plt.subplots(figsize=(7.4, 5.0))
    for grp, g in frame.groupby("industry_grp"):
        highlight = str(grp) in ("Athlete",)
        ax.scatter(
            g["S"],
            g["G"],
            s=30,
            alpha=0.75,
            color="#2B6CB0" if highlight else "#B0B7C3",
            edgecolor="none",
            label=str(grp) if highlight else None,
        )
    b0 = fit["coefs"]["Intercept"]
    b1 = fit["coefs"]["S"]
    xs = np.linspace(frame["S"].min(), frame["S"].max(), 100)
    ax.plot(xs, b0 + b1 * xs, color="#D97706", linewidth=2.4)
    ax.axhline(0, color="#BBBBBB", linewidth=0.8, linestyle=":")
    ax.axvline(0, color="#BBBBBB", linewidth=0.8, linestyle=":")
    ax.set_xlabel(r"Surprise $S = Z^{Judge}_t - Z^{Fan}_{t-1}$")
    ax.set_ylabel(r"Fan-vote growth $G = Z^{Fan}_t - Z^{Fan}_{t-1}$")
    ax.set_title(
        f"Surprise effect on vote growth  (β₁={b1:+.3f}, p={fit['pvalues']['S']:.1e}, "
        f"n={fit['n']})",
        pad=12,
        weight="semibold",
    )
    ax.legend(frameon=True, loc="lower right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_surprise_nonlinear(
    frame: pd.DataFrame,
    fit: dict[str, Any],
    pred_grid: pd.DataFrame,
    output_path: str | Path,
) -> Path:
    """P-071: nonlinearity and interaction panels.

    Left: scatter with the fitted quadratic ``G(S)`` curve (β₂>0 Matthew effect).
    Right: fitted ``G`` vs ``S`` for a rookie (H_exp=0) and veteran (H_exp=2)
    partner to isolate the β₃ interaction.  ``pred_grid`` is the saved
    ``predict_quadratic`` table.
    """
    import matplotlib
    import matplotlib.pyplot as plt

    matplotlib.use("Agg")
    _apply_talk_style(plt)

    b1 = fit["coefs"]["S"]
    b2 = fit["coefs"]["I(S ** 2)"]
    b3 = fit["coefs"]["S:H_exp"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.0, 4.8))

    ax1.scatter(frame["S"], frame["G"], s=26, alpha=0.7, color="#B0B7C3")
    xs = np.linspace(frame["S"].min(), frame["S"].max(), 120)
    curve = fit["coefs"]["Intercept"] + b1 * xs + b2 * xs**2 + b3 * xs * np.median(frame["H_exp"])
    ax1.plot(xs, curve, color="#D97706", linewidth=2.6)
    ax1.axhline(0, color="#BBBBBB", linewidth=0.8, linestyle=":")
    ax1.axvline(0, color="#BBBBBB", linewidth=0.8, linestyle=":")
    ax1.set_xlabel(r"Surprise $S$")
    ax1.set_ylabel(r"Growth $G$")
    ax1.set_title(
        f"Nonlinearity: β₂={b2:+.3f} (p={fit['pvalues']['I(S ** 2)']:.2g})",
        pad=10,
        weight="semibold",
    )
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    for h, color, label in (
        (0, "#4C72B0", "Rookie partner (H_exp=0)"),
        (2, "#DD8452", "Veteran partner (H_exp=2)"),
    ):
        g = pred_grid[pred_grid["H_exp"].eq(h)]
        ax2.plot(g["S"], g["G_pred"], color=color, linewidth=2.4, label=label)
    ax2.axhline(0, color="#BBBBBB", linewidth=0.8, linestyle=":")
    ax2.axvline(0, color="#BBBBBB", linewidth=0.8, linestyle=":")
    ax2.set_xlabel(r"Surprise $S$")
    ax2.set_ylabel(r"Fitted growth $G$")
    ax2.set_title(
        f"Interaction with partner tenure: β₃={b3:+.3f} (p={fit['pvalues']['S:H_exp']:.2g})",
        pad=10,
        weight="semibold",
    )
    ax2.legend(frameon=True, loc="lower right")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    fig.suptitle("Surprise nonlinearity and interaction (P-071)", y=1.02, weight="semibold")
    fig.tight_layout()
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out
