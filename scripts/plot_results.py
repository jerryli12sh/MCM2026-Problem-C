#!/usr/bin/env python3
"""Build the project figures from saved analysis tables."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "results/tables"
FIGURES = ROOT / "results/figures"
NAVY, TEAL, GOLD, GREY = "#203653", "#168A8A", "#D6A24C", "#8693A3"


def table(name):
    return pd.read_csv(TABLES / name)


def finish(fig, name, source):
    fig.text(0.02, 0.012, source, fontsize=8, color="#637083")
    fig.savefig(FIGURES / name, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(name)


def reconstruction():
    p = table("problem1_top1_by_season_P.csv").set_index("season")
    r = table("problem1_top1_by_season_R.csv").set_index("season")
    fig, ax = plt.subplots(figsize=(13, 4.7))
    x = np.arange(len(p))
    ax.bar(x - 0.2, p.top1_accuracy, 0.4, color=NAVY, label="P: two-stage model")
    ax.bar(
        x + 0.2, r.loc[p.index].top1_accuracy, 0.4, color=TEAL, label="R: marginal-likelihood model"
    )
    ax.set(
        xticks=x,
        xticklabels=p.index,
        xlabel="Season",
        ylabel="Top-1 reconstruction rate",
        ylim=(0, 1.12),
    )
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.set_title("Historical elimination reconstruction", loc="left", pad=30, fontweight="bold")
    ax.text(
        0,
        1.06,
        "218 single-elimination weeks  |  P: 207 / 218 (94.95%)  |  R: 182 / 218 (83.49%)",
        transform=ax.transAxes,
        color="#566579",
        fontsize=10,
    )
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.31), ncol=2)
    fig.subplots_adjust(bottom=0.28, top=0.78)
    finish(
        fig,
        "reconstruction.png",
        "Source: problem1_top1_by_season_{P,R}.csv. Each bar uses that season's eligible single-elimination weeks.",
    )


def fan_support():
    d = table("problem1_posterior_summary_P.csv")
    d = d[d.season.eq(8)]
    order = (
        d.groupby("celebrity_name")
        .agg(last=("week", "max"), mean=("p_mean", "mean"))
        .sort_values(["last", "mean"], ascending=False)
        .index
    )
    grid = d.pivot(index="celebrity_name", columns="week", values="p_mean").loc[order]
    fig, ax = plt.subplots(figsize=(11, 6.4))
    cmap = plt.get_cmap("YlGnBu").copy()
    cmap.set_bad("#EDF1F5")
    im = ax.imshow(grid, aspect="auto", cmap=cmap, vmin=0)
    ax.set(
        xticks=range(len(grid.columns)),
        xticklabels=grid.columns,
        yticks=range(len(grid)),
        yticklabels=grid.index,
        xlabel="Week",
    )
    ax.set_title("Season 8 | weekly fan-support estimates", loc="left", fontweight="bold", pad=20)
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax, pad=0.03)
    cb.set_label("Posterior mean fan share")
    cb.ax.yaxis.set_major_formatter(PercentFormatter(1))
    fig.subplots_adjust(left=0.23, bottom=0.15, right=0.92)
    finish(
        fig,
        "fan_support.png",
        "Source: problem1_posterior_summary_P.csv. Grey cells have no estimate; shares normalize within the weekly active roster.",
    )


def rules():
    d = table("problem2_season_metrics_P.csv")
    wide = d.pivot(index="season", columns="metric", values="posterior_mean")
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), gridspec_kw={"width_ratios": [1.65, 1]})
    ax = axes[0]
    ax.plot(wide.index, wide["dr"], color=NAVY, marker="o", markersize=3)
    ax.set(xlabel="Season", ylabel="Posterior mean disagreement", ylim=(0, 1))
    ax.set_title("Rank and percentage rules", loc="left", fontweight="bold")
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax = axes[1]
    vals = wide[["override_rank", "override_pct"]].mean()
    bars = ax.bar(["Rank", "Percentage"], vals, color=[NAVY, TEAL], width=0.55)
    ax.bar_label(bars, labels=[f"{v:.1%}" for v in vals], padding=6)
    ax.set(ylabel="Fan override rate", ylim=(0, max(vals) * 1.3))
    ax.set_title("Average across 34 seasons", loc="left", fontweight="bold")
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    fig.subplots_adjust(wspace=0.3, bottom=0.2, top=0.86)
    finish(
        fig,
        "rule_comparison.png",
        "Source: problem2_season_metrics_P.csv. Within-season posterior means; right panel gives equal weight to each season.",
    )


def attributes():
    d = table("problem3_demo_coefs_P3.csv")
    d = d[d.term.eq("celebrity_age_during_season")].set_index("outcome")
    names = ["judge_w1", "judge_w6", "judge_w11", "fan_w1", "fan_w6", "fan_final"]
    labels = [
        "Judge | week 1",
        "Judge | week 6",
        "Judge | week 11",
        "Fan | week 1",
        "Fan | week 6",
        "Fan | late stage",
    ]
    d = d.loc[names]
    fig, ax = plt.subplots(figsize=(11, 5.6))
    for i, (_, row) in enumerate(d.iterrows()):
        ax.errorbar(
            row.coef * 10,
            i,
            xerr=1.96 * row.se * 10,
            fmt="o",
            color=NAVY if i < 3 else TEAL,
            capsize=4,
            markersize=7,
        )
        ax.text(
            1.01,
            i,
            f"n = {int(row.n)}",
            transform=ax.get_yaxis_transform(),
            color="#566579",
            va="center",
            fontsize=9,
        )
    ax.axvline(0, color=GREY, lw=1, ls="--")
    ax.set(
        yticks=range(6),
        yticklabels=labels,
        xlabel="Age coefficient per 10 years (standardized outcome units)",
    )
    ax.invert_yaxis()
    ax.set_title(
        "Age associations across judge and fan signals", loc="left", fontweight="bold", pad=22
    )
    fig.subplots_adjust(left=0.2, right=0.88, bottom=0.19, top=0.85)
    finish(
        fig,
        "attribute_effects.png",
        "Source: problem3_demo_coefs_P3.csv. OLS controlling for industry; intervals are 95% HC3 normal-approximation intervals.",
    )


def simulation():
    d = pd.concat(
        [table("problem4_shock_rates_V1.csv"), table("problem4_shock_rates_V2.csv")],
        ignore_index=True,
    )
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.4), gridspec_kw={"width_ratios": [1, 1.6]})
    ax = axes[0]
    bars = ax.bar(d.scheme, d.shock_k3, color=[GREY, GREY, NAVY, GOLD, TEAL], width=0.62)
    ax.bar_label(bars, labels=[f"{v:.2%}" for v in d.shock_k3], padding=5, fontsize=9)
    ax.set(ylabel="Share of elimination events", ylim=(0, max(d.shock_k3) * 1.3 + 0.005))
    ax.set_title("Technical shock: judge top 3", loc="left", fontweight="bold")
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    c = table("problem4_case_summary_V2.csv")
    wide = c.pivot(index="celebrity_name", columns="scheme", values="final_alive_rate")
    ax = axes[1]
    y = np.arange(len(wide))
    ax.barh(y + 0.18, wide.V4, 0.36, color=GOLD, label="V4: weighted baseline")
    ax.barh(y - 0.18, wide.V5, 0.36, color=TEAL, label="V5: transformed signals + bonus")
    ax.set(
        yticks=y,
        yticklabels=wide.index,
        xlabel="Last-observed-week survival in simulation",
        xlim=(0, 1.07),
    )
    ax.xaxis.set_major_formatter(PercentFormatter(1))
    ax.set_title("Six case studies", loc="left", fontweight="bold")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.43), fontsize=8)
    fig.subplots_adjust(wspace=0.68, bottom=0.29, top=0.87)
    finish(
        fig,
        "mechanism_simulation.png",
        "Source: problem4_shock_rates_* and problem4_case_summary_V2.csv. 300 simulations per scheme per season; fixed model inputs.",
    )


def sensitivity():
    d = table("sensitivity_summary_all_SA.csv")
    fig, ax = plt.subplots(figsize=(11, 5.3))
    families = [
        ("A1_", "Temperature / concentration"),
        ("A2_", "Regularization"),
        ("A3_", "Judge signal"),
        ("A4_", "Drop one season"),
    ]
    for i, (prefix, label) in enumerate(families):
        rows = d[d.scenario_id.str.startswith(prefix)]
        offsets = np.linspace(-0.16, 0.16, len(rows)) if len(rows) > 1 else np.array([0.0])
        ax.scatter(
            np.full(len(rows), i) + offsets,
            rows.accuracy,
            color=[NAVY, TEAL, GOLD, GREY][i],
            alpha=0.8,
            s=24,
            label=f"{len(rows)} fits",
        )
    baseline = d[d.scenario_id.eq("baseline")].accuracy.iloc[0]
    ax.axhline(baseline, color=NAVY, ls="--", lw=1, label=f"Baseline: {baseline:.2%}")
    ax.set(
        xticks=range(4), xticklabels=[label for _, label in families], ylabel="Reconstruction rate"
    )
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.set_title(
        "Sensitivity across parameters and season composition",
        loc="left",
        fontweight="bold",
        pad=22,
    )
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.3), ncol=5, fontsize=9)
    fig.subplots_adjust(bottom=0.28, top=0.84)
    finish(
        fig,
        "sensitivity.png",
        "Source: sensitivity_summary_all_SA.csv. Dropped-season scenarios evaluate the retained seasons.",
    )


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 14,
            "axes.labelcolor": NAVY,
            "text.color": NAVY,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#C8D0DA",
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": "#E8EDF2",
            "grid.alpha": 0.8,
            "legend.frameon": False,
        }
    )
    for plot in (reconstruction, fan_support, rules, attributes, simulation, sensitivity):
        plot()
    summary = []
    for tag in ("P", "R"):
        s = json.loads((TABLES / f"problem1_summary_{tag}.json").read_text())
        summary.append(
            {
                "model": tag,
                **{
                    k: s[k]
                    for k in (
                        "overall_top1_accuracy",
                        "mean_pcp_weighted",
                        "mean_ess_ratio",
                        "mean_ci_rel_width",
                        "s_bar",
                    )
                },
            }
        )
    pd.DataFrame(summary).to_csv(TABLES / "model_comparison.csv", index=False)


if __name__ == "__main__":
    main()
