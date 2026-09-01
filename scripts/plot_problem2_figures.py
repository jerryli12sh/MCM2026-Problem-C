#!/usr/bin/env python3
"""Render the paper's Problem 2 figures from saved source tables (Track P/R).

Only reads the track-tagged CSVs written by ``scripts/problem2_run.py`` (and
that run's manifest); it never re-runs inference, satisfying the "figures only
from saved source tables and run manifests" rule.  Rendering follows the legacy
notebook cells (``../src/2_rank_vs_pct_cross_season.ipynb``) and the paper's
figure filenames in ``../paper_Latex/2107542.tex``:

- ``2_posterior_probability.png``      (P-042)  hist of weekly ``P_agree``;
- ``2_weekly_disagreement.png``        (P-043)  season ``DR_s`` bar;
- ``2_posterior_delta.png``            (P-045)  season ``delta`` with bootstrap 95% CI;
- ``2_threshold_fan_share.png``        (P-046)  alive-size vs threshold fan share;
- ``3_Reversal_Heatmap_Rank.png``      (P-049)  rank-rule reversal heatmap;
- ``3_Reversal_Heatmap_Percent.png``   (P-050)  percentage-rule reversal heatmap;
- ``3_Discrepancy_Scatter.png``        (P-051)  judge-fan disagreement landscape;
- ``4_EliminationProbability_<C>.png`` (P-053)  per-case weekly flip + elim-prob delta;
- ``4_SurvivalCurves_<C>.png``         (P-054)  per-case survival curves (Solid=Save, Dashed=Direct);
- ``4_RankVsPercentage_<C>.png``       (P-055)  per-case rank traces with Bottom-2 annotations.

Figures are written under ``outputs/figures_{track}/`` so the P and R variants
keep the paper-exact filenames without colliding.  A figure manifest
``problem2_fig_manifest_{track}.json`` records every file, its traceability id,
and its sha256.

Usage:
    python scripts/plot_problem2_figures.py [--track P|R] [--output-dir outputs]
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

# matplotlib is imported lazily so that non-rendering imports stay light.
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

_SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(_SRC))

from dwts_reproduction.config import load_paths  # noqa: E402
from dwts_reproduction.hashing import sha256_file  # noqa: E402

# (season, celebrity_name, paper-filename label) for the six named cases.
# Labels drop spaces to match the ``2107542.tex`` figure names.
CASE_FIGURES: list[tuple[int, str, str]] = [
    (2, "Jerry Rice", "JerryRice"),
    (4, "Billy Ray Cyrus", "BillyRayCyrus"),
    (11, "Bristol Palin", "BristolPalin"),
    (27, "Bobby Bones", "BobbyBones"),
    (27, "Tinashe", "Tinashe"),
    (31, "Vinny Guadagnino", "VinnyGuadagnino"),
]
# Highlighted cases on the disagreement landscape (legacy cell 23).
LANDSCAPE_CASES: list[tuple[int, str]] = [
    (2, "Jerry Rice"),
    (4, "Billy Ray Cyrus"),
    (11, "Bristol Palin"),
    (27, "Bobby Bones"),
]

ERA_COLORS = {"rank": "#4C72B0", "percent": "#DD8452"}

BOOTSTRAP_SEED = 42
BOOTSTRAP_N = 1000
BOOTSTRAP_Q = (0.025, 0.975)


def _load(table: str, tag: str, output_dir: Path) -> pd.DataFrame:
    return pd.read_csv(output_dir / f"problem2_{table}_{tag}.csv")


def plot_p042_histogram(agree: pd.DataFrame, out: Path) -> None:
    """P-042: posterior P(agree) histogram (legacy cell 10)."""
    fig, ax = plt.subplots(figsize=(6.8, 3.6))
    ax.hist(agree["P_agree"], bins=20, color="#4C72B0", edgecolor="white")
    ax.set_title("Posterior P(rank == percent)")
    ax.set_xlabel("P(agree)")
    ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(out, dpi=300)
    plt.close(fig)


def plot_p043_disagreement(season_metrics: pd.DataFrame, out: Path) -> None:
    """P-043: weekly disagreement rate DR_s per season (legacy cell 5)."""
    dr = season_metrics[season_metrics["metric"] == "dr"].sort_values("season")
    fig, ax = plt.subplots(figsize=(8.4, 3.6))
    ax.bar(dr["season"].astype(str), dr["point"], color="#4C72B0")
    ax.set_title("Weekly disagreement rate: rank vs percent")
    ax.set_xlabel("Season")
    ax.set_ylabel("Disagree rate")
    ax.tick_params(axis="x", rotation=90)
    fig.tight_layout()
    fig.savefig(out, dpi=300)
    plt.close(fig)


def plot_p045_delta(agree: pd.DataFrame, out: Path) -> None:
    """P-045: season delta = mean(P_override_rank) - mean(P_override_pct).

    Bootstrap 95% CI over weeks within each season (legacy cell 16) using a
    deterministic generator seeded ``BOOTSTRAP_SEED`` (the legacy created
    ``np.random.default_rng(42)`` for exactly this purpose); D-20260902-01.
    """
    boot_rows: list[dict[str, object]] = []
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    for s, g in agree.groupby("season"):
        g = g.reset_index(drop=True)
        n = len(g)
        if n < 2:
            continue
        rank = g["P_override_rank"].to_numpy()
        pct = g["P_override_pct"].to_numpy()
        delta = float(rank.mean() - pct.mean())
        deltas = np.empty(BOOTSTRAP_N)
        for b in range(BOOTSTRAP_N):
            idx = rng.choice(n, size=n, replace=True)
            deltas[b] = float(rank[idx].mean() - pct[idx].mean())
        lo, hi = np.quantile(deltas, BOOTSTRAP_Q)
        boot_rows.append({"season": int(s), "delta": delta, "lo": float(lo), "hi": float(hi)})
    boot = pd.DataFrame(boot_rows)

    fig, ax = plt.subplots(figsize=(8.6, 3.6))
    ax.errorbar(
        boot["season"].astype(str),
        boot["delta"],
        yerr=[boot["delta"] - boot["lo"], boot["hi"] - boot["delta"]],
        fmt="o",
        color="#4C72B0",
        ecolor="#4C72B0",
        capsize=3,
    )
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("Posterior Delta (rank - percent) with 95% interval")
    ax.set_xlabel("Season")
    ax.set_ylabel("Delta")
    ax.tick_params(axis="x", rotation=90)
    fig.tight_layout()
    fig.savefig(out, dpi=300)
    plt.close(fig)


def plot_p046_threshold(thr: pd.DataFrame, out: Path) -> None:
    """P-046: alive-set size vs threshold fan share (legacy cell 13)."""
    fig, ax = plt.subplots(figsize=(6.6, 3.6))
    for era, g in thr.groupby("era"):
        ax.scatter(g["alive_size"], g["thr_pct"], s=35, alpha=0.7, color=ERA_COLORS[era], label=era)
    ax.set_title("Threshold fan share (percent rule) for judge-worst to survive")
    ax.set_xlabel("Alive-set size")
    ax.set_ylabel("Threshold fan share")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=300)
    plt.close(fig)


def plot_p049_p050_heatmaps(rev: pd.DataFrame, fig_dir: Path) -> list[tuple[str, str]]:
    """P-049/P-050: reversal-rate heatmaps by season-week (legacy cell 27).

    Returns ``[(filename, trace_id)]`` for manifest recording.
    """
    records: list[tuple[str, str]] = []
    for col, trace_id, fname, title in (
        (
            "rev_rate_rank",
            "P-049",
            "3_Reversal_Heatmap_Rank.png",
            "Reversal rate vs direct-min (Rank rule, Bottom-2)",
        ),
        (
            "rev_rate_pct",
            "P-050",
            "3_Reversal_Heatmap_Percent.png",
            "Reversal rate vs direct-min (Percent rule, Bottom-2)",
        ),
    ):
        pivot = rev.pivot(index="season", columns="week", values=col)
        fig, ax = plt.subplots(figsize=(10, 6))
        im = ax.imshow(pivot.to_numpy(), aspect="auto", cmap="YlOrRd", vmin=0, vmax=1)
        ax.set_xticks(range(pivot.shape[1]), pivot.columns)
        ax.set_yticks(range(pivot.shape[0]), pivot.index)
        ax.set_title(title)
        ax.set_xlabel("Week")
        ax.set_ylabel("Season")
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        fig.savefig(fig_dir / fname, dpi=300)
        plt.close(fig)
        records.append((fname, trace_id))
        print(f"    {trace_id:<8} {fname}")
    return records


def plot_p051_discrepancy(di: pd.DataFrame, out: Path) -> None:
    """P-051: judge-fan disagreement landscape (legacy cell 23)."""
    fig, ax = plt.subplots(figsize=(6.8, 5.2))
    for era, g in di.groupby("era"):
        ax.scatter(
            g["delta_r_bar"], g["delta_s_bar"], s=35, alpha=0.35, color=ERA_COLORS[era], label=era
        )
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title("Judge–Fan disagreement: Δr̄ vs Δs̄")
    ax.set_xlabel("Δr̄ (fan rank − judge rank)")
    ax.set_ylabel("Δs̄ (fan share − judge share)")

    highlight = di.merge(
        pd.DataFrame(LANDSCAPE_CASES, columns=["season", "celebrity_name"]),
        on=["season", "celebrity_name"],
        how="inner",
    )
    ax.scatter(
        highlight["delta_r_bar"],
        highlight["delta_s_bar"],
        s=130,
        c="crimson",
        edgecolors="black",
        linewidths=0.8,
        zorder=5,
        label="Highlighted controversy cases",
    )
    for _, r in highlight.iterrows():
        ax.text(
            r["delta_r_bar"] + 0.15,
            r["delta_s_bar"] + 0.005,
            r["celebrity_name"],
            fontsize=9,
            weight="bold",
            color="black",
        )
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=300)
    plt.close(fig)


def plot_p053_elimination(weekly: pd.DataFrame, out: Path) -> None:
    """P-053: per-case weekly flip and elimination-probability delta (cell 34 Plot A)."""
    fig, ax = plt.subplots(figsize=(9.2, 3.6))
    delta = weekly["p_elim_pct"] - weekly["p_elim_rank"]
    ax.plot(
        weekly["week"],
        weekly["rev_rate_rank_vs_pct"],
        label="Pr(e_rank ≠ e_pct)",
        color="#4C72B0",
        marker="o",
    )
    ax.plot(
        weekly["week"], delta, label="Pr(elim pct) - Pr(elim rank)", color="#C44E52", marker="o"
    )
    ax.axhline(0, color="black", linewidth=0.8)
    name = weekly["celebrity_name"].iloc[0]
    season = int(weekly["season"].iloc[0])
    ax.set_title(f"{name} (Season {season}) — weekly flip & elim-prob delta")
    ax.set_xlabel("Week")
    ax.set_ylabel("Probability")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=300)
    plt.close(fig)


def plot_p054_survival(surv: pd.DataFrame, out: Path) -> None:
    """P-054: per-case survival curves, Solid=Save / Dashed=Direct (cell 34 Plot B)."""
    styles = {
        "rank_direct": dict(color="#4C72B0", linestyle="--", label="Rank (direct)"),
        "rank_bottom2": dict(color="#4C72B0", linestyle="-", label="Rank (save)"),
        "pct_direct": dict(color="#DD8452", linestyle="--", label="Pct (direct)"),
        "pct_bottom2": dict(color="#DD8452", linestyle="-", label="Pct (save)"),
    }
    fig, ax = plt.subplots(figsize=(8.6, 4.2))
    for mech, g in surv.groupby("mechanism"):
        g = g.sort_values("week")
        st = styles[mech]
        ax.plot(g["week"], g["S"], **st)
        ax.fill_between(g["week"], g["S_lo"], g["S_hi"], color=st["color"], alpha=0.15)
    name = surv["celebrity_name"].iloc[0]
    season = int(surv["season"].iloc[0])
    ax.set_title(f"{name} (Season {season}) — survival curves")
    ax.set_xlabel("Week")
    ax.set_ylabel("Survival probability")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=300)
    plt.close(fig)


def plot_p055_rank_traces(traces: pd.DataFrame, out: Path) -> None:
    """P-055: per-case rank traces with Bottom-2 annotations (legacy cell 39)."""
    fig, ax = plt.subplots(figsize=(9.2, 5.4))
    for _, r in traces.iterrows():
        ax.fill_between(
            [r["week"] - 0.5, r["week"] + 0.5],
            r["alive_n"] - 1 - 0.5,
            r["alive_n"] + 0.5,
            color="#f2f2f2",
            alpha=0.7,
            zorder=0,
        )
    ax.plot(
        traces["week"],
        traces["rJ"],
        color="#1f77b4",
        marker="o",
        linewidth=1.6,
        alpha=0.45,
        label="Judge rank",
    )
    ax.fill_between(
        traces["week"],
        traces["rF_lo"],
        traces["rF_hi"],
        color="#ff7f0e",
        alpha=0.1,
        label="Fan rank band (90% CI)",
    )
    ax.plot(
        traces["week"],
        traces["rF_mean"],
        color="#ff7f0e",
        marker="o",
        linewidth=1.6,
        alpha=0.45,
        label="Fan rank mean",
    )
    ax.plot(
        traces["week"],
        traces["rRank_mean"],
        color="#2ca02c",
        marker="o",
        linewidth=3,
        label="Combined rank-rule rank",
    )
    ax.plot(
        traces["week"],
        traces["rPct_mean"],
        color="#d62728",
        marker="o",
        linewidth=3,
        label="Combined percent-rule rank",
    )

    def _annotate_bottom2(x, y, in_b2, elim, y_offset):
        for wk, yv, b2, el in zip(x, y, in_b2, elim, strict=True):
            if not b2:
                continue
            ax.scatter(
                [wk], [yv], s=260, facecolors="none", edgecolors="black", linewidths=2.0, zorder=8
            )
            ax.text(
                wk + 0.10,
                yv + y_offset,
                "eliminated" if el else "saved",
                fontsize=8,
                color="black",
                zorder=9,
            )

    _annotate_bottom2(
        traces["week"].to_numpy(),
        traces["rRank_mean"].to_numpy(),
        traces["in_bottom2_rank"].to_numpy(),
        traces["elim_under_save_rank"].to_numpy(),
        y_offset=+0.15,
    )
    _annotate_bottom2(
        traces["week"].to_numpy(),
        traces["rPct_mean"].to_numpy(),
        traces["in_bottom2_pct"].to_numpy(),
        traces["elim_under_save_pct"].to_numpy(),
        y_offset=-0.15,
    )

    name = traces["celebrity_name"].iloc[0]
    season = int(traces["season"].iloc[0])
    ax.set_title(f"{name} (Season {season}) — Rank vs Percentage (weekly ranks)")
    ax.set_xlabel("Week")
    ax.set_ylabel("Rank (1 = best)")
    ax.invert_yaxis()
    ax.grid(alpha=0.25)
    ax.legend(loc="lower right", ncol=2, fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=300)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Problem 2 figure rendering (Track P / Track R)")
    parser.add_argument(
        "--track",
        choices=["P", "R"],
        default="P",
        help="P = paper-faithful replay; R = review-corrected replay",
    )
    parser.add_argument("--output-dir", default="outputs", help="where track-tagged artifacts live")
    args = parser.parse_args()
    tag = args.track

    started = datetime.now(UTC).isoformat(timespec="seconds")
    paths = load_paths()
    output_dir = (paths.repo_root / args.output_dir).resolve()
    fig_dir = output_dir / f"figures_{tag}"
    fig_dir.mkdir(parents=True, exist_ok=True)

    agree = _load("agree_week", tag, output_dir)
    reversal = _load("reversal_week", tag, output_dir)
    thr = _load("threshold_week", tag, output_dir)
    di = _load("divergence_index", tag, output_dir)
    surv = _load("case_survival", tag, output_dir)
    traces = _load("case_rank_traces", tag, output_dir)
    weekly = _load("case_weekly_probs", tag, output_dir)
    season_metrics = _load("season_metrics", tag, output_dir)

    figures: list[tuple[str, str]] = []  # (relative path, traceability id)

    def _record(rel: str, trace_id: str) -> None:
        figures.append((rel, trace_id))

    print(f"[problem2 figures {tag}] rendering from {output_dir.name}/ problem2_*_{tag}.csv")
    plot_p042_histogram(agree, fig_dir / "2_posterior_probability.png")
    _record(f"figures_{tag}/2_posterior_probability.png", "P-042")
    plot_p043_disagreement(season_metrics, fig_dir / "2_weekly_disagreement.png")
    _record(f"figures_{tag}/2_weekly_disagreement.png", "P-043")
    plot_p045_delta(agree, fig_dir / "2_posterior_delta.png")
    _record(f"figures_{tag}/2_posterior_delta.png", "P-045")
    plot_p046_threshold(thr, fig_dir / "2_threshold_fan_share.png")
    _record(f"figures_{tag}/2_threshold_fan_share.png", "P-046")
    plot_p051_discrepancy(di, fig_dir / "3_Discrepancy_Scatter.png")
    _record(f"figures_{tag}/3_Discrepancy_Scatter.png", "P-051")
    for fname, trace_id in plot_p049_p050_heatmaps(reversal, fig_dir):
        _record(f"figures_{tag}/{fname}", trace_id)

    for season, name, label in CASE_FIGURES:
        case_surv = surv[(surv["season"] == season) & (surv["celebrity_name"] == name)]
        case_traces = traces[(traces["season"] == season) & (traces["celebrity_name"] == name)]
        case_weekly = weekly[(weekly["season"] == season) & (weekly["celebrity_name"] == name)]
        plot_p053_elimination(case_weekly, fig_dir / f"4_EliminationProbability_{label}.png")
        _record(f"figures_{tag}/4_EliminationProbability_{label}.png", "P-053")
        plot_p054_survival(case_surv, fig_dir / f"4_SurvivalCurves_{label}.png")
        _record(f"figures_{tag}/4_SurvivalCurves_{label}.png", "P-054")
        plot_p055_rank_traces(case_traces, fig_dir / f"4_RankVsPercentage_{label}.png")
        _record(f"figures_{tag}/4_RankVsPercentage_{label}.png", "P-055")

    manifest = {
        "track": tag,
        "generated_from": f"problem2_run_manifest_{tag}.json",
        "source_tables": {
            table: f"problem2_{table}_{tag}.csv"
            for table in (
                "agree_week",
                "reversal_week",
                "threshold_week",
                "divergence_index",
                "case_survival",
                "case_rank_traces",
                "case_weekly_probs",
                "season_metrics",
            )
        },
        "bootstrap": {
            "seed": BOOTSTRAP_SEED,
            "n_resamples": BOOTSTRAP_N,
            "quantiles": list(BOOTSTRAP_Q),
        },
        "started_at": started,
        "ended_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "command": " ".join(sys.argv),
        "figures": {
            rel: {
                "traceability_id": trace_id,
                "sha256": sha256_file(fig_dir / Path(rel).name),
            }
            for rel, trace_id in figures
        },
    }
    manifest_path = output_dir / f"problem2_fig_manifest_{tag}.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"  figure manifest {manifest_path.name}  ({len(figures)} files)")
    for rel, trace_id in figures:
        print(f"    {trace_id:<8} {rel}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
