#!/usr/bin/env python3
"""Plot the Problem 1 evaluation-extra figures from saved source tables.

Every chart is a pure function of a CSV written by
``scripts/problem1_extras_run.py`` — no figure is generated from live model
state (CLAUDE.md: figures only from saved source tables).  The ranking-gap
figure recomputes its quadratic fit from the saved un-jittered CSV rather than
re-reading the JSON, so the chart and its 95% band are reproducible from the
table alone.

Produced figures (PNG under the output dir):
  problem1_fig_accuracy_line_P1E.png      <- P-029
  problem1_fig_crowded_field_weighted_P1E.png / ..._unweighted_P1E.png  <- P-033
  problem1_fig_ranking_gap_P1E.png        <- P-035
  problem1_fig_s8_heatmap_P1E.png         <- P-025
  problem1_fig_s21_heatmap_P1E.png        <- P-037

Usage:
    python scripts/plot_problem1_figures.py [--output-dir outputs]
"""

from __future__ import annotations

import argparse
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

_SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(_SRC))

from dwts_reproduction.config import load_paths  # noqa: E402
from dwts_reproduction.hashing import sha256_file  # noqa: E402
from dwts_reproduction.problem1.figures import (  # noqa: E402
    plot_accuracy_line,
    plot_crowded_field,
    plot_heatmap,
    plot_ranking_gap,
)
from dwts_reproduction.problem1.structural import quadratic_fit_with_ci  # noqa: E402
from dwts_reproduction.run_manifest import RunManifest  # noqa: E402

TAG = "P1E"


def _git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        )
        return out.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _environment() -> dict[str, str]:
    def ver(name: str) -> str:
        try:
            return getattr(__import__(name), "__version__", "unknown")
        except ImportError:
            return "not-installed"

    return {
        "python": platform.python_version(),
        "numpy": ver("numpy"),
        "pandas": ver("pandas"),
        "matplotlib": ver("matplotlib"),
        "platform": platform.platform(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot Problem 1 evaluation-extra figures")
    parser.add_argument("--output-dir", default="outputs", help="where the P1E tables + PNGs live")
    args = parser.parse_args()

    started = datetime.now(UTC).isoformat(timespec="seconds")
    paths = load_paths()
    out = (paths.repo_root / args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)

    def _read(name: str) -> pd.DataFrame:
        p = out / name
        if not p.exists():
            raise FileNotFoundError(f"missing saved source table for figure: {p}")
        return pd.read_csv(p)

    rendered: list[tuple[str, str]] = []  # (png_name, source_table)

    by_season = _read(f"problem1_extras_inseason_by_season_{TAG}.csv")
    png = out / f"problem1_fig_accuracy_line_{TAG}.png"
    plot_accuracy_line(by_season, png)
    rendered.append((png.name, f"problem1_extras_inseason_by_season_{TAG}.csv"))

    crowded = _read(f"problem1_extras_crowded_field_{TAG}.csv")
    for variant in ["weighted", "unweighted"]:
        png = out / f"problem1_fig_crowded_field_{variant}_{TAG}.png"
        plot_crowded_field(crowded, png, pcp_col=f"pcp_{variant}")
        rendered.append((png.name, f"problem1_extras_crowded_field_{TAG}.csv"))

    gap = _read(f"problem1_extras_ranking_gap_{TAG}.csv")
    fit = quadratic_fit_with_ci(
        gap["result_minus_judge"].to_numpy(), gap["audience_rank"].to_numpy(), order=2
    )
    png = out / f"problem1_fig_ranking_gap_{TAG}.png"
    plot_ranking_gap(gap, fit, png)
    rendered.append((png.name, f"problem1_extras_ranking_gap_{TAG}.csv"))

    s8 = _read(f"problem1_extras_s8_heatmap_{TAG}.csv")
    png = out / f"problem1_fig_s8_heatmap_{TAG}.png"
    plot_heatmap(s8, season=8, metric="p_mean", output_path=png)
    rendered.append((png.name, f"problem1_extras_s8_heatmap_{TAG}.csv"))

    s21 = _read(f"problem1_extras_s21_heatmap_{TAG}.csv")
    png = out / f"problem1_fig_s21_heatmap_{TAG}.png"
    plot_heatmap(s21, season=21, metric="ci_rel_width", output_path=png)
    rendered.append((png.name, f"problem1_extras_s21_heatmap_{TAG}.csv"))

    for png_name, source in rendered:
        print(f"  {png_name:<52} <- {source}")
    print(f"  ranking gap     R^2 = {fit.r_squared:.4f} (recomputed from saved table)")

    ended = datetime.now(UTC).isoformat(timespec="seconds")
    outputs = {name: sha256_file(out / name) for name, _ in rendered if (out / name).exists()}
    manifest = RunManifest(
        track=TAG,
        config_path="scripts/plot_problem1_figures.py",
        input_manifest_sha256=sha256_file(out / f"problem1_extras_run_manifest_{TAG}.json"),
        git_commit=_git_commit(),
        environment=_environment(),
        seeds={"plot_jitter_seed": 42, "ranking_gap_order": 2},
        command=" ".join(sys.argv),
        started_at=started,
        ended_at=ended,
        status="success",
        outputs=outputs,
    )
    manifest_path = out / f"problem1_fig_manifest_{TAG}.json"
    manifest.write(manifest_path)
    print(
        f"  figure manifest {manifest_path.name}  "
        f"(inputs sha {manifest.input_manifest_sha256[:12]}…)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
