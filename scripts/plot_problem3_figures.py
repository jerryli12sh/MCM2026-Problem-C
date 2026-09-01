#!/usr/bin/env python3
"""Plot the Problem 3 figures from saved source tables.

Every chart is a pure function of a CSV/JSON written by
``scripts/problem3_run.py`` — no figure is generated from live model state
(CLAUDE.md: figures only from saved source tables).  The surprise figures
re-fit their trend lines from the saved S/G frame and the saved fit JSONs.

Produced figures (PNG under the output dir):
  problem3_fig_success_factors_P3.png        <- P-061 (paper_demo_model coefs)
  problem3_fig_partner_correlation_P3.png    <- P-065 (trait x outcome r)
  problem3_fig_partner_heterogeneity_P3.png  <- P-066 (raw ability vs FE)
  problem3_fig_surprise_linear_P3.png        <- P-070 (S vs G, linear fit)
  problem3_fig_surprise_nonlinear_P3.png     <- P-071 (quadratic + interaction)

Usage:
    python scripts/plot_problem3_figures.py [--output-dir outputs]
"""

from __future__ import annotations

import argparse
import json
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
from dwts_reproduction.problem3.figures import (  # noqa: E402
    plot_partner_correlation_heatmap,
    plot_partner_heterogeneity,
    plot_success_factors_heatmap,
    plot_surprise_linear,
    plot_surprise_nonlinear,
)
from dwts_reproduction.run_manifest import RunManifest  # noqa: E402

TAG = "P3"


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
    parser = argparse.ArgumentParser(description="Plot Problem 3 figures")
    parser.add_argument("--output-dir", default="outputs", help="where the P3 tables + PNGs live")
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

    def _read_json(name: str) -> dict:
        p = out / name
        if not p.exists():
            raise FileNotFoundError(f"missing saved fit JSON for figure: {p}")
        return json.loads(p.read_text(encoding="utf-8"))

    rendered: list[tuple[str, str]] = []  # (png_name, source table)

    demo = _read(f"problem3_demo_coefs_{TAG}.csv")
    png = out / f"problem3_fig_success_factors_{TAG}.png"
    plot_success_factors_heatmap(demo, png)
    rendered.append((png.name, f"problem3_demo_coefs_{TAG}.csv"))

    corr = _read(f"problem3_partner_correlations_{TAG}.csv")
    png = out / f"problem3_fig_partner_correlation_{TAG}.png"
    plot_partner_correlation_heatmap(corr, png)
    rendered.append((png.name, f"problem3_partner_correlations_{TAG}.csv"))

    fe_params = _read(f"problem3_partner_fe_params_{TAG}.csv")
    png = out / f"problem3_fig_partner_heterogeneity_{TAG}.png"
    plot_partner_heterogeneity(fe_params, png)
    rendered.append((png.name, f"problem3_partner_fe_params_{TAG}.csv"))

    frame = _read(f"problem3_surprise_frame_tw6_{TAG}.csv")
    linear = _read_json(f"problem3_surprise_linear_tw6_{TAG}.json")
    png = out / f"problem3_fig_surprise_linear_{TAG}.png"
    plot_surprise_linear(frame, linear, png)
    rendered.append((png.name, f"problem3_surprise_frame_tw6_{TAG}.csv"))

    quad = _read_json(f"problem3_surprise_quadratic_tw6_{TAG}.json")
    grid = _read(f"problem3_surprise_predict_grid_{TAG}.csv")
    png = out / f"problem3_fig_surprise_nonlinear_{TAG}.png"
    plot_surprise_nonlinear(frame, quad, grid, png)
    rendered.append((png.name, f"problem3_surprise_frame_tw6_{TAG}.csv"))

    for png_name, source in rendered:
        print(f"  {png_name:<52} <- {source}")

    ended = datetime.now(UTC).isoformat(timespec="seconds")
    outputs = {name: sha256_file(out / name) for name, _ in rendered if (out / name).exists()}
    manifest = RunManifest(
        track=TAG,
        config_path="scripts/plot_problem3_figures.py",
        input_manifest_sha256=sha256_file(out / f"problem3_run_manifest_{TAG}.json"),
        git_commit=_git_commit(),
        environment=_environment(),
        seeds={"min_cat_count": 6, "cv_min_train_seasons": 3},
        command=" ".join(sys.argv),
        started_at=started,
        ended_at=ended,
        status="success",
        outputs=outputs,
    )
    manifest_path = out / f"problem3_fig_manifest_{TAG}.json"
    manifest.write(manifest_path)
    print(
        f"  figure manifest {manifest_path.name}  "
        f"(inputs sha {manifest.input_manifest_sha256[:12]}…)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
