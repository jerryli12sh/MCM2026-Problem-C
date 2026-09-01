#!/usr/bin/env python3
"""Render Problem-4 paper Figure 8 charts from saved source tables only.

Reads the tables produced by ``scripts/problem4_run.py`` (sim summaries,
per-case weekly trends, shock rates) and renders the four chart families the
paper embeds with an ``8_`` filename prefix:

- ``8_V1_plot1.png`` — week-by-archetype heatmap of Δ(−rank̄) (S3 vs S1).
- ``8_fig_diff_contour_avg_rank_S3_minus_S1.png`` — diff contour (4_plot_1
  cell-4 recipe).
- ``8_fig_ribbon_survival_by_archetype.png`` — V4 vs V5 survival ribbon
  (4_plot_2 cell-1 recipe).
- ``8_fig_sim2_trend_<season>_<name>.png`` — the six V2 named-case trends.

Writes ``outputs/problem4_fig_manifest_P4.json`` listing every file with its
SHA-256, so no figure exists without a provenance record.

Usage:
    python scripts/plot_problem4_figures.py [--output-dir outputs]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

_SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(_SRC))

from dwts_reproduction.config import load_paths  # noqa: E402
from dwts_reproduction.hashing import sha256_file  # noqa: E402
from dwts_reproduction.problem4.cases import (  # noqa: E402
    plot_case_rank_weekly,
    plot_case_score_weekly,
    sanitize_filename,
)
from dwts_reproduction.problem4.figures import (  # noqa: E402
    diff_contour_rank,
    heatmap_delta_rank,
    ribbon_survival,
)

TAG = "P4"


def _save_fig(path: Path) -> None:
    print(f"  rendered        {path.name}  ({path.stat().st_size} bytes)")


def main() -> int:
    parser = argparse.ArgumentParser(description="Problem 4 figure rendering")
    parser.add_argument("--output-dir", default="outputs", help="where P4 artifacts go")
    args = parser.parse_args()

    paths = load_paths()
    out_dir = (paths.repo_root / args.output_dir).resolve()
    fig_dir = out_dir
    fig_dir.mkdir(parents=True, exist_ok=True)

    v1_summary = pd.read_csv(out_dir / "problem4_sim_summary_V1.csv")
    v2_summary = pd.read_csv(out_dir / "problem4_sim_summary_V2.csv")
    v1_weekly = pd.read_csv(out_dir / "problem4_case_weekly_V1.csv")
    v2_weekly = pd.read_csv(out_dir / "problem4_case_weekly_V2.csv")
    print(
        f"[plot_problem4] v1_summary rows={len(v1_summary)}  "
        f"v2_summary rows={len(v2_summary)}  "
        f"v1_case_weekly rows={len(v1_weekly)}  v2_case_weekly rows={len(v2_weekly)}"
    )

    written: list[str] = []

    # ---- composite charts
    written.append(heatmap_delta_rank(v1_summary, fig_dir / "8_V1_plot1.png").name)
    _save_fig(fig_dir / "8_V1_plot1.png")
    written.append(
        diff_contour_rank(v1_summary, fig_dir / "8_fig_diff_contour_avg_rank_S3_minus_S1.png").name
    )
    _save_fig(fig_dir / "8_fig_diff_contour_avg_rank_S3_minus_S1.png")
    written.append(
        ribbon_survival(v2_summary, fig_dir / "8_fig_ribbon_survival_by_archetype.png").name
    )
    _save_fig(fig_dir / "8_fig_ribbon_survival_by_archetype.png")

    # ---- V1 case trends (legacy naming, parity outputs — not paper-embedded)
    if not v1_weekly.empty:
        for (season, name), g in v1_weekly.groupby(["season", "celebrity_name"]):
            fname = f"fig_sim_trend_{int(season)}_{sanitize_filename(name)}.png"
            plot_case_rank_weekly(g, int(season), name, fig_dir / fname)
            written.append(fname)
            _save_fig(fig_dir / fname)

    # ---- V2 case trends (paper embeds these with the "8_" prefix)
    if not v2_weekly.empty:
        for (season, name), g in v2_weekly.groupby(["season", "celebrity_name"]):
            fname = f"8_fig_sim2_trend_{int(season)}_{sanitize_filename(name)}.png"
            plot_case_score_weekly(g, int(season), name, fig_dir / fname)
            written.append(fname)
            _save_fig(fig_dir / fname)

    manifest = {
        "track": TAG,
        "source_tables": {
            "v1_summary": "problem4_sim_summary_V1.csv",
            "v2_summary": "problem4_sim_summary_V2.csv",
            "v1_case_weekly": "problem4_case_weekly_V1.csv",
            "v2_case_weekly": "problem4_case_weekly_V2.csv",
            "run_manifest": "problem4_run_manifest_P4.json",
        },
        "files": {name: sha256_file(fig_dir / name) for name in sorted(written)},
    }
    manifest_path = out_dir / f"problem4_fig_manifest_{TAG}.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"  figure manifest {manifest_path.name}  ({len(written)} files)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
