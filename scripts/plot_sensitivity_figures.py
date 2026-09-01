#!/usr/bin/env python3
"""Render the paper's Figure 10 sensitivity panels from saved source tables.

Only reads the CSVs written by ``scripts/sensitivity_run.py`` (and that run's
manifest); it never re-runs inference, satisfying the "figures only from saved
source tables and run manifests" rule.  The three panels match the paper's
Figure 10(a)-(c):

- ``10_stability_scatter.png`` (P-091): scenario vs baseline ``p_mean``.
- ``10_tornado_pcp_mean.png`` (P-092): relative range of ``pcp_mean`` per family.
- ``10_A1_line_pcp_mean_by_kappa.png`` (P-093): ``pcp_mean`` vs ``tau``, by kappa.

Writes a figure manifest JSON (``sensitivity_fig_manifest_SA.json``) listing each
panel, its traceability id, source tables, and output hash.

Usage:
    python scripts/plot_sensitivity_figures.py [--output-dir outputs]
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(_SRC))

from dwts_reproduction.config import load_paths  # noqa: E402
from dwts_reproduction.hashing import sha256_file  # noqa: E402
from dwts_reproduction.sensitivity.viz import (  # noqa: E402
    line_plot_tau_kappa,
    load_csv,
    stability_scatter,
    tornado_plot,
)

TAG = "SA"


def main() -> int:
    parser = argparse.ArgumentParser(description="Render Figure 10 sensitivity panels")
    parser.add_argument("--output-dir", default="outputs", help="where SA artifacts live")
    args = parser.parse_args()

    started = datetime.now(UTC).isoformat(timespec="seconds")
    paths = load_paths()
    output_dir = (paths.repo_root / args.output_dir).resolve()

    def p(name: str) -> Path:
        return Path(output_dir) / f"{name}_{TAG}.csv"

    baseline_post = load_csv(p("sensitivity_baseline_post"))
    baseline_summary = load_csv(p("sensitivity_baseline_summary"))
    a1_summary = load_csv(p("sensitivity_A1_grid_summary"))
    summary_all = load_csv(p("sensitivity_summary_all"))

    all_post_files = [
        p("sensitivity_A1_grid_post"),
        p("sensitivity_A2_lambda_post"),
        p("sensitivity_A3_judge_post"),
        p("sensitivity_A4_leave_one_season_post"),
    ]

    figures: list[tuple[str, str, Path | None]] = [
        ("10_stability_scatter.png", "P-091", None),
        ("10_tornado_pcp_mean.png", "P-092", None),
        ("10_A1_line_pcp_mean_by_kappa.png", "P-093", None),
    ]

    out = stability_scatter(
        baseline_post, all_post_files, summary_all, output_dir, k=4, max_points=4000
    )
    figures[0] = (figures[0][0], figures[0][1], out)
    out = tornado_plot(summary_all, baseline_summary, output_dir, metric="pcp_mean")
    figures[1] = (figures[1][0], figures[1][1], out)
    out = line_plot_tau_kappa(a1_summary, output_dir, metric="pcp_mean")
    figures[2] = (figures[2][0], figures[2][1], out)

    manifest = {
        "track": TAG,
        "generated_from": "sensitivity_run_manifest_SA.json",
        "started_at": started,
        "ended_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "command": " ".join(sys.argv),
        "figures": {
            name: {
                "traceability_id": trace_id,
                "sha256": sha256_file(output_dir / name),
            }
            for name, trace_id, path in figures
            if path is not None
        },
    }
    manifest_path = output_dir / f"sensitivity_fig_manifest_{TAG}.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"  figure manifest {manifest_path.name}")
    for _name, trace_id, path in figures:
        status = f"OK {path.name}" if path else "skipped"
        print(f"    {trace_id:<8} {status}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
