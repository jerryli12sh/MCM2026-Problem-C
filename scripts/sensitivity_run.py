#!/usr/bin/env python3
"""Check temperature, concentration, regularization, judge-signal and season stability."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

_SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(_SRC))

from dwts_reproduction.config import load_paths  # noqa: E402
from dwts_reproduction.preprocess import build_all_tables  # noqa: E402
from dwts_reproduction.problem1.config import load_problem1_config  # noqa: E402
from dwts_reproduction.problem1.panel import build_problem1_panel, build_train_weeks  # noqa: E402
from dwts_reproduction.problem1.track_p import fit_pooled_softmin  # noqa: E402
from dwts_reproduction.sensitivity import (  # noqa: E402
    check_all,
    compute_metrics,
    run_a1_grid,
    run_a2_lambda_scan,
    run_a3_judge_transform,
    run_a4_leave_one_season_out,
)

TAG = "SA"

# Legacy sensitivity_analysis_a.py defaults (../src/sensitivity_analysis_a.py).
GRID_N = 15
TAU_MULT = [0.5, 1, 1.5, 2, 3, 4]
KAPPA_MULT = [0.5, 1, 2, 3, 5, 10]
LAMBDA_RATIOS = [0.25, 0.5, 1, 2, 4]
SOFTMAX_TEMPS = [0.5, 1, 2]
INCLUDE_PERCENTILE = True


def _save_csv(df: pd.DataFrame, path: Path, label: str) -> None:
    df.to_csv(path, index=False)
    print(f"  {label:<36} {path.name}  ({len(df):>5} rows)")


def _save_json(obj: object, path: Path, label: str) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"  {label:<36} {path.name}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Sensitivity analysis (A1-A4, Figure 10)")
    parser.add_argument("--output-dir", default="results/tables", help="where SA artifacts go")
    parser.add_argument(
        "--grid-n", type=int, default=GRID_N, help="A1: number of nearby (tau, kappa) grid points"
    )
    args = parser.parse_args()

    paths = load_paths()
    output_dir = (paths.repo_root / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    config = load_problem1_config()
    print(f"[sensitivity {config.era_mode}] building tables from {paths.raw_data_csv.name}")

    tables = build_all_tables(paths.raw_data_csv)
    warnings: list[str] = []
    panel = build_problem1_panel(tables, config.era_mode, warnings)
    train_weeks = build_train_weeks(panel)
    print(f"  panel rows      {len(panel):>5} (reference 4199)")
    print(f"  train weeks     {len(train_weeks):>5} (reference 218)")
    for w in warnings:
        print(f"  [warn] {w}")

    # ---- baseline (Track P posterior, B = config.B)
    t0 = time.monotonic()
    base_fit = fit_pooled_softmin(panel, train_weeks, config)
    baseline_post, baseline_week, baseline_summary = compute_metrics(
        panel, base_fit, train_weeks, config, "baseline"
    )
    print(
        f"[sensitivity] baseline pcp_mean={baseline_summary['pcp_mean']:.4f} "
        f"accuracy={baseline_summary['accuracy']:.4f} ({time.monotonic() - t0:.1f}s)"
    )

    # ---- A1: tau x kappa grid
    t0 = time.monotonic()
    tau_vals = [config.tau_train * m for m in TAU_MULT]
    kappa_vals = [config.kappa * m for m in KAPPA_MULT]
    a1_summary, a1_week, a1_post = run_a1_grid(
        panel,
        config,
        train_weeks,
        tau_vals=tau_vals,
        kappa_vals=kappa_vals,
        grid_n=args.grid_n,
        baseline_post=baseline_post,
    )
    print(f"[sensitivity] A1 grid {len(a1_summary)} scenarios ({time.monotonic() - t0:.1f}s)")

    # ---- A2: lambda ratio scan
    t0 = time.monotonic()
    a2_summary, a2_week, a2_post = run_a2_lambda_scan(
        panel, config, train_weeks, ratios=LAMBDA_RATIOS, baseline_post=baseline_post
    )
    print(
        f"[sensitivity] A2 lambda scan {len(a2_summary)} scenarios ({time.monotonic() - t0:.1f}s)"
    )

    # ---- A3: judge transforms
    t0 = time.monotonic()
    a3_summary, a3_week, a3_post = run_a3_judge_transform(
        tables,
        config,
        temperatures=SOFTMAX_TEMPS,
        include_percentile=INCLUDE_PERCENTILE,
        baseline_post=baseline_post,
    )
    print(
        f"[sensitivity] A3 judge transform {len(a3_summary)} scenarios ({time.monotonic() - t0:.1f}s)"
    )

    # ---- A4: leave-one-season-out
    t0 = time.monotonic()
    a4_summary, a4_week, a4_post = run_a4_leave_one_season_out(
        panel, config, baseline_post=baseline_post
    )
    print(
        f"[sensitivity] A4 leave-one-season-out {len(a4_summary)} scenarios ({time.monotonic() - t0:.1f}s)"
    )

    # ---- combine, claim checks, write outputs
    # baseline_summary is a dict; promote it to a one-row frame for the all-table.
    baseline_row = pd.DataFrame([baseline_summary])
    summary_all = pd.concat(
        [baseline_row, a1_summary, a2_summary, a3_summary, a4_summary],
        ignore_index=True,
    )
    claims = check_all(summary_all, baseline_row, a1_summary)
    print(f"  claim checks: {len(claims)} rows  pass={int((claims['status'] == 'pass').sum())}")

    _save_csv(baseline_week, output_dir / f"sensitivity_baseline_week_{TAG}.csv", "baseline_week")
    _save_csv(
        baseline_row, output_dir / f"sensitivity_baseline_summary_{TAG}.csv", "baseline_summary"
    )
    _save_csv(a1_summary, output_dir / f"sensitivity_A1_grid_summary_{TAG}.csv", "A1_grid_summary")
    _save_csv(a1_week, output_dir / f"sensitivity_A1_grid_week_{TAG}.csv", "A1_grid_week")
    _save_csv(
        a2_summary, output_dir / f"sensitivity_A2_lambda_summary_{TAG}.csv", "A2_lambda_summary"
    )
    _save_csv(a2_week, output_dir / f"sensitivity_A2_lambda_week_{TAG}.csv", "A2_lambda_week")
    _save_csv(
        a3_summary, output_dir / f"sensitivity_A3_judge_summary_{TAG}.csv", "A3_judge_summary"
    )
    _save_csv(a3_week, output_dir / f"sensitivity_A3_judge_week_{TAG}.csv", "A3_judge_week")
    _save_csv(
        a4_summary,
        output_dir / f"sensitivity_A4_leave_one_season_summary_{TAG}.csv",
        "A4_leave_one_season_summary",
    )
    _save_csv(
        a4_week,
        output_dir / f"sensitivity_A4_leave_one_season_week_{TAG}.csv",
        "A4_leave_one_season_week",
    )
    _save_csv(summary_all, output_dir / f"sensitivity_summary_all_{TAG}.csv", "summary_all")
    _save_csv(claims, output_dir / f"sensitivity_claims_{TAG}.csv", "claim_checks")

    run_config = {
        "track": TAG,
        "tau_vals": tau_vals,
        "kappa_vals": kappa_vals,
        "grid_n": args.grid_n,
        "lambda_ratios": LAMBDA_RATIOS,
        "softmax_temperatures": SOFTMAX_TEMPS,
        "include_percentile": INCLUDE_PERCENTILE,
        "seed": config.seed,
        "n_weeks": int(len(baseline_week)),
        "baseline_pcp_mean": float(baseline_summary["pcp_mean"]),
        "baseline_accuracy": float(baseline_summary["accuracy"]),
    }
    _save_json(run_config, output_dir / f"sensitivity_run_config_{TAG}.json", "run_config")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
