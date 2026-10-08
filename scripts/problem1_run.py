#!/usr/bin/env python3
"""Fit latent fan-support models and save weekly posterior summaries (P/R)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(_SRC))

from dwts_reproduction.config import load_paths  # noqa: E402
from dwts_reproduction.preprocess import build_all_tables  # noqa: E402
from dwts_reproduction.problem1 import (  # noqa: E402
    build_event_tables,
    build_problem1_panel,
    build_train_weeks,
    compute_cumulative_consistency,
    evaluate_top1_accuracy,
    fit_pooled_softmin,
    infer_all_weekly_fan_support,
    load_problem1_config,
    s_bar,
    summarize_posterior,
)
from dwts_reproduction.problem1.config import Problem1Config  # noqa: E402
from dwts_reproduction.problem1.track_r import (  # noqa: E402
    fit_integrated_marginal,
    fit_sensitivity,
)


def _save_csv(df, path: Path, label: str) -> None:
    df.to_csv(path, index=False)
    print(f"  {label:<34} {path.name}  ({len(df):>5} rows)")


def _save_npz(path: Path, fit, *, float64: bool = False) -> None:
    import numpy as np

    dtype = np.float64 if float64 else np.float32
    np.savez_compressed(
        path,
        beta=fit.beta.astype(dtype),
        bias=dtype(fit.bias),
        u=fit.u.astype(dtype),
        loss_history=np.asarray(fit.loss_history, dtype=np.float64),
    )


def _run_track_p(args: argparse.Namespace, paths, output_dir: Path):
    """Track P: pooled softmin fit + Dirichlet IS posterior (paper-faithful)."""
    config_path = (paths.repo_root / args.config).resolve()
    config = load_problem1_config(config_path)
    print(f"[problem1 {config.era_mode}] building tables from {paths.raw_data_csv.name}")

    tables = build_all_tables(paths.raw_data_csv)
    warnings: list[str] = []
    panel = build_problem1_panel(tables, config.era_mode, warnings)
    train_weeks = build_train_weeks(panel)
    print(f"  panel rows      {len(panel):>5} (reference 4199)")
    print(f"  train weeks     {len(train_weeks):>5} (reference 218)")
    for w in warnings:
        print(f"  [warn] {w}")

    fit = fit_pooled_softmin(panel, train_weeks, config)
    print(
        f"  fitted beta     {[round(float(v), 4) for v in fit.beta]}"
        f"  bias {fit.bias:.4f}  final loss {fit.loss_history[-1]:.4f}"
    )
    extra = {}
    return panel, train_weeks, config, fit, tables, extra


def _run_track_r(args: argparse.Namespace, paths, output_dir: Path):
    """Track R: integrated marginal-likelihood fit + MC diagnostics."""
    config = Problem1Config.for_track("R")
    print(f"[problem1 {config.era_mode}] building tables from {paths.raw_data_csv.name}")

    tables = build_all_tables(paths.raw_data_csv)
    warnings: list[str] = []
    panel = build_problem1_panel(tables, config.era_mode, warnings)
    train_weeks = build_train_weeks(panel)
    print(f"  panel rows      {len(panel):>5} (reference 4199)")
    print(f"  train weeks     {len(train_weeks):>5} (reference 218)")
    for w in warnings:
        print(f"  [warn] {w}")

    fit, diagnostics = fit_integrated_marginal(panel, train_weeks, config, mc_B=args.mc_B)
    print(
        f"  fitted beta     {[round(float(v), 4) for v in fit.beta]}"
        f"  bias {fit.bias:.4f}  final loss {fit.loss_history[-1]:.4f}"
    )
    print(
        f"  MC diagnostics  logL {diagnostics['mc_log_l']:.4f}"
        f"  se {diagnostics['mc_se']:.5f}"
        f"  rel_se {diagnostics['mc_se_relative']:.5f}"
        f"  ess {diagnostics['ess_mean']:.1f}"
    )

    sensitivity: dict[str, object] = {}
    seeds = tuple(int(config.seed) + i for i in range(1, args.sensitivity_seeds + 1))
    fit_Bs = (
        tuple(int(b) for b in args.sensitivity_fit_B.split(",")) if args.sensitivity_fit_B else ()
    )
    if seeds or fit_Bs:
        sensitivity = fit_sensitivity(
            panel, train_weeks, config, seeds=seeds, fit_Bs=fit_Bs, mc_B=args.mc_B
        )
        print(f"  sensitivity     {json.dumps(sensitivity['variants'], indent=4)}")
    extra = {"diagnostics": diagnostics, "sensitivity": sensitivity}
    return panel, train_weeks, config, fit, tables, extra


def main() -> int:
    parser = argparse.ArgumentParser(description="Problem 1 Track P / Track R pipeline")
    parser.add_argument(
        "--track",
        choices=["P", "R"],
        default="P",
        help="P = paper two-stage; R = integrated marginal",
    )
    parser.add_argument(
        "--output-dir", default="results/tables", help="where track-tagged artifacts go"
    )
    parser.add_argument(
        "--config",
        default="configs/problem1.yaml",
        help="Problem1Config YAML (Track P only; Track R uses Problem1Config.for_track('R'))",
    )
    parser.add_argument(
        "--mc-B",
        type=int,
        default=None,
        help="Track R: Dirichlet draws per choice set in the MC fit (default: config.B)",
    )
    parser.add_argument(
        "--sensitivity-seeds",
        type=int,
        default=0,
        help="Track R: number of extra seeds to refit for stability (0 = off)",
    )
    parser.add_argument(
        "--sensitivity-fit-B",
        default="",
        help="Track R: comma-separated fit sample counts to refit, e.g. '400,600'",
    )
    args = parser.parse_args()

    paths = load_paths()
    output_dir = (paths.repo_root / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.track == "P":
        panel, train_weeks, config, fit, tables, extra = _run_track_p(args, paths, output_dir)
    else:
        panel, train_weeks, config, fit, tables, extra = _run_track_r(args, paths, output_dir)
    tag = args.track

    posterior = infer_all_weekly_fan_support(panel, fit, config)
    by_week, by_season, accuracy = evaluate_top1_accuracy(panel, posterior, train_weeks)
    summary = summarize_posterior(posterior, accuracy)

    event_tables = build_event_tables(panel, fit, tables.elim_events, config)
    cum_consistency = compute_cumulative_consistency(
        event_tables["event_long"], event_tables["event_table"]
    )
    summary["s_bar"] = s_bar(cum_consistency)
    summary["n_panel_rows"] = int(len(panel))
    summary["n_train_weeks"] = int(len(train_weeks))
    summary["track"] = tag
    if extra:
        summary.update(extra)

    print("  metrics:")
    for key, value in summary.items():
        print(f"    {key:<26} {value}")

    # ---- write track-tagged outputs --------------------------------------
    _save_csv(panel, output_dir / f"problem1_panel_{tag}.csv", "panel")
    _save_csv(train_weeks, output_dir / f"problem1_train_weeks_{tag}.csv", "train_weeks")
    _save_csv(posterior, output_dir / f"problem1_posterior_summary_{tag}.csv", "posterior_summary")
    _save_csv(by_week, output_dir / f"problem1_top1_by_week_{tag}.csv", "top1_by_week")
    _save_csv(by_season, output_dir / f"problem1_top1_by_season_{tag}.csv", "top1_by_season")
    _save_csv(
        event_tables["event_table"],
        output_dir / f"problem1_event_table_{tag}.csv",
        "event_table",
    )
    _save_csv(
        event_tables["event_long"],
        output_dir / f"problem1_event_long_{tag}.csv",
        "event_long",
    )
    _save_csv(
        cum_consistency,
        output_dir / f"problem1_cumulative_consistency_{tag}.csv",
        "cum_consistency",
    )

    fit_arrays = output_dir / f"problem1_fit_arrays_{tag}.npz"
    _save_npz(fit_arrays, fit, float64=(tag == "R"))
    fit_meta = output_dir / f"problem1_fit_meta_{tag}.json"
    fit_meta.write_text(
        json.dumps(fit.as_dict(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    summary_path = output_dir / f"problem1_summary_{tag}.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"  fit_arrays_{tag}.npz ({fit_arrays.stat().st_size / 1e6:.1f} MB)")
    print(f"  fit_meta_{tag}.json / problem1_summary_{tag}.json")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
