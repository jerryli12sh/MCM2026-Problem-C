#!/usr/bin/env python3
"""Evaluate within-season baselines, field-size effects, and ranking gaps."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

_SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(_SRC))

from dwts_reproduction.config import load_paths  # noqa: E402
from dwts_reproduction.preprocess import build_all_tables  # noqa: E402
from dwts_reproduction.problem1 import (  # noqa: E402
    accuracy_by_season,
    build_problem1_panel,
    build_train_weeks,
    crowded_field_from_posterior,
    evaluate_inseason_accuracy,
    fit_pooled_softmin,
    infer_all_weekly_fan_support,
    load_problem1_config,
    quadratic_fit_with_ci,
    ranking_gap_frame,
)

TAG = "P1E"


def _save_csv(df, path: Path, label: str) -> None:
    df.to_csv(path, index=False)
    print(f"  {label:<34} {path.name}  ({len(df):>5} rows)")


def main() -> int:
    parser = argparse.ArgumentParser(description="Problem 1 evaluation extras")
    parser.add_argument("--output-dir", default="results/tables", help="where P1E artifacts go")
    parser.add_argument(
        "--config", default="configs/problem1.yaml", help="Track P Problem1Config YAML"
    )
    parser.add_argument("--xgb-seed", type=int, default=42, help="base seed for the xgb line")
    args = parser.parse_args()

    paths = load_paths()
    output_dir = (paths.repo_root / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    config_path = (paths.repo_root / args.config).resolve()
    config = load_problem1_config(config_path)
    print(f"[problem1_extras] Track P config ({config.era_mode} era) + saved posterior")

    tables = build_all_tables(paths.raw_data_csv)
    warnings: list[str] = []
    panel = build_problem1_panel(tables, config.era_mode, warnings)
    train_weeks = build_train_weeks(panel)
    print(f"  panel rows      {len(panel):>5} (reference 4199)")
    print(f"  train weeks     {len(train_weeks):>5} (reference 218)")
    for w in warnings:
        print(f"  [warn] {w}")

    # ---- Track P posterior summary (saved table used by P-033/P-035/P-025/P-037)
    fit = fit_pooled_softmin(panel, train_weeks, config)
    posterior_summary = infer_all_weekly_fan_support(panel, fit, config)

    # ---- In-season baselines (P-027 / P-029)
    xgb_by_week = evaluate_inseason_accuracy(
        panel,
        "xgb",
        seed=args.xgb_seed,
        kappa=config.kappa,
        tau_like=config.tau_like,
        B=config.B,
    )
    torch_by_week = evaluate_inseason_accuracy(
        panel,
        "torch",
        config=config,
    )
    inseason_by_week = pd.concat([xgb_by_week, torch_by_week], ignore_index=True)
    inseason_by_season = accuracy_by_season(inseason_by_week)
    xgb_overall = float(xgb_by_week["accuracy"].mean())
    torch_overall = float(torch_by_week["accuracy"].mean())
    print(f"  xgb overall acc {xgb_overall:.6f}  (paper B-01 0.806554)")
    print(f"  torch overall   {torch_overall:.6f}  (visual line, no registered target)")

    # ---- P-033: PCP vs alive-set size
    crowded = crowded_field_from_posterior(posterior_summary)
    print(
        f"  crowded field   {len(crowded)} season-weeks, "
        f"mean pcp_weighted {crowded['pcp_weighted'].mean():.4f}"
    )

    # ---- P-035: ranking gap + quadratic fit
    gap = ranking_gap_frame(tables.weekly, posterior_summary)
    fit_res = quadratic_fit_with_ci(
        gap["result_minus_judge"].to_numpy(), gap["audience_rank"].to_numpy(), order=2
    )
    print(f"  ranking gap     n={fit_res.n}  R^2={fit_res.r_squared:.4f}  (paper claim >0.6)")

    # ---- P-025 / P-037: heatmap source tables (S8 p_mean, S21 ci_rel_width)
    s8 = posterior_summary[posterior_summary["season"].eq(8)].copy()
    s21 = posterior_summary[posterior_summary["season"].eq(21)].copy()

    # ---- summary json
    summary = {
        "track": TAG,
        "n_panel_rows": int(len(panel)),
        "n_train_weeks": int(len(train_weeks)),
        "inseason_xgb_overall_accuracy": xgb_overall,
        "inseason_torch_overall_accuracy": torch_overall,
        "paper_b01_xgb_target": 0.806554,
        "b01_rel_error": abs(xgb_overall - 0.806554) / 0.806554,
        "ranking_gap_n": int(fit_res.n),
        "ranking_gap_r_squared": fit_res.r_squared,
        "ranking_gap_claim_r2_gt_0_6": bool(fit_res.r_squared > 0.6),
        "ranking_gap_coeffs": fit_res.coeffs.tolist(),
        "crowded_mean_pcp_weighted": float(crowded["pcp_weighted"].mean()),
        "crowded_mean_pcp_unweighted": float(crowded["pcp_unweighted"].mean()),
        "s8_heatmap_metric": "p_mean",
        "s21_heatmap_metric": "ci_rel_width",
    }
    print("  metrics:")
    for key, value in summary.items():
        print(f"    {key:<40} {value}")

    # ---- write outputs
    _save_csv(
        inseason_by_week,
        output_dir / f"problem1_extras_inseason_by_week_{TAG}.csv",
        "inseason_by_week",
    )
    _save_csv(
        inseason_by_season,
        output_dir / f"problem1_extras_inseason_by_season_{TAG}.csv",
        "inseason_by_season",
    )
    _save_csv(crowded, output_dir / f"problem1_extras_crowded_field_{TAG}.csv", "crowded_field")
    _save_csv(gap, output_dir / f"problem1_extras_ranking_gap_{TAG}.csv", "ranking_gap")
    _save_csv(s8, output_dir / f"problem1_extras_s8_heatmap_{TAG}.csv", "s8_heatmap")
    _save_csv(s21, output_dir / f"problem1_extras_s21_heatmap_{TAG}.csv", "s21_heatmap")

    fit_json = output_dir / f"problem1_extras_ranking_gap_fit_{TAG}.json"
    fit_json.write_text(
        json.dumps(
            {
                "order": int(fit_res.order),
                "coeffs": fit_res.coeffs.tolist(),
                "r_squared": float(fit_res.r_squared),
                "n": int(fit_res.n),
                "ci_method": "polyfit(cov=True); 1.96 * sqrt(diag(V cov V^T))",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    summary_path = output_dir / f"problem1_extras_summary_{TAG}.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"  fit json        {fit_json.name} / summary {summary_path.name}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
