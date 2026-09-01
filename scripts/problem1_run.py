#!/usr/bin/env python3
"""Run the Problem 1 Track P pipeline end to end.

Builds the contestant-week panel, fits the pooled softmin popularity prior, draws
the weekly Dirichlet + importance-sampling fan-support posteriors, evaluates
top-1 reconstruction on training weeks, and reproduces the paper's cumulative
consistency ``S_bar`` — writing every artifact under ``outputs/`` with a ``_P``
track tag plus a run manifest recording inputs, config, seed, git commit, and
output hashes.

Usage:
    python scripts/problem1_run.py [--output-dir outputs] [--config configs/problem1.yaml]
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(_SRC))

from dwts_reproduction.config import load_paths  # noqa: E402
from dwts_reproduction.hashing import sha256_file  # noqa: E402
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
from dwts_reproduction.run_manifest import RunManifest  # noqa: E402


def _git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _environment() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "numpy": _import_version("numpy"),
        "pandas": _import_version("pandas"),
        "scipy": _import_version("scipy"),
        "platform": platform.platform(),
    }


def _import_version(name: str) -> str:
    try:
        mod = __import__(name)
        return getattr(mod, "__version__", "unknown")
    except ImportError:
        return "not-installed"


def _save_csv(df, path: Path, label: str) -> None:
    df.to_csv(path, index=False)
    print(f"  {label:<34} {path.name}  ({len(df):>5} rows)")


def _save_npz(path: Path, fit) -> None:
    import numpy as np

    np.savez_compressed(
        path,
        beta=fit.beta.astype(np.float32),
        bias=np.float32(fit.bias),
        u=fit.u.astype(np.float32),
        loss_history=np.asarray(fit.loss_history, dtype=np.float64),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Problem 1 Track P pipeline")
    parser.add_argument("--output-dir", default="outputs", help="where track-tagged artifacts go")
    parser.add_argument(
        "--config", default="configs/problem1.yaml", help="Problem1Config YAML (Track P defaults)"
    )
    args = parser.parse_args()

    started = datetime.now(UTC).isoformat(timespec="seconds")
    paths = load_paths()
    output_dir = (paths.repo_root / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
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

    print("  metrics:")
    for key, value in summary.items():
        print(f"    {key:<26} {value}")

    # ---- write track-tagged outputs --------------------------------------
    _save_csv(panel, output_dir / "problem1_panel_P.csv", "panel")
    _save_csv(train_weeks, output_dir / "problem1_train_weeks_P.csv", "train_weeks")
    _save_csv(posterior, output_dir / "problem1_posterior_summary_P.csv", "posterior_summary")
    _save_csv(by_week, output_dir / "problem1_top1_by_week_P.csv", "top1_by_week")
    _save_csv(by_season, output_dir / "problem1_top1_by_season_P.csv", "top1_by_season")
    _save_csv(event_tables["event_table"], output_dir / "problem1_event_table_P.csv", "event_table")
    _save_csv(event_tables["event_long"], output_dir / "problem1_event_long_P.csv", "event_long")
    _save_csv(
        cum_consistency, output_dir / "problem1_cumulative_consistency_P.csv", "cum_consistency"
    )

    fit_arrays = output_dir / "problem1_fit_arrays_P.npz"
    _save_npz(fit_arrays, fit)
    fit_meta = output_dir / "problem1_fit_meta_P.json"
    fit_meta.write_text(
        json.dumps(fit.as_dict(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    summary_path = output_dir / "problem1_summary_P.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"  fit_arrays_P.npz ({fit_arrays.stat().st_size / 1e6:.1f} MB)")
    print("  fit_meta_P.json / problem1_summary_P.json")

    ended = datetime.now(UTC).isoformat(timespec="seconds")
    outputs = {
        p.relative_to(output_dir).as_posix(): sha256_file(p)
        for p in output_dir.glob("problem1_*_P.*")
    }
    manifest = RunManifest(
        track="P",
        config_path=config_path.relative_to(paths.repo_root).as_posix(),
        input_manifest_sha256=sha256_file(paths.manifest_dir / "input_manifest.sha256"),
        git_commit=_git_commit(),
        environment=_environment(),
        seeds={"seed": config.seed},
        command=" ".join(sys.argv),
        started_at=started,
        ended_at=ended,
        status="success",
        outputs=outputs,
    )
    manifest_path = output_dir / "problem1_run_manifest_P.json"
    manifest.write(manifest_path)
    print(
        f"  run manifest    {manifest_path.name}  (inputs sha {manifest.input_manifest_sha256[:12]}…)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
