#!/usr/bin/env python3
"""Run the Problem 2 replay pipeline end to end for Track P or Track R.

The Problem 2 replay consumes a *serialized* Problem 1 fit
(``outputs/problem1_fit_meta_{track}.json`` + ``problem1_fit_arrays_{track}.npz``,
written by ``scripts/problem1_run.py``), rebuilds the matching contestant-week
panel from the fit's ``era_mode``, and reproduces the paper's Problem 2 outputs:

- ``season_rule_metrics`` — paper Eqs. 3-6 (override / reversal / fan-worst
  rates and their difference), with point and posterior-propagated values;
- ``case_divergence`` — paper Table 1 ``|d|`` / ``Flip`` for the six named
  controversy cases;
- ``case_weekly_probs`` — per-eligible-week elimination probabilities under the
  rank and percentage rules for each named case;
- ``b2_case_metrics`` — the bottom-2 + judges'-save reference metrics;
- ``mechanism_phase_metrics`` / ``phase_claim_checks`` — the mechanism phase
  diagram (paper Fig. 5) backing table and quantitative checks of the paper's
  and review's phase-diagram claims (P-056/P-057, R-040).

For Track P the run additionally compares against the registered reproduction
targets (paper Table 1 values in ``TABLE1_REFERENCE`` and the reference
``../data/metrics_b2_save.csv``) and reports the deviations in the summary.
Track R runs the same replay under the review's judge-signal mapping; its
numbers have no legacy reference and are reported without a comparison.

Artifacts are written under ``outputs/`` with a ``_P`` / ``_R`` tag and a run
manifest recording inputs, config source, seeds, git commit, and output hashes.

Usage:
    python scripts/problem2_run.py [--track P|R] [--output-dir outputs]
        [--fit-dir outputs] [--B-div 1200] [--B-flip 600] [--B-metrics 600]
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
from dwts_reproduction.problem2 import (  # noqa: E402
    B_DIVERGENCE,
    B_MECHANISM,
    TABLE1_CASES,
    TABLE1_REFERENCE,
    b2_case_metrics,
    build_replay_inputs,
    case_divergence,
    case_weekly_probs,
    config_from_fit,
    load_pooled_fit,
    mechanism_phase_metrics,
    phase_claim_checks,
    season_rule_metrics,
)
from dwts_reproduction.run_manifest import RunManifest  # noqa: E402

# Registered reproduction tolerance for the b2 reference CSV (B=600 MC):
# p_b2 / p_rev are Bernoulli proportions (0.002), p_rev_given_b2 a conditional
# proportion (0.005), and dE_T / dP_finals draw-aligned trajectory means
# (0.010 / 0.005).  Matches tests/test_problem2_replay.py.
B2_TOLERANCE = {
    "p_b2": 0.002,
    "p_rev": 0.002,
    "p_rev_given_b2": 0.005,
    "dE_T": 0.010,
    "dP_finals": 0.005,
}
# Paper Table 1 |d| / Flip reproduction tolerance (B-08).
ABS_D_TOL = 0.02
FLIP_TOL = 0.01


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


def _save_csv(df: pd.DataFrame, path: Path, label: str) -> None:
    df.to_csv(path, index=False)
    print(f"  {label:<32} {path.name}  ({len(df):>5} rows)")


def _compare_table1(got: pd.DataFrame) -> dict[str, object]:
    """Deviation report against the paper Table 1 reference values."""
    rows = []
    for _, row in got.iterrows():
        key = (int(row["season"]), row["celebrity_name"])
        ref_d, ref_flip = TABLE1_REFERENCE[key]
        rows.append(
            {
                "season": int(row["season"]),
                "celebrity_name": row["celebrity_name"],
                "abs_d_ref": ref_d,
                "abs_d_got": float(row["abs_d"]),
                "abs_d_dev": float(row["abs_d"]) - ref_d,
                "abs_d_within_tol": bool(abs(float(row["abs_d"]) - ref_d) <= ABS_D_TOL),
                "flip_ref": ref_flip,
                "flip_got": float(row["flip"]),
                "flip_dev": float(row["flip"]) - ref_flip,
                "flip_within_tol": bool(abs(float(row["flip"]) - ref_flip) <= FLIP_TOL),
                "flip_week_got": None if pd.isna(row["flip_week"]) else int(row["flip_week"]),
            }
        )
    return {"table": rows}


def _compare_b2(got: pd.DataFrame, ref: pd.DataFrame) -> dict[str, object]:
    """Deviation report against the reference ``metrics_b2_save.csv``."""
    ref = ref[ref["unit_type"] == "individual"].copy()
    key = ["season", "celebrity_name", "baseline_mode"]
    cols = ["p_b2", "p_rev_given_b2", "p_rev", "dE_T", "dP_finals"]
    merged = ref[key + cols].merge(got[key + cols], on=key, suffixes=("_ref", "_got"))
    dev = {c: float((merged[f"{c}_got"] - merged[f"{c}_ref"]).abs().max()) for c in cols}
    return {
        "n_compared": int(len(merged)),
        "max_abs_deviation": dev,
        "within_tolerance": {c: bool(dev[c] <= tol) for c, tol in B2_TOLERANCE.items()},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Problem 2 replay (Track P / Track R)")
    parser.add_argument(
        "--track",
        choices=["P", "R"],
        default="P",
        help="P = paper-faithful legacy replay; R = review-corrected official replay",
    )
    parser.add_argument("--output-dir", default="outputs", help="where track-tagged artifacts go")
    parser.add_argument(
        "--fit-dir",
        default="outputs",
        help="directory holding problem1_fit_meta/arrays_{track} (default: outputs)",
    )
    parser.add_argument(
        "--B-div", type=int, default=B_DIVERGENCE, help="draws for |d| (paper cell 20: 1200)"
    )
    parser.add_argument(
        "--B-flip", type=int, default=B_MECHANISM, help="draws for Flip (paper cell 29: 600)"
    )
    parser.add_argument(
        "--B-metrics",
        type=int,
        default=B_MECHANISM,
        help="draws for season metrics / case weekly probs / b2 metrics (reference: 600)",
    )
    parser.add_argument(
        "--B-phase",
        type=int,
        default=B_MECHANISM,
        help="draws for the mechanism phase diagram (paper Fig. 5; reference: 600)",
    )
    args = parser.parse_args()

    started = datetime.now(UTC).isoformat(timespec="seconds")
    paths = load_paths()
    output_dir = (paths.repo_root / args.output_dir).resolve()
    fit_dir = (paths.repo_root / args.fit_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    tag = args.track

    meta_path = fit_dir / f"problem1_fit_meta_{tag}.json"
    arrays_path = fit_dir / f"problem1_fit_arrays_{tag}.npz"
    if not (meta_path.exists() and arrays_path.exists()):
        print(
            f"error: serialized Problem 1 fit not found in {fit_dir}; "
            "run `python scripts/problem1_run.py --track <tag>` first",
            file=sys.stderr,
        )
        return 1

    fit = load_pooled_fit(meta_path, arrays_path)
    panel, train_weeks = build_replay_inputs(paths, fit)
    cfg = config_from_fit(fit, B=max(args.B_div, args.B_flip, args.B_metrics))
    print(f"[problem2 {tag}] replayed fit era_mode={fit.era_mode} model_type={fit.model_type}")
    print(f"  panel rows      {len(panel):>5}   train weeks {len(train_weeks):>5}")
    print(f"  beta            {[round(float(v), 4) for v in fit.beta]}  bias {fit.bias:.4f}")
    print(f"  B               div={args.B_div} flip={args.B_flip} metrics={args.B_metrics}")

    print("  computing season rule metrics (Eqs. 3-6)...")
    season_metrics = season_rule_metrics(panel, fit, cfg, B=args.B_div)
    print("  computing case divergence (Table 1 |d| / Flip)...")
    divergence = case_divergence(
        panel, fit, cfg, TABLE1_CASES, B_div=args.B_div, B_flip=args.B_flip
    )
    print("  computing per-week case probabilities...")
    weekly_parts = [
        case_weekly_probs(panel, fit, cfg, season, name, B=args.B_metrics)
        for season, name in TABLE1_CASES
    ]
    weekly = pd.concat(weekly_parts, ignore_index=True)
    print("  computing b2-save metrics...")
    b2 = b2_case_metrics(panel, fit, cfg, TABLE1_CASES, B=args.B_metrics)
    print("  computing mechanism phase diagram (paper Fig. 5)...")
    phase = mechanism_phase_metrics(panel, fit, B=args.B_phase, alpha=0.10)
    claim_checks = phase_claim_checks(phase)
    skipped = season_metrics.attrs.get("skipped_weeks", [])

    print("  writing track-tagged outputs:")
    _save_csv(season_metrics, output_dir / f"problem2_season_metrics_{tag}.csv", "season_metrics")
    _save_csv(divergence, output_dir / f"problem2_case_divergence_{tag}.csv", "case_divergence")
    _save_csv(weekly, output_dir / f"problem2_case_weekly_probs_{tag}.csv", "case_weekly_probs")
    _save_csv(b2, output_dir / f"problem2_b2_metrics_{tag}.csv", "b2_metrics")
    _save_csv(phase, output_dir / f"problem2_phase_metrics_{tag}.csv", "phase_metrics")
    _save_csv(
        claim_checks,
        output_dir / f"problem2_phase_claim_checks_{tag}.csv",
        "phase_claim_checks",
    )

    def _records(df: pd.DataFrame) -> list[dict[str, object]]:
        """Records with NaN/NaT -> None so JSON stays valid."""
        return df.astype(object).where(pd.notna(df), None).to_dict(orient="records")

    summary: dict[str, object] = {
        "track": tag,
        "era_mode": fit.era_mode,
        "model_type": fit.model_type,
        "n_season_metrics_rows": int(len(season_metrics)),
        "n_case_rows": int(len(divergence)),
        "n_weekly_rows": int(len(weekly)),
        "n_b2_rows": int(len(b2)),
        "n_phase_rows": int(len(phase)),
        "n_phase_claim_rows": int(len(claim_checks)),
        "phase_claim_checks": _records(claim_checks),
        "n_skipped_weeks": len(skipped),
        "skipped_weeks": [[int(s), int(w)] for s, w in skipped],
        "B_div": args.B_div,
        "B_flip": args.B_flip,
        "B_metrics": args.B_metrics,
        "B_phase": args.B_phase,
    }
    if tag == "P":
        summary["table1_comparison"] = _compare_table1(divergence)
        ref_b2 = pd.read_csv(paths.data_dir / "metrics_b2_save.csv")
        summary["b2_comparison"] = _compare_b2(b2, ref_b2)
        print("  Track P reproduction comparison:")
        t1 = summary["table1_comparison"]
        for r in t1["table"]:
            print(
                f"    {r['season']}/{r['celebrity_name']:<16} |d| {r['abs_d_got']:.4f} "
                f"(ref {r['abs_d_ref']}, ok={r['abs_d_within_tol']})  "
                f"Flip {r['flip_got']:.4f} (ref {r['flip_ref']}, ok={r['flip_within_tol']})"
            )
        b2c = summary["b2_comparison"]
        print(
            f"    b2 metrics vs reference: {b2c['n_compared']} rows, within_tol={b2c['within_tolerance']}"
        )
    else:
        print("  Track R: no legacy reference; structural outputs only.")
    print("  phase claim checks (paper P-056/P-057, review R-040):")
    for _, r in claim_checks.iterrows():
        supporting = "" if pd.isna(r.get("supporting")) else f"  {r['supporting']}"
        print(
            f"    {r['claim']:<48} mean_delta_y={r['mean_delta_y']:+.4f} "
            f"n={int(r['n_seasons'])}{supporting}".rstrip()
        )

    summary_path = output_dir / f"problem2_summary_{tag}.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"  summary         {summary_path.name}")

    ended = datetime.now(UTC).isoformat(timespec="seconds")
    outputs = {
        p.relative_to(output_dir).as_posix(): sha256_file(p)
        for p in output_dir.glob(f"problem2_*_{tag}.*")
    }
    manifest = RunManifest(
        track=tag,
        config_path=f"{meta_path.relative_to(paths.repo_root).as_posix()} "
        f"+ {arrays_path.relative_to(paths.repo_root).as_posix()} "
        f"(reconstructed via config_from_fit)",
        input_manifest_sha256=sha256_file(paths.manifest_dir / "input_manifest.sha256"),
        git_commit=_git_commit(),
        environment=_environment(),
        seeds={
            "fit_seed": fit.seed,
            "B_div": args.B_div,
            "B_flip": args.B_flip,
            "B_metrics": args.B_metrics,
            "B_phase": args.B_phase,
        },
        command=" ".join(sys.argv),
        started_at=started,
        ended_at=ended,
        status="success",
        outputs=outputs,
    )
    manifest_path = output_dir / f"problem2_run_manifest_{tag}.json"
    manifest.write(manifest_path)
    print(
        f"  run manifest    {manifest_path.name}  (inputs sha {manifest.input_manifest_sha256[:12]}…)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
