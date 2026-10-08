#!/usr/bin/env python3
"""Simulate five voting schemes across 34 seasons using fitted Dirichlet fan draws."""

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
from dwts_reproduction.problem4.cases import build_case_summary, build_case_weekly  # noqa: E402
from dwts_reproduction.problem4.claims import check_all, shock_table  # noqa: E402
from dwts_reproduction.problem4.features import (  # noqa: E402
    V1_DEFAULTS,
    V2_DEFAULTS,
    load_pooled_fit_dict,
)
from dwts_reproduction.problem4.v1 import SimConfig as V1Config  # noqa: E402
from dwts_reproduction.problem4.v1 import load_inputs as load_inputs_v1  # noqa: E402
from dwts_reproduction.problem4.v1 import run_simulation as run_v1  # noqa: E402
from dwts_reproduction.problem4.v1 import summarize_results as summarize_v1  # noqa: E402
from dwts_reproduction.problem4.v2 import SimConfig as V2Config  # noqa: E402
from dwts_reproduction.problem4.v2 import run_simulation as run_v2  # noqa: E402
from dwts_reproduction.problem4.v2 import summarize_results as summarize_v2  # noqa: E402

TAG = "P4"

# All 34 aired seasons (the legacy simulators iterated every season present in
# the weekly table; the repo pins the season list for determinism).
SEASONS = list(range(1, 35))


def _save_csv(df: pd.DataFrame, path: Path, label: str) -> None:
    df.to_csv(path, index=False)
    print(f"  {label:<30} {path.name}  ({len(df):>5} rows)")


def _save_json(obj: object, path: Path, label: str) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"  {label:<30} {path.name}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Problem 4 mechanism simulators")
    parser.add_argument("--output-dir", default="results/tables", help="where P4 artifacts go")
    parser.add_argument("--n-sims", type=int, default=V1_DEFAULTS["n_sims"])
    args = parser.parse_args()

    paths = load_paths()
    output_dir = (paths.repo_root / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # The pooled fit is loaded once and shared by both simulators.  A
    # simulated season with a single active contestant logs a RuntimeWarning
    # from the Dirichlet draw; it is benign (no elimination happens) and is
    # recorded, not silenced.
    pooled_fit = load_pooled_fit_dict(
        output_dir / "problem1_fit_meta_P.json",
        output_dir / "problem1_fit_arrays_P.npz",
    )

    weekly, clean, archetypes = load_inputs_v1(
        paths.data_dir / "df_weekly.csv",
        paths.data_dir / "df_clean.csv",
        paths.data_dir / "contestant_archetypes.csv",
    )
    print(f"[problem4] weekly rows={len(weekly)}  archetypes rows={len(archetypes)}")

    # ---- V1 simulator (paper Mechanism I, schemes S1/S2/S3)
    v1_cfg = V1Config(**{**V1_DEFAULTS, "n_sims": args.n_sims})
    t0 = time.monotonic()
    v1_detail = run_v1(weekly, archetypes, clean, pooled_fit, v1_cfg, SEASONS)
    v1_elapsed = time.monotonic() - t0
    v1_summary = summarize_v1(v1_detail)
    print(
        f"[problem4] V1 detail rows={len(v1_detail)}  summary rows={len(v1_summary)}  "
        f"({v1_elapsed:.1f}s)"
    )

    # ---- V2 simulator (paper Mechanism II, schemes V4/V5)
    v2_cfg = V2Config(**{**V2_DEFAULTS, "n_sims": args.n_sims})
    t0 = time.monotonic()
    v2_detail = run_v2(weekly, archetypes, clean, pooled_fit, v2_cfg, SEASONS)
    v2_elapsed = time.monotonic() - t0
    v2_summary = summarize_v2(v2_detail)
    print(
        f"[problem4] V2 detail rows={len(v2_detail)}  summary rows={len(v2_summary)}  "
        f"({v2_elapsed:.1f}s)"
    )

    # ---- case studies, shock rates, claim checks
    v1_case_summary = build_case_summary(v1_detail, "V1")
    v2_case_summary = build_case_summary(v2_detail, "V2")
    v1_case_weekly = build_case_weekly(v1_detail, "V1")
    v2_case_weekly = build_case_weekly(v2_detail, "V2")
    v1_shock = shock_table(v1_detail, "V1")
    v2_shock = shock_table(v2_detail, "V2")
    claims = check_all(v1_detail, v2_detail)
    print(f"  claim checks: {len(claims)} rows  pass={int((claims['status'] == 'pass').sum())}")

    # ---- legacy regression check (V1 vs saved sim_summary.csv / sim_case_summary.csv)
    #
    # A 0.005 rank tolerance allows small Monte Carlo differences from the saved fit.
    legacy_summary = pd.read_csv(paths.reference_dir / "sim_summary.csv")
    merged = v1_summary.merge(
        legacy_summary, on=["scheme", "week", "archetype"], suffixes=("_repo", "_legacy")
    )
    tol = 5e-3
    max_abs = float((merged["avg_rank_repo"] - merged["avg_rank_legacy"]).abs().max())
    within = bool(max_abs <= tol)
    legacy_parity = {
        "target_rows": int(len(legacy_summary)),
        "matched_rows": int(len(merged)),
        "max_abs_avg_rank_diff": round(max_abs, 7),
        "tol": tol,
        "within_tol": within,
    }
    print(f"  V1 legacy parity: {legacy_parity}")

    # ---- summary JSON
    summary = {
        "track": TAG,
        "v1_schemes": list(V1_DEFAULTS["schemes"]),
        "v2_schemes": list(V2_DEFAULTS["schemes"]),
        "n_sims": args.n_sims,
        "seasons": SEASONS,
        "v1_detail_rows": int(len(v1_detail)),
        "v2_detail_rows": int(len(v2_detail)),
        "v1_summary_rows": int(len(v1_summary)),
        "v2_summary_rows": int(len(v2_summary)),
        "v1_elapsed_s": round(v1_elapsed, 1),
        "v2_elapsed_s": round(v2_elapsed, 1),
        "legacy_parity": legacy_parity,
        "claim_status": {
            str(claim_id): str(status)
            for claim_id, status in zip(claims["claim_id"], claims["status"], strict=True)
        },
        "simulation_scope": (
            "Fan shares are Dirichlet draws around the fitted support center; "
            "all schemes use the same fixed model parameters."
        ),
    }
    print("  claim checks (Track P):")
    for claim_id, status in summary["claim_status"].items():
        print(f"    {claim_id:<10} {status}")

    # ---- write outputs
    _save_csv(v1_summary, output_dir / "problem4_sim_summary_V1.csv", "v1_summary")
    _save_csv(v2_summary, output_dir / "problem4_sim_summary_V2.csv", "v2_summary")
    _save_csv(v1_case_summary, output_dir / "problem4_case_summary_V1.csv", "v1_case_summary")
    _save_csv(v2_case_summary, output_dir / "problem4_case_summary_V2.csv", "v2_case_summary")
    _save_csv(v1_case_weekly, output_dir / "problem4_case_weekly_V1.csv", "v1_case_weekly")
    _save_csv(v2_case_weekly, output_dir / "problem4_case_weekly_V2.csv", "v2_case_weekly")
    _save_csv(v1_shock, output_dir / "problem4_shock_rates_V1.csv", "v1_shock_rates")
    _save_csv(v2_shock, output_dir / "problem4_shock_rates_V2.csv", "v2_shock_rates")
    _save_csv(claims, output_dir / "problem4_claims_P4.csv", "claim_checks")
    _save_json(summary, output_dir / "problem4_summary_P4.json", "summary")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
