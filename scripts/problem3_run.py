#!/usr/bin/env python3
"""Estimate celebrity, partner-history, and surprise-growth associations."""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import pandas as pd

_SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(_SRC))

from dwts_reproduction.config import load_paths  # noqa: E402
from dwts_reproduction.problem3 import (  # noqa: E402
    LATE_TFINAL,
    PRIMARY_TW6,
    cv_table,
    engineer_features,
    extract_key_coefs,
    fit_all_ols,
    fit_growth_linear,
    fit_growth_quadratic,
    incremental_r2_table,
    judge_fan_supporting,
    load_data,
    paper_demo_model,
    predict_quadratic,
    surprise_claim_checks,
    surprise_growth_frame,
)

TAG = "P3"


def _save_csv(df: pd.DataFrame, path: Path, label: str) -> None:
    df.to_csv(path, index=False)
    print(f"  {label:<32} {path.name}  ({len(df):>5} rows)")


def _save_json(obj: object, path: Path, label: str) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"  {label:<32} {path.name}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Problem 3 survival determinants")
    parser.add_argument("--output-dir", default="results/tables", help="where P3 artifacts go")
    args = parser.parse_args()

    paths = load_paths()
    output_dir = (paths.repo_root / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # The partner-FE model has singleton-partner rows whose HC3 leverage is 1
    # (statsmodels warns "divide by zero" in het_scale).  Coefficients are
    # unaffected; the robust SEs of those rows are not used in any claim.  The
    # See docs/METHODS.md for the interpretation of singleton partner effects.
    warnings.filterwarnings(
        "ignore",
        message="divide by zero encountered in divide",
        category=RuntimeWarning,
    )

    # ---- input
    df = load_data(paths.data3_csv)
    eng = engineer_features(df)
    print(f"[problem3] data_3.csv rows={len(df)}  engineered rows={len(eng)}")

    # ---- demographic divergence (P-058..P-061)
    ols_summary, fitted = fit_all_ols(eng)
    key_base = extract_key_coefs(fitted, "base")
    key_fe = extract_key_coefs(fitted, "seasonFE")
    incr = incremental_r2_table(eng)
    cvf = cv_table(eng)
    demo = paper_demo_model(eng)

    # ---- partner effects (P-062..P-066)
    partner = judge_fan_supporting(eng)
    print(
        f"  partner FE summary {len(partner['fe_summary'])} rows; "
        f"per-partner FE {len(partner['fe_params'])} partners"
    )

    # ---- surprise/growth (P-067..P-071)
    frames = {}
    fits = {}
    for label, cfg in (("tw6", PRIMARY_TW6), ("tfinal", LATE_TFINAL)):
        frames[label] = surprise_growth_frame(eng, **cfg)
        fits[label] = {
            "linear": fit_growth_linear(frames[label]),
            "quadratic": fit_growth_quadratic(frames[label]),
        }
    claims_tw6 = surprise_claim_checks(frames["tw6"])
    claims_tfinal = surprise_claim_checks(frames["tfinal"])
    claims = pd.concat([claims_tw6, claims_tfinal], ignore_index=True)
    grid_tw6 = predict_quadratic(frames["tw6"], fits["tw6"]["quadratic"])
    print(
        f"  surprise t=W6 n={len(frames['tw6'])}  "
        f"beta1={fits['tw6']['linear'].coefs['S']:.4f}  "
        f"beta2={fits['tw6']['quadratic'].coefs['I(S ** 2)']:.4f}  "
        f"beta3={fits['tw6']['quadratic'].coefs['S:H_exp']:.4f}"
    )

    # ---- summary metrics
    demo_pick = demo[demo["term"].eq("celebrity_age_during_season")]
    age_judge = {
        o: float(demo_pick[demo_pick["outcome"].eq(o)]["coef"].iloc[0])
        for o in ("judge_w1", "judge_w6", "judge_w11")
    }
    actor = demo[demo["term"].str.contains("Actor/Actress", na=False)]
    actor_judge_w1 = float(actor[actor["outcome"].eq("judge_w1")]["coef"].iloc[0])
    actor_fan_w6 = float(actor[actor["outcome"].eq("fan_w6")]["coef"].iloc[0])
    r_hexp_judge = float(
        partner["corr"][
            partner["corr"]["trait"].eq("H_exp") & partner["corr"]["outcome"].eq("judge_w1")
        ]["r"].iloc[0]
    )
    b1 = fits["tw6"]["linear"].coefs["S"]
    b2 = fits["tw6"]["quadratic"].coefs["I(S ** 2)"]
    b3 = fits["tw6"]["quadratic"].coefs["S:H_exp"]

    summary = {
        "track": TAG,
        "n_data3_rows": int(len(df)),
        "placement_source": "official raw placement column",
        "partner_rate_denominator": "number of prior observed seasons",
        "paper_P059_age_judge": {k: round(v, 4) for k, v in age_judge.items()},
        "paper_P059_age_within_abs_0_02": bool(
            all(abs(v - (-0.04)) <= 0.02 for v in age_judge.values())
        ),
        "paper_P060_actor_judge_w1": round(actor_judge_w1, 4),
        "paper_P060_actor_fan_w6": round(actor_fan_w6, 4),
        "paper_P060_within_abs_0_1": bool(
            abs(actor_judge_w1 - 0.16) <= 0.1 and abs(actor_fan_w6 - (-0.87)) <= 0.1
        ),
        "paper_P064_r_Hexp_judge_w1": round(r_hexp_judge, 4),
        "paper_P064_within_abs_0_05": bool(abs(r_hexp_judge - 0.23) <= 0.05),
        "paper_P069_beta1_tw6": round(b1, 4),
        "paper_P069_within_abs_0_05": bool(abs(b1 - 0.34) <= 0.05),
        "paper_P070_beta2_gt_0": bool(b2 > 0),
        "paper_P071_beta3_gt_0": bool(b3 > 0),
        "partner_n_fe_partners": int(len(partner["fe_params"])),
    }
    print("  claim checks (Track P):")
    for key, value in summary.items():
        print(f"    {key:<40} {value}")

    # ---- write outputs
    _save_csv(ols_summary, output_dir / f"problem3_ols_summary_{TAG}.csv", "ols_summary")
    _save_csv(key_base, output_dir / f"problem3_key_coefs_base_{TAG}.csv", "key_coefs_base")
    _save_csv(key_fe, output_dir / f"problem3_key_coefs_seasonFE_{TAG}.csv", "key_coefs_fe")
    _save_csv(incr, output_dir / f"problem3_incremental_r2_{TAG}.csv", "incremental_r2")
    _save_csv(cvf, output_dir / f"problem3_cv_forward_{TAG}.csv", "cv_forward")
    _save_csv(demo, output_dir / f"problem3_demo_coefs_{TAG}.csv", "demo_coefs")
    _save_csv(
        partner["fe_summary"],
        output_dir / f"problem3_partner_fe_summary_{TAG}.csv",
        "partner_fe_summary",
    )
    _save_csv(
        partner["fe_params"],
        output_dir / f"problem3_partner_fe_params_{TAG}.csv",
        "partner_fe_params",
    )
    _save_csv(
        partner["corr"], output_dir / f"problem3_partner_correlations_{TAG}.csv", "partner_corr"
    )
    _save_csv(claims, output_dir / f"problem3_surprise_claims_{TAG}.csv", "surprise_claims")
    _save_csv(grid_tw6, output_dir / f"problem3_surprise_predict_grid_{TAG}.csv", "surprise_grid")
    for label in ("tw6", "tfinal"):
        _save_csv(
            frames[label],
            output_dir / f"problem3_surprise_frame_{label}_{TAG}.csv",
            f"surprise_frame_{label}",
        )
        for spec, fit in fits[label].items():
            _save_json(
                fit.to_dict(),
                output_dir / f"problem3_surprise_{spec}_{label}_{TAG}.json",
                f"surprise_{spec}_{label}",
            )
    summary_path = output_dir / f"problem3_summary_{TAG}.json"
    _save_json(summary, summary_path, "summary")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
