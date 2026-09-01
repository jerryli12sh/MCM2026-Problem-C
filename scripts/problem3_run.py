#!/usr/bin/env python3
"""Problem 3 survival-determinant analysis run (Track P).

Reproduces the three sub-analyses on the registered ``data/data_3.csv`` input:

- Demographic divergence (P-058..P-061): the faithful port of the legacy
  ``dwts_pro_celeb_regression.py`` OLS pipeline (base + season-FE spec, HC3,
  incremental R2, forward-CV) plus the paper's exact Eq. (demo_model)
  coefficients with ``Other`` as the industry reference.
- Professional-partner effects (P-062..P-066): H_abil/H_exp, partner-FE model,
  trait correlations, per-partner FE.
- Surprise/growth dynamics (P-067..P-071): S/G construction at t=W6 (primary)
  and t=final (late), linear + quadratic fits, claim checks.

Figures are NOT drawn here — ``scripts/plot_problem3_figures.py`` reads the
saved CSVs/JSON so every chart is backed by a registered source table.

Usage:
    python scripts/problem3_run.py [--output-dir outputs]
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import warnings
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

_SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(_SRC))

from dwts_reproduction.config import load_paths  # noqa: E402
from dwts_reproduction.hashing import sha256_file  # noqa: E402
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
from dwts_reproduction.run_manifest import RunManifest  # noqa: E402

TAG = "P3"

# Legacy OLS baseline (from the registered legacy run of
# ``../src/dwts_pro_celeb_regression.py`` on the same data_3.csv).
LEGACY_REFERENCE_R2 = {
    "placement_z": 0.054987,
    "judge_w1": 0.227111,
    "judge_w6": 0.275234,
    "judge_w11": 0.310374,
    "fan_w1": 0.155339,
    "fan_w6": 0.155171,
    "fan_final": 0.111478,
}


def _git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        )
        return out.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _import_version(name: str) -> str:
    try:
        mod = __import__(name)
        return getattr(mod, "__version__", "unknown")
    except ImportError:
        return "not-installed"


def _environment() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "numpy": _import_version("numpy"),
        "pandas": _import_version("pandas"),
        "scipy": _import_version("scipy"),
        "statsmodels": _import_version("statsmodels"),
        "sklearn": _import_version("sklearn"),
        "platform": platform.platform(),
    }


def _save_csv(df: pd.DataFrame, path: Path, label: str) -> None:
    df.to_csv(path, index=False)
    print(f"  {label:<32} {path.name}  ({len(df):>5} rows)")


def _save_json(obj: object, path: Path, label: str) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"  {label:<32} {path.name}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Problem 3 survival determinants")
    parser.add_argument("--output-dir", default="outputs", help="where P3 artifacts go")
    args = parser.parse_args()

    started = datetime.now(UTC).isoformat(timespec="seconds")
    paths = load_paths()
    output_dir = (paths.repo_root / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # The partner-FE model has singleton-partner rows whose HC3 leverage is 1
    # (statsmodels warns "divide by zero" in het_scale).  Coefficients are
    # unaffected; the robust SEs of those rows are not used in any claim.  The
    # caveat is recorded in D-20260901-17 and the summary JSON.
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

    # ---- legacy parity check
    parity_rows = []
    for _, row in ols_summary[ols_summary["spec"].eq("base")].iterrows():
        expected = LEGACY_REFERENCE_R2.get(row["outcome"])
        if expected is None:
            continue
        parity_rows.append(
            {
                "outcome": row["outcome"],
                "legacy_r2": expected,
                "repo_r2": float(row["R2"]),
                "abs_diff": abs(float(row["R2"]) - expected),
                "within_1e-4": abs(float(row["R2"]) - expected) < 1e-4,
            }
        )
    parity = pd.DataFrame(parity_rows)
    print(f"  legacy R2 parity: {int(parity['within_1e-4'].sum())}/{len(parity)} within 1e-4")

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
        "legacy_r2_parity_within_1e-4": bool(parity["within_1e-4"].all()),
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
    _save_csv(parity, output_dir / f"problem3_legacy_parity_{TAG}.csv", "legacy_parity")
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

    ended = datetime.now(UTC).isoformat(timespec="seconds")
    outputs = {
        p.relative_to(output_dir).as_posix(): sha256_file(p)
        for p in output_dir.glob(f"problem3_*_{TAG}.*")
    }
    manifest = RunManifest(
        track=TAG,
        config_path="configs/paths.yaml",
        input_manifest_sha256=sha256_file(paths.manifest_dir / "input_manifest.sha256"),
        git_commit=_git_commit(),
        environment=_environment(),
        seeds={"min_cat_count": 6, "cv_min_train_seasons": 3},
        command=" ".join(sys.argv),
        started_at=started,
        ended_at=ended,
        status="success",
        outputs=outputs,
    )
    manifest_path = output_dir / f"problem3_run_manifest_{TAG}.json"
    manifest.write(manifest_path)
    print(
        f"  run manifest    {manifest_path.name}  (inputs sha {manifest.input_manifest_sha256[:12]}…; "
        f"data_3 sha {sha256_file(paths.data3_csv)[:12]}…)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
