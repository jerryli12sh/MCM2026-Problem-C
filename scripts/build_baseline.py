#!/usr/bin/env python3
"""Author the registered paper-output baseline.

Writes ``manifests/baseline.csv`` (canonical) and ``docs/BASELINE_PAPER_OUTPUTS.md``
(human-readable). Each row registers a paper metric/figure/table with its source location,
legacy producer, expected value or visual target, a *proposed* tolerance, and status.

Tolerances are proposed defaults and are flagged for owner approval — Phase 0 does not
approve them. Numeric tolerances are relative where the pipeline is deterministic;
figures/tables are structural/visual.

Columns: ``id, item, source_loc, legacy_producer, expected_value, proposed_tolerance, status``.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(_SRC))

from dwts_reproduction.config import load_paths  # noqa: E402

HEADER = [
    "id",
    "item",
    "source_loc",
    "legacy_producer",
    "expected_value",
    "proposed_tolerance",
    "status",
]


def b(
    idx: int,
    item: str,
    loc: str,
    producer: str,
    expected: str,
    tol: str,
    status: str = "registered",
) -> list[str]:
    return [f"B-{idx:02d}", item, loc, producer, expected, tol, status]


ROWS: list[list[str]] = [
    b(
        1,
        "Top-1 accuracy (XGBoost baseline)",
        "Problem 1 eval",
        "../src/xgb_baseline.py",
        "0.806554 (paper target; NOT reproducible from current legacy — legacy line 0.821101 week-mean / 0.817496 season-mean, see D-20260901-11 / C-07)",
        "rel 1e-3 (proposed)",
        "not-reproduced (paper target preserved)",
    ),
    b(
        2,
        "Top-1 accuracy (torch_model)",
        "Problem 1 eval",
        "../src/model.py; ../src/model_new.py",
        "0.952092",
        "rel 1e-3 (proposed)",
    ),
    b(
        3,
        "Top-1 accuracy (review rebuild)",
        "problem1_summary.json",
        "../review/problem1_rebuild/problem1_fan_support.py",
        "0.949541",
        "rel 1e-3 (proposed)",
    ),
    b(
        4,
        "Cumulative consistency S-bar",
        "Problem 1 eval",
        "../src/eval_metrics_viz.py",
        "0.78",
        "abs 0.02 (proposed)",
    ),
    b(
        5,
        "Mean PCP (weighted)",
        "problem1_summary.json",
        "../review/problem1_rebuild/outputs/problem1_summary.json",
        "0.6043",
        "abs 0.02 (proposed)",
    ),
    b(
        6,
        "Mean ESS ratio",
        "problem1_summary.json",
        "../review/problem1_rebuild/outputs/problem1_summary.json",
        "0.9625",
        "abs 0.02 (proposed)",
    ),
    b(
        7,
        "Mean CI relative width",
        "problem1_summary.json",
        "../review/problem1_rebuild/outputs/problem1_summary.json",
        "3.117",
        "abs 0.05 (proposed)",
    ),
    b(
        8,
        "Case-study table (|d|, Flip)",
        "Problem 2 table",
        "../src/sim_rank_trend_cases.py",
        "Jerry Rice 3.69/0.87; B.R.Cyrus 3.25/0.75; B.Palin 4.30/0.97; Bobby Bones 4.00/0.57; Tinashe 8.50/0.57; Vinny G. 9.88/0.33",
        "structural (proposed)",
    ),
    b(
        9,
        "V1 parameters",
        "Problem 4",
        "../src/season_simulator.py",
        "K=3; early/late split m=8",
        "exact (proposed)",
    ),
    b(
        10,
        "V2 parameters",
        "Problem 4",
        "../src/season_simulator2.py",
        "w_J=0.80 w_F=0.20 gamma=0.45 delta=1.35 mu=0.01 L in {2,3}",
        "exact (proposed)",
    ),
    b(
        11,
        "Problem 3 age coefficient (judge)",
        "Problem 3",
        "../src/dwts_pro_celeb_regression.py",
        "beta_Age ~ -0.04",
        "abs 0.02 (proposed)",
        "reproduced (got -0.0301/-0.0329/-0.0359, D-20260901-17)",
    ),
    b(
        12,
        "Problem 3 actor coefficients",
        "Problem 3",
        "../src/dwts_pro_celeb_regression.py",
        "judge W1 ~0.16; fan W6 -0.87",
        "abs 0.1 (proposed)",
        "direction-confirmed only (got +0.254 / -1.0221; |Δ|>0.1, D-20260901-17)",
    ),
    b(
        13,
        "Problem 3 partner tenure r",
        "Problem 3",
        "../src/dwts_pro_celeb_regression.py",
        "0.23",
        "abs 0.05 (proposed)",
        "direction-confirmed only (got 0.134; |Δ|>0.05, D-20260901-17)",
    ),
    b(
        14,
        "Problem 3 surprise beta1",
        "Problem 3",
        "../src/dwts_pro_celeb_regression.py",
        "0.34",
        "abs 0.05 (proposed)",
        "reproduced (got 0.3419, p<0.001, D-20260901-17)",
    ),
    b(
        15,
        "Model hyperparameters",
        "Problem 1",
        "../data/超参数.md",
        "tau=0.05 l2b=0.05 l2u=0.05 kappa=10 lr=0.02 steps=600 bs=32 B=1200",
        "exact (proposed)",
    ),
    b(
        16,
        "Preprocessing validation targets",
        "refactor prompt",
        "../review/srcs_0/dwts_preprocess.py",
        "see manifests/traceability_review.csv (R-01..R-19)",
        "exact (proposed)",
    ),
    b(
        17,
        "Paper figures",
        "paper_Latex/img/",
        "various ../src/*",
        "visual; see manifests/traceability_paper.csv",
        "visual (proposed)",
    ),
    b(
        18,
        "Problem 2 b2/phase output tables",
        "outputs/problem2_{b2_metrics,season_metrics,case_divergence,case_weekly_probs,phase_metrics,phase_claim_checks}_{P,R}.csv",
        "scripts/problem2_run.py",
        "136 phase rows (34 seasons x 4 mechanisms); b2 12 rows; see outputs/problem2_run_manifest_{P,R}.json for hashes",
        "structural (proposed)",
    ),
]


def _esc(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")


def main() -> int:
    paths = load_paths()

    csv_path = paths.manifest_dir / "baseline.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(HEADER)
        writer.writerows(ROWS)

    lines = [
        "# Registered paper-output baseline",
        "",
        "> Canonical data: `manifests/baseline.csv`. Generated by `scripts/build_baseline.py`.",
        "> Tolerances are **proposed** and flagged for owner approval; Phase 0 does not approve them.",
        "> Figures/tables are structural or visual targets; numeric items use relative/absolute tolerances.",
        "",
        f"| {' | '.join(HEADER)} |",
        f"|{'|'.join(['---'] * len(HEADER))}|",
    ]
    for row in ROWS:
        lines.append("| " + " | ".join(_esc(cell) for cell in row) + " |")
    lines.append("")

    (paths.repo_root / "docs" / "BASELINE_PAPER_OUTPUTS.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    print(f"wrote {len(ROWS)} baseline rows to {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
