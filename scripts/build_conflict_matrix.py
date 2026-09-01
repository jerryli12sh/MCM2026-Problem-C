#!/usr/bin/env python3
"""Author the paper-vs-review conflict matrix.

Writes ``manifests/conflict_matrix.csv`` (canonical) and ``docs/CONFLICT_MATRIX.md``
(human-readable). Each conflict maps to a decision ID in ``docs/DECISIONS.md``. A conflict is
recorded, not silently resolved: Track P and Track R behaviors stay separate.

Columns: ``id, paper_stmt, review_stmt, legacy_evidence, track_P_behavior, track_R_behavior,
numerical_impact, conclusion_impact, decision_id``.
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
    "paper_stmt",
    "review_stmt",
    "legacy_evidence",
    "track_P_behavior",
    "track_R_behavior",
    "numerical_impact",
    "conclusion_impact",
    "decision_id",
]

ROWS: list[list[str]] = [
    [
        "C-01",
        "Problem 2 states Season 28+ uses Bottom-2 + judges save; Problem 1 does not state a per-season era mapping explicitly.",
        "Official problem-statement mapping: Seasons 1-2 rank, 3-27 percent, 28-34 rank (Bottom-2 ~28+).",
        "../src/model.py reportedly assigns era via season>=28 -> 'percent' (opposite direction); refactor prompt flags 'suspicious opposite direction'.",
        "Reproduce legacy era mapping in an explicit `era_mode='legacy'` variant to match paper numbers; default remains under audit until the legacy code is inspected.",
        "Use the official mapping (`era_mode='official'`).",
        "Changes which judge signal (rank vs percent) is used per season, altering j_metric and all downstream accuracies.",
        "Determines whether Track P can reproduce the reported top-1 accuracy at all.",
        "D-20260901-01 (suspected — pending legacy inspection)",
    ],
    [
        "C-02",
        "Track P is two-stage: fit q from eliminations, then condition weekly p on the same observed elimination.",
        "review_all.md: fit the integrated marginal P(Y|beta,u)=integral P(Y|p,J) Dirichlet(p|kappa q) dp to avoid using the elimination outcome twice.",
        "../src/posterior_uncertainty.py reweights weekly draws by the same softmin likelihood used to fit q.",
        "Two-stage procedure; reconstruction metrics labeled in-sample/explanatory.",
        "Integrated marginal-likelihood formulation; leakage-safe.",
        "Track P overstates internal consistency; Track R may reduce top-1 accuracy.",
        "Track P accuracy is internal validity, not independent prediction.",
        "D-20260901-02",
    ],
    [
        "C-03",
        "Paper writes a single temperature tau in both the penalized NLL and the posterior likelihood.",
        "Legacy uses two temperatures: tau_train=0.05 (fit) and tau_like=0.15 (posterior reweight).",
        "../data/超参数.md lists tau=0.05; problem1_summary.json lists tau_train=0.05 and tau_like=0.15.",
        "Preserve the paper's single tau OR the legacy dual tau behind an explicit named variant.",
        "Explicit single temperature in the integrated model.",
        "Changes posterior concentration and credible-interval widths.",
        "Uncertainty magnitude and CI width differ between the two settings.",
        "D-20260901-03",
    ],
    [
        "C-04",
        "Paper reports torch_model top-1 accuracy 0.952092 (XGBoost 0.806554).",
        "Review rebuild reports overall top-1 accuracy 0.949541.",
        "../review/problem1_rebuild/outputs/problem1_summary.json overall_top1_accuracy=0.949541.",
        "Reproduce the paper pipeline; target 0.952092.",
        "Reproduce the review rebuild; target 0.949541.",
        "~0.003 absolute gap (0.952092 vs 0.949541).",
        "Whether the gap is within tolerance or a real pipeline/seed difference.",
        "D-20260901-04",
    ],
    [
        "C-05",
        "Paper specifies eta = x^T beta + u with x = {current-week J, age}.",
        "Legacy adds a bias intercept b and an era_is_percent feature (X_cols = j_metric_z, age_z, era_is_percent).",
        "../review/problem1_rebuild/outputs/problem1_fit_metadata.json X_cols + cs2idx; ../src/model.py bias term.",
        "Preserve the paper's stated features OR the legacy features as a named paper-implementation variant.",
        "Explicit, documented feature set.",
        "era_is_percent changes era-specific fan support; intercept shifts the baseline.",
        "Minor, but affects identifiability and coefficient interpretation.",
        "D-20260901-05",
    ],
    [
        "C-06",
        "Paper reports inferred fan shares p_hat as estimated fan votes.",
        "Review requires: do not claim p_mean is the true official fan vote; it is a model-based posterior estimate.",
        "../review/problem1_rebuild/outputs/problem1_readme.md: 'model-based posterior fan-share summaries, not official vote totals.'",
        "Label p_hat as posterior estimates constrained by observed outcomes.",
        "Same labeling.",
        "None (labeling only).",
        "Prevents overclaiming latent fan votes as ground truth.",
        "D-20260901-06",
    ],
    [
        "C-07",
        "Paper reports XGBoost in-season baseline A = 0.806554 (same features as torch; Fig. 1).",
        "Review does not re-run the xgb baseline; review rebuild outputs only torch top-1 accuracy 0.949541.",
        "Legacy src/xgb_baseline.py + compare_models_cv.py run today produce xgb week-mean 0.821101 / season-mean 0.817496 (see /tmp/p1e_legacy/xgb_by_week_legacy.csv); the repo port is bit-for-bit identical.",
        "Preserve the paper's registered target 0.806554 in BASELINE_PAPER_OUTPUTS.md but report the legacy-reproduced line honestly (xgb week-mean 0.821101 / season-mean 0.817496); label the paper number as not reproducible from current legacy code/data.",
        "Track R has no xgb baseline requirement; it reports the torch rebuild (target 0.949541).",
        "~1.8% relative gap (0.821101 vs 0.806554); outside the proposed 1e-3 tolerance for B-01.",
        "The xgb baseline is a comparison line for the torch claim; the discrepancy does not change the torch line (0.952092 reproduced exactly).",
        "D-20260901-11",
    ],
]


def _esc(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")


def main() -> int:
    paths = load_paths()

    csv_path = paths.manifest_dir / "conflict_matrix.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(HEADER)
        writer.writerows(ROWS)

    lines = [
        "# Conflict matrix (paper vs review vs legacy)",
        "",
        "> Canonical data: `manifests/conflict_matrix.csv`. Generated by "
        "`scripts/build_conflict_matrix.py`.",
        "> Conflicts are recorded and kept separate as Track P vs Track R — never silently merged.",
        "> `decision_id` points into `docs/DECISIONS.md`.",
        "",
        f"| {' | '.join(HEADER)} |",
        f"|{'|'.join(['---'] * len(HEADER))}|",
    ]
    for row in ROWS:
        lines.append("| " + " | ".join(_esc(cell) for cell in row) + " |")
    lines.append("")

    (paths.repo_root / "docs" / "CONFLICT_MATRIX.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {len(ROWS)} conflicts to {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
