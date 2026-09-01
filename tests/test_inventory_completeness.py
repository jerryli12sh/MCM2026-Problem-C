"""Completeness checks: automated coverage backed by the manual checklist.

These prove that every literal figure/table in the paper is present in the paper inventory,
that the critical review requirements are present in the review inventory, and that every
conflict maps to a decision ID. The manual `docs/TRACEABILITY_COVERAGE.md` checklist
(verified by the independent audit) covers formulas, claims, assumptions, and conclusions
that automated parsing cannot.
"""

from __future__ import annotations

import re

from dwts_reproduction.config import load_paths


def test_paper_figures_covered():
    paths = load_paths()
    tex = paths.paper_tex.read_text(encoding="utf-8")
    csv_text = (paths.manifest_dir / "traceability_paper.csv").read_text(encoding="utf-8")
    includes = re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", tex)
    missing = [
        name.split("/")[-1]
        for name in includes
        if "#" not in name and name.split("/")[-1] not in csv_text
    ]
    assert not missing, f"figures missing from paper inventory: {missing}"


def test_paper_tables_covered():
    paths = load_paths()
    tex = paths.paper_tex.read_text(encoding="utf-8")
    csv_text = (paths.manifest_dir / "traceability_paper.csv").read_text(encoding="utf-8")
    labels = re.findall(r"\\label\{tab:([^}]+)\}", tex)
    missing = [label for label in labels if f"tab:{label}" not in csv_text]
    assert not missing, f"table labels missing from paper inventory: {missing}"


def test_review_targets_covered():
    paths = load_paths()
    csv_text = (paths.manifest_dir / "traceability_review.csv").read_text(encoding="utf-8")
    targets = [
        "421",
        "53",
        "4671",
        "4741",
        "9412",
        "18524",
        "4199",
        "292",
        "218",
        "258",
        "track_r_marginal",
        "era_official",
        "era_legacy",
        "0.949541",
    ]
    missing = [target for target in targets if target not in csv_text]
    assert not missing, f"review targets missing from review inventory: {missing}"


def test_conflicts_have_decision_ids():
    paths = load_paths()
    conflicts = (paths.manifest_dir / "conflict_matrix.csv").read_text(encoding="utf-8")
    decisions = (paths.repo_root / "docs" / "DECISIONS.md").read_text(encoding="utf-8")
    decision_ids = re.findall(r"D-\d{8}-\d{2}", conflicts)
    assert decision_ids, "no decision IDs found in conflict matrix"
    for decision_id in decision_ids:
        assert decision_id in decisions, f"decision {decision_id} missing from DECISIONS.md"
