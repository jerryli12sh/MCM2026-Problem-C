"""Hermetic tests for the release comparison module (synthetic fixtures only)."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from dwts_reproduction.release.compare import (
    VERDICT_FAIL,
    VERDICT_INFO,
    VERDICT_PASS,
    compare,
    parse_tolerance,
    summarize,
)


def _write_baseline(tmp_path, rows: list[dict]) -> str:
    df = pd.DataFrame(rows)
    path = tmp_path / "baseline.csv"
    df.to_csv(path, index=False)
    return str(path)


def _write_torch_inseason(tmp_path, acc: float) -> None:
    df = pd.DataFrame(
        {
            "model": ["torch_model"] * 3 + ["xgboost_baseline"] * 2,
            "season": [1, 2, 3, 1, 2],
            "accuracy": [acc] * 3 + [0.8, 0.81],
        }
    )
    df.to_csv(tmp_path / "problem1_extras_inseason_by_season_P1E.csv", index=False)


def _write_p1_summary(tmp_path, **values: float) -> None:
    payload = {
        "overall_top1_accuracy": 0.9495412844,
        "s_bar": 0.7785,
        "mean_pcp_weighted": 0.6043173,
        "mean_ess_ratio": 0.9625174,
        "mean_ci_rel_width": 3.1171359,
        **values,
    }
    (tmp_path / "problem1_summary_P.json").write_text(json.dumps(payload))


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("rel 1e-3 (proposed)", ("rel", 0.001)),
        ("abs 0.02 (proposed)", ("abs", 0.02)),
        ("abs 5e-3 (recalibrated, proposed)", ("abs", 0.005)),
        ("exact", ("exact", None)),
        ("structural (proposed)", ("structural", None)),
        ("visual (proposed)", ("visual", None)),
    ],
)
def test_parse_tolerance(text: str, expected: tuple[str, float | None]) -> None:
    assert parse_tolerance(text) == expected


def test_numeric_check_passes(tmp_path) -> None:
    _write_torch_inseason(tmp_path, 0.952092)
    baseline = _write_baseline(
        tmp_path,
        [
            {
                "id": "B-02",
                "item": "Top-1 accuracy (torch_model)",
                "proposed_tolerance": "rel 1e-3 (proposed)",
            }
        ],
    )
    results = compare(baseline, str(tmp_path), str(tmp_path), repo_root=tmp_path)
    assert len(results) == 1
    assert results[0].verdict == VERDICT_PASS
    assert "0.952092" in results[0].observed


def test_numeric_check_fails_when_drifted(tmp_path) -> None:
    _write_torch_inseason(tmp_path, 0.88)  # well outside rel 1e-3 of 0.952092
    baseline = _write_baseline(
        tmp_path,
        [
            {
                "id": "B-02",
                "item": "Top-1 accuracy (torch_model)",
                "proposed_tolerance": "rel 1e-3 (proposed)",
            }
        ],
    )
    results = compare(baseline, str(tmp_path), str(tmp_path), repo_root=tmp_path)
    assert results[0].verdict == VERDICT_FAIL


def test_unknown_row_reports_info(tmp_path) -> None:
    baseline = _write_baseline(
        tmp_path,
        [{"id": "Z-99", "item": "placeholder", "proposed_tolerance": "abs 1"}],
    )
    results = compare(baseline, str(tmp_path), str(tmp_path), repo_root=tmp_path)
    assert len(results) == 1
    assert results[0].verdict == VERDICT_INFO


def test_b19_parity_within_tolerance(tmp_path) -> None:
    rows = [
        {"scheme": s, "week": w, "archetype": a, "avg_rank": r, "alive_rate": 0.5, "elim_rate": 0.1}
        for s in ("S1", "S2", "S3")
        for w in range(1, 12)
        for a in range(1, 4)
        for r in [float(w) + 0.001]
    ]
    pd.DataFrame(rows).to_csv(tmp_path / "problem4_sim_summary_V1.csv", index=False)
    # legacy copy with a tiny drift below the 5e-3 tolerance
    legacy = pd.DataFrame(rows)
    legacy["avg_rank"] = legacy["avg_rank"] + 0.002
    legacy.to_csv(tmp_path / "sim_summary.csv", index=False)
    baseline = _write_baseline(
        tmp_path,
        [
            {
                "id": "B-19",
                "item": "V1 simulator legacy parity",
                "proposed_tolerance": "abs 5e-3 (recalibrated, proposed)",
            }
        ],
    )
    results = compare(baseline, str(tmp_path), str(tmp_path), repo_root=tmp_path)
    assert results[0].verdict == VERDICT_PASS


def test_b19_parity_fails_when_drifted(tmp_path) -> None:
    rows = [
        {
            "scheme": s,
            "week": w,
            "archetype": a,
            "avg_rank": float(w) + 0.001,
            "alive_rate": 0.5,
            "elim_rate": 0.1,
        }
        for s in ("S1", "S2", "S3")
        for w in range(1, 12)
        for a in range(1, 4)
    ]
    pd.DataFrame(rows).to_csv(tmp_path / "problem4_sim_summary_V1.csv", index=False)
    legacy = pd.DataFrame(rows)
    legacy.loc[0, "avg_rank"] += 0.05  # exceed 5e-3
    legacy.to_csv(tmp_path / "sim_summary.csv", index=False)
    baseline = _write_baseline(
        tmp_path,
        [
            {
                "id": "B-19",
                "item": "V1 simulator legacy parity",
                "proposed_tolerance": "abs 5e-3 (recalibrated, proposed)",
            }
        ],
    )
    results = compare(baseline, str(tmp_path), str(tmp_path), repo_root=tmp_path)
    assert results[0].verdict == VERDICT_FAIL


def test_p1_summary_checks(tmp_path) -> None:
    _write_p1_summary(tmp_path, s_bar=0.7785)
    baseline = _write_baseline(
        tmp_path,
        [
            {
                "id": "B-03",
                "item": "Top-1 (review rebuild)",
                "proposed_tolerance": "rel 1e-3 (proposed)",
            },
            {
                "id": "B-04",
                "item": "Cumulative consistency S-bar",
                "proposed_tolerance": "abs 0.02 (proposed)",
            },
        ],
    )
    results = compare(baseline, str(tmp_path), str(tmp_path), repo_root=tmp_path)
    assert [r.verdict for r in results] == [VERDICT_PASS, VERDICT_PASS]


def test_summarize_counts() -> None:
    from dwts_reproduction.release.compare import CheckResult

    results = [
        CheckResult("B-02", "x", "rel 1e-3", "0.952", VERDICT_PASS, ""),
        CheckResult("B-12", "y", "abs 0.1", "0.254", VERDICT_PASS, ""),
        CheckResult("Z-9", "z", "abs 1", "", VERDICT_INFO, ""),
    ]
    assert summarize(results) == {
        "checked": 3,
        "pass": 2,
        "fail": 0,
        "info": 1,
        "release_ok": True,
    }


def _write_review_traceability(tmp_path, statuses: dict[str, str]) -> None:
    """Write a minimal traceability_review.csv (B-16's read path)."""
    header = [
        "id",
        "source",
        "loc",
        "item_type",
        "requirement",
        "target_value",
        "legacy_producer",
        "track_P_module",
        "track_R_module",
        "acceptance_test",
        "tolerance",
        "status",
    ]
    rows = [
        [
            idx,
            "src",
            "loc",
            "validation",
            "req",
            "",
            "",
            "preprocessing",
            "preprocessing",
            "test_preprocess",
            "",
            status,
        ]
        for idx, status in statuses.items()
    ]
    (tmp_path / "manifests").mkdir(exist_ok=True)
    pd.DataFrame(rows, columns=header).to_csv(
        tmp_path / "manifests" / "traceability_review.csv", index=False
    )


def test_b16_scoped_to_r001_r019(tmp_path) -> None:
    # In-scope targets implemented; an out-of-scope planned row (R-021) must not fail B-16.
    statuses = {f"R-{i:03d}": "implemented" for i in range(1, 36)}
    statuses["R-021"] = "planned"
    _write_review_traceability(tmp_path, statuses)
    baseline = _write_baseline(
        tmp_path,
        [
            {
                "id": "B-16",
                "item": "Preprocessing validation targets",
                "proposed_tolerance": "exact (proposed)",
            }
        ],
    )
    results = compare(baseline, str(tmp_path), str(tmp_path), repo_root=tmp_path)
    assert results[0].verdict == VERDICT_PASS
    assert "19 preprocessing targets" in results[0].observed


def test_b16_fails_when_in_scope_row_planned(tmp_path) -> None:
    statuses = {f"R-{i:03d}": "implemented" for i in range(1, 36)}
    statuses["R-007"] = "planned"  # in scope -> must fail
    _write_review_traceability(tmp_path, statuses)
    baseline = _write_baseline(
        tmp_path,
        [
            {
                "id": "B-16",
                "item": "Preprocessing validation targets",
                "proposed_tolerance": "exact (proposed)",
            }
        ],
    )
    results = compare(baseline, str(tmp_path), str(tmp_path), repo_root=tmp_path)
    assert results[0].verdict == VERDICT_FAIL


def _write_fig_manifest(tmp_path, stem: str, kind: str, n: int) -> None:
    payload = {"track": "test", kind: {f"f{i}.png": "sha" for i in range(n)}}
    (tmp_path / f"{stem}_fig_manifest_{kind}.json").write_text(json.dumps(payload))


def test_b17_counts_all_manifest_schemas(tmp_path) -> None:
    # 6 manifests across the three schema variants (figures/outputs/files), 10 each.
    for i in range(6):
        _write_fig_manifest(tmp_path, f"p{i}", ("figures", "outputs", "files")[i % 3], 10)
    baseline = _write_baseline(
        tmp_path,
        [{"id": "B-17", "item": "Paper figures", "proposed_tolerance": "visual (proposed)"}],
    )
    results = compare(baseline, str(tmp_path), str(tmp_path), repo_root=tmp_path)
    assert results[0].verdict == VERDICT_PASS
    assert "60 PNGs" in results[0].observed


def test_b17_fails_when_pngs_undercount(tmp_path) -> None:
    for i in range(6):
        _write_fig_manifest(tmp_path, f"p{i}", ("figures", "outputs", "files")[i % 3], 5)
    baseline = _write_baseline(
        tmp_path,
        [{"id": "B-17", "item": "Paper figures", "proposed_tolerance": "visual (proposed)"}],
    )
    results = compare(baseline, str(tmp_path), str(tmp_path), repo_root=tmp_path)
    assert results[0].verdict == VERDICT_FAIL


def test_b17_ignores_unregistered_schema_key(tmp_path) -> None:
    # A manifest using an unrecognized key is not counted at all (the original B-17 bug).
    for i in range(6):
        _write_fig_manifest(tmp_path, f"p{i}", "pngs", 10)
    baseline = _write_baseline(
        tmp_path,
        [{"id": "B-17", "item": "Paper figures", "proposed_tolerance": "visual (proposed)"}],
    )
    results = compare(baseline, str(tmp_path), str(tmp_path), repo_root=tmp_path)
    assert results[0].verdict == VERDICT_FAIL
    assert "0 PNGs" in results[0].observed
