"""Tests for the Phase 0 smoke test (no uncontrolled generated files)."""

from __future__ import annotations

from dwts_reproduction.config import load_paths
from dwts_reproduction.run_manifest import RunManifest
from dwts_reproduction.smoke import RAW_EXPECTED_SHAPE, read_raw_shape, run_smoke


def test_raw_shape():
    paths = load_paths()
    assert read_raw_shape(paths) == RAW_EXPECTED_SHAPE


def test_run_smoke_checks_pass():
    paths = load_paths()
    summary = run_smoke(paths)
    assert summary["raw_shape_ok"] is True
    assert summary["legacy_hash_unchanged"] is True


def test_run_manifest_written_to_tmp(tmp_path):
    manifest = RunManifest(
        track="P",
        config_path="configs/phase0.yaml",
        input_manifest_sha256="c" * 64,
        git_commit="deadbeef",
        environment={"python": "3.13"},
        seeds={"seed": 42},
        command="make smoke",
        started_at="2026-09-01T00:00:00Z",
        ended_at="2026-09-01T00:00:01Z",
        status="success",
        outputs={},
    )
    manifest.write(tmp_path / "smoke.manifest.json")
    assert (tmp_path / "smoke.manifest.json").exists()
