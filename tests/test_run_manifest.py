"""Tests for the dependency-free run-manifest validator."""

from __future__ import annotations

import json

import pytest

from dwts_reproduction.run_manifest import RunManifest


def _valid(**overrides) -> RunManifest:
    base = dict(
        track="P",
        config_path="configs/phase0.yaml",
        input_manifest_sha256="a" * 64,
        git_commit="deadbeef",
        environment={"python": "3.13"},
        seeds={"seed": 42},
        command="python scripts/x.py",
        started_at="2026-09-01T00:00:00Z",
        ended_at="2026-09-01T00:00:01Z",
        status="success",
        outputs={"out.csv": "b" * 64},
    )
    base.update(overrides)
    return RunManifest(**base)


def test_valid_manifest_passes():
    assert _valid().validate() == []


def test_rejects_bad_track():
    assert any("track" in e for e in _valid(track="X").validate())


def test_rejects_bad_status():
    assert any("status" in e for e in _valid(status="done").validate())


def test_rejects_empty_string_fields():
    errors = _valid(config_path="").validate()
    assert any("config_path" in e for e in errors)


def test_rejects_non_mapping_outputs():
    assert any("outputs" in e for e in _valid(outputs=["a", "b"]).validate())  # type: ignore[arg-type]


def test_write_refuses_empty_input_manifest(tmp_path):
    with pytest.raises(ValueError):
        _valid(input_manifest_sha256="").write(tmp_path / "m.json")


def test_write_roundtrip(tmp_path):
    _valid().write(tmp_path / "m.json")
    data = json.loads((tmp_path / "m.json").read_text(encoding="utf-8"))
    assert data["track"] == "P"
    assert data["outputs"] == {"out.csv": "b" * 64}
    assert RunManifest.from_dict(data).validate() == []
