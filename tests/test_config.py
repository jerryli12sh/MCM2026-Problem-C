"""Tests for path resolution (config.py)."""

from __future__ import annotations

import pytest

from dwts_reproduction.config import REPO_ROOT, load_paths


def test_load_paths_resolves_anchors():
    paths = load_paths()
    assert paths.repo_root == REPO_ROOT
    assert paths.raw_data_csv.name == "2026_MCM_Problem_C_Data.csv"
    assert paths.paper_tex.name == "2107542.tex"
    assert paths.review_all.name == "review_all.md"


def test_sources_resolve_outside_repo():
    paths = load_paths()
    for directory in (
        paths.source_root,
        paths.data_dir,
        paths.src_dir,
        paths.review_dir,
        paths.paper_latex_dir,
        paths.figure_dir,
    ):
        assert not directory.resolve().is_relative_to(REPO_ROOT.resolve())


def test_missing_source_root_raises(tmp_path):
    bad = tmp_path / "paths.yaml"
    bad.write_text("unrelated: 1\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_paths(bad)
