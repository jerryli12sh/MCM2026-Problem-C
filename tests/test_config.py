"""Tests for path resolution (config.py)."""

from __future__ import annotations

import pytest

from dwts_reproduction.config import REPO_ROOT, load_paths
from dwts_reproduction.problem1.config import load_problem1_config


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


def test_problem1_config_registered_defaults_coerce_types():
    """The registered YAML uses exponent-only floats (``eps: 1e-6``); PyYAML 1.1
    parses those as strings, so the loader must coerce to the declared types."""
    config = load_problem1_config()
    assert isinstance(config.eps, float) and config.eps == 1e-6
    assert isinstance(config.tau_train, float) and isinstance(config.lr, float)
    assert isinstance(config.n_steps, int) and isinstance(config.batch_size, int)
    assert config.era_mode == "legacy"
    assert config.B == 1200


def test_problem1_config_accepts_both_float_spellings(tmp_path):
    """``1e-6`` and ``1.0e-6`` must resolve identically after coercion."""
    exponent_only = tmp_path / "exp.yaml"
    exponent_only.write_text("eps: 1e-6\n", encoding="utf-8")
    decimal = tmp_path / "dec.yaml"
    decimal.write_text("eps: 1.0e-6\n", encoding="utf-8")
    assert load_problem1_config(exponent_only).eps == load_problem1_config(decimal).eps == 1e-6
