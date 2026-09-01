"""Tests for immutable-input hashing and role classification."""

from __future__ import annotations

from dwts_reproduction.hashing import (
    build_manifest,
    classify_role,
    format_manifest,
    sha256_file,
    validate_manifest,
)

_SHA256_HELLO = "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"


def test_sha256_is_deterministic_and_correct(tmp_path):
    target = tmp_path / "a.txt"
    target.write_bytes(b"hello")
    assert sha256_file(target) == _SHA256_HELLO
    assert sha256_file(target) == sha256_file(target)


def test_classify_role_known_paths():
    cases = {
        "data/2026_MCM_Problem_C_Data.csv": "raw_input",
        "data/2026_MCM_Problem_C.pdf": "raw_input",
        "data/df_clean.csv": "legacy_output",
        "data/数据说明.md": "raw_input",
        "paper_Latex/2107542.tex": "paper_spec",
        "paper_Latex/2107542.pdf": "paper_spec",
        "paper_Latex/easymcm.sty": "paper_spec",
        "paper_Latex/img/1_model_accuracy_line.png": "paper_spec",
        "review/notes/review_all.md": "review_spec",
        "review/plan.md": "review_spec",
        "review/srcs_0/dwts_preprocess.py": "legacy_impl",
        "review/srcs_0/df_clean.csv": "legacy_output",
        "review/problem1_rebuild/problem1_fan_support.py": "legacy_impl",
        "review/problem1_rebuild/outputs/x.json": "legacy_output",
        "src/model.py": "legacy_impl",
        "src/0data_cleaning.ipynb": "legacy_impl",
        "figure/fig_phase_diagram.png": "legacy_output",
        "essay.pdf": "reference",
        "2623768_1.pdf": "reference",
    }
    for path, expected in cases.items():
        assert classify_role(path) == expected, path


def test_classify_role_excluded():
    assert classify_role("paper_Latex/2107542.aux") is None
    assert classify_role("paper_Latex/2107542.log") is None
    assert classify_role("paper_Latex/注意事项.txt") is None
    assert classify_role("src/__pycache__/model.cpython-311.pyc") is None
    assert classify_role("data/.DS_Store") is None
    assert classify_role("review/problem1_rebuild/.venv/bin/python") is None
    assert classify_role("Latex/test.tex") is None  # not a hash target


def test_build_and_validate_roundtrip(tmp_path):
    (tmp_path / "data").mkdir()
    (tmp_path / "src").mkdir()
    (tmp_path / "data" / "2026_MCM_Problem_C_Data.csv").write_text("a,b\n1,2\n", encoding="utf-8")
    (tmp_path / "data" / "df_clean.csv").write_text("x\n1\n", encoding="utf-8")
    (tmp_path / "src" / "model.py").write_text("print(1)\n", encoding="utf-8")

    rows, skipped = build_manifest(tmp_path)
    roles = {rel: role for role, _, rel in rows}
    assert roles["data/2026_MCM_Problem_C_Data.csv"] == "raw_input"
    assert roles["data/df_clean.csv"] == "legacy_output"
    assert roles["src/model.py"] == "legacy_impl"
    assert len(rows) == 3
    assert skipped == []

    manifest = tmp_path / "manifest.sha256"
    manifest.write_text(format_manifest(rows), encoding="utf-8")
    assert validate_manifest(manifest, tmp_path) == []

    (tmp_path / "data" / "df_clean.csv").write_text("x\n2\n", encoding="utf-8")
    errors = validate_manifest(manifest, tmp_path)
    assert len(errors) == 1
    assert "df_clean.csv" in errors[0]
