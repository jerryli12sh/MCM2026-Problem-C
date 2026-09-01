"""Repository-scope safety: no Git operation stages paths outside ``repo/``.

The Git repository is rooted at the parent MCM directory, which also tracks the read-only
sources and unrelated files. These tests prove the git root is the parent and that the
``check_scope`` gate reports success (nothing staged outside ``repo/``).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], capture_output=True, text=True, check=True).stdout


def test_git_toplevel_is_parent():
    toplevel = _git("rev-parse", "--show-toplevel").strip()
    assert Path(toplevel).resolve() == REPO.resolve().parent


def test_check_scope_gate_passes():
    result = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "check_scope.py")],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_no_staged_paths_outside_repo():
    toplevel = _git("rev-parse", "--show-toplevel").strip()
    staged = _git("-C", toplevel, "diff", "--cached", "--name-only").splitlines()
    outside = [p for p in staged if p and Path(p).parts[0] != "repo"]
    assert not outside, f"staged paths outside repo/: {outside}"
