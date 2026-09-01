#!/usr/bin/env python3
"""Verify that no Git operation stages paths outside ``repo/``.

The Git repository is rooted at the *parent* of this repository (the MCM directory), which
also tracks the read-only sources and various unrelated files. Any accidental ``git add .``
or ``git commit -a`` would stage those. This gate asserts every staged path lives under
``repo/``.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _git_output(args: list[str]) -> str:
    result = subprocess.run(args, capture_output=True, text=True, check=True)
    return result.stdout


def main() -> int:
    try:
        toplevel = _git_output(["git", "rev-parse", "--show-toplevel"]).strip()
    except subprocess.CalledProcessError as error:
        print(f"git error: {error}")
        return 2

    top = Path(toplevel).resolve()
    repo = REPO_ROOT.resolve()
    if not repo.is_relative_to(top):
        print(f"ERROR: repo/ ({repo}) is not inside the git root ({top})")
        return 1

    # Staged paths relative to the git root (e.g. ``repo/foo.py``, ``paper.pdf``).
    staged = _git_output(["git", "-C", str(top), "diff", "--cached", "--name-only"]).splitlines()

    outside = [p for p in staged if p and Path(p).parts[0] != "repo"]
    if outside:
        print(f"ERROR: {len(outside)} staged path(s) outside repo/:")
        for path in outside:
            print(f"  - {path}")
        return 1

    print(f"OK: {len(staged)} staged path(s), all inside repo/ (git root: {top})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
