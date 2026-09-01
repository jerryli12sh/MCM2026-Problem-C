#!/usr/bin/env python3
"""Run the Phase 0 smoke test and report pass/fail (exit code 0/1)."""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(_SRC))

from dwts_reproduction.config import load_paths  # noqa: E402
from dwts_reproduction.smoke import run_smoke  # noqa: E402


def main() -> int:
    paths = load_paths()
    summary = run_smoke(paths)

    print(f"raw CSV:        {summary['raw_csv']}")
    print(f"raw shape:      {summary['raw_shape']} (expected (421, 53))")
    print(f"shape OK:       {summary['raw_shape_ok']}")
    print(f"legacy file:    {summary['legacy_file']}")
    print(f"hash unchanged: {summary['legacy_hash_unchanged']}")

    ok = bool(summary["raw_shape_ok"]) and bool(summary["legacy_hash_unchanged"])
    print("SMOKE TEST:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
