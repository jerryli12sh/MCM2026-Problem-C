"""Phase 0 smoke test: prove a fresh environment can read inputs without modifying them.

The smoke test reads the raw contestant-season CSV and confirms its shape, and verifies
that touching (reading) a legacy file leaves its hash unchanged. It is dependency-light on
purpose: only the standard library is required to run it.
"""

from __future__ import annotations

import csv
from pathlib import Path

from .config import Paths
from .hashing import sha256_file

RAW_EXPECTED_SHAPE = (421, 53)


def read_raw_shape(paths: Paths) -> tuple[int, int]:
    """Return ``(n_rows, n_columns)`` of the raw contestant-season CSV (BOM-safe)."""
    with paths.raw_data_csv.open(newline="", encoding="utf-8-sig") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        n_cols = len(header)
        n_rows = sum(1 for _ in reader)
    return n_rows, n_cols


def verify_legacy_unchanged(paths: Paths, sample: Path | None = None) -> tuple[str, str]:
    """Hash a legacy file, read it in full, hash again; return ``(before, after)``."""
    target = sample if sample is not None else paths.paper_tex
    before = sha256_file(target)
    target.read_bytes()
    after = sha256_file(target)
    return before, after


def run_smoke(paths: Paths) -> dict[str, object]:
    """Run the smoke checks and return a summary dictionary."""
    shape = read_raw_shape(paths)
    before, after = verify_legacy_unchanged(paths)
    return {
        "raw_csv": str(paths.raw_data_csv),
        "raw_shape": shape,
        "raw_shape_ok": shape == RAW_EXPECTED_SHAPE,
        "legacy_file": str(paths.paper_tex),
        "legacy_hash_unchanged": before == after,
    }
