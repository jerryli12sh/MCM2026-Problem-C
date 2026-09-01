#!/usr/bin/env python3
"""Build or validate the immutable-input manifest.

Build mode walks the covered read-only sources, classifies each file by artifact role, and
writes ``manifests/input_manifest.sha256``. Validate mode re-hashes every covered file and
reports any missing file, new file, or hash/role mismatch.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(_SRC))

from dwts_reproduction.config import load_paths  # noqa: E402
from dwts_reproduction.hashing import (  # noqa: E402
    build_manifest,
    format_manifest,
    validate_manifest,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", help="paths.yaml override")
    parser.add_argument("--out", help="manifest output path (build mode only)")
    parser.add_argument("--build", action="store_true", help="build the manifest (default)")
    parser.add_argument("--validate", action="store_true", help="validate the manifest")
    args = parser.parse_args(argv)

    paths = load_paths(args.config)
    manifest_path = Path(args.out) if args.out else paths.manifest_dir / "input_manifest.sha256"

    if args.validate:
        errors = validate_manifest(manifest_path, paths.source_root)
        if errors:
            print(f"INVALID ({len(errors)} problem(s)):")
            for error in errors:
                print(f"  - {error}")
            return 1
        print(f"OK: {manifest_path} matches {paths.source_root}")
        return 0

    rows, skipped = build_manifest(paths.source_root)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(format_manifest(rows), encoding="utf-8")
    print(f"wrote {len(rows)} entries to {manifest_path}")
    if skipped:
        print(f"skipped {len(skipped)} excluded path(s):")
        for item in skipped:
            print(f"  - {item}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
