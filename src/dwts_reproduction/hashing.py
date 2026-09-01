"""Immutable-input hashing and provenance manifest.

Hashes every authoritative and reference input under the read-only source tree and
classifies each file into an ``artifact_role`` so that raw inputs, the paper/review
specifications, legacy implementation evidence, and derived legacy outputs are never
conflated. Caches, LaTeX build artifacts, and non-authoritative files are excluded.

The manifest is a tab-separated, header-commented text file with one row per covered file::

    # role\tsha256\trelative_path
    raw_input\t<sha256>\tdata/2026_MCM_Problem_C_Data.csv
"""

from __future__ import annotations

import hashlib
import os
from collections.abc import Iterator
from pathlib import Path, PurePosixPath

# Top-level paths (relative to ``source_root``) that the manifest covers. The repository
# itself (``repo/``) and unrelated loose files (``Latex/``, ``.idea/``, ``*.zip``,
# ``*.docx``) are deliberately NOT hashed.
HASH_TARGETS: tuple[str, ...] = (
    "data",
    "paper_Latex",
    "review",
    "src",
    "figure",
    "essay.pdf",
    "2623768_1.pdf",
)

# LaTeX build artifacts are regenerable and not authoritative; they are excluded.
_LATEX_BUILD_SUFFIXES = frozenset({".aux", ".log", ".out", ".toc", ".synctex.gz"})

# Authoritative paper files under paper_Latex/ (everything else there is a build artifact
# or an unrelated note).
_PAPER_SPEC_NAMES = frozenset({"2107542.tex", "2107542.pdf", "easymcm.sty"})

MANIFEST_HEADER = "# role\tsha256\trelative_path"


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    """Return the hexadecimal SHA-256 digest of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_excluded(parts: tuple[str, ...], name: str) -> bool:
    """Return True for caches, editor droppings, and other non-authoritative files."""
    if name == ".DS_Store":
        return True
    if name.endswith(".pyc"):
        return True
    if "__pycache__" in parts or ".venv" in parts or ".git" in parts:
        return True
    return False


def classify_role(rel_path: str) -> str | None:
    """Return the artifact role for a source-relative path, or ``None`` if excluded.

    Roles are one of ``raw_input``, ``paper_spec``, ``review_spec``, ``legacy_impl``,
    ``legacy_output``, or ``reference``. Returns ``None`` for caches and non-authoritative
    files so the caller can record them as skipped rather than silently hashing them.
    """
    p = PurePosixPath(rel_path)
    parts = p.parts
    if not parts:
        return None
    name = p.name
    if _is_excluded(parts, name):
        return None

    top = parts[0]

    if top == "data":
        if name in {"2026_MCM_Problem_C_Data.csv", "2026_MCM_Problem_C.pdf"}:
            return "raw_input"
        if p.suffix == ".md":
            return "raw_input"
        if p.suffix in {".csv", ".json"}:
            return "legacy_output"
        return None

    if top == "paper_Latex":
        if p.suffix in _LATEX_BUILD_SUFFIXES:
            return None
        if name in _PAPER_SPEC_NAMES:
            return "paper_spec"
        if "img" in parts:
            return "paper_spec"
        return None  # e.g. 注意事项.txt

    if top == "review":
        if "srcs_0" in parts:
            return "legacy_impl" if p.suffix == ".py" else "legacy_output"
        if "problem1_rebuild" in parts:
            if p.suffix == ".py":
                return "legacy_impl"
            if "outputs" in parts:
                return "legacy_output"
            return None
        # notes/*.md, notes/*.html, plan.md, and root-level refactor prompts.
        return "review_spec"

    if top == "src":
        return "legacy_impl" if p.suffix in {".py", ".ipynb"} else None

    if top == "figure":
        return "legacy_output" if p.suffix == ".png" else None

    if top in {"essay.pdf", "2623768_1.pdf"}:
        return "reference"

    return None


_PRUNED_DIRS = frozenset({".venv", "__pycache__", ".git", ".idea"})


def iter_files(source_root: Path) -> Iterator[Path]:
    """Yield every file under the covered hash targets (recursively for directories).

    Virtual environments, caches, and VCS/IDE directories are pruned during traversal so
    their contents are never enumerated (let alone hashed).
    """
    for target in HASH_TARGETS:
        candidate = source_root / target
        if candidate.is_file():
            yield candidate
            continue
        if not candidate.is_dir():
            continue
        for dirpath, dirnames, filenames in os.walk(candidate):
            dirnames[:] = sorted(d for d in dirnames if d not in _PRUNED_DIRS)
            for filename in sorted(filenames):
                yield Path(dirpath) / filename


def build_manifest(source_root: Path) -> tuple[list[tuple[str, str, str]], list[str]]:
    """Build the manifest.

    Returns a tuple of ``(rows, skipped)`` where each row is ``(role, sha256, rel_path)``
    sorted by path, and ``skipped`` lists excluded paths so nothing is dropped silently.
    """
    rows: list[tuple[str, str, str]] = []
    skipped: list[str] = []
    for file_path in iter_files(source_root):
        rel = str(file_path.relative_to(source_root))
        role = classify_role(rel)
        if role is None:
            skipped.append(rel)
            continue
        rows.append((role, sha256_file(file_path), rel))
    rows.sort(key=lambda r: r[2])
    skipped.sort()
    return rows, skipped


def format_manifest(rows: list[tuple[str, str, str]]) -> str:
    """Serialize manifest rows to the tab-separated text format."""
    lines = [MANIFEST_HEADER]
    lines.extend(f"{role}\t{digest}\t{rel}" for role, digest, rel in rows)
    return "\n".join(lines) + "\n"


def parse_manifest(text: str) -> list[tuple[str, str, str]]:
    """Parse a manifest's text back into ``(role, sha256, rel_path)`` rows."""
    rows: list[tuple[str, str, str]] = []
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        role, digest, rel = line.split("\t")
        rows.append((role, digest, rel))
    return rows


def validate_manifest(manifest_path: Path, source_root: Path) -> list[str]:
    """Re-hash the sources and compare against a recorded manifest.

    Returns a list of human-readable error strings; an empty list means the manifest is
    valid (every covered file is present and unchanged, with the correct role).
    """
    current_rows, _ = build_manifest(source_root)
    recorded_rows = parse_manifest(manifest_path.read_text(encoding="utf-8"))

    current = {rel: (role, digest) for role, digest, rel in current_rows}
    recorded = {rel: (role, digest) for role, digest, rel in recorded_rows}

    errors: list[str] = []
    for rel in sorted(set(current) | set(recorded)):
        if rel not in current:
            errors.append(f"recorded but missing on disk: {rel}")
        elif rel not in recorded:
            errors.append(f"on disk but not recorded: {rel}")
        elif recorded[rel] != current[rel]:
            errors.append(f"mismatch for {rel}: recorded={recorded[rel]} actual={current[rel]}")
    return errors
