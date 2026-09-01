"""DWTS MCM reproduction package.

This package reproduces the submitted DWTS MCM paper as a clean, testable Python
repository, with two explicit analytical tracks:

- **Track P** — paper-faithful reproduction (primary).
- **Track R** — review-corrected implementation (secondary).

Phase 0 modules here (``config``, ``hashing``, ``run_manifest``) provide the provenance and
reproducibility plumbing that every later phase inherits.
"""

from .config import Paths, load_paths
from .hashing import (
    HASH_TARGETS,
    build_manifest,
    classify_role,
    parse_manifest,
    sha256_file,
    validate_manifest,
)
from .run_manifest import RunManifest

__all__ = [
    "HASH_TARGETS",
    "Paths",
    "RunManifest",
    "build_manifest",
    "classify_role",
    "load_paths",
    "parse_manifest",
    "sha256_file",
    "validate_manifest",
]

__version__ = "0.1.0"
