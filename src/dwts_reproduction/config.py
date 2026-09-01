"""Path resolution for the DWTS reproduction repository.

All external (read-only) inputs — raw data, the paper, the review notes, and the legacy
implementation — live in the *parent* of this repository. This module resolves them from a
single ``source_root`` so that no absolute, machine-specific path appears in library code.

The repository root is discovered from the location of this file (``parents[2]`` of
``src/dwts_reproduction/config.py``), so editable installs and direct imports both work.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

# repo/ = src/dwts_reproduction/config.py -> parents[2]
REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "configs"
DEFAULT_PATHS_YAML = CONFIG_DIR / "paths.yaml"


@dataclass(frozen=True)
class Paths:
    """Resolved paths for the repository and its read-only sources."""

    repo_root: Path
    source_root: Path
    data_dir: Path
    src_dir: Path
    review_dir: Path
    paper_latex_dir: Path
    figure_dir: Path

    @property
    def raw_data_csv(self) -> Path:
        """Raw contestant-season wide table."""
        return self.data_dir / "2026_MCM_Problem_C_Data.csv"

    @property
    def data3_csv(self) -> Path:
        """Problem 3 engineered table (one row per season-celebrity).

        Hashed legacy output produced by the original analysis (registered in
        ``manifests/legacy_inventory.csv`` as role=legacy_output, sha
        ``72ca124e…``); it is the input read by the legacy
        ``dwts_pro_celeb_regression.py`` producer.
        """
        return self.data_dir / "data_3.csv"

    @property
    def paper_tex(self) -> Path:
        """Authoritative paper source (Track P)."""
        return self.paper_latex_dir / "2107542.tex"

    @property
    def review_all(self) -> Path:
        """Authoritative consolidated review note (Track R)."""
        return self.review_dir / "notes" / "review_all.md"

    @property
    def manifest_dir(self) -> Path:
        """Directory holding generated manifests."""
        return self.repo_root / "manifests"


def load_paths(paths_yaml: Path | None = None) -> Paths:
    """Load ``configs/paths.yaml`` and build a :class:`Paths`.

    Args:
        paths_yaml: Optional override for the paths configuration file. Defaults to
            ``configs/paths.yaml`` inside the repository.

    Raises:
        ValueError: If ``source_root`` is missing, or if any derived source directory
            resolves to a location inside the repository (sources must be read-only and
            live outside ``repo/``).
    """
    yaml_path = Path(paths_yaml) if paths_yaml else DEFAULT_PATHS_YAML
    data = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
    raw_root = data.get("source_root")
    if raw_root is None:
        raise ValueError(f"{yaml_path} must declare a `source_root` key")

    source_root = (REPO_ROOT / str(raw_root)).resolve()

    paths = Paths(
        repo_root=REPO_ROOT.resolve(),
        source_root=source_root,
        data_dir=source_root / "data",
        src_dir=source_root / "src",
        review_dir=source_root / "review",
        paper_latex_dir=source_root / "paper_Latex",
        figure_dir=source_root / "figure",
    )

    # Enforce that the read-only sources live outside the repository.
    for name, path in (
        ("source_root", source_root),
        ("data_dir", paths.data_dir),
        ("src_dir", paths.src_dir),
        ("review_dir", paths.review_dir),
        ("paper_latex_dir", paths.paper_latex_dir),
        ("figure_dir", paths.figure_dir),
    ):
        if path.resolve().is_relative_to(REPO_ROOT.resolve()):
            raise ValueError(
                f"{name} resolves to {path}, which is inside the repository; "
                "read-only sources must live outside repo/"
            )
    return paths
