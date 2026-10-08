"""Repository-relative locations shared by the analysis scripts."""

from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "configs"


@dataclass(frozen=True)
class Paths:
    repo_root: Path = REPO_ROOT
    data_dir: Path = REPO_ROOT / "data/intermediate"
    reference_dir: Path = REPO_ROOT / "data/reference"

    @property
    def raw_data_csv(self) -> Path:
        return self.repo_root / "data/2026_MCM_Problem_C_Data.csv"

    @property
    def data3_csv(self) -> Path:
        return self.data_dir / "data_3.csv"


def load_paths() -> Paths:
    return Paths()
