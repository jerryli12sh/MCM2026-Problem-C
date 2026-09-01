"""Problem 1 configuration.

A single frozen dataclass holds every hyperparameter of the latent fan-support
models.  Track P and Track R share this schema; only the defaults differ
(``era_mode`` and the presence of an integrated marginal-likelihood fit).
The YAML file under ``configs/problem1.yaml`` records the registered Track P
settings; ``load_problem1_config`` is the only loader used by production code.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from dwts_reproduction.config import CONFIG_DIR

DEFAULT_PROBLEM1_YAML = CONFIG_DIR / "problem1.yaml"


@dataclass(frozen=True)
class Problem1Config:
    """Hyperparameters for the Problem 1 latent fan-support models.

    Attributes:
        era_mode: Judge-aggregation era mapping. ``"legacy"`` (``season >= 28 ->
            percent``) reproduces the reference outputs; ``"official"`` follows the
            problem statement (seasons 1-2 rank, 3-27 percent, 28-34 rank).
        seed: Seed for the pooled fit and per-week posterior sampling.
        tau_train: Softmin temperature used to fit ``q``.
        tau_like: Softmin temperature used to reweight posterior draws.
        kappa: Dirichlet concentration multiplier for the weekly prior
            ``p ~ Dirichlet(kappa q)``.
        l2_beta: L2 penalty on the shared coefficients ``beta``.
        l2_u: L2 penalty on the contestant-season effects ``u``.
        lr: Adam learning rate for the hand-written optimizer.
        n_steps: Number of pooled-fit minibatch steps.
        batch_size: Choice sets per minibatch.
        B: Number of Dirichlet draws for posterior approximation.
        eps: Floor for numerical stability (q clipping, CI denominator).
        alpha_floor: Minimum Dirichlet concentration for the Track R MC score
            gradient.  Below ~0.1 the gamma sampler underflows individual draws
            to exact ``0.0`` (``log 0 -> -inf`` corrupts the score estimate) and
            the score variance ``trigamma(alpha_i)`` grows like ``1/alpha_i``;
            ``0.1`` keeps both bounded while leaving the fit essentially
            unchanged for any contestant with plausible fan support.
    """

    era_mode: str = "legacy"
    seed: int = 42
    tau_train: float = 0.05
    tau_like: float = 0.15
    kappa: float = 10.0
    l2_beta: float = 0.05
    l2_u: float = 0.05
    lr: float = 0.020
    n_steps: int = 600
    batch_size: int = 32
    B: int = 1200
    eps: float = 1e-6
    alpha_floor: float = 0.1

    def __post_init__(self) -> None:
        if self.era_mode not in {"legacy", "official"}:
            raise ValueError("era_mode must be 'legacy' or 'official'.")
        if self.B < 1 or self.n_steps < 1 or self.batch_size < 1:
            raise ValueError("B, n_steps, and batch_size must be positive.")
        if not (0.0 < self.tau_train and 0.0 < self.tau_like):
            raise ValueError("temperatures must be positive.")
        if self.kappa <= 0.0:
            raise ValueError("kappa must be positive.")
        if not (0.0 < self.alpha_floor <= 1.0):
            raise ValueError("alpha_floor must be in (0, 1].")

    @classmethod
    def for_track(cls, track: str) -> Problem1Config:
        """Return the registered defaults for a track.

        Track P keeps the legacy era mapping (matches the reference rebuild and the
        paper's reported outputs).  Track R uses the official problem-statement
        mapping, which is what the review's integrated model is specified against.
        """
        if track not in {"P", "R"}:
            raise ValueError("track must be 'P' or 'R'")
        era = "legacy" if track == "P" else "official"
        return cls(era_mode=era)


@dataclass(frozen=True)
class Problem1Paths:
    """Inputs and output directory for one Problem 1 run."""

    raw_data_csv: Path
    output_dir: Path


def load_problem1_config(path: Path | None = None) -> Problem1Config:
    """Load a :class:`Problem1Config` from YAML (defaults to ``configs/problem1.yaml``).

    Each known field is coerced to its declared dataclass type.  PyYAML (YAML 1.1)
    parses exponent-only notation like ``eps: 1e-6`` as a string because its float
    resolver requires a decimal point, so ``int()`` / ``float()`` are applied to
    fields declared as numbers to keep the loader robust to either spelling.
    """
    yaml_path = Path(path) if path else DEFAULT_PROBLEM1_YAML
    data = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
    known: dict[str, Any] = {}
    for f, field_def in Problem1Config.__dataclass_fields__.items():
        if f not in data:
            continue
        value = data[f]
        # ``from __future__ import annotations`` makes ``field_def.type`` a string.
        if field_def.type in ("float", "int"):
            known[f] = float(value) if field_def.type == "float" else int(value)
        else:
            known[f] = value
    return Problem1Config(**known)
