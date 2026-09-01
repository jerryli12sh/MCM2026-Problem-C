"""Run-manifest schema: records how a generated output was produced.

Every non-trivial generated output (tables, figures, metrics) must be accompanied by a
run manifest capturing the track, configuration, input hashes, Git commit, environment,
seeds, command, timing, status, and output hashes. Validation is dependency-free: a
hand-written :meth:`RunManifest.validate` enforces the required fields and types so the
schema lives in one place and cannot drift from a separate JSON Schema document.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# "P" and "R" are the two analytical tracks; "P1E" tags Problem 1 evaluation
# extras (in-season baselines, PCP, ranking gap, heatmaps) that build on the
# Track P posterior summary — see docs/TRACEABILITY.md. The list is kept closed
# so an unexpected tag cannot be recorded silently.
VALID_TRACKS = ("P", "R", "P1E")
VALID_STATUSES = ("pending", "running", "success", "failed", "skipped")

_REQUIRED_FIELDS = (
    "track",
    "config_path",
    "input_manifest_sha256",
    "git_commit",
    "environment",
    "seeds",
    "command",
    "started_at",
    "ended_at",
    "status",
    "outputs",
)


@dataclass
class RunManifest:
    """One run's provenance record.

    Attributes:
        track: ``"P"`` (paper-faithful) or ``"R"`` (review-corrected).
        config_path: Path to the configuration that produced this run.
        input_manifest_sha256: SHA-256 of the input manifest used (immutable inputs).
        git_commit: Full Git commit hash of the repository at run time.
        environment: Mapping of Python version and package versions.
        seeds: Mapping of named random seeds used.
        command: The exact command line that produced the run.
        started_at: ISO-8601 start timestamp.
        ended_at: ISO-8601 end timestamp.
        status: One of ``pending``, ``running``, ``success``, ``failed``, ``skipped``.
        outputs: Mapping of output path (relative) to its SHA-256.
    """

    track: str
    config_path: str
    input_manifest_sha256: str
    git_commit: str
    environment: dict[str, Any]
    seeds: dict[str, Any]
    command: str
    started_at: str
    ended_at: str
    status: str
    outputs: dict[str, str] = field(default_factory=dict)

    def validate(self) -> list[str]:
        """Return a list of validation errors; empty means the manifest is valid."""
        errors: list[str] = []
        for field_name in _REQUIRED_FIELDS:
            value = getattr(self, field_name, None)
            if value is None:
                errors.append(f"missing required field: {field_name}")
                continue
            if field_name in {
                "track",
                "config_path",
                "input_manifest_sha256",
                "git_commit",
                "command",
                "started_at",
                "ended_at",
                "status",
            }:
                if not isinstance(value, str) or not value.strip():
                    errors.append(f"field {field_name} must be a non-empty string")

        if self.track not in VALID_TRACKS:
            errors.append(f"track must be one of {VALID_TRACKS}, got {self.track!r}")
        if self.status not in VALID_STATUSES:
            errors.append(f"status must be one of {VALID_STATUSES}, got {self.status!r}")
        if not isinstance(self.environment, dict):
            errors.append("environment must be a mapping")
        if not isinstance(self.seeds, dict):
            errors.append("seeds must be a mapping")
        if not isinstance(self.outputs, dict):
            errors.append("outputs must be a mapping")
        elif not all(isinstance(k, str) and isinstance(v, str) for k, v in self.outputs.items()):
            errors.append("outputs must map str path -> str sha256")
        return errors

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-ready dictionary."""
        return {
            "track": self.track,
            "config_path": self.config_path,
            "input_manifest_sha256": self.input_manifest_sha256,
            "git_commit": self.git_commit,
            "environment": self.environment,
            "seeds": self.seeds,
            "command": self.command,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "status": self.status,
            "outputs": self.outputs,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RunManifest:
        """Construct a :class:`RunManifest` from a decoded JSON dictionary."""
        return cls(
            track=data["track"],
            config_path=data["config_path"],
            input_manifest_sha256=data["input_manifest_sha256"],
            git_commit=data["git_commit"],
            environment=data.get("environment", {}),
            seeds=data.get("seeds", {}),
            command=data["command"],
            started_at=data["started_at"],
            ended_at=data["ended_at"],
            status=data["status"],
            outputs=data.get("outputs", {}),
        )

    def write(self, path: Path) -> None:
        """Validate then write the manifest as JSON.

        Raises:
            ValueError: If validation fails or ``input_manifest_sha256`` is empty.
        """
        errors = self.validate()
        if errors:
            raise ValueError("run manifest is invalid: " + "; ".join(errors))
        if not self.input_manifest_sha256:
            raise ValueError("input_manifest_sha256 must not be empty")
        path.write_text(
            json.dumps(self.to_dict(), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
