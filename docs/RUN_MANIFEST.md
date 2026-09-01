# Run-manifest schema

Every non-trivial generated output (table, figure, metric) must be accompanied by a run
manifest recording how it was produced. The schema is defined in
`src/dwts_reproduction/run_manifest.py` (a dataclass plus a hand-written `validate()`); there
is **no** separate JSON Schema document, so the contract cannot drift from the validator.

## Fields

| field | type | meaning |
|---|---|---|
| `track` | str | `P` (paper-faithful) or `R` (review-corrected) |
| `config_path` | str | path to the configuration that produced the run |
| `input_manifest_sha256` | str | SHA-256 of the immutable-input manifest used |
| `git_commit` | str | full Git commit hash at run time |
| `environment` | map | Python version + frozen package versions |
| `seeds` | map | named random seeds used |
| `command` | str | exact command line that produced the run |
| `started_at` | str | ISO-8601 start timestamp |
| `ended_at` | str | ISO-8601 end timestamp |
| `status` | str | `pending` / `running` / `success` / `failed` / `skipped` |
| `outputs` | map[str, str] | output path (relative) → SHA-256 |

## Validation rules (dependency-free)

- Every field in the table above is required and must not be `None`.
- `track` ∈ {`P`, `R`}; `status` ∈ {`pending`, `running`, `success`, `failed`, `skipped`}.
- The scalar string fields must be non-empty strings.
- `environment`, `seeds`, and `outputs` must be mappings; `outputs` maps `str → str`.
- `RunManifest.write()` refuses to write unless `input_manifest_sha256` is non-empty.

## Usage

```python
from pathlib import Path
from dwts_reproduction.run_manifest import RunManifest

m = RunManifest(
    track="P",
    config_path="configs/phase0.yaml",
    input_manifest_sha256="<sha256 of manifests/input_manifest.sha256>",
    git_commit="<git rev-parse HEAD>",
    environment={"python": "3.13.3", "packages": {"numpy": "2.5.2"}},
    seeds={"seed": 42},
    command="python scripts/... ",
    started_at="2026-09-01T00:00:00Z",
    ended_at="2026-09-01T00:00:10Z",
    status="success",
    outputs={"outputs/foo.csv": "<sha256>"},
)
m.write(Path("outputs/foo.manifest.json"))  # validates then writes
```

Generated outputs are not evidence unless their run manifest records the configuration,
seeds, input hashes, Git commit, environment, and command used.
