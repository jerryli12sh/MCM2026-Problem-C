#!/usr/bin/env python
"""End-to-end release reproduction driver.

Runs every pipeline stage (Problem 1 P/R, extras, Problem 2 P/R, Problem 3,
Problem 4, sensitivity, and all plot scripts), regenerates the baseline and
traceability documents, then compares every registered baseline row against the
freshly produced artifacts and writes ``release_manifest.json`` +
``release_comparison.json`` under the output directory.

Modes:
- default: run every stage, then compare.
- ``--verify-only``: do not recompute anything; compare the existing artifacts
  against the registered baseline (fast re-check, used by CI / post-review).
- ``--skip NAME``: skip a named stage (repeatable). Stage names are the first
  column of the STEPS table below.

All pipeline scripts are invoked with the interpreter running this driver, so
the repository-local environment is used. Determinism is enforced by the
pipeline scripts' fixed seeds; this driver only orchestrates and records.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from dwts_reproduction.config import load_paths
from dwts_reproduction.hashing import sha256_file
from dwts_reproduction.release.compare import compare, summarize

REPO_ROOT = Path(__file__).resolve().parents[1]

# name -> argv (relative paths resolved against REPO_ROOT).
STEPS: list[tuple[str, list[str]]] = [
    ("problem1_run P", [sys.executable, "scripts/problem1_run.py", "--track", "P"]),
    ("problem1_run R", [sys.executable, "scripts/problem1_run.py", "--track", "R"]),
    ("problem1_extras_run", [sys.executable, "scripts/problem1_extras_run.py"]),
    ("problem2_run P", [sys.executable, "scripts/problem2_run.py", "--track", "P"]),
    ("problem2_run R", [sys.executable, "scripts/problem2_run.py", "--track", "R"]),
    ("problem3_run", [sys.executable, "scripts/problem3_run.py"]),
    ("problem4_run", [sys.executable, "scripts/problem4_run.py"]),
    ("sensitivity_run", [sys.executable, "scripts/sensitivity_run.py"]),
    ("plot_problem1", [sys.executable, "scripts/plot_problem1_figures.py"]),
    ("plot_problem2 P", [sys.executable, "scripts/plot_problem2_figures.py", "--track", "P"]),
    ("plot_problem2 R", [sys.executable, "scripts/plot_problem2_figures.py", "--track", "R"]),
    ("plot_phase_diagram P", [sys.executable, "scripts/plot_phase_diagram.py", "--tag", "P"]),
    ("plot_phase_diagram R", [sys.executable, "scripts/plot_phase_diagram.py", "--tag", "R"]),
    ("plot_problem3", [sys.executable, "scripts/plot_problem3_figures.py"]),
    ("plot_problem4", [sys.executable, "scripts/plot_problem4_figures.py"]),
    ("plot_sensitivity", [sys.executable, "scripts/plot_sensitivity_figures.py"]),
    ("build_baseline", [sys.executable, "scripts/build_baseline.py"]),
    ("build_traceability", [sys.executable, "scripts/build_traceability.py"]),
    ("build_conflict_matrix", [sys.executable, "scripts/build_conflict_matrix.py"]),
]


def _git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip()
    except Exception:
        return "unknown"


def _run_step(name: str, argv: list[str]) -> dict:
    started = datetime.now(UTC).isoformat()
    t0 = time.monotonic()
    proc = subprocess.run(
        argv,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    duration_s = round(time.monotonic() - t0, 2)
    tail = (proc.stdout or "").strip().splitlines()[-8:]
    err_tail = (proc.stderr or "").strip().splitlines()[-8:]
    return {
        "stage": name,
        "command": " ".join(argv),
        "started_at": started,
        "ended_at": datetime.now(UTC).isoformat(),
        "duration_s": duration_s,
        "exit_code": proc.returncode,
        "stdout_tail": tail,
        "stderr_tail": err_tail,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default="outputs")
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="skip all pipeline stages; only re-run the baseline comparison",
    )
    parser.add_argument(
        "--skip", action="append", default=[], help="stage name to skip (repeatable)"
    )
    args = parser.parse_args()

    outputs = Path(args.out_dir)
    outputs.mkdir(parents=True, exist_ok=True)

    run_manifest: dict = {
        "track": "release",
        "generated_by": "scripts/run_release.py",
        "started_at": datetime.now(UTC).isoformat(),
        "git_commit": _git_commit(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "steps": [],
    }

    steps_done = 0
    if not args.verify_only:
        for name, argv in STEPS:
            if name in args.skip:
                run_manifest["steps"].append(
                    {"stage": name, "skipped": True, "command": " ".join(argv)}
                )
                continue
            record = _run_step(name, argv)
            run_manifest["steps"].append(record)
            steps_done += 1
            status = "ok" if record["exit_code"] == 0 else "FAILED"
            print(f"[{status:6s}] {name} ({record['duration_s']}s)")
            if record["exit_code"] != 0:
                print("  stderr tail:", "\n  ".join(record["stderr_tail"]))
                run_manifest["ended_at"] = datetime.now(UTC).isoformat()
                (outputs / "release_manifest.json").write_text(json.dumps(run_manifest, indent=2))
                return 1
        if steps_done == 0:
            print("nothing to run; add a stage or drop --verify-only")

    # Comparison against the registered baseline.
    paths = load_paths()
    results = compare(
        REPO_ROOT / "manifests" / "baseline.csv",
        outputs,
        paths.data_dir,
        repo_root=REPO_ROOT,
    )
    summary = summarize(results)
    comparison = {
        "track": "release",
        "baseline": "manifests/baseline.csv",
        "summary": summary,
        "rows": [
            {
                "id": r.id,
                "item": r.item,
                "tolerance": r.tolerance,
                "observed": r.observed,
                "verdict": r.verdict,
                "detail": r.detail,
            }
            for r in results
        ],
    }
    comp_path = outputs / "release_comparison.json"
    comp_path.write_text(json.dumps(comparison, indent=2))

    run_manifest.update(
        {
            "ended_at": datetime.now(UTC).isoformat(),
            "comparison": {
                "file": str(comp_path),
                "sha256": sha256_file(comp_path),
                "summary": summary,
            },
            "all_stages_exit_zero": all(s.get("exit_code", 0) == 0 for s in run_manifest["steps"]),
        }
    )
    run_path = outputs / "release_manifest.json"
    run_path.write_text(json.dumps(run_manifest, indent=2))

    print("\n=== release comparison ===")
    for r in results:
        print(f"[{r.verdict:4s}] {r.id:6s} {r.observed}  {r.detail}")
    print(
        f"\nchecked={summary['checked']} pass={summary['pass']} "
        f"fail={summary['fail']} info={summary['info']} "
        f"release_ok={summary['release_ok']}"
    )
    return 0 if summary["release_ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
