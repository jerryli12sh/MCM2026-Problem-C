#!/usr/bin/env python3
"""Run the analyses in dependency order with the active Python environment."""

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STAGES = {
    "preprocess": ["preprocess_run.py"],
    "model-p": ["problem1_run.py", "--track", "P"],
    "model-r": ["problem1_run.py", "--track", "R"],
    "baselines": ["problem1_extras_run.py"],
    "rules-p": ["problem2_run.py", "--track", "P"],
    "rules-r": ["problem2_run.py", "--track", "R"],
    "regression": ["problem3_run.py"],
    "simulation": ["problem4_run.py"],
    "sensitivity": ["sensitivity_run.py"],
    "figures": ["plot_results.py"],
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--core", action="store_true", help="preprocess and fit Track P")
    group.add_argument("--stage", choices=STAGES, help="run one stage; earlier inputs must exist")
    args = parser.parse_args()
    selected = (
        [args.stage] if args.stage else (["preprocess", "model-p"] if args.core else list(STAGES))
    )
    started = time.monotonic()
    for name in selected:
        script, *options = STAGES[name]
        print(f"\n>>> {name}", flush=True)
        subprocess.run(
            [sys.executable, "-u", str(ROOT / "scripts" / script), *options], cwd=ROOT, check=True
        )
    print(f"\nCompleted {len(selected)} stages in {time.monotonic() - started:.1f}s", flush=True)


if __name__ == "__main__":
    main()
