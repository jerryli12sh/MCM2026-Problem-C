#!/usr/bin/env python3
"""Render the mechanism phase diagram (paper Fig. 5) from a saved source table.

The figure is a *pure function of the saved CSV*
``outputs/problem2_phase_metrics_{tag}.csv`` written by ``scripts/problem2_run.py``:
no model or sampler runs here, and the only inputs read are that table plus the
matching run manifest (whose output hashes pin the table). A JSON sidecar records
the source table hash, manifest path, and y-column definition so the figure can be
traced back to a specific run.

Axes (decision D-20260901-10):
- ``x`` = fan influence ``mu(|Ds|)`` = ``x_posterior_mean`` (shared by both tracks).
- ``y`` = Track P: the paper's ``y = 1 - mu(|Dr|)`` (``y_posterior_mean``, raw
  within-week rank differences — not bounded below); Track R: the review's
  ``y = 1 - mu(|r_Final - r_J|)`` (``y_review_posterior_mean``).

Season points for the four mechanisms (Pct/Rank x Direct/Bottom-2+Save) are drawn as
markers; a thin line connects the same season across mechanisms so the reader can
trace the paper's Direct -> Bottom-2+Save lift.  The vertical dashed line marks the
paper's "high fan-influence" threshold ``x >= 0.3`` (P-057).

Usage:
    python scripts/plot_phase_diagram.py [--tag P|R] [--table outputs/problem2_phase_metrics_P.csv]
        [--figure-dir outputs/figures] [--dpi 150]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

_SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(_SRC))

from dwts_reproduction.config import load_paths  # noqa: E402
from dwts_reproduction.hashing import sha256_file  # noqa: E402

# Mechanism -> (color, marker).  Order matches MECHANISMS in rules.py.
_MECHANISM_STYLE = {
    "rank_direct": ("tab:blue", "o"),
    "rank_bottom2": ("tab:orange", "s"),
    "pct_direct": ("tab:green", "^"),
    "pct_bottom2": ("tab:red", "D"),
}
_HIGH_FAN_INFLUENCE_X = 0.3  # paper P-057 threshold, mirrors HIGH_FAN_INFLUENCE_X


def _git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _y_column(tag: str) -> str:
    """Track P uses the paper's y; Track R uses the review's y (D-20260901-10)."""
    return "y_review_posterior_mean" if tag == "R" else "y_posterior_mean"


def _manifest_for(paths, tag: str) -> Path:
    return paths.repo_root / "outputs" / f"problem2_run_manifest_{tag}.json"


def main() -> int:
    parser = argparse.ArgumentParser(description="Mechanism phase diagram from saved table")
    parser.add_argument("--tag", choices=["P", "R"], default="P")
    parser.add_argument(
        "--table",
        default=None,
        help="source table (default: outputs/problem2_phase_metrics_{tag}.csv)",
    )
    parser.add_argument("--figure-dir", default="outputs/figures")
    parser.add_argument("--dpi", type=int, default=150)
    args = parser.parse_args()

    paths = load_paths()
    tag = args.tag
    table_path = (
        Path(args.table)
        if args.table
        else paths.repo_root / "outputs" / f"problem2_phase_metrics_{tag}.csv"
    )
    if not table_path.exists():
        print(
            f"error: source table not found: {table_path}\n"
            "run `python scripts/problem2_run.py --track <tag>` first",
            file=sys.stderr,
        )
        return 1

    df = pd.read_csv(table_path)
    ycol = _y_column(tag)
    required = {"season", "mechanism", "x_posterior_mean", ycol}
    missing = required - set(df.columns)
    if missing:
        print(f"error: table missing columns {sorted(missing)}", file=sys.stderr)
        return 1

    fig, ax = plt.subplots(figsize=(9.5, 7.0))
    mechanisms = [m for m in _MECHANISM_STYLE if m in set(df["mechanism"])]
    # Per-season polyline first so markers stay legible on top.
    for _, g in df.groupby("season"):
        order = [r for r in mechanisms if r in set(g["mechanism"])]
        g = g.set_index("mechanism").loc[order]
        ax.plot(
            g["x_posterior_mean"],
            g[ycol],
            "-",
            color="0.75",
            lw=0.8,
            zorder=1,
            alpha=0.55,
        )
    for mech in mechanisms:
        sub = df[df["mechanism"] == mech]
        color, marker = _MECHANISM_STYLE[mech]
        ax.scatter(
            sub["x_posterior_mean"],
            sub[ycol],
            s=42,
            color=color,
            marker=marker,
            label=mech,
            zorder=3,
        )
        # 10-90% posterior intervals (light whiskers) on both axes.
        for _, row in sub.iterrows():
            ax.plot(
                [row["x_ci_lo_10"], row["x_ci_hi_10"]],
                [row[ycol], row[ycol]],
                color=color,
                lw=0.7,
                alpha=0.45,
                zorder=2,
            )
            lo, hi = (
                row[f"{ycol.replace('posterior_mean', 'ci_lo_10')}"],
                row[f"{ycol.replace('posterior_mean', 'ci_hi_10')}"],
            )
            ax.plot(
                [row["x_posterior_mean"], row["x_posterior_mean"]],
                [lo, hi],
                color=color,
                lw=0.7,
                alpha=0.45,
                zorder=2,
            )

    ax.axvline(_HIGH_FAN_INFLUENCE_X, color="0.3", ls="--", lw=1.0)
    ax.text(
        _HIGH_FAN_INFLUENCE_X,
        ax.get_ylim()[0],
        f" high fan-influence (x >= {_HIGH_FAN_INFLUENCE_X})",
        fontsize=8,
        va="bottom",
    )
    ax.set_xlabel(r"Fan influence  $x=\mu(|\Delta s|)$  (mean $|p-J|$)")
    ydef = (
        r"Judge consistency  $y=1-\mu(|\Delta r|)$  (paper: within-week rank diff)"
        if tag == "P"
        else r"Judge consistency  $y=1-\mu(|r_{Final}-r_J|)$  (review R-040)"
    )
    ax.set_ylabel(ydef)
    ax.set_title(f"Mechanism phase diagram — Track {tag} (34 seasons, B=600)")
    ax.legend(title="mechanism", loc="best", framealpha=0.9, fontsize=8)
    ax.grid(True, which="both", alpha=0.25)
    fig.tight_layout()

    figure_dir = paths.repo_root / Path(args.figure_dir)
    figure_dir.mkdir(parents=True, exist_ok=True)
    out_png = figure_dir / f"problem2_phase_diagram_{tag}.png"
    fig.savefig(out_png, dpi=args.dpi)
    plt.close(fig)

    manifest_path = _manifest_for(paths, tag)
    manifest = {}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    sidecar = {
        "track": tag,
        "source_table": str(table_path.relative_to(paths.repo_root)),
        "source_table_sha256": sha256_file(table_path),
        "y_column": ycol,
        "x_column": "x_posterior_mean",
        "run_manifest": (
            str(manifest_path.relative_to(paths.repo_root)) if manifest_path.exists() else None
        ),
        "run_manifest_outputs_match": (
            manifest.get("outputs", {}).get(f"problem2_phase_metrics_{tag}.csv")
            == sha256_file(table_path)
            if manifest_path.exists()
            else None
        ),
        "git_commit": _git_commit(),
        "n_seasons": int(df["season"].nunique()),
        "n_rows": int(len(df)),
    }
    out_json = figure_dir / f"problem2_phase_diagram_{tag}.json"
    out_json.write_text(json.dumps(sidecar, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {out_png}")
    print(f"wrote {out_json}")
    print(
        f"  y column: {ycol}; source {table_path.name} (sha {sidecar['source_table_sha256'][:12]}…)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
