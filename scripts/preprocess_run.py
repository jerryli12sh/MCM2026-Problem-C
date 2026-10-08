#!/usr/bin/env python3
"""Rebuild weekly rosters and elimination events from the contest CSV."""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dwts_reproduction.config import load_paths
from dwts_reproduction.preprocess import build_all_tables


def main():
    paths = load_paths()
    tables = build_all_tables(paths.raw_data_csv)
    paths.data_dir.mkdir(parents=True, exist_ok=True)
    for name in ("clean", "weekly", "roster", "elim_events"):
        frame = getattr(tables, name)
        frame.to_csv(paths.data_dir / f"df_{name}.csv", index=False)
        print(f"df_{name}.csv: {len(frame)} rows")
    tables.validation.to_csv(paths.data_dir / "validation.csv", index=False)
    # Preserve the original judge/fan signals while using official final placements.
    signals = pd.read_csv(paths.data_dir / "celebrity_signals.csv")
    signals["celebrity_name"] = signals["celebrity_name"].str.strip()
    raw = tables.raw.copy()
    raw["celebrity_name"] = raw["celebrity_name"].str.strip()
    features = signals.merge(
        raw[["season", "celebrity_name", "placement"]],
        on=["season", "celebrity_name"],
        validate="one_to_one",
        how="left",
    )
    if features["placement"].isna().any():
        raise ValueError("Every regression row must match an official final placement")
    season_size = raw.groupby("season").size()
    n = features["season"].map(season_size)
    beaten = (n - features["placement"]) / (n - 1)
    features["placement_z"] = (beaten - beaten.mean()) / beaten.std(ddof=0)
    features.to_csv(paths.data3_csv, index=False)
    print(f"data_3.csv: {len(features)} regression rows with official final placements")
    print(f"Raw data: {len(tables.raw)} contestant-seasons, {tables.raw.season.nunique()} seasons")


if __name__ == "__main__":
    main()
