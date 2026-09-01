"""V1 season simulator (paper Mechanism I, schemes S1/S2/S3).

Exact port of the legacy producer ``../src/season_simulator.py``
(paper ``2107542.tex`` lines 871-985): a Monte-Carlo replay of every season
driven by posterior fan-share draws from Problem 1.  Three controlled schemes
differ only in *when* the judge-gate elimination rule is active:

- ``S1`` baseline all season: Bottom-2 by ``combined_rank = wJ*j_rank + wF*f_rank``,
  with a judges' save on the bottom two (highest judge rank eliminated).
- ``S2`` early stage uses S1; late stage activates the judge-gate (nominee
  bottom-K by judges, fans eliminate the least-supported nominee).
- ``S3`` full-season judge gate (finale fixed, no elimination).

Every scheme keeps the finale unchanged (no elimination at ``stage == "final"``).

Operational details preserved from the legacy code (documented in
D-20260901-18): the early/late split is by *elimination count*
(``elim_so_far < m_early_elims``) not by alive-set size as the prose of the
paper suggests; the judge signal is judge-percent in percent-era seasons and
``softmax(-judge_rank)`` in rank-era seasons (era_cutoff = 28); ties in the
baseline Bottom-2 save break on fan support via ``np.lexsort``; the judge gate
uses ``K = min(cfg.K, len(names))`` nominees.

The pooled fit is loaded once (see :mod:`.features`) and never re-trained.
Seeds: ``cfg.seed + 100000*scheme_idx + 1000*b + season`` per
(scheme, sim, season), deterministic for a fixed fit.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .features import (
    age_map,
    compute_q_hat,
    get_week_scores,
    load_archetypes,
    load_clean,
    load_weekly,
    rank_desc,
    softmax,
)


@dataclass
class SimConfig:
    """Configuration for one V1 simulator run (legacy ``SimConfig``)."""

    schemes: tuple[str, ...]
    n_sims: int
    seed: int
    wJ: float
    wF: float
    K: int
    m_early_elims: int
    final_n: int
    era_cutoff: int
    kappa: float


def stage_label(elim_so_far: int, total: int, m_early: int, final_n: int) -> str:
    """Port of ``_stage_label``: ``final`` / ``early`` / ``late``."""
    if total <= final_n:
        return "final"
    if elim_so_far < m_early:
        return "early"
    return "late"


def simulate_one_season(
    season: int,
    df_weekly: pd.DataFrame,
    archetypes: pd.DataFrame,
    ages: dict[str, float],
    pooled_fit: dict[str, Any],
    cfg: SimConfig,
    rng: np.random.Generator,
) -> list[dict]:
    """Simulate one season under ``cfg.schemes[0]`` (port of ``_simulate_one_season``)."""
    df_s = df_weekly[df_weekly["season"] == season].copy()
    if df_s.empty:
        return []

    weeks = sorted(df_s["week"].unique().tolist())
    week1 = weeks[0]
    initial = df_s[(df_s["week"] == week1) & (df_s.get("eligible", True))]
    active = sorted(initial["celebrity_name"].astype(str).unique().tolist())
    if not active:
        return []

    season_scores = df_s["total_judge_score"].dropna()
    if season_scores.empty:
        season_scores = df_s["judge_total"].dropna()
    season_mean = float(season_scores.mean()) if not season_scores.empty else 0.0

    arche_map = archetypes.set_index(["season", "celebrity_name"])["archetype"].to_dict()

    rows: list[dict] = []
    elim_so_far = 0
    last_scores: dict[str, float] = {}

    for week in weeks:
        if len(active) <= 1:
            break

        stage = stage_label(elim_so_far, len(active), cfg.m_early_elims, cfg.final_n)
        scores = get_week_scores(df_weekly, season, week, active, last_scores, season_mean)
        for name, v in scores.items():
            last_scores[name] = v

        names = np.array(active)
        judge_score = np.array([scores[n] for n in names], dtype=float)
        if judge_score.sum() == 0:
            judge_pct = np.ones_like(judge_score) / len(judge_score)
        else:
            judge_pct = judge_score / judge_score.sum()
        judge_rank = rank_desc(judge_score)

        era = "percent" if season >= cfg.era_cutoff else "rank"
        if era == "percent":
            j_metric = judge_pct
        else:
            j_metric = softmax(-judge_rank)

        df_rows = pd.DataFrame(
            {
                "season": season,
                "week": week,
                "celebrity_name": names,
                "era": era,
                "j_metric": j_metric,
                "age": [ages.get(n, np.nan) for n in names],
            }
        )
        q_hat = compute_q_hat(df_rows, pooled_fit)
        alpha = cfg.kappa * q_hat
        p_draw = rng.dirichlet(alpha)
        fan_rank = rank_desc(p_draw)
        combined_rank = cfg.wJ * judge_rank + cfg.wF * fan_rank

        eliminated = None
        if stage != "final":
            scheme = cfg.schemes[0]
            use_baseline = scheme == "S1" or (scheme == "S2" and stage == "early")
            if use_baseline:
                risk = combined_rank
                order = np.argsort(-risk)
                bottom2 = order[:2]
                if len(bottom2) == 1:
                    eliminated = names[bottom2[0]]
                else:
                    b2_j = judge_rank[bottom2]
                    b2_p = p_draw[bottom2]
                    if b2_j[0] == b2_j[1]:
                        elim_idx = bottom2[np.lexsort((b2_p,))[0]]
                    else:
                        elim_idx = bottom2[np.argmax(b2_j)]
                    eliminated = names[elim_idx]
            else:
                k = min(cfg.K, len(names))
                nominee_idx = np.argsort(judge_rank)[-k:]
                nominee_p = p_draw[nominee_idx]
                elim_idx = nominee_idx[np.lexsort((nominee_p,))[0]]
                eliminated = names[elim_idx]

        for i, name in enumerate(names):
            rows.append(
                {
                    "season": season,
                    "week": week,
                    "celebrity_name": name,
                    "archetype": arche_map.get((season, name), "unknown"),
                    "stage": stage,
                    "judge_score": judge_score[i],
                    "judge_rank": judge_rank[i],
                    "fan_p": p_draw[i],
                    "fan_rank": fan_rank[i],
                    "combined_rank": combined_rank[i],
                    "eliminated_this_week": bool(eliminated == name),
                }
            )

        if eliminated is not None and eliminated in active:
            active.remove(eliminated)
            elim_so_far += 1
        if stage == "final":
            break

    return rows


def run_simulation(
    df_weekly: pd.DataFrame,
    archetypes: pd.DataFrame,
    df_clean: pd.DataFrame,
    pooled_fit: dict[str, Any],
    cfg: SimConfig,
    seasons: list[int],
) -> pd.DataFrame:
    """Run all schemes/sims/seasons; return the detail frame (port of ``run_simulation``)."""
    ages = age_map(df_clean)
    all_rows: list[dict] = []

    for scheme_idx, scheme in enumerate(cfg.schemes):
        cfg_one = SimConfig(
            schemes=(scheme,),
            n_sims=cfg.n_sims,
            seed=cfg.seed,
            wJ=cfg.wJ,
            wF=cfg.wF,
            K=cfg.K,
            m_early_elims=cfg.m_early_elims,
            final_n=cfg.final_n,
            era_cutoff=cfg.era_cutoff,
            kappa=cfg.kappa,
        )
        for b in range(cfg.n_sims):
            for season in seasons:
                seed = cfg.seed + 100000 * scheme_idx + 1000 * b + int(season)
                rng = np.random.default_rng(seed)
                rows = simulate_one_season(
                    season, df_weekly, archetypes, ages, pooled_fit, cfg_one, rng
                )
                for r in rows:
                    r["scheme"] = scheme
                    r["sim"] = b
                all_rows.extend(rows)

    return pd.DataFrame(all_rows)


def summarize_results(df: pd.DataFrame) -> pd.DataFrame:
    """Per (scheme, week, archetype) alive/rank summary (port of ``summarize_results``).

    ``avg_rank`` is the mean ``combined_rank`` (V1 semantics; V2 replaces it
    with ``avg_score`` — the two are not comparable across simulators).
    """
    if df.empty:
        return df
    df = df.copy()
    df["alive"] = ~df["eliminated_this_week"]
    summary = (
        df.groupby(["scheme", "week", "archetype"], as_index=False)
        .agg(
            alive_rate=("alive", "mean"),
            avg_rank=("combined_rank", "mean"),
            elim_rate=("eliminated_this_week", "mean"),
            n=("celebrity_name", "count"),
        )
        .sort_values(["scheme", "week", "archetype"])
        .reset_index(drop=True)
    )
    return summary


def load_inputs(
    weekly_path: str | Path,
    clean_path: str | Path,
    archetypes_path: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the three simulator inputs through the shared loaders."""
    return (
        load_weekly(weekly_path),
        load_clean(clean_path),
        load_archetypes(archetypes_path),
    )
