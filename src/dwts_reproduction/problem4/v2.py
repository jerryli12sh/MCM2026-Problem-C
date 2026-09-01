"""V2 season simulator (paper Mechanism II, schemes V4/V5).

Exact port of the legacy producer ``../src/season_simulator2.py`` (paper
``2107542.tex`` lines 986-1060).  The paper labels the two compared mechanisms
"V0" (baseline) and "V2" (proposed); the legacy code calls them ``V4`` and
``V5`` — the repo keeps the legacy scheme names and documents the mapping
(D-20260901-18).

- ``V4`` baseline: ``S = wJ*judge_share + wF*p_draw``.
- ``V5`` proposed: ``S = wJ*j_sharp + wF*p_temp + bonus`` with fan tempering
  ``p_temp = p^gamma/sum``, judge sharpening ``j_sharp = J^delta/sum`` and a
  momentum bonus ``mu*tanh(z/c)`` where ``z = m/(sd_T+eps)`` with
  ``m = T_now - mean(judge_score past L weeks)``.
- Elimination when ``len(active) > final_n``: Bottom-2 by ``S``, judges' save
  via ``argmin(judge_score[bottom2])``.  The finale (``len(active) <= final_n``)
  computes ``S`` but eliminates nobody and the loop runs through all weeks
  (no early ``break`` — unlike V1).

The momentum bonus depends on the past judge-score history of each contestant
(``history``), which is updated *after* the row is appended.  Judge share
fallback is uniform when the week's judge score sums to zero.  Details recorded
per row: ``score_S``, ``fan_share``, ``fan_share_tempered``,
``judge_share_sharp``, ``momentum_z``, ``bonus``, ``eliminated_this_week``.
The per-(scheme, week, archetype) summary aggregates ``avg_score`` (not
``avg_rank``) — the two simulators are not directly comparable.

One latent legacy quirk is preserved: ``scheme`` is a per-week loop variable
assigned in the elimination branch and reused in the finale branch.  The repo
assigns it once at the top of each week so a season that starts at
``final_n`` contestants cannot hit a ``NameError``; behaviour is otherwise
identical (``cfg.schemes[0]`` never changes inside a run).
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
    """Configuration for one V2 simulator run (legacy ``SimConfig``)."""

    schemes: tuple[str, ...]
    n_sims: int
    seed: int
    gamma: float
    delta: float
    wJ: float
    wF: float
    L: int
    mu: float
    c: float
    kappa: float
    final_n: int
    era_cutoff: int


def momentum_bonus(
    name: str,
    T_now: float,
    history: dict[str, list[float]],
    sd_T: float,
    L: int,
    mu: float,
    c: float,
    eps: float = 1e-8,
) -> tuple[float, float]:
    """Port of ``_momentum_bonus``: (bonus, z-score of the L-week judge trend)."""
    past = history.get(name, [])
    if len(past) < L:
        m = 0.0
    else:
        m = T_now - float(np.mean(past[-L:]))
    z = m / (sd_T + eps)
    bonus = mu * np.tanh(z / c)
    return bonus, z


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
    last_scores: dict[str, float] = {}
    history: dict[str, list[float]] = {n: [] for n in active}

    for week in weeks:
        if len(active) <= 1:
            break

        scores = get_week_scores(df_weekly, season, week, active, last_scores, season_mean)
        for name, v in scores.items():
            last_scores[name] = v

        names = np.array(active)
        judge_score = np.array([scores[n] for n in names], dtype=float)
        sd_T = float(np.std(judge_score, ddof=0))

        if judge_score.sum() == 0:
            judge_share = np.ones_like(judge_score) / len(judge_score)
        else:
            judge_share = judge_score / judge_score.sum()
        judge_rank = rank_desc(judge_score)

        era = "percent" if season >= cfg.era_cutoff else "rank"
        if era == "percent":
            j_metric = judge_share
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

        p_temp = p_draw**cfg.gamma
        p_temp = p_temp / p_temp.sum()
        j_sharp = judge_share**cfg.delta
        j_sharp = j_sharp / j_sharp.sum()

        bonuses = np.zeros(len(names), dtype=float)
        z_vals = np.zeros(len(names), dtype=float)
        for i, name in enumerate(names):
            bonus, z = momentum_bonus(name, judge_score[i], history, sd_T, cfg.L, cfg.mu, cfg.c)
            bonuses[i] = bonus
            z_vals[i] = z

        scheme = cfg.schemes[0]
        if scheme == "V4":
            S = cfg.wJ * judge_share + cfg.wF * p_draw
        else:
            S = cfg.wJ * j_sharp + cfg.wF * p_temp + bonuses

        eliminated = None
        if len(active) > cfg.final_n:
            order = np.argsort(S)
            bottom2 = order[:2]
            if len(bottom2) == 1:
                eliminated = names[bottom2[0]]
            else:
                b2_t = judge_score[bottom2]
                elim_idx = bottom2[np.argmin(b2_t)]
                eliminated = names[elim_idx]

        for i, name in enumerate(names):
            rows.append(
                {
                    "season": season,
                    "week": week,
                    "celebrity_name": name,
                    "archetype": arche_map.get((season, name), "unknown"),
                    "judge_score": judge_score[i],
                    "judge_share": judge_share[i],
                    "fan_share": p_draw[i],
                    "fan_share_tempered": p_temp[i],
                    "judge_share_sharp": j_sharp[i],
                    "momentum_z": z_vals[i],
                    "bonus": bonuses[i],
                    "score_S": S[i],
                    "eliminated_this_week": bool(eliminated == name),
                }
            )

        for i, name in enumerate(names):
            history.setdefault(name, [])
            history[name].append(judge_score[i])

        if eliminated is not None and eliminated in active:
            active.remove(eliminated)

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
            gamma=cfg.gamma,
            delta=cfg.delta,
            wJ=cfg.wJ,
            wF=cfg.wF,
            L=cfg.L,
            mu=cfg.mu,
            c=cfg.c,
            kappa=cfg.kappa,
            final_n=cfg.final_n,
            era_cutoff=cfg.era_cutoff,
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
    """Per (scheme, week, archetype) alive/score summary (port of ``summarize_results``).

    ``avg_score`` is the mean composite ``score_S``; V1's summary uses
    ``avg_rank`` instead, so summaries across simulators are not comparable.
    """
    if df.empty:
        return df
    df = df.copy()
    df["alive"] = ~df["eliminated_this_week"]
    summary = (
        df.groupby(["scheme", "week", "archetype"], as_index=False)
        .agg(
            alive_rate=("alive", "mean"),
            avg_score=("score_S", "mean"),
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
