"""Mechanism phase diagram and cross-mechanism rule comparison (paper Fig. 5).

The paper summarizes season-level rule behavior with a *Mechanism Phase
Diagram* (``paper_Latex/2107542.tex``, lines 738-749): each season is embedded
by **fan influence** on the x-axis, ``x = mu(|Ds|)`` (the magnitude of the
popularity cushion), and **judge consistency** on the y-axis,
``y = 1 - mu(|Dr|)``, where ``Ds = p - J`` and ``Dr = rF - rJ``.  Four
mechanisms (Pct/Rank x Direct/Bottom-2+Save) each yield a point per season;
arrows trace the same season under different rules.

Decision D-20260901-10: no legacy code producer exists for the phase diagram
(notebook ``src/2_rank_vs_pct_cross_season.ipynb`` has no such code and the
paper ships only the figure PNG), so the axes are reconstructed here as a
*counterfactual trajectory replay*.  Start from the observed week-1 alive set;
at every eligible single-elimination week eliminate the mechanism's choice; at
non-eligible weeks (finales, double-elimination, non-elimination) apply the
observed eliminations among the survivors still present.  ``x`` and ``y`` are
the contestant-week means of ``|p - J|`` and ``|rF - rJ|`` over the
counterfactual alive sets, ranks computed within each alive set.

A mechanism that disagrees with the observed elimination keeps contestants the
data had already eliminated, so counterfactual survivors can fall outside the
observed alive set of a later week.  Each roster contestant therefore carries a
week-indexed fan share and judge vector under **last-observation carry-forward**
(their final observed values from the most recent alive week with a finite
judge vector); ``p`` is the cell-3 point ``p_hat`` for the point trajectory and
the per-draw posterior for posterior propagation.  Weeks whose judge vector is
not fully finite are skipped, matching the rest of the Problem 2 replay
(D-20260901-09).

The paper and the review (``review_all.md`` lines 277-279) also give different
``y`` definitions: the paper's weekly ``Dr = rF - rJ`` (used for the Track P
figure) versus the review's contestant-level ``|r_Final - r_J|`` (final
placement vs. judge standing, used for the Track R comparison).  Both are
computed from the same replay trajectories and reported side by side; nothing
here asserts a causal reading of the recommendation (P-057) — it is reproduced
as a claim check on the replay output (D-20260901-06).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from dwts_reproduction.problem1.track_p import PooledFit
from dwts_reproduction.problem2.replay import (
    DrawCache,
    _week_p_hat,
    build_train_weeks,
    config_from_fit,
    eligible_weeks,
    week_judge_vector,
)
from dwts_reproduction.problem2.rules import (
    MECHANISMS,
    descending_rank,
    simulate_week,
)

# The paper's high fan-influence threshold for the tail-risk comparison
# (``mu(|Ds|) gtrsim 0.3``, paper line 740).
HIGH_FAN_INFLUENCE_X = 0.3


def _observed_eliminated(
    panel: pd.DataFrame, season: int, week: int, alive_weeks: list[int]
) -> set[str]:
    """Observed contestants eliminated at ``week`` (alive-set difference).

    Non-eligible weeks (finales, double-elimination, non-elimination) evolve
    the counterfactual alive set by the *observed* eliminations — the set of
    contestants alive at ``week`` but not alive at the next alive week.  The
    last alive week performs no further evolution, so it needs no elimination
    set (D-20260901-10).
    """
    i = alive_weeks.index(week)
    if i == len(alive_weeks) - 1:
        return set()
    nxt = alive_weeks[i + 1]
    alive_t = set(
        panel[(panel["season"] == season) & (panel["week"] == week) & panel["alive"]][
            "celebrity_name"
        ].astype(str)
    )
    alive_nxt = set(
        panel[(panel["season"] == season) & (panel["week"] == nxt) & panel["alive"]][
            "celebrity_name"
        ].astype(str)
    )
    return alive_t - alive_nxt


def _week_contribution(p: np.ndarray, j: np.ndarray) -> tuple[float, float]:
    """(sum_i |p_i - J_i|, sum_i |rF_i - rJ_i|) over one alive set.

    Ranks are within the alive set (``descending_rank``, 1 = best), matching
    the ``|d|``/``Flip`` semantics used elsewhere in Problem 2.
    """
    rF = descending_rank(p)
    rJ = descending_rank(j)
    return float(np.abs(p - j).sum()), float(np.abs(rF - rJ).sum())


def _season_alive_weeks(panel: pd.DataFrame, season: int) -> list[int]:
    """Distinct alive weeks of one season, chronological."""
    return [
        int(w) for w in sorted(panel[(panel["season"] == season) & panel["alive"]]["week"].unique())
    ]


def _season_roster(panel: pd.DataFrame, season: int) -> list[str]:
    """Name-sorted roster of every contestant observed alive in the season."""
    return sorted(
        panel[(panel["season"] == season) & panel["alive"]]["celebrity_name"]
        .astype(str)
        .unique()
        .tolist()
    )


def _season_judge_rank(panel: pd.DataFrame, season: int) -> pd.Series:
    """Descending judge standing ``r_J`` indexed by contestant name.

    ``r_J(i)`` is the descending rank of contestant ``i``'s mean within-week
    judge share over their alive weeks — a per-contestant technical standing
    used by the review's ``y = 1 - mu(|r_Final - r_J|)`` definition
    (R-040, D-20260901-10).  Contestants whose mean judge share is not finite
    are excluded by the caller via a validity mask.
    """
    g = (
        panel[(panel["season"] == season) & panel["alive"]]
        .groupby("celebrity_name", sort=True)["judge_percent"]
        .mean()
    )
    g.index = g.index.astype(str)
    return pd.Series(descending_rank(g.to_numpy(float)), index=g.index)


def _extended_weeks(
    panel: pd.DataFrame,
    fit: PooledFit,
    cache: DrawCache,
    season: int,
    alive_weeks: list[int],
    roster: list[str],
    *,
    B: int,
    train_keys: set[tuple[int, int]],
) -> list[dict[str, Any]]:
    """Per-alive-week snapshots over the full season roster, carry-forward.

    For each alive week with a fully finite judge vector, records the
    observed-alive mask, the judge vector, the point fan share, and the
    per-draw fan shares for *every* roster contestant.  Contestants the observed
    data has already eliminated carry forward their last observed values
    (last-observation carry-forward), so the replay can evaluate counterfactual
    survivors outside the observed alive set.  Weeks with a non-finite judge
    vector are skipped (no contribution, no evolution), matching the rest of the
    Problem 2 replay (D-20260901-09).
    """
    idx = {n: i for i, n in enumerate(roster)}
    cur_j = np.full(len(roster), np.nan)
    cur_p = np.zeros(len(roster))
    cur_d = np.zeros((B, len(roster)))
    out: list[dict[str, Any]] = []
    for w in alive_weeks:
        names, j = week_judge_vector(panel, season, w)
        if not np.isfinite(j).all():
            continue
        p_hat = _week_p_hat(panel, fit, cache, season, w, names, train_keys, B)
        p_draws, aligned = cache.aligned(season, w, names, B)
        if aligned != names:
            raise ValueError(f"week {season}/{w}: cache alignment {aligned} != judge set {names}")
        for i, n in enumerate(names):
            ci = idx[n]
            cur_j[ci] = j[i]
            cur_p[ci] = p_hat[i]
            cur_d[:, ci] = p_draws[:, i]
        alive_mask = np.zeros(len(roster), dtype=bool)
        for n in names:
            alive_mask[idx[n]] = True
        out.append(
            {
                "week": int(w),
                "alive_mask": alive_mask,
                "j": cur_j.copy(),
                "p_point": cur_p.copy(),
                "p_draws": cur_d.copy(),
                "obs_elim": _observed_eliminated(panel, season, w, alive_weeks),
            }
        )
    return out


def _replay_season_point(
    weeks: list[dict[str, Any]],
    season: int,
    mechanism: str,
    *,
    eligible_keys: set[tuple[int, int]],
    roster: list[str],
) -> tuple[float, float, np.ndarray, int, int]:
    """Deterministic point replay of one season under ``mechanism``.

    ``weeks`` is the snapshot list from :func:`_extended_weeks` (computed once
    per season and shared by the point and draw replays).  Returns
    ``(x, y_paper, surv, n_eligible, n_weeks)`` where ``surv`` counts the alive
    weeks each roster contestant survives (ranked to ``r_Final`` by
    ``descending_rank``).
    """
    idx = {n: i for i, n in enumerate(roster)}
    A = np.ones(len(roster), dtype=bool)
    x_num = y_num = 0.0
    n_terms = 0
    surv = np.zeros(len(roster), dtype=float)
    n_eligible = 0
    for we in weeks:
        j = we["j"]
        mask = A & np.isfinite(j)
        if not mask.any():
            continue
        p_b = we["p_point"][mask]
        j_b = j[mask]
        names_b = np.asarray(roster)[mask]
        dx, dy = _week_contribution(p_b, j_b)
        x_num += dx
        y_num += dy
        n_terms += int(mask.sum())
        surv[mask] += 1.0
        if (int(season), int(we["week"])) in eligible_keys:
            n_eligible += 1
            elim, _ = simulate_week(p_b, j_b, names_b, mechanism)
            A[np.where(mask)[0][int(np.where(names_b == elim)[0][0])]] = False
        else:
            elim_idx = np.array([idx[n] for n in we["obs_elim"] if n in idx], dtype=int)
            A[elim_idx] = False
    x = x_num / max(n_terms, 1)
    y_paper = 1.0 - y_num / max(n_terms, 1)
    return x, y_paper, surv, n_eligible, len(weeks)


def _replay_season_draws(
    weeks: list[dict[str, Any]],
    season: int,
    mechanism: str,
    *,
    eligible_keys: set[tuple[int, int]],
    roster: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    """Per-draw replay of one season under ``mechanism`` (B trajectories).

    Every draw carries its own alive mask; at eligible weeks each draw's fan
    vector ``p^b`` drives the elimination, so the counterfactual trajectories
    (and hence ``x``/``y_paper``/``r_Final``) diverge across draws.  Returns
    ``(x_draws, y_paper_draws, surv_draws, n_eligible, n_weeks)`` with
    ``surv_draws`` shape ``(B, roster)``.
    """
    idx = {n: i for i, n in enumerate(roster)}
    n = len(roster)
    B = weeks[0]["p_draws"].shape[0]
    A = np.ones((B, n), dtype=bool)
    x_num = np.zeros(B)
    y_num = np.zeros(B)
    n_terms = np.zeros(B)
    surv = np.zeros((B, n), dtype=float)
    n_eligible = 0
    for we in weeks:
        j = we["j"]
        finite = np.isfinite(j)
        if (int(season), int(we["week"])) in eligible_keys:
            n_eligible += 1
            for b in range(B):
                mask = A[b] & finite
                if not mask.any():
                    continue
                p_b = we["p_draws"][b, mask]
                j_b = j[mask]
                names_b = np.asarray(roster)[mask]
                dx, dy = _week_contribution(p_b, j_b)
                x_num[b] += dx
                y_num[b] += dy
                n_terms[b] += int(mask.sum())
                surv[b, mask] += 1.0
                elim, _ = simulate_week(p_b, j_b, names_b, mechanism)
                A[b, np.where(mask)[0][int(np.where(names_b == elim)[0][0])]] = False
        else:
            elim_idx = np.array([idx[n] for n in we["obs_elim"] if n in idx], dtype=int)
            for b in range(B):
                mask = A[b] & finite
                if not mask.any():
                    continue
                p_b = we["p_draws"][b, mask]
                j_b = j[mask]
                names_b = np.asarray(roster)[mask]
                dx, dy = _week_contribution(p_b, j_b)
                x_num[b] += dx
                y_num[b] += dy
                n_terms[b] += int(mask.sum())
                surv[b, mask] += 1.0
                A[b, elim_idx] = False
    x_draws = x_num / np.maximum(n_terms, 1.0)
    y_draws = 1.0 - y_num / np.maximum(n_terms, 1.0)
    return x_draws, y_draws, surv, n_eligible, len(weeks)


def _review_y_from_surv(surv: np.ndarray, r_J: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Review's ``y = 1 - mu(|r_Final - r_J|)`` over the season roster.

    ``r_Final`` is the descending rank of the survival-week counts (longest
    survivor = winner = rank 1); ties get the pandas 'average' rank, matching
    ``descending_rank``.  ``surv`` may be shape ``(roster,)`` (point) or
    ``(B, roster)`` (draws); ``r_J`` is the full-roster judge standing and
    ``valid`` masks contestants with non-finite judge means, so the mean runs
    over the intersection of both rankings.
    """
    if surv.ndim == 1:
        rF = descending_rank(surv)
        return np.asarray(1.0 - float(np.abs(rF[valid] - r_J[valid]).mean()))
    rF_draws = np.stack([descending_rank(s) for s in surv])
    return np.asarray(1.0 - np.abs(rF_draws[:, valid] - r_J[None, valid]).mean(axis=1))


def season_phase_metrics(
    panel: pd.DataFrame,
    fit: PooledFit,
    cache: DrawCache,
    season: int,
    mechanism: str,
    *,
    eligible_keys: set[tuple[int, int]],
    train_keys: set[tuple[int, int]],
    alive_weeks: list[int],
    roster: list[str],
    r_J: np.ndarray,
    B: int,
    alpha: float,
) -> dict[str, Any]:
    """Phase-diagram statistics for one (season, mechanism) pair.

    Returns point / posterior-mean / ``1 - alpha`` interval for the paper's
    ``x = mu(|Ds|)`` and ``y = 1 - mu(|Dr|)`` and the review's
    ``y_review = 1 - mu(|r_Final - r_J|)``, all from the same replay.
    """
    lo_q, hi_q = alpha / 2.0, 1.0 - alpha / 2.0
    valid = np.isfinite(r_J)
    weeks = _extended_weeks(
        panel, fit, cache, season, alive_weeks, roster, B=B, train_keys=train_keys
    )
    x_p, y_p, surv_p, n_elig, n_weeks = _replay_season_point(
        weeks,
        season,
        mechanism,
        eligible_keys=eligible_keys,
        roster=roster,
    )
    y_rev_p = _review_y_from_surv(surv_p, r_J, valid)
    x_d, y_d, surv_d, _, _ = _replay_season_draws(
        weeks,
        season,
        mechanism,
        eligible_keys=eligible_keys,
        roster=roster,
    )
    y_rev_d = _review_y_from_surv(surv_d, r_J, valid)
    x_lo, x_hi = np.quantile(x_d, [lo_q, hi_q])
    y_lo, y_hi = np.quantile(y_d, [lo_q, hi_q])
    y_rev_lo, y_rev_hi = np.quantile(y_rev_d, [lo_q, hi_q])
    return {
        "season": int(season),
        "mechanism": mechanism,
        "x_point": float(x_p),
        "x_posterior_mean": float(x_d.mean()),
        f"x_ci_lo_{int(alpha * 100):02d}": float(x_lo),
        f"x_ci_hi_{int(alpha * 100):02d}": float(x_hi),
        "y_point": float(y_p),
        "y_posterior_mean": float(y_d.mean()),
        f"y_ci_lo_{int(alpha * 100):02d}": float(y_lo),
        f"y_ci_hi_{int(alpha * 100):02d}": float(y_hi),
        "y_review_point": float(y_rev_p),
        "y_review_posterior_mean": float(y_rev_d.mean()),
        f"y_review_ci_lo_{int(alpha * 100):02d}": float(y_rev_lo),
        f"y_review_ci_hi_{int(alpha * 100):02d}": float(y_rev_hi),
        "n_weeks": n_weeks,
        "n_eligible_weeks": n_elig,
        "B": B,
    }


def mechanism_phase_metrics(
    panel: pd.DataFrame,
    fit: PooledFit,
    mechanisms: tuple[str, ...] = MECHANISMS,
    *,
    B: int = 600,
    alpha: float = 0.10,
) -> pd.DataFrame:
    """Phase-diagram table over seasons x mechanisms (paper Fig. 5 backing data).

    One row per (season, mechanism) with the paper's ``x``/``y`` and the
    review's ``y_review``, point + posterior-propagated.  Seasons without any
    eligible mechanism week are included (``n_eligible_weeks == 0``); their
    ``x``/``y`` are mechanism-independent by construction and marked as such.
    """
    cfg = config_from_fit(fit, B=B)
    cache = DrawCache(panel, fit, cfg, max_B=B)
    train_keys = set(
        build_train_weeks(panel)[["season", "week"]].itertuples(index=False, name=None)
    )
    eligible_keys = set(eligible_weeks(panel))
    rows: list[dict[str, Any]] = []
    for season in sorted(panel["season"].unique()):
        alive_weeks = _season_alive_weeks(panel, int(season))
        if not alive_weeks:
            continue
        roster = _season_roster(panel, int(season))
        r_J = _season_judge_rank(panel, int(season)).reindex(roster).to_numpy(dtype=float)
        for mechanism in mechanisms:
            rows.append(
                season_phase_metrics(
                    panel,
                    fit,
                    cache,
                    int(season),
                    mechanism,
                    eligible_keys=eligible_keys,
                    train_keys=train_keys,
                    alive_weeks=alive_weeks,
                    roster=roster,
                    r_J=r_J,
                    B=B,
                    alpha=alpha,
                )
            )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Claim checks on the reproduced phase data (paper P-056/P-057, review R-040)
# --------------------------------------------------------------------------- #
def phase_claim_checks(
    df: pd.DataFrame,
    x_threshold: float = HIGH_FAN_INFLUENCE_X,
    alpha: float = 0.10,
) -> pd.DataFrame:
    """Quantitative checks of the paper's and review's phase-diagram claims.

    Rows (claim, y_column, n_seasons, mean_delta_y, mean_delta_x,
    supporting): the Direct -> Bottom-2 y lift and x change for each base rule
    (paper P-056), the tail-risk comparison on high fan-influence seasons
    (paper P-057), and the review's "Perc+Bottom2 highest overall x and y"
    ranking (R-040).  All deltas are over ``posterior_mean`` columns, so the
    comparison carries posterior-mean uncertainty rather than point estimates.
    """
    p = df.pivot_table(
        index="season",
        columns="mechanism",
        values=["x_posterior_mean", "y_posterior_mean", "y_review_posterior_mean"],
    )
    base_rank_d, base_rank_b = "rank_direct", "rank_bottom2"
    base_pct_d, base_pct_b = "pct_direct", "pct_bottom2"
    rows: list[dict[str, Any]] = []

    def _lift(base_d: str, base_b: str, ycol: str) -> None:
        d_y = p[(ycol, base_b)] - p[(ycol, base_d)]
        d_x = p[("x_posterior_mean", base_b)] - p[("x_posterior_mean", base_d)]
        rows.append(
            {
                "claim": f"Direct->Bottom2 y lift ({base_d.split('_')[0]})",
                "y_column": ycol,
                "n_seasons": int(d_y.notna().sum()),
                "mean_delta_y": float(d_y.mean()),
                "mean_delta_x": float(d_x.mean()),
                "mean_delta_y_lo": float(d_y.quantile(alpha / 2)),
                "mean_delta_y_hi": float(d_y.quantile(1 - alpha / 2)),
            }
        )

    _lift(base_rank_d, base_rank_b, "y_posterior_mean")
    _lift(base_pct_d, base_pct_b, "y_posterior_mean")

    hi = p[p[("x_posterior_mean", base_pct_d)] >= x_threshold]
    if not hi.empty:
        d_y = hi[("y_posterior_mean", base_pct_b)] - hi[("y_posterior_mean", base_pct_d)]
        rows.append(
            {
                "claim": f"Pct tail-risk: Bottom2 vs Direct y (x >= {x_threshold})",
                "y_column": "y_posterior_mean",
                "n_seasons": int(d_y.notna().sum()),
                "mean_delta_y": float(d_y.mean()),
                "mean_delta_x": float(
                    (
                        hi[("x_posterior_mean", base_pct_b)] - hi[("x_posterior_mean", base_pct_d)]
                    ).mean()
                ),
                "mean_delta_y_lo": float(d_y.quantile(alpha / 2)),
                "mean_delta_y_hi": float(d_y.quantile(1 - alpha / 2)),
            }
        )
    else:
        rows.append(
            {
                "claim": f"Pct tail-risk: Bottom2 vs Direct y (x >= {x_threshold})",
                "y_column": "y_posterior_mean",
                "n_seasons": 0,
                "mean_delta_y": float("nan"),
                "mean_delta_x": float("nan"),
                "mean_delta_y_lo": float("nan"),
                "mean_delta_y_hi": float("nan"),
                "supporting": (
                    "not testable: no season reaches the high fan-influence "
                    "threshold on the reproduced data"
                ),
            }
        )

    # Review claim R-040: Perc+Bottom2 has the highest overall x and y.
    for ycol in ("y_posterior_mean", "y_review_posterior_mean"):
        means = {m: float(p[(ycol, m)].mean()) for m in MECHANISMS}
        top = max(means.items(), key=lambda kv: kv[1])[0]
        rows.append(
            {
                "claim": f"Top mechanism by overall {ycol}",
                "y_column": ycol,
                "n_seasons": int(len(p)),
                "mean_delta_y": means[top],
                "mean_delta_x": float(p[("x_posterior_mean", top)].mean()),
                "mean_delta_y_lo": means[top],
                "mean_delta_y_hi": means[top],
                "supporting": (
                    "supports review claim"
                    if top == "pct_bottom2"
                    else f"disagrees with review claim (top = {top})"
                ),
            }
        )
    x_means = {m: float(p[("x_posterior_mean", m)].mean()) for m in MECHANISMS}
    top_x = max(x_means.items(), key=lambda kv: kv[1])[0]
    rows.append(
        {
            "claim": "Top mechanism by overall x",
            "y_column": "x_posterior_mean",
            "n_seasons": int(len(p)),
            "mean_delta_y": x_means[top_x],
            "mean_delta_x": x_means[top_x],
            "mean_delta_y_lo": x_means[top_x],
            "mean_delta_y_hi": x_means[top_x],
            "supporting": (
                "supports review claim"
                if top_x == "pct_bottom2"
                else f"disagrees with review claim (top = {top_x})"
            ),
        }
    )
    return pd.DataFrame(rows)
