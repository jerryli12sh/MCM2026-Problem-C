"""Problem 1 evaluation metrics.

Two families of metrics:

- ``evaluate_top1_accuracy``: posterior-mean reconstruction of the observed
  eliminatee over single-elimination training weeks (the reference rebuild's
  headline number), plus per-week PCP / ESS columns.
- ``build_event_tables`` + ``compute_cumulative_consistency``: the paper's
  season-path consistency ``S_s`` (B-04, ``S_bar ~ 0.78``) reproduced from the
  legacy ``eval_metrics_viz.py``.  The event tables use the ``"legacy"``
  posterior mode so final-week single eliminations are softmin-reweighted exactly
  as the legacy pipeline that produced the paper number did.

PCP / credible-interval / relative-width columns already live on
``posterior_summary`` (see ``track_p.infer_all_weekly_fan_support``); the
aggregates here mirror ``problem1_summary.json``.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from dwts_reproduction.problem1.config import Problem1Config
from dwts_reproduction.problem1.softmin import softmax_np
from dwts_reproduction.problem1.track_p import PooledFit, posterior_draws_for_week


# --------------------------------------------------------------------------- #
# Top-1 accuracy on training weeks
# --------------------------------------------------------------------------- #
def evaluate_top1_accuracy(
    panel: pd.DataFrame, posterior_summary: pd.DataFrame, train_weeks: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Evaluate whether posterior mean reconstructs observed eliminations."""
    by_week_rows: list[dict[str, Any]] = []
    train_keys = train_weeks.set_index(["season", "week"])["true_eliminatee"].to_dict()
    for (season, week), true_elim in train_keys.items():
        g = posterior_summary[
            (posterior_summary["season"].eq(season)) & (posterior_summary["week"].eq(week))
        ].copy()
        if g.empty:
            continue
        g["combined_score"] = g["j_metric"] + g["p_mean"]
        pred = g.sort_values(["combined_score", "celebrity_name"]).iloc[0]["celebrity_name"]
        pcp_w = (
            float(g["pcp_weighted"].dropna().iloc[0]) if g["pcp_weighted"].notna().any() else np.nan
        )
        pcp_u = (
            float(g["pcp_unweighted"].dropna().iloc[0])
            if g["pcp_unweighted"].notna().any()
            else np.nan
        )
        by_week_rows.append(
            {
                "season": int(season),
                "week": int(week),
                "true_eliminatee": true_elim,
                "pred_eliminatee": pred,
                "correct": bool(pred == true_elim),
                "alive_n": int(g["alive_n"].iloc[0]),
                "pcp_unweighted": pcp_u,
                "pcp_weighted": pcp_w,
                "ess_ratio": float(g["ess_ratio"].iloc[0]),
            }
        )
    by_week = pd.DataFrame(by_week_rows).sort_values(["season", "week"])
    by_season = (
        by_week.groupby("season", as_index=False)
        .agg(
            n_weeks=("correct", "size"),
            top1_accuracy=("correct", "mean"),
            mean_pcp_weighted=("pcp_weighted", "mean"),
            mean_ess_ratio=("ess_ratio", "mean"),
        )
        .sort_values("season")
    )
    summary = {
        "n_eval_weeks": int(len(by_week)),
        "overall_top1_accuracy": float(by_week["correct"].mean()) if len(by_week) else math.nan,
        "mean_pcp_unweighted": float(by_week["pcp_unweighted"].mean())
        if len(by_week)
        else math.nan,
        "mean_pcp_weighted": float(by_week["pcp_weighted"].mean()) if len(by_week) else math.nan,
        "mean_ess_ratio": float(by_week["ess_ratio"].mean()) if len(by_week) else math.nan,
    }
    return by_week, by_season, summary


# --------------------------------------------------------------------------- #
# Event tables (legacy semantics) for the paper's cumulative consistency
# --------------------------------------------------------------------------- #
def _parse_elim_list(s: str | list[Any]) -> list[str]:
    if isinstance(s, list):
        return [str(x) for x in s]
    return [x for x in str(s).split("|") if x != ""]


def build_event_tables(
    panel: pd.DataFrame,
    fit: PooledFit,
    df_elim_events: pd.DataFrame,
    config: Problem1Config,
) -> dict[str, pd.DataFrame]:
    """Build ``event_table`` / ``event_long`` from every elimination event.

    Mirrors ``src/build_event_table.py``: one row per elimination event (all 292,
    including finales and multi-eliminations) with the posterior-mean combined
    score ``c_hat = j_metric + p_mean`` for each alive contestant.  Posterior
    draws use the ``"legacy"`` mode so final-week single eliminations are
    softmin-reweighted exactly as the pipeline that produced ``S_bar = 0.78``.
    """
    tmp = df_elim_events.copy()
    if "Unnamed: 0" in tmp.columns:
        tmp = tmp.drop(columns=["Unnamed: 0"])
    if "elim_at_end_of_week" in tmp.columns:
        tmp = tmp.rename(columns={"elim_at_end_of_week": "week"})

    def _elim_list(value: Any) -> list[Any]:
        if isinstance(value, str):
            return list(_parse_elim_list(value)) if "|" in value else _literal_eval(value)
        return list(value)

    event_rows: list[dict[str, Any]] = []
    long_rows: list[dict[str, Any]] = []
    for _, row in tmp.iterrows():
        season = int(row["season"])
        week = int(row["week"])
        elim_list = [str(x) for x in _elim_list(row["eliminated"])]
        if len(elim_list) == 0:
            continue

        g = panel[
            (panel["season"] == season) & (panel["week"] == week) & (panel["alive"].astype(bool))
        ].copy()
        if g.empty:
            continue

        res = posterior_draws_for_week(
            panel, fit, season, week, config, has_posterior_mode="legacy"
        )
        alive = res["alive"].copy()
        p_mean = res["weights"] @ res["samples"]
        j = alive["j_metric"].to_numpy(dtype=np.float32)
        c_hat = j + p_mean
        pi_hat = softmax_np(-c_hat / config.tau_like)

        g = alive.copy()
        g["p_mean"] = p_mean
        g["c_hat"] = c_hat
        g["pi_hat"] = pi_hat
        g = g.sort_values("celebrity_name").reset_index(drop=True)

        alive_list = g["celebrity_name"].astype(str).tolist()
        event_rows.append(
            {
                "season": season,
                "week": week,
                "alive_list": "|".join(alive_list),
                "elim_obs_list": "|".join(elim_list),
                "m_elim": len(elim_list),
                "alive_n": int(len(alive_list)),
                "c_hat_list": "|".join(f"{x:.10f}" for x in g["c_hat"].to_numpy()),
                "pi_hat_list": "|".join(f"{x:.10f}" for x in g["pi_hat"].to_numpy()),
                "p_mean_list": "|".join(f"{x:.10f}" for x in g["p_mean"].to_numpy()),
                "has_posterior": bool(res["has_posterior"]),
                "ess": float(res["ess"]),
                "B": int(config.B),
            }
        )

        c_rank = pd.Series(g["c_hat"].to_numpy()).rank(method="average", ascending=True).to_numpy()
        g["risk_rank"] = c_rank
        for _, r in g.iterrows():
            long_rows.append(
                {
                    "season": season,
                    "week": week,
                    "celebrity_name": str(r["celebrity_name"]),
                    "c_hat": float(r["c_hat"]),
                    "pi_hat": float(r["pi_hat"]),
                    "p_mean": float(r["p_mean"]),
                    "j_metric": float(r["j_metric"]),
                    "risk_rank": float(r["risk_rank"]),
                    "alive_n": int(len(alive_list)),
                    "m_elim": int(len(elim_list)),
                    "is_elim_obs": str(r["celebrity_name"]) in elim_list,
                }
            )

    return {
        "event_table": pd.DataFrame(event_rows),
        "event_long": pd.DataFrame(long_rows),
    }


def _literal_eval(value: str) -> list[Any]:
    import ast

    parsed: list[Any] = ast.literal_eval(value)
    return parsed


# --------------------------------------------------------------------------- #
# Cumulative consistency S_s (paper P-030/P-031)
# --------------------------------------------------------------------------- #
def compute_cumulative_consistency(
    event_long: pd.DataFrame, event_table: pd.DataFrame, *, require_posterior: bool = False
) -> pd.DataFrame:
    """Season-path cumulative consistency ``S_s`` (paper equation, B-04).

    Mirrors ``eval_metrics_viz.compute_cum_consistency`` exactly: within each
    season, event ``k`` contributes ``(1/k) * |pred_cum & obs_cum| / |obs_cum|``
    where ``pred_set`` is the bottom-``m`` alive contestants by ``c_hat``; the
    season score is the harmonic normalization ``S_s = sum(terms) / H_K``.
    """
    rows: list[dict[str, Any]] = []
    for season, evs in event_table.groupby("season"):
        evs = evs.sort_values("week")
        obs_cum: set[str] = set()
        pred_cum: set[str] = set()
        terms: list[float] = []
        k = 0
        for _, e in evs.iterrows():
            s = int(e["season"])
            w = int(e["week"])
            elim_obs = set(_parse_elim_list(e["elim_obs_list"]))
            m = int(e["m_elim"])
            if require_posterior and ("has_posterior" in e) and (not bool(e["has_posterior"])):
                continue
            g = event_long[(event_long["season"] == s) & (event_long["week"] == w)].copy()
            if g.empty or m <= 0:
                continue
            k += 1
            pred_set = set(
                g.sort_values("c_hat", ascending=True).head(m)["celebrity_name"].tolist()
            )
            obs_cum |= elim_obs
            pred_cum |= pred_set
            n_prime = len(pred_cum & obs_cum)
            n = len(obs_cum)
            term = 0.0 if n == 0 else (1.0 / k) * (n_prime / n)
            terms.append(term)
        if k == 0:
            continue
        h_k = sum(1.0 / i for i in range(1, k + 1))
        s_s = sum(terms) / h_k if h_k > 0 else np.nan
        rows.append({"season": int(season), "K_s": int(k), "S_s": float(s_s)})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Summary helpers
# --------------------------------------------------------------------------- #
def summarize_posterior(
    posterior_summary: pd.DataFrame, accuracy_summary: dict[str, Any]
) -> dict[str, Any]:
    """Aggregate the metrics reported in ``problem1_summary.json``.

    ``mean_pcp_weighted`` / ``mean_ess_ratio`` / ``mean_ci_rel_width`` are pandas
    means over the full posterior summary (NaNs skipped), matching the reference
    ``save_problem1_outputs``.
    """
    return {
        "overall_top1_accuracy": accuracy_summary["overall_top1_accuracy"],
        "mean_pcp_weighted": float(posterior_summary["pcp_weighted"].mean()),
        "mean_ess_ratio": float(posterior_summary["ess_ratio"].mean()),
        "mean_ci_rel_width": float(posterior_summary["ci_rel_width"].mean()),
    }


def s_bar(summary: pd.DataFrame) -> float:
    """Mean cumulative consistency across seasons."""
    if summary.empty or summary["S_s"].isna().all():
        return math.nan
    return float(summary["S_s"].mean())
