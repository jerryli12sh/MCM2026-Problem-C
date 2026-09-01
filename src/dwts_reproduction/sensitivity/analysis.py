"""Sensitivity analysis over the Problem 1 fan-vote inference (paper A1-A4).

A faithful port of ``../src/sensitivity_analysis_a.py`` onto the repo's Problem 1
pipeline.  The four perturbation families from the paper's sensitivity section
(``2107542.tex`` lines 1060-1110):

- A1: grid over the likelihood temperature ``tau`` and prior tightness ``kappa``.
- A2: scan of the regularization balance ``lambda_u / lambda_beta``.
- A3: alternative judge-to-share transforms (softmax temperature, percentile).
- A4: leave-one-season-out refits.

Metric definitions
------------------
``pcp_mean`` (primary) is the mean over weeks of the paper-consistent PCP
(weighted fraction of posterior draws where the eliminated contestant has the
minimum combined score ``j_metric + p``), identical to
``infer_all_weekly_fan_support.pcp_weighted`` and the repo's registered baseline
PCP (B-05, 0.6043).  The legacy sensitivity script instead reported the weighted
softmin probability of the eliminated contestant; that is kept as the secondary
column ``pcp_softmin`` so the comparison stays explicit (D-20260901-20).
Similarly ``accuracy`` is the Problem-1 top-1 (argmin of ``j + p_mean``) and
``accuracy_softmin`` the legacy argmax-of-softmin-probability definition.

Seeds
-----
Each fit reseeds the global numpy RNG from ``config.seed`` (identical batch
sequences per scenario).  Posterior draws use the repo scheme
``config.seed + season*1000 + week``; the legacy script used ``seed + s*100 + w``.
No saved legacy sensitivity outputs exist to regress bit-for-bit, so the
reproduction target is the paper's qualitative claims (P-087..P-093); the seed
difference is immaterial and recorded in D-20260901-20.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np
import pandas as pd

from dwts_reproduction.preprocess import PreprocessTables
from dwts_reproduction.problem1.config import Problem1Config
from dwts_reproduction.problem1.panel import (
    build_feature_frame,
    build_problem1_panel,
    build_train_weeks,
)
from dwts_reproduction.problem1.track_p import (
    PooledFit,
    fit_pooled_softmin,
    posterior_draws_for_week,
    weighted_quantile,
)

EPS_TAU = 1e-12


# --------------------------------------------------------------------------- #
# Small numeric helpers
# --------------------------------------------------------------------------- #
def softmax_rows(z: np.ndarray) -> np.ndarray:
    """Row-wise softmax with max-shift stability."""
    z = np.asarray(z, dtype=float)
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def js_distance(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    """Jensen-Shannon divergence between two (possibly unnormalized) vectors."""
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    p = p / (p.sum() + eps)
    q = q / (q.sum() + eps)
    m = 0.5 * (p + q)
    kl_pm = float(np.sum(p * np.log((p + eps) / (m + eps))))
    kl_qm = float(np.sum(q * np.log((q + eps) / (m + eps))))
    return 0.5 * (kl_pm + kl_qm)


def spearman_corr(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman rank correlation, ``nan`` when undefined (constant input)."""
    a = np.asarray(a)
    b = np.asarray(b)
    if a.size == 0 or b.size == 0:
        return float("nan")
    ra = pd.Series(a).rank(method="average").to_numpy()
    rb = pd.Series(b).rank(method="average").to_numpy()
    if np.std(ra) < 1e-12 or np.std(rb) < 1e-12:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


# --------------------------------------------------------------------------- #
# Judge-signal variants (A3)
# --------------------------------------------------------------------------- #
def build_judge_rank_share_variant(
    df_long_judge: pd.DataFrame,
    base: pd.DataFrame,
    *,
    method: str = "softmax",
    temperature: float = 1.0,
) -> pd.DataFrame:
    """Rank-era judge shares under an alternative transform.

    ``method="softmax"`` divides the negative summed ranks by ``temperature``
    before the alive-set softmax (temperature 1.0 reproduces the default share);
    ``method="percentile"`` replaces them with the descending rank-percentile
    weights.  Mirrors ``sensitivity_analysis_a.build_judge_rank_share_variant``.
    """
    dj = df_long_judge.copy()
    for c in ["Unnamed: 0"]:
        if c in dj.columns:
            dj = dj.drop(columns=c)
    if "eligible" in dj.columns:
        dj = dj[dj["eligible"] == True].copy()  # noqa: E712
    if "is_show_week" in dj.columns:
        dj = dj[dj["is_show_week"] == True].copy()  # noqa: E712
    if "judge_score" not in dj.columns:
        raise KeyError("df_long_judge must have `judge_score`.")
    dj = dj[dj["judge_score"].notna()].copy()

    dj["judge_rank"] = dj.groupby(["season", "week", "judge"])["judge_score"].rank(
        ascending=False, method="average"
    )
    rank_sum = (
        dj.groupby(["season", "week", "celebrity_name"])
        .agg(rank_sum=("judge_rank", "sum"), n_judges=("judge_rank", "count"))
        .reset_index()
        .merge(
            base[["season", "week", "celebrity_name", "alive"]],
            on=["season", "week", "celebrity_name"],
            how="left",
        )
    )
    rank_sum = rank_sum[rank_sum["alive"] == True].copy()  # noqa: E712

    if method == "softmax":
        score = -rank_sum["rank_sum"].astype(float).to_numpy() / max(1e-8, float(temperature))

        def softmax_group(x: np.ndarray) -> np.ndarray:
            z = x - np.max(x)
            e = np.exp(z)
            return np.asarray(e / e.sum())

        rank_sum["judge_rank_share"] = (
            rank_sum.assign(_score=score)
            .groupby(["season", "week"])["_score"]
            .transform(softmax_group)
        )
    elif method == "percentile":

        def pct_group(x: pd.Series) -> np.ndarray:
            x = x.to_numpy(dtype=float)
            n = len(x)
            if n <= 1:
                return np.ones(n)
            order = x.argsort()
            ranks = np.empty_like(order)
            ranks[order] = np.arange(n)
            pct = 1.0 - ranks / (n - 1)
            s = pct.sum()
            if s <= 0:
                return np.ones(n) / n
            return np.asarray(pct / s)

        rank_sum["judge_rank_share"] = rank_sum.groupby(["season", "week"])["rank_sum"].transform(
            pct_group
        )
    else:
        raise ValueError(f"Unknown method: {method}")

    return rank_sum[["season", "week", "celebrity_name", "judge_rank_share"]].copy()


def build_panel_with_variant(
    tables: PreprocessTables,
    era_mode: str = "legacy",
    rank_method: str = "softmax",
    temperature: float = 1.0,
) -> pd.DataFrame:
    """Reassemble the Problem 1 panel with an alternative rank-era judge share.

    The panel is rebuilt through ``build_problem1_panel`` and the
    ``judge_rank_share`` column is replaced by the variant, then ``j_metric`` is
    recomputed with the era rule.  Rows missing from the variant (should not
    happen for alive rows) fall back to the default share.
    """
    base_panel = build_problem1_panel(tables, era_mode)
    orig_jrs = base_panel["judge_rank_share"].copy()
    base = base_panel[["season", "week", "celebrity_name", "alive"]].copy()
    jrs = build_judge_rank_share_variant(
        tables.long_judge, base, method=rank_method, temperature=temperature
    )
    panel = (
        base_panel.drop(columns=["judge_rank_share"])
        .merge(jrs, on=["season", "week", "celebrity_name"], how="left")
        .copy()
    )
    panel["judge_rank_share"] = panel["judge_rank_share"].fillna(orig_jrs)
    panel["j_metric"] = np.where(
        panel["era"].eq("percent"), panel["judge_percent"], panel["judge_rank_share"]
    )
    return panel


# --------------------------------------------------------------------------- #
# Weekly posterior and scenario metrics
# --------------------------------------------------------------------------- #
def infer_week_posterior(
    panel: pd.DataFrame,
    fit: PooledFit,
    config: Problem1Config,
    season: int,
    week: int,
) -> tuple[pd.DataFrame, dict[str, Any]] | None:
    """Conditional fan-share posterior for one week plus its metric row.

    Reuses the repo stage-2 sampler (``posterior_draws_for_week``) so the draws
    and importance weights are bit-identical to ``infer_all_weekly_fan_support``
    at the same configuration.  Returns ``None`` when the week is not a usable
    single-elimination, non-final week.
    """
    res = posterior_draws_for_week(
        panel, fit, int(season), int(week), config, has_posterior_mode="rebuild"
    )
    if not res["has_posterior"] or res["elim_pos"] is None:
        return None
    alive = res["alive"].reset_index(drop=True)
    p_samps = np.asarray(res["samples"], dtype=float)  # (B, n)
    w = np.asarray(res["weights"], dtype=float)  # (B,)
    elim_pos = int(res["elim_pos"])
    j = alive["j_metric"].to_numpy(dtype=float)
    n = len(alive)

    p_mean = w @ p_samps
    diff = p_samps - p_mean[None, :]
    sd = np.sqrt(np.sum(w[:, None] * diff**2, axis=0))
    cv = sd / (p_mean + config.eps)

    ci_lo = np.zeros(n, dtype=float)
    ci_hi = np.zeros(n, dtype=float)
    for i in range(n):
        lo, hi = weighted_quantile(p_samps[:, i], [0.05, 0.95], w)
        ci_lo[i] = float(lo)
        ci_hi[i] = float(hi)
    ciw = ci_hi - ci_lo

    # Paper-consistent argmin PCP (matches Problem 1's pcp_weighted).
    pred_pos_by_sample = np.argmin(j[None, :] + p_samps, axis=1)
    pcp_weighted = float(np.sum(w * (pred_pos_by_sample == elim_pos)))
    # Legacy sensitivity definition: weighted softmin probability of the eliminatee.
    soft = softmax_rows(-(j[None, :] + p_samps) / max(1e-8, float(config.tau_like)))
    elim_prob_post = np.sum(w[:, None] * soft, axis=0)
    pcp_softmin = float(elim_prob_post[elim_pos])

    accuracy = int(np.argmin(j + p_mean) == elim_pos)
    accuracy_softmin = int(np.argmax(elim_prob_post) == elim_pos)

    out = alive[
        ["season", "week", "celebrity_name", "era", "j_metric", "q_hat", "elim_this_week_end"]
    ].copy()
    out["p_mean"] = p_mean
    out["p_cv"] = cv
    out["p_ci_lo"] = ci_lo
    out["p_ci_hi"] = ci_hi
    out["p_ci_width"] = ciw

    week_info: dict[str, Any] = {
        "season": int(season),
        "week": int(week),
        "pcp_weighted": pcp_weighted,
        "pcp_softmin": pcp_softmin,
        "accuracy": accuracy,
        "accuracy_softmin": accuracy_softmin,
        "mean_cv": float(np.mean(cv)),
        "median_cv": float(np.median(cv)),
        "p25_cv": float(np.quantile(cv, 0.25)),
        "p75_cv": float(np.quantile(cv, 0.75)),
        "mean_ci_width": float(np.mean(ciw)),
        "median_ci_width": float(np.median(ciw)),
        "p25_ci_width": float(np.quantile(ciw, 0.25)),
        "p75_ci_width": float(np.quantile(ciw, 0.75)),
        "n_alive": int(n),
        "ess": float(res["ess"]),
        "ess_ratio": float(res["ess_ratio"]),
    }
    return out, week_info


def compute_metrics(
    panel: pd.DataFrame,
    fit: PooledFit,
    train_weeks: pd.DataFrame,
    config: Problem1Config,
    scenario_id: str,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Aggregate per-week metrics over a scenario's training weeks.

    Returns ``(post_df, week_df, summary)``.  ``post_df`` is the long
    contestant-week posterior frame (used by stability scatter and the saved
    tables); ``week_df`` is one row per processed week; ``summary`` is the
    scenario-level metric row.
    """
    week_rows: list[dict[str, Any]] = []
    post_rows: list[pd.DataFrame] = []
    for s, w in train_weeks[["season", "week"]].itertuples(index=False):
        res = infer_week_posterior(panel, fit, config, int(s), int(w))
        if res is None:
            continue
        post, info = res
        info["scenario_id"] = scenario_id
        week_rows.append(info)
        post["scenario_id"] = scenario_id
        post_rows.append(post)

    week_df = pd.DataFrame(week_rows)
    post_df = pd.concat(post_rows, ignore_index=True) if post_rows else pd.DataFrame()

    n_weeks = int(len(week_df))
    if week_df.empty:
        summary: dict[str, Any] = {
            "scenario_id": scenario_id,
            "n_weeks": 0,
            "n_weeks_total": int(len(train_weeks)),
            "pcp_mean": float("nan"),
            "pcp_median": float("nan"),
            "pcp_p25": float("nan"),
            "pcp_p75": float("nan"),
            "pcp_softmin_mean": float("nan"),
            "accuracy": float("nan"),
            "accuracy_softmin": float("nan"),
            "mean_cv": float("nan"),
            "median_cv": float("nan"),
            "mean_ci_width": float("nan"),
            "median_ci_width": float("nan"),
            "mean_ess_ratio": float("nan"),
        }
        return post_df, week_df, summary

    summary = {
        "scenario_id": scenario_id,
        "n_weeks": n_weeks,
        "n_weeks_total": int(len(train_weeks)),
        "pcp_mean": float(week_df["pcp_weighted"].mean()),
        "pcp_median": float(week_df["pcp_weighted"].median()),
        "pcp_p25": float(week_df["pcp_weighted"].quantile(0.25)),
        "pcp_p75": float(week_df["pcp_weighted"].quantile(0.75)),
        "pcp_softmin_mean": float(week_df["pcp_softmin"].mean()),
        "accuracy": float(week_df["accuracy"].mean()),
        "accuracy_softmin": float(week_df["accuracy_softmin"].mean()),
        "mean_cv": float(week_df["mean_cv"].mean()),
        "median_cv": float(week_df["median_cv"].median()),
        "mean_ci_width": float(week_df["mean_ci_width"].mean()),
        "median_ci_width": float(week_df["median_ci_width"].median()),
        "mean_ess_ratio": float(week_df["ess_ratio"].mean()),
    }
    return post_df, week_df, summary


def add_stability_metrics(
    summary: dict[str, Any],
    post_df: pd.DataFrame,
    baseline_post: pd.DataFrame,
) -> dict[str, Any]:
    """Append rank-stability metrics (Spearman, JS distance) vs the baseline."""
    if post_df.empty or baseline_post.empty:
        summary.update(
            {"spearman_p": float("nan"), "js_mean": float("nan"), "js_median": float("nan")}
        )
        return summary
    merged = post_df.merge(
        baseline_post[["season", "week", "celebrity_name", "p_mean"]],
        on=["season", "week", "celebrity_name"],
        how="inner",
        suffixes=("", "_base"),
    )
    if merged.empty:
        summary.update(
            {"spearman_p": float("nan"), "js_mean": float("nan"), "js_median": float("nan")}
        )
        return summary
    spearman = spearman_corr(merged["p_mean"].to_numpy(), merged["p_mean_base"].to_numpy())
    js_vals = [
        js_distance(g["p_mean"].to_numpy(), g["p_mean_base"].to_numpy())
        for (_, _), g in merged.groupby(["season", "week"])
    ]
    js_arr = np.asarray(js_vals, dtype=float)
    summary.update(
        {
            "spearman_p": float(spearman),
            "js_mean": float(np.mean(js_arr)) if js_arr.size else float("nan"),
            "js_median": float(np.median(js_arr)) if js_arr.size else float("nan"),
        }
    )
    return summary


def compute_u_var_share(panel: pd.DataFrame, fit: PooledFit, train_weeks: pd.DataFrame) -> float:
    """Share of logit variance explained by the contestant-season random effect."""
    df_feat, _ = build_feature_frame(panel, train_weeks)
    X = df_feat[fit.X_cols].to_numpy(dtype=float)
    beta = np.asarray(fit.beta, dtype=float)
    xbeta = X @ beta
    u = np.asarray(fit.u, dtype=float)
    cs_idx = df_feat["cs_idx"].to_numpy(dtype=int)
    u_part = u[cs_idx]
    var_u = float(np.var(u_part))
    var_x = float(np.var(xbeta))
    return var_u / (var_u + var_x + 1e-12)


# --------------------------------------------------------------------------- #
# Scenario families
# --------------------------------------------------------------------------- #
def build_grid_values(
    base: float, values: list[float] | None, multipliers: list[float]
) -> list[float]:
    """Explicit grid values, else base times the multipliers (deduplicated)."""
    if values:
        return sorted(set(values))
    return sorted({max(1e-6, base * m) for m in multipliers})


def select_nearby_grid(
    tau_vals: list[float],
    kappa_vals: list[float],
    *,
    base_tau: float,
    base_kappa: float,
    grid_n: int | None,
) -> list[tuple[float, float]]:
    """Keep the ``grid_n`` points nearest (tau, kappa) to the baseline pair."""
    grid = [(t, k) for t in tau_vals for k in kappa_vals]
    if not grid_n or grid_n >= len(grid):
        return grid

    def dist(pair: tuple[float, float]) -> float:
        t, k = pair
        dt = (t / base_tau) - 1.0
        dk = (k / base_kappa) - 1.0
        return float(dt * dt + dk * dk)

    return sorted(grid, key=dist)[: max(6, grid_n)]


def run_a1_grid(
    panel: pd.DataFrame,
    base_config: Problem1Config,
    train_weeks: pd.DataFrame,
    *,
    tau_vals: list[float],
    kappa_vals: list[float],
    grid_n: int | None,
    baseline_post: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """A1: refit per ``tau``, then posterior over the (tau, kappa) grid."""
    grid = select_nearby_grid(
        tau_vals,
        kappa_vals,
        base_tau=base_config.tau_train,
        base_kappa=base_config.kappa,
        grid_n=grid_n,
    )
    summaries: list[dict[str, Any]] = []
    all_week: list[pd.DataFrame] = []
    all_post: list[pd.DataFrame] = []

    by_tau: dict[float, PooledFit] = {}
    for tau, _ in grid:
        if tau in by_tau:
            continue
        fit_cfg = replace(base_config, tau_train=tau)
        by_tau[tau] = fit_pooled_softmin(panel, train_weeks, fit_cfg)

    for tau, kappa in grid:
        fit = by_tau[tau]
        post_cfg = replace(base_config, kappa=kappa, tau_like=tau)
        scenario_id = f"A1_tau{tau:.4g}_kappa{kappa:.4g}"
        post_df, week_df, summary = compute_metrics(panel, fit, train_weeks, post_cfg, scenario_id)
        summary.update({"scenario": "A1_grid", "tau": tau, "kappa": kappa})
        summary = add_stability_metrics(summary, post_df, baseline_post)
        summaries.append(summary)
        all_week.append(week_df)
        all_post.append(post_df)

    return (
        pd.DataFrame(summaries),
        pd.concat(all_week, ignore_index=True) if all_week else pd.DataFrame(),
        pd.concat(all_post, ignore_index=True) if all_post else pd.DataFrame(),
    )


def run_a2_lambda_scan(
    panel: pd.DataFrame,
    base_config: Problem1Config,
    train_weeks: pd.DataFrame,
    *,
    ratios: list[float],
    baseline_post: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """A2: scan ``lambda_u = lambda_beta * ratio``, reporting u variance share."""
    summaries: list[dict[str, Any]] = []
    all_week: list[pd.DataFrame] = []
    all_post: list[pd.DataFrame] = []

    for r in ratios:
        fit_cfg = replace(base_config, l2_u=base_config.l2_beta * r)
        fit = fit_pooled_softmin(panel, train_weeks, fit_cfg)
        scenario_id = f"A2_ratio{r:.3g}"
        post_df, week_df, summary = compute_metrics(
            panel, fit, train_weeks, base_config, scenario_id
        )
        summary.update(
            {
                "scenario": "A2_lambda_ratio",
                "l2_beta": base_config.l2_beta,
                "l2_u": fit_cfg.l2_u,
                "lambda_ratio": r,
                "u_var_share": compute_u_var_share(panel, fit, train_weeks),
            }
        )
        summary = add_stability_metrics(summary, post_df, baseline_post)
        summaries.append(summary)
        all_week.append(week_df)
        all_post.append(post_df)

    return (
        pd.DataFrame(summaries),
        pd.concat(all_week, ignore_index=True) if all_week else pd.DataFrame(),
        pd.concat(all_post, ignore_index=True) if all_post else pd.DataFrame(),
    )


def run_a3_judge_transform(
    tables: PreprocessTables,
    base_config: Problem1Config,
    *,
    temperatures: list[float],
    include_percentile: bool,
    baseline_post: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """A3: rebuild the panel with alternative judge transforms and refit."""
    summaries: list[dict[str, Any]] = []
    all_week: list[pd.DataFrame] = []
    all_post: list[pd.DataFrame] = []

    era_mode = base_config.era_mode

    for T in temperatures:
        panel = build_panel_with_variant(
            tables, era_mode=era_mode, rank_method="softmax", temperature=T
        )
        train_weeks_v = build_train_weeks(panel)
        fit = fit_pooled_softmin(panel, train_weeks_v, base_config)
        scenario_id = f"A3_softmaxT{T:.3g}"
        post_df, week_df, summary = compute_metrics(
            panel, fit, train_weeks_v, base_config, scenario_id
        )
        summary.update(
            {"scenario": "A3_judge_transform", "rank_method": "softmax", "temperature": T}
        )
        summary = add_stability_metrics(summary, post_df, baseline_post)
        summaries.append(summary)
        all_week.append(week_df)
        all_post.append(post_df)

    if include_percentile:
        panel = build_panel_with_variant(
            tables, era_mode=era_mode, rank_method="percentile", temperature=1.0
        )
        train_weeks_v = build_train_weeks(panel)
        fit = fit_pooled_softmin(panel, train_weeks_v, base_config)
        scenario_id = "A3_percentile"
        post_df, week_df, summary = compute_metrics(
            panel, fit, train_weeks_v, base_config, scenario_id
        )
        summary.update(
            {
                "scenario": "A3_judge_transform",
                "rank_method": "percentile",
                "temperature": float("nan"),
            }
        )
        summary = add_stability_metrics(summary, post_df, baseline_post)
        summaries.append(summary)
        all_week.append(week_df)
        all_post.append(post_df)

    return (
        pd.DataFrame(summaries),
        pd.concat(all_week, ignore_index=True) if all_week else pd.DataFrame(),
        pd.concat(all_post, ignore_index=True) if all_post else pd.DataFrame(),
    )


def run_a4_leave_one_season_out(
    panel: pd.DataFrame,
    base_config: Problem1Config,
    *,
    baseline_post: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """A4: leave-one-season-out refits; metrics on the held-in seasons only."""
    seasons = sorted(panel["season"].unique().tolist())
    summaries: list[dict[str, Any]] = []
    all_week: list[pd.DataFrame] = []
    all_post: list[pd.DataFrame] = []

    for s in seasons:
        panel_sub = panel[panel["season"] != s].copy()
        train_weeks_sub = build_train_weeks(panel_sub)
        fit = fit_pooled_softmin(panel_sub, train_weeks_sub, base_config)
        scenario_id = f"A4_leaveS{s}"
        post_df, week_df, summary = compute_metrics(
            panel_sub, fit, train_weeks_sub, base_config, scenario_id
        )
        summary.update({"scenario": "A4_leave_one_season_out", "dropped_season": int(s)})
        summary = add_stability_metrics(summary, post_df, baseline_post)
        summaries.append(summary)
        all_week.append(week_df)
        all_post.append(post_df)

    return (
        pd.DataFrame(summaries),
        pd.concat(all_week, ignore_index=True) if all_week else pd.DataFrame(),
        pd.concat(all_post, ignore_index=True) if all_post else pd.DataFrame(),
    )
