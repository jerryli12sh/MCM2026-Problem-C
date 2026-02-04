import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

import model


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def weighted_quantile(values: np.ndarray, quantiles: Iterable[float], weights: np.ndarray) -> np.ndarray:
    values = np.asarray(values)
    quantiles = np.asarray(quantiles)
    w = np.asarray(weights)
    sorter = np.argsort(values)
    v = values[sorter]
    w = w[sorter]
    cdf = np.cumsum(w)
    if cdf[-1] <= 0:
        return np.quantile(values, quantiles)
    cdf = cdf / cdf[-1]
    return np.interp(quantiles, cdf, v)


def softmax_rows(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def js_distance(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    p = p / (p.sum() + eps)
    q = q / (q.sum() + eps)
    m = 0.5 * (p + q)
    kl_pm = np.sum(p * np.log((p + eps) / (m + eps)))
    kl_qm = np.sum(q * np.log((q + eps) / (m + eps)))
    return 0.5 * (kl_pm + kl_qm)


def spearman_corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a)
    b = np.asarray(b)
    if a.size == 0 or b.size == 0:
        return np.nan
    ra = pd.Series(a).rank(method="average").to_numpy()
    rb = pd.Series(b).rank(method="average").to_numpy()
    if np.std(ra) < 1e-12 or np.std(rb) < 1e-12:
        return np.nan
    return float(np.corrcoef(ra, rb)[0, 1])


def build_judge_rank_share_variant(
    df_long_judge: pd.DataFrame,
    base: pd.DataFrame,
    *,
    method: str = "softmax",
    temperature: float = 1.0,
) -> pd.DataFrame:
    dj = df_long_judge.copy()
    for c in ["Unnamed: 0"]:
        if c in dj.columns:
            dj = dj.drop(columns=c)
    if "eligible" in dj.columns:
        dj = dj[dj["eligible"] == True].copy()
    if "is_show_week" in dj.columns:
        dj = dj[dj["is_show_week"] == True].copy()
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
    )

    rank_sum = rank_sum.merge(
        base[["season", "week", "celebrity_name", "alive"]],
        on=["season", "week", "celebrity_name"], how="left"
    )
    rank_sum = rank_sum[rank_sum["alive"] == True].copy()

    if method == "softmax":
        score = -rank_sum["rank_sum"].astype(float).to_numpy() / max(1e-8, float(temperature))

        def softmax_group(x):
            z = x - np.max(x)
            e = np.exp(z)
            return e / e.sum()

        rank_sum["judge_rank_share"] = (
            rank_sum.assign(_score=score)
            .groupby(["season", "week"])["_score"]
            .transform(softmax_group)
        )
    elif method == "percentile":
        def pct_group(x):
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
            return pct / s

        rank_sum["judge_rank_share"] = (
            rank_sum.groupby(["season", "week"])["rank_sum"].transform(pct_group)
        )
    else:
        raise ValueError(f"Unknown method: {method}")

    return rank_sum[["season", "week", "celebrity_name", "judge_rank_share"]].copy()


def build_panel_with_variant(
    *,
    era_cutoff: int = 28,
    rank_method: str = "softmax",
    temperature: float = 1.0,
) -> pd.DataFrame:
    df_elim_events, df_roster, df_weekly, df_long_judge, df_clean = model.load_tables()
    elim_long = model.build_elim_long(df_elim_events)
    base = model.build_base(df_roster, elim_long, df_clean)
    judge_percent = model.build_judge_percent(df_weekly, base)
    judge_rank_share = build_judge_rank_share_variant(
        df_long_judge, base, method=rank_method, temperature=temperature
    )
    panel = model.build_panel(base, judge_percent, judge_rank_share, era_cutoff=era_cutoff)
    return panel


def infer_week_posterior(
    panel: pd.DataFrame,
    pooled_fit: dict,
    season: int,
    week: int,
    *,
    kappa: float,
    tau_like: float,
    B: int,
    seed: int,
) -> Optional[Tuple[pd.DataFrame, dict]]:
    rng = np.random.default_rng(seed)
    g = model.pooled_q_for_week(panel, pooled_fit, season=season, week=week).copy()
    alive = g.copy()
    if alive.shape[0] <= 1:
        return None
    if alive["elim_this_week_end"].sum() != 1:
        return None

    elim_name = alive.loc[alive["elim_this_week_end"], "celebrity_name"].iloc[0]
    names = alive["celebrity_name"].to_numpy()
    elim_pos = int(np.where(names == elim_name)[0][0])

    q = alive["q_hat"].to_numpy(dtype=float)
    alpha = np.maximum(1e-8, kappa * q)
    p_samps = rng.dirichlet(alpha, size=B)

    j = alive["j_metric"].to_numpy(dtype=float)
    cost = j[None, :] + p_samps
    elim_prob = softmax_rows(-cost / max(1e-8, float(tau_like)))

    w = elim_prob[:, elim_pos]
    if np.sum(w) <= 0:
        w = np.ones(B, dtype=float) / B
    else:
        w = w / np.sum(w)

    p_mean = (w[:, None] * p_samps).sum(axis=0)
    var = (w[:, None] * (p_samps - p_mean) ** 2).sum(axis=0)
    sd = np.sqrt(var)
    cv = sd / (p_mean + 1e-12)

    ci_lo = np.zeros_like(p_mean)
    ci_hi = np.zeros_like(p_mean)
    for i in range(p_mean.shape[0]):
        lo, hi = weighted_quantile(p_samps[:, i], [0.05, 0.95], w)
        ci_lo[i] = lo
        ci_hi[i] = hi
    ciw = ci_hi - ci_lo

    elim_prob_post = (w[:, None] * elim_prob).sum(axis=0)
    pcp = float(elim_prob_post[elim_pos])
    pred_elim_pos = int(np.argmax(elim_prob_post))
    accuracy = 1 if pred_elim_pos == elim_pos else 0

    out = alive[[
        "season", "week", "celebrity_name", "era", "j_metric", "q_hat", "elim_this_week_end"
    ]].copy()
    out["p_mean"] = p_mean
    out["p_cv"] = cv
    out["p_ci_lo"] = ci_lo
    out["p_ci_hi"] = ci_hi
    out["p_ci_width"] = ciw

    week_info = {
        "season": int(season),
        "week": int(week),
        "pcp": pcp,
        "accuracy": accuracy,
        "mean_cv": float(np.mean(cv)),
        "median_cv": float(np.median(cv)),
        "p25_cv": float(np.quantile(cv, 0.25)),
        "p75_cv": float(np.quantile(cv, 0.75)),
        "mean_ci_width": float(np.mean(ciw)),
        "median_ci_width": float(np.median(ciw)),
        "p25_ci_width": float(np.quantile(ciw, 0.25)),
        "p75_ci_width": float(np.quantile(ciw, 0.75)),
        "n_alive": int(len(cv)),
    }
    return out, week_info


def compute_metrics(
    panel: pd.DataFrame,
    pooled_fit: dict,
    train_weeks: pd.DataFrame,
    *,
    kappa: float,
    tau_like: float,
    B: int,
    seed: int,
    scenario_id: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
    week_rows = []
    post_rows = []

    for _, row in train_weeks.iterrows():
        s = int(row["season"])
        w = int(row["week"])
        wk_seed = seed + s * 100 + w
        res = infer_week_posterior(
            panel, pooled_fit, s, w, kappa=kappa, tau_like=tau_like, B=B, seed=wk_seed
        )
        if res is None:
            continue
        post, info = res
        info["scenario_id"] = scenario_id
        week_rows.append(info)
        post["scenario_id"] = scenario_id
        post_rows.append(post)

    week_df = pd.DataFrame(week_rows)
    post_df = pd.concat(post_rows, ignore_index=True) if post_rows else pd.DataFrame()

    if week_df.empty:
        summary = {
            "scenario_id": scenario_id,
            "pcp_mean": np.nan,
            "pcp_median": np.nan,
            "pcp_p25": np.nan,
            "pcp_p75": np.nan,
            "accuracy": np.nan,
            "mean_cv": np.nan,
            "median_cv": np.nan,
            "mean_ci_width": np.nan,
            "median_ci_width": np.nan,
            "n_weeks": 0,
        }
        return post_df, week_df, summary

    summary = {
        "scenario_id": scenario_id,
        "pcp_mean": float(week_df["pcp"].mean()),
        "pcp_median": float(week_df["pcp"].median()),
        "pcp_p25": float(week_df["pcp"].quantile(0.25)),
        "pcp_p75": float(week_df["pcp"].quantile(0.75)),
        "accuracy": float(week_df["accuracy"].mean()),
        "mean_cv": float(week_df["mean_cv"].mean()),
        "median_cv": float(week_df["median_cv"].median()),
        "mean_ci_width": float(week_df["mean_ci_width"].mean()),
        "median_ci_width": float(week_df["median_ci_width"].median()),
        "n_weeks": int(len(week_df)),
    }
    return post_df, week_df, summary


def add_stability_metrics(
    summary: dict,
    post_df: pd.DataFrame,
    baseline_post: pd.DataFrame,
) -> dict:
    if post_df.empty or baseline_post.empty:
        summary.update({"spearman_p": np.nan, "js_mean": np.nan, "js_median": np.nan})
        return summary

    merged = post_df.merge(
        baseline_post[["season", "week", "celebrity_name", "p_mean"]],
        on=["season", "week", "celebrity_name"],
        how="inner",
        suffixes=("", "_base"),
    )
    if merged.empty:
        summary.update({"spearman_p": np.nan, "js_mean": np.nan, "js_median": np.nan})
        return summary

    spearman = spearman_corr(merged["p_mean"].to_numpy(), merged["p_mean_base"].to_numpy())

    js_vals = []
    for (s, w), g in merged.groupby(["season", "week"]):
        js = js_distance(g["p_mean"].to_numpy(), g["p_mean_base"].to_numpy())
        js_vals.append(js)

    js_vals = np.asarray(js_vals)
    summary.update({
        "spearman_p": float(spearman),
        "js_mean": float(np.mean(js_vals)) if js_vals.size else np.nan,
        "js_median": float(np.median(js_vals)) if js_vals.size else np.nan,
    })
    return summary


def build_grid_values(base: float, values: Optional[List[float]], multipliers: List[float]) -> List[float]:
    if values:
        return values
    out = sorted({max(1e-6, base * m) for m in multipliers})
    return out


def select_nearby_grid(
    tau_vals: List[float],
    kappa_vals: List[float],
    *,
    base_tau: float,
    base_kappa: float,
    grid_n: Optional[int],
) -> List[Tuple[float, float]]:
    grid = [(t, k) for t in tau_vals for k in kappa_vals]
    if not grid_n or grid_n >= len(grid):
        return grid
    def dist(pair):
        t, k = pair
        dt = (t / base_tau) - 1.0
        dk = (k / base_kappa) - 1.0
        return dt * dt + dk * dk
    grid_sorted = sorted(grid, key=dist)
    min_n = max(6, grid_n)
    return grid_sorted[:min_n]


def compute_u_var_share(panel: pd.DataFrame, pooled_fit: dict) -> float:
    train_weeks, train_rows, _ = model.build_train_weeks(panel)
    df_feat, _ = model._build_features(panel, train_rows)
    X = df_feat[pooled_fit["X_cols"]].to_numpy(dtype=float)
    beta = pooled_fit["beta_hat"].astype(float)
    xbeta = X @ beta
    u = pooled_fit["u_hat"].astype(float)
    cs_idx = df_feat["cs_idx"].to_numpy(dtype=int)
    u_part = u[cs_idx]
    var_u = float(np.var(u_part))
    var_x = float(np.var(xbeta))
    denom = var_u + var_x + 1e-12
    return var_u / denom


def run_a1_grid(
    panel: pd.DataFrame,
    base_fit: dict,
    base_train_weeks: pd.DataFrame,
    *,
    tau_vals: List[float],
    kappa_vals: List[float],
    grid_n: Optional[int],
    B: int,
    seed: int,
    outdir: Path,
    baseline_post: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    grid = select_nearby_grid(
        tau_vals, kappa_vals, base_tau=base_fit["tau"], base_kappa=base_fit["kappa"], grid_n=grid_n
    )
    summaries = []
    all_week = []
    all_post = []

    by_tau: Dict[float, Tuple[dict, pd.DataFrame]] = {}
    for tau, _ in grid:
        if tau in by_tau:
            continue
        model_tau, fit_tau, train_weeks_tau = model.train_pooled_model(
            panel,
            seed=seed,
            tau=tau,
            l2_beta=base_fit["hyperparams"]["l2_beta"],
            l2_u=base_fit["hyperparams"]["l2_u"],
            kappa=base_fit["kappa"],
            lr=base_fit["hyperparams"]["lr"],
            n_steps=base_fit["hyperparams"]["n_steps"],
            batch_size=base_fit["hyperparams"]["batch_size"],
        )
        by_tau[tau] = (fit_tau, train_weeks_tau)

    for tau, kappa in grid:
        fit_tau, train_weeks_tau = by_tau[tau]
        scenario_id = f"A1_tau{tau:.4g}_kappa{kappa:.4g}"
        post_df, week_df, summary = compute_metrics(
            panel, fit_tau, train_weeks_tau, kappa=kappa, tau_like=tau, B=B, seed=seed, scenario_id=scenario_id
        )
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
    base_fit: dict,
    base_train_weeks: pd.DataFrame,
    *,
    ratios: List[float],
    B: int,
    seed: int,
    baseline_post: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summaries = []
    all_week = []
    all_post = []

    l2_beta = base_fit["hyperparams"]["l2_beta"]

    for r in ratios:
        l2_u = l2_beta * r
        model_r, fit_r, train_weeks_r = model.train_pooled_model(
            panel,
            seed=seed,
            tau=base_fit["tau"],
            l2_beta=l2_beta,
            l2_u=l2_u,
            kappa=base_fit["kappa"],
            lr=base_fit["hyperparams"]["lr"],
            n_steps=base_fit["hyperparams"]["n_steps"],
            batch_size=base_fit["hyperparams"]["batch_size"],
        )
        scenario_id = f"A2_ratio{r:.3g}"
        post_df, week_df, summary = compute_metrics(
            panel, fit_r, train_weeks_r, kappa=base_fit["kappa"], tau_like=base_fit["tau"],
            B=B, seed=seed, scenario_id=scenario_id
        )
        summary.update({
            "scenario": "A2_lambda_ratio",
            "l2_beta": l2_beta,
            "l2_u": l2_u,
            "lambda_ratio": r,
            "u_var_share": compute_u_var_share(panel, fit_r),
        })
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
    *,
    base_fit: dict,
    era_cutoff: int,
    temperatures: List[float],
    include_percentile: bool,
    B: int,
    seed: int,
    baseline_post: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summaries = []
    all_week = []
    all_post = []

    for T in temperatures:
        panel = build_panel_with_variant(era_cutoff=era_cutoff, rank_method="softmax", temperature=T)
        model_t, fit_t, train_weeks_t = model.train_pooled_model(
            panel,
            seed=seed,
            tau=base_fit["tau"],
            l2_beta=base_fit["hyperparams"]["l2_beta"],
            l2_u=base_fit["hyperparams"]["l2_u"],
            kappa=base_fit["kappa"],
            lr=base_fit["hyperparams"]["lr"],
            n_steps=base_fit["hyperparams"]["n_steps"],
            batch_size=base_fit["hyperparams"]["batch_size"],
        )
        scenario_id = f"A3_softmaxT{T:.3g}"
        post_df, week_df, summary = compute_metrics(
            panel, fit_t, train_weeks_t, kappa=base_fit["kappa"], tau_like=base_fit["tau"],
            B=B, seed=seed, scenario_id=scenario_id
        )
        summary.update({"scenario": "A3_judge_transform", "rank_method": "softmax", "temperature": T})
        summary = add_stability_metrics(summary, post_df, baseline_post)

        summaries.append(summary)
        all_week.append(week_df)
        all_post.append(post_df)

    if include_percentile:
        panel = build_panel_with_variant(era_cutoff=era_cutoff, rank_method="percentile", temperature=1.0)
        model_p, fit_p, train_weeks_p = model.train_pooled_model(
            panel,
            seed=seed,
            tau=base_fit["tau"],
            l2_beta=base_fit["hyperparams"]["l2_beta"],
            l2_u=base_fit["hyperparams"]["l2_u"],
            kappa=base_fit["kappa"],
            lr=base_fit["hyperparams"]["lr"],
            n_steps=base_fit["hyperparams"]["n_steps"],
            batch_size=base_fit["hyperparams"]["batch_size"],
        )
        scenario_id = "A3_percentile"
        post_df, week_df, summary = compute_metrics(
            panel, fit_p, train_weeks_p, kappa=base_fit["kappa"], tau_like=base_fit["tau"],
            B=B, seed=seed, scenario_id=scenario_id
        )
        summary.update({"scenario": "A3_judge_transform", "rank_method": "percentile", "temperature": np.nan})
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
    base_fit: dict,
    *,
    B: int,
    seed: int,
    baseline_post: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    seasons = sorted(panel["season"].unique().tolist())
    summaries = []
    all_week = []
    all_post = []

    for s in seasons:
        panel_sub = panel[panel["season"] != s].copy()
        model_s, fit_s, train_weeks_s = model.train_pooled_model(
            panel_sub,
            seed=seed,
            tau=base_fit["tau"],
            l2_beta=base_fit["hyperparams"]["l2_beta"],
            l2_u=base_fit["hyperparams"]["l2_u"],
            kappa=base_fit["kappa"],
            lr=base_fit["hyperparams"]["lr"],
            n_steps=base_fit["hyperparams"]["n_steps"],
            batch_size=base_fit["hyperparams"]["batch_size"],
        )
        scenario_id = f"A4_leaveS{s}"
        post_df, week_df, summary = compute_metrics(
            panel_sub, fit_s, train_weeks_s, kappa=base_fit["kappa"], tau_like=base_fit["tau"],
            B=B, seed=seed, scenario_id=scenario_id
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


def parse_float_list(text: Optional[str]) -> Optional[List[float]]:
    if text is None or text.strip() == "":
        return None
    return [float(x) for x in text.split(",")]


def main() -> None:
    parser = argparse.ArgumentParser(description="Sensitivity analysis for fan vote inference (A1-A4).")
    parser.add_argument("--outdir", type=str, default="sensitivity_outputs", help="Output directory.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--B", type=int, default=800, help="Posterior draws per week.")
    parser.add_argument("--era-cutoff", type=int, default=28)
    parser.add_argument("--run-a1", action="store_true", help="Run A1 grid sensitivity.")
    parser.add_argument("--run-a2", action="store_true", help="Run A2 lambda ratio scan.")
    parser.add_argument("--run-a3", action="store_true", help="Run A3 judge transform sensitivity.")
    parser.add_argument("--run-a4", action="store_true", help="Run A4 leave-one-season-out.")
    parser.add_argument("--tau-values", type=str, default=None, help="Comma list for tau grid.")
    parser.add_argument("--kappa-values", type=str, default=None, help="Comma list for kappa grid.")
    parser.add_argument("--grid-n", type=int, default=15, help="Nearest grid points to keep.")
    parser.add_argument("--ratio-values", type=str, default="0.25,0.5,1,2,4", help="Comma list for l2_u/l2_beta ratios.")
    parser.add_argument("--temperatures", type=str, default="0.5,1,2", help="Comma list for rank softmax temperatures.")
    parser.add_argument("--no-percentile", action="store_true", help="Skip percentile rank variant.")

    args = parser.parse_args()

    run_any = args.run_a1 or args.run_a2 or args.run_a3 or args.run_a4
    if not run_any:
        args.run_a1 = args.run_a2 = args.run_a3 = args.run_a4 = True

    outdir = Path(args.outdir)
    ensure_dir(outdir)

    panel = build_panel_with_variant(era_cutoff=args.era_cutoff, rank_method="softmax", temperature=1.0)

    base_model, base_fit, base_train_weeks = model.train_pooled_model(
        panel,
        seed=args.seed,
    )

    baseline_post, baseline_week, baseline_summary = compute_metrics(
        panel, base_fit, base_train_weeks, kappa=base_fit["kappa"], tau_like=base_fit["tau"],
        B=args.B, seed=args.seed, scenario_id="baseline"
    )

    baseline_summary.update({"scenario": "baseline", "tau": base_fit["tau"], "kappa": base_fit["kappa"]})

    baseline_week.to_csv(outdir / "baseline_week.csv", index=False)
    baseline_post.to_csv(outdir / "baseline_post.csv", index=False)
    pd.DataFrame([baseline_summary]).to_csv(outdir / "baseline_summary.csv", index=False)

    summary_tables = [pd.DataFrame([baseline_summary])]

    if args.run_a1:
        tau_vals = build_grid_values(base_fit["tau"], parse_float_list(args.tau_values), [0.5, 1, 1.5, 2, 3, 4])
        kappa_vals = build_grid_values(base_fit["kappa"], parse_float_list(args.kappa_values), [0.5, 1, 2, 3, 5, 10])
        a1_sum, a1_week, a1_post = run_a1_grid(
            panel, base_fit, base_train_weeks, tau_vals=tau_vals, kappa_vals=kappa_vals,
            grid_n=args.grid_n, B=args.B, seed=args.seed, outdir=outdir, baseline_post=baseline_post
        )
        a1_sum.to_csv(outdir / "A1_grid_summary.csv", index=False)
        a1_week.to_csv(outdir / "A1_grid_week.csv", index=False)
        a1_post.to_csv(outdir / "A1_grid_post.csv", index=False)
        summary_tables.append(a1_sum)

    if args.run_a2:
        ratios = parse_float_list(args.ratio_values) or [0.25, 0.5, 1, 2, 4]
        a2_sum, a2_week, a2_post = run_a2_lambda_scan(
            panel, base_fit, base_train_weeks, ratios=ratios, B=args.B, seed=args.seed,
            baseline_post=baseline_post
        )
        a2_sum.to_csv(outdir / "A2_lambda_summary.csv", index=False)
        a2_week.to_csv(outdir / "A2_lambda_week.csv", index=False)
        a2_post.to_csv(outdir / "A2_lambda_post.csv", index=False)
        summary_tables.append(a2_sum)

    if args.run_a3:
        temps = parse_float_list(args.temperatures) or [0.5, 1.0, 2.0]
        a3_sum, a3_week, a3_post = run_a3_judge_transform(
            base_fit=base_fit,
            era_cutoff=args.era_cutoff,
            temperatures=temps,
            include_percentile=not args.no_percentile,
            B=args.B,
            seed=args.seed,
            baseline_post=baseline_post,
        )
        a3_sum.to_csv(outdir / "A3_judge_summary.csv", index=False)
        a3_week.to_csv(outdir / "A3_judge_week.csv", index=False)
        a3_post.to_csv(outdir / "A3_judge_post.csv", index=False)
        summary_tables.append(a3_sum)

    if args.run_a4:
        a4_sum, a4_week, a4_post = run_a4_leave_one_season_out(
            panel, base_fit, B=args.B, seed=args.seed, baseline_post=baseline_post
        )
        a4_sum.to_csv(outdir / "A4_leave_one_season_summary.csv", index=False)
        a4_week.to_csv(outdir / "A4_leave_one_season_week.csv", index=False)
        a4_post.to_csv(outdir / "A4_leave_one_season_post.csv", index=False)
        summary_tables.append(a4_sum)

    all_summary = pd.concat(summary_tables, ignore_index=True)
    all_summary.to_csv(outdir / "summary_all.csv", index=False)

    config = {
        "seed": args.seed,
        "B": args.B,
        "era_cutoff": args.era_cutoff,
        "base_fit": base_fit["hyperparams"],
        "grid_n": args.grid_n,
        "tau_values": parse_float_list(args.tau_values),
        "kappa_values": parse_float_list(args.kappa_values),
        "ratio_values": parse_float_list(args.ratio_values),
        "temperatures": parse_float_list(args.temperatures),
        "include_percentile": not args.no_percentile,
        "ran": {
            "A1": args.run_a1,
            "A2": args.run_a2,
            "A3": args.run_a3,
            "A4": args.run_a4,
        },
    }
    with open(outdir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
