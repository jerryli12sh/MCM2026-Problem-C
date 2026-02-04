import argparse
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

import model


@dataclass
class PosteriorWeekResult:
    season: int
    week: int
    names: np.ndarray
    p_hat: np.ndarray
    ci_lo: np.ndarray
    ci_hi: np.ndarray
    ci_width: np.ndarray
    ci_rel_width: np.ndarray
    ess: float
    ess_ratio: float
    has_posterior: bool
    alive_n: int


def posterior_draws_for_week(panel: pd.DataFrame, pooled_fit: Dict, season: int, week: int,
                             *, kappa: float = None, B: int = 1200, tau_like: float = 0.15,
                             seed: int = 42) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, float, bool]:
    rng = np.random.default_rng(seed)
    g = model.pooled_q_for_week(panel, pooled_fit, season=season, week=week).copy()
    alive = g.copy()
    n = alive.shape[0]
    if n <= 1:
        return None, None, None, 0.0, False

    if alive["elim_this_week_end"].sum() == 1:
        elim_name = alive.loc[alive["elim_this_week_end"], "celebrity_name"].iloc[0]
        names = alive["celebrity_name"].to_numpy()
        elim_pos = int(np.where(names == elim_name)[0][0])
        has_posterior = True
    else:
        elim_pos = None
        has_posterior = False

    kappa = pooled_fit.get("kappa") if kappa is None else kappa
    q = alive["q_hat"].to_numpy()
    alpha = kappa * q
    p_samps = rng.dirichlet(alpha, size=B)

    if elim_pos is not None:
        j = alive["j_metric"].to_numpy()
        cost = j[None, :] + p_samps
        logp = model.softmin_logprob_elim(cost, elim_pos, tau=tau_like)
        w = np.exp(logp - logp.max())
        w = w / w.sum()
    else:
        w = np.ones(B) / B

    ess = 1.0 / np.sum(w ** 2)
    return alive, p_samps, w, ess, has_posterior


def summarize_week(panel: pd.DataFrame, pooled_fit: Dict, season: int, week: int,
                   *, kappa: float = None, B: int = 1200, tau_like: float = 0.15,
                   seed: int = 42, eps: float = 1e-6) -> PosteriorWeekResult:
    alive, p_samps, w, ess, has_posterior = posterior_draws_for_week(
        panel, pooled_fit, season, week, kappa=kappa, B=B, tau_like=tau_like, seed=seed
    )
    if alive is None:
        return None

    names = alive["celebrity_name"].to_numpy()
    p_hat = (w[:, None] * p_samps).sum(axis=0)

    ci_lo = np.array([model.weighted_quantile(p_samps[:, k], 0.05, w) for k in range(p_samps.shape[1])])
    ci_hi = np.array([model.weighted_quantile(p_samps[:, k], 0.95, w) for k in range(p_samps.shape[1])])

    ci_width = ci_hi - ci_lo
    ci_rel_width = ci_width / (p_hat + eps)

    ess_ratio = ess / float(p_samps.shape[0])
    alive_n = alive.shape[0]

    return PosteriorWeekResult(
        season=int(season),
        week=int(week),
        names=names,
        p_hat=p_hat,
        ci_lo=ci_lo,
        ci_hi=ci_hi,
        ci_width=ci_width,
        ci_rel_width=ci_rel_width,
        ess=float(ess),
        ess_ratio=float(ess_ratio),
        has_posterior=bool(has_posterior),
        alive_n=int(alive_n),
    )


def build_exit_week(panel: pd.DataFrame) -> pd.DataFrame:
    base = panel[panel["alive"] == True].copy()
    max_week_by_season = base.groupby("season")["week"].max().to_dict()

    def _exit_week(g: pd.DataFrame) -> int:
        elim_rows = g[g["elim_this_week_end"] == True]
        if len(elim_rows) > 0:
            return int(elim_rows["week"].iloc[0])
        return int(max_week_by_season[int(g["season"].iloc[0])])

    out = (base.groupby(["season", "celebrity_name"])
           .apply(_exit_week)
           .reset_index(name="exit_week"))
    return out


def compute_all_metrics(panel: pd.DataFrame, pooled_fit: Dict,
                        *, kappa: float = None, B: int = 1200, tau_like: float = 0.15,
                        seed: int = 42, eps: float = 1e-6) -> pd.DataFrame:
    alive_panel = panel[panel["alive"] == True].copy()
    weeks = (alive_panel[["season", "week"]]
             .drop_duplicates()
             .sort_values(["season", "week"]))

    rows: List[Dict] = []

    exit_week = build_exit_week(panel)

    for _, row in weeks.iterrows():
        s = int(row["season"])
        wk = int(row["week"])
        wk_seed = seed + s * 1000 + wk

        res = summarize_week(panel, pooled_fit, s, wk, kappa=kappa, B=B, tau_like=tau_like, seed=wk_seed, eps=eps)
        if res is None:
            continue

        for i, name in enumerate(res.names):
            rows.append({
                "season": res.season,
                "week": res.week,
                "celebrity_name": name,
                "p_hat": float(res.p_hat[i]),
                "ci_lo": float(res.ci_lo[i]),
                "ci_hi": float(res.ci_hi[i]),
                "ci_width": float(res.ci_width[i]),
                "ci_rel_width": float(res.ci_rel_width[i]),
                "ess": float(res.ess),
                "ess_ratio": float(res.ess_ratio),
                "has_posterior": bool(res.has_posterior),
                "alive_n": int(res.alive_n),
                "B": int(B),
                "tau_like": float(tau_like),
                "kappa": float(pooled_fit.get("kappa") if kappa is None else kappa),
            })

    out = pd.DataFrame(rows)
    out = out.merge(exit_week, on=["season", "celebrity_name"], how="left")
    return out


def main():
    parser = argparse.ArgumentParser(description="Compute posterior uncertainty metrics for contestant-week fan shares.")
    parser.add_argument("--B", type=int, default=1200)
    parser.add_argument("--tau-like", type=float, default=0.15)
    parser.add_argument("--kappa", type=float, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eps", type=float, default=1e-6)
    parser.add_argument("--output", type=str, default="posterior_uncertainty.csv")
    args = parser.parse_args()

    df_elim_events, df_roster, df_weekly, df_long_judge, df_clean = model.load_tables()
    elim_long = model.build_elim_long(df_elim_events)
    base = model.build_base(df_roster, elim_long, df_clean)
    judge_percent = model.build_judge_percent(df_weekly, base)
    judge_rank_share = model.build_judge_rank_share(df_long_judge, base)
    panel = model.build_panel(base, judge_percent, judge_rank_share)

    _, pooled_fit, _ = model.train_pooled_model(panel)

    metrics = compute_all_metrics(
        panel,
        pooled_fit,
        kappa=args.kappa,
        B=args.B,
        tau_like=args.tau_like,
        seed=args.seed,
        eps=args.eps,
    )
    metrics.to_csv(args.output, index=False)
    print(f"Wrote {args.output} with {len(metrics)} rows")


if __name__ == "__main__":
    main()
