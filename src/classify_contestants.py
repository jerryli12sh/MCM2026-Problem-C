import argparse
from pathlib import Path
import numpy as np
import pandas as pd

import model


def build_panel():
    df_elim_events, df_roster, df_weekly, df_long_judge, df_clean = model.load_tables()
    elim_long = model.build_elim_long(df_elim_events)
    base = model.build_base(df_roster, elim_long, df_clean)
    judge_percent = model.build_judge_percent(df_weekly, base)
    judge_rank_share = model.build_judge_rank_share(df_long_judge, base)
    panel = model.build_panel(base, judge_percent, judge_rank_share)
    return panel


def compute_p_hat(panel: pd.DataFrame, pooled_fit: dict, *, seed: int, B: int, tau_like: float) -> pd.DataFrame:
    alive = panel[panel["alive"] == True].copy()
    weeks = (
        alive[["season", "week"]]
        .drop_duplicates()
        .sort_values(["season", "week"])
        .values.tolist()
    )

    out_rows = []
    for season, week in weeks:
        post = model.posterior_mean_for_week(
            panel,
            pooled_fit,
            season=season,
            week=week,
            B=B,
            tau_like=tau_like,
            seed=seed,
        )
        if post is None:
            q = model.pooled_q_for_week(panel, pooled_fit, season=season, week=week)
            q = q[["season", "week", "celebrity_name", "q_hat"]].copy()
            q["p_hat"] = q["q_hat"]
            q["has_posterior"] = False
            out_rows.append(q[["season", "week", "celebrity_name", "p_hat", "has_posterior"]])
        else:
            post = post[["season", "week", "celebrity_name", "p_mean", "q_hat", "has_posterior"]].copy()
            post["p_hat"] = post["p_mean"]
            missing = post["p_hat"].isna()
            if missing.any():
                post.loc[missing, "p_hat"] = post.loc[missing, "q_hat"]
                post.loc[missing, "has_posterior"] = False
            out_rows.append(post[["season", "week", "celebrity_name", "p_hat", "has_posterior"]])

    return pd.concat(out_rows, ignore_index=True)


def assign_archetypes(delta_mean: pd.Series) -> pd.Series:
    n = len(delta_mean)
    if n == 0:
        return pd.Series([], dtype=str)

    order = delta_mean.sort_values(kind="mergesort")
    n_tech = int(np.floor(n * 0.25))
    n_pop = int(np.floor(n * 0.25))

    labels = pd.Series(index=delta_mean.index, dtype=object)

    if n_tech > 0:
        tech_idx = order.iloc[:n_tech].index
        labels.loc[tech_idx] = "relative_technical"

    if n_pop > 0:
        pop_idx = order.iloc[-n_pop:].index
        labels.loc[pop_idx] = "relative_popular"

    labels = labels.fillna("balanced")
    return labels


def main():
    parser = argparse.ArgumentParser(description="Classify contestants by fan-vs-judge preference gap.")
    parser.add_argument("--out", default="data/contestant_archetypes.csv", help="Output CSV path")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--B", type=int, default=600, help="Posterior samples per week")
    parser.add_argument("--tau-like", type=float, default=0.15)
    parser.add_argument("--n-steps", type=int, default=600, help="Training steps for pooled model")
    args = parser.parse_args()

    panel = build_panel()

    _, pooled_fit, _ = model.train_pooled_model(panel, seed=args.seed, n_steps=args.n_steps)

    p_hat = compute_p_hat(panel, pooled_fit, seed=args.seed, B=args.B, tau_like=args.tau_like)

    alive = panel[panel["alive"] == True][["season", "week", "celebrity_name", "j_metric"]].copy()
    merged = alive.merge(p_hat, on=["season", "week", "celebrity_name"], how="left")

    merged["delta"] = merged["p_hat"] - merged["j_metric"]

    by_contestant = (
        merged.groupby(["season", "celebrity_name"], as_index=False)
        .agg(delta_mean=("delta", "mean"), n_weeks=("delta", "count"))
    )

    by_contestant["archetype"] = assign_archetypes(by_contestant["delta_mean"])

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    by_contestant.to_csv(out_path, index=False)

    print(f"Wrote {len(by_contestant)} rows to {out_path}")


if __name__ == "__main__":
    main()
