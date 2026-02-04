import argparse
import ast
from typing import Dict, List

import numpy as np
import pandas as pd

import model
import posterior_uncertainty


def _softmin_prob(cost: np.ndarray, tau: float) -> np.ndarray:
    z = -cost / tau
    z = z - np.max(z)
    e = np.exp(z)
    return e / e.sum()


def build_event_tables(panel: pd.DataFrame, pooled_fit: Dict, df_elim_events: pd.DataFrame,
                       *, tau_like: float = 0.15, B: int = 1200, seed: int = 42) -> Dict[str, pd.DataFrame]:
    tmp = df_elim_events.copy()
    if "Unnamed: 0" in tmp.columns:
        tmp = tmp.drop(columns=["Unnamed: 0"])
    if "elim_at_end_of_week" in tmp.columns:
        tmp = tmp.rename(columns={"elim_at_end_of_week": "week"})

    tmp["eliminated_list"] = tmp["eliminated"].apply(ast.literal_eval)

    event_rows: List[Dict] = []
    long_rows: List[Dict] = []

    for _, row in tmp.iterrows():
        season = int(row["season"])
        week = int(row["week"])
        elim_list = [str(x) for x in row["eliminated_list"]]
        if len(elim_list) == 0:
            continue

        g = panel[(panel["season"] == season) & (panel["week"] == week) & (panel["alive"] == True)].copy()
        if g.empty:
            continue

        wk_seed = seed + season * 1000 + week
        alive, p_samps, w, ess, has_posterior = posterior_uncertainty.posterior_draws_for_week(
            panel, pooled_fit, season, week, B=B, tau_like=tau_like, seed=wk_seed
        )
        if alive is None:
            continue
        g = alive.copy()
        p_mean = (w[:, None] * p_samps).sum(axis=0)

        j = g["j_metric"].to_numpy(dtype=np.float32)
        c_hat = j + p_mean
        pi_hat = _softmin_prob(c_hat, tau_like)

        g = g.copy()
        g["p_mean"] = p_mean
        g["c_hat"] = c_hat
        g["pi_hat"] = pi_hat

        g = g.sort_values("celebrity_name").reset_index(drop=True)
        alive_list = g["celebrity_name"].astype(str).tolist()
        c_list = g["c_hat"].to_numpy().tolist()
        pi_list = g["pi_hat"].to_numpy().tolist()
        p_list = g["p_mean"].to_numpy().tolist()

        event_rows.append({
            "season": season,
            "week": week,
            "alive_list": "|".join(alive_list),
            "elim_obs_list": "|".join(elim_list),
            "m_elim": len(elim_list),
            "alive_n": int(len(alive_list)),
            "c_hat_list": "|".join(f"{x:.10f}" for x in c_list),
            "pi_hat_list": "|".join(f"{x:.10f}" for x in pi_list),
            "p_mean_list": "|".join(f"{x:.10f}" for x in p_list),
            "has_posterior": bool(has_posterior),
            "ess": float(ess),
            "B": int(B),
        })

        c_rank = pd.Series(g["c_hat"].to_numpy()).rank(method="average", ascending=True).to_numpy()
        g["risk_rank"] = c_rank

        for i, r in g.iterrows():
            long_rows.append({
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
            })

    event_table = pd.DataFrame(event_rows)
    event_long = pd.DataFrame(long_rows)
    return {"event_table": event_table, "event_long": event_long}


def main():
    parser = argparse.ArgumentParser(description="Build evaluation event tables from model outputs.")
    parser.add_argument("--tau-like", type=float, default=0.15)
    parser.add_argument("--B", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--event-out", type=str, default="event_table.csv")
    parser.add_argument("--long-out", type=str, default="event_long.csv")
    args = parser.parse_args()

    df_elim_events, df_roster, df_weekly, df_long_judge, df_clean = model.load_tables()
    elim_long = model.build_elim_long(df_elim_events)
    base = model.build_base(df_roster, elim_long, df_clean)
    judge_percent = model.build_judge_percent(df_weekly, base)
    judge_rank_share = model.build_judge_rank_share(df_long_judge, base)
    panel = model.build_panel(base, judge_percent, judge_rank_share)

    _, pooled_fit, _ = model.train_pooled_model(panel)

    tables = build_event_tables(
        panel, pooled_fit, df_elim_events, tau_like=args.tau_like, B=args.B, seed=args.seed
    )
    tables["event_table"].to_csv(args.event_out, index=False)
    tables["event_long"].to_csv(args.long_out, index=False)
    print(f"Wrote {args.event_out} and {args.long_out}")


if __name__ == "__main__":
    main()
