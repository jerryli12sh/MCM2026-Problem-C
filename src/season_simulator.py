import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

import model


@dataclass
class SimConfig:
    schemes: Tuple[str, ...]
    n_sims: int
    seed: int
    wJ: float
    wF: float
    K: int
    m_early_elims: int
    final_n: int
    era_cutoff: int
    kappa: float


def _rank_desc(x: np.ndarray) -> np.ndarray:
    return pd.Series(x).rank(ascending=False, method="average").to_numpy()


def _softmax(x: np.ndarray) -> np.ndarray:
    z = x - np.max(x)
    e = np.exp(z)
    return e / e.sum()


def _load_archetypes(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "archetype" not in df.columns:
        raise KeyError("contestant_archetypes.csv must contain `archetype`.")
    df = df[["season", "celebrity_name", "archetype"]].copy()
    df["season"] = df["season"].astype(int)
    df["celebrity_name"] = df["celebrity_name"].astype(str)
    df["archetype"] = df["archetype"].astype(str)
    return df


def _age_map(df_clean: pd.DataFrame) -> Dict[str, float]:
    if "celebrity_age_during_season" in df_clean.columns:
        age_col = "celebrity_age_during_season"
    elif "celebrity_age" in df_clean.columns:
        age_col = "celebrity_age"
    else:
        return {}
    tmp = df_clean[["celebrity_name", age_col]].dropna()
    tmp = tmp.drop_duplicates(subset=["celebrity_name"])
    return tmp.set_index("celebrity_name")[age_col].astype(float).to_dict()


def _compute_q_hat(df_rows: pd.DataFrame, pooled_fit: dict) -> np.ndarray:
    feat = model.build_features_for_rows(df_rows, pooled_fit)
    X = feat[pooled_fit["X_cols"]].to_numpy(dtype=np.float32)
    beta = pooled_fit["beta_hat"].astype(np.float32)
    logits = pooled_fit["bias_hat"] + X @ beta

    u = pooled_fit.get("u_hat")
    if u is not None:
        cs_idx = feat["cs_idx"].to_numpy()
        mask = cs_idx >= 0
        add_u = np.zeros(len(feat), dtype=np.float32)
        add_u[mask] = u[cs_idx[mask]]
        logits = logits + add_u

    z = logits - logits.max()
    q = np.exp(z)
    q = q / q.sum()
    return q


def _prepare_weekly_table(df_weekly: pd.DataFrame) -> pd.DataFrame:
    df = df_weekly.copy()
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
    df["season"] = df["season"].astype(int)
    df["week"] = df["week"].astype(int)
    df["celebrity_name"] = df["celebrity_name"].astype(str)
    return df


def _get_week_scores(
    df_weekly: pd.DataFrame,
    season: int,
    week: int,
    active_names: List[str],
    last_scores: Dict[str, float],
    season_mean: float,
) -> Dict[str, float]:
    g = df_weekly[(df_weekly["season"] == season) & (df_weekly["week"] == week)]
    g = g[g["celebrity_name"].isin(active_names)]
    score_map = {}
    if "total_judge_score" in g.columns:
        score_map = g.set_index("celebrity_name")["total_judge_score"].to_dict()
    elif "judge_total" in g.columns:
        score_map = g.set_index("celebrity_name")["judge_total"].to_dict()
    else:
        raise KeyError("df_weekly must contain total_judge_score or judge_total.")

    scores = {}
    for name in active_names:
        v = score_map.get(name)
        if v is not None and not pd.isna(v):
            scores[name] = float(v)
        elif name in last_scores:
            scores[name] = float(last_scores[name])
        else:
            scores[name] = float(season_mean)
    return scores


def _stage_label(elim_so_far: int, total: int, m_early: int, final_n: int) -> str:
    if total <= final_n:
        return "final"
    if elim_so_far < m_early:
        return "early"
    return "late"


def _simulate_one_season(
    season: int,
    df_weekly: pd.DataFrame,
    archetypes: pd.DataFrame,
    age_map: Dict[str, float],
    pooled_fit: dict,
    cfg: SimConfig,
    rng: np.random.Generator,
) -> List[dict]:
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

    rows: List[dict] = []
    elim_so_far = 0
    last_scores: Dict[str, float] = {}

    for week in weeks:
        if len(active) <= 1:
            break

        stage = _stage_label(elim_so_far, len(active), cfg.m_early_elims, cfg.final_n)
        scores = _get_week_scores(df_weekly, season, week, active, last_scores, season_mean)
        for name, v in scores.items():
            last_scores[name] = v

        names = np.array(active)
        judge_score = np.array([scores[n] for n in names], dtype=float)
        if judge_score.sum() == 0:
            judge_pct = np.ones_like(judge_score) / len(judge_score)
        else:
            judge_pct = judge_score / judge_score.sum()
        judge_rank = _rank_desc(judge_score)

        era = "percent" if season >= cfg.era_cutoff else "rank"
        if era == "percent":
            j_metric = judge_pct
        else:
            j_metric = _softmax(-judge_rank)

        df_rows = pd.DataFrame(
            {
                "season": season,
                "week": week,
                "celebrity_name": names,
                "era": era,
                "j_metric": j_metric,
                "age": [age_map.get(n, np.nan) for n in names],
            }
        )
        q_hat = _compute_q_hat(df_rows, pooled_fit)
        alpha = cfg.kappa * q_hat
        p_draw = rng.dirichlet(alpha)
        fan_rank = _rank_desc(p_draw)
        combined_rank = cfg.wJ * judge_rank + cfg.wF * fan_rank

        eliminated = None
        if stage != "final":
            scheme = cfg.schemes[0]
            use_baseline = (
                scheme == "S1"
                or (scheme == "S2" and stage == "early")
            )
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
    pooled_fit: dict,
    cfg: SimConfig,
    seasons: List[int],
) -> pd.DataFrame:
    age_map = _age_map(df_clean)
    all_rows: List[dict] = []

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
                rows = _simulate_one_season(
                    season, df_weekly, archetypes, age_map, pooled_fit, cfg_one, rng
                )
                for r in rows:
                    r["scheme"] = scheme
                    r["sim"] = b
                all_rows.extend(rows)

    return pd.DataFrame(all_rows)


def summarize_results(df: pd.DataFrame) -> pd.DataFrame:
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


def main():
    parser = argparse.ArgumentParser(description="Season simulator for DWTS rule variants.")
    parser.add_argument("--schemes", default="S1,S2,S3")
    parser.add_argument("--n-sims", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wJ", type=float, default=0.5)
    parser.add_argument("--wF", type=float, default=0.5)
    parser.add_argument("--K", type=int, default=3)
    parser.add_argument("--m", type=int, default=8, help="Early stage cutoff (elims before late stage).")
    parser.add_argument("--final-n", type=int, default=3)
    parser.add_argument("--era-cutoff", type=int, default=28)
    parser.add_argument("--kappa", type=float, default=10.0)
    parser.add_argument("--n-steps", type=int, default=600)
    parser.add_argument("--out-dir", default="sim_results")
    parser.add_argument("--seasons", default="all")
    args = parser.parse_args()

    schemes = tuple([s.strip() for s in args.schemes.split(",") if s.strip()])
    cfg = SimConfig(
        schemes=schemes,
        n_sims=args.n_sims,
        seed=args.seed,
        wJ=args.wJ,
        wF=args.wF,
        K=args.K,
        m_early_elims=args.m,
        final_n=args.final_n,
        era_cutoff=args.era_cutoff,
        kappa=args.kappa,
    )

    df_weekly = _prepare_weekly_table(model.load_csv("df_weekly.csv"))
    df_clean = model.load_csv("df_clean.csv")
    archetypes = _load_archetypes(Path("data/contestant_archetypes.csv"))

    # Train pooled model for audience distribution
    df_elim_events, df_roster, df_weekly_raw, df_long_judge, df_clean_raw = model.load_tables()
    elim_long = model.build_elim_long(df_elim_events)
    base = model.build_base(df_roster, elim_long, df_clean_raw)
    judge_percent = model.build_judge_percent(df_weekly_raw, base)
    judge_rank_share = model.build_judge_rank_share(df_long_judge, base)
    panel = model.build_panel(base, judge_percent, judge_rank_share, era_cutoff=args.era_cutoff)
    _, pooled_fit, _ = model.train_pooled_model(panel, seed=args.seed, n_steps=args.n_steps)

    if args.seasons == "all":
        seasons = sorted(df_weekly["season"].unique().tolist())
    else:
        seasons = [int(s) for s in args.seasons.split(",") if s.strip()]

    results = run_simulation(df_weekly, archetypes, df_clean, pooled_fit, cfg, seasons)
    summary = summarize_results(results)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(out_dir / "sim_detail.csv", index=False)
    summary.to_csv(out_dir / "sim_summary.csv", index=False)

    print(f"Wrote {len(results)} rows to {out_dir / 'sim_detail.csv'}")
    print(f"Wrote {len(summary)} rows to {out_dir / 'sim_summary.csv'}")


if __name__ == "__main__":
    main()
