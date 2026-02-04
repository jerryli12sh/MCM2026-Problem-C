import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import model


@dataclass
class SimConfigB:
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
    fan_temp: float
    judge_save_rule: str  # "judge" | "fan" | "improve"


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


def _apply_fan_temp(p: np.ndarray, temp: float) -> np.ndarray:
    t = max(1e-6, float(temp))
    adj = np.power(p, 1.0 / t)
    s = adj.sum()
    if s <= 0:
        return np.ones_like(adj) / len(adj)
    return adj / s


def _choose_elim_from_bottom(
    names: np.ndarray,
    bottom_idx: np.ndarray,
    judge_rank: np.ndarray,
    fan_p: np.ndarray,
    improvement: np.ndarray,
    rule: str,
) -> str:
    if len(bottom_idx) == 1:
        return names[bottom_idx[0]]
    if rule == "fan":
        elim_idx = bottom_idx[np.lexsort((fan_p[bottom_idx],))[0]]
    elif rule == "improve":
        elim_idx = bottom_idx[np.lexsort((fan_p[bottom_idx], improvement[bottom_idx]))[0]]
    else:
        # judge: eliminate worst judge rank
        elim_idx = bottom_idx[np.argmax(judge_rank[bottom_idx])]
    return names[elim_idx]


def _simulate_one_season(
    season: int,
    df_weekly: pd.DataFrame,
    archetypes: pd.DataFrame,
    age_map: Dict[str, float],
    pooled_fit: dict,
    cfg: SimConfigB,
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
        improvement = {}
        for name, v in scores.items():
            improvement[name] = float(v) - float(last_scores.get(name, v))
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
        fan_p = _apply_fan_temp(p_draw, cfg.fan_temp)
        fan_rank = _rank_desc(fan_p)
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
                bottom_k = order[: min(cfg.K, len(names))]
                elim_name = _choose_elim_from_bottom(
                    names,
                    bottom_k,
                    judge_rank,
                    fan_p,
                    np.array([improvement[n] for n in names], dtype=float),
                    cfg.judge_save_rule,
                )
                eliminated = elim_name
            else:
                k = min(cfg.K, len(names))
                nominee_idx = np.argsort(judge_rank)[-k:]
                elim_name = _choose_elim_from_bottom(
                    names,
                    nominee_idx,
                    judge_rank,
                    fan_p,
                    np.array([improvement[n] for n in names], dtype=float),
                    cfg.judge_save_rule,
                )
                eliminated = elim_name

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
                    "fan_p": fan_p[i],
                    "fan_rank": fan_rank[i],
                    "combined_rank": combined_rank[i],
                    "improvement": improvement[name],
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
    cfg: SimConfigB,
    seasons: List[int],
) -> pd.DataFrame:
    age_map = _age_map(df_clean)
    all_rows: List[dict] = []

    for scheme_idx, scheme in enumerate(cfg.schemes):
        cfg_one = SimConfigB(
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
            fan_temp=cfg.fan_temp,
            judge_save_rule=cfg.judge_save_rule,
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


def elimination_name_by_week(df: pd.DataFrame) -> pd.DataFrame:
    g = df[df["eliminated_this_week"]].copy()
    return g[["scheme", "sim", "season", "week", "celebrity_name"]].rename(
        columns={"celebrity_name": "eliminated_name"}
    )


def reversal_rate(df: pd.DataFrame, baseline_scheme: str = "S1") -> pd.DataFrame:
    elim = elimination_name_by_week(df)
    base = elim[elim["scheme"] == baseline_scheme].rename(
        columns={"eliminated_name": "elim_base"}
    )
    merged = elim.merge(
        base[["sim", "season", "week", "elim_base"]],
        on=["sim", "season", "week"],
        how="left",
    )
    merged["reversed"] = merged["eliminated_name"] != merged["elim_base"]
    res = (
        merged.groupby(["scheme", "season", "week"], as_index=False)
        .agg(reversal_rate=("reversed", "mean"))
    )
    return res


def elimination_week_table(df: pd.DataFrame) -> pd.DataFrame:
    weeks = df.groupby(["scheme", "sim", "season"])["week"].max().reset_index()
    elim = df[df["eliminated_this_week"]].copy()
    elim = elim.groupby(["scheme", "sim", "season", "celebrity_name"])["week"].min().reset_index()
    elim = elim.merge(weeks, on=["scheme", "sim", "season"], how="left", suffixes=("", "_max"))
    elim["elim_week"] = elim["week"]
    elim["elim_week"] = elim["elim_week"].fillna(elim["week_max"] + 1)
    return elim[["scheme", "sim", "season", "celebrity_name", "elim_week"]]


def controversy_flip_rate(
    df: pd.DataFrame, contestants: List[str], baseline_scheme: str = "S1"
) -> pd.DataFrame:
    if not contestants:
        return pd.DataFrame()
    elim_week = elimination_week_table(df)
    elim_week = elim_week[elim_week["celebrity_name"].isin(contestants)]
    base = elim_week[elim_week["scheme"] == baseline_scheme].rename(
        columns={"elim_week": "elim_week_base"}
    )
    merged = elim_week.merge(
        base[["sim", "season", "celebrity_name", "elim_week_base"]],
        on=["sim", "season", "celebrity_name"],
        how="left",
    )
    merged["flip"] = merged["elim_week"] != merged["elim_week_base"]
    return (
        merged.groupby(["scheme", "celebrity_name"], as_index=False)
        .agg(flip_rate=("flip", "mean"))
    )


def compute_metrics(df: pd.DataFrame, final_rank_threshold: int = 4) -> pd.DataFrame:
    if df.empty:
        return df
    res = []
    for scheme, g in df.groupby("scheme"):
        non_final = g[g["stage"] != "final"]
        extreme_unfair = (non_final["eliminated_this_week"] & (non_final["judge_rank"] <= final_rank_threshold)).mean()

        # fan-favoring survival: bottom judge quartile + top fan quartile and survived
        def fan_favoring_rate(group: pd.DataFrame) -> float:
            if group.empty:
                return np.nan
            q_j = group["judge_rank"].quantile(0.75)
            q_f = group["fan_p"].quantile(0.75)
            mask = (group["judge_rank"] >= q_j) & (group["fan_p"] >= q_f)
            if mask.sum() == 0:
                return 0.0
            surv = (~group["eliminated_this_week"]) & mask
            return float(surv.mean())

        fan_favoring = non_final.groupby(["season", "week"]).apply(fan_favoring_rate).mean()

        # technical finalist rate
        final_rows = g[g["stage"] == "final"]
        if final_rows.empty:
            technical_finalist = np.nan
        else:
            technical_finalist = (final_rows["judge_rank"] <= final_rank_threshold).mean()

        # entertainment volatility: mean entropy of elimination week
        elim_week = elimination_week_table(g)
        entropies = []
        for (season, name), sub in elim_week.groupby(["season", "celebrity_name"]):
            counts = sub["elim_week"].value_counts().to_numpy(dtype=float)
            probs = counts / counts.sum()
            ent = -np.sum(probs * np.log(probs + 1e-12))
            entropies.append(ent)
        volatility = float(np.mean(entropies)) if entropies else np.nan

        res.append(
            {
                "scheme": scheme,
                "extreme_unfair_rate": float(extreme_unfair),
                "fan_favoring_rate": float(fan_favoring),
                "technical_finalist_rate": float(technical_finalist),
                "entertainment_volatility": float(volatility),
            }
        )
    return pd.DataFrame(res)


def fairness_score(df_metrics: pd.DataFrame) -> pd.Series:
    return (
        df_metrics["technical_finalist_rate"]
        - df_metrics["extreme_unfair_rate"]
        - df_metrics["fan_favoring_rate"]
    )


def build_pooled_fit(era_cutoff: int, seed: int, n_steps: int) -> dict:
    df_elim_events, df_roster, df_weekly_raw, df_long_judge, df_clean_raw = model.load_tables()
    elim_long = model.build_elim_long(df_elim_events)
    base = model.build_base(df_roster, elim_long, df_clean_raw)
    judge_percent = model.build_judge_percent(df_weekly_raw, base)
    judge_rank_share = model.build_judge_rank_share(df_long_judge, base)
    panel = model.build_panel(base, judge_percent, judge_rank_share, era_cutoff=era_cutoff)
    _, pooled_fit, _ = model.train_pooled_model(panel, seed=seed, n_steps=n_steps)
    return pooled_fit


def run_b1(
    df_weekly: pd.DataFrame,
    archetypes: pd.DataFrame,
    df_clean: pd.DataFrame,
    args,
    seasons: List[int],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summaries = []
    reversal_rows = []
    controversy_rows = []

    for cutoff in args.era_cutoffs:
        pooled_fit = build_pooled_fit(cutoff, seed=args.seed, n_steps=args.n_steps)
        cfg = SimConfigB(
            schemes=tuple(args.schemes),
            n_sims=args.n_sims,
            seed=args.seed,
            wJ=args.wJ,
            wF=1.0 - args.wJ,
            K=args.bottom_k,
            m_early_elims=args.m,
            final_n=args.final_n,
            era_cutoff=cutoff,
            kappa=args.kappa,
            fan_temp=args.fan_temp,
            judge_save_rule=args.judge_save_rule,
        )
        results = run_simulation(df_weekly, archetypes, df_clean, pooled_fit, cfg, seasons)

        rev = reversal_rate(results, baseline_scheme=args.baseline_scheme)
        rev["era_cutoff"] = cutoff
        reversal_rows.append(rev)

        metrics = compute_metrics(results, final_rank_threshold=args.tech_rank)
        metrics["era_cutoff"] = cutoff
        metrics["wJ"] = args.wJ
        metrics["fan_temp"] = args.fan_temp
        metrics["bottom_k"] = args.bottom_k
        metrics["judge_save_rule"] = args.judge_save_rule
        metrics["fairness_score"] = fairness_score(metrics)
        summaries.append(metrics)

        if args.controversy:
            cf = controversy_flip_rate(results, args.controversy, baseline_scheme=args.baseline_scheme)
            cf["era_cutoff"] = cutoff
            controversy_rows.append(cf)

    return (
        pd.concat(summaries, ignore_index=True) if summaries else pd.DataFrame(),
        pd.concat(reversal_rows, ignore_index=True) if reversal_rows else pd.DataFrame(),
        pd.concat(controversy_rows, ignore_index=True) if controversy_rows else pd.DataFrame(),
    )


def run_b2(
    df_weekly: pd.DataFrame,
    archetypes: pd.DataFrame,
    df_clean: pd.DataFrame,
    args,
    seasons: List[int],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    pooled_fit = build_pooled_fit(args.era_cutoff, seed=args.seed, n_steps=args.n_steps)
    summaries = []
    map_rows = []

    for wJ in args.wJ_grid:
        for fan_temp in args.fan_temp_grid:
            cfg = SimConfigB(
                schemes=tuple(args.schemes),
                n_sims=args.n_sims,
                seed=args.seed,
                wJ=wJ,
                wF=1.0 - wJ,
                K=args.bottom_k,
                m_early_elims=args.m,
                final_n=args.final_n,
                era_cutoff=args.era_cutoff,
                kappa=args.kappa,
                fan_temp=fan_temp,
                judge_save_rule=args.judge_save_rule,
            )
            results = run_simulation(df_weekly, archetypes, df_clean, pooled_fit, cfg, seasons)
            metrics = compute_metrics(results, final_rank_threshold=args.tech_rank)
            metrics["wJ"] = wJ
            metrics["fan_temp"] = fan_temp
            metrics["fairness_score"] = fairness_score(metrics)
            summaries.append(metrics)

            best = (
                metrics.sort_values("fairness_score", ascending=False)
                .iloc[0]["scheme"]
            )
            map_rows.append({"w_j": wJ, "fan_temp": fan_temp, "winner": best})

    return (
        pd.concat(summaries, ignore_index=True) if summaries else pd.DataFrame(),
        pd.DataFrame(map_rows),
    )


def run_b3(
    df_weekly: pd.DataFrame,
    archetypes: pd.DataFrame,
    df_clean: pd.DataFrame,
    args,
    seasons: List[int],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    pooled_fit = build_pooled_fit(args.era_cutoff, seed=args.seed, n_steps=args.n_steps)
    summaries = []
    controversies = []

    for k in args.bottom_k_grid:
        for rule in args.judge_save_rules:
            cfg = SimConfigB(
                schemes=tuple(args.schemes),
                n_sims=args.n_sims,
                seed=args.seed,
                wJ=args.wJ,
                wF=1.0 - args.wJ,
                K=k,
                m_early_elims=args.m,
                final_n=args.final_n,
                era_cutoff=args.era_cutoff,
                kappa=args.kappa,
                fan_temp=args.fan_temp,
                judge_save_rule=rule,
            )
            results = run_simulation(df_weekly, archetypes, df_clean, pooled_fit, cfg, seasons)
            metrics = compute_metrics(results, final_rank_threshold=args.tech_rank)
            metrics["bottom_k"] = k
            metrics["judge_save_rule"] = rule
            metrics["fairness_score"] = fairness_score(metrics)
            summaries.append(metrics)

            if args.controversy:
                cf = controversy_flip_rate(results, args.controversy, baseline_scheme=args.baseline_scheme)
                cf["bottom_k"] = k
                cf["judge_save_rule"] = rule
                controversies.append(cf)

    return (
        pd.concat(summaries, ignore_index=True) if summaries else pd.DataFrame(),
        pd.concat(controversies, ignore_index=True) if controversies else pd.DataFrame(),
    )


def parse_float_list(text: str) -> List[float]:
    return [float(x) for x in text.split(",") if x.strip()]


def parse_int_list(text: str) -> List[int]:
    return [int(x) for x in text.split(",") if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Sensitivity analysis for rule simulator (B1-B3).")
    parser.add_argument("--outdir", type=str, default="sensitivity_b_outputs")
    parser.add_argument("--n-sims", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--schemes", type=str, default="S1,S2,S3")
    parser.add_argument("--baseline-scheme", type=str, default="S1")
    parser.add_argument("--wJ", type=float, default=0.5)
    parser.add_argument("--kappa", type=float, default=10.0)
    parser.add_argument("--fan-temp", type=float, default=1.0)
    parser.add_argument("--bottom-k", type=int, default=2)
    parser.add_argument("--m", type=int, default=8)
    parser.add_argument("--final-n", type=int, default=3)
    parser.add_argument("--era-cutoff", type=int, default=28)
    parser.add_argument("--n-steps", type=int, default=600)
    parser.add_argument("--tech-rank", type=int, default=4)
    parser.add_argument("--judge-save-rule", type=str, default="judge")
    parser.add_argument("--seasons", type=str, default="all")
    parser.add_argument("--controversy", type=str, default="")

    parser.add_argument("--run-b1", action="store_true")
    parser.add_argument("--run-b2", action="store_true")
    parser.add_argument("--run-b3", action="store_true")

    parser.add_argument("--era-cutoffs", type=str, default="26,27,28,29,30")
    parser.add_argument("--wJ-grid", type=str, default="0.2,0.3,0.4,0.5,0.6,0.7,0.8")
    parser.add_argument("--fan-temp-grid", type=str, default="0.5,1.0,2.0")
    parser.add_argument("--bottom-k-grid", type=str, default="2,3")
    parser.add_argument("--judge-save-rules", type=str, default="judge,fan,improve")

    args = parser.parse_args()

    run_any = args.run_b1 or args.run_b2 or args.run_b3
    if not run_any:
        args.run_b1 = args.run_b2 = args.run_b3 = True

    args.schemes = [s.strip() for s in args.schemes.split(",") if s.strip()]
    args.era_cutoffs = parse_int_list(args.era_cutoffs)
    args.wJ_grid = parse_float_list(args.wJ_grid)
    args.fan_temp_grid = parse_float_list(args.fan_temp_grid)
    args.bottom_k_grid = parse_int_list(args.bottom_k_grid)
    args.judge_save_rules = [s.strip() for s in args.judge_save_rules.split(",") if s.strip()]
    args.controversy = [s.strip() for s in args.controversy.split(",") if s.strip()]

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df_weekly = _prepare_weekly_table(model.load_csv("df_weekly.csv"))
    df_clean = model.load_csv("df_clean.csv")
    archetypes = _load_archetypes(Path("data/contestant_archetypes.csv"))

    if args.seasons == "all":
        seasons = sorted(df_weekly["season"].unique().tolist())
    else:
        seasons = [int(s) for s in args.seasons.split(",") if s.strip()]

    if args.run_b1:
        b1_sum, b1_rev, b1_con = run_b1(df_weekly, archetypes, df_clean, args, seasons)
        b1_sum.to_csv(outdir / "B1_cutoff_summary.csv", index=False)
        b1_rev.to_csv(outdir / "B1_reversal_week.csv", index=False)
        if not b1_con.empty:
            b1_con.to_csv(outdir / "B1_controversy.csv", index=False)

    if args.run_b2:
        b2_sum, b2_map = run_b2(df_weekly, archetypes, df_clean, args, seasons)
        b2_sum.to_csv(outdir / "B2_grid_summary.csv", index=False)
        b2_map.to_csv(outdir / "robustness_map.csv", index=False)

    if args.run_b3:
        b3_sum, b3_con = run_b3(df_weekly, archetypes, df_clean, args, seasons)
        b3_sum.to_csv(outdir / "B3_bottom_summary.csv", index=False)
        if not b3_con.empty:
            b3_con.to_csv(outdir / "B3_controversy.csv", index=False)

    config = {
        "schemes": args.schemes,
        "baseline_scheme": args.baseline_scheme,
        "seed": args.seed,
        "n_sims": args.n_sims,
        "wJ": args.wJ,
        "kappa": args.kappa,
        "fan_temp": args.fan_temp,
        "bottom_k": args.bottom_k,
        "m": args.m,
        "final_n": args.final_n,
        "era_cutoff": args.era_cutoff,
        "tech_rank": args.tech_rank,
        "judge_save_rule": args.judge_save_rule,
        "era_cutoffs": args.era_cutoffs,
        "wJ_grid": args.wJ_grid,
        "fan_temp_grid": args.fan_temp_grid,
        "bottom_k_grid": args.bottom_k_grid,
        "judge_save_rules": args.judge_save_rules,
        "controversy": args.controversy,
        "ran": {"B1": args.run_b1, "B2": args.run_b2, "B3": args.run_b3},
    }
    with open(outdir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
