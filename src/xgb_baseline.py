import ast
from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd

try:
    import xgboost as xgb
except Exception as exc:  # pragma: no cover - runtime import guard
    xgb = None
    _XGB_IMPORT_ERROR = exc


# -------------------- utils --------------------

def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)


def load_csv(name: str) -> pd.DataFrame:
    """Load a CSV by searching common locations."""
    candidates = [
        Path(name),
        Path.cwd() / name,
        Path.cwd() / "data" / name,
        Path.cwd().parent / name,
        Path.cwd().parent / "data" / name,
        Path("/mnt/data") / name,
    ]
    for p in candidates:
        if p.exists():
            return pd.read_csv(p)
    raise FileNotFoundError(
        f"Could not find {name}. Tried: " + ", ".join(str(p) for p in candidates)
    )


def load_tables():
    df_elim_events = load_csv("df_elim_events.csv")
    df_roster = load_csv("df_roster.csv")
    df_weekly = load_csv("df_weekly.csv")
    df_long_judge = load_csv("df_long_judge.csv")
    df_clean = load_csv("df_clean.csv")
    return df_elim_events, df_roster, df_weekly, df_long_judge, df_clean


# -------------------- panel construction --------------------

def build_elim_long(df_elim_events: pd.DataFrame) -> pd.DataFrame:
    tmp = df_elim_events.copy()
    if "Unnamed: 0" in tmp.columns:
        tmp = tmp.drop(columns=["Unnamed: 0"])
    tmp["eliminated_list"] = tmp["eliminated"].apply(ast.literal_eval)
    elim_long = (
        tmp.rename(columns={"elim_at_end_of_week": "week"})
        .explode("eliminated_list")
        .rename(columns={"eliminated_list": "celebrity_name"})
        [["season", "week", "celebrity_name", "is_final_week_end"]]
    )
    elim_long["elim_this_week_end"] = True
    return elim_long


def build_base(
    df_roster: pd.DataFrame,
    elim_long: pd.DataFrame,
    df_clean: pd.DataFrame,
) -> pd.DataFrame:
    base = df_roster.copy()
    for c in ["Unnamed: 0"]:
        if c in base.columns:
            base = base.drop(columns=c)

    base = base.merge(elim_long, on=["season", "week", "celebrity_name"], how="left")
    base["elim_this_week_end"] = base["elim_this_week_end"].fillna(False).astype(bool)

    if "eligible" in base.columns:
        base["alive"] = base["eligible"].astype(bool)
    elif "alive" not in base.columns:
        raise KeyError("df_roster must contain `eligible` or `alive` to define alive set.")

    max_week_by_season = base.loc[base["alive"]].groupby("season")["week"].max()
    base["max_week"] = base["season"].map(max_week_by_season)
    base["is_final_week"] = base["week"].eq(base["max_week"])

    if "celebrity_age_during_season" in base.columns:
        base["age"] = pd.to_numeric(base["celebrity_age_during_season"], errors="coerce")
    elif "celebrity_age" in base.columns:
        base["age"] = pd.to_numeric(base["celebrity_age"], errors="coerce")
    elif "age" in base.columns:
        base["age"] = pd.to_numeric(base["age"], errors="coerce")
    else:
        if df_clean is not None and "celebrity_name" in df_clean.columns:
            if "celebrity_age_during_season" in df_clean.columns:
                age_series = df_clean["celebrity_age_during_season"]
            elif "celebrity_age" in df_clean.columns:
                age_series = df_clean["celebrity_age"]
            else:
                age_series = None

            if age_series is not None:
                age_map = (
                    df_clean.assign(_age=age_series)
                    .dropna(subset=["_age"])
                    .drop_duplicates(subset=["celebrity_name"])
                    .set_index("celebrity_name")["_age"]
                    .to_dict()
                )
                base["age"] = base["celebrity_name"].map(age_map)
            else:
                base["age"] = np.nan
        else:
            base["age"] = np.nan

    return base


def build_judge_percent(df_weekly: pd.DataFrame, base: pd.DataFrame) -> pd.DataFrame:
    w = df_weekly.copy()
    for c in ["Unnamed: 0"]:
        if c in w.columns:
            w = w.drop(columns=c)
    if "judge_percent" in w.columns and w["judge_percent"].notna().any():
        judge_percent = w[["season", "week", "celebrity_name", "judge_percent"]].copy()
    else:
        if "judge_total" not in w.columns and "total_judge_score" not in w.columns:
            raise KeyError("df_weekly must contain judge_percent or judge_total/total_judge_score.")
        score_col = "total_judge_score" if "total_judge_score" in w.columns else "judge_total"
        w2 = w.merge(
            base[["season", "week", "celebrity_name", "alive"]],
            on=["season", "week", "celebrity_name"],
            how="left",
        )
        w2 = w2[w2["alive"] == True].copy()
        denom = w2.groupby(["season", "week"])[score_col].transform("sum")
        w2["judge_percent"] = w2[score_col] / denom
        judge_percent = w2[["season", "week", "celebrity_name", "judge_percent"]].copy()
    return judge_percent


def build_judge_rank_share(df_long_judge: pd.DataFrame, base: pd.DataFrame) -> pd.DataFrame:
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
        on=["season", "week", "celebrity_name"],
        how="left",
    )
    rank_sum = rank_sum[rank_sum["alive"] == True].copy()

    rank_sum["rank_score"] = -rank_sum["rank_sum"].astype(float)

    def softmax_group(x):
        z = x - np.max(x)
        e = np.exp(z)
        return e / e.sum()

    rank_sum["judge_rank_share"] = rank_sum.groupby(["season", "week"])["rank_score"].transform(
        softmax_group
    )
    judge_rank_share = rank_sum[["season", "week", "celebrity_name", "judge_rank_share"]].copy()
    return judge_rank_share


def build_panel(
    base: pd.DataFrame,
    judge_percent: pd.DataFrame,
    judge_rank_share: pd.DataFrame,
    *,
    era_cutoff: int = 28,
) -> pd.DataFrame:
    panel = base.copy()
    panel["era"] = np.where(panel["season"] >= era_cutoff, "percent", "rank")
    panel = panel.merge(judge_percent, on=["season", "week", "celebrity_name"], how="left")
    panel = panel.merge(judge_rank_share, on=["season", "week", "celebrity_name"], how="left")
    panel["j_metric"] = np.where(
        panel["era"] == "percent", panel["judge_percent"], panel["judge_rank_share"]
    )
    return panel


def build_train_weeks(panel: pd.DataFrame):
    alive_rows = panel[panel["alive"] == True].copy()
    elim_cnt = (
        alive_rows.groupby(["season", "week"])["elim_this_week_end"]
        .sum()
        .rename("elim_cnt")
        .reset_index()
    )
    alive_n = (
        alive_rows.groupby(["season", "week"]).size().rename("alive_n").reset_index()
    )
    train_weeks = elim_cnt.merge(alive_n, on=["season", "week"], how="left")
    train_weeks = train_weeks.merge(
        alive_rows.groupby("season")["week"].max().rename("max_week").reset_index(),
        on="season",
        how="left",
    )
    train_weeks = train_weeks[
        (train_weeks["elim_cnt"] == 1) & (train_weeks["week"] < train_weeks["max_week"])
    ].copy()

    train_rows = alive_rows.merge(
        train_weeks[["season", "week"]], on=["season", "week"], how="inner"
    ).copy()
    elim_lookup = (
        train_rows[train_rows["elim_this_week_end"]]
        .groupby(["season", "week"])["celebrity_name"]
        .apply(lambda x: x.iloc[0])
        .to_dict()
    )
    return train_weeks, train_rows, elim_lookup


# -------------------- features --------------------

def _build_features(panel: pd.DataFrame, train_rows: pd.DataFrame):
    df_feat = train_rows.copy()
    df_feat["j_metric"] = pd.to_numeric(df_feat["j_metric"], errors="coerce")
    if df_feat["j_metric"].isna().any():
        raise ValueError("j_metric has NaN in training rows. Fix panel construction first.")

    jm_mean = df_feat["j_metric"].mean()
    jm_std = df_feat["j_metric"].std(ddof=0) + 1e-12
    df_feat["j_metric_z"] = (df_feat["j_metric"] - jm_mean) / jm_std

    use_age = df_feat["age"].notna().any()
    if use_age:
        df_feat["age"] = pd.to_numeric(df_feat["age"], errors="coerce")
        age_mean = df_feat["age"].mean()
        age_std = df_feat["age"].std(ddof=0) + 1e-12
        df_feat["age_z"] = (df_feat["age"] - age_mean) / age_std
    else:
        df_feat["age_z"] = 0.0
        age_mean = None
        age_std = None

    df_feat["era_is_percent"] = (df_feat["era"].astype(str) == "percent").astype(float)

    X_cols = ["j_metric_z", "age_z", "era_is_percent"]

    meta = {
        "jm_mean": jm_mean,
        "jm_std": jm_std,
        "use_age": use_age,
        "age_mean": float(age_mean) if use_age else None,
        "age_std": float(age_std) if use_age else None,
        "X_cols": X_cols,
    }
    return df_feat, meta


def build_features_for_rows(df_rows: pd.DataFrame, pooled_fit: dict) -> pd.DataFrame:
    out = df_rows.copy()
    out["j_metric"] = pd.to_numeric(out["j_metric"], errors="coerce")
    out["j_metric_z"] = (out["j_metric"] - pooled_fit["jm_mean"]) / (
        pooled_fit["jm_std"] + 1e-12
    )
    out["era_is_percent"] = (out["era"].astype(str) == "percent").astype(float)

    if pooled_fit["use_age"] and "age" in out.columns and out["age"].notna().any():
        out["age"] = pd.to_numeric(out["age"], errors="coerce")
        out["age_z"] = (out["age"] - pooled_fit["age_mean"]) / (
            pooled_fit["age_std"] + 1e-12
        )
    else:
        out["age_z"] = 0.0

    return out


# -------------------- XGBoost baseline --------------------

def train_pooled_model(
    panel: pd.DataFrame,
    *,
    seed: int = 42,
    kappa: float = 10.0,
    n_estimators: int = 300,
    max_depth: int = 4,
    learning_rate: float = 0.05,
    subsample: float = 0.9,
    colsample_bytree: float = 0.9,
    reg_lambda: float = 1.0,
    reg_alpha: float = 0.0,
    min_child_weight: float = 1.0,
):
    if xgb is None:  # pragma: no cover - runtime import guard
        raise ImportError(
            "xgboost is required for xgb_baseline.py. "
            f"Original import error: {_XGB_IMPORT_ERROR}"
        )

    set_seed(seed)
    train_weeks, train_rows, _ = build_train_weeks(panel)
    df_feat, meta = _build_features(panel, train_rows)

    X = df_feat[meta["X_cols"]].to_numpy(dtype=np.float32)
    # Predict "not eliminated" as positive to align with model.py's popularity q_hat.
    y = (1 - df_feat["elim_this_week_end"].astype(int)).to_numpy()

    n_pos = int(y.sum())
    n_neg = int((1 - y).sum())
    scale_pos_weight = float(n_neg / max(1, n_pos))

    model = xgb.XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        reg_lambda=reg_lambda,
        reg_alpha=reg_alpha,
        min_child_weight=min_child_weight,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=seed,
        scale_pos_weight=scale_pos_weight,
    )
    model.fit(X, y)

    pooled_fit = {
        "model_type": "xgboost",
        "model": model,
        "X_cols": meta["X_cols"],
        "jm_mean": meta["jm_mean"],
        "jm_std": meta["jm_std"],
        "use_age": meta["use_age"],
        "age_mean": meta["age_mean"],
        "age_std": meta["age_std"],
        "seed": seed,
        "kappa": kappa,
        "hyperparams": {
            "kappa": kappa,
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "reg_lambda": reg_lambda,
            "reg_alpha": reg_alpha,
            "min_child_weight": min_child_weight,
            "scale_pos_weight": scale_pos_weight,
        },
    }

    return model, pooled_fit, train_weeks


def pooled_q_for_week(
    panel: pd.DataFrame, pooled_fit: dict, season: int, week: int
) -> pd.DataFrame:
    if xgb is None:  # pragma: no cover - runtime import guard
        raise ImportError(
            "xgboost is required for xgb_baseline.py. "
            f"Original import error: {_XGB_IMPORT_ERROR}"
        )

    g = panel[
        (panel["season"] == season)
        & (panel["week"] == week)
        & (panel["alive"] == True)
    ].copy()
    g = build_features_for_rows(g, pooled_fit)

    X = g[pooled_fit["X_cols"]].to_numpy(dtype=np.float32)
    proba = pooled_fit["model"].predict_proba(X)[:, 1]
    proba = np.clip(proba, 1e-8, 1 - 1e-8)
    logit_hat = np.log(proba / (1 - proba))

    z = logit_hat - logit_hat.max()
    q = np.exp(z)
    q = q / q.sum()

    g["q_hat"] = q
    g["logit_hat"] = logit_hat
    return g[
        [
            "season",
            "week",
            "celebrity_name",
            "era",
            "j_metric",
            "q_hat",
            "logit_hat",
            "elim_this_week_end",
            "alive",
        ]
    ]


def weighted_quantile(values, quantiles, sample_weight):
    values = np.asarray(values)
    quantiles = np.asarray(quantiles)
    w = np.asarray(sample_weight)
    sorter = np.argsort(values)
    v = values[sorter]
    w = w[sorter]
    cdf = np.cumsum(w)
    cdf = cdf / cdf[-1]
    return np.interp(quantiles, cdf, v)


def softmin_logprob_elim(cost, elim_pos, tau=0.15):
    z = -cost / tau
    z = z - z.max(axis=1, keepdims=True)
    log_denom = np.log(np.exp(z).sum(axis=1))
    return z[:, elim_pos] - log_denom


def posterior_mean_for_week(
    panel: pd.DataFrame,
    pooled_fit: dict,
    season: int,
    week: int,
    *,
    kappa: Optional[float] = None,
    B: int = 1200,
    tau_like: float = 0.15,
    seed: int = 42,
):
    rng = np.random.default_rng(seed)
    g = pooled_q_for_week(panel, pooled_fit, season=season, week=week).copy()
    alive = g.copy()
    n = alive.shape[0]
    if n <= 1:
        return None

    if alive["elim_this_week_end"].sum() == 1:
        elim_name = alive.loc[alive["elim_this_week_end"], "celebrity_name"].iloc[0]
        names = alive["celebrity_name"].to_numpy()
        elim_pos = int(np.where(names == elim_name)[0][0])
    else:
        elim_pos = None

    kappa = pooled_fit.get("kappa") if kappa is None else kappa
    q = alive["q_hat"].to_numpy()
    alpha = kappa * q
    p_samps = rng.dirichlet(alpha, size=B)

    if elim_pos is not None:
        j = alive["j_metric"].to_numpy()
        cost = j[None, :] + p_samps
        logp = softmin_logprob_elim(cost, elim_pos, tau=tau_like)
        w = np.exp(logp - logp.max())
        w = w / w.sum()
    else:
        w = np.ones(B) / B

    p_mean = (w[:, None] * p_samps).sum(axis=0)

    out = alive[
        ["season", "week", "celebrity_name", "era", "j_metric", "q_hat", "elim_this_week_end"]
    ].copy()
    out["p_mean"] = p_mean
    out["has_posterior"] = bool(elim_pos is not None)
    return out
