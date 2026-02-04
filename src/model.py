
import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
import ast
from pathlib import Path

# -------------------- utils --------------------

def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


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


def build_base(df_roster: pd.DataFrame, elim_long: pd.DataFrame, df_clean: pd.DataFrame) -> pd.DataFrame:
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
        # fallback: use df_clean if available
        if df_clean is not None and "celebrity_name" in df_clean.columns:
            if "celebrity_age_during_season" in df_clean.columns:
                age_series = df_clean["celebrity_age_during_season"]
            elif "celebrity_age" in df_clean.columns:
                age_series = df_clean["celebrity_age"]
            else:
                age_series = None

            if age_series is not None:
                age_map = (df_clean.assign(_age=age_series)
                           .dropna(subset=["_age"])
                           .drop_duplicates(subset=["celebrity_name"])
                           .set_index("celebrity_name")["_age"].to_dict())
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
        w2 = w.merge(base[["season", "week", "celebrity_name", "alive"]],
                     on=["season", "week", "celebrity_name"], how="left")
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
        on=["season", "week", "celebrity_name"], how="left"
    )
    rank_sum = rank_sum[rank_sum["alive"] == True].copy()

    rank_sum["rank_score"] = -rank_sum["rank_sum"].astype(float)

    def softmax_group(x):
        z = x - np.max(x)
        e = np.exp(z)
        return e / e.sum()

    rank_sum["judge_rank_share"] = rank_sum.groupby(["season", "week"])["rank_score"].transform(softmax_group)
    judge_rank_share = rank_sum[["season", "week", "celebrity_name", "judge_rank_share"]].copy()
    return judge_rank_share


def build_panel(base: pd.DataFrame, judge_percent: pd.DataFrame, judge_rank_share: pd.DataFrame,
                *, era_cutoff: int = 28) -> pd.DataFrame:
    panel = base.copy()
    panel["era"] = np.where(panel["season"] >= era_cutoff, "percent", "rank")
    panel = panel.merge(judge_percent, on=["season", "week", "celebrity_name"], how="left")
    panel = panel.merge(judge_rank_share, on=["season", "week", "celebrity_name"], how="left")
    panel["j_metric"] = np.where(panel["era"] == "percent",
                                 panel["judge_percent"], panel["judge_rank_share"])
    return panel


def build_train_weeks(panel: pd.DataFrame):
    alive_rows = panel[panel["alive"] == True].copy()
    elim_cnt = (
        alive_rows.groupby(["season", "week"])["elim_this_week_end"]
        .sum().rename("elim_cnt").reset_index()
    )
    alive_n = (
        alive_rows.groupby(["season", "week"]).size()
        .rename("alive_n").reset_index()
    )
    train_weeks = elim_cnt.merge(alive_n, on=["season", "week"], how="left")
    train_weeks = train_weeks.merge(
        alive_rows.groupby("season")["week"].max().rename("max_week").reset_index(),
        on="season", how="left"
    )
    train_weeks = train_weeks[
        (train_weeks["elim_cnt"] == 1) & (train_weeks["week"] < train_weeks["max_week"])
    ].copy()

    train_rows = alive_rows.merge(train_weeks[["season", "week"]], on=["season", "week"], how="inner").copy()
    elim_lookup = (
        train_rows[train_rows["elim_this_week_end"]]
        .groupby(["season", "week"])["celebrity_name"]
        .apply(lambda x: x.iloc[0])
        .to_dict()
    )
    return train_weeks, train_rows, elim_lookup


# -------------------- pooled model --------------------

class PooledElimModel(nn.Module):
    def __init__(self, n_cs: int, p: int):
        super().__init__()
        self.beta = nn.Parameter(torch.zeros(p))
        self.bias = nn.Parameter(torch.tensor(0.0))
        self.u = nn.Embedding(n_cs, 1)
        nn.init.zeros_(self.u.weight)

    def forward_eta(self, X, cs_idx):
        return self.bias + X @ self.beta + self.u(cs_idx).squeeze(-1)


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

    all_alive = panel[panel["alive"] == True].copy()
    all_alive["_cs_key"] = list(zip(all_alive["season"].astype(int), all_alive["celebrity_name"].astype(str)))
    cs_levels = sorted(all_alive["_cs_key"].unique().tolist())
    cs2idx = {k: i for i, k in enumerate(cs_levels)}
    n_cs = len(cs_levels)
    df_feat["_cs_key"] = list(zip(df_feat["season"].astype(int), df_feat["celebrity_name"].astype(str)))
    df_feat["cs_idx"] = df_feat["_cs_key"].map(cs2idx).astype(int)

    X_cols = ["j_metric_z", "age_z", "era_is_percent"]

    meta = {
        "jm_mean": jm_mean,
        "jm_std": jm_std,
        "use_age": use_age,
        "age_mean": float(age_mean) if use_age else None,
        "age_std": float(age_std) if use_age else None,
        "cs2idx": cs2idx,
        "n_cs": n_cs,
        "X_cols": X_cols,
    }
    return df_feat, meta


def _pack_week_tensors(df_feat: pd.DataFrame, elim_lookup: dict, X_cols):
    group_keys = df_feat[["season", "week"]].drop_duplicates().sort_values(["season", "week"]).values.tolist()
    X_list = []
    J_list = []
    cs_list = []
    elim_pos_list = []
    st_list = []

    for s, wk in group_keys:
        g = df_feat[(df_feat["season"] == s) & (df_feat["week"] == wk)].copy()
        g = g.sort_values("celebrity_name")
        X = g[X_cols].to_numpy(dtype=np.float32)
        cs_idx = g["cs_idx"].to_numpy(dtype=np.int64)
        J = g["j_metric"].to_numpy(dtype=np.float32)
        elim_name = elim_lookup[(s, wk)]
        elim_pos = int(np.where(g["celebrity_name"].to_numpy() == elim_name)[0][0])
        X_list.append(torch.tensor(X))
        J_list.append(torch.tensor(J))
        cs_list.append(torch.tensor(cs_idx))
        elim_pos_list.append(elim_pos)
        st_list.append((int(s), int(wk)))

    return X_list, J_list, cs_list, elim_pos_list, st_list


def train_pooled_model(panel: pd.DataFrame, *, seed: int = 42,
                       tau: float = 0.05, l2_beta: float = 0.05, l2_u: float = 0.05,
                       kappa: float = 10.0, lr: float = 0.020, n_steps: int = 600,
                       batch_size: int = 32):
    set_seed(seed)
    train_weeks, train_rows, elim_lookup = build_train_weeks(panel)
    df_feat, meta = _build_features(panel, train_rows)
    X_list, J_list, cs_list, elim_pos_list, st_list = _pack_week_tensors(df_feat, elim_lookup, meta["X_cols"])

    p = len(meta["X_cols"])

    def train_one_model():
        model = PooledElimModel(n_cs=meta["n_cs"], p=p)
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        n_weeks = len(X_list)
        idx_all = np.arange(n_weeks)

        def pooled_softmin_nll_minibatch(batch_ids):
            nll = 0.0
            for k in batch_ids:
                X = X_list[k]
                J = J_list[k]
                cs_idx = cs_list[k]
                elim_pos = elim_pos_list[k]
                eta = model.forward_eta(X, cs_idx)
                q = torch.softmax(eta, dim=0)
                C = J + q
                logp = F.log_softmax(-C / tau, dim=0)
                nll = nll - logp[elim_pos]
            nll = nll / max(1, len(batch_ids))
            reg = l2_beta * torch.mean(model.beta ** 2) + l2_u * torch.mean(model.u.weight.squeeze(-1) ** 2)
            return nll + reg

        for _ in range(1, n_steps + 1):
            batch_ids = np.random.choice(idx_all, size=min(batch_size, n_weeks), replace=False)
            opt.zero_grad()
            loss = pooled_softmin_nll_minibatch(batch_ids)
            loss.backward()
            opt.step()
        return model

    model = train_one_model()
    beta_hat = model.beta.detach().cpu().numpy().copy()
    bias_hat = float(model.bias.detach().cpu().numpy())
    u_hat = model.u.weight.detach().cpu().numpy().squeeze(-1)

    pooled_fit = {
        "beta_hat": beta_hat,
        "bias_hat": bias_hat,
        "u_hat": u_hat,
        "X_cols": meta["X_cols"],
        "tau": tau,
        "jm_mean": meta["jm_mean"],
        "jm_std": meta["jm_std"],
        "use_age": meta["use_age"],
        "age_mean": meta["age_mean"],
        "age_std": meta["age_std"],
        "cs2idx": meta["cs2idx"],
        "kappa": kappa,
        "seed": seed,
        "hyperparams": {
            "tau": tau,
            "l2_beta": l2_beta,
            "l2_u": l2_u,
            "kappa": kappa,
            "lr": lr,
            "n_steps": n_steps,
            "batch_size": batch_size,
        },
    }

    return model, pooled_fit, train_weeks


# -------------------- inference helpers --------------------

def build_features_for_rows(df_rows: pd.DataFrame, pooled_fit: dict) -> pd.DataFrame:
    out = df_rows.copy()
    out["j_metric"] = pd.to_numeric(out["j_metric"], errors="coerce")
    out["j_metric_z"] = (out["j_metric"] - pooled_fit["jm_mean"]) / (pooled_fit["jm_std"] + 1e-12)
    out["era_is_percent"] = (out["era"].astype(str) == "percent").astype(float)

    if pooled_fit["use_age"] and "age" in out.columns and out["age"].notna().any():
        out["age"] = pd.to_numeric(out["age"], errors="coerce")
        out["age_z"] = (out["age"] - pooled_fit["age_mean"]) / (pooled_fit["age_std"] + 1e-12)
    else:
        out["age_z"] = 0.0

    out["_cs_key"] = list(zip(out["season"].astype(int), out["celebrity_name"].astype(str)))
    out["cs_idx"] = out["_cs_key"].map(pooled_fit["cs2idx"]).fillna(-1).astype(int)
    return out


def pooled_q_for_week(panel: pd.DataFrame, pooled_fit: dict, season: int, week: int) -> pd.DataFrame:
    g = panel[(panel["season"] == season) & (panel["week"] == week) & (panel["alive"] == True)].copy()
    g = build_features_for_rows(g, pooled_fit)
    X = g[pooled_fit["X_cols"]].to_numpy(dtype=np.float32)
    beta = pooled_fit["beta_hat"].astype(np.float32)
    logits = pooled_fit["bias_hat"] + X @ beta

    u = pooled_fit["u_hat"]
    add_u = np.zeros(len(g), dtype=np.float32)
    mask = g["cs_idx"].to_numpy() >= 0
    add_u[mask] = u[g.loc[mask, "cs_idx"].to_numpy()]
    logits = logits + add_u

    z = logits - logits.max()
    q = np.exp(z)
    q = q / q.sum()

    g["q_hat"] = q
    g["logit_hat"] = logits
    return g[["season", "week", "celebrity_name", "era", "j_metric", "q_hat", "logit_hat",
              "elim_this_week_end", "alive"]]


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


def posterior_mean_for_week(panel: pd.DataFrame, pooled_fit: dict, season: int, week: int,
                            *, kappa: float = None, B: int = 1200, tau_like: float = 0.15,
                            seed: int = 42):
    rng = np.random.default_rng(seed)
    g = pooled_q_for_week(panel, pooled_fit, season=season, week=week).copy()
    alive = g.copy()
    n = alive.shape[0]
    if n <= 1:
        return None

    # only do posterior reweighting for single-elim non-final weeks when label exists
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

    out = alive[["season", "week", "celebrity_name", "era", "j_metric", "q_hat", "elim_this_week_end"]].copy()
    out["p_mean"] = p_mean
    out["has_posterior"] = bool(elim_pos is not None)
    return out