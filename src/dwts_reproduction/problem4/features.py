"""Problem 4 shared inputs, pooled-fit reload, and q-hat projection.

Faithful ports of the helpers duplicated in the two legacy producers
``../src/season_simulator.py`` and ``../src/season_simulator2.py`` plus the
feature builder in ``../src/model.py`` (``build_features_for_rows``).  The
legacy simulators each duplicated these helpers verbatim; the repo ports them
once here and re-exports them through ``v1``/``v2`` so behaviour stays
identical.

The pooled fit is **not** re-trained: :func:`load_pooled_fit_dict` rebuilds the
exact ``pooled_fit`` dict ``model.train_pooled_model`` returns by reading the
saved Problem 1 Track P fit (``problem1_fit_meta_P.json`` + the float32 arrays
in ``problem1_fit_arrays_P.npz``).  The legacy simulator training defaults match
the saved fit hyperparameters bit-for-bit (seed 42, tau 0.05, l2 0.05,
kappa 10, lr 0.02, 600 steps, batch 32), so the simulators run on the *same*
posterior center the legacy runs used — no hidden re-training.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Legacy CLI defaults (problem-4 mechanism simulators).
V1_DEFAULTS: dict[str, Any] = {
    "schemes": ("S1", "S2", "S3"),
    "n_sims": 300,
    "seed": 42,
    "wJ": 0.5,
    "wF": 0.5,
    "K": 3,
    "m_early_elims": 8,
    "final_n": 3,
    "era_cutoff": 28,
    "kappa": 10.0,
}
V2_DEFAULTS: dict[str, Any] = {
    "schemes": ("V4", "V5"),
    "n_sims": 300,
    "seed": 42,
    "gamma": 0.45,
    "delta": 1.35,
    "wJ": 0.80,
    "wF": 0.20,
    "L": 2,
    "mu": 0.01,
    "c": 2.0,
    "kappa": 10.0,
    "final_n": 3,
    "era_cutoff": 28,
}

# Paper Table-1 named controversy cases (shared by both simulators).
CONTROVERSY_CASES: tuple[tuple[int, str], ...] = (
    (2, "Jerry Rice"),
    (4, "Billy Ray Cyrus"),
    (11, "Bristol Palin"),
    (27, "Bobby Bones"),
    (27, "Tinashe"),
    (31, "Vinny Guadagnino"),
)
# Popularity-driven cases the paper claims V2 corrects (P-084/P-085).
POPULARITY_CASES: tuple[tuple[int, str], ...] = (
    (2, "Jerry Rice"),
    (4, "Billy Ray Cyrus"),
    (11, "Bristol Palin"),
    (27, "Bobby Bones"),
)


def load_pooled_fit_dict(meta_path: str | Path, arrays_path: str | Path) -> dict[str, Any]:
    """Rebuild the legacy ``pooled_fit`` dict from the saved Problem 1 fit.

    Returns the dict shape ``model.train_pooled_model`` returns
    (``beta_hat``, ``bias_hat``, ``u_hat``, ``X_cols``, ``jm_mean``,
    ``jm_std``, ``use_age``, ``age_mean``, ``age_std``, ``cs2idx``, ``kappa``,
    ``seed``, ``hyperparams``).  The numeric arrays are taken from the
    ``.npz`` so they are float32 bit-for-bit equal to what training returned;
    the scalars and the ``"season::name" -> idx`` map come from the JSON meta.
    """
    meta = json.loads(Path(meta_path).read_text(encoding="utf-8"))
    arrays = np.load(arrays_path)
    cs2idx = {
        (int(int(key.split("::")[0])), key.split("::", 1)[1]): int(idx)
        for key, idx in meta["cs2idx_json"].items()
    }
    return {
        "beta_hat": np.asarray(arrays["beta"], dtype=np.float32),
        "bias_hat": float(arrays["bias"]),
        "u_hat": np.asarray(arrays["u"], dtype=np.float32),
        "X_cols": list(meta["X_cols"]),
        "tau": float(meta.get("tau", meta["hyperparameters"].get("tau_train", 0.05))),
        "jm_mean": float(meta["jm_mean"]),
        "jm_std": float(meta["jm_std"]),
        "use_age": bool(meta["use_age"]),
        "age_mean": None if meta["age_mean"] is None else float(meta["age_mean"]),
        "age_std": None if meta["age_std"] is None else float(meta["age_std"]),
        "cs2idx": cs2idx,
        "kappa": float(meta["hyperparameters"].get("kappa", 10.0)),
        "seed": int(meta["seed"]),
        "hyperparams": {k: float(v) for k, v in meta["hyperparameters"].items()},
    }


def build_features_for_rows(df_rows: pd.DataFrame, pooled_fit: dict) -> pd.DataFrame:
    """Port of ``model.build_features_for_rows`` (legacy ``model.py``).

    Builds ``j_metric_z``, ``era_is_percent``, ``age_z`` and the contestant
    index ``cs_idx`` used to add the contestant random effect ``u``.  The
    ``age`` column, when present and non-null, is standardized with the fit's
    ``age_mean``/``age_std``; otherwise ``age_z`` is zero.
    """
    out = df_rows.copy()
    out["j_metric"] = pd.to_numeric(out["j_metric"], errors="coerce")
    out["j_metric_z"] = (out["j_metric"] - pooled_fit["jm_mean"]) / (pooled_fit["jm_std"] + 1e-12)
    out["era_is_percent"] = (out["era"].astype(str) == "percent").astype(float)

    if pooled_fit["use_age"] and "age" in out.columns and out["age"].notna().any():
        out["age"] = pd.to_numeric(out["age"], errors="coerce")
        out["age_z"] = (out["age"] - pooled_fit["age_mean"]) / (pooled_fit["age_std"] + 1e-12)
    else:
        out["age_z"] = 0.0

    out["_cs_key"] = list(
        zip(out["season"].astype(int), out["celebrity_name"].astype(str), strict=True)
    )
    out["cs_idx"] = out["_cs_key"].map(pooled_fit["cs2idx"]).fillna(-1).astype(int)
    return out


def compute_q_hat(df_rows: pd.DataFrame, pooled_fit: dict) -> np.ndarray:
    """Project the posterior center ``q = softmax(X beta + u)`` (both simulators).

    Exact port of ``_compute_q_hat`` in ``season_simulator.py`` /
    ``season_simulator2.py``.  Returns the float64 Dirichlet concentration
    mean vector (``logits`` stay float32; ``numpy`` promotes the subsequent
    ``rng.dirichlet`` draws to float64, as in the legacy runs).
    """
    feat = build_features_for_rows(df_rows, pooled_fit)
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
    return np.asarray(q)


def load_weekly(path: str | Path) -> pd.DataFrame:
    """Read and normalize the weekly table (``_prepare_weekly_table``)."""
    df = pd.read_csv(path)
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
    df["season"] = df["season"].astype(int)
    df["week"] = df["week"].astype(int)
    df["celebrity_name"] = df["celebrity_name"].astype(str)
    return df


def load_clean(path: str | Path) -> pd.DataFrame:
    """Read the contestant clean table (ages)."""
    return pd.read_csv(path)


def load_archetypes(path: str | Path) -> pd.DataFrame:
    """Read the archetype table (``_load_archetypes``)."""
    df = pd.read_csv(path)
    if "archetype" not in df.columns:
        raise KeyError("contestant_archetypes.csv must contain `archetype`.")
    df = df[["season", "celebrity_name", "archetype"]].copy()
    df["season"] = df["season"].astype(int)
    df["celebrity_name"] = df["celebrity_name"].astype(str)
    df["archetype"] = df["archetype"].astype(str)
    return df


def age_map(df_clean: pd.DataFrame) -> dict[str, float]:
    """Port of ``_age_map``: name -> age at season, deduplicated."""
    if "celebrity_age_during_season" in df_clean.columns:
        age_col = "celebrity_age_during_season"
    elif "celebrity_age" in df_clean.columns:
        age_col = "celebrity_age"
    else:
        return {}
    tmp = df_clean[["celebrity_name", age_col]].dropna()
    tmp = tmp.drop_duplicates(subset=["celebrity_name"])
    return dict(tmp.set_index("celebrity_name")[age_col].astype(float).to_dict())


def rank_desc(x: np.ndarray) -> np.ndarray:
    """Descending average-rank of an array (best = 1)."""
    return np.asarray(pd.Series(x).rank(ascending=False, method="average").to_numpy())


def softmax(x: np.ndarray) -> np.ndarray:
    """Stable softmax (used for the rank-era judge signal)."""
    z = x - np.max(x)
    e = np.exp(z)
    return np.asarray(e / e.sum())


def get_week_scores(
    df_weekly: pd.DataFrame,
    season: int,
    week: int,
    active_names: list[str],
    last_scores: dict[str, float],
    season_mean: float,
) -> dict[str, float]:
    """Port of ``_get_week_scores`` with last-score/season-mean fallbacks."""
    g = df_weekly[(df_weekly["season"] == season) & (df_weekly["week"] == week)]
    g = g[g["celebrity_name"].isin(active_names)]
    score_map: dict[str, float] = {}
    if "total_judge_score" in g.columns:
        score_map = g.set_index("celebrity_name")["total_judge_score"].to_dict()
    elif "judge_total" in g.columns:
        score_map = g.set_index("celebrity_name")["judge_total"].to_dict()
    else:
        raise KeyError("df_weekly must contain total_judge_score or judge_total.")

    scores: dict[str, float] = {}
    for name in active_names:
        v = score_map.get(name)
        if v is not None and not pd.isna(v):
            scores[name] = float(v)
        elif name in last_scores:
            scores[name] = float(last_scores[name])
        else:
            scores[name] = float(season_mean)
    return scores
