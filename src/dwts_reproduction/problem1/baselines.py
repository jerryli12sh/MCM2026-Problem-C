"""In-season accuracy baselines (paper Fig. 1 / P-027, P-029).

The paper reports a per-season accuracy line comparing ``torch_model``
(``A = 0.952092``) against an XGBoost baseline (``A = 0.806554``).  Both lines
come from the legacy in-season cross-validation loop ``src/compare_models_cv.py``:
for each season a *per-season* pooled model is fitted on that season's
single-elimination training weeks, then each training week's posterior fan mean
``p_mean`` is compared against the observed eliminatee (``C_hat = J + p_mean``,
``argmin``).

This module reproduces both lines behind an explicit ``model_kind`` so the paper
figure is a pure function of the fitted tables:

- ``model_kind='xgb'`` — an exact port of ``src/xgb_baseline.py``.  The feature
  builder deliberately does **not** impute missing ``age`` (XGBoost treats NaN as
  a missing value natively), which differs from the Track P feature frame in
  ``panel.build_feature_frame`` (that one fills ``age`` with the mean).  The
  posterior seed scheme is the legacy ``seed + 100*season + week``
  (``compare_models_cv.py``), not the Track P ``config.seed + 1000*season +
  week``.
- ``model_kind='torch'`` — the repository's numpy rebuild of the torch model
  (``track_p.fit_pooled_softmin`` + ``track_p.posterior_draws_for_week``),
  fitted per season exactly as the legacy loop did.  This line is a visual item
  (P-029) with no registered numeric target; see D-20260901-13.

Neither line reports a claim about latent fan votes: ``p_mean`` is a posterior
estimate constrained by observed eliminations (D-20260901-06).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from dwts_reproduction.problem1.config import Problem1Config
from dwts_reproduction.problem1.panel import build_train_weeks
from dwts_reproduction.problem1.track_p import fit_pooled_softmin, posterior_draws_for_week

_XGB_IMPORT_ERROR: Exception | None = None
try:  # xgboost is an analysis extra; the module must import without it
    import xgboost as xgb  # noqa: F401
except Exception as exc:  # pragma: no cover - runtime import guard
    xgb = None  # type: ignore[assignment]
    _XGB_IMPORT_ERROR = exc

_XGB_EPS = 1e-12
XGB_X_COLS = ["j_metric_z", "age_z", "era_is_percent"]


# --------------------------------------------------------------------------- #
# XGBoost feature builder (no age imputation)
# --------------------------------------------------------------------------- #
def build_xgb_features(
    panel: pd.DataFrame, train_rows: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build standardized XGBoost row features from training rows.

    Faithful port of ``xgb_baseline._build_features`` (``src/xgb_baseline.py``):
    ``j_metric_z``, ``age_z``, ``era_is_percent``.  Unlike
    ``panel.build_feature_frame``, ``age`` is **not** mean-imputed; missing values
    stay NaN so XGBoost's native missing-value handling is used (this is what the
    paper's ``0.806554`` baseline was produced under).
    """
    df_feat = train_rows.copy()
    df_feat["j_metric"] = pd.to_numeric(df_feat["j_metric"], errors="coerce")
    if df_feat["j_metric"].isna().any():
        bad = df_feat.loc[df_feat["j_metric"].isna(), ["season", "week", "celebrity_name"]]
        raise ValueError(
            f"j_metric has NaN in XGBoost training rows. Examples: {bad.head().to_dict('records')}"
        )

    jm_mean = float(df_feat["j_metric"].mean())
    jm_std = float(df_feat["j_metric"].std(ddof=0) + _XGB_EPS)
    df_feat["j_metric_z"] = (df_feat["j_metric"] - jm_mean) / jm_std

    df_feat["age"] = pd.to_numeric(df_feat["age"], errors="coerce")
    use_age = bool(df_feat["age"].notna().any())
    if use_age:
        age_mean = float(df_feat["age"].mean())
        age_std = float(df_feat["age"].std(ddof=0) + _XGB_EPS)
        # No fillna: keep NaN for XGBoost missing-value handling (matches legacy).
        df_feat["age_z"] = (df_feat["age"] - age_mean) / age_std
    else:
        age_mean = None
        age_std = None
        df_feat["age_z"] = 0.0

    df_feat["era_is_percent"] = df_feat["era"].eq("percent").astype(float)

    meta: dict[str, Any] = {
        "jm_mean": jm_mean,
        "jm_std": jm_std,
        "use_age": use_age,
        "age_mean": age_mean,
        "age_std": age_std,
        "X_cols": list(XGB_X_COLS),
    }
    return df_feat, meta


def build_xgb_features_for_rows(df_rows: pd.DataFrame, fit: XgbPooledFit) -> pd.DataFrame:
    """Standardize rows for inference with the fitted XGBoost moments.

    Mirrors ``xgb_baseline.build_features_for_rows`` exactly: when the fit used
    ``age``, each row is mean-centered with the fitted moments (so an individual
    row with a missing ``age`` stays NaN and is treated by XGBoost as missing,
    matching training); ``age_z = 0`` is only used when the fit had no age data at
    all.
    """
    out = df_rows.copy()
    out["j_metric"] = pd.to_numeric(out["j_metric"], errors="coerce")
    out["j_metric_z"] = (out["j_metric"] - fit.jm_mean) / (fit.jm_std + _XGB_EPS)
    out["era_is_percent"] = out["era"].eq("percent").astype(float)
    if fit.use_age and "age" in out.columns and out["age"].notna().any():
        out["age"] = pd.to_numeric(out["age"], errors="coerce")
        assert fit.age_mean is not None and fit.age_std is not None
        out["age_z"] = (out["age"] - fit.age_mean) / (fit.age_std + _XGB_EPS)
    else:
        out["age_z"] = 0.0
    return out


# --------------------------------------------------------------------------- #
# XGBoost fitted-model container
# --------------------------------------------------------------------------- #
@dataclass
class XgbPooledFit:
    """Learned per-season XGBoost popularity prior (``model_type='xgboost'``)."""

    model: Any
    X_cols: list[str]
    jm_mean: float
    jm_std: float
    use_age: bool
    age_mean: float | None
    age_std: float | None
    seed: int
    kappa: float
    model_type: str = "xgboost"
    hyperparams: dict[str, float] = field(default_factory=dict)


def _require_xgb() -> Any:
    if xgb is None:
        raise ImportError(
            "xgboost is required for the XGBoost baseline. Install with "
            f"`pip install .[analysis]`. Original import error: {_XGB_IMPORT_ERROR}"
        )
    return xgb


# --------------------------------------------------------------------------- #
# XGBoost stage 1: per-season pooled fit
# --------------------------------------------------------------------------- #
def fit_xgb_pooled(
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
) -> XgbPooledFit:
    """Fit a pooled XGBoost classifier on one season's training weeks.

    Exact port of ``xgb_baseline.train_pooled_model``.  ``y = 1 - elim`` predicts
    "not eliminated" (positive = survives), matching the legacy popularity prior
    ``q_hat`` orientation.  The seed mirrors the legacy ``set_seed`` + XGBoost
    ``random_state``.
    """
    xgb_mod = _require_xgb()
    np.random.seed(seed)
    train_weeks = build_train_weeks(panel)
    keys = train_weeks[["season", "week"]]
    train_rows = panel[panel["alive"]].merge(keys, on=["season", "week"], how="inner")
    df_feat, meta = build_xgb_features(panel, train_rows)

    X = df_feat[meta["X_cols"]].to_numpy(dtype=np.float32)
    y = (1 - df_feat["elim_this_week_end"].astype(int)).to_numpy()

    n_pos = int(y.sum())
    n_neg = int((1 - y).sum())
    scale_pos_weight = float(n_neg / max(1, n_pos))

    model = xgb_mod.XGBClassifier(
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

    return XgbPooledFit(
        model=model,
        X_cols=list(meta["X_cols"]),
        jm_mean=meta["jm_mean"],
        jm_std=meta["jm_std"],
        use_age=bool(meta["use_age"]),
        age_mean=meta["age_mean"],
        age_std=meta["age_std"],
        seed=seed,
        kappa=kappa,
        hyperparams={
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
    )


# --------------------------------------------------------------------------- #
# XGBoost inference
# --------------------------------------------------------------------------- #
def xgb_q_for_week(panel: pd.DataFrame, fit: XgbPooledFit, season: int, week: int) -> pd.DataFrame:
    """Compute the XGBoost popularity-prior mean ``q_hat`` for one alive week.

    Mirrors ``xgb_baseline.pooled_q_for_week``: ``predict_proba[:, 1]`` clipped,
    logit-transformed, then softmaxed into a simplex ``q_hat``.
    """
    g = panel[(panel["season"].eq(season)) & (panel["week"].eq(week)) & (panel["alive"])].copy()
    if g.empty:
        raise ValueError(f"No alive rows for season {season}, week {week}.")
    g = build_xgb_features_for_rows(g, fit).sort_values("celebrity_name").copy()

    X = g[fit.X_cols].to_numpy(dtype=np.float32)
    proba = fit.model.predict_proba(X)[:, 1]
    proba = np.clip(proba, 1e-8, 1.0 - 1e-8)
    logit_hat = np.log(proba / (1.0 - proba))

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
    ].copy()


def _softmin_logprob_elim(cost: np.ndarray, elim_pos: int, tau: float) -> np.ndarray:
    """Stable per-draw softmin log-probability of the observed eliminatee."""
    z = -np.asarray(cost, dtype=float) / tau
    z = z - z.max(axis=1, keepdims=True)
    log_denom = np.log(np.exp(z).sum(axis=1))
    return np.asarray(z[:, elim_pos] - log_denom)


def xgb_posterior_mean_for_week(
    panel: pd.DataFrame,
    fit: XgbPooledFit,
    season: int,
    week: int,
    *,
    kappa: float | None = None,
    B: int = 1200,
    tau_like: float = 0.15,
    seed: int = 42,
) -> pd.DataFrame | None:
    """Condition a weekly ``Dirichlet(kappa q_hat)`` posterior on the elimination.

    Exact port of ``xgb_baseline.posterior_mean_for_week``.  The seed is the
    caller-provided posterior seed (the legacy in-season loop passes
    ``seed + 100*season + week``).  Returns ``None`` for degenerate alive sets.
    """
    rng = np.random.default_rng(seed)
    g = xgb_q_for_week(panel, fit, season=season, week=week).copy()
    n = g.shape[0]
    if n <= 1:
        return None

    if g["elim_this_week_end"].sum() == 1:
        elim_name = g.loc[g["elim_this_week_end"], "celebrity_name"].iloc[0]
        names = g["celebrity_name"].to_numpy()
        elim_pos = int(np.where(names == elim_name)[0][0])
    else:
        elim_pos = None

    kappa_eff = fit.kappa if kappa is None else kappa
    q = g["q_hat"].to_numpy()
    alpha = kappa_eff * q
    p_samps = rng.dirichlet(alpha, size=B)

    if elim_pos is not None:
        j = g["j_metric"].to_numpy()
        cost = j[None, :] + p_samps
        logp = _softmin_logprob_elim(cost, elim_pos, tau=tau_like)
        weights = np.exp(logp - logp.max())
        weights = weights / weights.sum()
    else:
        weights = np.ones(B) / B

    p_mean = (weights[:, None] * p_samps).sum(axis=0)

    out = g[
        ["season", "week", "celebrity_name", "era", "j_metric", "q_hat", "elim_this_week_end"]
    ].copy()
    out["p_mean"] = p_mean
    out["has_posterior"] = bool(elim_pos is not None)
    return out


# --------------------------------------------------------------------------- #
# Week accuracy and the in-season evaluation loop
# --------------------------------------------------------------------------- #
def week_accuracy_from_posterior(post_df: pd.DataFrame | None) -> int | None:
    """1 if the posterior-mean ``argmin(J + p_mean)`` is the observed eliminatee.

    Returns ``None`` when the week is not a single-elimination week.  This is the
    exact definition from ``compare_models_cv.week_accuracy_from_posterior``.
    """
    if post_df is None or post_df["elim_this_week_end"].sum() != 1:
        return None
    post_df = post_df.copy()
    post_df["C_hat"] = post_df["j_metric"] + post_df["p_mean"]
    pred_pos = int(np.argmin(post_df["C_hat"].to_numpy()))
    actual_pos = int(np.where(post_df["elim_this_week_end"].to_numpy())[0][0])
    return 1 if pred_pos == actual_pos else 0


def evaluate_inseason_accuracy(
    panel: pd.DataFrame,
    model_kind: str,
    *,
    seed: int = 42,
    kappa: float = 10.0,
    tau_like: float = 0.15,
    B: int = 1200,
    config: Problem1Config | None = None,
) -> pd.DataFrame:
    """Return one row per training week with the model's reconstruction accuracy.

    Ports ``compare_models_cv.evaluate_model_inseason``.  For each season a
    per-season model is fitted; each single-elimination, non-final training week
    then gets a posterior fan mean and a 0/1 accuracy.  ``model_kind`` is
    ``'xgb'`` or ``'torch'``.

    - ``'xgb'`` uses the exact legacy posterior seed ``seed + 100*season + week``.
    - ``'torch'`` uses the repository rebuild's per-week posterior draws
      (``config.seed + 1000*season + week`` internally); ``config`` is required.
    """
    if model_kind not in {"xgb", "torch"}:
        raise ValueError("model_kind must be 'xgb' or 'torch'")
    if model_kind == "torch" and config is None:
        raise ValueError("config is required for model_kind='torch'")

    model_label = "xgboost_baseline" if model_kind == "xgb" else "torch_model"
    rows: list[dict[str, Any]] = []
    seasons = sorted(panel["season"].dropna().astype(int).unique().tolist())
    for s in seasons:
        panel_s = panel[panel["season"].eq(s)].copy()
        weeks_s = build_train_weeks(panel_s)
        if weeks_s.empty:
            continue

        if model_kind == "xgb":
            xgb_fit = fit_xgb_pooled(panel_s, seed=seed, kappa=kappa)
            for _, row in weeks_s.iterrows():
                w = int(row["week"])
                post = xgb_posterior_mean_for_week(
                    panel_s,
                    xgb_fit,
                    season=s,
                    week=w,
                    kappa=kappa,
                    tau_like=tau_like,
                    B=B,
                    seed=seed + s * 100 + w,
                )
                acc = week_accuracy_from_posterior(post)
                if acc is None:
                    continue
                rows.append({"model": model_label, "season": s, "week": w, "accuracy": acc})
        else:
            assert config is not None
            pooled_fit = fit_pooled_softmin(panel_s, weeks_s, config)
            for _, row in weeks_s.iterrows():
                w = int(row["week"])
                res = posterior_draws_for_week(panel_s, pooled_fit, season=s, week=w, config=config)
                post = res["alive"].copy()
                post["p_mean"] = res["weights"] @ res["samples"]
                acc = week_accuracy_from_posterior(post)
                if acc is None:
                    continue
                rows.append({"model": model_label, "season": s, "week": w, "accuracy": acc})
    return pd.DataFrame(rows)


def accuracy_by_season(by_week: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-(model, season) mean accuracy (the paper Fig. 1 line data)."""
    return (
        by_week.groupby(["model", "season"], as_index=False)["accuracy"]
        .mean()
        .sort_values(["model", "season"])
        .reset_index(drop=True)
    )
