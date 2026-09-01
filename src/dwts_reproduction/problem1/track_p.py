"""Track P: the paper's two-stage latent fan-support reconstruction.

Stage 1 fits a pooled popularity prior ``q = softmax(bias + X beta + u)`` by
penalized softmin negative-log-likelihood over single-elimination weeks (see
``fit_pooled_softmin``).  Stage 2 conditions a weekly ``p ~ Dirichlet(kappa q)``
on the observed elimination via importance sampling (see
``posterior_draws_for_week`` / ``infer_all_weekly_fan_support``).

The pooled fit mirrors the reference ``torch`` implementation with hand-written
numpy + Adam (see ``softmin.NumpyAdam``) so the reference outputs can be
reproduced without a torch dependency.  The fit runs in float32 (matching torch
tensors); inference runs in float64 (matching the reference inference path).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from dwts_reproduction.problem1.config import Problem1Config
from dwts_reproduction.problem1.softmin import NumpyAdam, logsumexp, softmax_np

EPS_TAU = 1e-12


# --------------------------------------------------------------------------- #
# Fitted model container
# --------------------------------------------------------------------------- #
@dataclass
class PooledFit:
    """Learned pooled popularity prior (Track P stage 1)."""

    beta: np.ndarray
    bias: float
    u: np.ndarray
    X_cols: list[str]
    jm_mean: float
    jm_std: float
    use_age: bool
    age_mean: float | None
    age_std: float | None
    cs2idx: dict[tuple[int, str], int]
    n_cs: int
    seed: int
    era_mode: str
    loss_history: list[float] = field(default_factory=list)
    hyperparameters: dict[str, float] = field(default_factory=dict)
    train_choice_sets: list[tuple[int, int]] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        """JSON-ready serialization of the fit metadata (arrays excluded)."""
        return {
            "model_type": "pooled_softmin_numpy",
            "beta_hat": self.beta.astype(float).tolist(),
            "bias_hat": float(self.bias),
            "u_hat": self.u.astype(float).tolist(),
            "X_cols": list(self.X_cols),
            "jm_mean": self.jm_mean,
            "jm_std": self.jm_std,
            "use_age": self.use_age,
            "age_mean": self.age_mean,
            "age_std": self.age_std,
            "cs2idx_json": {
                f"{int(s)}::{name}": int(idx) for (s, name), idx in self.cs2idx.items()
            },
            "n_cs": self.n_cs,
            "seed": self.seed,
            "era_mode": self.era_mode,
            "loss_history": [float(v) for v in self.loss_history],
            "train_choice_sets": [tuple(int(x) for x in st) for st in self.train_choice_sets],
            "hyperparameters": {k: float(v) for k, v in self.hyperparameters.items()},
        }


# --------------------------------------------------------------------------- #
# Choice-set packing
# --------------------------------------------------------------------------- #
@dataclass
class WeekPack:
    """One single-elimination choice set in canonical (celebrity-name) order."""

    X: np.ndarray  # (n, p) float32
    J: np.ndarray  # (n,) float32
    cs: np.ndarray  # (n,) int64 contestant-season indices
    elim_pos: int
    st: tuple[int, int]


def pack_week_tensors(
    df_feat: pd.DataFrame, train_weeks: pd.DataFrame, X_cols: list[str]
) -> list[WeekPack]:
    """Pack each training week into a canonical-order choice set.

    Ordering within a week is by ``celebrity_name`` so ``elim_pos`` is stable and
    matches the reference rebuild exactly.
    """
    elim_lookup = train_weeks.set_index(["season", "week"])["true_eliminatee"].to_dict()
    packs: list[WeekPack] = []
    for s, wk in train_weeks[["season", "week"]].itertuples(index=False):
        g = df_feat[(df_feat["season"] == s) & (df_feat["week"] == wk)].sort_values(
            "celebrity_name"
        )
        names = g["celebrity_name"].to_numpy()
        true_elim = elim_lookup[(s, wk)]
        pos = np.where(names == true_elim)[0]
        if len(pos) != 1:
            raise ValueError(
                f"Could not locate true eliminatee for season {s}, week {wk}: {true_elim}"
            )
        packs.append(
            WeekPack(
                X=g[X_cols].to_numpy(dtype=np.float32),
                J=g["j_metric"].to_numpy(dtype=np.float32),
                cs=g["cs_idx"].to_numpy(dtype=np.int64),
                elim_pos=int(pos[0]),
                st=(int(s), int(wk)),
            )
        )
    return packs


# --------------------------------------------------------------------------- #
# Stage 1: pooled penalized softmin fit
# --------------------------------------------------------------------------- #
def fit_pooled_softmin(
    panel: pd.DataFrame, train_weeks: pd.DataFrame, config: Problem1Config
) -> PooledFit:
    """Fit the pooled popularity prior by penalized softmin likelihood.

    Replicates the reference torch model and optimizer (float32 arithmetic, Adam
    lr 0.02, betas (0.9, 0.999), eps 1e-8, minibatch choice without replacement
    from a ``RandomState(config.seed)`` sequence).  Each training week is one
    choice set; elimination is the softmin of ``J + q`` at temperature
    ``tau_train``.
    """
    from dwts_reproduction.problem1.panel import build_feature_frame

    df_feat, meta = build_feature_frame(panel, train_weeks)
    packs = pack_week_tensors(df_feat, train_weeks, meta["X_cols"])
    n_weeks = len(packs)
    if n_weeks == 0:
        raise ValueError("No training weeks to fit.")
    n_cs = meta["n_cs"]
    p = len(meta["X_cols"])
    tau = float(config.tau_train)

    st_list = [wk.st for wk in packs]

    beta = np.zeros(p, dtype=np.float32)
    bias = np.zeros((), dtype=np.float32)
    u = np.zeros(n_cs, dtype=np.float32)
    optimizer = NumpyAdam([beta, bias, u], lr=float(config.lr))

    # The reference rebuild seeds the global numpy RNG and draws each batch with
    # ``np.random.choice(..., replace=False)``; the same seed reproduces the same
    # batch sequence (legacy MT19937).  Alive-set sizes vary across weeks, so the
    # minibatch loss is summed week-by-week exactly as in the torch reference.
    np.random.seed(config.seed)
    idx_all = np.arange(n_weeks)
    batch_size = min(config.batch_size, n_weeks)
    loss_history: list[float] = []
    l2_beta = float(config.l2_beta)
    l2_u = float(config.l2_u)
    n_beta = float(beta.size)
    n_u = float(n_cs)
    inv_tau = np.float32(1.0 / tau)

    for step in range(1, config.n_steps + 1):
        batch_ids = np.random.choice(idx_all, size=batch_size, replace=False)
        grad_beta = np.zeros_like(beta)
        grad_bias = np.zeros((), dtype=np.float32)
        grad_u = np.zeros(n_cs, dtype=np.float32)
        nll = 0.0
        for k in batch_ids:
            wk = packs[int(k)]
            eta = bias + wk.X @ beta + u[wk.cs]  # (n,) float32
            q = eta - eta.max()
            q = np.exp(q)
            q = q / q.sum()

            z = -(wk.J + q) / np.float32(tau)
            z = z - z.max()
            # stable log-softmax, exactly as ``F.log_softmax`` in the torch
            # reference (no epsilon floor on the probability).
            logp = z - np.log(np.exp(z).sum())
            nll += -logp[wk.elim_pos]
            psoft = np.exp(z)
            psoft = psoft / psoft.sum()

            # softmin gradient dL/dz = -(1/tau)(psoft - e_elim), chained through
            # q = softmax(eta) by the softmax Jacobian: grad_eta = q o (g - (q.g)).
            onehot = np.zeros_like(psoft)
            onehot[wk.elim_pos] = np.float32(1.0)
            g = -inv_tau * (psoft - onehot)
            dot = float((q * g).sum())
            grad_eta = q * (g - dot)  # (n,) float32

            grad_beta += wk.X.T @ grad_eta
            grad_bias += grad_eta.sum()
            np.add.at(grad_u, wk.cs, grad_eta)
        nll = nll / max(1, len(batch_ids))
        reg = float(l2_beta * np.mean(beta**2) + l2_u * np.mean(u**2))

        grad_beta = grad_beta / np.float32(batch_size) + np.float32(2.0 * l2_beta / n_beta) * beta
        grad_bias = grad_bias / np.float32(batch_size)
        grad_u = grad_u / np.float32(batch_size) + np.float32(2.0 * l2_u / n_u) * u

        optimizer.step([grad_beta, grad_bias, grad_u])
        if step == 1 or step == config.n_steps or step % 50 == 0:
            loss_history.append(nll + reg)

    return PooledFit(
        beta=beta,
        bias=float(bias),
        u=u,
        X_cols=list(meta["X_cols"]),
        jm_mean=meta["jm_mean"],
        jm_std=meta["jm_std"],
        use_age=bool(meta["use_age"]),
        age_mean=meta["age_mean"],
        age_std=meta["age_std"],
        cs2idx=meta["cs2idx"],
        n_cs=int(n_cs),
        seed=config.seed,
        era_mode=config.era_mode,
        loss_history=loss_history,
        hyperparameters={
            "tau_train": config.tau_train,
            "tau_like": config.tau_like,
            "kappa": config.kappa,
            "l2_beta": config.l2_beta,
            "l2_u": config.l2_u,
            "lr": config.lr,
            "n_steps": config.n_steps,
            "batch_size": config.batch_size,
            "B": config.B,
            "eps": config.eps,
        },
        train_choice_sets=st_list,
    )


# --------------------------------------------------------------------------- #
# Stage 1 inference: pooled prior q_hat for one alive week
# --------------------------------------------------------------------------- #
def _build_features_for_rows(df_rows: pd.DataFrame, fit: PooledFit) -> pd.DataFrame:
    """Standardize rows using the fitted moments (float64-capable, inference path)."""
    out = df_rows.copy()
    out["j_metric"] = pd.to_numeric(out["j_metric"], errors="coerce")
    # Non-training weeks sometimes lack a comparable judge signal; fall back to
    # the training mean so q_hat stays a valid simplex vector.  The posterior
    # likelihood is only used when the whole single-elimination week has finite J.
    j_for_prior = out["j_metric"].fillna(fit.jm_mean)
    out["j_metric_z"] = (j_for_prior - fit.jm_mean) / (fit.jm_std + EPS_TAU)
    out["era_is_percent"] = out["era"].eq("percent").astype(float)
    if fit.use_age:
        assert fit.age_mean is not None and fit.age_std is not None
        age = pd.to_numeric(out["age"], errors="coerce").fillna(fit.age_mean)
        out["age_z"] = (age - fit.age_mean) / (fit.age_std + EPS_TAU)
    else:
        out["age_z"] = 0.0
    out["_cs_key"] = list(
        zip(out["season"].astype(int), out["celebrity_name"].astype(str), strict=True)
    )
    out["cs_idx"] = out["_cs_key"].map(fit.cs2idx).fillna(-1).astype(int)
    return out


def pooled_q_for_week(panel: pd.DataFrame, fit: PooledFit, season: int, week: int) -> pd.DataFrame:
    """Compute the trained popularity-prior mean ``q_hat`` for one alive set (float64)."""
    g = panel[(panel["season"].eq(season)) & (panel["week"].eq(week)) & (panel["alive"])].copy()
    if g.empty:
        raise ValueError(f"No alive rows for season {season}, week {week}.")
    g = _build_features_for_rows(g, fit).sort_values("celebrity_name").copy()

    X = g[fit.X_cols].to_numpy(dtype=np.float64)
    beta = np.asarray(fit.beta, dtype=np.float64)
    logits = fit.bias + X @ beta
    cs_idx = g["cs_idx"].to_numpy()
    mask = cs_idx >= 0
    u = np.asarray(fit.u, dtype=np.float64)
    logits[mask] = logits[mask] + u[cs_idx[mask]]
    q = softmax_np(logits)

    g["q_hat"] = q
    g["logit_hat"] = logits
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
            "is_final_week",
            "max_week",
        ]
    ].copy()


# --------------------------------------------------------------------------- #
# Stage 2: weekly Dirichlet + softmin importance-sampling posterior
# --------------------------------------------------------------------------- #
def posterior_draws_for_week(
    panel: pd.DataFrame,
    fit: PooledFit,
    season: int,
    week: int,
    config: Problem1Config,
    *,
    has_posterior_mode: str = "rebuild",
) -> dict[str, Any]:
    """Draw ``Dirichlet(kappa q_hat)`` fan-share samples and reweight them.

    ``has_posterior_mode`` selects which weeks receive the softmin reweighting:

    - ``"rebuild"`` (default, matches the review rebuild): reweight only
      single-elimination, non-final weeks with a full finite judge signal.
    - ``"legacy"`` (matches ``posterior_uncertainty.py``, used to reproduce the
      paper's cumulative-consistency event tables): reweight every
      single-elimination week, including finales.

    Returns the alive frame, samples, weights, ESS, and flags.
    """
    rng = np.random.default_rng(config.seed + int(season) * 1000 + int(week))
    alive = pooled_q_for_week(panel, fit, season, week)
    q = np.clip(alive["q_hat"].to_numpy(dtype=float), config.eps, 1.0)
    q = q / q.sum()
    alpha = np.clip(config.kappa * q, config.eps, None)
    p_samps = rng.dirichlet(alpha, size=config.B)

    single_elim = bool(alive["elim_this_week_end"].sum() == 1)
    if has_posterior_mode == "rebuild":
        has_posterior = bool(
            single_elim and (not alive["is_final_week"].iloc[0]) and alive["j_metric"].notna().all()
        )
    elif has_posterior_mode == "legacy":
        has_posterior = single_elim
    else:
        raise ValueError("has_posterior_mode must be 'rebuild' or 'legacy'.")

    if has_posterior:
        elim_name = alive.loc[alive["elim_this_week_end"], "celebrity_name"].iloc[0]
        elim_pos = int(np.where(alive["celebrity_name"].to_numpy() == elim_name)[0][0])
        j = alive["j_metric"].to_numpy(dtype=float)
        logp = _softmin_logprob_elim(j[None, :] + p_samps, elim_pos, config.tau_like)
        weights = np.exp(logp - logp.max())
        weights = weights / weights.sum()
    else:
        elim_pos = None
        weights = np.ones(config.B, dtype=float) / config.B

    if not np.isfinite(weights).all():
        raise FloatingPointError(f"Non-finite posterior weights for season {season}, week {week}.")
    ess = float(1.0 / np.sum(weights**2))
    return {
        "alive": alive,
        "samples": p_samps,
        "weights": weights,
        "ess": ess,
        "ess_ratio": ess / float(config.B),
        "has_posterior": has_posterior,
        "elim_pos": elim_pos,
    }


def _softmin_logprob_elim(cost: np.ndarray, elim_pos: int, tau: float) -> np.ndarray:
    z = -np.asarray(cost, dtype=float) / tau
    z = z - z.max(axis=1, keepdims=True)
    return np.asarray(z[:, elim_pos] - logsumexp(z, axis=1))


def infer_all_weekly_fan_support(
    panel: pd.DataFrame, fit: PooledFit, config: Problem1Config
) -> pd.DataFrame:
    """Infer posterior fan-share summaries for every alive season-week."""
    rows: list[dict[str, Any]] = []
    weeks = (
        panel[panel["alive"]][["season", "week"]]
        .drop_duplicates()
        .sort_values(["season", "week"])
        .itertuples(index=False)
    )
    for season, week in weeks:
        res = posterior_draws_for_week(panel, fit, int(season), int(week), config)
        alive = res["alive"].reset_index(drop=True)
        p_samps = res["samples"]
        weights = res["weights"]
        p_mean = weights @ p_samps

        pcp_indicator = None
        if res["elim_pos"] is not None:
            j = alive["j_metric"].to_numpy(dtype=float)
            pred_pos_by_sample = np.argmin(j[None, :] + p_samps, axis=1)
            pcp_indicator = pred_pos_by_sample == int(res["elim_pos"])
            pcp_unweighted = float(pcp_indicator.mean())
            pcp_weighted = float(np.sum(weights * pcp_indicator))
        else:
            pcp_unweighted = np.nan
            pcp_weighted = np.nan

        for i, r in alive.iterrows():
            lo, hi = weighted_quantile(p_samps[:, i], [0.05, 0.95], weights)
            ci_width = float(hi - lo)
            pm = float(p_mean[i])
            rows.append(
                {
                    "season": int(r["season"]),
                    "week": int(r["week"]),
                    "celebrity_name": r["celebrity_name"],
                    "era": r["era"],
                    "j_metric": float(r["j_metric"]),
                    "q_hat": float(r["q_hat"]),
                    "p_mean": pm,
                    "ci_lo_05": float(lo),
                    "ci_hi_95": float(hi),
                    "ci_width": ci_width,
                    "ci_rel_width": float(ci_width / (pm + config.eps)),
                    "ess": float(res["ess"]),
                    "ess_ratio": float(res["ess_ratio"]),
                    "pcp_unweighted": pcp_unweighted,
                    "pcp_weighted": pcp_weighted,
                    "has_posterior": bool(res["has_posterior"]),
                    "alive_n": int(len(alive)),
                    "elim_this_week_end": bool(r["elim_this_week_end"]),
                    "B": int(config.B),
                    "kappa": float(config.kappa),
                    "tau_like": float(config.tau_like),
                }
            )
    return pd.DataFrame(rows)


def weighted_quantile(
    values: np.ndarray, quantiles: float | list[float], weights: np.ndarray
) -> np.ndarray:
    """Weighted empirical quantiles via linear interpolation of the weighted CDF."""
    values = np.asarray(values, dtype=float)
    q = np.atleast_1d(np.asarray(quantiles, dtype=float))
    weights = np.asarray(weights, dtype=float)
    sorter = np.argsort(values)
    v = values[sorter]
    w = weights[sorter]
    cdf = np.cumsum(w)
    cdf = cdf / cdf[-1]
    return np.asarray(np.interp(q, cdf, v))
