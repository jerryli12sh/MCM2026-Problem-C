"""Track R: the review's integrated marginal-likelihood fit.

Track P fits the pooled popularity prior ``q = softmax(bias + X beta + u)`` by
treating the observed elimination as the softmin of ``J + q`` and then
reconditions a weekly ``p ~ Dirichlet(kappa q)`` posterior on the *same* observed
elimination (the paper's two-stage procedure).  That uses the outcome twice.
Track R instead fits the global parameters directly against the integrated
marginal likelihood

    P(Y | beta, u) = integral P(Y | p, J) Dirichlet(p | kappa q(beta, u)) dp

so the observed elimination is used exactly once per choice set.  The integral is
a fixed-noise Monte Carlo estimate (``B`` Dirichlet draws per choice set) and the
gradient is a score-function (REINFORCE) estimator: the softmin likelihood is the
unnormalized weight, the Dirichlet score ``d/dalpha log Dir(p | alpha)`` is the
influence, and the gradient chains through ``alpha = kappa q`` by the softmax
Jacobian.  See ``docs/METHOD_SPEC.md`` (Track R) and D-20260901-08.

Convergence and Monte Carlo error are reported with the fit
(``marginal_likelihood_diagnostics``) and stability across seeds / sample counts
is reported by ``fit_sensitivity``.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np
import pandas as pd

from dwts_reproduction.problem1.config import Problem1Config
from dwts_reproduction.problem1.softmin import (
    NumpyAdam,
    dirichlet_log_density_grad,
    log_mean_exp,
    softmax_np,
    softmin_logprob,
)
from dwts_reproduction.problem1.track_p import (
    PooledFit,
    WeekPack,
    pack_week_tensors,
)

EPS = 1e-12


# --------------------------------------------------------------------------- #
# One choice set: MC marginal log-likelihood and its score gradient
# --------------------------------------------------------------------------- #
def integrated_week_terms(
    wk: WeekPack,
    beta: np.ndarray,
    bias: float,
    u: np.ndarray,
    config: Problem1Config,
    rng: np.random.Generator,
    B: int,
) -> dict[str, Any]:
    """Monte Carlo terms for one single-elimination choice set.

    Draws ``p_b ~ Dirichlet(kappa q(beta, u))`` for ``b = 1..B``, computes each
    softmin log-likelihood of the observed elimination ``log f_b``, and returns

    - ``log_l``: the MC marginal log-likelihood ``log(mean_b f_b)``;
    - ``grad_eta``: the score-function gradient ``d log_l / d eta``;
    - ``se_log_l``: delta-method Monte Carlo standard error of ``log_l``;
    - ``ess``: effective sample size of the self-normalized weights ``w_b``;
    - ``q``/``alpha``/``p_samps``/``log_f`` for diagnostics and tests.

    The score gradient of ``log mean_b f_b`` with respect to the Dirichlet
    concentration is ``sum_b w_b * d/dalpha log Dir(p_b | alpha)`` where
    ``w_b = f_b / sum f_b`` (self-normalized importance weights); it chains to
    ``eta`` through ``alpha = kappa q`` and the softmax Jacobian.
    """
    X = wk.X.astype(float)
    J = wk.J.astype(float)
    eta = bias + X @ beta + u[wk.cs]
    q = softmax_np(eta)
    # ``alpha_floor`` (not ``eps``): concentrations below ~0.1 make the gamma
    # sampler underflow to exact zeros whose ``log`` corrupts the score gradient.
    alpha = np.clip(config.kappa * q, config.alpha_floor, None)

    p_samps = rng.dirichlet(alpha, size=B)
    log_f = softmin_logprob(J[None, :] + p_samps, wk.elim_pos, config.tau_like)
    log_l = float(log_mean_exp(log_f))
    # Self-normalized importance weights summing to 1 (``exp(log_f - log_l)`` is
    # ``f / mean(f)``, which sums to ``B``, so renormalize).
    w = np.exp(log_f - log_l)
    w = w / w.sum()
    score = dirichlet_log_density_grad(p_samps, alpha)  # (B, n)
    dlogL_dalpha = w @ score  # (n,)
    g_q = config.kappa * dlogL_dalpha
    grad_eta = q * (g_q - float(q @ g_q))  # softmax Jacobian chain

    f = np.exp(log_f)
    f_mean = float(f.mean())
    se_log_l = float(np.sqrt(f.var(ddof=0) / B) / max(f_mean, EPS))
    ess = float(1.0 / np.sum(w**2))
    return {
        "log_l": log_l,
        "grad_eta": grad_eta,
        "se_log_l": se_log_l,
        "ess": ess,
        "q": q,
        "alpha": alpha,
        "p_samps": p_samps,
        "log_f": log_f,
    }


# --------------------------------------------------------------------------- #
# Stage 1: integrated marginal-likelihood fit
# --------------------------------------------------------------------------- #
def fit_integrated_marginal(
    panel: pd.DataFrame,
    train_weeks: pd.DataFrame,
    config: Problem1Config,
    *,
    mc_B: int | None = None,
) -> tuple[PooledFit, dict[str, Any]]:
    """Fit ``q = softmax(bias + X beta + u)`` against the integrated marginal likelihood.

    A single explicit softmin temperature (``config.tau_like``, D-20260901-03)
    governs the elimination likelihood.  Each minibatch step draws fresh Dirichlet
    samples (the Monte Carlo noise averages out over steps), and gradients are
    accumulated week-by-week exactly as the Track P fit.  Returns
    ``(fit, diagnostics)`` where ``fit`` is a :class:`PooledFit` (its
    posterior/evaluation chain is shared with Track P) and ``diagnostics`` carries
    the full-data Monte Carlo error and effective sample sizes at the fitted point.
    """
    from dwts_reproduction.problem1.panel import build_feature_frame

    df_feat, meta = build_feature_frame(panel, train_weeks)
    packs = pack_week_tensors(df_feat, train_weeks, meta["X_cols"])
    n_weeks = len(packs)
    if n_weeks == 0:
        raise ValueError("No training weeks to fit.")
    n_cs = int(meta["n_cs"])
    p = len(meta["X_cols"])
    B = int(config.B if mc_B is None else mc_B)
    seed = int(config.seed)
    batch_size = min(int(config.batch_size), n_weeks)

    beta = np.zeros(p, dtype=np.float64)
    bias = np.zeros((), dtype=np.float64)
    u = np.zeros(n_cs, dtype=np.float64)
    optimizer = NumpyAdam([beta, bias, u], lr=float(config.lr))
    rng = np.random.default_rng(seed)
    st_list = [wk.st for wk in packs]
    loss_history: list[float] = []
    l2_beta = float(config.l2_beta)
    l2_u = float(config.l2_u)
    n_beta = float(p)
    n_u = float(n_cs)

    for step in range(1, int(config.n_steps) + 1):
        batch_ids = rng.choice(n_weeks, size=batch_size, replace=False)
        grad_beta = np.zeros_like(beta)
        grad_bias = np.zeros_like(bias)
        grad_u = np.zeros_like(u)
        nll = 0.0
        for k in batch_ids:
            wk = packs[int(k)]
            # Fresh, deterministic per-step-per-week draws so the MC noise averages
            # across steps while the whole fit stays reproducible from the seed.
            wk_seed = seed + step * 100_000 + wk.st[0] * 10_000 + wk.st[1]
            wk_rng = np.random.default_rng(wk_seed)
            terms = integrated_week_terms(wk, beta, float(bias), u, config, wk_rng, B)
            nll -= terms["log_l"]
            # ``grad_eta`` is d log_l / d eta (the *positive* log-likelihood
            # gradient); the objective is nll = -log_l, so accumulate the negative
            # exactly as Track P does (its -1/tau chain factor already flips sign).
            grad_eta = -terms["grad_eta"]
            grad_beta += wk.X.astype(float).T @ grad_eta
            grad_bias += grad_eta.sum()
            np.add.at(grad_u, wk.cs, grad_eta)
        nll = nll / max(1, len(batch_ids))
        reg = l2_beta * float(np.mean(beta**2)) + l2_u * float(np.mean(u**2))

        grad_beta = grad_beta / batch_size + (2.0 * l2_beta / n_beta) * beta
        grad_bias = grad_bias / batch_size
        grad_u = grad_u / batch_size + (2.0 * l2_u / n_u) * u

        optimizer.step([grad_beta, grad_bias, grad_u])
        if step == 1 or step == int(config.n_steps) or step % 50 == 0:
            loss_history.append(float(nll + reg))

    diagnostics = marginal_likelihood_diagnostics(packs, beta, float(bias), u, config, B=B)

    fit = PooledFit(
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
        n_cs=n_cs,
        seed=seed,
        era_mode=config.era_mode,
        loss_history=loss_history,
        hyperparameters={
            "fit_temperature": config.tau_like,  # single explicit temperature (D-20260901-03)
            "tau_like": config.tau_like,
            "kappa": config.kappa,
            "l2_beta": config.l2_beta,
            "l2_u": config.l2_u,
            "lr": config.lr,
            "n_steps": config.n_steps,
            "batch_size": config.batch_size,
            "B": B,
            "eps": config.eps,
            "alpha_floor": config.alpha_floor,
        },
        train_choice_sets=st_list,
        model_type="integrated_marginal_mc",
    )
    return fit, diagnostics


# --------------------------------------------------------------------------- #
# Diagnostics: Monte Carlo error, ESS, convergence, sensitivity
# --------------------------------------------------------------------------- #
def marginal_likelihood_diagnostics(
    packs: list[WeekPack],
    beta: np.ndarray,
    bias: float,
    u: np.ndarray,
    config: Problem1Config,
    *,
    B: int | None = None,
    seed: int | None = None,
) -> dict[str, Any]:
    """Per-choice-set MC error and ESS of the marginal log-likelihood at a fit.

    Fresh, fixed-seed draws per week give the delta-method standard error of
    ``log L_k`` and the self-normalized importance ESS ``1 / sum w_b^2``; both are
    averaged over the training weeks and reported with the total MC marginal
    log-likelihood.
    """
    B = int(B if B is not None else config.B)
    base_seed = int(seed if seed is not None else config.seed)
    week_stats = [
        integrated_week_terms(
            wk,
            beta,
            bias,
            u,
            config,
            np.random.default_rng(base_seed + 777 + wk.st[0] * 10_000 + wk.st[1]),
            B,
        )
        for wk in packs
    ]
    total_log_l = float(sum(s["log_l"] for s in week_stats))
    mc_se = float(np.mean([s["se_log_l"] for s in week_stats]))
    mc_se_relative = float(np.mean([s["se_log_l"] / max(abs(s["log_l"]), EPS) for s in week_stats]))
    ess_mean = float(np.mean([s["ess"] for s in week_stats]))
    return {
        "n_weeks": len(packs),
        "mc_log_l": total_log_l,
        "mc_se": mc_se,
        "mc_se_relative": mc_se_relative,
        "ess_mean": ess_mean,
        "B": B,
    }


def fit_sensitivity(
    panel: pd.DataFrame,
    train_weeks: pd.DataFrame,
    config: Problem1Config,
    *,
    seeds: tuple[int, ...] = (),
    fit_Bs: tuple[int, ...] = (),
    mc_B: int | None = None,
) -> dict[str, Any]:
    """Refit the integrated marginal across seeds and sample counts.

    Each variant is a full refit plus posterior reconstruction; the report gives
    the final training loss, the full-data MC marginal log-likelihood, and the
    top-1 reconstruction accuracy so a stability verdict can be read directly.
    """
    from dwts_reproduction.problem1.evaluate import evaluate_top1_accuracy
    from dwts_reproduction.problem1.track_p import infer_all_weekly_fan_support

    def _run(cfg: Problem1Config, fit_B: int | None) -> dict[str, Any]:
        fit, diag = fit_integrated_marginal(panel, train_weeks, cfg, mc_B=fit_B)
        posterior = infer_all_weekly_fan_support(panel, fit, cfg)
        _, _, acc = evaluate_top1_accuracy(panel, posterior, train_weeks)
        return {
            "final_loss": fit.loss_history[-1],
            "mc_log_l": diag["mc_log_l"],
            "top1_accuracy": acc["overall_top1_accuracy"],
        }

    results: dict[str, Any] = {"variants": {}}
    for s in seeds:
        results["variants"][f"seed_{s}"] = _run(replace(config, seed=s), mc_B)
    for b in fit_Bs:
        results["variants"][f"fit_B_{b}"] = _run(config, b)
    values = [v["final_loss"] for v in results["variants"].values()]
    if values:
        results["final_loss_spread"] = float(max(values) - min(values))
    return results
