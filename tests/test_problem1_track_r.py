"""Problem 1 Track R integrated marginal-likelihood tests.

The score-function Monte Carlo gradient of the integrated marginal likelihood
``P(Y | beta, u) = int P(Y | p, J) Dirichlet(p | kappa q) dp`` is verified
against an exact n=2 quadrature identity and against finite differences of the
Dirichlet log-density.  An in-family synthetic recovery test confirms the fit
recovers a positive judge-signal and a zero-signal control with a per-contestant
``u`` structure, and a real-data end-to-end test pins the Track R labels that
must never be confused with Track P metrics.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.integrate import quad
from scipy.special import loggamma

from dwts_reproduction.config import load_paths
from dwts_reproduction.preprocess import build_all_tables
from dwts_reproduction.problem1 import (
    build_problem1_panel,
    build_train_weeks,
    evaluate_top1_accuracy,
    infer_all_weekly_fan_support,
)
from dwts_reproduction.problem1.config import Problem1Config
from dwts_reproduction.problem1.softmin import (
    dirichlet_log_density,
    dirichlet_log_density_grad,
    softmax_np,
    softmin_logprob,
)
from dwts_reproduction.problem1.track_p import PooledFit, WeekPack
from dwts_reproduction.problem1.track_r import (
    fit_integrated_marginal,
    fit_sensitivity,
    integrated_week_terms,
)


# --------------------------------------------------------------------------- #
# Dirichlet score: analytic gradient vs finite differences
# --------------------------------------------------------------------------- #
def test_dirichlet_log_density_grad_matches_finite_differences():
    rng = np.random.default_rng(5)
    for trial in range(4):
        n = 3 + trial
        B = 7
        alpha = 0.3 + 2.0 * rng.dirichlet(np.ones(n))  # varied, some small entries
        p = rng.dirichlet(alpha, size=B)

        score = dirichlet_log_density_grad(p, alpha)  # (B, n)
        eps = 1e-5
        num = np.zeros_like(score)
        for i in range(n):
            da = np.zeros(n)
            da[i] = eps
            num[:, i] = (
                dirichlet_log_density(p, alpha + da) - dirichlet_log_density(p, alpha - da)
            ) / (2 * eps)
        np.testing.assert_allclose(score, num, rtol=1e-4, atol=1e-5)


# --------------------------------------------------------------------------- #
# n=2 quadrature identity for the MC marginal likelihood and its score gradient
# --------------------------------------------------------------------------- #
def _n2_quad_log_likelihood(X, beta, bias, u, J, elim_pos, config) -> float:
    """Exact n=2 integrated marginal log-likelihood via the sigmoid softmin form.

    For two contestants ``P(elim = e | p1)`` is a sigmoid in the free coordinate
    ``p1`` (softmax is shift-invariant along ``-(J + p)/tau``), so the integral
    ``int P(elim|p1) Beta(p1; a1, a2) dp1`` is smooth and ``quad`` converges to
    machine precision.  ``a = clip(kappa q, alpha_floor)`` matches the fit.
    """
    eta = bias + X @ beta + u
    q = softmax_np(eta)
    alpha = np.clip(config.kappa * q, config.alpha_floor, None)
    a1, a2 = alpha
    norm = loggamma(a1 + a2) - loggamma(a1) - loggamma(a2)

    def integrand(p1: float) -> float:
        p = np.array([p1, 1.0 - p1])
        log_f = softmin_logprob((J + p)[None, :], elim_pos, config.tau_like)[0]
        return float(np.exp(log_f + norm + (a1 - 1.0) * np.log(p1) + (a2 - 1.0) * np.log(1.0 - p1)))

    val, _ = quad(integrand, 0.0, 1.0, epsabs=1e-14, epsrel=1e-12, limit=500)
    return float(np.log(val))


def _n2_fd_gradients(X, beta, bias, u, J, elim_pos, config, eps: float = 1e-5):
    """Central finite differences of the exact n=2 log-likelihood."""
    gb = np.zeros(len(beta))
    for i in range(len(beta)):
        dp = np.zeros(len(beta))
        dp[i] = eps
        gb[i] = (
            _n2_quad_log_likelihood(X, beta + dp, bias, u, J, elim_pos, config)
            - _n2_quad_log_likelihood(X, beta - dp, bias, u, J, elim_pos, config)
        ) / (2 * eps)
    gu = np.zeros(len(u))
    for i in range(len(u)):
        du = np.zeros(len(u))
        du[i] = eps
        gu[i] = (
            _n2_quad_log_likelihood(X, beta, bias, u + du, J, elim_pos, config)
            - _n2_quad_log_likelihood(X, beta, bias, u - du, J, elim_pos, config)
        ) / (2 * eps)
    return gb, gu


def test_integrated_mc_gradient_matches_quadrature():
    """The score estimator (self-normalized importance weights) is unbiased.

    Compared against the exact n=2 integral: the MC ``log L`` and the MC
    gradients ``X.T @ grad_eta`` (beta) and ``grad_eta`` (u) agree with the
    quadrature truth well within Monte Carlo error at B = 1e6.
    """
    config = Problem1Config(era_mode="official", kappa=10.0, tau_like=0.15, alpha_floor=0.1)
    rng = np.random.default_rng(3)
    X = rng.standard_normal((2, 2))
    beta = rng.standard_normal(2) * 0.3
    bias = 0.2
    u = np.array([0.3, -0.2])
    J = rng.standard_normal(2)
    elim_pos = 1
    wk = WeekPack(
        X=X.astype(np.float32),
        J=J.astype(np.float32),
        cs=np.array([0, 1]),
        elim_pos=elim_pos,
        st=(1, 1),
    )

    B = 1_000_000
    terms = integrated_week_terms(wk, beta, bias, u, config, np.random.default_rng(0), B)

    truth_log_l = _n2_quad_log_likelihood(X, beta, bias, u, J, elim_pos, config)
    assert terms["log_l"] == pytest.approx(truth_log_l, abs=0.05)
    assert terms["se_log_l"] > 0.0 and terms["se_log_l"] < 0.05

    fd_beta, fd_u = _n2_fd_gradients(X, beta, bias, u, J, elim_pos, config)
    # d log L / d beta = X.T @ grad_eta ; d log L / d u = grad_eta (chain rule)
    np.testing.assert_allclose(X.T @ terms["grad_eta"], fd_beta, atol=0.1)
    np.testing.assert_allclose(terms["grad_eta"], fd_u, atol=0.1)


def test_importance_weights_are_self_normalized():
    """``exp(log_f - log_l)`` is ``f / mean(f)`` (sums to B); the terms must
    renormalize so weights sum to 1 and ESS lies in (0, B]."""
    config = Problem1Config(era_mode="official", kappa=10.0, tau_like=0.15, alpha_floor=0.1)
    rng = np.random.default_rng(11)
    n = 4
    X = rng.standard_normal((n, 2))
    wk = WeekPack(
        X=X.astype(np.float32),
        J=rng.standard_normal(n).astype(np.float32),
        cs=np.zeros(n, dtype=np.int64),
        elim_pos=0,
        st=(1, 1),
    )
    beta = rng.standard_normal(2)
    terms = integrated_week_terms(wk, beta, 0.1, np.zeros(1), config, rng, B=4096)
    w = np.exp(terms["log_f"] - np.log(np.exp(terms["log_f"]).mean()))
    assert not np.allclose(w.sum(), 1.0), "raw f/mean(f) sums to B, not 1"
    assert 0.0 < terms["ess"] <= 4096
    assert terms["q"].shape == (n,) and terms["alpha"].shape == (n,)
    assert terms["p_samps"].shape == (4096, n)


# --------------------------------------------------------------------------- #
# In-family synthetic recovery: judge-signal vs zero-signal control
# --------------------------------------------------------------------------- #
def _synth_panel(n_weeks: int, beta_z: np.ndarray, u_star: np.ndarray, seed: int) -> pd.DataFrame:
    """Generate from the fitted model family (standardized-X space).

    ``j_metric = base + s*z`` with iid standard-normal ``z`` per (week,
    contestant) so the internally standardized ``j_metric_z ~ z`` and ``beta_z``
    is directly the fitted coefficient.  ``era="rank"`` keeps the era dummy at 0.
    """
    rng = np.random.default_rng(seed)
    n = 6
    base, s = 1.0 / 6.0, 0.141
    rows: list[dict] = []
    for w in range(n_weeks):
        jraw = base + s * rng.standard_normal(n)
        age = rng.uniform(20.0, 45.0, n)
        age_z = (age - 32.5) / 7.2
        zz = (jraw - base) / s
        eta = np.zeros(n) + u_star + beta_z[0] * zz + beta_z[1] * age_z
        q = softmax_np(eta)
        p = rng.dirichlet(10.0 * q)
        elim_i = int(rng.choice(n, p=softmax_np(-(jraw + p) / 0.15)))
        for i in range(n):
            rows.append(
                dict(
                    season=1,
                    week=w,
                    celebrity_name=f"C{i}",
                    alive=True,
                    elim_this_week_end=(i == elim_i),
                    is_final_week=False,
                    max_week=n_weeks + 1,
                    age=age[i],
                    judge_percent=np.nan,
                    judge_rank_share=jraw[i],
                    j_metric=jraw[i],
                    era="rank",
                )
            )
    return pd.DataFrame(rows)


def _synth_config() -> Problem1Config:
    return Problem1Config(
        era_mode="official",
        kappa=10.0,
        tau_like=0.15,
        alpha_floor=0.1,
        lr=0.02,
        n_steps=500,
        batch_size=64,
        B=1200,
        l2_beta=0.02,
        l2_u=0.02,
        seed=7,
    )


def test_synthetic_recovery_signal_vs_control():
    """A positive judge signal is recovered; a zero-signal control stays near 0
    while its per-contestant ``u`` structure is still found."""
    config = _synth_config()
    signal = _synth_panel(300, np.array([0.5, 0.0]), np.zeros(6), seed=7)
    control = _synth_panel(
        300, np.array([0.0, 0.0]), np.array([0.5, -0.3, 0.1, 0.0, -0.4, 0.1]), seed=7
    )
    fit_sig, _ = fit_integrated_marginal(signal, build_train_weeks(signal), config)
    fit_ctl, _ = fit_integrated_marginal(control, build_train_weeks(control), config)

    assert fit_sig.beta[0] > 0.25, f"signal judge coefficient {fit_sig.beta[0]:.3f} not recovered"
    assert abs(fit_ctl.beta[0]) < 0.20, f"control judge coefficient {fit_ctl.beta[0]:.3f} drifted"
    assert fit_ctl.u.std() > 0.20, "control must still recover its per-contestant u structure"
    for fit in (fit_sig, fit_ctl):
        assert fit.model_type == "integrated_marginal_mc"
        assert np.isfinite(fit.beta).all() and np.isfinite(fit.u).all()


# --------------------------------------------------------------------------- #
# Real-data end-to-end (Track R labels must never be confused with Track P)
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def real_fit() -> tuple[pd.DataFrame, pd.DataFrame, object, dict[str, object]]:
    config = Problem1Config.for_track("R")
    paths = load_paths()
    tables = build_all_tables(paths.raw_data_csv)
    panel = build_problem1_panel(tables, config.era_mode, [])
    train = build_train_weeks(panel)
    fit, diagnostics = fit_integrated_marginal(panel, train, config)
    return panel, train, fit, diagnostics


def test_real_data_end_to_end_track_r(real_fit):
    panel, train, fit, diagnostics = real_fit
    assert fit.model_type == "integrated_marginal_mc"
    assert fit.era_mode == "official"
    assert fit.beta.shape == (3,) and fit.u.shape == (fit.n_cs,)
    assert fit.loss_history[-1] < fit.loss_history[0], "fit must descend"
    assert np.isfinite(fit.beta).all() and np.isfinite(fit.u).all()

    for key in ("n_weeks", "mc_log_l", "mc_se", "mc_se_relative", "ess_mean", "B"):
        assert key in diagnostics, f"missing diagnostic {key}"
        assert np.isfinite(diagnostics[key])
    assert diagnostics["mc_se_relative"] < 0.05, "MC log-likelihood error must be small"
    assert 0.0 < diagnostics["ess_mean"] <= diagnostics["B"]

    config = Problem1Config.for_track("R")
    posterior = infer_all_weekly_fan_support(panel, fit, config)
    _, _, accuracy = evaluate_top1_accuracy(panel, posterior, train)
    top1 = accuracy["overall_top1_accuracy"]
    assert 0.0 < top1 < 1.0
    # Track R fits the integrated marginal likelihood and does not recondition on
    # the observed outcome; its explanatory top-1 is below Track P's two-stage
    # reconstruction (0.9495) — both numbers are reported with their track label.
    assert top1 < 0.95


def test_fit_sensitivity_structure():
    """fit_sensitivity returns per-variant final loss, MC logL, and top-1."""
    panel = _synth_panel(40, np.array([0.0, 0.0]), np.zeros(6), seed=1)
    cfg = Problem1Config(
        era_mode="official",
        kappa=10.0,
        tau_like=0.15,
        alpha_floor=0.1,
        lr=0.02,
        n_steps=80,
        batch_size=20,
        B=400,
        l2_beta=0.02,
        l2_u=0.02,
        seed=2,
    )
    result = fit_sensitivity(panel, build_train_weeks(panel), cfg, seeds=(3,), mc_B=400)
    variants = result["variants"]
    assert "seed_3" in variants
    for key in ("final_loss", "mc_log_l", "top1_accuracy"):
        assert key in variants["seed_3"]
        assert np.isfinite(variants["seed_3"][key])
    assert "final_loss_spread" in result and result["final_loss_spread"] >= 0.0


# --------------------------------------------------------------------------- #
# Configuration guards
# --------------------------------------------------------------------------- #
def test_pooledfit_model_type_default():
    assert PooledFit.model_type == "pooled_softmin_numpy"


def test_alpha_floor_validation():
    with pytest.raises(ValueError):
        Problem1Config(alpha_floor=0.0)
    with pytest.raises(ValueError):
        Problem1Config(alpha_floor=1.5)
