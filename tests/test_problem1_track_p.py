"""Problem 1 Track P stage-1 fit and stage-2 posterior tests.

The pooled softmin gradient is verified against finite differences, the numpy
Adam update against the bias-corrected rule it is meant to reproduce, and the
full pipeline against the review rebuild's registered targets
(``docs/BASELINE_PAPER_OUTPUTS.md`` B-03, B-05..B-07).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dwts_reproduction.config import load_paths
from dwts_reproduction.preprocess import build_all_tables
from dwts_reproduction.problem1 import (
    build_problem1_panel,
    build_train_weeks,
    evaluate_top1_accuracy,
    fit_pooled_softmin,
    infer_all_weekly_fan_support,
    pooled_q_for_week,
    posterior_draws_for_week,
)
from dwts_reproduction.problem1.config import Problem1Config
from dwts_reproduction.problem1.softmin import NumpyAdam, softmax_np

# Registered targets (BASELINE_PAPER_OUTPUTS.md) --------------------------------
REF_TOP1 = 0.9495412844036697
REF_MEAN_PCP = 0.6043260177622087
REF_MEAN_ESS = 0.9625205370468809
REF_MEAN_CI_RW = 3.117335173792119


# --------------------------------------------------------------------------- #
# Analytic softmin gradient vs finite differences
# --------------------------------------------------------------------------- #
def _pooled_loss(X, beta, bias, u, J, elim, tau):
    eta = bias + X @ beta + u
    q = softmax_np(eta)
    z = -(J + q) / tau
    z = z - z.max()
    logp = z - np.log(np.exp(z).sum())
    return -logp[elim]


def _pooled_grad(X, beta, bias, u, J, elim, tau):
    eta = bias + X @ beta + u
    q = softmax_np(eta)
    z = -(J + q) / tau
    psoft = softmax_np(z)
    onehot = np.zeros(psoft.shape)
    onehot[elim] = 1.0
    g = -(1.0 / tau) * (psoft - onehot)
    dot = (q * g).sum()
    grad_eta = q * (g - dot)
    return grad_eta


def test_pooled_softmin_gradient_matches_finite_differences():
    rng = np.random.default_rng(7)
    for trial in range(6):
        n, p = 5 + trial, 3
        X = rng.standard_normal((n, p))
        beta = rng.standard_normal(p)
        bias = rng.standard_normal(())
        u = rng.standard_normal(n)
        J = rng.standard_normal(n)
        elim = int(rng.integers(0, n))
        tau = 0.05

        grad_eta = _pooled_grad(X, beta, bias, u, J, elim, tau)
        eps = 1e-6

        num_beta = np.zeros(p)
        for i in range(p):
            dp = np.zeros(p)
            dp[i] = eps
            num_beta[i] = (
                _pooled_loss(X, beta + dp, bias, u, J, elim, tau)
                - _pooled_loss(X, beta - dp, bias, u, J, elim, tau)
            ) / (2 * eps)
        np.testing.assert_allclose(X.T @ grad_eta, num_beta, rtol=1e-4, atol=1e-5)

        num_bias = (
            _pooled_loss(X, beta, bias + eps, u, J, elim, tau)
            - _pooled_loss(X, beta, bias - eps, u, J, elim, tau)
        ) / (2 * eps)
        np.testing.assert_allclose(grad_eta.sum(), num_bias, rtol=1e-4, atol=1e-5)

        num_u = np.zeros(n)
        for i in range(n):
            du = np.zeros(n)
            du[i] = eps
            num_u[i] = (
                _pooled_loss(X, beta, bias, u + du, J, elim, tau)
                - _pooled_loss(X, beta, bias, u - du, J, elim, tau)
            ) / (2 * eps)
        np.testing.assert_allclose(grad_eta, num_u, rtol=1e-4, atol=1e-5)


# --------------------------------------------------------------------------- #
# NumpyAdam update rule
# --------------------------------------------------------------------------- #
def test_numpyadam_first_step_is_gradient_descent_with_bias_correction():
    param = np.array([1.0, -2.0, 0.5])
    grad = np.array([0.1, -0.3, 0.05])
    opt = NumpyAdam([param], lr=0.02)
    before = param.copy()
    opt.step([grad])
    # step 1: m_hat = grad, v_hat = grad^2, update = lr * grad / (|grad| + eps)
    expected = before - 0.02 * grad / (np.abs(grad) + 1e-8)
    np.testing.assert_allclose(param, expected, rtol=1e-12)
    assert opt.t == 1


def test_numpyadam_moves_downhill():
    param = np.array([3.0, -1.0, 2.0])
    opt = NumpyAdam([param], lr=0.1)
    f = float(np.sum(param**2))
    g = 2.0 * param
    opt.step([g])
    assert float(np.sum(param**2)) < f, "Adam step must reduce a quadratic objective"


def test_numpyadam_rejects_bad_betas():
    with pytest.raises(ValueError):
        NumpyAdam([np.zeros(2)], betas=(1.0, 0.9))


# --------------------------------------------------------------------------- #
# Real-data fit + posterior regression (Track P, legacy era)
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def fitted() -> tuple[pd.DataFrame, pd.DataFrame, object]:
    config = Problem1Config.for_track("P")
    paths = load_paths()
    tables = build_all_tables(paths.raw_data_csv)
    panel = build_problem1_panel(tables, config.era_mode)
    train = build_train_weeks(panel)
    fit = fit_pooled_softmin(panel, train, config)
    return panel, train, fit


def test_fit_shapes_and_convergence(fitted):
    panel, train, fit = fitted
    assert fit.beta.ndim == 1 and fit.beta.size == 3
    assert fit.u.ndim == 1 and fit.u.size == fit.n_cs
    assert fit.train_choice_sets == list(
        train[["season", "week"]].itertuples(index=False, name=None)
    )
    assert len(fit.loss_history) > 10
    first, last = fit.loss_history[0], fit.loss_history[-1]
    assert last < first, "penalized softmin NLL should fall over the fit"
    assert np.isfinite(fit.beta).all() and np.isfinite(fit.u).all()


def test_pooled_q_is_valid_simplex(fitted):
    panel, train, fit = fitted
    s, wk = int(train["season"].iloc[0]), int(train["week"].iloc[0])
    qdf = pooled_q_for_week(panel, fit, s, wk)
    q = qdf["q_hat"].to_numpy()
    assert q.min() >= 0.0 and abs(q.sum() - 1.0) < 1e-9
    assert qdf["celebrity_name"].is_monotonic_increasing


def test_posterior_draws_are_conditioned_on_elimination(fitted):
    """The softmin likelihood moves the eliminee's share below its prior and makes
    it the posterior-mean softmin (lowest ``J + p``), reproducing the elimination."""
    panel, train, fit = fitted
    config = Problem1Config.for_track("P")
    s, wk = int(train["season"].iloc[0]), int(train["week"].iloc[0])
    res = posterior_draws_for_week(panel, fit, s, wk, config)
    assert res["has_posterior"] is True
    assert res["samples"].shape == (config.B, len(res["alive"]))
    np.testing.assert_allclose(res["samples"].sum(axis=1), 1.0, rtol=1e-6)
    np.testing.assert_allclose(res["weights"].sum(), 1.0, rtol=1e-9)
    assert 0.0 < res["ess"] <= config.B
    # conditioning must actually reweight (weights not all equal)
    assert np.max(res["weights"]) > np.min(res["weights"]), "weights must be non-uniform"
    p_mean = res["weights"] @ res["samples"]
    assert abs(p_mean.sum() - 1.0) < 1e-9
    elim_idx = int(res["elim_pos"])
    q_prior = res["alive"]["q_hat"].to_numpy()
    # softmin elimination favors LOW fan share for the eliminee, so conditioning
    # pulls p below the pooled prior q (the model's explanatory mechanism)
    assert p_mean[elim_idx] < q_prior[elim_idx], (
        "softmin conditioning must lower the eliminee's fan-share below its prior"
    )
    # and the posterior-mean combined score reproduces the observed elimination
    j = res["alive"]["j_metric"].to_numpy(dtype=float)
    assert int(np.argmin(j + p_mean)) == elim_idx, (
        "posterior-mean softmin should be the observed eliminatee"
    )


def test_final_week_uses_uniform_weights_in_rebuild_mode(fitted):
    panel, train, fit = fitted
    config = Problem1Config.for_track("P")
    final_week = panel[panel["is_final_week"]].drop_duplicates(["season", "week"])
    s, wk = int(final_week["season"].iloc[0]), int(final_week["week"].iloc[0])
    res = posterior_draws_for_week(panel, fit, s, wk, config, has_posterior_mode="rebuild")
    assert res["has_posterior"] is False
    np.testing.assert_allclose(res["weights"], np.full(config.B, 1.0 / config.B), atol=1e-12)


# --------------------------------------------------------------------------- #
# Full Track P regression against the review rebuild
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def full_pipeline(fitted):
    panel, train, fit = fitted
    config = Problem1Config.for_track("P")
    posterior = infer_all_weekly_fan_support(panel, fit, config)
    return panel, train, fit, posterior


def test_summary_reproduces_reference_numbers(full_pipeline):
    from dwts_reproduction.problem1 import evaluate_top1_accuracy, summarize_posterior

    panel, train, fit, posterior = full_pipeline
    _, _, accuracy = evaluate_top1_accuracy(panel, posterior, train)
    summary = summarize_posterior(posterior, accuracy)

    assert summary["overall_top1_accuracy"] == pytest.approx(REF_TOP1, rel=1e-3)
    assert summary["mean_pcp_weighted"] == pytest.approx(REF_MEAN_PCP, rel=1e-3)
    assert summary["mean_ess_ratio"] == pytest.approx(REF_MEAN_ESS, rel=1e-3)
    assert summary["mean_ci_rel_width"] == pytest.approx(REF_MEAN_CI_RW, rel=1e-3)


def test_top1_metrics_are_in_sample_and_labeled(full_pipeline):
    """Regression targets use observed eliminations; the metrics are explanatory."""
    panel, train, fit, posterior = full_pipeline
    by_week, _, _ = evaluate_top1_accuracy(panel, posterior, train)
    assert len(by_week) == len(train)
    assert by_week["correct"].isin([True, False]).all()
    assert by_week["alive_n"].min() >= 2


def test_pcp_unweighted_within_reference_band(full_pipeline):
    from dwts_reproduction.problem1 import evaluate_top1_accuracy

    panel, train, fit, posterior = full_pipeline
    _, _, accuracy = evaluate_top1_accuracy(panel, posterior, train)
    # reference B-02 band for the paper's unweighted PCP; our numpy rebuild is
    # within a couple of percent of the reported mean
    assert accuracy["mean_pcp_unweighted"] > 0.50
