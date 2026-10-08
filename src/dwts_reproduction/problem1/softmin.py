"""Shared numerical utilities for the Problem 1 models.

Everything here is pure numpy with no stochastic state.  The two softmin
likelihoods (tracking elimination risk), the weighted quantile used for credible
intervals, the Dirichlet density helpers used by the Track R marginal likelihood,
and the hand-written Adam optimizer (matching ``torch.optim.Adam`` defaults so the
reference fit is reproducible without a torch dependency) live in this module.
"""

from __future__ import annotations

import numpy as np


# --------------------------------------------------------------------------- #
# Softmax / softmin likelihood
# --------------------------------------------------------------------------- #
def logsumexp(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable ``log(sum(exp(x)))`` along ``axis``."""
    a = np.asarray(x, dtype=float)
    m = np.max(a, axis=axis, keepdims=True)
    return np.asarray(m.squeeze(axis) + np.log(np.exp(a - m).sum(axis=axis)))


def softmax_np(x: np.ndarray) -> np.ndarray:
    """Softmax over the last axis; stable via max-shift.  Result sums to 1."""
    z = np.asarray(x, dtype=float)
    z = z - np.max(z)
    e = np.exp(z)
    return np.asarray(e / e.sum())


def softmin_logprob(cost: np.ndarray, elim_pos: int, tau: float) -> np.ndarray:
    """Log-probability that position ``elim_pos`` is the softmin of each row of ``cost``.

    Args:
        cost: ``(B, n)`` combined scores ``J + p``.
        elim_pos: Column index of the observed eliminatee.
        tau: Softmin temperature.
    """
    z = -np.asarray(cost, dtype=float) / tau
    z = z - z.max(axis=1, keepdims=True)
    return np.asarray(z[:, elim_pos] - np.log(np.exp(z).sum(axis=1)))


def log_mean_exp(x: np.ndarray) -> np.ndarray:
    """Stable ``log(mean(exp(x)))`` along the last axis (base-``e``)."""
    a = np.asarray(x, dtype=float)
    m = np.max(a, axis=-1)
    return np.asarray(m + np.log(np.exp(a - m[..., None]).mean(axis=-1)))


# --------------------------------------------------------------------------- #
# Weighted quantiles (posterior credible intervals)
# --------------------------------------------------------------------------- #
def weighted_quantile(
    values: np.ndarray, quantiles: float | list[float] | np.ndarray, weights: np.ndarray
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


# --------------------------------------------------------------------------- #
# Dirichlet density helpers (Track R marginal likelihood)
# --------------------------------------------------------------------------- #
def dirichlet_log_density(p: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """Row-wise log density of ``Dirichlet(alpha)`` at each row of ``p``.

    Args:
        p: ``(B, n)`` simplex rows.
        alpha: ``(n,)`` positive concentration vector (broadcast per row).
    """
    a = np.asarray(alpha, dtype=float)
    x = np.asarray(p, dtype=float)
    return float(_lgamma(a.sum()) - _lgamma(a).sum()) + ((a - 1.0) * np.log(x)).sum(axis=1)


def dirichlet_log_density_grad(p: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """``d/dalpha log Dirichlet(p | alpha)`` as a ``(B, n)`` matrix.

    The gradient is ``psi(alpha0) - psi(alpha_i) + log(p_i)`` per row, where
    ``psi`` is the digamma function.  This is the score of the Dirichlet in the
    Fisher-identity gradient used by the Track R fit.

    A draw component that the RNG rounds to exact ``0.0`` (gamma underflow under
    a low concentration) would otherwise feed ``log 0 = -inf`` into the score;
    it is clamped to the smallest positive float so the gradient stays finite
    (the Track R fit additionally floors ``alpha`` to avoid such draws).
    """
    a = np.asarray(alpha, dtype=float)
    x = np.asarray(p, dtype=float)
    from scipy.special import digamma  # analysis extra; see pyproject.toml

    return np.asarray(
        digamma(a.sum()) - digamma(a)[None, :] + np.log(np.maximum(x, np.finfo(float).tiny))
    )


def _lgamma(x: np.ndarray) -> np.ndarray:
    from scipy.special import gammaln

    return np.asarray(gammaln(np.asarray(x, dtype=float)))


# --------------------------------------------------------------------------- #
# Hand-written Adam optimizer
# --------------------------------------------------------------------------- #
class NumpyAdam:
    """Adam optimizer matching ``torch.optim.Adam`` default hyperparameters.

    The reference rebuild fits the pooled softmin model with ``torch.optim.Adam``
    (lr 0.02, betas (0.9, 0.999), eps 1e-8).  Torch is intentionally not a
    dependency of this repository, so this class reproduces the same update rule
    with plain numpy.
    """

    def __init__(
        self,
        params: list[np.ndarray],
        lr: float = 0.020,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
    ) -> None:
        if not 0.0 <= betas[0] < 1.0 or not 0.0 <= betas[1] < 1.0:
            raise ValueError("Adam betas must be in [0, 1).")
        self.params = params
        self.lr = float(lr)
        self.b1, self.b2 = float(betas[0]), float(betas[1])
        self.eps = float(eps)
        self.t = 0
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]

    def step(self, grads: list[np.ndarray]) -> None:
        """Apply one Adam update using the given gradients (same order as ``params``)."""
        if len(grads) != len(self.params):
            raise ValueError("grads must match params length")
        self.t += 1
        b1t = 1.0 - self.b1**self.t
        b2t = 1.0 - self.b2**self.t
        for param, grad, m, v in zip(self.params, grads, self.m, self.v, strict=True):
            m *= self.b1
            m += (1.0 - self.b1) * grad
            v *= self.b2
            v += (1.0 - self.b2) * grad * grad
            m_hat = m / b1t
            v_hat = v / b2t
            param -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
