"""Structural problem-1 analyses: PCP-vs-alive-set size and the ranking gap.

Two paper claims are reproduced here from *saved* tables (never from the raw
draws, which are not persisted):

- ``crowded_field_from_posterior`` (P-033): per-season-week PCP against the
  alive-set size ``|A_{s,t}|``.  The legacy notebook computed PCP by weighted
  importance sampling with ``kappa=30``/``B=2500``; the repository's saved
  ``posterior_summary`` carries both ``pcp_weighted`` and ``pcp_unweighted`` at
  ``kappa=10``/``B=1200`` (registered config).  The figure therefore reports both
  PCP variants and records the parameter discrepancy in D-20260901-14.  PCP is a
  posterior-mean reconstruction metric, not a claim about latent fan votes.

- ``ranking_gap_frame`` + ``quadratic_fit_with_ci`` (P-035): the paper's
  "inferred fan-vote ranking vs. final placement - average judge ranking" gap
  figure (paper claim ``R^2 > 0.6``).  This is an exact port of notebook cell 56:
  ``placement`` (final placement, first row per contestant-season), the per-
  contestant mean ``judge_rank`` over alive weeks, and the per-contestant mean
  ``p_mean`` from the saved posterior summary.  The quadratic regression and its
  95% CI are computed on the **un-jittered** data; jittering (``0.15``) is a
  plot-only cosmetic, applied downstream in the figure module, never here.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# P-033: PCP vs alive-set size
# --------------------------------------------------------------------------- #
def crowded_field_from_posterior(posterior_summary: pd.DataFrame) -> pd.DataFrame:
    """Collapse the per-contestant posterior summary into one row per season-week.

    PCP / alive size are per-week quantities repeated on every alive contestant's
    row; take the first row's value per ``(season, week)``.  ``era`` is taken from
    that row too (constant within a season-week).  Only weeks with a posterior
    reweighting keep a finite PCP; the column stays ``NaN`` elsewhere.
    """
    cols = ["season", "week", "era", "alive_n", "pcp_weighted", "pcp_unweighted", "has_posterior"]
    g = (
        posterior_summary.sort_values(["season", "week", "celebrity_name"])
        .groupby(["season", "week"], as_index=False)[cols]
        .first()
        .sort_values(["season", "week"])
        .reset_index(drop=True)
    )
    return g


# --------------------------------------------------------------------------- #
# P-035: ranking gap
# --------------------------------------------------------------------------- #
def ranking_gap_frame(weekly: pd.DataFrame, posterior_summary: pd.DataFrame) -> pd.DataFrame:
    """Build the ``(placement - mean judge rank)`` vs ``inferred fan rank`` frame.

    Exact port of notebook cell 56.  ``weekly`` is the repository's weekly table
    (placement, judge_rank, judge_percent per contestant-season-week — the exact
    equivalent of the legacy ``df_weekly``); ``posterior_summary`` is the saved
    Track P posterior mean table.  All three inputs are dropped on their own NaN
    before grouping, exactly as the notebook did, and the merge is inner.
    """
    placement_df = (
        weekly.dropna(subset=["placement"])
        .sort_values(["season", "celebrity_name", "week"])
        .groupby(["season", "celebrity_name"], as_index=False)
        .first()[["season", "celebrity_name", "placement"]]
    )
    judge_avg = (
        weekly.dropna(subset=["judge_rank"])
        .groupby(["season", "celebrity_name"])["judge_rank"]
        .mean()
        .rename("judge_avg_rank")
        .reset_index()
    )
    audience_mean = (
        posterior_summary.dropna(subset=["p_mean"])
        .groupby(["season", "celebrity_name"])["p_mean"]
        .mean()
        .rename("audience_mean")
        .reset_index()
    )

    frame = placement_df.merge(judge_avg, on=["season", "celebrity_name"], how="inner")
    frame = frame.merge(audience_mean, on=["season", "celebrity_name"], how="inner")
    frame["result_minus_judge"] = frame["placement"] - frame["judge_avg_rank"]
    frame["audience_rank"] = frame.groupby("season")["audience_mean"].rank(
        ascending=False, method="average"
    )
    return frame.sort_values(["season", "result_minus_judge"]).reset_index(drop=True)


@dataclass
class QuadFit:
    """Polynomial fit with pointwise 95% confidence band (un-jittered data)."""

    order: int
    coeffs: np.ndarray
    cov: np.ndarray
    r_squared: float
    n: int
    x_grid: np.ndarray
    y_fit: np.ndarray
    ci_lo: np.ndarray
    ci_hi: np.ndarray


def quadratic_fit_with_ci(x: np.ndarray, y: np.ndarray, order: int = 2) -> QuadFit:
    """Fit a polynomial by least squares and return ``R^2`` plus a pointwise 95% CI.

    The CI follows the standard covariance propagation used by ``seaborn.regplot``
    (``polyfit(cov=True)``): ``CI = fit ± 1.96 * sqrt(diag(V @ cov @ V.T))`` where
    ``V`` is the polynomial-basis (Vandermonde) design matrix.  The fit is
    computed on the un-jittered data; any plot-time jitter must be applied only to
    the scatter, never to ``x``/``y`` here (D-20260901-15).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    n = int(x.size)
    if n <= order:
        raise ValueError(f"Need more than {order} finite points for a degree-{order} fit; got {n}.")

    coeffs, cov = np.polyfit(x, y, order, cov=True)
    y_fit = np.polyval(coeffs, x)

    ss_res = float(np.sum((y - y_fit) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    x_grid = np.linspace(float(x.min()), float(x.max()), 200)
    vdm = np.vander(x_grid, order + 1, increasing=True)  # rows x [1, x, x^2, ...]
    std = np.sqrt(np.clip(np.einsum("ij,jk,ik->i", vdm, cov, vdm), 0, None))
    band = 1.96 * std

    return QuadFit(
        order=order,
        coeffs=np.asarray(coeffs, dtype=float),
        cov=np.asarray(cov, dtype=float),
        r_squared=float(r_squared),
        n=n,
        x_grid=x_grid,
        y_fit=np.asarray(np.polyval(coeffs, x_grid), dtype=float),
        ci_lo=np.asarray(np.polyval(coeffs, x_grid) - band, dtype=float),
        ci_hi=np.asarray(np.polyval(coeffs, x_grid) + band, dtype=float),
    )
