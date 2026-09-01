"""Problem 3 professional-partner effects (P-062..P-066).

The paper defines two leakage-safe partner traits over PRIOR seasons only:

- ``H_abil``: prior-season ability  = ``pro_hist_mean_placez`` (expanding mean of
  placement_z over seasons < t)
- ``H_exp``:  prior-season tenure    = ``pro_hist_n_prev`` (prior-season count)

Reproduced artifacts:

- P-062/P-063  Eq. (partner)/Eq. (fe_model): ``Z = mu + gamma1 H_abil
  + gamma2 H_exp + alpha_p + eps`` with partner fixed effects, one fit per
  outcome (robust HC3 SEs).  The paper reports ``r(H_exp, judge) = 0.23*`` for
  this setup (P-064).
- P-064  Pearson ``r`` (and p-value) between each trait and each standardized
  judge/fan outcome, at the observation (season-celebrity) level.
- P-065  Partner-trait correlation heatmap source table.
- P-066  Partner-heterogeneity source tables: per-partner raw ability
  (``H_abil`` mean) and judge-outcome partner fixed effects.

There is no legacy producer for this analysis in ``../src/`` (confirmed by a
full-source search); the reproduction is therefore from the paper's formulas on
the registered ``data/data_3.csv`` input (see D-20260901-17).
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import pearsonr

from .regression import FAN_OUTCOMES, JUDGE_OUTCOMES, OUTCOMES

# Robust SE convention for the partner-FE fits.  HC1 (the classic White
# estimator, finite) is used instead of the HC3 used by the demographic OLS
# pipeline because partners with a single appearance have leverage h=1, which
# makes HC3's (1-h)^-2 scaling degenerate (SEs -> inf).  Coefficients are
# identical under either convention; only the SEs differ (D-20260901-17).
ROBUST_SE = "HC1"


def partner_fe_regressions(
    df: pd.DataFrame, outcomes: tuple[str, ...] | list[str] = JUDGE_OUTCOMES
) -> pd.DataFrame:
    """Fit Eq. (fe_model) per outcome: ``Z ~ H_abil + H_exp + C(ballroom_partner)``.

    Returns a tidy table with one row per (outcome, term) for the trait terms and
    per-outcome model statistics.  The partner fixed effects themselves are
    returned separately by :func:`partner_fe_params`.
    """
    rows: list[dict[str, Any]] = []
    for key in outcomes:
        y = OUTCOMES[key]
        d = df.dropna(subset=[y, "pro_hist_mean_placez", "pro_hist_n_prev"]).copy()
        if len(d) < 30:
            continue
        m = smf.ols(
            f"{y} ~ pro_hist_mean_placez + pro_hist_n_prev + C(ballroom_partner)", data=d
        ).fit(cov_type=ROBUST_SE)
        for term in ("pro_hist_mean_placez", "pro_hist_n_prev"):
            rows.append(
                {
                    "outcome": key,
                    "trait": "H_abil" if term == "pro_hist_mean_placez" else "H_exp",
                    "coef": float(m.params[term]),
                    "se": float(m.bse[term]),
                    "p": float(m.pvalues[term]),
                    "n": int(len(d)),
                    "r2": float(m.rsquared),
                    "n_partners": int(d["ballroom_partner"].nunique()),
                }
            )
    return pd.DataFrame(rows).sort_values(["outcome", "trait"]).reset_index(drop=True)


def partner_fe_params(df: pd.DataFrame, outcome: str = "judge_w1") -> pd.DataFrame:
    """Extract per-partner fixed effects ``alpha_p`` for one outcome (P-066).

    Fits ``Z ~ H_abil + H_exp + C(ballroom_partner)`` and returns each partner's
    absolute FE (reference partner gets the intercept; others get intercept plus
    their dummy).  Also carries each partner's per-row ability and tenure means.
    """
    y = OUTCOMES[outcome]
    d = df.dropna(subset=[y, "pro_hist_mean_placez", "pro_hist_n_prev"]).copy()
    m = smf.ols(f"{y} ~ pro_hist_mean_placez + pro_hist_n_prev + C(ballroom_partner)", data=d).fit(
        cov_type=ROBUST_SE
    )
    params = m.params
    intercept = float(params["Intercept"])
    fe_by_partner: dict[str, float] = {}
    for term, coef in params.items():
        if term.startswith("C(ballroom_partner)[T."):
            name = re.match(r"C\(ballroom_partner\)\[T\.(.*)\]$", term)
            if name:
                fe_by_partner[name.group(1)] = intercept + float(coef)
    fe_by_partner.setdefault(str(d["ballroom_partner"].iloc[0]), intercept)  # reference partner

    traits = d.groupby("ballroom_partner", as_index=False).agg(
        n=("ballroom_partner", "size"),
        H_abil_mean=("pro_hist_mean_placez", "mean"),
        H_exp_mean=("pro_hist_n_prev", "mean"),
        outcome_mean=(y, "mean"),
    )
    traits["alpha_p"] = traits["ballroom_partner"].map(fe_by_partner)
    return traits.sort_values("alpha_p", ascending=False).reset_index(drop=True)


def partner_trait_correlations(df: pd.DataFrame) -> pd.DataFrame:
    """Row-level Pearson ``r`` of each partner trait with each outcome (P-064).

    ``H_abil`` and ``H_exp`` are the leakage-safe features already present on
    ``df`` (rookies carry ``0``, as in the legacy port).  Returns
    ``(trait, outcome, r, p, n)`` rows.
    """
    traits = ("H_abil", "H_exp")
    trait_col = {"H_abil": "pro_hist_mean_placez", "H_exp": "pro_hist_n_prev"}
    rows: list[dict[str, Any]] = []
    for key, y in OUTCOMES.items():
        if key == "placement_z":
            continue
        d = df.dropna(subset=[y]).copy()
        for trait in traits:
            x = d[trait_col[trait]].to_numpy(dtype=float)
            yy = d[y].to_numpy(dtype=float)
            mask = np.isfinite(x) & np.isfinite(yy)
            if mask.sum() < 5:
                continue
            r, p = pearsonr(x[mask], yy[mask])
            rows.append(
                {
                    "trait": trait,
                    "outcome": key,
                    "r": float(r),
                    "p": float(p),
                    "n": int(mask.sum()),
                }
            )
    return pd.DataFrame(rows).sort_values(["trait", "outcome"]).reset_index(drop=True)


def partner_tenure_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Per-outcome summary of the H_exp correlation claim (P-064 narrative)."""
    corr = partner_trait_correlations(df)
    return (
        corr[corr["trait"].eq("H_exp")]
        .assign(
            significant=lambda d: d["p"].lt(0.05),
            sign=lambda d: np.sign(d["r"]),
        )
        .sort_values("outcome")
        .reset_index(drop=True)
    )


def judge_fan_supporting(
    df: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """Bundle the partner-source tables needed by the P-065/P-066 figures.

    Returns ``{"corr": ..., "fe_params": ..., "traits": ...}`` where ``corr`` is
    the P-064 correlation table and ``fe_params`` is the per-partner FE table
    (P-066) for the primary judge outcome.
    """
    return {
        "corr": partner_trait_correlations(df),
        "fe_params": partner_fe_params(df, "judge_w1"),
        "fe_summary": partner_fe_regressions(df, JUDGE_OUTCOMES + FAN_OUTCOMES),
    }
