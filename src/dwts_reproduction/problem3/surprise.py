"""Surprise and fan-support growth with HC3 regression inference.

Surprise is the current judge signal minus the previous fan signal; growth
is the change in the fan signal. The primary model compares weeks 1 and 6.
The late-stage variant combines judge week 11, fan week 6 and final fan signal.
The quadratic specification includes surprise squared and its interaction
with professional-partner experience."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
import statsmodels.formula.api as smf

# Primary (t = Week 6) and late-stage (t = final) constructions.  ``fan_t_col``
# is the Z_Fan,t signal, ``fan_prev_col`` the Z_Fan,t-1 baseline.
PRIMARY_TW6: dict[str, str] = {
    "judge_col": "week6_judge_score_placement_z",
    "fan_t_col": "week6_p_score_placement_z",
    "fan_prev_col": "week1_p_score_placement_z",
    "label": "t=W6",
}
LATE_TFINAL: dict[str, str] = {
    "judge_col": "week11_judge_score_placement_z",
    "fan_t_col": "week_final_p_score_placement_z",
    "fan_prev_col": "week6_p_score_placement_z",
    "label": "t=final",
}


@dataclass
class FitSummary:
    """Compact OLS fit summary used by the figures and claim checks."""

    spec: str
    n: int
    r2: float
    terms: list[str]
    coefs: dict[str, float]
    pvalues: dict[str, float]
    stderr: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "spec": self.spec,
            "n": int(self.n),
            "r2": float(self.r2),
            "terms": self.terms,
            "coefs": {k: float(v) for k, v in self.coefs.items()},
            "pvalues": {k: float(v) for k, v in self.pvalues.items()},
            "stderr": {k: float(v) for k, v in self.stderr.items()},
        }


def surprise_growth_frame(
    df: pd.DataFrame,
    *,
    judge_col: str,
    fan_t_col: str,
    fan_prev_col: str,
    label: str,
) -> pd.DataFrame:
    """Build the (S, G, H_exp, industry) analysis frame for one construction.

    S = judge_z - fan_prev_z; G = fan_t_z - fan_prev_z; NaN rows dropped.
    """
    out = pd.DataFrame(
        {
            "season": df["season"],
            "celebrity_name": df["celebrity_name"],
            "industry_grp": df["industry_grp"],
            "ballroom_partner": df["ballroom_partner"],
            "H_exp": pd.to_numeric(df["pro_hist_n_prev"], errors="coerce"),
            "S": pd.to_numeric(df[judge_col], errors="coerce")
            - pd.to_numeric(df[fan_prev_col], errors="coerce"),
            "G": pd.to_numeric(df[fan_t_col], errors="coerce")
            - pd.to_numeric(df[fan_prev_col], errors="coerce"),
        }
    )
    out = out.dropna(subset=["S", "G"]).reset_index(drop=True)
    out.attrs["label"] = label
    return out


def fit_growth_linear(frame: pd.DataFrame) -> FitSummary:
    """Linear ``G ~ S`` with robust (HC3) SEs (P-069 claim check)."""
    m = smf.ols("G ~ S", data=frame).fit(cov_type="HC3")
    return FitSummary(
        spec="G ~ S",
        n=int(len(frame)),
        r2=float(m.rsquared),
        terms=["Intercept", "S"],
        coefs={k: float(m.params[k]) for k in ("Intercept", "S")},
        pvalues={k: float(m.pvalues[k]) for k in ("Intercept", "S")},
        stderr={k: float(m.bse[k]) for k in ("Intercept", "S")},
    )


def fit_growth_quadratic(frame: pd.DataFrame) -> FitSummary:
    """Paper Eq. (growth_model): ``G ~ S + S^2 + S:H_exp`` (HC3).

    The ``S:H_exp`` term is the pure interaction (no H_exp main effect), matching
    the paper's equation exactly.
    """
    m = smf.ols("G ~ S + I(S**2) + S:H_exp", data=frame).fit(cov_type="HC3")
    terms = ["Intercept", "S", "I(S ** 2)", "S:H_exp"]
    return FitSummary(
        spec="G ~ S + S^2 + S:H_exp",
        n=int(len(frame)),
        r2=float(m.rsquared),
        terms=terms,
        coefs={k: float(m.params[k]) for k in terms},
        pvalues={k: float(m.pvalues[k]) for k in terms},
        stderr={k: float(m.bse[k]) for k in terms},
    )


def surprise_claim_checks(frame: pd.DataFrame) -> pd.DataFrame:
    """Tidy claim-check rows for the paper's P-069..P-071 numeric claims."""
    linear = fit_growth_linear(frame)
    quad = fit_growth_quadratic(frame)
    label = frame.attrs.get("label", "t=W6")

    rows = [
        {
            "label": label,
            "claim": "beta1 (S) ~ 0.34, p<0.001",
            "term": "S",
            "spec": "linear",
            "coef": linear.coefs["S"],
            "p": linear.pvalues["S"],
            "n": linear.n,
            "r2": linear.r2,
            "within_tol_abs_0_05": abs(linear.coefs["S"] - 0.34) <= 0.05,
        },
        {
            "label": label,
            "claim": "beta2 (S^2) > 0 (Matthew effect)",
            "term": "S^2",
            "spec": "quadratic",
            "coef": quad.coefs["I(S ** 2)"],
            "p": quad.pvalues["I(S ** 2)"],
            "n": quad.n,
            "r2": quad.r2,
            "within_tol_abs_0_05": bool(quad.coefs["I(S ** 2)"] > 0),
        },
        {
            "label": label,
            "claim": "beta3 (S x H_exp) > 0 (veteran amplifies)",
            "term": "S:H_exp",
            "spec": "quadratic",
            "coef": quad.coefs["S:H_exp"],
            "p": quad.pvalues["S:H_exp"],
            "n": quad.n,
            "r2": quad.r2,
            "within_tol_abs_0_05": bool(quad.coefs["S:H_exp"] > 0),
        },
    ]
    return pd.DataFrame(rows)


def predict_quadratic(frame: pd.DataFrame, fit: FitSummary) -> pd.DataFrame:
    """Grid of fitted G vs S for H_exp in {0, 2} under the quadratic model.

    Used by the P-071 interaction panel: the veteran (H_exp=2) curve versus the
    rookie (H_exp=0) curve isolates ``beta3``.
    """
    s_lo = float(frame["S"].min())
    s_hi = float(frame["S"].max())
    grid = pd.DataFrame({"S": pd.Series([s_lo + (s_hi - s_lo) * i / 199 for i in range(200)])})
    rows = []
    for h in (0, 2):
        g = grid.copy()
        g["H_exp"] = h
        b0 = fit.coefs["Intercept"]
        b1 = fit.coefs["S"]
        b2 = fit.coefs["I(S ** 2)"]
        b3 = fit.coefs["S:H_exp"]
        g["G_pred"] = b0 + b1 * g["S"] + b2 * g["S"] ** 2 + b3 * g["S"] * h
        g["H_exp"] = h
        rows.append(g)
    return pd.concat(rows, ignore_index=True)
