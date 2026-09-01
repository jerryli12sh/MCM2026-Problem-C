"""Problem 3 survival-determinant regressions (P-058..P-060).

Faithful port of the legacy producer ``../src/dwts_pro_celeb_regression.py``,
which fits parallel OLS regressions of standardized judge/fan outcomes on
celebrity age, industry, country and leakage-safe pro-history features, then
decomposes variance and runs forward (by-season) predictive CV.

The ported functions reproduce the legacy outputs bit-for-bit on the registered
``data/data_3.csv`` input (one row per (season, celebrity)).  ``paper_demo_model``
additionally fits the paper's *exact* Eq. (demo_model) — age + industry dummies
only, with ``Other`` as the reference category so every industry delta including
``Actor/Actress`` is directly estimable — to check the P-059/P-060 coefficient
claims against honest numbers.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import pearsonr
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from statsmodels.regression.linear_model import RegressionResultsWrapper

# Outcome key -> column in the engineered ``data_3.csv`` frame.  The fan late
# stage uses ``week_final_p_score_placement_z`` (data_3 provides no week-11 fan
# signal); the legacy comment interprets it as a late-stage fan proxy.
OUTCOMES: dict[str, str] = {
    "placement_z": "placement_z",
    "judge_w1": "week1_judge_score_placement_z",
    "judge_w6": "week6_judge_score_placement_z",
    "judge_w11": "week11_judge_score_placement_z",
    "fan_w1": "week1_p_score_placement_z",
    "fan_w6": "week6_p_score_placement_z",
    "fan_final": "week_final_p_score_placement_z",
}

# Ordering used by the legacy script for ``fit_all_ols``/``cv_table`` output.
OUTCOME_ORDER: tuple[str, ...] = (
    "placement_z",
    "judge_w1",
    "judge_w6",
    "judge_w11",
    "fan_w1",
    "fan_w6",
    "fan_final",
)

KEY_NUMERIC: list[str] = [
    "celebrity_age_during_season",
    "pro_hist_mean_placez",
    "pro_hist_win_rate",
    "pro_hist_top3_rate",
]

# Terms reported in ``extract_key_coefs`` (base spec).
KEY_TERMS: tuple[str, ...] = (
    "celebrity_age_during_season",
    "pro_hist_mean_placez",
    "pro_hist_win_rate",
    "pro_hist_top3_rate",
)

# Industry reference category for ``paper_demo_model`` so ``Actor/Actress`` is a
# directly estimable delta (see docstring).  Registered in the run manifest.
PAPER_INDUSTRY_REFERENCE = "Other"

# Judge/fan outcome keys grouped by signal family (for claim checks / figures).
JUDGE_OUTCOMES: tuple[str, ...] = ("judge_w1", "judge_w6", "judge_w11")
FAN_OUTCOMES: tuple[str, ...] = ("fan_w1", "fan_w6", "fan_final")


@dataclass
class OLSResult:
    """Compact legacy-compatible OLS summary row."""

    outcome: str
    spec: str
    n: int
    r2: float
    adj_r2: float
    aic: float


def load_data(path: str | Path) -> pd.DataFrame:
    """Read the engineered ``data_3.csv`` table (one row per season/celebrity)."""
    df = pd.read_csv(path)
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
    df["season"] = df["season"].astype(int)
    df["placement"] = pd.to_numeric(df["placement"], errors="coerce")
    df["placement_z"] = pd.to_numeric(df["placement_z"], errors="coerce")
    df["celebrity_age_during_season"] = pd.to_numeric(
        df["celebrity_age_during_season"], errors="coerce"
    )
    return df


def group_rare_categories(
    s: pd.Series, min_count: int = 6, other_label: str = "Other"
) -> pd.Series:
    """Keep categories appearing >= ``min_count``; map the rest to ``other_label``.

    ``NaN`` is filled with ``"Unknown"`` before counting (legacy behaviour).
    """
    s2 = s.fillna("Unknown").astype(str)
    counts = s2.value_counts(dropna=False)
    keep = set(counts[counts >= min_count].index)
    return s2.where(s2.isin(keep), other_label)


def add_pro_history_features(df: pd.DataFrame) -> pd.DataFrame:
    """Pro-dancer history based on PRIOR seasons only (leakage-safe).

    For each pro dancer ``p`` and season ``s``:

    - ``pro_hist_mean_placez(p, s)``: expanding mean of ``placement_z`` over
      seasons ``< s`` (H_abil in z units)
    - ``pro_hist_win_rate(p, s)``: expanding mean of ``I(placement == 1)``
    - ``pro_hist_top3_rate(p, s)``: expanding mean of ``I(placement <= 3)``
    - ``pro_hist_n_prev(p, s)``: number of prior appearances (H_exp)

    Rows with no history are set to 0.  Matches the legacy implementation.
    """
    out = df.copy()
    out = out.sort_values(["ballroom_partner", "season", "celebrity_name"]).reset_index(drop=True)
    g = out.groupby("ballroom_partner", sort=False)

    out["pro_hist_n_prev"] = g.cumcount()

    out["pro_hist_mean_placez"] = (
        g["placement_z"]
        .apply(lambda s: s.shift(1).expanding().mean())
        .reset_index(level=0, drop=True)
    )

    out["pro_hist_win_rate"] = (
        g["placement"]
        .apply(lambda s: (s.shift(1).eq(1)).expanding().mean())
        .reset_index(level=0, drop=True)
    )

    out["pro_hist_top3_rate"] = (
        g["placement"]
        .apply(lambda s: (s.shift(1).le(3)).expanding().mean())
        .reset_index(level=0, drop=True)
    )

    for c in ["pro_hist_mean_placez", "pro_hist_win_rate", "pro_hist_top3_rate"]:
        out[c] = out[c].fillna(0.0)

    return out.sort_values(["season", "placement", "celebrity_name"]).reset_index(drop=True)


def engineer_features(df: pd.DataFrame, min_cat_count: int = 6) -> pd.DataFrame:
    """Add industry/country groupings and leakage-safe pro-history features."""
    out = df.copy()
    out["industry_grp"] = group_rare_categories(out["celebrity_industry"], min_count=min_cat_count)
    out["country_grp"] = group_rare_categories(
        out["celebrity_homecountry_region"], min_count=min_cat_count
    )
    return add_pro_history_features(out)


def fit_ols(
    df: pd.DataFrame, y: str, rhs: str, robust: str = "HC3"
) -> tuple[RegressionResultsWrapper, int]:
    """Fit ``y ~ rhs`` on ``df`` with robust SEs; return ``(model, n)``."""
    cols_needed = [
        y,
        "celebrity_age_during_season",
        "industry_grp",
        "country_grp",
        "season",
        "pro_hist_mean_placez",
        "pro_hist_win_rate",
        "pro_hist_top3_rate",
    ]
    d = df[cols_needed].copy()
    d = d.dropna(subset=[y, "celebrity_age_during_season"])
    model = smf.ols(f"{y} ~ {rhs}", data=d).fit(cov_type=robust)
    return model, len(d)


def _base_rhs() -> str:
    return (
        "celebrity_age_during_season + pro_hist_mean_placez + pro_hist_win_rate"
        " + pro_hist_top3_rate + C(industry_grp) + C(country_grp)"
    )


def _fe_rhs() -> str:
    return _base_rhs() + " + C(season)"


def fit_all_ols(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[tuple[str, str], RegressionResultsWrapper]]:
    """Fit base and season-FE OLS for every outcome; return (summary, fitted)."""
    rows: list[dict] = []
    fitted: dict[tuple[str, str], RegressionResultsWrapper] = {}

    for outcome in OUTCOME_ORDER:
        y = OUTCOMES[outcome]
        m_base, n_base = fit_ols(df, y, _base_rhs())
        m_fe, n_fe = fit_ols(df, y, _fe_rhs())

        fitted[(outcome, "base")] = m_base
        fitted[(outcome, "seasonFE")] = m_fe

        rows.append(
            {
                "outcome": outcome,
                "spec": "base",
                "n": n_base,
                "R2": m_base.rsquared,
                "adjR2": m_base.rsquared_adj,
                "AIC": m_base.aic,
            }
        )
        rows.append(
            {
                "outcome": outcome,
                "spec": "seasonFE",
                "n": n_fe,
                "R2": m_fe.rsquared,
                "adjR2": m_fe.rsquared_adj,
                "AIC": m_fe.aic,
            }
        )

    summary = pd.DataFrame(rows).sort_values(["outcome", "spec"]).reset_index(drop=True)
    return summary, fitted


def extract_key_coefs(
    fitted: dict[tuple[str, str], RegressionResultsWrapper], spec: str = "base"
) -> pd.DataFrame:
    """Extract the four numeric-term coefficients from fitted models."""
    rows = []
    for outcome in OUTCOME_ORDER:
        m = fitted.get((outcome, spec))
        if m is None:
            continue
        for term in KEY_TERMS:
            if term in m.params.index:
                rows.append(
                    {
                        "outcome": outcome,
                        "term": term,
                        "coef": float(m.params[term]),
                        "se": float(m.bse[term]),
                        "p": float(m.pvalues[term]),
                    }
                )
    return pd.DataFrame(rows).sort_values(["outcome", "term"]).reset_index(drop=True)


def r2_for_formula(df: pd.DataFrame, y: str, rhs: str) -> tuple[float, int]:
    """OLS ``R^2`` (non-robust) for an outcome/formula; returns ``(r2, n)``."""
    d = df.dropna(subset=[y, "celebrity_age_during_season"]).copy()
    m = smf.ols(f"{y} ~ {rhs}", data=d).fit()
    return float(m.rsquared), int(len(d))


def incremental_r2_table(df: pd.DataFrame) -> pd.DataFrame:
    """Delta-R2 decomposition: pro-history vs industry/country contributions."""
    base_rhs = _base_rhs()
    rhs_no_pro = "celebrity_age_during_season + C(industry_grp) + C(country_grp)"
    rhs_no_cat = (
        "celebrity_age_during_season + pro_hist_mean_placez + pro_hist_win_rate"
        " + pro_hist_top3_rate"
    )

    rows = []
    for outcome in OUTCOME_ORDER:
        y = OUTCOMES[outcome]
        r2_full, n = r2_for_formula(df, y, base_rhs)
        r2_no_pro, _ = r2_for_formula(df, y, rhs_no_pro)
        r2_no_cat, _ = r2_for_formula(df, y, rhs_no_cat)
        rows.append(
            {
                "outcome": outcome,
                "n": n,
                "R2_full": r2_full,
                "ΔR2_pro_given_celeb": r2_full - r2_no_pro,
                "ΔR2_industry_country": r2_full - r2_no_cat,
            }
        )
    return pd.DataFrame(rows).sort_values("outcome").reset_index(drop=True)


def make_X(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot design matrix for the Ridge forward-CV."""
    X = df[
        [
            "celebrity_age_during_season",
            "pro_hist_mean_placez",
            "pro_hist_win_rate",
            "pro_hist_top3_rate",
            "industry_grp",
            "country_grp",
        ]
    ].copy()
    X = pd.get_dummies(X, columns=["industry_grp", "country_grp"], drop_first=True)
    return X


def season_forward_cv(
    df: pd.DataFrame,
    ycol: str,
    *,
    alpha: float = 1.0,
    min_train_seasons: int = 3,
) -> dict:
    """Forward (by-season) Ridge CV; report pooled Pearson r and RMSE."""
    seasons = sorted(df["season"].unique())
    preds: list[np.ndarray] = []
    trues: list[np.ndarray] = []
    season_used: list[int] = []

    for s in seasons:
        train = df[df["season"] < s].copy()
        test = df[df["season"] == s].copy()

        if train["season"].nunique() < min_train_seasons:
            continue

        train = train.dropna(subset=[ycol, "celebrity_age_during_season"])
        test = test.dropna(subset=[ycol, "celebrity_age_during_season"])
        if len(train) < 50 or len(test) < 5:
            continue

        X_train = make_X(train)
        X_test = make_X(test).reindex(columns=X_train.columns, fill_value=0)

        y_train = train[ycol].to_numpy(dtype=float)
        y_test = test[ycol].to_numpy(dtype=float)

        model = Ridge(alpha=alpha, random_state=0)
        model.fit(X_train, y_train)
        y_hat = model.predict(X_test)

        preds.append(y_hat)
        trues.append(y_test)
        season_used.append(int(s))

    if not preds:
        return {
            "pearson_r": np.nan,
            "rmse": np.nan,
            "n_test_total": 0,
            "n_seasons_tested": 0,
        }

    yhat = np.concatenate(preds)
    ytrue = np.concatenate(trues)
    r = float(pearsonr(ytrue, yhat).statistic)
    rmse = float(math.sqrt(mean_squared_error(ytrue, yhat)))
    return {
        "pearson_r": r,
        "rmse": rmse,
        "n_test_total": int(len(ytrue)),
        "n_seasons_tested": int(len(season_used)),
    }


def cv_table(df: pd.DataFrame) -> pd.DataFrame:
    """Forward-CV result for every outcome."""
    rows = []
    for outcome in OUTCOME_ORDER:
        res = season_forward_cv(df, OUTCOMES[outcome])
        rows.append({"outcome": outcome, **res})
    return pd.DataFrame(rows).sort_values("outcome").reset_index(drop=True)


def paper_demo_model(df: pd.DataFrame) -> pd.DataFrame:
    """Fit the paper's exact Eq. (demo_model): age + industry dummies only.

    ``Y = alpha + beta_Age Age + sum delta_k I(industry=k) + eps`` with robust
    (HC3) SEs and ``Other`` as the industry reference so ``Actor/Actress`` is a
    directly estimable delta (registered in the run manifest).  The paper's
    P-059/P-060 coefficient claims (age ~ -0.04; actor judge W1 ~0.16; fan W6
    -0.87) are checked against the honest values returned here.
    """
    rhs = f"celebrity_age_during_season + C(industry_grp, Treatment({PAPER_INDUSTRY_REFERENCE!r}))"
    rows: list[dict] = []
    for outcome in OUTCOME_ORDER:
        y = OUTCOMES[outcome]
        d = df.dropna(subset=[y, "celebrity_age_during_season"]).copy()
        if len(d) < 20:
            continue
        m = smf.ols(f"{y} ~ {rhs}", data=d).fit(cov_type="HC3")
        for term in m.params.index:
            rows.append(
                {
                    "outcome": outcome,
                    "term": term,
                    "coef": float(m.params[term]),
                    "se": float(m.bse[term]),
                    "p": float(m.pvalues[term]),
                    "n": int(len(d)),
                    "r2": float(m.rsquared),
                }
            )
    out = pd.DataFrame(rows)
    # Keep age rows plus the Actor/Athlete deltas for the claim checks.
    return out.sort_values(["outcome", "term"]).reset_index(drop=True)
