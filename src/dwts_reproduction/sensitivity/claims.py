"""Sensitivity claim checks (paper P-091, P-092, P-093) and effect sizes.

The paper's sensitivity section (``2107542.tex`` lines 1060-1110) makes three
checkable statements about Figure 10:

- P-091: inferred fan support is *stable* under perturbation — the stability
  scatter shows high Spearman rank correlation between baseline and scenario
  ``p_mean``.
- P-092: the PCP tornado shows ``tau`` dominates, then ``kappa`` — i.e. the
  relative range of ``pcp_mean`` across the tau family is the largest.
- P-093: on the A1 lines, ``pcp_mean`` *decreases monotonically with tau* and
  *larger kappa uniformly shifts PCP upward*.

Each check operates only on the saved scenario summary tables (no re-fit), so a
figure and its claim share one source of truth.  Targets are operationalized from
the paper's prose; a failed row is reported honestly, not hidden.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

CLAIM_COLUMNS = [
    "claim_id",
    "track",
    "statement",
    "metric",
    "value",
    "target",
    "status",
]

_SLACK = 1e-6


def _row(claim_id: str, statement: str, metric: str, value, target: str, status: str) -> dict:
    return {
        "claim_id": claim_id,
        "track": "P",
        "statement": statement,
        "metric": metric,
        "value": value,
        "target": target,
        "status": status,
    }


def effect_sizes(
    summary_all: pd.DataFrame,
    baseline_summary: pd.DataFrame,
    metric: str = "pcp_mean",
) -> dict[str, float]:
    """Relative ranges ``(max - min) / baseline`` per perturbation family.

    Mirrors the legacy tornado computation exactly: the A1 tau/kappa families use
    the mean over the other axis; the A2/A3/A4 families use the raw per-scenario
    values.  Used by both the tornado figure and the P-092 claim so they agree.
    """
    if (
        summary_all is None
        or summary_all.empty
        or baseline_summary is None
        or baseline_summary.empty
    ):
        return {}
    base_val = float(baseline_summary.iloc[0].get(metric, float("nan")))
    if not pd.notna(base_val) or abs(base_val) < 1e-12:
        return {}
    out: dict[str, float] = {}

    a1 = summary_all[summary_all["scenario"] == "A1_grid"]
    if not a1.empty:
        tau_means = a1.groupby("tau")[metric].mean()
        kappa_means = a1.groupby("kappa")[metric].mean()
        out["tau"] = float((tau_means.max() - tau_means.min()) / base_val)
        out["kappa"] = float((kappa_means.max() - kappa_means.min()) / base_val)

    for key, label in [
        ("A2_lambda_ratio", "lambda_ratio"),
        ("A3_judge_transform", "judge_transform"),
        ("A4_leave_one_season_out", "leave_one_season_out"),
    ]:
        fam = summary_all[summary_all["scenario"] == key]
        if not fam.empty:
            out[label] = float((fam[metric].max() - fam[metric].min()) / base_val)
    return out


def check_p091(summary_all: pd.DataFrame, spearman_floor: float = 0.90) -> pd.DataFrame:
    """P-091: rank ordering of p_mean is stable (high Spearman) under perturbation."""
    stable = summary_all.dropna(subset=["spearman_p"])
    if stable.empty:
        return pd.DataFrame(columns=CLAIM_COLUMNS)
    min_sp = float(stable["spearman_p"].min())
    med_sp = float(stable["spearman_p"].median())
    return pd.DataFrame(
        [
            _row(
                "P-091",
                "Stability of inferred fan support: high Spearman between baseline and scenarios",
                "min/median Spearman (n scenarios)",
                {"min": round(min_sp, 4), "median": round(med_sp, 4), "n": int(len(stable))},
                f"min >= {spearman_floor} (high rank stability)",
                "pass" if min_sp >= spearman_floor else "fail",
            )
        ]
    )


def check_p092(
    summary_all: pd.DataFrame,
    baseline_summary: pd.DataFrame,
    metric: str = "pcp_mean",
) -> pd.DataFrame:
    """P-092: the tornado shows tau dominates, then kappa."""
    effects = effect_sizes(summary_all, baseline_summary, metric)
    if not effects or "tau" not in effects or "kappa" not in effects:
        return pd.DataFrame(columns=CLAIM_COLUMNS)
    tau_eff = effects["tau"]
    kappa_eff = effects["kappa"]
    return pd.DataFrame(
        [
            _row(
                "P-092",
                "PCP sensitivity tornado: tau dominates, then kappa",
                f"rel range of {metric} (tau vs kappa)",
                {
                    "tau": round(tau_eff, 4),
                    "kappa": round(kappa_eff, 4),
                    "all": {k: round(v, 4) for k, v in effects.items()},
                },
                "effect(tau) > effect(kappa)",
                "pass" if tau_eff > kappa_eff else "fail",
            )
        ]
    )


def _monotone_decreasing(series: pd.Series) -> bool:
    """True when the series is non-increasing (allowing ``_SLACK`` float noise)."""
    v = series.sort_index().to_numpy(dtype=float)
    if v.size < 2:
        return True
    return bool((np.diff(v) <= _SLACK).all())


def check_p093(a1_summary: pd.DataFrame) -> pd.DataFrame:
    """P-093: PCP_mean decreases monotonically in tau; larger kappa shifts up.

    Checks every A1 kappa line for monotone decrease in ``tau`` (P-093a) and every
    pair of consecutive kappa values at shared tau points for an upward shift
    (P-093b).  A line with fewer than two points passes trivially.
    """
    if a1_summary is None or a1_summary.empty:
        return pd.DataFrame(columns=CLAIM_COLUMNS)
    df = a1_summary.copy()

    # P-093a: per-kappa monotone decrease in tau.
    n_lines = int(df["kappa"].nunique())
    mono_ok = 0
    for _, g in df.groupby("kappa"):
        line = g.sort_values("tau").set_index("tau")["pcp_mean"]
        if _monotone_decreasing(line):
            mono_ok += 1

    # P-093b: at each tau, pcp_mean(kappa_hi) >= pcp_mean(kappa_lo).
    pairs_total = 0
    pairs_ok = 0
    for _tau, g in df.groupby("tau"):
        g = g.dropna(subset=["pcp_mean"]).sort_values("kappa")
        kappas = g["kappa"].tolist()
        vals = dict(zip(kappas, g["pcp_mean"].tolist(), strict=True))
        for lo, hi in zip(kappas[:-1], kappas[1:], strict=True):
            pairs_total += 1
            if vals[hi] >= vals[lo] - _SLACK:
                pairs_ok += 1

    rows = [
        _row(
            "P-093a",
            "PCP_mean decreases monotonically with tau (per kappa line)",
            "monotone non-increasing lines",
            f"{mono_ok}/{n_lines}",
            "all kappa lines monotone non-increasing",
            "pass" if mono_ok == n_lines else "fail",
        ),
        _row(
            "P-093b",
            "Larger kappa uniformly shifts PCP_mean upward",
            "upward kappa pairs (at fixed tau)",
            f"{pairs_ok}/{pairs_total}" if pairs_total else "n/a",
            "all kappa pairs shift upward",
            "pass" if pairs_total and pairs_ok == pairs_total else "fail",
        ),
    ]
    return pd.DataFrame(rows)


def check_all(
    summary_all: pd.DataFrame,
    baseline_summary: pd.DataFrame,
    a1_summary: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Run all sensitivity claim checks on the saved summary tables."""
    frames: list[pd.DataFrame] = [
        check_p091(summary_all),
        check_p092(summary_all, baseline_summary),
    ]
    if a1_summary is not None and not a1_summary.empty:
        frames.append(check_p093(a1_summary))
    frames = [f for f in frames if not f.empty]
    if not frames:
        return pd.DataFrame(columns=CLAIM_COLUMNS)
    return pd.concat(frames, ignore_index=True)
