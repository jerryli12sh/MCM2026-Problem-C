"""Sensitivity-analysis tests (paper A1-A4 + Figure 10, P-087..P-093).

Unit invariants (hand-built fixtures) pin the pure helpers -- the A3 judge-share
variant (softmax temperature 1.0 must reproduce the default share exactly),
``js_distance`` / ``spearman_corr``, the A1 grid value construction and
nearby-grid selection, the tornado ``effect_sizes``, and the P-091/P-092/P-093
claim checks in both their pass and fail paths.  One real-data test checks that
``build_panel_with_variant`` preserves panel shape against the default panel.

Track: the sensitivity section reproduces the paper on top of the Track P
posterior, so every claim row carries ``track="P"`` (D-20260901-20).  Metrics are
argmin-weighted PCP (paper-consistent) with the legacy softmin PCP kept separate.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dwts_reproduction.config import load_paths
from dwts_reproduction.preprocess import build_all_tables
from dwts_reproduction.problem1.panel import build_judge_rank_share, build_problem1_panel
from dwts_reproduction.sensitivity.analysis import (
    build_grid_values,
    build_judge_rank_share_variant,
    build_panel_with_variant,
    js_distance,
    select_nearby_grid,
    spearman_corr,
)
from dwts_reproduction.sensitivity.claims import (
    check_p091,
    check_p092,
    check_p093,
    effect_sizes,
)


def test_js_distance_symmetric_and_zero() -> None:
    p = np.array([0.7, 0.2, 0.1])
    q = np.array([0.2, 0.3, 0.5])
    assert js_distance(p, q) == pytest.approx(js_distance(q, p), abs=1e-12)
    assert js_distance(p, p) == pytest.approx(0.0, abs=1e-12)
    # Unnormalized inputs are normalized first.
    assert js_distance(p, q) == pytest.approx(js_distance(10 * p, 0.5 * q), abs=1e-12)


def test_js_distance_bounds() -> None:
    # Deterministic vs deterministic, different supports -> JS divergence < 1.
    p = np.array([1.0, 0.0, 0.0])
    q = np.array([0.0, 1.0, 0.0])
    assert 0.0 < js_distance(p, q) < 1.0


def test_spearman_corr_known_values() -> None:
    a = np.array([1.0, 2.0, 3.0, 4.0])
    assert spearman_corr(a, a) == pytest.approx(1.0)
    assert spearman_corr(a, a[::-1]) == pytest.approx(-1.0)
    b = np.array([1.0, 3.0, 2.0, 4.0])
    assert spearman_corr(a, b) == pytest.approx(0.8, abs=1e-12)
    # Ties use average rank (a rank, not a value, quantity).
    assert spearman_corr(np.array([1.0, 1.0, 2.0]), np.array([1.0, 2.0, 3.0])) == pytest.approx(
        0.8660254038, abs=1e-6
    )


def test_spearman_corr_constant_is_nan() -> None:
    assert np.isnan(spearman_corr(np.array([2.0, 2.0, 2.0]), np.array([1.0, 2.0, 3.0])))
    assert np.isnan(spearman_corr(np.array([], dtype=float), np.array([], dtype=float)))


def _synthetic_long_judge() -> pd.DataFrame:
    """One season, two weeks, three judges, four contestants (two alive)."""
    rows = []
    # week 1: judge scores; rank era shares matter, not the percentage era.
    for judge in ("J1", "J2", "J3"):
        rows.append([1, 1, "A", judge, 90.0, True, True])
        rows.append([1, 1, "B", judge, 85.0, True, True])
        rows.append([1, 1, "C", judge, 70.0, True, True])
        rows.append([1, 1, "D", judge, 60.0, True, True])
    return pd.DataFrame(
        rows,
        columns=[
            "season",
            "week",
            "celebrity_name",
            "judge",
            "judge_score",
            "eligible",
            "is_show_week",
        ],
    )


def _synthetic_base() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "season": [1, 1, 1, 1],
            "week": [1, 1, 1, 1],
            "celebrity_name": ["A", "B", "C", "D"],
            "alive": [True, True, False, False],
        }
    )


def test_variant_softmax_temperature_one_matches_default() -> None:
    """A3 softmax temperature 1.0 must reproduce the default judge_rank_share."""
    long_judge = _synthetic_long_judge()
    base = _synthetic_base()
    default = build_judge_rank_share(long_judge, base)
    variant = build_judge_rank_share_variant(long_judge, base, method="softmax", temperature=1.0)
    merged = default.merge(
        variant, on=["season", "week", "celebrity_name"], suffixes=("_default", "_variant")
    )
    assert len(merged) == len(default)
    assert np.allclose(merged["judge_rank_share_default"], merged["judge_rank_share_variant"])


def test_variant_percentile_method_sums_to_one() -> None:
    long_judge = _synthetic_long_judge()
    base = _synthetic_base()
    variant = build_judge_rank_share_variant(long_judge, base, method="percentile")
    by_week = variant.groupby(["season", "week"])["judge_rank_share"].sum()
    assert np.allclose(by_week.to_numpy(), 1.0)
    # Descending support: A > B > C (dead) excluded, so A's percentile weight > B's.
    a_share = float(variant.loc[variant["celebrity_name"] == "A", "judge_rank_share"].iloc[0])
    b_share = float(variant.loc[variant["celebrity_name"] == "B", "judge_rank_share"].iloc[0])
    assert a_share > b_share


def test_variant_rejects_unknown_method() -> None:
    with pytest.raises(ValueError, match="Unknown method"):
        build_judge_rank_share_variant(_synthetic_long_judge(), _synthetic_base(), method="bogus")


def test_build_grid_values_explicit_and_multipliers() -> None:
    assert build_grid_values(0.05, [0.1, 0.2], []) == [0.1, 0.2]
    got = build_grid_values(0.05, None, [1.0, 2.0, 2.0, 4.0])
    assert got == [0.05, 0.1, 0.2]
    # Zero floor: negative multipliers are clamped away.
    assert min(build_grid_values(0.05, None, [-2.0, 1.0])) > 0


def test_select_nearby_grid_keeps_nearest() -> None:
    tau_vals = [0.025, 0.05, 0.1]
    kappa_vals = [5.0, 10.0, 20.0]
    # grid_n small -> min 6 points; the 6 nearest to (0.05, 10).
    nearby = select_nearby_grid(tau_vals, kappa_vals, base_tau=0.05, base_kappa=10.0, grid_n=5)
    assert len(nearby) == 6
    # Baseline pair itself is the closest point.
    assert (0.05, 10.0) in nearby
    # grid_n large -> full grid.
    full = select_nearby_grid(tau_vals, kappa_vals, base_tau=0.05, base_kappa=10.0, grid_n=20)
    assert len(full) == 9
    # Distance ordering: (0.05, 10) < (0.025, 10) < (0.05, 5) < (0.025, 5).
    dists = {
        (t, k): (t / 0.05 - 1.0) ** 2 + (k / 10.0 - 1.0) ** 2 for t in tau_vals for k in kappa_vals
    }
    assert dists[(0.05, 10.0)] < dists[(0.025, 10.0)] < dists[(0.025, 5.0)]


def _baseline_summary(pcp: float = 0.6) -> pd.DataFrame:
    return pd.DataFrame([{"scenario_id": "baseline", "scenario": "baseline", "pcp_mean": pcp}])


def _a1_family() -> pd.DataFrame:
    """9-row A1 grid with a large tau effect and a small kappa effect."""
    tau_base = {0.025: 0.6, 0.05: 0.45, 0.1: 0.3}
    kappa_adj = {5.0: 0.005, 10.0: 0.0, 20.0: -0.005}
    rows = []
    for tau in (0.025, 0.05, 0.1):
        for kappa in (5.0, 10.0, 20.0):
            rows.append(
                {
                    "scenario_id": f"A1_tau{tau}_kappa{kappa}",
                    "scenario": "A1_grid",
                    "tau": tau,
                    "kappa": kappa,
                    "pcp_mean": tau_base[tau] + kappa_adj[kappa],
                    "spearman_p": 0.98,
                }
            )
    return pd.DataFrame(rows)


def test_effect_sizes_tau_dominates_kappa() -> None:
    summary_all = _a1_family()
    effects = effect_sizes(summary_all, _baseline_summary())
    assert set(effects) == {"tau", "kappa"}
    assert effects["tau"] > effects["kappa"] > 0


def test_effect_sizes_empty_without_baseline() -> None:
    assert effect_sizes(pd.DataFrame(), _baseline_summary()) == {}
    assert effect_sizes(_a1_family(), pd.DataFrame()) == {}


def test_check_p091_pass_and_fail() -> None:
    df = _a1_family()
    rows = check_p091(df)
    assert len(rows) == 1
    assert rows.iloc[0]["claim_id"] == "P-091"
    assert rows.iloc[0]["status"] == "pass"
    low = df.copy()
    low.loc[0, "spearman_p"] = 0.5
    assert check_p091(low).iloc[0]["status"] == "fail"
    # All-NaN spearman -> no claim row at all.
    missing = df.copy()
    missing["spearman_p"] = np.nan
    assert check_p091(missing).empty


def test_check_p092_pass_and_fail() -> None:
    assert check_p092(_a1_family(), _baseline_summary()).iloc[0]["status"] == "pass"
    # Flip kappa to dominate -> fail.
    flipped = _a1_family().copy()
    flipped["pcp_mean"] = 0.45 + 0.15 * flipped["kappa"] / 20.0  # kappa effect 0.15
    flipped.loc[flipped["tau"] == 0.025, "pcp_mean"] -= 0.05  # small tau effect
    assert check_p092(flipped, _baseline_summary()).iloc[0]["status"] == "fail"


def test_check_p093_monotone_in_tau_and_kappa_shift() -> None:
    # A1 lines: pcp decreases in tau, larger kappa shifts up (pass).
    rows = []
    for tau in (0.05, 0.1, 0.15):
        for kappa in (10.0, 20.0, 30.0):
            rows.append(
                {
                    "scenario": "A1_grid",
                    "tau": tau,
                    "kappa": kappa,
                    "pcp_mean": 0.6 - 1.0 * tau + 0.002 * kappa,
                }
            )
    a1 = pd.DataFrame(rows)
    out = check_p093(a1)
    statuses = dict(zip(out["claim_id"], out["status"], strict=True))
    assert statuses["P-093a"] == "pass"
    assert statuses["P-093b"] == "pass"

    # Break monotonicity in one kappa line -> P-093a fails.
    bad = a1.copy()
    bad.loc[(bad["kappa"] == 10.0) & (bad["tau"] == 0.1), "pcp_mean"] += 0.1
    out_bad = check_p093(bad)
    assert dict(zip(out_bad["claim_id"], out_bad["status"], strict=True))["P-093a"] == "fail"

    # Empty input -> empty frame.
    assert check_p093(pd.DataFrame()).empty


@pytest.mark.slow
def test_panel_with_variant_preserves_shape_on_real_data() -> None:
    """A3 rebuild via variant must keep the panel's row set and column schema."""
    paths = load_paths()
    tables = build_all_tables(paths.raw_data_csv)
    base_panel = build_problem1_panel(tables, "legacy")
    variant_panel = build_panel_with_variant(
        tables, era_mode="legacy", rank_method="softmax", temperature=1.0
    )
    # Same column schema; order may differ because the merged share is appended.
    assert set(variant_panel.columns) == set(base_panel.columns)
    assert len(variant_panel) == len(base_panel)
    key = ["season", "week", "celebrity_name"]
    merged = base_panel[key].merge(variant_panel[key], on=key)
    assert len(merged) == len(base_panel)
    # j_metric is recomputed from the (default) variant share at T=1.0.
    assert np.allclose(
        base_panel["j_metric"].fillna(0.0).to_numpy(),
        variant_panel["j_metric"].fillna(0.0).to_numpy(),
    )
