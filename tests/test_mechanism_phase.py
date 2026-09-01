"""Mechanism phase diagram replay tests (paper Fig. 5 / review R-040).

Pins the counterfactual trajectory replay behind the phase diagram
(``mechanism_phase.py``, decision D-20260901-10):

- synthetic panels exercise the roster-indexed carry-forward replay: no
  ``KeyError`` when a mechanism keeps a contestant the observed data already
  eliminated (the counterfactual alive set is *not* a subset of the observed
  alive set), monotone alive-set shrinking, mechanism independence when no
  eligible week exists, and posterior-propagated bounds;
- real-data structural checks (Track P fit only) confirm the full
  ``mechanism_phase_metrics`` table builds for every season and every mechanism
  without divergence errors.

Synthetic fits use ``beta_j`` to shape the popularity prior ``q_hat``: 0 gives
uniform fan share (observed-elimination-neutral), a large negative value makes
fan support anti-correlate with the judge signal (the paper's finding) so the
four mechanisms genuinely diverge.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dwts_reproduction.config import load_paths
from dwts_reproduction.problem1.track_p import PooledFit
from dwts_reproduction.problem2 import (
    MECHANISMS,
    build_replay_inputs,
    eligible_weeks,
    load_pooled_fit,
    mechanism_phase_metrics,
    phase_claim_checks,
)
from dwts_reproduction.problem2.mechanism_phase import (
    _replay_season_point,
    _review_y_from_surv,
    _week_contribution,
)
from dwts_reproduction.problem2.replay import (
    DrawCache,
    build_train_weeks,
    config_from_fit,
)

# Small draw count for fast synthetic / structural tests.  First-B equivalence
# (the seed scheme) means these are the same draws a B-sized run would use.
B_SMALL = 60
ALPHA = 0.10


# --------------------------------------------------------------------------- #
# Synthetic fixtures
# --------------------------------------------------------------------------- #
def _make_panel(season: int = 99) -> pd.DataFrame:
    """Five-contestant season: w1 single-elim (eligible), w2 double-elim, w3 final.

    Judge vectors sum to 1 within each alive set.  Observed eliminations:
    w1 -> A, w2 -> B and C, w3 (final) -> no one.
    """
    rows: list[dict[str, object]] = []

    def add(week: int, name: str, alive: bool, elim: bool, is_final: bool, j: float) -> None:
        rows.append(
            {
                "season": season,
                "week": week,
                "celebrity_name": name,
                "alive": alive,
                "elim_this_week_end": elim,
                "is_final_week": is_final,
                "max_week": 3,
                "judge_percent": j,
                "j_metric": j,
                "era": "percent",
                "age": 30.0,
            }
        )

    j1 = {"A": 0.05, "B": 0.15, "C": 0.20, "D": 0.35, "E": 0.25}
    for n in ("A", "B", "C", "D", "E"):
        add(1, n, True, n == "A", False, j1[n])
    j2 = {"B": 0.15, "C": 0.20, "D": 0.40, "E": 0.25}
    for n in ("B", "C", "D", "E"):
        add(2, n, True, n in ("B", "C"), False, j2[n])
    j3 = {"D": 0.55, "E": 0.45}
    for n in ("D", "E"):
        add(3, n, True, False, True, j3[n])
    return pd.DataFrame(rows)


def _make_panel_no_eligible(season: int = 98) -> pd.DataFrame:
    """Four-contestant season where every week is non-eligible (final only)."""
    rows: list[dict[str, object]] = []

    def add(week: int, name: str, alive: bool, elim: bool, j: float) -> None:
        rows.append(
            {
                "season": season,
                "week": week,
                "celebrity_name": name,
                "alive": alive,
                "elim_this_week_end": elim,
                "is_final_week": week == 1,
                "max_week": 1,
                "judge_percent": j,
                "j_metric": j,
                "era": "percent",
                "age": 30.0,
            }
        )

    j1 = {"A": 0.10, "B": 0.20, "C": 0.30, "D": 0.40}
    for n in ("A", "B", "C", "D"):
        add(1, n, True, False, j1[n])
    return pd.DataFrame(rows)


def _make_fit(season: int, beta_j: float = 0.0) -> PooledFit:
    """Hand-built pooled fit over one season's roster.

    ``X_cols`` matches ``build_feature_frame``; ``jm_mean=0``/``jm_std=1`` make
    ``j_metric_z == judge_percent``, so ``beta_j`` directly shapes ``q_hat``.
    ``u = 0`` and ``bias = 0`` keep every other logit zero.
    """
    names = ["A", "B", "C", "D", "E"]
    cs2idx = {(season, n): i for i, n in enumerate(names)}
    return PooledFit(
        beta=np.array([float(beta_j), 0.0, 0.0], dtype=float),
        bias=0.0,
        u=np.zeros(len(names)),
        X_cols=["j_metric_z", "age_z", "era_is_percent"],
        jm_mean=0.0,
        jm_std=1.0,
        use_age=False,
        age_mean=None,
        age_std=None,
        cs2idx=cs2idx,
        n_cs=len(names),
        seed=42,
        era_mode="legacy",
        hyperparameters={"kappa": 200.0, "tau_like": 0.30, "B": B_SMALL},
        model_type="pooled_softmin_numpy",
    )


@pytest.fixture(scope="module")
def synth() -> tuple[pd.DataFrame, PooledFit, list[tuple[int, int]]]:
    """Standard synthetic panel + strong-divergence fit + eligible weeks."""
    panel = _make_panel(99)
    fit = _make_fit(99, beta_j=-10.0)
    eligible = eligible_weeks(panel)
    return panel, fit, eligible


def _phase_df(panel: pd.DataFrame, fit: PooledFit, B: int = B_SMALL) -> pd.DataFrame:
    return mechanism_phase_metrics(panel, fit, B=B, alpha=ALPHA)


# --------------------------------------------------------------------------- #
# Panel / eligible-week integrity
# --------------------------------------------------------------------------- #
def test_synthetic_panel_train_and_eligible(synth):
    """The synthetic season has exactly one training (eligible) week: w1."""
    panel, _, eligible = synth
    train = build_train_weeks(panel)
    assert [(int(s), int(w)) for s, w in train[["season", "week"]].itertuples(index=False)] == [
        (99, 1)
    ]
    assert eligible == [(99, 1)]


def test_no_eligible_season_empty(synth):
    """The final-only season has no training/eligible weeks."""
    panel = _make_panel_no_eligible(98)
    assert build_train_weeks(panel).empty
    assert eligible_weeks(panel) == []


# --------------------------------------------------------------------------- #
# Well-formedness, bounds, determinism
# --------------------------------------------------------------------------- #
def test_phase_metrics_shape_and_bounds(synth):
    """One row per (season, mechanism); paper y and review y stay in [0, 1]."""
    panel, _, _ = synth
    df = _phase_df(panel, _make_fit(99, beta_j=0.0))
    assert list(df["mechanism"]) == list(MECHANISMS)
    assert df["season"].unique().tolist() == [99]
    assert df["n_eligible_weeks"].unique().tolist() == [1]
    assert df["n_weeks"].unique().tolist() == [3]
    assert df["B"].unique().tolist() == [B_SMALL]
    # ``x = mu(|Ds|)`` is a share difference -> [0, 1].  The paper's ``y =
    # 1 - mu(|Dr|)`` and the review's ``y_review`` use *raw* within-week rank
    # differences (ranks span 1..n), so they are not bounded below; the paper
    # only makes comparative claims about them (D-20260901-10).
    for col in ("x_point", "x_posterior_mean"):
        assert ((df[col] >= 0.0) & (df[col] <= 1.0)).all(), col
    for col in (
        "y_point",
        "y_posterior_mean",
        "y_review_point",
        "y_review_posterior_mean",
        "x_ci_lo_10",
        "x_ci_hi_10",
        "y_ci_lo_10",
        "y_ci_hi_10",
        "y_review_ci_lo_10",
        "y_review_ci_hi_10",
    ):
        assert np.isfinite(df[col]).all(), col
    for lo, hi in (
        ("x_ci_lo_10", "x_ci_hi_10"),
        ("y_ci_lo_10", "y_ci_hi_10"),
        ("y_review_ci_lo_10", "y_review_ci_hi_10"),
    ):
        assert np.isfinite(df[[lo, hi]].to_numpy()).all(), (lo, hi)
        assert (df[lo] <= df[hi] + 1e-12).all(), (lo, hi)


def test_phase_metrics_deterministic(synth):
    """Same inputs -> identical phase table (seeded per-week draws)."""
    panel, _, _ = synth
    fit = _make_fit(99, beta_j=-10.0)
    a = _phase_df(panel, fit)
    b = _phase_df(panel, fit)
    pd.testing.assert_frame_equal(a, b)


def test_no_eligible_week_mechanism_independence(synth):
    """No eligible week -> every mechanism reproduces the observed trajectory."""
    panel = _make_panel_no_eligible(98)
    df = _phase_df(panel, _make_fit(98, beta_j=-10.0))
    assert df["n_eligible_weeks"].unique().tolist() == [0]
    by_mech = df.set_index("mechanism")
    base = by_mech.loc["pct_bottom2"]
    for col in (
        "x_point",
        "y_point",
        "y_review_point",
        "x_posterior_mean",
        "y_posterior_mean",
        "y_review_posterior_mean",
    ):
        for mech in MECHANISMS:
            assert by_mech.loc[mech, col] == pytest.approx(base[col], abs=1e-12), (mech, col)


# --------------------------------------------------------------------------- #
# Carry-forward / divergence under a strong fan-judge conflict
# --------------------------------------------------------------------------- #
def test_divergence_and_carry_forward(synth):
    """With fan anti-correlated to judge, mechanisms diverge and no KeyError fires.

    The observed data eliminates A at w1.  A strong negative ``beta_j`` gives A
    the highest fan prior, so at least one mechanism keeps A past w1 (carry
    forward: A is not in any later observed alive set) and at least one other
    eliminates A at w1.  Both trajectories must run without error and yield
    different phase points and survival vectors.
    """
    panel, _, _ = synth
    fit = _make_fit(99, beta_j=-10.0)
    df = _phase_df(panel, fit)
    pts = df.set_index("mechanism")[["x_point", "y_point"]]
    assert len({tuple(row) for _, row in pts.iterrows()}) >= 2, (
        "all four mechanisms coincided on a fan-judge-conflict season"
    )

    train_keys = set(
        build_train_weeks(panel)[["season", "week"]].itertuples(index=False, name=None)
    )
    eligible_keys = set(eligible_weeks(panel))
    roster = ["A", "B", "C", "D", "E"]
    cache = DrawCache(panel, fit, config_from_fit(fit, B=B_SMALL), max_B=B_SMALL)
    alive_weeks = [1, 2, 3]
    surv_by_mech = {}
    for mechanism in MECHANISMS:
        weeks = _weeks_for(panel, fit, cache, alive_weeks, roster, train_keys)
        _, _, surv, _, _ = _replay_season_point(
            weeks, 99, mechanism, eligible_keys=eligible_keys, roster=roster
        )
        surv_by_mech[mechanism] = surv

    observed_alive = {
        n: int(panel[(panel["celebrity_name"] == n) & panel["alive"]]["week"].nunique())
        for n in roster
    }
    # A is observed alive only in w1 (surv 1); at least one mechanism keeps A
    # past its observed elimination -> carry-forward survives the divergence.
    assert observed_alive["A"] == 1
    assert max(float(surv_by_mech[m][0]) for m in MECHANISMS) >= 2.0
    # Survival counts are weeks alive: bounded by the number of weeks, and the
    # two observed finalists (D, E) survive all 3 weeks unless a mechanism
    # pre-eliminates them at w1 (the counterfactual may shrink the finale set).
    for m in MECHANISMS:
        assert ((surv_by_mech[m] >= 0.0) & (surv_by_mech[m] <= 3.0)).all(), m
    # Divergence: mechanisms produce different survival vectors on this season.
    assert len({tuple(np.round(surv_by_mech[m], 6)) for m in MECHANISMS}) >= 2


def _weeks_for(panel, fit, cache, alive_weeks, roster, train_keys):
    from dwts_reproduction.problem2.mechanism_phase import _extended_weeks

    return _extended_weeks(
        panel, fit, cache, 99, alive_weeks, roster, B=B_SMALL, train_keys=train_keys
    )


# --------------------------------------------------------------------------- #
# Unit checks on the axis formulas
# --------------------------------------------------------------------------- #
def test_week_contribution_hand_values():
    """x = sum|p - J|, y = 1 - mean|rF - rJ| over a known alive set."""
    p = np.array([0.50, 0.30, 0.20])
    j = np.array([0.10, 0.40, 0.50])
    x, dy = _week_contribution(p, j)
    assert x == pytest.approx(np.abs(p - j).sum(), abs=1e-12)
    rF = np.array([1, 2, 3])
    rJ = np.array([3, 2, 1])
    assert dy == pytest.approx(np.abs(rF - rJ).sum(), abs=1e-12)
    # The paper's y = 1 - mu(|Dr|) is not bounded below (raw rank differences).
    assert (1.0 - dy / 3.0) == pytest.approx(-1.0 / 3.0, abs=1e-12)


def test_review_y_hand_values():
    """y_review = 1 - mean|r_Final - r_J|; perfect agreement -> 1, reversal low."""
    r_J = np.arange(1.0, 6.0)  # A best judge standing, E worst
    valid = np.ones(5, dtype=bool)

    surv_perfect = np.arange(5.0, 0.0, -1.0)  # A wins -> A r_Final 1 == r_J 1
    assert float(_review_y_from_surv(surv_perfect, r_J, valid)) == pytest.approx(1.0, abs=1e-12)

    surv_reversed = np.arange(1.0, 6.0)  # A last -> r_Final 5 vs r_J 1
    y = float(_review_y_from_surv(surv_reversed, r_J, valid))
    assert y < 1.0  # reversed rankings lower judge consistency

    # Valid mask drops contestants with non-finite judge standing.
    r_J_nan = np.array([1.0, np.nan, 3.0, 4.0, 5.0])
    valid2 = np.isfinite(r_J_nan)
    y2 = _review_y_from_surv(surv_perfect, r_J_nan, valid2)
    assert np.isfinite(y2).all()
    assert float(y2) == pytest.approx(1.0, abs=1e-12)

    # 2-D draw input: per-draw ranks, one y per draw, finite and in [0, 1].
    surv_draws = np.tile(surv_perfect, (4, 1))
    y_d = _review_y_from_surv(surv_draws, r_J, valid)
    assert y_d.shape == (4,)
    assert np.isfinite(y_d).all() and ((y_d >= 0.0) & (y_d <= 1.0)).all()


# --------------------------------------------------------------------------- #
# Claim checks
# --------------------------------------------------------------------------- #
def test_phase_claim_checks_wellformed(synth):
    """Claim checks produce one row per claim with the documented columns."""
    panel, _, _ = synth
    df = _phase_df(panel, _make_fit(99, beta_j=-10.0))
    checks = phase_claim_checks(df, alpha=ALPHA)
    assert {
        "claim",
        "n_seasons",
        "mean_delta_y",
        "mean_delta_x",
        "mean_delta_y_lo",
        "mean_delta_y_hi",
    }.issubset(set(checks.columns))
    # The P-057 tail-risk row may report n_seasons == 0 when no season reaches
    # the high fan-influence threshold (explicit "not testable" row); every
    # other claim must have a non-empty season subset.
    tail = checks["claim"].str.startswith("Pct tail-risk")
    assert (checks.loc[~tail, "n_seasons"] >= 1).all()
    # The lift rows (Direct -> Bottom2) report finite deltas on a one-season table.
    lifts = checks[checks["claim"].str.startswith("Direct->Bottom2")]
    assert len(lifts) == 2
    assert np.isfinite(lifts["mean_delta_y"]).all()


# --------------------------------------------------------------------------- #
# Real-data structural check (Track P fit; skips until problem1_run has run)
# --------------------------------------------------------------------------- #
def test_real_phase_metrics_structural_p():
    """The full phase table builds for every season/mechanism on Track P data."""
    paths = load_paths()
    out = paths.repo_root / "outputs"
    meta, arrays = out / "problem1_fit_meta_P.json", out / "problem1_fit_arrays_P.npz"
    if not (meta.exists() and arrays.exists()):
        pytest.skip("run `python scripts/problem1_run.py --track P` first")
    fit = load_pooled_fit(meta, arrays)
    panel, _ = build_replay_inputs(paths, fit)
    df = mechanism_phase_metrics(panel, fit, B=60, alpha=ALPHA)
    seasons = sorted(panel["season"].unique())
    assert len(df) == len(seasons) * len(MECHANISMS)
    assert df["mechanism"].value_counts().to_dict() == {m: len(seasons) for m in MECHANISMS}
    assert np.isfinite(
        df[["x_point", "y_point", "x_posterior_mean", "y_posterior_mean"]].to_numpy()
    ).all()
    assert ((df["x_posterior_mean"] >= 0.0) & (df["x_posterior_mean"] <= 1.0)).all()
    for col in ("y_point", "y_posterior_mean", "y_review_posterior_mean"):
        assert np.isfinite(df[col]).all(), col
    # Seasons with at least one eligible week should show at least some
    # cross-mechanism spread in the point trajectory (replay is live).
    eligible_seasons = df.groupby("season")["n_eligible_weeks"].max() > 0
    live = df[df["season"].isin(eligible_seasons[eligible_seasons].index)]
    if not live.empty:
        spread = live.groupby(["season"])["y_point"].nunique()
        assert (spread >= 2).any(), "no season shows any cross-mechanism spread"
