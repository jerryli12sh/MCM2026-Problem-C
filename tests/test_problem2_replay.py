"""Problem 2 replay / case-study tests.

Pins the Problem 2 replay machinery to its registered reproduction targets:

- :class:`PooledFit` reload roundtrip from ``outputs/problem1_fit_meta_P.json``
  + ``problem1_fit_arrays_P.npz`` (the serialized form problem1_run.py writes);
- ``config_from_fit`` hyperparameter reconstruction with exact int/float types;
- ``DrawCache`` determinism and first-B-slice equivalence (the seed scheme makes
  the first ``B`` draws of a larger cache identical to a ``B``-sized cache);
- ``DrawCache.weighted_mean`` equals the ``p_mean`` column of
  ``outputs/problem1_posterior_summary_P.csv`` on training weeks;
- paper Table 1 ``|d|`` / ``Flip`` reproduction (B-08, registered tolerance);
- the reference b2-save CSV (``../data/metrics_b2_save.csv``) within MC tolerance.

Track P (``era_mode='legacy'``) is the registered reproduction target for the
paper Table 1 and b2-save numbers; Track R (``era_mode='official'``) runs the
same replay under the review's judge-signal definition and is checked
structurally only (no legacy reference exists for its numbers).

Real-data tests are skipped until ``scripts/problem1_run.py`` has produced the
track outputs.  These tests use the *reloaded* fit on purpose: the replay must
work from the serialized artifacts, not from an in-memory fit.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dwts_reproduction.config import load_paths
from dwts_reproduction.problem2 import (
    TABLE1_CASES,
    TABLE1_REFERENCE,
    DrawCache,
    b2_case_metrics,
    build_replay_inputs,
    case_divergence,
    config_from_fit,
    eligible_weeks,
    load_pooled_fit,
    p_hat_unweighted,
    season_rule_metrics,
    week_judge_vector,
)

# --------------------------------------------------------------------------- #
# Registered reproduction targets (docs/BASELINE_PAPER_OUTPUTS.md)
# --------------------------------------------------------------------------- #
# B-08: Table 1 case-study |d| / Flip.  |d| reproduced within 0.02 (measured
# 0.0025 / 0.005 on the two rounding-margin cases); Flip within 0.01 (measured
# 0.0033).  flip_week is exact (deterministic from the draw stream).
ABS_D_TOL = 0.02
FLIP_TOL = 0.01
FLIP_WEEK_REFERENCE = {
    (2, "Jerry Rice"): 6,
    (4, "Billy Ray Cyrus"): 9,
    (11, "Bristol Palin"): 9,
    (27, "Bobby Bones"): 4,
    (27, "Tinashe"): 4,
    (31, "Vinny Guadagnino"): 7,
}

# b2-save CSV (../data/metrics_b2_save.csv): MC tolerance for B=600 Bernoulli
# proportions and draw-aligned trajectory means.  Measured max-abs deviations
# were p_b2 3.3e-4, p_rev 2.4e-4, p_rev_given_b2 3.7e-3, dE_T 6.7e-3,
# dP_finals 1.7e-3; the registered bounds leave a margin for the reference
# producer's independent RNG stream.
B2_TOLERANCE = {
    "p_b2": 0.002,
    "p_rev": 0.002,
    "p_rev_given_b2": 0.005,
    "dE_T": 0.010,
    "dP_finals": 0.005,
}


# --------------------------------------------------------------------------- #
# Real-data fixtures (Track P = registered reproduction; Track R = structural)
# --------------------------------------------------------------------------- #
def _load_track(track: str):
    """Load the serialized fit + rebuilt panel for a track, or skip."""
    paths = load_paths()
    out = paths.repo_root / "outputs"
    meta = out / f"problem1_fit_meta_{track}.json"
    arrays = out / f"problem1_fit_arrays_{track}.npz"
    if not (meta.exists() and arrays.exists()):
        pytest.skip(
            f"outputs/problem1_fit_*_{track}.* not present; "
            "run `python scripts/problem1_run.py --track P [--track R]` first"
        )
    fit = load_pooled_fit(meta, arrays)
    panel, train_weeks = build_replay_inputs(paths, fit)
    return panel, train_weeks, fit, paths


@pytest.fixture(scope="module")
def p_fit():
    """Track P (legacy era) fit + panel + paths."""
    return _load_track("P")


@pytest.fixture(scope="module")
def r_fit():
    """Track R (official era) fit + panel + paths."""
    return _load_track("R")


# --------------------------------------------------------------------------- #
# Fit reload and config reconstruction
# --------------------------------------------------------------------------- #
def test_load_pooled_fit_roundtrip(p_fit):
    """The serialized fit reloads with identical arrays, era, and contestant map."""
    _, _, fit, _ = p_fit
    assert fit.era_mode == "legacy"
    assert fit.seed == 42
    assert fit.model_type == "pooled_softmin_numpy"
    assert fit.beta.shape == (3,)
    assert fit.X_cols == ["j_metric_z", "age_z", "era_is_percent"]
    assert fit.u.shape == (fit.n_cs,)
    assert fit.n_cs == 421
    assert len(fit.cs2idx) == 421
    assert fit.cs2idx[(2, "Jerry Rice")] == 9
    assert isinstance(fit.bias, float) and np.isfinite(fit.bias)
    assert fit.loss_history and all(np.isfinite(v) for v in fit.loss_history)
    assert fit.hyperparameters["kappa"] == pytest.approx(10.0)
    # The npz arrays are the authoritative bytes.
    paths = load_paths()
    arrays = np.load(paths.repo_root / "outputs" / "problem1_fit_arrays_P.npz")
    np.testing.assert_array_equal(fit.beta, arrays["beta"])
    assert fit.bias == float(arrays["bias"])
    np.testing.assert_array_equal(fit.u, arrays["u"])


def test_config_from_fit_hyperparameters(p_fit):
    """Reconstructed config matches the stored hyperparameters for Track P."""
    _, _, fit, _ = p_fit
    cfg = config_from_fit(fit)
    assert cfg.era_mode == fit.era_mode == "legacy"
    assert cfg.seed == fit.seed == 42
    assert cfg.kappa == pytest.approx(10.0)
    assert cfg.tau_like == pytest.approx(0.15)
    assert cfg.tau_train == pytest.approx(0.05)
    assert cfg.l2_beta == pytest.approx(0.05)
    assert cfg.l2_u == pytest.approx(0.05)
    assert cfg.alpha_floor == pytest.approx(0.1)


def test_config_from_fit_int_casts(p_fit):
    """n_steps / batch_size / B / seed survive the JSON round trip as ints."""
    _, _, fit, _ = p_fit
    cfg = config_from_fit(fit)
    assert cfg.n_steps == 600 and isinstance(cfg.n_steps, int)
    assert cfg.batch_size == 32 and isinstance(cfg.batch_size, int)
    assert cfg.B == 1200 and isinstance(cfg.B, int)
    assert isinstance(cfg.seed, int)
    cfg600 = config_from_fit(fit, B=600)
    assert cfg600.B == 600 and isinstance(cfg600.B, int)


def test_config_from_fit_track_r(r_fit):
    """Track R hyperparameters reconstruct (fit_temperature/tau_train absence)."""
    _, _, fit, _ = r_fit
    cfg = config_from_fit(fit)
    assert cfg.era_mode == fit.era_mode == "official"
    assert cfg.tau_like == pytest.approx(0.15)  # stored explicitly
    assert cfg.tau_train == pytest.approx(0.05)  # tolerated absence -> default
    assert cfg.alpha_floor == pytest.approx(0.1)  # stored explicitly


# --------------------------------------------------------------------------- #
# DrawCache determinism and slice equivalence
# --------------------------------------------------------------------------- #
def test_draw_cache_is_deterministic(p_fit):
    """Two identically-configured caches draw byte-identical samples."""
    panel, train_weeks, fit, _ = p_fit
    cfg = config_from_fit(fit, B=600)
    c1 = DrawCache(panel, fit, cfg, max_B=600)
    c2 = DrawCache(panel, fit, cfg, max_B=600)
    s, w = int(train_weeks.iloc[0]["season"]), int(train_weeks.iloc[0]["week"])
    n1, p1 = c1.week(s, w)
    n2, p2 = c2.week(s, w)
    assert n1 == n2
    np.testing.assert_array_equal(p1, p2)


def test_draw_cache_first_B_slice_equivalence(p_fit):
    """A B=1200 cache's first 600 draws equal a B=600 cache's draws."""
    panel, train_weeks, fit, _ = p_fit
    cfg600 = config_from_fit(fit, B=600)
    cfg1200 = config_from_fit(fit, B=1200)
    c600 = DrawCache(panel, fit, cfg600, max_B=600)
    c1200 = DrawCache(panel, fit, cfg1200, max_B=1200)
    s, w = int(train_weeks.iloc[0]["season"]), int(train_weeks.iloc[0]["week"])
    n600, p600 = c600.week(s, w)
    n1200, p1200 = c1200.week(s, w)
    assert n600 == n1200
    assert p1200.shape[0] == 1200
    np.testing.assert_array_equal(p1200[:600], p600)


def test_draw_cache_aligned_matches_week_columns(p_fit):
    """``aligned`` reindexes by name to the alive judge vector without drift."""
    panel, train_weeks, fit, _ = p_fit
    cfg = config_from_fit(fit, B=600)
    cache = DrawCache(panel, fit, cfg, max_B=600)
    s, w = int(train_weeks.iloc[0]["season"]), int(train_weeks.iloc[0]["week"])
    names, j = week_judge_vector(panel, s, w)
    p, aligned_names = cache.aligned(s, w, names, 600)
    assert aligned_names == names
    store_names, full = cache.week(s, w)
    idx = {n: i for i, n in enumerate(store_names)}
    np.testing.assert_array_equal(p, full[:600, [idx[n] for n in names]])
    assert np.isfinite(j).all()


def test_p_hat_unweighted_is_draw_mean():
    """The Flip-engine point summary is the plain per-draw mean."""
    rng = np.random.default_rng(3)
    p = rng.dirichlet(np.ones(5), size=40)
    np.testing.assert_allclose(p_hat_unweighted(p), p.mean(axis=0))


def test_eligible_weeks_are_single_elim_nonfinal(p_fit):
    """Eligible weeks carry alive_n >= 3 on non-final single-elimination weeks."""
    panel, train_weeks, fit, _ = p_fit
    weeks = eligible_weeks(panel, train_weeks)
    assert len(weeks) > 0
    for s, w in weeks:
        g = panel[(panel["season"] == s) & (panel["week"] == w)]
        assert bool(g["alive"].sum() >= 3)
        assert bool(g["elim_this_week_end"].sum() == 1)
        assert not bool(g["is_final_week"].iloc[0])


# --------------------------------------------------------------------------- #
# weighted_mean vs posterior_summary p_mean
# --------------------------------------------------------------------------- #
def test_weighted_mean_matches_posterior_summary(p_fit):
    """Training-week weighted mean equals the stored posterior ``p_mean``.

    The posterior summary (``outputs/problem1_posterior_summary_P.csv``) was
    written by ``scripts/problem1_run.py`` from the *in-memory* fit; this test
    recomputes it from the *reloaded* fit and must agree to float precision.
    """
    panel, _, fit, paths = p_fit
    summary = pd.read_csv(paths.repo_root / "outputs" / "problem1_posterior_summary_P.csv")
    train = summary[summary["has_posterior"]].iloc[0]
    s, w = int(train["season"]), int(train["week"])
    names, _ = week_judge_vector(panel, s, w)
    cache = DrawCache(panel, fit, config_from_fit(fit, B=1200), max_B=1200)
    p_mean, aligned = cache.weighted_mean(s, w, names, 1200)
    assert aligned == names
    ref = summary[(summary["season"] == s) & (summary["week"] == w)].set_index("celebrity_name")[
        "p_mean"
    ]
    assert set(names) == set(ref.index)
    for name, pm in zip(names, p_mean, strict=True):
        assert pm == pytest.approx(ref[name], rel=1e-5, abs=1e-9)


# --------------------------------------------------------------------------- #
# Paper Table 1 case studies (|d| / Flip) — Track P reproduction
# --------------------------------------------------------------------------- #
def test_case_divergence_reproduces_table1(p_fit):
    """B-08: the six named cases reproduce |d| and Flip within tolerance."""
    panel, _, fit, _ = p_fit
    cfg = config_from_fit(fit, B=1200)
    df = case_divergence(panel, fit, cfg, TABLE1_CASES, B_div=1200, B_flip=600)
    assert len(df) == len(TABLE1_CASES) == 6
    for _, row in df.iterrows():
        key = (int(row["season"]), row["celebrity_name"])
        ref_d, ref_flip = TABLE1_REFERENCE[key]
        assert key in FLIP_WEEK_REFERENCE
        assert abs(row["abs_d"] - ref_d) <= ABS_D_TOL, (
            f"{key} |d| {row['abs_d']:.4f} vs ref {ref_d}"
        )
        assert abs(row["flip"] - ref_flip) <= FLIP_TOL, (
            f"{key} Flip {row['flip']:.4f} vs ref {ref_flip}"
        )
        assert row["flip_week"] == FLIP_WEEK_REFERENCE[key]
        # The 90% interval bands the per-draw |d| statistic.  The *point* |d|
        # uses the importance-weighted posterior mean p_hat, a different
        # estimator from the raw draws, so it need not fall inside the band
        # (e.g. S2 Jerry Rice point 3.6875 vs band [2.19, 3.44]); only
        # well-formedness and scale agreement are asserted (D-20260901-09).
        assert row["abs_d_ci_lo"] <= row["abs_d_ci_hi"]
        assert np.isfinite(
            [row["abs_d_ci_lo"], row["abs_d_ci_hi"], row["abs_d_posterior_mean"]]
        ).all()
        assert abs(row["abs_d"] - row["abs_d_posterior_mean"]) <= 0.5 * max(1.0, row["abs_d"])
        assert row["n_d_weeks"] >= 1


def test_case_divergence_table1_shapes(p_fit):
    """|d| is the max of |mean(delta_r)| and |mean(delta_s)|; bounds are sane."""
    panel, _, fit, _ = p_fit
    cfg = config_from_fit(fit, B=1200)
    df = case_divergence(panel, fit, cfg, TABLE1_CASES, B_div=1200, B_flip=600)
    for _, row in df.iterrows():
        assert row["abs_d"] == pytest.approx(max(abs(row["delta_r_bar"]), abs(row["delta_s_bar"])))
        assert 0.0 <= row["flip"] <= 1.0
        assert row["n_flip_weeks"] >= 1
        assert row["flip_week"] is not None and row["flip_week"] >= 1


def test_case_divergence_track_r_runs_and_bounds(r_fit):
    """Track R replays the same cases under the official era mapping.

    No registered numeric reference exists for Track R (the paper Table 1 values
    are legacy-era); this test only checks that the pipeline runs and that the
    outputs are well-formed.
    """
    panel, _, fit, _ = r_fit
    cfg = config_from_fit(fit, B=1200)
    df = case_divergence(panel, fit, cfg, TABLE1_CASES, B_div=1200, B_flip=600)
    assert len(df) == 6
    for _, row in df.iterrows():
        assert row["abs_d"] >= 0.0 and np.isfinite(row["abs_d"])
        assert 0.0 <= row["flip"] <= 1.0
        # Same estimator distinction as the Track P test: the point uses the
        # weighted posterior mean, the band the per-draw statistic.
        assert row["abs_d_ci_lo"] <= row["abs_d_ci_hi"]
        assert np.isfinite(
            [row["abs_d_ci_lo"], row["abs_d_ci_hi"], row["abs_d_posterior_mean"]]
        ).all()


# --------------------------------------------------------------------------- #
# Season-level rule metrics (paper Eqs. 3-6)
# --------------------------------------------------------------------------- #
def test_season_rule_metrics_shapes_and_bounds(p_fit):
    """Season rows carry point/posterior estimates and 90% intervals."""
    panel, _, fit, _ = p_fit
    cfg = config_from_fit(fit, B=600)
    df = season_rule_metrics(panel, fit, cfg, B=600)
    assert not df.empty
    assert {"season", "metric", "point", "posterior_mean", "ci_lo_10", "ci_hi_10"} <= set(
        df.columns
    )
    for _, row in df.iterrows():
        assert row["metric"] in {
            "dr",
            "override_rank",
            "override_pct",
            "fanworst_rank",
            "fanworst_pct",
            "delta",
        }
        assert 0.0 <= row["point"] <= 1.0 or row["metric"] == "delta"
        assert row["ci_lo_10"] <= row["ci_hi_10"]
        assert row["n_weeks"] >= 1
    # The delta sign convention (Eq. 6) is a difference of season-level means.
    sel = df[df["metric"].isin(["override_rank", "override_pct"])]
    ranks = sel[sel["metric"] == "override_rank"].set_index("season")
    pcts = sel[sel["metric"] == "override_pct"].set_index("season")
    deltas = df[df["metric"] == "delta"].set_index("season")
    assert set(deltas.index) == set(ranks.index)
    for s in deltas.index:
        assert deltas.loc[s, "point"] == pytest.approx(ranks.loc[s, "point"] - pcts.loc[s, "point"])


# --------------------------------------------------------------------------- #
# Reference b2-save metrics (../data/metrics_b2_save.csv) — Track P reproduction
# --------------------------------------------------------------------------- #
def test_b2_case_metrics_matches_reference_csv(p_fit):
    """All 12 individual rows reproduce the reference CSV within MC tolerance."""
    panel, _, fit, paths = p_fit
    cfg = config_from_fit(fit, B=600)
    got = b2_case_metrics(panel, fit, cfg, TABLE1_CASES, B=600)
    assert len(got) == 12  # 6 cases x {rank, pct}
    ref = pd.read_csv(paths.data_dir / "metrics_b2_save.csv")
    ref = ref[ref["unit_type"] == "individual"].copy()
    assert len(ref) == 12

    key = ["season", "celebrity_name", "baseline_mode"]
    compare_cols = ["p_b2", "p_rev_given_b2", "p_rev", "dE_T", "dP_finals"]
    merged = ref[key + compare_cols].merge(
        got[key + compare_cols], on=key, suffixes=("_ref", "_got")
    )
    assert len(merged) == 12
    for col, tol in B2_TOLERANCE.items():
        dev = (merged[f"{col}_got"] - merged[f"{col}_ref"]).abs()
        assert dev.max() <= tol, (
            f"{col}: max abs deviation {dev.max():.6f} exceeds registered tolerance {tol}"
        )
    # Denomination sanity: p_rev_given_b2 is conditioned on bottom-two presence.
    assert got["denom_b2"].gt(0).all()
    assert got["B"].eq(600).all()


def test_b2_case_metrics_track_r_runs_and_bounds(r_fit):
    """Track R replays the b2 metrics under the official era mapping (no ref)."""
    panel, _, fit, _ = r_fit
    cfg = config_from_fit(fit, B=600)
    df = b2_case_metrics(panel, fit, cfg, TABLE1_CASES, B=600)
    assert len(df) == 12
    for _, row in df.iterrows():
        assert 0.0 <= row["p_b2"] <= 1.0
        assert 0.0 <= row["p_rev"] <= 1.0
        assert np.isfinite(row["dE_T"]) and np.isfinite(row["dP_finals"])
