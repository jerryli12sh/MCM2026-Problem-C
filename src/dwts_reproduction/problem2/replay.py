"""Problem 2 historical replay and named case studies.

Replays the fitted weekly fan-support posteriors (Problem 1, either track) under
the paper's rank and percentage elimination rules and the bottom-2 + judges'
save mechanism, quantifying overrides, reversals, and the named controversy case
studies of ``paper_Latex/2107542.tex`` (Table 1).

Two distinct rule engines are kept apart on purpose (never merged silently):

- ``simulate_week`` (``rules.py``) reproduces the legacy notebook cell 29 engine
  used for the case-study mechanisms and ``Flip``.  Its judge signal is the raw
  within-week judge-score share ``judge_percent`` (which *is* the paper's
  ``T_i / sum_k T_k``), and the bottom-2 save compares raw judge shares.
- ``risk_and_bottom2`` (``rules.py``) reproduces ``src/b2_save_metrics.py``, the
  producer of the reference ``../data/metrics_b2_save.csv``.  Its judge signal is
  era-appropriate (``j_metric``: judge percent in percent-era weeks, judge-rank
  share in rank-era weeks) and the save compares the mode-specific judge signal.

Point fan-share semantics follow the legacy notebook cell-by-cell
(D-20260901-09): the season-rate metrics (Eqs. 3-6) and the ``|d|`` case-study
input use the *importance-weighted* posterior mean for training weeks and the
fitted pooled prior ``q_hat`` for every other alive week, over all alive weeks
including finales (cells 3/4/7/12/20); ``Flip`` and per-week mechanism
reversals use the *unweighted* per-draw realizations (cells 29/34).
Posterior-propagated estimates average each mechanism outcome over the
per-draw realizations (training weeks vary per draw; other alive weeks are
fixed at ``q_hat``).

Historical replay (conditioning on observed outcomes through the fitted
posterior) is deliberately distinct from counterfactual simulation assumptions
(synthetic ``p`` grids, counterfactual judge scores); the latter belong to
Problem 4.  Nothing here claims the inferred ``p`` is the official fan vote
(D-20260901-06).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from dwts_reproduction.config import Paths
from dwts_reproduction.preprocess import build_all_tables
from dwts_reproduction.problem1.config import Problem1Config
from dwts_reproduction.problem1.panel import build_problem1_panel, build_train_weeks
from dwts_reproduction.problem1.track_p import (
    PooledFit,
    pooled_q_for_week,
    posterior_draws_for_week,
)
from dwts_reproduction.problem2.rules import (
    descending_rank,
    elim_pct_idx,
    elim_rank_idx,
    fan_worst_idx,
    judge_vectors_from_shares,
    judge_worst_idx,
    risk_and_bottom2,
    simulate_week,
)

# Default draw counts used by the legacy notebook for the two case-study inputs.
# |d| (notebook cell 20) uses B = 1200 posterior means over training weeks;
# Flip (cell 29) uses B = 600 per-draw mechanism simulations on eligible weeks.
B_DIVERGENCE = 1200
B_MECHANISM = 600

# Paper Table 1 named controversy cases.
TABLE1_CASES = [
    (2, "Jerry Rice"),
    (4, "Billy Ray Cyrus"),
    (11, "Bristol Palin"),
    (27, "Bobby Bones"),
    (27, "Tinashe"),
    (31, "Vinny Guadagnino"),
]
# Paper Table 1 |d| / Flip reference values (legacy reproduction targets).
TABLE1_REFERENCE = {
    (2, "Jerry Rice"): (3.69, 0.87),
    (4, "Billy Ray Cyrus"): (3.25, 0.75),
    (11, "Bristol Palin"): (4.30, 0.97),
    (27, "Bobby Bones"): (4.00, 0.57),
    (27, "Tinashe"): (8.50, 0.57),
    (31, "Vinny Guadagnino"): (9.88, 0.33),
}


# --------------------------------------------------------------------------- #
# Fit reload and configuration reconstruction
# --------------------------------------------------------------------------- #
def load_pooled_fit(meta_path: Path, arrays_path: Path) -> PooledFit:
    """Rebuild a :class:`PooledFit` from its serialized metadata + arrays.

    Inverts ``PooledFit.as_dict``: ``cs2idx_json`` keys are ``"season::name"``
    strings and the arrays live in the ``.npz`` (``beta``, ``bias``, ``u``,
    ``loss_history``).  The fit's ``era_mode`` and ``hyperparameters`` are
    preserved so the replay can rebuild the matching panel and config.
    """
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    arrays = np.load(arrays_path)
    cs2idx = {
        (int(int(key.split("::")[0])), key.split("::", 1)[1]): int(idx)
        for key, idx in meta["cs2idx_json"].items()
    }
    return PooledFit(
        beta=np.asarray(arrays["beta"]),
        bias=float(arrays["bias"]),
        u=np.asarray(arrays["u"]),
        X_cols=list(meta["X_cols"]),
        jm_mean=float(meta["jm_mean"]),
        jm_std=float(meta["jm_std"]),
        use_age=bool(meta["use_age"]),
        age_mean=None if meta["age_mean"] is None else float(meta["age_mean"]),
        age_std=None if meta["age_std"] is None else float(meta["age_std"]),
        cs2idx=cs2idx,
        n_cs=int(meta["n_cs"]),
        seed=int(meta["seed"]),
        era_mode=str(meta["era_mode"]),
        loss_history=[float(v) for v in meta["loss_history"]],
        hyperparameters={k: float(v) for k, v in meta["hyperparameters"].items()},
        train_choice_sets=[(int(st[0]), int(st[1])) for st in meta["train_choice_sets"]],
        model_type=str(meta["model_type"]),
    )


def config_from_fit(fit: PooledFit, *, B: int | None = None) -> Problem1Config:
    """Rebuild a :class:`Problem1Config` from the stored fit hyperparameters.

    ``B`` overrides the stored draw count (used to request the mechanism/Flip
    ``B=600`` or the divergence ``B=1200`` count while keeping the seed scheme).
    """
    hp = fit.hyperparameters
    kwargs: dict[str, Any] = {
        "era_mode": fit.era_mode,
        "seed": fit.seed,
        # Tolerated absences: Track R stores no ``tau_train`` and Track P no
        # ``alpha_floor``; both fall back to the registered defaults.
        "tau_train": float(hp.get("tau_train", 0.05)),
        "tau_like": float(hp.get("tau_like", 0.15)),
        "kappa": float(hp.get("kappa", 10.0)),
        "l2_beta": float(hp.get("l2_beta", 0.05)),
        "l2_u": float(hp.get("l2_u", 0.05)),
        "lr": float(hp.get("lr", 0.02)),
        "n_steps": int(hp.get("n_steps", 600)),
        "batch_size": int(hp.get("batch_size", 32)),
        "B": int(B if B is not None else hp.get("B", 600)),
        "eps": float(hp.get("eps", 1e-6)),
        "alpha_floor": float(hp.get("alpha_floor", 0.1)),
    }
    return Problem1Config(**kwargs)


def build_replay_inputs(paths: Paths, fit: PooledFit) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build the panel and training weeks matching the fit's ``era_mode``.

    ``era_mode`` travels with the fit (Track P = ``'legacy'``, Track R =
    ``'official'``) so a replayed fit is never evaluated under the other track's
    judge-signal definition (D-20260901-01).
    """
    tables = build_all_tables(paths.raw_data_csv)
    warnings: list[str] = []
    panel = build_problem1_panel(tables, fit.era_mode, warnings)
    train_weeks = build_train_weeks(panel)
    return panel, train_weeks


def eligible_weeks(
    panel: pd.DataFrame, train_weeks: pd.DataFrame | None = None
) -> list[tuple[int, int]]:
    """Eligible mechanism weeks: single-elimination, non-final, alive set > 2.

    Matches ``eligible_week_selector`` in the legacy b2-save producer and the
    notebook cell 29 mechanism loop.  Weeks with any missing ``judge_percent``
    among the alive set are *not* silently simulated; callers detect and report
    them (D-20260901-09: a rule needs a full judge vector).
    """
    if train_weeks is None:
        train_weeks = build_train_weeks(panel)
    out: list[tuple[int, int]] = []
    for season, week in train_weeks[["season", "week"]].itertuples(index=False):
        alive_n = int(
            ((panel["season"] == season) & (panel["week"] == week) & panel["alive"]).sum()
        )
        if alive_n >= 3:
            out.append((int(season), int(week)))
    return out


@dataclass
class DrawCache:
    """Memoized unweighted posterior draws keyed by (season, week)."""

    panel: pd.DataFrame
    fit: PooledFit
    config: Problem1Config
    max_B: int = B_DIVERGENCE

    def __post_init__(self) -> None:
        # The cache always draws ``max_B`` (the shared seed makes the first ``B``
        # draws identical to a ``B``-sized run), so config.B must be at least as
        # large as every requested slice.
        if self.config.B < self.max_B:
            self.config = self.config.__class__(**{**self.config.__dict__, "B": self.max_B})
        self._store: dict[tuple[int, int], tuple[list[str], np.ndarray]] = {}
        self._weights: dict[tuple[int, int], np.ndarray] = {}

    def week(self, season: int, week: int) -> tuple[list[str], np.ndarray]:
        """Unweighted draws aligned to the name-sorted alive set of one week."""
        key = (int(season), int(week))
        if key not in self._store:
            res = posterior_draws_for_week(
                self.panel,
                self.fit,
                int(season),
                int(week),
                self.config,
                has_posterior_mode="legacy",
            )
            alive = res["alive"].reset_index(drop=True)
            self._store[key] = (alive["celebrity_name"].astype(str).tolist(), res["samples"])
            self._weights[key] = np.asarray(res["weights"], dtype=float)
        return self._store[key]

    def aligned(
        self, season: int, week: int, names: list[str], B: int
    ) -> tuple[np.ndarray, list[str]]:
        """Slice and reindex the week's draws to ``names`` (name order preserved)."""
        store_names, p = self.week(season, week)
        idx = {n: i for i, n in enumerate(store_names)}
        cols = [idx[n] for n in names if n in idx]
        return p[:B, cols], [n for n in names if n in idx]

    def weighted_mean(
        self, season: int, week: int, names: list[str], B: int
    ) -> tuple[np.ndarray, list[str]]:
        """Importance-weighted posterior mean over the first ``B`` draws.

        Training (single-elimination, non-final) weeks carry softmin importance
        weights, so this reproduces the legacy ``posterior_mean_for_week``
        ``p_mean`` used for ``|d|`` and the season-rate metrics (notebook cells
        3/4/7/12/20).  Weights are renormalized over the slice in case
        ``B < max_B``.  Non-training weeks draw uniform weights here; callers
        that need the cell-3 ``q_hat`` fallback must use ``_week_p_hat``.
        """
        key = (int(season), int(week))
        store_names, p = self.week(season, week)
        w = self._weights[key]
        idx = {n: i for i, n in enumerate(store_names)}
        cols = [idx[n] for n in names if n in idx]
        w_b = w[:B]
        w_b = w_b / w_b.sum()
        return (w_b[:, None] * p[:B, cols]).sum(axis=0), [n for n in names if n in idx]


def week_judge_vector(
    panel: pd.DataFrame, season: int, week: int, j_col: str = "judge_percent"
) -> tuple[list[str], np.ndarray]:
    """Alive (name, judge vector) for one week; NaN judge rows are kept visible."""
    g = panel[(panel["season"] == season) & (panel["week"] == week) & panel["alive"]].copy()
    g = g.sort_values("celebrity_name")
    return g["celebrity_name"].astype(str).tolist(), g[j_col].to_numpy(dtype=float)


def p_hat_unweighted(p_samps: np.ndarray) -> np.ndarray:
    """Unweighted draw mean — the summary of the draws used by ``Flip``.

    Only the per-draw mechanism engine (cells 29/34) used the raw
    ``p_draws.mean(axis=0)``.  The ``|d|`` case-study input (cell 20) and the
    season-rate metrics (cells 4/7/12) instead use the *importance-weighted*
    posterior mean for training weeks and ``q_hat`` for all other alive weeks
    (see ``_week_p_hat``), so this helper is not their point estimator
    (D-20260901-09).
    """
    return np.asarray(p_samps, dtype=float).mean(axis=0)


# --------------------------------------------------------------------------- #
# Season-level override / reversal / fan-worst metrics (paper Eqs. 3-6)
# --------------------------------------------------------------------------- #
_METRICS = ("dr", "override_rank", "override_pct", "fanworst_rank", "fanworst_pct", "delta")


def season_rule_metrics(
    panel: pd.DataFrame,
    fit: PooledFit,
    config: Problem1Config,
    *,
    B: int = B_DIVERGENCE,
    alpha: float = 0.10,
) -> pd.DataFrame:
    """Paper Eqs. 3-6 season metrics with point and posterior-propagated values.

    Faithful to the legacy season tables (notebook cells 4/7/12), which evaluate
    every *alive* week — finales included — at the cell-3 point fan share
    ``p_hat`` (importance-weighted posterior mean for training weeks, ``q_hat``
    otherwise; D-20260901-09).  Season rows report:

    - ``dr``: ``DR_s = mean_t 1{e_rank != e_pct}`` (paper Eq. 3);
    - ``override_rank`` / ``override_pct``: ``mean_t 1{e_m != argmin_k T_k}``
      (Eq. 4);
    - ``fanworst_rank`` / ``fanworst_pct``: ``mean_t 1{e_m == argmin_k p_k}``
      (Eq. 5);
    - ``delta``: ``E[Override_rank] - E[Override_pct]`` (Eq. 6 sign convention:
      negative => percentage rule more prone to fan-driven overrides).

    Each metric carries ``point`` (evaluated at ``p_hat``), ``posterior_mean``
    and a ``1 - alpha`` interval averaged over per-draw season realizations:
    training weeks vary per draw, non-training weeks are fixed at the point.
    """
    cfg = config_from_fit(fit, B=B)
    cache = DrawCache(panel, fit, cfg, max_B=B)
    train_weeks = build_train_weeks(panel)
    train_keys = set(train_weeks[["season", "week"]].itertuples(index=False, name=None))
    alive_weeks = list(
        panel[panel["alive"]][["season", "week"]]
        .drop_duplicates()
        .sort_values(["season", "week"])
        .itertuples(index=False, name=None)
    )
    by_season: dict[int, list[dict[str, Any]]] = {}

    skipped: list[tuple[int, int]] = []
    for season, week in alive_weeks:
        names, j = week_judge_vector(panel, season, week)
        if not np.isfinite(j).all():
            skipped.append((int(season), int(week)))
            continue
        p_hat = _week_p_hat(panel, fit, cache, season, week, names, train_keys, B)
        jw = judge_worst_idx(j)
        fw_hat = fan_worst_idx(p_hat)
        point = {
            "dr": float(elim_rank_idx(j, p_hat) != elim_pct_idx(j, p_hat)),
            "override_rank": float(elim_rank_idx(j, p_hat) != jw),
            "override_pct": float(elim_pct_idx(j, p_hat) != jw),
            "fanworst_rank": float(elim_rank_idx(j, p_hat) == fw_hat),
            "fanworst_pct": float(elim_pct_idx(j, p_hat) == fw_hat),
        }
        if (int(season), int(week)) in train_keys:
            p_draws, _ = cache.aligned(season, week, names, B)
            er = np.array([elim_rank_idx(j, pb) for pb in p_draws])
            ep = np.array([elim_pct_idx(j, pb) for pb in p_draws])
            fw_d = np.array([fan_worst_idx(pb) for pb in p_draws])
            draws = {
                "dr": (er != ep).astype(float),
                "override_rank": (er != jw).astype(float),
                "override_pct": (ep != jw).astype(float),
                "fanworst_rank": (er == fw_d).astype(float),
                "fanworst_pct": (ep == fw_d).astype(float),
            }
        else:
            draws = {m: np.full(B, float(v)) for m, v in point.items()}
        by_season.setdefault(int(season), []).append({"point": point, "draws": draws})

    rows: list[dict[str, Any]] = []
    for season, weeks_data in sorted(by_season.items()):
        n_weeks = len(weeks_data)
        if n_weeks == 0:
            continue
        draws_arr = {
            m: np.stack([w["draws"][m] for w in weeks_data], axis=0)
            for m in _METRICS
            if m != "delta"
        }
        # delta is a difference of season-level means, per draw.
        draws_arr["delta"] = draws_arr["override_rank"].mean(axis=0) - draws_arr[
            "override_pct"
        ].mean(axis=0)
        point = {
            m: float(np.mean([w["point"][m] for w in weeks_data])) for m in _METRICS if m != "delta"
        }
        point["delta"] = point["override_rank"] - point["override_pct"]
        lo_q, hi_q = alpha / 2.0, 1.0 - alpha / 2.0
        for m in _METRICS:
            pm = float(draws_arr[m].mean())
            lo, hi = np.quantile(draws_arr[m], [lo_q, hi_q])
            rows.append(
                {
                    "season": season,
                    "metric": m,
                    "point": point[m],
                    "posterior_mean": pm,
                    f"ci_lo_{int(alpha * 100):02d}": float(lo),
                    f"ci_hi_{int(alpha * 100):02d}": float(hi),
                    "n_weeks": n_weeks,
                    "B": B,
                }
            )
    out = pd.DataFrame(rows)
    out.attrs["skipped_weeks"] = skipped
    return out


# --------------------------------------------------------------------------- #
# Case studies: |d| (cell 20) and Flip (cell 29)
# --------------------------------------------------------------------------- #
def _week_p_hat(
    panel: pd.DataFrame,
    fit: PooledFit,
    cache: DrawCache,
    season: int,
    week: int,
    names: list[str],
    train_keys: set[tuple[int, int]],
    B: int,
) -> np.ndarray:
    """Legacy point fan-share vector for one alive week (notebook cell 3).

    Training weeks (single-elimination, non-final) use the importance-weighted
    posterior mean over the first ``B`` draws; every other alive week — finales
    and multi-elimination weeks — uses the fitted pooled prior ``q_hat``.  This
    is exactly ``week_fanshare``'s ``p_hat`` column (D-20260901-09).
    """
    if (int(season), int(week)) in train_keys:
        p_hat, aligned = cache.weighted_mean(season, week, names, B)
        if aligned != names:
            raise ValueError(
                f"week {season}/{week}: posterior alive set {aligned} != judge set {names}"
            )
        return p_hat
    q = pooled_q_for_week(panel, fit, int(season), int(week))
    out = np.asarray(q.set_index("celebrity_name").reindex(names)["q_hat"].to_numpy(dtype=float))
    if not np.isfinite(out).all():
        raise ValueError(f"week {season}/{week}: q_hat not finite for all alive contestants")
    return out


def _week_delta_row(
    panel: pd.DataFrame,
    fit: PooledFit,
    cache: DrawCache,
    season: int,
    week: int,
    name: str,
    train_keys: set[tuple[int, int]],
    B: int,
) -> dict[str, Any] | None:
    """Per-week (delta_r, delta_s) for one contestant at the legacy ``p_hat``.

    ``p_hat`` follows cell 3 (weighted posterior mean for training weeks,
    ``q_hat`` otherwise) and the week is any alive week of the contestant —
    matching cell 20's ``disagree_index`` (finales included, via ``q_hat``).
    ``delta_r = rF - rJ`` and ``delta_s = p_hat - J_pct`` with ``J_pct`` the
    within-week judge-score share (``judge_percent`` IS the paper's ``T/sum T``).
    Training weeks additionally carry per-draw deltas for posterior propagation;
    non-training weeks are fixed at the point across draws.
    """
    names, j = week_judge_vector(panel, season, week)
    if name not in names or not np.isfinite(j).all():
        return None
    p_hat = _week_p_hat(panel, fit, cache, season, week, names, train_keys, B)
    i = names.index(name)
    rJ = descending_rank(j)
    point = {
        "delta_r": float(descending_rank(p_hat)[i] - rJ[i]),
        "delta_s": float(p_hat[i] - j[i]),
    }
    if (int(season), int(week)) in train_keys:
        p_draws, aligned_names = cache.aligned(season, week, names, B)
        i_d = aligned_names.index(name)
        rF_d = np.stack([descending_rank(pb) for pb in p_draws])
        return {
            "point": point,
            "delta_r_draws": rF_d[:, i_d] - rJ[i],
            "delta_s_draws": p_draws[:, i_d] - j[i],
        }
    return {
        "point": point,
        "delta_r_draws": np.full(B, point["delta_r"], dtype=float),
        "delta_s_draws": np.full(B, point["delta_s"], dtype=float),
    }


def case_divergence(
    panel: pd.DataFrame,
    fit: PooledFit,
    config: Problem1Config,
    cases: list[tuple[int, str]],
    *,
    B_div: int = B_DIVERGENCE,
    B_flip: int = B_MECHANISM,
    alpha: float = 0.10,
) -> pd.DataFrame:
    """Reproduce the paper Table 1 case-study ``|d|`` and ``Flip``.

    ``|d|`` (notebook cell 20): contestant-season means of the weekly
    ``delta_r = rF - rJ`` and ``delta_s = p_hat - J_pct``, then
    ``|d| = max(|mean(delta_r)|, |mean(delta_s)|)``, over *all* alive weeks of
    the contestant (finales included).  ``p_hat`` is the cell-3 point: the
    importance-weighted B=1200 posterior mean on training weeks and ``q_hat``
    otherwise (D-20260901-09).

    ``Flip`` (cell 29/34): the *season-wide* maximum of the weekly share of
    B=600 draws for which the rank rule and the percentage rule pick different
    eliminatees, over all non-finale single-elimination weeks of the season.
    Legacy ``_infer_flags`` marks every ``df_weekly`` row alive, so this maximum
    is not restricted to weeks the contestant was alive (e.g. Billy Ray Cyrus,
    eliminated S4 w8, carries the S4 w9 rate 0.75).  Per-draw elimination uses
    ``simulate_week`` (the legacy lexsort tie-breaking of notebook cell 29),
    *not* the paper-formula first-index ``elim_rank_idx``/``elim_pct_idx``
    (D-20260901-09).

    Each row also carries the posterior-propagated 90% interval of ``|d|`` over
    the B=1200 draws (per-draw analogue of the point definition: training weeks
    vary per draw, other alive weeks are fixed at ``q_hat``).
    """
    cfg_div = config_from_fit(fit, B=B_div)
    cfg_flip = config_from_fit(fit, B=B_flip)
    cache_div = DrawCache(panel, fit, cfg_div, max_B=B_div)
    cache_flip = DrawCache(panel, fit, cfg_flip, max_B=B_flip)
    train_weeks = build_train_weeks(panel)
    weeks_eligible = eligible_weeks(panel, train_weeks)
    train_keys = set(train_weeks[["season", "week"]].itertuples(index=False, name=None))
    alive_weeks = list(
        panel[panel["alive"]][["season", "week"]]
        .drop_duplicates()
        .sort_values(["season", "week"])
        .itertuples(index=False, name=None)
    )

    rows: list[dict[str, Any]] = []
    for season, name in cases:
        season = int(season)
        # --- |d| over ALL alive weeks (cell 20) ---------------------------
        d_rows = []
        n_train_d_weeks = 0
        for s, wk in alive_weeks:
            if s != season:
                continue
            row = _week_delta_row(panel, fit, cache_div, season, wk, name, train_keys, B_div)
            if row is None:
                continue
            if (int(season), int(wk)) in train_keys:
                n_train_d_weeks += 1
            d_rows.append(row)
        if d_rows:
            delta_r_bar = float(np.mean([r["point"]["delta_r"] for r in d_rows]))
            delta_s_bar = float(np.mean([r["point"]["delta_s"] for r in d_rows]))
            d_point = float(max(abs(delta_r_bar), abs(delta_s_bar)))
            dr_d = np.mean([r["delta_r_draws"] for r in d_rows], axis=0)
            ds_d = np.mean([r["delta_s_draws"] for r in d_rows], axis=0)
            d_draws = np.maximum(np.abs(dr_d), np.abs(ds_d))
            lo_q, hi_q = alpha / 2.0, 1.0 - alpha / 2.0
            d_lo, d_hi = np.quantile(d_draws, [lo_q, hi_q])
        else:
            d_point = d_lo = d_hi = np.nan
            delta_r_bar = delta_s_bar = np.nan
            dr_d = ds_d = d_draws = np.array([np.nan])

        # --- Flip over the season's eligible weeks (cell 29/34) ----------
        # Legacy ``_infer_flags`` marks *every* ``df_weekly`` row alive (the
        # frame has no ``alive`` column), so ``eligible``/``contestant_week``
        # span all non-finale single-elim weeks of the season and ``Flip`` is
        # the season-wide max reversal rate — *not* restricted to weeks the
        # contestant was alive.  E.g. Billy Ray Cyrus (eliminated S4 w8) still
        # shows the S4 w9 rate 0.75.  D-20260901-09.
        flip_weeks: list[tuple[int, float]] = []
        for s, wk in weeks_eligible:
            if s != season:
                continue
            names, j = week_judge_vector(panel, season, wk)
            if not np.isfinite(j).all():
                continue
            p, _ = cache_flip.aligned(season, wk, names, B_flip)
            er = np.array([simulate_week(pb, j, names, "rank_direct")[0] for pb in p])
            ep = np.array([simulate_week(pb, j, names, "pct_direct")[0] for pb in p])
            flip_weeks.append((int(wk), float((er != ep).mean())))
        flip = float(max(w[1] for w in flip_weeks)) if flip_weeks else np.nan
        flip_week = int(max(flip_weeks, key=lambda t: t[1])[0]) if flip_weeks else None

        rows.append(
            {
                "season": season,
                "celebrity_name": name,
                "abs_d": d_point,
                "delta_r_bar": delta_r_bar,
                "delta_s_bar": delta_s_bar,
                "abs_d_posterior_mean": float(np.nanmean(d_draws)),
                "abs_d_ci_lo": d_lo,
                "abs_d_ci_hi": d_hi,
                "flip": flip,
                "flip_week": flip_week,
                "n_d_weeks": len(d_rows),
                "n_train_d_weeks": n_train_d_weeks,
                "n_flip_weeks": len(flip_weeks),
                "B_div": B_div,
                "B_flip": B_flip,
            }
        )
    return pd.DataFrame(rows)


def case_weekly_probs(
    panel: pd.DataFrame,
    fit: PooledFit,
    config: Problem1Config,
    season: int,
    name: str,
    *,
    B: int = B_MECHANISM,
    alpha: float = 0.10,
) -> pd.DataFrame:
    """Per-eligible-week elimination probability under each rule (posterior mean
    and interval over B draws) plus the weekly rank-vs-pct reversal rate."""
    cfg = config_from_fit(fit, B=B)
    cache = DrawCache(panel, fit, cfg, max_B=B)
    rows: list[dict[str, Any]] = []
    for season_i, week in eligible_weeks(panel):
        if season_i != int(season):
            continue
        names, j = week_judge_vector(panel, season_i, week)
        if name not in names or not np.isfinite(j).all():
            continue
        p, aligned_names = cache.aligned(season_i, week, names, B)
        i = aligned_names.index(name)
        er = np.array([names.index(simulate_week(pb, j, names, "rank_direct")[0]) for pb in p])
        ep = np.array([names.index(simulate_week(pb, j, names, "pct_direct")[0]) for pb in p])
        lo_q, hi_q = alpha / 2.0, 1.0 - alpha / 2.0
        p_rank, p_pct = (er == i).astype(float), (ep == i).astype(float)
        rows.append(
            {
                "season": season_i,
                "week": week,
                "alive_n": len(names),
                "p_elim_rank": float(p_rank.mean()),
                "p_elim_pct": float(p_pct.mean()),
                f"p_elim_rank_lo_{int(alpha * 100):02d}": float(np.quantile(p_rank, lo_q)),
                f"p_elim_rank_hi_{int(alpha * 100):02d}": float(np.quantile(p_rank, hi_q)),
                f"p_elim_pct_lo_{int(alpha * 100):02d}": float(np.quantile(p_pct, lo_q)),
                f"p_elim_pct_hi_{int(alpha * 100):02d}": float(np.quantile(p_pct, hi_q)),
                "rev_rate_rank_vs_pct": float((er != ep).mean()),
                "B": B,
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Bottom-2 + judges' save reference metrics (src/b2_save_metrics.py)
# --------------------------------------------------------------------------- #
def _judge_vectors_legacy(
    g: pd.DataFrame, names: list[str], baseline_mode: str
) -> tuple[np.ndarray, np.ndarray]:
    """Era-appropriate judge vectors for the b2-save reference metrics.

    Faithful port of ``_judge_vectors`` in ``src/b2_save_metrics.py``: the
    score column is ``j_metric`` (percent-era weeks store judge percent, rank-era
    weeks judge-rank share); the pct mode normalizes it to a within-week share,
    the rank mode derives the share from the descending rank.
    """
    score_col = "j_metric"
    v = g.set_index("celebrity_name")[score_col].reindex(names).to_numpy(float)
    if baseline_mode == "pct":
        share = v / v.sum() if v.sum() != 0 else np.zeros_like(v)
        return judge_vectors_from_shares(share, "pct")
    return judge_vectors_from_shares(v, "rank")


def b2_case_metrics(
    panel: pd.DataFrame,
    fit: PooledFit,
    config: Problem1Config,
    cases: list[tuple[int, str]],
    *,
    B: int = B_MECHANISM,
    wJ: float = 0.5,
    wF: float = 0.5,
) -> pd.DataFrame:
    """Replay the reference b2-save CSV (``../data/metrics_b2_save.csv``).

    Exact port of ``compute_b2_save_metrics`` in ``src/b2_save_metrics.py`` for
    the named case contestants, sourcing posterior draws from ``DrawCache``.  Per
    ``(season, name, mode)``: ``p_b2`` (P(name in bottom two)), ``p_rev``
    (P(judges' save flips the eliminee)), ``p_rev_given_b2``, and the season-path
    trajectory deltas ``dE_T`` / ``dP_finals`` (draw-b aligned), exactly as the
    reference producer defines them.  The reference's trailing NaN group rows
    (an artifact of merging an empty ``contestant_types`` frame) are not
    reproduced; only the meaningful individual rows are emitted (D-20260901-09).
    """
    cfg = config_from_fit(fit, B=B)
    cache = DrawCache(panel, fit, cfg, max_B=B)
    df = panel.copy()
    Tmax_by_season = df.groupby("season")["week"].max().to_dict()

    rows: list[dict[str, Any]] = []
    for baseline_mode in ("rank", "pct"):
        for season, name in cases:
            season = int(season)
            Tmax = Tmax_by_season.get(season)
            if Tmax is None:
                continue
            weeks = sorted(
                wk
                for wk in df[df["season"] == season]["week"].unique()
                if wk < Tmax
                and eligible_week_selector(df[(df["season"] == season) & (df["week"] == wk)])
            )

            p_b2_hits = 0
            p_rev_given_b2_hits = 0
            p_rev_hits = 0
            denom_b2 = 0
            T_i = 0
            n_draws: int | None = None

            for wk in weeks:
                g = df[(df["season"] == season) & (df["week"] == wk) & df["alive"]].copy()
                if g.empty:
                    continue
                names = g["celebrity_name"].astype(str).tolist()
                if name not in names:
                    continue
                store_names, p_full = cache.week(season, wk)
                name_idx = {n: i for i, n in enumerate(store_names)}
                cols = [name_idx[n] for n in names]
                p_draws = p_full[:B, cols]
                n_draws = p_draws.shape[0]
                g = _align_to_names(g, names)
                judge_pct, judge_rank = _judge_vectors_legacy(g, names, baseline_mode)
                for b in range(n_draws):
                    _, bottom2, elim_base, elim_save = risk_and_bottom2(
                        p_draws[b], names, judge_pct, judge_rank, baseline_mode, wJ=wJ, wF=wF
                    )
                    if elim_base is None or bottom2 is None:
                        continue
                    if name in bottom2:
                        p_b2_hits += 1
                        denom_b2 += 1
                        if elim_base != elim_save:
                            p_rev_given_b2_hits += 1
                    if elim_base != elim_save:
                        p_rev_hits += 1
                T_i += 1

            if T_i == 0 or n_draws is None:
                continue
            p_b2 = p_b2_hits / (T_i * n_draws)
            p_rev = p_rev_hits / (T_i * n_draws)
            p_rev_given_b2 = p_rev_given_b2_hits / denom_b2 if denom_b2 > 0 else np.nan

            dE_T, dP_finals = _b2_season_path_deltas(
                panel, cache, season, weeks, name, baseline_mode, Tmax, B, wJ, wF
            )
            rows.append(
                {
                    "season": season,
                    "celebrity_name": name,
                    "baseline_mode": baseline_mode,
                    "p_b2": p_b2,
                    "p_rev_given_b2": p_rev_given_b2,
                    "p_rev": p_rev,
                    "dE_T": dE_T,
                    "dP_finals": dP_finals,
                    "n_weeks": T_i,
                    "B": int(n_draws),
                    "denom_b2": int(denom_b2),
                    "Tmax": int(Tmax),
                    "unit_type": "individual",
                    "unit_id": name,
                }
            )
    return pd.DataFrame(rows)


def eligible_week_selector(df_week: pd.DataFrame) -> bool:
    """Legacy ``eligible_week_selector``: non-final, single-elim, alive > 2."""
    if "is_final_week" in df_week.columns and df_week["is_final_week"].iloc[0]:
        return False
    if "elim_this_week_end" in df_week.columns:
        if df_week["elim_this_week_end"].sum() != 1:
            return False
    if "alive" in df_week.columns:
        return bool(df_week["alive"].sum() > 2)
    return True


def _align_to_names(g: pd.DataFrame, names: list[str]) -> pd.DataFrame:
    order = pd.Index(names)
    d = g.copy()
    d["celebrity_name"] = pd.Categorical(
        d["celebrity_name"].astype(str), categories=order, ordered=True
    )
    return d.sort_values("celebrity_name")


def _b2_season_path_deltas(
    panel: pd.DataFrame,
    cache: DrawCache,
    season: int,
    weeks: list[int],
    name: str,
    baseline_mode: str,
    Tmax: int,
    B: int,
    wJ: float,
    wF: float,
) -> tuple[float, float]:
    """Season-path trajectory deltas for the b2-save reference metrics.

    Draw-b aligned simulation: for every draw ``b`` the season is replayed under
    the direct risk rule (``elim_base``) and under the bottom-2 + judges' save
    rule (``elim_save``); ``t_base`` / ``t_save`` are the elimination weeks of
    ``name`` (or ``Tmax`` when it reaches the final).  Returns
    ``(dE_T, dP_finals)`` as the mean differences over draws.  Ported exactly
    from the trajectory block of ``compute_b2_save_metrics`` — note the alive set
    is trimmed by ``celebrity_name.isin(...)`` on actual week rows, not by the
    ``alive`` flag, and weeks with fewer than 3 survivors are skipped.
    """
    if not weeks:
        return np.nan, np.nan
    alive0 = set(
        panel[(panel["season"] == season) & (panel["week"] == weeks[0]) & panel["alive"]][
            "celebrity_name"
        ].astype(str)
    )
    t_base_draws = np.full(B, float(Tmax), dtype=float)
    t_save_draws = np.full(B, float(Tmax), dtype=float)
    # Per-draw alive sets (mutable per week).
    alive_base: list[set[str]] = [set(alive0) for _ in range(B)]
    alive_save: list[set[str]] = [set(alive0) for _ in range(B)]

    # Per-week draws are fixed for the whole week and identical across the
    # base/save paths and across draws; only the alive subset varies per draw.
    store_by_week: dict[int, tuple[list[str], np.ndarray]] = {}
    for wk in weeks:
        store_by_week[wk] = cache.week(season, wk)

    for wk in weeks:
        store_names, p_full = store_by_week[wk]
        name_idx = {n: i for i, n in enumerate(store_names)}
        # All panel rows for the week, exactly like the legacy
        # ``df[(season) & (week==wk) & isin(alive)]``.  Data-eliminated
        # contestants still carry a row (``alive`` flag False), so the
        # ``len(g) >= 3`` guard below uses the *untrimmed* roster exactly as the
        # reference producer does; stale contestants are then trimmed away by the
        # draws-column intersection (legacy ``get_week_draws`` +
        # ``g_base[g_base[...].isin(names)]``), which also matches the
        # elimination-week semantics (D-20260901-09).
        present = panel[(panel["season"] == season) & (panel["week"] == wk)].copy()
        for label, alive_sets, t_out in (
            ("base", alive_base, t_base_draws),
            ("save", alive_save, t_save_draws),
        ):
            for b in range(B):
                g = present[present["celebrity_name"].astype(str).isin(alive_sets[b])].copy()
                if len(g) < 3:
                    continue
                names = g["celebrity_name"].astype(str).tolist()
                # Trim to the draws' columns exactly like the legacy
                # ``get_week_draws`` + ``g_base[g_base[...].isin(names)]``.
                cols = [name_idx[n] for n in names if n in name_idx]
                if not cols:
                    continue
                names = [n for n in names if n in name_idx]
                g = g[g["celebrity_name"].astype(str).isin(names)].copy()
                g = _align_to_names(g, names)
                p_b = p_full[b, cols]
                judge_pct, judge_rank = _judge_vectors_legacy(g, names, baseline_mode)
                _, _, elim_base, elim_save = risk_and_bottom2(
                    p_b, names, judge_pct, judge_rank, baseline_mode, wJ=wJ, wF=wF
                )
                if label == "base":
                    if elim_base is None:
                        continue
                    if name in alive_sets[b] and elim_base == name and t_out[b] == Tmax:
                        t_out[b] = float(wk)
                    alive_sets[b].discard(elim_base)
                else:
                    if elim_save is None:
                        continue
                    if name in alive_sets[b] and elim_save == name and t_out[b] == Tmax:
                        t_out[b] = float(wk)
                    alive_sets[b].discard(elim_save)
    dE_T = float(np.mean(t_save_draws) - np.mean(t_base_draws))
    dP_finals = float(np.mean(t_save_draws == Tmax) - np.mean(t_base_draws == Tmax))
    return dE_T, dP_finals
