"""Pure elimination-rule functions for Problem 2.

Implements the paper's rank and percentage rules and the Bottom-2 + judges'
save mechanism (``paper_Latex/2107542.tex``, Problem 2) as separately tested
pure functions, together with the exact tie-breaking semantics of the legacy
notebook ``src/2_rank_vs_pct_cross_season.ipynb`` (cells 20/29/34) and the
reference b2-save metrics producer ``src/b2_save_metrics.py``.

Conventions
-----------
- ``j`` is the judge signal. For the paper formulas and the case-study replay
  it is the within-week share of raw judge scores (``judge_percent`` = ``T /
  sum T``); ranks and argmin are invariant to that normalization.
- ``p`` is a fan-support vector (a single posterior draw or the posterior
  mean). Ranks are descending so rank 1 = best (largest value).
- Rank rule (paper Eq. 1): ``argmax_i(rank(-T_i) + rank(-p_i))``.
- Percentage rule (paper Eq. 2): ``argmin_i(T_i/sum T + p_i)``.
- Bottom-2 + judges' save (paper): ``B = arg two-min_i S_i`` then
  ``e_B2 = argmin_{i in B} J_i`` where ``S`` is the mechanism score (rank sum
  under ``rank``, weighted share sum under ``pct``).

Tie-breaking policy (D-20260901-09)
-----------------------------------
- The paper-formula ``argmax``/``argmin`` helpers use first-index ties on the
  caller-provided (``celebrity_name``-sorted) ordering.
- ``simulate_week`` reproduces the legacy notebook's lexsort exactly
  (``(name_key, p, j, score)`` keys; ``score`` primary) so a bottom-2 save
  eliminates the worse-judged contestant with the fan share and name as
  secondary ties.
- ``compute_risk_and_bottom2`` reproduces ``src/b2_save_metrics.py`` exactly
  (primary ``-risk`` then ``judge_pct`` then ``p_draw`` then name), which is
  the producer of ``../data/metrics_b2_save.csv``.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

# Mechanisms accepted by ``simulate_week``.
MECHANISMS = ("rank_direct", "rank_bottom2", "pct_direct", "pct_bottom2")


def descending_rank(x: np.ndarray) -> np.ndarray:
    """Rank ``x`` so the largest value gets rank 1 (pandas 'average' method).

    Matches the legacy ``_rank_desc`` used throughout the Problem 2 notebooks
    and ``src/b2_save_metrics.py``.
    """
    return np.asarray(
        pd.Series(np.asarray(x, dtype=float))
        .rank(ascending=False, method="average")
        .to_numpy(dtype=float)
    )


def ascending_rank(x: np.ndarray) -> np.ndarray:
    """Rank ``x`` so the smallest value gets rank 1 (pandas 'average' method)."""
    return np.asarray(
        pd.Series(np.asarray(x, dtype=float))
        .rank(ascending=True, method="average")
        .to_numpy(dtype=float)
    )


def _order_by(
    score: np.ndarray,
    j: np.ndarray,
    p: np.ndarray,
    names: np.ndarray,
    *,
    worst: str,
) -> np.ndarray:
    """Legacy ordering: worst first, with ``(j, p, name)`` secondary ties.

    ``worst='min'`` orders ascending ``score`` (smallest worst); ``'max'``
    orders descending (largest worst).  Matches ``_order_by`` in notebook cell
    29 exactly: keys ``(name_key, p, j, score_key)`` with ``score`` primary.
    """
    names = np.asarray(names)
    name_key = np.argsort(names)
    score_key = score if worst == "min" else -score
    return np.lexsort((name_key, np.asarray(p), np.asarray(j), score_key))


def _worst_among_bottom2(
    order: np.ndarray, j: np.ndarray, p: np.ndarray, names: np.ndarray
) -> tuple[int, str]:
    """Among the bottom-2, the one with the worse judge signal.

    ``j`` is oriented so *smaller is worse*.  Tie-breaks fan share (smaller
    worse) then name, matching ``simulate_week`` and ``b2_save_metrics``.
    """
    b2_idx = order[:2]
    jvals = j[b2_idx]
    pvals = p[b2_idx]
    b2_names = names[b2_idx]
    elim_idx = b2_idx[np.lexsort((np.argsort(b2_names), pvals, jvals))[0]]
    return int(elim_idx), names[elim_idx]


def simulate_week(
    p: np.ndarray,
    j: np.ndarray,
    names: Sequence[str] | np.ndarray,
    mechanism: str,
    *,
    wJ: float = 0.5,
    wF: float = 0.5,
) -> tuple[str, list[str]]:
    """Simulate one week under a mechanism: returns ``(elim, bottom2)``.

    Faithful port of the legacy ``simulate_week`` (notebook cell 29).  Under
    ``rank*`` the score is the sum of descending judge and fan ranks (worst is
    the *largest* sum); under ``pct*`` the score is ``wJ*J + wF*p`` (worst is
    the *smallest*).  ``*_direct`` eliminates the single worst contestant;
    ``*_bottom2`` first isolates the bottom two and then eliminates the worse
    judge among them (the judges' save).
    """
    p = np.asarray(p, dtype=float)
    j = np.asarray(j, dtype=float)
    names = np.asarray(names)
    if mechanism not in MECHANISMS:
        raise ValueError(f"mechanism must be one of {MECHANISMS}, got {mechanism!r}")
    if mechanism.startswith("rank"):
        rj = descending_rank(j)
        rf = descending_rank(p)
        score = rj + rf
        order = _order_by(score, j, p, names, worst="max")
    else:
        score = wJ * j + wF * p
        order = _order_by(score, j, p, names, worst="min")

    bottom2 = [str(n) for n in names[order[:2]]]
    if mechanism.endswith("direct"):
        elim = str(names[order[0]])
    else:
        # Judges' save: eliminate the worse judge among the bottom two.  The
        # legacy compares raw ``j`` (smaller = worse) for both mechanisms.
        _, elim = _worst_among_bottom2(order, j, p, names)
    return elim, bottom2


# --------------------------------------------------------------------------- #
# Paper-formula point helpers (first-index tie-break on name-sorted input)
# --------------------------------------------------------------------------- #
def elim_rank_idx(judge_score: np.ndarray, fan_share: np.ndarray) -> int:
    """Paper Eq. 1: ``argmax_i(rank(-T_i) + rank(-p_i))``.

    ``judge_score`` may be the raw judge total or any within-week positive
    multiple of it; ranks are invariant to that scale.
    """
    rj = descending_rank(judge_score)
    rf = descending_rank(fan_share)
    return int(np.argmax(rj + rf))


def elim_pct_idx(judge_share: np.ndarray, fan_share: np.ndarray) -> int:
    """Paper Eq. 2: ``argmin_i(T_i/sum T + p_i)``.

    ``judge_share`` is the within-week share ``T_i / sum_k T_k``.
    """
    return int(np.argmin(np.asarray(judge_share, dtype=float) + np.asarray(fan_share, dtype=float)))


def judge_worst_idx(judge_score: np.ndarray) -> int:
    """Index of the judge-only loser ``argmin_k T_k`` (paper Override metric)."""
    return int(np.argmin(np.asarray(judge_score, dtype=float)))


def fan_worst_idx(fan_share: np.ndarray) -> int:
    """Index of the least fan-supported contestant ``argmin_k p_k``."""
    return int(np.argmin(np.asarray(fan_share, dtype=float)))


# --------------------------------------------------------------------------- #
# Bottom-2 + judges' save under the reference b2-save metrics semantics
# --------------------------------------------------------------------------- #
def risk_and_bottom2(
    p_draw: np.ndarray,
    names: Sequence[str] | np.ndarray,
    judge_pct: np.ndarray,
    judge_rank: np.ndarray,
    baseline_mode: str,
    *,
    wJ: float = 0.5,
    wF: float = 0.5,
) -> tuple[np.ndarray | None, list[str] | None, str | None, str | None]:
    """Exact port of ``compute_risk_and_bottom2`` (src/b2_save_metrics.py).

    Returns ``(risk, bottom2, elim_base, elim_save)``.  ``baseline_mode`` is
    ``'pct'`` or ``'rank'``; ``judge_pct`` is a within-week share and
    ``judge_rank`` a descending rank of the judge signal (1 = best).
    """
    names = np.asarray(names)
    p_draw = np.asarray(p_draw, dtype=float)
    judge_pct = np.asarray(judge_pct, dtype=float)
    judge_rank = np.asarray(judge_rank, dtype=float)
    n = len(names)
    if n == 0 or len(p_draw) != n or len(judge_pct) != n or len(judge_rank) != n:
        return None, None, None, None

    if baseline_mode == "pct":
        risk = wJ * (1.0 - judge_pct) + wF * (1.0 - p_draw)
        judge_signal = judge_pct
    else:
        fan_rank = descending_rank(p_draw)
        risk = wJ * judge_rank + wF * fan_rank
        judge_signal = -judge_rank

    name_key = np.argsort(names)
    order = np.lexsort((name_key, p_draw, judge_pct, -risk))
    if order.size == 0:
        return None, None, None, None

    bottom2_idx = order[:2]
    bottom2 = [str(n) for n in names[bottom2_idx]]
    elim_base = str(names[order[0]])

    b2_j = judge_signal[bottom2_idx]
    b2_p = p_draw[bottom2_idx]
    b2_names = names[bottom2_idx]
    elim_idx = bottom2_idx[np.lexsort((np.argsort(b2_names), b2_p, b2_j))[0]]
    elim_save = str(names[elim_idx])
    return risk, bottom2, elim_base, elim_save


def judge_vectors_from_shares(
    judge_pct: np.ndarray, baseline_mode: str
) -> tuple[np.ndarray, np.ndarray]:
    """Derive ``(judge_pct, judge_rank)`` from a within-week share.

    For ``'pct'`` the share is used directly and its descending rank becomes
    ``judge_rank``; for ``'rank'`` the descending rank is primary and the
    derived share is ``1 - (rank - 1) / (n - 1)`` — exactly the legacy
    ``_judge_vectors`` in src/b2_save_metrics.py.
    """
    judge_pct = np.asarray(judge_pct, dtype=float)
    if baseline_mode == "pct":
        judge_rank = descending_rank(judge_pct)
        return judge_pct, judge_rank
    judge_rank = descending_rank(judge_pct)  # rank of the *share* is the same
    n = len(judge_rank)
    judge_pct = 1.0 - (judge_rank - 1.0) / max(n - 1, 1)
    return judge_pct, judge_rank
