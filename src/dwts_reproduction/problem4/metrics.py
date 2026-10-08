"""Mechanism metrics over simulator outputs (Shock_k, survival).

Implements the paper's two headline criteria on saved simulator frames:

- **Shock_k** (paper Eq. p. ``2107542.tex`` line 883): the *technical shock*
  rate ``Shock_k = Pr(elim(s,t) in Top-k by judges at week (s,t))``, computed
  over the elimination events of every simulated week.  A "fair" rule keeps it
  small (judge-strong couples are rarely eliminated).
- **Cumulative survival** ``S(t)`` per (scheme, archetype): the unconditional
  survival function used in Fig. 8 (paper lines 917/933/1003).

Judge rank: V1 sim_detail already records ``judge_rank`` (descending average
rank, 1 = best).  V2 sim_detail only records ``judge_score``, so
:func:`add_judge_rank` recomputes the identical rank within each
(scheme, sim, season, week) group.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULT_K = 3


def add_judge_rank(df: pd.DataFrame, detail_kind: str = "V1") -> pd.DataFrame:
    """Guarantee a ``judge_rank`` column (1 = best judge) on a detail frame.

    V1 frames already carry it; V2 frames recompute it from ``judge_score``
    via descending average rank inside each (scheme, sim, season, week) group —
    the same definition ``rank_desc`` uses everywhere in this package.
    """
    if detail_kind == "V1" and "judge_rank" in df.columns:
        return df
    df = df.copy()
    df["judge_rank"] = df.groupby(["scheme", "sim", "season", "week"])["judge_score"].rank(
        ascending=False, method="average"
    )
    return df


def elimination_events(df: pd.DataFrame) -> pd.DataFrame:
    """Rows where a contestant was eliminated this week."""
    return df[df["eliminated_this_week"].astype(bool)].copy()


def shock_rates(
    df: pd.DataFrame,
    ks: tuple[int, ...] = (1, 2, 3),
    detail_kind: str = "V1",
    by_archetype: bool = False,
) -> pd.DataFrame:
    """Per-scheme (and optionally archetype) technical-shock rates ``Shock_k``.

    For each elimination event, ``Shock_k`` is 1 when the eliminated
    contestant's judge rank is ``<= k``.  Columns: ``scheme`` [, ``archetype``]
    then ``shock_k{k}`` for every k in ``ks``.
    """
    df = add_judge_rank(df, detail_kind)
    ev = elimination_events(df)
    if ev.empty:
        cols = ["scheme"] + ([f"shock_k{k}" for k in ks])
        if by_archetype:
            cols = ["scheme", "archetype"] + [f"shock_k{k}" for k in ks]
        return pd.DataFrame(columns=cols)
    groups = ["scheme"] + (["archetype"] if by_archetype else [])
    out = (
        ev.groupby(groups)
        .apply(
            lambda g: pd.Series(
                {f"shock_k{k}": float(np.mean(g["judge_rank"].to_numpy() <= k)) for k in ks}
            ),
            include_groups=False,
        )
        .reset_index()
    )
    return out


def cum_alive_rate(summary: pd.DataFrame) -> pd.DataFrame:
    """Cumulative survival ``S(t)`` per (scheme, archetype).

    ``alive_rate`` in a summary row is the mean of ``~eliminated_this_week``
    across sims in that (scheme, week, archetype) cell, i.e. one minus the
    weekly elimination rate; ``S(t)`` is the cumulative product over weeks.
    """
    if summary.empty:
        return summary
    out = summary.copy()
    out = out.sort_values(["scheme", "archetype", "week"])
    out["cum_alive_rate"] = out.groupby(["scheme", "archetype"])["alive_rate"].cumprod()
    return out.reset_index(drop=True)


def add_nominated(df: pd.DataFrame, K: int = DEFAULT_K) -> pd.DataFrame:
    """Mark judge-gate nominees (``nominated``) on a V1 detail frame.

    The paper's judge gate (``B^J_{s,t} = arg K-min J``, paper line 903)
    nominates the ``K`` worst-judged contestants; fans then eliminate the
    least-supported nominee.  The gate is active in S3 all season and in S2
    only in the late stage (paper line 911); finale weeks have no gate.  A
    ``nominated`` flag is added for those rows so ``pre-filter immunity``
    (paper line 942) can be checked: popular contestants should rarely enter
    the nominee set even under full-season activation.
    """
    df = df.copy()
    df["nominated"] = False
    if "stage" not in df.columns:
        # Synthetic fixtures may omit the stage column; fall back to S3 only.
        gate = df[df["scheme"] == "S3"].copy()
    else:
        # The finale is fixed (no elimination, paper line 918), so the gate's
        # risk set only applies to non-final weeks; a contestant who reaches
        # the finale must not be counted as "nominated" there.
        gate = df[
            (df["stage"] != "final")
            & ((df["scheme"] == "S3") | ((df["scheme"] == "S2") & (df["stage"] == "late")))
        ].copy()
    if gate.empty:
        return df

    def _mark(g: pd.DataFrame) -> pd.DataFrame:
        g = g.copy()
        k = min(K, len(g))
        worst = np.argsort(g["judge_rank"].to_numpy())[-k:]
        flag = np.zeros(len(g), dtype=bool)
        flag[worst] = True
        g["nominated"] = flag
        return g

    marked = gate.groupby(["scheme", "sim", "season", "week"], group_keys=False).apply(
        _mark, include_groups=False
    )
    df.loc[marked.index, "nominated"] = marked["nominated"]
    return df
