"""Problem 2 elimination-rule tests.

Hand-computed fixtures pin the paper-formula helpers (Eqs. 1-5 of
``paper_Latex/2107542.tex``), the legacy ``simulate_week`` lexsort semantics
(notebook cell 29), and the reference b2-save producer
(``compute_risk_and_bottom2`` in ``src/b2_save_metrics.py``).  A hand-chosen
panel demonstrates that the rank and percentage rules can pick *different*
eliminatees — the phenomenon the paper's ``DR`` / ``Flip`` metrics measure.
"""

from __future__ import annotations

import numpy as np
import pytest

from dwts_reproduction.problem2 import (
    ascending_rank,
    descending_rank,
    elim_pct_idx,
    elim_rank_idx,
    fan_worst_idx,
    judge_vectors_from_shares,
    judge_worst_idx,
    risk_and_bottom2,
    simulate_week,
)


# --------------------------------------------------------------------------- #
# Ranks
# --------------------------------------------------------------------------- #
def test_descending_rank_largest_is_first_with_average_ties():
    x = np.array([3.0, 10.0, 10.0, 1.0])
    r = descending_rank(x)
    # 10s share ranks 1 and 2 -> 1.5 each; then 3 -> 3, 1 -> 4.
    np.testing.assert_allclose(r, [3.0, 1.5, 1.5, 4.0])
    assert ascending_rank(x)[0] == pytest.approx(2.0)  # 3 is second-smallest


# --------------------------------------------------------------------------- #
# Paper-formula helpers (Eqs. 1-5)
# --------------------------------------------------------------------------- #
def test_elim_rank_idx_paper_eq1():
    # judge_score may be the raw total; ranks are invariant to positive scaling.
    r1 = elim_rank_idx(np.array([8.0, 5.0, 3.0]), np.array([0.3, 0.5, 0.2]))
    r2 = elim_rank_idx(np.array([80.0, 50.0, 30.0]), np.array([0.3, 0.5, 0.2]))
    assert r1 == 2 and r2 == 2
    # descending_rank([8,5,3])=[1,2,3]; desc([0.3,0.5,0.2])=[2,1,3]; sum=[3,3,6].
    assert r1 == 2


def test_elim_pct_idx_paper_eq2():
    # share + p: [0.5,0.3125,0.1875]+[0.3,0.5,0.2] = [0.8,0.8125,0.3875] -> argmin 2.
    assert elim_pct_idx(np.array([0.5, 0.3125, 0.1875]), np.array([0.3, 0.5, 0.2])) == 2


def test_judge_and_fan_worst_helpers():
    j = np.array([8.0, 5.0, 3.0])
    p = np.array([0.3, 0.5, 0.2])
    assert judge_worst_idx(j) == 2  # argmin T_k (Eq. 4 reference)
    assert fan_worst_idx(p) == 2  # argmin p_k (Eq. 5 reference)


def test_rank_and_pct_rules_diverge():
    """A hand-chosen panel where the two paper rules eliminate different people.

    j=[12,23,10,7], p=[0.822,0.028,0.073,0.077]:
    rank rule eliminates index 2 (rank-sum [3,5,6,6]); the pct rule index 3
    (share-sum [1.053,0.470,0.265,0.212]).  This divergence is exactly what
    ``DR``/``Flip`` measure.
    """
    j = np.array([12.0, 23.0, 10.0, 7.0])
    p = np.array([0.822, 0.028, 0.073, 0.077])
    assert elim_rank_idx(j, p) == 2
    assert elim_pct_idx(j / j.sum(), p) == 3
    assert judge_worst_idx(j) == 3
    assert fan_worst_idx(p) == 1


# --------------------------------------------------------------------------- #
# Legacy simulate_week (notebook cell 29 lexsort semantics)
# --------------------------------------------------------------------------- #
@pytest.fixture
def week_fixture():
    names = np.array(["Alice", "Bob", "Carol"])
    j = np.array([10.0, 5.0, 3.0])
    p = np.array([0.2, 0.5, 0.3])
    return names, j, p


def test_simulate_week_rank_direct(week_fixture):
    names, j, p = week_fixture
    # score = rj + rf = [4,3,5]; worst='max' -> Carol first (score 5).
    elim, bottom2 = simulate_week(p, j, names, "rank_direct")
    assert elim == "Carol"
    assert bottom2 == ["Carol", "Alice"]


def test_simulate_week_rank_bottom2(week_fixture):
    names, j, p = week_fixture
    # Judges' save: of {Carol, Alice}, Carol has the worse judge score (3 < 10).
    elim, bottom2 = simulate_week(p, j, names, "rank_bottom2")
    assert elim == "Carol"
    assert bottom2 == ["Carol", "Alice"]


def test_simulate_week_pct_direct(week_fixture):
    names, j, p = week_fixture
    # score = 0.5 j + 0.5 p = [5.1,2.75,1.65]; worst='min' -> Carol first.
    elim, bottom2 = simulate_week(p, j, names, "pct_direct")
    assert elim == "Carol"
    assert bottom2 == ["Carol", "Bob"]


def test_simulate_week_pct_bottom2(week_fixture):
    names, j, p = week_fixture
    elim, bottom2 = simulate_week(p, j, names, "pct_bottom2")
    assert elim == "Carol"
    assert bottom2 == ["Carol", "Bob"]


def test_simulate_week_validates_mechanism(week_fixture):
    names, j, p = week_fixture
    with pytest.raises(ValueError):
        simulate_week(p, j, names, "rank_something")


# --------------------------------------------------------------------------- #
# Reference b2-save producer (src/b2_save_metrics.py semantics)
# --------------------------------------------------------------------------- #
def test_risk_and_bottom2_pct_mode():
    names = np.array(["A", "B", "C"])
    judge_pct = np.array([0.5, 0.3, 0.2])
    judge_rank = np.array([1.0, 2.0, 3.0])
    p_draw = np.array([0.3, 0.4, 0.3])
    risk, bottom2, elim_base, elim_save = risk_and_bottom2(
        p_draw, names, judge_pct, judge_rank, "pct", wJ=0.5, wF=0.5
    )
    # risk = 0.5(1-J) + 0.5(1-p) = [0.60,0.65,0.75]; C most at risk.
    np.testing.assert_allclose(risk, [0.60, 0.65, 0.75])
    assert bottom2 == ["C", "B"]
    assert elim_base == "C"
    assert elim_save == "C"  # C also has the worse judge share of {B, C}


def test_risk_and_bottom2_rank_mode():
    names = np.array(["A", "B", "C"])
    judge_pct = np.array([0.5, 0.3, 0.2])
    judge_rank = np.array([1.0, 2.0, 3.0])
    p_draw = np.array([0.3, 0.4, 0.3])
    risk, bottom2, elim_base, elim_save = risk_and_bottom2(
        p_draw, names, judge_pct, judge_rank, "rank", wJ=0.5, wF=0.5
    )
    # risk = 0.5*judge_rank + 0.5*desc_rank(p) = [1.75,1.50,2.75]; C most at risk.
    np.testing.assert_allclose(risk, [1.75, 1.50, 2.75])
    assert bottom2 == ["C", "A"]
    assert elim_base == "C"
    assert elim_save == "C"  # C is also the worse-judged of {C, A}


def test_judge_vectors_from_shares():
    share = np.array([0.5, 0.3, 0.2])
    pct_j, pct_r = judge_vectors_from_shares(share, "pct")
    np.testing.assert_allclose(pct_j, share)
    np.testing.assert_allclose(pct_r, [1.0, 2.0, 3.0])

    rank_j, rank_r = judge_vectors_from_shares(share, "rank")
    np.testing.assert_allclose(rank_r, [1.0, 2.0, 3.0])
    np.testing.assert_allclose(rank_j, [1.0, 0.5, 0.0])  # 1 - (r-1)/(n-1)


def test_bottom2_save_worse_judge_is_identical_across_modes():
    """Judges' save is mode-consistent: both modes eliminate the worse judge
    of the bottom two, so they save the same contestant whenever the
    bottom-two pair coincides.

    ``elim_base`` is deliberately NOT compared across modes.  The rank risk
    ``wJ*jr + wF*fr`` and the pct risk ``wJ*(1-J) + wF*(1-p)`` are different
    objective functions that legitimately pick different direct eliminatees —
    that divergence *is* the paper's rank-vs-percentage premise (decision
    D-20260901-09; see also ``test_rank_and_pct_rules_diverge``).  What is
    invariant is the save rule (``compute_risk_and_bottom2`` in
    src/b2_save_metrics.py): of the bottom two, the contestant with the worse
    judge signal (larger ``judge_rank``, smaller judge share) is eliminated in
    both modes, with fan share then name as tie-breaks.
    """
    rng = np.random.default_rng(7)
    any_base_divergence = False
    for _ in range(20):
        n = 5
        j = rng.uniform(1, 20, n)
        p = rng.dirichlet(np.ones(n))
        judge_pct = j / j.sum()
        _, judge_rank = judge_vectors_from_shares(judge_pct, "pct")
        names = np.arange(n).astype(str)
        by_name = {nm: i for i, nm in enumerate(names)}

        _, r_b2, r_base, r_save = risk_and_bottom2(p, names, judge_pct, judge_rank, "rank")
        _, p_b2, p_base, p_save = risk_and_bottom2(p, names, judge_pct, judge_rank, "pct")

        # Within-mode save rule: the worse-judged contestant of the bottom-two
        # pair is eliminated (larger judge_rank; fan share tie-break).
        for b2, save in ((r_b2, r_save), (p_b2, p_save)):
            i, k = (by_name[nm] for nm in b2)
            worse = i if judge_rank[i] > judge_rank[k] else k
            if judge_rank[i] == judge_rank[k]:
                worse = i if p[i] < p[k] else k
            assert save == names[worse]

        # Cross-mode: rank vs pct baseline legitimately diverge on some draws…
        if r_base != p_base:
            any_base_divergence = True
        # …but when both modes isolate the same bottom-two pair, the judges'
        # save agrees (identical judge ordering).
        if r_b2 == p_b2:
            assert r_save == p_save

    # The fixed seed-7 draw set must exhibit the rank/pct baseline divergence
    # the paper's DR/Flip metrics exist to measure.
    assert any_base_divergence
