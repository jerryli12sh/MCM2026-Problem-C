"""Problem 4: mechanism design and optimization (paper lines 871-1060).

Two Monte-Carlo season simulators driven by posterior fan-share draws from the
Problem 1 pooled model (``q_hat`` -> ``alpha = kappa*q_hat`` ->
``rng.dirichlet(alpha)``):

- :mod:`.v1` — the paper's Mechanism I (schemes ``S1``/``S2``/``S3``), exact
  port of the legacy ``../src/season_simulator.py``.
- :mod:`.v2` — the paper's Mechanism II (schemes ``V4``/``V5``), exact port of
  the legacy ``../src/season_simulator2.py``; the paper labels the two compared
  mechanisms "V0" (baseline) and "V2" (proposed), the legacy code calls them
  ``V4``/``V5`` (D-20260901-18).

Shared helpers live in :mod:`.features`; named-case studies in :mod:`.cases`;
mechanism metrics (``Shock_k``, survival, gate nomination) in :mod:`.metrics`;
the paper's P-084/P-085/P-086 claim checks in :mod:`.claims`; the composite
paper Figure 8 charts in :mod:`.figures`.

Track: Problem 4 is Track P primary.  The V2 mechanism and ``Shock_k`` are
shared with the review's "design a new mechanism" section (review_all.md), so
rows that involve them carry track ``"P/R"``.  There is no separate Track R
mechanism for Problem 4 (no review-critique rows) — the paper-vs-review
comparison lives in the P-086 claim check.
"""

from __future__ import annotations
