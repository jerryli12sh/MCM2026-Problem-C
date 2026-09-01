"""Problem-4 claim checks (paper P-084, P-085, P-086).

Reproduces the paper's mechanism-design claims as checkable rows over saved
simulator detail frames.  The paper states these qualitatively; each row
operationalizes a claim with a metric, the reproduced value, a target derived
from the paper's prose, and a pass/fail status.  Nothing here claims that the
simulated fan shares are ground truth — they are posterior draws conditioned on
observed outcomes (see :mod:`.features`).

Track: Problem 4 is Track P primary (the paper's mechanism design), but the V2
mechanism and the ``Shock_k`` metric are shared with the review, so rows that
involve them carry track ``"P/R"`` (D-20260901-18).
"""

from __future__ import annotations

import pandas as pd

from .cases import add_week_ranks, case_summary_table_rank, case_summary_table_score
from .features import POPULARITY_CASES
from .metrics import add_nominated

CLAIM_COLUMNS = [
    "claim_id",
    "track",
    "statement",
    "metric",
    "value",
    "target",
    "status",
]


def _row(claim_id: str, track: str, statement: str, metric: str, value, target: str, status: str):
    return {
        "claim_id": claim_id,
        "track": track,
        "statement": statement,
        "metric": metric,
        "value": value,
        "target": target,
        "status": status,
    }


def exit_week_by_scheme(df_case: pd.DataFrame, threshold: float = 0.5) -> dict[str, float | None]:
    """First week whose elimination rate crosses ``threshold``, per scheme.

    Mirrors the V2 legacy case script's ``ELIM_THRESHOLD`` split; ``None`` when
    the case is never eliminated at rate >= threshold.
    """
    out: dict[str, float | None] = {}
    for scheme, g in df_case.groupby("scheme"):
        by_week = g.groupby("week")["eliminated_this_week"].mean()
        candidates = by_week[by_week >= threshold]
        out[scheme] = float(candidates.index[0]) if not candidates.empty else None
    return out


def case_placement_table(df_detail: pd.DataFrame, detail_kind: str = "V1") -> pd.DataFrame:
    """Per (season, celebrity_name, scheme) mean_rank / final survival.

    Reuses the legacy case-summary recipes: V1 reads ``combined_rank``, V2 adds
    ``rank_S`` from ``score_S``.  Returns the per-scheme rows for every named
    case present in the detail frame.
    """
    if detail_kind == "V2":
        # Rank the full frame once (within scheme/sim/season/week), then group;
        # matches legacy ``sim_rank_trend_cases_2.main``.  Ranking a
        # single-contestant slice would collapse ``rank_S`` to 1.
        df_detail = add_week_ranks(df_detail)
    rows = []
    for (season, name), g in df_detail.groupby(["season", "celebrity_name"]):
        n_sims = int(g["sim"].nunique())
        if detail_kind == "V1":
            tbl = case_summary_table_rank(g, n_sims)
        else:
            tbl = case_summary_table_score(g, n_sims)
        tbl.insert(0, "celebrity_name", name)
        tbl.insert(0, "season", int(season))
        rows.append(tbl)
    if not rows:
        return pd.DataFrame(columns=["season", "celebrity_name", "scheme"])
    return pd.concat(rows, ignore_index=True)


def nominee_rates(
    v1_detail: pd.DataFrame, cases: tuple[tuple[int, str], ...] = POPULARITY_CASES
) -> pd.DataFrame:
    """Per popularity case: S3 nominee (risk-set) rate, S1/S3 final survival."""
    df = add_nominated(v1_detail)
    rows = []
    for season, name in cases:
        sub = df[(df["season"] == season) & (df["celebrity_name"] == name)]
        if sub.empty:
            continue
        s3 = sub[sub["scheme"] == "S3"]
        nominee_rate = (
            float(s3["nominated"].mean()) if not s3.empty and "nominated" in s3 else float("nan")
        )
        placed = case_placement_table(sub, detail_kind="V1")
        final = dict(
            zip(placed["scheme"], placed["final_alive_rate"], strict=True)
            if not placed.empty
            else []
        )
        rows.append(
            {
                "season": int(season),
                "celebrity_name": name,
                "nominee_rate_S3": nominee_rate,
                "final_alive_S1": final.get("S1"),
                "final_alive_S3": final.get("S3"),
            }
        )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def check_p084(v1_detail: pd.DataFrame) -> pd.DataFrame:
    """Paper line 942: pre-filter immunity + the gate does not correct controversy.

    Claim A: highly popular contestants rarely enter the S3 judge-gate risk set
    (``nominee_rate_S3`` small) — pre-filter immunity.
    Claim B: they still survive deep into the season under S3
    (``final_alive_S3`` high) — the mechanism does not correct the canonical
    controversy.  Thresholds: nominee rate <= 0.2 ("rarely"), final survival
    >= 0.25 ("deep").
    """
    nom = nominee_rates(v1_detail)
    if nom.empty:
        return pd.DataFrame(columns=CLAIM_COLUMNS)
    rows = []
    for _, r in nom.iterrows():
        nr = r["nominee_rate_S3"]
        fa = r["final_alive_S3"]
        rows.append(
            _row(
                "P-084a",
                "P",
                "Pre-filter immunity: popularity-driven cases rarely enter the S3 risk set",
                f"nominee_rate_S3 ({r['celebrity_name']})",
                round(float(nr), 4) if pd.notna(nr) else None,
                "<= 0.20 (rarely in risk set)",
                "pass" if (pd.notna(nr) and nr <= 0.20) else "fail",
            )
        )
        rows.append(
            _row(
                "P-084b",
                "P",
                "The S3 gate does not correct the canonical controversy: survival stays deep",
                f"final_alive_S3 ({r['celebrity_name']})",
                round(float(fa), 4) if pd.notna(fa) else None,
                ">= 0.25 (survives deep)",
                "pass" if (pd.notna(fa) and fa >= 0.25) else "fail",
            )
        )
    return pd.DataFrame(rows)


def check_p085(v2_detail: pd.DataFrame) -> pd.DataFrame:
    """Paper line 1050: V2 corrects popularity cases; protects a technical case.

    - Bobby Bones: baseline V4 mean rank ~ champion (<= 2), V5 ~ 6th (in
      [4, 8]) — "drops from champion to 6th".
    - Tinashe: baseline V4 exit ~ week 7 (<= 7), V5 survives to week 8+
      (>= 8) — "protected by the judges' save (Week 7 -> Week 8)".
    """
    placed = case_placement_table(v2_detail, detail_kind="V2")
    rows: list[dict] = []
    bb = placed[(placed["celebrity_name"] == "Bobby Bones") & (placed["season"] == 27)]
    if not bb.empty:
        v4 = bb.loc[bb["scheme"] == "V4", "mean_rank"]
        v5 = bb.loc[bb["scheme"] == "V5", "mean_rank"]
        v4v = float(v4.iloc[0]) if not v4.empty else None
        v5v = float(v5.iloc[0]) if not v5.empty else None
        rows.append(
            _row(
                "P-085a",
                "P/R",
                "Bobby Bones drops from champion to ~6th under V2",
                "mean_rank V4 (baseline)",
                round(v4v, 3) if v4v is not None else None,
                "<= 2 (champion)",
                "pass" if v4v is not None and v4v <= 2.0 else "fail",
            )
        )
        rows.append(
            _row(
                "P-085b",
                "P/R",
                "Bobby Bones drops from champion to ~6th under V2",
                "mean_rank V5 (V2)",
                round(v5v, 3) if v5v is not None else None,
                "in [4, 8] (~6th)",
                "pass" if v5v is not None and 4.0 <= v5v <= 8.0 else "fail",
            )
        )
    ti = placed[(placed["celebrity_name"] == "Tinashe") & (placed["season"] == 27)]
    if not ti.empty:
        exits = exit_week_by_scheme(
            v2_detail[(v2_detail["season"] == 27) & (v2_detail["celebrity_name"] == "Tinashe")]
        )
        v4x = exits.get("V4")
        v5x = exits.get("V5")
        rows.append(
            _row(
                "P-085c",
                "P/R",
                "Tinashe protected by the judges' save (Week 7 -> Week 8)",
                "exit week V4 (baseline)",
                v4x,
                "<= 7",
                "pass" if v4x is not None and v4x <= 7 else "fail",
            )
        )
        rows.append(
            _row(
                "P-085d",
                "P/R",
                "Tinashe protected by the judges' save (Week 7 -> Week 8)",
                "exit week V5 (V2)",
                v5x,
                ">= 8",
                "pass" if v5x is not None and v5x >= 8 else "fail",
            )
        )
    if not rows:
        return pd.DataFrame(columns=CLAIM_COLUMNS)
    return pd.DataFrame(rows)


def check_p086(v1_detail: pd.DataFrame, v2_detail: pd.DataFrame) -> pd.DataFrame:
    """Paper Shock_k definition (line 883) and V2 fairness claims (lines 1048, 1056).

    ``Shock_k`` = Pr(eliminated contestant is in Top-k by judges) — "should
    remain small" (line 885).  Paper line 1048(i) claims the judges' save
    "prevents severe technical injustice at exit", i.e. V2 has a *smaller* shock
    than its own baseline (V0 == V4).  We check that first, then the
    cross-mechanism diagnostic that V2 is not worse than the V1 full-season gate
    (supports the line 1058 "V2 for legitimacy" recommendation), and finally
    report all four schemes' shock rates as context.  The recommendation itself
    remains a design judgment; these rows only report whether the reproduced
    shock evidence is consistent with it, without overstating it.
    """
    v1_shock = shock_table(v1_detail, "V1")
    v2_shock = shock_table(v2_detail, "V2")
    rows: list[dict] = []

    def _lookup(tbl: pd.DataFrame, scheme: str) -> float | None:
        if tbl.empty:
            return None
        hit = tbl[tbl["scheme"] == scheme]
        return float(hit["shock_k3"].iloc[0]) if not hit.empty else None

    s3 = _lookup(v1_shock, "S3")
    s1 = _lookup(v1_shock, "S1")
    v5 = _lookup(v2_shock, "V5")
    v4 = _lookup(v2_shock, "V4")
    rows.append(
        _row(
            "P-086a",
            "P/R",
            "V2's judges' save keeps technical shock small vs the V2 baseline",
            "Shock_k3 V5 vs V4",
            (round(v5, 4) if v5 is not None else None, round(v4, 4) if v4 is not None else None),
            "Shock_k3(V5) <= Shock_k3(V4)",
            "pass" if (v5 is not None and v4 is not None and v5 <= v4) else "fail",
        )
    )
    rows.append(
        _row(
            "P-086b",
            "P/R",
            "Cross-mechanism: V2 shock is not higher than the V1 full-season gate",
            "Shock_k3 V5 vs S3",
            (round(v5, 4) if v5 is not None else None, round(s3, 4) if s3 is not None else None),
            "Shock_k3(V5) <= Shock_k3(S3)",
            "pass" if (v5 is not None and s3 is not None and v5 <= s3) else "fail",
        )
    )
    rows.append(
        _row(
            "P-086c",
            "P/R",
            "All four schemes' Shock_k3 reported (line 885 'should remain small')",
            "Shock_k3 {S1,S3,V4,V5}",
            {
                k: (round(v, 4) if v is not None else None)
                for k, v in {"S1": s1, "S3": s3, "V4": v4, "V5": v5}.items()
            },
            "reported",
            "reported",
        )
    )
    return pd.DataFrame(rows)


def shock_table(df: pd.DataFrame, detail_kind: str = "V1") -> pd.DataFrame:
    """Per-scheme Shock_k rates (k=1,2,3) over all elimination events."""
    from .metrics import shock_rates

    return shock_rates(df, ks=(1, 2, 3), detail_kind=detail_kind)


def check_all(v1_detail: pd.DataFrame, v2_detail: pd.DataFrame) -> pd.DataFrame:
    """Run all Problem-4 claim checks; return the joined check table."""
    frames = [check_p084(v1_detail), check_p085(v2_detail), check_p086(v1_detail, v2_detail)]
    frames = [f for f in frames if not f.empty]
    if not frames:
        return pd.DataFrame(columns=CLAIM_COLUMNS)
    return pd.concat(frames, ignore_index=True)
