"""Release comparison: check the produced artifacts against the registered baseline.

Each row of ``manifests/baseline.csv`` records an expected value (paper target, legacy
value, or honest observed value), a proposed tolerance, and a status
(``registered`` / ``reproduced`` / ``not-reproduced`` / ``direction-confirmed only``).
``compare`` reads the currently produced outputs and reports, per row, whether the
artifact still matches what was registered. A row is ``PASS`` when the produced number
agrees with the baseline's stated value *as registered* (including rows whose paper
target is honestly not reproduced — the embedded observed value is what is checked);
``FAIL`` when the produced artifact has drifted from what the baseline registered.

This module is pure: it only reads files at the paths it is given (outputs directory,
legacy data directory, baseline CSV). It never writes.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

VERDICT_PASS = "PASS"
VERDICT_FAIL = "FAIL"
VERDICT_INFO = "INFO"


@dataclass(frozen=True)
class CheckResult:
    """One baseline row checked against the produced artifacts."""

    id: str
    item: str
    tolerance: str
    observed: str
    verdict: str
    detail: str


@dataclass(frozen=True)
class CheckContext:
    """Everything a per-row check may read."""

    outputs: Path
    legacy: Path
    repo_root: Path


def parse_tolerance(tol: str) -> tuple[str, float | None]:
    """Return (kind, value) for a tolerance string.

    Examples: ``"rel 1e-3 (proposed)"`` -> ``("rel", 0.001)``,
    ``"exact"`` -> ``("exact", None)``, ``"abs 0.02 (proposed)"`` -> ``("abs", 0.02)``.
    """
    m = re.match(r"^(rel|abs)\s+([0-9][0-9.eE+-]*)", str(tol).strip())
    if m:
        return m.group(1).lower(), float(m.group(2))
    kind = re.match(r"^([A-Za-z-]+)", str(tol).strip())
    return (kind.group(1).lower() if kind else "unknown"), None


def _json(path: Path) -> dict[str, object]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    assert isinstance(data, dict), f"{path} is not a JSON object"
    return data


def _csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def _fmt(x: float) -> str:
    return f"{x:.6g}"


def _num(payload: dict[str, object], key: str) -> float:
    """Extract a numeric value from a JSON payload with a runtime guard."""
    value = payload[key]
    assert isinstance(value, (int, float)), f"{key} is not numeric"
    return float(value)


def _abs_err(obs: float, exp: float) -> float:
    return abs(obs - exp)


def _rel_err(obs: float, exp: float) -> float:
    return abs(obs - exp) / abs(exp) if exp else float("inf")


# ---------------------------------------------------------------------------
# per-row checks
# ---------------------------------------------------------------------------


def _check_b01(ctx: CheckContext, row: pd.Series) -> CheckResult:
    """XGBoost baseline: paper target preserved; honest legacy line checked."""
    p1e = _json(ctx.outputs / "problem1_extras_summary_P1E.json")
    xgb_week = _num(p1e, "inseason_xgb_overall_accuracy")
    paper_target = _num(p1e, "paper_b01_xgb_target")
    inseason = _csv(ctx.outputs / "problem1_extras_inseason_by_season_P1E.csv")
    xgb_season = float(
        inseason.loc[
            inseason["model"].astype(str).str.contains("xgboost", case=False),
            "accuracy",
        ].mean()
    )
    ok = _rel_err(xgb_week, 0.821101) < 1e-3 and abs(paper_target - 0.806554) < 1e-6
    detail = (
        f"paper target {_fmt(paper_target)} preserved; honest legacy line "
        f"week-mean {_fmt(xgb_week)} / season-mean {_fmt(xgb_season)} "
        f"(not-reproduced, D-20260901-11 / C-07)"
    )
    return CheckResult(
        row["id"],
        str(row["item"]),
        str(row["proposed_tolerance"]),
        f"week-mean {_fmt(xgb_week)}, season-mean {_fmt(xgb_season)}",
        VERDICT_PASS if ok else VERDICT_FAIL,
        detail,
    )


def _torch_season_mean(ctx: CheckContext) -> float:
    inseason = _csv(ctx.outputs / "problem1_extras_inseason_by_season_P1E.csv")
    return float(
        inseason.loc[
            inseason["model"].astype(str).str.contains("torch", case=False),
            "accuracy",
        ].mean()
    )


def _check_b02(ctx: CheckContext, row: pd.Series) -> CheckResult:
    obs = _torch_season_mean(ctx)
    err = _rel_err(obs, 0.952092)
    kind, tol = parse_tolerance(row["proposed_tolerance"])
    ok = err < (tol if tol is not None else 1e-3)
    return CheckResult(
        row["id"],
        str(row["item"]),
        str(row["proposed_tolerance"]),
        f"torch season-mean {_fmt(obs)}",
        VERDICT_PASS if ok else VERDICT_FAIL,
        f"rel err {err:.2e} vs target 0.952092",
    )


def _p1_summary(ctx: CheckContext) -> dict:
    return _json(ctx.outputs / "problem1_summary_P.json")


def _numeric_check(
    ctx: CheckContext,
    row: pd.Series,
    observed: float,
    expected: float,
) -> CheckResult:
    kind, tol = parse_tolerance(row["proposed_tolerance"])
    if kind == "rel":
        ok = _rel_err(observed, expected) < (tol if tol is not None else 1e-3)
        err_str = f"rel err {_rel_err(observed, expected):.2e}"
    else:  # abs
        ok = _abs_err(observed, expected) < (tol if tol is not None else 1e-3)
        err_str = f"abs err {_abs_err(observed, expected):.2e}"
    return CheckResult(
        row["id"],
        str(row["item"]),
        str(row["proposed_tolerance"]),
        _fmt(observed),
        VERDICT_PASS if ok else VERDICT_FAIL,
        f"{err_str} vs target {_fmt(expected)}",
    )


def _check_b03(ctx: CheckContext, row: pd.Series) -> CheckResult:
    return _numeric_check(ctx, row, _num(_p1_summary(ctx), "overall_top1_accuracy"), 0.949541)


def _check_b04(ctx: CheckContext, row: pd.Series) -> CheckResult:
    return _numeric_check(ctx, row, _num(_p1_summary(ctx), "s_bar"), 0.78)


def _check_b05(ctx: CheckContext, row: pd.Series) -> CheckResult:
    return _numeric_check(ctx, row, _num(_p1_summary(ctx), "mean_pcp_weighted"), 0.6043)


def _check_b06(ctx: CheckContext, row: pd.Series) -> CheckResult:
    return _numeric_check(ctx, row, _num(_p1_summary(ctx), "mean_ess_ratio"), 0.9625)


def _check_b07(ctx: CheckContext, row: pd.Series) -> CheckResult:
    return _numeric_check(ctx, row, _num(_p1_summary(ctx), "mean_ci_rel_width"), 3.117)


def _check_b08(ctx: CheckContext, row: pd.Series) -> CheckResult:
    """Named controversy cases: |d| / Flip numeric contract against the registered table.

    The baseline registers 2-decimal |d|/Flip values ("Jerry Rice 3.69/0.87; ...",
    abbreviated names). Produced values are the unrounded posterior quantities; the
    check asserts each within abs 1e-2 (the registration's rounding resolution).
    """
    # Registered |d|/Flip keyed by the produced full celebrity name; the baseline's
    # abbreviations are Jerry Rice / B.R.Cyrus / B.Palin / Bobby Bones / Tinashe / Vinny G.
    registered = {
        "Jerry Rice": (3.69, 0.87),
        "Billy Ray Cyrus": (3.25, 0.75),
        "Bristol Palin": (4.30, 0.97),
        "Bobby Bones": (4.00, 0.57),
        "Tinashe": (8.50, 0.57),
        "Vinny Guadagnino": (9.88, 0.33),
    }
    div = _csv(ctx.outputs / "problem2_case_divergence_P.csv")
    names = sorted(str(x) for x in div["celebrity_name"])
    ok = len(div) == 6 and names == sorted(registered)
    deltas: list[str] = []
    tol = 1e-2
    if ok:
        for name, (exp_d, exp_flip) in registered.items():
            found = div[div["celebrity_name"].astype(str) == name]
            if found.empty:
                ok = False
                deltas.append(f"{name}: MISSING")
                continue
            d = float(found.iloc[0]["abs_d"])
            flip = float(found.iloc[0]["flip"])
            d_ok = abs(d - exp_d) <= tol
            flip_ok = abs(flip - exp_flip) <= tol
            if not (d_ok and flip_ok):
                ok = False
                deltas.append(
                    f"{name}:|d|={d}(Δ{abs(d - exp_d):.4f})/flip={flip}(Δ{abs(flip - exp_flip):.4f})"
                )
            else:
                deltas.append(f"{name}:|d|={d}/flip={flip}")
    vals = "; ".join(deltas) if deltas else "6 rows"
    return CheckResult(
        row["id"],
        str(row["item"]),
        str(row["proposed_tolerance"]),
        f"{len(div)} rows",
        VERDICT_PASS if ok else VERDICT_FAIL,
        vals,
    )


def _config_values(ctx: CheckContext) -> tuple[dict, dict]:
    from dwts_reproduction.problem4.features import V1_DEFAULTS, V2_DEFAULTS  # noqa: PLC0415

    return dict(V1_DEFAULTS), dict(V2_DEFAULTS)


def _check_b09(ctx: CheckContext, row: pd.Series) -> CheckResult:
    v1, _ = _config_values(ctx)
    ok = v1["K"] == 3 and v1["m_early_elims"] == 8
    return CheckResult(
        row["id"],
        str(row["item"]),
        str(row["proposed_tolerance"]),
        f"K={v1['K']}, m_early={v1['m_early_elims']}",
        VERDICT_PASS if ok else VERDICT_FAIL,
        "V1 defaults match legacy (K=3, early/late split m=8)",
    )


def _check_b10(ctx: CheckContext, row: pd.Series) -> CheckResult:
    _, v2 = _config_values(ctx)
    ok = (
        abs(v2["wJ"] - 0.80) < 1e-9
        and abs(v2["wF"] - 0.20) < 1e-9
        and abs(v2["gamma"] - 0.45) < 1e-9
        and abs(v2["delta"] - 1.35) < 1e-9
        and abs(v2["mu"] - 0.01) < 1e-9
        and v2["L"] in (2, 3)
    )
    return CheckResult(
        row["id"],
        str(row["item"]),
        str(row["proposed_tolerance"]),
        f"wJ={v2['wJ']} wF={v2['wF']} gamma={v2['gamma']} delta={v2['delta']} "
        f"mu={v2['mu']} L={v2['L']}",
        VERDICT_PASS if ok else VERDICT_FAIL,
        "V2 defaults match the paper/legacy (D-20260901-18)",
    )


def _p3_summary(ctx: CheckContext) -> dict:
    return _json(ctx.outputs / "problem3_summary_P3.json")


def _check_b11(ctx: CheckContext, row: pd.Series) -> CheckResult:
    s = _p3_summary(ctx)
    age = s["paper_P059_age_judge"]
    ok = bool(s["paper_P059_age_within_abs_0_02"])
    return CheckResult(
        row["id"],
        str(row["item"]),
        str(row["proposed_tolerance"]),
        " / ".join(f"{k}={_fmt(v)}" for k, v in age.items()),
        VERDICT_PASS if ok else VERDICT_FAIL,
        "age coefficients within abs 0.02 of paper target (reproduced)",
    )


def _check_b12(ctx: CheckContext, row: pd.Series) -> CheckResult:
    s = _p3_summary(ctx)
    w1, w6 = float(s["paper_P060_actor_judge_w1"]), float(s["paper_P060_actor_fan_w6"])
    ok = abs(w1 - 0.254) < 1e-6 and abs(w6 - (-1.0221)) < 1e-6
    return CheckResult(
        row["id"],
        str(row["item"]),
        str(row["proposed_tolerance"]),
        f"judge W1 {_fmt(w1)}, fan W6 {_fmt(w6)}",
        VERDICT_PASS if ok else VERDICT_FAIL,
        "matches registered values; paper target 0.16/-0.87 not within abs 0.1 "
        "(direction-confirmed only, D-20260901-17)",
    )


def _check_b13(ctx: CheckContext, row: pd.Series) -> CheckResult:
    s = _p3_summary(ctx)
    r = float(s["paper_P064_r_Hexp_judge_w1"])
    ok = abs(r - 0.134) < 1e-6
    return CheckResult(
        row["id"],
        str(row["item"]),
        str(row["proposed_tolerance"]),
        f"r={_fmt(r)}",
        VERDICT_PASS if ok else VERDICT_FAIL,
        "matches registered value; paper target 0.23 not within abs 0.05 "
        "(direction-confirmed only, D-20260901-17)",
    )


def _check_b14(ctx: CheckContext, row: pd.Series) -> CheckResult:
    s = _p3_summary(ctx)
    b1 = float(s["paper_P069_beta1_tw6"])
    ok = bool(s["paper_P069_within_abs_0_05"])
    return CheckResult(
        row["id"],
        str(row["item"]),
        str(row["proposed_tolerance"]),
        f"beta1={_fmt(b1)}",
        VERDICT_PASS if ok else VERDICT_FAIL,
        "surprise beta1 within abs 0.05 (reproduced)",
    )


def _check_b15(ctx: CheckContext, row: pd.Series) -> CheckResult:
    import yaml  # noqa: PLC0415

    cfg = yaml.safe_load((ctx.repo_root / "configs" / "problem1.yaml").read_text())
    expect = {
        "tau_train": 0.05,
        "l2_beta": 0.05,
        "l2_u": 0.05,
        "kappa": 10.0,
        "lr": 0.020,
        "n_steps": 600,
        "batch_size": 32,
        "B": 1200,
    }
    got = {k: cfg.get(k) for k in expect}
    ok = all((got[k] is not None and abs(float(got[k]) - v) < 1e-9) for k, v in expect.items())
    return CheckResult(
        row["id"],
        str(row["item"]),
        str(row["proposed_tolerance"]),
        "; ".join(f"{k}={got[k]}" for k in expect),
        VERDICT_PASS if ok else VERDICT_FAIL,
        "hyperparameters match the paper (B-15 registered)",
    )


def _check_b16(ctx: CheckContext, row: pd.Series) -> CheckResult:
    rev = _csv(ctx.repo_root / "manifests" / "traceability_review.csv")
    ids = rev["id"].astype(str)
    # Registered scope (baseline expected_value): preprocessing validation targets R-001..R-019.
    subset = rev[ids.str.fullmatch(r"R-0(0[1-9]|1[0-9])")].copy()
    subset.loc[:, "status"] = subset["status"].fillna("").astype(str).str.lower()
    not_impl = subset[subset["status"] != "implemented"]
    ok = len(subset) == 19 and len(not_impl) == 0
    return CheckResult(
        row["id"],
        str(row["item"]),
        str(row["proposed_tolerance"]),
        f"{len(subset)} preprocessing targets, {len(not_impl)} not implemented",
        VERDICT_PASS if ok else VERDICT_FAIL,
        "preprocessing validation targets R-001..R-019 implemented",
    )


def _fig_count(d: dict[str, object]) -> int:
    """Count figure entries across the manifest schema variants used in this repo.

    Problem 2 / sensitivity manifests store ``{"figures": {name: meta}}``;
    Problem 1 P1E / Problem 3 store ``{"outputs": {name: sha}}``; Problem 4 stores
    ``{"files": {name: sha}}``.
    """
    for key in ("figures", "outputs", "files"):
        value = d.get(key)
        if isinstance(value, dict):
            return len(value)
    return 0


def _check_b17(ctx: CheckContext, row: pd.Series) -> CheckResult:
    manifests = sorted(ctx.outputs.glob("*_fig_manifest_*.json"))
    total_pngs = 0
    details: list[str] = []
    for m in manifests:
        n = _fig_count(_json(m))
        total_pngs += n
        details.append(f"{m.name}: {n} PNGs")
    # 6 manifests must exist and every schema variant must actually be counted
    # (the known reproduction renders 79 PNGs; floor guards against under-counting).
    ok = len(manifests) >= 6 and total_pngs >= 60
    return CheckResult(
        row["id"],
        str(row["item"]),
        str(row["proposed_tolerance"]),
        f"{len(manifests)} manifests, {total_pngs} PNGs",
        VERDICT_PASS if ok else VERDICT_FAIL,
        "; ".join(details),
    )


def _check_b18(ctx: CheckContext, row: pd.Series) -> CheckResult:
    phase = _csv(ctx.outputs / "problem2_phase_metrics_P.csv")
    b2 = _csv(ctx.outputs / "problem2_b2_metrics_P.csv")
    ok = len(phase) == 136 and len(b2) == 12
    return CheckResult(
        row["id"],
        str(row["item"]),
        str(row["proposed_tolerance"]),
        f"{len(phase)} phase rows, {len(b2)} b2 rows",
        VERDICT_PASS if ok else VERDICT_FAIL,
        "Phase 2 tables have the registered structural shape (34 seasons x 4 mechanisms)",
    )


def _check_b19(ctx: CheckContext, row: pd.Series) -> CheckResult:
    """V1 simulator parity against the legacy sim_summary.csv (99 cells)."""
    v1 = _csv(ctx.outputs / "problem4_sim_summary_V1.csv")
    legacy = _csv(ctx.legacy / "sim_summary.csv")
    cols = ["scheme", "week", "archetype"]
    v1 = v1.sort_values(cols).reset_index(drop=True)
    legacy = legacy.sort_values(cols).reset_index(drop=True)
    if len(v1) != 99 or len(legacy) != 99:
        return CheckResult(
            row["id"],
            str(row["item"]),
            str(row["proposed_tolerance"]),
            f"repo {len(v1)} rows / legacy {len(legacy)} rows",
            VERDICT_FAIL,
            "row count mismatch",
        )
    worst = 0.0
    worst_col = ""
    for c in ["avg_rank", "alive_rate", "elim_rate"]:
        if c not in v1.columns or c not in legacy.columns:
            continue
        diff = float((v1[c] - legacy[c]).abs().max())
        if diff > worst:
            worst, worst_col = diff, c
    ok = worst < 5e-3
    return CheckResult(
        row["id"],
        str(row["item"]),
        str(row["proposed_tolerance"]),
        f"max abs diff {worst:.4f} ({worst_col}) over 99 cells",
        VERDICT_PASS if ok else VERDICT_FAIL,
        "V1 avg_rank/alive_rate/elim_rate within 5e-3 of legacy sim_summary.csv "
        "(recalibrated, D-20260901-19)",
    )


def _check_b20(ctx: CheckContext, row: pd.Series) -> CheckResult:
    """V2 summaries + Shock_k claim checks (P-086a/b pass on the seeded run)."""
    v2 = _csv(ctx.outputs / "problem4_sim_summary_V2.csv")
    claims = _csv(ctx.outputs / "problem4_claims_P4.csv")
    schemes = set(v2["scheme"]) if "scheme" in v2.columns else set()
    has_v4v5 = {"V4", "V5"} <= schemes
    claim_ok = True
    for cid in ("P-086a", "P-086b"):
        sub = claims[claims["claim_id"] == cid]
        if sub.empty or (sub["status"].astype(str) != "pass").any():
            claim_ok = False
    ok = has_v4v5 and claim_ok and len(v2) == 66
    status = claims["status"].value_counts().to_dict()
    return CheckResult(
        row["id"],
        str(row["item"]),
        str(row["proposed_tolerance"]),
        f"V2 {len(v2)} rows, schemes={sorted(schemes)}, claims {status}",
        VERDICT_PASS if ok else VERDICT_FAIL,
        "V4/V5 summaries present; Shock_k claims P-086a/b pass on the seeded run",
    )


_Check = Callable[[CheckContext, pd.Series], CheckResult]

_CHECKS: dict[str, _Check] = {
    "B-01": _check_b01,
    "B-02": _check_b02,
    "B-03": _check_b03,
    "B-04": _check_b04,
    "B-05": _check_b05,
    "B-06": _check_b06,
    "B-07": _check_b07,
    "B-08": _check_b08,
    "B-09": _check_b09,
    "B-10": _check_b10,
    "B-11": _check_b11,
    "B-12": _check_b12,
    "B-13": _check_b13,
    "B-14": _check_b14,
    "B-15": _check_b15,
    "B-16": _check_b16,
    "B-17": _check_b17,
    "B-18": _check_b18,
    "B-19": _check_b19,
    "B-20": _check_b20,
}


def compare(
    baseline_csv: str | Path,
    outputs_dir: str | Path,
    legacy_dir: str | Path,
    *,
    repo_root: str | Path | None = None,
) -> list[CheckResult]:
    """Check every baseline row against the produced artifacts.

    ``baseline_csv`` is the registered baseline table; ``outputs_dir`` holds the
    produced artifacts; ``legacy_dir`` holds read-only legacy outputs (used by the
    B-19 parity check). Returns one :class:`CheckResult` per baseline row, in row
    order.
    """
    if repo_root is None:
        repo_root = Path(__file__).resolve().parents[3]
    baseline = pd.read_csv(baseline_csv)
    ctx = CheckContext(
        outputs=Path(outputs_dir), legacy=Path(legacy_dir), repo_root=Path(repo_root)
    )
    results: list[CheckResult] = []
    for _, row in baseline.iterrows():
        check = _CHECKS.get(str(row["id"]))
        if check is None:
            results.append(
                CheckResult(
                    str(row["id"]),
                    str(row["item"]),
                    str(row["proposed_tolerance"]),
                    "",
                    VERDICT_INFO,
                    "no automated check registered",
                )
            )
            continue
        results.append(check(ctx, row))
    return results


def summarize(results: list[CheckResult]) -> dict:
    """Aggregate verdict counts and the overall release-health flag."""
    counts: dict[str, int] = {VERDICT_PASS: 0, VERDICT_FAIL: 0, VERDICT_INFO: 0}
    for r in results:
        counts[r.verdict] += 1
    return {
        "checked": len(results),
        "pass": counts[VERDICT_PASS],
        "fail": counts[VERDICT_FAIL],
        "info": counts[VERDICT_INFO],
        "release_ok": counts[VERDICT_FAIL] == 0,
    }
