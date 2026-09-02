# Phase 7 acceptance packet — Release reproduction

Phase 7 (release reproduction) is implemented and its gate is evidenced below. This packet is
evidence, not a narrative claim: the owner reviews it and the independent audit verifies it.
Track P and Track R are implemented and compared explicitly; no paper formula, sample definition,
or hyperparameter was altered, and every deviation from the paper, review, or legacy results is
registered in `docs/DECISIONS.md`.

## Scope completed

A single documented command reproduces the full release from the current checkout: every pipeline
stage (Problem 1 Track P/R, extras, Problem 2 Track P/R, Problem 3, Problem 4, sensitivity, and all
plot scripts), the regenerated baseline / traceability / conflict-matrix documents, and a comparison
of every registered baseline row (B-01..B-20) against the freshly produced artifacts. The comparison
and the run manifest are saved and their content hash is pinned.

## Gate result

The Phase 7 gate — *one documented command recreates the release; another runs the full test suite;
all deviations appear in `docs/DECISIONS.md`* — passes.

```
# full release (≈23 min, all 19 stages, exit 0) — at the committed HEAD (569994b)
.venv/bin/python scripts/run_release.py
  -> outputs/release_manifest.json   (19/19 stages exit 0; total 1375.8 s;
                                       git 569994b, python 3.13.3, platform darwin;
                                       comparison sha256 pinned and matching)
  -> outputs/release_comparison.json (checked=20 pass=20 fail=0 info=0 release_ok=True;
                                       B-08 enforced as a numeric |d|/flip contract)

# fast re-verification (comparison only, current module)
.venv/bin/python scripts/run_release.py --verify-only   -> 20/20 PASS, release_ok=True

# full test suite
.venv/bin/python -m pytest -q      -> 229 passed
.venv/bin/ruff format --check .    -> clean
.venv/bin/ruff check .             -> All checks passed!
.venv/bin/mypy src/dwts_reproduction -> Success (no issues)
```

### Baseline rows (20/20 PASS)

| Row | Item | Observed | Status |
|-----|------|----------|--------|
| B-01 | XGBoost top-1 | week 0.821101 / season 0.817496; paper target 0.806554 preserved | not-reproduced (D-20260901-11/C-07) |
| B-02 | torch in-season top-1 | 0.952092 (rel 3.7e-07) | reproduced |
| B-03 | Top-1 (review rebuild) | 0.949541 (rel 3.0e-07) | reproduced |
| B-04 | Cumulative consistency S-bar | 0.778531 (abs 1.5e-03) | reproduced |
| B-05 | Mean PCP (weighted) | 0.604317 (abs 1.7e-05) | reproduced |
| B-06 | Mean ESS ratio | 0.962517 (abs 1.7e-05) | reproduced |
| B-07 | Mean CI relative width | 3.11714 (abs 1.4e-04) | reproduced |
| B-08 | Case-study (\|d\|, Flip) | 6 rows; every registered \|d\|/flip within abs 1e-2 | reproduced |
| B-09 | V1 parameters | K=3, m_early=8 | reproduced |
| B-10 | V2 parameters | wJ=0.80 wF=0.20 gamma=0.45 delta=1.35 mu=0.01 L=2 | reproduced |
| B-11 | Problem 3 age coefficient | -0.0301 / -0.0329 / -0.0359 | reproduced |
| B-12 | Problem 3 actor coefficient | 0.254 / -1.0221 vs paper 0.16/-0.87 | direction-confirmed only (D-20260901-17) |
| B-13 | Problem 3 partner r | 0.134 vs paper 0.23 | direction-confirmed only (D-20260901-17) |
| B-14 | Problem 3 surprise beta1 | 0.3419 (p<0.001) | reproduced |
| B-15 | Problem 1 hyperparameters | tau 0.05/0.15, l2 0.05, kappa 10, lr 0.02, steps 600, B 1200 | reproduced |
| B-16 | Preprocessing validation targets | R-001..R-019: 19 targets, 0 not implemented | reproduced |
| B-17 | Paper figures | 6 manifests, 79 PNGs (P1E 6, P2 P/R 25+25, P3 5, P4 15, SA 3) | reproduced |
| B-18 | Phase tables | 136 phase rows, 12 b2 rows | reproduced |
| B-19 | V1 legacy parity | max abs diff 0.0026 over 99 cells (tol 5e-3) | reproduced |
| B-20 | V2 claims | 66 rows, claims pass 9 / fail 5 / reported 1 (P-086a/b pass) | reproduced |

## Files created / modified (Phase 7 work, all under `repo/`)

- Release framework: `scripts/run_release.py`, `src/dwts_reproduction/release/compare.py`,
  `tests/test_release_compare.py` (22 hermetic tests), `scripts/build_baseline.py`,
  `scripts/build_traceability.py`, `scripts/build_conflict_matrix.py`.
- Documents: `manifests/baseline.csv` (20 rows), `docs/BASELINE_PAPER_OUTPUTS.md`,
  `manifests/traceability_{paper,review}.csv` (96 / 40 rows, all `implemented`),
  `manifests/conflict_matrix.csv`, `docs/PHASE7_ACCEPTANCE.md` (this packet).
- Reproduced artifacts (gitignored, regenerable): `outputs/*` saved tables, run manifests, figure
  manifests, and `outputs/figures{,_{P,R}}/*.png` (79 PNGs), each recorded in a figure manifest with
  its input-table hash.

## Track separation

- **Track P** reproduces the paper's two-stage procedure (fit `q` from elimination outcomes, then
  condition weekly `p` on the observed elimination). Its reconstruction metrics are labeled
  **internal/explanatory** (double-use limitation documented in D-20260901-02): top1 0.9495, PCP
  0.6043, CI width 3.117, S_bar 0.7785.
- **Track R** implements the review's integrated marginal-likelihood formulation (single use of each
  outcome): top1 0.8349, PCP 0.5342, CI width 3.378, S_bar 0.6331 (labeled separately; sensitivity
  spans top-1 0.803–0.844). The lower performance and the provisional negative `β_j` at the
  marginal-likelihood optimum are **provisional until the final independent audit**; evidence is
  preserved and the limitation is not hidden (D-20260901-02, D-20260901-24).
- Never reported without a track label.

## Honest limitations (not hidden, all decisioned)

1. **B-01 XGBoost paper target 0.806554 is not reproducible** from the current legacy code/data; the
   repo port is bit-for-bit identical to a live legacy run (C-07). Reported with the honest legacy
   line 0.821101 / 0.817496 (D-20260901-11).
2. **Paper ranking-gap `R² > 0.6` claim not reproducible** — exact cell-56 port gives R² = 0.2704,
   n=421 (D-20260901-12; jitter is plot-only, D-20260901-15).
3. **Review claim R-040** (Perc+Bottom2-highest) is **not supported** on the reproduced data; the
   top mechanism by all reported quantities is `rank_bottom2` on both tracks (D-20260901-10).
4. **P-057** (paper's `x >= 0.3` fan-influence subset) is **not testable** — the subset is empty on
   the reproduced data (D-20260901-10).
5. **B-12 actor** and **B-13 partner r** are **direction-confirmed only**; paper targets are outside
   tolerance (D-20260901-17). Problem 3 `beta3` (S×H_exp) is directional only (0.0104, p=0.51).
6. **Latent fan votes are never ground truth** — reported as posterior estimates constrained by
   observed outcomes (D-20260901-06).
7. Phase 0 (baseline/provenance) technically passes but still awaits owner acceptance and the
   independent audit (`PLAN.md` Phase 0 note). This is a process gate, not a technical failure; the
   release reproduction does not depend on it.

## Assumptions / decisions (summary of `docs/DECISIONS.md`)

D-20260901-01 era mapping · -02 Track P double-use limitation + Track R provisional framing ·
-04 numpy-vs-torch gap · -06 fan votes not ground truth · -07 posterior reweighting mode · -08 MC
estimator/optimizer choice · -09 tie/b2-save semantics · -10 phase-diagram axis definition and
claim checks · -11 XGBoost not reproducible · -12 ranking-gap not reproducible · -13 torch
aggregation is mean-of-season-means · -14 PCP formula variants · -15 plot-only jitter · -16
S8/S21 heatmap adaption · -17 Problem 3 honest claim status · -18 V2 defaults · -19 V1 parity
tolerance · -20 sensitivity definitions · -21/22 Problem 2 figure rendering · -23 B-16/B-17
comparison scoping · -24 B-08 numeric contract + provisional Track R framing.

## Rerun commands

```bash
.venv/bin/python scripts/run_release.py            # full release, 1375.8 s (≈ 23 min)
.venv/bin/python scripts/run_release.py --verify-only   # fast re-check
.venv/bin/python -m pytest -q                      # full test suite (229)
.venv/bin/ruff format --check . && .venv/bin/ruff check . && .venv/bin/mypy src/dwts_reproduction
```

Runtime cost of the full release: 1375.8 s (≈ 23 min) wall-clock on this machine (problem4 sims
≈ 15 min, problem2 Track P/R ≈ 6.4 min); "≈ 24 min" is the coarse planning estimate. No API/token
cost beyond local compute.
