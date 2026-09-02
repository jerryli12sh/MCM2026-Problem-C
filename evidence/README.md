# Evidence figures — representative snapshot

This directory is a **frozen, representative snapshot** of figures from the recorded release, so a
reader can see headline results without running the ~23-minute pipeline. It is a *deliberate copy*:
the scripts that regenerate the full figure set write to gitignored `outputs/`, never here, so these
bytes only change when the owner deliberately refreshes the snapshot after accepting a new recorded
release.

## Snapshot origin

| Item | Value |
|---|---|
| Release run | git `569994b330177e77fea731936ed6b1bdfeb5c5d5` (see `outputs/release_manifest.json`) |
| Python / platform | 3.13.3 · macOS darwin/arm64 |
| Rendered from | saved source tables (figures are pure functions of tables; never live model state) |
| Full set | 79 figures across 6 manifests, regenerable under `outputs/` via `scripts/run_release.py` |
| Cross-platform note | matplotlib PNG bytes are **not** guaranteed bit-identical across platforms/versions; the recorded *numbers* reproduce within registered tolerances (see `docs/ENVIRONMENT.md`). This snapshot records what the recorded release produced on its own platform. |

## Figures and provenance

Each committed PNG below is byte-for-byte the file the recorded release wrote to `outputs/`. The
SHA-256 is computed on the committed bytes and matches the recorded figure-manifest hash for the
eight figures that carry one; the two phase diagrams carry a sidecar (`problem2_phase_diagram_{P,R}.json`)
that pins their source-table hash, git commit, and run-manifest match.

| Committed figure | Shows (traceability) | SHA-256 (first 16) |
|---|---|---|
| `problem1_fig_accuracy_line_P1E.png` | In-season top-1 accuracy line across the season — torch & XGBoost baselines, week-by-week (P-029) | `cac471da3665b77a` |
| `problem1_fig_ranking_gap_P1E.png` | Ranking-gap quadratic fit — **honest** R²=0.2704, n=421, *not* the paper's >0.6 (P-035; D-20260901-12) | `d4a80f6fd4405b69` |
| `problem1_fig_crowded_field_weighted_P1E.png` | Weighted PCP by crowd size — crowded-field analysis (P-033) | `35f729a32b8d92b0` |
| `problem2_phase_diagram_P.png` | Mechanism phase diagram, **Track P** — rank/percentage × direct/bottom-2+save (P-056/P-057) | `0717af647ae10915` |
| `problem2_phase_diagram_R.png` | Mechanism phase diagram, **Track R** — same axes, review-integrated posterior | `12e49c6d3d0fe330` |
| `problem3_fig_success_factors_P3.png` | Problem-3 demographic divergence — paper demo-model coefficients (age/industry) with CIs (P-061) | `f1ae53fce5b26a13` |
| `problem4_V1_plot1.png` | V1 new-rule simulation — week-by-archetype Δ(−rank̄) heatmap (S3 vs S1) | `153783d36420e7b7` |
| `problem4_sim2_trend_season2_Jerry_Rice.png` | V2 named-case trend — Season 2 Jerry Rice controversy under simulated rules | `288d79b2f35b3191` |
| `sensitivity_tornado_pcp_mean.png` | Sensitivity tornado — relative range of PCP by parameter family (P-092) | `57388b3ed1ca428d` |
| `sensitivity_A1_line_pcp_mean_by_kappa.png` | Sensitivity line — PCP vs τ by κ (P-093) | `67559c48e7152984` |

## Policy

- **`outputs/` is gitignored and regenerable** — the authoritative evidence for a given commit is the
  *recipe + manifests*, not committed bytes: rerun `scripts/run_release.py` (or `--verify-only`) and
  compare against the 20 registered baseline rows and the figure manifests.
- **This snapshot answers "what does the result look like?"** at a glance. It is not the comparison
  evidence; `outputs/release_comparison.json` and `docs/PHASE7_ACCEPTANCE.md` are.
- Refresh procedure (owner only): run the full release, review the git diff of the 20 registered
  baselines, then re-copy the desired figures from `outputs/` into `evidence/figures/` and update the
  SHA-256 column above. Do not refresh silently.
