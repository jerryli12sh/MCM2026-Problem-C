# Traceability coverage checklist

Automated parsing (see `tests/test_inventory_completeness.py`) only proves that every
`\includegraphics` in the paper appears in the paper inventory. It cannot prove that every
**formula, numeric claim, assumption, and conclusion** was captured. This manual checklist
fills that gap. It is a section-by-section sweep of the paper and the review notes; the
independent audit (`../prompts/PHASE0_AUDIT.md`) must verify each row against the source
documents and report any section with an uncaptured formula/claim/assumption/conclusion.

`captured` means: the item has a row in `manifests/traceability_paper.csv` (Track P) or
`manifests/traceability_review.csv` (Track R).

## Paper (Track P) — `../paper_Latex/2107542.tex`

| Section | Material content to verify | captured |
|---|---|---|
| Abstract | four task summaries; primary metric (disagreement/override); Bottom-2 recommendation; V1/V2 hybrid | ✅ |
| Introduction · Background | rank vs percentage vs Bottom-2 aggregation history | ✅ |
| Introduction · Restatement | four tasks (infer, compare, explain, design) | ✅ |
| Introduction · Our work | per-problem method one-liners | ✅ |
| Preparation · Assumptions | A1–A9 (nine assumptions) | ✅ |
| Preparation · Notations | notations table symbols | ✅ |
| Preparation · Preprocessing | 5 steps incl. `Z^J` rank normalization | ✅ |
| Problem 1 · Standardization | `J` percentage + rank-softmax | ✅ |
| Problem 1 · Latent | `η = xᵀβ + u`; `q = softmax`; feature choice (J, age) | ✅ |
| Problem 1 · Likelihood | `C = J + q`; softmin; penalized NLL | ✅ |
| Problem 1 · Posterior | Dirichlet prior `κq`; Bayes update; importance sampling | ✅ |
| Problem 1 · Internal eval | top-1 (`0.806554` / `0.952092`); cumulative consistency `0.78` | ✅ |
| Problem 1 · External eval | PCP; crowded-field; popularity premium `R²>0.6` | ✅ |
| Problem 1 · Uncertainty | `CI⁹⁰`; `RW`; heterogeneous uncertainty; ESS | ✅ |
| Problem 2 · Rules | rank rule; percentage rule | ✅ |
| Problem 2 · Divergence | `DR_s`; override; fan-worst; `Δ_s`; threshold effect | ✅ |
| Problem 2 · Week-level | Bottom-2; `rev_rate`; heatmaps | ✅ |
| Problem 2 · Cases | 6-case table (`\|d\|`/Flip); trajectory figures; asymmetric impact | ✅ |
| Problem 2 · Systemic | phase diagram; recommendation | ✅ |
| Problem 3 · Demographics | OLS; age/industry coefficients (`−0.04`, `−0.87`) | ✅ |
| Problem 3 · Partners | `H_abil`/`H_exp`; FE model; `r=0.23` | ✅ |
| Problem 3 · Dynamics | surprise/vote-growth; quadratic interaction (`0.34`) | ✅ |
| Problem 4 · Objectives | fairness `Shock_k`; excitement | ✅ |
| Problem 4 · V1 | judge gate `K=3`; `m=8` split; archetypes | ✅ |
| Problem 4 · V2 | tempering/sharpening/momentum; defaults; Bones→6th; Tinashe | ✅ |
| Problem 4 · Policy | V2/V1 recommendation; hybrid | ✅ |
| Sensitivity | A1–A4; stability; tornado; monotonic PCP | ✅ |
| Model Analysis | strengths; weaknesses (identifiability, V2 ablation) | ✅ |
| Memorandum | Actor's curse; casting; V2 recommendation | ✅ |
| References | bibliography entries | ✅ (not tracked as numeric) |
| AI-tools report | ChatGPT/Codex disclosures | ✅ (provenance only) |

## Review (Track R)

| Source | Material content to verify | captured |
|---|---|---|
| `review_all.md` · data | alive set, season length, horizon, eligibility, elim events | ✅ |
| `review_all.md` · model | judge signal; `q`; `C`; softmin; likelihood; regularization | ✅ |
| `review_all.md` · posterior | Dirichlet; importance sampling; **integrated marginal (Track R)** | ✅ |
| `review_all.md` · evaluation | top-1; cumulative consistency; PCP; CI/RW; override; reversal | ✅ |
| `review_all.md` · mechanism | tempering/sharpening/momentum; explanation (Problem 3) | ✅ |
| `00_problem_understanding_prep.md` | problem restatement; data semantics | ✅ |
| `00_codex_preprocessing_refactor_prompt.md` | all validation targets (shapes/counts/m_elim); era audit | ✅ |
| `01_codex_problem1_refactor_prompt.md` | panel/train-weeks/features/hyperparams/posterior/eval; overclaim guard | ✅ |
| `plan.md` | project layout (notes/notebooks/src) | ✅ (process, not tracked) |
| HTML notes (×2) | preprocessing + Problem 1 reviews | ✅ (superseded by md + prompts) |

## Status semantics

- `captured` in Phase 0 means the item is **inventoried and mapped to a planned target**, not
  implemented. `track_P_module` / `track_R_module` / `acceptance_test` fields hold planned
  Phase 1–7 targets; `status = planned`.
- The independent audit must re-open each source document and attempt to find an uncaptured
  formula, numeric claim, assumption, figure, table, or conclusion. Any such finding is a
  gate failure for Phase 0.
