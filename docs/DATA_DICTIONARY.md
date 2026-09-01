# Data dictionary

Canonical unit of analysis: contestant-season-week `(contestant_id, season, week)`. Raw and
derived data live under `../data/` (read-only); the review's independent rebuilds live under
`../review/srcs_0/` and `../review/problem1_rebuild/outputs/`. This document describes the
semantics of each table and its key columns. File/column names here are the raw/legacy names;
the clean implementation may use stable internal identifiers.

## 1. Raw input

### `../data/2026_MCM_Problem_C_Data.csv` — contestant-season wide table (`raw_input`)

Grain: one row = one celebrity in one season. Shape `421 × 53`.

| column | type | meaning |
|---|---|---|
| `celebrity_name` | str | contestant display name |
| `ballroom_partner` | str | professional partner name |
| `celebrity_industry` | str | industry category (Actor/Actress, Athlete, …) |
| `celebrity_homestate` | str | US state |
| `celebrity_homecountry/region` | str | country/region (note: legacy `/` in name) |
| `celebrity_age_during_season` | num | age during the season |
| `season` | int | season index, 1..34 |
| `results` | str | textual outcome: `1st Place`, `Eliminated Week k`, `Withdrew`, … |
| `placement` | int | final placement |
| `week{1..11}_judge{1..4}_score` | num | judge score (44 columns); `N/A` = missing judge/season-not-run; literal `0` = structural placeholder after elimination |

Data facts (from `../data/数据说明.md`): 44 score columns; 4671 literal zeros (structural
post-elimination fillers, not real scores); 4741 `N/A`. Season 1..34, variable weeks and
contestants per season.

### `../data/2026_MCM_Problem_C.pdf` — problem statement (`raw_input`)

Official 2026 MCM Problem C statement (DWTS). Defines the two aggregation rules (rank vs.
percentage) and the Bottom-2 judges' save, and the four tasks.

### Documentation (`raw_input`)

- `../data/数据说明.md` (`data的说明.md`) — the five-table design and their grain.
- `../data/超参数.md` — the eight hyperparameters and their roles (see §5).
- `../data/题面中文翻译.md` — Chinese translation of the problem statement.

## 2. Derived legacy tables (in `../data/`, `legacy_output`)

### `df_clean.csv` — cleaned contestant-season master + wide scores

Grain: one row per season × celebrity. Adds parsed outcome fields and structural-zero cleaning:

| column | meaning |
|---|---|
| `elim_week_result` | parsed from `Eliminated Week k` |
| `is_withdrew` / `is_place` | flags distinguishing withdrawal vs. final placement |
| `season_max_week` | last week any contestant has a positive judge total |
| `active_until` | per-contestant activity horizon |
| `last_week_positive` | last week with a positive score |
| `week*_judge*_score` | structural post-elimination zeros converted to `NaN` |

### `df_long_judge.csv` — judge-level long table

Grain: `(season, celebrity_name, week, judge)`. Columns: `judge_score`, `is_show_week`,
`eligible`, plus copied static attributes.

### `df_weekly.csv` — weekly aggregation (modeling base)

Grain: `(season, celebrity_name, week)`. Columns: `total_judge_score`, `mean_judge_score`,
`n_judges_scored`, `performed` (positive total), `eligible`, `judge_rank` (lower is better),
`judge_percent` (share of alive-set judge total), plus static attributes.

### `df_roster.csv` — weekly alive-set table

Grain: `(season, celebrity_name, week)`. Columns: `eligible`, `season_max_week`,
`eligible_next`. Defines `A_{s,t}` (the weekly competition set).

### `df_elim_events.csv` — elimination-event table

Grain: `(season, week_end)`. Columns: `eliminated` (list of names, supports multi-elimination),
`is_final_week_end`, `m_elim` (count). Finale endings are marked so they are not treated as
ordinary single-elimination events.

### Fan-vote / judge-normalized outputs

- `df_clean_with_vote.csv`, `df_clean_with_vote_judge_norm.csv` — `df_clean` joined with the
  inferred fan-vote and judge-normalized signals produced by the legacy Problem 1 model.

### Simulation / mechanism outputs

- `contestant_archetypes.csv` — `season, celebrity_name, delta_mean, n_weeks, archetype`
  (`relative_popular` / `relative_technical` / `balanced`).
- `metrics_b2_save.csv` — per-contestant Bottom-2+Save metrics (`p_b2`, `p_rev`, `dE_T`,
  `dP_finals`, …).
- `sim_case_summary.csv` — per-case Monte-Carlo summaries by scheme (`mean_rank`,
  `mean_alive_rate`, `final_alive_rate`).
- `sim_summary.csv` — scheme × week × archetype summaries (`alive_rate`, `avg_rank`,
  `elim_rate`, `n`).
- `data_3.csv` — Problem 3 pooled contestant-season table (judge/fan pathways).

## 3. Review rebuild outputs

### `../review/srcs_0/` (`legacy_impl` + `legacy_output`)

Independent preprocessing rebuild (`dwts_preprocess.py`, `0_dwts_preprocess.py`) that
reproduces the five tables (`df_clean`, `df_long_judge`, `df_weekly`, `df_roster`,
`df_elim_events`) plus `zero_audit.csv` and `validation_report.csv`.

### `../review/problem1_rebuild/outputs/` (`legacy_output`)

Independent Problem 1 rebuild (`problem1_fan_support.py`): `problem1_panel.csv`,
`problem1_train_weeks.csv`, `problem1_posterior_summary.csv` (weekly `p_mean`, CI, ESS, PCP),
`problem1_accuracy_by_week.csv`, `problem1_accuracy_by_season.csv`, `problem1_summary.json`
(overall Top-1 accuracy `0.949541`), `problem1_fit_metadata.json`, `problem1_fit_arrays.npz`.

## 4. Legacy figures

`../figure/` (`legacy_output`) — `fig_rank_vs_pct_*.png`, `fig_sim_trend_*.png`,
`fig_rev_heatmap_*.png`, `fig_archetypes.png`, `fig_phase_diagram.png`, and simulation
surfaces. Distinct from the paper's embedded figures in `../paper_Latex/img/` (`paper_spec`).

## 5. Hyperparameters (`../data/超参数.md`)

| name | value | role |
|---|---|---|
| `tau` (train) | 0.05 | softmin elimination temperature (fit) |
| `tau_like` | 0.15 | softmin temperature for importance weights (posterior) |
| `l2b` (`λ_β`) | 0.05 | L2 penalty on feature coefficients |
| `l2u` (`λ_u`) | 0.05 | L2 penalty on contestant-season effects `u` |
| `kappa` (`κ`) | 10.0 | Dirichlet concentration (prior tightness) |
| `lr` | 0.02 | Adam learning rate |
| `steps` | 600 | training iterations |
| `bs` | 32 | batch size (choice sets per step) |
| `B` | 1200 | importance-sampling draws |

Note the paper writes a single temperature `τ`; the legacy code uses two (`tau_train` vs
`tau_like`). This is recorded as a conflict (see `CONFLICT_MATRIX.md`).
