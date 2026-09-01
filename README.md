# DWTS MCM Reproduction

This directory is the clean, reproducible implementation of the submitted DWTS MCM paper, with a
separate review-corrected variant where the final review notes identify a methodological issue.

## Authority order

1. `../paper_Latex/2107542.tex` defines **Track P**, the primary faithful paper reproduction.
2. `../review/notes/review_all.md` defines **Track R**, the reviewed/corrected variant.
3. Original code in `../src/` and `../review/srcs_0/` is implementation evidence used to resolve how
   the paper was actually computed, but it may not silently override the written paper.
4. When paper and review disagree, implement both, label them clearly, and compare their assumptions,
   outputs, and conclusions. Never merge them into an undocumented hybrid.

The legacy directories and raw data must never be modified by implementation work.

## Workflow

Read `CLAUDE.md` and `docs/RUNBOOK.md`, then execute `PLAN.md` one gated phase at a time. Every phase ends with tests,
recorded outputs, and a small Git commit. Do not start the next phase until the current acceptance
gate passes.

## Intended repository layout

```text
configs/                 versioned experiment settings
docs/                    specification, decisions, and acceptance criteria
src/dwts_reproduction/   reusable Python package
tests/                   unit, invariant, integration, and regression tests
scripts/                 thin command-line entry points
outputs/                 generated tables, figures, metrics, and run manifests
```

Generated outputs are not evidence unless their run manifest records the configuration, random
seed, input hashes, Git commit, environment, and command used.
