# Phase 0 autonomous implementation brief

Read `CLAUDE.md`, `PLAN.md`, `docs/METHOD_SPEC.md`, `docs/ACCEPTANCE.md`, and `docs/RUNBOOK.md` in full.
Read the paper source `../../paper_Latex/2107542.tex`, final review
`../../review/notes/review_all.md`, and inventory the relevant legacy code/data without modifying them.

Execute only Phase 0. You have autonomy to create/edit files inside this repository, create `.venv`,
install declared Python dependencies locally, run safe read-only inspection commands, run tests, fix
failures, self-review the complete diff, and make local checkpoint commits. Do not push or modify any
file outside this repository.

Required deliverables:

1. Python 3.11+ package/test/config skeleton with reproducible dependency declaration.
2. Immutable input manifest with SHA-256 hashes and a command to revalidate it.
3. Data dictionary and legacy asset/producer inventory.
4. Paper traceability inventory, review traceability inventory, and explicit conflict matrix.
5. Registered paper-output baseline: metric/figure/table, source location, legacy producer, expected
   value or visual target, tolerance proposal, and current status.
6. Run-manifest schema recording track, config, input hashes, git commit, environment, seeds, command,
   timing, status, and output hashes.
7. Formatting, linting, unit-test, manifest-validation, and smoke-test commands.
8. Phase 0 acceptance packet containing evidence, not just a narrative claim.

Before finalizing, perform a hostile self-review: inspect all changed files; try to falsify the input
inventory and traceability completeness; verify no legacy file changed; verify no secret or generated
environment is tracked; run every Phase 0 gate; fix all in-scope failures. Record unresolved conflicts
in `docs/DECISIONS.md` and continue independent work.

Stop only for the conditions in `docs/RUNBOOK.md`. At completion, report commits, files, commands,
tests, inventory counts, known gaps, decisions, runtime, available token/cost information, and the exact
independent-audit command. Do not enter Phase 1.
