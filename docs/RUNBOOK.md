# Owner runbook

This is the human operating procedure. The owner approves scope and evidence; Claude Code performs
bounded autonomous implementation and self-review.

## One-time local setup

1. Work on branch `reproduction/main`, never directly on `main`.
2. Start Claude Code from this `repo/` directory so writes are confined here.
3. Keep raw data, paper, review, and legacy code outside this directory and read-only.
4. Do not push until Phase 0 establishes CI and a secret-free clean repository.

## Per-phase operating loop

1. **Plan run:** ask Claude to inspect the phase, produce an executable plan, tests, risks, and stop
   conditions. No edits.
2. **Owner review:** approve only scope, formulas, acceptance gates, and unresolved decisions—not each
   ordinary file edit.
3. **Autonomous implementation run:** Claude implements, tests, self-reviews its diff, fixes failures,
   writes evidence, and makes local checkpoint commits. It may continue without confirmation for safe
   in-scope work.
4. **Independent verification run:** start a fresh Claude session with no implementation narrative;
   ask it to audit the phase against source documents, tests, and git diff. It must not edit first.
5. **Owner acceptance:** inspect the acceptance packet and several hand-checked examples. Mark the phase
   accepted only when its gate passes.

## When Claude must stop

- destructive command, raw-data modification, secret handling, network publication, push, or merge;
- formula, sample definition, target metric, or conclusion change;
- paper/review/code conflict not already represented as Track P and Track R;
- failed gate that cannot be corrected without changing scope;
- suspected leakage, circular evaluation, non-identifiability, or numerical instability.

Claude should not stop for routine file creation, formatting, dependency installation in `.venv`,
test execution, safe refactoring, documentation, or local checkpoint commits within `repo/`.

## Git and GitHub progression

Use local Git first. A phase is developed in small local commits. After Phase 0 passes, create the
GitHub branch and push it, then use pull requests and CI. Never put the DeepSeek token in the repo or
GitHub workflow. CI should reproduce tests deterministically; AI review in GitHub is optional and
separate from CI correctness.

Recommended flow:

`local branch -> local tests -> local independent audit -> push branch -> GitHub CI -> pull request ->
human acceptance -> merge`

## Owner acceptance checklist

- Scope matches the active phase.
- Paper-faithful and review-corrected outputs are distinctly labeled.
- Tests fail when a hand-worked example is deliberately broken.
- No generated conclusion exists without a source table and run manifest.
- No training/test leakage or outcome-conditioned metric is mislabeled predictive.
- Re-running the documented command produces results within tolerance.
- Git diff contains no raw data, secret, virtual environment, cache, or unrelated file.
- Cost, runtime, failures, assumptions, and unresolved risks are reported.
