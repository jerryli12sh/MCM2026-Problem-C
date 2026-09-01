# Phase 0 independent audit brief

Audit Phase 0 from scratch. Do not trust the implementation session's summary and do not edit first.
Read the paper, review, repository instructions, Phase 0 gate, git history, complete diff, tests, input
manifest, traceability inventories, conflict matrix, and acceptance packet.

Try to disprove completion. Look for missing paper claims/figures, missing review requirements,
paper/review conflicts silently merged, mutable or unhashed inputs, incorrect paths, weak tests,
unregistered outputs, secrets, generated environments, misleading predictive language, and changes
outside `repo/`. Run all Phase 0 verification commands independently.

Return findings ordered by severity with file/line evidence, then a gate decision: PASS, PASS WITH
CONDITIONS, or FAIL. Do not fix findings unless the owner explicitly starts a separate remediation run.
