# Acceptance criteria

## Data correctness

- Raw files are immutable and validated by hashes.
- Keys are unique at their declared grain; shares sum to one within numerical tolerance.
- No eliminated contestant appears in a later alive set.
- Structural zero, true zero, and missing judge scores are distinguishable.
- Season finales, withdrawals, double eliminations, and ordinary eliminations are separately counted.

## Statistical correctness

- Unit tests cover normalization, ties, event parsing, likelihood direction, posterior weighting, and
  every mechanism using hand-calculated fixtures.
- Synthetic data tests recover known direction/sign and produce calibrated uncertainty.
- Season-grouped validation prevents contestant/week leakage.
- Random seeds, sample counts, convergence, effective sample size, and Monte Carlo error are saved.
- In-sample explanation is never labeled out-of-sample prediction.

## Reproducibility

- A clean environment can run preprocessing, tests, a fast end-to-end sample, and the full pipeline.
- Each run writes configuration, input hashes, Git commit, package versions, seed, timing, and status.
- Tables are the source for figures; no manual figure edits are required.
- Important numerical outputs have explicit absolute/relative tolerances and reasons for them.

## Review traceability

Create two traceability inventories and a conflict matrix. Every material paper method, result, figure,
and claim must map to code, test, generated artifact, tolerance, and status. Every review requirement
must map to the same fields. A release is unacceptable while a required row is unmapped, silently
changed, or supported only by a notebook cell.

For every paper/review conflict, acceptance requires: paper statement; review statement; legacy-code
evidence; Track P behavior; Track R behavior; numerical impact; conclusion impact; and decision ID.

Track P must reproduce registered paper numbers/figures within explicit tolerance or document why it
cannot. Track R must pass its own statistical tests; it is not required to match paper numbers.

## Human acceptance packet

For every phase, present: scope completed; file diff; test results; old-versus-new metrics; visual
comparison where relevant; runtime and API cost; assumptions/decisions; known failures; exact rerun
command. The user approves formula and conclusion changes explicitly.
