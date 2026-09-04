# Development journey

The repository was rebuilt in seven evidence-driven stages. This page keeps the useful engineering
story without retaining temporary agent prompts or stale phase checklists.

| Stage | Main work | Durable evidence |
|---|---|---|
| 0. Inventory and provenance | Classified 174 source files, hashed immutable inputs, mapped paper/review requirements, registered conflicts and baselines | `manifests/`, `DATA_DICTIONARY.md`, `CONFLICT_MATRIX.md` |
| 1. Canonical preprocessing | Parsed result states, inferred season/activity horizons, converted structural zeros, built alive sets and elimination events | `preprocess.py`, preprocessing tests |
| 2. Latent fan support | Implemented the paper-faithful two-stage fit and review-corrected integrated likelihood | `problem1/track_p.py`, `track_r.py`, numerical tests |
| 3. Evaluation and uncertainty | Added reconstruction, PCP, credible intervals, season paths, XGBoost comparison, and crowded-field analysis | `problem1/evaluate.py`, `baselines.py`, evidence figures |
| 4. Mechanism replay | Implemented rank/percentage and direct/Bottom-2 rules, including ties and named cases | `problem2/`, hand-worked rule tests |
| 5. Mechanism design | Simulated fan compression, judge amplification, momentum, and judges' save | `problem4/`, claim checks and simulations |
| 6. Explanation and robustness | Modeled celebrity, partner, surprise/growth effects and varied key assumptions | `problem3/`, `sensitivity/` |
| 7. Release engineering | Added a 19-stage driver, 20 registered comparisons, environment capture, CI, representative evidence, and audit documentation | `run_release.py`, `release/compare.py`, `evidence/`, `VERIFICATION.md` |

## Decisions that mattered

The most important work was not adding more model complexity. It was making ambiguous choices
visible:

- official season rules and legacy code did not always agree, so both mappings were labeled;
- the paper reused elimination outcomes during fit and posterior conditioning, so the behavior was
  preserved as Track P and corrected separately as Track R;
- negative reproduction results were registered instead of forcing a target match;
- plot jitter was separated from fitted data, ties received deterministic policies, and all random
  procedures received fixed seeds;
- every figure was changed to read saved tables rather than hidden live state;
- raw sources remained external and hash-verified.

The full decision log is [`DECISIONS.md`](DECISIONS.md). It is deliberately more detailed than this
summary because it records alternatives, evidence, consequences, and affected files.

## What changed from the competition workspace

The original workspace mixed notebooks, one-off scripts, data, compiled paper files, figures,
caches, and exploratory outputs. The publishable repository extracts the stable intent into typed
functions, keeps command-line wrappers thin, separates source from generated artifacts, and attaches
tests to the fragile parts of the analysis. This is why the public tree is smaller than the original
workspace while preserving more verifiable knowledge.
