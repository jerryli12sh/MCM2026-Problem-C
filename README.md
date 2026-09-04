# MCM 2026 Problem C — auditable DWTS re-analysis

This repository turns our *Dancing with the Stars* MCM solution into a reproducible,
interviewer-friendly case study. It preserves the original modeling logic, implements a stricter
review-corrected alternative, and records where the submitted paper could—or could not—be reproduced.

The best starting point is [`review/notes/review_all.md`](review/notes/review_all.md). That Chinese
note is the conceptual spine of the project: it reframes the task as **inverse mechanism design**,
defines the latent fan-vote problem, explains the competing elimination rules, and proposes the
counterfactual mechanism. The code, tests, decisions, and evidence in this repository turn that note
into an auditable implementation.

For a detailed Chinese tour of every directory and file type, see
[`docs/REPOSITORY_GUIDE.zh-CN.md`](docs/REPOSITORY_GUIDE.zh-CN.md).

## The question in one minute

DWTS reveals judge scores and elimination outcomes, but not fan-vote totals. The analysis therefore
does not simply predict an observed label. It asks:

1. What latent weekly fan-vote shares are compatible with the observed eliminations?
2. How would rank, percentage, and Bottom-2 judges'-save rules change those outcomes?
3. Which celebrity and professional-partner characteristics relate differently to judge and fan
   signals?
4. Can a new rule balance technical fairness with meaningful audience influence?

The basic weekly object is a probability vector over contestants still alive in the competition:

```text
known judge signal J + unknown fan share p -> elimination mechanism -> observed elimination Y
```

Because many different values of `p` can explain the same `Y`, the project uses a pooled support
model and a Dirichlet distribution rather than pretending the fan vote is uniquely observed.

## Two analysis tracks

| Track | Purpose | Statistical treatment |
|---|---|---|
| **P — paper-faithful** | Reproduce the submitted solution and its legacy computation | Fits a pooled support center from eliminations, then conditions weekly fan shares on the same eliminations. Useful for historical explanation, but its fit metrics are in-sample/internal. |
| **R — review-corrected** | Test the main correction proposed in `review_all.md` | Integrates weekly fan shares out of the elimination likelihood, so each elimination is used once. Structurally cleaner, but not automatically higher-scoring. |

The two tracks are never silently mixed. Configurations, outputs, metrics, and manifests carry their
track label. See [`docs/METHOD_SPEC.md`](docs/METHOD_SPEC.md) for notation and
[`docs/DECISIONS.md`](docs/DECISIONS.md) for the 24 recorded modeling decisions.

## Results and honest limits

| Quantity | Track P | Track R |
|---|---:|---:|
| Top-1 elimination reconstruction | 0.9495 | 0.8349 |
| Mean weighted PCP | 0.6043 | 0.5342 |
| Mean relative credible-interval width | 3.117 | 3.378 |
| Season-path consistency, `S-bar` | 0.7785 | 0.6331 |

Track P's higher reconstruction score is not evidence that it is the better predictive model: it
uses the observed elimination in both fitting and weekly posterior conditioning. Track R pays the
expected fit cost of removing that double use.

Important negative findings are kept, not polished away:

- the paper's XGBoost target `0.806554` was not reproducible from the available legacy code/data;
- the claimed ranking-gap `R² > 0.6` reproduced as `R² = 0.2704` (`n = 421`);
- the review's “percentage + Bottom-2 is best” claim was not supported by the reproduced data;
- some actor, partner, and interaction effects were directional only rather than statistically
  conclusive;
- inferred fan shares are posterior estimates constrained by outcomes, never ground truth.

The reconciled status is in [`docs/STATUS.md`](docs/STATUS.md). A representative ten-figure snapshot
is in [`evidence/figures/`](evidence/figures/), with provenance explained in
[`evidence/README.md`](evidence/README.md).

## Repository structure

```text
review/notes/review_all.md   conceptual core and review-corrected specification
src/dwts_reproduction/      reusable, typed analysis package
  preprocess.py             alive sets, structural zeros, elimination events
  problem1/                 latent fan-support inference: Tracks P and R
  problem2/                 rule replay and controversy case studies
  problem3/                 celebrity/partner pathway regressions
  problem4/                 counterfactual rule simulations
  sensitivity/              robustness analyses
  release/                  registered baseline comparison
scripts/                    command-line entry points for runs, plots, and manifests
configs/                    human-readable path and model configuration
tests/                      unit, invariant, numerical, and integration tests
manifests/                  machine-readable provenance and traceability records
evidence/                   committed headline figures for readers
docs/                       methods, findings, decisions, data dictionary, and verification
outputs/                    regenerated tables/figures; intentionally ignored by Git
pyproject.toml              package metadata, dependencies, and tool configuration
Makefile                    short commands for install, checks, and release runs
.github/workflows/ci.yml    source-free public continuous-integration gate
```

The full data flow is:

```text
external read-only source bundle
        |
        v
canonical preprocessing -> Problem 1 P/R -> Problems 2–4 + sensitivity
        |                                         |
        +----------------> generated outputs <----+
                                      |
                                      v
                      baseline comparison + evidence snapshot
```

## Reproduce or inspect

Python 3.13 is the verified environment; Python 3.11–3.13 is supported.

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[analysis,dev]"
```

For the exact recorded dependency set, install `requirements-lock.txt` first and then install this
package with `pip install -e . --no-deps`.

Run the public, source-free checks:

```bash
make check
```

The full numerical pipeline also needs the private/read-only source bundle containing the official
data, submitted paper source, review materials, and legacy implementation. Keep that bundle outside
this repository and either place it in the parent directory or point `DWTS_SOURCE_ROOT` to it. Then:

```bash
make verify-data       # data-bound tests, input hashes, and smoke checks
make release           # regenerate the complete 19-stage release (~23 minutes on the recorded Mac)
```

The public repository intentionally does **not** redistribute the contest dataset, paper source, or
legacy workspace. This keeps provenance clear and avoids making an unsupported licensing decision.
See [`docs/ENVIRONMENT.md`](docs/ENVIRONMENT.md) and [`docs/CI.md`](docs/CI.md) for the exact boundary.

## What this project demonstrates

- translating an informal modeling argument into testable statistical software;
- recognizing identification limits and leakage/double-use in latent-variable inference;
- preserving a paper-faithful baseline while implementing a competing corrected model;
- rule replay and Monte Carlo counterfactual mechanism design;
- sensitivity analysis, traceability, reproducible manifests, and explicit negative results;
- software-engineering discipline: typed modules, deterministic seeds, CI, tests, and generated
  artifacts separated from source evidence.

## Reuse and citation

No open-source license is granted in this repository. The code is visible for portfolio,
verification, and academic-review purposes; reuse requires the owner's permission. The official
contest dataset and submitted paper are not included. When discussing the work, cite both this
repository and the original MCM submission rather than treating inferred fan-vote shares as observed
data.
