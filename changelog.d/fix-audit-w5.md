#### Test-suite precision (audit wave W5)

Tests, CI and the environment only; no library code changed and no public number moved.

- **A public exact-oracle suite runs in CI** (`tests/test_public_oracles.py`, references in `tests/oracles_public/refs.py`,
  mpmath only, no package code): linear Gaussian SDEs (non-normal 3-field drift, correlated noise, a Jordan block, poles
  1e-7 apart; Lyapunov + expm), Levy-driven linear SDEs at k = 2..4 (closed form), linear Hawkes rates and two-point
  functions (Markov embedding), 1-D polynomial drifts loop by loop at unequal times (backward generator; the Boltzmann
  series 1, -3, 24, -297), the delayed OU process (exact BVP), Erlang kernels, the switch-on transient (Van Loan),
  colored noise (exact one-loop mass shift) and the 1-D Allen-Cahn tree and one loop. Tolerance 1e-12 relative to the
  cell's scale on both the state-space and the pole engine. The equal-time
  one loop (2e-6..7e-6 off before the exact equal-time loop rule) agrees to 3.7e-13.
- **Mutation-killing tests** (`tests/test_audit_w5_mutation_kills.py`) for the audit's surviving mutants: Laurent
  residues at double and triple poles, the pole-cluster cancellation factor, the Taylor degree selection of the
  state-space exponential, and the exact delayed tree (rational K0, dynamic delayed K1, against a spectral integral; the
  neutral-system refusal). Each test was checked to fail on its mutant.
- **Tolerances tightened to the measured accuracy**: 327 assertions that were >= 100x looser than the worst error the
  suite achieves (or exact under a loose tolerance) now sit at max(1e-12, the decade above 100x the worst error on
  either the local or the CI stack); 12 were left alone (semantic thresholds, or already at the floor). Ten `pytest.approx` checks on values far
  below 1 get `abs=0` (the default `abs=1e-12` made them vacuous, e.g. a 2e-20 residue), and a stochastic check whose
  expected value was below its own 5-sigma band now runs at a lag where it is not. Three assertions that ran within
  2x of their tolerance on CI's stack got a decade of room.
- **New tests for untested public paths**: `dd.prediagram_mappings`, `RationalKernel` at a delta contact,
  `Cumulant.smooth`, phase-type jumps from floats and a symbolic rate, the simulator's delayed-kernel ring buffer and
  `record=`, a lift-the-bound test for the tree enumeration's |V2| bound, exact counts in the type-assignment tests,
  and the README's Python snippets run as written (`tests/test_readme_snippets.py`).
- **Slow tests run weekly and on demand** (`.github/workflows/slow-tests.yml`, three shards); the two tests whose
  footprint exceeds 12 GB are marked `bigmem` and run only when asked. The analyzer's cold-cache counterparts of its
  warm-cache skips are slow tests, so they now run there. A test that needs a git checkout skips outside one.
- **`environment.yml` pins the exact stack CI runs** (python 3.12, numpy 2.3.5, scipy 1.17.1, sympy 1.13.3,
  numba 0.65.1; the supported ranges stay in a comment), and CI prints the resolved versions.
