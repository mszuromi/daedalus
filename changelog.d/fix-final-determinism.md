#### Results are byte-identical across processes; direct tests of the mean-field root tolerance; doc fix (final determinism pass)

- **Cross-process determinism (FINAL N9).** Runs were identical within a process but scattered by 1-4 ulp from one process
  to the next (dendritic 4 variants, spike_reset 2-3, cubic_alpha 2, quad_exp rarely), whatever `PYTHONHASHSEED` was.
  Root cause: Pynac (Sage's symbolic engine) orders the operands of a product by a symbol hash that is drawn afresh in every
  process, so `(a1*a2*vstar2).subs({a1: 2.5, a2: 3.7, ...})` multiplies the floats in a process-dependent order. A
  four-line script reproduces it. The Phase J prefactor of every diagram was formed that way (`cp.subs(num_params)`), so
  the last bit of the prefactor, and of everything built from it, differed between processes. `engine.core.detnum.numeric_subs`
  now evaluates a closed form in the parameters itself: Python doubles, product operands in a canonical order (sorted by
  text), sums rounded exactly (`math.fsum`); an expression with a symbol the substitution does not cover falls back to
  `.subs`. The prefactor sites of `final_integral.py` use it, and so do the other numeric substitutions there (kernel rates, noise-kernel
  coefficients, the retardation probe); the ones with a free time or frequency symbol keep `.subs`. Verified: spike_reset one loop 9 of 9 processes byte-identical
  (3 variants before), dendritic and quad_exp 6 of 6 (4 and 2 before). Zoo values against the previous main
  (one process each): dendritic 1.5e-16, quad_exp 1.4e-16, quadratic_hawkes_alpha 1.2e-16 relative, every other entry
  bit-identical; three separate serial processes of the whole zoo agree to the byte on every entry that runs. New tests (`tests/test_audit_w9_determinism.py`) run separate processes
  and compare bytes: the substitution on its own, and the one-loop spike_reset cumulant end to end.
- **Direct tests of the mean-field root tolerance (FINAL N12).** `_ROOT_RTOL` 1e-7 -> 1e-4 survived every mean-field test
  file (1e-2 was killed by one indirect test). `tests/test_audit_w9_mf_mutants.py` feeds `_refine_root` closed-form
  systems (two rows disagreeing by a relative 3e-5 / 3e-3, the same in units of 1e9 and 1e-9, and a 1e-8 gap that must be
  accepted) and kills the mutants 1e-4, 1e-2 (too loose) and 1e-9, 1e-12 (too tight).
- `docs/phase_j_unbounded_exact.md` no longer describes the retired Phase J switches in the present tense (FINAL N10).
- **The deduplication representative no longer depends on `PYTHONHASHSEED` (FINAL decision W3, audit CON-10).** The external-leg
  permutations of a typed-diagram enumeration came from a `set` of tuples of strings, whose iteration order changes with the
  hash seed; the deduplication keeps the first diagram of each class, so a two-field model's loop cumulant took two byte
  variants across seeds 0-5. `type_assignment._distinct_permutations` now returns them in a canonical content order
  (`sorted`). Verified: the two-field one-loop and two-loop cumulants are byte-identical across seeds 0-5 (two variants
  before); eight zoo entries (spike_reset, linear_hawkes, multipopulation_test, quad_exp, dendritic, ou_quartic_colored,
  quadratic_hawkes_alpha, ou_quartic k3 l2) identical across seeds 0-3. No frozen baseline needed refreezing: every
  baseline test passes at its tolerance. New test in `tests/test_audit_w9_determinism.py` (six processes, seeds 0-5).
