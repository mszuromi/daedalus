#### Fixed: non-local noise kernels with slow tails (silent-wrong since the kernel-derived τ box), quadrature error in G6, a budget for the quadrature fallback

**Numbers move** for non-local noise sources (`declare_cgf_term(..., markovianize=False)`) whose kernel decays slowly
(a power-law tail). The τ box read off the kernel (wave 3c) reaches ~1e4 for `1/(1+τ^4)` and ~1e8 for `1/(1+τ^2)`,
while the kernel's peak is O(1) wide. The hardened quadrature had no breakpoint near the peak, so it sampled past
it. With equal external times the only breakpoint is 0, and the peak sat at the end of two panels as wide as the
box; QUADPACK's nearest node lies 0.0022 panel widths from an end, so it returned 0 with error 0, accepted. The
status stayed 'ok'. Tree of `x' = -x + η`, `D = 1/2`, before → after (exact: mpmath convolution):

| kernel | τ | before | after = exact |
|---|---|---|---|
| `1/(1+τ^4)` | 0 / 0.7 / 3 | 0.6304769 / 0.3593010 / 0.0786357 | 0.6304778 / 0.5398611 / 0.0814718 |
| `1/(1+τ^2)` | 0 / 0.7 | 0.0 / 0.326 | 0.6214 / 0.542 |
| `(1+τ^2)^(-3/4)` | 0 | 0.0 | 0.6792 |
| `1/(1+(τ/40)^2)` | 0 | 0.0 | 0.9988 |
| `(1+τ^2)^(-1/2)` | 0 | 0.0 | finite (the kernel's integral diverges, C does not: the line decays) |

After: ≤ 1.8e-13 relative against mpmath at τ = 0, 0.7, 3, 10, and for a time-rescaled twin (S = 100).

- **The fix.** When a source's box is more than 64 core widths (`_noise_kernel_scales`: the |τ| where the kernel
  falls to half its peak), its τ variable gets the breakpoints ±core·16·4^j out to the box (`_noise_kernel_ladder`).
  Kernels that decay fast (`exp(-|τ|)`, `exp(-τ^2)`) get none, so they integrate exactly as before.
- **G6.** The hardened fallback's QUADPACK error estimates now enter the accuracy estimate (counter
  `nquad_hardened_err_abs_sum`): each level's own estimate plus its inner levels' estimates, integrated over its
  nodes. This share is judged against 1e-8 (the fallback's own contract) unless `question.accuracy` is declared;
  `G6_ACCURACY_LOW` names it when it exceeds that. Before, quadrature regions were not in the estimate at all.
- **A budget.** A hardened region that needs more than `NQUAD_EVAL_BUDGET = 20,000,000` integrand evaluations
  raises `PhaseJQuadratureBudgetError` (a `DaedalusError`) that names the region and the options; `None` removes the
  limit. Before, the one-loop colored-noise OU (exp, gaussian and power-law kernels alike) ran for more than 30 minutes
  (exp: over 40, killed), with QUADPACK non-convergence. Now it is refused after about 3 minutes. The counter `nquad_hardened_evals_max` records
  the largest region.

Not refused: a bounded kernel whose integral diverges (`(1+τ^2)^(-1/2)`) gives a finite, correct C wherever the
lines decay. A diverging zero-frequency noise power only shows where nothing filters it.

Zoo A/B (17 public entries that run, `use_cache=False`, base main a46ddcf): all bit-identical, analyzer codes
unchanged; five more refuse identically on both sides. No zoo entry has a slow-tailed non-Markov noise kernel.
Full suite: 0 failures.

Tests: `tests/test_noise_kernel_box_quadrature.py` (eight of its ten cases fail on the code before; exp and gaussian
pass on both).
