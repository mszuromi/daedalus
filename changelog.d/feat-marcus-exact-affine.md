#### Marcus rule: exact noise terms for affine couplings, and a sharper truncation advisory

Under `interpretation('marcus')`, a jump source whose coupling is affine in its own field now gets its noise term
exactly from the jump law's mgf. Before, it was a moment series in the jump size. This covers `(a + b x) <> dL`,
and also vector couplings where each component is affine in its own field or constant.

- **The exact term.** The flow is closed form: `phi_z(x) - x = (x + a/b)(e^{bz} - 1)`. So the coefficient of
  `(xt (x + a/b))^n / n!` is `lam E[(e^{bz} - 1)^n]`, built from mgf values. It is kept to `SDE.series_order`
  response legs. A run needs `SDE.series_order >= k + max_ell`, because that is how many response legs its vertices
  can use; `compute_cumulants` refuses a lower order by name. The series route still needs `k + 2*max_ell`.
- **Why it matters.** The old series diverges at degree `n > 1/(|b| th)` for a multiplicative coupling with
  exponential jumps, even though the exact coefficient is finite. Take `b x <> dL` with `Exponential(th)` jumps. The
  advisory fired for essentially every such model, and the high-degree vertices were unreliable. On the exact route
  that model's saddle equals the exact mean to `1e-16`, and the loop series converges to the exact covariance with
  the `ell >= 4` tail (`3.1e-8` at `max_ell = 3`). No advisory fires.
- **No cancellation for the common laws.** Exponential, Poisson and other exp-sum or rational mgfs give one-fraction
  coefficients, for example `lam n! th^n / prod_j (1 - j b th)`.
- **Laws with no one-fraction form** (Gaussian or non-integer Gamma jumps) give an alternating mgf sum. That sum can
  lose digits at small `b`. `compute_cumulants` measures this rounding and warns past `1e-10`, naming the remedy:
  the new `SDE.marcus_route('series')` (also `sde.marcus_route` in configs). The series converges fast there.
- **Missing mgf.** If the jump law's mgf does not exist where a usable vertex needs it (`E[e^{n b z}] = inf`), the
  run is refused before the mean-field solve. The message names the degree. A degree that `(k, max_ell)` cannot
  reach is recorded as `mgf_degree_limit` instead. Poles of even order (Gamma(2) jumps, for example) are found too:
  the check uses the real roots of polynomial bases and the unexpanded CGF. The refusal is conservative: a
  covariance that happens to need only low mgf values, such as one with a linear drift, is refused as well.

Other couplings (non-affine ones, or affine ones that read another moved field, such as a rotation) keep the series
route.

The advisory (`result['marcus_truncation']`, `MarcusTruncationWarning`, G6) now counts only the vertices the run can
use. Each term is expanded about the saddle into vertices `xt^r dx^q`, and a vertex counts only if
`r + q <= k + 2*max_ell` (loop counting) and `r <= k + max_ell` (loop grading of the response legs). For example, an
omitted degree-8 vertex no longer warns at `k = 2, max_ell = 1`. Rows now carry `route` (`'series'` or `'exact'`)
and `admissible`, and `relative_by_degree` is keyed by the response degree `r`.

Gaussian-only and additive models are unchanged, still byte-identical to `'stratonovich'` and Ito.
