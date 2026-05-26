"""
field_theory_sage.py
====================
Model-agnostic MSRJD field theory expansion framework — SageMath version.

The action is specified as a callable returning an SR expression. All field
variables are SR symbolic variables. The framework automatically Taylor-expands
every nonlinear function of the fields (exp, tanh, formal functions, ...) to
the requested order, renames formal-function derivative symbols to clean names,
applies MF background conditions, then classifies terms by bigrade (n_tilde, n_phys)
using PolynomialRing(SR, ...) exponent vectors.

Usage
-----
    load('field_theory_sage.py')
    load('models/hawkes_sage.py')

    ft = FieldTheory(HAWKES_MODEL, taylor_order=4)
    ft.expand()
    ft.sanity_check()
    ft.summary()
"""

import warnings

from sage.all import (
    SR, PolynomialRing, factorial, QQ, latex, LatexExpr,
    diff, function, exp, dirac_delta, heaviside, integrate, oo, I, pi, taylor
)
from IPython.display import display, Math as _Math


def fourier_transform(f, t, s):
    r"""
    Symbolic Fourier transform (angular-frequency convention):

        F(s) = \int_{-\infty}^{\infty} f(t) e^{-i s t} dt

    No 2π in the exponent.  Gives  δ(t) → 1,  δ'(t) → iω.

    For causal integrands of the form ``g(t) * heaviside(t)`` — which
    is virtually every neural-style kernel — we replace ``heaviside(t)``
    with ``1`` and restrict the integration to ``[0, ∞)``.  This avoids
    Maxima's request for an explicit sign on ``omega`` (it can't decide
    where the heaviside argument cuts unless told).  For non-causal
    integrands the full real line is used.

    Tries the SymPy backend first (it handles ``positive=True``
    assumptions on time constants automatically) and falls back to
    Maxima's default integrator.
    """
    f = SR(f)

    # Detect the causal case: replace heaviside(t) with 1 and split
    # the integration domain at zero.  Bounds are wrapped in SR() so
    # the SymPy integrator (which calls ``a._sympy_()``) accepts them.
    has_heaviside_t = bool(f.has(heaviside(t)))
    if has_heaviside_t:
        integrand = f.subs({heaviside(t): 1}) * exp(-I * s * t)
        a, b = SR(0), oo
    else:
        integrand = f * exp(-I * s * t)
        a, b = -oo, oo

    # SymPy first (better at handling symbolic-positive parameters);
    # then Maxima.  If both fail, return the unevaluated integral.
    for algo in ('sympy', 'maxima'):
        try:
            return integrate(integrand, t, a, b, algorithm=algo)
        except (ValueError, RuntimeError, TypeError):
            continue
    return integrate(integrand, t, a, b)


def inverse_fourier_transform(F, s, t):
    r"""
    Inverse Fourier transform (angular-frequency convention):

        f(t) = \frac{1}{2\pi} \int_{-\infty}^{\infty} F(s) e^{i s t} ds

    Paired with fourier_transform():  FT uses e^{-ist}, IFT uses e^{+ist}/(2π).
    Tries SymPy backend first (handles symbolic parameters without asking for
    assumptions), falls back to Maxima.
    """
    integrand = F * exp(I * s * t)
    for algo in ('sympy', 'maxima'):
        try:
            return integrate(integrand, s, -oo, oo, algorithm=algo) / (2 * pi)
        except (ValueError, RuntimeError):
            continue
    # Last resort: return unevaluated integral
    return integrate(integrand, s, -oo, oo) / (2 * pi)


def _show(expr):
    """Display a SageMath expression or Conv/IP object as rendered LaTeX."""
    display(_Math(latex(expr)))


# ---------------------------------------------------------------------------
# Display helpers — Conv and IP with _latex_() for show()/display()
# ---------------------------------------------------------------------------

class _ConvIPBase:
    def __add__(self, other):
        return _DisplaySum([self, other])
    def __radd__(self, other):
        return _DisplaySum([self]) if other == 0 else _DisplaySum([other, self])
    def __sub__(self, other):
        return _DisplaySum([self, _Neg(other)])
    def __neg__(self):
        return _Neg(self)


class Conv(_ConvIPBase):
    """Conv(kappa, f)  — convolution (κ∗f)(t)."""
    def __init__(self, kappa, f):
        self.kappa = kappa
        self.f     = f
    def _latex_(self):
        kappa_str = latex(self.kappa)
        # Wrap sum kernels in parens so (κ₁+κ₂)∗f reads unambiguously.
        # SageMath may return operator.add or its own add_vararg; check __name__.
        # Fallback: if latex contains '+' the kernel is a sum.
        is_sum = False
        try:
            op = self.kappa.operator()
            is_sum = op is not None and 'add' in getattr(op, '__name__', '')
        except AttributeError:
            pass
        if not is_sum:
            is_sum = ('+' in kappa_str)
        if is_sum:
            kappa_str = r'\left(' + kappa_str + r'\right)'
        return r'\left(' + kappa_str + r' \ast ' + latex(self.f) + r'\right)'
    def __repr__(self):
        return f'Conv({self.kappa!r}, {self.f!r})'


class IP(_ConvIPBase):
    """IP(a, b)  — inner product ∫ a(t) b(t) dt."""
    def __init__(self, a, b):
        self.a = a
        self.b = b
    def _latex_(self):
        return latex(self.a) + r'^\top ' + latex(self.b)
    def __repr__(self):
        return f'IP({self.a!r}, {self.b!r})'


class _Neg(_ConvIPBase):
    def __init__(self, inner):
        self.inner = inner
    def _latex_(self):
        return r'-' + latex(self.inner)
    def __repr__(self):
        return f'-{self.inner!r}'


class _DisplaySum:
    def __init__(self, terms):
        self.terms = list(terms)
    def __add__(self, other):
        return _DisplaySum(self.terms + (other.terms if isinstance(other, _DisplaySum) else [other]))
    def __radd__(self, other):
        return self if other == 0 else _DisplaySum([other] + self.terms)
    def __sub__(self, other):
        return _DisplaySum(self.terms + [_Neg(other)])
    def _latex_(self):
        parts = []
        for t in self.terms:
            s = latex(t)
            if parts and not s.startswith('-'):
                parts.append('+')
            parts.append(s)
        return ' '.join(parts)
    def __repr__(self):
        return ' + '.join(repr(t) for t in self.terms)


# ---------------------------------------------------------------------------
# Namespace object
# ---------------------------------------------------------------------------

class _Namespace:
    pass


# ---------------------------------------------------------------------------
# SR polynomial helper  (used for legacy 'taylor_coeffs' path)
# ---------------------------------------------------------------------------

def _poly_taylor(coeffs, x):
    """
    Build Taylor polynomial  sum_n  (coeffs[n] / n!)  x^n  as an SR expression.
    coeffs[n] = f^(n)(background)  — the n-th derivative at the expansion point.
    """
    return sum(SR(c) * QQ(1)/factorial(n) * x**n
               for n, c in enumerate(coeffs))


def _iter_multi_indices(n_args, max_total):
    """Yield every multi-index (k_1, ..., k_{n_args}) of non-negative
    integers with  sum k_j  ≤ ``max_total``.

    Used by the formal-function rename machinery to enumerate every
    partial derivative up to a given total order.  For ``n_args=1``
    this collapses to ``(0,), (1,), ..., (max_total,)`` — i.e. the
    single-argument behavior the framework previously hardcoded.

    Implemented iteratively (via ``itertools.product``) rather than
    recursively so it survives ``%autoreload`` in Jupyter — a
    self-recursive generator would lose its reference to itself
    when the module's globals get hot-swapped.
    """
    if n_args == 0:
        yield ()
        return
    from itertools import product
    for combo in product(range(max_total + 1), repeat=n_args):
        if sum(combo) <= max_total:
            yield combo


def _multi_index_suffix(multi_idx):
    """Encode a multi-index as the rename-target suffix used in
    ``f<suffix>_<i+1>`` symbol names.

    For single-arg (``n_args=1``) the suffix is just the derivative
    order ``<k>`` — keeping the legacy ``f0_1``, ``f1_1``, ...
    naming intact.  For multi-arg we concatenate the per-argument
    derivative orders, giving ``f<k1><k2>...<kn>_<i+1>``.
    """
    return ''.join(str(k) for k in multi_idx)


# ---------------------------------------------------------------------------
# SR → PolynomialRing conversion
# ---------------------------------------------------------------------------

def _sr_to_ring(sr_expr, R, ring_var_names: list):
    """
    Convert an SR expression that is polynomial in ring_var_names to a
    PolynomialRing(SR, ring_var_names) element.

    For each summand of the expanded SR expression, extracts the degree in each
    ring variable via .degree() / .coefficient(v, d), accumulating SR coefficients
    (parameters, kernels, operator symbols) separately from ring monomials.

    Handles the sum/single-term distinction explicitly so that symbolic
    coefficients like tau*Dt are never lost.
    """
    import operator as _op
    gen_sr = [SR.var(name) for name in ring_var_names]
    result = R.zero()

    expanded = SR(sr_expr).expand()
    if expanded.is_zero():
        return result

    # Collect summands.  For a sum, .operands() gives the addends correctly.
    # For a non-sum (single product, single atom), wrap in a list so we don't
    # accidentally iterate over factors instead of summands.
    # Use name-based check (not identity) because SageMath uses add_vararg,
    # not operator.add — identity check with _op.add silently fails.
    _op_obj = expanded.operator()
    if _op_obj is not None and 'add' in getattr(_op_obj, '__name__', '').lower():
        summands = expanded.operands()
    else:
        summands = [expanded]

    for term in summands:
        exponents = []
        coeff = SR(term)
        for v_sr in gen_sr:
            deg = int(coeff.degree(v_sr))
            exponents.append(deg)
            if deg > 0:
                coeff = coeff.coefficient(v_sr, deg)
        result += SR(coeff) * R.monomial(*exponents)
    return result


# ---------------------------------------------------------------------------
# Bigrade classification
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Correlated-noise cumulant injection  (Option A: read declarative
# ``correlated_noises`` block, append -W_m[mt] to the action symbolically)
# ---------------------------------------------------------------------------

def _build_cumulant_action(ns, model):
    r"""
    Construct the symbolic contribution

        S_cum  =  - sum_n  (1/n!)  \int dt_1 ... dt_n
                    \sum_{i_1...i_n}  kappa^{(n)}_{i_1...i_n}(\tau\text{'s})
                                      \tilde m_{i_1}(t_1) ... \tilde m_{i_n}(t_n)

    for each noise process declared in ``model['correlated_noises']``.

    Per ordered leg-tuple, the user-supplied kernel function is called
    with a placeholder relative-time variable ``\tau``.  The result is
    decomposed as

        K(tau)  =  c_local * dirac_delta(tau)  +  K_smooth(tau)

    via ``K.coefficient(dirac_delta(tau))``:

      * ``c_local`` (dirac residue) collapses the time integral and
        contributes  -(1/2) * c_local * mt[i] * mt[j]  directly to S_cum
        (a *local* noise-kernel term — same format as the cortical
        Poisson  -1/2 nstar_i * nt_i^2);
      * ``K_smooth`` introduces an SR coefficient symbol
        ``z_kappa_<noise>_<order>_<i+1>_<j+1>`` (analogous to ``z_g``
        for the synaptic filter) representing the implicit two-time
        integration; the kernel function is registered on
        ``ns._cumulant_kernels[(noise, order, (i, j))]`` for the
        downstream pipeline to consume.

    Only order n = 2 is implemented in this commit (covers the GTaS
    Bernoulli + Gaussian case).  Higher orders fire a warning and are
    skipped — that's fine for N=2 GTaS where κ^(n)=0 for n≥3.

    Returns an SR expression; SR(0) when the model has no
    ``correlated_noises`` declaration (i.e. plain quad_expg, linear, …).
    """
    cn = model.get('correlated_noises', None)
    ns._cumulant_kernels = {}
    if not cn:
        return SR(0)

    S_cum = SR(0)

    for noise_name, spec in cn.items():
        # ``physical_field`` is OPTIONAL — present for GTaS-style external
        # noise that introduces its own physical field (MSR bilinear
        # ``mt · m`` is registered elsewhere), absent for inline
        # cumulants on existing response fields (no new field involved).
        phys_name = spec.get('physical_field')
        if phys_name and not hasattr(ns, phys_name):
            raise ValueError(
                f"correlated_noises[{noise_name!r}]: physical_field "
                f"{phys_name!r} is not declared in physical_fields"
            )

        # Per-leg response field resolution:
        #   * ``response_legs`` (NEW; dict keyed by order, value = list of
        #     response-field names of length ``order``) — supports
        #     cross-field cumulants (legs on multiple response fields).
        #   * ``response_field`` (legacy; single string broadcast to all
        #     legs at every order) — kept for back-compat.
        response_legs_by_order = spec.get('response_legs') or {}
        legacy_resp_name = spec.get('response_field')
        if legacy_resp_name and not hasattr(ns, legacy_resp_name):
            raise ValueError(
                f"correlated_noises[{noise_name!r}]: response_field "
                f"{legacy_resp_name!r} is not declared in response_fields"
            )

        # κ^(1) (mean) is informational — already absorbed into the saddle
        # by the model's mf_bg_conditions.  Skip explicit injection.

        for order, kernel_fn in spec.get('cumulants', {}).items():
            if order < 2:
                continue

            # Resolve per-leg response field names for THIS order.  If
            # the new ``response_legs`` block has an entry, use it;
            # otherwise fall back to the legacy single-field broadcast.
            leg_field_names = response_legs_by_order.get(order)
            if leg_field_names is None:
                if not legacy_resp_name:
                    raise ValueError(
                        f"correlated_noises[{noise_name!r}] order={order}: "
                        f"no response_legs and no legacy response_field — "
                        f"can't resolve which response field each leg "
                        f"sits on."
                    )
                leg_field_names = [legacy_resp_name] * int(order)

            if len(leg_field_names) != int(order):
                raise ValueError(
                    f"correlated_noises[{noise_name!r}] order={order}: "
                    f"response_legs has {len(leg_field_names)} entries "
                    f"but order is {order}.")

            # Per-leg SR-var arrays and index ranges.  All entries are
            # wrapped to lists so a non-indexed (scalar) response field
            # still iterates uniformly.
            leg_arrays: list = []
            leg_ranges: list = []
            for fname in leg_field_names:
                if not hasattr(ns, fname):
                    raise ValueError(
                        f"correlated_noises[{noise_name!r}] order={order}: "
                        f"response-leg field {fname!r} not declared.")
                arr = getattr(ns, fname)
                if not isinstance(arr, list):
                    arr = [arr]
                leg_arrays.append(arr)
                leg_ranges.append(range(len(arr)))

            # Cumulant order n needs n-1 relative time variables.
            tau_syms = [SR.var(f'_tau_{noise_name}_{k}',
                               latex_name=rf'\tau_{{{k}}}')
                        for k in range(order - 1)]
            n_fact = factorial(order)
            from itertools import product as _iter_product
            from itertools import permutations as _iter_perms

            # Enumerate distinct permutations of the leg-field tuple.
            # For homogeneous leg-fields (e.g. ['xt', 'xt']) there's
            # only one — the existing leg-INDEX product over the field's
            # range already covers all index orderings (the 1/n! factor
            # symmetrizes them).  For heterogeneous leg-fields
            # (e.g. ['xt', 'yt']) the cumulant series sums over BOTH
            # (xt, yt) and (yt, xt) ordered tuples but the user only
            # writes one canonical row; the framework reconstructs the
            # other orderings here.
            seen_perms = set()
            distinct_field_perms = []
            for perm_indices in _iter_perms(range(order)):
                perm_fields = tuple(leg_field_names[p] for p in perm_indices)
                if perm_fields in seen_perms:
                    continue
                seen_perms.add(perm_fields)
                # Record the per-leg array, range, and original-position
                # mapping for this permutation.
                perm_arrays = [leg_arrays[p] for p in perm_indices]
                perm_ranges = [leg_ranges[p] for p in perm_indices]
                distinct_field_perms.append({
                    'fields': perm_fields,
                    'arrays': perm_arrays,
                    'ranges': perm_ranges,
                })

            for perm_spec in distinct_field_perms:
              perm_leg_arrays = perm_spec['arrays']
              perm_leg_ranges = perm_spec['ranges']
              perm_leg_fields = list(perm_spec['fields'])
              for idx_tuple in _iter_product(*perm_leg_ranges):
                # Evaluate the kernel at placeholder τ symbols.
                try:
                    K = SR(kernel_fn(ns, *idx_tuple, *tau_syms))
                except TypeError:
                    # Backward-compat: order 2 callers may use the old
                    # (ns, i, j, tau) signature with a single τ; expand.
                    if order == 2 and len(tau_syms) == 1:
                        K = SR(kernel_fn(ns, idx_tuple[0], idx_tuple[1],
                                         tau_syms[0]))
                    else:
                        raise
                K = K.expand()

                # Iteratively peel off δ(τ_k) coefficients.  After n-1
                # steps, what remains in ``c_local`` is the multiplier
                # of the FULL delta product  ∏ δ(τ_k).
                c_local = K
                for tau_k in tau_syms:
                    try:
                        c_local = c_local.coefficient(dirac_delta(tau_k))
                    except (AttributeError, TypeError):
                        c_local = SR(0)
                        break
                # Residual = K minus the all-delta contribution
                delta_product = SR(1)
                for tau_k in tau_syms:
                    delta_product = delta_product * dirac_delta(tau_k)
                K_residual = (K - c_local * delta_product)
                try:
                    K_residual = K_residual.simplify_full()
                except (ValueError, RuntimeError, AttributeError):
                    pass

                # ── Local (fully delta-correlated) contribution ──────
                # All time integrals collapse;  -(1/n!) c_local m̃_{i₁}…m̃_{iₙ}
                # at a single time gets injected directly into S_cum.
                # Each leg's SR variable comes from THAT leg's response
                # field — leg_arrays[k] is the SR-var list for the k-th
                # leg's named response field.
                #
                # Zero check: ``c_local != 0`` returns False under
                # Sage when the expression contains unbound parameters
                # without positivity assumptions (e.g. ``rho`` with
                # ``domain='real'``).  ``is_trivial_zero()`` is the
                # right test: True iff the expression is syntactically
                # zero, regardless of param assumptions.
                if not SR(c_local).is_trivial_zero():
                    factor = -SR(1) / n_fact * SR(c_local)
                    for k, k_idx in enumerate(idx_tuple):
                        factor = factor * perm_leg_arrays[k][k_idx]
                    S_cum = S_cum + factor

                # ── Smooth (non-local) residual ──────────────────────
                # Currently only handled at order 2 (the integrator uses
                # one τ_v per noise vertex).  For order ≥ 3 with a
                # smooth residual, warn — the integrator's per-leg time
                # map is ordered-by-leg; we'd need an n-leg time map.
                # For Bernoulli + Gaussian GTaS at N=2, all order-≥3
                # cumulants are FULLY LOCAL so this branch never fires
                # in practice; future non-local higher-order kernels
                # need integrator extension before they can be used.
                #
                # Symbol naming for cross-field at order 2: encode each
                # leg's field name in the suffix so the same noise can
                # produce both ``z_kappa_X_2_mt_1_mt_2`` (auto, cells 1/2
                # of mt) and ``z_kappa_X_2_xt_1_yt_1`` (cross between
                # different response fields).
                if not SR(K_residual).is_trivial_zero():
                    if order == 2:
                        leg_a_name = perm_leg_fields[0]
                        leg_b_name = perm_leg_fields[1]
                        sym_name = (
                            f'z_kappa_{noise_name}_{order}'
                            f'_{leg_a_name}_{idx_tuple[0]+1}'
                            f'_{leg_b_name}_{idx_tuple[1]+1}'
                        )
                        latex_name = (
                            rf'\kappa^{{({order})}}_'
                            rf'{{{leg_a_name}_{idx_tuple[0]+1},'
                            rf'{leg_b_name}_{idx_tuple[1]+1}}}'
                        )
                        sym = SR.var(sym_name, latex_name=latex_name)
                        # The leg-FIELD permutation outer loop has
                        # already enumerated (xt, yt) and (yt, xt) as
                        # distinct iterations, so the (1/n!) factor and
                        # the iteration's symmetry combine to give the
                        # correct overall coefficient.  At the canonical
                        # ordered single-field case (legs = ['mt', 'mt']),
                        # only ONE permutation exists; the index-product
                        # iteration over (i, j) + (j, i) provides the
                        # within-field symmetry as before.
                        S_cum = S_cum + (
                            -SR(1) / n_fact * sym
                            * perm_leg_arrays[0][idx_tuple[0]]
                            * perm_leg_arrays[1][idx_tuple[1]]
                        )
                        ns._cumulant_kernels[
                            (noise_name, order,
                             tuple(zip(perm_leg_fields, idx_tuple)))
                        ] = {
                            'symbol':    sym,
                            'kernel_fn': kernel_fn,
                            'legs':      tuple(idx_tuple),
                            'leg_fields': tuple(perm_leg_fields),
                            'tau_var':   tau_syms[0],
                        }
                    else:
                        warnings.warn(
                            f"correlated_noises[{noise_name!r}] "
                            f"cumulant order {order} legs "
                            f"{idx_tuple}: kernel has a non-local "
                            f"smooth residual that requires an n-leg "
                            f"time map in the integrator (currently "
                            f"only n=2 implemented).  Skipping.",
                            stacklevel=3,
                        )

    return S_cum


def _collect_bigrade(poly, n_tilde: int) -> dict:
    """
    Split a PolynomialRing element into {(n_tilde, n_phys): poly} sectors.
    Ring generators are ordered [tilde_gens..., phys_gens...].
    """
    R = poly.parent()
    sectors: dict = {}
    for exp_vec, coeff in poly.dict().items():
        n_t = int(sum(exp_vec[:n_tilde]))
        n_p = int(sum(exp_vec[n_tilde:]))
        key = (n_t, n_p)
        sectors[key] = sectors.get(key, R.zero()) + SR(coeff) * R.monomial(*exp_vec)
    return sectors


def _verify_and_zero_mf_sector(by_tp, mf_subs, spec_subs, ns, R, model,
                               mf_sector_keys=((0, 0), (1, 0), (0, 1)),
                               num_tol=1e-9):
    """Apply ``mf_subs`` to the saddle-eq (MF) sector and verify it
    vanishes.  After verification each MF sector entry is replaced by
    ``R.zero()`` in-place — at the saddle the sector is zero by
    construction, and downstream consumers (sanity_check, propagator
    extractor) rely on it.

    Two-tier check, per the user's spec:

      1. **Symbolic**: ``coeff.subs(mf_subs).simplify_full() == 0``.
         Cheap and complete for closed-form saddles.
      2. **Numerical fallback** (only when symbolic fails): bind every
         remaining free symbol to a numerical saddle from the model's
         own MF solver and check ``|residual| < num_tol``.  Catches the
         case where the MF equation is solved iteratively rather than
         in closed form, so ``simplify_full`` can't see the
         cancellation but the numerics confirm it.

    Failures raise :class:`AssertionError` with the offending bigrade,
    the post-subs symbolic form, and the numerical residual when
    available.  This is the structural test that catches both a
    miswired MF solver and an action whose bigrade-≤1 sector is not
    actually the saddle-eq sector.
    """
    failures: list = []
    soft_passes: list = []

    for key in mf_sector_keys:
        sector_poly = by_tp.get(key)
        if sector_poly is None or sector_poly == R.zero():
            continue
        for exp_vec, coeff in sector_poly.dict().items():
            c_subbed = SR(coeff).subs(mf_subs)
            if spec_subs:
                c_subbed = c_subbed.subs(spec_subs)
            try:
                c_simpl = c_subbed.simplify_full()
                if c_simpl == 0:
                    continue
            except Exception:
                c_simpl = c_subbed
            num_resid = _mf_numerical_residual(c_subbed, ns, model)
            if num_resid is not None and abs(num_resid) < num_tol:
                soft_passes.append((key, exp_vec, c_simpl, num_resid))
            else:
                failures.append((key, exp_vec, c_simpl, num_resid))

    if failures:
        lines = ["MF sector does not vanish at saddle:"]
        for key, exp_vec, c_simpl, num_resid in failures:
            num_str = (f"{num_resid:.3e}"
                       if num_resid is not None else "N/A")
            lines.append(
                f"  bigrade={key}  monomial_exponents={exp_vec}\n"
                f"    symbolic residual: {c_simpl}\n"
                f"    numerical residual: {num_str}")
        lines.append(
            "Either the MF solver is wrong or the action's bigrade-≤1 "
            "sector is not the saddle-eq sector.")
        raise AssertionError('\n'.join(lines))

    if soft_passes:
        warnings.warn(
            f"MF sector vanishes only numerically (not symbolically) for "
            f"{len(soft_passes)} monomial(s).  This is normal when the "
            f"MF equation is solved iteratively rather than in closed "
            f"form; flagging for visibility.",
            stacklevel=3,
        )

    for key in mf_sector_keys:
        if key in by_tp:
            by_tp[key] = R.zero()


def _mf_numerical_residual(expr, ns, model):
    """Numerical residual of an SR expression at the MF saddle.

    Returns the magnitude of ``expr`` after binding every free symbol
    to its numerical value at the saddle for a representative
    parameter point (``model['fundamental_defaults']`` if provided,
    else all-ones).  Returns ``None`` if the residual can't be built
    (no MF solver, fundamental incomplete, etc.) — the caller falls
    back to ``failures`` rather than ``soft_passes`` in that case.
    """
    fundamental = (model.get('fundamental_defaults') or
                   _default_fundamental_point(ns, model))
    if fundamental is None:
        return None

    # Prefer the DAE-based solver when the model has declared
    # ``.equation(...)``-style residuals — the legacy iterative
    # solver doesn't know how to iterate self-referential
    # implicit equations (e.g. ``xstar = -eps*xstar^3``), so it
    # returns no saddle values and the residual check would default
    # to xstar=1.0 from the all-ones fallback (wildly wrong).
    mf = None
    if model.get('equations'):
        try:
            from pipeline._mean_field_dae import solve_mean_field_dae_compat
            ft_proxy = _MFProxyForSolver(
                ns, model, taylor_order=getattr(ns, '_taylor_order', 4))
            mf = solve_mean_field_dae_compat(
                ft_proxy, model, fundamental, verbose=False)
        except Exception:
            mf = None

    if mf is None:
        try:
            from pipeline._mean_field import solve_mean_field
        except ImportError:
            return None
        try:
            ft_proxy = _MFProxyForSolver(
                ns, model, taylor_order=getattr(ns, '_taylor_order', 4))
            mf = solve_mean_field(ft_proxy, model, fundamental, verbose=False)
        except Exception:
            return None

    num_subs = dict(mf.get('num_params', {}))
    try:
        bound = SR(expr).subs(num_subs)
        free_syms = list(bound.free_variables())
    except Exception:
        return None

    if free_syms:
        for sym in free_syms:
            num_subs[sym] = 1.0
        try:
            bound = SR(expr).subs(num_subs)
        except Exception:
            return None

    try:
        return abs(complex(bound))
    except Exception:
        try:
            return abs(float(bound))
        except Exception:
            return None


def _default_fundamental_point(ns, model):
    """Build a ``fundamental`` dict for the verification's numerical
    fallback.  Prefers the theory author's per-parameter ``default``
    values (which the theory ships specifically because they admit a
    well-behaved MF solution); falls back to all-ones only when a
    parameter has no default declared.

    All-ones is a poor universal fallback for non-linear closures —
    e.g. quad ``phi(v) = a·v²`` with ``Em=a=w=1`` yields the saddle
    equation ``2v² - v + 1 = 0`` (no real root).  The theory's own
    defaults sidestep this.
    """
    fundamental: dict = {}
    pop_size_map = getattr(ns, '_pop_size', {}) or {}
    for pspec in model.get('parameters', []):
        if pspec.get('mean_field'):
            continue
        pname = pspec['name']
        default = pspec.get('default')
        if default is not None:
            fundamental[pname] = default
            continue
        ib = pspec.get('indexed_by') or []
        if not ib:
            fundamental[pname] = 1.0
        elif len(ib) == 1:
            n = pop_size_map.get(ib[0], 2)
            fundamental[pname] = [1.0] * n
        else:
            n_rows = pop_size_map.get(ib[0], 2)
            n_cols = pop_size_map.get(ib[1], 2)
            fundamental[pname] = [[1.0] * n_cols for _ in range(n_rows)]
    return fundamental or None


class _MFProxyForSolver:
    """Minimal FieldTheory-shaped proxy that ``solve_mean_field`` can
    consume during MF-sector verification.  We can't call
    ``solve_mean_field(self, ...)`` from inside ``expand()`` because
    ``self`` isn't fully populated yet — this proxy exposes just the
    ``_ns``, ``taylor_order``, and ``model`` attributes the solver needs.
    """
    __slots__ = ('_ns', 'taylor_order', 'model')

    def __init__(self, ns, model, taylor_order=4):
        self._ns = ns
        self.model = model
        self.taylor_order = taylor_order


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class FieldTheory:
    """
    Model-agnostic MSRJD field theory expander (SageMath version).

    The action callable in the model dict returns an SR expression. Field
    variables are SR symbolic variables. Nonlinear functions (exp, formal
    function symbols, etc.) are Taylor-expanded automatically.

    Parameters
    ----------
    model : dict
        Model specification dict (see models/hawkes_sage.py).
    taylor_order : int
        Truncation order for all Taylor expansions.
    """

    def __init__(self, model: dict, taylor_order: int = 4):
        self.model        = model
        self.taylor_order = taylor_order
        self._ns      = None
        self._R       = None
        self._n_tilde = 0
        self._S_raw   = None
        self._by_tp   = None
        self._mf_sector_raw = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def expand(self) -> None:
        """
        Build the namespace, evaluate the action as an SR expression,
        auto-expand all nonlinear functions of the field variables via
        sequential taylor(), rename formal derivative symbols, apply MF
        background conditions and specializations, then classify by bigrade.
        """
        ns, R, n_tilde = self._build_namespace()
        self._ns      = ns
        self._R       = R
        self._n_tilde = n_tilde

        # Evaluate action — result is an SR expression
        S_sr = SR(self.model['action'](ns))

        # ── Conv(kernel, field) reduction ─────────────────────────────
        # Resolve every ``Conv(g, X)`` atom according to the time-domain
        # convolution rules:
        #   * linearity in arg 2  ⇒  Conv(g, a+b) = Conv(g, a) + Conv(g, b)
        #   * pull constant prefactors out of arg 2
        #   * Conv(g, constant) = ĝ(0) · constant = constant   (normalised)
        #   * Conv(g, fluctuation) → g · fluctuation   (kernel symbol stays
        #     for the FT pipeline to substitute ĝ(ω) later).
        #
        # The asymmetry between Conv's two arguments matters for any
        # action term where a kernel is convolved with a field that
        # ALSO multiplies another field outside the convolution — e.g.
        # conductance-style ``v · Conv(g, n)``, which must expand under
        # v=vstar+dv, n=nstar+dn as
        #    vstar·nstar + vstar·g·dn + dv·nstar + dv·g·dn
        # The naïve ``Conv(g, n) → g·n`` flatten gives the wrong
        # ``g·nstar`` bilinear coefficient on the dv side.
        from msrjd.core.convolution import reduce_conv_in_action
        try:
            fluct_vars = set(ns._all_field_sr_vars)
            # Pass taylor_order so nonlinear Conv arguments
            # (``Conv(g, h(v, n))`` with non-polynomial h) get Taylor-
            # expanded at the same order the action itself is about to
            # be expanded at downstream.  For polynomial-of-degree-≤N
            # arguments the Taylor is a no-op, so existing theories
            # using simple ``Conv(g, n[j])`` form keep bit-identical
            # output.
            #
            # ``attachments`` collects ``kernel_symbol → set of attached
            # fluct vars`` for every rule-4 ``Conv(g, fluct) → g·fluct``
            # emission.  Stashed on ``ns`` for downstream vertex
            # extraction (``extract_vertex_types``) to identify which
            # leg each surviving kernel symbol in an interaction-vertex
            # coefficient is attached to — needed for the time-domain
            # Phase J integrator's per-leg τ scaffolding.
            attachments = {}
            S_sr = reduce_conv_in_action(
                S_sr, fluct_vars, taylor_order=self.taylor_order,
                attachments_out=attachments,
            )
            ns._kernel_attachments = attachments
        except (AttributeError, TypeError):
            # No Conv atoms / non-SR action / missing ns attribute —
            # treat as identity transformation.
            pass

        # Inject -W_m[mt] cumulant series from correlated_noises declarations
        # (Option A: user writes ns.dm[i] / ns.mt[i] in the action, the
        # framework appends the non-local cumulant terms here from the
        # declarative spec).  Returns SR(0) when the model has no
        # correlated_noises block, so existing models are untouched.
        S_sr = S_sr + _build_cumulant_action(ns, self.model)

        # Multivariate Taylor in all field variables around 0 (one-shot).
        # Truncates at TOTAL degree taylor_order, which matches the
        # diagrammatic vertex-leg-count interpretation.  Replaces the
        # previous sequential single-variable loop, which broke for
        # multi-arg formal functions (chained .taylor() doesn't compose
        # at non-zero expansion points — the inner-arg substructures
        # get frozen by the outer call).  For single-arg formal
        # functions, the result is identical to the old sequential
        # loop (since each formal call only depends on one fluctuation
        # variable, multivariate Taylor at total order N produces the
        # same monomials as per-variable order N).
        if ns._all_field_sr_vars:
            S_sr = taylor(
                S_sr,
                *[(v, 0) for v in ns._all_field_sr_vars],
                self.taylor_order,
            )

        # Rename formal function derivative symbols to clean SR names
        # e.g. D[0](phi_1)(0) → phi1_1,  phi_1(0) → phi0_1
        if ns._deriv_rename_subs:
            S_sr = S_sr.subs(ns._deriv_rename_subs)

        # Apply specializations (e.g. phi0_<i+1> → a*vstar_<i+1>,
        # phi1_<i+1> → a, quadratic phi, delta g).  These are pure
        # closure renames and are safe to apply to every sector — they
        # do not introduce parameter dependence that should be confined
        # to the saddle-eq sector.
        if 'specializations' in self.model:
            S_sr = S_sr.subs(self.model['specializations'](ns))

        # Coerce to polynomial ring and bigrade-classify FIRST, BEFORE
        # applying MF saddle substitutions.  This is critical: MF subs
        # of the form ``vstar → (Em + Σwg·nstar)/(1+τ·nstar)`` rewrites
        # ``vstar`` everywhere it appears, which would otherwise inject
        # Em-dependence (and other saddle-eq algebra) into the (1,1)
        # bilinear propagator kernel and the ≥2 interaction vertices,
        # where ``vstar``/``nstar`` should remain as free symbolic
        # parameters bound numerically downstream via num_params.
        S_poly = _sr_to_ring(S_sr.expand(), R, ns._ring_var_names)
        by_tp = _collect_bigrade(S_poly, n_tilde)

        # Apply MF background conditions ONLY to the MF sector — the
        # bigrade-≤1-in-each-index slots (0,0), (1,0), (0,1) that hold
        # the saddle-eq residual.  At the saddle this sector must
        # vanish; the substitution is the *test* of that condition,
        # not a structural rewrite of the propagator.
        mf_bg_key = ('mf_bg_conditions_action'
                     if 'mf_bg_conditions_action' in self.model
                     else 'mf_bg_conditions')
        # Preserve the pre-zero MF sector polynomials for diagnostics.
        # Downstream code reads ``_by_tp`` (which has the MF sector
        # zeroed by construction); ``_mf_sector_raw`` lets a debug
        # print show what was there before verification.
        self._mf_sector_raw = {
            key: by_tp.get(key, R.zero())
            for key in [(0, 0), (1, 0), (0, 1)]
        }

        if mf_bg_key in self.model:
            mf_subs = self.model[mf_bg_key](ns)
            # Re-apply specializations after mf_subs to resolve any
            # phi-Taylor symbols (``phi0_<i+1>``, ``phi1_<i+1>``, …)
            # reintroduced by the vstar substitution's RHS — the
            # mf_bg lambda builds RHSes BEFORE specializations runs,
            # so ``vstar → (Em + Σwg·phi0_<j+1>)/(1+τ·phi0_<i+1>)``
            # carries raw phi-Taylor tokens that need a second pass.
            spec_subs = (self.model['specializations'](ns)
                         if 'specializations' in self.model else None)
            _verify_and_zero_mf_sector(
                by_tp, mf_subs, spec_subs, ns, R, self.model,
                mf_sector_keys=[(0, 0), (1, 0), (0, 1)],
            )

        # Rebuild S_poly from by_tp (MF sector now zeroed) so
        # downstream consumers that read self._S_raw see a consistent
        # action whose (≤1, ≤1) sectors are exactly zero.
        S_raw = R.zero()
        for poly in by_tp.values():
            S_raw = S_raw + poly

        self._S_raw = S_raw
        self._by_tp = by_tp

    def sanity_check(self) -> bool:
        """
        Verify zero sectors:
          (0,0) — constant
          (1,0) — tadpole (must vanish at MF saddle)
          (0,1) — EOM residual (must vanish at background solution)
        """
        self._require_expanded()
        checks = [
            ((0, 0), 'constant term'),
            ((1, 0), 'tadpole — must vanish at MF saddle'),
            ((0, 1), 'linear physical-only — must vanish at EOM'),
        ]
        all_pass = True
        print('=== Sanity checks ===')
        for key, label in checks:
            val = self._by_tp.get(key, self._R.zero())
            ok  = (val == self._R.zero())
            print(f'  [{"PASS" if ok else "FAIL"}]  (n_tilde={key[0]}, n_phys={key[1]})  {label}')
            if not ok:
                _show(val)
            all_pass = all_pass and ok
        return all_pass

    def summary(self) -> None:
        """Print and display all non-zero bigrade sectors."""
        self._require_expanded()
        print('=== Action sectors ===')
        for (n_t, n_p), expr in sorted(self._by_tp.items()):
            if expr == self._R.zero():
                continue
            label = self._sector_label(n_t, n_p)
            print(f'  (n_tilde={n_t}, n_phys={n_p})  [{label}]:')
            _show(expr)

    def free_action(self):
        """Return the (1,1) sector polynomial."""
        self._require_expanded()
        return self._by_tp.get((1, 1), self._R.zero())

    def noise_kernel(self) -> dict:
        """Return all (≥2, 0) sectors."""
        self._require_expanded()
        return {k: v for k, v in self._by_tp.items()
                if k[0] >= 2 and k[1] == 0 and v != self._R.zero()}

    def vertices(self) -> dict:
        """Return all sectors with total degree ≥ 3."""
        self._require_expanded()
        return {k: v for k, v in self._by_tp.items()
                if k[0] + k[1] >= 3 and v != self._R.zero()}

    def sectors(self) -> dict:
        """Return full bigrade dict (non-zero sectors only)."""
        self._require_expanded()
        return {k: v for k, v in self._by_tp.items()
                if v != self._R.zero()}

    def ring(self):
        self._require_expanded()
        return self._R

    # ------------------------------------------------------------------
    # Namespace builder
    # ------------------------------------------------------------------

    def _build_namespace(self):
        """
        Construct the _Namespace with:
          - SR symbolic variables for all field variables
          - SR symbolic variables for all parameters, kernels, operators
          - Callables for nonlinear functions (auto-expand or taylor_coeffs)
          - _deriv_rename_subs: {formal_deriv_at_0 → nice_SR_var} for renaming
          - _all_field_sr_vars: list of SR vars to expand in

        Also builds PolynomialRing(SR, ring_var_names) for bigrade analysis.
        Returns (ns, R, n_tilde).
        """
        m   = self.model
        ns  = _Namespace()
        idx = m['index_sets']
        for name, lst in idx.items():
            setattr(ns, name, list(lst))
        primary_idx = list(list(idx.values())[0])

        # ── Population-aware sizing ────────────────────────────
        # For heterogeneous theories, every population in
        # ``model['populations']`` has its own index range
        # ``[0, ..., size-1]`` exposed on the namespace under both
        # ``pop_<name>`` and (for ergonomic action text) ``<name>``.
        # Legacy theories have an empty ``populations`` list and
        # fall back to the single flat ``pop`` index.
        populations = m.get('populations') or []
        pop_size = {p['name']: int(p.get('size', 1)) for p in populations}
        pop_local_idx = {name: list(range(sz))
                         for name, sz in pop_size.items()}
        # Bind plain-name iterables for use in action text:
        # ``for i in E`` and ``for j in I`` work as written.
        for pname, plist in pop_local_idx.items():
            if not hasattr(ns, pname):
                setattr(ns, pname, list(plist))
            # Also bind under the ``pop_<name>`` alias for callers
            # that prefer the explicit form.
            alias = f'pop_{pname}'
            if not hasattr(ns, alias):
                setattr(ns, alias, list(plist))

        def _field_indices(fspec):
            """Indices over which a field's SR vars range.  Uses
            ``fspec['population']`` when given (per-pop sizing); falls
            back to ``primary_idx`` for legacy fields."""
            pop = fspec.get('population')
            if pop and pop in pop_local_idx:
                return pop_local_idx[pop]
            return primary_idx

        def _entity_axis_sizes(spec):
            """Resolve the per-axis size list for a parameter / kernel.

            Returns ``[]`` for scalar, ``[N_a]`` for vector, ``[N_a, N_b]``
            for matrix.  ``indexed_by`` (heterogeneous-pop) wins over
            legacy ``indexed=`` when both are present.
            """
            ib = spec.get('indexed_by')
            if ib:
                return [pop_size.get(p, len(primary_idx)) for p in ib]
            indexed = spec.get('indexed', False)
            if indexed == 'matrix':
                return [len(primary_idx), len(primary_idx)]
            if indexed is True or indexed == 'vector':
                return [len(primary_idx)]
            return []

        # Stash for downstream code (mf_substitutions, _mean_field, etc.).
        ns._populations    = populations
        ns._pop_size       = pop_size
        ns._pop_local_idx  = pop_local_idx
        # Expose the axis-size helper so per-pop param / kernel iterators
        # in pipeline code can reuse the same resolution rules.
        ns._entity_axis_sizes = _entity_axis_sizes

        # ---- Field variables as SR symbols ----
        tilde_sr_vars: list = []
        phys_sr_vars:  list = []
        tilde_names:   list = []
        phys_names:    list = []

        for fspec in m['response_fields']:
            fname   = fspec['name']
            indexed = fspec.get('indexed', True)
            lx      = fspec.get('latex', fname)
            if indexed:
                idx_list = _field_indices(fspec)
                arr = [SR.var(f"{fname}{i+1}", latex_name=f'{lx}_{{{i+1}}}')
                       for i in idx_list]
                setattr(ns, fname, arr)
                tilde_sr_vars.extend(arr)
                tilde_names.extend(f"{fname}{i+1}" for i in idx_list)
            else:
                v = SR.var(fname, latex_name=lx)
                setattr(ns, fname, v)
                tilde_sr_vars.append(v)
                tilde_names.append(fname)

        for fspec in m['physical_fields']:
            fname   = fspec['name']
            indexed = fspec.get('indexed', True)
            lx      = fspec.get('latex', fname)
            if indexed:
                idx_list = _field_indices(fspec)
                arr = [SR.var(f"{fname}{i+1}", latex_name=f'{lx}_{{{i+1}}}')
                       for i in idx_list]
                setattr(ns, fname, arr)
                phys_sr_vars.extend(arr)
                phys_names.extend(f"{fname}{i+1}" for i in idx_list)
            else:
                v = SR.var(fname, latex_name=lx)
                setattr(ns, fname, v)
                phys_sr_vars.append(v)
                phys_names.append(fname)

        ns._tilde_sr_vars    = tilde_sr_vars
        ns._phys_sr_vars     = phys_sr_vars
        ns._all_field_sr_vars = tilde_sr_vars + phys_sr_vars

        # ---- Polynomial ring (tilde generators first) ----
        ring_var_names       = tilde_names + phys_names
        ns._ring_var_names   = ring_var_names
        n_tilde              = len(tilde_names)
        R                    = PolynomialRing(SR, ring_var_names)

        # ---- SR parameters ----
        # Vector params get a flat list of SR vars sized by their
        # ``indexed_by`` population (or the legacy primary_idx).
        # Matrix params get a 1D placeholder list; the actual 2D
        # element-naming grid is set later by ``mf_substitutions``,
        # which also knows about ``indexed_by`` for per-pop sizing.
        for pspec in m.get('parameters', []):
            pname   = pspec['name']
            domain  = pspec.get('domain', None)
            axis_sizes = _entity_axis_sizes(pspec)
            if not axis_sizes:
                sym = SR.var(pname, domain=domain) if domain else SR.var(pname)
                setattr(ns, pname, sym)
            else:
                # For vector: axis_sizes = [N_a].
                # For matrix: take the first axis; the 2D structure
                # gets installed by mf_substitutions.
                n = axis_sizes[0]
                arr = ([SR.var(f"{pname}{i+1}", domain=domain)
                        for i in range(n)]
                       if domain else
                       [SR.var(f"{pname}{i+1}") for i in range(n)])
                setattr(ns, pname, arr)

        # ---- Kernels and operators ----
        # Kernels can be:
        #   * scalar (default):  one SR symbol  ns.g  used as ``g`` in action.
        #   * vector (indexed=True/'vector' or indexed_by=['X']):
        #     N_X SR symbols ``g_<i+1>``, namespace exposes
        #     ``ns.g = [g_1, ...]`` for ``g[i]`` syntax.
        #   * matrix (indexed='matrix' or indexed_by=['X', 'Y']):
        #     N_X × N_Y SR symbols ``g_<i+1>_<j+1>``, namespace
        #     exposes ``ns.g`` as a list-of-lists for ``g[i, j]`` syntax.
        #     The companion ``kernel_ft_image`` lambda evaluates the
        #     frequency image per (i, j) pair so per-pair parameters
        #     like ``tau_g[i, j]`` resolve correctly.
        for kspec in m.get('kernels', []):
            kname    = kspec['name']
            klx_base = kspec.get('latex_name', None) or kname
            axis_sizes = _entity_axis_sizes(kspec)
            if len(axis_sizes) == 2:
                n_rows, n_cols = axis_sizes
                rows = []
                for i in range(n_rows):
                    row = []
                    for j in range(n_cols):
                        sn = (f'{kspec.get("sage_name", "z_" + kname)}'
                              f'_{i+1}_{j+1}')
                        lx = f'{klx_base}_{{{i+1}{j+1}}}'
                        row.append(SR.var(sn, latex_name=lx))
                    rows.append(row)
                setattr(ns, kname, rows)
            elif len(axis_sizes) == 1:
                arr = []
                for i in range(axis_sizes[0]):
                    sn = f'{kspec.get("sage_name", "z_" + kname)}_{i+1}'
                    lx = f'{klx_base}_{{{i+1}}}'
                    arr.append(SR.var(sn, latex_name=lx))
                setattr(ns, kname, arr)
            else:
                kn_sage = kspec.get('sage_name', kname)
                setattr(ns, kname,
                        SR.var(kn_sage, latex_name=klx_base)
                        if klx_base else SR.var(kn_sage))
        for ospec in m.get('operators', []):
            oname = ospec.get('sage_name', ospec['name'])
            olx   = ospec.get('latex_name', None)
            setattr(ns, ospec['name'],
                    SR.var(oname, latex_name=olx) if olx else SR.var(oname))

        # Internal names start with 'z' so they sort after tau ('t') and phi ('p')
        # in any SR product, giving the canonical order: φ … τ … δ/δ'.
        ns.delta_D  = SR.var('z_delta',   latex_name=r'\delta')
        ns.delta_Dp = SR.var('z_delta_p', latex_name=r"\delta'")

        # ---- Nonlinear functions ----
        # Accumulate derivative-rename substitutions for formal function symbols.
        # The rename machinery is multi-arg-aware: for an n-argument
        # function, every partial derivative (k_1, ..., k_n) with
        # sum ≤ taylor_order gets registered as a rename to
        # ``<prefix><k_1><k_2>...<k_n>_<i+1>``.  Single-arg
        # (``n_args=1``, the default) collapses to the legacy
        # ``<prefix><k>_<i+1>`` naming, so existing models are
        # unaffected.
        ns._deriv_rename_subs = {}
        order = self.taylor_order

        for fspec in m.get('functions', []):
            fname        = fspec['name']
            indexed      = fspec.get('indexed', True)
            deriv_prefix = fspec.get('deriv_prefix', fname)
            n_args       = int(fspec.get('n_args', 1))

            if 'expression' in fspec:
                # ---- Auto-expand path ----
                # ``fspec['expression'](i, x_1, ..., x_n)`` returns an SR
                # expression in the n formal arguments.  Partial
                # derivatives at the all-zero point get renamed to
                # ``<prefix><multi-idx-suffix>_<i+1>``.
                fn_latex = fspec.get('latex', deriv_prefix)
                arg_dums = [SR.var(f'_xdum_{deriv_prefix}_{j}')
                            for j in range(n_args)]

                def _deriv_latex(base, multi_idx, sub):
                    """LaTeX name for the partial-derivative symbol."""
                    total = sum(multi_idx)
                    if total == 0:
                        return (f'{base}_{{{sub}}}' if sub != '' else base)
                    if len(multi_idx) == 1:
                        # Legacy single-arg notation:  f', f'', f^{(k)}.
                        k = multi_idx[0]
                        primes = "'" if k == 1 else ("''" if k == 2 else None)
                        if primes:
                            return (f"{base}{primes}_{{{sub}}}"
                                    if sub != '' else f"{base}{primes}")
                        return (f'{base}^{{({k})}}_{{{sub}}}'
                                if sub != '' else f'{base}^{{({k})}}')
                    # Multi-arg: superscript shows the multi-index tuple.
                    idx_str = ','.join(str(k) for k in multi_idx)
                    return (f'{base}^{{({idx_str})}}_{{{sub}}}'
                            if sub != '' else f'{base}^{{({idx_str})}}')

                def _build_target(multi_idx, sub_label):
                    suffix = _multi_index_suffix(multi_idx)
                    base   = f'{deriv_prefix}{suffix}'
                    return (f'{base}_{sub_label}' if sub_label != ''
                            else base)

                def _register_renames(fe, sub_label):
                    """Register a rename for every multi-derivative of
                    ``fe`` (an SR expression in the ``arg_dums``)
                    evaluated at the all-zero expansion point."""
                    zero_subs = {arg_dums[j]: 0 for j in range(n_args)}
                    for multi_idx in _iter_multi_indices(n_args, order):
                        deriv = fe
                        for j, kj in enumerate(multi_idx):
                            if kj > 0:
                                deriv = diff(deriv, arg_dums[j], kj)
                        try:
                            deriv_at_0 = SR(deriv).subs(zero_subs)
                            if SR(deriv_at_0).is_numeric():
                                continue
                            lname = _deriv_latex(fn_latex, multi_idx, sub_label)
                            nice  = SR.var(_build_target(multi_idx, sub_label),
                                           latex_name=lname)
                            ns._deriv_rename_subs[deriv_at_0] = nice
                        except Exception:
                            pass

                if indexed:
                    fn_exprs = []
                    for i in primary_idx:
                        fe = fspec['expression'](i, *arg_dums)
                        fn_exprs.append(fe)
                        _register_renames(fe, str(i + 1))

                    def _make_fn(exprs, dums):
                        def fn(i, *xs):
                            sub = {dums[j]: xs[j] for j in range(len(dums))}
                            return exprs[i].subs(sub)
                        return fn
                    setattr(ns, fname, _make_fn(fn_exprs, arg_dums))

                else:
                    fe = fspec['expression'](*arg_dums)
                    _register_renames(fe, '')
                    def _make_scalar_fn(fe_, dums):
                        def fn(*xs):
                            sub = {dums[j]: xs[j] for j in range(len(dums))}
                            return fe_.subs(sub)
                        return fn
                    setattr(ns, fname, _make_scalar_fn(fe, arg_dums))

            elif 'taylor_coeffs' in fspec:
                # ---- Legacy path: manually supplied derivative coefficients ----
                coeffs = fspec['taylor_coeffs']
                if indexed:
                    def _make_tc(c_list, o):
                        def fn(x):
                            return _poly_taylor(c_list[:o+1], x)
                        return fn
                    fns = [_make_tc(coeffs[i], order) for i in range(len(primary_idx))]
                    setattr(ns, fname, lambda i, x, _fns=fns: _fns[i](x))
                else:
                    def _make_scalar(c_list, o):
                        def fn(x):
                            return _poly_taylor(c_list[:o+1], x)
                        return fn
                    setattr(ns, fname, _make_scalar(coeffs, order))

        # Expose taylor_order on namespace (used by specializations lambdas)
        ns._taylor_order = self.taylor_order

        # ---- MF substitutions (computed once at namespace build) ----
        for sub in m.get('mf_substitutions', []):
            setattr(ns, sub['name'], sub['value'](ns))

        return ns, R, n_tilde

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _require_expanded(self):
        if self._by_tp is None:
            raise RuntimeError("Call expand() first.")

    @staticmethod
    def _sector_label(n_t: int, n_p: int) -> str:
        total = n_t + n_p
        if (n_t, n_p) == (1, 1):  return 'free action'
        if n_t >= 2 and n_p == 0: return 'noise kernel'
        if total == 1:             return 'tadpole / background'
        if total >= 3:             return f'vertex (order {total})'
        return f'sector ({n_t},{n_p})'
