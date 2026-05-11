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
    diff, function, exp, dirac_delta, integrate, oo, I, pi, taylor
)
from IPython.display import display, Math as _Math


def fourier_transform(f, t, s):
    r"""
    Symbolic Fourier transform (angular-frequency convention):

        F(s) = \int_{-\infty}^{\infty} f(t) e^{-i s t} dt

    No 2π in the exponent.  Gives  δ(t) → 1,  δ'(t) → iω.
    Uses SageMath's symbolic integrate, which delegates to Maxima/SymPy and
    handles distributions (dirac_delta, diff(dirac_delta, t), ...) via the
    sifting property and integration by parts.
    """
    return integrate(f * exp(-I * s * t), t, -oo, oo)


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
        phys_name = spec['physical_field']
        resp_name = spec['response_field']
        if not hasattr(ns, resp_name):
            raise ValueError(
                f"correlated_noises[{noise_name!r}]: response_field "
                f"{resp_name!r} is not declared in response_fields"
            )
        if not hasattr(ns, phys_name):
            raise ValueError(
                f"correlated_noises[{noise_name!r}]: physical_field "
                f"{phys_name!r} is not declared in physical_fields"
            )
        resp_field = getattr(ns, resp_name)
        if not isinstance(resp_field, list):
            resp_field = [resp_field]
        legs = list(range(len(resp_field)))

        # κ^(1) (mean) is informational — already absorbed into the saddle
        # by the model's mf_bg_conditions.  Skip explicit injection.

        for order, kernel_fn in spec.get('cumulants', {}).items():
            if order < 2:
                continue

            # Cumulant order n needs n-1 relative time variables.
            tau_syms = [SR.var(f'_tau_{noise_name}_{k}',
                               latex_name=rf'\tau_{{{k}}}')
                        for k in range(order - 1)]
            n_fact = factorial(order)
            from itertools import product as _iter_product

            for idx_tuple in _iter_product(legs, repeat=order):
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
                if c_local != 0:
                    factor = -SR(1) / n_fact * SR(c_local)
                    for k_idx in idx_tuple:
                        factor = factor * resp_field[k_idx]
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
                if K_residual != 0:
                    if order == 2:
                        sym_name = (
                            f'z_kappa_{noise_name}_{order}'
                            f'_{idx_tuple[0]+1}_{idx_tuple[1]+1}'
                        )
                        latex_name = (
                            rf'\kappa^{{({order})}}_'
                            rf'{{{idx_tuple[0]+1}{idx_tuple[1]+1}}}'
                        )
                        sym = SR.var(sym_name, latex_name=latex_name)
                        S_cum = S_cum + (
                            -SR(1) / n_fact * sym
                            * resp_field[idx_tuple[0]]
                            * resp_field[idx_tuple[1]]
                        )
                        ns._cumulant_kernels[
                            (noise_name, order, tuple(idx_tuple))
                        ] = {
                            'symbol':    sym,
                            'kernel_fn': kernel_fn,
                            'legs':      tuple(idx_tuple),
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

        # Apply MF background conditions (SR substitutions)
        # Prefer the action-specific mf_bg dict (closure-baked, so
        # the symbolic (1, 0) tadpole vanishes for any saddle EOM
        # form).  Falls back to the legacy ``mf_bg_conditions`` key
        # for old hand-written model files.
        mf_bg_key = ('mf_bg_conditions_action'
                     if 'mf_bg_conditions_action' in self.model
                     else 'mf_bg_conditions')
        if mf_bg_key in self.model:
            S_sr = S_sr.subs(self.model[mf_bg_key](ns))

        # Apply optional specializations (e.g. quadratic phi, delta g)
        if 'specializations' in self.model:
            S_sr = S_sr.subs(self.model['specializations'](ns))

        # Expand products and coerce to polynomial ring for bigrade analysis
        S_poly = _sr_to_ring(S_sr.expand(), R, ns._ring_var_names)

        self._S_raw = S_poly
        self._by_tp = _collect_bigrade(S_poly, n_tilde)

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
                arr = [SR.var(f"{fname}{i+1}", latex_name=f'{lx}_{{{i+1}}}')
                       for i in primary_idx]
                setattr(ns, fname, arr)
                tilde_sr_vars.extend(arr)
                tilde_names.extend(f"{fname}{i+1}" for i in primary_idx)
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
                arr = [SR.var(f"{fname}{i+1}", latex_name=f'{lx}_{{{i+1}}}')
                       for i in primary_idx]
                setattr(ns, fname, arr)
                phys_sr_vars.extend(arr)
                phys_names.extend(f"{fname}{i+1}" for i in primary_idx)
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
        for pspec in m.get('parameters', []):
            pname   = pspec['name']
            domain  = pspec.get('domain', None)
            indexed = pspec.get('indexed', False)
            if indexed:
                arr = ([SR.var(f"{pname}{i+1}", domain=domain) for i in primary_idx]
                       if domain else
                       [SR.var(f"{pname}{i+1}") for i in primary_idx])
                setattr(ns, pname, arr)
            else:
                sym = SR.var(pname, domain=domain) if domain else SR.var(pname)
                setattr(ns, pname, sym)

        # ---- Kernels and operators ----
        # Kernels can be:
        #   * scalar (default):  one SR symbol  ns.g  used as ``g`` in action.
        #   * vector (indexed=True/'vector'):  N SR symbols ``g_<i+1>``,
        #     namespace exposes ``ns.g = [g_1, g_2, ...]`` for ``g[i]`` syntax.
        #   * matrix (indexed='matrix'):  N×N SR symbols ``g_<i+1>_<j+1>``,
        #     namespace exposes ``ns.g`` as a list-of-lists wrapper for
        #     ``g[i, j]`` syntax.  The companion ``kernel_ft_image`` lambda
        #     populates each symbol's frequency image (potentially with
        #     per-pair parameters like ``tau_g[i, j]``).
        for kspec in m.get('kernels', []):
            kname    = kspec['name']
            klx_base = kspec.get('latex_name', None) or kname
            indexed  = kspec.get('indexed', False)
            matrix_k = kspec.get('matrix', False) or (indexed == 'matrix')
            vector_k = (indexed is True or indexed == 'vector') and not matrix_k
            if matrix_k:
                rows = []
                for i in primary_idx:
                    row = []
                    for j in primary_idx:
                        sn   = f'{kspec.get("sage_name", "z_" + kname)}_{i+1}_{j+1}'
                        lx   = f'{klx_base}_{{{i+1}{j+1}}}'
                        row.append(SR.var(sn, latex_name=lx))
                    rows.append(row)
                setattr(ns, kname, rows)
            elif vector_k:
                arr = []
                for i in primary_idx:
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
