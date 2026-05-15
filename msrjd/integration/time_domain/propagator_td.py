"""
msrjd.integration.time_domain.propagator_td
===========================================
Time-domain retarded propagator utilities.

This module exposes the symbolic time-domain propagator matrix G(t) and a
per-edge lookup `G_t_entry` that returns the retarded propagator for an
edge `(u -> v)` as `G_R_{phys,resp}(t_v - t_u)`.

Conventions (fixed pipeline-wide)
---------------------------------
1. **Fourier convention**:

        G(t) = (1 / 2π) ∫ dω  exp(i ω t)  Ĝ(ω)

   Under this convention, poles with Im(ω) > 0 yield decaying
   exponentials for t > 0 and growing exponentials for t < 0. The
   causality filter in `msrjd.diagrams.causality` guarantees all
   `propagator_data['pole_vals']` have Im > 0 and thus parameterize the
   retarded sector.

2. **Retarded boundary condition**: this module's `build_G_t_matrix`
   returns the ANALYTIC pole-residue sum — it does *not* enforce t > 0.
   The caller obtains the physical retarded propagator via

        G_R(t) = Θ(t) · G_analytic(t)

   and in practice `G_t_entry` applies that multiplication for you.

3. **Heaviside at zero**: the pipeline's numerical integration path
   uses `Θ(0) = 0` (boundary `Δt = 0` is STRICTLY EXCLUDED from the
   retarded support).  This is enforced by:
   - The Heaviside-filter integrand wrapper
     (`_make_heaviside_filtered_integrand` in `final_integral.py`)
     which returns 0 whenever any `Δt <= 0`.
   - Polytope feasibility checks using strict inequality
     (`c_eff <= 0` → infeasible for degenerate pure-axis constraints).

   Note: SageMath's symbolic `heaviside(0)` returns `1/2` by default —
   that's used ONLY in the symbolic-display path (e.g. cell 26's
   optional LaTeX render when `SHOW_SYMBOLIC = True`).  It does NOT
   enter the JIT-compiled numerical integrand, where retardation is
   enforced by polytope bounds + the Heaviside filter above.

4. **Index transpose**: the retarded propagator is "response of physical
   field j to response-field source i" — `G^R_{j ← i}`. The kernel matrix
   K and its inverse G = K^{-1} both have layout [resp_row, phys_col], so
   to obtain `G^R_{j ← i}` we read `G[j, i]` (physical row, response col).
   `G_t_entry(phys_idx=j, resp_idx=i, ...)` applies this transpose — it
   matches `_get_propagator_entry` in `msrjd/integration/symbolic.py`.
"""

from sage.all import SR, I, exp, heaviside, matrix, CDF


# Stage 4a optim (2026-05-15): the smooth-matrix simplify_full() pass
# below is a Maxima round-trip per entry — measured at ~16% of
# integrate_diagram wall on the k=2 ell=1 quad config.  After Stage 4a
# opt #1, the smooth matrix is unused for analytic-eligible subsets
# (which is the common case for plain rational-propagator diagrams),
# so the simplification is dead weight in that path.  It's still
# beneficial for NoiseSourceType-kernel diagrams whose SR-path
# ``subset_factor.expand()`` walks the smooth entries.  Flip to
# ``False`` to disable; the simplify can also be turned back on
# locally if a diagnostic shows a non-trivial expand() slowdown on
# the residual SR path.
USE_SIMPLIFY_FULL_IN_GT = False


def build_G_t_matrix(propagator_data, t_var, num_params=None):
    r"""
    Build the full time-domain retarded propagator `G_R(t)` as a
    decomposition

        G_R[i, j](t)  =  delta_coeffs[i, j] · δ(t)
                       + heaviside(t) · (Σ_k C_mats[k][i, j] · exp(I · p_k · t))

    The SECOND piece is the "smooth" pole-residue sum and is returned as
    a SageMath SR matrix. The FIRST piece is the instantaneous
    δ-function response that shows up for any entry whose
    frequency-domain propagator has a nonzero limit at `ω → ∞` (e.g.
    the `ñ_i × δn_i` coupling in the MSR-JD action, where a ñ source at
    time `t` produces an *immediate* δn response at the same time `t`).
    Its coefficients are returned as a SageMath matrix of complex
    constants `delta_coeffs[i, j] = lim_{ω→∞} Ĝ[i, j](ω)`.

    The caller is responsible for handling the Heaviside (by
    multiplication or by polytope constraint) and the δ-function (by
    subset enumeration in the tree integrator, where a δ edge collapses
    one integration variable).

    Fourier convention (fixed pipeline-wide): the causality filter
    guarantees every pole in `propagator_data['pole_vals']` has
    `Im(p) > 0`, so each summand `C_k · exp(I·p_k·t)` decays for t > 0
    and grows for t < 0. The Heaviside in the retarded convention is
    what makes this well-defined on the real line.

    Parameters
    ----------
    propagator_data : dict
        Must contain keys 'pole_vals' (list of SR), 'C_mats' (list of
        SageMath matrices, one per pole), and 'G_ft' (the full
        frequency-domain propagator, required to compute the
        ω→∞ limits for the delta coefficients). This is the same dict
        consumed by `msrjd.integration.symbolic`.
    t_var : SR variable
        The symbolic time variable to build the smooth G(t) in.
    num_params : dict or None
        If provided, each pole value, residue matrix entry, and
        delta-coefficient entry is substituted with these numerical
        parameters BEFORE the matrices are assembled.

    Returns
    -------
    dict with keys:
        'smooth' : SageMath matrix (SR)
            `G_smooth[i, j](t) = Σ_k C_k[i,j] · exp(I·p_k·t)`
            The "analytic part" — caller must multiply by `heaviside(t)`
            to enforce retardation.
        'delta'  : SageMath matrix (SR)
            `delta[i, j] = lim_{ω→∞} Ĝ[i, j](ω)`, as a matrix of
            complex constants. Most entries are zero; any nonzero
            entry encodes a δ(t) component of `G_R[i, j](t)`.
        't_var'  : SR variable
            The time variable used to build `smooth` (so downstream
            code can substitute into it).

    Notes
    -----
    For backward compatibility, passing the returned dict back into
    `G_t_entry` is supported; `G_t_entry` also accepts a bare matrix
    (treated as the smooth part only).
    """
    pole_vals = propagator_data['pole_vals']
    C_mats = propagator_data['C_mats']
    G_ft = propagator_data.get('G_ft')

    # Normalise raw Python ``complex`` / Sage ``ComplexDoubleElement``
    # values into SR expressions of the form ``a + b*I`` with Python
    # ``float`` components.  This MUST run before ``.subs(num_params)``
    # because ``SR(python_complex)`` otherwise wraps the complex as an
    # opaque GiNaC node; when that node propagates into a later product
    # or expansion, GiNaC's term sort tries to compare two ``complex``
    # objects with ``<`` and raises:
    #
    #   TypeError: '<' not supported between instances of
    #              'complex' and 'complex'.
    #
    # Symbolic SR expressions (like ``i*(1 - sqrt(6)/10)`` from a
    # closed-form pole solve) pass through unchanged so their exact
    # algebraic content is preserved -- casting those through
    # ``complex(...)`` would silently collapse to a double-precision
    # approximation and produce a small systematic bias in any
    # correlator that compounds across many tau evaluations.
    try:
        from sage.rings.complex_double import ComplexDoubleElement
    except ImportError:
        ComplexDoubleElement = ()  # unlikely on real Sage installs

    def _to_sr_ab(value):
        """Normalise only raw complex / CDF scalars; leave SR alone."""
        if isinstance(value, complex):
            return SR(float(value.real)) + SR(float(value.imag)) * I
        if isinstance(value, ComplexDoubleElement):
            c = complex(value)
            return SR(float(c.real)) + SR(float(c.imag)) * I
        # Symbolic SR expression, Sage number, etc. -- preserve exactly.
        return SR(value)

    pole_vals = [_to_sr_ab(p) for p in pole_vals]
    C_mats = [
        C.apply_map(_to_sr_ab)
        for C in C_mats
    ]

    if num_params:
        pole_vals = [SR(p).subs(num_params) for p in pole_vals]
        C_mats = [
            C.apply_map(lambda e: SR(e).subs(num_params))
            for C in C_mats
        ]

    # Determine the matrix size ``nf`` so we can always build ``smooth``
    # as a proper SR matrix -- even when ``pole_vals`` is empty (which
    # would make ``sum(empty_generator)`` collapse to Python ``int(0)``,
    # breaking every downstream call to ``G_t_matrix.dimensions()``).
    n_propagator = propagator_data.get('nf')
    if n_propagator is None:
        if C_mats:
            n_propagator = C_mats[0].nrows()
        else:
            _Dd = propagator_data.get('D_delta')
            if _Dd is not None:
                n_propagator = _Dd.nrows()
            else:
                raise ValueError(
                    "build_G_t_matrix: propagator_data is missing 'nf' "
                    "and provides neither 'C_mats' nor 'D_delta', so the "
                    "matrix size cannot be inferred."
                )

    if not pole_vals:
        import warnings as _warnings
        _warnings.warn(
            "build_G_t_matrix: propagator_data['pole_vals'] is empty. "
            "This usually means the upstream characteristic-polynomial "
            "solve returned no retarded roots (e.g., the kernel matrix "
            "has no poles in Im(omega) > 0 after num_params were "
            "substituted).  The smooth propagator will be identically "
            "zero, so the only nonvanishing contribution will come from "
            "propagator_data['D_delta'].  Check compute_poles_and_residues "
            "in notebook cell 8.",
            stacklevel=2,
        )

    smooth = matrix(SR, n_propagator, n_propagator, 0)
    for k in range(len(pole_vals)):
        smooth = smooth + C_mats[k] * exp(I * pole_vals[k] * t_var)
    if USE_SIMPLIFY_FULL_IN_GT:
        try:
            smooth = smooth.apply_map(lambda e: e.simplify_full())
        except Exception:
            # simplify_full may fail on expressions with numerical
            # complex coefficients; fall back to unsimplified form.
            pass

    # Delta coefficients: the polynomial (non-proper) part of Ĝ(ω).
    #
    # The key identity:
    #   Ĝ(ω) = Q(iω) + Ĝ_proper(ω)
    # where Q is polynomial and Ĝ_proper is strictly proper. Then:
    #   G(t) = Q(∂_t) δ(t) + Θ(t) · [residue sum]
    #
    # For the common case Q = constant:
    #   D[i,j] = lim_{ω→∞} Ĝ[i,j](ω) = coefficient of δ(t)
    #
    # If propagator_data already has 'D_delta' (computed upstream via
    # symbolic polynomial division / limit), use it directly. Otherwise
    # compute it here via Sage's symbolic limit.
    from sage.all import limit as _limit, oo as _oo

    D_precomputed = propagator_data.get('D_delta')
    if D_precomputed is not None:
        # Use the precomputed delta matrix. Apply num_params if needed.
        if num_params:
            delta_coeffs = D_precomputed.apply_map(
                lambda e: SR(e).subs(num_params) if not SR(e).is_zero() else SR(0)
            )
        else:
            delta_coeffs = D_precomputed
    else:
        # Compute from G_ft via symbolic limit.
        nrows, ncols = (G_ft.dimensions() if G_ft is not None
                        else smooth.dimensions())
        delta_data = [[SR(0)] * ncols for _ in range(nrows)]
        if G_ft is not None:
            omega_sym = _infer_omega_variable(G_ft, num_params)
            if omega_sym is not None:
                for i in range(nrows):
                    for j in range(ncols):
                        entry = SR(G_ft[i, j])
                        if entry.is_zero():
                            continue
                        try:
                            entry_sub = (entry.subs(num_params)
                                         if num_params else entry)
                            lim_val = _limit(entry_sub,
                                             **{str(omega_sym): _oo})
                            if not SR(lim_val).is_zero():
                                if num_params:
                                    lim_val = SR(lim_val).subs(num_params)
                                delta_data[i][j] = SR(lim_val)
                        except Exception:
                            pass
        delta_coeffs = matrix(SR, delta_data)

    return {
        'smooth': smooth,
        'delta': delta_coeffs,
        't_var': t_var,
    }


def _infer_omega_variable(G_ft, num_params):
    """
    Find the unique free variable in G_ft (after num_params substitution)
    that is assumed to be the frequency symbol ω.
    """
    free = set()
    nrows, ncols = G_ft.dimensions()
    for i in range(nrows):
        for j in range(ncols):
            entry = SR(G_ft[i, j])
            if num_params:
                entry = entry.subs(num_params)
            try:
                free.update(entry.variables())
            except Exception:
                pass
    if not free:
        return None
    # Prefer a variable literally named 'omega' if one exists.
    for v in free:
        if str(v) == 'omega':
            return v
    # Otherwise just take the first one.
    return sorted(free, key=str)[0]


def G_t_entry(G_t_obj, phys_idx, resp_idx, t_expr,
              include_heaviside=True):
    r"""
    Look up the SMOOTH retarded-propagator entry 'response of physical
    field `phys_idx` to response-field source `resp_idx`' at time
    `t_expr`. This returns **only the smooth (pole-residue) part** —
    any δ(t) component of the full retarded propagator is handled
    separately by the tree evaluator (see `final_integral.py`).

    Returns `smooth[phys_idx, resp_idx]` with its internal time
    variable substituted to `t_expr`, optionally multiplied by
    `heaviside(t_expr)` to enforce retarded time ordering.

    This reads the (phys, resp) entry — i.e., the TRANSPOSE of the
    (resp, phys) convention natural to the kernel matrix K. This
    transpose matches `_get_propagator_entry` in
    `msrjd/integration/symbolic.py` line ~305; both paths (Phase I and
    Phase J) must use the same transpose convention.

    Parameters
    ----------
    G_t_obj : dict or SageMath matrix (SR)
        Output of `build_G_t_matrix`. If a dict (the current format),
        the `'smooth'` entry is used. If a bare SR matrix is passed,
        it is treated as the smooth part directly (backward compat for
        tests and external callers).
    phys_idx : int
        Row index (physical field at the head of the edge).
    resp_idx : int
        Column index (response field at the tail of the edge).
    t_expr : SR expression or number
        The time argument. For an edge u -> v this is `t_v - t_u`.
    include_heaviside : bool
        If True (default), multiply by `heaviside(t_expr)`. Pass False
        only if the caller is managing causality separately.

    Returns
    -------
    SR expression
    """
    if isinstance(G_t_obj, dict):
        G_t_matrix = G_t_obj['smooth']
    else:
        G_t_matrix = G_t_obj

    t_var = _infer_time_variable(G_t_matrix)
    entry = SR(G_t_matrix[phys_idx, resp_idx])
    if t_var is not None:
        entry = entry.subs({t_var: t_expr})
    if include_heaviside:
        entry = entry * heaviside(t_expr)
    return entry


def G_t_delta_coeff(G_t_obj, phys_idx, resp_idx):
    """
    Return the δ(t) coefficient of the retarded propagator entry
    `G_R[phys_idx, resp_idx](t)` — i.e., `lim_{ω→∞} Ĝ[phys_idx,
    resp_idx](ω)`. This is the instantaneous-response weight that the
    tree evaluator uses when enumerating δ-edge subsets.

    Returns a Python complex number (or a real number if the imaginary
    part is negligible). Returns 0 if there is no δ component or the
    input is a bare smooth matrix without delta info.
    """
    if not isinstance(G_t_obj, dict):
        return 0.0 + 0.0j
    delta_matrix = G_t_obj.get('delta')
    if delta_matrix is None:
        return 0.0 + 0.0j
    val = delta_matrix[phys_idx, resp_idx]
    try:
        c = complex(CDF(val))
    except Exception:
        return 0.0 + 0.0j
    if abs(c.imag) < 1e-12 * max(abs(c.real), 1.0):
        return float(c.real)
    return c


def _infer_time_variable(G_t_matrix):
    """
    Recover the time variable used to build `G_t_matrix`.

    G(t) is a symbolic matrix built from a sum of `C_k * exp(I p_k t)`
    terms. When all parameters are numerically substituted, the only
    remaining free variable in each entry is `t`. We scan entries for a
    nonzero expression and return its first free variable.

    Returns None if no free variable is present (e.g., if `t_expr` was
    already substituted to a number before the matrix was constructed).

    The ``entry == 0`` test is avoided: for entries carrying Python
    ``complex`` coefficients (from a numerically-substituted CDF
    C-matrix + complex pole), Sage's equality falls back to Maxima,
    which cannot compare ``complex`` objects and raises.  Here we
    check ``entry.variables()`` directly -- any expression with free
    variables is non-zero for our purposes -- and only consult an
    explicit ``is_zero()`` test for scalar entries.
    """
    # Defensive: caller may hand us a degenerate scalar (Python int 0,
    # from ``sum(empty_generator)`` in ``build_G_t_matrix`` when
    # ``pole_vals`` is empty).  In that case there is no free time
    # variable to infer; return None and let the caller substitute.
    if not hasattr(G_t_matrix, 'dimensions'):
        return None

    nrows, ncols = G_t_matrix.dimensions()
    for i in range(nrows):
        for j in range(ncols):
            entry = G_t_matrix[i, j]
            try:
                free_vars = entry.variables()
            except AttributeError:
                continue
            if free_vars:
                return free_vars[0]
            # No free variables -> scalar.  Check zero without triggering
            # Maxima simplification that fails on embedded complex coefs.
            try:
                if entry.is_trivial_zero():
                    continue
            except AttributeError:
                pass
    return None
