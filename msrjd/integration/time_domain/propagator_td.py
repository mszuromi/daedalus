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

3. **Heaviside at zero**: SageMath's `heaviside(0)` returns `1/2`
   (the default). This module treats that convention as FIXED across the
   entire pipeline. No module may monkey-patch, override, or substitute
   `unit_step` for `heaviside`; coincident-time evaluations must all use
   the same convention.

4. **Index transpose**: the retarded propagator is "response of physical
   field j to response-field source i" — `G^R_{j ← i}`. The kernel matrix
   K and its inverse G = K^{-1} both have layout [resp_row, phys_col], so
   to obtain `G^R_{j ← i}` we read `G[j, i]` (physical row, response col).
   `G_t_entry(phys_idx=j, resp_idx=i, ...)` applies this transpose — it
   matches `_get_propagator_entry` in `msrjd/integration/symbolic.py`.
"""

from sage.all import SR, I, exp, heaviside


def build_G_t_matrix(propagator_data, t_var, num_params=None):
    r"""
    Build the symbolic time-domain propagator matrix G(t) via the
    pole-residue sum.

    The pole-residue form gives the ANALYTIC exponential part of the
    time-domain propagator:

        G_analytic[i, j](t) = Σ_k  C_mats[k][i, j] · exp(I · pole_vals[k] · t)

    Causality (retarded boundary condition) is **not** applied here — the
    caller must multiply by `heaviside(t)` to obtain the physical
    retarded propagator G_R(t) = Θ(t) · G_analytic(t). Under the Fourier
    convention documented at the top of this module, the causality
    filter guarantees Im(pole_vals[k]) > 0, so `G_analytic(t)` decays
    for t > 0 and grows for t < 0; the Heaviside is what makes the
    product well-defined on the whole real line and enforces retarded
    time ordering.

    Parameters
    ----------
    propagator_data : dict
        Must contain keys 'pole_vals' (list of SR) and 'C_mats' (list of
        SageMath matrices, one per pole). This is the same dict consumed
        by `msrjd.integration.symbolic`.
    t_var : SR variable
        The symbolic time variable to build G(t) in.
    num_params : dict or None
        If provided, each pole value and residue matrix entry is
        substituted with these numerical parameters BEFORE the symbolic
        matrix is assembled. This is usually the right call for numerical
        work — symbolic propagator entries with fully symbolic parameters
        produce a blow-up of terms downstream.

    Returns
    -------
    G_t : SageMath matrix (SR)
        Symbolic time-domain propagator matrix (analytic part; no
        Heaviside applied).
    """
    pole_vals = propagator_data['pole_vals']
    C_mats = propagator_data['C_mats']

    if num_params:
        pole_vals = [SR(p).subs(num_params) for p in pole_vals]
        C_mats = [
            C.apply_map(lambda e: SR(e).subs(num_params))
            for C in C_mats
        ]

    G_t = sum(
        C_mats[k] * exp(I * pole_vals[k] * t_var)
        for k in range(len(pole_vals))
    )
    try:
        G_t = G_t.apply_map(lambda e: e.simplify_full())
    except Exception:
        # simplify_full may fail on expressions with numerical complex
        # coefficients; fall back to unsimplified form.
        pass
    return G_t


def G_t_entry(G_t_matrix, phys_idx, resp_idx, t_expr,
              include_heaviside=True):
    r"""
    Look up the retarded propagator 'response of physical field
    phys_idx to response-field source resp_idx' at time `t_expr`.

    Returns `G_t_matrix[phys_idx, resp_idx]` with `t_var` substituted to
    `t_expr`, optionally multiplied by `heaviside(t_expr)` to enforce
    retarded time ordering (the default).

    This reads the (phys, resp) entry of the matrix — i.e., the
    TRANSPOSE of the (resp, phys) convention natural to the kernel
    matrix K. This transpose matches `_get_propagator_entry` in
    `msrjd/integration/symbolic.py` line ~305; both paths (Phase I and
    Phase J) must use the same transpose convention.

    Parameters
    ----------
    G_t_matrix : SageMath matrix (SR)
        Output of `build_G_t_matrix`. The caller passes the matrix
        explicitly (rather than the raw `propagator_data` dict) so that
        numerically-substituted matrices can be reused across many edge
        lookups without recomputing.
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
    # G_t_matrix was built in terms of a single time variable (often
    # called 't'); recover it so we can substitute t -> t_expr.
    # We locate the unique variable in G_t_matrix that is not among the
    # numerical-parameter symbols by inspecting any nonzero entry.
    t_var = _infer_time_variable(G_t_matrix)
    entry = SR(G_t_matrix[phys_idx, resp_idx])
    if t_var is not None:
        entry = entry.subs({t_var: t_expr})
    if include_heaviside:
        entry = entry * heaviside(t_expr)
    return entry


def _infer_time_variable(G_t_matrix):
    """
    Recover the time variable used to build `G_t_matrix`.

    G(t) is a symbolic matrix built from a sum of `C_k * exp(I p_k t)`
    terms. When all parameters are numerically substituted, the only
    remaining free variable in each entry is `t`. We scan entries for a
    nonzero expression and return its first free variable.

    Returns None if no free variable is present (e.g., if `t_expr` was
    already substituted to a number before the matrix was constructed).
    """
    nrows, ncols = G_t_matrix.dimensions()
    for i in range(nrows):
        for j in range(ncols):
            entry = G_t_matrix[i, j]
            if entry == 0:
                continue
            try:
                free_vars = entry.variables()
            except AttributeError:
                continue
            if free_vars:
                return free_vars[0]
    return None
