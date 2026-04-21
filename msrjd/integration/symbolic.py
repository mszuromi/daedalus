"""
msrjd.integration.symbolic
============================
Construct and evaluate diagram integral expressions symbolically
for **stationary** systems using frequency-domain methods.

Mathematical procedure (Helias & Dahmen Ch. 9)
-----------------------------------------------
For a unique typed diagram Γ contributing to the k-point cumulant
⟨x_{a₁}(t₁) ⋯ x_{aₖ}(tₖ)⟩:

1. **Assign frequency variables**: each directed edge e → ω_e.

2. **Frequency conservation at internal vertices**: at each internal
   vertex (interaction or source), the sum of incoming edge frequencies
   equals the sum of outgoing edge frequencies.  Source vertices
   (no incoming edges) impose Σ ω_out = 0.

3. **Solve conservation**: the conservation equations express internal
   edge frequencies in terms of the external edge frequencies (the
   frequencies on edges connected to leaves).  If the diagram has
   loops, some internal edge frequencies remain free — these are the
   loop integration variables.

4. **Build the integrand**: for each edge, look up the propagator
   entry Ĝ_{i,j}(ω_e).  Each external leg j contributes an
   exponential e^{±iω_j t_j} from the inverse Fourier transform.
   The sign depends on the leaf directionality:
     - tail leaf (outgoing): e^{-iω t}
     - head leaf (incoming): e^{+iω t}

5. **Integrate by residues**: integrate over all free frequency
   variables.  The result is C(t₁, ..., tₖ), a function of the
   external times.  For stationary systems, this depends only on
   time differences.

FT convention:  F̂(ω) = ∫ f(t) e^{iωt} dt,  IFT: f(t) = ∫ F̂(ω) e^{-iωt} dω/(2π).

Build Phase H.
"""

from collections import defaultdict

from sage.all import SR, I, pi, exp, matrix, QQ

from msrjd.diagrams.symmetry import classify_coefficient_factors


# ═══════════════════════════════════════════════════════════════════════════
# Prerequisite checks
# ═══════════════════════════════════════════════════════════════════════════

def check_propagator_available(propagator_data):
    """
    Verify that frequency-domain propagator data is available.

    Returns
    -------
    'explicit' or 'implicit'
    """
    if propagator_data.get('G_ft') is not None:
        return 'explicit'
    if (propagator_data.get('adj_ft') is not None
            and propagator_data.get('D_omega') is not None):
        return 'implicit'
    raise ValueError(
        'No frequency-domain propagator available. '
        'Need either G_ft (explicit inverse) or adj_ft + D_omega '
        '(adjugate and determinant of the kernel matrix).'
    )


# ═══════════════════════════════════════════════════════════════════════════
# Frequency assignment and conservation
# ═══════════════════════════════════════════════════════════════════════════

def assign_frequencies(typed_diagram, k):
    r"""
    Create a symbolic frequency variable for every edge.

    Each edge gets a unique key ``(idx, u, v)`` where ``idx`` is the
    position in ``D.edges()``.

    Returns
    -------
    edge_freqs : dict
        {(idx, u, v): SR variable} for every edge.
    leaf_edge_freq : dict
        {leaf_vertex: SR variable} — the frequency on the edge at each leaf.
    """
    D = typed_diagram.prediagram[0]
    leaves = typed_diagram.prediagram[2]

    edge_freqs = {}
    for idx, (u, v, lbl) in enumerate(D.edges()):
        ek = (idx, u, v)
        edge_freqs[ek] = SR.var(f'omega_e{idx}',
                                latex_name=rf'\omega_{{e_{{{idx}}}}}')

    leaf_edge_freq = {}
    for lf in leaves:
        for ek, w in edge_freqs.items():
            _, u, v = ek
            if u == lf or v == lf:
                leaf_edge_freq[lf] = w
                break

    return edge_freqs, leaf_edge_freq


def solve_conservation(typed_diagram, edge_freqs, leaf_edge_freq):
    r"""
    Solve frequency conservation at internal vertices using linear algebra.

    At each internal vertex v the conservation equation is
    ``Σ ω_in − Σ ω_out = 0``.  These are linear in the edge
    frequencies, so they can be written as a matrix equation

        M_int · ω_int + M_ext · ω_ext = 0

    where ``ω_ext`` are the (known) external-leg frequencies and
    ``ω_int`` are the internal-edge frequencies to solve for.
    Row-reducing the augmented matrix ``[M_int | M_ext]`` over ℚ
    yields:

    * **pivot columns < n_int** → determined internal frequencies
      (expressed in terms of free + external frequencies),
    * **non-pivot columns < n_int** → free (loop) frequencies,
    * **pivot in external columns** → overall conservation among
      external frequencies.

    Returns
    -------
    substitutions : dict
        ``{ω_internal: expression_in_ext_and_free}``.
    free_freqs : list of SR variable
        Loop integration variables (one per independent loop).
    overall_conservation : SR expression or None
        Constraint among external frequencies, if one exists.
    """
    D = typed_diagram.prediagram[0]
    leaves = typed_diagram.prediagram[2]
    leaf_set = set(leaves)

    leaf_freq_set = set(leaf_edge_freq.values())

    # Partition edge keys into external (touch a leaf) and internal.
    all_ekeys = sorted(edge_freqs.keys())          # sorted by edge index
    ext_ekeys = [ek for ek in all_ekeys if edge_freqs[ek] in leaf_freq_set]
    int_ekeys = [ek for ek in all_ekeys if edge_freqs[ek] not in leaf_freq_set]

    internal_vertices = sorted(v for v in D.vertices() if v not in leaf_set)

    n_eqs = len(internal_vertices)
    n_int = len(int_ekeys)
    n_ext = len(ext_ekeys)

    if n_eqs == 0:
        return {}, [], None

    # ── Build incidence sub-matrices over ℚ ──────────────────────────────
    # Convention: edge entering v → +1, edge leaving v → −1.
    M_int = matrix(QQ, n_eqs, max(n_int, 1))  # at least 1 col for Sage
    M_ext = matrix(QQ, n_eqs, max(n_ext, 1))

    for i, v in enumerate(internal_vertices):
        for j, ek in enumerate(int_ekeys):
            _, u, w = ek
            if w == v:
                M_int[i, j] += 1
            if u == v:
                M_int[i, j] -= 1
        for j, ek in enumerate(ext_ekeys):
            _, u, w = ek
            if w == v:
                M_ext[i, j] += 1
            if u == v:
                M_ext[i, j] -= 1

    # Trim dummy columns if n_int or n_ext was 0
    if n_int == 0:
        M_int = matrix(QQ, n_eqs, 0)
    if n_ext == 0:
        M_ext = matrix(QQ, n_eqs, 0)

    # ── Row-reduce augmented matrix [M_int | M_ext] ─────────────────────
    aug = M_int.augment(M_ext)
    rref = aug.echelon_form()
    pivots = rref.pivots()

    pivot_set = set(pivots)
    free_int_cols = [j for j in range(n_int) if j not in pivot_set]
    free_freqs = [edge_freqs[int_ekeys[j]] for j in free_int_cols]

    # ── Read off substitutions and overall conservation ──────────────────
    substitutions = {}
    overall_conservation = None

    for row_idx, pivot_col in enumerate(pivots):
        if row_idx >= rref.nrows():
            break

        if pivot_col < n_int:
            # Pivot is an internal frequency → solve for it.
            expr = SR(0)
            for j in range(rref.ncols()):
                if j == pivot_col:
                    continue
                coeff = rref[row_idx, j]
                if coeff == 0:
                    continue
                if j < n_int:
                    expr -= SR(coeff) * edge_freqs[int_ekeys[j]]
                else:
                    expr -= SR(coeff) * edge_freqs[ext_ekeys[j - n_int]]
            substitutions[edge_freqs[int_ekeys[pivot_col]]] = expr

        else:
            # Pivot in an external column → overall conservation.
            if overall_conservation is None:
                cons = SR(0)
                for j in range(n_int, rref.ncols()):
                    coeff = rref[row_idx, j]
                    if coeff != 0:
                        cons += SR(coeff) * edge_freqs[ext_ekeys[j - n_int]]
                if not cons.is_zero():
                    overall_conservation = cons

    # If no pivot fell in external columns, check for a redundant-row
    # conservation (all internal entries zero, external entries nonzero).
    if overall_conservation is None and n_int == 0 and n_ext >= 2:
        for row_idx in range(rref.nrows()):
            cons = SR(0)
            for j in range(rref.ncols()):
                coeff = rref[row_idx, j]
                if coeff != 0:
                    cons += SR(coeff) * edge_freqs[ext_ekeys[j]]
            if not cons.is_zero():
                overall_conservation = cons
                break

    return substitutions, free_freqs, overall_conservation


# ═══════════════════════════════════════════════════════════════════════════
# Edge matching helper (shared by build_integrand and extract_prop_factors)
# ═══════════════════════════════════════════════════════════════════════════

def _resolve_edge_propagator_data(typed_diagram, edge_freqs, substitutions):
    """
    Match each edge in D.edges() to its typed_diagram propagator info
    and resolve the frequency via conservation substitutions.

    Returns
    -------
    list of (td_edge_key, matrix_row, matrix_col, resolved_freq)
        One entry per edge in D.edges() order.
    """
    D = typed_diagram.prediagram[0]
    td_edge_keys = list(typed_diagram.edge_types.keys())

    td_key_for_edge = {}
    for td_ek in td_edge_keys:
        u, v = td_ek[0], td_ek[1]
        lbl = td_ek[2] if len(td_ek) > 2 else None
        td_key_for_edge.setdefault((u, v, lbl), []).append(td_ek)

    result = []
    for idx, (u, v, lbl) in enumerate(D.edges()):
        ef_key = (idx, u, v)
        if ef_key not in edge_freqs:
            continue

        candidates = td_key_for_edge.get((u, v, lbl), [])
        if not candidates:
            candidates = td_key_for_edge.get((u, v, None), [])
        if not candidates:
            for k_try in td_edge_keys:
                if k_try[0] == u and k_try[1] == v:
                    candidates = [k_try]
                    break
        if not candidates:
            raise KeyError(f'No type info for edge ({u}, {v}, {lbl})')

        td_ek = candidates.pop(0)
        ri, pi = typed_diagram.propagator_indices[td_ek]
        omega_val = edge_freqs[ef_key].subs(substitutions)
        result.append((td_ek, ri, pi, omega_val))

    return result


# ═══════════════════════════════════════════════════════════════════════════
# Propagator lookup
# ═══════════════════════════════════════════════════════════════════════════

def _get_propagator_entry(i, j, omega_var, propagator_data, omega_symbol):
    r"""Look up the retarded propagator Ĝ^R_{j,i}(ω) and substitute omega_var.

    The kernel matrix K has rows = response fields, cols = physical fields,
    so G = K^{-1} has the same layout: G[resp, phys].  But the *retarded*
    propagator — "response of physical field j to response-field source i" —
    is G^R_{j←i} = G[j, i] (physical row, response col) = G^T[i, j].

    To obtain the correct Feynman-rule factor we therefore read G[j, i],
    transposing the (resp, phys) indices supplied by the typed diagram.
    """
    G_ft = propagator_data.get('G_ft')
    if G_ft is not None:
        entry = SR(G_ft[j, i])          # transposed: G[phys, resp]
    else:
        adj = propagator_data['adj_ft']
        det = propagator_data['D_omega']
        entry = SR(adj[j, i]) / SR(det)  # transposed
    return entry.subs({omega_symbol: omega_var})


# ═══════════════════════════════════════════════════════════════════════════
# Build integrand
# ═══════════════════════════════════════════════════════════════════════════

def build_integrand(typed_diagram, edge_freqs, substitutions,
                    propagator_data, omega_symbol=None,
                    noise_structure=None):
    r"""
    Build the frequency-domain integrand: ∏_e Ĝ_{i_e,j_e}(ω_e).

    All edge frequencies are substituted via the conservation solution,
    so the result depends only on external frequencies and any free
    (loop) frequencies.
    """
    if omega_symbol is None:
        omega_symbol = SR.var('omega')

    D = typed_diagram.prediagram[0]
    ns = noise_structure or {'temporal_type': 'white'}
    noise_type = ns.get('temporal_type', 'white')

    resolved = _resolve_edge_propagator_data(
        typed_diagram, edge_freqs, substitutions
    )

    integrand = SR(1)
    for td_ek, ri, pi, omega_val in resolved:
        prop_entry = _get_propagator_entry(
            ri, pi, omega_val, propagator_data, omega_symbol
        )
        integrand *= prop_entry

    # Colored noise kernel factors
    if noise_type == 'colored':
        kernel_ft_expr = ns.get('kernel_ft')
        kernel_ft_omega = ns.get('kernel_ft_omega', omega_symbol)
        if kernel_ft_expr is None:
            raise ValueError(
                "Colored noise requires 'kernel_ft' in noise_structure."
            )
        for v, vtype in typed_diagram.vertex_assignments.items():
            if D.in_degree(v) > 0:
                continue
            out_ekeys = [ek for ek in edge_freqs if ek[1] == v]
            if out_ekeys:
                omega_val = edge_freqs[out_ekeys[0]].subs(substitutions)
                integrand *= SR(kernel_ft_expr).subs(
                    {kernel_ft_omega: omega_val}
                )

    return integrand


# ═══════════════════════════════════════════════════════════════════════════
# Full assembly
# ═══════════════════════════════════════════════════════════════════════════

def build_integrand_stationary(typed_diagram, propagator_data, k,
                                omega_symbol=None,
                                time_dep_params=None,
                                noise_structure=None):
    r"""
    Full integrand assembly for a stationary diagram.

    Produces everything needed to evaluate the time-domain cumulant:

        C(t₁,...,tₖ) = scalar_prefactor × ∏ ∫(dω_free/(2π))
                     × [∏ e^{±iω_leaf t_j}] × ∏ Ĝ(ω_e)

    Returns
    -------
    dict with keys:
        'scalar_prefactor', 'integrand', 'full_integrand',
        'ext_freqs', 'ext_times', 'free_freqs',
        'loop_number', 'edge_freqs', 'substitutions',
        'coefficient_info', 'fourier_prefactor',
        'overall_conservation', 'leaf_edge_freq'
    """
    check_propagator_available(propagator_data)

    if omega_symbol is None:
        omega_symbol = SR.var('omega')

    coeff_info = classify_coefficient_factors(
        typed_diagram,
        time_dep_params=time_dep_params,
        noise_structure=noise_structure,
    )
    if not coeff_info['is_stationary']:
        raise ValueError(
            'Diagram has nonstationary features. '
            'Use time-domain integrand builder instead.'
        )

    D = typed_diagram.prediagram[0]
    leaves = typed_diagram.prediagram[2]

    edge_freqs, leaf_edge_freq = assign_frequencies(typed_diagram, k)

    subs, free_freqs, overall_cons = solve_conservation(
        typed_diagram, edge_freqs, leaf_edge_freq
    )

    ext_freqs_all = [leaf_edge_freq[lf] for lf in leaves]
    ext_times = []
    for j in range(k):
        ext_times.append(SR.var(f't_{j+1}', latex_name=rf't_{{{j+1}}}'))

    overall_cons_sub = {}
    if overall_cons is not None:
        from sage.all import solve as sage_solve
        # Pick the last external freq as the elimination target (for k>=2
        # this expresses ω_k in terms of ω_1..ω_{k-1}; for k=1 it sets
        # the single external freq to zero).
        target = ext_freqs_all[-1]
        cons_sol = sage_solve(overall_cons == 0, target, solution_dict=True)
        if cons_sol:
            overall_cons_sub = cons_sol[0]
            # Substitute the eliminated external freq into existing subs.
            for var_key in list(subs):
                subs[var_key] = subs[var_key].subs(overall_cons_sub)
            subs.update(overall_cons_sub)

    ext_freqs = [w for w in ext_freqs_all if w not in overall_cons_sub]

    integrand = build_integrand(
        typed_diagram, edge_freqs, subs,
        propagator_data, omega_symbol, noise_structure,
    )

    # Build exponential factor for each external leg
    exp_factor = SR(1)
    for j, lf in enumerate(leaves):
        omega_j = leaf_edge_freq[lf].subs(subs)
        is_tail = (D.out_degree(lf) > 0 and D.in_degree(lf) == 0)
        if is_tail:
            exp_factor *= exp(-I * omega_j * ext_times[j])
        else:
            exp_factor *= exp(+I * omega_j * ext_times[j])

    full_integrand = integrand * exp_factor

    loop_number = len(free_freqs)
    n_integrals = len(free_freqs) + len(ext_freqs)
    fourier_prefactor = SR(1) / (2 * pi) ** n_integrals

    return {
        'scalar_prefactor':     coeff_info['scalar_prefactor'],
        'integrand':            integrand,
        'full_integrand':       full_integrand,
        'ext_freqs':            ext_freqs,
        'ext_times':            ext_times,
        'free_freqs':           free_freqs,
        'loop_number':          loop_number,
        'edge_freqs':           edge_freqs,
        'substitutions':        subs,
        'leaf_edge_freq':       leaf_edge_freq,
        'coefficient_info':     coeff_info,
        'fourier_prefactor':    fourier_prefactor,
        'overall_conservation': overall_cons,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Residue integration helpers
# ═══════════════════════════════════════════════════════════════════════════

def _pole_order(expr, var, pole):
    """Determine the order of a pole by testing (var - pole)^n × expr."""
    from sage.all import limit
    for n in range(1, 10):
        try:
            test = ((var - pole)**n * expr).simplify_rational()
            val = limit(test, **{str(var): pole})
            val_str = str(val)
            if ('Infinity' not in val_str and 'infinity' not in val_str
                    and 'ind' not in val_str.lower()
                    and 'und' not in val_str.lower()):
                return n
        except Exception:
            continue
    return 1


def _residue_at_pole(expr, var, pole):
    """
    Compute the residue of expr at var = pole (arbitrary order).
    Res = (1/(n-1)!) × lim d^{n-1}/dz^{n-1} [(z-p)^n f(z)]
    """
    from sage.all import limit, diff, factorial

    n = _pole_order(expr, var, pole)
    try:
        g = ((var - pole)**n * expr).simplify_rational()
        if n == 1:
            return limit(g, **{str(var): pole})
        dg = g
        for _ in range(n - 1):
            dg = diff(dg, var)
        return limit(dg, **{str(var): pole}) / factorial(n - 1)
    except Exception:
        try:
            return limit((var - pole) * expr, **{str(var): pole})
        except Exception:
            return SR(0)


def _find_poles(expr, var):
    """Find poles of a rational expression in var."""
    from sage.all import solve as sage_solve
    try:
        expr_s = expr.simplify_rational()
    except Exception:
        expr_s = expr
    try:
        _, denom = expr_s.numerator_denominator()
        sols = sage_solve(denom == 0, var, solution_dict=True)
        return [s[var] for s in sols]
    except Exception:
        return []


def _integrate_by_residues(expr, var, close_upper=True):
    r"""
    Evaluate ∫_{-∞}^{∞} expr dvar via the residue theorem.

    Returns 2πi × Σ Res (upper) or −2πi × Σ Res (lower).
    """
    poles = _find_poles(expr, var)
    try:
        expr_s = expr.simplify_rational()
    except Exception:
        expr_s = expr

    residue_sum = SR(0)
    for p in poles:
        try:
            p_imag = p.imag_part()
            in_upper = bool(p_imag > 0)
            in_lower = bool(p_imag < 0)
        except (TypeError, ValueError):
            continue

        if close_upper and in_upper:
            residue_sum += _residue_at_pole(expr_s, var, p)
        elif not close_upper and in_lower:
            residue_sum += _residue_at_pole(expr_s, var, p)

    if close_upper:
        return 2 * pi * I * residue_sum
    else:
        return -2 * pi * I * residue_sum


# ═══════════════════════════════════════════════════════════════════════════
# Time-domain integration
# ═══════════════════════════════════════════════════════════════════════════

def _extract_time_argument(full_integrand, integrand, omega_var):
    r"""
    Extract the effective time argument τ from the exponential factor.
    """
    from sage.all import log
    try:
        exp_factor = (full_integrand / integrand).simplify_rational()
        log_exp = log(exp_factor)
        d_log = log_exp.diff(omega_var).simplify()
        tau = (-I * d_log).simplify()
        return tau
    except Exception:
        return None


def integrate_to_time_domain(integrand_result):
    r"""
    Evaluate the full time-domain cumulant contribution from a diagram.

    Performs all frequency integrals (free loop + external) by residues.
    """
    from sage.all import heaviside

    free_freqs = integrand_result['free_freqs']
    ext_freqs = integrand_result['ext_freqs']
    ext_times = integrand_result['ext_times']
    prefactor = integrand_result['scalar_prefactor']
    fourier_pf = integrand_result['fourier_prefactor']
    full_integrand = integrand_result['full_integrand']
    integrand = integrand_result['integrand']

    int_vars = list(free_freqs) + list(ext_freqs)

    current_expr = full_integrand
    status = 'ok'

    for lf in free_freqs:
        try:
            current_expr = current_expr.simplify_rational()
        except Exception:
            pass
        result = _integrate_by_residues(current_expr, lf, close_upper=True)
        if result == 0:
            result = _integrate_by_residues(current_expr, lf, close_upper=False)
        current_expr = result

    for omega_ext in ext_freqs:
        try:
            current_expr = current_expr.simplify_rational()
        except Exception:
            pass

        try:
            result_upper = _integrate_by_residues(
                current_expr, omega_ext, close_upper=True)
            result_lower = _integrate_by_residues(
                current_expr, omega_ext, close_upper=False)
        except Exception:
            status = 'partial'
            break

        tau_arg = _extract_time_argument(full_integrand, integrand, omega_ext)

        if tau_arg is not None:
            current_expr = (
                heaviside(tau_arg) * result_upper
                + heaviside(-tau_arg) * result_lower
            )
        else:
            current_expr = result_upper
            status = 'partial'

    result = prefactor * fourier_pf * current_expr
    try:
        result = result.simplify_full()
    except Exception:
        try:
            result = result.simplify_rational()
        except Exception:
            pass

    return {
        'time_domain_result': result,
        'frequency_domain_integrand': integrand,
        'integration_variables': int_vars,
        'ext_times': ext_times,
        'status': status,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Convenience wrappers
# ═══════════════════════════════════════════════════════════════════════════

def integrate_tree_level(integrand_result):
    r"""Evaluate a tree-level diagram (ℓ = 0) in the time domain."""
    if integrand_result['loop_number'] != 0:
        raise ValueError(
            f"Expected tree-level (ℓ=0), got ℓ={integrand_result['loop_number']}"
        )
    return integrate_to_time_domain(integrand_result)['time_domain_result']


def integrate_one_loop_residues(integrand_result, pole_vals=None,
                                 omega_symbol=None, close_upper=True):
    r"""Evaluate a one-loop diagram (ℓ = 1) in the time domain."""
    if integrand_result['loop_number'] != 1:
        raise ValueError(
            f"Expected one-loop (ℓ=1), got ℓ={integrand_result['loop_number']}"
        )
    return integrate_to_time_domain(integrand_result)['time_domain_result']


# ═══════════════════════════════════════════════════════════════════════════
# Propagator factor extraction (opaque representation)
# ═══════════════════════════════════════════════════════════════════════════

def extract_propagator_factors(typed_diagram, edge_freqs, substitutions,
                               noise_structure=None):
    r"""
    Extract the structural propagator factor list for the integrand,
    WITHOUT expanding propagator entries into their rational form.

    Each edge produces an opaque factor ``('prop', row, col, ω_resolved)``
    representing ``Ĝ_{row,col}(ω_resolved)``.  Colored-noise source
    vertices produce ``('noise', ω_resolved)``.

    Parameters
    ----------
    typed_diagram : TypedDiagram
    edge_freqs : dict
        From ``assign_frequencies``.
    substitutions : dict
        From ``solve_conservation`` (with overall conservation applied).
    noise_structure : dict or None

    Returns
    -------
    list of tuples
        Each element is ``('prop', int, int, SR)`` or ``('noise', SR)``.
    """
    D = typed_diagram.prediagram[0]
    ns = noise_structure or {'temporal_type': 'white'}
    noise_type = ns.get('temporal_type', 'white')

    resolved = _resolve_edge_propagator_data(
        typed_diagram, edge_freqs, substitutions
    )

    factors = []
    for td_ek, ri, pi, omega_val in resolved:
        factors.append(('prop', ri, pi, omega_val))

    # Colored noise kernel factors
    if noise_type == 'colored':
        for v, vtype in typed_diagram.vertex_assignments.items():
            if D.in_degree(v) > 0:
                continue
            out_ekeys = [ek for ek in edge_freqs if ek[1] == v]
            if out_ekeys:
                omega_val = edge_freqs[out_ekeys[0]].subs(substitutions)
                factors.append(('noise', omega_val))

    return factors


# ═══════════════════════════════════════════════════════════════════════════
# Canonicalization and loop kernel signatures
# ═══════════════════════════════════════════════════════════════════════════

def canonicalize_prop_factors(prop_factors, ext_freqs, free_freqs):
    r"""
    Replace diagram-specific frequency variable names with canonical ones.

    External frequencies → ``w_0, w_1, ...`` (in the order given).
    Loop frequencies → ``L_0, L_1, ...`` (in the order given).

    Parameters
    ----------
    prop_factors : list of tuples
        From ``extract_propagator_factors``.
    ext_freqs : list of SR variable
    free_freqs : list of SR variable

    Returns
    -------
    canonical_factors : list of tuples
        Same structure as input but with canonical frequency variables.
    canonical_ext : list of SR variable
        ``[w_0, w_1, ...]``
    canonical_loop : list of SR variable
        ``[L_0, L_1, ...]``
    """
    canonical_ext = [SR.var(f'w_{i}', latex_name=rf'w_{{{i}}}')
                     for i in range(len(ext_freqs))]
    canonical_loop = [SR.var(f'L_{i}', latex_name=rf'L_{{{i}}}')
                      for i in range(len(free_freqs))]

    var_map = {}
    for old, new in zip(ext_freqs, canonical_ext):
        var_map[old] = new
    for old, new in zip(free_freqs, canonical_loop):
        var_map[old] = new

    canonical_factors = []
    for factor in prop_factors:
        if factor[0] == 'prop':
            _, ri, pi, omega_val = factor
            canonical_factors.append(
                ('prop', ri, pi, omega_val.subs(var_map))
            )
        elif factor[0] == 'noise':
            _, omega_val = factor
            canonical_factors.append(
                ('noise', omega_val.subs(var_map))
            )
        else:
            canonical_factors.append(factor)

    return canonical_factors, canonical_ext, canonical_loop


def _factor_depends_on(factor, variables):
    """Check if a propagator factor depends on any of the given variables."""
    var_set = set(variables)
    if factor[0] == 'prop':
        return bool(set(factor[3].variables()) & var_set)
    elif factor[0] == 'noise':
        return bool(set(factor[1].variables()) & var_set)
    return False


def _factor_to_hashable(factor):
    """Convert a propagator factor to a hashable form for signatures."""
    if factor[0] == 'prop':
        return ('prop', factor[1], factor[2], str(factor[3].expand()))
    elif factor[0] == 'noise':
        return ('noise', str(factor[1].expand()))
    return factor


def loop_kernel_signature(prop_factors_canonical, free_freqs_canonical):
    r"""
    Build a hierarchical, hashable signature for the loop integrand.

    The signature is constructed level by level, from the innermost
    loop variable to the outermost:

    - At each level, factors are partitioned into those that depend on
      the current loop variable (``loop_factors``) and those that do
      not (``outer_factors``).
    - The ``loop_factors`` form the signature at this nesting level.
    - The ``outer_factors`` are passed to the next (outer) level.

    For a 1-loop diagram, the result is::

        (external_sig, loop_0_sig)

    For a 2-loop diagram (future)::

        (external_sig, loop_0_sig, loop_1_sig)

    where each ``*_sig`` is a sorted tuple of hashable factor
    representations.  Two diagrams with the same signature have
    identical loop integrals (as functions of the external frequency).

    Parameters
    ----------
    prop_factors_canonical : list of tuples
        Canonicalized propagator factors.
    free_freqs_canonical : list of SR variable
        ``[L_0, L_1, ...]`` — ordered innermost-first.

    Returns
    -------
    tuple
        Hashable hierarchical signature.
    """
    remaining = list(prop_factors_canonical)
    level_sigs = []

    # Process from innermost loop variable to outermost.
    for loop_var in reversed(free_freqs_canonical):
        loop_factors = [f for f in remaining
                        if _factor_depends_on(f, [loop_var])]
        remaining = [f for f in remaining
                     if not _factor_depends_on(f, [loop_var])]
        level_sig = tuple(sorted(_factor_to_hashable(f) for f in loop_factors))
        level_sigs.append(level_sig)

    # Whatever's left depends only on external frequencies.
    ext_sig = tuple(sorted(_factor_to_hashable(f) for f in remaining))

    return (ext_sig,) + tuple(level_sigs)


def loop_only_signature(full_signature):
    r"""
    Extract only the loop-dependent part of a kernel signature.

    ``full_signature`` has the form ``(ext_sig, loop_0_sig, ...)``.
    This returns ``(loop_0_sig, ...)`` — the part that determines the
    numerical loop integral.  Two diagrams with the same loop-only
    signature require only **one** numerical integration, even if their
    external propagator factors differ.

    Tree-level diagrams (no loop levels) return ``()``.
    """
    # Element 0 is ext_sig; everything after is loop levels.
    return full_signature[1:]


# ═══════════════════════════════════════════════════════════════════════════
# Diagram grouping by loop kernel
# ═══════════════════════════════════════════════════════════════════════════

def group_diagrams_by_kernel(unique_diagrams, propagator_data, k,
                              omega_symbol=None, time_dep_params=None,
                              noise_structure=None):
    r"""
    Build integrands for all diagrams and group by loop kernel signature.

    Diagrams sharing the same loop kernel (same product of propagator
    entries with the same frequency routing) differ only in their scalar
    prefactors.  Summing the prefactors within each group eliminates
    redundant numerical integrations.

    Parameters
    ----------
    unique_diagrams : list of TypedDiagram
    propagator_data : dict
    k : int
        Number of external legs.
    omega_symbol, time_dep_params, noise_structure :
        Passed through to ``build_integrand_stationary``.

    Returns
    -------
    list of dict
        One entry per unique kernel, with keys:

        ``'signature'`` : tuple
            The hierarchical loop kernel signature.
        ``'combined_prefactor'`` : SR expression
            Sum of scalar prefactors over all diagrams in the group.
        ``'representative_ir'`` : dict
            Integrand result from ``build_integrand_stationary`` for
            one representative diagram (use for numerical evaluation).
        ``'diagrams'`` : list of TypedDiagram
            All diagrams in the group.
        ``'individual_prefactors'`` : list of SR expression
        ``'n_diagrams'`` : int
        ``'loop_number'`` : int
        ``'prop_factors'`` : list of tuples
            Canonical propagator factor list for this kernel.
    """
    groups = defaultdict(lambda: {
        'prefactors': [],
        'diagrams': [],
        'ir': None,
        'prop_factors': None,
        'loop_number': None,
    })

    for td in unique_diagrams:
        ir = build_integrand_stationary(
            td, propagator_data, k,
            omega_symbol=omega_symbol,
            time_dep_params=time_dep_params,
            noise_structure=noise_structure,
        )
        pf = extract_propagator_factors(
            td, ir['edge_freqs'], ir['substitutions'],
            noise_structure=noise_structure,
        )
        cpf, c_ext, c_loop = canonicalize_prop_factors(
            pf, ir['ext_freqs'], ir['free_freqs']
        )
        sig = loop_kernel_signature(cpf, c_loop)

        g = groups[sig]
        g['prefactors'].append(ir['scalar_prefactor'])
        g['diagrams'].append(td)
        if g['ir'] is None:
            g['ir'] = ir
            g['prop_factors'] = cpf
            g['loop_number'] = ir['loop_number']

    result = []
    for sig, g in groups.items():
        combined = sum(g['prefactors'][1:], g['prefactors'][0])
        result.append({
            'signature': sig,
            'combined_prefactor': combined,
            'representative_ir': g['ir'],
            'diagrams': g['diagrams'],
            'individual_prefactors': g['prefactors'],
            'n_diagrams': len(g['diagrams']),
            'loop_number': g['loop_number'],
            'prop_factors': g['prop_factors'],
        })

    return result


# ═══════════════════════════════════════════════════════════════════════════
# Compute full correction (with deduplication)
# ═══════════════════════════════════════════════════════════════════════════

def compute_correction(typed_diagrams, propagator_data, k,
                        omega_symbol=None, pole_vals=None,
                        time_dep_params=None, noise_structure=None):
    r"""
    Sum time-domain contributions from all diagrams.

    Returns
    -------
    results : list of dict
        Per-diagram results.
    total : SR expression
        Sum of all contributions C(t₁,...,tₖ).
    """
    if omega_symbol is None:
        omega_symbol = SR.var('omega')

    results = []
    total = SR(0)

    for td in typed_diagrams:
        ir = build_integrand_stationary(
            td, propagator_data, k,
            omega_symbol=omega_symbol,
            time_dep_params=time_dep_params,
            noise_structure=noise_structure,
        )
        try:
            td_result = integrate_to_time_domain(ir)
            contribution = td_result['time_domain_result']
            status = td_result['status']
        except Exception as exc:
            contribution = None
            status = f'error: {exc}'

        results.append({
            'diagram': td,
            'integrand_result': ir,
            'contribution': contribution,
            'status': status,
        })
        if contribution is not None:
            total += contribution

    return results, total
