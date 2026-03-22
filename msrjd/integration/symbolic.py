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

from sage.all import SR, I, pi, exp

from msrjd.diagrams.symmetry import classify_coefficient_factors


# ═══════════════════════════════════════════════════════════════════════════
# Prerequisite checks
# ═══════════════════════════════════════════════════════════════════════════

def check_propagator_available(propagator_data):
    """
    Verify that frequency-domain propagator data is available.

    Parameters
    ----------
    propagator_data : dict
        Must contain 'G_ft' or both 'adj_ft' and 'D_omega'.

    Returns
    -------
    'explicit' or 'implicit'

    Raises
    ------
    ValueError if neither form is available.
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

def _edge_key(u, v, lbl, has_multiedges):
    """Canonical dict key for an edge."""
    return (u, v, lbl) if has_multiedges else (u, v)


def assign_frequencies(typed_diagram, k):
    r"""
    Create a symbolic frequency variable for every edge.

    Each leaf has exactly one edge connecting it to the diagram.
    The frequency on that edge is the external frequency for that leaf.

    Parameters
    ----------
    typed_diagram : TypedDiagram
    k : int
        Number of external legs (= number of leaves).

    Returns
    -------
    edge_freqs : dict
        {edge_key: SR variable} for every edge.
    leaf_edge_freq : dict
        {leaf_vertex: SR variable} — the frequency on the edge at each leaf.
    """
    D = typed_diagram.prediagram[0]
    leaves = typed_diagram.prediagram[2]
    leaf_set = set(leaves)
    has_me = (D.has_multiple_edges()
              if hasattr(D, 'has_multiple_edges') else False)

    edge_freqs = {}
    for idx, (u, v, lbl) in enumerate(D.edges()):
        ek = _edge_key(u, v, lbl, has_me)
        edge_freqs[ek] = SR.var(f'omega_e{idx}',
                                latex_name=rf'\omega_{{e_{{{idx}}}}}')

    # Identify the frequency variable on the edge at each leaf
    leaf_edge_freq = {}
    for lf in leaves:
        for ek, w in edge_freqs.items():
            u, v = ek[0], ek[1]
            if u == lf or v == lf:
                leaf_edge_freq[lf] = w
                break

    return edge_freqs, leaf_edge_freq


def solve_conservation(typed_diagram, edge_freqs, leaf_edge_freq):
    r"""
    Solve frequency conservation at internal vertices.

    Conservation at each internal vertex:
        Σ(incoming ω_e) = Σ(outgoing ω_e)

    The unknowns are the internal edge frequencies (edges not connected
    to any leaf).  The leaf-edge frequencies are the external frequencies
    — they are parameters, not unknowns.

    If the diagram has loops, some internal edge frequencies remain
    free after solving.  These are the loop integration variables.

    Parameters
    ----------
    typed_diagram : TypedDiagram
    edge_freqs : dict
        {edge_key: ω_e} from assign_frequencies.
    leaf_edge_freq : dict
        {leaf_vertex: ω_e} from assign_frequencies.

    Returns
    -------
    substitutions : dict
        {ω_e: expr(external freqs, free freqs)} for all solved edges.
    free_freqs : list of SR variable
        Internal edge frequencies not determined by conservation
        (one per independent loop).
    overall_conservation : SR expression or None
        Redundant equation among external frequencies (= 0).
    """
    D = typed_diagram.prediagram[0]
    leaves = typed_diagram.prediagram[2]
    leaf_set = set(leaves)
    has_me = (D.has_multiple_edges()
              if hasattr(D, 'has_multiple_edges') else False)

    # Identify which edge frequency variables are external vs internal
    leaf_freq_set = set(leaf_edge_freq.values())
    internal_freq_vars = [w for w in edge_freqs.values()
                          if w not in leaf_freq_set]

    # Build conservation equations at internal vertices only
    in_ekeys = {v: [] for v in D.vertices()}
    out_ekeys = {v: [] for v in D.vertices()}
    for (u, v, lbl) in D.edges():
        ek = _edge_key(u, v, lbl, has_me)
        out_ekeys[u].append(ek)
        in_ekeys[v].append(ek)

    equations = []
    for v in D.vertices():
        if v in leaf_set:
            continue  # skip leaves — their edge freqs are external
        in_sum = sum(edge_freqs[ek] for ek in in_ekeys[v])
        out_sum = sum(edge_freqs[ek] for ek in out_ekeys[v])
        equations.append(in_sum - out_sum)

    # Solve for internal edge frequencies using sequential elimination.
    # SageMath's solve() returns [] for underdetermined systems,
    # so we solve one equation at a time, substituting as we go.
    from sage.all import solve as sage_solve

    substitutions = {}
    free_freqs = []
    overall_conservation = None

    if not internal_freq_vars:
        # No internal edges (e.g. source directly to leaves)
        if equations:
            overall_conservation = equations[0]
        return substitutions, free_freqs, overall_conservation

    remaining_unknowns = list(internal_freq_vars)
    remaining_eqs = list(equations)

    # Sequential elimination: for each equation, solve for one unknown
    changed = True
    while changed and remaining_eqs and remaining_unknowns:
        changed = False
        for eq_idx, eq in enumerate(remaining_eqs):
            # Apply current substitutions
            eq_sub = eq.subs(substitutions)
            if eq_sub == 0:
                # Redundant equation
                remaining_eqs.pop(eq_idx)
                changed = True
                break
            # Try to solve for any remaining unknown
            for unk in remaining_unknowns:
                if unk in set(eq_sub.variables()):
                    sol = sage_solve(eq_sub == 0, unk, solution_dict=True)
                    if sol:
                        substitutions[unk] = sol[0][unk]
                        remaining_unknowns.remove(unk)
                        remaining_eqs.pop(eq_idx)
                        changed = True
                        break
            if changed:
                break

    # Any remaining unknowns are free (loop frequencies)
    free_freqs = list(remaining_unknowns)

    # Check for overall conservation: substitute solution back into
    # a leaf-vertex conservation equation to find the constraint
    # among external frequencies.
    for lf in leaves:
        in_sum = sum(edge_freqs[ek] for ek in in_ekeys[lf])
        out_sum = sum(edge_freqs[ek] for ek in out_ekeys[lf])
        if in_sum != 0 or out_sum != 0:
            cons = (in_sum - out_sum).subs(substitutions)
            # If this equation involves only external freqs,
            # it's an overall conservation relation
            cons_vars = set(cons.variables())
            if cons_vars and cons_vars.issubset(leaf_freq_set):
                overall_conservation = cons
                break

    return substitutions, free_freqs, overall_conservation


# ═══════════════════════════════════════════════════════════════════════════
# Propagator lookup
# ═══════════════════════════════════════════════════════════════════════════

def _get_propagator_entry(i, j, omega_var, propagator_data, omega_symbol):
    """Look up Ĝ_{i,j}(ω) and substitute omega_var for ω."""
    G_ft = propagator_data.get('G_ft')
    if G_ft is not None:
        entry = SR(G_ft[i, j])
    else:
        adj = propagator_data['adj_ft']
        det = propagator_data['D_omega']
        entry = SR(adj[i, j]) / SR(det)
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

    Parameters
    ----------
    typed_diagram : TypedDiagram
    edge_freqs : dict
    substitutions : dict
    propagator_data : dict
    omega_symbol : SR variable or None
    noise_structure : dict or None

    Returns
    -------
    SR expression
    """
    if omega_symbol is None:
        omega_symbol = SR.var('omega')

    D = typed_diagram.prediagram[0]
    ns = noise_structure or {'temporal_type': 'white'}
    noise_type = ns.get('temporal_type', 'white')

    integrand = SR(1)

    for edge_key in typed_diagram.edge_types:
        ri, pi = typed_diagram.propagator_indices[edge_key]

        # Find matching edge frequency
        if edge_key in edge_freqs:
            omega_e = edge_freqs[edge_key]
        else:
            u, v = edge_key[0], edge_key[1]
            matched = [ek for ek in edge_freqs
                       if ek[0] == u and ek[1] == v]
            if matched:
                omega_e = edge_freqs[matched[0]]
            else:
                raise KeyError(f'No frequency variable for edge {edge_key}')

        omega_val = omega_e.subs(substitutions)
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
            out_ekeys = [ek for ek in edge_freqs if ek[0] == v]
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

    The exponential factors come from the IFT at each external leg.
    The sign depends on the leaf directionality:
      - tail (outgoing into diagram): e^{-iωt}
      - head (incoming from diagram): e^{+iωt}

    Parameters
    ----------
    typed_diagram : TypedDiagram
    propagator_data : dict
    k : int
        Number of external legs.
    omega_symbol : SR variable or None
    time_dep_params : list of str or None
    noise_structure : dict or None

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

    # Coefficient classification
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

    # Frequency assignment
    edge_freqs, leaf_edge_freq = assign_frequencies(typed_diagram, k)

    # Solve conservation at internal vertices
    subs, free_freqs, overall_cons = solve_conservation(
        typed_diagram, edge_freqs, leaf_edge_freq
    )

    # External frequencies and times
    ext_freqs_all = [leaf_edge_freq[lf] for lf in leaves]
    ext_times = []
    for j in range(k):
        ext_times.append(SR.var(f't_{j+1}', latex_name=rf't_{{{j+1}}}'))

    # Apply overall conservation to reduce external frequencies.
    # Overall conservation is an equation among the leaf-edge frequencies
    # (e.g. ω_e0 - ω_e1 = 0 for the 2-pt tree).
    # Solve for the last ext freq in terms of the others.
    overall_cons_sub = {}
    if overall_cons is not None and len(ext_freqs_all) >= 2:
        from sage.all import solve as sage_solve
        target = ext_freqs_all[-1]
        cons_sol = sage_solve(overall_cons == 0, target, solution_dict=True)
        if cons_sol:
            overall_cons_sub = cons_sol[0]
            subs.update(overall_cons_sub)

    ext_freqs = [w for w in ext_freqs_all if w not in overall_cons_sub]

    # Build propagator integrand (with all substitutions applied)
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

    # Integration variables = free internal freqs + independent ext freqs
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

    The full_integrand = integrand × exp(i ω τ + ...).
    We extract τ as the coefficient of (i × omega_var) in the exponent.

    Returns τ (SR expression) or None.
    """
    from sage.all import log
    try:
        exp_factor = (full_integrand / integrand).simplify_rational()
        log_exp = log(exp_factor)
        d_log = log_exp.diff(omega_var).simplify()
        # d_log = i*τ → τ = -i * d_log
        tau = (-I * d_log).simplify()
        return tau
    except Exception:
        return None


def integrate_to_time_domain(integrand_result):
    r"""
    Evaluate the full time-domain cumulant contribution from a diagram.

    Performs all frequency integrals (free loop + external) by residues,
    returning C(t₁, ..., tₖ) — a function of the external times.

    For the 2-point function, the result is:
        C(t₁, t₂) = scalar_prefactor × (1/2π)^n
            × ∫ dω  e^{iω(t₂−t₁)} × Ĝ(ω)²
        = scalar_prefactor × Θ(t₂−t₁) × [time-domain expression]

    Parameters
    ----------
    integrand_result : dict
        Output from build_integrand_stationary.

    Returns
    -------
    dict with keys:
        'time_domain_result' : SR expression
        'frequency_domain_integrand' : SR expression
        'integration_variables' : list of SR variable
        'ext_times' : list of SR variable
        'status' : str
    """
    from sage.all import heaviside

    free_freqs = integrand_result['free_freqs']
    ext_freqs = integrand_result['ext_freqs']
    ext_times = integrand_result['ext_times']
    prefactor = integrand_result['scalar_prefactor']
    fourier_pf = integrand_result['fourier_prefactor']
    full_integrand = integrand_result['full_integrand']
    integrand = integrand_result['integrand']

    # All integration variables: free (loop) frequencies, then external
    int_vars = list(free_freqs) + list(ext_freqs)

    current_expr = full_integrand
    status = 'ok'

    # Integrate over free (loop) frequencies first
    for lf in free_freqs:
        try:
            current_expr = current_expr.simplify_rational()
        except Exception:
            pass
        result = _integrate_by_residues(current_expr, lf, close_upper=True)
        if result == 0:
            result = _integrate_by_residues(current_expr, lf, close_upper=False)
        current_expr = result

    # Integrate over external frequencies
    # The exponentials e^{±iωt} determine contour closure direction.
    # Extract the time argument for each external frequency to decide.
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

        # Determine time argument τ for this frequency
        tau_arg = _extract_time_argument(full_integrand, integrand, omega_ext)

        if tau_arg is not None:
            current_expr = (
                heaviside(tau_arg) * result_upper
                + heaviside(-tau_arg) * result_lower
            )
        else:
            current_expr = result_upper
            status = 'partial'

    # Multiply by prefactors
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
# Compute full correction
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
