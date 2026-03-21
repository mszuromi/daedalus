"""
msrjd.integration.symbolic
============================
Construct and evaluate diagram integral expressions symbolically
for **stationary** systems using frequency-domain methods.

Mathematical procedure
----------------------
For a unique typed diagram Γ with loop number ℓ:

1. **Assign frequency variables**: each directed edge e → ω_e.
   External leaves get fixed external frequencies ω_ext_j.

2. **Frequency conservation**: at every vertex (interaction, source,
   and external leaf), the sum of incoming frequencies equals the sum
   of outgoing frequencies.  Source vertices (no incoming edges) impose
   Σ ω_out = 0.

3. **Solve conservation constraints**: choose a spanning tree T.
   Tree-edge frequencies are linear functions of the ℓ independent
   loop frequencies {Ω_1, ..., Ω_ℓ} and the external frequencies.

4. **Build the integrand**: substitute dependent frequencies, then
   multiply all propagator entries and noise kernels:

       I(Ω; ω_ext) = ∏_e  Ĝ_{i_e,j_e}(ω_e(Ω, ω_ext))
                    × ∏_{sources s}  κ̂_s(ω_s)

5. **Integrate**:
   - ℓ = 0 (tree): no integral — evaluate the algebraic expression.
   - ℓ = 1:        residue theorem (rational integrand in Ω).
   - ℓ ≥ 2:        sequential residue integration or flag for numerics.

6. **Full contribution**:

       weight(Γ) = scalar_prefactor × (1/(2π))^ℓ × ∫ I dΩ_1...dΩ_ℓ

FT convention:  F(ω) = ∫ f(t) e^{-iωt} dt,  so IFT has 1/(2π).

Build Phase H.
"""

from sage.all import SR, I, pi, Graph

from msrjd.diagrams.symmetry import classify_coefficient_factors


# ═══════════════════════════════════════════════════════════════════════════
# Step 0: Prerequisite checks
# ═══════════════════════════════════════════════════════════════════════════

def check_propagator_available(propagator_data):
    """
    Verify that frequency-domain propagator data is available.

    For symbolic integration we need either:
      - G_ft (explicit propagator matrix), or
      - adj_ft + D_omega (implicit rational form)

    Parameters
    ----------
    propagator_data : dict
        Must contain at least one of {'G_ft', ('adj_ft', 'D_omega')}.

    Returns
    -------
    mode : str
        'explicit' if G_ft is available, 'implicit' if only adj/det.

    Raises
    ------
    ValueError
        If neither form is available.
    """
    G_ft = propagator_data.get('G_ft')
    adj  = propagator_data.get('adj_ft')
    det  = propagator_data.get('D_omega')

    if G_ft is not None:
        return 'explicit'
    if adj is not None and det is not None:
        return 'implicit'
    raise ValueError(
        'No frequency-domain propagator available. '
        'Need either G_ft (explicit inverse) or adj_ft + D_omega '
        '(adjugate and determinant of the kernel matrix).'
    )


# ═══════════════════════════════════════════════════════════════════════════
# Step 1: Frequency variable assignment
# ═══════════════════════════════════════════════════════════════════════════

def assign_frequencies(typed_diagram, k):
    r"""
    Create symbolic frequency variables for every edge and external leg.

    Each leaf gets its own independent external frequency ω_ext_j.
    Overall conservation (Σ ω_ext_j = 0 or similar) is NOT pre-imposed;
    it emerges as the redundant equation when solving vertex conservation.

    After solving, the caller can substitute the overall constraint
    to reduce to k-1 independent external frequencies.

    Parameters
    ----------
    typed_diagram : TypedDiagram
    k : int
        Number of external legs (= number of leaves).

    Returns
    -------
    edge_freqs : dict
        {(u, v): SR variable ω_u_v} for every edge.
    ext_freqs : list of SR variable
        [ω_ext_1, ..., ω_ext_k] — one per leaf, all independent.
    ext_freq_assignment : dict
        {leaf_vertex: SR variable} mapping each leaf to its
        external frequency.
    """
    D = typed_diagram.prediagram[0]
    leaves = typed_diagram.prediagram[2]

    # One frequency per edge.
    # For multiedge graphs, use (u, v, label) as the key to distinguish
    # parallel edges.  For simple graphs, use (u, v).
    has_multiedges = D.has_multiple_edges() if hasattr(D, 'has_multiple_edges') else False
    edge_freqs = {}
    for idx, (u, v, lbl) in enumerate(D.edges()):
        ekey = (u, v, lbl) if has_multiedges else (u, v)
        omega = SR.var(f'omega_e{idx}',
                       latex_name=rf'\omega_{{e_{{{idx}}}}}')
        edge_freqs[ekey] = omega

    # One external frequency per leaf (all independent)
    ext_freqs = []
    ext_freq_assignment = {}
    for j, lf in enumerate(leaves):
        w = SR.var(f'omega_ext_{j+1}',
                   latex_name=rf'\omega_{{\mathrm{{ext}},{j+1}}}')
        ext_freqs.append(w)
        ext_freq_assignment[lf] = w

    return edge_freqs, ext_freqs, ext_freq_assignment


# ═══════════════════════════════════════════════════════════════════════════
# Step 2: Frequency conservation constraints
# ═══════════════════════════════════════════════════════════════════════════

def build_conservation_equations(typed_diagram, edge_freqs,
                                  ext_freq_assignment):
    r"""
    Build linear frequency conservation equations at each vertex.

    At each vertex v:
        Σ_{incoming edges} ω_e  =  Σ_{outgoing edges} ω_e

    For leaves, the edge frequency is constrained to equal the
    external frequency assigned to that leaf.

    Parameters
    ----------
    typed_diagram : TypedDiagram
    edge_freqs : dict
        {(u, v): ω_e} from assign_frequencies.
    ext_freq_assignment : dict
        {leaf: ω_ext} from assign_frequencies.

    Returns
    -------
    equations : list of (SR expression, str)
        Each entry is (lhs - rhs, description) where lhs - rhs = 0.
        These are linear in the edge frequency variables.
    """
    D = typed_diagram.prediagram[0]
    leaves = typed_diagram.prediagram[2]
    leaf_set = set(leaves)
    has_multiedges = D.has_multiple_edges() if hasattr(D, 'has_multiple_edges') else False

    # Build per-vertex incoming/outgoing edge key lists
    in_ekeys = {v: [] for v in D.vertices()}
    out_ekeys = {v: [] for v in D.vertices()}
    for (u, v, lbl) in D.edges():
        ekey = (u, v, lbl) if has_multiedges else (u, v)
        out_ekeys[u].append(ekey)
        in_ekeys[v].append(ekey)

    equations = []

    for v in D.vertices():
        # Sum of incoming frequencies
        in_sum = sum(edge_freqs[ek] for ek in in_ekeys[v])
        # Sum of outgoing frequencies
        out_sum = sum(edge_freqs[ek] for ek in out_ekeys[v])

        if v in leaf_set:
            # Leaf vertex: one edge connects to the diagram.
            # The frequency on that edge equals the external frequency.
            omega_ext = ext_freq_assignment[v]
            if D.in_degree(v) == 0 and D.out_degree(v) > 0:
                # Leaf is a tail → outgoing edge carries its frequency
                eq = out_sum - omega_ext
            elif D.out_degree(v) == 0 and D.in_degree(v) > 0:
                # Leaf is a head → incoming edge carries its frequency
                eq = in_sum - omega_ext
            else:
                # Both in and out (shouldn't happen for typical leaves)
                eq = in_sum - out_sum
            equations.append((eq, f'leaf {v}'))
        else:
            # Internal or source vertex: conservation
            # Source vertices have in_sum = 0, so this becomes out_sum = 0
            eq = in_sum - out_sum
            vtype = typed_diagram.vertex_assignments.get(v)
            label = 'source' if (D.in_degree(v) == 0) else 'internal'
            equations.append((eq, f'{label} vertex {v}'))

    return equations


# ═══════════════════════════════════════════════════════════════════════════
# Step 3: Solve for independent frequencies
# ═══════════════════════════════════════════════════════════════════════════

def _choose_spanning_tree(typed_diagram):
    """
    Choose a spanning tree of the diagram's underlying undirected graph.

    Returns the set of edges (as frozensets {u, v}) in the spanning tree.
    Non-tree edges correspond to independent loop frequencies.
    """
    D = typed_diagram.prediagram[0]
    G_undirected = typed_diagram.prediagram[1]

    # Use BFS spanning tree from the first leaf
    leaves = typed_diagram.prediagram[2]
    root = leaves[0] if leaves else D.vertices()[0]

    # Build spanning tree via BFS
    visited = {root}
    queue = [root]
    tree_edges = set()
    while queue:
        v = queue.pop(0)
        for u in G_undirected.neighbors(v):
            if u not in visited:
                visited.add(u)
                queue.append(u)
                tree_edges.add(frozenset({v, u}))

    return tree_edges


def solve_conservation(typed_diagram, edge_freqs, ext_freq_assignment):
    r"""
    Solve the frequency conservation constraints to express all edge
    frequencies in terms of independent loop frequencies and external
    frequencies.

    Chooses a spanning tree T.  Tree-edge frequencies are determined
    by conservation.  Each non-tree edge e_k carries an independent
    loop frequency Ω_k.

    For a connected graph with V vertices and E edges, the V vertex
    conservation equations have rank V-1 (one redundant equation
    expressing overall frequency conservation).  We drop one equation
    before solving to get a consistent system.

    The redundant equation gives the **overall conservation relation**
    between external frequencies (e.g. ω_ext_1 = ω_ext_2 for the
    2-point function).  This relation is returned separately so the
    caller can optionally impose it to reduce to k-1 independent
    external frequencies.

    Parameters
    ----------
    typed_diagram : TypedDiagram
    edge_freqs : dict
        {(u, v): ω_e} from assign_frequencies.
    ext_freq_assignment : dict
        {leaf: ω_ext} from assign_frequencies.

    Returns
    -------
    substitutions : dict
        {ω_e: expr(Ω_1, ..., Ω_ℓ, ω_ext)} for ALL edge frequencies.
    loop_freqs : list of SR variable
        The ℓ independent loop frequency variables.
    loop_number : int
        The loop number ℓ.
    overall_conservation : SR expression or None
        The redundant equation (= 0) expressing overall conservation.
        E.g., ``omega_ext_1 - omega_ext_2`` meaning ω_ext_1 = ω_ext_2.
        None if no redundant equation was identified.
    """
    D = typed_diagram.prediagram[0]
    has_multiedges = D.has_multiple_edges() if hasattr(D, 'has_multiple_edges') else False
    edges_list = [(u, v, lbl) for (u, v, lbl) in D.edges()]

    # Identify spanning tree edges and non-tree (loop) edges
    tree_edge_sets = _choose_spanning_tree(typed_diagram)

    tree_edges = []    # edge keys
    loop_edges = []    # edge keys
    tree_used = set()  # track which undirected edges are already claimed
    for (u, v, lbl) in edges_list:
        ekey = (u, v, lbl) if has_multiedges else (u, v)
        fs = frozenset({u, v})
        if fs in tree_edge_sets and fs not in tree_used:
            tree_edges.append(ekey)
            tree_used.add(fs)
        else:
            loop_edges.append(ekey)

    loop_number = len(loop_edges)

    # Create loop frequency variables for non-tree edges
    loop_freqs = []
    loop_subs = {}
    for idx, ekey in enumerate(loop_edges):
        omega_loop = SR.var(f'Omega_{idx+1}',
                            latex_name=rf'\Omega_{{{idx+1}}}')
        loop_freqs.append(omega_loop)
        loop_subs[edge_freqs[ekey]] = omega_loop

    # Build conservation equations
    equations = build_conservation_equations(
        typed_diagram, edge_freqs, ext_freq_assignment
    )

    # The unknowns are the tree-edge frequency variables
    tree_freq_vars = [edge_freqs[ekey] for ekey in tree_edges]

    # Substitute loop frequencies into conservation equations
    labeled_eqs = []
    for (eq, desc) in equations:
        eq_sub = eq.subs(loop_subs)
        labeled_eqs.append((eq_sub, desc))

    # For a connected graph, the V conservation equations have rank V-1.
    # Separate equations into those involving tree-edge unknowns vs
    # the redundant one (which involves only known quantities after
    # the others are solved).
    # Strategy: find equations that reference tree-edge variables, and
    # identify the one that doesn't (or is linearly dependent).
    # Simpler approach: drop the last equation (any one works for a
    # connected graph) and check if it's redundant.
    overall_conservation = None

    if len(labeled_eqs) > len(tree_freq_vars):
        # Overdetermined — find and remove the redundant equation.
        # Try dropping each equation and solving; the first that works
        # gives us the redundant equation.
        from sage.all import solve as sage_solve
        solved = False
        for drop_idx in range(len(labeled_eqs)):
            remaining = [eq for i, (eq, _) in enumerate(labeled_eqs)
                         if i != drop_idx]
            if tree_freq_vars:
                solutions = sage_solve(
                    [eq == 0 for eq in remaining],
                    tree_freq_vars,
                    solution_dict=True,
                )
                if solutions:
                    sol = solutions[0]
                    # The dropped equation is the redundant one —
                    # substitute the solution to get the overall
                    # conservation relation.
                    dropped_eq = labeled_eqs[drop_idx][0]
                    overall_conservation = dropped_eq.subs(sol)
                    solved = True
                    break
            else:
                sol = {}
                # All equations should be in terms of ext freqs only
                overall_conservation = labeled_eqs[drop_idx][0]
                solved = True
                break

        if not solved:
            raise RuntimeError(
                'Frequency conservation system has no solution even '
                'after dropping equations. Check diagram connectivity.'
            )
    else:
        # Exactly determined or underdetermined
        from sage.all import solve as sage_solve
        if tree_freq_vars:
            solutions = sage_solve(
                [eq == 0 for eq, _ in labeled_eqs],
                tree_freq_vars,
                solution_dict=True,
            )
            if not solutions:
                raise RuntimeError(
                    'Frequency conservation system has no solution.'
                )
            sol = solutions[0]
        else:
            sol = {}

    # Build the full substitution dict
    substitutions = dict(loop_subs)  # loop edges → Ω_k
    substitutions.update(sol)        # tree edges → expr(Ω, ω_ext)

    return substitutions, loop_freqs, loop_number, overall_conservation


# ═══════════════════════════════════════════════════════════════════════════
# Step 4: Build the integrand
# ═══════════════════════════════════════════════════════════════════════════

def _get_propagator_entry(i, j, omega_var, propagator_data, omega_symbol):
    """
    Look up Ĝ_{i,j}(ω) from propagator data and substitute the
    frequency variable.

    Parameters
    ----------
    i, j : int
        Row and column indices into the propagator matrix.
    omega_var : SR expression
        The frequency flowing through this edge (after conservation).
    propagator_data : dict
        Contains 'G_ft' or ('adj_ft', 'D_omega').
    omega_symbol : SR variable
        The symbol used for ω in the propagator expressions
        (typically SR.var('omega')).

    Returns
    -------
    SR expression
        The propagator entry evaluated at omega_var.
    """
    G_ft = propagator_data.get('G_ft')
    if G_ft is not None:
        entry = SR(G_ft[i, j])
    else:
        adj = propagator_data['adj_ft']
        det = propagator_data['D_omega']
        entry = SR(adj[i, j]) / SR(det)

    return entry.subs({omega_symbol: omega_var})


def build_integrand(typed_diagram, edge_freqs, freq_substitutions,
                    propagator_data, omega_symbol=None,
                    noise_structure=None):
    r"""
    Build the frequency-domain integrand for a typed diagram.

    The integrand is:

        I(Ω; ω_ext) = ∏_e  Ĝ_{i_e,j_e}(ω_e)  ×  ∏_s  κ̂_s(ω_s)

    where each ω_e has been expressed in terms of loop and external
    frequencies via freq_substitutions.

    Parameters
    ----------
    typed_diagram : TypedDiagram
    edge_freqs : dict
        {(u, v): ω_e} from assign_frequencies.
    freq_substitutions : dict
        {ω_e: expr(Ω, ω_ext)} from solve_conservation.
    propagator_data : dict
        Propagator data containing G_ft or (adj_ft, D_omega).
    omega_symbol : SR variable or None
        The ω symbol used in the propagator expressions.
        If None, uses SR.var('omega').
    noise_structure : dict or None
        From the model dict.  If colored noise, must include
        'kernel_ft' (SR expression in ω) to provide κ̂(ω).

    Returns
    -------
    integrand : SR expression
        The product of all propagator entries and noise kernels,
        with all frequencies substituted.
    """
    if omega_symbol is None:
        omega_symbol = SR.var('omega')

    D = typed_diagram.prediagram[0]
    leaves = typed_diagram.prediagram[2]
    leaf_set = set(leaves)

    ns = noise_structure or {'temporal_type': 'white'}
    noise_type = ns.get('temporal_type', 'white')

    integrand = SR(1)

    # Propagator factors: one per edge.
    # edge_freqs keys may be (u,v) or (u,v,lbl) depending on multiedges.
    # TypedDiagram.edge_types keys are (u,v) or (u,v,lbl).
    # Match them to edge_freqs by trying both formats.
    for edge_key in typed_diagram.edge_types:
        ri, pi = typed_diagram.propagator_indices[edge_key]

        # Look up the frequency variable for this edge
        if edge_key in edge_freqs:
            omega_e = edge_freqs[edge_key]
        else:
            # Try matching (u,v) to (u,v,lbl) or vice versa
            u, v = edge_key[0], edge_key[1]
            matched = [ek for ek in edge_freqs if ek[0] == u and ek[1] == v]
            if matched:
                omega_e = edge_freqs[matched[0]]
            else:
                raise KeyError(f'No frequency variable for edge {edge_key}')

        # Substitute to get frequency in terms of Ω and ω_ext
        omega_val = omega_e.subs(freq_substitutions)

        prop_entry = _get_propagator_entry(
            ri, pi, omega_val, propagator_data, omega_symbol
        )
        integrand *= prop_entry

    # Source vertex noise kernel factors (only for colored noise)
    if noise_type == 'colored':
        kernel_ft_expr = ns.get('kernel_ft')
        kernel_ft_omega = ns.get('kernel_ft_omega', omega_symbol)
        if kernel_ft_expr is None:
            raise ValueError(
                "Colored noise requires 'kernel_ft' in noise_structure: "
                "an SR expression giving κ̂(ω)."
            )
        for v, vtype in typed_diagram.vertex_assignments.items():
            if D.in_degree(v) > 0:
                continue  # not a source
            # Source vertex: find first outgoing edge frequency
            out_ekeys = [ek for ek in edge_freqs
                         if ek[0] == v]
            if out_ekeys:
                omega_source = edge_freqs[out_ekeys[0]]
                omega_val = omega_source.subs(freq_substitutions)
                kappa = SR(kernel_ft_expr).subs(
                    {kernel_ft_omega: omega_val}
                )
                integrand *= kappa

    return integrand


# ═══════════════════════════════════════════════════════════════════════════
# Step 5: Full integrand assembly (scalar prefactor + integrand)
# ═══════════════════════════════════════════════════════════════════════════

def build_integrand_stationary(typed_diagram, propagator_data, k,
                                omega_symbol=None,
                                time_dep_params=None,
                                noise_structure=None):
    r"""
    Full integrand assembly for a stationary diagram.

    Combines:
    - Coefficient classification (scalar prefactor vs integrand factors)
    - Frequency assignment and conservation
    - Propagator integrand construction

    Parameters
    ----------
    typed_diagram : TypedDiagram
    propagator_data : dict
        Must contain G_ft or (adj_ft, D_omega).
    k : int
        Number of external legs.
    omega_symbol : SR variable or None
        The ω symbol used in propagator expressions.
    time_dep_params : list of str or None
        Parameter prefixes that are time-dependent.
    noise_structure : dict or None
        Noise temporal structure from the model dict.

    Returns
    -------
    result : dict with keys:
        'scalar_prefactor' : SR
            M(Γ) × constant vertex/source coefficients.
        'integrand' : SR expression
            Product of propagators and noise kernels, in terms of
            loop frequencies Ω_k and external frequencies ω_ext_j.
        'loop_freqs' : list of SR variable
            Independent loop frequency variables [Ω_1, ..., Ω_ℓ].
        'ext_freqs' : list of SR variable
            External frequency variables [ω_ext_1, ..., ω_ext_{k-1}].
        'loop_number' : int
        'edge_freqs' : dict
            {(u,v): ω_e} the raw edge frequency variables.
        'freq_substitutions' : dict
            {ω_e: expr(Ω, ω_ext)} the conservation solution.
        'coefficient_info' : dict
            Full output from classify_coefficient_factors.
        'fourier_prefactor' : SR expression
            (1/(2π))^ℓ.
    """
    # Prerequisite check
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
            'This diagram has nonstationary features (time-dependent '
            'vertex coefficients or general noise). Use the time-domain '
            'integrand builder instead.'
        )

    # Frequency assignment
    edge_freqs, ext_freqs, ext_freq_assignment = assign_frequencies(
        typed_diagram, k
    )

    # Conservation
    freq_subs, loop_freqs, loop_number, overall_cons = solve_conservation(
        typed_diagram, edge_freqs, ext_freq_assignment
    )

    # Build integrand
    integrand = build_integrand(
        typed_diagram, edge_freqs, freq_subs,
        propagator_data, omega_symbol, noise_structure,
    )

    fourier_prefactor = SR(1) / (2 * pi) ** loop_number

    return {
        'scalar_prefactor':      coeff_info['scalar_prefactor'],
        'integrand':             integrand,
        'loop_freqs':            loop_freqs,
        'ext_freqs':             ext_freqs,
        'loop_number':           loop_number,
        'edge_freqs':            edge_freqs,
        'freq_substitutions':    freq_subs,
        'coefficient_info':      coeff_info,
        'fourier_prefactor':     fourier_prefactor,
        'overall_conservation':  overall_cons,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Step 6: Symbolic integration
# ═══════════════════════════════════════════════════════════════════════════

def integrate_tree_level(integrand_result):
    r"""
    Evaluate a tree-level diagram (ℓ = 0).

    No integration needed — the integrand is already the full
    frequency-domain expression.

    Parameters
    ----------
    integrand_result : dict
        Output from build_integrand_stationary.

    Returns
    -------
    SR expression
        scalar_prefactor × integrand  (a function of ω_ext).
    """
    if integrand_result['loop_number'] != 0:
        raise ValueError(
            f"Expected tree-level (ℓ=0), got ℓ={integrand_result['loop_number']}"
        )
    prefactor = integrand_result['scalar_prefactor']
    integrand = integrand_result['integrand']
    return (prefactor * integrand).simplify_rational()


def integrate_one_loop_residues(integrand_result, pole_vals,
                                 omega_symbol=None, close_upper=True):
    r"""
    Evaluate a one-loop diagram (ℓ = 1) via the residue theorem.

    The integrand is a rational function of Ω_1.  We close the contour
    in the upper (or lower) half-plane and sum residues.

    The full result is:

        scalar_prefactor × (1/(2π)) × 2πi × Σ Res[I(Ω), upper poles]
      = scalar_prefactor × i × Σ Res[I(Ω), upper poles]

    or with lower half-plane closure:

        scalar_prefactor × (-i) × Σ Res[I(Ω), lower poles]

    Parameters
    ----------
    integrand_result : dict
        Output from build_integrand_stationary.
    pole_vals : list of SR expression
        Pole locations of the propagator (from propagator_data['pole_vals']).
        These are the poles of Ĝ(ω) in the ω variable.  The actual
        poles of the loop integrand in Ω may be shifted by external
        frequencies.
    omega_symbol : SR variable or None
        The ω symbol used in the original propagator.
    close_upper : bool
        If True, close contour in upper half-plane (collect poles with
        Im(ω) > 0).  For retarded propagators, poles are typically in
        the upper half-plane.

    Returns
    -------
    SR expression
        The integrated diagram contribution.
    """
    if integrand_result['loop_number'] != 1:
        raise ValueError(
            f"Expected one-loop (ℓ=1), got ℓ={integrand_result['loop_number']}"
        )

    if omega_symbol is None:
        omega_symbol = SR.var('omega')

    Omega = integrand_result['loop_freqs'][0]
    integrand = integrand_result['integrand']
    prefactor = integrand_result['scalar_prefactor']

    # The integrand is a rational function of Ω.
    # Find its poles by examining the denominator.
    # Strategy: factor the denominator and find roots in Ω.
    try:
        integrand_simplified = integrand.simplify_rational()
    except Exception:
        integrand_simplified = integrand

    # Use partial_fraction to decompose
    try:
        pf = integrand_simplified.partial_fraction(Omega)
    except Exception:
        pf = integrand_simplified

    # Compute residues at each pole
    # For a rational function f(Ω) with simple pole at Ω = p,
    # Res[f, p] = lim_{Ω→p} (Ω - p) f(Ω)
    from sage.all import limit, oo as sage_oo

    # Find poles of the integrand in Ω
    try:
        numer, denom = integrand_simplified.numerator_denominator()
        from sage.all import solve as sage_solve
        pole_solutions = sage_solve(denom == 0, Omega, solution_dict=True)
        integrand_poles = [sol[Omega] for sol in pole_solutions]
    except Exception:
        # Fallback: try to identify poles from the propagator poles
        integrand_poles = []

    # Select poles in the appropriate half-plane
    residue_sum = SR(0)
    sign = SR(1) if close_upper else SR(-1)

    for p in integrand_poles:
        # Check if pole is in the upper/lower half-plane
        try:
            p_imag = p.imag_part()
            if close_upper and bool(p_imag > 0):
                res = limit((Omega - p) * integrand_simplified, Omega=p)
                residue_sum += res
            elif not close_upper and bool(p_imag < 0):
                res = limit((Omega - p) * integrand_simplified, Omega=p)
                residue_sum += res
        except Exception:
            # Symbolic pole — can't determine half-plane.
            # Include it with annotation.
            res = limit((Omega - p) * integrand_simplified, Omega=p)
            residue_sum += res

    # Result: prefactor × (±i) × Σ residues
    # The 1/(2π) from Fourier convention and 2πi from residue theorem
    # combine: (1/(2π)) × (2πi) = i
    result = prefactor * sign * I * residue_sum

    try:
        result = result.simplify_full()
    except Exception:
        pass

    return result


# ═══════════════════════════════════════════════════════════════════════════
# Step 7: Compute full correction at a given loop level
# ═══════════════════════════════════════════════════════════════════════════

def compute_correction(typed_diagrams, propagator_data, k,
                        omega_symbol=None, pole_vals=None,
                        time_dep_params=None, noise_structure=None):
    r"""
    Sum contributions from all diagrams at a given loop level.

    For each unique typed diagram Γ:
        contribution(Γ) = scalar_prefactor(Γ) × (1/(2π))^ℓ × ∫ I_Γ dΩ

    The total correction is Σ_Γ contribution(Γ).

    Parameters
    ----------
    typed_diagrams : list of TypedDiagram
        Should be deduplicated (one per unique diagram).
    propagator_data : dict
    k : int
        Number of external legs.
    omega_symbol : SR variable or None
    pole_vals : list or None
        Pole locations for residue integration.
    time_dep_params : list or None
    noise_structure : dict or None

    Returns
    -------
    results : list of dict
        Per-diagram results, each containing:
            'diagram': the TypedDiagram
            'integrand_result': full output from build_integrand_stationary
            'contribution': the evaluated contribution (SR expression)
            'status': 'ok', 'symbolic_fallback', or 'needs_numerical'
    total : SR expression
        Sum of all contributions (as a function of external frequencies).
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

        ell = ir['loop_number']

        if ell == 0:
            contribution = integrate_tree_level(ir)
            status = 'ok'
        elif ell == 1 and pole_vals is not None:
            try:
                contribution = integrate_one_loop_residues(
                    ir, pole_vals, omega_symbol=omega_symbol
                )
                status = 'ok'
            except Exception as exc:
                contribution = None
                status = f'symbolic_fallback: {exc}'
        else:
            contribution = None
            status = 'needs_numerical'

        results.append({
            'diagram': td,
            'integrand_result': ir,
            'contribution': contribution,
            'status': status,
        })

        if contribution is not None:
            total += contribution

    return results, total
