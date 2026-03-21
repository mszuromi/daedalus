"""
msrjd.integration.symbolic
============================
Construct and evaluate diagram integral expressions symbolically
for **stationary** systems using frequency-domain methods.

Mathematical procedure (Helias & Dahmen Ch. 9)
-----------------------------------------------
For a unique typed diagram Γ contributing to the k-point cumulant
⟨x_{a₁}(t₁) ⋯ x_{aₖ}(tₖ)⟩, with loop number ℓ:

1. **Assign frequency variables**: each directed edge e → ω_e.
   External leaves get independent external frequencies ω_ext_j.

2. **Frequency conservation**: at every vertex, the sum of incoming
   frequencies equals the sum of outgoing frequencies.  This yields
   V−1 independent constraints (one redundant = overall conservation).

3. **Solve conservation constraints**: choose a spanning tree T.
   Tree-edge frequencies are linear functions of the ℓ independent
   loop frequencies {Ω_1, ..., Ω_ℓ} and the k external frequencies.
   Overall conservation gives ω_ext_k = −Σ_{j<k} ω_ext_j, reducing
   to k−1 independent external frequencies.

4. **Build the frequency-domain integrand**: substitute resolved
   frequencies, then multiply all propagator entries and noise kernels:

       F(Ω; ω_ext) = ∏_e  Ĝ_{i_e,j_e}(ω_e(Ω, ω_ext))
                    × ∏_{sources s}  κ̂_s(ω_s)

5. **Include external-leg exponentials**: each external leg ℓ
   contributes a factor e^{−iω_ℓ t_ℓ} from the functional derivative
   δW/δj(t_ℓ) = ∫(dω/2π) e^{−iωt_ℓ} δW/δJ(ω).

6. **Integrate**: the k-point cumulant is

       C(t₁,...,tₖ) = scalar_prefactor
           × ∏_{j=1}^{k-1} ∫(dω_ext_j / 2π)
           × ∏_{m=1}^{ℓ} ∫(dΩ_m / 2π)
           × [∏_{j=1}^{k} e^{−iω_ext_j t_j}]
           × F(Ω; ω_ext)

   where ω_ext_k = −ω_ext_1 − ⋯ − ω_ext_{k-1} from overall
   conservation.

   The result is a function of times t₁,...,tₖ — NOT frequencies.

   For stationary systems, C depends only on time *differences*:
   e.g. C(t₁,t₂) = C(t₂−t₁).

   Integration is performed by residues (contour closure in upper
   or lower half-plane depending on the sign of the time argument).

FT convention:  F(ω) = ∫ f(t) e^{iωt} dt,  so IFT has e^{-iωt}/(2π).

Build Phase H.
"""

from sage.all import SR, I, pi, Graph, exp, assume, forget

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
    overall_conservation = None

    if len(labeled_eqs) > len(tree_freq_vars):
        # Overdetermined — find and remove the redundant equation.
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
                    dropped_eq = labeled_eqs[drop_idx][0]
                    overall_conservation = dropped_eq.subs(sol)
                    solved = True
                    break
            else:
                sol = {}
                overall_conservation = labeled_eqs[drop_idx][0]
                solved = True
                break

        if not solved:
            raise RuntimeError(
                'Frequency conservation system has no solution even '
                'after dropping equations. Check diagram connectivity.'
            )
    else:
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

        F(Ω; ω_ext) = ∏_e  Ĝ_{i_e,j_e}(ω_e)  ×  ∏_s  κ̂_s(ω_s)

    where each ω_e has been expressed in terms of loop and external
    frequencies via freq_substitutions.

    This does NOT include the e^{−iωt} external-leg factors — those
    are added during integration (see integrate_to_time_domain).

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

    for edge_key in typed_diagram.edge_types:
        ri, pi = typed_diagram.propagator_indices[edge_key]

        # Look up the frequency variable for this edge
        if edge_key in edge_freqs:
            omega_e = edge_freqs[edge_key]
        else:
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
    - External time variables t_1, ..., t_k

    The full k-point cumulant contribution from this diagram is:

        C(t₁,...,tₖ) = scalar_prefactor
            × ∏_{j=1}^{k-1} ∫(dω_j / 2π)
            × ∏_{m=1}^{ℓ} ∫(dΩ_m / 2π)
            × [∏_{j=1}^{k} e^{−iω_j t_j}]
            × F(Ω; ω)

    where ω_k = −ω_1 − ⋯ − ω_{k-1} from overall conservation.

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
        'full_integrand' : SR expression
            integrand × ∏ e^{−iω_j t_j} (the complete integrand
            including time-domain exponentials).
        'loop_freqs' : list of SR variable
            Independent loop frequency variables [Ω_1, ..., Ω_ℓ].
        'ext_freqs' : list of SR variable
            External frequency variables [ω_ext_1, ..., ω_ext_k].
        'ext_freqs_independent' : list of SR variable
            Independent external frequencies after overall conservation
            [ω_ext_1, ..., ω_ext_{k-1}].
        'ext_times' : list of SR variable
            External time variables [t_1, ..., t_k].
        'overall_conservation_sub' : dict or None
            Substitution {ω_ext_k: −Σ ω_ext_j} from overall conservation.
        'loop_number' : int
        'edge_freqs' : dict
            {(u,v): ω_e} the raw edge frequency variables.
        'freq_substitutions' : dict
            {ω_e: expr(Ω, ω_ext)} the conservation solution.
        'coefficient_info' : dict
            Full output from classify_coefficient_factors.
        'fourier_prefactor' : SR expression
            (1/(2π))^{ℓ + k − 1}  (one per integration variable).
        'overall_conservation' : SR expression or None
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

    # Build propagator integrand F(Ω; ω_ext)
    integrand = build_integrand(
        typed_diagram, edge_freqs, freq_subs,
        propagator_data, omega_symbol, noise_structure,
    )

    # Create external time variables t_1, ..., t_k
    ext_times = []
    for j in range(k):
        t = SR.var(f't_{j+1}', latex_name=rf't_{{{j+1}}}')
        ext_times.append(t)

    # Apply overall conservation to reduce to k-1 independent ext freqs.
    # Solve overall_cons == 0 for ext_freqs[-1] (the last ext freq).
    overall_cons_sub = None
    ext_freqs_independent = list(ext_freqs)  # copy
    if overall_cons is not None and len(ext_freqs) >= 2:
        from sage.all import solve as sage_solve
        # Solve for the last external frequency
        target = ext_freqs[-1]
        cons_sol = sage_solve(overall_cons == 0, target, solution_dict=True)
        if cons_sol:
            overall_cons_sub = cons_sol[0]
            ext_freqs_independent = ext_freqs[:-1]

    # Build the full integrand with exponential factors from the
    # inverse Fourier transform at each external leg.
    #
    # The sign convention is determined by the direction of the
    # external leg relative to the diagram:
    #   - TAIL leaf (outgoing into diagram): e^{-iω t}
    #     (from δ/δj(t) = ∫(dω/2π) e^{-iωt} δ/δJ(ω))
    #   - HEAD leaf (incoming from diagram): e^{+iω t}
    #     (the propagator G(ω) connects J(ω) to J(-ω),
    #      so at the receiving end the frequency has opposite sign)
    #
    # This ensures that for the 2-pt function 0→2→1 with ω_1 = ω_2,
    # the combined exponential is e^{-iωt₁}×e^{+iωt₂} = e^{iω(t₂-t₁)},
    # giving the correct IFT: C(t₁,t₂) = ∫(dω/2π) e^{iω(t₂-t₁)} Ĝ(ω).
    D = typed_diagram.prediagram[0]
    leaves = typed_diagram.prediagram[2]

    exp_factor = SR(1)
    for j in range(k):
        omega_j = ext_freqs[j]
        leaf_v = leaves[j]
        # If this is the dependent frequency, substitute
        if overall_cons_sub is not None and omega_j in overall_cons_sub:
            omega_j_resolved = overall_cons_sub[omega_j]
        else:
            omega_j_resolved = omega_j

        # Determine sign based on leaf directionality
        is_tail = (D.out_degree(leaf_v) > 0 and D.in_degree(leaf_v) == 0)
        if is_tail:
            exp_factor *= exp(-I * omega_j_resolved * ext_times[j])
        else:
            exp_factor *= exp(+I * omega_j_resolved * ext_times[j])

    # Also apply overall conservation to the propagator integrand
    integrand_resolved = integrand
    if overall_cons_sub is not None:
        integrand_resolved = integrand.subs(overall_cons_sub)

    full_integrand = integrand_resolved * exp_factor

    # Fourier prefactor: (1/(2π)) per integration variable
    # Integration variables = (k-1) independent ext freqs + ℓ loop freqs
    n_integrals = len(ext_freqs_independent) + loop_number
    fourier_prefactor = SR(1) / (2 * pi) ** n_integrals

    return {
        'scalar_prefactor':          coeff_info['scalar_prefactor'],
        'integrand':                 integrand_resolved,
        'full_integrand':            full_integrand,
        'loop_freqs':                loop_freqs,
        'ext_freqs':                 ext_freqs,
        'ext_freqs_independent':     ext_freqs_independent,
        'ext_times':                 ext_times,
        'overall_conservation_sub':  overall_cons_sub,
        'loop_number':               loop_number,
        'edge_freqs':                edge_freqs,
        'freq_substitutions':        freq_subs,
        'coefficient_info':          coeff_info,
        'fourier_prefactor':         fourier_prefactor,
        'overall_conservation':      overall_cons,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Step 6: Residue integration helpers
# ═══════════════════════════════════════════════════════════════════════════

def _find_poles(expr, var):
    """Find poles of a rational expression in the given variable."""
    from sage.all import solve as sage_solve
    try:
        expr_s = expr.simplify_rational()
    except Exception:
        expr_s = expr
    try:
        numer, denom = expr_s.numerator_denominator()
        pole_solutions = sage_solve(denom == 0, var, solution_dict=True)
        return [sol[var] for sol in pole_solutions]
    except Exception:
        return []


def _pole_order(expr, var, pole):
    """Determine the order of a pole by testing successive multiplications."""
    from sage.all import limit
    for n in range(1, 10):
        try:
            test = ((var - pole)**n * expr).simplify_rational()
            val = limit(test, **{str(var): pole})
            val_str = str(val)
            # Check if the limit is finite (not infinity or indeterminate)
            if ('Infinity' not in val_str and 'infinity' not in val_str
                    and 'ind' not in val_str.lower()
                    and 'und' not in val_str.lower()):
                return n
        except Exception:
            continue
    return 1  # fallback


def _residue_at_pole(expr, var, pole):
    """
    Compute the residue of expr at var = pole.

    Handles poles of arbitrary order n:
        Res[f, p] = (1/(n-1)!) × lim_{z→p} d^{n-1}/dz^{n-1} [(z-p)^n f(z)]
    """
    from sage.all import limit, diff, factorial

    n = _pole_order(expr, var, pole)

    try:
        g = ((var - pole)**n * expr).simplify_rational()
        if n == 1:
            res = limit(g, **{str(var): pole})
        else:
            dg = g
            for _ in range(n - 1):
                dg = diff(dg, var)
            res = limit(dg, **{str(var): pole}) / factorial(n - 1)
        return res
    except Exception:
        # Fallback for simple pole
        try:
            return limit((var - pole) * expr, **{str(var): pole})
        except Exception:
            return SR(0)


def _integrate_by_residues(expr, var, close_upper=True):
    r"""
    Evaluate ∫_{-∞}^{∞} expr dvar  via the residue theorem.

    Closes contour in the upper (Im > 0) or lower (Im < 0) half-plane.

    Returns 2πi × Σ Res (upper) or -2πi × Σ Res (lower).
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
            # Cannot determine — skip (symbolic pole with unknown sign)
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
# Step 7: Integrate to time domain
# ═══════════════════════════════════════════════════════════════════════════

def integrate_to_time_domain(integrand_result):
    r"""
    Evaluate the full time-domain cumulant contribution from a diagram.

    Performs all frequency integrals (external + loop) by residues,
    returning C(t₁, ..., tₖ) — a function of the external times.

    For the 2-point function (k=2), the result is:

        C(t₁, t₂) = scalar_prefactor × (1/2π)^{ℓ+1}
            × ∫ dω e^{iω(t₂−t₁)} × F(ω, Ω)  × ∫ dΩ₁ ... dΩ_ℓ

    For a stationary system this depends only on τ = t₂ − t₁.

    The contour is closed in the upper half-plane for τ > 0 and
    lower half-plane for τ < 0.  Both cases are computed and
    assembled with Heaviside step functions Θ(τ) and Θ(−τ).

    Parameters
    ----------
    integrand_result : dict
        Output from build_integrand_stationary.

    Returns
    -------
    dict with keys:
        'time_domain_result' : SR expression
            The cumulant C(t₁,...,tₖ) as a function of external times.
            For stationary systems, expressed in terms of time differences.
        'frequency_domain_integrand' : SR expression
            The integrand F(ω_ext, Ω) before Fourier transform.
        'integration_variables' : list of SR variable
            [ω_ext_1, ..., ω_ext_{k-1}, Ω_1, ..., Ω_ℓ].
        'status' : str
            'ok', 'partial', or 'needs_numerical'.
    """
    loop_number = integrand_result['loop_number']
    loop_freqs = integrand_result['loop_freqs']
    ext_freqs_indep = integrand_result['ext_freqs_independent']
    ext_times = integrand_result['ext_times']
    prefactor = integrand_result['scalar_prefactor']
    fourier_pf = integrand_result['fourier_prefactor']
    full_integrand = integrand_result['full_integrand']

    # All integration variables: loop freqs first, then ext freqs
    # (order matters for sequential residue evaluation)
    int_vars = list(loop_freqs) + list(ext_freqs_indep)

    # Sequential residue integration over all variables.
    # For each variable, we integrate ∫ dv/(2π) [integrand].
    # The 1/(2π) factors are already in fourier_pf, so we just
    # compute the raw contour integrals and multiply by fourier_pf
    # at the end.

    current_expr = full_integrand

    status = 'ok'

    # Integrate over loop frequencies first (close in upper half-plane;
    # for retarded propagators, poles are in upper half-plane).
    for lf in loop_freqs:
        try:
            current_expr = current_expr.simplify_rational()
        except Exception:
            pass
        integral_result = _integrate_by_residues(current_expr, lf,
                                                  close_upper=True)
        if integral_result == 0:
            # Try the other half-plane
            integral_result = _integrate_by_residues(current_expr, lf,
                                                      close_upper=False)
        current_expr = integral_result

    # Integrate over external frequencies.
    # The exponentials e^{-iωt} determine which half-plane to close in:
    # for e^{-iωt} with t > 0, close in lower half-plane (convergent);
    # for t < 0, close in upper half-plane.
    # Since t₁, t₂ have unknown sign, we compute both cases and use
    # Heaviside (step function) to combine them.
    #
    # For the 2-point function with one independent ext freq ω:
    # the exponential is e^{iω(t₂−t₁)} so:
    #   τ = t₂ − t₁ > 0 → close upper (for e^{iωτ})
    #   τ = t₂ − t₁ < 0 → close lower
    #
    # More generally for k-point, we integrate sequentially, and
    # at each step the exponential determines the closure direction.
    # The general case produces piecewise expressions in time orderings.

    for omega_ext in ext_freqs_indep:
        try:
            current_expr = current_expr.simplify_rational()
        except Exception:
            pass

        # Determine the exponent coefficient of this frequency variable
        # in the exponential factor.  For e^{-iωt}, the "time argument"
        # is the coefficient of (-iω) in the exponent, which determines
        # the sign for contour closure.
        #
        # Strategy: compute both closures and combine with step functions.
        try:
            result_upper = _integrate_by_residues(current_expr, omega_ext,
                                                   close_upper=True)
            result_lower = _integrate_by_residues(current_expr, omega_ext,
                                                   close_upper=False)
        except Exception:
            status = 'partial'
            break

        # Determine the time argument from the exponential.
        # For k=2 with overall conservation ω₂ = ω₁, the full exponential
        # is e^{-iω₁t₁} × e^{-iω₁t₂} ... but after conservation ω₂=ω₁
        # the combined exponent is e^{-iω₁(t₁-t₂)} = e^{iω₁(t₂-t₁)}.
        # The coefficient of iω₁ is (t₂-t₁) = τ.
        # Close upper for τ > 0, lower for τ < 0.
        #
        # General method: extract the time argument as the coefficient of
        # (-i × omega_ext) in the exponent of the full integrand.
        # This works by differentiating the log of the exponential part
        # w.r.t. omega_ext, but a simpler approach: look at the exponential
        # structure directly.

        # For the stationary 2-point case:
        # The upper closure gives the τ > 0 part, lower gives τ < 0.
        # Use the SageMath unit_step (Heaviside) function.
        from sage.all import heaviside

        # Identify the "time argument" τ such that the exponential
        # factor in omega_ext is e^{i ω τ}.
        # Method: take derivative of full_integrand w.r.t. omega_ext,
        # evaluate at omega_ext=0, and extract the coefficient.
        # Alternative (more robust): examine the coefficient of i*omega_ext
        # in the exponent.

        # For the simple case where the exponential part has a clear
        # time argument, we use: upper for τ > 0, lower for τ < 0.
        # Compute the effective τ from the exponentials in the original
        # full_integrand.

        tau_arg = _extract_time_argument(
            integrand_result, omega_ext
        )

        if tau_arg is not None:
            current_expr = (
                heaviside(tau_arg) * result_upper
                + heaviside(-tau_arg) * result_lower
            )
        else:
            # Cannot determine time argument; return upper closure
            # (appropriate for retarded response, τ > 0)
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
        'frequency_domain_integrand': integrand_result['integrand'],
        'integration_variables': int_vars,
        'ext_times': ext_times,
        'status': status,
    }


def _extract_time_argument(integrand_result, omega_var):
    r"""
    Extract the effective time argument τ such that the external-leg
    exponential factor, as a function of omega_var, has the form
    e^{i ω τ} (plus terms independent of ω).

    For the 2-point function 0→2→1 with ω_ext_2 = ω_ext_1:
    - Leaf 0 (tail): e^{-iω₁t₁}
    - Leaf 1 (head): e^{+iω₁t₂}  (ω₂ = ω₁ from conservation)
    - Combined: e^{iω₁(t₂-t₁)}
    - So τ = t₂ - t₁.

    Contour closure: for e^{iωτ} with τ > 0, close in the upper
    half-plane (convergent for Im(ω) > 0).

    Returns
    -------
    SR expression or None
        The time argument τ such that the exponential is e^{iωτ}.
    """
    # Extract the exponential factor from the full integrand.
    # The full_integrand = propagator_terms × exp_factor
    # The exponent in the exp_factor is a linear function of omega_var.
    # We need the coefficient of (i * omega_var) in that exponent.
    #
    # Compute by differentiating the full_integrand's log w.r.t. omega_var
    # and extracting the part from the exponential.
    # Simpler: rebuild the exponent from the stored information.

    ext_times = integrand_result['ext_times']
    ext_freqs = integrand_result['ext_freqs']
    overall_sub = integrand_result.get('overall_conservation_sub', None)

    # Rebuild the exponent using the same sign logic as build_integrand_stationary
    D = None
    leaves = None
    # We need the diagram to determine tail vs head
    # Access it from the typed diagram stored in coefficient_info
    # Actually we don't store the diagram directly, but we can recover
    # the sign from the full_integrand.
    #
    # Alternative: just differentiate the exponent w.r.t. omega_var.
    # The full_integrand = F(ω) × e^{g(ω)} where g is linear in ω.
    # We want g'(omega_var) = i × τ, so τ = g'(omega_var) / i.
    #
    # Extract from the full_integrand directly:
    full_int = integrand_result['full_integrand']
    try:
        # Take the derivative and divide by the full integrand to get
        # the exponent's derivative: d/dω [log(full_int)] contains
        # both the rational part (from propagators) and i*τ (from exp).
        # But we only want the exp part.
        # Better approach: differentiate the exp factor alone.
        # Rebuild the exp factor.
        pass
    except Exception:
        pass

    # Rebuild the exponent directly from ext_times, ext_freqs, overall_sub
    # and the leaf directions (which we stored implicitly in the overall_cons).
    # For now, use a numerical approach: evaluate the full_integrand
    # at two omega_var values and look at the ratio to extract the exponent.
    #
    # Simplest robust approach: compute d(full_integrand)/d(omega_var),
    # evaluate at a safe point, divide by full_integrand, and extract
    # the imaginary coefficient.
    #
    # Actually the CLEANEST approach: just look at the coefficient of
    # omega_var in the exponent by examining full_integrand.
    # SageMath can extract this via operands of the exponential.

    # Most robust: differentiate the full_integrand w.r.t. omega_var,
    # divide by full_integrand, and the result should be
    # (rational function derivative) + i*τ.
    # But for large expressions this is fragile.

    # Use the direct approach: compute the exponent coefficient from
    # the structure we know.
    # The exponent is Σ_j (sign_j) * (-i) * ω_j_resolved * t_j
    # where sign_j = +1 for tail, -1 for head.
    # We need d(exponent)/d(omega_var).
    # The exponent as a function of omega_var is linear, so the
    # coefficient of omega_var in the exponent is constant.

    # Parse the coefficient of omega_var from the exponent of the
    # full_integrand by looking at the exponential part.
    # Extract all exp() operands.

    try:
        # Strategy: factor the full integrand into rational × exponential
        # The exponential part is e^{i*omega*tau + ...}
        # We differentiate log(full_int) w.r.t. omega_var
        # and take the limit as omega_var → 0.
        # The rational part's derivative vanishes asymptotically, leaving i*τ.
        # But this is not clean.

        # CLEAN approach: since we BUILT the exp_factor, we know its structure.
        # The exponent's derivative w.r.t. omega_var is constant (linear exponent).
        # Extract it by differentiating the exponential part.

        # Reconstruct the exponent from the full integrand by finding
        # the exponential operand.
        from sage.all import log
        # full_integrand = propagator_rational × exp(exponent)
        # So log(full_integrand) = log(propagator_rational) + exponent
        # d/d(omega_var) [log(full_integrand)] = d/d(omega_var)[log(R)] + d(exponent)/d(omega_var)
        # At omega_var = 0 (if safe), the rational derivative is finite.
        # The exponent derivative is i*τ (constant in omega_var).

        # Simpler: since the exponent is linear in omega_var,
        # d(exponent)/d(omega_var) = coefficient of omega_var in exponent.
        # We can get this by: d(full_integrand)/d(omega_var) / full_integrand
        # minus d(log(R))/d(omega_var).

        # Simplest: just try all time combinations.
        # For k=2 stationary, τ = ±(t₂ - t₁).
        # Determine the sign from the direction of propagation.

        # PRAGMATIC approach: compute the exponent directly.
        # The full_integrand = integrand_resolved * exp_factor
        # where exp_factor = product of e^{±iω_j t_j}
        # Extract by dividing: exp_factor = full_integrand / integrand_resolved
        integrand_resolved = integrand_result['integrand']
        exp_factor = (full_int / integrand_resolved).simplify_rational()
        # exp_factor should be a pure exponential
        # d(log(exp_factor))/d(omega_var) = i*τ
        log_exp = log(exp_factor)
        d_log = log_exp.diff(omega_var).simplify()
        # d_log = i*τ → τ = d_log / i = -i * d_log
        tau = (-I * d_log).simplify()
        return tau
    except Exception:
        return None


# ═══════════════════════════════════════════════════════════════════════════
# Step 7b: Convenience wrappers (backward-compatible)
# ═══════════════════════════════════════════════════════════════════════════

def integrate_tree_level(integrand_result):
    r"""
    Evaluate a tree-level diagram (ℓ = 0) in the time domain.

    For a tree diagram there are no loop integrals, but there are
    still k−1 external frequency integrals with e^{−iωt} factors.
    The result is C(t₁,...,tₖ), a function of external times.

    Parameters
    ----------
    integrand_result : dict
        Output from build_integrand_stationary.

    Returns
    -------
    SR expression
        C(t₁,...,tₖ) — the time-domain contribution.
    """
    if integrand_result['loop_number'] != 0:
        raise ValueError(
            f"Expected tree-level (ℓ=0), got ℓ={integrand_result['loop_number']}"
        )
    result = integrate_to_time_domain(integrand_result)
    return result['time_domain_result']


def integrate_one_loop_residues(integrand_result, pole_vals=None,
                                 omega_symbol=None, close_upper=True):
    r"""
    Evaluate a one-loop diagram (ℓ = 1) in the time domain.

    Performs both the loop integral and external frequency integrals
    via residues. The result is C(t₁,...,tₖ).

    Parameters
    ----------
    integrand_result : dict
        Output from build_integrand_stationary.
    pole_vals : list or None
        (Unused — poles are found automatically from the integrand.)
    omega_symbol : SR variable or None
        (Unused — kept for API compatibility.)
    close_upper : bool
        (Unused — closure direction determined by time arguments.)

    Returns
    -------
    SR expression
        C(t₁,...,tₖ) — the time-domain contribution.
    """
    if integrand_result['loop_number'] != 1:
        raise ValueError(
            f"Expected one-loop (ℓ=1), got ℓ={integrand_result['loop_number']}"
        )
    result = integrate_to_time_domain(integrand_result)
    return result['time_domain_result']


# ═══════════════════════════════════════════════════════════════════════════
# Step 8: Compute full correction at a given loop level
# ═══════════════════════════════════════════════════════════════════════════

def compute_correction(typed_diagrams, propagator_data, k,
                        omega_symbol=None, pole_vals=None,
                        time_dep_params=None, noise_structure=None):
    r"""
    Sum contributions from all diagrams.

    For each unique typed diagram Γ, compute the time-domain cumulant
    contribution C_Γ(t₁,...,tₖ) and sum them.

    Parameters
    ----------
    typed_diagrams : list of TypedDiagram
        Should be deduplicated (one per unique diagram).
    propagator_data : dict
    k : int
        Number of external legs.
    omega_symbol : SR variable or None
    pole_vals : list or None
        (Kept for API compatibility; poles are found automatically.)
    time_dep_params : list or None
    noise_structure : dict or None

    Returns
    -------
    results : list of dict
        Per-diagram results, each containing:
            'diagram': the TypedDiagram
            'integrand_result': full output from build_integrand_stationary
            'contribution': the time-domain contribution (SR expression)
            'status': 'ok', 'partial', or 'needs_numerical'
    total : SR expression
        Sum of all contributions (as a function of external times).
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
