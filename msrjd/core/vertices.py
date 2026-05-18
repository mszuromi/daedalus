"""
msrjd.core.vertices
====================
Decompose bigrade polynomial sectors into individual typed monomials
with field-leg metadata (VertexType, SourceType data structures).

Each monomial from the interacting action becomes a VertexType; each
monomial from the noise kernel becomes a SourceType.  These are the
atomic building blocks used by the type-assignment engine (Phase E).

Build Phase B.
"""

from sage.all import SR


# ── Data structures ──────────────────────────────────────────────────────────

class VertexType:
    """
    One monomial from an interacting-action sector (total degree >= 3).

    Attributes
    ----------
    coefficient : SR expression
        Coupling constant * combinatorial prefactor (the SR coefficient
        from the polynomial ring).
    response_legs : list of (str, int)
        Each entry is (field_base_name, population_index).  Repeated if
        the monomial has exponent > 1 in that generator.
    physical_legs : list of (str, int)
        Same format as response_legs, for physical field generators.
    bigrade : (int, int)
        (n_tilde, n_phys).
    """

    __slots__ = ('coefficient', 'response_legs', 'physical_legs', 'bigrade')

    def __init__(self, coefficient, response_legs, physical_legs, bigrade):
        self.coefficient   = coefficient
        self.response_legs = list(response_legs)
        self.physical_legs = list(physical_legs)
        self.bigrade       = tuple(bigrade)

    # Pickle support for __slots__
    def __getstate__(self):
        return {s: getattr(self, s) for s in self.__slots__}

    def __setstate__(self, state):
        for s, v in state.items():
            object.__setattr__(self, s, v)

    @property
    def in_degree(self):
        """Number of physical (incoming) legs."""
        return len(self.physical_legs)

    @property
    def out_degree(self):
        """Number of response (outgoing) legs."""
        return len(self.response_legs)

    @property
    def total_degree(self):
        return len(self.response_legs) + len(self.physical_legs)

    def __repr__(self):
        return (f'VertexType(bigrade={self.bigrade}, '
                f'resp={self.response_legs}, phys={self.physical_legs}, '
                f'coeff={self.coefficient})')


class ConvVertexType(VertexType):
    """
    Interaction vertex (n_phys ≥ 1) where one or more PHYSICAL legs
    sit at independent times linked to the vertex's main time by a
    synaptic kernel ``g(τ)`` — i.e. a conductance-style term in the
    original action.

    Parallels :class:`NoiseSourceType` (which does the same for the
    response legs of a noise source via the cumulant kernel
    ``κ(τ)``).  The Phase J integrator treats both by allocating one
    extra τ integration variable per kernel attachment and inserting
    the kernel factor into the integrand.

    Origin
    ------
    Produced by :func:`extract_vertex_types` when a vertex
    coefficient contains kernel SR symbols recorded by
    :func:`msrjd.core.convolution.reduce_conv_in_action`'s
    ``attachments_out`` mechanism.  The kernel symbol's attached
    field is resolved against the vertex's actual physical-leg list
    to identify which leg sits at ``anchor_time − τ``.

    Attributes
    ----------
    kernel_attachments : list of dict
        One entry per kernel SR symbol present in ``coefficient``.
        Each dict has:

          * ``'symbol'``         — the kernel SR symbol (e.g.
                                   ``z_g_1_2``); same symbol the
                                   coefficient still references.
          * ``'leg'``            — the physical leg the kernel
                                   attaches to, as a leg-tuple
                                   ``(base_name, pop_idx_1based)``
                                   matching entries in
                                   ``physical_legs``.
          * ``'leg_index'``      — int, position of that leg within
                                   ``physical_legs`` (0-based).  When
                                   the same leg-tuple appears more
                                   than once in ``physical_legs``
                                   (kernel coupling to a duplicated
                                   leg) this is the first matching
                                   index; treat as canonical.
          * ``'kernel_td_fn'``   — callable ``(tau) -> SR``; the
                                   time-domain kernel ``g(τ)``
                                   evaluated at the per-vertex τ
                                   integration symbol.  Pre-bound
                                   to the kernel's specific indices
                                   from ``model['kernel_td_image']``.
    """

    __slots__ = ('kernel_attachments',)

    def __init__(self, coefficient, response_legs, physical_legs, bigrade,
                 kernel_attachments):
        super().__init__(coefficient, response_legs, physical_legs, bigrade)
        self.kernel_attachments = list(kernel_attachments)

    # Pickle support — extend VertexType's via __slots__ chain
    def __getstate__(self):
        state = super().__getstate__()
        state['kernel_attachments'] = self.kernel_attachments
        return state

    def __setstate__(self, state):
        super().__setstate__(
            {s: state[s] for s in VertexType.__slots__ if s in state}
        )
        object.__setattr__(self, 'kernel_attachments',
                           state.get('kernel_attachments', []))

    def __repr__(self):
        att_summary = [
            f'{att["symbol"]}→{att["leg"]}'
            for att in self.kernel_attachments
        ]
        return (f'ConvVertexType(bigrade={self.bigrade}, '
                f'resp={self.response_legs}, phys={self.physical_legs}, '
                f'attach=[{", ".join(att_summary)}], '
                f'coeff={self.coefficient})')


class SourceType:
    """
    One monomial from a noise-kernel sector (n_tilde >= 2, n_phys = 0).

    Attributes
    ----------
    coefficient : SR expression
    response_legs : list of (str, int)
    bigrade : (int, int)
        (n_tilde, 0).
    """

    __slots__ = ('coefficient', 'response_legs', 'bigrade')

    def __init__(self, coefficient, response_legs, bigrade):
        self.coefficient   = coefficient
        self.response_legs = list(response_legs)
        self.bigrade       = tuple(bigrade)

    # Pickle support for __slots__
    def __getstate__(self):
        return {s: getattr(self, s) for s in self.__slots__}

    def __setstate__(self, state):
        for s, v in state.items():
            object.__setattr__(self, s, v)

    @property
    def out_degree(self):
        """Number of response (outgoing) legs."""
        return len(self.response_legs)

    def __repr__(self):
        return (f'SourceType(bigrade={self.bigrade}, '
                f'resp={self.response_legs}, coeff={self.coefficient})')


class NoiseSourceType(SourceType):
    """
    Source vertex backed by a non-local cumulant kernel.

    Same role as SourceType (n_tilde response legs, no physical legs)
    but the response legs sit at *independent* times — they are
    coupled by the cumulant kernel ``κ^{(n)}(τ_1, …, τ_{n-1})``.  This
    arises from the GTaS / correlated-input cumulant generating
    functional ``-W_m[mt]`` injected by ``FieldTheory.expand``.

    Locally-correlated (delta) cumulants stay as plain ``SourceType``
    (their τ-integral collapses inside ``_build_cumulant_action``);
    only the smooth, non-local part requires per-leg time treatment
    in the Phase J integrator.

    Attributes
    ----------
    cumulant_specs : list of dict
        One entry per ``z_kappa`` placeholder symbol that contributes
        to ``coefficient``.  Each dict has:

          * ``'symbol'``     — the SR placeholder (the same symbol the
                               coefficient still references; the
                               integrator substitutes it with 1 and
                               multiplies in the actual kernel_fn).
          * ``'kernel_fn'``  — callable ``(ns, i, j, ..., tau) -> SR``;
                               evaluated with the cumulant's relative-
                               time variable to produce the integrand
                               kernel factor.
          * ``'legs'``       — leg-index tuple, 0-based, matching the
                               kernel_fn's leg arguments (e.g.
                               ``(0, 1)`` for cross-cumulant 1↔2).
          * ``'tau_var'``    — the SR symbol used for the cumulant's
                               relative time when the kernel was
                               registered (for hashability of bounds).
          * ``'sign'``       — SR scalar pulled out of the source's
                               coefficient as the prefactor of
                               ``symbol`` (typically ``-1/2``).
          * ``'noise'``      — noise-process name (e.g. ``'X'``).
          * ``'order'``      — cumulant order (e.g. ``2``).
    """

    __slots__ = ('cumulant_specs',)

    def __init__(self, coefficient, response_legs, bigrade,
                 cumulant_specs):
        super().__init__(coefficient, response_legs, bigrade)
        self.cumulant_specs = list(cumulant_specs)

    # Pickle support — extend SourceType's via __slots__ chain
    def __getstate__(self):
        state = super().__getstate__()
        state['cumulant_specs'] = self.cumulant_specs
        return state

    def __setstate__(self, state):
        super().__setstate__(
            {s: state[s] for s in SourceType.__slots__ if s in state}
        )
        object.__setattr__(self, 'cumulant_specs',
                           state.get('cumulant_specs', []))

    def __repr__(self):
        legs = [s.get('legs') for s in self.cumulant_specs]
        return (f'NoiseSourceType(bigrade={self.bigrade}, '
                f'resp={self.response_legs}, '
                f'cumulant_legs={legs}, coeff={self.coefficient})')


# ── Ring variable name parsing ───────────────────────────────────────────────

def _parse_field_name(ring_var_name):
    """
    Parse a ring variable name like 'nt1', 'dn2', 'vt12' into
    (base_name, population_index).

    Convention: the name is a string of letters followed by digits.
    The digits are the 1-based population index.
    """
    # Find where digits start
    i = len(ring_var_name)
    while i > 0 and ring_var_name[i - 1].isdigit():
        i -= 1
    if i == len(ring_var_name) or i == 0:
        # No trailing digits or all digits — use full name, index 0
        return ring_var_name, 0
    base = ring_var_name[:i]
    idx  = int(ring_var_name[i:])
    return base, idx


# ── Decomposition ────────────────────────────────────────────────────────────

def decompose_sector(sector_poly, n_tilde, ring_var_names):
    """
    Decompose one bigrade sector polynomial into individual monomials.

    Parameters
    ----------
    sector_poly : PolynomialRing element
        One sector from FieldTheory.sectors(), e.g. the (2,1) sector.
    n_tilde : int
        Number of response-field generators (first n_tilde generators
        in the ring are response fields).
    ring_var_names : list of str
        Ring generator names in order, e.g. ['vt1','vt2','nt1','nt2','dv1','dv2','dn1','dn2'].

    Returns
    -------
    list of (VertexType or SourceType)
    """
    results = []

    for exp_vec, coeff in sector_poly.dict().items():
        resp_legs = []
        phys_legs = []

        for gen_idx, exponent in enumerate(exp_vec):
            if exponent == 0:
                continue
            name = ring_var_names[gen_idx]
            base, pop_idx = _parse_field_name(name)
            leg = (base, pop_idx)

            # Repeat for exponent multiplicity
            if gen_idx < n_tilde:
                resp_legs.extend([leg] * int(exponent))
            else:
                phys_legs.extend([leg] * int(exponent))

        n_t = len(resp_legs)
        n_p = len(phys_legs)
        bigrade = (n_t, n_p)

        if n_p == 0:
            results.append(SourceType(SR(coeff), resp_legs, bigrade))
        else:
            results.append(VertexType(SR(coeff), resp_legs, phys_legs, bigrade))

    return results


def _kernel_symbol_to_pop_indices(ksym):
    """Parse a kernel SR symbol like ``z_g_1_2`` into the (i, j)
    population indices (1-based, matching the matrix convention).

    Returns ``(i, j)`` for a matrix kernel ``z_g_i_j``, ``(i,)`` for a
    vector kernel ``z_g_i``, or ``()`` for a scalar kernel.  Returns
    ``None`` when the name doesn't match the kernel naming
    convention.

    Used by :func:`extract_vertex_types` to resolve which physical
    leg of a vertex a kernel symbol attaches to: the second index of
    a matrix kernel ``g_i_j`` matches the pop-idx of the field that
    appeared inside the original ``Conv(g[i,j], n[j])``.
    """
    name = str(ksym)
    # Trailing digits parsed greedily as a sequence of ``_<int>``.
    parts = name.split('_')
    indices = []
    for tok in reversed(parts):
        if tok.isdigit():
            indices.append(int(tok))
        else:
            break
    indices.reverse()
    if not indices:
        return ()
    return tuple(indices)


def _resolve_kernel_attachment_to_leg(ksym, attached_fluct_set,
                                       physical_legs, ring_var_names,
                                       n_tilde):
    """Pick the physical leg a kernel symbol attaches to.

    Strategy:
      1. The attachments dict maps ``ksym → {fluct_vars}`` (the SR
         symbols that appeared as ``Conv(g, fluct)``'s second arg in
         the original action).  We translate each fluct SR var into a
         leg-tuple ``(base_name, pop_idx)`` and intersect with the
         vertex's actual ``physical_legs``.
      2. Failing intersection (e.g. the attached fluct vars are at
         different pop-idx than any of this vertex's legs), fall back
         to **index matching**: match the kernel's trailing pop-index
         against the leg's pop-idx.  E.g. ``z_g_1_2`` matches a leg
         with pop-idx 2.  This handles the common case where the
         attachments dict has the kernel paired with a generic field
         name but the bigrade-classified monomial's leg list is
         specialised by pop-idx.

    Returns ``(leg_tuple, leg_index_in_physical_legs)`` or ``None``
    when no resolution is possible.
    """
    if not physical_legs:
        return None

    # Build a map from SR var name → leg-tuple by parsing the var name.
    fluct_leg_candidates = set()
    for fv in attached_fluct_set:
        base, idx = _parse_field_name(str(fv))
        fluct_leg_candidates.add((base, idx))

    # Direct intersection with physical_legs.
    for i, leg in enumerate(physical_legs):
        if leg in fluct_leg_candidates:
            return leg, i

    # Fall back to kernel-index matching.  For a matrix kernel
    # ``z_g_i_j``, the SECOND index (j) is the "incoming" leg's
    # pop-idx — the field that was inside the Conv.  Vector kernels
    # use their single index.
    kernel_idx = _kernel_symbol_to_pop_indices(ksym)
    if kernel_idx:
        target_pop = kernel_idx[-1]  # last index = "incoming" leg
        for i, leg in enumerate(physical_legs):
            if leg[1] == target_pop:
                return leg, i

    return None


def _flatten_kernel_symbols(ns, model):
    """Walk ``model['kernels']`` and return the flat list of every
    kernel SR symbol registered in the namespace.  Used by vertex
    extraction to scope the kernel-detection scan."""
    out = []
    for spec in model.get('kernels', []):
        kname = spec['name']
        ksym_obj = getattr(ns, kname, None)
        if ksym_obj is None:
            continue
        # Scalar (single SR var), vector (list), or matrix (list-of-lists)
        if hasattr(ksym_obj, '__iter__') and not isinstance(ksym_obj, str):
            for row in ksym_obj:
                if hasattr(row, '__iter__') and not isinstance(row, str):
                    out.extend(row)
                else:
                    out.append(row)
        else:
            out.append(ksym_obj)
    return out


def extract_vertex_types(ft):
    """
    Extract all VertexType objects from a FieldTheory's interacting action.

    Returns plain :class:`VertexType` for local interaction vertices.
    For conductance-style vertices — those whose coefficient still
    contains a kernel SR symbol because the original action had a
    ``Conv(g, X)`` factor that survived bigrade classification — the
    returned record is a :class:`ConvVertexType` that carries the
    kernel-attachment metadata the time-domain Phase J integrator
    consumes (per-leg τ allocation and ``g(τ)`` factor insertion).

    Parameters
    ----------
    ft : FieldTheory
        Must have been expanded (ft.expand() called).  When the
        action used the ``Conv(...)`` operator, ``ft._ns`` carries the
        ``_kernel_attachments`` dict populated by
        ``reduce_conv_in_action``; this drives the upgrade decision.

    Returns
    -------
    list of VertexType (with ConvVertexType instances mixed in for
    conductance vertices)
    """
    from msrjd.core.convolution import kernel_attachments_in_coefficient

    ft._require_expanded()
    ns = ft._ns
    model = ft.model

    attachments = getattr(ns, '_kernel_attachments', None) or {}
    kernel_symbols = (_flatten_kernel_symbols(ns, model)
                      if attachments else [])
    kernel_td_image_lambda = model.get('kernel_td_image')

    # Pre-build the {ksym: SR_in_tau} dict once per call so each
    # ConvVertexType can reference its bound time-domain expression
    # rather than re-evaluating the lambda per attachment.  The τ
    # symbol used here is a placeholder; the integrator substitutes
    # its actual per-vertex τ at integration time.
    if kernel_td_image_lambda is not None and attachments:
        from sage.all import SR as _SR
        _tau_placeholder = _SR.var('_conv_tau_placeholder')
        kernel_td_map = kernel_td_image_lambda(ns, _tau_placeholder)
    else:
        _tau_placeholder = None
        kernel_td_map = {}

    vtypes = []
    for (n_t, n_p), poly in ft.vertices().items():
        # vertices() returns sectors with total degree >= 3
        # Some may be pure noise-kernel (n_p == 0) — skip those
        if n_p == 0:
            continue
        monomials = decompose_sector(
            poly, ft._n_tilde, list(ns._ring_var_names)
        )
        for m in monomials:
            if not isinstance(m, VertexType):
                continue
            if not attachments:
                vtypes.append(m)
                continue

            # Scan the coefficient for kernel symbols recorded by
            # the Conv reducer.
            detected = kernel_attachments_in_coefficient(
                m.coefficient, attachments, kernel_symbols
            )
            if not detected:
                vtypes.append(m)
                continue

            # Resolve each detected kernel symbol to a specific
            # physical leg of THIS vertex.  The attachments dict can
            # only narrow it to a set of candidate fields; the
            # vertex's leg-tuples disambiguate by pop-idx.
            ring_var_names = list(ns._ring_var_names)
            kernel_attachments_for_vertex = []
            for ksym, leg_info in detected.items():
                if isinstance(leg_info, frozenset):
                    attached_set = set(leg_info)
                else:
                    attached_set = {leg_info}
                resolved = _resolve_kernel_attachment_to_leg(
                    ksym, attached_set, m.physical_legs,
                    ring_var_names, ft._n_tilde,
                )
                if resolved is None:
                    # Unresolvable — leave the kernel symbol in the
                    # coefficient unhandled.  Fall back to plain
                    # VertexType so existing-pipeline diagnostics flag
                    # the surviving symbol rather than silently
                    # treating it as zero.
                    continue
                leg, leg_idx = resolved
                if _tau_placeholder is not None and ksym in kernel_td_map:
                    # Bind the placeholder τ to a per-call closure so
                    # callers can re-substitute their own τ symbol.
                    td_expr = kernel_td_map[ksym]
                    def _td_fn(tau, _td_expr=td_expr,
                               _placeholder=_tau_placeholder):
                        return SR(_td_expr).subs({_placeholder: tau})
                else:
                    # No time-domain image available (kernel declared
                    # via freq_image only).  Caller will see this as
                    # None and either invert-FT or error out.
                    _td_fn = None
                kernel_attachments_for_vertex.append({
                    'symbol':       ksym,
                    'leg':          leg,
                    'leg_index':    leg_idx,
                    'kernel_td_fn': _td_fn,
                })

            if kernel_attachments_for_vertex:
                vtypes.append(ConvVertexType(
                    coefficient        = m.coefficient,
                    response_legs      = m.response_legs,
                    physical_legs      = m.physical_legs,
                    bigrade            = m.bigrade,
                    kernel_attachments = kernel_attachments_for_vertex,
                ))
            else:
                vtypes.append(m)
    return vtypes


def extract_source_types(ft):
    """
    Extract all SourceType objects from a FieldTheory's noise kernel.

    Sources whose coefficient carries a non-local cumulant placeholder
    symbol ``z_kappa_<noise>_<order>_<i>_<j>`` (registered on
    ``ft._ns._cumulant_kernels`` by ``_build_cumulant_action``) are
    upgraded to ``NoiseSourceType`` records carrying the kernel
    function and leg metadata, so the Phase J integrator can treat
    them with per-leg times and a kernel factor.

    Local (delta-collapsed) auto-cumulants stay as plain ``SourceType`` —
    their kernel placeholder was already eliminated at extraction time
    in ``_build_cumulant_action``.

    Parameters
    ----------
    ft : FieldTheory
        Must have been expanded (ft.expand() called).

    Returns
    -------
    list of SourceType (with NoiseSourceType subclass instances mixed
    in for non-local sources)
    """
    ft._require_expanded()
    stypes = []

    ns = ft._ns
    cumulant_kernels = getattr(ns, '_cumulant_kernels', {}) or {}
    correlated_noises = ft.model.get('correlated_noises', {}) or {}
    # Look up the response-field name for each registered (noise, order)
    # so we can match leg-tuples in the source's response_legs.
    noise_resp_field = {
        noise_name: spec['response_field']
        for noise_name, spec in correlated_noises.items()
    }

    for (n_t, n_p), poly in ft.noise_kernel().items():
        monomials = decompose_sector(poly, ft._n_tilde, list(ft._ns._ring_var_names))
        for m in monomials:
            if not isinstance(m, SourceType):
                continue

            # Match the response-leg multiset against every registered
            # cumulant spec.  Specs whose (noise, order, legs) leg-tuple
            # produces the SAME leg multiset as ``m.response_legs`` AND
            # whose placeholder symbol still appears in ``m.coefficient``
            # contribute to this source.
            matched_specs = []
            m_leg_multiset = sorted(m.response_legs)
            for (noise_name, order, leg_tuple), spec in cumulant_kernels.items():
                resp_field_name = noise_resp_field.get(noise_name)
                if resp_field_name is None:
                    continue
                # 0-based legs → 1-based pop_idx leg-tuple to compare
                spec_legs = sorted(
                    [(resp_field_name, k + 1) for k in leg_tuple]
                )
                if spec_legs != m_leg_multiset:
                    continue
                # Verify the placeholder symbol is actually in coeff
                sym = spec['symbol']
                try:
                    sign = SR(m.coefficient).coefficient(sym)
                except (ValueError, TypeError):
                    sign = SR(0)
                if sign == 0:
                    continue
                # Bind `ns` into the kernel-fn closure so the
                # downstream Phase J integrator can call
                # ``kernel_fn(i, j, tau)`` without needing the
                # FieldTheory namespace.  The user's lambda has
                # signature ``(ns, i, j, tau) -> SR``.
                _user_kf = spec['kernel_fn']
                bound_kernel = (
                    lambda *args, _kf=_user_kf, _ns=ns: _kf(_ns, *args)
                )
                matched_specs.append({
                    'symbol':    sym,
                    'kernel_fn': bound_kernel,
                    'legs':      spec['legs'],
                    'tau_var':   spec['tau_var'],
                    'sign':      sign,
                    'noise':     noise_name,
                    'order':     order,
                })

            if matched_specs:
                stypes.append(NoiseSourceType(
                    coefficient   = m.coefficient,
                    response_legs = m.response_legs,
                    bigrade       = m.bigrade,
                    cumulant_specs= matched_specs,
                ))
            else:
                stypes.append(m)
    return stypes


def available_degrees(vertex_types, source_types):
    """
    Compute the sets of available degree signatures.

    Parameters
    ----------
    vertex_types : list of VertexType
    source_types : list of SourceType

    Returns
    -------
    interaction_degrees : set of (int, int)
        Set of (in_degree, out_degree) pairs from vertex types.
    source_degrees : set of int
        Set of out_degree values from source types.
    """
    interaction_degrees = {(vt.in_degree, vt.out_degree) for vt in vertex_types}
    source_degrees      = {st.out_degree for st in source_types}
    return interaction_degrees, source_degrees
