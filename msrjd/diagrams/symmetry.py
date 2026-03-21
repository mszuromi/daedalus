"""
msrjd.diagrams.symmetry
========================
Combinatorial factor M(Γ) for fully-typed labeled diagrams, and
deduplication of typed diagrams into unique representatives.

Definition (Attachment)
-----------------------
Given a typed diagram skeleton Γ with directed graph D = (V, E),
vertex leg multisets L_v^out (response) and L_v^in (physical), and
a fixed propagator type on each edge, an *attachment* is a collection
of bijections {f_v^out, f_v^in}_{v in V} such that:

  1. f_v^out : L_v^out → {outgoing edges of v}  is a bijection.
  2. f_v^in  : L_v^in  → {incoming edges of v}  is a bijection.
  3. Each edge e = (u → v) is matched by exactly one response leg
     at u and one physical leg at v.
  4. The resulting (response_leg, physical_leg) pair on each edge
     corresponds to a nonzero propagator component.
  5. Edges to/from external vertices match the assigned external fields.

The combinatorial factor is:

    M(Γ) = number of valid attachments.

Since identical legs (same field type) map to the same propagator
row or column, swapping identical legs among their edges never changes
the propagator assignment.  Therefore:

    M(Γ) = ∏_v  ∏_{groups of k identical response legs at v}  k!
              ×  ∏_{groups of k identical physical legs at v}  k!

The diagram's contribution to the k-point function is:

    weight(Γ) = M(Γ) × ∏_v coeff(v) × ∫(propagators)

where the vertex coefficients already contain 1/n! from the Taylor
expansion of the action.

Reference: Helias & Dahmen, "Statistical Field Theory for Neural
Networks", Ch. 9 (Springer, 2020).

Build Phase G.
"""

from collections import Counter
from functools import reduce
from math import factorial
from operator import mul

from sage.all import SR


# ── Combinatorial factor ────────────────────────────────────────────────────

def _vertex_attachment_count(vertex_type):
    """
    Count the number of valid leg-to-edge bijections at a single vertex.

    For each group of k identical response legs, any permutation among
    their k outgoing edges yields the same propagator indices → k! ways.
    Same for physical legs.

    Parameters
    ----------
    vertex_type : VertexType or SourceType

    Returns
    -------
    int
        Number of distinct attachments at this vertex (always >= 1).
    """
    resp_legs = vertex_type.response_legs
    has_phys = hasattr(vertex_type, 'physical_legs')
    phys_legs = vertex_type.physical_legs if has_phys else []

    m = 1
    for count in Counter(resp_legs).values():
        m *= factorial(count)
    for count in Counter(phys_legs).values():
        m *= factorial(count)
    return m


def combinatorial_factor(typed_diagram):
    r"""
    Compute M(Γ) — the number of distinct valid attachments (leg-to-edge
    bijections) that realize the typed diagram Γ.

    M(\Gamma) = \prod_{v} \prod_{\text{groups of } k \text{ identical legs}} k!

    This factor **multiplies** the diagram's contribution:

        weight = M(Γ) × ∏(vertex coefficients) × ∫(propagators)

    Parameters
    ----------
    typed_diagram : TypedDiagram

    Returns
    -------
    int
        The combinatorial factor (always >= 1).
    """
    m = 1
    for v, vtype in typed_diagram.vertex_assignments.items():
        m *= _vertex_attachment_count(vtype)
    return m


def compute_all_combinatorial_factors(typed_diagrams):
    """
    Compute M(Γ) for each typed diagram.

    Parameters
    ----------
    typed_diagrams : list of TypedDiagram

    Returns
    -------
    list of int
        Combinatorial factor for each diagram, same order as input.
    """
    return [combinatorial_factor(td) for td in typed_diagrams]


# ── Deduplication ───────────────────────────────────────────────────────────

def diagram_signature(td):
    """
    Build a hashable canonical signature for a typed diagram.

    Two typed diagrams with the same signature are identical — they
    represent the same Feynman diagram Γ and differ only in the
    internal choice of which identical leg was assigned to which edge
    (an attachment degree of freedom).

    The signature encodes:
      - External leg assignments  (which field at each leaf)
      - Vertex type at each internal vertex  (coefficient, legs, bigrade)
      - Propagator indices on every edge

    Parameters
    ----------
    td : TypedDiagram

    Returns
    -------
    tuple
        Hashable canonical signature.
    """
    # External legs: sorted (leaf, field) pairs
    ext = tuple(sorted(td.external_legs.items()))

    # Vertex assignments: sorted (vertex, type_key) pairs
    verts = []
    for v, vtype in sorted(td.vertex_assignments.items()):
        tname = type(vtype).__name__
        resp = tuple(vtype.response_legs)
        phys = tuple(vtype.physical_legs) if hasattr(vtype, 'physical_legs') else ()
        verts.append((v, tname, str(vtype.coefficient), vtype.bigrade, resp, phys))
    verts = tuple(verts)

    # Edge propagator assignments: sorted (edge, prop_indices) pairs
    edges = tuple(sorted(
        ((u, v), td.propagator_indices[(u, v)])
        for (u, v) in td.edge_types
    ))

    return (ext, verts, edges)


def deduplicate_typed_diagrams(typed_diagrams):
    """
    Remove duplicate typed diagrams, keeping one representative per
    unique diagram Γ.

    Two TypedDiagrams are duplicates if they have identical external
    leg assignments, vertex type assignments, and propagator indices
    on every edge — i.e. they differ only in the internal leg-to-edge
    bijection (attachment).

    Parameters
    ----------
    typed_diagrams : list of TypedDiagram

    Returns
    -------
    unique : list of TypedDiagram
        One representative per unique diagram.
    """
    seen = set()
    unique = []
    for td in typed_diagrams:
        sig = diagram_signature(td)
        if sig not in seen:
            seen.add(sig)
            unique.append(td)
    return unique


# ── Coefficient classification ──────────────────────────────────────────────

def _symbols_matching_prefixes(expr, prefixes):
    """
    Return the set of free SR variables in *expr* whose string name
    starts with any of the given prefixes.

    >>> _symbols_matching_prefixes(SR('nstar1 * phi1_1'), ['nstar'])
    {nstar1}
    """
    if not prefixes:
        return set()
    matches = set()
    for sym in expr.variables():
        name = str(sym)
        if any(name.startswith(p) for p in prefixes):
            matches.add(sym)
    return matches


def _is_source_type(vtype):
    """Check if a vertex type is a SourceType (has no physical_legs)."""
    return not hasattr(vtype, 'physical_legs')


def classify_coefficient_factors(typed_diagram, time_dep_params=None,
                                 noise_structure=None):
    r"""
    Partition each vertex coefficient into constant and time-dependent
    symbolic factors, respecting the different time structures of
    interaction vertices vs source vertices.

    **Interaction vertices** are local in time: all legs share one time
    variable $t_v$.  If any symbol in the coefficient matches
    ``time_dep_params``, it becomes $c_v(t_v)$ inside the integral.

    **Source vertices** represent multi-point cumulant densities: each
    outgoing leg carries its **own** time variable.  A source with $k$
    legs contributes $\kappa(t_1, \ldots, t_k)$ to the integrand.  The
    temporal structure (white, colored, general) determines how the
    amplitude and the time dependence factor:

    - ``'white'``:  $\kappa(t_1, t_2) = c \cdot \delta(t_1 - t_2)$.
      The $\delta$ collapses the two leg-times; in frequency domain the
      source contributes no $\omega$ dependence.
    - ``'colored'``:  $\kappa(t_1, t_2) = C(t_1 - t_2)$ (stationary
      but not delta-correlated).  The kernel Fourier transform enters
      the frequency integral.
    - ``'general'``:  $\kappa(t_1, t_2)$ with no simplification — both
      leg-times are independent integration variables.

    Parameters
    ----------
    typed_diagram : TypedDiagram
    time_dep_params : list of str or None
        Parameter name prefixes that are time-dependent at interaction
        vertices (e.g. ``['nstar', 'phi1', 'phi2']``).
        If None or empty, all interaction coefficients are constant.
    noise_structure : dict or None
        Noise temporal structure from the model dict.  Expected keys:
            ``'temporal_type'``: ``'white'``, ``'colored'``, or ``'general'``
            ``'amplitude_params'``: list of parameter prefixes in the
                noise amplitude that may be time-dependent
        If None, defaults to ``{'temporal_type': 'white',
        'amplitude_params': []}``.

    Returns
    -------
    dict with keys:
        ``'M'`` : int
            Combinatorial factor M(Γ).
        ``'scalar_prefactor'`` : SR expression
            Product of all constant (time-independent) factors across
            all vertices.  In the fully stationary case this equals
            M(Γ) × ∏_v coeff(v).
        ``'vertex_time_factors'`` : dict
            ``{vertex_id: SR expression}`` for each **interaction**
            vertex whose coefficient contains time-dependent symbols.
            Each factor depends on a single vertex time $t_v$.
            Empty dict in the stationary case.
        ``'source_time_info'`` : dict
            ``{vertex_id: info_dict}`` for each **source** vertex.
            Each ``info_dict`` has keys:
                ``'n_legs'``: int — number of outgoing legs (= number of
                    independent time variables, before any δ collapse)
                ``'temporal_type'``: str — ``'white'``, ``'colored'``,
                    or ``'general'``
                ``'amplitude'``: SR expression — the amplitude factor
                    (may be time-dependent or constant)
                ``'amplitude_is_time_dep'``: bool — whether amplitude
                    depends on leg times
            Empty dict when no source vertices are present.
        ``'is_stationary'`` : bool
            True when no vertex has time-dependent symbols AND
            noise is white with constant amplitude.
    """
    prefixes = list(time_dep_params or [])
    ns = noise_structure or {'temporal_type': 'white', 'amplitude_params': []}
    noise_type = ns.get('temporal_type', 'white')
    noise_amp_prefixes = list(ns.get('amplitude_params', []))

    M = combinatorial_factor(typed_diagram)

    scalar_parts = [SR(M)]
    vertex_time_factors = {}
    source_time_info = {}
    has_time_dep = False

    for v, vtype in typed_diagram.vertex_assignments.items():
        coeff = SR(vtype.coefficient)

        if _is_source_type(vtype):
            # ── Source vertex: per-leg time structure ──
            # Each outgoing leg gets its own time variable (unlike
            # interaction vertices which share one vertex time).
            #
            # The amplitude is time-dependent only if the noise
            # amplitude symbols are ALSO declared time-dependent
            # in the model's time_dep_params list.
            n_legs = len(vtype.response_legs)
            # Symbols in coeff that are noise amplitudes
            amp_syms = _symbols_matching_prefixes(coeff, noise_amp_prefixes)
            # Of those, which are also declared time-dependent?
            amp_td_syms = _symbols_matching_prefixes(coeff,
                [p for p in noise_amp_prefixes if p in prefixes])
            amp_is_td = len(amp_td_syms) > 0

            if not amp_is_td and noise_type == 'white':
                # Fully stationary white noise: amplitude is constant,
                # δ(t₁-t₂) collapses times → no time dependence
                scalar_parts.append(coeff)
            else:
                # Factor out constant part of amplitude
                const_part = coeff.subs({s: SR(1) for s in amp_td_syms}) if amp_td_syms else coeff
                td_part = coeff / const_part if (amp_td_syms and not const_part.is_zero()) else coeff

                if amp_td_syms and not const_part.is_one() and not const_part.is_zero():
                    scalar_parts.append(const_part)
                elif not amp_td_syms:
                    scalar_parts.append(coeff)
                    td_part = SR(1)

                has_time_dep = True

            source_time_info[v] = {
                'n_legs': n_legs,
                'temporal_type': noise_type,
                'amplitude': coeff,
                'amplitude_is_time_dep': amp_is_td,
            }

        else:
            # ── Interaction vertex: single vertex time ──
            td_syms = _symbols_matching_prefixes(coeff, prefixes)

            if not td_syms:
                scalar_parts.append(coeff)
            else:
                const_part = coeff.subs({s: SR(1) for s in td_syms})
                td_part = coeff / const_part if not const_part.is_zero() else coeff

                if not const_part.is_one() and not const_part.is_zero():
                    scalar_parts.append(const_part)

                vertex_time_factors[v] = td_part.simplify_rational()
                has_time_dep = True

    scalar_prefactor = reduce(mul, scalar_parts, SR(1))

    # Stationary = no time-dep interaction factors AND noise is white
    # with constant amplitude
    is_stationary = (
        len(vertex_time_factors) == 0
        and all(not info['amplitude_is_time_dep'] for info in source_time_info.values())
        and all(info['temporal_type'] == 'white' for info in source_time_info.values())
    )

    return {
        'M': M,
        'scalar_prefactor': scalar_prefactor,
        'vertex_time_factors': vertex_time_factors,
        'source_time_info': source_time_info,
        'is_stationary': is_stationary,
    }
