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
    Partition each vertex coefficient into factors that can be pulled
    outside the integral vs factors that must stay inside, respecting
    the different time structures of interaction and source vertices.

    **Interaction vertices** are local in time: all legs share one time
    variable $t_v$.  The coefficient $c_v$ can be pulled out of the
    integral only if it contains no time-dependent symbols.  If any
    symbol matches ``time_dep_params``, it becomes $c_v(t_v)$ inside
    the integral.

    **Source vertices** represent multi-point cumulant densities:
    each outgoing leg carries its **own** time variable.  A source with
    $k$ legs contributes $\kappa(t_1, \ldots, t_k)$ to the integrand.

    A source coefficient can be pulled outside the integral **only** if
    ALL of the following hold:

    1. The noise is **white**: $\kappa \propto \delta(t_1 - t_2)$, so
       the $\delta$ collapses the leg-times and contributes no
       $\omega$ dependence in the frequency domain.
    2. The amplitude is a **constant** (not time-dependent).

    Otherwise — for colored noise ($\kappa(t_1 - t_2)$), general noise
    ($\kappa(t_1, t_2)$), or time-dependent amplitude — the source
    factor stays **inside** the integral.  Note that colored stationary
    noise is still Fourier-transformable ($\hat{\kappa}(\omega)$ enters
    the frequency integrand), but it is NOT a scalar that can be pulled
    out.

    **Stationarity** is a separate concept: a system is stationary when
    ALL time dependencies are through time *differences* only (making
    the problem Fourier-transformable).  This is true when:
    - No interaction vertex has time-dependent coefficients, AND
    - Noise is ``'white'`` or ``'colored'`` (both are stationary), AND
    - The noise amplitude is constant.

    A stationary system can still have source factors inside the
    integral (colored noise), but those factors depend only on
    $\omega$ in the frequency domain.

    Parameters
    ----------
    typed_diagram : TypedDiagram
    time_dep_params : list of str or None
        Parameter name prefixes that are time-dependent
        (e.g. ``['nstar', 'phi1', 'phi2']``).
        If None or empty, all coefficients are treated as constant.
    noise_structure : dict or None
        Noise temporal structure from the model dict.  Expected keys:
            ``'temporal_type'``: ``'white'``, ``'colored'``, or
                ``'general'``
            ``'amplitude_params'``: list of parameter prefixes that
                enter the noise amplitude (e.g. ``['nstar']``).
                These are only treated as time-dependent if they also
                appear in ``time_dep_params``.
        If None, defaults to white noise with no amplitude params.

    Returns
    -------
    dict with keys:
        ``'M'`` : int
            Combinatorial factor M(Γ).
        ``'scalar_prefactor'`` : SR expression
            M(Γ) × product of all factors that can be pulled outside
            the integral.  For interaction vertices, this includes
            coefficients with no time-dependent symbols.  For source
            vertices, this includes the amplitude only when the noise
            is white AND the amplitude is constant.
        ``'vertex_time_factors'`` : dict
            ``{vertex_id: SR expression}`` for each **interaction**
            vertex whose coefficient contains time-dependent symbols.
            Each factor depends on a single vertex time $t_v$.
        ``'source_time_info'`` : dict
            ``{vertex_id: info_dict}`` for each **source** vertex.
            Each ``info_dict`` has keys:
                ``'n_legs'``: int — number of outgoing legs (= number
                    of independent time variables before any δ collapse)
                ``'temporal_type'``: ``'white'``, ``'colored'``, or
                    ``'general'``
                ``'amplitude'``: SR expression — the full source coeff
                ``'amplitude_is_time_dep'``: bool — whether amplitude
                    symbols are in time_dep_params
                ``'in_integrand'``: bool — whether this source's
                    contribution stays inside the integral (True for
                    colored/general noise, or time-dep amplitude)
        ``'is_stationary'`` : bool
            True when all time dependencies are through time differences
            only — the system is Fourier-transformable.  This does NOT
            mean all factors can be pulled out (colored noise is
            stationary but stays in the integrand as $\hat{\kappa}(\omega)$).
    """
    prefixes = list(time_dep_params or [])
    ns = noise_structure or {'temporal_type': 'white', 'amplitude_params': []}
    noise_type = ns.get('temporal_type', 'white')
    noise_amp_prefixes = list(ns.get('amplitude_params', []))

    M = combinatorial_factor(typed_diagram)

    scalar_parts = [SR(M)]
    vertex_time_factors = {}
    source_time_info = {}

    for v, vtype in typed_diagram.vertex_assignments.items():
        coeff = SR(vtype.coefficient)

        if _is_source_type(vtype):
            # ── Source vertex: per-leg time structure ──
            n_legs = len(vtype.response_legs)

            # Check if amplitude symbols are declared time-dependent
            amp_td_syms = _symbols_matching_prefixes(coeff,
                [p for p in noise_amp_prefixes if p in prefixes])
            amp_is_td = len(amp_td_syms) > 0

            # Source coeff can be pulled out ONLY if white + constant amp
            can_pull_out = (noise_type == 'white' and not amp_is_td)

            if can_pull_out:
                scalar_parts.append(coeff)
            else:
                # Factor out constant part of amplitude
                if amp_td_syms:
                    const_part = coeff.subs({s: SR(1) for s in amp_td_syms})
                    if not const_part.is_one() and not const_part.is_zero():
                        scalar_parts.append(const_part)
                # Time-dep part and/or noise kernel stays in integrand

            source_time_info[v] = {
                'n_legs': n_legs,
                'temporal_type': noise_type,
                'amplitude': coeff,
                'amplitude_is_time_dep': amp_is_td,
                'in_integrand': not can_pull_out,
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

    scalar_prefactor = reduce(mul, scalar_parts, SR(1))

    # Stationary = Fourier-transformable.  Requires:
    #   1. No interaction vertex has time-dependent coefficients
    #   2. Noise is white or colored (both depend only on time differences)
    #   3. Noise amplitude is constant (not time-dependent)
    # Note: colored noise IS stationary — κ(t₁-t₂) → κ̂(ω) — but the
    # source factor still enters the integral as κ̂(ω).
    is_stationary = (
        len(vertex_time_factors) == 0
        and all(not info['amplitude_is_time_dep']
                for info in source_time_info.values())
        and all(info['temporal_type'] in ('white', 'colored')
                for info in source_time_info.values())
    )

    return {
        'M': M,
        'scalar_prefactor': scalar_prefactor,
        'vertex_time_factors': vertex_time_factors,
        'source_time_info': source_time_info,
        'is_stationary': is_stationary,
    }
