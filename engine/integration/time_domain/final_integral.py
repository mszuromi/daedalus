"""
engine.integration.time_domain.final_integral
=============================================
Vertex-time integration on a tree-level diagram via explicit numerical
quadrature, with proper handling of the δ(t) component of any
"instantaneous" propagator entry.

MVP scope
---------
Only tree-level (loop_number == 0) typed diagrams are handled. Tree-level
is the right proving ground for the Phase J evaluation layer because it
exercises:

- time-domain retarded propagator lookup per edge,
- polytope extraction from explicit Heaviside factors,
- **δ-edge subset enumeration** (δ(t) components of instantaneous
  couplings like `ñ × δn` in MSR-JD),
- vertex-time integration over the retarded polytope,
- global translation invariance / origin pinning,
- numerical dispatch,

**without** needing the kernel reduction / caching / contraction
machinery. Loop cases are deferred to Extension 1.

Integration strategy
--------------------
The retarded propagator for an edge `(u -> v)` with matrix index
`(phys=p, resp=r)` decomposes into two pieces:

    G_R[p, r](t_v - t_u)
      =  delta_coeff[p, r] · δ(t_v - t_u)
       + Θ(t_v - t_u) · smooth[p, r](t_v - t_u)

The δ piece encodes an instantaneous response (nonzero whenever the
frequency-domain entry has a nonzero `ω → ∞` limit, which is the case
for any instantaneous coupling in the MSR-JD action). The smooth piece
is the usual pole-residue sum.

For a tree diagram with `|E|` edges, the full integrand is a product
of `|E|` such factors. Expanding the product yields `2^|E|` terms; we
enumerate them by picking a subset `S ⊆ edges` to take in its δ form
and the complement in its smooth form:

  Σ_{S ⊆ edges}  ∫ ds_1 … ds_m
                 · (∏_{e ∈ S} delta_coeff[e] · δ(t_{v_e} - t_{u_e}))
                 · (∏_{e ∉ S} Θ(t_{v_e} - t_{u_e}) · smooth[e](t_{v_e} - t_{u_e}))
                 · combined_prefactor

For each subset:

1. **Delta-edge equations** `t_{v_e} − t_{u_e} = 0` for `e ∈ S` are
   solved to eliminate integration variables by substitution. In the
   MVP star-tree case (single source vertex, all edges connecting source
   to leaves), one δ-edge pins the source time to a specific leaf time;
   two or more δ-edges force equality among several leaf times, which
   is a shot-noise δ(τ=0) contribution and is **skipped** for the
   continuous-callable return type.

2. **Smooth edges** contribute both a `fast_callable`-compiled factor
   (JIT'd over `CDF`) and a linear retardation constraint `t_v > t_u`
   that's added to the polytope.

3. The per-subset contribution is then a numerical quadrature over the
   reduced set of integration variables with the reduced polytope.

4. All subset contributions are summed into the final callable.

The public entry point `integrate_tree_diagram` returns a Python
callable `contribution(*ext_time_values) -> complex`.

Numerical overflow mitigation
-----------------------------
Each per-subset smooth product is `.expand()`'d into a sum of
single-exponential terms before JIT-compilation, so
`scipy.integrate.quad` cannot overflow IEEE doubles when sampling at
large negative `s`. (See the 2026-04-08 overflow fix in the CHANGELOG.)
"""

import math
import functools as _functools
from collections import namedtuple
from dataclasses import dataclass
from fractions import Fraction as _Fraction

from sage.all import SR, fast_callable, CDF, solve as sage_solve
from sage.rings.complex_double import (
    ComplexDoubleElement as _ComplexDoubleElement)

from engine.integration.time_domain.propagator_td import (
    build_G_t_matrix,
    G_t_entry,
    G_t_delta_coeff,
)
from engine.core.vertices import NoiseSourceType, ConvVertexType


# ───────────────────────────────────────────────────────────────────────
# Canonical mode-sum representation of a propagator edge
# ───────────────────────────────────────────────────────────────────────
# Every retarded-propagator edge in a Feynman diagram decomposes as:
#
#     G_R[pi, ri](Δt)  =  delta_coeff · δ(Δt)
#                       + Θ(Δt) · Σ_α  C_α · exp(λ_α · Δt)
#
# where ``λ_α = i·p_α`` for our Fourier convention and ``C_α`` is the
# residue of the propagator matrix at pole ``p_α`` at position (pi, ri).
#
# The smooth-part data is currently re-extracted inside
# ``_build_fast_subset_evaluator`` on every (diagram, subset) call,
# even though it depends only on the (pi, ri) of the edge.  Lifting
# the extraction to a per-edge step at the top of
# ``integrate_diagram`` eliminates the redundant work and gives the
# downstream integrators a clean, JSON-able data structure to consume.
#
# Spatial-extension hook: in a future spatial model ``λ_α`` and
# ``C_α`` become callables of momentum ``k`` rather than complex
# scalars.  The integrator backends will then gain an outer loop over
# momentum, but the EdgeModeSum interface stays the same.

@dataclass(frozen=True)
class EdgeModeSum:
    """Numerical mode-sum representation of one propagator edge.

    Fields
    ------
    ri, pi : int
        Response-column and physical-row indices into the propagator
        matrix.  Convention: ``G[pi, ri] = ⟨φ_pi  ñ_ri⟩``.
    delta_coeff : complex
        Coefficient of the δ(Δt) component (= the ω → ∞ limit of
        ``G_FT[pi, ri]``, captured in ``propagator_data['D_delta']``).
    modes : tuple of (complex C_α, complex λ_α)
        Pole-residue pairs.  For our codebase's Fourier convention
        (e^{-iωt}, retarded poles in Im(ω) > 0), ``λ_α = i · p_α``
        where ``p_α ∈ propagator_data['pole_vals']`` and
        ``C_α = C_mats[α][pi, ri]``.  The smooth-time-domain
        propagator is ``Σ_α C_α · exp(λ_α · Δt)``.
    dt_c0, dt_int_pairs, dt_ext_pairs :
        Sparse linear form for ``Δt`` in terms of (integration vars,
        free external times).  Filled in at the per-subset stage —
        leave empty in the per-edge build, populate when the subset
        is known.  Kept here so the same EdgeModeSum carries through
        the whole evaluation chain.
    """
    ri: int
    pi: int
    delta_coeff: complex
    modes: tuple                 # tuple[tuple[complex, complex], ...]
    dt_c0: float = 0.0
    dt_int_pairs: tuple = ()     # tuple[tuple[int, float], ...]
    dt_ext_pairs: tuple = ()     # tuple[tuple[int, float], ...]


def _build_edge_mode_sums(edge_info, propagator_data, prop_td=None):
    """Build one EdgeModeSum per entry of ``edge_info`` by extracting
    the per-pole residue from ``propagator_data['C_mats']`` ONCE per
    edge.

    The ``dt_*`` fields are NOT populated here — those depend on
    which integration variables survive δ-elimination in each subset
    and are filled in at the subset level (see
    ``_attach_subset_dt`` below).

    ``prop_td`` (M5 L2, a :class:`PropagatorTD` of THIS propagator data):
    the poles and each ``(pi, ri)`` entry's ``(residue, λ)`` tuple come from
    its tables, built once per ``compute_correction_td`` call with the very
    same conversions below, instead of once per diagram.

    Returns a list parallel to ``edge_info`` (same length, same
    order).  If the propagator data is incomplete (missing ``pole_vals``
    or ``C_mats``), returns ``None`` to signal that the caller should
    fall back to the SR-symbolic path.
    """
    pole_vals = propagator_data.get('pole_vals')
    C_mats = propagator_data.get('C_mats')
    if pole_vals is None or C_mats is None:
        return None

    if prop_td is not None:
        edge_mode_sums = []
        for ei in edge_info:
            ri, pi = ei['ri'], ei['pi']
            modes = prop_td.entry_modes(pi, ri)
            if modes is None:
                return None
            try:
                d_c = complex(ei['delta_coeff'])
            except Exception:
                try:
                    d_c = complex(CDF(SR(ei['delta_coeff'])))
                except Exception:
                    return None
            edge_mode_sums.append(EdgeModeSum(
                ri=ri, pi=pi, delta_coeff=d_c, modes=modes))
        return edge_mode_sums

    # Convert poles to complex once.
    try:
        modes_lambdas = tuple(complex(CDF(SR(p))) * 1j for p in pole_vals)
    except Exception:
        return None
    n_poles = len(modes_lambdas)

    edge_mode_sums = []
    for ei in edge_info:
        ri, pi = ei['ri'], ei['pi']
        try:
            residues = tuple(
                complex(CDF(SR(C_mats[k][pi, ri])))
                for k in range(n_poles)
            )
        except Exception:
            return None
        modes = tuple(zip(residues, modes_lambdas))
        try:
            d_c = complex(ei['delta_coeff'])
        except Exception:
            try:
                d_c = complex(CDF(SR(ei['delta_coeff'])))
            except Exception:
                return None
        edge_mode_sums.append(EdgeModeSum(
            ri=ri, pi=pi,
            delta_coeff=d_c,
            modes=modes,
        ))
    return edge_mode_sums


def _is_exact_zero(x):
    r"""True iff the stored number ``x`` is exactly zero, decided on ``x``
    itself, with no tolerance: a Python / CDF number compares exactly with 0
    (a residue that is merely tiny is NOT zero); anything else goes through
    ``SR(x).is_trivial_zero()`` (structural: a symbolic 0, never a proof).
    A CDF number is compared as a Python complex (an exact conversion):
    Sage's CDF compares a NaN in either part EQUAL to 0, Python does not.
    NaN and inf are never zero; undecidable counts as nonzero."""
    try:
        if isinstance(x, _ComplexDoubleElement):
            return complex(x) == 0
        if isinstance(x, (int, float, complex)):
            return bool(x == 0)
        return bool(SR(x).is_trivial_zero())
    except Exception:
        return False


class PoleFreePropagatorError(ValueError):
    r"""Phase J was handed a propagator WITHOUT any pole
    (``propagator_data['pole_vals']`` is empty) and a diagram needs an
    entry whose smooth (non-instantaneous) part cannot be shown to vanish.

    The propagator builder keeps only strictly retarded poles
    (Im ω > 1e-9, ``api/_propagator.py``).  A marginal or non-retarded mode
    -- e.g. λ = 0 at q = 0 for a massless field -- is dropped from the
    list although the entry still has a smooth part, which then cannot be
    evaluated from the pole list.  Raised (with ``STRUCTURAL_ZEROS`` on)
    instead of returning a value: before M2a the per-diagram path failed
    here with a bare ``IndexError``.  ``entries``: the offending
    ``(pi, ri)`` entries; ``reasons``: ``{(pi, ri): why}``."""

    def __init__(self, message, entries=(), reasons=None):
        super().__init__(message)
        self.entries = tuple(entries)
        self.reasons = dict(reasons or {})


def _entry_is_provably_instantaneous(propagator_data, num_params, pi, ri):
    r"""Decide whether entry ``G[pi, ri]`` is PROVABLY a pure δ (or 0),
    i.e. whether its smooth part ``G(ω) − G(ω → ∞)`` is identically zero.
    That is the case exactly when the entry does not depend on ω.

    Decided on ``propagator_data['G_ft']`` -- the propagator BEFORE any
    pole was found or filtered -- with ``num_params`` substituted, by a
    structural test with no tolerance and no simplification: the entry
    must not contain ``propagator_data['omega']`` at all.  (A pure-δ entry
    such as the spike train of an undriven population of
    ``single_population_linear_delta_spikes_test`` passes: substituting
    its zero couplings removes every ω.)  Anything else -- ω present,
    ``G_ft`` or ``omega`` missing, an error -- is "not proven".  Returns
    ``(True, None)`` or ``(False, reason)``."""
    G_ft = propagator_data.get('G_ft')
    omega = propagator_data.get('omega')
    if G_ft is None or omega is None:
        return False, ("propagator_data has no 'G_ft' / 'omega' to decide "
                       "it from")
    try:
        e = SR(G_ft[pi, ri])
        if num_params:
            e = e.subs(num_params)
        if SR(omega) not in e.variables():
            return True, None
        text = str(e)
        if len(text) > 160:
            text = text[:157] + '...'
        return False, (f'its G_ft entry still depends on {omega} after the '
                       f'numeric parameters are substituted: {text}')
    except Exception as exc:                               # noqa: BLE001
        return False, (f'its G_ft entry could not be examined '
                       f'({type(exc).__name__}: {exc})')


def _pole_free_forced_delta_edges(edge_info, propagator_data, zero_by_entry,
                                  num_params):
    r"""``_forced_delta_edges`` for a propagator with an EMPTY pole list:
    every edge whose entry is provably a pure δ
    (``_entry_is_provably_instantaneous``) is forced-δ.  If any edge's
    entry is not provably a pure δ, raise ``PoleFreePropagatorError``
    naming every such entry: its smooth part exists but is not in the
    (empty) pole list, so no value can be computed for this diagram."""
    out = set()
    bad = {}
    for i, ei in enumerate(edge_info):
        key = (ei['pi'], ei['ri'])
        if zero_by_entry.get(key) is True:
            out.add(i)
            continue
        if key in bad:
            continue
        ok, why = _entry_is_provably_instantaneous(
            propagator_data, num_params, key[0], key[1])
        if ok:
            zero_by_entry[key] = True
            out.add(i)
        else:
            bad[key] = why
    if bad:
        nf = propagator_data.get('nf')
        lines = '; '.join(f'G[pi={p}, ri={r}]: {why}'
                          for (p, r), why in sorted(bad.items()))
        raise PoleFreePropagatorError(
            f'Phase J: the propagator has no pole at all '
            f"(propagator_data['pole_vals'] is empty"
            f"{'' if nf is None else f'; nf = {nf}'}), but the smooth "
            f'(non-instantaneous) part of {len(bad)} entr'
            f"{'y' if len(bad) == 1 else 'ies'} used by this diagram is "
            f'not provably zero -- {lines}.  The propagator builder keeps '
            f'only strictly retarded poles (Im ω > 1e-9, '
            f'api/_propagator.py), so a marginal or non-retarded mode '
            f'(e.g. λ = 0 at q = 0 for a massless field) was dropped from '
            f'the pole list, and this smooth part cannot be evaluated '
            f'from it, so no value is returned (rather than a wrong one).  '
            f'(An entry whose G_ft contains no ω after the numeric '
            f'parameters are substituted is a pure δ and is evaluated.)',
            entries=sorted(bad), reasons=bad)
    return frozenset(out)


def _smooth_part_is_exact_zero(C_mats, n_poles, pi, ri):
    r"""True if the propagator has at least one pole and every pole residue
    ``C_mats[k][pi, ri]`` is an EXACT zero (``_is_exact_zero``).

    With no poles at all this residue test decides nothing (False): an
    empty pole list is not evidence of a zero smooth part, because the
    propagator builder keeps only strictly retarded poles (Im ω > 1e-9,
    ``api/_propagator.py``) and drops a marginal λ = 0 mode (e.g. a
    massless spatial mode at q = 0) although its smooth part is not zero.
    ``_forced_delta_edges`` decides a pole-free propagator from ``G_ft``
    instead (``_entry_is_provably_instantaneous``).  Any failure to
    decide counts as "not zero" too (the conservative answer: nothing is
    skipped).  Exact zeros here are the builder's stored values, not a
    proof: it keeps only the retarded poles and sets a residue to 0 by
    numeric tests, so ``_forced_delta_edges`` also asks ``G_ft``."""
    if n_poles < 1:
        return False
    for k in range(n_poles):
        try:
            x = C_mats[k][pi, ri]
        except Exception:
            return False
        if not _is_exact_zero(x):
            return False
    return True


def _forced_delta_edges(edge_info, propagator_data, zero_by_entry=None,
                        num_params=None):
    r"""Indices of the edges in ``edge_info`` whose smooth propagator part
    is identically zero (M2a, plan §3.1 L1: "forced-δ" edges).

    For such an edge ``G_R = δ_coeff·δ(Δt) + Θ(Δt)·0``, so every δ-subset
    that keeps it smooth has an identically-zero integrand: the mirror image
    of the forced-smooth edges (zero δ part), whose δ branch the subset
    enumeration already skips.  Every evaluator of a subset (the mode-sum
    cache ``_build_edge_mode_sums``, the fast pole/residue closure and the SR
    ``smooth_factor`` of ``build_G_t_matrix``) reads the same
    ``propagator_data['C_mats']`` entry, so an exact zero there is a zero
    for all of them.  Returns a frozenset (empty when the pole data is
    missing).

    With ``G_ft`` available, an edge is forced-δ only when its smooth part
    is provably zero, decided before any pole was filtered: every stored
    residue of its entry is an exact zero (``_smooth_part_is_exact_zero``)
    AND its ``G_ft`` entry contains no ω once ``num_params`` are
    substituted (``_entry_is_provably_instantaneous``).  The stored
    residues alone are no proof: the builder keeps only strictly retarded
    poles, so an entry whose own mode was dropped as marginal has
    exact-zero residues at the kept poles although its smooth part is not
    zero; such an edge is kept (its subsets are evaluated as before M2a).
    Without ``G_ft`` nothing can be proven, and the stored residues alone
    decide.  ``api/_propagator.py`` leaves ``G_ft`` as None when it skips
    the symbolic inverse for a "rich" propagator (nf ≥ 6 or more than 20
    free symbols; e.g. ``multipopulation_test``, nf = 8), and when that
    inverse exceeds its time budget or fails.  The stored residues are what
    every evaluator reads, so the skipped integrand is exactly the zero it
    would have evaluated (a mode the builder filtered stays lost, as it is
    without the prune).

    A propagator with an EMPTY pole list is decided from ``G_ft`` alone
    (``_pole_free_forced_delta_edges``): an edge is forced-δ when its entry
    provably contains no ω after ``num_params`` are substituted (a pure δ);
    if some edge's entry is not provably a pure δ (its modes were dropped
    as non-retarded, e.g. λ = 0 at q = 0), ``PoleFreePropagatorError`` is
    raised.  ``zero_by_entry``: an optional ``{(pi, ri): bool}`` cache
    shared between calls on the same ``propagator_data`` and
    ``num_params``."""
    pole_vals = propagator_data.get('pole_vals')
    C_mats = propagator_data.get('C_mats')
    if pole_vals is None or C_mats is None:
        return frozenset()
    n_poles = len(pole_vals)
    if zero_by_entry is None:
        zero_by_entry = {}
    if n_poles < 1:
        return _pole_free_forced_delta_edges(edge_info, propagator_data,
                                             zero_by_entry, num_params)
    have_g_ft = (propagator_data.get('G_ft') is not None
                 and propagator_data.get('omega') is not None)
    out = set()
    for i, ei in enumerate(edge_info):
        key = (ei['pi'], ei['ri'])
        z = zero_by_entry.get(key)
        if z is None:
            z = _smooth_part_is_exact_zero(C_mats, n_poles, key[0], key[1])
            if z and have_g_ft:
                z = _entry_is_provably_instantaneous(
                    propagator_data, num_params, key[0], key[1])[0]
            zero_by_entry[key] = z
        if z:
            out.add(i)
    return frozenset(out)


def _extract_exp_mode(sr_expr, tau_sym):
    """Extract ``(C, λ)`` from an SR expression of the form
    ``C · exp(λ · tau_sym)`` (single-exponential kernel).

    Uses the log-derivative trick: λ = d/dτ log(g(τ)) evaluated at
    τ=0, then C = g(0).  Works for any single-exponential expression
    after Heaviside-stripping and num-params substitution.

    Returns ``(C_complex, λ_complex)`` or ``None`` if the expression
    can't be reduced to this form (e.g. polynomial-prefactor kernels
    like the alpha kernel ``τ/τ_g² · exp(-τ/τ_g)``, which would need
    a multi-mode decomposition).
    """
    try:
        c_at_zero = sr_expr.subs({tau_sym: 0})
        C = complex(CDF(SR(c_at_zero)))
    except Exception:
        return None
    if C == 0:
        # Polynomial-prefactor kernel (e.g. alpha) — single-exponential
        # extraction doesn't apply.  Caller should fall back to the
        # SR + scipy path until multi-mode kernels are supported.
        return None
    try:
        deriv = sr_expr.diff(tau_sym)
        lam_sr = (deriv / sr_expr).subs({tau_sym: 0})
        lam = complex(CDF(SR(lam_sr)))
    except Exception:
        return None
    return (C, lam)


def _attach_subset_dt(edge_mode_sum, a_int, a_ext, c0):
    """Return a copy of ``edge_mode_sum`` with the per-subset Δt
    linear form (sparse coefficients on integration vars + free
    external times, plus constant) populated.
    """
    int_pairs = tuple(
        (i, float(a)) for i, a in enumerate(a_int)
        if abs(float(a)) > 1e-15
    )
    ext_pairs = tuple(
        (i, float(a)) for i, a in enumerate(a_ext)
        if abs(float(a)) > 1e-15
    )
    return EdgeModeSum(
        ri=edge_mode_sum.ri,
        pi=edge_mode_sum.pi,
        delta_coeff=edge_mode_sum.delta_coeff,
        modes=edge_mode_sum.modes,
        dt_c0=float(c0),
        dt_int_pairs=int_pairs,
        dt_ext_pairs=ext_pairs,
    )


# ───────────────────────────────────────────────────────────────────────
# Analytic ∫∫_polygon exp(α·x + β·y) dA  (Stage 3a-full)
# ───────────────────────────────────────────────────────────────────────
# Replaces scipy.nquad on the m=2 polytope.  The integrand factors
# after pole-expansion as a sum of single-exponential terms
# A·exp(α·s_0 + β·s_1 + γ).  Each term is integrated analytically:
#
#   1. The polytope is a convex polygon (intersection of half-planes
#      from the retardation constraints).  Computed once per
#      (subset, τ-point) via Sutherland-Hodgman clipping.
#   2. Fan-triangulate the polygon from vertex 0.
#   3. Per triangle: affine-map to the unit triangle 0 ≤ u, w ≤ 1,
#      u + w ≤ 1.  The integrand reduces to ``exp(α₀ + p·u + q·w)``
#      with α₀, p, q expressible from (α, β, vertices).
#   4. Unit-triangle integral ``J(p, q) = ∫₀¹ ∫₀^{1-u} exp(p·u + q·w)
#      dw du`` has the closed form
#
#         J(p, q) = [(eᵖ − e^q)/(p − q) − (eᵖ − 1)/p] / q
#
#      with stable Taylor fallbacks when |p|, |q|, or |p−q| → 0.
#   5. Sum over pole tuples ``(α_e)_e``: each contributes
#      A · exp(γ) · |det| · exp(α₀) · J(p, q).
#
# Compared to scipy.nquad on the un-expanded integrand: O(n_poles^|E_smooth|
# × n_triangles) complex-exp evaluations per (subset, τ-point) instead
# of ~10⁴ adaptive samples per (subset, τ-point), each itself doing
# n_poles · |E_smooth| complex-exps.  Net speedup typically 10-100×.

USE_POLYGON_M2_INTEGRATOR = True
POLYGON_BBOX_CAP = 200.0  # bounding-box for unbounded polygons
# The three analytic modesum integrators take ``bbox_cap=None`` and resolve it
# at CALL TIME from this module attribute (``_resolve_bbox_cap``), so
# ``final_integral.POLYGON_BBOX_CAP = x`` takes effect immediately, in every
# Θ(0) mode.  Before M1 the default was bound at definition time
# (``bbox_cap=POLYGON_BBOX_CAP``) and changing the attribute did nothing.


# ───────────────────────────────────────────────────────────────────────
# Phase J flags (read at CALL TIME) and the legacy umbrella
# ───────────────────────────────────────────────────────────────────────
# docs/integration_speedup_plan.md §3.1 "Rollback and flag hygiene".  Every
# Phase J flag is a module attribute; its environment variable is read ONCE,
# at import, only to initialise it.  Call sites read the attribute at call
# time (never through a default argument or a closure capture), so
# ``monkeypatch.setattr(final_integral, FLAG, value)`` takes effect at once,
# including for closures built before the change.  ``grouped_integral``
# reads them through the module object (``_fi_mod``), never by name.
#
# ``DAEDALUS_PHASE_J_LEGACY=1`` (the umbrella) sets EVERY Phase J flag to its
# legacy value, reproducing the pre-M1 numbers bit-for-bit within one
# process.  That is ``THETA0_CONST_ROW_MODE = 'legacy_clip'``,
# ``STRUCTURAL_ZEROS = False`` (M2a), ``NQUAD_HARDENED = False`` (M2b) and
# ``USE_DBM_FALLBACK = False`` (M3; see ``_initial_phase_j_flags``).  The
# bounding box is not a flag: the umbrella also reads ``POLYGON_BBOX_CAP`` at
# call time (so it reproduces a pre-M1 run at any cap, e.g. the cap-12 trap).
import os as _os


def _env_truthy(name, environ=None):
    env = _os.environ if environ is None else environ
    return env.get(name, '').strip().lower() not in (
        '', '0', 'false', 'no', 'off')


# ── Θ(0) convention for constant constraint rows (M1; plan §2.4, §3.1 L0) ──
# After δ-elimination a smooth edge can have Δt ≡ const (both endpoints
# merged by δ edges, or the edge parallel to a δ edge), i.e. a constraint row
# with a zero normal: a_int ≡ 0, so the row reads ``c_eff > 0`` with
# ``c_eff = c0 + Σ_j a_ext_j t_j``.  Its Heaviside is Θ(c_eff).  The package
# convention is the Itô one, Θ(0) = 0 -- the same convention that evaluates
# the k=2 τ = 0 grid point at the left limit τ = −``api.compute._ITO_EPS``
# (memory note ``project_ito_equal_time_tau0``).  ``_const_row_verdict`` is
# the ONE place that applies it; every integrator (m=0, m=1, m=2 polygon,
# m≥3 poset, scipy.nquad fallback, grouped m0/m1) calls it.
#
# The rule is EXACT: c_eff > 0 keeps the row (DROP), c_eff < 0 empties the
# region, and only c_eff == 0.0 is a tie.  (A relative tie tolerance was
# tried and rejected in the M1 review: ``contribution()`` evaluates each
# Wick permutation on time differences taken relative to a different origin
# leg, so a tolerance on those differences declared a rounding-level
# near-tie a tie in some permutations and not in others, giving a value
# that was neither the distinct-time nor the tied-time one.  The sign of a
# difference of two floats is exact, so the exact rule is the same in every
# permutation.)  A tie is resolved as follows:
#
# * a row whose Δt does not depend on the external times (a_ext ≡ 0,
#   c0 == 0: both endpoints merged into one vertex time by δ edges) is
#   Θ(0) = 0 (``THETA_AT_ZERO_CONST_ROW``): the Itô rule proper;
# * a row whose Δt is the difference of two external legs' times,
#   Δt = t_p − t_q, is decided by the order of the legs' RAW times
#   (``_TieContext``, built per Wick permutation by ``contribution()``), and
#   when those are exactly equal by the leg order: a leg with a larger
#   index counts as infinitesimally EARLIER (``_tie_order_sign``).  So of
#   Θ(t_p − t_q) and Θ(t_q − t_p) exactly one holds at t_p == t_q, and the
#   value at a tie is the one-sided limit in which the later leg approaches
#   from below -- for k = 2 exactly the Itô left limit τ = t_1 − t_0 → 0⁻
#   that the τ grid samples at −_ITO_EPS, and for k ≥ 3 its natural
#   extension (Θ(0) = 0 on both orientations would drop both and give a
#   value that is neither limit);
# * any other constant row with c_eff == 0.0 (no row context, or an
#   unusual row shape) is Θ(0) = 0.
THETA_AT_ZERO_CONST_ROW = 0
# A row has a zero normal when every |a_int_j| <= _ROW_COEF_ATOL.  The δ-solve
# produces small-integer coefficients, so real rows are exactly 0 or ±1.
_ROW_COEF_ATOL = 1e-12
# Emptiness of an m=2 polygon is decided structurally and exactly too (no
# area tolerance: a genuine strip of any width, e.g. between external times
# 1e-12 apart, is integrated as before 0.2.0).  See
# ``_polygon_from_2d_constraints``: two rows with exactly opposite normals
# whose constants sum to <= 0 (the 2-cycle rows s1 > s0, s0 > s1, or a strip
# between two coincident external times) empty the region, and a clipped
# polygon with fewer than 3 vertices or an exactly zero computed area is
# degenerate.

# THETA0_CONST_ROW_MODE:
#   'ito'          (default) Θ(0) = 0 everywhere through ``_const_row_verdict``,
#                  with the external-time tie order above; an m=2 polygon
#                  with an exactly contradictory pair of opposite rows or an
#                  exactly zero area, and a directed cycle of m≥3 order rows
#                  whose shifts sum to <= 0, are empty; a δ-subset with a
#                  τ-independent EMPTY row returns 0 without integration
#                  (decided at call time, see ``integrate_diagram``).
#   'legacy_clip'  the pre-M1 behaviour, bit-for-bit: the m=2 polygon clip
#                  keeps a zero-normal row with c_eff == 0 (Θ(0) = 1), every
#                  other path uses its own absolute tolerances.
# The two modes can differ only on an evaluation where 'ito' decides a
# constant row with the helper, in any path (``theta0_const_empty`` /
# ``theta0_const_drop``; ``zero_normal_rows_seen`` counts only the polygon
# and poset paths), skips a τ-independent EMPTY subset
# (``theta0_subsets_pruned``), or makes an exact structural emptiness
# decision that needs no constant row (``polygon_zero_area``,
# ``poset_empty_cycle``; these may turn a rounding-level value, or an nquad
# value / exception, into an exact 0).  With all five counters at 0 the
# modes are bit-identical in-process.
# Environment: DAEDALUS_PHASE_J_THETA0_CONST_ROW = ito | legacy_clip.
_THETA0_MODES = ('ito', 'legacy_clip')

# ── Structural zeros (M2a; plan §3.1 L1) ──
# STRUCTURAL_ZEROS (bool, read at call time):
#   True   (default) two kinds of δ-subset work that is identically zero are
#          skipped:
#          * ``_integrate_polytope`` (the scipy.nquad fallback, whoever calls
#            it: an m=2 guard bail, a poset extraction that bailed, grouped
#            m2/m≥3, the SR path of a non-analytic subset, a direct caller)
#            returns 0j at once when the rows contain a directed cycle of
#            unit-coefficient pair rows whose shifts sum to <= 0 in exact
#            arithmetic (``_region_structurally_empty``; counter
#            ``polytope_empty_cycle``).  The strict inequalities around such
#            a cycle cannot all hold, so the region is empty.  The fallback
#            used to hand it to scipy.nquad anyway: an identically-zero
#            filtered integrand, or (m=2 with exact bounds, no filter)
#            zero-length inner intervals whose quadrature still evaluates
#            the integrand outside the region, where fast poles overflow
#            (OverflowError, plan Appendix C.2).
#          * a δ-subset that keeps smooth an edge whose smooth part is
#            provably zero (a pure-δ entry, or an entry that vanishes: every
#            stored pole residue is an exact zero
#            (``_smooth_part_is_exact_zero``) AND the ``G_ft`` entry -- the
#            propagator before any pole filtering -- contains no ω once
#            ``num_params`` are substituted
#            (``_entry_is_provably_instantaneous``); without ``G_ft`` the
#            stored residues decide) is never built
#            (``_forced_delta_edges``; counter ``forced_delta_pruned``), per
#            diagram and in grouped builds -- the mirror image of the
#            forced-smooth edges (zero δ part) that the subset enumeration
#            already skips.  Such a subset's integrand is a product with an
#            identically-zero factor.  A propagator WITHOUT any pole is
#            decided from ``G_ft`` alone: an entry is a pure δ only if it
#            contains no ω once ``num_params`` are substituted (e.g. a
#            spike model with all couplings 0); if a diagram uses an entry
#            that is not provably a pure δ, its smooth part was dropped
#            with a non-retarded mode (λ = 0 at q = 0 for a massless field)
#            and ``PoleFreePropagatorError`` is raised -- the per-diagram
#            path used to fail there with a bare IndexError, and the grouped
#            path used to drop that smooth part silently.
#          Both are value-neutral by construction: every skipped piece
#          contributes an exact zero (the analytic paths skip zero-residue
#          pole tuples, nquad of the zero function is exactly 0.0, and
#          adding a signed zero to the running sum never changes it).  They
#          remove the exceptions / NaN the legacy code could raise on those
#          pieces.  (Constant EMPTY rows need no rule of their own here:
#          'ito' decides them by ``_const_row_verdict`` at the entry of
#          ``_integrate_polytope``, and the legacy inner integrators return
#          0 on them before any quadrature -- ``_outer_bounds``,
#          ``_resolve_1d_bounds``, ``_integrate_2d_polytope``.)
#   False  the pre-M2a behaviour, bit-for-bit.
# The prune is decided when the subsets are built (the flag is read at the
# call of ``integrate_diagram`` / ``integrate_grouped_diagram``); a closure
# built with the prune never re-creates the pruned subsets, which are
# identically zero anyway.  The cycle test is read at every call.
# Environment: DAEDALUS_PHASE_J_STRUCTURAL_ZEROS = 1 | 0 (also true/false,
# yes/no, on/off); the umbrella DAEDALUS_PHASE_J_LEGACY=1 sets it to 0.
_STRUCTURAL_ZEROS_ON = ('', '1', 'true', 'yes', 'on')
_STRUCTURAL_ZEROS_OFF = ('0', 'false', 'no', 'off')

# ── Hardened scipy quadrature fallback (M2b; plan §3.1 L1b) ──
# NQUAD_HARDENED (bool, read at call time):
#   True   (default) every ``_integrate_polytope`` region that reaches scipy
#          quadrature (m >= 1, after the constant-row verdicts and the
#          structural-zero test) is integrated by
#          ``_integrate_polytope_hardened``: nested ``scipy.integrate.quad``
#          over the region's OWN bounds (a difference-bound-matrix closure
#          of its rows, not the ±200 / ±50 box), the innermost variable in
#          closed form when the integrand's modes are known, with an empty
#          interval answered 0 without sampling, the integrand
#          Heaviside-filtered, breakpoints at the external times, at the
#          kinks of the inner bounds and geometrically towards the ends of
#          wide panels, epsabs =
#          ``NQUAD_EPSABS_FACTOR`` × a rigorous bound of |integrand| from
#          the modes and epsrel = ``NQUAD_EPSREL``.  A direction that the
#          rows leave open is truncated at ``NQUAD_TAIL_K`` / κ_min from
#          its finite end (κ_min: the slowest decay rate of the modes), or,
#          without a decay certificate, at the legacy ±200.  See that
#          function for the details and the counters.
#   False  the pre-M2b default-tolerance scipy.nquad routines, bit-for-bit.
# Environment: DAEDALUS_PHASE_J_NQUAD_HARDENED = 1 | 0 (also true/false,
# yes/no, on/off); the umbrella DAEDALUS_PHASE_J_LEGACY=1 sets it to 0.

# ── Poset lower-bound inheritance and the exact DBM route (M3; plan §3.1
# L7, problem P3) ──
# USE_DBM_FALLBACK (bool, read at call time; acts only while
# THETA0_CONST_ROW_MODE is 'ito', see ``_dbm_route_on``):
#   True   (default) two changes to the m≥3 subsets:
#          * P3, the inheritance rule.  The poset path integrates every
#            variable from ONE shared scalar lower bound L.  A variable
#            without a scalar lower of its own may inherit L only through a
#            predecessor in the transitive closure of the strict order that
#            has one (s_v > s_u > L).  A poset with a variable that has
#            neither extends below L, down to the domain cap; the poset path
#            now bails on it (``'poset_lower_not_inherited'``; counter
#            ``poset_lower_not_inherited``) instead of cutting that region
#            off at L.
#          * The DBM route.  Every m≥3 subset whose poset path bails (for
#            any reason: the P3 rule above, unequal scalar lowers, shifted
#            order rows, an order cycle with a positive total shift, an
#            overflow or degenerate chain) is integrated exactly by
#            ``_integrate_subset_dbm`` (``dbm_integral.py``: case-split
#            Fourier-Motzkin elimination of the difference constraints,
#            log-domain terms).  A direction the rows leave open is closed
#            at ±OUTER_CAP (``_nquad_outer_cap``, ±200: the box of the
#            legacy default-tolerance scipy.nquad routines; the M3 brief
#            fixes this domain); a finite bound is kept as it is, even
#            beyond the box.  The box is absolute (from the time origin), so
#            a region bounded above near or below -OUTER_CAP with a time open
#            below is truncated or emptied (answered 0) by it, as by the
#            legacy routine.  The default hardened fallback
#            (``NQUAD_HARDENED``) truncates a direction with a decay
#            certificate at K/κ beyond its breakpoints instead, which is
#            nearly box-free: for modes slower than κ ≈ 0.15 a non-P3 bail
#            that M3 moves from it to the DBM can change by the box
#            truncation (none in the measured public zoo: no m≥3 region
#            reached the fallback there).
#            Constant rows are decided by ``_const_row_decision`` (Θ(0) = 0
#            and the external-time tie order; the side-effect-free form, as
#            the poset path has already counted them); every other row
#            enters the region as it is: the integral is continuous in a
#            non-constant row's shift, so it needs no tie rule.  A subset
#            the DBM cannot take (a row that is not a difference row, e.g.
#            a ConvVertex 3-term row; rows and modes that do not match; an
#            overflowing term; a closed form whose rounding error estimate
#            exceeds max(1e-10 |value|, 1e-14 × the integrand's scale), as
#            close poles or a thin region can give) goes on to
#            ``_integrate_polytope`` as before (counters ``dbm_declined_*``).
#   False  the pre-M3 behaviour, bit-for-bit (the shared L is inherited by
#          every variable, and a poset bail goes to ``_integrate_polytope``).
# Under THETA0_CONST_ROW_MODE 'legacy_clip' neither change applies (that
# mode reproduces the pre-M1 numbers bit-for-bit, and the DBM decides
# constant rows by the Itô rule).
# Environment: DAEDALUS_PHASE_J_DBM = 1 | 0 (also true/false, yes/no,
# on/off); the umbrella DAEDALUS_PHASE_J_LEGACY=1 sets it to 0.


def _initial_phase_j_flags(environ=None):
    """The import-time values of the Phase J flags from ``environ``
    (default ``os.environ``): ``{'THETA0_CONST_ROW_MODE': ...,
    'STRUCTURAL_ZEROS': ..., 'NQUAD_HARDENED': ...,
    'USE_DBM_FALLBACK': ...}``.  The umbrella wins over the per-flag
    variables.  An unknown value raises (a silent fallback to the default
    would hide a requested rollback)."""
    env = _os.environ if environ is None else environ
    if _env_truthy('DAEDALUS_PHASE_J_LEGACY', env):
        return {'THETA0_CONST_ROW_MODE': 'legacy_clip',
                'STRUCTURAL_ZEROS': False,
                'NQUAD_HARDENED': False,
                'USE_DBM_FALLBACK': False}
    mode = (env.get('DAEDALUS_PHASE_J_THETA0_CONST_ROW', '')
            .strip().lower() or 'ito')
    if mode not in _THETA0_MODES:
        raise ValueError(
            f'DAEDALUS_PHASE_J_THETA0_CONST_ROW={mode!r}: expected one of '
            f'{_THETA0_MODES}')
    sz = env.get('DAEDALUS_PHASE_J_STRUCTURAL_ZEROS', '').strip().lower()
    if sz not in _STRUCTURAL_ZEROS_ON + _STRUCTURAL_ZEROS_OFF:
        raise ValueError(
            f'DAEDALUS_PHASE_J_STRUCTURAL_ZEROS={sz!r}: expected 1 or 0')
    nh = env.get('DAEDALUS_PHASE_J_NQUAD_HARDENED', '').strip().lower()
    if nh not in _STRUCTURAL_ZEROS_ON + _STRUCTURAL_ZEROS_OFF:
        raise ValueError(
            f'DAEDALUS_PHASE_J_NQUAD_HARDENED={nh!r}: expected 1 or 0')
    db = env.get('DAEDALUS_PHASE_J_DBM', '').strip().lower()
    if db not in _STRUCTURAL_ZEROS_ON + _STRUCTURAL_ZEROS_OFF:
        raise ValueError(
            f'DAEDALUS_PHASE_J_DBM={db!r}: expected 1 or 0')
    return {'THETA0_CONST_ROW_MODE': mode,
            'STRUCTURAL_ZEROS': sz in _STRUCTURAL_ZEROS_ON,
            'NQUAD_HARDENED': nh in _STRUCTURAL_ZEROS_ON,
            'USE_DBM_FALLBACK': db in _STRUCTURAL_ZEROS_ON}


_PHASE_J_FLAGS_AT_IMPORT = _initial_phase_j_flags()
THETA0_CONST_ROW_MODE = _PHASE_J_FLAGS_AT_IMPORT['THETA0_CONST_ROW_MODE']
STRUCTURAL_ZEROS = _PHASE_J_FLAGS_AT_IMPORT['STRUCTURAL_ZEROS']
NQUAD_HARDENED = _PHASE_J_FLAGS_AT_IMPORT['NQUAD_HARDENED']
USE_DBM_FALLBACK = _PHASE_J_FLAGS_AT_IMPORT['USE_DBM_FALLBACK']


def _dbm_fallback_on():
    """``USE_DBM_FALLBACK`` (call time), validated."""
    flag = USE_DBM_FALLBACK
    if flag is True or flag is False:
        return flag
    raise ValueError(f'final_integral.USE_DBM_FALLBACK={flag!r}: expected '
                     f'True or False')


def _dbm_route_on():
    """True when the M3 changes apply (call time): ``USE_DBM_FALLBACK`` on
    AND ``THETA0_CONST_ROW_MODE`` 'ito' (see the flag's comment)."""
    return _dbm_fallback_on() and not _theta0_legacy()


def _nquad_hardened_on():
    """True when the hardened quadrature fallback is enabled (call time)."""
    flag = NQUAD_HARDENED
    if flag is True or flag is False:
        return flag
    raise ValueError(f'final_integral.NQUAD_HARDENED={flag!r}: expected '
                     f'True or False')


def _structural_zeros_on():
    """True when the M2a structural-zero skips are enabled (call time)."""
    flag = STRUCTURAL_ZEROS
    if flag is True or flag is False:
        return flag
    raise ValueError(f'final_integral.STRUCTURAL_ZEROS={flag!r}: expected '
                     f'True or False')


def _theta0_legacy():
    """True when the pre-M1 constant-row behaviour is selected (call time)."""
    mode = THETA0_CONST_ROW_MODE
    if mode == 'ito':
        return False
    if mode == 'legacy_clip':
        return True
    raise ValueError(f'final_integral.THETA0_CONST_ROW_MODE={mode!r}: '
                     f'expected one of {_THETA0_MODES}')


def _resolve_bbox_cap(bbox_cap):
    """``bbox_cap`` if given, else ``POLYGON_BBOX_CAP`` (read at call time)."""
    return POLYGON_BBOX_CAP if bbox_cap is None else bbox_cap

# ───────────────────────────────────────────────────────────────────────
# Physical fallback margin for unbounded polytope sides (Stage 3b)
# ───────────────────────────────────────────────────────────────────────
# When the polytope has no explicit scalar lower/upper, the chain
# simplex needs *some* finite L / U.  POLYGON_BBOX_CAP=200 is too loose:
# combined with retarded poles (Re β < 0), |Re β · L|=β·200 overflows
# the exp guard at |Re β|>3 (sum of 3–4 poles).  Physically, retarded
# propagators decay over a few correlation times beyond the earliest
# external time, so the integrand is negligible past
# ``min(0, free_ext_vals) − POSET_PHYSICAL_MARGIN`` (lower) and
# ``max(0, free_ext_vals) + POSET_PHYSICAL_MARGIN`` (upper).
# 50 is generous (≈ several correlation times for typical Hawkes
# τ_v ~ 10) but tight enough to avoid the overflow guard at typical
# pole magnitudes.
POSET_PHYSICAL_MARGIN = 50.0

# ───────────────────────────────────────────────────────────────────────
# Runtime path counters (diagnostic; zero perf impact when not read)
# ───────────────────────────────────────────────────────────────────────
# Increment at decision points inside the analytic evaluators so we can
# tell whether the intended analytic path actually completed or fell
# back to scipy.nquad at runtime.  The ``_evaluator_label`` recorded in
# ``subset_diagnostics`` is INTENT (set at subset setup time); these
# counters are RUNTIME.  Call ``_reset_runtime_counters()`` before a
# timed run, then read ``_RUNTIME_COUNTERS`` after.
_RUNTIME_COUNTERS = {
    # m=2 polygon path
    'polygon_attempted': 0,
    'polygon_returned_none': 0,
    # m≥3 poset path
    'poset_attempted': 0,
    'poset_extract_returned_none': 0,
    'poset_consistent_lower_failed': 0,
    'poset_maximality_failed': 0,
    'chain_simplex_fast_returned_none': 0,
    'chain_simplex_polynomial_called': 0,
    'chain_simplex_polynomial_returned_none': 0,
    # memoisation (see the wrappers below)
    'chain_simplex_memo_hits': 0,
    'chain_simplex_memo_misses': 0,
    # M4: the ``_chain_with_intermediate_uppers`` memo
    # (``USE_CHAIN_UPPERS_MEMO``); ``evictions`` counts the table clears at
    # the size cap:
    'chain_uppers_memo_hits': 0,
    'chain_uppers_memo_misses': 0,
    'chain_uppers_memo_evictions': 0,
    'poset_returned_none_total': 0,
    # m=1 interval path
    'interval_attempted': 0,
    'interval_returned_none': 0,
    # scipy.nquad fallback (counted at _integrate_polytope entry for m≥1)
    'scipy_nquad_called_m1': 0,
    'scipy_nquad_called_m2': 0,
    'scipy_nquad_called_mge3': 0,
    # ── M0.1 observational counters (docs/integration_speedup_plan.md
    # §2.4 item 8, §4.1).  Purely observational: nothing reads them to
    # decide control flow.
    #
    # Entries into the scipy.nquad fallback: ``_integrate_polytope`` calls
    # with m >= 1, so ``nquad_calls == scipy_nquad_called_m1 + _m2 + _mge3``
    # always.  (An entry may still return before invoking
    # scipy.integrate.nquad, and an invoked quadrature integrates the real
    # and imaginary parts separately: invocations <= 2 * nquad_calls.)
    # The plan's "0 nquad calls on the model zoo" gate (§2.4 item 8, M9) is
    # keyed on this counter.
    'nquad_calls': 0,
    # ``_integrate_polytope`` entries with m = 0: a direct constraint check
    # + evaluation of a δ-collapsed subset, NOT quadrature.
    'polytope_m0_direct': 0,
    # Why an m≥1 subset call reached ``_integrate_polytope`` (counted by the
    # per-diagram and grouped dispatch closures, not by direct callers).
    # Keys are exactly ``_NQUAD_FALLBACK_REASONS``; the fine-grained reason
    # (e.g. 'polygon_gamma_overflow') is passed to ``_SUBSET_HOOK`` instead.
    'nquad_fallback_by_reason': {
        'polygon_guard': 0, 'polygon_other': 0,
        'poset_extract_none': 0, 'poset_no_extension': 0,
        'poset_chain_none': 0, 'other': 0,
    },
    # Constraint rows with every |a_int| <= 1e-12 (a smooth edge whose Δt is
    # constant after δ-elimination) processed by ``_polygon_from_2d_
    # constraints`` or ``_extract_causal_poset``.
    'zero_normal_rows_seen': 0,
    # ── M1 Θ(0) counters (plan §2.4 item 8).  One count per
    # ``_const_row_verdict`` call (i.e. per zero-normal row per evaluation;
    # never incremented in 'legacy_clip' mode):
    #   theta0_const_empty -- verdict EMPTY (c_eff < 0, or a tie decided
    #                         empty);
    #   theta0_const_drop  -- verdict DROP (c_eff > 0, or a tie decided by
    #                         the leg order to hold);
    #   theta0_tie         -- c_eff == 0.0 exactly, decided EMPTY by
    #                         Θ(0) = 0 itself (no external-time dependence);
    #   theta0_tie_ordered -- c_eff == 0.0 exactly on a difference of two
    #                         legs' times, decided by their raw times / the
    #                         leg order (``_tie_order_sign``; DROP or EMPTY).
    'theta0_const_empty': 0,
    'theta0_const_drop': 0,
    'theta0_tie': 0,
    'theta0_tie_ordered': 0,
    # δ-subset evaluations that returned 0 without integration because a
    # smooth-edge row is constant, τ-independent and EMPTY (a_int ≡ 0,
    # a_ext ≡ 0, c0 <= 0).  The subset is flagged when ``integrate_diagram``
    # builds it (in any mode) and the mode is read at call time.
    'theta0_subsets_pruned': 0,
    # m=2 polygons that survived the clip but are empty or degenerate by an
    # exact test, answered 0 ('ito' only; ``_polygon_from_2d_constraints``):
    # two rows with exactly opposite normals whose constants sum to <= 0
    # (e.g. the 2-cycle rows s1 > s0, s0 > s1, which the closed clip leaves
    # as the segment s0 = s1), or fewer than 3 vertices / an exactly zero
    # computed area.
    'polygon_zero_area': 0,
    # m≥3 posets answered 0 without integration: a constant EMPTY row, or a
    # directed cycle of order rows whose shifts sum to <= 0 (strict
    # inequalities around such a cycle cannot all hold: empty).
    'poset_empty_const': 0,
    'poset_empty_cycle': 0,
    # ── M2a structural-zero counters (``STRUCTURAL_ZEROS``) ──
    # ``_integrate_polytope`` entries (m >= 2, counted in ``nquad_calls`` /
    # ``scipy_nquad_called_*`` as before) answered 0j without quadrature:
    # a directed cycle of unit-coefficient pair rows whose shifts sum to
    # <= 0 (``_region_structurally_empty``).
    'polytope_empty_cycle': 0,
    # δ-subsets never built (per-diagram and grouped builds) because they
    # keep smooth an edge whose smooth part is provably zero (every stored
    # residue of its entry is an exact zero and its ``G_ft`` entry contains
    # no ω; for a pole-free propagator: the ``G_ft`` test alone).  Counted
    # once per subset per build, not per call.
    'forced_delta_pruned': 0,
    # M5 (P6): the per-diagram setup levers (see the section before
    # ``integrate_diagram``).  ``setup_zero_exit``: diagrams whose
    # numerically-zero prefactor returned before any setup (L1).
    'setup_zero_exit': 0,
    # L2 (``USE_SETUP_PROP_TD``): ``integrate_diagram`` calls served from the
    # per-call ``PropagatorTD``; calls handed a ``PropagatorTD`` that no
    # longer matched their propagator data / ``num_params`` (built locally);
    # lazy ``build_G_t_matrix`` builds and per-entry residue-table builds.
    'setup_prop_td_used': 0,
    'setup_prop_td_stale': 0,
    'setup_prop_td_g_t_builds': 0,
    'setup_prop_td_entry_builds': 0,
    # ── M2b hardened quadrature counters (``NQUAD_HARDENED``) ──
    # ``_integrate_polytope`` entries (m >= 1) handed to
    # ``_integrate_polytope_hardened`` (still counted in ``nquad_calls``):
    'nquad_hardened_calls': 0,
    # ... of which answered 0 without quadrature because the closure of
    # their difference rows is infeasible (an exact test; a constant row
    # <= 0 counts here too when the entry did not decide it):
    'nquad_hardened_empty': 0,
    # nested-quadrature intervals found empty (lower >= upper) and answered
    # 0 WITHOUT evaluating the integrand (one count per such interval):
    'nquad_hardened_empty_intervals': 0,
    # entries in which an open direction was truncated at NQUAD_TAIL_K / κ
    # beyond its farthest breakpoint (decay-certified truncation; κ the
    # certified decay rate along it):
    'nquad_hardened_capped': 0,
    # entries in which an open direction had no decay certificate (no mode
    # data, a mode with Re λ >= 0, or, for rows that are not unit
    # difference rows that are edges, a failed linear-programming
    # certificate) and the legacy ±200 cap was used:
    'nquad_hardened_uncertified': 0,
    # entries without mode data (epsabs from a pre-sample of the integrand,
    # open sides at the legacy cap, no geometric breakpoints):
    'nquad_hardened_no_modes': 0,
    # entries with a row that is not a difference row (exact level bounds
    # by Fourier-Motzkin elimination, breakpoints at vertex projections):
    'nquad_hardened_general_rows': 0,
    # ... of which the elimination exceeded _NQUAD_FM_MAX_ROWS rows (level
    # bounds: the difference-row closure and the rows filed by innermost
    # variable, a superset of the region's projection):
    'nquad_hardened_fm_capped': 0,
    # levels at which the vertex projections were not enumerated (more
    # than _NQUAD_VERTEX_MAX_COMBOS subsets of rows): kinks incomplete:
    'nquad_hardened_kinks_incomplete': 0,
    # scipy.integrate.quad calls of a region's final pass that still
    # returned a nonzero ``ier`` (roundoff detected, ..., or the limit
    # NQUAD_LIMIT_MAX reached) after the retries, and whose error estimate,
    # times the widths of the enclosing levels' intervals, exceeds
    # NQUAD_EPSREL × the region's scale (|result| in a second pass) and the
    # requested tolerance (or the limit reached); the value is still used
    # and a PhaseJNquadFallbackWarning reports them (once per model, source
    # and loop order):
    'nquad_hardened_quad_flags': 0,
    # quad calls that reached their subinterval limit and were repeated
    # with a larger one (every pass):
    'nquad_hardened_quad_retries': 0,
    # entries integrated a second time with epsabs from the first result
    # (its magnitude times NQUAD_EPSABS_FACTOR), the first pass's epsabs
    # (from the bound / sample of |integrand|) having been larger:
    'nquad_hardened_reruns': 0,
    # entries whose outermost open side was cut farther out (the added strip
    # integrated and added): the bound of the tail dropped by the first cut
    # (K = NQUAD_TAIL_K) was above _NQUAD_TAIL_REL × |result|; or whose
    # inner levels' cuts moved out (``region_tail``; the region integrated
    # again):
    'nquad_hardened_tail_widened': 0,
    # entries whose inner levels were cut but whose cuts could not be
    # checked (a vertex enumeration incomplete, no bound from linear
    # programming, or ``_NQUAD_TAIL_ANCHOR`` off):
    'nquad_hardened_inner_unchecked': 0,
    # entries whose second-pass absolute tolerance was set by the rounding
    # floor (1e-9 × sampled max) rather than by the result,
    # with an implied relative tolerance above 1e-8: reported in the
    # aggregated warning, never silently accepted
    'nquad_hardened_floor_limited': 0,
    # closed-form innermost integrals (``_NquadModes.integrate_innermost``)
    # whose exponentials overflowed and were integrated by quad instead:
    'nquad_hardened_innermost_overflow': 0,
    # entries whose closed-form innermost expansion would exceed
    # _NQUAD_INNERMOST_MAX_TERMS terms (s_0 integrated by quad):
    'nquad_hardened_innermost_capped': 0,
    # largest truncation distance K / κ used (0.0: none):
    'nquad_hardened_cap_span_max': 0.0,
    # ── M3 counters (``USE_DBM_FALLBACK``) ──
    # m≥3 posets refused by the inheritance rule (P3): a variable without a
    # scalar lower and without a lower-bounded predecessor in the transitive
    # closure of the strict order; the poset path bails
    # ('poset_lower_not_inherited'):
    'poset_lower_not_inherited': 0,
    # ``_integrate_subset_dbm`` calls (m≥3 subsets whose poset path bailed):
    'dbm_attempted': 0,
    # ... answered by the DBM route (a value, counted once):
    'dbm_answered': 0,
    # ... of which the region is empty or of measure zero (a constant EMPTY
    # row, or a closed-DBM cycle of weight <= 0, exact): exactly 0:
    'dbm_empty': 0,
    # ... of which the poset path had bailed on the inheritance rule (P3):
    'dbm_answered_p3': 0,
    # ... of which the first closed form failed the conditioning test and the
    # thin-interval retry passed it (``dbm_integral.integrate_exp_sum``):
    'dbm_answered_thin_retry': 0,
    # calls declined (the subset goes on to ``_integrate_polytope``): a row
    # that is not a difference row (e.g. a ConvVertex 3-term row):
    'dbm_declined_rows': 0,
    # ... a final term beyond the log-overflow limit:
    'dbm_declined_overflow': 0,
    # ... a closed form whose estimated rounding error (1e-15 × the summed
    # magnitudes of its expansion's terms) exceeds max(1e-10 |value|,
    # 1e-14 × the integrand's scale), also after the thin-interval retry:
    # close poles or a thin region (``dbm_integral.STATUS_ILL_CONDITIONED``):
    'dbm_declined_ill_conditioned': 0,
    # ... anything else (rows and modes that do not match, a non-finite
    # shift or exponent, more than dbm_integral.MAX_CASES elimination
    # cases):
    'dbm_declined_other': 0,
    # largest (estimated rounding error) / (accepted error) of an answered
    # DBM value (<= 1 by construction; 0.0: none):
    'dbm_error_ratio_max': 0.0,
}

_NQUAD_FALLBACK_REASONS = (
    'polygon_guard', 'polygon_other',
    'poset_extract_none', 'poset_no_extension', 'poset_chain_none',
    'other',
)


def _reset_runtime_counters():
    """Zero out ``_RUNTIME_COUNTERS`` before a timed run.

    Nested dict counters (``nquad_fallback_by_reason``) are reset in place
    to exactly their canonical keys, so a reference held by a caller stays
    valid.
    """
    for k in list(_RUNTIME_COUNTERS):
        v = _RUNTIME_COUNTERS[k]
        if isinstance(v, dict):
            v.clear()
            if k == 'nquad_fallback_by_reason':
                v.update((r, 0) for r in _NQUAD_FALLBACK_REASONS)
        else:
            _RUNTIME_COUNTERS[k] = 0


# ───────────────────────────────────────────────────────────────────────
# The shared Θ(0) = 0 rule for constant constraint rows (M1)
# ───────────────────────────────────────────────────────────────────────
# Row provenance (``row_kinds``, parallel to ``subset_constraint_data``;
# built by ``integrate_diagram`` and the grouped build, ``None`` = every row
# is 'edge', which is what external callers and the spatial path get):
#   'edge'         a smooth propagator edge's Δt_e > 0;
#   'conv_pseudo'  a ConvVertex kernel pseudo-edge Δt = +τ > 0 (a decaying
#                  kernel mode λ = −1/τ_g by construction: edge semantics);
#   'noise_box'    a NoiseSource τ_v box row, ±TAU_KERNEL_CAP;
#   'conv_tau'     a ConvVertex τ box row ([0, TAU_KERNEL_CAP)).
# The box kinds always keep τ as an integration variable (a_int[τ] = ±1), so
# they can never be constant (plan §7.6, M0.7: 0 occurrences in the zoo); a
# constant one would mean an unexpected δ-forcing of a kernel variable, and
# the helper raises.  No code path produces a split two-sided kernel row
# (M0.7), so there is no 'kernel_split' rule.
_EDGE_ROW_KINDS = ('edge', 'conv_pseudo')
_BOX_ROW_KINDS = ('noise_box', 'conv_tau')
ROW_KINDS = _EDGE_ROW_KINDS + _BOX_ROW_KINDS


def _row_kind(row_kinds, idx):
    """Kind of row ``idx`` (``'edge'`` when no provenance was threaded)."""
    return 'edge' if row_kinds is None else row_kinds[idx]


class _TieContext(namedtuple('_TieContext',
                             ('times', 'free_legs', 'origin_leg'))):
    """Which external legs the free external variables of ONE Wick
    permutation evaluation stand for (built by ``contribution()``).

    * ``times``      -- the raw external times, one per external leg, in the
                        caller's leg order (``contribution(*times)``),
                        exactly as passed: NOT converted to float, so a
                        time argument that is never compared (every k=1
                        call, any evaluation without an exact tie) may be
                        of any type the pre-0.2.0 code accepted;
    * ``free_legs``  -- the leg whose time each entry of ``free_ext_vals``
                        carries (same order as ``free_ext_vals``);
    * ``origin_leg`` -- the leg pinned at 0 (every free value is
                        ``times[leg] − times[origin_leg]``), or ``None``.

    ``_tie_order_sign`` uses it to decide a constant row that is exactly 0.0
    by the raw order of the two legs it compares (converting only those two
    times).  Purely descriptive: it never changes a time value.
    """
    __slots__ = ()


def _raw_time_order(t_p, t_q):
    """``t_p − t_q`` reduced to its order: a float whose sign is that of the
    difference (0.0 for equal times), NaN when the times have no order, or
    ``None`` when neither the times nor their difference is a real number.

    Float-convertible times (every API call) are compared as floats, as
    before.  Otherwise (e.g. translation-invariant symbolic times such as
    ``(t, t)``, whose free values are numeric although the times are not)
    the difference is converted instead."""
    try:
        a, b = float(t_p), float(t_q)
    except (TypeError, ValueError):
        try:
            return float(t_p - t_q)
        except (TypeError, ValueError):
            return None
    if a > b:
        return 1.0
    if a < b:
        return -1.0
    if a == b:
        return 0.0
    return float('nan')


def _tie_order_sign(a_ext, c0, tie_ctx):
    r"""Decide ``Θ(Δt)`` for a constant row whose ``c_eff`` is exactly 0.0.

    Returns ``+1`` (the row holds: DROP) or ``-1`` (it fails: EMPTY) when
    ``Δt = c0 + Σ_j a_ext_j t_j`` is the difference ``t_p − t_q`` of two
    external legs' times (``c0 == 0``; mapped to legs through ``tie_ctx``,
    including the origin leg the free values are measured from), and ``0``
    otherwise (no context, or another row shape: Θ(0) = 0 applies).

    The two legs are compared on their RAW times, so a difference that
    rounded to 0.0 only after the translation to the origin leg still gets
    its true sign, the same in every Wick permutation.  Exactly equal raw
    times are ordered by leg index: a leg with a LARGER index counts as
    infinitesimally EARLIER (the k=2 Itô left limit τ = t_1 − t_0 → 0⁻,
    extended to every tie).  Exactly one of ``Θ(t_p − t_q)`` and
    ``Θ(t_q − t_p)`` therefore holds at a tie.  (The leg index is the
    position in the caller's argument list, i.e. in ``external_fields``:
    at coincident times of legs of DIFFERENT fields the value depends on
    the order in which they are listed, exactly as the k=2 τ = 0 grid
    value does -- C_AB(τ → 0⁻) and C_BA(τ → 0⁻) are the two one-sided
    limits of the same physical function.)

    Only the two compared times are converted, and only here
    (``_raw_time_order``); times without a real order give ``0``.
    """
    if tie_ctx is None or float(c0) != 0.0:
        return 0
    free_legs = tie_ctx.free_legs
    if len(a_ext) > len(free_legs):
        return 0
    coef = {}
    total = 0.0
    for j, a in enumerate(a_ext):
        a = float(a)
        if a == 0.0:
            continue
        leg = free_legs[j]
        coef[leg] = coef.get(leg, 0.0) + a
        total += a
    if tie_ctx.origin_leg is not None and total != 0.0:
        leg = tie_ctx.origin_leg
        coef[leg] = coef.get(leg, 0.0) - total
    legs = [(leg, c) for leg, c in coef.items() if c != 0.0]
    if len(legs) != 2:
        return 0
    (leg_a, c_a), (leg_b, c_b) = legs
    if not (c_a == -c_b and abs(c_a) == 1.0):
        return 0
    p, q = (leg_a, leg_b) if c_a > 0 else (leg_b, leg_a)   # Δt = t_p − t_q
    d = _raw_time_order(tie_ctx.times[p], tie_ctx.times[q])
    if d is None or d != d:                 # no real order (NaN, complex)
        return 0
    if d > 0.0:
        return 1
    if d < 0.0:
        return -1
    return 1 if p < q else -1


def _const_row_decision(a_int, a_ext, c0, free_ext_vals, kind='edge',
                        tie_ctx=None):
    r"""The Θ(0) = 0 rule for one row, without side effects:
    ``(verdict, basis)``.

    ``verdict`` is ``None`` for a row with a nonzero normal (a genuine
    half-space), else ``'DROP'`` (the constant condition holds everywhere:
    drop it from the GEOMETRY only, the edge still contributes
    ``exp(λ_e·c_eff)`` to γ) or ``'EMPTY'`` (the region is empty).
    ``basis`` says how: ``'sign'`` (c_eff ≠ 0), ``'theta0'`` (c_eff == 0.0,
    Θ(0) = 0) or ``'tie_order'`` (c_eff == 0.0 on a two-leg time
    difference, ``_tie_order_sign``).  ``c_eff = c0 + Σ_j a_ext_j t_j`` is
    the same arithmetic as every call site.  A constant 'noise_box' /
    'conv_tau' row, or an unknown ``kind``, raises.
    """
    for a in a_int:
        if abs(float(a)) > _ROW_COEF_ATOL:
            return None, None
    if kind not in _EDGE_ROW_KINDS:
        if kind in _BOX_ROW_KINDS:
            raise ValueError(
                f'constant {kind!r} constraint row (a_int={list(a_int)!r}, '
                f'a_ext={list(a_ext)!r}, c0={c0!r}): a kernel/box '
                f'integration variable was eliminated by a δ edge, which no '
                f'model is expected to produce (plan §7.6); refusing to '
                f'guess its Θ(0) convention')
        raise ValueError(f'unknown constraint-row kind {kind!r}; expected '
                         f'one of {ROW_KINDS}')
    c_eff = float(c0) + sum(float(a_ext[j]) * float(free_ext_vals[j])
                            for j in range(len(a_ext)))
    if c_eff > 0.0:
        return 'DROP', 'sign'
    if c_eff != 0.0:                        # < 0 (or NaN): empty
        return 'EMPTY', 'sign'
    s = _tie_order_sign(a_ext, c0, tie_ctx)
    if s > 0:
        return 'DROP', 'tie_order'
    if s < 0:
        return 'EMPTY', 'tie_order'
    return 'EMPTY', 'theta0'                # THETA_AT_ZERO_CONST_ROW = 0


def _const_row_verdict(a_int, a_ext, c0, free_ext_vals, kind='edge',
                       tie_ctx=None):
    r"""Θ(0) = 0 verdict for one constraint row ``a_int·s + c_eff > 0``.

    Returns ``None`` if the row has a nonzero normal (some
    ``|a_int_j| > _ROW_COEF_ATOL``): it is a genuine half-space, not this
    helper's business.  Otherwise the row is the constant condition
    ``c_eff > 0`` with ``c_eff = c0 + Σ_j a_ext_j t_j``, and the verdict is

    * ``'DROP'``  if ``c_eff > 0``: the condition holds everywhere.  Drop it
      from the GEOMETRY only -- the row stays in the per-edge data, so the
      edge still contributes ``exp(λ_e·c_eff)`` to γ;
    * ``'EMPTY'`` if ``c_eff < 0``: the region is empty;
    * a tie, ``c_eff == 0.0`` exactly: ``'EMPTY'`` by Θ(0) = 0
      (``THETA_AT_ZERO_CONST_ROW``, the Itô convention), except a difference
      of two external legs' times, which ``tie_ctx`` orders
      (``_tie_order_sign``: exactly one orientation holds).

    There is no tie tolerance (see the THETA_AT_ZERO_CONST_ROW comment).
    ``kind`` is the row's provenance (see ``ROW_KINDS``): edge kinds get the
    rule above; a constant 'noise_box' / 'conv_tau' row raises.  Counts
    ``theta0_*`` in ``_RUNTIME_COUNTERS`` (``_const_row_decision`` is the
    side-effect-free form).

    Mode-agnostic: callers select legacy behaviour themselves
    (``_theta0_legacy()``) and never call this in 'legacy_clip' mode.
    """
    verdict, basis = _const_row_decision(a_int, a_ext, c0, free_ext_vals,
                                         kind, tie_ctx)
    if verdict is None:
        return None
    if verdict == 'DROP':
        _RUNTIME_COUNTERS['theta0_const_drop'] += 1
    else:
        _RUNTIME_COUNTERS['theta0_const_empty'] += 1
    if basis == 'theta0':
        _RUNTIME_COUNTERS['theta0_tie'] += 1
    elif basis == 'tie_order':
        _RUNTIME_COUNTERS['theta0_tie_ordered'] += 1
    return verdict


def _any_const_row_empty(subset_constraint_data, free_ext_vals,
                         row_kinds=None, tie_ctx=None):
    """True if some constant row of the subset has verdict ``'EMPTY'``."""
    for idx, (a_int, a_ext, c0) in enumerate(subset_constraint_data):
        if _const_row_verdict(a_int, a_ext, c0, free_ext_vals,
                              _row_kind(row_kinds, idx),
                              tie_ctx) == 'EMPTY':
            return True
    return False


# ───────────────────────────────────────────────────────────────────────
# Analytic-path bail reasons (M0.1; observational)
# ───────────────────────────────────────────────────────────────────────
# Every ``return None`` of the three analytic modesum integrators (and of the
# grouped m0/m1 helpers) goes through ``_bail(reason)``, which records WHY the
# path gave up and returns ``None`` -- control flow is unchanged.  The
# dispatch closures read the reason with ``_pop_bail_reason()`` only after a
# ``None``, to fill ``nquad_fallback_by_reason`` and the ``_SUBSET_HOOK``
# payload.  Thread-local so a threaded caller cannot mislabel another
# thread's fallback.
import threading as _threading

_BAIL_STATE = _threading.local()


def _bail(reason):
    """Record ``reason`` as the latest analytic-path bail; return ``None``."""
    _BAIL_STATE.reason = reason
    return None


def _pop_bail_reason():
    """Return and clear the latest bail reason of this thread (or ``None``)."""
    reason = getattr(_BAIL_STATE, 'reason', None)
    _BAIL_STATE.reason = None
    return reason


# Fine-grained bail reason -> ``nquad_fallback_by_reason`` key.
_BAIL_REASON_CATEGORY = {
    'polygon_triangle_guard': 'polygon_guard',
    'poset_extract_none': 'poset_extract_none',
    'poset_no_extension': 'poset_no_extension',
    'poset_chain_none': 'poset_chain_none',
}


def _bail_category(reason):
    """Map a fine-grained bail reason (or ``None``) to a counter key."""
    cat = _BAIL_REASON_CATEGORY.get(reason)
    if cat is not None:
        return cat
    if reason is not None and reason.startswith('polygon_'):
        return 'polygon_other'
    return 'other'


def _count_nquad_fallback(reason, m):
    """Dispatch-side counter: an m≥1 subset call is about to reach
    ``_integrate_polytope`` because of ``reason`` (``None`` = no analytic
    path was attempted).  m=0 calls are not counted (not a fallback)."""
    if m >= 1:
        _RUNTIME_COUNTERS['nquad_fallback_by_reason'][
            _bail_category(reason)] += 1


# ───────────────────────────────────────────────────────────────────────
# Per-subset observation hook (M0.1; docs/integration_speedup_plan.md §4.1)
# ───────────────────────────────────────────────────────────────────────
# ``_SUBSET_HOOK`` is ``None`` in production.  When set to a callable, the
# per-diagram dispatch (``integrate_diagram``'s ``_contrib``) and the grouped
# dispatch (``grouped_integral.integrate_grouped_diagram``'s ``_contrib``)
# call ``_SUBSET_HOOK(payload)`` once per (diagram, δ-subset, free_ext_vals,
# Wick permutation) evaluation, AFTER computing the value, with a dict
# payload built by ``_emit_subset_hook``.  It is read at CALL TIME through
# the module global (never a default argument), so
# ``final_integral._SUBSET_HOOK = f`` takes effect immediately, including
# for closures built before it was set.  With the hook ``None`` the only
# cost is one global ``is None`` check per subset call and one per
# ``contribution()`` call; values are bit-identical either way (the hook
# only observes -- it must not mutate the payload's referenced objects).
#
# Contract: subset contributions are evaluated in the calling process.  A
# hook set in a parent process does not see calls made in forked
# ``total_C_batch`` workers -- run with ``parallel=False`` when observing.
# Exceptions raised by the hook propagate (it is a diagnostic tool).  The
# consumer is ``tests/tools/phase_j_subset_diff.py``.
_SUBSET_HOOK = None

import itertools as _itertools_hook
_DIAGRAM_SERIAL = _itertools_hook.count()
_HOOK_EVAL_SERIAL = _itertools_hook.count()
_HOOK_CALL_SERIAL = _itertools_hook.count()


def _next_diagram_serial():
    """Monotone id for one ``integrate_diagram`` / grouped-group build."""
    return next(_DIAGRAM_SERIAL)


def _next_hook_call_serial():
    """Monotone id for one hooked ``contribution(*ext_time_values)`` call."""
    return next(_HOOK_CALL_SERIAL)


def _hook_eval_context(diagram_serial, ext_time_values, perm_index, perm,
                       n_perms, compensation, call_serial=None):
    """Per-(contribution() call, Wick perm) context handed to ``_contrib``
    when the hook is set.  ``call_serial`` is shared by every perm of one
    ``contribution()`` call; ``eval_serial`` is unique per (call, perm)."""
    return {
        'call_serial': call_serial,
        'eval_serial': next(_HOOK_EVAL_SERIAL),
        'diagram_serial': diagram_serial,
        'ext_time_values': tuple(float(x) for x in ext_time_values),
        'perm_index': perm_index,
        'perm': tuple(perm),
        'n_perms': n_perms,
        'compensation': compensation,
    }


def _modes_summary(modes, pole_tuples):
    """Small, picklable summary of the smooth-edge pole/residue data."""
    out = {'n_smooth': None, 'n_modes_per_edge': None, 'edge_modes': None,
           'lambdas': None, 'n_pole_tuples': None,
           'max_abs_tuple_coeff': None}
    try:
        if modes is not None:
            out['n_smooth'] = len(modes)
            out['n_modes_per_edge'] = [len(ms.modes) for ms in modes]
            out['edge_modes'] = [
                [(complex(C), complex(lam)) for (C, lam) in ms.modes]
                for ms in modes
            ]
            out['lambdas'] = sorted(
                {complex(lam) for ms in modes for (_C, lam) in ms.modes},
                key=lambda z: (z.real, z.imag),
            )
        if pole_tuples is not None:
            out['n_pole_tuples'] = len(pole_tuples)
            out['max_abs_tuple_coeff'] = max(
                (abs(complex(C)) for (C, _l) in pole_tuples), default=0.0)
    except Exception as exc:          # observational only: never raise
        out['error'] = repr(exc)
    return out


def _emit_subset_hook(hook, meta, ctx, free_vals, m, cdata, *, path,
                      evaluator, branch, value, bail_reason, attempted,
                      modes, prefactor, plan, pole_tuples, integrand,
                      row_kinds=None, tie_ctx=None):
    """Build the ``_SUBSET_HOOK`` payload and call the hook.

    Only ever called when the hook is set.  Payload keys:

    * identity: ``source`` ('per_diagram' | 'grouped'), ``diagram_serial``,
      ``diagram`` (TypedDiagram, or the list of them for a grouped build),
      ``loop_number``, ``subset_id`` (δ-subset bitmask), ``subset_index``
      (position in the diagram's subset list), ``delta_edges``,
      ``smooth_edges``; plus the Wick/evaluation context ``ctx`` (``None``
      when the subset closure is called outside ``contribution()``):
      ``eval_serial``, ``ext_time_values``, ``perm_index``, ``perm``,
      ``n_perms``, ``compensation``.
    * geometry: ``m``, ``constraints`` (the subset_constraint_data rows
      ``(a_int, a_ext, c0)``), ``row_kinds`` (M1 row provenance: a tuple
      parallel to ``constraints``, entries from ``ROW_KINDS``),
      ``free_ext_vals``, ``tie_ctx`` (the ``_TieContext`` the constant rows
      were decided with -- ``None`` outside ``contribution()``).
    * integrand: ``modes_summary`` (picklable), and live references
      ``modes`` / ``plan`` / ``pole_tuples`` / ``integrand`` (the closure
      scipy.nquad integrates) / ``prefactor``.  The live references are
      NOT picklable; copy what you need.
    * outcome: ``path`` ('m0' | 'm1' | 'polygon' | 'poset' | 'dbm' |
      'nquad'; 'dbm': the M3 DBM route answered after the poset path
      bailed),
      ``evaluator`` (function-level name), ``branch`` ('plan' | 'noplan' |
      'grouped' | None), ``attempted`` (analytic path tried, or ``None``),
      ``bail_reason`` (fine-grained reason the analytic path returned
      ``None``; 'not_eligible' when none was attempted for an m≥1 call),
      ``bail_category`` (the ``nquad_fallback_by_reason`` key, or ``None``
      when an analytic path answered), ``value`` (complex, exactly what
      the dispatch returns).
    """
    summ = meta.get('_modes_summary')
    if summ is None:
        summ = _modes_summary(modes, pole_tuples)
        meta['_modes_summary'] = summ
    loop_number = meta.get('loop_number')
    if loop_number is None and meta.get('diagram') is not None:
        try:
            loop_number = _loop_number_from_graph(meta['diagram'])
        except Exception:
            loop_number = None
        meta['loop_number'] = loop_number
    answered_by_analytic = path != 'nquad' and not (
        path == 'm0' and evaluator == 'polytope_m0')
    payload = {
        'source': meta.get('source'),
        'diagram_serial': meta.get('diagram_serial'),
        'diagram': meta.get('diagram'),
        'loop_number': loop_number,
        'subset_id': meta.get('subset_id'),
        'subset_index': meta.get('subset_index'),
        'delta_edges': meta.get('delta_edges'),
        'smooth_edges': meta.get('smooth_edges'),
        'ctx': ctx,
        'm': m,
        'constraints': cdata,
        'row_kinds': (None if row_kinds is None
                      else tuple(row_kinds)),
        'free_ext_vals': tuple(float(x) for x in free_vals),
        'tie_ctx': tie_ctx,
        'modes_summary': summ,
        'modes': modes,
        'plan': plan,
        'pole_tuples': pole_tuples,
        'integrand': integrand,
        'prefactor': prefactor,
        'path': path,
        'evaluator': evaluator,
        'branch': branch,
        'attempted': attempted,
        'bail_reason': bail_reason,
        'bail_category': (None if answered_by_analytic or m < 1
                          else _bail_category(
                              None if bail_reason == 'not_eligible'
                              else bail_reason)),
        'value': value,
    }
    hook(payload)

# ───────────────────────────────────────────────────────────────────────
# Analytic ∫_L^U exp(α·s + γ) ds  (Stage 4a-perdiag, m=1)
# ───────────────────────────────────────────────────────────────────────
# Per-pole-tuple closed-form 1D exponential integral.  Replaces
# scipy.quad on the pole-residue closure callable for m=1 subsets.
# Same correctness guarantees as the polygon/poset paths (exact for
# rational propagators).
USE_1D_INTEGRATOR = True
# Threshold below which we switch to a 4th-order Taylor expansion of
# J(p, q) to avoid catastrophic cancellation in the formula's denominator.
_J_TAYLOR_EPS = 1e-6


def _exp_over_unit_triangle(p, q):
    r"""Closed-form value of

        J(p, q) = ∫₀¹ du ∫₀^{1-u} dw  exp(p·u + q·w)

    for complex ``p``, ``q``.

    Derivation: do the w-integral first to get
    ``∫₀¹ du exp(p·u) · (exp(q(1-u)) − 1) / q``, then split and
    integrate in u.  Falls back to a 4th-order Taylor expansion when
    any of |p|, |q|, |p − q| drops below ``_J_TAYLOR_EPS`` to avoid
    catastrophic cancellation.
    """
    import cmath
    eps = _J_TAYLOR_EPS
    abs_p = abs(p)
    abs_q = abs(q)
    abs_pq = abs(p - q)

    if abs_p < eps and abs_q < eps:
        # 4th-order Taylor of the double integral.  Coefficients are
        # the moments of the unit triangle:
        #   ∫∫ 1 = 1/2
        #   ∫∫ u  = ∫∫ w = 1/6
        #   ∫∫ u² = ∫∫ w² = 1/12,   ∫∫ uw = 1/24
        #   ∫∫ u³ = ∫∫ w³ = 1/20,   ∫∫ u²w = ∫∫ uw² = 1/60
        return (0.5
                + (p + q) / 6.0
                + (p * p) / 24.0 + (q * q) / 24.0 + (p * q) / 24.0
                + (p**3 + q**3) / 120.0
                + (p * p * q + p * q * q) / 120.0)

    if abs_p < eps:
        # J(0, q) = (e^q − 1 − q) / q²
        return (cmath.exp(q) - 1 - q) / (q * q)

    if abs_q < eps:
        # J(p, 0) = (e^p − 1 − p) / p²
        return (cmath.exp(p) - 1 - p) / (p * p)

    if abs_pq < eps:
        # J(p, p) = ((p − 1) e^p + 1) / p²
        return ((p - 1) * cmath.exp(p) + 1) / (p * p)

    # General case.
    ep = cmath.exp(p)
    eq = cmath.exp(q)
    return ((ep - eq) / (p - q) - (ep - 1) / p) / q


def _exp_over_triangle(v0, v1, v2, alpha, beta):
    r"""``∫∫_T exp(α·x + β·y) dA`` for triangle ``T = (v0, v1, v2)``.

    The vertices ``v0``, ``v1``, ``v2`` are 2-tuples of floats; ``α``,
    ``β`` are complex.  Affine-maps to the unit triangle and uses
    ``_exp_over_unit_triangle`` for the closed-form integral.

    Returns ``None`` if the per-term exponential would overflow IEEE
    double range — the polygon-modesum caller treats ``None`` as a
    signal to abort the analytic path and fall back to scipy.nquad.
    """
    import cmath
    e1x = v1[0] - v0[0]
    e1y = v1[1] - v0[1]
    e2x = v2[0] - v0[0]
    e2y = v2[1] - v0[1]
    det = e1x * e2y - e1y * e2x  # signed parallelogram area
    if abs(det) < 1e-15:
        return 0.0 + 0.0j  # degenerate
    p = alpha * e1x + beta * e1y
    q = alpha * e2x + beta * e2y
    v0_term = alpha * v0[0] + beta * v0[1]
    # Overflow guard — bilateral.  ``cmath.exp(z)`` blows past double
    # range for Re(z) > ~709 (overflow) AND underflows to 0 for Re(z)
    # < ~-745.  For most analytic integrators the underflow direction
    # is harmless (term decays correctly), but ``_exp_over_unit_
    # triangle`` (called below) has structural cancellation —
    # ``(exp(p) - exp(q)) / (p - q) - (exp(p) - 1) / p`` — which can
    # produce floating-point-noise-dominated values when one of
    # ``exp(p)`` or ``exp(q)`` underflows and the other doesn't.
    # Stage 4a opt #3 (2026-05-15 commit 7f0bf05) relaxed this to one-
    # sided which appears to have introduced wrong-direction loop
    # corrections in spike-reset k=2 ell=1 (m=2 polygon path is
    # exercised heavily there).  Reverting per Agent 4's audit
    # recommendation: keep the guard bilateral so previously-bailed
    # cases route to scipy.nquad as they did in Stage 3b.
    EXP_REAL_LIMIT = 600.0
    if (abs(p.real) > EXP_REAL_LIMIT
            or abs(q.real) > EXP_REAL_LIMIT
            or abs(v0_term.real) > EXP_REAL_LIMIT):
        return None
    try:
        J = _exp_over_unit_triangle(p, q)
        return abs(det) * cmath.exp(v0_term) * J
    except (OverflowError, ValueError):
        return None


def _clip_polygon_to_halfplane(polygon, a, b, c):
    r"""Sutherland-Hodgman clip a CCW polygon against the half-plane
    ``a·x + b·y + c > 0``.

    ``polygon`` is a list of ``(x, y)`` tuples.  Returns the clipped
    polygon as a new list; may be empty if the polygon lies entirely
    in the rejected half-space.

    Vertices exactly on the boundary (``a·x + b·y + c = 0``) are
    treated as "on the inside" — for analytic integration over the
    polygon interior, the measure-zero boundary contribution is 0
    regardless of which side we assign.

    The normal ``(a, b)`` must be nonzero.  A zero normal makes the
    condition the constant ``c > 0``, which this closed-half-plane clip
    would treat as Θ(0) = 1 at ``c == 0`` (keeping the whole polygon);
    constant rows are decided by ``_const_row_verdict`` instead (M1).
    """
    if a == 0.0 and b == 0.0:
        raise ValueError(
            '_clip_polygon_to_halfplane: zero normal (a = b = 0); constant '
            'constraint rows must go through _const_row_verdict')
    if not polygon:
        return []
    n = len(polygon)
    output = []
    prev = polygon[-1]
    prev_f = a * prev[0] + b * prev[1] + c
    for i in range(n):
        curr = polygon[i]
        curr_f = a * curr[0] + b * curr[1] + c
        prev_in = prev_f >= 0.0
        curr_in = curr_f >= 0.0
        if curr_in:
            if not prev_in:
                # Edge enters: add intersection.
                t = prev_f / (prev_f - curr_f)
                ix = prev[0] + t * (curr[0] - prev[0])
                iy = prev[1] + t * (curr[1] - prev[1])
                output.append((ix, iy))
            output.append(curr)
        else:
            if prev_in:
                # Edge exits: add intersection.
                t = prev_f / (prev_f - curr_f)
                ix = prev[0] + t * (curr[0] - prev[0])
                iy = prev[1] + t * (curr[1] - prev[1])
                output.append((ix, iy))
            # curr discarded
        prev = curr
        prev_f = curr_f
    return output


def _polygon_from_2d_constraints(constraint_data, free_ext_vals, bbox_cap,
                                 row_kinds=None, tie_ctx=None):
    r"""Build the 2D convex polygon defined by the retardation
    constraints, starting from a CCW bounding box ``±bbox_cap`` and
    clipping with each constraint.

    Each constraint is ``(a_int, a_ext, c0)`` with
    ``a_int·(s_0, s_1) + (c0 + a_ext·free_ext_vals) > 0``.

    Returns a CCW polygon vertex list (possibly empty if the
    intersection is empty).

    Constant rows (zero normal) follow ``THETA0_CONST_ROW_MODE`` (call
    time).  'ito': ``_const_row_verdict`` decides -- EMPTY returns ``[]``,
    DROP skips only the clip (the row stays in the caller's per-edge / γ
    data).  After the clip, 'ito' also returns ``[]`` (counted in
    ``polygon_zero_area``) for a region that is empty or degenerate by an
    EXACT test, never a tolerance:

    * two rows with exactly opposite normals, ``a_j == −a_i``, and
      ``c_i + c_j <= 0.0`` (``_opposite_rows_empty``): ``a·s > −c_i`` and
      ``a·s < c_j`` cannot both hold.  E.g. the 2-cycle rows s1 > s0,
      s0 > s1, which the closed clip leaves as the segment s0 = s1, or a
      strip between two exactly coincident external times;
    * a clipped polygon with fewer than 3 vertices or an exactly zero
      computed area (``_polygon_has_zero_area``).

    A genuine strip, however thin (e.g. between external times 1e-12
    apart), is integrated exactly as before 0.2.0.
    'legacy_clip': the pre-M1 closed-half-plane clip, bit-for-bit (a
    zero-normal row with c_eff >= 0 keeps the polygon: Θ(0) = 1).
    ``row_kinds`` is the rows' provenance (``None``: all 'edge');
    ``tie_ctx`` (a ``_TieContext`` or ``None``) orders an exact external-time
    tie (``_tie_order_sign``).
    """
    polygon = [
        (-bbox_cap, -bbox_cap),
        (bbox_cap, -bbox_cap),
        (bbox_cap, bbox_cap),
        (-bbox_cap, bbox_cap),
    ]
    legacy = _theta0_legacy()
    clip_rows = []
    for idx, (a_int, a_ext, c0) in enumerate(constraint_data):
        c_eff = float(c0) + sum(
            float(a_ext[j]) * float(free_ext_vals[j])
            for j in range(len(a_ext))
        )
        a0 = float(a_int[0])
        a1 = float(a_int[1])
        if abs(a0) <= _ROW_COEF_ATOL and abs(a1) <= _ROW_COEF_ATOL:
            # Observational (M0.1): a zero-normal row reaching the clip.
            _RUNTIME_COUNTERS['zero_normal_rows_seen'] += 1
            if not legacy:
                verdict = _const_row_verdict(
                    (a0, a1), a_ext, c0, free_ext_vals,
                    _row_kind(row_kinds, idx), tie_ctx)
                if verdict == 'EMPTY':
                    return []
                continue            # DROP: no clip; γ keeps exp(λ·c_eff)
            if a0 == 0.0 and a1 == 0.0:
                # legacy_clip: exactly what the closed-half-plane clip does
                # with f ≡ c_eff (all vertices kept iff c_eff >= 0).
                if not (c_eff >= 0.0):
                    return []
                continue
        polygon = _clip_polygon_to_halfplane(polygon, a0, a1, c_eff)
        if not polygon:
            return []
        clip_rows.append((a0, a1, c_eff))
    if not legacy and (_opposite_rows_empty(clip_rows)
                       or _polygon_has_zero_area(polygon)):
        _RUNTIME_COUNTERS['polygon_zero_area'] += 1
        return []
    return polygon


def _opposite_rows_empty(rows):
    r"""True if two of the half-planes ``a0·s0 + a1·s1 + c > 0`` in ``rows``
    (``(a0, a1, c)`` triples, nonzero normals) have EXACTLY opposite
    normals, ``(a0_j, a1_j) == (−a0_i, −a1_i)``, and ``c_i + c_j <= 0.0``.
    Then ``a·s > −c_i`` and ``a·s < c_j`` cannot both hold: the region is
    empty (with ``c_i + c_j == 0.0`` the closed clip would leave a segment).
    Exact comparisons only; a pair with ``c_i + c_j > 0`` is a genuine
    strip, however thin, and is left to the clip."""
    n = len(rows)
    for i in range(n):
        a0, a1, ci = rows[i]
        for j in range(i + 1, n):
            b0, b1, cj = rows[j]
            if b0 == -a0 and b1 == -a1 and ci + cj <= 0.0:
                return True
    return False


def _polygon_has_zero_area(polygon):
    r"""True if the clipped polygon is degenerate: fewer than 3 vertices, or
    twice its area (summed over the fan from ``polygon[0]``) is exactly
    0.0.  No tolerance: a genuine thin polygon keeps its (tiny) area and is
    integrated as before 0.2.0."""
    n = len(polygon)
    if n < 3:
        return True
    x0, y0 = polygon[0]
    area2 = 0.0
    for i in range(1, n - 1):
        ax = polygon[i][0] - x0
        ay = polygon[i][1] - y0
        bx = polygon[i + 1][0] - x0
        by = polygon[i + 1][1] - y0
        area2 += ax * by - ay * bx
    return area2 == 0.0


def _enumerate_pole_tuples(edge_mode_sums):
    r"""Cartesian product over per-edge modes.

    Yields ``(C_product, lambdas)`` per pole tuple:
      * ``C_product``: complex, the product of residues ``∏_e C_α_e``
      * ``lambdas``:  tuple of complex ``(λ_α₁, λ_α₂, …)`` — one per
        smooth edge, in the same order as ``edge_mode_sums``.

    For ``len(edge_mode_sums) == 0`` yields exactly one tuple
    ``(1.0+0j, ())`` representing the empty product / no exponential.
    """
    if not edge_mode_sums:
        yield (1.0 + 0.0j, ())
        return
    n_edges = len(edge_mode_sums)
    mode_counts = [len(ems.modes) for ems in edge_mode_sums]
    # Stack-based product (avoids itertools.product overhead).
    idx = [0] * n_edges
    while True:
        C_prod = 1.0 + 0.0j
        lambdas = []
        for e in range(n_edges):
            C, lam = edge_mode_sums[e].modes[idx[e]]
            C_prod *= C
            lambdas.append(lam)
        yield C_prod, tuple(lambdas)
        # Advance index.
        e = 0
        while e < n_edges:
            idx[e] += 1
            if idx[e] < mode_counts[e]:
                break
            idx[e] = 0
            e += 1
        else:
            return  # all overflowed


# ───────────────────────────────────────────────────────────────────────
# Per-subset plan cache (Stage 4a-plan, 2026-05-15)
# ───────────────────────────────────────────────────────────────────────
# All three analytic modesum integrators (m=1 interval, m=2 polygon,
# m≥3 poset) iterate over pole tuples and per tuple compute
#
#   α_s[v]     = Σ_e λ_e · a_int_e[v]      for v in 0..m-1
#   γ_const[t] = Σ_e λ_e · c0_e
#   γ_slope[t][j] = Σ_e λ_e · a_ext_e[j]   for j in 0..n_ext-1
#
# and finally
#
#   γ(free_vals) = γ_const + Σ_j γ_slope[j] · free_vals[j].
#
# ``λ_e``, ``a_int_e[v]``, ``c0_e``, ``a_ext_e[j]`` are all functions of
# ``(smooth_edge_modes, subset_constraint_data)`` — strictly per-subset.
# ``free_vals`` is the only τ-grid-varying input.  Previously each
# integrator rebuilt the per-tuple α_s / γ_const / γ_slope on every
# ``_contrib(free_vals)`` call, which compounds linearly across the τ
# grid.  The plan caches them once at subset setup and threads the
# cached arrays into the per-call inner loop, so the per-τ work
# reduces to the much cheaper
#
#   γ_per_tuple[t] = γ_const[t] + Σ_j γ_slope[t][j] · free_vals[j]
#
# while reusing α_s, polygon-vertex constraints, and the EdgeModeSum
# cache.  Scales with (N_τ − 1) — biggest win on τ-dense sweeps
# (k=1 max_ell=2 with 10+ probes, etc.).
def _build_modesum_plan(smooth_edge_modes, subset_constraint_data,
                        m, n_ext):
    """Pre-compute the τ-invariant per-pole-tuple data used by the
    analytic modesum integrators.

    Returns a dict with the following keys; all values are tuples
    (immutable, cheap to share across closures):

    ``pole_tuples``: tuple of (C_prod, lambdas)
        Pre-enumerated cartesian product over per-edge modes; identical
        contents to ``_enumerate_pole_tuples(smooth_edge_modes)`` but
        materialised so iteration over the τ grid pays the construction
        cost once.

    ``alphas_per_tuple``: tuple, len = n_tuples
        Each entry is a tuple of ``m`` complex values
        ``(α_s[0], …, α_s[m-1])`` for the corresponding pole tuple.
        Replaces the per-call inner edge-loop
        ``α_s[v] += λ_e · a_int_e[v]``.

    ``gamma_const_per_tuple``: tuple, len = n_tuples
        Each entry is the complex constant ``Σ_e λ_e · c0_e`` for that
        pole tuple — i.e. γ at ``free_vals = 0``.

    ``gamma_slope_per_tuple_per_ext``: tuple of tuples
        ``slope[t][j] = Σ_e λ_e · a_ext_e[j]`` for tuple t, external-
        time index j.  γ at general free_vals is then
        ``γ_const[t] + Σ_j slope[t][j] · free_vals[j]``.
    """
    pole_tuples = tuple(_enumerate_pole_tuples(smooth_edge_modes))
    # Pre-extract per-edge linear coefficients in floats.  The arity
    # of ``a_int_e`` is ``m`` and ``a_ext_e`` is ``n_ext``.  We tolerate
    # short ``a_int`` lists (degenerate constraints with no internal
    # coefficients) by padding with 0.0.
    a_int_per_edge = []
    a_ext_per_edge = []
    c0_per_edge = []
    for (a_int, a_ext, c0) in subset_constraint_data:
        a_int_pad = tuple(
            float(a_int[v]) if v < len(a_int) else 0.0 for v in range(m)
        )
        a_ext_pad = tuple(
            float(a_ext[j]) if j < len(a_ext) else 0.0
            for j in range(n_ext)
        )
        a_int_per_edge.append(a_int_pad)
        a_ext_per_edge.append(a_ext_pad)
        c0_per_edge.append(float(c0))

    alphas_per_tuple = []
    gamma_const_per_tuple = []
    gamma_slope_per_tuple_per_ext = []
    n_smooth = len(smooth_edge_modes)
    for (C_prod, lambdas) in pole_tuples:
        alphas = [0.0 + 0.0j] * m
        gamma_const = 0.0 + 0.0j
        gamma_slope = [0.0 + 0.0j] * n_ext
        for e in range(n_smooth):
            lam = lambdas[e]
            for v in range(m):
                alphas[v] += lam * a_int_per_edge[e][v]
            gamma_const += lam * c0_per_edge[e]
            for j in range(n_ext):
                gamma_slope[j] += lam * a_ext_per_edge[e][j]
        alphas_per_tuple.append(tuple(alphas))
        gamma_const_per_tuple.append(gamma_const)
        gamma_slope_per_tuple_per_ext.append(tuple(gamma_slope))

    # ── group tuples that share an alpha vector ──────────────────────
    # The chain-simplex integral depends on the pole tuple ONLY through
    # ``alphas`` -- ``C_prod`` and gamma enter as a scalar prefactor.  So
    # tuples with identical alphas yield identical chain values and can be
    # evaluated once, with their prefactors summed.
    #
    # This is where multi-pole cost actually lives: the tuple count is a
    # cartesian product, n_poles ** n_edges (measured: 6 tuples at one pole
    # vs 211,968 at two, k=4 ell=1), but the DISTINCT alpha vectors grow far
    # more slowly (2,240,064 chain calls collapsing onto 4,729 distinct sums
    # on a two-field model).  Grouping converts the per-tuple work into
    # per-group work.
    #
    # Keys are the EXACT complex tuples, never rounded: identical alphas give
    # identical chain values, whereas near-equal ones do not, and merging
    # those would silently change the answer.
    groups = {}
    for t_idx, alphas in enumerate(alphas_per_tuple):
        groups.setdefault(alphas, []).append(t_idx)
    tuple_groups = tuple((alphas, tuple(idxs))
                         for alphas, idxs in groups.items())

    return {
        'pole_tuples':                  pole_tuples,
        'alphas_per_tuple':             tuple(alphas_per_tuple),
        'gamma_const_per_tuple':        tuple(gamma_const_per_tuple),
        'gamma_slope_per_tuple_per_ext': tuple(gamma_slope_per_tuple_per_ext),
        'tuple_groups':                 tuple_groups,
    }


# ───────────────────────────────────────────────────────────────────────
# Causal poset extraction + linear-extension enumeration (Stage 3b prep)
# ───────────────────────────────────────────────────────────────────────
# For m ≥ 3 (deeply-nested integrals), the retardation constraint set
# typically forms a directed acyclic graph (DAG) on the integration
# variables — each constraint  s_v − s_u > 0  is an edge ``u → v``.
# The full polytope decomposes into a disjoint union of simplex
# regions, one per linear extension (topological sort) of the DAG.
# Each simplex region has the form
#
#     L ≤ s_{σ(1)} ≤ s_{σ(2)} ≤ … ≤ s_{σ(N)} ≤ U
#
# (when all variables share the same scalar lower bound L and the
# top variable has a scalar upper bound U).  The integral over each
# region factors into nested 1D exponential integrals — a closed
# form lives in ``_exp_over_chain`` (Stage 3b-nested).
#
# This file implements the structural part: extract the DAG + scalar
# bounds from the constraint list, enumerate linear extensions.

@dataclass(frozen=True)
class _CausalPoset:
    """DAG on ``m`` integration variables plus per-variable scalar
    bounds, extracted from a list of retardation constraints.

    Fields
    ------
    m : int
        Number of integration variables.
    edges : tuple of (int, int)
        Pairs ``(u, v)`` with ``s_v > s_u``.  Duplicate edges removed.
    scalar_lowers : tuple of (int, float)
        ``(var_idx, c)`` meaning ``s_var > c``.  Multiple lower bounds
        on the same variable are kept; the effective lower is the
        max.
    scalar_uppers : tuple of (int, float)
        ``(var_idx, c)`` meaning ``s_var < c``.  Effective upper is
        the min.

    Notes
    -----
    Edge (u, v) means u precedes v in the integration time ordering.
    """
    m: int
    edges: tuple
    scalar_lowers: tuple
    scalar_uppers: tuple


class _EmptyPosetSentinel:
    """Type of ``_EMPTY_POSET`` (private to FI's poset-modesum path)."""
    __slots__ = ()

    def __repr__(self):
        return '_EMPTY_POSET'


# Returned by ``_extract_causal_poset`` (THETA0_CONST_ROW_MODE 'ito' only)
# when the region is EMPTY by construction: a constant row with verdict
# EMPTY (Θ(0) = 0), or a directed cycle of order rows whose shifts sum to
# <= 0 (s_a > s_b + … > s_a with strict inequalities cannot all hold).  The
# poset-modesum caller returns 0j for it.  ``_CausalPoset`` and
# ``_enumerate_linear_extensions`` are untouched
# (``spatial/causal_chambers.py`` imports them).
_EMPTY_POSET = _EmptyPosetSentinel()


def _order_rows_infeasible(m, order_rows):
    r"""True if the strict order rows cannot all hold.

    ``order_rows`` holds ``(lo, up, c)`` for rows ``s_up − s_lo + c > 0``,
    i.e. the difference constraints ``s_lo − s_up < c``.  Such a system is
    infeasible exactly when some directed cycle of constraints has total
    shift ``Σ c <= 0`` (summing the rows around the cycle gives
    ``0 < Σ c``).  Floyd–Warshall on the m variables; exact for exact
    shifts (an unshifted cycle sums to 0.0 exactly).  Any cycle it finds is
    also a cycle of the unshifted order edges, so the poset has no linear
    extension: a subset decided here would otherwise have bailed to the
    scipy.nquad fallback (which integrates an identically-zero filtered
    integrand over it).
    """
    if len(order_rows) < 2:
        return False
    inf = math.inf
    dist = [[inf] * m for _ in range(m)]
    for (lo, up, c) in order_rows:
        # s_lo − s_up < c: an edge up -> lo of weight c.
        if c < dist[up][lo]:
            dist[up][lo] = c
    for k in range(m):
        dk = dist[k]
        for i in range(m):
            dik = dist[i][k]
            if dik == inf:
                continue
            di = dist[i]
            for j in range(m):
                v = dik + dk[j]
                if v < di[j]:
                    di[j] = v
    return any(dist[i][i] <= 0.0 for i in range(m))


def _region_structurally_empty(s_constraints, m):
    r"""``'cycle'`` if the resolved rows ``s_constraints`` (``(a_int, shift)``
    pairs for ``a_int·s + shift > 0``, as ``_integrate_polytope`` receives
    them) contain a directed cycle of unit-coefficient pair rows whose
    shifts sum to <= 0 in EXACT arithmetic; ``None`` otherwise.

    A pair row has exactly two nonzero coefficients, exactly ``+1.0`` (at
    ``up``) and exactly ``-1.0`` (at ``lo``): ``s_up − s_lo + shift > 0``.
    Summing such rows around a directed cycle gives ``0 < Σ shift``, so a
    cycle with ``Σ shift <= 0`` empties the region whatever the other rows
    are.  The shifts are the resolved floats, summed as exact rationals
    (``fractions.Fraction``, through ``_order_rows_infeasible``): the
    verdict is exact for the region these rows define, with no tolerance.
    A cycle whose shifts sum to > 0 encloses a thin nonempty strip and is
    not decided; rows of any other shape (scalar bounds, constant rows,
    non-unit or 3+-term rows) are ignored, never a reason for a verdict.
    A structural pre-check (Kahn's algorithm on the unshifted order graph)
    returns ``None`` for an acyclic order without any exact arithmetic.
    """
    if m < 2:
        return None
    pairs = []
    try:
        for (a_int, shift) in s_constraints:
            up = lo = None
            clean = True
            for j, a in enumerate(a_int):
                a = float(a)
                if a == 0.0:
                    continue
                if a == 1.0 and up is None:
                    up = j
                elif a == -1.0 and lo is None:
                    lo = j
                else:
                    clean = False
                    break
            if clean and up is not None and lo is not None:
                pairs.append((lo, up, shift))
    except (TypeError, ValueError):
        return None
    if len(pairs) < 2:
        return None
    # Kahn's algorithm on the order graph (an edge up -> lo per pair row):
    # only a graph with a directed cycle can hold an infeasible one.
    succ = [[] for _ in range(m)]
    indeg = [0] * m
    for (lo, up, _shift) in pairs:
        succ[up].append(lo)
        indeg[lo] += 1
    stack = [v for v in range(m) if indeg[v] == 0]
    removed = 0
    while stack:
        v = stack.pop()
        removed += 1
        for w in succ[v]:
            indeg[w] -= 1
            if indeg[w] == 0:
                stack.append(w)
    if removed == m:
        return None
    try:
        exact = [(lo, up, _Fraction(shift)) for (lo, up, shift) in pairs]
    except (TypeError, ValueError, OverflowError):
        return None                     # a NaN / inf / non-real shift
    return 'cycle' if _order_rows_infeasible(m, exact) else None


def _extract_causal_poset(subset_constraint_data, free_ext_vals, m,
                          tol=1e-12, row_kinds=None, tie_ctx=None):
    r"""Build a ``_CausalPoset`` from a list of retardation
    constraints.

    Each constraint is ``(a_int, a_ext, c0)`` representing
    ``a_int · s + c_eff > 0`` where
    ``c_eff = c0 + a_ext · free_ext_vals``.

    Constraint shape recognised:
    * **Inter-axis**: ``a_int`` has exactly two nonzero entries
      summing to 0 (one +1, one −1), and ``c_eff ≈ 0``.
      Adds edge ``(u, v)`` where ``a_int[u] = −1`` (lower) and
      ``a_int[v] = +1`` (upper).  "≈ 0" is the pre-existing SHIFT
      tolerance ``|c_eff| <= 1000·tol`` (1e-9): a row inside it is
      integrated as the unshifted order ``s_v > s_u``, an approximation
      of ``s_v > s_u − c_eff`` (plan §2.4 item 1: route such rows to the
      exact DBM path at M3).  It is not a tie rule; ties of constant rows
      are decided exactly by ``_const_row_verdict``.
    * **Scalar lower**: ``a_int`` has exactly one +1 entry, all else 0.
      Adds (var, −c_eff) to ``scalar_lowers``.
    * **Scalar upper**: ``a_int`` has exactly one −1 entry, all else 0.
      Adds (var, c_eff) to ``scalar_uppers``.
    * **Constant** (zero normal): THETA0_CONST_ROW_MODE 'ito' (call time)
      applies ``_const_row_verdict`` (with ``tie_ctx``) to every constant
      row FIRST: one EMPTY row returns ``_EMPTY_POSET``, DROP rows are
      skipped.  'legacy_clip' (pre-M1): a row with ``c_eff <= 0`` returns
      ``None``.
    * **Anything else** (multiple inter-axis couplings, mixed
      coefficients, inter-axis with a shift beyond the window, etc.) →
      return ``None``.  Caller falls back to scipy.nquad.

    In 'ito' mode, inter-axis rows that form a directed cycle whose shifts
    sum to <= 0 (``_order_rows_infeasible``; e.g. an unshifted 2-cycle, or
    one shifted by a rounding-level −1e-16) also return ``_EMPTY_POSET``.
    A cycle with a positive total shift encloses a thin nonempty strip and
    is not decided here.

    Returns ``_CausalPoset``, ``None`` or (``'ito'`` only) ``_EMPTY_POSET``.
    ``row_kinds``: the rows' provenance (``None``: all 'edge');
    ``tie_ctx``: see ``_tie_order_sign``.
    """
    legacy = _theta0_legacy()
    const_verdict = {}
    if not legacy:
        # Constant rows first: one EMPTY row empties the region whatever
        # the other rows are (so it never reaches a fallback).
        for idx, (a_int, a_ext, c0) in enumerate(subset_constraint_data):
            if any(abs(float(x)) > _ROW_COEF_ATOL for x in a_int):
                continue
            _RUNTIME_COUNTERS['zero_normal_rows_seen'] += 1
            verdict = _const_row_verdict(a_int, a_ext, c0, free_ext_vals,
                                         _row_kind(row_kinds, idx), tie_ctx)
            if verdict == 'EMPTY':
                _RUNTIME_COUNTERS['poset_empty_const'] += 1
                return _EMPTY_POSET
            const_verdict[idx] = verdict      # 'DROP'
    edges = []
    order_rows = []
    scalar_lowers = []
    scalar_uppers = []
    for idx, (a_int, a_ext, c0) in enumerate(subset_constraint_data):
        c_eff = float(c0) + sum(
            float(a_ext[j]) * float(free_ext_vals[j])
            for j in range(len(a_ext))
        )
        # Find positions and signs of nonzero entries in a_int.
        nz = [(i, float(a_int[i])) for i in range(len(a_int))
              if abs(float(a_int[i])) > tol]
        if not nz:
            if not legacy:
                if idx in const_verdict:
                    continue            # DROP (decided above)
                # Coefficients in (_ROW_COEF_ATOL, tol]: only a direct
                # caller with a looser ``tol`` gets here.  Not a clean row.
                return None
            # Observational (M0.1): a zero-normal row reaching the
            # extractor (``tol`` may differ from 1e-12 for direct callers,
            # so re-check against the counter's definition).
            if all(abs(float(x)) <= 1e-12 for x in a_int):
                _RUNTIME_COUNTERS['zero_normal_rows_seen'] += 1
            # Pure constant constraint.  Should be satisfied
            # (c_eff > 0); if violated, polytope is empty (signal
            # via None — caller falls back, which will also detect
            # the empty polytope correctly).
            if c_eff <= 0:
                return None
            continue
        if len(nz) == 1:
            (var_idx, coef) = nz[0]
            # Pure scalar bound.  Only accept ±1 coefficients.
            if abs(coef - 1.0) < tol:
                # +1 · s_var + c_eff > 0  ⇔  s_var > −c_eff
                scalar_lowers.append((var_idx, -c_eff))
            elif abs(coef + 1.0) < tol:
                # −1 · s_var + c_eff > 0  ⇔  s_var < c_eff
                scalar_uppers.append((var_idx, c_eff))
            else:
                # Non-unit coefficient — not a clean ±1 constraint.
                return None
            continue
        if len(nz) == 2:
            # Need exactly one +1 and one −1, plus c_eff inside the shift
            # window (see the docstring).
            (i, a_i), (j, a_j) = nz
            if abs(c_eff) > 1000 * tol:
                # Inter-axis constraint with extra constant — would
                # be a "shifted" ordering like s_v > s_u + c.  Not
                # supported in the simple poset model; bail.
                return None
            if abs(a_i - 1.0) < tol and abs(a_j + 1.0) < tol:
                # +1 at i, −1 at j  ⇔  s_i > s_j  ⇔  edge (j → i)
                edge = (j, i)
            elif abs(a_i + 1.0) < tol and abs(a_j - 1.0) < tol:
                # −1 at i, +1 at j  ⇔  s_j > s_i  ⇔  edge (i → j)
                edge = (i, j)
            else:
                return None
            edges.append(edge)
            order_rows.append((edge[0], edge[1], c_eff))
            continue
        # 3+ nonzero entries — mixed constraint not supported.
        return None

    if not legacy and _order_rows_infeasible(m, order_rows):
        _RUNTIME_COUNTERS['poset_empty_cycle'] += 1
        return _EMPTY_POSET
    # Deduplicate edges.
    edges_set = tuple(sorted(set(edges)))
    return _CausalPoset(
        m=m,
        edges=edges_set,
        scalar_lowers=tuple(scalar_lowers),
        scalar_uppers=tuple(scalar_uppers),
    )


def _enumerate_linear_extensions(poset):
    r"""Yield each linear extension (topological ordering) of a
    ``_CausalPoset``.

    A linear extension is a permutation σ of [0, m) such that for
    every edge (u, v) of the poset, σ⁻¹(u) < σ⁻¹(v) — i.e., u appears
    before v in the ordering.

    Standard recursive Kahn-style enumeration: at each step, pick
    one of the currently-source nodes (no remaining predecessors)
    and recurse.

    Yields tuples of length ``poset.m``.
    """
    m = poset.m
    # Predecessor counts (per variable).
    in_count = [0] * m
    successors = [[] for _ in range(m)]
    for (u, v) in poset.edges:
        in_count[v] += 1
        successors[u].append(v)

    available = sorted(i for i in range(m) if in_count[i] == 0)

    def _recurse(order, in_count_local, available_local):
        if len(order) == m:
            yield tuple(order)
            return
        for var in list(available_local):
            new_available = [a for a in available_local if a != var]
            new_in_count = list(in_count_local)
            for succ in successors[var]:
                new_in_count[succ] -= 1
                if new_in_count[succ] == 0:
                    # Insert preserving sorted order for deterministic
                    # output (canonical lex enumeration).
                    inserted = False
                    for k, x in enumerate(new_available):
                        if x > succ:
                            new_available.insert(k, succ)
                            inserted = True
                            break
                    if not inserted:
                        new_available.append(succ)
            order.append(var)
            yield from _recurse(order, new_in_count, new_available)
            order.pop()

    yield from _recurse([], in_count, available)


def _causal_poset_consistent_scalar_lower(poset, tol=1e-9):
    r"""Compute the single effective scalar lower bound shared by
    variables that have one.

    Returns ``(L, True)`` if every variable WITH A SCALAR LOWER has
    the same value ``L`` (within ``tol``), or no scalar lower at all
    (returns ``(None, True)`` — caller uses a bbox cap).  Variables
    WITHOUT an explicit scalar lower inherit ``L`` via the chain
    ordering of any linear extension: the chain-simplex form
    integrates each non-bottom variable from L (or the previous
    variable's value, whichever is greater).  For retardation-style
    constraint structure where every internal vertex is reachable
    from an external-leaf-pinned ancestor, this is correct.

    Returns ``(None, False)`` if scalar lowers exist on multiple
    variables with DIFFERENT values — in that case the chain
    simplex over a single L would over- or under-include regions
    and the caller should fall back to scipy.nquad.

    M3 (P3, ``USE_DBM_FALLBACK``; ``_dbm_route_on`` at call time): also
    ``(None, False)`` when some variable has neither a scalar lower of its
    own nor a predecessor with one in the transitive closure of the strict
    order (``_poset_lower_not_inherited``).  Such a variable is not bounded
    below by L (only by the domain cap), so the single-L chain simplex
    would cut off the region below L.  With the flag off every variable
    inherits L, as before.
    """
    return _poset_scalar_lower_verdict(poset, tol)[:2]


def _poset_lower_not_inherited(poset, lower_vars):
    r"""True if some variable of ``poset`` has no scalar lower (it is not in
    ``lower_vars``) and no predecessor in ``lower_vars`` in the transitive
    closure of the strict order.

    An edge ``(u, v)`` means ``s_v > s_u``, so a variable inherits a lower
    bound L exactly from its ancestors: ``s_v > s_u > L``.  Order shifts
    play no role (the extractor accepts only unshifted order rows)."""
    m = poset.m
    preds = [[] for _ in range(m)]
    for (u, v) in poset.edges:
        preds[v].append(u)
    bounded = set(lower_vars)
    for v in range(m):
        if v in bounded:
            continue
        seen = {v}
        stack = list(preds[v])
        found = False
        while stack:
            u = stack.pop()
            if u in seen:
                continue
            if u in lower_vars:
                found = True
                break
            seen.add(u)
            stack.extend(preds[u])
        if not found:
            return True
    return False


def _poset_scalar_lower_verdict(poset, tol=1e-9):
    """``(L, ok, reason)``: ``_causal_poset_consistent_scalar_lower``'s
    ``(L, ok)`` plus why it failed (``None``, ``'inconsistent'``: unequal
    scalar lowers, ``'not_inherited'``: the M3 inheritance rule)."""
    per_var_max = {}
    for (var, c) in poset.scalar_lowers:
        cur = per_var_max.get(var)
        per_var_max[var] = c if cur is None else max(cur, c)
    if not per_var_max:
        return (None, True, None)
    vals = list(per_var_max.values())
    Lmin, Lmax = min(vals), max(vals)
    if Lmax - Lmin > tol:
        return (None, False, 'inconsistent')
    # All variables that have a scalar lower agree on value Lmax.
    # Variables without one inherit it via the chain ordering -- M3: only
    # from a lower-bounded ancestor.
    if _dbm_route_on() and _poset_lower_not_inherited(poset, per_var_max):
        return (None, False, 'not_inherited')
    return (Lmax, True, None)


def _causal_poset_consistent_scalar_upper(poset, tol=1e-9):
    r"""Companion to ``_causal_poset_consistent_scalar_lower`` for
    the upper-bound side, but only the TOP variable in each linear
    extension needs an upper bound (others are bounded by the next
    variable in the chain).

    Returns the smallest scalar upper found (across any variable),
    intended to be used as the cap for whichever variable ends up
    at the top of the linear extension — IF that variable has its
    own scalar upper.  When no variable has a scalar upper, the
    caller must supply a fallback cap.
    """
    per_var_min = {}
    for (var, c) in poset.scalar_uppers:
        cur = per_var_min.get(var)
        per_var_min[var] = c if cur is None else min(cur, c)
    return per_var_min


def _exp_over_chain_simplex(alphas, lower, upper, eps=1e-9):
    r"""Closed-form value of the nested integral on the chain
    simplex  ``{lower ≤ s_1 ≤ s_2 ≤ … ≤ s_N ≤ upper}``::

       ∫_{lower}^{upper}        ds_N  exp(α_N · s_N)
         · ∫_{lower}^{s_N}      ds_{N-1}  exp(α_{N-1} · s_{N-1})
         · …
         · ∫_{lower}^{s_2}      ds_1  exp(α_1 · s_1)

    where ``alphas = [α_1, α_2, …, α_N]`` with ``α_k`` the exponent
    coefficient for the k-th variable in the chain (1-indexed
    mathematically, 0-indexed in the list).

    Derivation: integrate inside-out.  After each step, the running
    integrand is a sum of terms of the form
    ``C · exp(β · s_outer + (constant depending on already-integrated
    bounds))``.  Each inner integration produces two new terms — one
    "upper-bound" piece (whose ``β`` merges with the next-outer
    variable's coefficient) and one "lower-bound" piece (a numerical
    prefactor ``exp(β · lower)`` falls out, leaving the outer
    coefficient unchanged).  After N steps the sum has 2^N constant
    terms; their sum is the integral.

    Parameters
    ----------
    alphas : sequence of complex
        Effective exponent coefficients, innermost first.
    lower, upper : float
        Common scalar lower bound and the cap on the outermost var.
    eps : float
        Threshold below which a ``β`` is treated as degenerate (the
        formula's ``1/β`` factor would amplify roundoff).  Returns
        ``None`` in that case so the caller falls back to scipy.

    Returns
    -------
    complex or None
        The integral value, or ``None`` if any intermediate
        coefficient ``β`` is too close to zero.
    """
    import cmath
    N = len(alphas)
    if N == 0:
        # Empty product of integrals — by convention 1.
        return 1.0 + 0.0j
    # Empty chain simplex (``upper ≤ lower``): integration domain has
    # zero measure, integral is 0.  Crucial for τ < 0 configurations
    # where the chain-bottom scalar lower (e.g. from leaf t = 0) and
    # the chain-top scalar upper (e.g. from leaf t = τ < 0) cross.
    if upper <= lower:
        return 0.0 + 0.0j

    # Each term: (complex coefficient, list of remaining β values).
    # At level k (about to integrate the (k+1)-th variable in the
    # chain, which is the innermost remaining), ``beta[0]`` is the
    # effective coefficient on that variable, ``beta[1:]`` are the
    # coefficients on the outer variables we haven't touched yet.
    terms = [(1.0 + 0.0j, list(alphas))]

    # Overflow guard.  The 2^N term expansion can produce intermediate
    # ``exp(b · L_or_U)`` factors whose Re(b · arg) exceeds the IEEE
    # double's exp range (~709).  The TRUE integral is finite — the
    # individual large terms cancel — but the cancellation is fragile
    # and depends on bit-exact arithmetic the closed-form can't
    # guarantee.  Safe path: detect the overflow risk, return None,
    # let the caller fall back to scipy.nquad which handles the
    # well-conditioned integral natively.
    # ``EXP_REAL_LIMIT = 600`` leaves a margin below the 709 hard
    # limit so accumulated roundoff doesn't push us over.
    EXP_REAL_LIMIT = 600.0

    try:
        # Integrate variables 1, 2, …, N-1 (each bounded above by next
        # variable in the chain).
        for _ in range(N - 1):
            new_terms = []
            for (C, beta) in terms:
                b_inner = beta[0]
                if abs(b_inner) < eps:
                    return None
                if abs((b_inner * lower).real) > EXP_REAL_LIMIT:
                    return None
                # Term A — upper-bound piece.  exp(b_inner · s_outer)
                # merges with the existing exp(beta[1] · s_outer)
                # factor, so the new β on the next-outer variable is
                # ``b_inner + beta[1]``.
                beta_A = list(beta[1:])
                beta_A[0] = b_inner + beta_A[0]
                new_terms.append((C / b_inner, beta_A))
                # Term B — lower-bound piece.  Just pulls out a
                # constant ``exp(b_inner · lower)`` factor; outer β
                # unchanged.
                beta_B = list(beta[1:])
                new_terms.append((
                    -C * cmath.exp(b_inner * lower) / b_inner,
                    beta_B,
                ))
            terms = new_terms

        # Outermost integration: s_N from lower to upper (both
        # constants).
        total = 0.0 + 0.0j
        for (C, beta) in terms:
            b = beta[0]
            if abs(b) < eps:
                # ∫_L^U exp(0 · s) ds = U − L
                total += C * (upper - lower)
            else:
                if (abs((b * upper).real) > EXP_REAL_LIMIT
                        or abs((b * lower).real) > EXP_REAL_LIMIT):
                    return None
                total += C * (cmath.exp(b * upper)
                              - cmath.exp(b * lower)) / b
    except (OverflowError, ValueError):
        return None
    return total


# ───────────────────────────────────────────────────────────────────────
# Numba-compiled chain simplex (fast-path companion to the function above)
# ───────────────────────────────────────────────────────────────────────
# Same algorithm as ``_exp_over_chain_simplex`` translated to a numba
# ``@njit`` function operating on pre-allocated complex128 numpy
# buffers.  Targets the chain simplex inner loop which post-Stage-3b is
# the dominant per-(diagram, subset, pole-tuple, linear-extension)
# cost.  Expected 30-100× per call vs the pure-Python version.
#
# Semantics MUST match the Python version bit-for-bit on every
# converged result.  The Python version stays in place as the
# reference / fallback (selected by ``USE_NUMBA_CHAIN_SIMPLEX = False``
# or by the wrapper on a numba-import failure).
import numpy as np

try:
    import numba as _numba
    _HAVE_NUMBA = True
except ImportError:
    _HAVE_NUMBA = False
    _numba = None

USE_NUMBA_CHAIN_SIMPLEX = True

# ─── Chain-simplex precision-loss fix (2026-05-16) ────────────────────
#
# When the chain simplex contains close-paired effective coefficients
# (e.g. spike-reset propagator with poles 0.329i, 0.351i, so some
# ``b_inner`` in the 2^N recursion is ~0.022), the closed-form formula
# in ``_exp_over_chain_simplex`` suffers cancellation-driven precision
# loss.  Empirically observed as a 4× aggregate overestimate of the
# 1-loop value at spike-reset k=2 ell=1 (76% rel diff vs the converged
# grouped-scipy reference), with per-subset audit measuring an aggregate
# analytic/scipy ratio of 1.086 across 30 sampled m=3 subsets that
# extrapolates to the observed 4× across 313 subsets.
#
# Fix: when any subset sum of ``alphas`` falls below a threshold (cheap
# 2^N scan of subset-sum magnitudes), route the entire chain-simplex
# evaluation to ``_exp_over_chain_simplex_mpmath`` running at 50-digit
# precision.  The mpmath path is 50-500× slower per call but only fires
# when the float64 path was already wrong; unaffected configurations
# (quad model, spike-reset k≤1 ell≤1, etc.) skip the gate entirely.
#
# Full audit at ``docs/m_ge3_precision_bug_audit.md``;
# design at ``docs/m_ge3_chain_simplex_fix_proposal.md``.
#
# To revert exactly to the pre-fix behaviour (bit-identical), set the
# flag to ``False``.  No other code changes required.
# Set to True to enable the close-pole mpmath dispatch in chain simplex.
# Empirically this did NOT address the spike-reset k=2 ell=1 bug (which
# turned out to be a cap-mismatch issue, not chain-simplex precision —
# see USE_POSET_CAP_MATCH_SCIPY below).  Keeping the code path for
# possible future use.  Off by default.
USE_CHAIN_SIMPLEX_PRECISION_FIX = False

# Threshold below which any subset-sum magnitude of ``alphas`` triggers
# the high-precision (mpmath) path.  Calibrated to catch the spike-reset
# 0.022i pole-difference case (which causes ~10% per-subset bias) while
# not unduly slowing well-conditioned models.  Lower → fewer mpmath
# calls (faster but less safe); higher → more mpmath calls (safer but
# slower).
_CHAIN_SIMPLEX_CANCEL_THRESHOLD = 0.1


def _min_subset_sum_abs(alphas):
    """Minimum |Σ_{i ∈ S} α_i| over non-empty subsets S of ``alphas``.

    Cheap 2^N scan used to detect when the chain-simplex recursion will
    encounter a small ``b_inner`` (cumulative-pole-sum) at some level,
    triggering precision-loss in the closed-form formula.

    Returns ``inf`` for empty input so the threshold gate is a no-op.

    Cached via ``_min_subset_sum_abs_cached`` keyed on the alphas tuple,
    since the chain-simplex dispatcher is called many times with the
    same α-vector.  Cache eliminates the per-call Python overhead that
    would otherwise erase the numba speedup on well-conditioned inputs.
    """
    n = len(alphas)
    if n == 0:
        return float('inf')
    # Use the alphas directly as a tuple; complex elements are hashable.
    # On numpy arrays this requires a manual tuple() conversion.
    return _min_subset_sum_abs_cached(tuple(alphas))


@_functools.lru_cache(maxsize=4096)
def _min_subset_sum_abs_cached(alphas_tuple):
    """Cached 2^N subset-sum-magnitude scan."""
    n = len(alphas_tuple)
    min_abs = float('inf')
    for mask in range(1, 1 << n):
        s = 0.0 + 0.0j
        for i in range(n):
            if mask & (1 << i):
                s = s + alphas_tuple[i]
        a = abs(s)
        if a < min_abs:
            min_abs = a
    return min_abs


def _exp_over_chain_simplex_mpmath(alphas, lower, upper, eps=1e-9, dps=50):
    r"""High-precision (mpmath, default 50 digits) evaluation of the
    same closed-form integral as ``_exp_over_chain_simplex``.

    Used as the dispatch target for ``_exp_over_chain_simplex_fast``
    when ``USE_CHAIN_SIMPLEX_PRECISION_FIX`` is enabled AND the
    cheap subset-sum scan detects a close-pole condition.

    Semantics match ``_exp_over_chain_simplex`` exactly — same input/
    output types, same ``None`` return on overflow / degenerate β at
    the float64-effective threshold ``eps``.  The difference is that
    intermediate arithmetic uses ``mpmath.mpc`` at 50 decimal digits,
    so the 2^N cancellation that loses ~14 digits of float64 precision
    leaves ~36 digits intact — well below any plausible aggregation
    tolerance.

    Parameters
    ----------
    alphas, lower, upper, eps : as in ``_exp_over_chain_simplex``.
    dps : int, default 50
        Decimal-digit precision for the mpmath workspace.  50 digits
        comfortably exceeds float64's 16, with margin for cancellation
        of any plausible close-pole pair (e.g., spike-reset's 0.022i
        loses ~14 digits → 50 − 14 = 36 left).
    """
    try:
        from mpmath import mp, mpc, exp as mp_exp
    except ImportError:
        # mpmath unavailable → behave as if the fix were off; caller
        # already has bit-exact recovery via flag.
        return _exp_over_chain_simplex(alphas, lower, upper, eps)

    n = len(alphas)
    if n == 0:
        return 1.0 + 0.0j
    if upper <= lower:
        return 0.0 + 0.0j

    EXP_REAL_LIMIT = 600.0

    saved_dps = mp.dps
    mp.dps = dps
    try:
        L = mpc(float(lower), 0.0)
        U = mpc(float(upper), 0.0)
        alphas_mp = [mpc(complex(a).real, complex(a).imag) for a in alphas]
        terms = [(mpc(1, 0), list(alphas_mp))]

        for _ in range(n - 1):
            new_terms = []
            for (C, beta) in terms:
                b_inner = beta[0]
                # Use the float64 magnitude for the threshold check —
                # mpmath's abs is exact but slow; complex(b_inner) is
                # fine since the test is "is this MUCH bigger than eps".
                if abs(complex(b_inner)) < eps:
                    return None
                bl_real = float((b_inner * L).real)
                if abs(bl_real) > EXP_REAL_LIMIT:
                    return None
                beta_A = list(beta[1:])
                beta_A[0] = b_inner + beta_A[0]
                new_terms.append((C / b_inner, beta_A))
                beta_B = list(beta[1:])
                new_terms.append((
                    -C * mp_exp(b_inner * L) / b_inner,
                    beta_B,
                ))
            terms = new_terms

        total = mpc(0, 0)
        for (C, beta) in terms:
            b = beta[0]
            if abs(complex(b)) < eps:
                total = total + C * (U - L)
            else:
                bu_real = float((b * U).real)
                bl_real = float((b * L).real)
                if abs(bu_real) > EXP_REAL_LIMIT or abs(bl_real) > EXP_REAL_LIMIT:
                    return None
                total = total + C * (mp_exp(b * U) - mp_exp(b * L)) / b
        return complex(float(total.real), float(total.imag))
    except Exception:
        return None
    finally:
        mp.dps = saved_dps


if _HAVE_NUMBA:
    @_numba.njit(cache=True)
    def _exp_over_chain_simplex_numba_core(alphas, lower, upper, eps):
        """Returns ``(status, value)``:
            status = 0  → ``value`` is the integral
            status = 1  → degenerate β (caller should fall back to
                          polynomial-prefactor path)
            status = 2  → overflow (caller returns None)
        """
        N = alphas.shape[0]
        if N == 0:
            return 0, 1.0 + 0.0j
        if upper <= lower:
            return 0, 0.0 + 0.0j

        EXP_REAL_LIMIT = 600.0

        # Buffer size = max possible term count = 2^(N-1).
        max_terms = 1 << max(N - 1, 0)

        # Ping-pong buffers for the doubling term list.
        coefs_a = np.zeros(max_terms, dtype=np.complex128)
        betas_a = np.zeros((max_terms, N), dtype=np.complex128)
        coefs_b = np.zeros(max_terms, dtype=np.complex128)
        betas_b = np.zeros((max_terms, N), dtype=np.complex128)

        coefs_a[0] = 1.0 + 0.0j
        for j in range(N):
            betas_a[0, j] = alphas[j]
        n_terms = 1

        for level in range(N - 1):
            n_remaining = N - level
            new_n = 0
            for i in range(n_terms):
                b_inner = betas_a[i, 0]
                if abs(b_inner) < eps:
                    return 1, 0.0 + 0.0j
                if abs((b_inner * lower).real) > EXP_REAL_LIMIT:
                    return 2, 0.0 + 0.0j
                exp_lower_val = np.exp(b_inner * lower)
                C_i = coefs_a[i]

                # Term A (upper-bound piece):
                #   coef  = C / b_inner
                #   β_new = (b_inner + β_old[1], β_old[2], …)
                coefs_b[new_n] = C_i / b_inner
                betas_b[new_n, 0] = b_inner + betas_a[i, 1]
                for j in range(2, n_remaining):
                    betas_b[new_n, j - 1] = betas_a[i, j]
                new_n += 1

                # Term B (lower-bound piece):
                #   coef  = -C · exp(b_inner · lower) / b_inner
                #   β_new = (β_old[1], β_old[2], …)
                coefs_b[new_n] = -C_i * exp_lower_val / b_inner
                for j in range(1, n_remaining):
                    betas_b[new_n, j - 1] = betas_a[i, j]
                new_n += 1

            # Swap a ↔ b for next level.
            tmp_c = coefs_a
            coefs_a = coefs_b
            coefs_b = tmp_c
            tmp_b = betas_a
            betas_a = betas_b
            betas_b = tmp_b
            n_terms = new_n

        # Outermost integration: s_N from lower to upper.
        total = 0.0 + 0.0j
        for i in range(n_terms):
            b = betas_a[i, 0]
            C_i = coefs_a[i]
            if abs(b) < eps:
                total += C_i * (upper - lower)
            else:
                if (abs((b * upper).real) > EXP_REAL_LIMIT
                        or abs((b * lower).real) > EXP_REAL_LIMIT):
                    return 2, 0.0 + 0.0j
                total += C_i * (np.exp(b * upper) - np.exp(b * lower)) / b

        return 0, total


def _exp_over_chain_simplex_fast_uncached(alphas, lower, upper, eps=1e-9):
    """Dispatcher: numba version when available + enabled, else Python.

    Returns the integral value or ``None`` (degenerate β / overflow),
    matching ``_exp_over_chain_simplex`` semantics exactly.

    When ``USE_CHAIN_SIMPLEX_PRECISION_FIX`` is enabled (default) and
    a cheap subset-sum scan detects a close-pole condition (any non-
    empty subset of ``alphas`` summing to magnitude
    < ``_CHAIN_SIMPLEX_CANCEL_THRESHOLD``), the entire call is routed
    to ``_exp_over_chain_simplex_mpmath`` for high-precision evaluation.
    The float64 path is precision-limited in that regime and produces
    a systematic overestimate that compounds across many subsets at
    spike-reset k≥2 ell≥1.  Set the flag to ``False`` to disable the
    routing and recover the pre-fix behaviour bit-identically.
    """
    if (USE_CHAIN_SIMPLEX_PRECISION_FIX
            and len(alphas) >= 3
            and _min_subset_sum_abs(alphas) < _CHAIN_SIMPLEX_CANCEL_THRESHOLD):
        return _exp_over_chain_simplex_mpmath(alphas, lower, upper, eps)
    if not (_HAVE_NUMBA and USE_NUMBA_CHAIN_SIMPLEX):
        return _exp_over_chain_simplex(alphas, lower, upper, eps)
    try:
        arr = np.asarray(alphas, dtype=np.complex128)
        if arr.ndim != 1:
            return _exp_over_chain_simplex(alphas, lower, upper, eps)
        status, val = _exp_over_chain_simplex_numba_core(
            arr, float(lower), float(upper), float(eps),
        )
    except Exception:
        return _exp_over_chain_simplex(alphas, lower, upper, eps)
    if status != 0:
        return None
    return complex(val)


def _exp_over_chain_simplex_polynomial_uncached(alphas, lower, upper, eps=1e-9):
    r"""Polynomial-prefactor extension of ``_exp_over_chain_simplex``.

    Same closed-form integral as ``_exp_over_chain_simplex`` but DOES
    NOT return ``None`` when an intermediate cumulative-pole-sum β
    vanishes.  In that case the level's antiderivative is a polynomial
    in s (rather than ``exp(β·s)/β``); the polynomial is carried
    through subsequent levels, with closed-form treatment of
    ``∫ u^k · exp(β·u) du`` at non-degenerate levels.

    Math sketch
    -----------
    Each level produces one of two outcomes:

    * **Non-degenerate (β ≠ 0).**  For each ``u^k · exp(β·s)`` term in
      the running integrand:

          ∫_lower^{s_outer} (s - lower)^k · exp(β·s) ds
          = exp(β·lower) · Σ_{j=0}^{k} (-1)^j · k!/(k-j)! · u_outer^{k-j}
                                       · exp(β·u_outer) / β^{j+1}
            + (-1)^{k+1} · k! · exp(β·lower) / β^{k+1}

      yields (k+1) "upper" terms carrying β into the next-outer
      variable's exponent, plus 1 "lower" constant term with no β.

    * **Degenerate (β ≈ 0).**  The integrand is just a polynomial:

          ∫_lower^{s_outer} (s - lower)^k ds  =  u_outer^{k+1} / (k+1)

      yields a single term whose polynomial degree has grown by 1 and
      whose β-list is unchanged (no exp factor carries forward).

    The final outermost integration replaces ``s_outer`` with a numeric
    upper bound and accumulates the per-term contributions.

    Term representation
    -------------------
    Each running term is a tuple ``(C, poly, β_list)``:
    * ``C`` — complex scalar coefficient.
    * ``poly`` — tuple of complex polynomial coefficients in basis
      ``(s - lower)^k``, index = degree.
    * ``β_list`` — tuple of complex β values for the variables
      remaining to integrate, innermost first.

    Initial state: ``[(1+0j, (1+0j,), tuple(alphas))]``.

    Parameters and return value mirror ``_exp_over_chain_simplex``;
    returns ``None`` only on overflow (real-exponent magnitude beyond
    ``EXP_REAL_LIMIT`` = 600), never on degenerate β.
    """
    import cmath
    import math

    N = len(alphas)
    if N == 0:
        return 1.0 + 0.0j
    if upper <= lower:
        return 0.0 + 0.0j

    EXP_REAL_LIMIT = 600.0

    terms = [(1.0 + 0.0j, (1.0 + 0.0j,), tuple(alphas))]

    try:
        # Inner-loop integrations: s_1, s_2, …, s_{N-1}.
        for _level in range(N - 1):
            new_terms = []
            for (C, poly, β_list) in terms:
                β_inner = β_list[0]
                β_rest = β_list[1:]
                # β_rest always has ≥ 1 element while in this inner loop.

                if abs(β_inner) < eps:
                    # Degenerate level: polynomial integration.
                    new_poly = [0.0 + 0.0j] * (len(poly) + 1)
                    for k, a in enumerate(poly):
                        new_poly[k + 1] = a / (k + 1)
                    new_terms.append((C, tuple(new_poly), β_rest))
                else:
                    # Non-degenerate: closed-form polynomial × exp.
                    if abs((β_inner * lower).real) > EXP_REAL_LIMIT:
                        return None
                    exp_lower = cmath.exp(β_inner * lower)
                    # Upper β list: β_inner merges into next-outer β.
                    β_upper_rest = (β_inner + β_rest[0],) + β_rest[1:]
                    β_lower_rest = β_rest

                    for k, a in enumerate(poly):
                        if a == 0:
                            continue
                        k_fact = math.factorial(k)
                        # (k+1) upper terms carrying β forward.
                        for j in range(k + 1):
                            sign = -1 if (j & 1) else 1
                            falling = k_fact // math.factorial(k - j)
                            coef = C * a * sign * falling / (β_inner ** (j + 1))
                            deg = k - j
                            up_poly = tuple(
                                (1.0 + 0.0j) if i == deg else (0.0 + 0.0j)
                                for i in range(deg + 1)
                            )
                            new_terms.append((coef, up_poly, β_upper_rest))
                        # 1 lower term (constant, β-list reduced by one).
                        sign_last = -1 if ((k + 1) & 1) else 1
                        coef_lower = (C * a * sign_last * k_fact
                                       / (β_inner ** (k + 1)) * exp_lower)
                        new_terms.append(
                            (coef_lower, (1.0 + 0.0j,), β_lower_rest)
                        )
            terms = new_terms

        # Outermost integration: s_N from `lower` to `upper`.
        total = 0.0 + 0.0j
        u_top = upper - lower
        for (C, poly, β_list) in terms:
            β = β_list[0] if β_list else (0.0 + 0.0j)
            if abs(β) < eps:
                # Pure polynomial: ∫_lower^{upper} Σ a_k (s-lower)^k ds
                for k, a in enumerate(poly):
                    if a == 0:
                        continue
                    total += C * a * (u_top ** (k + 1)) / (k + 1)
            else:
                # ∫_lower^{upper} Σ a_k (s-lower)^k · exp(β·s) ds
                # = Σ_k a_k · [Σ_{j=0..k} (-1)^j k!/(k-j)! u_top^{k-j} exp(β·upper)/β^{j+1}
                #              + (-1)^{k+1} k! · exp(β·lower)/β^{k+1}]
                if abs((β * lower).real) > EXP_REAL_LIMIT:
                    return None
                if abs((β * upper).real) > EXP_REAL_LIMIT:
                    return None
                exp_top = cmath.exp(β * upper)
                exp_low = cmath.exp(β * lower)
                for k, a in enumerate(poly):
                    if a == 0:
                        continue
                    k_fact = math.factorial(k)
                    contrib = 0.0 + 0.0j
                    for j in range(k + 1):
                        sign = -1 if (j & 1) else 1
                        falling = k_fact // math.factorial(k - j)
                        contrib += (sign * falling * (u_top ** (k - j))
                                    * exp_top / (β ** (j + 1)))
                    sign_last = -1 if ((k + 1) & 1) else 1
                    contrib += sign_last * k_fact * exp_low / (β ** (k + 1))
                    total += C * a * contrib
        return total
    except (OverflowError, ValueError, ZeroDivisionError):
        return None


# ───────────────────────────────────────────────────────────────────────
# Memoised chain-simplex evaluation
# ───────────────────────────────────────────────────────────────────────
# Both chain-simplex routines are pure functions of (alphas, lower, upper,
# eps), and the poset walk calls them with a TINY set of distinct arguments
# over and over.  Measured on ou_quartic k=4, ell=2: 508,176 calls to each of
# the fast and polynomial routines, with just 9 DISTINCT argument tuples --
# a redundancy of 56,464x.  (The fast routine returns None on 100% of those
# calls because this model has a single retarded pole, so every alpha is
# degenerate and the polynomial path runs every time; memoising the None is
# worth as much as memoising the value.)
#
# Keys use the raw float/complex values -- the repeats are bit-identical
# because they come from the same pole data, so no rounding is applied and
# no result can be silently aliased onto a nearby-but-different argument.
_CHAIN_SIMPLEX_MEMO_MAX = 200_000
_chain_simplex_memo_fast = {}
_chain_simplex_memo_poly = {}

# ── M4 (P5): memo for ``_chain_with_intermediate_uppers`` ──────────────
# That function is pure and deterministic of (alphas, L, uppers, U) and the
# kernel flags below, and the poset walk calls it many times with identical
# arguments (the m >= 3 chain path at higher loop order; models with repeated
# poles).  The memo changes NO number: a hit returns the very object the
# first call computed, so every result is bit-identical to the memo off.
#
# ``USE_CHAIN_UPPERS_MEMO`` (bool, read at call time); environment
# ``DAEDALUS_CHAIN_UPPERS_MEMO`` = 1 | 0 (also true/false, yes/no, on/off;
# default 1).  It is a pure speed switch, so the umbrella
# ``DAEDALUS_PHASE_J_LEGACY`` (which reproduces pre-M1 NUMBERS) does not
# touch it.
#
# Thread safety: the spatial path runs Phase J under a ThreadPoolExecutor.
# Lookup (with its counter) and miss-insert (with eviction) hold
# ``_CHAIN_UPPERS_LOCK``; the value is computed OUTSIDE the lock, so two
# threads racing on one key both compute it (harmless: equal values).  At the
# size cap the table is cleared in one locked ``clear()``.
_CHAIN_UPPERS_MEMO_MAX = 200_000
_chain_uppers_memo = {}
_CHAIN_UPPERS_LOCK = _threading.Lock()
_CHAIN_UPPERS_MISS = object()


def _initial_chain_uppers_memo_flag(environ=None):
    """The import-time value of ``USE_CHAIN_UPPERS_MEMO`` from ``environ``
    (default ``os.environ``).  An unknown value raises."""
    env = _os.environ if environ is None else environ
    v = env.get('DAEDALUS_CHAIN_UPPERS_MEMO', '').strip().lower()
    if v in ('', '1', 'true', 'yes', 'on'):
        return True
    if v in ('0', 'false', 'no', 'off'):
        return False
    raise ValueError(
        f'DAEDALUS_CHAIN_UPPERS_MEMO={v!r}: expected 1 or 0')


USE_CHAIN_UPPERS_MEMO = _initial_chain_uppers_memo_flag()


def _chain_uppers_memo_on():
    """``USE_CHAIN_UPPERS_MEMO`` (call time), validated."""
    flag = USE_CHAIN_UPPERS_MEMO
    if flag is True or flag is False:
        return flag
    raise ValueError(f'final_integral.USE_CHAIN_UPPERS_MEMO={flag!r}: '
                     f'expected True or False')


def _chain_uppers_kernel_tag():
    """The module state that can change ``_chain_with_intermediate_uppers``,
    read at call time (part of every memo key, so toggling any of it misses).

    On the function's call path (``_exp_over_chain_simplex_fast`` ->
    ``_exp_over_chain_simplex_fast_uncached``, and the polynomial routine):
      * ``USE_NUMBA_CHAIN_SIMPLEX`` / ``_HAVE_NUMBA``: numba core vs the
        Python reference (they agree to rounding, not bit-for-bit);
      * ``USE_CHAIN_SIMPLEX_PRECISION_FIX`` and
        ``_CHAIN_SIMPLEX_CANCEL_THRESHOLD``: the mpmath dispatch for
        close-pole chains and its threshold.
    Included defensively although they sit in the CALLER
    (``_integrate_nd_polytope_poset_modesum``), not on this function's path,
    so no result of this function depends on them today:
      * ``USE_POSET_MPMATH_ACCUMULATION``, ``USE_POSET_CAP_MATCH_SCIPY``.
    ``CHAIN_UPPERS_KERNEL`` (the M6 transfer-kernel selector) does not exist
    yet: ADD IT HERE when it lands, or the memo would serve one kernel's
    values to the other."""
    return (USE_NUMBA_CHAIN_SIMPLEX, _HAVE_NUMBA,
            USE_CHAIN_SIMPLEX_PRECISION_FIX, _CHAIN_SIMPLEX_CANCEL_THRESHOLD,
            USE_POSET_MPMATH_ACCUMULATION, USE_POSET_CAP_MATCH_SCIPY,
            # CHAIN_UPPERS_KERNEL goes here (M6)
            )


def _chain_simplex_memo_clear():
    """Drop every chain-simplex memo table (call when poles change): the
    fast and polynomial tables and the M4 ``_chain_with_intermediate_uppers``
    table."""
    _chain_simplex_memo_fast.clear()
    _chain_simplex_memo_poly.clear()
    with _CHAIN_UPPERS_LOCK:
        _chain_uppers_memo.clear()


def _chain_simplex_fast_tag():
    """The module state ``_exp_over_chain_simplex_fast_uncached`` reads, at
    call time: the numba switch (and whether numba imported), the
    close-pole precision gate and its threshold.  Part of the fast table's
    key, so a toggle can never be answered from a stale entry (M4; before it
    the key held only the arguments).  The polynomial routine reads no
    module flag, so its key carries no tag."""
    return (USE_NUMBA_CHAIN_SIMPLEX, _HAVE_NUMBA,
            USE_CHAIN_SIMPLEX_PRECISION_FIX, _CHAIN_SIMPLEX_CANCEL_THRESHOLD)


def _chain_simplex_key(alphas, lower, upper, eps, tag=()):
    return (tuple(alphas), lower, upper, eps, tag)


def _exp_over_chain_simplex_fast(alphas, lower, upper, eps=1e-9):
    """Memoised wrapper -- see ``_exp_over_chain_simplex_fast_uncached``."""
    try:
        key = _chain_simplex_key(alphas, lower, upper, eps,
                                 _chain_simplex_fast_tag())
    except TypeError:            # unhashable argument: skip the cache
        return _exp_over_chain_simplex_fast_uncached(alphas, lower, upper, eps)
    memo = _chain_simplex_memo_fast
    if key in memo:
        _RUNTIME_COUNTERS['chain_simplex_memo_hits'] += 1
        val = memo[key]
        if val is None:
            _RUNTIME_COUNTERS['chain_simplex_fast_returned_none'] += 1
        return val
    _RUNTIME_COUNTERS['chain_simplex_memo_misses'] += 1
    val = _exp_over_chain_simplex_fast_uncached(alphas, lower, upper, eps)
    if val is None:
        _RUNTIME_COUNTERS['chain_simplex_fast_returned_none'] += 1
    if len(memo) < _CHAIN_SIMPLEX_MEMO_MAX:
        memo[key] = val
    return val


def _exp_over_chain_simplex_polynomial(alphas, lower, upper, eps=1e-9):
    """Memoised wrapper -- see ``_exp_over_chain_simplex_polynomial_uncached``."""
    try:
        key = _chain_simplex_key(alphas, lower, upper, eps)
    except TypeError:
        return _exp_over_chain_simplex_polynomial_uncached(
            alphas, lower, upper, eps)
    _RUNTIME_COUNTERS['chain_simplex_polynomial_called'] += 1
    memo = _chain_simplex_memo_poly
    if key in memo:
        _RUNTIME_COUNTERS['chain_simplex_memo_hits'] += 1
        val = memo[key]
        if val is None:
            _RUNTIME_COUNTERS['chain_simplex_polynomial_returned_none'] += 1
        return val
    _RUNTIME_COUNTERS['chain_simplex_memo_misses'] += 1
    val = _exp_over_chain_simplex_polynomial_uncached(
        alphas, lower, upper, eps)
    if val is None:
        _RUNTIME_COUNTERS['chain_simplex_polynomial_returned_none'] += 1
    if len(memo) < _CHAIN_SIMPLEX_MEMO_MAX:
        memo[key] = val
    return val


def _chain_uppers_key_suffix(L, upper_per_position, U_chain_top):
    """The (L, uppers, U) part of a ``_chain_with_intermediate_uppers`` memo
    key.  Independent of the alphas, so the poset plan loop builds it once
    per linear extension instead of once per pole-tuple group (M4 "A2").
    ``uppers`` is sorted (dict-order invariant) and keeps a ``None``-valued
    entry as ``None``: the uncached function treats a key mapped to ``None``
    as absent when forming effective uppers but as a direct constraint when
    placing cuts, so dropping it could alias two different results.
    Raises ``TypeError`` on an input that cannot be keyed, among them a
    position that is not an integer (``2.5`` must not share the key of
    ``2``: the callee treats them differently)."""
    if upper_per_position:
        items = []
        for k, v in upper_per_position.items():
            if k != int(k):
                raise TypeError(f'non-integer upper position {k!r}')
            items.append((int(k), None if v is None else float(v)))
        uppers = tuple(sorted(items))
    else:
        uppers = ()
    return (float(L), uppers, float(U_chain_top))


def _chain_with_intermediate_uppers(
    alphas_chain,
    L,
    upper_per_position,
    U_chain_top,
    _key_suffix=None,
):
    """Memoised wrapper (M4; ``USE_CHAIN_UPPERS_MEMO``) -- see
    ``_chain_with_intermediate_uppers_uncached``.  ``_key_suffix`` is an
    optional precomputed ``_chain_uppers_key_suffix(L, upper_per_position,
    U_chain_top)`` (the plan loop hoists it)."""
    if not _chain_uppers_memo_on():
        return _chain_with_intermediate_uppers_uncached(
            alphas_chain, L, upper_per_position, U_chain_top)
    try:
        if _key_suffix is None:
            _key_suffix = _chain_uppers_key_suffix(
                L, upper_per_position, U_chain_top)
        tag = _chain_uppers_kernel_tag()
        key = (tuple(alphas_chain), _key_suffix, tag)
        with _CHAIN_UPPERS_LOCK:
            # hashing the key (a ``TypeError`` for an unhashable alpha)
            # happens in this first lookup, before any counter moves
            val = _chain_uppers_memo.get(key, _CHAIN_UPPERS_MISS)
            if val is not _CHAIN_UPPERS_MISS:
                _RUNTIME_COUNTERS['chain_uppers_memo_hits'] += 1
                return val
            _RUNTIME_COUNTERS['chain_uppers_memo_misses'] += 1
    except (TypeError, ValueError, OverflowError):
        # unhashable / unkeyable (the callee's own handling of the input
        # applies): skip the cache
        return _chain_with_intermediate_uppers_uncached(
            alphas_chain, L, upper_per_position, U_chain_top)
    val = _chain_with_intermediate_uppers_uncached(
        alphas_chain, L, upper_per_position, U_chain_top)
    if _chain_uppers_kernel_tag() != tag:
        # a kernel flag moved while this call was computing (another thread
        # toggling it): the value may belong to either state, so do not
        # store it under the tag read before the compute
        return val
    with _CHAIN_UPPERS_LOCK:
        if len(_chain_uppers_memo) >= _CHAIN_UPPERS_MEMO_MAX:
            _chain_uppers_memo.clear()
            _RUNTIME_COUNTERS['chain_uppers_memo_evictions'] += 1
        _chain_uppers_memo[key] = val
    return val


def _chain_with_intermediate_uppers_uncached(
    alphas_chain,
    L,
    upper_per_position,
    U_chain_top,
):
    r"""Chain-simplex integral with scalar uppers at arbitrary positions
    (Stage 3b-maximality, generalisation of ``_exp_over_chain_simplex``).

    Computes

       ∫_{L ≤ s_{σ(0)} ≤ … ≤ s_{σ(m-1)} ≤ U_chain_top  ∧  s_{σ(k)} ≤ U_k for k ∈ keys}
            ∏_k exp(α_k · s_{σ(k)})  ds

    where ``upper_per_position[k] = U_k`` is the scalar upper bound on
    position ``k`` in the chain (positions not in the dict are bounded
    only by chain ordering + ``U_chain_top``).

    Math
    ----
    Chain ordering + per-position scalar uppers give an "effective upper"
    at each position:

        effective_upper[k] = min over j ≥ k of (upper_per_position[j], U_chain_top)

    which is non-decreasing in ``k``.  Group consecutive positions
    with identical ``effective_upper`` into "levels"; for ``q+1``
    levels the top level has the largest upper (= ``effective_upper[m-1]``).

    For each lower level (i = 0..q-1, value ``U_i < U_top``), the chain
    crosses ``U_i`` at some position ``c_i ∈ [level_i.start, m-1]``.
    By chain ordering, the crossings are monotonic:
    ``c_0 ≤ c_1 ≤ … ≤ c_{q-1}``.  Enumerating valid tuples
    ``(c_0, c_1, …, c_{q-1})`` and computing a piece-product per
    tuple decomposes the integral exactly.

    Pieces from a cut tuple:
    * Piece ``i`` (``i = 0..q-1``): positions ``[c_{i-1}+1 … c_i]``
      with bounds ``[U_{i-1}, U_i]`` (where ``U_{-1} = L``).  May be
      empty when ``c_{i-1} = c_i`` (consecutive cuts at same position).
    * Final piece: positions ``[c_{q-1}+1 … m-1]`` with bounds
      ``[U_{q-1}, U_top]``.  May be empty when ``c_{q-1} = m-1``.

    Each non-empty piece is an independent chain simplex; the product
    gives the case's contribution.

    Returns ``None`` if every case produces ``None`` from the
    underlying chain simplex (genuine overflow); otherwise returns
    the analytic closed-form sum.
    """
    m = len(alphas_chain)
    if m == 0:
        return 1.0 + 0.0j
    if upper_per_position is None:
        upper_per_position = {}

    # Compute effective upper at each position.
    effective_upper = [U_chain_top] * m
    running_min = U_chain_top
    for k in range(m - 1, -1, -1):
        u_k = upper_per_position.get(k)
        if u_k is not None:
            running_min = min(running_min, float(u_k))
        effective_upper[k] = running_min

    if any(effective_upper[k] <= L for k in range(m)):
        return 0.0 + 0.0j

    # Group consecutive positions with identical effective_upper into
    # levels.  ``levels[i] = (start, end_inclusive, upper_value)``.
    levels = []
    k = 0
    while k < m:
        v = effective_upper[k]
        start = k
        while k < m and effective_upper[k] == v:
            k += 1
        levels.append((start, k - 1, v))

    # If only one level, no intermediate uppers: standard chain.
    if len(levels) == 1:
        top = levels[0][2]
        v = _exp_over_chain_simplex_fast(alphas_chain, L, top)
        if v is None:
            v = _exp_over_chain_simplex_polynomial(alphas_chain, L, top)
        return v

    # q non-top levels (with strictly smaller uppers than the top level).
    q = len(levels) - 1

    # For each non-top level, find the LATEST position with a direct
    # constraint (i.e., upper_per_position[k] is set).  This is the
    # minimum cut position for the level: the cut must happen at or
    # after the constraint's original position, not just at the level's
    # extended start (positions before the constraint inherit the
    # upper via chain ordering, not via a direct constraint).
    cut_min_per_level = []
    for i in range(q):
        start, end, _ = levels[i]
        latest_direct = None
        for k in range(start, end + 1):
            if k in upper_per_position:
                latest_direct = k
        cut_min_per_level.append(
            latest_direct if latest_direct is not None else start
        )

    total = 0.0 + 0.0j
    any_returned = False

    # Enumerate cut tuples (c_0, c_1, …, c_{q-1}) with each
    # c_i ∈ [cut_min_per_level[i], m-1] and c_0 ≤ c_1 ≤ … ≤ c_{q-1}.
    def _gen_cuts(i, prev_c):
        if i == q:
            yield ()
            return
        lo = max(prev_c, cut_min_per_level[i])
        for c in range(lo, m):
            for rest in _gen_cuts(i + 1, c):
                yield (c,) + rest

    for cuts in _gen_cuts(0, 0):
        pieces = []
        prev_end = -1
        prev_upper = L
        for i, c in enumerate(cuts):
            positions = list(range(prev_end + 1, c + 1))
            upper_val = levels[i][2]
            pieces.append({
                'positions': positions,
                'lower': prev_upper,
                'upper': upper_val,
            })
            prev_end = c
            prev_upper = upper_val
        # Final piece: top-level positions with [last_cut_upper, top_upper].
        top_upper = levels[-1][2]
        pieces.append({
            'positions': list(range(prev_end + 1, m)),
            'lower': prev_upper,
            'upper': top_upper,
        })

        # Multiply chain simplex evaluations across non-empty pieces.
        case_value = 1.0 + 0.0j
        case_ok = True
        for piece in pieces:
            if not piece['positions']:
                continue
            if piece['upper'] <= piece['lower']:
                case_value = 0.0 + 0.0j
                break
            alphas_piece = [alphas_chain[k] for k in piece['positions']]
            v = _exp_over_chain_simplex_fast(
                alphas_piece, piece['lower'], piece['upper'],
            )
            if v is None:
                v = _exp_over_chain_simplex_polynomial(
                    alphas_piece, piece['lower'], piece['upper'],
                )
            if v is None:
                case_ok = False
                break
            case_value *= v

        if case_ok:
            total += case_value
            any_returned = True

    return total if any_returned else None


USE_POSET_INTEGRATOR = True

# ─── 2026-05-17: bbox-cap consistency between analytic and scipy ──────
#
# Before this fix, the analytic m≥3 poset evaluator used
# ``L = earliest_ext - POSET_PHYSICAL_MARGIN`` (default 50.0) for the
# lower bound on the chain when no scalar lower constraint was present,
# while the scipy fallback used ``L = -OUTER_CAP`` (default 200.0).
# The two paths therefore integrated different domains on the unbounded
# direction of the polytope.  For models with strictly retarded poles
# (Re β ≪ 0) the smooth integrand decays fast enough that the difference
# is negligible.  But for marginal-stability models (Re β ≈ 0 — e.g.
# spike-reset near the firing-rate fixed point) the integrand oscillates
# without decay and the integral is genuinely cap-dependent: analytic
# at L=-50 and scipy at L=-200 disagree by ~10-13% per m≥3 subset, which
# compounds to a 4× aggregate error in the 1-loop value at spike-reset
# k=2 ell=1.
#
# When True (default), the scipy.nquad m≥3 path uses ``OUTER_CAP =
# POSET_PHYSICAL_MARGIN`` (50.0) instead of the hard-coded 200.0,
# matching the analytic poset path's lower-bound fallback.  This keeps
# the analytic path fast (it never overflows at the tighter cap) and
# brings the grouped Phase J scipy reference into agreement with the
# per-diag analytic value.  Both paths now compute the same regularized
# integral for unbounded-below polytopes.
#
# Set the flag to False to recover the pre-fix behaviour bit-identically
# (scipy at cap=200, analytic at cap=50).
# See ``docs/m_ge3_precision_bug_audit.md`` for the full evidence chain.
# Empirically the cap mismatch is NOT the dominant source of the
# spike-reset k=2 ell=1 disagreement: aligning the caps at 50 does not
# bring per-diag (+2.66e-3) into agreement with grouped (+6.38e-4).  The
# grouped path's analytic poset uses pre-summed (cancelled-within-group)
# pole tuples, which produces a smaller integrand magnitude and hence a
# different value than per-diag's individual-diagram pole tuples
# summed AFTER integration.  By linearity these should be equal, so the
# 4× discrepancy points to a bookkeeping difference in pole-tuple
# construction between the two paths.  Off by default; see
# ``docs/m_ge3_precision_bug_audit.md``.
USE_POSET_CAP_MATCH_SCIPY = False


def _nquad_outer_cap():
    """Half-width of the box on which the legacy default-tolerance
    scipy.nquad routine (``_integrate_nd_polytope``) closes an open
    direction: ``OUTER_CAP`` (read at call time).  The DBM route closes the
    directions its rows leave open at the same ±cap.  (The hardened
    fallback truncates certified directions at K/κ instead; see the
    ``USE_DBM_FALLBACK`` comment.)"""
    return POSET_PHYSICAL_MARGIN if USE_POSET_CAP_MATCH_SCIPY else 200.0

# ─── Experimental: mpmath accumulation in the m≥3 poset evaluator ─────
#
# Hypothesis (2026-05-16): the per-diag analytic 1-loop at spike-reset
# k=2 ell=1 is 4× too high because ``_integrate_nd_polytope_poset_modesum``
# sums O(6^n_smooth) pole-tuple terms whose individual magnitudes can be
# large (close-paired poles produce O(1/Δp) residues; products of 5 such
# can be O(10^8)).  In float64, the cancellation down to the actual O(1)
# integral loses many digits per subset; the bias compounds across 313
# m≥3 subsets to give the observed 4× aggregate error.
#
# When the flag is True, the OUTER pole-tuple sum is accumulated in
# mpmath at 50-digit precision while individual term computation stays
# in float64.  Costs only the additions (negligible vs the chain simplex
# work).  When False (default), behaviour is bit-identical to pre-fix.
#
# This flag is provisional pending validation against
# ``test_grouped_vs_perdiag.py`` and the spike-reset k=2 ell=1 fixture.
# Empirically this did NOT address the spike-reset bug either — the
# poset accumulation has cancellation factor only ~21× (well within
# float64 precision).  Keeping the code path for possible future use
# on configs with bigger cancellation.  Off by default.
USE_POSET_MPMATH_ACCUMULATION = False


def _integrate_nd_polytope_poset_modesum(
    smooth_edge_modes,
    prefactor_complex,
    subset_constraint_data,
    free_ext_vals,
    m,
    bbox_cap=None,
    pole_tuples=None,
    plan=None,
    row_kinds=None,
    tie_ctx=None,
):
    r"""Analytic ``∫_{polytope} Π_e [Σ_α C_α exp(λ_α · Δt_e)] · pref
                                ds_1 … ds_m`` for m ≥ 3 via causal-
    poset decomposition.

    ``bbox_cap=None`` resolves at call time (``_resolve_bbox_cap``);
    ``row_kinds`` is the rows' provenance (``None``: all 'edge');
    ``tie_ctx`` orders an exact external-time tie (``_tie_order_sign``).
    With THETA0_CONST_ROW_MODE 'ito' a region that is empty by construction
    (``_EMPTY_POSET``: a constant EMPTY row, or an order cycle whose shifts
    sum to <= 0) returns 0j, in both the plan and the no-plan branch.

    Procedure:
      1. Extract the causal poset (DAG + scalar bounds) from the
         retardation constraints.  Fail (return ``None``) if any
         constraint is mixed (not a clean inter-axis or scalar
         bound).
      2. Resolve the COMMON scalar lower bound ``L`` for all
         integration variables.  Fail if scalar lowers differ across
         variables (the simple chain simplex form would over- or
         under-include regions), and (M3, ``USE_DBM_FALLBACK``) if a
         variable has neither a scalar lower nor a lower-bounded
         predecessor (``'poset_lower_not_inherited'``; the dispatch then
         takes the exact DBM route).  Fall back to ``-bbox_cap`` when no
         scalar lower is present.
      3. Enumerate every linear extension σ of the poset.  Each is a
         disjoint chain simplex
           L ≤ s_{σ(0)} ≤ s_{σ(1)} ≤ … ≤ s_{σ(m-1)} ≤ U_σ
         where U_σ is the scalar upper of σ(m-1) (or ``bbox_cap``).
      4. For each pole tuple (α_e)_e:
           α_v = Σ_e λ_α_e · a_int_e[v]                (per orig var)
           γ   = Σ_e λ_α_e · c_ext_e                   (ext-time part)
           α_chain[k] = α_{σ(k)}                       (permute)
           contribution = pref · ∏ C_α_e · exp(γ) ·
                          _exp_over_chain_simplex(α_chain, L, U_σ)
      5. Sum across extensions and tuples.  Fail if any chain
         simplex returns ``None`` (degenerate β).

    Returns ``complex`` or ``None``.  ``None`` triggers a fallback to
    scipy.nquad in the caller.
    """
    import cmath
    if m < 3:
        return _bail('poset_m_lt_3')  # m=2 has its own dedicated path
    _RUNTIME_COUNTERS['poset_attempted'] += 1
    bbox_cap = _resolve_bbox_cap(bbox_cap)
    n_smooth = len(smooth_edge_modes)
    if len(subset_constraint_data) != n_smooth:
        _RUNTIME_COUNTERS['poset_returned_none_total'] += 1
        return _bail('poset_rows_mismatch')

    poset = _extract_causal_poset(
        subset_constraint_data, free_ext_vals, m, row_kinds=row_kinds,
        tie_ctx=tie_ctx,
    )
    if poset is _EMPTY_POSET:
        return 0.0 + 0.0j
    if poset is None:
        _RUNTIME_COUNTERS['poset_extract_returned_none'] += 1
        _RUNTIME_COUNTERS['poset_returned_none_total'] += 1
        return _bail('poset_extract_none')

    L_value, lower_ok, lower_why = _poset_scalar_lower_verdict(poset)
    if not lower_ok:
        _RUNTIME_COUNTERS['poset_consistent_lower_failed'] += 1
        _RUNTIME_COUNTERS['poset_returned_none_total'] += 1
        if lower_why == 'not_inherited':
            # M3 (P3): a variable is not bounded below by L; the DBM route
            # (``_integrate_subset_dbm``) takes the subset.
            _RUNTIME_COUNTERS['poset_lower_not_inherited'] += 1
            return _bail('poset_lower_not_inherited')
        return _bail('poset_lower_inconsistent')
    # ── Lower bound: tight physical fallback (Stage 3b-bounds) ──────
    # When ``_causal_poset_consistent_scalar_lower`` returns no scalar
    # lower (L_value is None), the integrand still extends "to the
    # past" only as far as the retarded propagator chain decays.
    # Using ``-bbox_cap = -200`` here was too loose: combined with
    # cumulative pole sums |Re β| > 3 (sum of 3+ retarded poles),
    # ``exp(β · L)`` in the closed form's lower-bound term overflows
    # past the 600 real-exponent threshold and the path bails to
    # scipy.  A physical bound — earliest external time minus a few
    # correlation times — is more than adequate.
    if L_value is not None:
        L = float(L_value)
    else:
        earliest_ext = min(list(free_ext_vals) + [0.0])
        L = earliest_ext - POSET_PHYSICAL_MARGIN

    upper_per_var = _causal_poset_consistent_scalar_upper(poset)

    # Pre-extract per-edge linear data (constant in pole tuple).
    c_ext_per_edge = [
        float(c0) + sum(
            float(a_ext[j]) * float(free_ext_vals[j])
            for j in range(len(a_ext))
        )
        for (a_int, a_ext, c0) in subset_constraint_data
    ]
    a_int_per_edge = [
        tuple(float(a_int[i]) for i in range(m))
        for (a_int, _a_ext, _c0) in subset_constraint_data
    ]

    pref = complex(prefactor_complex)
    # Experimental: accumulate the pole-tuple sum in mpmath to preserve
    # precision when the per-tuple terms have large opposite-signed
    # magnitudes that cancel down to a small result.  When the flag is
    # off (default), behaviour is bit-identical to the float64 path.
    use_mp_accum = USE_POSET_MPMATH_ACCUMULATION
    if use_mp_accum:
        from mpmath import mp as _mp, mpc as _mpc
        _saved_dps = _mp.dps
        _mp.dps = 50
        total_mp = _mpc(0, 0)
    total = 0.0 + 0.0j
    extensions = list(_enumerate_linear_extensions(poset))
    if not extensions:
        return _bail('poset_no_extension')

    # Pre-resolve the chain-top upper bound per extension.
    # The upper-bound fallback stays at bbox_cap — retarded β gives
    # Re(β · U) < 0 here, which the exp underflows safely; the
    # closed-form's truncation at U=bbox_cap is negligible for typical
    # Hawkes pole magnitudes.  The lower-bound fix (above) is what
    # addresses the overflow guard firing in the closed form.
    upper_for_ext = [
        upper_per_var.get(sigma[m - 1], float(bbox_cap))
        for sigma in extensions
    ]
    # Pre-resolve the per-position upper map per extension.  Maps
    # chain position k → scalar upper on σ[k] (for the intermediate-
    # upper-aware chain integrator below).
    upper_per_position_per_ext = []
    for sigma in extensions:
        upp = {}
        for k in range(m):
            v = sigma[k]
            if v in upper_per_var:
                upp[k] = float(upper_per_var[v])
        upper_per_position_per_ext.append(upp)

    # ── Plan-cache fast path (Stage 4a-plan, 2026-05-15) ────────────
    # When the caller threads a pre-built plan through, the per-tuple
    # ``alphas_orig`` and the γ decomposition have been computed once
    # at subset setup.  Per call we only contract γ_slope with
    # free_ext_vals and reorder alphas per linear extension.  The
    # poset structure, scalar bounds, linear extensions and cut tuples
    # are all τ-dependent (they read free_ext_vals via c_eff) and
    # stay above this branch — they're rebuilt per call regardless.
    if plan is not None:
        pole_iter = plan['pole_tuples']
        alphas_per_tuple = plan['alphas_per_tuple']
        gamma_const_per_tuple = plan['gamma_const_per_tuple']
        gamma_slope_per_tuple_per_ext = plan[
            'gamma_slope_per_tuple_per_ext'
        ]
        n_ext = len(free_ext_vals)
        # M4 (A2): the (L, uppers, U) memo-key suffix depends on the
        # extension only, so build it once here, not per tuple group.
        # ``None`` (memo off, or an unkeyable input) = the callee decides.
        chain_key_suffixes = [None] * len(extensions)
        if _chain_uppers_memo_on():
            for i_ext, (U_ext, upp_per_pos) in enumerate(
                    zip(upper_for_ext, upper_per_position_per_ext)):
                try:
                    chain_key_suffixes[i_ext] = _chain_uppers_key_suffix(
                        L, upp_per_pos, float(U_ext))
                except (TypeError, ValueError, OverflowError):
                    pass
        # Iterate GROUPS of pole tuples sharing an alpha vector: the chain
        # integral is evaluated once per group with the group's summed
        # prefactor, instead of once per tuple.  See ``_build_modesum_plan``.
        for alphas_orig, t_idxs in plan['tuple_groups']:
            term_const = 0.0 + 0.0j
            for t_idx in t_idxs:
                C_prod = pole_iter[t_idx][0]
                gamma = gamma_const_per_tuple[t_idx]
                slope_row = gamma_slope_per_tuple_per_ext[t_idx]
                for j in range(n_ext):
                    gamma = gamma + slope_row[j] * free_ext_vals[j]
                if gamma.real > 600.0:
                    return _bail('poset_gamma_overflow')
                try:
                    term_const = term_const + pref * C_prod * cmath.exp(gamma)
                except (OverflowError, ValueError):
                    return _bail('poset_exp_overflow')
            if term_const == 0:
                continue
            for sigma, U_ext, upp_per_pos, key_suffix in zip(
                    extensions, upper_for_ext,
                    upper_per_position_per_ext, chain_key_suffixes):
                alphas_chain = [alphas_orig[sigma[k]] for k in range(m)]
                chain_val = _chain_with_intermediate_uppers(
                    alphas_chain, L, upp_per_pos, float(U_ext),
                    _key_suffix=key_suffix,
                )
                if chain_val is None:
                    _RUNTIME_COUNTERS['poset_returned_none_total'] += 1
                    _RUNTIME_COUNTERS[
                        'chain_simplex_polynomial_returned_none'
                    ] += 1
                    if use_mp_accum:
                        _mp.dps = _saved_dps
                    return _bail('poset_chain_none')
                if use_mp_accum:
                    _term_f64 = term_const * chain_val
                    total_mp = total_mp + _mpc(_term_f64.real, _term_f64.imag)
                else:
                    total += term_const * chain_val
        if use_mp_accum:
            _result = complex(float(total_mp.real), float(total_mp.imag))
            _mp.dps = _saved_dps
            return _result
        return total

    # ── Legacy path (no plan) ─────────────────────────────────────
    pole_iter = (
        pole_tuples if pole_tuples is not None
        else _enumerate_pole_tuples(smooth_edge_modes)
    )
    for C_prod, lambdas in pole_iter:
        # α_v for each ORIGINAL integration variable.
        alphas_orig = [0.0 + 0.0j] * m
        gamma = 0.0 + 0.0j
        for e in range(n_smooth):
            lam = lambdas[e]
            for v in range(m):
                alphas_orig[v] += lam * a_int_per_edge[e][v]
            gamma += lam * c_ext_per_edge[e]
        # Overflow guard on the γ-prefactor.  Stage 4a optim
        # (2026-05-15): only positive Re(γ) overflows ``cmath.exp``;
        # negative direction underflows to 0 (correct for decayed
        # integrand).  Matches the polygon/interval guards.
        if gamma.real > 600.0:
            return _bail('poset_gamma_overflow')
        try:
            term_const = pref * C_prod * cmath.exp(gamma)
        except (OverflowError, ValueError):
            return _bail('poset_exp_overflow')
        if term_const == 0:
            continue
        # Sum across linear extensions of the poset.
        for sigma, U_ext, upp_per_pos in zip(
                extensions, upper_for_ext, upper_per_position_per_ext):
            alphas_chain = [alphas_orig[sigma[k]] for k in range(m)]

            # Check whether any non-maximal variable has a scalar upper.
            # If so, route through the intermediate-uppers helper which
            # handles the chain split via 2^p case enumeration.  When
            # all scalar uppers (if any) sit on the chain-top variable
            # σ[m-1], the helper short-circuits to the standard chain
            # closed form (no case enumeration).  This subsumes the
            # old "maximality bail" — we never return None on that
            # ground anymore.
            chain_val = _chain_with_intermediate_uppers(
                alphas_chain, L, upp_per_pos, float(U_ext),
            )
            if chain_val is None:
                # All sub-pieces overflowed.  Fall back to scipy.
                _RUNTIME_COUNTERS['poset_returned_none_total'] += 1
                _RUNTIME_COUNTERS[
                    'chain_simplex_polynomial_returned_none'
                ] += 1
                if use_mp_accum:
                    _mp.dps = _saved_dps
                return _bail('poset_chain_none')
            if use_mp_accum:
                _term_f64 = term_const * chain_val
                total_mp = total_mp + _mpc(_term_f64.real, _term_f64.imag)
            else:
                total += term_const * chain_val
    if use_mp_accum:
        _result = complex(float(total_mp.real), float(total_mp.imag))
        _mp.dps = _saved_dps
        return _result
    return total


# ───────────────────────────────────────────────────────────────────────
# Exact DBM route for m≥3 subsets (M3; ``USE_DBM_FALLBACK``)
# ───────────────────────────────────────────────────────────────────────
from engine.integration.time_domain import dbm_integral as _dbm


def _integrate_subset_dbm(
    smooth_edge_modes,
    prefactor_complex,
    subset_constraint_data,
    free_ext_vals,
    m,
    pole_tuples=None,
    plan=None,
    row_kinds=None,
    tie_ctx=None,
    cap=None,
    poset_bail_reason=None,
):
    r"""Exact ``∫_{region ∩ box} pref · Π_e [Σ_α C_α exp(λ_α · Δt_e)] ds``
    for an m ≥ 3 subset whose poset path bailed: the DBM route
    (``dbm_integral.integrate_exp_sum``).

    The integrand and its arguments are those of
    ``_integrate_nd_polytope_poset_modesum`` (one row of
    ``subset_constraint_data`` per smooth edge; the pole tuples from
    ``plan``, ``pole_tuples`` or ``smooth_edge_modes``).  A direction the
    rows leave open is closed at ±``cap``, ``_nquad_outer_cap()`` (the
    scipy fallback's ``OUTER_CAP``) unless given; a finite bound is kept as
    it is, even beyond the box.  The box is absolute (from the time
    origin), as in the legacy scipy routine: a region bounded above near or
    below -cap with a time open below is truncated or emptied by it.

    Rows: a constant row (zero normal) is decided by the Θ(0) = 0 rule with
    the external-time tie order (``_const_row_decision`` with ``tie_ctx``):
    EMPTY returns 0j, DROP leaves the geometry (its edge still contributes
    ``exp(λ·c_eff)``).  Every other row enters the region with its resolved
    shift ``c_eff``; a tie needs no rule there (the integral is continuous
    in such a shift).

    Returns ``complex``, or ``None`` (through ``_bail``) when the DBM cannot
    take the subset: a row that is not a difference row
    (``'dbm_not_difference_rows'``, e.g. a ConvVertex 3-term row), rows
    that do not match the modes (``'dbm_rows_mismatch'``), a non-finite
    shift or exponent (``'dbm_nonfinite'``), an overflowing final term
    (``'dbm_overflow'``), more than ``dbm_integral.MAX_CASES`` elimination
    cases (``'dbm_too_many_cases'``) or a closed form whose rounding error
    estimate exceeds max(1e-10 |value|, 1e-14 × Σ|pole-tuple coefficient|)
    (``'dbm_ill_conditioned'``: close poles, a thin region).  The caller then goes on to
    ``_integrate_polytope``.  ``poset_bail_reason`` (why the poset path
    bailed) is only counted (``dbm_answered_p3``).
    """
    import cmath
    _RUNTIME_COUNTERS['dbm_attempted'] += 1
    n_rows = len(subset_constraint_data)
    # Row shapes first, so that a region outside the DBM's scope (e.g. a
    # ConvVertex 3-term row) is counted as such even when its rows and
    # modes do not match either (a ConvVertex τ box row has no mode).
    for (a_int, _a_ext, _c0) in subset_constraint_data:
        nz = [float(a) for a in a_int if abs(float(a)) > _ROW_COEF_ATOL]
        if len(nz) > 2 or (len(nz) == 2 and nz[0] != -nz[1]):
            _RUNTIME_COUNTERS['dbm_declined_rows'] += 1
            return _bail('dbm_not_difference_rows')
    if smooth_edge_modes is not None and len(smooth_edge_modes) != n_rows:
        _RUNTIME_COUNTERS['dbm_declined_other'] += 1
        return _bail('dbm_rows_mismatch')
    n_ext = len(free_ext_vals)
    rows = []
    c_eff_per_edge = []
    a_int_per_edge = []
    for idx, (a_int, a_ext, c0) in enumerate(subset_constraint_data):
        c_eff = float(c0) + sum(
            float(a_ext[j]) * float(free_ext_vals[j])
            for j in range(len(a_ext))
        )
        a_f = tuple(float(a_int[i]) for i in range(m))
        c_eff_per_edge.append(c_eff)
        a_int_per_edge.append(a_f)
        verdict, _basis = _const_row_decision(
            a_int, a_ext, c0, free_ext_vals, _row_kind(row_kinds, idx),
            tie_ctx)
        if verdict == 'EMPTY':
            _RUNTIME_COUNTERS['dbm_answered'] += 1
            _RUNTIME_COUNTERS['dbm_empty'] += 1
            if poset_bail_reason == 'poset_lower_not_inherited':
                _RUNTIME_COUNTERS['dbm_answered_p3'] += 1
            return 0.0 + 0.0j
        if verdict == 'DROP':
            continue
        if not math.isfinite(c_eff):
            _RUNTIME_COUNTERS['dbm_declined_other'] += 1
            return _bail('dbm_nonfinite')
        rows.append((a_f, c_eff))

    # Seeds (C, E, β): one per pole tuple, E = γ kept in the log domain
    # (``integrate_exp_sum`` merges the tuples that share β).
    pref = complex(prefactor_complex)
    seeds = []
    if plan is not None:
        pt = plan['pole_tuples']
        g_const = plan['gamma_const_per_tuple']
        g_slope = plan['gamma_slope_per_tuple_per_ext']
        for alphas_orig, t_idxs in plan['tuple_groups']:
            for t_idx in t_idxs:
                gamma = g_const[t_idx]
                slope_row = g_slope[t_idx]
                for j in range(n_ext):
                    gamma = gamma + slope_row[j] * free_ext_vals[j]
                seeds.append((pref * pt[t_idx][0], gamma, alphas_orig))
    else:
        pole_iter = (
            pole_tuples if pole_tuples is not None
            else _enumerate_pole_tuples(smooth_edge_modes)
        )
        for C_prod, lambdas in pole_iter:
            if len(lambdas) != n_rows:
                _RUNTIME_COUNTERS['dbm_declined_other'] += 1
                return _bail('dbm_rows_mismatch')
            alphas_orig = [0.0 + 0.0j] * m
            gamma = 0.0 + 0.0j
            for e in range(n_rows):
                lam = lambdas[e]
                a_e = a_int_per_edge[e]
                for v in range(m):
                    alphas_orig[v] += lam * a_e[v]
                gamma += lam * c_eff_per_edge[e]
            seeds.append((pref * C_prod, gamma, alphas_orig))
    for (C, E, _b) in seeds:
        if not (cmath.isfinite(C) and cmath.isfinite(E)):
            _RUNTIME_COUNTERS['dbm_declined_other'] += 1
            return _bail('dbm_nonfinite')

    # The integrand's scale for the conditioning test's absolute floor:
    # Σ |pole-tuple coefficient| (prefactor included), the bound of the
    # integrand where every Δt >= 0 and Re λ <= 0.
    scale = sum(abs(C) for (C, _E, _b) in seeds)
    res = _dbm.integrate_exp_sum(
        rows, m, seeds, _nquad_outer_cap() if cap is None else cap,
        scale=scale)
    if res.status == _dbm.STATUS_NOT_DBM:
        _RUNTIME_COUNTERS['dbm_declined_rows'] += 1
        return _bail('dbm_not_difference_rows')
    if res.status == _dbm.STATUS_OVERFLOW:
        _RUNTIME_COUNTERS['dbm_declined_overflow'] += 1
        return _bail('dbm_overflow')
    if res.status == _dbm.STATUS_TOO_MANY_CASES:
        _RUNTIME_COUNTERS['dbm_declined_other'] += 1
        return _bail('dbm_too_many_cases')
    if res.status == _dbm.STATUS_ILL_CONDITIONED:
        _RUNTIME_COUNTERS['dbm_declined_ill_conditioned'] += 1
        return _bail('dbm_ill_conditioned')
    _RUNTIME_COUNTERS['dbm_answered'] += 1
    if res.status == _dbm.STATUS_EMPTY:
        _RUNTIME_COUNTERS['dbm_empty'] += 1
    elif res.error_ratio > _RUNTIME_COUNTERS['dbm_error_ratio_max']:
        _RUNTIME_COUNTERS['dbm_error_ratio_max'] = res.error_ratio
    if res.thin_retry:
        _RUNTIME_COUNTERS['dbm_answered_thin_retry'] += 1
    if poset_bail_reason == 'poset_lower_not_inherited':
        _RUNTIME_COUNTERS['dbm_answered_p3'] += 1
    return res.value


def _integrate_1d_polytope_modesum(
    smooth_edge_modes,
    prefactor_complex,
    subset_constraint_data,
    free_ext_vals,
    bbox_cap=None,
    pole_tuples=None,
    plan=None,
    row_kinds=None,
    tie_ctx=None,
):
    r"""Analytic ``∫_L^U Π_e [Σ_α C_α exp(λ_α · Δt_e)] · prefactor ds``
    for ``m = 1``.

    After pole-expansion the integrand is a sum of single-exponential
    terms ``A · exp(α_s · s + γ)`` where
        α_s = Σ_e λ_α_e · a_int_e[0]
        γ   = Σ_e λ_α_e · c_ext_e,  c_ext_e = c0_e + Σ_j a_ext_e[j]·t_free[j]
    Each term is integrated in closed form over the polytope interval
    ``[L, U]``.

    The interval bounds come from the smooth-edge retardation
    constraints ``a_int_e[0]·s + c_ext_e > 0``.  Unbounded sides are
    tracked as ±∞: the corresponding closed-form boundary term
    evaluates to 0 if the integrand decays in that direction
    (``sign(Re α_s)`` matches), else ``None`` (caller falls back to
    scipy.quad which handles the unbounded endpoint via the standard
    adaptive-quadrature substitution).

    ``pole_tuples`` (optional): pre-built iterable of ``(C_prod,
    lambdas)`` pairs that replaces ``_enumerate_pole_tuples
    (smooth_edge_modes)``.  Used by the grouped Phase J path to
    inject merged residues ``B_α = Σ_td cp_td · Π_e C^{(td)}_{α_e, e}``.

    Returns ``complex`` or ``None`` on overflow / divergent unbounded
    integrand.

    Constant rows: THETA0_CONST_ROW_MODE 'ito' (call time) uses
    ``_const_row_verdict`` (EMPTY -> 0j, DROP -> no bound; the row still
    enters γ); 'legacy_clip' the pre-M1 ``|a| < 1e-15``, ``c_eff <= 0``
    rule.  ``bbox_cap`` is unused (both sides are tracked exactly); it is
    accepted for signature symmetry.  ``row_kinds``: the rows' provenance;
    ``tie_ctx``: see ``_tie_order_sign``.
    """
    import cmath
    import math
    n_smooth = len(smooth_edge_modes)
    _RUNTIME_COUNTERS['interval_attempted'] += 1
    bbox_cap = _resolve_bbox_cap(bbox_cap)          # (unused; see docstring)
    if len(subset_constraint_data) != n_smooth:
        _RUNTIME_COUNTERS['interval_returned_none'] += 1
        return _bail('interval_rows_mismatch')

    # Resolve the feasible interval — track unboundedness exactly.
    legacy = _theta0_legacy()
    L = -math.inf
    U = +math.inf
    for idx, (a_int, a_ext, c0) in enumerate(subset_constraint_data):
        a = float(a_int[0]) if a_int else 0.0
        c_eff = float(c0) + sum(
            float(a_ext[i]) * float(free_ext_vals[i])
            for i in range(len(a_ext))
        )
        if not legacy:
            if abs(a) <= _ROW_COEF_ATOL:
                if _const_row_verdict(
                        (a,), a_ext, c0, free_ext_vals,
                        _row_kind(row_kinds, idx), tie_ctx) == 'EMPTY':
                    return 0.0 + 0.0j
                continue
        elif abs(a) < 1e-15:
            if c_eff <= 0:
                return 0.0 + 0.0j
            continue
        bound = -c_eff / a
        if a > 0:
            if bound > L:
                L = bound
        else:
            if bound < U:
                U = bound
    if L >= U:
        return 0.0 + 0.0j
    L_inf = math.isinf(L)
    U_inf = math.isinf(U)

    pref = complex(prefactor_complex)
    total = 0.0 + 0.0j

    # ── Plan-cache fast path (Stage 4a-plan, 2026-05-15) ────────────
    # When the caller threads a pre-built plan through, the per-tuple
    # ``α_s`` and ``γ`` decomposition has been computed once at
    # subset setup — only the γ slope contraction with free_ext_vals
    # remains.  See ``_build_modesum_plan``.
    if plan is not None:
        pole_iter = plan['pole_tuples']
        alphas_per_tuple = plan['alphas_per_tuple']
        gamma_const_per_tuple = plan['gamma_const_per_tuple']
        gamma_slope_per_tuple_per_ext = plan[
            'gamma_slope_per_tuple_per_ext'
        ]
        n_ext = len(free_ext_vals)
        for t_idx, (C_prod, _lambdas) in enumerate(pole_iter):
            alpha_s = alphas_per_tuple[t_idx][0]
            gamma = gamma_const_per_tuple[t_idx]
            slope_row = gamma_slope_per_tuple_per_ext[t_idx]
            for j in range(n_ext):
                gamma = gamma + slope_row[j] * free_ext_vals[j]
            # Same overflow guard / closed-form as the non-plan path
            # below.  Duplicated rather than fall-through so the hot
            # path stays branch-free on (plan is not None).
            if gamma.real > 600.0:
                return _bail('interval_gamma_overflow')
            try:
                term_const = pref * C_prod
            except (OverflowError, ValueError):
                return _bail('interval_exp_overflow')
            if term_const == 0:
                continue
            try:
                if abs(alpha_s) < 1e-15:
                    if L_inf or U_inf:
                        return _bail('interval_divergent_flat')
                    contrib = (U - L) * cmath.exp(gamma)
                else:
                    if U_inf:
                        if alpha_s.real >= 0:
                            return _bail('interval_divergent_upper')
                        term_U = 0.0 + 0.0j
                    else:
                        arg = alpha_s * U + gamma
                        if arg.real > 600.0:
                            return _bail('interval_arg_overflow')
                        term_U = cmath.exp(arg)
                    if L_inf:
                        if alpha_s.real <= 0:
                            return _bail('interval_divergent_lower')
                        term_L = 0.0 + 0.0j
                    else:
                        arg = alpha_s * L + gamma
                        if arg.real > 600.0:
                            return _bail('interval_arg_overflow')
                        term_L = cmath.exp(arg)
                    contrib = (term_U - term_L) / alpha_s
            except (OverflowError, ValueError):
                return _bail('interval_exp_overflow')
            total += term_const * contrib
        return total

    # ── Legacy path (no plan): rebuild per-edge data per call ──────
    c_ext_per_edge = [
        float(c0) + sum(
            float(a_ext[j]) * float(free_ext_vals[j])
            for j in range(len(a_ext))
        )
        for (a_int, a_ext, c0) in subset_constraint_data
    ]
    a_int_per_edge = [
        float(a_int[0]) if a_int else 0.0
        for (a_int, _a_ext, _c0) in subset_constraint_data
    ]

    pole_iter = (
        pole_tuples if pole_tuples is not None
        else _enumerate_pole_tuples(smooth_edge_modes)
    )
    for C_prod, lambdas in pole_iter:
        alpha_s = 0.0 + 0.0j
        gamma = 0.0 + 0.0j
        for e in range(n_smooth):
            lam = lambdas[e]
            alpha_s += lam * a_int_per_edge[e]
            gamma += lam * c_ext_per_edge[e]
        # Overflow guard on the γ-prefactor (matches polygon path).
        # Stage 4a optim (2026-05-15): only positive Re(γ) overflows
        # ``cmath.exp``; negative direction underflows to 0 (which
        # gives the correct result for a fully-decayed integrand).
        if gamma.real > 600.0:
            return _bail('interval_gamma_overflow')
        try:
            term_const = pref * C_prod
        except (OverflowError, ValueError):
            return _bail('interval_exp_overflow')
        if term_const == 0:
            continue
        try:
            if abs(alpha_s) < 1e-15:
                # α_s ≈ 0: integrand is constant exp(γ) in s.
                if L_inf or U_inf:
                    return _bail('interval_divergent_flat')  # diverges
                contrib = (U - L) * cmath.exp(gamma)
            else:
                if U_inf:
                    if alpha_s.real >= 0:
                        # would diverge at +∞
                        return _bail('interval_divergent_upper')
                    term_U = 0.0 + 0.0j
                else:
                    arg = alpha_s * U + gamma
                    if arg.real > 600.0:
                        return _bail('interval_arg_overflow')
                    term_U = cmath.exp(arg)
                if L_inf:
                    if alpha_s.real <= 0:
                        # would diverge at -∞
                        return _bail('interval_divergent_lower')
                    term_L = 0.0 + 0.0j
                else:
                    arg = alpha_s * L + gamma
                    if arg.real > 600.0:
                        return _bail('interval_arg_overflow')
                    term_L = cmath.exp(arg)
                contrib = (term_U - term_L) / alpha_s
        except (OverflowError, ValueError):
            return _bail('interval_exp_overflow')
        total += term_const * contrib
    return total


def _integrate_2d_polygon_modesum(
    smooth_edge_modes,
    prefactor_complex,
    subset_constraint_data,
    free_ext_vals,
    bbox_cap=None,
    pole_tuples=None,
    plan=None,
    row_kinds=None,
    tie_ctx=None,
):
    r"""Analytic ∫∫_polygon Π_e [Σ_α C_α exp(λ_α · Δt_e)] · prefactor
                  ds_0 ds_1.

    Each per-edge mode sum is pole-expanded; the resulting sum of
    single-exponential terms ``A · exp(α·s_0 + β·s_1 + γ)`` is
    integrated analytically over the polygon defined by
    ``subset_constraint_data``.

    Returns ``complex`` or ``None`` if construction fails (e.g.
    polygon empty / degenerate, or the per-edge data is missing).

    Δt_e for each smooth edge is expressed as
        Δt_e = c0_e + a_int_e[0]·s_0 + a_int_e[1]·s_1 + Σ_j a_ext_e[j]·t_free[j]
    Substituting into ``Σ_α λ_α · Δt_e`` and rearranging gives the
    exponent ``γ + α_s·s_0 + β_s·s_1`` where
        α_s = Σ_e a_int_e[0] · λ_α_e
        β_s = Σ_e a_int_e[1] · λ_α_e
        γ   = Σ_e λ_α_e · (c0_e + Σ_j a_ext_e[j]·t_free[j])

    ``pole_tuples`` (optional): pre-built iterable of ``(C_prod, lambdas)``
    pairs that replaces ``_enumerate_pole_tuples(smooth_edge_modes)``.
    Used by the grouped Phase J path
    (``engine.integration.time_domain.grouped_integral``) to inject a
    merged-residue tensor ``B_α = Σ_td cp_td · Π_e C^{(td)}_{α_e, e}``
    in place of the per-edge Cartesian product.  When ``None``, the
    per-diagram default iterator runs.

    ``bbox_cap=None`` resolves at call time (``_resolve_bbox_cap``).  The
    polygon (shared by the plan and the no-plan branch) follows
    THETA0_CONST_ROW_MODE through ``_polygon_from_2d_constraints``; a
    constant DROP row only skips the clip and still enters γ.
    ``row_kinds``: the rows' provenance (``None``: all 'edge');
    ``tie_ctx``: see ``_tie_order_sign``.
    """
    import cmath
    n_smooth = len(smooth_edge_modes)
    _RUNTIME_COUNTERS['polygon_attempted'] += 1
    bbox_cap = _resolve_bbox_cap(bbox_cap)
    if len(subset_constraint_data) != n_smooth:
        # Smooth-edges-to-constraints mismatch shouldn't happen — the
        # caller built both from ``smooth_edges`` in lock-step.  Bail
        # to scipy.nquad fallback.
        _RUNTIME_COUNTERS['polygon_returned_none'] += 1
        return _bail('polygon_rows_mismatch')

    # Polygon is shared across all pole tuples.
    polygon = _polygon_from_2d_constraints(
        subset_constraint_data, free_ext_vals, bbox_cap,
        row_kinds=row_kinds, tie_ctx=tie_ctx,
    )
    if len(polygon) < 3:
        # Empty or degenerate polygon → integral is zero.
        return 0.0 + 0.0j

    # Fan triangulation.
    triangles = [
        (polygon[0], polygon[i], polygon[i + 1])
        for i in range(1, len(polygon) - 1)
    ]

    total = 0.0 + 0.0j
    pref = complex(prefactor_complex)

    # ── Plan-cache fast path (Stage 4a-plan, 2026-05-15) ────────────
    # Per-tuple α_s, β_s and γ decomposition pre-computed at subset
    # setup; only γ_slope contraction and the per-triangle integral
    # remain per τ.  See ``_build_modesum_plan``.
    if plan is not None:
        pole_iter = plan['pole_tuples']
        alphas_per_tuple = plan['alphas_per_tuple']
        gamma_const_per_tuple = plan['gamma_const_per_tuple']
        gamma_slope_per_tuple_per_ext = plan[
            'gamma_slope_per_tuple_per_ext'
        ]
        n_ext = len(free_ext_vals)
        for t_idx, (C_prod, _lambdas) in enumerate(pole_iter):
            alpha_s, beta_s = alphas_per_tuple[t_idx]
            gamma = gamma_const_per_tuple[t_idx]
            slope_row = gamma_slope_per_tuple_per_ext[t_idx]
            for j in range(n_ext):
                gamma = gamma + slope_row[j] * free_ext_vals[j]
            if gamma.real > 600.0:
                return _bail('polygon_gamma_overflow')
            try:
                term_const = pref * C_prod * cmath.exp(gamma)
            except (OverflowError, ValueError):
                return _bail('polygon_exp_overflow')
            if term_const == 0:
                continue
            tri_sum = 0.0 + 0.0j
            for (v0, v1, v2) in triangles:
                tri_contrib = _exp_over_triangle(
                    v0, v1, v2, alpha_s, beta_s
                )
                if tri_contrib is None:
                    return _bail('polygon_triangle_guard')
                tri_sum += tri_contrib
            total += term_const * tri_sum
        return total

    # ── Legacy path (no plan): rebuild per-edge data per call ──────
    # Precompute per-edge "ext-time c-contribution":
    #   c_ext_e = c0_e + Σ_j a_ext_e[j] · t_free[j]
    # so that γ for a given pole tuple is Σ_e λ_α_e · c_ext_e.
    c_ext_per_edge = [
        float(c0) + sum(
            float(a_ext[j]) * float(free_ext_vals[j])
            for j in range(len(a_ext))
        )
        for (a_int, a_ext, c0) in subset_constraint_data
    ]
    a_int_per_edge = [
        (float(a_int[0]), float(a_int[1]))
        for (a_int, _a_ext, _c0) in subset_constraint_data
    ]

    pole_iter = (
        pole_tuples if pole_tuples is not None
        else _enumerate_pole_tuples(smooth_edge_modes)
    )
    for C_prod, lambdas in pole_iter:
        # α_s, β_s, γ for this pole tuple.
        alpha_s = 0.0 + 0.0j
        beta_s = 0.0 + 0.0j
        gamma = 0.0 + 0.0j
        for e in range(n_smooth):
            lam = lambdas[e]
            a0, a1 = a_int_per_edge[e]
            alpha_s += lam * a0
            beta_s += lam * a1
            gamma += lam * c_ext_per_edge[e]
        # Overflow guard on the γ-prefactor.  cmath.exp overflows for
        # Re(γ) > ~709 and underflows to 0 for Re(γ) < ~-745;
        # underflow is the right behaviour (term decays to 0).
        # Stage 4a optim (2026-05-15): check only the positive-
        # overflow side, matching the fixed-direction guard inside
        # ``_exp_over_triangle``.
        if gamma.real > 600.0:
            return _bail('polygon_gamma_overflow')
        try:
            term_const = pref * C_prod * cmath.exp(gamma)
        except (OverflowError, ValueError):
            return _bail('polygon_exp_overflow')
        if term_const == 0:
            continue
        # Triangle sum.  ``_exp_over_triangle`` returns ``None`` when
        # any per-term exp would overflow — propagate that as a
        # whole-subset fallback signal.
        tri_sum = 0.0 + 0.0j
        for (v0, v1, v2) in triangles:
            tri_contrib = _exp_over_triangle(v0, v1, v2, alpha_s, beta_s)
            if tri_contrib is None:
                return _bail('polygon_triangle_guard')
            tri_sum += tri_contrib
        total += term_const * tri_sum
    return total


# ───────────────────────────────────────────────────────────────────────
# Quadrature accuracy knob
# ───────────────────────────────────────────────────────────────────────
# Controls scipy.integrate.quad / nquad parameters for the vertex-time
# integrals. Loosen these for fast iterative checks; tighten for
# publication-quality results.
#
# Usage from a notebook cell (BEFORE running the numerics cell):
#   from engine.integration.time_domain import final_integral
#   final_integral.QUAD_OPTS = {'limit': 30, 'epsrel': 1e-3}
#
QUAD_OPTS = {
    'limit': 200,      # max subintervals for scipy.integrate.quad / nquad
}

# ───────────────────────────────────────────────────────────────────────
# Cumulant-kernel τ_v integration cap (non-local noise sources)
# ───────────────────────────────────────────────────────────────────────
# Diagrams with a NoiseSourceType vertex carry an extra integration
# variable τ_v parametrising the relative time between the source's
# legs (per-leg time map for non-local cumulant kernels).  The
# kernel itself decays on its natural timescale (e.g., σ for a
# Gaussian), so integrating τ_v over a half-infinite range
# (retard_L, +∞) — which is what the polytope alone gives —
# leaves scipy.quad's tan-substitution coordinate transform free
# to compress the kernel's central peak near the boundary, where
# adaptive sampling intermittently misses it.  Capping τ_v ∈
# (-CAP, +CAP) collapses the range to a finite interval where
# adaptive quadrature is well-behaved.  ±50 is safe for kernels
# with σ ≤ 5; loosen to ±200 (the polytope OUTER_CAP) if your
# kernel has heavy tails:
#   from engine.integration.time_domain import final_integral
#   final_integral.TAU_KERNEL_CAP = 200.0
TAU_KERNEL_CAP = 50.0


# ───────────────────────────────────────────────────────────────────────
# Heaviside guard mode
# ───────────────────────────────────────────────────────────────────────
# The polytope integrators wrap their integrand in
# ``_make_heaviside_filtered_integrand`` which returns 0 whenever any
# retarded ``Δt_e`` constraint is violated.  This is a defensive belt-
# and-braces measure: the polytope BOUNDS we pass to scipy.quad / nquad
# should already constrain the integration to the feasible region, so
# the wrapper SHOULD be redundant — except in cases where the bounds
# fall back to ``±OUTER_CAP`` (m=2 without a pure-s_1 constraint, or
# m≥3 with deferred-inner constraints).  In those cases the cap is a
# superset of the true polytope and the filter is what enforces
# correctness.
#
# For paths where the bounds are EXACT (m=1, m=2 with pure_s_1), the
# wrapper is pure overhead — ~few µs per integrand call.  At millions
# of calls per τ sweep the cumulative cost is meaningful.
#
# Default: ``DEBUG_HEAVISIDE_GUARD = False`` skips the filter on the
# exact-bound paths; the cap-fallback paths always apply it regardless
# (correctness is non-negotiable).
#
# Set to ``True`` to force the filter on every path — useful when
# validating a refactor that touches the polytope bound logic and you
# want a belt-and-braces sanity check.
DEBUG_HEAVISIDE_GUARD = False


# ── M5 (P6): per-diagram setup levers (pure speed-ups, no number moves) ──
# Four independent levers cut the setup that ``integrate_diagram`` pays per
# diagram.  Each has a module flag (bool, read at CALL time, so
# ``monkeypatch.setattr`` works) and an environment variable read at import
# (1 | 0, also true/false, yes/no, on/off; default ON; anything else raises):
#
#   USE_SETUP_ZERO_EXIT   DAEDALUS_SETUP_ZERO_EXIT   L1: a diagram whose
#       prefactor is numerically exactly 0 returns a zero contribution before
#       any setup.
#   USE_SETUP_PROP_TD     DAEDALUS_SETUP_PROP_TD     L2: model-level data
#       built once per ``compute_correction_td`` call.
#   USE_SETUP_LAZY_SR     DAEDALUS_SETUP_LAZY_SR     L3: the SR objects only
#       the shot-noise / SR-integrand branches need are built on demand.
#   USE_AUT_MEMO (engine.diagrams.symmetry)  DAEDALUS_SETUP_AUT_MEMO   L5.
#
# ``DAEDALUS_PHASE_J_LEGACY_SETUP=1`` (read at call time) forces every one of
# them off.  It is NOT ``DAEDALUS_PHASE_J_LEGACY``: that one reproduces older
# NUMBERS, this one only restores the old setup path, which gives the same
# numbers.  Neither umbrella touches the other.  A lever on is ``np.array_equal``
# to the lever off on every result.
_SETUP_TRUE = ('', '1', 'true', 'yes', 'on')
_SETUP_FALSE = ('0', 'false', 'no', 'off')


def _initial_setup_flag(env_name, environ=None):
    """The import-time value of a setup-lever flag from ``environ`` (default
    ``os.environ``).  An unknown value raises."""
    env = _os.environ if environ is None else environ
    v = env.get(env_name, '').strip().lower()
    if v in _SETUP_TRUE:
        return True
    if v in _SETUP_FALSE:
        return False
    raise ValueError(f'{env_name}={v!r}: expected 1 or 0')


USE_SETUP_ZERO_EXIT = _initial_setup_flag('DAEDALUS_SETUP_ZERO_EXIT')
USE_SETUP_PROP_TD = _initial_setup_flag('DAEDALUS_SETUP_PROP_TD')


def _setup_lever_on(flag_name):
    """The module flag ``flag_name`` (call time, validated), unless the
    umbrella ``DAEDALUS_PHASE_J_LEGACY_SETUP`` is set."""
    if _env_truthy('DAEDALUS_PHASE_J_LEGACY_SETUP'):
        return False
    flag = globals()[flag_name]
    if flag is True or flag is False:
        return flag
    raise ValueError(f'final_integral.{flag_name}={flag!r}: '
                     f'expected True or False')


class _LazyDict(dict):
    """A dict some of whose keys hold a DEFERRED value: the key is present
    (``in``, ``len``, ``keys()`` and the insertion order are those of the
    eager dict) and its value is computed on the first read, then stored.
    Every read path resolves (``[]``, ``get``, ``items``, ``values``,
    ``copy``, ``pop``, iteration through ``dict(...)``); a pickle or deep
    copy resolves everything first and is a plain dict.  Thread safe in the
    way the memos are: two readers racing on one key both compute it and
    store equal values."""

    __slots__ = ('_thunks',)

    def __init__(self, items, thunks):
        dict.__init__(self, items)
        self._thunks = dict(thunks)

    def _force(self, key):
        th = self._thunks.get(key)
        if th is not None:
            dict.__setitem__(self, key, th())
            self._thunks.pop(key, None)

    def _force_all(self):
        for k in list(self._thunks):
            self._force(k)

    def __getitem__(self, key):
        self._force(key)
        return dict.__getitem__(self, key)

    def get(self, key, default=None):
        self._force(key)
        return dict.get(self, key, default)

    def pop(self, key, *default):
        self._force(key)
        return dict.pop(self, key, *default)

    def setdefault(self, key, default=None):
        self._force(key)
        return dict.setdefault(self, key, default)

    def __setitem__(self, key, value):
        self._thunks.pop(key, None)
        dict.__setitem__(self, key, value)

    def __delitem__(self, key):
        self._thunks.pop(key, None)
        dict.__delitem__(self, key)

    def __iter__(self):
        return dict.__iter__(self)      # a Python-level iter: no C fast copy

    def items(self):
        self._force_all()
        return dict.items(self)

    def values(self):
        self._force_all()
        return dict.values(self)

    def copy(self):
        self._force_all()
        return dict(dict.items(self))

    def __repr__(self):
        self._force_all()
        return dict.__repr__(self)

    def __eq__(self, other):
        self._force_all()
        return dict.__eq__(self, other)

    __hash__ = None

    def __reduce__(self):
        self._force_all()
        return (dict, (dict(dict.items(self)),))


# ───────────────────────────────────────────────────────────────────────
# Tree-level vertex-time integration
# ───────────────────────────────────────────────────────────────────────

def _loop_number_from_graph(typed_diagram):
    """Compute loop number from the diagram's graph structure.

    For a connected graph: L = |E| - |V| + 1.
    This avoids any frequency-domain dependency.
    """
    D = typed_diagram.prediagram[0]
    return D.num_edges() - D.num_verts() + 1


_UNSET = object()


class PropagatorTD:
    """Model-level data of one ``(propagator_data, num_params)``, built once
    per ``compute_correction_td`` call and shared by every
    ``integrate_diagram`` of that call (M5 L2, ``USE_SETUP_PROP_TD``).

    Holds, each built lazily and with the very conversions the per-diagram
    code uses (so every value is bit-identical to a per-diagram build):

    * ``G_t_obj()``   the ``build_G_t_matrix`` result (smooth SR matrix and
      the delta matrix);
    * ``poles()``     ``tuple(complex(CDF(SR(p))) * 1j)``, the mode λ's;
    * ``entry_modes(pi, ri)``  the tuple of ``(residue, λ)`` pairs of entry
      ``(pi, ri)``, residues ``complex(CDF(SR(C_mats[k][pi, ri])))``.  A
      conversion that fails is cached as ``None`` (the caller then returns
      ``None`` exactly as the per-diagram build does);
    * ``delta_coeff(pi, ri)``  ``G_t_delta_coeff(G_t_obj, pi, ri)``.

    SCOPE: one call.  ``matches`` recognises the data it was built from by
    identity (the propagator-data dict, ``num_params`` and the pole / residue
    lists); ``integrate_diagram`` builds locally, as without the object, when
    handed one that does not match (counter ``setup_prop_td_stale``).  Never
    keep one across calls and never key it by ``id()`` in a global: the
    spatial bridge re-solves the poles for each q sample, so an object that
    outlived its call would serve stale data.  Thread safe: builds hold the
    lock, the finished tables are read-only.
    """

    def __init__(self, propagator_data, num_params=None):
        self.propagator_data = propagator_data
        self.num_params = num_params
        self._pole_vals = propagator_data.get('pole_vals')
        self._C_mats = propagator_data.get('C_mats')
        self._lock = _threading.RLock()
        self._g_t_obj = _UNSET
        self._lam = _UNSET
        self._entries = {}
        self._deltas = {}

    def matches(self, propagator_data, num_params):
        pd = self.propagator_data
        return (propagator_data is pd and num_params is self.num_params
                and pd.get('pole_vals') is self._pole_vals
                and pd.get('C_mats') is self._C_mats)

    def G_t_obj(self):
        with self._lock:
            if self._g_t_obj is _UNSET:
                self._g_t_obj = build_G_t_matrix(
                    self.propagator_data, SR.var('_t_td_'),
                    num_params=self.num_params)
                _RUNTIME_COUNTERS['setup_prop_td_g_t_builds'] += 1
            return self._g_t_obj

    def poles(self):
        with self._lock:
            if self._lam is _UNSET:
                try:
                    self._lam = tuple(complex(CDF(SR(p))) * 1j
                                      for p in self._pole_vals)
                except Exception:
                    self._lam = None
            return self._lam

    def entry_modes(self, pi, ri):
        key = (pi, ri)
        with self._lock:
            hit = self._entries.get(key, _UNSET)
            if hit is not _UNSET:
                return hit
            lam = self.poles()
            if lam is None:
                modes = None
            else:
                try:
                    residues = tuple(
                        complex(CDF(SR(self._C_mats[k][pi, ri])))
                        for k in range(len(lam)))
                    modes = tuple(zip(residues, lam))
                except Exception:
                    modes = None
            self._entries[key] = modes
            _RUNTIME_COUNTERS['setup_prop_td_entry_builds'] += 1
            return modes

    def delta_coeff(self, pi, ri):
        key = (pi, ri)
        with self._lock:
            hit = self._deltas.get(key, _UNSET)
            if hit is _UNSET:
                hit = G_t_delta_coeff(self.G_t_obj(), pi, ri)
                self._deltas[key] = hit
            return hit


def _zero_exit_applies(typed_diagram, propagator_data, cp, external_fields,
                       leaves):
    """True iff ``integrate_diagram`` may return the zero result at once
    (M5 L1).  ``cp`` is the numeric prefactor (``num_params`` substituted).
    All of these must hold, else the full path runs unchanged:

    * no noise-source (``cumulant_specs``) and no ``ConvVertexType`` vertex:
      their kernels are substituted into the prefactor, so the plain
      prefactor decides nothing for them;
    * ``external_fields`` is given and matches the leaves (the full path
      warns about an unmapped mixed-field diagram; that stays);
    * the propagator has at least one pole (a pole-free propagator raises
      ``PoleFreePropagatorError`` / warns in the full path; that stays);
    * ``complex(CDF(cp))`` evaluates and is exactly 0 -- the very test the
      subset loop applies to ``cp`` times the delta coefficients (an
      exception or leftover symbols: no early exit).
    """
    if external_fields is None or len(external_fields) != len(leaves):
        return False
    for vtype in (getattr(typed_diagram, 'vertex_assignments', None)
                  or {}).values():
        if isinstance(vtype, ConvVertexType):
            return False
        if isinstance(vtype, NoiseSourceType) and vtype.cumulant_specs:
            return False
    try:
        if not len(propagator_data.get('pole_vals') or ()):
            return False
    except (AttributeError, TypeError):
        return False
    try:
        zero = complex(CDF(SR(cp))) == 0
    except Exception:
        return False
    return bool(zero)


def _zero_exit_result(D, leaves, ext_time_vars, diag_serial, rerun):
    """The result ``integrate_diagram`` returns for a diagram whose every
    subset was skipped, built without the setup (M5 L1).  The keys the
    pipeline reads are eager; ``stripped_integrand``, ``constraints`` and
    ``edge_info`` (display data that need the propagator) are filled on first
    read by re-running the full path (``rerun``)."""
    _RUNTIME_COUNTERS['setup_zero_exit'] += 1
    leaf_set = set(leaves)
    integration_vars = [
        SR.var(f's_v{v}_td_', latex_name=rf's_{{v_{{{v}}}}}')
        for v in D.vertices() if v not in leaf_set]
    n_ext = len(ext_time_vars)

    def contribution(*ext_time_values):
        if len(ext_time_values) != n_ext:
            raise ValueError(
                f"contribution() expects {n_ext} positional "
                f"arguments (one per ext_time_var); got "
                f"{len(ext_time_values)}."
            )
        return 0.0 + 0.0j

    full = []

    def _full():
        if not full:
            full.append(rerun(diag_serial))
        return full[0]

    return _LazyDict(
        [('status', 'ok'),
         ('contribution', contribution),
         ('delta_contributions', []),
         ('integration_vars', integration_vars),
         ('stripped_integrand', None),
         ('constraints', None),
         ('edge_info', None),
         ('n_subsets_evaluated', 0),
         ('n_delta_contributions', 0),
         ('n_shotnoise_skipped', 0),
         ('subset_diagnostics', []),
         ('cumulant_prefactor', None),
         ('has_cumulant_kernel', False)],
        {'stripped_integrand': lambda: _full()['stripped_integrand'],
         'constraints': lambda: _full()['constraints'],
         'edge_info': lambda: _full()['edge_info']})


def integrate_diagram(
    typed_diagram,
    propagator_data,
    combined_prefactor,
    ext_time_vars,
    num_params=None,
    origin_leaf_idx=0,
    external_fields=None,
    representative_ir=None,  # deprecated, kept for backward compat
    edge_mode_sums_builder=None,
    prop_td=None,
    _setup_full=None,
):
    r"""
    Vertex-time integration for a typed Feynman diagram at ANY loop
    order, evaluated via explicit numerical quadrature.

    Previously this function was named ``integrate_tree_diagram`` and
    asserted tree-level.  The core algorithm -- assign a time to each
    internal vertex, integrate over those times, and enforce the
    retarded Heaviside on every edge -- works identically for loop
    diagrams because our enumeration produces DAGs (the "loop" in
    Feynman terminology is a topological cycle in the underlying
    undirected graph, not a cyclic directed path).  Multi-edges
    between the same vertex pair are already supported through the
    3-tuple ``(u, v, label)`` edge keys used by
    ``_lookup_prop_indices``.

    Returns a dict whose `contribution` value is a Python callable

        f(*ext_time_values) -> complex

    taking `k` positional arguments in **canonical** order: position i
    is the time of ``external_fields[i]``.  If `origin_leaf_idx` is
    not None, the value supplied at that position is ignored (it was
    pinned to zero during integrand construction).

    Parameters
    ----------
    typed_diagram : TypedDiagram
        The typed diagram.  Any loop order is accepted; the
        integrator treats every internal vertex as an integration
        variable and every edge as a retarded-propagator factor.
        Needed for the prediagram `D`, leaf list, and
        ``propagator_indices``.
    propagator_data : dict
        Must contain `'pole_vals'`, `'C_mats'`, and optionally
        `'D_delta'` (for delta-coefficient detection).
    combined_prefactor : SR or numeric
        Sum of scalar prefactors over diagrams in the kernel group.
    ext_time_vars : list of SR
        `k` external time variables in canonical order:
        ``ext_time_vars[i]`` is the time of ``external_fields[i]``.
    num_params : dict or None
        Numerical parameter substitutions for the propagator matrix AND
        the combined prefactor. Required if either the propagator
        entries or the prefactor contain free symbolic parameters — the
        JIT-compiled integrand cannot be built until every symbol
        except the integration variables and external times has been
        substituted.
    origin_leaf_idx : int or None
        Which canonical position to pin to zero (i.e., which entry of
        `external_fields` provides the base time). Default 0.
    external_fields : list of tuple or None
        The canonical external field list as specified by the user,
        e.g. ``[('dn',1), ('dn',1), ('dn',2)]``.  Used to map each
        leaf to its canonical position so that ``contribution(t_1,
        t_2, t_3)`` always has position i = time of
        ``external_fields[i]``, regardless of the diagram's internal
        leaf ordering.  If None, falls back to position-based mapping
        (leaf j → ext_time_vars[j]).
    prop_td : PropagatorTD or None
        Model-level data shared by the diagrams of one
        ``compute_correction_td`` call (M5 L2).  ``None`` (the default):
        everything is built here, per diagram, as before.  Used only when
        ``USE_SETUP_PROP_TD`` is on and it matches ``propagator_data`` /
        ``num_params``.

    Returns
    -------
    dict with keys:
        'status' : 'ok' | 'empty_polytope' | 'failed'
        'contribution' : callable
            `f(*ext_time_values) -> complex`. For status != 'ok' this
            may be `None`.
        'integration_vars' : list of SR
            The non-leaf vertex time symbols that were integrated out.
        'stripped_integrand' : SR
            The symbolic integrand WITHOUT the Heaviside factors, for
            debugging.
        'constraints' : list of SR
            One SR expression per edge; each expression `dt_e` is the
            linear combination `t_head - t_tail` that must be positive
            for the integrand to be nonzero.
    """
    loop_number = _loop_number_from_graph(typed_diagram)
    # Identity of this build for the ``_SUBSET_HOOK`` payload (M0.1).
    _diag_serial = (_next_diagram_serial() if _setup_full is None
                    else _setup_full)
    # The model, for the hardened fallback's one-time warnings (M2b).
    _model_id = _model_identity(propagator_data)

    D = typed_diagram.prediagram[0]
    leaves = list(typed_diagram.prediagram[2])
    leaf_set = set(leaves)

    if len(ext_time_vars) != len(leaves):
        raise ValueError(
            f"ext_time_vars has length {len(ext_time_vars)} but "
            f"the diagram has {len(leaves)} leaves."
        )

    # ── 0. Zero-prefactor early exit (M5 L1, ``USE_SETUP_ZERO_EXIT``) ──
    # A diagram whose prefactor is numerically exactly 0 contributes 0 at
    # every point: every subset is skipped at the "numerical zero-skip"
    # below.  Return that result before ``build_G_t_matrix`` and the
    # per-edge setup (most of this function's time on a model with many
    # structurally-zero diagrams).  ``_setup_full`` is set by the lazy keys
    # of the early result: they re-run the full path to build themselves.
    _cp_pre = None
    if (_setup_full is None and _SUBSET_HOOK is None
            and _setup_lever_on('USE_SETUP_ZERO_EXIT')):
        _cp_pre = SR(combined_prefactor) if combined_prefactor is not None \
            else SR(1)
        if num_params:
            _cp_pre = _cp_pre.subs(num_params)
        if _zero_exit_applies(typed_diagram, propagator_data, _cp_pre,
                              external_fields, leaves):
            return _zero_exit_result(
                D, leaves, ext_time_vars, _diag_serial,
                lambda serial: integrate_diagram(
                    typed_diagram, propagator_data, combined_prefactor,
                    ext_time_vars, num_params=num_params,
                    origin_leaf_idx=origin_leaf_idx,
                    external_fields=external_fields,
                    representative_ir=representative_ir,
                    edge_mode_sums_builder=edge_mode_sums_builder,
                    prop_td=prop_td,
                    _setup_full=serial))

    # ── 1. Numerical G(t) matrix (smooth + delta parts) ──────────
    # M5 L2: the matrix, the poles and the per-entry residues come from the
    # call-scoped ``PropagatorTD`` when one that matches this propagator data
    # was handed in.
    _ptd = None
    if prop_td is not None and _setup_lever_on('USE_SETUP_PROP_TD'):
        if prop_td.matches(propagator_data, num_params):
            _ptd = prop_td
            _RUNTIME_COUNTERS['setup_prop_td_used'] += 1
        else:
            _RUNTIME_COUNTERS['setup_prop_td_stale'] += 1
    t_sym = SR.var('_t_td_')
    if _ptd is not None:
        G_t_obj = _ptd.G_t_obj()
    else:
        G_t_obj = build_G_t_matrix(propagator_data, t_sym,
                                   num_params=num_params)

    # ── 2. Enumerate inter-vertex Wick contractions ───────────────
    # For correlators with repeated external field types (e.g. two
    # dn₁ legs at different spacetime points), each DISTINCT way to
    # assign canonical positions to leaves is a separate Wick
    # contraction that contributes to the connected correlator.
    #
    # The dedup in symmetry.py merges diagrams that differ only in
    # which same-type leaf connects to which vertex (they have the
    # same field-type multiset per vertex).  We compensate here by
    # summing over all such permutations.
    #
    # For same-vertex permutations (e.g. 2 dn₁ both at one vertex),
    # the integrand is invariant under swap (commutative product),
    # so summing 2! mappings gives 2x overcounting → divide by 2!.
    # For cross-vertex permutations (dn₁ at different vertices), the
    # integrands differ → summing gives the full answer.
    import itertools as _itertools

    if external_fields is not None and len(external_fields) == len(leaves):
        _leaf_fields = [typed_diagram.external_legs.get(lf) for lf in leaves]
        # Group canonical positions and leaves by field type
        _cp_by_field = {}
        _leaves_by_field = {}
        for cp, field in enumerate(external_fields):
            _cp_by_field.setdefault(field, []).append(cp)
        for j, field in enumerate(_leaf_fields):
            _leaves_by_field.setdefault(field, []).append(j)

        # Enumerate all canonical-to-leaf mappings
        _all_mappings = [{}]
        _mapping_fallback = False
        for field in sorted(_cp_by_field.keys(), key=str):
            cps = _cp_by_field[field]
            lfs = _leaves_by_field.get(field, [])
            if len(cps) != len(lfs):
                _all_mappings = [{j: j for j in range(len(leaves))}]
                _mapping_fallback = True
                break
            perms = list(_itertools.permutations(lfs))
            new_mappings = []
            for m in _all_mappings:
                for perm in perms:
                    nm = dict(m)
                    for cp, lf_idx in zip(cps, perm):
                        nm[cp] = lf_idx
                    new_mappings.append(nm)
            _all_mappings = new_mappings

        # Compensation factor: divides the ``_all_mappings`` sum to
        # remove ext-leaf permutations that are ALREADY graph
        # automorphisms of the typed diagram.
        #
        # By orbit–stabilizer, the mapping sum counts every distinct
        # pinned-external diagram exactly
        # ``|Aut(Γ, leaves free)| / |Aut(Γ, leaves fixed)|`` times, so
        # that index is the exact divisor — see
        # ``external_wick_compensation`` in symmetry.py.
        #
        # (A previous heuristic divided by ∏N! over leaves grouped by
        # ``vertex_role_signature`` of their attachment vertex; same
        # role signature does NOT imply the leaf swap is realizable as
        # an automorphism, so the heuristic over-divided — ×⅓/×½
        # deficits on the k=4 OU+εx³ 1-loop cascades/double-exchange
        # while k=2 stayed exact by coincidence.)
        if _mapping_fallback:
            # Field-count mismatch: only the identity mapping is
            # summed, so there is no permutation overcounting to
            # divide out.  (The old code applied the heuristic divisor
            # even here, silently shrinking the single-mapping sum.)
            _compensation = 1
        else:
            from engine.diagrams.symmetry import external_wick_compensation
            _compensation = external_wick_compensation(typed_diagram)
    else:
        # SAFETY WARNING: falling back to identity leaf→position
        # mapping.  For diagrams whose leaves have MIXED field types
        # (e.g. one δn_1 and one δv_2), the enumeration's leaf order
        # is not guaranteed to match ``external_fields`` — a mismatch
        # here produces a τ → −τ mirror image of the physical
        # correlator.  Always pass ``external_fields`` when k ≥ 2.
        _leaf_field_list = [typed_diagram.external_legs.get(lf)
                            for lf in leaves]
        if len(set(_leaf_field_list)) > 1:
            import warnings as _warnings
            _warnings.warn(
                "integrate_tree_diagram: external_fields not provided "
                "for a diagram with mixed leaf field types "
                f"({_leaf_field_list}).  The canonical leaf→position "
                "mapping will fall back to identity, which may produce "
                "a τ → −τ mirror image of the physical correlator.  "
                "Pass external_fields to fix.",
                stacklevel=2,
            )
        _all_mappings = [{j: j for j in range(len(leaves))}]
        _compensation = 1

    # Build vertex_time for the FIRST mapping; subsequent mappings
    # handled by permuting the positional arguments in the wrapper.
    vertex_time = {}
    _first_mapping = _all_mappings[0]
    for cp, leaf_idx in _first_mapping.items():
        lf = leaves[leaf_idx]
        t_ext = ext_time_vars[cp]
        if origin_leaf_idx is not None and cp == origin_leaf_idx:
            t_ext = SR(0)
        vertex_time[lf] = t_ext

    internal_vertices = [v for v in D.vertices() if v not in leaf_set]
    integration_vars = []
    for v in internal_vertices:
        s_v = SR.var(f's_v{v}_td_', latex_name=rf's_{{v_{{{v}}}}}')
        vertex_time[v] = s_v
        integration_vars.append(s_v)

    # ── 2b. Non-local cumulant noise sources: per-leg time map ────
    # For each NoiseSourceType vertex, the response legs sit at
    # *independent* times coupled by a non-local kernel κ^{(n)}(τ).
    # We anchor the vertex at its existing time symbol (= leg-0
    # time) and introduce one extra integration variable τ per
    # noise source, with leg-1 time = anchor − τ.  Edges leaving
    # this vertex through leg 1 (matched by their resp_leg pop_idx)
    # are routed through the leg-1 time symbol.  Plain SourceType
    # vertices (cortical Poisson, GTaS auto-cumulant) keep the
    # single-time semantics — vertex_leg_time stays empty for them
    # and the existing edge-build code path is unaffected.
    #
    # ConvVertexType vertices (conductance-style interaction
    # vertices, e.g. ``vt·v·Conv(g,n)``) extend the same scaffold
    # to one PHYSICAL leg per kernel attachment.  The shape of
    # ``vertex_leg_time[v]`` is identical (``{pop_idx_0based: SR time}``)
    # but ``vertex_leg_kind[v]`` tells the edge-routing code below
    # whether to match against ``edge_resp_leg`` (NoiseSourceType)
    # or ``edge_phys_leg`` (ConvVertexType).
    vertex_leg_time = {}        # {v: {leg_pop_idx_0based: SR time}}
    vertex_leg_kind = {}        # {v: 'response' | 'physical'}
    noise_source_specs = {}     # {v: list of cumulant_specs dicts}
    conv_vertex_specs  = {}     # {v: list of (tau_sym, att_dict) pairs}
    extra_tau_syms = []         # list of (tau_sym, vertex) pairs
    vertex_assignments = (
        getattr(typed_diagram, 'vertex_assignments', None) or {}
    )
    for v, vtype in vertex_assignments.items():
        if not isinstance(vtype, NoiseSourceType):
            continue
        if not vtype.cumulant_specs:
            continue
        # All specs on this vertex share the same leg multiset
        # (they were grouped by extract_source_types).  Read the
        # 0-based leg ordering and field-name tuple from the first
        # spec (extract_source_types caches leg_fields on each entry).
        legs0 = vtype.cumulant_specs[0]['legs']            # e.g. (0, 0)
        leg_fields_0 = vtype.cumulant_specs[0].get('leg_fields')
        # Anchor time = the existing internal-vertex symbol
        anchor_time = vertex_time[v]
        # τ symbol — distinct per noise vertex
        tau_sym = SR.var(
            f's_v{v}_tau_td_', latex_name=rf'\tau_{{v_{{{v}}}}}'
        )
        extra_tau_syms.append((tau_sym, v))
        integration_vars.append(tau_sym)

        # Edge-keyed per-leg time map.  The typed diagram identifies
        # legs by (field_base, pop_idx) — that's a coarser granularity
        # than the cumulant action needs.  For a homogeneous auto-
        # cumulant ``Cxx`` (legs ``['xt','xt']``) BOTH source-side
        # endpoints have the same ``(field, pop_idx)``, so a pop_idx
        # keyed map (the old design) collapsed both legs to the same
        # time symbol and dropped the τ-coupling between propagators —
        # the kernel ``K(τ_v)`` then became a multiplicative weight
        # decoupled from the propagator structure, yielding the
        # *white-limit* answer ``D/μ`` for every τc.
        #
        # Fix: route by EDGE-KEY directly.  The typed diagram's
        # parallel multi-edges have distinct labels, so two edges to
        # the same source can always be disambiguated by their label.
        # Strategy:
        #   * heterogeneous-field legs (e.g. ``['xt','yt']``): match
        #     each incident edge's response-leg field+pop_idx to the
        #     source's leg-field tuple.  Edge whose resp_leg matches
        #     ``leg_fields_0[0]`` lands at ``anchor_time``; the other
        #     lands at ``anchor_time - τ``.
        #   * homogeneous-field legs (e.g. ``['xt','xt']``): no field
        #     info distinguishes them — use label ordering instead.
        #     𝒮(Γ) already counts both orderings of indistinguishable
        #     legs, so the specific choice of "first label → anchor"
        #     just picks ONE representative of the orbit.
        incident_edges = [
            ek for ek in D.edges() if ek[0] == v or ek[1] == v
        ]
        edge_to_time = {}
        homogeneous = (
            leg_fields_0 is None
            or len(legs0) < 2
            or (len(legs0) == 2
                and leg_fields_0[0] == leg_fields_0[1]
                and legs0[0] == legs0[1])
        )
        if not homogeneous and len(legs0) == 2:
            # Heterogeneous: match each edge by its response leg
            # field+pop_idx against ``leg_fields_0``.
            target_anchor = (leg_fields_0[0], legs0[0] + 1)
            target_other  = (leg_fields_0[1], legs0[1] + 1)
            for ek in incident_edges:
                edge_resp_leg, _ = typed_diagram.edge_types[ek]
                if edge_resp_leg == target_anchor:
                    edge_to_time[ek] = anchor_time
                elif edge_resp_leg == target_other:
                    edge_to_time[ek] = anchor_time - tau_sym
                else:
                    # Shouldn't happen if extract_source_types matched
                    # legs correctly, but fall back to anchor.
                    edge_to_time[ek] = anchor_time
        else:
            # Homogeneous (or order≠2): use label ordering to
            # disambiguate parallel edges.  𝒮(Γ) absorbs the choice.
            sorted_edges = sorted(incident_edges)
            for i, ek in enumerate(sorted_edges):
                edge_to_time[ek] = (
                    anchor_time if (i == 0) else (anchor_time - tau_sym)
                )
        vertex_leg_time[v] = edge_to_time
        vertex_leg_kind[v] = 'response'
        noise_source_specs[v] = list(vtype.cumulant_specs)

    # ── 2c. Conductance-style interaction vertices ────────────────
    # ConvVertexType is the interaction-vertex analogue of
    # NoiseSourceType: one PHYSICAL leg per kernel attachment sits
    # at ``anchor_time − τ`` linked to the rest of the vertex via
    # the synaptic kernel ``g(τ)``.  See
    # ``docs/conductance_vertex_kernels_design.md`` for the math.
    #
    # A single ConvVertexType can carry several attachments (e.g.
    # ``vt · v · Conv(g1, n1) · Conv(g2, n2)``) — one τ per
    # attachment.  Each kernel-attached leg's pop-idx is recorded
    # under leg_index in the attachment dict so vertex_leg_time
    # can route edges through the right time symbol.
    for v, vtype in vertex_assignments.items():
        if not isinstance(vtype, ConvVertexType):
            continue
        if not vtype.kernel_attachments:
            continue
        if v in vertex_leg_time:
            # Vertex was already promoted by the NoiseSourceType
            # block — bigrade overlap is structurally impossible
            # (NoiseSourceType is n_phys=0, ConvVertexType is
            # n_phys≥1) but guard against the misclassification
            # rather than silently merging two leg maps.
            raise RuntimeError(
                f"vertex {v} appears both as NoiseSourceType and "
                f"ConvVertexType — these are mutually exclusive."
            )
        anchor_time = vertex_time[v]
        leg_time_map = {}
        # Track this vertex's τ↔attachment pairs locally — DON'T mutate
        # the attachment dict.  The same ConvVertexType instance can be
        # bound to multiple graph vertices in the typing engine, so any
        # mutation here would leak τ symbols across vertices in the
        # same diagram (vertex 2's τ overwriting vertex 4's, etc.).
        att_tau_pairs = []
        for att_idx, att in enumerate(vtype.kernel_attachments):
            # ``leg_index`` is the position within physical_legs
            # the kernel attaches to.  We key vertex_leg_time on
            # the leg's pop_idx (matching the edge-routing
            # convention used for NoiseSourceType).
            leg_tuple = att['leg']
            phys_pop_idx = leg_tuple[1] - 1  # 0-based
            tau_sym = SR.var(
                f's_v{v}_gtau{att_idx}_td_',
                latex_name=rf'\tau^{{g}}_{{v_{{{v}}},{att_idx}}}'
            )
            extra_tau_syms.append((tau_sym, v))
            integration_vars.append(tau_sym)
            # Kernel-attached leg sits at anchor − τ.
            leg_time_map[phys_pop_idx] = anchor_time - tau_sym
            att_tau_pairs.append((tau_sym, att))
        vertex_leg_time[v] = leg_time_map
        vertex_leg_kind[v] = 'physical'
        conv_vertex_specs[v] = att_tau_pairs

    # ── 3. Gather per-edge info: ri, pi, dt, delta_coeff, smooth factor
    def _resolve_leg_time(vert, edge_key, default_time):
        """Pick the right time symbol for ``vert``'s end of an edge.

        For NoiseSourceType (``vertex_leg_kind == 'response'``) the
        per-leg map is keyed on the ``edge_key`` directly — the
        typed-diagram representation collapses indistinguishable legs
        (e.g. the two ``xt`` legs of an auto-cumulant ``Cxx``) under a
        single (field, pop_idx) identifier, so we must use the edge's
        own label to disambiguate two parallel edges into the source.
        See the noise-vertex setup block above for the heterogeneous /
        homogeneous routing strategy.

        For ConvVertexType (``vertex_leg_kind == 'physical'``) the map
        is keyed on the PHYSICAL leg pop-idx — the kernel attachment
        already binds a specific physical-leg slot, no ambiguity.

        Plain vertices keep their single ``vertex_time`` entry — the
        routing falls through to ``default_time``.
        """
        if vert not in vertex_leg_time:
            return default_time
        kind = vertex_leg_kind.get(vert)
        if kind == 'response':
            return vertex_leg_time[vert].get(edge_key, default_time)
        # ConvVertexType ('physical') path — keyed by phys-leg pop_idx
        edge_resp_leg, edge_phys_leg = typed_diagram.edge_types[edge_key]
        leg = edge_phys_leg
        pop_idx = leg[1] - 1  # 0-based
        return vertex_leg_time[vert].get(pop_idx, default_time)

    edges = list(D.edges())
    edge_info = []
    for (u, v, lbl) in edges:
        ri, pi = _lookup_prop_indices(typed_diagram, (u, v, lbl))
        edge_key = (u, v, lbl)
        # Route edge tail / head through per-leg time when the
        # vertex carries a non-local kernel (noise source or
        # conductance interaction); otherwise use the standard
        # single-time map.
        t_u = _resolve_leg_time(u, edge_key, vertex_time[u])
        t_v = _resolve_leg_time(v, edge_key, vertex_time[v])
        dt = SR(t_v - t_u)
        delta_c = (_ptd.delta_coeff(pi, ri) if _ptd is not None
                   else G_t_delta_coeff(G_t_obj, pi, ri))
        smooth_factor = G_t_entry(G_t_obj, pi, ri, dt, include_heaviside=False)
        edge_info.append({
            'u': u, 'v': v, 'lbl': lbl,
            'ri': ri, 'pi': pi,
            'dt_sym': dt,
            'delta_coeff': delta_c,
            'smooth_factor': smooth_factor,
        })

    # ── 3a. Mode-sum cache (Stage 2 of Phase J refactor) ─────────
    # Build one ``EdgeModeSum`` per edge, extracting the per-pole
    # residue from ``propagator_data['C_mats']`` ONCE.  The fast
    # subset evaluator (the hot path for plain diagrams) reuses
    # this across all 2^|branch| subsets instead of re-extracting
    # residues from SR on every call.  ``None`` if the propagator
    # data is incomplete, in which case the fast path stays on
    # the legacy tuple-based extractor.
    # Spatial loop integrator hook (Stage C.5): when a builder is supplied,
    # it constructs the per-edge EdgeModeSums from the routed per-edge momenta
    # (each edge ``ei`` carries its ``(u, v, lbl)`` key).  Defaults to the
    # standard global-propagator extractor, so the time-only path is unchanged.
    if edge_mode_sums_builder is not None:
        edge_mode_sums = edge_mode_sums_builder(edge_info, propagator_data)
    else:
        edge_mode_sums = _build_edge_mode_sums(edge_info, propagator_data,
                                               prop_td=_ptd)

    # Combined prefactor (numerical)
    if _cp_pre is not None:
        cp = _cp_pre            # the very expression the early-exit test built
    else:
        cp = SR(combined_prefactor) if combined_prefactor is not None \
            else SR(1)
        if num_params:
            cp = cp.subs(num_params)

    # ── 3b. Non-local cumulant kernel substitution ────────────────
    # For each NoiseSourceType vertex, replace each placeholder
    # symbol ``z_kappa_<noise>_<order>_<i>_<j>`` in ``cp`` with the
    # actual kernel SR expression returned by the user's kernel_fn,
    # evaluated at the per-vertex τ integration symbol.  The signs
    # and combinatorial factors that ``_build_cumulant_action``
    # multiplied onto each placeholder (typically -1/2) are already
    # in ``cp``; the substitution carries them through.  The result
    # is a cp that is now an explicit function of the τ symbols,
    # which the existing fast_callable / nquad path handles
    # naturally because each τ is in ``integration_vars``.
    if noise_source_specs:
        kappa_subs = {}
        for v, specs in noise_source_specs.items():
            tau_sym_v = next(
                (ts for ts, vv in extra_tau_syms if vv == v), None
            )
            if tau_sym_v is None:
                continue
            for spec in specs:
                i_leg, j_leg = spec['legs'][0], spec['legs'][1]
                kappa_subs[spec['symbol']] = SR(
                    spec['kernel_fn'](i_leg, j_leg, tau_sym_v)
                )
        if kappa_subs:
            cp = cp.subs(kappa_subs)
            if num_params:
                # Re-substitute num_params now that kernel symbols
                # like ns.lambda_X, ns.mu_shift_diff have been
                # introduced via kernel_fn evaluation.
                cp = cp.subs(num_params)
            # Try to combine like terms (e.g. ordered pairs (i,j) and
            # (j,i) with symmetric kernels collapse to a single term).
            # simplify_full can be expensive but the cumulant prefactor
            # is a small SR expression so it's cheap here.
            try:
                cp = cp.simplify_full()
            except (ValueError, RuntimeError, AttributeError):
                pass

    # ── 3c. Conductance-vertex kernel substitution ────────────────
    # For each ConvVertexType, replace the kernel SR symbol in ``cp``
    # with the time-domain kernel ``g(τ)`` evaluated at the
    # per-attachment τ symbol introduced in section 2c.  Same shape
    # as the noise-source path above — once substituted, ``cp`` is
    # an explicit function of the τ symbols and flows through
    # fast_callable / nquad without further special handling.
    # ``conv_kernel_extracted`` flags whether all ConvVertex kernels were
    # successfully decomposed into single-exponential pseudo-edges.  When
    # True, ``cp`` has the kernel symbols replaced by ``1`` and the kernel
    # weights live in the per-subset mode-sum (analytic path).  When False
    # (e.g. polynomial-prefactor alpha kernel that single-exp extraction
    # rejects), we fall back to the legacy ``cp.subs(g(τ))`` substitution
    # and the slower SR + scipy.nquad path.
    conv_kernel_extracted = False
    # ``conv_extracted_modes`` collects ``(tau_sym, C, lam)`` triples for
    # the per-subset pseudo-edge build.  Populated only when extraction
    # succeeds; empty otherwise.
    conv_extracted_modes = []
    if conv_vertex_specs:
        from sage.all import heaviside as _sage_heaviside
        _all_extractable = True
        _modes_buffer = []
        for v, att_tau_pairs in conv_vertex_specs.items():
            for tau_sym, att in att_tau_pairs:
                td_fn = att.get('kernel_td_fn')
                if td_fn is None:
                    _all_extractable = False
                    break
                kernel_sr = SR(td_fn(tau_sym))
                kernel_sr = kernel_sr.substitute_function(
                    _sage_heaviside, lambda _x: SR(1)
                )
                if num_params:
                    kernel_sr = kernel_sr.subs(num_params)
                mode = _extract_exp_mode(kernel_sr, tau_sym)
                if mode is None:
                    _all_extractable = False
                    break
                C, lam = mode
                _modes_buffer.append((tau_sym, C, lam))
            if not _all_extractable:
                break
        conv_kernel_extracted = _all_extractable
        if _all_extractable:
            conv_extracted_modes = _modes_buffer

        # Build the kernel-symbol → SR substitution dict.  When the
        # kernel will live in a pseudo-edge, substitute with ``1`` so
        # the analytic mode-sum doesn't double-count it.  Otherwise
        # substitute with the full ``g(τ)`` SR expression so the SR +
        # scipy.nquad path sees a complete integrand.
        g_subs = {}
        for v, att_tau_pairs in conv_vertex_specs.items():
            for tau_sym, att in att_tau_pairs:
                td_fn = att.get('kernel_td_fn')
                if td_fn is None:
                    continue
                if conv_kernel_extracted:
                    g_subs[att['symbol']] = SR(1)
                else:
                    td_expr = SR(td_fn(tau_sym))
                    td_expr = td_expr.substitute_function(
                        _sage_heaviside, lambda _x: SR(1)
                    )
                    g_subs[att['symbol']] = td_expr
        if g_subs:
            cp = cp.subs(g_subs)
            if num_params:
                cp = cp.subs(num_params)
            try:
                cp = cp.simplify_full()
            except (ValueError, RuntimeError, AttributeError):
                pass

    # ── 4. External-time bookkeeping ─────────────────────────────
    free_ext_idx = [
        j for j in range(len(ext_time_vars))
        if (origin_leaf_idx is None or j != origin_leaf_idx)
    ]
    free_ext_syms = [ext_time_vars[j] for j in free_ext_idx]

    n_edges = len(edge_info)

    # Pre-classify edges by whether they CAN be chosen as δ.  An edge
    # with ``|delta_coeff| < 1e-15`` contributes nothing if placed in
    # the δ subset, so the inner subset loop only branches on edges
    # that have a nonzero δ part.  Edges with zero δ are always in
    # ``smooth_edges``.  Pre-classification turns a 2^|E| enumeration
    # into 2^|branch| with no behaviour change relative to the old
    # ``continue`` guard inside the loop.
    branch_edge_indices: list[int] = []
    forced_smooth_indices: list[int] = []
    for i in range(n_edges):
        if abs(complex(edge_info[i]['delta_coeff'])) < 1e-15:
            forced_smooth_indices.append(i)
        else:
            branch_edge_indices.append(i)
    n_branch = len(branch_edge_indices)
    n_subsets_total = 2 ** n_branch

    # M2a structural zeros (``STRUCTURAL_ZEROS``, read when the subsets are
    # built): the mirror image of the forced-smooth edges.  An edge whose
    # smooth part is provably zero (every stored residue of its entry is an
    # exact zero and its ``G_ft`` entry contains no ω; for a propagator
    # without any pole: the ``G_ft`` test alone) contributes nothing when
    # kept smooth, so a δ-subset
    # that keeps one smooth is never built (it is skipped before the
    # δ-elimination; counter ``forced_delta_pruned``).  A pole-free
    # propagator whose entry is not provably a pure δ raises
    # ``PoleFreePropagatorError`` here.  Only for the time-only propagator
    # (no ``edge_mode_sums_builder``), where every evaluator reads the same
    # ``propagator_data['C_mats']``.  A pruned subset that would have been
    # a shot-noise subset contributed a zero-coefficient entry to
    # ``delta_contributions``; it is not emitted.
    _forced_delta = (
        _forced_delta_edges(edge_info, propagator_data,
                            num_params=num_params)
        if edge_mode_sums_builder is None and _structural_zeros_on()
        else frozenset())

    # Expose the |S|=0 (all smooth) symbolic integrand and constraints
    # for debugging / display, matching the pre-fix return shape.  Kept as
    # the unexpanded product: nothing reads it numerically, and expanding it
    # blows up combinatorially for multi-mode kernels (alpha-function
    # synapses: >10 min per diagram on quadratic_hawkes_alpha at ell=1).
    display_stripped = cp
    for ei in edge_info:
        display_stripped = display_stripped * ei['smooth_factor']
    display_constraints = [ei['dt_sym'] for ei in edge_info]

    # Accumulators
    subset_contributions = []   # continuous smooth contributions (callable)
    delta_contributions = []    # shot-noise δ spikes (structured dicts)
    n_shotnoise_skipped = 0
    subset_diagnostics = []

    for branch_bits in range(n_subsets_total):
        # Δ subset: branch-edges with bit set, in original edge-index order.
        delta_edges = [
            branch_edge_indices[k] for k in range(n_branch)
            if (branch_bits >> k) & 1
        ]
        # Smooth subset: forced-smooth edges + branch-edges with bit unset.
        # Preserve original edge-index order so downstream constraint
        # extraction and zip(edge_info, ...) sees the same ordering as
        # the pre-Stage-1a code path.
        branch_smooth = [
            branch_edge_indices[k] for k in range(n_branch)
            if not ((branch_bits >> k) & 1)
        ]
        smooth_edges = sorted(forced_smooth_indices + branch_smooth)

        if _forced_delta and not _forced_delta.isdisjoint(smooth_edges):
            _RUNTIME_COUNTERS['forced_delta_pruned'] += 1
            subset_diagnostics.append({
                'delta_edges': delta_edges,
                'smooth_edges': smooth_edges,
                'status': 'forced_delta_pruned',
            })
            continue

        # ── Solve the δ-edge equalities: eliminate integration vars
        # by substitution. For each δ edge, set dt_e = 0 and solve for
        # an integration variable appearing in the equation; if no
        # integration variable is available, the equation becomes a
        # constraint among external times (shot-noise δ, skip).
        substitutions = {}
        remaining_int_vars = list(integration_vars)
        ext_time_equalities = []  # residual constraints on ext times

        subset_infeasible = False
        for ei_idx in delta_edges:
            eq_expr = edge_info[ei_idx]['dt_sym'].subs(substitutions)
            eq_expr = SR(eq_expr)
            # Find an integration variable to solve for
            int_var_to_eliminate = None
            try:
                eq_vars = set(eq_expr.variables())
            except AttributeError:
                eq_vars = set()
            for iv in remaining_int_vars:
                if iv in eq_vars:
                    int_var_to_eliminate = iv
                    break
            if int_var_to_eliminate is not None:
                try:
                    sol = sage_solve(
                        eq_expr == 0, int_var_to_eliminate,
                        solution_dict=True,
                    )
                except Exception:
                    sol = []
                if not sol:
                    subset_infeasible = True
                    break
                new_rhs = sol[0][int_var_to_eliminate]
                substitutions[int_var_to_eliminate] = new_rhs
                remaining_int_vars.remove(int_var_to_eliminate)
                # Resolve transitively: apply the new substitution to
                # the RHS of every existing entry so a chain like
                # ``{a: f(b), b: g(c)}`` collapses to ``{a: f(g(c)), b: g(c)}``.
                # Sage's ``.subs(dict)`` is a parallel one-pass operation
                # and does NOT chain substitutions — without this fixup
                # ``cp.subs(substitutions)`` would leave ``b`` exposed in
                # the result, breaking the integrator's free-symbol
                # audit downstream.  Pre-existing concern that becomes
                # load-bearing with multi-τ ConvVertexType diagrams.
                # Cheap early-skip: only chain-resolve if some EXISTING
                # RHS actually mentions the variable we just eliminated.
                # For typical non-ConvVertex diagrams the chain almost
                # never forms (the per-edge ``eq_expr.subs(substitutions)``
                # above already applies prior subs before solving), so
                # the inner SR.subs() loop is pure overhead.  Walk
                # variables once per existing entry — much cheaper than
                # blindly calling SR.subs.
                affected_keys = []
                for _k, _rhs in substitutions.items():
                    if _k == int_var_to_eliminate:
                        continue
                    try:
                        _rhs_vars = SR(_rhs).variables()
                    except (AttributeError, TypeError):
                        continue
                    if int_var_to_eliminate in _rhs_vars:
                        affected_keys.append(_k)
                if affected_keys:
                    _chain_subs = {int_var_to_eliminate: new_rhs}
                    for _k in affected_keys:
                        substitutions[_k] = SR(
                            substitutions[_k]
                        ).subs(_chain_subs)
            else:
                # No integration variable to eliminate → this is a
                # constraint on external times alone. If it's
                # identically zero, the δ is satisfied trivially; if
                # not, it's a shot-noise δ(τ)-style contribution.
                ext_time_equalities.append(eq_expr)

        if subset_infeasible:
            continue

        # Shot-noise check: any nontrivial residual equality among
        # external times means this subset contributes a δ(τ) spike
        # at the hypersurface where that equality holds. Instead of
        # skipping it, compute a structured δ-contribution: a numeric
        # coefficient together with the linear equality and any
        # retardation half-space constraints, so downstream code can
        # insert it into a discrete τ grid.
        has_shotnoise = False
        nontrivial_equalities = []
        for eq in ext_time_equalities:
            try:
                if bool(eq.is_zero()):
                    continue
            except Exception:
                pass
            # Nontrivial equation → shot-noise
            has_shotnoise = True
            nontrivial_equalities.append(SR(eq))
        if has_shotnoise:
            n_shotnoise_skipped += 1
            subset_diagnostics.append({
                'delta_edges': delta_edges,
                'smooth_edges': smooth_edges,
                'status': 'shotnoise',
                'ext_time_equalities': ext_time_equalities,
            })

            # For the MVP we only support the single-equality case
            # (one δ(a · τ + c) spike per subset). Multi-equality cases
            # would correspond to δ(τ_a − τ_b) · δ(τ_c − τ_d) style
            # "double-delta" spikes which are rare at tree level and
            # deferred to Extension 1.
            if len(nontrivial_equalities) != 1:
                continue

            # Build the numeric coefficient: combined_pf × ∏ δ-coeffs
            #                               × (smooth factors with δ subs applied)
            subset_factor_delta = cp
            for ei_idx in delta_edges:
                subset_factor_delta = (
                    subset_factor_delta
                    * SR(edge_info[ei_idx]['delta_coeff'])
                )
            for ei_idx in smooth_edges:
                subset_factor_delta = (
                    subset_factor_delta
                    * edge_info[ei_idx]['smooth_factor']
                )
            subset_factor_delta = subset_factor_delta.subs(substitutions)
            # NOTE: ``.subs(num_params)`` used to run here but it is
            # redundant: ``build_G_t_matrix(propagator_data, t_sym,
            # num_params=num_params)`` has already substituted
            # ``num_params`` into every ``edge_info[i]['smooth_factor']``
            # and ``edge_info[i]['delta_coeff']``, and ``cp`` was
            # substituted at the top of ``integrate_diagram``.  The only
            # remaining free variables here are ext-time symbols and
            # integration variables -- neither of which is in
            # ``num_params``.  Removed 2026-04-21 (audit Fix #A, ~6%
            # speedup on k=2 ell=1 quadratic Hawkes).
            try:
                subset_factor_delta = subset_factor_delta.expand()
            except Exception:
                pass

            # At the shot-noise hypersurface, any remaining integration
            # variables must already have been eliminated (otherwise it
            # wouldn't be a pure δ contribution — m_sub > 0 after all
            # deltas applied means we'd need to integrate further).
            # For the MVP star tree, this condition always holds.
            if remaining_int_vars:
                subset_diagnostics.append({
                    'status': 'shotnoise_with_remaining_int_vars',
                    'delta_edges': delta_edges,
                    'remaining': list(remaining_int_vars),
                })
                continue

            # Extract the linear form of the equality in terms of
            # free_ext_syms: a · x + c = 0.
            eq = nontrivial_equalities[0]
            try:
                eq_a = [float(eq.coefficient(s)) for s in free_ext_syms]
                eq_c = float(eq.subs({s: 0 for s in free_ext_syms}))
            except (TypeError, ValueError):
                continue

            # Check the equality is actually nontrivial (not all zero)
            if all(abs(a) < 1e-15 for a in eq_a):
                # Pure numeric residual: if nonzero, no contribution;
                # if zero, it's trivially satisfied (shouldn't happen
                # because we already filtered `is_zero()` above).
                continue

            # The symbolic factor evaluated at the δ surface is just
            # subset_factor_delta — it already has all the pin-substitutions
            # applied and can be evaluated numerically once free_ext_vals
            # are supplied. For the MVP case where remaining_int_vars is
            # empty, subset_factor_delta depends only on free_ext_syms
            # (or is a constant).
            try:
                coeff_free_vars = set(subset_factor_delta.variables())
            except AttributeError:
                coeff_free_vars = set()
            unexpected_in_coeff = coeff_free_vars - set(free_ext_syms)
            if unexpected_in_coeff:
                # Can't build a callable coefficient with free params left
                continue

            try:
                coeff_fc = fast_callable(
                    subset_factor_delta,
                    vars=list(free_ext_syms),
                    domain=CDF,
                )
            except Exception:
                continue

            # Retardation constraints at the δ point: for the MVP's
            # shot-noise subset (all edges δ, no smooth), there are
            # no retardation constraints. But in principle a mixed
            # δ+smooth shot-noise subset could have them.
            retard_data_delta = []
            for ei_idx in smooth_edges:
                c_retard = SR(
                    edge_info[ei_idx]['dt_sym']
                ).subs(substitutions)
                try:
                    a_ext = [
                        float(c_retard.coefficient(s))
                        for s in free_ext_syms
                    ]
                    c0 = float(c_retard.subs(
                        {s: 0 for s in free_ext_syms}
                    ))
                except (TypeError, ValueError):
                    continue
                retard_data_delta.append((a_ext, c0))

            delta_contributions.append({
                'coeff_fc': coeff_fc,
                'equality_a': eq_a,
                'equality_c': eq_c,
                'equality_symbolic': eq,
                'retardation_data': retard_data_delta,
                'delta_edges': list(delta_edges),
                'free_ext_idx': list(free_ext_idx),
            })
            continue

        # ── Stage 4a optim (2026-05-15): early prefactor build +
        # analytic-eligibility check.  When the subset can be served
        # by `_fast_eval` (pole/residue closure) + analytic modesum
        # integrators, the SR-based `subset_factor` build + `.expand()`
        # + `fast_callable()` compile is dead weight (Stage-3b
        # profiling: ~52% of integrate_diagram wall time on the k=2
        # ell=1 quad config, JIT tree never queried).  Skip it when
        # eligible; fall back to the full SR build only when needed.
        prefactor_num = cp
        for _ei_idx in delta_edges:
            prefactor_num = prefactor_num * SR(
                edge_info[_ei_idx]['delta_coeff']
            )

        try:
            _prefactor_c = complex(CDF(SR(prefactor_num)))
            _prefactor_is_numerical = True
        except Exception:
            _prefactor_c = None
            _prefactor_is_numerical = False

        # Numerical zero-skip (structural; no simplify_full).
        if _prefactor_is_numerical and _prefactor_c == 0:
            continue

        # Build retardation constraints for smooth edges (with δ subs applied).
        # Always needed for the polytope path.
        subset_retard = []
        for ei_idx in smooth_edges:
            c = SR(edge_info[ei_idx]['dt_sym']).subs(substitutions)
            subset_retard.append(c)

        # ── M1 Θ(0): τ-independent EMPTY subsets (plan §3.1 L0).  A smooth
        # edge whose Δt is a numeric constant after δ-elimination (no
        # integration variable, no external time: a_int ≡ 0, a_ext ≡ 0) with
        # c0 <= 0 empties this subset at EVERY τ under Θ(0) = 0.  The subset
        # is still built (its closure is what 'legacy_clip' integrates); the
        # flag makes the closure return 0 without integrating while the mode
        # read AT CALL TIME is 'ito' (so flipping the mode after the build
        # takes effect, as for every other Phase J flag).
        _theta0_always_empty = False
        for c in subset_retard:
            try:
                if SR(c).variables():
                    continue
                _c0_const = float(SR(c))
            except (AttributeError, TypeError, ValueError):
                continue
            if _const_row_decision((), (), _c0_const, (),
                                   'edge')[0] == 'EMPTY':
                _theta0_always_empty = True
                break

        m_sub = len(remaining_int_vars)
        fc_vars_sub = list(remaining_int_vars) + list(free_ext_syms)

        # `_analytic_eligible` ⇒ the per-call evaluator goes through
        # `_fast_eval` (built below from pole/residue cache) for any
        # residual scipy.nquad path, and the analytic modesum
        # integrators (m=1/2/≥3) handle the closed-form path.
        # Neither needs `integrand_fc_sub`, so we skip the entire
        # SR + `.expand()` + `fast_callable()` build chain.
        #
        # Conductance vertices (ConvVertexType) are eligible whenever
        # their kernels decompose into single-exponential pseudo-edges
        # (``conv_kernel_extracted``); the per-attachment ``(C, λ)``
        # mode is appended to ``smooth_edge_modes`` below, alongside
        # a polytope constraint ``τ > 0`` from the pseudo-edge's
        # ``dt = +τ`` linear form.  NoiseSourceType kernels remain on
        # the slow path until they get an analogous extraction.
        _conv_only_leg_times = (
            bool(conv_vertex_specs)
            and not noise_source_specs
            and conv_kernel_extracted
        )
        _analytic_eligible = (
            (not vertex_leg_time or _conv_only_leg_times)
            and edge_mode_sums is not None
            and _prefactor_is_numerical
        )

        if _analytic_eligible:
            subset_factor = None
            integrand_fc_sub = None
        else:
            # NoiseSourceType kernel diagrams, or non-numerical
            # prefactor, or missing edge_mode_sums cache — build the
            # full SR + fast_callable path for the residual scipy
            # integrand.  NOTE: ``.subs(num_params)`` removed
            # 2026-04-21 (audit Fix #A); see commit notes.
            subset_factor = cp
            for ei_idx in delta_edges:
                subset_factor = subset_factor * SR(
                    edge_info[ei_idx]['delta_coeff']
                )
            for ei_idx in smooth_edges:
                subset_factor = (
                    subset_factor * edge_info[ei_idx]['smooth_factor']
                )
            subset_factor = subset_factor.subs(substitutions)
            try:
                subset_factor = subset_factor.expand()
            except Exception:
                pass

            # Structural zero-check (avoids Maxima simplify_full,
            # which can hang or blow up for complex Hawkes integrands).
            try:
                if subset_factor.is_trivial_zero():
                    continue
            except AttributeError:
                if str(subset_factor) == '0':
                    continue

            # Free-symbol audit — catches num_params pass-through bugs.
            try:
                subset_free_vars = set(subset_factor.variables())
            except AttributeError:
                subset_free_vars = set()
            unexpected = subset_free_vars - set(fc_vars_sub)
            if unexpected:
                return {
                    'status': 'failed',
                    'contribution': None,
                    'integration_vars': integration_vars,
                    'stripped_integrand': display_stripped,
                    'constraints': display_constraints,
                    'reason': (
                        f"[subset {bin(branch_bits)}] stripped integrand "
                        f"contains unexpected free symbols {unexpected}; "
                        f"pass them via num_params."
                    ),
                }

            try:
                integrand_fc_sub = fast_callable(
                    subset_factor, vars=fc_vars_sub, domain=CDF,
                )
            except Exception as exc:
                return {
                    'status': 'failed',
                    'contribution': None,
                    'integration_vars': integration_vars,
                    'stripped_integrand': display_stripped,
                    'constraints': display_constraints,
                    'reason': (
                        f"[subset {bin(branch_bits)}] fast_callable "
                        f"failed: {exc}"
                    ),
                }

        # Extract linear coefficients for the polytope
        subset_constraint_data = []
        constraint_err = None
        for c in subset_retard:
            c_sr = SR(c)
            try:
                a_int = [float(c_sr.coefficient(v))
                         for v in remaining_int_vars]
                a_ext = [float(c_sr.coefficient(s)) for s in free_ext_syms]
                zero_subs = {v: 0
                             for v in list(remaining_int_vars)
                             + list(free_ext_syms)}
                c0 = float(c_sr.subs(zero_subs))
            except (TypeError, ValueError) as exc:
                constraint_err = exc
                break
            subset_constraint_data.append((a_int, a_ext, c0))
        # M1 row provenance, parallel to ``subset_constraint_data`` (see
        # ``ROW_KINDS``): the smooth-edge rows so far, then the τ box rows
        # and the conv pseudo-edge rows appended below.
        subset_row_kinds = ['edge'] * len(subset_constraint_data)

        # ── Cap each surviving τ_v integration variable to a finite
        # range ──────────────────────────────────────────────────────
        # Cumulant kernels (Gaussian, etc.) decay rapidly on a
        # kernel-natural timescale.  Without a finite cap, scipy.quad
        # integrates over (retard_L, +∞), and the adaptive Cauchy /
        # tan-substitution coordinate transform compresses the
        # kernel's central peak near the upper boundary of the
        # transformed parameter range.  The peak then gets
        # intermittently missed by the sampling — producing the
        # spurious spikes seen in non-local diagram contributions
        # at τ values where the external time t puts the kernel
        # peak in the "danger zone."  Capping τ_v ∈ (−CAP, +CAP)
        # collapses the integration to a finite interval where
        # adaptive quad is well-behaved.  Outside ±5σ a Gaussian
        # is < 1e-6 of its peak; ±50 with σ ~ 1 is overkill and
        # safe for any kernel with σ < 10.
        if extra_tau_syms and remaining_int_vars:
            n_iv = len(remaining_int_vars)
            n_ext = len(free_ext_syms)
            # Which τ symbols come from a ConvVertexType.  These are
            # causal-synaptic-kernel τ = vertex_time − leg_time, which
            # by construction have support τ ≥ 0; the heaviside(τ) was
            # stripped from cp above and the lower bound 0 (not −CAP)
            # is what carries the causality constraint into the
            # polytope.
            conv_tau_syms = set()
            for _v, _att_pairs in conv_vertex_specs.items():
                for _tau, _att in _att_pairs:
                    conv_tau_syms.add(_tau)
            for (tau_s, _v) in extra_tau_syms:
                if tau_s not in remaining_int_vars:
                    continue
                idx = remaining_int_vars.index(tau_s)
                is_conv_tau = tau_s in conv_tau_syms
                # Upper cap:  -τ_v + CAP > 0  ⇒  τ_v < CAP
                a_up  = [0.0] * n_iv
                a_up[idx] = -1.0
                subset_constraint_data.append(
                    (a_up, [0.0] * n_ext, TAU_KERNEL_CAP)
                )
                # Lower cap.  For NoiseSourceType: +τ_v + CAP > 0
                # (symmetric around 0).  For ConvVertexType: +τ_v > 0
                # (causal — kernel support starts at τ = 0).
                a_lo  = [0.0] * n_iv
                a_lo[idx] = +1.0
                lo_c0 = 0.0 if is_conv_tau else TAU_KERNEL_CAP
                subset_constraint_data.append(
                    (a_lo, [0.0] * n_ext, lo_c0)
                )
                _box_kind = 'conv_tau' if is_conv_tau else 'noise_box'
                subset_row_kinds.extend((_box_kind, _box_kind))
        if constraint_err is not None:
            return {
                'status': 'failed',
                'contribution': None,
                'integration_vars': integration_vars,
                'stripped_integrand': display_stripped,
                'constraints': display_constraints,
                'reason': (
                    f"[subset {bin(branch_bits)}] constraint not "
                    f"linear: {constraint_err}"
                ),
            }

        # ── Conv-vertex kernel pseudo-edges ──────────────────────────
        # Each surviving ``(τ, C, λ)`` mode contributes one synthetic
        # smooth edge with ``Δt = +τ`` and a single mode-sum pole at
        # ``λ = −1/τ_g``.  Appended to ``smooth_edge_modes`` and
        # ``subset_constraint_data`` at the analytic-mode-sum call
        # sites below.  The pseudo-edge's polytope constraint
        # ``Δt > 0`` SUBSUMES the one-sided ``τ > 0`` cap added above,
        # so we drop the redundant cap entry to avoid double-counting
        # the constraint in the poset extraction.
        conv_pseudo_edges = []
        conv_pseudo_constraints = []
        if conv_kernel_extracted and conv_extracted_modes:
            n_iv_sub = len(remaining_int_vars)
            n_ext_sub = len(free_ext_syms)
            for (tau_sym, C, lam) in conv_extracted_modes:
                try:
                    tau_idx_sub = remaining_int_vars.index(tau_sym)
                except ValueError:
                    # τ was eliminated by δ-edge substitution — can't
                    # happen for a ConvVertex kernel τ (no edge ever
                    # has dt = τ_kernel alone in this code path), but
                    # guard defensively.
                    conv_pseudo_edges = None
                    conv_pseudo_constraints = None
                    break
                a_int_pe = [0.0] * n_iv_sub
                a_int_pe[tau_idx_sub] = 1.0
                conv_pseudo_edges.append(EdgeModeSum(
                    ri=-1, pi=-1,            # synthetic — never indexed
                    delta_coeff=complex(0.0),
                    modes=((C, lam),),
                ))
                conv_pseudo_constraints.append(
                    (a_int_pe, [0.0] * n_ext_sub, 0.0)
                )
            # Drop the redundant one-sided ``τ > 0`` cap (added in the
            # bounds loop above) so the polytope doesn't carry the same
            # constraint twice — the poset integrator treats them as
            # independent dim-1 retardation walls and rejects identical
            # ones.
            if conv_pseudo_edges is not None:
                _conv_tau_set = {tau for (tau, _, _) in conv_extracted_modes}
                _to_keep = []
                _kinds_to_keep = []
                for c_tuple, _c_kind in zip(subset_constraint_data,
                                            subset_row_kinds):
                    a_int_c, a_ext_c, c0_c = c_tuple
                    # Identify "+τ + 0 > 0" rows for our τs (a_ext all
                    # zero, c0 == 0, a_int has a single +1 at the τ
                    # column).
                    nonzero = [(j, v) for j, v in enumerate(a_int_c) if v != 0]
                    is_pure_tau_pos = (
                        c0_c == 0.0
                        and not any(v != 0 for v in a_ext_c)
                        and len(nonzero) == 1
                        and nonzero[0][1] == 1.0
                        and remaining_int_vars[nonzero[0][0]] in _conv_tau_set
                    )
                    if not is_pure_tau_pos:
                        _to_keep.append(c_tuple)
                        _kinds_to_keep.append(_c_kind)
                subset_constraint_data = _to_keep
                subset_row_kinds = _kinds_to_keep

        # Fix E (2026-04-21): direct numerical per-edge evaluator
        # reconstructs P · Π_e Σ_k C_e^{(k)} · exp(I · p_k · Δt_e)
        # from the propagator's pole / residue data plus the
        # already-extracted ``subset_constraint_data``, without
        # materialising the distributed |edges|^|poles|-term sum
        # that fast_callable would have to compile.  Overflow-safe
        # by edge-product bound Σ_k |C_e^{(k)}| for Δt_e ≥ 0.
        #
        # Stage 4a optim (2026-05-15): ``prefactor_num`` is now
        # computed early above (used for the analytic-eligibility
        # check).  Only the ri/pi lookup remains here.
        smooth_edges_ri_pi = [
            (edge_info[_ei_idx]['ri'], edge_info[_ei_idx]['pi'])
            for _ei_idx in smooth_edges
        ]
        if vertex_leg_time and not _conv_only_leg_times:
            # NoiseSourceType vertex with non-rational (Gaussian, etc.)
            # kernel — the fast pole/residue evaluator assumes
            # P · Π_e Σ_k C_e^{(k)} exp(i p_k Δt_e), which the
            # cumulant κ-factor breaks.  Fall through to the generic
            # fast_callable path.  ConvVertex kernels with successful
            # single-exp extraction take the pseudo-edge analytic
            # path instead (handled below).
            _fast_eval = None
        else:
            # Stage 2: prefer the pre-built ``edge_mode_sums`` cache
            # (residues + λ_α extracted once per edge at the top of
            # integrate_diagram).  Falls back to the legacy in-call
            # extraction path if the cache wasn't built (incomplete
            # propagator_data).  For ConvVertex-only diagrams we
            # extend the smooth-edge list with the per-attachment
            # kernel pseudo-edges + their dt constraints.
            if edge_mode_sums is not None:
                smooth_edge_modes = [
                    edge_mode_sums[_ei_idx] for _ei_idx in smooth_edges
                ]
                if conv_pseudo_edges:
                    smooth_edge_modes = smooth_edge_modes + conv_pseudo_edges
                    subset_constraint_data = (
                        subset_constraint_data + conv_pseudo_constraints
                    )
                    subset_row_kinds = (
                        subset_row_kinds
                        + ['conv_pseudo'] * len(conv_pseudo_constraints)
                    )
                _fast_eval = _build_fast_subset_evaluator_from_modes(
                    prefactor_num,
                    smooth_edge_modes,
                    subset_constraint_data,
                    m_sub,
                )
            else:
                _fast_eval = _build_fast_subset_evaluator(
                    propagator_data,
                    prefactor_num,
                    smooth_edges_ri_pi,
                    subset_constraint_data,
                    m_sub,
                )
        integrand_for_quad = (
            _fast_eval if _fast_eval is not None else integrand_fc_sub
        )

        # Capture the smooth-edge EdgeModeSum subset + complex
        # prefactor for the analytic mode-sum paths:
        #   m = 1  → ``_integrate_1d_polytope_modesum``     (Stage 4a-perdiag)
        #   m = 2  → ``_integrate_2d_polygon_modesum``      (Stage 3a-full)
        #   m ≥ 3  → ``_integrate_nd_polytope_poset_modesum`` (Stage 3b)
        # ``None`` means the closure-only fallback path applies for
        # both (the constraints either can't be extracted or have
        # a non-numerical prefactor still).
        _smooth_edge_modes = None
        _modesum_prefactor_c = None
        _modesum_enabled = (
            (USE_1D_INTEGRATOR and m_sub == 1)
            or (USE_POLYGON_M2_INTEGRATOR and m_sub == 2)
            or (USE_POSET_INTEGRATOR and m_sub >= 3)
        )
        # ``_prefactor_c`` was already computed early (Stage 4a optim
        # 2026-05-15); reuse it instead of re-converting through SR.
        _pole_tuples_cache = None
        _modesum_plan = None
        if (_modesum_enabled
                and (not vertex_leg_time or _conv_only_leg_times)
                and edge_mode_sums is not None
                and _prefactor_is_numerical):
            _modesum_prefactor_c = _prefactor_c
            _smooth_edge_modes = [
                edge_mode_sums[_ei_idx] for _ei_idx in smooth_edges
            ]
            if conv_pseudo_edges:
                # Conv-vertex kernels enter as single-mode pseudo-edges
                # appended to the smooth-edge list, with their own
                # ``Δt = +τ`` polytope constraint already merged into
                # subset_constraint_data above.
                _smooth_edge_modes = _smooth_edge_modes + conv_pseudo_edges
            # Stage 4a-plan (2026-05-15): pre-compute the τ-invariant
            # per-pole-tuple data (alphas, γ_const, γ_slope) for the
            # entire τ grid.  Each analytic integrator threads this
            # plan through its inner loop and skips the per-call
            # edge-loop that would otherwise rebuild α_s / γ on every
            # ``_contrib(free_vals)`` invocation.  Trades a single
            # subset-setup cost for (N_τ − 1) call-time recomputations.
            _modesum_plan = _build_modesum_plan(
                _smooth_edge_modes,
                subset_constraint_data,
                m_sub,
                len(free_ext_syms),
            )
            # Keep the legacy cache populated so any caller still
            # passing ``pole_tuples=`` keeps working unchanged.
            _pole_tuples_cache = _modesum_plan['pole_tuples']

        # Build this subset's contribution callable
        def _make_subset_contrib(fc, cdata, m_val,
                                  modes=None, pref_c=None,
                                  pole_tuples=None, plan=None,
                                  hook_meta=None, row_kinds=None,
                                  theta0_empty=False):
            # ``_hook_ctx`` is passed by ``contribution()`` only while
            # ``_SUBSET_HOOK`` is set (M0.1); the hook itself is read from
            # the module global at call time.  The hook does not change the
            # value: the extra work is bookkeeping on the ``None``
            # (fallback) branches and, only with a hook, the payload.
            # ``row_kinds`` (M1) is the subset's row provenance and
            # ``_tie_ctx`` (a ``_TieContext`` from ``contribution()``) the
            # legs behind ``free_vals``; both are threaded to every
            # integrator for the Θ(0) rule (``_const_row_verdict``).
            # ``theta0_empty``: the subset has a τ-independent EMPTY row;
            # it returns 0 while the call-time mode is 'ito'.
            def _contrib(free_vals, _hook_ctx=None, _tie_ctx=None):
                if theta0_empty and not _theta0_legacy():
                    _RUNTIME_COUNTERS['theta0_subsets_pruned'] += 1
                    return 0.0 + 0.0j
                _hook = _SUBSET_HOOK
                _attempted = None
                _bail_reason = None
                # m=1 analytic 1D interval (Stage 4a-perdiag).
                if (modes is not None and pref_c is not None
                        and m_val == 1):
                    interval_val = _integrate_1d_polytope_modesum(
                        smooth_edge_modes=modes,
                        prefactor_complex=pref_c,
                        subset_constraint_data=cdata,
                        free_ext_vals=free_vals,
                        pole_tuples=pole_tuples,
                        plan=plan,
                        row_kinds=row_kinds,
                        tie_ctx=_tie_ctx,
                    )
                    if interval_val is not None:
                        if _hook is not None:
                            _emit_subset_hook(
                                _hook, hook_meta, _hook_ctx, free_vals,
                                m_val, cdata, path='m1',
                                evaluator='_integrate_1d_polytope_modesum',
                                branch='plan' if plan is not None
                                else 'noplan',
                                value=interval_val, bail_reason=None,
                                attempted='m1', modes=modes,
                                prefactor=pref_c, plan=plan,
                                pole_tuples=pole_tuples, integrand=fc,
                                row_kinds=row_kinds, tie_ctx=_tie_ctx)
                        return interval_val
                    _attempted = 'm1'
                    _bail_reason = _pop_bail_reason()
                # m=2 analytic polygon (Stage 3a-full).
                if (modes is not None and pref_c is not None
                        and m_val == 2):
                    poly_val = _integrate_2d_polygon_modesum(
                        smooth_edge_modes=modes,
                        prefactor_complex=pref_c,
                        subset_constraint_data=cdata,
                        free_ext_vals=free_vals,
                        pole_tuples=pole_tuples,
                        plan=plan,
                        row_kinds=row_kinds,
                        tie_ctx=_tie_ctx,
                    )
                    if poly_val is not None:
                        if _hook is not None:
                            _emit_subset_hook(
                                _hook, hook_meta, _hook_ctx, free_vals,
                                m_val, cdata, path='polygon',
                                evaluator='_integrate_2d_polygon_modesum',
                                branch='plan' if plan is not None
                                else 'noplan',
                                value=poly_val, bail_reason=None,
                                attempted='polygon', modes=modes,
                                prefactor=pref_c, plan=plan,
                                pole_tuples=pole_tuples, integrand=fc,
                                row_kinds=row_kinds, tie_ctx=_tie_ctx)
                        return poly_val
                    _attempted = 'polygon'
                    _bail_reason = _pop_bail_reason()
                # m≥3 analytic causal-poset chain simplex (Stage 3b).
                if (modes is not None and pref_c is not None
                        and m_val >= 3):
                    poset_val = _integrate_nd_polytope_poset_modesum(
                        smooth_edge_modes=modes,
                        prefactor_complex=pref_c,
                        subset_constraint_data=cdata,
                        free_ext_vals=free_vals,
                        m=m_val,
                        pole_tuples=pole_tuples,
                        plan=plan,
                        row_kinds=row_kinds,
                        tie_ctx=_tie_ctx,
                    )
                    if poset_val is not None:
                        if _hook is not None:
                            _emit_subset_hook(
                                _hook, hook_meta, _hook_ctx, free_vals,
                                m_val, cdata, path='poset',
                                evaluator=(
                                    '_integrate_nd_polytope_poset_modesum'),
                                branch='plan' if plan is not None
                                else 'noplan',
                                value=poset_val, bail_reason=None,
                                attempted='poset', modes=modes,
                                prefactor=pref_c, plan=plan,
                                pole_tuples=pole_tuples, integrand=fc,
                                row_kinds=row_kinds, tie_ctx=_tie_ctx)
                        return poset_val
                    _attempted = 'poset'
                    _bail_reason = _pop_bail_reason()
                    # M3: the exact DBM route (``USE_DBM_FALLBACK``).
                    if _dbm_route_on():
                        dbm_val = _integrate_subset_dbm(
                            smooth_edge_modes=modes,
                            prefactor_complex=pref_c,
                            subset_constraint_data=cdata,
                            free_ext_vals=free_vals,
                            m=m_val,
                            pole_tuples=pole_tuples,
                            plan=plan,
                            row_kinds=row_kinds,
                            tie_ctx=_tie_ctx,
                            poset_bail_reason=_bail_reason,
                        )
                        if dbm_val is not None:
                            if _hook is not None:
                                _emit_subset_hook(
                                    _hook, hook_meta, _hook_ctx, free_vals,
                                    m_val, cdata, path='dbm',
                                    evaluator='_integrate_subset_dbm',
                                    branch='plan' if plan is not None
                                    else 'noplan',
                                    value=dbm_val, bail_reason=None,
                                    attempted='poset', modes=modes,
                                    prefactor=pref_c, plan=plan,
                                    pole_tuples=pole_tuples, integrand=fc,
                                    row_kinds=row_kinds, tie_ctx=_tie_ctx)
                            return dbm_val
                        # Declined (its reason is in the dbm_declined_*
                        # counters); the poset's reason labels the fallback.
                        _pop_bail_reason()
                # Closure-only fallback via scipy.nquad.
                _count_nquad_fallback(_bail_reason, m_val)
                resolved = []
                for (a_int, a_ext, c0) in cdata:
                    c_eff = c0 + sum(a_ext[i] * free_vals[i]
                                     for i in range(len(a_ext)))
                    resolved.append((list(a_int), c_eff))
                _val = _integrate_polytope(fc, resolved, free_vals, m_val,
                                           raw_rows=cdata,
                                           row_kinds=row_kinds,
                                           tie_ctx=_tie_ctx,
                                           diag_meta=hook_meta)
                if _hook is not None:
                    if m_val < 1:
                        _reason = None
                    elif _attempted is None:
                        _reason = 'not_eligible'
                    else:
                        _reason = _bail_reason or 'unknown'
                    _emit_subset_hook(
                        _hook, hook_meta, _hook_ctx, free_vals, m_val,
                        cdata, path='nquad' if m_val >= 1 else 'm0',
                        evaluator='_integrate_polytope' if m_val >= 1
                        else 'polytope_m0',
                        branch=('plan' if plan is not None else 'noplan')
                        if _attempted is not None else None,
                        value=_val, bail_reason=_reason,
                        attempted=_attempted, modes=modes,
                        prefactor=pref_c, plan=plan,
                        pole_tuples=pole_tuples, integrand=fc,
                        row_kinds=row_kinds, tie_ctx=_tie_ctx)
                return _val
            return _contrib

        subset_contributions.append(
            _make_subset_contrib(
                integrand_for_quad, subset_constraint_data, m_sub,
                modes=_smooth_edge_modes,
                pref_c=_modesum_prefactor_c,
                pole_tuples=_pole_tuples_cache,
                plan=_modesum_plan,
                row_kinds=tuple(subset_row_kinds),
                theta0_empty=_theta0_always_empty,
                hook_meta={
                    'source': 'per_diagram',
                    'diagram_serial': _diag_serial,
                    'model': _model_id,
                    'diagram': typed_diagram,
                    'loop_number': loop_number,
                    'subset_id': branch_bits,
                    'subset_index': len(subset_contributions),
                    'delta_edges': tuple(delta_edges),
                    'smooth_edges': tuple(smooth_edges),
                },
            )
        )
        # ``_evaluator_label`` tags the INTENDED analytic path for
        # this subset.  At runtime the closure may still fall back
        # to scipy.nquad if the analytic path returns None (mixed
        # constraint, non-uniform bounds, degenerate β).  The label
        # records the design intent; runtime falls through to
        # 'fast_numpy' silently.
        if _smooth_edge_modes is not None and m_sub == 1:
            _evaluator_label = 'interval_modesum'
        elif _smooth_edge_modes is not None and m_sub == 2:
            _evaluator_label = 'polygon_modesum'
        elif _smooth_edge_modes is not None and m_sub >= 3:
            _evaluator_label = 'poset_modesum'
        elif _fast_eval is not None:
            _evaluator_label = 'fast_numpy'
        else:
            _evaluator_label = 'fast_callable'
        subset_diagnostics.append({
            'delta_edges': delta_edges,
            'smooth_edges': smooth_edges,
            'status': 'evaluated',
            'm_after_delta': m_sub,
            'evaluator': _evaluator_label,
            'theta0_always_empty': _theta0_always_empty,
        })

    # ── Build the final contribution callable ─────────────────────
    # The subset_contributions were built using the FIRST mapping.
    # To include all inter-vertex Wick contractions, we evaluate the
    # same integrand with the input arguments permuted for each
    # alternative mapping, then sum and divide by the compensation
    # factor (which removes overcounting of same-vertex permutations).
    _m0 = _all_mappings[0]
    _m0_inv = {v: k for k, v in _m0.items()}
    _k = len(ext_time_vars)

    # For each alternative mapping m, compute the permutation that
    # converts canonical-order inputs into the order the first-mapping
    # integrand expects.
    _perms = []
    for _m in _all_mappings:
        _perm = [0] * _k
        for _cp in range(_k):
            _perm[_m0_inv[_m[_cp]]] = _cp
        _perms.append(tuple(_perm))

    _comp = _compensation

    def contribution(*ext_time_values):
        if len(ext_time_values) != len(ext_time_vars):
            raise ValueError(
                f"contribution() expects {len(ext_time_vars)} positional "
                f"arguments (one per ext_time_var); got "
                f"{len(ext_time_values)}."
            )
        # Wick-contraction permutations for identical-field externals.
        # The integrand was BUILT with ``origin_leaf_idx`` pinning the
        # origin leaf at t=0, so the integrand really computes a
        # function of the time DIFFERENCES (t_j − t_origin) for the
        # non-origin leaves.  When we apply a non-identity permutation
        # ``perm`` (which relabels which leaf occupies which canonical
        # position), the origin leaf may end up at a non-zero time in
        # the user's input.  We then need to time-translate so the
        # origin leaf returns to 0; this shifts the other leaves by
        # the negative of the origin's permuted time.  Equivalently,
        # the free-time argument fed to the cfn is the time of the
        # free leaf MINUS the time of the origin leaf in that
        # permutation's frame.
        #
        # For symmetric integrands (e.g. tree-level identical-leaf
        # cumulants) every permutation gives the same value, so the
        # sum reduces to ``len(_perms) × identity-value``; ``_comp``
        # cancels it exactly.  For asymmetric integrands (every
        # 1-loop and higher diagram where the prediagram has a
        # distinguished leaf, like cubic-vertex tadpoles), the
        # different permutations are physically distinct evaluations
        # (e.g. ``V(τ) + V(−τ)`` for k=2 swap) and MUST be summed,
        # not just counted.  Before this fix the swap permutation was
        # fed free_val=0 (the pinned origin's actual time), producing
        # spurious asymmetry in C(τ) for any non-tree-level k=2
        # identical-externals case.
        #
        # M1: ``tie_ctx`` records which caller leg each free value belongs
        # to (and the origin leg), so a constant row that is exactly 0.0 is
        # ordered by the legs' raw times -- identically in every
        # permutation -- instead of by Θ(0) = 0 on both orientations
        # (``_tie_order_sign``).  It never changes a time value, and it
        # keeps the times as passed (no float() here): only the two times
        # of an exact tie are ever converted, so the argument types the
        # pre-0.2.0 callable accepted (e.g. any value at k=1, whose single
        # leg is the origin, or translation-invariant symbolic times) are
        # still accepted, in every mode.
        _hook_on = _SUBSET_HOOK is not None     # M0.1: one check per call
        _call_serial = _next_hook_call_serial() if _hook_on else None
        _raw_times = tuple(ext_time_values)
        total = 0.0 + 0.0j
        for _perm_idx, perm in enumerate(_perms):
            permuted = [ext_time_values[perm[j]] for j in range(_k)]
            if origin_leaf_idx is not None:
                t_origin = permuted[origin_leaf_idx]
                permuted = [pt - t_origin for pt in permuted]
            free_vals = [float(permuted[j]) for j in free_ext_idx]
            tie_ctx = _TieContext(
                _raw_times, tuple(perm[j] for j in free_ext_idx),
                None if origin_leaf_idx is None else perm[origin_leaf_idx])
            if not _hook_on:
                for cfn in subset_contributions:
                    total = total + complex(cfn(free_vals, None, tie_ctx))
            else:
                _ctx = _hook_eval_context(
                    _diag_serial, ext_time_values, _perm_idx, perm,
                    len(_perms), _comp, _call_serial)
                for cfn in subset_contributions:
                    total = total + complex(cfn(free_vals, _ctx, tie_ctx))
        return total / _comp

    # If non-local cumulant kernels were substituted in cp, expose
    # the τ-dependent prefactor (= the substituted, simplified cp,
    # already num_params-substituted) so the display layer can place
    # it inside the integral with the propagator factors.  When no
    # NoiseSourceType vertex is present, this is just cp (==
    # combined_prefactor.subs(num_params)) and the display layer
    # treats it as a τ-independent prefactor outside the integral.
    has_cumulant_kernel = bool(noise_source_specs)
    return {
        'status': 'ok',
        'contribution': contribution,
        'delta_contributions': delta_contributions,
        'integration_vars': integration_vars,
        'stripped_integrand': display_stripped,
        'constraints': display_constraints,
        'edge_info': edge_info,
        'n_subsets_evaluated': len(subset_contributions),
        'n_delta_contributions': len(delta_contributions),
        'n_shotnoise_skipped': n_shotnoise_skipped,
        'subset_diagnostics': subset_diagnostics,
        'cumulant_prefactor':       cp if has_cumulant_kernel else None,
        'has_cumulant_kernel':      has_cumulant_kernel,
    }


def eval_delta_contributions_on_tau_grid(
    delta_contributions,
    tau_grid,
    free_ext_dim=1,
    vary_index=0,
    fixed_values=None,
):
    r"""
    Convert a list of symbolic δ-spike contributions (from
    `integrate_tree_diagram` or `compute_correction_td`) into a
    discretized contribution on a 1D τ grid, optionally restricted to
    a slice of a higher-dimensional external-time space.

    Each delta contribution stores a linear equality
    `a · x + c = 0` in the free-external-time vector `x` (of length
    `free_ext_dim`). On a 1D slice where all-but-one entry of `x` is
    fixed, the equality collapses to `a_vary · τ + c' = 0` where
    `c' = c + Σ_{j ≠ vary_index} a_j · fixed_values[j]`. We solve
    `τ_fire = −c' / a_vary`, evaluate the coefficient callable at the
    (fixed + varying) point, check any retardation half-spaces, and
    deposit `coeff / |a_vary| / Δτ` into the nearest bin on
    `tau_grid`.

    Parameters
    ----------
    delta_contributions : list of dict
        As returned by `integrate_tree_diagram` under
        `delta_contributions` or `compute_correction_td` under the
        same key. Each entry has `equality_a`, `equality_c`,
        `coeff_fc`, and `retardation_data`.
    tau_grid : 1-D numpy array
        Uniformly spaced grid of τ values along the axis being varied.
        Bin width is inferred as `tau_grid[1] - tau_grid[0]`.
    free_ext_dim : int, default 1
        Number of free-external-time dimensions that each
        `delta_contribution['equality_a']` is parameterized over.
        For k=2 with one leaf pinned this is 1; for k=3 it is 2.
    vary_index : int, default 0
        Which component of the free-external-time vector is swept by
        `tau_grid`. All other components are pinned to the
        corresponding value in `fixed_values`.
    fixed_values : dict or None
        Mapping `{j: value}` for indices `j ≠ vary_index`. Any index
        not supplied is pinned to 0.0. For the common k=3 "slice
        through the origin" the default (all zeros) is usually what
        the caller wants.

    Returns
    -------
    numpy.ndarray (complex)
        An array the same length as `tau_grid`, with zeros everywhere
        except at the bins where δ contributions fire.

    Notes
    -----
    A δ contribution whose equality is IDENTICALLY zero on the chosen
    slice (i.e., `a_vary == 0` but the remaining slice residual is
    also zero) corresponds to a "δ along the whole slice" — the
    contribution is continuous along the varying axis rather than
    concentrated at a single bin. These contributions are **skipped**
    by this helper with a silent pass; they should be handled by a
    full 2D grid evaluator or by adding an explicit continuous
    contribution to the smooth total. Callers can inspect
    `delta_contributions` directly to detect this case.
    """
    import numpy as np

    if free_ext_dim < 1:
        raise ValueError(f"free_ext_dim must be >= 1, got {free_ext_dim}")
    if not (0 <= vary_index < free_ext_dim):
        raise ValueError(
            f"vary_index={vary_index} out of range for "
            f"free_ext_dim={free_ext_dim}"
        )
    if fixed_values is None:
        fixed_values = {}

    tau_grid = np.asarray(tau_grid, dtype=float)
    if tau_grid.size < 2:
        raise ValueError("tau_grid must have at least 2 points")
    dtau = float(tau_grid[1] - tau_grid[0])
    out = np.zeros_like(tau_grid, dtype=complex)

    # Build the full fixed-values vector (length free_ext_dim, with
    # vary_index slot filled in per-evaluation).
    other_indices = [j for j in range(free_ext_dim) if j != vary_index]
    fixed_vec_template = [0.0] * free_ext_dim
    for j in other_indices:
        fixed_vec_template[j] = float(fixed_values.get(j, 0.0))

    for dc in delta_contributions:
        eq_a = dc['equality_a']
        eq_c = dc['equality_c']
        coeff_fc = dc['coeff_fc']
        if len(eq_a) != free_ext_dim:
            # Dimension mismatch between the delta contribution and
            # the caller's advertised free_ext_dim. Silently skip.
            continue

        a_vary = eq_a[vary_index]
        c_eff = eq_c + sum(
            eq_a[j] * fixed_vec_template[j] for j in other_indices
        )

        if abs(a_vary) < 1e-15:
            if abs(c_eff) > 1e-12:
                # Infeasible: the δ surface doesn't intersect this
                # slice at all → zero contribution.
                continue
            # DEGENERATE: the δ equality is satisfied EVERYWHERE on
            # this slice. The contribution is NOT a spike — it's a
            # smooth continuous function along the varying axis:
            #   C_degenerate(τ) = coeff_fc(τ) (no 1/dtau divisor)
            # This is the "pair-driven" piece: e.g. for two identical
            # pop-1 fields at the same time (δ(τ₁) on the τ₁=0
            # slice), the remaining smooth propagator to pop-2 gives
            # a decaying function of τ₂.
            for i_tau, tau_val in enumerate(tau_grid):
                eval_vec = list(fixed_vec_template)
                eval_vec[vary_index] = float(tau_val)
                try:
                    val = complex(coeff_fc(*eval_vec))
                except Exception:
                    continue
                # Check retardation constraints at this point
                retard_ok = True
                for (a_list, c0) in dc.get('retardation_data', []):
                    if len(a_list) != free_ext_dim:
                        retard_ok = False
                        break
                    r_val = c0 + sum(
                        a_list[j] * eval_vec[j]
                        for j in range(free_ext_dim)
                    )
                    if r_val <= 0:
                        retard_ok = False
                        break
                if retard_ok:
                    out[i_tau] = out[i_tau] + val
            continue
        tau_fire = -c_eff / a_vary

        # Build the full point in free-ext-time space for evaluating
        # the coefficient callable
        eval_vec = list(fixed_vec_template)
        eval_vec[vary_index] = float(tau_fire)

        coeff_fc = dc['coeff_fc']
        try:
            coeff_val = complex(coeff_fc(*eval_vec))
        except Exception:
            continue

        # Check retardation constraints at the fire point
        retard_ok = True
        for (a_list, c0) in dc.get('retardation_data', []):
            if len(a_list) != free_ext_dim:
                retard_ok = False
                break
            val = c0 + sum(a_list[j] * eval_vec[j] for j in range(free_ext_dim))
            if val <= 0:
                retard_ok = False
                break
        if not retard_ok:
            continue

        # δ(a_vary · τ + c_eff) = δ(τ − τ_fire) / |a_vary|
        weight = coeff_val / abs(a_vary)

        # Find the nearest grid bin and add weight / dtau
        idx = int(np.argmin(np.abs(tau_grid - tau_fire)))
        out[idx] = out[idx] + weight / dtau

    return out


def eval_delta_contributions_on_2d_grid(
    delta_contributions,
    tau1_grid,
    tau2_grid,
    free_ext_dim=3,
    grid_axes=(1, 2),
    fixed_values=None,
):
    r"""
    Discretize δ-spike contributions onto a 2D grid for heatmap display.

    For each delta contribution with linear equality `a · x + c = 0`
    in the free-external-time vector `x`, this helper finds all 2D
    grid points where the equality is approximately satisfied (within
    half a grid step) and deposits the coefficient value there.

    Parameters
    ----------
    delta_contributions : list of dict
        As returned by `compute_correction_td` under `delta_contributions`.
    tau1_grid, tau2_grid : 1-D numpy arrays
        The two axes of the 2D heatmap grid. Must be uniformly spaced.
    free_ext_dim : int
        Total number of free external-time dimensions.
    grid_axes : tuple of two ints
        Which components of the free-ext vector correspond to the two
        grid axes. Default (1, 2) for k=3 with t_a=0 pinned.
    fixed_values : dict or None
        Values for any free-ext components NOT on the grid axes.
        Default: {0: 0.0} (the reference field at time 0).

    Returns
    -------
    numpy.ndarray (complex), shape (len(tau1_grid), len(tau2_grid))
        The discretized delta contributions. Add to the smooth `total_C`
        heatmap to get the full theory prediction.
    """
    import numpy as np

    if fixed_values is None:
        fixed_values = {0: 0.0}

    tau1 = np.asarray(tau1_grid, dtype=float)
    tau2 = np.asarray(tau2_grid, dtype=float)
    dt1 = float(tau1[1] - tau1[0]) if len(tau1) > 1 else 1.0
    dt2 = float(tau2[1] - tau2[0]) if len(tau2) > 1 else 1.0
    ax1, ax2 = grid_axes

    out = np.zeros((len(tau1), len(tau2)), dtype=complex)

    for dc in delta_contributions:
        eq_a = dc['equality_a']
        eq_c = dc['equality_c']
        coeff_fc = dc['coeff_fc']
        if len(eq_a) != free_ext_dim:
            continue

        a1 = eq_a[ax1]  # coefficient on the first grid axis
        a2 = eq_a[ax2]  # coefficient on the second grid axis

        # Contribution from fixed (non-grid) axes to the equality
        c_fixed = eq_c
        for idx in range(free_ext_dim):
            if idx != ax1 and idx != ax2:
                c_fixed += eq_a[idx] * float(fixed_values.get(idx, 0.0))

        # The equality on the grid is: a1·τ₁ + a2·τ₂ + c_fixed = 0
        # This is a line in the (τ₁, τ₂) plane. We find grid bins
        # that this line passes through.

        if abs(a1) < 1e-15 and abs(a2) < 1e-15:
            # No dependence on grid axes — either always or never fires
            if abs(c_fixed) < 1e-12:
                # Fires everywhere — evaluate coeff at every point
                for i, t1 in enumerate(tau1):
                    for j, t2 in enumerate(tau2):
                        eval_vec = [0.0] * free_ext_dim
                        for k_idx, v in fixed_values.items():
                            eval_vec[k_idx] = float(v)
                        eval_vec[ax1] = float(t1)
                        eval_vec[ax2] = float(t2)
                        try:
                            val = complex(coeff_fc(*eval_vec))
                        except Exception:
                            continue
                        # Check retardation
                        retard_ok = True
                        for (a_list, c0) in dc.get('retardation_data', []):
                            if len(a_list) != free_ext_dim:
                                retard_ok = False; break
                            rv = c0 + sum(a_list[m] * eval_vec[m]
                                          for m in range(free_ext_dim))
                            if rv <= 0:
                                retard_ok = False; break
                        if retard_ok:
                            # This is a 2D degenerate — the delta fires
                            # on the full 2D grid. No 1/dt divisor
                            # (the delta is in a direction orthogonal
                            # to the grid plane).
                            out[i, j] += val
            continue

        # The delta line a1·τ₁ + a2·τ₂ + c_fixed = 0 crosses the grid.
        # For each row i (fixed τ₁), solve for τ₂:
        #   τ₂_fire = -(a1·τ₁[i] + c_fixed) / a2
        # OR for each column j (fixed τ₂), solve for τ₁:
        #   τ₁_fire = -(a2·τ₂[j] + c_fixed) / a1
        # Use whichever axis has the larger coefficient (better resolved).

        if abs(a2) >= abs(a1):
            # Sweep along τ₁ axis, solve for τ₂ at each row
            for i, t1 in enumerate(tau1):
                t2_fire = -(a1 * t1 + c_fixed) / a2
                j = int(np.argmin(np.abs(tau2 - t2_fire)))
                if abs(tau2[j] - t2_fire) > dt2:
                    continue  # not within a grid cell

                eval_vec = [0.0] * free_ext_dim
                for k_idx, v in fixed_values.items():
                    eval_vec[k_idx] = float(v)
                eval_vec[ax1] = float(t1)
                eval_vec[ax2] = float(t2_fire)
                try:
                    val = complex(coeff_fc(*eval_vec))
                except Exception:
                    continue

                retard_ok = True
                for (a_list, c0) in dc.get('retardation_data', []):
                    if len(a_list) != free_ext_dim:
                        retard_ok = False; break
                    rv = c0 + sum(a_list[m] * eval_vec[m]
                                  for m in range(free_ext_dim))
                    if rv <= 0:
                        retard_ok = False; break
                if not retard_ok:
                    continue

                # The delta δ(a2·τ₂ + ...) has Jacobian 1/|a2|.
                # On the grid with spacing dt2, the bin density is
                # coeff / (|a2| · dt2).
                weight = val / (abs(a2) * dt2)
                out[i, j] += weight
        else:
            # Sweep along τ₂ axis, solve for τ₁ at each column
            for j, t2 in enumerate(tau2):
                t1_fire = -(a2 * t2 + c_fixed) / a1
                i = int(np.argmin(np.abs(tau1 - t1_fire)))
                if abs(tau1[i] - t1_fire) > dt1:
                    continue

                eval_vec = [0.0] * free_ext_dim
                for k_idx, v in fixed_values.items():
                    eval_vec[k_idx] = float(v)
                eval_vec[ax1] = float(t1_fire)
                eval_vec[ax2] = float(t2)
                try:
                    val = complex(coeff_fc(*eval_vec))
                except Exception:
                    continue

                retard_ok = True
                for (a_list, c0) in dc.get('retardation_data', []):
                    if len(a_list) != free_ext_dim:
                        retard_ok = False; break
                    rv = c0 + sum(a_list[m] * eval_vec[m]
                                  for m in range(free_ext_dim))
                    if rv <= 0:
                        retard_ok = False; break
                if not retard_ok:
                    continue

                weight = val / (abs(a1) * dt1)
                out[i, j] += weight

    return out


# ───────────────────────────────────────────────────────────────────────
# Fast numerical subset-integrand evaluator (Fix E, 2026-04-21)
# ───────────────────────────────────────────────────────────────────────

def _build_fast_subset_evaluator(
    propagator_data,
    prefactor_num,
    smooth_edges_ri_pi,
    subset_constraint_data,
    m_sub,
):
    r"""Return a Python callable that evaluates a subset's smooth
    integrand numerically, without going through
    ``fast_callable(subset_factor.expand())``.

    The subset integrand is structurally

        P · Π_e  Σ_k  C_e^{(k)} · exp(I · p_k · Δt_e)

    where ``P`` is the product of the combined prefactor and all
    δ-edge coefficients, the outer product runs over smooth edges,
    and the inner sum runs over poles.  ``fast_callable``'s overflow-
    safe form (Fix from 2026-04-08, commit 388fa7c) was to first call
    ``subset_factor.expand()``, distributing the ``|edges|^|poles|``
    cross-product into a sum of single exponentials.  For a 5-edge
    diagram with 4 poles that's ~988 single-exp terms, all compiled
    into the JIT tree.  cProfile of the k=2 ell=1 quadratic Hawkes
    V=5 diagram (2026-04-21) shows ~18 µs per ``fast_callable`` call,
    × 750 k samples per τ point, = the full 13 s of Phase J wall time.

    This evaluator skips the expansion entirely: each edge contributes
    ``Σ_k C_e^{(k)} exp(I p_k Δt_e)`` computed independently and
    multiplied in.  The product is bounded term-by-term by ``|C_e^{(k)}|``
    for Δt_e ≥ 0 (which the Heaviside filter guarantees -- see
    ``_make_heaviside_filtered_integrand``), so the pre-cancellation
    overflow that motivated the expand fix cannot occur.

    Returns None if the numerical extraction fails (e.g., ``prefactor``
    isn't purely numerical after ``num_params`` subs, or the propagator
    data lacks ``pole_vals`` / ``C_mats``).  Callers should fall back
    to ``fast_callable(subset_factor.expand())`` in that case.
    """
    import cmath as _cmath

    # ── Prefactor → complex scalar ──
    try:
        pref_c = complex(CDF(SR(prefactor_num)))
    except Exception:
        return None

    # ── Pole list (shared across edges) ──
    pole_vals = propagator_data.get('pole_vals')
    C_mats = propagator_data.get('C_mats')
    if pole_vals is None or C_mats is None:
        return None
    try:
        poles_tuple = tuple(complex(CDF(SR(p))) for p in pole_vals)
    except Exception:
        return None
    n_poles = len(poles_tuple)

    # ── Per-edge (residues, c0, int_pairs, ext_pairs) ──
    edge_data = []
    for (ri, pi), (a_int, a_ext, c0) in zip(
        smooth_edges_ri_pi, subset_constraint_data
    ):
        try:
            residues = tuple(
                complex(CDF(SR(C_mats[k][pi, ri])))
                for k in range(n_poles)
            )
        except Exception:
            return None
        # Sparse (position, coef) pairs — retardation ``Δt`` vectors
        # have exactly 1–2 nonzero entries regardless of m_sub.
        int_pairs = tuple(
            (i, float(a)) for i, a in enumerate(a_int)
            if abs(float(a)) > 1e-15
        )
        ext_pairs = tuple(
            (i, float(a)) for i, a in enumerate(a_ext)
            if abs(float(a)) > 1e-15
        )
        edge_data.append(
            (poles_tuple, residues, float(c0), int_pairs, ext_pairs)
        )
    edge_data_t = tuple(edge_data)
    m_offset = m_sub          # index into args where external times begin
    _cexp = _cmath.exp

    def evaluator(*args):
        # args = (s_0, ..., s_{m_sub-1}, t_free_0, t_free_1, ...)
        result = pref_c
        for (poles, residues, c0, int_pairs, ext_pairs) in edge_data_t:
            dt = c0
            for (i, a) in int_pairs:
                dt += a * args[i]
            for (i, a) in ext_pairs:
                dt += a * args[m_offset + i]
            # Σ_k r_k · exp(i·p_k·dt)
            edge_val = 0.0 + 0.0j
            for (p, r) in zip(poles, residues):
                edge_val += r * _cexp(1j * p * dt)
            result *= edge_val
        return result

    # Mode data for the hardened quadrature fallback (M2b): what this
    # closure evaluates, P · Π_e Σ_k r_k exp(λ_k Δt_e) with λ_k = i p_k.
    evaluator._nquad_modes = _NquadModes.from_edge_data(pref_c, edge_data_t)
    return evaluator


def _build_fast_subset_evaluator_from_modes(
    prefactor_num,
    smooth_edge_modes,
    subset_constraint_data,
    m_sub,
):
    """Stage 2 variant of ``_build_fast_subset_evaluator``.

    Same per-call evaluator semantics as the legacy function, but
    consumes pre-built ``EdgeModeSum`` objects (residues + λ_α
    extracted once per edge at the top of ``integrate_diagram``)
    instead of re-extracting them from ``propagator_data`` on every
    call.  ``smooth_edge_modes`` is the subset of the diagram's
    per-edge mode sums corresponding to the current smooth-set;
    ``subset_constraint_data`` carries the Δt linear forms after
    δ-elimination.

    The two builders return numerically identical closures (same
    edge-product loop, same complex-exp arithmetic); this one just
    skips the per-call SR → complex coercion in the legacy hot path.
    """
    import cmath as _cmath

    # ── Prefactor → complex scalar ──
    try:
        pref_c = complex(CDF(SR(prefactor_num)))
    except Exception:
        return None

    # Edge-data tuples are constructed once here from the pre-built
    # mode-sum cache + per-subset Δt constraint data.  No SR → CDF
    # coercion happens — that already ran when the EdgeModeSum list
    # was built.
    edge_data = []
    for ems, (a_int, a_ext, c0) in zip(
        smooth_edge_modes, subset_constraint_data
    ):
        # Modes are already (C_α, λ_α) with λ_α = i·p_α; the legacy
        # evaluator multiplies ``1j * p`` per-call which would double
        # the imaginary factor.  Split modes back into separate
        # poles/residues tuples for the SAME inner loop shape as the
        # legacy evaluator.
        residues = tuple(C for (C, _lam) in ems.modes)
        # λ_α = i·p_α  ⇒  p_α = -i·λ_α = λ_α / 1j
        poles = tuple((_lam / 1j) for (_C, _lam) in ems.modes)
        int_pairs = tuple(
            (i, float(a)) for i, a in enumerate(a_int)
            if abs(float(a)) > 1e-15
        )
        ext_pairs = tuple(
            (i, float(a)) for i, a in enumerate(a_ext)
            if abs(float(a)) > 1e-15
        )
        edge_data.append(
            (poles, residues, float(c0), int_pairs, ext_pairs)
        )
    edge_data_t = tuple(edge_data)
    m_offset = m_sub
    _cexp = _cmath.exp

    def evaluator(*args):
        result = pref_c
        for (poles, residues, c0, int_pairs, ext_pairs) in edge_data_t:
            dt = c0
            for (i, a) in int_pairs:
                dt += a * args[i]
            for (i, a) in ext_pairs:
                dt += a * args[m_offset + i]
            edge_val = 0.0 + 0.0j
            for (p, r) in zip(poles, residues):
                edge_val += r * _cexp(1j * p * dt)
            result *= edge_val
        return result

    # Mode data for the hardened quadrature fallback (M2b).  The fallback
    # never calls this closure outside the closed region: an empty interval
    # is answered 0 without sampling and its integrand is Heaviside-filtered
    # (``_integrate_polytope_hardened``), so ``_cexp(1j*p*dt)`` is never
    # evaluated at the far-outside points where fast poles overflow.
    evaluator._nquad_modes = _NquadModes.from_edge_data(pref_c, edge_data_t)
    return evaluator


# ───────────────────────────────────────────────────────────────────────
# Polytope-integration helpers
# ───────────────────────────────────────────────────────────────────────

def _integrate_polytope(integrand_callable, s_constraints, free_ext_vals, m,
                        raw_rows=None, row_kinds=None, tie_ctx=None,
                        mode_info=None, diag_meta=None):
    """
    Integrate `integrand_callable(s_1, ..., s_m, *free_ext_vals)` over
    the polytope `{s : a_int · s + c_eff > 0 for all constraints}`.

    s_constraints is a list of tuples `(a_int_list_of_len_m, c_eff)`.

    Constant rows (M1): with THETA0_CONST_ROW_MODE 'ito' (call time) every
    zero-normal row is first given its ``_const_row_verdict``.  An EMPTY
    verdict returns 0j at once; for c_eff != 0 the downstream checks (the
    Heaviside filter's ``always_empty``, ``_integrate_2d_polytope``,
    ``_resolve_1d_bounds``, ``_outer_bounds``, the m = 0 loop) agree with
    the verdict, and return exactly 0 for c_eff < 0 as well.  A tie
    (c_eff == 0.0) that the external-time order resolves as DROP
    (``_tie_order_sign``) is removed from ``s_constraints`` before them,
    since they would read it as Θ(0) = 0.  ``raw_rows`` -- the
    ``(a_int, a_ext, c0)`` rows behind ``s_constraints``, same order --
    let the helper see which legs a row compares (with ``tie_ctx``);
    without them (external callers) a tie is Θ(0) = 0.  ``row_kinds``: the
    rows' provenance (``None``: all 'edge').  'legacy_clip' skips all of
    this (pre-M1 behaviour).

    Structural zeros (M2a, ``STRUCTURAL_ZEROS``, call time): after the
    constant rows, an m >= 2 region whose pair rows contain a directed
    cycle with shifts summing to <= 0 (``_region_structurally_empty``,
    exact) returns 0j without quadrature (counter ``polytope_empty_cycle``;
    the entry is still counted in ``nquad_calls`` / ``scipy_nquad_called_*``).
    This covers every caller: an m=2 guard bail, a poset extraction that
    bailed before its own cycle test, grouped m2/m≥3, the SR path of a
    non-analytic subset, and direct callers.

    Hardened quadrature (M2b, ``NQUAD_HARDENED``, call time): every m >= 1
    region left after those exact tests is integrated by
    ``_integrate_polytope_hardened`` instead of the default-tolerance
    scipy.nquad routines below.  ``mode_info`` (an ``_NquadModes``; default:
    the ``_nquad_modes`` attribute of the fast evaluators) supplies the
    integrand's modes for its tolerance and truncation; ``diag_meta`` (the
    dispatch's hook metadata) names the diagram in its one-time warning.
    Neither changes anything with the flag off.
    """
    # M0.1 observational counters: m=0 is a direct evaluation, m>=1 is
    # real scipy.nquad quadrature.
    if m == 0:
        _RUNTIME_COUNTERS['polytope_m0_direct'] += 1
    else:
        _RUNTIME_COUNTERS['nquad_calls'] += 1
    if m == 1:
        _RUNTIME_COUNTERS['scipy_nquad_called_m1'] += 1
    elif m == 2:
        _RUNTIME_COUNTERS['scipy_nquad_called_m2'] += 1
    elif m >= 3:
        _RUNTIME_COUNTERS['scipy_nquad_called_mge3'] += 1
    if not _theta0_legacy():
        rows = (raw_rows if raw_rows is not None else
                [(a_int, (), c_eff) for (a_int, c_eff) in s_constraints])
        fv = free_ext_vals if raw_rows is not None else ()
        keep = None
        for idx, (a_int, a_ext, c0) in enumerate(rows):
            verdict = _const_row_verdict(a_int, a_ext, c0, fv,
                                         _row_kind(row_kinds, idx),
                                         tie_ctx)
            if verdict == 'EMPTY':
                return 0.0 + 0.0j
            if verdict == 'DROP' and not float(s_constraints[idx][1]) > 0.0:
                keep = keep if keep is not None else set(
                    range(len(s_constraints)))
                keep.discard(idx)              # a tie ordered as holding
        if keep is not None:
            s_constraints = [row for i, row in enumerate(s_constraints)
                             if i in keep]
    if (m >= 2 and _structural_zeros_on()
            and _region_structurally_empty(s_constraints, m) == 'cycle'):
        _RUNTIME_COUNTERS['polytope_empty_cycle'] += 1
        return 0.0 + 0.0j
    if m >= 1 and _nquad_hardened_on():
        val = _integrate_polytope_hardened(
            integrand_callable, s_constraints, free_ext_vals, m,
            mode_info=mode_info, diag_meta=diag_meta)
        if val is not None:
            return val
        # ``None``: a non-finite row shift; the legacy routines below.
    if m == 0:
        # Zero integration variables — the "integrand" is just a number.
        # Still have to check the constraints (they may be vacuous or
        # infeasible).  Θ(0) = 0 convention: boundary c_eff = 0 is OUTSIDE
        # the feasible region (the half-space is strictly open at Δt = 0).
        for (a_int, c_eff) in s_constraints:
            if c_eff <= 0:
                return 0.0 + 0.0j
        val = integrand_callable(*free_ext_vals)
        return complex(val)

    if m == 1:
        return _integrate_1d_polytope(
            integrand_callable, s_constraints, free_ext_vals
        )

    if m == 2:
        return _integrate_2d_polytope(
            integrand_callable, s_constraints, free_ext_vals
        )

    return _integrate_nd_polytope(
        integrand_callable, s_constraints, free_ext_vals, m
    )


def _make_heaviside_filtered_integrand(integrand_callable, s_constraints,
                                        free_ext_vals, m):
    r"""
    Wrap `integrand_callable` with an explicit Heaviside-product check.

    The polytope bounds we pass to `scipy.nquad` are only an approximation
    of the true polytope when some constraints couple multiple integration
    axes: cross-axis constraints must be deferred to an inner axis, and
    when the bounds function for an outer axis loses its lower or upper
    bound from such deferred constraints, we fall back to ±OUTER_CAP.
    That fallback admits regions geometrically OUTSIDE the true polytope.

    Physically the retarded propagator `G^R(Δt) = Θ(Δt) · G^sm(Δt)`
    vanishes on those regions via the Heaviside.  But our JIT-compiled
    integrand contains ONLY `G^sm`, never `Θ` — it relies entirely on the
    polytope bounds for retardation.  When the bounds overshoot, the
    integrand evaluates `G^sm` on a region where it should be zero, and
    for retarded poles (Im(ω) > 0) `G^sm(Δt) = C · exp(-γ Δt)` GROWS for
    `Δt < 0`, producing a spurious positive contribution.

    This wrapper explicitly multiplies by the Heaviside product, i.e.
    returns 0.0 whenever any `a_int · s + c_eff < 0`.  The polytope
    bounds then serve only as an optimization: they tighten the
    quadrature domain for speed, but correctness no longer depends on
    them being exact.

    Parameters
    ----------
    integrand_callable : callable
        The JIT-compiled smooth integrand `f(s_0, ..., s_{m-1},
        *free_ext_vals)`.
    s_constraints : list of (a_int, c_eff)
        Polytope constraints `a_int · s + c_eff > 0` for each retarded
        edge still active at this subset (strict inequality per Θ(0) = 0
        convention — the boundary Δt = 0 is OUTSIDE the feasible region).
        `c_eff` already has the current external-time values substituted
        in (via the caller).
    free_ext_vals : list
        Free external time values, passed through to `integrand_callable`.
    m : int
        Number of integration axes (`s_0, ..., s_{m-1}`).

    Returns
    -------
    callable f(*s_vals) → complex, with the Heaviside filter applied.
    """
    # Convention: Θ(0) = 0.  A constraint `a_int · s + c_eff > 0` is
    # STRICTLY required: the boundary `Δt = 0` is excluded.  Use `dt <= 0`
    # to kill both the exterior (strictly infeasible) AND the boundary.
    #
    # Fix D (2026-04-21): pre-extract constraints in SPARSE form so the
    # hot loop skips zero-coefficient axes entirely.  For retarded
    # propagators each constraint is `t_v − t_u > 0` which has exactly
    # two nonzero entries in `a_int` (regardless of `m`), so this
    # collapses an inner `for j in range(m)` loop into 2 iterations.
    #
    # A pre-check here handles constraints that are purely trivial
    # (all `a` zero): if `c_eff > 0` the constraint is always satisfied
    # and we drop it; if `c_eff <= 0` the polytope is empty and the
    # filter always returns 0.  (M1: with THETA0_CONST_ROW_MODE 'ito',
    # ``_integrate_polytope`` has already applied ``_const_row_verdict``,
    # returned 0 for an EMPTY constant row and removed a tie ordered as
    # holding, so only rows with c_eff > 0 reach this check from there.)
    sparse = []
    always_empty = False
    for (a_int, c_eff) in s_constraints:
        c_eff_f = float(c_eff)
        pairs = tuple((j, float(a)) for j, a in enumerate(a_int)
                      if abs(float(a)) > 1e-15)
        if not pairs:
            # Pure constant constraint.  Θ(0) = 0: strict c_eff > 0.
            if c_eff_f <= 0.0:
                always_empty = True
                break
            # Trivially satisfied — drop.
            continue
        sparse.append((c_eff_f, pairs))
    # Tuple-of-tuples for slightly faster iteration than list-of-tuples
    # (CPython's FOR_ITER has a specialized path for tuples).
    sparse_constraints = tuple(sparse)
    free_ext_tuple = tuple(free_ext_vals)

    if always_empty:
        def filtered_empty(*s_vals):
            return 0.0 + 0.0j
        return filtered_empty

    # Capture `free_ext_vals` as a LIST (not tuple) in the closure so
    # the argument-packing style matches the pre-Fix D code exactly —
    # `integrand_callable(*args)` with `args = list(s_vals) +
    # free_ext_list`.  Measured: `integrand_callable(*s_vals,
    # *free_ext_tuple)` (Python 3.5+ multi-unpack) triggers a slow
    # path inside Sage's fast_callable on the 1-loop m=3 workload
    # (~2× wall-clock regression vs. baseline) for reasons not yet
    # root-caused.  Empirically the single-unpack form matches
    # baseline performance while still benefiting from the sparse-
    # scan speedup below.
    free_ext_list = list(free_ext_vals)

    def filtered(*s_vals):
        # Heaviside check: every constraint must be > 0 (Θ(0) = 0).
        for c_eff, nzs in sparse_constraints:
            dt = c_eff
            for (j, a) in nzs:
                dt += a * s_vals[j]
            if dt <= 0.0:
                return 0.0 + 0.0j
        return complex(integrand_callable(*(list(s_vals) + free_ext_list)))

    return filtered


def _complex_quad(integrand_callable, s_slot_index, other_args, lower, upper):
    """
    1D quadrature of `integrand_callable` along `s_slot_index`, with
    the other arguments fixed to `other_args`, over `[lower, upper]`.

    Real and imaginary parts are integrated separately via
    `scipy.integrate.quad`, which handles ±inf bounds natively.
    """
    from scipy.integrate import quad

    def _eval(s_val):
        args = list(other_args)
        args.insert(s_slot_index, float(s_val))
        val = integrand_callable(*args)
        return complex(val)

    def f_re(s_val):
        return _eval(s_val).real

    def f_im(s_val):
        return _eval(s_val).imag

    re_val, _ = quad(f_re, lower, upper, **QUAD_OPTS)
    try:
        im_val, _ = quad(f_im, lower, upper, **QUAD_OPTS)
    except Exception:
        im_val = 0.0
    return complex(re_val, im_val)


def _resolve_1d_bounds(s_constraints, s_index):
    """
    For the given integration-variable index `s_index`, intersect all
    half-line constraints `a_i s_{s_index} + (other terms already
    substituted) + c_eff > 0` into a single interval `[L, U]`.

    Each element of `s_constraints` is `(a_int_list, c_eff_scalar)`
    where `a_int_list` has length equal to the number of integration
    variables. For this 1D-on-one-axis pass we assume the other axes'
    coefficients are zero (i.e., the caller has already substituted
    them).

    Returns (L, U). If the intersection is infeasible we return a
    DEGENERATE empty interval `(0.0, 0.0)` rather than a flipped
    (inf, -inf) sentinel. This matters because `scipy.quad(f, 0, 0)`
    returns 0 correctly, while `scipy.quad(f, +inf, -inf)` returns
    `-quad(f, -inf, +inf)` — the full real-line integral with a
    sign flip, which silently poisons any outer quadrature
    (`scipy.nquad`) that feeds this bounds function into
    `scipy.quad`. The 1D code path catches infeasible ranges
    up-front via `if L >= U: return 0`, so this degenerate form is
    indistinguishable from the old sentinel for the 1D path; but
    the 2D path needs the degenerate-empty form so that the inner
    integral returns 0 where the projection is empty.
    """
    L, U = -math.inf, math.inf
    infeasible = False
    for (a_int, c_eff) in s_constraints:
        a = a_int[s_index]
        if abs(a) < 1e-15:
            # Degenerate constraint (no dependence on s_index).  Under
            # Θ(0) = 0, the inequality is strict: `c_eff > 0` required.
            # Boundary c_eff = 0 is infeasible.
            if c_eff <= 0:
                infeasible = True
                break
            continue
        bound = -c_eff / a
        if a > 0:
            if bound > L:
                L = bound
        else:
            if bound < U:
                U = bound
    if infeasible or L >= U:
        return 0.0, 0.0   # degenerate empty interval
    return L, U


def _integrate_1d_polytope(integrand_callable, s_constraints, free_ext_vals):
    """Single integration variable. The polytope is an interval.

    A single axis is always cleanly bounded by the polytope (no deferred
    constraints possible since there's no inner axis to defer to), so
    the bounds (L, U) returned by ``_resolve_1d_bounds`` are exact.
    When ``DEBUG_HEAVISIDE_GUARD`` is False (production default) we
    skip the Heaviside-filter wrapper entirely — it's redundant given
    exact bounds and adds ~few µs per integrand call.  Flip the flag
    to True if you want belt-and-braces validation.
    """
    L, U = _resolve_1d_bounds(s_constraints, s_index=0)
    if L >= U:
        return 0.0 + 0.0j

    if DEBUG_HEAVISIDE_GUARD:
        # Wrapped path: filter every integrand call against the
        # retarded constraints.  No-op on the (L, U) interior but
        # catches any drift if the bound resolver is wrong.
        filt = _make_heaviside_filtered_integrand(
            integrand_callable, s_constraints, free_ext_vals, m=1,
        )

        def f_re(s_0):
            return filt(s_0).real

        def f_im(s_0):
            return filt(s_0).imag
    else:
        # Fast path: bounds are exact so the filter is redundant.
        free_ext_list = list(free_ext_vals)

        def f_re(s_0):
            return complex(integrand_callable(s_0, *free_ext_list)).real

        def f_im(s_0):
            return complex(integrand_callable(s_0, *free_ext_list)).imag

    from scipy.integrate import quad
    re_val, _ = quad(f_re, L, U, **QUAD_OPTS)
    try:
        im_val, _ = quad(f_im, L, U, **QUAD_OPTS)
    except Exception:
        im_val = 0.0
    return complex(re_val, im_val)


def _integrate_2d_polytope(integrand_callable, s_constraints, free_ext_vals):
    """
    Two integration variables s_0, s_1.

    Use `scipy.integrate.nquad` with `s_0` as the innermost integral
    and `s_1` as the outermost. The `s_0` bounds depend on s_1; the
    `s_1` bounds are extracted from constraints with zero coefficient
    on `s_0` (if any), defaulting to `(-inf, +inf)` if there are no
    pure-`s_1` constraints.
    """
    from scipy.integrate import nquad

    # ── Pre-split 2D constraints by role w.r.t. s_0 (Fix D) ──────
    # For each constraint `a_0 s_0 + a_1 s_1 + c_eff > 0` we
    # classify:
    #   pure_0      : a_1 ≈ 0, a_0 ≠ 0 → precomputed (L, U) on s_0.
    #   mixed       : both a_0 ≠ 0 and a_1 ≠ 0 → resolve per s_1 call.
    #   s1_only     : a_0 ≈ 0, a_1 ≠ 0 → pure residual inequality in s_1.
    #   constant    : a_0 ≈ 0, a_1 ≈ 0 → constant check (handled at build).
    pure_0_L = -math.inf
    pure_0_U = math.inf
    mixed_s0 = []       # list of (a_0, c_eff, a_1)
    s1_residual = []    # list of (a_1, c_eff): constraint reduces to a_1 s_1 + c_eff > 0
    bounds_s0_always_empty = False
    for (a_int, c_eff) in s_constraints:
        a_0 = float(a_int[0])
        a_1 = float(a_int[1])
        c_f = float(c_eff)
        if abs(a_0) < 1e-15 and abs(a_1) < 1e-15:
            # Constant constraint; Θ(0) = 0 wants c_eff > 0.
            if c_f <= 0.0:
                bounds_s0_always_empty = True
                break
            # Trivially satisfied — drop.
            continue
        if abs(a_0) < 1e-15:
            s1_residual.append((a_1, c_f))
            continue
        if abs(a_1) < 1e-15:
            bound = -c_f / a_0
            if a_0 > 0.0:
                if bound > pure_0_L:
                    pure_0_L = bound
            else:
                if bound < pure_0_U:
                    pure_0_U = bound
        else:
            mixed_s0.append((a_0, c_f, a_1))

    mixed_s0_t = tuple(mixed_s0)
    s1_residual_t = tuple(s1_residual)

    if bounds_s0_always_empty or pure_0_L >= pure_0_U:
        # Entire polytope empty.
        def bounds_s0(s_1_val):
            return 0.0, 0.0
    else:
        def bounds_s0(s_1_val):
            # Check s_1-only residual constraints (pure a_1 s_1 + c > 0).
            for (a_1, c_f) in s1_residual_t:
                if a_1 * s_1_val + c_f <= 0.0:
                    return 0.0, 0.0
            L = pure_0_L
            U = pure_0_U
            for (a_0, c_f, a_1) in mixed_s0_t:
                bound = -(c_f + a_1 * s_1_val) / a_0
                if a_0 > 0.0:
                    if bound > L:
                        L = bound
                else:
                    if bound < U:
                        U = bound
            if L >= U:
                return 0.0, 0.0
            return L, U

    # Bounds on s_1: use constraints where a_0 = 0 AND a_1 != 0
    # (genuinely pure-s_1); else fall back to the ±OUTER_CAP
    # Heaviside-filtered quadrature domain.
    #
    # Subtle: a constraint with BOTH a_int[0] ≈ 0 and a_int[1] ≈ 0 is
    # NOT a bound on s_1 at all — it's a pure-external inequality that
    # either kills the polytope (c_eff ≤ 0 under Θ(0) = 0) or is
    # trivially satisfied (c_eff > 0).  Such constraints must NOT set
    # the `pure_s1_found` flag, otherwise we skip the OUTER_CAP fallback
    # and scipy.nquad runs on an unbounded axis, oversampling the
    # infinite-domain transform grid and biasing the integral.  At k=4
    # with distinct fields, δ-sifting can pin an integration variable
    # to external times and leave residual pure-external constraints
    # that trigger this path — the overshoot source for ~12% of
    # model-vs-sim at k=4.
    L1, U1 = math.inf, -math.inf
    pure_s1_found = False
    tmp_L, tmp_U = -math.inf, math.inf
    for (a_int, c_eff) in s_constraints:
        if abs(a_int[0]) < 1e-15:
            a = a_int[1]
            if abs(a) < 1e-15:
                # Pure-external constraint: NOT a bound on s_1.
                # Θ(0) = 0: strict c_eff > 0 required; boundary infeasible.
                if c_eff <= 0:
                    return 0.0 + 0.0j
                continue
            # Genuine pure-s_1 constraint.
            pure_s1_found = True
            bound = -c_eff / a
            if a > 0 and bound > tmp_L:
                tmp_L = bound
            elif a < 0 and bound < tmp_U:
                tmp_U = bound
    if pure_s1_found:
        L1, U1 = tmp_L, tmp_U
        if L1 >= U1:
            return 0.0 + 0.0j
    else:
        # Fallback cap: ±200 is ample with Heaviside-filtered integrand
        # (see _make_heaviside_filtered_integrand — correctness doesn't
        # depend on the cap being tight, only large enough to contain
        # the decaying tail).
        L1, U1 = -200.0, 200.0

    # Heaviside-filtered integrand: correctness no longer depends on
    # the (L1, U1) cap being exactly the true polytope projection.
    # Skip the filter when bounds are EXACT (pure_s1_found) — the
    # inner ``bounds_s0`` callable is exact too, so the integration
    # domain matches the polytope precisely.  Cap-fallback path
    # always needs the filter regardless of DEBUG_HEAVISIDE_GUARD.
    needs_filter = (not pure_s1_found) or DEBUG_HEAVISIDE_GUARD
    if needs_filter:
        filt = _make_heaviside_filtered_integrand(
            integrand_callable, s_constraints, free_ext_vals, m=2,
        )

        def f_re(s_0, s_1):
            return filt(s_0, s_1).real

        def f_im(s_0, s_1):
            return filt(s_0, s_1).imag
    else:
        free_ext_list = list(free_ext_vals)

        def f_re(s_0, s_1):
            return complex(
                integrand_callable(s_0, s_1, *free_ext_list)
            ).real

        def f_im(s_0, s_1):
            return complex(
                integrand_callable(s_0, s_1, *free_ext_list)
            ).imag

    re_val, _ = nquad(f_re, [bounds_s0, (L1, U1)], opts=QUAD_OPTS)
    try:
        im_val, _ = nquad(f_im, [bounds_s0, (L1, U1)], opts=QUAD_OPTS)
    except Exception:
        im_val = 0.0
    return complex(re_val, im_val)


def _integrate_nd_polytope(integrand_callable, s_constraints, free_ext_vals, m):
    """
    General `m >= 3` case via `scipy.integrate.nquad` with nested
    bound functions.

    Variable ordering for nquad: the FIRST argument of the integrand
    is the INNERMOST integration variable.  We integrate s_0 first
    (innermost), then s_1, ..., up to s_{m-1} (outermost).

    For each variable s_k, its bounds are computed by:
      1. Substituting all OUTER variables (s_{k+1}, ..., s_{m-1}) and
         the external times into the linear constraints
         `a · s + c_eff > 0`.
      2. Resolving the remaining 1D polytope on s_k via
         `_resolve_1d_bounds`.
    """
    from scipy.integrate import nquad

    def _make_bound_fn(k_var):
        """Return a function `bounds_k(*outer_vals)` that returns (L, U)
        for variable s_{k_var} given all the OUTER variable values.

        nquad calls bound functions with the outer variables in
        REVERSE order (s_{k+1}, s_{k+2}, ..., s_{m-1}).

        Important subtlety — constraints that still couple to a MORE-
        INNER axis s_j (j < k_var) must be SKIPPED here.  Those
        constraints are genuine bounds on the inner variable, not on
        s_{k_var}, and they will be resolved when nquad nests into the
        deeper (smaller-index) integration axis and s_j becomes the
        resolution target.  Failing to filter them is a latent bug:
        `_resolve_1d_bounds` inspects only `a_int[s_index]` and treats
        `|a_int[k_var]| < 1e-15` as a pure-residual check, which then
        spuriously declares the polytope infeasible whenever the
        accumulated residual (still containing the unresolved inner
        coefficient) is negative.  This mirrors the filter
        `abs(a_int[0]) < 1e-15` that `_integrate_2d_polytope` already
        applies to the outer-axis bound (lines ~1240–1255).

        Regression: test_phase_J_nd_polytope_preserves_deferred_constraints
        in tests/test_time_domain.py.

        Fix D (2026-04-21): constraint classification and sparse-
        coefficient extraction happen ONCE at closure-build time, not
        on every call.  Constraints with only `s_{k_var}` nonzero
        (no outer coupling) contribute a fixed slice to (L, U) that we
        precompute.  The per-call path now only iterates constraints
        whose bound actually varies with the outer values.
        """
        # ── Classify constraints by their role w.r.t. axis k_var ──
        # Four buckets (after filtering deferred-inner constraints):
        #   pure_k      : a[k_var] != 0, no outer coupling → precomputed
        #                 contribution to (L, U).
        #   mixed       : a[k_var] != 0, some outer axes coupled →
        #                 resolve per call using outer vals.
        #   outer_only  : a[k_var] == 0, some outer axes coupled →
        #                 pure residual inequality; kills polytope when
        #                 negative at this outer point.
        #   (trivial satisfied or trivially infeasible constraints
        #    are resolved at build time.)
        pure_k_L = -math.inf
        pure_k_U = math.inf
        mixed = []         # list of (a_k, c_eff, outer_pairs)
        outer_only = []    # list of (c_eff, outer_pairs)
        infeasible_at_build = False
        for (a_int, c_eff) in s_constraints:
            # Skip constraints that still couple to a more-inner axis.
            deferred = False
            for j in range(k_var):
                if abs(a_int[j]) >= 1e-15:
                    deferred = True
                    break
            if deferred:
                continue
            a_k = float(a_int[k_var])
            # Sparse outer-axis coefficients: tuple of (outer_index, coeff)
            # where outer_index is the position in *outer_vals (not the
            # absolute axis index).  nquad passes outer_vals with
            # outer_vals[0] = s_{k_var+1}, [1] = s_{k_var+2}, etc.,
            # which matches the original `j = k_var + 1 + i_outer`
            # indexing.
            outer_pairs = tuple(
                (j - (k_var + 1), float(a_int[j]))
                for j in range(k_var + 1, len(a_int))
                if abs(float(a_int[j])) >= 1e-15
            )
            c_eff_f = float(c_eff)
            if abs(a_k) < 1e-15:
                if not outer_pairs:
                    # Constant constraint: Θ(0)=0 wants c_eff > 0.
                    if c_eff_f <= 0.0:
                        infeasible_at_build = True
                        break
                    # Trivially satisfied → drop.
                    continue
                outer_only.append((c_eff_f, outer_pairs))
            else:
                if outer_pairs:
                    mixed.append((a_k, c_eff_f, outer_pairs))
                else:
                    bound = -c_eff_f / a_k
                    if a_k > 0.0:
                        if bound > pure_k_L:
                            pure_k_L = bound
                    else:
                        if bound < pure_k_U:
                            pure_k_U = bound

        # If the polytope is empty under Θ(0)=0 irrespective of outer
        # values, or the pure-k_var bounds are crossed, return a
        # constant-zero bounds function — scipy.quad(a, a) is 0, so
        # this cleanly kills the outer integral over the empty region.
        if infeasible_at_build or pure_k_L >= pure_k_U:
            def _bounds_infeas(*outer_vals):
                return 0.0, 0.0
            return _bounds_infeas

        mixed_t = tuple(mixed)
        outer_only_t = tuple(outer_only)

        def bounds_k(*outer_vals):
            # Start from the precomputed pure-axis bounds (may be ±inf
            # if no pure-k constraint is present on that side — mixed
            # constraints below may finitely bound the axis instead).
            L = pure_k_L
            U = pure_k_U
            # Check outer-only constraints first (cheap polytope kill).
            for c_eff, outer_pairs in outer_only_t:
                val = c_eff
                for (oi, a) in outer_pairs:
                    val += a * outer_vals[oi]
                if val <= 0.0:
                    return 0.0, 0.0
            # Apply mixed (outer-coupled on-axis) constraints.
            for a_k, c_eff, outer_pairs in mixed_t:
                total_c = c_eff
                for (oi, a) in outer_pairs:
                    total_c += a * outer_vals[oi]
                bound = -total_c / a_k
                if a_k > 0.0:
                    if bound > L:
                        L = bound
                else:
                    if bound < U:
                        U = bound
            if L >= U:
                return 0.0, 0.0
            # Cap infinite bounds only when still unbounded.  Matches
            # the original post-`_resolve_1d_bounds` behaviour where a
            # cap never clips a finite constraint-derived bound.
            if math.isinf(L):
                L = -OUTER_CAP
            if math.isinf(U):
                U = OUTER_CAP
            return L, U
        return bounds_k

    # Fallback cap for any axis whose bounds resolve to (-inf, +inf).
    # With the Heaviside-filtered integrand (below), correctness does
    # NOT depend on this cap being tight — the filter kills any
    # contribution from the region outside the true polytope.  The cap
    # only needs to be large enough that the decaying integrand has
    # effectively vanished by the boundary.  For retarded propagators
    # with time constant τ, ~10τ gives exp(-10) ≈ 5e-5 tail.  Hawkes
    # τ=10 ⟹ ±200 is ample.  A wider cap is harmless (the filter
    # zeros out the extra volume) but slightly slower to quadrature
    # through.
    #
    # 2026-05-17: when USE_POSET_CAP_MATCH_SCIPY is enabled (default),
    # we tie OUTER_CAP to POSET_PHYSICAL_MARGIN so the m≥3 scipy
    # fallback integrates over the same domain as the analytic poset
    # path's unbounded-below fallback.  For marginal-stability models
    # (Re β ≈ 0) the integral is genuinely cap-dependent and the two
    # paths must agree on the cap or the grouped/per-diag comparison
    # disagrees.  For strongly-decaying models the integrand is
    # negligible at |s| > 50 anyway, so this is harmless.
    OUTER_CAP = _nquad_outer_cap()

    # Outermost variable s_{m-1}: bounds computed from constraints with
    # zero coefficient on all inner variables (pure-s_{m-1}).  If no
    # such constraints exist, fall back to ±OUTER_CAP.
    L_out, U_out = _outer_bounds(s_constraints, m - 1)
    if L_out >= U_out:
        return 0.0 + 0.0j
    if not math.isfinite(L_out):
        L_out = -OUTER_CAP
    if not math.isfinite(U_out):
        U_out = OUTER_CAP

    # Build the list of bound specifications for nquad.
    # Order: innermost first.  For variables s_0, ..., s_{m-2}, use
    # callable bounds (functions of outer vars).  For s_{m-1}, use the
    # constant tuple (L_out, U_out).
    bound_specs = [_make_bound_fn(k) for k in range(m - 1)]
    bound_specs.append((L_out, U_out))

    # Heaviside-filtered integrand: evaluates to 0 outside the true
    # polytope, guarding against any cap overshoot or deferred-
    # constraint leak.
    filt = _make_heaviside_filtered_integrand(
        integrand_callable, s_constraints, free_ext_vals, m=m,
    )

    def f_re(*all_args):
        # nquad passes integration variables (s_0, ..., s_{m-1}).
        return filt(*all_args).real

    def f_im(*all_args):
        return filt(*all_args).imag

    re_val, _ = nquad(f_re, bound_specs, opts=QUAD_OPTS)
    try:
        im_val, _ = nquad(f_im, bound_specs, opts=QUAD_OPTS)
    except Exception:
        im_val = 0.0
    return complex(re_val, im_val)


def _outer_bounds(s_constraints, k_var):
    """Compute bounds on s_{k_var} from constraints whose coefficients on
    all OTHER integration variables vanish (pure-s_{k_var} constraints).

    Used for the outermost integration variable in nD polytope
    integration, where there are no further outer variables to
    substitute.  If no pure constraint exists, returns (-inf, +inf).
    """
    L, U = -math.inf, math.inf
    pure_found = False
    for (a_int, c_eff) in s_constraints:
        # Check that a_int is purely k_var-dependent
        is_pure = True
        for j in range(len(a_int)):
            if j == k_var:
                continue
            if abs(a_int[j]) >= 1e-15:
                is_pure = False
                break
        if not is_pure:
            continue
        pure_found = True
        a = a_int[k_var]
        if abs(a) < 1e-15:
            # Degenerate pure-k_var constraint.  Θ(0) = 0: boundary
            # c_eff = 0 is infeasible; strict c_eff > 0 required.
            if c_eff <= 0:
                return math.inf, -math.inf  # infeasible
            continue
        bound = -c_eff / a
        if a > 0:
            if bound > L:
                L = bound
        else:
            if bound < U:
                U = bound
    if pure_found and L >= U:
        return math.inf, -math.inf
    if not pure_found:
        # No pure constraints — let inner bounds clip via callables.
        return -math.inf, math.inf
    return L, U


# ───────────────────────────────────────────────────────────────────────
# Hardened quadrature fallback (M2b; plan §3.1 L1b; flag NQUAD_HARDENED)
# ───────────────────────────────────────────────────────────────────────
# The default-tolerance scipy.nquad routines above integrate a region over
# a ±200 (m = 2 outer axis, m >= 3) box with the integrand Heaviside-
# filtered, scipy's default epsabs (1.49e-8) and no breakpoints.  A narrow
# peak (fast poles: width ~1/12 against a box of 400) can then fall between
# every Gauss-Kronrod node of the first panel: both rules give ~0, the error
# estimate is ~0 and the panel is accepted -- the whole region is lost
# (tests/test_phase_j_nquad_hardening.py, section B).  An empty
# inner interval (bounds (0, 0)) is still sampled, at s = 0 outside the
# region, where fast poles overflow.  ``_integrate_polytope_hardened``
# replaces them (flag on):
#
# * Bounds: the rows' own.  Every level of the nested quadrature integrates
#   its variable over the exact projection of the region onto it given the
#   outer variables (rounded outward by one ulp).  The difference rows (one
#   nonzero coefficient, or two exactly opposite ones) are closed exactly
#   (``_dbm_closure``: Floyd-Warshall on rational bounds), which gives that
#   projection when they are the only rows; with any other row, every row
#   enters an exact Fourier-Motzkin elimination (``_fm_level_rows``,
#   rationals), whose rows at each level bound it exactly (counter
#   ``nquad_hardened_general_rows``; beyond _NQUAD_FM_MAX_ROWS rows the
#   other rows only bound the level of their innermost variable, a superset,
#   counted in ``nquad_hardened_fm_capped``).  The Heaviside filter enforces
#   every row pointwise.  An infeasible closure or elimination -> 0 without
#   quadrature.
# * An empty interval returns 0 without evaluating anything, and the
#   integrand is never evaluated where a row is <= 0 (Θ(0) = 0).
# * A side that the rows leave open is truncated at K / κ (K =
#   NQUAD_TAIL_K) beyond the farthest of the interval's finite end and its
#   breakpoints on that side (kinks of the inner integral, external times;
#   beyond them the slices change only affinely), κ a certified decay rate
#   along it.  Decay certificate: every mode with a nonzero coefficient has
#   Re λ < 0 (κ_min > 0, the slowest rate −Re λ), so |integrand| <=
#   S·exp(−Σ_e κ_e·Δt_e) where every Δt_e >= 0, and
#   - when every row is a unit difference row and an edge, and every edge
#     is a row (``_unit_edge_rows``; every Phase J region measured), κ =
#     κ_min: the rows are totally unimodular, so along an open direction Σ_e
#     κ_e·Δt_e grows at least at the rate κ_min (or not at all, and the
#     integral would diverge).  The tail beyond the outermost cut, at the
#     distance d from the finite end, is at most (S/κ_min^m)·Γ(m, κ_min·d)/
#     Γ(m) (= S·κ_min^{-m}·e^{-K}·Σ_{j<m} K^j/j!, K = κ_min·d: for d = K/κ,
#     1.7e-16·S/κ^m at m = 2, 3.6e-15 at m = 3, 2.8e-11 at m = 7): in the
#     values of a spanning tree of rows that contains the closure's
#     shortest path from that variable to its finite end (|Jacobian| = 1),
#     the path's values add up to d, every tree value decays at least at
#     the rate κ_min and the other rows' factors are <= 1, whatever the
#     shape of the slices (S: the modes' bound on the box of the region's
#     level intervals, at most |pref|·Π_e Σ_k |C_ek|);
#   - otherwise (other coefficients, rows that are not edges)
#     ``_GeneralDecayCertificate``: linear programs give the decay rate of
#     min φ (φ = Σ_e κ_e·Δt_e) along every open side of every level, κ =
#     min(κ_min·a, those rates) (a <= 1: the smallest |coefficient|; the
#     rate depends on how the rows combine, not on how a row is scaled),
#     and bound the outermost tail by S·e^{-v}·∫ e^{-ρy}·Π_k w_k(y) dy from
#     the minimum v of φ and the widths w_k of the slices beyond the cut
#     (the slice's volume, not a chain's).  Edges that can be negative on
#     the region, a side without decay, slices of the outermost level that
#     are unbounded in an inner variable, or an incomplete vertex
#     enumeration of that level: no certificate.
#   These bounds are relative to S, not to the integral, which is far
#   smaller when the mass sits far from the finite end (an integral of s_1
#   peaked at a kink 60 from it: ~S·e^{-κ·60}).  The cut beyond the
#   farthest breakpoint reaches such mass (the inner integral is a sum of
#   exponentials times polynomials between breakpoints); and after a pass
#   the outermost cut's tail bound is compared with _NQUAD_TAIL_REL ×
#   |result|: above it, the cut moves out and the strip between the two
#   cuts is integrated and added (``nquad_hardened_tail_widened``).  With
#   the LP certificate the inner levels' cuts are checked too, by a bound of
#   everything the cuts drop (``region_tail``): above _NQUAD_TAIL_REL ×
#   |result| their K moves out and the region is integrated again; with
#   unit difference rows that are edges the same certificate is built on
#   first need for this check (``tail_in``, ``_inner_certificate``).
#   Without a certificate the legacy cap ±200 is used and counted
#   (``nquad_hardened_uncertified``).
# * Tolerance: epsrel = NQUAD_EPSREL and epsabs = NQUAD_EPSABS_FACTOR × a
#   scale, at every level.  First pass: the scale is the smaller of B, a
#   rigorous bound of |integrand| on the region from the modes
#   (``_NquadModes.bound``), and the largest |integrand| at
#   _NQUAD_PRESAMPLES points of the region (from the modes when known; a
#   lower bound of the supremum), but not below _NQUAD_SCALE_FLOOR × B (a
#   sample spread over long truncated sides can miss the integrand's peak
#   by many orders).  Without mode data only the sample (0 -> epsabs 0, a
#   pure relative tolerance).  B takes each edge separately
#   over a box and sums |C| over the modes, so it can exceed the integral
#   by many orders (close poles: residues ±1/ε; decay across the region;
#   oscillation), and so can the sample (oscillation, cancellation); with
#   epsabs above epsrel·|I| QUADPACK stops at its first estimates.  So if
#   the first pass's epsabs exceeds _NQUAD_RERUN_RATIO ×
#   NQUAD_EPSABS_FACTOR × |result|, the region is integrated again with
#   epsabs = NQUAD_EPSABS_FACTOR × |result| (floored at 1e-9 × the sampled
#   maximum, warned when that floor allows more than 1e-8 relative error;
#   up to _NQUAD_RERUN_MAX passes; counter
#   ``nquad_hardened_reruns``).
# * Subinterval limit: NQUAD_LIMIT (at least twice the breakpoints), raised
#   to NQUAD_LIMIT + ω·(U − L)/π when the modes oscillate (ω from
#   ``_NquadModes.osc_bound``) and that count exceeds _NQUAD_OSC_MIN.  A
#   quad call that reaches its limit is repeated once with the limit
#   NQUAD_LIMIT_MAX (``nquad_hardened_quad_retries``; QUADPACK's bisections
#   do not depend on the limit until half of it is used, so the larger
#   limit costs only the subintervals needed); a call of the final pass
#   that still ends with a nonzero ier (roundoff detected, ..., or that
#   limit reached) is counted (``nquad_hardened_quad_flags``) and reported
#   by a PhaseJNquadFallbackWarning (aggregated per model, source and loop
#   order: ``_nquad_note_region``), unless its error
#   estimate meets the requested tolerance or, integrated over the
#   enclosing levels' intervals, stays within NQUAD_EPSREL × the region's
#   scale (an inner call of a second pass asks for 1e-13·|result|
#   absolute, which the rounding of its own values can prevent).
# * Breakpoints: the external times (0 and the free external times), every
#   kink of the inner integral as a function of the variable, and, in every
#   panel wider than _NQUAD_GEOM_MIN_WIDTH/κ_fast (κ_fast: the fastest decay
#   rate), 4^j/κ_fast (j >= 0) from each of its ends that is not a
#   truncation, so no panel next to a peak is much wider than the peak.
#   The kinks at level k (given the outer values) are the s_k-coordinates of
#   the vertices of the region's slice over (s_0, ..., s_k).  Between them
#   the inner integral is smooth (exponentials times polynomials); the
#   narrow peaks that the geometric breakpoints must reach sit at, or a few
#   1/κ_fast from, an interval end or such a kink (for one exponential term
#   the inner integral is log-concave, and its log has corners only there).
#   A kink missing from the breakpoints can leave a whole peak inside a wide
#   panel, unseen by every node.  With difference rows only, a vertex
#   coordinate is an anchor (0 or an outer variable) plus an alternating sum
#   of closure entries along a simple path through inner variables (a run of
#   tight rows in one direction is tight as one closure entry), so these are
#   enumerated from the closure (``_dbm_path_kinks``; the one-intermediate
#   paths are the direct switches of an inner bound).  With other rows the
#   vertices are enumerated at every level call (``_VertexKinks``: every
#   (k+1)-subset of the rows, solved once per region; beyond
#   _NQUAD_VERTEX_MAX_COMBOS subsets that level keeps only the closure's
#   kinks, counted in ``nquad_hardened_kinks_incomplete``).
# * Real and imaginary parts are integrated by separate adaptive passes
#   over one cache of complex values per level, so the inner integrals are
#   computed once per node.  The part that is larger at the centre of the
#   first panel goes first; the other is integrated with epsabs raised to
#   epsrel × |the first part| (the complex value's tolerance), so a part
#   that is only rounding noise (the imaginary part of a real integrand
#   evaluated from pole tuples, or the real part of an imaginary one) does
#   not drive the subdivision.  A part that ends with a QUADPACK message is
#   accepted once its error estimate meets max(epsabs, epsrel × |the
#   level's complex value|).
# * With mode data the innermost variable s_0 is integrated in closed form
#   (``_NquadModes.integrate_innermost``): every row that involves s_0 bounds
#   it in ``_level_interval`` (s_0 has no inner variable), so on that
#   interval the integrand is a sum of exponentials in s_0, and the filter
#   reduces to the rows without s_0.  Quadrature then runs on m − 1 levels
#   (none for m = 1); without mode data, or if an exponential overflows, s_0
#   is integrated by quad like the other levels.  The expansion (per
#   diagram: Π_e (modes of e) terms over the edges with s_0; grouped: one
#   term per pole tuple) is built once per mode object and evaluated with
#   numpy; beyond _NQUAD_INNERMOST_MAX_TERMS product terms s_0 is integrated
#   by quad (``nquad_hardened_innermost_capped``).

#: epsabs = NQUAD_EPSABS_FACTOR × (scale): first the smaller of a rigorous
#: bound of |integrand| and its largest sampled value, then, if that was
#: larger, the first result's magnitude (a second pass).
NQUAD_EPSABS_FACTOR = 1e-13
#: epsrel of every quad call of the hardened fallback.
NQUAD_EPSREL = 1e-10
#: Base subinterval limit of every quad call (raised with the breakpoints,
#: and from the modes' oscillation count).
NQUAD_LIMIT = 200
#: A quad call that reaches its subinterval limit is repeated once with
#: this limit.
NQUAD_LIMIT_MAX = 12800
#: An open side is truncated at NQUAD_TAIL_K / κ beyond its farthest
#: breakpoint (κ: the certified decay rate along it, κ_min for unit
#: difference rows that are edges).
NQUAD_TAIL_K = 40.0
#: Cap of an open side without a decay certificate (the legacy ±200).
NQUAD_UNCERTIFIED_CAP = 200.0
# Geometric breakpoints at RATIO^j / κ_fast (j = 0, 1, ...) from both ends of
# every panel (between the interval's ends and its other breakpoints) wider
# than _NQUAD_GEOM_MIN_WIDTH / κ_fast.  Below that width the adaptive rule
# sees a peak at a panel end: its outermost Gauss-Kronrod node lies 0.0022
# panel widths from the end, where an exp(−κ·x) peak still has e^(−0.56) of
# its height (a panel is accepted unseen only beyond ~1e4 / κ).
_NQUAD_GEOM_RATIO = 4.0
_NQUAD_GEOM_J0 = 0
_NQUAD_GEOM_MAX = 40
_NQUAD_GEOM_MIN_WIDTH = 256.0
# With mode data, the innermost variable is integrated in closed form
# (``_NquadModes.integrate_innermost``) over its exact interval instead of by
# a quad call; False forces quadrature on every level (tests compare both).
_NQUAD_ANALYTIC_INNERMOST = True
# Regions with a row that is not a difference row: the Fourier-Motzkin
# elimination stops beyond this many rows (``_fm_level_rows``), and a level's
# vertex enumeration beyond this many (k+1)-subsets of rows
# (``_VertexKinks``); both are counted.
_NQUAD_FM_MAX_ROWS = 4096
_NQUAD_VERTEX_MAX_COMBOS = 20000
# Points of the region at which |integrand| is sampled for the absolute
# tolerance (from the modes when they are known, else the integrand).
_NQUAD_PRESAMPLES = 64
# With mode data the first pass's scale (the smaller of the bound and the
# sampled maximum) is never below this fraction of the bound.
_NQUAD_SCALE_FLOOR = 1e-3
# Second pass: when the first pass's epsabs exceeds _NQUAD_RERUN_RATIO ×
# NQUAD_EPSABS_FACTOR × |result|, the region is integrated again with
# epsabs = NQUAD_EPSABS_FACTOR × |result| (at most _NQUAD_RERUN_MAX more
# passes, each with the latest result).
_NQUAD_RERUN_RATIO = 10.0
# After a pass that cut the outermost open side: the rigorous bound of the
# dropped tail must be at most _NQUAD_TAIL_REL × |result| (an order below
# NQUAD_EPSREL), else the cut moves out (``tail_k``) and the strip between
# the two cuts is integrated and added; _NQUAD_TAIL_K_MAX caps the widened
# constant (e^{-700} is near the smallest double).
_NQUAD_TAIL_REL = 1e-11
_NQUAD_TAIL_K_MAX = 700.0
# An open side is cut K/κ beyond its farthest breakpoint (kink or external
# time), not beyond its finite end; False measures from the finite end
# (tests compare both).
_NQUAD_TAIL_ANCHOR = True
_NQUAD_RERUN_MAX = 2
# The closed-form innermost integral expands a per-diagram integrand into
# Π_e (modes of edge e) terms over the edges that involve s_0; beyond this
# many terms s_0 is integrated by quad instead (counted).
_NQUAD_INNERMOST_MAX_TERMS = 1 << 16
# The closed form is evaluated by a scalar loop below this many terms per
# edge with s_0 (per diagram) or half as many terms (grouped): numpy loses
# to it there (measured crossovers: 2 edges x 8 modes, 3 x 4 vs 3 x 8; 16 vs
# 64 pole tuples).
_NQUAD_INNERMOST_NUMPY_MIN = 64
# The starting subinterval limit of a quad call is raised to NQUAD_LIMIT +
# the oscillation count ω·(U − L)/π when that count exceeds this.
_NQUAD_OSC_MIN = 100


class PhaseJNquadFallbackWarning(UserWarning):
    """Phase J integrated regions with scipy quadrature (the hardened
    fallback of ``_integrate_polytope``) because no analytic path answered
    them.  Issued once per model, source (per-diagram or grouped) and loop
    order per process (``_nquad_note_region``): the model is its name
    (``_model_identity``), so another ``compute_cumulants`` call or other
    parameter values do not repeat it, while another model does.  Within
    one evaluation (``nquad_warning_scope``, opened by the callables that
    ``compute_cumulants`` returns and around its own grid evaluation) the
    regions are aggregated, and the warning gives their counts: regions and
    diagrams served, regions without mode data, without a decay
    certificate, and (a second warning, also once per model, source and
    loop order) regions whose quadrature did not reach its tolerance
    (``nquad_hardened_quad_flags``).  Every region is counted in
    ``_RUNTIME_COUNTERS['nquad_hardened_calls']``; with logging at DEBUG
    the module's logger names each diagram once.
    """


_NQUAD_WARNED = set()
_NQUAD_WARNED_LOCK = _threading.Lock()
_NQUAD_MODEL_SERIAL = _itertools_hook.count(1)


def _diagram_identity(diag_meta):
    """An identity of the diagram behind ``diag_meta`` that does not change
    when the same model is built again (the warnings count distinct
    diagrams by it): the model (``diag_meta['model']``, see
    ``_model_identity``), the source, the loop order and, per typed
    diagram, its vertex count, its edges (endpoints, label and the two
    field legs), its external legs and the legs and bigrade of every vertex
    type (not the coefficients, which carry parameter values).  Without a
    typed diagram, the build's ``diagram_serial``."""
    model = diag_meta.get('model')
    td = diag_meta.get('diagram')
    tds = td if isinstance(td, (list, tuple)) else (td,)
    sig = []
    try:
        for x in tds:
            if x is None:
                continue
            verts = []
            for v, a in x.vertex_assignments.items():
                verts.append(repr((v, type(a).__name__,
                                   tuple(getattr(a, 'response_legs', ())),
                                   tuple(getattr(a, 'physical_legs', ())),
                                   tuple(getattr(a, 'bigrade', ())))))
            sig.append((len(x.prediagram[0].vertices()),
                        tuple(sorted(repr(e) for e in x.edge_types.items())),
                        tuple(sorted(repr(e)
                                     for e in x.external_legs.items())),
                        tuple(sorted(verts))))
    except Exception:                                   # noqa: BLE001
        sig = []
    if not sig:
        return (model, diag_meta.get('source'), 'serial',
                diag_meta.get('diagram_serial'))
    return (model, diag_meta.get('source'), diag_meta.get('loop_number'),
            tuple(sig))


def _model_identity(propagator_data):
    """The model part of the warnings' key (``_nquad_group_key``) for a
    build with this ``propagator_data``: its ``'model_name'`` (set by
    ``compute_cumulants`` from the model's ``name``), so that rebuilding the
    same model with other parameter values does not repeat a warning; for
    an unnamed model or a direct caller, a serial stamped into that
    ``propagator_data`` dict on first use (each call of ``compute_cumulants``
    builds a new one, so it warns again)."""
    try:
        name = propagator_data.get('model_name')
    except AttributeError:
        return ('unnamed', None)
    if isinstance(name, str) and name:
        return ('model', name)
    try:
        s = propagator_data.get('_nquad_warn_serial')
        if s is None:
            s = next(_NQUAD_MODEL_SERIAL)
            propagator_data['_nquad_warn_serial'] = s
    except (AttributeError, TypeError):
        s = None
    return ('unnamed', s)


def _describe_diagram(diag_meta, m):
    """A short human-readable name of the diagram behind ``diag_meta`` (the
    dispatch's hook metadata), for warnings."""
    if not diag_meta:
        return f'an integration region with m = {m} (no diagram information)'
    src = diag_meta.get('source', '?')
    loop = diag_meta.get('loop_number', '?')
    sid = diag_meta.get('subset_id')
    td = diag_meta.get('diagram')
    desc = ''
    try:
        if isinstance(td, (list, tuple)):
            desc = f'; a group of {len(td)} typed diagrams'
            td = td[0] if td else None
        if td is not None:
            D = td.prediagram[0]
            legs = sorted(f'{f}{p}' for (f, p) in td.external_legs.values())
            edges = sorted(f'{r[0]}{r[1]}<-{q[0]}{q[1]}'
                           for (r, q) in td.edge_types.values())
            desc += (f'; {len(D.vertices())} vertices, external legs '
                     f'{legs}, edges {edges}')
    except Exception:                                   # noqa: BLE001
        pass
    sid_s = bin(sid) if isinstance(sid, int) else str(sid)
    return (f'δ-subset {sid_s} (m = {m}) of a {src} diagram '
            f'(loop order {loop}{desc})')


# Aggregation of the fallback's warnings.  ``_nquad_note_region`` records
# every region served (after its quadrature) under the key (model, source,
# loop order); at the end of the outermost ``nquad_warning_scope`` (or at
# once, outside any scope or in another process than the scope's, e.g. a
# forked worker) ``_nquad_flush`` issues one PhaseJNquadFallbackWarning
# per key not yet warned in this process, and one more for its flagged
# regions.
_NQUAD_PENDING = {}
_NQUAD_SCOPE = {'depth': 0, 'pid': None}
_NQUAD_LOGGED = set()
_NQUAD_LOG = None


def _nquad_logger():
    global _NQUAD_LOG
    if _NQUAD_LOG is None:
        import logging
        _NQUAD_LOG = logging.getLogger(__name__)
    return _NQUAD_LOG


class nquad_warning_scope:
    """Context manager: the hardened fallback's warnings for every region
    served inside it are aggregated and issued when the outermost scope of
    this process ends (``_nquad_flush``).  Reentrant; a forked child that
    inherits an open scope warns at once."""

    def __enter__(self):
        pid = _os.getpid()
        with _NQUAD_WARNED_LOCK:
            if _NQUAD_SCOPE['pid'] != pid:
                _NQUAD_SCOPE['pid'] = pid
                _NQUAD_SCOPE['depth'] = 0
            _NQUAD_SCOPE['depth'] += 1
        return self

    def __exit__(self, *exc):
        with _NQUAD_WARNED_LOCK:
            if _NQUAD_SCOPE['pid'] == _os.getpid():
                _NQUAD_SCOPE['depth'] = max(0, _NQUAD_SCOPE['depth'] - 1)
            done = _NQUAD_SCOPE['depth'] == 0
        if done:
            _nquad_flush()
        return False


def _nquad_group_key(diag_meta):
    if not diag_meta:
        return (None, 'direct', None)
    return (diag_meta.get('model'), diag_meta.get('source'),
            diag_meta.get('loop_number'))


def _nquad_note_region(diag_meta, m, *, no_modes=False, uncertified=False,
                       flag_msgs=()):
    """Record one region served by the hardened fallback (see
    ``nquad_warning_scope``)."""
    key = _nquad_group_key(diag_meta)
    ident = (_diagram_identity(diag_meta) if diag_meta
             else ('direct', m))
    with _NQUAD_WARNED_LOCK:
        st = _NQUAD_PENDING.get(key)
        if st is None:
            st = {'diagrams': set(), 'regions': 0, 'no_modes': 0,
                  'uncertified': 0, 'flagged': 0, 'flag_calls': 0,
                  'first': (diag_meta, m), 'first_flag': None}
            _NQUAD_PENDING[key] = st
        st['diagrams'].add(ident)
        st['regions'] += 1
        st['no_modes'] += bool(no_modes)
        st['uncertified'] += bool(uncertified)
        if flag_msgs:
            st['flagged'] += 1
            st['flag_calls'] += len(flag_msgs)
            if st['first_flag'] is None:
                st['first_flag'] = (diag_meta, m, flag_msgs[0])
        new_ident = ident not in _NQUAD_LOGGED
        if new_ident:
            _NQUAD_LOGGED.add(ident)
        now = (_NQUAD_SCOPE['depth'] == 0
               or _NQUAD_SCOPE['pid'] != _os.getpid())
    if new_ident:
        log = _nquad_logger()
        if log.isEnabledFor(10):                        # logging.DEBUG
            log.debug('Phase J hardened quadrature fallback: %s',
                      _describe_diagram(diag_meta, m))
    if now:
        _nquad_flush()


def _nquad_whose(key):
    model, src, loop = key
    if isinstance(model, tuple) and model[:1] == ('model',):
        who = f"model '{model[1]}'"
    elif isinstance(model, tuple) and model[:1] == ('unnamed',):
        who = 'an unnamed model'
    elif src == 'direct':
        return 'a direct call (no diagram information)'
    else:
        who = 'an unidentified model'
    if src == 'direct':
        return who
    return f'{who} ({src}, loop order {loop})'


def _nquad_flush():
    """Issue the aggregated warnings of ``_NQUAD_PENDING`` (once per key and
    process) and clear it."""
    import warnings
    with _NQUAD_WARNED_LOCK:
        items = list(_NQUAD_PENDING.items())
        _NQUAD_PENDING.clear()
        todo = []
        for key, st in items:
            kf = ('fallback',) + key
            if kf not in _NQUAD_WARNED:
                _NQUAD_WARNED.add(kf)
                todo.append(('fallback', key, st))
            kq = ('flags',) + key
            if st['flagged'] and kq not in _NQUAD_WARNED:
                _NQUAD_WARNED.add(kq)
                todo.append(('flags', key, st))
    for kind, key, st in todo:
        who = _nquad_whose(key)
        if kind == 'fallback':
            extra = ''
            if st['no_modes']:
                extra += (f'; {st["no_modes"]} without the integrand\'s '
                          'exponential modes (epsabs from a sample of the '
                          'integrand, open directions capped at the legacy '
                          f'{NQUAD_UNCERTIFIED_CAP:g}, no geometric '
                          'breakpoints)')
            if st['uncertified']:
                extra += (f'; {st["uncertified"]} with an open direction '
                          'without a decay certificate (capped at the '
                          f'legacy {NQUAD_UNCERTIFIED_CAP:g})')
            dm, m = st['first']
            msg = (
                f'Phase J: no analytic path answered {st["regions"]} '
                f'integration region(s) of {len(st["diagrams"])} '
                f'diagram(s) of {who} in this evaluation; they were '
                'integrated by the hardened scipy quadrature fallback '
                f'(exact region bounds, epsrel {NQUAD_EPSREL:g}, open '
                f'directions truncated at {NQUAD_TAIL_K:g}/κ beyond their '
                f'farthest breakpoint{extra}).  First: '
                f'{_describe_diagram(dm, m)}.  Issued once per model, '
                'source and loop order and process; every region is '
                "counted in final_integral._RUNTIME_COUNTERS"
                "['nquad_hardened_calls'] (logging at DEBUG names each "
                'diagram).')
        else:
            dm, m, first = st['first_flag']
            msg = (
                'Phase J: the hardened quadrature fallback did not reach '
                f'its tolerance on {st["flagged"]} of {st["regions"]} '
                f'region(s) of {who} in this evaluation: '
                f'{st["flag_calls"]} scipy.integrate.quad call(s) of their '
                'final passes ended with a QUADPACK warning (first, on '
                f'{_describe_diagram(dm, m)}: "{first}"); the values are '
                'used as computed and may be inaccurate.  Issued once per '
                'model, source and loop order and process; every such '
                "call is counted in final_integral._RUNTIME_COUNTERS"
                "['nquad_hardened_quad_flags'].")
        warnings.warn(msg, PhaseJNquadFallbackWarning, stacklevel=2)


import cmath


def _exp_over_interval(o, al, L, U):
    """∫_L^U exp(o + al·s) ds for complex o, al (the scalar path of
    ``_NquadModes.integrate_innermost``)."""
    h = U - L
    z = al * h
    if abs(z) < 0.25:
        phi = 1.0 + 0.0j
        term = 1.0 + 0.0j
        for n in range(2, 20):
            term *= z / n
            phi += term
        return cmath.exp(o + al * L) * h * phi
    return (cmath.exp(o + al * U) - cmath.exp(o + al * L)) / al


def _sum_exp_over_interval(K, EU, EL, AL, L, U):
    r"""Σ_j K_j ∫_L^U exp(o_j + α_j s) ds from EU_j = exp(o_j + α_j U),
    EL_j = exp(o_j + α_j L) and AL_j = α_j (complex arrays): each integral
    is (EU_j − EL_j)/α_j, or EL_j·(U − L)·φ1(α_j (U − L)) with
    φ1(z) = (e^z − 1)/z by its series where |α_j (U − L)| < 1/4."""
    h = U - L
    z = AL * h
    small = np.abs(z) < 0.25
    if small.all():
        out = EL * h * _phi1_series(z)
    elif small.any():
        out = np.empty(EU.shape, complex)
        big = ~small
        out[big] = (EU[big] - EL[big]) / AL[big]
        out[small] = EL[small] * h * _phi1_series(z[small])
    else:
        out = (EU - EL) / AL
    return complex(np.dot(K, out))


def _phi1_series(zs):
    """φ1(z) = (e^z − 1)/z = Σ_{n>=0} z^n/(n+1)! for |z| < 1/4 (an array;
    in place, the same operations as the scalar series)."""
    phi = np.ones(zs.shape, complex)
    term = np.ones(zs.shape, complex)
    for n in range(2, 20):
        np.multiply(term, zs, out=term)
        np.divide(term, n, out=term)
        np.add(phi, term, out=phi)
    return phi


class _NquadModes:
    r"""The modes of a Phase J subset integrand, for the hardened fallback.

    ``kind='product'`` (per-diagram fast evaluators): the integrand is
    ``pref · Π_e Σ_k C_ek exp(λ_ek Δt_e)``; ``cterms[e]`` is the tuple of
    ``(C_ek, λ_ek)``.  ``kind='sum'`` (grouped merged pole tuples): the
    integrand is ``pref · Σ_α B_α Π_e exp(λ_αe Δt_e)``; ``cterms`` is the
    tuple of ``(B_α, (λ_αe for each edge e))``.  ``edges[e]`` is the edge's
    row ``Δt_e = c0 + Σ a_int·s + Σ a_ext·t`` in the sparse form
    ``(int_pairs, ext_pairs, c0)``.  ``terms`` keeps the magnitudes and real
    parts (|C|, Re λ) that the tolerance and the truncation need.  The
    expansion of the integrand in the innermost variable s_0 (arrays for
    ``integrate_innermost``) is built once per object, on first use
    (``_expansion0``), and so are the oscillation rate (``osc_bound``) and
    the decay rates (``decay_rates``).
    """
    __slots__ = ('kind', 'pref', 'pref_abs', 'edges', 'cterms', 'terms',
                 '_x0', '_osc', '_rates')

    def __init__(self, kind, pref, edges, cterms):
        self.kind = kind
        self.pref = complex(pref)
        self.pref_abs = abs(self.pref)
        self.edges = edges
        self.cterms = cterms
        self._x0 = None
        self._osc = None
        self._rates = None
        if kind == 'product':
            self.terms = tuple(tuple((abs(C), lam.real) for (C, lam) in per)
                               for per in cterms)
        else:
            self.terms = tuple((abs(B), tuple(lam.real for lam in lams))
                               for (B, lams) in cterms)

    @classmethod
    def from_edge_data(cls, pref_c, edge_data):
        """From a fast evaluator's ``(poles, residues, c0, int_pairs,
        ext_pairs)`` tuples (``G_e(t) = Σ r exp(i p t)``: λ = i p).  ``None``
        if the data are not numeric."""
        try:
            edges, cterms = [], []
            for (poles, residues, c0, int_pairs, ext_pairs) in edge_data:
                edges.append((tuple(int_pairs), tuple(ext_pairs), float(c0)))
                cterms.append(tuple((complex(r), 1j * complex(p))
                                    for p, r in zip(poles, residues)))
            return cls('product', pref_c, tuple(edges), tuple(cterms))
        except (TypeError, ValueError, OverflowError):
            return None

    @classmethod
    def from_pole_tuples(cls, rows, pole_tuples, pref=1.0):
        """Grouped: ``pole_tuples`` = ``[(B_α, (λ per smooth edge))]``, the
        smooth edges being the first rows of ``rows`` (``(a_int, a_ext,
        c0)``, the subset's constraint data).  ``None`` if unusable."""
        try:
            pt = list(pole_tuples or ())
            if not pt:
                return None
            n = len(pt[0][1])
            if n > len(rows):
                return None
            edges = []
            for (a_int, a_ext, c0) in list(rows)[:n]:
                ip = tuple((i, float(a)) for i, a in enumerate(a_int)
                           if abs(float(a)) > 1e-15)
                ep = tuple((i, float(a)) for i, a in enumerate(a_ext)
                           if abs(float(a)) > 1e-15)
                edges.append((ip, ep, float(c0)))
            cterms = tuple((complex(B), tuple(complex(lam) for lam in lams))
                           for (B, lams) in pt)
            if any(len(lams) != n for (_b, lams) in cterms):
                return None
            return cls('sum', pref, tuple(edges), cterms)
        except (TypeError, ValueError, IndexError, OverflowError):
            return None

    def _expansion0(self):
        r"""The integrand as a sum of exponentials in s_0, built once.

        Per edge: ``(a0, c0, ext_pairs, outer_pairs)`` with
        Δt_e = a0·s_0 + c_e, c_e = c0 + Σ a·free + Σ a·s_{i} (i >= 1;
        ``outer_pairs`` index ``outer`` = (s_1, ...)).

        * ``'product'``: the edges with a0 != 0 expand into
          n = Π_e (number of modes of e) terms: K = ⊗_e (C_ek)_k (Kronecker
          product) and AL = the matching sums Σ_e a0_e·λ_ek; per call the
          exponentials are formed per edge and combined by Kronecker
          products.  ``('capped', n)`` if n > _NQUAD_INNERMOST_MAX_TERMS
          (the caller then integrates s_0 by quad).
        * ``'sum'``: B (one entry per pole tuple), LAM (tuples × edges) and
          AL = LAM @ a0.
        """
        x = self._x0
        if x is not None:
            return x
        edges_c = []
        for (ip, ep, c0) in self.edges:
            a0 = 0.0
            op = []
            for (i, a) in ip:
                if i == 0:
                    a0 = a
                else:
                    op.append((i - 1, a))
            edges_c.append((a0, c0, tuple(ep), tuple(op)))
        edges_c = tuple(edges_c)
        if self.kind == 'product':
            s0 = tuple(e for e, ec in enumerate(edges_c) if ec[0] != 0.0)
            rest = tuple(e for e, ec in enumerate(edges_c) if ec[0] == 0.0)
            n = 1
            for e in s0:
                n *= len(self.cterms[e])
            if n > _NQUAD_INNERMOST_MAX_TERMS:
                x = ('capped', n)
            else:
                K = np.ones(1, complex)
                AL = np.zeros(1, complex)
                lams = []
                for e in s0:
                    Ce = np.array([C for (C, _l) in self.cterms[e]], complex)
                    Le = np.array([lam for (_C, lam) in self.cterms[e]],
                                  complex)
                    K = np.kron(K, Ce)
                    AL = (AL[:, None] + (Le * edges_c[e][0])[None, :]).ravel()
                    lams.append(Le)
                x = ('product', n, edges_c, s0, rest, K, AL, tuple(lams))
        else:
            ne = len(edges_c)
            a0v = np.array([ec[0] for ec in edges_c], float)
            B = np.array([b for (b, _l) in self.cterms], complex)
            LAM = np.array([list(lams) for (_b, lams) in self.cterms],
                           complex).reshape(len(B), ne)
            x = ('sum', len(B), edges_c, B, LAM, LAM @ a0v)
        self._x0 = x
        return x

    def innermost_terms(self):
        """The number of exponential terms of the closed-form innermost
        integral (``integrate_innermost`` is available only when it is at
        most _NQUAD_INNERMOST_MAX_TERMS, i.e. not ``'capped'``)."""
        x = self._expansion0()
        return x[1]

    def innermost_available(self):
        return self._expansion0()[0] != 'capped'

    @staticmethod
    def _cval(ec, outer, free):
        _a0, c, ep, op = ec
        for (i, a) in ep:
            c += a * free[i]
        for (i, a) in op:
            c += a * outer[i]
        return c

    def integrate_innermost(self, L, U, outer, free):
        r"""∫_L^U integrand(s_0, *outer, *free) ds_0 in closed form (s_0 the
        innermost variable, ``outer`` = (s_1, ..., s_{m-1})), with NO
        Heaviside factor: the caller passes the exact interval of s_0 and
        checks the other rows.  Each product of modes is an exponential
        exp(o + α s_0); its integral is exp(o + α L)·(U − L)·φ1(α(U − L)),
        φ1(z) = (e^z − 1)/z (series for |z| < 1/4, else the difference of
        the two exponentials over α).  The expansion is built once per
        object (``_expansion0``) and evaluated with numpy; per call the
        exponentials are formed per edge (product kind: Kronecker products
        of the per-edge factors exp(λ_ek Δt_e) at s_0 = L and U) or per
        pole tuple (sum kind).  Below _NQUAD_INNERMOST_NUMPY_MIN terms per
        edge with s_0 (product kind) or _NQUAD_INNERMOST_NUMPY_MIN / 2 terms
        (sum kind) the scalar loop is faster (numpy's per-call and per-edge
        overhead) and is used instead (``_integrate_innermost_scalar``).
        Raises OverflowError if
        an exponential overflows (the caller then integrates
        numerically)."""
        x = self._expansion0()
        kind = x[0]
        if (kind == 'product' and x[1] < _NQUAD_INNERMOST_NUMPY_MIN
                * max(1, len(x[3]))) or (
                    kind == 'sum' and 2 * x[1] < _NQUAD_INNERMOST_NUMPY_MIN):
            return self._integrate_innermost_scalar(L, U, outer, free)
        cval = self._cval
        try:
            with np.errstate(over='raise', invalid='raise'):
                if kind == 'product':
                    (_k, _n, edges_c, s0, rest, K, AL, lams) = x
                    val = self.pref
                    for e in rest:
                        c = cval(edges_c[e], outer, free)
                        g = 0.0j
                        for (C, lam) in self.cterms[e]:
                            g += C * cmath.exp(lam * c)
                        val *= g
                    if not s0:
                        return val * (U - L)
                    EU = EL = None
                    for e, Le in zip(s0, lams):
                        ec = edges_c[e]
                        c = cval(ec, outer, free)
                        eu = np.exp(Le * (ec[0] * U + c))
                        el = np.exp(Le * (ec[0] * L + c))
                        if EU is None:
                            EU, EL = eu, el
                        else:
                            # == np.kron for 1-D arrays, bit for bit, without
                            # its reshaping overhead
                            EU = np.multiply.outer(EU, eu).ravel()
                            EL = np.multiply.outer(EL, el).ravel()
                    return val * _sum_exp_over_interval(K, EU, EL, AL, L, U)
                if kind == 'sum':
                    (_k, _n, edges_c, B, LAM, AL) = x
                    a0v = np.array([ec[0] for ec in edges_c], float)
                    cv = np.array([cval(ec, outer, free) for ec in edges_c],
                                  float)
                    EU = np.exp(LAM @ (a0v * U + cv))
                    EL = np.exp(LAM @ (a0v * L + cv))
                    return self.pref * _sum_exp_over_interval(B, EU, EL, AL,
                                                              L, U)
        except FloatingPointError as exc:
            raise OverflowError(str(exc)) from exc
        raise ValueError('closed-form innermost integral not available '
                         f'({x[1]} terms)')

    def _integrate_innermost_scalar(self, L, U, outer, free):
        """``integrate_innermost`` by a scalar loop over the expanded terms
        (cmath; for few terms)."""
        if self.kind == 'product':
            val = self.pref
            terms = [(1.0 + 0.0j, 0.0j, 0.0j)]
            for (ip, ep, c0), per in zip(self.edges, self.cterms):
                c = c0
                for (i, a) in ep:
                    c += a * free[i]
                a0 = 0.0
                for (i, a) in ip:
                    if i == 0:
                        a0 = a
                    else:
                        c += a * outer[i - 1]
                if a0 == 0.0:
                    g = 0.0j
                    for (C, lam) in per:
                        g += C * cmath.exp(lam * c)
                    val *= g
                else:
                    terms = [(k * C, o + lam * c, al + lam * a0)
                             for (k, o, al) in terms for (C, lam) in per]
            tot = 0.0j
            for (k, o, al) in terms:
                tot += k * _exp_over_interval(o, al, L, U)
            return val * tot
        cs, a0s = [], []
        for (ip, ep, c0) in self.edges:
            c = c0
            for (i, a) in ep:
                c += a * free[i]
            a0 = 0.0
            for (i, a) in ip:
                if i == 0:
                    a0 = a
                else:
                    c += a * outer[i - 1]
            cs.append(c)
            a0s.append(a0)
        tot = 0.0j
        for (B, lams) in self.cterms:
            o = al = 0.0j
            for lam, c, a0 in zip(lams, cs, a0s):
                o += lam * c
                al += lam * a0
            tot += B * _exp_over_interval(o, al, L, U)
        return self.pref * tot

    def value(self, s, free):
        """The integrand at ``s`` = (s_0, ..., s_{m-1}) from the modes (no
        Heaviside factor; used for the tolerance's sample of |integrand|).
        May raise OverflowError."""
        if self.kind == 'product':
            val = self.pref
            for (ip, ep, c0), per in zip(self.edges, self.cterms):
                dt = c0
                for (i, a) in ip:
                    dt += a * s[i]
                for (i, a) in ep:
                    dt += a * free[i]
                g = 0.0j
                for (C, lam) in per:
                    g += C * cmath.exp(lam * dt)
                val *= g
            return val
        x = self._expansion0()
        (_k, _n, edges_c, B, LAM, _AL) = x
        dts = []
        for (ip, ep, c0) in self.edges:
            dt = c0
            for (i, a) in ip:
                dt += a * s[i]
            for (i, a) in ep:
                dt += a * free[i]
            dts.append(dt)
        try:
            with np.errstate(over='raise', invalid='raise'):
                return self.pref * complex(np.dot(
                    B, np.exp(LAM @ np.array(dts, float))))
        except FloatingPointError as exc:
            raise OverflowError(str(exc)) from exc

    def osc_bound(self):
        """An estimate (upper bound for unit-coefficient rows) of the
        angular frequency of the integrand, and of every nested inner
        integral, in any integration variable: max over the products of
        modes of Σ_e ‖a_e‖₁·|Im λ_e| (‖a_e‖₁: the sum of |coefficients| of
        the integration variables in edge e's row; an inner bound that
        depends on s_k carries the inner variables' frequencies into the
        function of s_k).  0.0 when nothing oscillates.  Used for the
        starting subinterval limit of the quad calls."""
        w = self._osc
        if w is not None:
            return w
        norms = [sum(abs(a) for (_i, a) in ip) for (ip, _ep, _c0)
                 in self.edges]
        if self.kind == 'product':
            w = 0.0
            for nrm, per in zip(norms, self.cterms):
                if nrm and per:
                    w += nrm * max(abs(lam.imag) for (_C, lam) in per)
        else:
            LAM = self._expansion0()[4]
            w = (float(np.max(np.abs(LAM.imag) @ np.array(norms, float)))
                 if LAM.size else 0.0)
        if not math.isfinite(w):
            w = 0.0
        self._osc = w
        return w

    def sup_bound(self):
        """|pref|·Π_e Σ_k |C_ek| (product) or |pref|·Σ_α |B_α| (sum): a
        bound of |integrand| wherever every Δt_e >= 0, when no mode grows
        (the decay certificate); the S of the truncation's tail bound."""
        if self.kind == 'product':
            s = self.pref_abs
            for per in self.terms:
                s *= sum(c for (c, _re) in per)
            return s
        return self.pref_abs * sum(b for (b, _res) in self.terms)

    def min_coefficient(self):
        """The smallest |coefficient| of an integration variable in the
        edges' rows (1.0 if none)."""
        a = [abs(x) for (ip, _ep, _c0) in self.edges for (_i, x) in ip
             if x != 0.0]
        return min(a) if a else 1.0

    def _modes(self):
        if self.kind == 'product':
            for per_edge in self.terms:
                for (c, re) in per_edge:
                    yield c, re
        else:
            for (b, res) in self.terms:
                for re in res:
                    yield b, re

    def decay_rates(self):
        """``(κ_min, κ_fast)``: the smallest and the largest decay rate
        −Re λ over the modes with a nonzero coefficient (κ_min <= 0 when
        some such mode does not decay); ``(None, None)`` without any such
        mode or with a non-finite one.  Computed once per object."""
        r = self._rates
        if r is not None:
            return r
        kmin = kfast = None
        for c, re in self._modes():
            if c == 0.0:
                continue
            if not (math.isfinite(c) and math.isfinite(re)):
                kmin = kfast = None
                break
            k = -re
            if kmin is None or k < kmin:
                kmin = k
            if kfast is None or abs(re) > kfast:
                kfast = abs(re)
        self._rates = (kmin, kfast)
        return self._rates

    def edge_decay(self, free):
        """For the linear-programming decay certificate
        (``_GeneralDecayCertificate``): ``(S, edges)`` with ``edges`` =
        ``[(a, c, κ_e)]`` for every edge that involves an integration
        variable -- its coefficients ``a`` (a tuple of ``(j, a_j)``), its
        shift ``c`` at the free times and its slowest decay rate κ_e (the
        smallest −Re λ over its modes with a nonzero coefficient, per
        diagram; over the pole tuples with a nonzero B, grouped) -- and S
        with |integrand| <= S·exp(−Σ_e κ_e·Δt_e) wherever every such
        Δt_e >= 0: |pref|·Π_e Σ_k |C_ek| over those edges times the
        constant edges' bounds (per diagram), or |pref|·Σ_α |B_α|·exp(the
        constant edges' Re λ_αe·Δt_e) (grouped).  ``None`` if a rate or a
        coefficient is not finite."""
        ints, consts = [], []
        for e, (ip, ep, c0) in enumerate(self.edges):
            c = c0 + sum(a * free[i] for i, a in ep)
            prs = tuple(sorted((i, a) for (i, a) in ip if abs(a) > 1e-15))
            (ints if prs else consts).append((e, prs, c))
        try:
            if self.kind == 'product':
                S = self.pref_abs
                out = []
                for (e, prs, c) in ints:
                    per = [(cc, re) for (cc, re) in self.terms[e]
                           if cc != 0.0]
                    if not per:
                        return 0.0, []
                    out.append((prs, c, min(-re for (_cc, re) in per)))
                    S *= sum(cc for (cc, _re) in per)
                for (e, _prs, c) in consts:
                    S *= sum(cc * math.exp(re * c)
                             for (cc, re) in self.terms[e] if cc != 0.0)
            else:
                rows = [(b, res) for (b, res) in self.terms if b != 0.0]
                if not rows:
                    return 0.0, []
                out = [(prs, c, min(-res[e] for (_b, res) in rows))
                       for (e, prs, c) in ints]
                S = self.pref_abs * sum(
                    b * math.exp(sum(res[e] * c for (e, _p, c) in consts))
                    for (b, res) in rows)
        except OverflowError:
            return None
        if not (math.isfinite(S) and all(math.isfinite(k)
                                         for (_a, _c, k) in out)):
            return None
        return S, out

    def bound(self, lo, hi, free):
        r"""A rigorous upper bound of |integrand| on the region, which lies
        in the box [lo, hi].  Per edge, Δt_e takes values in
        [max(0, min_box Δt_e), max_box Δt_e] (its row is a constraint of the
        region: Δt_e > 0), and each exp(Re λ Δt) is bounded at the end of
        that range that maximises it (Re λ < 0: the lower end, so the
        factor is <= 1).  An edge whose Δt_e is <= 0 on the whole box is not
        a constraint there -- a constant row at an exact tie that the
        external-time order kept (``_const_row_verdict``: DROP, e.g.
        Δt_e ≡ 0, where the edge contributes G(0)) -- and keeps its box
        range unclipped.  0.0 only for an identically zero integrand."""
        ranges = []
        for (ip, ep, c0) in self.edges:
            c = c0 + sum(a * free[i] for i, a in ep)
            dmin = dmax = c
            for (i, a) in ip:
                if a > 0.0:
                    dmin += a * lo[i]
                    dmax += a * hi[i]
                else:
                    dmin += a * hi[i]
                    dmax += a * lo[i]
            ranges.append((max(dmin, 0.0) if dmax > 0.0 else dmin, dmax))

        def ex(x):
            try:
                return math.exp(x)
            except OverflowError:
                return math.inf

        if self.pref_abs == 0.0:
            return 0.0
        if self.kind == 'product':
            B = self.pref_abs
            for (dmin, dmax), per_edge in zip(ranges, self.terms):
                s = 0.0
                for (c, re) in per_edge:
                    if c == 0.0:
                        continue
                    if re < 0.0:
                        s += c * ex(re * dmin)
                    elif re > 0.0:
                        s += c * ex(re * dmax)
                    else:
                        s += c
                if s == 0.0:
                    return 0.0
                B *= s
            return B
        B = 0.0
        for (b, res) in self.terms:
            if b == 0.0:
                continue
            e = 0.0
            for (dmin, dmax), re in zip(ranges, res):
                if re < 0.0:
                    e += re * dmin
                elif re > 0.0:
                    e += re * dmax
            B += b * ex(e)
        return self.pref_abs * B


def _frac_up(fr):
    """The smallest float >= the rational ``fr`` (``inf`` beyond range)."""
    try:
        f = float(fr)
    except OverflowError:
        return math.inf if fr > 0 else -math.nextafter(math.inf, 0.0)
    if _Fraction(f) < fr:
        f = math.nextafter(f, math.inf)
    return f


def _dbm_closure(s_constraints, m):
    r"""Exact difference-bound-matrix closure of the resolved rows
    ``(a_int, shift)`` (each ``a_int·s + shift > 0``, as
    ``_integrate_polytope`` receives them).

    Returns ``(D, general, verdict)``:

    * ``D``: (m+1)×(m+1) nested lists, ``D[u][v]`` an exact ``Fraction``
      with ``x_u − x_v < D[u][v]`` on the region of the difference rows, or
      ``None`` (no bound); node ``m`` is the constant 0 (``x_m ≡ 0``), so
      ``D[k][m]`` / ``−D[m][k]`` are the upper / lower bound of ``s_k``.
      After the Floyd-Warshall closure every entry is the shortest path,
      i.e. the exact supremum of x_u − x_v over that region.
    * ``general``: every other non-constant row, ``(shift, ((j, a_j), ...))``
      with floats; they are NOT in ``D``.
    * ``verdict``: ``'EMPTY'`` when the region is empty (a constant row with
      shift <= 0 under Θ(0) = 0, or a cycle of total weight <= 0: strict
      inequalities that cannot all hold), ``'NONFINITE'`` for a shift or
      coefficient that is not a finite float, else ``None``.

    Difference rows: one coefficient |a_j| > 1e-15 (``a·s_j + c > 0``), or two
    with ``a_i == −a_j`` exactly (``a_i·(s_i − s_j) + c > 0``); each gives
    the bound ``c/|a|`` as an exact rational (no rounding anywhere).
    Coefficients with |a| <= 1e-15 count as 0, as in the Heaviside filter.
    """
    n = m + 1
    z = m
    D = [[None] * n for _ in range(n)]
    general = []
    try:
        for (a_int, shift) in s_constraints:
            c = float(shift)
            coefs = [float(a) for a in a_int]
            if not math.isfinite(c) or not all(math.isfinite(a)
                                               for a in coefs):
                return None, None, 'NONFINITE'
            prs = tuple((j, a) for j, a in enumerate(coefs)
                        if abs(a) > 1e-15)
            if not prs:
                if c <= 0.0:
                    return None, None, 'EMPTY'
                continue
            if len(prs) == 1:
                (j, a), = prs
                w = _Fraction(c) / _Fraction(abs(a))
                u, v = (z, j) if a > 0.0 else (j, z)
            elif len(prs) == 2 and prs[0][1] == -prs[1][1]:
                (i, ai), (j, _aj) = prs
                w = _Fraction(c) / _Fraction(abs(ai))
                # ai·(s_i − s_j) + c > 0
                u, v = (j, i) if ai > 0.0 else (i, j)
            else:
                general.append((c, prs))
                continue
            if D[u][v] is None or w < D[u][v]:
                D[u][v] = w
    except (TypeError, ValueError, OverflowError):
        return None, None, 'NONFINITE'
    for k in range(n):
        Dk = D[k]
        for i in range(n):
            dik = D[i][k]
            if dik is None:
                continue
            Di = D[i]
            for j in range(n):
                dkj = Dk[j]
                if dkj is None:
                    continue
                s = dik + dkj
                if Di[j] is None or s < Di[j]:
                    Di[j] = s
    for v in range(n):
        if D[v][v] is not None and D[v][v] <= 0:
            return D, general, 'EMPTY'
    return D, general, None


def _dbm_level_tables(D, m):
    """Per variable k: ``(L0, U0, lo_outer, up_outer)`` -- the closure's
    constant bounds of s_k (floats rounded outward, ±inf if none) and its
    bounds relative to the OUTER variables s_j, j > k: ``lo_outer`` =
    ``((j − k − 1, d), ...)`` for ``s_k > s_j − d``, ``up_outer`` =
    ``((j − k − 1, d), ...)`` for ``s_k < s_j + d`` (d rounded up)."""
    z = m
    tabs = []
    for k in range(m):
        U0 = math.inf if D[k][z] is None else _frac_up(D[k][z])
        L0 = -math.inf if D[z][k] is None else -_frac_up(D[z][k])
        up_o = tuple((j - k - 1, _frac_up(D[k][j]))
                     for j in range(k + 1, m) if D[k][j] is not None)
        lo_o = tuple((j - k - 1, _frac_up(D[j][k]))
                     for j in range(k + 1, m) if D[j][k] is not None)
        tabs.append((L0, U0, lo_o, up_o))
    return tabs


def _dbm_interval(tab, outer):
    """``(L, U)`` of the difference rows' projection onto s_k given the
    outer values ``outer = (s_{k+1}, ..., s_{m-1})`` (``tab`` from
    ``_dbm_level_tables``); exact up to the outward rounding of the
    bounds and of the one addition per candidate."""
    L, U, lo_o, up_o = tab
    for (oi, d) in up_o:
        v = outer[oi] + d
        if v < U:
            U = v
    for (oi, d) in lo_o:
        v = outer[oi] - d
        if v > L:
            L = v
    return L, U


def _general_rows_by_level(general, m):
    """The non-difference rows of ``_dbm_closure``, filed under the level of
    their innermost variable k as ``(a_k, shift, ((j − k − 1, a_j), ...))``
    over the outer variables j > k: at that level they involve no inner
    variable, so they bound s_k exactly given the outer values."""
    gen = [[] for _ in range(m)]
    for (c, prs) in general:
        k = min(j for j, _a in prs)
        a_k = [a for j, a in prs if j == k][0]
        gen[k].append((a_k, c, tuple((j - k - 1, a) for j, a in prs
                                      if j != k)))
    return gen


def _level_interval(tab, gen_k, outer):
    """``(L, U)`` of s_k given the outer values: the difference rows'
    projection (``_dbm_interval``) intersected with the rows of ``gen_k``
    (``_general_rows_by_level``).  Never narrower than the region's slice,
    up to the rounding of one addition / division per candidate (the
    caller widens by one ulp)."""
    L, U = _dbm_interval(tab, outer)
    for (a_k, c, prs) in gen_k:
        tot = c
        for (oi, a) in prs:
            tot += a * outer[oi]
        b = -tot / a_k
        if a_k > 0.0:
            if b > L:
                L = b
        elif b < U:
            U = b
    return L, U


class _FMEmpty(Exception):
    """``_fm_level_rows``: a positive combination of the rows is a constant
    row <= 0, so the region is empty."""


def _fm_level_rows(s_constraints, m, max_rows=None):
    r"""Exact per-level bounds of the region ``{s : a_int·s + shift > 0}``
    by Fourier-Motzkin elimination of s_0, s_1, ... in rational arithmetic.

    The rows of the system after eliminating s_0..s_{k-1} whose innermost
    variable is s_k bound s_k exactly given the outer variables: the
    interval ``_level_interval`` builds from them is the projection of the
    region onto s_k given s_{k+1}, ..., s_{m-1} (each elimination step is
    exact: s_v exists iff every lower bound lies below every upper bound).
    Returns per level a list of ``(a_k, shift, ((j − k − 1, a_j), ...))``
    in the format of ``_general_rows_by_level``, with a_k = ±1 (each row is
    scaled by its innermost |coefficient|; per direction only the tightest
    row is kept).  ``'EMPTY'`` if a positive combination of the rows is a
    constant <= 0 (strict inequalities: the region is empty); ``None`` if
    the system grows beyond ``max_rows`` (default _NQUAD_FM_MAX_ROWS).
    Coefficients with |a| <= 1e-15 count as 0, as in the Heaviside filter.
    The rows are exact; their float form (rounded to nearest) can make an
    interval narrower than the projection by a few ulps, a sliver the
    caller's one-ulp widening may not cover (negligible next to the
    quadrature tolerance).
    """
    if max_rows is None:
        max_rows = _NQUAD_FM_MAX_ROWS
    zero = _Fraction(0)
    sys_ = {}

    def add(a, c):
        for j, x in enumerate(a):
            if x != 0:
                break
        else:
            if c <= 0:
                raise _FMEmpty
            return
        sc = abs(a[j])
        if sc != 1:
            a = tuple(x / sc for x in a)
            c = c / sc
        old = sys_.get(a)
        if old is None or c < old:
            sys_[a] = c

    levels = [[] for _ in range(m)]
    try:
        for (a_int, shift) in s_constraints:
            a = tuple(_Fraction(float(x)) if abs(float(x)) > 1e-15 else zero
                      for x in a_int)
            add(a, _Fraction(float(shift)))
        for v in range(m):
            pos, neg, rest = [], [], {}
            for a, c in sys_.items():
                if a[v] > 0:
                    pos.append((a, c))
                elif a[v] < 0:
                    neg.append((a, c))
                else:
                    rest[a] = c
            for (a, c) in pos + neg:
                levels[v].append((float(a[v]), float(c), tuple(
                    (j - v - 1, float(a[j])) for j in range(v + 1, m)
                    if a[j] != 0)))
            if len(rest) + len(pos) * len(neg) > max_rows:
                return None
            sys_ = rest
            for (ap, cp) in pos:          # ap[v] = +1, an[v] = −1
                for (an, cn) in neg:
                    add(tuple(x + y for x, y in zip(ap, an)), cp + cn)
    except _FMEmpty:
        return 'EMPTY'
    except (TypeError, ValueError, OverflowError):
        return None
    return levels


def _dbm_path_kinks(D, m):
    r"""Kinks of the inner integral at levels k >= 2 from the closed
    difference-bound matrix ``D`` (``_dbm_closure``), beyond the direct
    switches of one inner bound.

    At level k, with the outer variables fixed, every vertex of the slice
    over (s_0, ..., s_k) is cut out by tight rows that connect each of
    these variables to an anchor (the constant node, value 0, or an outer
    variable).  Along the path from s_k to its anchor, two consecutive
    tight rows in the same direction (s_u − s_v = D[u][v] and
    s_v − s_w = D[v][w]) make s_u − s_w = D[u][v] + D[v][w], which the
    closure bounds by D[u][w]: the path is also tight as the single entry
    D[u][w].  So s_k = anchor + an alternating sum of closure entries along
    a simple path through inner variables.  The paths through one inner
    variable are the kinks the caller already has; this returns, per level,
    ``(consts, offsets)`` for the paths through two or more: constants
    (anchor 0) and ``(j − k − 1, offset)`` pairs (anchor s_j, j > k).  A
    superset of the vertex coordinates (not every such path is a vertex);
    floats, deduplicated.
    """
    z = m
    Df = [[None if x is None else float(x) for x in row] for row in D]
    out = [((), ()) for _ in range(m)]
    for k in range(2, m):
        consts, offs = set(), set()
        outer_nodes = tuple(range(k + 1, m))

        def walk(u, val, d, seen, nseg):
            # one step from u in direction d: d = +1 a tight upper bound of
            # s_u above the next node (s_u = s_v + D[u][v]), d = −1 a tight
            # upper bound of the next node above s_u (s_u = s_v − D[v][u]).
            if nseg >= 2:                     # this step ends at an anchor
                w = Df[u][z] if d > 0 else Df[z][u]
                if w is not None:
                    consts.add(val + w if d > 0 else val - w)
                for j in outer_nodes:
                    w = Df[u][j] if d > 0 else Df[j][u]
                    if w is not None:
                        offs.add((j - k - 1, val + w if d > 0 else val - w))
            for v in range(k):
                if v in seen:
                    continue
                w = Df[u][v] if d > 0 else Df[v][u]
                if w is None:
                    continue
                nv = val + w if d > 0 else val - w
                seen.add(v)
                walk(v, nv, -d, seen, nseg + 1)
                seen.discard(v)

        walk(k, 0.0, 1, set(), 0)
        walk(k, 0.0, -1, set(), 0)
        out[k] = (tuple(sorted(consts)), tuple(sorted(offs)))
    return out


def _dbm_kinks(D, m):
    """``(kinks_c, kinks_o)``: per level k, the candidate kinks of the inner
    integral as a function of s_k from the closure ``D`` of the difference
    rows -- constants, and ``(j − k − 1, offset)`` pairs for s_j + offset
    (j > k).  First the direct switches (a vertex coordinate through one
    inner variable): where a bound of an inner variable s_i (i < k) that
    moves with s_k meets one that does not (a constant, or one relative to
    an outer variable); then, for k >= 2, the paths through two or more
    (``_dbm_path_kinks``).  With difference rows only, every vertex
    coordinate of the slice over (s_0, ..., s_k) is among them or is an end
    of s_k's own interval."""
    z = m
    kinks_c = [[] for _ in range(m)]
    kinks_o = [[] for _ in range(m)]
    for k in range(m):
        for i in range(k):
            if D[i][k] is not None:            # s_i < s_k + D[i][k]
                if D[i][z] is not None:
                    kinks_c[k].append(float(D[i][z] - D[i][k]))
                for j in range(k + 1, m):
                    if D[i][j] is not None:
                        kinks_o[k].append((j - k - 1,
                                           float(D[i][j] - D[i][k])))
            if D[k][i] is not None:            # s_i > s_k − D[k][i]
                if D[z][i] is not None:
                    kinks_c[k].append(float(D[k][i] - D[z][i]))
                for j in range(k + 1, m):
                    if D[j][i] is not None:
                        kinks_o[k].append((j - k - 1,
                                           float(D[k][i] - D[j][i])))
    if m >= 3:
        for k, (pc, po) in enumerate(_dbm_path_kinks(D, m)):
            kinks_c[k].extend(pc)
            kinks_o[k].extend(po)
    return kinks_c, kinks_o


class _VertexKinks:
    r"""The s_k-coordinates of the vertices of the slice
    ``{(s_0, ..., s_k) : every row > 0}`` of a region given the outer values
    s_{k+1}, ..., s_{m-1} (levels k >= 1), for regions with rows that are
    not difference rows.  Every (k+1)-subset of the rows that involve
    s_0..s_k and are linearly independent there is solved once per region
    (affinely in the outer values); ``at(k, outer)`` returns the coordinates
    of the solutions that satisfy every row (to a relative 1e-9: a superset
    of the vertices).  ``complete[k]`` is False where more than
    _NQUAD_VERTEX_MAX_COMBOS subsets would be needed (no kinks there).
    """

    def __init__(self, rows_t, m):
        import itertools
        n = len(rows_t)
        A = np.zeros((n, m))
        C = np.zeros(n)
        for r, (c, prs) in enumerate(rows_t):
            C[r] = c
            for (j, a) in prs:
                A[r, j] = a
        self.A, self.C, self.m = A, C, m
        self.data = [None] * m
        self.complete = [True] * m
        for k in range(1, m):
            d = k + 1
            Ain = A[:, :d]
            idx = np.nonzero(np.any(Ain != 0.0, axis=1))[0]
            if len(idx) < d:
                continue
            if math.comb(len(idx), d) > _NQUAD_VERTEX_MAX_COMBOS:
                self.complete[k] = False
                continue
            combos = np.array(list(itertools.combinations(idx, d)))
            Mx = Ain[combos]                                 # (nc, d, d)
            scale = np.prod(np.linalg.norm(Mx, axis=2), axis=1)
            good = np.abs(np.linalg.det(Mx)) > 1e-12 * scale
            if not np.any(good):
                continue
            combos = combos[good]
            P = np.linalg.inv(Mx[good])
            G = -np.einsum('nij,nj->ni', P, C[combos])
            H = -np.einsum('nij,njo->nio', P, A[:, d:][combos])
            self.data[k] = (G, H, Ain, A[:, d:])

    def at(self, k, outer):
        dat = self.data[k]
        if dat is None:
            return ()
        G, H, Ain, Aout = dat
        if Aout.shape[1]:
            o = np.asarray(outer, float)
            X = G + H @ o
            Cv = self.C + Aout @ o
        else:
            X = G
            Cv = self.C
        slack = X @ Ain.T + Cv
        feas = np.all(slack >= -1e-9 * (1.0 + np.abs(Cv)), axis=1)
        return X[feas, k]


def _edge_rows_at(mode_info, free):
    """``[(a, c)]`` for every edge of ``mode_info`` that involves an
    integration variable: its coefficients ``a`` (sorted ``(j, a_j)``
    pairs) and its shift ``c`` at the free times; ``None`` without edge
    data."""
    edges = getattr(mode_info, 'edges', None)
    if edges is None:
        return None
    out = []
    try:
        for (ip, ep, c0) in edges:
            prs = tuple(sorted((i, a) for (i, a) in ip if abs(a) > 1e-15))
            if prs:
                out.append((prs, c0 + sum(a * free[i] for i, a in ep)))
    except (TypeError, ValueError, IndexError):
        return None
    return out


def _unit_edge_rows(rows_t, edge_rows):
    r"""True when the decay certificate and the tail bound of the
    hardened fallback hold as stated (``_integrate_polytope_hardened``):

    (i) every row with an integration variable is a unit difference row
    (one coefficient ±1, or +1 and −1), so the rows form a totally
    unimodular system;  (ii) every such row has the coefficients of an
    edge (its value is that edge's Δt, which decays);  (iii) every edge
    with an integration variable has a row with its coefficients and a
    shift at most its own, so Δt_e >= that row's value > 0 on the region.

    Then every vertex of a cone section of the region is integral, so the
    decay rate along any open direction is 0 or at least κ_min; and the
    change of variables to the values of a spanning tree of rows that
    contains the closure's shortest path from the open variable to its
    finite end (|Jacobian| = 1) bounds the tail beyond a cut at distance
    d from that end by (S/κ_min^m)·Γ(m, κ_min·d)/Γ(m) (``tail_k``).  Every
    region of the public Phase J models measured satisfies (i)-(iii)."""
    if edge_rows is None:
        return False
    rowmin = {}
    for (c, prs) in rows_t:
        if len(prs) == 1:
            if abs(prs[0][1]) != 1.0:
                return False
        elif len(prs) == 2:
            if abs(prs[0][1]) != 1.0 or prs[0][1] != -prs[1][1]:
                return False
        else:
            return False
        old = rowmin.get(prs)
        if old is None or c < old:
            rowmin[prs] = c
    evecs = set()
    for (prs, c) in edge_rows:
        r = rowmin.get(prs)
        if r is None or r > c + 1e-12 * (1.0 + abs(c)):
            return False
        evecs.add(prs)
    return all(p in evecs for p in rowmin)


class _GeneralDecayCertificate:
    r"""The decay certificate of an open direction, and the bound of the
    tail an outermost cut drops, by linear programming -- for regions whose
    rows are not all unit difference rows that are edges
    (``_unit_edge_rows`` False: other coefficients, rows that are not
    edges).  ``|integrand| <= S·exp(−φ(s))``, φ = Σ_e κ_e·Δt_e over the
    edges with an integration variable (``_NquadModes.edge_decay``), where
    every such Δt_e >= 0.

    * Every such edge must be >= 0 on the region (a row with its
      coefficients and a shift at most its own, or an LP minimum >= 0).
    * Rates: for each level k and side σ, the cone section
      {r : A r >= 0 (every row), r_k = σ, r_j = 0 for j > k} -- the
      directions in which the slice of s_0..s_k given the outer values is
      unbounded -- and ρ = min φ'(r) over it (an LP; φ' the linear part of
      φ).  Beyond the farthest vertex of a slice the minimum of φ over it
      grows exactly at the rate ρ along that side, whatever the rows'
      scaling.  ``rate`` = the smallest ρ (``inf`` if no side is open); a
      ρ <= 0 (no decay) or an unbounded LP fails.
    * Tail of the outermost cut (``tail_bound``, outermost side ``side``
      open; the vertex enumeration of the outermost level must be
      complete): beyond the farthest vertex coordinate (the anchor of the
      cut) v(x) = min φ over the slice at s_{m-1} = x and the slice's range
      [lo_k(x), hi_k(x)] in every inner variable are affine in the
      distance; they are measured by LPs at the first cut and one decay
      length beyond it.  Bounded slices: the tail beyond a cut at distance D
      past the first one is at most S·exp(−v₁ − ρD)·∫_0^∞ e^{−ρy}·Π_k
      (w_k + β_k·(D + y)) dy (the slice's volume is at most the product of
      its widths w_k).  A slice unbounded in s_k (on one side only, along
      which e^{−φ} decays: φ's coefficient g_k of the right sign; else no
      certificate): the region lies in the box fibration
      {s_k in [lo_k(x), hi_k(x)]}, on which ∫ e^{−φ} factorises: per
      variable at most e^{−g_k·(the end where g_k·s_k is smallest)}/|g_k|
      (g_k != 0) or e^{0}·w_k (g_k = 0), then the affine exponent is
      integrated over x in closed form (``math.inf`` if it does not
      decay).

    * Everything the cuts drop (``region_tail``, for the inner levels'
      cuts, which ``tail_bound`` does not see): a point that a cut of level
      k drops lies K/κ (κ <= ρ) beyond the farthest vertex coordinate of
      its slice over (s_0..s_k), so φ there is at least φ_min + K (φ_min =
      min φ over the region): the point is a vertex combination v (φ(v) >=
      φ_min) plus a direction r of the slice's recession cone with |r_k| >=
      K/κ, along which φ grows at least at ρ·|r_k|.  The integral of
      S·e^{−φ} over {φ >= φ_min + K} is at most S·e^{−φ_min}·∫_K^∞ e^{−τ}
      V(τ) dτ, V(τ) the volume of {φ <= φ_min + τ} on the region, which
      is at most the product of its widths W_j(τ) in every variable; each
      W_j is concave in τ (a parametric LP), so beyond τ = K it is at most
      W_j(K) + (W_j(K) − W_j(K − 1))·(τ − K) (LPs at both).

    ``ok`` False (``why`` says why) -> the caller treats the region as
    uncertified.  LP values carry a relative 1e-7 margin (1e-6 for the
    widths of ``region_tail``).
    """

    _MARGIN = 1e-7

    def __init__(self, rows_t, m, edge_decay, side, kinks_complete):
        self.m = m
        self.ok = False
        self.rate = None
        self.why = ''
        self.side = side
        self._tail = None
        self._rtail = None
        if edge_decay is None:
            self.why = 'edge data not finite'
            return
        self.S, edges = edge_decay
        n = len(rows_t)
        A = np.zeros((n, m))
        C = np.zeros(n)
        for r, (c, prs) in enumerate(rows_t):
            C[r] = c
            for (j, a) in prs:
                A[r, j] = a
        self.A, self.C = A, C
        g = np.zeros(m)
        phi0 = 0.0
        self._edges = edges
        for (prs, c, kap) in edges:
            for (j, a) in prs:
                g[j] += kap * a
            phi0 += kap * c
        self.g, self.phi0 = g, phi0
        try:
            self._certify(rows_t, kinks_complete)
        except Exception as exc:                         # noqa: BLE001
            self.ok = False
            self.why = f'linear programming failed ({exc!r})'

    @staticmethod
    def _lp(c, A_ub, b_ub, bounds):
        from scipy.optimize import linprog
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds,
                      method='highs')
        return res.status, (res.fun if res.status == 0 else None), res.x

    def _certify(self, rows_t, kinks_complete):
        m, A, C, g = self.m, self.A, self.C, self.g
        free = [(None, None)] * m
        rowmin = {}
        for (c, prs) in rows_t:
            old = rowmin.get(prs)
            if old is None or c < old:
                rowmin[prs] = c
        # every edge with an integration variable is >= 0 on the region
        for (prs, c, _k) in self._edges:
            r = rowmin.get(prs)
            if r is not None and r <= c + 1e-12 * (1.0 + abs(c)):
                continue
            a = np.zeros(m)
            for (j, x) in prs:
                a[j] = x
            st, val, _x = self._lp(a, -A, C, free)
            if st == 2:
                continue                              # empty region
            if st != 0 or val + c < -1e-9 * (1.0 + abs(c)):
                self.why = 'an edge is not >= 0 on the region'
                return
        # decay rate along every open side of every level
        rates = []
        for k in range(m):
            for sg in (-1.0, 1.0):
                keep = np.any(A[:, :k + 1] != 0.0, axis=1)
                Ak = A[keep]
                if k == 0:
                    if np.all(sg * Ak[:, 0] >= 0.0):
                        rates.append(sg * g[0])
                    else:
                        continue
                else:
                    st, val, _x = self._lp(g[:k], -Ak[:, :k], sg * Ak[:, k],
                                           [(None, None)] * k)
                    if st == 2:
                        continue                      # that side is bounded
                    if st != 0:
                        self.why = f'decay LP status {st} (level {k})'
                        return
                    rates.append(val + sg * g[k])
                if not rates[-1] > 1e-12 * (1.0 + float(np.abs(g).sum())):
                    self.why = f'no decay along a side of level {k}'
                    return
        self.rate = (float(min(rates)) * (1.0 - self._MARGIN) if rates
                     else math.inf)
        self.slice_open = [(False, False)] * max(0, m - 1)
        if self.side is not None and m >= 2:
            if not kinks_complete:
                self.why = ('vertex enumeration of the outermost level '
                            'incomplete')
                return
            Ai = A[:, :m - 1]
            keep = np.any(Ai != 0.0, axis=1)
            Ai = Ai[keep]
            for k in range(m - 1):
                unb = []
                for sg in (-1.0, 1.0):
                    b = [(None, None)] * (m - 1)
                    b[k] = (sg, sg)
                    st, _v, _x = self._lp(np.zeros(m - 1), -Ai,
                                          np.zeros(len(Ai)), b)
                    if st not in (0, 2):
                        self.why = f'slice LP status {st}'
                        return
                    unb.append(st == 0)
                lo_u, hi_u = unb
                # the box fibration needs e^{-g_k s_k} to decay on the open
                # side of s_k
                if (lo_u and hi_u) or (lo_u and not g[k] < 0.0) or (
                        hi_u and not g[k] > 0.0):
                    self.why = ('the slice of the outermost level is '
                                f'unbounded in s_{k} without decay')
                    return
                self.slice_open[k] = (lo_u, hi_u)
        self.ok = True

    def _slice(self, x):
        """``(v, lo, hi)`` of the slice at s_{m-1} = x: v = min φ over it
        and the range of every inner variable (±inf on an open side);
        ``None`` if it is empty."""
        m, A, C, g = self.m, self.A, self.C, self.g
        if m == 1:
            return g[0] * x + self.phi0, [], []
        A_ub = -A[:, :m - 1]
        b_ub = C + A[:, m - 1] * x
        bounds = [(None, None)] * (m - 1)
        st, val, _x = self._lp(g[:m - 1], A_ub, b_ub, bounds)
        if st == 2:
            return None
        if st != 0:
            raise ValueError(f'slice LP status {st}')
        v = val + g[m - 1] * x + self.phi0
        lo, hi = [], []
        for k in range(m - 1):
            e = np.zeros(m - 1)
            e[k] = 1.0
            lo_u, hi_u = self.slice_open[k]
            if lo_u:
                lo.append(-math.inf)
            else:
                s1, val, _x = self._lp(e, A_ub, b_ub, bounds)
                if s1 != 0:
                    raise ValueError(f'range LP status {s1}')
                lo.append(val)
            if hi_u:
                hi.append(math.inf)
            else:
                s2, val, _x = self._lp(-e, A_ub, b_ub, bounds)
                if s2 != 0:
                    raise ValueError(f'range LP status {s2}')
                hi.append(-val)
        return v, lo, hi

    def tail_bound(self, x1, x):
        """Bound of the integral beyond the cut x of the outermost variable
        (on the side ``side``), from the slices at the first cut x1 and one
        decay length beyond it; ``math.inf`` if it cannot be bounded."""
        if not self.ok or self.side is None:
            return math.inf
        sgn = -1.0 if self.side == 'lo' else 1.0
        mg = self._MARGIN
        try:
            if self._tail is None or self._tail[0] != x1:
                dl = 1.0 / self.rate if math.isfinite(self.rate) else 1.0
                s1 = self._slice(x1)
                s2 = self._slice(x1 + sgn * dl)
                if s1 is None or s2 is None:
                    self._tail = (x1, None)
                elif not any(lu or hu for (lu, hu) in self.slice_open):
                    # bounded slices: min φ and the widths
                    w1 = [h - l for l, h in zip(s1[1], s1[2])]
                    w2 = [h - l for l, h in zip(s2[1], s2[2])]
                    expo = (-(s1[0] - mg * (1.0 + abs(s1[0]))),
                            (s2[0] - s1[0]) / dl * (1.0 - mg))
                    ws = [max(0.0, w) * (1.0 + mg) + mg for w in w1]
                    bs = [max(0.0, (b - a) / dl) * (1.0 + mg) + mg
                          for a, b in zip(w1, w2)]
                    self._tail = (x1, (expo, 1.0, ws, bs))
                else:
                    # the box fibration: per inner variable e^{-g_k·E_k}/|g_k|
                    # (E_k the end where g_k·s_k is smallest) or the width
                    g = self.g

                    def ex(sl, xx):
                        e = g[self.m - 1] * xx + self.phi0
                        for k in range(self.m - 1):
                            if g[k] > 0.0:
                                e += g[k] * sl[1][k]
                            elif g[k] < 0.0:
                                e += g[k] * sl[2][k]
                        return e
                    e1, e2 = ex(s1, x1), ex(s2, x1 + sgn * dl)
                    if not (math.isfinite(e1) and math.isfinite(e2)):
                        raise ValueError('box fibration not finite')
                    fac = 1.0
                    ws, bs = [], []
                    for k in range(self.m - 1):
                        if g[k] != 0.0:
                            fac /= abs(g[k])
                        else:
                            a = s1[2][k] - s1[1][k]
                            b = s2[2][k] - s2[1][k]
                            ws.append(max(0.0, a) * (1.0 + mg) + mg)
                            bs.append(max(0.0, (b - a) / dl) * (1.0 + mg)
                                      + mg)
                    expo = (-(e1 - mg * (1.0 + abs(e1))),
                            (e2 - e1) / dl * (1.0 - mg))
                    self._tail = (x1, (expo, fac * (1.0 + mg), ws, bs))
        except (ValueError, OverflowError):
            return math.inf
        data = self._tail[1]
        if data is None:
            return 0.0                    # nothing beyond the first cut
        (e0, rho), fac, ws, bs = data
        if not rho > 0.0:
            return math.inf
        D = max(0.0, sgn * (x - x1))
        poly = [1.0]
        for w, b in zip(ws, bs):
            c0 = w + b * D
            new = [0.0] * (len(poly) + 1)
            for j, p in enumerate(poly):
                new[j] += p * c0
                new[j + 1] += p * b
            poly = new
        tot = 0.0
        for j, p in enumerate(poly):
            tot += p * math.exp(math.lgamma(j + 1) - (j + 1) * math.log(rho))
        try:
            return self.S * fac * math.exp(e0 - rho * D) * tot
        except OverflowError:
            return math.inf

    def _region_tail_data(self, K0):
        """``(K0, φ_min, w, β)``: the widths W_j(K0) of {φ <= φ_min + K0}
        on the region, rounded up, and their slopes from W_j(K0 − 1) (or
        W_j(0) if K0 < 1), rounded up; ``None`` if an LP fails."""
        m, A, C, g = self.m, self.A, self.C, self.g
        free = [(None, None)] * m
        st, val, _x = self._lp(g, -A, C, free)
        if st != 0:
            return None
        phimin = val + self.phi0
        A_ub = np.vstack([-A, g[None, :]])

        def widths(tau):
            b_ub = np.concatenate([C, [phimin + tau - self.phi0]])
            out = []
            for j in range(m):
                e = np.zeros(m)
                e[j] = 1.0
                s1, lo, _x = self._lp(e, A_ub, b_ub, free)
                s2, hi, _x = self._lp(-e, A_ub, b_ub, free)
                if s1 != 0 or s2 != 0:
                    return None
                hi = -hi
                out.append((max(0.0, hi - lo),
                            1e-6 * (1.0 + abs(lo) + abs(hi))))
            return out
        t1 = max(0.0, K0 - 1.0)
        w1, w2 = widths(t1), widths(K0)
        if w1 is None or w2 is None:
            return None
        ws = [w + e for (w, e) in w2]
        bs = [max(0.0, (w + e) - max(0.0, w1_ - e1)) / (K0 - t1)
              for ((w, e), (w1_, e1)) in zip(w2, w1)]
        return (K0, phimin, ws, bs)

    def region_tail(self, K):
        """A bound of S·∫ e^{−φ} over the points of the region where φ >=
        φ_min + K -- every point that a cut at K/κ (κ <= the certified
        rate) beyond the farthest vertex coordinate of its slice drops, at
        any level (see the class docstring); ``math.inf`` if it cannot be
        bounded.  The LPs run at the first K asked for (and again only for
        a smaller K)."""
        if not self.ok or not self.rate > 0.0:
            return math.inf
        if self._rtail is None or (self._rtail and K < self._rtail[0]):
            try:
                self._rtail = self._region_tail_data(K) or False
            except (ValueError, OverflowError):
                self._rtail = False
        if not self._rtail:
            return math.inf
        K0, phimin, ws, bs = self._rtail
        D = max(0.0, K - K0)
        poly = [1.0]
        for w, b in zip(ws, bs):
            c0 = w + b * D
            new = [0.0] * (len(poly) + 1)
            for j, p in enumerate(poly):
                new[j] += p * c0
                new[j + 1] += p * b
            poly = new
        tot = 0.0
        for j, p in enumerate(poly):
            tot += p * math.factorial(j)
        lo_phi = phimin - self._MARGIN * (1.0 + abs(phimin))
        try:
            return self.S * math.exp(-lo_phi - K) * tot
        except OverflowError:
            return math.inf


def _integrate_polytope_hardened(integrand_callable, s_constraints,
                                 free_ext_vals, m, mode_info=None,
                                 diag_meta=None):
    r"""The hardened scipy quadrature of ``integrand_callable(s_0..s_{m-1},
    *free_ext_vals)`` over ``{s : a_int·s + shift > 0 for every row}``
    (``NQUAD_HARDENED``; the design is in the comment block above).  s_0 is
    the innermost variable, s_{m-1} the outermost.

    ``mode_info``: an ``_NquadModes`` (default: the integrand's
    ``_nquad_modes`` attribute, set by the fast evaluators); with it the
    innermost variable is integrated in closed form, the first pass's
    epsabs comes from the smaller of a rigorous bound and a sample of
    |integrand| (evaluated from the modes), floored at _NQUAD_SCALE_FLOOR ×
    the bound, open sides are truncated by the
    decay certificate (``_unit_edge_rows``, else
    ``_GeneralDecayCertificate``) and oscillating modes raise the starting
    subinterval limit.  ``None`` -> every level by quad, epsabs from the largest
    |integrand| at _NQUAD_PRESAMPLES points of the region (0 if all of them
    are 0), no geometric breakpoints, open sides capped at the legacy ±200
    (counters ``nquad_hardened_no_modes``, ``nquad_hardened_uncertified``).
    Either way a second pass uses epsabs = NQUAD_EPSABS_FACTOR × |first
    result| when the first epsabs was more than _NQUAD_RERUN_RATIO times
    that, and when only the outermost cut has to move (``tail_k``) the
    strip between the cuts is integrated and added; when the inner levels'
    cuts have to move (``tail_in``; the LP certificate, built on first need
    for unit difference rows that are edges), the region is integrated
    again.  Every region served
    is recorded for the aggregated warnings (``_nquad_note_region``).
    Returns ``None`` (the caller then uses the legacy routines) only
    for a row with a non-finite shift or coefficient, or a time without a
    float value.
    """
    from scipy.integrate import quad
    D, general, verdict = _dbm_closure(s_constraints, m)
    if verdict == 'NONFINITE':
        return None
    try:
        free_f = [float(x) for x in free_ext_vals]
    except (TypeError, ValueError):
        return None                    # e.g. a symbolic time: legacy route
    ctr = _RUNTIME_COUNTERS
    ctr['nquad_hardened_calls'] += 1
    if verdict == 'EMPTY':
        ctr['nquad_hardened_empty'] += 1
        return 0.0 + 0.0j
    if mode_info is None:
        mode_info = getattr(integrand_callable, '_nquad_modes', None)
    tabs = _dbm_level_tables(D, m)
    rows_t = tuple([(c, prs) for (c, prs) in
                    ((float(sh), tuple((j, float(a))
                                       for j, a in enumerate(a_int)
                                       if abs(float(a)) > 1e-15))
                     for (a_int, sh) in s_constraints) if prs])

    vkinks = None
    if general:
        # Rows that are not difference rows: exact level bounds from a
        # Fourier-Motzkin elimination of every row, and the kinks from the
        # vertices of the slices (the closure does not see these rows).
        ctr['nquad_hardened_general_rows'] += 1
        gen = _fm_level_rows(s_constraints, m)
        if gen == 'EMPTY':
            ctr['nquad_hardened_empty'] += 1
            return 0.0 + 0.0j
        if gen is None:
            ctr['nquad_hardened_fm_capped'] += 1
            # a superset of the projection: each such row bounds the level
            # of its innermost variable (it involves no inner one there)
            gen = _general_rows_by_level(general, m)
        if m >= 2:
            vkinks = _VertexKinks(rows_t, m)
            ctr['nquad_hardened_kinks_incomplete'] += sum(
                1 for k in range(1, m) if not vkinks.complete[k])
    else:
        gen = [()] * m
    kinks_c, kinks_o = _dbm_kinks(D, m)
    ext_pts = tuple(sorted(set([0.0] + [t for t in free_f
                                         if math.isfinite(t)])))

    kmin = kfast = None
    if mode_info is not None:
        kmin, kfast = mode_info.decay_rates()
    else:
        ctr['nquad_hardened_no_modes'] += 1
    certified = (kmin is not None and kmin > 0.0 and math.isfinite(kmin))
    span = None
    unit = False
    gcert = None
    if certified:
        # Along an open direction |integrand| <= S·exp(−κ·d) beyond its
        # farthest breakpoint.  Unit difference rows that are edges
        # (``_unit_edge_rows``, every Phase J region measured): κ = κ_min.
        # Otherwise the smallest |coefficient| a <= 1 of an integration
        # variable in the rows and the edges gives κ_min·a, lowered to the
        # rate that linear programming finds along every open side
        # (``_GeneralDecayCertificate``; the rate depends on how the rows
        # combine, not on their scaling) -- or no certificate.
        a_min = 1.0
        mc = getattr(mode_info, 'min_coefficient', None)
        if mc is not None:
            a_min = min(a_min, mc())
        for (_c, prs) in rows_t:
            for (_j, a) in prs:
                if abs(a) < a_min:
                    a_min = abs(a)
        if a_min >= 1.0 - 1e-12:
            kap = kmin
            span = NQUAD_TAIL_K / kmin
        else:
            kap = kmin * a_min
            span = NQUAD_TAIL_K / kap
        unit = _unit_edge_rows(rows_t, _edge_rows_at(mode_info, free_f))
        if not unit:
            Lo, Uo = _level_interval(tabs[m - 1], gen[m - 1], ())
            side = (('lo' if Lo == -math.inf else 'hi')
                    if (Lo == -math.inf) != (Uo == math.inf) else None)
            ed = getattr(mode_info, 'edge_decay', None)
            gcert = _GeneralDecayCertificate(
                rows_t, m, ed(free_f) if ed is not None else None, side,
                vkinks is None or vkinks.complete[m - 1])
            if not gcert.ok:
                certified = False
                span = None
            elif gcert.rate < kap * (1.0 - 1e-6):
                kap = gcert.rate
                span = NQUAD_TAIL_K / kap
        # S of the tail bound: |integrand| on the box of the region's level
        # intervals (open sides infinite), which credits each edge with its
        # smallest Δt on the region (e.g. an edge to a free time 60 away);
        # at most |pref|·Π_e Σ_k |C_ek| (``sup_bound``).
        try:
            s_tail = mode_info.bound([tabs[k][0] for k in range(m)],
                                     [tabs[k][1] for k in range(m)], free_f)
        except (AttributeError, TypeError, ValueError, OverflowError):
            s_tail = math.inf
        sb = getattr(mode_info, 'sup_bound', None)
        if sb is not None and not s_tail <= sb():
            s_tail = sb()
    geo = ()
    geo_min_width = math.inf
    if kfast is not None and kfast > 0.0 and math.isfinite(kfast):
        geo = tuple(_NQUAD_GEOM_RATIO ** j / kfast for j in range(
            _NQUAD_GEOM_J0, _NQUAD_GEOM_J0 + _NQUAD_GEOM_MAX))
        geo_min_width = _NQUAD_GEOM_MIN_WIDTH / kfast
    wosc = 0.0
    if mode_info is not None and hasattr(mode_info, 'osc_bound'):
        wosc = mode_info.osc_bound()

    # First-pass tolerance scale: a rigorous bound of |integrand| on a box
    # that contains every interval the nested quadrature can visit (an open
    # side of s_k is reached within m truncation distances of its finite
    # bound), lowered to the largest |integrand| sampled in the region
    # (below).
    if mode_info is not None:
        lo, hi = [], []
        for k in range(m):
            L0, U0 = tabs[k][0], tabs[k][1]
            if L0 == -math.inf and certified and U0 < math.inf:
                L0 = U0 - m * span
            if U0 == math.inf and certified and L0 > -math.inf:
                U0 = L0 + m * span
            lo.append(L0)
            hi.append(U0)
        B = mode_info.bound(lo, hi, free_f)
        if B == 0.0:                   # identically zero integrand
            return 0.0 + 0.0j
        scale = B if math.isfinite(B) else math.inf
    else:
        scale = math.inf
    epsrel = NQUAD_EPSREL
    # Closed-form innermost level: the rows that involve s_0 are exactly
    # those that bound it in ``_level_interval`` (difference rows through the
    # closure, other rows through ``gen[0]``: the elimination's level-0 rows
    # are every row with s_0, or, past its cap, the rows filed under level
    # 0), so on that interval the filter only has to check the rows WITHOUT
    # s_0 (outer variables only).
    analytic0 = (_NQUAD_ANALYTIC_INNERMOST and mode_info is not None
                 and getattr(mode_info, 'cterms', None) is not None)
    if analytic0:
        avail = getattr(mode_info, 'innermost_available', None)
        if avail is not None and not avail():
            analytic0 = False          # too many terms: s_0 by quad
            ctr['nquad_hardened_innermost_capped'] += 1
    rows_no0 = tuple((c, tuple((j - 1, a) for (j, a) in prs))
                     for (c, prs) in rows_t if all(j != 0 for j, _a in prs))
    free_list = list(free_ext_vals)
    f = integrand_callable
    # [empty intervals, capped (0/1), uncertified (0/1), (unused),
    #  closed-form innermost integrals that overflowed (done by quad),
    #  quad calls retried with a larger limit, pass with epsabs from the
    #  result (0/1), pass with a wider truncation (0/1), inner cuts that
    #  could not be checked (0/1)]
    stats = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    pflags = []                        # QUADPACK messages of this pass
    eps = [0.0]                        # epsabs of the current pass
    kmin_eff = [math.inf]              # κ·(finite end -> outermost cut)
    # The outermost level's truncation: its span (moved out by ``tail_k``)
    # and, after a pass, (side, anchor, cut) of its cut (None: not cut).
    # The inner levels' span (moved out by ``tail_in``) and
    # whether a pass cut an inner level.
    span_out = [span]
    span_in = [span]
    cutinfo = [None]
    inner_cut = [False]
    inf = math.inf

    def g0(*s):
        # The Heaviside filter: Θ(0) = 0, nothing is evaluated outside.
        for (c, prs) in rows_t:
            dt = c
            for (j, a) in prs:
                dt += a * s[j]
            if dt <= 0.0:
                return 0.0 + 0.0j
        return complex(f(*(list(s) + free_list)))

    def gm(*s):
        # The same from the modes (the tolerance's sample: no integrand
        # call when the modes are known).
        for (c, prs) in rows_t:
            dt = c
            for (j, a) in prs:
                dt += a * s[j]
            if dt <= 0.0:
                return 0.0 + 0.0j
        return mode_info.value(s, free_f)

    def truncate(L, U, lo_open, hi_open, count=True, anchor=None,
                 outermost=False):
        # An open side: cut at span = K / κ beyond the farthest of the
        # finite end and ``anchor`` (the farthest breakpoint on that side: a
        # kink of the inner integral or an external time, where its mass
        # can sit far from the finite end), by the decay certificate; else
        # the legacy cap.  For the outermost level (``span_out``), records
        # the effective κ·(distance from the finite end to the cut) in
        # kmin_eff and the cut in cutinfo (for ``tail_k`` and the strip);
        # for an inner level (``span_in``), that it was cut (``tail_in``).
        if not (lo_open or hi_open):
            return L, U
        if certified and not (lo_open and hi_open):
            sp = span_out[0] if outermost else span_in[0]
            if lo_open:
                a = U if anchor is None or not anchor < U else anchor
                lo, hi = a - sp, U
                d = U - lo
                cut = ('lo', a, lo)
            else:
                a = L if anchor is None or not anchor > L else anchor
                lo, hi = L, a + sp
                d = hi - L
                cut = ('hi', a, hi)
            if count:
                stats[1] = 1
                if outermost:
                    ke = kap * d
                    if ke < kmin_eff[0]:
                        kmin_eff[0] = ke
                    cutinfo[0] = cut
                else:
                    inner_cut[0] = True
            return lo, hi
        if count:
            stats[2] = 1
        cap = NQUAD_UNCERTIFIED_CAP
        if lo_open and hi_open:
            return -cap, cap
        if lo_open:
            return min(-cap, U - cap), U
        return L, max(cap, L + cap)

    # The largest |integrand| at points spread over the region (each
    # coordinate uniform in its level's interval given the outer ones;
    # Kronecker sequence): a lower bound of its supremum, so never a looser
    # tolerance than the rigorous bound; without mode data it is the only
    # scale.
    alphas = [math.sqrt(p) % 1.0 for p in (2, 3, 5, 7, 11, 13, 17, 19,
                                           23, 29, 31, 37)]
    sample_f = gm if mode_info is not None and hasattr(mode_info,
                                                       'value') else g0
    Ms = 0.0
    for i in range(1, _NQUAD_PRESAMPLES + 1):
        outer = ()
        for k in range(m - 1, -1, -1):
            L, U = _level_interval(tabs[k], gen[k], outer)
            if not L < U:
                break
            L, U = truncate(L, U, L == -inf, U == inf, count=False)
            u = (i * alphas[k % len(alphas)] + 0.5 * (k // len(alphas))
                 ) % 1.0
            outer = (L + u * (U - L),) + outer
        else:
            try:
                v = abs(sample_f(*outer))
            except (OverflowError, ZeroDivisionError, ValueError):
                continue
            if math.isfinite(v) and v > Ms:
                Ms = v
    if 0.0 < Ms < scale:
        scale = Ms
        # ... but never below _NQUAD_SCALE_FLOOR × the bound: a sample that
        # misses the integrand's peak (at a finite end or a kink, the
        # samples spread over a long truncated side) can sit many orders
        # below it and below the integral, and an epsabs below the rounding
        # noise of the level values ran calls to NQUAD_LIMIT_MAX
        # subintervals.  Between the bound and that floor, the second pass
        # tightens epsabs from the result.
        if (mode_info is not None and math.isfinite(B)
                and scale < _NQUAD_SCALE_FLOOR * B):
            scale = _NQUAD_SCALE_FLOOR * B
    eps[0] = NQUAD_EPSABS_FACTOR * scale if math.isfinite(scale) else 0.0

    def qcall(fun, L, U, p, lim, wprod, other=None, defer=False,
              first=None, ea_min=0.0):
        # One adaptive quad call of the real or the imaginary part of a
        # level (``other``: the other part's value, when known), with
        # epsabs = max(the pass's epsabs, ``ea_min``); returns (value,
        # None), or with ``defer`` (value, (result, limit)) for a call that
        # ended with a QUADPACK message its own tolerances do not excuse:
        # the caller repeats the judgement with ``first`` = that result once
        # the other part is known.
        # * A call whose error estimate meets the level's complex value's
        #   tolerance, max(epsabs, epsrel·|re + i·im|), is accepted: a part
        #   that is only the rounding noise of the other (the imaginary
        #   part of a real integrand) is not refined to an epsabs below its
        #   noise, which runs each such call to its subinterval limit.
        # * Otherwise a call that reaches its subinterval limit is repeated
        #   once with the limit NQUAD_LIMIT_MAX (QUADPACK's bisections do
        #   not depend on the limit until half of it is used, so a larger
        #   limit costs only the subintervals actually needed; the level's
        #   cache keeps the values already computed).  Any other nonzero
        #   ier, or the limit NQUAD_LIMIT_MAX reached, is recorded (counter
        #   + warning) -- unless the call's own error estimate, integrated
        #   over the enclosing levels' intervals (``wprod``, their widths'
        #   product; 1 at the outermost level), stays within the region's
        #   relative tolerance, NQUAD_EPSREL × the pass's scale (eps /
        #   NQUAD_EPSABS_FACTOR: |result| in a second pass), or meets the
        #   requested tolerance itself.  (Inner calls of a second pass ask
        #   for 1e-13·|result| absolute, which rounding of their own values
        #   can prevent: QUADPACK's roundoff ier = 2 there is no loss at the
        #   region's tolerance.)
        ea = max(eps[0], ea_min)
        r = first
        while True:
            if r is None:
                r = quad(fun, L, U, epsabs=ea, epsrel=epsrel, limit=lim,
                         points=p, full_output=1)
            if len(r) <= 3:
                return r[0], None
            if other is not None and r[1] <= max(
                    ea, epsrel * abs(complex(r[0], other))):
                return r[0], None
            info = r[2] if isinstance(r[2], dict) else {}
            hit = info.get('last', 0) >= lim
            if not hit and (
                    r[1] <= max(ea, epsrel * abs(r[0]))
                    or r[1] * wprod <= NQUAD_EPSREL * eps[0]
                    / NQUAD_EPSABS_FACTOR):
                return r[0], None
            if defer:
                return r[0], (r, lim)
            if hit and lim < NQUAD_LIMIT_MAX:
                lim = NQUAD_LIMIT_MAX
                stats[5] += 1
                r = None
                continue
            pflags.append(str(r[3]).split('\n')[0].strip())
            return r[0], None

    def level(k, outer, wprod=1.0, window=None):
        # ``window``: integrate the outermost level over this sub-interval
        # (a strip between two cuts, ``tail_k``) instead of its truncated
        # interval; the breakpoints inside it are kept.
        L, U = _level_interval(tabs[k], gen[k], outer)
        if not L < U:
            stats[0] += 1              # empty: no sampling at all
            return 0.0 + 0.0j
        lo_open, hi_open = L == -inf, U == inf
        if k == 0 and analytic0:
            for (c, prs) in rows_no0:
                dt = c
                for (oi, a) in prs:
                    dt += a * outer[oi]
                if dt <= 0.0:
                    return 0.0 + 0.0j       # Θ(0) = 0, as the filter
            La, Ua = (window if window is not None else
                      truncate(L, U, lo_open, hi_open, outermost=(m == 1)))
            try:
                return mode_info.integrate_innermost(La, Ua, outer, free_f)
            except (OverflowError, ZeroDivisionError):
                stats[4] += 1               # numerically, below
        # One ulp outward: never cut the slice; the filter removes the rest.
        if not lo_open:
            L = math.nextafter(L, -inf)
        if not hi_open:
            U = math.nextafter(U, inf)
        # Breakpoints in the (possibly open) interval: the external times and
        # every kink of the inner integral (vertex coordinates of the slice).
        pts = [p for p in ext_pts if L < p < U]
        for v in kinks_c[k]:
            if L < v < U:
                pts.append(v)
        for (oi, off) in kinks_o[k]:
            v = outer[oi] + off
            if L < v < U:
                pts.append(v)
        if vkinks is not None and k >= 1:
            for v in vkinks.at(k, outer):
                v = float(v)
                if L < v < U:
                    pts.append(v)
        anchor = None
        if pts and (lo_open or hi_open) and _NQUAD_TAIL_ANCHOR:
            fin = [p for p in pts if math.isfinite(p)]
            if fin:
                anchor = min(fin) if lo_open else max(fin)
        if window is None:
            L, U = truncate(L, U, lo_open, hi_open, anchor=anchor,
                            outermost=(k == m - 1))
            lo_t, hi_t = lo_open, hi_open
        else:
            (L, U), lo_t, hi_t = window, True, True
        pts = [p for p in pts if L < p < U]
        if geo and U - L > geo_min_width:
            # A peak sits at a finite end or at a breakpoint (the integrand
            # is a sum of exponentials between kinks).  Every panel wider
            # than geo_min_width gets geometric points towards each of its
            # ends that can carry one (not a truncated side or a strip's
            # end), up to its middle, so no panel next to a peak is much
            # wider than the peak's scale.
            anchors = sorted(set(pts))
            anchors = [L] + anchors + [U]
            extra = []
            last = len(anchors) - 2
            for i in range(len(anchors) - 1):
                a, b = anchors[i], anchors[i + 1]
                if not b - a > geo_min_width:
                    continue
                mid = 0.5 * (a + b)
                if not (i == 0 and lo_t):
                    for g in geo:
                        v = a + g
                        if not v < mid:
                            break
                        extra.append(v)
                if not (i == last and hi_t):
                    for g in geo:
                        v = b - g
                        if not v > mid:
                            break
                        extra.append(v)
            pts.extend(extra)
        if pts:
            # Drop breakpoints within a relative 1e-12 of an end or of each
            # other (e.g. an external time equal to an end that the
            # one-ulp widening moved): a sliver panel makes QUADPACK stop
            # with ier = 3 before it converges.
            sep = 1e-12 * max(1.0, abs(L), abs(U))
            kept = []
            for v in sorted(pts):
                if v - L > sep and U - v > sep and (
                        not kept or v - kept[-1] > sep):
                    kept.append(v)
            pts = kept
        lim = max(NQUAD_LIMIT, 2 * (len(pts) + 2))
        if wosc > 0.0:
            # Oscillating modes: start with the limit their oscillation
            # count over the interval needs.
            n_osc = wosc * (U - L) / math.pi
            if n_osc > _NQUAD_OSC_MIN:
                lim = max(lim, min(NQUAD_LIMIT_MAX,
                                   NQUAD_LIMIT + int(math.ceil(n_osc))))
        cache = {}
        if k == 0:
            def F(x):
                v = cache.get(x)
                if v is None:
                    v = g0(x, *outer)
                    cache[x] = v
                return v
        else:
            km1 = k - 1
            wk = wprod * (U - L)

            def F(x):
                v = cache.get(x)
                if v is None:
                    v = level(km1, (x,) + outer, wk)
                    cache[x] = v
                return v
        p = pts if pts else None
        # The part (real or imaginary) that is larger at the centre of the
        # first panel (a node of quad's first rule, so no extra evaluation)
        # first, its judgement deferred if QUADPACK reported a problem; then
        # the other part, with epsabs raised to epsrel·|first part| (the
        # complex value's tolerance) unless the first was deferred, and
        # judged against the complex value; then the first part again, so
        # judged, if it was deferred.  A part that is only the rounding
        # noise of the other then does not drive the subdivision (measured:
        # a grouped model-free m = 3 region ran past three million
        # evaluations when only the judging was relative to the complex
        # value, and takes 0.3 s with this order and epsabs).
        v0 = F(0.5 * (L + (pts[0] if pts else U)))
        swap = abs(v0.imag) > abs(v0.real)
        if swap:
            pa, pb = (lambda x: F(x).imag), (lambda x: F(x).real)
        else:
            pa, pb = (lambda x: F(x).real), (lambda x: F(x).imag)
        a_, held = qcall(pa, L, U, p, lim, wprod, defer=True)
        b_, _h = qcall(pb, L, U, p, lim, wprod, other=a_,
                       ea_min=epsrel * abs(a_) if held is None else 0.0)
        if held is not None:
            a_, _h = qcall(pa, L, U, p, held[1], wprod, other=b_,
                           first=held[0])
        return complex(b_, a_) if swap else complex(a_, b_)

    def strip(lo_, hi_):
        # The outermost level over [lo_, hi_], the strip between two cuts,
        # at the current pass's tolerance (the inner levels keep their cut).
        return level(m - 1, (), 1.0, window=(lo_, hi_))

    gx1 = [None]                       # the first pass's outermost cut

    def tail_k(a, K):
        # The bound of the tail dropped by the outermost cut (at K/κ past
        # its anchor) must be at most _NQUAD_TAIL_REL·|result|; returns K
        # if it is, else the smallest K + j (j = 1, 2, ...; capped at
        # _NQUAD_TAIL_K_MAX) at which it is.
        # * Unit difference rows that are edges (``unit``): the tail is at
        #   most (S/κ^m)·Q(m, K_eff), Q(m, K) = e^{-K}·Σ_{j<m} K^j/j! and
        #   K_eff = κ·(distance from the finite end to the cut) >= K: with
        #   the values of a spanning tree of rows that contains the
        #   closure's shortest path from the variable to its finite end as
        #   coordinates (|Jacobian| = 1; the path's values add up to that
        #   distance, every tree value decays at least at the rate κ, the
        #   other rows' factors are <= 1), whatever the shape of the inner
        #   slices.  S credits each edge with its smallest Δt on the box of
        #   the region (0 on the path).
        # * Other rows: ``_GeneralDecayCertificate.tail_bound`` (the minimum
        #   of φ and the widths of the slices, by linear programming).
        # Relative to S that says nothing against the integral, which can be
        # far below S.  (The inner levels' cuts: ``tail_in``.)
        need = _NQUAD_TAIL_REL * a
        ci = cutinfo[0]
        if ci is None or not need > 0.0:
            return K
        if unit:
            K_eff = kmin_eff[0]
            if not (math.isfinite(s_tail) and math.isfinite(K_eff)):
                return K
            lscale = math.log(s_tail) - m * math.log(kap) - math.log(need)

            def ok(Kp):
                Ke = Kp + (K_eff - K)
                q = sum(math.exp(j * math.log(Ke) - math.lgamma(j + 1) - Ke)
                        for j in range(m))
                return q <= 0.0 or math.log(q) + lscale <= 0.0
        elif gcert is not None:
            side, anc, cut = ci
            if gx1[0] is None:
                gx1[0] = cut

            def ok(Kp):
                x = anc - Kp / kap if side == 'lo' else anc + Kp / kap
                return gcert.tail_bound(gx1[0], x) <= need
        else:
            return K
        if ok(K):
            return K
        Kp = K
        while Kp < _NQUAD_TAIL_K_MAX and not ok(Kp):
            Kp += 1.0
        return Kp

    # Every inner level's cut (given the outer values, at K/κ beyond the
    # farthest vertex coordinate of its slice) is checked with rows that are
    # not unit difference rows that are edges (``gcert``): the bound of
    # everything the cuts of a pass drop,
    # ``_GeneralDecayCertificate.region_tail`` (φ >= φ_min + K on every
    # dropped point; K·min(1, rate/κ) if κ is a hair above the certified
    # rate), must be at most _NQUAD_TAIL_REL·|result|, else the inner K
    # moves out (the outermost cut at least as far) and the region is
    # integrated again.  Measured before: slices widening with the distance
    # (rows that are not edges: volume ~ distance^j) with modes that cancel
    # to 2^-24 of their parts lost up to 2.4e-7 of the integral at the inner
    # cuts.  The bound needs a complete vertex enumeration at every inner
    # level (else the cut's anchor may not be the farthest vertex) and the
    # anchored cuts; without them the cuts stay unchecked, counted in
    # ``nquad_hardened_inner_unchecked``.  Unit difference rows that are
    # edges are checked the same way, with the general certificate built
    # on first need (``_inner_certificate``).
    inner_checkable = _NQUAD_TAIL_ANCHOR and (
        vkinks is None or all(vkinks.complete[k] for k in range(1, m)))

    gcert_in = [gcert]

    def _inner_certificate():
        # The unit path has no certificate of its own (its outermost cut is
        # bounded by Q(m, K)); the inner levels' cuts are checked with the
        # general one, built on first need.  Measured before: on unit rows an
        # open inner slice holding a chain of edges lost up to 4e-8 of the
        # integral at K = 40 with nothing counted.
        if gcert_in[0] is None and unit:
            try:
                ed = getattr(mode_info, 'edge_decay', None)
                # side=None: ``region_tail`` uses neither the outermost
                # side nor its slices; with the outermost side open, a slice
                # unbounded in an inner variable would otherwise fail the
                # certificate and leave the inner cuts unchecked.
                gc = _GeneralDecayCertificate(
                    rows_t, m, ed(free_f) if ed is not None else None, None,
                    vkinks is None or vkinks.complete[m - 1])
                gcert_in[0] = gc if gc.ok else False
            except (ValueError, ArithmeticError, np.linalg.LinAlgError):
                gcert_in[0] = False
        return gcert_in[0] or None

    def tail_in(a, K):
        if not inner_cut[0]:
            return K
        gc = _inner_certificate()
        if gc is None:
            stats[8] = 1               # counted: nquad_hardened_inner_unchecked
            return K
        need = _NQUAD_TAIL_REL * a
        fac = min(1.0, gc.rate / kap)
        if not (inner_checkable
                and math.isfinite(gc.region_tail(K * fac))):
            stats[8] = 1               # counted: nquad_hardened_inner_unchecked
            return K

        def ok(Kp):
            return gc.region_tail(Kp * fac) <= need
        if ok(K):
            return K
        Kp = K
        while Kp < _NQUAD_TAIL_K_MAX and not ok(Kp):
            Kp += 1.0
        return Kp

    try:
        floor_limited = [0.0]
        val = level(m - 1, ())
        # Further passes, the tolerances relative to the result:
        # * epsabs: the first pass's scale (bound or sample of |integrand|)
        #   can exceed the integral by orders of magnitude (close poles,
        #   oscillation, decay across the region); QUADPACK then stops at
        #   its first estimates.  The target is NQUAD_EPSABS_FACTOR·|result|,
        #   floored at NQUAD_EPSABS_FACTOR·1e-9·(sampled max) so that regions
        #   whose integral sits far below rounding (e.g. 1e-50 on a near-tie
        #   sliver) are not chased; where that floor allows more than 1e-8
        #   relative error (thin regions) the region is counted
        #   (``nquad_hardened_floor_limited``) and warned about.  The region
        #   is integrated again (with the outermost cut of ``tail_k``).
        # * the inner levels' cuts (``tail_in``): when they
        #   move, the region is integrated again (the outermost cut at least
        #   as far out).
        # * the outermost cut (``tail_k``), when only it moves: the strip
        #   between the old and the new cut is integrated and added.
        used = eps[0]
        K_used = NQUAD_TAIL_K
        K_in = NQUAD_TAIL_K
        for _ in range(_NQUAD_RERUN_MAX):
            a = abs(val)
            if not (a > 0.0 and math.isfinite(a)):
                break
            target = NQUAD_EPSABS_FACTOR * max(a, 1e-9 * Ms)
            # The floor keeps regions whose value lies far below the
            # integrand's size (near-coincident times: values ~1e-50) from
            # being chased into rounding noise, but for a thin region it can
            # allow more than the 1e-8 relative contract.  Never accept that
            # silently: count it and warn (resolving such regions exactly is
            # the box-free integration's job).
            floor_rel = (NQUAD_EPSABS_FACTOR * 1e-9 * Ms / a
                         if a > 0.0 else 0.0)
            if 1e-9 * Ms > a and floor_rel > 1e-8:
                floor_limited[0] = floor_rel
            K_new = tail_k(a, K_used) if stats[1] else K_used
            K_in_new = tail_in(a, K_in) if stats[1] else K_in
            if K_new > K_used or K_in_new > K_in:
                stats[7] = 1
            if used > _NQUAD_RERUN_RATIO * target or K_in_new > K_in:
                if used > _NQUAD_RERUN_RATIO * target:
                    stats[6] = 1
                    eps[0] = used = target
                if K_in_new > K_in:
                    K_in = K_in_new
                    span_in[0] = K_in / kap
                    K_new = max(K_new, K_in)
                if K_new > K_used:
                    K_used = K_new
                    span_out[0] = K_new / kap
                del pflags[:]
                kmin_eff[0] = math.inf
                cutinfo[0] = None
                inner_cut[0] = False
                val = level(m - 1, ())
            elif K_new > K_used:
                side, anc, cut = cutinfo[0]
                new = anc - K_new / kap if side == 'lo' else anc + K_new / kap
                val = val + (strip(new, cut) if side == 'lo'
                             else strip(cut, new))
                kmin_eff[0] += K_new - K_used
                cutinfo[0] = (side, anc, new)
                K_used = K_new
                span_out[0] = K_new / kap
            else:
                break
    finally:
        ctr['nquad_hardened_empty_intervals'] += stats[0]
        ctr['nquad_hardened_capped'] += stats[1]
        ctr['nquad_hardened_uncertified'] += stats[2]
        ctr['nquad_hardened_quad_flags'] += len(pflags)
        ctr['nquad_hardened_innermost_overflow'] += stats[4]
        ctr['nquad_hardened_quad_retries'] += stats[5]
        ctr['nquad_hardened_reruns'] += stats[6]
        ctr['nquad_hardened_tail_widened'] += stats[7]
        ctr['nquad_hardened_inner_unchecked'] += stats[8]
        if stats[1]:
            sp_max = max(span, span_out[0], span_in[0])
            if sp_max > ctr['nquad_hardened_cap_span_max']:
                ctr['nquad_hardened_cap_span_max'] = sp_max
    if floor_limited[0] > 0.0:
        ctr['nquad_hardened_floor_limited'] += 1
        pflags.append('absolute tolerance limited by the rounding floor of '
                      'the region: requested relative accuracy only '
                      f'{floor_limited[0]:.1e}')
    _nquad_note_region(diag_meta, m, no_modes=mode_info is None,
                       uncertified=bool(stats[2]), flag_msgs=pflags)
    return val


# ───────────────────────────────────────────────────────────────────────
# Edge → propagator-index lookup (matches _resolve_edge_propagator_data)
# ───────────────────────────────────────────────────────────────────────

def format_td_integral_latex(
    tree_result,
    typed_diagram=None,
    combined_prefactor=None,
    ext_time_vars=None,
    label=None,
):
    r"""
    Build a LaTeX string describing the time-domain integral that
    Phase J evaluates for a given tree diagram, in the same spirit as
    the notebook's `show_integral` helper for the frequency-domain
    Phase I integrand.

    This helper is intended for debugging and documentation only —
    numerical evaluation goes through `tree_result['contribution']`.

    Parameters
    ----------
    tree_result : dict
        Output of `integrate_tree_diagram`. Must contain the
        `edge_info`, `integration_vars`, and `constraints` keys.
    typed_diagram : TypedDiagram, optional
        Used for the leaf/internal-vertex display. If not supplied the
        vertex-time assignment is inferred from `tree_result`.
    combined_prefactor : SR or numeric, optional
        Prefactor to display in front of the integral. If None, the
        prefactor is folded into the integrand and not displayed
        separately.
    ext_time_vars : list of SR, optional
        External time symbols, used for display only. If not provided,
        default labels `t_1, t_2, ...` are inferred.
    label : str, optional
        A short label for the diagram (e.g., 'Tree-1', 'Hawkes-k2').
        Rendered as a prefix in the output.

    Returns
    -------
    str
        A LaTeX string, ready to pass to `display(Math(...))` in a
        Jupyter notebook. Includes:
          - vertex-time assignment block
          - the integral expression
            `C_Γ = pref · ∫ ds_1 ... ds_m  ∏ G_R[…](…)`
          - the retarded polytope constraints `Θ(t_v − t_u)`
          - any δ-edge summary (edges with a nonzero δ coefficient)
    """
    from sage.all import latex

    edge_info = tree_result.get('edge_info', [])
    integration_vars = tree_result.get('integration_vars', [])
    constraints = tree_result.get('constraints', [])

    lines = []
    if label:
        lines.append(r'\text{' + str(label) + r'} \;:')

    # Integration variables
    if integration_vars:
        int_vars_tex = r' \wedge '.join(latex(v) for v in integration_vars)
        lines.append(
            r'\text{non-leaf vertex times: } \{' + int_vars_tex + r'\}'
        )
        integrals_tex = ''
        for v in integration_vars:
            integrals_tex += r'\int\!d' + latex(v) + r'\;'
    else:
        integrals_tex = ''

    # Combined prefactor.  For diagrams with a non-local cumulant
    # kernel (NoiseSourceType vertex), the substituted, simplified
    # τ-dependent prefactor lives in ``tree_result['cumulant_prefactor']``
    # and goes INSIDE the integral as a kernel factor.  For ordinary
    # diagrams the kappa machinery is absent and the user-supplied
    # ``combined_prefactor`` (τ-independent scalar) goes OUTSIDE.
    pref_tex = ''
    inside_kernel_tex = ''
    cumulant_pref = tree_result.get('cumulant_prefactor', None)
    has_cumulant = tree_result.get('has_cumulant_kernel', False)
    if has_cumulant and cumulant_pref is not None:
        # The cumulant prefactor is τ-dependent — render it inside the
        # integral, parenthesised, before the propagator product.
        inside_kernel_tex = (
            r'\bigl[' + latex(SR(cumulant_pref)) + r'\bigr] \cdot '
        )
    elif combined_prefactor is not None:
        pref_tex = latex(combined_prefactor) + r'\;'

    # Build the edge-factor product
    factor_bits = []
    delta_summary_bits = []
    for ei in edge_info:
        u = ei['u']
        v = ei['v']
        ri = ei['ri']
        pi = ei['pi']
        dt = ei['dt_sym']
        delta_c = ei.get('delta_coeff', 0)
        # G_R[pi, ri](dt)
        factor_bits.append(
            r'G^{R}_{' + str(pi) + ',' + str(ri) + r'}\!\bigl('
            + latex(dt) + r'\bigr)'
        )
        try:
            if abs(complex(delta_c)) > 1e-15:
                delta_summary_bits.append(
                    r'c_{' + str(pi) + ',' + str(ri) + r'} = '
                    + latex(delta_c)
                )
        except Exception:
            pass

    product_tex = r' \cdot '.join(factor_bits) if factor_bits else r'1'

    # Main equation: C_Γ = pref × ∫ds × [κ(τ)] × ∏ G_R × Θ
    lines.append(
        r'C_{\Gamma}(t) \;=\; '
        + pref_tex
        + integrals_tex
        + inside_kernel_tex
        + product_tex
    )

    # Retardation constraints (as Θ factors)
    if constraints:
        theta_bits = [
            r'\Theta\bigl(' + latex(SR(c)) + r'\bigr)'
            for c in constraints
        ]
        lines.append(
            r'\text{retardation: } '
            + r' \cdot '.join(theta_bits)
        )

    # δ-edge summary, if any
    if delta_summary_bits:
        lines.append(
            r'\text{instantaneous } \delta\text{-edge coefficients: } '
            + r', \; '.join(delta_summary_bits)
        )

    n_subsets = tree_result.get('n_subsets_evaluated')
    n_skipped = tree_result.get('n_shotnoise_skipped', 0)
    if n_subsets is not None:
        lines.append(
            r'\text{δ-edge subsets evaluated: } '
            + str(n_subsets)
            + (
                r'\;\;(\text{shot-noise skipped: } '
                + str(n_skipped) + r')'
                if n_skipped else ''
            )
        )

    return r' \\ '.join(lines)


def _lookup_prop_indices(typed_diagram, edge_key):
    """
    Look up (resp_row, phys_col) propagator indices for a prediagram
    edge, using the same fallback order as
    `_resolve_edge_propagator_data` in `engine/integration/symbolic.py`.
    """
    u, v, lbl = edge_key
    prop_indices = typed_diagram.propagator_indices
    td_edge_keys = list(typed_diagram.edge_types.keys())

    # 1. Exact match on (u, v, lbl)
    for td_ek in td_edge_keys:
        if td_ek == (u, v, lbl):
            return prop_indices[td_ek]

    # 2. Match on (u, v, None)
    for td_ek in td_edge_keys:
        if (td_ek[0], td_ek[1]) == (u, v) and (
            len(td_ek) < 3 or td_ek[2] is None
        ):
            return prop_indices[td_ek]

    # 3. First edge matching (u, v) regardless of label
    for td_ek in td_edge_keys:
        if (td_ek[0], td_ek[1]) == (u, v):
            return prop_indices[td_ek]

    raise KeyError(
        f"No propagator indices found for edge ({u}, {v}, {lbl}) in "
        f"typed diagram."
    )


# ───────────────────────────────────────────────────────────────────────
# Backward-compatibility alias
# ───────────────────────────────────────────────────────────────────────
# ``integrate_tree_diagram`` was the original name for what is now
# ``integrate_diagram`` (generalised to any loop order).  Keep the old
# name resolvable so tests and external callers don't break.
integrate_tree_diagram = integrate_diagram
