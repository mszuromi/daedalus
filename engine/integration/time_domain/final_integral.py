"""
engine.integration.time_domain.final_integral
=============================================
Phase J: the vertex-time integration of a typed Feynman diagram, at any loop
order, with the δ(t) component of instantaneous propagator entries.

Integrand
---------
Every retarded propagator edge ``(u -> v)`` with matrix index
``(phys=p, resp=r)`` is

    G_R[p, r](Δt) = delta_coeff · δ(Δt) + Θ(Δt) · Σ_α C_α exp(λ_α Δt)

(``EdgeModeSum``: residues ``C_α`` and rates ``λ_α`` on numeric poles).
Expanding the product over the edges gives ``2^|E|`` subsets ``S``: the
edges of ``S`` take their δ form, which eliminates integration times (or
pins them to external times; the shot-noise δ(τ) terms are returned
separately, never added to ``C_tau``), and the others their smooth form,
whose Heaviside factors make each remaining region a polytope of difference
rows ``a · s + c > 0``.  After the pole expansion every subset integrand is
a sum of single exponentials ``C · exp(α · s + γ)``.

Integration routes (closed forms on numeric poles; no symbolic integral)
------------------------------------------------------------------------
* m = 1: the interval; m = 2: the polygon; m >= 3: the chain sum over the
  linear extensions of the causal poset (``_integrate_1d_polytope_
  modesum``, ``_integrate_2d_polygon_modesum``, ``_integrate_nd_polytope_
  poset_modesum``), with a per-subset plan and memoised chain integrals.
* They run box-free on the divided-difference kernels of ``expdd``
  (``boxfree.py``): the exact unbounded region (finite vertices and
  recession rays), the semi-infinite chain, γ folded into the logarithmic
  scale before any exp, and one overflow test on the final exponent;
  many-tuple regions take the batched compiled kernels.  An open direction
  must decay (the convergence certificate); a failure is dropped when its
  pole-tuple group cancels exactly, and otherwise handled by the divergence
  policy (``on_divergence``: 'fallback' / 'warn' / 'raise').
* A subset no closed form takes goes to the exact difference-constraint
  route (``dbm_integral``) and only then to the hardened ``scipy``
  quadrature fallback (``_integrate_polytope``), which warns.  On the model
  zoo no region reaches it.
* Θ(0) = 0 (Itô) for constant rows left by δ-elimination; an exact tie of
  external times follows their tie order (``_const_row_verdict``).

Module attributes read at call time: ``PHASE_J_T_MIN`` (a finite lower
bound of every time, for switch-on transients) and the tuning constants
below.  The pre-0.2.0 rollback switches (``DAEDALUS_PHASE_J_LEGACY`` and the
per-milestone ones, ``_RETIRED_PHASE_J_ENV``) were retired in audit W7: set,
each warns once and is ignored.  The public entry point is
``integrate_diagram``; its ``contribution(*ext_time_values) -> complex``
callable evaluates the diagram.
"""

import math
import os as _os
import itertools
import functools as _functools
from collections import namedtuple
from dataclasses import dataclass
from fractions import Fraction as _Fraction

import numpy as np
from sage.all import SR, fast_callable, CDF, solve as sage_solve
from sage.all import QQ as _QQ, ZZ as _ZZ
from sage.symbolic.operators import (
    add_vararg as _sr_add_op, mul_vararg as _sr_mul_op)
from sage.rings.complex_double import (
    ComplexDoubleElement as _ComplexDoubleElement)

from engine.integration.time_domain.propagator_td import (
    build_G_t_matrix,
    G_t_entry,
    G_t_delta_coeff,
)
from engine.core.vertices import NoiseSourceType, ConvVertexType
from engine import _env


# ───────────────────────────────────────────────────────────────────────
# Retired environment switches (audit W7)
# ───────────────────────────────────────────────────────────────────────
# Each restored a pre-0.2.0 route or a slower setup path.  Set, it warns once
# per process (``FutureWarning``) and is otherwise ignored.
_RETIRED_PHASE_J_ENV = (
    ('DAEDALUS_PHASE_J_LEGACY',
     'the pre-0.2.0 Phase J numbers are no longer reproducible in-process; '
     'they are kept as frozen baselines (tests/fixtures/'
     'phase_j_legacy_baseline.npz, tests/phase_j_refactor_fixtures/legacy/) '
     '(audit W7)'),
    ('DAEDALUS_PHASE_J_THETA0_CONST_ROW',
     "the Ito rule Theta(0) = 0 with the external-time tie order is the only "
     "constant-row convention ('legacy_clip' is gone) (audit W7)"),
    ('DAEDALUS_PHASE_J_STRUCTURAL_ZEROS',
     'identically-zero subsets are always skipped (audit W7)'),
    ('DAEDALUS_PHASE_J_NQUAD_HARDENED',
     'the hardened quadrature fallback is the only one (audit W7)'),
    ('DAEDALUS_PHASE_J_DBM',
     'the poset lower-bound inheritance rule and the exact DBM route are '
     'always on (audit W7)'),
    ('DAEDALUS_PHASE_J_DD_KERNELS',
     'the divided-difference kernels are the only Phase J kernels '
     '(audit W7)'),
    ('DAEDALUS_CHAIN_UPPERS_KERNEL',
     'the divided-difference chain kernels replaced both chain evaluators '
     '(audit W7)'),
    ('DAEDALUS_CHAIN_UPPERS_MEMO',
     'the chain integrals are always memoised (audit W7)'),
    ('DAEDALUS_PHASE_J_SEMIINF_CHAIN',
     'the semi-infinite chain is the only m >= 3 route (audit W7)'),
    ('DAEDALUS_PHASE_J_UNBOUNDED_2D',
     'the box-free closed forms are the only route (audit W7)'),
    ('DAEDALUS_PRUNE_ZERO_MODES',
     'pole tuples with an exactly zero residue are always pruned '
     '(audit W7)'),
    ('DAEDALUS_PHASE_J_LEGACY_SETUP',
     'the fast setup path is the only one; it gives the same numbers '
     '(audit W7)'),
    ('DAEDALUS_SETUP_ZERO_EXIT',
     'the zero-prefactor early exit is always on (audit W7)'),
    ('DAEDALUS_SETUP_PROP_TD',
     'the call-scoped propagator data are always used (audit W7)'),
    ('DAEDALUS_SETUP_LAZY_SR',
     'the SR objects are always built on demand (audit W7)'),
    ('DAEDALUS_SETUP_DELTA_SOLVE',
     'the delta edges are always eliminated exactly '
     '(DAEDALUS_PHASE_J_VALIDATE_DELTA still cross-checks sage_solve) '
     '(audit W7)'),
    ('DAEDALUS_SHOTNOISE_INTEGRATED',
     'integrated shot-noise records are always built; the off path dropped '
     'shot-noise subsets silently (audit W7)'),
)
for _name, _note in _RETIRED_PHASE_J_ENV:
    _env.retired(_name, _note)


def _warn_retired_env():
    """The retired switches are read again at the start of every Phase J
    pass (``compute_correction_td``), as their readers were: one set after
    import warns too (once per process and name)."""
    for name, note in _RETIRED_PHASE_J_ENV + (
            ('DAEDALUS_SETUP_AUT_MEMO',
             'the automorphism-order memo is always on (audit W7)'),
            ('DAEDALUS_POLE_CLUSTERS',
             'the near-degenerate pole deformation is always on (audit W7)')):
        _env.retired(name, note)
del _name, _note


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
    (Im ω > 1e-12 of the largest |root|, ``api/_propagator.py``).  A
    marginal or non-retarded mode -- e.g. λ = 0 at q = 0 for a massless
    field -- is dropped from the list although the entry still has a
    smooth part, which then cannot be evaluated from the pole list.  Raised
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
            f'only strictly retarded poles (Im ω > 1e-12 max|root|, '
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
    propagator builder keeps only strictly retarded poles (Im ω > 1e-12 of
    the largest |root|, ``api/_propagator.py``) and drops a marginal λ = 0 mode (e.g. a
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
    # The log-derivative at 0 fits ANY kernel; check that it is the kernel
    # (a sum of exponentials would otherwise get a wrong single mode), at
    # times on the kernel's OWN scale (audit W4b, ACC-60: fixed times 0.37,
    # 1.9, 4.3 underflowed for a fast kernel, so r e^{-rt} cos(rt) at
    # r = 2000 passed with nothing compared and was fitted as one real mode;
    # its r = 2 twin was refused).  The scale is |lambda| together with the
    # curvature of log g at 0 (which a pure exponential lacks), so that
    # cos(w t) (lambda = 0) is probed at t ~ 1/w; both scale like a rate,
    # so the probe times are fixed multiples of the kernel's time unit.
    import cmath
    rate = abs(lam)
    if rate == 0:
        # no rate to set the unit (a constant, or cos(w t)): the curvature
        # of log g at 0, which an exponential lacks (computed only here, so
        # an ordinary kernel takes exactly the symbolic steps it took before)
        try:
            curv = complex(CDF(SR(sr_expr.diff(tau_sym, 2) / sr_expr
                                  - (sr_expr.diff(tau_sym) / sr_expr) ** 2)
                           .subs({tau_sym: 0})))
        except Exception:
            return None
        rate = math.sqrt(abs(curv))
    unit = 1.0 / rate if rate > 0 and math.isfinite(rate) else 1.0
    try:
        for t0 in _EXP_MODE_PROBES:
            t = t0 * unit
            fit = C * cmath.exp(lam * t)
            if abs(fit) < 1e-250 * abs(C):
                continue                    # underflow: nothing to compare
            g_t = complex(CDF(SR(sr_expr.subs({tau_sym: t}))))
            if abs(g_t - fit) > 1e-12 * max(abs(g_t), abs(fit)):
                return None
    except Exception:
        return None
    return (C, lam)


#: Probe times of the single-exponential fit check, in units of the kernel's
#: own time scale (``_extract_exp_mode``).
_EXP_MODE_PROBES = (0.37, 1.9, 4.3)


def _trig_as_exp(expr):
    """``expr`` with cos, sin, cosh and sinh written as exponentials, so that
    a damped-oscillatory kernel ``r e^{-rt} cos(rt)`` splits into its two
    complex modes (``_extract_exp_modes``) on the pole route as on the
    state-space route (audit W4b, ACC-60: written with cos it was refused,
    at r = 2, or fitted as one real mode, at r = 2000).  The SAME object
    when none of them occurs (an operator walk: nothing is built for a
    kernel without a trig factor)."""
    from sage.all import cos, sin, cosh, sinh, exp, I
    trig = (cos, sin, cosh, sinh)
    stack = [expr]
    while stack:
        x = stack.pop()
        op = x.operator()
        if op is None:
            continue
        if any(op is f for f in trig):
            break
        stack.extend(x.operands())
    else:
        return expr
    w = SR.wild(0)
    return expr.subs({cos(w): (exp(I * w) + exp(-I * w)) / 2,
                      sin(w): (exp(I * w) - exp(-I * w)) / (2 * I),
                      cosh(w): (exp(w) + exp(-w)) / 2,
                      sinh(w): (exp(w) - exp(-w)) / 2})


def _extract_exp_modes(sr_expr, tau_sym, max_modes=16):
    """``[(C_k, λ_k), ...]`` with ``sr_expr == Σ_k C_k exp(λ_k tau_sym)``
    (distinct λ_k, numeric), or ``None`` when the expression is not a
    finite exponential sum (a polynomial prefactor ``τ e^{-τ}``, a Gaussian,
    ...).  Each additive term is fitted by ``_extract_exp_mode`` (which
    checks the fit), equal rates are merged, and the sum is checked again
    at three points."""
    import cmath
    try:
        expr0 = SR(sr_expr)
        expr = _trig_as_exp(expr0)
        from_trig = expr is not expr0
        expr = expr.expand()
    except Exception:
        return None
    try:
        from sage.symbolic.operators import add_vararg
        terms = (list(expr.operands()) if expr.operator() is add_vararg
                 else [expr])
    except Exception:
        terms = [expr]
    if len(terms) > max_modes:
        return None
    acc = {}
    for term in terms:
        mode = _extract_exp_mode(term, tau_sym)
        if mode is None:
            return None
        C, lam = mode
        key = (round(lam.real, 14), round(lam.imag, 14))
        C0, lam0 = acc.get(key, (0j, lam))
        acc[key] = (C0 + C, lam0)
    modes = [(C, lam) for (C, lam) in acc.values() if C != 0]
    if not modes or _kernel_rates_clustered(modes):
        return None
    if from_trig and any(lam.real >= 0 for (_C, lam) in modes):
        # a trig kernel that does not decay (cos(w t), cosh(t)): refused as
        # before the rewrite (``_trig_as_exp``) existed
        return None
    # probe on every mode's own time scale (ACC-60: fixed times underflow
    # for fast modes and check nothing)
    rates = sorted({abs(lam) for (_C, lam) in modes if abs(lam) > 0}) or [1.0]
    try:
        for t in sorted({t0 / r for r in rates for t0 in _EXP_MODE_PROBES}):
            g_t = complex(CDF(SR(SR(sr_expr).subs({tau_sym: t}))))
            fit = sum(C * cmath.exp(lam * t) for C, lam in modes)
            scale = sum(abs(C * cmath.exp(lam * t)) for C, lam in modes)
            if abs(g_t - fit) > 1e-12 * max(scale, 1e-300):
                return None
    except Exception:
        return None
    return modes


#: Kernel rates closer than this (relative) count as one repeated rate:
#: ``t e^{μt}`` terms, which a mode sum cannot hold (a numerically split
#: double root gives residues ~1/gap of rounding size); such a kernel is
#: refused by name.  Between separated rates the ±1/gap coefficients cancel
#: in double precision with an error ~1e-15/gap² (measured against the
#: Markov embedding: 1.6e-13 at an 11 % gap, 1.6e-12 at 6 %, 4e-12 at 1.4 %,
#: 4e-10 at 0.14 %), so 10 % keeps
#: an accepted kernel at the 1e-12 level.
KERNEL_RATE_CLUSTER_RTOL = 1e-1


def _kernel_rates_clustered(modes):
    """True when two kernel rates are within ``KERNEL_RATE_CLUSTER_RTOL``."""
    lams = [complex(lam) for (_c, lam) in modes]
    for i in range(len(lams)):
        for j in range(i):
            if abs(lams[i] - lams[j]) <= KERNEL_RATE_CLUSTER_RTOL * max(
                    abs(lams[i]), abs(lams[j]), 1e-300):
                return True
    return False


def _modes_from_freq_image(ft_fn, num_params=None, direct=False):
    """The exponential modes ``[(C, λ)]`` of a causal kernel given by its
    rational frequency image ``ĝ(ω)`` (``ft_fn(omega)``; the codebase's
    convention ``ĝ(ω) = ∫ g(t) e^{-iωt} dt``, retarded poles in Im ω > 0):
    each simple pole ω_α gives ``λ_α = i ω_α`` and ``C_α = i Res ĝ(ω_α)``.
    ``None`` when the image is not a proper rational function of ω with
    simple poles in the upper half plane (a δ part, a repeated or an
    acausal pole, a non-rational image).

    ``direct=True`` (item 56): also a kernel with a direct term, ``ĝ = c +
    (strictly proper)``, i.e. ``g = c δ + Θ Σ C_α e^{λ_α t}`` (the all-pass
    Pade image of a bare delay); returns ``(c, modes)`` (``c = 0`` for a
    strictly proper image; the modes as above, of the strictly proper
    remainder), or ``None``."""
    from sage.all import CDF as _CDF, PolynomialRing as _PR
    w = SR.var('convkernelomegatd')
    try:
        g = SR(ft_fn(w))
        if num_params:
            g = g.subs(num_params)
        g = g.rational_simplify()
        if set(g.variables()) - {w}:
            return None
        R = _PR(_CDF, 'w_')
        num = R(SR(g.numerator()).polynomial(_CDF).list())
        den = R(SR(g.denominator()).polynomial(_CDF).list())
    except Exception:
        return None
    c_dir = 0j
    if direct and den.degree() >= 1 and num.degree() == den.degree():
        # the direct term: the constant polynomial part c = lc(num)/lc(den)
        c_dir = complex(num.leading_coefficient()) / complex(
            den.leading_coefficient())
        # the remainder, its (cancelled) top coefficient dropped exactly
        num = R((num - num.leading_coefficient() / den.leading_coefficient()
                 * den).list()[:den.degree()])
    if den.degree() < 1 or num.degree() >= den.degree():
        return None                         # δ part or not a pole sum
    roots = den.roots(multiplicities=True)
    if sum(mult for (_r, mult) in roots) != den.degree() or any(
            mult != 1 for (_r, mult) in roots):
        return None
    dden = den.derivative()
    modes = []
    for (om, _m) in roots:
        om = complex(om)
        if not om.imag > 0:
            return None                     # not a causal (retarded) kernel
        res = complex(num(om)) / complex(dden(om))
        modes.append((1j * res, 1j * om))
    if not modes or _kernel_rates_clustered(modes):
        return None                         # repeated / near-repeated pole
    return (c_dir, modes) if direct else modes


def _kernel_direct_split(att, num_params=None):
    """Item 56: ``(c, modes)`` for a conductance kernel attachment whose
    frequency image has a direct term, ``ĝ = c + (strictly proper)`` with
    ``c != 0`` (``g = c δ + Θ Σ C_α e^{λ_α t}``: the all-pass Pade (n, n)
    image of a bare hard delay at a vertex leg), or ``None`` (no frequency
    image, a time image, no direct term, or not such a pole sum)."""
    if att.get('kernel_td_fn') is not None:
        return None
    ft_fn = att.get('kernel_ft_fn')
    if ft_fn is None:
        return None
    out = _modes_from_freq_image(ft_fn, num_params, direct=True)
    if out is None or out[0] == 0:
        return None
    return out


def _kernel_modes(att, num_params=None):
    """The exponential modes of a conductance kernel attachment's
    time-domain image (Heaviside stripped, ``num_params`` substituted),
    or ``None`` (no time-domain image, or not a finite exponential sum).
    A kernel declared only by its frequency image (``kernel_ft_fn``) is
    read from the poles of that image (``_modes_from_freq_image``)."""
    from sage.all import heaviside as _sage_heaviside
    td_fn = att.get('kernel_td_fn')
    if td_fn is None:
        ft_fn = att.get('kernel_ft_fn')
        return (_modes_from_freq_image(ft_fn, num_params)
                if ft_fn is not None else None)
    tau = SR.var('_conv_kernel_modes_tau_td_')
    try:
        g = SR(td_fn(tau)).substitute_function(_sage_heaviside,
                                                lambda _x: SR(1))
        if num_params:
            g = g.subs(num_params)
    except Exception:
        return None
    return _extract_exp_modes(g, tau)


#: Item 27: fold a conductance kernel that is a finite exponential sum into
#: the propagator of the line it sits on (``(g ∗ G)``, a pole sum: no extra
#: time variable).  Off: every kernel takes the leg-time path.  Read at call
#: time; env ``DAEDALUS_PHASE_J_DRESS_KERNELS=0`` switches it off at import
#: (``engine._env.flag``: any other word raises).
DRESS_CONV_KERNELS = _env.flag('DAEDALUS_PHASE_J_DRESS_KERNELS', True)

# Item 35 (F1): every modesum plan leaves out the pole tuples that take a
# mode whose residue is EXACTLY zero (no tolerance; ``_build_pruned_modesum_
# plan``).  Driven filters that do not feed back (block-triangular K) give
# such entries; at k = 3 one loop the full product n_poles^E was 1e2-1e3
# times larger than the nonzero one.  The values are bit-identical to the
# unpruned plan's.  (The switch ``DAEDALUS_PRUNE_ZERO_MODES`` was retired in
# audit W7; ``_build_modesum_plan(prune=False)`` still builds the unpruned
# plan.)

#: Item 35: a pruned plan replays the group order of the full product (so
#: that the group sums run in the unpruned order, bit for bit) when that
#: product has at most this many pole tuples; beyond it (an unpruned plan of
#: ~70 GB that could never be built) the groups keep the order of their
#: first NONZERO tuple (counter ``zero_mode_order_unreplayed``).
PRUNE_ORDER_MAX_TUPLES = 1 << 26


# ── Item 35: the Phase J memory budget ──────────────────────────────────
#: Bytes one pole tuple of a modesum plan holds: its plan entries (the
#: (C, lambdas) pair, alphas, gamma constant and slopes as Python complex:
#: ~800 B) plus the batched kernels' numpy rows (~270 B), measured at m = 6
#: (item 34).  Plans live as long as the result's callables.
PLAN_BYTES_PER_TUPLE = 1100

#: The pre-pass (``estimate_modesum_plans``) is an upper bound that can be
#: several times the plans actually built (δ-subsets that never build one:
#: 5x on a spike-only two-neuron Hawkes model, 9.3 M estimated vs 1.8 M
#: built).  It refuses outright only above this multiple of the budget (a
#: hopeless run, e.g. 108 GB of unpruned plans); below it the running guard
#: (``_plan_guard``, exact at each allocation) decides.
PHASE_J_PREPASS_REFUSE_FACTOR = 8


class PhaseJResourceError(MemoryError):
    r"""Phase J would hold more modesum-plan memory than the budget
    ``PHASE_J_MEM_BUDGET_GB`` (default: ``default_mem_budget``) allows: raised
    BEFORE the plan that crosses it is allocated (and, from
    ``compute_cumulants``, before Phase J starts when its estimate already
    crosses it), under every analyzer policy.  ``estimate``: the numbers
    (tuples, bytes, budget, the offending diagram / subset)."""

    def __init__(self, message, estimate=None):
        super().__init__(message)
        self.estimate = estimate


# ── the default memory budget (item 53) ───────────────────────────────────
# A fixed 6 GB is more than a small machine can give one run on top of the
# Python + Sage baseline.  With the variable unset the budget is
#     min(MEM_BUDGET_CAP_GB, MEM_BUDGET_RAM_FRACTION x RAM - this process now)
# and at least max(MEM_BUDGET_FLOOR_GB, MEM_BUDGET_FLOOR_RAM_FRACTION x RAM)
# (a refusal, not a crash, is then the usual outcome on a machine already
# short of memory).  compute_cumulants reads it once per run.  The same rule backs
# SPATIAL_MEM_BUDGET_GB.  An explicit value is used as given.
MEM_BUDGET_CAP_GB = 6.0
MEM_BUDGET_RAM_FRACTION = 0.6
MEM_BUDGET_FLOOR_GB = 0.5
#: ... and at least this fraction of RAM (capped at the cap): a large
#: machine running a large process keeps a usable budget (64 GB with 40 GB
#: held: 6 GB, not the 0.5 GB floor)
MEM_BUDGET_FLOOR_RAM_FRACTION = 0.1


def _ram_bytes():
    """Physical memory of the machine in bytes, or ``None``."""
    from engine.enumeration.prediagram_cache import _physical_memory
    return _physical_memory()


@_functools.lru_cache(maxsize=1)
def _libproc():
    import ctypes
    import ctypes.util
    return ctypes.CDLL(ctypes.util.find_library('proc')
                       or '/usr/lib/libSystem.B.dylib')


def process_footprint_bytes():
    """What this process holds now, in bytes, or ``None`` when unknown.
    macOS: ``phys_footprint`` (``proc_pid_rusage``), not the resident size,
    which leaves out compressed pages (6 GB resident was 18 GB of footprint
    in one measurement).  Linux: resident + swapped (``/proc/self/status``)."""
    import sys
    if sys.platform == 'darwin':
        try:
            import ctypes
            lib = _libproc()
            buf = ctypes.create_string_buffer(512)   # rusage_info_v2 + room
            if lib.proc_pid_rusage(_os.getpid(), 2, buf) != 0:
                return None
            # rusage_info_v2: 16-byte uuid, then uint64 fields; the 8th
            # (ri_phys_footprint) follows user/system time, the two wakeup
            # counts, pageins, wired and resident size
            return int.from_bytes(buf.raw[16 + 7 * 8:16 + 8 * 8], 'little')
        except Exception:
            return None
    try:
        total = 0
        with open('/proc/self/status') as fh:
            for line in fh:
                if line.startswith(('VmRSS:', 'VmSwap:')):
                    total += int(line.split()[1]) * 1024
        return total or None
    except (OSError, ValueError, IndexError):
        return None


def default_mem_budget(ram=None, footprint=None):
    """``(bytes, rule)``: the default budget of the memory guards and the
    sentence that says how it was set (``ram`` / ``footprint``: bytes, read
    from the machine when ``None``)."""
    cap = MEM_BUDGET_CAP_GB * 1e9
    ram = _ram_bytes() if ram is None else ram
    if not ram:
        return cap, (f'default {_fmt_gb(cap)} (the machine\'s RAM is '
                     'unknown)')
    footprint = process_footprint_bytes() if footprint is None else footprint
    fp = footprint or 0
    floor = min(cap, max(MEM_BUDGET_FLOOR_GB * 1e9,
                         MEM_BUDGET_FLOOR_RAM_FRACTION * ram))
    room = MEM_BUDGET_RAM_FRACTION * ram - fp
    budget = max(floor, min(cap, room))
    rule = (f'default min({_fmt_gb(cap)}, {MEM_BUDGET_RAM_FRACTION:g} x '
            f'{_fmt_gb(ram)} RAM - {_fmt_gb(fp)} already held by this '
            f'process{"" if footprint else " (unknown, taken as 0)"})'
            + (f', at least {_fmt_gb(floor)}' if room < floor else '')
            + f' = {_fmt_gb(budget)}')
    return budget, rule


def mem_budget(var, environ=None):
    """``(bytes or None, rule)`` of the memory guard read from the
    environment variable ``var`` (``engine._env.budget_gb``: a positive
    number of GB; ``None`` = no limit for ``off``, ``none``, ``inf`` or
    ``unlimited``; ``0`` and negative values raise -- they meant "no limit"
    here and "refuse everything" for the spatial guard, audit RED-05);
    unset or empty: the default rule (``default_mem_budget``)."""
    gb = _env.budget_gb(var, environ)
    if gb is None:
        return default_mem_budget()
    env = _os.environ if environ is None else environ
    raw = str(env[var]).strip().lower()
    if gb == float('inf'):
        return None, f'{var}={raw}'
    return gb * 1e9, f'{var}={raw}'


def phase_j_mem_budget_bytes(environ=None):
    """The budget in bytes from ``PHASE_J_MEM_BUDGET_GB``; unset: the
    default rule (``default_mem_budget``: at most 6 GB, less on a machine
    with little RAM); ``None`` (no limit) for ``off``, ``none``, ``inf`` or
    ``unlimited``."""
    return mem_budget('PHASE_J_MEM_BUDGET_GB', environ)[0]


_BUDGET_HOW = ('raise the budget with the environment variable '
               'PHASE_J_MEM_BUDGET_GB (in GB; e.g. '
               "os.environ['PHASE_J_MEM_BUDGET_GB'] = '24' before the run, or "
               "'off' for no limit) if the machine has that much free memory, "
               'or lower k / max_ell')


def _fmt_gb(nbytes):
    return f'{nbytes / 1e9:.2f} GB'


def _fmt_budget(budget, rule=None):
    """The budget in a refusal message, with how it was set (``rule``: from
    the ``mem_budget`` call that set it)."""
    if rule is None or rule.startswith('default'):
        # the default rule's sentence already ends in '= <budget>'
        return rule or _fmt_gb(budget)
    return f'{_fmt_gb(budget)} ({rule})'


import threading as _threading_budget

#: The running total of one Phase J run (``phase_j_memory_scope``): bytes of
#: the plans built so far, the budget, and what the run is.
_PLAN_MEM = {'active': 0, 'used': 0, 'tuples': 0, 'plans': 0,
             'budget': None, 'rule': None, 'what': ''}
_PLAN_MEM_LOCK = _threading_budget.Lock()


class phase_j_memory_scope:
    """Context for one Phase J run: plans built inside count against one
    running total (``budget``: bytes, default ``phase_j_mem_budget_bytes()``;
    ``what``: names the run in the message).  Outside any scope every plan is
    checked on its own against the budget.  Nested scopes share the outer
    total."""

    def __init__(self, budget=None, what='', use_env=True, rule=None):
        # ``use_env=False`` with ``budget=None``: no limit; ``rule``: how
        # the caller's budget was set (for the messages)
        if budget is None and use_env:
            self.budget, self.rule = mem_budget('PHASE_J_MEM_BUDGET_GB')
        else:
            self.budget = budget
            self.rule = rule or 'set by the caller'
        self.what = what
        self._outer = False

    def __enter__(self):
        with _PLAN_MEM_LOCK:
            if _PLAN_MEM['active'] == 0:
                _PLAN_MEM.update(used=0, tuples=0, plans=0,
                                 budget=self.budget, rule=self.rule,
                                 what=self.what)
                self._outer = True
            _PLAN_MEM['active'] += 1
        return self

    def __exit__(self, *exc):
        with _PLAN_MEM_LOCK:
            _PLAN_MEM['active'] -= 1
            self.used = _PLAN_MEM['used']
            self.tuples = _PLAN_MEM['tuples']
            self.plans = _PLAN_MEM['plans']
        return False


def _budget_now():
    """``(scoped, budget, rule, used)`` for a charge now: the running
    scope's, or outside any scope the environment's (``mem_budget``) with
    nothing used.  The caller holds ``_PLAN_MEM_LOCK``."""
    if _PLAN_MEM['active'] > 0:
        return True, _PLAN_MEM['budget'], _PLAN_MEM['rule'], _PLAN_MEM['used']
    budget, rule = mem_budget('PHASE_J_MEM_BUDGET_GB')
    return False, budget, rule, 0


def _plan_guard(n_tuples, n_full, where=None):
    """The running guard (item 35): account ``n_tuples`` pole tuples of the
    plan about to be built (``n_full``: before pruning) and raise
    ``PhaseJResourceError`` BEFORE it is allocated when the run's total (or,
    outside a ``phase_j_memory_scope``, the plan alone) would exceed the
    budget."""
    nbytes = n_tuples * PLAN_BYTES_PER_TUPLE
    with _PLAN_MEM_LOCK:
        scoped, budget, rule, used = _budget_now()
        if budget is not None and used + nbytes > budget:
            est = {'tuples': n_tuples, 'tuples_unpruned': n_full,
                   'bytes': nbytes, 'bytes_so_far': used,
                   'plans_so_far': _PLAN_MEM['plans'] if scoped else 0,
                   'budget_bytes': budget, 'where': where,
                   'run': _PLAN_MEM['what'] if scoped else ''}
            run = f' of {est["run"]}' if est['run'] else ''
            raise PhaseJResourceError(
                f'Phase J memory budget exceeded{run}: the next modesum '
                f'plan ({where or "diagram subset"}) holds {n_tuples:,} pole '
                f'tuples (~{_fmt_gb(nbytes)}; {n_full:,} before the '
                f'exact-zero modes were left out) on top of '
                f'{_fmt_gb(used)} in {est["plans_so_far"]} plans built so '
                f'far, over the budget of {_fmt_budget(budget, rule)}.  '
                f'Nothing was '
                f'allocated for it.  Options: {_BUDGET_HOW}.', est)
        if scoped:
            _PLAN_MEM['used'] += nbytes
            _PLAN_MEM['tuples'] += n_tuples
            _PLAN_MEM['plans'] += 1


def _plan_refund(n_tuples):
    """Take back a ``_plan_guard`` charge for a plan that was not kept (a
    pruned plan abandoned for the unpruned one)."""
    with _PLAN_MEM_LOCK:
        if _PLAN_MEM['active'] > 0:
            _PLAN_MEM['used'] -= n_tuples * PLAN_BYTES_PER_TUPLE
            _PLAN_MEM['tuples'] -= n_tuples
            _PLAN_MEM['plans'] -= 1

#: Relative gap below which a kernel rate counts as equal to a propagator
#: pole (the dressed line would need a ``Δt e^{λΔt}`` term): leg-time path,
#: unless the line's mode there is negligible (``_negligible_mode``: its
#: residue below ``DRESS_ZERO_RTOL`` of the line's and its integral weight
#: ``|C| / |Re λ|`` below ``DRESS_ZERO_RTOL`` of the line's L1 norm).  A
#: dressed line drops its negligible modes (rounding-level residues of poles
#: the propagator inherits from a kernel image with residue 0; they multiply
#: the pole-tuple count for nothing); a slow mode with a small residue has a
#: large weight and is kept.  The other lines of a diagram with a dressed
#: line drop their negligible modes by the same rule.  Every dropped mode is
#: counted
#: (``conv_dress_zero_modes_dropped``).
DRESS_RATE_RTOL = 1e-6
DRESS_ZERO_RTOL = 1e-12


def _mode_l1(C, lam):
    """``∫_0^∞ |C e^{λ t}| dt = |C| / |Re λ|`` (``inf`` for an undamped or
    growing mode)."""
    C, lam = complex(C), complex(lam)
    if C == 0:
        return 0.0
    return abs(C) / -lam.real if lam.real < 0 else float('inf')


def _negligible_mode(C, lam, sup_scale, l1_scale):
    """Whether the mode ``C e^{λ t}`` of a line is negligible: its residue at
    most ``DRESS_ZERO_RTOL`` of ``sup_scale`` (a bound of the line's sup
    norm) AND its integral weight at most ``DRESS_ZERO_RTOL`` of
    ``l1_scale`` (a bound of the line's L1 norm).  Both ratios are unchanged
    by a rescaling of time or of the fields."""
    C = complex(C)
    if C == 0:
        return True
    if not (math.isfinite(sup_scale) and math.isfinite(l1_scale)):
        return False        # an undamped mode somewhere: exact zeros only
    if abs(C) > DRESS_ZERO_RTOL * sup_scale:
        return False
    return _mode_l1(C, lam) <= DRESS_ZERO_RTOL * l1_scale


def _dress_kernels_on():
    flag = DRESS_CONV_KERNELS
    if flag is True or flag is False:
        return flag
    raise ValueError(f'final_integral.DRESS_CONV_KERNELS={flag!r}: expected '
                     f'True or False')


def _dress_edge_modes(ems, kernel_modes, direct=0):
    """The ``EdgeModeSum`` of ``(g ∗ G)(Δ) = ∫_0^Δ g(Δ − x) G(x) dx`` for
    ``G = D δ + Θ Σ_α C_α e^{λ_α x}`` (``ems``) and ``g = Θ Σ_k c_k
    e^{μ_k x}``:  ``D g(Δ) + Σ_{α,k} C_α c_k (e^{λ_α Δ} − e^{μ_k Δ}) /
    (λ_α − μ_k)``, no δ part (but see ``direct``).  Rates are merged when
    equal.  A λ_α equal
    to a μ_k (``DRESS_RATE_RTOL``) with a rounding-level residue is dropped
    as an exact zero; with a real residue the result is ``None`` (the line
    then takes the leg-time path).

    ``direct`` (item 56): the kernel's direct term ``c`` (``g = c δ + Θ Σ_k
    c_k e^{μ_k x}``, ``_kernel_direct_split``) adds ``c G``: ``c C_α`` at
    every ``λ_α`` and the δ part ``c D`` (nonzero only on a line with a δ
    part, a point-process field)."""
    acc = {}

    def add(C, lam):
        key = (lam.real, lam.imag)
        C0, lam0 = acc.get(key, (0j, lam))
        acc[key] = (C0 + C, lam0)

    D_ = complex(ems.delta_coeff)
    direct = complex(direct)
    kernel_modes = [(complex(c), complex(mu)) for (c, mu) in kernel_modes]
    # sup / L1 norm bounds of G and g (the δ parts count with their weights)
    G_sup = sum(abs(complex(C)) for (C, _l) in ems.modes)
    G_l1 = sum(_mode_l1(C, l) for (C, l) in ems.modes) + abs(D_)
    g_sup = sum(abs(c) for (c, _m) in kernel_modes)
    g_l1 = sum(_mode_l1(c, m) for (c, m) in kernel_modes) + abs(direct)
    # a rounding-level mode of the line: an exact zero (e.g. a pole of the
    # kernel's image that the propagator inherits with residue 0)
    line_modes = []
    for (C, lam) in ems.modes:
        C, lam = complex(C), complex(lam)
        if C != 0 and _negligible_mode(C, lam, G_sup + abs(D_), G_l1):
            _RUNTIME_COUNTERS['conv_dress_zero_modes_dropped'] += 1
            continue
        line_modes.append((C, lam))
    if direct != 0:
        for (C, lam) in line_modes:
            add(direct * C, lam)
    for (c, mu) in kernel_modes:
        if D_ != 0:
            add(D_ * c, mu)
        for (C, lam) in line_modes:
            if C == 0:
                continue
            if abs(lam - mu) <= DRESS_RATE_RTOL * max(abs(lam), abs(mu),
                                                      1e-300):
                return None                 # a real coincidence: Δt e^{λΔt}
            f = C * c / (lam - mu)
            add(f, lam)
            add(-f, mu)
    # A kernel rate where G has a zero (a pole of the kernel's image is a
    # zero of G = (1 - c ĝ)^{-1}-type propagators) cancels exactly: its
    # coefficient is rounding.  Drop it, as the voltage form never has it.
    # The scales bound g ∗ G itself (sup ≤ ‖g‖₁ sup G + |D| sup g, L1 ≤
    # ‖g‖₁ ‖G‖₁): the sum of the coefficients is inflated by the pairs
    # C c / (λ − μ) of a near-coincident rate, which cancel.
    out_sup = g_l1 * G_sup + abs(D_) * g_sup + abs(direct) * G_sup
    out_l1 = g_l1 * G_l1
    modes = []
    for (C, lam) in acc.values():
        if _negligible_mode(C, lam, out_sup, out_l1):
            if C != 0:
                _RUNTIME_COUNTERS['conv_dress_zero_modes_dropped'] += 1
            continue
        modes.append((C, lam))
    return EdgeModeSum(ri=ems.ri, pi=ems.pi,
                       delta_coeff=(direct * D_ if direct != 0 else 0j),
                       modes=tuple(modes))


def _sr_complex(z):
    """A Python/numpy complex as an SR number (``SR(complex)`` does not
    combine with symbolic expressions)."""
    from sage.all import I as _I
    z = complex(z)
    return SR(z.real) + SR(z.imag) * _I if z.imag else SR(z.real)


class _ScaledCoeff:
    """``coeff_fc`` scaled by a constant (an averaged δ record)."""
    __slots__ = ('_f', '_w')

    def __init__(self, f, w):
        self._f = f
        self._w = w

    def __call__(self, *args):
        return self._w * complex(self._f(*args))


def _conv_monomial_split(vertex_assignments, combined_prefactor):
    """Item 60 (bug 14): ``None`` when every conductance vertex's coefficient
    is a single kernel monomial (the slots follow from it); else the list of
    ``(prefactor, {vertex: term})`` over every choice of one term (grouped
    by the powers of the vertex's kernel symbols) per vertex, the prefactor's
    factor of each vertex replaced by its term; or a refusal reason (str)
    when the prefactor is not the product of the vertex coefficients times a
    factor free of kernel symbols."""
    groups = {}
    for v, vt in vertex_assignments.items():
        if not isinstance(vt, ConvVertexType) or not vt.kernel_attachments:
            continue
        syms = []
        for att in vt.kernel_attachments:
            if att['symbol'] not in syms:
                syms.append(att['symbol'])
        coeff = SR(vt.coefficient)
        e = coeff.expand()
        op = e.operator()
        terms = (e.operands() if op is not None
                 and getattr(op, '__name__', '') == 'add_vararg' else [e])
        by = {}
        for t in terms:
            key = tuple(int(t.degree(SR(z))) for z in syms)
            by[key] = by.get(key, SR(0)) + t
        groups[v] = (coeff, [by[k] for k in sorted(by)])
    if not groups or all(len(ts) == 1 for _c, ts in groups.values()):
        return None
    prod = SR(1)
    for coeff, _ts in groups.values():
        prod = prod * coeff
    rest = (SR(combined_prefactor) / prod).simplify_rational()
    kern = {str(att['symbol']) for vt in vertex_assignments.values()
            if isinstance(vt, ConvVertexType)
            for att in (vt.kernel_attachments or ())}
    if any(str(x) in kern for x in rest.variables()):
        return ('conductance vertex coefficients mixing plain and kernel '
                'terms: the diagram prefactor is not their product times a '
                f'kernel-free factor ({combined_prefactor}); not integrated')
    verts = list(groups)
    out = []
    for combo in itertools.product(*[groups[v][1] for v in verts]):
        pf = rest
        for t in combo:
            pf = pf * t
        out.append((pf, dict(zip(verts, combo))))
    return out


def _sum_conv_monomials(split, kw):
    """``integrate_diagram`` as the sum over the coefficient terms of
    ``_conv_monomial_split`` (each with its own prefactor and kernel slots);
    the closure carries the parts as ``phase_j_placements`` of weight 1 (the
    state-space triples, S4b)."""
    results = []
    for pf, mono in split:
        if SR(pf).is_zero():
            continue
        r = integrate_diagram(combined_prefactor=pf, _conv_monomial=mono, **kw)
        if r.get('status') != 'ok':
            return r
        results.append(r)
    if not results:
        return {'status': 'failed', 'contribution': None,
                'reason': 'every coefficient term of the conductance '
                          'vertices is zero'}
    fns = [r['contribution'] for r in results]

    def contribution(*ext_time_values):
        return sum(complex(f(*ext_time_values)) for f in fns)
    contribution.phase_j_placements = tuple((1.0, f) for f in fns)
    first = results[0]
    out = {k: (None if k == 'stripped_integrand' else first[k])
           for k in first.keys()}
    out['contribution'] = contribution
    for key in ('delta_contributions', 'delta_contributions_multi'):
        out[key] = [rec for r in results for rec in (r.get(key) or [])]
    for key in ('n_subsets_evaluated', 'n_delta_contributions',
                'n_shotnoise_skipped', 'n_delta_subsets_dropped'):
        out[key] = sum(int(r.get(key) or 0) for r in results)
    out['subset_diagnostics'] = [d for r in results
                                 for d in (r.get('subset_diagnostics') or [])]
    out['conv_monomials'] = len(results)
    return out


def _average_conv_placements(choices, kw):
    """``integrate_diagram`` for a diagram with a conductance vertex whose
    kernel slots can be filled in several distinct ways (a mixed leg tuple,
    see section 2c): build every placement and average them.  The diagram
    weight M counts the copies of a leg as identical, so M times this
    average is the sum over the Wick contractions."""
    combos = list(itertools.product(*[pl for (_key, pl) in choices]))
    keys = [key for (key, _pl) in choices]
    if len(combos) > 256:
        return {'status': 'failed', 'contribution': None,
                'reason': (f'conductance vertices: {len(combos)} distinct '
                           f'kernel-slot placements (more than 256)')}
    results = []
    for combo in combos:
        r = integrate_diagram(_conv_assign=dict(zip(keys, combo)), **kw)
        if r.get('status') != 'ok':
            return r
        results.append(r)
    w = 1.0 / len(results)
    fns = [r['contribution'] for r in results]

    def contribution(*ext_time_values):
        return w * sum(complex(f(*ext_time_values)) for f in fns)
    # the placements and their weight, for the state-space triples (S4b)
    contribution.phase_j_placements = tuple((w, f) for f in fns)

    first = results[0]
    out = {k: (None if k == 'stripped_integrand' else first[k])
           for k in first.keys()}            # no lazy display build
    out['contribution'] = contribution
    for key in ('delta_contributions', 'delta_contributions_multi'):
        recs = []
        for r in results:
            for rec in r.get(key) or []:
                rec = dict(rec)
                rec['coeff_fc'] = _ScaledCoeff(rec['coeff_fc'], w)
                recs.append(rec)
        out[key] = recs
    for key in ('n_subsets_evaluated', 'n_delta_contributions',
                'n_shotnoise_skipped', 'n_delta_subsets_dropped'):
        out[key] = sum(int(r.get(key) or 0) for r in results)
    out['subset_diagnostics'] = [d for r in results
                                 for d in (r.get('subset_diagnostics') or [])]
    out['conv_placements'] = len(results)
    return out


def _conv_kernel_unsupported(att):
    """Why a conductance kernel's time-domain image cannot be integrated as
    a causal kernel (a message), or ``None``.  The Heaviside factor is
    stripped and replaced by the Δt > 0 row, so only ``heaviside(t)`` itself
    is allowed (a delay ``heaviside(t − d)`` would be lost); a δ part is
    allowed only as the whole kernel (``c·δ(t)``)."""
    from sage.all import heaviside, dirac_delta
    td_fn = att.get('kernel_td_fn')
    if td_fn is None:
        return None
    t = SR.var('_conv_shape_t_td_')
    w = SR.wild(0)
    try:
        g = SR(td_fn(t))
        for h in g.find(heaviside(w)):
            if not bool((h.operands()[0] - t).is_zero()):
                return (f'the kernel {att["symbol"]} has a delayed step '
                        f'{h}; only heaviside(t) is supported')
        if g.has(dirac_delta(w)) and _conv_kernel_local_weight(att) is None:
            return (f'the kernel {att["symbol"]} mixes a δ part with a '
                    f'smooth part; not supported')
    except Exception:
        return None
    return None


def _conv_kernel_local_weight(att, num_params=None):
    """The weight ``c`` of a conductance kernel attachment ``att`` whose
    time-domain image is ``c·δ(t)`` (``Conv(c·δ, n) = c·n``: no delay), or
    ``None`` for a delayed kernel (or one without a time-domain image)."""
    from sage.all import dirac_delta
    td_fn = att.get('kernel_td_fn')
    if td_fn is None:
        return None
    t = SR.var('_conv_local_t_td_')
    try:
        g = SR(td_fn(t))
        if num_params:
            g = g.subs(num_params)
        if not g.has(dirac_delta(t)):
            return None
        c = (g / dirac_delta(t)).simplify()
        if c.has(dirac_delta(t)) or t in c.variables():
            return None
        return c
    except Exception:
        return None


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


#: The m = 2 analytic polygon route (``_integrate_2d_polygon_modesum``);
#: ``False`` sends every m = 2 subset to the quadrature fallback (tests).
USE_POLYGON_M2_INTEGRATOR = True


def _env_truthy(name, environ=None):
    """An on/off switch, default off (``engine._env.flag``)."""
    return _env.flag(name, False, environ)


# ── Θ(0) convention for constant constraint rows (M1; plan §2.4, §3.1 L0) ──
# After δ-elimination a smooth edge can have Δt ≡ const (both endpoints
# merged by δ edges, or the edge parallel to a δ edge), i.e. a constraint row
# with a zero normal: a_int ≡ 0, so the row reads ``c_eff > 0`` with
# ``c_eff = c0 + Σ_j a_ext_j t_j``.  Its Heaviside is Θ(c_eff).  The package
# convention is the Itô one, Θ(0) = 0 -- the left limit the k=2 τ = 0 grid
# point takes (since Oct 2026 by evaluating AT the tie with the rule below;
# ``api.compute._tau0_mode``).  ``_const_row_verdict`` is
# the ONE place that applies it; every integrator (m=0, m=1, m=2 polygon,
# m≥3 poset, DBM, scipy fallback) calls it.
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
#   (the τ grid's τ = 0 point, evaluated here at the tie), and for k ≥ 3 its natural
#   extension (Θ(0) = 0 on both orientations would drop both and give a
#   value that is neither limit);
# * any other constant row with c_eff == 0.0 (no row context, or an
#   unusual row shape) is Θ(0) = 0.
THETA_AT_ZERO_CONST_ROW = 0
# A row has a zero normal when every |a_int_j| <= _ROW_COEF_ATOL.  The δ-solve
# produces small-integer coefficients, so real rows are exactly 0 or ±1.
_ROW_COEF_ATOL = 1e-12
# Emptiness of an m=2 region is decided structurally and exactly too (no
# area tolerance: a genuine strip of any width, e.g. between external times
# 1e-12 apart, is integrated).  See ``boxfree.region_2d``: two rows with
# exactly opposite normals whose constants sum to <= 0 (the 2-cycle rows
# s1 > s0, s0 > s1, or a strip between two coincident external times,
# ``_opposite_rows_empty``) empty the region, and so does a region of
# measure zero.

# The constant-row rule above is the only one: the pre-M1 'legacy_clip'
# mode (``THETA0_CONST_ROW_MODE``, ``DAEDALUS_PHASE_J_THETA0_CONST_ROW``) was
# retired in audit W7.

# ── Structural zeros (M2a; plan §3.1 L1) ──
# Two kinds of δ-subset work that is identically zero are always skipped
# (the switch ``STRUCTURAL_ZEROS`` / ``DAEDALUS_PHASE_J_STRUCTURAL_ZEROS``
# was retired in audit W7):
#   * ``_integrate_polytope`` (the scipy fallback, whoever calls it: an
#     analytic bail, the SR path of a non-analytic subset, a direct caller)
#     returns 0j at once when the rows contain a directed cycle of
#     unit-coefficient pair rows whose shifts sum to <= 0 in exact
#     arithmetic (``_region_structurally_empty``; counter
#     ``polytope_empty_cycle``).  The strict inequalities around such a
#     cycle cannot all hold, so the region is empty.  (Before M2a the
#     fallback handed it to scipy.nquad anyway, which could overflow
#     evaluating fast poles outside the region, plan Appendix C.2.)
#   * a δ-subset that keeps smooth an edge whose smooth part is provably
#     zero (a pure-δ entry, or an entry that vanishes: every stored pole
#     residue is an exact zero (``_smooth_part_is_exact_zero``) AND the
#     ``G_ft`` entry -- the propagator before any pole filtering -- contains
#     no ω once ``num_params`` are substituted
#     (``_entry_is_provably_instantaneous``); without ``G_ft`` the stored
#     residues decide) is never built (``_forced_delta_edges``; counter
#     ``forced_delta_pruned``) -- the mirror image of the forced-smooth
#     edges (zero δ part) that the subset enumeration already skips.  A
#     propagator WITHOUT any pole is decided from ``G_ft`` alone: an entry
#     is a pure δ only if it contains no ω once ``num_params`` are
#     substituted; if a diagram uses an entry that is not provably a pure
#     δ, its smooth part was dropped with a non-retarded mode (λ = 0 at
#     q = 0 for a massless field) and ``PoleFreePropagatorError`` is raised
#     (before M2a: a bare IndexError).
# Both are value-neutral by construction: every skipped piece contributes an
# exact zero.

# ── The quadrature fallback (M2b; plan §3.1 L1b) ──
# Every ``_integrate_polytope`` region that reaches scipy quadrature (m >= 1,
# after the constant-row verdicts and the structural-zero test) is
# integrated by ``_integrate_polytope_hardened``: nested
# ``scipy.integrate.quad`` over the region's OWN bounds (a
# difference-bound-matrix closure of its rows, not a ±200 / ±50 box), the
# innermost variable in closed form when the integrand's modes are known,
# with an empty interval answered 0 without sampling, the integrand
# Heaviside-filtered, breakpoints at the external times, at the kinks of the
# inner bounds and geometrically towards the ends of wide panels, epsabs =
# ``NQUAD_EPSABS_FACTOR`` × a rigorous bound of |integrand| from the modes
# and epsrel = ``NQUAD_EPSREL``.  A direction that the rows leave open is
# truncated at ``NQUAD_TAIL_K`` / κ_min from its finite end (κ_min: the
# slowest decay rate of the modes), or, without a decay certificate, at
# ±200.  See that function for the details and the counters.  Only a region
# with a non-finite row shift (which the hardened routine declines) reaches
# the default-tolerance scipy.nquad routines (``_integrate_1d_polytope``,
# ``_integrate_2d_polytope``, ``_integrate_nd_polytope``).  (The switch
# ``NQUAD_HARDENED`` / ``DAEDALUS_PHASE_J_NQUAD_HARDENED``, which selected
# those routines for every region, was retired in audit W7.)

# ── Poset lower-bound inheritance and the exact DBM route (M3; plan §3.1
# L7, problem P3) ──
# Two rules for the m≥3 subsets (the switch ``USE_DBM_FALLBACK`` /
# ``DAEDALUS_PHASE_J_DBM`` was retired in audit W7):
#   * P3, the inheritance rule.  The poset path integrates every variable
#     from ONE shared scalar lower bound L.  A variable without a scalar
#     lower of its own may inherit L only through a predecessor in the
#     transitive closure of the strict order that has one (s_v > s_u > L).
#     A poset with a variable that has neither extends below L; the poset
#     path bails on it (``'poset_lower_not_inherited'``; counter
#     ``poset_lower_not_inherited``) instead of cutting that region off at L.
#   * The DBM route.  Every m≥3 subset whose poset path bails (for any
#     reason: the P3 rule above, unequal scalar lowers, shifted order rows,
#     an order cycle with a positive total shift, a divergent or overflowing
#     chain) is integrated exactly by ``_integrate_subset_dbm``
#     (``dbm_integral.py``: case-split Fourier-Motzkin elimination of the
#     difference constraints, log-domain terms), box-free.  Constant rows
#     are decided by ``_const_row_decision`` (Θ(0) = 0 and the external-time
#     tie order); every other row enters the region as it is.  A subset the
#     DBM cannot take (a row that is not a difference row, e.g. a ConvVertex
#     3-term row; rows and modes that do not match; an overflowing term; a
#     divergent region; a closed form whose rounding error estimate exceeds
#     max(1e-10 |value|, 1e-14 × the integrand's scale) even at
#     ``dbm_integral.MP_DPS`` digits) goes on to ``_integrate_polytope``
#     (counters ``dbm_declined_*``).


#: Item 27: a DBM closed form that fails the conditioning test is
#: re-evaluated in ``dbm_integral.MP_DPS``-digit arithmetic before the
#: subset may fall to the quadrature fallback (read at call time; env
#: ``DAEDALUS_PHASE_J_DBM_MP=0`` switches it off at import; any other word
#: than an on/off value raises).  Acts only inside the DBM route.
USE_DBM_HIGH_PRECISION = _env.flag('DAEDALUS_PHASE_J_DBM_MP', True)


def _dbm_high_precision_on():
    flag = USE_DBM_HIGH_PRECISION
    if flag is True or flag is False:
        return flag
    raise ValueError(f'final_integral.USE_DBM_HIGH_PRECISION={flag!r}: '
                     f'expected True or False')


# ───────────────────────────────────────────────────────────────────────
# Physical margin of the scipy fallback's box (``USE_POSET_CAP_MATCH_SCIPY``)
# ───────────────────────────────────────────────────────────────────────
# The pre-0.2.0 chain routes closed an unbounded lower side at
# ``min(0, free_ext_vals) − POSET_PHYSICAL_MARGIN``; the box-free routes have
# no margin.  It survives as the alternative half-width of the scipy
# fallback's box (``_nquad_outer_cap``).
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
    # M4: the chain memo (``_chain_dd_log``); ``evictions`` counts the
    # table clears at the size cap:
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
    # ── M0.1 observational counters (the internal Phase J plan (not distributed)
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
        'divergent': 0, 'dd_overflow': 0, 'dd_unsupported': 0,
    },
    # Constraint rows with every |a_int| <= 1e-12 (a smooth edge whose Δt is
    # constant after δ-elimination) processed by ``_polygon_from_2d_
    # constraints`` or ``_extract_causal_poset``.
    'zero_normal_rows_seen': 0,
    # ── M1 Θ(0) counters (plan §2.4 item 8).  One count per
    # ``_const_row_verdict`` call (i.e. per zero-normal row per
    # evaluation):
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
    # builds it.
    'theta0_subsets_pruned': 0,
    # m=2 regions that are empty or degenerate by an exact test, answered 0
    # (``_polygon_region_dd``): two rows with exactly opposite normals whose
    # constants sum to <= 0 (e.g. the 2-cycle rows s1 > s0, s0 > s1), or a
    # region of measure zero.
    'polygon_zero_area': 0,
    # m≥3 posets answered 0 without integration: a constant EMPTY row, or a
    # directed cycle of order rows whose shifts sum to <= 0 (strict
    # inequalities around such a cycle cannot all hold: empty).
    'poset_empty_const': 0,
    'poset_empty_cycle': 0,
    # ── M2a structural-zero counters ──
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
    # L2: ``integrate_diagram`` calls served from the
    # per-call ``PropagatorTD``; calls handed a ``PropagatorTD`` that no
    # longer matched their propagator data / ``num_params`` (built locally);
    # lazy ``build_G_t_matrix`` builds and per-entry residue-table builds.
    'setup_prop_td_used': 0,
    'setup_prop_td_stale': 0,
    'setup_prop_td_g_t_builds': 0,
    'setup_prop_td_entry_builds': 0,
    # L3: edge ``smooth_factor`` SR objects whose
    # build was deferred / built on demand, and the same for the display-only
    # ``stripped_integrand``.
    'setup_lazy_sr_deferred': 0,
    'setup_lazy_sr_built': 0,
    'setup_lazy_display_deferred': 0,
    'setup_lazy_display_built': 0,
    # L5 (``engine.diagrams.symmetry``): automorphism-order memo
    # hits / misses / table clears, and the ``external_wick_compensation``
    # calls skipped because the diagram has a single Wick mapping.
    'setup_aut_memo_hits': 0,
    'setup_aut_memo_misses': 0,
    'setup_aut_memo_evictions': 0,
    'setup_aut_wick_skipped': 0,
    # M10 (L4a): δ-edge eliminations solved by
    # exact rational linear algebra / handed to ``sage_solve`` because the
    # equation is not an exactly-rational linear form in the eliminated
    # variable; and, under ``VALIDATE_DELTA``, eliminations checked against
    # ``sage_solve`` (substitution and constraint rows exactly equal).
    'setup_delta_solve_fast': 0,
    'setup_delta_solve_fallback': 0,
    'setup_delta_validated': 0,
    # ── M2b hardened quadrature counters ──
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
    'nquad_hardened_floor_cut': 0,
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
    # ── M3 counters ──
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
    # item 27: ill-conditioned closed forms re-evaluated at MP_DPS digits
    'dbm_mp_attempted': 0,
    'dbm_mp_answered': 0,
    # item 27: conductance kernels folded into lines / exact-zero modes
    'conv_lines_dressed': 0,
    'conv_dress_zero_modes_dropped': 0,
    'conv_kernel_leg_edges': 0,
    # item 56: lines dressed with a kernel that has a direct term (c δ + a
    # pole sum), and such kernels refused on a kernel edge
    'conv_lines_dressed_direct': 0,
    'conv_kernel_direct_refused': 0,
    # item 35: exact-zero residue modes left out of the modesum plans
    # (modes, and the pole tuples of the full product that took one), the
    # plans built pruned, and pruned plans whose group order could not be
    # replayed from the full product (beyond ``PRUNE_ORDER_MAX_TUPLES``)
    'zero_modes_pruned': 0,
    'zero_mode_tuples_pruned': 0,
    'zero_mode_plans_pruned': 0,
    'zero_mode_order_unreplayed': 0,
    'zero_mode_order_mismatch': 0,
    # item 35 F2: pole-tuple groups whose folded coefficient is 0, skipped
    # by the batched poset kernels before any chain is computed
    'dd_batch_zero_groups_skipped': 0,
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
    # ── M7 counters (box-free routing; all 0 with the M7 flags off) ──
    # subset evaluations answered on the DD kernels, per path:
    'dd_polygon': 0, 'dd_interval': 0, 'dd_poset': 0, 'dd_grouped_m0': 0,
    # ... of which the region was unbounded and integrated exactly (m=2
    # ideal-vertex fan / m≥3 chain with L = -inf or U = +inf):
    'polygon_exact_unbounded': 0,
    'poset_semiinf_chains': 0,
    # DD-path bails: the integral itself overflows (log|value| > 700, the
    # one-sided final-exponent test); a shape the box-free kernels do not
    # take (an open-top chain with intermediate uppers, a chain state that
    # underflowed); an open direction that does not decay (the divergence
    # policy is M8's):
    'dd_overflow': 0,
    'dd_unsupported': 0,
    'divergent_subsets': 0,
    # M8: pole-tuple groups that fail the convergence certificate but whose
    # merged coefficient is exactly 0 (dropped, not divergent):
    'divergent_cancelled': 0,
    # Error accounting of the box-free subset sums (``_dd_accuracy``): the
    # worst estimated-error / accepted-error ratio, and the subsets above 1.
    'dd_err_ratio_max': 0.0,
    'dd_ill_conditioned': 0,
    # D2 (G6 of ``api.analyze``): the summed absolute rounding-error
    # estimates (``COND_ERR_FACTOR`` x the expansion's magnitude) of the
    # box-free subset sums and of the answered DBM closed forms, for the
    # total-relative accuracy bound at the ``compute_cumulants`` level.
    # Bookkeeping only.
    'dd_err_abs_sum': 0.0,
    'dbm_err_abs_sum': 0.0,
    # DBM calls without a box, with the slow-mode box ±max(200, 40/κ_min)
    # wider than ±200 (the pre-0.2.0 DD route, retired in audit W7: always
    # 0), and declined as divergent (box-free):
    'dbm_box_free': 0,
    'dbm_slow_mode_box': 0,
    'dbm_declined_divergent': 0,
    # legacy ``_chain_with_intermediate_uppers`` evaluations (M6 F7) in which
    # at least one cut case returned None while others did not, and those
    # returning a finite value beyond e^700 (that evaluator was retired in
    # audit W7: always 0):
    'chain_uppers_legacy_partial': 0,
    'chain_uppers_legacy_huge': 0,
}

_NQUAD_FALLBACK_REASONS = (
    'polygon_guard', 'polygon_other',
    'poset_extract_none', 'poset_no_extension', 'poset_chain_none',
    'other',
    # M8: the box-free (DD) routes' bails, from any route (``_DD_BAILS``)
    'divergent', 'dd_overflow', 'dd_unsupported',
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
#   'conv_pseudo'  a ConvVertex kernel edge Δt = anchor − s > 0 (the kernel
#                  g(Δt) is causal: edge semantics);
#   'noise_box'    a NoiseSource τ_v box row, ±TAU_KERNEL_CAP;
#   'conv_tau'     a ConvVertex τ box row ([0, TAU_KERNEL_CAP)); no longer
#                  produced (the kernel delay is an edge since item 27).
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
    if reason is not None:
        # M8: the DD routes' bails (``_DD_BAILS``: <route>_divergent,
        # <route>_dd_overflow, <route>_dd_unsupported)
        for suffix, key in (('_divergent', 'divergent'),
                            ('_dd_overflow', 'dd_overflow'),
                            ('_dd_unsupported', 'dd_unsupported')):
            if reason.endswith(suffix):
                return key
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
# Divergence policy (M8; plan §3.1 L5)
# ───────────────────────────────────────────────────────────────────────
# The box-free routes certify convergence before they integrate: every open
# direction must decay (m = 1: the sign of Re α on an infinite side; m = 2:
# Re(α·r) < -tol on every recession ray, and no line in the region; m >= 3:
# every prefix sum (L = -inf) or suffix sum (open top) with Re beyond tol;
# the DBM checks its own eliminations).  A pole-tuple group that fails it but
# whose merged coefficient is EXACTLY zero (Σ C e^{γ - γ_max} == 0 in floating
# point, e.g. a zero-rate mode with opposite residues) has an identically
# zero integrand and is dropped (counter ``divergent_cancelled``; the m >= 3
# chain fold and the grouped path merge such groups before any chain is
# evaluated).  Any other failure is a divergent subset (counter
# ``divergent_subsets``, bail ``<route>_divergent``), and what happens next
# is the caller's divergence policy:
#
# * 'fallback' (default): the subset goes on to the next route, as every
#   other bail does (the DBM for m >= 3, then the boxed ``scipy.nquad``
#   fallback), and ONE ``PhaseJDivergenceWarning`` per process says that a
#   box-truncated value was used;
# * 'warn': the same, with one warning per divergent subset naming the
#   diagram, the subset, the external times and the reason;
# * 'raise': ``PhaseJDivergenceError`` with the same details.
#
# The policy is a keyword (``on_divergence``) of ``integrate_diagram``,
# ``compute_correction_td`` and ``grouped_integral.integrate_grouped_diagram``;
# ``None`` takes ``PHASE_J_ON_DIVERGENCE`` (read at call time; environment
# ``DAEDALUS_PHASE_J_ON_DIVERGENCE`` = fallback | warn | raise, default
# fallback).  The spatial certification (``pipeline_bridge``) passes
# 'fallback' explicitly, so a later change of the temporal default does not
# reach it.
_DIVERGENCE_POLICIES = ('fallback', 'warn', 'raise')
#: the bail reasons that are divergence verdicts (``_DD_BAILS`` and the DBM)
_DIVERGENT_BAIL_REASONS = frozenset((
    'polygon_divergent', 'interval_divergent', 'poset_divergent',
    'grouped_m0_divergent', 'grouped_m1_divergent', 'dbm_divergent'))


def _is_divergence_verdict(reason):
    """True for a bail reason that says the subset integral diverges
    (``_DIVERGENT_BAIL_REASONS``)."""
    return reason in _DIVERGENT_BAIL_REASONS


class PhaseJDivergenceError(ArithmeticError):
    """A Phase J subset integral diverges (an open direction in which the
    integrand does not decay) and the divergence policy is 'raise'."""


class PhaseJDivergenceWarning(UserWarning):
    """A Phase J subset integral diverges and was replaced by the boxed
    fallback (divergence policy 'fallback' or 'warn')."""


def _initial_on_divergence(environ=None):
    """The import-time value of ``PHASE_J_ON_DIVERGENCE`` from ``environ``
    (default ``os.environ``); an unknown value raises."""
    return _env.choice('DAEDALUS_PHASE_J_ON_DIVERGENCE',
                          _DIVERGENCE_POLICIES, 'fallback', environ)


PHASE_J_ON_DIVERGENCE = _initial_on_divergence()


def _divergence_policy(on_divergence=None):
    """The effective policy: ``on_divergence``, or ``PHASE_J_ON_DIVERGENCE``
    (call time) when ``None``; validated."""
    policy = PHASE_J_ON_DIVERGENCE if on_divergence is None else on_divergence
    if isinstance(policy, str) and policy in _DIVERGENCE_POLICIES:
        return policy
    raise ValueError(f'on_divergence={policy!r}: expected None or one of '
                     f'{_DIVERGENCE_POLICIES}')


# one copy, shared with ``api.save`` (audit W7, RED-07)
from engine._warn_helpers import (                    # noqa: E402
    ensure_warning_shown as _ensure_warning_shown)


_DIVERGENCE_FALLBACK_WARNED = [False]


def _divergence_verdict(on_divergence, reason, m, meta, free_vals):
    """Apply the divergence policy to a subset that bailed with the
    divergence verdict ``reason`` (``_DIVERGENT_BAIL_REASONS``) and is about
    to take the boxed fallback.  Returns nothing; raises
    ``PhaseJDivergenceError`` under 'raise'."""
    import warnings as _warnings
    policy = _divergence_policy(on_divergence)
    meta = meta or {}
    diagram = str(meta.get('diagram'))
    if len(diagram) > 200:              # the grouped path passes a list
        diagram = diagram[:197] + '...'
    where = (f"diagram {diagram} ({meta.get('source')}, "
             f"loop {meta.get('loop_number')}), subset "
             f"{meta.get('subset_id')!s} (m={m}), external times "
             f"{tuple(float(x) for x in free_vals)}")
    if policy == 'raise':
        raise PhaseJDivergenceError(
            f'Phase J: divergent subset integral ({reason}): {where}.  An '
            f'open direction of the integration region does not decay (a '
            f'pole with Re >= 0, or a non-edge row).')
    if policy == 'warn':
        _msg = (f'Phase J: divergent subset integral ({reason}) replaced by '
                f'the boxed fallback: {where}.')
        _ensure_warning_shown(PhaseJDivergenceWarning, message=_msg,
                              stacklevel=3)
        _warnings.warn(_msg, PhaseJDivergenceWarning, stacklevel=3)
        return
    if not _DIVERGENCE_FALLBACK_WARNED[0]:
        _DIVERGENCE_FALLBACK_WARNED[0] = True
        _msg = ('Phase J: a subset integral diverges (an open direction does '
                'not decay); its boxed fallback value is used (divergence '
                "policy 'fallback'; counter divergent_subsets; "
                "on_divergence='warn' names every subset, 'raise' stops).")
        _ensure_warning_shown(PhaseJDivergenceWarning, message=_msg,
                              stacklevel=3)
        _warnings.warn(_msg, PhaseJDivergenceWarning, stacklevel=3)


# ───────────────────────────────────────────────────────────────────────
# Per-subset observation hook (M0.1; the internal Phase J plan (not distributed) §4.1)
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
      ``None``; 'not_eligible' when none was attempted for an m≥1 call;
      for 'dbm', the reason the poset path passed the subset on),
      ``bail_category`` (the ``nquad_fallback_by_reason`` key, or ``None``
      when an analytic path answered), ``value`` (complex, exactly what
      the dispatch returns).
    * engine (S3): ``engine`` ('statespace' for ``path == 'ss'``, else
      'poles'); ``ss_fallback``: the reason when this pole-route evaluation
      is the state-space route's call-time fallback, else ``None``.
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
        'engine': 'statespace' if path == 'ss' else 'poles',
        'ss_fallback': meta.get('ss_fallback'),
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


def _opposite_rows_empty(rows):
    r"""True if two of the half-planes ``a0·s0 + a1·s1 + c > 0`` in ``rows``
    (``(a0, a1, c)`` triples, nonzero normals) have EXACTLY opposite
    normals, ``(a0_j, a1_j) == (−a0_i, −a1_i)``, and ``c_i + c_j <= 0.0``.
    Then ``a·s > −c_i`` and ``a·s < c_j`` cannot both hold: the region is
    empty (with ``c_i + c_j == 0.0`` the closed clip would leave a segment).
    Exact comparisons only; a pair with ``c_i + c_j > 0`` is a genuine
    strip, however thin.  (``boxfree.region_2d``'s first test.)"""
    n = len(rows)
    for i in range(n):
        a0, a1, ci = rows[i]
        for j in range(i + 1, n):
            b0, b1, cj = rows[j]
            if b0 == -a0 and b1 == -a1 and ci + cj <= 0.0:
                return True
    return False


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
                        m, n_ext, prune=None, where=None):
    """Pre-compute the τ-invariant per-pole-tuple data used by the
    analytic modesum integrators.

    ``prune`` (default ``True``): leave out every pole tuple
    that takes a mode whose residue is exactly zero (item 35, see
    ``_build_pruned_modesum_plan``); the plan's numbers are bit-identical
    to the unpruned plan's.  ``n_tuples_full`` / ``n_groups_full``: the
    tuple and group counts of the UNPRUNED plan (the group count exact below
    ``_DD_BATCH_MIN``, else a lower bound at or above it), which pick the
    batched or scalar kernels exactly as before.  Before any tuple is allocated the
    running memory guard (``_plan_guard``; ``where`` names the subset in
    its message) may raise ``PhaseJResourceError``.

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
    if prune is None:
        prune = True
    if prune and smooth_edge_modes:
        pruned = _build_pruned_modesum_plan(
            smooth_edge_modes, subset_constraint_data, m, n_ext, where)
        if pruned is not None:
            return pruned
    n_full = math.prod(len(ems.modes) for ems in smooth_edge_modes)
    _plan_guard(n_full, n_full, where)
    plan = _plan_from_tuples(
        tuple(_enumerate_pole_tuples(smooth_edge_modes)),
        len(smooth_edge_modes), subset_constraint_data, m, n_ext)
    plan['n_tuples_full'] = len(plan['pole_tuples'])
    plan['n_groups_full'] = len(plan['tuple_groups'])
    return plan


def _plan_coefficients(subset_constraint_data, m, n_ext):
    """Per-edge ``(a_int, a_ext, c0)`` of the subset rows as floats, the
    internal row padded to ``m`` and the external one to ``n_ext``."""
    a_int_per_edge = []
    a_ext_per_edge = []
    c0_per_edge = []
    for (a_int, a_ext, c0) in subset_constraint_data:
        a_int_per_edge.append(tuple(
            float(a_int[v]) if v < len(a_int) else 0.0 for v in range(m)))
        a_ext_per_edge.append(tuple(
            float(a_ext[j]) if j < len(a_ext) else 0.0
            for j in range(n_ext)))
        c0_per_edge.append(float(c0))
    return a_int_per_edge, a_ext_per_edge, c0_per_edge


def _plan_from_tuples(pole_tuples, n_smooth, subset_constraint_data, m,
                      n_ext):
    """The plan dict of ``_build_modesum_plan`` for the given pole tuples,
    groups in order of first occurrence."""
    # Pre-extract per-edge linear coefficients in floats.  The arity
    # of ``a_int_e`` is ``m`` and ``a_ext_e`` is ``n_ext``.  We tolerate
    # short ``a_int`` lists (degenerate constraints with no internal
    # coefficients) by padding with 0.0.
    a_int_per_edge, a_ext_per_edge, c0_per_edge = _plan_coefficients(
        subset_constraint_data, m, n_ext)

    alphas_per_tuple = []
    gamma_const_per_tuple = []
    gamma_slope_per_tuple_per_ext = []
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


def _mode_is_zero(C):
    """True iff the stored residue ``C`` is exactly zero: ``complex(C) ==
    0``, no tolerance (a residue that is merely tiny is kept; NaN is not
    zero; a value that does not convert is kept)."""
    try:
        return complex(C) == 0
    except (TypeError, ValueError, OverflowError):
        return False


class _KeptModes:
    """The ``modes`` of an edge with its exact-zero modes left out (all
    ``_enumerate_pole_tuples`` reads of an ``EdgeModeSum``)."""
    __slots__ = ('modes',)

    def __init__(self, modes):
        self.modes = modes


def _build_pruned_modesum_plan(smooth_edge_modes, subset_constraint_data,
                               m, n_ext, where=None):
    r"""The modesum plan of ``smooth_edge_modes`` over the pole tuples whose
    residues are all nonzero (item 35, F1), or ``None`` when no edge has an
    exact-zero mode, when an edge has no nonzero mode at all (the whole
    product is zero; the unpruned plan keeps the old behaviour), or when the
    group order cannot be replayed (the caller then builds the unpruned plan).

    Why it is exact: a tuple's coefficient is ``∏_e C_e``, so a tuple that
    takes an exact-zero residue has coefficient exactly 0, and every consumer
    of a plan already skips zero coefficients (``c != 0`` in the folds,
    ``term_const == 0`` in the interval / polygon / legacy poset loops,
    ``coefs != 0`` in the batched kernels).  Where the zeros come from: the
    residues are those of ``PropagatorTD.entry_modes`` (``complex(CDF(SR(
    C_mats[k][pi, ri])))``), and the propagator builder (``api/_propagator``)
    stores an exact 0 where an entry of ``G`` has no denominator, where a
    pole is not a root of the entry's own denominator (the exact tier-1
    membership test over QQ[i], item 31: a driven filter that does not feed
    back makes ``K`` block triangular, and the entries below it miss its
    poles), and, on the numerical cofactor tiers only, where a residue is
    below the 1e-12 noise clip relative to the largest at its pole.  The
    test here is a plain ``== 0`` on the stored value: whatever is left out
    contributed exactly 0 before.

    Bit-identity with the unpruned plan: tuples keep their order (a
    subsequence of the full product), each group keeps its nonzero members
    in that order, and the groups are put in the order of their FIRST
    OCCURRENCE IN THE FULL PRODUCT, which a zero tuple may define (the group
    sums of the poset routes run over groups, so this order fixes the
    rounding).  That order is replayed on numpy from the full product's
    alphas (``_full_product_first_occurrence``), ~100x cheaper per tuple
    than the plan entries and never materialised as Python objects; beyond
    ``PRUNE_ORDER_MAX_TUPLES`` the groups keep their pruned order.
    """
    kept = []
    dropped = False
    for ems in smooth_edge_modes:
        k = tuple((C, lam) for (C, lam) in ems.modes if not _mode_is_zero(C))
        if not k:
            return None
        if len(k) < len(ems.modes):
            dropped = True
        kept.append(_KeptModes(k))
    if not dropped:
        return None
    n_full = math.prod(len(ems.modes) for ems in smooth_edge_modes)
    n_keep = math.prod(len(kk.modes) for kk in kept)
    _plan_guard(n_keep, n_full, where)
    plan = _plan_from_tuples(tuple(_enumerate_pole_tuples(kept)),
                             len(kept), subset_constraint_data, m, n_ext)
    groups = plan['tuple_groups']
    n_groups_full = None
    if n_full <= PRUNE_ORDER_MAX_TUPLES:
        a_int_per_edge = _plan_coefficients(subset_constraint_data, m, 0)[0]
        replay = _full_product_first_occurrence(
            [[lam for (_C, lam) in ems.modes] for ems in smooth_edge_modes],
            a_int_per_edge, m, [a for (a, _i) in groups])
        if replay is None:
            _RUNTIME_COUNTERS['zero_mode_order_mismatch'] += 1
            _plan_refund(n_keep)
            return None
        first, n_groups_full = replay
        order = sorted(range(len(groups)), key=first.__getitem__)
        plan['tuple_groups'] = tuple(groups[i] for i in order)
    else:
        _RUNTIME_COUNTERS['zero_mode_order_unreplayed'] += 1
        n_groups_full = n_full
    # the decay margin from ALL modes, as the unpruned plan has it
    _decay_tol(smooth_edge_modes, None, plan, subset_constraint_data)
    n_kept = len(plan['pole_tuples'])
    plan['n_tuples_full'] = n_full
    plan['n_groups_full'] = n_groups_full
    _RUNTIME_COUNTERS['zero_mode_plans_pruned'] += 1
    _RUNTIME_COUNTERS['zero_modes_pruned'] += sum(
        len(ems.modes) - len(kk.modes)
        for ems, kk in zip(smooth_edge_modes, kept))
    _RUNTIME_COUNTERS['zero_mode_tuples_pruned'] += n_full - n_kept
    return plan


#: Rows of the full product per numpy block in the order replay
#: (``_full_product_first_occurrence``): 2^18 rows x m x 16 bytes.
_PRUNE_ORDER_BLOCK = 1 << 18


def _full_product_first_occurrence(lams_per_edge, a_int_per_edge, m,
                                   alphas_list):
    r"""For each alpha vector of ``alphas_list``: the index, in the full
    product over ``lams_per_edge`` (``_enumerate_pole_tuples`` order, edge 0
    fastest), of the first pole tuple with exactly that alpha; and the
    number of distinct alphas of the product as far as ``_DD_BATCH_MIN``
    needs it (exact below it, else a lower bound >= it).  ``None`` when a
    value is not finite or an alpha is not found (the arithmetic did not
    reproduce).

    The alphas are formed as ``_plan_from_tuples`` forms them: ``λ · a``
    (a complex times a real: both parts rounded once, as CPython does) summed
    edge by edge from the left, so the keys are equal as Python complex
    tuples are (signed zeros normalised by ``+ 0.0``, as ``-0.0 == 0.0``).
    Rows are matched through a complex projection (elementwise, so a row's
    value does not depend on its position) and confirmed byte for byte on
    the candidates only; the scan stops once every alpha is found."""
    if m == 0 or not alphas_list:
        return [0] * len(alphas_list), (1 if alphas_list else 0)
    try:
        terms = [np.asarray(lams, dtype=np.complex128)[:, None]
                 * np.asarray(a, dtype=np.float64)[None, :]
                 for lams, a in zip(lams_per_edge, a_int_per_edge)]
        targ = np.asarray(alphas_list, dtype=np.complex128).reshape(-1, m)
    except (TypeError, ValueError):
        return None
    if not (all(np.all(np.isfinite(t)) for t in terms)
            and np.all(np.isfinite(targ))):
        return None
    vd = np.dtype((np.void, 16 * m))
    w = [complex(0.7548776662466927 + 0.1093 * v, 0.5698402909980532
                 - 0.0721 * v) for v in range(m)]

    def proj(rows):
        sp = rows[:, 0] * w[0]
        for v in range(1, m):
            sp = sp + rows[:, v] * w[v]
        return sp

    targ = np.ascontiguousarray(targ + 0.0)
    t_proj = proj(targ)
    t_keys = targ.view(vd).ravel()
    # the distinct count only matters below _DD_BATCH_MIN
    need = _DD_BATCH_MIN if len(alphas_list) < _DD_BATCH_MIN else 0
    seen = set()
    E = len(terms)
    k, P = 1, len(terms[0])
    inner = terms[0]
    while k < E and P * len(terms[k]) <= _PRUNE_ORDER_BLOCK:
        inner = (inner[None, :, :] + terms[k][:, None, :]).reshape(-1, m)
        P *= len(terms[k])
        k += 1
    first = np.full(len(targ), -1, dtype=np.int64)
    base = 0
    outer = [range(len(terms[e])) for e in range(E - 1, k - 1, -1)]
    for combo in itertools.product(*outer):
        block = inner
        for j, i in enumerate(reversed(combo)):
            block = block + terms[k + j][i]
        block = np.ascontiguousarray(block + 0.0)
        sp = proj(block)
        if need and len(seen) < need:
            # distinct projections are distinct rows: enough of them settle
            # it; otherwise count the rows themselves
            if len(np.unique(sp)) >= need:
                seen.update(range(-need, 0))
            else:
                seen.update(x.tobytes()
                            for x in np.unique(block.view(vd).ravel()))
        miss = np.flatnonzero(first < 0)
        if len(miss):
            ts = np.sort(t_proj[miss])
            at = np.minimum(np.searchsorted(ts, sp), len(ts) - 1)
            cand = np.flatnonzero(ts[at] == sp)
            if len(cand):
                u, fi = np.unique(block[cand].view(vd).ravel(),
                                  return_index=True)
                pos = np.minimum(np.searchsorted(u, t_keys[miss]),
                                 len(u) - 1)
                hit = u[pos] == t_keys[miss]
                first[miss[hit]] = base + cand[fi[pos[hit]]]
        base += P
        if not np.any(first < 0) and not (need and len(seen) < need):
            break
    if np.any(first < 0):
        return None
    return first.tolist(), (min(len(seen), need) if need and len(seen) >= need
                            else len(seen) if need else len(alphas_list))


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


# Returned by ``_extract_causal_poset``
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
    * **Constant** (zero normal): ``_const_row_verdict`` (with
      ``tie_ctx``) decides every constant row FIRST: one EMPTY row returns
      ``_EMPTY_POSET``, DROP rows are skipped.
    * **Anything else** (multiple inter-axis couplings, mixed
      coefficients, inter-axis with a shift beyond the window, etc.) →
      return ``None``.  Caller falls back to scipy.nquad.

    Inter-axis rows that form a directed cycle whose shifts
    sum to <= 0 (``_order_rows_infeasible``; e.g. an unshifted 2-cycle, or
    one shifted by a rounding-level −1e-16) also return ``_EMPTY_POSET``.
    A cycle with a positive total shift encloses a thin nonempty strip and
    is not decided here.

    Returns ``_CausalPoset``, ``None`` or ``_EMPTY_POSET``.
    ``row_kinds``: the rows' provenance (``None``: all 'edge');
    ``tie_ctx``: see ``_tie_order_sign``.
    """
    const_verdict = {}
    # Constant rows first: one EMPTY row empties the region whatever the
    # other rows are (so it never reaches a fallback).
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
            if idx in const_verdict:
                continue            # DROP (decided above)
            # Coefficients in (_ROW_COEF_ATOL, tol]: only a direct caller
            # with a looser ``tol`` gets here.  Not a clean row.
            return None
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

    if _order_rows_infeasible(m, order_rows):
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


#: Relative tolerance of "equal" scalar lowers (``_poset_scalar_lower_verdict``),
#: against the largest scalar bound of the poset (its time scale).
POSET_LOWER_RTOL = 1e-12


def _causal_poset_consistent_scalar_lower(poset, tol=POSET_LOWER_RTOL):
    r"""Compute the single effective scalar lower bound shared by
    variables that have one.

    Returns ``(L, True)`` if every variable WITH A SCALAR LOWER has
    the same value ``L`` (within ``tol`` RELATIVE to the largest scalar
    bound of the poset), or no scalar lower at all
    (returns ``(None, True)`` — the caller's lower bound).  Variables
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

    M3 (P3): also ``(None, False)`` when some variable has neither a
    scalar lower of its own nor a predecessor with one in the transitive
    closure of the strict order (``_poset_lower_not_inherited``).  Such a
    variable is not bounded below by L, so the single-L chain would cut off
    the region below L.  (Before M3 every variable inherited L; the switch
    that restored that, ``USE_DBM_FALLBACK``, was retired in audit W7.)
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


def _poset_scalar_lower_verdict(poset, tol=POSET_LOWER_RTOL):
    """``(L, ok, reason)``: ``_causal_poset_consistent_scalar_lower``'s
    ``(L, ok)`` plus why it failed (``None``, ``'inconsistent'``: unequal
    scalar lowers, ``'not_inherited'``: the M3 inheritance rule).

    Lowers count as equal within ``tol`` times the poset's time scale (its
    largest scalar bound, lower or upper), a rounding-level margin in any
    time unit.  Audit W4b (ACC-63): an ABSOLUTE 1e-9 merged the distinct
    lowers 0 and 5e-10 of a region with upper 2e-9 (the exact 1e9-faster
    twin of 0, 0.5, 2, which bails): the chain simplex from one L gave the
    m = 3 value 36.6 % low."""
    per_var_max = {}
    for (var, c) in poset.scalar_lowers:
        cur = per_var_max.get(var)
        per_var_max[var] = c if cur is None else max(cur, c)
    if not per_var_max:
        return (None, True, None)
    vals = list(per_var_max.values())
    Lmin, Lmax = min(vals), max(vals)
    t_scale = max(abs(c) for (_v, c) in
                  tuple(poset.scalar_lowers) + tuple(poset.scalar_uppers))
    if Lmax - Lmin > tol * t_scale:
        return (None, False, 'inconsistent')
    # All variables that have a scalar lower agree on value Lmax.
    # Variables without one inherit it via the chain ordering -- M3: only
    # from a lower-bounded ancestor.
    if _poset_lower_not_inherited(poset, per_var_max):
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


# ── M4 (P5): memo of the m >= 3 chain integrals (``_chain_dd_log``) ────
# A chain integral is a pure, deterministic function of (alphas, L, uppers,
# U, decay margin), and the poset walk evaluates it many times with
# identical arguments (higher loop orders; models with repeated poles).  The
# memo changes NO number: a hit returns the very object the first call
# computed.  (Its switch, ``USE_CHAIN_UPPERS_MEMO`` /
# ``DAEDALUS_CHAIN_UPPERS_MEMO``, was retired in audit W7: always on.)
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
# M7b: the DD route's chains of one linear extension for EVERY pole-tuple
# group of a plan, ``(status, log_scale, mant)`` arrays, filled from the
# per-chain memo above (so the same values) and keyed EXACTLY on the bytes
# of the plan's group alphas, the extension, the (L, uppers, U) suffix and
# the DD tag.  Same lock; cleared with it; bounded by the number of chains
# held (``_CHAIN_DD_BATCH_MAX``).
_CHAIN_DD_BATCH_MAX = 2_000_000
_chain_dd_batch_memo = {}
_chain_dd_batch_size = [0]
# The batched DD branches take a plan with at least this many pole-tuple
# groups (poset) or pole tuples (polygon); below it their fixed cost per
# call and per linear extension exceeds the scalar loop's (measured: one
# group and ~500 extensions per region, ``ou_quartic`` k = 4 ℓ = 2, ran
# 1.4x slower batched).
_DD_BATCH_MIN = 8
#: Item 35 (F2): the batched poset kernels compute chains only for the
#: groups whose folded coefficient is nonzero (the fold skips the others).
DD_BATCH_SKIP_ZERO_GROUPS = True




# ── M7: the box-free exact routes (the only Phase J routes since audit W7) ─
# the internal Phase J plan (not distributed) §3.1 (layers L3, L4) and §14.
#   * the m=2 region is built exactly -- finite vertices and the recession
#     rays, no box -- and integrated by the ideal-vertex fan (triangles,
#     half-strips, wedges; ``boxfree.region_2d``) on the ``expdd`` triangle
#     kernel anchored at the vertex with the largest real exponent, the pole
#     tuple's γ folded into the exponent;
#   * the m=1 interval is a first divided difference (no 1/α cancellation
#     for a small α);
#   * the m≥3 chain, intermediate uppers included, is ``expdd``'s (transfer
#     matrix), γ folded into the chain's log scale; a chain without a scalar
#     lower starts at ``PHASE_J_T_MIN`` (default −inf: the semi-infinite
#     chain, started from the dominant eigenvector), a chain top without a
#     scalar upper is +inf (mirror form);
#   * the DBM route has no box (open directions integrated to infinity).
#   The bilateral |Re| > 600 guards of the pre-0.2.0 routes are replaced by
#   ONE one-sided test on the FINAL exponent: log|value| > 700 means the
#   integral itself overflows; the path then bails with a ``*_dd_overflow``
#   reason (counter ``dd_overflow``) to the next route.  Never a silent
#   inf/nan.
# ``PHASE_J_T_MIN`` (float, default −inf; no environment variable): the lower
#   bound of every integration time.  The finite-lower-bound closed forms are
#   kept for it (switch-on transients, feature plan S6): a finite value is
#   honoured by the m=1 interval, the exact polygon, the semi-infinite chain
#   and the DBM route (a ``ValueError`` when it reaches the scipy fallback,
#   which cannot honour it).
# An open direction that does not decay (Re(α·r) >= 0 on a recession ray, a
# line in the region, a chain prefix sum with Re <= 0) is reported -- bail
# ``*_divergent``, counter ``divergent_subsets`` -- and the subset goes on to
# the next route (DBM, then the scipy fallback).  The divergence POLICY is
# M8's.
# The pre-0.2.0 boxed routes and their switches (``USE_PHASE_J_DD_KERNELS``,
# ``CHAIN_UPPERS_KERNEL``, ``SEMIINF_CHAIN``, ``UNBOUNDED_2D``,
# ``POLYGON_BBOX_CAP``, the chain-simplex routines) were retired in audit W7.
PHASE_J_T_MIN = float('-inf')


def _t_min():
    """``PHASE_J_T_MIN`` (call time): ``-inf`` or a finite float."""
    t = PHASE_J_T_MIN
    try:
        t = float(t)
    except (TypeError, ValueError):
        raise ValueError(f'final_integral.PHASE_J_T_MIN={PHASE_J_T_MIN!r}: '
                         f'expected a float') from None
    if t == float('-inf'):
        return t
    if not math.isfinite(t):
        raise ValueError(f'final_integral.PHASE_J_T_MIN={t!r}: expected '
                         f'-inf or a finite float')
    return t


def _t_min_frame(free_ext_vals, tie_ctx=None):
    """``PHASE_J_T_MIN`` in the time frame of one evaluation (validated by
    ``_t_min``).  ``contribution()`` translates each Wick permutation so that
    its origin leg sits at 0, so an absolute bound T is ``T - t_origin``
    there, ``t_origin`` the origin leg's raw time from the tie context (0
    without one: a direct call is in absolute time).  -inf stays -inf.  A
    finite bound after an external time raises: the cumulant would be
    asked for before the switch-on."""
    t = _t_min()
    if t == float('-inf'):
        return t
    t0 = 0.0
    if tie_ctx is not None and tie_ctx.origin_leg is not None:
        try:
            t0 = float(tie_ctx.times[tie_ctx.origin_leg])
        except (TypeError, ValueError):
            raise ValueError(
                'a finite final_integral.PHASE_J_T_MIN needs numeric external '
                f'times, got {tie_ctx.times[tie_ctx.origin_leg]!r}') from None
    t_rel = t - t0
    if t_rel > min([0.0] + [float(x) for x in free_ext_vals]):
        raise ValueError(
            f'final_integral.PHASE_J_T_MIN={t!r} is after an external time '
            f'(the cumulant is asked for before the switch-on)')
    return t_rel


def _m7_t_min_is_finite():
    """True when ``PHASE_J_T_MIN`` is not ``-inf`` (no validation)."""
    try:
        return float(PHASE_J_T_MIN) != float('-inf')
    except (TypeError, ValueError):
        return True


_BOXFREE = []


def _bf():
    """The ``boxfree`` module (imported on first use, so the default
    routes never import it)."""
    if not _BOXFREE:
        from engine.integration.time_domain import boxfree
        _BOXFREE.append(boxfree)
    return _BOXFREE[0]


# The DD routes' bail reasons, written out (overflow, divergent,
# unsupported) so that every reason string is a literal ``_bail`` argument.
_DD_BAILS = {
    'polygon': (lambda: _bail('polygon_dd_overflow'),
                lambda: _bail('polygon_divergent'),
                lambda: _bail('polygon_dd_unsupported')),
    'interval': (lambda: _bail('interval_dd_overflow'),
                 lambda: _bail('interval_divergent'),
                 lambda: _bail('interval_dd_unsupported')),
    'poset': (lambda: _bail('poset_dd_overflow'),
              lambda: _bail('poset_divergent'),
              lambda: _bail('poset_dd_unsupported')),
    'grouped_m0': (lambda: _bail('grouped_m0_dd_overflow'),
                   lambda: _bail('grouped_m0_divergent'),
                   lambda: _bail('grouped_m0_dd_unsupported')),
    'grouped_m1': (lambda: _bail('grouped_m1_dd_overflow'),
                   lambda: _bail('grouped_m1_divergent'),
                   lambda: _bail('grouped_m1_dd_unsupported')),
}


def _decay_tol(smooth_edge_modes=None, pole_tuples=None, plan=None,
               subset_constraint_data=None):
    """The margin by which an open direction's exponent must decay on the
    box-free routes: ``64 ε · Σ_e max|λ_e| · max_row Σ|a|``, the rounding
    scale of the exponents formed from the modes, so that the residue of an
    exponent that is exactly 0 (the integrand constant along the direction)
    never passes as decay.  From the per-edge modes, else from the pole
    tuples (cached in the plan)."""
    if plan is not None:
        tol = plan.get('_m7_decay_tol')
        if tol is not None:
            return tol
    per_edge = None
    if smooth_edge_modes is not None:
        try:
            per_edge = [max((abs(complex(lam)) for (_c, lam) in ems.modes),
                            default=0.0) for ems in smooth_edge_modes]
        except (AttributeError, TypeError):
            per_edge = None
    if per_edge is None:
        tuples = (plan['pole_tuples'] if plan is not None else pole_tuples)
        per_edge = []
        for (_C, lams) in (tuples or ()):
            for e, lam in enumerate(lams):
                a = abs(complex(lam))
                if e >= len(per_edge):
                    per_edge.append(a)
                elif a > per_edge[e]:
                    per_edge[e] = a
    amax = 1.0
    for (a_int, _a_ext, _c0) in (subset_constraint_data or ()):
        amax = max(amax, sum(abs(float(x)) for x in a_int))
    tol = 64.0 * 2.220446049250313e-16 * sum(per_edge) * amax
    if plan is not None:
        plan['_m7_decay_tol'] = tol
    return tol


def _dd_status_bail(status, prefix):
    """Count a non-OK ``boxfree`` status and bail with
    ``<prefix>_dd_overflow`` / ``<prefix>_divergent`` /
    ``<prefix>_dd_unsupported``."""
    bf = _bf()
    if status == bf.OVERFLOW:
        _RUNTIME_COUNTERS['dd_overflow'] += 1
        return _DD_BAILS[prefix][0]()
    if status == bf.DIVERGENT:
        _RUNTIME_COUNTERS['divergent_subsets'] += 1
        return _DD_BAILS[prefix][1]()
    _RUNTIME_COUNTERS['dd_unsupported'] += 1
    return _DD_BAILS[prefix][2]()


def _chain_simplex_memo_clear():
    """Drop the chain memo tables (call when poles change): the M4 chain
    table of ``_chain_dd_log`` and the M7b batch table."""
    with _CHAIN_UPPERS_LOCK:
        _chain_uppers_memo.clear()
        _chain_dd_batch_memo.clear()
        _chain_dd_batch_size[0] = 0


def _chain_uppers_key_suffix(L, upper_per_position, U_chain_top):
    """The (L, uppers, U) part of a chain memo key (``_chain_dd_log``).
    Independent of the alphas, so the poset loop builds it once per linear
    extension instead of once per pole-tuple group (M4 "A2").  ``uppers`` is
    sorted (dict-order invariant) and keeps a ``None``-valued entry as
    ``None``, so two different upper maps never alias.
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
    """Half-width of the box on which the default-tolerance scipy.nquad
    routine (``_integrate_nd_polytope``) closes an open direction:
    ``OUTER_CAP`` (read at call time).  The box-free DBM route takes its
    β-zero threshold from it (``_integrate_subset_dbm``).  (The hardened
    fallback truncates certified directions at K/κ instead.)"""
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
    pole_tuples=None,
    plan=None,
    row_kinds=None,
    tie_ctx=None,
):
    r"""Analytic ``∫_{polytope} Π_e [Σ_α C_α exp(λ_α · Δt_e)] · pref
                                ds_1 … ds_m`` for m ≥ 3 via causal-
    poset decomposition.

    ``row_kinds`` is the rows' provenance (``None``: all 'edge');
    ``tie_ctx`` orders an exact external-time tie (``_tie_order_sign``).
    A region that is empty by construction (``_EMPTY_POSET``: a constant
    EMPTY row, or an order cycle whose shifts sum to <= 0) returns 0j, in
    both the plan and the no-plan branch.

    Procedure:
      1. Extract the causal poset (DAG + scalar bounds) from the
         retardation constraints.  Fail (return ``None``) if any
         constraint is mixed (not a clean inter-axis or scalar
         bound).
      2. Resolve the COMMON scalar lower bound ``L`` for all
         integration variables.  Fail if scalar lowers differ across
         variables (the single chain form would over- or under-include
         regions), and (M3) if a variable has neither a scalar lower nor a
         lower-bounded predecessor (``'poset_lower_not_inherited'``; the
         dispatch then takes the exact DBM route).  Without a scalar lower,
         ``L = PHASE_J_T_MIN`` (default -inf: the semi-infinite chain).
      3. Enumerate every linear extension σ of the poset.  Each is a
         disjoint chain
           L ≤ s_{σ(0)} ≤ s_{σ(1)} ≤ … ≤ s_{σ(m-1)} ≤ U_σ
         where U_σ is the scalar upper of σ(m-1) (or +inf).
      4. Per linear extension and pole-tuple group, the chain integral on
         the DD kernels in the log domain (``_poset_modesum_dd``), the
         tuples' γ folded in before any exp.
      5. Sum across extensions and tuples.  Fail if a chain is divergent or
         overflows.

    Returns ``complex`` or ``None``.  ``None`` sends the subset on to the
    DBM route, then the scipy fallback.
    """
    if m < 3:
        return _bail('poset_m_lt_3')  # m=2 has its own dedicated path
    _t_min()                    # M7: a finite PHASE_J_T_MIN is never ignored
    _RUNTIME_COUNTERS['poset_attempted'] += 1
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
    # M7: no margin; the chain starts at PHASE_J_T_MIN (default -inf:
    # semi-infinite) unless a scalar lower bounds it.
    t_min = _t_min_frame(free_ext_vals, tie_ctx)
    L = float(L_value) if L_value is not None else t_min
    if t_min > L:
        L = t_min

    upper_per_var = _causal_poset_consistent_scalar_upper(poset)

    # M7: the DD kernels, both branches.
    return _poset_modesum_dd(
        poset, L, upper_per_var, m, float('inf'),
        smooth_edge_modes, prefactor_complex, subset_constraint_data,
        free_ext_vals, pole_tuples, plan)


def _chain_dd_log(alphas_chain, L, upper_per_position, U_chain_top,
                  _key_suffix=None, tol=0.0):
    """``boxfree.chain_log`` (``(status, log_scale, mant)``), memoised in the
    M4 table under a key that carries the ``'dd'`` tag, the kernel backend
    and the decay margin ``tol``; an infinite ``L`` / ``U`` is part of the
    key as such."""
    bf = _bf()
    try:
        if _key_suffix is None:
            _key_suffix = _chain_uppers_key_suffix(
                L, upper_per_position, U_chain_top)
        key = (tuple(alphas_chain), _key_suffix,
               ('dd', bf.expdd.BACKEND, float(tol)))
        with _CHAIN_UPPERS_LOCK:
            val = _chain_uppers_memo.get(key, _CHAIN_UPPERS_MISS)
            if val is not _CHAIN_UPPERS_MISS:
                _RUNTIME_COUNTERS['chain_uppers_memo_hits'] += 1
                return val
            _RUNTIME_COUNTERS['chain_uppers_memo_misses'] += 1
    except (TypeError, ValueError, OverflowError):
        return bf.chain_log(alphas_chain, L, upper_per_position, U_chain_top,
                            tol)
    val = bf.chain_log(alphas_chain, L, upper_per_position, U_chain_top, tol)
    with _CHAIN_UPPERS_LOCK:
        if len(_chain_uppers_memo) >= _CHAIN_UPPERS_MEMO_MAX:
            _chain_uppers_memo.clear()
            _RUNTIME_COUNTERS['chain_uppers_memo_evictions'] += 1
        _chain_uppers_memo[key] = val
    return val


def _m7b_plan_arrays(plan):
    """The plan's per-tuple and per-group data as numpy arrays for the
    batched DD kernels (M7b), built once and kept in the plan: ``C`` (the
    tuple coefficients), ``gc``/``S`` (γ constant and slopes), ``A`` (the
    alphas, ``(T, m)``), the groups' members (``member_idx`` in tuple order,
    ``member_off``) and the exact bytes of the group alphas (the batch memo
    key).  ``None`` when the plan does not convert (the scalar loops run)."""
    arrs = plan.get('_m7b_arrays')
    if arrs is not None:
        return arrs or None
    try:
        T = len(plan['pole_tuples'])
        C = np.array([complex(c) for (c, _l) in plan['pole_tuples']],
                      dtype=np.complex128)
        gc = np.array([complex(g) for g in plan['gamma_const_per_tuple']],
                       dtype=np.complex128)
        slopes = plan['gamma_slope_per_tuple_per_ext']
        n_ext = len(slopes[0]) if T else 0
        S = np.array([[complex(x) for x in row] for row in slopes],
                      dtype=np.complex128).reshape(T, n_ext)
        A = np.array([[complex(x) for x in a]
                       for a in plan['alphas_per_tuple']],
                      dtype=np.complex128)
        groups = plan['tuple_groups']
        GA = np.array([[complex(x) for x in a] for (a, _i) in groups],
                       dtype=np.complex128)
        if GA.ndim != 2:
            GA = GA.reshape(len(groups), -1)
        member_idx = np.array([t for (_a, idxs) in groups for t in idxs],
                               dtype=np.int64)
        member_off = np.zeros(len(groups) + 1, dtype=np.int64)
        member_off[1:] = np.cumsum([len(idxs) for (_a, idxs) in groups])
        arrs = {'C': C, 'gc': gc, 'S': S, 'A': A.reshape(T, -1) if T else A,
                'member_idx': member_idx, 'member_off': member_off,
                'group_alpha_bytes': GA.tobytes(), 'n_groups': len(groups),
                'GA': GA}
        if not (np.all(np.isfinite(gc)) and np.all(np.isfinite(S))):
            arrs = {}
    except (TypeError, ValueError, IndexError, KeyError):
        arrs = {}
    plan['_m7b_arrays'] = arrs
    return arrs or None


def _m7b_gammas(arrs, free_ext_vals):
    """γ per tuple at ``free_ext_vals`` (as ``_modesum_tuples``: the
    constant plus each slope times its external time, in order)."""
    gamma = arrs['gc']
    S = arrs['S']
    for j in range(S.shape[1]):
        gamma = gamma + S[:, j] * float(free_ext_vals[j])
    return gamma


def _chain_dd_batch(plan, arrs, m, sigma, L, upp, U_ext, key_suffix, tol,
                    live=None):
    """``(status, log_scale, mant)`` arrays of the chain of extension
    ``sigma`` for every group of ``plan`` (``_chain_dd_log`` per group, so
    the very values of the per-chain memo), memoised per (group alphas,
    ``sigma``, suffix, tag).  ``None`` when a chain cannot be evaluated (the
    scalar loop then meets the same input and decides).  ``live`` (item 35, F2):
    the indices of the groups to compute, in order (``None``: all); the
    arrays then hold those groups only, keyed on their alphas."""
    bf = _bf()
    groups = plan['tuple_groups']
    if live is not None:
        groups = [groups[g] for g in live]
    ga_bytes = (arrs['group_alpha_bytes'] if live is None
                else arrs['GA'][live].tobytes())
    key = (ga_bytes, m, tuple(sigma), key_suffix,
           ('dd', bf.expdd.BACKEND, float(tol)))
    with _CHAIN_UPPERS_LOCK:
        val = _chain_dd_batch_memo.get(key)
    if val is not None:
        return val
    G = len(groups)
    ST = np.empty(G, dtype=np.int64)
    LS = np.empty(G, dtype=np.float64)
    MANT = np.empty(G, dtype=np.complex128)
    try:
        for g, (alphas_orig, _idxs) in enumerate(groups):
            alphas_chain = [alphas_orig[sigma[k]] for k in range(m)]
            ST[g], LS[g], MANT[g] = _chain_dd_log(
                alphas_chain, L, upp, U_ext, _key_suffix=key_suffix, tol=tol)
    except (TypeError, ValueError, ArithmeticError):
        return None
    val = (ST, LS, MANT)
    with _CHAIN_UPPERS_LOCK:
        if _chain_dd_batch_size[0] + G > _CHAIN_DD_BATCH_MAX:
            _chain_dd_batch_memo.clear()
            _chain_dd_batch_size[0] = 0
        _chain_dd_batch_memo[key] = val
        _chain_dd_batch_size[0] += G
    return val


def _poset_dd_batch(plan, m, extensions, L, upper_for_ext,
                    upper_per_position_per_ext, key_suffixes, pref,
                    free_ext_vals, tol):
    """The plan branch of ``_poset_modesum_dd`` on the batched kernels: the
    groups' folded prefactors (``boxfree.group_fold``), the chains per
    extension for all groups at once (``_chain_dd_batch``) and their fold
    (``boxfree.chain_fold``), in the scalar loop's order: ``(value,
    Σ|term|, scale)`` (``_dd_accuracy``).  ``None`` when any term leaves the
    plain case (the scalar loop then runs and decides)."""
    # the UNPRUNED group count decides (item 35), as before the pruning
    if (plan.get('n_groups_full', len(plan['tuple_groups'])) < _DD_BATCH_MIN
            or any(k is None for k in key_suffixes)):
        return None
    arrs = _m7b_plan_arrays(plan)
    if arrs is None:
        return None
    bf = _bf()
    gamma = _m7b_gammas(arrs, free_ext_vals)
    coef = pref * arrs['C']
    tc, gmax, tcabs = bf.group_fold_mag(coef, gamma, arrs['member_idx'],
                                        arrs['member_off'])
    if not np.any(tc != 0):
        return 0.0 + 0.0j, 0.0, 0.0
    # F2 (item 35): chains only for the groups whose folded coefficient is
    # nonzero; ``chain_fold`` skips the others, so the sum is the same
    live = np.flatnonzero(tc != 0)
    if len(live) == len(tc) or not DD_BATCH_SKIP_ZERO_GROUPS:
        live = None
    else:
        _RUNTIME_COUNTERS['dd_batch_zero_groups_skipped'] += (
            len(tc) - len(live))
        tc, gmax, tcabs = tc[live], gmax[live], tcabs[live]
    rows = []
    for sigma, U_ext, upp, key in zip(extensions, upper_for_ext,
                                      upper_per_position_per_ext,
                                      key_suffixes):
        b = _chain_dd_batch(plan, arrs, m, sigma, L, upp, U_ext, key, tol,
                            live)
        if b is None:
            return None
        rows.append(b)
    E = len(rows)
    G = len(tc)
    ST = np.empty((E, G), dtype=np.int64)
    LS = np.empty((E, G), dtype=np.float64)
    MANT = np.empty((E, G), dtype=np.complex128)
    for e, (st, ls, mant) in enumerate(rows):
        ST[e] = st
        LS[e] = ls
        MANT[e] = mant
    r = bf.chain_fold_mag(tc, gmax, ST, LS, MANT, tcabs)
    if r is None:
        return None
    # the scalar loop's floor: Σ |coef| e^{Re γ} times the time volume of
    # the fastest exponent of the tuples with a nonzero coefficient
    A = np.asarray(arrs['A'])
    nz = np.asarray(coef) != 0
    rate = (float(np.max(np.abs(A[nz]))) if A.size and np.any(nz) else 0.0)
    return r[0], r[1], _dd_scale_np(coef, gamma) * _dbm.time_volume(
        rate, m, _chain_widths(L, upper_for_ext, m))


def _poset_modesum_dd(poset, L, upper_per_var, m, U_default,
                      smooth_edge_modes, prefactor_complex,
                      subset_constraint_data, free_ext_vals, pole_tuples,
                      plan):
    """The chain sum of ``_integrate_nd_polytope_poset_modesum`` on the DD
    kernels (M7): per linear extension the chain integral in the log domain
    (``_chain_dd_log``; ``L`` and the chain top may be infinite), folded with
    the pole tuples' γ before any exp: a group of tuples sharing alphas
    contributes ``e^{g + ls} · Σ_t pref C_t e^{γ_t - g} · mant`` with ``g``
    the group's largest Re γ.  A non-OK status bails ('poset_dd_overflow',
    'poset_divergent', 'poset_dd_unsupported')."""
    import cmath
    bf = _bf()
    OK = bf.OK
    OVERFLOW = bf.OVERFLOW
    LOG_OVERFLOW = bf.LOG_OVERFLOW
    extensions = list(_enumerate_linear_extensions(poset))
    if not extensions:
        return _bail('poset_no_extension')
    upper_for_ext = [
        float(upper_per_var.get(sigma[m - 1], U_default))
        for sigma in extensions
    ]
    upper_per_position_per_ext = []
    for sigma in extensions:
        upp = {}
        for k in range(m):
            v = sigma[k]
            if v in upper_per_var:
                upp[k] = float(upper_per_var[v])
        upper_per_position_per_ext.append(upp)
    key_suffixes = [None] * len(extensions)
    for i_ext, (U_ext, upp) in enumerate(
            zip(upper_for_ext, upper_per_position_per_ext)):
        try:
            key_suffixes[i_ext] = _chain_uppers_key_suffix(L, upp, U_ext)
        except (TypeError, ValueError, OverflowError):
            pass
    pref = complex(prefactor_complex)
    tol = _decay_tol(smooth_edge_modes, pole_tuples, plan,
                     subset_constraint_data)
    use_mp_accum = USE_POSET_MPMATH_ACCUMULATION
    # M7b: all groups of the plan at once; None = the scalar loop below
    if plan is not None and not use_mp_accum:
        res = _poset_dd_batch(plan, m, extensions, L, upper_for_ext,
                              upper_per_position_per_ext, key_suffixes,
                              pref, free_ext_vals, tol)
        if res is not None:
            return _poset_dd_finish(res[0], L, upper_for_ext, res[1],
                                    res[2])
    if plan is not None:
        terms = list(_modesum_tuples(None, subset_constraint_data,
                                     free_ext_vals, m, None, plan))
        groups = [(alphas, [(pref * terms[t][0], terms[t][2])
                            for t in t_idxs])
                  for alphas, t_idxs in plan['tuple_groups']]
    else:
        terms = list(_modesum_tuples(smooth_edge_modes,
                                     subset_constraint_data, free_ext_vals,
                                     m, pole_tuples, None))
        # M8: an α shared by several tuples whose merged coefficient is
        # exactly 0 is dropped, as the plan branch's fold drops it
        cancelling = _cancelling_alphas(terms, pref)
        groups = ((alphas, [(pref * C_prod, gamma)])
                  for C_prod, alphas, gamma in terms
                  if alphas not in cancelling)
    if use_mp_accum:
        from mpmath import mp as _mp, mpc as _mpc
        with _mp.workdps(50):
            total_mp = _mpc(0, 0)
    total = 0.0 + 0.0j
    mag = scale = 0.0
    rate = 0.0
    for alphas_orig, members in groups:
        members = [(c, g) for (c, g) in members if c != 0]
        if not members:
            continue
        scale += sum(_dd_scale_term(c, g) for (c, g) in members)
        rate = max([rate] + [abs(complex(a)) for a in alphas_orig])
        g_max = max(g.real for (_c, g) in members)
        tc = 0.0 + 0.0j
        tcabs = 0.0
        for (c, g) in members:
            x = c * cmath.exp(g - g_max)
            tc += x
            tcabs += abs(x)
        if tc == 0:
            continue
        for sigma, U_ext, upp, key in zip(
                extensions, upper_for_ext, upper_per_position_per_ext,
                key_suffixes):
            alphas_chain = [alphas_orig[sigma[k]] for k in range(m)]
            st, ls, mant = _chain_dd_log(alphas_chain, L, upp, U_ext,
                                         _key_suffix=key, tol=tol)
            # OVERFLOW is the chain's own verdict, before γ: its (ls, mant)
            # are valid, and the test that counts is the folded one below
            if st != OK and st != OVERFLOW:
                return _dd_status_bail(st, 'poset')
            zr = g_max + ls
            if (-LOG_OVERFLOW <= zr <= LOG_OVERFLOW and abs(mant) < 1e150
                    and abs(tc) < 1e150):
                v = (tc * mant) * math.exp(zr)
            else:
                st, v = bf.exp_times(complex(zr, 0.0) + cmath.log(tc), mant)
                if st != OK:
                    return _dd_status_bail(st, 'poset')
            mag += abs(v) * (tcabs / abs(tc))
            if use_mp_accum:
                with _mp.workdps(50):
                    total_mp = total_mp + _mpc(v.real, v.imag)
            else:
                total += v
    if use_mp_accum:
        total = complex(float(total_mp.real), float(total_mp.imag))
    return _poset_dd_finish(total, L, upper_for_ext, mag,
                            scale * _dbm.time_volume(
                                rate, m, _chain_widths(L, upper_for_ext, m)))


def _poset_dd_finish(total, L, upper_for_ext, mag=0.0, scale=0.0):
    """The tail of ``_poset_modesum_dd``: the overflow test on the total,
    the error accounting (``_dd_accuracy``) and the DD counters."""
    total = _finite_or_overflow(total, 'poset')
    if total is not None:
        _dd_accuracy(total, mag, scale)
        _RUNTIME_COUNTERS['dd_poset'] += 1
        if L == float('-inf') or any(u == float('inf')
                                     for u in upper_for_ext):
            _RUNTIME_COUNTERS['poset_semiinf_chains'] += 1
    return total


# ───────────────────────────────────────────────────────────────────────
# Exact DBM route for m≥3 subsets (M3)
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
    r"""Exact ``∫_{region} pref · Π_e [Σ_α C_α exp(λ_α · Δt_e)] ds`` for an
    m ≥ 3 subset whose poset path bailed: the DBM route
    (``dbm_integral.integrate_exp_sum``).

    The integrand and its arguments are those of
    ``_integrate_nd_polytope_poset_modesum`` (one row of
    ``subset_constraint_data`` per smooth edge; the pole tuples from
    ``plan``, ``pole_tuples`` or ``smooth_edge_modes``).  By default
    (``cap=None``) the region is box-free: a direction the rows leave open
    is integrated to infinity (``'dbm_divergent'`` when it does not decay
    by more than the rounding margin of ``_decay_tol``).  An explicit
    ``cap`` closes such directions at ±``cap`` instead (absolute, from the
    time origin; a finite bound is kept as it is, even beyond the box):
    the boxed reference of a test, the box of the pre-0.2.0 route at
    ``cap=200.0``.

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

    # The conditioning test's absolute floor: the DBM's own default
    # (``integral_scale``: Σ |C| e^{Re E} times a volume in the fastest
    # exponent's time unit, capped by the region), in the units of the
    # integral.  Audit W4b (ACC-64): the Σ |C| passed here lacked e^{Re E}
    # (an ill-conditioned closed form in a correlator tail passed, its mp
    # retry skipped) and [time]^m (the verdict changed with the time unit).
    scale = None
    # M7: no box (open directions integrated to infinity, DIVERGENT when
    # one does not decay by more than the rounding margin of
    # ``_decay_tol``), with the β-zero threshold of the box
    # ±max(_nquad_outer_cap(), 40/κ_min + max|free time|) that the
    # pre-0.2.0 DD route used (κ_min the slowest decay rate of the decaying
    # modes).  A finite PHASE_J_T_MIN (``_t_min``) adds the rows
    # s_v > PHASE_J_T_MIN.  An explicit ``cap``: that box, no β threshold.
    eps = None
    beta_tol = None
    t_min = _t_min_frame(free_ext_vals, tie_ctx)
    if t_min != -math.inf:
        for v in range(m):
            e_v = tuple(1.0 if i == v else 0.0 for i in range(m))
            rows.append((e_v, -t_min))
    if cap is None:
        base = _nquad_outer_cap()
        # every mode of every edge (item 35: from the edges when they are
        # given, since a pruned plan or grouped tuple list lacks the
        # exact-zero modes; the same set as the full product's)
        if smooth_edge_modes is not None:
            lam_all = [lam for ems in smooth_edge_modes
                       for (_C, lam) in ems.modes]
        else:
            lam_all = [lam for (_C, lams) in (
                plan['pole_tuples'] if plan is not None
                else pole_tuples) for lam in lams]
        # decay rates above the rounding margin only (a slower one cannot
        # count as decay; it would also overflow 40/κ)
        k_floor = _decay_tol(smooth_edge_modes, pole_tuples, plan,
                             subset_constraint_data)
        kmin = min((-complex(lam).real for lam in lam_all
                    if -complex(lam).real > k_floor),
                   default=math.inf)
        t_far = max((abs(float(t)) for t in free_ext_vals), default=0.0)
        slow_cap = (max(base, 40.0 / kmin + t_far) if kmin < math.inf
                    else base)
        cap = math.inf
        eps = _dbm.BETA_ZERO_SPAN / max(slow_cap, 1.0)
        beta_tol = _decay_tol(smooth_edge_modes, pole_tuples, plan,
                              subset_constraint_data)
        _RUNTIME_COUNTERS['dbm_box_free'] += 1
    res = _dbm.integrate_exp_sum(rows, m, seeds, cap, scale=scale, eps=eps,
                                 beta_tol=beta_tol)
    if res.status == _dbm.STATUS_DIVERGENT:
        _RUNTIME_COUNTERS['dbm_declined_divergent'] += 1
        if poset_bail_reason != 'poset_divergent':
            _RUNTIME_COUNTERS['divergent_subsets'] += 1
        return _bail('dbm_divergent')
    if res.status == _dbm.STATUS_NOT_DBM:
        _RUNTIME_COUNTERS['dbm_declined_rows'] += 1
        return _bail('dbm_not_difference_rows')
    if res.status == _dbm.STATUS_OVERFLOW:
        _RUNTIME_COUNTERS['dbm_declined_overflow'] += 1
        return _bail('dbm_overflow')
    if res.status == _dbm.STATUS_TOO_MANY_CASES:
        _RUNTIME_COUNTERS['dbm_declined_other'] += 1
        return _bail('dbm_too_many_cases')
    err_factor = _dbm.COND_ERR_FACTOR
    if (res.status == _dbm.STATUS_ILL_CONDITIONED
            and _dbm_high_precision_on()):
        # The same exact elimination in MP_DPS-digit arithmetic: the
        # expansion's cancellation (near-equal rates) costs digits of the
        # double closed form, not of the integral (item 27: a conductance
        # kernel rate next to a propagator pole sent such subsets to days
        # of nested quadrature).
        _RUNTIME_COUNTERS['dbm_mp_attempted'] += 1
        res_mp = _dbm.integrate_exp_sum_mp(rows, m, seeds, cap, scale=scale,
                                           eps=eps, beta_tol=beta_tol)
        if res_mp.status == _dbm.STATUS_OK:
            _RUNTIME_COUNTERS['dbm_mp_answered'] += 1
            res = res_mp
            err_factor = 10.0 ** (2 - _dbm.MP_DPS)
    if res.status == _dbm.STATUS_ILL_CONDITIONED:
        _RUNTIME_COUNTERS['dbm_declined_ill_conditioned'] += 1
        return _bail('dbm_ill_conditioned')
    _RUNTIME_COUNTERS['dbm_answered'] += 1
    _RUNTIME_COUNTERS['dbm_err_abs_sum'] += err_factor * res.magnitude
    if res.status == _dbm.STATUS_EMPTY:
        _RUNTIME_COUNTERS['dbm_empty'] += 1
    elif res.error_ratio > _RUNTIME_COUNTERS['dbm_error_ratio_max']:
        _RUNTIME_COUNTERS['dbm_error_ratio_max'] = res.error_ratio
    if res.thin_retry:
        _RUNTIME_COUNTERS['dbm_answered_thin_retry'] += 1
    if poset_bail_reason == 'poset_lower_not_inherited':
        _RUNTIME_COUNTERS['dbm_answered_p3'] += 1
    return res.value


# ───────────────────────────────────────────────────────────────────────
# M7: the DD-kernel / box-free evaluation of the analytic routes
# ───────────────────────────────────────────────────────────────────────
def _modesum_tuples(smooth_edge_modes, subset_constraint_data, free_ext_vals,
                    m, pole_tuples=None, plan=None):
    """Yield ``(C_prod, alphas, gamma)`` per pole tuple: the coefficient,
    the ``m`` exponent coefficients of the integration variables and the
    constant part of the exponent at ``free_ext_vals`` -- the quantities the
    legacy routes form inline, from the plan when there is one."""
    if plan is not None:
        alphas_per_tuple = plan['alphas_per_tuple']
        gamma_const_per_tuple = plan['gamma_const_per_tuple']
        gamma_slope_per_tuple_per_ext = plan['gamma_slope_per_tuple_per_ext']
        n_ext = len(free_ext_vals)
        for t_idx, (C_prod, _lambdas) in enumerate(plan['pole_tuples']):
            gamma = gamma_const_per_tuple[t_idx]
            slope_row = gamma_slope_per_tuple_per_ext[t_idx]
            for j in range(n_ext):
                gamma = gamma + slope_row[j] * free_ext_vals[j]
            yield C_prod, alphas_per_tuple[t_idx], gamma
        return
    c_ext_per_edge = [
        float(c0) + sum(
            float(a_ext[j]) * float(free_ext_vals[j])
            for j in range(len(a_ext))
        )
        for (_a_int, a_ext, c0) in subset_constraint_data
    ]
    a_int_per_edge = [
        tuple(float(a_int[v]) if v < len(a_int) else 0.0 for v in range(m))
        for (a_int, _a_ext, _c0) in subset_constraint_data
    ]
    n_rows = len(subset_constraint_data)
    pole_iter = (
        pole_tuples if pole_tuples is not None
        else _enumerate_pole_tuples(smooth_edge_modes)
    )
    for C_prod, lambdas in pole_iter:
        alphas = [0.0 + 0.0j] * m
        gamma = 0.0 + 0.0j
        for e in range(n_rows):
            lam = lambdas[e]
            a_e = a_int_per_edge[e]
            for v in range(m):
                alphas[v] += lam * a_e[v]
            gamma += lam * c_ext_per_edge[e]
        yield C_prod, tuple(alphas), gamma


def _cancelling_alphas(terms, pref):
    """The alphas shared by two or more of ``terms`` (``(C, alphas, γ)``)
    whose merged coefficient ``Σ pref·C e^{γ - γ_max}`` is exactly 0 (one
    pass; ``_divergent_group_cancels`` for one alpha)."""
    seen = {}
    dup = set()
    for (C, a, g) in terms:
        if a in seen:
            dup.add(a)
        else:
            seen[a] = True
    if not dup:
        return frozenset()
    members = {a: [] for a in dup}
    for (C, a, g) in terms:
        if a in members and pref * C != 0:
            members[a].append((pref * C, g))
    out = set()
    for a, mem in members.items():
        if mem:
            g_max = max(g.real for (_c, g) in mem)
            w = 0.0 + 0.0j
            for (c, g) in mem:
                w += c * cmath.exp(g - g_max)
            if w != 0:
                continue
        out.add(a)
    return out


def _divergent_group_cancels(terms, alphas, pref):
    """True when the pole tuples of ``terms`` (``(C, alphas, γ)``) with
    exactly these ``alphas`` have the merged coefficient
    ``Σ pref·C e^{γ - γ_max}`` exactly 0: their integrand is identically
    zero, so a convergence-certificate failure of that group is not a
    divergence (M8; the m >= 3 chain fold merges the same way before it
    evaluates a chain)."""
    members = [(pref * C, g) for (C, a, g) in terms
               if a == alphas and pref * C != 0]
    if not members:
        return True
    g_max = max(g.real for (_c, g) in members)
    w = 0.0 + 0.0j
    for (c, g) in members:
        w += c * cmath.exp(g - g_max)
    return w == 0


def _chain_widths(L, uppers, m):
    """Extent of a chain region ``L < s_1 < ... < s_m < U`` along each
    variable for the DD floor's volume (``dbm_integral.time_volume``): the
    largest finite upper minus ``L`` (infinite when open)."""
    U = max((float(u) for u in uppers), default=math.inf)
    return [U - float(L)] * m


def _region2d_widths(region):
    """Extent of a 2-D DD region along each axis (``boxfree.Region2D``): its
    vertices' bounding box, infinite along an axis a recession ray moves."""
    out = []
    for ax in range(2):
        if any(abs(r[ax]) > 0 for r in getattr(region, 'rays', ())):
            out.append(math.inf)
            continue
        xs = [float(v[ax]) for v in getattr(region, 'vertices', ())]
        out.append(max(xs) - min(xs) if xs else math.inf)
    return out


def _dd_scale_term(coef, gamma):
    """``|coef| e^{Re γ}``: one pole tuple's share of the integrand scale
    of a DD route (the DBM's ``scale``, capped at e^LOG_OVERFLOW)."""
    return abs(coef) * math.exp(min(gamma.real, 700.0))


def _dd_scale_np(coefs, gammas):
    """``Σ |coef| e^{Re γ}`` over arrays (``_dd_scale_term``)."""
    with np.errstate(over='ignore'):     # inf, as the scalar sum gives
        return float(np.sum(np.abs(coefs)
                            * np.exp(np.minimum(np.real(gammas), 700.0))))


def _dd_accuracy(total, mag, scale):
    """Error accounting of a box-free subset sum (M7 report: stiff mode
    ratios and cancelling sums lose accuracy with an OK status): the DBM's
    own test -- rounding error ``COND_ERR_FACTOR · Σ|term|`` against
    ``max(COND_RTOL · |value|, COND_ATOL · scale)``, ``scale`` in the units
    of the integral as the DBM's ``integral_scale`` (the callers multiply
    Σ |coef| e^{Re γ} by ``dbm_integral.time_volume``: audit W4b, ACC-64)
    -- recorded in the
    counters ``dd_err_ratio_max`` (the worst ratio) and ``dd_ill_conditioned``
    (subsets above 1).  Bookkeeping only: the value is used as computed (the
    DD kernels are the most accurate route there is).  The test is relative
    to the SUBSET, so a flagged subset may be negligible for the cumulant
    (quad_exp: 8 regions near a cancellation, ~1e-21 absolute against a
    one-loop of 2e-4); hence counters, not a warning."""
    accepted = max(_dbm.COND_RTOL * abs(total), _dbm.COND_ATOL * scale)
    err = _dbm.COND_ERR_FACTOR * mag
    if err == 0:
        return
    _RUNTIME_COUNTERS['dd_err_abs_sum'] += err
    ratio = err / accepted if accepted > 0 else math.inf
    if ratio > _RUNTIME_COUNTERS['dd_err_ratio_max']:
        _RUNTIME_COUNTERS['dd_err_ratio_max'] = ratio
    if ratio > 1.0:
        _RUNTIME_COUNTERS['dd_ill_conditioned'] += 1


def _finite_or_overflow(total, prefix):
    """``total`` if finite, else the explicit overflow bail."""
    if math.isfinite(total.real) and math.isfinite(total.imag):
        return total
    _RUNTIME_COUNTERS['dd_overflow'] += 1
    return _DD_BAILS[prefix][0]()


def _interval_dd_total(terms, pref, L, U, prefix, tol=0.0,
                       t_min=float('-inf')):
    """``Σ pref·C ∫_L^U exp(α s + γ) ds`` over ``terms`` (``(C, (α,), γ)``)
    on the DD kernels (``boxfree.interval``): ``L``, ``U`` may be infinite
    (``tol``: see ``_decay_tol``); ``t_min`` (``PHASE_J_T_MIN`` in the
    call's frame, ``_t_min_frame``) raises ``L``.  A non-OK status bails
    (``prefix``)."""
    bf = _bf()
    if t_min > L:
        L = t_min
    if not L < U:
        return 0.0 + 0.0j
    terms = list(terms)
    dropped = set()
    total = 0.0 + 0.0j
    mag = scale = 0.0
    rate = 0.0
    for C_prod, alphas, gamma in terms:
        coef = pref * C_prod
        if coef == 0 or alphas in dropped:
            continue
        scale += _dd_scale_term(coef, gamma)
        rate = max(rate, abs(complex(alphas[0])))
        st, v = bf.interval(alphas[0], gamma, L, U, tol)
        if st == bf.DIVERGENT and _divergent_group_cancels(terms, alphas,
                                                           pref):
            dropped.add(alphas)
            _RUNTIME_COUNTERS['divergent_cancelled'] += 1
            continue
        if st == bf.OVERFLOW:
            # the integral alone overflows: decide on the whole term
            st, v = bf.interval(alphas[0], gamma + cmath.log(coef), L, U,
                                tol)
            if st == bf.OK:
                total += v
                mag += abs(v)
                continue
        if st != bf.OK:
            return _dd_status_bail(st, prefix)
        total += coef * v
        mag += abs(coef * v)
    total = _finite_or_overflow(total, prefix)
    if total is not None:
        _dd_accuracy(total, mag, scale * _dbm.time_volume(rate, 1, [U - L]))
        _RUNTIME_COUNTERS['dd_interval'] += 1
    return total


def _integrate_1d_polytope_modesum(
    smooth_edge_modes,
    prefactor_complex,
    subset_constraint_data,
    free_ext_vals,
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
    ``[L, U]`` (a first divided difference on the DD kernels,
    ``_interval_dd_total``; ``L`` is at least ``PHASE_J_T_MIN``).

    The interval bounds come from the smooth-edge retardation
    constraints ``a_int_e[0]·s + c_ext_e > 0``.  Unbounded sides are
    tracked as ±∞: a term that does not decay in that direction is a
    divergence (bail ``'interval_divergent'``).

    ``pole_tuples`` (optional): pre-built iterable of ``(C_prod,
    lambdas)`` pairs that replaces ``_enumerate_pole_tuples
    (smooth_edge_modes)``.

    Returns ``complex`` or ``None`` (through ``_bail``) on overflow or a
    divergent unbounded integrand.

    Constant rows: ``_const_row_verdict`` (EMPTY -> 0j, DROP -> no bound;
    the row still enters γ).  ``row_kinds``: the rows' provenance;
    ``tie_ctx``: see ``_tie_order_sign``.
    """
    import math
    _t_min()                    # M7: a finite PHASE_J_T_MIN is never ignored
    n_smooth = len(smooth_edge_modes)
    _RUNTIME_COUNTERS['interval_attempted'] += 1
    if len(subset_constraint_data) != n_smooth:
        _RUNTIME_COUNTERS['interval_returned_none'] += 1
        return _bail('interval_rows_mismatch')

    # Resolve the feasible interval — track unboundedness exactly.
    L = -math.inf
    U = +math.inf
    for idx, (a_int, a_ext, c0) in enumerate(subset_constraint_data):
        a = float(a_int[0]) if a_int else 0.0
        c_eff = float(c0) + sum(
            float(a_ext[i]) * float(free_ext_vals[i])
            for i in range(len(a_ext))
        )
        if abs(a) <= _ROW_COEF_ATOL:
            if _const_row_verdict(
                    (a,), a_ext, c0, free_ext_vals,
                    _row_kind(row_kinds, idx), tie_ctx) == 'EMPTY':
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

    pref = complex(prefactor_complex)

    # M7: the DD kernels, both branches.
    return _interval_dd_total(
        _modesum_tuples(smooth_edge_modes, subset_constraint_data,
                        free_ext_vals, 1, pole_tuples, plan),
        pref, L, U, 'interval',
        _decay_tol(smooth_edge_modes, pole_tuples, plan,
                   subset_constraint_data),
        _t_min_frame(free_ext_vals, tie_ctx))


def _integrate_2d_polygon_modesum(
    smooth_edge_modes,
    prefactor_complex,
    subset_constraint_data,
    free_ext_vals,
    pole_tuples=None,
    plan=None,
    row_kinds=None,
    tie_ctx=None,
):
    r"""Analytic ∫∫_region Π_e [Σ_α C_α exp(λ_α · Δt_e)] · prefactor
                  ds_0 ds_1.

    Each per-edge mode sum is pole-expanded; the resulting sum of
    single-exponential terms ``A · exp(α·s_0 + β·s_1 + γ)`` is
    integrated analytically over the exact (possibly unbounded) region
    defined by ``subset_constraint_data`` (``_polygon_modesum_dd``).

    Returns ``complex`` or ``None`` (through ``_bail``) when the rows do not
    match the modes, the region is divergent or the value overflows.

    Δt_e for each smooth edge is expressed as
        Δt_e = c0_e + a_int_e[0]·s_0 + a_int_e[1]·s_1 + Σ_j a_ext_e[j]·t_free[j]
    Substituting into ``Σ_α λ_α · Δt_e`` and rearranging gives the
    exponent ``γ + α_s·s_0 + β_s·s_1`` where
        α_s = Σ_e a_int_e[0] · λ_α_e
        β_s = Σ_e a_int_e[1] · λ_α_e
        γ   = Σ_e λ_α_e · (c0_e + Σ_j a_ext_e[j]·t_free[j])

    ``pole_tuples`` (optional): pre-built iterable of ``(C_prod, lambdas)``
    pairs that replaces ``_enumerate_pole_tuples(smooth_edge_modes)``.

    The region (shared by the plan and the no-plan branch) decides constant
    rows by the Θ(0) rule (``_polygon_region_dd``); a constant DROP row only
    leaves the geometry and still enters γ.  ``row_kinds``: the rows'
    provenance (``None``: all 'edge'); ``tie_ctx``: see ``_tie_order_sign``.
    """
    _t_min()                    # M7: a finite PHASE_J_T_MIN is never ignored
    n_smooth = len(smooth_edge_modes)
    _RUNTIME_COUNTERS['polygon_attempted'] += 1
    if len(subset_constraint_data) != n_smooth:
        # Smooth-edges-to-constraints mismatch shouldn't happen — the
        # caller built both from ``smooth_edges`` in lock-step.  Bail
        # to the scipy fallback.
        _RUNTIME_COUNTERS['polygon_returned_none'] += 1
        return _bail('polygon_rows_mismatch')

    # M7: the DD kernels on the exact (box-free) region, both branches.
    return _polygon_modesum_dd(
        smooth_edge_modes, prefactor_complex, subset_constraint_data,
        free_ext_vals, pole_tuples, plan, row_kinds, tie_ctx)


def _polygon_region_dd(subset_constraint_data, free_ext_vals,
                       row_kinds=None, tie_ctx=None):
    """The ``boxfree.Region2D`` of an m=2 subset: the exact region
    (``boxfree.region_2d``) of the rows, constant rows decided by
    ``_const_row_verdict`` (EMPTY empties the region; DROP skips the row,
    which stays in γ), plus ``s_i > PHASE_J_T_MIN`` when that is finite."""
    bf = _bf()
    rows = []
    for idx, (a_int, a_ext, c0) in enumerate(subset_constraint_data):
        c_eff = float(c0) + sum(
            float(a_ext[j]) * float(free_ext_vals[j])
            for j in range(len(a_ext))
        )
        a0 = float(a_int[0])
        a1 = float(a_int[1])
        if abs(a0) <= _ROW_COEF_ATOL and abs(a1) <= _ROW_COEF_ATOL:
            _RUNTIME_COUNTERS['zero_normal_rows_seen'] += 1
            verdict = _const_row_verdict(
                (a0, a1), a_ext, c0, free_ext_vals,
                _row_kind(row_kinds, idx), tie_ctx)
            if verdict == 'EMPTY':
                return bf.Region2D('empty')
            continue                # DROP: no geometry; γ keeps exp(λ·c_eff)
        rows.append((a0, a1, c_eff))
    t_min = _t_min_frame(free_ext_vals, tie_ctx)
    if t_min != float('-inf'):
        rows.append((1.0, 0.0, -t_min))
        rows.append((0.0, 1.0, -t_min))
    region = bf.region_2d(rows)
    if region.kind == 'empty' and region.reason in (
            'opposite', 'measure_zero'):
        # the exact emptiness tests (``_opposite_rows_empty``, measure zero)
        _RUNTIME_COUNTERS['polygon_zero_area'] += 1
    return region


def _polygon_modesum_dd(smooth_edge_modes, prefactor_complex,
                        subset_constraint_data, free_ext_vals,
                        pole_tuples, plan, row_kinds, tie_ctx):
    """``_integrate_2d_polygon_modesum`` on the DD kernels (M7): the region
    once per call, then ``Σ pref·C ∫ exp(α s_0 + β s_1 + γ)`` with γ folded
    into each piece's exponent (``boxfree.region_integral``); a non-OK status
    bails ('polygon_dd_overflow' / 'polygon_divergent')."""
    bf = _bf()
    region = _polygon_region_dd(subset_constraint_data, free_ext_vals,
                                row_kinds, tie_ctx)
    if region.kind == 'empty':
        return 0.0 + 0.0j
    if region.kind == 'lineality':
        # every exponent diverges along a line, so the integral is finite
        # only when every pole-tuple group cancels exactly (M8)
        pref = complex(prefactor_complex)
        terms = list(_modesum_tuples(
            smooth_edge_modes, subset_constraint_data, free_ext_vals, 2,
            pole_tuples, plan))
        live = {a for (C, a, _g) in terms if pref * C != 0}
        if live <= _cancelling_alphas(terms, pref):
            _RUNTIME_COUNTERS['divergent_cancelled'] += len(live)
            return 0.0 + 0.0j
        return _dd_status_bail(bf.DIVERGENT, 'polygon')
    pref = complex(prefactor_complex)
    tol = _decay_tol(smooth_edge_modes, pole_tuples, plan,
                     subset_constraint_data)
    total = None
    if plan is not None and plan.get(
            'n_tuples_full', len(plan['pole_tuples'])) >= _DD_BATCH_MIN:
        # M7b: every tuple at once when each term is plain (all OK);
        # None = the scalar loop below
        arrs = _m7b_plan_arrays(plan)
        A = None if arrs is None else arrs['A']
        if A is not None and A.ndim == 2 and A.shape[1] == 2:
            coefs = pref * arrs['C']
            gammas = _m7b_gammas(arrs, free_ext_vals)
            r = bf.region_modesum_mag(region, coefs, A[:, 0], A[:, 1],
                                      gammas, tol)
            if r is not None:
                total, mag = r
                scale = _dd_scale_np(coefs, gammas)
                nz = np.asarray(coefs) != 0      # as the scalar loop below
                rate = (float(np.max(np.abs(A[nz]))) if A.size and np.any(nz)
                        else 0.0)
    if total is None:
        terms = list(_modesum_tuples(
            smooth_edge_modes, subset_constraint_data, free_ext_vals, 2,
            pole_tuples, plan))
        dropped = set()
        total = 0.0 + 0.0j
        mag = scale = 0.0
        rate = 0.0
        for C_prod, alphas, gamma in terms:
            coef = pref * C_prod
            if coef == 0 or alphas in dropped:
                continue
            scale += _dd_scale_term(coef, gamma)
            rate = max(rate, abs(complex(alphas[0])), abs(complex(alphas[1])))
            st, v = bf.region_integral(region, alphas[0], alphas[1], gamma,
                                       tol)
            if st == bf.DIVERGENT and _divergent_group_cancels(
                    terms, alphas, pref):
                dropped.add(alphas)
                _RUNTIME_COUNTERS['divergent_cancelled'] += 1
                continue
            if st == bf.OVERFLOW:
                # the integral alone overflows: decide on the whole term
                st, v = bf.region_integral(region, alphas[0], alphas[1],
                                           gamma + cmath.log(coef), tol)
                if st == bf.OK:
                    total += v
                    mag += abs(v)
                    continue
            if st != bf.OK:
                return _dd_status_bail(st, 'polygon')
            total += coef * v
            mag += abs(coef * v)
    total = _finite_or_overflow(total, 'polygon')
    if total is not None:
        _dd_accuracy(total, mag, scale * _dbm.time_volume(
            rate, 2, _region2d_widths(region)))
        _RUNTIME_COUNTERS['dd_polygon'] += 1
        if region.kind == 'unbounded':
            _RUNTIME_COUNTERS['polygon_exact_unbounded'] += 1
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
#
# ACC-15: an absolute ±50 truncates a kernel wider than ~10 time units (and
# is not invariant under a change of the time unit).  ``None`` (the
# default): the box of each source is read off its kernel,
# ``_noise_kernel_halfwidth`` -- the |τ| beyond which |κ| stays below
# ``TAU_KERNEL_RTOL`` of its peak.  A number restores that absolute box.
TAU_KERNEL_CAP = None
TAU_KERNEL_RTOL = 1e-17
#: The legacy box, for an integrand whose kernel cannot be sampled.
_TAU_KERNEL_CAP_LEGACY = 50.0
# |τ| sampled for the kernel's extent: quarter decades over 1e-12 .. 1e12
_TAU_KERNEL_GRID = 10.0 ** (np.arange(-48, 49) / 4.0)


def _noise_kernel_halfwidth(kernels, tau_sym):
    """The half-width of the τ box of a non-local noise source whose
    cumulant kernels are the SR expressions ``kernels`` in ``tau_sym``
    (parameters substituted): the first sampled |τ| (quarter decades) beyond
    which every |κ(±τ)| stays below ``TAU_KERNEL_RTOL`` of the largest
    sampled |κ| -- a length of the kernel itself, so a change of the time
    unit moves it with the kernel.  ``None`` when a kernel cannot be
    sampled (the caller uses the legacy box)."""
    grid = _TAU_KERNEL_GRID
    # τ = 0 itself is not sampled (heaviside(0), δ(0) have no value; the
    # smallest |τ| stands for it)
    pts = np.concatenate([-grid[::-1], grid])
    mags = np.zeros(pts.size)
    n_ok = 0
    for k in kernels:
        e = SR(k)
        fk = None
        try:
            from sage.all import fast_callable, CDF
            fk = fast_callable(e, vars=[tau_sym], domain=CDF)
        except Exception:                                # noqa: BLE001
            fk = None
        vals = np.full(pts.size, np.nan)
        for i, t in enumerate(pts):
            try:
                v = (fk(float(t)) if fk is not None
                     else e.subs({tau_sym: float(t)}))
                vals[i] = abs(complex(v))
            except Exception:                            # noqa: BLE001
                continue                    # a point without a value
        ok = np.isfinite(vals)
        n_ok += int(ok.sum())
        mags = np.maximum(mags, np.where(ok, vals, 0.0))
    if n_ok == 0:
        return None
    peak = float(np.max(mags))
    if not math.isfinite(peak) or peak <= 0.0:
        return None
    big = np.nonzero(mags > TAU_KERNEL_RTOL * peak)[0]
    reach = float(np.max(np.abs(pts[big])))
    nxt = grid[grid > reach]
    return float(nxt[0]) if nxt.size else float(grid[-1])


def _line_decay_floor(propagator_data, num_params):
    """The slowest decay rate of the propagator's lines (the smallest
    damping ``|Im ω_p|`` of its poles), or ``None`` (no poles, an undamped
    one, or no value)."""
    try:
        poles = propagator_data.get('pole_vals')
    except AttributeError:
        return None
    if not poles:
        return None
    rates = []
    for p in poles:
        try:
            z = SR(p)
            if num_params:
                z = z.subs(num_params)
            rates.append(abs(complex(z).imag))
        except Exception:                                # noqa: BLE001
            return None
    r = min(rates)
    return r if r > 0.0 and math.isfinite(r) else None


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
# Four levers cut the setup that ``integrate_diagram`` pays per diagram; all
# are always on (their switches -- ``USE_SETUP_*``, ``DAEDALUS_SETUP_*`` and
# the umbrella ``DAEDALUS_PHASE_J_LEGACY_SETUP`` -- were retired in audit
# W7; each was ``np.array_equal`` to its legacy path on every result):
#
#   L1: a diagram whose prefactor is numerically exactly 0 returns a zero
#       contribution before any setup (``_zero_exit_applies``).
#   L2: model-level data built once per ``compute_correction_td`` call
#       (``PropagatorTD``).
#   L3: the SR objects only the shot-noise / SR-integrand branches need are
#       built on demand.
#   L5: the automorphism-order memo (``engine.diagrams.symmetry``).
#   M10 (L4a): the δ-edge eliminations by exact rational linear algebra
#       instead of Maxima (``sage_solve``) and SR ``==`` tests (see
#       ``_delta_solve_exact``).

# M10 validation mode (default OFF): every δ-subset is also solved the
# legacy way (``sage_solve``) and the two must
# agree EXACTLY: same eliminated variables, every substitution RHS equal as a
# polynomial and the same SR tree, the same residual external-time verdicts, and every constraint
# row (``a_int``, ``a_ext``, ``c0``) built from them equal with ``==``.  A
# mismatch raises ``DeltaSolveValidationError``.  On when the module bool is
# True or ``DAEDALUS_PHASE_J_VALIDATE_DELTA`` is truthy at CALL time.
VALIDATE_DELTA = _env_truthy('DAEDALUS_PHASE_J_VALIDATE_DELTA')


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


#: The pole engine's refusal of a kernel without separated rates names the
#: default engine (audit LOUD-05); it realizes such kernels when the kernel
#: has a state-space realization (repeated / close rates, polynomial factors).
_POLES_KERNEL_HINT = ('; the default state-space engine (unset '
                      'DAEDALUS_PHASE_J_ENGINE) can compute such kernels '
                      'exactly when they have a state-space realization: '
                      'the pole engine cannot (audit LOUD-05)')


class PropagatorTD:
    """Model-level data of one ``(propagator_data, num_params)``, built once
    per ``compute_correction_td`` call and shared by every
    ``integrate_diagram`` of that call (M5 L2).

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


def _lazy_smooth_thunk(G_t_obj, pi, ri, dt):
    """The deferred ``edge_info[i]['smooth_factor']`` (M5 L3): the very call
    the eager path makes, made on the first read."""
    def thunk():
        _RUNTIME_COUNTERS['setup_lazy_sr_built'] += 1
        return G_t_entry(G_t_obj, pi, ri, dt, include_heaviside=False)
    return thunk


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
    * ``cp`` is a plain number (``is_numeric()``) and ``complex(CDF(cp))``
      is exactly 0 -- the test the subset loop applies to ``cp`` times the
      delta coefficients, which for a plain 0 is the same (an exception or
      a symbolic expression: no early exit).
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
        pole_vals = propagator_data.get('pole_vals')
        C_mats = propagator_data.get('C_mats')
        if not len(pole_vals or ()):
            return False
        # incomplete residue data: the full path fails (or falls back) in
        # its setup; keep it
        if C_mats is None or len(C_mats) < len(pole_vals):
            return False
    except (AttributeError, TypeError):
        return False
    try:
        # a plain number only (what ``.subs(num_params)`` leaves of a
        # numeric prefactor): a symbolic expression that merely evaluates to
        # 0.0 could still multiply a delta coefficient into a nonzero value
        # in the subset loop
        if not SR(cp).is_numeric():
            return False
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

    # C0 output (S4): no subset contributes
    contribution.phase_j_parts = {
        'subsets': [], 'perms': [], 'compensation': 1, 'origin_leaf_idx': None,
        'free_ext_idx': [], 'k': n_ext}

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
         ('n_delta_subsets_dropped', 0),
         ('delta_contributions_multi', []),
         ('subset_diagnostics', []),
         ('cumulant_prefactor', None),
         ('has_cumulant_kernel', False)],
        {'stripped_integrand': lambda: _full()['stripped_integrand'],
         'constraints': lambda: _full()['constraints'],
         'edge_info': lambda: _full()['edge_info']})


# ── M10 (L4a): exact δ-edge elimination ──
# Every δ edge of a subset sets ``dt_e = 0``; ``integrate_diagram`` solves it
# for the first remaining integration variable that occurs.  On every model
# in the repo ``dt_e`` is (after the previous substitutions) a linear form in
# the vertex times with exactly rational coefficients, so the solution is
# ``x = -rest/a`` computed in QQ: the same polynomial ``sage_solve`` (Maxima)
# returns, rebuilt as an SR sum of ``Rational * symbol`` terms, which Pynac
# stores in its canonical form -- so every later ``.subs`` / ``.coefficient``
# / ``float`` sees the same expression and every row and number is
# unchanged.  Variables are matched by NAME (``str``), never by SR ``==``
# (whose ``bool`` may run randomized zero proofs).  Anything that is not an
# exactly-rational linear form in ``x`` (a float, complex or symbolic
# coefficient, a nonlinear term, a zero coefficient, an exception) goes to
# ``sage_solve`` unchanged (counter ``setup_delta_solve_fallback``).  No
# floating point is used in the elimination: a coefficient that is exactly 0
# stays exactly 0 (the exact-zero contract of ``_const_row_verdict`` and the
# Θ(0) rule).


class DeltaSolveValidationError(AssertionError):
    """``VALIDATE_DELTA``: the exact δ-elimination and ``sage_solve``
    disagree on a substitution, a residual verdict or a constraint row."""


def _validate_delta_on():
    """True when the M10 validation mode is on (call time)."""
    flag = VALIDATE_DELTA
    if flag is not True and flag is not False:
        raise ValueError(f'final_integral.VALIDATE_DELTA={flag!r}: expected '
                         f'True or False')
    return flag or _env_truthy('DAEDALUS_PHASE_J_VALIDATE_DELTA')


def _exact_rational(x):
    """The value of the numeric SR ``x`` as an element of QQ when it is an
    exact integer or rational number, else None (float, complex, ...)."""
    try:
        v = x.pyobject()
    except Exception:
        return None
    if isinstance(v, bool) or isinstance(v, float) or isinstance(v, complex):
        return None
    if isinstance(v, int):
        return _QQ(v)
    try:
        parent = v.parent()
    except AttributeError:
        return None
    if parent is _QQ or parent is _ZZ:
        return _QQ(v)
    return None


def _exact_linear_form(expr):
    """``(terms, const)`` when the SR expression ``expr`` is a linear form
    sum_i c_i * s_i + c whose coefficients and constant are all exact
    rationals: ``terms`` maps each symbol NAME to ``[symbol, c_i]`` (QQ;
    repeated symbols accumulate, a coefficient may come out 0) and ``const``
    is in QQ.  None for anything else (a float / complex / symbolic
    coefficient, a product of symbols, a power, a function, a constant such
    as ``pi``).  Structural only: no SR comparison, no simplification."""
    terms = {}
    const = _QQ(0)
    stack = [(expr, _QQ(1))]
    while stack:
        e, k = stack.pop()
        if e.is_numeric():
            q = _exact_rational(e)
            if q is None:
                return None
            const += k * q
            continue
        if e.is_symbol():
            ent = terms.get(str(e))
            if ent is None:
                terms[str(e)] = [e, k]
            else:
                ent[1] += k
            continue
        op = e.operator()
        if op is _sr_add_op:
            stack.extend((o, k) for o in e.operands())
            continue
        if op is _sr_mul_op:
            coef = k
            factor = None
            for o in e.operands():
                if o.is_numeric():
                    q = _exact_rational(o)
                    if q is None:
                        return None
                    coef *= q
                elif factor is None:
                    factor = o
                else:
                    return None             # product of two non-numbers
            if factor is None:
                const += coef
            else:
                stack.append((factor, coef))
            continue
        return None
    return terms, const


def _delta_eliminate_exact(eq_expr, var):
    """The SR RHS of ``var = RHS`` solving ``eq_expr == 0`` by exact
    rational linear algebra, or None when ``eq_expr`` is not an exactly
    rational linear form with a nonzero coefficient on ``var`` (by name)."""
    lf = _exact_linear_form(eq_expr)
    if lf is None:
        return None
    terms, const = lf
    name = str(var)
    ent = terms.get(name)
    if ent is None or ent[1] == 0:
        return None
    a = ent[1]
    rhs = SR(-const / a)
    for nm, (sym, c) in terms.items():
        if nm == name or c == 0:
            continue
        rhs = rhs + SR(-c / a) * sym
    return rhs


# ── Feature B: integrated shot-noise δ records ─────────────────────────
# A shot-noise subset (one residual external-time equality) that keeps
# integration variables after the δ elimination becomes a
# ``delta_contributions`` record whose ``coeff_fc`` is the subset's polytope
# integral, and one with several independent equalities a
# ``delta_contributions_multi`` record.  Neither enters ``contribution`` /
# ``C_tau``.  (The switch ``USE_SHOTNOISE_INTEGRATED`` /
# ``DAEDALUS_SHOTNOISE_INTEGRATED``, whose off path dropped those subsets
# silently, was retired in audit W7.)


def _shot_equality(eqs, free_ext_syms):
    """``(head, why)`` for the residual equalities ``eqs`` (each
    ``a·x + c = 0`` in ``free_ext_syms``).  One equality: ``head`` is
    ``{'equality_a', 'equality_c', 'equality_symbolic'}``, built exactly as
    for a tree-level record.  Several (a δ of codimension > 1, e.g. the
    t_1 = t_2 = t_3 spike of a third cumulant): ``{'equalities': [(a, c),
    ...], 'equalities_symbolic': [...]}``, a record of
    ``delta_contributions_multi``.  ``head`` is None with ``why`` =
    'not_linear' (no float coefficients: dropped), 'empty' (an equality
    with no external time: the subset never fires, a legitimate zero) or
    'dependent' (linearly dependent equalities: a δ of a δ, dropped)."""
    rows = []
    for eq in eqs:
        try:
            eq_a = [float(eq.coefficient(s)) for s in free_ext_syms]
            eq_c = float(eq.subs({s: 0 for s in free_ext_syms}))
        except (TypeError, ValueError):
            return None, 'not_linear'
        if all(abs(a) < 1e-15 for a in eq_a):
            return None, 'empty'
        rows.append((eq_a, eq_c))
    if len(rows) == 1:
        return {'equality_a': rows[0][0], 'equality_c': rows[0][1],
                'equality_symbolic': eqs[0]}, None
    import numpy as _np
    if _np.linalg.matrix_rank(_np.array([r[0] for r in rows])) < len(rows):
        return None, 'dependent'
    return {'equalities': rows, 'equalities_symbolic': list(eqs)}, None


def _shot_skip(subset_diagnostics, delta_edges, smooth_edges, reason):
    """Diagnostic for an integrated shot-noise subset that could not be
    built (skipped, as every such subset was before Feature B)."""
    subset_diagnostics.append({
        'delta_edges': delta_edges,
        'smooth_edges': smooth_edges,
        'status': 'shotnoise_with_remaining_int_vars',
        'reason': reason,
    })


def _rows_zero_on_surface(head, cdata):
    """Number of constraint rows of an integrated δ subset that involve no
    integration variable and vanish IDENTICALLY on the δ surface (the row is
    a combination of the surface's equalities).  ``_SurfaceCoeff`` weighs
    each such row Θ(0) = 0 (Itô: the coincidence is imposed by a δ, like a
    constant row left by the δ elimination), so the weight leaves out the
    term whose edge would need an equal-time retarded propagator; a nonzero
    count says the weight relies on that convention."""
    import numpy as _np
    rows = (head['equalities'] if 'equalities' in head
            else [(head['equality_a'], head['equality_c'])])
    E = _np.array([list(a) + [c] for a, c in rows], dtype=float)
    r0 = _np.linalg.matrix_rank(E)
    n = 0
    for a_int, a_ext, c0 in cdata:
        if any(abs(a) > 0 for a in a_int):
            continue
        if not any(abs(a) > 0 for a in a_ext):
            continue
        R = _np.vstack([E, list(a_ext) + [c0]])
        if _np.linalg.matrix_rank(R) == r0:
            n += 1
    return n


class _SurfaceCoeff:
    """``coeff_fc`` of an integrated δ record: ``coeff_fc(*x)`` is the
    subset integral at the free external times ``x`` (taken on the δ
    surface by the caller), with Θ(0) = 0 for a constant row that is exactly
    0 there (no tie context: the record is not one Wick permutation of a
    caller's times, and the coincidence is imposed by the δ itself, the
    0.2.0 Itô rule for rows left by the δ elimination; the record's
    ``theta0_rows_on_surface`` counts such rows)."""
    __slots__ = ('_cfn',)

    def __init__(self, cfn):
        self._cfn = cfn

    def __call__(self, *free_vals):
        return complex(self._cfn([float(v) for v in free_vals], None, None))


def _residual_is_zero_exact(eq):
    """True / False when the residual external-time equality ``eq`` is an
    exactly rational linear form that is / is not identically zero; None
    when it is not such a form (the caller then keeps the SR test)."""
    try:
        lf = _exact_linear_form(SR(eq))
    except Exception:
        return None
    if lf is None:
        return None
    terms, const = lf
    return const == 0 and all(c == 0 for _, c in terms.values())


def _delta_solve_legacy(edge_info, delta_edges, integration_vars):
    """The pre-M10 δ-edge elimination of ``integrate_diagram`` for one subset
    (``sage_solve`` + SR ``==``), factored out verbatim.  Returns
    ``(substitutions, remaining_int_vars, ext_time_equalities)``, or None
    when a solve fails (the subset is infeasible and skipped)."""
    substitutions = {}
    remaining_int_vars = list(integration_vars)
    ext_time_equalities = []  # residual constraints on ext times

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
                return None
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
    return substitutions, remaining_int_vars, ext_time_equalities


def _delta_solve_exact(edge_info, delta_edges, integration_vars):
    """``_delta_solve_legacy`` with the M10 lever: the same elimination
    order and the same substitutions, solved by ``_delta_eliminate_exact``
    (``sage_solve`` only as the fallback) with every variable test by name.
    Same return value."""
    substitutions = {}
    remaining_int_vars = list(integration_vars)
    ext_time_equalities = []

    for ei_idx in delta_edges:
        eq_expr = SR(edge_info[ei_idx]['dt_sym'].subs(substitutions))
        try:
            eq_names = {str(v) for v in eq_expr.variables()}
        except AttributeError:
            eq_names = set()
        int_var_to_eliminate = None
        for iv in remaining_int_vars:
            if str(iv) in eq_names:
                int_var_to_eliminate = iv
                break
        if int_var_to_eliminate is None:
            ext_time_equalities.append(eq_expr)
            continue
        try:
            new_rhs = _delta_eliminate_exact(eq_expr, int_var_to_eliminate)
        except Exception:
            new_rhs = None
        if new_rhs is not None:
            _RUNTIME_COUNTERS['setup_delta_solve_fast'] += 1
        else:
            _RUNTIME_COUNTERS['setup_delta_solve_fallback'] += 1
            try:
                sol = sage_solve(
                    eq_expr == 0, int_var_to_eliminate,
                    solution_dict=True,
                )
            except Exception:
                sol = []
            if not sol:
                return None
            new_rhs = sol[0][int_var_to_eliminate]
        x_name = str(int_var_to_eliminate)
        substitutions[int_var_to_eliminate] = new_rhs
        # first match only, as the legacy ``list.remove``
        del remaining_int_vars[next(
            i for i, iv in enumerate(remaining_int_vars)
            if str(iv) == x_name)]
        # Chain-resolve exactly as the legacy path (same SR ``.subs``), with
        # the "does this RHS mention x" test by name.
        affected_keys = []
        for _k, _rhs in substitutions.items():
            if str(_k) == x_name:
                continue
            try:
                _rhs_vars = SR(_rhs).variables()
            except (AttributeError, TypeError):
                continue
            if any(str(v) == x_name for v in _rhs_vars):
                affected_keys.append(_k)
        if affected_keys:
            _chain_subs = {int_var_to_eliminate: new_rhs}
            for _k in affected_keys:
                substitutions[_k] = SR(substitutions[_k]).subs(_chain_subs)
    return substitutions, remaining_int_vars, ext_time_equalities


def _delta_constraint_rows(edge_info, smooth_edges, substitutions,
                           remaining_int_vars, free_ext_syms):
    """The ``(a_int, a_ext, c0)`` float rows ``integrate_diagram`` extracts
    from the smooth edges of a subset (same expressions), or an
    ``('error', name)`` marker where that extraction raises."""
    rows = []
    for ei_idx in smooth_edges:
        c_sr = SR(SR(edge_info[ei_idx]['dt_sym']).subs(substitutions))
        try:
            a_int = [float(c_sr.coefficient(v)) for v in remaining_int_vars]
            a_ext = [float(c_sr.coefficient(s)) for s in free_ext_syms]
            zero_subs = {v: 0 for v in list(remaining_int_vars)
                         + list(free_ext_syms)}
            c0 = float(c_sr.subs(zero_subs))
        except (TypeError, ValueError) as exc:
            rows.append(('error', type(exc).__name__))
            continue
        rows.append((a_int, a_ext, c0))
    return rows


def _residual_rows(eqs, free_ext_syms):
    """The shot-noise ``(equality_a, equality_c)`` extraction of each
    residual equality, or an ``('error', name)`` marker."""
    rows = []
    for eq in eqs:
        eq = SR(eq)
        try:
            rows.append(([float(eq.coefficient(s)) for s in free_ext_syms],
                         float(eq.subs({s: 0 for s in free_ext_syms}))))
        except (TypeError, ValueError) as exc:
            rows.append(('error', type(exc).__name__))
    return rows


def _legacy_residual_is_zero(eq):
    try:
        return bool(eq.is_zero())
    except Exception:
        return False


def _validate_delta_subset(fast, legacy, edge_info, smooth_edges,
                           free_ext_syms, where):
    """``VALIDATE_DELTA``: raise ``DeltaSolveValidationError`` unless the
    exact solve ``fast`` and the ``sage_solve`` one ``legacy`` (both as
    returned by ``_delta_solve_*``) agree exactly; count the validated
    eliminations in ``setup_delta_validated``."""
    def fail(msg):
        raise DeltaSolveValidationError(f'M10 δ-solve mismatch ({where}): '
                                        f'{msg}')

    if (fast is None) != (legacy is None):
        fail(f'feasibility differs: exact {fast is not None}, '
             f'sage_solve {legacy is not None}')
    if fast is None:
        return
    f_subs, f_rem, f_ext = fast
    l_subs, l_rem, l_ext = legacy
    if [str(v) for v in f_rem] != [str(v) for v in l_rem]:
        fail(f'remaining variables differ: {f_rem} vs {l_rem}')
    f_by = {str(k): v for k, v in f_subs.items()}
    l_by = {str(k): v for k, v in l_subs.items()}
    if list(f_by) != list(l_by):
        fail(f'eliminated variables differ: {list(f_by)} vs {list(l_by)}')
    for name in f_by:
        diff = (SR(f_by[name]) - SR(l_by[name])).expand()
        if not diff.is_trivial_zero():
            fail(f'{name} = {f_by[name]} (exact) vs {l_by[name]} '
                 f'(sage_solve): difference {diff}')
        # the same SR tree too: the RHS reaches ``.subs`` / ``fast_callable``
        if not SR(f_by[name]).is_trivially_equal(SR(l_by[name])):
            fail(f'{name} = {f_by[name]} (exact) is the same polynomial as '
                 f'{l_by[name]} (sage_solve) but not the same SR tree')
    if len(f_ext) != len(l_ext):
        fail(f'{len(f_ext)} vs {len(l_ext)} residual equalities')
    for fe, le in zip(f_ext, l_ext):
        diff = (SR(fe) - SR(le)).expand()
        if not diff.is_trivial_zero():
            fail(f'residual equality {fe} vs {le}')
        verdict = _residual_is_zero_exact(fe)
        if verdict is not None and verdict != _legacy_residual_is_zero(le):
            fail(f'residual {fe}: exact zero-test {verdict}, SR is_zero '
                 f'{not verdict}')
    if (_residual_rows(f_ext, free_ext_syms)
            != _residual_rows(l_ext, free_ext_syms)):
        fail('residual equality rows differ')
    f_rows = _delta_constraint_rows(edge_info, smooth_edges, f_subs, f_rem,
                                    free_ext_syms)
    l_rows = _delta_constraint_rows(edge_info, smooth_edges, l_subs, l_rem,
                                    free_ext_syms)
    if f_rows != l_rows:
        fail(f'constraint rows differ: {f_rows} vs {l_rows}')
    _RUNTIME_COUNTERS['setup_delta_validated'] += len(f_by)


class _LazyIntegrand:
    """S3: the pole routes' closure of a subset the state-space route takes,
    built on its first call (the hook payload's reference ``integrand``)."""
    __slots__ = ('_mk', '_sr', '_fc', '_fn')

    def __init__(self, mk_fast, sr_build, fc):
        self._mk, self._sr, self._fc, self._fn = mk_fast, sr_build, fc, None

    def __call__(self, *args):
        if self._fn is None:
            fn = self._mk()
            if fn is None:
                fn = self._fc
            if fn is None:
                kind, val = self._sr()
                if kind == 'zero':
                    fn = (lambda *_a: 0j)
                elif kind != 'ok':
                    raise RuntimeError(
                        f'the pole routes\' integrand of this subset cannot '
                        f'be built ({kind}: '
                        f'{val if isinstance(val, str) else val.get("reason")})')
                else:
                    fn = val
            self._fn = fn
        return self._fn(*args)


def _spec_leg_keys(spec):
    """``((field, leg), (field, leg))`` of an order-2 cumulant spec (field
    ``None`` when the spec does not name its leg fields)."""
    lf = spec.get('leg_fields') or (None, None)
    return ((lf[0], spec['legs'][0]), (lf[1], spec['legs'][1]))


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
    on_divergence=None,
    _conv_assign=None,
    _conv_force_leg=None,
    _conv_monomial=None,
):
    r"""
    Vertex-time integration for a typed Feynman diagram at ANY loop
    order: closed forms on numeric poles per δ-subset (box-free by
    default), the exact difference-constraint route and, last, a hardened
    quadrature fallback (see the module docstring).

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
        it matches ``propagator_data`` / ``num_params``.
    on_divergence : None or 'fallback' | 'warn' | 'raise'
        The divergence policy of the box-free routes (M8; see the
        "Divergence policy" block).  ``None``: ``PHASE_J_ON_DIVERGENCE``
        at evaluation time.

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
    if on_divergence is not None:
        _divergence_policy(on_divergence)          # validate early
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

    # ── 0. Zero-prefactor early exit (M5 L1) ──
    # A diagram whose prefactor is numerically exactly 0 contributes 0 at
    # every point: every subset is skipped at the "numerical zero-skip"
    # below.  Return that result before ``build_G_t_matrix`` and the
    # per-edge setup (most of this function's time on a model with many
    # structurally-zero diagrams).  ``_setup_full`` is set by the lazy keys
    # of the early result: they re-run the full path to build themselves.
    _cp_pre = None
    if _setup_full is None and _SUBSET_HOOK is None:
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
                    _setup_full=serial,
                    on_divergence=on_divergence))

    # ── 1. Numerical G(t) matrix (smooth + delta parts) ──────────
    # M5 L2: the matrix, the poles and the per-entry residues come from the
    # call-scoped ``PropagatorTD`` when one that matches this propagator data
    # was handed in.
    _ptd = None
    if prop_td is not None:
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
            from engine.diagrams import symmetry as _symmetry
            if len(_all_mappings) == 1:
                # M5 L5: one mapping means every external field occurs once,
                # so the leaves are pairwise distinguished by their field in
                # ``_colored_incidence_digraph`` whether or not they are
                # fixed: Aut(leaves free) = Aut(leaves fixed) and the
                # compensation |Aut_free| / |Aut_fixed| is exactly 1.
                _compensation = 1
                _RUNTIME_COUNTERS['setup_aut_wick_skipped'] += 1
            else:
                _compensation = _symmetry.external_wick_compensation(
                    typed_diagram)
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
    noise_tau_box = {}          # {tau_sym: half-width of its box}
    leg_kernel_rates = []       # decay rates of the conductance kernels
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
    # NoiseSourceType: a PHYSICAL leg ``y`` that a kernel attaches to
    # (``x·(g ∗ y)``) sees ``∫ ds g(t − s) y(s)``, ``t`` the vertex time
    # (the anchor, where the response leg and every other physical leg sit).
    # See ``docs/conductance_vertex_kernels_design.md``.
    #
    # Each copy of a leg tuple is a SLOT: the k-th kernel symbol's power in
    # the vertex coefficient is the number of copies it occupies, the rest
    # are plain.  The edges whose HEAD is this vertex at that leg tuple are
    # the lines to place in the slots.  When all slots of a leg tuple are
    # alike the placement is unique; otherwise (``x·(g ∗ x)``: one plain
    # and one kernel copy of the same leg) the diagram's weight M (which
    # counts the copies as identical) multiplies the AVERAGE over the
    # distinct placements, which this function computes by building each
    # placement (``_conv_assign``) and averaging.
    #
    # A line placed in a kernel slot is integrated one of two ways:
    # * 'dress' (the default): a kernel that is a finite exponential sum
    #   (``_extract_exp_modes``) is folded into that line's propagator,
    #   ``(g ∗ G)(Δt)`` with Δt = t − t_tail, again a finite pole sum
    #   (``_dress_edge_modes``) with no δ part (item 56: a kernel with a
    #   direct term c δ keeps c times the line's δ part): no extra time, no
    #   extra row;
    # * 'leg': otherwise (a kernel that is not an exponential sum, a kernel
    #   rate equal to a propagator pole, no mode data) the attached leg gets
    #   its own integration time ``s`` and the kernel becomes one more
    #   smooth edge ``s → t`` (section 3a'), every row a difference row.
    conv_leg_time = {}          # {(v, edge_key): s symbol}   ('leg')
    kernel_edge_specs = []      # [(v, att, s_sym)]             ('leg')
    conv_dress = {}             # {edge_key: kernel modes}      ('dress')
    conv_dress_att = {}         # {edge_key: the attachment}    ('dress', S3)
    conv_dress_direct = {}      # {edge_key: c} a direct term c δ (item 56)
    # S3: the state-space route realizes a kernel exactly (equal or close
    # rates, a Jordan-type chain; ``api._realization.realize_kernel``), so a
    # kernel without separated modes is no longer refused when that route is
    # on: it takes the 'leg' path and the subsets without a pole route go to
    # the state-space route (``_ss_only_kernels``: kernels with no image the
    # pole routes can use at all).
    _engine_poles = _env.phase_j_engine() == 'poles'
    _ss_only_kernels = []
    _ss_only_idx = set()        # their kernel edges (indices into edge_info)

    def _ss_only_failure(bits):
        return {
            'status': 'failed',
            'contribution': None,
            'integration_vars': integration_vars,
            'reason': (
                f'[subset {bin(bits)}] the conductance kernel(s) '
                f'{sorted(set(_ss_only_kernels))} have no exponential '
                f'modes (a repeated or near-repeated rate) and only the '
                f'state-space route realizes them, but this subset needs '
                f'the pole routes (the state-space route declined it: see '
                f'the ss_declined_by_reason counters); write the kernel as '
                f'an auxiliary linear field (Markov embedding) or give it a '
                f'time-domain image'),
        }

    def _ss_realizes_kernel(att):
        if _engine_poles:
            return False
        from engine.integration.time_domain import statespace as _ssk
        if not _ssk.engine_requested():
            return False
        _model, _why = _ssk.realization_for(propagator_data, num_params)
        return _model is not None and not isinstance(
            _model.kernel_entry(att), str)
    # Item 60 (bug 14): a vertex coefficient that is a SUM of terms with
    # different powers of the kernel symbols (``-ee*x*x - c*x*Conv(h, x)``:
    # coefficient ``c*z_h + ee`` on one leg tuple) is linear in its terms:
    # each term has its own slot count (a plain term has no kernel slot).
    # The diagram is integrated once per choice of one term per such vertex
    # (``_conv_monomial``) with the prefactor's factor of that vertex
    # replaced by the term, and the results summed.  Before, the slots were
    # counted on the whole coefficient and the plain term was convolved too.
    if _conv_monomial is None:
        _split = _conv_monomial_split(vertex_assignments, combined_prefactor)
        if isinstance(_split, str):
            return {'status': 'failed', 'contribution': None,
                    'integration_vars': integration_vars, 'reason': _split}
        if _split is not None:
            return _sum_conv_monomials(
                _split,
                dict(typed_diagram=typed_diagram,
                     propagator_data=propagator_data,
                     ext_time_vars=ext_time_vars, num_params=num_params,
                     origin_leaf_idx=origin_leaf_idx,
                     external_fields=external_fields,
                     edge_mode_sums_builder=edge_mode_sums_builder,
                     prop_td=prop_td, on_divergence=on_divergence,
                     _conv_assign=_conv_assign,
                     _conv_force_leg=_conv_force_leg))
    conv_refusal = None
    _conv_choices = []          # [((v, leg), [placement, ...])]
    _conv_slots = {}            # {(v, leg): (in_edges, base placement)}
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
        phys_legs = [tuple(lg) for lg in vtype.physical_legs]
        mono = None if _conv_monomial is None else _conv_monomial.get(v)
        coeff_sr = SR(vtype.coefficient if mono is None else mono)
        by_leg = {}
        for att_idx, att in enumerate(vtype.kernel_attachments):
            try:
                q = int(coeff_sr.degree(SR(att['symbol'])))
            except Exception:
                q = 1
            # one term of a split coefficient: exactly its power (a plain
            # term, power 0, fills no kernel slot)
            by_leg.setdefault(tuple(att['leg']), []).extend(
                [att_idx] * (max(q, 1) if mono is None else max(q, 0)))
        for leg, slots in by_leg.items():
            n_copies = phys_legs.count(leg)
            hits = sorted(ek for ek in D.edges() if ek[1] == v
                          and tuple(typed_diagram.edge_types[ek][1]) == leg)
            if len(slots) > n_copies or len(hits) != n_copies:
                conv_refusal = (
                    f'conductance vertex {v}: {len(slots)} kernel slot(s) on '
                    f'the leg {leg}, which occurs {n_copies} times among the '
                    f'vertex\'s physical legs {phys_legs} and is met by '
                    f'{len(hits)} edge(s)')
                break
            slots = tuple(slots) + (None,) * (n_copies - len(slots))
            placements = sorted(set(itertools.permutations(slots)),
                                key=lambda pl: tuple(-1 if x is None else x
                                                     for x in pl))
            _conv_slots[(v, leg)] = (hits, placements[0])
            if len(placements) > 1:
                _conv_choices.append(((v, leg), placements))
        if conv_refusal is not None:
            break
    if conv_refusal is not None:
        return {
            'status': 'failed',
            'contribution': None,
            'integration_vars': integration_vars,
            'reason': conv_refusal,
        }
    if _conv_choices and _conv_assign is None:
        return _average_conv_placements(
            _conv_choices,
            dict(typed_diagram=typed_diagram, propagator_data=propagator_data,
                 combined_prefactor=combined_prefactor,
                 ext_time_vars=ext_time_vars, num_params=num_params,
                 origin_leaf_idx=origin_leaf_idx,
                 external_fields=external_fields,
                 edge_mode_sums_builder=edge_mode_sums_builder,
                 prop_td=prop_td, on_divergence=on_divergence,
                 _conv_force_leg=_conv_force_leg,
                 _conv_monomial=_conv_monomial))
    _assign = dict(_conv_assign or {})
    _can_dress = (edge_mode_sums_builder is None
                  and propagator_data.get('C_mats') is not None
                  and _dress_kernels_on())
    for (v, leg), (hits, base) in _conv_slots.items():
        placement = _assign.get((v, leg), base)
        vtype = vertex_assignments[v]
        att_pairs = conv_vertex_specs.setdefault(v, [])
        for ek, att_idx in zip(hits, placement):
            if att_idx is None:
                continue
            att = vtype.kernel_attachments[att_idx]
            why = _conv_kernel_unsupported(att)
            if why is not None:
                return {
                    'status': 'failed',
                    'contribution': None,
                    'integration_vars': integration_vars,
                    'reason': f'conductance vertex {v}: {why}',
                }
            if _conv_kernel_local_weight(att, num_params) is not None:
                # ``Conv(c·δ, n) = c·n``: no delay, the line stays as it is
                # (the kernel symbol becomes its weight in section 3c).
                att_pairs.append((None, att))
                continue
            kmodes = _kernel_modes(att, num_params)
            kdirect = None
            if kmodes is None:
                # item 56: a direct term, g = c δ + (a pole sum): the
                # all-pass Pade (n, n) image of a bare hard delay at a
                # vertex leg (x*x(t - d)).  Dressed, the line becomes
                # c G + (g_smooth ∗ G), on both routes; a kernel edge
                # ('leg') cannot hold it (refused by name below)
                _split = _kernel_direct_split(att, num_params)
                if _split is not None:
                    kdirect, kmodes = _split
            if kmodes is None:
                leg_kernel_rates.append(None)
            else:
                leg_kernel_rates.extend(-complex(mu).real
                                        for (_c, mu) in kmodes)
            if (kdirect is not None
                    and not (_can_dress
                             and ek not in (_conv_force_leg or ()))):
                _RUNTIME_COUNTERS['conv_kernel_direct_refused'] += 1
                return {
                    'status': 'failed',
                    'contribution': None,
                    'integration_vars': integration_vars,
                    'reason': (f'conductance vertex {v}: the kernel '
                               f'{att["symbol"]} has a direct term '
                               f'({complex(kdirect):.6g} δ next to a pole '
                               f'sum, e.g. the Pade (n, n) image of a bare '
                               f'delay) and this line cannot be dressed '
                               f'with it (a kernel rate equal to a pole of '
                               f'the line, or no propagator mode data); a '
                               f'kernel edge holds only a strictly proper '
                               f'kernel'),
                }
            if kmodes is None and _ss_realizes_kernel(att):
                # S3: no separated modes (a repeated / close rate, a
                # polynomial factor), but the state-space route realizes the
                # kernel: a kernel edge ('leg'), no mode sum
                _RUNTIME_COUNTERS['conv_kernel_ss_lifted'] = (
                    _RUNTIME_COUNTERS.get('conv_kernel_ss_lifted', 0) + 1)
            elif kmodes is None:
                return {
                    'status': 'failed',
                    'contribution': None,
                    'integration_vars': integration_vars,
                    'reason': (f'conductance vertex {v}: the kernel '
                               f'{att["symbol"]} is not a finite sum of '
                               f'exponentials with separated rates (a '
                               f'repeated or near-repeated rate, a '
                               f'polynomial factor such as t e^{{-t/tau}}, '
                               f'or no usable image); write it as an '
                               f'auxiliary linear field (Markov embedding)'
                               + (_POLES_KERNEL_HINT if _engine_poles
                                  else '')),
                }
            if (kmodes is not None and _can_dress
                    and ek not in (_conv_force_leg or ())):
                conv_dress[ek] = kmodes
                conv_dress_att[ek] = att
                if kdirect is not None:
                    conv_dress_direct[ek] = kdirect
                    _RUNTIME_COUNTERS['conv_lines_dressed_direct'] += 1
                att_pairs.append(('dress', att))
                continue
            s_sym = SR.var(
                f's_v{v}_gleg{len(kernel_edge_specs)}_td_',
                latex_name=rf's^{{g}}_{{v_{{{v}}},{len(kernel_edge_specs)}}}'
            )
            integration_vars.append(s_sym)
            conv_leg_time[(v, ek)] = s_sym
            att_pairs.append((s_sym, att))
            kernel_edge_specs.append((v, att, s_sym))
            _RUNTIME_COUNTERS['conv_kernel_leg_edges'] += 1

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

        For ConvVertexType the map ``conv_leg_time`` is keyed on
        ``(vertex, edge_key)`` and holds only the edges whose HEAD is the
        vertex at its kernel-attached physical leg (section 2c); every
        other end of an edge at that vertex sits at the anchor.

        Plain vertices keep their single ``vertex_time`` entry — the
        routing falls through to ``default_time``.
        """
        if conv_leg_time and edge_key[1] == vert:
            s_leg = conv_leg_time.get((vert, edge_key))
            if s_leg is not None:
                return s_leg
        if vert not in vertex_leg_time:
            return default_time
        return vertex_leg_time[vert].get(edge_key, default_time)

    # M5 L3: ``smooth_factor`` (an SR object from ``G_t_entry``) is read only
    # by the shot-noise branch and the SR-integrand branch below; build it on
    # first read.  The key stays present in every ``edge_info`` record.
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
        _RUNTIME_COUNTERS['setup_lazy_sr_deferred'] += 1
        edge_info.append(_LazyDict(
            [('u', u), ('v', v), ('lbl', lbl), ('ri', ri), ('pi', pi),
             ('dt_sym', dt), ('delta_coeff', delta_c),
             ('smooth_factor', None)],
            {'smooth_factor': _lazy_smooth_thunk(G_t_obj, pi, ri, dt)}))

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

    # ── 3a'. Conductance kernels as edges ─────────────────────────
    # Each delayed kernel attachment (section 2c) is one more smooth edge
    # ``s → anchor`` with Δt = anchor − s, no δ part and smooth factor
    # g(Δt) (Heaviside stripped: the Δt > 0 row is the edge's retardation
    # row, as for a propagator).  Appended AFTER the propagator edges, so
    # the propagator-indexed passes (mode sums above, forced-δ pruning
    # below) see only ``edge_info[:n_prop_edges]``.  A kernel that is one
    # exponential C·e^{λΔt} also gets its mode, and the diagram stays on
    # the mode-sum routes; any other kernel (alpha function, ...) sets
    # ``edge_mode_sums`` to None and the diagram takes the SR path.
    n_prop_edges = len(edge_info)
    # Dressed lines (section 2c 'dress'): the line's propagator G becomes
    # (g ∗ G), a pole sum (its δ part c D only for a kernel with a direct
    # term c, item 56), at the same times.
    dressed_idx = set()
    if conv_dress:
        if edge_mode_sums is None:
            return {
                'status': 'failed',
                'contribution': None,
                'integration_vars': integration_vars,
                'reason': ('conductance kernel: the propagator has no '
                           'mode data to dress a line with'),
            }
        # item 56 (D10): each line's δ part after its own dressing (a
        # strictly proper kernel removes it, a direct term c keeps c D)
        _delta_after = {}
        if conv_dress_direct:
            for j, ej in enumerate(edge_info):
                kj = (ej['u'], ej['v'], ej['lbl'])
                Dj = complex(edge_mode_sums[j].delta_coeff)
                _delta_after[j] = (Dj if kj not in conv_dress
                                   else Dj * complex(conv_dress_direct.get(kj, 0)))
        for i, ei in enumerate(edge_info):
            kmodes = conv_dress.get((ei['u'], ei['v'], ei['lbl']))
            if kmodes is None:
                continue
            kdirect = conv_dress_direct.get((ei['u'], ei['v'], ei['lbl']))
            if (kdirect is not None and abs(_delta_after.get(i, 0)) >= 1e-15
                    and any(abs(_delta_after.get(j, 0)) >= 1e-15
                            for j, ej in enumerate(edge_info)
                            if j != i and ej['v'] == ei['v'])):
                # item 56 (D10): the direct term passes this line's δ part
                # (a spike train) to a vertex where another physical line
                # has one too: a product of δ functions at one time (exp of
                # a delayed spike train), which is not defined
                _RUNTIME_COUNTERS['conv_kernel_direct_refused'] += 1
                return {
                    'status': 'failed',
                    'contribution': None,
                    'integration_vars': integration_vars,
                    'reason': (f'conductance vertex {ei["v"]}: the kernel '
                               f'{conv_dress_att[(ei["u"], ei["v"], ei["lbl"])]["symbol"]} '
                               f'has a direct term (a bare delay) on a line '
                               f'with a delta part (a point-process field) at '
                               f'a vertex where another line has one too: it '
                               f'multiplies delta functions, which is not '
                               f'defined; give the delay a filter'),
                }
            dressed = (_dress_edge_modes(edge_mode_sums[i], kmodes)
                       if kdirect is None else
                       _dress_edge_modes(edge_mode_sums[i], kmodes,
                                         direct=kdirect))
            if dressed is None:
                return integrate_diagram(
                    typed_diagram, propagator_data, combined_prefactor,
                    ext_time_vars, num_params=num_params,
                    origin_leaf_idx=origin_leaf_idx,
                    external_fields=external_fields,
                    edge_mode_sums_builder=edge_mode_sums_builder,
                    prop_td=prop_td, on_divergence=on_divergence,
                    _conv_assign=_conv_assign,
                    # the coefficient term this call integrates (a split
                    # coefficient, item 60): without it the rerun splits
                    # the term's own prefactor again and refuses it
                    _conv_monomial=_conv_monomial,
                    _conv_force_leg=(set(_conv_force_leg or ())
                                     | {(ei['u'], ei['v'], ei['lbl'])}))
            _RUNTIME_COUNTERS['conv_lines_dressed'] += 1
            edge_mode_sums[i] = dressed
            dt = ei['dt_sym']
            sf = SR(0)
            for (C, lam) in dressed.modes:
                sf = sf + _sr_complex(C) * (_sr_complex(lam) * dt).exp()
            edge_info[i] = {
                'u': ei['u'], 'v': ei['v'], 'lbl': ei['lbl'],
                'ri': ei['ri'], 'pi': ei['pi'],
                'dt_sym': dt,
                # item 56: a kernel direct term c on a line with a δ part D
                # leaves c D δ (zero otherwise, as before)
                'delta_coeff': (0.0 if kdirect is None
                                or complex(dressed.delta_coeff) == 0
                                else complex(dressed.delta_coeff)),
                'smooth_factor': sf,
            }
            dressed_idx.add(i)
        for i in range(n_prop_edges):
            if i in dressed_idx:
                continue
            ems_i = edge_mode_sums[i]
            # an undressed line: its negligible modes against its own norms
            # (a small residue at a slow rate is a real tail and is kept)
            sup_i = sum(abs(complex(C)) for (C, _l) in ems_i.modes)
            l1_i = sum(_mode_l1(C, l) for (C, l) in ems_i.modes)
            keep = tuple((C, lam) for (C, lam) in ems_i.modes
                         if not _negligible_mode(C, lam, sup_i, l1_i))
            if len(keep) < len(ems_i.modes):
                _RUNTIME_COUNTERS['conv_dress_zero_modes_dropped'] += (
                    len(ems_i.modes) - len(keep))
                edge_mode_sums[i] = EdgeModeSum(
                    ri=ems_i.ri, pi=ems_i.pi,
                    delta_coeff=ems_i.delta_coeff, modes=keep)
    if kernel_edge_specs:
        from sage.all import heaviside as _sage_heaviside
        _tau_ph = SR.var('_conv_kernel_tau_td_')
        for (v, att, s_sym) in kernel_edge_specs:
            td_fn = att.get('kernel_td_fn')
            g_tau = None
            if td_fn is None:
                fmodes = (_kernel_modes(att, num_params)
                          if att.get('kernel_ft_fn') is not None else None)
                if fmodes is None and _ss_realizes_kernel(att):
                    # S3: a frequency image without separated modes: only
                    # the state-space route has this kernel (no time-domain
                    # expression for the pole routes' SR integrand)
                    _ss_only_kernels.append(str(att['symbol']))
                elif fmodes is None:
                    return {
                        'status': 'failed',
                        'contribution': None,
                        'integration_vars': integration_vars,
                        'reason': (f'conductance vertex {v}: the kernel '
                                   f'{att["symbol"]} has no time-domain '
                                   f'image and its frequency image is not '
                                   f'a proper rational function with simple '
                                   f'retarded poles (declare it with '
                                   f'time_expr)'
                                   + (_POLES_KERNEL_HINT if _engine_poles
                                      else '')),
                    }
                else:
                    g_tau = SR(0)
                    for (C, lam) in fmodes:
                        g_tau = g_tau + _sr_complex(C) * (
                            _sr_complex(lam) * _tau_ph).exp()
            else:
                g_tau = SR(td_fn(_tau_ph)).substitute_function(
                    _sage_heaviside, lambda _x: SR(1))
            if num_params and g_tau is not None:
                g_tau = g_tau.subs(num_params)
            dt = SR(vertex_time[v] - s_sym)
            if g_tau is None:
                # S3: only the state-space route has this kernel; a formal
                # function stands for it in the symbolic factors (display;
                # an SR integrand of such a subset is refused by name)
                from sage.all import function as _sage_function
                _ss_only_idx.add(len(edge_info))
                _sf = _sage_function(
                    f'{att["symbol"]}_statespace_only')(dt)
            else:
                _sf = g_tau.subs({_tau_ph: dt})
            edge_info.append({
                'u': None, 'v': v, 'lbl': ('kernel', str(att['symbol'])),
                'ri': None, 'pi': None,
                'dt_sym': dt,
                'delta_coeff': 0.0,
                'smooth_factor': _sf,
            })
            kmodes = (None if g_tau is None
                      else _extract_exp_modes(g_tau, _tau_ph))
            if kmodes is None or edge_mode_sums is None:
                edge_mode_sums = None
            else:
                edge_mode_sums.append(EdgeModeSum(
                    ri=-1, pi=-1, delta_coeff=complex(0.0),
                    modes=tuple((complex(C), complex(lam))
                                for (C, lam) in kmodes)))

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
            kv = []
            # tau_sym_v is oriented by the FIRST spec's legs (2b: its leg 1
            # sits at anchor - tau); a spec with the legs the other way
            # round reads its kernel at -tau (an asymmetric cross kernel,
            # k_ij(tau) != k_ij(-tau), was evaluated at the wrong sign and
            # the pair came out symmetrized: audit W6, found with GEN-05)
            ref = _spec_leg_keys(specs[0])
            for spec in specs:
                i_leg, j_leg = spec['legs'][0], spec['legs'][1]
                own = _spec_leg_keys(spec)
                tau_eff = (-tau_sym_v if own != ref and own == ref[::-1]
                           else tau_sym_v)
                kappa_subs[spec['symbol']] = SR(
                    spec['kernel_fn'](i_leg, j_leg, tau_eff)
                )
                kv.append(kappa_subs[spec['symbol']].subs(num_params)
                          if num_params else kappa_subs[spec['symbol']])
            if TAU_KERNEL_CAP is not None:
                noise_tau_box[tau_sym_v] = float(TAU_KERNEL_CAP)
            else:
                hw = _noise_kernel_halfwidth(kv, tau_sym_v)
                noise_tau_box[tau_sym_v] = (
                    hw if hw is not None else _TAU_KERNEL_CAP_LEGACY)
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

    # ── 3c. Conductance-vertex kernel symbols ─────────────────────
    # The vertex coefficient still carries the kernel symbol (``z_g_*``,
    # what ``Conv(g, dn) → z_g · dn`` left).  A delayed kernel lives in its
    # edge (section 3a'), so its symbol is 1 here; a δ kernel (``Conv(δ, n)
    # = n``) has no edge and its symbol is its weight.
    if conv_vertex_specs:
        g_subs = {}
        for v, att_pairs in conv_vertex_specs.items():
            for s_sym, att in att_pairs:
                g_subs[att['symbol']] = (
                    SR(1) if s_sym is not None
                    else SR(_conv_kernel_local_weight(att, num_params)))
        cp = cp.subs(g_subs)
        if num_params:
            cp = cp.subs(num_params)

    # The hardened fallback's decay certificate for an integrand without
    # mode data: the slowest line and kernel rate (``None``: unknown), and
    # the parameter point its warnings are keyed by.
    _kappa_floor = _line_decay_floor(propagator_data, num_params)
    if _kappa_floor is not None and leg_kernel_rates:
        if any(r is None or not r > 0.0 for r in leg_kernel_rates):
            _kappa_floor = None
        else:
            _kappa_floor = min(_kappa_floor, min(leg_kernel_rates))
    try:
        _params_fp = hash(tuple(sorted(
            (str(k), repr(v)) for k, v in (num_params or {}).items())))
    except Exception:                                    # noqa: BLE001
        _params_fp = None

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

    # M2a structural zeros (decided when the subsets are built): the
    # mirror image of the forced-smooth edges.  An edge whose
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
        _forced_delta_edges(edge_info[:n_prop_edges], propagator_data,
                            num_params=num_params)
        if edge_mode_sums_builder is None
        else frozenset())
    if dressed_idx and _forced_delta:
        # a dressed line always has a smooth part (D g + c G + g ∗ G), so it
        # is never a forced δ edge (item 56: its δ part is c D, zero unless
        # its kernel has a direct term)
        _forced_delta = frozenset(_forced_delta) - dressed_idx

    # Expose the |S|=0 (all smooth) symbolic integrand and constraints
    # for debugging / display, matching the pre-fix return shape.  Kept as
    # the unexpanded product: nothing reads it numerically, and expanding it
    # blows up combinatorially for multi-mode kernels (alpha-function
    # synapses: >10 min per diagram on quadratic_hawkes_alpha at ell=1).
    # (M5 L3: display only -- nothing in the package reads it -- so it is
    # built on first read of the result's ``stripped_integrand``.)
    _ds_cell = []

    def _display_stripped_value():
        if not _ds_cell:
            _ds = cp
            for ei in edge_info:
                _ds = _ds * ei['smooth_factor']
            _ds_cell.append(_ds)
            _RUNTIME_COUNTERS['setup_lazy_display_built'] += 1
        return _ds_cell[0]

    display_constraints = [ei['dt_sym'] for ei in edge_info]

    # Accumulators
    subset_contributions = []   # continuous smooth contributions (callable)
    delta_contributions = []    # shot-noise δ spikes (structured dicts)
    # Feature B: δ spikes of codimension > 1 (several equalities)
    delta_contributions_multi = []
    n_shotnoise_skipped = 0
    # Feature B: shot-noise subsets that produced NO record although their
    # weight may be nonzero (several equalities, a coefficient that could
    # not be built, ...); ``api.cumulant`` reports a result with any as
    # incomplete.
    n_delta_dropped = 0
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
        # M10: exact rational elimination, the same substitutions as
        # ``sage_solve`` (``_delta_solve_exact``); ``VALIDATE_DELTA`` also
        # runs the legacy solve and compares.
        _solved = _delta_solve_exact(edge_info, delta_edges,
                                     integration_vars)
        if _validate_delta_on():
            _validate_delta_subset(
                _solved,
                _delta_solve_legacy(edge_info, delta_edges,
                                    integration_vars),
                edge_info, smooth_edges, free_ext_syms,
                f'diagram serial {_diag_serial}, subset '
                f'{bin(branch_bits)}, δ edges {delta_edges}')
        if _solved is None:
            continue
        substitutions, remaining_int_vars, ext_time_equalities = _solved

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
            # M10: an exactly rational linear form is zero iff all its
            # coefficients are; anything else keeps the SR test.
            _eq_zero = _residual_is_zero_exact(eq)
            if _eq_zero is None:
                try:
                    if bool(eq.is_zero()):
                        continue
                except Exception:
                    pass
            elif _eq_zero:
                continue
            # Nontrivial equation → shot-noise
            has_shotnoise = True
            nontrivial_equalities.append(SR(eq))
        # Feature B: a single-equality
        # shot-noise subset that keeps integration variables (a loop
        # correction to a δ weight, e.g. the one-loop mean rate under the
        # spike) used to be skipped.  Its weight is the subset's polytope
        # integral at the δ surface, so it goes through the smooth-subset
        # evaluator below and becomes a ``delta_contributions`` record
        # (``integrated=True``) instead of a ``subset_contributions`` term.
        _shot_pending = None
        if has_shotnoise and remaining_int_vars:
            _shot_pending = _shot_equality(nontrivial_equalities,
                                           free_ext_syms)[0]
            if _shot_pending is not None:
                n_shotnoise_skipped += 1
        if has_shotnoise and _shot_pending is None:
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
            # (Feature B: several independent equalities give a
            # ``delta_contributions_multi`` record instead.)

            if _ss_only_idx & set(smooth_edges):
                # a kernel only the state-space route realizes (S3): this
                # record's coefficient would need its symbolic image
                return _ss_only_failure(branch_bits)
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
                if _shot_equality(nontrivial_equalities,
                                  free_ext_syms)[1] != 'empty':
                    n_delta_dropped += 1
                continue

            # Extract the linear form of the equality in terms of
            # free_ext_syms: a · x + c = 0 (several: ``_shot_equality``).
            # An equality without external times ('empty': a pure numeric
            # residual, nonzero because `is_zero()` filtered above) never
            # fires: no contribution.
            _head, _why = _shot_equality(nontrivial_equalities,
                                         free_ext_syms)
            if _head is None:
                if _why != 'empty':
                    n_delta_dropped += 1
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
                n_delta_dropped += 1
                continue

            try:
                coeff_fc = fast_callable(
                    subset_factor_delta,
                    vars=list(free_ext_syms),
                    domain=CDF,
                )
            except Exception:
                n_delta_dropped += 1
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

            _rec = {'coeff_fc': coeff_fc}
            _rec.update(_head)
            _rec.update({
                'retardation_data': retard_data_delta,
                'delta_edges': list(delta_edges),
                'free_ext_idx': list(free_ext_idx),
            })
            (delta_contributions_multi if 'equalities' in _head
             else delta_contributions).append(_rec)
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
        # is still built (as before audit W7, when a retired Θ(0) mode could
        # integrate it); the flag makes the closure return 0 without
        # integrating.
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
        # Conductance kernels are ordinary smooth edges here (section 3a'):
        # a single-exponential kernel keeps ``edge_mode_sums`` and the
        # diagram eligible; any other kernel made it None.  Only
        # NoiseSourceType kernels (``vertex_leg_time``) stay on the slow
        # path until they get an analogous extraction.
        _analytic_eligible = (
            not vertex_leg_time
            and edge_mode_sums is not None
            and _prefactor_is_numerical
        )

        def _sr_build(_cp=cp, _delta_edges=delta_edges,
                      _smooth_edges=smooth_edges, _subs=substitutions,
                      _fc_vars=fc_vars_sub, _bits=branch_bits,
                      _shot=_shot_pending):
            """The SR integrand of this subset for the residual quadrature:
            ``('ok', fast_callable)``, ``('zero', None)`` (structurally
            zero), ``('skip', reason)`` (an integrated δ subset that cannot
            be built: skipped, as before Feature B) or ``('fail', result
            dict)``.  Bound to this subset (S3: also called lazily, by the
            state-space route's fallback)."""
            if _ss_only_idx & set(_smooth_edges):
                return ('fail', _ss_only_failure(_bits))
            # NoiseSourceType kernel diagrams, or non-numerical
            # prefactor, or missing edge_mode_sums cache — build the
            # full SR + fast_callable path for the residual scipy
            # integrand.  NOTE: ``.subs(num_params)`` removed
            # 2026-04-21 (audit Fix #A); see commit notes.
            subset_factor = _cp
            for ei_idx in _delta_edges:
                subset_factor = subset_factor * SR(
                    edge_info[ei_idx]['delta_coeff']
                )
            for ei_idx in _smooth_edges:
                subset_factor = (
                    subset_factor * edge_info[ei_idx]['smooth_factor']
                )
            subset_factor = subset_factor.subs(_subs)
            try:
                subset_factor = subset_factor.expand()
            except Exception:
                pass

            # Structural zero-check (avoids Maxima simplify_full,
            # which can hang or blow up for complex Hawkes integrands).
            try:
                if subset_factor.is_trivial_zero():
                    return ('zero', None)
            except AttributeError:
                if str(subset_factor) == '0':
                    return ('zero', None)

            # Free-symbol audit — catches num_params pass-through bugs.
            try:
                subset_free_vars = set(subset_factor.variables())
            except AttributeError:
                subset_free_vars = set()
            unexpected = subset_free_vars - set(_fc_vars)
            if unexpected and _shot is not None:
                # an integrated δ subset never fails the diagram: before
                # Feature B it was skipped, so it is skipped (and said so)
                return ('skip', f'unexpected free symbols {unexpected}')
            if unexpected:
                return ('fail', {
                    'status': 'failed',
                    'contribution': None,
                    'integration_vars': integration_vars,
                    'stripped_integrand': _display_stripped_value(),
                    'constraints': display_constraints,
                    'reason': (
                        f"[subset {bin(_bits)}] stripped integrand "
                        f"contains unexpected free symbols {unexpected}; "
                        f"pass them via num_params."
                    ),
                })

            try:
                return ('ok', fast_callable(
                    subset_factor, vars=_fc_vars, domain=CDF,
                ))
            except Exception as exc:
                if _shot is not None:
                    return ('skip', f'fast_callable failed: {exc}')
                return ('fail', {
                    'status': 'failed',
                    'contribution': None,
                    'integration_vars': integration_vars,
                    'stripped_integrand': _display_stripped_value(),
                    'constraints': display_constraints,
                    'reason': (
                        f"[subset {bin(_bits)}] fast_callable "
                        f"failed: {exc}"
                    ),
                })

        # S3: a subset the state-space route may take defers the SR build
        # (needed by the pole routes' quadrature only): built after the
        # dispatch when the route declines it, or by its fallback.
        _sr_deferred = False
        if _analytic_eligible:
            subset_factor = None
            integrand_fc_sub = None
        elif (not _engine_poles and not vertex_leg_time
              and edge_mode_sums_builder is None
              and _prefactor_is_numerical):
            integrand_fc_sub = None
            _sr_deferred = True
        else:
            _sr_kind, _sr_val = _sr_build()
            if _sr_kind == 'zero':
                continue
            if _sr_kind == 'skip':
                _shot_skip(subset_diagnostics, delta_edges, smooth_edges,
                           _sr_val)
                n_delta_dropped += 1
                continue
            if _sr_kind == 'fail':
                return _sr_val
            integrand_fc_sub = _sr_val

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
        # ``ROW_KINDS``): the smooth-edge rows so far (a conductance kernel
        # edge is 'conv_pseudo'), then the τ box rows appended below.
        subset_row_kinds = [
            'conv_pseudo' if ei_idx >= n_prop_edges else 'edge'
            for ei_idx in smooth_edges[:len(subset_constraint_data)]]

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
            for (tau_s, _v) in extra_tau_syms:
                if tau_s not in remaining_int_vars:
                    continue
                idx = remaining_int_vars.index(tau_s)
                # Upper cap:  -τ_v + CAP > 0  ⇒  τ_v < CAP
                cap = noise_tau_box.get(tau_s, _TAU_KERNEL_CAP_LEGACY)
                a_up  = [0.0] * n_iv
                a_up[idx] = -1.0
                subset_constraint_data.append(
                    (a_up, [0.0] * n_ext, cap)
                )
                # Lower cap:  +τ_v + CAP > 0 (symmetric around 0).
                a_lo  = [0.0] * n_iv
                a_lo[idx] = +1.0
                subset_constraint_data.append(
                    (a_lo, [0.0] * n_ext, cap)
                )
                subset_row_kinds.extend(('noise_box', 'noise_box'))
        if constraint_err is not None and _shot_pending is not None:
            _shot_skip(subset_diagnostics, delta_edges, smooth_edges,
                       f'constraint not linear: {constraint_err}')
            n_delta_dropped += 1
            continue
        if constraint_err is not None:
            return {
                'status': 'failed',
                'contribution': None,
                'integration_vars': integration_vars,
                'stripped_integrand': _display_stripped_value(),
                'constraints': display_constraints,
                'reason': (
                    f"[subset {bin(branch_bits)}] constraint not "
                    f"linear: {constraint_err}"
                ),
            }

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
        def _mk_fast_eval(_cdata=subset_constraint_data, _m=m_sub,
                          _smooth=smooth_edges, _pn=prefactor_num,
                          _ripi=smooth_edges_ri_pi):
            """The pole routes' per-edge evaluator of this subset (or
            ``None``); bound to it (S3: built lazily for a subset the
            state-space route takes, by its fallback)."""
            if vertex_leg_time or (kernel_edge_specs
                                   and edge_mode_sums is None):
                # NoiseSourceType vertex with non-rational (Gaussian, etc.)
                # kernel — the fast pole/residue evaluator assumes
                # P · Π_e Σ_k C_e^{(k)} exp(i p_k Δt_e), which the
                # cumulant κ-factor breaks.  Fall through to the generic
                # fast_callable path.  Likewise a conductance kernel edge
                # without a mode (it has no propagator entry to read).
                return None
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
                        edge_mode_sums[_ei_idx] for _ei_idx in _smooth
                    ]
                    return _build_fast_subset_evaluator_from_modes(
                        _pn,
                        smooth_edge_modes,
                        _cdata,
                        _m,
                    )
                else:
                    return _build_fast_subset_evaluator(
                        propagator_data,
                        _pn,
                        _ripi,
                        _cdata,
                        _m,
                    )

        # S3: a subset the state-space route may take builds the pole
        # routes' evaluator only if it is needed (the route declined it, or
        # its fallback); with ``poles`` it is built here, as before
        _fast_eval = None if not _engine_poles else _mk_fast_eval()
        integrand_for_quad = (
            _fast_eval if _fast_eval is not None else integrand_fc_sub
        )
        if not _engine_poles:
            # the hook's reference closure, built on its first call
            integrand_for_quad = _LazyIntegrand(_mk_fast_eval, _sr_build,
                                                integrand_fc_sub)

        # Capture the smooth-edge EdgeModeSum subset + complex
        # prefactor for the analytic mode-sum paths:
        #   m = 1  → ``_integrate_1d_polytope_modesum``     (Stage 4a-perdiag)
        #   m = 2  → ``_integrate_2d_polygon_modesum``      (Stage 3a-full)
        #   m ≥ 3  → ``_integrate_nd_polytope_poset_modesum`` (Stage 3b)
        # ``None`` means the closure-only fallback path applies for
        # both (the constraints either can't be extracted or have
        # a non-numerical prefactor still).
        # The state-space subset integrator (S2; the default since S3,
        # ``DAEDALUS_PHASE_J_ENGINE=poles`` switches it off) takes an
        # eligible subset (every row an edge with a realization: a
        # propagator entry, a conductance kernel, a dressed line); no
        # pole-tuple plan is built for it unless its call-time fallback
        # needs one (``_pole_route_builder``).  With ``poles``, nothing
        # below changes.
        _ss_make = None
        if not _engine_poles:
            from engine.integration.time_domain import statespace as _ss
            _ss_specs = []
            for _i in smooth_edges:
                if _i >= n_prop_edges:
                    _ss_specs.append(
                        ('kernel', kernel_edge_specs[_i - n_prop_edges][1]))
                elif _i in dressed_idx:
                    _ei = edge_info[_i]
                    _att = conv_dress_att.get((_ei['u'], _ei['v'], _ei['lbl']))
                    _ss_specs.append(
                        ('dress', _ei['pi'], _ei['ri'], _att) if _att is not None
                        else ('unsupported', 'dressed_line_without_kernel'))
                else:
                    _ss_specs.append(None)
            _ss_make = _ss.subset_maker(
                propagator_data=propagator_data, num_params=num_params,
                pi_ri=[(edge_info[_i]['pi'], edge_info[_i]['ri'])
                       for _i in smooth_edges],
                plain_edges=(not vertex_leg_time
                             and edge_mode_sums_builder is None),
                prefactor=_prefactor_c if _prefactor_is_numerical else None,
                cdata=subset_constraint_data, m=m_sub,
                row_kinds=subset_row_kinds, n_free=len(free_ext_syms),
                modes=(None if edge_mode_sums is None else
                       [edge_mode_sums[_i] for _i in smooth_edges]),
                edge_specs=_ss_specs)
        if _ss_make is None and _ss_only_idx & set(smooth_edges):
            return _ss_only_failure(branch_bits)
        if _ss_make is None and not _engine_poles:
            _fast_eval = _mk_fast_eval()
            integrand_for_quad = (
                _fast_eval if _fast_eval is not None else integrand_fc_sub
            )
        if _ss_make is None and _sr_deferred:
            _sr_kind, _sr_val = _sr_build()
            if _sr_kind == 'zero':
                continue
            if _sr_kind == 'skip':
                _shot_skip(subset_diagnostics, delta_edges, smooth_edges,
                           _sr_val)
                n_delta_dropped += 1
                continue
            if _sr_kind == 'fail':
                return _sr_val
            integrand_fc_sub = _sr_val
            integrand_for_quad = (
                _fast_eval if _fast_eval is not None else integrand_fc_sub
            )
        _modesum_enabled = (
            (USE_1D_INTEGRATOR and m_sub == 1)
            or (USE_POLYGON_M2_INTEGRATOR and m_sub == 2)
            or (USE_POSET_INTEGRATOR and m_sub >= 3)
        )

        def _pole_plan(_cdata=subset_constraint_data, _m=m_sub,
                       _smooth_edges=smooth_edges, _bits=branch_bits,
                       _n_free=len(free_ext_syms),
                       _pref_ok=_prefactor_is_numerical,
                       _pref_c=_prefactor_c, _enabled=_modesum_enabled):
            """``(modes, prefactor, pole_tuples, plan)`` of the analytic
            mode-sum routes for this subset (all ``None`` when they do not
            apply).  Bound to this subset (S3: also called lazily, by the
            state-space route's fallback)."""
            if not (_enabled and not vertex_leg_time
                    and edge_mode_sums is not None and _pref_ok):
                return None, None, None, None
            _modes = [edge_mode_sums[_ei_idx] for _ei_idx in _smooth_edges]
            # Stage 4a-plan (2026-05-15): pre-compute the τ-invariant
            # per-pole-tuple data (alphas, γ_const, γ_slope) for the
            # entire τ grid.  Each analytic integrator threads this
            # plan through its inner loop and skips the per-call
            # edge-loop that would otherwise rebuild α_s / γ on every
            # ``_contrib(free_vals)`` invocation.  Trades a single
            # subset-setup cost for (N_τ − 1) call-time recomputations.
            _plan = _build_modesum_plan(
                _modes,
                _cdata,
                _m,
                _n_free,
                where=(f'diagram #{_diag_serial}, delta-subset '
                       f'{_bits}, m = {_m}, '
                       f'{len(_modes)} smooth edges with '
                       f'{[len(e.modes) for e in _modes]} '
                       f'modes'),
            )
            # Keep the legacy cache populated so any caller still
            # passing ``pole_tuples=`` keeps working unchanged.
            return _modes, _pref_c, _plan['pole_tuples'], _plan

        # ``_prefactor_c`` was already computed early (Stage 4a optim
        # 2026-05-15); reuse it instead of re-converting through SR.
        if _ss_make is None:
            (_smooth_edge_modes, _modesum_prefactor_c, _pole_tuples_cache,
             _modesum_plan) = _pole_plan()
        else:
            _smooth_edge_modes = _modesum_prefactor_c = None
            _pole_tuples_cache = _modesum_plan = None

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
            # it returns 0.
            def _contrib(free_vals, _hook_ctx=None, _tie_ctx=None):
                if theta0_empty:
                    _RUNTIME_COUNTERS['theta0_subsets_pruned'] += 1
                    return 0.0 + 0.0j
                _hook = _SUBSET_HOOK
                _attempted = None
                _bail_reason = None
                _dbm_reason = None
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
                    # M3: the exact DBM route.
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
                                value=dbm_val, bail_reason=_bail_reason,
                                attempted='poset', modes=modes,
                                prefactor=pref_c, plan=plan,
                                pole_tuples=pole_tuples, integrand=fc,
                                row_kinds=row_kinds, tie_ctx=_tie_ctx)
                        return dbm_val
                    # Declined (its reason is in the dbm_declined_*
                    # counters); the poset's reason labels the fallback.
                    _dbm_reason = _pop_bail_reason()
                # Closure-only fallback via scipy.nquad.
                if _is_divergence_verdict(_bail_reason):
                    _divergence_verdict(on_divergence, _bail_reason, m_val,
                                        hook_meta, free_vals)
                elif _dbm_reason == 'dbm_divergent':
                    _divergence_verdict(on_divergence, _dbm_reason, m_val,
                                        hook_meta, free_vals)
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

        _hook_meta = {
            'source': 'per_diagram',
            'diagram_serial': _diag_serial,
            'model': _model_id,
            # the hardened fallback's decay certificate without mode data
            # and its warnings' parameter point (ACC-17)
            'kappa_floor': _kappa_floor,
            'params': _params_fp,
            'diagram': typed_diagram,
            'loop_number': loop_number,
            'subset_id': branch_bits,
            'subset_index': len(subset_contributions),
            'delta_edges': tuple(delta_edges),
            'smooth_edges': tuple(smooth_edges),
            # Feature B: an integrated δ record, not a smooth term
            **({'delta_record': True}
               if _shot_pending is not None else {}),
        }
        if _ss_make is None:
            _subset_cfn = _make_subset_contrib(
                integrand_for_quad, subset_constraint_data, m_sub,
                modes=_smooth_edge_modes,
                pref_c=_modesum_prefactor_c,
                pole_tuples=_pole_tuples_cache,
                plan=_modesum_plan,
                row_kinds=tuple(subset_row_kinds),
                theta0_empty=_theta0_always_empty,
                hook_meta=_hook_meta,
            )
        else:
            def _pole_route_builder(meta, _mk=_make_subset_contrib,
                                    _fc=integrand_fc_sub,
                                    _mk_fast=_mk_fast_eval,
                                    _cdata=subset_constraint_data, _m=m_sub,
                                    _rk=tuple(subset_row_kinds),
                                    _t0=_theta0_always_empty,
                                    _plan_fn=_pole_plan, _sr_fn=_sr_build,
                                    _deferred=_sr_deferred,
                                    _ss_only=bool(_ss_only_idx
                                                  & set(smooth_edges))):
                """S3: the pole-route closure of this subset, built when
                the state-space route falls back on it (its modesum plan,
                and the SR integrand if it was deferred); ``None`` when the
                subset has no pole route."""
                if _ss_only:
                    return None
                fast = _mk_fast()
                fc = fast if fast is not None else _fc
                if _deferred and fast is None:
                    kind, val = _sr_fn()
                    if kind == 'zero':
                        return lambda *_a, **_k: 0.0 + 0.0j
                    if kind != 'ok':
                        raise RuntimeError(
                            f'the state-space route refused this subset at '
                            f'call time and its pole route cannot be built '
                            f'({kind}: '
                            f'{val if isinstance(val, str) else val.get("reason")})')
                    fc = val
                modes_, pref_, tuples_, plan_ = _plan_fn()
                return _mk(fc, _cdata, _m, modes=modes_, pref_c=pref_,
                           pole_tuples=tuples_, plan=plan_, hook_meta=meta,
                           row_kinds=_rk, theta0_empty=_t0)
            _subset_cfn = _ss_make(
                integrand_for_quad, subset_constraint_data, m_sub,
                row_kinds=tuple(subset_row_kinds),
                theta0_empty=_theta0_always_empty,
                hook_meta=_hook_meta,
                fallback=_pole_route_builder,
            )
        if _shot_pending is not None:
            # Feature B: the subset's integral is the δ weight (evaluated
            # on the δ surface by the caller), not a smooth term.
            (delta_contributions_multi if 'equalities' in _shot_pending
             else delta_contributions).append(dict(
                _shot_pending,
                coeff_fc=_SurfaceCoeff(_subset_cfn),
                retardation_data=[],
                delta_edges=list(delta_edges),
                free_ext_idx=list(free_ext_idx),
                integrated=True,
                m_after_delta=m_sub,
                theta0_rows_on_surface=_rows_zero_on_surface(
                    _shot_pending, subset_constraint_data),
            ))
            subset_diagnostics.append({
                'delta_edges': delta_edges,
                'smooth_edges': smooth_edges,
                'status': 'shotnoise_integrated',
                'm_after_delta': m_sub,
                'theta0_always_empty': _theta0_always_empty,
            })
            continue
        subset_contributions.append(_subset_cfn)
        # ``_evaluator_label`` tags the INTENDED analytic path for
        # this subset.  At runtime the closure may still fall back
        # to scipy.nquad if the analytic path returns None (mixed
        # constraint, non-uniform bounds, degenerate β).  The label
        # records the design intent; runtime falls through to
        # 'fast_numpy' silently.
        if _ss_make is not None:
            _evaluator_label = 'statespace'
        elif _smooth_edge_modes is not None and m_sub == 1:
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

    # Feature B: what a δ record needs to be put in the caller's frame
    # (``api.cumulant``): the record is written in the FIRST mapping's free
    # times, exactly like one ``subset_contributions`` term, so the physical
    # spike is Σ_perm (record in that permutation's free times) / _comp.
    for _dc in delta_contributions + delta_contributions_multi:
        _dc['wick_perms'] = [tuple(p) for p in _perms]
        _dc['wick_compensation'] = _comp
        _dc['origin_leaf_idx'] = origin_leaf_idx
        _dc['n_ext'] = _k
        _dc['loop_number'] = loop_number

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

    # C0 output (S4, ``api.me_output``): how ``contribution`` combines its
    # subset closures, so the matrix-exponential form of the same sum can be
    # assembled term by term (read only; nothing above uses it)
    contribution.phase_j_parts = {
        'subsets': subset_contributions, 'perms': list(_perms),
        'compensation': _comp, 'origin_leaf_idx': origin_leaf_idx,
        'free_ext_idx': list(free_ext_idx), 'k': _k}

    # If non-local cumulant kernels were substituted in cp, expose
    # the τ-dependent prefactor (= the substituted, simplified cp,
    # already num_params-substituted) so the display layer can place
    # it inside the integral with the propagator factors.  When no
    # NoiseSourceType vertex is present, this is just cp (==
    # combined_prefactor.subs(num_params)) and the display layer
    # treats it as a τ-independent prefactor outside the integral.
    has_cumulant_kernel = bool(noise_source_specs)
    _ok_items = [
        ('status', 'ok'),
        ('contribution', contribution),
        ('delta_contributions', delta_contributions),
        ('integration_vars', integration_vars),
        ('stripped_integrand',
         None),
        ('constraints', display_constraints),
        ('edge_info', edge_info),
        ('n_subsets_evaluated', len(subset_contributions)),
        ('n_delta_contributions', len(delta_contributions)),
        ('n_shotnoise_skipped', n_shotnoise_skipped),
        ('n_delta_subsets_dropped', n_delta_dropped),
        ('delta_contributions_multi', delta_contributions_multi),
        ('subset_diagnostics', subset_diagnostics),
        ('cumulant_prefactor', cp if has_cumulant_kernel else None),
        ('has_cumulant_kernel', has_cumulant_kernel),
    ]
    _RUNTIME_COUNTERS['setup_lazy_display_deferred'] += 1
    return _LazyDict(_ok_items,
                     {'stripped_integrand': _display_stripped_value})


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

    Constant rows (M1): every zero-normal row is first given its
    ``_const_row_verdict``.  An EMPTY
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
    rows' provenance (``None``: all 'edge').

    Structural zeros (M2a): after the
    constant rows, an m >= 2 region whose pair rows contain a directed
    cycle with shifts summing to <= 0 (``_region_structurally_empty``,
    exact) returns 0j without quadrature (counter ``polytope_empty_cycle``;
    the entry is still counted in ``nquad_calls`` / ``scipy_nquad_called_*``).
    This covers every caller: an analytic bail, a poset extraction that
    bailed before its own cycle test, the SR path of a non-analytic subset,
    and direct callers.

    Hardened quadrature (M2b): every m >= 1 region left after those exact
    tests is integrated by ``_integrate_polytope_hardened``; only a region
    it declines (a non-finite row shift) reaches the default-tolerance
    scipy.nquad routines below.  ``mode_info`` (an ``_NquadModes``; default:
    the ``_nquad_modes`` attribute of the fast evaluators) supplies the
    integrand's modes for its tolerance and truncation; ``diag_meta`` (the
    dispatch's hook metadata) names the diagram in its one-time warning.
    """
    # M0.1 observational counters: m=0 is a direct evaluation, m>=1 is
    # real scipy.nquad quadrature.
    if m == 0:
        _RUNTIME_COUNTERS['polytope_m0_direct'] += 1
    else:
        _RUNTIME_COUNTERS['nquad_calls'] += 1
        if _m7_t_min_is_finite():
            # M7: a finite PHASE_J_T_MIN is honoured by the box-free routes
            # only; the scipy routines' box would silently ignore it.  Known
            # case (M7 report): a region next to PHASE_J_T_MIN that the exact
            # DBM route declines as ill-conditioned (spike reset with T within
            # ~2 of the earliest external time; T = -5 works there).
            raise ValueError(
                f'a finite final_integral.PHASE_J_T_MIN '
                f'(= {PHASE_J_T_MIN!r}) reached the scipy fallback, which '
                f'cannot honour it (an m>=1 subset no box-free route '
                f'answered: {_describe_diagram(diag_meta, m)}, external '
                f'times {tuple(float(x) for x in free_ext_vals)}; declined '
                f'by the exact DBM route as ill-conditioned '
                f'{_RUNTIME_COUNTERS["dbm_declined_ill_conditioned"]} '
                f'time(s) so far).  This happens when PHASE_J_T_MIN is close '
                f'to the external times (a few decay lengths); move it '
                f'further back, or leave it at -inf (the stationary state).')
    if m == 1:
        _RUNTIME_COUNTERS['scipy_nquad_called_m1'] += 1
    elif m == 2:
        _RUNTIME_COUNTERS['scipy_nquad_called_m2'] += 1
    elif m >= 3:
        _RUNTIME_COUNTERS['scipy_nquad_called_mge3'] += 1
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
    if (m >= 2
            and _region_structurally_empty(s_constraints, m) == 'cycle'):
        _RUNTIME_COUNTERS['polytope_empty_cycle'] += 1
        return 0.0 + 0.0j
    if m >= 1:
        val = _integrate_polytope_hardened(
            integrand_callable, s_constraints, free_ext_vals, m,
            mode_info=mode_info, diag_meta=diag_meta)
        if val is not None:
            return val
        # ``None``: a non-finite row shift; the default-tolerance routines
        # below.
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
    # filter always returns 0.  (M1: ``_integrate_polytope`` has already
    # applied ``_const_row_verdict``,
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
# Hardened quadrature fallback (M2b; plan §3.1 L1b)
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
    them.  Issued once per model, source (per-diagram or grouped), loop
    order, set of diagrams and parameter point per process
    (``_nquad_note_region``): the model is its name (``_model_identity``), so
    repeating the same call does not repeat it, while other diagrams, other
    parameter values or another model do.  Within
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
    ``compute_cumulants`` from the model's ``name``; the warnings add the
    diagrams and the parameter point to it, ``_nquad_flush``); for
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
                       floor_cut=False, flag_msgs=()):
    """Record one region served by the hardened fallback (see
    ``nquad_warning_scope``)."""
    key = _nquad_group_key(diag_meta)
    ident = (_diagram_identity(diag_meta) if diag_meta
             else ('direct', m))
    with _NQUAD_WARNED_LOCK:
        st = _NQUAD_PENDING.get(key)
        if st is None:
            st = {'diagrams': set(), 'params': set(), 'regions': 0,
                  'no_modes': 0, 'uncertified': 0, 'floor_cut': 0,
                  'flagged': 0, 'flag_calls': 0,
                  'first': (diag_meta, m), 'first_flag': None}
            _NQUAD_PENDING[key] = st
        st['diagrams'].add(ident)
        st['params'].add((diag_meta or {}).get('params'))
        st['regions'] += 1
        st['no_modes'] += bool(no_modes)
        st['uncertified'] += bool(uncertified)
        st['floor_cut'] += bool(floor_cut)
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
            # once per model, source, loop order, set of diagrams and
            # parameter point (another parameter set or other diagrams of
            # the same model warn again)
            which = (frozenset(st['diagrams']), frozenset(st['params']))
            kf = ('fallback',) + key + which
            if kf not in _NQUAD_WARNED:
                _NQUAD_WARNED.add(kf)
                todo.append(('fallback', key, st))
            kq = ('flags',) + key + which
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
                          'integrand, no geometric breakpoints)')
            if st['floor_cut']:
                extra += (f'; {st["floor_cut"]} with open directions cut at '
                          f'{NQUAD_TAIL_K:g}/κ beyond their farthest '
                          'breakpoint, κ the slowest line or kernel rate of '
                          'the model')
            if st['uncertified']:
                extra += (f'; {st["uncertified"]} with an open direction '
                          'without a decay certificate, capped at the '
                          f'absolute ±{NQUAD_UNCERTIFIED_CAP:g}: their '
                          'values are truncated wherever the integrand '
                          'decays more slowly than that')
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
                'source, loop order, set of diagrams and parameter point '
                'in a process; every region is '
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
                'model, source, loop order, set of diagrams and parameter '
                'point in a process; every such '
                "call is counted in final_integral._RUNTIME_COUNTERS"
                "['nquad_hardened_quad_flags'].")
        # shown also under the Sage launcher's PYTHONWARNINGS=ignore
        _ensure_warning_shown(PhaseJNquadFallbackWarning, message=msg,
                              stacklevel=2)
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
    (the design is in the comment block above).  s_0 is
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
    floor_span = None
    if mode_info is not None:
        kmin, kfast = mode_info.decay_rates()
    else:
        ctr['nquad_hardened_no_modes'] += 1
        kf = (diag_meta or {}).get('kappa_floor')
        if kf is not None and kf > 0.0 and math.isfinite(kf):
            a_min = 1.0
            for (_c, prs) in rows_t:
                for (_j, a) in prs:
                    a_min = min(a_min, abs(a))
            floor_span = NQUAD_TAIL_K / (kf * a_min)
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
    # [empty intervals, capped (0/1), uncertified (0/1), cut by the
    #  model's decay floor (0/1),
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
        if floor_span is not None:
            # no certificate from the integrand, but the model's slowest
            # line / kernel rate bounds its decay (ACC-17): cut at
            # K / κ_floor beyond the farthest breakpoint
            far_lo = min((x for x in (L, anchor, ext_pts[0])
                          if x is not None and math.isfinite(x)),
                         default=0.0)
            far_hi = max((x for x in (U, anchor, ext_pts[-1])
                          if x is not None and math.isfinite(x)),
                         default=0.0)
            if count:
                stats[3] = 1
            if lo_open and hi_open:
                return far_lo - floor_span, far_hi + floor_span
            if lo_open:
                return min(far_lo, U) - floor_span, U
            return L, max(far_hi, L) + floor_span
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
        ctr['nquad_hardened_floor_cut'] += stats[3]
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
                       uncertified=bool(stats[2]), floor_cut=bool(stats[3]),
                       flag_msgs=pflags)
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


def _prefactor_exact_zero(cp, num_params):
    """True iff the prefactor with ``num_params`` substituted is a plain
    number exactly 0 (the zero-prefactor exit's test, ``CDF`` value == 0);
    anything symbolic or failing counts as nonzero."""
    try:
        cp = SR(cp)
        if num_params:
            cp = cp.subs(num_params)
        return bool(cp.is_numeric() and complex(CDF(cp)) == 0)
    except Exception:
        return False


def estimate_modesum_plans(typed_diagrams, propagator_data, num_params=None,
                           prune=None, prefactors=None, budget_bytes=None,
                           ss_estimate=None):
    r"""Estimate of the pole tuples the modesum plans of ``typed_diagrams``
    will hold (item 35's memory pre-pass), without integrating anything: an
    upper bound on the per-diagram route for propagator edges (not for
    conductance-kernel placements, dressed lines or the residues of a
    pole-deformed propagator, which the running guard covers).

    Per diagram, each propagator edge ``(pi, ri)`` has ``n_e`` modes: its
    NONZERO residues when the exact-zero modes are pruned (``prune``,
    default ``True``), else all poles; an entry without any
    nonzero residue has no smooth part to keep (``n_e = 0``: the
    structural-zero rule never builds a subset that keeps it smooth).  An edge with a δ part (``|delta_coeff| >= 1e-15``) is smooth in
    some δ-subsets and δ in the others, so the sum over the subsets of the
    product over their smooth edges is ``∏_δ (1 + n_e) · ∏_smooth n_e``
    (dropping edges only removes tuples; conductance-kernel and dressed
    lines are not counted: the running guard ``_plan_guard`` catches what
    this misses).  A diagram whose prefactor (``prefactors``, with
    ``num_params`` substituted) is exactly 0 builds no plan (the
    zero-prefactor exit and the subsets' zero-skip); that test costs a
    substitution per diagram, so it runs only when the count without it is
    over ``budget_bytes`` (or when no budget is given), and
    ``prefactors_tested`` says whether it ran.  Returns a dict: ``tuples``, ``tuples_unpruned``,
    ``bytes`` (x ``PLAN_BYTES_PER_TUPLE``), ``n_diagrams`` and ``worst``
    (the diagram index with its per-edge ``(n_e, n_all, has_delta)``), or
    ``None`` when the propagator has no usable pole data.

    S3: ``ss_estimate(td, has_delta)`` -> ``(stationary, transient)`` bytes
    or ``None`` (``statespace.diagram_estimate``): a diagram the
    state-space route takes is charged that estimate instead of its pole
    tuples when it is the smaller of the two (``ss_diagrams``, ``ss_bytes``:
    the stationary states summed, in ``bytes``; ``ss_transient_bytes``: the
    largest segment working set, reported only: one segment at a time, each
    checked before it is built); ``tuples`` counts the diagrams charged
    their pole tuples."""
    if prune is None:
        prune = True
    try:
        ptd = PropagatorTD(propagator_data, num_params)
        lam = ptd.poles()
    except Exception:
        return None
    if not lam:
        return None
    D_delta = propagator_data.get('D_delta')
    entries = {}

    def counts(pi, ri):
        key = (pi, ri)
        hit = entries.get(key)
        if hit is None:
            modes = ptd.entry_modes(pi, ri)
            n_all = len(modes) if modes is not None else len(lam)
            n_nz = (sum(1 for (C, _l) in modes if not _mode_is_zero(C))
                    if modes is not None else n_all)
            try:
                # the δ coefficient as ``build_G_t_matrix`` takes it
                # (``D_delta`` with num_params substituted), without
                # building G(t)
                if D_delta is not None:
                    e = SR(D_delta[pi, ri])
                    dc = (0j if e.is_zero() else
                          complex(CDF(e.subs(num_params) if num_params
                                      else e)))
                else:
                    dc = complex(ptd.delta_coeff(pi, ri))
                has_delta = abs(dc) >= 1e-15
            except Exception:
                has_delta = True
            hit = ((n_nz if prune else n_all) if n_nz else 0,
                   n_all if n_nz else 0, has_delta)
            entries[key] = hit
        return hit

    per_diagram = []            # (tuples, tuples_unpruned, per-edge, ss)
    delta_rec = []              # may give shot-noise δ records (item 61)
    for td in typed_diagrams:
        per = []
        pidx = td.propagator_indices
        parent = {}

        def find(x):
            while parent.get(x, x) != x:
                x = parent[x]
            return x
        for ek in td.prediagram[0].edges():
            hit = pidx.get(ek)          # the exact key: _lookup's first rule
            ri, pi = (hit if hit is not None
                      else _lookup_prop_indices(td, ek))
            per.append(counts(pi, ri))
            if per[-1][2]:
                parent[find(ek[0])] = find(ek[1])
        # a δ-subset whose δ edges tie two external legs together is a
        # shot-noise record, which keeps its subsets for later evaluation
        roots = [find(v) for v in td.prediagram[2]]
        delta_rec.append(len(set(roots)) < len(roots))
        t = math.prod((1 + n) if dl else n for (n, _a, dl) in per)
        tf = math.prod((1 + a) if dl else a for (_n, a, dl) in per)
        ss = None
        if ss_estimate is not None:
            est_ss = ss_estimate(td, lambda pi_, ri_: counts(pi_, ri_)[2])
            # the cheaper engine: a subset whose stationary states do not
            # fit declines to the pole route when its plan does
            # (``statespace._pole_plan_fits``)
            if est_ss is not None and (
                    est_ss[0] <= t * PLAN_BYTES_PER_TUPLE):
                ss = est_ss
        per_diagram.append((t, tf, per, ss))

    def _bytes(rec):
        t, _tf, _per, ss = rec
        return ss[0] if ss is not None else t * PLAN_BYTES_PER_TUPLE
    tested = False
    if prefactors is not None and (
            budget_bytes is None
            or sum(_bytes(r) for r in per_diagram) > budget_bytes):
        # item 61: the diagrams charged the state-space estimate are tested
        # too (an exactly-zero diagram builds no subset on either route)
        tested = True
        memo = {}
        for i, cp in enumerate(prefactors):
            if _bytes(per_diagram[i]) == 0:
                continue
            try:
                key = SR(cp)
                zero = memo.get(key)
            except TypeError:
                key, zero = None, None
            if zero is None:
                zero = _prefactor_exact_zero(cp, num_params)
                if key is not None:
                    memo[key] = zero
            if zero:
                per_diagram[i] = (0, 0, per_diagram[i][2], None)
    total = total_full = 0
    ss_n = ss_stat = ss_trans = 0
    worst = None
    max_diag = 0
    delta_bytes = 0
    for i, (t, tf, per, ss) in enumerate(per_diagram):
        max_diag = max(max_diag, _bytes(per_diagram[i]))
        if delta_rec[i]:
            delta_bytes += _bytes(per_diagram[i])
        if ss is not None:
            ss_n += 1
            ss_stat += ss[0]
            ss_trans = max(ss_trans, ss[1])
            t = tf = 0
            per = []
        total += t
        total_full += tf
        if worst is None or t > worst['tuples']:
            worst = {'index': i, 'tuples': t, 'tuples_unpruned': tf,
                     'edges': per}
    out = {'tuples': total, 'tuples_unpruned': total_full,
           'bytes': total * PLAN_BYTES_PER_TUPLE,
           'n_diagrams': len(typed_diagrams), 'worst': worst,
           'pruned': bool(prune), 'prefactors_tested': tested,
           # the largest single diagram (pole plans or stationary states):
           # what one diagram at a time holds (item 61, ``discard``), and
           # the diagrams that may keep theirs (shot-noise δ records)
           'max_diagram_bytes': max_diag,
           'delta_record_bytes': delta_bytes}
    if ss_estimate is not None:
        # the transient term (one segment system at a time) is reported, not
        # added: it is not held across subsets, and each segment is checked
        # against the budget before it is built (``_check_transient``)
        out.update(ss_diagrams=ss_n, ss_bytes=ss_stat,
                   ss_transient_bytes=ss_trans,
                   bytes=out['bytes'] + ss_stat)
    return out


def _lookup_prop_indices(typed_diagram, edge_key):
    """
    Look up (resp_row, phys_col) propagator indices for a prediagram
    edge, using the same fallback order as
    `_resolve_edge_propagator_data` in `tests/oracles/symbolic.py`.
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
