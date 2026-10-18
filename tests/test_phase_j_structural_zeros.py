"""M2a (the internal Phase J plan (not distributed) §3.1 L1): structural zeros.

Two kinds of Phase J work that is identically zero are always skipped
(the switch ``final_integral.STRUCTURAL_ZEROS`` /
``DAEDALUS_PHASE_J_STRUCTURAL_ZEROS``, and the umbrella
``DAEDALUS_PHASE_J_LEGACY``, that turned it off were retired in audit W7):

* **Order cycles at the scipy.nquad fallback.**  ``_integrate_polytope``
  returns 0j without quadrature when its rows contain a directed cycle of
  unit-coefficient pair rows (``s_up − s_lo + shift > 0``) whose shifts sum
  to <= 0 in exact arithmetic (``_region_structurally_empty``; counter
  ``polytope_empty_cycle``).  Summing the rows around the cycle gives
  ``0 < Σ shift <= 0``: the region is empty.  This covers every caller that
  reaches the fallback without the analytic paths' own emptiness tests: an
  m=2 guard bail, a poset extraction that bailed before its cycle test, a
  rows/modes mismatch, the SR path of a non-analytic subset
  and direct callers.  Before M2a the fallback handed such a region to
  scipy.nquad, which integrated the identically-zero filtered integrand, or
  raised OverflowError evaluating the unfiltered one outside the region
  (plan Appendix C.2).
* **Forced-δ edges.**  An edge whose smooth propagator part is provably
  zero (every stored pole residue of its entry is an exact zero, and its
  ``G_ft`` entry -- the propagator before the builder drops non-retarded
  poles -- contains no ω once the numeric parameters are substituted;
  without ``G_ft``, which the builder skips for a propagator with nf >= 6
  or more than 20 free symbols, the stored residues decide, and only an
  exact zero counts: a tiny nonzero residue is never pruned) contributes
  nothing when kept smooth, so the δ-subsets that keep one smooth are never
  built (``_forced_delta_edges``; counter ``forced_delta_pruned``).  An
  entry whose residues are exact zeros only because its own mode was
  dropped (its ``G_ft`` entry still has ω) is kept.
* **Propagators without any pole.**  An empty pole list is not evidence of
  a zero smooth part: the builder drops non-retarded modes (λ = 0 at q = 0
  for a massless field).  Such a propagator is decided from ``G_ft``: an
  entry that contains no ω once the numeric parameters are substituted is a
  pure δ (forced-δ, e.g. the linear delta spike model with every coupling
  0); a diagram that uses any other entry raises
  ``PoleFreePropagatorError`` (the per-diagram path used to fail with a
  bare IndexError there).

Both skips remove exact zeros only, so values were bit-identical with the
retired switch on and off (checked here and against the pre-M2a engine of
git revision 8aad964 until audit W7), except where the pre-M2a code raised
(an OverflowError at the fallback, a bare IndexError on an empty pole list)
or returned NaN on a skipped piece (a fixed crash, not a value change), and
a pole-free propagator with a dropped mode now raises the named error.

Everything here is model-free (hand-built constraint rows, single-pole modes
and hand-built typed diagrams with a hand-built propagator) except sections F
and G.  They use the public ``single_population_linear_delta_spikes_test``
model with population 2 receiving no input (``w = [[0, 0.25], [0, 0]]``: its
spike train is then a constant-rate process whose own propagator entry is
pure δ) and with every coupling 0 (a propagator without any pole), the
public ``single_population_spike_reset_test`` with population 2 receiving no
input (``w = [[0.55, 0.65], [0, 0]]``: typings that are zero on different
subsets) and the massless ``edwards_wilkinson_1d`` probe of the model zoo (a
λ = 0 mode at q = 0), each in a private empty cwd.

Run:  sage -python -m pytest tests/test_phase_j_structural_zeros.py -q
"""
import contextlib
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                '..')))

import engine.integration.time_domain.final_integral as FI       # noqa: E402

VARIANTS = ('plan', 'noplan')


def _counter(name):
    return FI._RUNTIME_COUNTERS[name]


def _ems(*modes):
    """An ``EdgeModeSum`` with the given ``(C, λ)`` modes."""
    return FI.EdgeModeSum(ri=0, pi=0, delta_coeff=0j, modes=tuple(
        (complex(C), complex(lam)) for C, lam in modes))


@pytest.fixture
def no_quadrature(monkeypatch):
    """Fail if any scipy quadrature routine behind ``_integrate_polytope``
    is reached."""
    def boom(*_a, **_k):
        raise AssertionError('quadrature reached')
    for name in ('_integrate_polytope_hardened', '_integrate_1d_polytope',
                 '_integrate_2d_polytope', '_integrate_nd_polytope'):
        monkeypatch.setattr(FI, name, boom)


def _never_called(*_a, **_k):
    raise AssertionError('integrand evaluated on an empty region')


def _resolved(rows, free):
    return [(list(a), c0 + sum(x * t for x, t in zip(e, free)))
            for (a, e, c0) in rows]


# ═══════════════════════════════════════════════════════════════════════
# B. the exact cycle test
# ═══════════════════════════════════════════════════════════════════════

def _pair(m, up, lo, shift):
    a = [0.0] * m
    a[up], a[lo] = 1.0, -1.0
    return (tuple(a), shift)       # s_up - s_lo + shift > 0


def test_cycles_with_nonpositive_shift_sum_are_empty():
    E = FI._region_structurally_empty
    assert E([_pair(2, 0, 1, 0.0), _pair(2, 1, 0, 0.0)], 2) == 'cycle'
    assert E([_pair(3, 1, 0, 0.0), _pair(3, 2, 1, 0.0),
              _pair(3, 0, 2, 0.0)], 3) == 'cycle'
    # opposite shifts of an external-time difference cancel exactly
    d = 0.7 - 0.3
    assert E([_pair(2, 0, 1, d), _pair(2, 1, 0, -d)], 2) == 'cycle'
    # a cycle inside a larger system, among scalar and 3-term rows
    rows = [((1.0, 0.0, 0.0, 0.0), 5.0), ((1.0, 1.0, -1.0, 0.0), 0.3),
            _pair(4, 3, 1, 0.0), _pair(4, 2, 3, 0.0), _pair(4, 1, 2, -0.5)]
    assert E(rows, 4) == 'cycle'


def test_cycles_with_positive_shift_sum_are_not_decided():
    E = FI._region_structurally_empty
    assert E([_pair(2, 0, 1, 1e-12), _pair(2, 1, 0, 0.0)], 2) is None
    assert E([_pair(2, 0, 1, 0.5), _pair(2, 1, 0, -0.25)], 2) is None


def test_cycle_sums_are_exact():
    """The shifts are summed as exact rationals: 1 + 1e-16 - 1 is 1e-16 > 0
    (a thin strip, not decided) although float addition in this order gives
    0.0; with -1e-16 the exact sum is negative (empty)."""
    E = FI._region_structurally_empty
    assert (1.0 + 1e-16) - 1.0 == 0.0
    cyc = [_pair(3, 1, 0, 1.0), _pair(3, 2, 1, 1e-16), _pair(3, 0, 2, -1.0)]
    assert E(cyc, 3) is None
    cyc[1] = _pair(3, 2, 1, -1e-16)
    assert E(cyc, 3) == 'cycle'


def test_rows_that_are_not_unit_pair_rows_are_ignored():
    E = FI._region_structurally_empty
    assert E([_pair(3, 1, 0, 0.0), _pair(3, 2, 1, 0.0)], 3) is None   # chain
    two = [((2.0, -2.0), 0.0), ((-2.0, 2.0), 0.0)]                    # scaled
    assert E(two, 2) is None
    assert E([((1.0, -1.0, 1.0), 0.0), ((-1.0, 1.0, 0.0), 0.0)], 3) is None
    assert E([((0.0, 0.0), -1.0), ((0.0, 0.0), -2.0)], 2) is None     # const
    assert E([_pair(2, 0, 1, 0.0), _pair(2, 1, 0, 0.0)], 1) is None   # m < 2
    for bad in (float('nan'), float('inf'), 1j):
        assert E([_pair(2, 0, 1, bad), _pair(2, 1, 0, 0.0)], 2) is None
    # negative-zero coefficients are zeros
    assert E([((1.0, -1.0, -0.0), 0.0), ((-1.0, 1.0, 0.0), 0.0)], 3) \
        == 'cycle'


# ═══════════════════════════════════════════════════════════════════════
# C. _integrate_polytope and the dispatch shapes
# ═══════════════════════════════════════════════════════════════════════

# Rows s1 < 0, s0 < t, s1 > s0, s0 > s1 (the 2-cycle of plan Appendix C.2) at
# t = 0.2.  With fast poles the pre-M2a fallback evaluated the unfiltered
# integrand at s0 = 0 on the unbounded s1 interval and overflowed.
TWO_CYCLE = [((0.0, -1.0), (0.0,), 0.0), ((-1.0, 0.0), (1.0,), 0.0),
             ((-1.0, 1.0), (0.0,), 0.0), ((1.0, -1.0), (0.0,), 0.0)]
FAST = _ems((1.0, -12.0))
SLOW = _ems((1.0, -1.0))
# m = 3: the chain s0 < s1 < s2 < t, the reversed pair s0 > s1 (a 2-cycle)
# and a 3-term row that makes the poset extractor bail before its own
# cycle test.
CYCLE3_MIXED = [((-1.0, 1.0, 0.0), (0.0,), 0.0),
                ((0.0, -1.0, 1.0), (0.0,), 0.0),
                ((0.0, 0.0, -1.0), (1.0,), 0.0),
                ((1.0, -1.0, 0.0), (0.0,), 0.0),
                ((1.0, 1.0, -1.0), (0.0,), 3.0)]


def _fast_eval(modes, rows, m):
    return FI._build_fast_subset_evaluator_from_modes(1.0 + 0j, list(modes),
                                                      rows, m)


def test_cycle_region_is_zero_without_quadrature(no_quadrature):
    for rows, m, free in ((TWO_CYCLE, 2, [0.2]), (CYCLE3_MIXED, 3, [0.0])):
        FI._reset_runtime_counters()
        v = FI._integrate_polytope(_never_called, _resolved(rows, free),
                                   free, m, raw_rows=rows)
        assert v == 0 and isinstance(v, complex)
        # still an entry of the fallback, answered structurally
        assert _counter('nquad_calls') == 1
        assert _counter('polytope_empty_cycle') == 1
    # a direct caller without the raw rows (no external-time context)
    FI._reset_runtime_counters()
    assert FI._integrate_polytope(_never_called, _resolved(TWO_CYCLE, [0.2]),
                                  [0.2], 2) == 0
    assert _counter('polytope_empty_cycle') == 1


def test_positive_shift_cycle_reaches_quadrature(monkeypatch):
    calls = []
    # the pre-M2b quadrature routine, or the hardened one (M2b, default)
    for name in ('_integrate_2d_polytope', '_integrate_polytope_hardened'):
        monkeypatch.setattr(FI, name,
                            lambda *a, **k: calls.append(a) or 0.25 + 0j)
    rows = [((1.0, -1.0), 1e-9), ((-1.0, 1.0), 0.0)]
    FI._reset_runtime_counters()
    assert FI._integrate_polytope(_never_called, rows, [], 2) == 0.25
    assert len(calls) == 1 and _counter('polytope_empty_cycle') == 0


def _analytic(variant, modes, rows, free, m):
    """The analytic modesum integrator in one of the production call
    shapes (per-diagram with / without a plan)."""
    fn = (FI._integrate_2d_polygon_modesum if m == 2
          else FI._integrate_nd_polytope_poset_modesum)
    kw = dict(smooth_edge_modes=list(modes), prefactor_complex=1.0 + 0j,
              subset_constraint_data=rows, free_ext_vals=list(free))
    if m >= 3:
        kw['m'] = m
    if variant == 'plan':
        kw['plan'] = FI._build_modesum_plan(list(modes), rows, m, len(free))
    return fn(**kw)


def _dispatch(variant, modes, rows, free, m):
    """What the per-diagram ``_contrib`` closure does: the analytic
    path first, the scipy fallback on a bail.  (Since M3 an m≥3 poset bail
    tries the exact DBM route in between; it is left out here, as these
    tests pin the bail reasons and the fallback.)"""
    v = _analytic(variant, modes, rows, free, m)
    if v is not None:
        return 'analytic', v, None
    reason = FI._pop_bail_reason()
    return 'nquad', FI._integrate_polytope(
        _fast_eval(modes, rows, m) if len(modes) == len(rows) else
        _never_called, _resolved(rows, free), free, m, raw_rows=rows), reason


@pytest.mark.parametrize('variant', VARIANTS)
def test_m3_extraction_bail_route(variant, no_quadrature):
    FI._reset_runtime_counters()
    path, v, reason = _dispatch(variant, [SLOW] * 5, CYCLE3_MIXED, [0.0], 3)
    assert (path, v, reason) == ('nquad', 0, 'poset_extract_none')
    assert _counter('polytope_empty_cycle') == 1
    assert _counter('poset_empty_cycle') == 0


@pytest.mark.parametrize('variant', VARIANTS)
def test_rows_mismatch_route(variant, no_quadrature):
    """More rows than smooth-edge modes (e.g. conv τ box rows) make every
    analytic path bail before its emptiness tests."""
    rows = TWO_CYCLE[:2] + [((-1.0, 1.0), (0.0,), 0.0),
                            ((1.0, -1.0), (0.0,), 0.0)]
    FI._reset_runtime_counters()
    path, v, reason = _dispatch(variant, [SLOW] * 3, rows, [0.2], 2)
    assert (path, v, reason) == ('nquad', 0, 'polygon_rows_mismatch')
    assert _counter('polytope_empty_cycle') == 1


# ═══════════════════════════════════════════════════════════════════════
# D. forced-δ edges: the exact-zero test
# ═══════════════════════════════════════════════════════════════════════

def test_exact_zero_is_decided_on_the_stored_residue():
    from sage.all import SR, CDF, var
    Z = FI._is_exact_zero
    nan, inf = float('nan'), float('inf')
    for z in (0, 0.0, -0.0, 0j, complex(-0.0, -0.0), CDF(0), CDF(-0.0),
              CDF(0.0, -0.0), SR(0), SR(0.0), np.float64(0.0),
              np.complex128(0.0)):
        assert Z(z), z
    for nz in (1e-300, CDF(1e-300), SR(1e-300), complex(0, 1e-300),
               var('zz'), nan, SR(2) - SR(1),
               # a NaN or inf residue (e.g. an overflow upstream) is never an
               # exact zero; Sage's CDF compares NaN in either part EQUAL
               # to 0, so these are the regression cases for that
               CDF(nan), CDF(0, nan), CDF(nan, 0), CDF(nan, nan), CDF(inf),
               CDF(0, inf), complex(nan, 0.0), complex(0.0, nan), inf,
               np.complex128(complex(nan, 0.0))):
        assert not Z(nz), nz


def _prop(symbolic=False, rate=1.0):
    """A hand-built 2-field propagator with one pole (λ = −rate):
    entry (pi=0, ri=0) pure δ (weight 1, no residue), (1, 0) smooth only,
    (1, 1) δ 0.3 plus a smooth part, (0, 1) identically zero.
    ``symbolic``: the residues are symbols bound by ``num_params``, so no
    numeric mode sum can be built (every subset takes the SR path)."""
    from sage.all import SR, I, matrix
    if symbolic:
        c10, c11 = SR.var('zc10'), SR.var('zc11')
        num = {c10: SR(0.5), c11: SR(0.7)}
    else:
        c10, c11, num = SR(0.5), SR(0.7), None
    C = matrix(SR, [[0, 0], [c10, c11]])
    D = matrix(SR, [[1, 0], [0, SR(3) / 10]])
    return {'pole_vals': [SR(rate) * I], 'C_mats': [C], 'D_delta': D,
            'nf': 2}, num


def _pole_free_prop(pure, with_g_ft=True):
    """A hand-built 2-field propagator WITHOUT any pole (``pole_vals`` and
    ``C_mats`` empty), with its frequency-domain matrix ``G_ft`` in a
    symbolic ``omega``.  ``pure``: every entry is ω-free (a pure-δ
    propagator: δ weights 1 and 3/10 on the diagonal, the off-diagonal
    entries 0).  Otherwise entry (pi=1, ri=0) is ``1/(-i ω)``: a marginal
    λ = 0 mode, which a builder that keeps only strictly retarded poles
    drops from the pole list although the entry has a smooth part."""
    from sage.all import SR, I, matrix
    om = SR.var('omega')
    G = matrix(SR, [[1, 0], [0, SR(3) / 10]])
    if not pure:
        G[1, 0] = 1 / (-I * om)
    pd = {'pole_vals': [], 'C_mats': [],
          'D_delta': matrix(SR, [[1, 0], [0, SR(3) / 10]]), 'nf': 2}
    if with_g_ft:
        pd.update(G_ft=G, omega=om)
    return pd


def _prop_with_g_ft(dropped_mode=False, coupling=None):
    """``_prop()`` (one pole at λ = −1) together with a ``G_ft`` in a
    symbolic ``omega``: entry (pi=0, ri=0) is 1 (pure δ), (0, 1) is 0,
    (1, 0) and (1, 1) have the kept pole.  ``dropped_mode``: entry (0, 0)
    is ``1 + 1/(-i ω)`` instead -- a λ = 0 mode that a builder keeping
    only strictly retarded poles drops, so its stored residue at the kept
    pole is still an exact 0 although its smooth part is not zero.
    ``coupling``: entry (0, 0) is ``1 + zw/(1 - i ω)`` with ``num_params``
    binding ``zw`` to this value (the stored residue stays 0, as a builder
    with ``zw = 0`` would store it)."""
    from sage.all import SR, I, matrix
    pd, num = _prop()
    om = SR.var('omega')
    G = matrix(SR, [[1, 0], [SR(1) / (1 - I * om),
                             SR(3) / 10 + SR(1) / (1 - I * om)]])
    if dropped_mode:
        G[0, 0] = 1 + 1 / (-I * om)
    if coupling is not None:
        zw = SR.var('zw')
        G[0, 0] = 1 + zw / (1 - I * om)
        num = {zw: SR(coupling)}
    pd.update(G_ft=G, omega=om)
    return pd, num


ENTRY = {'D': (0, 0), 'S': (0, 1), 'B': (1, 1), '0': (1, 0)}   # (ri, pi)


def test_forced_delta_edges():
    pd, _ = _prop()
    info = [{'pi': ENTRY[k][1], 'ri': ENTRY[k][0]} for k in 'DSB0D']
    assert FI._forced_delta_edges(info, pd) == frozenset({0, 3, 4})
    cache = {}
    assert FI._forced_delta_edges(info, pd, cache) == frozenset({0, 3, 4})
    assert cache == {(0, 0): True, (1, 0): False, (1, 1): False,
                     (0, 1): True}
    # no poles at all: the residue test decides nothing (an empty pole list
    # is not evidence of a zero smooth part: the propagator builder drops
    # non-retarded modes, e.g. λ = 0 at q = 0 for a massless field).  The
    # entries are decided from G_ft instead (section E2).
    assert FI._smooth_part_is_exact_zero([], 0, 0, 0) is False
    with pytest.raises(FI.PoleFreePropagatorError):     # no G_ft to decide
        FI._forced_delta_edges(info, dict(pd, pole_vals=[], C_mats=[]))
    assert FI._forced_delta_edges(info, _pole_free_prop(pure=True)) \
        == frozenset(range(5))
    assert FI._forced_delta_edges(info, {}) == frozenset()
    spd, _ = _prop(symbolic=True)
    assert FI._forced_delta_edges(info, spd) == frozenset({0, 3, 4})


def test_forced_delta_edges_need_an_omega_free_g_ft_entry():
    """With poles AND a ``G_ft``: an exact-zero stored residue is not enough,
    the ``G_ft`` entry (before pole filtering, ``num_params`` substituted)
    must also be ω-free.  A dropped λ = 0 mode leaves exact-zero residues at
    the kept pole on its entry: that edge is NOT forced-δ.  A coupling bound
    to 0 by ``num_params`` removes ω: forced-δ; bound to 0.2: not."""
    info = [{'pi': ENTRY[k][1], 'ri': ENTRY[k][0]} for k in 'DSB0D']
    pd, num = _prop_with_g_ft()
    cache = {}
    assert FI._forced_delta_edges(info, pd, cache, num) == frozenset({0, 3, 4})
    assert cache == {(0, 0): True, (1, 0): False, (1, 1): False,
                     (0, 1): True}
    pd, num = _prop_with_g_ft(dropped_mode=True)
    cache = {}
    assert FI._forced_delta_edges(info, pd, cache, num) == frozenset({3})
    assert cache[(0, 0)] is False and cache[(0, 1)] is True
    assert FI._smooth_part_is_exact_zero(pd['C_mats'], 1, 0, 0) is True
    ok, why = FI._entry_is_provably_instantaneous(pd, num, 0, 0)
    assert not ok and 'omega' in why
    for coupling, expect in ((0, frozenset({0, 3, 4})),
                             (0.2, frozenset({3}))):
        pd, num = _prop_with_g_ft(coupling=coupling)
        assert FI._forced_delta_edges(info, pd, None, num) == expect


# ═══════════════════════════════════════════════════════════════════════
# E. hand-built typed diagrams (per-diagram builds)
# ═══════════════════════════════════════════════════════════════════════

def _td(edges, leaves, internal, prediagram=None):
    """A TypedDiagram on ``edges`` = {(u, v): entry letter}."""
    from sage.all import DiGraph
    from engine.diagrams.type_assignment import TypedDiagram
    if prediagram is None:
        D = DiGraph(multiedges=True)
        D.add_vertices(sorted(set(leaves) | set(internal)))
        for (u, v) in edges:
            D.add_edge(u, v, None)
        prediagram = (D, D.to_undirected(), list(leaves), list(internal))
    keys = {(u, v, None): ENTRY[k] for (u, v), k in edges.items()}
    return TypedDiagram(prediagram=prediagram, vertex_assignments={},
                        edge_types={k: (('x', 1), ('y', 1)) for k in keys},
                        external_legs={lf: ('y', 1) for lf in leaves},
                        propagator_indices=keys)


def _t2():
    from sage.all import SR
    return [SR.var('t_1'), SR.var('t_2')]


# u -> w -> v with the δ-capable shortcut u -> v: its δ subset merges u and v
# and leaves the 2-cycle s_w > s_v > s_w.  Leaves 3 (after v) and 4 (after w).
CYC = {(0, 1): 'S', (1, 2): 'S', (0, 2): 'B', (2, 3): 'S', (1, 4): 'S'}


def _perdiag(edges, rate=1.0, symbolic=False, t=(0.0, 0.7)):
    pd, num = _prop(symbolic, rate)
    FI._reset_runtime_counters()
    r = FI.integrate_diagram(_td(edges, (3, 4), (0, 1, 2)), pd, 1.0, _t2(),
                             num_params=num)
    assert r['status'] == 'ok'
    return r['contribution'](*t), dict(FI._RUNTIME_COUNTERS), r


def test_perdiag_cycle_through_the_fallback(monkeypatch):
    """With the m=2 polygon integrator off, the cycle subset reaches the
    scipy fallback and is answered 0 by the structural test.  (Before M2a
    fast poles overflowed there: OverflowError at rate 12; slow poles gave
    the same value as now.)"""
    monkeypatch.setattr(FI, 'USE_POLYGON_M2_INTEGRATOR', False)
    for rate in (12.0, 1.0):
        v, c, _r = _perdiag(CYC, rate=rate)
        assert np.isfinite(complex(v)) and c['polytope_empty_cycle'] == 1


@pytest.mark.slow
def test_perdiag_sr_path_cycle(monkeypatch):
    """Symbolic residues: no numeric mode sum, every subset takes the SR
    path straight to the fallback (the cycle subset is answered there;
    before M2a it was integrated, to the same value)."""
    v_on, c_on, _r = _perdiag(CYC, symbolic=True)
    assert v_on != 0
    assert c_on['polytope_empty_cycle'] == 1
    assert c_on['nquad_calls'] == 2


# A tree whose leaf-3 edge is pure δ: of its four δ-subsets (δ parts on
# 0 -> 3 and 2 -> 1) the two that keep 0 -> 3 smooth are identically zero and
# never built.  (No two δ edges share an internal vertex: kept from the
# retired grouped builder, whose one-pass substitutions refused that.)
FD_TREE = {(0, 3): 'D', (0, 4): 'S', (1, 0): 'S', (2, 1): 'B'}


def _perdiag_fd(t=(0.0, 0.7)):
    pd, _ = _prop()
    FI._reset_runtime_counters()
    r = FI.integrate_diagram(_td(FD_TREE, (3, 4), (0, 1, 2)), pd, 1.0, _t2())
    return r, r['contribution'](*t), dict(FI._RUNTIME_COUNTERS)


def test_perdiag_forced_delta_prune():
    """(The values equalled those without the prune, bit for bit, while the
    retired switch could turn it off.)"""
    r_on, v_on, c_on = _perdiag_fd()
    assert v_on != 0
    (fd,) = [i for i, ei in enumerate(r_on['edge_info'])
             if (ei['u'], ei['v']) == (0, 3)]
    assert c_on['forced_delta_pruned'] == 2       # 2 of the 2**2 subsets
    pruned = [d for d in r_on['subset_diagnostics']
              if d['status'] == 'forced_delta_pruned']
    assert len(pruned) == 2 and all(fd in d['smooth_edges'] for d in pruned)
    assert r_on['n_subsets_evaluated'] <= 2


def test_a_dropped_mode_with_poles_is_not_pruned(monkeypatch):
    """The pure-δ leaf edge of ``FD_TREE`` is
    forced-δ with a consistent ``G_ft`` (2 subsets pruned) but not when its
    ``G_ft`` entry carries a dropped λ = 0 mode: then nothing is pruned (the
    subsets are evaluated exactly as before M2a)."""
    for dropped, n_pd in ((False, 2), (True, 0)):
        pd, _num = _prop_with_g_ft(dropped_mode=dropped)
        FI._reset_runtime_counters()
        r = FI.integrate_diagram(_td(FD_TREE, (3, 4), (0, 1, 2)), pd,
                                 1.0, _t2())
        v = complex(r['contribution'](0.0, 0.7))
        assert v != 0, v
        assert FI._RUNTIME_COUNTERS['forced_delta_pruned'] == n_pd, dropped


def _prop_cdf(delta_entry_residue):
    """``_prop()`` with its numeric residues stored as CDF numbers (as the
    API's propagator builder stores them) and the pure-δ entry's residue
    replaced by ``delta_entry_residue``."""
    from sage.all import SR, CDF, I, matrix
    C = matrix(CDF, [[delta_entry_residue, 0.0], [0.5, 0.7]])
    D = matrix(SR, [[1, 0], [0, SR(3) / 10]])
    return {'pole_vals': [SR(1.0) * I], 'C_mats': [C], 'D_delta': D, 'nf': 2}


@pytest.mark.parametrize('residue', [
    float('nan'), complex(float('nan'), 0.0), complex(0.0, float('nan'))],
    ids=['nan', 'nan_real', 'nan_imag'])
def test_a_nan_residue_is_never_pruned(monkeypatch, residue):
    """A NaN residue must stay visible: it is not an exact zero, so no
    δ-subset is pruned, and the box-free routes refuse the non-finite
    exponent (``ValueError``, never a silent number; the retired legacy
    routes returned NaN, also visible).  A CDF zero in the same place is
    pruned and gives a value (control)."""
    from sage.all import CDF
    for bad, expect_raise in ((residue, True), (0.0, False)):
        pd = _prop_cdf(bad)
        assert isinstance(pd['C_mats'][0][0, 0], type(CDF(0)))
        FI._reset_runtime_counters()
        r = FI.integrate_diagram(_td(FD_TREE, (3, 4), (0, 1, 2)), pd, 1.0,
                                 _t2())
        pruned = FI._RUNTIME_COUNTERS['forced_delta_pruned']
        if expect_raise:
            assert pruned == 0
            with pytest.raises(ValueError, match='finite'):
                r['contribution'](0.0, 0.7)
        else:
            assert pruned == 2
            assert complex(r['contribution'](0.0, 0.7)) != 0


@pytest.mark.parametrize('residue, symbolic', [
    (1e-300, False), (1e-14, False), (3e-13, False),
    (complex(0.0, 1e-200), False), (1e-300, True), (None, True)],
    ids=['1e-300', '1e-14', '3e-13', '1e-200j', 'SR_1e-300', 'SR_10^-40'])
def test_a_tiny_nonzero_residue_is_never_pruned(monkeypatch, residue,
                                                symbolic):
    """Only an EXACT zero residue counts: a tiny nonzero residue on the
    pure-δ leaf entry of ``FD_TREE`` (no ``G_ft``, so the stored residues
    alone decide) prunes nothing.  (A tolerance in the residue test would
    prune 2 subsets here and move the values, e.g. for 1e-14 at (0, -0.4)
    from 8.4e-16 to 0.)  ``symbolic``: the residues are stored as SR
    numbers (``None``: the exact SR(10)**-40)."""
    from sage.all import SR, matrix
    if symbolic:
        res = SR(10) ** -40 if residue is None else SR(residue)
    for t in ((0.0, 0.7), (0.0, -0.4)):
        if symbolic:
            pd = _prop_cdf(0.0)
            pd['C_mats'] = [matrix(SR, [[res, 0], [SR(0.5), SR(0.7)]])]
        else:
            pd = _prop_cdf(residue)
        assert 'G_ft' not in pd
        FI._reset_runtime_counters()
        r = FI.integrate_diagram(_td(FD_TREE, (3, 4), (0, 1, 2)), pd,
                                 1.0, _t2())
        v = complex(r['contribution'](*t))
        assert FI._RUNTIME_COUNTERS['forced_delta_pruned'] == 0, t
        assert np.isfinite(v), (t, v)


# ═══════════════════════════════════════════════════════════════════════
# E2. propagators without any pole
# ═══════════════════════════════════════════════════════════════════════

# Every edge δ-capable (entries 'D' and 'B'): with a pure-δ propagator only
# the all-δ subset survives, a shot-noise subset (both leaves merged).
ALL_DELTA_TREE = {(0, 3): 'D', (0, 4): 'B', (1, 0): 'D', (2, 1): 'B'}


def _outcome(fn):
    try:
        return ('value', fn())
    except Exception as exc:                    # noqa: BLE001
        return ('raises', type(exc).__name__, exc)


def _pole_free_perdiag(pd, edges=FD_TREE, t=(0.0, 0.7)):
    FI._reset_runtime_counters()
    r = FI.integrate_diagram(_td(edges, (3, 4), (0, 1, 2)), pd, 1.0, _t2())
    return r, complex(r['contribution'](*t))


def test_a_pure_delta_propagator_without_poles_is_evaluated(monkeypatch):
    """No pole, and every G_ft entry ω-free: a pure-δ propagator.  Every
    subset that keeps an edge smooth is identically zero and skipped; the
    all-δ subset survives (a shot-noise term), so the value at distinct
    times is an exact 0.  (Before M2a the per-diagram path failed here with
    a bare IndexError on the empty pole list.)"""
    pd = _pole_free_prop(pure=True)
    for edges, n_pruned in ((FD_TREE, 4), (ALL_DELTA_TREE, 15)):
        r, v = _pole_free_perdiag(pd, edges)
        assert v == 0
        assert FI._RUNTIME_COUNTERS['forced_delta_pruned'] == n_pruned
        if edges is ALL_DELTA_TREE:          # the surviving shot-noise term
            assert len(r['delta_contributions']) == 1


@pytest.mark.parametrize('with_g_ft', [True, False], ids=['dropped_mode',
                                                          'no_g_ft'])
def test_a_pole_free_propagator_with_a_smooth_part_raises(monkeypatch,
                                                          with_g_ft):
    """No pole, but entry (pi=1, ri=0) is 1/(-i ω) (a dropped λ = 0 mode),
    or no G_ft at all to decide the entries from: the smooth part cannot be
    evaluated, so no value is returned -- ``PoleFreePropagatorError``,
    naming the entry and the reason.  (Before this
    rule an empty pole list counted as an identically-zero smooth part and
    the value became 0.)  Diagrams that use only provably pure-δ entries
    are still evaluated (``ALL_DELTA_TREE`` with ``G_ft``).  (Before M2a:
    a bare IndexError.)"""
    pd = _pole_free_prop(pure=False, with_g_ft=with_g_ft)
    for fn in (lambda: _pole_free_perdiag(pd),):
        with pytest.raises(FI.PoleFreePropagatorError) as info:
            fn()
        err = info.value
        assert isinstance(err, ValueError)
        assert (1, 0) in err.entries
        msg = str(err)
        assert 'G[pi=1, ri=0]' in msg and 'retarded' in msg
        assert ('depends on omega' in msg) if with_g_ft \
            else ("no 'G_ft'" in msg)
        assert FI._RUNTIME_COUNTERS['forced_delta_pruned'] == 0
    if with_g_ft:
        _r, v = _pole_free_perdiag(pd, ALL_DELTA_TREE)
        assert v == 0


# ═══════════════════════════════════════════════════════════════════════
# F. public models end to end (private cwd)
# ═══════════════════════════════════════════════════════════════════════

_LD = 'single_population_linear_delta_spikes_test'
_LD_ONE_WAY = {'Em': [0.8, 0.78], 'tau': [10.0, 9.0],
               'w': [[0.0, 0.25], [0.0, 0.0]]}
_SPIKE1_PARAMS = {'Em': [3.5], 'tau': [10.0], 'a': [2.5], 'w': [[0.55]]}


@pytest.fixture(scope='module')
def private_cwd(tmp_path_factory):
    prev = os.getcwd()
    os.chdir(tmp_path_factory.mktemp('phase_j_structural_zeros_cache'))
    try:
        yield
    finally:
        os.chdir(prev)


@pytest.fixture(scope='module')
def ld_model(private_cwd):
    import daedalus as dd
    return dd.load_model(_LD)[0]


def _run(model, *, k, points=None, tau=None, max_ell=1,
         params=None, fields=None):
    from api import compute_cumulants
    FI._reset_runtime_counters()
    res = compute_cumulants(
        model, k=k, max_ell=max_ell,
        external_fields=fields or ([('n', 1), ('n', 2), ('n', 1)][:k]),
        parameters=params or _LD_ONE_WAY,
        tau_grid=np.array(tau if tau is not None else [0.0]),
        use_cache=True, parallel=False, verbose=False)
    if points is None:
        vals = {ell: np.asarray(a, dtype=complex)
                for ell, a in res['C_tau_by_ell'].items()}
    else:
        vals = {ell: np.array([complex(fn(*p)) for p in points])
                for ell, fn in res['total_C_by_ell'].items()}
    return vals, dict(FI._RUNTIME_COUNTERS)


def _bits(x):
    return np.ascontiguousarray(x, dtype=complex).view(np.uint64)


def _assert_bit_identical(a, b):
    assert set(a) == set(b)
    for ell in a:
        assert np.array_equal(_bits(a[ell]), _bits(b[ell])), (ell, a[ell],
                                                             b[ell])


K3_POINTS = [(0.0, 0.4, 1.0), (0.0, 0.7, 0.7), (0.0, -2e-6, -1e-6),
             (0.3, 0.1 + 0.2, 0.7)]


# (Sections F and G checked these values bit for bit against the prune off
# and against the pre-M2a engine until audit W7 retired the switch.)
def test_public_one_way_linear_delta_k2(ld_model):
    tau = [0.0, 2.5, 10.0, -2.5]
    on, c_on = _run(ld_model, k=2, tau=tau)
    again, _c = _run(ld_model, k=2, tau=tau)
    _assert_bit_identical(on, again)
    assert np.any(on[0] != 0)
    assert c_on['forced_delta_pruned'] > 0


def test_public_one_way_linear_delta_k3(ld_model):
    on, c_on = _run(ld_model, k=3, points=K3_POINTS, max_ell=0)
    assert np.any(on[0] != 0)
    # 31 of the 33 k=3 tree subsets keep a pure-δ edge smooth
    assert c_on['forced_delta_pruned'] == 31


# Every coupling 0: two independent constant-rate spike trains.  The
# propagator has no pole at all and every entry is a pure δ (its G_ft has no
# ω once w = 0 is substituted), so every smooth part is provably zero.
_LD_DECOUPLED = {'Em': [0.8, 0.78], 'tau': [10.0, 9.0],
                 'w': [[0.0, 0.0], [0.0, 0.0]]}


@pytest.mark.parametrize('fields', [[('n', 1), ('n', 2)], [('n', 2), ('n', 2)]],
                         ids=['n1n2', 'n2n2'])
def test_public_pure_delta_propagator_without_poles(ld_model, fields):
    """The pole-free propagator of a decoupled spike model is a pure δ: its
    continuous cumulant at τ ≠ 0 (and at the τ = 0 grid point, sampled at
    −1e−6) is exactly 0.  Before M2a the
    per-diagram path raised a bare IndexError on the empty pole list."""
    tau = [0.0, 2.5, -2.5]
    on, c_on = _run(ld_model, k=2, tau=tau,
                    max_ell=0, params=_LD_DECOUPLED, fields=fields)
    assert np.array_equal(on[0], np.zeros(len(tau), dtype=complex))
    assert c_on['forced_delta_pruned'] > 0


def test_public_massless_spatial_mode_raises_the_named_error(tmp_path):
    """The massless Edwards–Wilkinson probe of the model zoo (μ = 0): at the
    q = 0 certification sample its only mode has λ = 0, which the propagator
    builder drops as non-retarded, so the pole list is empty although the
    propagator 1/(−iω) has a smooth part.  Phase J raises
    ``PoleFreePropagatorError`` (before M2a: a bare IndexError)."""
    from api import compute_cumulants
    from tests.tools import phase_j_zoo_baseline as Z
    e = next(x for x in Z.ZOO
             if x['name'] == 'spatial-edwards_wilkinson_1d-massless-l0')
    with _in_dir(tmp_path):
        model, module, _src = Z._load_model(e)
        params = Z.resolve_params(e, model, module)

        def run():
            return compute_cumulants(
                model, k=2, max_ell=e['max_ell'],
                external_fields=[tuple(x) for x in e['ext']],
                parameters=params, tau_grid=np.asarray(e['tau_grid']),
                chi_grid=np.asarray(e['chi_grid']), parallel=False,
                verbose=False, use_cache=True, spatial_parallel=False)
        with pytest.raises(FI.PoleFreePropagatorError) as info:
            run()
        assert 'retarded' in str(info.value) and info.value.entries


_SPIKE = 'single_population_spike_reset_test'
# Population 2 receives no input (row 2 of w is 0): the propagator entries
# from population 1 into population 2 vanish, so some typings of a
# prediagram keep an identically-zero edge smooth in a subset where other
# typings of it do not.
_SPIKE_ONE_WAY = {'Em': [3.5, 3.5], 'tau': [10.0, 9.0], 'a': [2.5, 2.5],
                  'w': [[0.55, 0.65], [0.0, 0.0]]}


@pytest.fixture(scope='module')
def spike_dir(tmp_path_factory):
    """A private cache dir of its own (entered per test, never left as the
    cwd): the one-population spike model of this section shares the repo
    spike model's cache directory name, so the two never share a cwd."""
    return tmp_path_factory.mktemp('phase_j_structural_zeros_spike')


@contextlib.contextmanager
def _in_dir(path):
    prev = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev)


@pytest.mark.slow
@pytest.mark.parametrize('cfg', [
    pytest.param(dict(k=3, max_ell=0, points=K3_POINTS[:2]), id='k3_tree'),
    pytest.param(dict(k=2, max_ell=1, tau=[0.0, 1.0, 3.0, -3.0]),
                 id='k2_ell1')])
def test_public_spike_reset_one_way_unchanged(spike_dir, cfg):
    """A prediagram whose typings are zero on different subsets (ported from
    the retired grouped mixed-group test): each typed diagram prunes its own
    forced-δ subsets (bit-identical to the prune off while the retired
    switch existed), and two warm builds are bit-identical.  (Slow: ~13 s
    for the model build in a fresh cwd.)"""
    import daedalus as dd
    kw = dict(k=cfg['k'], max_ell=cfg['max_ell'],
              points=cfg.get('points'), tau=cfg.get('tau'),
              params=_SPIKE_ONE_WAY)
    with _in_dir(spike_dir):
        model = dd.load_model(_SPIKE)[0]
        # warm-up: a build from a fresh cache and one loaded from it can
        # differ at rounding level (plan §4.4), so compare two warm builds
        _run(model, **kw)
        on, c_on = _run(model, **kw)
        again, _c = _run(model, **kw)
    _assert_bit_identical(on, again)
    assert any(np.any(v != 0) for v in on.values())
    assert c_on['forced_delta_pruned'] > 0


# (G: the A/B against the pre-M2a engine of git revision 8aad964, flag on,
# off and the legacy umbrella exec'd side by side, and the spike1 on/off
# check were retired with the switch in audit W7.)


# ═══════════════════════════════════════════════════════════════════════
# H. the harness's quadrature-level nquad stub
# ═══════════════════════════════════════════════════════════════════════

def test_harness_quadrature_stub_keeps_the_entry_checks():
    from tests.tools.phase_j_subset_diff import NquadStub
    with pytest.raises(ValueError):
        NquadStub('scipy')
    stub = NquadStub('quadrature')
    fe = _fast_eval([SLOW] * 2, [((-1.0, 1.0), (0.0,), 0.0),
                                 ((0.0, -1.0), (1.0,), 0.0)], 2)
    open_rows = _resolved([((-1.0, 1.0), (0.0,), 0.0),
                           ((0.0, -1.0), (1.0,), 0.0)], [0.0])
    with stub.installed():
        FI._reset_runtime_counters()
        # a cycle is answered at the (real) entry: never reaches the stub
        assert FI._integrate_polytope(_never_called,
                                      _resolved(TWO_CYCLE, [0.2]), [0.2],
                                      2) == 0
        # an open region reaches the (stubbed) quadrature
        assert FI._integrate_polytope(fe, open_rows, [0.0], 2) == 0
        assert FI._integrate_polytope(fe, [((1.0,), 0.0)], [0.0], 1) == 0
        assert _counter('polytope_empty_cycle') == 1
        assert _counter('nquad_calls') == 3
    assert dict(stub.calls) == {2: 1, 1: 1}
    assert FI._integrate_2d_polytope.__name__ == '_integrate_2d_polytope'
