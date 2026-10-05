"""M1 (docs/integration_speedup_plan.md §2.4, §3.1 L0): the Θ(0) = 0 (Itô)
convention for constant constraint rows, row provenance, and call-time flags.

After δ-elimination a smooth edge can have Δt ≡ const: a constraint row with
a zero normal, ``(a_int ≡ 0, a_ext, c0)``, i.e. the condition
``c_eff = c0 + a_ext·t > 0``.  Before M1 the m=2 polygon clip kept such a row
at ``c_eff == 0`` (Θ(0) = 1) while every other integrator dropped it
(Θ(0) = 0).  ``final_integral._const_row_verdict`` is now the one rule, and
it is exact (no tie tolerance):

* ``c_eff > 0``  -> 'DROP'  (skip the row's GEOMETRY only; the edge still
  contributes ``exp(λ_e·c_eff)`` to γ);
* ``c_eff < 0``  -> 'EMPTY';
* ``c_eff == 0.0``: Θ(0) = 0 ('EMPTY') for a row that does not depend on the
  external times; a difference of two external legs' times is ordered by the
  legs' raw times and, at an exact tie, by leg index (a larger index counts
  as infinitesimally earlier), so exactly one orientation holds and a tie
  evaluates to the one-sided limit (``_tie_order_sign``).

An m=2 polygon with two exactly opposite rows whose constants sum to <= 0
(or an exactly zero area) and an m >= 3 order cycle whose shifts sum to <= 0
are empty as well; both tests are exact, so a genuine thin region (e.g. a
strip between external times 1e-12 apart) is integrated as before M1.
``THETA0_CONST_ROW_MODE = 'legacy_clip'`` (or ``DAEDALUS_PHASE_J_LEGACY=1``)
restores the pre-M1 behaviour.

Everything here is model-free (hand-built constraint rows and single-pole
modes whose integrals are known in closed form, and two hand-built typed
diagrams for the row provenance of τ box rows) except the last sections,
which use the public one-population spike-reset model (the ``spike1``
configuration of ``tests/test_phase_j_subset_hook.py``), ``ou_quartic``
and ``single_population_linear_delta_spikes_test`` in a private empty cwd.
Each analytic-integrator test runs on the three branches production uses:
``plan`` (per-diagram, pre-built plan), ``noplan`` (per-diagram without a
plan) and ``grouped`` (merged ``pole_tuples`` and no plan -- the call shape
of ``grouped_integral``'s dispatch; for m=1 and m=0 the grouped helpers
themselves).

Run:  sage -python -m pytest tests/test_phase_j_theta0.py -q
"""
import ast
import cmath
import collections
import inspect
import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                '..')))

import engine.integration.time_domain.final_integral as FI       # noqa: E402
import engine.integration.time_domain.grouped_integral as GI     # noqa: E402

VARIANTS = ('plan', 'noplan', 'grouped')
RTOL = 1e-13


def _ems(*modes):
    """An ``EdgeModeSum`` with the given ``(C, λ)`` modes."""
    return FI.EdgeModeSum(ri=0, pi=0, delta_coeff=0j, modes=tuple(
        (complex(C), complex(lam)) for C, lam in modes))


def _analytic(fn, variant, modes, rows, free, m, row_kinds=None, **kw):
    """Call an FI modesum integrator in one of the production call shapes."""
    args = dict(smooth_edge_modes=list(modes), prefactor_complex=1.0 + 0j,
                subset_constraint_data=rows, free_ext_vals=list(free),
                row_kinds=row_kinds, **kw)
    if fn is FI._integrate_nd_polytope_poset_modesum:
        args['m'] = m
    if variant == 'plan':
        args['plan'] = FI._build_modesum_plan(list(modes), rows, m,
                                              len(free))
    elif variant == 'grouped':
        args['pole_tuples'] = list(FI._enumerate_pole_tuples(list(modes)))
    else:
        assert variant == 'noplan'
    return fn(**args)


def _m1(variant, modes, rows, free, row_kinds=None, tie_ctx=None):
    if variant == 'grouped':
        return GI._integrate_grouped_m1_modesum(
            list(FI._enumerate_pole_tuples(list(modes))), rows, list(free),
            row_kinds=row_kinds, tie_ctx=tie_ctx)
    return _analytic(FI._integrate_1d_polytope_modesum, variant, modes, rows,
                     free, 1, row_kinds, tie_ctx=tie_ctx)


def _m2(variant, modes, rows, free, row_kinds=None, tie_ctx=None):
    return _analytic(FI._integrate_2d_polygon_modesum, variant, modes, rows,
                     free, 2, row_kinds, tie_ctx=tie_ctx)


def _m3(variant, modes, rows, free, row_kinds=None, tie_ctx=None):
    return _analytic(FI._integrate_nd_polytope_poset_modesum, variant, modes,
                     rows, free, 3, row_kinds, tie_ctx=tie_ctx)


def _counter(name):
    return FI._RUNTIME_COUNTERS[name]


@pytest.fixture
def legacy(monkeypatch):
    """Switch to the pre-M1 behaviour for one test (call-time flag)."""
    monkeypatch.setattr(FI, 'THETA0_CONST_ROW_MODE', 'legacy_clip')


def _never_called(*_a, **_k):
    raise AssertionError('integrand evaluated on an EMPTY region')


# Base subsets with closed-form values.  One free external time t = 0 (the
# first entry of ``free``); λ = -1 on every edge.
LAM = -1.0 + 0j
BASE1 = [((-1.0,), (1.0,), 0.0)]                          # s < t: ∫ e^{s} = 1
BASE2 = [((-1.0, 1.0), (0.0,), 0.0),                      # s1 - s0 > 0
         ((0.0, -1.0), (1.0,), 0.0)]                      # t - s1 > 0   -> 1
BASE3 = [((-1.0, 1.0, 0.0), (0.0,), 0.0),                 # chain, -> 1
         ((0.0, -1.0, 1.0), (0.0,), 0.0),
         ((0.0, 0.0, -1.0), (1.0,), 0.0)]
BASES = {1: (_m1, BASE1), 2: (_m2, BASE2), 3: (_m3, BASE3)}
EXTRA = (2.0 + 1.0j, -3.0 + 0.5j)       # the extra constant edge's mode

# Tie contexts for the one free value t = t_1 − t_0 of a two-leg evaluation
# at t_0 == t_1 == 0.  IDENTITY: the free value is leg 1, the origin leg 0;
# SWAPPED: the other Wick permutation (free value = leg 0, origin = leg 1).
IDENTITY = FI._TieContext((0.0, 0.0), (1,), 0)
SWAPPED = FI._TieContext((0.0, 0.0), (0,), 1)


def _with_const_row(m, c0, a_ext=(0.0,)):
    """BASE_m plus one constant row (a_int ≡ 0) on an extra edge of mode
    EXTRA; returns (integrator, modes, rows)."""
    fn, base = BASES[m]
    rows = list(base) + [((0.0,) * m, tuple(a_ext), c0)]
    modes = [_ems((1.0, LAM))] * len(base) + [_ems(EXTRA)]
    return fn, modes, rows


# ═══════════════════════════════════════════════════════════════════════
# A. the helper
# ═══════════════════════════════════════════════════════════════════════

def test_constants_and_signature_defaults():
    assert FI.THETA_AT_ZERO_CONST_ROW == 0
    assert FI._ROW_COEF_ATOL == 1e-12
    # no area tolerance either (M1 review): polygon emptiness is exact
    assert not hasattr(FI, '_ZERO_AREA_RTOL')
    # the three analytic integrators resolve the box at CALL time and take
    # the row provenance and the tie context as keywords
    for fn in (FI._integrate_1d_polytope_modesum,
               FI._integrate_2d_polygon_modesum,
               FI._integrate_nd_polytope_poset_modesum):
        params = inspect.signature(fn).parameters
        assert params['bbox_cap'].default is None
        assert params['row_kinds'].default is None
        assert params['tie_ctx'].default is None
    # no snapping of external times anywhere (M1 review): the translation
    # to the origin leg is plain subtraction
    assert not hasattr(FI, '_translate_to_origin')


def test_verdicts_are_exact():
    """No tie tolerance: only c_eff == 0.0 is a tie."""
    V = FI._const_row_verdict
    assert V((1.0, 0.0), (), 0.0, ()) is None              # a real half-space
    assert V((0.0, 2e-12), (), 0.0, ()) is None
    assert V((0.0, 0.0), (0.0,), 0.0, (0.0,)) == 'EMPTY'   # Θ(0) = 0
    assert V((0.0, 1e-13), (), 0.3, ()) == 'DROP'          # |a| <= 1e-12
    assert V((), (), -0.3, ()) == 'EMPTY'
    assert V((), (), 1e-300, ()) == 'DROP'
    # rounding-level values are decided by their sign, like any other
    assert V((0.0,), (1.0, -1.0), 1e-17, (10.0, 10.0)) == 'DROP'
    assert V((0.0,), (1.0, -1.0), -1e-17, (10.0, 10.0)) == 'EMPTY'
    t = np.nextafter(10.0, 11.0)
    assert V((0.0,), (1.0, -1.0), 0.0, (t, 10.0)) == 'DROP'
    assert V((0.0,), (1.0, -1.0), 0.0, (10.0, t)) == 'EMPTY'
    assert 0.1 + 0.2 - 0.3 > 0          # 5.55e-17: distinct floats, not a tie
    assert V((0.0,), (1.0, 1.0, -1.0), 0.0, (0.1, 0.2, 0.3)) == 'DROP'
    assert V((0.0,), (1.0, -1.0), 0.0, (10.0 + 1e-9, 10.0)) == 'DROP'
    assert V((0.0,), (1.0,), 0.0, (float('inf'),)) == 'DROP'
    assert V((0.0,), (1.0,), 0.0, (float('-inf'),)) == 'EMPTY'
    # the side-effect-free form agrees and names the basis
    D = FI._const_row_decision
    assert D((0.0,), (1.0,), 0.0, (-1e-17,)) == ('EMPTY', 'sign')
    assert D((0.0,), (0.0,), 0.0, (0.0,)) == ('EMPTY', 'theta0')
    assert D((1.0,), (), 0.0, ()) == (None, None)


def test_tie_order_two_legs():
    """k=2 at t_0 == t_1: exactly one orientation holds, the same in both
    Wick permutations -- leg 1 counts as infinitesimally earlier, i.e. the
    Itô left limit τ = t_1 − t_0 → 0⁻."""
    S = FI._tie_order_sign
    V = FI._const_row_verdict
    # identity permutation: free value = t_1 − t_0
    assert S((1.0,), 0.0, IDENTITY) == -1               # Θ(t_1 − t_0) = 0
    assert S((-1.0,), 0.0, IDENTITY) == +1              # Θ(t_0 − t_1) = 1
    # swapped permutation: free value = t_0 − t_1 (same physical rows)
    assert S((1.0,), 0.0, SWAPPED) == +1                # Θ(t_0 − t_1) = 1
    assert S((-1.0,), 0.0, SWAPPED) == -1               # Θ(t_1 − t_0) = 0
    for ctx in (IDENTITY, SWAPPED):
        verdicts = {V((0.0,), (s,), 0.0, (0.0,), 'edge', ctx)
                    for s in (1.0, -1.0)}
        assert verdicts == {'DROP', 'EMPTY'}
    # no context, a pure constant, a shifted row: Θ(0) = 0
    assert S((1.0,), 0.0, None) == 0
    assert S((0.0,), 0.0, IDENTITY) == 0
    assert S((1.0,), 0.5, IDENTITY) == 0
    assert V((0.0,), (-1.0,), 0.0, (0.0,)) == 'EMPTY'   # no ctx
    assert FI._const_row_decision((0.0,), (-1.0,), 0.0, (0.0,), 'edge',
                                  IDENTITY) == ('DROP', 'tie_order')


def test_tie_order_three_legs_and_raw_times():
    S = FI._tie_order_sign
    # legs 1 and 2 tied (free values t_1 − t_0, t_2 − t_0): leg 2 is earlier
    ctx = FI._TieContext((0.0, 0.7, 0.7), (1, 2), 0)
    assert S((1.0, -1.0), 0.0, ctx) == +1               # t_1 − t_2 > 0
    assert S((-1.0, 1.0), 0.0, ctx) == -1
    # the order comes from the RAW times: a difference that rounds to 0.0
    # only after the translation to the origin leg keeps its true sign.
    # Here t_a = 1e-17 > t_b = 0 but (t_a − 1) == (t_b − 1) == -1.0; leg a
    # has the LARGER index, so the index rule alone would say the opposite.
    t_o, t_a, t_b = 1.0, 1e-17, 0.0
    assert (t_a - t_o) == (t_b - t_o)
    ctx = FI._TieContext((t_o, t_b, t_a), (2, 1), 0)    # legs: o=0, b=1, a=2
    free = [t_a - t_o, t_b - t_o]
    assert FI._const_row_decision((0.0,), (1.0, -1.0), 0.0, free, 'edge',
                                  ctx) == ('DROP', 'tie_order')
    assert FI._const_row_decision((0.0,), (-1.0, 1.0), 0.0, free, 'edge',
                                  ctx) == ('EMPTY', 'tie_order')
    # three-term or non-unit rows are not two-leg differences
    ctx = FI._TieContext((0.0, 0.0, 0.0), (1, 2), 0)
    assert S((1.0, 1.0), 0.0, ctx) == 0                 # t1 + t2 − 2 t0
    assert S((2.0, -2.0), 0.0, ctx) == 0


def test_tie_order_converts_only_the_compared_times():
    """The tie context keeps the times as the caller passed them; only the
    two times of a tie are converted, in ``_tie_order_sign``.  Float-like
    times are compared as floats; times with no float value (symbolic
    ones, whose free values are still numeric) are ordered by their
    difference; times without a real order (complex, NaN) fall back to
    Θ(0) = 0."""
    from sage.all import SR
    S = FI._tie_order_sign
    t = SR.var('t')
    ctx = FI._TieContext((t, t), (1,), 0)               # a symbolic exact tie
    assert S((1.0,), 0.0, ctx) == S((1.0,), 0.0, IDENTITY) == -1
    assert S((-1.0,), 0.0, ctx) == S((-1.0,), 0.0, IDENTITY) == +1
    assert S((1.0,), 0.0, FI._TieContext((t, t + 1), (1,), 0)) == +1
    assert S((1.0,), 0.0, FI._TieContext((0j, 0j), (1,), 0)) == 0
    assert S((1.0,), 0.0, FI._TieContext((0.0, math.nan), (1,), 0)) == 0
    R = FI._raw_time_order
    assert R(np.float64(0.7), 0.7) == 0.0
    assert R(0.7, float(np.nextafter(0.7, 0.0))) == 1.0
    assert R(-1, 0.5) == -1.0
    assert R(t + 2, t) == 2.0 and R(t, t) == 0.0
    assert R(0j, 1j) is None


def test_verdict_counters_and_reset():
    FI._reset_runtime_counters()
    V = FI._const_row_verdict
    V((0.0,), (), 0.0, ())                  # tie, Θ(0) = 0
    V((0.0,), (), -1.0, ())                 # strictly empty
    V((0.0,), (), 1.0, ())                  # drop
    V((1.0,), (), 0.0, ())                  # not constant: not counted
    V((0.0,), (-1.0,), 0.0, (0.0,), 'edge', IDENTITY)   # ordered tie: drop
    assert (_counter('theta0_const_empty'), _counter('theta0_const_drop'),
            _counter('theta0_tie'), _counter('theta0_tie_ordered')) == (
        2, 2, 1, 1)
    FI._reset_runtime_counters()
    for k in ('theta0_const_empty', 'theta0_const_drop', 'theta0_tie',
              'theta0_tie_ordered', 'theta0_subsets_pruned',
              'polygon_zero_area', 'poset_empty_const', 'poset_empty_cycle'):
        assert _counter(k) == 0, k


def test_row_kinds():
    V = FI._const_row_verdict
    assert V((0.0,), (), 0.3, (), 'conv_pseudo') == 'DROP'
    for kind in ('noise_box', 'conv_tau'):
        assert V((1.0,), (), 50.0, (), kind) is None     # the normal case
        with pytest.raises(ValueError, match='constant'):
            V((0.0,), (), 50.0, (), kind)
    with pytest.raises(ValueError, match='unknown'):
        V((0.0,), (), 1.0, (), 'kernel_split')
    # provenance is threaded to the integrators
    fn, modes, rows = _with_const_row(2, 0.3)
    kinds = ('edge', 'edge', 'noise_box')
    with pytest.raises(ValueError, match="constant 'noise_box'"):
        _m2('plan', modes, rows, (0.0,), row_kinds=kinds)


# Row provenance as BUILT by ``integrate_diagram`` (review round 3): no
# public model runs a colored noise source (they are markovianized) or a
# conductance-style convolution vertex cheaply, so two hand-built k = 2
# tree diagrams on one field x with one decaying pole (G(t) = e^{-t}, no δ
# part) carry them: a colored noise source feeding both leaves (its second
# response leg sits at anchor − τ, κ(τ) = e^{-|τ|}), and a convolution
# vertex whose kernel g(τ) = e^{-τ/2}/2 is extracted into a pseudo-edge.
# The quadrature fallback is stubbed: only the recorded rows matter here.

def _hand_built_prop():
    from sage.all import SR, I, matrix
    return {'pole_vals': [SR(1.0) * I], 'C_mats': [matrix(SR, [[1]])],
            'D_delta': matrix(SR, [[0]]), 'nf': 1}


def _colored_noise_td():
    from sage.all import SR, DiGraph, exp
    from engine.core.vertices import NoiseSourceType
    from engine.diagrams.type_assignment import TypedDiagram
    D = DiGraph(multiedges=True, loops=False)
    D.add_edge(2, 0, 'a')
    D.add_edge(2, 1, 'b')
    z = SR.var('z_kappa_X_2_0_0')
    src = NoiseSourceType(
        coefficient=z, response_legs=[('xt', 1), ('xt', 1)], bigrade=(2, 0),
        cumulant_specs=[{
            'symbol': z, 'kernel_fn': lambda i, j, tau: exp(-abs(tau)),
            'legs': (0, 0), 'leg_fields': ('xt', 'xt'),
            'tau_var': SR.var('tau'), 'sign': SR(-1) / 2, 'noise': 'X',
            'order': 2}])
    et = {(2, 0, 'a'): (('xt', 1), ('x', 1)),
          (2, 1, 'b'): (('xt', 1), ('x', 1))}
    td = TypedDiagram((D, D, [0, 1], [2]), {2: src}, et,
                      {0: ('x', 1), 1: ('x', 1)}, {e: (0, 0) for e in et})
    return td, z


def _conv_vertex_td():
    from sage.all import SR, DiGraph, exp
    from engine.core.vertices import ConvVertexType, SourceType
    from engine.diagrams.type_assignment import TypedDiagram
    D = DiGraph(multiedges=True, loops=False)
    D.add_edge(2, 0, 'a')
    D.add_edge(3, 2, 'b')
    D.add_edge(3, 1, 'c')
    g = SR.var('g')
    conv = ConvVertexType(
        coefficient=g, response_legs=[('xt', 1)], physical_legs=[('x', 1)],
        bigrade=(1, 1),
        kernel_attachments=[{'symbol': g, 'leg': ('x', 1), 'leg_index': 0,
                             'kernel_td_fn': lambda tau: exp(-tau / 2) / 2}])
    src = SourceType(coefficient=SR(1), response_legs=[('xt', 1), ('xt', 1)],
                     bigrade=(2, 0))
    et = {(2, 0, 'a'): (('xt', 1), ('x', 1)),
          (3, 2, 'b'): (('xt', 1), ('x', 1)),
          (3, 1, 'c'): (('xt', 1), ('x', 1))}
    td = TypedDiagram((D, D, [0, 1], [2, 3]), {2: conv, 3: src}, et,
                      {0: ('x', 1), 1: ('x', 1)}, {e: (0, 0) for e in et})
    return td, g


@pytest.mark.parametrize('build, expected', [
    (_colored_noise_td, ('edge', 'edge', 'noise_box', 'noise_box')),
    # the conv τ keeps its upper cap (τ < 50); its lower cap τ > 0 is
    # replaced by the kernel pseudo-edge's row Δt = +τ > 0
    (_conv_vertex_td, ('edge', 'edge', 'edge', 'conv_tau', 'conv_pseudo')),
], ids=['noise_source', 'conv_vertex'])
def test_integrate_diagram_labels_every_row(build, expected, monkeypatch):
    """The provenance ``integrate_diagram`` builds for its τ box rows and
    conv pseudo-edges reaches the hook payload, one kind per constraint
    row; a box row made constant there raises (plan §7.6)."""
    from sage.all import SR

    def stub(fc, rows, free, m, **kw):
        assert m >= 1
        return 0j
    monkeypatch.setattr(FI, '_integrate_polytope', stub)
    seen = []
    monkeypatch.setattr(FI, '_SUBSET_HOOK', seen.append)
    td, sym = build()
    res = FI.integrate_diagram(td, _hand_built_prop(), SR(0.3) * sym,
                               [SR.var('t_1'), SR.var('t_2')],
                               external_fields=[('x', 1), ('x', 1)])
    assert res['status'] == 'ok', res.get('reason')
    res['contribution'](0.0, 0.5)
    assert seen and {p['path'] for p in seen} == {'nquad'}
    for p in seen:
        kinds = p['row_kinds']
        assert tuple(kinds) == expected
        assert len(kinds) == len(p['constraints'])
        for (a_int, _e, _c), kind in zip(p['constraints'], kinds):
            if kind in ('noise_box', 'conv_tau', 'conv_pseudo'):
                # a τ row: exactly one nonzero coefficient, on τ (last)
                assert [abs(x) for x in a_int[:-1]] == [0.0] * (
                    len(a_int) - 1) and abs(a_int[-1]) == 1.0
        rows = [list(r) for r in p['constraints']]
        box = kinds.index(expected[-2])
        rows[box][0] = [0.0] * len(rows[box][0])        # δ-forced τ
        with pytest.raises(ValueError, match='constant'):
            FI._any_const_row_empty(rows, p['free_ext_vals'], kinds)
    # the grouped build never sees these rows: its guard refuses colored
    # noise sources and convolution vertices (the per-diagram path runs
    # them), so its 'noise_box' labels are unreachable while it is there
    from engine.integration.time_domain.grouped_integral import (
        integrate_grouped_diagram)
    gres = integrate_grouped_diagram(
        typed_diagrams=[td], combined_prefactors=[SR(0.3) * sym],
        propagator_data=_hand_built_prop(),
        ext_time_vars=[SR.var('t_1'), SR.var('t_2')],
        external_fields=[('x', 1), ('x', 1)])
    assert gres['status'] == 'failed'


# ═══════════════════════════════════════════════════════════════════════
# B. analytic integrators: EMPTY -> 0, DROP keeps γ, ties, legacy
# ═══════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize('m', [1, 2, 3])
@pytest.mark.parametrize('variant', VARIANTS)
def test_base_values(m, variant):
    fn, base = BASES[m]
    v = fn(variant, [_ems((1.0, LAM))] * len(base), base, (0.0,))
    assert abs(v - 1.0) < 1e-12, v


@pytest.mark.parametrize('m', [1, 2, 3])
@pytest.mark.parametrize('variant', VARIANTS)
def test_empty_const_row_gives_exact_zero(m, variant):
    """([0..0], [0], 0): exactly 0, matching ``_integrate_polytope`` (which
    decides without evaluating the integrand)."""
    fn, modes, rows = _with_const_row(m, 0.0)
    FI._reset_runtime_counters()
    v = fn(variant, modes, rows, (0.0,))
    assert v == 0 and isinstance(v, complex), v
    assert FI._pop_bail_reason() is None
    assert _counter('theta0_tie') >= 1
    resolved = [(list(a), c0 + sum(x * t for x, t in zip(e, (0.0,))))
                for (a, e, c0) in rows]
    assert FI._integrate_polytope(_never_called, resolved, [0.0], m,
                                  raw_rows=rows) == 0
    assert FI._integrate_polytope(_never_called, resolved, [0.0], m) == 0


@pytest.mark.parametrize('m', [1, 2, 3])
@pytest.mark.parametrize('variant', VARIANTS)
def test_drop_const_row_keeps_gamma(m, variant):
    """([0..0], [0], +0.3) on an extra edge of mode (C, λ): the value is
    multiplied by exactly C·e^{0.3 λ} -- the row leaves the geometry, not γ."""
    fn, base = BASES[m]
    v0 = fn(variant, [_ems((1.0, LAM))] * len(base), base, (0.0,))
    _fn, modes, rows = _with_const_row(m, 0.3)
    v = fn(variant, modes, rows, (0.0,))
    C, lam = EXTRA
    expect = v0 * C * cmath.exp(0.3 * lam)
    assert abs(v - expect) <= RTOL * abs(expect), (v, expect)


@pytest.mark.parametrize('m', [1, 2, 3])
@pytest.mark.parametrize('variant', VARIANTS)
def test_ordered_external_tie(m, variant):
    """An exact tie t_1 == t_0 on an extra edge Δt = ±(t_1 − t_0): with the
    tie context exactly one orientation holds (Θ(t_0 − t_1) = 1, i.e. the
    left limit in τ = t_1 − t_0), and the holding one keeps its γ = e^{0}
    factor; without the context both are Θ(0) = 0."""
    fn, base = BASES[m]
    v0 = fn(variant, [_ems((1.0, LAM))] * len(base), base, (0.0,))
    C = EXTRA[0]
    for ctx, holds in ((IDENTITY, -1.0), (SWAPPED, 1.0)):
        for s in (1.0, -1.0):
            _fn, modes, rows = _with_const_row(m, 0.0, a_ext=(s,))
            v = fn(variant, modes, rows, (0.0,), tie_ctx=ctx)
            expect = v0 * C if s == holds else 0.0
            assert abs(v - expect) <= RTOL * abs(v0 * C), (ctx, s, v)
            assert fn(variant, modes, rows, (0.0,)) == 0         # no ctx


@pytest.mark.parametrize('variant', VARIANTS)
def test_legacy_clip_polygon_keeps_theta0_equal_one(variant, legacy):
    """Pre-M1 (Θ(0) = 1 in the polygon clip): the EMPTY row of
    ``test_empty_const_row_gives_exact_zero`` keeps the whole region, so the
    value is C·e^{0} times the base value -- the bug M1 fixes."""
    _fn, modes, rows = _with_const_row(2, 0.0)
    v = _m2(variant, modes, rows, (0.0,))
    assert abs(v - EXTRA[0]) < 1e-12, v
    # ... and legacy never consults the tie context
    _fn, modes, rows = _with_const_row(2, 0.0, a_ext=(1.0,))
    v = _m2(variant, modes, rows, (0.0,), tie_ctx=IDENTITY)
    assert abs(v - EXTRA[0]) < 1e-12, v


@pytest.mark.parametrize('variant', ['plan', 'noplan'])
def test_legacy_clip_m1_and_poset(variant, legacy):
    """Pre-M1 elsewhere: m=1 already used Θ(0) = 0; the poset extractor
    returned ``None`` (bail to scipy.nquad) on a violated constant row."""
    fn, modes, rows = _with_const_row(1, 0.0)
    assert fn(variant, modes, rows, (0.0,)) == 0
    fn, modes, rows = _with_const_row(3, 0.0)
    assert fn(variant, modes, rows, (0.0,)) is None
    assert FI._pop_bail_reason() == 'poset_extract_none'


def test_m0_rows():
    """m = 0 (every row constant): ``_integrate_polytope`` and the grouped
    m0 helper -- exact rule, and the tie context orders an external tie."""
    pt = [(3.0 + 0j, (LAM,))]
    free = [0.0]
    for (a_ext, ctx, value) in (((1.0,), IDENTITY, 0.0),
                                ((-1.0,), IDENTITY, 3.0),
                                ((-1.0,), None, 0.0),
                                ((1.0,), SWAPPED, 3.0)):
        rows = [((), a_ext, 0.0)]
        resolved = [([], 0.0)]
        f = (lambda *a: 3.0) if value else _never_called
        assert FI._integrate_polytope(f, resolved, free, 0, raw_rows=rows,
                                      tie_ctx=ctx) == value
        assert GI._evaluate_grouped_m0_modesum(pt, rows, free,
                                               tie_ctx=ctx) == value
    # a rounding-level positive value is not a tie
    rows = [((), (1.0, 1.0, -1.0), 0.0)]
    v = GI._evaluate_grouped_m0_modesum(pt, rows, [0.1, 0.2, 0.3])
    assert abs(v - 3.0 * cmath.exp(LAM * (0.1 + 0.2 - 0.3))) < 1e-15
    assert v != 0                       # DROP, not EMPTY
    drop = [((), (1.0,), 0.0)]
    v = GI._evaluate_grouped_m0_modesum(pt, drop, [0.5])
    assert abs(v - 3.0 * cmath.exp(-0.5)) < 1e-15


def test_m0_legacy(legacy):
    tie = [((), (1.0,), 0.0)]
    assert FI._integrate_polytope(lambda *a: 7.0, [([], 0.0)], [0.0], 0,
                                  raw_rows=tie, tie_ctx=SWAPPED) == 0
    v = GI._evaluate_grouped_m0_modesum([(3.0 + 0j, (LAM,))], tie, [0.0],
                                        tie_ctx=SWAPPED)
    assert v == 0


def test_nquad_fallback_applies_the_helper():
    """``_integrate_polytope`` with m >= 1: an EMPTY constant row returns 0
    before any quadrature; an external tie ordered as holding is removed
    from the rows the quadrature sees (which would read it as Θ(0) = 0)."""
    rows = [((-1.0,), (1.0,), 0.0), ((0.0,), (-1.0,), 0.0)]
    free = [0.0]
    resolved = [(list(a), c0 + sum(x * t for x, t in zip(e, free)))
                for (a, e, c0) in rows]
    n0 = _counter('nquad_calls')
    assert FI._integrate_polytope(_never_called, resolved, free, 1,
                                  raw_rows=rows) == 0
    assert _counter('nquad_calls') == n0 + 1      # an entry, not quadrature
    v = FI._integrate_polytope(lambda s, t: math.exp(s - t), resolved, free,
                               1, raw_rows=rows, tie_ctx=IDENTITY)
    assert abs(v - 1.0) < 1e-8, v                 # ∫_{s<0} e^{s} ds
    assert FI._integrate_polytope(_never_called, resolved, free, 1,
                                  raw_rows=rows, tie_ctx=SWAPPED) == 0


# ═══════════════════════════════════════════════════════════════════════
# C. structural emptiness: zero-area polygon, order cycles
# ═══════════════════════════════════════════════════════════════════════

# Rows s1 < 0, s0 < t, s1 > s0, s0 > s1 at t = 0.2 (in this order): the
# closed-half-plane clip leaves the segment s0 = s1 from (2.8e-17, 0) to
# (-200, -200).  With fast poles (λ = -12) its far-corner fan triangles trip
# the |Re| > 600 guard: pre-M1 the subset went to scipy.nquad, whose
# evaluator overflowed on the unbounded box (plan Appendix C.2).
TWO_CYCLE = [((0.0, -1.0), (0.0,), 0.0), ((-1.0, 0.0), (1.0,), 0.0),
             ((-1.0, 1.0), (0.0,), 0.0), ((1.0, -1.0), (0.0,), 0.0)]
FAST = _ems((1.0, -12.0))


@pytest.mark.parametrize('variant', VARIANTS)
def test_zero_area_two_cycle_polygon_is_empty(variant):
    assert FI._polygon_from_2d_constraints(TWO_CYCLE, [0.2], 200.0) == []
    FI._reset_runtime_counters()
    v = _m2(variant, [FAST] * 4, TWO_CYCLE, (0.2,))
    assert v == 0 and FI._pop_bail_reason() is None
    assert _counter('polygon_zero_area') == 1
    assert _counter('nquad_calls') == 0


def test_zero_area_two_cycle_polygon_legacy(legacy):
    """Pre-M1: a non-empty zero-area segment, and a guard bail (-> nquad)."""
    poly = FI._polygon_from_2d_constraints(TWO_CYCLE, [0.2], 200.0)
    assert len(poly) >= 3
    assert _m2('plan', [FAST] * 4, TWO_CYCLE, (0.2,)) is None
    assert FI._pop_bail_reason() == 'polygon_triangle_guard'


def test_zero_area_check_keeps_thin_genuine_regions():
    """A strip of width 1e-6 (the τ_eval scale) is not degenerate; the
    degeneracy test is exact (a segment is, a sliver of area 1e-30 is not)."""
    rows = [((1.0, 0.0), (0.0,), 1e-6), ((-1.0, 0.0), (0.0,), 0.0)]
    poly = FI._polygon_from_2d_constraints(rows, [0.0], 200.0)
    assert len(poly) == 4 and not FI._polygon_has_zero_area(poly)
    assert FI._polygon_has_zero_area([(0.0, 0.0), (1.0, 1.0), (2.0, 2.0)])
    assert FI._polygon_has_zero_area([(0.0, 0.0), (1.0, 1.0)])
    assert not FI._polygon_has_zero_area([(0.0, 0.0), (1.0, 0.0),
                                          (1.0, 1e-30)])


def test_opposite_rows_empty_is_exact():
    """Two rows with exactly opposite normals are contradictory iff their
    constants sum to <= 0.0 -- no tolerance."""
    D = FI._opposite_rows_empty
    assert D([(1.0, -1.0, 0.0), (-1.0, 1.0, 0.0)])            # 2-cycle
    assert D([(1.0, 0.0, -0.7), (-1.0, 0.0, 0.7 - 1e-16)])    # reversed
    assert not D([(1.0, 0.0, -0.7), (-1.0, 0.0, 0.7 + 1e-12)])
    assert not D([(1.0, 0.0, -1.0), (-2.0, 0.0, 0.5)])        # not opposite
    assert not D([(1.0, -1.0, 0.0), (0.0, -1.0, 0.0)])
    assert not D([])


# A strip of width w = t_b − t_a between two external times and no constant
# row: s0 − t_a > 0, t_b − s0 > 0, s0 − s1 > 0 (s1 unbounded below, cut by
# the box).  Its integral is w·e^{−w} (to e^{−200}).  Before the M1 review
# an area tolerance zeroed it for w below ~1e-12; the exact tests keep the
# pre-M1 value bit-for-bit (legacy_clip, whose clip is the pre-M1 one).
def _strip_rows():
    return [((1.0, 0.0), (-1.0, 0.0), 0.0),
            ((-1.0, 0.0), (0.0, 1.0), 0.0),
            ((1.0, -1.0), (0.0, 0.0), 0.0)]


@pytest.mark.parametrize('variant', VARIANTS)
@pytest.mark.parametrize('w', [1e-9, 3e-12, 1e-12, 3e-13, 1e-13, 1e-15])
def test_thin_strip_without_constant_rows_equals_legacy(variant, w,
                                                        monkeypatch):
    modes = [_ems((1.0, -1.0))] * 3
    free = (0.7, 0.7 + w)
    FI._reset_runtime_counters()
    new = _m2(variant, modes, _strip_rows(), free)
    ctrs = {k: _counter(k) for k in ('zero_normal_rows_seen',
                                     'polygon_zero_area', 'nquad_calls')}
    monkeypatch.setattr(FI, 'THETA0_CONST_ROW_MODE', 'legacy_clip')
    old = _m2(variant, modes, _strip_rows(), free)
    assert new is not None and new == old, (w, new, old)
    assert ctrs == {'zero_normal_rows_seen': 0, 'polygon_zero_area': 0,
                    'nquad_calls': 0}
    wt = free[1] - free[0]
    assert new.real == pytest.approx(wt * math.exp(-wt), rel=0.5)


@pytest.mark.parametrize('variant', VARIANTS)
def test_strip_between_coincident_times_is_empty(variant):
    """w = 0 exactly (and reversed, t_b < t_a): the opposite pair is
    contradictory, so the strip is empty -- exactly 0, no integration."""
    modes = [_ems((1.0, -1.0))] * 3
    for free in ((0.7, 0.7), (0.7, 0.7 - 1e-16)):
        FI._reset_runtime_counters()
        assert _m2(variant, modes, _strip_rows(), free) == 0
        assert _counter('polygon_zero_area') == 1
        assert _counter('zero_normal_rows_seen') == 0


def test_clip_rejects_a_zero_normal():
    square = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
    with pytest.raises(ValueError, match='zero normal'):
        FI._clip_polygon_to_halfplane(square, 0.0, 0.0, 1.0)


CYCLE3 = BASE3 + [((1.0, -1.0, 0.0), (0.0,), 0.0)]      # s0 > s1 and s1 > s0


@pytest.mark.parametrize('variant', VARIANTS)
def test_poset_cycle_and_constant_rows_are_zero_without_fallback(variant):
    """m >= 3: an unshifted order cycle and a constant EMPTY row both
    return 0j from the analytic path (so the dispatch never reaches
    scipy.nquad); the extractor returns the private ``_EMPTY_POSET``."""
    FI._reset_runtime_counters()
    assert _m3(variant, [_ems((1.0, LAM))] * 4, CYCLE3, (0.0,)) == 0
    assert FI._pop_bail_reason() is None
    assert _counter('poset_empty_cycle') == 1
    _fn, modes, rows = _with_const_row(3, 0.0)
    assert _m3(variant, modes, rows, (0.0,)) == 0
    assert FI._pop_bail_reason() is None
    assert _counter('poset_empty_const') == 1
    assert FI._extract_causal_poset(CYCLE3, [0.0], 3) is FI._EMPTY_POSET
    assert FI._extract_causal_poset(rows, [0.0], 3) is FI._EMPTY_POSET
    assert _counter('nquad_calls') == 0


def _shifted_cycle(t1, t2):
    """The chain s0 < s1 < s2 < t plus s0 − s1 + (t1 − t2) > 0 (free values
    (t, t1, t2)): a 2-cycle whose total shift is t1 − t2."""
    pad = (0.0, 0.0)
    rows = [(a, tuple(e) + pad, c) for (a, e, c) in BASE3]
    rows.append(((1.0, -1.0, 0.0), (0.0, 1.0, -1.0), 0.0))
    return rows, (0.0, t1, t2)


@pytest.mark.parametrize('variant', VARIANTS)
def test_poset_cycle_with_nonpositive_shift_is_empty(variant):
    """Exact and rounding-level external-time ties behave the same: a
    cycle whose shifts sum to 0 or to -1.1e-16 (t2 = 0.1 + 0.2 + 0.4, one
    ulp above t1 = 0.7) is empty, with no fallback."""
    for t2 in (0.7, 0.1 + 0.2 + 0.4):
        rows, free = _shifted_cycle(0.7, t2)
        assert FI._extract_causal_poset(rows, list(free), 3) \
            is FI._EMPTY_POSET
        assert _m3(variant, [_ems((1.0, LAM))] * 4, rows, free) == 0
        assert FI._pop_bail_reason() is None


def test_poset_cycle_with_positive_shift_is_not_decided():
    """A cycle with a positive total shift (accepted up to the 1e-9 shift
    window) encloses a thin nonempty strip: not decided here, the path
    bails as before (to the exact DBM route at M3)."""
    for rows, free in (
            _shifted_cycle(0.7, float(np.nextafter(0.7, 0.0))),
            (BASE3 + [((1.0, -1.0, 0.0), (0.0,), 1e-10)], (0.0,))):
        assert _m3('plan', [_ems((1.0, LAM))] * 4, rows, free) is None
        assert FI._pop_bail_reason() == 'poset_no_extension'


def test_order_rows_infeasible():
    f = FI._order_rows_infeasible
    # (lo, up, c): s_up − s_lo + c > 0
    assert f(3, [(0, 1, 0.0), (1, 0, 0.0)])                 # s1 > s0 > s1
    assert f(3, [(0, 1, 0.5), (1, 2, -0.2), (2, 0, -0.3)])  # Σ c = 0
    assert not f(3, [(0, 1, 0.5), (1, 2, -0.2), (2, 0, -0.2)])
    assert not f(3, [(0, 1, 0.0), (1, 2, 0.0)])             # a chain
    assert not f(3, [(0, 1, 1e-16), (1, 0, 0.0)])           # a thin strip
    assert f(3, [(0, 1, -1e-16), (1, 0, 0.0)])


def test_poset_cycle_legacy(legacy):
    assert _m3('plan', [_ems((1.0, LAM))] * 4, CYCLE3, (0.0,)) is None
    assert FI._pop_bail_reason() == 'poset_no_extension'
    assert FI._extract_causal_poset(CYCLE3, [0.0], 3) is not FI._EMPTY_POSET


def test_causal_chambers_unchanged():
    """``spatial/causal_chambers`` reuses ``_CausalPoset`` and
    ``_enumerate_linear_extensions``: pin its chamber enumeration on the
    diagrams of ``tests/test_causal_chambers.py`` (M1 must not change it)."""
    from engine.integration.spatial.causal_chambers import causal_chambers
    assert causal_chambers(2, []) == [(0, 1), (1, 0)]
    assert causal_chambers(3, []) == [
        (0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)]
    assert causal_chambers(3, [(0, 1)]) == [(0, 1, 2), (0, 2, 1), (2, 0, 1)]
    assert causal_chambers(3, [(0, 1), (1, 2)]) == [(0, 1, 2)]
    assert causal_chambers(4, [(0, 1), (0, 2)]) == [
        (0, 1, 2, 3), (0, 1, 3, 2), (0, 2, 1, 3), (0, 2, 3, 1),
        (0, 3, 1, 2), (0, 3, 2, 1), (3, 0, 1, 2), (3, 0, 2, 1)]
    # a cyclic order has no chamber (unchanged: the private _EMPTY_POSET
    # sentinel lives only in the temporal modesum path)
    assert causal_chambers(2, [(0, 1), (1, 0)]) == []
    assert FI._CausalPoset.__dataclass_fields__.keys() == {
        'm', 'edges', 'scalar_lowers', 'scalar_uppers'}


# ═══════════════════════════════════════════════════════════════════════
# D. flags: call time, the umbrella, the default
# ═══════════════════════════════════════════════════════════════════════

def test_flag_initialisation_from_the_environment():
    # (the other Phase J flags of the same dict are checked by their own
    # tests, e.g. STRUCTURAL_ZEROS in tests/test_phase_j_structural_zeros.py)
    def f(env):
        return {'THETA0_CONST_ROW_MODE':
                FI._initial_phase_j_flags(env)['THETA0_CONST_ROW_MODE']}
    assert f({}) == {'THETA0_CONST_ROW_MODE': 'ito'}
    assert f({'DAEDALUS_PHASE_J_THETA0_CONST_ROW': 'legacy_clip'}) == {
        'THETA0_CONST_ROW_MODE': 'legacy_clip'}
    for v in ('1', 'true', 'yes'):
        assert f({'DAEDALUS_PHASE_J_LEGACY': v,
                  'DAEDALUS_PHASE_J_THETA0_CONST_ROW': 'ito'}) == {
            'THETA0_CONST_ROW_MODE': 'legacy_clip'}
    assert f({'DAEDALUS_PHASE_J_LEGACY': '0'})['THETA0_CONST_ROW_MODE'] \
        == 'ito'
    with pytest.raises(ValueError):
        f({'DAEDALUS_PHASE_J_THETA0_CONST_ROW': 'stratonovich'})


def test_default_configuration_is_never_legacy():
    """Guard: the shipped default is the Itô rule.  (If this process was
    started with a Phase J override in the environment, the module values
    are the override's; the environment-free default is checked either
    way.)"""
    assert FI._initial_phase_j_flags({})['THETA0_CONST_ROW_MODE'] == 'ito'
    overridden = any(k in os.environ for k in (
        'DAEDALUS_PHASE_J_LEGACY', 'DAEDALUS_PHASE_J_THETA0_CONST_ROW'))
    if not overridden:
        assert FI.THETA0_CONST_ROW_MODE == 'ito'
        assert not FI._theta0_legacy()


def test_invalid_mode_attribute_raises(monkeypatch):
    monkeypatch.setattr(FI, 'THETA0_CONST_ROW_MODE', 'stratonovich')
    with pytest.raises(ValueError):
        _m2('plan', [_ems((1.0, LAM))] * 2, BASE2, (0.0,))


@pytest.mark.parametrize('variant', VARIANTS)
def test_mode_is_read_at_call_time_by_a_built_plan(variant, monkeypatch):
    """A plan built under one mode, evaluated under the other."""
    _fn, modes, rows = _with_const_row(2, 0.0)
    plan = FI._build_modesum_plan(modes, rows, 2, 1)
    kw = dict(smooth_edge_modes=modes, prefactor_complex=1.0 + 0j,
              subset_constraint_data=rows, free_ext_vals=[0.0])
    if variant == 'plan':
        kw['plan'] = plan
    elif variant == 'grouped':
        kw['pole_tuples'] = list(FI._enumerate_pole_tuples(modes))
    assert FI._integrate_2d_polygon_modesum(**kw) == 0
    monkeypatch.setattr(FI, 'THETA0_CONST_ROW_MODE', 'legacy_clip')
    assert abs(FI._integrate_2d_polygon_modesum(**kw) - EXTRA[0]) < 1e-12


# An unbounded wedge with slow decay (λ = -0.05): ∫_{s0<s1<0} = 400 on the
# infinite domain, so the ±cap box visibly matters.
SLOW = _ems((1.0, -0.05))
# A chain s0 < s1 < s2 with s0 > t and no upper row: the poset integrates up
# to the box (upper fallback), so the value depends on it.
UPPER_OPEN = [((-1.0, 1.0, 0.0), (0.0,), 0.0), ((0.0, -1.0, 1.0), (0.0,), 0.0),
              ((1.0, 0.0, 0.0), (-1.0,), 0.0)]


@pytest.mark.parametrize('variant', VARIANTS)
@pytest.mark.parametrize('mode', ['ito', 'legacy_clip'])
def test_bbox_cap_is_read_at_call_time(variant, mode, monkeypatch):
    """In both modes: the legacy umbrella also reads the cap at call time,
    so it reproduces a pre-M1 run at any cap (plan §3.1: the cap-12 trap
    through the call-time cap)."""
    monkeypatch.setattr(FI, 'THETA0_CONST_ROW_MODE', mode)

    def values():
        return (_m2(variant, [SLOW, SLOW], BASE2, (0.0,)),
                _m3(variant, [_ems((1.0, LAM))] * 3, UPPER_OPEN, (0.0,)))
    v200 = values()
    assert abs(v200[0] - 399.80024030904514) < 1e-9
    monkeypatch.setattr(FI, 'POLYGON_BBOX_CAP', 12.0)
    v12 = values()
    assert abs(v12[0] - 48.760552899823075) < 1e-9
    assert v12[1] != v200[1]


@pytest.mark.parametrize('variant', VARIANTS)
def test_legacy_theta0_value_follows_the_call_time_cap(variant, legacy,
                                                       monkeypatch):
    """The trap mechanism, model-free: under 'legacy_clip' the tied
    constant row keeps the whole (box-bounded) region, so the value moves
    with the cap; the umbrella must see that through the call-time cap."""
    _fn, modes, rows = _with_const_row(2, 0.0)
    modes = [SLOW, SLOW, _ems(EXTRA)]
    v200 = _m2(variant, modes, rows, (0.0,))
    monkeypatch.setattr(FI, 'POLYGON_BBOX_CAP', 12.0)
    v12 = _m2(variant, modes, rows, (0.0,))
    C = EXTRA[0]
    assert abs(v200 - C * 399.80024030904514) < 1e-9 * abs(v200)
    assert abs(v12 - C * 48.760552899823075) < 1e-9 * abs(v12)
    monkeypatch.setattr(FI, 'THETA0_CONST_ROW_MODE', 'ito')
    assert _m2(variant, modes, rows, (0.0,)) == 0


# ═══════════════════════════════════════════════════════════════════════
# E. one rule, one place: no other ad-hoc constant-row checks (§2.4 item 1)
# ═══════════════════════════════════════════════════════════════════════

_ENGINE_DIR = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), 'engine', 'integration', 'time_domain')

# Every comparison of a constant-row value (a name ``c_eff*`` or ``c_f``)
# with 0 in the Phase J integrators, by enclosing function.  Anything not
# listed is a new ad-hoc constant-row check: route it through
# ``final_integral._const_row_verdict`` instead (or, if it is a legacy
# branch or downstream of the helper, add it here with the reason).
_ALLOWED_CONST_ROW_CHECKS = {
    'final_integral.py': {
        # the rule itself
        '_const_row_decision': 2,
        # 'legacy_clip' branches (pre-M1 behaviour, kept verbatim)
        '_polygon_from_2d_constraints': 1,
        '_extract_causal_poset': 1,
        '_integrate_1d_polytope_modesum': 1,
        # the scipy.nquad fallback: downstream of the helper in
        # ``_integrate_polytope`` (which returns 0 for an EMPTY row and
        # removes a tie ordered as holding), so for 'ito' these only ever
        # see rows with c_eff != 0, on which they agree with the rule
        '_integrate_polytope': 1,
        '_make_heaviside_filtered_integrand': 1,
        '_integrate_2d_polytope': 2,
        '_integrate_nd_polytope._make_bound_fn': 1,
        '_outer_bounds': 1,
        '_resolve_1d_bounds': 1,
    },
    'grouped_integral.py': {
        # 'legacy_clip' branches
        '_evaluate_grouped_m0_modesum': 1,
        '_integrate_grouped_m1_modesum': 1,
    },
}


def _const_row_checks(path):
    """{function: count} of ``<c_eff-like name> <op> 0`` comparisons."""
    tree = ast.parse(open(path, encoding='utf-8').read())
    found = collections.Counter()

    def is_value(node):
        if isinstance(node, ast.Call) and getattr(node.func, 'id', '') \
                == 'float' and node.args:
            node = node.args[0]
        return isinstance(node, ast.Name) and (
            node.id.startswith('c_eff') or node.id == 'c_f')

    def is_zero(node):
        return (isinstance(node, ast.Constant)
                and isinstance(node.value, (int, float))
                and node.value == 0)

    def visit(node, fn):
        for child in ast.iter_child_nodes(node):
            f = fn
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                f = child.name if fn is None else f'{fn}.{child.name}'
            if isinstance(child, ast.Compare):
                sides = [child.left] + list(child.comparators)
                if any(is_value(s) for s in sides) and any(
                        is_zero(s) for s in sides):
                    found[f] += 1
            visit(child, f)
    visit(tree, None)
    return dict(found)


@pytest.mark.parametrize('fname', sorted(_ALLOWED_CONST_ROW_CHECKS))
def test_no_other_ad_hoc_constant_row_checks(fname):
    got = _const_row_checks(os.path.join(_ENGINE_DIR, fname))
    assert got == _ALLOWED_CONST_ROW_CHECKS[fname]


# ═══════════════════════════════════════════════════════════════════════
# F. public models end to end (one-population spike reset, ou_quartic)
# ═══════════════════════════════════════════════════════════════════════

_SPIKE1_PARAMS = {'Em': [3.5], 'tau': [10.0], 'a': [2.5], 'w': [[0.55]]}
_E2E = {}


@pytest.fixture(scope='module')
def private_cwd(tmp_path_factory):
    """A private empty cwd for the model caches (fresh-clone state)."""
    prev = os.getcwd()
    os.chdir(tmp_path_factory.mktemp('phase_j_theta0_cache'))
    try:
        yield
    finally:
        os.chdir(prev)


@pytest.fixture(scope='module')
def spike1_model(private_cwd):
    """The model, built once, in the private cwd."""
    from tests.test_phase_j_subset_hook import _spike_reset_one_population
    return _spike_reset_one_population()


def _spike1(model, k, max_ell, grouped, tau_grid=(0.0, 3.0), mode='ito'):
    key = (k, max_ell, grouped, tuple(tau_grid), mode)
    if key not in _E2E:
        from api import compute_cumulants
        saved = FI.THETA0_CONST_ROW_MODE
        saved_nh = FI.NQUAD_HARDENED
        FI.THETA0_CONST_ROW_MODE = mode
        # 'legacy_clip' reaches the scipy quadrature fallback here: the
        # pre-M1 captures were made with the pre-M2b (default-tolerance)
        # fallback, so it is used too (M2b, NQUAD_HARDENED).
        FI.NQUAD_HARDENED = saved_nh if mode == 'ito' else False
        FI._reset_runtime_counters()
        try:
            res = compute_cumulants(
                model, k=k, max_ell=max_ell,
                external_fields=[('n', 1)] * k, parameters=_SPIKE1_PARAMS,
                tau_grid=np.array(tau_grid), use_cache=True, parallel=False,
                verbose=False, use_grouped_phase_j=grouped)
        finally:
            FI.THETA0_CONST_ROW_MODE = saved
            FI.NQUAD_HARDENED = saved_nh
        counters = {k_: (dict(v) if isinstance(v, dict) else v)
                    for k_, v in FI._RUNTIME_COUNTERS.items()}
        _E2E[key] = (res, counters)
    return _E2E[key]


def _arrays(res):
    return {int(ell): np.asarray(v, dtype=complex)
            for ell, v in res['C_tau_by_ell'].items()}


@pytest.mark.parametrize('grouped', [False, True], ids=['perdiag', 'grouped'])
def test_spike1_legacy_reproduces_pre_m1_values(spike1_model, grouped):
    """'legacy_clip' reproduces the values captured before M0.1/M1 (rtol
    1e-13: cross-process ulp jitter, plan §4.4)."""
    from tests.test_phase_j_subset_hook import _BEFORE
    res, _c = _spike1(spike1_model, 2, 1, grouped, mode='legacy_clip')
    got = _arrays(res)
    for ell, vals in _BEFORE[('spike1', grouped)].items():
        np.testing.assert_allclose(got[ell], np.array(vals, dtype=complex),
                                   rtol=1e-13, atol=0)


@pytest.mark.parametrize('grouped', [False, True], ids=['perdiag', 'grouped'])
def test_spike1_ito_moves_one_loop_only_and_never_reaches_nquad(
        spike1_model, grouped):
    """M1 moves the one-loop (Θ(0) rows in the polygon path), not the tree;
    with the Itô rule every subset is answered analytically."""
    res_i, c_i = _spike1(spike1_model, 2, 1, grouped)
    res_l, c_l = _spike1(spike1_model, 2, 1, grouped, mode='legacy_clip')
    ito, leg = _arrays(res_i), _arrays(res_l)
    np.testing.assert_allclose(ito[0], leg[0], rtol=1e-14, atol=0)
    assert np.all(np.abs(ito[1] - leg[1]) > 1e-3 * np.abs(leg[1]))
    assert c_i['nquad_calls'] == 0 < c_l['nquad_calls']
    assert c_i['theta0_const_empty'] > 0
    # δ-subsets with a τ-independent EMPTY row return 0 without
    # integration (Itô only; the flag is read at call time)
    if not grouped:
        assert c_i['theta0_subsets_pruned'] > 0
    else:
        assert c_i['theta0_tie'] > 0
    for key in ('theta0_const_empty', 'theta0_const_drop', 'theta0_tie',
                'theta0_tie_ordered', 'theta0_subsets_pruned',
                'polygon_zero_area', 'poset_empty_const',
                'poset_empty_cycle'):
        assert c_l[key] == 0, key


def test_spike1_ito_perdiag_equals_grouped(spike1_model):
    pd = _arrays(_spike1(spike1_model, 2, 1, False)[0])
    gr = _arrays(_spike1(spike1_model, 2, 1, True)[0])
    for ell in pd:
        np.testing.assert_allclose(pd[ell], gr[ell], rtol=1e-12, atol=1e-17)


@pytest.mark.parametrize('grouped', [False, True], ids=['perdiag', 'grouped'])
def test_spike1_mode_flip_after_the_build(spike1_model, grouped,
                                          monkeypatch):
    """The mode is read at CALL time, also for the τ-independent EMPTY
    subsets: a callable built under one mode and evaluated under the other
    equals the callable built under the other mode."""
    fn_i = _spike1(spike1_model, 2, 1, grouped)[0]['total_C_by_ell'][1]
    fn_l = _spike1(spike1_model, 2, 1, grouped,
                   mode='legacy_clip')[0]['total_C_by_ell'][1]
    pts = [(0.0, -1e-6), (0.0, 3.0)]
    out = {}
    for mode in ('ito', 'legacy_clip'):
        monkeypatch.setattr(FI, 'THETA0_CONST_ROW_MODE', mode)
        out[mode] = (np.array([complex(fn_i(*p)) for p in pts]),
                     np.array([complex(fn_l(*p)) for p in pts]))
    for mode, (built_ito, built_legacy) in out.items():
        np.testing.assert_allclose(built_ito, built_legacy, rtol=1e-13,
                                   atol=0, err_msg=mode)
    assert np.all(np.abs(out['ito'][0] - out['legacy_clip'][0])
                  > 1e-3 * np.abs(out['legacy_clip'][0]))


def test_spike1_hook_records_row_kinds(spike1_model):
    """The hook payload carries the row provenance and the tie context."""
    from tests.tools import phase_j_subset_diff as PSD
    _spike1(spike1_model, 2, 1, False)                   # warm cache
    run = PSD.run(spike1_model, k=2, max_ell=1,
                  external_fields=[('n', 1), ('n', 1)],
                  parameters=_SPIKE1_PARAMS, tau_grid=[3.0], keep_live=False)
    recs = run['records']
    assert recs
    for r in recs:
        assert isinstance(r['row_kinds'], tuple)
        assert len(r['row_kinds']) == len(r['constraints'])
        assert set(r['row_kinds']) <= set(FI.ROW_KINDS)
        times, free_legs, origin = r['tie_ctx']
        assert len(free_legs) == len(r['free_ext_vals'])
        assert origin not in free_legs
        # τ-independent EMPTY subsets return 0 before the hook
        for (a, e, c0) in r['constraints']:
            assert not (all(x == 0.0 for x in a) and all(x == 0.0 for x in e)
                        and c0 <= 0.0), r['constraints']
    assert run['counters']['nquad_calls'] == 0


@pytest.mark.parametrize('mode', ['ito', 'legacy_clip'])
def test_callables_accept_the_pre_0_2_0_time_arguments(spike1_model, mode,
                                                       monkeypatch):
    """The raw per-ℓ callables take every time argument the pre-0.2.0 code
    took (review round 3): ``contribution()`` no longer converts the times
    to float up front (the tie context keeps them as passed), so at k=1,
    whose single leg is the origin, the time can be anything, and at k=2
    translation-invariant symbolic times give the numeric value -- in the
    Itô mode through the tie order at an exact symbolic tie."""
    from sage.all import SR
    t = SR.var('t')
    monkeypatch.setattr(FI, 'THETA0_CONST_ROW_MODE', mode)
    res1, _c = _spike1(spike1_model, 1, 1, False, tau_grid=(0.0,))
    fn1 = res1['phase_j_by_ell'][1]['total_C']
    ref = complex(fn1(0.0))
    for arg in (0j, np.array([0.0, 1.0]), t):
        assert complex(fn1(arg)) == ref, arg
    for grouped in (False, True):
        res2, _c = _spike1(spike1_model, 2, 1, grouped)
        for ell in (0, 1):
            fn = res2['phase_j_by_ell'][ell]['total_C']
            FI._reset_runtime_counters()
            tie = complex(fn(0.0, 0.0))
            n_num = _counter('theta0_tie_ordered')
            FI._reset_runtime_counters()
            assert complex(fn(t, t)) == tie, (grouped, ell)
            assert _counter('theta0_tie_ordered') == n_num, (grouped, ell)
            assert (n_num > 0) == (mode == 'ito'), (grouped, ell)
            assert complex(fn(t, t + 3)) == complex(fn(0.0, 3.0))


# k=3 ties on a spike-train field, tree level.  Legs 1 and 2 coincide
# (non-anchor legs reach the engine as exact ties, also through dd.run: on
# the diagonals of its full k-point grid and between the non-swept legs of
# its k>=4 slices).  The regular part is continuous across t1 = t2, so
# with exactly one orientation of each tied edge holding the value at the
# tie is the limit -- from both sides here.
_K3_TIE = (0.0, 0.7, 0.7)
_H = np.array([1e-4, 1e-5, 1e-6])


def _richardson(vals, hs=None):
    hs = _H if hs is None else np.asarray(hs, dtype=float)
    A = np.vstack([np.ones(3), hs, hs ** 2]).T
    return float(np.linalg.solve(A, np.array(vals))[0])


@pytest.fixture(scope='module', params=[False, True],
                ids=['perdiag', 'grouped'])
def k3_tree(request, spike1_model):
    res, _c = _spike1(spike1_model, 3, 0, request.param, tau_grid=(0.0,))
    fn = res['total_C_by_ell'][0]
    return lambda *t: complex(fn(*t)).real


def test_k3_exact_tie_is_the_one_sided_limit(k3_tree):
    f = k3_tree
    FI._reset_runtime_counters()
    tie = f(*_K3_TIE)
    assert _counter('nquad_calls') == 0 and _counter('theta0_tie_ordered') > 0
    below = _richardson([f(0.0, 0.7, 0.7 - h) for h in _H])  # leg 2 earlier
    above = _richardson([f(0.0, 0.7, 0.7 + h) for h in _H])
    assert abs(tie - below) <= 1e-9 * abs(below)
    assert abs(tie - above) <= 1e-9 * abs(above)
    # a tie with the anchor leg: leg 1 counts as earlier than leg 0
    anchor = f(0.0, 0.0, 0.7)
    left = _richardson([f(0.0, -h, 0.7) for h in _H])
    assert abs(anchor - left) <= 1e-9 * abs(left)
    # both non-anchor legs at -1e-6: the origin of dd.run's full k=3 grid
    # (each leg at 0 -> -1e-6; the pre-0.2.0 slice point at τ = 0 too):
    # the limit with leg 2 below leg 1
    nudged = f(0.0, -1e-6, -1e-6)
    lim = _richardson([f(0.0, -1e-6, -1e-6 - h) for h in _H])
    assert abs(nudged - lim) <= 1e-9 * abs(lim)


def test_k3_near_ties_are_continuous_and_permutation_consistent(k3_tree):
    """Rounding-level near-ties (one ulp either side, user arithmetic, the
    tie with its legs swapped) are ordinary distinct times: the same value
    as the tie up to the regular part's slope times the gap."""
    f = k3_tree
    tie = f(*_K3_TIE)
    for pt in ((0.0, 0.7, 0.1 + 0.2 + 0.4),
               (0.0, 0.7, float(np.nextafter(0.7, 0.0))),
               (0.0, 0.1 + 0.2 + 0.4, 0.7)):
        assert abs(f(*pt) - tie) <= 1e-12 * abs(tie), pt
    # an anchor near-tie, both sides of it, against its exact tie
    anchor_tie = f(0.3, 0.3, 1.0)
    for pt in ((0.3, 0.1 + 0.2, 1.0), (0.1 + 0.2, 0.3, 1.0)):
        assert abs(f(*pt) - anchor_tie) <= 1e-12 * abs(anchor_tie), pt


def test_k3_translation_invariance(k3_tree):
    """t -> t + 1e6 (no time snapping: a gap far below 1e6·eps·64 stays a
    gap).  Tolerances: the translated times carry ~1e-10 rounding."""
    f = k3_tree
    for pt, rtol in (((0.0, 1e-8, 0.4), 1e-6), ((0.0, 0.7, 0.7), 1e-8),
                     ((0.0, 0.3, 1.1), 1e-8)):
        shifted = tuple(1e6 + t for t in pt)
        assert abs(f(*shifted) - f(*pt)) <= rtol * abs(f(*pt)), pt


def test_ou_quartic_near_ties_do_not_depend_on_the_mode(private_cwd):
    """A model without δ parts has no constant rows, so the Θ(0) mode and
    the tie handling can never change its numbers: bit-identical in-process
    at exact ties and rounding-level near-ties (no time is ever altered)."""
    import daedalus as dd
    from api import compute_cumulants
    model, _ = dd.load_model('ou_quartic')
    res = compute_cumulants(model, k=4, max_ell=1,
                            external_fields=[('dx', 1)] * 4,
                            parameters={'mu': 1.0, 'D': 1.0, 'eps': 0.1},
                            tau_grid=np.array([0.0]), use_cache=True,
                            parallel=False, verbose=False)
    fns = res['total_C_by_ell']
    pts = [(0.3, 0.1 + 0.2, 0.6, 0.9), (10.0, 10.0 + 1e-13, 10.5, 11.0),
           (0.0, 0.5, 0.5, 0.5), (1e6, 1e6, 1e6 + 0.5, 1e6 + 1.0)]
    vals = {}
    saved = FI.THETA0_CONST_ROW_MODE
    try:
        for mode in ('ito', 'legacy_clip'):
            FI.THETA0_CONST_ROW_MODE = mode
            FI._reset_runtime_counters()
            vals[mode] = np.array([[complex(fns[ell](*p)) for p in pts]
                                   for ell in (0, 1)])
            assert _counter('zero_normal_rows_seen') == 0
            assert _counter('nquad_calls') == 0
    finally:
        FI.THETA0_CONST_ROW_MODE = saved
    assert np.array_equal(vals['ito'], vals['legacy_clip'])
    assert np.all(np.abs(vals['ito'][1]) > 0)


# Legs of DIFFERENT fields: the regular part of ⟨n1 n2 n1⟩ jumps where
# the n2 leg meets an n1 leg, so the value at the coincident point is a
# convention.  It must be the one-sided limit in which the leg with the
# larger index approaches from below (the k=2 Itô left limit, extended),
# not the other one and not the pre-M1 value, which was neither.  Checked
# on the per-diagram AND the grouped path (grouped_integral builds its own
# ``_TieContext``; a single-field model cannot tell the two orientations
# apart, this one can).
_LD_PARAMS = {'Em': [0.8, 0.78], 'tau': [10.0, 9.0],
              'w': [[0.0, 0.25], [0.2, 0.0]]}


@pytest.fixture(scope='module', params=[False, True],
                ids=['perdiag', 'grouped'])
def ld_cross_k3(request, private_cwd):
    import daedalus as dd
    from api import compute_cumulants
    model, _ = dd.load_model('single_population_linear_delta_spikes_test')
    res = compute_cumulants(model, k=3, max_ell=0,
                            external_fields=[('n', 1), ('n', 2), ('n', 1)],
                            parameters=_LD_PARAMS, tau_grid=np.array([0.0]),
                            use_cache=True, parallel=False, verbose=False,
                            use_grouped_phase_j=request.param)
    fn = res['total_C_by_ell'][0]
    return lambda *t: complex(fn(*t)).real


# Step sizes of the one-sided samples.  At the full-grid origin the legs sit
# at −1e−6, so the samples must stay below the anchor t₀ = 0 (h <= 1e−7).
_H_API = np.array([1e-7, 1e-8, 1e-9])


@pytest.mark.parametrize('tie, larger_index_earlier, other_side, hs', [
    ((0.0, 0.7, 0.7), lambda h: (0.0, 0.7, 0.7 - h),
     lambda h: (0.0, 0.7, 0.7 + h), _H),
    ((0.3, 0.3, 0.7), lambda h: (0.3, 0.3 - h, 0.7),
     lambda h: (0.3, 0.3 + h, 0.7), _H),
    # dd.run's full-grid origin (the pre-0.2.0 k=3 slice point at τ = 0)
    ((0.0, -1e-6, -1e-6), lambda h: (0.0, -1e-6, -1e-6 - h),
     lambda h: (0.0, -1e-6, -1e-6 + h), _H_API),
], ids=['legs12', 'legs01', 'grid_origin'])
def test_cross_field_tie_takes_the_tie_order_side(
        ld_cross_k3, monkeypatch, tie, larger_index_earlier, other_side, hs):
    f = ld_cross_k3
    value = f(*tie)
    lim = _richardson([f(*larger_index_earlier(h)) for h in hs], hs)
    other = _richardson([f(*other_side(h)) for h in hs], hs)
    assert abs(other - lim) > 1e-2 * abs(lim)          # a genuine jump
    assert abs(value - lim) <= 1e-12 * abs(lim)
    monkeypatch.setattr(FI, 'THETA0_CONST_ROW_MODE', 'legacy_clip')
    legacy = f(*tie)
    assert abs(legacy - lim) > 1e-2 * abs(lim)
    assert abs(legacy - other) > 1e-2 * abs(other)


# The documented consequence (CHANGELOG 0.2.0, review round 3): the leg
# index is the argument position, so at coincident times of legs of
# DIFFERENT fields the value depends on the order in which
# ``external_fields`` lists them -- listing the same (field, time) pairs in
# another order can give the other one-sided limit.  The k = 2 τ = 0 point
# has had that convention since 0.1.0 (C_AB is sampled at τ = −1e−6: B just
# before A; C_BA(0) is the other limit).  Pinned here so that a change of
# convention is deliberate.

@pytest.fixture(scope='module')
def ld_k3_orders(private_cwd):
    import daedalus as dd
    from api import compute_cumulants
    model, _ = dd.load_model('single_population_linear_delta_spikes_test')
    out = {}
    for ext in ([('n', 1), ('n', 2), ('n', 1)], [('n', 1), ('n', 1), ('n', 2)],
                [('n', 1), ('n', 2)], [('n', 2), ('n', 1)]):
        res = compute_cumulants(model, k=len(ext), max_ell=0,
                                external_fields=ext, parameters=_LD_PARAMS,
                                tau_grid=np.array([0.0]), use_cache=True,
                                parallel=False, verbose=False)
        out[tuple(i for _f, i in ext)] = (res['total_C_by_ell'][0],
                                          res['C_tau'])
    return out


def test_cross_field_tie_depends_on_the_argument_order(ld_k3_orders):
    def fn(order):
        f = ld_k3_orders[order][0]
        return lambda *t: complex(f(*t)).real
    a, b = fn((1, 2, 1)), fn((1, 1, 2))
    # distinct times: {n1@0, n2@0.9, n1@0.7} listed both ways, one value
    assert abs(a(0.0, 0.9, 0.7) - b(0.0, 0.7, 0.9)) <= 1e-12 * abs(
        a(0.0, 0.9, 0.7))
    # {n1@0, n2@0.7, n1@0.7}: the later-listed of the tied legs is earlier
    n1_first = _richardson([a(0.0, 0.7, 0.7 - h) for h in _H])
    n2_first = _richardson([a(0.0, 0.7 - h, 0.7) for h in _H])
    assert abs(n1_first - n2_first) > 1e-2 * abs(n1_first)
    assert abs(a(0.0, 0.7, 0.7) - n1_first) <= 1e-12 * abs(n1_first)
    assert abs(b(0.0, 0.7, 0.7) - n2_first) <= 1e-12 * abs(n2_first)
    # k = 2, the τ grid's τ = 0 point (τ = −1e−6), unchanged since 0.1.0:
    # ⟨n1 n2⟩(0) is n2 just before n1, ⟨n2 n1⟩(0) n1 just before n2
    c12 = complex(ld_k3_orders[(1, 2)][1][0]).real
    c21 = complex(ld_k3_orders[(2, 1)][1][0]).real
    f12 = ld_k3_orders[(1, 2)][0]
    assert abs(c12 - c21) > 1e-2 * abs(c12)
    assert abs(c12 - complex(f12(0.0, -1e-6)).real) == 0
    assert abs(c21 - complex(f12(0.0, 1e-6)).real) <= 1e-12 * abs(c21)
