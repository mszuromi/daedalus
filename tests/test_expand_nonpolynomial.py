"""FieldTheory.expand on non-polynomial actions: truncated per-atom Taylor.

The action of the inline model below has two sigmoids ``1/(1+exp(u))`` of
linear combinations of two fields, linear in the drift and squared /
multiplied in a state-dependent noise covariance, and a positive parameter
(an assumption symbol).  The legacy single ``taylor()`` + ``PolynomialRing``
zero tests took ~2 s / 9 s / 110 s at taylor_order 2 / 3 / 4 on it; the
truncated route takes well under a second.

Checks:
  * the expansion equals the one of the hand-expanded polynomial form
    (the sigmoid replaced by its Taylor polynomial at the saddle u = 0,
    exact to the order used) -- exactly, coefficient by coefficient;
  * the retired legacy switch (``DAEDALUS_EXPAND_LEGACY``, audit W7) only
    warns and changes nothing;
  * a time budget at taylor_order 4;
  * unit checks of the truncated Taylor, of the fast nonzero test and of
    the dict-to-ring construction.
"""
import os
import random
import time

import pytest

sage = pytest.importorskip('sage.all')
from sage.all import (SR, RealField, PolynomialRing, taylor, exp, log, sin,
                      cos, tanh, function, heaviside, sqrt, assume, forget)

import engine.core.field_theory as F
from api.model import TemporalModelBuilder

_SIG = {'exp': '(1/(1+exp({u})))',
        # Taylor polynomial of 1/(1+exp(u)) at u = 0, exact through u^4
        'poly': '(1/2 - ({u})/4 + ({u})^3/48)'}


def _model(form='exp'):
    s = _SIG[form].format
    s1, s2 = s(u='2*x[i] + y[i]'), s(u='x[i] + 2*y[i]')
    f1, f2 = f'(1/2 - {s1})', f'(1/2 - {s2})'
    action = (f'sum(xt[i]*(Dt*x[i] + {f1}) + yt[i]*(Dt*y[i] + {f2})'
              f' - qamp/2*({s1}^2*xt[i]^2 + 2*{s1}*{s2}*xt[i]*yt[i]'
              f' + {s2}^2*yt[i]^2) for i in pop)')
    return (TemporalModelBuilder(f'two-field sigmoid drift ({form})')
            .population('pop', size=1)
            .physical_field('x', population='pop')
            .physical_field('y', population='pop')
            .parameter('qamp', default=0.3, domain='positive')
            .set_action_text(action)
            .equation(lhs='(Dt+1)*x[i]', rhs=f'x[i] - {f1}', population='pop')
            .equation(lhs='(Dt+1)*y[i]', rhs=f'y[i] - {f2}', population='pop')
            .build())


def _expand(model, order, monkeypatch=None):
    ft = F.FieldTheory(model, taylor_order=order)
    ft.expand()
    return ft


def _frozen_legacy(key):
    """The retired legacy route's table (``DAEDALUS_EXPAND_LEGACY=1``, frozen
    at main dca8804: tests/fixtures/expand_legacy_taylor.json)."""
    import json
    path = os.path.join(os.path.dirname(__file__), 'fixtures',
                        'expand_legacy_taylor.json')
    return json.load(open(path))[key]


def _eval_frozen(text, values):
    """A frozen coefficient's value at ``values`` ({symbol name: number}):
    parsed with only those names bound (``SR(text)`` would read a name that
    another test declared as a symbol, e.g. ``e``, as that symbol)."""
    from sage.all import sage_eval
    return sage_eval(text, locals=dict(values))


def _table(ft):
    out = {}
    for key, poly in ft._by_tp.items():
        for mono, c in poly.dict().items():
            out[(key, tuple(mono))] = SR(c)
    return out


def _saddle_zero(ft):
    """The saddle of the inline model is x* = y* = 0."""
    return {v: 0 for v in SR(sum(_table(ft).values())).variables()
            if repr(v).endswith('star1')}


@pytest.mark.parametrize('order', [2, 3, 4])
def test_matches_hand_expanded_polynomial_form(order, monkeypatch):
    ft_e = _expand(_model('exp'), order, monkeypatch=monkeypatch)
    ft_p = _expand(_model('poly'), order, monkeypatch=monkeypatch)
    te, tp = _table(ft_e), _table(ft_p)
    sub = _saddle_zero(ft_e)
    assert sub, 'expected saddle symbols in the coefficients'
    keys = set(te) | set(tp)
    assert len(keys) > 5
    for k in keys:
        d = (te.get(k, SR(0)).subs(sub) - tp.get(k, SR(0)).subs(sub)).expand()
        assert d.is_trivial_zero(), (k, d)


def test_retired_legacy_switch_warns_and_changes_nothing(monkeypatch):
    from engine import _env
    order = 2
    model = _model('exp')
    tn = _table(_expand(model, order, monkeypatch=monkeypatch))
    # the frozen legacy table (the route the switch selected) still agrees
    fl = _frozen_legacy('exp2')
    assert {repr(k) for k in tn} == set(fl)
    RF = RealField(200)
    syms = sorted({v for c in tn.values() for v in c.variables()}, key=repr)
    pt = {s: RF(random.Random(0).uniform(0.2, 0.8)) for s in syms}
    by_name = {repr(s): v for s, v in pt.items()}
    for k in tn:
        a = RF(tn[k].subs(pt))
        b = RF(_eval_frozen(fl[repr(k)], by_name))
        assert abs(a - b) <= RF(10) ** -50 * max(abs(a), RF(1)), k
    _env._RETIRED_SEEN.discard(F._EXPAND_LEGACY_ENV)
    monkeypatch.setenv(F._EXPAND_LEGACY_ENV, '1')
    with pytest.warns(FutureWarning, match='DAEDALUS_EXPAND_LEGACY is retired'):
        tl = _table(_expand(model, order, monkeypatch=monkeypatch))
    assert set(tn) == set(tl)
    assert all((tn[k] - tl[k]).is_trivial_zero() for k in tn)


def test_equals_legacy_taylor_order3(monkeypatch):
    model = _model('exp')
    tn = _table(_expand(model, 3, monkeypatch=monkeypatch))
    tl = _frozen_legacy('exp3')
    assert {repr(k) for k in tn} == set(tl)
    RF = RealField(200)
    syms = sorted({v for c in tn.values() for v in c.variables()}, key=repr)
    pt = {s: RF(random.Random(5).uniform(0.2, 0.8)) for s in syms}
    by_name = {repr(s): v for s, v in pt.items()}
    for k in tn:
        a = RF(tn[k].subs(pt))
        b = RF(_eval_frozen(tl[repr(k)], by_name))
        assert abs(a - b) <= RF(10) ** -50 * max(abs(a), RF(1)), k


def test_time_budget_order4(monkeypatch):
    model = _model('exp')
    t0 = time.perf_counter()
    ft = _expand(model, 4, monkeypatch=monkeypatch)
    dt = time.perf_counter() - t0
    assert sum(len(p.dict()) for p in ft._by_tp.values()) == 36
    assert dt < 60.0, f'expand at taylor_order 4 took {dt:.1f} s'


def test_expand_cache_roundtrip_fast(monkeypatch, tmp_path):
    """Loading the cached expansion re-tests every coefficient; with the
    legacy ``R(dict)`` that was ~100 s on a model like this one."""
    from api import _expand_cache as EC
    monkeypatch.chdir(tmp_path)
    model = _model('exp')
    ft = _expand(model, 4, monkeypatch=monkeypatch)
    EC.save_expand(model, ft)
    for target in (4, 3):
        ft2 = F.FieldTheory(model, taylor_order=target)
        EC.prepare_for_load(ft2)
        t0 = time.perf_counter()
        assert EC.load_expand(model, ft2, target_order=target, cached_order=4)
        assert time.perf_counter() - t0 < 30.0
        for key, poly in ft2._by_tp.items():
            ref = ft._by_tp[key].dict()
            assert list(poly.dict()) == list(ref)
            assert all(SR(c).is_trivially_equal(SR(ref[m]))
                       for m, c in poly.dict().items())
        assert ft2._S_raw.dict().keys() == {
            m for p in ft2._by_tp.values() for m in p.dict()}


def test_polynomial_action_identical_to_legacy(monkeypatch):
    """Polynomial actions: the truncated route returns the very same ring
    element (coefficients trivially equal), not only an equal one."""
    model = _model('poly')
    tn = _table(_expand(model, 4, monkeypatch=monkeypatch))
    tl = _frozen_legacy('poly4')
    assert {repr(k) for k in tn} == set(tl)
    for k in tn:
        assert str(tn[k]) == tl[repr(k)], k


# ── unit checks ──────────────────────────────────────────────────────────

def _check_against_taylor(expr, fvars, order):
    got = F._truncated_taylor(expr, fvars, order)
    ref = taylor(expr, *[(v, 0) for v in fvars], order)
    mine = sum(c * sage.prod(v ** k for v, k in zip(fvars, e))
               for e, c in got.items())
    assert all(sum(e) <= order for e in got)
    if SR(ref).operator() is not None and any(
            type(o).__name__ in ('NewSymbolicFunction', 'FDerivativeOperator')
            for o in _operators(SR(ref))):
        # formal functions: not evaluable, compare the expanded forms
        assert (SR(mine) - SR(ref)).expand().is_trivial_zero(), expr
        return
    RF = RealField(150)
    syms = sorted(set(SR(ref).variables()) | set(SR(mine).variables()), key=repr)
    for trial in range(3):
        rnd = random.Random(10 + trial)
        pt = {s: RF(rnd.uniform(0.1, 0.4)) for s in syms}
        a, b = RF(SR(mine).subs(pt)), RF(SR(ref).subs(pt))
        assert abs(a - b) <= RF(10) ** -40 * max(abs(b), RF(1)), (expr, a, b)


def _operators(e):
    op = e.operator()
    if op is None:
        return []
    out = [op]
    for o in e.operands():
        out += _operators(o)
    return out


def test_truncated_taylor_matches_taylor():
    x, y, a, b = SR.var('x y a b')
    phi = function('phi_tt')
    cases = [
        exp(a + x - 2 * y) * (1 + x * y),
        1 / (1 + exp(a + 3 * x + y)) ** 2,
        log(1 + exp(b + x)) * y,
        (a + x + y ** 2) ** (sage.QQ(-3) / 2),
        sqrt(a + x) + (1 + x * y) ** 5,
        tanh(b + x - y) + sin(x) * cos(a + y),
        phi(a + x + y) * x,
        x ** 3 * y ** 2 + a * x - b,
    ]
    for e in cases:
        for order in (2, 4):
            _check_against_taylor(e, [x, y], order)


def test_non_polynomial_atom_falls_back_to_taylor():
    """An atom whose expansion at 0 is a Laurent / Puiseux series: the
    whole action goes through the legacy ``taylor()``, unchanged."""
    x, a = SR.var('x a')
    for e in (exp(a + x) / x, sqrt(x) * exp(x)):
        with pytest.raises(F._TruncatedTaylorUnsupported):
            F._truncated_taylor(e, [x], 3)
        assert F._taylor_action(e, [x], 3).is_trivially_equal(
            taylor(e, (x, 0), 3))


def test_any_atom_is_expanded_by_taylor():
    x, y, a = SR.var('x y a')
    e = heaviside(a + x) * exp(x + y) + y ** 2 * abs(a - x)
    got = F._truncated_taylor(e, [x, y], 3)
    mine = sum(c * x ** i * y ** j for (i, j), c in got.items())
    ref = taylor(e, (x, 0), (y, 0), 3)
    assert (SR(mine) - SR(ref)).expand().is_trivial_zero()


def test_sr_nonzero_agrees_with_bool():
    u, w = SR.var('u_nz w_nz')
    assume(u > 0)
    try:
        cases = [
            u * (w - 3) + exp(w) / (1 + exp(u)),   # nonzero (fast path)
            sin(w) ** 2 + cos(w) ** 2 - 1,         # zero, not trivially
            SR(0),
            u - u,
            sqrt(u ** 2) - u,                      # zero only for u > 0
            sage.RR(0.1) * u * w + sage.RR(0.3),   # float literals
            sage.erf(w) + sage.erfc(w) - 1,        # no rigorous interval
            u * w - sage.RR(0.5) * u * w * 2,      # float residue
            -(u + 1) * w / 10 + (sage.RR(0.1) * u + sage.RR(0.1)) * w,
        ]
        for c in cases:
            assert F._sr_nonzero(c) == bool(c), c
    finally:
        forget(u > 0)


def test_ring_from_nonzero_dict_is_R_of_dict():
    R = PolynomialRing(SR, ['p1', 'p2'])
    a, b = SR.var('a_r b_r')
    d = {(1, 0): a + 1, (0, 2): exp(b), (1, 1): SR(3)}
    el = F._ring_from_nonzero_dict(R, d)
    ref = R(d)
    assert type(el) is type(ref)
    assert el.parent() is R
    assert el.dict() == ref.dict()
    assert el == ref
    assert F._ring_from_nonzero_dict(R, {}) == R.zero()
    assert el + ref == 2 * ref
