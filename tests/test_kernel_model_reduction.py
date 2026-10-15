"""
tests/test_kernel_model_reduction.py
====================================
Balanced truncation of the per-entry realizations (``api._realization``,
item 45) and the delay route's order controller (``api.delay``).  Inline
public models only:

* ``balanced_truncation`` on systems with known Hankel singular values: the
  states it discards, the H-infinity bound ``2 sum(discarded)`` against the
  error on a dense frequency grid, stability, real arithmetic, and the
  identity (the same object) when nothing can be discarded;
* graded Pade-16 entries (``|b| ~ 1e12``): the directly computed Gramian
  factors (Hammarling) keep the truncation within its tolerance;
* complex coefficients;
* the tolerance in effect: default, environment, ``truncation_scope`` (which
  clears the state-space route's realization cache), invalid values;
* ``realize_K`` / ``realize_kernel``: the truncation record, the provenance,
  the entry state counts, exact (``truncation=0``) entries unchanged;
* end to end: a delayed OU one loop with and without truncation, and the
  order controller's record (state counts, predicted walls, cost exponent,
  why it stopped).
"""
import math
import os
import sys
import warnings

import numpy as np
import pytest

pytest.importorskip('sage.all')

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from sage.all import SR, I, matrix                                      # noqa: E402

from api import compute_cumulants                                      # noqa: E402
from api import _realization as RZ                                     # noqa: E402
from api import delay as DL                                            # noqa: E402
from api.delay import pade_coefficients                                # noqa: E402
from api.noise import Gaussian                                         # noqa: E402
from api.sde import SDE                                                # noqa: E402
import engine.integration.time_domain.statespace as SS                 # noqa: E402

ENV = RZ.TRUNCATION_ENV


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    monkeypatch.delenv(ENV, raising=False)
    SS._clear_caches()
    yield
    SS._clear_caches()


def _grid_err(e0, e1, w=None):
    """max_w |g0(i w) - g1(i w)| on a dense grid (both signs of w)."""
    if w is None:
        w = np.concatenate([[0.0], np.geomspace(1e-4, 1e4, 4000)])
    s = 1j * np.concatenate([w, -w[w > 0]])
    g0 = RZ._siso_transfer(e0.a, e0.b, e0.c, s)
    g1 = RZ._siso_transfer(e1.a, e1.b, e1.c, s) if e1.n_states else 0 * g0
    return float(np.max(np.abs(g0 - g1)))


def _ent(a, b, c, d=0.0):
    a = np.asarray(a)
    return RZ.EntryRealization(a=a, b=np.asarray(b), c=np.asarray(c), d=d,
                               n_states=a.shape[0])


def _pade(n, z):
    c = [SR(x.numerator) / SR(x.denominator) for x in pade_coefficients(n)]
    return (sum(ck * (-z) ** k for k, ck in enumerate(c))
            / sum(ck * z ** k for k, ck in enumerate(c)))


def _delayed_alpha_K(n, d=SR(3) / 2):
    """A scalar loop ``K = i w + 1 - (1/2) P_n(i w d) / (1 + i w)^2``."""
    W = SR.var('omega')
    z = I * W * d
    return matrix(SR, [[I * W + 1 - SR(1) / 2 * _pade(n, z) / (1 + I * W) ** 2]]), W


# ── balanced truncation on known systems ─────────────────────────────────


def test_tiny_mode_is_discarded_within_its_bound():
    # g = 1/(s + 1) + 1e-12/(s + 5): one Hankel singular value ~ 1e-13
    e = _ent(np.diag([-1.0, -5.0]), [1.0, 1.0], [1.0, 1e-12])
    e2, info = RZ.truncate_entry(e, 1e-10)
    assert (info['n'], info['r'], e2.n_states) == (2, 1, 1)
    assert info['bound_rel'] <= 1e-10 and info['err_rel'] <= 1e-10
    assert np.isclose(e2.a[0, 0], -1.0, rtol=1e-12)
    assert _grid_err(e, e2) <= info['bound'] * (1 + 1e-6) + 1e-16
    assert e2.a.dtype == np.float64 and e2.b.dtype == np.float64


def test_nothing_to_discard_returns_the_same_object():
    e = _ent(np.diag([-1.0, -3.0]), [1.0, 1.0], [1.0, -0.5])
    for tol in (1e-10, 1e-3):
        e2, info = RZ.truncate_entry(e, tol)
        assert e2 is e and info['kept'] == 'no_small_hsv' and info['r'] == 2
    e2, info = RZ.truncate_entry(e, 0.0)
    assert e2 is e and info['kept'] == 'tol_zero'
    e1 = _ent([[-2.0]], [1.0], [1.0])
    assert RZ.truncate_entry(e1, 1.0)[0] is e1


def test_unstable_entry_is_never_truncated():
    e = _ent(np.diag([-1.0, 0.5]), [1.0, 1.0], [1.0, 1e-14])
    e2, info = RZ.truncate_entry(e, 1e-6)
    assert e2 is e and info['kept'] == 'not_hurwitz'


def test_hsv_of_a_first_order_section():
    # 1/(s + a): the single Hankel singular value is 1/(2a)
    for a in (0.3, 2.0, 17.0):
        sv = RZ.hankel_singular_values(np.array([[-a]]), np.array([1.0]), np.array([1.0]))[0]
        assert sv[0] == pytest.approx(1 / (2 * a), rel=1e-14)


# ── rational kernels: delays, Pade blocks, graded entries ────────────────


@pytest.mark.parametrize('n,tol', [(8, 1e-5), (12, 1e-6), (16, 1e-7)])
def test_delayed_alpha_loop_truncates_within_tolerance(n, tol):
    K, W = _delayed_alpha_K(n)
    exact = RZ.realize_K(K, W, {}, truncation=0)
    red = RZ.realize_K(K, W, {}, truncation=tol)
    e0, e1 = exact.entries[(0, 0)], red.entries[(0, 0)]
    assert e0.n_states == n + 3
    assert exact.truncation['n_truncated'] == 0 and exact.truncation['tol'] == 0
    rec = red.truncation['entries'][(0, 0)]
    assert e1.n_states == rec['r'] < e0.n_states            # something was discarded
    s1 = rec['hsv'][0]
    err = _grid_err(e0, e1)
    assert err <= tol * s1 * (1 + 1e-6) and rec['err_rel'] <= tol
    assert np.all(np.linalg.eigvals(e1.a).real < 0)
    assert red.entry_states[0, 0] == e1.n_states
    assert any('balanced truncation' in p for p in red.provenance)
    # the global realization is untouched (the exact tree and the Hurwitz
    # test use it)
    assert np.array_equal(red.A, exact.A) and np.array_equal(red.B, exact.B)


def test_graded_pade16_entry_needs_the_direct_factors():
    """S1's Pade-16 entries are graded (here |b| ~ 3e11 after balancing ``a``
    alone; on the two-neuron demo factors taken from eigh(P) lost everything
    below ~1e-8 relative and the truncation failed its check).  The direct
    (Hammarling) factors keep the truncation within its tolerance."""
    K, W = _delayed_alpha_K(16)
    e0 = RZ.realize_K(K, W, {}, truncation=0).entries[(0, 0)]
    assert np.abs(e0.b).max() > 1e10                          # graded
    e1, info = RZ.truncate_entry(e0, 1e-7)
    assert info['kept'] is None and e1.n_states < e0.n_states
    assert _grid_err(e0, e1) <= 1e-7 * info['hsv'][0] * (1 + 1e-6)
    # the factors themselves: Z Z^T solves the Lyapunov equation
    a, b = e0.a, e0.b
    Z = RZ._lyap_factor(a, b)
    P = Z @ Z.T
    res = a @ P + P @ a.T + np.outer(b, b)
    scale = 2 * np.abs(a).max() * np.abs(P).max() + np.abs(np.outer(b, b)).max()
    assert np.abs(res).max() <= 1e-13 * scale


def test_complex_coefficients():
    W = SR.var('omega')
    z = I * W
    K = matrix(SR, [[I * W + 1 + (2 + I / 2) * _pade(8, z) / (1 + I * W) ** 2]])
    exact = RZ.realize_K(K, W, {}, truncation=0)
    red = RZ.realize_K(K, W, {}, truncation=1e-6)
    e0, e1 = exact.entries[(0, 0)], red.entries[(0, 0)]
    assert red.field_name == 'QQ[I]' and np.iscomplexobj(e1.a)
    assert e1.n_states < e0.n_states
    rec = red.truncation['entries'][(0, 0)]
    assert _grid_err(e0, e1) <= 1e-6 * rec['hsv'][0] * (1 + 1e-6)


def test_conductance_kernel_realization_truncates():
    W = SR.var('wk')
    # a delayed alpha kernel alone: Hankel singular values decay only
    # algebraically (an all-pass factor), so a loose tolerance
    img = _pade(12, I * W * SR(3) / 2) / (1 + I * W) ** 2
    att = {'symbol': 'g_test45', 'kernel_ft_fn': lambda w, _e=img: _e.subs({W: w})}
    e0, f0 = RZ.realize_kernel(att, {}, truncation=0)
    e1, f1 = RZ.realize_kernel(att, {}, truncation=5e-3)
    assert f0 == f1 == 'QQ' and e0.n_states == 14 and e1.n_states < 14
    hsv = RZ.hankel_singular_values(e0.a, e0.b, e0.c)[0]
    assert _grid_err(e0, e1) <= 5e-3 * hsv[0] * (1 + 1e-6)
    # the tolerance in effect applies when none is passed
    with RZ.truncation_scope(0):
        assert RZ.realize_kernel(att, {})[0].n_states == 14


# ── the tolerance in effect ──────────────────────────────────────────────


def test_tolerance_default_environment_and_scope(monkeypatch):
    assert RZ.truncation_tol() == RZ.TRUNCATION_TOL_DEFAULT == 1e-10
    for v, want in (('0', 0.0), ('off', 0.0), ('exact', 0.0), ('1e-7', 1e-7),
                    ('default', 1e-10), ('', 1e-10)):
        monkeypatch.setenv(ENV, v)
        assert RZ.truncation_tol() == want, v
    # a malformed value warns (once) and the default applies (a ValueError
    # would be swallowed by the state-space route and send every model to
    # the pole route silently)
    for bad, msg in (('lots', ENV), ('-1', 'finite')):
        monkeypatch.setenv(ENV, bad)
        with pytest.raises(ValueError, match=msg):
            RZ._env_truncation_tol()
        with pytest.warns(UserWarning, match=msg):
            assert RZ.truncation_tol() == RZ.TRUNCATION_TOL_DEFAULT
        with RZ.truncation_scope(0):                      # an explicit scope still works
            assert RZ.truncation_tol() == 0.0
    monkeypatch.setenv(ENV, '1e-9')
    with RZ.truncation_scope(1e-4):
        assert RZ.truncation_tol() == 1e-4
        with RZ.truncation_scope(None):                     # back to the environment
            assert RZ.truncation_tol() == 1e-9
        with RZ.truncation_scope(0):
            assert RZ.truncation_tol() == 0.0
        assert RZ.truncation_tol() == 1e-4
    assert RZ.truncation_tol() == 1e-9
    with pytest.raises(ValueError):
        RZ.truncation_scope(-1e-3)


def test_scope_clears_the_state_space_realization_cache():
    SS._REAL_CACHE['probe'] = (None, None, 'x')
    with RZ.truncation_scope(RZ.truncation_tol()):          # no change: kept
        assert 'probe' in SS._REAL_CACHE
    with RZ.truncation_scope(1e-6):                          # a change: cleared
        assert 'probe' not in SS._REAL_CACHE
        SS._REAL_CACHE['probe'] = (None, None, 'x')
    assert 'probe' not in SS._REAL_CACHE                     # and on exit


def test_delay_option_validation():
    o = DL.resolve_options({'truncation': 0})
    assert o['truncation'] == 0.0 and DL._truncation_tol(o) == 0.0
    o = DL.resolve_options({})
    assert o['truncation'] == 'auto'
    assert DL._truncation_tol(o) == min(RZ.TRUNCATION_TOL_DEFAULT, 0.01 * o['tol'])
    assert DL._truncation_tol(DL.resolve_options({'tol': 1e-12})) == pytest.approx(1e-14)
    with RZ.truncation_scope(0):                              # 'auto' follows the exact setting
        assert DL._truncation_tol(DL.resolve_options({})) == 0.0
    with pytest.raises(ValueError, match='truncation'):
        DL.resolve_options({'truncation': -1})


# ── end to end ───────────────────────────────────────────────────────────

TAUS = np.array([0.0, 0.5, 1.0, 2.0, 4.0])


def _ou_delayed(name, d=0.5):
    return (SDE(name).parameter('a', default=1.0).parameter('c', default=-2.0)
            .parameter('d', default=d, domain='positive').parameter('eps', default=0.02)
            .parameter('D', default=0.1).field('x')
            .drift('x', '-a*x + c*x(t - d) - eps*x^3')
            .noise('xi', Gaussian(var='2*D'), couples={'x': 1}).build())


def _cc(model, ell, **kw):
    kw.setdefault('use_cache', False)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return compute_cumulants(model, k=2, max_ell=ell, external_fields=[('x', 1)] * 2,
                                 tau_grid=TAUS, parallel=False, verbose=False,
                                 taylor_order=4, **kw)


def test_delayed_ou_one_loop_truncated_vs_exact():
    m = _ou_delayed('red45 delayed ou')
    r0 = _cc(m, 1, delay={'order': 8, 'truncation': 0})
    r1 = _cc(m, 1, delay={'order': 8})
    i0, i1 = r0['delay'], r1['delay']
    assert i0['truncation_tol'] == 0 and i1['truncation_tol'] == 1e-10
    loop0, loop1 = (np.real(r['C_tau_by_ell'][1]) for r in (r0, r1))
    tot0 = np.real(r0['C_tau'])
    # the per-entry bound (relative 1e-10) carries to the observable
    assert np.max(np.abs(loop1 - loop0)) <= 1e-9 * np.max(np.abs(tot0))
    assert np.array_equal(np.real(r0['C_tau_by_ell'][0]), np.real(r1['C_tau_by_ell'][0]))
    assert i1['stop_reason'] == 'fixed order n = 8'
    led = [e for e in r1['analysis'].ledger if e.name.startswith('delay:')]
    assert led and 'balanced-truncated' in led[0].detail
    led0 = [e for e in r0['analysis'].ledger if e.name.startswith('delay:')]
    assert led0 and 'balanced-truncated' not in led0[0].detail


def test_order_controller_record_and_prediction():
    m = _ou_delayed('red45 delayed ou auto')
    r = _cc(m, 1, delay={'tol': 1e-6})
    info = r['delay']
    assert info['orders_run'] == [4, 8] and info['target_met'] is True
    assert 'target 1e-06 met at n = 8' in info['stop_reason']
    # the controller built the realizations of n = 4 and n = 8 (truncated as
    # those runs were) before running 8
    cb = info['cost_by_order']
    assert set(cb) == {4, 8} and cb[8]['states'] > cb[4]['states'] > 0
    assert cb[8]['proxy'] == pytest.approx(cb[8]['states'] ** 2 * (1 + cb[8]['stiffness']),
                                           rel=1e-2)
    assert set(info['predicted_walls_s']) == {8} and info['cost_correction'] == 1.0
    want = info['walls_s'][4] * cb[8]['proxy'] / cb[4]['proxy']
    assert info['predicted_walls_s'][8] == pytest.approx(want, rel=0.02, abs=0.2)
    # the first order at the base tolerance, the second adaptive: 1 % of the
    # loop error predicted for it, between the base and TRUNCATION_MAX
    tb = info['truncation_by_order']
    assert tb[4] == 1e-10 and info['truncation_tol'] == tb[8]
    e8 = info['loop_error_pred_by_order'][8]
    assert tb[8] == pytest.approx(max(1e-10, min(DL.TRUNCATION_MAX, DL.TRUNCATION_KAPPA * e8)))
    # what matters: the truncation moved the n = 8 observable by far less
    # than the Pade error the controller estimates (exact realizations here)
    r0 = _cc(m, 1, delay={'order': 8, 'truncation': 0})
    d = np.max(np.abs(np.real(r['C_tau']) - np.real(r0['C_tau'])))
    assert d / np.max(np.abs(np.real(r0['C_tau']))) <= 1e-2 * info['est_by_order'][8]


def test_stiffness_is_the_taylor_driver():
    # a normal matrix: the spectral spread about the mean
    assert DL._stiffness(np.diag([-1.0, -3.0])) == pytest.approx(1.0)
    # a non-normal block: min over p of ||(a - mu)^p||^(1/p), small though
    # ||a - mu|| = 101 (here (a - mu)^2 = I)
    assert DL._stiffness(np.array([[-2.0, 100.0], [0.0, -4.0]])) == pytest.approx(1.0)
    # a Jordan block: the nilpotent part ends the series (as in the Taylor action)
    assert DL._stiffness(np.array([[-2.0, 100.0], [0.0, -2.0]])) == 0.0
    assert DL._stiffness(np.zeros((0, 0))) == 0.0


def test_prediction_from_the_realizations_and_fallbacks():
    from api import analyze as AN
    m = _ou_delayed('red45 delayed ou ctx')
    _rep, ctx = AN.analyze_with_context(
        m, AN.Question(k=2, external_fields=(('x', 1), ('x', 1)), max_ell=1), None,
        use_cache=False)
    cache = {}
    ctrl = {'cost_by_order': {}, 'predicted_walls_s': {}, 'cost_correction': None}
    g, how = DL._predicted_growth(ctx, 4, 8, None, {4: 10.0}, cache, ctrl)
    t = RZ.truncation_tol()
    c4, c8 = cache[(4, t)], cache[(8, t)]
    assert c8['states'] > c4['states'] and g == pytest.approx(c8['proxy'] / c4['proxy'])
    assert 'states' in how and ctrl['cost_correction'] == 1.0
    # two measured orders: the proxy is corrected by its error on that step
    walls = {4: 10.0, 8: 10.0 * 1.5 * c8['proxy'] / c4['proxy']}
    g2, how2 = DL._predicted_growth(ctx, 8, 16, 4, walls, cache, ctrl)
    c16 = cache[(16, t)]
    assert ctrl['cost_correction'] == pytest.approx(1.5, rel=1e-3)
    assert g2 == pytest.approx(1.5 * c16['proxy'] / c8['proxy']) and 'last step' in how2
    # per-order tolerances: the next order's realization truncated as it will run
    g3, _ = DL._predicted_growth(ctx, 4, 8, None, {4: 10.0}, cache, ctrl,
                                 tols={4: t, 8: 1e-3})
    assert (8, 1e-3) in cache and cache[(8, 1e-3)]['states'] <= c8['states']
    # no realization (a context without the operator): the rule before item
    # 45, the larger of (n_next / n)^4 and the measured growth
    g4, how4 = DL._predicted_growth(None, 4, 8, None, {4: 1.0}, {}, ctrl)
    assert g4 == 16.0 and 'n_next / n' in how4 and 'no realization' in how4
    g5, how5 = DL._predicted_growth(None, 8, 16, 4, {4: 1.0, 8: 6.0}, {}, ctrl)
    assert g5 == 16.0 and 'n_next / n' in how5
    g6, how6 = DL._predicted_growth(None, 8, 16, 4, {4: 1.0, 8: 20.0}, {}, ctrl)
    assert g6 == 20.0 and 'measured' in how6
    # the last run did not take the state-space route everywhere: the old rule
    r_poles = {'phase_j_engine': {'engine': 'statespace', 'statespace_share': 0.5}}
    g7, how7 = DL._predicted_growth(ctx, 4, 8, None, {4: 1.0}, cache, ctrl, r_last=r_poles)
    assert g7 == 16.0 and 'state-space route' in how7
    # a realization that does not grow with the order (the delay only in a
    # nonlinear term): the old rule
    flat = {(4, t): {'states': 3.0, 'stiffness': 1.0, 'proxy': 18.0},
            (8, t): {'states': 3.0, 'stiffness': 1.0, 'proxy': 18.0}}
    g8, how8 = DL._predicted_growth(ctx, 4, 8, None, {4: 1.0}, flat, ctrl)
    assert g8 == 16.0 and 'nonlinear term' in how8


# ── structural redundancy: what S1's exact Kalman reduction already removes ─


def test_identical_kernels_and_zero_weights_leave_no_redundant_states():
    """Field 3 drives fields 1 and 2 through the SAME alpha kernel (two
    copies in the pencil), and a third kernel enters with weight w0 = 0 at
    the parameter point.  The duplicate chain is exactly uncontrollable and
    the zero-weight kernel never becomes a state: the exact Kalman reduction
    (S1) leaves the minimal counts, so the truncation has nothing structural
    left to remove (it only removes numerically negligible states)."""
    W = SR.var('omega')
    w0 = SR.var('w0')
    g = 1 / (1 + I * W) ** 2
    K = matrix(SR, [[I * W + 1, 0, -SR(1) / 2 * g],
                    [0, I * W + 1, -SR(1) / 2 * g],
                    [w0 * g, 0, I * W + 2]])
    real = RZ.realize_K(K, W, {w0: 0.0})
    # pencil: 3 fields + 2 x 2 kernel states (the w0 entry is 0: no chain)
    assert real.conditioning['n_kernel_states'] == 4
    assert real.n_states == 5                        # the duplicate chain is gone
    es = real.entry_states
    assert es[0, 2] == es[1, 2] == 4 and es[0, 0] == es[1, 1] == 1 and es[2, 2] == 1
    assert real.truncation['n_truncated'] == 0


def test_next_truncation_rules():
    o = DL.resolve_options({})
    ctrl = {'loop_error_pred': {}}
    tot = np.array([1.0, 0.5, 0.2])
    r = {'C_tau': tot, 'C_tau_by_ell': {0: tot - np.array([0.1, 0.05, 0.0]), 1: np.array([0.1, 0.05, 0.0])},
         'delay_tree_vs_pade': 1e-2}
    # from the tree: 1e-2 * share 0.1 * (4/8)^3; from the estimate: 1e-3 * (2/8)^3
    t = DL._next_truncation(o, 1e-10, r, 2, 4, 8, 1e-3, 2, ctrl)   # est measures n = 2
    e = min(1e-2 * 0.1 / 8, 1e-3 / 64)
    assert ctrl['loop_error_pred'][8] == pytest.approx(e) and t == pytest.approx(1e-2 * e)
    # capped above, floored at the base, untouched when pinned or exact
    assert DL._next_truncation(o, 1e-10, dict(r, delay_tree_vs_pade=10.0), 2, 4, 8, None,
                               None, ctrl) == DL.TRUNCATION_MAX
    assert DL._next_truncation(o, 1e-10, dict(r, delay_tree_vs_pade=1e-14), 2, 4, 8, None,
                               None, ctrl) == 1e-10
    assert DL._next_truncation(o, 1e-10, {'C_tau': tot}, 2, 4, 8, None, None, ctrl) == 1e-10
    # after a skipped order the estimate still measures the order it compared
    # with: order 16 after 4 (8 skipped), the estimate of 4 measuring 2
    t16 = DL._next_truncation(o, 1e-10, {'C_tau': tot}, 2, 4, 16, 1e-3, 2, ctrl)
    assert ctrl['loop_error_pred'][16] == pytest.approx(1e-3 * (2 / 16) ** 3)
    assert DL._next_truncation(DL.resolve_options({'truncation': 1e-9}), 1e-9, r, 2, 4, 8,
                               1e-3, 2, ctrl) == 1e-9
    assert DL._next_truncation(o, 0.0, r, 2, 4, 8, 1e-3, 2, ctrl) == 0.0
    with pytest.raises(ValueError, match='truncation'):
        DL.resolve_options({'truncation': None})


def test_scope_exits_interleaved_across_threads():
    """Scopes opened and closed out of order (two threads): each exit removes
    only its own entry."""
    a, b = RZ.truncation_scope(0.0), RZ.truncation_scope(1e-6)
    a.__enter__()
    b.__enter__()
    assert RZ.truncation_tol() == 1e-6
    a.__exit__(None, None, None)                  # A leaves first: B stays in force
    assert RZ.truncation_tol() == 1e-6
    b.__exit__(None, None, None)
    assert RZ.truncation_tol() == RZ.TRUNCATION_TOL_DEFAULT


def test_tiny_tolerance_is_floored_and_kept():
    e = _ent(np.diag([-1.0, -5.0]), [1.0, 1.0], [1.0, 1e-12])
    e2, info = RZ.truncate_entry(e, 1e-16)
    assert info['tol'] == RZ.TRUNCATION_TOL_MIN and e2 is e     # 1.8e-13 > 1e-13: kept
    e2, info = RZ.truncate_entry(e, 1e-12)
    assert e2.n_states == 1 and info['err_rel'] <= 1e-12


def test_symbolic_loop_refuses_a_truncated_reference():
    from api import symbolic_loop as SL
    m = _ou_delayed('red45 symbolic ref')        # not a symbolic-route model: only the guard
    r = _cc(m, 0, delay={'order': 16, 'tree': 'pade', 'truncation': 0})
    with RZ.truncation_scope(1e-3):
        with pytest.raises(SL.SymbolicLoopRefused, match='truncated_reference'):
            SL._refuse_truncated_reference(r)
    with RZ.truncation_scope(0):
        SL._refuse_truncated_reference(r)        # exact setting: nothing to refuse
