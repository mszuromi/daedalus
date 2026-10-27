"""Mean-field roots at an exact 0 whose rows cancel only to round-off.

A two-node shot-noise model: node i jumps by +/- s along the coupling vector
(k_ii, k_ji) at state-dependent rates

    up_i   = (p_i + c_i e^{-x_i}) / s,     p_i = (g_i + a_i - c_i) / 2,
    down_i = (a_i e^{x_i} + n_i) / s,      n_i = (g_i - a_i + c_i) / 2,

so the drift is ``K h(x)`` with ``h_i = (a_i - c_i) + c_i e^{-x_i} - a_i e^{x_i}``:
the origin is the exact, stable saddle at every parameter point, while each
row evaluates there to a few ulps of its O(g) terms, never to an exact 0.

The solver rejected it ("no MF fixed point found from 64 seeds"): hybr's stop
test is a step relative to ``x`` (it reports no progress when converging to 0
against a round-off residual floor), and the acceptance scale of a component
whose every converged point is 0 is 0 as well.  A row whose linear terms are
round-off is now satisfied at the round-off of its own terms; rows with a linear
scale keep the relative test (a residual above round-off at 0, or a point in a
decaying drift's flat tail, is still refused, below).
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest
from scipy.linalg import expm, solve_continuous_lyapunov

pytest.importorskip('sage.all')

import daedalus as dd                                                 # noqa: E402
from daedalus.api import _mean_field_dae as MF                                 # noqa: E402
from daedalus.api.model import TemporalModelBuilder                            # noqa: E402
from daedalus.api.noise import Poisson                                         # noqa: E402
from daedalus.api.sde import SDE                                               # noqa: E402

_NAMES = ['a1', 'c1', 'g1', 'a2', 'c2', 'g2', 's', 'k11', 'k12']
_R = 0.2
_P = dict(a1=0.13201444627312386, c1=0.13201444627312386, g1=3.9022068858229244,
          a2=10.185708437929115, c2=0.005351009345574485, g2=12.712463325126727,
          s=0.02099778393266374, k11=1 / (1 - _R * _R), k12=_R / (1 - _R * _R))
# g_i enters only the split of the rates (it cancels from the drift): another
# split of the same drift, the case where no hybr start reported success
_P_SHIFTED = dict(_P, g1=_P['g1'] + 1e-3, g2=_P['g2'] + 1e-3)
_BOX = {'u': (-0.05, 0.05), 'w': (-0.05, 0.05)}


def _sde_model():
    m = SDE('two-node shot noise')
    for p in _NAMES:
        m = m.parameter(p, default=1.0, domain='positive')
    m = m.field('u').field('w').drift('u', '0').drift('w', '0')
    for i, x, cpl in ((1, 'u', {'u': 'k11', 'w': 'k12'}), (2, 'w', {'u': 'k12', 'w': 'k11'})):
        p, n = f'((g{i}+a{i}-c{i})/2)', f'((g{i}-a{i}+c{i})/2)'
        m = (m.noise(f'up{i}', Poisson(rate=f'({p} + c{i}*exp(-{x}))/s', size='s'), couples=cpl)
              .noise(f'dn{i}', Poisson(rate=f'(a{i}*exp({x}) + {n})/s', size='-s'), couples=cpl))
    return m.build()


def _action_model():
    """The same process by hand: the jump action, and drift rows written with
    the same cancelling terms (this route had the same defect)."""
    P1, P2 = '(k11*ut[i] + k12*wt[i])', '(k12*ut[i] + k11*wt[i])'
    jumps = ' + '.join(
        f'(((g{i}+a{i}-c{i})/2) + c{i}*exp(-{x}[i]))*(exp(s*{P}) - 1)'
        f' + (a{i}*exp({x}[i]) + ((g{i}-a{i}+c{i})/2))*(exp(-s*{P}) - 1)'
        for i, x, P in ((1, 'u', P1), (2, 'w', P2)))
    H1 = '((g1+a1-c1)/2 + c1*exp(-u[i]) - a1*exp(u[i]) - (g1-a1+c1)/2)'
    H2 = '((g2+a2-c2)/2 + c2*exp(-w[i]) - a2*exp(w[i]) - (g2-a2+c2)/2)'
    b = (TemporalModelBuilder('two-node shot noise (action)')
         .population('pop', size=1)
         .physical_field('u', population='pop')
         .physical_field('w', population='pop'))
    for p in _NAMES:
        b = b.parameter(p, default=1.0, domain='positive')
    return (b.set_action_text(f'sum(ut[i]*Dt*u[i] + wt[i]*Dt*w[i] - ({jumps})/s for i in pop)')
            .mf_residual(lhs='(Dt+1)*u[i]', rhs=f'u[i] + k11*{H1} + k12*{H2}', population='pop')
            .mf_residual(lhs='(Dt+1)*w[i]', rhs=f'w[i] + k12*{H1} + k11*{H2}', population='pop')
            .build())


def _run(model, params, ext):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return dd.run(model, dd.Config(
            k=len(ext), max_ell=1, external_fields=ext, parameters=params,
            tau_grid=np.array([0.0]) if len(ext) == 1 else np.array([0.0, 1.0]),
            parallel=False, verbose=False, mf_dae_seed_box=_BOX))


def _tree_cross(params, taus):
    """<u(0) w(tau)> of the linearized (Kramers-Moyal) process."""
    a = np.array([params['a1'], params['a2']])
    c = np.array([params['c1'], params['c2']])
    g = np.array([params['g1'], params['g2']])
    K = np.array([[params['k11'], params['k12']], [params['k12'], params['k11']]])
    A = -K @ np.diag(a + c)
    D = params['s'] * K @ np.diag(g + a + c) @ K
    S = solve_continuous_lyapunov(A, -D)
    return np.array([(S @ expm(A.T * t))[0, 1] for t in taus])


def _sde_residual(params):
    model = _sde_model()
    pn = MF._numpy_params(model, params)
    sv = MF._state_variables(model)
    sl = MF._state_slices(sv)
    R, _ = MF._build_residual(model, pn, MF._make_phi_callables(model, pn),
                              MF._index_sets(model), sv, sl)
    return R


@pytest.mark.parametrize('params', [_P, _P_SHIFTED], ids=['split', 'other_split'])
def test_the_residual_at_the_exact_root_is_round_off_not_zero(params):
    """The premise: the rows cancel to round-off only (else the test is void)."""
    R = _sde_residual(params)
    m = np.empty(2)
    f = R(np.zeros(2), m)
    assert np.any(f != 0)
    assert np.all(np.abs(f) <= 16 * np.finfo(float).eps * m)


@pytest.mark.parametrize('params', [_P, _P_SHIFTED], ids=['split', 'other_split'])
def test_sde_front_end_finds_the_origin_and_runs_one_loop(params):
    r = _run(_sde_model(), params, [('du', 1), ('dw', 1)])
    assert r['mf_values'] == {'ustar': [0.0], 'wstar': [0.0]}
    C = r['C_tau_by_ell']
    assert all(np.all(np.isfinite(np.asarray(v))) for v in C.values())
    np.testing.assert_allclose(np.real(C[0]), _tree_cross(params, [0.0, 1.0]), rtol=1e-10)
    assert np.all(np.abs(np.real(C[1])) > 0)


@pytest.mark.parametrize('params', [_P, _P_SHIFTED], ids=['split', 'other_split'])
def test_action_text_route_agrees_with_the_sde_front_end(params):
    for ext in ([('dw', 1)], [('du', 1), ('dw', 1)]):
        a = _run(_sde_model(), params, ext)
        b = _run(_action_model(), params, ext)
        assert b['mf_values'] == {'ustar': [0.0], 'wstar': [0.0]}
        for ell in (0, 1):
            np.testing.assert_allclose(np.real(b['C_tau_by_ell'][ell]),
                                       np.real(a['C_tau_by_ell'][ell]), rtol=1e-10, atol=1e-15)


# ── the acceptance itself ───────────────────────────────────────────

def _rows_R(rows):
    """``R(x, m=None)`` of rows ``rows(x) -> [(lhs, [rhs terms]), ...]``; ``m`` (when
    given) is filled with the term sizes ``|lhs| + sum |term|``, as the package's rows."""
    def R(x, m=None):
        rs = rows(np.asarray(x, dtype=float))
        if m is not None:
            m[:] = [abs(l) + sum(abs(t) for t in ts) for l, ts in rs]
        return np.array([l - sum(ts) for l, ts in rs])
    return R


def _cancelling(offset=0.0):
    """x = x + (g + a - c)/2 + c e^{-x} - a e^{x} - (g - a + c)/2 + offset (the
    builder's ``x = x + f(x)`` form; the root is 0 for offset 0)."""
    a, c, g = 10.185708437929115, 0.005351009345574485, 12.712463325126727
    return _rows_R(lambda x: [(x[0], [x[0], (g + a - c) / 2, c * np.exp(-x[0]),
                                      -a * np.exp(x[0]), -(g - a + c) / 2, offset])])


def test_an_exact_zero_at_round_off_is_accepted_without_a_scale():
    R = _cancelling()
    assert R(np.zeros(1))[0] != 0
    hit = MF._refine_root(R, np.zeros(1), np.zeros(1))
    assert hit is not None and hit[0][0] == 0.0


def test_hybr_point_keeps_a_round_off_zero_it_reports_as_stalled():
    x = MF._hybr_point(_cancelling(), np.array([0.03]))
    assert x is not None and abs(x[0]) <= 1e-14


@pytest.mark.parametrize('offset', [1e-12, 1e-9])
def test_a_residual_above_round_off_at_zero_is_still_refused(offset):
    """0 is NOT a root of the offset rows (their root is offset/(a + c) to
    first order): accepting round-off must not accept 0, whose residual is 10x
    and more above it; refinement from 0 may only return the true root."""
    a, c = 10.185708437929115, 0.005351009345574485
    hit = MF._refine_root(_cancelling(offset), np.zeros(1), np.zeros(1))
    assert hit is None or (hit[0][0] != 0 and hit[0][0] == pytest.approx(
        offset / (a + c), rel=1e-6, abs=10 * hit[1][0]))


def test_a_shifted_root_with_large_cancelling_terms():
    """Root v* = 0.3 of rows whose terms are ~1e3 times the drift: found to
    round-off, and 0 (a residual of ~1e-1 there) is not accepted as a root."""
    a, c, L = 2.7, 1.9, 4.0e3
    v0 = 0.3

    def terms(x):
        y = x - v0
        return [x, L + (a - c), c * np.exp(-y), -a * np.exp(y), -L]

    R = _rows_R(lambda x: [(x[0], terms(x[0]))])
    hit = MF._refine_root(R, np.array([v0 + 1e-6]), np.array([v0]))
    assert hit is not None and hit[0][0] == pytest.approx(v0, rel=1e-11)
    hit = MF._refine_root(R, np.zeros(1), np.zeros(1))
    assert hit is None or hit[0][0] == pytest.approx(v0, rel=1e-11)


# ── through the solver: a zero next to a nonzero field, and no tail roots ──

def _solve(drifts, params, stability=False):
    from daedalus.api.noise import Gaussian
    m = SDE('mf probe')
    for p in params:
        m = m.parameter(p, default=1.0)
    m = m.parameter('D', default=0.1, domain='positive')
    for f in drifts:
        m = m.field(f)
    for f, d in drifts.items():
        m = m.drift(f, d)
    m = m.noise('n', Gaussian(var='2*D'), couples={f: 1 for f in drifts})
    if stability:
        m = m.stability_analysis(True)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return MF.solve_mean_field_dae(m.build(), dict(params, D=0.1))


_H = '((g+a-c)/2 + c*exp(-(X)) - a*exp(X) - (g-a+c)/2)'     # root X = 0, to round-off only
_HP = dict(a=10.185708437929115, c=0.005351009345574485, g=12.712463325126727)


def test_a_round_off_zero_next_to_a_nonzero_field():
    """u* = 0 by a row that cancels to round-off, w* = 2: the u row's linear scale
    is round-off (its converged points are ~1e-17, not all exactly 0)."""
    r = _solve({'u': _H.replace('X', 'u'), 'w': '-(w - 2) + u/2'}, _HP)
    assert r['mf_values']['ustar'] == [0.0]
    assert r['mf_values']['wstar'][0] == pytest.approx(2.0, rel=1e-14)
    assert len(r['mf_all_roots']) == 1


def test_a_round_off_zero_among_other_roots():
    """x (x - 1)(x + 2) with the x factor written as cancelling terms: all three roots."""
    r = _solve({'x': f'(x - 1)*(x + 2)*{_H.replace("X", "x")}/(a + c)'}, _HP)
    roots = sorted(d['values']['xstar'][0] for d in r['mf_all_roots'])
    assert roots[0] == pytest.approx(-2.0, rel=1e-12)
    assert roots[1] == 0.0
    assert roots[2] == pytest.approx(1.0, rel=1e-12)


def test_no_flat_tail_point_displaces_the_stable_root():
    """A bistable drift with a Gaussian tail: far out, ``x = x + f(x)`` is at the
    round-off of |x| although f has no root there.  Accepting a round-off residual
    must not admit such points (one, at x ~ -5.7, was selected as stable)."""
    r = _solve({'x': '-(x + 1.301)*(x - 0.613)*exp(-(x - 0.469)^2)'}, {}, stability=True)
    assert r['mf_values']['xstar'][0] == pytest.approx(0.613, rel=1e-12)
    roots = sorted(d['values']['xstar'][0] for d in r['mf_all_roots'])
    assert roots == [pytest.approx(-1.301, rel=1e-12), pytest.approx(0.613, rel=1e-12)]
