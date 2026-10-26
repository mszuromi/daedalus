"""Switch-on transients (non-stationary stage 1): ``compute_cumulants(t_min=..., times=...)``.

The fluctuations about the mean-field saddle start at zero at the absolute time ``t_min``; every vertex-time
integral has the lower bound ``t_min`` and the cumulant is a function of the absolute external times.  Computed on
the state-space Phase J route only (a finite first segment from the empty state).

References, none sharing code with Daedalus:
* linear models: the closed-form time-dependent Lyapunov solution
  ``Sigma(s) = Sigma_inf - e^{As} (Sigma_inf - C0) e^{A^T s}``, ``C(t1, t2) = Sigma(t1 - t_min) e^{A^T (t2 - t1)}``;
* nonlinear models: the exact perturbative moment hierarchy of the Ito generator started at ``X(t_min) = saddle``
  (``tests/_ito_moments.py``, vendored from sft-wick, BSD-3), extended here to multi-time moments by freezing the
  earlier values as extra coordinates with no drift and no noise (the Markov property: ``E[f(X_s) X_t^a] =
  E[f(X_s) E[X_t^a | X_s]]``), and turned into joint cumulants by the set-partition formula, order by order in the
  coupling.
"""
import math

import numpy as np
import pytest

pytest.importorskip('sage.all')
from api.theory import TemporalTheoryBuilder                      # noqa: E402
import api.compute as AC                                          # noqa: E402
import tests._ito_moments as im                                   # noqa: E402
from engine.integration.time_domain import final_integral as FI   # noqa: E402
from engine.integration.time_domain import statespace as SS       # noqa: E402


# ── the multi-time moment oracle ─────────────────────────────────────────────

def _augment(sde, n_frozen):
    """The process with ``n_frozen`` extra coordinates that neither drift nor diffuse."""
    z = (0,) * n_frozen
    aug = im.PolySDE(D=sde.D + n_frozen, n_tags=sde.n_tags)
    aug.drift = [(i, c, tuple(p) + z, tag) for (i, c, p, tag) in sde.drift]
    aug.diffusion = [(i, j, c, tuple(p) + z, tag) for (i, j, c, p, tag) in sde.diffusion]
    return aug


def multi_time_moments(sde, comps, times, tags):
    """``{tag: coefficient of E[prod_j X_{comps[j]}(times[j])]}`` for ``X(0) = 0`` (times >= 0, any order, ties
    allowed).  Phase ``j`` runs from the ``j``-th to the ``(j+1)``-th time with the earlier values frozen; its
    initial moments are the previous phase's, with the newest frozen coordinate folded back into ``X``."""
    order = sorted(range(len(times)), key=lambda j: times[j])
    cs = [comps[j] for j in order]
    ts = [float(times[j]) for j in order]
    D0 = sde.D
    aug = [_augment(sde, j) for j in range(len(ts))]

    def phase(j, nodes):
        req = []
        im.solve(aug[j], nodes, [0.0], initial=lambda a, t: req.append((a, t)) or 0.0)
        req = list(dict.fromkeys(req))
        if j == 0:
            zero = ((0,) * D0, (0,) * sde.n_tags)
            init = {nd: float(nd == zero) for nd in req}
        else:
            prev = []
            for (alpha, tag) in req:
                x, y = list(alpha[:D0]), list(alpha[D0:])
                x[cs[j - 1]] += y.pop()
                prev.append((tuple(x + y), tag))
            pv = phase(j - 1, list(dict.fromkeys(prev)))
            init = {nd: pv[pn] for nd, pn in zip(req, prev)}
        L = ts[j] - (0.0 if j == 0 else ts[j - 1])
        out = im.solve(aug[j], nodes, [L], initial=lambda a, t: init[(a, t)])
        return {nd: float(v[0]) for nd, v in out.items()}

    x = [0] * D0
    x[cs[-1]] += 1
    tgt = tuple(x) + (1,) * (len(ts) - 1)
    vals = phase(len(ts) - 1, [(tgt, tuple(t)) for t in tags])
    return {tuple(t): vals[(tgt, tuple(t))] for t in tags}


def _partitions(items):
    if not items:
        yield []
        return
    first, rest = items[0], items[1:]
    for part in _partitions(rest):
        for i in range(len(part)):
            yield part[:i] + [[first] + part[i]] + part[i + 1:]
        yield [[first]] + part


def cumulant_series(sde, comps, times, order):
    """The joint cumulant of ``X_{comps[j]}(times[j])`` as coefficients of powers ``0..order`` of the single
    bookkeeping parameter (``n_tags = 1``)."""
    tags = [(q,) for q in range(order + 1)]
    cache = {}

    def mom(block):
        if block not in cache:
            m = multi_time_moments(sde, [comps[b] for b in block], [times[b] for b in block], tags)
            cache[block] = np.array([m[(q,)] for q in range(order + 1)])
        return cache[block]

    out = np.zeros(order + 1)
    for part in _partitions(list(range(len(comps)))):
        ser = np.zeros(order + 1)
        ser[0] = 1.0
        for B in part:
            ser = np.convolve(ser, mom(tuple(sorted(B))))[:order + 1]
        out += (-1) ** (len(part) - 1) * math.factorial(len(part) - 1) * ser
    return out


# ── inline models ─────────────────────────────────────────────────────────────

def _ou_cubic():
    return (TemporalTheoryBuilder('ou cubic (switch-on)', n_populations=0)
            .physical_field('x', indexed=True)
            .parameter('mu', default=1.0, domain='positive').parameter('eps', default=0.02, domain='positive')
            .parameter('D', default=1.0, domain='positive')
            .set_action_text('xt*((Dt+mu)*x + eps*x^3) - D*xt^2')
            .equation(lhs='(Dt+mu)*x', rhs='-eps*x^3')
            .stability_analysis(True).build())


def _quadratic():
    return (TemporalTheoryBuilder('quadratic drift (switch-on)', n_populations=0)
            .physical_field('x', indexed=True)
            .parameter('mu', default=1.0, domain='positive').parameter('q', default=0.1, domain='real')
            .parameter('D', default=1.0, domain='positive')
            .set_action_text('xt*((Dt+mu)*x + q*x^2) - D*xt^2')
            .equation(lhs='(Dt+mu)*x', rhs='-q*x^2')
            .stability_analysis(True).build())


def _colored_embedded():
    """OU + eps x^3 driven by an OU noise y (the Markov embedding of exponentially correlated noise)."""
    return (TemporalTheoryBuilder('colored OU embedded (switch-on)', n_populations=0)
            .physical_field('x', indexed=True).physical_field('y', indexed=True)
            .parameter('mu', default=1.0, domain='positive').parameter('gam', default=0.5, domain='positive')
            .parameter('eps', default=0.05, domain='positive').parameter('D', default=1.0, domain='positive')
            .set_action_text('xt*((Dt+mu)*x + eps*x^3 - y) + yt*((Dt+gam)*y) - D*yt^2')
            .equation(lhs='(Dt+mu)*x', rhs='-eps*x^3 + y')
            .equation(lhs='(Dt+gam)*y', rhs='0')
            .stability_analysis(True).build())


def _coupled_cubic():
    """Two noisy fields with non-reciprocal coupling (no Boltzmann solution)."""
    return (TemporalTheoryBuilder('coupled cubic (switch-on)', n_populations=0)
            .physical_field('x', indexed=True).physical_field('y', indexed=True)
            .parameter('mu1', default=1.0, domain='positive').parameter('mu2', default=1.7, domain='positive')
            .parameter('eps', default=0.05, domain='positive')
            .parameter('J1', default=0.5, domain='real').parameter('J2', default=0.3, domain='real')
            .parameter('D1', default=1.0, domain='positive').parameter('D2', default=0.6, domain='positive')
            .set_action_text('xt*((Dt+mu1)*x + eps*x^3 - J1*y) - D1*xt^2 + yt*((Dt+mu2)*y - J2*x) - D2*yt^2')
            .equation(lhs='(Dt+mu1)*x', rhs='-eps*x^3 + J1*y')
            .equation(lhs='(Dt+mu2)*y', rhs='J2*x')
            .stability_analysis(True).build())


def _run(model, k, max_ell, legs, par, t_min, times, **kw):
    return AC.compute_cumulants(model, k=k, max_ell=max_ell, external_fields=legs, parameters=par, t_min=t_min,
                                times=np.asarray(times, dtype=float), use_cache=False, parallel=False,
                                verbose=False, **kw)


def _close(got, want, rtol, atol=0.0):
    assert abs(got - want) <= rtol * abs(want) + atol, (got, want, abs(got - want) / max(abs(want), 1e-300))


X, Y = ('dx', 1), ('dy', 1)
OU_P = {'mu': 1.3, 'eps': 0.05, 'D': 0.7}
COL_P = {'mu': 1.2, 'gam': 0.45, 'eps': 0.04, 'D': 0.9}
CPL_P = {'mu1': 1.0, 'mu2': 1.7, 'J1': 0.5, 'J2': 0.3, 'D1': 1.0, 'D2': 0.6, 'eps': 0.05}


def _ou_sde(mu, D):
    sde = im.PolySDE(D=1, n_tags=1)
    sde.add_linear_drift(np.array([[-mu]]))
    sde.drift.append((0, -1.0, (3,), (1,)))                 # -eps x^3, tag = power of eps
    sde.add_constant_diffusion(np.array([[2 * D]]))
    return sde


# ── A: exact references ───────────────────────────────────────────────────────

def test_ou_tree_is_the_time_dependent_lyapunov_solution():
    mu, D, tm = OU_P['mu'], OU_P['D'], 0.4
    T = [[0.9, 0.9], [1.0, 2.5], [2.5, 1.0], [3.0, 3.0], [0.5, 7.0], [12.0, 12.4]]
    res = _run(_ou_cubic(), 2, 0, [X, X], OU_P, tm, T)
    assert res['C_tau'] is None and res['tau_grid'] is None and res['t_min'] == tm
    assert res['nonstationary']['protocol'] == 'switch_on'
    assert res['phase_j_engine']['ss_declined'] == 0 and res['phase_j_engine']['fallback_evals'] == 0
    for (t1, t2), got in zip(T, res['C_times']):
        want = D / mu * (math.exp(-mu * abs(t1 - t2)) - math.exp(-mu * (t1 + t2 - 2 * tm)))
        _close(got.real, want, 1e-13)
        assert got.imag == 0.0
    # the callable takes absolute times and agrees with the array
    assert res['total_C'](1.0, 2.5) == res['C_times'][1]


def test_ou_cubic_one_loop_against_the_moment_hierarchy():
    mu, D, eps, tm = OU_P['mu'], OU_P['D'], OU_P['eps'], -0.5
    T = [[0.0, 0.0], [0.0, 0.7], [0.3, -0.2], [2.0, 2.0], [-0.4, 1.5]]
    res = _run(_ou_cubic(), 2, 1, [X, X], OU_P, tm, T)
    sde = _ou_sde(mu, D)
    for i, (t1, t2) in enumerate(T):
        m = multi_time_moments(sde, [0, 0], [t1 - tm, t2 - tm], [(0,), (1,)])     # <x> = 0: moment = cumulant
        _close(res['C_times_by_ell'][0][i].real, m[(0,)], 1e-13)
        _close(res['C_times_by_ell'][1][i].real, eps * m[(1,)], 1e-10)
    assert np.allclose(res['C_times'], res['C_times_by_ell'][0] + res['C_times_by_ell'][1], rtol=0, atol=0)


def test_fluctuations_vanish_at_t_min():
    """``t1 = t2 = t_min``: zero (no initial-state term), at every order; a k = 3 leg at t_min too."""
    res = _run(_ou_cubic(), 2, 1, [X, X], OU_P, 0.25, [[0.25, 0.25], [0.25, 3.0]])
    for ell in (0, 1):
        assert np.all(res['C_times_by_ell'][ell] == 0)
    r3 = _run(_quadratic(), 3, 1, [X, X, X], {'mu': 1.1, 'q': 0.3, 'D': 0.8}, -0.3, [[-0.3, 1.0, 1.0]])
    assert r3['C_times'][0] == 0


QP = {'mu': 1.1, 'q': 0.3, 'D': 0.8}


def _quad_sde():
    sde = im.PolySDE(D=1, n_tags=1)
    sde.add_linear_drift(np.array([[-QP['mu']]]))
    sde.drift.append((0, -1.0, (2,), (1,)))                 # -q x^2, tag = power of q
    sde.add_constant_diffusion(np.array([[2 * QP['D']]]))
    return sde


def test_quadratic_mean_one_loop_transient():
    """k = 1: the one-loop mean <x>(t) relaxes from 0 (the saddle) at t_min to its stationary value (B)."""
    tm, T = -0.3, [-0.3, 0.0, 0.5, 2.0, 8.0, 30.0]
    res = _run(_quadratic(), 1, 1, [X], QP, tm, T)
    assert res['mf_values']['xstar'] == [0.0]
    sde = _quad_sde()
    for i, t in enumerate(T):
        want = QP['q'] * cumulant_series(sde, [0], [t - tm], 1)[1]
        assert res['C_times_by_ell'][0][i] == 0
        _close(res['C_times_by_ell'][1][i].real, want, 1e-10, 1e-16)
    stat = AC.compute_cumulants(_quadratic(), k=1, max_ell=1, external_fields=[X], parameters=QP,
                                tau_grid=np.array([0.0]), use_cache=False, parallel=False, verbose=False)
    gap = np.abs(res['C_times'][1:] - stat['C_tau'][0])
    assert np.all(np.diff(gap) < 0) and gap[-1] < 1e-12 * abs(stat['C_tau'][0])


@pytest.mark.parametrize('ell', [0, 1])
def test_quadratic_third_cumulant(ell):
    """k = 3, tree (order q) and one loop (order q^3), several time orders and ties."""
    tm = -0.3
    T = np.array([[0.2, 0.2, 0.2], [0.0, 0.7, 1.5], [1.5, 0.0, 0.7], [0.5, 0.5, 2.0]])
    res = _run(_quadratic(), 3, ell, [X, X, X], QP, tm, T)
    sde = _quad_sde()
    for i, row in enumerate(T):
        ser = cumulant_series(sde, [0, 0, 0], list(row - tm), 3)
        _close(res['C_times_by_ell'][0][i].real, QP['q'] * ser[1], 1e-12)
        if ell:
            _close(res['C_times_by_ell'][1][i].real, QP['q'] ** 3 * ser[3], 1e-10)


def test_quadratic_two_point_one_loop_subtracts_the_means():
    tm = -0.3
    T = np.array([[0.0, 0.0], [0.5, 1.5], [3.0, 0.2]])
    res = _run(_quadratic(), 2, 1, [X, X], QP, tm, T)
    sde = _quad_sde()
    for i, row in enumerate(T):
        ser = cumulant_series(sde, [0, 0], list(row - tm), 2)
        _close(res['C_times_by_ell'][0][i].real, ser[0], 1e-13)
        _close(res['C_times_by_ell'][1][i].real, QP['q'] ** 2 * ser[2], 1e-10)


def _col_exact(i, j, t1, t2, tm, C0=None):
    """Colored OU (linear part): A = [[-mu, 1], [0, -gam]], Q = diag(0, 2D); closed forms, no expm."""
    mu, gam, D = COL_P['mu'], COL_P['gam'], COL_P['D']

    def eA(t):
        return np.array([[math.exp(-mu * t), (math.exp(-gam * t) - math.exp(-mu * t)) / (mu - gam)],
                         [0.0, math.exp(-gam * t)]])
    syy = D / gam
    sxy = syy / (mu + gam)
    Sinf = np.array([[sxy / mu, sxy], [sxy, syy]])
    C0 = np.zeros((2, 2)) if C0 is None else C0
    a, b = t1 - tm, t2 - tm
    s = min(a, b)
    S = Sinf - eA(s) @ (Sinf - C0) @ eA(s).T
    return (S @ eA(b - a).T)[i, j] if a <= b else (eA(a - b) @ S)[i, j]


@pytest.mark.parametrize('legs', [(X, X), (X, Y), (Y, X)])
def test_colored_ou_markov_embedded(legs):
    tm = 0.0
    T = [[0.5, 0.5], [0.3, 2.0], [2.0, 0.3], [4.0, 4.0]]
    res = _run(_colored_embedded(), 2, 1, list(legs), COL_P, tm, T)
    i, j = (0 if legs[0] == X else 1), (0 if legs[1] == X else 1)
    A = np.array([[-COL_P['mu'], 1.0], [0.0, -COL_P['gam']]])
    sde = im.PolySDE(D=2, n_tags=1)
    sde.add_linear_drift(A)
    sde.drift.append((0, -1.0, (3, 0), (1,)))
    sde.add_constant_diffusion(np.diag([0.0, 2 * COL_P['D']]))
    for p, (t1, t2) in enumerate(T):
        _close(res['C_times_by_ell'][0][p].real, _col_exact(i, j, t1, t2, tm), 1e-13)
        m = multi_time_moments(sde, [i, j], [t1 - tm, t2 - tm], [(1,)])
        _close(res['C_times_by_ell'][1][p].real, COL_P['eps'] * m[(1,)], 1e-10)


def test_repeated_pole_jordan_block():
    """mu = gam: a repeated pole (Jordan block; the pole engine's deformation is bypassed): still exact."""
    mu = gam = 0.8
    D, eps = 0.9, 0.04
    par = {'mu': mu, 'gam': gam, 'eps': eps, 'D': D}

    def eA(t):
        e = math.exp(-mu * t)
        return np.array([[e, t * e], [0.0, e]])
    syy = D / gam
    sxy = syy / (mu + gam)
    Sinf = np.array([[sxy / mu, sxy], [sxy, syy]])
    T = [[0.5, 0.5], [0.3, 2.0], [2.0, 0.3], [4.0, 4.0]]
    res = _run(_colored_embedded(), 2, 1, [X, Y], par, 0.0, T)
    assert res['phase_j_engine']['pole_deformation'] == 'bypassed'
    sde = im.PolySDE(D=2, n_tags=1)
    sde.add_linear_drift(np.array([[-mu, 1.0], [0.0, -gam]]))
    sde.drift.append((0, -1.0, (3, 0), (1,)))
    sde.add_constant_diffusion(np.diag([0.0, 2 * D]))
    for p, (t1, t2) in enumerate(T):
        s_ = min(t1, t2)
        S = Sinf - eA(s_) @ Sinf @ eA(s_).T
        want = (S @ eA(t2 - t1).T)[0, 1] if t1 <= t2 else (eA(t1 - t2) @ S)[0, 1]
        _close(res['C_times_by_ell'][0][p].real, want, 1e-13)
        m = multi_time_moments(sde, [0, 1], [t1, t2], [(1,)])
        _close(res['C_times_by_ell'][1][p].real, eps * m[(1,)], 1e-10)


def test_coupled_two_field_cross_legs_one_loop():
    tm = 1.0
    T = np.array([[1.5, 1.5], [1.2, 3.0], [3.0, 1.2], [6.0, 6.5]])
    res = _run(_coupled_cubic(), 2, 1, [X, Y], CPL_P, tm, T)
    sde = im.PolySDE(D=2, n_tags=1)
    sde.add_linear_drift(np.array([[-CPL_P['mu1'], CPL_P['J1']], [CPL_P['J2'], -CPL_P['mu2']]]))
    sde.drift.append((0, -1.0, (3, 0), (1,)))
    sde.add_constant_diffusion(np.diag([2 * CPL_P['D1'], 2 * CPL_P['D2']]))
    for p, (t1, t2) in enumerate(T):
        m = multi_time_moments(sde, [0, 1], [t1 - tm, t2 - tm], [(0,), (1,)])
        _close(res['C_times_by_ell'][0][p].real, m[(0,)], 1e-13)
        _close(res['C_times_by_ell'][1][p].real, CPL_P['eps'] * m[(1,)], 1e-10)


def test_initial_covariance_tree_term():
    C0 = np.array([[0.3, -0.1], [-0.1, 0.5]])
    T = [[0.0, 0.0], [0.5, 0.5], [0.3, 2.0], [2.0, 0.3], [4.0, 4.0]]
    res = _run(_colored_embedded(), 2, 0, [X, Y], COL_P, 0.0, T, initial_covariance=C0)
    for (t1, t2), got in zip(T, res['C_times']):
        _close(got.real, _col_exact(0, 1, t1, t2, 0.0, C0), 1e-13)
    assert res['C_times'][0].real == pytest.approx(C0[0, 1], abs=1e-15)
    assert res['nonstationary']['initial_term'] is not None
    assert res['total_C'](0.3, 2.0) == pytest.approx(res['C_times'][2], rel=1e-15)
    # the ell = 0 Phase J record's total carries the term too (review round 2)
    assert res['phase_j_by_ell'][0]['total_C'](0.3, 2.0) == pytest.approx(res['C_times'][2], rel=1e-15)
    assert res['phase_j_by_ell'][0]['total_C_batch']([(0.3, 2.0)]) == [res['phase_j_by_ell'][0]['total_C'](0.3, 2.0)]


# ── B / C: the stationary limit and the stationary path ──────────────────────

def test_t_min_to_minus_infinity_is_the_stationary_result():
    stat = AC.compute_cumulants(_colored_embedded(), k=2, max_ell=1, external_fields=[X, Y], parameters=COL_P,
                                tau_grid=np.array([0.8, -1.3]), use_cache=False, parallel=False, verbose=False)
    assert stat.get('t_min') is None and 'C_times' not in stat
    res = _run(_colored_embedded(), 2, 1, [X, Y], COL_P, -80.0, [[0.0, 0.0], [0.0, 0.8], [0.0, -1.3]])
    for ell in (0, 1):
        # tau = 0: the stationary callable at an exact tie (C_tau's tau = 0 point is the Ito nudge 0-)
        _close(res['C_times_by_ell'][ell][0].real, stat['phase_j_by_ell'][ell]['total_C'](0.0, 0.0).real, 1e-13)
        for p in (1, 2):
            _close(res['C_times_by_ell'][ell][p].real, stat['C_tau_by_ell'][ell][p - 1].real, 1e-13)


def test_relaxation_to_the_stationary_value_is_monotone():
    s = np.array([1.0, 2.0, 4.0, 8.0, 16.0, 32.0])
    res = _run(_ou_cubic(), 2, 1, [X, X], OU_P, 0.0, s)
    stat = AC.compute_cumulants(_ou_cubic(), k=2, max_ell=1, external_fields=[X, X], parameters=OU_P,
                                tau_grid=np.array([0.5]), use_cache=False, parallel=False, verbose=False)
    for ell in (0, 1):
        ref = stat['phase_j_by_ell'][ell]['total_C'](0.0, 0.0).real
        gap = np.abs(res['C_times_by_ell'][ell].real - ref) / abs(ref)
        assert np.all(np.diff(gap[:4]) < 0), gap         # s >= 16: at rounding
        assert np.all(gap[4:] < 1e-13), gap
    # the tree gap is exactly e^{-2 mu s}
    np.testing.assert_allclose(np.abs(res['C_times_by_ell'][0].real - OU_P['D'] / OU_P['mu'])
                               / (OU_P['D'] / OU_P['mu']), np.exp(-2 * OU_P['mu'] * s), rtol=1e-9, atol=2e-15)


def test_scope_is_thread_local_and_restores():
    assert FI.PHASE_J_T_MIN == float('-inf') and not SS.switch_on_active()
    _run(_ou_cubic(), 2, 0, [X, X], OU_P, 0.0, [[1.0, 1.0]])
    assert FI.PHASE_J_T_MIN == float('-inf') and not SS.switch_on_active()
    with pytest.raises(ValueError, match='inside'):
        with SS.switch_on_scope(2.0):
            # the pole engine's module bound is never touched
            assert FI.PHASE_J_T_MIN == float('-inf') and SS._switch_on_t_min() == 2.0
            with SS.switch_on_scope(2.0):            # same bound: nests
                assert SS._switch_on_t_min() == 2.0
            assert SS._switch_on_t_min() == 2.0
            with pytest.raises(ValueError, match='one switch-on time'):
                with SS.switch_on_scope(3.0):
                    pass
            raise ValueError('inside')
    assert FI.PHASE_J_T_MIN == float('-inf') and not SS.switch_on_active()


def test_threads_do_not_leak_the_bound():
    """Review finding: scopes closing out of order in threads used to leave the bound set, and a later stationary
    run returned zeros.  Callables of a switch-on result evaluated from 8 threads, while another thread runs a
    stationary evaluation: the stationary values are untouched and nothing leaks."""
    from concurrent.futures import ThreadPoolExecutor
    res = _run(_ou_cubic(), 2, 0, [X, X], OU_P, 0.0, [[1.0, 1.0]])
    stat = AC.compute_cumulants(_ou_cubic(), k=2, max_ell=0, external_fields=[X, X], parameters=OU_P,
                                tau_grid=np.array([0.5, 1.0]), use_cache=False, parallel=False, verbose=False)
    st_fn = stat['total_C_by_ell'][0]
    want = [complex(st_fn(0.0, 0.5)), complex(st_fn(0.0, 1.0))]

    def job(i):
        if i % 10 == 0:
            return ('stat', complex(st_fn(0.0, 0.5)))
        return ('so', complex(res['total_C'](1.0, 1.0)))
    with ThreadPoolExecutor(8) as ex:
        out = list(ex.map(job, range(400)))
    assert {v for kind, v in out if kind == 'so'} == {res['C_times'][0]}
    assert {v for kind, v in out if kind == 'stat'} == {want[0]}
    assert FI.PHASE_J_T_MIN == float('-inf') and not SS.switch_on_active()
    again = AC.compute_cumulants(_ou_cubic(), k=2, max_ell=0, external_fields=[X, X], parameters=OU_P,
                                 tau_grid=np.array([0.5, 1.0]), use_cache=False, parallel=False, verbose=False)
    np.testing.assert_array_equal(again['C_tau'], stat['C_tau'])
    assert again['C_tau'][0].real == pytest.approx(OU_P['D'] / OU_P['mu'] * math.exp(-OU_P['mu'] * 0.5), rel=1e-14)


def test_every_stored_callable_is_the_switch_on_one():
    """Review finding: the closures kept in ``phase_j_by_ell`` used to evaluate the stationary integrals."""
    from api import me_output
    from api.save import save_npz
    T = [[1.0, 1.0], [1.0, 2.0]]
    res = _run(_ou_cubic(), 2, 1, [X, X], OU_P, 0.0, T)
    for ell in (0, 1):
        td = res['phase_j_by_ell'][ell]
        for p, (t1, t2) in enumerate(T):
            assert complex(td['total_C'](t1, t2)) == res['C_times_by_ell'][ell][p]
            assert complex(res['total_C_by_ell'][ell](t1, t2)) == res['C_times_by_ell'][ell][p]
            assert sum(complex(g['contribution'](t1, t2)) for g in td['groups']) == pytest.approx(
                res['C_times_by_ell'][ell][p], rel=1e-14, abs=1e-300)
        assert td['total_C_batch']([(1.0, 2.0)], parallel=True) == [res['C_times_by_ell'][ell][1]]
        vals, _loops = td['eval_per_diagram_batch']([(1.0, 2.0)], parallel=True)
        assert complex(sum(vals[0])) == pytest.approx(res['C_times_by_ell'][ell][1], rel=1e-14)
    with pytest.raises(me_output.MEUnavailable, match='switch-on'):
        me_output.from_result(res)
    with pytest.raises(me_output.MEUnavailable, match='switch-on'):
        me_output.slice_me(res)
    with pytest.raises(NotImplementedError, match='C_times'):
        save_npz(res, 'never_written.npz')
    from api.report import generate_report
    with pytest.raises(NotImplementedError, match='C_times'):
        generate_report(_ou_cubic(), 2, output_pdf='never_written.pdf', result=res)
    from api.symbolic_out import tree_covariance_from_result, SymbolicOutError
    with pytest.raises(SymbolicOutError, match='switch-on'):
        tree_covariance_from_result(res, model=_ou_cubic())


# ── refusals: never silent, never the pole engine ─────────────────────────────

def test_pole_engine_refuses_t_min(monkeypatch):
    monkeypatch.setenv('DAEDALUS_PHASE_J_ENGINE', 'poles')
    with pytest.raises(SS.SwitchOnUnsupported, match='state-space route only'):
        _run(_ou_cubic(), 2, 0, [X, X], OU_P, 0.0, [[1.0, 1.0]])
    with pytest.raises(SS.SwitchOnUnsupported, match='engine_poles'):
        SS.switch_on_scope(0.0)


def _cgf_model(kernel):
    return (TemporalTheoryBuilder('ou noise kernel (switch-on)', n_populations=0)
            .physical_field('x', indexed=True)
            .parameter('mu', default=1.0, domain='positive').parameter('eps', default=0.05, domain='positive')
            .parameter('D', default=1.0, domain='positive').parameter('tauc', default=1.0, domain='positive')
            .declare_cgf_term('CXX', response_legs=['xt', 'xt'], order=2, coefficient='2*D/tauc', kernel=kernel)
            .set_action_text('xt*((Dt+mu)*x + eps*x^3)')
            .equation(lhs='(Dt+mu)*x', rhs='-eps*x^3')
            .stability_analysis(True).build())


def test_declared_exponential_noise_kernel_is_switched_on_with_its_embedding():
    """An exponential CGF kernel is Markov-embedded when the model is built (an auxiliary OU field): under t_min
    that auxiliary state starts at zero too, i.e. the noise is switched on with the system.  Same numbers as the
    explicit embedding (gam = 1/tauc, D_y = 2 D / tauc^2)."""
    mu, eps, D, tauc = 1.2, 0.04, 0.8, 1.0 / 0.45
    T = [[0.5, 0.5], [0.3, 2.0], [4.0, 4.0]]
    res = _run(_cgf_model('exp(-abs(tau)/tauc)'), 2, 1, [X, X], {'mu': mu, 'eps': eps, 'D': D, 'tauc': tauc},
               0.0, T)
    assert res['nonstationary']['embedded_noise_states']
    ref = _run(_colored_embedded(), 2, 1, [X, X], {'mu': mu, 'gam': 1 / tauc, 'eps': eps, 'D': 2 * D / tauc ** 2},
               0.0, T)
    for ell in (0, 1):
        np.testing.assert_allclose(res['C_times_by_ell'][ell], ref['C_times_by_ell'][ell], rtol=1e-12, atol=0)


def test_non_markov_noise_kernel_refuses_t_min():
    """A noise kernel without a realization (a Gaussian kernel: a noise-kernel box) is not on the state-space
    route: refused, never sent to the pole engine."""
    with pytest.raises(SS.SwitchOnUnsupported, match='noise_kernel'):
        _run(_cgf_model('exp(-tau^2/tauc^2)'), 2, 0, [X, X], {'mu': 1.0, 'eps': 0.05, 'D': 1.0, 'tauc': 1.0},
             0.0, [[1.0, 1.0]])
    assert FI.PHASE_J_T_MIN == float('-inf') and not SS.switch_on_active()


def test_delay_kernel_model_runs_under_t_min():
    """A delay kernel is computed with the Pade tree and loops (tests/test_delay_switch_on.py has the
    references); ``initial_covariance`` stays refused there."""
    from api.noise import Gaussian
    from api.sde import SDE
    model = (SDE('delayed OU (switch-on)').parameter('a', default=1.0).parameter('c', default=-0.5)
             .parameter('d', default=0.5, domain='positive').parameter('D', default=0.1).field('x')
             .drift('x', '-a*x + c*x(t - d)')
             .noise('xi', Gaussian(var='2*D'), couples={'x': 1}).build())
    kw = dict(k=2, max_ell=0, external_fields=[('x', 1), ('x', 1)], t_min=0.0, times=[[1.0, 1.0]],
              use_cache=False, parallel=False, verbose=False)
    r = AC.compute_cumulants(model, delay={'order': 4}, **kw)
    assert r['delay']['tree'] == 'pade' and np.isfinite(np.real(r['C_times'])).all()
    with pytest.raises(SS.SwitchOnUnsupported, match='initial_covariance'):
        AC.compute_cumulants(model, initial_covariance=np.eye(1), **kw)


@pytest.mark.parametrize('kw, exc, match', [
    (dict(t_min=0.0, times=[[1.0, -0.5]]), ValueError, 'after|>= t_min'),
    (dict(t_min=0.0), ValueError, 'needs times'),
    (dict(times=[[1.0, 1.0]]), ValueError, 'need t_min'),
    (dict(t_min=float('-inf'), times=[[1.0, 1.0]]), ValueError, 'finite'),
    (dict(t_min=0.0, times=[[1.0, 1.0, 1.0]]), ValueError, 'shape'),
    (dict(t_min=0.0, times=[[1.0, 1.0]], output_npz='x.npz'), ValueError, 'output_npz'),
    (dict(t_min=0.0, times=[[1.0, 1.0]], max_ell=1, initial_covariance=np.eye(1)), NotImplementedError, 'loops'),
    (dict(t_min=0.0, times=[[1.0, 1.0]], tau_grid=np.array([0.5])), ValueError, 'tau_grid'),
    (dict(t_min=0.0, times=[[1.0, 1.0]], initial_covariance=np.ones(2)), ValueError, 'square'),
])
def test_argument_refusals(kw, exc, match):
    kw = dict(kw)
    max_ell = kw.pop('max_ell', 0)
    with pytest.raises(exc, match=match):
        AC.compute_cumulants(_ou_cubic(), k=2, max_ell=max_ell, external_fields=[X, X], parameters=OU_P,
                             use_cache=False, parallel=False, verbose=False, **kw)


def test_retired_grouped_flag_only_warns_with_t_min():
    """Grouped Phase J is retired: ``use_grouped_phase_j=True`` with ``t_min`` is no longer refused, it
    warns (FutureWarning) and runs the per-diagram route, with the same numbers."""
    kw = dict(k=2, max_ell=0, external_fields=[X, X], parameters=OU_P, t_min=0.0,
              times=[[1.0, 1.0], [0.5, 2.0]], use_cache=False, parallel=False, verbose=False)
    ref = AC.compute_cumulants(_ou_cubic(), **kw)
    with pytest.warns(FutureWarning, match='use_grouped_phase_j is retired'):
        res = AC.compute_cumulants(_ou_cubic(), use_grouped_phase_j=True, **kw)
    assert np.array_equal(np.asarray(res['C_times']), np.asarray(ref['C_times']))
    assert res['config']['use_grouped_phase_j'] is False


def test_initial_state_needs_identity_response():
    """``R(0+) != I`` (here ``1/c``: a ``c Dt`` term) cannot carry an initial state of the fluctuations."""
    model = (TemporalTheoryBuilder('scaled time derivative (switch-on)', n_populations=0)
             .physical_field('x', indexed=True)
             .parameter('mu', default=1.0, domain='positive').parameter('c', default=2.0, domain='positive')
             .parameter('D', default=1.0, domain='positive')
             .set_action_text('xt*((c*Dt + mu)*x) - D*xt^2')
             .equation(lhs='(c*Dt + mu)*x', rhs='0')
             .stability_analysis(True).build())
    with pytest.raises(ValueError, match='identity'):
        _run(model, 2, 0, [X, X], {'mu': 1.0, 'c': 2.0, 'D': 1.0}, 0.0, [[1.0, 1.0]],
             initial_covariance=np.eye(1))


# ── the dd layer ─────────────────────────────────────────────────────────────

def test_dd_config_switch_on():
    import daedalus as dd
    with pytest.raises(ValueError, match='need t_min'):
        dd.Config(times=(0.0, 1.0, 3))
    late = dd.Config(k=2, external_fields=[('x', 1), ('x', 1)], parameters=OU_P)
    late.times = [[1.0, 1.0]]                     # set after construction: still refused
    with pytest.raises(ValueError, match='need t_min'):
        dd.run(_ou_cubic(), late)
    cfg = dd.Config(k=2, max_ell=1, external_fields=[('x', 1), ('x', 1)], parameters=OU_P,
                    t_min=0.0, times=(0.0, 4.0, 5), verbose=False)
    res = dd.run(_ou_cubic(), cfg)
    ref = _run(_ou_cubic(), 2, 1, [X, X], OU_P, 0.0, np.linspace(0.0, 4.0, 5))
    np.testing.assert_array_equal(res['times'], np.repeat(np.linspace(0.0, 4.0, 5)[:, None], 2, axis=1))
    np.testing.assert_allclose(res['C_times'], ref['C_times'], rtol=1e-15, atol=0)
    assert res['C_times'][0] == 0 and 'C_tau_slices' not in res
    with pytest.raises(NotImplementedError, match='C_times'):
        dd.plot_cumulant(res, cfg)
    with pytest.raises(NotImplementedError, match='C_times'):
        dd.cumulant(res)
