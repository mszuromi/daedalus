"""The exact master-equation oracle (daedalus/api/master_equation.py, ``dd.master_equation``).

Checks, each against an independent closed form:
  * immigration-death: Poisson law, every cumulant = lambda/mu, C(tau) = (lambda/mu) e^{-mu |tau|},
    the integrated correlation lambda/mu^2, direct vs Krylov ``e^{tau Q}``;
  * a two-species network with a joint immigration channel (correlated species) against the exact
    linear-noise (Lyapunov) solution -- exact for linear rates -- incl. lag correlations and the
    automatic box;
  * a stiff two-timescale network: Krylov vs the direct march and the Lyapunov result, with its
    own error estimate honest;
  * truncation diagnostics (a box that cuts the law; a box whose edge has vanishing rate; the
    refusals);
  * the Kramers-Moyal extrapolation (immigration-death: diffusion kappa_4 = 3 lambda/(2 mu));
  * the bridge: a Schlogl-type nonlinear birth-death process against the one-loop cumulants of
    ``compute_cumulants``, the error shrinking as the system size grows.
"""
import warnings

import numpy as np
import pytest
from scipy.linalg import expm, solve_continuous_lyapunov
from scipy.stats import poisson

import daedalus as dd
from daedalus.api.master_equation import (MasterEquation, MasterEquationError, TruncationWarning,
                                 Transition)


# ── immigration-death ──────────────────────────────────────────────────────────────────────

LAM, MU = 20.0, 1.0


def _imm_death(lam=LAM, **kw):
    kw.setdefault('box', (0, 100))
    return dd.master_equation([('lam', +1), ('mu*x', -1)], params=dict(lam=lam, mu=MU), **kw)


def test_immigration_death_poisson_law_and_cumulants():
    me = _imm_death()
    st = me.stationary()
    assert st.residual < 1e-12 and st.truncation_flux < 1e-30 and not st.truncated
    np.testing.assert_allclose(st.p, poisson.pmf(np.arange(101), LAM / MU), atol=1e-14, rtol=1e-10)
    cm = me.cumulants(4)
    for k in (1, 2, 3, 4):
        assert cm[k].shape == (1,) * k
        np.testing.assert_allclose(cm[k].ravel(), LAM / MU, rtol=1e-10)
    assert cm.cumulant('x', 'x', 'x') == pytest.approx(LAM / MU, rel=1e-10)


@pytest.mark.parametrize('method', ['direct', 'krylov', 'auto'])
def test_immigration_death_lag_correlation(method):
    me = _imm_death()
    taus = np.array([0.0, 0.1, 0.7, 2.0, 5.0, -1.3])
    cc = me.correlation(taus, method=method)
    np.testing.assert_allclose(cc.C[:, 0, 0], LAM / MU * np.exp(-MU * np.abs(taus)), rtol=1e-9)
    assert cc.krylov_error <= 1e-10
    assert me.integrated_correlation()[0, 0] == pytest.approx(LAM / MU ** 2, rel=1e-10)


def test_rate_string_forms_and_callable_agree():
    a = _imm_death().cumulants(2)[2]
    b = dd.master_equation([(LAM, 1), (lambda x: MU * x, -1)], box=(0, 100)).cumulants(2)[2]
    c = dd.master_equation([Transition('lam', (1,)), Transition('mu*x', (-1,))],
                           params=dict(lam=LAM, mu=MU), box=[(0, 100)]).cumulants(2)[2]
    np.testing.assert_allclose([a, b, c], LAM / MU, rtol=1e-10)


# ── two-species network against the exact linear-noise (Lyapunov) result ──────────────────────

NET = dict(lj=1.5, l1=6.0, m1=1.0, k=0.8, c=0.3, m2=0.5)
NET_TR = [('lj', (1, 1)), ('l1', (1, 0)), ('m1*x', (-1, 0)), ('k*x', (-1, 1)),
          ('c*y', (1, -1)), ('m2*y', (0, -1))]


def _net_exact(P):
    A = np.array([[-(P['m1'] + P['k']), P['c']], [P['k'], -(P['m2'] + P['c'])]])
    mu = -np.linalg.solve(A, np.array([P['l1'] + P['lj'], P['lj']]))
    D = np.zeros((2, 2))
    for r, j in [(P['lj'], (1, 1)), (P['l1'], (1, 0)), (P['m1'] * mu[0], (-1, 0)),
                 (P['k'] * mu[0], (-1, 1)), (P['c'] * mu[1], (1, -1)), (P['m2'] * mu[1], (0, -1))]:
        D += r * np.outer(j, j)
    return A, mu, solve_continuous_lyapunov(A, -D)


@pytest.fixture(scope='module')
def net():
    return dd.master_equation(NET_TR, fields=['x', 'y'], params=NET, box='auto', lower=0)


def test_two_species_linear_network_matches_lyapunov(net):
    A, mu, S = _net_exact(NET)
    lna = net.lna()
    np.testing.assert_allclose(lna['mean'], mu, rtol=1e-8)
    np.testing.assert_allclose(lna['cov'], S, rtol=1e-6)
    assert net.n_states > 500
    cm = net.cumulants(2)
    np.testing.assert_allclose(cm[1], mu, atol=1e-6)
    np.testing.assert_allclose(cm[2], S, atol=1e-6)
    assert S[0, 1] > 0.1                                       # genuinely correlated
    taus = np.array([0.0, 0.5, 2.0, 7.0, -0.5])
    ex = np.array([S @ expm(A.T * abs(t)) if t >= 0 else (S @ expm(A.T * abs(t))).T for t in taus])
    for method in ('direct', 'krylov'):
        cc = net.correlation(taus, method=method)
        np.testing.assert_allclose(cc.C, ex, atol=1e-6)
    np.testing.assert_allclose(net.integrated_correlation(), S @ np.linalg.inv(-A).T, atol=1e-5)


def test_two_species_all_cumulants_have_diagnostics(net):
    cm = net.cumulants(4)
    assert cm[3].shape == (2, 2, 2) and cm[4].shape == (2, 2, 2, 2)
    np.testing.assert_allclose(cm[3], cm[3].transpose(1, 0, 2))
    np.testing.assert_allclose(cm[4], cm[4].transpose(3, 1, 2, 0))
    assert cm.residual < 1e-10 and cm.truncation_flux < 1e-8 and cm.edge_mass < 1e-7


def test_dd_lazy_export_and_state_listing(net):
    assert dd.master_equation is not None and dd.MasterEquation is MasterEquation
    X = net.states()
    assert X.shape == (net.n_states, 2) and X.min() == 0.0
    Q = net.generator()
    assert abs(Q.sum(axis=1)).max() < 1e-12                  # rows of a generator sum to zero


# ── stiff two-timescale network ───────────────────────────────────────────────────────────────

def test_stiff_two_timescale_krylov_is_accurate_and_error_is_honest():
    P = dict(l=0.6, ms=0.05, kf=100.0, mf=100.0, c=0.02)
    tr = [('l', (1, 0)), ('ms*x', (-1, 0)), ('kf*x', (0, 1)), ('mf*y', (0, -1)), ('c*y', (1, -1))]
    me = dd.master_equation(tr, fields=['x', 'y'], params=P, box='auto', K=7.0, lower=0)
    A = np.array([[-P['ms'], P['c']], [P['kf'], -(P['mf'] + P['c'])]])
    mu = -np.linalg.solve(A, np.array([P['l'], 0]))
    D = np.zeros((2, 2))
    for r, j in [(P['l'], (1, 0)), (P['ms'] * mu[0], (-1, 0)), (P['kf'] * mu[0], (0, 1)),
                 (P['mf'] * mu[1], (0, -1)), (P['c'] * mu[1], (1, -1))]:
        D += r * np.outer(j, j)
    S = solve_continuous_lyapunov(A, -D)
    # fast mode ~100, slow mode ~0.03: the grid spans four decades
    taus = np.array([0.0, 1e-4, 1e-3, 1e-2, 0.05, 0.3, 2.0, 10.0, 40.0, 120.0])
    ex = np.array([S @ expm(A.T * t) for t in taus])
    cc = me.correlation(taus, method='krylov', krylov_tol=1e-10)
    scale = np.abs(ex).max(axis=(1, 2))
    err = np.abs(cc.C - ex).max(axis=(1, 2)) / scale
    trunc = me.stationary().truncation_flux
    assert err.max() < 1e-5 + 1e3 * trunc                    # box truncation dominates
    assert cc.krylov_error < 1e-9 and cc.krylov_dim > 60       # stiffness needed a long Krylov space
    # the Krylov dimension is the only difference from the direct march: compare on the short lags
    short = taus[taus <= 0.05]
    cd = me.correlation(short, method='direct')
    ck = me.correlation(short, method='krylov')
    np.testing.assert_allclose(ck.C, cd.C, rtol=1e-8, atol=1e-8 * np.abs(cd.C).max())
    # an under-resolved projection must say so
    with pytest.warns(Warning) as rec:
        bad = me.correlation(taus, method='krylov', m_max=20)
    assert bad.krylov_error > 1e-6
    assert any('Krylov' in str(w.message) for w in rec)


# ── truncation diagnostics and refusals ───────────────────────────────────────────────────────

def test_truncation_diagnostic_fires_when_box_too_small():
    me = _imm_death(box=(0, 12))
    with pytest.warns(TruncationWarning, match='cuts the stationary law'):
        st = me.stationary()
    assert st.truncated and st.truncation_flux > 1e-2 and st.edge_mass > 1e-2
    assert me.cumulants(2).truncation_flux == st.truncation_flux
    assert me.cumulants(1)[1][0] < LAM / MU                      # the law is visibly cut
    with pytest.raises(MasterEquationError, match='cuts the stationary law'):
        _imm_death(box=(0, 12), on_truncation='raise').stationary()
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        _imm_death(box=(0, 12), on_truncation='ignore').stationary()


def test_truncation_quiet_when_box_edge_has_zero_rate():
    # the death rate vanishes at 0: the lower edge costs nothing, the upper edge is 8+ sigma away
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        st = _imm_death(box=(0, 80)).stationary()
    assert st.truncation_flux < 1e-20 and st.edge_mass < 1e-20
    # a box that cuts below the mean: mass piles on the lower edge and is reported
    with pytest.warns(TruncationWarning):
        _imm_death(box=(26, 60)).stationary()


def test_pin_far_in_the_tail_does_not_poison_the_solve():
    # a box whose centre is 100 sigma from the bulk: the LU pin must follow the mean field
    for B in (120, 1000):
        me = dd.master_equation([('5.0', 1), ('1.0*x', -1)], box=(0, B))
        st = me.stationary()
        assert st.residual < 1e-10 and st.negative_mass < 1e-8
        np.testing.assert_allclose(me.cumulants(2)[1][0], 5.0, rtol=1e-8)
        np.testing.assert_allclose(me.cumulants(2)[2][0, 0], 5.0, rtol=1e-8)
        assert np.all(np.isfinite(me.correlation([0.0, 1.0]).C))
    for guess in (60.0, 150.0):                          # a bad guess is only a starting point
        me = dd.master_equation([('5.0', 1), ('1.0*x', -1)], box=(0, 400), guess=guess)
        np.testing.assert_allclose(me.cumulants(1)[1][0], 5.0, rtol=1e-8)
    # a stiff chain whose box centre is far from the bulk of the slow species
    for G in (1e2, 1e5):
        tr = [(f'{G}', (1, 0)), (f'{G}*x1', (-1, 0)), (f'x1/{G}', (0, 1)), (f'x2/{G}', (0, -1))]
        me = dd.master_equation(tr, fields=['x1', 'x2'], box=[(0, 15), (0, 40)])
        np.testing.assert_allclose(me.cumulants(1)[1], [1.0, 1.0], rtol=1e-5)


def test_refusals():
    with pytest.raises(MasterEquationError, match='negative rate'):
        dd.master_equation([('x - 5', 1), ('1', -1)], box=(0, 10)).generator()
    with pytest.raises(MasterEquationError, match='undeclared'):
        dd.master_equation([('lam', 1), ('x', -1)], box=(0, 10))
    with pytest.raises(MasterEquationError, match='closed communicating classes'):
        dd.master_equation([('1', 2), ('x', -2)], box=(0, 20)).stationary()       # even / odd lattices
    with pytest.raises(MasterEquationError, match='zero jump'):
        dd.master_equation([('1', 0)], box=(0, 5))
    with pytest.raises(MasterEquationError, match='box'):
        dd.master_equation([('1', 1), ('x', -1)])
    with pytest.raises(MasterEquationError, match='max_states'):
        dd.master_equation([('1', 1), ('x', -1)], box=(0, 100), max_states=50)


def test_transient_states_are_dropped_and_periodic_policy_is_uniform():
    # no death below x = 5: the states 0..3 are left for good and carry no stationary mass
    me = dd.master_equation([('3', 1), (lambda x: np.where(x >= 5, x, 0.0), -1)], box=(0, 40))
    st = me.stationary()
    assert st.closed_class_size == 37 and st.p[:4].sum() == 0.0
    assert st.p.sum() == pytest.approx(1.0, abs=1e-14)
    ring = dd.master_equation([('2', 1), ('2', -1)], box=(0, 9), policy='periodic')
    st = ring.stationary()
    np.testing.assert_allclose(st.p, 0.1, atol=1e-14)
    assert st.truncation_flux == 0.0


# ── Kramers-Moyal (diffusion) limit ───────────────────────────────────────────────────────────

def test_kramers_moyal_extrapolation_immigration_death():
    lam = 60.0                                        # 7.7 sigma from the boundary x = 0
    me = _imm_death(lam=lam, box=(0, 200))
    km = me.kramers_moyal(refine=(1, 2, 4))
    # m = 1 is the jump process itself
    for k in (1, 2, 3, 4):
        np.testing.assert_allclose(km.cumulants_by_m[1][k], me.cumulants(4)[k], rtol=1e-12)
    # the diffusion dx = (lam - mu x) dt + sqrt(lam + mu x) dW: Ito moment hierarchy
    #   kappa_2 = kappa_3 = lam/mu,  kappa_4 = 3 lam / (2 mu)
    np.testing.assert_allclose(km.cumulants[1].ravel(), lam / MU, rtol=1e-8)
    np.testing.assert_allclose(km.cumulants[2].ravel(), lam / MU, rtol=1e-6)
    np.testing.assert_allclose(km.cumulants[3].ravel(), lam / MU, rtol=1e-4)
    np.testing.assert_allclose(km.cumulants[4].ravel(), 1.5 * lam / MU, rtol=3e-4)
    assert km.error_estimate[4].ravel()[0] < 0.1 * 1.5 * lam
    # the jump process is NOT its diffusion limit beyond the second cumulant
    assert abs(me.cumulants(4)[4].ravel()[0] - km.cumulants[4].ravel()[0]) > 20.0
    assert max(km.truncation_flux.values()) < 1e-10
    # an explicit box overrides the parent's (here a wider upper edge: same law to the truncation)
    wide = me.kramers_moyal(refine=(1, 2, 4), box=(0, 240))
    np.testing.assert_allclose(wide.cumulants[4], km.cumulants[4], rtol=1e-9)
    with pytest.raises(MasterEquationError, match='negative rate'):      # m = 1 is the jump process
        me.kramers_moyal(refine=(1, 2), box=(-10, 200))


def test_kramers_moyal_correlation_matches_exact_linear():
    me = _imm_death(box=(0, 130))
    taus = np.array([0.0, 0.5, 2.0])
    km = me.kramers_moyal(refine=(1, 2), taus=taus)
    np.testing.assert_allclose(km.correlation[:, 0, 0], LAM / MU * np.exp(-MU * taus), rtol=2e-4)


# ── bridge to compute_cumulants (SDE front end with Poisson sources) ──────────────────────────

def _schlogl(Om):
    # birth Om (x^2 + 1), death Om (x^3 + x): one stable fixed point x* = 1, h = 1/Om
    return dd.master_equation(
        [('Om*(a*x^2 + c)', 1), ('Om*(b*x^3 + d*x)', -1)],
        params=dict(Om=Om, a=1.0, c=1.0, b=1.0, d=1.0), h=1.0 / Om, box=(0, int(4 * Om)))


def test_schlogl_one_loop_converges_to_the_oracle_as_system_size_grows():
    model = _schlogl(25).to_sde('schlogl').build()
    errs = {}
    for Om in (25, 50, 100):
        me = _schlogl(Om)
        assert not me.stationary().truncated
        cmp = me.compare_with_daedalus(model, ks=(1, 2), max_ell=1)
        assert [r['cumulant'] for r in cmp.rows] == ['kappa_1(x)', 'kappa_2(x,x)']
        errs[Om] = [(r['rel_error'][0], r['rel_error'][1]) for r in cmp.rows]
    for i in (0, 1):                                    # kappa_1, kappa_2
        tree = [errs[Om][i][0] for Om in (25, 50, 100)]
        loop = [errs[Om][i][1] for Om in (25, 50, 100)]
        assert all(l < t / 10 for l, t in zip(loop, tree))      # one loop is far better than the tree
        assert tree[0] / tree[1] == pytest.approx(2.0, rel=0.15)  # tree error ~ h
        assert tree[1] / tree[2] == pytest.approx(2.0, rel=0.15)
        assert loop[0] / loop[1] > 2.4 and loop[1] / loop[2] > 2.8  # one-loop error ~ h^2
    assert 'asymptotic' in cmp.notes[0] and 'kappa_2(x,x)' in cmp.table()


def test_bridge_linear_network_tree_is_exact_for_the_first_cumulants(net):
    cmp = net.compare_with_daedalus(ks=(1, 2, 3), max_ell=1)
    assert len(cmp.rows) == 2 + 3 + 4
    for r in cmp.rows:                                   # linear rates: the tree is exact, loops vanish
        assert r['rel_error'][0] < 1e-6 and r['rel_error'][1] < 1e-6, r
    with pytest.raises(MasterEquationError, match='string rates'):
        dd.master_equation([(lambda x: 3.0 + 0 * x, 1), ('x', -1)], box=(0, 20)).to_sde()
