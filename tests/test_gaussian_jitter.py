"""
Gaussian jitter of an exogenous input: ``Kernel.gaussian(sigma, tol=...)``.

The Gaussian density is acausal; on an exogenous stationary input it is
replaced by a rational fit of the density shifted by ``T`` with the shift
dropped (``daedalus.api._gaussian_fit``), which changes no stationary statistic of the
network.  Checked here against references that use the TRUE Gaussian:

* the fit itself: L1 error (independent quadrature), image and spectral
  errors, exact unit area;
* k = 2 tree of a linear network: the frequency integral of
  ``lam p^2 |H(omega)|^2 exp(-sigma^2 omega^2)`` (``scipy.integrate.quad``),
  within ``bound(2)`` times the unjittered variance; a narrow sigma
  (``sigma = 0.01`` against time constants of order 1) too;
* k = 3 tree: the time-domain triple product of the exactly convolved
  impulse response (exponentially modified Gaussians), within ``bound(3)``,
  and a loose fit (``tol = 1e-3``) whose error shows;
* a 1-population rate model (logistic rate, recurrent): tree + one loop
  against an independent simulation with the true Gaussian bumps;
* refusals: a source with feedback into its rate, a continuous field, a use
  without the jitter, two jitters, an observed jittered input, switch-on
  runs, the builder, an indexed sigma.
"""
from __future__ import annotations

import os
import sys
from fractions import Fraction

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from scipy.integrate import quad                                    # noqa: E402
from scipy.special import log_ndtr, roots_legendre                  # noqa: E402

from daedalus.api import compute_cumulants                                   # noqa: E402
from daedalus.api._gaussian_fit import gaussian_fit                          # noqa: E402
from daedalus.api.delay import Kernel                                        # noqa: E402
from daedalus.api.noise import Gaussian, Poisson                             # noqa: E402
from daedalus.api.sde import SDE, SDEError                                   # noqa: E402

TAU, TAUX, LAM, P, W = 1.0, 0.5, 2.0, 0.7, 1.3
KW = dict(parallel=False, verbose=False, use_cache=False)


def _linear(sigma=0.5, tol=1e-6, name='jitter linear'):
    return (SDE(name).parameter('tau', default=TAU, domain='positive')
            .parameter('tauX', default=TAUX, domain='positive')
            .parameter('lam', default=LAM, domain='positive')
            .parameter('p', default=P).parameter('w', default=W)
            .parameter('sigma', default=sigma, domain='positive')
            .define_kernel('kX', kernel=Kernel.exponential('tauX'))
            .define_kernel('g', kernel=Kernel.gaussian('sigma', tol=tol))
            .point_process('n', 'lam', jump_size='p')
            .field('v').equation('v', lhs='(tau*Dt + 1)*v', rhs='w*Conv(kX*g, n)'))


def _h(t, sigma):
    """Impulse response n -> v with the TRUE Gaussian: w (e^{-t/tau} -
    e^{-t/tauX})/(tau - tauX) convolved with the Gaussian (exponentially
    modified Gaussians)."""
    def emg(a):
        return np.exp(-a * t + 0.5 * (a * sigma) ** 2 + log_ndtr(t / sigma - a * sigma))
    return W / (TAU - TAUX) * (emg(1 / TAU) - emg(1 / TAUX))


# ── the fit ───────────────────────────────────────────────────────────
@pytest.mark.parametrize('tol', [1e-3, 1e-6, 1e-9])
def test_fit_errors(tol):
    f = gaussian_fit(tol)
    assert f.l1 <= tol and f.sup <= f.l1 and f.spectral <= 2 * f.sup + f.sup ** 2
    assert all(p.real < 0 for p in f.poles) and f.order == len(f.poles)
    assert f.truncated_mass <= tol / 10
    # L1 by an independent quadrature (panels of Gauss-Legendre, the impulse
    # response from the exact fraction's partial fractions in numpy)
    x, w = roots_legendre(20)
    edges = np.linspace(0.0, f.shift + 40.0, 4001)
    h = np.diff(edges)
    t = (edges[:-1, None] + 0.5 * h[:, None] * (x[None, :] + 1.0)).ravel()
    ww = (0.5 * h[:, None] * w[None, :]).ravel()
    gauss = np.exp(-0.5 * (t - f.shift) ** 2) / np.sqrt(2 * np.pi)
    l1 = ww @ np.abs(f.impulse(t) - gauss) + f.truncated_mass
    assert l1 == pytest.approx(f.l1, rel=1e-3)
    # the kernel string: one exact fraction with unit area, the sections' function
    N, D = f.fraction()
    assert N[0] == D[0] and D[-1] == 1 and isinstance(N[0], Fraction)
    # (the exact fraction evaluated at 40 digits: its coefficients reach 6e15 at
    # tol = 1e-9, so a double Horner evaluation carries ~1e-12 of rounding)
    import mpmath
    with mpmath.workdps(40):
        zs = [mpmath.mpc(0, w) for w in np.linspace(0.0, 8.0, 33)]
        ex = [mpmath.polyval([mpmath.mpf(c.numerator) / c.denominator for c in N[::-1]], q)
              / mpmath.polyval([mpmath.mpf(c.numerator) / c.denominator for c in D[::-1]], q)
              for q in zs]
        ex = np.array([complex(v) for v in ex])
    # f.transfer sums the sections in double: rounding ~ eps * sum |residues|
    rnd = 1e-15 * sum(abs(r) for r in f.residues) + 1e-14
    assert np.max(np.abs(ex - f.transfer(1j * np.linspace(0.0, 8.0, 33)))) < rnd
    assert 'omega' in Kernel.gaussian('sigma', tol=tol).freq_image


def test_fit_orders_grow_with_accuracy():
    orders = [gaussian_fit(t).order for t in (1e-2, 1e-4, 1e-6, 1e-8)]
    assert orders == sorted(orders) and orders[2] <= 18


# ── trees against the true Gaussian ───────────────────────────────────
def _k2_reference(taus, sigma, jitter=True):
    def S(om):
        g2 = np.exp(-(sigma * om) ** 2) if jitter else 1.0
        return LAM * P * P * W * W * g2 / ((1 + (om * TAUX) ** 2) * (1 + (om * TAU) ** 2))
    return np.array([quad(lambda o: S(o) * np.cos(o * t), 0, np.inf, epsabs=1e-14,
                          limit=2000)[0] / np.pi for t in taus])


@pytest.mark.parametrize('sigma', [0.5, 0.3])
def test_k2_tree_matches_the_true_gaussian(sigma):
    taus = np.array([0.0, 0.3, 1.0, 2.5])
    r = compute_cumulants(_linear(sigma).build(), k=2, max_ell=0,
                          external_fields=[('v', 1), ('v', 1)], tau_grid=taus, **KW)
    assert r['status'] == 'ok'
    C = np.real(np.asarray(r['C_tau_by_ell'][0]))
    ref = _k2_reference(taus, sigma)
    var0 = _k2_reference([0.0], sigma, jitter=False)[0]       # unjittered variance
    b2 = r['input_jitter']['n']['fit']['bound_by_order'][2]
    assert np.max(np.abs(C - ref)) <= b2 * var0
    # the jitter matters at this width (the check is not vacuous)
    assert np.max(np.abs(ref - _k2_reference(taus, sigma, jitter=False))) > 1e3 * b2 * var0
    led = [e for e in r['analysis'].ledger if e.name == 'jitter:n']
    assert led and led[0].kind == 'approximation' and '16 states' in led[0].detail


def test_k2_narrow_sigma():
    """Adversarial width: sigma = 0.01 against time constants 0.5 and 1 (the
    fit's poles ~ 1/sigma, 100x faster than the network)."""
    sigma, taus = 0.01, np.array([0.0, 0.01, 0.3, 2.5])
    r = compute_cumulants(_linear(sigma).build(), k=2, max_ell=0,
                          external_fields=[('v', 1), ('v', 1)], tau_grid=taus, **KW)
    C = np.real(np.asarray(r['C_tau_by_ell'][0]))
    ref = _k2_reference(taus, sigma)
    var0 = _k2_reference([0.0], sigma, jitter=False)[0]
    assert np.max(np.abs(C - ref)) <= r['input_jitter']['n']['fit']['bound_by_order'][2] * var0


def _k3_reference(pts, sigma):
    out = []
    for a, b, c in pts:
        lo = min(a, b, c)
        out.append(LAM * P ** 3 * quad(lambda u: _h(a - u, sigma) * _h(b - u, sigma) * _h(c - u, sigma),
                                       lo - 40.0, lo + 10 * sigma, points=[lo - 10 * sigma, lo],
                                       epsabs=1e-15, epsrel=1e-12, limit=2000)[0])
    return np.array(out)


@pytest.mark.parametrize('tol', [1e-6, 1e-3])
def test_k3_tree_matches_the_true_gaussian(tol):
    sigma = 0.5
    pts = [(0.0, 0.0, 0.0), (0.0, 0.4, 1.1), (0.0, -0.7, 0.3), (0.0, 1.5, 3.0)]
    r = compute_cumulants(_linear(sigma, tol=tol).build(), k=3, max_ell=0,
                          external_fields=[('v', 1)] * 3, points=pts, **KW)
    C = np.real(np.array([complex(r['total_C_by_ell'][0](*q)) for q in pts]))
    ref = _k3_reference(pts, sigma)
    rel = np.max(np.abs(C - ref)) / np.max(np.abs(ref))
    fit = r['input_jitter']['n']['fit']
    assert rel <= fit['bound_by_order'][3]
    if tol == 1e-3:
        assert rel > 1e-6                     # the fit error shows (6e-5 measured)


def test_mean_is_exact():
    r = compute_cumulants(_linear(0.3).build(), k=1, max_ell=0, external_fields=[('v', 1)],
                          tau_grid=np.array([0.0]), **KW)
    assert r['mf_values']['vstar'][0] == pytest.approx(W * LAM * P, rel=1e-14)


# ── a nonlinear rate model against a simulation with the true Gaussian ─
RATE_P = dict(tau=1.0, tauX=0.5, lam=4.0, p=0.5, w=1.0, J=1.0, sigma=0.4)


def _rate_model():
    s = SDE('jitter rate model')
    for k, v in RATE_P.items():
        s = s.parameter(k, default=v, domain='positive' if k in ('tau', 'tauX', 'lam', 'sigma')
                        else None)
    return (s.define_kernel('kX', kernel=Kernel.exponential('tauX'))
            .define_kernel('g', kernel=Kernel.gaussian('sigma'))
            .define_function('phi', args=['v'], expression='1/(1 + exp(-v))')
            .point_process('n', 'lam', jump_size='p')
            .field('v').equation('v', lhs='(tau*Dt + 1)*v',
                                 rhs='J*phi(v) + w*Conv(kX*g, n) - w*lam*p'))


def _simulate(lags, n_paths=500, T=400.0, burn=20.0, dt=0.01, seed=1):
    """Euler paths of tau dv/dt = -v + J phi(v) + w x(t) - w lam p, with
    x = p sum_j (kX * g)(t - t_j) built from the TRUE Gaussian (an
    exponentially modified Gaussian per Poisson event; event times on the dt
    grid)."""
    tau, tauX, lam, p, w, J, sg = (RATE_P[k] for k in ('tau', 'tauX', 'lam', 'p', 'w', 'J', 'sigma'))
    rng = np.random.default_rng(seed)
    nt, nb = int((T + burn) / dt), int(burn / dt)
    off = int(6 * sg / dt)
    kt = np.arange(-off, int(25 * tauX / dt)) * dt
    emg = np.exp(-kt / tauX + sg * sg / (2 * tauX * tauX) + log_ndtr(kt / sg - sg / tauX)) / tauX
    x = np.empty((n_paths, nt))
    for i in range(n_paths):
        te = rng.uniform(-10.0, T + burn, rng.poisson(lam * (T + burn + 10.0)))
        idx = np.round(te / dt).astype(int) + kt.size
        cnt = np.bincount(idx[idx >= 0], minlength=nt + 2 * kt.size)
        x[i] = p * np.convolve(cnt, emg)[kt.size + off: kt.size + off + nt]
    v = np.zeros(n_paths)
    vs = np.empty((n_paths, nt - nb))
    for k in range(nt):
        v = v + dt * (-v + J / (1 + np.exp(-v)) + w * x[:, k] - w * lam * p) / tau
        if k >= nb:
            vs[:, k - nb] = v
    vc = vs - vs.mean()                       # pooled mean: no per-path bias
    cov, se = [], []
    for L in lags:
        l_ = int(round(L / dt))
        c = (vc[:, :vc.shape[1] - l_] * vc[:, l_:]).mean(axis=1)
        cov.append(c.mean())
        se.append(c.std() / np.sqrt(n_paths))
    return vs.mean(), vs.mean(axis=1).std() / np.sqrt(n_paths), np.array(cov), np.array(se)


def test_rate_model_tree_and_one_loop_vs_simulation():
    lags = np.array([0.0, 1.0, 2.0])
    m = _rate_model().build()
    r1 = compute_cumulants(m, k=1, max_ell=1, external_fields=[('v', 1)],
                           tau_grid=np.array([0.0]), **KW)
    r2 = compute_cumulants(m, k=2, max_ell=1, external_fields=[('v', 1), ('v', 1)],
                           tau_grid=lags, **KW)
    assert r1['status'] == r2['status'] == 'ok'
    mean = r1['mf_values']['vstar'][0] + np.real(r1['C_tau_by_ell'][1][0])
    tree = np.real(np.asarray(r2['C_tau_by_ell'][0]))
    C = tree + np.real(np.asarray(r2['C_tau_by_ell'][1]))
    mu, se_mu, cov, se = _simulate(lags)
    assert abs(mu - mean) < 4 * se_mu
    assert np.all(np.abs(cov - C) < 4 * se + 0.005 * C[0])
    # the loop is resolved: the tree alone is off by more than the error bars
    assert abs(cov[0] - tree[0]) > 4 * se[0]


# ── refusals and scope ────────────────────────────────────────────────
def _base(name='jitter refusals'):
    return (SDE(name).parameter('lam', default=1.0, domain='positive')
            .parameter('sigma', default=0.3, domain='positive')
            .parameter('tauX', default=0.5, domain='positive').parameter('w', default=0.5)
            .define_kernel('kX', kernel=Kernel.exponential('tauX'))
            .define_kernel('g', kernel=Kernel.gaussian('sigma')))


def test_refuses_feedback_into_the_source_rate():
    s = (_base().define_kernel('r', kernel=Kernel.exponential('tauX'))
         .point_process('n', 'lam*exp(w*Conv(r, n))'))
    s = s.field('v').equation('v', lhs='Dt*v', rhs='-v + w*Conv(kX*g, n)')
    with pytest.raises(SDEError, match='feedback into the source'):
        s.build()
    s = (_base().point_process('n', 'lam*exp(v)').field('v')
         .equation('v', lhs='Dt*v', rhs='-v + w*Conv(kX*g, n)'))
    with pytest.raises(SDEError, match=r"depends on the field\(s\) \['v'\]"):
        s.build()


def test_refuses_a_continuous_field_and_a_use_without_the_jitter():
    s = _base().field('v').equation('v', lhs='Dt*v', rhs='-v + w*Conv(kX*g, v)')
    with pytest.raises(SDEError, match='is not such a source'):
        s.build()
    s = (_base().point_process('n', 'lam').field('v')
         .equation('v', lhs='Dt*v', rhs='-v + w*Conv(kX*g, n) + w*Conv(kX, n)'))
    with pytest.raises(SDEError, match='used without its Gaussian jitter'):
        s.build()


def test_refuses_two_jitters_on_one_input():
    s = (_base().define_kernel('g2', kernel=Kernel.gaussian('sigma', tol=1e-4))
         .point_process('n', 'lam').field('v').field('u')
         .equation('v', lhs='Dt*v', rhs='-v + w*Conv(kX*g, n)')
         .equation('u', lhs='Dt*u', rhs='-u + w*Conv(kX*g2, n)'))
    with pytest.raises(SDEError, match='share one Gaussian'):
        s.build()
    with pytest.raises(ValueError, match='two Gaussian jitters'):
        Kernel.gaussian('sigma') * Kernel.gaussian('sigma')


def test_refuses_observing_the_input_and_switch_on_runs():
    m = _linear(name='jitter observed').build()
    with pytest.raises(ValueError, match='Gaussian-jittered input'):
        compute_cumulants(m, k=2, max_ell=0, external_fields=[('v', 1), ('n', 1)],
                          tau_grid=np.array([0.0]), **KW)
    with pytest.raises(ValueError, match='switch-on transients'):
        compute_cumulants(m, k=2, max_ell=0, external_fields=[('v', 1), ('v', 1)],
                          t_min=0.0, times=np.array([1.0]), **KW)


def test_declaration_scope():
    from daedalus.api.model import TemporalModelBuilder
    with pytest.raises(ValueError, match='SDE front end'):
        TemporalModelBuilder('b').define_kernel('g', kernel=Kernel.gaussian('sigma'))
    with pytest.raises(SDEError, match='is indexed'):
        (SDE('ix').population('E', size=2).parameter('s', default=[0.1, 0.2], indexed_by=['E'])
         .define_kernel('g', kernel=Kernel.gaussian('s[i]')))
    with pytest.raises(ValueError, match='outside'):
        Kernel.gaussian('sigma', tol=1e-12)
    # an exposed noise source with a field-free rate is an exogenous input too
    m = (_base('jitter exposed').noise('X', Poisson('lam'), as_field='n').field('v')
         .equation('v', lhs='Dt*v', rhs='-v + w*Conv(kX*g, n)')).build()
    assert set(m['exogenous_jitter']) == {'n'}
    assert m['exogenous_jitter']['n']['fit']['order'] == 16


def test_a_gaussian_time_kernel_is_refused_with_a_pointer():
    from daedalus.api.analyze import AnalyzerError
    s = (SDE('acausal gaussian').parameter('lam', default=1.0, domain='positive')
         .parameter('sigma', default=0.5, domain='positive')
         .define_kernel('g', time_expr='exp(-t^2/(2*sigma^2))/sqrt(2*pi*sigma^2)')
         .point_process('n', 'lam').field('v').equation('v', lhs='Dt*v', rhs='-v + Conv(g, n)'))
    with pytest.raises(AnalyzerError, match='Kernel.gaussian'):
        compute_cumulants(s.build(), k=2, max_ell=0, external_fields=[('v', 1), ('v', 1)],
                          tau_grid=np.array([0.0]), **KW)


def test_refuses_the_bypasses_of_the_exogeneity_check():
    # an exposed input that also drives a field directly (into=): the direct
    # path would not be shifted with the jittered one
    with pytest.raises(SDEError, match='also drives'):
        (_base('jitter direct').noise('X', Poisson('lam'), into={'v': 'w'}, as_field='n')
         .field('v').equation('v', lhs='Dt*v', rhs='-v + w*Conv(kX*g, n)')).build()
    # a parenthesised kernel argument is still a jitter use (on a field: refused)
    for kern in ('(g)', '(kX*g)'):
        s = (_base(f'jitter paren {kern}').field('u').field('v')
             .drift('u', '-u + 0.8*v').equation('v', lhs='Dt*v', rhs=f'-v - w*Conv({kern}, u)'))
        with pytest.raises(SDEError, match='is not such a source'):
            s.build()
    # a jitter inside a noise coefficient is seen too
    s = (_base('jitter coefficient').field('u').field('v').drift('u', '-u').drift('v', '-v')
         .noise('W', Gaussian(var='0.1'), into={'v': '1 + 0.1*Conv(kX*g, u)'}))
    with pytest.raises(SDEError, match='is not such a source'):
        s.build()
    # sigma must not hold a field
    s = (SDE('jitter sigma field').parameter('lam', default=1.0, domain='positive')
         .field('v').define_kernel('g', kernel=Kernel.gaussian('v')).point_process('n', 'lam')
         .equation('v', lhs='Dt*v', rhs='-v + Conv(g, n)'))
    with pytest.raises(SDEError, match='sigma is a parameter expression'):
        s.build()


def test_modular_kalman_falls_back_on_a_denominator_divisible_by_its_prime():
    from sage.all import QQ, matrix
    from daedalus.api import _realization as R
    A = matrix(QQ, [[1, QQ(1) / R._KRYLOV_PRIME], [0, 2]])
    B = matrix(QQ, [[1], [1]])
    assert R._controllable_part_modular(A, B) is None
    assert R._controllable_part(A, B, QQ)[2] is None            # exact ranks: controllable


def test_a_jitter_of_a_nonlinear_argument_is_not_embedded():
    """The Markov embedding of kernels of nonlinear arguments leaves a Gaussian
    jitter to the exogeneity check (here: on a field, refused)."""
    s = _base('jitter nonlinear').field('v').equation('v', lhs='Dt*v', rhs='-v + w*Conv(kX*g, v^2)')
    with pytest.raises(SDEError, match='is not such a source'):
        s.build()
