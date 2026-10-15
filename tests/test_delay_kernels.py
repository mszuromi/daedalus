"""
tests/test_delay_kernels.py
===========================
Delay kernels (``api.delay``, ``api._kernel_stability``, ``api._delay_exact``,
``simulations.delay_sde_sim``).  Inline public models only:

* the front end: ``x(t - d)`` in an SDE drift, ``define_kernel(delay=...)``,
  ``kernel=Kernel.delay(d) * Kernel.exponential(tau)``, a shifted
  ``time_expr`` and a hand-written ``exp(-I*omega*d)`` image give the same
  ``K(omega)``;
* the Pade (n, n) approximant (all-pass, order of contact) and its
  substitution;
* G2-K: Lambert W against the argument principle at seven delays, the
  2-population delayed cycle against its Pade roots, an unstable delayed
  saddle refused by name, neutral / acausal statuses;
* the exact tree (boundary-value problem + method of steps) against the
  spectral integral and a closed form (delayed noise), and as the pipeline's
  tree; the one loop against the exact O(eps) (Hartree) correction; the
  automatic order, the ledger and the switch to the Pade tree;
* the simulator against the exact covariance of its own scheme (statistics)
  and, extrapolated in dt, against the exact tree;
* a rational-kernel model keeps its numbers (G2-K adds an info finding).

Scalar delayed OU: dx = (-a x + c x(t - d) - eps x^3) dt + sqrt(2D) dW,
a = 1, c = -2, D = 0.1 (Hopf at d* = 2 pi / (3 sqrt 3) = 1.2092).
"""
import math
import os
import sys
import warnings

import numpy as np
import pytest

pytest.importorskip('sage.all')

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from sage.all import SR, I, exp, matrix                                 # noqa: E402

from api import compute_cumulants                                      # noqa: E402
from api.analyze import (AnalyzerError, Question, analyze,             # noqa: E402
                         analyze_with_context)
from api.delay import (Kernel, delay_factors, pade_coefficients,       # noqa: E402
                       pade_np, pade_substitute, split_shifted_time_expr)
from api._delay_exact import DelayedStateSpace, ExactTree              # noqa: E402
from api._kernel_stability import (count_rhp_roots, kernel_stability,  # noqa: E402
                                   numeric_K)
from api.model import TemporalModelBuilder                             # noqa: E402
from api.noise import Gaussian                                         # noqa: E402
from api.sde import SDE                                                # noqa: E402

A_, C_, D_, EPS = 1.0, -2.0, 0.1, 0.02
TAUS = np.array([0.0, 0.5, 1.0, 2.0, 4.0])
EXT = [('x', 1), ('x', 1)]


def _ou(name, d=0.5, style='x(t-d)', eps=EPS):
    s = (SDE(name).parameter('a', default=A_).parameter('c', default=C_)
         .parameter('d', default=d, domain='positive').parameter('eps', default=eps)
         .parameter('D', default=D_).field('x'))
    if style == 'x(t-d)':
        s = s.drift('x', '-a*x + c*x(t - d) - eps*x^3')
    else:
        kw = {'delay=': dict(delay='d'),
              'freq_image': dict(freq_image='exp(-I*omega*d)'),
              'time_expr': dict(time_expr='dirac_delta(t - d)'),
              'Kernel': dict(kernel=Kernel.delay('d'))}[style]
        s = s.define_kernel('g', **kw).drift('x', '-a*x + c*Conv(g, x) - eps*x^3')
    return s.noise('xi', Gaussian(var='2*D'), couples={'x': 1}).build()


def _ou_state_space(d, a=A_, c=C_, D=D_):
    one = np.array([[1.0]])
    return DelayedStateSpace(np.array([[-a]]), np.array([[c]]), one, np.array([[0.0]]),
                             one, np.array([[0.0]]), d, np.array([[2 * D]]))


def _exact_ou(taus, d, a=A_, c=C_, D=D_):
    return ExactTree(_ou_state_space(d, a, c, D), 0, 0).values(taus)


def _hartree(taus, d, h=1e-5):
    """Exact O(eps) one-loop correction at x* = 0: 3 eps C0(0) dC0/da."""
    c00 = _exact_ou([0.0], d)[0]
    return 3 * EPS * c00 * (_exact_ou(taus, d, a=A_ + h) - _exact_ou(taus, d, a=A_ - h)) / (2 * h)


def _spectral_ou(taus, d):
    """(1/pi) int_0^oo 2D cos(w tau) / |K(i w)|^2 dw, the 2D/(w^2 + a^2) part
    analytically (QAWF on the rest, which decays like w^-3)."""
    from scipy.integrate import quad
    R = lambda w: 2 * D_ * (1 / abs(1j * w + A_ - C_ * np.exp(-1j * w * d)) ** 2
                            - 1 / (w * w + A_ * A_))
    return np.array([(D_ / A_) * math.exp(-A_ * t)
                     + quad(R, 0, np.inf, weight='cos', wvar=t, limlst=400, limit=4000,
                            epsabs=1e-16)[0] / math.pi for t in taus])


def _cc(model, ell, taus=TAUS, **kw):
    kw.setdefault('use_cache', False)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return compute_cumulants(model, k=2, max_ell=ell, external_fields=EXT,
                                 tau_grid=np.asarray(taus), parallel=False,
                                 verbose=False, taylor_order=4, **kw)


# ── Pade approximant and recognition ────────────────────────────────────


def test_pade_all_pass_and_contact_order():
    for n in (1, 4, 9):
        c = pade_coefficients(n)
        assert c[0] == 1 and len(c) == n + 1
        x = np.linspace(-30, 30, 61)
        assert np.max(np.abs(np.abs(pade_np(n, 1j * x, 0.7)) - 1.0)) < 1e-12
        # contact of order 2n + 1: error (n!)^2 / ((2n)! (2n+1)!) z^(2n+1) (leading term)
        z = 0.3
        lead = math.factorial(n) ** 2 / (math.factorial(2 * n) * math.factorial(2 * n + 1)) * z ** (2 * n + 1)
        assert abs(pade_np(n, z, 1.0) - math.exp(-z)) <= 1.5 * lead + 4e-16


def test_recognition_and_substitution():
    om, d, tau = SR.var('omega'), SR.var('d'), SR.var('tau')
    e = exp(-I * om * d) / (1 + I * om * tau) + 3
    assert [str(x) for x in delay_factors(e, om)] == ['d']
    p = pade_substitute(e, om, 3)
    assert not delay_factors(p, om)
    assert SR(p).is_polynomial(om) is False and SR(p).numerator().is_polynomial(om)
    assert split_shifted_time_expr('heaviside(t - d)*exp(-(t - d)/tau)/tau') == \
        ('d', str(SR('heaviside(t)*exp(-t/tau)/tau').expand()))
    assert split_shifted_time_expr('dirac_delta(t - d)')[0] == 'd'
    assert split_shifted_time_expr('exp(-t/tau)*heaviside(t)') is None
    # an instantaneous part next to a delayed one is not one delayed kernel
    assert split_shifted_time_expr('dirac_delta(t) + 0.5*dirac_delta(t - d)') is None
    assert split_shifted_time_expr('dirac_delta(t - d1) + dirac_delta(t - d2)') is None
    assert split_shifted_time_expr('0.3*dirac_delta(t - d) + heaviside(t - d)*exp(-(t - d)/tau)/tau')[0] == 'd'


def test_kernel_composition():
    k = Kernel.delay('d') * Kernel.exponential('tau')
    assert k.delay == 'd' and 'tau' in k.freq_image
    assert (Kernel.delay('d1') * Kernel.delay('d2')).delay == '(d1) + (d2)'
    assert Kernel.gamma(3, 'm').freq_image == '(1 + I*omega*(m)/3)^(-3)'


def test_front_end_forms_give_the_same_operator():
    ref = None
    for style in ('x(t-d)', 'delay=', 'freq_image', 'time_expr', 'Kernel'):
        rep, ctx = analyze_with_context(_ou(f'delay front end {style}', style=style),
                                        Question(k=2, external_fields=tuple(EXT), max_ell=1))
        assert rep.traits.propagator_class == 'delay', (style, rep.summary())
        assert rep.has('G3_DELAY_KERNEL') and not rep.errors, (style, rep.summary())
        K = SR(ctx.K_ft[0, 0]).subs(ctx.num_params)
        if ref is None:
            ref = K
        assert SR(K - ref).simplify_full().is_zero(), (style, K, ref)
    # a delayed exponential: time_expr shifted by d == delay= with the unshifted image
    imgs = []
    for kw in (dict(time_expr='heaviside(t - d)*exp(-(t - d)/tau)/tau'),
               dict(delay='d', freq_image='1/(1 + I*omega*tau)'),
               dict(kernel=Kernel.delay('d') * Kernel.exponential('tau'))):
        m = (TemporalModelBuilder('delayed exponential ' + str(len(imgs))).physical_field('x')
             .parameter('d', default=0.5, domain='positive')
             .parameter('tau', default=0.3, domain='positive')
             .parameter('J', default=0.4).parameter('D', default=0.1)
             .define_kernel('g', **kw)
             .set_action_text('xt*((Dt + 1)*x - J*g*x) - D*xt^2')
             .set_mf_equation('xstar', '0').build())
        rep, ctx = analyze_with_context(m, Question(k=2, external_fields=tuple(EXT), max_ell=0))
        assert rep.traits.propagator_class == 'delay', rep.summary()
        imgs.append(SR(ctx.K_ft[0, 0]).subs(ctx.num_params))
    for e in imgs[1:]:
        assert SR(e - imgs[0]).simplify_full().is_zero(), (e, imgs[0])


# ── G2-K ────────────────────────────────────────────────────────────────


def test_g2k_lambert_matches_argument_principle():
    om, a, c, d = SR.var('omega'), SR.var('a'), SR.var('c'), SR.var('d')
    K = matrix(SR, [[I * om + a - c * exp(-I * om * d)]])
    expect = {0.5: 0, 1.1: 0, 1.2092: 2, 1.3: 2, 1.4: 2, 3.0: 2, 6.0: 4}
    for dv, n_exp in expect.items():
        npar = {a: A_, c: C_, d: dv}
        lw = kernel_stability(K, om, npar)
        assert lw.method == 'lambert_w' and lw.n_rhp == n_exp, (dv, lw)
        Kf, _ = numeric_K(K, om, npar)
        ap = count_rhp_roots(lambda s: np.linalg.det(Kf(s)))
        assert ap.n_rhp == n_exp, (dv, ap)
        for z in ap.roots:
            assert min(abs(z - w) for w in lw.roots) < 1e-8 * max(1.0, abs(z))
    lw = kernel_stability(K, om, {a: A_, c: C_, d: 1.4})
    assert abs(lw.rightmost.real - 0.048369) < 1e-6 and abs(abs(lw.rightmost.imag) - 1.547343) < 1e-6


def test_g2k_two_population_cycle_matches_pade_roots():
    from numpy.polynomial import Polynomial as Poly
    om, d = SR.var('omega'), SR.var('d')
    tau, aa, W = [1.0, 1.5], [1.0, 1.0], [[0.3, -1.6], [1.4, 0.2]]
    K = matrix(SR, 2, 2, lambda i, j: (tau[i] * I * om + 1 if i == j else 0)
               - aa[i] * W[i][j] * exp(-I * om * d))
    for dv, n_exp in ((0.5, 0), (1.0, 2), (2.0, 2)):
        ks = kernel_stability(K, om, {d: dv})
        assert ks.status == ('unstable' if n_exp else 'stable') and ks.n_rhp == n_exp
        c = [float(x) for x in pade_coefficients(12)]
        P = Poly([ck * (-dv) ** k for k, ck in enumerate(c)])
        Q = Poly([ck * dv ** k for k, ck in enumerate(c)])
        M = [[Poly([1, tau[i]]) * Q * (1 if i == j else 0) - aa[i] * W[i][j] * P
              for j in range(2)] for i in range(2)]
        roots = (M[0][0] * M[1][1] - M[0][1] * M[1][0]).roots()
        rhp = [z for z in roots if z.real > 1e-9]
        assert len(rhp) == n_exp
        for z in ks.roots:
            assert min(abs(z - w) for w in rhp) < 1e-6
    ks = kernel_stability(K, om, {d: 1.0})
    assert abs(ks.rightmost - complex(0.07346, 0.72058)) < 1e-4 or \
        abs(ks.rightmost - complex(0.07346, -0.72058)) < 1e-4


def test_g2k_neutral_and_acausal():
    om, d = SR.var('omega'), SR.var('d')
    neutral = matrix(SR, [[I * om * (1 - SR(0.5) * exp(-I * om * d)) + 1]])
    assert kernel_stability(neutral, om, {d: 1.0}).status == 'neutral'
    acausal = matrix(SR, [[I * om + 1 - SR(0.3) / (1 - I * om * 2)]])
    assert kernel_stability(acausal, om, {}).status == 'acausal'
    advance = matrix(SR, [[I * om + 1 - SR(0.3) * exp(I * om * 2)]])
    assert kernel_stability(advance, om, {}).status == 'acausal'


def test_unstable_delayed_saddle_refused_by_name():
    with pytest.raises(AnalyzerError) as ei:
        _cc(_ou('delay unstable saddle', d=1.4), 0)
    assert 'G2_UNSTABLE_SADDLE' in ei.value.codes
    assert '0.048369' in str(ei.value) and 'delay' in str(ei.value)
    rep = analyze(_ou('delay unstable saddle (analyze)', d=1.4),
                  Question(k=2, external_fields=tuple(EXT), max_ell=0))
    assert rep.has('G2_UNSTABLE_SADDLE') and rep.traits.kernel_stability == 'unstable'
    assert rep.traits.n_unstable_roots == 2


# ── exact tree ──────────────────────────────────────────────────────────


@pytest.mark.parametrize('d', [0.5, 1.1])
def test_exact_tree_matches_spectral_integral(d):
    t = TAUS[1:]
    bvp, spec = _exact_ou(t, d), _spectral_ou(t, d)
    # QAWF-limited (a 30-digit mpmath quadrature agrees with the BVP to 3e-14)
    assert np.max(np.abs(bvp - spec)) <= 5e-10, (bvp, spec)
    # tau = 0: the equal-time value of the same covariance
    assert abs(_exact_ou([0.0], d)[0] - _exact_ou([1e-9], d)[0]) < 1e-9


def test_delayed_noise_matches_closed_form():
    from scipy.integrate import quad
    a, beta, d = 1.3, 0.6, 1.0
    one = np.array([[1.0]])
    ss = DelayedStateSpace(np.array([[-a]]), np.array([[0.0]]), one, np.array([[beta]]),
                           one, np.array([[0.0]]), d, one)
    h = lambda u: math.exp(-a * u) + (beta * math.exp(-a * (u - d)) if u > d else 0.0)
    taus = [0.0, 0.3, 0.7, 1.6, 2.5]
    cf = [quad(lambda u: h(t + u) * h(u), 0, 60, points=[d, max(d - t, 0)],
               limit=500, epsabs=1e-15)[0] for t in taus]
    assert np.max(np.abs(ExactTree(ss, 0, 0).values(taus) - cf)) < 1e-13


def test_pipeline_tree_exact_and_one_loop_vs_hartree():
    r = _cc(_ou('delay pipeline n8'), 1, delay={'order': 8})
    tree = np.real(r['C_tau_by_ell'][0])
    exact = _exact_ou(np.where(TAUS == 0, -1e-6, TAUS), 0.5)   # tau = 0: the Ito nudge
    assert np.max(np.abs(tree - exact)) <= 1e-12 * np.max(np.abs(exact))
    loop = np.real(r['C_tau_by_ell'][1])
    ref = _hartree(TAUS, 0.5)
    assert np.max(np.abs(loop - ref)) / np.max(np.abs(ref)) < 1e-4    # Pade-8: 2.9e-5
    info = r['delay']
    assert info['n'] == 8 and info['tree'] == 'exact' and info['tree_method'] == 'bvp'
    assert info['target_met'] is None and info['stability']['status'] == 'stable'
    assert 1e-4 < info['tree_vs_pade_rel'] < 1e-3                       # Pade-8 tree: 3.2e-4
    assert r['analysis'].has('G3_DELAY_APPROXIMATED')
    assert r['analysis'].traits.propagator_class == 'delay-rational'
    led = [e for e in r['analysis'].ledger if e.name.startswith('delay:')]
    assert led and led[0].kind == 'approximation' and 'Pade(8,8)' in led[0].detail
    assert r['total_C'](0.0, 0.5) == pytest.approx(tree[1] + loop[1], rel=1e-12)
    assert r['total_C_by_ell'][0](1.0, 1.0) == pytest.approx(tree[0], rel=1e-13)


def test_auto_order_meets_its_target():
    r = _cc(_ou('delay auto order'), 1, delay={'tol': 1e-6})
    info = r['delay']
    assert info['orders_run'] == [4, 8] and info['n'] == 8
    assert info['target_met'] is True and info['est_rel_error'] <= 1e-6
    # the estimate is conservative: the true loop error of n = 8 (vs the exact
    # Hartree correction) relative to max |C| is below it
    tot = np.real(r['C_tau'])
    true = np.max(np.abs(np.real(r['C_tau_by_ell'][1]) - _hartree(TAUS, 0.5)))
    assert true / np.max(np.abs(tot)) < info['est_rel_error']


def test_pade_tree_switch(monkeypatch):
    r = _cc(_ou('delay pade tree'), 0, delay={'order': 4, 'tree': 'pade'})
    exact = _exact_ou(TAUS, 0.5)
    err = np.max(np.abs(np.real(r['C_tau_by_ell'][0]) - exact)) / np.max(np.abs(exact))
    assert 1e-3 < err < 3e-3 and r['delay']['tree'] == 'pade'           # Pade-4: 2.2e-3
    monkeypatch.setenv('DAEDALUS_DELAY_EXACT_TREE', '0')
    r2 = _cc(_ou('delay pade tree env'), 0, delay={'order': 4})
    assert r2['delay']['tree'] == 'pade'
    assert np.allclose(np.real(r2['C_tau_by_ell'][0]), np.real(r['C_tau_by_ell'][0]), rtol=1e-12)


# ── simulator ───────────────────────────────────────────────────────────


def test_simulator_against_exact_covariances():
    from simulations.delay_sde_sim import simulate, autocovariance, em_linear_autocovariance
    d, dt = 0.5, 0.005
    taus = np.array([0.0, 0.5, 1.0, 2.0])
    rec = simulate(lambda x, xd: -A_ * x + C_ * xd[0], [d], [[math.sqrt(2 * D_)]], [0.0],
                   dt=dt, t_max=200.0, n_paths=400, burn_in=20.0,
                   record_every=10, seed=7)
    est, se = autocovariance(rec, 0, 0, np.round(taus / rec['dt_rec']).astype(int))
    em = em_linear_autocovariance(A_, C_, 2 * D_, dt, int(round(d / dt)), 400)
    em_half = em_linear_autocovariance(A_, C_, 2 * D_, dt / 2, int(round(d / (dt / 2))), 800)
    idx = np.round(taus / dt).astype(int)
    # statistics: the run against the exact covariance of its own scheme
    assert np.all(np.abs(est - em[idx]) <= 4 * se), (est, em[idx], se)
    # discretisation: first order in dt; extrapolated, the exact tree
    exact = _exact_ou(taus, d)
    rich = 2 * em_half[2 * idx] - em[idx]
    assert np.max(np.abs(rich - exact)) < 3e-5 < np.max(np.abs(em[idx] - exact))


# ── models without delays ───────────────────────────────────────────────


def test_rational_kernel_model_unchanged_and_g2k_info():
    m = (TemporalModelBuilder('delay test exp kernel OU').physical_field('x')
         .parameter('mu', default=1.0, domain='positive').parameter('J', default=0.5)
         .parameter('D', default=0.1, domain='positive')
         .parameter('tg', default=0.7, domain='positive')
         .define_kernel('g', freq_image='1/(1 + I*omega*tg)')
         .set_action_text('xt*((Dt + mu)*x - J*g*x) - D*xt^2')
         .equation(lhs='(Dt + mu)*x', rhs='0').build())
    r = _cc(m, 0, taus=[0.0, 0.5, 2.0])
    assert 'delay' not in r
    assert r['analysis'].has('G2_KERNEL_STABILITY')
    assert r['analysis'].traits.kernel_stability == 'stable'
    # exact: x = G xi with G = (1 + i w tg) / ((i w + mu)(1 + i w tg) - J)
    from scipy.integrate import quad
    S = lambda w: 0.2 * abs((1 + 1j * w * 0.7) / ((1j * w + 1.0) * (1 + 1j * w * 0.7) - 0.5)) ** 2 / np.pi
    ref = [quad(S, 0, np.inf, limit=500)[0]] + [
        quad(S, 0, np.inf, weight='cos', wvar=t, limlst=200, limit=2000)[0] for t in (0.5, 2.0)]
    got = np.real(r['C_tau_by_ell'][0])
    assert np.allclose(got[1:], ref[1:], rtol=1e-8), (got, ref)
    assert got[0] == pytest.approx(ref[0], rel=2e-6)          # tau = 0 at -1e-6 (Ito)


def test_k3_uses_the_pade_tree_and_converges():
    """k = 3 (a quadratic drift makes it non-zero): no exact tree (k = 2 only),
    the Pade tree converges in n and the record says so."""
    m = (SDE('delay k3 quadratic').parameter('a', default=A_).parameter('c', default=C_)
         .parameter('d', default=0.5, domain='positive').parameter('b', default=0.3)
         .parameter('D', default=D_).field('x').drift('x', '-a*x + c*x(t - d) - b*x^2')
         .noise('xi', Gaussian(var='2*D'), couples={'x': 1}).build())
    pts = [(0.0, 0.3, 0.8), (0.0, 0.5, 0.5)]
    vals = {}
    for n in (2, 4, 8):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            r = compute_cumulants(m, k=3, max_ell=0, external_fields=[('x', 1)] * 3,
                                  parallel=False, verbose=False, use_cache=False,
                                  delay={'order': n})
        assert r['delay']['tree'] == 'pade' and 'k = 3' in r['delay']['tree_note']
        vals[n] = np.array([complex(r['total_C'](*p)) for p in pts])
        assert np.all(np.isfinite(vals[n])) and np.max(np.abs(vals[n])) > 0
    assert np.max(np.abs(vals[8] - vals[4])) < np.max(np.abs(vals[4] - vals[2]))


# ── the state-space realization's a posteriori check (api._realization) ──


def test_aposteriori_check_flags_a_broken_realization():
    from api._realization import _aposteriori_check
    A = np.array([[-1.0, 2.0], [0.0, -3.0]])
    B = np.array([[1.0, 0.0], [0.5, 1.0]])
    C = np.eye(2)
    assert _aposteriori_check(A, B, C) is None
    # a rounding that leaves the matrix non-Hurwitz: the Lyapunov covariance
    # is not positive (and disagrees with the transfer function's integral)
    assert _aposteriori_check(np.array([[1e-3, 2.0], [0.0, -3.0]]), B, C) is not None
    # a wrong output map against the same transfer evaluation: still consistent
    # (the check compares the realization with itself, so only a broken
    # time-domain solve fails it)
    assert _aposteriori_check(A, B, 2 * C) is None


def _hawkes_linear(name, delay_kw):
    return (TemporalModelBuilder(name).population('E', size=2)
            .physical_field('n', population='E').physical_field('v', population='E')
            .parameter('Em', default=[0.8, 0.78], indexed_by=['E'], domain='positive')
            .parameter('tau', default=[10, 9], indexed_by=['E'], domain='positive')
            .parameter('dg', default=2.0, domain='positive')
            .parameter('a', default=[0.44, 0.44], indexed_by=['E'], domain='positive')
            .parameter('w', default=[[0.25, 0.25], [0.2, 0.3]], indexed_by=['E', 'E'], domain='positive')
            .define_function('phi', args=['v'], expression='a[i]*v', population='E')
            .define_kernel('g', **delay_kw)
            .set_action_text('''sum( nt[i]*n[i] - (exp(nt[i])-1)*phi[i](v[i])
                + vt[i]*((tau[i]*Dt + 1)*v[i] - Em[i] - sum(w[i,j]*g*n[j] for j in E)) for i in E)''')
            .set_mf_equation('vstar', '(Em[i] + sum(w[i, j]*g*nstar[j] for j in E))')
            .set_mf_equation('nstar', 'phi[i](vstar[i])').build())


@pytest.mark.slow
def test_graded_pade24_realization_falls_back_to_the_pole_route(monkeypatch):
    """A 4-field model with a Pade-24 kernel on its cycle (the delay route at a
    fixed order, Pade tree): the exact realization is right but its rounded
    form is unusable (the state-space route returned C ~ 1e13 before); it is
    refused ('ill_conditioned'), the run takes the pole route (the pole
    engine's value) and says so."""
    m = _hawkes_linear('delay test graded pade24', dict(delay='dg'))
    kw = dict(k=2, max_ell=0, external_fields=[('n', 1), ('n', 2)], tau_grid=np.array([0.5, 5.0]),
              parallel=False, verbose=False, use_cache=False, delay={'order': 24, 'tree': 'pade'})
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        r = compute_cumulants(m, **kw)
        monkeypatch.setenv('DAEDALUS_PHASE_J_ENGINE', 'poles')
        rp = compute_cumulants(m, **kw)
    assert r['analysis'].has('G6_SS_ILL_CONDITIONED')
    assert 'realization_ill_conditioned' in r['phase_j_engine']['ss_declined_by_reason']
    led = [e for e in r['analysis'].ledger if e.name == 'realization_conditioning']
    assert led and led[0].satisfied is False
    assert np.allclose(np.real(r['C_tau']), np.real(rp['C_tau']), rtol=1e-12)
    assert np.max(np.abs(r['C_tau'])) < 1.0          # (1e13 before the check)


# ── review fixes (regressions) ──────────────────────────────────────────


def _cond_model(name, **kw):
    return (SDE(name).parameter('a', default=1.0).parameter('b', default=1.0)
            .parameter('c', default=0.3).parameter('d', default=0.5, domain='positive')
            .parameter('tau', default=0.4, domain='positive').parameter('D', default=0.05)
            .field('x').define_kernel('g', **kw)
            .drift('x', '-a*x + b - c*x*Conv(g, x)')
            .noise('xi', Gaussian(var='2*D'), couples={'x': 1}).build())


def test_delayed_kernel_in_a_nonlinear_term_keeps_its_delay():
    """A shifted time_expr kernel inside a field x Conv vertex: the loops use
    the Pade image (the unshifted time image never reaches the vertex), equal
    to the same kernel declared with delay= + freq_image."""
    kw = dict(k=2, max_ell=1, external_fields=EXT, tau_grid=np.array([0.0, 0.5, 2.0]),
              parallel=False, verbose=False, use_cache=False, delay={'order': 4})
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        a = compute_cumulants(_cond_model('delay cond A', time_expr='heaviside(t - d)*exp(-(t - d)/tau)/tau'), **kw)
        b = compute_cumulants(_cond_model('delay cond B', delay='d', freq_image='1/(1 + I*omega*tau)'), **kw)
    assert a['status'] == 'ok' and b['status'] == 'ok'
    assert np.allclose(np.real(a['C_tau_by_ell'][1]), np.real(b['C_tau_by_ell'][1]), rtol=1e-10, atol=0)
    assert np.max(np.abs(a['C_tau_by_ell'][1])) > 0


def test_hard_delay_in_a_nonlinear_term_is_never_silent():
    """x * x(t - d): the all-pass image is not strictly proper, Phase J skips
    those diagrams; the record says so (target not met) and warns."""
    m = (SDE('delay hard nonlinear').parameter('a', default=1.0).parameter('b', default=1.0)
         .parameter('c', default=0.3).parameter('d', default=0.5, domain='positive')
         .parameter('D', default=0.05).field('x').drift('x', '-a*x + b - c*x*x(t - d)')
         .noise('xi', Gaussian(var='2*D'), couples={'x': 1}).build())
    from api.delay import DelayOrderWarning
    with warnings.catch_warnings(record=True) as ws:
        warnings.simplefilter('always')
        r = compute_cumulants(m, k=2, max_ell=1, external_fields=EXT, tau_grid=np.array([0.0, 1.0]),
                              parallel=False, verbose=False, use_cache=False)
    assert r['status'] == 'incomplete' and r['delay']['incomplete']
    assert r['delay']['target_met'] is False and 'strictly proper' in r['delay']['cap']
    assert any(issubclass(w.category, DelayOrderWarning) for w in ws)


def test_long_delay_steps_up_the_order():
    """A stable oscillator with a long delayed velocity feedback (d = 13): low
    Pade orders put the tree far from the exact one; the search moves on
    instead of raising, and the tree is exact."""
    m = (SDE('delay long feedback').parameter('g', default=0.5).parameter('k', default=0.45)
         .parameter('d', default=13.0, domain='positive').parameter('D', default=0.05)
         .parameter('eps', default=0.05)
         .field('x').field('v').equation('x', lhs='(Dt + 1)*x', rhs='v + x')
         .drift('v', '-x - g*v - k*v(t - d) - eps*v^3')
         .noise('xi', Gaussian(var='2*D'), couples={'v': 1}).build())
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        r = compute_cumulants(m, k=2, max_ell=1, external_fields=[('x', 1)] * 2,
                              tau_grid=np.array([0.0, 1.0, 2.0]), parallel=False, verbose=False,
                              use_cache=False)
    info = r['delay']
    assert info['tree'] == 'exact' and np.all(np.isfinite(np.real(r['C_tau'])))
    assert info['n'] >= 8 and (info['orders_skipped_unstable'] or info['orders_run'][0] > 4
                               or len(info['orders_run']) > 1)


def test_output_files_hold_the_final_numbers(tmp_path):
    from api.save import load_npz
    p = str(tmp_path / 'delay.npz')
    r = _cc(_ou('delay npz'), 0, delay={'order': 4}, output_npz=p)
    d = load_npz(p)
    tree = [v for k_, v in d.items() if 'tree' in k_.lower() or k_ == 'C_tree']
    assert tree, list(d)
    assert np.allclose(np.real(np.asarray(tree[0])), np.real(r['C_tau_by_ell'][0]), rtol=1e-14)


def test_indexed_delayed_kernel_and_caret_time_expr():
    m = (TemporalModelBuilder('delay indexed kernel').population('E', size=2)
         .physical_field('x', population='E')
         .parameter('d', default=0.5, domain='positive').parameter('D', default=0.1)
         .parameter('w', default=[[0.2, -0.3], [0.25, 0.1]], indexed_by=['E', 'E'])
         .define_kernel('g', delay='d', freq_image='1/(1 + I*omega*0.3)', indexed_by=['E', 'E'])
         .set_action_text('sum(xt[i]*((Dt + 1)*x[i] - sum(w[i,j]*g[i,j]*x[j] for j in E)) - D*xt[i]^2 for i in E)')
         .set_mf_equation('xstar', '0').build())
    rep = analyze(m, Question(k=2, external_fields=(('x', 1), ('x', 2)), max_ell=0))
    assert rep.traits.propagator_class == 'delay' and not rep.errors, rep.summary()
    assert split_shifted_time_expr('heaviside(t - d)*(t - d)*exp(-(t - d)/tau)/tau^2')[0] == 'd'


def test_spatial_delay_is_refused_by_name():
    from api.delay import compute_delayed
    with pytest.raises(ValueError, match='spatial'):
        compute_delayed({'name': 'x', 'spatial': {'dim': 1}}, {'k': 2}, {'z_g': ['d']})


def test_g2k_review_cases_real_roots_aliasing_long_delays():
    """Numerics review: real right-half-plane roots (box edges on the real
    axis), long delays (the contour grid resolves e^{-s d}: no aliasing), and
    the Lambert W path at a d >> 1 (no overflow, enough branches)."""
    om, d, c = SR.var('omega'), SR.var('d'), SR.var('c')
    bp = matrix(SR, [[I * om + 1 - c * exp(-I * om * d) * (1 / (1 + I * om) - 1 / (1 + I * om) ** 2)]])
    ks = kernel_stability(bp, om, {c: 9.5, d: 0.5})
    assert ks.status == 'unstable' and ks.n_rhp == 2       # Pade-12/20 roots: 0.64119, 0.20259
    assert sorted(round(z.real, 5) for z in ks.roots) == [0.20259, 0.64119]
    assert kernel_stability(bp, om, {c: 1.0, d: 0.5}).status == 'stable'
    al = matrix(SR, [[I * om + 1 + 2 * exp(-I * om * d), 0], [0, I * om + 3]])
    assert kernel_stability(al, om, {d: 523.6}).n_rhp == 290
    lw = matrix(SR, [[I * om + 10 + 5 * exp(-I * om * d)]])
    assert kernel_stability(lw, om, {d: 80.0}).status == 'stable'
    lw2 = matrix(SR, [[I * om + 1 + 2 * exp(-I * om * d)]])
    assert kernel_stability(lw2, om, {d: 300.0}).n_rhp == 166
    one = np.array([[1.0]])
    from api._delay_exact import ExactTreeUnavailable
    with pytest.raises(ExactTreeUnavailable, match='not positive'):
        ExactTree(DelayedStateSpace(np.array([[-1.0]]), np.array([[-2.0]]), one, np.array([[0.0]]),
                                    one, np.array([[0.0]]), 1.4, np.array([[0.2]])), 0, 0)
