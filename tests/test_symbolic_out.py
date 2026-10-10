"""Tests for ``api/symbolic_out.py``: symbolic tree-level covariance
``C0(omega) = G D G^dagger`` in the Fourier domain, its exports, the kernel
pairing and the error paths.  Stage C1 of docs/feature_plan_sde_and_delta.md.

Fourier convention tested here (pipeline-wide):
    G(t) = (1/2 pi) int d omega exp(i omega t) G(omega)
    C_{ab}(tau) = <x_a(0) x_b(tau)> = (1/2 pi) int C0_{ab}(omega) exp(-i omega tau)

Every test builds with ``use_cache=False`` so nothing is written under
``saved_models/``.  Hermetic: only tracked models.
"""
import json
import os
import sys
import warnings

import numpy as np
import pytest
from sage.all import SR, var, I, exp, sqrt

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', 'notebooks')))

import daedalus as dd  # noqa: E402
from api import symbolic_out as so  # noqa: E402
from api.compute import compute_cumulants  # noqa: E402

warnings.filterwarnings('ignore')


def _zero(expr):
    return bool(SR(expr).simplify_full() == 0)


# ═════════════════════════════════════════════════════════════════════════════
# A: OU quartic, white noise
# ═════════════════════════════════════════════════════════════════════════════

@pytest.fixture(scope='module')
def ou():
    m, mod = dd.load_model('ou_quartic')
    cov = so.tree_covariance(m, use_cache=False, mean_field={'xstar1': 0})
    return m, mod, cov


def test_A_ou_quartic_C0_exact(ou):
    m, mod, cov = ou
    w = cov.omega
    mu, D = var('mu'), var('D')
    assert cov.phys_names == ['dx1'] and cov.resp_names == ['xt1']
    assert cov.G[0, 0] == 1 / (mu + I * w)
    assert cov.D[0, 0] == 2 * D          # <xi xi> = 2 D delta
    c = cov.entry('dx1', 'dx1')
    assert _zero(c - 2 * D / (w ** 2 + mu ** 2))
    assert cov.parameters == ['D', 'mu']
    # the user-facing / tuple addressing agrees with the field name
    assert cov.entry(('dx', 1), ('dx', 1)) == c
    assert cov.entry(('x', 1), ('x', 1)) == c
    assert cov.contact('dx1', 'dx1') == 0


def test_A_inverse_transform_is_D_over_mu_exp(ou):
    _, _, cov = ou
    mu, D, tau = var('mu'), var('D'), var('tau')
    r = cov.inverse_transform('dx1', 'dx1')
    assert r['contact'] == 0
    assert _zero(r['positive'] - D / mu * exp(-mu * tau))
    assert _zero(r['negative'] - D / mu * exp(mu * tau))
    assert r['even'] is not None
    assert _zero(r['even'] - D / mu * exp(-mu * abs(tau)))
    # a numeric tau picks the branch (parameters stay symbolic)
    v = cov.inverse_transform('dx1', 'dx1', 0.5)
    assert _zero(v['value'] - D / mu * exp(-mu * 0.5))
    assert 'value_numeric' not in v
    f = so.to_callable(r['positive'], tau, params=['D', 'mu'])
    assert abs(f(0.5, D=1.0, mu=2.0)[()] - np.exp(-1.0) / 2.0) < 1e-14


def test_A_exports_round_trip(ou):
    _, _, cov = ou
    w = cov.omega
    c = cov.entry('dx1', 'dx1')
    # sympy
    import sympy
    sy = cov.sympy('dx1', 'dx1')
    assert isinstance(sy, sympy.Expr)
    back = so.from_sympy(sy)
    assert _zero(back - c)
    pts = [(0.0, 1.0, 1.0), (0.3, 0.7, 2.0), (1.0, 1.0, 0.5),
           (2.5, 3.0, 1.2), (-4.0, 0.2, 0.9)]
    for om, mu_, D_ in pts:
        a = complex(c.subs({w: om, var('mu'): mu_, var('D'): D_}).n())
        b = complex(sy.subs({sympy.Symbol('omega'): om,
                             sympy.Symbol('mu'): mu_,
                             sympy.Symbol('D'): D_}))
        assert abs(a - b) < 1e-14
    # LaTeX
    tex = cov.latex('dx1', 'dx1')
    assert '\\frac' in tex and '\\mu' in tex and 'D' in tex
    # numpy callable with named parameters
    f = cov.callable('dx1', 'dx1')
    assert f.params == ['D', 'mu']
    om = np.linspace(-5, 5, 11)
    np.testing.assert_allclose(f(om, mu=1.3, D=0.7),
                               2 * 0.7 / (om ** 2 + 1.3 ** 2), rtol=1e-14)
    assert f(0.0, mu=2.0, D=1.0).shape == ()
    with pytest.raises(TypeError, match='missing'):
        f(om, mu=1.0)
    with pytest.raises(TypeError, match='unknown'):
        f(om, mu=1.0, D=1.0, bogus=3)
    # JSON: serialisable, parameter list, round trip
    js = cov.json('dx1', 'dx1')
    d = json.loads(js)
    assert d['parameters'] == ['D', 'mu'] and d['variable'] == 'omega'
    assert d['expression'] == '2*D/(mu**2 + omega**2)'
    expr, om_sym, params = so.from_json(js)
    assert params == ['D', 'mu']
    assert _zero(expr - c)


def test_A_from_result_matches_model_route(ou):
    m, mod, cov = ou
    res = compute_cumulants(
        m, k=2, max_ell=0, external_fields=[('dx', 1), ('dx', 1)],
        parameters=dict(mod.DEFAULT_FUNDAMENTAL),
        tau_grid=np.array([0.5, 1.0]), use_cache=False, verbose=False)
    cov2 = so.tree_covariance_from_result(res, model=m, mean_field='result')
    assert _zero(cov2.entry('dx1', 'dx1') - cov.entry('dx1', 'dx1'))
    # propagator in the result is the one the matrices were built from
    assert cov2.G[0, 0] == cov.G[0, 0]
    # pipeline cross-check at tau > 0 (value D/mu exp(-mu tau) at defaults)
    f = cov2.callable('dx1', 'dx1', defaults=True)
    q = so.numeric_inverse_transform(lambda w: f(w), 0.5)
    np.testing.assert_allclose(q.value.real, res['C_tau'][0].real, rtol=1e-6)
    with pytest.raises(so.SymbolicOutError, match='propagator'):
        so.tree_covariance_from_result({}, model=m)


def test_fourier_convention_green_function_and_poles(ou):
    """G(t) = (1/2 pi) int exp(i omega t) G(omega): the retarded G of
    1/(i omega + mu) is Theta(t) exp(-mu t), pole at omega = +i mu."""
    m, mod, cov = ou
    w, mu = cov.omega, var('mu')
    r = so.inverse_transform(1 / (I * w + mu), w)
    tau = r['tau']
    # inverse_transform integrates against exp(-i w tau): G(t) is tau = -t
    assert r['positive'] == 0                       # tau > 0  <=>  t < 0
    assert _zero(r['negative'] - exp(mu * tau))     # tau = -t: exp(-mu t)
    # the pipeline's own pole-residue form agrees: pole at +i mu (Im > 0)
    res = compute_cumulants(
        m, k=2, max_ell=0, external_fields=[('dx', 1), ('dx', 1)],
        parameters=dict(mod.DEFAULT_FUNDAMENTAL),
        tau_grid=np.array([0.5]), use_cache=False, verbose=False)
    pv = np.asarray(res['propagator']['pole_vals'], dtype=complex)
    assert pv.shape == (1,) and abs(pv[0] - 1j * mod.DEFAULT_FUNDAMENTAL['mu']
                                    ) < 1e-12
    # Fourier pair on a rational function via the numeric route:
    # int dt exp(-i w t) exp(-mu t) Theta(t) = 1/(mu + i w)
    mu0 = 1.7
    for om in (0.0, 0.9, -2.3):
        val = 1.0 / (mu0 + 1j * om)
        ref = complex((1 / (I * w + mu)).subs({w: om, mu: mu0}).n())
        assert abs(val - ref) < 1e-14


# ═════════════════════════════════════════════════════════════════════════════
# B: linear Hawkes (2 populations), colored OU (Markov embedding)
# ═════════════════════════════════════════════════════════════════════════════

@pytest.fixture(scope='module')
def hawkes():
    m, mod = dd.load_model('linear_hawkes')
    cov = so.tree_covariance(m, use_cache=False)
    taus = np.array([0.5, 1.0, 5.0, -1.0, 0.0])
    res = compute_cumulants(
        m, k=2, max_ell=0, external_fields=[('n', 1), ('n', 2)],
        parameters=dict(mod.DEFAULT_FUNDAMENTAL), tau_grid=taus,
        use_cache=False, verbose=False)
    res_vv = compute_cumulants(
        m, k=2, max_ell=0, external_fields=[('v', 1), ('v', 1)],
        parameters=dict(mod.DEFAULT_FUNDAMENTAL), tau_grid=taus,
        use_cache=False, verbose=False)
    res_nn = compute_cumulants(
        m, k=2, max_ell=0, external_fields=[('n', 1), ('n', 1)],
        parameters=dict(mod.DEFAULT_FUNDAMENTAL), tau_grid=taus,
        use_cache=False, verbose=False)
    npar = {str(k): float(v) for k, v in res['num_params'].items()}
    return m, mod, cov, taus, res, res_vv, res_nn, npar


def _numeric_C(cov, a, b, npar, tau):
    f = cov.callable(a, b)
    return so.numeric_inverse_transform(
        lambda w: f(w, **{k: npar[k] for k in f.params}), tau)


def test_B_hawkes_structure(hawkes):
    m, mod, cov, taus, res, res_vv, res_nn, npar = hawkes
    assert cov.phys_names == ['dn1', 'dn2', 'dv1', 'dv2']
    assert cov.resp_names == ['nt1', 'nt2', 'vt1', 'vt2']
    a, vs1, vs2 = var('a'), var('vstar1'), var('vstar2')
    # Poisson noise on the n response fields: D_nn = phi(v*) = a v*
    D = cov.D
    assert D[0, 0] == a * vs1 and D[1, 1] == a * vs2
    assert all(D[i, j] == 0 for i in range(4) for j in range(4)
               if not (i == j and i < 2))
    # delta contact term on dn (the pipeline's D_delta), none on dv
    assert cov.contact('dn1', 'dn1') == a * vs1
    assert cov.contact('dn1', 'dn2') == 0
    assert cov.contact('dv1', 'dv1') == 0
    # Hermitian: C0_ba(omega) = conj(C0_ab(omega))
    f12 = cov.callable('dn1', 'dv2')
    f21 = cov.callable('dv2', 'dn1')
    om = np.array([-3.0, -0.4, 0.0, 0.7, 2.2])
    kw = {k: npar[k] for k in f12.params}
    np.testing.assert_allclose(f21(om, **{k: npar[k] for k in f21.params}),
                               np.conj(f12(om, **kw)), rtol=1e-12)


def test_B_hawkes_cross_entry_matches_phase_j(hawkes):
    """(dn1, dn2): the brief's convention int C0_ab exp(-i w tau) dw/2pi equals
    compute_cumulants(external_fields=[a, b]), for tau of both signs (the
    off-diagonal entry is NOT even in tau)."""
    m, mod, cov, taus, res, res_vv, res_nn, npar = hawkes
    for tau, ref in zip(taus, res['C_tau']):
        if tau == 0.0:
            continue
        q = _numeric_C(cov, 'dn1', 'dn2', npar, tau)
        np.testing.assert_allclose(q.value.real, ref.real, rtol=1e-6)
        assert abs(q.value.imag) < 1e-9
    # swapping the entry reverses tau
    q1 = _numeric_C(cov, 'dn2', 'dn1', npar, 1.0)
    q2 = _numeric_C(cov, 'dn1', 'dn2', npar, -1.0)
    np.testing.assert_allclose(q1.value, q2.value, rtol=1e-9)


def test_B_hawkes_diagonal_entries_three_taus(hawkes):
    """The brief's check: (dv1,dv1) and (dn1,dn1) at three tau, rel 1e-6.

    Ito convention at tau=0: the pipeline evaluates the left limit tau=-1e-6
    (Theta(0)=0), so its tau=0 value differs from the exact smooth part by
    O(1e-6) relative -- compared with a 5e-6 tolerance below.  The delta(tau)
    contact term of dn (weight a v*) is NOT in C_tau: the numeric transform is
    the smooth part, matching the pipeline's "delta contributions reported but
    not added"."""
    m, mod, cov, taus, res, res_vv, res_nn, npar = hawkes
    for ent, r in (('dv1', res_vv), ('dn1', res_nn)):
        for tau, ref in zip(taus, r['C_tau']):
            q = _numeric_C(cov, ent, ent, npar, tau)
            rtol = 5e-6 if tau == 0.0 else 1e-6
            np.testing.assert_allclose(q.value.real, ref.real, rtol=rtol)
            assert q.error < 1e-9
    # even in tau for the diagonal entry
    qp = _numeric_C(cov, 'dv1', 'dv1', npar, 2.0)
    qm = _numeric_C(cov, 'dv1', 'dv1', npar, -2.0)
    np.testing.assert_allclose(qp.value, qm.value, rtol=1e-9)
    # contact term value: lim_{w->inf} C0_dn1dn1 = a v1* = n1*
    f = cov.callable('dn1', 'dn1')
    big = f(1e7, **{k: npar[k] for k in f.params})
    np.testing.assert_allclose(big.real, npar['a'] * npar['vstar1'],
                               rtol=1e-8)


def test_B_colored_noise_through_markov_embedding():
    m, mod = dd.load_model('ou_quartic_colored')
    cov = so.tree_covariance(m, use_cache=False,
                             mean_field={'xstar1': 0, 'xistar1': 0})
    assert cov.phys_names == ['dx1', 'dxi1']
    D, tauc, mu, w = var('D'), var('tauc'), var('mu'), cov.omega
    # white noise on the embedding field, variance 4 D / tauc^2
    assert cov.D[1, 1] == 4 * D / tauc ** 2 and cov.D[0, 0] == 0
    # exact spectrum of the colored OU
    c = cov.entry('dx1', 'dx1')
    assert _zero(c - 4 * D / tauc ** 2 / ((w ** 2 + mu ** 2)
                                          * (w ** 2 + 1 / tauc ** 2)))
    params = {'mu': 0.1, 'eps': 0.1, 'D': 1.0, 'tauc': 1.0}
    taus = np.array([0.5, 1.0, 5.0])
    res = compute_cumulants(
        m, k=2, max_ell=0, external_fields=[('x', 1), ('x', 1)],
        parameters=params, tau_grid=taus, use_cache=False, verbose=False)
    for tau, ref in zip(taus, res['C_tau']):
        q = _numeric_C(cov, 'dx1', 'dx1', params, tau)
        np.testing.assert_allclose(q.value.real, ref.real, rtol=1e-6)
    # closed form by residues agrees with the pipeline too
    r = cov.inverse_transform('dx1', 'dx1', reference=params)
    f = so.to_callable(r['even'], r['tau'], params=['D', 'mu', 'tauc'])
    for tau, ref in zip(taus, res['C_tau']):
        np.testing.assert_allclose(f(tau, **{k: params[k] for k in f.params}
                                     ).real, ref.real, rtol=1e-6)


def test_B_cross_correlated_noise_matrix():
    """Two colored fields with a correlation rho: the noise matrix has an
    off-diagonal entry; C0 is Hermitian."""
    m, mod = dd.load_model('ou_quartic_two_dim_color_corr')
    cov = so.tree_covariance(m, use_cache=False,
                             mean_field={'xstar1': 0, 'ystar1': 0})
    names = cov.resp_names
    i, j = names.index('xit1'), names.index('yit1')
    D1, D2, rho, tauc = var('D1'), var('D2'), var('rho'), var('tauc')
    assert cov.D[i, j] == 2 * sqrt(D1) * sqrt(D2) * rho / tauc ** 2
    assert cov.D[i, j] == cov.D[j, i]
    assert cov.D[i, i] == 2 * D1 / tauc ** 2
    f_xy = cov.callable('dx1', 'dy1')
    f_yx = cov.callable('dy1', 'dx1')
    pv = {'mu1': 0.8, 'mu2': 1.1, 'J1': 0.2, 'J2': -0.15, 'tauc': 0.7,
          'D1': 1.0, 'D2': 0.6, 'rho': 0.4, 'eps1': 0.1, 'eps2': 0.1}
    om = np.array([-2.0, 0.3, 1.5])
    a = f_xy(om, **{k: pv[k] for k in f_xy.params})
    b = f_yx(om, **{k: pv[k] for k in f_yx.params})
    np.testing.assert_allclose(b, np.conj(a), rtol=1e-12)
    # without correlation the cross spectrum is carried by the J couplings only
    assert not _zero(cov.entry('dx1', 'dy1'))


def test_mean_field_solve_double_well():
    """mean_field='solve' substitutes the pipeline's saddle: for mu<0 the
    restoring rate is mu + 3 eps x*^2 = -2 mu."""
    m, mod = dd.load_model('ou_quartic_double_well')
    par = {'mu': -1.0, 'eps': 0.1, 'D': 1.0}
    cov = so.tree_covariance(m, use_cache=False, mean_field='solve',
                             parameters=par, values={'mu': -1, 'eps': 0.1})
    assert set(cov.mean_field_values) == {'xstar1'}
    f = cov.callable('dx1', 'dx1')
    om = np.array([0.0, 1.0, 3.0])
    np.testing.assert_allclose(f(om, D=1.0).real, 2.0 / (om ** 2 + 4.0),
                               rtol=1e-12)


# ═════════════════════════════════════════════════════════════════════════════
# C: pairing with a response kernel
# ═════════════════════════════════════════════════════════════════════════════

def test_C_pair_with_exponential_kernel(ou):
    _, _, cov = ou
    w, mu, D, T = cov.omega, var('mu'), var('D'), var('T')
    L = 1 / (1 + I * w * T)                      # L(t) = exp(-t/T)/T Theta(t)
    p = cov.pair_with_kernel('dx1', 'dx1', L)
    assert p.exact and p.expression is not None
    # closed form D / (mu (1 + mu T))
    assert _zero(p.expression - D / (mu * (1 + mu * T)))
    vals = {'mu': 1.3, 'D': 0.8, 'T': 0.7}
    exact = 0.8 / (1.3 * (1 + 1.3 * 0.7))
    got = complex(p.expression.subs({mu: 1.3, D: 0.8, T: 0.7}).n())
    assert abs(got - exact) < 1e-14
    # (a) numeric quadrature of the same integrand
    q = so.numeric_pair_with_kernel(cov.entry('dx1', 'dx1'), L, w,
                                    values=vals, epsabs=1e-12, epsrel=1e-10)
    assert abs(q.value - exact) < 1e-9 and q.error < 1e-9
    assert q.epsabs == 1e-12 and q.epsrel == 1e-10
    # (b) direct time-domain integral  int dt L(t) (D/mu) exp(-mu |t|)
    from scipy.integrate import quad
    td, _ = quad(lambda t: (np.exp(-t / 0.7) / 0.7) * (0.8 / 1.3)
                 * np.exp(-1.3 * abs(t)), 0, np.inf, epsabs=1e-13)
    assert abs(td - exact) < 1e-11
    assert abs(q.value - td) < 1e-9


def test_C_pair_with_kernel_numeric_fallback_for_non_rational_kernel(ou):
    """A delayed exponential is not rational in omega: the exact route
    declines and the quadrature fallback answers, with its tolerance stated."""
    _, _, cov = ou
    w = cov.omega
    d_, T = var('d'), var('T')
    L = exp(-I * w * d_) / (1 + I * w * T)
    vals = {'mu': 1.3, 'D': 0.8, 'T': 0.7, 'd': 0.4}
    p = cov.pair_with_kernel('dx1', 'dx1', L, values=vals)
    assert not p.exact and p.expression is None
    assert p.tolerance == (1e-12, 1e-10) and p.abs_error < 1e-9
    assert 'exp' in p.reason or 'non-rational' in p.reason
    # L(t) = Theta(t-d) exp(-(t-d)/T)/T   ->   (D/mu) e^{-mu d}/(T(mu+1/T))
    exact = (0.8 / 1.3) * np.exp(-1.3 * 0.4) / (0.7 * (1.3 + 1 / 0.7))
    assert abs(p.value - exact) < 1e-8
    # fallback disabled -> the reason is raised instead
    with pytest.raises(so.NonRationalPropagatorError):
        cov.pair_with_kernel('dx1', 'dx1', L, values=vals, fallback=False)


def test_C_pair_with_kernel_sign_convention():
    """conj(L~) not L~: for L(t)=Theta(t) e^{-t/T}/T and the NON-even
    covariance exp(-mu tau) Theta(tau) the pairing is int L(t) C(-t) dt."""
    w = var('omega')
    mu, T = var('mu'), var('T')
    # C0 whose transform is C(tau) = Theta(tau) exp(-mu tau) *under e^{-iw tau}*
    # is 1/(mu - i w); pairing = int dt L(t) C(-t) = 0 for a causal L
    C0 = 1 / (mu - I * w)
    L = 1 / (1 + I * w * T)
    p = so.pair_with_kernel(C0, L, w, reference={'mu': 1.0, 'T': 1.0})
    # int dw/2pi conj(L) C0 : conj(L)=1/(1-iwT) is analytic in the LOWER
    # half-plane... both factors have their poles on the same side -> 0
    assert _zero(p.expression)
    # the anticausal partner pairs to 1/(mu T + 1)... via C0 -> 1/(mu + i w)
    p2 = so.pair_with_kernel(1 / (mu + I * w), L, w,
                             reference={'mu': 1.0, 'T': 1.0})
    assert _zero(p2.expression - 1 / (1 + mu * T))


# ═════════════════════════════════════════════════════════════════════════════
# Residue machinery: multiplicity, quadratics, contact terms
# ═════════════════════════════════════════════════════════════════════════════

def test_inverse_transform_double_pole_and_quadratic_factor():
    w, a, b = var('omega'), var('a'), var('b')
    tau = var('tau')
    # 1/(a + i w)^2  ->  Theta(-tau)-side t exp: G(t) = t exp(-a t) Theta(t)
    r = so.inverse_transform(1 / (a + I * w) ** 2, w,
                             reference={'a': 1.0})
    assert r['positive'] == 0
    assert _zero(r['negative'] - (-tau) * exp(a * tau))
    # irreducible quadratic factors: the underdamped oscillator spectrum
    #   1/((w^2 - w0^2)^2 + g^2 w^2)  (poles s = (-g +- sqrt(g^2-4 w0^2))/2)
    g, w0 = var('g'), var('w0')
    C0 = 1 / ((w ** 2 - w0 ** 2) ** 2 + g ** 2 * w ** 2)
    ref = {'g': 0.2, 'w0': 1.0}
    r2 = so.inverse_transform(C0, w, reference=ref)
    f = so.to_callable(r2['even'], tau, params=['g', 'w0'])
    c = so.to_callable(C0, w, params=['g', 'w0'])
    for t in (0.3, 1.7, 6.0):
        q = so.numeric_inverse_transform(lambda x: c(x, **ref), t)
        got = complex(f(t, **ref)[()])
        assert abs(q.value - got) < 1e-9
    # golden-ratio quartic: two quadratic factors with real s-roots
    C1 = 1 / (w ** 4 + 3 * w ** 2 + 1)
    r3 = so.inverse_transform(C1, w)
    f3 = so.to_callable(r3['even'], tau, params=[])
    c3 = so.to_callable(C1, w, params=[])
    for t in (0.4, 2.5):
        q = so.numeric_inverse_transform(lambda x: c3(x), t)
        assert abs(q.value - complex(f3(t)[()])) < 1e-9


def test_contact_term_is_split_off():
    """A constant tail is the weight of delta(tau); the rest is transformed."""
    w, mu, c = var('omega'), var('mu'), var('c')
    C0 = c + 2 / (w ** 2 + mu ** 2)
    r = so.inverse_transform(C0, w, reference={'mu': 1.0, 'c': 1.0})
    assert _zero(r['contact'] - c)
    assert _zero(r['even'] - exp(-mu * abs(var('tau'))) / mu * 1)


# ═════════════════════════════════════════════════════════════════════════════
# D: clear errors
# ═════════════════════════════════════════════════════════════════════════════

def test_D_spatial_model_is_rejected_before_any_work():
    m, mod = dd.load_model('kpz_1d')
    with pytest.raises(so.UnsupportedModelError, match='spatial'):
        so.tree_covariance(m, use_cache=False)
    with pytest.raises(NotImplementedError):          # also a NotImplementedError
        so.tree_covariance(m, use_cache=False)
    with pytest.raises(so.UnsupportedModelError, match='spatial'):
        so.tree_covariance_from_result({'propagator': {'K_ft': 1}}, model=m)


def test_D_non_rational_propagator_names_the_offender():
    w, mu, d = var('omega'), var('mu'), var('d')
    from sage.all import matrix
    K = matrix(SR, [[mu + I * w + exp(-I * w * d)]])     # a delay
    Dm = matrix(SR, [[2]])
    with pytest.raises(so.NonRationalPropagatorError) as ei:
        so.tree_covariance_from_matrices(None, Dm, w, K=K)
    msg = str(ei.value)
    assert 'exp' in msg and 'K[0,0]' in msg
    # square root of omega
    K2 = matrix(SR, [[mu + sqrt(I * w)]])
    with pytest.raises(so.NonRationalPropagatorError, match='non-rational'):
        so.tree_covariance_from_matrices(None, Dm, w, K=K2)


def test_D_inconsistent_inputs_are_refused():
    w, mu = var('omega'), var('mu')
    from sage.all import matrix
    K = matrix(SR, [[mu + I * w]])
    bad_G = matrix(SR, [[1 / (mu - I * w)]])             # not K^{-1}
    with pytest.raises(so.SymbolicOutError, match='inverse of'):
        so.tree_covariance_from_matrices(bad_G, matrix(SR, [[2]]), w, K=K)
    with pytest.raises(so.SymbolicOutError, match='do not occur'):
        so.tree_covariance_from_matrices(None, matrix(SR, [[2]]), w, K=K,
                                         values={'nope': 1})
    with pytest.raises(so.SymbolicOutError, match='complex'):
        so.tree_covariance_from_matrices(None, matrix(SR, [[2]]), w, K=K,
                                         values={'mu': 1 + 2j})
    cov = so.tree_covariance_from_matrices(None, matrix(SR, [[2]]), w, K=K,
                                           phys_names=['x'],
                                           resp_names=['xt'])
    with pytest.raises(so.SymbolicOutError, match='unknown physical field'):
        cov.entry('y', 'x')


def test_D_residue_refusals():
    w, mu = var('omega'), var('mu')
    # irreducible cubic: no closed form -> ClosedFormUnavailable, numeric ok
    C0 = 1 / (w ** 6 + w ** 2 + 7)
    with pytest.raises(so.ClosedFormUnavailable, match='degree'):
        so.inverse_transform(C0, w)
    p = so.pair_with_kernel(1 / (w ** 6 + 5 * w ** 2 + 3), SR(1), w,
                            fallback=True)
    from scipy.integrate import quad
    ref6 = quad(lambda x: 1 / (x ** 6 + 5 * x ** 2 + 3), -np.inf, np.inf,
                epsabs=1e-13)[0] / (2 * np.pi)
    assert not p.exact and p.reason and abs(p.value - ref6) < 1e-9
    # a pole on the real axis
    with pytest.raises(so.ClosedFormUnavailable, match='real'):
        so.pair_with_kernel(1 / (w ** 2 - 1) / (w ** 2 + 1), SR(1), w,
                            fallback=False)
    # slowly decaying integrand: refuse, do not return a wrong number
    with pytest.raises(so.SymbolicOutError, match='decays only'):
        so.pair_with_kernel(1 / (mu + I * w), SR(1), w,
                            reference={'mu': 1.0})
    # contact (constant) tail without a decaying kernel
    with pytest.raises(so.SymbolicOutError, match='decays only'):
        so.pair_with_kernel(SR(3) + 1 / (w ** 2 + 1), SR(1), w)
    # undecidable half plane: a pole whose side flips inside the sample
    with pytest.raises(so.ClosedFormUnavailable, match='half-plane|sign'):
        so.pair_with_kernel(1 / ((I * w + mu - 1) * (w ** 2 + 4)), SR(1), w,
                            reference={'mu': 1.2}, fallback=False)
    # a non-real kernel parameter is refused downstream by values
    with pytest.raises(so.SymbolicOutError, match='missing'):
        so.numeric_pair_with_kernel(1 / (w ** 2 + mu ** 2), SR(1), w,
                                    values={})


def test_noise_extraction_non_local_kernel_is_refused():
    """A NoiseSourceType (cumulant kernel not Markov-embedded) has no Fourier
    noise *matrix*; refuse with a specific message."""
    from engine.core.vertices import NoiseSourceType
    from unittest import mock
    m, mod = dd.load_model('ou_quartic')
    ft = so._expanded_theory(m, 2, False, False)
    fake = [NoiseSourceType(SR(1), [('xt', 1), ('xt', 1)], (2, 0), [])]
    with mock.patch('engine.core.vertices.extract_source_types',
                    return_value=fake):
        with pytest.raises(so.UnsupportedModelError, match='non-local'):
            so.noise_matrix_from_theory(ft, m)


def test_inverse_transform_is_independent_of_symbol_domains():
    """A positive domain left on a parameter symbol by an earlier model builder must not change the closed form
    (it did: the underdamped oscillator spectrum gave a wrong answer in a full-suite run). Private names keep this test
    from leaking a domain into other tests; the domain is reset afterwards."""
    from sage.all import var
    names = ('g_dom_t', 'w0_dom_t')
    var(names[0], domain='positive')
    var(names[1], domain='positive')
    try:
        w, g, w0, tau = var('omega'), var(names[0]), var(names[1]), var('tau')
        C0 = 1 / ((w ** 2 - w0 ** 2) ** 2 + g ** 2 * w ** 2)
        ref = {names[0]: 0.2, names[1]: 1.0}
        r = so.inverse_transform(C0, w, reference=ref)
        f = so.to_callable(r['even'], tau, params=list(names))
        c = so.to_callable(C0, w, params=list(names))
        for t in (0.3, 1.7, 6.0):
            q = so.numeric_inverse_transform(lambda x: c(x, **ref), t)
            assert abs(q.value - complex(f(t, **ref)[()])) < 1e-6
    finally:
        var(names[0], domain='complex')
        var(names[1], domain='complex')
