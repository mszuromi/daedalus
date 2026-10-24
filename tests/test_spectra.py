"""
tests/test_spectra.py
=====================
Item 70 (G6): exact spectral and integrated outputs of k = 2 results
(``api/spectra.py``; ``compute_cumulants(spectra=...)``, ``dd.Config(spectra=)``,
``dd.spectra(result, ...)``, YAML ``question.spectra``).

* OU: the exact Lorentzian ``S = 2D / (ω² + μ²)``, ``∫_0^T C``, ``τ_c = 1/μ``;
* OU quartic at one and two loops against a fine Gauss-Legendre quadrature of
  the package's own ``C_tau`` / ``total_C_by_ell`` (~1e-10 asked, ~1e-15 seen);
* a two-field cross-spectrum: ``H(iω) Q H(iω)ᴴ``, conjugation and
  ``τ → −τ`` symmetries, and a nonlinear two-field model at one loop;
* shot noise (linear and quadratic-rate Hawkes): the δ at τ = 0 (the flat
  floor of ``S``, ``delta0``), ``S(0) = ∫_0^∞ + ∫_{−∞}^0 + delta0``;
* ``S(0) = 2 ∫_0^∞ C`` for symmetric autocorrelations;
* adversarial near-degenerate and exactly repeated poles (colour noise with
  the filter rate 1e-9 from the relaxation rate, and equal), against
  50-digit references;
* the quadrature fallback (pole engine) with error estimates, refusals, the
  large-block route, the delay route, saving and the YAML config.

Inline public models or tracked ones only.
"""
import os
import sys
import warnings

import numpy as np
import pytest

pytest.importorskip('sage.all')

HERE = os.path.dirname(__file__)
ROOT = os.path.join(HERE, '..')
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'notebooks'))

import daedalus as dd                                                # noqa: E402
from api import compute_cumulants                                    # noqa: E402
from api import me_output as ME                                      # noqa: E402
from api import spectra as SP                                        # noqa: E402
from api.noise import Gaussian, Poisson                              # noqa: E402
from api.sde import SDE                                              # noqa: E402
import engine.integration.time_domain.statespace as SS               # noqa: E402

OM = np.array([-3.0, -0.4, 0.0, 0.25, 1.0, 6.0])
TS = np.array([0.0, 1e-3, 0.3, 1.0, 4.0, 25.0])


@pytest.fixture(autouse=True)
def _default_engine(monkeypatch):
    monkeypatch.delenv(SS.ENGINE_ENV, raising=False)


def _cc(model, ext, ell, spectra=None, taus=np.array([0.0]), **kw):
    kw.setdefault('use_cache', False)
    return compute_cumulants(model, k=2, max_ell=ell, external_fields=ext,
                             tau_grid=taus, parallel=False, verbose=False,
                             spectra=spectra, **kw)


def _rel(a, b):
    a, b = np.asarray(a, dtype=complex), np.asarray(b, dtype=complex)
    return float(np.max(np.abs(a - b)) / max(np.max(np.abs(b)), 1e-300))


def _params(res):
    return {str(k): float(v) for k, v in res['num_params'].items()}


def _ou():
    return (SDE('spectra OU').parameter('mu', default=1.3)
            .parameter('D', default=0.7).field('x')
            .drift('x', '-mu*x')
            .noise('xi', Gaussian(var='2*D'), couples={'x': 1})).build()


def _rotation():
    return (SDE('spectra underdamped OU').parameter('g', default=0.2)
            .parameter('w0', default=1.0).parameter('D', default=1.0)
            .field('x').field('y')
            .drift('x', '-g*x - w0*y').drift('y', 'w0*x - g*y')
            .noise('xi', Gaussian(var='2*D'), couples={'x': 1})).build()


def _coupled_quartic():
    return (SDE('spectra coupled quartic').parameter('mu', default=1.0)
            .parameter('nu', default=2.0).parameter('k', default=0.5)
            .parameter('eps', default=0.05).parameter('D', default=1.0)
            .field('x').field('y')
            .drift('x', '-mu*x + k*y - eps*x^3')
            .drift('y', '-nu*y + k*x - eps*x^2*y')
            .noise('xi', Gaussian(var='2*D'), couples={'x': 1})
            .noise('zeta', Gaussian(var='D'), couples={'y': 1})).build()


def _hawkes(quadratic=False):
    s = (SDE('spectra Hawkes' + (' quadratic' if quadratic else ''))
         .parameter('Em', default=0.5).parameter('tau', default=2.0)
         .parameter('a', default=1.0).parameter('b', default=0.1)
         .parameter('w', default=0.4 if not quadratic else 0.3)
         .define_function('phi', args=['v'],
                          expression='a*v + b*v^2' if quadratic else 'a*v')
         .noise('spikes', Poisson(rate='phi(v)'), expose='n')
         .field('v').equation('v', lhs='(tau*Dt + 1)*v', rhs='Em + w*n'))
    return s.build()


def _colour(gap):
    """x' = -mu x + y - eps x^3, y' = -b y + noise, b = mu (1 + gap)."""
    return (SDE('spectra colour OU').parameter('mu', default=1.0)
            .parameter('b', default=1.0 + gap).parameter('eps', default=0.02)
            .parameter('D', default=1.0).field('x').field('y')
            .drift('x', '-mu*x + y - eps*x^3').drift('y', '-b*y')
            .noise('xi', Gaussian(var='2*D'), couples={'y': 1})).build()


# ═════════════════════════════════════════════════════════════════════════════
# A. exact closed forms
# ═════════════════════════════════════════════════════════════════════════════

def test_ou_exact_lorentzian():
    res = _cc(_ou(), [('x', 1)] * 2, 0, {'omega': OM, 'T': TS})
    sp = res['spectra']
    p = _params(res)
    mu, D = p['mu'], p['D']
    assert sp['status'] == 'ok' and sp['method_by_ell'] == {0: 'exact'}
    assert _rel(sp['S'], 2 * D / (OM ** 2 + mu ** 2)) <= 1e-14
    assert abs(sp['S0'] - 2 * D / mu ** 2) <= 1e-14 * 2 * D / mu ** 2
    I_T = D / mu ** 2 * -np.expm1(-mu * TS)        # no cancellation at small T
    assert _rel(sp['integral'], I_T) <= 1e-14
    assert _rel(sp['integral_neg'], I_T) <= 1e-14
    assert sp['integral'][0] == 0
    assert abs(sp['integral'][1] - I_T[1]) <= 1e-14 * I_T[1]   # T = 1e-3
    assert abs(sp['C0'] - D / mu) <= 1e-15
    assert abs(sp['tau_c'] - 1 / mu) <= 1e-14
    assert sp['delta0'] == 0 and sp['abs_error'] is None
    assert np.max(np.abs(np.imag(sp['S']))) <= 1e-16


def test_long_windows_exact_and_quadrature():
    """Windows of many correlation times: the exact route returns ∫_0^∞
    once e^{GT} has decayed (no expm of a huge G T: it stalled at T = 1e40,
    NaN at 1e300); the quadrature route takes ∫_0^∞ − ∫_T^∞ (QUADPACK on a
    long [0, T] never sampled C and reported 0 ± 0)."""
    res = _cc(_ou(), [('x', 1)] * 2, 0)
    p = _params(res)
    mu, D = p['mu'], p['D']
    Ts = np.array([0.5, 30.0, 1e4, 1e5, 1e6, 1e40, 1e300])
    ref = D / mu ** 2 * -np.expm1(-mu * Ts)
    ex = dd.spectra(res, None, Ts, method='exact')
    assert _rel(ex['integral'], ref) <= 1e-14
    assert _rel(ex['integral_neg'], ref) <= 1e-14
    q = dd.spectra(res, None, Ts[:5], method='quad')
    assert _rel(q['integral'], ref[:5]) <= 1e-10
    assert _rel(q['integral_neg'], ref[:5]) <= 1e-10
    err = q['abs_error']['integral']
    assert np.all(np.abs(q['integral'] - ref[:5]) <= np.maximum(10 * err,
                                                                1e-12))


def test_spectra_true_and_config_route():
    """``spectra=True``: S(0), the integrals to infinity and tau_c only; the
    same numbers through dd.Config / dd.run and post hoc dd.spectra."""
    res = _cc(_ou(), [('x', 1)] * 2, 0, True)
    sp = res['spectra']
    assert sp['omega'] is None and sp['S'] is None and sp['T'] is None
    assert sp['integral'] is None and sp['S_by_ell'] == {}
    cfg = dd.Config(k=2, max_ell=0, external_fields=[('x', 1)] * 2,
                    tau_grid=np.array([0.0]), use_cache=False,
                    spectra={'omega': (0.0, 2.0, 5), 'T': 2.0})
    r2 = dd.run(_ou(), cfg)
    s2 = r2['spectra']
    assert s2['S0'] == sp['S0'] and s2['tau_c'] == sp['tau_c']
    assert np.array_equal(s2['omega'], np.linspace(0.0, 2.0, 5))
    post = dd.spectra(r2, (0.0, 2.0, 5), 2.0)
    assert np.array_equal(post['S'], s2['S'])
    assert np.array_equal(post['integral'], s2['integral'])


# ═════════════════════════════════════════════════════════════════════════════
# B. loops against a fine quadrature of the package's own C(τ)
# ═════════════════════════════════════════════════════════════════════════════

def _gl_panels(L=40.0, npan=8, ng=24):
    x, w = np.polynomial.legendre.leggauss(ng)
    edges = np.linspace(0.0, L, npan + 1)
    u = np.concatenate([0.5 * (b - a) * x + 0.5 * (a + b)
                        for a, b in zip(edges[:-1], edges[1:])])
    wu = np.concatenate([0.5 * (b - a) * w
                         for a, b in zip(edges[:-1], edges[1:])])
    return u, wu


@pytest.mark.parametrize('ell', [1, 2])
def test_ou_quartic_loops_against_fine_quadrature(ell):
    u, wu = _gl_panels()
    grid = np.concatenate([-u[::-1], u])
    om = np.array([0.0, 0.7, 2.5])
    Ts = np.array([0.3, 1.0, 4.0])
    res = _cc(dd.load_model('ou_quartic')[0], [('dx', 1)] * 2, ell,
              {'omega': om, 'T': Ts}, taus=grid)
    sp = res['spectra']
    assert set(sp['method_by_ell'].values()) == {'exact'}
    xg, wg = np.polynomial.legendre.leggauss(30)
    tot = {'S': 0, 'Iinf': 0, 'IT': 0}
    for e in range(ell + 1):
        C = np.asarray(res['C_tau_by_ell'][e])
        Cn, Cp = C[:u.size][::-1], C[u.size:]
        Iinf = np.sum(wu * Cp)
        S = np.array([np.sum(wu * (Cp * np.exp(1j * o * u)
                                   + Cn * np.exp(-1j * o * u))) for o in om])
        fn = res['total_C_by_ell'][e]
        IT = np.array([np.sum(0.5 * T * wg * np.array(
            [complex(fn(0.0, t)) for t in 0.5 * T * (xg + 1)])) for T in Ts])
        scale = abs(sp['S0_by_ell'][e])
        assert abs(sp['integral_inf_by_ell'][e] - Iinf) <= 1e-10 * scale
        assert abs(sp['integral_inf_neg_by_ell'][e] - np.sum(wu * Cn)) \
            <= 1e-10 * scale
        assert np.max(np.abs(sp['S_by_ell'][e] - S)) <= 1e-10 * scale
        assert np.max(np.abs(sp['integral_by_ell'][e] - IT)) <= 1e-10 * scale
        # C0 is C_tau's τ = 0 value (Itô left limit)
        assert abs(sp['C0_by_ell'][e] - complex(fn(0.0, 0.0))) <= 1e-15
        # symmetric autocorrelation: S(0) = 2 ∫_0^∞ C
        assert abs(sp['S0_by_ell'][e] - 2 * sp['integral_inf_by_ell'][e]) \
            <= 1e-13 * scale
        tot['S'] += S
        tot['Iinf'] += Iinf
        tot['IT'] += IT
    assert np.max(np.abs(sp['S'] - tot['S'])) <= 1e-10 * abs(sp['S0'])
    assert abs(sp['integral_inf'] - tot['Iinf']) <= 1e-10 * abs(sp['S0'])
    # tau_c: total, and the truncated orders
    c0 = sum(sp['C0_by_ell'].values())
    assert abs(sp['tau_c'] - tot['Iinf'] / c0) <= 1e-10
    for e in range(ell + 1):
        num = sum(sp['integral_inf_by_ell'][j] for j in range(e + 1))
        den = sum(sp['C0_by_ell'][j] for j in range(e + 1))
        assert sp['tau_c_by_ell'][e] == num / den
    assert sp['tau_c_by_ell'][ell] == sp['tau_c']


def test_ou_quartic_one_loop_closed_form():
    """OU quartic at x* = 0, one loop: the tadpole shifts mu by 3 eps C(0) =
    3 eps D / mu, so S_1(ω) = dS_tree/dmu · 3 eps D / mu =
    −4 D mu / (ω² + mu²)² · 3 eps D / mu."""
    res = _cc(dd.load_model('ou_quartic')[0], [('dx', 1)] * 2, 1,
              {'omega': OM, 'T': TS})
    p = _params(res)
    mu, D, eps = p['mu'], p['D'], p['eps']
    dmu = 3 * eps * D / mu                       # 3 eps C_tree(0)
    S_tree = lambda m: 2 * D / (OM ** 2 + m ** 2)          # noqa: E731
    dS = -2 * D * 2 * mu / (OM ** 2 + mu ** 2) ** 2 * dmu
    assert _rel(res['spectra']['S_by_ell'][1], dS) <= 1e-13
    assert _rel(res['spectra']['S_by_ell'][0], S_tree(mu)) <= 1e-14


# ═════════════════════════════════════════════════════════════════════════════
# C. cross-spectra
# ═════════════════════════════════════════════════════════════════════════════

def test_two_field_cross_spectrum():
    m = _rotation()
    Ts = np.array([0.5, 2.0, 8.0])
    out = {}
    for ext in ([('x', 1), ('y', 1)], [('y', 1), ('x', 1)]):
        out[ext[0][0]] = _cc(m, ext, 0, {'omega': OM, 'T': Ts})
    xy, yx = out['x']['spectra'], out['y']['spectra']
    A = np.array([[-0.2, -1.0], [1.0, -0.2]])
    Q = np.diag([2.0, 0.0])
    ref = []
    for w in OM:
        H = np.linalg.inv(1j * w * np.eye(2) - A)
        ref.append((H @ Q @ H.conj().T)[0, 1])
    ref = np.array(ref)
    # S_xy = <x^ y^*> = [H Q H^H]_{xy}
    assert _rel(xy['S'], ref) <= 1e-13
    # conjugation: S_yx = conj(S_xy); S_xy(-ω) = conj(S_xy(ω))
    assert _rel(yx['S'], np.conj(xy['S'])) <= 1e-13
    sym = dd.spectra(out['x'], -OM)['S']
    assert _rel(sym, np.conj(xy['S'])) <= 1e-13
    # τ -> -τ swaps the windows of the two orderings
    assert _rel(xy['integral'], yx['integral_neg']) <= 1e-13
    assert _rel(xy['integral_inf'], yx['integral_inf_neg']) <= 1e-13
    # S(0) = ∫_0^∞ + ∫_{-∞}^0 (no δ), real for this cross pair
    assert abs(xy['S0'] - (xy['integral_inf'] + xy['integral_inf_neg'])) \
        <= 1e-14 * abs(xy['S0'])
    # the opposite sign convention of Cumulant.spectrum: S(ω) = spectrum(−ω)
    c = dd.cumulant(out['x'])
    assert _rel(c.spectrum(-OM), xy['S']) <= 1e-14
    # against the quadrature of the package's own C(τ)
    q = dd.spectra(out['x'], OM, Ts, method='quad')
    for key in ('S', 'integral', 'integral_neg'):
        assert np.max(np.abs(q[key] - xy[key])) <= 1e-10 * abs(xy['S0'])


def test_nonlinear_cross_spectrum_one_loop():
    om = np.array([-1.0, 0.0, 0.6, 2.0])
    Ts = np.array([0.5, 3.0])
    res = _cc(_coupled_quartic(), [('x', 1), ('y', 1)], 1,
              {'omega': om, 'T': Ts})
    sp = res['spectra']
    assert sp['method_by_ell'] == {0: 'exact', 1: 'exact'}
    assert np.max(np.abs(sp['S_by_ell'][1])) > 1e-4          # a real loop
    q = dd.spectra(res, om, Ts, method='quad')
    for key in ('S', 'integral', 'integral_neg', 'integral_inf',
                'integral_inf_neg', 'S0'):
        err = np.max(np.abs(np.asarray(q['abs_error'][key])))
        assert np.max(np.abs(np.asarray(q[key]) - np.asarray(sp[key]))) \
            <= max(1e-10 * abs(sp['S0']), 10 * err), key
    # reversed order: conjugate spectrum, mirrored windows
    rev = _cc(_coupled_quartic(), [('y', 1), ('x', 1)], 1,
              {'omega': om, 'T': Ts})['spectra']
    assert _rel(rev['S'], np.conj(sp['S'])) <= 1e-12
    assert _rel(rev['integral_neg'], sp['integral']) <= 1e-12


# ═════════════════════════════════════════════════════════════════════════════
# D. shot noise: the δ at τ = 0
# ═════════════════════════════════════════════════════════════════════════════

def test_shot_noise_delta_floor_linear_hawkes(monkeypatch):
    om = np.array([0.0, 0.3, 1.0, 4.0, 200.0])
    res = _cc(_hawkes(), [('n', 1)] * 2, 1, {'omega': om, 'T': TS})
    sp = res['spectra']
    p = _params(res)
    b = p['a'] * p['w']
    lam = p['a'] * p['Em'] / (1 - b)
    tau = p['tau']
    S = lam * (1 + om ** 2 * tau ** 2) / ((1 - b) ** 2 + om ** 2 * tau ** 2)
    assert _rel(sp['S'], S) <= 1e-13
    # the δ at τ = 0 is the mean rate; it is in S (the floor), not in the
    # windows; S(0) = ∫_0^∞ + ∫_{-∞}^0 + delta0
    assert abs(sp['delta0'] - lam) <= 1e-14 * lam
    assert abs(sp['S'][-1] - lam) <= 1e-3 * lam
    assert abs(sp['S0'] - (sp['integral_inf'] + sp['integral_inf_neg']
                           + sp['delta0'])) <= 1e-14 * abs(sp['S0'])
    assert abs(sp['S0'] - (2 * sp['integral_inf'] + sp['delta0'])) \
        <= 1e-13 * abs(sp['S0'])
    # the windows are of the smooth part: ∫_0^T -> 0 as T -> 0
    assert sp['integral'][0] == 0 and abs(sp['integral'][1]) < 1e-3
    # the smooth part's C(0) is C_tau's
    assert abs(sp['C0'] - res['C_tau'][0]) <= 1e-15
    # the spikes as Cumulant.deltas
    c = dd.cumulant(res)
    assert abs(sum(d.weight for d in c.deltas) - sp['delta0']) <= 1e-15
    # a spike that is not a point of the lag is refused, never dropped
    monkeypatch.setattr(type(c.deltas[0]), 'location',
                        property(lambda self: None))
    with pytest.raises(ME.MEUnavailable, match='not a point'):
        dd.spectra(res, om)


def test_shot_noise_cross_with_a_jump_at_zero():
    """C_{nv} jumps at τ = 0 (a spike kicks v): C0 is the Itô left limit, so
    C0 differs between the two orders (C_{vn}(0⁻) = C_{nv}(0⁺))."""
    m = _hawkes()
    nv = _cc(m, [('n', 1), ('v', 1)], 0, {'omega': OM, 'T': TS})['spectra']
    vn = _cc(m, [('v', 1), ('n', 1)], 0, {'omega': OM, 'T': TS})['spectra']
    assert nv['delta0'] == 0
    assert abs(nv['C0'] - vn['C0']) > 0.1 * abs(vn['C0'])
    assert _rel(vn['S'], np.conj(nv['S'])) <= 1e-13
    assert _rel(vn['integral'], nv['integral_neg']) <= 1e-13


def test_shot_noise_nonlinear_one_loop_exact_vs_quadrature():
    om = np.array([0.0, 0.4, 3.0])
    Ts = np.array([0.5, 3.0])
    res = _cc(_hawkes(quadratic=True), [('n', 1)] * 2, 1,
              {'omega': om, 'T': Ts})
    sp = res['spectra']
    assert sp['method_by_ell'] == {0: 'exact', 1: 'exact'}
    assert abs(sp['delta0_by_ell'][1]) > 1e-4         # the loop shifts the rate
    q = dd.spectra(res, om, Ts, method='quad')
    assert q['method_by_ell'] == {0: 'quad', 1: 'quad'}
    for key in ('S', 'integral', 'integral_inf', 'S0'):
        assert np.max(np.abs(np.asarray(q[key]) - np.asarray(sp[key]))) \
            <= 1e-10 * abs(sp['S0']), key
        assert np.all(np.asarray(q['abs_error'][key]) < 1e-9)
    for e in (0, 1):
        assert abs(sp['S0_by_ell'][e] - 2 * sp['integral_inf_by_ell'][e]
                   - sp['delta0_by_ell'][e]) <= 1e-13 * abs(sp['S0'])


# ═════════════════════════════════════════════════════════════════════════════
# E. adversarial: near-degenerate and repeated poles
# ═════════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize('gap', [1e-9, 0.0])
def test_near_degenerate_poles_against_50_digit_reference(gap):
    """Colour noise with the filter rate b = mu (1 + gap): the tree C(τ) =
    D/(b² − mu²) (e^{−mu τ}/mu − e^{−b τ}/b) cancels to 9 digits (gap
    1e-9) or is the double pole D e^{−mu τ}(1 + mu τ)/(2 mu³) (gap 0); its
    mode coefficients are ~1/gap.  The triples are exact."""
    import mpmath as mp
    om = np.array([0.0, 0.5, 3.0])
    Ts = np.array([1e-3, 0.5, 3.0])
    res = _cc(_colour(gap), [('x', 1)] * 2, 1, {'omega': om, 'T': Ts})
    sp = res['spectra']
    with mp.workdps(50):
        M = mp.mpf(1)
        B = mp.mpf(1.0 + gap)                  # the double the run used
        D = mp.mpf(1)

        def C(t):
            if B == M:
                return D * mp.e ** (-M * t) * (1 + M * t) / (2 * M ** 3)
            return D / (B ** 2 - M ** 2) * (mp.e ** (-M * t) / M
                                            - mp.e ** (-B * t) / B)
        S_ref = [float(2 * D / ((mp.mpf(w) ** 2 + M ** 2)
                                * (mp.mpf(w) ** 2 + B ** 2))) for w in om]
        I_ref = [float(mp.quad(C, [0, mp.mpf(T)])) for T in Ts]
        Iinf_ref = float(D / (M ** 2 * B ** 2))
        C0_ref = float(C(0))
    assert _rel(sp['S_by_ell'][0], S_ref) <= 1e-13
    assert _rel(sp['integral_by_ell'][0], I_ref) <= 1e-13
    assert abs(sp['integral_inf_by_ell'][0] - Iinf_ref) <= 1e-14 * Iinf_ref
    assert abs(sp['C0_by_ell'][0] - C0_ref) <= 1e-14 * C0_ref
    # the loop: exact against quadrature of the package's own C(τ)
    q = dd.spectra(res, om, Ts, method='quad')
    for key in ('S', 'integral', 'integral_inf'):
        assert np.max(np.abs(np.asarray(q[key]) - np.asarray(sp[key]))) \
            <= 1e-10 * abs(sp['S0']), key
    if gap:
        # the mode form is ill-conditioned here (warned); the outputs above
        # never go through it
        with pytest.warns(ME.MEConditioningWarning):
            dd.cumulant(res).me.modes(0)


# ═════════════════════════════════════════════════════════════════════════════
# F. fallbacks and refusals
# ═════════════════════════════════════════════════════════════════════════════

def test_pole_engine_falls_back_on_quadrature_with_error_estimates(
        monkeypatch):
    monkeypatch.setenv(SS.ENGINE_ENV, 'poles')
    om = np.array([0.0, 0.7])
    Ts = np.array([0.5, 2.0])
    with pytest.warns(ME.MEFallbackWarning):
        res = _cc(dd.load_model('ou_quartic')[0], [('dx', 1)] * 2, 1,
                  {'omega': om, 'T': Ts})
    sp = res['spectra']
    assert sp['method_by_ell'] == {0: 'quad', 1: 'quad'}
    assert sp['notes'] and 'poles' in sp['notes'][0]
    for e in (0, 1):
        er = sp['error_by_ell'][e]
        assert set(er) >= {'S', 'integral', 'integral_neg', 'integral_inf',
                           'integral_inf_neg', 'S0'}
        assert np.all(np.asarray(er['S']) < 1e-9)
    monkeypatch.delenv(SS.ENGINE_ENV)
    ex = _cc(dd.load_model('ou_quartic')[0], [('dx', 1)] * 2, 1,
             {'omega': om, 'T': Ts})['spectra']
    for key in ('S', 'integral', 'integral_inf', 'S0', 'C0'):
        assert np.max(np.abs(np.asarray(sp[key]) - np.asarray(ex[key]))) \
            <= 1e-10, key
    # 'exact' refuses: after the run, the result is kept with the reason
    monkeypatch.setenv(SS.ENGINE_ENV, 'poles')
    with pytest.warns(SP.SpectraWarning):
        r2 = _cc(dd.load_model('ou_quartic')[0], [('dx', 1)] * 2, 0,
                 {'omega': om, 'method': 'exact'})
    assert r2['spectra']['status'] == 'refused'
    assert 'triples' in r2['spectra']['reason']
    assert r2['C_tau'] is not None
    with pytest.raises(ME.MEUnavailable):
        dd.spectra(r2, om, method='exact')


@pytest.mark.parametrize('kw, match', [
    (dict(k=3), 'k = 2'),
    (dict(t_min=-5.0, times=np.array([0.0])), 't_min'),
    (dict(discard=True), 'discard'),
])
def test_refused_before_the_run(kw, match):
    m = dd.load_model('ou_quartic')[0]
    k = kw.pop('k', 2)
    with pytest.raises(ValueError, match=match):
        compute_cumulants(m, k=k, max_ell=0, external_fields=[('dx', 1)] * k,
                          parallel=False, verbose=False, use_cache=False,
                          spectra=True,
                          **({} if 't_min' in kw else
                             {'tau_grid': np.array([0.0])}), **kw)


def test_refused_spatial_and_bad_options():
    m = dd.load_model('edwards_wilkinson_1d')[0]
    with pytest.raises(ValueError, match='temporal'):
        compute_cumulants(m, k=2, max_ell=0,
                          external_fields=[(dd.field_names(m)[0], 1)] * 2,
                          chi_grid=np.array([0.0]), parallel=False,
                          verbose=False, spectra=True)
    for bad, match in (({'omg': [1.0]}, 'unknown key'),
                       ({'method': 'fast'}, 'method'),
                       ({'T': -1.0}, '>= 0'),
                       ({'omega': [0.0, np.nan]}, 'finite'),
                       ({'omega': [-5, 5, 51]}, 'TUPLE'),
                       ('yes', 'dict')):
        with pytest.raises(ValueError, match=match):
            SP.resolve_spectra_options(bad)
    with pytest.raises(ValueError, match='output'):
        dd.Config(k=2, spectra=True, output='moment')
    with pytest.raises(ValueError, match='unknown key'):
        dd.Config(k=2, spectra={'omega_grid': [1.0]})


def test_large_block_route_equals_dense(monkeypatch):
    """Blocks beyond the dense forms: ∫_0^T by the segment's own action and
    resolvent (``c (−G)^{-1}(x0 − e^{GT} x0)``), the resolvents matrix free."""
    om = np.array([0.0, 0.7])
    Ts = np.array([0.01, 1.0, 4.0])
    res = _cc(dd.load_model('ou_quartic')[0], [('dx', 1)] * 2, 2)
    a = dd.spectra(res, om, Ts)
    monkeypatch.setattr(ME, 'MAX_BLOCK', 0)
    b = dd.spectra(res, om, Ts)
    for key in ('S', 'integral', 'integral_inf', 'C0'):
        assert np.max(np.abs(np.asarray(a[key]) - np.asarray(b[key]))) \
            <= 1e-13, key


def test_delay_model_exact_tree_and_pade_loops():
    """OU with a hard delay: the exact delayed tree's S(ω) = 2D / |iω + a −
    c e^{−iωd}|² (exact), its windows by quadrature (with errors), the Pade
    loops exact for the approximant (noted)."""
    a, c, d, D = 1.0, -2.0, 0.5, 0.1
    m = (SDE('spectra delay OU').parameter('a', default=a)
         .parameter('c', default=c).parameter('d', default=d, domain='positive')
         .parameter('eps', default=0.02).parameter('D', default=D).field('x')
         .drift('x', '-a*x + c*x(t - d) - eps*x^3')
         .noise('xi', Gaussian(var='2*D'), couples={'x': 1})).build()
    om = np.array([-2.0, 0.0, 0.5, 3.0])
    Ts = np.array([0.5, 2.0])
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        res = _cc(m, [('x', 1)] * 2, 1, {'omega': om, 'T': Ts},
                  taylor_order=4)
    sp = res['spectra']
    S = 2 * D / np.abs(1j * om + a - c * np.exp(-1j * om * d)) ** 2
    assert _rel(sp['S_by_ell'][0], S) <= 1e-13
    assert sp['method_by_ell'] == {0: 'exact+quad', 1: 'exact'}
    assert np.all(sp['error_by_ell'][0]['integral'] < 1e-10)
    assert any('Padé' in n for n in sp['notes'])
    assert abs(sp['S0_by_ell'][0] - 2 * sp['integral_inf_by_ell'][0]) \
        <= 1e-13 * abs(sp['S0'])
    # the windows of the tree tend to ∫_0^∞
    big = dd.spectra(res, None, np.array([30.0]))
    assert abs(big['integral_by_ell'][0][0] - sp['integral_inf_by_ell'][0]) \
        <= 1e-10 * abs(sp['S0'])
    with pytest.raises(ME.MEUnavailable, match='exact delayed'):
        dd.spectra(res, om, method='quad')


# ═════════════════════════════════════════════════════════════════════════════
# G. saving and the YAML config
# ═════════════════════════════════════════════════════════════════════════════

def test_save_npz_and_yaml_config(tmp_path):
    from api.save import save_npz, load_npz
    from api.config_run import run_config, load_config
    res = _cc(dd.load_model('ou_quartic')[0], [('dx', 1)] * 2, 1,
              {'omega': OM, 'T': TS})
    p = save_npz(res, str(tmp_path / 'r.npz'))
    z = load_npz(p)
    sp = res['spectra']
    assert np.array_equal(z['spectra_omega'], OM)
    assert np.array_equal(z['spectra_S'], sp['S'])
    assert np.array_equal(z['spectra_S_by_ell'][1], sp['S_by_ell'][1])
    assert np.array_equal(z['spectra_integral'], sp['integral'])
    assert z['spectra_tau_c'][0] == sp['tau_c']
    assert list(z['spectra_ells']) == [0, 1]
    cfg = tmp_path / 'c.yaml'
    cfg.write_text(
        'model: ou_quartic\n'
        'question:\n'
        '  k: 2\n'
        '  max_ell: 1\n'
        '  tau_grid: {values: [0.0]}\n'
        '  spectra: {omega: {start: -3, stop: 3, num: 7}, T: [1, 4],'
        ' method: exact}\n'
        'policy: {use_cache: false}\n'
        'outputs: {npz: out/c.npz}\n')
    result, resolved = run_config(str(cfg))
    s2 = result['spectra']
    assert np.array_equal(s2['omega'], np.linspace(-3, 3, 7))
    assert np.array_equal(s2['T'], np.array([1.0, 4.0]))
    ref = dd.spectra(res, np.linspace(-3, 3, 7), np.array([1.0, 4.0]))
    assert np.max(np.abs(s2['S'] - ref['S'])) <= 1e-14
    z2 = load_npz(str(tmp_path / 'out' / 'c.npz'))
    assert 'spectra_S' in z2 and z2['spectra_status'][0] == 'ok'
    again = load_config(str(tmp_path / 'out' / 'config_resolved.yaml'))[0]
    assert again == resolved
    bad = tmp_path / 'b.yaml'
    bad.write_text('model: ou_quartic\nquestion:\n  spectra: {omega: [1.0],'
                   ' window: 3}\n')
    from api.config_run import ConfigError
    with pytest.raises(ConfigError, match='line 3'):
        load_config(str(bad))


def test_non_decaying_blocks():
    """Constant (u-independent) blocks: cancelling ones are dropped from the
    integrals to infinity, others make S and ∫_0^∞ not exist (refused by
    name); a block with a non-negative rate is refused too."""
    from types import SimpleNamespace as NS

    def blk(w, const=True, rate=None):
        T = None if rate is None else np.array([[rate]], dtype=complex)
        return NS(weight=w, constant=const, n=1,
                  tri=lambda: (T, np.ones(1), np.ones(1)))
    side = NS(ell=1, side='+', blocks=[blk(1.0), blk(-1.0)])
    assert SP._side_split(side) == ([], 0j)
    side.blocks = [blk(1.0), blk(-0.5)]
    with pytest.raises(ME.MEUnavailable, match='constant'):
        SP._side_split(side)
    side.blocks = [blk(1.0, False, 0.0)]
    with pytest.raises(ME.MEUnavailable, match='does not'):
        SP._side_split(side)


def test_dd_exports():
    assert dd.spectra is SP.spectra
    assert dd.SpectraWarning is SP.SpectraWarning
    assert 'spectra' in [f for f in dd.Config.__dataclass_fields__]
