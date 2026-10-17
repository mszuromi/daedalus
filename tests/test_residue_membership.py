"""
Residues of high-degree rational propagators (``api._propagator``).

The exact tier decided "pole p_k is not a pole of entry (i, j)" by an
ABSOLUTE test ``|Q_ij(p_k)| > 1e-9`` on the monic denominator; at high degree
its rounding alone exceeds that, every pole was skipped and C = 0 without a
word.  Scalar OU with an Erlang-n kernel, ``dx = (-a x + c (g*x)) dt +
sqrt(2D) dW`` (a = 1, c = -2, D = 0.1, kernel time scale d = 1), tree C(0)
before the fix:

    n    pipeline        exact
    1    0.0666665667    0.0666666667
    2    0.0809522810    0.0809523810
    4    0.1024145982    0.1024146982
    8    0.0000000000    0.1354441967

(the 1e-7 offsets are the tau = 0 Ito left limit, evaluated at -1e-6).  Pole
membership is now exact (the entry denominators split Q into exact coprime
factors), the double-precision residues carry an accuracy certificate (else
the exact 50-digit route), and nothing is decided by an absolute threshold.
The numerical tier's siblings (a leading-coefficient trim that drops genuine
coefficients, an absolute |det| acceptance) are fixed the same way.

Kernels: Erlang-n, ``(1 + s d/n)^-n``, and the all-pass ``B_n(-s d) /
B_n(s d)`` (``bessel_coeffs``), both of denominator degree n.

References: the exact linear state-space embedding of the rational kernel
(the controllable canonical form),
stationary covariance by an eigendecomposition Lyapunov solve and the matrix
exponential at 80 digits (mpmath) -- independent of the pole/residue code.
"""
from __future__ import annotations

import os
import sys
import warnings
from math import factorial

import mpmath as mp
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from sage.all import SR, var, matrix, I                             # noqa: E402

from api import compute_cumulants                                   # noqa: E402
from api import _propagator as PR                                   # noqa: E402
from api.noise import Gaussian                                      # noqa: E402
from api.sde import SDE                                             # noqa: E402

A, C, D, TS = 1.0, -2.0, 0.1, 1.0
TAU = np.array([0.0, 0.5, 1.0, 2.0, 4.0])
ITO = 0.0           # tau = 0: the exact Ito left limit (was evaluated at -1e-6 before Oct 2026)
RTOL = 1e-11


# ── kernels (frequency images, s = I omega) and their exact realizations ──

def erlang_image(n):
    return f'(1 + I*omega*d/{n})^(-{n})'


def bessel_coeffs(n):
    """The normalised reverse Bessel polynomial B_n(z) = sum_k b_k z^k: the
    all-pass kernel B_n(-s d) / B_n(s d) has equal degrees (a delta part)."""
    return [factorial(2 * n - k) * factorial(n) / (factorial(2 * n) * factorial(k) * factorial(n - k))
            for k in range(n + 1)]


def allpass_image(n):
    c = bessel_coeffs(n)
    z = '(I*omega*d)'
    num = ' + '.join(f'({ck!r})*(-1)^{k}*{z}^{k}' for k, ck in enumerate(c))
    den = ' + '.join(f'({ck!r})*{z}^{k}' for k, ck in enumerate(c))
    return f'({num})/({den})'


def _erlang_NM(n, d):
    r = mp.mpf(n) / mp.mpf(d)
    return [r ** n], [mp.mpf(factorial(n) // (factorial(k) * factorial(n - k))) * r ** (n - k)
                     for k in range(n + 1)]


def _allpass_NM(n, d):
    co = [mp.mpf(factorial(2 * n - k) * factorial(n)) / (factorial(2 * n) * factorial(k) * factorial(n - k))
          for k in range(n + 1)]
    return ([co[k] * (-mp.mpf(d)) ** k for k in range(n + 1)],
            [co[k] * mp.mpf(d) ** k for k in range(n + 1)])


def _ou_oracle(taus, N, M, a=A, c=C, Dn=D, ou_rate=None, ou_D=None, raw=False):
    """C_xx(tau) of x' = -a x + c (g*x) [+ z] + sqrt(2 Dn) xi, g = N/M (ascending
    coefficients in s), optionally driven by an OU z' = -ou_rate z + sqrt(2 ou_D) xi'."""
    with mp.workdps(80):
        n = len(M) - 1
        lc = M[-1]
        Mn = [m / lc for m in M]
        Nn = [x / lc for x in N] + [mp.mpf(0)] * (n + 1 - len(N))
        Dg = Nn[n]
        Nt = [Nn[k] - Dg * Mn[k] for k in range(n)]
        extra = 0 if ou_rate is None else 1
        m = n + 1 + extra
        Am = mp.zeros(m, m)
        Am[0, 0] = -mp.mpf(a) + mp.mpf(c) * Dg
        for k in range(n):
            Am[0, 1 + k] = mp.mpf(c) * Nt[k]
            if k < n - 1:
                Am[1 + k, 2 + k] = 1
            Am[n, 1 + k] = -Mn[k]
        Am[n, 0] += 1
        Qn = mp.zeros(m, m)
        Qn[0, 0] = 2 * mp.mpf(Dn)
        if extra:
            Am[0, m - 1] = 1
            Am[m - 1, m - 1] = -mp.mpf(ou_rate)
            Qn[m - 1, m - 1] = 2 * mp.mpf(ou_D)
        E, V = mp.eig(Am)
        Vi = mp.inverse(V)
        Qh = Vi * Qn * Vi.T
        X = mp.matrix(m, m)
        for i in range(m):
            for j in range(m):
                X[i, j] = -Qh[i, j] / (E[i] + E[j])
        W = X * V.T
        out = [mp.re(sum(V[0, k] * mp.exp(E[k] * abs(mp.mpf(t))) * W[k, 0] for k in range(m)))
               for t in taus]
        return out if raw else np.array([float(v) for v in out])


def _model(image, name, ou=False):
    b = (SDE(name).parameter('a', default=A).parameter('c', default=C)
         .parameter('d', default=TS, domain='positive').parameter('D', default=D).field('x'))
    if ou:
        b = (b.parameter('b', default=0.7).parameter('Dz', default=0.05).field('z')
             .define_kernel('g', freq_image=image)
             .drift('x', '-a*x + c*Conv(g, x) + z').drift('z', '-b*z')
             .noise('xi', Gaussian(var='2*D'), couples={'x': 1})
             .noise('zeta', Gaussian(var='2*Dz'), couples={'z': 1}))
    else:
        b = (b.define_kernel('g', freq_image=image).drift('x', '-a*x + c*Conv(g, x)')
             .noise('xi', Gaussian(var='2*D'), couples={'x': 1}))
    return b.build()


def _run(model, tau=TAU):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        r = compute_cumulants(model, k=2, max_ell=0, external_fields=[('x', 1), ('x', 1)], parallel=False,
                              verbose=False, use_cache=False, tau_grid=np.asarray(tau, dtype=float))
    r['_warnings'] = caught
    return r


def _tree(r):
    return np.array([complex(v).real for v in r['C_tau']])


def _ref_taus(tau=TAU):
    return [ITO if t == 0 else float(t) for t in tau]


def _route(r):
    return (r['propagator'].get('pole_analysis') or {}).get('residue_route')


# ── acceptance: the table above and its extension ──────────────────────

@pytest.mark.parametrize('n', [1, 2, 4, 8, 16, 32])
def test_erlang_ou_tree_matches_linear_chain(n):
    r = _run(_model(erlang_image(n), f'erlang{n}'))
    ref = _ou_oracle(_ref_taus(), *_erlang_NM(n, TS))
    got = _tree(r)
    assert np.all(np.abs(got - ref) <= RTOL * np.abs(ref)), (n, got, ref)


@pytest.mark.parametrize('n', [4, 8, 12, 16, 24])
def test_allpass_ou_tree_matches_state_space(n):
    r = _run(_model(allpass_image(n), f'allpass{n}'))
    ref = _ou_oracle(_ref_taus(), *_allpass_NM(n, TS))
    got = _tree(r)
    assert np.all(np.abs(got - ref) <= RTOL * np.abs(ref)), (n, got, ref)


def test_distinct_entry_denominators_high_degree():
    """x driven by an OU z: the entries of G have different denominators
    (G_zz carries only z's pole), so membership is labelled per entry."""
    r = _run(_model(erlang_image(8), 'erlang8_ou', ou=True))
    pa = r['propagator']['pole_analysis']
    assert pa['membership'] == 'labelled'
    ref = _ou_oracle(_ref_taus(), *_erlang_NM(8, TS), ou_rate=0.7, ou_D=0.05)
    got = _tree(r)
    assert np.all(np.abs(got - ref) <= RTOL * np.abs(ref)), (got, ref)


def test_residue_route_and_finding():
    """Low degree: certified double-precision residues; high degree: the
    certificate fails and the exact 50-digit route runs, with a G3 finding."""
    r4 = _run(_model(erlang_image(4), 'erlang4'), tau=[0.5])
    assert _route(r4) == 'double'
    assert not r4['analysis'].has('G3_RESIDUES_HIGH_PRECISION')
    r16 = _run(_model(erlang_image(16), 'erlang16'), tau=[0.5])
    assert _route(r16) == 'exact'
    assert r16['analysis'].has('G3_RESIDUES_HIGH_PRECISION')
    assert r16['analysis'].traits.residue_route == 'exact'


def test_one_loop_erlang8_matches_hartree():
    """-eps x^3 at x* = 0: the exact O(eps) term is the Hartree shift
    3 eps C0(0) dC0/da (the tadpole), here from the 80-digit oracle."""
    eps, d = 0.02, 0.5
    image = erlang_image(8)
    m = (SDE('erlang8_loop').parameter('a', default=A).parameter('c', default=C)
         .parameter('d', default=d, domain='positive').parameter('D', default=D).parameter('eps', default=eps)
         .field('x').define_kernel('g', freq_image=image).drift('x', '-a*x + c*Conv(g, x) - eps*x^3')
         .noise('xi', Gaussian(var='2*D'), couples={'x': 1}).build())
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        r = compute_cumulants(m, k=2, max_ell=1, external_fields=[('x', 1), ('x', 1)], parallel=False,
                              verbose=False, use_cache=False, tau_grid=TAU)
    taus = _ref_taus()
    NM = _erlang_NM(8, d)
    with mp.workdps(80):
        c00 = _ou_oracle([0], *NM, raw=True)[0]
        h = mp.mpf(10) ** -30
        up = _ou_oracle(taus, *NM, a=mp.mpf(A) + h, raw=True)
        dn = _ou_oracle(taus, *NM, a=mp.mpf(A) - h, raw=True)
        ref = np.array([float(3 * eps * c00 * (u - v) / (2 * h)) for u, v in zip(up, dn)])
    got = np.array([complex(v).real for v in r['C_tau_by_ell'][1]])
    assert np.max(np.abs(got - ref)) <= RTOL * np.max(np.abs(ref)), (got, ref)


# ── never silent ───────────────────────────────────────────────────────

def _diag_K(rate2):
    """K(omega) of two decoupled OU fields with rates 1 and ``rate2``."""
    om = var('omega')
    return matrix(SR, [[I * om + 1, 0], [0, I * om + SR(rate2)]]), om


def test_membership_labelling_clear_and_undecided():
    from sage.all import CyclotomicField, PolynomialRing, CDF
    CF = CyclotomicField(4, 'I_')
    P = PolynomialRing(CF, 'w')
    w = P.gen()
    i_ = CF.gen()
    b1, b2 = w - i_, w - 2 * i_
    Q = b1 * b2
    dens = {(0, 0): b1, (1, 1): b2}
    to_cdf = lambda p: PolynomialRing(CDF, 'wc')([complex(c) for c in p.list()])
    member, how = PR._entry_pole_membership(Q, dens, [1j, 2j], to_cdf)
    assert how == 'labelled' and member == {(0, 0): frozenset({0}), (1, 1): frozenset({1})}
    # roots equidistant from both factors: no clear attribution -> undecided
    member, how = PR._entry_pole_membership(Q, dens, [1.5j, 1.5j + 1e-9], to_cdf)
    assert member is None and how == 'undecided'
    # one entry with Q itself: no numerical decision at all
    member, how = PR._entry_pole_membership(Q, {(0, 0): Q}, [1j, 2j], to_cdf)
    assert how == 'shared' and member == {(0, 0): None}


def test_unclassifiable_poles_take_the_exact_route():
    """Roots 1e-12 apart in different entries: double precision cannot tell
    which root is whose; the exact route can, and says so."""
    K, om = _diag_K('1 + 1/1000000000000')
    an = {}
    pv, Cm, _ = PR._compute_residues_via_polynomial_fracfield(K, om, {}, 2, analysis_out=an)
    assert an['residue_route'] == 'exact' and 'attributed' in an['residue_route_reason']
    assert len(pv) == 2
    # each residue sits in its own entry, value i * (1/(-i)) = -1 ... times i: C = 1
    for C_k in Cm:
        nz = [(i, j) for i in range(2) for j in range(2) if abs(complex(C_k[i, j])) > 0]
        assert len(nz) == 1 and abs(complex(C_k[nz[0]]) - 1) < 1e-14


def test_exact_route_failure_warns_and_is_recorded(monkeypatch):
    K, om = _diag_K('1 + 1/1000000000000')

    def boom(*a, **k):
        raise ArithmeticError('forced failure')
    monkeypatch.setattr(PR, '_exact_route', boom)
    monkeypatch.setattr(PR, '_cluster_path', lambda *a, **k: None)
    an = {}
    with pytest.warns(PR.PoleResidueWarning, match='forced failure'):
        res = PR._compute_residues_via_polynomial_fracfield(K, om, {}, 2, analysis_out=an)
    assert res == (None, None, None)
    assert an['residue_route'] == 'failed'


def test_failed_route_reaches_the_analyzer(monkeypatch):
    """End to end: the exact tier cannot compute the residues -> warning,
    G3_RESIDUES_NUMERICAL_TIER, and the numerical tier's numbers."""
    monkeypatch.setattr(PR, '_entry_pole_membership', lambda *a, **k: (None, 'undecided'))

    def boom(*a, **k):
        raise ArithmeticError('forced failure')
    monkeypatch.setattr(PR, '_exact_route', boom)
    monkeypatch.setattr(PR, '_cluster_path', lambda *a, **k: None)
    r = _run(_model(erlang_image(2), 'erlang2_ou_fail', ou=True), tau=[0.5, 1.0])
    assert any(issubclass(w.category, PR.PoleResidueWarning) for w in r['_warnings'])
    assert r['analysis'].has('G3_RESIDUES_NUMERICAL_TIER')
    ref = _ou_oracle([0.5, 1.0], *_erlang_NM(2, TS), ou_rate=0.7, ou_D=0.05)
    assert np.allclose(_tree(r), ref, rtol=1e-8, atol=0)


# ── the numerical tier's siblings ──────────────────────────────────────

@pytest.mark.parametrize('path', ['legacy', 'rich'])
@pytest.mark.parametrize('kind,n,rtol', [('erlang', 16, 1e-5), ('allpass', 16, 1e-9)])
def test_numerical_tier_keeps_every_pole(monkeypatch, kind, n, rtol, path):
    """Forced numerical tier (both of its paths: with the symbolic inverse,
    'legacy', and without, 'rich'): the leading-coefficient trim of
    num(det K) dropped genuine coefficients (Erlang-16: 4 of 17 poles kept,
    C 25 % off; all-pass-16: 7 of 17)."""
    monkeypatch.setattr(PR, '_compute_residues_via_polynomial_fracfield', lambda *a, **k: (None, None, None))
    if path == 'rich':
        import api.compute as CO
        orig = PR.compute_poles_and_residues

        def no_inverse(prop, *a, **k):
            prop['adj_ft'] = None
            return orig(prop, *a, **k)
        monkeypatch.setattr(CO, 'compute_poles_and_residues', no_inverse)
    image, NM = (erlang_image, _erlang_NM) if kind == 'erlang' else (allpass_image, _allpass_NM)
    r = _run(_model(image(n), f'{kind}{n}_tier2'), tau=[0.5, 1.0, 4.0])
    assert len(r['propagator']['pole_vals']) == n + 1
    ref = _ou_oracle([0.5, 1.0, 4.0], *NM(n, TS))
    assert np.allclose(_tree(r), ref, rtol=rtol, atol=0)


@pytest.mark.parametrize('scale', [1e-30, 1.0, 1e30])
def test_singular_at_is_scale_free(scale):
    roots = [0.3 + 1j, -2 + 0.5j]
    det = lambda w: scale * (w - roots[0]) * (w - roots[1])
    assert all(PR._singular_at(det, r) for r in roots)
    assert not PR._singular_at(det, 1 + 1j)
    assert not PR._singular_at(det, roots[0] + 1e-2)
