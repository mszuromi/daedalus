"""
tests/test_realization.py
=========================
State-space realization of the propagator, ``api/_realization.py``:
``K(omega)^{-1} = C (sI - A)^{-1} B + D`` (s = i omega) built without an
eigenvalue or a numerical root.  Inline public models only:

* first-order two-field system (class A), incl. an exact double pole;
* descriptor with a delta part (class B, index 1): ``D`` exact;
* Erlang-8 kernel (class C, chain from the given rate) and a balanced
  Pade-8 kernel (class C, companion block), both end to end from the SDE
  front end through ``build_propagator``;
* refusals by name: index 2, non-rational kernel, unbound symbol, spatial;
* the triangular ``expm`` used by ``impulse`` at a near-double eigenvalue.
"""
from math import factorial

import numpy as np
import pytest

pytest.importorskip('sage.all')
import mpmath as mp                                                   # noqa: E402
from sage.all import SR, I, matrix, var, exp                          # noqa: E402

from api import _realization as RZ                                    # noqa: E402

W = var('omega')


def _mp_impulse(R, t, dps=40):
    """``C e^{At} B`` from the EXACT (unrounded) realization, mpmath."""
    with mp.workdps(dps):
        def conv(M):
            return mp.matrix([[mp.mpf(int(M[i, j].numerator())) / int(M[i, j].denominator())
                               for j in range(M.ncols())] for i in range(M.nrows())])
        ex = R.exact
        G = conv(ex['C']) * mp.expm(conv(ex['A']) * t) * conv(ex['B'])
        return np.array([[float(G[i, j]) for j in range(G.cols)] for i in range(G.rows)])


# ── class A: first order ──────────────────────────────────────────────────

@pytest.mark.parametrize('gap', [0.5, 1e-8, 0.0])
def test_first_order_two_field_exact_through_double_pole(gap):
    # x' = -x + eta, eta' = -(1 + gap) eta: G_{x, eta}(t) = (e^{-t} - e^{-(1+gap)t})/gap
    lam = SR(1 + gap)
    K = matrix(SR, [[I * W + 1, -1], [0, I * W + lam]])
    R = RZ.realize_K(K, W, {}, phys_names=('x', 'eta'), resp_names=('xt', 'etat'))
    assert R.cls == 'A' and not R.descriptor and R.n_states == 2
    assert R.entry_states.tolist() == [[1, 2], [0, 1]]
    assert R.hurwitz is True and R.lyapunov_margin > 0
    assert np.all(np.asarray(R.D) == 0)
    v = RZ.verify_against_K(R, K, W, {}, n_points=20, seed=1)
    assert v['max_err'] <= 1e-13 and v['max_err_exact'] <= 1e-40
    lamf = float(1 + gap)
    for t in (0.1, 1.0, 5.0, 20.0):
        with mp.workdps(40):
            lm = mp.mpf(lamf)
            ref = float(t * mp.exp(-t) if gap == 0 else (mp.exp(-t) - mp.exp(-lm * t)) / (lm - 1))
        assert abs(R.impulse(t)[0, 1] - ref) <= 1e-15 * 0.37 + 1e-14 * abs(ref)
        assert abs(R.impulse_entry(0, 1, t) - ref) <= 1e-15 * 0.37 + 1e-14 * abs(ref)
    assert np.all(R.impulse(-0.5) == 0)
    # stationary covariance (noise on eta, variance 2): exact Lyapunov value
    Q = np.diag([0.0, 2.0])
    S = R.stationary_covariance(Q)
    ref = 1.0 / (lamf * (1.0 + lamf))            # Var x of the OU-driven OU
    assert abs(S[0, 0] - ref) <= 1e-14 * ref


# ── class B: index-1 descriptor with a delta part ─────────────────────────

def test_descriptor_delta_part_exact_D():
    # n = a v (algebraic, delta part), tau v' = -v + w n + y_v
    a, w_, tau = var('a w tau')
    K = matrix(SR, [[1, -a], [-w_, I * W * tau + 1]])
    P = {a: 0.5, w_: 0.25, tau: 2.0}
    R = RZ.realize_K(K, W, P)
    assert R.cls == 'B' and R.descriptor and R.n_states == 1
    D_delta = matrix(SR, [[1, 0], [0, 0]])            # lim_{omega -> inf} K^{-1}
    cd = RZ.compare_D(R, D_delta, P)
    assert cd['exact_equal'] is True and cd['max_rel'] == 0.0
    assert R.exact['D'] == matrix(R.exact['field'], [[1, 0], [0, 0]])
    v = RZ.verify_against_K(R, K, W, P, n_points=20, seed=2)
    assert v['max_err'] <= 1e-13 and v['max_err_entries'] <= 1e-13
    # G_vv(t) = e^{-(1 - a w) t / tau} / tau
    for t in (0.0, 0.3, 3.0):
        assert abs(R.impulse(t)[1, 1] - np.exp(-(1 - 0.125) * t / 2.0) / 2.0) <= 1e-15


# ── class C: rational kernels, end to end from the front end ──────────────

def _pade_str(n, d='d'):
    c = [factorial(2 * n - k) * factorial(n) / (factorial(2 * n) * factorial(k) * factorial(n - k))
         for k in range(n + 1)]
    z = f'(I*omega*{d})'
    num = ' + '.join(f'({ck!r})*(-1)^{k}*{z}^{k}' for k, ck in enumerate(c))
    den = ' + '.join(f'({ck!r})*{z}^{k}' for k, ck in enumerate(c))
    return f'({num})/({den})'


def _kernel_prop(freq_image, name):
    """x' = -a x + c (g * x) - eps x^3 + noise; propagator data + parameters at x* = 0."""
    from api.sde import SDE
    from api.noise import Gaussian
    from api._propagator import build_propagator
    from engine.core.field_theory import FieldTheory
    m = (SDE(name).parameter('a', default=1.0).parameter('c', default=-2.0)
         .parameter('d', default=1.0, domain='positive').parameter('eps', default=0.1)
         .parameter('D', default=0.1).field('x').define_kernel('g', freq_image=freq_image)
         .drift('x', '-a*x + c*Conv(g, x) - eps*x^3')
         .noise('xi', Gaussian(var='2*D'), couples={'x': 1}).build())
    ft = FieldTheory(m, taylor_order=2)
    ft.expand()
    prop = build_propagator(ft, m, use_cache=False, verbose=False)
    vals = {'a': 1.0, 'c': -2.0, 'd': 1.0, 'eps': 0.1, 'D': 0.1}
    syms = set()
    for e in prop['K_ft'].list():
        syms.update(SR(e).variables())
    params = {}
    for s in syms:
        if str(s) == 'omega':
            continue
        params[s] = vals.get(str(s), 0.0)            # the saddle x* = 0
    return prop, params


def test_erlang8_kernel_chain_from_given_rate():
    n = 8
    prop, params = _kernel_prop(f'(1 + I*omega*d/{n})^(-{n})', 'erl8_realization_test')
    R = RZ.realize(prop, params)
    assert R.cls == 'C' and R.n_states == n + 1
    assert any(f'{n} first-order block(s) from 1 given rate(s) (multiplicities [{n}])' in p
               for p in R.provenance)
    v = RZ.verify_against_K(R, prop['K_ft'], prop['omega'], params, n_points=20, seed=3)
    assert v['max_err'] <= 1e-13 and v['max_err_exact'] <= 1e-40
    # exact Erlang-chain (Markov) embedding: x' = -x - 2 y_n, y_k' = 8 (y_{k-1} - y_k)
    A = np.zeros((n + 1, n + 1))
    A[0, 0], A[0, n] = -1.0, -2.0
    for k in range(1, n + 1):
        A[k, k], A[k, k - 1] = -float(n), float(n)
    Qc = np.zeros((n + 1, n + 1))
    Qc[0, 0] = 0.2
    import scipy.linalg as sl
    ref = sl.solve_continuous_lyapunov(A, -Qc)[0, 0]
    S = R.stationary_covariance(np.array([[0.2]]))
    assert abs(S[0, 0] - ref) <= 1e-13 * ref
    with mp.workdps(40):
        Am = mp.matrix(A.tolist())
        for t in (0.05, 0.7, 3.0):
            g_ref = float(mp.expm(Am * t)[0, 0])
            assert abs(R.impulse(t)[0, 0] - g_ref) <= 1e-14
    assert R.hurwitz is True


def test_pade8_kernel_balanced_companion():
    prop, params = _kernel_prop(_pade_str(8), 'pade8_realization_test')
    R = RZ.realize(prop, params)
    assert R.cls == 'C' and R.n_states == 9
    assert any('Schwarz form: 1' in p for p in R.provenance)
    cond = R.conditioning
    assert cond['norm_A'] < 1e-3 * cond['norm_A_unbalanced']          # balancing is not optional
    assert np.all(np.log2(R.balance) == np.round(np.log2(R.balance)))   # exact powers of two
    v = RZ.verify_against_K(R, prop['K_ft'], prop['omega'], params, n_points=20, seed=4)
    assert v['max_err'] <= 1e-13 and v['max_err_exact'] <= 1e-40
    for t in (0.2, 1.0, 4.0):
        ref = _mp_impulse(R, t)[0, 0]
        assert abs(R.impulse(t)[0, 0] - ref) <= 1e-13 * max(1.0, abs(ref))
    # tree C(0): refined Lyapunov solve vs a 40-digit one on the exact matrices
    with mp.workdps(40):
        ex = R.exact
        N = ex['A'].nrows()
        q = lambda x: mp.mpf(int(x.numerator())) / int(x.denominator())
        A = mp.matrix([[q(ex['A'][i, j]) for j in range(N)] for i in range(N)])
        B = mp.matrix([[q(ex['B'][i, 0])] for i in range(N)])
        Cm = mp.matrix([[q(ex['C'][0, j]) for j in range(N)]])
        Qm = B * mp.mpf('0.2') * B.T
        M = mp.matrix(N * N, N * N)
        rhs = mp.matrix(N * N, 1)
        for i in range(N):
            for j in range(N):
                rhs[i * N + j] = -Qm[i, j]
                for k in range(N):
                    M[i * N + j, k * N + j] += A[i, k]
                    M[i * N + j, i * N + k] += A[j, k]
        x = mp.lu_solve(M, rhs)
        X = mp.matrix(N, N)
        for i in range(N):
            for j in range(N):
                X[i, j] = x[i * N + j]
        ref = float((Cm * X * Cm.T)[0, 0])
    got = R.stationary_covariance(np.array([[0.2]]))[0, 0]
    assert abs(got - ref) <= 1e-14 * ref


def test_constants_and_float_literals_are_exact_doubles():
    # a transcendental constant (as a saddle value produces) and a float literal
    K = matrix(SR, [[I * W + exp(-SR(13) / 10) + SR(0.1) / (1 + I * W * SR(0.7))]])
    R = RZ.realize_K(K, W, {})
    assert R.cls == 'C' and R.field_name == 'QQ' and R.n_states == 2
    v = RZ.verify_against_K(R, K, W, {}, n_points=20, seed=5)
    assert v['max_err'] <= 1e-13 and v['max_err_exact'] <= 1e-40
    # the kernel rate is 1/0.7 with 0.7 the double, exactly (a chain state)
    from fractions import Fraction
    from sage.all import QQ
    fr = Fraction(0.7)
    assert -QQ(fr.denominator) / fr.numerator in list(R.exact['A'].diagonal())
    assert any('1 first-order block(s) from 1 given rate(s)' in p for p in R.provenance)


# ── refusals by name ───────────────────────────────────────────────────────

def test_index2_descriptor_refused():
    K = matrix(SR, [[I * W, 1], [1, 0]])                    # G = [[0, 1], [1, -s]]: delta'
    R, exc = RZ.try_realize({'K_ft': K, 'omega': W, 'nf': 2}, {})
    assert R is None and exc.reason == 'index_gt_1'
    with pytest.raises(RZ.RealizationRefused, match='index > 1'):
        RZ.realize_K(K, W, {})


def test_non_rational_kernel_refused():
    K = matrix(SR, [[I * W + 1 - exp(-I * W)]])             # a non-rational frequency image
    R, exc = RZ.try_realize({'K_ft': K, 'omega': W, 'nf': 1}, {})
    assert R is None and exc.reason == 'non_rational'
    assert 'K[0,0]' in str(exc)


def test_unbound_and_spatial_refused():
    b, lap = var('b Laplacian')
    _, e1 = RZ.try_realize({'K_ft': matrix(SR, [[I * W + b]]), 'omega': W, 'nf': 1}, {})
    assert e1.reason == 'unbound_symbol' and "'b'" in str(e1)
    _, e2 = RZ.try_realize({'K_ft': matrix(SR, [[I * W - lap + 1]]), 'omega': W, 'nf': 1}, {})
    assert e2.reason == 'spatial'
    _, e3 = RZ.try_realize({'K_ft': matrix(SR, [[I * W, 0], [0, 0]]), 'omega': W, 'nf': 2}, {})
    assert e3.reason == 'singular_pencil'


def test_improper_entry_refused():
    K = matrix(SR, [[-W ** 2 + I * W + 1]])                 # x'' + x' + x
    _, exc = RZ.try_realize({'K_ft': K, 'omega': W, 'nf': 1}, {})
    assert exc.reason == 'improper_entry'


# ── the triangular expm behind impulse() ───────────────────────────────────

@pytest.mark.parametrize('gap', [1e-4, 1e-8, 1e-12, 0.0])
def test_expm_upper_triangular_near_double_eigenvalue(gap):
    T = np.array([[-1.0, 1.0], [0.0, -1.0 - gap]], dtype=complex)
    for t in (0.5, 5.0, 30.0):
        E = RZ.expm_upper_triangular(T, t)
        with mp.workdps(40):
            a, b = mp.mpf(-1) * t, -(1 + mp.mpf(gap)) * t
            ref = float(t * mp.exp(a) if gap == 0 else (mp.exp(a) - mp.exp(b)) / ((a - b) / t))
        assert abs(E[0, 1].real - ref) <= 4e-15 * abs(ref)
        assert abs(E[0, 0].real - np.exp(-t)) <= 4e-15 * np.exp(-t)
