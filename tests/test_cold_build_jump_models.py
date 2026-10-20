"""Cold build of small multi-field jump models: no Maxima zero tests.

A network of binary (Glauber) units with numeric polynomial rates puts float
coefficients and the symbolic saddle values ``x<i>star1`` into every entry of
the kernel matrix (``I*omega + 1.0`` on the diagonal, float polynomials in the
saddle values off it).  Sage's ``bool(c)`` cannot certify a float expression
by its random-point test and asks Maxima on every truth test: the symbolic
inverse of the 3x3 kernel matrix of the three-unit network took 101 s (and the
whole symbolic block ~183 s, then gave up), and the zero test of the whole
expanded action of the same network at taylor order 3 overflowed the ECL
binding stack.

Checks:
  * ``_sr_nonzero`` gives ``bool``'s verdict on float, complex-float and
    float-dust expressions (incl. a complex ``(a+b)^2 - a^2 - 2ab - b^2``
    that is exactly nonzero below double rounding, zero to ``bool``);
  * ``_sr_inverse`` returns exactly the expressions of ``K.inverse()``
    (``repr`` of every entry), incl. a pivot swap and a singular matrix;
  * the three-unit network expands and builds its symbolic propagator in
    seconds, and its tree covariance equals the Lyapunov solution of the
    linearized dynamics (an independent reference).
"""
import math

import numpy as np
import pytest

sage = pytest.importorskip('sage.all')
from sage.all import SR, I, matrix, exp, sqrt, assume, forget

import engine.core.field_theory as F
import api._propagator as P
from api.model import TemporalTheoryBuilder
from tests._cputime import CpuClock  # noqa: E402

_J = [(0, 1, 1.0), (1, 2, -0.8), (0, 2, -0.5)]
_H = [0.3, -0.2, 0.1]
_BETA = 0.3


def _network(n=3, deg=3):
    """``n`` binary units, up-rate ``rp(I_i)`` and down-rate ``1 - rp(I_i)``,
    ``rp(u) = (1 + tanh u)/2`` replaced by its Taylor polynomial (numeric
    coefficients) about the mean-field input; ``I_i = beta (sum_j J_ij
    (2 x_j - 1) + h_i)``.  Returns the model, the saddle and the linearized
    drift / noise matrices."""
    from scipy.optimize import fsolve
    J = np.zeros((n, n))
    for i, j, v in _J:
        if i < n and j < n:
            J[i, j] = J[j, i] = v
    h = np.asarray(_H[:n])
    m = fsolve(lambda m: m - np.tanh(_BETA * (J @ m + h)), np.tanh(_BETA * h),
               xtol=1e-14)
    xs = (1 + m) / 2
    u = _BETA * (J @ m + h)
    b = TemporalTheoryBuilder(f'glauber net n{n} deg{deg}').population(
        'pop', size=1)
    for i in range(n):
        b = b.physical_field(f'x{i + 1}', population='pop')
    terms = []
    for i in range(n):
        t = math.tanh(u[i])
        # d^k tanh / du^k at u_i, as polynomials in t: P_{k+1} = P_k' (1 - t^2)
        P_k = np.polynomial.Polynomial([0, 1])
        c = []
        for k in range(deg + 1):
            c.append(float(P_k(t)) / math.factorial(k) / 2)
            P_k = P_k.deriv() * np.polynomial.Polynomial([1, 0, -1])
        c[0] += 0.5
        lin = ' + '.join(f'({float(2 * _BETA * J[i, j])!r})*(x{j + 1}[i] - '
                         f'({float(xs[j])!r}))' for j in range(n) if J[i, j])
        R = ' + '.join([repr(c[0])] + [f'({c[k]!r})*({lin})^{k}'
                                       for k in range(1, deg + 1)])
        xi, xt = f'x{i + 1}[i]', f'x{i + 1}t[i]'
        terms.append(f'{xt}*Dt*{xi} - (exp({xt}) - 1)*(1 - {xi})*({R})'
                     f' - (exp(-{xt}) - 1)*{xi}*(1 - ({R}))')
    b = b.set_action_text('sum( ' + ' + '.join(terms) + ' for i in pop)')
    for i in range(n):
        b = b.set_mf_equation(f'x{i + 1}star', repr(float(xs[i])))
    c1 = (1 - np.tanh(u) ** 2) / 2
    A = -np.eye(n) + np.diag(c1) @ (2 * _BETA * J)
    B = np.diag(2 * xs * (1 - xs))
    return b.build(), xs, A, B


def test_sr_nonzero_floats_complex_and_dust():
    x, y, w = SR.var('x_cb y_cb w_cb')
    p = SR.var('p_cb')
    assume(p > 0)
    try:
        a = SR(1e-17) * I * x * y + SR(0.9)
        bb = SR(0.8333333333333333) * I * w - SR(0.4) * I
        cases = [
            I * w + SR(1.0),                                 # complex, nonzero
            SR(0.3) * p * x ** 2 - SR(0.1) * x + SR(0.2) * I * w,
            SR(5.551115123125783e-17) * x,                   # tiny, nonzero
            (x + SR(0.1)) ** 2 - x ** 2 - SR(0.2) * x - SR(0.01),  # dust: True
            SR(0.1) * (x + y) * x - SR(0.1) * x ** 2 - SR(0.1) * x * y,
            (a + bb) ** 2 - a ** 2 - 2 * a * bb - bb ** 2,   # below rounding
            SR(0.5) * I * w - I * w / 2,
            exp(SR(0.3) * I * w) * x - x * exp(SR(0.3) * I * w),
            1 / (I * w + SR(0.7)) - 1 / (I * w + SR(0.7)),
            sqrt(I * w + 1) - SR(0.2),                       # branch cut: bool
        ]
        for c in cases:
            assert F._sr_nonzero(c) == bool(c), c
    finally:
        forget(p > 0)


def _same(G0, G1):
    return all(repr(G0[i, j]) == repr(G1[i, j])
               for i in range(G0.nrows()) for j in range(G0.ncols()))


def test_sr_inverse_is_sage_inverse():
    x, w = SR.var('x_ci w_ci')
    mats = [
        matrix(SR, [[I * w + SR(1.0), SR(-0.26) - SR(0.1) * x ** 2],
                    [SR(-0.29) + SR(0.05) * x, I * w + SR(1.0)]]),
        matrix(SR, [[0, I * w + 2, SR(0.3)],                # pivot swap
                    [x + SR(0.5), 1, 0],
                    [SR(0.2) * x, SR(0.1), I * w / 3 + 1]]),
        matrix(SR, [[x, 1], [2, 1 / (I * w + 1)]]),          # exact entries
        matrix(SR, [[SR(2.5)]]),
    ]
    for K in mats:
        assert _same(K.inverse(), P._sr_inverse(K)), K
    # symbolic parameters with assumptions, 4x4 and 5x5 (an I*omega
    # diagonal, positive rates, couplings in symbols and floats)
    a, b, c = SR.var('a_ci b_ci c_ci')
    assume(a > 0)
    assume(b > 0)
    try:
        for n, dense in ((4, True), (5, False)):
            rows = []
            for i in range(n):
                row = []
                for j in range(n):
                    if i == j:
                        row.append(I * w + a + i * b)
                    elif j == (i + 1) % n:
                        row.append(-c * SR(0.3 + 0.1 * i))
                    elif dense and (i + j) % 3 == 0:
                        row.append(b / (1 + I * w * a) - SR(0.05) * j)
                    else:
                        row.append(SR(0))
                rows.append(row)
            # the 5x5 is a ring: Sage's own inverse of the dense 5x5, the
            # reference here, takes minutes (what the replica removes)
            K = matrix(SR, rows)
            assert _same(K.inverse(), P._sr_inverse(K)), n
    finally:
        forget(a > 0)
        forget(b > 0)
    sing = matrix(SR, [[x, 2 * x], [SR(1), SR(2)]])
    with pytest.raises(ZeroDivisionError):
        sing.inverse()
    with pytest.raises(ZeroDivisionError):
        P._sr_inverse(sing)


def test_three_unit_network_cold_build_and_lyapunov():
    import scipy.linalg as sla
    from api.compute import compute_cumulants
    model, xs, A, B = _network(3)
    clock = CpuClock()                        # CPU seconds, Maxima child included
    ft = F.FieldTheory(model, taylor_order=3)
    ft.expand()
    prop = P.build_propagator(ft, model, use_cache=False, verbose=False)
    wall = clock.elapsed()
    assert prop['G_ft'] is not None          # the symbolic block completed
    assert wall < 60, wall                    # ~183 s (and no G_ft) before
    C = sla.solve_continuous_lyapunov(A, -B)
    for (a, b) in [(1, 1), (1, 3)]:
        r = compute_cumulants(model, k=2, max_ell=0,
                              external_fields=[(f'dx{a}', 1), (f'dx{b}', 1)],
                              tau_grid=np.array([0.0]), parallel=False,
                              verbose=False, use_cache=False)
        v = float(np.real(np.atleast_1d(r['C_tau_by_ell'][0]))[0])
        assert abs(v - C[a - 1, b - 1]) <= 1e-12 * abs(C[a - 1, a - 1]), (
            a, b, v, C[a - 1, b - 1])


def test_numpy_scalar_reprs_in_action_text():
    """``f'{x!r}'`` of a numpy >= 2 scalar is ``np.float64(0.25)``: the
    action text reads it as the number (it raised a NameError on ``np``)."""
    from api.model_compiler import _normalize_expr_text
    assert (_normalize_expr_text('np.float64(0.5)*x - numpy.int64(3) + '
                                 'np.float64(-3e-05)')
            == '(0.5)*x - (3) + (-3e-05)')
    a = np.float64(0.7)
    model = (TemporalTheoryBuilder('np repr ou').population('pop', size=1)
             .physical_field('x', population='pop')
             .set_action_text(f'sum(xt[i]*(Dt*x[i] + {a!r}*x[i]) '
                              f'- {np.float64(0.5)!r}*xt[i]^2 for i in pop)')
             .set_mf_equation('xstar', '0')
             .build())
    ft = F.FieldTheory(model, taylor_order=2)
    ft.expand()
    K = {k: v for k, v in ft._by_tp.items() if k == (1, 1)}
    assert K and '0.7' in str(K[(1, 1)])


def _network_symbolic(n=2, deg=3):
    """The network of :func:`_network` with its numbers as SYMBOLIC indexed
    parameters (couplings, saddle, Taylor coefficients of the rate)."""
    from scipy.optimize import fsolve
    J = np.zeros((n, n))
    for i, j, v in _J:
        if i < n and j < n:
            J[i, j] = J[j, i] = v
    h = np.asarray(_H[:n])
    m = fsolve(lambda m: m - np.tanh(_BETA * (J @ m + h)), np.tanh(_BETA * h),
               xtol=1e-14)
    xs = (1 + m) / 2
    u = _BETA * (J @ m + h)
    C = np.zeros((n, deg + 1))
    for i in range(n):
        t = math.tanh(u[i])
        P_k = np.polynomial.Polynomial([0, 1])
        for k in range(deg + 1):
            C[i, k] = float(P_k(t)) / math.factorial(k) / 2
            P_k = P_k.deriv() * np.polynomial.Polynomial([1, 0, -1])
        C[i, 0] += 0.5
    b = (TemporalTheoryBuilder(f'glauber net symbolic n{n} deg{deg}')
         .population('pop', size=n).physical_field('x', population='pop')
         .parameter('J', default=J.tolist(), indexed_by=['pop', 'pop'])
         .parameter('xs', default=xs.tolist(), indexed_by=['pop'])
         .parameter('beta', default=float(_BETA), domain='positive'))
    for k in range(deg + 1):
        b = b.parameter(f'c{k}', default=C[:, k].tolist(), indexed_by=['pop'])
    D = '(2*beta*sum(J[i,j]*(x[j] - xs[j]) for j in pop))'
    R = ' + '.join(['c0[i]'] + [f'c{k}[i]*{D}^{k}' for k in range(1, deg + 1)])
    return (b.set_action_text(f'sum( xt[i]*Dt*x[i] - (exp(xt[i]) - 1)*(1 - x[i])'
                              f'*({R}) - (exp(-xt[i]) - 1)*x[i]*(1 - ({R}))'
                              f' for i in pop)')
            .set_mf_equation('xstar', 'xs[i]').build())


def test_symbolic_coefficient_network_one_loop_timed():
    """Two units, symbolic indexed coefficients, k = 1 one loop: 474.7 s and
    5.3 GB before (the symbolic inverse block and the per-diagram source
    zero tests in Maxima); ~18 s now.  Same mean as the numeric network."""
    from api.compute import compute_cumulants
    kw = dict(k=1, max_ell=1, tau_grid=np.array([0.0]), parallel=False,
              verbose=False, use_cache=False)
    clock = CpuClock()
    r = compute_cumulants(_network_symbolic(2), external_fields=[('dx', 1)],
                          **kw)
    wall = clock.elapsed()
    assert wall < 120, wall
    rn = compute_cumulants(_network(2)[0], external_fields=[('dx1', 1)], **kw)
    v = float(np.real(np.atleast_1d(r['C_tau_by_ell'][1]))[0])
    vn = float(np.real(np.atleast_1d(rn['C_tau_by_ell'][1]))[0])
    assert abs(v - vn) <= 1e-12 * abs(vn), (v, vn)
