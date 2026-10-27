"""The x-space Dyson tree with a negative eigenvalue gap (fix/dyson-nan).

``dressed_tree_C_x`` integrates ``e^{-m0 t} Phi_n(t; nu)`` terms over s in [0, inf).  With a
gap ``nu = m_a - m_0 < 0`` the factor ``Phi_n`` alone grows like ``e^{|nu| t}``: at large s it
overflowed while ``e^{-m0 t}`` underflowed, 0 * inf = NaN, and the whole integral was NaN (the
unequal-D loop-dressing tests raised).  The combined exponential only decays.
"""
import numpy as np
import pytest
from scipy.integrate import quad
from scipy.linalg import expm, solve_continuous_lyapunov

from daedalus.engine.integration.spatial.dyson_dressing import _exp_phi_batch, dressed_tree_C_x


@pytest.mark.parametrize('m0,nus', [(2.0, [-1.5]), (4.0, [-3.0, -2.5]), (1.0, [3.0]), (1.3, [0.0, -0.9])])
def test_exp_phi_is_finite_and_equals_the_divided_difference(m0, nus):
    """e^{-m0 t} Phi_n = (-1)^n g[m0, m0 + nu_1, ...], g(z) = e^{-t z}: closed forms for n = 1, 2."""
    ts = np.array([0.0, 0.3, 2.0, 40.0, 5e3, 1e6])
    got = _exp_phi_batch(ts, m0, nus)
    assert np.all(np.isfinite(got))
    lam = [m0] + [m0 + v for v in nus]
    for t, g in zip(ts, got):
        e = [np.exp(-t * x) for x in lam]
        if len(nus) == 1:
            ref = t * e[0] if nus[0] == 0 else (e[0] - e[1]) / (lam[1] - lam[0])
        else:
            def dd(a, b, ea, eb, ta):
                return -ta * ea if a == b else (eb - ea) / (b - a)
            d01, d12 = dd(lam[0], lam[1], e[0], e[1], t), dd(lam[1], lam[2], e[1], e[2], t)
            ref = (d12 - d01) / (lam[2] - lam[0]) if lam[2] != lam[0] else None
        if ref is not None:
            assert abs(g - ref) <= 1e-12 * max(abs(ref), 1e-300) + 1e-300, (t, g, ref)


def _exact_unequal_D_line(M, Dm, N, tau, x, legs):
    i, j = legs
    ii, jj = (j, i) if tau >= 0 else (i, j)

    def Cq(q):
        A = M + Dm * q * q
        return np.real((expm(-A * abs(tau)) @ solve_continuous_lyapunov(A, N))[ii, jj])
    if x == 0.0:
        return quad(Cq, 0, np.inf, limit=400, epsabs=1e-14)[0] / np.pi
    return quad(Cq, 0, np.inf, weight='cos', wvar=abs(x), limit=400)[0] / np.pi


def test_dressed_tree_with_a_large_gap_is_finite_and_exact():
    """Masses 1 and 4 (gaps +-3, so strings with nu < 0), weak coupling, unequal D: the Dyson
    tree is finite (it was NaN) and converges geometrically to the exact Lyapunov tree on the
    line (truncation ratio ||D_hat|| / D0 = 0.25)."""
    M = np.array([[1.0, 0.05], [-0.04, 4.0]])
    Dm = np.diag([0.6, 1.0])
    D0 = 0.8
    N = np.diag([2.0, 1.4])
    xs = np.array([0.0, 0.8])
    for legs in ((0, 0), (0, 1)):
        for tau in (0.6, -0.6):
            ex = np.array([_exact_unequal_D_line(M, Dm, N, tau, x, legs) for x in xs])
            errs = []
            for n in (1, 3, 6):
                got = np.real(dressed_tree_C_x(xs, tau, M, Dm - D0 * np.eye(2), D0, N, n, legs))
                assert np.all(np.isfinite(got))
                errs.append(np.max(np.abs(got - ex)) / np.max(np.abs(ex)))
            assert errs[0] > errs[1] > errs[2] and errs[2] < 2e-5, (legs, tau, errs)
