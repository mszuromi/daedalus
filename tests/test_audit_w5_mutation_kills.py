"""Tests that kill the audit's surviving mutants (audit W5, TST-03).

Mutation testing of the numerical core left lines whose result can change with no test
noticing.  Each test below pins one of them against an independent value:

  * Laurent residues at repeated roots (``api._pole_clusters._laurent_residue``): double and
    triple pole residues against the closed-form 1/(lambda - rho) coefficients;
  * the cancellation factor ``kappa`` of a pole cluster (``_kappa``): decoupled close poles do not
    cancel (kappa = 1 exactly);
  * the Taylor degree selection of the state-space exponential (``_taylor_params``, Al-Mohy &
    Higham: max(d_p, d_{p+1})): a stiff non-normal matrix against the 50-digit exponential;
  * the exact delayed tree (``api._delay_exact``): a rational non-delayed part K0 with a nontrivial
    denominator and a dynamic delayed kernel K1 against an independent spectral integral, and the
    route actually taken ('exact', not the Pade fallback);
  * the neutral-system guard of the exact delayed tree (D0 D1 != 0 is refused).
"""
import math

import mpmath as mp
import numpy as np
import pytest
import scipy.linalg as sl

import api.noise as N
from api.compute import compute_cumulants
from api.delay import Kernel
from api.sde import SDE


# ----------------------------------------------------------------------------- pole clusters

def _fracfield_matrix(entries_lam, nf):
    """A ``Frac(QQ[i][omega])`` propagator from entries given as functions of lambda = i omega
    (the convention of ``api._pole_clusters``: ``p(omega) -> p(-i lambda)``), with the common
    denominator."""
    from sage.all import CyclotomicField, PolynomialRing, matrix
    CF = CyclotomicField(4)
    R = PolynomialRing(CF, 'omega')
    F = R.fraction_field()
    lam = CF.gen() * F(R.gen())
    G = matrix(F, nf, nf, lambda i, j: entries_lam[i][j](lam) if entries_lam[i][j] else 0)
    Q = None
    for i in range(nf):
        for j in range(nf):
            if G[i, j] != 0:
                d = G[i, j].denominator()
                Q = d if Q is None else Q.lcm(d)
    return G, Q


def _residue_at(an, pole_lambda, i, j):
    """C[i][j] of the nominal residues at the retarded root lambda = pole_lambda."""
    pole_vals, C_mats = an.nominal
    target = complex(-1j * pole_lambda)
    k = int(np.argmin([abs(p - target) for p in pole_vals]))
    assert abs(pole_vals[k] - target) < 1e-12
    return complex(C_mats[k][i, j])


def test_laurent_residues_at_double_and_triple_poles():
    """x' = -a x + y, y' = -a y + z, z' = -a z + w, w' = -b w (a chain): G_xz has a double root and
    G_xw a triple root at lambda = -a.  The 1/(lambda + a) coefficients are
    d/dl (l + b)^-1 = -1/(b - a)^2 and (1/2) d^2/dl^2 (l + b)^-1 = 1/(b - a)^3."""
    from sage.all import QQ
    from api import _pole_clusters as PC
    a, b = QQ(1) / 2, QQ(2)
    E = [[None] * 4 for _ in range(4)]
    E[0][0] = lambda l: 1 / (l + a)
    E[0][2] = lambda l: 1 / ((l + a) ** 2 * (l + b))
    E[0][3] = lambda l: 1 / ((l + a) ** 3 * (l + b))
    E[1][1] = lambda l: 1 / (l + a)
    G, Q = _fracfield_matrix(E, 4)
    an = PC.analyze(G, Q, 4)
    fa, fb = float(a), float(b)
    assert abs(_residue_at(an, -fa, 0, 2) - (-1 / (fb - fa) ** 2)) < 1e-14
    assert abs(_residue_at(an, -fa, 0, 3) - 1 / (fb - fa) ** 3) < 1e-14
    assert abs(_residue_at(an, -fa, 0, 0) - 1.0) < 1e-14       # simple pole inside the triple root
    # the simple root -b: residue of 1/((l + a)^3 (l + b)) is 1/(a - b)^3
    assert abs(_residue_at(an, -fb, 0, 3) - 1 / (fa - fb) ** 3) < 1e-14
    assert abs(_residue_at(an, -fb, 0, 2) - 1 / (fa - fb) ** 2) < 1e-14


def test_decoupled_close_poles_do_not_cancel():
    """G = diag(-3/(lambda + a), 1/(lambda + a + delta)): the two close poles never meet in one
    entry, so nothing cancels and the cluster's cancellation factor is exactly 1."""
    from sage.all import QQ
    from api import _pole_clusters as PC
    a, delta = QQ(1), QQ(1) / 1000
    E = [[lambda l: -3 / (l + a), None], [None, lambda l: 1 / (l + a + delta)]]
    G, Q = _fracfield_matrix(E, 2)
    an = PC.analyze(G, Q, 2)
    assert an.clusters, 'the pair must form a cluster'
    assert abs(an.kappa_max - 1.0) < 1e-12, an.report()


def test_close_poles_in_one_entry_cancel_by_the_known_factor():
    """G = 1/((lambda + a)(lambda + a + delta)): residues +-1/delta cancel.  With s the cluster's
    scale, kappa = sum|C| / max_r |sum C ((lambda_k - c)/s)^r| = (2/delta) / (1/s) = 2 s / delta."""
    from sage.all import QQ
    from api import _pole_clusters as PC
    a, delta = QQ(1), QQ(1) / 1000
    G, Q = _fracfield_matrix([[lambda l: 1 / ((l + a) * (l + a + delta))]], 1)
    an = PC.analyze(G, Q, 1)
    (c,) = an.clusters
    s = max(c['d_lim'], float(delta))          # 2 x the cluster radius delta/2
    assert abs(an.kappa_max / (2 * s / float(delta)) - 1) < 1e-12, an.report()


# ----------------------------------------------------------------------------- state-space Taylor action

@pytest.mark.parametrize('t', [2.0, 5.33])
def test_taylor_action_on_a_stiff_non_normal_matrix(t):
    """Al-Mohy & Higham choose the degree and the number of steps from
    alpha_p = max(d_p, d_{p+1}), d_p = ||A^p||^{1/p}.  Here (a rate-1.9e3 decay fed from a
    zero-rate state, off-diagonal 254) a selection from d_p alone loses two to three digits.
    Reference: the 50-digit matrix exponential."""
    from engine.integration.time_domain import statespace as S
    A = np.array([[0.0, 0.0], [254.0, -1927.0]])
    x = np.array([1.12, -0.9])
    mu = float(np.trace(A)) / 2
    As = A - mu * np.eye(2)
    got = S._taylor_action(lambda v: As @ v, mu, S._onenorm_powers(As), x, t)
    with mp.workdps(50):
        r = mp.expm(mp.matrix(A.tolist()) * t) * mp.matrix(x.tolist())
    ref = np.array([float(r[0]), float(r[1])])
    assert np.max(np.abs(got - ref)) <= 1e-14 * np.max(np.abs(ref)), (got, ref)


# ----------------------------------------------------------------------------- exact delayed tree

DK = dict(a=1.5, b=0.4, tk=0.7, c=-0.6, d=0.8, m=0.5, D=0.2)


def _delayed_kernel_model():
    """x' = -a x + b (g * x) + c (g_d * x) + xi: g exponential (K0 = s + a - b/(1 + s tk), a
    nontrivial denominator), g_d a delayed Erlang-2 kernel (K1 = -c e^{-sd}/(1 + s m/2)^2, dynamic)."""
    s = SDE('w5 delayed kernel tree')
    for k, v in DK.items():
        s = s.parameter(k, default=v, **({'domain': 'positive'} if k in ('tk', 'd', 'm', 'D') else {}))
    return (s.define_kernel('g', kernel=Kernel.exponential('tk'))
            .define_kernel('gd', kernel=Kernel.delay('d') * Kernel.gamma(2, 'm'))
            .field('x').drift('x', '-a*x + b*Conv(g, x) + c*Conv(gd, x)')
            .noise('xi', N.Gaussian(var='2*D'), couples={'x': 1}).build())


def _spectral_reference(tau):
    """C(tau) = (1/pi) int_0^oo cos(w tau) 2D / |K(i w)|^2 dw, with the OU part 2D/(w^2 + a^2)
    subtracted (its integral is (D/a) e^{-a tau}) so the remainder decays like w^-4."""
    with mp.workdps(30):
        a, b, tk, c, d, m, D = (mp.mpf(DK[k]) for k in ('a', 'b', 'tk', 'c', 'd', 'm', 'D'))
        tau = mp.mpf(tau)

        def K(w):
            s = 1j * w
            return s + a - b / (1 + s * tk) - c * mp.exp(-s * d) / (1 + s * m / 2) ** 2
        f = lambda w: mp.cos(w * tau) * (2 * D / abs(K(w)) ** 2 - 2 * D / (w * w + a * a))
        pts = [0] + [mp.pi * j / d for j in range(1, 400)] + [mp.inf]
        return mp.quad(f, pts) / mp.pi + D / a * mp.exp(-a * tau)


def test_exact_delayed_tree_with_a_dynamic_delayed_kernel_matches_the_spectral_integral():
    taus = [0.0, 0.3, 1.1, 2.0]
    r = compute_cumulants(_delayed_kernel_model(), k=2, max_ell=0, external_fields=[('x', 1), ('x', 1)],
                          tau_grid=np.array(taus), parallel=False, verbose=False)
    assert r['delay']['tree'] == 'exact', r['delay'].get('tree_note')
    got = np.asarray(r['C_tau_by_ell'][0], complex)
    ref = np.array([complex(_spectral_reference(t)) for t in taus])
    # the reference quadrature is good to ~5e-12 of C(0) here
    assert np.max(np.abs(got - ref)) <= 1e-10 * abs(ref[0]), (got, ref)


def test_exact_delayed_tree_refuses_an_instantaneous_path_through_the_delay():
    """y = x + c y(t - d) + noise (algebraic y with a delayed self-term): D0 D1 != 0, a neutral
    system, which the exact tree does not cover."""
    from sage.all import SR, I, exp, matrix
    from api._delay_exact import ExactTreeUnavailable, delayed_state_space
    omega = SR.var('omega')
    K = matrix(SR, [[I * omega + SR(3) / 2, 0], [-1, 1 - SR(2) / 5 * exp(-I * omega * SR(4) / 5)]])
    with pytest.raises(ExactTreeUnavailable, match='D0 D1'):
        delayed_state_space(K, omega, {}, np.eye(2))
    # the same system without the instantaneous path (the delayed term on x) is accepted
    K_ok = matrix(SR, [[I * omega + SR(3) / 2 - SR(2) / 5 * exp(-I * omega * SR(4) / 5), 0], [-1, 1]])
    ss = delayed_state_space(K_ok, omega, {}, np.eye(2))
    assert math.isclose(ss.d, 0.8, rel_tol=1e-15)
