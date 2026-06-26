"""
tests/test_kinematic_spectral.py
================================
Dyson 3c-1: the per-segment-mass spectral kinematic
(``full_integrator.diagram_kinematic_spectral`` + ``spectral_rows``).

  * TREE C edge closed form (exact oracle):
    I(q,τ) = e^{−(m_v+Dq²)τ} / (m_u+m_v+2Dq²)  — incl. COMPLEX masses;
  * uniform-mass limit: I_orig = 2^{n_C}·I_spec on a hand-built bubble AND
    tadpole, q-path and xs-path (the two-segment C representation halves each
    σ integral; the single-field path folds that 2 into its 2^{−n_C});
  * xs-path tree vs direct σ-quadrature oracle;
  * mass-table batching: many assignments in one call == one-at-a-time.

Run:  sage -python -m pytest tests/test_kinematic_spectral.py -q
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from scipy.integrate import quad                                  # noqa: E402

from engine.integration.spatial.diagram_descriptor import (        # noqa: E402
    CEdge, CStackDiagram,
)
from engine.integration.spatial.full_integrator import (           # noqa: E402
    diagram_kinematic, diagram_kinematic_spectral, spectral_rows,
)


def _tree():
    """Single C edge between the two external legs (t=0 at leg 0, t=τ at leg 1)."""
    return CStackDiagram(
        internal_vertices=(),
        external_legs=(0, 1),
        edges=(CEdge(a=(), b=(1.0,), kind='C', u=0, v=1, external=True),),
        n_loops=0)


def _bubble():
    """Hand-built 1-loop bubble: legs 0 (t=0), 1 (t=τ); internal 2, 3.
    Momentum: q in at leg 0 → loop ℓ on the C edge, q−ℓ on the internal R."""
    return CStackDiagram(
        internal_vertices=(2, 3),
        external_legs=(0, 1),
        edges=(
            CEdge(a=(0.0,), b=(1.0,), kind='R', u=0, v=2, external=True),
            CEdge(a=(0.0,), b=(1.0,), kind='R', u=3, v=1, external=True),
            CEdge(a=(1.0,), b=(0.0,), kind='C', u=2, v=3, external=False),
            CEdge(a=(-1.0,), b=(1.0,), kind='R', u=2, v=3, external=False),
        ),
        n_loops=1)


def _tadpole():
    """C self-loop on one internal vertex between the legs."""
    return CStackDiagram(
        internal_vertices=(2,),
        external_legs=(0, 1),
        edges=(
            CEdge(a=(0.0,), b=(1.0,), kind='R', u=0, v=2, external=True),
            CEdge(a=(0.0,), b=(1.0,), kind='R', u=2, v=1, external=True),
            CEdge(a=(1.0,), b=(0.0,), kind='C', u=2, v=2, external=False),
        ),
        n_loops=1)


def test_spectral_rows_expansion():
    rows = spectral_rows(_bubble())
    assert [h for _, _, h in rows] == ['R', 'R', 'Cu', 'Cv', 'R']
    assert len(spectral_rows(_tree())) == 2                       # Cu + Cv


@pytest.mark.parametrize('m_u,m_v', [
    (1.0, 1.0), (0.8, 2.3),
    (1.0 + 0.7j, 1.0 - 0.7j),                                     # conjugate pair
])
def test_tree_closed_form_q(m_u, m_v):
    """Tree C edge: I(q,τ) = e^{−(m_v+Dq²)τ}/(m_u+m_v+2Dq²) exactly."""
    D, q, tau = 0.6, 0.9, 0.7
    et = {0: 0.0, 1: tau}
    table = np.array([m_u, m_v], dtype=complex)
    val = diagram_kinematic_spectral(_tree(), [q], et, table, D, n_s=40)[0]
    mq = D * q * q
    expected = np.exp(-(m_v + mq) * tau) / (m_u + m_v + 2 * mq)
    assert abs(val - expected) < 1e-7 * abs(expected)


def test_tree_xs_vs_quadrature():
    """Tree xs-path vs direct σ-quadrature of the two-segment representation."""
    D, tau = 0.6, 0.5
    m_u, m_v = 0.9, 1.7
    xs = np.array([0.0, 0.8, 1.6])
    et = {0: 0.0, 1: tau}
    val = diagram_kinematic_spectral(_tree(), [0.0], et,
                                     np.array([m_u, m_v]), D, n_s=40, xs=xs)[0]

    def oracle(x):
        def f(s):
            B = D * (tau + 2 * s)
            hk = (4 * np.pi * B) ** -0.5 * np.exp(-x * x / (4 * B))
            return np.exp(-m_u * s - m_v * (tau + s)) * hk
        return quad(f, 0, 60.0, limit=400)[0]

    for i, x in enumerate(xs):
        assert abs(val[i].real - oracle(x)) < 2e-7
        assert abs(val[i].imag) < 1e-12


@pytest.mark.parametrize('mk', ['bubble', 'tadpole'])
@pytest.mark.parametrize('path', ['q', 'xs'])
def test_uniform_limit_matches_single_field(mk, path):
    """Uniform masses: diagram_kinematic == 2^{n_C} · diagram_kinematic_spectral
    (same descriptor, same grids; only the C-edge σ representation differs)."""
    descr = _bubble() if mk == 'bubble' else _tadpole()
    mu, D, tau = 1.3, 0.7, 0.6
    et = {0: 0.0, 1: tau}
    n_C = sum(1 for e in descr.edges if e.kind == 'C')
    table = np.full(len(spectral_rows(descr)), mu, dtype=complex)
    kw = dict(n_t=26, n_s=40)
    if path == 'q':
        ref = diagram_kinematic(descr, [0.8], et, mu, D, **kw)
        new = diagram_kinematic_spectral(descr, [0.8], et, table, D, **kw)[0]
        assert new.imag == pytest.approx(0.0, abs=1e-14)
        assert (2.0 ** n_C) * new.real == pytest.approx(ref, rel=2e-5)
    else:
        xs = np.array([0.0, 1.0])
        ref = diagram_kinematic(descr, [0.0], et, mu, D, xs=xs, **kw)
        new = diagram_kinematic_spectral(descr, [0.0], et, table, D, xs=xs, **kw)[0]
        assert np.allclose((2.0 ** n_C) * new.real, ref, rtol=2e-5)
        assert np.allclose(new.imag, 0.0, atol=1e-14)


def test_mass_table_batching():
    """A (n_rows, n_assign) batch returns exactly the per-column results."""
    descr = _bubble()
    et = {0: 0.0, 1: 0.5}
    rows = spectral_rows(descr)
    rng = np.random.default_rng(7)
    cols = [1.0 + 0.5 * rng.random(len(rows)) + 0.2j * rng.standard_normal(len(rows))
            for _ in range(4)]
    table = np.stack(cols, axis=1)
    ms = float(np.min(table.real))                     # pin one shared grid scale
    batch = diagram_kinematic_spectral(descr, [0.7], et, table, 0.6, mu_scale=ms)
    for j, c in enumerate(cols):
        single = diagram_kinematic_spectral(descr, [0.7], et, c, 0.6,
                                            mu_scale=ms)[0]
        assert abs(batch[j] - single) < 1e-12 * max(1.0, abs(single))


def test_rejects_nonpositive_mass():
    with pytest.raises(ValueError, match='Re m > 0'):
        diagram_kinematic_spectral(_tree(), [0.0], {0: 0.0, 1: 0.5},
                                   np.array([1.0, -0.2]), 0.5)


# ── Dyson loop dressing primitives (D-3 loop, order n=1) ─────────────────────

def test_insertion_identity_vs_finite_difference():
    """The closed-form n=1 insertion factor (−|k_r|²) == (1/D)∂/∂w_r of the
    momentum factor — pinned per-sample against a central finite difference of
    _momentum_factor_batch, for a LOOP row and an EXTERNAL row of the bubble."""
    from engine.integration.spatial.full_integrator import _momentum_factor_batch

    descr = _bubble()
    rows = spectral_rows(descr)
    a = np.array([r[1].a for r in rows], dtype=float).reshape(len(rows), -1)
    b = np.array([r[1].b for r in rows], dtype=float).reshape(len(rows), -1)
    D, qv = 0.7, 0.9
    rng = np.random.default_rng(3)
    w = 0.3 + rng.random((5, len(rows)))                  # (P, n_rows)
    base, Lam, N, ok = _momentum_factor_batch(a, b, w, [qv], D, 1,
                                              return_gaussian=True)
    assert np.all(ok)
    for r in (0, 2):                                      # external row, loop row
        # closed form: (1/D)[−(1/2)g_r − q²·D(b_r − a_rᵀΛ⁻¹N)²]
        LamiA = np.linalg.solve(Lam, np.broadcast_to(a[r], (5, a.shape[1]))[..., None])[..., 0]
        g = LamiA @ a[r]
        u = np.einsum('pl,pl->p', LamiA, N[:, :, 0])
        fac = (-(0.5) * g - qv * qv * D * (b[r, 0] - u) ** 2) / D
        # finite difference of the momentum factor in w_r
        h = 1e-6
        wp, wm = w.copy(), w.copy()
        wp[:, r] += h
        wm[:, r] -= h
        fd = (_momentum_factor_batch(a, b, wp, [qv], D, 1)
              - _momentum_factor_batch(a, b, wm, [qv], D, 1)) / (2 * h * D)
        assert np.allclose(base * fac, fd, rtol=2e-6), (r, base * fac, fd)


def test_insertion_on_external_row_is_d2dx2():
    """Inserting (−|k_r|²) on a purely-external row (k_r = q) multiplies by −q²
    in q-space ⇒ the xs-path insertion equals ∂²/∂x² of the base result."""
    descr = _tree()
    D, tau = 0.6, 0.5
    table = np.array([0.9, 1.7], dtype=complex)
    et = {0: 0.0, 1: tau}
    hx = 0.05
    x0s = np.array([0.4, 1.0])
    xs = np.concatenate([[x - 2 * hx, x - hx, x, x + hx, x + 2 * hx]
                         for x in x0s])
    base = diagram_kinematic_spectral(descr, [0.0], et, table, D, n_s=48,
                                      xs=xs)[0].real
    ins = diagram_kinematic_spectral(descr, [0.0], et, table, D, n_s=48,
                                     xs=x0s, insert_row=0)[0].real
    scale = float(np.max(np.abs(ins)))
    for k, x in enumerate(x0s):
        f = base[5 * k:5 * k + 5]
        # 5-point central second derivative, O(h⁴)
        d2 = (-f[0] + 16 * f[1] - 30 * f[2] + 16 * f[3] - f[4]) / (12 * hx ** 2)
        assert abs(ins[k] - d2) < 2e-3 * scale, (x, ins[k], d2)
    # q-path: trivially −q² × base
    qv = 0.8
    b_q = diagram_kinematic_spectral(descr, [qv], et, table, D, n_s=48)[0]
    i_q = diagram_kinematic_spectral(descr, [qv], et, table, D, n_s=48,
                                     insert_row=1)[0]
    assert i_q == pytest.approx(-qv * qv * b_q, rel=1e-10)


def test_power_table_confluent_amplitude():
    """κ=1 on the late C half: ∫dσ (τ+σ)·e^{−m_uσ−m_v(τ+σ)}e^{−Dq²(τ+2σ)} —
    the confluent dressed-segment form w·e^{−mw}, vs direct quadrature."""
    descr = _tree()
    D, qv, tau = 0.6, 0.7, 0.5
    m_u, m_v = 0.9, 1.4
    table = np.array([m_u, m_v], dtype=complex)
    powers = np.array([0.0, 1.0])
    val = diagram_kinematic_spectral(descr, [qv], {0: 0.0, 1: tau}, table, D,
                                     n_s=48, power_table=powers)[0]

    def f(s):
        return ((tau + s) * np.exp(-m_u * s - m_v * (tau + s))
                * np.exp(-D * qv * qv * (tau + 2 * s)))
    oracle = quad(f, 0, 80.0, limit=400)[0]
    assert val.real == pytest.approx(oracle, rel=1e-7)
    assert abs(val.imag) < 1e-14


def test_insertion_order2_vs_fd():
    """Order-2 Dyson insertion factors (same-row n=2 AND cross-row 1+1)
    vs Richardson-extrapolated finite differences of the closed-form
    momentum factor, on BOTH the q-path and the analytic-IFT xs-path:

        factor(r,s) = E_r E_s + (1/D^2) d2 lnV / dw_r dw_s,
        E_r = (1/D) d lnV / dw_r .

    The implemented pieces (g_r, v_r, g_rs and the h1/h2 B-derivatives)
    are replicated here independently from first principles."""
    import numpy as np
    from engine.integration.spatial.full_integrator import (
        _momentum_factor_batch, _symanzik_kernel_batch)

    rng = np.random.default_rng(3)
    E, L, D, d, qv, xv = 6, 2, 1.3, 1.0, 0.7, 0.9
    a = rng.normal(size=(E, L))
    b = rng.normal(size=(E, 1))
    w0 = rng.uniform(0.3, 1.5, size=(1, E))

    def Vq(w):
        return complex(_momentum_factor_batch(
            a, b, w.reshape(1, E), [qv], D, 1)[0]).real

    def Vx(w):
        pref, B, ok = _symanzik_kernel_batch(a, b, w.reshape(1, E), D, 1)
        B = float(B[0])
        return float(pref[0]) * (4*np.pi*B)**(-0.5) * np.exp(-xv**2/(4*B))

    def fd2(V, r, s, eps):
        if r == s:
            wp = w0.copy(); wp[0, r] += eps
            wm = w0.copy(); wm[0, r] -= eps
            num = (V(wp) - 2*V(w0) + V(wm)) / eps**2
        else:
            num = 0.0
            for sr, ss in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
                wp = w0.copy(); wp[0, r] += sr*eps; wp[0, s] += ss*eps
                num += sr*ss*V(wp)
            num /= 4*eps**2
        return num / V(w0) / D**2

    def rich(V, r, s):
        return (4*fd2(V, r, s, 2e-4) - fd2(V, r, s, 4e-4)) / 3

    w = w0[0]
    Lam = (w[:, None]*a).T @ a
    N = (w[:, None]*a).T @ b
    Q = float((w * b[:, 0]**2).sum())
    Lami = np.linalg.inv(Lam)
    B = D * (Q - float(N[:, 0] @ Lami @ N[:, 0]))
    g = lambda r_, s_: float(a[r_] @ Lami @ a[s_])
    v = lambda r_: float(b[r_, 0] - a[r_] @ Lami @ N[:, 0])
    h1 = -(d/2)/B + xv**2/(4*B**2)
    h2 = (d/2)/B**2 - xv**2/(2*B**3)

    def f_q(r, s):
        Er = (-(d/2)*g(r, r) - qv**2*D*v(r)**2)/D
        Es = (-(d/2)*g(s, s) - qv**2*D*v(s)**2)/D
        X = ((d/2)*g(r, s)**2 + 2*D*qv**2*g(r, s)*v(r)*v(s))/D**2
        return Er*Es + X

    def f_x(r, s):
        Er = (-(d/2)*g(r, r) + h1*D*v(r)**2)/D
        Es = (-(d/2)*g(s, s) + h1*D*v(s)**2)/D
        X = ((d/2)*g(r, s)**2 + h2*(D*v(r)**2)*(D*v(s)**2)
             - 2*D*g(r, s)*v(r)*v(s)*h1)/D**2
        return Er*Es + X

    for (r, s) in ((0, 0), (1, 3), (2, 2), (4, 5), (0, 5)):
        for V, f in ((Vq, f_q), (Vx, f_x)):
            fd = rich(V, r, s)
            fo = f(r, s)
            assert abs(fd - fo)/max(abs(fo), 1e-300) < 1e-5, (r, s, fd, fo)


def test_insertion_general_order_vs_symbolic():
    """ANY-order insertion factors vs EXACT symbolic differentiation of
    lnV(w) (sympy, no FD noise): orders 1-4, same/mixed/repeated rows,
    q-path and xs-path.  Pins the partition expansion
    (1/D^n) d^n V / V = sum_{partitions} prod_blocks (1/D^|b|) d^b lnV
    with d^b lnU = closed g-chains and d^b B = open v-g-v chains."""
    import numpy as np
    import sympy as sp
    from math import factorial as fct
    from engine.integration.spatial.full_integrator import (
        _set_partitions, _dlnU_block, _dB_block)

    rng = np.random.default_rng(5)
    E, L, D, d, qv, xv = 5, 2, 1.3, 1.0, 0.7, 0.9
    a = rng.normal(size=(E, L))
    b = rng.normal(size=(E, 1))
    w0 = rng.uniform(0.3, 1.5, size=E)
    ws = sp.symbols('w0:%d' % E, positive=True)
    Lam_s = sp.Matrix(L, L, lambda i, j: sum(ws[e]*a[e, i]*a[e, j]
                                             for e in range(E)))
    N_s = sp.Matrix(L, 1, lambda i, j: sum(ws[e]*a[e, i]*b[e, 0]
                                           for e in range(E)))
    Q_s = sum(ws[e]*b[e, 0]**2 for e in range(E))
    U_s = Lam_s.det()
    B_s = D*(Q_s - (N_s.T*Lam_s.inv()*N_s)[0, 0])
    lnV = {'q': -sp.Rational(1, 2)*d*sp.log(U_s) - qv**2*B_s,
           'x': (-sp.Rational(1, 2)*d*sp.log(U_s)
                 - sp.Rational(1, 2)*d*sp.log(B_s) - xv**2/(4*B_s))}
    subs0 = {ws[e]: float(w0[e]) for e in range(E)}

    def oracle(path, rows):
        expr = sp.exp(lnV[path])
        for r in rows:
            expr = sp.diff(expr, ws[r])
        return float(sp.N(expr.subs(subs0)
                          / sp.exp(lnV[path]).subs(subs0), 30)) / D**len(rows)

    Lam = (w0[:, None]*a).T @ a
    N = (w0[:, None]*a).T @ b
    Lami = np.linalg.inv(Lam)
    Gd = {(r, s): np.array([float(a[r] @ Lami @ a[s])])
          for r in range(E) for s in range(E)}
    Vd = {r: np.array([float(b[r, 0] - a[r] @ Lami @ N[:, 0])])
          for r in range(E)}
    Bv = D*(float((w0*b[:, 0]**2).sum()) - float(N[:, 0] @ Lami @ N[:, 0]))

    def hm(m):
        return (-(0.5*d)*((-1.0)**(m-1))*fct(m-1)/Bv**m
                - 0.25*xv**2*((-1.0)**m)*fct(m)/Bv**(m+1))

    def factor(rows, path):
        tot = 0.0
        for part in _set_partitions(list(range(len(rows)))):
            term = 1.0
            for blk in part:
                rb = [rows[i] for i in blk]
                if path == 'q':
                    dlnf = -qv**2*float(_dB_block(rb, Gd, Vd, D)[0])
                else:
                    dlnf = 0.0
                    for p2 in _set_partitions(list(range(len(rb)))):
                        t2 = hm(len(p2))
                        for b2 in p2:
                            t2 *= float(_dB_block([rb[i] for i in b2],
                                                  Gd, Vd, D)[0])
                        dlnf += t2
                term *= (-(0.5*d)*float(_dlnU_block(rb, Gd)[0])
                         + dlnf)/D**len(rb)
            tot += term
        return tot

    for rows in [(0,), (1, 3), (2, 2), (0, 1, 2), (1, 1, 3), (4, 4, 4),
                 (0, 1, 2, 3), (2, 2, 3, 3)]:
        for path in ('q', 'x'):
            o = oracle(path, rows)
            f = factor(rows, path)
            assert abs(f - o)/max(abs(o), 1e-300) < 1e-10, (rows, path, f, o)
