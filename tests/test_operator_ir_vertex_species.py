"""
tests/test_operator_ir_vertex_species.py
========================================
A derivative vertex and a plain local vertex of the same fields stay distinct
SPECIES in the spatial operator IR.

``c1*Lap(phi^2)`` and ``c2*phi^2`` both unfold to a ``phit*phi^2`` vertex.  The
lowering used to merge them into ONE coefficient ``c1 + c2`` carrying the
Laplacian form factor of the derivative vertex: ``c1 = -c2`` cancelled the
vertex (the loop run then failed as a free model), and any other pair put the
Laplacian on the local coupling as well (silently wrong numbers).  Each species
now keeps its own coupling and form factor in ``ns._operator_ir_vertex_terms``
(a local species has an empty chain, form factor 1), and the unfolded action
carries the group norm ``sqrt(c1^2 + c2^2)``.

Model: ``dphi/dt = -mu phi + D Lap phi + c1 Lap(phi^2) + c2 phi^2 + eta``,
``<eta eta> = 2T delta delta``, infinite line.

References (independent of the package):

* k=3 tree: the tree-level third cumulant by direct Fourier quadrature of
  ``sum_a int ds dy R(x_a - y, t_a - s) (c2 + c1 Lap_y) 2 C(y - x_b) C(y - x_c)``;
* k=2 tree at a non-zero saddle (a source ``h``): the erfc closed form of the
  linear response with ``m = mu - 2 c2 phi*`` and ``D_eff = D + 2 c1 phi*``;
* k=2 one loop: the spectral lattice simulator, paired free / interacting runs
  on common noise (slow).

Run:  PYTHONHASHSEED=0 sage -python -m pytest tests/test_operator_ir_vertex_species.py -q
"""
from __future__ import annotations

import math
import os
import sys
import warnings

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from scipy.special import erfc
from sage.all import SR

from api.model import SpatialModelBuilder
from api import compute_cumulants

_MODELS = os.path.join(os.path.dirname(__file__), '..', 'models')
_ACTION = ('phit*(Dt(phi) + mu*phi - D*Lap(phi) - c1*Lap(phi^2) - c2*phi^2'
           '{src}) - T*phit^2')


def _built(P, rhs, action):
    b = (SpatialModelBuilder('lap and local phi^2', n_populations=0)
         .physical_field('phi', spatial_dim=1))
    for k, v in P.items():
        b = b.parameter(k, default=v,
                        domain='positive' if k in ('mu', 'D', 'T') else 'real')
    # rhs: the homogeneous saddle (Lap / Dx terms vanish on it)
    return (b.equation(lhs='(Dt + mu - D*Laplacian)*phi', rhs=rhs)
            .set_action_text(action)
            .operator_ir().boundary('infinite').initial('stationary').build())


def _mixed(c1, c2, mu=1.0, D=1.0, h=None, action=_ACTION):
    P = {'mu': mu, 'D': D, 'c1': c1, 'c2': c2, 'T': 1.0}
    if h is not None:
        P['h'] = h
    rhs = 'c2*phi^2' + (' + h' if h is not None else '')
    m = _built(P, rhs, action.format(src=' - h' if h is not None else ''))
    return m, P


def _k2(m, P, xs, taus=(0.0,), max_ell=1):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        r = compute_cumulants(m, k=2, max_ell=max_ell,
                              external_fields=[('phi', 1), ('phi', 1)],
                              fundamental=P, tau_grid=np.array(taus),
                              spatial_grid=np.array(xs), use_cache=False,
                              verbose=False)
    return {el: np.asarray(v, dtype=float).reshape(len(taus), len(xs))
            for el, v in r['C_tau_x_by_ell'].items()}


def _k3(m, P, pts):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        r = compute_cumulants(m, k=3, max_ell=0,
                              external_fields=[('phi', 1)] * 3, fundamental=P,
                              spatial_points=pts, use_cache=False,
                              parallel=False, verbose=False)
    return np.asarray(r['C_kpoint'], dtype=float).ravel()


def _table(m, P):
    from engine.core.field_theory import FieldTheory
    ft = FieldTheory(m, taylor_order=4)
    ft.expand()
    sub = {SR.var(k): v for k, v in P.items()}
    sub[SR.var('phistar1')] = 0.0
    return ft, [(t['chain'], t['mode'], float(SR(t['weight']).subs(sub)))
                for t in ft._ns._operator_ir_vertex_terms]


# ── the table: two species, own couplings ─────────────────────────────
def test_lap_and_local_vertex_are_two_species():
    m, P = _mixed(0.3, -0.3)
    _ft, tbl = _table(m, P)
    by_chain = {chain: (mode, w) for chain, mode, w in tbl}
    assert set(by_chain) == {(('Lap',),), ()}, tbl
    # action coefficients -c1 (Lap) and -c2 (local), over sqrt(c1^2 + c2^2)
    assert by_chain[(('Lap',),)][1] == pytest.approx(-1 / math.sqrt(2), abs=1e-14)
    assert by_chain[()] == ('composite', pytest.approx(1 / math.sqrt(2), abs=1e-14))


def test_cancelling_pair_does_not_collapse():
    """c1 = -c2 used to cancel the phit*phi^2 vertex: the loop run raised
    'no live loop diagrams' and the k=3 tree came out exactly 0."""
    m, P = _mixed(0.3, -0.3)
    C = _k2(m, P, [0.0, 1.0])
    assert C[1][0, 0] > 0.1 * C[0][0, 0], C
    assert np.all(np.abs(_k3(m, P, _PTS)) > 1e-3)


# ── k=3 tree vs an independent Fourier quadrature ─────────────────────
_PTS = np.array([[[0.5, 0.3], [-0.4, 0.7]], [[1.2, 0.4], [0.2, 0.9]],
                 [[0.0, 0.6], [0.9, 0.2]]])     # distinct external times


def _time_integral(ta, tb, tc, wa, wb, wc):
    """int_{-inf}^{ta} ds e^{-wa (ta - s)} e^{-wb |s - tb|} e^{-wc |s - tc|}."""
    edges = [-np.inf] + sorted(t for t in (tb, tc) if t < ta) + [ta]
    out = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mid = hi - 0.5 if lo == -np.inf else 0.5 * (lo + hi)
        sb = 1.0 if mid > tb else -1.0
        sc = 1.0 if mid > tc else -1.0
        beta = wa - sb * wb - sc * wc               # integrand exp(alpha + beta s)
        alpha = -wa * ta + sb * wb * tb + sc * wc * tc
        if lo == -np.inf:
            out = out + np.exp(alpha + beta * hi) / beta
            continue
        # evaluated at the larger end, so nothing overflows
        peak = np.exp(np.maximum(alpha + beta * hi, alpha + beta * lo))
        ab = np.abs(beta) * (hi - lo)
        out = out + peak * np.where(
            ab > 1e-12, -np.expm1(-ab) / np.maximum(np.abs(beta), 1e-300), hi - lo)
    return out


def _kappa3_tree(point, c1, c2, kp=0.0, mu=1.0, D=1.0, T=1.0, K=60.0, n=2401):
    """Tree third cumulant <phi(0,0) phi(x1,t1) phi(x2,t2)>: the second-order
    response to the drift c2 phi^2 + c1 Lap(phi^2) + (kp/2) (Dx phi)^2, in
    Fourier space (R(k,t) = e^{-w t}, C(k,t) = T/w e^{-w|t|}, w = mu + D k^2;
    the gradient legs give (i kb)(i kc))."""
    k = np.linspace(-K, K, n)
    dk = k[1] - k[0]
    kb, kc = np.meshgrid(k, k, indexing='ij')
    ka = kb + kc
    w = lambda q: mu + D * q * q
    wa, wb, wc = w(ka), w(kb), w(kc)
    V = 2.0 * (c2 - c1 * ka * ka) - kp * kb * kc
    pts = [(0.0, 0.0)] + [tuple(p) for p in point]
    tot = 0.0
    for a in range(3):
        b, c = [i for i in range(3) if i != a]
        (xa, ta), (xb, tb), (xc, tc) = pts[a], pts[b], pts[c]
        f = (V * (T / wb) * (T / wc) * _time_integral(ta, tb, tc, wa, wb, wc)
             * np.cos(ka * xa - kb * xb - kc * xc))
        tot += (np.sum(f) - 0.5 * (f[0].sum() + f[-1].sum()
                                   + f[:, 0].sum() + f[:, -1].sum())) * dk * dk
    return tot / (2.0 * np.pi) ** 2


@pytest.fixture(scope='module')
def k3_basis():
    """The tree is linear in (c1, c2): the Lap and the local responses."""
    return (np.array([_kappa3_tree(p, 1.0, 0.0) for p in _PTS]),
            np.array([_kappa3_tree(p, 0.0, 1.0) for p in _PTS]))


@pytest.mark.parametrize('c1,c2', [(0.3, -0.3), (0.3, 0.2), (0.0, 0.2)])
def test_k3_tree_vs_fourier_quadrature(c1, c2, k3_basis):
    m, P = _mixed(c1, c2)
    got = _k3(m, P, _PTS)
    want = c1 * k3_basis[0] + c2 * k3_basis[1]
    # the merged coefficient gave (c1+c2)*Lap-response: 0 at c1=-c2, and the
    # wrong sign at (0.3, 0.2)
    assert np.max(np.abs(got - want)) < 1e-7, (got, want)


def test_split_local_coupling_is_one_species(k3_basis):
    """c2a*phi^2 + c2b*phi^2 stay two terms after expand(); both belong to
    the local species (only the last one used to)."""
    P = {'mu': 1.0, 'D': 1.0, 'c1': 0.3, 'c2a': -0.1, 'c2b': -0.2, 'T': 1.0}
    m = _built(P, 'c2a*phi^2 + c2b*phi^2',
               'phit*(Dt(phi) + mu*phi - D*Lap(phi) - c1*Lap(phi^2)'
               ' - c2a*phi^2 - c2b*phi^2) - T*phit^2')
    got = _k3(m, P, _PTS)
    want = 0.3 * k3_basis[0] - 0.3 * k3_basis[1]
    assert np.max(np.abs(got - want)) < 1e-7, (got, want)


def test_gradient_and_local_vertices_at_a_nonzero_saddle():
    """A per-leg gradient vertex (kp/2)(Dx phi)^2 next to c2 phi^2 and
    lam phi^3 at a non-zero saddle: the local phit*dphi^2 coupling is
    c2 + 3 lam phi* (two terms), the mass mu - 2 c2 phi* - 3 lam phi*^2."""
    kp, c2, lam, h = 0.4, 0.15, -0.2, 0.5
    P = {'mu': 1.0, 'D': 1.0, 'kp': kp, 'c2': c2, 'lam': lam, 'h': h, 'T': 1.0}
    m = _built(P, 'c2*phi^2 + lam*phi^3 + h',
               'phit*(Dt(phi) + mu*phi - D*Lap(phi) - (kp/2)*Dx(phi,0)^2'
               ' - c2*phi^2 - lam*phi^3 - h) - T*phit^2')
    roots = np.roots([lam, c2, -1.0, h])                  # h - phi + c2 phi^2 + lam phi^3
    ps = min((r.real for r in roots if abs(r.imag) < 1e-12),
             key=lambda r: abs(r - h))                    # the branch from phi* = h
    mass = 1.0 - 2.0 * c2 * ps - 3.0 * lam * ps ** 2
    got = _k3(m, P, _PTS)
    want = np.array([_kappa3_tree(p, 0.0, c2 + 3.0 * lam * ps, kp=kp, mu=mass)
                     for p in _PTS])
    assert np.max(np.abs(got - want)) < 1e-7, (got, want)


# ── k=2 tree at a non-zero saddle vs the linear-response closed form ──
def _C_ou(x, tau, mu, D, T):
    x, tau = abs(float(x)), abs(float(tau))
    a = math.sqrt(mu / D)
    if tau == 0.0:
        return T / (2.0 * math.sqrt(mu * D)) * math.exp(-a * x)
    s, r = math.sqrt(mu * tau), x / (2.0 * math.sqrt(D * tau))
    return T / (4.0 * math.sqrt(mu * D)) * (
        math.exp(-a * x) * erfc(s - r) + math.exp(a * x) * erfc(s + r))


def test_tree_at_nonzero_saddle_vs_closed_form():
    c1, c2, h = 0.3, -0.3, 0.5
    m, P = _mixed(c1, c2, h=h)
    taus, xs = [0.0, 0.5, 1.0], [0.0, 0.6, 1.5]
    C = _k2(m, P, xs, taus, max_ell=1)
    ps = (1.0 - math.sqrt(1.0 - 4.0 * c2 * h)) / (2.0 * c2)   # stable root
    ref = np.array([[_C_ou(x, t, 1.0 - 2.0 * c2 * ps, 1.0 + 2.0 * c1 * ps, 1.0)
                     for x in xs] for t in taus])
    assert np.max(np.abs(C[0] - ref)) < 1e-12, (C[0], ref)
    assert np.all(np.isfinite(C[1])) and np.max(np.abs(C[1])) > 1e-3


# ── one species switched off = the one-species model ─────────────────
def test_one_coupling_zero_matches_the_one_species_model():
    xs = [0.0, 1.0, 2.0]
    m, P = _mixed(0.0, 0.2)
    loc, Ploc = _mixed(0.0, 0.2, action='phit*(Dt(phi) + mu*phi - D*Lap(phi)'
                                         ' - c2*phi^2{src}) - T*phit^2')
    a, b = _k2(m, P, xs)[1], _k2(loc, Ploc, xs)[1]
    # the merged coefficient put the Laplacian form factor on c2 here
    # (the two take different integrator routes: ~1e-6 relative apart)
    assert np.max(np.abs(a - b)) < 1e-5 * np.max(np.abs(b)), (a, b)
    m, P = _mixed(0.3, 0.0)
    lap, Plap = _mixed(0.3, 0.0, action='phit*(Dt(phi) + mu*phi - D*Lap(phi)'
                                         ' - c1*Lap(phi^2){src}) - T*phit^2')
    a, b = _k2(m, P, xs)[1], _k2(lap, Plap, xs)[1]
    assert np.max(np.abs(a - b)) < 1e-12, (a, b)


# ── models with one species are unchanged ─────────────────────────────
def _hand(term):
    return (SpatialModelBuilder('one species', n_populations=0)
            .physical_field('phi', spatial_dim=1)
            .parameter('mu', default=1.0, domain='positive')
            .parameter('D', default=1.0, domain='positive')
            .parameter('c', default=0.3, domain='real')
            .parameter('T', default=1.0, domain='positive')
            .equation(lhs='(Dt + mu - D*Laplacian)*phi', rhs='0')
            .set_action_text(f'phit*(Dt(phi) + mu*phi - D*Lap(phi) {term}) - T*phit^2')
            .operator_ir().boundary('infinite').initial('stationary').build())


def _tracked(name):
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        name, os.path.join(_MODELS, f'{name}.model.py'))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# lowered action and form-factor table, recorded before the fix (main a01bf39)
_GOLDEN = {
    'burgers_1d': (
        '-T*phit1^2 - 1/2*(2*D*Laplacian*dphi1 - 2*Dt*dphi1 - (2*GradX*dphi1*phistar1'
        ' + dphi1^2)*lam - 2*(dphi1 + phistar1)*mu)*phit1',
        [['1', 2, "(('Dx', 0),)", 'composite']]),
    'kpz_1d': (
        '-T*ht1^2 - 1/2*(2*D*Laplacian*dh1 + dh1^2*lam - 2*Dt*dh1 - 2*(dh1 + hstar1)*mu)*ht1',
        [['1', 2, "(('Dx', 0),)", 'perleg']]),
    'reaction_diffusion_conserved_1d': (
        '-T*phit1^2 - (D*Laplacian*dphi1 - Dt*dphi1 + (2*Laplacian*dphi1*phistar1'
        ' + dphi1^2)*g - (dphi1 + phistar1)*mu)*phit1',
        [['1', 2, "(('Lap',),)", 'composite']]),
    '- c*Lap(phi^2)': (
        '-T*phit1^2 - (D*Laplacian*dphi1 + (2*Laplacian*dphi1*phistar1 + dphi1^2)*c'
        ' - Dt*dphi1 - (dphi1 + phistar1)*mu)*phit1',
        [['1', 2, "(('Lap',),)", 'composite']]),
    '- c*phi^2': (
        '-T*phit1^2 - (c*(dphi1 + phistar1)^2 + D*Laplacian*dphi1 - Dt*dphi1'
        ' - (dphi1 + phistar1)*mu)*phit1', []),
    '- (c/2)*Dx(phi,0)^2': (
        '-T*phit1^2 - 1/2*(2*D*Laplacian*dphi1 + c*dphi1^2 - 2*Dt*dphi1'
        ' - 2*(dphi1 + phistar1)*mu)*phit1',
        [['1', 2, "(('Dx', 0),)", 'perleg']]),
}


@pytest.mark.parametrize('name', sorted(_GOLDEN))
def test_one_species_models_are_unchanged(name):
    from engine.core.field_theory import FieldTheory
    m = _tracked(name).build() if name[0].isalpha() else _hand(name)
    ft = FieldTheory(m, taylor_order=4)
    ft.expand()
    action = str(m['action'](ft._ns))
    vt = [[str(t['weight']), t['n_phys'], str(t['chain']), t['mode']]
          for t in ft._ns._operator_ir_vertex_terms]
    assert (action, vt) == _GOLDEN[name]


# one-loop values recorded before the fix (main a01bf39), x = 0, 0.5, 1, 2
_GOLDEN_LOOP = {
    'reaction_diffusion_conserved_1d': [0.003704714023253397, 0.0009100770857724816,
                                        -7.612447777172182e-05, -0.00043231353080740165],
    'kpz_1d': [0.0009471760800918716, 0.0008827924984700641,
               0.0007333349450833471, 0.00040963609945601775],
    'burgers_1d': [-0.0001294607130037928, -0.00012969151202624207,
                   -9.66928744966511e-05, 5.5670129388424305e-06],
}


@pytest.mark.parametrize('name', sorted(_GOLDEN_LOOP))
def test_one_species_loop_values_are_unchanged(name):
    mod = _tracked(name)
    ext = mod.METADATA['recommended_external_fields']
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        r = compute_cumulants(mod.build(), k=2, max_ell=1, external_fields=ext,
                              fundamental=mod.DEFAULT_FUNDAMENTAL,
                              tau_max=0.0, tau_step=1.0,
                              spatial_grid=np.array([0.0, 0.5, 1.0, 2.0]),
                              use_cache=False, verbose=False)
    got = np.asarray(r['C_tau_x_by_ell'][1], dtype=float).ravel()
    np.testing.assert_allclose(got, _GOLDEN_LOOP[name], rtol=1e-12, atol=1e-17)


# ── one loop vs the lattice simulator (paired runs, common noise) ────
@pytest.mark.slow
@pytest.mark.parametrize('c1,c2', [(0.3, -0.3), (0.3, -0.15)])
def test_one_loop_vs_simulator(c1, c2):
    """delta C(x) = C_interacting - C_free at equal times from paired lattice
    runs (same noise), against the package one-loop term.  mu=2, D=4 keep the
    lattice field far from the blow-up thresholds (D + 2 c1 phi > 0 and the
    cubic escape point -mu/c2).  dt = 0.0025: the ETD1 splitting bias of the
    nonlinear step is below the statistical error there (dt = 0.01 is ~15 %
    low).  Two loops are ~1 % of one loop here."""
    from simulations.spatial_field_1d_sim import simulate
    mu, D, L, N = 2.0, 4.0, 20.0, 200
    dx = L / N
    xs = [0.0, 0.5, 1.0]
    ms = [int(round(x / dx)) for x in xs]

    def conn(snaps):
        F = np.fft.rfft(snaps - snaps.mean(), axis=1)
        return (np.fft.irfft(np.mean(np.abs(F) ** 2, axis=0), n=N) / N)[ms]

    d = []
    for seed in (1, 2, 3, 4):
        kw = dict(L=L, N=N, mu=mu, D=D, T=1.0, n_steps=800000, burn_in=20000,
                  record_every=10, seed=seed, dt=0.0025)
        free, _, _ = simulate(**kw)
        inter, _, _ = simulate(g=-c2, g_lap=c1, **kw)     # sim drift -g phi^2
        d.append(conn(inter) - conn(free))
    d = np.asarray(d)
    sim, err = d.mean(axis=0), d.std(axis=0, ddof=1) / math.sqrt(len(d))
    m, P = _mixed(c1, c2, mu=mu, D=D)
    loop = _k2(m, P, xs)[1][0]
    assert np.all(np.abs(sim - loop) < 3.0 * err + 0.03 * np.abs(loop)), \
        (sim, err, loop)
