"""
tests/test_dyson_dressing.py
============================
Dyson dressing D-2/D-3 core + the D-5 tree-level validation LADDER
(``engine.integration.spatial.dyson_dressing``).

  * ``𝓗_0(t) = e^{−Mt}`` (the bare matrix decay; B27 at n=0);
  * ``dressed_GR`` order-convergence: ``G^{(N)}(t,q) → expm(−(M+𝒟q²)t)``
    geometrically in ``N`` for unequal diffusion (the B24 series);
  * ``dressed_tree_C`` ladder N=0,1,2,3 vs the EXACT unequal-D oracle
    ``C(q,τ) = expm(−A(q)τ)·Σ(q)``, ``A Σ+Σ Aᵀ = N`` (the same classical result
    the 2-species Langevin sim was pinned against in test_coupled_rd_sim);
  * 𝒟̂=0 reduction: order-0 dressed tree == coupled_two_point (3a) exactly.

Run:  sage -python -m pytest tests/test_dyson_dressing.py -q
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from scipy.linalg import expm, solve_continuous_lyapunov         # noqa: E402

from engine.integration.spatial.dyson_dressing import (           # noqa: E402
    hcal_n, dressed_GR, dressed_tree_C,
)
from engine.integration.spatial.spectral_propagator import (      # noqa: E402
    build_reference, coupled_two_point, split_reference_diffusion,
)

_M = np.array([[1.5, 0.5], [0.3, 1.2]])
_NZ = np.array([[1.0, 0.2], [0.2, 0.8]])
_DVEC = np.array([0.9, 0.5])                # unequal ⇒ 𝒟̂ = diag(+0.2, −0.2)


def _exact_C(q, tau, Dvec=_DVEC):
    A = _M + np.diag(Dvec) * q * q
    Sig = solve_continuous_lyapunov(A, _NZ)
    C = expm(-A * abs(tau)) @ Sig
    return C if tau >= 0 else C.T


def test_hcal0_is_matrix_exponential():
    ts = np.array([0.0, 0.3, 1.1])
    H0 = hcal_n(ts, _M, np.zeros((2, 2)), 0)
    for i, t in enumerate(ts):
        assert np.allclose(H0[i], expm(-_M * t), atol=1e-12)


@pytest.mark.parametrize('qsq,t', [(1.0, 0.5), (2.5, 0.8)])
def test_dressed_GR_converges_to_full_matrix_heat_kernel(qsq, t):
    """‖G^{(N)} − expm(−(M+𝒟q²)t)‖ decreases geometrically with N (B24)."""
    D0, Dhat = split_reference_diffusion(np.diag(_DVEC))
    target = expm(-(_M + np.diag(_DVEC) * qsq) * t)
    errs = []
    for order in range(5):
        G = dressed_GR(np.array([t]), qsq, _M, Dhat, D0, order)[0]
        errs.append(np.max(np.abs(G - target)))
    assert errs[4] < 1e-4, f'order-4 error {errs[4]:.2e}'
    for n in range(4):
        assert errs[n + 1] < 0.75 * errs[n], \
            f'no geometric decay at N={n}: {errs}'


@pytest.mark.parametrize('q,tau', [(0.0, 0.4), (1.0, 0.0), (1.5, 0.6)])
def test_dressed_tree_ladder_vs_exact(q, tau):
    """D-5 tree ladder: C^{(N)}(q,τ) → exact expm/Lyapunov C(q,τ), errors
    shrinking with N; N=3 within 2e-3 relative."""
    D0, Dhat = split_reference_diffusion(np.diag(_DVEC))
    exact = _exact_C(q, tau)
    scale = np.max(np.abs(exact))
    errs = []
    for order in range(4):
        C = dressed_tree_C(q, tau, _M, Dhat, D0, _NZ, order, n_s=64)
        errs.append(np.max(np.abs(C - exact)) / scale)
    # at q=0 the insertion 𝒟̂q² vanishes — every order is exact
    if q == 0.0:
        assert errs[0] < 1e-8
    else:
        assert errs[3] < 2e-3, f'ladder: {[f"{e:.2e}" for e in errs]}'
        assert errs[3] < errs[0], f'no improvement: {errs}'
        assert errs[2] < errs[0]


def test_order0_scalar_diffusion_reduces_to_coupled_two_point():
    """𝒟̂=0 at order 0: the dressed tree == the (sim-validated) Lyapunov
    coupled_two_point — pins the s-integral normalization."""
    Dsc = 0.7 * np.eye(2)
    D0, Dhat = split_reference_diffusion(Dsc)
    assert np.allclose(Dhat, 0.0)
    ref = build_reference(_M, Dsc)
    for q, tau in ((0.0, 0.0), (0.8, 0.5), (1.4, 1.0)):
        C_d = dressed_tree_C(q, tau, _M, Dhat, D0, _NZ, 0, n_s=64)
        C_l = coupled_two_point(ref, _NZ, q * q, tau)
        assert np.allclose(C_d, C_l, atol=2e-6 * max(1.0, np.max(np.abs(C_l)))), \
            f'(q={q},τ={tau}):\n{C_d}\nvs\n{C_l}'


def test_tau_negative_transpose():
    D0, Dhat = split_reference_diffusion(np.diag(_DVEC))
    Cp = dressed_tree_C(1.0, 0.5, _M, Dhat, D0, _NZ, 2)
    Cm = dressed_tree_C(1.0, -0.5, _M, Dhat, D0, _NZ, 2)
    assert np.allclose(Cm, Cp.T, atol=1e-12)


# ── D-4 policy wiring + D-5 end-to-end (unequal diffusion) ───────────────────

_MU1, _MU2, _D1, _D2, _G, _T1, _T2 = 1.5, 1.2, 0.9, 0.5, 0.4, 0.5, 0.4


def _unequal_model(order=None, periodic_L=None):
    from api.model import SpatialModelBuilder
    b = (SpatialModelBuilder('coupled2_unequalD')
         .physical_field('a', spatial_dim=1)
         .physical_field('b', spatial_dim=1)
         .parameter('mua', default=_MU1, domain='positive')
         .parameter('mub', default=_MU2, domain='positive')
         .parameter('Da', default=_D1, domain='positive')
         .parameter('Db', default=_D2, domain='positive')
         .parameter('g', default=_G)
         .parameter('Ta', default=_T1, domain='positive')
         .parameter('Tb', default=_T2, domain='positive'))
    action = ('at*((Dt+mua-Da*Laplacian)*a + g*b) '
              '+ bt*((Dt+mub-Db*Laplacian)*b - g*a) '
              '- Ta*at^2 - Tb*bt^2')
    b = (b.set_action_text(action)
         .equation(lhs='(Dt+mua-Da*Laplacian)*a + g*b', rhs='0')
         .equation(lhs='(Dt+mub-Db*Laplacian)*b - g*a', rhs='0'))
    if periodic_L is not None:
        b = b.boundary('periodic', length=float(periodic_L))
    else:
        b = b.boundary('infinite')
    if order is not None:
        b = b.dyson_order(order)
    return b.initial('stationary').build()


_UF = {'mua': _MU1, 'mub': _MU2, 'Da': _D1, 'Db': _D2, 'g': _G,
       'Ta': _T1, 'Tb': _T2}


def _exact_box_C(xs, taus, L, i, j, n_modes=600):
    """Inline exact unequal-D oracle on the periodic box: per-mode Lyapunov +
    expm (the same classical result the Langevin sim is pinned against)."""
    Mm = np.array([[_MU1, _G], [-_G, _MU2]])
    Dv = np.array([_D1, _D2])
    Nz = np.array([[2 * _T1, 0.0], [0.0, 2 * _T2]])
    qs = 2.0 * np.pi * np.arange(-n_modes, n_modes + 1) / L
    out = np.zeros((len(taus), len(xs)), dtype=complex)
    for it, tau in enumerate(taus):
        Cq = np.empty(qs.size, dtype=complex)
        for iq, q in enumerate(qs):
            A = Mm + np.diag(Dv) * q * q
            Sig = solve_continuous_lyapunov(A, Nz)
            Cm = expm(-A * abs(tau)) @ Sig
            Cm = Cm if tau >= 0 else Cm.T
            Cq[iq] = Cm[i, j]
        for ix, x in enumerate(xs):
            out[it, ix] = np.sum(np.exp(1j * qs * x) * Cq) / L
    return out.real


def test_unequal_D_e2e_compute_cumulants_periodic():
    """D-5 e2e: unequal-D coupled model + .dyson_order(3) flows through
    compute_cumulants (periodic box) and matches the exact per-mode
    expm/Lyapunov oracle to the truncation accuracy."""
    from api import compute_cumulants
    L = 20.0
    model = _unequal_model(order=3, periodic_L=L)
    xs = np.array([0.0, 1.0])
    th = compute_cumulants(model=model, k=2, max_ell=0, fundamental=_UF,
                           external_fields=[('a', 1), ('a', 1)],
                           tau_max=0.5, tau_step=0.5, spatial_grid=xs,
                           parallel=False, verbose=False, use_cache=False)
    info = th['spatial_info']
    assert info.get('coupled') and info.get('dyson_order') == 3
    assert np.max(np.abs(info['Dhat'])) > 0.1            # genuinely unequal D
    C = np.asarray(th['C_tau_x']).real
    taus = np.asarray(th['tau_grid'])
    oracle = _exact_box_C(xs, taus, L, 0, 0)
    scale = np.max(np.abs(oracle))
    assert np.max(np.abs(C - oracle)) < 6e-3 * scale, \
        f'dressed N=3 vs exact:\n{C}\nvs\n{oracle}'


def test_unequal_D_default_policy_raises_actionable():
    """No .dyson_order ⇒ the default {'mode':'off'} policy still raises the
    clean unequal-D error (mentioning dyson_order)."""
    from api import compute_cumulants
    model = _unequal_model(order=None)
    with pytest.raises(NotImplementedError, match='scalar-diffusion'):
        compute_cumulants(model=model, k=2, max_ell=0, fundamental=_UF,
                          external_fields=[('a', 1), ('a', 1)],
                          tau_max=0.5, tau_step=0.5,
                          spatial_grid=np.array([0.0, 1.0]),
                          parallel=False, verbose=False, use_cache=False)


@pytest.mark.slow   # coupled-Dyson order-3 loop path: minutes-long; opt in with -m slow
def test_unequal_D_loops_order3_runs():
    """Loop insertions have NO order cap (exact at every order via the
    ln-derivative partition expansion + generalized-partial-fraction
    H_n labels): dyson_order(3) with max_ell=1 must RUN and return
    finite real values.  (Periodic loop corrections are still gated —
    use the infinite-boundary ladder model.)"""
    from api import compute_cumulants
    model = _ladder_coupled_model(3)
    th = compute_cumulants(model=model, k=2, max_ell=1, fundamental=_LUF,
                           external_fields=[('a', 1), ('a', 1)],
                           tau_max=0.6, tau_step=0.6,
                           spatial_grid=np.array([0.0, 0.8]),
                           parallel=False, verbose=False, use_cache=False)
    assert th['spatial_info'].get('coupled_loop')
    C = np.asarray(th['C_tau_x'])
    assert np.all(np.isfinite(C))


# ── D-3 LOOP dressing: the decoupled-unequal-D ladder (the sharp referee) ───

def _ladder_coupled_model(order):
    """(Near-)decoupled 2-field theory, UNEQUAL D, with BOTH a² (bubble,
    momentum-mixing) and a³ (tadpole) vertices on field a.  eps=1e-8 keeps the
    coupled routing (symbolically nonzero K_ft off-diagonals) while the exact
    physics is two independent fields — field a's answer is the trusted
    single-field generic loop path at D=Da."""
    from api.model import SpatialModelBuilder
    b = (SpatialModelBuilder('ladder_unequalD')
         .physical_field('a', spatial_dim=1)
         .physical_field('b', spatial_dim=1)
         .parameter('mua', default=_MU1, domain='positive')
         .parameter('mub', default=_MU2, domain='positive')
         .parameter('Da', default=_D1, domain='positive')
         .parameter('Db', default=_D2, domain='positive')
         .parameter('eps', default=1e-8)
         .parameter('gq', default=0.3)
         .parameter('ga', default=0.3)
         .parameter('Ta', default=_T1, domain='positive')
         .parameter('Tb', default=_T2, domain='positive'))
    action = ('at*((Dt+mua-Da*Laplacian)*a + eps*b + gq*a^2 + ga*a^3) '
              '+ bt*((Dt+mub-Db*Laplacian)*b - eps*a) '
              '- Ta*at^2 - Tb*bt^2')
    return (b.set_action_text(action)
            .equation(lhs='(Dt+mua-Da*Laplacian)*a + eps*b + gq*a^2 + ga*a^3',
                      rhs='0')
            .equation(lhs='(Dt+mub-Db*Laplacian)*b - eps*a', rhs='0')
            .boundary('infinite').initial('stationary')
            .dyson_order(order).build())


_LUF = {'mua': _MU1, 'mub': _MU2, 'Da': _D1, 'Db': _D2, 'eps': 1e-8,
        'gq': 0.3, 'ga': 0.3, 'Ta': _T1, 'Tb': _T2}


@pytest.mark.slow   # coupled-Dyson loop-dressing ladder: minutes-long; opt in with -m slow
def test_unequal_D_loop_dressing_ladder():
    """THE loop-dressing referee: the decoupled-unequal-D loop correction dC
    must converge to the single-field answer at D=Da — order 0 (no dressing,
    the wrong D₀=(Da+Db)/2 heat kernel, error O(ρ)) → order 1 (the (−|k|²)
    insertion + 𝓗₁ poles, error O(ρ²)); ρ = ‖𝒟̂‖/D₀ ≈ 0.29.  Covers tadpole
    AND momentum-mixing bubble topologies (a² + a³ vertices)."""
    from api import compute_cumulants
    from api.model import SpatialModelBuilder

    sg = np.array([0.0, 0.8])
    kw = dict(k=2, max_ell=1, tau_max=0.6, tau_step=0.6,
              parallel=False, verbose=False, use_cache=False)

    def loop_piece(th):
        info = th['spatial_info']
        return (np.asarray(info['C_by_order'][1])
                - np.asarray(info['C_by_order'][0]))

    dC = {}
    for order in (0, 1, 2):
        th = compute_cumulants(model=_ladder_coupled_model(order),
                               fundamental=_LUF,
                               external_fields=[('a', 1), ('a', 1)],
                               spatial_grid=sg, **kw)
        assert th['spatial_info'].get('coupled_loop')
        dC[order] = loop_piece(th)

    scalar = (SpatialModelBuilder('ladder_single')
              .physical_field('phi', spatial_dim=1)
              .parameter('mu', default=_MU1, domain='positive')
              .parameter('D', default=_D1, domain='positive')
              .parameter('gq', default=0.3)
              .parameter('ga', default=0.3)
              .parameter('T', default=_T1, domain='positive')
              .set_action_text('phit*((Dt+mu-D*Laplacian)*phi + gq*phi^2 '
                               '+ ga*phi^3) - T*phit^2')
              .equation(lhs='(Dt+mu-D*Laplacian)*phi + gq*phi^2 + ga*phi^3',
                        rhs='0')
              .boundary('infinite').initial('stationary').build())
    th_s = compute_cumulants(model=scalar,
                             fundamental={'mu': _MU1, 'D': _D1, 'gq': 0.3,
                                          'ga': 0.3, 'T': _T1},
                             external_fields=[('phi', 1), ('phi', 1)],
                             spatial_grid=sg, **kw)
    dC_ref = loop_piece(th_s)

    scale = np.max(np.abs(dC_ref))
    err0 = np.max(np.abs(dC[0] - dC_ref)) / scale
    err1 = np.max(np.abs(dC[1] - dC_ref)) / scale
    err2 = np.max(np.abs(dC[2] - dC_ref)) / scale
    # the LADDER: each insertion order must remove the leading 𝒟̂-power of
    # the error — O(ρ) → O(ρ²) → O(ρ³), ρ ≈ 0.29
    assert err1 < 0.55 * err0, f'no ladder improvement: err0={err0:.4f} err1={err1:.4f}'
    assert err2 < 0.60 * err1, f'order-2 no improvement: err1={err1:.4f} err2={err2:.4f}'
    assert err1 < 0.15, f'order-1 dressing too far off: err1={err1:.4f} (err0={err0:.4f})'
    assert err0 > 0.10, f'order-0 suspiciously accurate (test not discriminating): {err0:.4f}'
