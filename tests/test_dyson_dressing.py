"""
tests/test_dyson_dressing.py
============================
Dyson dressing D-2/D-3 core + the D-5 tree-level validation LADDER
(``msrjd.integration.spatial.dyson_dressing``).

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

from msrjd.integration.spatial.dyson_dressing import (           # noqa: E402
    hcal_n, dressed_GR, dressed_tree_C,
)
from msrjd.integration.spatial.spectral_propagator import (      # noqa: E402
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
    from pipeline.theory import SpatialTheoryBuilder
    b = (SpatialTheoryBuilder('coupled2_unequalD')
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
    """D-5 e2e: unequal-D coupled theory + .dyson_order(3) flows through
    compute_cumulants (periodic box) and matches the exact per-mode
    expm/Lyapunov oracle to the truncation accuracy."""
    from pipeline import compute_cumulants
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
    from pipeline import compute_cumulants
    model = _unequal_model(order=None)
    with pytest.raises(NotImplementedError, match='scalar-diffusion'):
        compute_cumulants(model=model, k=2, max_ell=0, fundamental=_UF,
                          external_fields=[('a', 1), ('a', 1)],
                          tau_max=0.5, tau_step=0.5,
                          spatial_grid=np.array([0.0, 1.0]),
                          parallel=False, verbose=False, use_cache=False)


def test_unequal_D_loops_still_gated():
    """Loops + unequal D = the per-edge loop dressing (future work): max_ell=1
    on a dressed unequal-D theory raises cleanly."""
    from pipeline import compute_cumulants
    model = _unequal_model(order=2, periodic_L=20.0)
    with pytest.raises(NotImplementedError):
        compute_cumulants(model=model, k=2, max_ell=1, fundamental=_UF,
                          external_fields=[('a', 1), ('a', 1)],
                          tau_max=0.5, tau_step=0.5,
                          spatial_grid=np.array([0.0, 1.0]),
                          parallel=False, verbose=False, use_cache=False)
