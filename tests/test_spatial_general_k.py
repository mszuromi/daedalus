"""Trust-the-process battery for the SPATIAL general-k machinery.

All k cannot be tested e2e, so this battery pins the k-generic code path
by construction instead:

1. **k=2 reduction** — the general mapping-sum driver
   (``diagram_correlator_pts`` + ``field_respecting_mappings`` +
   ``external_wick_compensation``) reproduces the production k=2 path
   (``diagram_correlator_x`` with its retarded ±τ completion) to machine
   precision on every enumerated diagram.  The k=2 path is sim-validated
   (Allen-Cahn/KPZ/Model B notebooks), so this transfers that trust.
2. **Per-sample Symanzik identity** — for n_ext=2 the matrix
   ``𝓑 = D·Q_eff`` from ``_symanzik_kernel_batch`` satisfies
   ``momfac(q⃗) = pref·exp(−q⃗ᵀ𝓑q⃗)`` exactly against the independent
   (already n_ext-general, k=2-trusted) ``_momentum_factor_batch`` on
   random Schwinger samples — tree and loop descriptors.
3. **Multivariate IFT formula** — ``_heat_kernel_x_general`` equals the
   direct 2-d quadrature of ``∫d²q e^{iq·X} e^{−q⃗ᵀ𝓑q⃗}/(2π)²`` for
   asymmetric test matrices.
4. **Full-value route equivalence at k=3** — analytic-IFT value equals
   the numerical Fourier transform of the q-path over a (q0, q1) grid,
   for the tree AND a 1-loop diagram (same chamber quadrature on both
   routes, so the only difference is the FT discretization).
5. **Permutation invariance** — the mapping-sum cumulant is symmetric
   under exchanging identical-field external events.

The k-independent pieces these tests pin (momentum-conservation routing
widths, the (k−1)-dim Gaussian reduction, the orbit–stabilizer Wick sum)
are exactly what varies with k; everything downstream (causal chambers,
Schwinger/Symanzik, heat kernels) is k-blind and already validated at
k=2.  The temporal pipeline's identical Wick architecture is anchored to
exact Boltzmann series at k ≤ 5 (tests/test_all_k_boltzmann.py).
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))

from sage.all import SR  # noqa: E402


def _build_records(k, max_ell=1):
    from pipeline.theory import TheoryBuilder
    from pipeline.compute import FieldTheory
    from pipeline._propagator import build_propagator
    from msrjd.integration.spatial.diagram_descriptor import diagram_to_cstack
    from msrjd.integration.spatial.pipeline_bridge import (
        build_pipeline_records, _legs_to_phys_idx)
    from msrjd.diagrams.type_assignment import build_field_index_map

    b = (TheoryBuilder('rd-quad-generalk', n_populations=0)
         .physical_field('p', spatial_dim=1)
         .parameter('mu', default=1.0, domain='positive')
         .parameter('DD', default=1.0, domain='positive')
         .parameter('g', default=0.1, domain='real')
         .parameter('T', default=1.0, domain='positive')
         .equation(lhs='(Dt+mu-DD*Laplacian)*p', rhs='0')
         .set_action_text('pt*(Dt(p)+mu*p-DD*Lap(p)+g*p^2) - T*pt^2')
         .operator_ir().boundary('infinite').initial('stationary').build())
    ft = FieldTheory(b, taylor_order=k + 2)
    ft.expand()
    prop = build_propagator(ft, b, use_cache=False, verbose=False)
    rvn = list(ft._ns._ring_var_names)
    _, pidx = build_field_index_map(rvn, ft._n_tilde)
    ext = _legs_to_phys_idx([('p', 1)] * k, pidx)
    base = {SR.var('mu'): 1., SR.var('DD'): 1., SR.var('g'): 0.1,
            SR.var('T'): 1., SR.var('pstar1'): 0.}
    be = build_pipeline_records(ft, b, prop, ext, max_ell=max_ell, k=k,
                                verbose=False)
    recs = []
    for ell in sorted(be):
        for td, p in be[ell]:
            pv = float(SR(p).subs(base))
            if abs(pv) < 1e-14:
                continue
            recs.append((td, diagram_to_cstack(td), pv, ell))
    return recs


@pytest.fixture(scope='module')
def recs2():
    return _build_records(2)


@pytest.fixture(scope='module')
def recs3():
    return _build_records(3)


def test_k2_reduction_general_driver(recs2):
    """General mapping-sum driver == production k=2 path, every diagram."""
    from msrjd.diagrams.symmetry import external_wick_compensation
    from msrjd.integration.spatial.full_integrator import (
        diagram_correlator_x, diagram_correlator_pts,
        field_respecting_mappings)
    xs = np.array([0.0, 0.7, 1.5])
    maps = field_respecting_mappings(['p', 'p'], ['p', 'p'])
    assert len(maps) == 2
    for tau in (0.0, 0.6):
        for td, d, pv, ell in recs2:
            old = diagram_correlator_x(d, pv, xs, tau, 1.0, 1.0,
                                       spatial_dim=1)
            comp = external_wick_compensation(td)
            xp = np.stack([np.zeros_like(xs), xs], axis=1)
            new = diagram_correlator_pts(d, pv, xp, [0.0, tau], 1.0, 1.0,
                                         spatial_dim=1, mappings=maps,
                                         comp=comp)
            assert np.max(np.abs(new - old)) < 1e-12, (ell, tau, comp)


def test_symanzik_matrix_per_sample_identity(recs3):
    """momfac(q⃗) == pref·exp(−q⃗ᵀ𝓑q⃗) exactly, n_ext=2, random w samples."""
    from msrjd.integration.spatial.full_integrator import (
        _symanzik_kernel_batch, _momentum_factor_batch)
    rng = np.random.default_rng(0)
    for td, d, pv, ell in [recs3[0], recs3[1], recs3[-1]]:
        E = len(d.edges)
        a = np.array([e.a for e in d.edges], float).reshape(E, -1)
        bm = np.array([e.b for e in d.edges], float).reshape(E, -1)
        assert bm.shape[1] == 2                      # k−1 = 2 routed momenta
        w = rng.uniform(0.05, 2.0, size=(40, E))
        pref, B, ok = _symanzik_kernel_batch(a, bm, w, 1.0, 1)
        for qv in ([0.3, -0.8], [1.2, 0.5]):
            mf = _momentum_factor_batch(a, bm, w, qv, 1.0, 1)
            q = np.asarray(qv)
            quad = B * q[0] ** 2 if B.ndim == 1 \
                else np.einsum('j,pjk,k->p', q, B, q)
            rhs = pref * np.exp(-quad)
            rel = np.max(np.abs(mf[ok] - rhs[ok])
                         / np.maximum(np.abs(rhs[ok]), 1e-300))
            assert rel < 1e-12, (ell, qv, rel)


def test_multivariate_ift_vs_quadrature():
    """_heat_kernel_x_general == direct 2-d quadrature, asymmetric B."""
    from scipy.integrate import dblquad
    from msrjd.integration.spatial.full_integrator import (
        _heat_kernel_x_general)
    Bm = np.array([[[0.9, 0.3], [0.3, 1.4]],
                   [[2.0, -0.5], [-0.5, 0.7]]])
    X = np.array([[0.5, -0.4], [0.0, 1.1]])
    hk = _heat_kernel_x_general(Bm, X, 1)
    for p in range(2):
        for ix in range(2):
            f = (lambda q1, q0, p=p, ix=ix:
                 np.cos(q0 * X[ix, 0] + q1 * X[ix, 1])
                 * np.exp(-(Bm[p, 0, 0] * q0 * q0
                            + 2 * Bm[p, 0, 1] * q0 * q1
                            + Bm[p, 1, 1] * q1 * q1)))
            val, _ = dblquad(f, -np.inf, np.inf, -np.inf, np.inf)
            val /= (2 * np.pi) ** 2
            assert abs(hk[p, ix] - val) / abs(val) < 1e-8


def test_k3_route_equivalence_tree(recs3):
    """Analytic xs-IFT == numerical FT of the q-path (k=3 tree)."""
    from msrjd.integration.spatial.full_integrator import diagram_kinematic
    td, d, pv, ell = recs3[0]
    assert ell == 0
    legs = list(d.external_legs)
    et = {legs[j]: t for j, t in enumerate((0.0, 0.3, -0.2))}
    X = np.array([[0.5, -0.4]])
    kin_x = diagram_kinematic(d, [0.0, 0.0], et, 1.0, 1.0, spatial_dim=1,
                              xs=X)
    qm, nq = 12.0, 161
    qg = np.linspace(-qm, qm, nq)
    dq = qg[1] - qg[0]
    acc = 0.0
    for q0 in qg:
        row = np.array([diagram_kinematic(d, [q0, q1], et, 1.0, 1.0,
                                          spatial_dim=1) for q1 in qg])
        acc += np.sum(np.cos(q0 * X[0, 0] + qg * X[0, 1]) * row) * dq * dq
    acc /= (2 * np.pi) ** 2
    assert abs(kin_x[0] - acc) / abs(acc) < 1e-10


@pytest.mark.slow
def test_k3_route_equivalence_1loop(recs3):
    """Analytic xs-IFT == numerical FT of the q-path (k=3 1-loop diagram).
    Coarse chamber quadrature on BOTH routes (shared w-grid) so the only
    difference is the q-FT discretization."""
    from msrjd.integration.spatial.full_integrator import diagram_kinematic
    td, d, pv, ell = recs3[1]
    assert ell == 1
    legs = list(d.external_legs)
    et = {legs[j]: t for j, t in enumerate((0.0, 0.3, -0.2))}
    X = np.array([[0.5, -0.4]])
    kw = dict(spatial_dim=1, n_t=6, n_s=8)
    kin_x = diagram_kinematic(d, [0.0, 0.0], et, 1.0, 1.0, xs=X, **kw)
    qm, nq = 10.0, 81
    qg = np.linspace(-qm, qm, nq)
    dq = qg[1] - qg[0]
    acc = 0.0
    for q0 in qg:
        row = np.array([diagram_kinematic(d, [q0, q1], et, 1.0, 1.0, **kw)
                        for q1 in qg])
        acc += np.sum(np.cos(q0 * X[0, 0] + qg * X[0, 1]) * row) * dq * dq
    acc /= (2 * np.pi) ** 2
    assert abs(kin_x[0] - acc) / max(abs(acc), 1e-300) < 5e-3


def test_k3_permutation_invariance(recs3):
    """The full mapping-sum cumulant is symmetric under permuting
    identical-field external events (any single diagram need not be)."""
    from msrjd.diagrams.symmetry import external_wick_compensation
    from msrjd.integration.spatial.full_integrator import (
        diagram_correlator_pts, field_respecting_mappings)
    maps = field_respecting_mappings(['p'] * 3, ['p'] * 3)
    assert len(maps) == 6
    events = [(0.0, 0.0), (0.8, 0.4), (-0.5, -0.3)]   # (x, t) per slot
    import itertools
    vals = []
    for perm in itertools.permutations(range(3)):
        x_pts = np.array([[events[p][0] for p in perm]])
        t_pts = [events[p][1] for p in perm]
        tot = 0.0
        for td, d, pv, ell in recs3:
            comp = external_wick_compensation(td)
            # Invariance is EXACT at any fixed quadrature (permuting the
            # input events permutes the mapping sum onto itself), so the
            # cheap chamber grid suffices — the default 22^3·24^3 grid
            # for n_V=3, n_C=3 loop diagrams is prohibitively large.
            kw = {} if ell == 0 else {'n_t': 6, 'n_s': 8}
            tot += diagram_correlator_pts(d, pv, x_pts, t_pts, 1.0, 1.0,
                                          spatial_dim=1, mappings=maps,
                                          comp=comp, **kw)[0]
        vals.append(tot)
    vals = np.array(vals)
    assert np.max(np.abs(vals - vals[0])) < 1e-12 * max(1.0, abs(vals[0]))


if __name__ == '__main__':
    class _R:
        pass
    r2, r3 = _build_records(2), _build_records(3)
    test_k2_reduction_general_driver(r2); print('k2 reduction OK')
    test_symanzik_matrix_per_sample_identity(r3); print('per-sample identity OK')
    test_multivariate_ift_vs_quadrature(); print('IFT vs quadrature OK')
    test_k3_route_equivalence_tree(r3); print('tree route equivalence OK')
    test_k3_permutation_invariance(r3); print('permutation invariance OK')
    test_k3_route_equivalence_1loop(r3); print('1-loop route equivalence OK')


@pytest.mark.slow
def test_k3_e2e_vs_simulator():
    """End-to-end anchor: the k=3 equal-time third cumulant
    kappa_3(x) = <dphi(0) dphi(x)^2> of the 1-d RD theory
    dphi/dt = -mu phi + DD d2phi - g phi^2 + sqrt(2T) eta
    (tree + 1-loop, mapping-sum driver over all enumerated diagrams)
    against the spectral ETD1 lattice simulator's translational-average
    estimator (models/spatial_field_1d_sim.third_cumulant_x)."""
    from msrjd.diagrams.symmetry import external_wick_compensation
    from msrjd.integration.spatial.full_integrator import (
        diagram_correlator_pts, field_respecting_mappings)
    from models.spatial_field_1d_sim import simulate, third_cumulant_x

    g = 0.25
    # theory records at this coupling (rebuild: module fixture used g=0.1)
    import tests.test_spatial_general_k as _self
    from pipeline.theory import TheoryBuilder
    from pipeline.compute import FieldTheory
    from pipeline._propagator import build_propagator
    from msrjd.integration.spatial.diagram_descriptor import diagram_to_cstack
    from msrjd.integration.spatial.pipeline_bridge import (
        build_pipeline_records, _legs_to_phys_idx)
    from msrjd.diagrams.type_assignment import build_field_index_map
    b = (TheoryBuilder('rd-quad-k3-e2e', n_populations=0)
         .physical_field('p', spatial_dim=1)
         .parameter('mu', default=1.0, domain='positive')
         .parameter('DD', default=1.0, domain='positive')
         .parameter('g', default=g, domain='real')
         .parameter('T', default=1.0, domain='positive')
         .equation(lhs='(Dt+mu-DD*Laplacian)*p', rhs='0')
         .set_action_text('pt*(Dt(p)+mu*p-DD*Lap(p)+g*p^2) - T*pt^2')
         .operator_ir().boundary('infinite').initial('stationary').build())
    ft = FieldTheory(b, taylor_order=5)
    ft.expand()
    prop = build_propagator(ft, b, use_cache=False, verbose=False)
    rvn = list(ft._ns._ring_var_names)
    _, pidx = build_field_index_map(rvn, ft._n_tilde)
    ext = _legs_to_phys_idx([('p', 1)] * 3, pidx)
    base = {SR.var('mu'): 1., SR.var('DD'): 1., SR.var('g'): g,
            SR.var('T'): 1., SR.var('pstar1'): 0.}
    be = build_pipeline_records(ft, b, prop, ext, max_ell=1, k=3,
                                verbose=False)
    recs = []
    for ell in sorted(be):
        for td, p in be[ell]:
            pv = float(SR(p).subs(base))
            if abs(pv) < 1e-14:
                continue
            recs.append((td, diagram_to_cstack(td), pv, ell))

    maps = field_respecting_mappings(['p'] * 3, ['p'] * 3)
    x_eval = np.array([0.0, 0.5, 1.0])
    theory = np.zeros((2, x_eval.size))            # rows: tree, +1loop
    for td, d, pv, ell in recs:
        comp = external_wick_compensation(td)
        x_pts = np.stack([np.zeros_like(x_eval), x_eval, x_eval], axis=1)
        # k=3 1-loop descriptors have n_V=3, n_C=3: the default chamber
        # grid (22^3 x 24^3 ~ 1.5e8 per chamber) is prohibitive.  The
        # route-equivalence test shows n_t=6/n_s=8 already agrees with
        # the q-path; n_t=10/n_s=12 is comfortably converged for a
        # correction that is itself O(g^3).
        kw = {} if ell == 0 else {'n_t': 10, 'n_s': 12}
        v = diagram_correlator_pts(d, pv, x_pts, [0.0, 0.0, 0.0], 1.0, 1.0,
                                   spatial_dim=1, mappings=maps, comp=comp,
                                   **kw)
        if ell == 0:
            theory[0] += v
        theory[1] += v if ell == 1 else (v if ell == 0 else 0)
    # theory[1] currently tree+1loop summed by the loop above for ell<=1
    theory_tot = theory[1]

    snaps, x_grid, meta = simulate(L=20.0, N=200, mu=1.0, D=1.0, T=1.0,
                                   g=g, n_steps=1200000, burn_in=80000,
                                   record_every=20, seed=7)
    k3, se = third_cumulant_x(snaps)
    dx = meta['dx']
    for i, xv in enumerate(x_eval):
        m = int(round(xv / dx))
        sim_v = k3[m]
        sim_e = se[m]
        diff = abs(theory_tot[i] - sim_v)
        tol = max(5.0 * sim_e, 0.05 * abs(sim_v))
        print('  x=%.1f: theory(tree)=%.5f tree+1loop=%.5f  sim=%.5f+-%.5f'
              % (xv, theory[0][i], theory_tot[i], sim_v, sim_e))
        assert diff < tol, (xv, theory_tot[i], sim_v, sim_e)
