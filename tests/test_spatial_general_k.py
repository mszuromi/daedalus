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
    from api.theory import TheoryBuilder
    from api.compute import FieldTheory
    from api._propagator import build_propagator
    from engine.integration.spatial.diagram_descriptor import diagram_to_cstack
    from engine.integration.spatial.pipeline_bridge import (
        build_pipeline_records, _legs_to_phys_idx)
    from engine.diagrams.type_assignment import build_field_index_map

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
    from engine.diagrams.symmetry import external_wick_compensation
    from engine.integration.spatial.full_integrator import (
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
    from engine.integration.spatial.full_integrator import (
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
    from engine.integration.spatial.full_integrator import (
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
    from engine.integration.spatial.full_integrator import diagram_kinematic
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
    from engine.integration.spatial.full_integrator import diagram_kinematic
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
    from engine.diagrams.symmetry import external_wick_compensation
    from engine.integration.spatial.full_integrator import (
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
    estimator (simulations/spatial_field_1d_sim.third_cumulant_x)."""
    from engine.diagrams.symmetry import external_wick_compensation
    from engine.integration.spatial.full_integrator import (
        diagram_correlator_pts, field_respecting_mappings)
    from simulations.spatial_field_1d_sim import simulate, third_cumulant_x

    g = 0.25
    # theory records at this coupling (rebuild: module fixture used g=0.1)
    import tests.test_spatial_general_k as _self
    from api.theory import TheoryBuilder
    from api.compute import FieldTheory
    from api._propagator import build_propagator
    from engine.integration.spatial.diagram_descriptor import diagram_to_cstack
    from engine.integration.spatial.pipeline_bridge import (
        build_pipeline_records, _legs_to_phys_idx)
    from engine.diagrams.type_assignment import build_field_index_map
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


@pytest.mark.slow
def test_k3_public_api_compute_cumulants():
    """k=3 through the public compute_cumulants(spatial_points=...) API
    reproduces the bridge-level (sim-anchored) values."""
    from api.theory import TheoryBuilder
    from api import compute_cumulants
    g = 0.25
    model = (TheoryBuilder('rd-quad-kpoint-api-test', n_populations=0)
             .physical_field('p', spatial_dim=1)
             .parameter('mu', default=1.0, domain='positive')
             .parameter('DD', default=1.0, domain='positive')
             .parameter('g', default=g, domain='real')
             .parameter('T', default=1.0, domain='positive')
             .equation(lhs='(Dt+mu-DD*Laplacian)*p', rhs='0')
             .set_action_text('pt*(Dt(p)+mu*p-DD*Lap(p)+g*p^2) - T*pt^2')
             .operator_ir().boundary('infinite').initial('stationary')
             .build())
    pts = np.array([[[0.0, 0.0], [0.0, 0.0]],
                    [[1.0, 0.0], [1.0, 0.0]]])
    th = compute_cumulants(
        model, k=3, max_ell=1, external_fields=[('p', 1)] * 3,
        fundamental={'mu': 1.0, 'DD': 1.0, 'g': g, 'T': 1.0},
        spatial_points=pts, use_cache=False, parallel=False, verbose=False)
    tree = th['C_kpoint_by_ell'][0]
    tot = th['C_kpoint']
    # bridge-level values, sim-anchored within 0.4 sigma (see
    # test_k3_e2e_vs_simulator)
    assert np.allclose(tree, [-0.041667, -0.025020], atol=2e-4), tree
    assert np.allclose(tot, [-0.050562, -0.031481], atol=3e-4), tot


@pytest.mark.slow
def test_k3_derivative_vertex_analytic_ift():
    """k=3 derivative-vertex (KPZ) analytic IFT (H1, commit f3221eb): the
    q-path evaluates every record finite with form factors active, the
    analytic xs-path (multivariate Wick moment, ff.moment_x_multi) now
    returns finite values for every derivative record (no longer gated),
    and on the tree record the analytic xs value matches the numerical
    Fourier transform of the q-path to FT-grid tolerance — the same
    route-equivalence check used for plain vertices."""
    from api.theory import TheoryBuilder
    from api.compute import FieldTheory
    from api._propagator import build_propagator
    from engine.integration.spatial.diagram_descriptor import diagram_to_cstack
    from engine.integration.spatial.pipeline_bridge import (
        build_pipeline_records, _legs_to_phys_idx, _formfactor_callable)
    from engine.diagrams.type_assignment import build_field_index_map
    from engine.integration.spatial.full_integrator import diagram_kinematic

    b = (TheoryBuilder('kpz-k3-ift-test', n_populations=0)
         .physical_field('h', spatial_dim=1)
         .parameter('mu', default=1.0, domain='positive')
         .parameter('D', default=1.0, domain='positive')
         .parameter('c', default=0.3, domain='real')
         .parameter('T', default=1.0, domain='positive')
         .equation(lhs='(Dt+mu-D*Laplacian)*h', rhs='0')
         .set_action_text('ht*(Dt(h)+mu*h-D*Lap(h)-(c/2)*Dx(h,0)^2)-T*ht^2')
         .operator_ir().boundary('infinite').initial('stationary').build())
    ft = FieldTheory(b, taylor_order=5)
    ft.expand()
    prop = build_propagator(ft, b, use_cache=False, verbose=False)
    rvn = list(ft._ns._ring_var_names)
    _, pidx = build_field_index_map(rvn, ft._n_tilde)
    ext3 = _legs_to_phys_idx([('h', 1)] * 3, pidx)
    base = {SR.var('mu'): 1., SR.var('D'): 1., SR.var('c'): 0.3,
            SR.var('T'): 1., SR.var('hstar1'): 0.}
    vt = [{'weight': float(SR(t['weight']).subs(base)),
           'n_phys': t['n_phys'], 'chain': t['chain'], 'mode': t['mode']}
          for t in ft._ns._operator_ir_vertex_terms]
    be = build_pipeline_records(ft, b, prop, ext3, max_ell=1, k=3,
                                verbose=False)
    X = np.array([[0.5, -0.4]])
    n_checked = 0
    tree_rec = None
    for ell in sorted(be):
        for td, p in be[ell]:
            pv = float(SR(p).subs(base))
            if abs(pv) < 1e-14:
                continue
            d = diagram_to_cstack(td)
            ff = _formfactor_callable(td, vt, d=1)
            legs = list(d.external_legs)
            et = {legs[j]: t for j, t in enumerate((0.0, 0.2, -0.1))}
            kw = {} if len(d.internal_vertices) < 3 else {'n_t': 6, 'n_s': 8}
            v = diagram_kinematic(d, [0.4, -0.7], et, 1.0, 1.0,
                                  spatial_dim=1, formfactor=ff, **kw)
            assert np.isfinite(complex(v).real), (ell, v)
            n_checked += 1
            if ff is not None:
                # analytic xs-path now WORKS (was NotImplementedError before
                # f3221eb) — finite, no raise
                assert ff.moment_x_multi is not None, (ell, 'no moment_x_multi')
                vx = diagram_kinematic(d, [0.0, 0.0], et, 1.0, 1.0,
                                       spatial_dim=1, xs=X, formfactor=ff,
                                       **kw)
                assert np.all(np.isfinite(np.real(vx))), (ell, vx)
                if ell == 0 and tree_rec is None:
                    tree_rec = (d, ff, et)
    assert n_checked >= 12

    # route equivalence on the tree record: analytic xs == numerical q-FT
    assert tree_rec is not None
    d, ff, et = tree_rec
    kin_x = diagram_kinematic(d, [0.0, 0.0], et, 1.0, 1.0, spatial_dim=1,
                              xs=X, formfactor=ff)
    qm, nq = 14.0, 161
    qg = np.linspace(-qm, qm, nq)
    dq = qg[1] - qg[0]
    acc = 0.0
    for q0 in qg:
        row = np.array([complex(diagram_kinematic(
            d, [q0, q1], et, 1.0, 1.0, spatial_dim=1, formfactor=ff))
            for q1 in qg])
        acc += np.sum(np.exp(1j * (q0 * X[0, 0] + qg * X[0, 1])) * row).real \
            * dq * dq
    acc /= (2 * np.pi) ** 2
    assert abs(kin_x[0] - acc) / max(abs(acc), 1e-300) < 1e-3, (kin_x[0], acc)


@pytest.mark.slow
def test_k3_spectral_uniform_mass_identity(recs3):
    """Coupled-field spectral kinematic at k=3: with UNIFORM masses the
    glued-segment (Lyapunov) convention must reduce to the single-field
    kinematic, I_spec = 2^{-n_C} * I_single, on BOTH the q-path and the
    analytic-IFT xs-path.  Pins the matrix-B branches added to
    diagram_kinematic_spectral (H2 core).  Residual is sigma-quadrature
    only (4e-2 -> 1e-5 at n_s = 8 -> 24); n_s=24 here."""
    from engine.integration.spatial.full_integrator import (
        diagram_kinematic, diagram_kinematic_spectral, spectral_rows)
    mu = 1.0
    worst = 0.0
    for td, d, pv, ell in recs3:
        n_C = sum(1 for e in d.edges if e.kind == 'C')
        mt = np.full((len(spectral_rows(d)), 1), mu, dtype=complex)
        legs = list(d.external_legs)
        et = {legs[j]: t for j, t in enumerate((0.0, 0.3, -0.2))}
        kw = {'n_t': 8, 'n_s': 24}
        ks = diagram_kinematic_spectral(d, [0.4, -0.7], et, mt, 1.0,
                                        spatial_dim=1, **kw)
        k1 = diagram_kinematic(d, [0.4, -0.7], et, mu, 1.0,
                               spatial_dim=1, **kw)
        ref = k1 / 2.0 ** n_C
        worst = max(worst, abs(complex(ks[0]).real - ref)
                    / max(abs(ref), 1e-300))
        X = np.array([[0.5, -0.4]])
        kxs = diagram_kinematic_spectral(d, [0.0, 0.0], et, mt, 1.0,
                                         spatial_dim=1, xs=X, **kw)
        kx1 = diagram_kinematic(d, [0.0, 0.0], et, mu, 1.0,
                                spatial_dim=1, xs=X, **kw)
        refx = kx1[0] / 2.0 ** n_C
        worst = max(worst, abs(complex(kxs[0, 0]).real - refx)
                    / max(abs(refx), 1e-300))
    assert worst < 5e-4, worst


@pytest.mark.slow
def test_k3_coupled_decoupled_limit():
    """H2 anchor: compute_coupled_kpoint on a DECOUPLED 2-species theory
    (m_a=1, m_b=1.7, nonlinearity g*a^2 in species a only, externals all
    species a) reproduces the single-field compute_spatial_kpoint values
    at matched quadrature (tree + 1-loop, 3 event configurations).
    Exercises the full coupled chain: 2-species enumeration, fpairs
    threading, spectral projector weights (b-mode assignments must drop
    out), mapping-sum externals, matrix-B spectral kinematic at
    n_ext=2.  Residual is sigma-quadrature only (~4e-5)."""
    from api.theory import TheoryBuilder
    from api.compute import FieldTheory
    from api._propagator import build_propagator
    from engine.integration.spatial.pipeline_bridge import (
        compute_spatial_kpoint, compute_coupled_kpoint, _legs_to_phys_idx)
    from engine.diagrams.type_assignment import build_field_index_map

    g, ma, mb, DD = 0.25, 1.0, 1.7, 1.0
    b2 = (TheoryBuilder('coup-k3-anchor-test', n_populations=0)
          .physical_field('a', spatial_dim=1)
          .physical_field('b', spatial_dim=1)
          .parameter('ma', default=ma, domain='positive')
          .parameter('mb', default=mb, domain='positive')
          .parameter('DD', default=DD, domain='positive')
          .parameter('g', default=g, domain='real')
          .parameter('T', default=1.0, domain='positive')
          .equation(lhs='(Dt+ma-DD*Laplacian)*a', rhs='0')
          .equation(lhs='(Dt+mb-DD*Laplacian)*b', rhs='0')
          .set_action_text('at*(Dt(a)+ma*a-DD*Lap(a)+g*a^2) - T*at^2'
                           ' + bt*(Dt(b)+mb*b-DD*Lap(b)) - T*bt^2')
          .operator_ir().boundary('infinite').initial('stationary').build())
    ft2 = FieldTheory(b2, taylor_order=5)
    ft2.expand()
    prop2 = build_propagator(ft2, b2, use_cache=False, verbose=False)
    np2 = {SR.var('ma'): ma, SR.var('mb'): mb, SR.var('DD'): DD,
           SR.var('g'): g, SR.var('T'): 1.0,
           SR.var('astar1'): 0.0, SR.var('bstar1'): 0.0}
    tree_info = {'M': np.diag([ma, mb]), 'Dhat': np.zeros((2, 2)),
                 'D0': DD, 'bc_mode': 'infinite'}
    pts = np.array([[[0.0, 0.0], [0.0, 0.0]],
                    [[0.8, 0.0], [0.8, 0.0]],
                    [[0.5, 0.3], [-0.4, -0.2]]])
    _, pidx2 = build_field_index_map(list(ft2._ns._ring_var_names),
                                     ft2._n_tilde)
    extA = _legs_to_phys_idx([('a', 1)] * 3, pidx2)
    vc, _ic = compute_coupled_kpoint(ft2, b2, prop2, np2, extA, pts,
                                     tree_info, max_ell=1, verbose=False,
                                     n_t_loop=10, n_s_loop=24)

    b1 = (TheoryBuilder('coup-k3-anchor-test-ref', n_populations=0)
          .physical_field('p', spatial_dim=1)
          .parameter('mu', default=ma, domain='positive')
          .parameter('DD', default=DD, domain='positive')
          .parameter('g', default=g, domain='real')
          .parameter('T', default=1.0, domain='positive')
          .equation(lhs='(Dt+mu-DD*Laplacian)*p', rhs='0')
          .set_action_text('pt*(Dt(p)+mu*p-DD*Lap(p)+g*p^2) - T*pt^2')
          .operator_ir().boundary('infinite').initial('stationary').build())
    ft1 = FieldTheory(b1, taylor_order=5)
    ft1.expand()
    prop1 = build_propagator(ft1, b1, use_cache=False, verbose=False)
    np1 = {SR.var('mu'): ma, SR.var('DD'): DD, SR.var('g'): g,
           SR.var('T'): 1.0, SR.var('pstar1'): 0.0}
    _, pidx1 = build_field_index_map(list(ft1._ns._ring_var_names),
                                     ft1._n_tilde)
    extP = _legs_to_phys_idx([('p', 1)] * 3, pidx1)
    vs, _is = compute_spatial_kpoint(ft1, b1, prop1, np1, extP, pts,
                                     max_ell=1, verbose=False)
    rel = np.max(np.abs(vc - vs) / np.maximum(np.abs(vs), 1e-300))
    assert rel < 5e-3, (vc, vs, rel)


@pytest.mark.slow
def test_k3_coupled_public_api_fallback():
    """The public compute_cumulants(k=3, spatial_points=...) coupled
    fallback (compute.py: catch 'single-field' NotImplementedError →
    extract M/D/V via reaction_diffusion_matrices + split_reference_
    diffusion → compute_coupled_kpoint).  Exercises the extraction chain
    that the direct-driver tests bypass.  A DECOUPLED 2-species theory
    (g*a^2 in species a only, externals all-a) through the public API
    must reproduce the single-field public-API tree value."""
    from api.theory import TheoryBuilder
    from api import compute_cumulants

    g, ma, mb, DD = 0.25, 1.0, 1.7, 1.0
    coupled = (TheoryBuilder('coup-k3-pubfallback', n_populations=0)
               .physical_field('a', spatial_dim=1)
               .physical_field('b', spatial_dim=1)
               .parameter('ma', default=ma, domain='positive')
               .parameter('mb', default=mb, domain='positive')
               .parameter('DD', default=DD, domain='positive')
               .parameter('g', default=g, domain='real')
               .parameter('T', default=1.0, domain='positive')
               .equation(lhs='(Dt+ma-DD*Laplacian)*a', rhs='0')
               .equation(lhs='(Dt+mb-DD*Laplacian)*b', rhs='0')
               .set_action_text('at*(Dt(a)+ma*a-DD*Lap(a)+g*a^2) - T*at^2'
                                ' + bt*(Dt(b)+mb*b-DD*Lap(b)) - T*bt^2')
               .operator_ir().boundary('infinite').initial('stationary')
               .build())
    single = (TheoryBuilder('coup-k3-pubfallback-ref', n_populations=0)
              .physical_field('p', spatial_dim=1)
              .parameter('mu', default=ma, domain='positive')
              .parameter('DD', default=DD, domain='positive')
              .parameter('g', default=g, domain='real')
              .parameter('T', default=1.0, domain='positive')
              .equation(lhs='(Dt+mu-DD*Laplacian)*p', rhs='0')
              .set_action_text('pt*(Dt(p)+mu*p-DD*Lap(p)+g*p^2) - T*pt^2')
              .operator_ir().boundary('infinite').initial('stationary')
              .build())
    pts = np.array([[[0.0, 0.0], [0.0, 0.0]], [[1.0, 0.0], [1.0, 0.0]]])
    kw = dict(k=3, max_ell=0, spatial_points=pts,
              use_cache=False, parallel=False, verbose=False)
    th_c = compute_cumulants(coupled, external_fields=[('a', 1)] * 3,
                             fundamental={'ma': ma, 'mb': mb, 'DD': DD,
                                          'g': g, 'T': 1.0}, **kw)
    th_s = compute_cumulants(single, external_fields=[('p', 1)] * 3,
                             fundamental={'mu': ma, 'DD': DD, 'g': g,
                                          'T': 1.0}, **kw)
    rel = np.max(np.abs(th_c['C_kpoint'] - th_s['C_kpoint'])
                 / np.maximum(np.abs(th_s['C_kpoint']), 1e-300))
    assert rel < 5e-3, (th_c['C_kpoint'], th_s['C_kpoint'], rel)


@pytest.mark.slow
def test_k3_coupled_cross_complex_modes():
    """Cross-coupled 2-species k=3 (M = [[1.5,.4],[-.3,1.2]], complex
    eigenvalue pair 1.35 +- 0.31i): the spectral sum over ALL 2^n_rows
    assignments must stay REAL to machine precision (conjugate-mode
    cancellation) at tree (3 external-field configs) AND 1-loop, with a
    sensible cross-cumulant hierarchy (nonlinearity in species a only).
    1-loop runs at the memory-safe grid: the (grid x assignments) amp
    array is the cost driver (17 GB at n_t=8/n_s=16 -> use 6/8)."""
    from api.theory import TheoryBuilder
    from api.compute import FieldTheory
    from api._propagator import build_propagator
    from engine.integration.spatial.pipeline_bridge import (
        compute_coupled_kpoint, _legs_to_phys_idx)
    from engine.diagrams.type_assignment import build_field_index_map

    ga, DD = 0.3, 1.0
    b2 = (TheoryBuilder('coup-k3-cross-test', n_populations=0)
          .physical_field('a', spatial_dim=1)
          .physical_field('b', spatial_dim=1)
          .parameter('DD', default=DD, domain='positive')
          .parameter('ga', default=ga, domain='real')
          .parameter('T', default=1.0, domain='positive')
          .equation(lhs='(Dt+1.5-DD*Laplacian)*a+0.4*b', rhs='0')
          .equation(lhs='(Dt+1.2-DD*Laplacian)*b-0.3*a', rhs='0')
          .set_action_text(
              'at*(Dt(a)+1.5*a+0.4*b-DD*Lap(a)+ga*a^2) - T*at^2'
              ' + bt*(Dt(b)-0.3*a+1.2*b-DD*Lap(b)) - T*bt^2')
          .operator_ir().boundary('infinite').initial('stationary').build())
    ft2 = FieldTheory(b2, taylor_order=5)
    ft2.expand()
    prop2 = build_propagator(ft2, b2, use_cache=False, verbose=False)
    np2 = {SR.var('DD'): DD, SR.var('ga'): ga, SR.var('T'): 1.0,
           SR.var('astar1'): 0.0, SR.var('bstar1'): 0.0}
    M = np.array([[1.5, 0.4], [-0.3, 1.2]])
    tree_info = {'M': M, 'Dhat': np.zeros((2, 2)), 'D0': DD,
                 'bc_mode': 'infinite'}
    _, pidx = build_field_index_map(list(ft2._ns._ring_var_names),
                                    ft2._n_tilde)
    pts2 = np.array([[[0.0, 0.0], [0.0, 0.0]], [[0.7, 0.0], [0.7, 0.0]]])
    vals = {}
    for name, legs in (('aaa', [('a', 1)] * 3),
                       ('aab', [('a', 1), ('a', 1), ('b', 1)]),
                       ('abb', [('a', 1), ('b', 1), ('b', 1)])):
        ext = _legs_to_phys_idx(legs, pidx)
        v, info = compute_coupled_kpoint(ft2, b2, prop2, np2, ext, pts2,
                                         tree_info, max_ell=0,
                                         verbose=False)
        assert info['max_abs_imag'] < 1e-12, (name, info['max_abs_imag'])
        vals[name] = v
    # nonlinearity is in species a only: cross-cumulants are suppressed
    assert abs(vals['aaa'][0]) > 10 * abs(vals['aab'][0]) > 0
    assert abs(vals['aab'][0]) > abs(vals['abb'][0]) > 0
    # 1-loop at the origin: stays real, sensible size relative to tree
    ext = _legs_to_phys_idx([('a', 1)] * 3, pidx)
    pts1 = np.array([[[0.0, 0.0], [0.0, 0.0]]])
    v1, i1 = compute_coupled_kpoint(ft2, b2, prop2, np2, ext, pts1,
                                    tree_info, max_ell=1, verbose=False,
                                    n_t_loop=6, n_s_loop=8)
    assert i1['max_abs_imag'] < 1e-12
    d1 = i1['per_ell'][1][0]
    assert 0 < abs(d1) < 0.5 * abs(i1['per_ell'][0][0]), (d1, i1['per_ell'])


@pytest.mark.slow
def test_k3_nongaussian_noise_source_tree():
    """Spatial NON-GAUSSIAN noise: a phi-tilde^3 source (third noise
    cumulant kappa^(3) = 3! S3) maps to an internal vertex with three
    retarded edges and flows through the generic chamber integrator.
    Tree kappa_3(x1, x2) vs the independent semi-analytic oracle
    (analytic q2 + s integrals, 1-d quadrature) to ~1e-4 at n_t=60.
    The equal-point value kappa_3(0,0) is UV log-divergent in d=1
    (integrand ~ ds/s) — excluded by design, like the d>=2 Gaussian
    divergences (bare values are cutoff-sensitive)."""
    from api.theory import TheoryBuilder
    from api.compute import FieldTheory
    from api._propagator import build_propagator
    from engine.integration.spatial.diagram_descriptor import diagram_to_cstack
    from engine.integration.spatial.pipeline_bridge import (
        build_pipeline_records, _legs_to_phys_idx)
    from engine.diagrams.type_assignment import build_field_index_map
    from engine.integration.spatial.full_integrator import (
        diagram_correlator_pts, field_respecting_mappings)
    from engine.diagrams.symmetry import external_wick_compensation
    from scipy.integrate import quad

    mu, DD, T, S3 = 1.0, 1.0, 1.0, 0.1
    b = (TheoryBuilder('ng-noise-test', n_populations=0)
         .physical_field('p', spatial_dim=1)
         .parameter('mu', default=mu, domain='positive')
         .parameter('DD', default=DD, domain='positive')
         .parameter('T', default=T, domain='positive')
         .parameter('S3', default=S3)
         .equation(lhs='(Dt+mu-DD*Laplacian)*p', rhs='0')
         .set_action_text('pt*(Dt(p)+mu*p-DD*Lap(p)) - T*pt^2 - S3*pt^3')
         .operator_ir().boundary('infinite').initial('stationary').build())
    ft = FieldTheory(b, taylor_order=3)
    ft.expand()
    prop = build_propagator(ft, b, use_cache=False, verbose=False)
    rvn = list(ft._ns._ring_var_names)
    _, pidx = build_field_index_map(rvn, ft._n_tilde)
    ext3 = _legs_to_phys_idx([('p', 1)] * 3, pidx)
    base = {SR.var('mu'): mu, SR.var('DD'): DD, SR.var('T'): T,
            SR.var('S3'): S3, SR.var('pstar1'): 0.}
    be = build_pipeline_records(ft, b, prop, ext3, max_ell=0, k=3,
                                verbose=False)
    td, p = be[0][0]
    pv = float(SR(p).subs(base))
    assert abs(pv - 6 * S3) < 1e-12          # kappa^(3) = 3! S3, S(Gamma)=3!
    d = diagram_to_cstack(td)
    assert sorted(e.kind for e in d.edges) == ['R', 'R', 'R']
    assert len(d.internal_vertices) == 1     # the source IS the vertex
    maps = field_respecting_mappings(['p'] * 3, ['p'] * 3)
    comp = external_wick_compensation(td)
    pts = [(0.4, 0.4), (0.6, 0.6), (0.3, 0.9), (0.0, 0.8)]
    x_pts = np.array([[0.0, a_, b_] for a_, b_ in pts])
    v = diagram_correlator_pts(d, pv, x_pts, [0.0, 0.0, 0.0], mu, DD,
                               spatial_dim=1, mappings=maps, comp=comp,
                               n_t=60)

    def oracle(x1, x2):
        def f(q1):
            al = 3*mu + 1.5*DD*q1*q1
            return (np.cos(q1*(x1 - x2/2.0)) * np.pi/np.sqrt(2*DD*al)
                    * np.exp(-abs(x2)*np.sqrt(al/(2*DD))))
        val, _ = quad(f, -np.inf, np.inf, epsabs=1e-13, epsrel=1e-12,
                      limit=400)
        return pv * val / (2*np.pi)**2

    o = np.array([oracle(a_, b_) for a_, b_ in pts])
    rel = np.max(np.abs(v - o) / np.abs(o))
    assert rel < 2e-4, (v, o, rel)
