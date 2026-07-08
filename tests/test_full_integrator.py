"""
tests/test_full_integrator.py
=============================
The genuine full-diagram integrator (``engine.integration.spatial.full_integrator``)
at the **tree** and **2-loop (ell=2)** level, in ``d=1`` AND ``d=2``.

  * tree ``Γ == C₀(q,τ)`` to machine precision;
  * the 2-loop **Keldysh sunset** (3 correlation lines between two vertices, two
    external retarded legs) — found by structure in the enumerated Allen-Cahn
    ``φ⁴`` ``ell=2`` set — validated against an INDEPENDENT direct brute-force:
    ``d=1``: explicit ``∫dℓ₁dℓ₂`` + the analytic 2-vertex time integral;
    ``d=2``: explicit ``∫d²ℓ₁d²ℓ₂`` (the UV-finite sunset; both schemes converge).

This proves the integrator is general in ``ell`` (the Symanzik handles ``L=2``)
and in ``d`` (a parameter flip), with no shortcuts.

Run:  sage -python -m pytest tests/test_full_integrator.py -q
"""
from __future__ import annotations

import importlib.util
import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
_REPO = os.path.join(os.path.dirname(__file__), '..')

from engine.integration.spatial.diagram_descriptor import diagram_to_cstack
from engine.integration.spatial.full_integrator import (
    diagram_value, diagram_kinematic, external_times_2pt,
)
from engine.integration.spatial.pipeline_bridge import (
    build_pipeline_records, _legs_to_phys_idx,
)
from engine.diagrams.type_assignment import build_field_index_map


def _allen_cahn():
    path = os.path.join(_REPO, 'models',
                        'allen_cahn_1d_subcritical_infinite.model.py')
    spec = importlib.util.spec_from_file_location('ac', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.build()


@pytest.fixture(scope='module')
def ac_records():
    from sage.all import SR
    from api._propagator import build_propagator
    from api.compute import FieldTheory
    b = _allen_cahn()
    ft = FieldTheory(b, taylor_order=6)            # k + 2·ell = 2 + 4
    ft.expand()
    ft.sanity_check(verbose=False)
    prop = build_propagator(ft, b, use_cache=False, verbose=False)
    rvn = list(ft._ns._ring_var_names)
    _, pidx = build_field_index_map(rvn, ft._n_tilde)
    ext = _legs_to_phys_idx([('phi', 1), ('phi', 1)], pidx)
    by_ell = build_pipeline_records(ft, b, prop, ext, max_ell=2, verbose=False)
    base = {SR.var('mu'): 1.0, SR.var('D'): 1.0, SR.var('T'): 1.0,
            SR.var('lam'): 0.1, SR.var('phistar1'): 0.0}
    return by_ell, base, SR


def _find_keldysh_sunset(by_ell):
    """A 2-loop diagram with 2 internal vertices joined by 3 correlation lines
    (each carrying loop momentum) and two external retarded legs."""
    for td, _pre in by_ell.get(2, []):
        d = diagram_to_cstack(td)
        if len(d.internal_vertices) != 2 or d.n_loops != 2:
            continue
        loop = d.loop_edges()
        ext = [e for e in d.edges if e.external]
        if (len(loop) == 3 and all(e.kind == 'C' and e.u != e.v for e in loop)
                and len(ext) == 2 and all(e.kind == 'R' for e in ext)):
            return d
    raise AssertionError('no Keldysh sunset found in the ell=2 set')


def _sunset_brute(descr, q, mu, D, d, L_cut, n):
    """Direct brute-force of the sunset's kinematic value at τ=0: ∫dᵈℓ₁dᵈℓ₂/(2π)^{2d}
    of (1/(m1 m2 m3))·2/(M−mq)·[1/(2mq)−1/(mq+M)] (the analytic 2-vertex time
    integral), reading the 3 loop C edges' routing (a,b) off the descriptor."""
    loop = descr.loop_edges()
    ax = np.linspace(-L_cut, L_cut, n)
    dl = ax[1] - ax[0]
    if d == 1:
        l1, l2 = np.meshgrid(ax, ax, indexing='ij')
        comps = [(l1, l2)]                          # one spatial component
    else:
        g = np.meshgrid(ax, ax, ax, ax, indexing='ij')
        comps = [(g[0], g[2]), (g[1], g[3])]        # (ℓ1·,ℓ2·) per axis; q on axis 0
    mk = lambda k2: mu + D * k2

    def m_edge(e):
        k2 = 0.0
        for ci, (a1, a2) in enumerate(comps):
            kc = e.a[0] * a1 + e.a[1] * a2 + (e.b[0] if (ci == 0 and e.b) else 0) * q
            k2 = k2 + kc * kc
        return mk(k2)
    m1, m2, m3 = (m_edge(e) for e in loop)
    M = m1 + m2 + m3
    mq = mk(q * q)
    integ = (1.0 / (m1 * m2 * m3)) * 2.0 / (M - mq) * (1.0 / (2 * mq) - 1.0 / (mq + M))
    return float(np.sum(integ)) * dl ** (2 * d) / (2 * math.pi) ** (2 * d)


def test_tree_is_C0(ac_records):
    by_ell, base, SR = ac_records
    mu, D, T = 1.0, 1.0, 1.0
    d0 = diagram_to_cstack(by_ell[0][0][0])
    pre = float(SR(by_ell[0][0][1]).subs(base))
    for q in (0.0, 0.7, 1.5):
        for tau in (0.0, 0.6):
            m = mu + D * q * q
            C0 = (T / m) * math.exp(-m * abs(tau))
            val = diagram_value(d0, pre, [q], external_times_2pt(d0, tau), mu, D, 1)
            assert abs(val - C0) <= 1e-9 * (abs(C0) + 1e-12)


def test_sunset_d1_vs_brute(ac_records):
    by_ell, _base, _SR = ac_records
    sun = _find_keldysh_sunset(by_ell)
    mu, D = 1.0, 1.0
    for q in (0.0, 0.6, 1.4):
        kin = diagram_kinematic(sun, [q], external_times_2pt(sun, 0.0), mu, D, 1,
                                n_t=22, n_s=22)
        brute = _sunset_brute(sun, q, mu, D, 1, L_cut=26.0, n=420)
        assert abs(kin - brute) <= 1e-4 * abs(brute), \
            f"d=1 sunset q={q}: {kin} vs {brute}"


def test_sunset_d2_vs_brute(ac_records):
    by_ell, _base, _SR = ac_records
    sun = _find_keldysh_sunset(by_ell)
    mu, D, q = 1.0, 1.0, 0.6
    kin = diagram_kinematic(sun, [q], external_times_2pt(sun, 0.0), mu, D, 2,
                            n_t=22, n_s=26)
    brute = _sunset_brute(sun, q, mu, D, 2, L_cut=20.0, n=64)   # UV-finite; both converge
    assert abs(kin - brute) <= 5e-3 * abs(brute), f"d=2 sunset: {kin} vs {brute}"


def test_formfactor_bubble_vs_oracle():
    """DERIVATIVE-vertex (Model-B conserved ``∇²(φ²)``) 1-loop bubble through the
    full integrator — the loop-momentum form factor ``F(ℓ,q)`` deposited by the ∇
    and averaged over the loop Gaussian by Gauss–Hermite — vs the INDEPENDENT
    ``loop_dyson`` oracle (itself sim-validated at B≈0.944).  Two completely
    different discretizations ⇒ agreement to ~2% locks in the new capability."""
    import importlib.util
    import numpy as np
    from sage.all import SR
    from api._propagator import build_propagator
    from api.compute import FieldTheory
    from engine.integration.spatial.diagram_descriptor import diagram_to_cstack
    from engine.integration.spatial.pipeline_bridge import (
        build_pipeline_records, _legs_to_phys_idx, _formfactor_callable)
    from engine.integration.spatial.full_integrator import diagram_correlator
    from engine.integration.spatial import loop_dyson
    from engine.diagrams.type_assignment import build_field_index_map

    path = os.path.join(_REPO, 'models',
                        'reaction_diffusion_conserved_1d.model.py')
    spec = importlib.util.spec_from_file_location('crd', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    b = mod.build()
    ft = FieldTheory(b, taylor_order=4)
    ft.expand()
    prop = build_propagator(ft, b, use_cache=False, verbose=False)
    rvn = list(ft._ns._ring_var_names)
    _, pidx = build_field_index_map(rvn, ft._n_tilde)
    ext = _legs_to_phys_idx([('phi', 1), ('phi', 1)], pidx)
    op = ft._ns._operator_ir_vertex_chain
    be = build_pipeline_records(ft, b, prop, ext, max_ell=1, verbose=False)
    mu, D, g, T = 1.0, 2.0, 0.3, 1.0
    base = {SR.var('mu'): mu, SR.var('D'): D, SR.var('g'): g, SR.var('T'): T,
            SR.var('phistar1'): 0.0}
    diags = [(diagram_to_cstack(td), float(SR(pre).subs(base)),
              _formfactor_callable(td, op)) for td, pre in be.get(1, [])]
    for q in (0.6, 1.2):
        full = sum(diagram_correlator(d, pv, q, 0.0, mu, D, 1, formfactor=ff,
                                      n_t=24, n_s=26) for d, pv, ff in diags)
        orc = float(np.real(loop_dyson.bubble_delta_C_q_tau(
            q, [0.0], mu, D, T, g=g,
            formfactor=lambda l: (q * q) * (l * l),
            formfactor_K=lambda l: q ** 4 * np.ones_like(l))[0]))
        assert abs(full - orc) <= 2e-2 * abs(orc), \
            f"formfactor bubble q={q}: full={full} oracle={orc}"


def test_formfactor_average_convolution_kernel():
    """The loop form-factor average integrates ARBITRARY smooth convolution
    kernels ``ŵ(ℓ)`` — not just the polynomial derivative-vertex case — so a
    spatiotemporal-convolution VERTEX (neural-field ``f(φ)⊛w``, nonlocal
    Allen-Cahn) flows through the SAME path.  Gauss–Hermite ``⟨ŵ⟩`` matches the
    brute Gaussian average ``∫ŵ·G/∫G``: a Gaussian kernel ~exact, a Lorentzian to
    ~1e-3.  (See ``docs/spatiotemporal_convolutions.md``.)"""
    import numpy as np
    from engine.integration.spatial.full_integrator import _formfactor_average

    D, q = 0.8, 0.7
    a = np.array([1.0, 1.0]); b = np.array([0.0, -1.0]); w = np.array([0.6, 0.9])
    M = np.array([[[float(np.sum(w * a * a))]]])      # (P=1, L=1, L=1)
    N = np.array([[[float(np.sum(w * a * b))]]])      # (P=1, L=1, n_ext=1)
    ok = np.array([True])

    def brute(kern, L=120.0, n=600001):
        ell = np.linspace(-L, L, n)
        G = np.exp(-D * (w[0] * ell ** 2 + w[1] * (ell - q) ** 2))
        return float(np.sum(kern(ell) * G) / np.sum(G))   # ⟨ŵ⟩ over the loop Gaussian

    cases = [('gaussian', lambda l: np.exp(-(0.9 ** 2) * l ** 2 / 2), 1e-6, 12),
             ('lorentzian', lambda l: 1.0 / (1.0 + (1.3 ** 2) * l ** 2), 1e-3, 24)]
    for name, kern, tol, n in cases:
        ff = lambda ell, qq, k=kern: k(ell[..., 0])
        avg = _formfactor_average(ff, M, N, [q], D, ok, gh_order=n)[0]
        assert abs(avg - brute(kern)) <= tol, \
            f"convolution kernel {name}: GH={avg} vs brute={brute(kern)}"


def test_diagram_form_factor_ell2_momentum():
    """Derivative-vertex form factors are GENERIC in loop number (NOT bubble-
    specific): on a real ℓ=2 (L=2) conserved-``∇²(φ²)`` diagram, ``MomFactor·⟨F⟩``
    matches a brute 2-D ``∫dℓ₀dℓ₁ F(ℓ,q)·Gaussian`` to machine precision —
    confirming the per-vertex form-factor product composes for any topology and the
    loop-basis ↔ integrator-column mapping is consistent at L=2."""
    import importlib.util
    import numpy as np
    from sage.all import SR
    from api._propagator import build_propagator
    from api.compute import FieldTheory
    from engine.integration.spatial.diagram_descriptor import diagram_to_cstack
    from engine.integration.spatial.pipeline_bridge import (
        build_pipeline_records, _legs_to_phys_idx, _formfactor_callable)
    from engine.integration.spatial.full_integrator import (
        _momentum_factor_batch, _formfactor_average)
    from engine.diagrams.type_assignment import build_field_index_map

    path = os.path.join(_REPO, 'models',
                        'reaction_diffusion_conserved_1d.model.py')
    spec = importlib.util.spec_from_file_location('crd', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    b = mod.build()
    ft = FieldTheory(b, taylor_order=6)            # k + 2·ell = 2 + 4 (ell=2)
    ft.expand()
    prop = build_propagator(ft, b, use_cache=False, verbose=False)
    rvn = list(ft._ns._ring_var_names)
    _, pidx = build_field_index_map(rvn, ft._n_tilde)
    ext = _legs_to_phys_idx([('phi', 1), ('phi', 1)], pidx)
    op = ft._ns._operator_ir_vertex_chain
    be = build_pipeline_records(ft, b, prop, ext, max_ell=2, verbose=False)
    base = {SR.var('mu'): 1.0, SR.var('D'): 2.0, SR.var('g'): 0.3,
            SR.var('T'): 1.0, SR.var('phistar1'): 0.0}
    td = descr = None
    for t, p in be.get(2, []):
        if abs(float(SR(p).subs(base))) <= 1e-14:
            continue
        d = diagram_to_cstack(t)
        if d.n_loops == 2:
            td, descr = t, d
            break
    assert td is not None, 'no live L=2 conserved-vertex diagram found'
    ff = _formfactor_callable(td, op)
    a = np.array([e.a for e in descr.edges], dtype=float).reshape(len(descr.edges), -1)
    bb = np.array([e.b for e in descr.edges], dtype=float).reshape(len(descr.edges), -1)
    E, L = a.shape
    assert L == 2
    D, q = 2.0, 0.7
    rng = np.random.default_rng(0)
    for _ in range(3):
        w = 0.4 + 0.8 * rng.random(E)
        momfac, M, N, ok = _momentum_factor_batch(a, bb, w[None, :], [q], D, 1,
                                                   return_gaussian=True)
        gh = float((momfac[0] * _formfactor_average(ff, M, N, [q], D, ok,
                                                    gh_order=10)[0]).real)
        Lc, n = 26.0, 340
        ax = np.linspace(-Lc, Lc, n); dl = ax[1] - ax[0]
        L0, L1 = np.meshgrid(ax, ax, indexing='ij')
        ell = np.stack([L0, L1], axis=-1)
        expo = np.zeros_like(L0)
        for e in range(E):
            ke = a[e, 0] * L0 + a[e, 1] * L1 + bb[e, 0] * q
            expo = expo + w[e] * ke * ke
        brute = float(np.sum(ff(ell, q) * np.exp(-D * expo)) * dl * dl
                      / (2 * np.pi) ** 2)
        assert abs(gh - brute) <= 1e-6 * (abs(brute) + 1e-30), \
            f"L=2 form-factor momentum integral: {gh} vs {brute}"


def test_perleg_and_complex_form_factor():
    """KPZ-type PER-LEG derivative form factors + COMPLEX (odd-∂) integrands —
    the new machinery for KPZ ``(∂φ)²`` / Burgers.  On a φ̃φ² bubble topology, the
    per-leg extraction (``∂_x`` on each physical/incoming leg → ``∏ i·p_leg``) and
    the complex Gauss–Hermite average match a brute ``∫dℓ`` to machine precision.
    (The end-to-end wiring of a KPZ/Burgers MODEL is separately gated on the v2
    k-explicit propagator kernel — see ``docs/spatial_kpz_burgers_plan.md``.)"""
    import importlib.util
    import numpy as np
    from sage.all import SR
    from api._propagator import build_propagator
    from api.compute import FieldTheory
    from engine.integration.spatial.diagram_descriptor import diagram_to_cstack
    from engine.integration.spatial.pipeline_bridge import (
        build_pipeline_records, _legs_to_phys_idx, _formfactor_callable)
    from engine.integration.spatial.full_integrator import (
        _momentum_factor_batch, _formfactor_average)
    from engine.diagrams.type_assignment import build_field_index_map

    path = os.path.join(_REPO, 'models',
                        'reaction_diffusion_conserved_1d.model.py')
    spec = importlib.util.spec_from_file_location('crd', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    b = mod.build()
    ft = FieldTheory(b, taylor_order=4)
    ft.expand()
    prop = build_propagator(ft, b, use_cache=False, verbose=False)
    rvn = list(ft._ns._ring_var_names)
    _, pidx = build_field_index_map(rvn, ft._n_tilde)
    ext = _legs_to_phys_idx([('phi', 1), ('phi', 1)], pidx)
    be = build_pipeline_records(ft, b, prop, ext, max_ell=1, verbose=False)
    base = {SR.var('mu'): 1.0, SR.var('D'): 2.0, SR.var('g'): 0.3,
            SR.var('T'): 1.0, SR.var('phistar1'): 0.0}
    td = descr = None
    for t, p in be.get(1, []):
        if abs(float(SR(p).subs(base))) <= 1e-14:
            continue
        d = diagram_to_cstack(t)
        if d.n_loops == 1:
            td, descr = t, d
            break
    assert td is not None
    a = np.array([e.a for e in descr.edges], dtype=float).reshape(len(descr.edges), -1)
    bb = np.array([e.b for e in descr.edges], dtype=float).reshape(len(descr.edges), -1)
    E = a.shape[0]
    D, q = 2.0, 0.7

    def brute(ff, w):
        Lc, n = 60.0, 200001
        ell = np.linspace(-Lc, Lc, n)
        dl = ell[1] - ell[0]
        expo = sum(w[e] * (a[e, 0] * ell + bb[e, 0] * q) ** 2 for e in range(E))
        return complex(np.sum(ff(ell[..., None], q) * np.exp(-D * expo))
                       * dl / (2 * np.pi))

    ff_kpz = _formfactor_callable(td, (('Dx', 0),), mode='perleg')   # KPZ per-leg
    ff_cplx = lambda ell, qq: 1j * ell[..., 0]                        # odd ∂ (imaginary)
    rng = np.random.default_rng(1)
    for ff in (ff_kpz, ff_cplx):
        w = 0.5 + 0.6 * rng.random(E)
        mf, M, N, ok = _momentum_factor_batch(a, bb, w[None, :], [q], D, 1,
                                              return_gaussian=True)
        gh = complex(mf[0] * _formfactor_average(ff, M, N, [q], D, ok,
                                                 gh_order=12)[0])
        br = brute(ff, w)
        assert abs(gh - br) <= 1e-6 * (abs(br) + 1e-30), \
            f"per-leg/complex form factor: {gh} vs {br}"


def _build_ff_model(kind, d):
    """Model B ∇²(φ²) (composite Lap) or KPZ (∇h)²=Σ_i(∂_i h)² (per-leg) at dim d."""
    from api.model import ModelBuilder
    tb = (ModelBuilder(kind, n_populations=0).physical_field('phi', spatial_dim=d)
          .parameter('mu', default=1.0, domain='positive')
          .parameter('D', default=1.0, domain='positive')
          .parameter('c', default=0.3, domain='real')
          .parameter('T', default=1.0, domain='positive'))
    if kind == 'modelb':
        return (tb.equation(lhs='(Dt + mu - D*Laplacian)*phi', rhs='c*Laplacian*phi^2')
                .set_action_text('phit*(Dt(phi)+mu*phi-D*Lap(phi)-c*Lap(phi^2))-T*phit^2')
                .operator_ir().boundary('infinite').initial('stationary').build())
    grad2 = ' + '.join(f'Dx(phi,{i})^2' for i in range(d))
    return (tb.equation(lhs='(Dt + mu - D*Laplacian)*phi', rhs='0')
            .set_action_text(f'phit*(Dt(phi)+mu*phi-D*Lap(phi)-(c/2)*({grad2}))-T*phit^2')
            .operator_ir().boundary('infinite').initial('stationary').build())


@pytest.mark.parametrize('kind,d,tol', [('modelb', 2, 1e-10), ('kpz', 2, 1e-10),
                                        ('modelb', 3, 3e-3), ('kpz', 3, 3e-3)])
def test_formfactor_d_ge_2_vs_brute(kind, d, tol):
    """The d≥2 derivative-vertex loop integral MomFactor·⟨F⟩ (the transverse-
    moment L·d-dim Gauss–Hermite average) matches a brute ``∫dᵈℓ F·Gaussian`` —
    Lap composite (Model B) AND full-gradient per-leg (KPZ ``(∇h)²``).  q on axis 0;
    d=2 is exact, d=3 to the (coarse) brute grid."""
    import numpy as np
    from sage.all import SR
    from api._propagator import build_propagator
    from api.compute import FieldTheory
    from engine.integration.spatial.diagram_descriptor import diagram_to_cstack
    from engine.integration.spatial.pipeline_bridge import (
        build_pipeline_records, _legs_to_phys_idx, _formfactor_callable)
    from engine.integration.spatial.full_integrator import (
        _momentum_factor_batch, _formfactor_average)
    from engine.diagrams.type_assignment import build_field_index_map

    b = _build_ff_model(kind, d)
    ft = FieldTheory(b, taylor_order=4); ft.expand()
    prop = build_propagator(ft, b, use_cache=False, verbose=False)
    rvn = list(ft._ns._ring_var_names)
    _, pidx = build_field_index_map(rvn, ft._n_tilde)
    ext = _legs_to_phys_idx([('phi', 1), ('phi', 1)], pidx)
    base = {SR.var('mu'): 1.0, SR.var('D'): 1.0, SR.var('c'): 0.3,
            SR.var('T'): 1.0, SR.var('phistar1'): 0.0}
    vt = [{'weight': float(SR(t['weight']).subs(base)), 'n_phys': t['n_phys'],
           'chain': t['chain'], 'mode': t['mode']}
          for t in ft._ns._operator_ir_vertex_terms]
    be = build_pipeline_records(ft, b, prop, ext, max_ell=1, verbose=False)
    td = descr = None
    for t, p in be.get(1, []):
        if abs(float(SR(p).subs(base))) <= 1e-14:
            continue
        dd = diagram_to_cstack(t)
        if dd.n_loops == 1:
            td, descr = t, dd; break
    assert td is not None, f'no live L=1 {kind} bubble at d={d}'
    ff = _formfactor_callable(td, vt, d=d)
    a = np.array([e.a for e in descr.edges], dtype=float).reshape(len(descr.edges), -1)
    bb = np.array([e.b for e in descr.edges], dtype=float).reshape(len(descr.edges), -1)
    E = a.shape[0]
    D, q = 1.0, 0.7
    rng = np.random.default_rng(1)
    for _ in range(2):
        w = 0.4 + 0.8 * rng.random(E)
        momfac, M, N, ok = _momentum_factor_batch(a, bb, w[None, :], [q], D, d,
                                                  return_gaussian=True)
        if M is None:
            continue
        gh = complex(momfac[0] * _formfactor_average(ff, M, N, [q], D, ok,
                                                     gh_order=8, spatial_dim=d)[0])
        Lc, n = (22.0, 160) if d == 2 else (15.0, 44)
        ax = np.linspace(-Lc, Lc, n); dl = ax[1] - ax[0]
        grids = np.meshgrid(*([ax] * d), indexing='ij')
        ell = np.stack(grids, axis=-1)[..., None, :]               # (…, L=1, d)
        expo = np.zeros_like(grids[0])
        for e in range(E):
            for al in range(d):
                ke = a[e, 0] * grids[al] + (bb[e, 0] * q if al == 0 else 0.0)
                expo = expo + w[e] * ke * ke
        qarr = np.zeros((1, d)); qarr[0, 0] = q
        brute = complex(np.sum(ff(ell, qarr) * np.exp(-D * expo))
                        * dl ** d / (2 * np.pi) ** d)
        assert abs(gh - brute) <= tol * (abs(brute) + 1e-30), \
            f"{kind} d={d}: GH {gh} vs brute {brute}"


def test_spatial_parallel_matches_serial():
    """The fork-parallel spatial integration (over q-points) is BIT-IDENTICAL to
    serial — same diagram-summation order, q-points independent.  Mirrors the
    temporal test_phase_J_total_C_batch_parallel_matches_serial."""
    import numpy as np
    from api.model import ModelBuilder
    from api.compute import compute_cumulants
    ac = (ModelBuilder('ac-par', n_populations=0).physical_field('phi', spatial_dim=1)
          .parameter('mu', default=1.0, domain='positive')
          .parameter('D', default=1.0, domain='positive')
          .parameter('lam', default=0.1, domain='positive')
          .parameter('T', default=1.0, domain='positive')
          .equation(lhs='(Dt + mu - D*Laplacian)*phi', rhs='-lam*phi^3')
          .set_action_text('phit*((Dt + mu - D*Laplacian)*phi + lam*phi^3) - T*phit^2')
          .boundary('infinite').initial('stationary').build())
    kw = dict(k=2, external_fields=[('phi', 1), ('phi', 1)],
              fundamental={'mu': 1, 'D': 1, 'lam': 0.1, 'T': 1},
              tau_max=0.0, tau_step=1.0, spatial_grid=np.linspace(0, 4, 7),
              use_cache=False, verbose=False, mf_dae_n_starts=2)
    # spatial MP is opt-in via spatial_parallel (default off); cap at 2 workers
    # (thread-pinned → 2 threads, safe in CI) to actually exercise the fork path.
    cs = np.asarray(compute_cumulants(ac, max_ell=1, spatial_parallel=False, **kw)['C_tau_x'])
    cp = np.asarray(compute_cumulants(ac, max_ell=1, spatial_parallel=True,
                                      n_workers=2, **kw)['C_tau_x'])
    assert np.array_equal(cs, cp), \
        f'spatial parallel != serial: max|Δ|={np.max(np.abs(cp - cs)):.2e}'


def test_analytic_ift_vs_numerical_ft():
    """The analytic heat-kernel IFT (Case A, plain vertices — correlator_2pt_x)
    reproduces the numerical q→x FT (correlator_2pt sampled on a fine q-grid) in
    the q_cut→∞ limit, with NO q-grid.  Allen-Cahn φ⁴ 1-loop (φ³ Hartree)."""
    import numpy as np, math
    from sage.all import SR
    from api.model import ModelBuilder
    from api.compute import FieldTheory
    from api._propagator import build_propagator
    from engine.integration.spatial.diagram_descriptor import diagram_to_cstack
    from engine.integration.spatial.pipeline_bridge import (
        build_pipeline_records, _legs_to_phys_idx)
    from engine.integration.spatial.full_integrator import (
        correlator_2pt, correlator_2pt_x)
    from engine.diagrams.type_assignment import build_field_index_map

    ac = (ModelBuilder('ac', n_populations=0).physical_field('phi', spatial_dim=1)
          .parameter('mu', default=1.0, domain='positive')
          .parameter('D', default=1.0, domain='positive')
          .parameter('lam', default=0.1, domain='positive')
          .parameter('T', default=1.0, domain='positive')
          .equation(lhs='(Dt + mu - D*Laplacian)*phi', rhs='-lam*phi^3')
          .set_action_text('phit*((Dt + mu - D*Laplacian)*phi + lam*phi^3) - T*phit^2')
          .boundary('infinite').initial('stationary').build())
    ft = FieldTheory(ac, taylor_order=4); ft.expand()
    prop = build_propagator(ft, ac, use_cache=False, verbose=False)
    rvn = list(ft._ns._ring_var_names)
    _, pidx = build_field_index_map(rvn, ft._n_tilde)
    ext = _legs_to_phys_idx([('phi', 1), ('phi', 1)], pidx)
    base = {SR.var('mu'): 1.0, SR.var('D'): 1.0, SR.var('lam'): 0.1,
            SR.var('T'): 1.0, SR.var('phistar1'): 0.0}
    be = build_pipeline_records(ft, ac, prop, ext, max_ell=1, verbose=False)
    descrs = [(diagram_to_cstack(td), float(SR(p).subs(base)))
              for td, p in be.get(1, []) if abs(float(SR(p).subs(base))) > 1e-14]
    assert descrs, 'no live Allen-Cahn 1-loop diagram'
    mu = D = 1.0
    xs = np.array([0.0, 0.5, 1.0, 2.0])
    an = correlator_2pt_x(descrs, xs, 0.0, mu, D, spatial_dim=1)        # analytic
    qg = np.linspace(0.0, 60.0, 3000)                                  # numerical FT
    dCq = np.array([correlator_2pt(descrs, float(q), 0.0, mu, D, spatial_dim=1)
                    for q in qg])
    num = np.array([np.trapz(np.cos(qg * x) * np.real(dCq), qg) / math.pi for x in xs])
    for i, x in enumerate(xs):
        assert abs(an[i] - num[i]) <= 1e-4 * (abs(num[i]) + 1e-12), \
            f"analytic IFT {an[i]} vs numerical FT {num[i]} at x={x}"


@pytest.mark.parametrize('kind,action,xs,qcut,nq,nt,ns,tol', [
    # Case C — per-leg derivative (KPZ (∇φ)²): F = ℓ²q(ℓ−q); the partial
    # cancellation in P(q)=−⅛q⁴+… keeps a light tail → modest q-grid converges.
    ('kpz', 'ht*(Dt(h)+mu*h-D*Lap(h)-(c/2)*Dx(h,0)^2)-T*ht^2',
     (0.0, 0.5, 1.0, 2.0), 50.0, 1500, 14, 12, 2e-3),
    # Case B — composite derivative (Model B ∇²(φ²)): F = q²(ℓ−q)²; P(q)=+¼q⁴+…
    # (reinforcing tail) → the numerical FT needs q_cut≳80 to converge to the
    # analytic value (the analytic IFT has NO such truncation — it is exact).
    ('modelb', 'ht*(Dt(h)+mu*h-D*Lap(h)-c*Lap(h^2))-T*ht^2',
     (0.0, 1.0), 85.0, 2200, 14, 12, 6e-3),
])
def test_analytic_ift_derivative_vs_numerical_ft(kind, action, xs, qcut, nq, nt, ns, tol):
    """Phase 2 — the analytic heat-kernel IFT of a DERIVATIVE-vertex form factor
    (``diagram_correlator_x`` with ``formfactor``: joint ``(ℓ,q)`` Gaussian →
    polynomial-fit + closed-form heat-kernel q-moments) reproduces the numerical
    ``q→x`` FT of ``diagram_correlator`` (same form factor) in the q_cut→∞ limit,
    with NO q-grid.  KPZ ``(∇φ)²`` (Case C) and Model B ``∇²(φ²)`` (Case B), 1-loop.

    Both paths share the SAME causal-chamber time quadrature (only the q-handling
    differs: closed-form moments vs trapz), so a modest ``n_t/n_s`` keeps the
    comparison exact while staying fast — the time-quadrature error cancels."""
    import numpy as np, math
    from sage.all import SR
    from api.model import ModelBuilder
    from api.compute import FieldTheory
    from api._propagator import build_propagator
    from engine.integration.spatial.diagram_descriptor import diagram_to_cstack
    from engine.integration.spatial.pipeline_bridge import (
        build_pipeline_records, _legs_to_phys_idx, _formfactor_callable)
    from engine.integration.spatial.full_integrator import (
        diagram_correlator, diagram_correlator_x)
    from engine.diagrams.type_assignment import build_field_index_map

    tb = (ModelBuilder(kind, n_populations=0).physical_field('h', spatial_dim=1)
          .parameter('mu', default=1.0, domain='positive')
          .parameter('D', default=1.0, domain='positive')
          .parameter('c', default=0.3, domain='real')
          .parameter('T', default=1.0, domain='positive'))
    rhs = 'c*Laplacian*h^2' if kind == 'modelb' else '0'
    b = (tb.equation(lhs='(Dt+mu-D*Laplacian)*h', rhs=rhs)
         .set_action_text(action).operator_ir()
         .boundary('infinite').initial('stationary').build())
    ft = FieldTheory(b, taylor_order=4); ft.expand()
    prop = build_propagator(ft, b, use_cache=False, verbose=False)
    rvn = list(ft._ns._ring_var_names)
    _, pidx = build_field_index_map(rvn, ft._n_tilde)
    ext = _legs_to_phys_idx([('h', 1), ('h', 1)], pidx)
    base = {SR.var('mu'): 1.0, SR.var('D'): 1.0, SR.var('c'): 0.3,
            SR.var('T'): 1.0, SR.var('hstar1'): 0.0}
    vt = [{'weight': float(SR(t['weight']).subs(base)), 'n_phys': t['n_phys'],
           'chain': t['chain'], 'mode': t['mode']}
          for t in ft._ns._operator_ir_vertex_terms]
    be = build_pipeline_records(ft, b, prop, ext, max_ell=1, verbose=False)
    diags = [(diagram_to_cstack(td), float(SR(p).subs(base)),
              _formfactor_callable(td, vt, d=1))
             for td, p in be.get(1, []) if abs(float(SR(p).subs(base))) > 1e-14]
    assert diags, f'no live {kind} 1-loop diagram'
    mu = D = 1.0
    xs_a = np.array(xs)
    an = np.zeros(len(xs_a), dtype=complex)
    for dd, pv, ff in diags:                                  # analytic (no q-grid)
        an += diagram_correlator_x(dd, pv, xs_a, 0.0, mu, D, spatial_dim=1,
                                   n_t=nt, n_s=ns, formfactor=ff)
    qg = np.linspace(0.0, qcut, nq)                           # numerical FT reference
    num = np.zeros(len(xs_a))
    for dd, pv, ff in diags:
        dCq = np.array([diagram_correlator(dd, pv, float(q), 0.0, mu, D,
                                           spatial_dim=1, n_t=nt, n_s=ns,
                                           formfactor=ff) for q in qg])
        num += np.array([np.trapz(np.cos(qg * x) * np.real(dCq), qg) / math.pi
                         for x in xs_a])
    for i, x in enumerate(xs_a):
        assert abs(np.real(an[i]) - num[i]) <= tol * (abs(num[i]) + 1e-12), \
            f"{kind} analytic IFT {np.real(an[i]):.7f} vs numerical FT " \
            f"{num[i]:.7f} at x={x} (rel {abs(np.real(an[i])-num[i])/(abs(num[i])+1e-30):.2e})"


def test_mc_integrator_matches_grid_plain():
    """The Monte-Carlo backend (method='mc') reproduces the deterministic grid for
    the PLAIN chamber/Schwinger integral (formfactor=None) — importance-sampled,
    O(1/√N), memory-bounded.  (Validated for plain vertices; derivative-vertex
    form-factor moments are biased by the det M→0 singularity and intentionally
    not asserted — see docs/spatial_loop_integral_analytic_mc.md.)"""
    import numpy as np
    from sage.all import SR
    from api.model import ModelBuilder
    from api.compute import FieldTheory
    from api._propagator import build_propagator
    from engine.integration.spatial.diagram_descriptor import diagram_to_cstack
    from engine.integration.spatial.pipeline_bridge import (
        build_pipeline_records, _legs_to_phys_idx)
    from engine.integration.spatial.full_integrator import diagram_kinematic
    from engine.diagrams.type_assignment import build_field_index_map

    tb = (ModelBuilder('kpz', n_populations=0).physical_field('h', spatial_dim=1)
          .parameter('mu', default=1.0, domain='positive')
          .parameter('D', default=1.0, domain='positive')
          .parameter('c', default=0.3, domain='real')
          .parameter('T', default=1.0, domain='positive'))
    b = (tb.equation(lhs='(Dt+mu-D*Laplacian)*h', rhs='0')
         .set_action_text('ht*(Dt(h)+mu*h-D*Lap(h)-(c/2)*Dx(h,0)^2)-T*ht^2')
         .operator_ir().boundary('infinite').initial('stationary').build())
    ft = FieldTheory(b, taylor_order=4); ft.expand()
    prop = build_propagator(ft, b, use_cache=False, verbose=False)
    rvn = list(ft._ns._ring_var_names)
    _, pidx = build_field_index_map(rvn, ft._n_tilde)
    ext = _legs_to_phys_idx([('h', 1), ('h', 1)], pidx)
    base = {SR.var('mu'): 1.0, SR.var('D'): 1.0, SR.var('c'): 0.3,
            SR.var('T'): 1.0, SR.var('hstar1'): 0.0}
    be = build_pipeline_records(ft, b, prop, ext, max_ell=1, verbose=False)
    raw = [td for td, p in be.get(1, []) if abs(float(SR(p).subs(base))) > 1e-14]
    assert raw, 'no live 1-loop diagram'
    dd = diagram_to_cstack(raw[0])
    et = {0: 0.0, 1: 0.0}
    grid = diagram_kinematic(dd, [0.7], et, 1.0, 1.0, spatial_dim=1,
                             n_t=24, n_s=26, formfactor=None)
    # default method must be the (untouched) grid path, bit-for-bit
    assert diagram_kinematic(dd, [0.7], et, 1.0, 1.0, spatial_dim=1, n_t=24,
                             n_s=26, formfactor=None, method='grid') == grid
    mc = diagram_kinematic(dd, [0.7], et, 1.0, 1.0, spatial_dim=1,
                           method='mc', mc_n=2_000_000, mc_seed=0, formfactor=None)
    assert grid != 0.0
    rel = abs(mc - grid) / abs(grid)
    assert rel < 0.02, f'MC {mc:.6e} vs grid {grid:.6e} (rel {rel:.2e})'


def test_bessel_integrator_matches_grid():
    """The Bessel-K backend (method='bessel') — radial λ analytically (a modified
    Bessel function), only the angular simplex sampled — reproduces the grid for the
    analytic-IFT δC(x) at x>0, for BOTH plain vertices (≈exact) AND derivative
    vertices (KPZ, where pure MC is biased by the det M→0 singularity).  x>0 only:
    x=0 equal-point is UV-sensitive (see docs/spatial_loop_integral_analytic_mc.md §3)."""
    import numpy as np
    from sage.all import SR
    from api.model import ModelBuilder
    from api.compute import FieldTheory
    from api._propagator import build_propagator
    from engine.integration.spatial.diagram_descriptor import diagram_to_cstack
    from engine.integration.spatial.pipeline_bridge import (
        build_pipeline_records, _legs_to_phys_idx, _formfactor_callable)
    from engine.integration.spatial.full_integrator import diagram_kinematic
    from engine.diagrams.type_assignment import build_field_index_map

    tb = (ModelBuilder('kpz', n_populations=0).physical_field('h', spatial_dim=1)
          .parameter('mu', default=1.0, domain='positive')
          .parameter('D', default=1.0, domain='positive')
          .parameter('c', default=0.3, domain='real')
          .parameter('T', default=1.0, domain='positive'))
    b = (tb.equation(lhs='(Dt+mu-D*Laplacian)*h', rhs='0')
         .set_action_text('ht*(Dt(h)+mu*h-D*Lap(h)-(c/2)*Dx(h,0)^2)-T*ht^2')
         .operator_ir().boundary('infinite').initial('stationary').build())
    ft = FieldTheory(b, taylor_order=4); ft.expand()
    prop = build_propagator(ft, b, use_cache=False, verbose=False)
    rvn = list(ft._ns._ring_var_names)
    _, pidx = build_field_index_map(rvn, ft._n_tilde)
    ext = _legs_to_phys_idx([('h', 1), ('h', 1)], pidx)
    base = {SR.var('mu'): 1.0, SR.var('D'): 1.0, SR.var('c'): 0.3,
            SR.var('T'): 1.0, SR.var('hstar1'): 0.0}
    vt = [{'weight': float(SR(t['weight']).subs(base)), 'n_phys': t['n_phys'],
           'chain': t['chain'], 'mode': t['mode']}
          for t in ft._ns._operator_ir_vertex_terms]
    be = build_pipeline_records(ft, b, prop, ext, max_ell=1, verbose=False)
    raw = [td for td, p in be.get(1, []) if abs(float(SR(p).subs(base))) > 1e-14]
    assert raw, 'no live 1-loop diagram'
    dd = diagram_to_cstack(raw[0]); ff = _formfactor_callable(raw[0], vt, d=1)
    assert getattr(ff, 'moment_bessel', None) is not None, 'no moment_bessel built'
    et = {0: 0.0, 1: 0.0}
    xs = np.array([2.0])                                 # one clean x>0 point
    # PLAIN: the radial reduction is a single Bessel-K → ≈exact
    gp = diagram_kinematic(dd, [0.0], et, 1.0, 1.0, spatial_dim=1, n_t=24, n_s=26,
                           xs=xs, formfactor=None)[0]
    bp = diagram_kinematic(dd, [0.0], et, 1.0, 1.0, spatial_dim=1, xs=xs,
                           formfactor=None, method='bessel', mc_n=2_000_000, mc_seed=0)[0]
    assert gp != 0.0
    assert abs(bp - gp) / abs(gp) < 0.01, f'plain bessel {bp:.6e} vs grid {gp:.6e}'
    # DERIVATIVE (KPZ): radial sum of Bessel-K's via the λ-graded moment
    gd = diagram_kinematic(dd, [0.0], et, 1.0, 1.0, spatial_dim=1, n_t=24, n_s=26,
                           xs=xs, formfactor=ff)[0]
    bd = diagram_kinematic(dd, [0.0], et, 1.0, 1.0, spatial_dim=1, xs=xs,
                           formfactor=ff, method='bessel', mc_n=4_000_000, mc_seed=0)[0]
    assert gd != 0.0
    assert abs(bd - gd) / abs(gd) < 0.06, f'KPZ bessel {bd:.6e} vs grid {gd:.6e}'
    # d-robustness: the radial Bessel-K + isotropic heat kernel work at d≥2 (plain)
    for dim in (2, 3):
        gpd = diagram_kinematic(dd, [0.0], et, 1.0, 1.0, spatial_dim=dim, n_t=24,
                                n_s=26, xs=xs, formfactor=None)[0]
        bpd = diagram_kinematic(dd, [0.0], et, 1.0, 1.0, spatial_dim=dim, xs=xs,
                                formfactor=None, method='bessel', mc_n=3_000_000, mc_seed=0)[0]
        assert abs(bpd - gpd) / abs(gpd) < 0.10, f'd={dim} plain bessel {bpd:.3e} vs {gpd:.3e}'
    # the backend must FAIL CLEANLY for a derivative vertex with no λ-graded moment
    # (e.g. d≥2 derivative, ff.moment_bessel is None) — never silently use plain K
    import pytest

    class _FFNoMoment:
        pass
    with pytest.raises(NotImplementedError):
        diagram_kinematic(dd, [0.0], et, 1.0, 1.0, spatial_dim=2, xs=xs,
                          formfactor=_FFNoMoment(), method='bessel', mc_n=10_000)
