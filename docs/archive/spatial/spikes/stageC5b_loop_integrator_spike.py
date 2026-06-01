"""
Stage C.5b spike — the per-edge momentum loop integrator END-TO-END.

STATUS (2026-05-29): the override path is CORRECT in structure (the C.5a spike
already validated the ∫dℓ to machine precision), and both seams are landed
(integrate_diagram + compute_correction_td accept the per-edge builder, time-
only path byte-unchanged).  This end-to-end spike is a PERFORMANCE WIP: it
re-runs the full integrate_diagram per ℓ-quadrature-node, but the time-polytope
(δ-subset sum + polygon/poset evaluation) is ℓ-INDEPENDENT, so doing it afresh
per node is too slow (the 1-loop integrate_diagram is seconds-scale on its own).
The fix (C.5b cont.): build the time-polytope structure ONCE and re-evaluate
only the per-edge (C_α, λ_α) per ℓ-node — i.e. cache the per-subset evaluator
and feed it node-dependent edge modes — or go momentum-first (Gaussian ∫dℓ at
fixed internal times).  See docs/spatial_stageC5_general_integrator_design.md.

Proves the override path: route each ell=1 diagram, build per-edge EdgeModeSums
at the routed k_e^2, run the SHARED Phase-J time integrator
(compute_correction_td with edge_mode_sums_builder_fn), and do the loop
integral ∫dℓ/2π by adaptive quad.

Validation (Allen-Cahn φ³ tadpole, μ=D=T=1, λ=0.1): the loop integrator must
reproduce the Stage-C tadpole δC(q,τ) = Σ·∂C₀(q,τ)/∂μ, Σ=3λ⟨φ²⟩₀=0.15.
At (q=0.8, τ=0.5):  m=1.64, ∂C₀/∂μ=-e^{-mτ}(1/m²+τ/m)=-0.2980 → δC=-0.04470.

This is the SAME machinery the bubble (φ̃φ²) needs; the tadpole is the
known-answer check before the bubble (whose loop momentum does not decouple).
"""
import importlib.util, os, sys, math
sys.path.insert(0, '.')
import numpy as np
import sympy as sp
from scipy import integrate
from sage.all import SR

from msrjd.core.field_theory import FieldTheory
from pipeline._propagator import build_propagator, compute_poles_and_residues
from msrjd.integration.time_domain.pipeline import compute_correction_td
from msrjd.integration.time_domain.final_integral import _build_edge_mode_sums
from msrjd.integration.spatial.pipeline_bridge import build_pipeline_records
from msrjd.integration.spatial.momentum_routing import route_momenta

Lap = SR.var('Laplacian')


def load(p):
    s = importlib.util.spec_from_file_location('m', p)
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m.build()


mu = D = T = 1.0
lam = 0.1
model = load('theories/allen_cahn_1d_subcritical_infinite.theory.py')
ft = FieldTheory(model, taylor_order=4); ft.expand()
prop = build_propagator(ft, model, use_cache=False, verbose=False)
base_np = {SR.var('mu'): mu, SR.var('D'): D, SR.var('T'): T,
           SR.var('lam'): lam, SR.var('phistar1'): 0.0}
ext = [('dphi', 1), ('dphi', 1)]

by_ell = build_pipeline_records(ft, model, prop, ext, max_ell=1)
ell1 = by_ell.get(1, [])
# Keep only diagrams with a NON-zero prefactor at the saddle (the φ*²-vanishing
# diagrams contribute 0; dropping them is a 4× speedup).
tds, prefs = [], []
for td, pf in ell1:
    if abs(complex(SR(pf).subs(base_np))) > 1e-14:
        tds.append(td); prefs.append(pf)
print(f'ell=1 diagrams: {len(ell1)} enumerated, {len(tds)} non-zero at saddle',
      flush=True)

# Precompute per-diagram routing (edge_key -> k^2 expr in q0, l0)
routings = [route_momenta(td) for td in tds]


def make_builder_fn(q_val, ell_val):
    """Return td -> builder(edge_info, propagator_data) at this (q, ℓ) node."""
    td_to_routing = {id(td): r for td, r in zip(tds, routings)}

    def builder_fn(td):
        r = td_to_routing[id(td)]
        subs = {r.q_syms[0]: q_val}
        if r.loop_syms:
            subs[r.loop_syms[0]] = ell_val
        k2 = {e: float(sp.Float(sp.expand(v).subs(subs))) for e, v in r.edge_k2().items()}

        def builder(edge_info, propagator_data):
            out = []
            for ei in edge_info:
                e = (ei['u'], ei['v'], ei['lbl'])
                nps = dict(base_np); nps[Lap] = -k2[e]
                compute_poles_and_residues(prop, nps, verbose=False)
                pd = {'pole_vals': list(prop['pole_vals']),
                      'C_mats': list(prop['C_mats'])}
                out.append(_build_edge_mode_sums([ei], pd)[0])
            return out
        return builder
    return builder_fn


# reference propagator_data (valid; its modes are overridden per edge)
compute_poles_and_residues(prop, {**base_np, Lap: 0.0}, verbose=False)
pdata = {kk: prop[kk] for kk in ('K_ker', 'K_ft', 'G_ft', 'adj_ft', 'D_omega',
                                 'D_delta', 't_var', 'omega', 'nf',
                                 'pole_vals', 'C_mats')}


def integrand_at(q_val, ell_val, tau):
    """total_C over the ell=1 diagrams at loop momentum ℓ_val (per-edge modes)."""
    bf = make_builder_fn(q_val, ell_val)
    res = compute_correction_td(
        typed_diagrams=tds, prefactors=prefs, k=2, propagator_data=pdata,
        external_fields=ext, num_params={**base_np, Lap: -(q_val**2)},
        origin_leaf_idx=0, edge_mode_sums_builder_fn=bf)
    return complex(res['total_C'](0.0, tau)).real


def loop_correction(q_val, tau, lim=14.0, n=48):
    """δC(q,τ) = ∫dℓ/2π total_C(q,ℓ,τ)  via fixed Gauss–Legendre on [-lim,lim]."""
    x, w = np.polynomial.legendre.leggauss(n)
    ells, wl = lim * x, lim * w
    s = 0.0
    for i in range(n):
        s += wl[i] * integrand_at(q_val, ells[i], tau)
    return s / (2 * np.pi)


q, tau = 0.8, 0.5
m = mu + D * q * q
dC0_dmu = -math.exp(-m * tau) * (1.0 / m**2 + tau / m)
target = 3 * lam * (T / (2 * math.sqrt(mu * D))) * dC0_dmu
print(f'\n(q={q}, τ={tau}):  m={m}, ∂C₀/∂μ={dC0_dmu:.5f}')
print(f'  target δC (Σ·∂C₀/∂μ, Σ=3λ⟨φ²⟩₀=0.15) = {target:.6f}')
got = loop_correction(q, tau)
print(f'  loop integrator ∫dℓ/2π            = {got:.6f}')
print(f'  rel err = {abs(got - target)/abs(target):.2e}')
print('\n(match → the override + ∫dℓ machinery reproduces the tadpole through the '
      'SHARED\n Phase-J integrator; the bubble is next.)')
