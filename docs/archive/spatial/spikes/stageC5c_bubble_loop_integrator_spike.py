"""
Stage C.5c spike — the cached, physical-range BUBBLE loop integrator.

Three stages:
  (A) PLUMBING check (no oracle): the per-edge override with EVERY edge forced
      to the same momentum q must reproduce the standard global-propagator path
      EXACTLY.  This validates the override mechanism + the k²-cache in
      isolation from any physics.
  (B) per-edge ∫dℓ on the φ̃φ² bubble diagrams, cached propagators, physical
      (Gaussian-damped, τ>0) range — check the integrand decays and the loop
      integral converges in (range, n_nodes), and stays fast (no large-pole
      slow path).
  (C) [next] compare to an independent (ω,k) oracle.

Theory: reaction_diffusion_quadratic_1d  (∂_tφ=(D∂²−μ)φ−gφ²+ξ).
"""
import importlib.util, os, sys, time, math
sys.path.insert(0, '.')
import numpy as np
import sympy as sp
from sage.all import SR

from msrjd.core.field_theory import FieldTheory
from pipeline._propagator import build_propagator, compute_poles_and_residues
from msrjd.integration.time_domain.pipeline import compute_correction_td
from msrjd.integration.time_domain.final_integral import _build_edge_mode_sums
from msrjd.integration.spatial.pipeline_bridge import build_pipeline_records
from msrjd.integration.spatial.momentum_routing import route_momenta

Lap = SR.var('Laplacian')
PDATA_KEYS = ('K_ker', 'K_ft', 'G_ft', 'adj_ft', 'D_omega', 'D_delta',
              't_var', 'omega', 'nf', 'pole_vals', 'C_mats')


def load(p):
    s = importlib.util.spec_from_file_location('m', p)
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m.build()


mu = D = T = 1.0
g = 0.3
model = load('theories/reaction_diffusion_quadratic_1d.theory.py')
ft = FieldTheory(model, taylor_order=3); ft.expand()
prop = build_propagator(ft, model, use_cache=False, verbose=False)
base_np = {SR.var('mu'): mu, SR.var('D'): D, SR.var('T'): T,
           SR.var('g'): g, SR.var('phistar1'): 0.0}
ext = [('dphi', 1), ('dphi', 1)]

by = build_pipeline_records(ft, model, prop, ext, max_ell=1)
ell1 = [(td, pf) for td, pf in by[1]
        if abs(complex(SR(pf).subs(base_np))) > 1e-14]
tds = [x[0] for x in ell1]
prefs = [x[1] for x in ell1]
print(f'ell=1 diagrams: {len(by[1])} enumerated, {len(tds)} non-zero', flush=True)

# ── k²-cached propagator snapshots ────────────────────────────────
_k2_cache = {}


def pd_at(k2):
    key = round(float(k2), 10)
    if key not in _k2_cache:
        compute_poles_and_residues(prop, {**base_np, Lap: -key}, verbose=False)
        _k2_cache[key] = {'pole_vals': list(prop['pole_vals']),
                          'C_mats': list(prop['C_mats'])}
    return _k2_cache[key]


def uniform_builder_fn(k2val):
    """Override: force EVERY edge to momentum² = k2val."""
    def bf(td):
        def builder(ei_list, pd0):
            return [_build_edge_mode_sums([ei], pd_at(k2val))[0] for ei in ei_list]
        return builder
    return bf


def routed_builder_fn(q, ell):
    """Override: each edge at its routed k_e²(q, ℓ)."""
    def bf(td):
        r = route_momenta(td)
        subs = {r.q_syms[0]: q}
        if r.loop_syms:
            subs[r.loop_syms[0]] = ell
        k2 = {e: float(sp.Float(sp.expand(v).subs(subs)))
              for e, v in r.edge_k2().items()}

        def builder(ei_list, pd0):
            return [_build_edge_mode_sums([ei], pd_at(k2[(ei['u'], ei['v'], ei['lbl'])]))[0]
                    for ei in ei_list]
        return builder
    return bf


def run(num_params, builder_fn=None, pdata=None):
    res = compute_correction_td(
        typed_diagrams=tds, prefactors=prefs, k=2, propagator_data=pdata,
        external_fields=ext, num_params=num_params, origin_leaf_idx=0,
        edge_mode_sums_builder_fn=builder_fn)
    return res['total_C']


# ── (A) PLUMBING check ────────────────────────────────────────────
q = 0.8
compute_poles_and_residues(prop, {**base_np, Lap: -(q*q)}, verbose=False)
pdata_q = {kk: prop[kk] for kk in PDATA_KEYS}
np_q = {**base_np, Lap: -(q*q)}
C_global = run(np_q, builder_fn=None, pdata=pdata_q)            # standard path
C_overr = run(np_q, builder_fn=uniform_builder_fn(q*q), pdata=pdata_q)  # override, uniform q
print('\n=== (A) plumbing: override(all edges @ q²) vs global path ===')
ok = True
for tau in [0.2, 0.5, 1.0]:
    a = complex(C_global(0.0, tau)); b = complex(C_overr(0.0, tau))
    d = abs(a - b)
    ok &= (d < 1e-10)
    print(f'  τ={tau}: global={a.real:.8f}  override={b.real:.8f}  |Δ|={d:.1e}')
print(f'  PLUMBING: {"PASS" if ok else "FAIL"}')

# ── (B) per-edge ∫dℓ on the bubble (physical range) ───────────────
print('\n=== (B) bubble loop integral ∫dℓ/2π (cached, physical range) ===')


def integrand(q, ell, tau):
    C = run({**base_np, Lap: -(q*q)}, builder_fn=routed_builder_fn(q, ell),
            pdata=pdata_q)
    return complex(C(0.0, tau)).real


# tail check: integrand vs ℓ at τ=0.5
print('  integrand(q=0.8, τ=0.5) vs ℓ (tail behaviour):')
for ell in [0.0, 1.0, 2.0, 3.0, 4.0, 6.0]:
    t0 = time.perf_counter()
    v = integrand(q, ell, 0.5)
    print(f'    ℓ={ell:>4}: {v:>14.8f}   ({time.perf_counter()-t0:.2f}s)', flush=True)


def loop_int(q, tau, lim, n):
    x, w = np.polynomial.legendre.leggauss(n)
    return sum(lim*w[i]*integrand(q, lim*x[i], tau) for i in range(n)) / (2*np.pi)


print('  convergence of ∫dℓ at (q=0.8, τ=0.5):')
for (lim, n) in [(5.0, 24), (6.0, 40), (8.0, 48)]:
    t0 = time.perf_counter()
    val = loop_int(q, 0.5, lim, n)
    print(f'    lim={lim} n={n}: δC={val:.8f}   ({time.perf_counter()-t0:.1f}s)',
          flush=True)
