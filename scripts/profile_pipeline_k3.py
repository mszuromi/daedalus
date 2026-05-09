"""
profile_pipeline_k3.py
======================
Profile the diagram enumeration pipeline for the quadratic Hawkes +
exponential synaptic filter model at k=3 external legs, ell<=1 loop order.

Uses time.perf_counter() to wall-clock each pipeline stage from a cold
start (no caches). Run via:

    sage -python profile_pipeline_k3.py
"""
from __future__ import annotations

import os
import sys
import time

# Resolve the project root so `msrjd.*` and `models.*` imports work when
# running the script from any cwd.
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

# ── Sage / SR imports ──
from sage.all import (
    SR, matrix, I, dirac_delta, diff, limit as sage_limit, oo as sage_oo,
)

# ── Library imports ──
from msrjd.core.field_theory import FieldTheory, fourier_transform
from msrjd.core.vertices import (
    extract_vertex_types, extract_source_types, available_degrees,
)
from msrjd.enumeration.loop_diagram_enumeration import enumerate_all as enumerate_prediagrams_all
from msrjd.enumeration.degree_scan import max_vertex_degree, scan_source_vertices
from msrjd.diagrams.filter import filter_prediagrams
from msrjd.diagrams.type_assignment import (
    enumerate_all as enumerate_all_typed,
    build_field_index_map,
)
from msrjd.diagrams.causality import filter_causal
from msrjd.diagrams.symmetry import (
    deduplicate_typed_diagrams, classify_coefficient_factors,
)

from models.hawkes_quad_expg import HAWKES_MODEL


# ── User workload configuration (exact match to request) ──
K                = 3
MAX_ELL          = 1
EXTERNAL_FIELDS  = [('dn', 1), ('dn', 2), ('dv', 2)]
TAYLOR_ORDER     = 4
NUM_PARAMS       = {
    'a':     0.44,
    'tau':   10.0,
    'tau_g': 5.0,
    'E':     [0.8, 0.8],
    'w':     [[0.3, 0.3], [0.3, 0.3]],
}


# ── Timing bookkeeping ──
stages: list[tuple[str, float, object]] = []  # (name, elapsed_s, output_size)


def _tick(name: str):
    """Start timing a stage; returns a (stop, tag) pair."""
    t0 = time.perf_counter()

    def _stop(size):
        dt = time.perf_counter() - t0
        stages.append((name, dt, size))
        print(f'  [{name}] {dt:8.3f} s  (output size: {size})')
        return dt

    print(f'--> {name} ...', flush=True)
    return _stop


# =====================================================================
# 1. FieldTheory construction + expand (cell 6 equivalent -- "1.1 expand")
# =====================================================================
stop = _tick('FieldTheory + expand')
ft = FieldTheory(HAWKES_MODEL, taylor_order=TAYLOR_ORDER)
ft.expand()
R  = ft.ring()
ns = ft._ns
stop(f'ring={R.ngens()} gens')


# =====================================================================
# 2. Propagator build (cell 8).  Replicates the exact construction: K
#    from free_action, kernel wrapping, FT, frequency-space image
#    substitution, inverse, adjugate, det, D_delta limits.
# =====================================================================
stop = _tick('Propagator build (K_ft/G_ft/adj_ft/D_omega/D_delta)')

S_free = ft.free_action()
ring_gen_names = [str(g) for g in R.gens()]
resp_names = ring_gen_names[:ft._n_tilde]
phys_names = ring_gen_names[ft._n_tilde:]
pos_to_row = {ring_gen_names.index(nm): i for i, nm in enumerate(resp_names)}
pos_to_col = {ring_gen_names.index(nm): j for j, nm in enumerate(phys_names)}

nf = len(resp_names)
K_data = [[SR(0)] * nf for _ in range(nf)]
for exp_vec, coeff in S_free.dict().items():
    row = col = None
    for idx in range(len(ring_gen_names)):
        if exp_vec[idx] > 0:
            if idx in pos_to_row: row = pos_to_row[idx]
            if idx in pos_to_col: col = pos_to_col[idx]
    if row is not None and col is not None:
        K_data[row][col] += SR(coeff)
K_mat = matrix(SR, K_data)

Dt       = ns.Dt
delta_D  = ns.delta_D
delta_Dp = ns.delta_Dp

def _to_kernel(c):
    c = SR(c)
    if c.has(delta_D):
        return c
    p0   = c.subs({Dt: 0})
    rest = (c - p0).subs({Dt: delta_Dp})
    return p0 * delta_D + rest

K_ker = matrix(SR, [[_to_kernel(K_mat[i, j]) for j in range(nf)]
                    for i in range(nf)])

t_var = SR.var('t')
omega = SR.var('omega', latex_name=r'\omega')
time_domain = {
    delta_D:  dirac_delta(t_var),
    delta_Dp: diff(dirac_delta(t_var), t_var),
}

K_ft_data = [[SR(0)] * nf for _ in range(nf)]
for i in range(nf):
    for j in range(nf):
        c = K_ker[i, j]
        if not c.is_zero():
            K_ft_data[i][j] = fourier_transform(
                SR(c).subs(time_domain), t_var, omega
            )
K_ft = matrix(SR, K_ft_data)

_kft_hook = HAWKES_MODEL.get('kernel_ft_image')
if _kft_hook is not None:
    _kft_subs = _kft_hook(ns, omega)
    K_ft = K_ft.apply_map(lambda e: SR(e).subs(_kft_subs))

G_ft   = K_ft.inverse().apply_map(lambda e: e.factor())
adj_ft = K_ft.adjugate()
D_omega = K_ft.det()

D_delta_data = [[SR(0)] * nf for _ in range(nf)]
for i in range(nf):
    for j in range(nf):
        entry = SR(G_ft[i, j])
        if entry.is_zero():
            continue
        try:
            lim_val = sage_limit(entry, **{str(omega): sage_oo})
            if not lim_val.is_zero():
                D_delta_data[i][j] = lim_val
        except Exception:
            pass
D_delta = matrix(SR, D_delta_data)
stop(f'nf={nf}, nonzero G_ft entries='
     f'{sum(1 for i in range(nf) for j in range(nf) if not G_ft[i,j].is_zero())}')


# =====================================================================
# 3. Pre-diagram enumeration (cell 11) -- needed before degree_scan
# =====================================================================
pds_by_ell = {}
counts_by_ell = {}
stop = _tick(f'Pre-diagram enumeration (k={K}, all ell<= {MAX_ELL})')
total_pd = 0
for ell in range(MAX_ELL + 1):
    _, _, pds, counts = enumerate_prediagrams_all(K, ell, verbose=False)
    pds_by_ell[ell] = pds
    counts_by_ell[ell] = counts
    total_pd += counts['n_prediagrams']
stop(f'ell=0: {counts_by_ell[0]["n_prediagrams"]} pds, '
     f'ell=1: {counts_by_ell[1]["n_prediagrams"]} pds '
     f'(total {total_pd})')


# =====================================================================
# 4. degree_scan (cell 13) -- max degree + source out-degrees.  Then
#    extract vertex types / source types.
# =====================================================================
stop = _tick('degree_scan (max_vertex_degree + scan_source_vertices)')
all_pds = [pd for pds in pds_by_ell.values() for pd in pds]
max_deg = max_vertex_degree(all_pds)
src_degs = scan_source_vertices(all_pds)
stop(f'max_deg={max_deg}, src_degs={src_degs}')

stop = _tick('Vertex type extraction (extract_vertex_types/source_types)')
vtypes = extract_vertex_types(ft)
stypes = extract_source_types(ft)
int_degs, src_degs_avail = available_degrees(vtypes, stypes)
stop(f'vtypes={len(vtypes)}, stypes={len(stypes)}')


# =====================================================================
# 5. filter_prediagrams (cell 15)
# =====================================================================
kept_by_ell = {}
stop = _tick('filter_prediagrams')
total_kept = 0
for ell in range(MAX_ELL + 1):
    kept, disc = filter_prediagrams(pds_by_ell[ell], vtypes, stypes)
    kept_by_ell[ell] = kept
    total_kept += len(kept)
stop(f'ell=0 kept={len(kept_by_ell[0])}, ell=1 kept={len(kept_by_ell[1])} '
     f'(total {total_kept})')


# =====================================================================
# 6. Build field index maps (cell 17 prep) -- near-instant but part of
#    the typed-enumeration setup.
# =====================================================================
stop = _tick('build_field_index_map')
ring_var_names_list = list(ns._ring_var_names)
n_tilde = ft._n_tilde
resp_idx, phys_idx = build_field_index_map(ring_var_names_list, n_tilde)
stop(f'resp_idx={len(resp_idx)}, phys_idx={len(phys_idx)}')


# =====================================================================
# 7. Typed-diagram enumeration (cell 18) -- enumerate_all_typed.
#    Do ell=0 and ell=1 separately so we can report the 2080 number
#    (which is ell=1 only; ell=0 contributes the other 2).
# =====================================================================
typed_by_ell = {}
stop = _tick('enumerate_all_typed (all ell)')
total_typed = 0
for ell in range(MAX_ELL + 1):
    typed = enumerate_all_typed(
        kept_by_ell[ell], EXTERNAL_FIELDS, vtypes, stypes,
        G_ft, resp_idx, phys_idx,
    )
    typed_by_ell[ell] = typed
    total_typed += len(typed)
stop(f'ell=0: {len(typed_by_ell[0])}, ell=1: {len(typed_by_ell[1])} '
     f'(total {total_typed})')


# =====================================================================
# 8. Causal filter (cell 18 continued) -- separate timing.
# =====================================================================
causal_by_ell = {}
stop = _tick('filter_causal')
total_causal = 0
for ell in range(MAX_ELL + 1):
    causal, n_disc, _ = filter_causal(typed_by_ell[ell])
    causal_by_ell[ell] = causal
    total_causal += len(causal)
stop(f'ell=0: {len(causal_by_ell[0])}, ell=1: {len(causal_by_ell[1])} '
     f'(total {total_causal}, n_discarded ell=1={MAX_ELL and n_disc})')


# =====================================================================
# 9. Deduplication (cell 18 continued).
# =====================================================================
unique_by_ell = {}
stop = _tick('deduplicate_typed_diagrams')
total_unique = 0
for ell in range(MAX_ELL + 1):
    unique = deduplicate_typed_diagrams(causal_by_ell[ell])
    unique_by_ell[ell] = unique
    total_unique += len(unique)
stop(f'ell=0: {len(unique_by_ell[0])}, ell=1: {len(unique_by_ell[1])} '
     f'(total {total_unique})')


# =====================================================================
# 10. Coefficient classification (cell 20/22) -- per-diagram prefactor.
# =====================================================================
time_dep_params = HAWKES_MODEL.get('time_dependent_parameters', [])
noise_structure = HAWKES_MODEL.get('noise_structure')

all_unique = [td for ell in range(MAX_ELL + 1) for td in unique_by_ell[ell]]

stop = _tick('classify_coefficient_factors (all unique)')
prefactors = []
for td in all_unique:
    coeff_info = classify_coefficient_factors(
        td,
        time_dep_params=time_dep_params,
        noise_structure=noise_structure,
    )
    prefactors.append(coeff_info['scalar_prefactor'])
stop(f'{len(prefactors)} prefactors')


# =====================================================================
# Summary table
# =====================================================================
print()
print('=' * 78)
print('PIPELINE PROFILE  --  k=3, ell<=1, quadratic Hawkes + exp filter, taylor=4')
print('=' * 78)
total = sum(dt for _, dt, _ in stages)
print(f'{"Stage":<50s}  {"Time (s)":>10s}  {"Cum (s)":>10s}  Output')
print('-' * 95)
cum = 0.0
for name, dt, size in stages:
    cum += dt
    print(f'{name:<50s}  {dt:>10.3f}  {cum:>10.3f}  {size}')
print('-' * 95)
print(f'{"TOTAL":<50s}  {total:>10.3f}  {total:>10.3f}')
print()

# Top-2 bottlenecks
ranked = sorted(stages, key=lambda s: s[1], reverse=True)
print('TOP BOTTLENECKS')
for name, dt, _ in ranked[:3]:
    print(f'  {name:<50s}  {dt:>8.3f} s  ({100*dt/total:5.1f}%)')
print()

# Enumerate-typed cost per output diagram
typed_ell1 = len(typed_by_ell.get(1, []))
enum_time = next(dt for name, dt, _ in stages if name.startswith('enumerate_all_typed'))
print(f'enumerate_all_typed cost-per-diagram: '
      f'{enum_time*1000:.1f} ms / {total_typed} diagrams = '
      f'{(enum_time/max(1,total_typed))*1e6:.1f} us each '
      f'(ell=1 alone: {typed_ell1} diagrams -> '
      f'{(enum_time/max(1,typed_ell1))*1e6:.1f} us each)')
