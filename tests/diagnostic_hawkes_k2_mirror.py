"""
Diagnostic: reproduce the τ → -τ mirror bug reported for the REAL Hawkes
linear model at k=2 with external_fields = [('dn', 1), ('dv', 2)] and
origin_leaf_idx = 0.

Expected (simulation): ⟨δn_1(0) · δv_2(τ)⟩ is causally larger at +τ (δn_1
drives δv_2 forward in time through synaptic coupling w_{21}).

Reported (theory via compute_correction_td with origin_leaf_idx=0):
produces the mirror image (peak at -τ).

This script:
  1. Loads the LINEAR Hawkes model (so phi' = 1, phi'' = 0)
  2. Builds FieldTheory, propagator_data, typed diagrams
  3. Calls compute_correction_td for origin=0 and origin=1
  4. Prints total_C(0, τ) and total_C(τ, 0) at τ = ±1, ±3, ±10
  5. Prints per-diagram contributions so a flipped diagram is visible

Run with:
    sage -python tests/diagnostic_hawkes_k2_mirror.py
"""

import os
import sys
import numpy as np

# Add project root to sys.path
_THIS = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_THIS, '..'))
sys.path.insert(0, _ROOT)

from sage.all import (
    SR, matrix, I, exp, diff, solve, dirac_delta, function,
    latex,
)
from scipy.optimize import fsolve

# ────────────────────────────────────────────────────────────────
# Config
# ────────────────────────────────────────────────────────────────
k = 2
max_ell = 0
external_fields = [('dn', 1), ('dv', 2)]

fundamental = {
    'E':   [1.0, 1.0],
    'w':   [[0.4, 0.5], [0.5, 0.4]],
    'tau': 10.0,
}

TAU_VALS = [-10.0, -3.0, -1.0, 1.0, 3.0, 10.0]

# ────────────────────────────────────────────────────────────────
# 1. Load model and expand field theory
# ────────────────────────────────────────────────────────────────
from models.hawkes_linear_sage import HAWKES_MODEL
from msrjd.core.field_theory import FieldTheory, fourier_transform
from msrjd.core.vertices import extract_vertex_types, extract_source_types
from msrjd.enumeration.loop_diagram_enumeration import enumerate_all
from msrjd.enumeration.degree_scan import max_vertex_degree, scan_source_vertices
from msrjd.diagrams.filter import filter_prediagrams
from msrjd.diagrams.type_assignment import (
    enumerate_all as enumerate_all_typed,
    build_field_index_map,
)
from msrjd.diagrams.causality import filter_causal
from msrjd.diagrams.symmetry import (
    deduplicate_typed_diagrams,
    classify_coefficient_factors,
)
from msrjd.integration.time_domain import compute_correction_td

print('='*70)
print('DIAGNOSTIC: Hawkes linear k=2 mirror bug')
print(f"  external_fields = {external_fields}")
print(f"  params = E={fundamental['E']}, w={fundamental['w']}, tau={fundamental['tau']}")
print('='*70)

ft = FieldTheory(HAWKES_MODEL, taylor_order=4)
ft.expand()
R = ft.ring()
ns = ft._ns
nf_resp = ft._n_tilde

# ────────────────────────────────────────────────────────────────
# 2. Build kernel K, invert to G, extract poles/residues/D_delta
# ────────────────────────────────────────────────────────────────
print('\n[1/5] Building propagator K → G ...')
S_free = ft.free_action()
ring_gen_names = [str(g) for g in R.gens()]
resp_names = ring_gen_names[:nf_resp]
phys_names = ring_gen_names[nf_resp:]
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

Dt = ns.Dt
delta_D = ns.delta_D
delta_Dp = ns.delta_Dp

def _to_kernel(c):
    c = SR(c)
    if c.has(delta_D) or c.has(ns.g):
        return c
    p0 = c.subs({Dt: 0})
    rest = (c - p0).subs({Dt: delta_Dp})
    return p0 * delta_D + rest

K_ker = matrix(SR, [[_to_kernel(K_mat[i, j]) for j in range(nf)]
                    for i in range(nf)])

t_var = SR.var('t')
omega = SR.var('omega', latex_name=r'\omega')
time_domain = {
    delta_D:  dirac_delta(t_var),
    delta_Dp: diff(dirac_delta(t_var), t_var),
    ns.g:     function('g')(t_var),
}

K_ft_data = [[SR(0)] * nf for _ in range(nf)]
for i in range(nf):
    for j in range(nf):
        c = K_ker[i, j]
        if not c.is_zero():
            K_ft_data[i][j] = fourier_transform(SR(c).subs(time_domain), t_var, omega)
K_ft = matrix(SR, K_ft_data)
G_ft = K_ft.inverse().apply_map(lambda e: e.factor())

adj_ft = K_ft.adjugate()
D_omega = K_ft.det().expand()
D_prime = diff(D_omega, omega)
pole_eqs = solve(D_omega == 0, omega)
pole_vals = [eq.rhs() if hasattr(eq, 'rhs') else eq for eq in pole_eqs]

C_mats = []
for omega_k in pole_vals:
    C_data = [[SR(0)] * nf for _ in range(nf)]
    for i in range(nf):
        for j in range(nf):
            n_ij = adj_ft[i, j]
            if not n_ij.is_zero():
                num = n_ij.subs({omega: omega_k})
                den = D_prime.subs({omega: omega_k})
                C_data[i][j] = (I * num / den).factor()
    C_mats.append(matrix(SR, C_data))

from sage.all import limit as _limit, oo as _oo
D_delta_data = [[SR(0)] * nf for _ in range(nf)]
for i in range(nf):
    for j in range(nf):
        entry = SR(G_ft[i, j])
        if entry.is_zero():
            continue
        try:
            lim_val = _limit(entry, **{str(omega): _oo})
            if not lim_val.is_zero():
                D_delta_data[i][j] = lim_val
        except Exception:
            pass
D_delta = matrix(SR, D_delta_data)

propagator_data = {
    'pole_vals': pole_vals,
    'C_mats': C_mats,
    'D_delta': D_delta,
    'nf': nf,
    'G_ft': G_ft,
}

print(f'  nf={nf}, {len(pole_vals)} poles')
print(f'  D_delta nonzero entries: '
      f'{sum(1 for i in range(nf) for j in range(nf) if not D_delta[i,j].is_zero())}')

# ────────────────────────────────────────────────────────────────
# 3. Enumerate and type diagrams
# ────────────────────────────────────────────────────────────────
print('\n[2/5] Enumerating diagrams ...')
vtypes = extract_vertex_types(ft)
stypes = extract_source_types(ft)
ring_var_names_list = list(ns._ring_var_names)
resp_idx, phys_idx = build_field_index_map(ring_var_names_list, nf_resp)

all_unique = []
for ell in range(max_ell + 1):
    _, _, pds, _ = enumerate_all(k, ell, verbose=False)
    kept, _ = filter_prediagrams(pds, vtypes, stypes)
    typed = enumerate_all_typed(
        kept, external_fields, vtypes, stypes,
        G_ft, resp_idx, phys_idx,
    )
    causal, _, _ = filter_causal(typed)
    unique = deduplicate_typed_diagrams(causal)
    all_unique.extend(unique)
    print(f'  ell={ell}: {len(pds)} PDs → {len(kept)} kept → {len(typed)} typed '
          f'→ {len(causal)} causal → {len(unique)} unique')

print(f'  Total unique tree diagrams: {len(all_unique)}')

# Per-diagram compact summary
time_dep_params = HAWKES_MODEL.get('time_dependent_parameters', [])
noise_structure = HAWKES_MODEL.get('noise_structure', {
    'temporal_type': 'white', 'amplitude_params': [],
})

diagram_prefactors = []
for i, td in enumerate(all_unique):
    ci = classify_coefficient_factors(td, time_dep_params, noise_structure)
    diagram_prefactors.append(ci['scalar_prefactor'])

print('\n[3/5] Diagram inventory:')
for i, (td, pf) in enumerate(zip(all_unique, diagram_prefactors)):
    ext = {lf: td.external_legs[lf] for lf in td.prediagram[2]}
    print(f'  D{i}: ext={ext}  prefactor={pf}')
    for v, vt in sorted(td.vertex_assignments.items()):
        tname = type(vt).__name__
        bigrade = vt.bigrade
        print(f'    v{v} ({tname}): bigrade={bigrade}')

# ────────────────────────────────────────────────────────────────
# 4. Solve MF and build num_params
# ────────────────────────────────────────────────────────────────
print('\n[4/5] Solving mean-field ...')
v_sym = SR.var('_v_mf_')
_param_subs = {}
for pspec in HAWKES_MODEL.get('parameters', []):
    pname = pspec['name']
    if pname in fundamental:
        if pspec.get('indexed', False):
            for i in ns.pop:
                sym = getattr(ns, pname)[i]
                _param_subs[sym] = fundamental[pname][i]
        else:
            sym = getattr(ns, pname)
            _param_subs[sym] = fundamental[pname]

phi_derivs = {}
for i in ns.pop:
    phi_expr = HAWKES_MODEL['phi_concrete'](ns, i, v_sym)
    phi_derivs[i] = {}
    for dk in range(ft.taylor_order + 1):
        if dk == 0:
            phi_derivs[i][dk] = phi_expr
        else:
            phi_derivs[i][dk] = diff(phi_expr, v_sym, dk)

def phi_num(i, v_val):
    return float(phi_derivs[i][0].subs(_param_subs).subs({v_sym: v_val}))

def mf_residual(nstar_vec):
    residuals = []
    for i in ns.pop:
        v_star_i = fundamental['E'][i] + sum(
            fundamental['w'][i][j] * nstar_vec[j] for j in ns.pop
        )
        residuals.append(nstar_vec[i] - phi_num(i, v_star_i))
    return residuals

nstar_sol = fsolve(mf_residual, [1.0] * len(ns.pop), full_output=True)
nstar_vals = [float(x) for x in nstar_sol[0]]
vstar_vals = []
phi_deriv_vals = {}
for i in ns.pop:
    v_star_i = fundamental['E'][i] + sum(
        fundamental['w'][i][j] * nstar_vals[j] for j in ns.pop
    )
    vstar_vals.append(v_star_i)
    for dk in range(ft.taylor_order + 1):
        phi_deriv_vals[(dk, i)] = float(
            phi_derivs[i][dk].subs(_param_subs).subs({v_sym: v_star_i})
        )

num_params = {}
for i in ns.pop:
    for j in ns.pop:
        num_params[SR.var(f'w{i+1}{j+1}')] = fundamental['w'][i][j]
num_params[SR.var('tau')] = fundamental['tau']
for i in ns.pop:
    num_params[ns.nstar[i]] = float(nstar_vals[i])
for i in ns.pop:
    for dk in range(1, ft.taylor_order + 1):
        sym = SR.var(f'phi{dk}_{i+1}')
        if (dk, i) in phi_deriv_vals:
            num_params[sym] = phi_deriv_vals[(dk, i)]

print(f'  n* = {nstar_vals}')
print(f'  v* = {vstar_vals}')

# ────────────────────────────────────────────────────────────────
# 5. Run compute_correction_td twice: origin=0 and origin=1
# ────────────────────────────────────────────────────────────────
print('\n[5/5] Running Phase J tree evaluator ...')

res_origin0 = compute_correction_td(
    typed_diagrams=all_unique,
    prefactors=diagram_prefactors,
    propagator_data=propagator_data,
    k=k,
    num_params=num_params,
    origin_leaf_idx=0,
    external_fields=external_fields,
)
res_origin1 = compute_correction_td(
    typed_diagrams=all_unique,
    prefactors=diagram_prefactors,
    propagator_data=propagator_data,
    k=k,
    num_params=num_params,
    origin_leaf_idx=1,
    external_fields=external_fields,
)

print(f'  origin=0: skipped_kernel_ids = {res_origin0["skipped_kernel_ids"]}')
print(f'  origin=1: skipped_kernel_ids = {res_origin1["skipped_kernel_ids"]}')
print(f'  n groups = {len(res_origin0["groups"])}')

total_C_0 = res_origin0['total_C']
total_C_1 = res_origin1['total_C']

# ────────────────────────────────────────────────────────────────
# Report: total_C values
# ────────────────────────────────────────────────────────────────
print('\n' + '='*70)
print('RESULT 1: total_C evaluated at ±τ for each origin choice')
print('='*70)
print(f"\nConvention: external_fields[0]=('dn',1) at position 0 (time t_1),")
print(f"            external_fields[1]=('dv',2) at position 1 (time t_2).")
print(f"\nWith origin=0 (t_1 pinned to 0, t_2 = τ free):")
print(f"  total_C(0, τ) = ⟨δn_1(0) · δv_2(τ)⟩  [should be causally > 0 at +τ]")
print(f"\nWith origin=1 (t_2 pinned to 0, t_1 = τ free):")
print(f"  total_C(τ, 0) = ⟨δn_1(τ) · δv_2(0)⟩  [should be causally > 0 at -τ]")
print()

header = f"{'τ':>8s}  {'origin=0 C(0,τ)':>22s}  {'origin=1 C(τ,0)':>22s}  {'origin=0 C(0,-τ)':>22s}"
print(header)
print('-' * len(header))

c0_table = {}
c1_table = {}
for tv in TAU_VALS:
    v0_pos = complex(total_C_0(0.0, tv)).real
    v1_pos = complex(total_C_1(tv, 0.0)).real
    v0_neg = complex(total_C_0(0.0, -tv)).real
    c0_table[tv] = v0_pos
    c1_table[tv] = v1_pos
    print(f"{tv:>8.2f}  {v0_pos:>22.6e}  {v1_pos:>22.6e}  {v0_neg:>22.6e}")

# Check peak direction
pos_side = max(c0_table[tv] for tv in TAU_VALS if tv > 0)
neg_side = max(c0_table[tv] for tv in TAU_VALS if tv < 0)

print()
print(f"origin=0 max|C(0,τ)| at τ>0: {pos_side:.6e}")
print(f"origin=0 max|C(0,τ)| at τ<0: {neg_side:.6e}")

if abs(neg_side) > abs(pos_side) * 1.5:
    verdict = 'origin=0 PEAKS AT NEGATIVE τ  → MIRROR BUG CONFIRMED'
elif abs(pos_side) > abs(neg_side) * 1.5:
    verdict = 'origin=0 peaks at POSITIVE τ  → (causally correct, no bug)'
else:
    verdict = 'origin=0 roughly symmetric (inconclusive)'
print(f'VERDICT: {verdict}')

# Stationarity check: origin=1 @ (τ,0) should == origin=0 @ (0,-τ).
print()
print('Stationarity consistency check:')
print(f"{'τ':>8s}  {'origin=1 C(τ,0)':>22s}  {'origin=0 C(0,-τ)':>22s}  {'rel_diff':>12s}")
for tv in TAU_VALS:
    v1 = complex(total_C_1(tv, 0.0)).real
    v0m = complex(total_C_0(0.0, -tv)).real
    rd = abs(v1 - v0m) / max(abs(v1), abs(v0m), 1e-12)
    print(f"{tv:>8.2f}  {v1:>22.6e}  {v0m:>22.6e}  {rd:>12.3e}")

# ────────────────────────────────────────────────────────────────
# Report: per-diagram contributions at τ = ±3
# ────────────────────────────────────────────────────────────────
print('\n' + '='*70)
print('RESULT 2: per-diagram contributions at τ = ±3 (origin=0)')
print('='*70)
print()
print(f"{'Diag':>6s}  {'ext legs':>30s}  {'C(0,+3)':>16s}  {'C(0,-3)':>16s}  "
      f"{'ratio -/+':>10s}")
print('-' * 90)

for i, g in enumerate(res_origin0['groups']):
    fn = g.get('contribution')
    if fn is None:
        print(f"{i:>6d}  {'(skipped: '+g.get('reason','?')+')':>30s}")
        continue
    td = all_unique[g['kernel_id']]
    ext = {lf: td.external_legs[lf] for lf in td.prediagram[2]}
    val_pos = complex(fn(0.0, 3.0)).real
    val_neg = complex(fn(0.0, -3.0)).real
    ratio = val_neg / val_pos if abs(val_pos) > 1e-15 else float('inf')
    print(f"{i:>6d}  {str(ext):>30s}  {val_pos:>16.6e}  {val_neg:>16.6e}  "
          f"{ratio:>10.3f}")

# Find diagrams whose contribution flips sign across τ = 0
print()
print('Per-diagram sign-flip analysis:')
for i, g in enumerate(res_origin0['groups']):
    fn = g.get('contribution')
    if fn is None:
        continue
    td = all_unique[g['kernel_id']]
    val_pos = complex(fn(0.0, 3.0)).real
    val_neg = complex(fn(0.0, -3.0)).real
    tag = ''
    if abs(val_pos) < 1e-12 and abs(val_neg) > 1e-12:
        tag = '← ONLY NONZERO AT NEGATIVE τ (suspicious)'
    elif abs(val_pos) > 1e-12 and abs(val_neg) < 1e-12:
        tag = '(only at +τ — causally correct)'
    if tag:
        ext = {lf: td.external_legs[lf] for lf in td.prediagram[2]}
        print(f"  D{i}: ext={ext}  C(0,+3)={val_pos:.3e}  C(0,-3)={val_neg:.3e}  {tag}")

print('\nDone.')
