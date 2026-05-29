"""
Stage A spike — spatial tree-level THROUGH THE SHARED PIPELINE (t,k).

RESULT (2026-05-29 — PASS, proves the re-architecture plumbing):
  (1) the SHARED pipeline (compute_poles_and_residues + compute_correction_td)
      with Laplacian -> -q^2 in num_params produces
      C(q,τ) = (T/m) e^{-m|τ|},  m = μ+D q^2,  to MACHINE PRECISION
      (max rel ≤ 1.5e-16 across q).  ONE typed diagram: the (2,0) noise
      source with its two φ̃ legs each joined to an external φ by G_R.
  (2) the q-FT of that pipeline C(q,τ) reproduces the bespoke oracle
      C(x,τ) to ~1e-13 (τ>0, Gaussian-damped clean numerical q-FT).

KEY UNBLOCK: the free 2-point IS a diagram — the earlier "0 typed" was a
wrong external-field spec.  A single-component field has only
phys_idx key ('dphi', 1), so BOTH external legs of the auto-correlation
sit at component 1: external_fields = [('dphi',1), ('dphi',1)].  Using
('dphi',2) (a non-existent component) is what gave 0 typed.

For PRODUCTION the q-FT becomes analytic (residue/erf = the G_tx/erf
closed forms) so τ=0 is exact and there is no ringing; this spike uses
a numerical q-FT only to validate the pipeline's C(q,τ) at τ>0.
""" + """
Stage A spike — spatial tree-level THROUGH THE SHARED PIPELINE (t,k).

Proves the re-architecture's core plumbing: the spatial linear-diffusion
theory flows through the SAME diagram→Phase-J machinery as a time-only
theory, by substituting Laplacian → -q^2 into the propagator
(q = external momentum, a num_param) so the pipeline sees a time-only
rational propagator at effective mass m(q) = mu + D q^2.  The pipeline
returns C(q, tau); we then q-FT to C(x, tau) and check the oracle.

Pipeline pieces used (mirrors pipeline/compute.py steps 4-7):
  build_propagator -> compute_poles_and_residues(prop, num_params with
  Laplacian=-q^2) -> enumerate_unique_diagrams -> classify_coefficient_
  factors -> compute_correction_td.   NO bespoke spatial code.

Checks:
  (1) C(q,tau) from the pipeline == (T/m) e^{-m|tau|},  m = mu+D q^2
  (2) q-FT of C(q,tau)  ==  oracle C(x,tau)   [tau>0, Gaussian-damped,
      clean numerical q-FT]; tau=0 noted (needs residue).
"""
import importlib.util, os, sys, math
import numpy as np

REPO = '/Users/matthewszuromi/Documents/Education/BU PhD/Ocker Lab/Automated Feynman Calculations'
sys.path.insert(0, REPO); os.chdir(REPO)

from sage.all import SR
from msrjd.core.field_theory import FieldTheory
from msrjd.core.vertices import extract_vertex_types, extract_source_types
from msrjd.diagrams.type_assignment import build_field_index_map
from pipeline._propagator import build_propagator, compute_poles_and_residues
from pipeline._diagrams import enumerate_unique_diagrams
from msrjd.integration.time_domain.pipeline import compute_correction_td
from msrjd.diagrams.symmetry import classify_coefficient_factors


def load(p):
    s = importlib.util.spec_from_file_location('m', p)
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m.build()


mu = D = T = 1.0
model = load('theories/linear_diffusion_test.theory.py')
ft = FieldTheory(model, taylor_order=2); ft.expand()
vtypes = extract_vertex_types(ft); stypes = extract_source_types(ft)
prop = build_propagator(ft, model, use_cache=False, verbose=False)

ring_var_names = list(ft._ns._ring_var_names)
n_tilde = ft._n_tilde
resp_idx, phys_idx = build_field_index_map(ring_var_names, n_tilde)
external_fields = [('dphi', 1), ('dphi', 1)]   # single component → both legs same component

# Diagram enumeration is q-INDEPENDENT (topology) -> do it once.
unique_by_ell, mult_by_ell, _ = enumerate_unique_diagrams(
    ft, model, k=2, max_ell=0, external_fields=external_fields,
    G_ft=prop['G_ft'], resp_idx=resp_idx, phys_idx=phys_idx,
    vtypes=vtypes, stypes=stypes, use_cache=False, verbose=False)
records = []
for ell in unique_by_ell:
    for td in unique_by_ell[ell]:
        info = classify_coefficient_factors(td, [], {'temporal_type': 'white',
                                                     'amplitude_params': []})
        records.append((td, SR(info['scalar_prefactor'])))
print(f'diagrams (k=2, ell=0): {len(records)}')

Lap = SR.var('Laplacian')
base_np = {SR.var('mu'): mu, SR.var('D'): D, SR.var('T'): T}


def C_q_tau(qval, taus):
    """Run the SHARED pipeline at Laplacian=-q^2 -> C(q, tau) for each tau."""
    nps = dict(base_np); nps[Lap] = -(qval ** 2)
    compute_poles_and_residues(prop, nps, verbose=False)
    pdata = {k: prop[k] for k in ('K_ker', 'K_ft', 'G_ft', 'adj_ft', 'D_omega',
                                  'D_delta', 't_var', 'omega', 'nf',
                                  'pole_vals', 'C_mats')}
    res = compute_correction_td(
        typed_diagrams=[r[0] for r in records],
        prefactors=[r[1] for r in records],
        k=2, propagator_data=pdata, external_fields=external_fields,
        num_params=nps, origin_leaf_idx=0)
    tC = res['total_C']
    return np.array([complex(tC(0.0, float(t))) for t in taus])


# ---- Check (1): pipeline C(q,tau) == (T/m) e^{-m|tau|} -------------
print('\n=== (1) pipeline C(q,τ) vs (T/m)e^{-m|τ|},  m=mu+D q^2 ===')
taus = np.array([0.0, 0.5, 1.0, 2.0])
for qv in [0.0, 0.8, 1.5]:
    cq = C_q_tau(qv, taus)
    m = mu + D * qv * qv
    ref = (T / m) * np.exp(-m * np.abs(taus))
    rel = np.max(np.abs(cq.real - ref) / np.maximum(np.abs(ref), 1e-30))
    print(f'  q={qv:>4}: m={m:.3f}  pipeline={np.round(cq.real,6)}  '
          f'closed={np.round(ref,6)}  max rel={rel:.2e}')

# ---- Check (2): q-FT of pipeline C(q,tau) == oracle C(x,tau) -------
from scipy import integrate
def oracle(x, tau):
    if abs(tau) < 1e-12:
        return T / (2 * math.sqrt(mu * D)) * math.exp(-abs(x) / math.sqrt(D / mu))
    f = lambda s: s ** -0.5 * math.exp(-x * x / (4 * D * s) - mu * s)
    v, _ = integrate.quad(f, abs(tau), np.inf)
    return T / math.sqrt(4 * math.pi * D) * v

print('\n=== (2) q-FT of pipeline C(q,τ) vs oracle C(x,τ)  (τ>0) ===')
Q, NQ = 8.0, 80
qgrid = np.linspace(-Q, Q, 2 * NQ + 1)
taus2 = [0.5, 1.0]
xs = [0.0, 1.0, 2.0]
# Precompute pipeline C(q,tau) on the q-grid (q-even -> half + mirror)
Cq = {t: np.array([C_q_tau(q, [t])[0].real for q in qgrid]) for t in taus2}
print(f'  (built C(q,τ) on {len(qgrid)} momentum points via the pipeline)')
print(f'{"τ":>5} {"x":>5} {"pipeline q-FT":>16} {"oracle":>12} {"rel":>10}')
for t in taus2:
    for x in xs:
        integ = np.cos(qgrid * x) * Cq[t]
        cxt = np.trapz(integ, qgrid) / (2 * np.pi)
        orc = oracle(x, t)
        print(f'{t:>5} {x:>5} {cxt:>16.8f} {orc:>12.8f} '
              f'{abs(cxt-orc)/max(abs(orc),1e-30):>10.2e}')
print('\n(τ=0 equal-time would need the residue closed form, not numeric q-FT)')
