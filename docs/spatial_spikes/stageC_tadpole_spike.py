"""
Stage C spike — spatial 1-loop TADPOLE through the shared pipeline.

The Allen-Cahn λφ³ tadpole is a CONSTANT mass shift Σ.  Its structure:

  * the pipeline (compute_correction_td) computes the ell=1 correction
    using the loop edge at the GLOBAL momentum, i.e. with loop value
    C₀(q,0) = N/(A+Bq²) — NOT the momentum-integrated ⟨φ²⟩₀;
  * but the loop appears LINEARLY and the tadpole acts as a pure mass
    shift, so the pipeline's ell1 obeys
        ell1(q,τ) = Σ_pipe(q) · ∂C₀(q,τ)/∂A ,   Σ_pipe(q) = g · C₀(q,0)
    with g = M(Γ)·(vertex coupling) a q-INDEPENDENT constant the pipeline
    supplies (the combinatorial 3·λ for Allen-Cahn — NOT hardcoded here);
  * the CORRECT self-energy replaces the loop value by the momentum
    integral:  Σ = g · ⟨φ²⟩₀,  ⟨φ²⟩₀ = ∫dℓ/2π C₀(ℓ,0) = free_two_point(A,B,N,0,0)
    (the §4c′ residue/erf closed form);
  * δC(x,τ) = Σ · ∂C₀(x,τ)/∂A  (mass-shift; external q-FT is automatic
    because C₀(x,τ) is already the q-FT'd tree).

Validation targets (μ=D=T=1):
  * g = 3λ  (M(Γ)=3, the dimension-independent tadpole factor)
  * ⟨φ²⟩₀ = 0.5
  * strict-1-loop ⟨φ²⟩ = C₁(0,0) = 0.5 + Σ·∂⟨φ²⟩₀/∂μ
  * self-consistent Hartree ⟨φ²⟩, and the simulator.
"""
import importlib.util, os, sys, math
import numpy as np
import scipy.optimize as opt

REPO = '/Users/matthewszuromi/Documents/Education/BU PhD/Ocker Lab/Automated Feynman Calculations'
sys.path.insert(0, REPO); os.chdir(REPO)

from sage.all import SR
from msrjd.core.field_theory import FieldTheory
from msrjd.core.vertices import extract_vertex_types, extract_source_types
from msrjd.diagrams.type_assignment import build_field_index_map
from msrjd.diagrams.symmetry import classify_coefficient_factors
from pipeline._propagator import build_propagator, compute_poles_and_residues
from pipeline._diagrams import enumerate_unique_diagrams
from msrjd.integration.time_domain.pipeline import compute_correction_td
from msrjd.integration.spatial.spatial_correlator import free_two_point


def load(p):
    s = importlib.util.spec_from_file_location('m', p)
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m.build()


mu = D = T = 1.0
A, B, N = mu, D, T          # single heat-kernel mode
Lap = SR.var('Laplacian')

model = load('theories/allen_cahn_1d_subcritical_infinite.theory.py')
ft = FieldTheory(model, taylor_order=4); ft.expand()
prop = build_propagator(ft, model, use_cache=False, verbose=False)
vt = extract_vertex_types(ft); st = extract_source_types(ft)
rv = list(ft._ns._ring_var_names); nt = ft._n_tilde
ri, pi = build_field_index_map(rv, nt)
ext = [('dphi', 1), ('dphi', 1)]
ub, _, _ = enumerate_unique_diagrams(
    ft, model, k=2, max_ell=1, external_fields=ext, G_ft=prop['G_ft'],
    resp_idx=ri, phys_idx=pi, vtypes=vt, stypes=st, use_cache=False, verbose=False)


def records(ell):
    out = []
    for td in ub[ell]:
        info = classify_coefficient_factors(
            td, [], {'temporal_type': 'white', 'amplitude_params': []})
        out.append((td, SR(info['scalar_prefactor'])))
    return out


def pipe_ell(ell, qval, lam, tau=0.0):
    nps = {SR.var('mu'): mu, SR.var('D'): D, SR.var('T'): T,
           SR.var('lam'): lam, SR.var('phistar1'): 0.0, Lap: -(qval ** 2)}
    compute_poles_and_residues(prop, nps, verbose=False)
    pdata = {k: prop[k] for k in ('K_ker', 'K_ft', 'G_ft', 'adj_ft', 'D_omega',
                                  'D_delta', 't_var', 'omega', 'nf',
                                  'pole_vals', 'C_mats')}
    r = records(ell)
    res = compute_correction_td(
        typed_diagrams=[x[0] for x in r], prefactors=[x[1] for x in r],
        k=2, propagator_data=pdata, external_fields=ext, num_params=nps,
        origin_leaf_idx=0)
    return complex(res['total_C'](0.0, tau)).real


def C0_mom(q, AA=A):
    return N / (AA + B * q * q)             # momentum-space equal-time


def dC0dA_mom(q, AA=A):
    return -N / (AA + B * q * q) ** 2       # ∂/∂A of C0_mom


def extract_g(lam, q_samples=(0.0, 0.8, 1.5)):
    """g = M(Γ)·λ from the pipeline (q-independent)."""
    gs = []
    for q in q_samples:
        e1 = pipe_ell(1, q, lam, tau=0.0)
        Sigma_pipe = e1 / dC0dA_mom(q)      # ell1 = Σ_pipe · ∂C0/∂A
        gs.append(Sigma_pipe / C0_mom(q))   # Σ_pipe = g · loop_value C0(q,0)
    return np.array(gs)


phi2_0 = free_two_point(A, B, N, 0.0, 0.0).real    # ⟨φ²⟩₀ = ∫dℓ/2π C0(ℓ,0)
print(f'⟨φ²⟩₀ (loop integral) = {phi2_0:.6f}  (closed 0.5)')

print(f'\n{"λ":>5} {"g=M·λ (per q)":>26} {"3λ":>6} {"Σ":>7} '
      f'{"strict ⟨φ²⟩":>12} {"Hartree":>9}')
for lam in [0.1, 0.2, 0.4]:
    gq = extract_g(lam)
    g = float(np.mean(gq))
    Sigma = g * phi2_0
    # strict 1-loop ⟨φ²⟩ = ⟨φ²⟩₀ + Σ·∂⟨φ²⟩₀/∂A  (finite diff in A)
    h = 1e-4
    dphi2_dA = (free_two_point(A + h, B, N, 0., 0.).real -
                free_two_point(A - h, B, N, 0., 0.).real) / (2 * h)
    strict = phi2_0 + Sigma * dphi2_dA
    # Hartree: A_eff = A + g·⟨φ²⟩₀(A_eff)
    f = lambda Ae: Ae - (A + g * free_two_point(Ae, B, N, 0., 0.).real)
    Ae = opt.brentq(f, 0.3, 6.0)
    hartree = free_two_point(Ae, B, N, 0., 0.).real
    print(f'{lam:>5} {np.array2string(gq, precision=4):>26} {3*lam:>6.3f} '
          f'{Sigma:>7.4f} {strict:>12.5f} {hartree:>9.5f}')

print('\n(g must equal 3λ and be q-independent → M(Γ)=3 from the pipeline;')
print(' strict-1-loop ⟨φ²⟩ at λ=0.1 should be ≈0.4625)')
