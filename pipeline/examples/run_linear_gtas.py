"""
End-to-end example: compute the k=2 cross-correlator
⟨δn_1(0) δn_2(τ)⟩ for the linear-GTaS model and produce a PDF report.

Run from the repo root:

    cd "/Users/.../Automated Feynman Calculations"
    OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES sage -python \
        pipeline/examples/run_linear_gtas.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

from models.hawkes_linear_expg_gtas import HAWKES_MODEL
from pipeline import compute_cumulants, generate_report


# ── 1. Define the working point ────────────────────────────────────
fundamental = {
    'E':                 [1.1, 1.05],
    'w':                 [[0.35, 0.4], [0.3, 0.5]],
    'tau':               10.0,
    'a':                 1.0,
    'tau_g':             2.5,
    'w_X':               0.3,
    'lambda_X':          2.5,
    'p_part':            0.6,
    'mu_shift_diff':     0.0,
    'sigma_shift_diff_sq': 1.0,
}


# ── 2. Run the pipeline + save CSV-friendly numerical output ──────
out_dir = 'pipeline_outputs'
os.makedirs(out_dir, exist_ok=True)

result = compute_cumulants(
    model           = HAWKES_MODEL,
    k               = 2,
    max_ell         = 0,
    fundamental     = fundamental,
    external_fields = [('dn', 1), ('dn', 2)],
    tau_max         = 50.0,
    tau_step        = 0.5,
    output_npz      = os.path.join(out_dir, 'linear_gtas_k2.npz'),
)

# ── 3. Generate the PDF diagram-by-diagram report ─────────────────
generate_report(
    model           = HAWKES_MODEL,
    k               = 2,
    fundamental     = fundamental,
    external_fields = [('dn', 1), ('dn', 2)],
    output_pdf      = os.path.join(out_dir, 'linear_gtas_k2_report.pdf'),
    result          = result,    # reuse the compute already done
)

print()
print('=' * 60)
print(f'NPZ:    {os.path.join(out_dir, "linear_gtas_k2.npz")}')
print(f'PDF:    {os.path.join(out_dir, "linear_gtas_k2_report.pdf")}')
print(f'k=2 slice: τ ∈ [{result["tau_grid"][0]:+g}, '
      f'{result["tau_grid"][-1]:+g}], '
      f'{len(result["tau_grid"])} points')
print(f'Diagrams: {len(result["diagrams"])} unique')
print(f'MF n*: {result["mf_values"]["nstar"]}')
print('=' * 60)
