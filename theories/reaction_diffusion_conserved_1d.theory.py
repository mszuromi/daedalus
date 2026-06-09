"""
1D conserved (Model-B-type) reaction-diffusion with a derivative vertex —
the spatial v2 operator-IR test theory.

A scalar density with a *conserving* nonlinearity (a density-dependent /
porous-medium-type flux) plus a linear loss and white noise:

    ∂_t φ  =  D ∇²φ  -  μ φ  +  g ∇²(φ²)  +  η ,
    ⟨η(x,t) η(x',t')⟩  =  2T δ(x-x') δ(t-t').

Equivalently ``∂_t φ = ∇·[(D + 2gφ)∇φ] − μφ + η`` — diffusion with a
density-dependent diffusivity.  The MSR-JD action carries the genuine
*derivative* vertex ``g·φ̃·∇²(φ²)`` (authored with the operator IR), whose
1-loop self-energy is a momentum-DEPENDENT bubble with form factors
``F_R=q²ℓ²``, ``F_K=q⁴`` — exactly the case a constant Hartree mass shift
cannot represent (the conservation law forces Σ(q→0) ∝ q²).

Validated this session (the spatial-v2 wiring): flows through
``compute_cumulants(max_ell=1)`` via the momentum-first bubble path, B≈0.944 vs
simulation (``docs/spatial_spikes/stageC5_derivative_vertex_validation.py``;
sim force ``g_lap`` in ``models/spatial_field_1d_sim.py``).  It is the
first-class backend-C test theory for derivative / form-factor vertices.
"""
from pipeline.theory import SpatialTheoryBuilder


def build():
    return (
        SpatialTheoryBuilder('1D conserved reaction-diffusion (derivative vertex)',
                      n_populations=0)
        .physical_field('phi', spatial_dim=1, description='conserved density')
        .parameter('mu', default=1.0, domain='positive')
        .parameter('D',  default=2.0, domain='positive')
        .parameter('g',  default=0.3, domain='real')
        .parameter('T',  default=1.0, domain='positive')
        # The deterministic EOM (for the MF saddle); the conserved forcing
        # g∇²(φ²) vanishes at the homogeneous φ*=0 saddle, so φ*=0.
        .equation(lhs='(Dt + mu - D*Laplacian)*phi', rhs='g*Laplacian*phi^2')
        # Operator-IR action: Lap(phi^2) is a genuine derivative vertex (the
        # ∇² acts on the φ² composite → per-leg momentum form factor).
        .set_action_text(
            'phit*(Dt(phi) + mu*phi - D*Lap(phi) - g*Lap(phi^2)) - T*phit^2')
        .operator_ir()
        .boundary('infinite')
        .initial('stationary')
        .build()
    )


DEFAULT_FUNDAMENTAL = {
    'mu': 1.0,
    'D':  2.0,
    'g':  0.3,
    'T':  1.0,
}


METADATA = {
    'k_default': 2,
    'ell_default': 1,
    'recommended_external_fields': [('phi', 1), ('phi', 1)],
    'tau_max': 2.0,
    'tau_step': 0.5,
    'spatial_grid': [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
}
