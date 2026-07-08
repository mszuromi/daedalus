"""
models/reaction_diffusion_2d.model.py
========================================
2D reaction–diffusion field with a QUADRATIC nonlinearity — the d=2
companion of ``reaction_diffusion_quadratic_1d``.  Its 1-loop self-energy
is a genuine momentum-dependent **bubble** (not a constant mass shift),
and the loop ``∫ d²ℓ`` is closed-form in any dimension via the analytic
Symanzik momentum reduction.

    ∂_t φ = D ∇²φ − μ φ − g φ²  + η,
    ⟨η(x,t) η(x',t')⟩ = 2T δ²(x−x') δ(t−t').

MSR–JD action (response field φ̃):

    S = φ̃ (∂_t + μ − D ∇²) φ  +  g φ̃ φ²  −  T φ̃².

The cubic MSR vertex ``g·φ̃ φ²`` gives, at 1-loop, the self-energy bubble:
two such vertices joined by two propagator lines carrying loop momentum ℓ
and q−ℓ, so Σ(q) is momentum-DEPENDENT and the loop correction is ∝ g².
Subcritical / stable fixed point φ* = 0 (the other root φ* = −μ/g is
unstable); we expand around φ* = 0.

Mirrors the inline build of
``notebooks/spatial/pipeline_reaction_diffusion_2d_loop_sim_compare.ipynb``
VERBATIM (same field, parameters, equation, action, boundary, initial).
"""

from api.model import ModelBuilder


def build():
    return (
        ModelBuilder('reaction-diffusion (d=2)', n_populations=0)
        .physical_field('phi', spatial_dim=2)
        .parameter('mu', default=1.0, domain='positive')
        .parameter('D',  default=1.0, domain='positive')
        .parameter('g',  default=0.2, domain='real')
        .parameter('T',  default=1.0, domain='positive')
        .equation(lhs='(Dt + mu - D*Laplacian)*phi', rhs='-g*phi^2')
        .set_action_text(
            'phit*((Dt + mu - D*Laplacian)*phi + g*phi^2) - T*phit^2')
        .boundary('infinite')
        .initial('stationary')
        .build())


# Run defaults — the numeric params the notebook ran at (mu=D=T=1, g=0.2),
# a mildly-perturbative regime around the stable fixed point φ*=0.
DEFAULT_FUNDAMENTAL = {
    'mu': 1.0,
    'D': 1.0,
    'g': 0.2,
    'T': 1.0,
}

METADATA = {
    'k_default': 2,
    'ell_default': 1,
    'recommended_external_fields': [('dphi', 1), ('dphi', 1)],
    # The notebook samples the radial separation r ∈ [0.4, 4.0] (14 points).
    'spatial_grid': [0.4, 4.0, 14],
    'tau_max': 1.0,
    'tau_step': 1.0,
}
