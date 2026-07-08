"""
models/reaction_diffusion_quadratic_1d.model.py
==================================================
1D reaction–diffusion field with a QUADRATIC nonlinearity — the minimal
model whose 1-loop self-energy is a genuine momentum-dependent **bubble**
(not the momentum-independent tadpole of the φ³ Allen–Cahn case).

    ∂_t φ = (D ∂_x² − μ) φ − g φ²  + ξ,
    ⟨ξ(x,t) ξ(x',t')⟩ = 2T δ(x−x') δ(t−t').

MSR–JD action (response field φ̃):

    S = φ̃ (∂_t + μ − D ∂_x²) φ  +  g φ̃ φ²  −  T φ̃².

The cubic MSR vertex ``g·φ̃ φ²`` (bigrade (1,2)) gives, at 1-loop, the
self-energy bubble: two such vertices joined by two propagator lines
carrying loop momentum ℓ and q−ℓ — so Σ(q) is momentum-DEPENDENT.  This is
the Stage C.5 test case for the per-edge ∫dℓ loop integrator.

Subcritical / stable fixed point: φ* = 0 (the other root φ* = −μ/g is
unstable); we expand around φ* = 0.
"""

from api.model import SpatialModelBuilder


def build():
    return (
        SpatialModelBuilder('1D reaction-diffusion (quadratic, bubble test)',
                      n_populations=0)
        .physical_field('phi', spatial_dim=1, description='density field')
        .parameter('mu', default=1.0, domain='positive',
                   description='linear relaxation / mass')
        .parameter('D', default=1.0, domain='positive',
                   description='diffusion constant')
        .parameter('g', default=0.3, domain='real',
                   description='quadratic reaction coupling')
        .parameter('T', default=1.0, domain='positive',
                   description='noise temperature')
        .set_action_text(
            'phit*((Dt + mu - D*Laplacian)*phi + g*phi^2) - T*phit^2')
        .equation(lhs='(Dt + mu - D*Laplacian)*phi', rhs='-g*phi^2')
        .boundary('infinite')
        .initial('stationary')
        .build())


# Run defaults — the build()'s param defaults (mu=1, D=1, g=0.3, T=1) are
# already a mildly-perturbative regime around the stable fixed point φ*=0,
# so DEFAULT_FUNDAMENTAL just pins them explicitly for the demo notebooks.
DEFAULT_FUNDAMENTAL = {
    'mu': 1.0,
    'D': 1.0,
    'g': 0.3,
    'T': 1.0,
}

METADATA = {
    'k_default': 2,
    'ell_default': 1,
    'recommended_external_fields': [('dphi', 1), ('dphi', 1)],
    'spatial_grid': [-6.0, 6.0, 49],
    'tau_max': 0.0,
    'tau_step': 1.0,
}
