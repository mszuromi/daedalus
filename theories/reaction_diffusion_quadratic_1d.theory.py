"""
theories/reaction_diffusion_quadratic_1d.theory.py
==================================================
1D reaction–diffusion field with a QUADRATIC nonlinearity — the minimal
theory whose 1-loop self-energy is a genuine momentum-dependent **bubble**
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

from pipeline.theory import SpatialTheoryBuilder


def build():
    return (
        SpatialTheoryBuilder('1D reaction-diffusion (quadratic, bubble test)',
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
