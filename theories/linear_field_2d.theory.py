"""
2D linear stochastic field (massive Edwards-Wilkinson / Ornstein-Uhlenbeck
field on the infinite plane) — spatial v1 theory.

    (∂_t + μ - D ∇²) φ  =  η,   ⟨η η⟩ = 2T δ² δ,   d = 2.

The simplest *two*-dimensional spatial field theory: a LINEAR Langevin
equation (no interaction vertex), so the tree-level correlator is exact.
The equal-time correlator has the closed 2-D screened (Yukawa) form

    C(r, 0) = (T / 2πD) K₀(r √(μ/D)).

Verbatim port of the inline ``TheoryBuilder('linear field (d=2)')`` build
in ``notebooks/spatial/pipeline_linear_field_2d_sim_compare.ipynb`` —
same parameters, action, equation, boundary and initial condition, with
``spatial_dim=2``.
"""
from pipeline.theory import SpatialTheoryBuilder


def build():
    return (
        SpatialTheoryBuilder('linear field (d=2)', n_populations=0)
        .physical_field('phi', spatial_dim=2)        # <- d=2
        .parameter('mu', default=1.0, domain='positive')
        .parameter('D',  default=1.0, domain='positive')
        .parameter('T',  default=1.0, domain='positive')
        .equation(lhs='(Dt + mu - D*Laplacian)*phi', rhs='0')
        .set_action_text('phit*((Dt + mu - D*Laplacian)*phi) - T*phit^2')
        .boundary('infinite')
        .initial('stationary')
        .build()
    )


DEFAULT_FUNDAMENTAL = {'mu': 1.0, 'D': 1.0, 'T': 1.0}


METADATA = {
    'k_default': 2,
    'ell_default': 0,
    'recommended_external_fields': [('phi', 1), ('phi', 1)],
    'tau_max': 2.0,
    'tau_step': 0.5,
    'spatial_grid': [0.3, 4.0, 16],
}
