"""
1D Edwards-Wilkinson interface (linear stochastic diffusion) — spatial
v1 test theory.

The simplest spatial field theory: a LINEAR Langevin equation (no
interaction vertex), so the tree-level correlator is exact.

    (∂_t + μ - D ∇²) φ  =  η,   ⟨η η⟩ = 2T δ δ.

Equal-time correlator  C(x, 0) = (T / 2√(μD)) e^{-|x|/ξ},  ξ = √(D/μ).
With μ → 0 this is the massless Edwards-Wilkinson surface; μ > 0 here
keeps it off-critical (a Wilkinson-with-mass / Ornstein-Uhlenbeck
field) so the correlator is normalizable.  Serves as a second
independent check of the spatial machinery (no φ³, distinct from
Allen-Cahn).
"""
from api.theory import SpatialTheoryBuilder


def build():
    return (
        SpatialTheoryBuilder('1D Edwards-Wilkinson (linear, massive)',
                      n_populations=0)
        .physical_field('phi', spatial_dim=1, description='height field')
        .parameter('mu', default=0.5, domain='positive')
        .parameter('D',  default=2.0, domain='positive')
        .parameter('T',  default=1.0, domain='positive')
        .set_action_text('phit*((Dt + mu - D*Laplacian)*phi) - T*phit^2')
        .equation(lhs='(Dt + mu - D*Laplacian)*phi', rhs='0')
        .boundary('infinite')
        .initial('stationary')
        .build()
    )


DEFAULT_FUNDAMENTAL = {'mu': 0.5, 'D': 2.0, 'T': 1.0}


METADATA = {
    'k_default': 2,
    'ell_default': 0,
    'recommended_external_fields': [('phi', 1), ('phi', 2)],
    'tau_max': 5.0,
    'tau_step': 0.5,
    'spatial_grid': [-6.0, -3.0, 0.0, 3.0, 6.0],
}
