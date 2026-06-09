"""
1D stochastic Allen-Cahn (subcritical, PERIODIC domain) — spatial v1
test theory.

Identical to ``allen_cahn_1d_subcritical_infinite`` except the spatial
domain is a periodic cell of length ``L`` (a sweepable parameter, so
the PBC→∞ limit is a one-parameter sweep — the single best sanity
check that the boundary + integration machinery is wired correctly).

    (∂_t + μ - D ∇²) φ  =  -λ φ³  +  η  on a ring of circumference L,
    ⟨η(x,t) η(x',t')⟩  =  2T Σ_n δ(x - x' + nL) δ(t - t').

The periodic propagator is the image-source sum
``G_PBC(t, x) = Σ_n G_inf(t, x + nL)``.
"""
from pipeline.theory import SpatialTheoryBuilder


def build():
    return (
        SpatialTheoryBuilder('1D stochastic Allen-Cahn (subcritical, periodic domain)',
                      n_populations=0)
        .physical_field('phi', spatial_dim=1, description='order parameter')
        .parameter('mu',  default=1.0, domain='positive')
        .parameter('D',   default=1.0, domain='positive')
        .parameter('lam', default=0.1, domain='positive')
        .parameter('T',   default=1.0, domain='positive')
        .parameter('L',   default=20.0, domain='positive')
        .set_action_text(
            'phit*((Dt + mu - D*Laplacian)*phi + lam*phi^3) - T*phit^2')
        .equation(lhs='(Dt + mu - D*Laplacian)*phi', rhs='-lam*phi^3')
        .stability_analysis(True)
        .boundary('periodic', length='L')
        .initial('stationary')
        .build()
    )


DEFAULT_FUNDAMENTAL = {
    'mu':  1.0,
    'D':   1.0,
    'lam': 0.1,
    'T':   1.0,
    'L':   20.0,
}


METADATA = {
    'k_default': 2,
    'ell_default': 0,
    'recommended_external_fields': [('phi', 1), ('phi', 2)],
    'tau_max': 5.0,
    'tau_step': 0.5,
    'spatial_grid': [-10.0, -5.0, 0.0, 5.0, 10.0],
}
