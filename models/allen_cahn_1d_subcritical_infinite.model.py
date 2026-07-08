"""
1D stochastic Allen-Cahn (subcritical, infinite domain) — spatial v1
test model.

The first spatial field theory in Daedalus.  Off-critical (μ > 0),
infinite domain, stationary initial condition.  Free part is an
Ornstein-Uhlenbeck-in-time × heat-kernel-in-space propagator; the
λφ³ vertex drives the 1-loop self-energy correction.

    (∂_t + μ - D ∇²) φ  =  -λ φ³  +  η,
    ⟨η(x,t) η(x',t')⟩  =  2T δ(x-x') δ(t-t').

The action's kinetic operator uses the inert ``Laplacian`` symbol
multiplicatively, exactly like ``Dt`` — the framework registers it
because ``phi`` declares ``spatial_dim=1``.  See
``docs/spatial_design_decisions_v1.md`` and
``docs/spatial_implementation_plan.md``.

Phase 1 status: this file ``build()``s and ``FieldTheory.expand()``s.
Propagator construction (heat kernel × exp decay) lands in Phase 2;
the (t, x) loop integral in Phase 5.
"""
from api.model import SpatialModelBuilder


def build():
    return (
        SpatialModelBuilder('1D stochastic Allen-Cahn (subcritical, infinite domain)',
                      n_populations=0)
        .physical_field('phi', spatial_dim=1, description='order parameter')
        .parameter('mu',  default=1.0, domain='positive')
        .parameter('D',   default=1.0, domain='positive')
        .parameter('lam', default=0.1, domain='positive')
        .parameter('T',   default=1.0, domain='positive')
        # White noise written directly in the action as the quadratic-
        # in-response term ``- T*phit^2`` (⟨η η⟩ = 2T δ δ), matching the
        # proven OU-quartic scalar pattern.  The spatial Laplacian rides
        # the drift exactly like Dt; the framework registers it because
        # ``phi`` is spatial.
        .set_action_text(
            'phit*((Dt + mu - D*Laplacian)*phi + lam*phi^3) - T*phit^2')
        .equation(lhs='(Dt + mu - D*Laplacian)*phi', rhs='-lam*phi^3')
        # φ⁴ at μ>0 has the single real saddle φ*=0; enable stability
        # filtering so ``fixed_point_index`` selects the stable origin
        # (harmless in the single-root regime).
        .stability_analysis(True)
        .boundary('infinite')
        .initial('stationary')
        .build()
    )


DEFAULT_FUNDAMENTAL = {
    'mu':  1.0,
    'D':   1.0,
    'lam': 0.1,
    'T':   1.0,
}


METADATA = {
    'k_default': 2,
    'ell_default': 0,
    'recommended_external_fields': [('phi', 1), ('phi', 2)],
    'tau_max': 5.0,
    'tau_step': 0.5,
    # Spatial grid for C(x, τ) output (consumed once Phase 6 lands the
    # spatial_grid kwarg in compute_cumulants).
    'spatial_grid': [-5.0, -2.5, 0.0, 2.5, 5.0],
}
