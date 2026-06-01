"""
1D stochastic φ⁶ theory (Allen-Cahn + quintic force, subcritical,
infinite domain) — a GENERALIZATION test for the spatial full-diagram
integrator.

Same structure as ``allen_cahn_1d_subcritical_infinite`` but with an
extra ``-γ φ⁵`` force term (a confining sextic potential for γ > 0):

    (∂_t + μ - D ∇²) φ  =  -λ φ³  -  γ φ⁵  +  η,
    ⟨η(x,t) η(x',t')⟩  =  2T δ(x-x') δ(t-t').

The deterministic force ``-μφ - λφ³ - γφ⁵`` has the single real saddle
φ*=0 for μ,λ,γ > 0 (μ + λφ² + γφ⁴ > 0), so the mean field is the same
trivial origin as the φ⁴ theory — only the interaction content differs.

Why this is a good generalization test
--------------------------------------
The action gains a ``φ̃ φ⁵`` interaction vertex of **total degree 6**.
``compute_cumulants`` auto-picks ``taylor_order = max(k + 2·max_ell, 2)``,
so the degree-6 vertex is only captured at ``max_ell ≥ 2``
(``taylor_order = 2 + 2·2 = 6``).  Consequences:

  * at ``max_ell = 1`` this theory is **identical** to pure φ⁴
    (the φ̃φ⁵ vertex is truncated away — γ does not appear);
  * at ``max_ell = 2`` the φ̃φ⁵ vertex enters as a **2-loop tadpole**
    ((5-1)/2 = 2 loops), so γ first corrects the correlator there.

Exercising it checks that the model-independent pipeline enumerates,
classifies, maps to a C-stack descriptor, and integrates a brand-new
higher-degree vertex with no special-casing.
"""
from pipeline.theory import TheoryBuilder


def build():
    return (
        TheoryBuilder('1D stochastic phi^6 (Allen-Cahn + quintic, infinite)',
                      n_populations=0)
        .physical_field('phi', spatial_dim=1, description='order parameter')
        .parameter('mu',    default=1.0, domain='positive')
        .parameter('D',     default=1.0, domain='positive')
        .parameter('lam',   default=0.1, domain='positive')
        .parameter('gamma', default=0.1, domain='positive')
        .parameter('T',     default=1.0, domain='positive')
        # White noise as the quadratic-in-response term -T*phit^2
        # (⟨η η⟩ = 2T δ δ).  The quintic force adds a +gamma*phi^5 term
        # to the drift inside the response bracket (action = phit·(drift)
        # with drift = (Dt + μ - D∇²)φ + λφ³ + γφ⁵).
        .set_action_text(
            'phit*((Dt + mu - D*Laplacian)*phi + lam*phi^3 + gamma*phi^5)'
            ' - T*phit^2')
        .equation(lhs='(Dt + mu - D*Laplacian)*phi',
                  rhs='-lam*phi^3 - gamma*phi^5')
        # μ,λ,γ > 0 → single real saddle φ*=0; enable stability filtering
        # so fixed_point_index selects the stable origin.
        .stability_analysis(True)
        .boundary('infinite')
        .initial('stationary')
        .build()
    )


DEFAULT_FUNDAMENTAL = {
    'mu':    1.0,
    'D':     1.0,
    'lam':   0.1,
    'gamma': 0.1,
    'T':     1.0,
}


METADATA = {
    'k_default': 2,
    'ell_default': 2,            # γ first contributes at 2 loops
    'recommended_external_fields': [('phi', 1), ('phi', 2)],
    'tau_max': 5.0,
    'tau_step': 0.5,
    'spatial_grid': [-5.0, -2.5, 0.0, 2.5, 5.0],
}
