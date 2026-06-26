"""
Combined model — Allen-Cahn ⊕ Model B ⊕ KPZ (d = 1).

One SPDE that stitches together the *unique* nonlinearity of each class:

    ∂_t φ = -μ φ + D ∂_x²φ
            - λ φ³                 (Allen-Cahn: φ⁴ potential, plain vertex)
            + g ∂_x²(φ²)           (Model B: conserved ∇² composite vertex)
            + (κ/2)(∂_x φ)²        (KPZ: per-leg gradient vertex)
            + η ,
    ⟨η(x,t) η(x',t')⟩ = 2T δ(x-x') δ(t-t').

This stresses three nonlinearities at once through the per-vertex form-factor
table (``ns._operator_ir_vertex_terms``): each φ̃φ² node sums the Model B (∇²)
and KPZ (∂_x) contributions coupling-weighted, and a mixed diagram reconstructs
every cross term — no single-mode gate.

Caveat for THIS combination: the Model B × KPZ *cross* bubble has a degree-4
loop form factor (𝔉_MB ∼ ℓ² × 𝔉_KPZ ∼ ℓ²), so its momentum loop is UV-divergent
in d=1 — mixing a *conserved* ∇²(φ²) with a *non-conserved* (∂φ)² generates a
UV-relevant cross operator.  The bare equal-time variance is therefore
UV-dominated (it can go negative) and needs renormalisation; the single-vertex
limits (g only, κ only, λ only) are each finite and physical.  The simulation
runs the full SPDE regardless — that is the clean object to look at.

Companions: ``theories/kpz_1d.theory.py`` (per-leg gradient only),
``theories/reaction_diffusion_conserved_1d.theory.py`` (Model B / composite ∇²
only), ``theories/allen_cahn_1d_subcritical_infinite.theory.py`` (φ³ only).
"""
from api.theory import TheoryBuilder


def build():
    return (
        TheoryBuilder('allen-cahn+modelB+kpz-1d', n_populations=0)
        .physical_field('phi', spatial_dim=1)
        .parameter('mu',  default=1.0,  domain='positive')
        .parameter('D',   default=1.0,  domain='positive')
        .parameter('lam', default=0.1,  domain='real')   # φ³  (Allen-Cahn)
        .parameter('g',   default=0.15, domain='real')   # ∇²(φ²)  (Model B)
        .parameter('kpz', default=0.3,  domain='real')   # (∂ₓφ)²  (KPZ)
        .parameter('T',   default=1.0,  domain='positive')
        # MF saddle: the ∇²/∂ₓ forcings vanish at the homogeneous φ*, so the
        # saddle is the Allen-Cahn one (−μφ*−λφ*³=0 ⇒ φ*=0).
        .equation(lhs='(Dt + mu - D*Laplacian)*phi', rhs='-lam*phi^3')
        .set_action_text(
            'phit*(Dt(phi) + mu*phi - D*Lap(phi) + lam*phi^3 '
            '- g*Lap(phi^2) - (kpz/2)*Dx(phi, 0)^2) - T*phit^2')
        .operator_ir()
        .boundary('infinite')
        .initial('stationary')
        .build()
    )


DEFAULT_FUNDAMENTAL = {
    'mu':  1.0,
    'D':   1.0,
    'lam': 0.1,
    'g':   0.15,
    'kpz': 0.3,
    'T':   1.0,
}


METADATA = {
    'k_default': 2,
    'ell_default': 1,
    'recommended_external_fields': [('dphi', 1), ('dphi', 1)],
    'tau_max': 0.0,
    'tau_step': 1.0,
    'spatial_grid': [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0,
                     2.25, 2.5, 2.75, 3.0, 3.25, 3.5, 3.75, 4.0, 4.25,
                     4.5, 4.75, 5.0, 5.25, 5.5, 5.75, 6.0],
}
