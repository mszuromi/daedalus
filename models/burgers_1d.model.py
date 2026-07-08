"""
1D stochastically-forced (damped) Burgers equation — a *composite* gradient
vertex test model for the spatial v2 operator IR.

    ∂_t φ  =  -μ φ  +  D ∇²φ  -  (λ/2) ∂_x(φ²)  +  η ,
    ⟨η(x,t) η(x',t')⟩  =  2T δ(x-x') δ(t-t').

Equivalently ``∂_tφ = -μφ + D∇²φ - λ φ ∂_xφ + η`` (Burgers, since
``φ ∂_xφ = ½ ∂_x(φ²)``).  The linear damping ``-μφ`` gives a massive, IR-safe
propagator and isolates the homogeneous saddle ``φ*=0`` (the gradient forcing
``∂_x(φ²)`` vanishes there).

The MSR-JD action carries a genuine **first-derivative** vertex
``(λ/2)·φ̃·∂_x(φ²)`` — the ``∂_x`` acts on the φ² *composite*, so the loop
form factor is ``mode='composite'`` on the response-leg momentum
(``𝔣 = i p`` → an *imaginary* form factor, unlike the real ``Lap → −k²`` of
the conserved Model-B model).  Expanding the vertex about the saddle also
produces a *bilinear* cross-term ``μ_drift·φ̃·∂_x(δφ)`` whose coefficient
``∝ φ*`` vanishes at ``φ*=0`` — handled by the drift-generalized heat kernel
(``extract_mass_diffusion`` returns mass μ, diffusion D, drift V→0).

Companion: ``models/kpz_1d.model.py`` (the *per-leg* ``(∂_xφ)²`` sibling).
"""
from api.model import SpatialModelBuilder


def build():
    return (
        SpatialModelBuilder('1D damped Burgers (composite gradient vertex)',
                      n_populations=0)
        .physical_field('phi', spatial_dim=1, description='velocity potential')
        .parameter('mu',  default=1.0, domain='positive')
        .parameter('D',   default=1.0, domain='positive')
        .parameter('lam', default=0.3, domain='real')
        .parameter('T',   default=1.0, domain='positive')
        # Deterministic EOM for the MF saddle.  The gradient forcing
        # −(λ/2)∂_x(φ²) vanishes at the homogeneous saddle (∂_x of a constant
        # is 0), so the saddle is fixed by the linear part: −μφ*=0 ⇒ φ*=0.
        # (The MF-equation parser handles Dt/Laplacian but not the Dx node,
        # and the term is identically 0 here, so we write the reduced RHS.)
        .equation(lhs='(Dt + mu - D*Laplacian)*phi', rhs='0')
        # Operator-IR action: Dx(phi^2, 0) is a first-derivative vertex on the
        # φ² composite (per-leg form factor, mode='composite').  The ``0`` is
        # the spatial axis (d=1).
        .set_action_text(
            'phit*(Dt(phi) + mu*phi - D*Lap(phi) + (lam/2)*Dx(phi^2, 0)) '
            '- T*phit^2')
        .operator_ir()
        .boundary('infinite')
        .initial('stationary')
        .build()
    )


DEFAULT_FUNDAMENTAL = {
    'mu':  1.0,
    'D':   1.0,
    'lam': 0.3,
    'T':   1.0,
}


METADATA = {
    'k_default': 2,
    'ell_default': 1,
    'recommended_external_fields': [('phi', 1), ('phi', 1)],
    'tau_max': 2.0,
    'tau_step': 0.5,
    'spatial_grid': [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
}
