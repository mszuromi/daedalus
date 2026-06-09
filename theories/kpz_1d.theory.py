"""
1D KPZ equation (with a confining mass) — a *per-leg* gradient vertex test
theory for the spatial v2 operator IR.

    ∂_t h  =  -μ h  +  D ∇²h  +  (λ/2) (∂_x h)²  +  η ,
    ⟨η(x,t) η(x',t')⟩  =  2T δ(x-x') δ(t-t').

The ``-μh`` damping (a confining/Edwards-Wilkinson mass) makes the propagator
massive and IR-safe and isolates the homogeneous saddle ``h*=0`` (the gradient
forcing ``(∂_x h)²`` vanishes there).  Pure KPZ is the ``μ→0`` limit.

The defining feature vs Burgers: the nonlinearity ``(∂_x h)²`` is a **per-leg**
derivative — ``∂_x`` acts on *each of the two physical legs separately*, not on
a composite ``∂_x(h²)``.  So the loop form factor is ``mode='perleg'``:
``𝔉 = ∏_legs (i p_leg)`` → on a φ̃φ² bubble ``F = ℓ²·q·(ℓ−q)``, the
``ℓ·(q−ℓ)`` dot-product KPZ signature (renormalizes the diffusion ``ν=D``).
This is genuinely ``(∂φ)² ≠ ∂(φ²)`` — the distinguishing case from Burgers.

Companion: ``theories/burgers_1d.theory.py`` (the *composite* ``∂_x(φ²)``
sibling).
"""
from pipeline.theory import SpatialTheoryBuilder


def build():
    return (
        SpatialTheoryBuilder('1D KPZ (per-leg gradient vertex)',
                      n_populations=0)
        .physical_field('h', spatial_dim=1, description='interface height')
        .parameter('mu',  default=1.0, domain='positive')
        .parameter('D',   default=1.0, domain='positive')
        .parameter('lam', default=0.3, domain='real')
        .parameter('T',   default=1.0, domain='positive')
        # Deterministic EOM for the MF saddle.  The gradient forcing
        # (λ/2)(∂_x h)² vanishes at the homogeneous saddle (∂_x of a constant
        # is 0), so the saddle is fixed by the linear part: −μh*=0 ⇒ h*=0.
        # (The MF-equation parser handles Dt/Laplacian but not the Dx node,
        # and the term is identically 0 here, so we write the reduced RHS.)
        .equation(lhs='(Dt + mu - D*Laplacian)*h', rhs='0')
        # Operator-IR action: Dx(h, 0)^2 is a per-leg first-derivative vertex
        # — ∂_x acts on EACH physical leg (mode='perleg'), not on a composite.
        # The response field for ``h`` is ``ht``; the ``0`` is the axis (d=1).
        .set_action_text(
            'ht*(Dt(h) + mu*h - D*Lap(h) - (lam/2)*Dx(h, 0)^2) - T*ht^2')
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
    'recommended_external_fields': [('h', 1), ('h', 1)],
    'tau_max': 2.0,
    'tau_step': 0.5,
    'spatial_grid': [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
}
