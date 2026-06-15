"""
2D KPZ equation (with a confining mass) — the *full-gradient* per-leg vertex
test theory for the spatial v2 operator IR in d ≥ 2.

    ∂_t h  =  -μ h  +  D ∇²h  +  (λ/2) (∇h)²  +  η ,
    (∇h)²  =  (∂_x h)² + (∂_y h)² ,
    ⟨η(x,t) η(x',t')⟩  =  2T δ²(x-x') δ(t-t').

The ``-μh`` damping (a confining/Edwards-Wilkinson mass) makes the propagator
massive and IR-safe and isolates the homogeneous saddle ``h*=0`` (the gradient
forcing ``(∇h)²`` vanishes there).  Pure KPZ is the ``μ→0`` limit.

The defining feature is the **full d=2 gradient** nonlinearity authored as the
per-axis sum ``Σ_i (∂_i h)² = Dx(h,0)^2 + Dx(h,1)^2`` — one per-leg first-
derivative vertex per axis, summed by the multi-vertex form-factor table into
the rotation-invariant dot product ``𝔉 = -p_1·p_2``.  The loop integral
averages this over the **d-dim** loop Gaussian (the ``L·d``-dim transverse-
moment Gauss–Hermite), validated vs brute ``∫d²ℓ`` to machine precision.

Companion: ``theories/kpz_1d.theory.py`` (the d=1 single-axis sibling).
"""
from pipeline.theory import SpatialTheoryBuilder


def build():
    return (
        SpatialTheoryBuilder('kpz-2d', n_populations=0)
        .physical_field('h', spatial_dim=2)
        .parameter('mu',  default=1.0, domain='positive')
        .parameter('D',   default=1.0, domain='positive')
        .parameter('lam', default=0.3, domain='real')
        .parameter('T',   default=1.0, domain='positive')
        # Deterministic EOM for the MF saddle.  The gradient forcing
        # (λ/2)(∇h)² vanishes at the homogeneous saddle (∂_i of a constant is
        # 0), so the saddle is fixed by the linear part: −μh*=0 ⇒ h*=0.
        .equation(lhs='(Dt + mu - D*Laplacian)*h', rhs='0')
        # Operator-IR action.  (∇h)² = Dx(h,0)^2 + Dx(h,1)^2 is the d=2 full
        # gradient: one per-axis per-leg first-derivative vertex (mode='perleg')
        # per spatial axis.  Response field for ``h`` is ``ht``; the axis index
        # is the second Dx argument (0=x, 1=y).
        .set_action_text(
            'ht*(Dt(h) + mu*h - D*Lap(h) - (lam/2)*(Dx(h, 0)^2 + Dx(h, 1)^2))'
            ' - T*ht^2')
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
    'recommended_external_fields': [('dh', 1), ('dh', 1)],
    'tau_max': 0.0,
    'tau_step': 1.0,
    'spatial_grid': [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0,
                     3.5, 4.0, 4.5, 5.0, 5.5, 6.0],
}
