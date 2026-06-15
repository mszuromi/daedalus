"""
Combined multi-vertex field theory in **d = 2** — Allen-Cahn ⊕ Model B ⊕ KPZ.

    ∂_t φ = -μ φ + D ∇²φ
            - λ φ³            (Allen-Cahn, plain φ³ vertex)
            + g ∇²(φ²)         (Model B, composite ∇²(φ²) vertex)
            + (κ/2) (∇φ)²      (KPZ, per-leg gradient vertex)
            + η ,
    (∇φ)² = (∂_x φ)² + (∂_y φ)² ,   ⟨η η⟩ = 2T δ δ ,   d = 2.

Three vertex types in one action: plain ``φ³`` (Allen-Cahn), composite
``∇²(φ²)`` (Model B, ``mode='composite'``), and per-leg ``(∇φ)²`` (KPZ,
``mode='perleg'``) summed by the per-vertex form-factor table, with the
``d=2`` transverse-moment Gauss–Hermite loop average.

The homogeneous saddle is ``φ*=0``: the gradient forcings vanish there
(``∂_x`` of a constant is 0) and the composite/cubic terms vanish, so the
saddle is fixed by the linear part ``-μ φ* = 0``.  The MF-equation parser
handles Dt/Laplacian but not the Dx/composite nodes, so the reduced RHS
``-λ φ³`` (also 0 at the saddle) is written; the gradient/Model-B terms live
only in the action.

**d=2 caveat (physics, expected):** the same-signature Model B × KPZ cross is
UV-divergent in d=2 (as in d=3, just milder), so the *bare* 1-loop is
cutoff-set.  v1 is not a renormalisation package — the deliverable is that the
machinery composes and computes.  See ``docs/spatial_d_ge_2.md``.

Companions: ``theories/kpz_1d.theory.py`` (per-leg single vertex),
``theories/burgers_1d.theory.py`` (composite single vertex).
"""
from pipeline.theory import SpatialTheoryBuilder

# mass, diffusion, φ³ (Allen-Cahn), ∇²(φ²) (Model B), (∇φ)² (KPZ), noise temp
mu, D, lam, g, kpz, T = 1.0, 1.0, 0.1, 0.1, 0.2, 1.0


def build():
    return (
        SpatialTheoryBuilder('allen-cahn+modelB+kpz-2d', n_populations=0)
        .physical_field('phi', spatial_dim=2)                    # ← d = 2
        .parameter('mu',  default=mu,  domain='positive')
        .parameter('D',   default=D,   domain='positive')
        .parameter('lam', default=lam, domain='real')   # φ³  (Allen-Cahn)
        .parameter('g',   default=g,   domain='real')   # ∇²(φ²)  (Model B)
        .parameter('kpz', default=kpz, domain='real')   # (∇φ)²  (KPZ)
        .parameter('T',   default=T,   domain='positive')
        .equation(lhs='(Dt + mu - D*Laplacian)*phi', rhs='-lam*phi^3')
        # (∇φ)² = Dx(phi,0)² + Dx(phi,1)² — the d=2 full gradient
        .set_action_text(
            'phit*(Dt(phi) + mu*phi - D*Lap(phi) + lam*phi^3 - g*Lap(phi^2) '
            '- (kpz/2)*(Dx(phi,0)^2 + Dx(phi,1)^2)) - T*phit^2')
        .operator_ir()
        .boundary('infinite')
        .initial('stationary')
        .build()
    )


DEFAULT_FUNDAMENTAL = {
    'mu':  1.0,
    'D':   1.0,
    'lam': 0.1,
    'g':   0.1,
    'kpz': 0.2,
    'T':   1.0,
}


METADATA = {
    'k_default': 2,
    'ell_default': 1,
    'recommended_external_fields': [('dphi', 1), ('dphi', 1)],
    'tau_max': 0.0,
    'tau_step': 1.0,
    # radial separations r ≥ 0 (xs = np.linspace(0, 5, 15))
    'spatial_grid': [0.0, 5.0, 15],
}
