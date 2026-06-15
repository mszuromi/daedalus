"""
Combined model in **d = 3** — Allen-Cahn ⊕ Model B ⊕ KPZ, the
full multi-derivative-vertex stress theory for the spatial pipeline.

    ∂_t φ = -μ φ + D ∇²φ
            - λ φ³                  (Allen-Cahn — plain φ³ vertex)
            + g ∇²(φ²)              (Model B — composite ∇² vertex)
            + (κ/2)(∇φ)²            (KPZ — per-leg ∂ vertex, summed over 3 axes)
            + η ,
    ⟨η(x,t) η(x',t')⟩ = 2T δ(x-x') δ(t-t'),   (∇φ)² = Σ_{i=0}^{2} (∂_i φ)²,  d = 3.

This exercises **everything at once**: three vertex types (a plain ``φ³``, a
composite ``∇²(φ²)``, and a per-leg ``(∇φ)²``), the per-vertex form-factor table
(mixed ``composite``/``perleg`` modes in one theory), and the d=3
transverse-moment loop average (the ``L·d``-dim Gauss–Hermite, validated vs
brute ``∫d³ℓ``).

**d=3 is UV-strong.**  The free ``⟨φ²⟩`` is already UV-divergent (cutoff-set),
and the same-signature Model B × KPZ cross makes the bare 1-loop *even more*
cutoff-sensitive — so the bare numbers are cutoff-dependent (they need
renormalisation).  The deliverable is that the machinery composes and computes.
See ``docs/spatial_d_ge_2.md``.

This reproduces, verbatim, the model that was built inline in
``notebooks/spatial/pipeline_combined_allencahn_modelb_kpz_3d_sim_compare.ipynb``.
"""
from pipeline.theory import TheoryBuilder

# mass, diffusion, φ³ (Allen-Cahn), ∇²(φ²) (Model B), (∇φ)² (KPZ), noise temp
mu, D, lam, g, kpz, T = 1.0, 1.0, 0.1, 0.1, 0.2, 1.0


def build():
    return (TheoryBuilder('allen-cahn+modelB+kpz-3d', n_populations=0)
            .physical_field('phi', spatial_dim=3)                    # ← d = 3
            .parameter('mu',  default=mu,  domain='positive')
            .parameter('D',   default=D,   domain='positive')
            .parameter('lam', default=lam, domain='real')   # φ³  (Allen-Cahn)
            .parameter('g',   default=g,   domain='real')   # ∇²(φ²)  (Model B)
            .parameter('kpz', default=kpz, domain='real')   # (∇φ)²  (KPZ)
            .parameter('T',   default=T,   domain='positive')
            .equation(lhs='(Dt + mu - D*Laplacian)*phi', rhs='-lam*phi^3')
            # (∇φ)² = Σ_i Dx(phi,i)²  — the d=3 full gradient (per-axis perleg vertices)
            .set_action_text(
                'phit*(Dt(phi) + mu*phi - D*Lap(phi) + lam*phi^3 - g*Lap(phi^2) '
                '- (kpz/2)*(Dx(phi,0)^2 + Dx(phi,1)^2 + Dx(phi,2)^2)) - T*phit^2')
            .operator_ir()
            .boundary('infinite').initial('stationary').build())


DEFAULT_FUNDAMENTAL = {
    'mu':  mu,
    'D':   D,
    'lam': lam,
    'g':   g,
    'kpz': kpz,
    'T':   T,
}


METADATA = {
    'k_default': 2,
    'ell_default': 1,                       # 0 = tree, 1 = +1-loop (d=3 loops are heavier)
    'recommended_external_fields': [('dphi', 1), ('dphi', 1)],
    'tau_max': 0.0,                          # equal-time only (τ=0)
    'tau_step': 1.0,
    'spatial_grid': [0.0, 5.0, 13],          # np.linspace(0.0, 5.0, 13): radial r ≥ 0
}
