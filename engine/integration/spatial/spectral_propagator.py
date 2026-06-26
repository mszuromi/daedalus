"""
engine.integration.spatial.spectral_propagator
==============================================
Step 1 of the Dyson–Duhamel integration (``docs/dyson_duhamel_integration_plan.md``):
the **spectral coupled-field reference propagator** ``G₀`` for the new
``SpatialTheoryBuilder`` machinery.  Self-contained numeric core — NOT yet wired into
the production propagator path (``heat_kernel.py`` still hard-gates to the diagonal
case); the wiring is a later increment.

Setup (paper Appendix B §B.15–B.23).  For an ``N``-component field the linearized
inverse propagator is

    K(ω, k) = −iω·I + M + 𝒟·|k|²,
        M = reaction (mass) matrix = diag(μ_i) − A⁽⁰⁾     (need not be diagonal),
        𝒟 = diffusion matrix      = diag(D_i) + A⁽²⁾       (need not be ∝ I).

Because ``M`` and ``𝒟`` need not commute (which would force a full matrix heat kernel
``e^{−𝒟|k|²t}``), we split off a **scalar reference diffusion**

    𝒟 = D₀·I + 𝒟̂        (D₀ ∈ ℝ;  𝒟̂ = residual, = 0 iff 𝒟 ∝ I),

so the reference kernel ``K₀ = −iω·I + M + D₀|k|²·I`` has scalar diffusion that
commutes with ``M``.  Diagonalizing ``M = Σ_α m_α P_α`` with spectral projectors
``P_α`` (``Σ_α P_α = I``, ``P_α P_β = δ_αβ P_α``), the retarded REFERENCE propagator is

    G₀(t, k) = Θ(t) · Σ_α P_α · e^{−(m_α + D₀|k|²) t}                 (eq. B23)
             = Θ(t) · e^{−M t} · e^{−D₀|k|² t}.

``G₀`` is the **n = 0 term** of the Dyson–Duhamel series; the ``𝒟̂`` corrections
(``n ≥ 1``) are layered on in step 3.  Two exactness facts (both validated in
``tests/test_spectral_propagator.py``):

  * **𝒟̂ = 0** (scalar diffusion, possibly coupled ``M``): ``G₀`` is the **exact**
    full propagator ``e^{−(M + D₀|k|²)t}`` — no Dyson series needed.  This already
    unlocks coupled-reaction / equal-diffusion theories.
  * **M and 𝒟 both diagonal**: ``G₀`` reduces to the per-field scalar heat kernel
    ``e^{−(μ_i + D_i|k|²)t}`` the current pipeline builds (``heat_kernel.py``).

Only the generic **diagonalizable** ``M`` is handled here; the defective case
(repeated eigenvalues with non-trivial Jordan blocks) would need the resolvent /
confluent form and is deferred.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import expm, solve_continuous_lyapunov

# Conditioning above which the eigenvector matrix is treated as too close to
# defective for a reliable spectral projector decomposition.
_COND_CAP = 1e10


def split_reference_diffusion(D_mat, D0: float | None = None):
    """Split ``𝒟 = D₀·I + 𝒟̂``.

    ``D0`` defaults to the isotropic part ``trace(𝒟)/N`` (the mean eigenvalue);
    pass a value to override (e.g. to minimise ``‖𝒟̂‖/D₀`` for Dyson convergence).
    Returns ``(D0, Dhat)`` with ``Dhat`` an ``(N, N)`` array (zero iff ``𝒟 ∝ I``).
    """
    D_mat = np.asarray(D_mat, dtype=float)
    n = D_mat.shape[0]
    if D0 is None:
        D0 = float(np.trace(D_mat) / n)
    Dhat = D_mat - D0 * np.eye(n)
    return float(D0), Dhat


def spectral_projectors(M):
    """Eigenvalues ``m_α`` and spectral projectors ``P_α`` of a diagonalizable
    ``M`` (``M = Σ_α m_α P_α``, ``Σ_α P_α = I``, ``P_α P_β = δ_αβ P_α``).

    Returns ``(eigvals (N,) complex, projectors list[(N,N) complex])``.  Raises
    ``ValueError`` if ``M`` is too close to defective (ill-conditioned eigenvectors).
    """
    M = np.asarray(M, dtype=complex)
    w, V = np.linalg.eig(M)
    cond = np.linalg.cond(V)
    if not np.isfinite(cond) or cond > _COND_CAP:
        raise ValueError(
            f'reaction matrix M is (near-)defective: eigenvector condition '
            f'number {cond:.3e} > {_COND_CAP:.0e}.  The spectral projector '
            f'decomposition needs a diagonalizable M; the defective/confluent '
            f'case (resolvent form) is deferred.')
    Vinv = np.linalg.inv(V)
    proj = [np.outer(V[:, a], Vinv[a, :]) for a in range(len(w))]
    return w, proj


@dataclass
class SpectralReference:
    """Cached spectral data for the reference propagator ``G₀``."""
    M: np.ndarray          # reaction matrix (N, N)
    D: np.ndarray          # full diffusion matrix (N, N)
    D0: float              # scalar reference diffusion
    Dhat: np.ndarray       # residual diffusion 𝒟̂ = 𝒟 − D₀·I (0 ⇒ G₀ exact)
    eigvals: np.ndarray    # m_α (N,)
    projectors: list       # [P_α]  (N, N) each

    @property
    def n_fields(self) -> int:
        return self.M.shape[0]

    @property
    def is_scalar_diffusion(self) -> bool:
        """True ⇒ 𝒟̂ = 0 ⇒ ``G₀`` is the EXACT full propagator (no Dyson needed)."""
        return bool(np.allclose(self.Dhat, 0.0))

    def G0(self, ksq: float, t: float) -> np.ndarray:
        """Reference propagator ``G₀(t, k) = Σ_α P_α e^{−(m_α + D₀·ksq)·t}``
        (matrix; caller applies ``Θ(t)``).  ``ksq = |k|²``."""
        return reference_propagator(self.eigvals, self.projectors, self.D0, ksq, t)


def build_reference(M, D, D0: float | None = None) -> SpectralReference:
    """Assemble the :class:`SpectralReference` from the reaction matrix ``M`` and
    diffusion matrix ``D`` (numeric ``(N, N)``).  ``D0`` overrides the default
    isotropic reference ``trace(D)/N``."""
    M = np.asarray(M, dtype=float)
    D = np.asarray(D, dtype=float)
    if M.shape != D.shape or M.ndim != 2 or M.shape[0] != M.shape[1]:
        raise ValueError(f'M and D must be square and same shape; got '
                         f'{M.shape} and {D.shape}.')
    d0, dhat = split_reference_diffusion(D, D0)
    w, proj = spectral_projectors(M)
    return SpectralReference(M=M, D=D, D0=d0, Dhat=dhat, eigvals=w, projectors=proj)


def reference_propagator(eigvals, projectors, D0: float, ksq: float,
                         t: float) -> np.ndarray:
    """``G₀(t, k) = Σ_α P_α e^{−(m_α + D₀·ksq)·t}`` (matrix).  Equivalent to
    ``e^{−M t}·e^{−D₀·ksq·t}``; for ``ksq=0`` it is ``e^{−M t}``."""
    G = np.zeros_like(projectors[0], dtype=complex)
    for m_a, P_a in zip(eigvals, projectors):
        G = G + P_a * np.exp(-(m_a + D0 * ksq) * t)
    return G


# ── Tree-level coupled 2-point: matrix Lyapunov / FDT (Dyson step 3a) ──────────
#
# The free 2-point of a coupled linear theory is NOT a sum of independent OU modes
# (that diagonal form is what pipeline_bridge._modes_C_q_tau builds).  With the
# relaxation matrix A(q) = M + 𝒟|q|² and the (symmetric) noise covariance N (the
# response-field (2,0) sector, ⟨ξξᵀ⟩ = N), the stationary covariance solves the
# Lyapunov equation and the 2-point follows by the fluctuation–regression theorem:
#
#     A(q) Σ(q) + Σ(q) A(q)ᵀ = N,           (stationary covariance)
#     C(q, τ) = e^{−A(q)|τ|} Σ(q)   (τ ≥ 0),   C(q, −τ) = Σ(q) e^{−A(q)ᵀ|τ|}.
#
# For SCALAR diffusion (𝒟 = D₀ I, i.e. ref.is_scalar_diffusion) A(q)=M+D₀|q|²I
# shares M's eigenprojectors, so e^{−A|τ|} = G₀(|q|², |τ|) is EXACT — no Dyson
# series.  Unequal diffusion (𝒟̂≠0) needs the n≥1 corrections and is rejected here.


def lyapunov_covariance(A, N) -> np.ndarray:
    """Stationary covariance ``Σ`` solving the Lyapunov equation
    ``A Σ + Σ Aᵀ = N`` (steady state of ``dx = −A x dt + ξ``, ``⟨ξξᵀ⟩ = N δ``).

    ``A`` must be a stable relaxation matrix (eigenvalues with positive real
    part); ``N`` is the symmetric noise covariance.  Uses
    ``scipy.linalg.solve_continuous_lyapunov`` (which solves ``A X + X Aᴴ = Q``;
    for the real ``A`` here ``Aᴴ = Aᵀ``)."""
    A = np.asarray(A, dtype=float)
    N = np.asarray(N, dtype=float)
    return solve_continuous_lyapunov(A, N)


def coupled_two_point(ref: SpectralReference, N, qsq: float,
                      tau: float) -> np.ndarray:
    """Free 2-point **matrix** ``C(q, τ)`` for a SCALAR-diffusion coupled theory.

    ``A(q) = M + D₀·qsq·I``; ``Σ`` solves ``A Σ + Σ Aᵀ = N``;
    ``C(q,τ) = e^{−A|τ|} Σ`` (``τ ≥ 0``), ``= Σ e^{−Aᵀ|τ|}`` (``τ < 0``, i.e.
    ``C(q,τ) = C(q,|τ|)ᵀ``).  Exact when ``ref.is_scalar_diffusion``; raises for
    unequal diffusion (that needs the Dyson–Duhamel series).  Returns a real
    ``(N, N)`` array (``C_ij`` = cross-correlation of fields ``i`` and ``j``)."""
    if not ref.is_scalar_diffusion:
        raise ValueError(
            'coupled_two_point needs scalar diffusion (𝒟̂ = 0); unequal '
            'diffusion requires the Dyson–Duhamel series (not yet wired).')
    n = ref.n_fields
    A = np.asarray(ref.M, dtype=float) + ref.D0 * float(qsq) * np.eye(n)
    Sigma = lyapunov_covariance(A, N)
    G = expm(-A * abs(float(tau)))                 # e^{−A|τ|}  (= G₀ for scalar 𝒟)
    return G @ Sigma if tau >= 0 else Sigma @ G.T


# ── Φ_n divided-difference evaluator (Dyson step D-1) ─────────────────────────
#
# The one genuinely new primitive of the Dyson–Duhamel dressing
# (``docs/dyson_duhamel_integration_plan.md``): the nested-simplex time integral
#
#     Φ_n(t; ν_1,…,ν_n) = ∫_{σ_n} tⁿ · e^{−t·Σᵢ uᵢ νᵢ} d𝐮,        Φ_0(t) = 1,
#
# over the standard simplex σ_n = {uᵢ ≥ 0, Σ uᵢ ≤ 1}.  It multiplies the
# projector strings in the 𝓗_n(w) assembly (paper Appendix B eq. B27, step D-2):
#
#     𝓗_n(t) = Σ_{α_0..α_n} P_{α_0}𝒟̂P_{α_1}⋯𝒟̂P_{α_n} · e^{−m_{α_0}t}
#               · Φ_n(t; m_{α_1}−m_{α_0}, …, m_{α_n}−m_{α_0}).
#
# By the Hermite–Genocchi formula with ``f(z) = e^{−tz}`` (so that
# ``f⁽ⁿ⁾(z) = (−t)ⁿ e^{−tz}``) and nodes ``{0, ν_1, …, ν_n}``, Φ_n is — up to
# sign — the n-th DIVIDED DIFFERENCE of ``f``, which we evaluate
# confluent-safely (no 0/0 at coincident or near-coincident nodes) via the
# **Opitz theorem**: ``f`` applied to the upper-bidiagonal matrix with the nodes
# on the diagonal and ones on the superdiagonal has the divided difference as
# its corner entry.


def _opitz_bidiagonal(nus: np.ndarray) -> np.ndarray:
    """Opitz matrix ``Z`` for nodes ``{0, ν_1, …, ν_n}``: the ``(n+1)×(n+1)``
    upper-bidiagonal with diagonal ``(0, ν_1, …, ν_n)`` and ones on the
    superdiagonal, so ``f(Z)[0, n] = f[0, ν_1, …, ν_n]`` for analytic ``f``."""
    n = nus.size
    Z = np.diag(np.ones(n, dtype=complex), k=1)
    Z[np.arange(1, n + 1), np.arange(1, n + 1)] = nus
    return Z


def phi_n(t: float, nus) -> complex:
    """The Dyson step D-1 primitive ``Φ_n(t; ν_1,…,ν_n)`` (a single ``t``).

    **Integral definition** (nested simplex ``σ_n = {uᵢ ≥ 0, Σ uᵢ ≤ 1}``):

        Φ_n(t; ν) = ∫_{σ_n} tⁿ · e^{−t·Σᵢ uᵢ νᵢ} d𝐮,        Φ_0(t) = 1.

    **Hermite–Genocchi**: with ``f(z) = e^{−tz}`` and nodes ``{0, ν_1, …, ν_n}``
    the formula ``f[x_0,…,x_n] = ∫_{σ_n} f⁽ⁿ⁾(x_0 + Σ uᵢ(xᵢ−x_0)) d𝐮`` gives

        Φ_n(t; ν) = (−1)ⁿ · f[0, ν_1, …, ν_n]    (n-th divided difference).

    **Opitz construction** (confluent-safe — exact at coincident nodes, stable
    at near-coincident ones; no 0/0): build the ``(n+1)×(n+1)``
    upper-bidiagonal ``Z`` with diagonal ``(0, ν_1, …, ν_n)`` and ones on the
    superdiagonal; then

        Φ_n(t; ν) = (−1)ⁿ · expm(−t·Z)[0, n].

    Worked ``n = 1`` check: ``Z = [[0, 1], [0, ν]]`` ⇒
    ``expm(−tZ)[0, 1] = −t·(1 − e^{−tν})/(tν) = −(1 − e^{−tν})/ν``; times
    ``(−1)`` ⇒ ``Φ_1 = (1 − e^{−tν})/ν``  ✓ (= ∫_0^1 t·e^{−tuν} du).

    ``t ≥ 0``; ``nus`` is a sequence (possibly empty ⇒ ``Φ_0 = 1``) of real or
    COMPLEX nodes — the eigenvalue differences ``m_{α_i} − m_{α_0}`` of a real
    ``M`` with complex spectrum come in conjugate pairs, so complex support is
    required.  Always returns a python ``complex``; callers decide whether to
    take the real part.  This is the time-side primitive feeding the
    ``𝓗_n(w)`` assembly of the Dyson–Duhamel dressing (paper Appendix B
    eq. B27, step D-2)."""
    nus = np.asarray(nus, dtype=complex).ravel()
    n = nus.size
    if n == 0:
        return complex(1.0)
    Z = _opitz_bidiagonal(nus)
    return complex((-1) ** n * expm(-float(t) * Z)[0, n])


def phi_n_batch(ts, nus) -> np.ndarray:
    """Vectorized :func:`phi_n` over an array of ``t`` values (shared nodes).

    ``ts`` is array-like (any shape); returns a complex array of
    ``Φ_n(t; ν_1,…,ν_n)`` with the same shape.  One ``expm(−t·Z)`` per ``t``
    on the shared ``(n+1)×(n+1)`` Opitz bidiagonal — tiny matrices for the
    practical ``n ≤ ~4`` truncation orders, so correctness over speed (see
    :func:`phi_n` for the integral / Hermite–Genocchi / Opitz math)."""
    ts = np.asarray(ts, dtype=float)
    nus = np.asarray(nus, dtype=complex).ravel()
    n = nus.size
    if n == 0:
        return np.ones(ts.shape, dtype=complex)
    Z = _opitz_bidiagonal(nus)
    sign = (-1) ** n
    out = np.array([sign * expm(-float(t) * Z)[0, n] for t in ts.ravel()],
                   dtype=complex)
    return out.reshape(ts.shape)
