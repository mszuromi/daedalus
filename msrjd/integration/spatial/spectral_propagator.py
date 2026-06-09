"""
msrjd.integration.spatial.spectral_propagator
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
