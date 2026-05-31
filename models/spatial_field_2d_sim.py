"""
2D spectral exponential-Euler (ETD1) simulator for a scalar stochastic field on a
periodic square (L×L, N×N grid) — the **d=2 backend-C validation oracle**.

Integrates

    ∂_t φ(x,t) = -μ φ + D ∇²φ - g φ² - λ φ³ + η,
    ⟨η(x,t) η(x',t')⟩ = 2T δ²(x - x') δ(t - t'),

with the linear + noise part propagated EXACTLY per Fourier mode (an
Ornstein-Uhlenbeck update — no dt bias), the nonlinear forcing via the ETD1
integrating factor.  Modeled on the validated 1D sim
(``models/spatial_field_1d_sim.py``).

Noise normalization (d-general): the stationary per-mode variance is
``⟨|φ̂_k|²⟩ = T N^{2d}/(L^d ω_k)`` (here d=2), so the equal-time structure factor
is ``S(k) = T/ω_k``.  The Hermitian structure is automatic: the per-step noise is
``fft2`` of a REAL white field, scaled so ``var(φ̂_k) = (1-e^{-2ω dt})·⟨|φ̂_k|²⟩``.

Self-consistency (λ=g=0): ``S(k)`` equals the exact lattice spectrum ``T/ω_k`` and
the radially-averaged ``C(r)`` → ``(T/2πD) K₀(r√(μ/D))`` in the continuum band.
"""
import numpy as np


def _dispersion_2d(N, dx, mu, D):
    """Finite-difference lattice dispersion on the full N×N fft2 grid:
    ``ω = μ + (2D/dx²)(2 − cos(kx·dx) − cos(ky·dx))``."""
    kd = 2.0 * np.pi * np.fft.fftfreq(N)          # the (k·dx) values, length N
    cx = np.cos(kd)[:, None]
    cy = np.cos(kd)[None, :]
    return mu + (2.0 * D / dx ** 2) * (2.0 - cx - cy)


def simulate_2d(L=20.0, N=64, mu=1.0, D=1.0, T=1.0, g=0.0, lam=0.0,
                dt=None, n_steps=60000, burn_in=10000, record_every=20,
                seed=1234):
    """Run the 2D simulator → ``(snaps, meta)``.

    snaps : (n_rec, N, N) recorded fields (post burn-in).
    meta  : dict with dx, dt, params, and ``k_max = π/dx`` (the physical UV cutoff).
    """
    N = int(N); n_steps = int(n_steps); burn_in = int(burn_in)
    record_every = int(record_every); seed = int(seed)
    dx = L / N
    if dt is None:
        dt = min(0.02 / mu, 0.05) if (g != 0.0 or lam != 0.0) else 0.05
    rng = np.random.default_rng(seed)

    omega = _dispersion_2d(N, dx, mu, D)          # (N,N), >0
    decay = np.exp(-omega * dt)
    etd1 = np.where(omega * dt > 1e-12, (1.0 - decay) / omega, dt)
    stat_var = T * N ** 4 / (L ** 2 * omega)      # ⟨|φ̂_k|²⟩ stationary (d=2)
    # per-step OU increment: fft2 of REAL unit white has ⟨|ξ̂_k|²⟩ = N² (sites),
    # so scaling by √(inc_var/N²) gives the per-mode increment variance and keeps
    # φ real automatically (ξ̂ Hermitian).
    inc_std = np.sqrt((1.0 - decay ** 2) * stat_var / N ** 2)

    def _run(a, nsteps, rec):
        out = []
        for s in range(nsteps):
            if g != 0.0 or lam != 0.0:
                phi = np.fft.ifft2(a).real
                F = np.fft.fft2(-g * phi ** 2 - lam * phi ** 3)
            else:
                F = 0.0
            xi = np.fft.fft2(rng.standard_normal((N, N)))
            a = decay * a + etd1 * F + inc_std * xi
            if rec and (s + 1) % rec == 0:
                out.append(np.fft.ifft2(a).real.copy())
        return a, (np.array(out) if out else np.empty((0, N, N)))

    a = np.fft.fft2(np.zeros((N, N)))
    if burn_in > 0:
        a, _ = _run(a, burn_in, 0)
    a, snaps = _run(a, n_steps, record_every)
    meta = {'dx': dx, 'dt': dt, 'L': L, 'N': N, 'mu': mu, 'D': D,
            'g': g, 'lam': lam, 'T': T, 'spatial_dim': 2,
            'record_every': record_every, 'n_rec': snaps.shape[0],
            'k_max': np.pi / dx}
    return snaps, meta


def structure_factor_2d(snaps, meta):
    """Per-mode equal-time structure factor ``S(k) = ⟨|φ̂_k|²⟩·(L²/N⁴)`` on the
    full fft2 grid (so the continuum tree gives ``S → T/(μ+D|k|²)``).  Returns
    ``(kx, ky, S)`` with ``kx, ky`` the fftfreq momentum axes (length N)."""
    L, N = meta['L'], int(meta['N'])
    F = np.fft.fft2(snaps, axes=(1, 2))           # (n_rec, N, N)
    S = np.mean(np.abs(F) ** 2, axis=0) * (L ** 2 / N ** 4)
    k = 2.0 * np.pi * np.fft.fftfreq(N, d=L / N)
    return k, k, S


def radial_structure_factor_2d(snaps, meta, n_bins=40):
    """Radially-binned ``S(|k|)`` → ``(k_centers, S_radial)`` (isotropic average)."""
    kx, ky, S = structure_factor_2d(snaps, meta)
    KX, KY = np.meshgrid(kx, ky, indexing='ij')
    kmag = np.sqrt(KX ** 2 + KY ** 2).ravel()
    Sf = S.ravel()
    kmax = float(meta['k_max'])
    edges = np.linspace(0.0, kmax, n_bins + 1)
    idx = np.digitize(kmag, edges) - 1
    kc = 0.5 * (edges[:-1] + edges[1:])
    Sr = np.array([Sf[idx == b].mean() if np.any(idx == b) else np.nan
                   for b in range(n_bins)])
    return kc, Sr


def radial_correlator_2d(snaps, meta, n_bins=40, r_max=None):
    """Real-space equal-time correlator ``C(r) = ⟨φ(0)φ(r)⟩`` radially averaged.

    Uses the circular autocorrelation (FFT) per snapshot, averaged, then bins by
    separation ``r=√(Δx²+Δy²)``.  ``C(0)=⟨φ²⟩``.  Returns ``(r_centers, C_radial)``
    — directly comparable to ``compute_cumulants``' d=2 ``C(r,0)``."""
    L, N = meta['L'], int(meta['N'])
    dx = L / N
    F = np.fft.fft2(snaps, axes=(1, 2))
    C2d = np.mean(np.fft.ifft2(np.abs(F) ** 2, axes=(1, 2)).real, axis=0) / N ** 2
    # separation grid wrapped to (−L/2, L/2]
    s = (np.arange(N) - N * (np.arange(N) > N // 2)) * dx
    SX, SY = np.meshgrid(s, s, indexing='ij')
    rmag = np.sqrt(SX ** 2 + SY ** 2).ravel()
    Cf = C2d.ravel()
    if r_max is None:
        r_max = 0.5 * L
    edges = np.linspace(0.0, r_max, n_bins + 1)
    idx = np.digitize(rmag, edges) - 1
    rc = 0.5 * (edges[:-1] + edges[1:])
    Cr = np.array([Cf[idx == b].mean() if np.any(idx == b) else np.nan
                   for b in range(n_bins)])
    return rc, Cr
