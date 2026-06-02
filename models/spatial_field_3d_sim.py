"""
3D spectral exponential-Euler (ETD1) simulator for a scalar stochastic field on a
periodic cube (L×L×L, N×N×N grid) — the **d=3 validation oracle**.

Integrates

    ∂_t φ(x,t) = -μ φ + D ∇²φ - g φ² - λ φ³ + g_lap ∇²(φ²) + (λ_kpz/2)(∇φ)² + η,
    ⟨η(x,t) η(x',t')⟩ = 2T δ³(x - x') δ(t - t'),

with the linear + noise part propagated EXACTLY per Fourier mode (an
Ornstein-Uhlenbeck update), the forcing via the ETD1 integrating factor.  A direct
generalization of ``spatial_field_2d_sim.py`` (fftn over the 3 spatial axes).

Noise normalization (d-general): stationary per-mode ``⟨|φ̂_k|²⟩ = T N^{2d}/(L^d ω_k)``
(here d=3), so ``S(k) = T/ω_k``; the per-step noise is ``fftn`` of a REAL white
field scaled so ``var(φ̂_k) = (1-e^{-2ω dt})·⟨|φ̂_k|²⟩`` (Hermitian automatically).
``(∇φ)² = Σ_{i=0,1,2}(∂_iφ)²`` is the full-gradient KPZ nonlinearity.
"""
import numpy as np


def _dispersion_3d(N, dx, mu, D):
    """FD lattice dispersion on the N×N×N fftn grid:
    ``ω = μ + (2D/dx²)(3 − cos(kx·dx) − cos(ky·dx) − cos(kz·dx))``."""
    kd = 2.0 * np.pi * np.fft.fftfreq(N)
    cx = np.cos(kd)[:, None, None]
    cy = np.cos(kd)[None, :, None]
    cz = np.cos(kd)[None, None, :]
    return mu + (2.0 * D / dx ** 2) * (3.0 - cx - cy - cz)


def simulate_3d(L=12.0, N=24, mu=1.0, D=1.0, T=1.0, g=0.0, lam=0.0,
                g_lap=0.0, lam_kpz=0.0,
                dt=None, n_steps=40000, burn_in=8000, record_every=20,
                seed=1234):
    """Run the 3D simulator → ``(snaps, meta)``.

    Forcings (added to ∂_tφ): ``−gφ²`` / ``−λφ³`` (plain), conserved Model-B
    ``+g_lap·∇²(φ²)``, and KPZ ``+(λ_kpz/2)(∇φ)² = (λ_kpz/2)Σ_i(∂_iφ)²``.

    snaps : (n_rec, N, N, N) recorded fields (post burn-in).
    meta  : dict with dx, dt, params, and ``k_max = π/dx``.
    """
    N = int(N); n_steps = int(n_steps); burn_in = int(burn_in)
    record_every = int(record_every); seed = int(seed)
    dx = L / N
    _nl = (g != 0.0 or lam != 0.0 or g_lap != 0.0 or lam_kpz != 0.0)
    if dt is None:
        dt = min(0.02 / mu, 0.05) if _nl else 0.05
    rng = np.random.default_rng(seed)

    omega = _dispersion_3d(N, dx, mu, D)          # (N,N,N), >0
    decay = np.exp(-omega * dt)
    etd1 = np.where(omega * dt > 1e-12, (1.0 - decay) / omega, dt)
    stat_var = T * N ** 6 / (L ** 3 * omega)      # ⟨|φ̂_k|²⟩ stationary (d=3)
    inc_std = np.sqrt((1.0 - decay ** 2) * stat_var / N ** 3)
    lap_eig = -(omega - mu) / D                   # ∇² eigenvalue ≤0 (Model B)
    _kd = 2.0 * np.pi * np.fft.fftfreq(N)
    ikx = 1j * np.sin(_kd)[:, None, None] / dx    # central-diff ∂ per axis (KPZ)
    iky = 1j * np.sin(_kd)[None, :, None] / dx
    ikz = 1j * np.sin(_kd)[None, None, :] / dx

    def _run(a, nsteps, rec):
        out = []
        for s in range(nsteps):
            if _nl:
                phi = np.fft.ifftn(a).real
                F = np.fft.fftn(-g * phi ** 2 - lam * phi ** 3)
                if g_lap != 0.0:
                    F = F + g_lap * lap_eig * np.fft.fftn(phi ** 2)
                if lam_kpz != 0.0:
                    dxphi = np.fft.ifftn(ikx * a).real
                    dyphi = np.fft.ifftn(iky * a).real
                    dzphi = np.fft.ifftn(ikz * a).real
                    F = F + 0.5 * lam_kpz * np.fft.fftn(
                        dxphi ** 2 + dyphi ** 2 + dzphi ** 2)
            else:
                F = 0.0
            xi = np.fft.fftn(rng.standard_normal((N, N, N)))
            a = decay * a + etd1 * F + inc_std * xi
            if rec and (s + 1) % rec == 0:
                out.append(np.fft.ifftn(a).real.copy())
        return a, (np.array(out) if out else np.empty((0, N, N, N)))

    a = np.fft.fftn(np.zeros((N, N, N)))
    if burn_in > 0:
        a, _ = _run(a, burn_in, 0)
    a, snaps = _run(a, n_steps, record_every)
    meta = {'dx': dx, 'dt': dt, 'L': L, 'N': N, 'mu': mu, 'D': D,
            'g': g, 'lam': lam, 'g_lap': g_lap, 'lam_kpz': lam_kpz, 'T': T,
            'spatial_dim': 3, 'record_every': record_every,
            'n_rec': snaps.shape[0], 'k_max': np.pi / dx}
    return snaps, meta


def radial_correlator_3d(snaps, meta, n_bins=30, r_max=None):
    """Real-space equal-time ``C(r)=⟨φ(0)φ(r)⟩`` radially averaged (FFT circular
    autocorrelation per snapshot, binned by ``r=√(Δx²+Δy²+Δz²)``).  ``C(0)=⟨φ²⟩``.
    Returns ``(r_centers, C_radial)`` — comparable to compute_cumulants' d=3 C(r,0)."""
    L, N = meta['L'], int(meta['N'])
    dx = L / N
    F = np.fft.fftn(snaps, axes=(1, 2, 3))
    C3d = np.mean(np.fft.ifftn(np.abs(F) ** 2, axes=(1, 2, 3)).real, axis=0) / N ** 3
    s = (np.arange(N) - N * (np.arange(N) > N // 2)) * dx
    SX, SY, SZ = np.meshgrid(s, s, s, indexing='ij')
    rmag = np.sqrt(SX ** 2 + SY ** 2 + SZ ** 2).ravel()
    Cf = C3d.ravel()
    if r_max is None:
        r_max = 0.5 * L
    edges = np.linspace(0.0, r_max, n_bins + 1)
    idx = np.digitize(rmag, edges) - 1
    rc = 0.5 * (edges[:-1] + edges[1:])
    Cr = np.array([Cf[idx == b].mean() if np.any(idx == b) else np.nan
                   for b in range(n_bins)])
    return rc, Cr
