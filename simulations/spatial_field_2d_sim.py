"""
2D spectral exponential-Euler (ETD1) simulator for a scalar stochastic field on a
periodic square (L×L, N×N grid) — the **d=2 backend-C validation oracle**.

Integrates

    ∂_t φ(x,t) = -μ φ + D ∇²φ - g φ² - λ φ³ + η,
    ⟨η(x,t) η(x',t')⟩ = 2T δ²(x - x') δ(t - t'),

with the linear + noise part propagated EXACTLY per Fourier mode (an
Ornstein-Uhlenbeck update — no dt bias), the nonlinear forcing via the ETD1
integrating factor.  Modeled on the validated 1D sim
(``simulations/spatial_field_1d_sim.py``).

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
                g_lap=0.0, lam_kpz=0.0,
                dt=None, n_steps=60000, burn_in=10000, record_every=20,
                seed=1234):
    """Run the 2D simulator → ``(snaps, meta)``.

    Forcings (added to ∂_tφ): ``−gφ²`` / ``−λφ³`` (plain), conserved Model-B
    ``+g_lap·∇²(φ²)`` (the composite ∇), and KPZ ``+(λ_kpz/2)(∇φ)²`` (the full
    gradient ``(∂_xφ)²+(∂_yφ)²`` — per-axis).

    snaps : (n_rec, N, N) recorded fields (post burn-in).
    meta  : dict with dx, dt, params, and ``k_max = π/dx`` (the physical UV cutoff).
    """
    N = int(N); n_steps = int(n_steps); burn_in = int(burn_in)
    record_every = int(record_every); seed = int(seed)
    dx = L / N
    _nl = (g != 0.0 or lam != 0.0 or g_lap != 0.0 or lam_kpz != 0.0)
    if dt is None:
        dt = min(0.02 / mu, 0.05) if _nl else 0.05
    rng = np.random.default_rng(seed)

    omega = _dispersion_2d(N, dx, mu, D)          # (N,N), >0
    decay = np.exp(-omega * dt)
    etd1 = np.where(omega * dt > 1e-12, (1.0 - decay) / omega, dt)
    stat_var = T * N ** 4 / (L ** 2 * omega)      # ⟨|φ̂_k|²⟩ stationary (d=2)
    inc_std = np.sqrt((1.0 - decay ** 2) * stat_var / N ** 2)
    # spectral operators on the fft2 grid (FD-consistent with _dispersion_2d):
    lap_eig = -(omega - mu) / D                   # ∇² eigenvalue ≤0 (Model B)
    _kd = 2.0 * np.pi * np.fft.fftfreq(N)
    ikx = 1j * np.sin(_kd)[:, None] / dx          # central-diff ∂_x (axis 0; KPZ)
    iky = 1j * np.sin(_kd)[None, :] / dx          # central-diff ∂_y (axis 1)

    def _run(a, nsteps, rec):
        out = []
        for s in range(nsteps):
            if _nl:
                phi = np.fft.ifft2(a).real
                F = np.fft.fft2(-g * phi ** 2 - lam * phi ** 3)
                if g_lap != 0.0:                  # Model B: +g·∇²(φ²)
                    F = F + g_lap * lap_eig * np.fft.fft2(phi ** 2)
                if lam_kpz != 0.0:                # KPZ: +(λ/2)[(∂ₓφ)²+(∂_yφ)²]
                    dxphi = np.fft.ifft2(ikx * a).real
                    dyphi = np.fft.ifft2(iky * a).real
                    F = F + 0.5 * lam_kpz * np.fft.fft2(dxphi ** 2 + dyphi ** 2)
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
            'g': g, 'lam': lam, 'g_lap': g_lap, 'lam_kpz': lam_kpz, 'T': T,
            'spatial_dim': 2, 'record_every': record_every,
            'n_rec': snaps.shape[0], 'k_max': np.pi / dx}
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


def space_time_correlator_2d(snaps, meta, max_lag=None, n_lags=None,
                             connected=True):
    """Full space-time correlator ``C(χ¹,χ²,τ) = ⟨φ(x,t) φ(x+χ,t+τ)⟩`` on the 2-D
    ring, averaged over x and t.

    Returns ``(tau, chi, C)`` where ``chi`` is the CENTRED lattice axis (length
    N, χ=0 at the middle), ``C`` has shape ``(n_tau, N, N)`` with
    ``C[i, a, b]`` the correlator at ``χ¹=chi[a], χ²=chi[b], τ=tau[i]``, and
    ``tau`` is the symmetric lag grid (C is even in τ by stationarity + parity,
    so τ<0 is mirrored from τ≥0).  ``connected`` subtracts ⟨φ⟩².  Its ``τ=0``
    plane equals the equal-time 2-D correlator (cf. :func:`radial_correlator_2d`),
    and it is directly comparable to ``compute_cumulants``' d=2 ``C(χ¹,χ²,τ)``.
    Lag spacing is the recording interval ``record_every·dt``."""
    snaps = np.asarray(snaps)
    n_rec, N, _ = snaps.shape
    dt_rec = float(meta['record_every']) * float(meta['dt'])
    dx = float(meta['L']) / N
    if n_lags is None:
        n_lags = (int(round(float(max_lag) / dt_rec)) + 1 if max_lag is not None
                  else max(1, min(n_rec // 4, 128)))
    n_lags = int(max(1, min(n_lags, n_rec - 1)))
    mean = float(snaps.mean()) if connected else 0.0
    F = np.fft.fft2(snaps - mean, axes=(1, 2))          # (n_rec, N, N)
    Cpos = np.empty((n_lags, N, N))
    for lag in range(n_lags):
        cps = np.mean(np.conj(F[:n_rec - lag]) * F[lag:], axis=0)
        Cpos[lag] = np.real(np.fft.ifft2(cps)) / N ** 2
    Cpos = np.fft.fftshift(Cpos, axes=(1, 2))           # centre χ=0
    chi = (np.arange(N) - (N // 2)) * dx
    tau_pos = np.arange(n_lags) * dt_rec
    tau = np.concatenate([-tau_pos[:0:-1], tau_pos])    # [-τmax … 0 … τmax]
    C = np.concatenate([Cpos[:0:-1], Cpos], axis=0)     # even in τ → mirror
    return tau, chi, C
