"""
Spectral exponential-Euler (ETD1) simulator for a 1D scalar stochastic
field on a ring (periodic boundary) — the spatial-v1 cross-check
simulator.

Integrates

    ∂_t φ(x,t) = -μ φ + D ∂_x² φ - λ φ³ + η(x,t),
    ⟨η(x,t) η(x',t')⟩ = 2T δ(x - x') δ(t - t'),

on a uniform periodic grid of N points, spacing dx = L/N.  The linear
+ noise part is propagated EXACTLY per Fourier mode (an
Ornstein-Uhlenbeck update), so the λ=0 stationary statistics are
unbiased in the time step dt — unlike plain Euler-Maruyama, which
inflates the fast (large-k) modes by O(dt·ω_max).  The nonlinear
forcing ``-λφ³`` is added via the ETD1 integrating factor.

Matches ``theories/allen_cahn_1d_subcritical_pbc.theory.py`` (and the
infinite-domain theory in the large-L limit) with action
``phit·((Dt + μ - D·Laplacian)φ + λφ³) - T·phit²``.

Self-consistency reference (λ=0): the stationary equal-time variance
equals the discretized lattice sum

    ⟨φ_j²⟩ = (T/L) Σ_n 1 / [ μ + (2D/dx²)(1 - cos(k_n dx)) ],
    k_n = 2π n / L,

which → the continuum ``(T/2√(μD)) coth((L/2)√(μ/D))`` as dx → 0.
The exponential integrator reproduces this lattice sum to sampling
noise (no dt bias).
"""
import numpy as np


def _dispersion(N, dx, mu, D):
    """Finite-difference lattice dispersion ω_k = μ + (2D/dx²)(1-cos)
    on the rfft grid (length N//2+1)."""
    M = N // 2 + 1
    m = np.arange(M)
    # finite-difference Laplacian eigenvalue: -(2/dx²)(1-cos(2πm/N))
    return mu + (2.0 * D / dx**2) * (1.0 - np.cos(2.0 * np.pi * m / N))


def _evolve(phi, n_steps, dt, mu, D, lam, T, dx, record_every, rng, g=0.0,
            g_lap=0.0, lam_burg=0.0, lam_kpz=0.0):
    """Spectral exponential-Euler (ETD1) integrator.

    Linear + noise part is propagated EXACTLY per Fourier mode (an
    Ornstein-Uhlenbeck update), so the λ=0 stationary statistics are
    UNBIASED in dt — they equal the discretized lattice spectrum
    exactly.  The nonlinear forcing ``-λφ³`` is added via the ETD1
    integrating factor ``(1-e^{-ω dt})/ω``.

    Noise normalization (numpy rfft convention): the full-FFT mode has
    stationary variance ⟨|φ̂_k|²⟩ = T N² / (L ω_k); the per-step OU
    increment has variance (1 - e^{-2 ω_k dt}) times that, split over
    real/imag parts for the interior (complex) modes.
    """
    N = phi.shape[0]
    L = N * dx
    n_rec = n_steps // record_every
    out = np.empty((n_rec, N), dtype=np.float64)

    omega = _dispersion(N, dx, mu, D)          # (M,)
    M = omega.shape[0]
    # finite-difference Laplacian eigenvalue on the rfft grid (≤0): the linear
    # dispersion is ω = μ − D·lap_eig, so lap_eig = −(ω−μ)/D.  Used for the
    # CONSERVED nonlinearity g_lap·∂ₓ²(φ²) (a derivative vertex), spectral form
    # g_lap·lap_eig·rfft(φ²) — the v2 Phase-4d derivative-vertex test forcing.
    lap_eig = -(omega - mu) / D
    # Central-difference FIRST-derivative eigenvalue on the rfft grid
    # (∂_x → i·sin(2πm/N)/dx ≈ i·k), consistent with the FD Laplacian above.
    # Used for the GRADIENT nonlinearities: Burgers −(λ/2)∂_x(φ²) and KPZ
    # +(λ/2)(∂_xφ)².  The Nyquist mode (sin π = 0) is correctly un-forced.
    ik_eig = 1j * np.sin(2.0 * np.pi * np.arange(M) / N) / dx
    decay = np.exp(-omega * dt)
    etd1 = np.where(omega * dt > 1e-12, (1.0 - decay) / omega, dt)
    stat_var = T * N**2 / (L * omega)          # ⟨|φ̂_k|²⟩ stationary
    inc_std = np.sqrt((1.0 - decay**2) * stat_var)   # per-step OU increment

    # rfft real/complex structure: modes 0 and (if N even) N/2 are
    # real; interior modes 1..M-2 are complex (split variance /2).
    is_real_mode = np.zeros(M, dtype=bool)
    is_real_mode[0] = True
    if N % 2 == 0:
        is_real_mode[M - 1] = True

    a = np.fft.rfft(phi)
    ri = 0
    for step in range(n_steps):
        # Nonlinear forcing F = rfft(-g φ² - λ φ³) (skip when both zero).
        # The g φ² term is the φ̃φ² bubble vertex; the λ φ³ term (→ +λφ⁴/4
        # potential) bounds the otherwise-unstable cubic potential.
        if (lam != 0.0 or g != 0.0 or g_lap != 0.0
                or lam_burg != 0.0 or lam_kpz != 0.0):
            phi_r = np.fft.irfft(a, n=N)
            F = np.fft.rfft(-g * phi_r**2 - lam * phi_r**3)
            if g_lap != 0.0:
                # conserved derivative vertex +g_lap·∂ₓ²(φ²) (∂_tφ EOM term)
                F = F + g_lap * lap_eig * np.fft.rfft(phi_r**2)
            if lam_burg != 0.0:
                # Burgers: −(λ/2)∂_x(φ²) → −(λ/2)·ik·rfft(φ²) (composite ∂_x)
                F = F - 0.5 * lam_burg * ik_eig * np.fft.rfft(phi_r**2)
            if lam_kpz != 0.0:
                # KPZ: +(λ/2)(∂_xφ)² (per-leg ∂_x: differentiate THEN square)
                dphi_r = np.fft.irfft(ik_eig * a, n=N)
                F = F + 0.5 * lam_kpz * np.fft.rfft(dphi_r**2)
        else:
            F = 0.0
        # OU noise increment with rfft Hermitian structure.  Vectorized
        # (real modes → inc_std·gr; interior complex modes → inc_std/√2·
        # (gr+i·gi)); draws gr,gi in the same order as the per-mode loop, so
        # the random stream — and hence every result — is bit-identical.
        gr = rng.standard_normal(M)
        gi = rng.standard_normal(M)
        noise = np.where(is_real_mode,
                         inc_std * gr,
                         inc_std / np.sqrt(2.0) * (gr + 1j * gi))
        a = decay * a + etd1 * F + noise
        if (step + 1) % record_every == 0:
            out[ri, :] = np.fft.irfft(a, n=N)
            ri += 1
    return out


def simulate(L=20.0, N=200, mu=1.0, D=1.0, lam=0.0, T=1.0,
             dt=None, n_steps=400000, burn_in=40000, record_every=20,
             seed=12345, g=0.0, g_lap=0.0, lam_burg=0.0, lam_kpz=0.0):
    """Run the simulator and return ``(snapshots, x_grid, meta)``.

    snapshots : (n_rec, N) recorded field configurations (post burn-in)
    x_grid    : (N,) spatial coordinates
    meta      : dict with dx, dt, params

    ``dt`` defaults to a stable explicit value:
    ``0.2 · dx² / D`` capped by ``0.1/μ`` (diffusive CFL + relaxation).
    """
    # Cast counts/seed to Python ints: under the SageMath *kernel* the
    # notebook preparser turns integer literals into Sage ``Integer``,
    # which ``np.random.default_rng`` (and some numpy paths) reject.
    N = int(N)
    n_steps = int(n_steps)
    burn_in = int(burn_in)
    record_every = int(record_every)
    seed = int(seed)
    dx = L / N
    if dt is None:
        # The linear part is exact (exponential integrator), so dt is
        # limited only by the ETD1 nonlinear splitting accuracy, not by
        # the diffusive CFL — a moderate dt suffices.
        _nl = (lam != 0.0 or g != 0.0 or g_lap != 0.0
               or lam_burg != 0.0 or lam_kpz != 0.0)
        dt = min(0.02 / mu, 0.05) if _nl else 0.05
    rng = np.random.default_rng(seed)
    phi0 = np.zeros(N, dtype=np.float64)
    # Burn-in (discarded).
    phi_burn = _evolve(phi0, burn_in, dt, mu, D, lam, T, dx, burn_in, rng,
                       g=g, g_lap=g_lap, lam_burg=lam_burg, lam_kpz=lam_kpz)
    phi_start = phi_burn[-1, :].copy()
    snaps = _evolve(phi_start, n_steps, dt, mu, D, lam, T, dx,
                    record_every, rng, g=g, g_lap=g_lap,
                    lam_burg=lam_burg, lam_kpz=lam_kpz)
    x_grid = np.arange(N) * dx
    meta = {'dx': dx, 'dt': dt, 'L': L, 'N': N, 'mu': mu, 'D': D,
            'lam': lam, 'g': g, 'g_lap': g_lap,
            'lam_burg': lam_burg, 'lam_kpz': lam_kpz, 'T': T,
            'record_every': record_every, 'n_rec': snaps.shape[0],
            # the PHYSICAL UV cutoff this grid imposes (Nyquist): k_max = π/dx =
            # πN/L.  A backend-C loop computed at THIS k_max is directly
            # comparable to this simulation (the "match the cutoff" principle).
            'k_max': np.pi / dx}
    return snaps, x_grid, meta


def structure_factor(snaps, meta):
    """Equal-time structure factor ``S(q) = ⟨|φ_q|²⟩`` on the rfft momentum grid,
    normalized as ``(L/N²)·⟨|FFT φ|²⟩`` (so the continuum tree gives
    ``S(q) → T/(μ+Dq²)``).  Returns ``(q_grid, S)`` with ``q_grid`` up to the
    physical cutoff ``meta['k_max']``.

    This is the matched-cutoff oracle for backend-C milestones III.0/III.2:
    compare a theory ``S(q)`` computed at ``meta['k_max']`` against this.
    """
    L, N = meta['L'], int(meta['N'])
    q_grid = 2.0 * np.pi * np.fft.rfftfreq(N, d=L / N)
    F = np.fft.rfft(snaps, axis=1)               # (n_rec, N//2+1)
    S = np.mean(np.abs(F) ** 2, axis=0) * (L / N ** 2)
    return q_grid, S


def equal_time_correlator(snaps):
    """Estimate C(x) = ⟨φ(x0) φ(x0 + x)⟩ averaged over x0 and time,
    using the periodic translational average.  Returns C of length N
    (C[m] = correlator at separation m·dx)."""
    n_rec, N = snaps.shape
    # Per-snapshot circular autocorrelation via FFT, averaged.
    acc = np.zeros(N)
    for r in range(n_rec):
        f = np.fft.rfft(snaps[r])
        ac = np.fft.irfft(np.abs(f) ** 2, n=N) / N
        acc += ac
    return acc / n_rec


def lattice_sum_variance(L, N, mu, D, T):
    """Discretized-theory exact equal-time variance ⟨φ²⟩ with the
    finite-difference dispersion — the simulator's λ=0 reference."""
    N = int(N)                 # guard against Sage-kernel Integer
    dx = L / N
    ks = 2.0 * np.pi * np.fft.fftfreq(N, d=dx)
    disp = mu + (2.0 * D / dx**2) * (1.0 - np.cos(ks * dx))
    return (T / L) * np.sum(1.0 / disp)
