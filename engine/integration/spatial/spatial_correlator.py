"""
engine.integration.spatial.spatial_correlator
=============================================
Tree-level (Gaussian) spatial two-point correlator ``C(x, τ)`` for
v1 spatial field theories.

For a free / tree-level theory the two-point function is the noise
driving two retarded propagators:

    C_ij(x, τ) = 2 D^noise_ij ∫_{-∞}^{min(t1,t2)} dt_v ∫ dx_v
                 G^R_i(t1 - t_v, x - x_v) G^R_j(t2 - t_v, -x_v).

For a diagonal Allen-Cahn-like propagator (each field an independent
heat-kernel mode with mass ``A_i`` and diffusion ``B_i``) the spatial
``x_v`` integral collapses by the heat-kernel semigroup and the
``t_v`` integral becomes the erf-closed-form

    C_ii(x, τ) = (D^noise_i / √(4π B_i)) ·
                 ∫_{|τ|}^∞ s^(-1/2) exp(-x²/(4 B_i s) - A_i s) ds,

with the integral evaluated by ``heat_kernel.erf_time_integral``
(Rescue A).  Verified 3 ways (closed form vs 2-D quadrature vs the
analytic ``C(x,0) = D^noise/(2√(AB)) e^{-|x|√(A/B)}``) in
``docs/spatial_spikes/phase5_erfsplit_semigroup_spike.py``.

This is the **tree-level / Gaussian** correlator: exact for the free
model, and the leading term for an interacting one (the λφ³ vertex
enters at 1-loop, which is the remaining Phase-5 work — see
``docs/spatial_implementation_plan.md`` §5).  Off-diagonal field
coupling and loops are deferred to v2.
"""
from __future__ import annotations

import cmath
import math

import numpy as np
from sage.all import SR

from engine.integration.spatial.heat_kernel import (
    SpatialPropagatorError, erf_time_integral, gaussian_heat_kernel,
)


# ── Noise extraction ──────────────────────────────────────────────
def _noise_value(coeff, sub, exp_vec):
    """Numeric value of one noise-sector coefficient.  Never silent: a
    coefficient that does not reduce to a number (a missing parameter / saddle
    value, or a momentum-dependent — e.g. conserved — noise the white-noise
    extractors cannot represent) raises, naming the leftover symbols; a
    dropped noise entry would change the correlator with no error."""
    val = SR(coeff).subs(sub)
    try:
        return complex(val)
    except (TypeError, ValueError):
        left = sorted(str(v) for v in val.free_variables())
        raise SpatialPropagatorError(
            f'noise coefficient {coeff} (monomial exponents {tuple(exp_vec)}) '
            f'does not reduce to a number: leftover symbol(s) {left}.  Supply '
            f'every parameter and saddle value; momentum-dependent noise is '
            f'not supported by the spatial white-noise extractors.')


def extract_noise_coefficients(ft, num_params):
    """Return ``{field_index: D_noise}`` — the white-noise spectral
    coefficient per (response) field, read from the action's (2,0)
    bigrade sector (the ``- D^noise · φ̃²`` term).

    Convention (matching the framework's OU models, e.g. action
    ``- D·x̃²`` ⇒ ⟨x²⟩ = D/μ): ``D_noise_i = -coeff(φ̃_i²)``.
    Cross-noise (i≠j, off-diagonal) is read but only the diagonal is
    used by the diagonal-propagator correlator.  ``num_params`` is
    substituted to turn the symbolic coefficient into a float.
    """
    by_tp = getattr(ft, '_by_tp', None)
    if by_tp is None:
        raise SpatialPropagatorError('FieldTheory not expanded (no _by_tp).')
    noise_sector = by_tp.get((2, 0))
    if noise_sector is None:
        return {}

    n_tilde = ft._n_tilde
    n_gens = len(ft.ring().gens())
    sub = {SR.var(str(kk)): vv for kk, vv in num_params.items()}

    # ``noise_sector`` is a polynomial in the FieldTheory ring; iterate
    # its monomials (like build_propagator does on S_free).  The
    # diagonal noise term for response field i is the monomial with
    # exponent 2 on ring position i (response fields are the first
    # n_tilde generators) and 0 elsewhere.
    out = {}
    try:
        terms = noise_sector.dict().items()
    except AttributeError:
        return {}
    for exp_vec, coeff in terms:
        # Identify a pure φ̃_i² monomial.
        nz = [idx for idx in range(n_gens) if exp_vec[idx] != 0]
        if len(nz) == 1 and nz[0] < n_tilde and exp_vec[nz[0]] == 2:
            i = nz[0]
            val = _noise_value(coeff, sub, exp_vec)
            Dn = -val.real       # action term is - D_noise φ̃²
            if abs(Dn) > 1e-300:
                out[i] = Dn
    return out


def extract_noise_matrix(ft, num_params):
    """Return the ``(n_tilde × n_tilde)`` noise covariance **matrix** ``N`` (numpy
    float array), ``⟨ξξᵀ⟩ = N``, from the action's ``(2, 0)`` response-field sector
    — the coupled-field generalization of :func:`extract_noise_coefficients`.

    Convention (calibrated so the diagonal reproduces the scalar OU result
    ``action −D·x̃² ⇒ ⟨x²⟩ = D/μ`` and the Lyapunov 2-point ``Σ = N/2A``):

        N_ii = −2·coeff(φ̃_i²),      N_ij = −coeff(φ̃_i φ̃_j)   (i ≠ j).

    So ``N_ii = 2·κ_i`` with ``κ_i`` the value :func:`extract_noise_coefficients`
    returns.  White noise only: a coefficient that does not reduce to a
    number (missing parameter, or q-dependent noise) raises (it used to be
    left 0 silently).
    """
    by_tp = getattr(ft, '_by_tp', None)
    if by_tp is None:
        raise SpatialPropagatorError('FieldTheory not expanded (no _by_tp).')
    n_tilde = ft._n_tilde
    n_gens = len(ft.ring().gens())
    N = np.zeros((n_tilde, n_tilde), dtype=float)
    noise_sector = by_tp.get((2, 0))
    if noise_sector is None:
        return N
    sub = {SR.var(str(kk)): vv for kk, vv in num_params.items()}
    try:
        terms = noise_sector.dict().items()
    except AttributeError:
        return N
    for exp_vec, coeff in terms:
        nz = [idx for idx in range(n_gens) if exp_vec[idx] != 0]
        c = _noise_value(coeff, sub, exp_vec).real
        if len(nz) == 1 and nz[0] < n_tilde and exp_vec[nz[0]] == 2:
            i = nz[0]
            N[i, i] = -2.0 * c
        elif (len(nz) == 2 and all(p < n_tilde for p in nz)
              and exp_vec[nz[0]] == 1 and exp_vec[nz[1]] == 1):
            i, j = nz
            N[i, j] = N[j, i] = -c
    return N


# ── Free two-point correlator ─────────────────────────────────────
def free_two_point(mu, D, kap, x, tau, bc_mode='infinite', L=None,
                   n_images_cap=200):
    """``C(x, τ)`` for one diagonal field.

    mu : mass (float/complex), D : diffusion (>0), kap : noise
    spectral coefficient.  Infinite domain → erf closed form;
    periodic → image sum over ``x → x + nL`` of the infinite-domain
    correlator (the ring's periodicity).
    """
    D = float(D)
    mu = complex(mu)
    kapf = float(kap)
    pref = kapf / math.sqrt(4.0 * math.pi * D)

    def _C_inf(xx):
        beta = (xx * xx) / (4.0 * D)
        integ = erf_time_integral(mu, beta, abs(tau), U_hi=None)
        return complex(pref * integ)

    if bc_mode == 'infinite':
        return _C_inf(x)
    if bc_mode == 'periodic':
        if L is None:
            raise SpatialPropagatorError('periodic correlator needs L.')
        L = float(L)
        total = _C_inf(x)
        ref = abs(total) if abs(total) > 0 else 1.0
        n = 1
        while n <= n_images_cap:
            tp = _C_inf(x + n * L)
            tm = _C_inf(x - n * L)
            total += tp + tm
            if abs(tp) < 1e-13 * ref and abs(tm) < 1e-13 * ref and n >= 2:
                break
            n += 1
        return complex(total)
    raise SpatialPropagatorError(f'unknown bc_mode {bc_mode!r}')


def heat_time_integral_d(mu, beta, t_lo, spatial_dim, dps=30, abs_tol=0.0):
    """``∫_{t_lo}^∞ s^{−d/2} e^{−μ s − β/s} ds`` (complex ``μ`` with Re μ > 0),
    the time integral behind the free correlator in ``d`` dimensions.

    ``t_lo = 0``: the closed form ``2 (β/μ)^{ν/2} K_ν(2√(μβ))``, ``ν = 1 − d/2``
    (needs ``β > 0`` for ``d ≥ 2``; at ``β = t_lo = 0`` the integral diverges —
    the equal-point variance is UV-divergent for ``d ≥ 2`` — and this raises).
    ``t_lo > 0``: adaptive mpmath quadrature at ``dps`` digits with its error
    estimate checked (raises if it is above 1e-12 relative and above
    ``abs_tol``, the absolute accuracy the caller needs)."""
    import mpmath as mp
    d = int(spatial_dim)
    mu = complex(mu)
    beta = float(beta)
    t_lo = float(t_lo)
    if not mu.real > 0.0:
        raise SpatialPropagatorError(
            f'free correlator needs a stable mass (Re mu > 0); got mu={mu}.')
    with mp.workdps(dps):
        if t_lo == 0.0:
            if beta <= 0.0:
                if d >= 2:
                    raise SpatialPropagatorError(
                        f'equal-point (x=0, tau=0) free correlator is '
                        f'UV-divergent at d={d} (no continuum value; the '
                        f'q-grid used to return a cutoff-dependent number).')
                return complex(mp.gamma(1.0 - 0.5 * d) * mp.mpc(mu) ** (0.5 * d - 1.0))
            nu = mp.mpf(1) - mp.mpf(d) / 2
            m_ = mp.mpc(mu)
            val = 2 * (beta / m_) ** (nu / 2) * mp.besselk(nu, 2 * mp.sqrt(m_ * beta))
            return complex(val)
        m_ = mp.mpc(mu)
        h = mp.mpf(d) / 2
        # s = t_lo + u, e^{−μ t_lo} factored out; breakpoints at the decay
        # scale 1/Re μ and around the peak of s^{−d/2} e^{−Re μ s − β/s}
        mr = mp.mpf(mu.real)
        s_pk = (-h + mp.sqrt(h * h + 4 * mr * beta)) / (2 * mr)
        f = lambda u: ((t_lo + u) ** (-h)
                       * mp.exp(-m_ * u - beta / (t_lo + u)))
        cuts = sorted({mp.mpf(0)} | {c for c in (
            s_pk - t_lo, (s_pk - t_lo) * 2, 1 / mr, 10 / mr, 60 / mr)
            if c > 0})
        val, err = mp.quad(f, cuts + [mp.inf], error=True, maxdegree=12)
        if (abs(err) > 1e-12 * abs(val)
                and abs(err) > max(mp.mpf(abs_tol), mp.mpf('1e-300'))):
            raise SpatialPropagatorError(
                f'free correlator time integral did not converge (estimate '
                f'{complex(val)}, error {float(abs(err)):.2e}).')
        val = val * mp.exp(-m_ * t_lo)
        return complex(val)


def free_two_point_d(mu, D, kap, x, tau, spatial_dim, abs_tol=0.0):
    """Exact infinite-domain free correlator in ``d`` dimensions,
    ``C(x,τ) = κ ∫_{|τ|}^∞ ds e^{−μs} (4πDs)^{−d/2} e^{−|x|²/4Ds}`` — the
    inverse FT of ``κ/(μ+Dq²)·e^{−(μ+Dq²)|τ|}`` with no q-grid or cutoff.
    ``d = 1`` delegates to :func:`free_two_point`; ``d ≥ 2`` refuses the
    UV-divergent equal point ``x = τ = 0``.  ``abs_tol``: absolute accuracy
    needed on ``C`` (an image term far below the total needs no 1e-12
    relative accuracy of its own)."""
    d = int(spatial_dim)
    if d == 1:
        return free_two_point(mu, D, kap, x, tau)
    D = float(D)
    r = float(np.linalg.norm(np.atleast_1d(np.asarray(x, dtype=float))))
    beta = r * r / (4.0 * D)
    pref = float(kap) * (4.0 * math.pi * D) ** (-0.5 * d)
    integ = heat_time_integral_d(mu, beta, abs(float(tau)), d,
                                 abs_tol=(abs_tol / abs(pref) if pref else 0.0))
    return complex(pref * integ)


def free_two_point_box(mu, D, kap, x, tau, spatial_dim, L, max_shells=80,
                       rtol=1e-14):
    """Exact free correlator on a periodic cubic box of period ``L`` (``x``
    along one axis): the image sum ``Σ_{n∈ℤ^d} C_∞(x·e₁ + nL, τ)`` of
    :func:`free_two_point_d`, summed in shells ``‖n‖_∞ = s`` until two
    consecutive shells add less than ``rtol`` of the total.  Raises when that
    needs more than ``max_shells`` shells (box much smaller than the
    correlation length) instead of truncating silently."""
    import itertools as _it
    d = int(spatial_dim)
    L = float(L)
    if d == 1:
        return free_two_point(mu, D, kap, x, tau, bc_mode='periodic', L=L)
    e1 = np.zeros(d)
    e1[0] = float(x)
    total = free_two_point_d(mu, D, kap, e1, tau, d)
    quiet = 0
    for sh in range(1, max_shells + 1):
        add = 0j
        for n in _it.product(range(-sh, sh + 1), repeat=d):
            if max(abs(c) for c in n) != sh:
                continue
            add += free_two_point_d(mu, D, kap, e1 + L * np.asarray(n, float),
                                    tau, d, abs_tol=1e-3 * rtol * abs(total))
        total += add
        quiet = quiet + 1 if abs(add) <= rtol * abs(total) else 0
        if quiet >= 2:
            return complex(total)
    raise SpatialPropagatorError(
        f'periodic image sum did not converge in {max_shells} shells (L={L} '
        f'is small against the correlation length sqrt(D/mu)); not '
        f'truncating silently.')


# ── Assembly over (τ, x) grid ─────────────────────────────────────
def compute_spatial_correlator_tree(ft, model, prop, num_params,
                                    external_fields, tau_grid,
                                    spatial_grid, verbose=False):
    """Build the tree-level ``C_tau_x`` array of shape
    ``(len(tau_grid), len(spatial_grid))`` for the 2-point function.

    external_fields : the two external legs (internal names) — both
    must reference the same field for a diagonal correlator (v1).
    num_params : parameter values (incl. saddle, e.g. phistar1).
    Returns ``(C_tau_x, info)`` where info records the field, A, B,
    D_noise actually used.
    """
    import numpy as np

    spatial = prop.get('spatial_dim')
    if not spatial:
        raise SpatialPropagatorError('propagator has no spatial block.')
    # ``num_params`` from compute_cumulants is keyed by SR symbol;
    # normalize to string keys for robust lookup (parameter names,
    # saddle names like 'phistar1', and the PBC length).
    nps = {str(kk): vv for kk, vv in num_params.items()}
    bc_mode = prop.get('bc_mode', 'infinite')
    bc_params = prop.get('bc_params', {}) or {}
    L = None
    if bc_mode == 'periodic':
        lname = bc_params.get('length')
        if isinstance(lname, str):
            if lname not in nps:
                raise SpatialPropagatorError(
                    f'periodic length parameter {lname!r} not in num_params '
                    f'(have {sorted(nps)}).')
            L = float(nps[lname])
        else:
            L = float(lname)

    # Resolve which field index the external legs sit on.  The
    # propagator's physical column names are phys_names; external
    # fields are given as internal physical names (e.g. 'dphi' or
    # 'phi'->'dphi').  Map to the diagonal field index in ac_mass.
    ring_names = prop['ring_gen_names']
    n_tilde = prop['nf']
    phys_names = ring_names[n_tilde:]

    # External legs resolve through the phys_idx (name, population) keys, so
    # a multi-population field's population 2 is its own column (the old
    # name-only match with the digits stripped returned column 0 for every
    # population; an unknown leg fell back to column 0 silently).
    from engine.diagrams.type_assignment import build_field_index_map
    from engine.integration.spatial.pipeline_bridge import _legs_to_phys_idx
    _, _pidx = build_field_index_map(list(ring_names), int(n_tilde))
    _keys = _legs_to_phys_idx(list(external_fields), _pidx)
    fi = int(_pidx[_keys[0]])
    if len(_keys) >= 2 and int(_pidx[_keys[1]]) != fi:
        raise NotImplementedError(
            'compute_spatial_correlator_tree: the external legs sit on two '
            'different fields (a cross-correlator); this diagonal oracle '
            'returns auto-correlators only — use '
            'compute_spatial_correlator_via_pipeline (coupled driver).')

    # Tier-1 guard: the closed-form heat-kernel block is only built for a
    # diagonal (decoupled) inverse propagator.  Off-diagonal multi-field
    # coupling leaves ``G_tx_sym`` None (build_propagator records the
    # reason in ``spatial_tier1_error``).  Raise a CLEAR error here rather
    # than a cryptic KeyError on 'G_tx_sym'.
    if prop.get('G_tx_sym') is None:
        why = prop.get('spatial_tier1_error', 'no closed-form spatial block')
        raise NotImplementedError(
            'spatial v1 supports only the Tier-1 diagonal heat-kernel '
            'propagator (dispersion λ = -(A + B·k²), one decoupled field per '
            'block).  This model is not Tier-1 — the usual causes are '
            'off-diagonal / coupled multi-field structure OR a higher-'
            'derivative operator (k⁴+, e.g. Laplacian²).  Both are v2 '
            'features — see docs/spatial_phase5_rearchitecture_plan.md. '
            f'(reason: {why})')

    A_expr, B_expr = prop['G_tx_sym'][(fi, fi)]
    sub = {SR.var(kk): vv for kk, vv in nps.items()}
    A = complex(SR(A_expr).subs(sub))
    Bsub = SR(B_expr).subs(sub)
    B = float(Bsub.real() if hasattr(Bsub, 'real') else Bsub)
    # A spatial field must carry a Laplacian kinetic term with a strictly
    # POSITIVE diffusion B.  Two distinct failure modes (the erf closed form
    # would otherwise divide by √(4πB) or take √ of a negative):
    #   B == 0 → no Laplacian at all: a time-only (spatial_dim=0) field;
    #   B <  0 → wrong-sign Laplacian: anti-diffusive / short-wavelength
    #            unstable (ill-posed heat kernel).
    if B <= 0.0:
        if B < 0.0:
            raise SpatialPropagatorError(
                f'field {phys_names[fi]!r} (index {fi}) has NEGATIVE diffusion '
                f'B={B}: the spatial operator is anti-diffusive, so the heat '
                f'kernel diverges (the model is short-wavelength unstable / '
                f'ill-posed).  Check the sign of the Laplacian term — it '
                f'should enter as "- D*Laplacian" with D>0 (dispersion +D·k²) '
                f'— and that D itself is positive.')
        raise SpatialPropagatorError(
            f'field {phys_names[fi]!r} (index {fi}) has zero diffusion (B=0): '
            f'it carries no spatial (Laplacian) kinetic term, i.e. it is a '
            f'time-only (spatial_dim=0) field.  A spatial correlator C(x, τ) '
            f'is only defined for spatial (dim >= 1) fields — request this '
            f"field's correlator without a spatial_grid (time-only path).")
    noise = extract_noise_coefficients(ft, nps)
    Dn = noise.get(fi)
    if Dn is None:
        raise SpatialPropagatorError(
            f'no white-noise coefficient found for field index {fi} '
            f'(noise sector empty?). Available: {noise}')

    if verbose:
        print(f'      spatial correlator: field#{fi} A(mass)={A} '
              f'B(diff)={B} D_noise={Dn} bc={bc_mode}'
              + (f' L={L}' if L else ''))

    C = np.zeros((len(tau_grid), len(spatial_grid)), dtype=np.complex128)
    for it, tau in enumerate(tau_grid):
        for ix, x in enumerate(spatial_grid):
            C[it, ix] = free_two_point(A, B, Dn, float(x), float(tau),
                                       bc_mode=bc_mode, L=L)
    info = {'field_index': fi, 'A_mass': A, 'B_diffusion': B,
            'D_noise': Dn, 'bc_mode': bc_mode, 'L': L}
    return C, info


# ── d-general radial q→x inverse FT (the d>1 output transform) ─────
def radial_inverse_ft(q_grid, Cq, x, spatial_dim):
    """Inverse spatial FT of an ISOTROPIC momentum-space function ``C(|q|)``
    sampled on ``q_grid`` (0 … k_max) to real space ``C(|x|)``, for d ∈ {1,2,3}::

        d=1:  C(x) = (1/π)    ∫₀^∞ cos(q x)  C(q) dq
        d=2:  C(x) = (1/2π)   ∫₀^∞ q J₀(q x) C(q) dq      (Hankel order 0)
        d=3:  C(x) = (1/2π²x) ∫₀^∞ q sin(q x) C(q) dq     (sinc; x→0 limit handled)

    This is the EXTERNAL output transform (separate from the loop): it turns a
    self-energy-dressed ``δC(|q|,τ)`` into the real-space correlator in any d.  It
    truncates at ``k_max = q_grid[-1]`` — i.e. evaluated AT a cutoff (Regime 1);
    the continuum value is recovered as ``k_max → ∞`` with a fine grid.  ``x`` may
    be scalar or array; returns the matching shape.  (The d=1 branch matches the
    cosine transform the bubble path already uses.)
    """
    import numpy as np
    from scipy.special import j0
    q = np.asarray(q_grid, dtype=float)
    Cq = np.asarray(Cq)                       # keep complex if C(q) is complex
    cplx = np.iscomplexobj(Cq)               # (e.g. a coupled cross-correlator)
    xs = np.atleast_1d(np.asarray(x, dtype=float))
    out = np.empty(xs.shape, dtype=complex if cplx else float)
    for i, xi in enumerate(xs):
        axi = abs(float(xi))
        if spatial_dim == 1:
            out[i] = np.trapz(np.cos(q * axi) * Cq, q) / math.pi
        elif spatial_dim == 2:
            out[i] = np.trapz(q * j0(q * axi) * Cq, q) / (2.0 * math.pi)
        elif spatial_dim == 3:
            if axi < 1e-12:                       # sin(qx)/x → q as x→0
                out[i] = np.trapz(q * q * Cq, q) / (2.0 * math.pi ** 2)
            else:
                out[i] = (np.trapz(q * np.sin(q * axi) * Cq, q)
                          / (2.0 * math.pi ** 2 * axi))
        else:
            raise SpatialPropagatorError(
                f'radial_inverse_ft supports d=1,2,3; got spatial_dim={spatial_dim}')
    if np.ndim(x):
        return out.reshape(np.asarray(x).shape)
    return complex(out[0]) if cplx else float(out[0])


def periodic_inverse_ft(q_grid, Cq, x, spatial_dim, L, n_cut=None):
    """Inverse spatial FT of an ISOTROPIC ``C(|q|)`` (sampled on ``q_grid``)
    on a PERIODIC cubic box of period ``L`` in each of ``spatial_dim`` dims —
    the discrete-momentum analogue of :func:`radial_inverse_ft`::

        C(x) = (1/L^d)  Σ_{n ∈ ℤ^d}  cos((2π/L) n₁ x) · C(|k_n|),
               k_n = (2π/L)·n,   summed for |k_n| ≤ k_max = q_grid[-1].

    ``x`` is the displacement along ONE axis (the separation the spatial grid
    encodes).  ``C(|k_n|)`` is obtained by interpolating ``Cq`` (the SAME radial
    array the infinite-domain path builds) onto the lattice magnitudes, so both
    boundary conditions share one momentum-space correlator.  The continuum
    limit ``(1/L^d)Σ → (1/2π)^d ∫d^dk`` recovers :func:`radial_inverse_ft` as
    ``L → ∞``.  ``n_cut`` defaults to ``ceil(k_max·L/2π)`` (the same UV cutoff
    as the radial path).  Returns the shape of ``x``; complex-safe."""
    import numpy as np
    q = np.asarray(q_grid, dtype=float)
    Cq = np.asarray(Cq)
    cplx = np.iscomplexobj(Cq)
    xs = np.atleast_1d(np.asarray(x, dtype=float))
    d = int(spatial_dim)
    L = float(L)
    two_pi_L = 2.0 * math.pi / L
    k_max = float(q[-1])
    if n_cut is None:
        n_cut = int(math.ceil(k_max / two_pi_L)) + 1
    ns = np.arange(-n_cut, n_cut + 1)
    if d not in (1, 2, 3):
        raise SpatialPropagatorError(
            f'periodic_inverse_ft supports d=1,2,3; got spatial_dim={d}')
    # The phase depends on n1 only, so the sum factorises:
    #     C(x) = (1/L^d) Σ_{n1} cos((2π/L) n1 x) · S(n1),
    #     S(n1) = Σ_{n_⊥} C(|k_n|)   (n_⊥: the other d-1 axes),
    # built one n1 slab at a time: memory O(n^{d-1}), not O(n^d) (the whole
    # d=3 lattice was ~9 arrays of (2 n_cut + 1)^3 doubles, 13.6 GB at
    # n_cut = 288; item 53).  The lattice magnitudes are exact integers
    # before the square root, so each term is as before; only the order of
    # the additions changed.
    n_sq1 = ns.astype(float) ** 2
    if d == 1:
        perp = np.zeros(1)
    elif d == 2:
        perp = n_sq1
    else:
        perp = (n_sq1[:, None] + n_sq1[None, :]).ravel()
    S = np.empty(len(ns), dtype=complex if cplx else float)
    for a, m1 in enumerate(n_sq1):
        flat = two_pi_L * np.sqrt(m1 + perp)           # |k_n| on the slab
        Sk = np.interp(flat, q, Cq.real)
        if cplx:
            Sk = Sk + 1j * np.interp(flat, q, Cq.imag)
        S[a] = np.sum(np.where(flat <= k_max, Sk, 0.0))
    kx = two_pi_L * ns.astype(float)                   # phase wavenumber on axis 1
    out = np.empty(xs.shape, dtype=complex if cplx else float)
    pref = 1.0 / L ** d
    for i, xi in enumerate(xs):
        out[i] = pref * np.sum(np.cos(kx * float(xi)) * S)
    if np.ndim(x):
        return out.reshape(np.asarray(x).shape)
    return complex(out[0]) if cplx else float(out[0])


def free_correlator_static_closed_form(r, mu, D, T, spatial_dim):
    """The exact static (τ=0) free correlator ``C(|r|)`` — the closed-form oracle
    for :func:`radial_inverse_ft` (the inverse FT of ``C(q,0)=T/(μ+Dq²)``)::

        d=1:  (T / 2√(μD))  e^{−|r|√(μ/D)}
        d=2:  (T / 2πD)     K₀(|r|√(μ/D))
        d=3:  (T / 4πD)     e^{−|r|√(μ/D)} / |r|
    """
    from scipy.special import k0
    kappa = math.sqrt(mu / D)
    r = abs(float(r))
    if spatial_dim == 1:
        return T / (2.0 * math.sqrt(mu * D)) * math.exp(-kappa * r)
    if spatial_dim == 2:
        return T / (2.0 * math.pi * D) * float(k0(kappa * r))
    if spatial_dim == 3:
        return T / (4.0 * math.pi * D) * math.exp(-kappa * r) / r
    raise SpatialPropagatorError(
        f'spatial_dim must be 1,2,3; got {spatial_dim}')
