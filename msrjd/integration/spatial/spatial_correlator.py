"""
msrjd.integration.spatial.spatial_correlator
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
theory, and the leading term for an interacting one (the λφ³ vertex
enters at 1-loop, which is the remaining Phase-5 work — see
``docs/spatial_implementation_plan.md`` §5).  Off-diagonal field
coupling and loops are deferred to v2.
"""
from __future__ import annotations

import cmath
import math

from sage.all import SR

from msrjd.integration.spatial.heat_kernel import (
    SpatialPropagatorError, erf_time_integral, gaussian_heat_kernel,
)


# ── Noise extraction ──────────────────────────────────────────────
def extract_noise_coefficients(ft, num_params):
    """Return ``{field_index: D_noise}`` — the white-noise spectral
    coefficient per (response) field, read from the action's (2,0)
    bigrade sector (the ``- D^noise · φ̃²`` term).

    Convention (matching the framework's OU theories, e.g. action
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
            try:
                val = complex(SR(coeff).subs(sub))
            except Exception:
                val = 0j
            Dn = -val.real       # action term is - D_noise φ̃²
            if abs(Dn) > 1e-300:
                out[i] = Dn
    return out


# ── Free two-point correlator ─────────────────────────────────────
def free_two_point(A, B, D_noise, x, tau, bc_mode='infinite', L=None,
                   n_images_cap=200):
    """``C(x, τ)`` for one diagonal field.

    A : mass (float/complex), B : diffusion (>0), D_noise : noise
    spectral coefficient.  Infinite domain → erf closed form;
    periodic → image sum over ``x → x + nL`` of the infinite-domain
    correlator (the ring's periodicity).
    """
    B = float(B)
    A = complex(A)
    Dn = float(D_noise)
    pref = Dn / math.sqrt(4.0 * math.pi * B)

    def _C_inf(xx):
        beta = (xx * xx) / (4.0 * B)
        integ = erf_time_integral(A, beta, abs(tau), U_hi=None)
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

    # External legs may be ('phi',1) style already normalized to the
    # internal fluctuation name with a numeric suffix (e.g. 'dphi1').
    def _field_index(legname):
        base = str(legname)
        # strip trailing population index digits to match phys col base
        for idx, pn in enumerate(phys_names):
            if base == pn or base.rstrip('0123456789') == pn.rstrip('0123456789'):
                return idx
        return 0
    leg_names = [f[0] if isinstance(f, (tuple, list)) else f
                 for f in external_fields]
    fi = _field_index(leg_names[0])

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
            'block).  This theory is not Tier-1 — the usual causes are '
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
                f'field {leg_names[0]!r} (index {fi}) has NEGATIVE diffusion '
                f'B={B}: the spatial operator is anti-diffusive, so the heat '
                f'kernel diverges (the theory is short-wavelength unstable / '
                f'ill-posed).  Check the sign of the Laplacian term — it '
                f'should enter as "- D*Laplacian" with D>0 (dispersion +D·k²) '
                f'— and that D itself is positive.')
        raise SpatialPropagatorError(
            f'field {leg_names[0]!r} (index {fi}) has zero diffusion (B=0): '
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
    Cq = np.asarray(Cq, dtype=float)
    xs = np.atleast_1d(np.asarray(x, dtype=float))
    out = np.empty(xs.shape, dtype=float)
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
    return out.reshape(np.asarray(x).shape) if np.ndim(x) else float(out[0])


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
