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

    A_expr, B_expr = prop['G_tx_sym'][(fi, fi)]
    sub = {SR.var(kk): vv for kk, vv in nps.items()}
    A = complex(SR(A_expr).subs(sub))
    Bsub = SR(B_expr).subs(sub)
    B = float(Bsub.real() if hasattr(Bsub, 'real') else Bsub)
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
