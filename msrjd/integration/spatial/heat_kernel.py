"""
msrjd.integration.spatial.heat_kernel
=====================================
Real-space (t, x) propagators for spatial field theories (v1).

For an Allen-Cahn-like inverse propagator (diagonal in field index,
first-order in time, with a scalar ``D·k²`` diffusion term) the
momentum-space propagator after the substitution ``Laplacian → -k²``
is

    G(ω, k) = 1 / (A + B·k² + iω)

with **mass** ``A`` (the ω=0, k=0 value of the inverse propagator —
``μ`` for the free theory, ``μ + 3λφ*²`` at a φ⁴ saddle) and
**diffusion** ``B`` (the coefficient of ``k²`` — the ``D`` in
``D·Laplacian``).  Closing the ω-contour and inverse-Fourier-
transforming in ``k`` gives the closed-form heat kernel × exponential
decay

    G(t, x) = θ(t) · (4π B t)^(-1/2) · exp[ -x²/(4 B t) - A t ]   (d=1)

verified to machine precision (incl. complex A, B) in
``docs/spatial_spikes/phase5_erfsplit_semigroup_spike.py``.

This module provides:
  * ``gaussian_heat_kernel`` — the closed-form kernel above (complex
    A, B tolerated; any spatial dimension d via the (4πBt)^(-d/2)
    prefactor).
  * ``image_sum`` — periodic-boundary wrapper Σ_n G_inf(t, x + nL)
    (Phase 3).
  * ``extract_mass_diffusion`` — read (A, B) symbolically from one
    diagonal entry of the inverse-propagator matrix ``K_ft``.
  * ``build_spatial_propagator`` — assemble the prop-dict spatial
    block (``G_tx`` callable, ``G_tx_sym`` symbolic, ``k_var``,
    ``spatial_dim``, ``bc_mode``, ``bc_params``) for the diagonal
    Allen-Cahn-like case; raises ``SpatialPropagatorError`` (caught
    by the caller, which can fall back) on the non-diagonal /
    non-Allen-Cahn case (Tier 2, deferred to v2).
"""
from __future__ import annotations

import cmath
import math
from typing import Callable, Optional

from sage.all import SR, I as SR_I


class SpatialPropagatorError(Exception):
    """Raised when the Tier-1 closed-form heat-kernel path does not
    apply (non-diagonal coupling, higher-derivative operator,
    non-unit iω coefficient, …).  The caller may fall back to a
    numerical inverse FT (Tier 2) or re-raise."""


# ── Closed-form kernels ───────────────────────────────────────────
def gaussian_heat_kernel(t, x, A, B, spatial_dim: int = 1):
    """θ(t) · (4π B t)^(-d/2) · exp(-x²/(4 B t) - A t).

    ``A`` (mass) and ``B`` (diffusion) may be complex.  ``x`` is the
    scalar separation in d=1; for d≥2 pass |x| (radial) — the
    prefactor uses d = ``spatial_dim``.  Returns a Python complex.
    """
    t = float(t.real) if hasattr(t, 'real') and not isinstance(t, complex) else t
    if (t.real if isinstance(t, complex) else t) <= 0.0:
        return 0j
    A = complex(A)
    B = complex(B)
    xx = float(x) ** 2
    pref = (4.0 * math.pi * B * t) ** (-0.5 * spatial_dim)
    return complex(pref * cmath.exp(-xx / (4.0 * B * t) - A * t))


def image_sum(t, x, A, B, L, spatial_dim: int = 1, eps: float = 1e-12,
              n_max_cap: int = 2000):
    """Periodic-boundary heat kernel via the image-source sum
    ``Σ_n G_inf(t, x + nL)`` (1D).

    The sum is truncated once the |n| terms fall below ``eps`` times
    the n=0 term (the Gaussian tail decays super-exponentially in n),
    capped at ``n_max_cap`` on each side.  Only the d=1 case is
    supported in v1 (PBC in higher d needs a lattice sum per axis).
    """
    if spatial_dim != 1:
        raise SpatialPropagatorError(
            'image_sum: periodic BC implemented for d=1 only in v1.')
    base = gaussian_heat_kernel(t, x, A, B, spatial_dim=1)
    total = base
    ref = abs(base) if abs(base) > 0 else 1.0
    n = 1
    while n <= n_max_cap:
        term_p = gaussian_heat_kernel(t, x + n * L, A, B, spatial_dim=1)
        term_m = gaussian_heat_kernel(t, x - n * L, A, B, spatial_dim=1)
        total += term_p + term_m
        if abs(term_p) < eps * ref and abs(term_m) < eps * ref and n >= 2:
            break
        n += 1
    return complex(total)


# ── Symbolic extraction ───────────────────────────────────────────
def extract_mass_diffusion(kft_entry, omega, k_var, lap_sym):
    """Read (A, B) from one diagonal inverse-propagator entry.

    ``kft_entry`` is expected to have the Allen-Cahn-like form
    ``A + B·k² + (coeff)·iω`` after ``lap_sym → -k_var²``.  Returns
    ``(A_expr, B_expr)`` as SR expressions in the model's parameters
    and saddle symbols.

    Raises ``SpatialPropagatorError`` if:
      * the entry is not linear in ``iω`` with unit coefficient
        (i.e. not a standard first-order-in-time MSR kernel), or
      * a residual ``k_var`` dependence beyond ``k²`` survives
        (higher-derivative operator), or
      * the entry still contains ``lap_sym`` after substitution.
    """
    e = SR(kft_entry).subs({lap_sym: -k_var**2}).expand()
    if e.has(lap_sym):
        raise SpatialPropagatorError(
            f'inverse-propagator entry still contains {lap_sym} after '
            f'k-substitution: {e}')

    # iω coefficient must be exactly 1 (standard MSR normalization).
    omega_coeff = e.coefficient(omega, 1)
    # Expect omega_coeff == I  (since the term is + I*omega).
    if (omega_coeff - SR_I).simplify_full() != 0:
        raise SpatialPropagatorError(
            f'inverse-propagator entry is not +i·ω-normalized '
            f'(∂/∂ω = {omega_coeff}, expected I): {e}')

    # The ω-independent part, as a polynomial in k_var.
    e0 = e.coefficient(omega, 0)
    # Highest k power must be 2.
    try:
        deg = e0.degree(k_var)
    except Exception:
        deg = 0
    if deg > 2:
        raise SpatialPropagatorError(
            f'inverse-propagator has k-power {deg} > 2 (higher-'
            f'derivative operator not supported in v1): {e0}')
    B = e0.coefficient(k_var, 2)
    # Reject odd power (k¹) — would break the Gaussian inverse FT.
    if e0.coefficient(k_var, 1) != 0:
        raise SpatialPropagatorError(
            f'inverse-propagator has a linear-in-k term: {e0}')
    A = e0.coefficient(k_var, 0)
    return A, B


def _make_numeric(expr, free_symbol_names):
    """Return a Python callable ``f(**num_params) -> complex`` for an
    SR ``expr`` whose free symbols are named in ``free_symbol_names``.
    Missing symbols at call time raise KeyError (caller supplies all
    parameters + saddle values)."""
    expr = SR(expr)
    syms = sorted(expr.variables(), key=str)

    def _f(**num_params):
        subs = {}
        for s in syms:
            nm = str(s)
            if nm not in num_params:
                raise KeyError(
                    f'heat-kernel parameter {nm!r} not supplied '
                    f'(have {sorted(num_params)})')
            subs[s] = num_params[nm]
        val = expr.subs(subs)
        return complex(val)
    return _f


# ── Builder ───────────────────────────────────────────────────────
def build_spatial_propagator(K_ft, omega, ns, model,
                             resp_names, phys_names, verbose=True):
    """Assemble the spatial block of the propagator dict.

    Parameters
    ----------
    K_ft : Sage matrix
        Inverse propagator in (ω, Laplacian), rows=response,
        cols=physical (from build_propagator).
    omega : SR var
        The ω symbol.
    ns : namespace
        Carries ``ns.Laplacian`` (the inert operator symbol).
    model : dict
        Must contain ``model['spatial']`` (dim, fields_with_spatial)
        and may contain ``model['boundary']`` / ``model['initial']``.

    Returns
    -------
    dict with keys ``G_tx`` (matrix of callables ``(t, x, **num) ->
    complex``), ``G_tx_sym`` (dict of per-entry (A, B) SR exprs),
    ``k_var``, ``spatial_dim``, ``bc_mode``, ``bc_params``,
    ``initial_mode``, ``ac_mass``, ``ac_diffusion`` (per-entry A/B).

    Raises ``SpatialPropagatorError`` on the non-Tier-1 case.
    """
    spatial = model.get('spatial') or {}
    d = int(spatial.get('dim', 1))
    if d != 1:
        raise SpatialPropagatorError(
            f'v1 supports spatial_dim=1 only (got {d}).')

    lap_sym = getattr(ns, 'Laplacian', None)
    if lap_sym is None:
        raise SpatialPropagatorError(
            'namespace has no Laplacian symbol (model not spatial?).')

    k_var = SR.var('k', latex_name='k')
    nf = K_ft.nrows()

    # Tier 1 requires a diagonal inverse propagator (each field is its
    # own independent Allen-Cahn mode).  Off-diagonal coupling (e.g. a
    # 2-field theory with cross terms) needs the full matrix inverse +
    # multi-pole heat-kernel decomposition — Tier 2, deferred.
    for i in range(nf):
        for j in range(nf):
            if i != j and not SR(K_ft[i, j]).is_zero():
                raise SpatialPropagatorError(
                    f'inverse propagator has off-diagonal coupling '
                    f'K_ft[{i},{j}] = {K_ft[i, j]} ≠ 0; Tier-1 closed-'
                    f'form heat kernel needs a diagonal K_ft.  '
                    f'(Multi-field coupled spatial theories are a v2 '
                    f'feature.)')

    bc = model.get('boundary') or {'mode': 'infinite'}
    bc_mode = bc.get('mode', 'infinite')
    bc_params = {kk: vv for kk, vv in bc.items() if kk != 'mode'}
    initial_mode = (model.get('initial') or {}).get('mode', 'stationary')

    # Periodic length: resolve the parameter name (or inline number).
    L_name = None
    if bc_mode == 'periodic':
        L_name = bc_params.get('length')

    ac_mass = {}        # (i,i) -> A expr
    ac_diffusion = {}   # (i,i) -> B expr
    G_sym = {}          # (i,j) -> (A_expr, B_expr) or None

    for i in range(nf):
        A_expr, B_expr = extract_mass_diffusion(
            K_ft[i, i], omega, k_var, lap_sym)
        ac_mass[i] = A_expr
        ac_diffusion[i] = B_expr
        G_sym[(i, i)] = (A_expr, B_expr)
    for i in range(nf):
        for j in range(nf):
            if (i, j) not in G_sym:
                G_sym[(i, j)] = None

    if verbose:
        print(f'      ── spatial propagator (d={d}, bc={bc_mode}) ──')
        for i in range(nf):
            print(f'        G_tx[{resp_names[i]},{phys_names[i]}]: '
                  f'A(mass)={ac_mass[i]}, B(diff)={ac_diffusion[i]}')

    # NOTE: this block is PICKLABLE (all SR exprs + plain data) so it
    # caches cleanly via PipelineCache.  The runtime ``G_tx`` callables
    # are NOT stored here — they're reconstructed from this symbolic
    # data by ``make_g_tx_callables(prop)`` after build / cache-load
    # (closures don't pickle).
    return {
        'G_tx_sym':     G_sym,         # dict[(i,j)] -> (A,B) SR or None
        'k_var':        k_var,
        'spatial_dim':  d,
        'bc_mode':      bc_mode,
        'bc_params':    bc_params,
        'initial_mode': initial_mode,
        'ac_mass':      ac_mass,
        'ac_diffusion': ac_diffusion,
    }


def make_g_tx_callables(prop):
    """(Re)build the runtime ``G_tx`` dict of callables from a prop
    dict's picklable spatial block (``G_tx_sym`` + bc info).

    Call this after ``build_propagator`` returns (fresh build OR cache
    load) for any spatial prop.  Returns ``dict[(i,j)] -> callable
    (t, x, **num_params) -> complex``, or ``None`` if the prop has no
    spatial block.  Idempotent / cheap.
    """
    G_sym = prop.get('G_tx_sym')
    if G_sym is None:
        return None
    d = int(prop.get('spatial_dim', 1))
    bc_mode = prop.get('bc_mode', 'infinite')
    bc_params = prop.get('bc_params', {}) or {}
    L_name = bc_params.get('length') if bc_mode == 'periodic' else None

    def _zero_entry(t, x, **num_params):
        return 0j

    G_entries = {}
    for key, ab in G_sym.items():
        if ab is None:
            G_entries[key] = _zero_entry
            continue
        A_expr, B_expr = ab
        A_num = _make_numeric(A_expr, None)
        B_num = _make_numeric(B_expr, None)

        def _make_entry(A_num=A_num, B_num=B_num):
            def _g(t, x, **num_params):
                A = A_num(**num_params)
                B = B_num(**num_params)
                if bc_mode == 'periodic':
                    if L_name is None:
                        raise SpatialPropagatorError(
                            'periodic BC missing length parameter.')
                    L = (num_params[L_name] if isinstance(L_name, str)
                         else float(L_name))
                    return image_sum(t, x, A, B, float(L), spatial_dim=d)
                return gaussian_heat_kernel(t, x, A, B, spatial_dim=d)
            return _g
        G_entries[key] = _make_entry()
    return G_entries
