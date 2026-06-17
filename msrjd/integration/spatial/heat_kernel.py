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

import mpmath as mp
from sage.all import SR, I as SR_I, matrix


class SpatialPropagatorError(Exception):
    """Raised when the Tier-1 closed-form heat-kernel path does not
    apply (non-diagonal coupling, higher-derivative operator,
    non-unit iω coefficient, …).  The caller may fall back to a
    numerical inverse FT (Tier 2) or re-raise."""


# ── Closed-form kernels ───────────────────────────────────────────
def gaussian_heat_kernel(t, x, A, B, spatial_dim: int = 1, V=0.0):
    """θ(t) · (4π B t)^(-d/2) · exp(−(x − v t)²/(4 B t) − A t),  v = V/i.

    ``A`` (mass) and ``B`` (diffusion) may be complex.  ``x`` is the
    scalar separation in d=1; for d≥2 pass |x| (radial) — the
    prefactor uses d = ``spatial_dim``.

    The optional **drift** ``V`` is the ``k¹`` coefficient of the inverse
    propagator (the advection term).  The exact inverse FT of
    ``e^{-(A+Bk²+Vk)t}`` carries an extra factor
    ``exp(−i V x /(2B) + V² t /(4B))`` — for ``V = i v`` this shifts the
    Gaussian centre to ``x = v t`` (transport at velocity ``v``).
    ``V = 0`` (Laplacian-only / even kernels) is **bit-identical** to the
    pure heat kernel.  Drift is supported for d=1 only (it is a vector in
    d≥2 — out of v1 scope).  Returns a Python complex.
    """
    t = float(t.real) if hasattr(t, 'real') and not isinstance(t, complex) else t
    if (t.real if isinstance(t, complex) else t) <= 0.0:
        return 0j
    A = complex(A)
    B = complex(B)
    V = complex(V)
    x_signed = float(x)
    xx = x_signed ** 2
    pref = (4.0 * math.pi * B * t) ** (-0.5 * spatial_dim)
    drift_exp = 0.0
    if V != 0:
        if spatial_dim != 1:
            raise SpatialPropagatorError(
                'drift heat kernel (V≠0) is implemented for d=1 only '
                '(drift is a vector in d≥2).')
        drift_exp = -1j * V * x_signed / (2.0 * B) + V * V * t / (4.0 * B)
    return complex(pref * cmath.exp(-xx / (4.0 * B * t) - A * t + drift_exp))


def erf_time_integral(alpha, beta, L_lo, U_hi=None, dps: int = 30):
    """∫_{L_lo}^{U_hi} s^(-1/2) · exp(-β/s - α·s) ds via the erf-split
    closed form (Rescue A / Path-1 m=1).

    The antiderivative of ``exp(-α w² - β/w²)`` (after s = w²) is

        F(w) = (√π / 4√α) [ e^{+2√(αβ)} erf(√α w + √β/w)
                          + e^{-2√(αβ)} erf(√α w - √β/w) ]

    so the integral is ``2[F(√U) - F(√L)]``.  ``α`` (mass) and ``β``
    (= X²/4B ≥ 0) may be complex/real; evaluated in mpmath at ``dps``
    digits then returned as a Python complex.  ``U_hi=None`` ⇒ the
    semi-infinite ``U → ∞`` limit (valid for Re √α > 0).

    Verified to machine precision (incl. complex α) in
    ``docs/spatial_spikes/phase5_erfsplit_semigroup_spike.py``.
    """
    with mp.workdps(dps):
        a = mp.sqrt(mp.mpc(alpha))
        b = mp.sqrt(mp.mpc(beta))
        pref = mp.sqrt(mp.pi) / (4 * a)

        def _F(w):
            if w == 0:
                # w→0⁺ limit: erf(±∞) = ±1 (β>0); erf(0)=0 (β=0).
                if beta == 0:
                    return mp.mpf(0)
                return pref * (mp.e ** (2 * a * b) - mp.e ** (-2 * a * b))
            return pref * (mp.e ** (2 * a * b) * mp.erf(a * w + b / w)
                           + mp.e ** (-2 * a * b) * mp.erf(a * w - b / w))

        wL = mp.sqrt(mp.mpf(L_lo)) if L_lo > 0 else mp.mpf(0)
        if U_hi is None:
            F_hi = pref * (mp.e ** (2 * a * b) + mp.e ** (-2 * a * b))
        else:
            F_hi = _F(mp.sqrt(mp.mpf(U_hi)))
        return complex(2 * (F_hi - _F(wL)))


def image_sum(t, x, A, B, L, spatial_dim: int = 1, eps: float = 1e-12,
              n_max_cap: int = 2000, V=0.0):
    """Periodic-boundary heat kernel via the image-source sum
    ``Σ_n G_inf(t, x + nL)`` (1D).

    The sum is truncated once the |n| terms fall below ``eps`` times
    the n=0 term (the Gaussian tail decays super-exponentially in n),
    capped at ``n_max_cap`` on each side.  Only the d=1 case is
    supported in v1 (PBC in higher d needs a lattice sum per axis).
    The drift ``V`` (default 0) is forwarded to ``gaussian_heat_kernel``.
    """
    if spatial_dim != 1:
        raise SpatialPropagatorError(
            'image_sum: periodic BC implemented for d=1 only in v1.')
    base = gaussian_heat_kernel(t, x, A, B, spatial_dim=1, V=V)
    total = base
    ref = abs(base) if abs(base) > 0 else 1.0
    n = 1
    while n <= n_max_cap:
        term_p = gaussian_heat_kernel(t, x + n * L, A, B, spatial_dim=1, V=V)
        term_m = gaussian_heat_kernel(t, x - n * L, A, B, spatial_dim=1, V=V)
        total += term_p + term_m
        if abs(term_p) < eps * ref and abs(term_m) < eps * ref and n >= 2:
            break
        n += 1
    return complex(total)


# ── Symbolic extraction ───────────────────────────────────────────
def extract_mass_diffusion(kft_entry, omega, k_var, lap_sym, grad_sym=None):
    """Read (A, B, V) from one diagonal inverse-propagator entry.

    ``kft_entry`` is expected to have the (drift-generalized) Allen-Cahn
    form ``A + B·k² + V·k + iω`` after ``lap_sym → −k_var²`` and (when a
    first-derivative ``grad_sym`` appears) ``grad_sym → i·k_var``.  Returns
    ``(A_expr, B_expr, V_expr)`` as SR expressions in the model's
    parameters and saddle symbols:

      * ``A`` — mass (the ``k⁰`` term, the relaxation rate),
      * ``B`` — diffusion (the ``k²`` term),
      * ``V`` — **drift** (the ``k¹`` term).  ``V`` is **zero** for an
        even/Laplacian-only kernel; it is the advection coefficient
        ``i·v`` for a first-derivative ``∂_x`` term.  For a *gradient
        nonlinearity* (Burgers ``∂_x(φ²)``, KPZ ``(∂_xφ)²``) the only
        ``∂_x`` reaching the bilinear sector is the saddle cross-term
        ``∝ φ*``, so ``V → 0`` at the homogeneous saddle ``φ*=0`` and the
        propagator is the pure heat kernel.

    Raises ``SpatialPropagatorError`` if:
      * the entry is not linear in ``iω`` with unit coefficient
        (i.e. not a standard first-order-in-time MSR kernel), or
      * a residual ``k_var`` dependence beyond ``k²`` survives
        (higher-derivative operator), or
      * the entry still contains ``lap_sym`` / ``grad_sym`` after
        substitution.
    """
    subs = {lap_sym: -k_var**2}
    if grad_sym is not None:
        # ∂_x → i k  (odd, imaginary): the first-derivative drift symbol.
        subs[grad_sym] = SR_I * k_var
    e = SR(kft_entry).subs(subs).expand()
    if e.has(lap_sym) or (grad_sym is not None and e.has(grad_sym)):
        raise SpatialPropagatorError(
            f'inverse-propagator entry still contains an operator symbol '
            f'after k-substitution: {e}')

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
    # The linear-in-k term is the DRIFT V (was rejected in v1; now carried
    # by the drift-generalized heat kernel).  V == 0 for Laplacian-only
    # (even) kernels → bit-identical to the pure heat kernel.
    V = e0.coefficient(k_var, 1)
    A = e0.coefficient(k_var, 0)
    return A, B, V


def reaction_diffusion_matrices(K_ft, omega, k_var, lap_sym, grad_sym=None):
    """Extract the (matrix) reaction ``M``, diffusion ``𝒟`` and drift ``V`` from
    the FULL inverse-propagator matrix ``K_ft`` — the coupled-field generalization
    of :func:`extract_mass_diffusion`.  After ``lap_sym → −k_var²`` (and, when
    present, ``grad_sym → i·k_var``) each entry must have the form

        K_ft[i,j] = i·ω·δ_ij + M[i,j] + V[i,j]·k_var + 𝒟[i,j]·k_var²,

    i.e. the time derivative ``i·ω`` is DIAGONAL and unit-normalized, the
    off-diagonal entries are ω-independent, and no entry has a ``k`` power > 2.
    Returns SR matrices ``(M, D, V)`` (each ``N×N``) in the model parameters /
    saddle symbols.  For a diagonal ``K_ft`` they reduce to
    ``diag(ac_mass / ac_diffusion / ac_drift)`` — bit-identical, per entry, to
    :func:`extract_mass_diffusion`.

    This feeds the spectral reference propagator (``spectral_propagator.py``,
    paper B23) once the parameters are numeric; it is the coupled-field input
    that the diagonal Tier-1 path (``build_spatial_propagator``) does not handle.

    Raises :class:`SpatialPropagatorError` on a non-MSR / higher-derivative
    kernel (residual operator symbol, wrong ω structure, or ``k`` power > 2).
    """
    n = int(K_ft.nrows())
    subs = {lap_sym: -k_var**2}
    if grad_sym is not None:
        subs[grad_sym] = SR_I * k_var
    Mrows, Drows, Vrows = [], [], []
    for i in range(n):
        Mr, Dr, Vr = [], [], []
        for j in range(n):
            e = SR(K_ft[i, j]).subs(subs).expand()
            if e.has(lap_sym) or (grad_sym is not None and e.has(grad_sym)):
                raise SpatialPropagatorError(
                    f'inverse-propagator entry [{i},{j}] still contains an '
                    f'operator symbol after k-substitution: {e}')
            omega_coeff = e.coefficient(omega, 1)
            expected = SR_I if i == j else SR(0)
            if (omega_coeff - expected).simplify_full() != 0:
                role = ('the diagonal +i·ω time derivative' if i == j
                        else 'no ω dependence (off-diagonal)')
                raise SpatialPropagatorError(
                    f'inverse-propagator entry [{i},{j}] has ∂/∂ω = '
                    f'{omega_coeff}, expected {expected} ({role}).')
            e0 = e.coefficient(omega, 0)
            try:
                deg = e0.degree(k_var)
            except Exception:
                deg = 0
            if deg > 2:
                raise SpatialPropagatorError(
                    f'inverse-propagator entry [{i},{j}] has k-power {deg} > 2 '
                    f'(higher-derivative operator not supported): {e0}')
            Dr.append(e0.coefficient(k_var, 2))
            Vr.append(e0.coefficient(k_var, 1))
            Mr.append(e0.coefficient(k_var, 0))
        Mrows.append(Mr)
        Drows.append(Dr)
        Vrows.append(Vr)
    return matrix(SR, Mrows), matrix(SR, Drows), matrix(SR, Vrows)


def _make_numeric(expr):
    """Return a Python callable ``f(**num_params) -> complex`` for an SR
    ``expr`` (its free symbols are read off the expression).  Missing symbols
    at call time raise KeyError (caller supplies all parameters + saddle
    values)."""
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
    if d not in (1, 2, 3):
        raise SpatialPropagatorError(
            f'spatial_dim must be 1, 2, or 3 (got {d}).')
    # NOTE: the per-mode (A,B) extraction below is d-INDEPENDENT (it reads the
    # symbolic kernel K_ft); only the OUTPUT q→x transform is d-specific (d=1
    # uses free_two_point, d≥2 the radial/Hankel radial_inverse_ft — handled in
    # the bridge).  Periodic BC image sums remain d=1-only (see image_sum).
    bc_d = (model.get('boundary') or {}).get('mode', 'infinite')
    if d != 1 and bc_d == 'periodic':
        raise SpatialPropagatorError(
            f'periodic BC is implemented for spatial_dim=1 only; use '
            f"'infinite' for d={d}.")

    lap_sym = getattr(ns, 'Laplacian', None)
    if lap_sym is None:
        raise SpatialPropagatorError(
            'namespace has no Laplacian symbol (model not spatial?).')
    # First-derivative drift symbol (∂_x → i·k); the bare symbol a bilinear
    # Dx lowers to (``pipeline.spatial_operator_ir.GRADX_SYM``).  Sage caches
    # symbols by name, so ``SR.var('GradX')`` IS that same object.  It is
    # absent from Laplacian-only kernels (the substitution is then a no-op
    # and the extracted drift V is 0 → pure heat kernel).
    grad_sym = SR.var('GradX')

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
    ac_drift = {}       # (i,i) -> V expr (drift; 0 for even/Laplacian kernels)
    G_sym = {}          # (i,j) -> (A_expr, B_expr) or None

    for i in range(nf):
        A_expr, B_expr, V_expr = extract_mass_diffusion(
            K_ft[i, i], omega, k_var, lap_sym, grad_sym)
        ac_mass[i] = A_expr
        ac_diffusion[i] = B_expr
        ac_drift[i] = V_expr
        # G_tx_sym stays a 2-tuple (A, B) for backward compatibility; the
        # drift travels in ac_drift (read by make_g_tx_callables).
        G_sym[(i, i)] = (A_expr, B_expr)
    for i in range(nf):
        for j in range(nf):
            if (i, j) not in G_sym:
                G_sym[(i, j)] = None

    if verbose:
        print(f'      ── spatial propagator (d={d}, bc={bc_mode}) ──')
        for i in range(nf):
            _drift = '' if SR(ac_drift[i]).is_zero() else \
                f', V(drift)={ac_drift[i]}'
            print(f'        G_tx[{resp_names[i]},{phys_names[i]}]: '
                  f'A(mass)={ac_mass[i]}, B(diff)={ac_diffusion[i]}{_drift}')

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
        'ac_drift':     ac_drift,      # dict[i] -> V (drift; 0 = pure heat)
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
    ac_drift = prop.get('ac_drift', {}) or {}   # dict[i] -> V (default 0)

    def _zero_entry(t, x, **num_params):
        return 0j

    G_entries = {}
    for key, ab in G_sym.items():
        if ab is None:
            G_entries[key] = _zero_entry
            continue
        A_expr, B_expr = ab
        A_num = _make_numeric(A_expr)
        B_num = _make_numeric(B_expr)
        # Drift V for this diagonal entry (0 for Laplacian-only kernels).
        V_expr = ac_drift.get(key[0], 0) if key[0] == key[1] else 0
        V_num = (_make_numeric(V_expr)
                 if not SR(V_expr).is_zero() else None)

        def _make_entry(A_num=A_num, B_num=B_num, V_num=V_num):
            def _g(t, x, **num_params):
                A = A_num(**num_params)
                B = B_num(**num_params)
                V = V_num(**num_params) if V_num is not None else 0.0
                if bc_mode == 'periodic':
                    if L_name is None:
                        raise SpatialPropagatorError(
                            'periodic BC missing length parameter.')
                    L = (num_params[L_name] if isinstance(L_name, str)
                         else float(L_name))
                    return image_sum(t, x, A, B, float(L), spatial_dim=d, V=V)
                return gaussian_heat_kernel(t, x, A, B, spatial_dim=d, V=V)
            return _g
        G_entries[key] = _make_entry()
    return G_entries
