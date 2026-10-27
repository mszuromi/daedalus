"""
daedalus.api.spectra
====================
Exact spectral and integrated outputs of a k = 2 result, without a lag
grid: the power / cross spectrum ``S(ω)`` on a frequency grid,
``S(0)``, the integrated correlator ``∫_0^T C(τ) dτ`` for given windows ``T``
(and its mirror ``∫_{-T}^0``), the one-sided integrals to infinity and the
correlation time ``τ_c``, each per loop order and in total.

Ask for them in the run (``compute_cumulants(..., spectra={...})``,
``dd.Config(spectra={...})``) or afterwards from a live result
(``dd.spectra(result, omega=..., T=...)``); the values are the same.  The run
does not need a dense lag grid for them: ``tau_grid=np.array([0.0])`` (the
equal-time point only) is enough.

Conventions (k = 2)
-------------------
* ``C(τ) = κ(x_1(t), x_2(t + τ))``: leg 1 at time 0, leg 2 at ``τ``, exactly
  ``C_tau`` (``external_fields`` order).  ``C`` is the connected cumulant of
  the fluctuations about the saddle; loop order ``ℓ`` is its ``ℓ``-loop
  INCREMENT (the ``C_tau_by_ell`` convention; the totals are the sums).
* ``S(ω) = ∫_{-∞}^{∞} C(τ) e^{+iωτ} dτ`` (no ``1/2π``; ω in radians per
  time unit), the full cumulant including its δ components.  With ``x̂(ω) =
  ∫ x(t) e^{-iωt} dt`` this is ``⟨x̂_1(ω) x̂_2(ω)*⟩`` per unit time, i.e. the
  entry ``(1, 2)`` of the matrix spectrum ``H(iω) Q H(iω)ᴴ`` of a linear
  system, and the tree level equals ``tree_covariance``'s ``C0_{12}(ω)``.
  Symmetries (real fields): ``S_12(-ω) = conj(S_12(ω))`` and ``S_21(ω) =
  conj(S_12(ω))``; an autocorrelation (``C`` even) has a real, even
  spectrum.  ``Cumulant.spectrum`` uses the opposite sign of the exponent
  (``∫ C e^{-iωτ}``): ``spectra()['S'](ω) = Cumulant.spectrum(-ω)``.
  The one-sided spectrum of an autocorrelation (``∫_0^∞ G df = C(0)``,
  ``f = ω/2π``) is ``G(f) = 2 S(2πf)``.
* ``S(0) = ∫ C dτ`` (the zero-frequency value, the diffusion constant of the
  time integral of ``x`` times 2 for an autocorrelation).
* δ components.  A shot-noise spike ``w δ(τ − τ0)`` (``dd.cumulant``'s
  :class:`~daedalus.api.cumulant.DeltaComponent`) adds ``w e^{iωτ0}`` to ``S(ω)`` (the
  flat floor ``w`` for ``τ0 = 0``).  The integrals count a spike at
  ``0 < τ0 <= T`` (``∫_0^T``) or ``-T <= τ0 < 0`` (``∫_{-T}^0``) fully; the
  spike AT ``τ = 0`` sits on the boundary of both windows and is in neither:
  it is reported as ``delta0``, so that always ``S(0) = integral_inf +
  integral_inf_neg + delta0``.  For an autocorrelation without δ part
  ``S(0) = 2 ∫_0^∞ C``.
* ``τ = 0``.  ``C0`` is ``C_tau``'s ``τ = 0`` value: the Itô left limit
  ``C(0⁻)`` (exact), of the smooth part (the δ at 0 is not in it).
* ``tau_c = integral_inf / C0 = ∫_0^∞ C(τ) dτ / C(0)`` of the total; for a
  cross-correlation it is the ``τ > 0`` side.  A ratio does not split into
  loop increments: ``tau_c_by_ell[ℓ]`` is the correlation time of the
  cumulant TRUNCATED at order ``ℓ`` (tree + ... + ``ℓ`` loops), so its last
  entry is ``tau_c``.

How they are computed
---------------------
``method='exact'`` (the default route of ``'auto'``): from the
matrix-exponential triples of the state-space Phase J engine
(:mod:`daedalus.api.me_output`): per side ``C(u) = c e^{G u} x0``, ``u = |τ|``,

* ``S(ω) = c_+ (−iω − G_+)^{-1} x0_+ + c_- (iω − G_-)^{-1} x0_-``, one
  triangular resolvent solve per block and frequency;
* ``∫_0^∞ C = c (−G)^{-1} x0`` (the resolvent at 0);
* ``∫_0^T C = c ∫_0^T e^{G u} du x0``, the top-right column of the augmented
  exponential ``exp([[G, x0], [0, 0]] T)`` (exact for any ``T``, no
  cancellation at small ``T``, no inverse of ``G``); a block beyond the dense
  forms (``me_output.MAX_BLOCK`` states) uses ``c (−G)^{-1} (x0 − e^{GT}
  x0)`` with the segment's own action and resolvent instead; once ``e^{GT}``
  is below double precision (``T`` times the slowest rate > 2000) the
  window is ``∫_0^∞``.

No pole, residue or quadrature is involved; the values agree with a fine
quadrature of ``C_tau`` to round-off.  The exact delayed tree of a delay
model (``daedalus.api.delay``, ``tree='exact'``) has no triple but an exact Laplace
transform: ``S(ω)`` and ``∫_0^∞`` stay exact, ``∫_0^T`` is computed by
adaptive quadrature of the tree with its error estimate.  Padé loops
(``daedalus.api.delay``) are triples of the Padé-approximated model: exact for it,
with the approximation error ``result['delay']['est_rel_error']``.

``method='quad'`` (and ``'auto'`` for a loop order without triples: a subset
on the pole route, ``DAEDALUS_PHASE_J_ENGINE=poles``, the pole-cluster
deformation): adaptive QUADPACK quadrature of the Phase J callables
(``quad`` on ``[0, ∞)``, on ``[0, T]`` up to ten correlation times and as
``∫_0^∞ − ∫_T^∞`` beyond, the Fourier-weight QAWF for ``S(ω ≠ 0)``), with the estimated absolute error of every value in
``error_by_ell`` / ``abs_error``; ``'auto'`` also issues an
:class:`daedalus.api.me_output.MEFallbackWarning` naming the orders and the reasons.
``'exact'`` refuses (:class:`daedalus.api.me_output.MEUnavailable`) instead.

Refused (``ValueError`` before a run, :class:`daedalus.api.me_output.MEUnavailable`
after it): k ≠ 2, spatial models (``C(x, τ)``: no lag-axis triple),
switch-on transients (``t_min``: not a function of the lag),
``discard=True`` / ``'auto'`` runs (their callables are released) and loaded
files.  A ``C`` that does not decay (a block with an eigenvalue of
non-negative real part, or constant blocks that do not cancel) has no
``S`` and no ``∫_0^∞``: those values are NaN, with the reason in ``notes``;
the finite windows are still given.
"""
from __future__ import annotations

import math
import operator
import warnings
from typing import Any, Optional

import numpy as np
import scipy.linalg as sl

from daedalus.api._grids import is_real_number

__all__ = ['spectra', 'resolve_spectra_options', 'SPECTRA_OPTIONS',
           'SpectraWarning']

#: the keys of the ``spectra=`` option (``compute_cumulants``, ``dd.Config``)
SPECTRA_OPTIONS = ('omega', 'T', 'method', 'epsabs', 'epsrel', 'limit')

#: default quadrature tolerances of the fallback route
_EPSABS = 1e-13
_EPSREL = 1e-11
_LIMIT = 500

#: a side's constant (non-decaying) blocks are taken as cancelling when the
#: magnitude of their summed weight is below this fraction of the side's
#: ``Σ |weight|``
_CONST_RTOL = 1e-12

#: a decaying block's window ``∫_0^T`` is its ``∫_0^∞`` once ``T`` times its
#: slowest rate exceeds this (``e^{-2000}`` is far below double precision,
#: with room for the polynomial factors of repeated eigenvalues)
_DECAYED = 2000.0

#: the quadrature window ``∫_0^T`` is taken directly up to this many
#: correlation times, beyond as ``∫_0^∞ − ∫_T^∞`` (QUADPACK on a long
#: ``[0, T]`` misses a ``C`` concentrated near 0 and reports ~0 error)
_DIRECT_SPANS = 10.0

CONVENTIONS = (
    'C(tau) = kappa(x_1(t), x_2(t + tau)) (C_tau); '
    'S(omega) = int C(tau) exp(+i omega tau) dtau (no 1/2pi), delta '
    'components included = <x1^(w) x2^(w)*> per unit time; '
    'integral(T) = int_(0,T] C, integral_neg(T) = int_[-T,0) C (spikes inside '
    'counted, the spike at tau = 0 in neither: delta0); '
    'S(0) = integral_inf + integral_inf_neg + delta0; '
    'C0 = C(0-) (Ito left limit, smooth part); '
    'tau_c = integral_inf / C0 (tau_c_by_ell: truncated at that order)')


class SpectraWarning(UserWarning):
    """``compute_cumulants(spectra=...)``: the spectral outputs could not be
    computed for this result (the reason is in ``result['spectra']``); the
    run itself is kept."""


# ─── options ────────────────────────────────────────────────────────────────

def resolve_spectra_options(spec, name='spectra'):
    """The ``spectra=`` option as a dict ``{omega, T, method, epsabs,
    epsrel, limit}`` (``omega`` / ``T`` float arrays or ``None``), or
    ``None`` when not requested.  ``True`` asks for ``S(0)``, the
    integrals to infinity and ``tau_c`` only; a dict gives ``omega`` (a
    frequency grid: an ``(lo, hi, n)`` TUPLE is a range, a list/array literal
    points, as ``tau_grid``) and ``T`` (one window length or several,
    ``>= 0``)."""
    if spec is None or spec is False:
        return None
    if spec is True:
        spec = {}
    if not isinstance(spec, dict):
        raise ValueError(f'{name} must be None, True or a dict with keys '
                         f'{list(SPECTRA_OPTIONS)}; got {spec!r}.')
    bad = sorted(str(k) for k in spec if k not in SPECTRA_OPTIONS)
    if bad:
        raise ValueError(f'{name}: unknown key(s) {bad}; the keys are '
                         f'{list(SPECTRA_OPTIONS)}.')
    from daedalus.api._grids import resolve_grid, require_finite
    out = {}
    om = spec.get('omega')
    out['omega'] = (None if om is None else
                    np.atleast_1d(resolve_grid(om, f'{name}.omega')).ravel())
    T = spec.get('T')
    if T is None:
        out['T'] = None
    else:
        Ta = np.atleast_1d(np.asarray(
            resolve_grid(T, f'{name}.T') if isinstance(T, (tuple, list))
            else T, dtype=float)).ravel()
        if Ta.size == 0:
            raise ValueError(f'{name}.T is empty.')
        require_finite(Ta, f'{name}.T')
        if np.any(Ta < 0):
            raise ValueError(f'{name}.T must be >= 0 (a window length); got '
                             f'{Ta[Ta < 0][:3].tolist()}.')
        out['T'] = Ta
    meth = spec.get('method', 'auto')
    if meth not in ('auto', 'exact', 'quad'):
        raise ValueError(f"{name}.method must be 'auto', 'exact' or 'quad'; "
                         f"got {meth!r}.")
    out['method'] = meth
    for key, dflt in (('epsabs', _EPSABS), ('epsrel', _EPSREL)):
        v = spec.get(key, dflt)
        if (not is_real_number(v)
                or not np.isfinite(float(v)) or float(v) <= 0):
            raise ValueError(f'{name}.{key} must be a finite number > 0; got '
                             f'{v!r}.')
        out[key] = float(v)
    lim = spec.get('limit', _LIMIT)
    try:                         # int, numpy and Sage integers; not a bool
        lim_ok = (not isinstance(lim, (bool, np.bool_))
                  and operator.index(lim) >= 1)
    except TypeError:
        lim_ok = False
    if not lim_ok:
        raise ValueError(f'{name}.limit must be an integer >= 1; got '
                         f'{lim!r}.')
    out['limit'] = int(lim)
    return out


def check_spectra_question(k, *, spatial=False, t_min=None, discard=False,
                           output='cumulant'):
    """Refuse, before a run, a question the spectral outputs do not cover."""
    if int(k) != 2:
        raise ValueError(f'spectra= is defined for k = 2 (a lag τ); got '
                         f'k = {k}.')
    if spatial:
        raise ValueError('spectra= is for temporal models: a spatial C(x, τ) '
                         'has no lag-axis triple (its spectrum is a function '
                         'of q and ω; not implemented).')
    if t_min is not None:
        raise ValueError('spectra= with t_min: a switch-on result is a '
                         'function of the absolute times, not of a lag.')
    if discard not in (False, None):
        raise ValueError(f'spectra= with discard={discard!r}: evaluate-and-'
                         f'discard releases the diagrams the spectra are '
                         f'built from; use discard=False.')
    if output not in (None, 'cumulant'):
        raise ValueError(f'spectra= with output={output!r}: the spectral '
                         f'outputs are those of the connected cumulant '
                         f"(output='cumulant').")


# ─── one block / one side, exact ────────────────────────────────────────────

def _decays(b):
    """True when every eigenvalue of the block's generator has a negative
    real part (the diagonal of its triangular form)."""
    from daedalus.api.me_output import MAX_BLOCK
    if b.constant:
        return False
    if b.n > MAX_BLOCK:
        seg = b.seg
        d = getattr(seg, 'diag', None)
        if d is None:
            return True                  # checked by the result's finiteness
        return bool(np.max(np.real(np.asarray(d))) < 0)
    T = b.tri()[0]
    return bool(np.max(np.real(np.diag(T))) < 0)


def _side_split(side):
    """``(blocks, constant)``: the decaying blocks of a side and the summed
    weight of its constant (non-decaying, ``u``-independent) blocks.
    Raises :class:`MEUnavailable` for a block with an eigenvalue of
    non-negative real part, or constant blocks that do not cancel."""
    from daedalus.api.me_output import MEUnavailable
    blocks, const, scale = [], 0j, 0.0
    for b in side.blocks:
        if b.constant:
            const += b.weight
            scale += abs(b.weight)
            continue
        if not _decays(b):
            raise MEUnavailable(
                f'ell={side.ell} side {side.side!r}: a block of C(τ) does not '
                f'decay (an eigenvalue of its generator has a non-negative '
                f'real part): S(ω) and ∫_0^∞ C do not exist')
        blocks.append(b)
    if scale and abs(const) > _CONST_RTOL * scale:
        raise MEUnavailable(
            f'ell={side.ell} side {side.side!r}: C(τ) tends to the constant '
            f'{complex(const):.6g} as |τ| → ∞ (non-decaying blocks that do '
            f'not cancel): S(ω) and ∫_0^∞ C do not exist')
    return blocks, const


def _laplace(side, zs):
    """``∫_0^∞ e^{−z u} C(u) du`` of one side for each ``z`` in ``zs``
    (``Re z >= 0``)."""
    from daedalus.api.me_output import ExactDelayTreeSide
    zs = np.atleast_1d(np.asarray(zs, dtype=complex))
    if isinstance(side, ExactDelayTreeSide):
        return np.array([side.laplace(z, 0) for z in zs], dtype=complex)
    blocks, _const = _side_split(side)
    out = np.zeros(zs.shape, dtype=complex)
    for b in blocks:
        out += np.array([b.resolvent(z, 1) for z in zs], dtype=complex)
    return out


def _decay_rate(b):
    """``min(−Re λ)`` over the eigenvalues of a block's generator (the
    diagonal of its triangular form), or ``None`` when not decaying or
    unknown (a block beyond the dense forms)."""
    from daedalus.api.me_output import MAX_BLOCK
    if b.constant or b.n > MAX_BLOCK:
        return None
    r = -float(np.max(np.real(np.diag(b.tri()[0]))))
    return r if r > 0 else None


def _block_window(b, Ts):
    """``∫_0^T`` of one block for every ``T`` in ``Ts``."""
    from daedalus.api.me_output import MAX_BLOCK, MEUnavailable
    Ts = np.asarray(Ts, dtype=float)
    out = np.zeros(Ts.shape, dtype=complex)
    if b.weight == 0:
        return out
    if b.constant:
        return b.weight * Ts.astype(complex)
    rate = _decay_rate(b)
    if rate is not None and np.any(rate * Ts > _DECAYED):
        # e^{GT} has decayed below double precision: ∫_0^T = ∫_0^∞ (the
        # exponential of a huge G T would overflow or stall)
        far = rate * Ts > _DECAYED
        out[far] = b.resolvent(0.0, 1)
        if not np.all(far):
            out[~far] = _block_window(b, np.where(far, 0.0, Ts))[~far]
        return out
    if b.n <= MAX_BLOCK:
        G, x, c = b.triple()
        n = G.shape[0]
        dt = np.result_type(G, x, float)
        M = np.zeros((n + 1, n + 1), dtype=dt)
        M[:n, :n] = G
        M[:n, n] = x
        for i, T in np.ndenumerate(Ts):
            if T == 0:
                continue
            out[i] = complex(np.asarray(c) @ sl.expm(M * T)[:n, n])
        return out
    if b.seg is None or b.rows or b.beta == 0:
        raise MEUnavailable(f'a block of {b.n} states with constant rows is '
                            f'beyond the dense forms (MAX_BLOCK={MAX_BLOCK})')
    # c (−G)^{-1} (x0 − e^{GT} x0), G = β G_seg
    x0 = np.asarray(b.x0, dtype=complex)
    for i, T in np.ndenumerate(Ts):
        if T == 0:
            continue
        v = b.seg.action(b.x0, b.beta * float(T), b.dtype)
        y = b.seg.resolve(x0 - np.asarray(v, dtype=complex), 0.0) / b.beta
        out[i] = complex(b.weight * (np.asarray(b.c) @ y))
    return out


def _window_exact(side, Ts):
    """``∫_0^T C(u) du`` of one side (triples), for each ``T``."""
    out = np.zeros(np.shape(Ts), dtype=complex)
    for b in side.blocks:
        out = out + _block_window(b, Ts)
    return out


# ─── quadrature ─────────────────────────────────────────────────────────────

def _quad_c(f, a, b, epsabs, epsrel, limit, weight=None, wvar=None):
    """``∫_a^b f`` of a complex function by QUADPACK on its real and
    imaginary parts: ``(value, abs_error)``."""
    from scipy.integrate import quad
    kw = dict(epsabs=epsabs, limit=limit, full_output=1)
    if weight is not None:
        kw.update(weight=weight, wvar=wvar)
        if np.isinf(b):
            kw['limlst'] = 200            # QAWF: epsrel is not used
        else:
            kw['epsrel'] = epsrel
    else:
        kw['epsrel'] = epsrel
    with warnings.catch_warnings():
        # QUADPACK's IntegrationWarning is folded into the error estimate
        warnings.simplefilter('ignore')
        re = quad(lambda s: f(s).real, a, b, **kw)
        im = quad(lambda s: f(s).imag, a, b, **kw)
    return complex(re[0], im[0]), float(abs(re[1]) + abs(im[1]))


class _QuadSide:
    """Quadrature of one loop order's ``C(τ)`` (a callable of the signed
    τ, from the Phase J callables or the exact delayed tree)."""

    def __init__(self, C, epsabs, epsrel, limit):
        self.C = C
        self.kw = (epsabs, epsrel, limit)
        self.scale = None
        self._inf = {}

    def _eabs(self):
        # an absolute tolerance on the scale of C
        if self.scale is None:
            s = max(abs(self.C(-1e-9)), abs(self.C(1e-9)), 1e-300)
            self.scale = s
        epsabs, epsrel, _l = self.kw
        return max(epsabs, 1e-2 * epsrel * self.scale)

    def window(self, sg, T, inf=None):
        """``∫_0^T C(sg u) du``: directly up to ``_DIRECT_SPANS`` times the
        side's correlation time ``|∫_0^∞ C| / |C(0)|``, beyond as ``∫_0^∞ −
        ∫_T^∞`` (QUADPACK on a long ``[0, T]`` never samples a ``C``
        concentrated near 0, and reports a ~0 error).  ``inf``: the side's
        ``(∫_0^∞, abs_error)`` when known (else :meth:`infinite`)."""
        _ea, epsrel, limit = self.kw
        T = float(T)
        if T == 0:
            return 0j, 0.0
        I, eI = self.infinite(sg) if inf is None else inf
        c0 = abs(self.C(sg * 1e-9))
        if not (c0 > 0 and np.isfinite(I)
                and T > _DIRECT_SPANS * abs(I) / c0):
            return _quad_c(lambda u: self.C(sg * u), 0.0, T,
                           self._eabs(), epsrel, limit)
        from daedalus.api.me_output import MEUnavailable
        try:
            tail, et = _quad_c(lambda s: self.C(sg * (T + s)), 0.0, np.inf,
                               self._eabs(), epsrel, limit)
        except MEUnavailable:
            # the exact delayed tree is capped at a finite lag: [0, T] only
            return _quad_c(lambda u: self.C(sg * u), 0.0, T,
                           self._eabs(), epsrel, limit)
        return complex(I) - tail, float(eI) + et

    def infinite(self, sg):
        if sg not in self._inf:
            _ea, epsrel, limit = self.kw
            self._inf[sg] = _quad_c(lambda u: self.C(sg * u), 0.0, np.inf,
                                    self._eabs(), epsrel, limit)
        return self._inf[sg]

    def spectrum(self, w):
        """``∫ C(τ) e^{iωτ} dτ = ∫_0^∞ cos(ωs)[C(s)+C(−s)] ds + i ∫_0^∞
        sin(ωs)[C(s)−C(−s)] ds``."""
        _ea, epsrel, limit = self.kw
        ea = self._eabs()

        def ev(s):
            return self.C(s) + self.C(-s)

        def od(s):
            return self.C(s) - self.C(-s)
        if w == 0.0:
            return _quad_c(ev, 0.0, np.inf, ea, epsrel, limit)
        a, e1 = _quad_c(ev, 0.0, np.inf, ea, epsrel, limit, 'cos', abs(w))
        b, e2 = _quad_c(od, 0.0, np.inf, ea, epsrel, limit, 'sin', abs(w))
        return a + 1j * math.copysign(1.0, w) * b, e1 + e2


# ─── the whole result ───────────────────────────────────────────────────────

def _orders(result):
    by = result.get('C_tau_by_ell')
    if by:
        return sorted(int(e) for e in by)
    return sorted(int(e) for e in (result.get('total_C_by_ell') or {}))


def _refuse_result(result):
    from daedalus.api.me_output import MEUnavailable
    if not isinstance(result, dict) or 'config' not in result:
        raise TypeError('spectra needs a compute_cumulants / dd.run result '
                        'dict')
    cfg = result['config']
    k = int(cfg.get('k', 0))
    if k != 2:
        raise MEUnavailable(f'spectra are defined for k = 2 (got k = {k})')
    if 'C_tau_x' in result or cfg.get('spatial'):
        raise MEUnavailable('spectra: a spatial result C(x, τ) has no '
                            'lag-axis triple (not implemented)')
    if result.get('t_min') is not None:
        raise MEUnavailable('spectra: a switch-on result (t_min) is a '
                            'function of the absolute times, not of a lag')
    pm = result.get('phase_j_memory') or {}
    if pm.get('discard'):
        raise MEUnavailable('spectra: this run released its diagrams '
                            '(evaluate-and-discard); rerun with discard=False')
    if not result.get('phase_j_by_ell') and not result.get('total_C_by_ell'):
        raise MEUnavailable('spectra: the result has no live Phase J '
                            'closures (a loaded file?)')


def spectra(result, omega=None, T=None, *, method='auto', epsabs=_EPSABS,
            epsrel=_EPSREL, limit=_LIMIT) -> dict:
    """The exact spectral and integrated outputs of a live k = 2 result
    (``dd.spectra``); see the module docstring for the conventions.

    Parameters
    ----------
    result : dict
        A ``compute_cumulants`` / ``dd.run`` result (k = 2, temporal,
        stationary, not loaded from a file).
    omega : array, (lo, hi, n) tuple or None
        Angular frequencies of ``S(ω)``; ``None``: no spectrum grid (``S0``
        is always given).
    T : float, array or None
        Window lengths of ``∫_0^T C`` and ``∫_{-T}^0 C``.
    method : 'auto' | 'exact' | 'quad'
        See the module docstring.
    epsabs, epsrel, limit
        Tolerances of the quadrature route.

    Returns
    -------
    dict
        ``omega``, ``S``, ``S_by_ell`` (complex arrays over ``omega``),
        ``S0``, ``S0_by_ell``, ``T``, ``integral``, ``integral_by_ell``,
        ``integral_neg``, ``integral_neg_by_ell`` (complex arrays over
        ``T``), ``integral_inf``, ``integral_inf_by_ell``,
        ``integral_inf_neg``, ``integral_inf_neg_by_ell``, ``delta0``,
        ``delta0_by_ell``, ``C0``, ``C0_by_ell``, ``tau_c``,
        ``tau_c_by_ell`` (truncated orders), ``method_by_ell``
        ('exact' / 'quad' / 'exact+quad'), ``error_by_ell`` (``None`` for an
        exact order, else ``{quantity: abs_error}``), ``abs_error`` (the
        summed quadrature errors per quantity, ``None`` when every order is
        exact), ``status`` ('ok'), ``notes`` and ``conventions``.  Values that
        do not exist (a non-decaying ``C``) are NaN with the reason in
        ``notes``.
    """
    from daedalus.api.me_output import (MEUnavailable, MEFallbackWarning,
                               ExactDelayTreeSide)
    from daedalus.api.cumulant import Cumulant, _memo_C
    _refuse_result(result)
    opts = resolve_spectra_options({'omega': omega, 'T': T, 'method': method,
                                    'epsabs': epsabs, 'epsrel': epsrel,
                                    'limit': limit}, 'spectra')
    om, Ts, method = opts['omega'], opts['T'], opts['method']
    qkw = (opts['epsabs'], opts['epsrel'], opts['limit'])
    c = Cumulant.from_result(result)
    orders = _orders(result)
    notes = []
    if not c.complete:
        notes.append('δ components may be incomplete: '
                     + '; '.join(c.incomplete_reasons))
    unplaced = [d for d in c.deltas if d.location is None]
    if unplaced:
        raise MEUnavailable(
            f'spectra: {len(unplaced)} δ component(s) of this result are not '
            f'a point of the lag τ (ell = '
            f'{sorted({d.loop_order for d in unplaced})}): their weight in '
            f'S(ω) and the windows is undefined')
    me = None if method == 'quad' else c.me
    quad_orders, why = [], {}
    for e in orders:
        if method == 'quad':
            quad_orders.append(e)
            why[e] = "method='quad'"
        elif me is None:
            quad_orders.append(e)
            why[e] = '; '.join(c.me_reasons) or 'no triples'
        elif me.reasons.get(e) or e not in me.sides:
            quad_orders.append(e)
            why[e] = '; '.join(me.reasons.get(e) or ['no triple'])
    from daedalus.api.me_output import _exact_delay_tree
    if method == 'quad' and 0 in quad_orders and _exact_delay_tree(result):
        raise MEUnavailable(
            "spectra(method='quad'): the k = 2 tree is the exact delayed "
            "covariance (method of steps over the delay), which cannot be "
            "integrated numerically to infinite lag; use method='auto' (its "
            "S(ω) and ∫_0^∞ are exact) or delay={'tree': 'pade'}")
    if quad_orders and method == 'exact':
        raise MEUnavailable('spectra: no matrix-exponential triples for '
                            + '; '.join(f'ell={e}: {why[e]}'
                                        for e in quad_orders))
    if quad_orders and method == 'auto':
        msg = ('spectra: no matrix-exponential triples for loop order(s) '
               f'{quad_orders} (' + '; '.join(f'ell={e}: {why[e]}'
                                              for e in quad_orders)
               + '); integrating their Phase J callables numerically '
               '(error estimates in error_by_ell)')
        warnings.warn(msg, MEFallbackWarning, stacklevel=2)
        notes.append(msg)

    nan = complex('nan')
    out_S, out_I, out_In, out_Iinf, out_Iinfn, out_C0 = {}, {}, {}, {}, {}, {}
    out_d0, meth, errs = {}, {}, {}
    for e in orders:
        # δ components of this order
        dl = [d for d in c.deltas if d.loop_order == e]
        d0 = sum((d.weight for d in dl if d.location == 0.0), 0j)
        out_d0[e] = complex(d0)
        dS = (np.zeros(0) if om is None else
              sum((d.weight * np.exp(1j * om * d.location) for d in dl),
                  np.zeros(om.shape, dtype=complex)))

        def d_in(lo, hi, dl=dl):
            return sum((d.weight for d in dl if lo < d.location <= hi), 0j)
        if e in quad_orders:
            fn = (c.smooth_fns or {}).get(e)
            if fn is None:
                raise MEUnavailable(f'spectra: ell={e} has neither a triple '
                                    f'nor a Phase J callable')
            q = _QuadSide(_memo_C([fn]), *qkw)
            er = {}
            meth[e] = 'quad'
            C0 = complex(fn(0.0, 0.0))
            if om is not None:
                vals = [q.spectrum(float(w)) for w in om]
                S = np.array([v for v, _ in vals], dtype=complex)
                er['S'] = np.array([x for _, x in vals])
            Ip, ep = q.infinite(1.0)
            In, en = q.infinite(-1.0)
            er['integral_inf'], er['integral_inf_neg'] = ep, en
            if Ts is not None:
                wp = [q.window(1.0, t) for t in Ts]
                wn = [q.window(-1.0, t) for t in Ts]
                IT = np.array([v for v, _ in wp], dtype=complex)
                ITn = np.array([v for v, _ in wn], dtype=complex)
                er['integral'] = np.array([x for _, x in wp])
                er['integral_neg'] = np.array([x for _, x in wn])
            er['S0'] = ep + en
            errs[e] = er
        else:
            sp, sm = me.sides[e]['+'], me.sides[e]['-']
            meth[e] = 'exact'
            er = None
            C0 = complex(me(np.array([0.0]), ell=e, at_zero='negative')[0])
            try:
                Ip, In = _laplace(sp, 0.0)[0], _laplace(sm, 0.0)[0]
                inf_ok = True
            except MEUnavailable as exc:
                notes.append(str(exc))
                Ip = In = nan
                inf_ok = False
            if om is not None:
                if not inf_ok:
                    S = np.full(om.shape, nan)
                elif isinstance(sp, ExactDelayTreeSide):
                    from daedalus.api.me_output import exact_delay_spectrum
                    S = exact_delay_spectrum(sp.tree, -om)
                else:
                    S = _laplace(sp, -1j * om) + _laplace(sm, 1j * om)
            if Ts is not None:
                if isinstance(sp, ExactDelayTreeSide):
                    # no triple: the windows by quadrature of the exact tree
                    q = _QuadSide(lambda t: complex(
                        (sp if t >= 0 else sm)(np.array([abs(t)]))[0]), *qkw)
                    wp = [q.window(1.0, t, (Ip, 0.0)) for t in Ts]
                    wn = [q.window(-1.0, t, (In, 0.0)) for t in Ts]
                    IT = np.array([v for v, _ in wp], dtype=complex)
                    ITn = np.array([v for v, _ in wn], dtype=complex)
                    er = {'integral': np.array([x for _, x in wp]),
                          'integral_neg': np.array([x for _, x in wn])}
                    meth[e] = 'exact+quad'
                    notes.append(f'ell={e}: the exact delayed tree has no '
                                 f'triple: its windows ∫_0^T are adaptive '
                                 f'quadrature (error_by_ell); S and ∫_0^∞ '
                                 f'are exact')
                else:
                    IT = _window_exact(sp, Ts)
                    ITn = _window_exact(sm, Ts)
            errs[e] = er
        out_C0[e] = C0
        out_Iinf[e] = complex(Ip) + d_in(0.0, np.inf)
        out_Iinfn[e] = complex(In) + sum(
            (d.weight for d in dl if d.location < 0.0), 0j)
        if om is not None:
            out_S[e] = S + dS
        if Ts is not None:
            out_I[e] = IT + np.array([d_in(0.0, t) for t in Ts],
                                     dtype=complex)
            out_In[e] = ITn + np.array(
                [sum((d.weight for d in dl if -t <= d.location < 0.0), 0j)
                 for t in Ts], dtype=complex)

    def tot(d):
        vals = [d[e] for e in orders if e in d]
        return None if not vals else sum(vals[1:], vals[0])
    # S(0) = ∫_0^∞ + ∫_{-∞}^0 (the resolvents at 0, with the spikes off
    # τ = 0) + the spike at τ = 0
    S0_by = {e: out_Iinf[e] + out_Iinfn[e] + out_d0[e] for e in orders}
    tau_by, cum_I, cum_C = {}, 0j, 0j
    for e in orders:
        cum_I += out_Iinf[e]
        cum_C += out_C0[e]
        tau_by[e] = cum_I / cum_C if cum_C != 0 else nan
    err_tot = None
    if any(v is not None for v in errs.values()):
        err_tot = {}
        for v in errs.values():
            if v is None:
                continue
            for kq, x in v.items():
                err_tot[kq] = (x if kq not in err_tot
                               else err_tot[kq] + x)
    res = {
        'omega': om,
        'S': tot(out_S),
        'S_by_ell': out_S if om is not None else {},
        'S0': tot(S0_by),
        'S0_by_ell': S0_by,
        'T': Ts,
        'integral': tot(out_I),
        'integral_by_ell': out_I if Ts is not None else {},
        'integral_neg': tot(out_In),
        'integral_neg_by_ell': out_In if Ts is not None else {},
        'integral_inf': tot(out_Iinf),
        'integral_inf_by_ell': out_Iinf,
        'integral_inf_neg': tot(out_Iinfn),
        'integral_inf_neg_by_ell': out_Iinfn,
        'delta0': tot(out_d0),
        'delta0_by_ell': out_d0,
        'C0': tot(out_C0),
        'C0_by_ell': out_C0,
        'tau_c': tau_by[orders[-1]] if orders else nan,
        'tau_c_by_ell': tau_by,
        'method_by_ell': meth,
        'error_by_ell': errs,
        'abs_error': err_tot,
        'status': 'ok',
        'notes': notes,
        'conventions': CONVENTIONS,
    }
    dl_rec = result.get('delay') if isinstance(result.get('delay'), dict) \
        else None
    if dl_rec is not None:
        est = dl_rec.get('est_rel_error')
        res['notes'].append(
            f"delay model: the loop orders are exact for the Padé "
            f"approximant of order n = {dl_rec.get('n')}, whose estimated "
            f"relative error is {est if est is not None else float('nan'):.3g} "
            f"(result['delay']); the tree is "
            f"{'the exact delayed covariance' if dl_rec.get('tree') == 'exact' else 'the Padé one'}")
    return res


def attach(result, opts):
    """``result['spectra']`` for ``compute_cumulants(spectra=opts)``
    (``opts`` from :func:`resolve_spectra_options`).  A refusal at this
    stage (the run is kept) is recorded as ``{'status': 'refused',
    'reason': ...}`` with a :class:`SpectraWarning`."""
    from daedalus.api.me_output import MEUnavailable
    try:
        result['spectra'] = spectra(
            result, opts['omega'], opts['T'], method=opts['method'],
            epsabs=opts['epsabs'], epsrel=opts['epsrel'],
            limit=opts['limit'])
    except (MEUnavailable, NotImplementedError) as exc:
        result['spectra'] = {'status': 'refused', 'reason': str(exc),
                             'conventions': CONVENTIONS}
        msg = (f"spectra could not be computed: {exc} -- see "
               f"result['spectra']")
        from daedalus.api.save import _ensure_warning_shown
        _ensure_warning_shown(SpectraWarning, 'always', message=msg,
                              stacklevel=3)
        warnings.warn(msg, SpectraWarning, stacklevel=3)
    return result
