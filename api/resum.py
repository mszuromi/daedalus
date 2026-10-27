"""
api.resum
=========
Opt-in resummations of the loop series (``dd.Config(resum=...)``):

* ``'hartree'``: the self-consistent Gaussian (:mod:`api.hartree`), the sum
  of every cactus diagram.  Bounded at any coupling; Gaussian (``κ_k = 0`` for
  ``k >= 3``) and a single matrix exponential in ``τ``.
* ``'screened'``: the existing loop orders re-expanded around a variational
  trial propagator (optimized / variational perturbation theory), with the
  trial fixed by the principle of minimal sensitivity.  It needs no new
  diagram: only the loop orders evaluated at a few trial propagators.

The screened series
-------------------
Write the bilinear part of the action about the mean-field saddle with a
trial drift ``A_t`` and noise ``Q_t`` in place of the tree-level ``J0`` and
``Q0``, and the difference as a two-point counterterm counted like one loop:

    ẋ = A(λ) x + ...,   A(λ) = A_t + λ (J0 - A_t),   Q(λ) = Q_t + λ (Q0 - Q_t).

At ``λ = 1`` this is the model.  Each loop order ``C_ℓ`` (vertices
unchanged) is evaluated with the propagator of ``A(λ)``; the counterterm
insertions are its ``λ``-derivatives, so the order-``N`` screened correlator
is

    C_N = Σ_{ℓ + n <= N} (1/n!) ∂_λ^n C_ℓ(A(λ), Q(λ)) |_{λ=0}.

The trial runs along the line ``A(κ) = J0 + κ (A_H - J0)`` (``Q`` alike)
through the Hartree closure (``κ = 1``) and the tree (``κ = 0``).  With
``F_ℓ(s) = C_ℓ(J0 + sΔ, Q0 + sΔ_Q)`` the series is the Taylor polynomial of
each ``F_ℓ`` about ``s = κ``, of degree ``N - ℓ``, evaluated at ``s = 0``:

    C_N(κ) = Σ_ℓ Σ_{n <= N-ℓ} (-κ)^n / n!  F_ℓ^{(n)}(κ).

``C_1`` at ``κ = 1`` is Hartree (the counterterm cancels the tadpole) for a
model whose closure keeps the mean at the saddle.

The counterterm is injected as extra bilinear action terms
``-Σ h_ab xt_a dx_b - ½ Σ q_ab xt_a xt_b`` with numeric parameters, so every
Phase J route runs unchanged; the model copy is named
``'<name> [screened]'`` (its own caches).

Choosing κ (``mass=``)
~~~~~~~~~~~~~~~~~~~~~~
* ``'pms'`` (default): the stationary point of ``C_N(κ)`` (equal time,
  ``observable``) nearest to the Hartree value ``κ = 1``.  The scan runs the
  loop orders at the equal-time point only, on Chebyshev nodes in
  ``log κ`` over ``kappa_range``; ``C_N(κ)`` and its κ-derivative are then
  analytic in the interpolant.  An order without a stationary point (even
  orders, generically) takes the mass of the next lower order that has one,
  and failing that Hartree's (``κ = 1``); ``resummation['kappa_source']``
  says which.
* ``'hartree'``: ``κ = 1``.  At strong coupling the series at a fixed mass
  diverges again from two orders on; PMS is the convergent choice.
* a number: that ``κ``.

The final ``C_N(τ)`` comes from a central stencil of runs in ``s`` about
``κ`` (relative step ``step``; 7 points through ``N = 3``, 9 at ``N = 4,
5``).

Error estimate: ``C_tau_spread = C_N - C_{N-1}`` at the same mass, and its
maximum relative to ``max |C_N|`` (``resummation['spread_last_two']``).
In the OU quartic tests it is conservative (7–280x the true error at equal
time from ``g = 0.01`` to ``1000``); it is not a bound.

Kernels and hard delays: the Hartree closure embeds them (``api.hartree``:
rational kernels exactly, delays through their Pade approximant), the trial
propagator is its block on the model's own fields, and the loop orders run
the loop path's kernel / delay routes (the options of ``Config.delay``).

Out of scope (future work): exact ``λ``-derivatives through Fréchet
derivatives of the state-space exponentials (in place of the stencil), the
self-consistent sunset of the 2PI expansion at next-to-leading order, spatial
models (the spatial tadpole needs a mass counterterm), exogenous input
fields.
"""
from __future__ import annotations

import math
import numbers
import time
import warnings

import numpy as np

__all__ = ['resolve_resum_options', 'screened', 'screened_model',
           'RESUM_METHODS', 'ResumWarning']

RESUM_METHODS = ('hartree', 'screened')

#: the keys of a ``resum={...}`` dict
RESUM_KEYS = ('method', 'mass', 'observable', 'step', 'kappa_range',
              'n_scan', 'taylor_order')

#: defaults of the screened series
SCREENED_DEFAULTS = {'mass': 'pms', 'observable': 'trace_cov', 'step': 5e-3,
                     'kappa_range': (0.25, 6.0), 'n_scan': 17}

#: the spectra keys that are linear in C (combined like C_tau)
_LINEAR_SPECTRA = ('S', 'S0', 'integral', 'integral_neg', 'integral_inf',
                   'integral_inf_neg', 'delta0', 'C0')

#: below this ``||Δ|| / ||J0||`` the trial equals the tree (a linear model):
#: the screened series is the loop series
_TRIVIAL_SHIFT = 1e-13


class ResumWarning(UserWarning):
    """A resummation note (no PMS stationary point, a narrowed κ range)."""


def resolve_resum_options(value, where='resum'):
    """``dd.Config.resum`` as a dict (``None`` when off): ``'hartree'``,
    ``'screened'`` or a dict with ``method`` and the screened options
    ``mass`` ('pms' | 'hartree' | κ), ``observable`` ('trace_cov' | 'C0' |
    a list of fields), ``step``, ``kappa_range``, ``n_scan``,
    ``taylor_order`` (of the Hartree closure)."""
    if value is None or value is False:
        return None
    if isinstance(value, str):
        value = {'method': value}
    if not isinstance(value, dict):
        raise ValueError(f"{where} must be None, 'hartree', 'screened' or a "
                         f"dict with keys {list(RESUM_KEYS)}; got {value!r}")
    unknown = sorted(set(value) - set(RESUM_KEYS))
    if unknown:
        raise ValueError(f'{where}: unknown key(s) {unknown}; known: '
                         f'{list(RESUM_KEYS)}')
    method = value.get('method')
    if method not in RESUM_METHODS:
        raise ValueError(f'{where}: method must be one of {RESUM_METHODS}; '
                         f'got {method!r}')
    out = {'method': method, 'taylor_order': value.get('taylor_order')}
    if method == 'hartree':
        extra = sorted(set(value) - {'method', 'taylor_order'})
        if extra:
            raise ValueError(f"{where}: {extra} are options of the screened "
                             f"series, not of method='hartree'")
        return out
    out.update({k: value.get(k, v) for k, v in SCREENED_DEFAULTS.items()})
    mass = out['mass']
    if isinstance(mass, numbers.Real) and not isinstance(mass, (bool,
                                                                 np.bool_)):
        mass = out['mass'] = float(mass)
    if not (mass in ('pms', 'hartree') or (
            isinstance(mass, float) and math.isfinite(mass) and mass > 0)):
        raise ValueError(f"{where}: mass must be 'pms', 'hartree' or a "
                         f"number kappa > 0; got {mass!r}")
    obs = out['observable']
    if not (obs in ('trace_cov', 'C0') or (isinstance(obs, (list, tuple))
                                           and len(obs) > 0)):
        raise ValueError(f"{where}: observable must be 'trace_cov', 'C0' or "
                         f"a list of fields; got {obs!r}")
    step = out['step']
    if not (isinstance(step, numbers.Real) and 0 < step < 0.1):
        raise ValueError(f'{where}: step must be in (0, 0.1); got {step!r}')
    try:
        lo, hi = (float(v) for v in out['kappa_range'])
    except (TypeError, ValueError) as exc:
        raise ValueError(f'{where}: kappa_range must be a pair (lo, hi); got '
                         f'{out["kappa_range"]!r}') from exc
    if not (0 < lo < 1 < hi):
        raise ValueError(f'{where}: kappa_range must satisfy 0 < lo < 1 < hi '
                         f'(it brackets the Hartree value 1); got '
                         f'{out["kappa_range"]!r}')
    out['kappa_range'] = (lo, hi)
    n = out['n_scan']
    if isinstance(n, numbers.Integral) and not isinstance(n, (bool,
                                                              np.bool_)):
        n = out['n_scan'] = int(n)
    if not (isinstance(n, int) and n >= 9 and n % 2 == 1):
        raise ValueError(f'{where}: n_scan must be an odd integer >= 9; got '
                         f'{n!r}')
    return out


def check_config(opts, *, k, max_ell, output, t_min, spatial_points,
                 covariance, where='Config.resum', spectra=None):
    """The ``dd.Config`` combinations a resummation refuses (``ValueError``)."""
    if opts is None:
        return
    for bad, why in ((output != 'cumulant',
                      f'output={output!r} (the resummed quantity is the '
                      f'connected cumulant)'),
                     (t_min is not None, 't_min (stationary state only)'),
                     (spatial_points is not None, 'spatial_points')):
        if bad:
            raise ValueError(f'{where} with {why} is not supported.')
    if covariance is not None and covariance.get('derivatives'):
        raise ValueError(f'{where}: covariance derivatives are not '
                         f'supported with a resummation.')
    if opts['method'] == 'hartree':
        if max_ell not in (None, 0):
            raise ValueError(
                f"{where}='hartree' is the self-consistent Gaussian on its "
                f"own: leave max_ell unset or 0 (got {max_ell!r}); the loop "
                f"orders on top of it are method='screened'.")
        if k not in (None, 1, 2):
            raise ValueError(
                f"{where}='hartree' is Gaussian: k must be 1 or 2 (its "
                f"connected k >= 3 cumulants vanish); use method='screened' "
                f"for the first non-Gaussian answer.")
    else:
        if k not in (None, 1, 2):
            raise ValueError(f"{where}: method='screened' is implemented for "
                             f"k = 1 (the resummed mean) and k = 2 (got "
                             f"k={k!r}).")
        if k == 1 and spectra is not None:
            raise ValueError(f"{where}: method='screened' at k = 1 has no "
                             f"spectra (the mean is a number).")
        if k == 1 and covariance is not None:
            raise ValueError(f"{where}: method='screened' at k = 1 has no "
                             f"covariance matrix (it is a k = 2 object).")
        if max_ell is not None and max_ell < 1:
            raise ValueError(f"{where}: method='screened' needs max_ell >= 1 "
                             f"(the screened order N = max_ell).")


# ── the counterterm model ──────────────────────────────────────────────────

def _free_name(base, taken):
    name = base
    k = 0
    while name in taken:
        k += 1
        name = f'{base}_{k}'
    taken.add(name)
    return name


def screened_model(model, h, *, rtol=1e-13):
    """The model with the screening counterterm, and the parameter values
    along the trial line.

    Returns ``(model_ct, values)``: ``model_ct`` adds the action terms
    ``-Σ h_ab xt_a dx_b - ½ Σ q_ab xt_a xt_b`` (``g``-coordinates of the
    action: ``h = M Δ``, ``q = M Δ_Q Mᵀ``) on the structurally nonzero
    entries of ``Δ = A_H - J0`` and ``Δ_Q = Q_H - Q0``; ``values(s)`` maps a
    trial position ``s`` to its parameter dict (``s = 0``: the model,
    ``s = 1``: the Hartree propagator).  ``model_ct`` is ``None`` when
    ``Δ`` and ``Δ_Q`` vanish (a linear model)."""
    import hashlib
    M = np.asarray(h.info['pencil'], dtype=float)
    D = M @ (h.A - h.J0)
    DQ = M @ (h.Q - h.Q0) @ M.T
    DQ = 0.5 * (DQ + DQ.T)
    scale = max(np.max(np.abs(h.J0)), np.max(np.abs(h.Q0)), 1e-300)
    n = D.shape[0]
    taken = {p['name'] for p in model.get('parameters', [])}
    entries = []                      # (name, kind, a, b, value)
    for a in range(n):
        for b in range(n):
            if abs(D[a, b]) > rtol * scale:
                entries.append((_free_name(f'scr_h_{a}_{b}', taken), 'h', a,
                                b, float(D[a, b])))
    for a in range(n):
        for b in range(a, n):
            if abs(DQ[a, b]) > rtol * scale:
                entries.append((_free_name(f'scr_q_{a}_{b}', taken), 'q', a,
                                b, float(DQ[a, b])))
    if not entries:
        return None, (lambda s: {})
    orig = model['action']

    def action(ns):
        from sage.all import SR
        t, p = ns._tilde_sr_vars, ns._phys_sr_vars
        S = SR(orig(ns))
        for name, kind, a, b, _v in entries:
            c = getattr(ns, name)
            if kind == 'h':
                S = S - c * t[a] * p[b]
            elif a == b:
                S = S - c * t[a] ** 2 / 2
            else:
                S = S - c * t[a] * t[b]
        return S

    pattern = ';'.join(f'{nm}:{kd}{a}{b}' for nm, kd, a, b, _v in entries)
    m = dict(model)
    m['action'] = action
    m['parameters'] = list(model.get('parameters', [])) + [
        {'name': nm, 'indexed': False, 'default': 0.0,
         'description': 'screened-series counterterm'}
        for nm, *_rest in entries]
    m['name'] = f"{model.get('name', 'model')} [screened]"
    m['spec_signature'] = hashlib.sha256(
        f"{model.get('spec_signature')}|screened|{pattern}".encode()
    ).hexdigest()[:16]
    m['_screened_counterterm'] = [e[:4] for e in entries]

    def values(s):
        return {nm: float(s) * v for nm, _k, _a, _b, v in entries}
    return m, values


# ── finite differences and Chebyshev derivatives ──────────────────────────

def fd_weights(z, x, m):
    """Fornberg's weights ``w[d, j]`` of the derivatives ``d = 0..m`` at
    ``z`` from the values at the nodes ``x``."""
    x = np.asarray(x, dtype=float)
    n = len(x) - 1
    c = np.zeros((n + 1, m + 1))
    c1, c4 = 1.0, x[0] - z
    c[0, 0] = 1.0
    for i in range(1, n + 1):
        mn = min(i, m)
        c2, c5, c4 = 1.0, c4, x[i] - z
        for j in range(i):
            c3 = x[i] - x[j]
            c2 *= c3
            if j == i - 1:
                for k in range(mn, 0, -1):
                    c[i, k] = c1 * (k * c[i - 1, k - 1] - c5 * c[i - 1, k]) / c2
                c[i, 0] = -c1 * c5 * c[i - 1, 0] / c2
            for k in range(mn, 0, -1):
                c[j, k] = (c4 * c[j, k] - k * c[j, k - 1]) / c3
            c[j, 0] = c4 * c[j, 0] / c3
        c1 = c2
    return c.T


def _stirling1(n):
    """Coefficients ``c[k]`` of ``θ(θ-1)...(θ-n+1) = Σ_k c[k] θ^k``."""
    poly = np.array([1.0])
    for j in range(n):
        poly = np.convolve(poly, [1.0, -float(j)])       # highest power first
    return poly[::-1]


class _LogCheb:
    """Chebyshev interpolant of ``F(s)`` in ``u = log s`` on Lobatto nodes,
    with derivatives in ``s``.  A ``F`` of one sign on the nodes is
    interpolated as ``log |F|`` (its values span many decades at strong
    coupling, where an interpolant of ``F`` itself loses the small end), and
    its derivatives are rebuilt by ``(e^L)^{(n)} = Σ_k C(n-1, k) L^{(k+1)}
    (e^L)^{(n-1-k)}``."""

    def __init__(self, s_nodes, values, u_lo, u_hi):
        from numpy.polynomial import chebyshev as Ch
        self.Ch = Ch
        self.lo, self.hi = u_lo, u_hi
        values = np.asarray(values, dtype=float)
        t = (2 * np.log(s_nodes) - (u_hi + u_lo)) / (u_hi - u_lo)
        self.sign = 0.0
        if np.all(values > 0) or np.all(values < 0):
            self.sign = float(np.sign(values[0]))
            values = np.log(np.abs(values))
        self.coef = Ch.chebfit(t, values, len(values) - 1)

    def _t(self, s):
        return (2 * np.log(s) - (self.hi + self.lo)) / (self.hi - self.lo)

    def _du(self, s, k):
        """``d^k g / du^k`` of the interpolated function ``g``."""
        t = self._t(s)
        if k == 0:
            return self.Ch.chebval(t, self.coef)
        return (2.0 / (self.hi - self.lo)) ** k * self.Ch.chebval(
            t, self.Ch.chebder(self.coef, k))

    def _Fu(self, s, n):
        """``[d^m F / du^m for m = 0..n]``."""
        if not self.sign:
            return [self._du(s, m) for m in range(n + 1)]
        L = [self._du(s, m) for m in range(n + 1)]
        F = [self.sign * np.exp(L[0])]
        for m in range(1, n + 1):
            F.append(sum(math.comb(m - 1, k) * L[k + 1] * F[m - 1 - k]
                         for k in range(m)))
        return F

    def deriv(self, s, n):
        """``d^n F / ds^n`` at ``s`` (array)."""
        s = np.asarray(s, dtype=float)
        Fu = self._Fu(s, n)
        if n == 0:
            return Fu[0]
        st = _stirling1(n)
        acc = np.zeros_like(s)
        for k in range(1, n + 1):
            if st[k] != 0:
                acc = acc + st[k] * Fu[k]
        return acc / s ** n


def _opt_value(derivs, kappa, N):
    """``C_N(κ) = Σ_ℓ Σ_{n <= N-ℓ} (-κ)^n/n! F_ℓ^{(n)}(κ)`` from
    ``derivs[ℓ][n]`` (arrays broadcasting with ``kappa``)."""
    tot = 0.0
    for ell in range(N + 1):
        for n in range(N - ell + 1):
            tot = tot + (-kappa) ** n / math.factorial(n) * derivs[ell][n]
    return tot


# ── one trial point ─────────────────────────────────────────────────────────

def _note_status(log, r, s):
    """Record a trial run that is not ``'ok'`` (diagrams left out)."""
    if log is not None and isinstance(r, dict) and r.get('status') not in (
            None, 'ok'):
        log.append({'s': float(s), 'status': r.get('status'),
                    'phase_j_incomplete': r.get('phase_j_incomplete')})


def _run_pairs(model_ct, values, s, pairs, max_ell, params, tau, run_kw,
               spectra=None, log=None, k=2):
    """The loop orders of every leg tuple (pairs for ``k = 2``, single legs
    for ``k = 1``) at trial position ``s``: a list of ``{ell: array}`` (one
    per tuple), and the spectra by ell of the first.  A run that is not
    ``'ok'`` is recorded in ``log``."""
    from api.compute import compute_cumulants, stage_memo
    par = dict(params)
    par.update(values(s))
    out, spec = [], None
    with stage_memo():
        for n, pr in enumerate(pairs):
            kw = dict(run_kw)
            if spectra is not None and n == 0:
                kw['spectra'] = spectra
            r = compute_cumulants(model_ct, k, max_ell, par, list(pr),
                                  tau_grid=tau, **kw)
            _note_status(log, r, s)
            out.append({int(e): np.asarray(v)
                        for e, v in r['C_tau_by_ell'].items()})
            if spectra is not None and n == 0:
                spec = r.get('spectra')
    return out, spec


def _run_cov(model_ct, values, s, cov, max_ell, params, tau, run_kw,
             log=None):
    from api.covariance import covariance_matrix
    par = dict(params)
    par.update(values(s))
    kw = {k: v for k, v in run_kw.items() if k != 'verbose'}
    r = covariance_matrix(model_ct, cov.get('fields'), max_ell,
                          parameters=par, pairs=cov.get('pairs', 'upper'),
                          tau_grid=tau, verbose=run_kw.get('verbose', False),
                          **kw)
    _note_status(log, r, s)
    return {int(e): np.asarray(v) for e, v in r['Sigma_by_ell'].items()}, r


# ── the screened series ────────────────────────────────────────────────────

def _trial_matrices(h):
    """``(A_H - J0, Q_H - Q0, J0, Q0)`` on the closure's whole state: the
    fields and, for a model with kernels, the kernel states (whose rows do
    not change along the trial line)."""
    if h.full is not None:
        f = h.full
        return f['A'] - f['J0'], f['Q'] - f['Q0'], f['J0'], f['Q0']
    return h.A - h.J0, h.Q - h.Q0, h.J0, h.Q0


def _stable_range(h, lo, hi):
    """The largest sub-interval of ``[lo, hi]`` around ``s = 1`` where the
    trial drift is stable and the trial noise positive semi-definite."""
    D, DQ, J0, Q0 = _trial_matrices(h)

    def ok(s):
        A = J0 + s * D
        Q = Q0 + s * DQ
        return (np.max(np.linalg.eigvals(A).real) < 0
                and np.min(np.linalg.eigvalsh(0.5 * (Q + Q.T)))
                >= -1e-12 * max(1.0, np.max(np.abs(Q))))
    grid = np.geomspace(lo, hi, 241)
    good = np.array([ok(s) for s in grid])
    i1 = int(np.argmin(np.abs(np.log(grid))))
    if not good[i1]:
        return None
    a = i1
    while a > 0 and good[a - 1]:
        a -= 1
    b = i1
    while b < len(grid) - 1 and good[b + 1]:
        b += 1
    # one grid cell of margin inside a boundary that was hit
    lo2 = grid[a] if a == 0 else grid[a + 1]
    hi2 = grid[b] if b == len(grid) - 1 else grid[b - 1]
    if not lo2 < 1.0 < hi2:
        return None                 # stable only in a sliver around s = 1
    return float(lo2), float(hi2)


def _stationary_points(derivs_fn, N, klo, khi):
    """The stationary points of ``C_N(κ)`` on ``[klo, khi]``:
    ``dC_N/dκ = Σ_ℓ F_ℓ^{(N-ℓ+1)}(κ) (-κ)^{N-ℓ} / (N-ℓ)!``."""
    from scipy.optimize import brentq

    def dC(k):
        k = np.asarray(k, dtype=float)
        tot = 0.0
        for ell in range(N + 1):
            m = N - ell
            tot = tot + derivs_fn(ell, m + 1, k) * (-k) ** m / math.factorial(m)
        return tot
    ks = np.geomspace(klo, khi, 801)
    d = dC(ks)
    roots = []
    for i in np.flatnonzero(np.sign(d[:-1]) * np.sign(d[1:]) < 0):
        try:
            roots.append(brentq(lambda k: float(dC(k)), ks[i], ks[i + 1],
                                xtol=1e-14, rtol=1e-13))
        except ValueError:
            continue
    return roots


def screened(model, max_ell, parameters=None, *, external_fields,
             options=None, tau_grid, spectra=None, covariance=None,
             fixed_point_index=0, mf_dae_n_starts=64, mf_dae_seed_box=None,
             hartree_result=None, verbose=False, k=2, **run_kw):
    """The screened (optimized-perturbation) loop series at order
    ``N = max_ell`` for one ``k = 2`` pair (``external_fields``), or for the
    covariance matrix (``covariance=``, the ``dd.Config.covariance`` dict), or
    -- ``k = 1`` -- for the resummed mean of one field.  The mean uses the
    SAME trial propagator as the covariance of that model: the principle of
    minimal sensitivity is read off the equal-time covariance (the
    ``observable`` option, ``'trace_cov'`` by default), so a ``k = 1`` run
    and a ``k = 2`` run at one parameter point have the same ``kappa``.

    ``options``: the screened keys of :func:`resolve_resum_options`.
    ``run_kw``: ``compute_cumulants`` options of every run (``use_cache``,
    ``parallel``, ``n_workers``, ``policy``, ``preflight``, ``backend``,
    ``accuracy``, ``taylor_order``).  The gates (``preflight``) run on the
    plain loop run only.

    Returns the plain run's result dict with ``C_tau`` (order ``N``),
    ``C_tau_by_order`` ``{N: C_N}``, ``C_tau_spread``, ``C_tau_loops`` /
    ``C_tau_loops_by_ell`` (the plain loop series) and ``resummation``
    (``kappa``, ``kappa_source``, ``A_trial``, ``Q_trial``, ``pms_curve``,
    ``spread_last_two``, ``hartree``, ...); with ``spectra``, ``spectra``
    holds the screened spectral values by order; with ``covariance``, the
    ``covariance_matrix`` layout with ``Sigma_by_order``.
    """
    from api.compute import compute_cumulants, as_model_dict
    from api.hartree import hartree, refusal_reason, HartreeUnsupported
    opts = resolve_resum_options(dict({'method': 'screened'},
                                      **(options or {})))
    model = as_model_dict(model)
    why = refusal_reason(model)
    if why is not None:
        raise HartreeUnsupported('resum: ' + why + '.')
    N = int(max_ell)
    if N < 1:
        raise ValueError('screened series: max_ell must be >= 1')
    t_start = time.perf_counter()
    tau = np.atleast_1d(np.asarray(tau_grid, dtype=float))
    params = dict(parameters or {})
    mf_kw = dict(fixed_point_index=fixed_point_index,
                 mf_dae_n_starts=mf_dae_n_starts,
                 mf_dae_seed_box=mf_dae_seed_box)
    h = hartree_result if hartree_result is not None else hartree(
        model, params, taylor_order=opts.get('taylor_order'),
        delay=run_kw.get('delay'), **mf_kw)
    model_ct, values = screened_model(model, h)
    inner = dict(run_kw, verbose=verbose, preflight=False, **mf_kw)
    plain_kw = dict(run_kw, verbose=verbose, **mf_kw)
    legs = [tuple(e) if isinstance(e, (list, tuple)) else e
            for e in external_fields]
    if k not in (1, 2):
        raise ValueError(f'screened series: k must be 1 (the mean) or 2; '
                         f'got {k!r}')
    if k == 1:
        if len(legs) != 1:
            raise ValueError(f'screened series: k = 1 takes one external '
                             f'field; got {legs}')
        if spectra is not None or covariance is not None:
            raise ValueError('screened series: k = 1 has no spectra or '
                             'covariance matrix')
    # PMS reads the equal-time covariance whatever k is: the mean of a field
    # is resummed with the propagator that is stationary for its variance
    scan_legs = legs if k == 2 else [legs[0], legs[0]]
    n_runs = 0
    trial_log = []

    # the plain loop series (the comparison, the result's metadata, the gates)
    if covariance is not None:
        from api.covariance import covariance_matrix
        base = covariance_matrix(
            model, covariance.get('fields'), N, parameters=params,
            pairs=covariance.get('pairs', 'upper'), tau_grid=tau,
            **{k: v for k, v in plain_kw.items()})
        plain_by_ell = {int(e): np.asarray(v)
                        for e, v in base['Sigma_by_ell'].items()}
    else:
        base = compute_cumulants(model, k, N, params, legs, tau_grid=tau,
                                 **plain_kw)
        plain_by_ell = {int(e): np.asarray(v)
                        for e, v in base['C_tau_by_ell'].items()}
    n_runs += 1
    info = {'method': 'screened', 'order': N, 'mass': opts['mass'],
            'observable': opts['observable'], 'step': opts['step'],
            'hartree': {'mean': h.mean, 'cov': h.cov, 'A': h.A, 'Q': h.Q,
                        'J0': h.J0, 'Q0': h.Q0, 'rates': h.rates,
                        'advisories': h.advisories,
                        'solutions': h.solutions, 'fields': h.fields},
            'counterterm': (None if model_ct is None
                            else model_ct['_screened_counterterm'])}

    if model_ct is None:
        # linear model: the trial is the tree, the series is the loop series
        orders = {M: sum(plain_by_ell[e] for e in range(M + 1)
                         if e in plain_by_ell) for M in range(N + 1)}
        spec_by = None
        info.update({'kappa': 1.0, 'kappa_source': 'trivial (A_H = J0)',
                     'pms_curve': None, 'stationary_points': {}})
    else:
        D, DQ = h.A - h.J0, h.Q - h.Q0
        Df, _DQf, J0f, _Q0f = _trial_matrices(h)
        rng = _stable_range(h, *opts['kappa_range'])
        if rng is None:
            raise ValueError('screened series: the Hartree trial propagator '
                             'is not stable')
        klo, khi = rng
        if (klo, khi) != tuple(opts['kappa_range']):
            warnings.warn(f'screened series: kappa range narrowed to '
                          f'[{klo:.4g}, {khi:.4g}] where the trial drift is '
                          f'stable', ResumWarning, stacklevel=3)
        info['kappa_range'] = (klo, khi)
        mass = opts['mass']
        pms = None
        if mass == 'pms':
            pms = _pms_scan(model_ct, values, h, scan_legs, N, params, inner,
                            opts, klo, khi, log=trial_log)
            n_runs += pms['n_runs']
            kappa, source = pms['kappa'], pms['source']
            info.update({'pms_curve': pms['curve'],
                         'stationary_points': pms['stationary'],
                         'pms_interp_change': pms['interp_change'],
                         'pms_linear_orders': pms['linear_orders'],
                         'pms_value': pms['value']})
        elif mass == 'hartree':
            kappa, source = 1.0, 'hartree'
        else:
            kappa, source = float(mass), 'given'
        if mass == 'pms' and source != f'pms order {N}':
            warnings.warn(f'screened series: {source}', ResumWarning,
                          stacklevel=3)
        # the final stencil about kappa
        half = max(3, N // 2 + 2)
        hstep = opts['step'] * kappa
        nodes = kappa + hstep * np.arange(-half, half + 1)
        if nodes[0] <= 0 or nodes[0] < klo * (1 - 1e-12) or \
                nodes[-1] > khi * (1 + 1e-12):
            if not all(np.max(np.linalg.eigvals(J0f + s * Df).real) < 0
                       for s in nodes):
                raise ValueError(
                    f'screened series: the stencil about kappa = {kappa:g} '
                    f'leaves the range where the trial drift is stable')
        W = fd_weights(kappa, nodes, N)
        point_vals, spec_pts = [], []
        for s in nodes:
            if covariance is not None:
                v, _r = _run_cov(model_ct, values, s, covariance, N, params,
                                 tau, inner, log=trial_log)
                point_vals.append(v)
            else:
                v, sp = _run_pairs(model_ct, values, s, [legs], N, params,
                                   tau, inner, spectra=spectra, log=trial_log,
                                   k=k)
                point_vals.append(v[0])
                spec_pts.append(sp)
            n_runs += 1
        derivs = {e: [sum(W[n, j] * point_vals[j][e]
                          for j in range(len(nodes)))
                      for n in range(N - e + 1)] for e in range(N + 1)}
        orders = {M: _opt_value(derivs, kappa, M) for M in range(N + 1)}
        spec_by = None
        if spectra is not None and all(sp is not None for sp in spec_pts):
            spec_by = _combine_spectra(spec_pts, W, kappa, N)
        A_t = h.J0 + kappa * D
        Q_t = h.Q0 + kappa * DQ
        info.update({'kappa': kappa, 'kappa_source': source,
                     'A_trial': A_t, 'Q_trial': Q_t,
                     'stencil': {'nodes': nodes, 'step': hstep}})
        if pms is not None and covariance is None and k == 2 and [
                [h.index(a) for a in pr] for pr in pms['pairs']] == [
                [h.index(a) for a in legs]]:
            v0 = _at_zero(orders[N], tau)
            if v0 is not None:
                info['pms_stencil_check'] = abs(v0 - pms['value']) / max(
                    abs(pms['value']), 1e-300)
    top = orders[N]
    spread = orders[N] - orders[N - 1]
    info['spread_last_two'] = float(np.max(np.abs(spread)) / max(
        np.max(np.abs(top)), 1e-300))
    info['n_runs'] = n_runs
    info['wall'] = time.perf_counter() - t_start

    out = dict(base)
    for key in ('C_tau_by_ell', 'total_C', 'total_C_by_ell', 'phase_j_by_ell',
                'spectra', 'Sigma_by_ell', 'derivatives'):
        out.pop(key, None)
    if covariance is not None:
        out['Sigma_loops'] = base['Sigma']
        out['Sigma_loops_by_ell'] = plain_by_ell
        out['Sigma'] = top
        out['Sigma_by_order'] = orders
        out['Sigma_spread'] = spread
        zero = np.flatnonzero(tau == 0.0)
        out['Sigma0'] = top[zero[0]] if zero.size else None
    else:
        out['C_tau_loops'] = base['C_tau']
        out['C_tau_loops_by_ell'] = plain_by_ell
        out['C_tau'] = top
        out['C_tau_by_order'] = orders
        out['C_tau_spread'] = spread
        if spec_by is not None:
            out['spectra'] = spec_by
        if k == 1:
            # the mean itself (the saddle plus the shift), as dd.hartree gives it
            i = h.index(legs[0])
            base_mean = float(h.saddle[i])
            out['mean'] = base_mean + float(np.real(np.asarray(top).ravel()[0]))
            out['mean_by_order'] = {
                M: base_mean + float(np.real(np.asarray(v).ravel()[0]))
                for M, v in orders.items()}
    out['resummation'] = info
    # the status covers every run the result is built from: the plain run
    # (already in ``base``) and each trial run of the scan and the stencil
    info['trial_runs_not_ok'] = trial_log
    if trial_log:
        out['status'] = 'incomplete'
    if isinstance(out.get('config'), dict):
        out['config'] = dict(out['config'], resum='screened')
    for a in h.advisories:
        if a['code'] == 'HARTREE_MULTIPLE_SOLUTIONS':
            info.setdefault('notes', []).append(
                'the Hartree closure is multistable; the trial propagator is '
                'that of one basin')
        elif a['code'] == 'HARTREE_LEFT_SADDLE':
            info.setdefault('notes', []).append(
                'the Hartree closure settled nearer another mean-field root '
                'than the expansion saddle: the trial propagator belongs to '
                'another basin than the vertices')
    return out


def _at_zero(arr, tau):
    z = np.flatnonzero(np.asarray(tau) == 0.0)
    if not z.size:
        return None
    return float(np.real(np.asarray(arr).ravel()[z[0]]))


def _observable_pairs(h, obs, legs):
    if obs == 'C0':
        return [legs]
    if obs == 'trace_cov':
        sel = list(h.legs)
    else:
        sel = [h.legs[h.index(f)] for f in obs]
    return [[(f, i if i is not None else 1), (f, i if i is not None else 1)]
            for f, i in sel]


def _pms_scan(model_ct, values, h, legs, N, params, inner, opts, klo, khi,
              log=None):
    """Equal-time loop orders at Chebyshev nodes in ``log s``, the screened
    curve ``C_N(κ)`` from the interpolant, its stationary points and the
    chosen κ (with the fallbacks of the module docstring)."""
    pairs = _observable_pairs(h, opts['observable'], legs)
    n = opts['n_scan']
    ulo, uhi = math.log(klo), math.log(khi)
    t = -np.cos(np.pi * np.arange(n) / (n - 1))
    s_nodes = np.exp(0.5 * (uhi + ulo) + 0.5 * (uhi - ulo) * t)
    F = np.zeros((N + 1, n))
    for k, s in enumerate(s_nodes):
        vals, _sp = _run_pairs(model_ct, values, s, pairs, N, params,
                               np.array([0.0]), inner, log=log)
        for v in vals:
            for e in range(N + 1):
                F[e, k] += float(np.real(np.asarray(v.get(e, 0.0)).ravel()[0]))
    inter = [_LogCheb(s_nodes, F[e], ulo, uhi) for e in range(N + 1)]
    sub = slice(None, None, 2)          # the nested Lobatto subset
    inter_h = [_LogCheb(s_nodes[sub], F[e][sub], ulo, uhi)
               for e in range(N + 1)]

    def dfn(ip):
        return lambda e, m, k: ip[e].deriv(k, m)
    ks = np.geomspace(klo, khi, 241)
    curve = {'kappa': ks}
    stat = {}
    for M in range(1, N + 1):
        der = {e: [inter[e].deriv(ks, m) for m in range(M - e + 1)]
               for e in range(M + 1)}
        curve[M] = _opt_value(der, ks, M)
        stat[M] = _stationary_points(dfn(inter), M, klo, khi)

    def nearest(roots):
        return min(roots, key=lambda k: abs(math.log(k))) if roots else None
    kappa, source = None, None
    for M in range(N, 0, -1):
        k = nearest(stat[M])
        if k is not None:
            kappa = k
            source = (f'pms order {N}' if M == N else
                      f'pms order {M} (order {N} has no stationary point)')
            break
    if kappa is None:
        kappa, source = 1.0, 'hartree (no order has a PMS stationary point)'
        M = N
    k_half = nearest(_stationary_points(dfn(inter_h), M, klo, khi))
    der = {e: [inter[e].deriv(np.array([kappa]), m)[0]
               for m in range(N - e + 1)] for e in range(N + 1)}
    der_h = {e: [inter_h[e].deriv(np.array([kappa]), m)[0]
                 for m in range(N - e + 1)] for e in range(N + 1)}
    value = float(_opt_value(der, kappa, N))
    value_h = float(_opt_value(der_h, kappa, N))
    return {'kappa': float(kappa), 'source': source, 'curve': curve,
            # loop orders that change sign over the scan are interpolated
            # directly (less accurate in the high derivatives than log |F|)
            'linear_orders': [e for e in range(N + 1) if not inter[e].sign],
            'stationary': stat, 'value': value,
            'interp_change': {
                'kappa': (None if k_half is None or source.startswith('hart')
                          else abs(k_half - kappa)),
                'value': abs(value - value_h) / max(abs(value), 1e-300)},
            'n_runs': n * len(pairs), 'nodes': s_nodes, 'F': F,
            'pairs': pairs}


def _combine_spectra(spec_pts, W, kappa, N):
    """The screened spectral values by order from the stencil runs'
    ``spectra`` dicts (the values linear in ``C``; ``tau_c`` re-derived)."""
    first = spec_pts[0]
    out = {'omega': first.get('omega'), 'T': first.get('T'),
           'conventions': first.get('conventions'), 'method': 'screened',
           'notes': list(first.get('notes') or [])}
    for key in _LINEAR_SPECTRA:
        bk = key + '_by_ell'
        if not first.get(bk):
            continue
        pts = [{int(e): np.asarray(v) for e, v in sp[bk].items()}
               for sp in spec_pts]
        derivs = {e: [sum(W[n, j] * pts[j][e] for j in range(len(pts)))
                      for n in range(N - e + 1)] for e in range(N + 1)}
        by = {M: _opt_value(derivs, kappa, M) for M in range(N + 1)}
        out[key + '_by_order'] = by
        out[key] = by[N]
    if 'integral_inf_by_order' in out and 'C0_by_order' in out:
        out['tau_c_by_order'] = {
            M: (complex(out['integral_inf_by_order'][M])
                / complex(out['C0_by_order'][M]))
            for M in out['C0_by_order']}
        out['tau_c'] = out['tau_c_by_order'][N]
    return out
