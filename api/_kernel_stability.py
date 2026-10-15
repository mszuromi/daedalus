r"""
Kernel-aware linear stability (the analyzer's G2-K gate)
========================================================

The mean-field linearization (G2) replaces every kernel by its integral
``g -> g_hat(0)``, so an oscillatory instability that a kernel (a delay, or a
rational kernel) creates is invisible to it.  This module decides stability
from the characteristic function

    chi(s) = det K(s),        K(s) = K_ft(omega = -i s)  at the saddle,

with the TRUE kernel images (``exp(-s d)`` for a delay): the saddle is
stable iff ``chi`` has no zero with ``Re s >= 0``.

* **Scalar, one delay** (``K(s) = alpha s + beta + gamma e^{-s X}``): the
  roots are ``s_k = W_k(z)/X - beta/alpha``, ``z = -(gamma/alpha) X
  e^{beta X/alpha}`` (Lambert W, branches ``k = -60..60``), exact.
* **General**: the argument principle on the boundary of the half disc
  ``{Re s >= 0, |s| <= R}``.  ``R`` is chosen so that ``|chi(s)/(c s^m) - 1|
  <= 1/2`` on the arc and on the imaginary axis beyond ``R`` (``c s^m``: the
  leading term of ``chi`` at large real ``s``); for a retarded system that
  ratio is bounded in the right half plane, so (Phragmen-Lindelof) ``chi``
  has no zero with ``Re s >= 0`` outside the disc and the winding number
  counts the unstable roots exactly.  The contour is refined adaptively
  until no step turns the phase by more than 0.3 rad.  A zero on (or within
  ``1e-9`` relative of) the axis is reported as marginal.  When no ``R``
  qualifies (a neutral system: a delay acting on a derivative), the status
  is ``'neutral'``.  The unstable roots are located (up to 6) by bisection
  of boxes with their own winding numbers and polished by the secant method.
* **Acausal kernel images** (a pole of an image with ``Re s > 0``) make
  ``chi`` meromorphic in the right half plane: status ``'acausal'`` (the G3
  gate names it), no count.

``kernel_stability(K_ft, omega, num_params)`` returns a ``KernelStability``.
"""
from __future__ import annotations

import cmath
import math
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

__all__ = ['KernelStability', 'kernel_stability', 'numeric_K',
           'lambert_roots', 'count_rhp_roots']

_PHASE_STEP = 0.3          # max phase turn per contour step (rad)
_MARGINAL_REL = 1e-9
_MAX_PTS = 400_000


@dataclass
class KernelStability:
    status: str                    # stable | unstable | marginal | neutral | acausal | undetermined
    n_rhp: Optional[int] = None    # zeros of det K with Re s > 0
    roots: tuple = ()              # located right-half-plane (or marginal) roots
    rightmost: Optional[complex] = None
    method: str = ''               # 'lambert_w' | 'argument_principle'
    radius: Optional[float] = None
    n_points: int = 0
    detail: str = ''
    wall_s: float = 0.0
    delays: tuple = ()

    def as_dict(self):
        return {'status': self.status, 'n_rhp': self.n_rhp,
                'roots': [complex(r) for r in self.roots],
                'rightmost': None if self.rightmost is None else complex(self.rightmost),
                'method': self.method, 'radius': self.radius,
                'n_points': self.n_points, 'detail': self.detail,
                'wall_s': round(self.wall_s, 4), 'delays': list(self.delays)}


# ── numeric evaluation of K(omega) ───────────────────────────────────────


def _np_compile(e, omega):
    """An SR expression in ``omega`` -> numpy function of a complex array."""
    from sage.all import SR, exp, sin, cos, sinh, cosh, tanh, log, sqrt
    e = SR(e)
    if not e.has(omega):
        v = complex(e)
        return lambda w, _v=v: np.full(np.shape(w), _v, dtype=complex)
    if e.is_symbol():
        return lambda w: np.asarray(w, dtype=complex)
    op = e.operator()
    fs = [_np_compile(o, omega) for o in e.operands()]
    name = getattr(op, '__name__', str(op))
    if name in ('add_vararg', 'add'):
        return lambda w, _f=fs: sum(f(w) for f in _f)
    if name in ('mul_vararg', 'mul'):
        def mul(w, _f=fs):
            out = _f[0](w)
            for f in _f[1:]:
                out = out * f(w)
            return out
        return mul
    if name == 'pow':
        base, ex = e.operands()
        if not SR(ex).has(omega):
            exv = complex(ex)
            if exv.imag == 0 and float(exv.real).is_integer():
                k = int(exv.real)
                return lambda w, _b=fs[0], _k=k: _b(w) ** _k
            return lambda w, _b=fs[0], _e=exv: _b(w) ** _e
        return lambda w, _b=fs[0], _e=fs[1]: _b(w) ** _e(w)
    table = {exp: np.exp, sin: np.sin, cos: np.cos, sinh: np.sinh, cosh: np.cosh,
             tanh: np.tanh, log: np.log, sqrt: np.sqrt}
    for sop, nf in table.items():
        if op == sop:
            return lambda w, _f=fs[0], _n=nf: _n(_f(w))
    raise ValueError(f'cannot evaluate {name} numerically in K(omega)')


def numeric_K(K_ft, omega, num_params):
    """``s-array -> (n, nf, nf)`` complex ``K(omega = -i s)`` (saddle and
    parameters substituted)."""
    from sage.all import SR
    nf = K_ft.nrows()
    fns = []
    for i in range(nf):
        row = []
        for j in range(nf):
            e = SR(K_ft[i, j])
            if e.is_trivial_zero():
                row.append(None)
                continue
            e = e.subs(num_params)
            free = set(map(str, e.variables())) - {str(omega)}
            if free:
                raise ValueError(f'K[{i},{j}] keeps symbols {sorted(free)}')
            row.append(_np_compile(e, omega))
        fns.append(row)

    def K_of_s(s):
        s = np.atleast_1d(np.asarray(s, dtype=complex))
        w = -1j * s
        M = np.zeros((s.size, nf, nf), dtype=complex)
        for i in range(nf):
            for j in range(nf):
                f = fns[i][j]
                if f is not None:
                    M[:, i, j] = f(w)
        return M

    return K_of_s, nf


# ── Lambert W (scalar, one delay) ────────────────────────────────────────


def lambert_roots(alpha, beta, gamma, X, branches=None):
    """Roots of ``alpha s + beta + gamma e^{-s X} = 0`` on the Lambert branches
    ``-K..K``.  ``K`` (default) covers every root with ``|Im s| <= R``,
    ``R = |beta/alpha| + |gamma/alpha| + 1`` (the bound on right-half-plane
    roots; ``Im W_k ~ 2 pi k``), at least 60.  ``z = -(gamma/alpha) X
    e^{beta X/alpha}`` is formed in mpmath (no overflow for large ``beta X``)."""
    import mpmath as mp
    alpha, beta, gamma, X = complex(alpha), complex(beta), complex(gamma), float(X)
    if gamma == 0:
        return [(-beta / alpha)]
    R = abs(beta / alpha) + abs(gamma / alpha) + 1.0
    K = int(branches) if branches is not None else max(60, int(R * X / (2 * math.pi)) + 10)
    with mp.workdps(30):
        z = -(mp.mpc(gamma) / mp.mpc(alpha)) * X * mp.exp(mp.mpc(beta) * X / mp.mpc(alpha))
        out = []
        for k in range(-K, K + 1):
            out.append(complex(mp.lambertw(z, k) / X - mp.mpc(beta) / mp.mpc(alpha)))
    return out


# ── the argument principle ──────────────────────────────────────────────


def _refined_phase(chi, path, t0, t1, n0, budget):
    """Total phase change of ``chi(path(t))`` for ``t`` from ``t0`` to ``t1``,
    refined until every step turns by at most ``_PHASE_STEP``; also the
    smallest ``|chi|`` seen (with its point) and the number of points."""
    t = np.linspace(t0, t1, n0)
    v = chi(path(t))
    n_eval = t.size
    for _ in range(40):
        if not np.all(np.isfinite(v)):
            bad = ~np.isfinite(v)
            raise FloatingPointError(f'chi not finite at {path(t[bad][:3])}')
        if np.any(v == 0):
            i = int(np.argmin(np.abs(v)))
            return float('nan'), 0.0, complex(path(np.array([t[i]]))[0]), n_eval
        dphi = np.angle(v[1:] / v[:-1])
        big = np.abs(dphi) > _PHASE_STEP
        if not big.any():
            break
        if n_eval > budget:
            raise RuntimeError('contour refinement budget exceeded')
        mids = 0.5 * (t[:-1][big] + t[1:][big])
        vm = chi(path(mids))
        n_eval += mids.size
        t = np.concatenate([t, mids])
        v = np.concatenate([v, vm])
        order = np.argsort(t, kind='stable')
        if t1 < t0:
            order = order[::-1]
        t, v = t[order], v[order]
    dphi = np.angle(v[1:] / v[:-1])
    i = int(np.argmin(np.abs(v)))
    return float(dphi.sum()), float(np.abs(v[i])), complex(path(np.array([t[i]]))[0]), n_eval


def _leading(chi):
    """``(c, m)`` with ``chi(s) ~ c s^m`` at large real ``s``: the first scale
    ``x0`` in 1e3 .. 1e13 where three successive doublings give the same
    integer power (the rates of the model set where that regime starts)."""
    for x0 in 10.0 ** np.arange(3, 14):
        xs = x0 * np.array([1.0, 2.0, 4.0, 8.0])
        with np.errstate(all='ignore'):
            v = chi(xs)
        if not np.all(np.isfinite(v)) or np.any(v == 0):
            continue
        ms = np.log(np.abs(v[1:] / v[:-1])) / math.log(2.0)
        m = int(round(float(ms[-1])))
        if np.max(np.abs(ms - m)) > 0.02:
            continue
        return complex(v[-1]) / xs[-1] ** m, m
    return None


def _ratio_ok(chi, c, m, R):
    th = np.linspace(-math.pi / 2, math.pi / 2, 2001)
    s_arc = R * np.exp(1j * th)
    y = np.geomspace(R, 1e3 * R, 1500)
    s_ax = np.concatenate([1j * y, -1j * y])
    s = np.concatenate([s_arc, s_ax])
    v = chi(s)
    if not np.all(np.isfinite(v)):
        return False
    lead = c * s ** m
    return bool(np.max(np.abs(v / lead - 1.0)) <= 0.5)


def _winding_box(chi, x0, x1, y0, y1, budget):
    """Winding number of ``chi`` around the rectangle (counterclockwise)."""
    total, n = 0.0, 0
    segs = [(lambda t: t + 1j * y0, x0, x1), (lambda t: x1 + 1j * t, y0, y1),
            (lambda t: t + 1j * y1, x1, x0), (lambda t: x0 + 1j * t, y1, y0)]
    vmin = np.inf
    for path, a, b in segs:
        ph, vm, _, ne = _refined_phase(chi, path, a, b, 64, budget)
        if not math.isfinite(ph):
            return 0, 0.0, n + ne
        total += ph
        n += ne
        vmin = min(vmin, vm)
    return int(round(total / (2 * math.pi))), vmin, n


def _polish(chi, s0, h):
    """Secant iterations on ``chi`` from ``s0``."""
    a, b = complex(s0), complex(s0) + h
    fa, fb = complex(chi(np.array([a]))[0]), complex(chi(np.array([b]))[0])
    for _ in range(60):
        if fb == fa:
            break
        c = b - fb * (b - a) / (fb - fa)
        a, fa = b, fb
        b, fb = c, complex(chi(np.array([c]))[0])
        if abs(b - a) <= 1e-14 * max(1.0, abs(b)):
            break
    return b


def _locate(chi, R, n_expect, budget, max_roots=6):
    """Up to ``max_roots`` roots in ``[0, R] x [-R, R]`` by box bisection."""
    roots = []
    stack = [(0.0, R, -R, R, n_expect)]
    used = 0
    while stack and len(roots) < max_roots and used < budget:
        x0, x1, y0, y1, cnt = stack.pop()
        if cnt <= 0:
            continue
        w = max(x1 - x0, y1 - y0)
        if cnt == 1 and w < 1e-2 * max(R, 1.0):
            roots.append(_polish(chi, complex(0.5 * (x0 + x1), 0.5 * (y0 + y1)), 1e-3 * w + 1e-9))
            continue
        if w < 1e-9 * max(R, 1.0):
            roots.append(complex(0.5 * (x0 + x1), 0.5 * (y0 + y1)))
            continue
        # off-centre splits: a real root never sits on a box edge
        xm, ym = x0 + 0.4973 * (x1 - x0), y0 + 0.5127 * (y1 - y0)
        for bx in ((x0, xm, y0, ym), (xm, x1, y0, ym), (x0, xm, ym, y1), (xm, x1, ym, y1)):
            try:
                c, vmin, ne = _winding_box(chi, *bx, budget)
            except (RuntimeError, FloatingPointError, ValueError):
                continue
            if vmin == 0.0:
                continue
            used += ne
            if c > 0:
                stack.append(bx + (c,))
    return sorted(roots, key=lambda z: -z.real)


def count_rhp_roots(chi, *, budget=_MAX_PTS, d_max=0.0):
    """``KernelStability`` of ``chi`` by the argument principle (see the
    module docstring).  ``d_max``: the longest delay; the initial contour grid
    resolves ``e^{-s d_max}`` (a phase step of 0.3 rad), so the refinement
    cannot alias over whole periods."""
    t0 = time.perf_counter()
    lead = _leading(chi)
    if lead is None:
        return KernelStability('undetermined', method='argument_principle',
                               detail='det K(s) has no power-law leading term '
                                      'at large real s', wall_s=time.perf_counter() - t0)
    c, m = lead
    R = 1.0
    while not _ratio_ok(chi, c, m, R):
        R *= 2.0
        if R > 1e7:
            return KernelStability(
                'neutral', method='argument_principle',
                detail='det K(s)/(c s^m) does not approach 1 along the imaginary '
                       'axis (a delay acting on a derivative: a neutral system); '
                       'the unstable roots cannot be bounded',
                wall_s=time.perf_counter() - t0)
    R = 1.5 * R
    n_ax = max(4001, int(2 * R * max(float(d_max), 0.0) / _PHASE_STEP) + 1)
    n_arc = max(2001, int(math.pi * R * max(float(d_max), 0.0) / _PHASE_STEP) + 1)
    if n_ax + n_arc > budget:
        return KernelStability(
            'undetermined', method='argument_principle', radius=R,
            detail=f'the contour needs {n_ax + n_arc} points to resolve the delay (R d = '
                   f'{R * d_max:.3g}): beyond the budget {budget}', wall_s=time.perf_counter() - t0)
    ph_ax, vmin_ax, s_min, n1 = _refined_phase(chi, lambda t: 1j * t, R, -R, n_ax, budget)
    ph_arc, _, _, n2 = _refined_phase(chi, lambda t: R * np.exp(1j * t),
                                      -math.pi / 2, math.pi / 2, n_arc, budget)
    scale = max(abs(complex(chi(np.array([0.0]))[0])), abs(c) * max(1.0, abs(s_min)) ** m)
    w = (ph_ax + ph_arc) / (2 * math.pi)
    if not math.isfinite(w):
        out = KernelStability('marginal', method='argument_principle', radius=R,
                              roots=(s_min,), rightmost=s_min, n_points=n1 + n2,
                              detail=f'det K(s) = 0 on the imaginary axis at s = {s_min:.6g}')
        out.wall_s = time.perf_counter() - t0
        return out
    nz = int(round(w))
    out = KernelStability('stable', n_rhp=nz, method='argument_principle', radius=R,
                          n_points=n1 + n2, detail=f'winding {w:.6f} on the half disc of '
                          f'radius {R:.4g} (leading term c s^{m})')
    if abs(w - nz) > 0.05 or nz < 0:
        out.status = 'undetermined'
        out.detail += ' (not an integer: the contour passes too close to a root)'
    elif nz > 0:
        out.status = 'unstable'
        try:
            roots = _locate(chi, R, nz, budget)
        except (RuntimeError, FloatingPointError, ValueError):
            roots = []
        out.roots = tuple(roots)
        out.rightmost = roots[0] if roots else None
        if vmin_ax <= _MARGINAL_REL * scale:
            out.detail += f'; also |det K| = {vmin_ax:.3g} on the axis at s = {s_min:.6g}'
    elif vmin_ax <= _MARGINAL_REL * scale:
        out.status = 'marginal'
        out.roots = (s_min,)
        out.rightmost = s_min
        out.detail += f'; |det K| = {vmin_ax:.3g} on the axis at s = {s_min:.6g}'
    out.wall_s = time.perf_counter() - t0
    return out


# ── the gate's entry point ───────────────────────────────────────────────


def _split_delay(e, omega):
    """``(K0, K1, X)`` with ``e = K0 + K1 exp(-I omega X)`` (one delay), or None."""
    from sage.all import SR
    from api.delay import delay_factors, _exp_delay
    ds = delay_factors(e, omega)
    if len(ds) != 1:
        return None
    X = ds[0]
    E = SR.var('_delayE_')

    def rebuild(x):
        x = SR(x)
        hit = _exp_delay(x, omega)
        if hit is not None:
            from sage.all import exp
            return exp(hit[0]) * E
        ops = x.operands()
        return x if not ops else x.operator()(*[rebuild(o) for o in ops])

    ee = SR(rebuild(e)).expand()
    K1 = SR(ee.coefficient(E, 1))
    K0 = SR(ee.coefficient(E, 0))
    if not SR(ee - K0 - K1 * E).expand().is_zero():
        return None
    return K0, K1, X


def _acausal_images(K_ft, omega, num_params):
    """Right-half-plane poles (in s) of the entries' denominators."""
    from sage.all import SR
    from api.delay import _exp_delay
    bad = []
    for e in K_ft.list():
        e = SR(e)
        if e.is_trivial_zero():
            continue
        e = e.subs(num_params)

        def strip(x):
            x = SR(x)
            if _exp_delay(x, omega) is not None:
                return SR(1)
            ops = x.operands()
            return x if not ops else x.operator()(*[strip(o) for o in ops])

        den = SR(strip(e)).rational_simplify().denominator()
        if not den.has(omega):
            continue
        try:
            co = [complex(c) for c in SR(den).expand().coefficients(omega, sparse=False)]
        except Exception:
            continue
        r = np.roots(co[::-1]) if len(co) > 1 else []
        for w in r:
            s = 1j * w
            if s.real > 1e-9 * max(1.0, abs(s)):
                bad.append(s)
    return bad


def kernel_stability(K_ft, omega, num_params) -> KernelStability:
    """G2-K: stability of the saddle with the true kernel images."""
    from sage.all import SR
    from api.delay import delay_factors
    t0 = time.perf_counter()
    delays = []
    for e in K_ft.list():
        for X in delay_factors(SR(e), omega):
            Xv = SR(X).subs(num_params)
            try:
                xv = float(Xv)
            except TypeError:
                return KernelStability('undetermined', detail=f'delay {X} has no value')
            if not any(abs(xv - y) <= 1e-14 * max(1.0, abs(y)) for y in delays):
                delays.append(xv)
    delays = [x for x in delays if x != 0.0]             # exp(0) = 1: no delay
    if any(x < 0 for x in delays):
        return KernelStability('acausal', delays=tuple(delays),
                               detail=f'negative delay(s) {[x for x in delays if x < 0]} '
                                      '(an advance, not a delay)',
                               wall_s=time.perf_counter() - t0)
    acausal = _acausal_images(K_ft, omega, num_params)
    if acausal:
        return KernelStability('acausal', delays=tuple(delays),
                               detail=f'kernel image pole(s) in the right half plane at s = '
                                      f'{[complex(a) for a in acausal[:3]]}',
                               wall_s=time.perf_counter() - t0)
    # Lambert W: one field, one delay, K = alpha s + beta + gamma e^{-s X}
    if K_ft.nrows() == 1 and len(delays) == 1:
        sp = _split_delay(SR(K_ft[0, 0]).subs(num_params), omega)
        if sp is not None:
            K0, K1, X = sp
            try:
                if (not K1.has(omega) and SR(K0.diff(omega, 2)).is_zero()):
                    alpha = complex(K0.diff(omega)) / 1j       # K0 = alpha (i omega) + beta
                    beta = complex(K0.subs({omega: 0}))
                    gamma = complex(K1)
                    Xv = float(SR(X).subs(num_params))
                    rts = lambert_roots(alpha, beta, gamma, Xv)
                    rhp = sorted([z for z in rts if z.real > 0], key=lambda z: -z.real)
                    marg = [z for z in rts if abs(z.real) <= _MARGINAL_REL * max(1.0, abs(z))]
                    right = max(rts, key=lambda z: z.real)
                    st = ('marginal' if marg else 'unstable' if rhp else 'stable')
                    return KernelStability(
                        st, n_rhp=len(rhp), roots=tuple(rhp or marg), rightmost=right,
                        method='lambert_w', delays=(Xv,),
                        detail=f'Lambert W roots s_k = W_k(z)/d - beta/alpha, '
                               f'branches -60..60; rightmost {right:.8g}',
                        wall_s=time.perf_counter() - t0)
            except (TypeError, ValueError, ZeroDivisionError, OverflowError, ArithmeticError):
                pass                                # the argument principle below
    K_of_s, nf = numeric_K(K_ft, omega, num_params)

    def chi(s):
        return np.linalg.det(K_of_s(s))

    try:
        out = count_rhp_roots(chi, d_max=max(delays) if delays else 0.0)
    except (RuntimeError, FloatingPointError, np.linalg.LinAlgError, ValueError,
            OverflowError) as exc:
        out = KernelStability('undetermined', method='argument_principle',
                              detail=f'{type(exc).__name__}: {exc}')
    out.delays = tuple(delays)
    out.wall_s = time.perf_counter() - t0
    return out
