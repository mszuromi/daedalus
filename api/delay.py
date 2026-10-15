r"""
Delay kernels: recognition, Pade (n, n) substitution with an automatic order,
and the driver that ``compute_cumulants`` hands a delayed model to
=============================================================================

A delay kernel is a kernel whose frequency image carries a factor
``exp(-I*omega*d)`` (``d > 0``): the hard delay ``g(t) = delta(t - d)``, a
delayed filter ``exp(-I*omega*d)/(1 + I*omega*tau)``, ...  Such an image is
not rational in ``omega``, so the pole/residue stage and the state-space
realization cannot take it as it is.  This module

* recognises the factors (``delay_factors``) in the model's kernel images
  (``model['kernel_ft_image']``, the hook every front end fills: the builder's
  ``define_kernel(freq_image=...)`` / ``time_expr='dirac_delta(t - d)'`` /
  ``delay='d'``, the SDE front end's ``x(t - d)`` and ``Kernel.delay(d)``);
* replaces each by the diagonal Pade approximant of order ``(n, n)``,

      exp(-z) ~ P_n(-z) / P_n(z),   P_n(z) = sum_k (2n-k)! n! / ((2n)! k! (n-k)!) z^k,
      z = I*omega*d,

  an all-pass function (``|P_n(-i x)/P_n(i x)| = 1``) with exact rational
  coefficients, whose poles lie in the left half plane for every ``n``
  (``pade_substitute``).  The approximated model is an ordinary rational
  kernel model: the state-space realization (``api._realization``) turns each
  Pade denominator into one Schwarz block and balances it, and the
  state-space Phase J route integrates it (its cost grows with the number of
  realization states, not with the number of pole tuples);
* runs ``compute_cumulants`` on the approximated model at increasing orders
  ``n = n_start, 2 n_start, ...`` (``compute_delayed``) until the estimated
  relative error of the observable, ``max_tau |C_n - C_2n| / max_tau |C_2n|``
  (conservative: item 29 measured it 3-5x above the true error of ``C_n``,
  and the reported value is ``C_2n``), meets the target (default ``1e-8``),
  or until ``n_max`` or the predicted wall time of the next order exceeds the
  cap (then a warning says the target was missed);
* takes the TREE (``k = 2``) from the exact covariance of the linear delayed
  system (``api._delay_exact``: a boundary-value problem on ``[0, d]`` plus
  the method of steps, exact up to rounding) instead of its Pade
  approximation, whose error decays only like ``n^-2`` at short lags.  Loops
  keep the Pade approximation (their error decays like ``n^-3`` and they are
  smaller).  ``delay={'tree': 'pade'}`` (or ``DAEDALUS_DELAY_EXACT_TREE=0``)
  switches the exact tree off.

Stability is decided with the TRUE exponential (``api._kernel_stability``,
the analyzer's G2-K gate), never from the approximant: an unstable delayed
saddle is refused by name before any approximation runs.

Options (``compute_cumulants(..., delay={...})``):

``order``    ``'auto'`` (default) or a fixed ``n``;
``tol``      target relative accuracy of the observable (default ``1e-8``);
``n_start``  first order of the doubling search (default 4);
``n_max``    largest order (default 16: one loop beyond it leaves the state-space
             route's fast range on c976a25, e.g. the scalar delayed OU takes 6 s at
             n = 16 and more than 30 min at n = 20);
``wall_max`` cap (s) on the predicted wall time of the next order (default 900);
``tree``     ``'exact'`` (default) or ``'pade'``.

The result carries ``result['delay']`` (orders tried, walls, the estimate,
whether the target was met, the tree used and its comparison with the Pade
tree), the analyzer finding ``G3_DELAY_APPROXIMATED`` and an
``approximation`` entry of the assumptions ledger (``delay:<kernel>``).
"""
from __future__ import annotations

import math
import os
import time
import warnings
from dataclasses import replace
from fractions import Fraction
from typing import Optional

import numpy as np

__all__ = ['Kernel', 'pade_coefficients', 'pade_image', 'delay_factors',
           'pade_substitute', 'kernel_delays', 'pade_model',
           'split_shifted_time_expr', 'compute_delayed', 'DelayOrderWarning',
           'DEFAULTS']

DEFAULTS = dict(order='auto', tol=1e-8, n_start=4, n_max=16, wall_max=900.0,
                tree='exact')


class DelayOrderWarning(UserWarning):
    """The automatic Pade order missed its accuracy target (cap or n_max)."""


# ── Pade approximants of exp(-z) ─────────────────────────────────────────


def pade_coefficients(n: int) -> list:
    """Exact coefficients ``c_0..c_n`` of ``P_n(z)``: ``exp(-z) ~ P_n(-z)/P_n(z)``."""
    n = int(n)
    if n < 1:
        raise ValueError(f'Pade order must be >= 1, got {n}')
    return [Fraction(math.factorial(2 * n - k) * math.factorial(n),
                     math.factorial(2 * n) * math.factorial(k) * math.factorial(n - k))
            for k in range(n + 1)]


def pade_image(z, n: int):
    """``P_n(-z)/P_n(z)`` as a Sage expression in ``z`` (exact coefficients)."""
    from sage.all import SR, QQ
    c = [QQ(x) for x in pade_coefficients(n)]
    z = SR(z)
    num = sum(SR(ck) * (-z) ** k for k, ck in enumerate(c))
    den = sum(SR(ck) * z ** k for k, ck in enumerate(c))
    return num / den


def pade_np(n: int, s, d):
    """numpy ``P_n(-s d)/P_n(s d)`` (``s`` the Laplace variable)."""
    c = [float(x) for x in pade_coefficients(n)]
    z = np.asarray(s) * d
    return (sum(ck * (-z) ** k for k, ck in enumerate(c))
            / sum(ck * z ** k for k, ck in enumerate(c)))


# ── recognition ──────────────────────────────────────────────────────────


def _exp_delay(e, omega):
    """``(alpha, X)`` when ``e = exp(alpha - I*omega*X)`` with ``X != 0`` free of
    ``omega`` (``alpha`` free of omega too), else ``None``."""
    from sage.all import SR, exp, I
    if e.operator() != exp:
        return None
    arg = SR(e.operands()[0])
    if not arg.has(omega):
        return None
    try:
        if not SR(arg.diff(omega, 2)).is_trivial_zero() and not SR(
                arg.diff(omega, 2)).simplify_full().is_zero():
            return None
        slope = SR(arg.diff(omega))
        if slope.has(omega):
            return None
        X = SR(slope * I).expand()
        alpha = SR(arg.subs({omega: 0})).expand()
    except Exception:
        return None
    if X.is_zero() or X.has(I):
        return None
    return alpha, X


def delay_factors(expr, omega) -> list:
    """The distinct delays ``X`` of the factors ``exp(alpha - I*omega*X)`` in
    ``expr`` (Sage expressions), in order of first appearance."""
    from sage.all import SR
    out = []

    def walk(e):
        e = SR(e)
        hit = _exp_delay(e, omega)
        if hit is not None:
            if not any(SR(hit[1] - x).is_zero() for x in out):
                out.append(hit[1])
            return
        for o in e.operands():
            walk(o)

    walk(expr)
    return out


def pade_substitute(expr, omega, n: int):
    """``expr`` with every ``exp(alpha - I*omega*X)`` replaced by
    ``exp(alpha) * P_n(-z)/P_n(z)``, ``z = I*omega*X``."""
    from sage.all import SR, I, exp

    def rebuild(e):
        e = SR(e)
        hit = _exp_delay(e, omega)
        if hit is not None:
            alpha, X = hit
            img = pade_image(I * omega * X, n)
            return img if alpha.is_zero() else exp(alpha) * img
        ops = e.operands()
        if not ops:
            return e
        return e.operator()(*[rebuild(o) for o in ops])

    return rebuild(expr)


def kernel_delays(model: dict, ns) -> dict:
    """``{kernel symbol name: [delays]}`` for the kernel images of ``model``
    that carry a delay factor (``ns``: the expanded theory's namespace, the
    one the propagator builder passes to the hook)."""
    from sage.all import SR
    hook = model.get('kernel_ft_image')
    if hook is None:
        return {}
    omega = SR.var('omega')
    try:
        imgs = hook(ns, omega) or {}
    except Exception:
        return {}
    out = {}
    for key, val in imgs.items():
        ds = delay_factors(SR(val), omega)
        if ds:
            out[str(key)] = ds
    return out


def pade_model(model: dict, n: int) -> dict:
    """A copy of ``model`` whose kernel images carry the Pade (n, n)
    approximant of every delay factor (its own name and fingerprint, so its
    caches never mix with the true model's or another order's).  The copy
    asks the propagator builder to skip the symbolic inverse (the numbers are
    substituted before inverting; the symbolic inverse of a Pade-n kernel
    matrix is the slow part and nothing downstream needs it)."""
    from sage.all import SR
    hook = model['kernel_ft_image']

    def hook_n(ns, omega, _hook=hook, _n=int(n)):
        out = _hook(ns, omega) or {}
        return {k: pade_substitute(SR(v), omega, _n) for k, v in out.items()}

    m = dict(model)
    m['kernel_ft_image'] = hook_n
    base = model.get('name', 'model')
    m['name'] = f'{base} [delay Pade {int(n)}]'
    m['spec_signature'] = f"{model.get('spec_signature', '')}-pade{int(n)}"
    m['_delay_pade'] = {'n': int(n), 'model': base}
    m['_numeric_first_propagator'] = True
    return m


# ── front end: delayed kernels from text ─────────────────────────────────


def split_shifted_time_expr(text: str):
    """``(delay_text, unshifted_time_expr)`` when ``text`` is a time-domain
    kernel shifted by a delay: it holds ``dirac_delta(t - X)`` or
    ``heaviside(t - X)`` factors with one ``X != 0`` free of ``t`` (then
    ``t -> t + X`` gives the unshifted kernel).  ``None`` otherwise."""
    import re
    # text pre-check: no Sage symbol is created for an ordinary kernel (a new
    # symbol shifts pynac's term order, and with it the rounding of later sums)
    if not re.search(r'\b(dirac_delta|heaviside|unit_step)\s*\(\s*t\s*[-+]', str(text)):
        return None
    from sage.all import SR, var, heaviside, dirac_delta
    names = set(re.findall(r'\b([A-Za-z_]\w*)\b', text)) - {
        'dirac_delta', 'heaviside', 'exp', 'log', 'sqrt', 'sin', 'cos', 'I',
        'pi', 'e', 'unit_step'}
    loc = {nm: var(nm) for nm in names}
    loc.update(dirac_delta=dirac_delta, heaviside=heaviside, unit_step=heaviside)
    try:
        e = SR(eval(text, {'__builtins__': {}}, dict(loc, **{
            'exp': __import__('sage.all', fromlist=['exp']).exp,
            'sqrt': __import__('sage.all', fromlist=['sqrt']).sqrt,
            'I': __import__('sage.all', fromlist=['I']).I,
            'pi': __import__('sage.all', fromlist=['pi']).pi})))
    except Exception:
        return None
    t = loc.get('t')
    if t is None:
        return None
    shifts = []

    def walk(x):
        op = x.operator()
        if op in (dirac_delta, heaviside) and x.operands():
            a = SR(x.operands()[0])
            if a.has(t):
                X = SR(-a.subs({t: 0}))
                # every step / delta factor counts, the unshifted ones too: a
                # kernel mixing an instantaneous and a delayed part is not ONE
                # delayed kernel (left to the compiler, which names it)
                shifts.append(X if SR(a.diff(t)).is_one() and not X.has(t) else None)
        for o in x.operands():
            walk(o)

    walk(e)
    if not shifts or any(x is None for x in shifts):
        return None
    X = shifts[0]
    if X.is_zero() or any(not SR(X - y).is_zero() for y in shifts):
        return None
    un = SR(e.subs({t: t + X})).expand()
    return str(X), str(un)


class Kernel:
    """Kernel declarations for ``define_kernel(name, kernel=...)``.

    ``Kernel.delay(d)``: the hard delay ``delta(t - d)`` (image ``exp(-I omega d)``);
    ``Kernel.exponential(tau)``: ``exp(-t/tau)/tau`` (image ``1/(1 + I omega tau)``);
    ``Kernel.gamma(n, mean)``: the Erlang density of order ``n`` and mean
    ``mean`` (image ``(1 + I omega mean/n)^-n``; rational: an exact gamma
    delay, no approximation);
    ``Kernel.image(text)``: any frequency image.
    Products compose (convolution): ``Kernel.delay('d') * Kernel.exponential('tau')``.
    Arguments are parameter names or expressions (text)."""

    def __init__(self, freq_image: str = '1', delay: Optional[str] = None):
        self.freq_image = str(freq_image)
        self.delay = None if delay is None else str(delay)

    @classmethod
    def delay(cls, d):
        return cls('1', delay=str(d))

    @classmethod
    def exponential(cls, tau):
        return cls(f'1/(1 + I*omega*({tau}))')

    @classmethod
    def gamma(cls, n, mean):
        n = int(n)
        if n < 1:
            raise ValueError('Kernel.gamma: n must be >= 1')
        return cls(f'(1 + I*omega*({mean})/{n})^(-{n})')

    @classmethod
    def image(cls, text):
        return cls(str(text))

    def __mul__(self, other):
        if not isinstance(other, Kernel):
            return NotImplemented
        if self.delay and other.delay:
            d = f'({self.delay}) + ({other.delay})'
        else:
            d = self.delay or other.delay
        fi = ('*'.join(f'({x})' for x in (self.freq_image, other.freq_image)
                       if x != '1') or '1')
        return Kernel(fi, delay=d)

    __rmul__ = __mul__

    def __repr__(self):
        return f'Kernel(freq_image={self.freq_image!r}, delay={self.delay!r})'


def delayed_image_text(freq_image: Optional[str], delay: Optional[str]) -> Optional[str]:
    """The frequency-image text of a kernel with an optional delay."""
    if delay is None:
        return freq_image
    base = freq_image if freq_image not in (None, '', '1') else None
    return f'exp(-I*omega*({delay}))' + (f'*({base})' if base else '')


def wrap_hook_with_delays(hook, delayed: dict, param_names=()):
    """``hook`` (``(ns, omega) -> {symbol: image}``) with the images of the
    kernels named in ``delayed`` (``{kernel name: delay text}``) multiplied by
    ``exp(-I*omega*X)``; ``X`` is evaluated in the parameter namespace (as the
    compiler evaluates a ``freq_image``)."""
    from sage.all import SR, I, exp

    def hook_d(ns, omega, _hook=hook, _delayed=dict(delayed),
               _pnames=list(param_names)):
        from api.model_compiler import (_builtin_namespace, _ns_var_namespace,
                                        _safe_eval)
        out = dict(_hook(ns, omega) or {}) if _hook is not None else {}
        env = {**_builtin_namespace(), **_ns_var_namespace(ns, [], _pnames, [])}
        for kname, dtext in _delayed.items():
            if not hasattr(ns, kname):
                continue
            ksym = getattr(ns, kname)
            X = SR(_safe_eval(dtext, env, f'kernel {kname} (delay)'))
            keys = list(ksym) if isinstance(ksym, (list, tuple)) else [ksym]
            for key in keys:
                out[key] = exp(-I * omega * X) * SR(out.get(key, SR(1)))
        return out

    return hook_d


# ── the driver ──────────────────────────────────────────────────────────


def resolve_options(opts) -> dict:
    o = dict(DEFAULTS)
    if opts:
        unknown = sorted(set(opts) - set(DEFAULTS))
        if unknown:
            raise TypeError(f'delay options: unknown key(s) {unknown}; known: '
                            f'{sorted(DEFAULTS)}')
        o.update(opts)
    if os.environ.get('DAEDALUS_DELAY_EXACT_TREE', '').strip() == '0':
        o['tree'] = 'pade'
    if o['tree'] not in ('exact', 'pade'):
        raise ValueError(f"delay tree must be 'exact' or 'pade', got {o['tree']!r}")
    if o['order'] != 'auto':
        o['order'] = int(o['order'])
        if o['order'] < 1:
            raise ValueError('delay order must be >= 1')
    o['n_start'] = max(1, int(o['n_start']))
    o['n_max'] = max(o['n_start'], int(o['n_max']))
    o['tol'] = float(o['tol'])
    o['wall_max'] = float(o['wall_max'])
    return o


_INSTABILITY_CODES = ('G2_UNSTABLE_SADDLE', 'G2_MARGINAL_MODE', 'G3_ACAUSAL_POLE')


def _observable(r, k):
    """Total and loop part of the observable on the grid (k <= 2), or at a few
    sample points of ``total_C`` (k >= 3)."""
    by = r.get('C_tau_by_ell') or {}
    if k <= 2 and r.get('C_tau') is not None:
        tot = np.asarray(r['C_tau'], dtype=complex)
        tree = np.asarray(by.get(0), dtype=complex) if by.get(0) is not None else 0 * tot
        return tot, tot - tree
    # k >= 3: total_C(t0, ..., t_{k-1}) at a few distinct, ordered times
    pts = [tuple([0.0] + [0.3 * (j + 1) * (i + 1) for i in range(k - 1)])
           for j in range(4)]
    tot = np.array([complex(r['total_C'](*p)) for p in pts])
    f0 = (r.get('total_C_by_ell') or {}).get(0)
    tree = np.array([complex(f0(*p)) for p in pts]) if f0 is not None else 0 * tot
    return tot, tot - tree


def _finite(r, k):
    try:
        tot, _ = _observable(r, k)
    except Exception:
        return False
    return bool(np.all(np.isfinite(tot)))


def _rel(a, b, scale):
    s = float(np.max(np.abs(scale))) if np.size(scale) else 0.0
    if s == 0.0:
        return float(np.max(np.abs(a - b))) if np.size(a) else 0.0
    return float(np.max(np.abs(a - b))) / s


def compute_delayed(model: dict, call: dict, delays: dict):
    """Run ``compute_cumulants`` for a model with delay kernels (``delays``:
    ``kernel_delays``); ``call``: the arguments of the original call."""
    from api.compute import compute_cumulants
    from api import analyze as _an
    from api import _delay_exact as _dx
    opts = resolve_options(call.get('delay'))
    k = int(call['k'])
    max_ell = int(call.get('max_ell') or 0)
    verbose = bool(call.get('verbose', True))
    t_all = time.perf_counter()

    # 1. the TRUE model: stability with the true exponential (G2-K), the
    #    inputs of the exact tree; refuses an unstable saddle by name
    pf_on = call.get('preflight')
    if pf_on is None:
        from api.compute import _preflight_default
        pf_on = _preflight_default()
    fundamental = call.get('parameters') if call.get('parameters') is not None \
        else call.get('fundamental')
    q = _an.Question(k=k, external_fields=tuple(call.get('external_fields') or ()),
                     max_ell=max_ell, accuracy=call.get('accuracy'),
                     parameters=fundamental,
                     fixed_point_index=call.get('fixed_point_index', 0),
                     taylor_order=call.get('taylor_order'))
    report_true, ctx = _an.analyze_with_context(model, q, call.get('policy'),
                                                use_cache=call.get('use_cache', True))
    if pf_on and report_true.errors:
        raise _an.AnalyzerError(_an._format_stop(report_true.errors, 'delay'),
                                report_true, report_true.errors, 'delay')

    # 2. the exact tree (k = 2 only), from the linear delayed system
    exact = None
    exact_note = None
    if opts['tree'] == 'exact' and k == 2:
        try:
            exact = _dx.ExactTree.from_context(ctx, call.get('external_fields'))
        except _dx.ExactTreeUnavailable as exc:
            exact_note = str(exc)
    elif opts['tree'] == 'exact':
        exact_note = f'k = {k}: the exact tree covers k = 2 only'

    # 3. the Pade orders
    if opts['order'] != 'auto':
        seq = [opts['order']]
    else:
        seq, n = [], opts['n_start']
        while n <= opts['n_max']:
            seq.append(n)
            n *= 2
        if not seq:
            seq = [opts['n_max']]
    loops_need_n = max_ell >= 1 or exact is None
    runs, walls, skipped, failed = {}, {}, {}, {}
    est, cap, n_used = None, None, None
    ests = {}
    prev = None
    sub_call = dict(call)
    sub_call.pop('model', None)
    sub_call.pop('delay', None)
    for i, n in enumerate(seq):
        if verbose:
            print(f'[delay] Pade ({n}, {n}) of exp(-i omega d): running the pipeline')
        t0 = time.perf_counter()
        try:
            r = compute_cumulants(pade_model(model, n), **sub_call)
        except _an.AnalyzerError as exc:
            codes = set(getattr(exc, 'codes', ()))
            if codes & set(_INSTABILITY_CODES) and i + 1 < len(seq):
                # the approximant moved a root across the axis (low n near a
                # Hopf point): the true model is stable (G2-K above)
                skipped[n] = sorted(codes)
                continue
            raise
        except Exception as exc:
            if prev is None:
                raise
            # a higher order the engine cannot finish (memory guard, ...):
            # the cost cap; the previous order stands
            failed[n] = f'{type(exc).__name__}: {str(exc)[:300]}'
            cap = f'order n = {n} failed ({type(exc).__name__})'
            break
        walls[n] = time.perf_counter() - t0
        if exact is not None:
            _install_exact_tree(r, exact, call)
        bad_tree = (exact is not None and r.get('delay_tree_vs_pade') is not None
                    and not r['delay_tree_vs_pade'] <= 1.0)
        if bad_tree or not _finite(r, k):
            why = ('its Pade tree is %.3g of max|C| away from the exact tree'
                   % r['delay_tree_vs_pade'] if bad_tree else 'non-finite values')
            if prev is None:
                raise RuntimeError(f'delay: the Pade ({n}, {n}) run is not usable '
                                   f'({why}); lower the order')
            failed[n] = why
            cap = f'order n = {n} is not usable ({why})'
            break
        if not loops_need_n:
            runs[n], n_used, est = r, n, 0.0
            break
        if prev is not None:
            tot_p, loop_p = _observable(runs[prev], k)
            tot_n, loop_n = _observable(r, k)
            e_n = (_rel(loop_p, loop_n, tot_n) if exact is not None
                   else _rel(tot_p, tot_n, tot_n))
            # divergence guard: the difference must shrink as n doubles; a
            # growing one (or one above 1) means the higher order lost
            # precision, and the previous pair stands
            last = ests.get(prev)
            if (last is not None and e_n > last) or e_n > 1.0:
                failed[n] = (f'the difference to n = {prev} grew to {e_n:.3g} '
                             f'(previous {last if last is not None else float("nan"):.3g})')
                cap = f'order n = {n} diverged (difference {e_n:.3g})'
                break
            ests[n] = e_n
            est = e_n
        runs[n], n_used = r, n
        if est is not None and est <= opts['tol']:
            break
        if i + 1 < len(seq):
            nxt = seq[i + 1]
            # cost of the next order: the measured growth, at least n^4 (the
            # one-loop pole-tuple rule; the state-space route is not cheaper
            # in n once the realization dominates)
            floor = (nxt / n) ** 4
            grow = max(floor, walls[n] / walls[prev]) if prev is not None and walls[prev] > 0 \
                else floor
            if walls[n] * grow > opts['wall_max']:
                cap = (f'next order n = {nxt} predicted at {walls[n] * grow:.0f} s '
                       f'> wall_max {opts["wall_max"]:.0f} s')
                break
        prev = n
    if n_used is None:
        raise RuntimeError(f'delay: no Pade order of {seq} gave a usable '
                           f'approximant (unstable: {skipped}; failed: {failed})')
    r = runs[n_used]
    fixed = opts['order'] != 'auto'
    target_met = (None if fixed and loops_need_n else
                  (est is not None and est <= opts['tol']) if loops_need_n else True)
    if est is None and loops_need_n and not fixed:
        cap = cap or 'a single order was run (no doubling estimate)'

    # 4. the record
    tree_vs_pade = None
    if exact is not None:
        tree_vs_pade = r['delay_tree_vs_pade'] if 'delay_tree_vs_pade' in r else None
    info = {
        'delays': {kname: [str(x) for x in ds] for kname, ds in delays.items()},
        'approximant': 'pade',
        'n': n_used,
        'orders_run': sorted(runs),
        'orders_skipped_unstable': skipped,
        'orders_failed': failed,
        'est_by_order': ests,
        'walls_s': {n: round(w, 3) for n, w in walls.items()},
        'est_rel_error': est,
        'tol': opts['tol'],
        'target_met': target_met,
        'cap': cap,
        'tree': ('exact' if exact is not None else 'pade'),
        'tree_method': (exact.method if exact is not None else None),
        'tree_note': exact_note,
        'tree_vs_pade_rel': tree_vs_pade,
        'stability': ctx.t.get('kernel_stability_detail'),
        'wall_s': round(time.perf_counter() - t_all, 3),
        'options': opts,
    }
    r['delay'] = info
    r.pop('delay_tree_vs_pade', None)
    _record_in_report(r, report_true, info)
    if loops_need_n and target_met is False:
        msg = (f'delay: the Pade approximation of the loops reached an '
               f'estimated relative error {est if est is not None else float("nan"):.3g} '
               f'at n = {n_used} (target {opts["tol"]:g}); {cap or "n_max reached"}.')
        from api.save import _ensure_warning_shown
        _ensure_warning_shown(DelayOrderWarning, 'always')
        warnings.warn(msg, DelayOrderWarning, stacklevel=3)
    return r


def _install_exact_tree(r, exact, call):
    """Replace the tree level of ``r`` by the exact covariance (k = 2)."""
    from api.compute import _ITO_EPS
    taus = np.asarray(r.get('tau_grid'), dtype=float)
    by = r.get('C_tau_by_ell') or {}
    pade_tree = by.get(0)
    # tau = 0 as on the pipeline's grid (and in its loops): the Ito left limit,
    # evaluated at tau = -_ITO_EPS
    c0 = exact.values(np.where(np.abs(taus) <= 1e-12, -_ITO_EPS, taus))
    if pade_tree is not None:
        pt = np.asarray(pade_tree, dtype=complex)
        scale = float(np.max(np.abs(c0))) or 1.0
        r['delay_tree_vs_pade'] = float(np.max(np.abs(pt - c0))) / scale
        r['delay_tree_pade'] = pt
    dtype = np.asarray(pade_tree).dtype if pade_tree is not None else float
    by[0] = np.asarray(c0, dtype=dtype)
    r['C_tau_by_ell'] = by
    r['C_tau'] = sum(np.asarray(v) for v in by.values() if v is not None)
    fby = r.get('total_C_by_ell') or {}
    f0 = exact.callable()
    fby[0] = f0
    r['total_C_by_ell'] = fby
    others = [f for ell, f in fby.items() if ell != 0 and f is not None]

    def total(*t, _f0=f0, _o=tuple(others)):
        return _f0(*t) + sum(f(*t) for f in _o)

    r['total_C'] = total


def _record_in_report(r, report_true, info):
    """The delay record in ``r['analysis']``: the G2-K findings of the true
    model, ``G3_DELAY_APPROXIMATED`` and the ledger's approximation entry."""
    from api import analyze as _an
    rep = r.get('analysis')
    n = info['n']
    names = ', '.join(f'{kn} (d = {", ".join(ds)})' for kn, ds in info['delays'].items())
    est = info['est_rel_error']
    loop_txt = ('the loops by Pade (%d, %d) of exp(-i omega d)' % (n, n)
                + (f', est. rel. error {est:.2g} (|C_n - C_2n|)' if est is not None else ''))
    tree_txt = ('the tree exact (linear delayed covariance, %s)' % info['tree_method']
                if info['tree'] == 'exact' else
                'the tree by the same Pade approximant' + (
                    f' ({info["tree_note"]})' if info.get('tree_note') else ''))
    level = 'warning' if info['target_met'] is False else 'info'
    finding = _an.Finding(
        code='G3_DELAY_APPROXIMATED', level=level, gate='G3',
        message=f'delay kernel(s) {names}: {tree_txt}; {loop_txt}'
                + ('' if info['target_met'] is not False else
                   f'; target {info["tol"]:g} missed ({info["cap"] or "n_max"})')
                + (f' (fixed order n = {n}: no estimate)' if info['target_met'] is None else '')
                + '.',
        remedy=('none needed' if info['target_met'] is not False else
                "raise delay={'wall_max': ...} / {'n_max': ...}, or accept the estimate"))
    entry = _an.LedgerEntry(
        kind='approximation', name='delay:' + ','.join(info['delays']),
        detail=(f'Pade({n},{n}) of exp(-i omega d) in the loops'
                + (f'; est. rel. error {est:.2g}' if est is not None else '')
                + ('; tree exact' if info['tree'] == 'exact' else '; tree Pade')),
        declared=True, satisfied=info['target_met'])
    if rep is None:
        return
    keep = [f for f in rep.findings
            if not (f.code.startswith('G2') and f.code in {g.code for g in report_true.findings})]
    g2k = [f for f in report_true.findings if f.code.startswith('G2')]
    findings = tuple(_an._sorted_findings(keep + g2k + [finding]))
    traits = replace(rep.traits, propagator_class='delay-rational')
    r['analysis'] = replace(rep, findings=findings, ledger=tuple(rep.ledger) + (entry,),
                            traits=traits)
    r['analysis_true_model'] = report_true
