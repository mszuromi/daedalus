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
  (conservative: it was measured 3-5x above the true error of ``C_n``,
  and the reported value is ``C_2n``), meets the target (default ``1e-8``),
  or until ``n_max`` or the predicted wall time of the next order exceeds the
  cap (then a warning says the target was missed).  The prediction is the
  measured wall of the last order times the growth of a cost proxy of the
  realizations, ``mean(states)^2 * (1 + mean(stiffness))`` per entry (the next
  order's realization is built and balanced-truncated as the run will be:
  ``_order_cost``), corrected by the proxy's own error on the step before;
* runs every order with its per-entry realizations balanced-truncated
  (``api._realization.truncation_scope``; option ``truncation``): the Pade
  blocks' fast, weakly coupled modes are where the state-space route's
  cost goes.  The first order uses ``min(truncation_tol(), tol / 100)``
  (``1e-12`` by default); every later order ``1 %`` of the relative loop
  error it is predicted to carry anyway (``_next_truncation``: from the
  exact-vs-Pade tree and the doubling estimate of the order before, at
  most ``1e-4``), so the per-entry error bound ``2 sum(discarded Hankel
  singular values)`` keeps the truncation far below the approximation
  error being estimated, while the high orders get the speed;
* takes the TREE (``k = 2``) from the exact covariance of the linear delayed
  system (``api._delay_exact``: a boundary-value problem on ``[0, d]`` plus
  the method of steps, exact up to rounding) instead of its Pade
  approximation, whose error decays only like ``n^-2`` at short lags.  Loops
  keep the Pade approximation (their error decays like ``n^-3`` and they are
  smaller).  ``delay={'tree': 'pade'}`` (or ``DAEDALUS_DELAY_EXACT_TREE=0``)
  switches the exact tree off.

A delay kernel at a vertex leg (a nonlinear term: ``x*x(t - d)``, a
delayed leg operator inside a rate nonlinearity) is routed like any other
conductance kernel when its approximated image is strictly proper (a delayed
filter, ``Kernel.delay(d) * Kernel.exponential(tau)``).  A bare hard delay
there has the all-pass (n, n) image ``(-1)^n + (strictly proper)``: Phase J
folds its direct term into the line it sits on (a dressed line, ``c G +
g_smooth * G``, on both routes), so the (n, n) approximant runs as
it is.  Only a line that cannot be dressed (a kernel rate equal to a pole of
the line) is skipped; ``compute_delayed`` then reruns the orders with the
strictly proper ``[n-1/n]`` approximant of every delay factor
(``pade_model(proper=True)``; the record says ``approximant='pade_proper'``).
A bare delay of a field whose propagator has a delta part (a spike train)
into a vertex with a second delta-carrying leg multiplies delta functions
and is refused by name (``bare_delays_on_delta_lines``); on a continuous
field, next to continuous legs only, or in a linear term, it runs.  The tree
stays exact.

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
``tree``     ``'exact'`` (default) or ``'pade'``;
``truncation`` ``'auto'`` (default: the first order at ``min(truncation_tol(),
             tol / 100)``, later orders adaptive, above), a fixed relative
             H-infinity tolerance for every order, or ``0`` (exact
             realizations).

The result carries ``result['delay']`` (orders tried, walls, the estimate,
whether the target was met, the tree used and its comparison with the Pade
tree), the analyzer finding ``G3_DELAY_APPROXIMATED`` and an
``approximation`` entry of the assumptions ledger (``delay:<kernel>``).  A
missed target (``n_max`` / ``wall_max`` reached, a higher order the engine
could not finish, e.g. refused by the memory budget, or a diverging
difference) sets ``result['status'] = 'target_missed'`` (``'incomplete'``
when Phase J left diagrams out) next to the ``DelayOrderWarning``.  At
k >= 3 the controller compares ``total_C`` at sample times in units of the
longest delay, capped by the slowest local relaxation time
(``result['delay']['sample_time_scale']``), so its estimate and the order it
picks do not depend on the time unit.
"""
from __future__ import annotations

import math
import os
import re
import time
import warnings
from dataclasses import replace
from fractions import Fraction
from typing import Optional

import numpy as np

__all__ = ['Kernel', 'pade_coefficients', 'pade_image', 'delay_factors',
           'pade_substitute', 'kernel_delays', 'pade_model',
           'split_shifted_time_expr', 'compute_delayed', 'DelayOrderWarning',
           'pade_mn_coefficients', 'bare_delay_kernels',
           'bare_delays_on_delta_lines', 'DEFAULTS']

DEFAULTS = dict(order='auto', tol=1e-8, n_start=4, n_max=16, wall_max=900.0,
                tree='exact', truncation='auto')

#: Cost model of the order controller: the wall of order ``m`` scales like
#: ``S^COST_EXPONENT_STATES * (1 + nu)^COST_EXPONENT_STIFFNESS``, ``S`` the mean
#: per-entry state count and ``nu`` the mean per-entry Taylor stiffness
#: ``min_p ||(a - mu)^p||_1^(1/p)`` (``mu`` = trace / n; the state-space
#: route's Taylor step count is proportional to it, the cost of a step to a
#: power of the tensor sizes; the 1 is a segment's fixed work, and keeps a
#: nilpotent Erlang chain, nu = 0, finite).  Least squares on the delayed-alpha two-neuron
#: demo (Pade-4 and 8, exact and truncated realizations, five runs from 71 s
#: to 634 s): exponents 2.16 and 0.95, every run within 10 %; the measured
#: wall ratio of the step before corrects the proxy (``_predicted_growth``).
COST_EXPONENT_STATES = 2.0
COST_EXPONENT_STIFFNESS = 1.0


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


def pade_mn_coefficients(m: int, n: int):
    """Exact coefficients of the Pade ``[m/n]`` approximant of ``exp(-z)``:
    ``(a, b)`` with ``exp(-z) ~ sum_k a_k (-z)^k / sum_k b_k z^k``,
    ``a_k = (m+n-k)! m! / ((m+n)! k! (m-k)!)``, ``b_k = (m+n-k)! n! /
    ((m+n)! k! (n-k)!)``; ``m = n`` gives ``pade_coefficients(n)`` twice.
    For ``n - 2 <= m <= n`` the denominator is Hurwitz (Ehle / Birkhoff-Varga)."""
    m, n = int(m), int(n)
    if n < 1 or not 0 <= m <= n:
        raise ValueError(f'Pade [m/n] needs n >= 1 and 0 <= m <= n, got [{m}/{n}]')
    f = math.factorial
    a = [Fraction(f(m + n - k) * f(m), f(m + n) * f(k) * f(m - k)) for k in range(m + 1)]
    b = [Fraction(f(m + n - k) * f(n), f(m + n) * f(k) * f(n - k)) for k in range(n + 1)]
    return a, b


def pade_image(z, n: int, m: Optional[int] = None):
    """``P_n(-z)/P_n(z)`` as a Sage expression in ``z`` (exact coefficients);
    with ``m`` (``< n``) the ``[m/n]`` approximant instead, strictly proper
    for ``m < n`` (``proper``: a hard delay at a vertex leg)."""
    from sage.all import SR, QQ
    a, b = pade_mn_coefficients(n if m is None else m, n)
    z = SR(z)
    num = sum(SR(QQ(ck)) * (-z) ** k for k, ck in enumerate(a))
    den = sum(SR(QQ(ck)) * z ** k for k, ck in enumerate(b))
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


def pade_substitute(expr, omega, n: int, m: Optional[int] = None):
    """``expr`` with every ``exp(alpha - I*omega*X)`` replaced by
    ``exp(alpha) * P_n(-z)/P_n(z)``, ``z = I*omega*X`` (with ``m``: the
    ``[m/n]`` approximant, ``pade_image``)."""
    from sage.all import SR, I, exp

    def rebuild(e):
        e = SR(e)
        hit = _exp_delay(e, omega)
        if hit is not None:
            alpha, X = hit
            img = pade_image(I * omega * X, n, m)
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
    specs = getattr(hook, '_kernel_specs', None)
    if specs is not None and not specs_may_delay(specs):
        return {}                       # a builder hook whose texts carry no delay
    omega = SR.var('omega')
    imgs = hook(ns, omega) or {}
    out = {}
    for key, val in imgs.items():
        ds = delay_factors(SR(val), omega)
        if ds:
            out[str(key)] = ds
    return out


_SHIFT_RE = r'\b(dirac_delta|heaviside|unit_step)\s*\(\s*t\s*[-+]'


def specs_may_delay(specs) -> bool:
    """Whether a builder's kernel declarations can produce a delay factor: a
    ``delay``, an ``exp(`` in a frequency image, or a shifted time image."""
    import re
    for k in specs or ():
        if k.get('delay') or re.search(r'\bexp\s*\(', str(k.get('freq_image') or '')) \
                or re.search(_SHIFT_RE, str(k.get('time_expr') or '')):
            return True
    return False


def pade_model(model: dict, n: int, proper: bool = False) -> dict:
    """A copy of ``model`` whose kernel images carry the Pade (n, n)
    approximant of every delay factor (``proper``: the strictly proper
    ``[n-1/n]`` instead, for a hard delay at a vertex leg; see
    ``compute_delayed``) (its own name and fingerprint, so its
    caches never mix with the true model's or another order's).  The copy
    asks the propagator builder to skip the symbolic inverse (the numbers are
    substituted before inverting; the symbolic inverse of a Pade-n kernel
    matrix is the slow part and nothing downstream needs it)."""
    from sage.all import SR
    hook = model['kernel_ft_image']

    m_ = int(n) - 1 if proper else None

    def hook_n(ns, omega, _hook=hook, _n=int(n), _m=m_):
        out = _hook(ns, omega) or {}
        return {k: pade_substitute(SR(v), omega, _n, _m) for k, v in out.items()}

    m = dict(model)
    m['kernel_ft_image'] = hook_n
    td = model.get('kernel_td_image')
    if td is not None:
        # a delayed kernel's time image (the unshifted one for a shifted
        # time_expr) must not reach a conductance vertex: without it the
        # vertex takes the (Pade) frequency image, like every other route
        def td_n(ns, t, _td=td, _hook=hook):
            out = dict(_td(ns, t) or {})
            om = SR.var('omega')
            delayed = {k for k, v in (_hook(ns, om) or {}).items()
                       if delay_factors(SR(v), om)}
            return {k: v for k, v in out.items() if k not in delayed}
        m['kernel_td_image'] = td_n
    base = model.get('name', 'model')
    tag = f'Pade [{int(n) - 1}/{int(n)}]' if proper else f'Pade {int(n)}'
    m['name'] = f'{base} [delay {tag}]'
    m['spec_signature'] = (f"{model.get('spec_signature', '')}-pade{int(n)}"
                           + ('sp' if proper else ''))
    m['_delay_pade'] = {'n': int(n), 'model': base, 'proper': bool(proper)}
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
    if not re.search(_SHIFT_RE, str(text)):
        return None
    from sage.all import SR, var, heaviside, dirac_delta
    names = set(re.findall(r'\b([A-Za-z_]\w*)\b', text)) - {
        'dirac_delta', 'heaviside', 'exp', 'log', 'sqrt', 'sin', 'cos', 'I',
        'pi', 'e', 'unit_step'}
    loc = {nm: var(nm) for nm in names}
    loc.update(dirac_delta=dirac_delta, heaviside=heaviside, unit_step=heaviside)
    try:
        e = SR(eval(str(text).replace('^', '**'), {'__builtins__': {}}, dict(loc, **{
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
    ``Kernel.alpha(tau)``: ``t/tau^2 exp(-t/tau)`` (image ``1/(1 + I omega tau)^2``);
    ``Kernel.gamma(n, mean)``: the Erlang density of order ``n`` and mean
    ``mean`` (image ``(1 + I omega mean/n)^-n``; rational: an exact gamma
    delay, no approximation);
    ``Kernel.image(text)``: any frequency image;
    ``Kernel.gaussian(sigma, tol=1e-6)``: the Gaussian jitter density
    ``exp(-t^2/(2 sigma^2))/sqrt(2 pi sigma^2)`` of an EXOGENOUS input, by a
    rational fit of the time-shifted density (``api._gaussian_fit``; the SDE
    front end refuses it on anything but an exogenous source).
    Products compose (convolution): ``Kernel.delay('d') * Kernel.exponential('tau')``.
    Arguments are parameter names or expressions (text)."""

    def __init__(self, freq_image: str = '1', delay: Optional[str] = None,
                 jitter: Optional[dict] = None):
        self.freq_image = str(freq_image)
        self.delay = None if delay is None else str(delay)
        self.jitter = jitter

    @classmethod
    def delay(cls, d):
        return cls('1', delay=str(d))

    @classmethod
    def exponential(cls, tau):
        return cls(f'1/(1 + I*omega*({tau}))')

    @classmethod
    def alpha(cls, tau):
        """The alpha function ``t/tau^2 exp(-t/tau)`` (image ``1/(1 + I omega tau)^2``)."""
        t = str(tau).strip()
        t = t if re.fullmatch(r'[\w\[\],\s]*', t) else f'({t})'
        return cls(f'1/(1 + I*omega*{t})^2')

    @classmethod
    def gamma(cls, n, mean):
        n = int(n)
        if n < 1:
            raise ValueError('Kernel.gamma: n must be >= 1')
        return cls(f'(1 + I*omega*({mean})/{n})^(-{n})')

    @classmethod
    def image(cls, text):
        return cls(str(text))

    @classmethod
    def gaussian(cls, sigma, tol=1e-6):
        """The Gaussian density ``exp(-t^2/(2 sigma^2))/sqrt(2 pi sigma^2)``
        (image ``exp(-sigma^2 omega^2/2)``), for the jitter of an EXOGENOUS
        input only: ``Conv(kX*g, n)`` with ``n`` a source whose rate no field
        feeds back into.  The density is acausal; it is shifted by
        ``T = shift*sigma`` (``Phi(-T/sigma) <= tol/10``) and replaced by a
        rational fit of the shifted density with L1 error ``<= tol``
        (``api._gaussian_fit``: Laguerre expansion, balanced truncation,
        least-squares numerators; 16 states at ``tol = 1e-6``).  The shift
        is dropped: shifting a stationary exogenous input in time changes no
        stationary statistic of the network.  ``sigma`` is a parameter name or
        an expression of unindexed parameters; ``tol`` in ``[1e-9, 1e-2]``.
        ``.jitter['fit']`` holds the fit and its errors (``bound(k)``: the
        relative error bound of a k-th order statistic)."""
        from api._gaussian_fit import gaussian_fit, TOL_RANGE
        tol = float(tol)
        if not TOL_RANGE[0] <= tol <= TOL_RANGE[1]:
            raise ValueError(f'Kernel.gaussian: tol={tol:g} outside [{TOL_RANGE[0]:g}, '
                             f'{TOL_RANGE[1]:g}]')
        sg = str(sigma).strip()
        if not sg:
            raise ValueError('Kernel.gaussian: empty sigma')
        sg = sg if re.fullmatch(r'\w+', sg) else f'({sg})'
        fit = gaussian_fit(tol)
        return cls(fit.image(f'I*omega*{sg}'),
                   jitter={'kind': 'gaussian', 'sigma': sg, 'tol': tol, 'fit': fit})

    def __mul__(self, other):
        if not isinstance(other, Kernel):
            return NotImplemented
        if self.jitter and other.jitter:
            raise ValueError('Kernel product: two Gaussian jitters; declare one '
                             'Kernel.gaussian with sigma = sqrt(sigma1^2 + sigma2^2).')
        if self.delay and other.delay:
            d = f'({self.delay}) + ({other.delay})'
        else:
            d = self.delay or other.delay
        facs = [x for x in (self.freq_image, other.freq_image) if x != '1']
        fi = facs[0] if len(facs) == 1 else ('*'.join(f'({x})' for x in facs) or '1')
        return Kernel(fi, delay=d, jitter=self.jitter or other.jitter)

    __rmul__ = __mul__

    def __repr__(self):
        if self.jitter:
            j = self.jitter
            return (f"Kernel(gaussian jitter sigma={j['sigma']!r}, tol={j['tol']:g}, "
                    f"{j['fit'].order} states, delay={self.delay!r})")
        return f'Kernel(freq_image={self.freq_image!r}, delay={self.delay!r})'


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
            def flat(x):
                if isinstance(x, (list, tuple)):
                    for y in x:
                        yield from flat(y)
                else:
                    yield x
            for key in flat(ksym):
                out[key] = exp(-I * omega * X) * SR(out.get(key, SR(1)))
        return out

    hook_d._kernel_specs = getattr(hook, '_kernel_specs', None)
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
    from engine import _env
    if not _env.flag('DAEDALUS_DELAY_EXACT_TREE', True):
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
    if o['truncation'] != 'auto':
        try:
            t = float(o['truncation'])
        except (TypeError, ValueError):
            t = float('nan')
        if not (t >= 0.0 and np.isfinite(t)):
            raise ValueError(f"delay truncation must be 'auto' or a tolerance >= 0, "
                             f"got {o['truncation']!r}")
        o['truncation'] = t
    return o


def _truncation_tol(opts) -> float:
    """The relative H-infinity tolerance of the per-entry balanced truncation
    of the first Pade order (and the floor of the later ones): ``'auto'`` =
    the tolerance in effect (environment / default, ``api._realization.
    truncation_tol``) capped at ``tol / 100``; ``0`` when that is exact."""
    from api._realization import truncation_tol
    if opts['truncation'] == 'auto':
        t = truncation_tol()
        return min(t, 0.01 * opts['tol']) if t > 0 else 0.0
    return float(opts['truncation'])


def _stiffness(a):
    """``min_p ||(a - mu I)^p||_1^(1/p)``, ``p = 1 .. 9``, ``mu = tr(a) / n``:
    the quantity the state-space route's Taylor step count follows."""
    a = np.asarray(a)
    n = a.shape[0]
    if n == 0:
        return 0.0
    A = a - (np.trace(a) / n) * np.eye(n)
    P = np.eye(n, dtype=A.dtype)
    best = math.inf
    for p in range(1, 10):
        P = P @ A
        best = min(best, float(np.abs(P).sum(axis=0).max()) ** (1.0 / p))
    return best


def _order_cost(ctx, m, cache, tol):
    """``{'states': S, 'stiffness': nu, 'proxy': S^2 (1 + nu)}`` (means over the
    nonzero entries) of the Pade-``m`` realization of the TRUE model's kernel
    matrix at its saddle, balanced-truncated to ``tol`` (as the run of that
    order is), or ``None`` when it cannot be built (the controller then
    falls back to the measured growth).  Cached per (order, tolerance)."""
    key = (int(m), float(tol))
    if key in cache:
        return cache[key]
    out = None
    try:
        from sage.all import SR
        from api._realization import realize_K
        if ctx is not None and ctx.K_ft is not None and ctx.num_params is not None:
            om = ctx.omega
            Km = ctx.K_ft.apply_map(lambda e: pade_substitute(SR(e), om, m))
            real = realize_K(Km, om, ctx.num_params, truncation=tol)
            ents = [e for e in real.entries.values() if e.n_states > 0]
            if ents:
                S = float(np.mean([e.n_states for e in ents]))
                nu = float(np.mean([_stiffness(e.a) for e in ents]))
                out = {'states': S, 'stiffness': nu,
                       'proxy': S ** COST_EXPONENT_STATES
                       * (1.0 + nu) ** COST_EXPONENT_STIFFNESS}
    except Exception:
        out = None
    cache[key] = out
    return out


#: The delay route's adaptive truncation (``delay={'truncation': 'auto'}``):
#: after the first order, each order's realizations are truncated to
#: ``TRUNCATION_KAPPA`` times the relative error its Pade loops are predicted
#: to carry (``_next_truncation``), at most ``TRUNCATION_MAX``.
TRUNCATION_KAPPA = 1e-2
TRUNCATION_MAX = 1e-4


def _next_truncation(opts, t_base, r, k, n, m, est, est_ref, ctrl):
    """The truncation tolerance of order ``m``, the last successful order
    being ``n`` (result ``r``; ``est``: its doubling estimate, which measures
    order ``est_ref``, the successful order before it).  A fixed
    ``delay={'truncation': t}`` (or the exact setting, ``t_base = 0``) is
    kept.  Otherwise ``max(t_base, min(TRUNCATION_MAX, TRUNCATION_KAPPA *
    e_m))`` with ``e_m`` the smaller of two predictions of the relative loop
    error of order ``m`` (in units of ``max |C|``, as the order estimate),
    each carried to order ``m`` by the loops' ``n^-3`` decay: the
    exact-vs-Pade tree of order ``n`` times the loops' share of ``max |C|``,
    and the doubling estimate (``est * (est_ref / m)^3``).  They are
    predictions, not bounds, and on the benchmarks they OVERestimate (the
    tree's Pade error decays more slowly than the loops'; the loops fall
    like ``n^-3.5``): measured 8x on the delayed-alpha demo and 11x on the
    delayed OU of the tests, so the per-entry bound is ~10 % of the loops'
    actual Pade error, and the observable moves by a small fraction of the
    per-entry bound (1.4e-2 on the demo at n = 8: 1.5e-8 of max |C| against
    a Pade error of ~1.4e-5).  ``None`` predictions (no exact tree, a single
    order): ``t_base``."""
    if opts['truncation'] != 'auto' or not t_base > 0:
        return t_base
    preds = []
    tvp = r.get('delay_tree_vs_pade')
    if tvp is not None and np.isfinite(tvp) and tvp > 0:
        try:
            tot, loop = _observable(r, k, ctrl.get('sample_time_scale') or 1.0)
            share = float(np.max(np.abs(loop))) / max(float(np.max(np.abs(tot))), 1e-300)
        except Exception:
            share = 1.0
        preds.append(float(tvp) * min(share, 1.0) * (n / m) ** 3)
    if est is not None and est_ref and np.isfinite(est) and est > 0:
        preds.append(float(est) * (est_ref / m) ** 3)
    preds = [p for p in preds if p > 0]
    if not preds:
        return t_base
    e_m = min(preds)
    ctrl['loop_error_pred'][m] = e_m
    return max(t_base, min(TRUNCATION_MAX, TRUNCATION_KAPPA * e_m))


_INSTABILITY_CODES = ('G2_UNSTABLE_SADDLE', 'G2_MARGINAL_MODE', 'G3_ACAUSAL_POLE')


def _delay_time_scale(ctx) -> float:
    """The time scale of the k >= 3 sample points of ``_observable``, in the
    model's time unit: the longest delay (numeric), capped by the slowest
    local relaxation time (``1 / min |Re lambda|`` of the saddle's
    linearization with every kernel at ``g_hat(0)``, the analyzer's G2
    eigenvalues), so the points stay inside the correlation window when the
    delay is long against the relaxation; 1.0 when neither can be read."""
    det = (getattr(ctx, 't', None) or {}).get('kernel_stability_detail') or {}
    vals = [abs(float(x)) for x in (det.get('delays') or ())]
    if not vals:
        try:
            from sage.all import SR
            for e in ctx.K_ft.list():
                for X in delay_factors(e, ctx.omega):
                    vals.append(abs(float(SR(X).subs(ctx.num_params))))
        except Exception:
            pass
    vals = [v for v in vals if np.isfinite(v) and v > 0]
    T = max(vals) if vals else None
    try:
        rates = [-complex(z).real for z in ((getattr(ctx, 't', None) or {}).get('eigenvalues')
                                            or ())]
    except (TypeError, ValueError):
        rates = []
    rates = [x for x in rates if np.isfinite(x) and x > 0]
    if rates:
        T = min(T, 1.0 / min(rates)) if T is not None else 1.0 / min(rates)
    return T if T is not None else 1.0


def _sample_points(k, t_scale=1.0):
    """The k >= 3 sample points of ``_observable`` (times in units of
    ``t_scale``)."""
    return [tuple([0.0] + [0.3 * (j + 1) * (i + 1) * t_scale for i in range(k - 1)])
            for j in range(4)]


def _with_sample_points(sub_call, k, t_scale):
    """``sub_call`` for a k >= 3 evaluate-and-discard run with fixed points
    (``points``, or ``tau_grid=[0.0]``: the equal-time point): the sample
    points of ``_observable`` are added, so the error estimate and the order
    the controller picks are those of the same run without discard."""
    if k < 3 or sub_call.get('discard', False) is False:
        return sub_call
    pts = sub_call.get('points')
    if pts is None:
        from api._grids import resolve_grid
        tg = resolve_grid(sub_call.get('tau_grid'), 'tau_grid')
        if tg is None:
            return sub_call
        tg = np.asarray(tg, dtype=float).ravel()
        if tg.size != 1 or float(tg[0]) != 0.0:
            return sub_call           # no fixed points: compute_cumulants says so
        pts = [(0.0,) * k]
    pts = [tuple(float(x) for x in p) for p in np.asarray(pts, dtype=float)]
    return dict(sub_call, points=pts + _sample_points(k, t_scale))


def _observable(r, k, t_scale=1.0):
    """Total and loop part of the observable on the grid (k <= 2), or at a few
    sample points of ``total_C`` (k >= 3), at times in units of ``t_scale``
    (``_delay_time_scale``: the points, the error estimate and the order the
    controller picks do not depend on the time unit)."""
    by = r.get('C_tau_by_ell') or {}
    if k <= 2 and r.get('C_tau') is not None:
        tot = np.asarray(r['C_tau'], dtype=complex)
        tree = np.asarray(by.get(0), dtype=complex) if by.get(0) is not None else 0 * tot
        return tot, tot - tree
    # k >= 3: total_C(t0, ..., t_{k-1}) at a few distinct, ordered times
    pts = _sample_points(k, t_scale)
    if r.get('C_points') is not None:
        # evaluate-and-discard: the callables answer the run's fixed points
        # only, and ``_run_orders`` added these sample points to them
        where = {tuple(p): i for i, p in
                 enumerate(np.asarray(r['points'], dtype=float).tolist())}
        sel = [where[p] for p in pts]
        tot = np.asarray(r['C_points'], dtype=complex)[sel]
        t0 = (r.get('C_points_by_ell') or {}).get(0)
        return tot, tot - (np.asarray(t0, dtype=complex)[sel]
                           if t0 is not None else 0 * tot)
    tot = np.array([complex(r['total_C'](*p)) for p in pts])
    f0 = (r.get('total_C_by_ell') or {}).get(0)
    tree = np.array([complex(f0(*p)) for p in pts]) if f0 is not None else 0 * tot
    return tot, tot - tree


def _finite(r, k, t_scale=1.0):
    try:
        tot, _ = _observable(r, k, t_scale)
    except Exception:
        return False
    return bool(np.all(np.isfinite(tot)))


def _rel(a, b, scale):
    s = float(np.max(np.abs(scale))) if np.size(scale) else 0.0
    if s == 0.0:
        return float(np.max(np.abs(a - b))) if np.size(a) else 0.0
    return float(np.max(np.abs(a - b))) / s


def _warn_explicit_module(filename):
    """The module name ``warnings.warn_explicit`` derives from ``filename``
    when it is given no ``module``."""
    module = filename or '<unknown>'
    return module[:-3] if module[-3:].lower() == '.py' else module


def _reemit(recorded):
    """Re-emit warnings held back by ``catch_warnings(record=True)``: each
    distinct one once, the package's own (``api.*`` / ``engine.*``) also where
    the environment ignores warnings (the filters the pipeline set with
    ``_ensure_warning_shown`` were rolled back with the context)."""
    from api.save import _ensure_warning_shown
    seen = set()
    for w in recorded:
        key = (w.category, str(w.message), w.filename, w.lineno)
        if key in seen:
            continue
        seen.add(key)
        if str(getattr(w.category, '__module__', '')).split('.')[0] in ('api', 'engine'):
            _ensure_warning_shown(w.category, 'always', message=w.message,
                                  module=_warn_explicit_module(w.filename))
        warnings.warn_explicit(w.message, w.category, w.filename, w.lineno)


def bare_delay_kernels(model: dict, ns) -> list:
    """The delayed kernels whose image without its delay factor is not
    strictly proper (a bare hard delay ``delta(t - d)``, or a delta part next to
    the delay), as sorted names; a kernel whose degree cannot be decided counts
    as bare (conservative)."""
    from sage.all import SR, exp
    hook = model.get('kernel_ft_image')
    if hook is None or ns is None:
        return []
    omega = SR.var('omega')

    def strip(e):
        e = SR(e)
        hit = _exp_delay(e, omega)
        if hit is not None:
            return exp(hit[0])
        ops = e.operands()
        return e if not ops else e.operator()(*[strip(o) for o in ops])

    out = []
    for key, val in (hook(ns, omega) or {}).items():
        if not delay_factors(SR(val), omega):
            continue
        try:
            e = SR(strip(SR(val))).rational_simplify()
            proper = SR(e.numerator()).degree(omega) < SR(e.denominator()).degree(omega)
        except Exception:
            proper = False
        if not proper:
            out.append(str(key))
    return sorted(out)


def bare_delays_on_delta_lines(ctx, bare, orders=(1, 2), ns=None):
    """D10: the bare delay kernels among ``bare`` (names,
    ``bare_delay_kernels``) that delay a field whose propagator has a delta
    part (a point-process field) inside a vertex with a second delta-carrying
    physical leg (a spike train inside a nonlinearity, e.g. ``exp(w n(t -
    d))`` or ``n(t) n(t - d)``: a product of delta functions), as sorted
    ``'kernel (field)'`` strings.  A bare delay of a continuous field, one in
    a vertex whose other physical legs are continuous (``x(t) n(t - d)``), or
    one that only enters vertices with a single physical leg (a linear term)
    is not listed.  ``ctx``: the true model's analyzer context (its vertex
    sectors: monomials over ``ctx.ring_names`` with their coefficients);
    ``ns``: the expanded theory's namespace (its kernel attachments; default
    ``ctx.ns``, which has them only when the analyzer expanded the theory
    itself).  The delta parts are those of ``G = K^{-1}`` with every delay
    factor replaced by its Pade (n, n) image for each ``n`` in ``orders``
    (both parities: the image tends to ``(-1)^n``, so a delayed spike train
    ``delta(t - d)`` counts too).  Raises when it cannot decide (the caller
    then falls back to the per-model test)."""
    from sage.all import SR
    from api import analyze as _an
    raw = None
    for ns_ in (ns, ctx.ns, ctx.pipeline_ns):
        raw = getattr(ns_, '_kernel_attachments', None) if ns_ is not None else None
        if raw:
            break
    att = {str(k): {str(f) for f in v} for k, v in (raw or {}).items()}
    if not att or ctx.K_ft is None or ctx.num_params is None:
        raise ValueError('no kernel attachments / linear operator to read')
    names = list(ctx.ring_names)
    nt = int(ctx.n_tilde)
    delta_fields = set()
    for n in orders:
        K = ctx.K_ft.apply_map(
            lambda e, _n=int(n): pade_substitute(SR(e).subs(ctx.num_params), ctx.omega, _n))
        res = _an._exact_poles(K, ctx.omega, nt)
        if res is None:
            raise ValueError('the propagator inverse did not finish within its budget')
        delta_fields |= {names[nt + i] for (i, _j) in res[3]}
    # the kernels in a vertex with at least two delta-carrying physical legs
    # (the delayed leg included: it is a leg of the field it delays)
    delta_idx = [j for j, nm in enumerate(names) if nm in delta_fields]
    in_delta_products = None
    if ctx.sectors is not None:
        in_delta_products = set()
        for (_nr, n_phys), terms in ctx.sectors.items():
            if n_phys < 2:
                continue
            for expo, coeff in terms.items():
                expo = tuple(expo)
                if len(expo) != len(names):
                    raise ValueError(f'a vertex monomial {expo} does not match the ring {names}')
                if sum(int(expo[j]) for j in delta_idx) >= 2:
                    in_delta_products |= {str(v) for v in SR(coeff).variables()}
    out = []
    for kname in bare:
        if in_delta_products is not None and kname not in in_delta_products:
            continue
        for f in sorted(att.get(kname, ())):
            if f in delta_fields:
                out.append(f'{kname} ({f})')
    return sorted(out)


def compute_delayed(model: dict, call: dict, delays: dict, ns=None):
    """Run ``compute_cumulants`` for a model with delay kernels (``delays``:
    ``kernel_delays``); ``call``: the arguments of the original call; ``ns``:
    the expanded theory's namespace (``bare_delay_kernels``)."""
    from api.compute import compute_cumulants
    from api import analyze as _an
    from api import _delay_exact as _dx
    opts = resolve_options(call.get('delay'))
    k = int(call['k'])
    max_ell = int(call.get('max_ell') or 0)
    verbose = bool(call.get('verbose', True))
    t_all = time.perf_counter()
    if model.get('spatial'):
        raise ValueError(
            f"model {model.get('name')!r}: delay kernels {sorted(delays)} in a spatial model are "
            'not supported (the delay route covers temporal models: Pade loops, exact tree)')

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
    if report_true.errors:                  # preflight=False: still never silent
        from api.save import _ensure_warning_shown
        _msg = ('delay: preflight=False, but the analyzer of the delayed model found '
                + '; '.join(f'[{f.code}] {f.message}' for f in report_true.errors))
        _ensure_warning_shown(DelayOrderWarning, 'always', message=_msg,
                              stacklevel=3)
        warnings.warn(_msg, DelayOrderWarning, stacklevel=3)

    # 2. the exact tree (k = 2 only), from the linear delayed system
    exact = None
    exact_note = None
    if opts['tree'] == 'exact' and k == 2:
        tg = call.get('tau_grid')
        tmax = (float(np.max(np.abs(np.asarray(tg, dtype=float)))) if tg is not None
                and np.size(tg) else float(call.get('tau_max') or 0.0))
        try:
            exact = _dx.ExactTree.from_context(ctx, call.get('external_fields'), tau_max=tmax)
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
    for key in ('model', 'delay', 'output_npz', 'output_csv', 'spectra'):
        sub_call.pop(key, None)            # the files and the spectra: once, from the result
    incomplete = None
    # every order runs with its per-entry realizations balanced-truncated
    # (api._realization; the tolerance of each order: _next_truncation)
    # a bare hard delay kernel can make the (n, n) pass incomplete (below):
    # only then are its warnings held back (a retry that completes replaces
    # them); every other model runs exactly as before
    bare = bare_delay_kernels(model, ns)
    w_first = []
    if bare:
        with warnings.catch_warnings(record=True) as w_first:
            warnings.simplefilter('always')
            out = _run_orders(model, sub_call, seq, opts, k, exact, loops_need_n, ctx,
                              runs, walls, skipped, failed, ests, verbose, call)
    else:
        out = _run_orders(model, sub_call, seq, opts, k, exact, loops_need_n, ctx,
                          runs, walls, skipped, failed, ests, verbose, call)
    est, cap, n_used, incomplete, ctrl = out
    approximant = 'pade'
    # the delta part is read off the (rational) approximated model's analysis
    # (the true model's G3 does not see poles through a delay factor)
    _an_first = (runs.get(n_used) or {}).get('analysis') if n_used is not None else None
    offenders = []
    if incomplete and bare:
        # D10: per line, not per model -- only a bare delay of a
        # field with a delta part into a vertex with a second such leg
        try:
            offenders = bare_delays_on_delta_lines(ctx, bare, ns=ns)
        except Exception as exc:          # undecided: the per-model test
            why = f'undecided: {type(exc).__name__}: {str(exc)[:120]}'
            offenders = [f'{b} ({why})' for b in bare] if any(
                getattr(rep_, 'has', lambda c: False)('G3_DELTA_COMPONENT')
                for rep_ in (report_true, _an_first) if rep_ is not None) else []
    if incomplete and offenders:
        # a hard delay of a point-process (delta) field into a vertex with a
        # second delta-carrying leg: the vertex multiplies delta functions
        # (exp of a delayed spike train), which is not defined; every strictly
        # proper approximant has a finite value that diverges as n grows, so
        # none is run
        cap = (f'refused: the hard delay kernel(s) {offenders} delay a field whose '
               'propagator has a delta part (a point-process field) into a vertex with a '
               'second such leg: it multiplies delta functions and is not defined; give '
               'the delay a filter, e.g. Kernel.delay(d) * Kernel.exponential(tau)')
    elif incomplete and bare:
        # Phase J skipped diagrams with a bare delay at a vertex leg: its
        # (n, n) image is all-pass, (-1)^n + strictly proper; Phase J dresses
        # the line with it, but a line it cannot dress (a kernel
        # rate equal to a pole of the line) is skipped.  Run the strictly
        # proper [n-1/n] approximant instead (it also converges to
        # exp(-s d), the tree stays exact); the first attempt's warnings are
        # dropped when it completes, re-emitted otherwise.
        if verbose:
            print('[delay] diagrams skipped with the (n, n) approximant: '
                  'running the strictly proper [n-1/n] approximant')
        first = (dict(runs), dict(walls), dict(skipped), dict(failed), dict(ests), out)
        for dct in (runs, walls, skipped, failed, ests):
            dct.clear()
        try:
            out2 = _run_orders(model, sub_call, seq, opts, k, exact, loops_need_n, ctx,
                               runs, walls, skipped, failed, ests, verbose, call,
                               proper=True)
        except Exception as exc:          # the first attempt's record stands
            out2 = None
            proper_error = f'{type(exc).__name__}: {str(exc)[:300]}'
        if out2 is not None and out2[2] is not None and not out2[3]:
            est, cap, n_used, incomplete, ctrl = out2
            approximant = 'pade_proper'
            w_first = []
        else:
            for dct, old in zip((runs, walls, skipped, failed, ests), first[:5]):
                dct.clear()
                dct.update(old)
            if out2 is None:
                cap = f'{cap}; the strictly proper [n-1/n] approximant failed ({proper_error})'
            else:
                cap = f'{cap}; the strictly proper [n-1/n] approximant left diagrams skipped too'
    _reemit(w_first)
    if n_used is None:
        raise RuntimeError(f'delay: no Pade order of {seq} gave a usable '
                           f'approximant (unstable: {skipped}; failed: {failed})')
    r = runs[n_used]
    fixed = opts['order'] != 'auto'
    target_met = (None if fixed and loops_need_n else
                  (est is not None and est <= opts['tol']) if loops_need_n else True)
    if incomplete:
        target_met = False
    elif target_met is False:
        # the requested accuracy was not reached (n_max / wall_max, a higher
        # order the engine could not finish, a diverging difference): the
        # result is complete but marked, not 'ok' (the record, the report and
        # a DelayOrderWarning say why)
        r['status'] = 'target_missed'
    if est is None and loops_need_n and not fixed:
        cap = cap or 'a single order was run (no doubling estimate)'
        if ctrl['stop'].startswith('n_max'):
            ctrl['stop'] = cap
    _finish(r, model, call, opts, delays, report_true, ctx, exact, exact_note, runs, walls,
            skipped, failed, ests, est, cap, n_used, incomplete, target_met, loops_need_n,
            t_all, ctrl, approximant)
    return r


def _run_orders(model, sub_call, seq, opts, k, exact, loops_need_n, ctx, runs, walls,
                skipped, failed, ests, verbose, call, proper=False):
    """The order loop of ``compute_delayed``; returns ``(est, cap, n_used,
    incomplete, ctrl)`` (``ctrl``: the controller's record: state counts,
    predictions, cost exponent, why it stopped)."""
    from api.compute import compute_cumulants
    from api import analyze as _an
    from api._realization import truncation_scope
    est, cap, n_used, incomplete, prev = None, None, None, None, None
    states_cache = {}
    t_base = _truncation_tol(opts)
    ctrl = {'cost_by_order': {}, 'predicted_walls_s': {}, 'cost_correction': None,
            'stop': None, 'truncation_by_order': {}, 'loop_error_pred': {},
            'sample_time_scale': _delay_time_scale(ctx) if k >= 3 else None}
    t_scale = ctrl['sample_time_scale'] or 1.0
    sub_call = _with_sample_points(sub_call, k, t_scale)
    last = None                    # (result, order, its estimate, the order that measures)
    for i, n in enumerate(seq):
        t_cur = t_base if last is None else _next_truncation(opts, t_base, *_lastargs(last, k, n),
                                                              ctrl)
        if verbose:
            print(f'[delay] Pade ({n - 1 if proper else n}, {n}) of exp(-i omega d): '
                  'running the pipeline'
                  + (f' (realizations truncated to {t_cur:.2g})' if t_cur else ''))
        ctrl['truncation_by_order'][n] = t_cur
        t0 = time.perf_counter()
        try:
            with truncation_scope(t_cur):
                r = compute_cumulants(pade_model(model, n, proper), **sub_call)
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
            if type(exc).__name__ == 'PhaseJResourceError':
                cap += (': the memory budget refused it (raise PHASE_J_MEM_BUDGET_GB '
                        'to run it)')
            break
        walls[n] = time.perf_counter() - t0
        if exact is not None:
            _install_exact_tree(r, exact, call)
        if r.get('status') == 'incomplete':
            # diagrams Phase J could not evaluate (a hard delay inside a
            # nonlinear term on a line Phase J cannot dress with its all-pass
            # image); no order fixes it, the caller (compute_delayed) retries
            # with the [n-1/n] approximant
            runs[n], n_used, incomplete = r, n, r.get('phase_j_incomplete')
            cap = ('Phase J skipped diagrams (see result["phase_j_incomplete"]; a delay '
                   'kernel at a vertex leg, e.g. x*x(t - d), on a line that cannot be '
                   'dressed with its direct term needs a strictly proper image)')
            est = None
            break
        bad_tree = (loops_need_n and exact is not None
                    and r.get('delay_tree_vs_pade') is not None
                    and not r['delay_tree_vs_pade'] <= 1.0)
        if bad_tree or not _finite(r, k, t_scale):
            why = ('its Pade tree is %.3g of max|C| away from the exact tree'
                   % r['delay_tree_vs_pade'] if bad_tree else 'non-finite values')
            if i + 1 < len(seq):
                skipped[n] = why            # too low for this delay: try the next order
                continue
            if prev is None:
                raise RuntimeError(f'delay: no Pade order up to n = {n} is usable ({why}); '
                                   "raise delay={'n_max': ...} (a long delay needs a higher order)")
            failed[n] = why
            cap = f'order n = {n} is not usable ({why})'
            break
        if not loops_need_n:
            runs[n], n_used, est = r, n, 0.0
            ctrl['stop'] = f'no loop needs the approximant (exact tree): n = {n}'
            break
        if prev is not None:
            tot_p, loop_p = _observable(runs[prev], k, t_scale)
            tot_n, loop_n = _observable(r, k, t_scale)
            e_n = (_rel(loop_p, loop_n, tot_n) if exact is not None
                   else _rel(tot_p, tot_n, tot_n))
            # divergence guard: the difference must shrink as n doubles; a
            # growing one (or one above 1) means the higher order lost
            # precision, and the previous pair stands
            last = ests.get(prev)
            if last is not None and e_n > last:
                failed[n] = (f'the difference to n = {prev} grew to {e_n:.3g} '
                             f'(previous {last if last is not None else float("nan"):.3g})')
                cap = f'order n = {n} diverged (difference {e_n:.3g})'
                break
            ests[n] = e_n
            est = e_n
        runs[n], n_used = r, n
        last = (r, n, ests.get(n), prev)
        if est is not None and est <= opts['tol']:
            ctrl['stop'] = f'target {opts["tol"]:g} met at n = {n} (estimate {est:.3g})'
            break
        if i + 1 < len(seq):
            nxt = seq[i + 1]
            t_nxt = _next_truncation(opts, t_base, *_lastargs(last, k, nxt), ctrl)
            grow, how = _predicted_growth(ctx, n, nxt, prev, walls, states_cache, ctrl,
                                          tols=ctrl['truncation_by_order'] | {nxt: t_nxt},
                                          r_last=r)
            pred = walls[n] * grow
            ctrl['predicted_walls_s'][nxt] = round(pred, 1)
            if verbose:
                print(f'[delay] n = {nxt} predicted at {pred:.0f} s ({how})')
            if pred > opts['wall_max']:
                cap = (f'next order n = {nxt} predicted at {pred:.0f} s '
                       f'> wall_max {opts["wall_max"]:.0f} s ({how})')
                break
        prev = n
    if ctrl['stop'] is None:
        ctrl['stop'] = cap or (f'fixed order n = {seq[0]}' if opts['order'] != 'auto'
                               else f'n_max = {opts["n_max"]} reached')
    return est, cap, n_used, incomplete, ctrl


def _lastargs(last, k, m):
    """``_next_truncation``'s ``(r, k, n, m, est, est_ref)`` for order ``m``
    from the last successful order ``last = (r, n, est, est_ref)``."""
    r, n, est, est_ref = last
    return r, k, n, m, est, est_ref


def _predicted_growth(ctx, n, nxt, prev, walls, cache, ctrl, tols=None, r_last=None):
    """``(growth, how)``: the predicted wall of order ``nxt`` over the
    measured wall of order ``n``: the ratio of the cost proxies of the two
    orders' (truncated) realizations (``_order_cost``), times the proxy's
    error on the step before when there is one (measured wall ratio of
    ``prev -> n`` over its proxy ratio, clamped to [0.5, 4]).  The proxy
    describes the state-space route's sweep of the linear operator's
    realization; it is not used (the earlier rule was: the larger of
    ``(nxt / n)^4`` and the measured growth of the last pair) when there is
    no realization of the next order, when the last run (``r_last``) did
    not take the state-space route for every subset, or when the proxy does
    not grow with the order (the delay sits only in a nonlinear term, e.g.
    ``x * Conv(g, x)`` with a delayed ``g``: the operator does not see it).
    ``tols``: {order: truncation tolerance} (default: the tolerance in
    effect)."""
    from api._realization import truncation_tol
    tols = dict(tols or {})
    tol_of = lambda m: tols.get(m, truncation_tol())         # noqa: E731
    c_n = _order_cost(ctx, n, cache, tol_of(n))
    c_x = _order_cost(ctx, nxt, cache, tol_of(nxt))
    for m, c in ((n, c_n), (nxt, c_x)):
        if c is not None:
            ctrl['cost_by_order'][m] = {k: round(v, 3) for k, v in c.items()}
    eng = (r_last or {}).get('phase_j_engine') or {}
    ss_all = r_last is None or (eng.get('engine') == 'statespace'
                                and float(eng.get('statespace_share') or 0.0) >= 1.0)
    why = None
    if not ss_all:
        why = 'the last order did not run entirely on the state-space route'
    elif c_n and c_x and c_n['proxy'] > 0 and not c_x['proxy'] > c_n['proxy'] * (1 + 1e-9):
        why = 'the realization does not grow with the order (a delay only in a nonlinear term)'
    if why is None and c_n and c_x and c_n['proxy'] > 0:
        g = c_x['proxy'] / c_n['proxy']
        corr = 1.0
        if prev is not None and walls.get(prev, 0) > 0:
            c_p = _order_cost(ctx, prev, cache, tol_of(prev))
            if c_p and c_p['proxy'] > 0:
                corr = min(4.0, max(0.5, (walls[n] / walls[prev])
                                    / (c_n['proxy'] / c_p['proxy'])))
        ctrl['cost_correction'] = round(corr, 3)
        return g * corr, (f'mean per-entry states {c_n["states"]:.1f} -> {c_x["states"]:.1f}, '
                          f'stiffness {c_n["stiffness"]:.1f} -> {c_x["stiffness"]:.1f}'
                          + (f', x {corr:.2f} from the last step' if corr != 1.0 else ''))
    why = why or 'no realization of the next order'
    floor = (nxt / n) ** 4
    if prev is not None and walls.get(prev, 0) > 0 and walls[n] / walls[prev] > floor:
        return walls[n] / walls[prev], f'{why}: the measured growth of the last two orders'
    return floor, f'{why}: the (n_next / n)^4 rule'


def _finish(r, model, call, opts, delays, report_true, ctx, exact, exact_note, runs, walls,
            skipped, failed, ests, est, cap, n_used, incomplete, target_met, loops_need_n,
            t_all, ctrl, approximant='pade'):
    """The record of ``compute_delayed`` (``r['delay']``, the report, the files,
    the warning)."""
    # 4. the record
    tree_vs_pade = None
    if exact is not None:
        tree_vs_pade = r['delay_tree_vs_pade'] if 'delay_tree_vs_pade' in r else None
    info = {
        'delays': {kname: [str(x) for x in ds] for kname, ds in delays.items()},
        # 'pade': (n, n); 'pade_proper': the strictly proper [n-1/n] (a hard
        # delay at a vertex leg: the (n, n) image is not strictly proper)
        'approximant': approximant,
        'n': n_used,
        'orders_run': sorted(runs),
        'orders_skipped_unstable': skipped,
        'orders_failed': failed,
        'incomplete': incomplete,
        'subrun_accuracy_low': bool(getattr(r.get('analysis'), 'has', lambda c: False)(
            'G6_ACCURACY_LOW')),
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
        # the per-entry balanced truncation of every order's
        # realization and the order controller's record
        'truncation_tol': ctrl['truncation_by_order'].get(n_used),
        'truncation_by_order': ctrl['truncation_by_order'],
        'loop_error_pred_by_order': ctrl['loop_error_pred'],
        'cost_by_order': ctrl['cost_by_order'],
        'predicted_walls_s': ctrl['predicted_walls_s'],
        'cost_correction': ctrl['cost_correction'],
        'stop_reason': ctrl['stop'],
        # k >= 3: the time unit of the controller's sample points of total_C
        'sample_time_scale': ctrl.get('sample_time_scale'),
    }
    r['delay'] = info
    r.pop('delay_tree_vs_pade', None)
    cfg = r.get('config')
    if isinstance(cfg, dict):
        cfg['delay_pade_model'] = cfg.get('model_name')
        cfg['model_name'] = model.get('name', cfg.get('model_name'))
    _record_in_report(r, report_true, info, call.get('policy'))
    if call.get('spectra') not in (None, False):
        # the spectral outputs of the final result (exact tree, Pade loops)
        from api import spectra as _sp
        _sp.attach(r, _sp.resolve_spectra_options(call['spectra']))
    if call.get('output_npz') or call.get('output_csv'):
        from api.save import save_npz, save_csv
        if call.get('output_npz'):
            save_npz(r, call['output_npz'])
        if call.get('output_csv'):
            save_csv(r, call['output_csv'])
    if (loops_need_n or incomplete) and target_met is False:
        _ml = int(call.get('max_ell') or 0)
        what = ('the loops' if exact is not None else
                'the tree and the loops' if _ml >= 1 else 'the tree')
        msg = (f'delay: the Pade approximation of {what} reached an '
               f'estimated relative error {est if est is not None else float("nan"):.3g} '
               f'at n = {n_used} (target {opts["tol"]:g}); {cap or "n_max reached"}.')
        from api.save import _ensure_warning_shown
        _ensure_warning_shown(DelayOrderWarning, 'always', message=msg,
                              stacklevel=4)
        warnings.warn(msg, DelayOrderWarning, stacklevel=4)
    return r


def _install_exact_tree(r, exact, call):
    """Replace the tree level of ``r`` by the exact covariance (k = 2)."""
    from api.compute import _tau0_lag
    taus = np.asarray(r.get('tau_grid'), dtype=float)
    by = r.get('C_tau_by_ell') or {}
    pade_tree = by.get(0)
    # tau = 0 as on the pipeline's grid (and in its loops): the Ito left limit,
    # exact (ExactTree.value(0)) or at tau = -_ITO_EPS (DAEDALUS_TAU0=nudge)
    c0 = exact.values(np.where(np.abs(taus) <= 1e-12, _tau0_lag(), taus))
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
    # the exact tree itself, for its exact pairing and spectrum
    # (api.me_output.ExactDelayTreeSide)
    f0.exact_tree = exact
    fby[0] = f0
    r['total_C_by_ell'] = fby
    others = [f for ell, f in fby.items() if ell != 0 and f is not None]

    def total(*t, _f0=f0, _o=tuple(others)):
        return _f0(*t) + sum(f(*t) for f in _o)

    r['total_C'] = total


def _fold_delay_into_g6(_an, findings, diag, info, pol):
    """The sub-run's G6 accuracy estimate covers Phase J of the Pade model
    only; the approximant's own estimated error (``est_rel_error``,
    |C_n - C_2n| relative, 0 when the exact tree makes it irrelevant) is part
    of the result's error too (R = 2e-14 reported next to a
    6.5e-5 Pade tree error).  Adds it to ``accuracy_estimate``, rewrites
    G6_ACCURACY and adds G6_ACCURACY_LOW when the sum exceeds the threshold."""
    est = info.get('est_rel_error')
    if est is None or not est > 0:
        return findings, diag
    R0 = diag.get('accuracy_estimate')
    R = float(est) + (float(R0) if R0 is not None else 0.0)
    thr = float(diag.get('accuracy_threshold', _an._G6_DEFAULT_RTOL))
    diag.update(accuracy_estimate=R, accuracy_estimate_phase_j=R0,
                delay_est_rel_error=float(est))
    out, had_low = [], False
    for f in findings:
        if f.code == 'G6_ACCURACY':
            f = replace(f, message=(
                f'accuracy estimate R = {R:.3g}: the delay approximant\'s '
                f'estimated relative error {est:.3g} (Pade n = {info["n"]}, '
                f'|C_n - C_2n|) plus Phase J\'s ' + f.message.split(': ', 1)[-1]
                .replace('accuracy estimate R = ', 'R = ')))
        had_low = had_low or f.code == 'G6_ACCURACY_LOW'
        out.append(f)
    if R > thr and not had_low:
        low = _an.Finding(
            code='G6_ACCURACY_LOW', level='warning', gate='G6',
            message=(f'the accuracy estimate R = {R:.3g} exceeds the threshold '
                     f'{thr:g}: the delay approximant\'s estimated relative error '
                     f'is {est:.3g} (Pade n = {info["n"]}).'),
            remedy=("raise delay={'n_max': ...} / {'wall_max': ...}, or a looser "
                    'question.accuracy accepts the result'))
        out.append(_an._apply_policy([low], pol)[0])
    return out, diag


def _record_in_report(r, report_true, info, policy=None):
    """The delay record in ``r['analysis']``: the G2-K findings of the true
    model, ``G3_DELAY_APPROXIMATED`` and the ledger's approximation entry."""
    from api import analyze as _an
    rep = r.get('analysis')
    n = info['n']
    names = ', '.join(f'{kn} (d = {", ".join(ds)})' for kn, ds in info['delays'].items())
    est = info['est_rel_error']
    m_ = n - 1 if info.get('approximant') == 'pade_proper' else n
    loop_txt = ('the loops by Pade (%d, %d) of exp(-i omega d)' % (m_, n)
                + (' (strictly proper: a delay at a vertex leg)' if m_ != n else '')
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
        detail=(f'Pade({m_},{n}) of exp(-i omega d) in the loops'
                + (f'; est. rel. error {est:.2g}' if est is not None else '')
                + ('; tree exact' if info['tree'] == 'exact' else '; tree Pade')
                + (f'; realizations balanced-truncated (relative H-inf tolerance '
                   f'{info["truncation_tol"]:.2g} per entry at n = {n})'
                   if info.get('truncation_tol') else '')),
        declared=True, satisfied=info['target_met'])
    if rep is None:
        return
    keep = [f for f in rep.findings
            if not (f.code.startswith('G2') and f.code in {g.code for g in report_true.findings})]
    g2k = [f for f in report_true.findings if f.code.startswith('G2')]
    pol = _an._coerce(policy, _an.Policy, 'policy')
    finding = _an._apply_policy([finding], pol)[0]          # strict: recorded as an error
    keep, diag = _fold_delay_into_g6(_an, keep, dict(rep.diagnostics or {}), info, pol)
    findings = tuple(_an._sorted_findings(keep + g2k + [finding]))
    tt = report_true.traits
    traits = replace(rep.traits, propagator_class='delay-rational', delays=tt.delays,
                     kernel_stability=tt.kernel_stability,
                     n_unstable_roots=tt.n_unstable_roots, stable=tt.stable,
                     stability_source=tt.stability_source or rep.traits.stability_source)
    r['analysis'] = replace(rep, findings=findings, ledger=tuple(rep.ledger) + (entry,),
                            traits=traits, diagnostics=diag)
    r['analysis_true_model'] = report_true
