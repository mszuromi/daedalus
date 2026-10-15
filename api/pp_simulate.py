r"""
Simulator for SDE front-end models with point processes (:meth:`api.sde.SDE.point_process`)
==========================================================================================

Time-discretised (step ``dt``), vectorised over ``n_paths`` independent
paths (numpy; no processes):

* a point process ``n[i]`` with conditional intensity ``rate_i`` emits
  ``Poisson(max(rate_i, 0) * dt)`` spikes in a step (the intensity read at
  the start of the step);
* every ``Conv(K, x)`` in a rate or a drift is the output ``C s`` of an exact
  state-space realization ``(A, B, C)`` of the kernel ``K`` (from its
  frequency image, or the Laplace transform of its ``time_expr``): a spike
  adds ``B`` to the state, a continuous input enters by a zero-order hold,
  and the state is propagated by ``exp(A dt)``; a delay ``d`` delays the
  filter's input by ``round(d / dt)`` steps (ring buffer);
* a continuous field with ``lhs = a*Dt*x + b*x`` takes the Euler step
  ``x += (rhs - b x)/a dt`` (+ its Gaussian sources, ``sqrt(var dt)``), a
  spike train in ``rhs`` contributing its count (a jump).

The scheme is first order in ``dt`` (a spike of step ``k`` reaches the
rates of step ``k + 1`` through ``g(dt)``); halve ``dt`` to see the bias.

``simulate(sde, T=..., dt=..., n_paths=...)`` returns binned spike counts
and sampled continuous fields; ``cross_covariance(rec, ('n', 0), ('n', 1),
lags)`` estimates ``Cov(dn_0(t), dn_1(t + tau)) / dt^2`` (the covariance
density the cumulant pipeline computes, ``k = 2``) with a batch-means
standard error over groups of paths.  ``SDE.simulate(...)`` calls
:func:`simulate`.
"""
from __future__ import annotations

import math
import re

import numpy as np

__all__ = ['simulate', 'cross_covariance', 'mean_rate', 'PPSimulationError']


class PPSimulationError(ValueError):
    """The model has a part this simulator does not handle."""


_NP = {'exp': np.exp, 'log': np.log, 'sqrt': np.sqrt, 'sin': np.sin, 'cos': np.cos,
       'tan': np.tan, 'tanh': np.tanh, 'sinh': np.sinh, 'cosh': np.cosh,
       'arctan': np.arctan, 'erf': None, 'abs': np.abs, 'pi': math.pi, 'e': math.e,
       'heaviside': lambda x: np.heaviside(x, 1.0)}
try:                                                        # optional
    from scipy.special import erf as _erf
    _NP['erf'] = _erf
except Exception:                                           # noqa: BLE001
    _NP.pop('erf')


def _py(text: str) -> str:
    return str(text).replace('^', '**')


class _Ref(np.ndarray):
    """A field value that remembers which field element it is (for Conv)."""

    def __array_finalize__(self, obj):
        self.ref = getattr(obj, 'ref', None)


def _tag(arr, ref):
    v = np.asarray(arr, dtype=float).view(_Ref)
    v.ref = ref
    return v


class _Table:
    def __init__(self, fn):
        self._fn = fn

    def __getitem__(self, idx):
        return self._fn(idx if isinstance(idx, tuple) else (idx,))


# ── kernels: exact state-space realizations ───────────────────────────

def _numeric_text(text, values, idx):
    """Substitute parameter values (indexed by the kernel's local i, j)."""
    loc = dict(zip(('i', 'j'), idx))

    def sub_indexed(m):
        name = m.group(1)
        if name not in values:
            return m.group(0)
        ix = tuple(int(eval(q.strip(), {}, loc)) for q in m.group(2).split(','))
        return repr(float(np.asarray(values[name])[ix]))
    text = re.sub(r'\b([A-Za-z_]\w*)\s*\[([^\]]*)\]', sub_indexed, str(text))

    def sub_bare(m):
        name = m.group(1)
        if name in values and np.ndim(values[name]) == 0:
            return repr(float(values[name]))
        return name
    return re.sub(r'\b([A-Za-z_]\w*)\b', sub_bare, text)


def _realize(kspec, values, idx, name):
    """``(A, B, C, D, delay)`` of kernel ``name`` at index ``idx``."""
    from sage.all import SR, I, sage_eval, laplace, heaviside, dirac_delta
    s = SR.var('s_')
    if kspec.get('freq_image') is not None:
        txt = _numeric_text(kspec['freq_image'], values, idx)
        om = SR.var('omega')
        H = SR(sage_eval(txt, locals={'omega': om, 'I': I})).subs({om: -I * s})
    else:
        txt = _numeric_text(kspec['time_expr'], values, idx)
        t = SR.var('t')
        g = SR(sage_eval(txt, locals={'t': t, 'heaviside': heaviside,
                                      'dirac_delta': dirac_delta}))
        g = g.subs({heaviside(t): 1})
        H = laplace(g, t, s)
    H = H.simplify_rational()
    num, den = H.numerator(), H.denominator()
    try:
        cn = [complex(c) for c in num.expand().coefficients(s, sparse=False)]
        cd = [complex(c) for c in den.expand().coefficients(s, sparse=False)]
    except TypeError as exc:
        raise PPSimulationError(f'kernel {name}{list(idx)}: the image is not rational '
                                f'({H}); the simulator needs a finite realization.') from exc
    if max(abs(c.imag) for c in cn + cd) > 1e-12 * max(abs(c) for c in cn + cd):
        raise PPSimulationError(f'kernel {name}{list(idx)}: complex coefficients.')
    cn = np.array([c.real for c in cn])
    cd = np.array([c.real for c in cd])
    m = len(cd) - 1
    if m < 0 or cd[-1] == 0:
        raise PPSimulationError(f'kernel {name}{list(idx)}: degenerate image {H}.')
    cn, cd = cn / cd[-1], cd / cd[-1]
    if len(cn) > m + 1:
        raise PPSimulationError(f'kernel {name}{list(idx)}: improper image {H}.')
    cn = np.concatenate([cn, np.zeros(m + 1 - len(cn))])
    D = cn[m] if m >= 0 else 0.0
    b = cn[:m] - D * cd[:m]                     # strictly proper part
    A = np.zeros((m, m))
    if m:
        A[:-1, 1:] = np.eye(m - 1)
        A[-1, :] = -cd[:m]
    B = np.zeros(m)
    if m:
        B[-1] = 1.0
    if m and np.max(np.linalg.eigvals(A).real) >= 0:
        raise PPSimulationError(f'kernel {name}{list(idx)}: not a stable (decaying) kernel.')
    delay = 0.0
    if kspec.get('delay'):
        delay = float(eval(_py(_numeric_text(kspec['delay'], values, idx)), dict(_NP)))
    return A, B, b, float(D), delay


class _Filter:
    def __init__(self, A, B, C, D, lag, dt, n_paths, spike):
        from scipy.linalg import expm
        m = A.shape[0]
        self.m, self.C, self.D, self.spike = m, C, D, spike
        self.Ad = expm(A * dt) if m else np.zeros((0, 0))
        if m and not spike:
            M = np.zeros((m + 1, m + 1))
            M[:m, :m], M[:m, m] = A * dt, B * dt
            self.Bd = expm(M)[:m, m]
        self.B = B
        self.x = np.zeros((m, n_paths))
        self.lag = int(lag)
        self.buf = np.zeros((max(self.lag, 1), n_paths))
        self.bp = 0
        self.direct = np.zeros(n_paths)

    def output(self):
        y = self.C @ self.x if self.m else np.zeros(self.x.shape[1])
        return y + self.direct

    def push(self, u):
        """``u``: this step's input (spike counts, or the field value)."""
        if self.lag:
            u, self.buf[self.bp] = self.buf[self.bp].copy(), u
            self.bp = (self.bp + 1) % self.lag
        if self.spike:
            if self.D:
                raise PPSimulationError('a kernel with a delta part acting on a spike '
                                        'train (an instantaneous rate jump) is not simulated.')
            if self.m:
                self.x = self.Ad @ (self.x + np.outer(self.B, u))
        else:
            self.direct = self.D * u
            if self.m:
                self.x = self.Ad @ self.x + np.outer(self.Bd, u)


# ── the simulator ─────────────────────────────────────────────────────

def _param_values(sde, params):
    vals = {}
    for (name,), kw in sde._params:
        v = (params or {}).get(name, kw.get('default'))
        if v is None:
            raise PPSimulationError(f'parameter {name!r} has no value; pass params={{...}}.')
        vals[name] = np.asarray(v, dtype=float) if not np.isscalar(v) else float(v)
    for k in (params or {}):
        if k not in vals:
            raise PPSimulationError(f'params: unknown parameter {k!r}.')
    return vals


def simulate(sde, *, T, dt, n_paths, burn_in=50.0, bin=None, params=None, seed=0,
             record=None):
    """Simulate ``sde`` (an :class:`api.sde.SDE` with point processes, drifts and
    Gaussian sources).  Returns ``{'counts': {name: (n_bins, size, n_paths)},
    'fields': {name: (n_bins, size, n_paths)}, 'bin': bin, 'dt': dt, ...}``:
    spike counts per bin of width ``bin`` (default ``dt``) and continuous fields
    sampled at the bin ends, after ``burn_in``."""
    from api.noise import expr_to_text
    from api.sde import SDEError
    sde._derive()                                   # validates; declares composites
    rng = np.random.default_rng(seed)
    vals = _param_values(sde, params)
    pops = {p['name']: p['size'] for p in sde._populations()}
    nsteps, nburn = int(round(T / dt)), int(round(burn_in / dt))
    bsteps = max(int(round((bin or dt) / dt)), 1)
    nb = nsteps // bsteps
    spikes = [f for f, info in sde._fields.items() if info['exposed'] is not None]
    for f in spikes:
        if not sde._sources[sde._fields[f]['exposed']].get('pp'):
            raise PPSimulationError(f'field {f!r}: only point_process() spike trains are '
                                    f'simulated (exposed {type(sde._sources[f]["source"]).__name__} '
                                    f'source).')
    conts = [f for f in sde._fields if f not in spikes]
    size = {f: pops[sde._pop_of_field(f)] for f in sde._fields}
    state = {f: np.zeros((size[f], n_paths)) for f in conts}
    counts_now = {f: np.zeros((size[f], n_paths)) for f in spikes}
    filters, order = {}, []
    registering = [True]

    def field_table(f):
        def get(idx):
            j = int(idx[0])
            if f in spikes:
                return _tag(counts_now[f][j] / dt, (f, j))
            return _tag(state[f][j], (f, j))
        return _Table(get)

    def kernel_table(k):
        return _Table(lambda idx: ('kernel', k, tuple(int(q) for q in idx)))

    def conv(kref, x):
        if not (isinstance(kref, tuple) and kref[:1] == ('kernel',)):
            raise PPSimulationError(f'Conv: first argument {kref!r} is not a kernel.')
        ref = getattr(x, 'ref', None)
        if ref is None:
            raise PPSimulationError('Conv(K, x): x must be a field element (x[j]).')
        key = (kref[1], kref[2], ref)
        if key not in filters:
            if not registering[0]:
                raise PPSimulationError(f'Conv{key}: input not seen at set-up.')
            A, B, C, D, d = _realize(sde._kernels[kref[1]], vals, kref[2], kref[1])
            lag = int(round(d / dt))
            if d and abs(lag * dt - d) > 1e-9 * max(d, 1):
                raise PPSimulationError(f'kernel {kref[1]}: delay {d} is not a multiple of '
                                        f'dt={dt}.')
            filters[key] = _Filter(A, B, C, D, lag, dt, n_paths, ref[0] in spikes)
            order.append(key)
        return filters[key].output()

    def func_table(fname, args, expr):
        txt = sde._rename_pop(expr)
        for pn in sde._vector_params():
            txt = re.sub(rf'\b{re.escape(pn)}\b(?!\s*[\[\(])', f'{pn}[i]', txt)
        code = compile(_py(txt), f'<function {fname}>', 'eval')

        def get(idx):
            i = int(idx[0])

            def call(*a):
                env = dict(base)
                env['i'] = i
                env.update(zip(args, a))
                return eval(code, env)
            return call
        return _Table(get)

    base = dict(_NP)
    base.update({'Conv': conv, 'sum': sum, 'I': 1j})
    base.update({p: range(n) for p, n in pops.items()})
    base['pop'] = range(pops.get('pop', 1))
    for name, v in vals.items():
        base[name] = v
    for f in sde._fields:
        base[f] = field_table(f)
    for k in sde._kernels:
        base[k] = kernel_table(k)
    for (fname, args, expr), _kw in sde._functions:
        base[fname] = func_table(fname, args, expr)

    def compiled(text, what):
        return compile(_py(text), what, 'eval')

    # rates of the point processes
    rate_code = {}
    for f in spikes:
        pp = sde._sources[sde._fields[f]['exposed']]['pp']
        rate_code[f] = (compiled(sde._index(sde._compose_kernels(pp['rate'], f)), f'<rate {f}>'),
                        compiled(sde._index(pp['size']), f'<size {f}>'))
    # continuous fields: lhs = a*Dt*x + b*x, rhs (+ Gaussian sources)
    from sage.all import SR
    from api.sde import _DT
    from api.noise import parse_expr
    drift = {}
    for f in conts:
        eq = sde._eqs[f]
        L = parse_expr(sde._index(eq['lhs'])).expand()
        x = sde._sym(f)
        a = L.coefficient(_DT, 1).coefficient(x, 1)
        rest0 = L.subs({_DT: 0})
        b = rest0.coefficient(x, 1)
        if (L - a * _DT * x - b * x).expand() != 0 or set(map(str, (a.variables() + b.variables()))) \
                & {g + '__i' for g in sde._fields}:
            raise PPSimulationError(f'equation({f!r}): lhs {eq["lhs"]!r} is not '
                                    f'a*Dt*{f} + b*{f} with constant a, b.')
        drift[f] = (compiled(expr_to_text(a), f'<a {f}>'), compiled(expr_to_text(b), f'<b {f}>'),
                    compiled(sde._index(eq['rhs']), f'<rhs {f}>'), [])
    for name, s in sde._sources.items():
        if s.get('pp'):
            continue
        src = s['source']
        if s['expose'] is not None or not src.is_gaussian:
            raise PPSimulationError(f'source {name!r} ({type(src).__name__}): the simulator '
                                    f'handles point_process() trains and Gaussian sources '
                                    f'coupled to fields.')
        if sde._interp != 'ito':
            raise PPSimulationError('simulate: Ito interpretation only.')
        mean = compiled(sde._index(expr_to_text(src.params['mean'])), f'<mean {name}>')
        var = compiled(sde._index(expr_to_text(src.params['var'])), f'<var {name}>')
        for f, c in s['couples'].items():
            drift[f][3].append((compiled(sde._index(str(c)), f'<c {name}>'), mean, var, name))

    def ev(code, i):
        env = dict(base)
        env['i'] = i
        return eval(code, env)

    # set-up pass: evaluate everything once to register the filters
    for f in spikes:
        for i in range(size[f]):
            ev(rate_code[f][0], i)
    for f in conts:
        for i in range(size[f]):
            ev(drift[f][2], i)
    registering[0] = False
    coef = {f: [(float(np.real(ev(drift[f][0], i))), float(np.real(ev(drift[f][1], i))))
                for i in range(size[f])] for f in conts}
    for f in conts:
        if any(a == 0 for a, _ in coef[f]):
            raise PPSimulationError(f'equation({f!r}): zero Dt coefficient.')

    out_counts = {f: np.zeros((nb, size[f], n_paths), dtype=np.int32) for f in spikes}
    out_fields = {f: np.zeros((nb, size[f], n_paths)) for f in (record or conts)}
    rate_sum = {f: np.zeros(size[f]) for f in spikes}
    for k in range(nburn + nsteps):
        # rates from the current state, then this step's spikes
        new = {}
        for f in spikes:
            c = np.empty((size[f], n_paths))
            for i in range(size[f]):
                lam = np.maximum(np.real(ev(rate_code[f][0], i)), 0.0)
                lam = np.broadcast_to(lam, (n_paths,))
                c[i] = rng.poisson(lam * dt) * float(np.real(ev(rate_code[f][1], i)))
                if k >= nburn:
                    rate_sum[f][i] += lam.mean()
            new[f] = c
        # continuous fields (Euler), reading the old state and this step's spikes
        for f in spikes:
            counts_now[f] = new[f]
        upd = {}
        for f in conts:
            x = state[f]
            nx = np.empty_like(x)
            for i in range(size[f]):
                a, b = coef[f][i]
                r = np.real(ev(drift[f][2], i))
                dx = (r - b * x[i]) / a * dt
                for (cc, mean, var, _nm) in drift[f][3]:
                    cv = np.real(ev(cc, i))
                    dx = dx + cv * (np.real(ev(mean, i)) * dt + np.sqrt(
                        np.maximum(np.real(ev(var, i)), 0.0) * dt) * rng.standard_normal(n_paths)) / a
                nx[i] = x[i] + dx
            upd[f] = nx
        # filters take this step's inputs (the field values before the update)
        for key in order:
            src_f, j = key[2]
            filters[key].push(new[src_f][j] if src_f in spikes else state[src_f][j])
        state.update(upd)
        for f in spikes:
            counts_now[f] = np.zeros_like(new[f])
        if k >= nburn:
            b = (k - nburn) // bsteps
            if b < nb:
                for f in spikes:
                    out_counts[f][b] += new[f].astype(np.int32)
                if (k - nburn) % bsteps == bsteps - 1:
                    for f in out_fields:
                        out_fields[f][b] = state[f]
    return {'counts': out_counts, 'fields': out_fields, 'bin': bsteps * dt, 'dt': dt,
            'n_paths': n_paths, 'T': nb * bsteps * dt,
            'mean_intensity': {f: rate_sum[f] / max(nsteps, 1) for f in spikes}}


def mean_rate(rec, f, i, n_batches=20):
    """Mean spike rate of ``f[i]`` and its batch-means standard error."""
    c = rec['counts'][f][:, i, :]
    per_path = c.sum(0) / rec['T']
    groups = np.array_split(per_path, min(n_batches, len(per_path)))
    gm = np.array([g.mean() for g in groups])
    return per_path.mean(), gm.std(ddof=1) / math.sqrt(len(gm))


def cross_covariance(rec, a, b, lags, n_batches=20):
    """``Cov(dn_a(t), dn_b(t + tau)) / dt^2`` at ``tau = lag * bin`` (``a``, ``b`` =
    ``(field, index)``) from the binned counts, the grand mean removed; returns
    ``(cov, se)``.  For ``a == b`` the lag-0 value includes the spikes'
    own (delta) variance ``rate / bin``."""
    xa = rec['counts'][a[0]][:, a[1], :].astype(float)
    xb = rec['counts'][b[0]][:, b[1], :].astype(float)
    xa -= xa.mean()
    xb -= xb.mean()
    nb, n_paths = xa.shape
    per_path = []
    for p in range(n_paths):
        row = []
        for l in lags:
            l = int(l)
            if l >= 0:
                row.append(np.dot(xa[:nb - l, p], xb[l:, p]) / (nb - l))
            else:
                row.append(np.dot(xa[-l:, p], xb[:nb + l, p]) / (nb + l))
        per_path.append(row)
    per_path = np.array(per_path) / rec['bin'] ** 2
    groups = np.array_split(per_path, min(n_batches, n_paths))
    gm = np.array([g.mean(0) for g in groups])
    return per_path.mean(0), gm.std(0, ddof=1) / math.sqrt(len(gm))
