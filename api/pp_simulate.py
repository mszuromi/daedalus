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
  ``x += (rhs - b x)/a dt`` (+ its Gaussian sources, ``sqrt(var dt)``, with
  the Ito drift correction under ``'stratonovich'``/``'marcus'``), a spike
  train in ``rhs`` contributing its count (a jump);
* a jump source coupled to fields (``noise(..., couples=...)``) draws its
  increment ``dL`` over the step (rates read at the start of the step):
  Ito moves the state by ``c(x) dL / a``; Marcus moves it along the flow of
  ``c/a`` for the time ``dL`` (exact for an affine coupling, a DOP853
  integration otherwise; the flows of one source compose, so the summed
  increment is the exact jump of all the step's jumps).

The scheme is first order in ``dt`` (a spike of step ``k`` reaches the
rates of step ``k + 1`` through ``g(dt)``; a delta part of a kernel on a
continuous input acts one step late); halve ``dt`` to see the bias.  Counts
are floats (a spike adds its ``size``).

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
from engine.errors import DaedalusError

__all__ = ['simulate', 'cross_covariance', 'mean_rate', 'PPSimulationError']


class PPSimulationError(DaedalusError, ValueError):
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
        # not inherited: an expression of a field (2*x, exp(x)) is not the field
        self.ref = None


def _tag(arr, ref):
    v = np.asarray(arr, dtype=float).view(_Ref)
    v.ref = ref
    return v


class _ImagPopulation(complex):
    """``I`` when a population is named ``I``: the imaginary unit in
    arithmetic, the population's index range in ``for j in I``."""

    def __new__(cls, n):
        obj = super().__new__(cls, 0.0, 1.0)
        obj._n = int(n)
        return obj

    def __iter__(self):
        return iter(range(self._n))

    def __len__(self):
        return self._n


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
        return f'({float(np.asarray(values[name])[ix])!r})'
    text = re.sub(r'\b([A-Za-z_]\w*)\s*\[([^\]]*)\]', sub_indexed, str(text))

    def sub_bare(m):
        name = m.group(1)
        if name in values and np.ndim(values[name]) == 0:
            return f'({float(values[name])!r})'
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
        if spike and D:
            raise PPSimulationError('a kernel with a delta part acting on a spike train (an '
                                    'instantaneous rate jump, e.g. a bare hard delay) is not '
                                    'simulated; give the delayed kernel a smooth image.')

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

def _as_seed(seed):
    """A seed for ``np.random.default_rng``: an integral number (also a Sage
    ``Integer``) as ``int``; ``None``, sequences, ``SeedSequence`` and
    generators as given."""
    if seed is None or isinstance(seed, (np.random.Generator,
                                         np.random.SeedSequence,
                                         np.random.BitGenerator)):
        return seed
    if hasattr(seed, '__index__') and np.ndim(seed) == 0:
        return int(seed.__index__())
    return seed


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
             record=None, max_bytes=None):
    """Simulate ``sde`` (an :class:`api.sde.SDE` with point processes, drifts and
    Gaussian sources).  Returns ``{'counts': {name: (n_bins, size, n_paths)},
    'fields': {name: (n_bins, size, n_paths)}, 'bin': bin, 'dt': dt, ...}``:
    spike counts per bin of width ``bin`` (default ``dt``) and continuous fields
    sampled at the bin ends, after ``burn_in``.  ``max_bytes``: the largest output
    allowed (default: ``PHASE_J_MEM_BUDGET_GB``, else the memory guards' default
    budget, ``default_mem_budget``; ``off`` there means no limit);
    a larger one raises ``MemoryError`` before anything is allocated."""
    from api.noise import expr_to_text
    from api.sde import SDEError
    sde._derive()                                   # validates; declares composites
    # Sage-preparsed literals (a notebook's Integer / RealNumber) as Python
    # numbers: numpy's seeding and round() do not take them
    T, dt, burn_in = float(T), float(dt), float(burn_in)
    bin = None if bin is None else float(bin)
    n_paths = int(n_paths)
    rng = np.random.default_rng(_as_seed(seed))
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
            raise PPSimulationError('Conv(K, x): x must be a field element (x[j]); filter '
                                    'an expression by declaring it as a field.')
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
    # a population named I (excitatory / inhibitory networks) must not hide
    # the imaginary unit: it iterates as the population and computes as 1j
    base.update({p: (_ImagPopulation(n) if p == 'I' else range(n))
                 for p, n in pops.items()})
    base['pop'] = range(pops.get('pop', 1))
    for name, v in vals.items():
        base[name] = v
    for f in sde._fields:
        base[f] = field_table(f)
    for k, spec in sde._kernels.items():
        indexed = spec.get('indexed_by') or spec.get('indexed')
        base[k] = kernel_table(k) if indexed else ('kernel', k, ())
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
        rhs = sde._index(eq['rhs'])
        corrs = sde._ito_drift.get(f, [])         # Stratonovich / Marcus: Ito drift correction
        if corrs:
            rhs = f'({rhs}) + ' + ' + '.join(f'({expr_to_text(c)})' for c in corrs)
        drift[f] = (compiled(expr_to_text(a), f'<a {f}>'), compiled(expr_to_text(b), f'<b {f}>'),
                    compiled(rhs, f'<rhs {f}>'), [])
    jumps = []                                     # (name, source, couples, flow or None)
    for name, s in sde._sources.items():
        if s.get('pp'):
            continue
        src = s['source']
        if s['expose'] is not None:
            raise PPSimulationError(f'source {name!r} ({type(src).__name__}): the simulator '
                                    f'handles point_process() trains and sources coupled '
                                    f'to fields (not exposed ones).')
        if sde._interp == 'marcus':
            coup = {f: sde._parse(str(c), f'coefficient of {name} in {f}')
                    for f, c in s['couples'].items()}
            units = sde._marcus_units(name, s, coup)
        else:
            units = [(name, src, False)]
        for uname, usrc, series in units:
            if usrc.is_gaussian:
                mean = compiled(sde._index(expr_to_text(usrc.params['mean'])), f'<mean {uname}>')
                var = compiled(sde._index(expr_to_text(usrc.params['var'])), f'<var {uname}>')
                for f, c in s['couples'].items():
                    drift[f][3].append((compiled(sde._index(str(c)), f'<c {uname}>'), mean, var,
                                        uname))
            else:
                # Ito: the jump moves the state by c(x_pre) dL; Marcus with a
                # constant coupling: the same translation; with a state-
                # dependent coupling: the flow of c/a for time dL
                cc = {f: compiled(sde._index(str(c)), f'<c {uname}>')
                      for f, c in s['couples'].items()}
                jumps.append((uname, usrc, cc,
                              _flow_spec(sde, s['couples'], compiled) if series else None))

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

    bad = [f for f in (record or []) if f not in conts]
    if bad:
        raise PPSimulationError(f'record={bad}: only continuous fields are sampled (spike '
                                f'trains are always recorded as counts).')
    # the output is 8 bytes per bin, unit and path of every recorded field:
    # refuse before allocating it (T=1e4, dt=1e-3, 100 paths is 8 GB)
    out_bytes = 8 * nb * n_paths * sum(size[f] for f in [*spikes, *(record or conts)])
    if max_bytes is None:
        # the Phase J budget rule: PHASE_J_MEM_BUDGET_GB, else the default
        # (``off`` / ``none`` / ``inf`` / ``unlimited``: no limit)
        from engine.integration.time_domain.final_integral import mem_budget
        max_bytes, rule = mem_budget('PHASE_J_MEM_BUDGET_GB')
    else:
        rule = f'max_bytes={float(max_bytes):g}'
    if max_bytes is not None and out_bytes > max_bytes:
        raise MemoryError(
            f'simulate: the output would hold {out_bytes / 1e9:.2f} GB ({nb:,} bins x '
            f'{n_paths} paths x {out_bytes // (8 * nb * n_paths)} recorded units, 8 bytes each), '
            f'over {max_bytes / 1e9:.2f} GB ({rule}); use a coarser bin= (counts are summed '
            f'per bin), fewer paths or a shorter T, record= fewer fields, or pass max_bytes=.')
    out_counts = {f: np.zeros((nb, size[f], n_paths)) for f in spikes}
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
                _check_runaway(f, i, lam, dt, k, nburn)
                c[i] = rng.poisson(lam * dt) * float(np.real(ev(rate_code[f][1], i)))
                if k >= nburn:
                    rate_sum[f][i] += lam.mean()
            new[f] = c
        # continuous fields (Euler), reading the old state and this step's spikes
        for f in spikes:
            counts_now[f] = new[f]
        upd = {}
        xi = {}                                     # one draw per (source, i): shared by its fields
        dL = {}                                     # jump-source increments, pre-step values
        for (uname, usrc, cc, flow) in jumps:
            for i in range(size[next(iter(cc))]):
                dL[(uname, i)] = np.broadcast_to(np.asarray(
                    usrc.sample(dt, n_paths, rng, _unit_values(vals, state, i)), dtype=float),
                    (n_paths,))
        for f in conts:
            x = state[f]
            nx = np.empty_like(x)
            for i in range(size[f]):
                a, b = coef[f][i]
                r = np.real(ev(drift[f][2], i))
                dx = (r - b * x[i]) / a * dt
                for (cc, mean, var, nm) in drift[f][3]:
                    cv = np.real(ev(cc, i))
                    if (nm, i) not in xi:
                        xi[(nm, i)] = rng.standard_normal(n_paths)
                    dx = dx + cv * (np.real(ev(mean, i)) * dt + np.sqrt(
                        np.maximum(np.real(ev(var, i)), 0.0) * dt) * xi[(nm, i)]) / a
                for (uname, usrc, cc, flow) in jumps:
                    if flow is None and f in cc:
                        dx = dx + np.real(ev(cc[f], i)) * dL[(uname, i)] / a
                nx[i] = x[i] + dx
            upd[f] = nx
        # filters take this step's inputs (the field values before the update)
        for key in order:
            src_f, j = key[2]
            filters[key].push(new[src_f][j] if src_f in spikes else state[src_f][j])
        state.update(upd)
        for (uname, usrc, cc, flow) in jumps:
            if flow is not None:
                for i in range(size[next(iter(cc))]):
                    _marcus_jump(flow, cc, coef, i, dL[(uname, i)], state, ev, n_paths)
        for f in spikes:
            counts_now[f] = np.zeros_like(new[f])
        if k >= nburn:
            b = (k - nburn) // bsteps
            if b < nb:
                for f in spikes:
                    out_counts[f][b] += new[f]
                if (k - nburn) % bsteps == bsteps - 1:
                    for f in out_fields:
                        out_fields[f][b] = state[f]
    return {'counts': out_counts, 'fields': out_fields, 'bin': bsteps * dt, 'dt': dt,
            'n_paths': n_paths, 'T': nb * bsteps * dt,
            'mean_intensity': {f: rate_sum[f] / max(nsteps, 1) for f in spikes}}


def _unit_values(vals, state, i):
    """Names a source's sampler may read for unit ``i``: the parameters (a
    vector parameter's entry ``i``) and the continuous fields' current values."""
    out = {}
    for k, v in vals.items():
        out[k] = v[i] if np.ndim(v) == 1 and len(v) > i else v
    for f, x in state.items():
        if len(x) > i:
            out[f] = x[i]
    return out


def _flow_spec(sde, couples, compiled):
    """The Marcus flow of a coupling ``{field: c}``: its fields and, when
    ``c`` is affine in the fields with a constant Jacobian, the compiled
    Jacobian entries ``d c_f / d x_g`` (closed-form flow), else ``None``
    (the flow is integrated)."""
    from api.noise import expr_to_text
    fl = list(couples)
    cs = {f: sde._parse(str(c), f'coefficient of {f}') for f, c in couples.items()}
    J = [[cs[f].diff(sde._sym(g)) for g in fl] for f in fl]
    if any(sde._field_dependent(e) for row in J for e in row):
        return {'fields': fl, 'A': None}
    return {'fields': fl, 'A': [[compiled(expr_to_text(e), f'<dc {f}/d{g}>')
                                 for g, e in zip(fl, row)] for f, row in zip(fl, J)]}


#: the Marcus flow's ODE fallback (non-affine couplings): DOP853 tolerances
MARCUS_FLOW_RTOL, MARCUS_FLOW_ATOL = 1e-11, 1e-13


def _marcus_jump(flow, cc, coef, i, z, state, ev, n_paths):
    """Move unit ``i`` of the coupled fields along the flow of ``b = c/a`` for
    the time ``z`` (per path): ``x -> x + psi_z(A) b(x)``, ``psi_z(lambda) =
    expm1(z lambda)/lambda``, for an affine ``b = A x + b0`` (exact; the
    eigenvectors of ``A`` must be well conditioned), else a DOP853 integration
    of ``dx/ds = z b(x)`` over ``s in [0, 1]``.  The flows of one source
    compose (``exp(z1 G) exp(z2 G) = exp((z1 + z2) G)``), so the step's
    summed increment is the exact Marcus jump of all its jumps."""
    mask = z != 0
    if not mask.any():
        return
    fl = flow['fields']
    n = len(fl)
    av = [coef[f][i][0] for f in fl]
    zz = z[mask]

    def bvec():
        return np.array([np.broadcast_to(np.real(ev(cc[f], i)), (n_paths,)) / av[r]
                         for r, f in enumerate(fl)])[:, mask]

    if flow['A'] is not None:
        A = np.array([[float(np.real(ev(flow['A'][r][q], i))) / av[r] for q in range(n)]
                      for r in range(n)])
        lam, V = np.linalg.eig(A)
        if np.linalg.cond(V) < 1e8:
            W = np.linalg.solve(V, bvec().astype(complex))
            L = lam[:, None] * zz[None, :]
            small = np.abs(L) < 1e-8
            safe = np.where(lam == 0, 1.0, lam)[:, None]
            psi = np.where(small, zz[None, :] * (1 + L / 2 + L * L / 6),
                           np.expm1(np.where(small, 0, L)) / safe)
            dx = np.real(V @ (psi * W))
            for r, f in enumerate(fl):
                state[f][i][mask] += dx[r]
            return
    from scipy.integrate import solve_ivp

    def rhs(_s, y):
        Y = y.reshape(n, -1)
        for r, f in enumerate(fl):
            state[f][i][mask] = Y[r]
        return (bvec() * zz[None, :]).ravel()

    y0 = np.concatenate([state[f][i][mask] for f in fl])
    sol = solve_ivp(rhs, (0.0, 1.0), y0, method='DOP853', rtol=MARCUS_FLOW_RTOL,
                    atol=MARCUS_FLOW_ATOL)
    if not sol.success:
        raise PPSimulationError(f'Marcus jump flow of {fl}: {sol.message} (the flow of '
                                f'the coupling may leave every bounded set within the jump).')
    Y = sol.y[:, -1].reshape(n, -1)
    for r, f in enumerate(fl):
        state[f][i][mask] = Y[r]


# An intensity times dt above this is no model of anything: the rate feedback
# has run away (numpy's own limit, 9e18, then fails with "lam value too large"
# from deep inside the sampler).
_RUNAWAY_MEAN = 1e12


def _check_runaway(f, i, lam, dt, k, nburn):
    """Refuse a diverged intensity with a message naming the cause."""
    m = float(np.max(lam)) * dt
    if np.isfinite(m) and m <= _RUNAWAY_MEAN:
        return
    when = ('during the burn-in' if k < nburn
            else f'at step {k - nburn} of the run')
    raise PPSimulationError(
        f'simulate: the intensity of {f}[{i}] diverged {when} (mean spikes '
        f'per step {m:g}): the rate feedback is unstable on this path '
        f'(supercritical self-excitation or an exponentially growing rate '
        f'function).  Reduce the coupling / the horizon T, or check the '
        f'rate function; a different seed only moves WHEN a supercritical '
        f'model diverges.')


def _batch_se(per_unit, n_batches, what):
    """Batch-means standard error of the mean over the axis-0 units; needs at
    least two units (one path has no spread, and ``std(ddof=1)`` of one batch
    is NaN).  Returns ``(mean, se)``."""
    n = len(per_unit)
    if n < 2:
        raise ValueError(
            f'{what}: a standard error needs at least 2 independent paths '
            f'(got {n}); run simulate(..., n_paths>=2).')
    groups = np.array_split(per_unit, max(2, min(int(n_batches), n)))
    gm = np.array([g.mean(0) for g in groups])
    return per_unit.mean(0), gm.std(0, ddof=1) / math.sqrt(len(gm))


def mean_rate(rec, f, i, n_batches=20):
    """Mean spike rate of ``f[i]`` and its batch-means standard error
    (needs at least 2 paths)."""
    c = rec['counts'][f][:, i, :]
    per_path = c.sum(0) / rec['T']
    return _batch_se(per_path, n_batches, 'mean_rate')


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
    return _batch_se(per_path, n_batches, 'cross_covariance')
