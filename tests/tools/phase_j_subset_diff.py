"""
tests/tools/phase_j_subset_diff.py
==================================
Per-subset differential harness for Phase J (``docs/integration_speedup_plan.md``
§4.1, milestone M0.1).  Importable API + CLI.  Built on the official,
call-time ``final_integral._SUBSET_HOOK`` (no exec'd module copy).

What it does
------------
1. **Record.**  Runs ``api.compute_cumulants`` with ``_SUBSET_HOOK`` installed
   and ``parallel=False`` (a hook set in this process never sees calls made in
   forked ``total_C_batch`` workers).  Every (diagram build, δ-subset,
   free_ext_vals, Wick permutation) evaluation of the per-diagram dispatch
   (``integrate_diagram``) and of the grouped dispatch
   (``integrate_grouped_diagram``) becomes one record: m, constraint rows,
   free_ext_vals, which path answered (``m0``/``m1``/``polygon``/``poset``/
   ``nquad``; ``plan``/``noplan``/``grouped`` branch; ``per_diagram``/
   ``grouped`` source), the fine-grained bail reason when an analytic path
   returned ``None``, the value, prefactor, pole/residue summary, and the Wick
   context (external times, perm index, compensation).
2. **Reconstruction check.**  Σ value / compensation over the records of the
   first ``contribution()`` call of each (diagram build, external-time point),
   grouped by (loop order, point), must reproduce the pipeline's C_ell at that
   point to ``RECON_RTOL`` = 1e-10 (plus a summation-order slack of
   1e-13 · Σ|terms| and 1e-14 absolute).  Each record is weighted by
   1/_compensation and summed over Wick perms.
3. **References** on the SAME closure (the callable production hands to
   scipy.nquad) and constraints, once per unique (diagram build, subset,
   free_ext_vals, tie context) -- the record's ``ref_key``.  The tie
   context is part of it because at an exact external-time tie several
   Wick permutations evaluate the same subset at the same free values with
   DIFFERENT tie contexts (other legs behind the free values), and the
   constant-row verdicts legitimately differ between them; every such
   orientation gets its own reference and its own rule cross-check, and
   the disagreement table and the attribution compare each record with
   the reference of its own tie context:

   ``a``    tight iterated quadrature (``scipy.integrate.quad_vec`` per
            level, epsabs = 1e-13 · scale, epsrel = 1e-10) on the ±200 box.
            Exact nested bounds; breakpoints at the external times (0 and
            free_ext_vals) and at every vertex projection of the slice
            polytope (this covers the chain top).  ``scale`` =
            |prefactor| · Π_e Σ_k |C_e,k| (per-diagram) or Σ_α |B_α|
            (grouped), i.e. the integrand bound for Δt ≥ 0; 1.0 when the
            subset has no pole data (non-rational kernels).
   ``b``    the same on the ±12 box.
   ``c``    m=2 only: 50-digit mpmath fan formula (anchored at the
            max-exponent vertex, adaptive extra precision in the
            divided difference) on box(150) -> ``c150`` and box(400) ->
            ``c400``.  Needs pole data.
   ``d``    DBM exact route at L = -3000 -- **NOT AVAILABLE until M3**:
            ``dbm_reference`` raises ``NotImplementedError``.
   ``e``    ``certify_box`` a-priori truncation bound S·Q(m, T) for the box
            actually used -- **NOT AVAILABLE until M8**:
            ``certify_box_reference`` raises ``NotImplementedError``.

   All references decide constant (zero-normal) rows by the 0.2.0 rule
   (CHANGELOG.md; plan Appendix D.1), implemented HERE independently of
   production (``reference_const_row_verdict``): exact sign of
   ``c_eff = c0 + Σ_i a_ext_i t_i`` (> 0: dropped from the geometry only;
   < 0: EMPTY; no tolerance); at ``c_eff == 0.0`` a row that is the
   difference t_p − t_q of two external legs' times (``c0 == 0``, mapped to
   legs through the record's ``tie_ctx``, origin leg included) takes the
   one-sided limit in which leg j sits at t_j − j·δ, δ → 0⁺ (raw times
   first, then the leg order: a later-listed leg is infinitesimally
   earlier); any other zero row is Θ(0) = 0 (EMPTY).  So they are what the
   code SHOULD return.  Every constant row is also given production's
   ``final_integral._const_row_decision``; a different verdict is recorded
   in ``RULE_DIVERGENCES`` / ``ref_meta['rule_divergences']`` and reported
   (the references keep the harness's own verdict).  Under
   ``THETA0_CONST_ROW_MODE = 'legacy_clip'`` the production m=2 polygon
   keeps zero-valued rows (Θ(0) = 1, plan P1), so disagreements on m=2
   subsets with a constant row are EXPECTED in that mode.
4. **Disagreement table**: |value - ref| > max(1e-8·|ref|, 1e-14), grouped by
   (reference, m, bail reason / answering path, row kind), plus the
   ``_RUNTIME_COUNTERS`` snapshot (``nquad_calls`` = entries into the
   scipy.nquad fallback with m>=1, ``polytope_m0_direct`` = m=0 δ-collapsed
   subsets evaluated directly, ``nquad_fallback_by_reason``,
   ``zero_normal_rows_seen``, ...).  Row kind: the M1 ``row_kinds``
   provenance (``edge`` without it), plus ``+zero_normal`` when the subset
   has a constant row (``--ref-scope zero_normal`` selects those subsets).
5. **--stub-nquad**: a counting stub that returns 0 for m>=1 in place of the
   scipy.nquad fallback.  For censuses only: totals are then meaningless,
   the reconstruction check still holds against the stubbed totals.  Two
   levels (``--stub-level``, ``run(stub_level=...)``):

   ``entry`` (default; the level of the M0/M1 censuses) replaces
            ``_integrate_polytope`` itself (in final_integral AND
            grouped_integral, which imports it by name); m=0 passes through
            (a direct evaluation, not quadrature).  The stubbed calls never
            reach the real entry, so ``nquad_calls`` and
            ``scipy_nquad_called_*`` stay 0; they are in ``stub_calls`` (by
            m), and the report labels the counter line accordingly.
   ``quadrature`` keeps the real entry -- its constant-row verdicts and the
            M2a structural-zero test (``polytope_empty_cycle``) run and are
            counted -- and replaces only the three quadrature routines it
            dispatches to (``_integrate_1d_polytope``, ``_integrate_2d_polytope``,
            ``_integrate_nd_polytope``).  ``stub_calls`` (by m) then counts
            the entries that would have reached scipy quadrature (some of them
            still return early inside those routines, e.g. on crossed scalar
            bounds), and ``nquad_calls`` counts every m>=1 entry as usual.
6. **fixture_delta_report()**: max abs / rel deltas of the four frozen
   Phase J fixtures (``tests/phase_j_refactor_fixtures``) between the
   current code and both the frozen ``.npz`` and the ``legacy/`` copies
   (absent until the M0.3 step creates them; reported as missing), per
   probe too.  ``mode='legacy'`` (CLI ``--fixture-mode legacy|both``)
   evaluates under the legacy umbrella's flags, set in-process.

Provenance: ``run()`` records in ``config['diagram_sources']`` where every
(k, ell) diagram list came from -- the typed-diagram cache
(``saved_models/...``) or ``load_prediagrams``' source (``v2`` / ``v1`` /
``shipped`` / ``computed`` / ``eager``) -- plus the cwd (every cache root is
cwd-relative).  MEASURED: the prediagram source changes the isomorphism
representatives and order of the typed diagrams, and with them the Phase J
routes (bail reasons, nquad counts) and ulp-level values, so path censuses
and nquad counts are only comparable between runs with the same sources.

τ convention: the k=2 grid point τ=0 is evaluated by the pipeline at
t1 = -``api.compute._ITO_EPS`` (Itô left limit).  Records and pipeline totals
are keyed by the external times actually passed to ``contribution()``, so the
"τ=0" entries carry -_ITO_EPS.

Budget / safety: a model with fast poles at max_ell=2 can spend hours in
scipy.nquad on unstubbed default code.  Use ``--stub-nquad`` or a hard
subprocess timeout.  Output defaults to ``scratch/phase_j_subset_diff/``
(gitignored); never point it at a tracked path for local-only models.

CLI examples (repo root)::

    sage -python -m tests.tools.phase_j_subset_diff --model ou_quartic \\
        --k 2 --max-ell 1 --taus 0,0.5 --fields dx:1,dx:1 \\
        --params '{"mu": 1.0, "D": 1.0, "eps": 0.02}'
    sage -python -m tests.tools.phase_j_subset_diff \\
        --model single_population_spike_reset_test \\
        --fields n:1,n:2 --taus 0,3 --refs a,b,c --ref-max-m 2 \\
        --params '{"Em": [3.5, 3.5], "tau": [10.0, 9.0], "a": [2.5, 2.5],
                   "w": [[0.55, 0.65], [0.7, 0.8]]}'
    sage -python -m tests.tools.phase_j_subset_diff --model ou_quartic \\
        --k 4 --max-ell 1 --points '0,0.3,0.6,0.9;0,0.5,0.5,0.5' \\
        --fields dx:1,dx:1,dx:1,dx:1 \\
        --params '{"mu": 1.0, "D": 1.0, "eps": 0.02}'
    sage -python -m tests.tools.phase_j_subset_diff --fixture-report \\
        --fixture-mode both          # current defaults AND legacy umbrella

``--model-file path/to/file.py:builder`` loads a model from any file instead
of ``--model``.
"""
from __future__ import annotations

import argparse
import cmath
import collections
import contextlib
import copy
import importlib.util
import itertools
import json
import math
import os
import pickle
import sys
import time

import numpy as np

_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..'))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

DEFAULT_OUT_DIR = os.path.join(_REPO_ROOT, 'scratch', 'phase_j_subset_diff')

# Tolerances (plan §4.4 / §2.4).  Constant rows have no tie tolerance: the
# references decide them with the harness's own ``reference_const_row_
# verdict`` (exact sign; the recorded ``tie_ctx`` orders an external-time
# tie), cross-checked against production's ``_const_row_decision``.
ROW_COEF_ATOL = 1e-12                       # "zero normal" threshold
DISAGREE_RTOL = 1e-8
DISAGREE_ATOL = 1e-14
RECON_RTOL = 1e-10
RECON_ATOL = 1e-14
RECON_SUM_SLACK = 1e-13                     # × Σ|terms|: summation order

REFERENCE_KINDS = {
    'a': 'tight iterated quadrature on the ±200 box',
    'b': 'tight iterated quadrature on the ±12 box',
    'c': 'm=2: 50-digit mpmath fan formula on box(150) [c150] and '
         'box(400) [c400]',
    'd': 'DBM exact route at L=-3000 (NOT AVAILABLE until M3)',
    'e': 'certify_box truncation bound (NOT AVAILABLE until M8)',
}

FIXTURE_NAMES = ('spike_reset_k1_ell1', 'spike_reset_k2_ell0',
                 'spike_reset_k2_ell1', 'quad_exp_k2_ell0')


# ═══════════════════════════════════════════════════════════════════════
# Recording
# ═══════════════════════════════════════════════════════════════════════

def _fi():
    import engine.integration.time_domain.final_integral as FI
    return FI


def _gi():
    import engine.integration.time_domain.grouped_integral as GI
    return GI


def _zero_normal(a_int):
    return all(abs(float(x)) <= ROW_COEF_ATOL for x in a_int)


def has_zero_normal_row(record):
    """True if the record's subset has a constant (zero-normal) row."""
    return any(_zero_normal(a) for (a, _e, _c) in record['constraints'])


def row_kind_tag(record):
    """Row provenance for grouping: M1's ``row_kinds`` when present (else
    ``edge``), plus ``+zero_normal`` when the subset has a constant row."""
    rk = record.get('row_kinds')
    tag = '+'.join(sorted(set(rk))) if rk else 'edge'
    if has_zero_normal_row(record):
        tag += '+zero_normal'
    return tag


def _hashable_time(t):
    """A time of a tie context as a hashable, comparable key entry."""
    try:
        return float(t)
    except (TypeError, ValueError):
        return repr(t)


def tie_ctx_key(tie_ctx):
    """Hashable form of a record's ``tie_ctx`` ``(times, free_legs,
    origin_leg)`` (``None`` stays ``None``)."""
    if tie_ctx is None:
        return None
    times, free_legs, origin = tie_ctx
    return (tuple(_hashable_time(t) for t in times), tuple(free_legs),
            origin)


def ref_key(record):
    """The reference key of a record: ``(key, tie context)``.  ``key`` =
    (diagram build, subset, free_ext_vals) identifies the integrand and the
    rows; the tie context decides the constant rows at an exact tie, which
    can differ between Wick permutations that share ``key``.  (Derived from
    ``key`` and ``tie_ctx`` for a record without the field, e.g. one
    pickled by an earlier version of this harness.)"""
    rk = record.get('ref_key')
    if rk is None:
        rk = (record['key'], tie_ctx_key(record.get('tie_ctx')))
    return rk


class SubsetRecorder:
    """``_SUBSET_HOOK`` callable.  Keeps a picklable record per call and,
    once per unique (diagram build, subset, free_ext_vals), the live
    references (integrand closure, modes, plan, pole tuples) needed to
    compute references afterwards (they do not depend on the Wick
    permutation, so ``live`` is keyed by ``key``, not ``ref_key``)."""

    def __init__(self, keep_live=True):
        self.records = []
        self.live = {}
        self.keep_live = keep_live

    def __call__(self, p):
        ctx = p.get('ctx') or {}
        fv = tuple(p['free_ext_vals'])
        key = (p['diagram_serial'], p['subset_index'], fv)
        tie_ctx = (None if p.get('tie_ctx') is None
                   else tuple(p['tie_ctx']))
        comp = ctx.get('compensation')
        pref = p.get('prefactor')
        rec = {
            'key': key,
            # (key, tie context): what a reference is computed for
            'ref_key': (key, tie_ctx_key(tie_ctx)),
            'source': p['source'],
            'diagram_serial': p['diagram_serial'],
            'loop_number': p['loop_number'],
            'subset_id': p['subset_id'],
            'subset_index': p['subset_index'],
            'delta_edges': p['delta_edges'],
            'smooth_edges': p['smooth_edges'],
            'call_serial': ctx.get('call_serial'),
            'eval_serial': ctx.get('eval_serial'),
            'ext_time_values': ctx.get('ext_time_values'),
            'perm_index': ctx.get('perm_index'),
            'n_perms': ctx.get('n_perms'),
            'compensation': None if comp is None else complex(comp),
            'm': p['m'],
            'constraints': [
                (tuple(float(x) for x in a), tuple(float(x) for x in e),
                 float(c))
                for (a, e, c) in p['constraints']
            ],
            'row_kinds': p.get('row_kinds'),
            'free_ext_vals': fv,
            # (times, free_legs, origin_leg) of the M1 ``_TieContext``, or
            # None: what ordered an exact external-time tie in this call.
            'tie_ctx': tie_ctx,
            'modes_summary': p.get('modes_summary'),
            'prefactor': None if pref is None else complex(pref),
            'path': p['path'],
            'evaluator': p['evaluator'],
            'branch': p['branch'],
            'attempted': p['attempted'],
            'bail_reason': p['bail_reason'],
            'bail_category': p['bail_category'],
            'value': complex(p['value']),
        }
        self.records.append(rec)
        if self.keep_live and key not in self.live:
            self.live[key] = {
                'integrand': p.get('integrand'),
                'modes': p.get('modes'),
                'plan': p.get('plan'),
                'pole_tuples': p.get('pole_tuples'),
                'prefactor': pref,
                'diagram': p.get('diagram'),
            }


@contextlib.contextmanager
def installed_hook(hook):
    """Install ``hook`` as ``final_integral._SUBSET_HOOK``; restore on exit."""
    FI = _fi()
    prev = FI._SUBSET_HOOK
    FI._SUBSET_HOOK = hook
    try:
        yield hook
    finally:
        FI._SUBSET_HOOK = prev


STUB_LEVELS = ('entry', 'quadrature')
# final_integral's quadrature routines behind ``_integrate_polytope`` (m >= 1).
_QUADRATURE_ROUTINES = {'_integrate_1d_polytope': 1,
                        '_integrate_2d_polytope': 2,
                        '_integrate_nd_polytope': None}


class NquadStub:
    """Counting stand-in for the scipy.nquad fallback (m>=1 -> 0j).

    ``level='entry'`` replaces ``_integrate_polytope`` (m=0 passes through);
    ``level='quadrature'`` keeps the real entry and replaces the three
    quadrature routines it dispatches to (module docstring, item 5)."""

    def __init__(self, level='entry'):
        if level not in STUB_LEVELS:
            raise ValueError(f'stub level must be one of {STUB_LEVELS}, '
                             f'got {level!r}')
        self.level = level
        self.calls = collections.Counter()
        self._orig = None

    def __call__(self, integrand_callable, s_constraints, free_ext_vals, m,
                 **kw):
        # ``**kw``: the M1 keywords (``raw_rows``, ``row_kinds``) are passed
        # through to the real m=0 evaluation.
        if m == 0:
            return self._orig(integrand_callable, s_constraints,
                              free_ext_vals, m, **kw)
        self.calls[m] += 1
        return 0.0 + 0.0j

    def _routine_stub(self, name):
        fixed_m = _QUADRATURE_ROUTINES[name]

        def stub(integrand_callable, s_constraints, free_ext_vals, *m):
            self.calls[fixed_m if fixed_m is not None else m[0]] += 1
            return 0.0 + 0.0j
        return stub

    @contextlib.contextmanager
    def installed(self):
        FI, GI = _fi(), _gi()
        if self.level == 'quadrature':
            saved = {n: getattr(FI, n) for n in _QUADRATURE_ROUTINES}
            for n in _QUADRATURE_ROUTINES:
                setattr(FI, n, self._routine_stub(n))
            try:
                yield self
            finally:
                for n, f in saved.items():
                    setattr(FI, n, f)
            return
        self._orig = FI._integrate_polytope
        saved = (FI._integrate_polytope, GI._integrate_polytope)
        FI._integrate_polytope = self
        GI._integrate_polytope = self
        try:
            yield self
        finally:
            FI._integrate_polytope, GI._integrate_polytope = saved


def _grid_point(k, t, ito_eps):
    """The external-time tuple compute_cumulants evaluates for grid τ=t."""
    t = float(t)
    if k == 1:
        return (t,)
    return (0.0, t if abs(t) > 1e-12 else -ito_eps)


def load_model(model=None, model_file=None):
    """``dd.load_model(name)`` for repo models, or ``path.py:callable``."""
    if model_file:
        path, _, fn = model_file.partition(':')
        path = os.path.abspath(path)
        spec = importlib.util.spec_from_file_location(
            '_psd_model_' + os.path.basename(path).replace('.', '_'), path)
        mod = importlib.util.module_from_spec(spec)
        sys.path.insert(0, os.path.dirname(path))
        spec.loader.exec_module(mod)
        return getattr(mod, fn or 'build')()
    import daedalus as dd
    m, _ = dd.load_model(model)
    return m


@contextlib.contextmanager
def recorded_diagram_sources():
    """Record where each (k, ell) diagram list comes from while active.

    Yields a list that fills with ``{'k', 'ell', 'source', ...}`` dicts:
    ``source`` is ``'typed_cache'`` when the typed-diagram cache
    (``saved_models/<model>/unique_typed_mult_*``) answered, else the source
    string ``load_prediagrams`` returned.  Wraps the module attributes the
    pipeline looks up at call time; restored on exit.
    """
    from engine.core import cache as _cache_mod
    from engine.enumeration import prediagram_cache as _pc
    seen = []
    orig_load_pd = _pc.load_prediagrams
    orig_cache_load = _cache_mod.PipelineCache.load

    def load_pd(root, k, ell, *a, **kw):
        out = orig_load_pd(root, k, ell, *a, **kw)
        seen.append({'k': int(k), 'ell': int(ell), 'source': out[1],
                     'prediagram_root': str(root)})
        return out

    def cache_load(self, stage, k=None, loop_order=None):
        out = orig_cache_load(self, stage, k=k, loop_order=loop_order)
        if str(stage).startswith('unique_typed_mult'):
            seen.append({'k': k, 'ell': loop_order, 'source': 'typed_cache',
                         'stage': str(stage), 'cache_root': self.root})
        return out

    _pc.load_prediagrams = load_pd
    _cache_mod.PipelineCache.load = cache_load
    try:
        yield seen
    finally:
        _pc.load_prediagrams = orig_load_pd
        _cache_mod.PipelineCache.load = orig_cache_load


def run(model, *, k, max_ell, external_fields, parameters=None,
        tau_grid=None, points=None, use_grouped_phase_j=False,
        stub_nquad=False, keep_live=True, compute_kwargs=None, label=None,
        stub_level='entry'):
    """Run ``compute_cumulants`` with the hook installed and return a run
    dict (records, pipeline totals keyed by (ell, point), counters, ...).

    ``tau_grid`` (k in {1, 2}) uses the pipeline's own grid values;
    ``points`` (any k) evaluates the RAW per-ell callables at the given
    external-time tuples (no Itô nudge).  ``parallel`` is forced False.

    With ``points`` and no ``tau_grid`` at k in {1, 2}, the pipeline is given
    the one-point grid ``[0.0]`` (otherwise it evaluates its full default
    grid, 201 points, and records every one of them), and the records and
    counters of that auxiliary grid evaluation are dropped (its counters are
    kept as ``counters_aux_grid``), so only the requested points are recorded,
    counted and reconstructed (the stub's ``stub_calls`` are reset too).

    ``stub_level`` ('entry' | 'quadrature') selects what ``stub_nquad``
    replaces (module docstring, item 5).
    """
    from api import compute_cumulants
    from api.compute import _ITO_EPS
    FI = _fi()
    rec = SubsetRecorder(keep_live=keep_live)
    stub = NquadStub(stub_level) if stub_nquad else None
    kw = dict(k=k, max_ell=max_ell, external_fields=list(external_fields),
              use_cache=True, verbose=False,
              use_grouped_phase_j=use_grouped_phase_j)
    if parameters is not None:
        kw['parameters'] = parameters
    if tau_grid is not None:
        kw['tau_grid'] = np.asarray(tau_grid, dtype=float)
    kw.update(compute_kwargs or {})
    kw['parallel'] = False
    aux_grid = (points is not None and tau_grid is None and k in (1, 2)
                and 'tau_grid' not in kw)
    if aux_grid:
        kw['tau_grid'] = np.array([0.0])
    FI._reset_runtime_counters()
    t0 = time.perf_counter()
    pipeline = {}
    with installed_hook(rec), (stub.installed() if stub
                               else contextlib.nullcontext()), \
            recorded_diagram_sources() as sources:
        res = compute_cumulants(model, **kw)
        aux_counters = None
        if aux_grid:
            # Drop the auxiliary one-point grid's records and counters (see
            # docstring); keep its counters separately for reference.
            del rec.records[:]
            rec.live.clear()
            aux_counters = copy.deepcopy(FI._RUNTIME_COUNTERS)
            FI._reset_runtime_counters()
            if stub is not None:
                stub.calls.clear()
        if tau_grid is not None and k in (1, 2):
            for ell, arr in (res.get('C_tau_by_ell') or {}).items():
                if arr is None:
                    continue
                for t, v in zip(np.asarray(tau_grid, dtype=float), arr):
                    pipeline[(int(ell), _grid_point(k, t, _ITO_EPS))] = (
                        complex(v))
        for pt in (points or []):
            pt = tuple(float(x) for x in pt)
            for ell, pj in (res.get('phase_j_by_ell') or {}).items():
                pipeline[(int(ell), pt)] = (
                    0j if pj is None else complex(pj['total_C'](*pt)))
    wall = time.perf_counter() - t0
    return {
        'label': label,
        'config': {
            'k': k, 'max_ell': max_ell,
            'external_fields': [tuple(f) for f in external_fields],
            'parameters': parameters,
            'tau_grid': None if tau_grid is None
            else [float(t) for t in tau_grid],
            'points': None if points is None
            else [tuple(map(float, p)) for p in points],
            'use_grouped_phase_j': use_grouped_phase_j,
            'stub_nquad': stub_nquad,
            'stub_level': stub_level if stub_nquad else None,
            'ito_eps': _ITO_EPS,
            'aux_tau_grid': [0.0] if aux_grid else None,
            'use_cache': kw.get('use_cache'),
            'cwd': os.getcwd(),
            'diagram_sources': sources,
        },
        'records': rec.records,
        'live': rec.live,
        'pipeline': pipeline,
        'counters': copy.deepcopy(FI._RUNTIME_COUNTERS),
        'counters_aux_grid': aux_counters,
        'stub_calls': dict(stub.calls) if stub else None,
        'wall': wall,
        'refs': {},
        'ref_meta': {},
    }


# ═══════════════════════════════════════════════════════════════════════
# Reconstruction
# ═══════════════════════════════════════════════════════════════════════

def reconstruct(run_):
    """(ell, point) -> (Σ value/comp, Σ|value/comp|, n_records), using only
    the first ``contribution()`` call per (diagram build, point)."""
    first = {}
    for r in run_['records']:
        if r['call_serial'] is None:
            continue
        kk = (r['diagram_serial'], r['ext_time_values'])
        cs = first.get(kk)
        if cs is None or r['call_serial'] < cs:
            first[kk] = r['call_serial']
    sums = collections.defaultdict(complex)
    abss = collections.defaultdict(float)
    counts = collections.Counter()
    for r in run_['records']:
        if r['call_serial'] is None:
            continue
        if first[(r['diagram_serial'], r['ext_time_values'])] != (
                r['call_serial']):
            continue
        g = (int(r['loop_number']), r['ext_time_values'])
        term = r['value'] / r['compensation']
        sums[g] += term
        abss[g] += abs(term)
        counts[g] += 1
    return {g: (sums[g], abss[g], counts[g]) for g in sums}


def reconstruction_check(run_):
    """Rows (ell, point, pipeline, reconstructed, |diff|, tol, ok)."""
    rec = reconstruct(run_)
    rows = []
    for (ell, pt), pipe in sorted(run_['pipeline'].items()):
        s, sabs, n = rec.get((ell, pt), (0j, 0.0, 0))
        diff = abs(s - pipe)
        tol = RECON_RTOL * abs(pipe) + RECON_SUM_SLACK * sabs + RECON_ATOL
        rows.append({'ell': ell, 'point': pt, 'pipeline': pipe,
                     'reconstructed': s, 'abs_diff': diff, 'tol': tol,
                     'n_records': n, 'ok': diff <= tol})
    orphans = [g for g in rec if g not in run_['pipeline']]
    return rows, orphans


# ═══════════════════════════════════════════════════════════════════════
# References
# ═══════════════════════════════════════════════════════════════════════

_EDGE_KINDS = ('edge', 'conv_pseudo')
_BOX_KINDS = ('noise_box', 'conv_tau')

#: Constant rows whose harness verdict differs from production's
#: ``final_integral._const_row_decision``: dicts with the row, the free
#: values, the tie context and both verdicts.  ``compute_references``
#: copies its share into ``run_['ref_meta']['rule_divergences']``.
RULE_DIVERGENCES = []


def reference_const_row_verdict(a_ext, c0, free_vals, kind='edge',
                                 tie_ctx=None):
    r"""The harness's own statement of the 0.2.0 constant-row rule
    (``'DROP'`` or ``'EMPTY'``), written from the rule's definition rather
    than from production's code, so that the references are an independent
    check of the decision.

    ``c_eff = c0 + Σ_i a_ext_i t_i`` (the row's value; exact sign, no
    tolerance): > 0 DROP, < 0 (or NaN) EMPTY.  At ``c_eff == 0.0``: if
    ``c0 == 0`` and the row, rewritten on the legs' RAW times through
    ``tie_ctx = (times, free_legs, origin_leg)`` (``free_ext_vals[i] =
    times[free_legs[i]] − times[origin_leg]``), is ``t_p − t_q`` for two
    legs, it is evaluated in the one-sided limit where leg j sits at
    ``t_j − j·δ``, δ → 0⁺: lexicographically, first the exactly rounded raw
    difference, then ``−Σ_j coef_j · j``.  Any other zero row is
    Θ(0) = 0: EMPTY.  A constant box row ('noise_box' / 'conv_tau')
    raises (never expected, plan §7.6).
    """
    if kind in _BOX_KINDS:
        raise ValueError(f'constant {kind!r} row: not expected (plan §7.6)')
    if kind not in _EDGE_KINDS:
        raise ValueError(f'unknown row kind {kind!r}')
    c = float(c0) + sum(float(a_ext[j]) * float(free_vals[j])
                        for j in range(len(a_ext)))
    if c > 0.0:
        return 'DROP'
    if not c == 0.0:
        return 'EMPTY'
    if tie_ctx is None or float(c0) != 0.0:
        return 'EMPTY'
    times, free_legs, origin = tie_ctx
    if len(a_ext) > len(free_legs):
        return 'EMPTY'
    coef = collections.defaultdict(float)
    for i, a in enumerate(a_ext):
        coef[free_legs[i]] += float(a)
        if origin is not None:
            coef[origin] -= float(a)
    coef = {leg: a for leg, a in coef.items() if a != 0.0}
    if sorted(coef.values()) != [-1.0, 1.0]:
        return 'EMPTY'
    raw = math.fsum(a * float(times[leg]) for leg, a in coef.items())
    if raw != raw:                      # NaN time: no order
        return 'EMPTY'
    if raw != 0.0:
        return 'DROP' if raw > 0.0 else 'EMPTY'
    pert = -sum(a * leg for leg, a in coef.items())
    return 'DROP' if pert > 0 else 'EMPTY'


def _resolve_rows(constraints, free_vals, tie_ctx=None, row_kinds=None):
    """Itô constant-row verdict + the non-constant rows.

    Constant rows are decided by ``reference_const_row_verdict`` (the
    harness's own rule); ``tie_ctx`` is a record's ``(times, free_legs,
    origin_leg)`` (or ``None``: an exact tie is Θ(0) = 0).  Each constant
    row is also given production's ``final_integral._const_row_decision``
    (side-effect free); a different verdict is appended to
    ``RULE_DIVERGENCES``.  Returns ('EMPTY', None) or
    ('OK', [(a ndarray, c_eff), ...]).
    """
    FI = _fi()
    tc = None if tie_ctx is None else FI._TieContext(*tie_ctx)
    out = []
    empty = False
    for idx, (a_int, a_ext, c0) in enumerate(constraints):
        c = float(c0) + sum(float(a_ext[j]) * float(free_vals[j])
                            for j in range(len(a_ext)))
        if _zero_normal(a_int):
            kind = 'edge' if not row_kinds else row_kinds[idx]
            verdict = reference_const_row_verdict(a_ext, c0, free_vals,
                                                  kind, tie_ctx)
            prod = FI._const_row_decision(a_int, a_ext, c0, free_vals,
                                          kind, tc)[0]
            if prod != verdict:
                RULE_DIVERGENCES.append({
                    'row': (tuple(a_int), tuple(a_ext), c0),
                    'free_ext_vals': tuple(free_vals), 'kind': kind,
                    'tie_ctx': tie_ctx, 'harness': verdict,
                    'production': prod})
            if verdict == 'EMPTY':
                empty = True                # Θ(0)=0 / ordered tie
            continue                        # DROP: geometry only
        out.append((np.array([float(x) for x in a_int]), c))
    if empty:
        return 'EMPTY', None
    return 'OK', out


def _vertex_coords(A, C, k_axis, max_combos=20000):
    """Coordinate ``k_axis`` of every vertex of {x : A x + C > 0} (A is
    n×d with d = k_axis+1).  ``None`` if too many combinations."""
    n, d = A.shape
    nz = np.where(np.any(np.abs(A) > 1e-15, axis=1))[0]
    if len(nz) < d:
        return []
    if math.comb(len(nz), d) > max_combos:
        return None
    combos = np.array(list(itertools.combinations(nz, d)))
    M = A[combos]                                   # (nc, d, d)
    rhs = -C[combos]                                # (nc, d)
    det = np.linalg.det(M)
    good = np.abs(det) > 1e-12
    if not np.any(good):
        return []
    X = np.linalg.solve(M[good], rhs[good][..., None])[..., 0]
    slack = X @ A.T + C                             # (ng, n)
    feas = np.all(slack >= -1e-9 * (1.0 + np.abs(C)), axis=1)
    return sorted(set(float(x) for x in X[feas, k_axis]))


def tight_quad_reference(integrand, constraints, free_vals, m, *, box,
                         scale=1.0, epsabs_factor=1e-13, epsrel=1e-10,
                         limit=400, tie_ctx=None, row_kinds=None):
    """Tight iterated quadrature of ``integrand(s_0..s_{m-1}, *free)`` over
    {rows > 0} ∩ [-box, box]^m (Itô constant rows).  Innermost s_0.

    Returns ``(value, info)``; ``info`` has the worst quad_vec error estimate
    and status (0 = converged) over all levels.
    """
    from scipy.integrate import quad_vec
    verdict, rows = _resolve_rows(constraints, free_vals, tie_ctx, row_kinds)
    info = {'verdict': verdict, 'max_err': 0.0, 'worst_status': 0,
            'vertex_breakpoints': True}
    if verdict == 'EMPTY' or m == 0:
        return 0j, info
    for i in range(m):
        e = np.zeros(m)
        e[i] = 1.0
        rows.append((e.copy(), float(box)))
        rows.append((-e, float(box)))
    A = np.array([a for a, _c in rows])
    C = np.array([c for _a, c in rows])
    free = tuple(float(x) for x in free_vals)
    ext_pts = sorted(set([0.0] + list(free)))
    epsabs = epsabs_factor * scale
    zero = np.zeros(2)

    def f0(x, outer):
        v = complex(integrand(x, *outer, *free))
        return np.array([v.real, v.imag])

    def level(k, outer):
        Ak = A[:, :k + 1]
        Ck = C + (A[:, k + 1:] @ np.array(outer) if outer else 0.0)
        nzk = np.any(np.abs(Ak) > 1e-15, axis=1)
        if np.any(~nzk & (Ck <= 0.0)):
            return zero
        pure = nzk & ~np.any(np.abs(Ak[:, :k]) > 1e-15, axis=1)
        L, U = -math.inf, math.inf
        for i in np.where(pure)[0]:
            a = Ak[i, k]
            b = -Ck[i] / a
            if a > 0:
                L = max(L, b)
            else:
                U = min(U, b)
        if not L < U:
            return zero
        pts = [p for p in ext_pts if L < p < U]
        if k >= 1:
            vc = _vertex_coords(Ak, Ck, k)
            if vc is None:
                info['vertex_breakpoints'] = False
            else:
                pts += [p for p in vc if L < p < U]
        pts = sorted(set(pts))
        if k == 0:
            g = lambda x: f0(x, outer)           # noqa: E731
        else:
            g = lambda x: level(k - 1, (x,) + outer)   # noqa: E731
        val, err, qinfo = quad_vec(g, L, U, epsabs=epsabs, epsrel=epsrel,
                                   points=pts or None, limit=limit,
                                   full_output=True)
        info['max_err'] = max(info['max_err'], float(err))
        info['worst_status'] = max(info['worst_status'],
                                   int(getattr(qinfo, 'status', 0)))
        return val

    try:
        v = level(m - 1, ())
    except (OverflowError, FloatingPointError, ValueError) as exc:
        info['error'] = repr(exc)
        return complex('nan'), info
    return complex(v[0], v[1]), info


def _pole_terms(live):
    """[(coefficient incl. prefactor, lambdas)] or ``None``."""
    pref = live.get('prefactor')
    pref = 1.0 + 0.0j if pref is None else complex(pref)
    plan = live.get('plan')
    pt = plan['pole_tuples'] if plan is not None else live.get('pole_tuples')
    if pt is not None:
        return [(pref * complex(C), tuple(complex(x) for x in lams))
                for (C, lams) in pt]
    modes = live.get('modes')
    if modes is None:
        return None
    out = []
    for combo in itertools.product(*[ms.modes for ms in modes]):
        C = pref
        for (c, _l) in combo:
            C *= complex(c)
        out.append((C, tuple(complex(l) for (_c, l) in combo)))
    return out


def _integrand_scale(live):
    terms = _pole_terms(live)
    if not terms:
        return 1.0
    s = sum(abs(c) for c, _l in terms)
    return s if s > 0 else 1.0


def _mp_J(p, q, mp):
    """∫_0^1 ∫_0^{1-u} exp(p u + q w) dw du  (= exp[0, p, q]), mp inputs,
    with extra working precision against cancellation."""
    ap, aq, ad = abs(p), abs(q), abs(p - q)
    big = max(mp.mpf(1), ap, aq)
    extra = 10
    for small in (ap, aq, ad):
        if small != 0 and small < big:
            extra += int(mp.ceil(mp.log10(big / small))) + 2
    extra = min(extra, 2000)
    with mp.extradps(extra):
        def E(x):
            return mp.expm1(x) / x if x != 0 else mp.mpf(1)
        if p == q:
            if p == 0:
                return mp.mpf(1) / 2
            return (mp.exp(p) * (p - 1) + 1) / (p * p)
        return (E(q) - E(p)) / (q - p)


def mp_fan_reference(terms, constraints, free_vals, *, box, dps=50,
                     tie_ctx=None, row_kinds=None):
    """m=2: Σ_terms coef · ∫∫_poly exp(α s0 + β s1 + γ) at ``dps`` digits,
    polygon = [-box, box]² clipped by the non-constant rows (Itô constant
    rows).  ``terms`` from ``_pole_terms``."""
    import mpmath
    mp = mpmath.mp
    with mp.workdps(dps):
        verdict, rows = _resolve_rows(constraints, free_vals, tie_ctx,
                                      row_kinds)
        if verdict == 'EMPTY':
            return 0j
        c_eff_all = []
        for (a_int, a_ext, c0) in constraints:
            c_eff_all.append(mp.mpf(float(c0)) + sum(
                mp.mpf(float(a_ext[j])) * mp.mpf(float(free_vals[j]))
                for j in range(len(a_ext))))
        B = mp.mpf(float(box))
        poly = [(-B, -B), (B, -B), (B, B), (-B, B)]
        for (a, _c), (a_int, a_ext, c0) in zip(
                rows, [r for r in constraints if not _zero_normal(r[0])]):
            ce = mp.mpf(float(c0)) + sum(
                mp.mpf(float(a_ext[j])) * mp.mpf(float(free_vals[j]))
                for j in range(len(a_ext)))
            a0, a1 = mp.mpf(float(a[0])), mp.mpf(float(a[1]))
            out = []
            n = len(poly)
            for i in range(n):
                P, Q = poly[i - 1], poly[i]
                fP = a0 * P[0] + a1 * P[1] + ce
                fQ = a0 * Q[0] + a1 * Q[1] + ce
                if fQ >= 0:
                    if fP < 0:
                        t = fP / (fP - fQ)
                        out.append((P[0] + t * (Q[0] - P[0]),
                                    P[1] + t * (Q[1] - P[1])))
                    out.append(Q)
                elif fP >= 0:
                    t = fP / (fP - fQ)
                    out.append((P[0] + t * (Q[0] - P[0]),
                                P[1] + t * (Q[1] - P[1])))
            poly = out
            if len(poly) < 3:
                return 0j
        total = mp.mpc(0)
        a_rows = [(mp.mpf(float(a[0])), mp.mpf(float(a[1])))
                  for (a, _e, _c) in constraints]
        for coef, lams in terms:
            lm = [mp.mpc(x.real, x.imag) for x in lams]
            al = sum(l * a[0] for l, a in zip(lm, a_rows))
            be = sum(l * a[1] for l, a in zip(lm, a_rows))
            ga = sum(l * c for l, c in zip(lm, c_eff_all))
            i0 = max(range(len(poly)),
                     key=lambda i: mp.re(al * poly[i][0] + be * poly[i][1]))
            pv = poly[i0:] + poly[:i0]
            v0 = pv[0]
            s = mp.mpc(0)
            for i in range(1, len(pv) - 1):
                v1, v2 = pv[i], pv[i + 1]
                e1 = (v1[0] - v0[0], v1[1] - v0[1])
                e2 = (v2[0] - v0[0], v2[1] - v0[1])
                det = e1[0] * e2[1] - e1[1] * e2[0]
                if det == 0:
                    continue
                p = al * e1[0] + be * e1[1]
                q = al * e2[0] + be * e2[1]
                s += abs(det) * mp.exp(al * v0[0] + be * v0[1] + ga) * \
                    _mp_J(p, q, mp)
            total += mp.mpc(coef.real, coef.imag) * s
        return complex(total)


def dbm_reference(*_args, **_kwargs):
    """Extension point (d): exact DBM route at L = -3000.  Lands with M3
    (plan §3.1 L7, ``poset_dbm_integrator.py`` promotion)."""
    raise NotImplementedError(
        'reference (d) DBM at L=-3000 is not available before milestone M3 '
        '(docs/integration_speedup_plan.md §3.1 L7).')


def certify_box_reference(*_args, **_kwargs):
    """Extension point (e): ``certify_box`` bound S·Q(m, T) for the box
    actually used.  Lands with M8 (plan §3.1 L5)."""
    raise NotImplementedError(
        'reference (e) certify_box is not available before milestone M8 '
        '(docs/integration_speedup_plan.md §3.1 L5).')


def compute_references(run_, kinds=('a', 'b', 'c'), *, ref_max_m=2,
                       scope='all', max_subsets=None, time_budget=None,
                       verbose=False):
    """Fill ``run_['refs'][ref_key][ref_name] = complex`` for unique subset
    evaluations with 1 <= m <= ref_max_m, one per ``ref_key`` = (diagram
    build, subset, free_ext_vals, tie context): every tie orientation of a
    subset is referenced, with its own tie context, and cross-checked
    against production's rule.  ``scope``: 'all', 'fallback' (answered by
    nquad), 'analytic' (answered analytically) or 'zero_normal' (has a
    constant row)."""
    for kd in kinds:
        if kd == 'd':
            dbm_reference()
        if kd == 'e':
            certify_box_reference()
        if kd not in ('a', 'b', 'c'):
            raise ValueError(f'unknown reference kind {kd!r}; '
                             f'known: {sorted(REFERENCE_KINDS)}')
    if run_['config'].get('stub_nquad'):
        raise ValueError('references are meaningless on a --stub-nquad run')
    uniq = {}
    for r in run_['records']:
        if not (1 <= r['m'] <= ref_max_m):
            continue
        if scope == 'fallback' and r['path'] != 'nquad':
            continue
        if scope == 'analytic' and r['path'] == 'nquad':
            continue
        if scope == 'zero_normal' and not has_zero_normal_row(r):
            continue
        uniq.setdefault(ref_key(r), r)
    t0 = time.perf_counter()
    n_done = 0
    n_div0 = len(RULE_DIVERGENCES)
    for rkey, r in uniq.items():
        if max_subsets is not None and n_done >= max_subsets:
            break
        if time_budget is not None and time.perf_counter() - t0 > time_budget:
            run_['ref_meta']['truncated_by_time_budget'] = True
            break
        live = run_['live'].get(r['key'])    # the integrand: per subset
        if live is None or live.get('integrand') is None:
            continue
        refs = run_['refs'].setdefault(rkey, {})
        meta = run_['ref_meta'].setdefault(rkey, {})
        scale = _integrand_scale(live)
        meta['scale'] = scale
        for kd in kinds:
            if kd in ('a', 'b'):
                box = 200.0 if kd == 'a' else 12.0
                ts = time.perf_counter()
                v, info = tight_quad_reference(
                    live['integrand'], r['constraints'], r['free_ext_vals'],
                    r['m'], box=box, scale=scale, tie_ctx=r.get('tie_ctx'),
                    row_kinds=r.get('row_kinds'))
                refs[kd] = v
                meta[kd] = dict(info, seconds=time.perf_counter() - ts)
            elif kd == 'c' and r['m'] == 2:
                terms = _pole_terms(live)
                if terms is None:
                    meta['c'] = 'no pole data'
                    continue
                for box in (150.0, 400.0):
                    refs[f'c{int(box)}'] = mp_fan_reference(
                        terms, r['constraints'], r['free_ext_vals'], box=box,
                        tie_ctx=r.get('tie_ctx'),
                        row_kinds=r.get('row_kinds'))
        n_done += 1
        if verbose and n_done % 25 == 0:
            print(f'  references: {n_done}/{len(uniq)} subsets '
                  f'({time.perf_counter() - t0:.0f}s)', flush=True)
    run_['ref_meta']['n_subsets_with_refs'] = n_done
    run_['ref_meta']['n_candidates'] = len(uniq)
    run_['ref_meta'].setdefault('rule_divergences', []).extend(
        RULE_DIVERGENCES[n_div0:])
    return run_


# ═══════════════════════════════════════════════════════════════════════
# Tables / reports
# ═══════════════════════════════════════════════════════════════════════

def _first_record_by_ref_key(run_):
    """``ref_key`` -> the first record with it (every record with the same
    ``ref_key`` evaluates the same subset, free values and tie context)."""
    out = {}
    for r in run_['records']:
        out.setdefault(ref_key(r), r)
    return out


def disagreement_table(run_):
    """Per (ref, m, bail-reason-or-path, row kind): n compared, n disagree
    (|value-ref| > max(1e-8|ref|, 1e-14)), non-finite refs, max abs/rel.
    One comparison per ``ref_key``: each tie orientation of a subset is
    compared with the reference of its own tie context."""
    by_key = _first_record_by_ref_key(run_)
    table = {}
    worst = []
    for key, refs in run_['refs'].items():
        r = by_key[key]
        reason = r['bail_reason'] or r['path']
        for name, ref in refs.items():
            g = (name, r['m'], reason, row_kind_tag(r))
            row = table.setdefault(g, {'n': 0, 'n_disagree': 0,
                                       'n_nonfinite_ref': 0,
                                       'max_abs': 0.0, 'max_rel': 0.0})
            row['n'] += 1
            if ref is None or not cmath.isfinite(ref):
                row['n_nonfinite_ref'] += 1
                continue
            d = abs(r['value'] - ref)
            rel = d / abs(ref) if ref != 0 else (0.0 if d == 0 else math.inf)
            row['max_abs'] = max(row['max_abs'], d)
            row['max_rel'] = max(row['max_rel'], rel)
            if d > max(DISAGREE_RTOL * abs(ref), DISAGREE_ATOL):
                row['n_disagree'] += 1
                worst.append((d, name, key, r['value'], ref))
    worst.sort(key=lambda x: -x[0])
    return table, worst


def attribution(run_, ref_name):
    """(ell, point) -> (Σ (value - ref)/comp, n_terms_with_ref,
    n_terms_without_ref) over the reconstruction records: how much of the
    pipeline total moves if every subset returned ``ref_name`` instead of
    its production value (the per-subset attribution).  Each record is
    compared with the reference of its own ``ref_key`` (its own tie
    context).  Only exact when every subset that differs has that
    reference (``ref_max_m``)."""
    first = {}
    for r in run_['records']:
        if r['call_serial'] is None:
            continue
        kk = (r['diagram_serial'], r['ext_time_values'])
        if kk not in first or r['call_serial'] < first[kk]:
            first[kk] = r['call_serial']
    out = {}
    for r in run_['records']:
        if r['call_serial'] is None or first[
                (r['diagram_serial'], r['ext_time_values'])] != r['call_serial']:
            continue
        g = (int(r['loop_number']), r['ext_time_values'])
        s, n_with, n_without = out.get(g, (0j, 0, 0))
        ref = run_['refs'].get(ref_key(r), {}).get(ref_name)
        if ref is None or not cmath.isfinite(ref):
            out[g] = (s, n_with, n_without + 1)
        else:
            out[g] = (s + (r['value'] - ref) / r['compensation'],
                      n_with + 1, n_without)
    return out


def path_census(run_):
    """Counter over (source, path, branch, m, bail_reason)."""
    return collections.Counter(
        (r['source'], r['path'], r['branch'], r['m'], r['bail_reason'])
        for r in run_['records'])


def format_report(run_, *, max_worst=15):
    lines = []
    cfg = run_['config']
    lines.append(f"== phase_j_subset_diff: {run_.get('label') or ''}  "
                 f"k={cfg['k']} max_ell={cfg['max_ell']} "
                 f"grouped={cfg['use_grouped_phase_j']} "
                 f"stub_nquad={cfg['stub_nquad']}  wall={run_['wall']:.1f}s  "
                 f"records={len(run_['records'])}")
    c = run_['counters']
    stub_calls = run_.get('stub_calls')
    entry_stub = cfg.get('stub_level', 'entry') in (None, 'entry')
    n_stub = (sum(n for m, n in stub_calls.items() if int(m) >= 1)
              if stub_calls is not None else 0)
    if stub_calls is None:
        stub_note = ''
    elif entry_stub:
        stub_note = (f" [nquad STUBBED: {n_stub} m>=1 calls intercepted "
                     f"before the real entry, so these stay 0; see 'stubbed "
                     f"nquad calls by m']")
    else:
        stub_note = (f" [quadrature STUBBED behind the real entry: {n_stub} "
                     f"of these reached a quadrature routine; see 'stubbed "
                     f"nquad calls by m']")
    lines.append(f"counters: nquad_calls={c.get('nquad_calls')} "
                 f"(m1={c.get('scipy_nquad_called_m1')} "
                 f"m2={c.get('scipy_nquad_called_m2')} "
                 f"m>=3={c.get('scipy_nquad_called_mge3')}){stub_note}  "
                 f"polytope_m0_direct={c.get('polytope_m0_direct')}  "
                 f"zero_normal_rows_seen={c.get('zero_normal_rows_seen')}  "
                 f"polytope_empty_cycle={c.get('polytope_empty_cycle')}  "
                 f"forced_delta_pruned={c.get('forced_delta_pruned')}")
    by = c.get('nquad_fallback_by_reason') or {}
    n_extra = n_stub if entry_stub else 0
    lines.append(f"nquad_fallback_by_reason (dispatch side; sum "
                 f"{sum(by.values())} vs nquad_calls {c.get('nquad_calls')}"
                 f" + stubbed {n_extra}): {by}")
    if stub_calls is not None:
        lines.append(f"stubbed nquad calls by m: {stub_calls}")
    srcs = cfg.get('diagram_sources')
    if srcs is not None:
        lines.append('diagram sources (k, ell, source): ' + (', '.join(
            f"({d['k']},{d['ell']},{d['source']})" for d in srcs)
            or 'none recorded') + f"  cwd={cfg.get('cwd')}")
    lines.append('path census (source, path, branch, m, bail_reason): n')
    for kk, n in sorted(path_census(run_).items(), key=lambda x: str(x[0])):
        lines.append(f'   {kk}: {n}')
    rows, orphans = reconstruction_check(run_)
    lines.append('reconstruction (Σ value/comp vs pipeline):')
    for row in rows:
        lines.append(
            f"   ell={row['ell']} pt={row['point']}: pipeline="
            f"{row['pipeline'].real:.15g}{row['pipeline'].imag:+.2e}j "
            f"recon={row['reconstructed'].real:.15g} "
            f"|d|={row['abs_diff']:.2e} tol={row['tol']:.1e} "
            f"n={row['n_records']} {'OK' if row['ok'] else 'FAIL'}")
    if orphans:
        lines.append(f'   records at points the pipeline did not report: '
                     f'{orphans}')
    if run_['refs']:
        table, worst = disagreement_table(run_)
        lines.append(f"references: {run_['ref_meta'].get('n_subsets_with_refs')}"
                     f" of {run_['ref_meta'].get('n_candidates')} candidate "
                     f"(subset, free values, tie context) evaluations")
        div = run_['ref_meta'].get('rule_divergences') or []
        lines.append(f"constant-row rule: {len(div)} verdict(s) where the "
                     f"harness differs from final_integral._const_row_"
                     f"decision" + (' (references use the harness verdict)'
                                    if div else ''))
        for d in div[:max_worst]:
            lines.append(f"   rule divergence: row={d['row']} "
                         f"free={d['free_ext_vals']} tie_ctx={d['tie_ctx']} "
                         f"harness={d['harness']} production="
                         f"{d['production']}")
        lines.append('disagreements (ref, m, bail_reason|path, row_kind): '
                     'n / n_disagree / nonfinite / max_abs / max_rel')
        for g, row in sorted(table.items(), key=lambda x: str(x[0])):
            lines.append(f"   {g}: {row['n']} / {row['n_disagree']} / "
                         f"{row['n_nonfinite_ref']} / {row['max_abs']:.2e} / "
                         f"{row['max_rel']:.2e}")
        for d, name, key, val, ref in worst[:max_worst]:
            lines.append(f'   worst {name} key={key}: value={val:.12g} '
                         f'ref={ref:.12g} |d|={d:.3e}')
        names = sorted({n for refs in run_['refs'].values() for n in refs})
        lines.append('attribution Σ (value - ref)/comp per (ell, point) '
                     '[n with ref / n without]:')
        for name in names:
            for g, (s, nw, nwo) in sorted(attribution(run_, name).items()):
                lines.append(f'   {name} ell={g[0]} pt={g[1]}: '
                             f'{s.real:+.13g}{s.imag:+.1e}j [{nw}/{nwo}]')
    return '\n'.join(lines)


def _jsonable(x):
    if isinstance(x, complex):
        return [x.real, x.imag]
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, float) and not math.isfinite(x):
        return str(x)
    if x is None or isinstance(x, (str, int, float, bool)):
        return x
    return repr(x)


def save(run_, out_prefix):
    """Write ``<out_prefix>.json`` (summary) and ``<out_prefix>.pkl``
    (records, refs, pipeline, counters; live references dropped)."""
    os.makedirs(os.path.dirname(os.path.abspath(out_prefix)), exist_ok=True)
    rows, orphans = reconstruction_check(run_)
    table, worst = disagreement_table(run_) if run_['refs'] else ({}, [])
    summary = {
        'label': run_.get('label'), 'config': run_['config'],
        'wall': run_['wall'], 'n_records': len(run_['records']),
        'counters': run_['counters'], 'stub_calls': run_.get('stub_calls'),
        'path_census': {str(k): v for k, v in path_census(run_).items()},
        'reconstruction': rows, 'reconstruction_orphans': orphans,
        'disagreements': {str(k): v for k, v in table.items()},
        'worst': [(d, n, str(k), v, r) for d, n, k, v, r in worst[:200]],
        'ref_meta_global': {k: v for k, v in run_['ref_meta'].items()
                            if not isinstance(k, tuple)},
    }
    with open(out_prefix + '.json', 'w') as fh:
        json.dump(_jsonable(summary), fh, indent=1)
    slim = {k: v for k, v in run_.items() if k != 'live'}
    with open(out_prefix + '.pkl', 'wb') as fh:
        pickle.dump(slim, fh)
    return out_prefix + '.json', out_prefix + '.pkl'


# ═══════════════════════════════════════════════════════════════════════
# Fixture delta report
# ═══════════════════════════════════════════════════════════════════════

@contextlib.contextmanager
def phase_j_flags(mode='default'):
    """Set the Phase J flags in-process for the duration of the block.

    ``'default'`` leaves the module attributes as they are; ``'legacy'``
    applies what ``DAEDALUS_PHASE_J_LEGACY=1`` selects
    (``final_integral._initial_phase_j_flags``).  The flags are read at call
    time, so this takes effect at once; restored on exit.
    """
    if mode == 'default':
        yield {}
        return
    if mode != 'legacy':
        raise ValueError(f"mode must be 'default' or 'legacy', got {mode!r}")
    FI = _fi()
    flags = FI._initial_phase_j_flags({'DAEDALUS_PHASE_J_LEGACY': '1'})
    saved = {k: getattr(FI, k) for k in flags}
    try:
        for k, v in flags.items():
            setattr(FI, k, v)
        yield flags
    finally:
        for k, v in saved.items():
            setattr(FI, k, v)


def fixture_delta_report(names=None, *, verbose=True, mode='default'):
    """Max abs / rel deltas of each frozen Phase J fixture: current code vs
    the frozen ``.npz`` and vs ``legacy/<name>.npz`` (reported 'missing'
    when that copy does not exist yet).

    ``mode='legacy'`` evaluates under the legacy umbrella's flags (set
    in-process, see ``phase_j_flags``): the pre-change numbers, which must
    reproduce the ``legacy/`` copies.  With ``verbose`` the per-probe values
    are printed as well."""
    from tests.phase_j_refactor_fixtures._configs import FIXTURES
    from tests.phase_j_refactor_fixtures._runner import evaluate, fixture_path
    names = tuple(names or FIXTURE_NAMES)
    out = {}
    for fx in FIXTURES:
        if fx.name not in names:
            continue
        with phase_j_flags(mode):
            cur = evaluate(fx)
        entry = {'wall': cur['wall_time'], 'current': cur['C_values'],
                 'mode': mode,
                 'tau_probes': [tuple(p) for p in cur['tau_probes'].tolist()]}
        for label, path in (
                ('frozen', fixture_path(fx.name)),
                ('legacy', os.path.join(os.path.dirname(fixture_path(fx.name)),
                                        'legacy', f'{fx.name}.npz'))):
            if not os.path.exists(path):
                entry[label] = 'missing'
                continue
            ref = np.load(path)['C_values']
            d = np.abs(cur['C_values'] - ref)
            rel = d / np.maximum(np.abs(ref), 1e-300)
            entry[label] = {'max_abs': float(d.max()),
                            'max_rel': float(rel.max()),
                            'per_probe_abs': d.tolist(),
                            'per_probe_rel': rel.tolist(),
                            'values': ref}
        out[fx.name] = entry
        if verbose:
            def fmt(e):
                return e if isinstance(e, str) else (
                    f"max_abs={e['max_abs']:.3e} max_rel={e['max_rel']:.3e}")
            print(f"{fx.name:22s} [{mode}] ({entry['wall']:.1f}s)  vs frozen: "
                  f"{fmt(entry['frozen'])}   vs legacy: "
                  f"{fmt(entry['legacy'])}", flush=True)
            ref = entry['legacy'] if not isinstance(
                entry['legacy'], str) else entry['frozen']
            if not isinstance(ref, str):
                for i, (pt, v) in enumerate(zip(entry['tau_probes'],
                                                cur['C_values'])):
                    r = ref['values'][i]
                    print(f"    probe {pt}: current {v.real:+.12e}  "
                          f"pre-M1 {r.real:+.12e}  |d|={ref['per_probe_abs'][i]:.3e}"
                          f"  rel={ref['per_probe_rel'][i]:.3e}", flush=True)
    return out


# ═══════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════

def _parse_fields(s):
    out = []
    for tok in s.split(','):
        name, _, idx = tok.strip().partition(':')
        out.append((name, int(idx or 1)))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(
        description='Per-subset Phase J differential harness (M0.1).')
    ap.add_argument('--model', help='repo model name (daedalus.load_model)')
    ap.add_argument('--model-file', help='path/to/file.py:builder')
    ap.add_argument('--k', type=int, default=2)
    ap.add_argument('--max-ell', type=int, default=1)
    ap.add_argument('--taus', default=None,
                    help='comma list; k in {1,2} grid (τ=0 -> -_ITO_EPS); '
                         "default '0,0.5' unless --points is given")
    ap.add_argument('--points', default=None,
                    help="explicit k-tuples 'a,b;c,d' (raw per-ell "
                         'callables); implies no τ grid unless --taus is '
                         'also given')
    ap.add_argument('--fields', help="e.g. 'dx:1,dx:1'")
    ap.add_argument('--params', default=None, help='JSON dict')
    ap.add_argument('--grouped', action='store_true')
    ap.add_argument('--stub-nquad', action='store_true')
    ap.add_argument('--stub-level', default='entry', choices=list(STUB_LEVELS),
                    help="what --stub-nquad replaces: 'entry' "
                         "(_integrate_polytope) or 'quadrature' (the "
                         'routines behind its entry checks)')
    ap.add_argument('--refs', default='a,b,c',
                    help="subset of a,b,c ('' for none; d,e raise)")
    ap.add_argument('--ref-max-m', type=int, default=2)
    ap.add_argument('--ref-scope', default='all',
                    choices=['all', 'fallback', 'analytic', 'zero_normal'])
    ap.add_argument('--ref-time-budget', type=float, default=600.0)
    ap.add_argument('--out', default=None, help='output prefix (no ext)')
    ap.add_argument('--fixture-report', action='store_true')
    ap.add_argument('--fixtures', default=None, help='comma list of names')
    ap.add_argument('--fixture-mode', default='default',
                    choices=['default', 'legacy', 'both'],
                    help="flags for --fixture-report: the current defaults, "
                         "the legacy umbrella's (in-process), or both")
    args = ap.parse_args(argv)

    if args.fixture_report:
        modes = (('default', 'legacy') if args.fixture_mode == 'both'
                 else (args.fixture_mode,))
        for mode in modes:
            fixture_delta_report(args.fixtures.split(',') if args.fixtures
                                 else None, mode=mode)
        return 0
    if not (args.model or args.model_file) or not args.fields:
        ap.error('--model or --model-file, and --fields, are required')
    model = load_model(args.model, args.model_file)
    taus_arg = args.taus if args.taus is not None else (
        None if args.points else '0,0.5')
    taus = ([float(x) for x in taus_arg.split(',')]
            if taus_arg and args.k in (1, 2) else None)
    points = ([tuple(float(x) for x in p.split(','))
               for p in args.points.split(';')] if args.points else None)
    label = (args.model or os.path.basename(args.model_file)).replace(
        '.py', '').replace(':', '_')
    run_ = run(model, k=args.k, max_ell=args.max_ell,
               external_fields=_parse_fields(args.fields),
               parameters=json.loads(args.params) if args.params else None,
               tau_grid=taus, points=points,
               use_grouped_phase_j=args.grouped,
               stub_nquad=args.stub_nquad, label=label,
               stub_level=args.stub_level)
    kinds = tuple(x for x in args.refs.split(',') if x)
    if kinds and not args.stub_nquad:
        compute_references(run_, kinds, ref_max_m=args.ref_max_m,
                           scope=args.ref_scope,
                           time_budget=args.ref_time_budget, verbose=True)
    print(format_report(run_))
    out = args.out or os.path.join(
        DEFAULT_OUT_DIR,
        f"{label}_k{args.k}_l{args.max_ell}"
        f"{'_grouped' if args.grouped else ''}"
        f"{'_stub' if args.stub_nquad else ''}_{time.strftime('%Y%m%d-%H%M%S')}")
    paths = save(run_, out)
    print('wrote', *paths)
    rows, _ = reconstruction_check(run_)
    return 0 if all(r['ok'] for r in rows) else 1


if __name__ == '__main__':
    sys.exit(main())
