"""
api.compute — single-call entry point for the full MSR-JD pipeline.

Wraps the notebook flow (cells 1–28 of hawkes_td_*.ipynb) into a
function that takes a model dict + parameter dict + run config, runs
the entire FT expansion → diagram enumeration → Phase J integration
chain, and returns a result dict (with optional .npz save).
"""
from __future__ import annotations

import contextlib
import contextvars
import os
import time
import warnings
from typing import Any

import math
import numpy as np
from sage.all import SR

# engine internals
from engine.core.field_theory import FieldTheory
from engine.core.vertices import (
    extract_vertex_types, extract_source_types, NoiseSourceType,
)
from engine.diagrams.type_assignment import build_field_index_map
from engine.diagrams.symmetry import classify_coefficient_factors
from engine.integration.time_domain.pipeline import (
    compute_correction_td,
    DiscardedDiagramsError,          # noqa: F401  (re-exported: daedalus)
)

# Pipeline-package helpers
from api._propagator import build_propagator, compute_poles_and_residues
from api._mean_field import solve_mean_field
from api._mean_field_dae import linear_stability
from api._diagrams  import enumerate_unique_diagrams
from engine.enumeration import prediagram_cache as _pd_cache
from api.save       import _phase_j_convention, _phase_j_flags
from api.access     import (
    MeanField, Parameters, normalize_external_fields, resolve_external_fields,
)
from engine.errors import DaedalusError

# Itô equal-time regularizer (magnitude).  Our SDEs use the Itô convention, which
# is LEFT-continuous (Θ(0)=Θ(0⁻)); the equal-time *observable* is therefore the
# LEFT limit.  Before Oct 2026 a τ=0 grid point was evaluated at −_ITO_EPS
# instead of exactly 0, so a coincident causal chamber survived the Θ(0)=0
# rule; Phase J's exact tie rule now does that at τ = 0 itself (the default,
# see ``_tau0_mode``; the −_ITO_EPS sample is DAEDALUS_TAU0=nudge).  (For
# the stationary/symmetric correlators here C is even, so the left limit equals the
# right; the sign only matters for a genuine τ=0 jump, where Itô takes the left.)
_ITO_EPS = 1e-6

# The k = 2 τ = 0 convention.  'exact':
# the τ = 0 point is evaluated AT equal times; Phase J's exact tie rule
# (``final_integral._const_row_verdict``: of Θ(t1 − t0) and Θ(t0 − t1) at
# t1 == t0 exactly the one with leg 1 earlier holds) gives the Itô left
# limit C(0⁻) exactly (<= 3e-14 against independent references on every
# route), and the exact delayed tree returns its own left limit.  Default
# since Oct 2026.  'nudge': the pre-Oct-2026 sample at τ = −_ITO_EPS, whose
# relative error is (rate × 1e-6) and therefore unit dependent (1e-3 at a
# rate of 1e3).  Environment variable DAEDALUS_TAU0 = 'exact' | 'nudge',
# read at call time.  ``dd.run``'s k ≥ 3 slices follow the same switch
# (since Oct 2026: 'exact' evaluates a tied point at the exact tie with the
# swept leg listed last, ``daedalus._kpoint_slice_point``; 'nudge' is the
# −_ITO_EPS placement of ``daedalus._kpoint_slice_times``).
# ``me_output.slice_me`` keeps the −_ITO_EPS placement.
_TAU0_MODES = ('exact', 'nudge')

# (model name, source, declared order) already warned about by
# compute_cumulants (CumulantTruncationWarning: once per process)
_CUMULANT_TRUNCATION_WARNED = set()


def _tau0_mode():
    """'exact' or 'nudge' (``DAEDALUS_TAU0``; see ``_TAU0_MODES``)."""
    from engine import _env
    return _env.choice('DAEDALUS_TAU0', _TAU0_MODES, 'exact')


def _tau0_lag():
    """The lag a k = 2 τ = 0 point is evaluated at: 0.0 ('exact') or
    −_ITO_EPS ('nudge').  'exact' uses Phase J's exact tie rule (the only
    constant-row rule since the pre-M1 'legacy_clip' mode was retired)."""
    return -_ITO_EPS if _tau0_mode() != 'exact' else 0.0


def _ito_nudge_callable(fn, k):
    """Wrap a user-facing C(τ) callable so its equal-time evaluation matches the
    Itô LEFT-limit used on the C_tau grid (see ``_ITO_EPS`` and the k=2 grid
    construction below).  Without this, ``total_C(t0, t0)`` would sample the
    measure-zero τ=0 step discontinuity and disagree with ``C_tau`` at τ=0.

    Only the k=2 single-axis slice has a well-defined "anchor leg" (leg 0); for
    other k the callable is returned unchanged (``daedalus.run`` places the
    k≥3 times itself: ``_kpoint_slice_times`` for the slices,
    ``_kpoint_pinned_time`` per leg for the full grid).  Mirrors the grid:
    a second-leg time within 1e-12 (relative) of the anchor leg's time is moved onto it
    (default, an exact tie) or to ``anchor − _ITO_EPS`` (DAEDALUS_TAU0=nudge).
    """
    if k != 2:
        return fn
    lag = _tau0_lag()
    # ``not (<=)`` rather than ``>``: a NaN second time compares False both
    # ways and must reach the engine as NaN (which refuses it), never be
    # moved to the equal-time left limit.  'exact': a second time within
    # 1e-12 of the first is moved ONTO it (an exact tie, decided by the
    # engine's tie rule)
    # (the tie is relative to the times themselves: an absolute 1e-12 tied
    # distinct times of a model in fast time units)
    # Only FINITE times can tie: with t1 = inf the relative threshold is itself
    # infinite and would snap inf onto tau = 0; non-finite times go to the
    # engine, which refuses them.
    def _tied(t0, t1):
        return (math.isfinite(t0) and math.isfinite(t1)
                and abs(t1 - t0) <= 1e-12 * max(abs(t0), abs(t1)))
    return lambda t0, t1: fn(t0, t0 + lag if _tied(t0, t1) else t1)


def _phase_j_warning_scope():
    """The scope that aggregates the Phase J hardened quadrature fallback's
    warnings over one evaluation (``final_integral.nquad_warning_scope``:
    one warning per model, source and loop order, with counts), or a null
    context for an engine without it."""
    import contextlib
    from engine.integration.time_domain import final_integral as _fi
    scope = getattr(_fi, 'nquad_warning_scope', None)
    return scope() if scope is not None else contextlib.nullcontext()


def _phase_j_scoped(fn):
    """``fn`` with every call inside ``_phase_j_warning_scope``."""
    def scoped(*args):
        with _phase_j_warning_scope():
            return fn(*args)
    return scoped


def _tau_extent(tau_grid, tau_max):
    """The largest |time difference| the run evaluates: the tau grid's
    extent, else ``tau_max`` (bounds the pole-cluster deformation radius)."""
    try:
        if tau_grid is not None and len(tau_grid):
            return float(np.max(np.abs(np.asarray(tau_grid, dtype=float))))
    except (TypeError, ValueError):
        pass
    return float(tau_max) if tau_max else None


def _ss_route_prediction(prop, num_params, vertex_types):
    """The run-level route prediction (``statespace.run_route``) plus
    ``bypass_deformation``: the run is covered by the state-space route and
    the propagator has a pole-cluster deformation, which then is not run
    (the route uses no poles or residues)."""
    from engine import _env
    if _env.phase_j_engine() == 'poles':
        return {'engine': 'poles', 'covers_run': False,
                'reason': 'engine_poles', 'bypass_deformation': False}
    from engine.integration.time_domain import statespace as _ssm
    pred = _ssm.run_route(prop, num_params, vertex_types)
    pred['bypass_deformation'] = bool(
        pred['covers_run'] and prop.get('pole_deformation') is not None)
    return pred


def _pole_cluster_preamble(prop, max_ell, ss_route=None):
    """The run's pole-cluster report (``None`` when no two poles are close)
    and the warning for a cluster the pipeline cannot resolve.  When
    the state-space route covers the run (``ss_route['covers_run']``) the
    poles are not used: status ``'bypassed'`` (the pole engine's status in
    ``status_poles``), no warning."""
    from api import _pole_clusters as _pc
    pc = prop.get('pole_clusters')
    if not pc:
        return None
    pc = dict(pc)
    if ss_route is not None and ss_route.get('covers_run'):
        pc['status_poles'] = pc['status']
        pc['reason_poles'] = pc.get('reason')
        pc['status'] = 'bypassed'
        pc['reason'] = ('the state-space Phase J route takes every subset of '
                        'the run; it uses no poles or residues')
        pc['accuracy'] = None
        return pc
    if pc['status'] == 'unresolved':
        digits = _pc.lost_digits(pc['kappa_max'], max_ell)
        pc['lost_digits'] = digits
        if digits <= _pc.LOST_DIGITS_WARN:
            return pc                     # recorded; the loss stays below 1e-6
        what = ('a repeated pole, which the pole-residue form cannot '
                'represent' if pc.get('repeated_pole') else
                f'near-degenerate poles (cancellation factor '
                f'{pc["kappa_max"]:.3g}, about {digits:.0f} digits lost at '
                f'max_ell={max_ell})')
        _pc.warn(f'the propagator has {what}; the pole-cluster deformation '
                 f'is {pc["status"]}: {pc["reason"]}.  The cumulant may be '
                 f'inaccurate (see result["pole_clusters"]).')
    return pc


class PhaseJIncompleteWarning(UserWarning):
    """``compute_cumulants`` left diagrams out: Phase J could not evaluate
    them, so the cumulant misses their contributions
    (``result['phase_j_incomplete']`` names them and the reason)."""


from engine.integration.time_domain.final_integral import (  # noqa: E402
    PhaseJResourceError, growth_scoped as _growth_scoped)


class SpatialAccuracyWarning(UserWarning):
    """A spatial result whose quadrature accuracy is not certified
    (``result['status'] == 'approximate'``; ``result['spatial_accuracy']``
    says why and gives the measured spread)."""


class SpatialAccuracyError(DaedalusError, RuntimeError):
    """Raised instead of :class:`SpatialAccuracyWarning` under
    ``policy={'mode': 'strict'}``; ``result`` is the approximate result."""

    def __init__(self, message, result=None):
        super().__init__(message)
        self.result = result


class MarcusTruncationWarning(UserWarning):
    """A Marcus-rule source (``SDE.interpretation('marcus')``) whose noise
    term may be off at the saddle by more than :data:`MARCUS_TRUNCATION_RTOL`
    in a vertex the run can use (``result['marcus_truncation']``).  On the
    series route the measure is the next term over the kept series; on the
    exact route (affine coupling) it is the rounding of the coefficients."""


#: relative size of the next Marcus term at the saddle above which
#: compute_cumulants warns (G6's default accuracy threshold)
MARCUS_TRUNCATION_RTOL = 1e-10


def _marcus_admissible(r, q, k, max_ell):
    """Can a noise vertex with ``r`` response and ``q`` physical
    (fluctuation) legs enter a ``k``-point ``max_ell``-loop diagram?  Two
    necessary conditions.  Loops: ``sum_v (r_v + q_v - 2) = k + 2*ell - 2``
    over the vertices (the xt-linear vertex vanishes at the saddle), so
    ``r + q <= k + 2*max_ell``.  Response legs: grading the action's
    ``r``-response vertices by ``hbar^(r-1)`` (``S/hbar`` is the loop
    expansion) makes an ``ell``-loop ``k``-point part ``hbar^(k-1+ell)``, so
    ``r <= k + max_ell``."""
    return r >= 1 and r + q <= k + 2 * max_ell and r <= k + max_ell


def _marcus_values(names, mf, par, i):
    """``{name: value}`` for the bare symbol names of a Marcus record at
    unit ``i``: a field ``x__i`` takes its saddle ``xstar[i]`` (``mf`` may be
    ``None``: before the mean-field solve), an indexed parameter its entry
    ``i``, a scalar parameter its value; names without a value are left
    out."""
    out = {}
    for nm in names:
        base, _, idx = nm.partition('__')
        val = None
        if idx:
            arr = (mf or {}).get(base + 'star', par.get(base))
            if arr is not None:
                a = np.atleast_1d(np.asarray(arr, dtype=complex))
                val = a[i] if a.size > i else None
        elif nm in par and np.ndim(par[nm]) == 0:
            val = par[nm]
        if val is not None:
            out[nm] = complex(val)
    return out


def _sr_num(v, field=None):
    v = complex(v)
    if field is None:
        return SR(v) if v.imag else SR(float(v.real))
    return SR(field(v.real, v.imag)) if v.imag else SR(field(v.real).real())


def _marcus_mgf_degree(rec, vals):
    """The exact route's mgf existence (``rec['mgf_bases']``): the smallest
    response degree ``n <= rec['order']`` whose vertices need the jump law's
    mgf at a point where it does not exist, with that point, or
    ``(None, None)``.  Degree ``n`` needs ``K`` on ``[n*min(s_f, 0),
    n*max(s_f, 0)]`` (``s_f`` the slopes).  The mgf of a law exists on an
    interval about 0 whose ends are singular points of its closed form
    (Widder), so it exists at ``s`` while no base changes sign or vanishes on
    ``[0, s]``."""
    from sage.all import CDF, fast_callable, log
    from api.noise import THETA, parse_expr
    U = SR.var('u__mgf')
    bases = [parse_expr(b) for b in rec.get('mgf_bases') or []]
    if not bases:
        return None, None
    sub = {SR.var(n): _sr_num(v) for n, v in vals.items()}
    slopes = [complex(parse_expr(s).subs(sub)).real for s in rec['slopes'].values()]
    hi, lo = max([0.0] + slopes), min([0.0] + slopes)
    fns = [SR(b).subs(sub) for b in bases]
    grid = np.linspace(0.0, 1.0, 2049)

    def exists(s):
        if s == 0.0:
            return True
        for f in fns:
            # a polynomial base in theta or in u = e^theta: its real roots on the
            # segment (an even-order root touches 0 without a sign change, which
            # the grid scan below cannot see)
            for var, g, a_, b_ in ((THETA, f, 0.0, s), (U, f.subs({THETA: log(U)}), 1.0, math.exp(s))):
                if not g.is_polynomial(var) or not g.has(var):
                    continue
                c = [complex(q) for q in reversed(g.coefficients(var, sparse=False))]
                # a k-fold root splits by ~eps^(1/k): a loose imaginary tolerance
                # (a complex pair that close to the segment is a near-pole anyway)
                if any(abs(r.imag) <= 1e-3 * max(1.0, abs(r))
                       and min(a_, b_) - 1e-12 <= r.real <= max(a_, b_) + 1e-12
                       for r in np.roots(c)):
                    return False
            try:
                fc = fast_callable(f, vars=[THETA], domain=CDF)
                vs = np.array([complex(fc(float(s * t))) for t in grid])
            except (ValueError, ZeroDivisionError, ArithmeticError, TypeError):
                return False
            re0 = vs[0].real
            if not np.all(np.isfinite(vs)) or re0 == 0.0 or np.any(vs.real * re0 <= 0.0):
                return False
        return True
    for n in range(1, int(rec['order']) + 1):
        for s in (n * hi, n * lo):
            if not exists(s):
                return n, s
    return None, None


def _marcus_refuse_mgf(name, i, n, s, k, max_ell):
    legs = 'leg' if n == 1 else 'legs'
    fix = ('the mean drift itself is infinite; change the parameters' if n == 1 else
           f'keep k + max_ell below {n}, or change the parameters')
    raise ValueError(
        f"Marcus source {name!r} (unit {i}): degree {n} -- the vertices with {n} "
        f"response {legs} of its exact (affine-flow) noise term need the jump "
        f"law's mgf at theta = {s:.6g}, where the mgf does not exist, so their "
        f"coefficients (rate * E[(e^(b z) - 1)^{n}]) are infinite, and with them "
        f"the stationary moments of order {n} and above. k = {k}, max_ell = "
        f"{max_ell} can use vertices with up to k + max_ell = {k + max_ell} "
        f"response legs; {fix}.")


def _marcus_precheck(model, par, k, max_ell):
    """Refuse, before the mean-field solve, an exact-route Marcus source
    whose jump law has no mgf where a usable vertex needs it -- for the
    sources whose slopes and law read only scalar parameters (the rest are
    checked at the saddle, after the run)."""
    series = model.get('marcus_series') if isinstance(model, dict) else None
    from api.noise import parse_expr
    for name, rec in (series or {}).items():
        if rec.get('route') != 'exact':
            continue
        names = set()
        for e in list(rec['slopes'].values()) + list(rec.get('mgf_bases') or []):
            names |= {str(v) for v in parse_expr(e).variables()} - {'theta'}
        vals = _marcus_values(names, None, par, 0)
        if set(vals) != names or any('__' in n for n in names):
            continue
        n, s = _marcus_mgf_degree(rec, vals)
        if n is not None and n <= k + max_ell:
            _marcus_refuse_mgf(name, 0, n, s, k, max_ell)


def _marcus_truncation(model, result, k, max_ell):
    """How far each Marcus noise term (``model['marcus_series']``) may be
    from exact at the saddle, per source and population unit, over the
    noise vertices the run can use (:func:`_marcus_admissible`; ``by
    response degree r`` below means the largest over the admissible physical
    degrees).  ``None`` when the model has no Marcus term.

    Series route: each term expanded about the saddle into vertices
    ``xt^r dx^q``; ``relative_by_degree[r]`` is ``sum |coefficient|`` of the
    first omitted term's admissible ``(r, q)`` vertices over the same for
    the kept series (the largest ratio over ``q``).  Exact route: the
    vertices are exact, so the measure is rounding -- the relative error of
    each kept coefficient evaluated in double precision against 212 bits;
    the smallest degree whose vertices need the mgf where it does not exist
    is ``mgf_degree_limit`` (refused when a usable vertex needs it).
    ``relative`` is the largest ratio; ``status`` ``'above'`` past
    :data:`MARCUS_TRUNCATION_RTOL`."""
    series = model.get('marcus_series') if isinstance(model, dict) else None
    if not series or not isinstance(result, dict):
        return None
    from api.noise import parse_expr
    mf = result.get('mf_values') or {}
    par = dict((result.get('config') or {}).get('fundamental') or {})
    out = []
    for name, rec in series.items():
        n_units = len(np.atleast_1d(mf.get(rec['responses'][0][:-1] + 'star', [0])))
        exact = rec.get('route') == 'exact'
        for i in range(n_units):
            row = {'source': name, 'unit': i, 'order': int(rec['order']),
                   'route': rec.get('route', 'series'),
                   'admissible': {'max_response_legs': int(k) + int(max_ell),
                                  'max_legs': int(k) + 2 * int(max_ell)},
                   'relative': None}
            if exact:
                _marcus_exact_row(row, name, rec, mf, par, i, k, max_ell)
            else:
                _marcus_series_row(row, rec, mf, par, i, k, max_ell)
            out.append(row)
    return out


def _marcus_detail(row):
    """One line on a ``result['marcus_truncation']`` row (G6, the warning)."""
    if row.get('route') == 'exact':
        return (f"Marcus source {row['source']!r} (unit {row['unit']}), exact affine-flow "
                f"route to {row['order']} response legs: its usable coefficients lose "
                f"{row['relative']:.2e} to rounding in double precision at the saddle")
    return (f"Marcus series of source {row['source']!r} (unit {row['unit']}) kept to "
            f"order {row['order']}: the first omitted term is {row['relative']:.2e} of "
            f"the kept series at the saddle, in the vertices the run can use")


def _marcus_remedy(row):
    if row.get('route') == 'exact':
        return ("the cancellation comes from a small slope times jump size; the series "
                "converges fast there: SDE.marcus_route('series')")
    return 'raise SDE.series_order'


def _rational_literals(text):
    """``text`` with each decimal literal (``0.1``, ``2.5e-3``) written as the
    exact fraction of its digits."""
    import re
    from fractions import Fraction

    def frac(m):
        f = Fraction(m.group(0))
        return f'({f.numerator}/{f.denominator})'
    return re.sub(r'(?<![\w.])\d+\.\d*(?:[eE][-+]?\d+)?|(?<![\w.])\d+[eE][-+]?\d+', frac, text)


def _marcus_exact_row(row, name, rec, mf, par, i, k, max_ell):
    from sage.all import ComplexField, RealField
    from api.noise import parse_expr
    coefs = {key: parse_expr(t) for key, t in rec['coefficients'].items()}
    # the 212-bit reference reads the decimal literals as exact rationals (a
    # 53-bit literal would cap the reference at 53 bits)
    exact = {key: parse_expr(_rational_literals(t)) for key, t in rec['coefficients'].items()}
    names = set()
    for e in list(coefs.values()) + [parse_expr(s) for s in rec['slopes'].values()] + \
            [parse_expr(b) for b in rec.get('mgf_bases') or []]:
        names |= {str(v) for v in SR(e).variables()} - {'theta'}
    vals = _marcus_values(names, mf, par, i)
    missing = sorted(names - set(vals))
    row['mgf_degree_limit'] = None
    if missing:
        row['status'] = 'unavailable: no values for ' + ', '.join(missing)
        return
    n0, s0 = _marcus_mgf_degree(rec, vals)
    row['mgf_degree_limit'] = n0
    if n0 is not None and n0 <= k + max_ell:
        _marcus_refuse_mgf(name, i, n0, s0, k, max_ell)
    RF, CF = RealField(212), ComplexField(212)
    sub_d = {SR.var(n): _sr_num(v) for n, v in vals.items()}
    sub_h = {SR.var(n): _sr_num(v, RF) for n, v in vals.items()}
    by = {}
    for key, P in coefs.items():
        r = sum(int(q) for q in key.split(','))
        if not _marcus_admissible(r, 0, k, max_ell):
            continue
        hp = CF(SR(exact[key]).subs(sub_h).n(212))
        try:
            d = complex(SR(P).subs(sub_d))
            rel = float(abs(CF(d) - hp) / abs(hp)) if hp != 0 else (0.0 if d == 0 else float('inf'))
        except (ValueError, ZeroDivisionError, ArithmeticError, TypeError, RuntimeError):
            rel = float('inf')
        if not math.isfinite(rel):
            rel = float('inf')
        by[r] = max(by.get(r, 0.0), rel)
    row['relative_by_degree'] = dict(sorted(by.items()))
    row['relative'] = max(by.values(), default=0.0)
    row['status'] = 'above' if row['relative'] > MARCUS_TRUNCATION_RTOL else 'ok'


def _marcus_series_row(row, rec, mf, par, i, k, max_ell):
    from sage.all import CDF, PolynomialRing
    from api.noise import parse_expr
    resp = [SR.var(f'{r}__i') for r in rec['responses']]
    kept = parse_expr(rec['kept'])
    nxt = None if rec.get('next') is None else parse_expr(rec['next'])
    fields = [SR.var(f'{f}__i') for f in rec.get('fields') or []]
    names = set()
    for e in (kept,) + ((nxt,) if nxt is not None else ()):
        names |= {str(v) for v in e.variables()}
    names -= {str(r) for r in resp}
    vals = _marcus_values(names, mf, par, i)
    missing = sorted(names - set(vals))
    if missing or nxt is None:
        row['status'] = ('unavailable: no values for ' + ', '.join(missing)
                         if missing else 'unavailable: the source has no cumulant '
                         'of the next order (a moment-series jump law)')
        return
    fld = [x for x in fields if str(x) in vals]
    sub = {SR.var(n): _sr_num(v) for n, v in vals.items() if SR.var(n) not in fld}
    cap = int(k) + 2 * int(max_ell)
    dx = [SR.var(f'd{x}') for x in fld]
    shift = {x: _sr_num(vals[str(x)]) + d for x, d in zip(fld, dx)}
    gens = [str(r) for r in resp] + [str(d) for d in dx]
    R = PolynomialRing(CDF, gens + (['zpad__'] if len(gens) == 1 else []))
    nr = len(resp)

    def size(e):
        """sum |coefficient| per admissible (r, q) vertex about the saddle"""
        e = SR(e).subs(sub)
        try:
            poly = R(e.subs(shift).expand())
        except (TypeError, ValueError):
            poly = None
        by = {}
        if poly is not None:
            items = poly.dict().items()
        else:                               # non-polynomial in the fields: Taylor about the saddle
            t = SR.var('t__marcus')
            ser = e.subs({x: _sr_num(vals[str(x)]) + t * d for x, d in zip(fld, dx)})
            ser = ser.series(t, cap + 1).truncate().subs({t: 1}).expand()
            items = R(ser).dict().items()
        for mono, c in items:
            mono = tuple(int(q) for q in mono)
            r, q = sum(mono[:nr]), sum(mono[nr:len(gens)])
            if _marcus_admissible(r, q, k, max_ell):
                by[(r, q)] = by.get((r, q), 0.0) + abs(complex(c))
        return by
    kb, nb = size(kept), size(nxt)
    rel = {}
    for rq in set(kb) | set(nb):
        ratio = (nb.get(rq, 0.0) / kb[rq] if kb.get(rq) else
                 (0.0 if not nb.get(rq) else float('inf')))
        rel[rq[0]] = max(rel.get(rq[0], 0.0), ratio)
    row['kept_abs'], row['next_abs'] = sum(kb.values()), sum(nb.values())
    row['relative_by_degree'] = dict(sorted(rel.items()))
    row['relative'] = max(rel.values(), default=0.0)
    row['status'] = 'above' if row['relative'] > MARCUS_TRUNCATION_RTOL else 'ok'


def _spatial_accuracy(result, tau_grid, max_ell):
    """The reasons a spatial result is approximate (empty: certified as far
    as the integrator knows).  Reads the drivers' records in
    ``result['spatial_info']``: derivative-vertex loops (the product grid
    converges ~n^-2 around the integrable loop-shrink singularity; the
    measured 0.7x-coarser spread), equal-time diagrams the radial-simplex
    rule could not take (above its dimension: the grid) or did not converge
    to 1e-6, and coupled loops at equal times (the spectral driver keeps the
    product grid there)."""
    info = result.get('spatial_info') if isinstance(result, dict) else None
    if not isinstance(info, dict):
        return []
    out = []
    da = info.get('derivative_accuracy')
    if da:
        out.append({'reason': 'derivative_vertex_loop', 'note': da['note'],
                    'coarse_spread': da.get('coarse_spread')})
    eq = info.get('equal_time')
    if eq and (eq.get('grid_calls') or (
            eq.get('unconverged') and (eq.get('max_rel_change') or 0) > 1e-6)):
        out.append({'reason': 'equal_time_grid',
                    'note': eq.get('grid_note') or (
                        'radial-simplex rule not converged to 1e-6 '
                        f'(max successive-order change '
                        f'{eq.get("max_rel_change"):.2e})'),
                    'coarse_spread': None})
    if info.get('integrator') in ('mc', 'bessel'):
        out.append({'reason': 'monte_carlo_integrator',
                    'note': (f"SPATIAL_INTEGRATOR={info['integrator']}: a "
                             'Monte-Carlo estimate (statistical error; bessel '
                             'is biased at tau != 0)'),
                    'coarse_spread': None})
    pts = result.get('points') if isinstance(result, dict) else None
    if info.get('coupled') and pts is not None and np.size(pts):
        # coupled k-point: the spectral driver keeps the product grid, which
        # converges only algebraically at equal-time events
        P_ = np.asarray(pts, dtype=float)
        if np.any(np.all(P_[..., 1] == 0.0, axis=-1)):
            out.append({'reason': 'equal_time_coupled_loop',
                        'note': ('coupled k-point events at equal times use '
                                 'the product grid (algebraic convergence; '
                                 'not certified)'),
                        'coarse_spread': None})
    taus = np.asarray(tau_grid if tau_grid is not None else [], dtype=float)
    if (info.get('coupled') and pts is None and max_ell >= 1 and taus.size
            and np.any(taus == 0.0)):
        out.append({'reason': 'equal_time_coupled_loop',
                    'note': ('coupled loops at tau = 0 use the product grid '
                             '(algebraic convergence at equal times; ~1e-5 '
                             'relative at the default grid on one-loop '
                             'diagrams; not certified)'),
                    'coarse_spread': None})
    return out


class PhaseJIncompleteError(DaedalusError, RuntimeError):
    """Raised instead of :class:`PhaseJIncompleteWarning` under
    ``policy={'mode': 'strict'}``.  ``result``: the (incomplete) result
    dict; ``failures``: ``{ell: [failure record, ...]}``."""

    def __init__(self, message, result=None, failures=None):
        super().__init__(message)
        self.result = result
        self.failures = failures or {}


def _phase_j_failures(phase_j_by_ell):
    """``{ell: [{'diagram', 'n_diagrams', 'reason'}, ...]}`` for every
    diagram that Phase J could not evaluate and therefore left out of
    ``C_tau``; ``{}`` when none.  Reads the ``skipped_kernel_ids`` of each
    loop order (ints with the reason in ``groups``; dicts in results of
    the retired grouped route), so every route reports through the same
    field."""
    out = {}
    for ell, pj in sorted((phase_j_by_ell or {}).items()):
        if not isinstance(pj, dict):
            continue
        reasons = {g.get('kernel_id'): g.get('reason')
                   for g in (pj.get('groups') or [])
                   if g.get('handled_by') == 'skipped'}
        recs = []
        for sk in pj.get('skipped_kernel_ids') or []:
            if isinstance(sk, dict):
                recs.append({
                    'diagram': sk.get('kernel_id', sk.get('group_idx')),
                    'n_diagrams': sk.get('n_diagrams', 1),
                    'reason': str(sk.get('reason', '')),
                })
            else:
                recs.append({'diagram': sk, 'n_diagrams': 1,
                             'reason': str(reasons.get(sk, ''))})
        if recs:
            out[ell] = recs
    return out


def _entry_int(name, v):
    """``v`` as an ``int``: an int / numpy integer (not a bool); anything
    else raises a ``TypeError`` naming the argument."""
    import operator
    if not isinstance(v, (bool, np.bool_)):
        try:
            return operator.index(v)      # int, numpy and Sage integers
        except TypeError:
            pass
    raise TypeError(f'{name} must be an integer; got {v!r} '
                    f'({type(v).__name__}).')


def as_model_dict(model):
    """The model dict of ``model``: a dict passes; an unbuilt builder (an
    ``api.sde.SDE`` or a ``ModelBuilder``/``TemporalModelBuilder``, anything
    with a ``build()``) is built, as ``dd.run`` does.  Anything else is a
    ``TypeError`` that says what was given."""
    if isinstance(model, dict):
        return model
    build = getattr(model, 'build', None)
    if callable(build):
        built = build()
        if isinstance(built, dict):
            return built
    raise TypeError(
        f'model must be a model dict (the result of .build()) or a builder '
        f'with a build() method (api.sde.SDE, ModelBuilder); got '
        f'{type(model).__name__}.')


def _is_count(v):
    import operator
    if isinstance(v, (bool, np.bool_)):
        return False
    try:
        operator.index(v)
        return True
    except TypeError:
        return False


def _policy_is_strict(policy):
    mode = (policy.get('mode') if isinstance(policy, dict)
            else getattr(policy, 'mode', None))
    return mode == 'strict'


def _phase_j_failure_message(failures, model_name):
    n = sum(r['n_diagrams'] for recs in failures.values() for r in recs)
    lines = [f'compute_cumulants({model_name!r}): Phase J could not evaluate '
             f'{n} diagram(s); C_tau is missing them (all of a diagram, or '
             f'its part at the pole-deformation nodes where it failed; '
             f'result["status"] = "incomplete", details in '
             f'result["phase_j_incomplete"]):']
    for ell, recs in failures.items():
        by_reason = {}
        for r in recs:
            by_reason.setdefault(r['reason'], []).append(r['diagram'])
        for reason, ids in by_reason.items():
            shown = ', '.join(map(str, ids[:12])) + (
                ', ...' if len(ids) > 12 else '')
            lines.append(f'  ell={ell}: diagram(s) {shown}: {reason[:300]}')
    return '\n'.join(lines)


def _real_inputs(num_params, diagram_records):
    """Real parameters and diagram prefactors: with a real propagator the
    Phase J result at conj(w) is conj of the one at w."""
    from sage.all import CDF
    try:
        for v in (num_params or {}).values():
            if complex(CDF(SR(v))).imag != 0:
                return False
        for r in diagram_records:
            pref = SR(r['combined_prefactor']).subs(num_params or {})
            if complex(CDF(pref)).imag != 0:
                return False
    except Exception:
        return False
    return True


def _combine_phase_j(results, nodes):
    """One ell's Phase J results of the deformation nodes, combined: the
    callables, the per-diagram contributions and the delta records are the
    node-weighted sums; ``pole_deformation_nodes`` lists the nodes."""
    from api import _pole_clusters as _pc
    if all(r is None for r in results):
        return None
    out = dict(results[0])
    out['total_C'] = _pc.combine_callables(
        [r['total_C'] for r in results], nodes)
    fns = [r.get('total_C_batch') for r in results]
    if all(f is not None for f in fns):
        def total_C_batch(tau_points, *a, **kw):
            vals = [np.asarray(f(tau_points, *a, **kw), dtype=complex)
                    for f in fns]
            return list(_pc.combine_values(vals, nodes))
        out['total_C_batch'] = total_C_batch
    epd = [r.get('eval_per_diagram_batch') for r in results]
    if all(f is not None for f in epd):
        def eval_per_diagram_batch(tau_points, *a, **kw):
            outs = [f(tau_points, *a, **kw) for f in epd]
            vals = [np.asarray(o[0], dtype=complex) for o in outs]
            return _pc.combine_values(vals, nodes).tolist(), outs[0][1]
        out['eval_per_diagram_batch'] = eval_per_diagram_batch
    groups = []
    for gs in zip(*[r.get('groups') or [] for r in results]):
        g = dict(gs[0])
        if all(x.get('contribution') is not None for x in gs):
            g['contribution'] = _pc.combine_callables(
                [x['contribution'] for x in gs], nodes)
        groups.append(g)
    out['groups'] = groups
    # A diagram that failed at ANY node is missing from that node's total:
    # report the union (the per-node records are in node order).
    sk_all, seen = [], set()
    for r in results:
        for sk in (r or {}).get('skipped_kernel_ids') or []:
            # by diagram / group id: the reason text may differ by node
            key = ((sk.get('kernel_id'), sk.get('group_idx'))
                   if isinstance(sk, dict) else sk)
            if key not in seen:
                seen.add(key)
                sk_all.append(sk)
    out['skipped_kernel_ids'] = sk_all
    for g, gs in zip(groups, zip(*[r.get('groups') or [] for r in results])):
        bad = [x for x in gs if x.get('handled_by') == 'skipped']
        if bad and g.get('handled_by') != 'skipped':
            g['handled_by'] = 'skipped'
            g['reason'] = bad[0].get('reason')
            g['contribution'] = None
    for key in ('delta_contributions', 'delta_contributions_multi'):
        if key not in out:
            continue
        dcs = []
        for r, (w, wt, paired) in zip(results, nodes):
            for rec in r.get(key) or []:
                rec = dict(rec)
                f = rec.get('coeff_fc')
                if callable(f):
                    rec['coeff_fc'] = _pc.combine_callables([f],
                                                            [(w, wt, paired)])
                rec['pole_deformation_node'] = w
                dcs.append(rec)
        out[key] = dcs
    out['pole_deformation_nodes'] = [dict(w=w, weight=wt, paired=paired)
                                     for w, wt, paired in nodes]
    return out


#: ``discard='auto'`` with the pre-pass estimate under the budget: the run
#: switches to evaluate-and-discard at the first diagram boundary where its
#: measured growth (``final_integral.run_growth``) exceeds this fraction of
#: the budget; the growth guard refuses at 1.
DISCARD_AUTO_GROWTH_FRACTION = 0.5

#: What a k >= 3 run without fixed evaluation points needs for
#: evaluate-and-discard (pre-pass and growth-guard messages).
_DISCARD_POINTS_HINT = (
    'needs the evaluation points up front: points= (the external-time '
    'tuples to evaluate; dd.run passes its slice points, tau_grid=[0.0] '
    'is the equal-time point).')


def _phase_j_memory_preflight(diagram_records, propagator_data, num_params,
                              prop, model, k, max_ell, verbose, preflight,
                              ss_route=None, n_tau=1, run_budget=None,
                              discard=False, discard_ok=False,
                              discard_hint=None):
    """The estimate of the modesum-plan memory of the whole Phase J
    run (``final_integral.estimate_modesum_plans`` over every diagram of
    every ell, times the pole-deformation node runs), recorded in the
    analyzer report (``diagnostics['phase_j_memory']`` and an assertion in
    the ledger) and returned for ``result['phase_j_memory']``.  The estimate
    is an upper bound (δ-subsets that build no plan are counted).  More
    than ``PHASE_J_PREPASS_REFUSE_FACTOR`` times over the budget
    ``PHASE_J_MEM_BUDGET_GB`` it raises ``PhaseJResourceError`` before any
    plan is built, under every policy; between 1x and that factor the
    running guard decides at each allocation.

    The state-space route (``ss_route``: the run's route prediction, ``_ss_route_prediction``):
    the diagrams the state-space route takes are charged its estimate
    (``statespace.diagram_estimate``: stationary states x the τ batch, and
    the largest segment working set) instead of pole tuples, and a run the
    route covers makes one Phase J pass (no deformation runs).

    Evaluate-and-discard (``discard``: the caller's option; ``discard_ok``: the run has
    fixed evaluation points and one pass): with evaluate-and-discard the
    run holds one diagram at a time, so the largest diagram plus one
    segment working set (``held_bytes``) is compared with the budget
    instead of the sum; ``'auto'`` turns it on exactly when the sum is over
    the budget and that fits.  ``est['discard']`` says whether it is on."""
    from engine.integration.time_domain import final_integral as _fi
    if not hasattr(_fi, 'estimate_modesum_plans'):
        return None             # an A/B harness running an older engine
    deform = prop.get('pole_deformation')
    if ss_route is not None and ss_route.get('bypass_deformation'):
        deform = None
    ss_est = None
    if ss_route is not None and ss_route.get('covers_run'):
        from engine.integration.time_domain import statespace as _ssm
        n_perm = math.factorial(max(1, int(k)))

        def ss_est(td, has_delta):
            return _ssm.diagram_estimate(td, propagator_data, num_params,
                                         has_delta, n_tau=n_tau,
                                         n_perm=n_perm)
    # the node runs ``_phase_j_deformed`` makes: at most ``n_nodes`` (fewer
    # for a real system, which costs a scan of every prefactor to decide;
    # done only when the bound is over budget)
    runs = int(getattr(deform, 'n_nodes', 1) or 1) if deform is not None \
        else 1
    # ``run_budget``: the run's ``(bytes, rule)``, read once
    budget, budget_rule = (run_budget if run_budget is not None
                           else _fi.mem_budget('PHASE_J_MEM_BUDGET_GB'))
    est = _fi.estimate_modesum_plans(
        [r['typed_diagram'] for r in diagram_records], propagator_data,
        num_params, prefactors=[r['combined_prefactor']
                                for r in diagram_records],
        budget_bytes=None if budget is None else budget / runs,
        **({} if ss_est is None else {'ss_estimate': ss_est}))
    if est is None:
        return None
    if (deform is not None and budget is not None
            and est['bytes'] * runs > budget):
        try:
            runs = len(deform.nodes(deform.real_system and _real_inputs(
                num_params, diagram_records)))
        except Exception:
            pass
    w = est['worst']
    worst_rec = diagram_records[w['index']] if w else None
    est.update(node_runs=runs, bytes=est['bytes'] * runs,
               tuples=est['tuples'] * runs,
               tuples_unpruned=est['tuples_unpruned'] * runs,
               budget_bytes=budget, budget_rule=budget_rule,
               worst_ell=worst_rec['ell'] if worst_rec else None,
               worst_index_in_ell=(
                   sum(1 for r in diagram_records[:w['index']]
                       if r['ell'] == worst_rec['ell'])
                   if worst_rec else None))
    factor = _fi.PHASE_J_PREPASS_REFUSE_FACTOR
    # one diagram at a time, plus the diagrams that keep their charge for
    # their shot-noise δ records, plus one segment working set
    held = min(est['bytes'] + est.get('ss_transient_bytes', 0),
               est.get('max_diagram_bytes', est['bytes'])
               + est.get('delta_record_bytes', 0)
               + est.get('ss_transient_bytes', 0))
    if discard == 'auto':
        on = bool(discard_ok and runs == 1 and budget is not None
                  and est['bytes'] > budget and held <= budget)
    else:
        on = bool(discard) and discard_ok and runs == 1
    est['discard'] = on
    est['held_bytes'] = held if on else est['bytes']
    charged = est['held_bytes']
    above = budget is not None and charged > budget
    over = budget is not None and charged > factor * budget
    est['above_budget'] = above
    detail = (f"modesum plans: <= {est['tuples']:,} pole tuples "
              f"(~{est['bytes'] / 1e9:.2f} GB; "
              f"{est['tuples_unpruned']:,} without the exact-zero pruning) "
              f"in {est['n_diagrams']} diagrams"
              + (f" x {runs} deformation runs" if runs > 1 else '')
              + (f"; state-space route: {est['ss_diagrams']} diagrams, "
                 f"<= {est['ss_bytes'] / 1e9:.3g} GB of stationary states "
                 f"+ {est['ss_transient_bytes'] / 1e9:.3g} GB of segment "
                 f"working set" if est.get('ss_diagrams') else '')
              + (f"; budget {budget / 1e9:.3g} GB ({budget_rule})"
                 if budget is not None else f'; no budget ({budget_rule})')
              + (f"; evaluate-and-discard: one diagram at a time, <= "
                 f"{held / 1e9:.3g} GB held (the largest diagram + one "
                 f"segment working set)" if on else '')
              + ('; above the budget: the plans are checked as they are '
                 'built' if above and not over else ''))
    if preflight is not None:
        preflight.diagnostics['phase_j_memory'] = dict(est)
        preflight.ctx.ledger_add('assertion', 'phase_j_memory_budget',
                                 detail, satisfied=not over)
    if verbose:
        print(f'  [memory] {detail}')
    if over:
        ram = _fi._ram_bytes()
        machine = (f"  This machine has {ram / 1e9:.1f} GB of RAM; the "
                   f"estimate is {charged / 1e9:.2f} GB "
                   f"({charged / ram:.0%} of it)." if ram else
                   f"  The estimate is {charged / 1e9:.2f} GB (this "
                   f"machine's RAM is unknown).")
        if not on and runs == 1 and held <= budget and (discard_ok
                                                        or discard_hint):
            machine += ("  The largest diagram alone fits: discard=True "
                        "(or 'auto') evaluates one diagram at a time."
                        if discard_ok else
                        f"  The largest diagram alone fits: discard=True "
                        f"evaluates one diagram at a time; at k >= 3 it "
                        f"{discard_hint}")
        if not w or not w['edges']:
            raise _fi.PhaseJResourceError(
                f"Phase J memory estimate more than {factor}x over budget "
                f"for {model.get('name', '<unnamed>')} at k={k}, "
                f"max_ell={max_ell}: {detail}.{machine}  Nothing was built.  "
                f"Options: {_fi._BUDGET_HOW}.", est)
        edges = ', '.join(
            f"{n}{'' if n == a else f'/{a}'}{'+delta' if dl else ''}"
            for (n, a, dl) in w['edges'])
        raise _fi.PhaseJResourceError(
            f"Phase J memory estimate more than {factor}x over budget for "
            f"{model.get('name', '<unnamed>')} at k={k}, max_ell={max_ell}: "
            f"{detail}.  Largest diagram: loop order {est['worst_ell']}, "
            f"#{est['worst_index_in_ell']} of that order, {w['tuples']:,} "
            f"tuples ({w['tuples_unpruned']:,} unpruned) over "
            f"{len(w['edges'])} propagator edges with [{edges}] modes "
            f"(nonzero/all).{machine}  Nothing was built.  Options: "
            f"{_fi._BUDGET_HOW}.", est)
    return est


def _phase_j_ss_bypass(phase_j_pass, propagator_data, engine_report):
    """One Phase J pass without the pole-cluster deformation, for a run
    the state-space route is predicted to cover.  Every subset must then
    take the route: a subset that declines it (its rows) or falls back on
    the pole route (which would use the undeformed poles and residues)
    makes this return ``None`` (the caller runs the deformation instead;
    the memory charged by this pass is taken back).  The pass's propagator
    data carry ``ss_no_pole_fallback``, so a fallback raises: in this pass
    (caught here) and when the returned callables are evaluated later."""
    from engine.integration.time_domain import final_integral as _fi
    from engine.integration.time_domain import statespace as _ssm
    C = _ssm._counters()
    before = (C.get('ss_subsets_declined', 0), C.get('ss_fallback_evals', 0))
    # a failed bypass leaves no trace in the run's counters (its pass is
    # discarded): restored below
    snap = {k_: (dict(v) if isinstance(v, dict) else v)
            for k_, v in C.items()}
    with _fi._PLAN_MEM_LOCK:
        mem0 = {k_: _fi._PLAN_MEM[k_] for k_ in ('used', 'tuples', 'plans')}
    why = None
    try:
        out = phase_j_pass(dict(
            propagator_data, ss_no_pole_fallback=(
                'the run skipped the pole-cluster deformation (the '
                'state-space route covers it), so the undeformed poles of '
                'this propagator cannot serve a pole-route fallback; set '
                'DAEDALUS_PHASE_J_ENGINE=poles for this model')))
    except (_ssm.SSNoPoleFallback, _ssm.SSLateRefusal) as exc:
        # a fallback refused in the pass (SSLateRefusal: with
        # DAEDALUS_PHASE_J_SS_FALLBACK=0); anything else propagates
        out, why = None, f'{type(exc).__name__}: {str(exc)[:200]}'
    after = (C.get('ss_subsets_declined', 0), C.get('ss_fallback_evals', 0))
    if out is not None and after != before:
        out, why = None, (f'{after[0] - before[0]} subset(s) declined the '
                          f'state-space route, {after[1] - before[1]} '
                          f'fallback evaluation(s)')
    if out is not None:
        # δ records of m = 0 subsets with smooth rows take their factors from
        # the pole-residue G(t) (not routed): with undeformed poles they
        # would be the clustered ones
        for res_ell in (out[0] or {}).values():
            for key in ('delta_contributions', 'delta_contributions_multi'):
                for rec in (res_ell or {}).get(key) or ():
                    if not rec.get('integrated') and rec.get(
                            'retardation_data'):
                        out, why = None, ('a delta record with smooth rows '
                                          'outside the state-space route')
                        break
                if out is None:
                    break
            if out is None:
                break
    if out is None:
        with _fi._PLAN_MEM_LOCK:
            if _fi._PLAN_MEM['active'] > 0:
                _fi._PLAN_MEM.update(mem0)
        for k_ in list(C):
            v = snap.get(k_)
            if isinstance(C[k_], dict):      # in place: callers hold them
                C[k_].clear()
                C[k_].update(v or {})
            else:
                C[k_] = 0 if v is None else v
        engine_report['pole_deformation'] = 'used'
        engine_report['bypass_failed'] = why
    return out


def _phase_j_deformed(phase_j_pass, propagator_data, deform, diagram_records,
                      num_params, tau_points, report):
    """Phase J for a propagator with resolved pole clusters: one pass per
    deformation node (``api._pole_clusters.PoleDeformation``), combined by
    the trapezoid rule on the unit circle; ``report`` gets the node count
    and the accuracy estimate (discrete Fourier tail, relative to the
    largest |C| on the grid), and a warning is issued above
    ``ACCURACY_WARN``."""
    from api import _pole_clusters as _pc
    symmetric = deform.real_system and _real_inputs(num_params,
                                                    diagram_records)
    nodes = deform.nodes(symmetric)
    runs = []
    for w, _, _ in nodes:
        pv, cm = deform.propagator_at(w)
        runs.append(phase_j_pass(dict(propagator_data, pole_vals=pv,
                                      C_mats=cm)))
    phase_j_by_ell, total_C_by_ell, C_tau_by_ell = {}, {}, {}
    acc_ell, err_tot, node_tot = {}, None, None
    for ell in runs[0][1]:
        total_C_by_ell[ell] = _pc.combine_callables(
            [r[1][ell] for r in runs], nodes)
        phase_j_by_ell[ell] = _combine_phase_j([r[0][ell] for r in runs],
                                               nodes)
        if tau_points is None:
            C_tau_by_ell[ell] = None
            continue
        vals = [np.asarray(r[2][ell], dtype=complex) for r in runs]
        C_tau_by_ell[ell] = _pc.combine_values(vals, nodes)
        err = _pc.tail_estimate(vals, nodes, deform.n_nodes)
        scale = float(np.max(np.abs(C_tau_by_ell[ell]))) if vals[0].size else 0
        acc_ell[ell] = (float(np.max(err)) / scale if scale > 0
                        else float(np.max(err)) if err is not None else None)
        node_tot = vals if node_tot is None else [a + b for a, b in
                                                  zip(node_tot, vals)]
    report.update(n_nodes=deform.n_nodes, node_runs=len(nodes),
                  symmetric=bool(symmetric))
    if tau_points is None or node_tot is None:
        report.update(accuracy=deform.a_priori_error, accuracy_source='a priori',
                      accuracy_by_ell=None, accuracy_by_tau=None,
                      accuracy_l1=deform.a_priori_error)
    else:
        err = _pc.tail_estimate(node_tot, nodes, deform.n_nodes)
        C = _pc.combine_values(node_tot, nodes)
        absC = np.abs(C)
        scale = float(np.max(absC)) if absC.size else 0.0
        acc = float(np.max(err)) / scale if scale > 0 else float(np.max(err))
        with np.errstate(divide='ignore', invalid='ignore'):
            by_tau = np.where(absC > 0, err / np.where(absC > 0, absC, 1),
                              np.where(err > 0, np.inf, 0.0))
        l1 = float(np.sum(err)) / float(np.sum(absC)) if np.sum(absC) > 0 \
            else float(np.sum(err))
        report.update(accuracy=acc, accuracy_source='fourier tail',
                      accuracy_by_ell=acc_ell,
                      accuracy_by_tau=[float(x) for x in by_tau],
                      accuracy_l1=l1)
    if report['accuracy'] is not None and report['accuracy'] > _pc.ACCURACY_WARN:
        _pc.warn(f'pole-cluster deformation: estimated relative error '
                 f'{report["accuracy"]:.3g} (above {_pc.ACCURACY_WARN:g}) '
                 f'with {deform.n_nodes} nodes; see result["pole_clusters"].')
    return phase_j_by_ell, total_C_by_ell, C_tau_by_ell


def _trunc(s, maxlen=200):
    s = str(s)
    return s if len(s) <= maxlen else s[:maxlen - 3] + '...'


def _print_action_sectors(ft):
    """Print MF / Free / Interaction action sectors after expand()."""
    ring_gen_names = [str(g) for g in ft.ring().gens()]

    def _fmt_poly(poly):
        lines = []
        for exp_vec, coeff in poly.dict().items():
            mono = '·'.join(
                f'{ring_gen_names[i]}^{exp_vec[i]}' if exp_vec[i] > 1
                else ring_gen_names[i]
                for i in range(len(exp_vec)) if exp_vec[i] > 0
            )
            mono = mono if mono else '1'
            lines.append(f'        {mono}  *  ({_trunc(coeff, 220)})')
        return lines

    print()
    print('      ── MF action (bigrade ≤1 in each index; vanishes at saddle) ──')
    mf_raw = getattr(ft, '_mf_sector_raw', None) or {}
    any_mf = False
    for key in [(0, 0), (1, 0), (0, 1)]:
        poly = mf_raw.get(key)
        if poly is None or not poly.dict():
            continue
        any_mf = True
        print(f'      bigrade {key}:')
        for line in _fmt_poly(poly):
            print(line)
    if not any_mf:
        print('        (empty)')

    print()
    print('      ── Free action (1,1) bilinear sector ──')
    S_free = ft.free_action()
    if not S_free.dict():
        print('        (empty)')
    else:
        for line in _fmt_poly(S_free):
            print(line)

    print()
    print('      ── Interaction action (total degree ≥ 2, excluding (1,1)) ──')
    sectors = ft.sectors()
    any_int = False
    for (n_t, n_p), poly in sorted(sectors.items()):
        if (n_t, n_p) == (1, 1) or n_t + n_p < 2:
            continue
        if not poly.dict():
            continue
        any_int = True
        print(f'      bigrade ({n_t},{n_p}):')
        for line in _fmt_poly(poly):
            print(line)
    if not any_int:
        print('        (empty)')
    print()


class _PipelineState:
    """The pipeline objects after the mean-field stage, handed to the
    chosen backend's ``run`` (``api.routing``).  Plain attributes."""

    def __init__(self, **kw):
        self.__dict__.update(kw)


# ── shared stages across calls (api.covariance) ─────────────────────────────
# Inside ``stage_memo()`` the parameter-point stages of compute_cumulants --
# the expansion [1], the propagator [2], the mean field [3] and the poles and
# residues [4] -- run once per (model, parameter point, options) and later
# calls with other external fields reuse them; outside it (the default)
# nothing is shared.  The same objects the first call built are handed on, so
# a later call computes exactly what it would have computed on its own.
_STAGE_MEMO = contextvars.ContextVar('daedalus_stage_memo', default=None)


def _value_key(v):
    """An exact, hashable key of a parameter value (numbers by their binary
    value, nested lists / arrays elementwise)."""
    if isinstance(v, dict):
        return ('dict', tuple(sorted((str(a), _value_key(b))
                                     for a, b in v.items())))
    if isinstance(v, np.ndarray):
        return ('array', v.shape, tuple(_value_key(x)
                                        for x in v.ravel().tolist()))
    if isinstance(v, (list, tuple)):
        return ('seq', tuple(_value_key(x) for x in v))
    try:
        z = complex(v)
        return (z.real.hex(), z.imag.hex())
    except (TypeError, ValueError, OverflowError):
        return ('repr', repr(v))


class _StageMemo:
    """The stage results of compute_cumulants for each parameter point seen
    inside one ``stage_memo()`` block.  Holds strong references to the model
    dicts, so a key's ``id`` cannot be recycled while the block is open."""

    def __init__(self):
        self.entries = {}
        self.hits = 0
        self.misses = 0

    @staticmethod
    def key(model, taylor_order, use_cache, fundamental, fixed_point_index,
            n_starts, seed_box):
        return (id(model), int(taylor_order), bool(use_cache),
                _value_key(fundamental), int(fixed_point_index),
                int(n_starts), _value_key(seed_box or {}))

    @staticmethod
    def _top(model):
        # the model's top-level values (strong references): a value replaced
        # in place since the stages were stored is a miss
        return tuple((str(k), v) for k, v in sorted(
            model.items(), key=lambda kv: str(kv[0])))

    def get(self, key, model, tau_extent):
        hit = self.entries.get(key)
        # the poles were resolved for the largest time difference of the
        # first call; a later call may use them for that extent or less
        if hit is None or hit['model'] is not model or len(
                hit['top']) != len(model) or any(
                a != b or u is not v for (a, u), (b, v)
                in zip(hit['top'], self._top(model))) or not (
                tau_extent is None or hit['tau_extent'] is None
                or tau_extent <= hit['tau_extent']):
            self.misses += 1
            return None
        self.hits += 1
        return hit

    def put(self, key, model, tau_extent, **stages):
        self.entries[key] = dict(stages, model=model, tau_extent=tau_extent,
                                 top=self._top(model))


def stage_memo():
    """Context manager: share the parameter-point stages (expansion,
    propagator, mean field, poles) of the ``compute_cumulants`` calls made
    inside it.  Yields the :class:`_StageMemo` (``hits`` / ``misses``)."""
    @contextlib.contextmanager
    def scope():
        memo = _StageMemo()
        token = _STAGE_MEMO.set(memo)
        try:
            yield memo
        finally:
            _STAGE_MEMO.reset(token)
    return scope()


def _preflight_default() -> bool:
    """``DAEDALUS_PREFLIGHT`` (read at call time): ``0``/``off``/``false``
    turns the analyzer gates off; default on; any other value raises."""
    from engine import _env
    return _env.flag('DAEDALUS_PREFLIGHT', True)


def _make_preflight(preflight, model, k, max_ell, external_fields_user,
                    fundamental, fixed_point_index, policy, accuracy,
                    taylor_order):
    """The analyzer for this call, or ``None`` when it is off."""
    if preflight is None:
        preflight = _preflight_default()
    if not preflight:
        return None
    from api.analyze import Preflight, Question
    q = Question(k=k, max_ell=max_ell,
                 external_fields=tuple(tuple(e) for e in external_fields_user),
                 parameters=dict(fundamental) if fundamental else None,
                 fixed_point_index=fixed_point_index, accuracy=accuracy)
    return Preflight(model, q, policy, taylor_order=taylor_order)


#: Phase J counters that are maxima (G6 resets them around a run and
#: restores ``max(before, run)``, so a caller that reads them after the
#: call sees the same value as without the analyzer).
_MAX_COUNTERS = ('dd_err_ratio_max', 'dbm_error_ratio_max',
                 'nquad_hardened_cap_span_max')
#: Float sums (the G6 error estimates): reset to 0 around a run and restored
#: to ``before + run`` like the maxima.  A difference of process-wide totals
#: lost the run's sum entirely when it was far below an earlier run's (a
#: model in slow time units after one in unit time: 1e-34 - 0 against 6.9e-17
#: gave R = 0).
_SUM_COUNTERS = ('dd_err_abs_sum', 'dbm_err_abs_sum', 'ss_err_abs_sum',
                 'nquad_hardened_err_abs_sum')


def _counter_snapshot():
    """The Phase J counters and worker-state identities before a run."""
    from engine.integration.time_domain import final_integral as _fi
    from engine.integration.time_domain import pipeline as _pl
    c = _fi._RUNTIME_COUNTERS
    snap = {k_: v for k_, v in c.items() if not isinstance(v, dict)}
    # the state-space route's counters by reason (dicts)
    for k_ in _SS_DICT_COUNTERS:
        if isinstance(c.get(k_), dict):
            snap[k_] = dict(c[k_])
    maxes = {k_: c[k_] for k_ in _MAX_COUNTERS if k_ in c}
    for k_ in maxes:
        c[k_] = 0.0
    sums = {k_: c.get(k_, 0.0) for k_ in _SUM_COUNTERS}
    for k_ in sums:
        c[k_] = 0.0
    workers = {k_: id(v) for k_, v in _pl._WORKER_STATE.items()}
    return snap, (maxes, sums), workers


_SS_DICT_COUNTERS = ('ss_declined_by_reason', 'ss_fallback_by_reason')


def _counter_delta(token):
    """``(counters of the run, complete)``: additive counters as
    differences, maxima as the run's maximum (the global value restored to
    ``max(before, run)``); ``complete`` is False when work went to forked
    workers whose counters stayed in the children (the Phase J batch now
    returns its workers' counters)."""
    from engine.integration.time_domain import final_integral as _fi
    from engine.integration.time_domain import pipeline as _pl
    snap, (maxes, sums), workers = token
    c = _fi._RUNTIME_COUNTERS
    out = {}
    for k_, v in c.items():
        if isinstance(v, dict):
            if k_ in _SS_DICT_COUNTERS:
                b = snap.get(k_) if isinstance(snap.get(k_), dict) else {}
                d = {r: n - b.get(r, 0) for r, n in v.items()
                     if n - b.get(r, 0)}
                out[k_] = d
            continue
        if k_ in maxes:
            out[k_] = v
            c[k_] = max(maxes[k_], v)
        elif k_ in sums:
            out[k_] = v
            c[k_] = sums[k_] + v
        else:
            out[k_] = v - snap.get(k_, 0)
    now = {k_: id(v) for k_, v in _pl._WORKER_STATE.items()}
    complete = now == workers
    return out, complete


class PhaseJEngineFallbackWarning(UserWarning):
    """``compute_cumulants``: the state-space route fell back on the pole
    route at call time for some subsets (``result['phase_j_engine']``
    counts them by reason; the values are the pole route's there)."""


def _realization_check(prop, num_params):
    """The state-space realization's a posteriori check
    (``api._realization``: run when the rounded realization is graded or
    not numerically Hurwitz) for the analyzer's G6 accuracy estimate:
    ``{'checked', 'err_rel', 'graded', 'bad_margin'}``, or ``None`` when the
    route has no realization of this model."""
    try:
        from engine.integration.time_domain import statespace as _ssr
        ok, _why = _ssr.model_route(prop, num_params)   # engine on, Itô rule
        if not ok:
            return None
        m, _why = _ssr.realization_for(prop, num_params)
    except Exception:
        return None
    if m is None:
        return None
    cond = getattr(m.real, 'conditioning', None) or {}
    return {'checked': bool(cond.get('aposteriori_checked')),
            'err_rel': cond.get('aposteriori_err_rel'),
            'graded': bool(cond.get('graded')),
            'bad_margin': bool(cond.get('bad_margin')),
            'truncation': _truncation_record(m.real),
            # the run's own realization, for the census to re-read the
            # truncation record once the dressed lines exist (popped there;
            # never part of the result)
            '_real': m.real}


def _truncation_record(real):
    """The balanced-truncation record of a realization for the result and
    the G6 estimate: ``{'tol', 'n_truncated', 'n_dressed_truncated',
    'max_bound_rel'}`` -- the per-entry truncation of the propagator and,
    once the run has built them, the truncated dressed lines
    (``Realization.truncation['dressed']``) -- or ``None`` when the
    realization has no record.  ``max_bound_rel`` is the largest relative
    H-infinity bound ``2 sum(discarded HSV) / sigma_1`` of any truncated
    entry or line (0.0 when nothing was cut)."""
    tr = getattr(real, 'truncation', None)
    if not isinstance(tr, dict):
        return None
    dressed = tr.get('dressed') or {}
    cut = [v for v in dressed.values() if v.get('r', 0) < v.get('n', 0)]
    worst = max([float(tr.get('max_bound_rel') or 0.0)]
                + [float(v.get('bound_rel') or 0.0) for v in cut])
    return {'tol': tr.get('tol'),
            'n_truncated': int(tr.get('n_truncated') or 0),
            'n_dressed_truncated': len(cut),
            'max_bound_rel': worst}


def _engine_census(result, counters):
    """``result['phase_j_engine']``: the run's prediction (set by the
    Phase J backend) plus the census of the counters: subsets built on the
    state-space route, declined by reason, fallbacks by reason, and
    ``statespace_share`` (routed / all subsets evaluated).  A fallback warns
    once (``PhaseJEngineFallbackWarning``: never silent).  When subsets
    declined the route although the run was predicted covered, a pole
    cluster report marked 'bypassed' gets its pole status back (those
    subsets used the poles)."""
    rep = result.get('phase_j_engine')
    if rep is None:
        return
    c = counters or {}
    n_ss = int(c.get('ss_subsets_built', 0))
    n_dec = int(c.get('ss_subsets_declined', 0))
    rep.update(
        ss_subsets=n_ss, ss_declined=n_dec,
        ss_declined_by_reason=dict(c.get('ss_declined_by_reason') or {}),
        ss_evals=int(c.get('ss_evals', 0)),
        fallback_subsets=int(c.get('ss_fallback_subsets', 0)),
        fallback_evals=int(c.get('ss_fallback_evals', 0)),
        fallback_by_reason=dict(c.get('ss_fallback_by_reason') or {}),
        statespace_share=(n_ss / (n_ss + n_dec)) if n_ss + n_dec else None)
    # the dressed lines are truncated while the run builds them: re-read the
    # realization's record now that the run is over
    rc = rep.get('realization_check')
    if isinstance(rc, dict):
        real = rc.pop('_real', None)
        if real is not None:
            rc['truncation'] = _truncation_record(real)
    pc = result.get('pole_clusters')
    if (isinstance(pc, dict) and pc.get('status') == 'bypassed'
            and rep.get('pole_deformation') != 'bypassed'
            and (n_dec or rep['fallback_evals'])):
        pc['status'] = pc.get('status_poles', pc['status'])
        pc['reason'] = pc.get('reason_poles', pc.get('reason'))
    if rep['fallback_evals']:
        from api.save import _ensure_warning_shown
        _msg = (f'Phase J: the state-space route fell back on the pole route '
                f'for {rep["fallback_subsets"]} subset(s) '
                f'({rep["fallback_evals"]} evaluation(s); reasons '
                f'{rep["fallback_by_reason"]}); see result["phase_j_engine"]')
        _ensure_warning_shown(PhaseJEngineFallbackWarning, 'always',
                              message=_msg, stacklevel=3)
        warnings.warn(_msg, PhaseJEngineFallbackWarning, stacklevel=3)


def _jitter_precheck(model, external_fields, t_min, times, initial_covariance):
    """A Gaussian-jittered exogenous input (``Kernel.gaussian``; the model's
    ``exogenous_jitter``) is fitted as the density shifted by ``T``, the
    shift dropped: only stationary statistics of the rest of the network
    are unchanged by it.  Refused: the input itself (or its response field)
    as an external leg (its cross-cumulants with the network move by ``T``),
    and switch-on runs (a transient is not invariant under a time shift)."""
    jit = model.get('exogenous_jitter') if isinstance(model, dict) else None
    if not jit:
        return
    names = {str(f[0]) if isinstance(f, (tuple, list)) else str(f)
             for f in (external_fields or ())}
    hit = sorted(n for n in names if n in jit or (n.endswith('t') and n[:-1] in jit))
    if hit:
        raise ValueError(
            f'external field(s) {hit}: the Gaussian-jittered input(s) {sorted(jit)} are '
            f'fitted with a dropped time shift ({", ".join(j["shift"] for j in jit.values())}), '
            f'free for the statistics of the rest of the network only; their own '
            f'cumulants with it would move by that shift.  Observe the fields the input '
            f'drives.')
    if t_min is not None or times is not None or initial_covariance is not None:
        raise ValueError(
            f'switch-on transients (t_min / times / initial_covariance) with the '
            f'Gaussian-jittered input(s) {sorted(jit)}: the jitter is fitted with a '
            f'dropped time shift, which only stationary statistics do not see.')


def _attach_jitter(result, model):
    """``result['input_jitter']`` and the ledger's approximation entry of a
    Gaussian-jittered exogenous input (``Kernel.gaussian``)."""
    jit = model.get('exogenous_jitter') if isinstance(model, dict) else None
    if not jit or not isinstance(result, dict):
        return
    result['input_jitter'] = {n: dict(j) for n, j in jit.items()}
    rep = result.get('analysis')
    if rep is None:
        return
    from dataclasses import replace
    from api import analyze as _an
    entries = tuple(
        _an.LedgerEntry(
            kind='approximation', name=f'jitter:{n}',
            detail=(f'Gaussian jitter of the exogenous input {n} (sigma = {j["sigma"]}) as '
                    f'a rational fit of {j["fit"]["order"]} states of the density shifted '
                    f'by {j["shift"]} (shift dropped); L1 error {j["fit"]["l1"]:.2g}, '
                    f'relative error bounds by order '
                    + ', '.join(f'{q}: {b:.2g}' for q, b in j['fit']['bound_by_order'].items()
                                if q <= 4)),
            declared=True, satisfied=True)
        for n, j in jit.items())
    result['analysis'] = replace(rep, ledger=tuple(rep.ledger) + entries)
    cfg = result.get('config')
    if isinstance(cfg, dict):
        cfg['assumptions'] = result['analysis'].ledger_dicts()


def _attach_provenance(result, route, pf):
    """The route and the analyzer report on the result (``result['route']``,
    ``result['analysis']``; ``config['route']`` / ``config['assumptions']``
    for ``api.save``)."""
    rep = pf.report() if pf is not None else None
    result['route'] = route.as_dict()
    result['analysis'] = rep
    cfg = result.get('config')
    if isinstance(cfg, dict):
        cfg['route'] = route.name
        cfg['assumptions'] = rep.ledger_dicts() if rep is not None else None



# ── switch-on transients ─────────────────────────────────────────

def _switch_on_request(model, k, max_ell, t_min, times, initial_covariance,
                       tau_grid, *, spatial, saving):
    """Validate the switch-on arguments of ``compute_cumulants`` (before any
    work).  ``None`` for a stationary run (``t_min=None``); otherwise the
    request ``{'t_min', 'times' (n, k), 'initial_covariance'}``."""
    extra = [n for n, v in (('times', times),
                            ('initial_covariance', initial_covariance))
             if v is not None]
    if t_min is None:
        if extra:
            raise ValueError(
                f'{", ".join(extra)} need t_min (a switch-on transient); '
                f'without t_min the run is stationary')
        return None
    try:
        t0 = float(t_min)
    except (TypeError, ValueError):
        raise ValueError(f't_min must be a finite float, got {t_min!r}') \
            from None
    if not math.isfinite(t0):
        raise ValueError(f't_min must be finite (omit it for the stationary '
                         f'state), got {t_min!r}')
    from engine.integration.time_domain import statespace as _ss
    if spatial:
        raise NotImplementedError(
            'switch-on transients (t_min) are implemented for temporal '
            'models only')
    if saving:
        raise ValueError(
            't_min: output_npz / output_csv do not support switch-on results '
            'yet; save result["times"] and result["C_times"] instead')
    if tau_grid is not None:
        raise ValueError(
            't_min: tau_grid is the stationary lag grid; a switch-on run is '
            'evaluated at the absolute times given by times=')
    if not _ss.engine_requested():
        raise _ss.SwitchOnUnsupported(
            'engine_poles', f'{_ss.ENGINE_ENV}=poles selects the pole engine')
    if times is None:
        raise ValueError('t_min needs times=: the absolute external times, '
                         f'shape (n, k={k}) or (n,)')
    T = np.asarray(times, dtype=float)
    if T.ndim == 0:
        T = T.reshape(1)
    if T.ndim == 1:
        T = np.repeat(T[:, None], k, axis=1)
    if T.ndim != 2 or T.shape[1] != k or T.shape[0] == 0:
        raise ValueError(f'times must have shape (n, k={k}) or (n,), got '
                         f'{np.shape(times)}')
    if not np.all(np.isfinite(T)):
        raise ValueError('times must be finite')
    if np.any(T < t0):
        raise ValueError(
            f'every time must be >= t_min = {t0!r} (the fluctuations start '
            f'at t_min); the earliest is {float(T.min())!r}')
    C0 = None
    if initial_covariance is not None:
        if k != 2:
            raise ValueError(f'initial_covariance applies to k = 2 only (got '
                             f'k = {k})')
        if max_ell != 0:
            raise NotImplementedError(
                f'initial_covariance with loops (max_ell = {max_ell}) is not '
                f'implemented: the initial-state term is the tree-level one '
                f'(loops need an initial-state vertex); pass max_ell=0')
        C0 = np.asarray(initial_covariance, dtype=float)
        if (C0.ndim != 2 or C0.shape[0] != C0.shape[1]
                or not np.all(np.isfinite(C0))):
            raise ValueError('initial_covariance must be a finite square '
                             'matrix')
    return {'t_min': t0, 'times': T, 'initial_covariance': C0}


def _switch_on_wrap_td(td, t_min):
    """Wrap the callables a switch-on run stores in ``phase_j_by_ell[ell]``
    (``total_C``, ``total_C_batch``, each group's ``contribution``) so that
    they, too, evaluate inside the scope (bare, they would give stationary
    values)."""
    if not isinstance(td, dict):
        return
    if callable(td.get('total_C')):
        fn = _switch_on_callable(td['total_C'], t_min)
        td['total_C'] = fn
        if 'total_C_batch' in td:
            td['total_C_batch'] = (
                lambda tau_points, *a, _fn=fn, **kw:
                [complex(_fn(*p)) for p in tau_points])
    for g in td.get('groups') or ():
        if isinstance(g, dict) and callable(g.get('contribution')):
            g['contribution'] = _switch_on_callable(g['contribution'], t_min)
    if callable(td.get('eval_per_diagram_batch')):
        # serial (no fork) and inside the scope
        _pd = td['eval_per_diagram_batch']
        _pd_scoped = _switch_on_callable(
            lambda tau_points: _pd(tau_points, parallel=False), t_min)
        td['eval_per_diagram_batch'] = (
            lambda tau_points, *a, _f=_pd_scoped, **kw: _f(tau_points))


def _switch_on_callable(fn, t_min):
    """A per-ell (or total) callable of absolute external times, evaluated
    inside ``statespace.switch_on_scope(t_min)`` (no Itô nudge: the
    state-space route applies the tie rule to equal times exactly)."""
    from engine.integration.time_domain import statespace as _ss

    def at(*ext_time_values):
        with _ss.switch_on_scope(t_min):
            return fn(*ext_time_values)
    return at


def _initial_state_term(so, propagator_data, num_params, external_fields,
                        phys_idx):
    """``(values at so['times'], callable)`` of the tree-level initial-state
    term ``R(t1-t_min) C0 R(t2-t_min)^T`` (``initial_covariance``), or
    ``None`` without one.  ``R``: the response from the model's realization
    (``statespace.realization_for``); it must be the identity at ``0+`` (a
    first-order model whose response index ``r`` is the equation of
    physical field ``r``), so that ``C0`` is the covariance of the
    fluctuations themselves.  ``C0`` enters like a noise vertex at
    ``t_min``, so at k = 2 this is the whole tree term."""
    C0 = so['initial_covariance']
    if C0 is None:
        return None
    from engine.integration.time_domain import statespace as _ss
    ops, why = _ss.realization_for(propagator_data, num_params)
    if ops is None:
        raise _ss.SwitchOnUnsupported(
            why, 'the initial-state term needs the model realization')
    nf = int(propagator_data['nf'])
    if C0.shape != (nf, nf):
        raise ValueError(
            f'initial_covariance must have shape {(nf, nf)} (the {nf} physical '
            f'field(s) in their internal order), got {C0.shape}')

    def R(p, r, t):
        ent = ops.entry(p, r)
        if ent is None or ent.n == 0:
            return 0.0
        v = ent.impulse(t)
        return float(np.real(v)) if abs(np.imag(v)) <= 1e-14 * (
            1 + abs(v)) else complex(v)

    for p in range(nf):
        for r in range(nf):
            v = R(p, r, 0.0)
            if abs(v - (1.0 if p == r else 0.0)) > 1e-12:
                raise ValueError(
                    'initial_covariance needs a first-order '
                    'model whose response at 0+ is the identity (response '
                    f'index r = the equation of physical field r); here '
                    f'R[{p}, {r}](0+) = {v!r}')
    legs = [phys_idx[f] for f in external_fields]
    t0 = so['t_min']

    def term(*ext):
        a = np.array([R(legs[0], r, ext[0] - t0) for r in range(nf)])
        b = np.array([R(legs[1], r, ext[1] - t0) for r in range(nf)])
        return complex(a @ C0 @ b)

    vals = np.array([term(*row) for row in so['times']], dtype=complex)
    return vals, term


def _embedded_noise_states(model):
    """Names of the auxiliary noise fields the model builder made from
    declared Markov-embeddable noise kernels (``correlated_noises`` entries
    ``*_markov_aux``).  Under a switch-on they start at zero like every
    fluctuation: the noise is switched on with the system."""
    cn = model.get('correlated_noises') or {} if isinstance(model, dict) \
        else {}
    return sorted(str(k) for k in cn if str(k).endswith('_markov_aux'))


def _ledger_switch_on(preflight, so):
    """The analyzer ledger records the non-stationary question: the
    stationary-state line is replaced by the switch-on one (the gates
    themselves are unchanged)."""
    if preflight is None:
        return
    ctx = preflight.ctx
    ctx.ledger[:] = [e for e in ctx.ledger if e.name != 'stationary_state']
    init = ('the fluctuations at t_min are zero'
            if so['initial_covariance'] is None else
            'plus the tree-level term of initial_covariance')
    emb = _embedded_noise_states(preflight.ctx.model)
    ctx.ledger_add('approximation', 'switch_on_state',
                   f'switch-on transient: fluctuations about the selected '
                   f'saddle start at t_min = {so["t_min"]!r} ({init}); every '
                   f'vertex time is > t_min; state-space Phase J route only'
                   + (f'; the Markov-embedded noise state(s) {emb} start at '
                      f'zero too (the noise is switched on at t_min)'
                      if emb else ''),
                   declared=True)


def _switch_on_check(result, failures, model_name):
    """A switch-on run is never partial: an unevaluated diagram, or δ
    components (whose weights are not yet derived for a transient), raise."""
    from engine.integration.time_domain import statespace as _ss
    if failures:
        raise PhaseJIncompleteError(
            'switch-on run (t_min): ' + _phase_j_failure_message(
                failures, model_name), result=result, failures=failures)
    for ell, td in (result.get('phase_j_by_ell') or {}).items():
        if td and (td.get('delta_contributions')
                   or td.get('delta_contributions_multi')
                   or td.get('n_delta_subsets_dropped')):
            raise _ss.SwitchOnUnsupported(
                'delta_components', f'ell = {ell} has δ(t) components (an '
                f'instantaneous response; recorded or dropped shot-noise '
                f'subsets); their switch-on weights are not implemented')


@_growth_scoped
def compute_cumulants(
    model: dict,
    k: int,
    max_ell: int = 0,
    fundamental: dict = None,
    external_fields: list[tuple[str, int]] = None,
    *,
    parameters: dict = None,         # canonical name for ``fundamental`` (wins if both given)
    tau_max: float = 50.0,
    tau_step: float = 0.5,
    tau_grid=None,                   # explicit τ grid (array); overrides tau_max/tau_step
    spatial_grid=None,
    chi_grid=None,                   # canonical name for ``spatial_grid`` (wins if both given)
    taylor_order: int = None,
    origin_leaf_idx: int = 0,
    output_npz: str = None,
    output_csv: str = None,
    use_cache: bool = True,
    parallel: bool = True,
    spatial_parallel: bool = True,   # spatial THREADS (no fork — safe); smart-gated to L≥2 where it helps (~2.5×)
    spatial_n_q: int = 64,           # spatial loop q-grid count — cost is LINEAR in it (64→16 = 4× faster; coarsens C(x) at large x)
    spatial_points=None,             # k>=3 spatial: (n_pts, k-1, 2) of (x_j, tau_j) offsets per non-anchor slot

    n_workers: int = None,
    use_grouped_phase_j: bool = False,
    fixed_point_index: int = 0,
    mf_dae_n_starts: int = 64,
    mf_dae_seed_box: dict | None = None,
    verbose: bool = True,
    policy=None,                     # api.analyze.Policy (or dict): strict / consents / downgrade
    preflight: bool = None,          # the analyzer gates (default on; DAEDALUS_PREFLIGHT=0 turns them off)
    backend: str = None,             # an explicit backend name (api.routing); None = the router picks
    accuracy: float = None,          # target relative accuracy (G6 threshold; default 1e-10)
    t_min: float = None,             # switch-on transients: fluctuations start at 0 at this absolute time
    times=None,                      # with t_min: absolute external times, (n, k) or (n,) (every leg at t)
    initial_covariance=None,         # with t_min (k=2, tree): covariance of the fluctuations at t_min
    delay: dict = None,              # delay kernels (api.delay): order / tol / n_start / n_max / wall_max / tree
    discard=False,                   # evaluate each diagram on the grid, then release it (True / 'auto')
    points=None,                     # k>=3 temporal: the (n, k) external-time tuples the result is read at
    spectra=None,                    # k=2: exact S(omega), S(0), int_0^T C, tau_c (api.spectra)
) -> dict[str, Any]:
    """
    Compute the k-point cumulant slice for ``model`` at the given
    parameter point, all the way through Phase J time-domain integration.

    Parameters
    ----------
    model : dict
        The model dict (``builder.build()``; an unbuilt ``api.sde.SDE`` or
        builder is built for you).  Must declare:
        ``response_fields``, ``physical_fields``, ``parameters``,
        ``kernels``, ``operators``, ``functions``, ``mf_substitutions``,
        ``mf_bg_conditions``, ``specializations``, ``kernel_ft_image``,
        ``phi_concrete``, ``mf_equations``, ``action``.
        Optional: ``correlated_noises``.
    k : int
        Number of external legs (k-point cumulant).
    max_ell : int, default 0
        Maximum loop order (0 = tree only).
    fundamental : dict
        Numerical parameter values keyed by param name.  E.g.::

            {'a': 1.0, 'tau': 10.0, 'tau_g': 2.5,
             'E': [1.1, 1.05], 'w': [[0.35, 0.4], [0.3, 0.5]],
             'w_X': 0.3, 'lambda_X': 2.5, 'p_part': 0.6,
             'mu_shift_diff': 0.0, 'sigma_shift_diff_sq': 1.0}

    external_fields : list
        The ``k`` legs, each a field name or a ``(field, unit)`` pair: the
        plain field name (``'x'``) is the documented form, its fluctuation
        name (``'dx'``) an accepted alias, and the two give the same result
        byte for byte.  A bare name (``['x', 'x']``) is the leg ``('x', 1)``
        of a field with one unit; a field in a population of several units
        needs the unit (``[('v', 1), ('v', 2)]``).  At ``k = 1`` the result is
        the loop SHIFT of the mean, ``<x> - x*`` (tree level 0): the mean is
        ``mf_values['xstar']`` plus the sum of the shifts over loop orders, and
        ``dd.Config(output='moment')`` returns that loop-corrected mean
        directly (``res['moment']``).
    tau_max, tau_step : float
        τ-grid extent and spacing for the Phase J slice evaluation.
    taylor_order : int or None
        Truncation order for FieldTheory's nonlinear-function Taylor.
        ``None`` (default) auto-picks ``max(k + 2 * max_ell, 2)``,
        which is the smallest order that captures every vertex needed
        for a connected diagram with ``k`` external legs at ``ell``
        loops (max vertex order ≤ ``k + 2·ell``).  The floor of 2 is
        the structural minimum — order 2 still has the (1,1) bilinear
        propagator kernel plus the (0,0)/(1,0)/(0,1) MF saddle sectors,
        which is everything the cache machinery needs even for the
        degenerate ``k=2, max_ell=0`` case (tree-level pair correlator
        = bare propagator, no interaction vertices).  Pass an explicit
        integer to override (e.g. for non-standard diagrammatic
        content or when probing higher-order vertices for cache
        invalidation testing).

        **Previously this floor was 4**, which forced models at
        ``k=2, max_ell=0`` to pay an order-4 expansion they didn't
        mathematically need — see ``CHANGELOG.md``
        (model-precompute-cache branch) for the cost it imposed on
        the dendritic-linear model.
    origin_leaf_idx : int
        Which canonical position to pin to t=0 for the slice (default 0).
    output_npz : str or None
        If given, save the result to this ``.npz`` path via
        ``api.save.save_npz``.  See that module for the schema —
        per-loop-order curves (``C_tree``, ``C_1_loop``, ...,
        ``C_total``), adaptive mean-field arrays, parameter keys.
    output_csv : str or None
        If given, save a wide-format CSV companion to ``output_npz``
        via ``api.save.save_csv`` (parameter / mean-field
        metadata as comment lines, then a τ-grid table).
    use_cache : bool
        Whether to reuse cached symbolic propagator (per ``(model, taylor)``)
        and unique typed diagrams (per ``(model, taylor, k, ell, ext_fields)``).
        Both caches live as sibling files under ``saved_models/<model>/``
        (e.g. ``saved_models/<model>/expand_taylor<N>.sobj``), one per
        Taylor order.  With ``False`` no cache file is read or written, and
        a temporal run takes its prediagrams from the eager enumerator, as
        before the streaming enumerator existed (``DAEDALUS_PREDIAGRAM_EAGER=0``
        selects the faster streaming one; see
        ``engine.enumeration.prediagram_cache.TEMPORAL_CACHE_OFF_STREAMS``).
        Cache on and cache off can use different labelled representatives of
        the same diagrams (in a different ``result['diagrams']`` order):
        totals then agree to rounding, or to the ``scipy.nquad`` fallback's
        tolerance on models whose regions fall back to it.  A spatial run
        never uses the prediagram cache; it streams its prediagrams in
        memory whatever this says.
    parallel : bool, default True
        Parallel evaluation where it helps, by a method safe for the
        platform; the numbers are the same bytes as ``parallel=False``.
        Phase J's τ grid (step [7]) fans its diagrams out: diagrams with
        large state blocks go to threads on every platform (notebooks
        included); a batch predicted to take seconds goes to spawned,
        Sage-free worker processes in a script on any platform (in a
        notebook only on request: ``DAEDALUS_SPAWN_IN_KERNEL=1``); a Linux
        fork pool only on request (``start_method='fork'``).  Type
        assignment (step [5]) forks on Linux only.  macOS never forks.
        Every pool fits the memory budget (fewer workers, then serial).
        ``DAEDALUS_PARALLEL=0`` turns every worker pool off, spatial
        threads included.  See ``docs/parallel.md``.
    n_workers : int or None, default None
        Worker count when ``parallel=True``.  ``None`` lets each stage
        pick its own default (Phase J threads: up to 8; spawned workers:
        4; fork workers: ``os.cpu_count()``; all within the memory
        budget).  Given, it also sets the process count of a streamed
        prediagram enumeration (Linux).
    discard : bool or 'auto', default False
        Evaluate-and-discard.  ``True``: each diagram is
        evaluated on the τ grid right after it is built and then
        released (its pole plans and stationary states are refunded to
        the memory budget), so Phase J holds one diagram at a time: the
        pre-pass compares the largest diagram plus one segment working
        set with the budget instead of the sum.  Needs fixed evaluation
        points: k = 1 or 2 with a τ grid, or k >= 3 with ``points``
        (or ``tau_grid=[0.0]``, the equal-time point); no switch-on, no
        pole-cluster deformation pass.  The values are the same; the
        result's callables (``total_C``, ``total_C_by_ell``, the
        per-diagram callables) then answer those points only and raise
        ``DiscardedDiagramsError`` elsewhere (rerun with
        ``discard=False``).  ``'auto'``: on exactly when the summed
        estimate is over the budget and the largest diagram fits; when
        the estimate is under the budget the run starts with the callables
        kept and switches to evaluate-and-discard at the first diagram
        boundary where the run's measured growth exceeds
        ``DISCARD_AUTO_GROWTH_FRACTION`` (0.5) of the budget (the diagrams
        built before then are evaluated on the points and released; the
        values are the same).  ``result['phase_j_memory']['discard']``
        says whether it was on, ``['discard_switched_at']`` where.
    points : array (n, k) or None
        k >= 3 temporal runs: the external-time tuples (``total_C``'s
        arguments, one time per leg in ``external_fields`` order) the
        result will be read at; duplicates are dropped.  It fixes the
        evaluation points for ``discard`` (``dd.run`` passes its slice
        points).  With evaluate-and-discard on, ``result['C_points']`` /
        ``result['C_points_by_ell']`` hold the values at ``result['points']``.
    spectra : None, True or dict
        k = 2 temporal stationary runs: the exact spectral and integrated
        outputs in ``result['spectra']`` (:mod:`api.spectra`), computed from
        the state-space triples after the run, per loop order and in total:
        the spectrum ``S(omega) = int C(tau) exp(+i omega tau) dtau`` (δ
        components included, no ``1/2pi``) on ``omega``, ``S(0)``, the
        windows ``int_0^T C`` / ``int_{-T}^0 C`` for each ``T``, the
        one-sided integrals to infinity, ``C(0)`` (the Itô left limit) and
        the correlation time ``tau_c = int_0^inf C / C(0)``.  A dict takes
        ``omega`` (an ``(lo, hi, n)`` tuple or an array, as ``tau_grid``),
        ``T`` (a number or an array), ``method`` (``'auto'`` default: exact
        where the triples exist, else adaptive quadrature of the Phase J
        callables with error estimates and an ``MEFallbackWarning``;
        ``'exact'`` refuses instead; ``'quad'``) and the quadrature
        tolerances ``epsabs`` / ``epsrel`` / ``limit``; ``True`` asks for
        ``S(0)``, the integrals to infinity and ``tau_c`` only.  The lag
        grid is not used: ``tau_grid=np.array([0.0])`` keeps the run to the
        equal-time point.  Refused before the run for k != 2, spatial
        models, ``t_min``, and ``discard`` other than False; a refusal after
        the run (``method='exact'`` for an order without triples, ...)
        keeps the result and records ``{'status': 'refused', 'reason':
        ...}`` with a :class:`api.spectra.SpectraWarning`; a ``C`` that does
        not decay has NaN for ``S`` and the integrals to infinity (reason in
        ``notes``).  Conventions: ``api.spectra``.
    verbose : bool
        Print progress messages.
    policy : api.analyze.Policy, dict or None
        What the model analyzer may accept without asking: ``mode``
        (``'permissive'`` default, ``'strict'`` makes warnings errors),
        consents (``allow_taylor_truncation``, ``allow_markov_embedding``,
        ``allow_numeric_fallback``), ``interpretation``, ``assertions`` and
        ``downgrade`` (error codes accepted as warnings).
    preflight : bool or None
        Run the analyzer gates (``api.analyze.Preflight``) at the stage
        boundaries: G0/G1 before the expansion, G1 after it, G3 (class)
        after the propagator, G2/G4 after the mean field, G3 (poles) after
        the residues, all before the diagram enumeration; any ``error``
        finding raises :class:`api.analyze.AnalyzerError` (a
        ``ValueError``) with its code and remedy.  ``None``: on, unless the
        environment sets ``DAEDALUS_PREFLIGHT=0``.  The gates read the
        pipeline's own objects and change no number.
    backend : str or None
        Force one registered backend (``api.routing.backends()``); it must
        still apply.  ``None``: the first applicable one.
    accuracy : float or None
        Target relative accuracy for the G6 total-relative accuracy check
        (``Question.accuracy``; default ``1e-10``, the per-subset acceptance
        of the Phase J routes).
    t_min : float or None
        ``None`` (default): the stationary state (time-translation
        invariant; nothing below applies).  A finite value asks for a
        **switch-on transient**: the fluctuations about the selected
        mean-field saddle start at zero at the absolute time ``t_min``
        (the state is the saddle exactly at ``t_min``), so every vertex-time
        integral has the lower bound ``t_min`` and the cumulant is a function
        of the absolute external times.  Computed on the state-space Phase J
        route only (``engine.integration.time_domain.statespace``: a finite
        first segment from the empty state); a model or subset it does not
        take, ``DAEDALUS_PHASE_J_ENGINE=poles``, spatial models and
        ``output_npz``/``output_csv`` raise.  As ``t_min ->
        -inf`` the result tends to the stationary one.  A noise kernel
        the model builder Markov-embedded (an auxiliary field) starts at
        zero too: the noise is switched on with the system
        (``result['nonstationary']['embedded_noise_states']``).
    times : array or None
        Required with ``t_min``: the absolute external times to evaluate,
        shape ``(n, k)`` (row ``i`` = the times of the ``k`` legs, in the
        order of ``external_fields``) or ``(n,)`` (every leg at ``t``: the
        equal-time cumulant); taken literally (a tuple too: the ``(lo, hi,
        n)`` range form is ``dd.Config``'s).  Every time must be ``>=
        t_min``; equal times
        are exact ties (no Itô nudge: the state-space route applies the
        tie rule).  Values: ``result['C_times']``.  ``tau_grid`` (the
        stationary lag grid) must not be given with ``t_min``.
    initial_covariance : array or None
        With ``t_min``, ``k = 2``, ``max_ell = 0``: an initial covariance
        ``C0`` of the fluctuations at ``t_min`` (square, over the physical
        fields in their internal order, ``result['propagator']['nf']``).
        The tree adds the deterministic term ``R(t1 - t_min) C0
        R(t2 - t_min)^T`` (``R``: the response), which is the whole tree
        term of a Gaussian initial state (``C0`` enters like a noise vertex
        at ``t_min``); loops with ``C0 != 0`` are not implemented (raises).
        Needs a first-order model whose response at ``0+`` is the identity.
        (A deterministic initial offset of the mean is not offered: its tree
        is the full nonlinear relaxation, a time-dependent saddle.)
    delay : dict or None
        Options for models with delay kernels (``exp(-i omega d)`` factors in
        a kernel image; see :mod:`api.delay`): ``order`` (``'auto'`` or a
        fixed Pade order ``n``), ``tol`` (target relative accuracy of the
        observable, default ``1e-8``), ``n_start`` / ``n_max`` (doubling search,
        default 4 / 16), ``wall_max`` (cap in s on the predicted wall time of
        the next order, default 900), ``tree`` (``'exact'``, default: the
        k = 2 tree from the exact delayed covariance; ``'pade'``),
        ``order_cache`` (default ``True``: an entry's repeated call replays its
        search; another entry at the same model and parameter point starts its
        own search at the cached order; ``False`` searches every call).  Ignored
        for models without delays.  The result then carries
        ``result['delay']``.

    Returns
    -------
    dict with keys:
        'total_C'         : callable(*tau) — full sum across ells
        'total_C_by_ell'  : {ell: callable(*tau)} — per-loop-order
        'C_tau'           : ndarray of total_C on the τ-grid (k=1, 2;
                             None for k≥3 — caller uses total_C / per-
                             ell callables instead).  k=2: the connected
                             cumulant C(τ) = <dφ(0) dφ(τ)> of the
                             fluctuation about the saddle.  k=1: the mean
                             SHIFT <φ> − φ* produced by the loops (0 at
                             tree level, since the cumulants expand
                             around the saddle); the mean itself is
                             ``mf_values`` + ``C_tau``.
        'C_tau_by_ell'    : {ell: ndarray or None} — per-loop-order
                             grid eval; ``C_tau == sum(values())``
        'tau_grid'        : ndarray of τ values
        'mf_values'       : raw {internal_name: [v_pop1, ...]}, adaptive
        'mf'              : :class:`api.access.MeanField` accessor —
                             ``mf['v', 1]`` returns ``v*_1``,
                             ``mf['n']`` returns the whole nstar vector
        'params'          : :class:`api.access.Parameters` accessor —
                             ``params['E', 1]``, ``params['w', 1, 2]``,
                             ``params['tau']`` (1-based indexing)
        'num_params'      : {SR symbol: float}
        'propagator'      : the propagator data dict
        'diagrams'        : list of dicts {'typed_diagram', 'classify',
                             'combined_prefactor', 'multiplicity' (M(F),
                             prod n_leg!/|Aut_ext|), 'dedup_class_size',
                             'ell'}
        'kernel_groups'   : list of dicts as fed to compute_correction_td
        'phase_j_by_ell'  : {ell: td_result dict}
        'spectra'         : (only with ``spectra=``) the dict of
                             :func:`api.spectra.spectra`: ``omega``, ``S``,
                             ``S_by_ell``, ``S0``, ``T``, ``integral``,
                             ``integral_neg``, ``integral_inf``,
                             ``integral_inf_neg``, ``delta0``, ``C0``,
                             ``tau_c`` (each also ``<name>_by_ell``;
                             ``tau_c_by_ell`` is the correlation time of
                             the cumulant truncated at that order),
                             ``method_by_ell``, ``error_by_ell``,
                             ``abs_error``, ``status``, ``notes``,
                             ``conventions``
        'status'          : (Phase J backends) ``'ok'``, or ``'incomplete'``
                             when Phase J could
                             not evaluate some diagram(s): they are left out
                             of ``C_tau``, a :class:`PhaseJIncompleteWarning`
                             names them (``policy={'mode': 'strict'}``
                             raises :class:`PhaseJIncompleteError`)
        'phase_j_incomplete' : {ell: [{'diagram', 'n_diagrams', 'reason'}]}
                             of those diagrams (``{}`` when complete)
        'route'           : the routing record (``api.routing.Route.
                             as_dict``: backend, rejected backends)
        'analysis'        : the :class:`api.analyze.ModelReport` of the
                             gates (traits, findings, assumptions ledger,
                             G6 run diagnostics); ``None`` with
                             ``preflight=False``
        'config'          : input args echoed back; in addition to the
                             obvious keys, ``external_fields_in`` echoes
                             the user-passed form (e.g. ``[('n', 1)]``)
                             while ``external_fields`` stores the
                             internal form (e.g. ``[('dn', 1)]``), and
                             ``phase_j_convention`` records the Phase J
                             equal-time rule the numbers were computed
                             with (``api.save`` stamps saved files with
                             it), ``route`` the backend name and
                             ``assumptions`` the ledger (saved too).

    SWITCH-ON (``t_min`` given): ``C_tau`` / ``C_tau_by_ell`` / ``tau_grid``
    are ``None``; instead
        'times'           : ndarray ``(n, k)`` of absolute external times
        'C_times'         : ndarray ``(n,)`` of the cumulant at each row
        'C_times_by_ell'  : {ell: ndarray (n,)}; ``C_times == sum(values())``
        't_min'           : the switch-on time
        'nonstationary'   : the record of the question (protocol, t_min,
                             initial-state terms and their values)
    and ``total_C`` / ``total_C_by_ell`` take ABSOLUTE times (each call runs
    inside :func:`statespace.switch_on_scope`).

    SPATIAL k=2 (a ``spatial`` model) returns the same schema where it
    applies, keyed by space as well:
        'C_tau_x'         : ndarray ``(n_tau, n_x)`` of C(x, τ) at the top
                             loop order
        'C_tau'           : the slice of ``C_tau_x`` at the x closest to 0
        'C_tau_x_by_ell'  : {ell: ndarray} the loop-order INCREMENTS of
                             C(x, τ): ell = 0 the tree, ell >= 1 the
                             ell-loop contribution, so
                             ``sum(C_tau_x_by_ell.values()) == C_tau_x``
                             (the temporal ``C_tau_by_ell`` convention)
        'C_tau_x_by_order': {ell: ndarray} the CUMULATIVE C(x, τ) through
                             each loop order (tree, tree + 1-loop, ...);
                             do not sum these
        'total_C_by_ell'  : {ell: callable(tau, x)} the loop-order increments
                             (the same data as ``C_tau_x_by_ell``)
        'tau_grid', 'chi_grid' : the grids (``'spatial_grid'`` is the older
                             name of ``chi_grid``)
        'status'          : ``'ok'``
        'spatial_info', 'mf_values', 'mf', 'params', 'num_params',
        'propagator', 'config' : as above
    (There is no ``C_tau_by_ell``: the per-order data are the x-dependent
    ``C_tau_x_by_ell``, increments like the temporal schema.)

    SPATIAL k≥3 (``spatial_points`` given) returns a DIFFERENT dict (an
    early return), keyed by the explicit evaluation events rather than a
    τ-grid:
        'C_kpoint'        : ndarray ``(n_pts,)`` — the k-point cumulant
                             at each event configuration
        'C_kpoint_by_ell' : {ell: ndarray (n_pts,)} per-loop-order
        'points'          : the ``spatial_points`` array echoed back
                             ``(n_pts, k-1, 2)`` of ``(x_j, τ_j)`` offsets
        'spatial_info'    : driver diagnostics (μ, D, per-ell, eig, …)
        'k', 'max_ell', 'mf'

    COVARIANCE MATRIX (``api.covariance.covariance_matrix``,
    ``dd.covariance_matrix``, ``dd.Config(covariance=...)``): one call per
    pair of fields inside :func:`stage_memo`, returning its own dict:
        'Sigma'           : complex ``(n_tau, N, N)``,
                             ``Sigma[n, i, j] = <dx_i(0) dx_j(tau_n)>``
                             (each computed entry is this function's
                             ``C_tau`` of the pair, bit for bit)
        'Sigma_by_ell'    : {ell: (n_tau, N, N)} the loop-order increments
        'Sigma0'          : the ``tau = 0`` matrix (``None`` if 0 is not on
                             the grid); ``Sigma0[j, i]`` is the left limit
                             ``C_ji(0-)``, not the transpose, where ``C_ij``
                             jumps at 0
        'fields', 'tau_grid', 'pairs', 'max_ell', 'status', 'mf_values',
        'params', 'wall', 'cpu'
        'runs'            : one record per call (fields, indices, tau_grid
                             'all' | 'points', n_tau, status,
                             phase_j_incomplete, n_diagrams, wall)
        'reflected'       : how many entries ``C_ji(tau) = C_ij(-tau)`` filled
        'shared_stages'   : {'hits', 'misses'} of the stage memo
        'entries'         : (``keep_entries=True``) {(i, j): result}
        'derivatives'     : (``derivatives=``) {label: {'dSigma', 'error',
                             'dSigma_by_ell', 'error_by_ell', 'step',
                             'value', 'dmf', 'dmf_fd', 'dmf_note', 'status',
                             'wall', 'cpu'}}
    """
    model = as_model_dict(model)         # an SDE / builder is built here
    # Grouped Phase J is retired: the flag is accepted
    # for one release and runs the per-diagram integrator (same numbers)
    if use_grouped_phase_j:
        warnings.warn(
            'use_grouped_phase_j is retired and has no effect: grouped Phase '
            'J was removed (the per-diagram integrator gives the same '
            'numbers and is what runs); drop the argument',
            FutureWarning, stacklevel=2)
        use_grouped_phase_j = False
    # the call as given (a model with delay kernels is handed to api.delay,
    # which re-enters compute_cumulants with Pade-approximated copies)
    _delay_call = dict(locals())
    # a misspelled DAEDALUS_PHASE_J_ENGINE fails here, not at the
    # first Phase J subset after the enumeration
    from engine.integration.time_domain import statespace as _ss_env
    _ss_env.engine_requested()
    _ss_env.fallback_enabled()      # (validated before the run too)
    # Accept the canonical names the dd.Config layer uses (``parameters`` /
    # ``chi_grid``), mirroring onto the internal names; canonical wins.
    if parameters is not None:
        fundamental = parameters
    if chi_grid is not None:
        spatial_grid = chi_grid
    # Fall back to the defaults declared via ``ModelBuilder.parameter(...,
    # default=...)`` for every parameter the caller did not give: an EMPTY
    # dict (several models ship ``DEFAULT_FUNDAMENTAL = {}``, and a caller
    # forwarding that verbatim must not sail past this and fail much later
    # with a bare ``NameError`` from inside the mean-field solve) and a
    # PARTIAL one alike -- the given values win.  Mean-field parameters
    # (``mean_field=True``, e.g. ``xstar``/``nstar``) are SOLVED, not
    # supplied, so they are never required here.
    _pspecs = model.get('parameters', []) or []
    _given = dict(fundamental) if fundamental else {}
    # (A saddle variable's default is never merged in: the legacy iteration
    # solver would take it as a given value and pin the saddle there.)
    fundamental = {p['name']: p['default'] for p in _pspecs
                   if p.get('default') is not None and not p.get('mean_field')}
    fundamental.update(_given)
    _missing = sorted(p['name'] for p in _pspecs
                      if p['name'] not in fundamental
                      and not p.get('mean_field'))
    # A misspelled name would otherwise be ignored and the run would use the
    # DEFAULT value of the parameter meant (same rule as ``dd.run``).
    _declared = {str(p['name']) for p in _pspecs}
    _unknown = sorted(str(n) for n in _given if str(n) not in _declared)
    if _declared and _unknown:
        raise ValueError(
            f'compute_cumulants: unknown parameter name(s) {_unknown}; the '
            f'model declares {sorted(_declared)}.')
    if _missing:
        raise ValueError(
            'compute_cumulants: ' +
            ('no ``parameters=`` given' if not _given else
             'the ``parameters=`` given do not include them') +
            ', and these model parameters declare no default: ' +
            ', '.join(_missing) + '. '
            'Pass parameters={...} or declare defaults via '
            'ModelBuilder.parameter(name, default=...).')
    k = _entry_int('k', k)
    max_ell = _entry_int('max_ell', max_ell)
    if external_fields is None:
        raise ValueError('external_fields is required')
    if isinstance(external_fields, str):
        raise ValueError(f'external_fields must be a list of k legs (e.g. [{external_fields!r}] * k); '
                         f'got the string {external_fields!r}.')
    external_fields = resolve_external_fields(model, external_fields)
    if len(external_fields) != k:
        raise ValueError(
            f'external_fields has {len(external_fields)} entries but k={k}'
        )
    if k < 1:
        raise ValueError(f'k must be >= 1; got {k}')
    if max_ell < 0:
        raise ValueError(f'max_ell must be >= 0; got {max_ell}')
    try:
        _acc = (None if accuracy is None or isinstance(accuracy, (bool, str))
                else float(accuracy))
    except (TypeError, ValueError):
        _acc = None
    if accuracy is not None and not (_acc is not None and np.isfinite(_acc)
                                     and _acc > 0):
        raise ValueError(
            f'accuracy must be a finite number > 0 (a target relative '
            f'accuracy, e.g. 1e-10) or None; got {accuracy!r}.')
    if _entry_int('fixed_point_index', fixed_point_index) < 0:
        raise ValueError(f'fixed_point_index must be >= 0; got '
                         f'{fixed_point_index!r}.')
    if n_workers is not None and _entry_int('n_workers', n_workers) < 1:
        raise ValueError(f'n_workers must be >= 1 or None; got {n_workers!r}.')
    if isinstance(discard, (bool, np.bool_)):
        discard = bool(discard)
    elif not (isinstance(discard, str) and discard == 'auto'):
        raise ValueError(f"discard={discard!r}: expected False, True or "
                         f"'auto'")
    from api import spectra as _spectra_mod
    _spectra_opts = _spectra_mod.resolve_spectra_options(spectra)
    if _spectra_opts is not None:
        _spectra_mod.check_spectra_question(
            k, spatial=bool(spatial_grid is not None or chi_grid is not None
                            or spatial_points is not None
                            or (isinstance(model, dict)
                                and model.get('spatial'))),
            t_min=t_min, discard=discard)
    if points is not None:
        _pa = np.asarray(points, dtype=float)
        if k < 3:
            raise ValueError(
                f'points= is for k >= 3 (the external-time tuples of a '
                f'k-point cumulant); k = {k} is read on tau_grid.')
        if t_min is not None:
            raise ValueError('points= with t_min: a switch-on run takes its '
                             'absolute external times from times=.')
        if spatial_grid is not None or spatial_points is not None or (
                isinstance(model, dict) and model.get('spatial')):
            raise ValueError('points= is for temporal runs; spatial k >= 3 '
                             'events are spatial_points=.')
        if _pa.ndim != 2 or _pa.shape[1] != k or _pa.shape[0] < 1:
            raise ValueError(
                f'points must have shape (n, k) = (n, {k}), n >= 1: one '
                f'time per external leg; got shape {_pa.shape}.')
        if not np.all(np.isfinite(_pa)):
            raise ValueError('points must be finite.')
        # in order, without duplicates (each is evaluated once)
        points = list(dict.fromkeys(tuple(float(x) for x in row)
                                    for row in _pa))
    # one rule for every entry point (dd.Config, YAML, here): an
    # (lo, hi, n) tuple is a range, a list/array is literal points, and a
    # 3-element integer-count list is refused (api._grids)
    from api._grids import resolve_grid
    tau_grid = resolve_grid(tau_grid, 'tau_grid')
    spatial_grid = resolve_grid(spatial_grid, 'spatial_grid')
    _nso = model.get('noise_series_order') if isinstance(model, dict) else None
    if _nso and k + 2 * max_ell > int(_nso['order']):
        # a CGF with a removable singularity at 0 (the uniform jump) enters
        # the action as its Taylor polynomial: vertices above that order are
        # absent, and a vertex of any order up to k + 2*max_ell can enter a
        # k-point max_ell-loop diagram
        raise ValueError(
            f"noise source(s) {', '.join(_nso.get('sources', []))}: the action "
            f"carries their CGF as a Taylor polynomial of order {_nso['order']} "
            f"(SDE.series_order), but k = {k}, max_ell = {max_ell} can use noise "
            f"vertices up to order k + 2*max_ell = {k + 2 * max_ell}; raise "
            f"SDE.series_order({k + 2 * max_ell}).")
    _meo = model.get('marcus_exact_order') if isinstance(model, dict) else None
    if _meo and k + max_ell > int(_meo['order']):
        # the exact Marcus noise term is kept to SDE.series_order response
        # legs, and a k-point max_ell-loop diagram uses vertices with up to
        # k + max_ell of them (_marcus_admissible)
        raise ValueError(
            f"Marcus source(s) {', '.join(_meo.get('sources', []))}: the action "
            f"carries their exact noise term to {_meo['order']} response legs "
            f"(SDE.series_order), but k = {k}, max_ell = {max_ell} can use noise "
            f"vertices with up to k + max_ell = {k + max_ell} response legs; raise "
            f"SDE.series_order({k + max_ell}).")
    # an exact-route Marcus source whose jump law has no mgf where a usable
    # vertex needs it: refused before the mean-field solve
    _marcus_precheck(model, fundamental, k, max_ell)
    _jitter_precheck(model, external_fields, t_min, times, initial_covariance)
    _ndo = model.get('noise_declared_orders') if isinstance(model, dict) else None
    for _src, _top in sorted((_ndo or {}).items()):
        _key = (model.get('name'), _src, int(_top))
        if k + 2 * max_ell > int(_top) and _key not in _CUMULANT_TRUNCATION_WARNED:
            # a Cumulants([...]) source: its CGF is a polynomial of the declared
            # order, so noise vertices above it are absent (a formal truncation,
            # not an error); one warning per model, source and process
            _CUMULANT_TRUNCATION_WARNED.add(_key)
            from api.noise import CumulantTruncationWarning
            from api.save import _ensure_warning_shown
            _msg = (f"noise source {_src!r} (Cumulants) declares cumulants up to order "
                    f"{int(_top)}; k = {k}, max_ell = {max_ell} can use noise vertices "
                    f"up to order {k + 2 * max_ell}, and those above order {int(_top)} "
                    f"are taken as 0. Give more kappas if they are not zero.")
            _ensure_warning_shown(CumulantTruncationWarning, 'always',
                                  message=_msg, stacklevel=2)
            warnings.warn(_msg, CumulantTruncationWarning, stacklevel=2)
    if backend is not None:
        from api import routing as _routing_check
        if backend not in {b.name for b in _routing_check.backends()}:
            raise ValueError(
                f'backend={backend!r} is not registered; known: '
                f'{[b.name for b in _routing_check.backends()]}')
    if tau_grid is None and not (tau_step > 0 and tau_max >= 0
                                 and np.isfinite(tau_step)
                                 and np.isfinite(tau_max)):
        raise ValueError(
            f'need finite tau_step > 0 and tau_max >= 0 (got tau_step='
            f'{tau_step}, tau_max={tau_max}), or pass an explicit tau_grid.')
    # A NaN lag is not "no lag": the Ito equal-time nudge turns a comparison
    # against it into a plain evaluation at tau = 0 and the result looks like
    # a valid number.  Refuse every grid / event array before any work.
    for _name, _arr in (('tau_grid', tau_grid), ('spatial_grid', spatial_grid),
                        ('spatial_points', spatial_points)):
        if _arr is None:
            continue
        _a = np.asarray(_arr, dtype=float)
        _bad = ~np.isfinite(_a)
        if _bad.any():
            _where = np.flatnonzero(_bad.ravel())[:5].tolist()
            raise ValueError(
                f'{_name} must be finite: {int(_bad.sum())} of {_a.size} '
                f'entries are NaN or infinite (flat positions {_where}'
                f'{"..." if _bad.sum() > 5 else ""}).')
    switch_on = _switch_on_request(
        model, k, max_ell, t_min, times, initial_covariance, tau_grid,
        spatial=(bool(model.get('spatial')) or spatial_grid is not None
                 or spatial_points is not None),
        saving=bool(output_npz or output_csv))

    # Auto-pick a Taylor budget that covers every vertex the
    # prediagram enumerator could ask for at the chosen (k, max_ell).
    # Connected diagrams with k external legs at ell loops have at
    # most ``k + 2·ell``-leg vertices, so that's the tight upper
    # bound.  The floor of 2 is structural (anything less wouldn't
    # capture the (1,1) bilinear propagator + (0,0)/(1,0)/(0,1) MF
    # sectors that downstream code unconditionally reads).
    #
    # Historically this floor was 4 — chosen to keep all 1-loop
    # 2-cumulant runs in one cache directory back when the layout
    # was ``saved_models/<model>_taylor<N>/``.  The new layout
    # (``saved_models/<model>/expand_taylor<N>.sobj``) sibling-
    # files different orders cleanly, so the floor is no longer
    # needed for cache-directory cohesion.  Dropping it from 4 → 2
    # saves ~90 min on heavy Bernoulli models at ``k=2, max_ell=0``
    # (the only case where the old floor exceeded the math minimum).
    if taylor_order is None:
        taylor_order = max(k + 2 * max_ell, 2)

    # Accept user-facing natural names and translate to the internal
    # fluctuation names the action / propagator-typing code expect.
    # The mapping comes from ``model['naming_convention']`` (declared
    # by ModelBuilder); pipeline falls back to a classic n/v/m map
    # when the model doesn't declare its own.
    naming_convention   = model.get('naming_convention')
    external_fields_user = list(external_fields)
    external_fields      = normalize_external_fields(
        external_fields, naming_convention=naming_convention)

    # ── Per-phase wall-time tracker (verbose only) ─────────────────
    # Recorded into a dict so callers / notebooks can inspect the
    # breakdown via ``th['phase_walls']`` after the run, and so the
    # final `Done.` line can echo a summary.  Stays None when
    # verbose=False so the bookkeeping is opt-in.
    phase_walls: dict[str, float] | None = {} if verbose else None

    def _phase_time(label: str, t0: float) -> None:
        if phase_walls is None:
            return
        dt = time.perf_counter() - t0
        phase_walls[label] = dt
        print(f'      [{label}] done in {dt:.2f}s')

    # ── 0. Model analyzer gates (api.analyze; G0/G1 before the expansion) ──
    _pf = _make_preflight(preflight, model, k, max_ell, external_fields_user,
                          fundamental, fixed_point_index, policy, accuracy,
                          taylor_order)
    if _pf is not None:
        _pf.declarations()

    # ── 1. FieldTheory expansion (cache-aware) ────────────────────
    # Try the on-disk expand cache before paying the full
    # multivariate-taylor cost.  The cache stores ft._by_tp (the
    # bigrade-classified action dict) at each previously-computed
    # taylor_order; we accept any cached order >= the requested one
    # and downgrade-filter to total degree <= target.
    # inside ``stage_memo()`` (api.covariance): stages [1]-[4] of an earlier
    # call at this parameter point, reused as built
    _memo = _STAGE_MEMO.get()
    _memo_key = (None if _memo is None or spatial_points is not None
                 or spatial_grid is not None else _StageMemo.key(
                     model, taylor_order, use_cache, fundamental,
                     fixed_point_index, mf_dae_n_starts, mf_dae_seed_box))
    _tau_ext = _tau_extent(tau_grid, tau_max)
    _shared = (None if _memo_key is None
               else _memo.get(_memo_key, model, _tau_ext))
    if verbose:
        print(f'[1/7] FieldTheory.expand (taylor_order={taylor_order})'
              + (' -- shared with an earlier call' if _shared else '') + '...')
    _t_phase = time.perf_counter()
    ft = (FieldTheory(model, taylor_order=taylor_order) if _shared is None
          else _shared['ft'])
    try:
        if _shared is None:
            from api import _expand_cache as _ec
            cache_hit = False
            if use_cache:
                cached_order = _ec.find_best_cached_order(model, taylor_order)
                if cached_order is not None:
                    _ec.prepare_for_load(ft)
                    cache_hit = _ec.load_expand(
                        model, ft,
                        target_order=taylor_order,
                        cached_order=cached_order,
                        verbose=verbose,
                    )
            if not cache_hit:
                ft.expand()
                if use_cache:
                    try:
                        _ec.save_expand(model, ft, verbose=verbose)
                    except Exception as e:
                        if verbose:
                            print(f'      [expand-cache] save failed ({e!r}); '
                                  f'continuing without persistence.')

            sanity_ok = ft.sanity_check(verbose=verbose)
            if not sanity_ok:
                raise RuntimeError(
                    'FieldTheory.sanity_check() failed — see printout above.'
                )
    except Exception as _exp_exc:
        if _pf is not None:               # e.g. no real saddle: name it
            _pf.failed('expanded', _exp_exc)
        raise
    if _pf is not None:
        _pf.expanded(ft)                  # G1: species, unpaired fields

    # ── 1b. Delay kernels: exp(-i omega d) in a kernel image (api.delay) ──
    if not model.get('_delay_pade') and model.get('kernel_ft_image') is not None:
        from api import delay as _delay
        _delays = _delay.kernel_delays(model, ft._ns)
        if _delays and switch_on is not None:
            # the delay driver's exact tree and its Pade-order search are
            # stationary (not combined yet)
            from engine.integration.time_domain import statespace as _ss_dl
            raise _ss_dl.SwitchOnUnsupported(
                'delay_kernels', 'models with delay kernels (api.delay) are '
                'computed for the stationary state only')
        if _delays:
            return _delay.compute_delayed(model, _delay_call, _delays, ns=ft._ns)

    if _shared is None:
        vtypes = extract_vertex_types(ft)
        stypes = extract_source_types(ft)
    else:
        vtypes, stypes = _shared['vtypes'], _shared['stypes']
    n_noise = sum(1 for s in stypes if isinstance(s, NoiseSourceType))
    if verbose:
        print(f'      vtypes: {len(vtypes)}, sources: {len(stypes)} '
              f'(NoiseSourceType: {n_noise})')
        _print_action_sectors(ft)
    _phase_time('expand', _t_phase)

    # ── 2. Propagator (symbolic, cached) ──────────────────────────
    if verbose:
        print('[2/7] Build propagator (K_ker → K_ft → G_ft → D_delta)...')
    _t_phase = time.perf_counter()
    prop = (build_propagator(ft, model, use_cache=use_cache, verbose=verbose)
            if _shared is None else _shared['prop'])  # poles filled in [4]
    _phase_time('propagator', _t_phase)
    if _pf is not None:
        _pf.propagator(prop)              # G3: propagator class

    # ── 3. Mean-field solve + num_params assembly ─────────────────
    if verbose:
        print('[3/7] Solve MF self-consistency...')
    _t_phase = time.perf_counter()
    try:
        # One multi-root solver for every model (``.equation`` rows or
        # ``.set_mf_equation`` saddle equations); its root
        # census (mf_all_roots, mf_index_used, ...) is surfaced below.
        mf = solve_mean_field(
            ft, model, fundamental, verbose=verbose,
            fixed_point_index=fixed_point_index,
            n_starts=mf_dae_n_starts,
            seed_box=mf_dae_seed_box,
        ) if _shared is None else _shared['mf']
    except Exception as _mf_exc:
        if _pf is not None:               # name the cause when a gate can
            _pf.failed('mean_field', _mf_exc)
        raise
    num_params = mf['num_params']
    _phase_time('mean_field', _t_phase)
    if _pf is not None:
        _pf.mean_field(mf)                # G2 / G3 spatial / G4 / undriven
    else:
        # the analyzer (G2_UNSTABLE_SADDLE) is off: an unstable selection is
        # refused here by name, not later as a pole-free propagator
        from api._mean_field_dae import refuse_unstable_saddle
        refuse_unstable_saddle(model, fundamental, mf)

    # ── 3.5 Spatial short-circuit ─────────────────────────────────
    # A spatial model's propagator carries the inert ``Laplacian``
    # symbol, which the (ω) pole-finder and time-domain Phase J can't
    # consume.  Spatial models route to the full-diagram momentum
    # integrator instead (Symanzik ∫dᵈℓ → causal chambers → q→x FT),
    # which returns the real-space correlator C(x, τ).  See
    # docs/spatial_pipeline.md.  Requires spatial_grid (the x points).
    if spatial_grid is not None and not model.get('spatial'):
        import warnings as _warnings
        _warnings.warn(
            'spatial_grid was provided but the model is not spatial (no field '
            'declares spatial_dim>=1); it is ignored and the temporal C(τ) is '
            'returned.', stacklevel=2)

    # ── Routing (api.routing): the first registered backend that applies.
    # The built-in table reproduces the dispatch of 0.2.0 exactly (spatial
    # short-circuit, per-diagram Phase J) and raises the same
    # errors when nothing applies.
    from api import routing as _routing
    _request = _routing.RouteRequest(
        model_name=model.get('name', '<unnamed>'),
        spatial=bool(model.get('spatial')),
        k=k, max_ell=max_ell,
        has_spatial_grid=spatial_grid is not None,
        has_spatial_points=spatial_points is not None,
        initial_mode=('switch_on' if switch_on is not None else
                      (model.get('initial') or {}).get('mode', 'stationary')),
        requested=backend,
        traits=_pf.traits_dict() if _pf is not None else {})
    route = _routing.select(_request)
    if _pf is not None:
        _pf.route = route
    state = _PipelineState(
        model=model, k=k, max_ell=max_ell, ft=ft, prop=prop,
        num_params=num_params, mf=mf, external_fields=external_fields,
        external_fields_user=external_fields_user, fundamental=fundamental,
        spatial_points=spatial_points, spatial_grid=spatial_grid,
        tau_grid=tau_grid, tau_max=tau_max, tau_step=tau_step,
        taylor_order=taylor_order, origin_leaf_idx=origin_leaf_idx,
        use_cache=use_cache, parallel=parallel,
        spatial_parallel=spatial_parallel, spatial_n_q=spatial_n_q,
        n_workers=n_workers,
        discard=discard,
        points=(points if points is not None or k < 3 or discard is False
                or tau_grid is None or np.asarray(tau_grid).size != 1
                or float(np.asarray(tau_grid).ravel()[0]) != 0.0
                else [(0.0,) * k]),     # tau_grid=[0.0]: the equal-time point
        verbose=verbose, naming_convention=naming_convention,
        vtypes=vtypes, stypes=stypes, phase_walls=phase_walls,
        phase_time=_phase_time, t_phase=_t_phase, preflight=_pf,
        route=route, evaluated_grid=False, switch_on=switch_on,
        # stage sharing (``stage_memo``): the poles of [4] are already in
        # ``prop``, or the stages are stored for later calls once they are
        poles_shared=_shared is not None,
        stage_store=(None if _memo_key is None or _shared is not None else
                     (lambda _m=_memo, _k=_memo_key: _m.put(
                         _k, model, _tau_ext, ft=ft, vtypes=vtypes,
                         stypes=stypes, prop=prop, mf=mf))))
    _snap = _counter_snapshot()
    try:
        result = route.backend.run(state)
    finally:                              # restores the maxima on a raise too
        _counters, _complete = _counter_delta(_snap)
    # the Phase J engine census of the run (subsets on the state-space
    # route, declined by reason, call-time fallbacks by reason)
    if isinstance(result, dict) and 'phase_j_by_ell' in result:
        _engine_census(result, _counters)
    # Never silent: a diagram Phase J could not evaluate is named (status,
    # warning; strict policy raises after the report is attached).
    _failures = None
    if isinstance(result, dict) and 'phase_j_by_ell' in result:
        _failures = _phase_j_failures(result['phase_j_by_ell'])
        result['phase_j_incomplete'] = _failures
        result['status'] = 'incomplete' if _failures else 'ok'
    # Spatial: a result whose quadrature accuracy is not certified is
    # 'approximate' (never 'ok'); G6 reports it; strict policy raises below
    _sp_acc = []
    if isinstance(result, dict) and route.backend.name == 'spatial_integrator':
        _sp_acc = _spatial_accuracy(result, result.get('tau_grid', tau_grid),
                                    max_ell)
        result['spatial_accuracy'] = _sp_acc
        result['status'] = 'approximate' if _sp_acc else 'ok'
    _marcus = _marcus_truncation(model, result, k, max_ell)
    if _marcus is not None:
        result['marcus_truncation'] = _marcus
        _over = [r for r in _marcus if r.get('status') == 'above']
        if _over:
            _w = max(_over, key=lambda r: r['relative'])
            _msg = (_marcus_detail(_w) + f" (threshold {MARCUS_TRUNCATION_RTOL:g}); "
                    + _marcus_remedy(_w) + " -- see result['marcus_truncation'].")
            from api.save import _ensure_warning_shown
            _ensure_warning_shown(MarcusTruncationWarning, 'always',
                                  message=_msg, stacklevel=2)
            warnings.warn(_msg, MarcusTruncationWarning, stacklevel=2)
    if _pf is not None:
        # G6 sizes its accuracy estimate on ``C_tau``; a switch-on run's
        # values are ``C_times`` (a read-only view: G6 never writes)
        _pf.after_run(
            dict(result, C_tau=result.get('C_times'))
            if switch_on is not None and isinstance(result, dict)
            else result, _counters, complete=_complete,
            evaluated=state.evaluated_grid)
    _attach_provenance(result, route, _pf)
    _attach_jitter(result, model)
    if isinstance(result, dict) and isinstance(result.get('route'), dict) \
            and result.get('phase_j_engine') is not None:
        result['route']['phase_j_engine'] = result['phase_j_engine']
    if switch_on is not None and isinstance(result, dict):
        _switch_on_check(result, _failures, model.get('name', '<unnamed>'))
    if _failures:
        _msg = _phase_j_failure_message(_failures,
                                        model.get('name', '<unnamed>'))
        if _policy_is_strict(policy):
            raise PhaseJIncompleteError(_msg, result=result,
                                        failures=_failures)
        # visible under ``sage -python`` / the Sage kernel too, whose
        # launcher sets PYTHONWARNINGS=ignore
        from api.save import _ensure_warning_shown
        _ensure_warning_shown(PhaseJIncompleteWarning, 'always',
                              message=_msg, stacklevel=2)
        warnings.warn(_msg, PhaseJIncompleteWarning, stacklevel=2)
    if _sp_acc:
        _msg = ('spatial result is approximate: ' + '; '.join(
            r['note'] + (f' (0.7x-coarser grid spread '
                         f'{r["coarse_spread"]:.2e} of max|C|)'
                         if r.get('coarse_spread') is not None else '')
            for r in _sp_acc) + ' -- see result["spatial_accuracy"].')
        if _policy_is_strict(policy):
            raise SpatialAccuracyError(_msg, result=result)
        from api.save import _ensure_warning_shown
        _ensure_warning_shown(SpatialAccuracyWarning, 'always',
                              message=_msg, stacklevel=2)
        warnings.warn(_msg, SpatialAccuracyWarning, stacklevel=2)
    if _spectra_opts is not None and isinstance(result, dict):
        _spectra_mod.attach(result, _spectra_opts)
    if route.backend.standard_tail:
        _standard_tail(state, result, output_npz, output_csv)
    return result


def _run_spatial(state):
    """Backend ``spatial_integrator`` (``api.routing``): the full-diagram
    momentum integrator.  ``state``: the pipeline objects after the
    mean-field stage (``_PipelineState``); the applicability checks
    (k = 2 needs ``spatial_grid``, k != 2 ``spatial_points``, ``max_ell <=
    2``, stationary initial state) are the backend's ``applies``."""
    (model, k, max_ell, ft, prop, num_params, mf, external_fields,
     spatial_points, spatial_grid, tau_grid, tau_max, tau_step, verbose,
     parallel, spatial_parallel, n_workers, spatial_n_q, fundamental,
     naming_convention, _phase_time, _t_phase) = (
        state.model, state.k, state.max_ell, state.ft, state.prop,
        state.num_params, state.mf, state.external_fields,
        state.spatial_points, state.spatial_grid, state.tau_grid,
        state.tau_max, state.tau_step, state.verbose, state.parallel,
        state.spatial_parallel, state.n_workers, state.spatial_n_q,
        state.fundamental, state.naming_convention, state.phase_time,
        state.t_phase)
    import numpy as _np
    from engine.integration.spatial.pipeline_bridge import (
        compute_spatial_correlator_via_pipeline,
    )
    from engine.integration.spatial.spatial_correlator import (
        free_two_point,
    )
    if k != 2:
        # ── general-k spatial path: explicit evaluation events ──
        from engine.integration.spatial.pipeline_bridge import (
            compute_spatial_kpoint,
        )
        try:
            vals, kp_info = compute_spatial_kpoint(
                ft, model, prop, num_params, external_fields,
                spatial_points, max_ell=max_ell, verbose=verbose)
        except NotImplementedError as _e:
            if 'single-field' not in str(_e):
                raise
            # ── COUPLED multi-field theory at k>=3: the spectral-
            # assignment driver (scalar diffusion, plain vertices).
            from engine.integration.spatial.heat_kernel import (
                reaction_diffusion_matrices)
            from engine.integration.spatial.pipeline_bridge import (
                compute_coupled_kpoint, _norm_sr)
            from engine.integration.spatial.spectral_propagator import (
                split_reference_diffusion)
            _ns = ft._ns
            _Msr, _Dsr, _Vsr = reaction_diffusion_matrices(
                prop['K_ft'], prop['omega'], SR.var('k'),
                _ns.Laplacian, getattr(_ns, 'GradX', None) or SR.var('GradX'))
            _nf = int(_Msr.nrows())
            _sub = _norm_sr(num_params)

            def _numm(sm):
                return _np.array(
                    [[complex(SR(sm[a_, b_]).subs(_sub))
                      for b_ in range(_nf)] for a_ in range(_nf)],
                    dtype=complex)
            _Mc, _Dc = _numm(_Msr), _numm(_Dsr)
            # M and D are real for a real action; never drop an imaginary
            # part silently (relative to the matrix's own scale)
            for _nm, _mc in (('reaction M', _Mc), ('diffusion D', _Dc)):
                if _np.max(_np.abs(_mc.imag)) > 1e-12 * max(
                        float(_np.max(_np.abs(_mc))), 1e-300):
                    raise NotImplementedError(
                        f'coupled k>=3: the {_nm} matrix is complex at the '
                        f'saddle; a real matrix is required.')
            _Mn, _Dn = _Mc.real, _Dc.real
            # the drift V = i*v*I (GradX -> i k, purely imaginary): a common
            # drift is boosted away exactly inside compute_coupled_kpoint;
            # any other V refuses there (pipeline_bridge._drift_velocity)
            _D0, _Dhat = split_reference_diffusion(_Dn)
            tree_info_k = {
                'M': _Mn, 'Dhat': _Dhat, 'D0': float(_D0),
                'bc_mode': (model.get('boundary')
                            or {}).get('mode', 'infinite')}
            vals, kp_info = compute_coupled_kpoint(
                ft, model, prop, num_params, external_fields,
                spatial_points, tree_info_k, max_ell=max_ell,
                verbose=verbose)
        _phase_time('spatial-kpoint', _t_phase)
        return {
            'C_kpoint':        vals,
            'C_kpoint_by_ell': kp_info['per_ell'],
            'points':          _np.asarray(spatial_points, dtype=float),
            'spatial_info':    kp_info,
            'k': k, 'max_ell': max_ell,
            'mf':              mf,
        }
    tau_grid = (_np.asarray(tau_grid, dtype=float) if tau_grid is not None
                else _np.arange(-tau_max, tau_max + tau_step * 0.5, tau_step))
    spatial_grid_arr = _np.asarray(spatial_grid, dtype=float)
    if verbose:
        print('[4/7] (spatial) Momentum stays symbolic (Laplacian) — skip the '
              'ω pole-finder + time-domain Phase J; route to the full-diagram '
              'integrator.')
        print(f'[spatial] {"loop" if max_ell >= 1 else "tree-level"} '
              f'C(x, τ): max_ell={max_ell}, {len(tau_grid)} τ × '
              f'{len(spatial_grid_arr)} x points...')
    # Route through the SHARED pipeline (Stage B/C): the bridge runs the
    # real diagram pipeline at sample momenta to CERTIFY the per-mode
    # (A,B,N) structure, then does the q→x FT.  At max_ell≥1 the FULL-DIAGRAM
    # integrator sums EVERY enumerated diagram up to max_ell loops through one
    # genuine integral (Symanzik ∫dᵈℓ → causal-chamber time integral →
    # ret+adv), weighted by the enumeration 𝒮(Γ) — no shortcut, no diagram
    # dropped (the internal design notes (not distributed)).
    if max_ell >= 1:
        from engine.integration.spatial.pipeline_bridge import (
            compute_spatial_correlator_generic,
        )
        # Spatial parallelism is THREAD-based (no fork → safe in Jupyter/macOS;
        # numpy releases the GIL for the per-diagram linalg/quadrature).  On by
        # default; disable with spatial_parallel=False.  (Fork is forbidden here.)
        C_tau_x, sp_info = compute_spatial_correlator_generic(
            ft, model, prop, num_params, external_fields,
            tau_grid, spatial_grid_arr, verbose=verbose, max_ell=max_ell,
            parallel=(parallel and spatial_parallel), n_workers=n_workers,
            n_q=int(spatial_n_q),
        )
    else:
        C_tau_x, sp_info = compute_spatial_correlator_via_pipeline(
            ft, model, prop, num_params, external_fields,
            tau_grid, spatial_grid_arr, verbose=verbose, stage_headers=True,
        )
    if verbose:
        _cmr = sp_info.get("certify_max_rel")
        _cmr_s = f'{_cmr:.1e}' if _cmr is not None else 'n/a (coupled driver)'
        print(f'[spatial] done — C(x,τ) ready; tree-mode certified='
              f'{sp_info.get("pipeline_certified")} '
              f'(max rel {_cmr_s})')
    # x=0 slice as the conventional C_tau (matches the time-only
    # API's C_tau shape).
    x0_idx = int(_np.argmin(_np.abs(spatial_grid_arr)))
    C_tau = C_tau_x[:, x0_idx].copy()

    # ── Result callables, built from the RETURNED grid ───────────────────
    # The spatial integrator evaluates C(x,τ) on the requested (tau_grid,
    # spatial_grid) only; the callables read that grid (the highest loop
    # order, and the per-order increments for total_C_by_ell), so
    # total_C(τ, x) == C_tau_x[iτ, ix] at every grid point.  Off the grid
    # they refuse — except the single-field TREE, which is exact anywhere
    # (free_two_point / free_two_point_d / the periodic image sum).
    _C_by_order = {int(el): _np.asarray(C) for el, C in
                   (sp_info.get('C_by_order') or {0: C_tau_x}).items()}
    _orders = sorted(_C_by_order)
    _incr = {el: (_C_by_order[el] - _C_by_order[_orders[i - 1]] if i
                  else _C_by_order[el])
             for i, el in enumerate(_orders)}
    _sp_d = int(sp_info.get('spatial_dim', 1))
    _tree_exact = (_orders == [0] and not sp_info.get('coupled')
                   and sp_info.get('modes'))
    _tau_arr = _np.asarray(tau_grid, dtype=float)
    # a constant drift was boosted away (C_v(x, tau) = C_0(x - v tau, tau)):
    # the off-grid exact tree is the drift-free mode at the boosted point
    _v_drift = float(sp_info.get('drift_velocity') or 0.0)

    def _spatial_callable(table, exact_tree):
        def fn(*tau_then_x):
            """C(τ) at x=0 (1 arg) or C(τ, x) (2 args), read from the
            evaluated (tau_grid, spatial_grid)."""
            if len(tau_then_x) not in (1, 2):
                raise TypeError('spatial total_C takes (tau) or (tau, x).')
            tau = float(tau_then_x[0])
            xq = float(tau_then_x[1]) if len(tau_then_x) == 2 else 0.0
            if not (_np.isfinite(tau) and _np.isfinite(xq)):
                raise ValueError(f'spatial total_C: tau and x must be finite '
                                 f'(got tau={tau}, x={xq}).')
            it = _np.flatnonzero(_np.abs(_tau_arr - tau)
                                 <= 1e-12 * (1.0 + abs(tau)))
            ix = _np.flatnonzero(_np.abs(spatial_grid_arr - xq)
                                 <= 1e-12 * (1.0 + abs(xq)))
            if it.size and ix.size:
                return complex(table[int(it[0]), int(ix[0])])
            if exact_tree:
                from engine.integration.spatial.spatial_correlator import (
                    free_two_point_d, free_two_point_box)
                val = 0j
                xq = xq - _v_drift * tau
                for (A, B, N) in sp_info['modes']:
                    if _sp_d == 1:
                        val += free_two_point(A, B, N, xq, tau,
                                              bc_mode=sp_info['bc_mode'],
                                              L=sp_info['L'])
                    elif sp_info['bc_mode'] == 'periodic':
                        val += free_two_point_box(A, B, N, xq, tau, _sp_d,
                                                  sp_info['L'])
                    else:
                        val += free_two_point_d(A, B, N, xq, tau, _sp_d)
                return val
            raise ValueError(
                f'spatial total_C: (tau={tau}, x={xq}) is not on the '
                f'evaluated grid.  The spatial integrator computes C(x, tau) '
                f'only at the requested tau_grid x spatial_grid points; '
                f'rerun with this point in tau_grid / spatial_grid '
                f'(or index C_tau_x / C_tau_x_by_order directly).')
        return fn

    total_C = _spatial_callable(_C_by_order[_orders[-1]], _tree_exact)
    total_C_by_ell = {el: _spatial_callable(_incr[el], _tree_exact and el == 0)
                      for el in _orders}

    # MF values (for the result dict) — reuse the saddle solve.
    mf_values_sp = {}
    for pname, vals in (mf.get('mf_values') or {}).items():
        mf_values_sp[pname] = vals

    return {
        'total_C':        total_C,
        'total_C_by_ell': total_C_by_ell,
        'C_tau':          C_tau,
        'C_tau_x':        C_tau_x,
        # cumulative C(x,τ) at each loop order {0: tree, 1: +1-loop, …, max_ell}
        # — the WHOLE progression from ONE call (the highest-ℓ run already
        # contains every lower order; no need to re-run per order).
        'C_tau_x_by_order': sp_info.get('C_by_order', {0: C_tau_x}),
        # the temporal schema's names: ``status`` and the per-loop-order
        # INCREMENTS keyed by ell (tree, then each loop order's own
        # contribution: they sum to C_tau_x; ``_by_order`` above is cumulative)
        'status':         'ok',
        'C_tau_x_by_ell': dict(_incr),
        'tau_grid':       tau_grid,
        'spatial_grid':   spatial_grid_arr,
        'chi_grid':       spatial_grid_arr,   # canonical alias of spatial_grid
        'spatial_info':   sp_info,
        'mf_values':      mf_values_sp,
        'mf':             MeanField(mf_values_sp,
                                    naming_convention=naming_convention),
        'params':         Parameters(fundamental),
        'num_params':     num_params,
        'propagator':     prop,
        'config': {
            'k':               k,
            'max_ell':         max_ell,
            'fundamental':     fundamental,
            'parameters':      fundamental,
            'external_fields': external_fields,
            'tau_max':         tau_max,
            'tau_step':        tau_step,
            'spatial':         True,
            'model_name':      model.get('name', '<unnamed>'),
            'phase_j_convention': _phase_j_convention(),
            'phase_j_flags':      _phase_j_flags(),
        },
    }


def _run_phase_j_per_diagram(state):
    """Backend ``phase_j_per_diagram`` (``api.routing``)."""
    return _run_phase_j(state)


def _run_phase_j(state):
    """The temporal Phase J stages [4/7]-[7/7]: poles and residues, the
    diagram enumeration, the prefactor classification and the time-domain
    integration, and the result dict.
    ``state``: the pipeline objects after the mean-field stage."""
    (model, k, max_ell, ft, prop, num_params, mf, external_fields,
     external_fields_user, fundamental, tau_grid, tau_max, tau_step,
     taylor_order, origin_leaf_idx, use_cache, parallel, n_workers,
     verbose, naming_convention, vtypes, stypes, phase_walls,
     _phase_time) = (
        state.model, state.k, state.max_ell, state.ft, state.prop,
        state.num_params, state.mf, state.external_fields,
        state.external_fields_user, state.fundamental, state.tau_grid,
        state.tau_max, state.tau_step, state.taylor_order,
        state.origin_leaf_idx, state.use_cache, state.parallel,
        state.n_workers, state.verbose, state.naming_convention,
        state.vtypes, state.stypes, state.phase_walls, state.phase_time)
    # ── 4. Numerical poles + residues (fills prop in place) ───────
    if verbose:
        print('[4/7] Compute numerical poles + residue matrices...')
    _t_phase = time.perf_counter()
    if not getattr(state, 'poles_shared', False):
        compute_poles_and_residues(prop, num_params, verbose=verbose,
                                   tau_max=_tau_extent(tau_grid, tau_max),
                                   resolve_clusters=True)
        if getattr(state, 'stage_store', None) is not None:
            state.stage_store()
    _phase_time('poles', _t_phase)
    # whether the state-space route takes the whole run (no noise-kernel
    # sources, every conductance kernel realizable, a Hurwitz realization)
    ss_route = _ss_route_prediction(
        prop, num_params, list(vtypes or ()) + list(stypes or ()))
    switch_on = getattr(state, 'switch_on', None)
    if switch_on is not None:
        if not ss_route.get('covers_run'):
            from engine.integration.time_domain import statespace as _ss_so
            raise _ss_so.SwitchOnUnsupported(
                ss_route.get('reason') or 'not_covered',
                'the run-level prediction of the state-space route')
        _ledger_switch_on(state.preflight, switch_on)
    if state.preflight is not None:
        # G3 pole statistics (api.analyze); informational when the
        # state-space route covers the run
        state.preflight.poles(prop, ss_route=ss_route)

    # ── 5. Field-index maps + prediagram + typed diagram pipeline ─
    if verbose:
        print(f'[5/7] Enumerate prediagrams (k={k}, max_ell={max_ell})...')
    _t_phase = time.perf_counter()
    ring_var_names = list(ft._ns._ring_var_names)
    n_tilde = ft._n_tilde
    resp_idx, phys_idx = build_field_index_map(ring_var_names, n_tilde)
    for f in external_fields:
        if f not in phys_idx:
            raise ValueError(
                f'external field {f} not in phys_idx '
                f'{sorted(phys_idx.keys())}'
            )

    unique_by_ell, multiplicity_by_ell, all_unique = enumerate_unique_diagrams(
        ft, model,
        k               = k,
        max_ell         = max_ell,
        external_fields = external_fields,
        G_ft            = prop['G_ft'],
        resp_idx        = resp_idx,
        phys_idx        = phys_idx,
        vtypes          = vtypes,
        stypes          = stypes,
        use_cache       = use_cache,
        # Cache off: the eager records, not the streamed ones, until Phase
        # J's nquad fallback is representative-independent (see the flag).
        stream_prediagrams = _pd_cache.TEMPORAL_CACHE_OFF_STREAMS,
        parallel        = parallel,
        n_workers       = n_workers,
        verbose         = verbose,
    )
    _phase_time('diagrams', _t_phase)

    # ── 6. Diagram-level prefactor classification + kernel groups ─
    if verbose:
        print('[6/7] Classify coefficient factors per diagram...')
    _t_phase = time.perf_counter()
    time_dep_params = model.get('time_dependent_parameters', []) or []
    noise_structure = model.get(
        'noise_structure', {'temporal_type': 'white', 'amplitude_params': []}
    )
    # Hard stop (not only the analyzer's G2_NOISE_STRUCTURE, which
    # preflight=False / a downgrade would skip): Phase J never applies a
    # temporal noise kernel, so a non-white temporal_type would be computed
    # as white noise with status 'ok'.
    _ntype = (noise_structure or {}).get('temporal_type', 'white')
    if _ntype != 'white':
        raise ValueError(
            f"noise_structure temporal_type {_ntype!r} is not supported by "
            "compute_cumulants (white noise only; the noise kernel would be "
            "dropped). Write colored noise as a Markov embedding (an extra "
            "OU-driven field) or a kernel Conv term instead.")

    diagram_records = []
    kernel_groups = []
    # Walk by ell so each record carries the ell tag — needed for the
    # per-loop-order Phase J decomposition in step [7].
    for ell in sorted(unique_by_ell.keys()):
        mults = multiplicity_by_ell.get(ell, [1] * len(unique_by_ell[ell]))
        for td, mult in zip(unique_by_ell[ell], mults):
            info = classify_coefficient_factors(
                td, time_dep_params, noise_structure
            )
            # Path A: ``combinatorial_factor`` in symmetry.py now
            # computes 𝒮(Γ) = ∏ n_leg! / |Aut_fixed_ext(Γ)| using the
            # full coloured incidence-graph automorphism order.  That
            # already accounts for every Feynman-rule symmetry —
            # same-type vertex swaps, parallel-edge swaps, self-loop
            # leg swaps — so multiplying by ``mult`` (the dedup
            # equivalence-class size) here would double-count and is
            # disabled.
            combined_prefactor = SR(info['scalar_prefactor'])
            kernel_groups.append({
                'diagrams':           [td],
                'combined_prefactor': combined_prefactor,
            })
            diagram_records.append({
                'typed_diagram':      td,
                'classify':           info,
                'combined_prefactor': combined_prefactor,
                # M(F) in the resolved convention of the paper's
                # Remark rem_multiplicity_convention: prod n_leg! /
                # |Aut_ext|, the weight that multiplies the literal
                # vertex coefficients (combined_prefactor = M * coeffs).
                'multiplicity':       info['Scal'],
                # size of the deduplication class the representative
                # stands for — diagnostic only, NOT a weight (it used
                # to be stored under 'multiplicity', which the TikZ
                # panel captions then printed as M).
                'dedup_class_size':   mult,
                'ell':                ell,
            })
    _phase_time('classify', _t_phase)

    # ── 7. Phase J time-domain integration (per ell) ──────────────
    if verbose:
        print(f'[7/7] Phase J: compute_correction_td per ell '
              f'(0..{max_ell})...')
    _t_phase = time.perf_counter()
    tau_grid = (np.asarray(tau_grid, dtype=float) if tau_grid is not None
                else np.arange(-tau_max, tau_max + tau_step * 0.5, tau_step))

    propagator_data = {
        'K_ker':   prop['K_ker'],
        'K_ft':    prop['K_ft'],
        'G_ft':    prop['G_ft'],
        'adj_ft':  prop['adj_ft'],
        'D_omega': prop['D_omega'],
        'D_delta': prop['D_delta'],
        't_var':   prop['t_var'],
        'omega':   prop['omega'],
        'nf':      prop['nf'],
        'pole_vals': prop['pole_vals'],
        'C_mats':    prop['C_mats'],
        # Identifies the model in Phase J's warnings (the hardened
        # quadrature fallback warns once per model, source and loop order).
        'model_name': model.get('name'),
    }

    # the memory pre-pass, before any modesum plan is built.  The
    # budget is read ONCE for the run (its default depends on what
    # the process holds), so the pre-pass and the running guard agree.
    from engine.integration.time_domain import final_integral as _fi_b
    _run_budget = (_fi_b.run_budget() if hasattr(_fi_b, 'run_budget')
                   else _fi_b.mem_budget('PHASE_J_MEM_BUDGET_GB')
                   if hasattr(_fi_b, 'mem_budget') else None)
    # evaluate-and-discard needs fixed evaluation points (k <= 2
    # on the τ grid, k >= 3 at ``points``) and a single pass
    discard = getattr(state, 'discard', False)
    if discard not in (False, True, 'auto'):
        raise ValueError(f"discard={discard!r}: expected False, True or "
                         f"'auto'")
    kpoints = getattr(state, 'points', None) if k > 2 else None
    _no_points = k > 2 and kpoints is None and switch_on is None
    _discard_why = (
        'switch-on transients (t_min)' if switch_on is not None else
        'k >= 3 without fixed evaluation points (pass points=, the '
        'external-time tuples to evaluate; dd.run passes its slice points)'
        if _no_points else
        'the pole-cluster deformation averages several passes'
        if (prop.get('pole_deformation') is not None
            and not ss_route.get('bypass_deformation')) else None)
    if discard is True and _discard_why is not None:
        raise ValueError(f'discard=True is not available for this run: '
                         f'{_discard_why}; use discard=False (or '
                         f"'auto', which then stays off)")
    mem_estimate = _phase_j_memory_preflight(
        diagram_records, propagator_data, num_params, prop, model, k,
        max_ell, verbose, state.preflight, ss_route=ss_route,
        n_tau=(len(tau_grid) if k <= 2 else 1) if switch_on is None
        else len(switch_on['times']), run_budget=_run_budget,
        discard=discard, discard_ok=_discard_why is None,
        discard_hint=(_DISCARD_POINTS_HINT if _no_points else None))
    _discard_on = (bool(mem_estimate.get('discard')) if mem_estimate
                   else (discard is True))
    # 'auto' with the estimate under the budget: switch on mid-run when
    # the measured growth crosses a fraction of the budget
    _discard_defer = (discard == 'auto' and not _discard_on
                      and _discard_why is None and mem_estimate is not None
                      and mem_estimate.get('node_runs', 1) == 1
                      and mem_estimate.get('budget_bytes') is not None)
    # the growth guard's refusal says what evaluate-and-discard would do
    _growth_hint = (
        None if _discard_on else
        "Evaluate-and-discard applies to this run: discard=True evaluates "
        "each diagram at the run's fixed points and releases it, so Phase J "
        "holds one diagram at a time." if _discard_why is None else
        f'Evaluate-and-discard (discard=True) would hold one diagram at a '
        f'time; at k >= 3 it {_DISCARD_POINTS_HINT}' if _no_points else None)

    # Pick the τ-grid evaluation pattern by k (None = no grid eval)
    if k == 1:
        tau_points = [(float(t),) for t in tau_grid]
    elif k == 2:
        # Single-axis slice: vary leaf 1 over tau_grid, leaf 0 pinned at 0.
        # Itô LEFT-limit at equal time (Itô ⇒ left-continuous, Θ(0)=Θ(0⁻)): the
        # τ=0 entry was sampled at t_1 = −_ITO_EPS rather than exactly 0.  The
        # propagators are strictly-causal retarded with Θ(0)=0; landing the τ-grid
        # on that measure-zero step discontinuity drops a coincident-ordering
        # contribution and produces a spurious single-point dip at τ=0.  Nudging
        # to 0⁻ keeps the surviving causal chamber and gives the left limit (= the
        # right limit for an even/stationary C; differs only at a genuine τ=0
        # jump, where Itô takes the left).  The grid LABEL stays 0.
        # (Since Oct 2026 the τ = 0 point is evaluated AT equal times by
        # default, where the engine's exact tie rule gives C(0⁻) exactly;
        # the −_ITO_EPS sample is DAEDALUS_TAU0=nudge, see ``_tau0_mode``.)
        _lag0 = _tau0_lag()
        # a grid value is the equal-time point when it is 0 to round-off of
        # the grid's own scale (an absolute 1e-12 turned every lag of a model
        # in fast time units, rates ~1e12, into tau = 0)
        _tz = 1e-12 * max([0.0] + [abs(float(t)) for t in tau_grid])
        tau_points = [(0.0, float(t) if abs(float(t)) > _tz else _lag0)
                      for t in tau_grid]
    else:
        tau_points = None   # k≥3: caller handles grid evaluation
    # the points each diagram is evaluated on under evaluate-and-discard:
    # the τ grid's (k <= 2, under the state-space τ batch, as
    # ``total_C_batch``), or ``points`` one call at a time (k >= 3, as the
    # result is read)
    _discard_pts = tau_points if k <= 2 else kpoints
    if switch_on is not None:
        # absolute times (``times``), evaluated below; no τ grid
        tau_points = None
        so_points = [tuple(float(x) for x in row)
                     for row in switch_on['times']]

    # Run Phase J once per ell so we get the per-loop-order
    # decomposition the saver / notebook plots need.  Diagrams within
    # an ell are summed in that ell's total_C; the master total_C is
    # the sum across ells.  One pass runs every ell on ``pdata``;
    # near-degenerate or repeated poles (``prop['pole_deformation']``)
    # average passes over a deformation of the propagator
    # (``api._pole_clusters``).
    def _phase_j_pass(pdata):
        phase_j_by_ell  = {}
        total_C_by_ell  = {}
        C_tau_by_ell    = {}
        for ell in sorted(unique_by_ell.keys()):
            records_ell = [r for r in diagram_records if r['ell'] == ell]
            if not records_ell:
                # No diagrams at this ell (e.g. tree-level for some
                # subset configs).  Contribute zero.
                total_C_by_ell[ell] = (lambda *t: 0.0 + 0.0j)
                phase_j_by_ell[ell] = None
                if tau_points is not None:
                    C_tau_by_ell[ell] = np.zeros(len(tau_grid), dtype=complex)
                elif switch_on is not None:
                    C_tau_by_ell[ell] = np.zeros(len(so_points), dtype=complex)
                else:
                    C_tau_by_ell[ell] = None
                continue

            if _discard_on or _discard_defer:
                # each diagram evaluated on the points, then released
                # (``_discard_defer``: from the switch on)
                with _phase_j_warning_scope():
                    td_result_ell = compute_correction_td(
                        typed_diagrams   = [r['typed_diagram']
                                            for r in records_ell],
                        prefactors       = [r['combined_prefactor']
                                            for r in records_ell],
                        k                = k,
                        propagator_data  = pdata,
                        external_fields  = external_fields,
                        num_params       = num_params,
                        origin_leaf_idx  = origin_leaf_idx,
                        discard_points   = _discard_pts,
                        **({} if k <= 2 else {'discard_prefetch': False}),
                        **({} if _discard_on else {
                            'discard_defer': DISCARD_AUTO_GROWTH_FRACTION}),
                    )
            else:
                td_result_ell = compute_correction_td(
                    typed_diagrams   = [r['typed_diagram']
                                        for r in records_ell],
                    prefactors       = [r['combined_prefactor']
                                        for r in records_ell],
                    k                = k,
                    propagator_data  = pdata,
                    external_fields  = external_fields,
                    num_params       = num_params,
                    origin_leaf_idx  = origin_leaf_idx,
                )
            if switch_on is not None:
                total_C_by_ell[ell] = _phase_j_scoped(_switch_on_callable(
                    td_result_ell['total_C'], switch_on['t_min']))
            else:
                total_C_by_ell[ell] = _phase_j_scoped(
                    _ito_nudge_callable(td_result_ell['total_C'], k))
            phase_j_by_ell[ell] = td_result_ell

            if switch_on is not None:
                with _phase_j_warning_scope():
                    C_tau_by_ell[ell] = np.array(
                        [complex(total_C_by_ell[ell](*pt))
                         for pt in so_points], dtype=complex)
                _switch_on_wrap_td(td_result_ell, switch_on['t_min'])
            elif tau_points is not None:
                with _phase_j_warning_scope():
                    C_tau_by_ell[ell] = np.array(
                        td_result_ell['total_C_batch'](
                            tau_points, parallel=parallel, n_workers=n_workers),
                        dtype=complex,
                    )
            else:
                C_tau_by_ell[ell] = None
        return phase_j_by_ell, total_C_by_ell, C_tau_by_ell

    pole_cluster_report = _pole_cluster_preamble(prop, max_ell, ss_route)
    engine_report = dict(ss_route, pole_deformation=(
        None if prop.get('pole_deformation') is None
        else 'bypassed' if ss_route['bypass_deformation'] else 'used'))
    engine_report['realization_check'] = _realization_check(prop, num_params)
    # the running guard: every plan of this run counts against one budget
    from engine.integration.time_domain import final_integral as _fi_mem
    import contextlib as _ctx
    _scope = (_fi_mem.phase_j_memory_scope(
        what=f"{model.get('name', '<unnamed>')} k={k} max_ell={max_ell}",
        **({} if _run_budget is None else dict(
            budget=_run_budget[0], rule=_run_budget[1], use_env=False)))
        if hasattr(_fi_mem, 'phase_j_memory_scope')       # older engine: A/B
        else _ctx.nullcontext())
    if switch_on is not None:
        from engine.integration.time_domain import statespace as _ss_so
        _so_scope = _ss_so.switch_on_scope(switch_on['t_min'])
    else:
        _so_scope = _ctx.nullcontext()
    _hint_scope = (_fi_mem.growth_hint(_growth_hint)
                   if _growth_hint and hasattr(_fi_mem, 'growth_hint')
                   else _ctx.nullcontext())
    with _scope as _mem_scope, _so_scope, _hint_scope:
        _bypassed = None
        if prop.get('pole_deformation') is not None and (
                ss_route['bypass_deformation'] or switch_on is not None):
            _bypassed = _phase_j_ss_bypass(_phase_j_pass, propagator_data,
                                           engine_report)
            if _bypassed is None and switch_on is not None:
                # the deformation averages pole-engine passes
                from engine.integration.time_domain import statespace as _ssd
                raise _ssd.SwitchOnUnsupported(
                    'pole_deformation', engine_report.get('bypass_failed'))
        if _bypassed is not None:
            phase_j_by_ell, total_C_by_ell, C_tau_by_ell = _bypassed
        elif prop.get('pole_deformation') is None:
            phase_j_by_ell, total_C_by_ell, C_tau_by_ell = _phase_j_pass(
                propagator_data)
        else:
            if pole_cluster_report is not None and pole_cluster_report.get(
                    'status') == 'bypassed':
                # the bypass did not hold (a subset needed the pole route):
                # the deformation runs after all
                pole_cluster_report = _pole_cluster_preamble(prop, max_ell)
            _discard_defer = False
            if _discard_on:
                # several passes are averaged: evaluate-and-discard is off
                # for them; the running guard still checks
                _discard_on = False
                if mem_estimate is not None:
                    mem_estimate['discard'] = False
                    mem_estimate['discard_dropped'] = 'pole_deformation'
            phase_j_by_ell, total_C_by_ell, C_tau_by_ell = _phase_j_deformed(
                _phase_j_pass, propagator_data, prop['pole_deformation'],
                diagram_records, num_params, tau_points, pole_cluster_report)
    if mem_estimate is not None and _mem_scope is not None:
        mem_estimate['built_tuples'] = _mem_scope.tuples
        mem_estimate['built_bytes'] = _mem_scope.used
        mem_estimate['built_plans'] = _mem_scope.plans
        _dis = [r.get('discard') for r in phase_j_by_ell.values()
                if isinstance(r, dict) and r.get('discard')]
        if _dis:
            # evaluate-and-discard: what was built over the run, the largest
            # diagram held at once, and what stays charged (δ records)
            if not mem_estimate.get('discard'):
                # 'auto' switched on mid-run by the measured growth
                mem_estimate['discard'] = True
                mem_estimate['discard_switched_at'] = {
                    ell: r['discard']['switched_at']
                    for ell, r in phase_j_by_ell.items()
                    if isinstance(r, dict) and r.get('discard')}
            mem_estimate['built_bytes'] = sum(d['built_bytes'] for d in _dis)
            mem_estimate['built_max_diagram_bytes'] = max(
                d['max_diagram_bytes'] for d in _dis)
            mem_estimate['kept_bytes'] = _mem_scope.used
            mem_estimate['discard_refunded'] = sum(d['refunded'] for d in _dis)
            mem_estimate['discard_kept_for_delta'] = sum(
                d['kept_for_delta'] for d in _dis)

    _phase_time('phase_j', _t_phase)

    # Master total_C: sum across ell (for caller convenience)
    @_phase_j_scoped
    def total_C(*ext_time_values):
        return sum(complex(fn(*ext_time_values))
                   for fn in total_C_by_ell.values())
    # ``dd.run`` may form total_C's values from the per-ell
    # values it evaluates anyway (the same sum, in this order)
    total_C.sum_of_by_ell = total_C_by_ell

    # k >= 3 evaluate-and-discard: the values at the fixed points (an ell
    # evaluated before a mid-run switch is evaluated there now)
    C_points = C_points_by_ell = None
    if kpoints is not None and any(
            isinstance(r, dict) and r.get('discard')
            for r in phase_j_by_ell.values()):
        with _phase_j_warning_scope():
            C_points_by_ell = {
                ell: np.array([complex(fn(*p)) for p in kpoints],
                              dtype=complex)
                for ell, fn in total_C_by_ell.items()}
        C_points = np.zeros(len(kpoints), dtype=complex)
        for i in range(len(kpoints)):        # total_C's sum, in its order
            C_points[i] = sum(complex(v[i]) for v in C_points_by_ell.values())

    # Aggregate τ-grid C(τ) = Σ_ell C_ell(τ)
    if tau_points is not None:
        C_tau = sum(C_tau_by_ell[ell] for ell in C_tau_by_ell)
    else:
        C_tau = None
    so_record = None
    if switch_on is not None:
        # switch-on: the values at ``times`` (C_tau_by_ell holds them); the
        # tree-level initial-state term joins ell = 0
        C_times_by_ell = C_tau_by_ell
        C_tau_by_ell = {ell: None for ell in C_times_by_ell}
        so_record = {
            'protocol': 'switch_on', 't_min': switch_on['t_min'],
            'engine': 'statespace',
            # Markov-embedded noise kernels (auxiliary fields made when the
            # model was built): their states start at zero as well
            'embedded_noise_states': _embedded_noise_states(model),
            'initial_covariance': switch_on['initial_covariance'],
            'initial_term': None}
        init = _initial_state_term(switch_on, propagator_data, num_params,
                                   external_fields, phys_idx)
        if init is not None:
            init_vals, init_fn = init
            so_record['initial_term'] = init_vals
            C_times_by_ell[0] = C_times_by_ell.get(0, 0) + init_vals
            _tree = total_C_by_ell.get(0, lambda *t: 0.0 + 0.0j)
            total_C_by_ell[0] = (lambda *t, _f=_tree, _g=init_fn:
                                 complex(_f(*t)) + _g(*t))
            # the ell = 0 Phase J record's totals too (its per-diagram
            # closures, ``groups`` / ``eval_per_diagram_batch``, are the
            # diagrams alone: the term is not a diagram)
            _td0 = phase_j_by_ell.get(0)
            if isinstance(_td0, dict) and callable(_td0.get('total_C')):
                _f0 = _td0['total_C']
                _td0['total_C'] = (lambda *t, _f=_f0, _g=init_fn:
                                   complex(_f(*t)) + _g(*t))
                _td0['total_C_batch'] = (
                    lambda tau_points, *a, _f=_td0['total_C'], **kw:
                    [complex(_f(*p)) for p in tau_points])
        C_times = sum(C_times_by_ell[ell] for ell in C_times_by_ell)

    # ── Adaptive mean-field dict ──────────────────────────────────
    # Discover MF saddle quantities from the model itself.  Two paths:
    #   1. ``model['naming_convention']['mf_parameters']`` (preferred,
    #      written by ModelBuilder when ``mean_field=True`` is set)
    #   2. ``parameters[i]['mean_field']`` flag (the same data)
    #   3. Classic n/v/m fallback for hand-written model files that
    #      pre-date the declarative API.
    npop = len(ft._ns.pop)

    from api.naming import saddle_names
    mf_param_names: list[str] = saddle_names(model)

    # Map each MF param name to the range of local indices it owns.
    # Heterogeneous models declare ``indexed_by=['<pop>']`` on each
    # saddle; the param's SR array is sized to ``len(pop_<pop>)``.
    # Legacy single-pop models use the flat ``ns.pop`` length.
    param_specs_by_name = {
        pspec['name']: pspec
        for pspec in (model.get('parameters', []) or [])
    }
    pop_size_map = getattr(ft._ns, '_pop_size', {}) or {}

    def _saddle_indices(pname):
        pspec = param_specs_by_name.get(pname, {})
        ib = pspec.get('indexed_by') or []
        if ib:
            # Heterogeneous: iterate over the saddle's own population
            # (single-population saddles only — physical_fields are
            # currently restricted to one population per field).
            n = pop_size_map.get(ib[0], 0)
            return list(range(n))
        # Legacy: flat pop index
        return list(ft._ns.pop)

    mf_values: dict[str, list] = {}
    for pname in mf_param_names:
        if not hasattr(ft._ns, pname):
            continue
        ns_syms = getattr(ft._ns, pname)
        # ns_syms may be a single SR var (un-indexed param) or a list
        if not isinstance(ns_syms, (list, tuple)):
            ns_syms = [ns_syms]
        vals: list[float] = []
        for i in _saddle_indices(pname):
            if i >= len(ns_syms):
                vals.append(float('nan'))
                continue
            sym = ns_syms[i]
            if sym in num_params:
                vals.append(float(num_params[sym]))
            else:
                vals.append(float('nan'))
        # Drop entries that are entirely NaN (model declared the
        # symbol but the MF solver never substituted — e.g. mstar
        # in a non-GTaS model that still ships an mstar declaration).
        if not all(v != v for v in vals):
            mf_values[pname] = vals

    result = {
        # the Phase J engine of the run (prediction, deformation; the
        # per-subset census is added by compute_cumulants)
        'phase_j_engine':  engine_report,
        'phase_j_memory':  mem_estimate,
        'total_C':         total_C,
        'total_C_by_ell':  total_C_by_ell,
        'C_tau':           C_tau,
        'C_tau_by_ell':    C_tau_by_ell,
        'tau_grid':        tau_grid if switch_on is None else None,
        # k >= 3 with evaluate-and-discard: the fixed points and the values
        **({} if C_points is None else {
            'points': np.array(kpoints, dtype=float),
            'C_points': C_points, 'C_points_by_ell': C_points_by_ell}),
        'mf_values':       mf_values,
        'mf':              MeanField(mf_values,
                                     naming_convention=naming_convention),
        'params':          Parameters(fundamental),
        'num_params':      num_params,
        'propagator':      prop,
        'diagrams':        diagram_records,
        'kernel_groups':   kernel_groups,
        'phase_j_by_ell':  phase_j_by_ell,
        'phase_walls':     phase_walls,  # dict[label → seconds], None if not verbose
        # near-degenerate / repeated poles (api._pole_clusters): None when no
        # two poles are close, else the cluster report (status, accuracy)
        'pole_clusters':   pole_cluster_report,
        'config': {
            'k':                  k,
            'max_ell':            max_ell,
            'fundamental':        fundamental,
            'parameters':         fundamental,
            'external_fields':    external_fields,        # internal names
            'external_fields_in': external_fields_user,   # as user passed
            'tau_max':            tau_max,
            'tau_step':           tau_step,
            'taylor_order':       taylor_order,
            'model_name':         model.get('name', '<unnamed>'),
            # Read after the τ-grid evaluation above: the rule these numbers
            # were computed with (the Phase J flags are read at call time).
            'phase_j_convention': _phase_j_convention(),
            'phase_j_flags':      _phase_j_flags(),
            # grouped Phase J is retired: kept so readers of old
            # result files see one key
            'use_grouped_phase_j': False,
        },
    }

    # ── DAE-MF extras (when the new equation-based solver ran) ────
    # ``mf_all_roots`` carries every distinct root with a per-root
    # stability annotation (``stable: bool``, ``eigenvalues_finite``).
    # ``fixed_point_index`` indexes over the STABLE subset only —
    # ``mf_stable_roots`` is that subset in the same sort order.
    # Unstable roots are surfaced via ``mf_unstable_roots`` for
    # inspection but are not selectable as expansion points.
    if 'mf_all_roots' in mf:
        result['mf_all_roots']        = mf['mf_all_roots']
        result['mf_stable_roots']     = mf.get('mf_stable_roots', [])
        result['mf_unstable_roots']   = mf.get('mf_unstable_roots', [])
        result['mf_index_used']       = mf['mf_index_used']
        result['mf_state_var_order']  = mf['state_var_order']
        # The selected root is guaranteed stable; its stability dict
        # (with full eigenvalue spectrum + A/B matrices) is the
        # canonical record for downstream consumers.
        try:
            # (saddle equations without .equation rows have no Dt terms
            # to linearize)
            result['mf_stability'] = (linear_stability(
                model, fundamental, mf['mf_values'], verbose=False)
                if model.get('equations') else None)
        except Exception as e:
            if verbose:
                print(f'  [stability] skipped: {type(e).__name__}: {e}')
            result['mf_stability'] = None
    if switch_on is not None:
        result['times'] = switch_on['times']
        result['C_times'] = C_times
        result['C_times_by_ell'] = C_times_by_ell
        result['t_min'] = switch_on['t_min']
        result['nonstationary'] = so_record
        result['config']['t_min'] = switch_on['t_min']
        result['config']['times'] = switch_on['times']
    state.all_unique = all_unique
    state.tau_grid_eval = (tau_grid if switch_on is None
                           else switch_on['times'][:, 0])
    state.evaluated_grid = tau_points is not None or switch_on is not None
    return result


def _standard_tail(state, result, output_npz, output_csv):
    """Save ``output_npz`` / ``output_csv`` and print the summary (the
    backends with ``standard_tail``: the temporal Phase J ones)."""
    verbose, phase_walls = state.verbose, state.phase_walls
    k, max_ell = state.k, state.max_ell
    all_unique, tau_grid = state.all_unique, state.tau_grid_eval
    # ── Optional NPZ / CSV save (api.save handles the schema) ─
    if output_npz or output_csv:
        from api.save import save_npz, save_csv
        if output_npz:
            save_npz(result, output_npz)
            if verbose:
                print(f'\nSaved NPZ: {output_npz}')
        if output_csv:
            save_csv(result, output_csv)
            if verbose:
                print(f'\nSaved CSV: {output_csv}')

    if verbose:
        print(f'\nDone.  k={k}, max_ell={max_ell}, '
              f'{len(all_unique)} unique diagrams, '
              f'{len(tau_grid)} '
              f'{"τ" if result.get("t_min") is None else "time"} points.')
        if phase_walls:
            total = sum(phase_walls.values())
            print(f'\n  Phase wall summary (Σ = {total:.1f}s):')
            for label, dt in phase_walls.items():
                pct = 100.0 * dt / total if total > 0 else 0.0
                print(f'    {label:12s}  {dt:6.2f}s  ({pct:4.1f}%)')
