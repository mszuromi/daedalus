"""
api.cumulant
============
A computed cumulant as two parts: the smooth (continuous) part, per loop
order, and the shot-noise δ components (Feature B).

``compute_cumulants`` keeps the δ spikes OUT of ``C_tau``: a spike
``w δ(a·t + c)`` sits on a hyperplane of the external times (for k = 2 the
point τ = 0 of an autocorrelation of a point process, weight = its mean rate)
and has no finite value there.  The Phase J records of the spikes are in
``result['phase_j_by_ell'][ell]['delta_contributions']``; :class:`Cumulant`
reads them, puts them in the caller's frame (the external times
``t_1 .. t_k`` in the order of ``external_fields``) and exposes them as
:class:`DeltaComponent` objects, without recomputing anything::

    c = dd.cumulant(result)          # or Cumulant.from_result(result)
    c.smooth, c.smooth_by_ell        # C_tau and C_tau_by_ell (k <= 2)
    c.deltas                         # [DeltaComponent, ...]
    c.deltas[0].weight, .location, .loop_order, .diagram, .equality
    c.total_on_grid(mode='bin')      # smooth + w/Δτ in the bin holding τ0
    c.spectrum(omega)                # FT[smooth] + Σ w e^{-iωτ0}
    c.to_frame(), c.deltas_frame()   # plain column dicts
    c.compare_to_sim(sim)            # vs a binned estimate, same convention

Conventions (k = 2): ``C(τ) = κ(x_1(t), x_2(t + τ))``, i.e. leg 1 at time 0
and leg 2 at τ, exactly as ``C_tau``.  A component's ``weight`` is the
coefficient of ``δ(τ − location)``.  ``total_on_grid(mode='bin')`` is what a
simulation estimator with bins centred on the grid points sees: the spike
adds ``weight / Δτ`` to the one bin that contains ``location`` (Δτ = that
bin's width).  ``mode='exclude'`` is ``C_tau`` (the default everywhere else:
``C(0)`` is the variance of the continuous part).

The spectrum is ``S(ω) = ∫ dτ e^{-iωτ} C(τ)`` with the same τ, so the δ
weights add the flat floor ``Σ w`` (times ``e^{-iω τ0}`` for a spike off
τ = 0).

C0 output (state-space stage S4, ``api.me_output``).  For a k = 2 result
computed on the state-space route the smooth part is also held as
matrix-exponential triples, one per loop order and side, assembled from the
same subsets, prefactors and Wick permutations as ``C_tau``::

    c.me                             # CumulantME: c.me.sides[ell]['+'].triple()
    c.me(tau)                        # C(τ) from the triples (any τ, exact τ = 0⁻)
    c.modes                          # {ell: [ModeTerm, ...]} derived on request
    c.integrate_against(kernel)      # ∫ L(τ) C(τ) dτ: resolvents + δ contact
    c.spectrum(omega)                # c (iω − G)^{-1} x0 + mirror + δ floor

One mode term is a :class:`ModeTerm` ``(coeff, rate, power, side)``:
``coeff · τ**power · exp(rate · τ)`` on ``side`` ('+' for τ > 0, '-' for
τ < 0).  The tree-level symbolic output (``ModeSum``) uses the
same field names (its sides are spelled 'positive' / 'negative').  Where the
triples are not available ``c.me_reasons`` says why (``c.me`` is ``None`` for
a loaded file or k ≠ 2, and lists per-loop-order ``reasons`` when, e.g., a
subset is on the pole route; k >= 3 slices, every chamber: ``c.me_slice(j)``
= ``api.me_output.slice_me``).  The exact delayed tree of a delay model
(``api.delay``, k = 2) has exact values, pairings and spectrum but no
triple and no finite mode list (``c.me.mode_reasons``);
``spectrum`` and
``integrate_against`` then integrate the Phase J callables numerically and
say so (:class:`~api.me_output.MEFallbackWarning`).
"""

from __future__ import annotations

import json
import math
from typing import Any, Callable, NamedTuple, Optional, Sequence

import numpy as np

__all__ = ['ModeTerm', 'DeltaComponent', 'Cumulant', 'cumulant',
           'DeltaRecordsIncompleteWarning']

#: two δ hyperplanes are the same when their normalised coefficients agree
#: to this absolute tolerance
_PLANE_TOL = 1e-12


class DeltaRecordsIncompleteWarning(UserWarning):
    """The δ components of a result may be incomplete (some shot-noise
    subsets produced no record); see :attr:`Cumulant.incomplete_reasons`."""


class ModeTerm(NamedTuple):
    """One term ``coeff · τ**power · exp(rate · τ)`` of an exponential-mode
    sum, valid on ``side`` ('+': τ > 0, '-': τ < 0, or an ordering label of
    the external times for k >= 3).  ``rate`` is complex (λ, Re λ < 0 for a
    decaying term on its side).  Filled by item C0; the layout is shared with
    the tree-level symbolic output."""
    coeff: complex
    rate: complex
    power: int
    side: str


def _rref(M, c):
    """Canonical form of the δ hyperplanes ``M t + c = 0`` (``M`` m×k, full
    row rank): ``(M', c', f)`` with ``M'`` in reduced row echelon form
    (pivots +1) and ``δ^m(M t + c) = f · δ^m(M' t + c')``,
    ``f = 1 / |det M[:, pivots]|``.  ``None`` when the rows are dependent."""
    M = np.atleast_2d(np.asarray(M, dtype=float))
    c = np.atleast_1d(np.asarray(c, dtype=float))
    m, k = M.shape
    aug = np.hstack([M, c[:, None]])
    piv, r = [], 0
    for col in range(k):
        if r == m:
            break
        p = r + int(np.argmax(np.abs(aug[r:, col])))
        if abs(aug[p, col]) < 1e-12:
            continue
        aug[[r, p]] = aug[[p, r]]
        aug[r] = aug[r] / aug[r, col]
        for q in range(m):
            if q != r:
                aug[q] = aug[q] - aug[q, col] * aug[r]
        piv.append(col)
        r += 1
    if r < m:
        return None
    aug[np.abs(aug) < 1e-15] = 0.0
    det = abs(float(np.linalg.det(M[:, piv])))
    return aug[:, :k], aug[:, k], 1.0 / det


def _plane_str(A, c, names=None):
    """``'t_1 - t_2 = 0'`` (``c`` moved to the right-hand side)."""
    names = names or [f't_{i + 1}' for i in range(len(A))]
    parts = []
    for a, nm in zip(A, names):
        if abs(a) < 1e-15:
            continue
        mag = abs(a)
        coef = '' if abs(mag - 1.0) < 1e-12 else f'{mag:.6g}*'
        sgn = '-' if a < 0 else '+'
        parts.append((sgn, f'{coef}{nm}'))
    if not parts:
        return f'{c:.6g} = 0'
    out = ('-' if parts[0][0] == '-' else '') + parts[0][1]
    for sgn, term in parts[1:]:
        out += f' {sgn} {term}'
    rhs = -c
    return f'{out} = {0.0 if abs(rhs) < 1e-15 else rhs:.6g}'


class DeltaComponent:
    """One shot-noise spike ``w(t) δ^m(M t + c)`` of a cumulant, in the
    caller's external times ``t = (t_1, .., t_k)`` (``m`` = :attr:`codim`
    equalities; ``m = 1`` is a hyperplane such as ``t_1 = t_2``, ``m = 2`` at
    k = 3 the point ``t_1 = t_2 = t_3`` of the relative times), with ``M`` in
    reduced row echelon form (pivots +1).

    Attributes
    ----------
    k : int
        Number of external legs.
    codim : int
        Number of equalities.
    equality : tuple
        ``(A, c)`` for ``codim == 1`` (the hyperplane ``A·t + c = 0``, ``A``
        a tuple of k floats); for ``codim > 1`` a tuple of such pairs.
    loop_order : int
        ℓ of the diagram that produced it.
    diagram : str
        Label ``'ell=<ℓ> #<i>'``: the i-th diagram of order ℓ (the order of
        ``[r for r in result['diagrams'] if r['ell'] == ℓ]``).
    diagram_index : int or None
        That ``i``.
    provenance : dict
        ``source`` ('phase_j_per_diagram' / 'phase_j_grouped'),
        ``integrated`` (the weight is an integral over internal times, not
        a product of propagator values), ``n_records`` (engine records
        summed: Wick permutations × pole-deformation nodes),
        ``pole_deformation_nodes``, ``delta_edges``, ``m_after_delta``
        (internal times integrated), ``theta0_rows_on_surface`` (rows taken
        as Θ(0) = 0 on the surface, see the engine's ``_SurfaceCoeff``),
        ``wick_info`` (the record carried its Wick permutations).
    """

    __slots__ = ('k', 'codim', '_M', '_c', 'loop_order', 'diagram',
                 'diagram_index', 'provenance', '_terms', '_weight_cache')

    def __init__(self, k, M, c, loop_order, diagram, diagram_index,
                 provenance, terms):
        self.k = int(k)
        self._M = np.atleast_2d(np.asarray(M, dtype=float))
        self._c = np.atleast_1d(np.asarray(c, dtype=float))
        self.codim = int(self._M.shape[0])
        self.loop_order = int(loop_order)
        self.diagram = str(diagram)
        self.diagram_index = diagram_index
        self.provenance = dict(provenance)
        # [(fn(t) -> complex, Jacobian factor)], summed
        self._terms = list(terms)
        self._weight_cache = None

    @property
    def equality(self):
        rows = tuple((tuple(float(a) for a in self._M[i]), float(self._c[i]))
                     for i in range(self.codim))
        return rows[0] if self.codim == 1 else rows

    @property
    def equality_str(self):
        return ', '.join(_plane_str(self._M[i], self._c[i])
                         for i in range(self.codim))

    # ── evaluation ──────────────────────────────────────────────────────
    def on_surface(self, times, *, tol=1e-9):
        """True when ``times`` (length k) satisfies every equality."""
        t = np.asarray(times, dtype=float)
        r = self._M @ t + self._c
        return bool(np.all(np.abs(r) <= tol * max(1.0, float(np.max(np.abs(
            t))) if t.size else 1.0)))

    def weight_at(self, times):
        """Coefficient of ``δ^m(M t + c)`` (the canonical form of
        :attr:`equality`) at ``times`` (length k, on the surface; raises
        ``ValueError`` otherwise).  For k >= 3 the weight of a hyperplane
        depends on the other times; this is how it is evaluated."""
        t = tuple(float(x) for x in times)
        if len(t) != self.k:
            raise ValueError(f'weight_at needs {self.k} times, got {len(t)}')
        if not self.on_surface(t):
            raise ValueError(f'{t} is not on the δ surface '
                             f'{self.equality_str}')
        return complex(sum(fn(t) * f for fn, f in self._terms))

    def _rel_block(self):
        """``M`` restricted to the relative times ``t_2 .. t_k`` (leg 1 at
        0): square when ``codim == k - 1``."""
        return self._M[:, 1:]

    @property
    def location(self):
        """Where the spike sits in the relative times (leg 1 at 0) when it
        is a point there (``codim == k - 1``): a float τ for k = 2
        (``δ(τ − location)``, τ = t_2 − t_1), a tuple for k >= 3; ``None``
        otherwise (a k >= 3 hyperplane: use :meth:`weight_at`)."""
        if self.codim != self.k - 1:
            return None
        B = self._rel_block()
        if abs(np.linalg.det(B)) < 1e-15:
            return None
        x = np.linalg.solve(B, -self._c) + 0.0
        return float(x[0]) if self.k == 2 else tuple(float(v) for v in x)

    @property
    def weight(self):
        """The coefficient of ``δ^{k-1}(τ − location)`` in the relative
        times (leg 1 at 0; complex, like ``C_tau``) when the spike is a point
        there (k = 2 always).  For a k >= 3 hyperplane use
        :meth:`weight_at`."""
        loc = self.location
        if loc is None:
            raise ValueError(
                f'this k={self.k} δ ({self.equality_str}) is not a point of '
                f'the relative times: its weight depends on the other '
                f'external times; use weight_at(times) on the surface')
        if self._weight_cache is None:
            t = (0.0,) + ((loc,) if self.k == 2 else tuple(loc))
            jac = abs(float(np.linalg.det(self._rel_block())))
            self._weight_cache = self.weight_at(t) / jac
        return self._weight_cache

    def __repr__(self):
        if self.location is not None:
            loc = (f'{self.location:.6g}' if self.k == 2
                   else str(tuple(round(v, 6) for v in self.location)))
            w = f'weight={self.weight:.6g}, location={loc}'
        else:
            w = self.equality_str
        return (f'DeltaComponent({w}, ell={self.loop_order}, '
                f'diagram={self.diagram!r})')


# ── building components from engine records ────────────────────────────

def _record_terms(rec, k):
    """The caller-frame pieces of one engine δ record: a list of
    ``(M, c, fn)`` with ``fn(t)`` the record's coefficient (already divided
    by the Wick compensation) at caller times ``t``, one per Wick
    permutation (``integrate_diagram`` writes the record in the FIRST
    mapping's free times; the physical spike sums the permutations, as
    ``contribution`` does for the smooth terms).  ``rec`` is a
    ``delta_contributions`` record (one equality) or a
    ``delta_contributions_multi`` one (``equalities``)."""
    rows = (rec['equalities'] if 'equalities' in rec
            else [(rec['equality_a'], rec['equality_c'])])
    rows = [([float(a) for a in ra], float(rc)) for ra, rc in rows]
    n_ext = int(rec.get('n_ext', k))
    origin = rec.get('origin_leaf_idx', 0)
    fidx = list(rec.get('free_ext_idx')
                or [j for j in range(n_ext) if j != origin])
    perms = rec.get('wick_perms') or [tuple(range(n_ext))]
    comp = rec.get('wick_compensation', 1) or 1
    coeff = rec['coeff_fc']
    retard = list(rec.get('retardation_data') or [])
    out = []
    for perm in perms:
        def free_of(t, perm=perm):
            o = t[perm[origin]] if origin is not None else 0.0
            return [t[perm[j]] - o for j in fidx]
        M = np.zeros((len(rows), n_ext))
        for r, (eq_a, _) in enumerate(rows):
            for j, a in zip(fidx, eq_a):
                M[r, perm[j]] += a
                if origin is not None:
                    M[r, perm[origin]] -= a
        c = np.array([rc for _, rc in rows])

        def fn(t, free_of=free_of):
            x = free_of(t)
            for a_list, c0 in retard:
                if c0 + sum(a * xv for a, xv in zip(a_list, x)) <= 0:
                    return 0j
            return complex(coeff(*x)) / comp
        out.append((M, c, fn))
    return out


def _components_from_records(records, k, ell, source, bad_out=None):
    """Merge the caller-frame pieces of ``records`` (one ℓ) into
    :class:`DeltaComponent` objects: one per (diagram, δ surface).  Pieces
    whose surface is degenerate (dependent rows) are left out and their
    diagram indices appended to ``bad_out``."""
    groups = []          # [(diagram_index, M', c', terms, recs, nodes)]
    bad = []
    for rec in records or []:
        di = rec.get('diagram_index')
        for M, c, fn in _record_terms(rec, k):
            canon = _rref(M, c)
            if canon is None:
                bad.append(rec.get('diagram_index'))
                continue
            Mn, cn, f = canon
            for g in groups:
                if g[0] == di and g[1].shape == Mn.shape \
                        and np.allclose(g[1], Mn, atol=_PLANE_TOL, rtol=0) \
                        and np.allclose(g[2], cn, atol=_PLANE_TOL, rtol=0):
                    break
            else:
                g = (di, Mn, cn, [], [], set())
                groups.append(g)
            g[3].append((fn, f))
            g[4].append(rec)
            if 'pole_deformation_node' in rec:
                g[5].add(complex(rec['pole_deformation_node']))
    out = []
    for di, Mn, cn, terms, recs, nodes in groups:
        r0 = recs[0]
        prov = {
            'source': source,
            'integrated': bool(r0.get('integrated', False)),
            'm_after_delta': int(r0.get('m_after_delta', 0)),
            'theta0_rows_on_surface': int(r0.get('theta0_rows_on_surface',
                                                 0)),
            'n_records': len(terms),
            'pole_deformation_nodes': len(nodes),
            'delta_edges': list(r0.get('delta_edges') or []),
            'wick_info': 'wick_perms' in r0,
        }
        label = f'ell={ell} #{di}' if di is not None else f'ell={ell}'
        out.append(DeltaComponent(k, Mn, cn, ell, label, di, prov, terms))
    if bad_out is not None:
        bad_out.extend(bad)
    return out


def _bin_deposits(tau, x):
    """``[(index, fraction / width), ...]``: how a unit spike at ``x`` is
    put on the grid ``tau`` in the bin convention.  Normally one bin (the
    one containing ``x``, centred on a grid point, edges at the midpoints);
    a spike on the edge between two bins (within 1e-9 of the bin width,
    e.g. τ = 0 on a grid without a τ = 0 point) is split half / half, so a
    symmetric spike stays symmetric; ``[]`` outside the grid."""
    i, width = _bin_index(tau, x)
    if i is None:
        return []
    tau = np.asarray(tau, dtype=float)
    below = tau[tau < tau[i]]
    above = tau[tau > tau[i]]
    tol = 1e-9 * width
    for nb in ((below.max(),) if below.size else ()) + \
            ((above.min(),) if above.size else ()):
        if abs(x - 0.5 * (tau[i] + nb)) <= tol:
            j, wj = _bin_index(tau, nb)
            return [(i, 0.5 / width), (j, 0.5 / wj)]
    return [(i, 1.0 / width)]


def _bin_index(tau, x):
    """Index of the bin (centred on the grid points, edges at the midpoints;
    the outer bins symmetric) that contains ``x``, or ``None`` outside the
    grid; and that bin's width."""
    tau = np.asarray(tau, dtype=float)
    if tau.size == 0:
        return None, None
    if tau.size == 1:
        return 0, None
    order = np.argsort(tau)
    ts = tau[order]
    mids = 0.5 * (ts[1:] + ts[:-1])
    lo = np.concatenate(([ts[0] - (mids[0] - ts[0])], mids))
    hi = np.concatenate((mids, [ts[-1] + (ts[-1] - mids[-1])]))
    hit = np.flatnonzero((lo <= x) & (x < hi))
    if hit.size == 0:
        return None, None
    i = int(hit[0])
    return int(order[i]), float(hi[i] - lo[i])



def _memo_C(fns):
    """``C(τ) = Σ f(0, τ)`` over the Phase J callables, memoized, for the
    quadrature routes.  A callable that cannot be evaluated at a large lag
    (the exact delayed tree of ``api.delay``: its method-of-steps stack is
    capped) raises :class:`api.me_output.MEUnavailable` naming it, instead
    of failing inside QUADPACK."""
    memo = {}

    def C(t):
        t = float(t)
        v = memo.get(t)
        if v is None:
            try:
                v = memo[t] = sum(complex(f(0.0, t)) for f in fns)
            except Exception as exc:
                if type(exc).__name__ != 'ExactTreeUnavailable':
                    raise
                from api.me_output import MEUnavailable
                raise MEUnavailable(
                    f'the exact delayed tree cannot be integrated numerically '
                    f'to infinite lag ({exc}); pair the loop orders alone '
                    f"(ell=...) or use delay={{'tree': 'pade'}}") from exc
        return v
    return C

class Cumulant:
    """A computed cumulant: smooth part per loop order plus δ components.
    Build with :meth:`from_result` (``compute_cumulants`` / ``dd.run``
    result) or :meth:`from_saved` (``load_npz`` output)."""

    def __init__(self, k, tau, smooth_by_ell, deltas, *, smooth_fns=None,
                 complete=True, incomplete_reasons=(), external_fields=None,
                 modes=None, source=None, result=None):
        self.k = int(k)
        self.tau = None if tau is None else np.asarray(tau, dtype=float)
        self.smooth_by_ell = {int(e): (None if v is None
                                       else np.asarray(v, dtype=complex))
                              for e, v in (smooth_by_ell or {}).items()}
        self.deltas = list(deltas)
        self.smooth_fns = dict(smooth_fns or {})
        self.complete = bool(complete)
        self.incomplete_reasons = list(incomplete_reasons)
        self.external_fields = external_fields
        # exponential-mode sum of the smooth part (item C0): given, or
        # derived from the triples on first use (``modes``)
        self._modes = modes
        self.source = source
        # the live result the triples are built from (``me``)
        self._result = result
        self._me = None
        self._me_reasons = None
        # loop orders paired / transformed by quadrature because they have
        # no triples: {(what, ell): count} (``integrate_against``,
        # ``spectrum``; each also warns, naming the reason)
        self.me_fallbacks = {}

    # ── construction ────────────────────────────────────────────────────
    @classmethod
    def from_result(cls, result, k=None):
        """From a ``compute_cumulants`` / ``dd.run`` result dict; nothing is
        recomputed (the δ weights are evaluated from the stored records on
        first use).  ``k``, when given, must match the result's."""
        if not isinstance(result, dict) or 'config' not in result:
            raise TypeError('Cumulant.from_result needs a compute_cumulants '
                            '/ dd.run result dict')
        cfg = result['config']
        k_res = int(cfg['k'])
        if k is not None and int(k) != k_res:
            raise ValueError(f'the result is a k={k_res} cumulant, not k={k}')
        if 'C_tau_x' in result or cfg.get('spatial'):
            raise NotImplementedError(
                'δ components are defined for temporal results; spatial '
                'results have no Phase J δ records')
        if result.get('t_min') is not None:
            raise NotImplementedError(
                'a switch-on result (t_min) is a function of the absolute '
                "times: use result['C_times'] / result['total_C'] (the "
                'Cumulant object describes stationary lags)')
        grouped = bool(cfg.get('use_grouped_phase_j'))
        source = 'phase_j_grouped' if grouped else 'phase_j_per_diagram'
        deltas, reasons = [], []
        if 'phase_j_by_ell' not in result:
            reasons.append('the result has no Phase J records')
        for ell, pj in sorted((result.get('phase_j_by_ell') or {}).items()):
            if not pj:
                continue
            n_rec = pj.get('n_delta_records_dropped', 0) or 0
            if n_rec:
                reasons.append(f'ell={ell}: the grouped path dropped {n_rec} '
                               f'δ record(s)')
            n_sub = pj.get('n_delta_subsets_dropped')
            if n_sub is None:
                reasons.append(f'ell={ell}: the engine did not report its '
                               f'dropped shot-noise subsets (pre-Feature-B '
                               f'result)')
            elif n_sub:
                reasons.append(f'ell={ell}: {n_sub} shot-noise subset(s) gave '
                               f'no δ record (several equalities, or a '
                               f'coefficient that could not be built)')
            sk = list(pj.get('skipped_kernel_ids') or [])
            if sk:
                reasons.append(f'ell={ell}: {len(sk)} diagram(s)/group(s) '
                               f'failed in Phase J (their smooth part and δ '
                               f'records are both missing; see '
                               f"phase_j_by_ell[{ell}]['groups'])")
            bad = []
            deltas.extend(_components_from_records(
                list(pj.get('delta_contributions') or [])
                + list(pj.get('delta_contributions_multi') or []),
                k_res, int(ell), source, bad))
            if bad:
                reasons.append(f'ell={ell}: {len(bad)} record piece(s) with a '
                               f'degenerate δ surface left out (diagrams '
                               f'{sorted(set(map(str, bad)))})')
        smooth_by_ell = dict(result.get('C_tau_by_ell') or {})
        return cls(k_res, result.get('tau_grid'), smooth_by_ell, deltas,
                   smooth_fns=result.get('total_C_by_ell') or {},
                   complete=not reasons, incomplete_reasons=reasons,
                   external_fields=cfg.get('external_fields_in',
                                           cfg.get('external_fields')),
                   source=source, result=result)

    @classmethod
    def from_saved(cls, data):
        """From :func:`api.save.load_npz` output.  The δ components come
        from the ``delta_*`` arrays (k = 2: weight and location; otherwise
        only the table, :meth:`deltas_frame`); a file written before
        Feature B has none and gives ``complete=False``."""
        k = int(np.asarray(data['k']).ravel()[0])
        by_ell = {}
        for key in data:
            if key == 'C_tree':
                by_ell[0] = data[key]
            elif key.startswith('C_') and key.endswith('_loop'):
                try:
                    by_ell[int(key[2:-5])] = data[key]
                except ValueError:
                    pass
        by_ell = {e: (v if np.asarray(v).size else None)
                  for e, v in by_ell.items()}
        deltas, reasons = [], []
        if 'delta_weight' not in data:
            reasons.append('the file has no delta_components table (written '
                           'before Feature B, or by a result without Phase J '
                           'records, e.g. spatial)')
        else:
            for i in range(np.asarray(data['delta_weight']).size):
                deltas.append(_SavedDelta.from_arrays(data, i, k))
            if 'delta_complete' in data and not bool(
                    np.asarray(data['delta_complete']).ravel()[0]):
                reasons.append('the saved result was incomplete')
        return cls(k, data.get('tau_grid'), by_ell, deltas,
                   complete=not reasons, incomplete_reasons=reasons,
                   source='saved')

    # ── smooth part ─────────────────────────────────────────────────────
    @property
    def smooth(self):
        """``Σ_ℓ`` smooth part on :attr:`tau` (``C_tau``), or None (k >= 3)."""
        vals = [v for v in self.smooth_by_ell.values() if v is not None]
        if not vals:
            return None
        return np.sum(vals, axis=0)

    def _smooth_on(self, tau, ell):
        """Smooth part on ``tau`` (k = 2), from the stored arrays when
        ``tau`` is the result's grid, else from the callables (leg 1 at 0,
        the τ = 0 point at the Itô left limit, like ``C_tau``)."""
        ells = sorted(self.smooth_by_ell) if ell is None else (
            [ell] if np.ndim(ell) == 0 else list(ell))
        unknown = [e for e in ells if e not in self.smooth_by_ell]
        if unknown:
            raise ValueError(f'loop order(s) {unknown} not in the result '
                             f'(orders: {sorted(self.smooth_by_ell)})')
        if self.tau is not None and tau.shape == self.tau.shape \
                and np.array_equal(tau, self.tau):
            out = np.zeros(tau.shape, dtype=complex)
            for e in ells:
                v = self.smooth_by_ell.get(e)
                if v is not None:
                    out = out + v
            return out
        if not self.smooth_fns:
            raise ValueError('the smooth part is only known on the stored '
                             'grid (no callables: a loaded file)')
        out = np.zeros(tau.shape, dtype=complex)
        for e in ells:
            fn = self.smooth_fns.get(e)
            if fn is not None:
                out = out + np.array([complex(fn(0.0, float(t))) for t in tau])
        return out

    # ── grids ───────────────────────────────────────────────────────────
    def _need_k2(self, what):
        if self.k != 2:
            raise NotImplementedError(f'{what} is defined for k = 2 (got '
                                      f'k = {self.k}); for k >= 3 use the '
                                      f'components\' weight_at')

    def delta_on_grid(self, tau=None, ell=None):
        """The spikes as a simulation bin sees them (k = 2): ``weight / Δτ``
        added to the bin of ``tau`` (centred on the points) that holds each
        ``location``; spikes outside the grid are left out."""
        self._need_k2('delta_on_grid')
        tau = self._grid(tau)
        out = np.zeros(tau.shape, dtype=complex)
        if tau.size < 2:
            raise ValueError('a bin needs a grid of at least two points')
        for d in self._deltas_of(ell):
            loc = d.location
            if loc is None:
                continue
            for i, f in _bin_deposits(tau, loc):
                out[i] += d.weight * f
        return out

    def _grid(self, tau):
        if tau is not None:
            return np.atleast_1d(np.asarray(tau, dtype=float))
        if self.tau is None:
            raise ValueError('no τ grid stored in the result: pass tau')
        return self.tau

    def total_on_grid(self, tau=None, mode='bin', ell=None):
        """Smooth part on ``tau`` (default: the result's grid) plus, with
        ``mode='bin'``, each spike as ``weight / Δτ`` in the bin containing
        it (the estimator convention of a binned simulation); with
        ``mode='exclude'`` the smooth part alone (``C_tau``).  ``ell``
        restricts to one loop order (or an iterable of them)."""
        self._need_k2('total_on_grid')
        if mode not in ('bin', 'exclude'):
            raise ValueError(f"mode must be 'bin' or 'exclude', not {mode!r}")
        tau = self._grid(tau)
        out = self._smooth_on(tau, ell)
        if mode == 'bin':
            self._warn_incomplete()
            out = out + self.delta_on_grid(tau, ell)
        return out

    def compare_to_sim(self, sim, ell=None):
        """Theory against a binned simulation estimate ``sim`` =
        ``{'tau', 'C', 'C_err'?}`` (k = 2): the estimate at ``sim['tau']``
        is a bin average, so it holds each spike as ``weight / Δτ`` in the
        bin containing it; the theory is put in the same convention
        (:meth:`total_on_grid` ``mode='bin'`` on the sim's points, with the
        sim's bin widths; the smooth part is taken at the bin centre, an
        O(Δτ²) approximation of the bin average).  Returns ``{'tau',
        'theory', 'sim', 'residual', 'z'}`` (``z`` = residual / C_err, None
        without errors)."""
        self._need_k2('compare_to_sim')
        t = np.asarray(sim['tau'], dtype=float)
        th = self.total_on_grid(t, mode='bin', ell=ell)
        sc = np.asarray(sim['C'])
        res = sc - th
        err = sim.get('C_err')
        z = None if err is None else np.real(res) / np.asarray(err, float)
        return {'tau': t, 'theory': th, 'sim': sc, 'residual': res, 'z': z}

    def _deltas_of(self, ell):
        if ell is None:
            return self.deltas
        ells = {ell} if np.ndim(ell) == 0 else set(ell)
        return [d for d in self.deltas if d.loop_order in ells]

    def delta_weights_by_ell(self):
        """k = 2: ``{ell: {location: Σ weight}}`` over the diagrams."""
        self._need_k2('delta_weights_by_ell')
        out = {}
        for d in self.deltas:
            loc = d.location
            if loc is None:
                continue
            slot = out.setdefault(d.loop_order, {})
            key = round(loc, 12) + 0.0
            slot[key] = slot.get(key, 0j) + d.weight
        return out

    def _warn_incomplete(self):
        if not self.complete:
            import warnings
            warnings.warn('δ components may be incomplete: '
                          + '; '.join(self.incomplete_reasons),
                          DeltaRecordsIncompleteWarning, stacklevel=3)

    # ── C0: matrix-exponential triples and modes ────────────────────────
    @property
    def me(self):
        """The smooth part as matrix-exponential triples
        (:class:`api.me_output.CumulantME`), built from the live result on
        first use; ``None`` when it cannot be built (see :attr:`me_reasons`).
        Per loop order gaps (a subset on the pole route, ...) are in
        ``me.reasons``."""
        if self._me is None and self._me_reasons is None:
            from api.me_output import from_result, MEUnavailable
            if self._result is None:
                self._me_reasons = ['no live result (a loaded file)']
            else:
                try:
                    self._me = from_result(self._result)
                    self._me_reasons = []
                except MEUnavailable as exc:
                    self._me_reasons = [str(exc)]
        return self._me

    def me_slice(self, j=1, base=None):
        """k >= 3: the matrix-exponential triples of ``dd.run``'s slice
        ``j`` (leg ``j`` swept, the others at ``base``), every chamber
        included (:func:`api.me_output.slice_me`)."""
        from api.me_output import slice_me, MEUnavailable
        if self.k < 3:
            raise MEUnavailable('me_slice is for k >= 3 (k = 2: .me)')
        if self._result is None:
            raise MEUnavailable('no live result (a loaded file)')
        return slice_me(self._result, j, base)

    @property
    def me_reasons(self):
        """Why :attr:`me` is ``None`` or incomplete: a list of strings (empty
        when every loop order has its triples)."""
        me = self.me
        if me is None:
            return list(self._me_reasons or [])
        return [f'ell={e}: {r}' for e, rs in sorted(me.reasons.items())
                for r in rs]

    @property
    def modes(self):
        """``{ell: [ModeTerm, ...]}``: the exponential-mode sum of the smooth
        part, derived from the triples on first use (eigenvalue clusters of
        the final generators; ``τ**n`` terms for repeated eigenvalues;
        :class:`api.me_output.MEConditioningWarning` when ill-conditioned).
        Loop orders without triples are left out, with an
        :class:`api.me_output.MEFallbackWarning` naming them
        (:attr:`me_reasons`); ``None`` when no loop order has triples."""
        if self._modes is None:
            me = self.me
            if me is None:
                return None
            mr = getattr(me, 'mode_reasons', {}) or {}
            ells = [e for e in me.orders
                    if not me.reasons.get(e) and not mr.get(e)]
            left = [e for e in me.orders if e not in ells]
            if left:
                import warnings
                from api.me_output import MEFallbackWarning
                why = self.me_reasons + [f'ell={e}: {r}' for e in left
                                         for r in mr.get(e, ())]
                warnings.warn(f'modes: loop order(s) {left} have no '
                              f'finite mode list and are left out ('
                              + '; '.join(why) + ')',
                              MEFallbackWarning, stacklevel=2)
            if not ells:
                return None
            self._modes = me.modes(ells)
        return self._modes

    @modes.setter
    def modes(self, value):
        self._modes = value

    def _delta_pairing(self, kernel, ell, contact, values):
        """``Σ w L(location)`` over the δ components (k = 2) with the
        contact rule of ``symbolic_out.integrate_against`` at τ = 0."""
        from api.me_output import kernel_value, kernel_at_zero
        tot = 0j
        deltas = self._deltas_of(ell)
        self._warn_incomplete()
        for d in deltas:
            loc = d.location
            if loc is None:
                continue
            w = d.weight
            if loc != 0.0:
                tot += w * kernel_value(kernel, loc, values)
                continue
            lp, lm, kscale = kernel_at_zero(kernel, values, with_scale=True)
            if contact == 'exclude':
                continue
            if contact is None:
                # a jump relative to the kernel's own size at 0 (the former
                # floor of 1.0 took a jump of a small kernel for round-off
                # and silently used L(0+): audit GEN-14)
                if abs(lp - lm) > 1e-14 * kscale:
                    raise ValueError(
                        f'the cumulant has a δ(τ) component (weight {w:.6g}) '
                        f'and the kernel jumps at 0 (L(0+) = {lp:.6g}, '
                        f"L(0-) = {lm:.6g}): choose contact='exclude', "
                        f"'positive', 'negative' or 'mid'")
                tot += w * lp
            elif contact == 'positive':
                tot += w * lp
            elif contact == 'negative':
                tot += w * lm
            elif contact == 'mid':
                tot += w * 0.5 * (lp + lm)
            else:
                raise ValueError("contact must be None, 'exclude', "
                                 "'positive', 'negative' or 'mid'")
        return tot

    def integrate_against(self, kernel, ell=None, contact=None, values=None,
                          method='auto', epsabs=1e-12, epsrel=1e-10,
                          limit=400):
        """``∫ L(τ) C(τ) dτ`` (k = 2): the smooth part plus the δ components.

        ``kernel``: a ``symbolic_out.Kernel`` (``Kernel.exponential``,
        ``Kernel.two_sided_exponential``, sums; polynomial × exponential
        terms), an :class:`api.me_output.RationalKernel`, or a list of them;
        ``values`` substitutes the kernel's symbols.  ``method='exact'``: one
        resolvent solve per triple block, ``q! c (zI − G)^{−q−1} x0`` (a
        Sylvester solve for a rational kernel): exact, no poles;
        ``'quad'``: adaptive quadrature of the Phase J callables;
        ``'auto'`` (default): exact when the triples cover ``ell``, else quad
        with an :class:`api.me_output.MEFallbackWarning`.  ``contact``: the
        δ(τ) term at τ = 0 as decided for the tree-level output (``None`` includes ``w L(0)``
        when the kernel is continuous at 0 and refuses otherwise;
        ``'exclude'`` / ``'positive'`` / ``'negative'`` / ``'mid'``); spikes
        off τ = 0 add ``w L(location)``.  Returns an
        :class:`api.me_output.Pairing` (``value``, ``smooth``, ``contact``,
        ``method``; ``complex(p)`` is the value)."""
        from api.me_output import Pairing, quad_pairing
        self._need_k2('integrate_against')
        if contact not in (None, 'exclude', 'positive', 'negative', 'mid'):
            raise ValueError("contact must be None, 'exclude', 'positive', "
                             "'negative' or 'mid'")
        method, ex, qu = self._split_method(method, ell, 'integrate_against')
        sm, err = 0j, None
        if ex:
            sm += self.me.integrate_smooth(kernel, ex, values)
        if qu is not False:
            s2, err = quad_pairing(self._quad_C(qu), kernel, values,
                                   epsabs, epsrel, limit)
            sm += s2
        ct = self._delta_pairing(kernel, ell, contact, values)
        return Pairing(sm + ct, sm, ct, method, err)

    def _split_method(self, method, ell, what, infinite=True):
        """``(label, exact_ells, quad_ells)``: the loop orders paired
        exactly (from the triples) and by quadrature (``None`` = every
        order the callables have, ``False`` = none).  ``'auto'`` splits per
        loop order: the orders without triples fall back on quadrature with
        an :class:`api.me_output.MEFallbackWarning` naming them and their
        reasons (counted in :attr:`me_fallbacks`); the label is then
        ``'exact+quad'`` (``'quad'`` when no order has triples)."""
        if method not in ('auto', 'exact', 'quad'):
            raise ValueError(f"method must be 'auto', 'exact' or 'quad', "
                             f"not {method!r}")
        if method == 'quad':
            qu = None if ell is None else (
                [ell] if np.ndim(ell) == 0 else list(ell))
            self._refuse_exact_delay_quad(qu, what, infinite)
            return 'quad', [], qu
        me = self.me
        if me is None:
            ex, why = [], '; '.join(self._me_reasons or ['no triples'])
            qu = None if ell is None else (
                [ell] if np.ndim(ell) == 0 else list(ell))
            left = qu
        else:
            ells = me.orders if ell is None else (
                [ell] if np.ndim(ell) == 0 else list(ell))
            missing = [e for e in ells if e not in me.sides]
            if missing:
                raise ValueError(f'loop order(s) {missing} not in the result '
                                 f'(orders: {me.orders})')
            ex = [e for e in ells if not me.reasons.get(e)]
            left = [e for e in ells if me.reasons.get(e)]
            if not left:
                return 'exact', ex, False
            qu = left
            why = '; '.join(f'ell={e}: ' + '; '.join(me.reasons[e])
                            for e in left)
        if method == 'exact':
            from api.me_output import MEUnavailable
            raise MEUnavailable(f'{what}: no matrix-exponential triples '
                                f'({why})')
        self._refuse_exact_delay_quad(qu, what, infinite, why)
        import warnings
        from api.me_output import MEFallbackWarning
        which = ('every loop order' if left is None
                 else f'loop order(s) {left}')
        warnings.warn(f'{what}: no matrix-exponential triples for {which} '
                      f'({why}); integrating the Phase J callables of '
                      f'{"those orders" if ex else "the cumulant"} '
                      f'numerically', MEFallbackWarning, stacklevel=3)
        for e in (left if left is not None else ['all']):
            key = (what, e)
            self.me_fallbacks[key] = self.me_fallbacks.get(key, 0) + 1
        return ('exact+quad' if ex else 'quad'), ex, qu

    def _refuse_exact_delay_quad(self, qu, what, infinite, why=None):
        """The exact delayed tree (``api.delay``) has no triple and its
        callable is a method-of-steps stack capped at a few hundred delay
        intervals: quadrature to infinite lag cannot finish (QUADPACK would
        build ever larger stacks first).  Refused up front, by name."""
        from api.me_output import _exact_delay_tree, MEUnavailable
        if not infinite or self._result is None or \
                not _exact_delay_tree(self._result):
            return
        if qu is not None and 0 not in qu:
            return
        raise MEUnavailable(
            f'{what}: the k = 2 tree is the exact delayed covariance '
            f'(method of steps over the delay), which has no triple'
            + (f' ({why})' if why else '')
            + ' and cannot be integrated numerically to infinite lag; '
            f'pair the loop orders alone (ell=...), use '
            f"delay={{'tree': 'pade'}}"
            + (", or a finite window (tau_max=...)"
               if what == 'spectrum' else ''))

    def _quad_C(self, ell):
        """``C(τ)`` from the Phase J callables (memoized), for quadrature."""
        ells = (sorted(self.smooth_fns) if ell is None else
                ([ell] if np.ndim(ell) == 0 else list(ell)))
        fns = [self.smooth_fns[e] for e in ells if e in self.smooth_fns]
        if not fns:
            raise ValueError('no smooth callables (a loaded file)')
        return _memo_C(fns)

    # ── spectrum ────────────────────────────────────────────────────────
    def spectrum(self, omega, ell=None, method='auto', tau_max=None,
                 epsabs=1e-12, epsrel=1e-10, limit=400):
        """Power spectrum ``S(ω) = ∫ dτ e^{-iωτ} C(τ)`` (k = 2): the Fourier
        transform of the smooth part plus ``Σ w e^{-iω·location}`` (the flat
        δ floor for spikes at τ = 0).  For a cross-correlation, C(τ) =
        κ(x_1(0), x_2(τ)), this is ``⟨x̂_1(ω)* x̂_2(ω)⟩`` per unit time with
        ``x̂(ω) = ∫ e^{-iωt} x(t) dt``.

        ``method='exact'`` (C0): ``c (iω − G_+)^{-1} x0 + c (−iω −
        G_−)^{-1} x0`` from the matrix-exponential triples of the two sides
        (one triangular resolvent solve per block and ω; no quadrature, no
        poles).  ``'quad'``: the Fourier-weight quadrature of
        ``scipy.integrate.quad`` (QAWF) on the Phase J callables on (-∞, 0)
        and (0, ∞) (``epsabs`` raised to ``epsrel · |C(0)|``; hundreds of
        callable evaluations per ω); ``tau_max`` truncates it to a finite
        window.  ``'grid'``: the trapezoid rule on the stored grid (display
        only).  ``'auto'`` (default): exact when the triples cover ``ell``,
        else quad with an :class:`api.me_output.MEFallbackWarning`.
        """
        self._need_k2('spectrum')
        omega = np.atleast_1d(np.asarray(omega, dtype=float))
        if method == 'grid':
            if self.tau is None:
                raise ValueError('no stored grid')
            order = np.argsort(self.tau)
            t = self.tau[order]
            c = self._smooth_on(self.tau, ell)[order]
            ph = np.exp(-1j * np.outer(omega, t))
            S = np.trapezoid(ph * c[None, :], t, axis=1) if hasattr(
                np, 'trapezoid') else np.trapz(ph * c[None, :], t, axis=1)
        else:
            if method not in ('auto', 'exact', 'quad'):
                raise ValueError(f"method must be 'auto', 'exact', 'quad' or "
                                 f"'grid', not {method!r}")
            if tau_max is not None and method == 'auto':
                method = 'quad'
            elif tau_max is not None and method == 'exact':
                raise ValueError("tau_max truncates the quadrature: use "
                                 "method='quad'")
            method, ex, qu = self._split_method(method, ell, 'spectrum',
                                                infinite=tau_max is None)
            S = np.zeros(omega.shape, dtype=complex)
            if ex:
                S = S + self.me.spectrum_smooth(omega, ex)
            if qu is not False:
                S = S + np.array([self._ft_quad(w, qu, tau_max, epsabs,
                                                epsrel, limit)
                                  for w in omega])
        self._warn_incomplete()
        for d in self._deltas_of(ell):
            loc = d.location
            if loc is not None:
                S = S + d.weight * np.exp(-1j * omega * loc)
        return S

    def _ft_quad(self, w, ell, tau_max, epsabs, epsrel, limit):
        from scipy.integrate import quad
        ells = (sorted(self.smooth_fns) if ell is None else
                ([ell] if np.ndim(ell) == 0 else list(ell)))
        fns = [self.smooth_fns[e] for e in ells if e in self.smooth_fns]
        if not fns:
            raise ValueError('no smooth callables (a loaded file): use '
                             "method='grid'")
        C = _memo_C(fns)
        # the absolute tolerance relative to the size of C
        scale = max(abs(C(-1e-9)), abs(C(1e-9)), 1e-300)
        eabs = max(epsabs, epsrel * scale)
        b = np.inf if tau_max is None else float(tau_max)

        def g_even(s):
            return C(s) + C(-s)

        def g_odd(s):
            return C(s) - C(-s)

        def part(g, wt):
            if wt is None:
                kw = dict(epsabs=eabs, epsrel=epsrel, limit=limit)
                return (quad(lambda s: g(s).real, 0.0, b, **kw)[0]
                        + 1j * quad(lambda s: g(s).imag, 0.0, b, **kw)[0])
            if np.isinf(b):
                # QAWF: epsrel is not used on an infinite range
                kw = dict(weight=wt, wvar=abs(w), epsabs=eabs, limlst=200,
                          limit=limit)
            else:
                kw = dict(weight=wt, wvar=abs(w), epsabs=eabs, epsrel=epsrel,
                          limit=limit)
            return (quad(lambda s: g(s).real, 0.0, b, **kw)[0]
                    + 1j * quad(lambda s: g(s).imag, 0.0, b, **kw)[0])

        if w == 0.0:
            return part(g_even, None)
        sw = 1.0 if w > 0 else -1.0
        # ∫_0^∞ [C(s) + C(-s)] cos(ωs) ds - i sgn(ω) ∫_0^∞ [C(s) - C(-s)]
        # sin(|ω|s) ds
        return part(g_even, 'cos') - 1j * sw * part(g_odd, 'sin')

    # ── export ──────────────────────────────────────────────────────────
    def to_frame(self, tau=None):
        """Columns as a plain dict of arrays (no pandas dependency):
        ``tau``, ``C_tree``, ``C_<ℓ>_loop``, ``C_smooth`` and, for k = 2,
        ``C_delta_bin`` (the spikes as a bin sees them) and ``C_total_bin``.
        ``pandas.DataFrame(c.to_frame())`` gives a frame."""
        if self.k == 2:
            tau = self._grid(tau)
        else:
            tau = self.tau if tau is None else np.asarray(tau, dtype=float)
        out = {'tau': tau}
        if self.k == 2:
            for e in sorted(self.smooth_by_ell):
                out['C_tree' if e == 0 else f'C_{e}_loop'] = \
                    self._smooth_on(tau, e)
            sm = self._smooth_on(tau, None)
            out['C_smooth'] = sm
            if tau.size >= 2:              # a bin needs two points
                dl = self.delta_on_grid(tau)
                out['C_delta_bin'] = dl
                out['C_total_bin'] = sm + dl
        else:
            for e, v in sorted(self.smooth_by_ell.items()):
                if v is not None:
                    out['C_tree' if e == 0 else f'C_{e}_loop'] = v
        return out

    def deltas_frame(self):
        """The δ table as a dict of columns: ``weight`` (k = 2, else NaN),
        ``location`` (k = 2, else NaN), ``ell``, ``diagram``, ``equality``,
        ``provenance`` (JSON)."""
        rows = [_delta_row(d, self.k) for d in self.deltas]
        cols = ('weight', 'location', 'ell', 'diagram', 'equality',
                'provenance')
        return {c: [r[c] for r in rows] for c in cols}

    def __repr__(self):
        return (f'Cumulant(k={self.k}, orders={sorted(self.smooth_by_ell)}, '
                f'deltas={len(self.deltas)}, complete={self.complete})')


def _delta_row(d, k):
    """One row of the δ table: ``weight`` when the spike is a point of the
    relative times (k = 2 always), else NaN; ``location`` the k = 2 τ, else
    NaN (the surface is in ``equality``)."""
    w = loc = complex('nan')
    try:
        w = d.weight                 # a point of the relative times
    except ValueError:               # a k >= 3 hyperplane: no number
        pass
    if k == 2 and d.location is not None:
        loc = d.location
    return {'weight': complex(w), 'location': float(np.real(loc)),
            'ell': int(d.loop_order), 'diagram': str(d.diagram),
            'equality': d.equality_str,
            'provenance': json.dumps(d.provenance, sort_keys=True,
                                     default=str)}


class _SavedDelta:
    """A δ component read back from a file: k = 2 weight / location and the
    table entries (the weight is a number, not a function of the times)."""
    __slots__ = ('k', 'weight', 'location', 'loop_order', 'diagram',
                 'equality_str', 'provenance', 'diagram_index', 'equality')

    @classmethod
    def from_arrays(cls, data, i, k):
        self = cls()
        self.k = k
        w = complex(np.asarray(data['delta_weight']).ravel()[i])
        loc = float(np.asarray(data['delta_location']).ravel()[i])
        self.weight = w
        self.location = None if math.isnan(loc) else loc
        self.loop_order = int(np.asarray(data['delta_ell']).ravel()[i])
        self.diagram = str(np.asarray(data['delta_diagram']).ravel()[i])
        self.equality_str = str(np.asarray(data['delta_equality']).ravel()[i])
        self.equality = None
        self.diagram_index = None
        try:
            self.provenance = json.loads(
                str(np.asarray(data['delta_provenance']).ravel()[i]))
        except ValueError:
            self.provenance = {}
        return self

    codim = None

    def weight_at(self, times):
        if not math.isnan(self.weight.real):
            return self.weight            # constant (a point spike)
        raise ValueError('a loaded k >= 3 hyperplane δ component has no '
                         'weight function (only the table)')

    def __repr__(self):
        return (f'DeltaComponent(saved, weight={self.weight:.6g}, '
                f'location={self.location}, ell={self.loop_order}, '
                f'diagram={self.diagram!r})')


def delta_table(result) -> dict:
    """The ``delta_components`` table of a result (columns of
    :meth:`Cumulant.deltas_frame` plus ``complete``) for the savers;
    ``None`` for results that have no Phase J records (spatial)."""
    try:
        c = Cumulant.from_result(result)
    except NotImplementedError:                   # spatial: no records
        return None
    tab = c.deltas_frame()
    tab['complete'] = c.complete
    return tab


def cumulant(result, k=None) -> Cumulant:
    """``dd.cumulant(result)``: the :class:`Cumulant` of a
    ``compute_cumulants`` / ``dd.run`` result (nothing is recomputed)."""
    return Cumulant.from_result(result, k=k)
