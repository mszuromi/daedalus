r"""
The exact stationary mean relation (Schwinger-Dyson), symbolic.

For an action whose drift sector is

    S  ⊃  xt_r [ K_{rp}(d/dt) dx_p  +  sum_{m >= 2} V^{(m)}_{r; p_1..p_m} dx_{p_1} ... dx_{p_m} ]

(the (1, 1) sector of the expanded ``FieldTheory`` and its (1, m) sectors,
``dx = x - x*`` the fluctuation about the saddle), the equation of motion
``< dS / d xt_r(t) > = 0`` reads, in the stationary state,

    K(omega = 0) <dx>  +  sum_m V^{(m)} : <dx^m>  =  0,

    <dx>  =  -G(omega = 0) . sum_m V^{(m)} : <dx^m>,      G = K^{-1}.

Every term with a response field left over has zero expectation (causality;
Itô: equal-time responses vanish), so the noise sectors never enter.  The
raw moments ``<dx_{p_1} ... dx_{p_m}>`` are written in CONNECTED cumulants by
the set-partition formula, so the relation is exact, at all loop orders, in
terms of the mean shift ``kappa(dx_p) = <x_p> - x*_p`` and the connected
cumulants ``kappa(dx_p, dx_q)``, ``kappa(dx_p, dx_q, dx_s)`` ... of the
stationary state.  It is a relation, not a solution: the cumulants of order
two and higher are unknowns of their own.

Loop expansion.  With ``kappa_n = sum_l kappa_n^{(l)}`` and
``kappa_n^{(l)} = O(hbar^{n - 1 + l})`` (``l >= 1`` for the mean, ``l >= 0``
otherwise; ``l = 0``, ``n = 2`` is the tree covariance) the relation gives
the mean at each loop order from lower orders only:

    kappa_1^{(1)} = -G(0) . V^{(2)} : kappa_2^{(0)}
    kappa_1^{(2)} = -G(0) . [ V^{(2)} : (kappa_2^{(1)} + kappa_1^{(1)} kappa_1^{(1)})
                             + V^{(3)} : (kappa_3^{(0)} + 3 kappa_2^{(0)} kappa_1^{(1)}) + V^{(4)} : ... ]

Loop order ``L`` needs the drift to degree ``2 L``.

One loop, diagram by diagram (:func:`one_loop_mean`).  The first line is a
sum of tadpoles, ``-G(0)_{a r} V^{(2)}_{r; pq} C0_pq(0)``, with the
equal-time tree covariance ``C0_pq(0) = kappa_{pq}^{(0)}`` named, as its
Fourier integral ``(1/2 pi) int C0_pq(omega) d omega``, exact (rational in
the parameters, by the state-space route) or as a residue sum over the poles
of ``C0_pq`` (``RootOf`` beyond degree two).

Polynomial drifts are exact.  A non-polynomial drift (``exp(v)``, a sigmoid)
is Taylor-expanded to ``order=`` and the relation holds up to the dropped
terms of degree ``order + 1`` and higher; :attr:`MeanRelation.truncation`
says which case applies.  The degree of the drift is read from the
mean-field sector of the action (its saddle dependence is the drift itself,
untruncated).

Refused (``UnsupportedModelError``): spatial models; vertices with a time
derivative or a kernel (``Conv``) in a drift coefficient, whose stationary
averages are not moments at one time; a singular ``K(omega = 0)``.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

import numpy as np
from sage.all import SR, matrix, SetPartitions, PolynomialRing

from daedalus.api.symbolic_out import (
    SymbolicOutError, UnsupportedModelError, _reject_spatial, _expanded_theory,
    _model_fundamental, _solve_mean_field, _mf_symbols_only, _as_name_dict,
    _expand_param_names, _is_saddle_symbol, to_sympy, to_latex,
)

__all__ = ['DriftTerm', 'MeanRelation', 'mean_relation', 'Tadpole', 'OneLoopMean',
           'one_loop_mean']


@dataclass(frozen=True)
class DriftTerm:
    """One monomial ``coefficient * xt_response * prod(legs)`` of a (1, m)
    drift sector, ``m = len(legs)``; ``legs`` are physical fluctuation
    fields with multiplicity, in field order."""
    response: str
    coefficient: Any
    legs: tuple

    @property
    def degree(self) -> int:
        return len(self.legs)


def _cumulant_name(legs, ell=None) -> str:
    base = 'kappa_' + '_'.join(legs)
    return base if ell is None else f'{base}__l{int(ell)}'


@dataclass
class MeanRelation:
    """The stationary Schwinger-Dyson relation of a model (module docstring).

    ``K0`` is ``K(omega = 0)`` (rows response fields, columns physical
    fields), ``G0`` its inverse; ``terms`` the drift monomials of degree 2 to
    ``order``; ``truncation`` states whether the relation is exact."""
    model_name: str
    phys_names: list
    resp_names: list
    K0: Any
    terms: list
    order: int
    drift_degree: Optional[int]
    values: dict = field(default_factory=dict)
    notes: list = field(default_factory=list)
    _latex: dict = field(default_factory=dict, repr=False)
    _cache: dict = field(default_factory=dict, repr=False)

    # ── status ──────────────────────────────────────────────────────────
    @property
    def exact(self) -> bool:
        """True when the drift is a polynomial of degree <= ``order``."""
        return self.drift_degree is not None and self.drift_degree <= self.order

    @property
    def truncation(self) -> str:
        if self.exact:
            return (f'exact: the drift is a polynomial of degree '
                    f'{self.drift_degree} in the fields')
        return (f'Taylor-truncated at drift degree {self.order}: the drift is '
                f'not a polynomial of degree <= {self.order} in the fields, '
                f'so the relation holds up to the dropped terms of degree '
                f'{self.order + 1} and higher')

    @property
    def G0(self):
        """``G(omega = 0) = K(omega = 0)^{-1}`` (physical x response)."""
        if 'G0' not in self._cache:
            self._cache['G0'] = self.K0.inverse().apply_map(
                lambda e: SR(e).simplify_rational())
        return self._cache['G0']

    # ── symbols ─────────────────────────────────────────────────────────
    def _canon(self, fields_) -> tuple:
        out = []
        for f in fields_:
            f = str(f)
            if f not in self.phys_names:
                raise SymbolicOutError(
                    f'unknown physical field {f!r}; known: {self.phys_names}')
            out.append(f)
        return tuple(sorted(out, key=self.phys_names.index))

    def kappa(self, *fields_, ell=None):
        """The connected-cumulant symbol ``kappa(fields)`` (``ell=None``: the
        full stationary value; ``ell=l``: its ``l``-loop part).  One field is
        the mean shift ``<x> - x*``."""
        legs = self._canon(fields_)
        name = _cumulant_name(legs, ell)
        if name not in self._latex:
            inner = ', '.join(self._field_latex(f) for f in legs)
            sup = '' if ell is None else f'^{{({int(ell)})}}'
            self._latex[name] = (rf'\langle {inner} \rangle{sup}' if len(legs) == 1
                                 else rf'\kappa{sup}({inner})')
        return SR.var(name, latex_name=self._latex[name])

    def _field_latex(self, f) -> str:
        return self._cache.get('field_latex', {}).get(f, f)

    # ── the relation ────────────────────────────────────────────────────
    def moment(self, legs, ell=None):
        """``<dx_{p_1} ... dx_{p_m}>`` in connected cumulants: exact
        (``ell=None``), or its part of loop order ``ell`` written with the
        loop-graded symbols."""
        legs = self._canon(legs)
        m = len(legs)
        if ell is None:
            total = SR(0)
            for part in SetPartitions(range(m)):
                term = SR(1)
                for block in part:
                    term *= self.kappa(*[legs[i] for i in block])
                total += term
            return total
        L = int(ell)
        P = PolynomialRing(SR, 'hbar__')
        h = P.gen()
        total = SR(0)
        for part in SetPartitions(range(m)):
            prod = P(1)
            for block in part:
                b = [legs[i] for i in block]
                n = len(b)
                series = P(0)
                for l in range(1 if n == 1 else 0, L - (n - 1) + 1):
                    series += self.kappa(*b, ell=l) * h ** (n - 1 + l)
                prod = (prod * series).truncate(L + 1)
                if prod == 0:
                    break
            total += prod[L]
        return total

    def drift_average(self, ell=None) -> list:
        """``sum_m V^{(m)} : <dx^m>`` per response row (exact, or its
        loop-order-``ell`` part)."""
        rows = [SR(0)] * len(self.resp_names)
        for t in self.terms:
            r = self.resp_names.index(t.response)
            rows[r] += t.coefficient * self.moment(t.legs, ell)
        return rows

    def equations(self) -> list:
        """One Sage relation per response field:
        ``(K(0) . kappa_1)_r == -(sum_m V^{(m)} : <dx^m>)_r``."""
        k1 = [self.kappa(p) for p in self.phys_names]
        rhs = self.drift_average()
        out = []
        for r in range(len(self.resp_names)):
            lhs = sum((self.K0[r, p] * k1[p] for p in range(len(k1))), SR(0))
            out.append(lhs == -rhs[r])
        return out

    def mean_shift(self) -> dict:
        """``kappa(dx_p) = -(G(0) . sum_m V^{(m)} : <dx^m>)_p`` (implicit: the
        right side contains the mean shift itself through the moments)."""
        return self._solve_rows(self.drift_average())

    def loop_order(self, ell: int) -> dict:
        """The ``ell``-loop mean shift, ``{field: expression}``, in the
        loop-graded cumulants of lower orders (``kappa(..., ell=l)``)."""
        ell = int(ell)
        if ell < 1:
            raise SymbolicOutError('the mean shift starts at one loop (ell >= 1)')
        if not self.exact and self.order < 2 * ell:
            raise SymbolicOutError(
                f'loop order {ell} needs the drift to degree {2 * ell}, and '
                f'this relation is truncated at degree {self.order} '
                f'({self.truncation}); build it with order={2 * ell} or more')
        return self._solve_rows(self.drift_average(ell))

    def _solve_rows(self, rows) -> dict:
        G0 = self.G0
        out = {}
        for p, name in enumerate(self.phys_names):
            e = -sum((G0[p, r] * rows[r] for r in range(len(rows))), SR(0))
            out[name] = SR(e).simplify_rational() if e != 0 else SR(0)
        return out

    # ── numbers and exports ─────────────────────────────────────────────
    def cumulant_symbols(self, expr) -> list:
        """Names of the cumulant symbols in an expression."""
        return sorted(str(v) for v in SR(expr).variables()
                      if str(v).startswith('kappa_'))

    def evaluate(self, expr, values: Optional[dict] = None,
                 cumulants: Optional[dict] = None):
        """Number of ``expr`` (one expression, a dict or a list of them) with
        the parameter ``values`` (pipeline names, ``{'tau': [..]}`` expanded)
        and the ``cumulants`` (``{symbol or name: number}``); every symbol
        must get a value."""
        sub = {}
        for k, v in _expand_param_names(_as_name_dict(values)).items():
            sub[SR.var(k)] = SR(v)
        for k, v in (cumulants or {}).items():
            sub[SR.var(str(k))] = SR(v)

        def one(e):
            e = SR(e).subs(sub)
            free = sorted(str(v) for v in e.variables())
            if free:
                raise SymbolicOutError(f'no value for {free}')
            return complex(e.n())
        if isinstance(expr, dict):
            return {k: one(v) for k, v in expr.items()}
        if isinstance(expr, (list, tuple)):
            return [one(e) for e in expr]
        return one(expr)

    def sympy(self, ell=None):
        """sympy image: ``ell=None`` the equations (``sympy.Eq``), else the
        ``ell``-loop mean shift ``{field: expr}``."""
        import sympy
        if ell is None:
            return [sympy.Eq(to_sympy(e.lhs()), to_sympy(e.rhs()))
                    for e in self.equations()]
        return {k: to_sympy(v) for k, v in self.loop_order(ell).items()}

    def latex(self, ell=None) -> str:
        """LaTeX: the equations (``ell=None``, one ``a = b`` line per response
        field joined by ``\\\\``) or the ``ell``-loop mean shift."""
        if ell is None:
            return r' \\ '.join(f'{to_latex(e.lhs())} = {to_latex(e.rhs())}'
                                for e in self.equations())
        return r' \\ '.join(
            f'{to_latex(self.kappa(k, ell=ell))} = {to_latex(v)}'
            for k, v in self.loop_order(ell).items())

    def __str__(self):
        lines = [f'MeanRelation({self.model_name!r}; {self.truncation})']
        lines += [f'  {e}' for e in self.equations()]
        return '\n'.join(lines)


# ═════════════════════════════════════════════════════════════════════════════
# Construction
# ═════════════════════════════════════════════════════════════════════════════

def _drift_degree(ft, model) -> Optional[int]:
    """Total degree of the drift in the fields, read from the response-linear
    mean-field sector (the drift at the saddle, before any Taylor
    truncation); ``None`` if it is not a polynomial in the saddle symbols, or
    no saddle symbol occurs there."""
    raw = (getattr(ft, '_mf_sector_raw', None) or {}).get((1, 0))
    if raw is None:
        return None
    t = SR.var('deg_probe__')
    deg, seen = 0, False
    for c in raw.dict().values():
        e = SR(c)
        sad = [v for v in e.variables() if _is_saddle_symbol(str(v), model)]
        if not sad:
            continue
        seen = True
        e = e.subs({v: t * v for v in sad}).expand()
        if not e.is_polynomial(t):
            return None
        deg = max(deg, int(e.degree(t)))
    return deg if seen else None


def _forbidden_symbols(model) -> set:
    return {str(o.get('sage_name') or o.get('name'))
            for o in (model.get('operators') or [])} | {'Dt'}


def mean_relation(model: dict, *, order: Optional[int] = None,
                  values: Optional[dict] = None, mean_field: Any = None,
                  parameters: Optional[dict] = None, use_cache: bool = True,
                  verbose: bool = False) -> MeanRelation:
    """The exact stationary mean relation of ``model`` (module docstring).

    Parameters
    ----------
    order : int, optional
        Highest drift degree kept.  Default: the drift's own degree when it
        is a polynomial; a non-polynomial drift needs ``order`` (loop order
        ``L`` needs ``order >= 2 L``).
    values : dict name -> number, optional
        Symbols to fix (pipeline names, ``{'tau': [1, 2]}`` expanded); the
        others stay symbolic.
    mean_field : None | 'solve' | dict
        ``None`` keeps the saddle symbols, ``'solve'`` substitutes the
        pipeline's saddle at ``parameters`` (the model defaults if omitted),
        a dict gives them.
    """
    if not isinstance(model, dict) and callable(getattr(model, 'build', None)):
        model = model.build()
    _reject_spatial(model)
    probe = _expanded_theory(model, 2, use_cache, verbose)
    degree = _drift_degree(probe, model)
    if order is None:
        if degree is None:
            raise SymbolicOutError(
                f"model {model.get('name', '<unnamed>')!r}: the drift is not a "
                f'polynomial in the fields (or its degree cannot be read off '
                f'the mean-field sector); give order= (the highest drift '
                f'degree to keep; loop order L needs order >= 2 L)')
        order = max(degree, 2)
    order = int(order)
    if order < 2:
        raise SymbolicOutError('order must be >= 2 (the first nonlinear drift term)')
    ft = _expanded_theory(model, order + 1, use_cache, verbose)
    from daedalus.api._propagator import build_propagator
    prop = build_propagator(ft, model, use_cache=use_cache, verbose=verbose)
    nf = prop['nf']
    names = prop['ring_gen_names']
    resp_names, phys_names = list(names[:nf]), list(names[nf:2 * nf])

    # substitutions: saddle (mean_field) then the user's values
    sub_names: dict = {}
    if mean_field == 'solve':
        npar = _solve_mean_field(ft, model, _model_fundamental(model, parameters))
        sub_names.update(_mf_symbols_only(model, npar))
    elif isinstance(mean_field, dict):
        sub_names.update(_expand_param_names(_as_name_dict(mean_field)))
    elif mean_field is not None:
        raise SymbolicOutError(
            f"mean_field must be None, 'solve' or a dict; got {mean_field!r}")
    uvals = _expand_param_names(_as_name_dict(values))
    sub_names.update(uvals)
    known = {str(v) for poly in list(ft._by_tp.values()) + list(
        (ft._mf_sector_raw or {}).values()) for c in poly.coefficients()
        for v in SR(c).variables()}
    known |= {str(v) for e in prop['K_ft'].list() for v in SR(e).variables()}
    # a declared parameter (a kernel's time constant enters only the
    # propagator); an indexed one with its unit
    declared = [re.escape(str(p['name'])) + (r'\d+' if p.get('indexed') else '')
                for p in model.get('parameters') or []]
    unknown = sorted(k for k in sub_names if k not in known and not any(
        re.fullmatch(d, k) for d in declared))
    if unknown:
        raise SymbolicOutError(
            f'values= / mean_field= name {unknown}, which the action does not '
            f'contain (indexed symbols carry their unit: xstar1, tau2, W12); '
            f'known: {sorted(known)}')
    sub = {SR.var(k): SR(v) for k, v in sub_names.items()}

    def fix(e):
        e = SR(e)
        return e.subs(sub) if sub else e

    omega = prop['omega']
    K0 = matrix(SR, [[fix(SR(prop['K_ft'][r, p]).subs({omega: 0}))
                      for p in range(nf)] for r in range(nf)])
    det = SR(K0.det()).simplify_rational()
    if det.is_trivial_zero() or det == 0:
        raise UnsupportedModelError(
            'K(omega = 0) is singular: G(omega = 0) does not exist (a '
            'marginal mode, e.g. a conserved quantity or a critical '
            'coupling), so the stationary mean relation has no solved form')

    forbidden = _forbidden_symbols(model)
    R = ft.ring()
    gens = [str(g) for g in R.gens()]
    terms = []
    for m in range(2, order + 1):
        sector = ft._by_tp.get((1, m), R.zero())
        for exps, coeff in sector.dict().items():
            resp = [gens[i] for i, e in enumerate(exps) if e and i < nf]
            legs = tuple(gens[i] for i, e in enumerate(exps) if i >= nf
                         for _ in range(int(e)))
            c = fix(coeff)
            bad = sorted(str(v) for v in c.variables()
                         if str(v) in forbidden or str(v).startswith('z_'))
            if bad:
                raise UnsupportedModelError(
                    f'drift vertex {coeff} * {resp[0]} * {"*".join(legs)} '
                    f'carries {bad}: a time derivative or a kernel on a '
                    f'nonlinear term, whose stationary average is not a '
                    f'moment at one time')
            if c != 0:
                terms.append(DriftTerm(resp[0], c, legs))
    rel = MeanRelation(
        model_name=model.get('name', '<unnamed>'), phys_names=phys_names,
        resp_names=resp_names, K0=K0, terms=terms, order=order,
        drift_degree=degree, values=dict(sub_names))
    fl = {}
    for pf in model.get('physical_fields') or []:
        nm, lx = str(pf.get('name')), pf.get('latex') or str(pf.get('name'))
        for p in phys_names:
            idx = p[len(nm):]
            if p.startswith(nm) and idx.isdigit():
                fl[p] = f'{lx}_{{{idx}}}'
    rel._cache['field_latex'] = fl
    if mean_field is None and any(_is_saddle_symbol(str(v), model)
                                  for t in terms for v in SR(t.coefficient).variables()):
        rel.notes.append('saddle symbols kept: the relation holds at the '
                         'mean-field saddle (mean_field=\'solve\' substitutes it)')
    if not rel.exact:
        rel.notes.append(rel.truncation)
    return rel


# ═════════════════════════════════════════════════════════════════════════════
# The one-loop mean, diagram by diagram
# ═════════════════════════════════════════════════════════════════════════════

C0_FORMS = ('named', 'integral', 'rational', 'residues')


@dataclass(frozen=True)
class Tadpole:
    """One one-loop mean diagram: the drift vertex
    ``coefficient * xt_response * dx_p * dx_q`` with its two physical legs
    closed by the equal-time tree covariance ``C0_pq(0)``, attached to the
    external field ``a`` by ``G_{a, response}(omega = 0)``:

        <dx_a>^{(1)}  ⊃  -G_{a, response}(0) * coefficient * C0_pq(0)

    ``legs = (p, q)`` (``p == q`` allowed); ``propagator`` maps each physical
    field ``a`` to ``G_{a, response}(0)``."""
    response: str
    legs: tuple
    coefficient: Any
    propagator: dict

    def weight(self, a, G0=None):
        """``-G_{a, response}(0) * coefficient``: the factor of
        ``C0_pq(0)`` in this diagram's contribution to ``<dx_a>`` (``G0``:
        ``{(a, response): entry}`` in place of the explicit entries)."""
        g = self.propagator[a] if G0 is None else G0[(a, self.response)]
        return -g * self.coefficient


@dataclass
class OneLoopMean:
    """The one-loop mean shift of a model as its tadpole diagrams
    (:class:`Tadpole`), with the equal-time tree covariance ``C0_pq(0)`` in
    one of the forms of :meth:`C0`:

    ``'named'``     the symbol ``kappa^{(0)}(dx_p, dx_q)`` (as in
                    :meth:`MeanRelation.loop_order`);
    ``'integral'``  ``(1/2 pi) int C0_pq(omega) d omega`` (held, unevaluated)
                    of the symbolic tree covariance;
    ``'rational'``  its exact value, rational in the parameters and the
                    saddle (state-space route, no roots);
    ``'residues'``  the residue sum over the poles of ``C0_pq``: explicit for
                    factors of degree <= 2, ``RootOf`` symbols above.

    The expressions are not simplified towards any closed form; ``relation``
    is the :class:`MeanRelation` they come from."""
    relation: MeanRelation
    tadpoles: list
    model: Any = field(repr=False, default=None)
    parameters: Optional[dict] = field(repr=False, default=None)
    notes: list = field(default_factory=list)
    _cache: dict = field(default_factory=dict, repr=False)

    @property
    def phys_names(self) -> list:
        return self.relation.phys_names

    @property
    def loops(self) -> list:
        """The distinct loops ``(p, q)`` of the tadpoles, in field order."""
        seen = []
        for t in self.tadpoles:
            if t.legs not in seen:
                seen.append(t.legs)
        return seen

    # ── C0(0) ───────────────────────────────────────────────────────────
    def _legs(self, p, q) -> tuple:
        return self.relation._canon((p, q))

    def covariance(self):
        """The symbolic tree covariance (:class:`TreeCovariance`) the
        ``'integral'`` and ``'residues'`` forms use (built once, with this
        relation's values and saddle substituted before it is simplified)."""
        if 'cov' not in self._cache:
            from daedalus.api.symbolic_out import tree_covariance
            self._cache['cov'] = tree_covariance(
                self.model, values=self.relation.values or None,
                parameters=self.parameters)
        return self._cache['cov']

    def _fix(self, e):
        sub = {SR.var(k): SR(v) for k, v in self.relation.values.items()}
        e = SR(e)
        return e.subs(sub) if sub else e

    def _entry(self, legs):
        """``C0_pq(omega)`` with this relation's substitutions; refused if it
        has a contact term (a white-noise-like field: ``C_pq(tau)`` has a
        ``delta(tau)`` and no equal-time value)."""
        cov = self.covariance()
        e = self._fix(cov.entry(*legs))
        tail = self._fix(cov.contact(*legs))
        if not SR(tail).is_trivial_zero() and SR(tail).simplify_rational() != 0:
            raise UnsupportedModelError(
                f'C0_{legs[0]},{legs[1]}(omega) tends to {tail} as omega -> '
                f'infinity: C(tau) has a delta(tau) term and no equal-time '
                f'value, so this loop has no finite C0(0)')
        return e

    def C0(self, p, q, form: str = 'named'):
        """The equal-time tree covariance ``C0_pq(tau = 0)`` in ``form``
        (class docstring)."""
        legs = self._legs(p, q)
        if form == 'named':
            return self.relation.kappa(*legs, ell=0)
        if form == 'integral':
            from sage.all import integral, pi, oo
            cov = self.covariance()
            w = cov.omega
            return integral(self._entry(legs), w, -oo, oo, hold=True) / (2 * pi)
        if form == 'rational':
            return self._rational(legs)
        if form == 'residues':
            ms = self.modes(*legs)
            return ms.expression('negative').subs({ms.tau: 0})
        raise SymbolicOutError(f'form must be one of {C0_FORMS}; got {form!r}')

    def modes(self, p, q):
        """The time-domain mode sum of ``C0_pq`` (:class:`ModeSum`, the
        ``'residues'`` form at ``tau = 0``); its half-plane split is decided at
        the model defaults (``parameters=``) and stated in its
        ``assumptions``."""
        legs = self._legs(p, q)
        key = ('modes', legs)
        if key not in self._cache:
            from daedalus.api.symbolic_out import tree_modes
            e = self._entry(legs)
            cov = self.covariance()
            free = {str(v) for v in e.variables()} - {str(cov.omega)}
            ref = {k: v for k, v in self._reference().items() if k in free}
            self._cache[key] = tree_modes(e, cov.omega, reference=ref or None)
        return self._cache[key]

    def _reference(self) -> dict:
        """The model defaults (``parameters=`` over them), with the solved
        saddle, as pipeline symbol names."""
        if 'ref' not in self._cache:
            cov = self.covariance()
            ref = {str(k): v for k, v in cov.default_values().items()}
            ref.update(self.relation.values)
            self._cache['ref'] = ref
        return self._cache['ref']

    def _rational(self, legs):
        key = ('rational', legs)
        if key in self._cache:
            return self._cache[key]
        from daedalus.api.symbolic_loop import symbolic_cumulant
        ext = [_external_field(self.model, f) for f in legs]
        vals = self.relation.values
        fixed = {k: v for k, v in vals.items()
                 if not _is_saddle_symbol(k, self.model)}
        saddle = {k: v for k, v in vals.items()
                  if _is_saddle_symbol(k, self.model)}
        params = _pack_parameters(self.model, self.parameters, fixed)
        symbols = [n for n in _expand_param_names(params) if n not in fixed]
        sc = symbolic_cumulant(self.model, k=2, max_ell=0, external_fields=ext,
                               parameters=params, symbols=symbols,
                               saddle=saddle or 'symbol')
        e = self._fix(SR(sc.by_ell[0]))
        self._cache[key] = e
        return e

    # ── the mean ────────────────────────────────────────────────────────
    def G0(self, a, r, propagator: str = 'explicit'):
        """``G_{a r}(omega = 0)``: its expression (``'explicit'``) or the
        symbol ``G0_a_r`` (``'named'``)."""
        rel = self.relation
        i, j = rel.phys_names.index(str(a)), rel.resp_names.index(str(r))
        if propagator == 'explicit':
            return rel.G0[i, j]
        if propagator != 'named':
            raise SymbolicOutError(f"propagator must be 'explicit' or 'named'; "
                                   f'got {propagator!r}')
        name = f'G0_{a}_{r}'
        lx = (rf'G_{{{rel._field_latex(str(a))}, '
              rf'{self._resp_latex(str(r))}}}(0)')
        return SR.var(name, latex_name=lx)

    def _resp_latex(self, r) -> str:
        for rf in (self.model or {}).get('response_fields') or []:
            nm, lx = str(rf.get('name')), rf.get('latex')
            idx = r[len(nm):]
            if lx and r.startswith(nm) and idx.isdigit():
                return f'{lx}_{{{idx}}}'
        return r

    def contributions(self, form: str = 'named',
                      propagator: str = 'explicit') -> list:
        """``[(tadpole, {a: its contribution to <dx_a>})]``; ``propagator``
        as in :meth:`G0`."""
        c0 = {legs: self.C0(*legs, form=form) for legs in self.loops}
        G0 = None
        if propagator != 'explicit':
            G0 = {(a, t.response): self.G0(a, t.response, propagator)
                  for t in self.tadpoles for a in self.phys_names}
        return [(t, {a: t.weight(a, G0) * c0[t.legs] for a in self.phys_names})
                for t in self.tadpoles]

    def mean_shift(self, form: str = 'named',
                   propagator: str = 'explicit') -> dict:
        """``<dx_a>^{(1)} = <x_a> - x*_a`` at one loop, ``{a: sum over the
        tadpoles}`` (not simplified); ``propagator`` as in :meth:`G0`."""
        out = {a: SR(0) for a in self.phys_names}
        for _, contrib in self.contributions(form, propagator):
            for a, v in contrib.items():
                out[a] += v
        return out

    def evaluate(self, form: str = 'rational', values: Optional[dict] = None
                 ) -> dict:
        """``{a: number}``: the one-loop mean shift with every remaining
        symbol given in ``values`` (pipeline names; the saddle too, unless the
        relation was built with ``mean_field=``).  ``'integral'`` integrates
        numerically (an independent check of ``'rational'``), ``'residues'``
        sums the modes at 256 bits."""
        vals = _expand_param_names(_as_name_dict(values))
        if form == 'integral':
            c0 = {legs: self._integral_number(legs, vals) for legs in self.loops}
        elif form == 'residues':
            c0 = {}
            for legs in self.loops:
                ms = self.modes(*legs)
                missing = sorted(k for k in ms.reference if k not in vals)
                if missing:
                    raise SymbolicOutError(f'no value for {missing}')
                c0[legs] = complex(ms.evaluate(
                    0.0, {k: vals[k] for k in ms.reference})[()])
        elif form in ('rational', 'named'):
            c0 = {legs: self.C0(*legs, form='rational') for legs in self.loops}
        else:
            raise SymbolicOutError(f'form must be one of {C0_FORMS}; got {form!r}')
        sub = {SR.var(k): SR(v) for k, v in vals.items()}
        out = {}
        for a in self.phys_names:
            tot = SR(0)
            for t in self.tadpoles:
                tot += t.weight(a) * c0[t.legs]
            tot = SR(tot).subs(sub)
            free = sorted(str(v) for v in tot.variables())
            if free:
                raise SymbolicOutError(f'no value for {free}')
            out[a] = complex(tot.n())
        return out

    def _integral_number(self, legs, vals) -> complex:
        from scipy.integrate import quad
        cov = self.covariance()
        w = cov.omega
        e = self._entry(legs).subs({SR.var(k): SR(v) for k, v in vals.items()})
        free = sorted(str(v) for v in e.variables() if str(v) != str(w))
        if free:
            raise SymbolicOutError(f'no value for {free}')
        f = e.function(w)
        re = quad(lambda x: float(f(x).real()), -np.inf, np.inf,
                  epsabs=0, epsrel=1e-13, limit=400)[0]
        im = quad(lambda x: float(f(x).imag()), -np.inf, np.inf,
                  epsabs=1e-300, epsrel=1e-13, limit=400)[0]
        return complex(re, im) / (2 * np.pi)

    # ── exports ─────────────────────────────────────────────────────────
    def sympy(self, form: str = 'named', propagator: str = 'explicit') -> dict:
        return {k: to_sympy(v)
                for k, v in self.mean_shift(form, propagator).items()}

    def latex(self, form: str = 'named', propagator: str = 'named') -> str:
        """One line per field: ``<dx_a>^{(1)} = sum of tadpoles`` (the
        propagator named by default: one term per diagram)."""
        rel = self.relation
        lines = []
        for a in self.phys_names:
            terms = [to_latex(c[a])
                     for _, c in self.contributions(form, propagator)
                     if not SR(c[a]).is_trivial_zero()]
            lines.append(f'{to_latex(rel.kappa(a, ell=1))} = '
                         + (' + '.join(terms) if terms else '0'))
        return r' \\ '.join(lines)

    def __str__(self):
        lines = [f'OneLoopMean({self.relation.model_name!r}; '
                 f'{len(self.tadpoles)} tadpoles)']
        for t in self.tadpoles:
            lines.append(f'  {t.response} <- ({t.coefficient}) * '
                         f'C0_{t.legs[0]},{t.legs[1]}(0)')
        return '\n'.join(lines)


def _external_field(model, phys: str) -> tuple:
    """``'dv2'`` -> ``('v', 2)``: the pipeline's external field of a physical
    fluctuation generator."""
    for pf in model.get('physical_fields') or []:
        nm = str(pf.get('name'))
        idx = phys[len(nm):]
        if phys.startswith(nm) and idx.isdigit():
            return (str(pf.get('natural_name') or nm), int(idx))
    raise SymbolicOutError(f'cannot map the field {phys!r} to an external '
                           f'field of the model')


def _pack_parameters(model, parameters: Optional[dict], fixed: dict) -> dict:
    """The model's parameter dict (``parameters=`` over the defaults) with
    the pipeline-named ``fixed`` values written into it (``W12`` into
    ``W[0][1]``, ``tau2`` into ``tau[1]``)."""
    import copy
    full = {p['name']: copy.deepcopy(p['default'])
            for p in model.get('parameters') or []
            if p.get('default') is not None and not p.get('mean_field')}
    full.update(copy.deepcopy(dict(parameters or {})))
    left = dict(fixed)
    for name, v in full.items():
        if isinstance(v, (list, tuple, np.ndarray)):
            v = [list(r) if isinstance(r, (list, tuple, np.ndarray)) else r
                 for r in v]
            for i, row in enumerate(v):
                if isinstance(row, list):
                    for j in range(len(row)):
                        k = f'{name}{i + 1}{j + 1}'
                        if k in left:
                            row[j] = left.pop(k)
                else:
                    k = f'{name}{i + 1}'
                    if k in left:
                        v[i] = left.pop(k)
            full[name] = v
        elif name in left:
            full[name] = left.pop(name)
    if left:
        raise SymbolicOutError(f'values= names {sorted(left)} are not model '
                               f'parameters')
    return full


def one_loop_mean(model: dict, *, values: Optional[dict] = None,
                  mean_field: Any = None, parameters: Optional[dict] = None,
                  use_cache: bool = True, verbose: bool = False) -> OneLoopMean:
    """The one-loop (``k = 1``, ``ell = 1``) mean shift of ``model`` as its
    tadpole diagrams (:class:`OneLoopMean`).

    Each tadpole is ``-G(omega = 0)_{a r} V^{(2)}_{r; pq} C0_pq(0)``: a
    quadratic drift vertex of the expanded action, its loop closed by the
    equal-time tree covariance.  One loop needs the drift to degree two only,
    so the result is exact at this order for any drift (a non-polynomial one
    included).  ``values``, ``mean_field`` and ``parameters`` as in
    :func:`mean_relation`."""
    if not isinstance(model, dict) and callable(getattr(model, 'build', None)):
        model = model.build()
    rel = mean_relation(model, order=2, values=values, mean_field=mean_field,
                        parameters=parameters, use_cache=use_cache,
                        verbose=verbose)
    G0 = rel.G0
    tads = []
    for t in rel.terms:
        if t.degree != 2:
            continue
        r = rel.resp_names.index(t.response)
        prop = {a: G0[i, r] for i, a in enumerate(rel.phys_names)}
        tads.append(Tadpole(t.response, t.legs, t.coefficient, prop))
    out = OneLoopMean(relation=rel, tadpoles=tads, model=model,
                      parameters=parameters)
    out.notes = [n for n in rel.notes if not n.startswith('Taylor-truncated')]
    return out
