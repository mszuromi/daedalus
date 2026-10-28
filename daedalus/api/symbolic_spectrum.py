r"""
The one-loop spectrum, diagram by diagram, symbolic.

The tree lines of the fluctuations ``dx`` about the saddle and the response
fields ``xt`` are

    <dx_i(t) xt_j(s)> = G_ij(t - s)       (retarded; Itô: G(0) = 0)
    <dx_i(t) dx_j(s)> = C0_ij(t - s)

with ``L(t) = int dw/2pi e^{i w t} L(w)``, ``G(w) = K(w)^{-1}`` and ``C0(w) = G D
G^H`` as in :func:`~daedalus.api.symbolic_out.tree_covariance`.  With the
interaction monomials ``c xt^p dx^m`` of the expanded action (weight
``e^{-S}``), Wick's theorem gives the one-loop increment of the spectrum
``S_ab(w) = int C_ab(tau) e^{i w tau} d tau``, ``C_ab(tau) = kappa(x_a(t), x_b(t +
tau))`` (the convention of :func:`daedalus.spectra`), as

    S^{(1)}_ab(w) = sum_{alpha beta} L_{a alpha}(w) Sigma_{alpha beta}(w) L_{b beta}(-w).

Here ``alpha`` and ``beta`` run over the response and the physical legs, with
``L_{a, xt_r} = G_{a r}`` and ``L_{a, dx_j} = C0_{a j}``.  ``Sigma`` is a sum of
three kinds of diagrams (:class:`SelfEnergyTerm`), read off the vertices of
degree three and four:

* ``tadpole``: one quartic vertex, whose other two legs are closed by the
  equal-time tree covariance ``C0_pq(0)``;
* ``mean_shift``: one cubic vertex, whose third leg is on the one-loop mean
  shift ``kappa^{(1)}_p`` (:func:`~daedalus.api.symbolic_mean.one_loop_mean`);
* ``bubble``: two cubic vertices joined by two lines,
  ``(1/2 pi) int L1(nu) L2(w - nu) d nu``.

Equal times are Itô's (``G(0) = 0``).  A line with an instantaneous part (a
spike train's ``G`` or contact term, ``L = L(inf) + L_r`` with ``L_r`` decaying)
makes a loop depend on that rule:

* a tadpole ``C0_pq(0)`` whose ``C_pq(tau)`` jumps at the tie takes its Itô
  value (:meth:`~daedalus.api.symbolic_mean.OneLoopMean.C0`);
* a bubble with ``L1 = L1(inf) + L1_r``: the ``delta(tau)`` of ``L1(inf)``
  puts both vertices at one time, where the other line takes its Itô value,
  zero for ``G`` and ``C0_pq(0)`` for ``C0``.  The bubble is
  ``(1/2 pi) int L1_r(nu) L2(w - nu) d nu + L1(inf) L2(tau = 0)``, which
  converges absolutely.

The block ``Sigma_{xt xt}`` renormalizes the noise (``G Sigma G^H``).  The blocks
``Sigma_{xt dx}`` and ``Sigma_{dx xt}`` renormalize the response (``G Sigma C0 +
C0 Sigma G^H``).  ``Sigma_{dx dx}`` vanishes by causality.

Lines and loops are kept in one of two forms:

* named functions (``G_dv1_vt1(w)``, ``C0_dv1_dv2(nu)``, ``kappa^{(0)}``,
  ``kappa^{(1)}``);
* written out: the rational entries of the symbolic propagator, with a bubble
  as its held Fourier integral.

The equal-time quantities take the forms of :meth:`OneLoopMean.C0`
(``named``, ``integral``, ``rational``, ``residues``).  Nothing is simplified
towards a closed form.  :meth:`OneLoopSpectrum.evaluate` gives numbers: the
equal-time parts exactly, the bubbles by quadrature in ``nu``.

One loop needs the vertices only up to degree four in all fields, so the
result is exact at this order for any drift and noise.

Refused (``UnsupportedModelError``):

* spatial models;
* a time derivative or a kernel on a vertex;
* a loop with no Itô value: a tadpole ``C0_pq(0)`` with a contact term
  (``delta(0)``), or a bubble of two lines that both have a ``delta(tau)``
  part (``delta(tau)^2``).
* a singular ``K(w = 0)``, which leaves no mean shift.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
from sage.all import SR, function, integral, pi, oo, matrix

from daedalus.api.symbolic_out import (
    SymbolicOutError, UnsupportedModelError, _expanded_theory, to_sympy,
    to_latex,
)
from daedalus.api.symbolic_mean import (
    C0_FORMS, OneLoopMean, one_loop_mean, _forbidden_symbols,
)

__all__ = ['Line', 'SelfEnergyTerm', 'OneLoopSpectrum', 'one_loop_spectrum']

LINE_FORMS = ('named', 'explicit')
KINDS = ('tadpole', 'mean_shift', 'bubble')


@dataclass(frozen=True)
class Line:
    """A tree line at frequency ``freq``: ``G_{row col}`` (``kind='G'``, row
    physical, column response) or ``C0_{row col}`` (``kind='C0'``)."""
    kind: str
    row: str
    col: str
    freq: Any

    def __str__(self):
        return f'{self.kind}_{self.row},{self.col}({self.freq})'


@dataclass(frozen=True)
class SelfEnergyTerm:
    """One diagram of ``Sigma_{left, right}(w)`` (module docstring).
    ``coefficient`` collects the vertex coefficients, the sign and the Wick
    multiplicity.  ``lines`` holds a bubble's two internal lines (the first at
    ``nu``), ``loop`` a tadpole's ``(p, q)`` of ``C0_pq(0)``, and ``shift`` the
    field ``p`` of a mean-shift insertion ``kappa^{(1)}_p``."""
    kind: str
    left: str
    right: str
    coefficient: Any
    lines: tuple = ()
    loop: Optional[tuple] = None
    shift: Optional[str] = None
    vertices: tuple = ()

    def __str__(self):
        if self.kind == 'tadpole':
            body = f'C0_{self.loop[0]},{self.loop[1]}(0)'
        elif self.kind == 'mean_shift':
            body = f'kappa1_{self.shift}'
        else:
            body = ' '.join(str(ln) for ln in self.lines)
        return (f'{self.kind:<10} Sigma[{self.left},{self.right}] <- '
                f'({self.coefficient}) * {body}')


@dataclass
class OneLoopSpectrum:
    """The one-loop spectrum ``S^{(1)}(w)`` of a model as the self-energy
    diagrams of the module docstring (:class:`SelfEnergyTerm`).

    Forms.  ``form`` picks the lines: ``'named'`` uses the functions
    ``G_a_r(w)`` and ``C0_a_b(w)``, ``'explicit'`` the rational entries of the
    symbolic propagator.  ``c0`` picks the equal-time loops ``C0_pq(0)`` and
    the mean shift ``kappa^{(1)}_p``, in the forms of :meth:`OneLoopMean.C0`;
    with ``'named'`` they are the symbols ``kappa^{(0)}`` and ``kappa^{(1)}``.
    ``mean`` is the :class:`OneLoopMean` the mean shifts and the tree
    covariance come from."""
    mean: OneLoopMean
    terms: list
    notes: list = field(default_factory=list)
    _cache: dict = field(default_factory=dict, repr=False)

    @property
    def phys_names(self) -> list:
        return self.mean.relation.phys_names

    @property
    def resp_names(self) -> list:
        return self.mean.relation.resp_names

    @property
    def legs(self) -> list:
        """The rows and columns of ``Sigma``: response legs, then physical."""
        return list(self.resp_names) + list(self.phys_names)

    @property
    def omega(self):
        return self.mean.covariance().omega

    @property
    def nu(self):
        if 'nu' not in self._cache:
            self._cache['nu'] = SR.var('nu_loop', latex_name=r'\nu')
        return self._cache['nu']

    def by_kind(self, kind: str) -> list:
        if kind not in KINDS:
            raise SymbolicOutError(f'kind must be one of {KINDS}; got {kind!r}')
        return [t for t in self.terms if t.kind == kind]

    # ── lines ───────────────────────────────────────────────────────────
    def _line_latex(self, kind, row, col) -> str:
        rel = self.mean.relation
        if kind == 'G':
            return (rf'G_{{{rel._field_latex(row)}, '
                    rf'{self.mean._resp_latex(col)}}}')
        return (rf'C^{{(0)}}_{{{rel._field_latex(row)}, '
                rf'{rel._field_latex(col)}}}')

    def _named_fn(self, kind, row, col):
        key = ('fn', kind, row, col)
        if key not in self._cache:
            self._cache[key] = function(f'{kind}_{row}_{col}',
                                        latex_name=self._line_latex(kind, row, col))
        return self._cache[key]

    def _explicit(self, kind, row, col):
        """The entry as a function of ``omega``, with the relation's values."""
        key = ('entry', kind, row, col)
        if key not in self._cache:
            cov = self.mean.covariance()
            if kind == 'G':
                e = cov.G[cov.phys_names.index(row), cov.resp_names.index(col)]
            else:
                e = cov.entry(row, col)
            self._cache[key] = self.mean._fix(e)
        return self._cache[key]

    def _decay(self, kind, row, col) -> int:
        """``deg Q - deg P`` of the entry ``P / Q`` in ``omega`` (``-1`` for a
        zero entry): 0 for a line with a ``delta(tau)`` part, 1 for one that
        jumps at ``tau = 0``."""
        key = ('decay', kind, row, col)
        if key not in self._cache:
            w = self.omega
            e = SR(self._explicit(kind, row, col))
            num, den = e.numerator(), e.denominator()
            deg = lambda p: int(p.degree(w)) if p.has(w) else 0
            self._cache[key] = (-1 if SR(num).is_trivial_zero()
                                else deg(den) - deg(num))
        return self._cache[key]

    def _tail(self, ln: Line, form: str):
        """The weight of ``delta(tau)`` in a line, the ``omega -> infinity``
        limit of its entry: the symbol ``G_a_r_inf`` / ``C0_a_b_inf``
        (``'named'``) or its value; zero for a line that decays."""
        if self._decay(ln.kind, ln.row, ln.col) != 0:
            return SR(0)
        if form == 'named':
            return SR.var(f'{ln.kind}_{ln.row}_{ln.col}_inf',
                          latex_name=self._line_latex(ln.kind, ln.row, ln.col)
                          + r'(\infty)')
        cov = self.mean.covariance()
        e = (cov.instantaneous(ln.row, ln.col) if ln.kind == 'G'
             else cov.contact(ln.row, ln.col))
        return self.mean._fix(e)

    def line(self, ln: Line, form: str = 'named'):
        """The value of a :class:`Line` in ``form`` (class docstring)."""
        if form == 'named':
            return self._named_fn(ln.kind, ln.row, ln.col)(ln.freq)
        if form != 'explicit':
            raise SymbolicOutError(f'form must be one of {LINE_FORMS}; '
                                   f'got {form!r}')
        return self._explicit(ln.kind, ln.row, ln.col).subs(
            {self.omega: ln.freq})

    def _outer(self, a, alpha, sign):
        kind = 'G' if alpha in self.resp_names else 'C0'
        return Line(kind, str(a), alpha, sign * self.omega)

    # ── equal-time parts ────────────────────────────────────────────────
    def _equal_time(self, t: SelfEnergyTerm, c0: str):
        if c0 not in C0_FORMS:
            raise SymbolicOutError(f'c0 must be one of {C0_FORMS}; got {c0!r}')
        if t.kind == 'tadpole':
            return self.mean.C0(*t.loop, form=c0)
        if c0 == 'named':
            return self.mean.relation.kappa(t.shift, ell=1)
        key = ('shift', c0)
        if key not in self._cache:
            self._cache[key] = self.mean.mean_shift(c0)
        return self._cache[key][t.shift]

    # ── Sigma and the spectrum ──────────────────────────────────────────
    def pairs(self) -> list:
        """The ``(left, right)`` entries of ``Sigma`` with diagrams."""
        seen = []
        for t in self.terms:
            if (t.left, t.right) not in seen:
                seen.append((t.left, t.right))
        return seen

    def _bubble_parts(self, left, right, form, c0):
        """The bubbles of ``Sigma_{left, right}``: the integrand over ``nu``
        and the equal-time part.  A line's ``delta(tau)`` part pins the two
        vertices to one time and takes the other line there, at its Itô value
        (module docstring); the rest of the product is integrated."""
        integrand, equal = SR(0), SR(0)
        for t in self.terms:
            if t.kind != 'bubble' or (t.left, t.right) != (left, right):
                continue
            tails = [self._tail(ln, form) for ln in t.lines]
            l1, l2 = (self.line(ln, form) - tl for ln, tl in zip(t.lines, tails))
            integrand += t.coefficient * l1 * l2
            for tl, other in zip(tails, reversed(t.lines)):
                if not SR(tl).is_trivial_zero() and other.kind == 'C0':
                    equal += (t.coefficient * tl
                              * self.mean.C0(other.row, other.col, form=c0))
        return integrand, equal

    def sigma(self, left, right, form: str = 'named', c0: str = 'named'):
        """``Sigma_{left, right}(omega)``: equal-time terms plus the bubbles as
        one held integral ``(1/2 pi) int ... d nu``."""
        left, right = str(left), str(right)
        out = SR(0)
        for t in self.terms:
            if (t.left, t.right) == (left, right) and t.kind != 'bubble':
                out += t.coefficient * self._equal_time(t, c0)
        bub, eq = self._bubble_parts(left, right, form, c0)
        out += eq
        if not SR(bub).is_trivial_zero():
            out += integral(bub, self.nu, -oo, oo, hold=True) / (2 * pi)
        return out

    def sigma_matrix(self, form: str = 'named', c0: str = 'named'):
        """``Sigma`` on :attr:`legs` x :attr:`legs`."""
        legs = self.legs
        M = matrix(SR, len(legs), len(legs))
        for l, r in self.pairs():
            M[legs.index(l), legs.index(r)] = self.sigma(l, r, form, c0)
        return M

    def spectrum(self, a, b, form: str = 'named', c0: str = 'named'):
        """``S^{(1)}_ab(omega) = sum L_{a alpha}(w) Sigma_{alpha beta}(w)
        L_{b beta}(-w)`` (not simplified)."""
        a, b = self._phys(a), self._phys(b)
        out = SR(0)
        for l, r in self.pairs():
            out += (self.line(self._outer(a, l, 1), form)
                    * self.sigma(l, r, form, c0)
                    * self.line(self._outer(b, r, -1), form))
        return out

    def _phys(self, a) -> str:
        if isinstance(a, (tuple, list)) and len(a) == 2:
            for p in self.phys_names:
                if _phys_of(self.mean.model, p) == (str(a[0]), int(a[1])):
                    return p
        if str(a) in self.phys_names:
            return str(a)
        raise SymbolicOutError(f'unknown physical field {a!r}; known: '
                               f'{self.phys_names}')

    # ── numbers ─────────────────────────────────────────────────────────
    def evaluate(self, a, b, omega, values: Optional[dict] = None,
                 c0: str = 'rational', epsrel: float = 1e-12):
        """``S^{(1)}_ab(omega)`` as numbers (complex; an array for an array of
        ``omega``), with ``values`` as in :meth:`OneLoopMean.evaluate` (a
        parameter not given takes its model default; one the build's saddle
        depends on re-solves it and evaluates the diagrams rebuilt there, see
        :meth:`at`).  The equal-time parts are taken in form ``c0``: exact for
        ``'rational'`` and ``'residues'``, by quadrature for ``'integral'``.
        The bubbles are integrated over ``nu`` by adaptive quadrature.  Every
        line is evaluated as a numeric ratio of polynomials (coefficients
        normalized at 256 bits): the exact rationals the tree covariance
        carries for float inputs overflow a double-precision evaluation of
        the expression."""
        a, b = self._phys(a), self._phys(b)
        moved = self.mean.relation.saddle_at(values)
        if moved is not None:
            return self.at(moved).evaluate(a, b, omega, values, c0, epsrel)
        vals = self.mean.relation.with_defaults(values)
        sub = {SR.var(k): SR(v) for k, v in vals.items()}
        w, nu = self.omega, self.nu
        lines: dict = {}

        def line(kind, row, col):
            if (kind, row, col) not in lines:
                lines[kind, row, col] = _numeric_rational(
                    SR(self._explicit(kind, row, col)).subs(sub), w)
            return lines[kind, row, col]

        def number(c):
            e = SR(c).subs(sub)
            free = sorted(str(v) for v in e.variables())
            if free:
                raise SymbolicOutError(f'no value for {free}')
            return complex(e.n())

        eq, bub = [], []   # (outer lines, value) and (outer, coeff, lines)
        for l, r in self.pairs():
            ol = self._outer(a, l, 1)
            orr = self._outer(b, r, -1)
            outer = (line(ol.kind, ol.row, ol.col), line(orr.kind, orr.row, orr.col))
            v = 0j
            for t in self.terms:
                if (t.left, t.right) != (l, r):
                    continue
                if t.kind == 'bubble':
                    c, lns = number(t.coefficient), []
                    for ln, other in zip(t.lines, reversed(t.lines)):
                        tail = number(self._tail(ln, 'explicit'))
                        if tail and other.kind == 'C0':
                            v += c * tail * self._c0_number(
                                (other.row, other.col), c0, vals)
                        lns.append((line(ln.kind, ln.row, ln.col),
                                    float(SR(ln.freq).coefficient(nu)),
                                    float(SR(ln.freq).coefficient(w)), tail))
                    bub.append((outer, c, lns))
                else:
                    v += number(t.coefficient) * self._equal_number(t, c0, vals)
            if v:
                eq.append((outer, v))
        oms = np.atleast_1d(np.asarray(omega, dtype=float))
        out = []
        for x in oms:
            tot = sum(A(x) * B(-x) * v for (A, B), v in eq)
            if bub:
                wts = [c * A(x) * B(-x) for (A, B), c, _ in bub]
                tot += _nu_integral(
                    lambda y: sum(wt * (L1(p1 * y + q1 * x) - t1)
                                  * (L2(p2 * y + q2 * x) - t2)
                                  for wt, (_, _, ((L1, p1, q1, t1),
                                                  (L2, p2, q2, t2)))
                                  in zip(wts, bub)), epsrel)
            out.append(complex(tot))
        out = np.array(out)
        return out if np.ndim(omega) else complex(out[0])

    def at(self, moved: dict) -> 'OneLoopSpectrum':
        """This spectrum rebuilt at the parameters and saddle of
        :meth:`~daedalus.api.symbolic_mean.MeanRelation.saddle_at` (cached)."""
        key = ('at', tuple(sorted(moved['saddle'].items())))
        if key not in self._cache:
            build = self.mean.relation._cache['build']
            self._cache[key] = one_loop_spectrum(
                self.mean.model, values=build['values'] or None,
                mean_field='solve', parameters=moved['parameters'])
        return self._cache[key]

    def _equal_number(self, t, c0, vals) -> complex:
        """A tadpole's ``C0_pq(0)`` or a mean shift, as a number."""
        if c0 == 'named':
            c0 = 'rational'
        if t.kind == 'mean_shift':
            key = ('shift_num', c0, tuple(sorted(vals.items())))
            if key not in self._cache:
                self._cache[key] = self.mean.evaluate(c0, vals)
            return self._cache[key][t.shift]
        return self._c0_number(t.loop, c0, vals)

    def _c0_number(self, legs, c0, vals) -> complex:
        """``C0_pq(0)`` (Itô) as a number, in form ``c0``."""
        if c0 == 'named':
            c0 = 'rational'
        legs = self.mean._legs(*legs)
        if c0 == 'integral':
            return self.mean._integral_number(legs, vals)
        if c0 == 'residues':
            ms = self.mean.modes(*legs)
            missing = sorted(k for k in ms.reference if k not in vals)
            if missing:
                raise SymbolicOutError(f'no value for {missing}')
            return complex(ms.evaluate(
                0.0, {k: vals[k] for k in ms.reference})[()])
        e = SR(self.mean.C0(*legs, form=c0)).subs(
            {SR.var(k): SR(v) for k, v in vals.items()})
        free = sorted(str(v) for v in e.variables())
        if free:
            raise SymbolicOutError(f'no value for {free}')
        return complex(e.n())

    # ── exports ─────────────────────────────────────────────────────────
    def sympy(self, a, b, form: str = 'named', c0: str = 'named'):
        return to_sympy(self.spectrum(a, b, form, c0))

    def latex(self, form: str = 'named', c0: str = 'named') -> str:
        """One line per nonzero entry of ``Sigma`` (named lines by default)."""
        rows = []
        for l, r in self.pairs():
            rows.append(rf'\Sigma_{{{l}, {r}}}(\omega) = '
                        + to_latex(self.sigma(l, r, form, c0)))
        return r' \\ '.join(rows)

    def __str__(self):
        counts = ', '.join(f'{len(self.by_kind(k))} {k}' for k in KINDS)
        lines = [f'OneLoopSpectrum({self.mean.relation.model_name!r}; {counts})']
        lines += [f'  {t}' for t in self.terms]
        return '\n'.join(lines)


def _numeric_rational(e, w):
    """``e`` (a rational function of ``w`` with numeric coefficients) as a
    numpy callable ``P(w) / Q(w)``, the coefficients normalized by the
    largest one of ``Q`` at 256 bits before they are rounded."""
    from sage.all import ComplexField
    free = sorted(str(v) for v in e.variables() if str(v) != str(w))
    if free:
        raise SymbolicOutError(f'no value for {free}')
    CF = ComplexField(256)
    num, den = SR(e).numerator().expand(), SR(e).denominator().expand()

    def coeffs(p):
        deg = int(p.degree(w)) if p.has(w) else 0
        return [CF(p.coefficient(w, k)) for k in range(deg, -1, -1)]

    nc, dc = coeffs(num), coeffs(den)
    s = max(abs(c) for c in dc)
    nc = np.array([complex(c / s) for c in nc])
    dc = np.array([complex(c / s) for c in dc])
    return lambda x: np.polyval(nc, x) / np.polyval(dc, x)


def _nu_integral(f, epsrel: float) -> complex:
    """``(1/2 pi) int f(nu) d nu`` over the real line."""
    from scipy.integrate import quad_vec
    def g(y):
        z = complex(f(y))
        return np.array([z.real, z.imag])
    v = quad_vec(g, -np.inf, np.inf, epsabs=1e-300, epsrel=epsrel,
                 norm='max', limit=2000)[0]
    return complex(v[0], v[1]) / (2 * np.pi)


def _phys_of(model, phys: str):
    from daedalus.api.symbolic_mean import _external_field
    try:
        return _external_field(model, phys)
    except SymbolicOutError:
        return None


def _vertices(ft, resp, phys, fix, forbidden) -> tuple:
    """The cubic and quartic interaction monomials ``(c, legs, label)``."""
    R = ft.ring()
    gens = [str(g) for g in R.gens()]
    cubic, quartic = [], []
    for (p, m), poly in sorted(ft._by_tp.items()):
        if p < 1 or p + m not in (3, 4):
            continue
        for exps, coeff in poly.dict().items():
            legs = tuple(gens[i] for i, e in enumerate(exps)
                         for _ in range(int(e)))
            other = [g for g in legs if g not in resp and g not in phys]
            c = fix(coeff)
            bad = sorted({str(v) for v in SR(c).variables()
                          if str(v) in forbidden or str(v).startswith('z_')}
                         | set(other))
            if bad:
                raise UnsupportedModelError(
                    f'vertex {coeff} * {"*".join(legs)} carries {bad}: a time '
                    f'derivative or a kernel on an interaction vertex is not '
                    f'covered')
            if SR(c).is_trivial_zero():
                continue
            v = (c, legs, f'({c})*{"*".join(legs)}')
            (cubic if len(legs) == 3 else quartic).append(v)
    return cubic, quartic


def _canonical_line(kind, row, col, f, phys_order, w, nu):
    """``C0_ij(f) = C0_ji(-f)`` (real fields): the lower index first, and a
    diagonal ``C0`` (even) at the frequency with ``+omega`` (else ``+nu``)."""
    if kind == 'C0':
        if phys_order.index(row) > phys_order.index(col):
            row, col, f = col, row, -f
        elif row == col:
            lead = f.coefficient(w) if f.has(w) else f.coefficient(nu)
            if lead < 0:
                f = -f
    return Line(kind, row, col, SR(f).expand())


def _canonical_bubble(lines, phys_order, w, nu) -> tuple:
    """The two lines of a bubble in a canonical form: ``int d nu`` over the
    real line is invariant under ``nu -> omega - nu`` (which swaps the two
    frequencies), so the smaller image (lines sorted) represents both."""
    best = None
    for image in (nu, w - nu):
        lns = sorted((_canonical_line(k, r, c, SR(f).subs({nu: image}),
                                      phys_order, w, nu)
                      for k, r, c, f in lines),
                     key=lambda ln: (ln.kind, ln.row, ln.col, str(ln.freq)))
        key = tuple(str(ln) for ln in lns)
        if best is None or key < best[0]:
            best = (key, tuple(lns))
    return best[1]


def _wick(cubic, quartic, resp, phys, canon, w, nu, phys_order) -> list:
    """Every one-loop diagram of ``Sigma`` (module docstring), merged per
    structure."""
    raw = []
    # first order, -c: one quartic vertex, two legs on the loop C0_pq(0)
    for c, legs, lab in quartic:
        for ia, ib in itertools.permutations(range(4), 2):
            rest = [legs[i] for i in range(4) if i not in (ia, ib)]
            if all(x in phys for x in rest):
                raw.append(SelfEnergyTerm('tadpole', legs[ia], legs[ib], -c,
                                          loop=canon(rest), vertices=(lab,)))
    # second order, (1/2) c1 c2 twice (either vertex can be the one at x_a)
    for c, legs, lab in cubic:
        for ia, ib in itertools.permutations(range(3), 2):
            k = legs[3 - ia - ib]
            if k in phys:
                # the third leg on the tadpole of the other vertex, through
                # G(0): -c times the one-loop mean shift <dx_k>
                raw.append(SelfEnergyTerm('mean_shift', legs[ia], legs[ib], -c,
                                          shift=k, vertices=(lab,)))
    for (c1, l1, lab1), (c2, l2, lab2) in itertools.product(cubic, repeat=2):
        for ia, ib in itertools.product(range(3), range(3)):
            r1 = [l1[i] for i in range(3) if i != ia]
            r2 = [l2[i] for i in range(3) if i != ib]
            for perm in ((0, 1), (1, 0)):
                lines = []   # (kind, row, col, head at the vertex of x_a)
                for u, v in ((r1[0], r2[perm[0]]), (r1[1], r2[perm[1]])):
                    if u in resp and v in resp:
                        break
                    if u in phys and v in phys:
                        lines.append(('C0', u, v, True))
                    elif u in phys:
                        lines.append(('G', u, v, True))
                    else:
                        lines.append(('G', v, u, False))
                else:
                    if (lines[0][0] == lines[1][0] == 'G'
                            and lines[0][3] != lines[1][3]):
                        continue   # a closed retarded loop
                    # omega flows from the vertex of x_b to that of x_a:
                    # nu through the first line, omega - nu through the second
                    flows = (nu, w - nu)
                    lns = _canonical_bubble(
                        [(k_, r_, c_, f if head else -f)
                         for (k_, r_, c_, head), f in zip(lines, flows)],
                        phys_order, w, nu)
                    raw.append(SelfEnergyTerm('bubble', l1[ia], l2[ib],
                                              c1 * c2, lines=lns,
                                              vertices=(lab1, lab2)))
    merged: dict = {}
    for t in raw:
        key = (t.kind, t.left, t.right, tuple(str(x) for x in t.lines),
               t.loop, t.shift)
        if key in merged:
            m = merged[key]
            verts = m.vertices + tuple(v for v in t.vertices
                                       if v not in m.vertices)
            merged[key] = SelfEnergyTerm(
                m.kind, m.left, m.right, m.coefficient + t.coefficient,
                m.lines, m.loop, m.shift, verts)
        else:
            merged[key] = t
    out = []
    for t in merged.values():
        c = SR(t.coefficient).simplify_rational()
        if not c.is_trivial_zero() and c != 0:
            out.append(SelfEnergyTerm(t.kind, t.left, t.right, c, t.lines,
                                      t.loop, t.shift, t.vertices))
    return out


def _check_loops(spec: OneLoopSpectrum) -> None:
    """Refuse loops without an Itô value (module docstring).  A line's decay
    order (:meth:`OneLoopSpectrum._decay`) is 0 when it has a ``delta(tau)``
    part and 1 when it jumps at the tie.  A tadpole loop at order 0 is a
    ``delta(0)``; a bubble whose two lines both have a ``delta(tau)`` part is
    a ``delta(tau)^2``.  Every other loop takes its Itô value."""
    for t in spec.terms:
        if t.kind == 'tadpole' and spec._decay('C0', *t.loop) == 0:
            raise UnsupportedModelError(
                f'the loop C0_{t.loop[0]},{t.loop[1]}(0) has a contact term '
                f'(a delta(tau) in C(tau)): no equal-time value')
        if t.kind == 'bubble' and all(
                spec._decay(ln.kind, ln.row, ln.col) == 0 for ln in t.lines):
            raise UnsupportedModelError(
                f'the bubble {t.lines[0]} {t.lines[1]} multiplies two '
                f'delta(tau) parts: delta(tau)^2 has no Itô value')


def one_loop_spectrum(model: dict, *, values: Optional[dict] = None,
                      mean_field: Any = None, parameters: Optional[dict] = None,
                      use_cache: bool = True,
                      verbose: bool = False) -> OneLoopSpectrum:
    """The one-loop (``k = 2``, ``ell = 1``) spectrum of ``model`` as its
    self-energy diagrams (:class:`OneLoopSpectrum`).  ``values``,
    ``mean_field`` and ``parameters`` as in
    :func:`~daedalus.api.symbolic_mean.mean_relation`; the tree covariance is
    built with ``values`` and the saddle substituted."""
    if not isinstance(model, dict) and callable(getattr(model, 'build', None)):
        model = model.build()
    lm = one_loop_mean(model, values=values, mean_field=mean_field,
                       parameters=parameters, use_cache=use_cache,
                       verbose=verbose)
    rel = lm.relation
    ft = _expanded_theory(model, 4, use_cache, verbose)
    resp, phys = set(rel.resp_names), set(rel.phys_names)
    cubic, quartic = _vertices(ft, resp, phys, lm._fix,
                               _forbidden_symbols(model))
    spec = OneLoopSpectrum(mean=lm, terms=[])
    canon = lambda legs: rel._canon(tuple(legs))
    spec.terms = _wick(cubic, quartic, resp, phys, canon, spec.omega, spec.nu,
                       list(rel.phys_names))
    _check_loops(spec)
    spec.notes = list(lm.notes)
    if any(t.left in phys and t.right in phys for t in spec.terms):
        spec.notes.append('WARNING: Sigma has a physical-physical entry, '
                          'which causality forbids')
    return spec
