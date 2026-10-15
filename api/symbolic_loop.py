r"""
Symbolic loop corrections as rational functions (stage C3)
==========================================================

On the state-space route (``engine/integration/time_domain/statespace.py``)
the stationary part of every Phase J subset integral is a composition of
linear solves: the states of the downsets of the subset's causal poset solve
``(-M_J) X_J = Σ_v F_v X_{J∖v}``, ``M_J`` a Kronecker sum of propagator
realizations and ``F_v`` contractions / openings by their ``c`` and ``b``
vectors.  With the parameters symbolic, therefore,

* the EQUAL-TIME cumulants (all external times equal; any k),
* the k = 2 pairings ``∫ L(τ) C(τ) dτ`` with exponential and polynomial ×
  exponential kernels (one more resolvent, ``q! c (z − βG)^{−(q+1)} x0``), and
* the k = 2 matrix-exponential triples ``(G, x0, c)`` of S4
  (``C(u) = c e^{G u} x0`` on each side of τ = 0)

are RATIONAL functions of the parameters (and of the saddle), at every loop
order: exact linear algebra over ``F = QQ(params)``, no polynomial roots, no
``RootOf``, no distinct-pole assumption (a double pole is an ordinary
companion matrix here).  Tested: OU quartic at one and two loops equals the
exact Boltzmann coefficients; colour-noise OU at the exact double pole; a
k = 3 one loop with three fields and five symbolic parameters.

Entry points: :func:`symbolic_cumulant` (equal time, or a pairing with
``kernel=``) -> :class:`SymbolicCumulant`; :func:`symbolic_triple` ->
:class:`SymbolicTriple`.  Also ``dd.symbolic_cumulant`` etc.

How
---
1. ``compute_cumulants`` runs once at a numeric REFERENCE point (the given
   parameters, completed by the model defaults): it supplies the diagrams,
   their symbolic prefactors (``result['diagrams'][...]['combined_prefactor']``,
   Sage expressions in the parameters and the saddle), the propagator data
   and the numbers the result is checked against.
2. ``G(s) = K(s)^{-1}`` (``s = iω``, ``K`` the pipeline's ``K_ft``) is
   inverted exactly over ``F(s)``; every entry is split ``G_pr = D_pr +
   r(s)/d(s)`` (``deg r < deg d``; a polynomial part of degree >= 1 is
   refused, ``improper_entry``) and realized by its controllable companion
   form (``d`` monic over ``F``): minimal, since ``r/d`` is reduced, and exact
   through repeated roots of ``d``.  ``D`` is the δ(t) coefficient (the
   pipeline's ``D_delta``).  This replaces S1's construction
   (``api._realization``: exact factorization of ``d`` into a cascade of
   rate / Schwarz blocks, Kalman reduction, balancing): those steps serve the
   floating-point conditioning of S2, are irrelevant in exact arithmetic,
   and factoring over ``QQ(params)`` is not needed; S1's exact conversion of
   inexact model numbers is reused in spirit (below).
3. Each diagram is integrated once more by ``integrate_diagram`` with a unit
   prefactor, only for its subsets: the state-space ``SubsetSS`` objects
   (poset, active edges, stationary downsets, event programs), their δ edges
   and the edge -> propagator-entry map.  The subset prefactor is the
   diagram's symbolic prefactor times the symbolic δ coefficients of its δ
   edges; the Wick permutations and the compensation are the contribution's
   own (``phase_j_parts``).
4. The stationary sweep and the event(s) are replayed with the symbolic
   realizations (numpy object tensors of field elements).  Every resolvent
   is one exact solve of a Kronecker-sum block by fraction-free (Bareiss)
   elimination over ``QQ[params]`` (:class:`_FFLU`: exact polynomial
   divisions, factored once per block and reused across downsets, subsets
   and diagrams; the right-hand side is replayed on a common denominator and
   divided once).  Sage's generic solve over the fraction field does not
   cancel common factors and swells beyond use already at 27 states.  At
   equal times the program is one event; on the k = 2 lag axis it is event,
   segment, event, and the segment's generator is built explicitly
   (diagonal blocks ``M_I``, firings below them).
5. Checks (never silent): each subset's symbolic δ factor, evaluated at the
   reference, equals the route's numeric prefactor; the symbolic δ
   coefficients equal ``D_delta`` and no symbolically nonzero entry vanishes
   at the reference (``degenerate_reference``: that run would have lost
   subsets); the total is compared with the pipeline at the reference
   (:meth:`SymbolicCumulant.check`, ``CHECK_RTOL``): equal time against
   ``phase_j_by_ell[ell]['total_C'](0, ..., 0)``, a pairing against S4's
   exact smooth ``Cumulant.integrate_against``.

Symbols, numbers and the saddle
-------------------------------
``symbols=None`` keeps every model parameter of the run symbolic (not the
saddle); a list keeps those, and the others enter as the SIMPLEST rational
that rounds to their double (``1/6`` for ``1/6.``; S1 takes the exact binary
value instead, whose 53-bit denominators make expressions unreadable; the
two differ by less than half an ulp).  Kernel coefficients and rates given
as floats are converted the same way; a kernel's own symbols join ``F``
(reference values in ``kernel_values=``).  The saddle (``xstar1`` ...):

* ``saddle='numeric'`` (default): the reference saddle as a rational (a
  value whose simplest form exceeds ``SADDLE_DIGITS`` digits is solver noise
  around a short rational -- ``1e-300`` for 0 -- and is snapped within
  ``SADDLE_ATOL``).  Exact for that saddle: the result is the symbolic
  function of the parameters only where the saddle does not move with them
  (``x* = 0``, or a saddle fixed by the model);
* ``saddle='symbol'``: the saddle symbols join ``F``; the result is rational
  in parameters and saddle and holds on the mean-field manifold;
* ``saddle={name: closed form}``: substituted, after a check against the
  reference saddle (``saddle_mismatch``).  ``'solve'`` is refused
  (``saddle_unsolved``): the pipeline's expanded action drops the stationary
  mean-field sector, so no symbolic equations are at hand here.

Conventions
-----------
* Equal time: the external times coincide; constant rows are decided by
  Phase J's own Θ(0) / tie rule (Itô).  For k = 2 this is the left limit
  ``C(0⁻)`` (``SymbolicTriple.at_zero(ell, '-')``; ``C_tau``'s τ = 0 sample
  is taken at ``τ = −_ITO_EPS`` instead).
* The SMOOTH part only: δ records of Feature B (shot-noise subsets on a δ
  surface) are not included, as in ``C_tau``; δ parts of the propagator
  inside a subset (its δ edges) are.
* Pairing (k = 2): ``∫ L(s) C(s) ds`` with ``C(τ) = κ(x_1(0), x_2(τ))`` and a
  C2 :class:`api.symbolic_out.Kernel` (terms ``coeff · s^q e^{rate s}`` on
  side 'positive' / 'negative'), no contact term (S4's
  ``contact='exclude'``).
* τ != 0: ``C(τ)`` itself needs ``e^{G τ}`` symbolically -- eigen data
  (``RootOf``, as C2) or the ``expm`` of an explicit rational matrix -- and
  is not built: :class:`SymbolicTriple` holds the rational triple (its
  ``laplace`` is rational, its ``C`` evaluates at numbers).

Limits and refusals (``SymbolicLoopRefused.reason``)
---------------------------------------------------
``too_large``: a Kronecker block over ``max_block`` states (default
``MAX_BLOCK``; the block of ``w`` active edges with ``n`` states each has
``n^w``) or more than ``max_symbols`` variables; ``spatial``;
``k_unsupported`` (a pairing or triple at k != 2, a switch-on result);
``complex_coefficients``; ``non_rational``; ``improper_entry``;
``singular_pencil``; ``pole_route_subset`` (a subset the state-space route
declined); ``kernel_edge`` (conductance kernels: kernel rows, dressed lines,
kernel symbols in a vertex) / ``dressed_line``; ``noise_kernel`` (a
non-local noise cumulant); ``offset_row`` / ``offset_slot`` / ``ext_ext_row``
(rows with constant time offsets or joining two external times on the lag
axis); ``degenerate_reference``; ``saddle_mismatch`` / ``saddle_unsolved``;
``prefactor_mismatch``; ``unbound_symbol``; ``failed_diagram``.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

import numpy as np

__all__ = ['SymbolicLoopRefused', 'SymbolicPropagator', 'SymbolicCumulant',
           'SymbolicTriple', 'symbolic_triple',
           'symbolic_propagator', 'symbolic_cumulant', 'expression_size',
           'MAX_BLOCK', 'MAX_SYMBOLS']

#: largest Kronecker-sum block (states) solved exactly
MAX_BLOCK = 64
#: most parameters kept symbolic at once
MAX_SYMBOLS = 12
#: relative tolerance of the reference checks (prefactors, totals)
CHECK_RTOL = 1e-12


class SymbolicLoopRefused(ValueError):
    """The symbolic route does not cover this model / request
    (``reason``: a short name, see the module docstring)."""

    def __init__(self, reason, message):
        super().__init__(f'{reason}: {message}')
        self.reason = reason


def _sage():
    import sage.all as S
    return S


# ─── the field ──────────────────────────────────────────────────────────────

class _Field:
    """``F = Frac(QQ[names])`` (``QQ`` when no name is symbolic) and the
    conversions SR <-> F."""

    def __init__(self, names):
        S = _sage()
        self.names = tuple(names)
        if self.names:
            # always multivariate (one name would give a univariate ring
            # with a different API)
            self.R = S.PolynomialRing(S.QQ, len(self.names), list(self.names))
            self.F = self.R.fraction_field()
        else:
            self.R = self.F = S.QQ
        self.sr_vars = {n: S.SR.var(n) for n in self.names}

    def from_sr(self, e, what='expression'):
        S = _sage()
        e = _exactify_simple(S.SR(e))
        free = sorted({str(v) for v in e.variables()} - set(self.names))
        if free:
            raise SymbolicLoopRefused(
                'unbound_symbol', f'{what} {str(e)[:120]} has symbols {free} '
                'that are neither symbolic nor given a value')
        if e.is_trivial_zero():
            return self.F(0)
        if not self.names:
            try:
                return S.QQ(e)
            except (TypeError, ValueError) as exc:
                raise SymbolicLoopRefused(
                    'non_rational', f'{what} {str(e)[:120]} is not a rational '
                    f'number ({exc})') from exc
        if e.has(S.I):
            raise SymbolicLoopRefused('complex_coefficients',
                                      f'{what} {str(e)[:120]} is complex')
        try:
            num, den = e.numerator(), e.denominator()
            return self.F(self.R(num.polynomial(ring=self.R))) / \
                self.F(self.R(den.polynomial(ring=self.R)))
        except Exception as exc:
            raise SymbolicLoopRefused(
                'non_rational', f'{what} {str(e)[:120]} is not a rational '
                f'function of {list(self.names)} ({type(exc).__name__}: '
                f'{str(exc)[:100]})') from exc

    def to_sr(self, x):
        S = _sage()
        if not self.names:
            return S.SR(x)
        return _poly_to_sr(x.numerator(), self) / _poly_to_sr(
            x.denominator(), self)

    def evaluate(self, x, point):
        """Complex value of ``x`` at ``point`` ({name: number})."""
        S = _sage()
        if not self.names:
            return complex(S.CDF(x))
        vals = [S.QQ(_exact(point[n])) for n in self.names]
        num = x.numerator()(*vals)
        den = x.denominator()(*vals)
        return complex(S.CDF(num / den))


#: a saddle value whose simplest rational (same double) has more digits than
#: this is solver noise around a short rational (e.g. 1e-300 for 0): it is
#: snapped to the simplest rational within SADDLE_ATOL * max(1, |v|)
SADDLE_DIGITS = 40
SADDLE_ATOL = 1e-13


def _rational_value(v, is_saddle=False):
    """The exact rational a numeric value enters ``F`` as: the simplest
    rational that rounds to the same double (``1/6`` for ``1/6.``, a short
    fraction for ``2*pi/48``); a saddle value whose form is longer than
    ``SADDLE_DIGITS`` digits is snapped within ``SADDLE_ATOL`` (module
    docstring)."""
    S = _sage()
    from sage.rings.rational import Rational
    from sage.rings.integer import Integer
    if isinstance(v, (int, Integer, Rational)) and not isinstance(v, bool):
        return S.QQ(v)
    try:
        sv = S.SR(v)
        if sv.is_numeric() and sv.pyobject().parent().is_exact():
            return S.QQ(sv.pyobject())
    except (TypeError, ValueError, AttributeError):
        pass
    c = complex(v)
    if c.imag != 0.0:
        raise SymbolicLoopRefused('complex_coefficients',
                                  f'complex parameter value {v!r}')
    x = float(c.real)
    q = S.RealField(53)(x).simplest_rational()
    if is_saddle and len(str(max(abs(q.numerator()), q.denominator()))) \
            > SADDLE_DIGITS:
        tol = SADDLE_ATOL * max(1.0, abs(x))
        q = S.RIF(x - tol, x + tol).simplest_rational()
    return q


def _exactify_simple(e):
    """Every inexact number of an SR expression replaced by
    :func:`_rational_value` (a kernel rate ``-0.5`` is ``-1/2``)."""
    S = _sage()
    if e.is_numeric():
        po = e.pyobject()
        par = getattr(po, 'parent', None)
        if isinstance(po, int) or (par is not None and par().is_exact()):
            return e
        return S.SR(_rational_value(complex(po)))
    if not e.variables() and not e.is_numeric():
        try:
            if not e.operands():
                return e
        except Exception:
            return e
    ops = e.operands()
    if not ops:
        return e
    return e.operator()(*[_exactify_simple(o) for o in ops])


def _red(x):
    """``x`` in lowest terms with a monic denominator (Sage's fraction field
    of a multivariate ring neither cancels common factors nor normalises
    the constant content by itself; the sizes reported are of this form)."""
    par = getattr(x, 'parent', None)
    if par is None:
        return x
    F = par()
    if not hasattr(x, 'denominator') or not hasattr(F, 'ring') or \
            not hasattr(x.numerator(), 'gcd'):
        return x
    num, den = x.numerator(), x.denominator()
    if num == 0:
        return F(0)
    g = num.gcd(den)
    if g != 1:
        num, den = num // g, den // g
    # integer coefficients without a common factor, positive leading
    # coefficient of the denominator
    from sage.all import lcm, gcd, QQ
    cs = list(num.coefficients()) + list(den.coefficients())
    mult = QQ(lcm([QQ(c).denominator() for c in cs]))
    mult = mult / QQ(gcd([QQ(c) * mult for c in cs]))
    if den.lc() < 0:
        mult = -mult
    if mult != 1:
        num, den = num * mult, den * mult
    from sage.rings.fraction_field_element import FractionFieldElement
    R = F.ring()
    return FractionFieldElement(F, R(num), R(den), coerce=False, reduce=False)


def _exact(v):
    from api._realization import _exact_number
    return _exact_number(v, allow_complex=False)


def _poly_to_sr(p, fld):
    S = _sage()
    if not fld.names:
        return S.SR(p)
    gens = [fld.sr_vars[n] for n in fld.names]
    out = S.SR(0)
    for c, m in zip(p.coefficients(), p.monomials()):
        exps = m.exponents()[0] if hasattr(m, 'exponents') else ()
        t = S.SR(c)
        for g, e in zip(gens, exps):
            if e:
                t = t * g ** e
        out = out + t
    return out


def expression_size(x) -> dict:
    """Size of a field element ``x``: terms and total degree of its numerator
    and denominator (``{'terms', 'degree', 'digits'}``; ``digits``: the
    largest coefficient's decimal digits)."""
    out = {'terms': 0, 'degree': 0, 'digits': 0}
    for p in (x.numerator(), x.denominator()):
        try:
            cs = p.coefficients()
            out['terms'] += len(cs)
            out['degree'] = max(out['degree'], int(p.total_degree())
                                if hasattr(p, 'total_degree') else 0)
            for c in cs:
                n = max(abs(c.numerator()), abs(c.denominator()))
                out['digits'] = max(out['digits'], len(str(n)))
        except AttributeError:          # a rational number
            out['terms'] += 1
            n = max(abs(p.numerator()), abs(p.denominator()))
            out['digits'] = max(out['digits'], len(str(n)))
    out = {k_: int(v) for k_, v in out.items()}
    return out


# ─── the symbolic propagator ────────────────────────────────────────────────

@dataclass
class _SymEntry:
    """``g(t) = c e^{a t} b`` (t > 0) over ``F`` (``n`` states, companion
    form) plus the δ coefficient ``d``."""
    a: Any
    b: Any
    c: Any
    d: Any
    n: int
    num: Any = None      # r(s) and d(s) of the smooth part (for display)
    den: Any = None


@dataclass
class SymbolicPropagator:
    """``G(s) = K(s)^{-1}`` over ``F(s)`` and its per-entry realizations.

    ``G[p, r]``: ``p`` a physical field, ``r`` a response field (the
    pipeline's ``(pi, ri)``); ``entries[(p, r)]`` the companion realization
    of the smooth part, ``D[p, r]`` the δ(t) coefficient."""
    G: Any
    entries: dict
    nf: int
    field: Any
    s: Any

    def D(self, p, r):
        return self.entries[(p, r)].d

    def transfer(self, p, r, s_value, point):
        """``G[p, r](s)`` at a complex ``s`` and parameter ``point``
        (numeric; for checks)."""
        S = _sage()
        e = self.entries[(p, r)]
        out = self.field.evaluate(e.d, point) if e.d != 0 else 0j
        if e.n:
            A = np.array([[self.field.evaluate(e.a[i, j], point)
                           for j in range(e.n)] for i in range(e.n)])
            bb = np.array([self.field.evaluate(x, point) for x in e.b])
            cc = np.array([self.field.evaluate(x, point) for x in e.c])
            out += cc @ np.linalg.solve(s_value * np.eye(e.n) - A, bb)
        return out


def symbolic_propagator(K_ft, omega, fld: _Field, subs: dict):
    """Exact ``G = K^{-1}`` over ``F(s)`` with the companion realization of
    every entry (module docstring, step 2).  ``subs``: {SR symbol: exact SR
    number} for the parameters that are not symbolic."""
    S = _sage()
    from api._realization import _exactify, _spatial_symbol
    nf = K_ft.nrows()
    if K_ft.ncols() != nf:
        raise SymbolicLoopRefused('singular_pencil', 'K_ft is not square')
    s = S.SR.var('s_c3')
    names = list(fld.names) + ['s_c3']
    Rs = S.PolynomialRing(S.QQ, len(names), names)
    Ls = Rs.fraction_field()
    s_gen = Rs.gen(len(names) - 1)
    fs = _Field(names)
    K = S.matrix(Ls, nf, nf)
    for i in range(nf):
        for j in range(nf):
            e = S.SR(K_ft[i, j])
            if e.is_trivial_zero():
                continue
            e = _exactify(e.subs(subs)).subs({S.SR(omega): -S.I * s})
            e = e.expand()
            if e.has(S.I):
                raise SymbolicLoopRefused(
                    'complex_coefficients', f'K[{i},{j}] = {str(K_ft[i, j])[:100]} '
                    'has complex coefficients in s = i omega')
            free = sorted({str(v) for v in e.variables()} - set(names))
            if any(_spatial_symbol(f) for f in free):
                raise SymbolicLoopRefused('spatial', f'K[{i},{j}] carries {free}')
            K[i, j] = _red(fs.from_sr(e, f'K[{i},{j}]'))
    try:
        det = K.det()
    except Exception as exc:
        raise SymbolicLoopRefused('non_rational', f'det K failed: {exc}') from exc
    if det == 0:
        raise SymbolicLoopRefused('singular_pencil', 'det K(s) == 0')
    G = K.inverse()
    # F[s] over F = Frac(QQ[params])
    F = fld.F
    Ps = S.PolynomialRing(F, 'sv')
    sv = Ps.gen()
    to_F = _ring_map(Rs, fld, Ps, sv)
    entries = {}
    for p in range(nf):
        for r in range(nf):
            g = G[p, r]
            num, den = to_F(g.numerator()), to_F(g.denominator())
            lc = den.leading_coefficient()
            num, den = num / lc, den / lc
            q, rem = num.quo_rem(den)
            if q.degree() >= 1:
                raise SymbolicLoopRefused(
                    'improper_entry', f'G[{p},{r}] has a polynomial part of '
                    f'degree {q.degree()} in s (a derivative of δ(t))')
            dcoef = q[0] if q != 0 else F(0)
            n = int(den.degree()) if rem != 0 else 0
            if n == 0:
                A = S.matrix(F, 0, 0)
                b = c = S.vector(F, 0)
            else:
                A = S.matrix(F, n, n)
                for i in range(n - 1):
                    A[i, i + 1] = 1
                dl = den.list()
                for j in range(n):
                    A[n - 1, j] = -dl[j]
                b = S.vector(F, [0] * (n - 1) + [1])
                rl = rem.list() + [0] * n
                c = S.vector(F, rl[:n])
            entries[(p, r)] = _SymEntry(A, b, c, dcoef, n, rem, den)
    return SymbolicPropagator(G, entries, nf, fld, sv)


def _ring_map(Rs, fld, Ps, sv):
    """QQ[params, s] -> F[s]."""
    S = _sage()
    F = fld.F
    k = len(fld.names)
    imgs = ([F(fld.R.gen(i)) for i in range(k)] if k else []) + [sv]

    def f(p):
        if p == 0:
            return Ps(0)
        out = Ps(0)
        for c, m in zip(p.coefficients(), p.monomials()):
            ex = m.exponents()[0]
            t = Ps(F(c))
            for g, e in zip(imgs, ex):
                if e:
                    t = t * g ** e
            out += t
        return out
    return f


# ─── symbolic tensors ───────────────────────────────────────────────────────

def _obj(x):
    a = np.empty((), dtype=object)
    a[()] = x
    return a


def _fire(x, act_in, ends, starts, act_out, ents):
    """Contract the edges ``ends`` by ``c_e``, open ``starts`` by ``b_e``
    (``ents[e]``: the edge's ``_SymEntry``); axes ``act_in`` -> ``act_out``."""
    axes = list(act_in)
    for e in ends:
        k = axes.index(e)
        cvec = np.array(list(ents[e].c), dtype=object)
        x = np.tensordot(x, cvec, axes=([k], [0]))
        axes.pop(k)
    for e in starts:
        bvec = np.array(list(ents[e].b), dtype=object)
        x = np.multiply.outer(x, bvec)
        axes.append(e)
    if sorted(axes) != sorted(act_out):
        raise RuntimeError(f'firing axes {axes} != {list(act_out)}')
    perm = [axes.index(e) for e in act_out]
    return np.transpose(x, perm) if perm else x


def _kron_sum(mats, F):
    """Dense Kronecker sum ``⊕ a_e`` (C order: the first axis is the most
    significant), over ``F``."""
    S = _sage()
    if not mats:
        return S.matrix(F, 1, 1)
    out = None
    dims = [m.nrows() for m in mats]
    n = int(np.prod(dims))
    out = S.matrix(F, n, n)
    for k, m in enumerate(mats):
        left = int(np.prod(dims[:k])) if k else 1
        right = int(np.prod(dims[k + 1:])) if k + 1 < len(dims) else 1
        term = S.identity_matrix(F, left).tensor_product(m).tensor_product(
            S.identity_matrix(F, right))
        out += term
    return out


class _Stats:
    def __init__(self, cap=None):
        self.cap = MAX_BLOCK if cap is None else int(cap)
        self.lu = {}
        self.kron = {}
        self.solves = 0
        self.delta_subsets = 0
        self.max_block = 0
        self.t_solve = 0.0


def _to_ring(x, R):
    """A field element with denominator 1 as an element of ``R``."""
    x = _red(x)
    d = x.denominator()
    if d != 1:
        raise RuntimeError('row scaling left a denominator')
    return R(x.numerator())


class _FFLU:
    """Fraction-free (Bareiss) LU of a square matrix over ``F = Frac(R)``,
    replayable on any right-hand side.

    The rows are scaled by the lcm of their denominators (polynomial rows),
    then eliminated with row pivoting (fewest terms); every update
    ``(p_k a_ij − a_ik a_kj) / p_{k−1}`` is an exact division in ``R`` (no
    fraction swell).  :meth:`solve` replays the same operations on a
    right-hand side brought to a common denominator (exact in ``R``), back
    substitutes fraction-free (``x̂ = det · x`` are the Cramer numerators,
    polynomials) and divides once.  The factorization depends on the matrix
    only, so a Kronecker block shared by many downsets and subsets is
    factored once."""

    def __init__(self, M, F):
        R = F.ring()
        n = M.nrows()
        self.F, self.R, self.n = F, R, n
        A, self.scale = [], []
        for i in range(n):
            row = [F(x) for x in M.row(i)]
            L = R(1)
            for x in row:
                if x != 0:
                    L = L.lcm(_red(x).denominator())
            self.scale.append(F(L))
            A.append([_to_ring(x * L, R) if x != 0 else R(0) for x in row])
        prev = R(1)
        self.steps = []                 # (piv, p_k, [(i, a_ik)], prev)
        for k in range(n):
            piv, best = None, None
            for i in range(k, n):
                a = A[i][k]
                if a != 0:
                    sz = len(a.coefficients())
                    if best is None or sz < best:
                        piv, best = i, sz
            if piv is None:
                raise SymbolicLoopRefused(
                    'singular_pencil', 'a singular resolvent block (a '
                    'non-Hurwitz realization?)')
            if piv != k:
                A[k], A[piv] = A[piv], A[k]
            p = A[k][k]
            rowk = A[k]
            mult = []
            for i in range(k + 1, n):
                ri = A[i]
                aik = ri[k]
                mult.append((i, aik))
                for j in range(k + 1, n):
                    v = ri[j] * p if ri[j] != 0 else ri[j]
                    if aik != 0 and rowk[j] != 0:
                        v = v - aik * rowk[j]
                    if v != 0 and prev != 1:
                        v = v // prev
                    ri[j] = v
                ri[k] = R(0)
            self.steps.append((piv, p, mult, prev))
            prev = p
        self.U = A
        self.det = A[n - 1][n - 1]

    def solve(self, rhs):
        F, R, n = self.F, self.R, self.n
        b = [F(x) * sc if x != 0 else F(0) for x, sc in zip(rhs, self.scale)]
        if all(x == 0 for x in b):
            return [F(0)] * n
        D = R(1)
        for x in b:
            if x != 0:
                D = D.lcm(_red(x).denominator())
        bn = [_to_ring(x * D, R) if x != 0 else R(0) for x in b]
        for k, (piv, p, mult, prev) in enumerate(self.steps):
            if piv != k:
                bn[k], bn[piv] = bn[piv], bn[k]
            bk = bn[k]
            for i, aik in mult:
                v = bn[i] * p if bn[i] != 0 else bn[i]
                if aik != 0 and bk != 0:
                    v = v - aik * bk
                if v != 0 and prev != 1:
                    v = v // prev
                bn[i] = v
        U, det = self.U, self.det
        xh = [R(0)] * n
        for i in range(n - 1, -1, -1):
            acc = det * bn[i] if bn[i] != 0 else R(0)
            for j in range(i + 1, n):
                if U[i][j] != 0 and xh[j] != 0:
                    acc = acc - U[i][j] * xh[j]
            if acc != 0:
                q, r = acc.quo_rem(U[i][i])
                if r != 0:
                    raise RuntimeError('fraction-free back substitution: '
                                       'inexact division')
                acc = q
            xh[i] = acc
        den = F(det) * F(D)
        return [_red(F(v) / den) if v != 0 else F(0) for v in xh]


def _solve(M, r, F, stats, key=None):
    """``M^{-1} r`` for a numpy object tensor ``r`` (flattened C order):
    QQ -> Sage's (flint) solve; QQ(params) -> :class:`_FFLU` (cached in
    ``stats.lu`` under ``key``)."""
    S = _sage()
    n = M.nrows()
    if n > stats.cap:
        raise SymbolicLoopRefused('too_large', f'a Kronecker block of {n} '
                                  f'states (max_block = {stats.cap})')
    stats.solves += 1
    stats.max_block = max(stats.max_block, n)
    t0 = time.perf_counter()
    shape = r.shape
    flat = [F(v) for v in r.reshape(-1)]
    if n == 1:
        out = [_red(flat[0] / M[0, 0])]
    elif F is S.QQ:
        out = list(M.solve_right(S.vector(F, flat)))
    else:
        lu = stats.lu.get(key) if key is not None else None
        if lu is None:
            lu = _FFLU(M, F)
            if key is not None:
                stats.lu[key] = lu
        out = lu.solve(flat)
    stats.t_solve += time.perf_counter() - t0
    y = np.empty(len(out), dtype=object)
    for i, v in enumerate(out):
        y[i] = v
    return y.reshape(shape)


# ─── one subset ─────────────────────────────────────────────────────────────

class _SymSubset:
    """The symbolic replay of one ``SubsetSS`` (module docstring, step 4)."""

    def __init__(self, sub, edge_entries, const_entries, F, stats):
        self.sub = sub
        self.ents = edge_entries          # edge id -> _SymEntry
        self.cents = const_entries        # const-row idx -> _SymEntry
        self.F = F
        self.stats = stats
        self._stat = None

    def _key(self, act):
        """A run-wide key of a Kronecker block: the identities of its
        entries (shared by every subset of the run)."""
        return tuple(id(self.ents[e]) for e in act)

    def _M(self, act):
        key = self._key(act)
        M = self.stats.kron.get(key)
        if M is None:
            M = _kron_sum([self.ents[e].a for e in act], self.F)
            self.stats.kron[key] = M
        return M

    def _dims(self, act):
        return tuple(self.ents[e].n for e in act)

    def stationary(self):
        if self._stat is not None:
            return self._stat
        sub = self.sub
        x = {frozenset(): _obj(self.F(1))}
        acts = {frozenset(): ()}
        for J in sub._stat_downsets:
            if not J:
                continue
            actJ = sub._act(J, None)
            acts[J] = actJ
            r = None
            for v in sub._maximal(J):
                I = J - {v}
                if I not in x:
                    continue
                term = _fire(x[I], acts[I], sub.in_edges[v], sub.out_edges[v],
                             actJ, self.ents)
                r = term if r is None else r + term
            if r is None:
                continue
            x[J] = _solve(-self._M(actJ), r, self.F, self.stats,
                          key=('neg',) + self._key(actJ))
        self._stat = x
        return x

    def scalar(self, free_vals, tie_ctx, FI):
        """Product of the constant rows (``None``: a row is EMPTY)."""
        out = self.F(1)
        for (idx, a_int, a_ext, c0, _ent) in self.sub.const_rows:
            v = FI._const_row_verdict(a_int, a_ext, c0, free_vals, 'edge',
                                      tie_ctx)
            if v == 'EMPTY':
                return None
            c_eff = float(c0) + sum(float(a) * float(f)
                                    for a, f in zip(a_ext, free_vals))
            if c_eff != 0.0:
                raise SymbolicLoopRefused(
                    'offset_row', f'a constant row with Δt = {c_eff} != 0 '
                    '(needs e^{a Δt} symbolically)')
            e = self.cents[idx]
            out *= (e.c * e.b) if e.n else self.F(0)
        return out

    def equal_time(self, perm_free_vals, tie_ctx, FI):
        sub = self.sub
        if sub.empty:
            return self.F(0)
        scal = self.scalar(perm_free_vals, tie_ctx, FI)
        if scal is None or scal == 0:
            return self.F(0)
        if sub.m == 0:
            return scal
        vals = sub._slot_values(perm_free_vals)
        pattern, _gv = sub._pattern(vals)
        if len(pattern) != 1:
            raise RuntimeError('equal times give more than one event')
        prog = sub._program(pattern, False)
        x = self.stationary()
        for kind, data in prog:
            if kind == 'event':
                nx = {}
                for I, f in data:
                    if I in x:
                        nx[I] = _fire(x[I], f.act_in, f.ends, f.starts,
                                      f.out, self.ents)
                x = nx
            elif kind == 'empty':
                return self.F(0)
            else:
                raise RuntimeError(f'unexpected step {kind} at equal times')
        v = x.get(sub.full)
        return self.F(0) if v is None else scal * self.F(v[()])

    # ── k = 2 pairing: event, segment, event ────────────────────────────
    def ray(self, free_base, free_dir, tie_ctx, FI):
        """``(beta, blocks, G, x0, cfun)`` on the ray ``f = free_base +
        u * free_dir`` (u > 0), or ``None`` (zero).  The value is
        ``scal * cfun(exp(β u G) x0)``."""
        sub = self.sub
        if sub.empty:
            return None
        for (idx, a_int, a_ext, c0, _e) in sub.const_rows:
            if any(float(a) * float(d) != 0.0 for a, d in zip(a_ext, free_dir)):
                raise SymbolicLoopRefused(
                    'ext_ext_row', 'a row joining two external times depends '
                    'on the lag (not covered by the symbolic pairing)')
        fv1 = [b + d for b, d in zip(free_base, free_dir)]
        scal = self.scalar(fv1, tie_ctx, FI)
        if scal is None or scal == 0:
            return None
        if sub.m == 0:
            raise SymbolicLoopRefused('ext_ext_row', 'a subset without '
                                      'internal times on the lag axis')
        v1 = sub._slot_values(fv1)
        v2 = sub._slot_values([b + 2 * d for b, d in zip(free_base, free_dir)])
        v0 = sub._slot_values(list(free_base))
        for a, b_, c in zip(v0, v1, v2):
            if abs((c - b_) - (b_ - a)) > 1e-12 or a != 0.0:
                raise SymbolicLoopRefused('offset_slot', 'an external time '
                                          'with a constant offset')
        pattern, gv = sub._pattern(v1)
        if len(pattern) == 1:
            raise RuntimeError('no lag on this ray')
        if len(pattern) != 2:
            raise SymbolicLoopRefused('k_unsupported', 'more than two events')
        beta = gv[1] - gv[0]
        prog = sub._program(pattern, False)
        x = self.stationary()
        steps = list(prog)
        if steps and steps[-1][0] == 'empty':
            return None
        kinds = [k for k, _ in steps]
        if kinds != ['event', 'seg', 'event']:
            raise SymbolicLoopRefused('k_unsupported', f'program {kinds}')
        ev0, seg, ev1 = steps[0][1], steps[1][1], steps[2][1]
        nx = {}
        for I, f in ev0:
            if I in x:
                nx[I] = _fire(x[I], f.act_in, f.ends, f.starts, f.out,
                              self.ents)
        # the segment's generator over F
        blocks, acts = seg.blocks, seg.acts
        dims = [int(np.prod(self._dims(a))) if a else 1 for a in acts]
        offs = np.cumsum([0] + dims)
        n = int(offs[-1])
        if n > 4 * self.stats.cap:
            raise SymbolicLoopRefused('too_large', f'a segment of {n} states '
                                      f'(4 max_block = {4 * self.stats.cap})')
        S = _sage()
        G = S.matrix(self.F, n, n)
        for k_, act in enumerate(acts):
            Mk = self._M(act)
            G.set_block(int(offs[k_]), int(offs[k_]), Mk)
        for kI, kJ, f in seg.trans:
            shapeI = self._dims(acts[kI])
            for col in range(dims[kI]):
                unit = np.zeros(dims[kI], dtype=object)
                for i in range(dims[kI]):
                    unit[i] = self.F(0)
                unit[col] = self.F(1)
                y = _fire(unit.reshape(shapeI) if shapeI else _obj(self.F(1)),
                          f.act_in, f.ends, f.starts, f.out, self.ents)
                yf = y.reshape(-1)
                for row in range(dims[kJ]):
                    G[int(offs[kJ]) + row, int(offs[kI]) + col] += yf[row]
        x0 = [self.F(0)] * n
        for k_, I in enumerate(blocks):
            if I in nx:
                for i, v in enumerate(nx[I].reshape(-1)):
                    x0[int(offs[k_]) + i] = self.F(v)
        cvec = [self.F(0)] * n
        for I, f in ev1:
            # the value is the full downset's scalar after the last event
            k_ = seg.index.get(I)
            if k_ is None or I != sub.full or f.out:
                continue
            shapeI = self._dims(acts[k_])
            for col in range(dims[k_]):
                unit = np.zeros(dims[k_], dtype=object)
                for i in range(dims[k_]):
                    unit[i] = self.F(0)
                unit[col] = self.F(1)
                y = _fire(unit.reshape(shapeI) if shapeI else _obj(self.F(1)),
                          f.act_in, f.ends, f.starts, f.out, self.ents)
                cvec[int(offs[k_]) + col] = self.F(y[()])
        return {'beta': beta, 'G': G, 'x0': S.vector(self.F, x0),
                'c': S.vector(self.F, cvec), 'scal': scal}


# ─── results ────────────────────────────────────────────────────────────────

@dataclass
class SymbolicCumulant:
    """Result of :func:`symbolic_cumulant`.

    ``kind``: 'equal_time' or 'pairing'; ``by_ell``: {ell: field element};
    ``total``: their sum; ``field``: the field (``names`` = its variables);
    ``reference``: {name: value} of the reference point; ``pipeline``:
    {ell: the pipeline's value there} (equal time: ``phase_j_by_ell[ell]
    ['total_C'](0, ..., 0)``; pairing: S4's exact smooth
    ``Cumulant.integrate_against``; empty when unavailable); ``stats``:
    sizes and timings; ``notes``: the decisions taken (saddle, numeric
    parameters)."""
    kind: str
    by_ell: dict
    field: Any
    reference: dict
    pipeline: dict
    k: int
    stats: dict
    notes: list
    _ctx: Any = None

    @property
    def names(self):
        return self.field.names

    @property
    def total(self):
        out = self.field.F(0)
        for v in self.by_ell.values():
            out += v
        return _red(out)

    def _x(self, ell):
        return self.total if ell is None else self.by_ell[ell]

    def expression(self, ell=None):
        """Sage ``SR`` expression (``ell=None``: the sum over loop orders)."""
        return self.field.to_sr(self._x(ell))

    def numerator(self, ell=None):
        return self._x(ell).numerator()

    def denominator(self, ell=None):
        return self._x(ell).denominator()

    def size(self, ell=None):
        """``expression_size`` of the (reduced) rational function."""
        return expression_size(self._x(ell))

    def sympy(self, ell=None):
        from api.symbolic_out import to_sympy
        return to_sympy(self.expression(ell))

    def latex(self, ell=None):
        from api.symbolic_out import to_latex
        return to_latex(self.expression(ell))

    def to_json(self, ell=None):
        """JSON string ``{"params": [...], "numerator": str, "denominator":
        str}`` (Sage polynomial syntax)."""
        import json
        x = self._x(ell)
        return json.dumps({'params': list(self.names),
                           'numerator': str(x.numerator()),
                           'denominator': str(x.denominator())})

    def callable(self, ell=None):
        """numpy callable ``f(**params)`` of the symbolic variables, through
        Horner forms of numerator and denominator (``sympy.lambdify``)."""
        import sympy
        ex = self.sympy(ell)
        syms = [sympy.Symbol(n) for n in self.names]
        by = {str(s): s for s in ex.free_symbols}
        fn = sympy.lambdify([by.get(n, s) for n, s in zip(self.names, syms)],
                            ex, modules='numpy')
        names = list(self.names)

        def f(**kw):
            missing = [n for n in names if n not in kw]
            extra = [n for n in kw if n not in names]
            if missing or extra:
                raise TypeError(f'parameters {names} expected; missing '
                                f'{missing}, unknown {extra}')
            return fn(*[kw[n] for n in names])
        f.params = names
        return f

    def evaluate(self, point=None, ell=None):
        """Value at ``point`` (default: the reference point), exact
        rational arithmetic then rounded."""
        pt = dict(self.reference)
        pt.update(point or {})
        return self.field.evaluate(self._x(ell), pt)

    def check(self, rtol=CHECK_RTOL):
        """``{ell: |symbolic − pipeline| / scale}`` at the reference point
        (scale: the largest |pipeline value|); raises ``AssertionError``
        beyond ``rtol``."""
        out = {}
        scale = max([abs(v) for v in self.pipeline.values()] + [1e-300])
        for ell, ref in self.pipeline.items():
            v = self.evaluate(ell=ell)
            d = abs(v - ref) / scale
            out[ell] = d
            if d > rtol:
                raise AssertionError(f'{self.kind} ell={ell}: symbolic {v} vs '
                                     f'pipeline {ref} ({d:.2e})')
        return out

    def __repr__(self):
        return (f'SymbolicCumulant({self.kind}, k={self.k}, ells='
                f'{sorted(self.by_ell)}, variables={list(self.names)})')


@dataclass
class SymbolicTriple:
    """Result of :func:`symbolic_triple` (k = 2): per loop order and side
    ('+': τ > 0, '-': τ < 0) a list of rational blocks ``(G, x0, c)`` with
    ``C_side(u) = Σ_blocks c e^{G u} x0`` (``u = |τ|``; weights and the
    segment slope folded in), the matrix-exponential triple of S4 over
    ``F``.  ``C(τ)`` itself needs ``e^{G u}``: :meth:`C` evaluates it at
    numbers (``scipy.linalg.expm`` of the evaluated blocks); symbolically it
    needs the eigen data of ``G`` (``RootOf`` as C2) or the ``expm`` of an
    explicit rational matrix, not built here.  :meth:`laplace` gives the
    rational ``Σ c (z − G)^{−(q+1)} x0`` (``z`` a number or a field
    element)."""
    blocks: dict
    field: Any
    reference: dict
    stats: dict
    notes: list

    @property
    def names(self):
        return self.field.names

    def n_states(self, ell, side):
        return sum(G.nrows() for G, _x, _c in self.blocks[ell][side])

    def dense(self, ell, side):
        """The block-diagonal triple ``(G, x0, c)`` over ``F`` (Sage)."""
        S = _sage()
        F = self.field.F
        bl = self.blocks[ell][side]
        if not bl:
            return S.matrix(F, 0, 0), S.vector(F, 0), S.vector(F, 0)
        G = S.block_diagonal_matrix([b[0] for b in bl])
        x0 = S.vector(F, [v for b in bl for v in b[1]])
        c = S.vector(F, [v for b in bl for v in b[2]])
        return G, x0, c

    def at_zero(self, ell, side='-'):
        """``Σ c x0``: ``C(0⁻)`` (side '-') or ``C(0⁺)``, a field element."""
        F = self.field.F
        out = F(0)
        for _G, x0, c in self.blocks[ell][side]:
            out += sum((a * b for a, b in zip(c, x0)), F(0))
        return _red(out)

    def laplace(self, z, ell, side, power=0):
        """``Σ_blocks c (z − G)^{−(power+1)} x0`` over ``F`` (``z``: a
        number, an SR expression in the variables, or a field element)."""
        S = _sage()
        F = self.field.F
        zf = z if getattr(z, 'parent', None) is not None and \
            z.parent() is F else self.field.from_sr(S.SR(z), 'z')
        out = F(0)
        for G, x0, c in self.blocks[ell][side]:
            M = zf * S.identity_matrix(F, G.nrows()) - G
            y = _solve_vec(M, list(x0), F, power + 1)
            out += sum((a * b for a, b in zip(c, y) if a != 0 and b != 0),
                       F(0))
        return _red(out)

    def C(self, tau, ell, point=None):
        """``C(τ)`` at numbers (array ``tau``; τ = 0 is the side '-' limit)."""
        import scipy.linalg as sl
        pt = dict(self.reference)
        pt.update(point or {})
        tau = np.atleast_1d(np.asarray(tau, dtype=float))
        out = np.zeros(tau.shape, dtype=complex)
        ev = self.field.evaluate
        for side in ('+', '-'):
            on = (tau > 0) if side == '+' else (tau <= 0)
            for G, x0, c in self.blocks[ell][side]:
                n = G.nrows()
                Gn = np.array([[ev(G[i, j], pt) if G[i, j] != 0 else 0j
                                for j in range(n)] for i in range(n)])
                xn = np.array([ev(v, pt) if v != 0 else 0j for v in x0])
                cn = np.array([ev(v, pt) if v != 0 else 0j for v in c])
                for i in np.nonzero(on)[0]:
                    out[i] += cn @ sl.expm(Gn * abs(tau[i])) @ xn
        return out


def _solve_vec(M, y, F, times=1):
    S = _sage()
    if F is S.QQ:
        for _ in range(times):
            y = list(M.solve_right(S.vector(F, y)))
        return y
    lu = _FFLU(M, F)
    for _ in range(times):
        y = lu.solve(y)
    return y


# ─── the driver ─────────────────────────────────────────────────────────────

def _saddle_names(model, num_params):
    from api.symbolic_out import _is_saddle_symbol
    return [str(k) for k in num_params if _is_saddle_symbol(str(k), model)]


def _propagator_data(prop, model):
    return {
        'K_ker': prop['K_ker'], 'K_ft': prop['K_ft'], 'G_ft': prop['G_ft'],
        'adj_ft': prop['adj_ft'], 'D_omega': prop['D_omega'],
        'D_delta': prop['D_delta'], 't_var': prop['t_var'],
        'omega': prop['omega'], 'nf': prop['nf'],
        'pole_vals': prop['pole_vals'], 'C_mats': prop['C_mats'],
        'model_name': (model or {}).get('name'),
    }


class _Setup:
    """The field, the substitutions and the symbolic propagator of one
    request (shared by the three entry points)."""

    def __init__(self, model, k, max_ell, external_fields, parameters,
                 symbols, saddle, result, kernel, kernel_values, verbose,
                 max_block=None, max_symbols=None):
        S = _sage()
        from api import compute_cumulants
        t_start = time.perf_counter()
        self.t_start = t_start
        if result is None:
            if parameters:
                # the model defaults for the names not given
                # (compute_cumulants takes a complete dict)
                full = {p['name']: p['default']
                        for p in (model.get('parameters') or [])
                        if p.get('default') is not None
                        and not p.get('mean_field')}
                full.update(parameters)
                parameters = full
            result = compute_cumulants(
                model, k=k, max_ell=max_ell, external_fields=external_fields,
                parameters=parameters, tau_grid=[0.0], parallel=False,
                verbose=verbose, use_cache=False)
        cfg = result.get('config') or {}
        if 'C_tau_x' in result or cfg.get('spatial'):
            raise SymbolicLoopRefused('spatial', 'spatial models are not '
                                      'covered (per-momentum realizations)')
        if result.get('t_min') is not None:
            raise SymbolicLoopRefused('k_unsupported', 'a switch-on result '
                                      '(t_min) is not stationary')
        if int(cfg.get('k', k)) != int(k):
            raise ValueError(f'result has k = {cfg.get("k")}, not {k}')
        self.result = result
        self.t_ref = time.perf_counter() - t_start
        num_params = result['num_params']
        self.num_params = num_params
        names_all = [str(p) for p in num_params]
        sad = _saddle_names(model, num_params)
        params = [n for n in names_all if n not in sad]
        if symbols is None:
            sym = list(params)
        else:
            sym = [str(x) for x in symbols]
            bad = [x for x in sym if x not in params]
            if bad:
                raise SymbolicLoopRefused(
                    'unbound_symbol', f'{bad} are not parameters of the run '
                    f'(parameters: {params}; saddle: {sad})')
        saddle_map = None
        if isinstance(saddle, dict):
            saddle_map = {str(a): b for a, b in saddle.items()}
            unknown = sorted(set(saddle_map) - set(sad))
            if unknown:
                raise ValueError(f'saddle={{...}} names {unknown}; the saddle '
                                 f'symbols of this run are {sad}')
            saddle = 'given'
        elif saddle == 'solve':
            raise SymbolicLoopRefused(
                'saddle_unsolved', "saddle='solve': the stationary mean-field "
                'equations are not available symbolically from the pipeline '
                "(its expanded action drops that sector); pass saddle="
                "{name: closed form} (checked at the reference) or use "
                "'numeric' / 'symbol'")
        elif saddle not in ('numeric', 'symbol'):
            raise ValueError("saddle must be 'numeric', 'symbol' or a dict "
                             "{saddle name: closed form}")
        notes = []
        vars_ = list(sym) + (list(sad) if saddle == 'symbol' else [])
        kvals = {str(a): b for a, b in (kernel_values or {}).items()}
        if kernel is not None:
            kv = set()
            for t in kernel.terms:
                for ex in (t.coeff, t.rate):
                    kv |= {str(v) for v in S.SR(ex).variables()}
            extra = sorted(kv - set(names_all))
            missing = [v for v in extra if v not in kvals]
            if missing:
                raise SymbolicLoopRefused(
                    'unbound_symbol', f'kernel symbols {missing} need '
                    'reference values (kernel_values=)')
            vars_ += [v for v in extra if v not in vars_]
        cap_s = MAX_SYMBOLS if max_symbols is None else int(max_symbols)
        if len(vars_) > cap_s:
            raise SymbolicLoopRefused('too_large', f'{len(vars_)} symbols '
                                      f'(max_symbols = {cap_s})')
        fld = _Field(vars_)
        self.field = fld
        reference, subs = {}, {}
        for key, v in num_params.items():
            n = str(key)
            if n in vars_:
                reference[n] = float(complex(v).real)
                continue
            if saddle_map is not None and n in saddle_map:
                continue
            subs[S.SR(key)] = S.SR(_rational_value(v, n in sad))
        for n in vars_:
            if n not in reference:
                reference[n] = float(kvals[n])
        if saddle_map is not None:
            for n, expr in saddle_map.items():
                e = S.SR(expr).subs(subs)
                x = fld.from_sr(e, f'saddle {n}')
                vref = complex(num_params[S.SR(n)] if S.SR(n) in num_params
                               else [v for kk, v in num_params.items()
                                     if str(kk) == n][0])
                got = fld.evaluate(x, reference)
                if abs(got - vref) > 1e-9 * max(1.0, abs(vref)):
                    raise SymbolicLoopRefused(
                        'saddle_mismatch', f'saddle {n} = {expr} gives {got} '
                        f'at the reference, the pipeline solved {vref}')
                subs[S.SR(n)] = fld.to_sr(x)
            rest = [n for n in sad if n not in saddle_map]
            notes.append(f'saddle {sorted(saddle_map)} by the given closed '
                         f'form(s) (checked at the reference)'
                         + (f'; {rest} at the reference value' if rest else ''))
        numeric_params = [n for n in params if n not in sym]
        if numeric_params:
            notes.append(f'numeric parameters (simplest rational of their '
                         f'double): {numeric_params}')
        if sad and saddle == 'numeric':
            notes.append(f'saddle {sad} at its reference value (simplest '
                         f'rational of its double; solver noise around a '
                         f'short rational snapped within {SADDLE_ATOL:g}): '
                         f'exact for that saddle, i.e. symbolic in the '
                         f'parameters only where the saddle does not move '
                         f'with them')
        elif sad and saddle == 'symbol':
            notes.append(f'saddle {sad} kept symbolic: rational in parameters '
                         f'and saddle, valid on the mean-field manifold')
        self.reference, self.subs, self.notes = reference, subs, notes
        self.sad, self.sym, self.kvals = sad, sym, kvals
        self.stats = _Stats(max_block)
        t0 = time.perf_counter()
        prop = result['propagator']
        self.sp = symbolic_propagator(prop['K_ft'], prop['omega'], fld, subs)
        self.t_prop = time.perf_counter() - t0
        _check_zero_pattern(self.sp, prop, fld, reference)
        self.ext = cfg.get('external_fields') or result.get('external_fields')
        self.origin = cfg.get('origin_leaf_idx', 0)
        self.pdata = _propagator_data(prop, model)
        self.k = int(k)
        self.ext_vars = [S.SR.var(f't_{j + 1}') for j in range(self.k)]

    def diagrams(self, ell):
        """``(index, field prefactor, integrate_diagram output)`` of the
        diagrams of one loop order with a nonzero prefactor."""
        S = _sage()
        from engine.integration.time_domain import final_integral as FI
        recs = [r for r in self.result['diagrams'] if int(r['ell']) == int(ell)]
        for di, rec in enumerate(recs):
            e = S.SR(rec['combined_prefactor']).subs(self.subs)
            kern = sorted(str(v) for v in e.variables()
                          if str(v).startswith('z_'))
            if kern:
                # a conductance vertex (``z_g_*``: its kernel is an edge or
                # a dressed line) or a non-local noise cumulant
                # (``z_kappa_*``): neither has a symbolic realization here
                raise SymbolicLoopRefused(
                    'noise_kernel' if any(x.startswith('z_kappa') for x in kern)
                    else 'kernel_edge',
                    f'ell={ell} diagram {di}: kernel symbol(s) {kern} in the '
                    'vertex coefficient')
            cp = self.field.from_sr(e, 'prefactor')
            if cp == 0:
                continue
            out = FI.integrate_diagram(
                typed_diagram=rec['typed_diagram'], propagator_data=self.pdata,
                combined_prefactor=S.SR(1), ext_time_vars=self.ext_vars,
                num_params=self.num_params, origin_leaf_idx=self.origin,
                external_fields=self.ext)
            if out.get('status') != 'ok':
                raise SymbolicLoopRefused(
                    'failed_diagram', f'ell={ell} diagram {di}: '
                    f'{out.get("status")} {out.get("reason", "")}')
            yield di, cp, out

    def stats_dict(self, per_ell):
        st = self.stats
        return {'reference_run_seconds': self.t_ref,
                'propagator_seconds': self.t_prop,
                'solves': st.solves, 'max_block': st.max_block,
                'solve_seconds': st.t_solve, 'lu_factorizations': len(st.lu),
                'delta_subsets': st.delta_subsets,
                'by_ell': per_ell,
                'total_seconds': time.perf_counter() - self.t_start,
                'entry_states': {key: e.n for key, e in self.sp.entries.items()},
                'variables': list(self.field.names)}


def symbolic_cumulant(model, k=2, max_ell=1, *, external_fields,
                      parameters=None, symbols=None, saddle='numeric',
                      ells=None, result=None, kernel=None, kernel_values=None,
                      reference_check=True, max_block=None, max_symbols=None,
                      verbose=False) -> SymbolicCumulant:
    """Equal-time cumulant (``kernel=None``; any k) or the k = 2 kernel
    pairing ``∫ L(s) C(s) ds`` (``kernel``: a C2
    :class:`api.symbolic_out.Kernel`), per loop order, as rational
    functions of the parameters (module docstring).

    ``parameters``: the reference values (model defaults otherwise);
    ``symbols``: the parameter names kept symbolic (``None``: all);
    ``saddle``: 'numeric' (default) | 'symbol' | {saddle name: closed form};
    ``ells``: the loop orders (default ``0..max_ell``); ``result``: a
    ``compute_cumulants`` result of the same request (its reference point);
    ``kernel_values``: reference values of the kernel's own symbols (they
    stay symbolic); ``reference_check``: compare with the pipeline at the
    reference point (``SymbolicCumulant.check``, raises beyond
    ``CHECK_RTOL``); ``max_block`` / ``max_symbols``: the size limits
    (defaults ``MAX_BLOCK`` / ``MAX_SYMBOLS``; beyond them ``too_large``)."""
    from engine.integration.time_domain import final_integral as FI
    if kernel is not None and int(k) != 2:
        raise SymbolicLoopRefused('k_unsupported',
                                  'kernel pairings are defined for k = 2')
    st = _Setup(model, k, max_ell, external_fields, parameters, symbols,
                saddle, result, kernel, kernel_values, verbose, max_block,
                max_symbols)
    F = st.field.F
    kterms = None
    if kernel is not None:
        S = _sage()
        kterms = []
        for t in kernel.terms:
            kterms.append((t.side, int(t.power),
                           st.field.from_sr(S.SR(t.coeff).subs(st.subs),
                                            'kernel coefficient'),
                           st.field.from_sr(S.SR(t.rate).subs(st.subs),
                                            'kernel rate')))
    if ells is None:
        ells = list(range(int(max_ell) + 1))
    by_ell, per_ell = {}, {}
    for ell in ells:
        te = time.perf_counter()
        total, n_sub = F(0), 0
        for _di, cp, out in st.diagrams(ell):
            val, ns = _diagram_value(out, st, FI, kterms=kterms)
            n_sub += ns
            total = _red(total + cp * val)
        by_ell[int(ell)] = total
        per_ell[int(ell)] = {'seconds': time.perf_counter() - te,
                             'subsets': n_sub, 'size': expression_size(total)}
    pipeline = _pipeline_values(st, ells, kernel) if reference_check else {}
    res = SymbolicCumulant(
        kind='equal_time' if kernel is None else 'pairing', by_ell=by_ell,
        field=st.field, reference=st.reference, pipeline=pipeline, k=st.k,
        stats=st.stats_dict(per_ell), notes=st.notes)
    res._ctx = {'propagator': st.sp, 'result': st.result, 'subs': st.subs}
    if reference_check:
        res.check()
    return res


def symbolic_triple(model, max_ell=1, *, external_fields, parameters=None,
                    symbols=None, saddle='numeric', ells=None, result=None,
                    max_block=None, max_symbols=None,
                    verbose=False) -> SymbolicTriple:
    """The rational matrix-exponential triples of a k = 2 result
    (:class:`SymbolicTriple`; arguments as :func:`symbolic_cumulant`)."""
    from engine.integration.time_domain import final_integral as FI
    st = _Setup(model, 2, max_ell, external_fields, parameters, symbols,
                saddle, result, None, None, verbose, max_block, max_symbols)
    if ells is None:
        ells = list(range(int(max_ell) + 1))
    blocks, per_ell = {}, {}
    for ell in ells:
        te = time.perf_counter()
        sink = {'+': [], '-': []}
        n_sub = 0
        for _di, cp, out in st.diagrams(ell):
            _v, ns = _diagram_value(out, st, FI, triple_sink=sink, weight=cp)
            n_sub += ns
        blocks[int(ell)] = sink
        per_ell[int(ell)] = {'seconds': time.perf_counter() - te,
                             'subsets': n_sub,
                             'states': {s: sum(b[0].nrows() for b in v)
                                        for s, v in sink.items()}}
    return SymbolicTriple(blocks=blocks, field=st.field,
                          reference=st.reference,
                          stats=st.stats_dict(per_ell), notes=st.notes)


def _pipeline_values(st, ells, kernel):
    """The pipeline's own numbers at the reference point (module docstring,
    step 5): equal time from the Phase J callables, pairings from S4's
    exact smooth ``integrate_against`` (omitted when S4 has no triples)."""
    out = {}
    pjs = st.result.get('phase_j_by_ell') or {}
    if kernel is None:
        for ell in ells:
            pj = pjs.get(ell)
            out[int(ell)] = (complex(pj['total_C'](*([0.0] * st.k)))
                             if pj else 0j)
        return out
    try:
        from api.cumulant import Cumulant
        c = Cumulant.from_result(st.result)
        vals = {a: float(b) for a, b in st.kvals.items()}
        for ell in ells:
            out[int(ell)] = complex(c.integrate_against(
                kernel, ell=ell, method='exact', contact='exclude',
                values=vals or None).smooth)
    except Exception as exc:                    # no triples: no reference
        st.notes.append(f'no pipeline reference for the pairing '
                        f'({type(exc).__name__}: {str(exc)[:120]})')
        return {}
    return out


def _check_zero_pattern(sp, prop, fld, reference):
    """The symbolic δ coefficients equal the pipeline's ``D_delta`` at the
    reference, and no symbolically nonzero entry (smooth or δ part)
    vanishes there (``degenerate_reference``: the reference run would have
    dropped its subsets)."""
    S = _sage()
    D_ref = prop.get('D_delta')
    pt = {S.SR.var(n): v for n, v in reference.items()}
    for (p, r), e in sp.entries.items():
        if D_ref is not None:
            try:
                dref = complex(S.CDF(S.SR(D_ref[p, r]).subs(pt)))
            except Exception:
                dref = None
            dsym = fld.evaluate(e.d, reference) if e.d != 0 else 0j
            if dref is not None and (dref == 0) != (e.d == 0):
                raise SymbolicLoopRefused(
                    'degenerate_reference', f'δ coefficient G[{p},{r}]: '
                    f'symbolic {"zero" if e.d == 0 else "nonzero"}, '
                    f'{dref} at the reference point')
            if dref is not None and abs(dsym - dref) > \
                    CHECK_RTOL * max(1.0, abs(dref)):
                raise SymbolicLoopRefused(
                    'prefactor_mismatch', f'δ coefficient G[{p},{r}]: '
                    f'{dsym} vs the pipeline {dref}')
        if e.n:
            num = e.num
            vals = {n: S.QQ(_exact(reference[n])) for n in fld.names}
            zero = all(
                (c if not fld.names else fld.F(c).numerator()(
                    *[vals[n] for n in fld.names])) == 0
                for c in num.list())
            if zero:
                raise SymbolicLoopRefused(
                    'degenerate_reference', f'G[{p},{r}] smooth part '
                    'vanishes at the reference point')


def _diagram_value(out, st, FI, kterms=None, triple_sink=None, weight=None):
    """One diagram (unit prefactor): ``Σ_perm Σ_subsets δ · scal · value /
    compensation`` over ``F`` (equal time or pairing), or its triple blocks
    appended to ``triple_sink`` (weighted by ``weight``); and the number of
    subsets."""
    S = _sage()
    fld, sp = st.field, st.sp
    F = fld.F
    contrib = out['contribution']
    parts = getattr(contrib, 'phase_j_parts', None)
    if parts is None:
        raise SymbolicLoopRefused('pole_route_subset', 'the contribution '
                                  'exposes no subset structure')
    diags = [d for d in out.get('subset_diagnostics') or []
             if isinstance(d, dict) and d.get('status') == 'evaluated']
    subs_ = parts['subsets']
    if len(diags) != len(subs_):
        raise RuntimeError('subset diagnostics do not match the closures')
    edge_info = out['edge_info']
    perms, comp = parts['perms'], parts['compensation']
    org, fidx, kk = parts['origin_leaf_idx'], parts['free_ext_idx'], parts['k']
    total = F(0)
    n = 0
    inv_comp = F(1) / F(S.QQ(comp))
    for si, (cfn, dg) in enumerate(zip(subs_, diags)):
        sub = getattr(cfn, 'ss_subset', None)
        if getattr(cfn, 'ss_theta0_empty', False):
            continue
        if sub is None:
            raise SymbolicLoopRefused(
                'pole_route_subset', 'a subset the state-space route '
                'declined (noise-kernel boxes, unrealizable kernels, ...)')
        smooth = list(dg['smooth_edges'])
        dfac = F(1)
        for e in dg['delta_edges']:
            ei = edge_info[e]
            dfac *= sp.D(int(ei['pi']), int(ei['ri']))
        # the route's numeric prefactor (unit diagram prefactor) is the δ
        # factor: the symbolic one must agree at the reference
        pref = complex(cfn.ss_prefactor)
        dnum = fld.evaluate(dfac, st.reference) if dfac != 0 else 0j
        if abs(dnum - pref) > CHECK_RTOL * max(1.0, abs(pref)):
            raise SymbolicLoopRefused(
                'prefactor_mismatch', f'subset {si}: symbolic δ factor {dnum} '
                f'vs the route\'s prefactor {pref}')
        if dfac == 0:
            continue
        if dg['delta_edges']:
            st.stats.delta_subsets += 1
        ents = {eid: _entry_of(sp, sub, edge_info, smooth[row], ent)
                for eid, (_t, _h, ent, row) in enumerate(sub.edges)}
        cents = {row: _entry_of(sp, sub, edge_info, smooth[row], ent)
                 for (row, _a, _b, _c, ent) in sub.const_rows}
        ss = _SymSubset(sub, ents, cents, F, st.stats)
        n += 1
        for perm in perms:
            if triple_sink is not None:
                _triple_subset(ss, perm, org, fidx, fld, FI, triple_sink,
                               weight * dfac * inv_comp)
            elif kterms is None:
                T = tuple([0.0] * kk)
                fv = [0.0 for _ in fidx]
                tie = FI._TieContext(T, tuple(perm[j] for j in fidx),
                                     None if org is None else perm[org])
                total = _red(total + dfac * ss.equal_time(fv, tie, FI))
            else:
                total = _red(total + dfac * _pair_subset(
                    ss, perm, org, fidx, kterms, fld, FI))
    return _red(total * inv_comp), n


def _entry_of(sp, sub, edge_info, idx, ent):
    """The symbolic entry of one smooth row; a conductance-kernel row or a
    dressed line (whose operator is not the plain propagator entry) is
    refused by name."""
    if idx >= len(edge_info):
        raise SymbolicLoopRefused('kernel_edge', 'a conductance kernel row '
                                  '(its kernel has no symbolic realization '
                                  'here)')
    ei = edge_info[idx]
    pi, ri = int(ei['pi']), int(ei['ri'])
    if ent is not sub.model.entry(pi, ri):
        raise SymbolicLoopRefused('dressed_line', 'a dressed line (kernel * '
                                  'propagator series connection)')
    return sp.entries[(pi, ri)]


def _ray_of(ss, perm, org, fidx, sg, FI):
    """The ray of one subset and Wick permutation on side ``sg`` (+1: τ > 0,
    −1: τ < 0) of the k = 2 lag axis, or ``None`` (zero)."""
    e = (0, sg)
    eo = 0.0 if org is None else float(e[perm[org]])
    fb = [0.0 for _ in fidx]
    fd = [float(e[perm[j]]) - eo for j in fidx]
    tie = FI._TieContext((0.0, float(sg)), tuple(perm[j] for j in fidx),
                         None if org is None else perm[org])
    return ss.ray(fb, fd, tie, FI)


def _pair_subset(ss, perm, org, fidx, kterms, fld, FI):
    """``Σ_terms ∫_side L C_subset`` for one Wick permutation:
    ``coeff · q! (±1)^q · c (−r_u − βG)^{−(q+1)} x0`` per kernel term
    ``(side, q, coeff, rate)`` (``r_u``: the rate in ``u = |s|``)."""
    S = _sage()
    F = fld.F
    out = F(0)
    for side, sg in (('positive', 1), ('negative', -1)):
        terms = [t for t in kterms if t[0] == side]
        if not terms:
            continue
        ray = _ray_of(ss, perm, org, fidx, sg, FI)
        if ray is None:
            continue
        bq = S.QQ(ray['beta'])
        G, x0, c = ray['G'], list(ray['x0']), list(ray['c'])
        n = G.nrows()
        for _side, q, coeff, rate in terms:
            ru = rate * sg
            M = (-ru) * S.identity_matrix(F, n) - bq * G
            y = _solve_vec(M, x0, F, q + 1)
            val = sum((ci * yi for ci, yi in zip(c, y)
                       if ci != 0 and yi != 0), F(0))
            out = _red(out + coeff * ray['scal'] * val
                       * S.factorial(q) * (sg ** q))
    return out


def _triple_subset(ss, perm, org, fidx, fld, FI, sink, w):
    """Append the rational blocks ``(βG, x0, w·scal·c)`` of one subset and
    Wick permutation to ``sink['+' / '-']``."""
    S = _sage()
    F = fld.F
    for side, sg in (('+', 1), ('-', -1)):
        ray = _ray_of(ss, perm, org, fidx, sg, FI)
        if ray is None:
            continue
        bq = S.QQ(ray['beta'])
        sc = w * ray['scal']
        sink[side].append((bq * ray['G'], list(ray['x0']),
                           [_red(sc * v) if v != 0 else F(0)
                            for v in ray['c']]))
