r"""
Symbolic loop corrections as rational functions (stage C3)
==========================================================

On the state-space route (``engine/integration/time_domain/statespace.py``)
the stationary part of every Phase J subset integral is a composition of
linear solves: the states of the downsets of the subset's causal poset solve
``(-M_J) X_J = Σ_v F_v X_{J∖v}`` with ``M_J`` a Kronecker sum of the
propagator realizations and ``F_v`` contractions / openings by their ``c``
and ``b`` vectors.  With the parameters symbolic, therefore,

* the EQUAL-TIME cumulants (all external times equal; k = 2 and k >= 3) and
* the pairings ``∫ L(τ) C(τ) dτ`` (k = 2) with exponential and
  polynomial × exponential kernels ``L`` (one more resolvent:
  ``q! c (z − βG)^{−(q+1)} x0``)

are RATIONAL functions of the parameters (and of the saddle), at every loop
order: exact linear algebra over ``F = QQ(params)``, no polynomial roots, no
``RootOf``, no distinct-pole assumption (a double pole is an ordinary
companion matrix here).  This module computes them.

How (``symbolic_cumulant``)
---------------------------
1. ``compute_cumulants`` runs once at a numeric REFERENCE point (the
   parameters given, or the model defaults): it supplies the diagrams, their
   symbolic prefactors (``result['diagrams'][...]['combined_prefactor']``,
   Sage expressions in the parameters and the saddle), the propagator data
   and the reference numbers the result is checked against.
2. The propagator ``G(s) = K(s)^{-1}`` (``s = iω``, ``K`` the pipeline's
   ``K_ft``) is inverted exactly over ``F(s)``; every entry is split
   ``G_pr = D_pr + r(s)/d(s)`` (``deg r < deg d``; a polynomial part of
   degree >= 1 is refused, ``improper_entry``) and realized by its
   controllable companion form (``d`` made monic over ``F``): minimal,
   because ``r/d`` is reduced, and exact through repeated roots of ``d``.
   ``D`` is the δ(t) coefficient (the pipeline's ``D_delta``).  Floats in the
   model (kernel literals, numeric parameters) enter as the exact rationals
   of their binary values (as in ``api._realization``, S1).
3. Each diagram is integrated once more by ``integrate_diagram`` with a unit
   prefactor, only to obtain its subsets: the state-space ``SubsetSS``
   objects (poset, active edges, stationary downsets, the event program),
   their δ edges and the edge -> propagator-entry map.  Nothing numeric of
   them is used except the structure.  The subset prefactor is the
   diagram's symbolic prefactor times the symbolic δ coefficients of its δ
   edges; the Wick permutations and the compensation are the
   contribution's own (``phase_j_parts``).
4. The stationary sweep and the event(s) are replayed with the symbolic
   companion realizations (numpy object tensors of field elements; each
   resolvent one exact ``solve_right`` of the dense Kronecker sum over
   ``F``).  At equal times the program is one event; for a pairing it is
   event, segment, event and the segment's generator ``G`` is built
   explicitly (diagonal blocks ``M_I``, firings below them).
5. Checks (never silent): every subset's symbolic prefactor, evaluated at
   the reference point, equals the route's numeric one (1e-12 relative);
   the symbolic zero pattern of ``G``'s smooth and δ parts equals the
   reference's (a reference point where an entry vanishes accidentally
   would lose subsets: ``degenerate_reference``); the total, evaluated at
   the reference, is compared with the pipeline's own value
   (``SymbolicCumulant.check``; the equal-time value against
   ``phase_j_by_ell[ell]['total_C'](0, ..., 0)``).

Which symbols stay symbolic
---------------------------
``symbols=None``: every model parameter in ``num_params`` (not the saddle);
a list keeps only those, the others enter at their exact reference values.
The saddle (``xstar1`` ...): ``saddle='numeric'`` (default) uses the exact
binary value of the reference saddle -- the result is then exact for the
REFERENCE saddle, i.e. it is the symbolic function of the remaining
parameters only where the saddle does not depend on them (e.g. ``x* = 0``);
``saddle='symbol'`` keeps the saddle symbols as variables of ``F`` (the
result is rational in parameters and saddle and holds on the mean-field
manifold, which ``mf_equations`` does not solve for you);
``saddle='solve'`` substitutes an exact solution of the mean-field
equations, refused (``saddle_unsolved``) unless every saddle value is a
rational function of the symbolic parameters (checked at the reference).

Conventions
-----------
* Equal time: the external times coincide; constant rows are decided by
  Phase J's own Θ(0) / tie rule (Itô).  For k = 2 this is the Itô left
  limit ``C(0⁻)`` of the pipeline (``C_tau``'s τ = 0 sample is taken at
  ``τ = −_ITO_EPS`` instead, so it differs from it by O(1e-6) where C has a
  slope), equal to ``api.me_output`` side '-' at u = 0.
* Only the SMOOTH part: δ records of Feature B (shot-noise subsets on a δ
  surface) are not included, as in ``C_tau``.
* Pairing (k = 2): ``∫ L(s) C(s) ds`` with ``C(τ) = κ(x_1(0), x_2(τ))`` and
  a C2 :class:`api.symbolic_out.Kernel` (terms ``coeff · s^q e^{rate s}`` on
  side 'positive' / 'negative'); the smooth part of ``C``; the kernel's
  coefficients and rates may carry their own symbols (they join ``F``).
* τ != 0 (``C(τ)`` itself) needs ``e^{Gτ}`` symbolically: not built here
  (see ``docs`` in the report: eigen data / ``RootOf`` as C2, or ``expm`` of
  the explicit rational ``G``); ``SymbolicCumulant`` exposes the rational
  triple ``(G, x0, c)`` per side for that purpose (:meth:`triple`).

Refusals (``SymbolicLoopRefused.reason``): ``spatial``, ``k_unsupported``,
``complex_coefficients``, ``non_rational``, ``improper_entry``,
``singular_pencil``, ``pole_route_subset`` (a subset the state-space route
declined: conductance kernels, noise-kernel boxes, ...), ``kernel_edge`` /
``dressed_line`` (rows the symbolic realization does not cover),
``offset_row`` / ``offset_slot`` (rows with a constant time offset),
``degenerate_reference``, ``saddle_unsolved``, ``prefactor_mismatch``,
``too_large`` (a Kronecker block over ``MAX_BLOCK`` states or more
symbolic parameters than ``MAX_SYMBOLS``), ``failed_diagram``.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

import numpy as np

__all__ = ['SymbolicLoopRefused', 'SymbolicPropagator', 'SymbolicCumulant',
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
            self.R = S.PolynomialRing(S.QQ, list(self.names))
            self.F = self.R.fraction_field()
        else:
            self.R = self.F = S.QQ
        self.sr_vars = {n: S.SR.var(n) for n in self.names}

    def from_sr(self, e, what='expression'):
        S = _sage()
        e = S.SR(e)
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
    Rs = S.PolynomialRing(S.QQ, names)
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
            n = den.degree() if rem != 0 else 0
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
    def __init__(self):
        self.lu = {}
        self.kron = {}
        self.solves = 0
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
    if n > MAX_BLOCK:
        raise SymbolicLoopRefused('too_large', f'a Kronecker block of {n} '
                                  f'states (MAX_BLOCK = {MAX_BLOCK})')
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
        if n > MAX_BLOCK * 4:
            raise SymbolicLoopRefused('too_large', f'a segment of {n} states')
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


# ─── the driver ─────────────────────────────────────────────────────────────

@dataclass
class SymbolicCumulant:
    """Result of :func:`symbolic_cumulant`.

    ``by_ell``: {ell: field element} (equal-time value per loop order);
    ``total``: their sum; ``field``: the ``_Field`` (``names`` = the
    symbolic variables); ``reference``: {name: value} of the reference
    point; ``pipeline``: {ell: the pipeline's equal-time value there};
    ``stats``: sizes and timings; ``notes``: the decisions taken
    (saddle, numeric parameters)."""
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

    def expression(self, ell=None):
        """Sage ``SR`` expression (``ell=None``: the sum over loop orders)."""
        x = self.total if ell is None else self.by_ell[ell]
        return self.field.to_sr(x)

    def size(self, ell=None):
        return expression_size(self.total if ell is None else self.by_ell[ell])

    def sympy(self, ell=None):
        from api.symbolic_out import to_sympy
        return to_sympy(self.expression(ell))

    def latex(self, ell=None):
        from api.symbolic_out import to_latex
        return to_latex(self.expression(ell))

    def callable(self, ell=None):
        """numpy callable ``f(**params)`` (via sympy.lambdify)."""
        import sympy
        ex = self.sympy(ell)
        syms = [sympy.Symbol(n) for n in self.names]
        by = {str(s): s for s in ex.free_symbols}
        fn = sympy.lambdify([by.get(n, s) for n, s in zip(self.names, syms)],
                            ex, modules='numpy')
        names = list(self.names)

        def f(**kw):
            missing = [n for n in names if n not in kw]
            if missing:
                raise TypeError(f'missing parameters {missing}')
            return fn(*[kw[n] for n in names])
        f.params = names
        return f

    def evaluate(self, point=None, ell=None):
        """Value at ``point`` (default: the reference point)."""
        pt = dict(self.reference)
        pt.update(point or {})
        x = self.total if ell is None else self.by_ell[ell]
        return self.field.evaluate(x, pt)

    def check(self, rtol=CHECK_RTOL):
        """``{ell: relative difference}`` to the pipeline at the reference;
        raises ``AssertionError`` beyond ``rtol``."""
        out = {}
        scale = max([abs(v) for v in self.pipeline.values()] + [1e-300])
        for ell, ref in self.pipeline.items():
            d = abs(self.evaluate(ell=ell) - ref) / scale
            out[ell] = d
            if d > rtol:
                raise AssertionError(f'ell={ell}: symbolic {self.evaluate(ell=ell)} '
                                     f'vs pipeline {ref} ({d:.2e})')
        return out


def _model_dict(model):
    return model


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


def symbolic_cumulant(model, k=2, max_ell=1, *, external_fields,
                      parameters=None, symbols=None, saddle='numeric',
                      ells=None, result=None, kernel=None,
                      reference_check=True, verbose=False):
    """Equal-time cumulant (``kernel=None``) or the k = 2 kernel pairing
    ``∫ L(s) C(s) ds`` (``kernel``: a C2 :class:`api.symbolic_out.Kernel`)
    per loop order, as rational functions (module docstring).

    ``parameters``: the reference values (model defaults otherwise);
    ``symbols``: the parameter names kept symbolic (``None``: all);
    ``saddle``: 'numeric' | 'symbol' | 'solve'; ``ells``: the loop orders
    (default ``range(max_ell + 1)``); ``result``: a ``compute_cumulants``
    result of the same request to reuse (its reference point)."""
    S = _sage()
    from api import compute_cumulants
    from engine.integration.time_domain import final_integral as FI
    from engine.integration.time_domain import statespace as SS
    from api._realization import _exact_number
    t_start = time.perf_counter()
    if kernel is not None and k != 2:
        raise SymbolicLoopRefused('k_unsupported',
                                  'kernel pairings are defined for k = 2')
    if result is None:
        result = compute_cumulants(
            model, k=k, max_ell=max_ell, external_fields=external_fields,
            parameters=parameters, tau_grid=[0.0], parallel=False,
            verbose=verbose, use_cache=False)
    if 'C_tau_x' in result or (result.get('config') or {}).get('spatial'):
        raise SymbolicLoopRefused('spatial', 'spatial models are not covered')
    t_ref = time.perf_counter() - t_start
    num_params = result['num_params']
    prop = result['propagator']
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
    if saddle not in ('numeric', 'symbol', 'solve'):
        raise ValueError("saddle must be 'numeric', 'symbol' or 'solve'")
    notes = []
    vars_ = list(sym) + (list(sad) if saddle == 'symbol' else [])
    kern_terms = []
    if kernel is not None:
        kv = set()
        for t in kernel.terms:
            for ex in (t.coeff, t.rate):
                kv |= {str(v) for v in S.SR(ex).variables()}
        extra = sorted(kv - set(vars_) - set(names_all))
        vars_ += extra
    if len(vars_) > MAX_SYMBOLS:
        raise SymbolicLoopRefused('too_large', f'{len(vars_)} symbols '
                                  f'(MAX_SYMBOLS = {MAX_SYMBOLS})')
    fld = _Field(vars_)
    F = fld.F
    reference = {}
    subs = {}
    for key, v in num_params.items():
        n = str(key)
        if n in vars_:
            reference[n] = float(v)
            continue
        subs[S.SR(key)] = S.SR(_rational_value(v, n in sad))
    numeric_params = [n for n in params if n not in sym]
    if numeric_params:
        notes.append(f'numeric parameters (exact binary values): '
                     f'{numeric_params}')
    if sad:
        if saddle == 'numeric':
            notes.append(f'saddle {sad} at its reference value (simplest rational '
                         f'of its double): the result is exact for that saddle')
        elif saddle == 'symbol':
            notes.append(f'saddle {sad} kept symbolic: rational in parameters '
                         f'and saddle, valid on the mean-field manifold')
        else:
            subs.update(_solve_saddle(model, result, sad, sym, fld, subs))
            notes.append(f'saddle {sad} substituted by an exact rational '
                         f'solution of the mean-field equations')
    if kernel is not None:
        for t in kernel.terms:
            for ex in (t.coeff, t.rate):
                for v in S.SR(ex).variables():
                    if str(v) in vars_ and str(v) not in reference:
                        raise SymbolicLoopRefused(
                            'unbound_symbol', f'kernel symbol {v} needs a '
                            'reference value (pass it in parameters=)')
    stats = _Stats()
    t0 = time.perf_counter()
    sp = symbolic_propagator(prop['K_ft'], prop['omega'], fld, subs)
    t_prop = time.perf_counter() - t0
    _check_zero_pattern(sp, prop, fld, reference)
    cfg = result.get('config') or {}
    ext = cfg.get('external_fields') or result.get('external_fields')
    origin = cfg.get('origin_leaf_idx', 0)
    pdata = _propagator_data(prop, model)
    ext_vars = [S.SR.var(f't_{j + 1}') for j in range(k)]
    if ells is None:
        ells = list(range(int(max_ell) + 1))
    by_ell, pipeline_ref = {}, {}
    per_ell_stats = {}
    for ell in ells:
        recs = [r for r in result['diagrams'] if int(r['ell']) == int(ell)]
        te = time.perf_counter()
        total = F(0)
        n_sub = 0
        for di, rec in enumerate(recs):
            cp = fld.from_sr(S.SR(rec['combined_prefactor']).subs(subs),
                             'prefactor')
            if cp == 0:
                continue
            out = FI.integrate_diagram(
                typed_diagram=rec['typed_diagram'], propagator_data=pdata,
                combined_prefactor=S.SR(1), ext_time_vars=ext_vars,
                num_params=num_params, origin_leaf_idx=origin,
                external_fields=ext)
            if out.get('status') != 'ok':
                raise SymbolicLoopRefused(
                    'failed_diagram', f'ell={ell} diagram {di}: '
                    f'{out.get("status")} {out.get("reason", "")}')
            val, ns = _diagram_value(out, sp, fld, FI, SS, k, stats, kernel,
                                     reference)
            n_sub += ns
            total = _red(total + cp * val)
        by_ell[int(ell)] = total
        per_ell_stats[int(ell)] = {'seconds': time.perf_counter() - te,
                                   'subsets': n_sub,
                                   'size': expression_size(total)}
        if reference_check and kernel is None:
            pj = (result.get('phase_j_by_ell') or {}).get(ell)
            if pj:
                pipeline_ref[int(ell)] = complex(pj['total_C'](*([0.0] * k)))
            else:
                pipeline_ref[int(ell)] = 0j
    st = {'reference_run_seconds': t_ref, 'propagator_seconds': t_prop,
          'solves': stats.solves, 'max_block': stats.max_block,
          'solve_seconds': stats.t_solve, 'by_ell': per_ell_stats,
          'total_seconds': time.perf_counter() - t_start,
          'entry_states': {key: e.n for key, e in sp.entries.items()}}
    out = SymbolicCumulant(by_ell=by_ell, field=fld, reference=reference,
                           pipeline=pipeline_ref, k=k, stats=st, notes=notes)
    out._ctx = {'propagator': sp, 'result': result, 'subs': subs}
    if reference_check and kernel is None:
        out.check()
    return out


def _solve_saddle(model, result, sad, sym, fld, subs):
    raise SymbolicLoopRefused('saddle_unsolved', "saddle='solve' is not "
                              'implemented yet')


def _check_zero_pattern(sp, prop, fld, reference):
    """Symbolic zero pattern of ``G`` (smooth part and δ part) == the
    reference's (``degenerate_reference`` otherwise)."""
    S = _sage()
    nf = sp.nf
    D_ref = prop.get('D_delta')
    for (p, r), e in sp.entries.items():
        if D_ref is not None:
            try:
                dref = complex(S.CDF(S.SR(D_ref[p, r]).subs(
                    {S.SR.var(n): v for n, v in reference.items()})))
            except Exception:
                dref = None
            dsym = fld.evaluate(e.d, reference) if e.d != 0 else 0j
            if dref is not None and (dref == 0) != (e.d == 0):
                raise SymbolicLoopRefused(
                    'degenerate_reference', f'δ coefficient G[{p},{r}]: '
                    f'symbolic {"zero" if e.d == 0 else "nonzero"} but '
                    f'{dref} at the reference point')
            if dref is not None and abs(dsym - dref) > 1e-12 * max(1.0, abs(dref)):
                raise SymbolicLoopRefused(
                    'prefactor_mismatch', f'δ coefficient G[{p},{r}]: '
                    f'{dsym} vs the pipeline {dref}')
        if e.n and e.num is not None:
            # the reference realization must not lose the entry
            val = sp.transfer(p, r, 0.37 + 0.11j, reference) - (
                fld.evaluate(e.d, reference) if e.d != 0 else 0)
            if val == 0:
                raise SymbolicLoopRefused(
                    'degenerate_reference', f'G[{p},{r}] smooth part '
                    'vanishes at the reference point')


def _diagram_value(out, sp, fld, FI, SS, k, stats, kernel, reference):
    """Σ_perm Σ_subsets δ · scal · value / compensation of one diagram (unit
    prefactor) over ``F``; and the number of subsets."""
    F = fld.F
    S = _sage()
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
    for si, (cfn, dg) in enumerate(zip(subs_, diags)):
        sub = getattr(cfn, 'ss_subset', None)
        if getattr(cfn, 'ss_theta0_empty', False):
            continue
        if sub is None:
            raise SymbolicLoopRefused(
                'pole_route_subset', 'a subset the state-space route '
                'declined (conductance kernels, noise-kernel boxes, ...)')
        smooth = list(dg['smooth_edges'])
        delta = list(dg['delta_edges'])
        dfac = F(1)
        for e in delta:
            ei = edge_info[e]
            dfac *= sp.D(int(ei['pi']), int(ei['ri']))
        # the route's numeric prefactor (unit diagram prefactor) == δ factor
        pref = complex(cfn.ss_prefactor)
        dnum = fld.evaluate(dfac, reference) if dfac != 0 else 0j
        if abs(dnum - pref) > CHECK_RTOL * max(1.0, abs(pref)):
            raise SymbolicLoopRefused(
                'prefactor_mismatch', f'subset {si}: symbolic δ factor {dnum} '
                f'vs the route\'s prefactor {pref}')
        if dfac == 0:
            continue
        ents, cents = {}, {}
        for eid, (_t, _h, _ent, row) in enumerate(sub.edges):
            ents[eid] = _entry_of(sp, edge_info[smooth[row]])
        for (row, *_rest) in sub.const_rows:
            cents[row] = _entry_of(sp, edge_info[smooth[row]])
        ss = _SymSubset(sub, ents, cents, F, stats)
        n += 1
        for perm in perms:
            if kernel is None:
                T = tuple([0.0] * kk)
                fv = [0.0 for _ in fidx]
                tie = FI._TieContext(T, tuple(perm[j] for j in fidx),
                                     None if org is None else perm[org])
                total = _red(total + dfac * ss.equal_time(fv, tie, FI))
            else:
                total = _red(total + dfac * _pair_subset(
                    ss, perm, org, fidx, kk, kernel, fld, FI))
    return _red(total / F(S.QQ(comp))), n


def _entry_of(sp, ei):
    if ei.get('kernel') is not None or ei.get('is_kernel'):
        raise SymbolicLoopRefused('kernel_edge', 'a conductance kernel row')
    return sp.entries[(int(ei['pi']), int(ei['ri']))]


def _pair_subset(ss, perm, org, fidx, kk, kernel, fld, FI):
    """``Σ_terms ∫_side L C_subset`` for one Wick permutation."""
    S = _sage()
    F = fld.F
    out = F(0)
    for side, sg in (('positive', 1), ('negative', -1)):
        terms = [t for t in kernel.terms if t.side == side]
        if not terms:
            continue
        e = (0, sg)
        T0 = (0.0, 0.0)
        eo = 0.0 if org is None else float(e[perm[org]])
        fb = [0.0 for _ in fidx]
        fd = [float(e[perm[j]]) - eo for j in fidx]
        T_ref = (0.0, float(sg))
        tie = FI._TieContext(T_ref, tuple(perm[j] for j in fidx),
                             None if org is None else perm[org])
        ray = ss.ray(fb, fd, tie, FI)
        if ray is None:
            continue
        beta = ray['beta']
        bq = S.QQ(beta)                    # an integer slope (0/±1 algebra)
        G, x0, c = ray['G'], ray['x0'], ray['c']
        n = G.nrows()
        for t in terms:
            q = int(t.power)
            coeff = fld.from_sr(S.SR(t.coeff), 'kernel coefficient')
            rate = fld.from_sr(S.SR(t.rate), 'kernel rate')
            # ∫_0^∞ u^q e^{rate_u u} c e^{β G u} x0 du, s = sg·u
            ru = rate * sg
            M = (-ru) * S.identity_matrix(F, n) - bq * G
            y = list(x0)
            if F is S.QQ:
                for _ in range(q + 1):
                    y = list(M.solve_right(S.vector(F, y)))
            else:
                lu = _FFLU(M, F)
                for _ in range(q + 1):
                    y = lu.solve(y)
            val = sum((ci * yi for ci, yi in zip(c, y) if ci != 0 and yi != 0),
                      F(0)) * S.factorial(q) * (sg ** q)
            out = _red(out + coeff * ray['scal'] * val)
    return out
