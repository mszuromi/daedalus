r"""
Symbolic tree-level covariance in the Fourier domain, with exports.

Stage C1 of the internal SDE/δ feature plan (not distributed).  The pipeline builds the
symbolic frequency-domain propagator ``G(omega)`` (``api/_propagator.py``) before
any number is substituted.  This module turns it, together with the noise
matrix read off the second-order response-field sector of the action, into the
tree-level covariance

    C0(omega) = G(omega) . D . G(omega)^dagger

as a matrix of Sage expressions in the symbolic parameters, and exports the
entries (sympy, LaTeX, a numpy callable, a JSON string), pairs them with a
response kernel by residues, and inverts them to the time domain.

Conventions (all of them tested in ``tests/test_symbolic_out.py``)
-------------------------------------------------------------------
Fourier:    G(t) = (1/2 pi) int d omega exp(i omega t) G(omega)
            G(omega) = int dt exp(-i omega t) G(t)         (so d/dt -> i omega)
This is the pipeline-wide convention (``engine/integration/time_domain``,
``engine.core.field_theory.fourier_transform``).  Poles with Im(omega) > 0
give decaying exponentials for t > 0.

Action:     the bilinear (1,1) sector is  S ⊃ xt_r K_{r p}(d/dt) dx_p  and the
            noise sector is               S ⊃ -(1/2) xt_r D_{r s} xt_s.
            ``G = K^{-1}`` (rows physical, columns response, exactly the
            pipeline's ``G_ft``) and ``D`` is the symmetric response-space
            noise matrix, ``D_rr = -2 c_rr``, ``D_rs = -c_rs`` (r != s) for a
            monomial ``c_rs xt_r xt_s`` of the action.  For the white-noise
            OU process ``S ⊃ -D xt^2`` this is ``D_xx = 2 D``, i.e.
            <xi(t) xi(t')> = 2 D delta(t - t').  For Poisson (Hawkes)
            dynamics ``D_nn = phi(v*)``, the mean rate.

Covariance: with x(omega) = G(omega) xi(omega) and <xi xi^*> = 2 pi delta D,
            C0 = G D G^dagger  (entries indexed by PHYSICAL fluctuation
            fields, e.g. ``dx1``, ``dn2``) and, for the pipeline's ordering
            (first external field at time 0, second at time tau),

                C_{ab}(tau) = <x_a(0) x_b(tau)>
                            = (1/2 pi) int d omega  C0_{ab}(omega) exp(-i omega tau).

            The brief's ``int C0 exp(-i omega tau) d omega / 2 pi`` and
            ``compute_cumulants(external_fields=[a, b])`` use this same
            ordering.  Diagonal entries are even in tau, off-diagonal ones are
            not: swapping the entry (a,b) -> (b,a) flips the sign of tau.
            C0 is Hermitian, C0_{ab}(-omega) = conj(C0_{ab}(omega)), for real
            parameters, which every pipeline parameter is.

Contact terms.  If the propagator has a delta(t) piece (``D_delta`` of the
pipeline; Poisson shot noise on a ``dn`` field) then C0_{ab}(omega) tends to a
constant c_inf as omega -> infinity and the covariance contains
``c_inf * delta(tau)``.  The inverse transforms below return the SMOOTH part
(everything but that delta) and report ``c_inf`` separately.  That is the
pipeline's ``tau = 0`` convention too: Theta(0) = 0 and the Ito left limit, with
delta contributions "reported but not added" to ``C_tau``.

What is covered
---------------
* Temporal models (no field with ``spatial_dim >= 1``); any number of fields
  and populations, as long as the propagator is a RATIONAL function of omega
  (rational kernel images such as ``1/(1 + i omega tau_g)`` are fine).
* Gaussian noise: white (any ``-sum c_rs xt_r xt_s`` sector, including
  cross-correlated noise between fields), and colored noise AFTER the
  pipeline's own Markov embedding (``declare_cgf_term``: the embedding
  appears as extra physical fields, e.g. ``dxi1``, and a white noise on them).
  Shot (Poisson) noise enters through its second cumulant, which is the whole
  story at tree level.  Higher noise cumulants do not enter the covariance.
* Mean-field symbols (``xstar1``, ``vstar2`` ...) stay symbolic by default;
  ``mean_field='solve'`` substitutes the saddle the pipeline would use.

Not covered (clear errors, never a silently wrong result)
---------------------------------------------------------
* Spatial models (the propagator carries the inert ``Laplacian`` symbol and
  depends on k): ``UnsupportedModelError``.
* Non-rational propagators (delays ``exp(-i omega d)``, square roots of omega,
  ...): ``NonRationalPropagatorError`` naming the offending sub-expression.
* Non-local noise left as a cumulant kernel (not Markov-embedded):
  ``UnsupportedModelError``.
* ``inverse_transform`` and ``pair_with_kernel`` run on the stage-C2 engine
  below (item 40): no degree limit, the half-plane split stated as
  assumptions.  Their pre-C2 routes (every irreducible factor of degree <= 2
  in omega, split sampled around ``reference``, ``ClosedFormUnavailable``
  otherwise) remain behind ``legacy=True`` / ``any_degree=False``.

Stage C2: time domain, any degree
---------------------------------
``tree_modes`` / ``TreeCovariance.modes`` give C(tau) as a ``ModeSum`` of
``Mode(coeff, rate, power, side)`` terms, ``coeff * tau**power *
exp(rate*tau)`` on ``side`` 'positive' (tau > 0) or 'negative', plus the
``contact`` weight of delta(tau).  Poles of irreducible factors of degree >= 3
are ``RootOf`` symbols (root ``index`` of a polynomial, roots ordered by real
part); coefficients are polynomials in them with coefficients rational in the
parameters.  Each result lists the ``assumptions`` it needs (Routh/Sturm sign
conditions, distinct poles) and refuses (``AssumptionViolation``) at values
that violate them.  ``integrate_against`` with a ``Kernel`` (exponential,
polynomial x exponential, two-sided exponential) gives ``int L(s) C(s) ds`` exactly;
it equals the Fourier route ``pair_with_kernel(C0, L~(-omega))`` on the
smooth part.  The numeric twin for a computed cumulant at any loop order
(matrix-exponential triples of the state-space route, no poles) is
``api.me_output`` / ``Cumulant.integrate_against`` (item 40).
"""
from __future__ import annotations

import json
import warnings
from collections import namedtuple
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Sequence

import numpy as np
from sage.all import (
    SR, QQ, I as _I, PolynomialRing, RealField, matrix, factorial, sqrt as _sqrt,
    exp as _exp, abs_symbolic as _abs,
)
from sage.rings.rational import Rational as _Rational
from sage.rings.integer import Integer as _Integer
from sage.rings.real_mpfr import RealNumber as _RealNumber

__all__ = [
    'SymbolicOutError', 'UnsupportedModelError', 'NonRationalPropagatorError',
    'ClosedFormUnavailable', 'TreeCovariance', 'SymbolicCallable',
    'KernelPairing', 'Quadrature', 'tree_covariance',
    'tree_covariance_from_result', 'tree_covariance_from_matrices',
    'noise_matrix_from_theory', 'pair_with_kernel', 'inverse_transform',
    'numeric_pair_with_kernel', 'numeric_inverse_transform', 'to_sympy',
    'from_sympy', 'to_latex', 'to_callable', 'to_json', 'from_json',
    # stage C2
    'AssumptionViolation', 'Mode', 'RootOf', 'Assumption', 'ModeSum',
    'Kernel', 'KernelIntegral', 'tree_modes', 'integrate_against', 'SIDES',
]

FORMAT_TAG = 'daedalus.symbolic_out/1'

# Ring generator standing for s = i*omega, and a pad generator that keeps the
# polynomial ring multivariate (uniform ``dict()`` / ``factor()`` behaviour).
_S = 'zs__'
_PAD = 'pad__'

# Numerical tolerance for deciding a pole sits ON the real axis.
_REAL_AXIS_TOL = 1e-9


# ═════════════════════════════════════════════════════════════════════════════
# Errors
# ═════════════════════════════════════════════════════════════════════════════

class SymbolicOutError(ValueError):
    """Base class of every error raised by this module."""


class UnsupportedModelError(SymbolicOutError, NotImplementedError):
    """The model is outside what the extraction covers (spatial, non-local
    noise, ...)."""


class NonRationalPropagatorError(SymbolicOutError):
    """The propagator or a kernel is not a rational function of omega."""


class ClosedFormUnavailable(SymbolicOutError):
    """The residue computation cannot be done in closed form (high-degree
    irreducible factor, undecidable pole location); use the numeric fallback
    with parameter values."""


# ═════════════════════════════════════════════════════════════════════════════
# Rational-function bridge  SR  <->  Frac(QQ[s, params])   with  s = i*omega
# ═════════════════════════════════════════════════════════════════════════════
#
# Every entry of K_ft, G_ft and the noise matrix is rational in omega with real
# coefficients in the parameters.  Writing s = i*omega makes the coefficients
# REAL rational functions of (s, params): complex conjugation then becomes
# s -> -s, and field arithmetic gives canonical (fully cancelled) forms.

def _opname(op) -> str:
    """'add' / 'mul' / 'pow' for the Sage SR operators (``add_vararg``,
    ``mul_vararg``, ``operator.pow``), else the raw name."""
    n = getattr(op, '__name__', '')
    return n.replace('_vararg', '')


def _qq_from_numeric(e) -> Any:
    """Exact rational from a numeric SR atom; floats map to the simplest
    rational inside their rounding interval (0.78 -> 39/50)."""
    x = e.pyobject()
    if isinstance(x, (_Integer, _Rational)):
        return QQ(x)
    if isinstance(x, _RealNumber):
        return QQ(x.simplest_rational())
    if isinstance(x, float):
        return QQ(RealField(53)(x).simplest_rational())
    # a real number that arrived as a Gaussian rational (I*I products)
    try:
        if x.imag() == 0:
            return QQ(x.real())
    except (AttributeError, TypeError, ValueError):
        pass
    raise NonRationalPropagatorError(
        f'complex or non-rational numeric coefficient {e!r}: the covariance '
        f'requires rational functions of omega with real coefficients in '
        f's = i*omega (for a C0 entry: C0(-omega) = conj(C0(omega)) at real '
        f'parameters).')


class _Bridge:
    """Conversion between Sage SR expressions in ``omega`` and the field
    ``QQ(s, params...)``."""

    def __init__(self, exprs: Sequence, omega):
        self.omega = omega
        self.s = SR.var(_S)
        syms: dict[str, Any] = {}
        pows: dict[str, tuple] = {}
        seen: set = set()
        for e in exprs:
            self._scan(SR(e), syms, pows, seen)
        syms.pop(str(omega), None)
        for reserved in (_S, _PAD):
            if reserved in syms:
                raise SymbolicOutError(
                    f'parameter name {reserved!r} is reserved by '
                    f'api.symbolic_out')
        self.sym_of = syms
        self.pow_of = pows
        names = [_S, _PAD] + sorted(syms) + sorted(pows)
        try:
            self.R = PolynomialRing(QQ, names)
        except Exception as exc:                       # bad identifier etc.
            raise SymbolicOutError(
                f'cannot build the polynomial ring for symbols {names}: '
                f'{exc}') from exc
        self.F = self.R.fraction_field()
        self.s_gen = self.R.gen(0)
        self._names = names

    # ── scan: collect symbols and fractional powers of symbols ──────────
    def _scan(self, e, syms, pows, seen):
        if e in seen:
            return
        seen.add(e)
        op = e.operator()
        if op is None:
            if e.is_symbol():
                syms[str(e)] = e
            return
        if _opname(op) == 'pow':
            base, ex = e.operands()
            if base.is_symbol() and ex.is_numeric():
                x = ex.pyobject()
                if isinstance(x, _Rational) and x.denominator() != 1:
                    nm = (f'pw__{base}__{"m" if x < 0 else "p"}'
                          f'{abs(x.numerator())}_{x.denominator()}')
                    pows[nm] = (base, QQ(x))
        for o in e.operands():
            self._scan(o, syms, pows, seen)

    # ── SR -> field ─────────────────────────────────────────────────────
    def to_field(self, expr):
        e = SR(expr).subs({self.omega: -_I * self.s})
        return self._conv(e, {})

    def _conv(self, e, memo):
        if e in memo:
            return memo[e]
        op = e.operator()
        if op is None:
            if e.is_symbol():
                name = str(e)
                if name == _S:
                    out = self.F(self.s_gen)
                else:
                    out = self.F(self.R(name))
            elif e.is_numeric():
                out = self.F(_qq_from_numeric(e))
            else:
                raise NonRationalPropagatorError(
                    f'non-rational constant {e!r} in the propagator.')
        elif _opname(op) == 'add':
            out = self.F(0)
            for o in e.operands():
                out += self._conv(o, memo)
        elif _opname(op) == 'mul':
            out = self.F(1)
            for o in e.operands():
                out *= self._conv(o, memo)
        elif _opname(op) == 'pow':
            base, ex = e.operands()
            if ex.is_numeric() and ex.is_integer():
                out = self._conv(base, memo) ** int(ex)
            elif ex.is_numeric() and base.is_symbol():
                x = QQ(ex.pyobject())
                nm = (f'pw__{base}__{"m" if x < 0 else "p"}'
                      f'{abs(x.numerator())}_{x.denominator()}')
                if nm not in self._names:
                    raise NonRationalPropagatorError(
                        f'non-rational power {e!r} of omega (or of a '
                        f'compound expression): the covariance requires a '
                        f'rational function of omega.')
                out = self.F(self.R(nm))
            else:
                raise NonRationalPropagatorError(
                    f'non-rational power {e!r}: the covariance requires a '
                    f'rational function of omega (e.g. no square roots of '
                    f'omega-dependent terms).')
        else:
            raise NonRationalPropagatorError(
                f'non-rational function {op} in {e!r}: the propagator must '
                f'be a rational function of omega (a delay exp(-I*omega*d) or '
                f'a transcendental kernel image cannot be handled in closed '
                f'form).')
        memo[e] = out
        return out

    # ── field -> SR ─────────────────────────────────────────────────────
    def _gen_sr(self, name):
        if name in self.sym_of:
            return self.sym_of[name]
        base, q = self.pow_of[name]
        return base ** q

    def poly_to_sr(self, P, *, in_omega=True):
        """Polynomial -> SR; ``s`` becomes ``I*omega`` (or must be absent)."""
        total = SR(0)
        for exps, c in P.dict().items():
            term = SR(c)
            for name, ex in zip(self._names, exps):
                if ex == 0 or name == _PAD:
                    continue
                if name == _S:
                    if not in_omega:
                        raise SymbolicOutError('polynomial depends on omega')
                    term *= (_I * self.omega) ** int(ex)
                else:
                    term *= self._gen_sr(name) ** int(ex)
            total += term
        return total

    def frac_to_sr(self, f):
        num = self.poly_to_sr(f.numerator()).expand()
        den = self.poly_to_sr(f.denominator()).expand()
        if den == 1:
            return num
        return num / den

    def neg_s(self, f):
        sub = {self.s_gen: -self.s_gen}
        return (self.F(f.numerator().subs(sub))
                / self.F(f.denominator().subs(sub)))

    def deg_s(self, P):
        return int(P.degree(self.s_gen))


def _is_real_valued_parameters(values: dict) -> None:
    for k, v in values.items():
        try:
            if abs(complex(v).imag) > 0:
                raise SymbolicOutError(
                    f'parameter {k!r}={v!r} is complex; the covariance '
                    f'G D G^dagger assumes real parameters.')
        except TypeError:
            continue


# ═════════════════════════════════════════════════════════════════════════════
# Building C0
# ═════════════════════════════════════════════════════════════════════════════

def noise_matrix_from_theory(ft, model: Optional[dict] = None):
    """Response-space noise matrix ``D`` from the (2,0) sector of an expanded
    ``FieldTheory``: the action contains ``-(1/2) xt_r D_{rs} xt_s``.

    Returns ``(D, resp_names)`` with ``D`` a symmetric Sage ``SR`` matrix.
    Raises ``UnsupportedModelError`` for non-local (kernel) noise."""
    from engine.core.vertices import extract_source_types, NoiseSourceType
    ft._require_expanded()
    for st in extract_source_types(ft):
        if isinstance(st, NoiseSourceType):
            raise UnsupportedModelError(
                'the model carries a non-local noise cumulant kernel that is '
                'not Markov-embedded (NoiseSourceType); the Fourier-domain '
                'noise matrix is a convolution kernel there.  Declare the '
                'colored noise with declare_cgf_term(...) so the pipeline '
                'embeds it, or use the time-domain pipeline.')
    R = ft.ring()
    names = [str(g) for g in R.gens()]
    n_t = ft._n_tilde
    resp = names[:n_t]
    Dm = [[SR(0)] * n_t for _ in range(n_t)]
    sector = ft._by_tp.get((2, 0), R.zero())
    for exps, coeff in sector.dict().items():
        idx = [(i, int(e)) for i, e in enumerate(exps) if e]
        if sum(e for _, e in idx) != 2 or any(i >= n_t for i, _ in idx):
            raise UnsupportedModelError(
                f'unexpected monomial in the (2,0) noise sector: {idx}')
        c = SR(coeff)
        if len(idx) == 1:
            r = idx[0][0]
            Dm[r][r] += -2 * c
        else:
            (r, _), (s_, _) = idx
            Dm[r][s_] += -c
            Dm[s_][r] += -c
    return matrix(SR, Dm), resp


def _reject_spatial(model: dict) -> None:
    sp = model.get('spatial')
    if sp and (sp.get('dim') if isinstance(sp, dict) else True):
        raise UnsupportedModelError(
            f"model {model.get('name', '<unnamed>')!r} is spatial: its "
            f"propagator carries the inert Laplacian operator and depends on "
            f"the wavevector k, so there is no scalar C0(omega).  "
            f"api.symbolic_out covers temporal models only; use "
            f"compute_cumulants(spatial_grid=...) for C(x, tau).")


def _expanded_theory(model: dict, taylor_order: int, use_cache: bool,
                     verbose: bool):
    """FieldTheory expansion exactly as ``compute_cumulants`` stage 1 does it
    (expand-cache aware, sanity-checked)."""
    from engine.core.field_theory import FieldTheory
    from api import _expand_cache as _ec
    ft = FieldTheory(model, taylor_order=taylor_order)
    hit = False
    if use_cache:
        cached_order = _ec.find_best_cached_order(model, taylor_order)
        if cached_order is not None:
            _ec.prepare_for_load(ft)
            hit = _ec.load_expand(model, ft, target_order=taylor_order,
                                  cached_order=cached_order, verbose=verbose)
    if not hit:
        ft.expand()
        if use_cache:
            try:
                _ec.save_expand(model, ft, verbose=verbose)
            except Exception:
                pass
    if not ft.sanity_check(verbose=verbose):
        raise SymbolicOutError('FieldTheory.sanity_check() failed')
    return ft


def _model_fundamental(model: dict, parameters: Optional[dict]) -> dict:
    """Default parameter dict, as ``compute_cumulants`` falls back to it."""
    if parameters:
        return dict(parameters)
    pspecs = model.get('parameters', []) or []
    fund = {p['name']: p['default'] for p in pspecs
            if p.get('default') is not None}
    missing = sorted(p['name'] for p in pspecs
                     if p.get('default') is None and not p.get('mean_field'))
    if missing:
        raise SymbolicOutError(
            'no parameters= given and these parameters declare no default: '
            + ', '.join(missing))
    return fund


def _solve_mean_field(ft, model: dict, fundamental: dict) -> dict:
    """``num_params`` of the pipeline's MF solve ({SR symbol: float})."""
    if model.get('equations'):
        from api._mean_field_dae import solve_mean_field_dae_compat
        mf = solve_mean_field_dae_compat(ft, model, fundamental,
                                         fixed_point_index=0, n_starts=64,
                                         seed_box=None, verbose=False)
    else:
        from api._mean_field import solve_mean_field
        mf = solve_mean_field(ft, model, fundamental, verbose=False)
    return mf['num_params']


def _clean_float(x: float):
    """Numbers from the MF solver: snap round-off dust to exactly 0."""
    x = float(x)
    return 0 if abs(x) < 1e-12 else x


def _as_name_dict(d: Optional[dict]) -> dict:
    out = {}
    for k, v in (d or {}).items():
        out[str(k)] = v
    return out


def tree_covariance_from_matrices(G, D, omega, *, K=None,
                                  phys_names: Sequence[str] | None = None,
                                  resp_names: Sequence[str] | None = None,
                                  values: Optional[dict] = None,
                                  model_name: str = '<matrices>',
                                  reference: Optional[dict] = None
                                  ) -> 'TreeCovariance':
    """Core constructor: ``G`` (phys x resp), ``D`` (resp x resp) and
    ``omega``, all Sage SR.  If ``G`` is ``None`` it is ``K^{-1}``.  When both
    are given the identity ``K G = 1`` is verified in the rational field."""
    if G is None and K is None:
        raise SymbolicOutError('need G or K')
    omega = SR(omega)
    vals = _as_name_dict(values)
    _is_real_valued_parameters(vals)
    exprs_all = []
    if G is not None:
        exprs_all += list(G.list())
    if K is not None:
        exprs_all += list(K.list())
    exprs_all += list(D.list())
    free = {str(v): v for e in exprs_all for v in SR(e).variables()}
    free.pop(str(omega), None)
    unknown = sorted(k for k in vals if k not in free)
    if unknown:
        raise SymbolicOutError(
            f'values= names {unknown} do not occur in G, K or D; known '
            f'symbols: {sorted(free)}')
    sub = {free[k]: SR(_clean_float(v) if isinstance(v, float) else v)
           for k, v in vals.items()}

    def _s(m):
        return m.apply_map(lambda e: SR(e).subs(sub)) if sub else m
    Gs = _s(G) if G is not None else None
    Ks = _s(K) if K is not None else None
    Ds = _s(D)
    n = (Gs or Ks).nrows()
    bridge = _Bridge([e for m in (Gs, Ks, Ds) if m is not None
                      for e in m.list()], omega)
    F = bridge.F

    def to_F(m, what):
        rows = []
        for i in range(m.nrows()):
            row = []
            for j in range(m.ncols()):
                try:
                    row.append(bridge.to_field(m[i, j]))
                except NonRationalPropagatorError as exc:
                    raise NonRationalPropagatorError(
                        f'{what}[{i},{j}]: {exc}') from None
                except (ZeroDivisionError, ArithmeticError) as exc:
                    raise SymbolicOutError(
                        f'{what}[{i},{j}] is singular: {exc}') from exc
            rows.append(row)
        return matrix(F, rows)

    Df = to_F(Ds, 'noise matrix D')
    if Gs is not None:
        Gf = to_F(Gs, 'G')
        if Ks is not None:
            Kf = to_F(Ks, 'K')
            if Kf * Gf != matrix(F, n, n, 1):
                raise SymbolicOutError(
                    'K * G != identity: the supplied G is not the inverse of '
                    'K (row/column convention mismatch?)')
    else:
        Kf = to_F(Ks, 'K')
        try:
            Gf = Kf.inverse()
        except ZeroDivisionError as exc:
            raise SymbolicOutError('kernel matrix K is singular') from exc
    if Df.nrows() != Gf.ncols() or Df != Df.transpose():
        raise SymbolicOutError(
            'the noise matrix must be symmetric with the size of the '
            'response space')
    Gneg = Gf.apply_map(bridge.neg_s)
    C0f = Gf * Df * Gneg.transpose()
    ph = list(phys_names) if phys_names else [f'x{i}' for i in range(n)]
    rs = list(resp_names) if resp_names else [f'xt{i}' for i in range(n)]
    return TreeCovariance(
        model_name=model_name, omega=omega, phys_names=ph, resp_names=rs,
        bridge=bridge, G_f=Gf, D_f=Df, C0_f=C0f,
        mean_field_values={}, values=dict(vals), reference=reference)


def tree_covariance(model: dict, *, values: Optional[dict] = None,
                    mean_field: Any = None, parameters: Optional[dict] = None,
                    taylor_order: int = 2, use_cache: bool = True,
                    verbose: bool = False) -> 'TreeCovariance':
    """Symbolic tree-level covariance of ``model`` (no diagram enumeration).

    Builds the propagator exactly as ``compute_cumulants`` stage 2 does
    (``FieldTheory.expand`` then ``build_propagator``) and reads the noise
    matrix from the (2,0) action sector.

    Parameters
    ----------
    values : dict name -> number, optional
        Symbols to fix before simplification, e.g. ``{'D': 1}``; all others
        stay symbolic.  Floats become the simplest rational in their rounding
        interval.
    mean_field : None | 'solve' | dict
        ``None`` keeps the saddle symbols (``xstar1`` ...) symbolic;
        ``'solve'`` runs the pipeline's mean-field solve at ``parameters`` (the
        model defaults if omitted) and substitutes the saddle symbols;
        a dict gives them explicitly (``{'xstar1': 0}``).
    """
    _reject_spatial(model)
    ft = _expanded_theory(model, taylor_order, use_cache, verbose)
    from api._propagator import build_propagator
    prop = build_propagator(ft, model, use_cache=use_cache, verbose=verbose)
    return _from_theory(model, ft, prop, values=values, mean_field=mean_field,
                        parameters=parameters)


def tree_covariance_from_result(result: dict, *, model: dict,
                                values: Optional[dict] = None,
                                mean_field: Any = None) -> 'TreeCovariance':
    """Same as :func:`tree_covariance` but reusing the propagator in a
    ``compute_cumulants`` result.  ``model`` is required: the result carries
    the kernel and ``G_ft`` but not the noise sector.

    ``mean_field='result'`` substitutes the saddle symbols from
    ``result['num_params']``; ``'solve'`` and dicts behave as in
    :func:`tree_covariance`."""
    _reject_spatial(model)
    prop = result.get('propagator') if isinstance(result, dict) else None
    if prop is None or prop.get('K_ft') is None:
        raise SymbolicOutError(
            "result has no 'propagator'['K_ft']; pass a dict returned by "
            "compute_cumulants (non-spatial)")
    ft = _expanded_theory(model, 2, False, False)
    mf = mean_field
    cov = _from_theory(model, ft, prop, values=values,
                       mean_field=None if mf == 'result' else mf,
                       parameters=(result.get('config') or {}).get(
                           'parameters'))
    if mf == 'result':
        npar = {str(k): v for k, v in result['num_params'].items()}
        cov = cov.fix_mean_field(npar)
        cov._mf_inputs = {k: v for k, v in npar.items()
                          if not _is_saddle_symbol(k, model)}
    cov._num_params = {str(k): float(v) for k, v in
                       (result.get('num_params') or {}).items()}
    return cov


def _from_theory(model, ft, prop, *, values, mean_field, parameters):
    D, resp = noise_matrix_from_theory(ft, model)
    nf = prop['nf']
    omega = prop['omega']
    names = prop['ring_gen_names']
    resp_names = names[:nf]
    phys_names = names[nf:2 * nf]
    mf_vals: dict = {}
    if mean_field == 'solve':
        fund = _model_fundamental(model, parameters)
        npar = _solve_mean_field(ft, model, fund)
        mf_vals = _mf_symbols_only(model, npar)
    elif isinstance(mean_field, dict):
        mf_vals = _as_name_dict(mean_field)
    elif mean_field is not None:
        raise SymbolicOutError(
            f"mean_field must be None, 'solve' or a dict; got {mean_field!r}")
    allvals = dict(mf_vals)
    allvals.update(_as_name_dict(values))
    G = prop.get('G_ft')
    cov = tree_covariance_from_matrices(
        G, D, omega, K=prop['K_ft'], phys_names=phys_names,
        resp_names=resp_names, values=_restrict(allvals, prop, D, omega),
        model_name=model.get('name', '<unnamed>'))
    cov.mean_field_values = dict(mf_vals)
    cov._model = model
    cov._ft = ft
    cov._parameters = parameters
    if mean_field == 'solve':
        cov._mf_inputs = _expand_param_names(
            _model_fundamental(model, parameters))
    return cov


def _restrict(vals: dict, prop, D, omega) -> dict:
    """Keep the names that actually occur (MF solves return extra symbols)."""
    occurring = {str(v) for m in (prop['K_ft'], D) for e in m.list()
                 for v in SR(e).variables()}
    if prop.get('G_ft') is not None:
        occurring |= {str(v) for e in prop['G_ft'].list()
                      for v in SR(e).variables()}
    return {k: v for k, v in vals.items() if k in occurring}


def _expand_param_names(d: Optional[dict]) -> dict:
    """``{'E': [a, b], 'w': [[..]]}`` -> ``{'E1': a, 'E2': b, 'w11': ..}``
    (the pipeline's symbol names); scalars pass through."""
    out = {}
    for k, v in (d or {}).items():
        if isinstance(v, (list, tuple, np.ndarray)):
            for i, row in enumerate(v):
                if isinstance(row, (list, tuple, np.ndarray)):
                    for j, x in enumerate(row):
                        out[f'{k}{i + 1}{j + 1}'] = x
                else:
                    out[f'{k}{i + 1}'] = row
        else:
            out[str(k)] = v
    return out


def _saddle_bases(model: Optional[dict]) -> Optional[list]:
    """Names of the model's mean-field (saddle) parameters, the ones
    declared with ``mean_field=True`` (``xstar``, ``nEstar`` ...); ``None``
    if the model declares none (then the ``<base>star`` naming rule
    applies)."""
    if not model:
        return None
    bases = [str(p.get('name')) for p in (model.get('parameters') or [])
             if p.get('mean_field')]
    return bases or None


def _is_saddle_symbol(name: str, model: Optional[dict]) -> bool:
    """Whether the symbol ``name`` is a mean-field value: a declared saddle
    parameter, plain or with a population index (``xstar`` / ``xstar1`` /
    ``nEstar2``), or, for models that declare none, a name ending in
    ``star`` plus an optional index -- the convention of
    ``describe_model``."""
    import re
    bases = _saddle_bases(model)
    if bases is None:
        return re.fullmatch(r'.*star\d*', name) is not None
    return any(re.fullmatch(re.escape(b) + r'\d*', name) for b in bases)


def _mf_symbols_only(model: Optional[dict], num_params) -> dict:
    """Entries of the MF ``num_params`` that are saddle values.  The solver
    returns parameters, saddles and Taylor-coefficient symbols together; we
    keep the saddle values (:func:`_is_saddle_symbol`) and let the caller's
    ``values=`` handle ordinary parameters."""
    out = {}
    for k, v in num_params.items():
        name = str(k)
        if _is_saddle_symbol(name, model):
            out[name] = _clean_float(v)
    return out


# ═════════════════════════════════════════════════════════════════════════════
# Exports
# ═════════════════════════════════════════════════════════════════════════════

def _free_param_names(expr, omega=None) -> list[str]:
    om = None if omega is None else str(omega)
    return sorted(str(v) for v in SR(expr).variables() if str(v) != om)


def to_sympy(expr):
    """Sage expression -> sympy expression (``expr._sympy_()``)."""
    return SR(expr)._sympy_()


def from_sympy(sym_expr):
    """sympy expression -> Sage ``SR`` (inverse of :func:`to_sympy`)."""
    return SR(sym_expr)


def to_latex(expr) -> str:
    """LaTeX string (``sage.all.latex``)."""
    from sage.all import latex
    return str(latex(SR(expr)))


class SymbolicCallable:
    """A numpy-callable ``f(omega, **named_parameters)`` built from a Sage
    expression through its sympy image (``sympy.lambdify``, numpy module).

    ``omega`` may be a scalar or an array; parameters are keywords by their
    pipeline names (``mu``, ``tau_g``, ``w12`` ...).  ``defaults`` supplies the
    ones not passed.  The result is a complex ndarray with the shape of
    ``omega``."""

    def __init__(self, expr, omega, params: Optional[Sequence[str]] = None,
                 defaults: Optional[dict] = None):
        import sympy
        self.expression = SR(expr)
        self.omega = SR(omega)
        self.params = (list(params) if params is not None
                       else _free_param_names(self.expression, self.omega))
        extra = set(_free_param_names(self.expression, self.omega)) - set(
            self.params)
        if extra:
            raise SymbolicOutError(
                f'expression has free symbols {sorted(extra)} that are not '
                f'in params={self.params}')
        self.defaults = {k: v for k, v in (defaults or {}).items()
                         if k in self.params}
        self.sympy_expr = to_sympy(self.expression)
        w = sympy.Symbol(str(self.omega))
        ps = [sympy.Symbol(p) for p in self.params]
        by_name = {str(s): s for s in self.sympy_expr.free_symbols}
        self._fn = sympy.lambdify(
            [by_name.get(str(self.omega), w)] +
            [by_name.get(p, q) for p, q in zip(self.params, ps)],
            self.sympy_expr, modules='numpy')

    def __call__(self, omega, **named):
        unknown = set(named) - set(self.params)
        if unknown:
            raise TypeError(f'unknown parameters {sorted(unknown)}; this '
                            f'expression takes {self.params}')
        vals = dict(self.defaults)
        vals.update(named)
        missing = [p for p in self.params if p not in vals]
        if missing:
            raise TypeError(f'missing parameters {missing}')
        w = np.asarray(omega, dtype=complex)
        out = self._fn(w, *[vals[p] for p in self.params])
        return np.broadcast_to(np.asarray(out, dtype=complex), w.shape).copy()

    def __repr__(self):
        return f'SymbolicCallable(params={self.params}, {self.expression})'


def to_callable(expr, omega, params: Optional[Sequence[str]] = None,
                defaults: Optional[dict] = None) -> SymbolicCallable:
    """numpy-callable with named parameters (see :class:`SymbolicCallable`)."""
    return SymbolicCallable(expr, omega, params, defaults)


def to_json(expr, omega, params: Optional[Sequence[str]] = None) -> str:
    """JSON string ``{"format", "variable", "parameters", "expression",
    "sage"}``.  ``expression`` is in python/sympy syntax (``**``), ``sage`` in
    Sage syntax (``^``); both parse back with :func:`from_json`."""
    import sympy
    e = SR(expr)
    plist = (list(params) if params is not None
             else _free_param_names(e, omega))
    return json.dumps({
        'format': FORMAT_TAG,
        'variable': str(omega),
        'parameters': plist,
        'expression': str(to_sympy(e)),
        'sage': str(e),
    })


def from_json(text: str):
    """Rebuild ``(expression, omega, params)`` from :func:`to_json` output
    (Sage ``SR`` expression; symbols named by the JSON, not model objects)."""
    import sympy
    d = json.loads(text) if isinstance(text, (str, bytes)) else dict(text)
    if d.get('format') != FORMAT_TAG:
        raise SymbolicOutError(f"not a {FORMAT_TAG} JSON document")
    names = [d['variable']] + list(d['parameters'])
    loc = {n: sympy.Symbol(n) for n in names}
    loc['I'] = sympy.I
    sym = sympy.sympify(d['expression'], locals=loc)
    expr = SR(sym)
    return expr, SR.var(d['variable']), list(d['parameters'])


# ═════════════════════════════════════════════════════════════════════════════
# TreeCovariance
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class TreeCovariance:
    """Symbolic ``G``, ``D`` and ``C0 = G D G^dagger`` of a temporal model.

    Attributes (matrices are Sage ``SR`` matrices, built lazily):
      ``G``   propagator, rows physical fields, columns response fields;
      ``D``   noise matrix, response x response (``-(1/2) xt D xt`` in S);
      ``C0``  tree-level covariance, physical x physical;
      ``phys_names`` / ``resp_names`` the field labels (``'dx1'``, ``'xt1'``);
      ``omega`` the frequency symbol; ``parameters`` the free symbol names.
    """
    model_name: str
    omega: Any
    phys_names: list
    resp_names: list
    bridge: Any = field(repr=False)
    G_f: Any = field(repr=False)
    D_f: Any = field(repr=False)
    C0_f: Any = field(repr=False)
    mean_field_values: dict = field(default_factory=dict)
    values: dict = field(default_factory=dict)
    reference: Optional[dict] = None
    _num_params: Optional[dict] = field(default=None, repr=False)
    _model: Any = field(default=None, repr=False)
    _ft: Any = field(default=None, repr=False)
    _parameters: Any = field(default=None, repr=False)
    _cache: dict = field(default_factory=dict, repr=False)
    # parameters a mean-field solve was run at (they are baked into the
    # saddle values); ModeSum refuses different values for them
    _mf_inputs: Optional[dict] = field(default=None, repr=False)

    # ── matrices ────────────────────────────────────────────────────────
    def _sr_matrix(self, key, f):
        if key not in self._cache:
            b = self.bridge
            self._cache[key] = matrix(
                SR, [[b.frac_to_sr(f[i, j]) for j in range(f.ncols())]
                     for i in range(f.nrows())])
        return self._cache[key]

    @property
    def G(self):
        return self._sr_matrix('G', self.G_f)

    @property
    def D(self):
        """Noise matrix.  Entries are omega-independent (white noise after
        Markov embedding), so ``frac_to_sr`` only expands the parameters."""
        return self._sr_matrix('D', self.D_f)

    @property
    def C0(self):
        return self._sr_matrix('C0', self.C0_f)

    @property
    def parameters(self) -> list:
        return sorted(str(k) for k in self.bridge.sym_of)

    # ── indexing ────────────────────────────────────────────────────────
    def index(self, field_) -> int:
        """Position of a physical field: name ``'dx1'``, pair ``('dx', 1)``,
        user-facing pair ``('x', 1)`` (via the model naming convention) or an
        integer."""
        if isinstance(field_, (int, np.integer)):
            return int(field_)
        if isinstance(field_, (tuple, list)) and len(field_) == 2:
            nm, pop = field_
            cands = [f'{nm}{pop}']
            conv = (self._model or {}).get('naming_convention')
            if conv is not None:
                from api.access import normalize_external_fields
                cands.append('%s%d' % normalize_external_fields(
                    [field_], naming_convention=conv)[0])
            for c in cands:
                if c in self.phys_names:
                    return self.phys_names.index(c)
            field_ = cands[-1]
        if field_ in self.phys_names:
            return self.phys_names.index(field_)
        raise SymbolicOutError(
            f'unknown physical field {field_!r}; known: {self.phys_names}')

    def entry(self, a, b):
        """``C0_{ab}(omega)`` as a Sage expression."""
        return self.C0[self.index(a), self.index(b)]

    def __getitem__(self, ab):
        return self.entry(*ab)

    def contact(self, a, b):
        """``lim_{omega->inf} C0_{ab}``: the weight of ``delta(tau)`` in
        ``C_{ab}(tau)`` (zero for entries that decay)."""
        return _contact_of(self.bridge, self.C0_f[self.index(a),
                                                  self.index(b)])[0]

    # ── exports ─────────────────────────────────────────────────────────
    def default_values(self) -> dict:
        """Numeric values for the remaining symbols: from a result's
        ``num_params`` if available, else the model defaults and the MF solve
        (cached).  May be empty."""
        if self._num_params:
            return dict(self._num_params)
        if 'defaults' in self._cache:
            return dict(self._cache['defaults'])
        out: dict = {}
        if self._model is not None and self._ft is not None:
            try:
                fund = _model_fundamental(self._model, self._parameters)
                npar = _solve_mean_field(self._ft, self._model, fund)
                out = {str(k): float(v) for k, v in npar.items()
                       if isinstance(v, (int, float, np.floating))}
            except Exception:
                out = {}
        self._cache['defaults'] = out
        return dict(out)

    def sympy(self, a, b):
        return to_sympy(self.entry(a, b))

    def latex(self, a, b) -> str:
        return to_latex(self.entry(a, b))

    def callable(self, a, b, defaults: Any = None) -> SymbolicCallable:
        """numpy callable ``f(omega, **params)`` for ``C0_{ab}``.  ``defaults``:
        ``None`` = none, ``True`` = :meth:`default_values`, or a dict."""
        e = self.entry(a, b)
        d = (self.default_values() if defaults is True
             else (defaults or {}))
        return to_callable(e, self.omega, defaults=d)

    def json(self, a, b) -> str:
        return to_json(self.entry(a, b), self.omega)

    def fix_mean_field(self, mf_values: dict) -> 'TreeCovariance':
        """New TreeCovariance with the saddle symbols replaced (the model's
        ``mean_field=True`` parameters, see :func:`_is_saddle_symbol`)."""
        vals = {k: v for k, v in mf_values.items()
                if k in self.bridge.sym_of and _is_saddle_symbol(k,
                                                                 self._model)}
        return _resubstitute(self, vals)

    # ── analysis ────────────────────────────────────────────────────────
    def inverse_transform(self, a, b, tau=None, **kw):
        """Exact ``C_{ab}(tau)`` by residues; see :func:`inverse_transform`."""
        kw.setdefault('reference', self.reference or self.default_values())
        return inverse_transform(self.entry(a, b), self.omega, tau, **kw)

    def pair_with_kernel(self, a, b, L_tilde, **kw):
        """See :func:`pair_with_kernel`."""
        kw.setdefault('reference', self.reference or self.default_values())
        return pair_with_kernel(self.entry(a, b), L_tilde, self.omega, **kw)

    def modes(self, a, b, *, values: Optional[dict] = None,
              reference: Optional[dict] = None,
              cross_check: bool = True) -> 'ModeSum':
        """``C_{ab}(tau)`` as a mode sum (any denominator degree); see
        :func:`tree_modes`.  The half-plane split is decided at
        ``reference`` (default: the result's ``num_params`` or the model
        defaults with the mean-field solve)."""
        if reference is None:
            reference = self.reference or self.default_values()
        ms = tree_modes(self.entry(a, b), self.omega, values=values,
                        reference=reference, cross_check=cross_check)
        ms._fixed = dict(self._mf_inputs or {})
        ms._fixed.update(_expand_param_names(self.values))
        return ms

    def integrate_against(self, a, b, kernel: 'Kernel', *,
                          values: Optional[dict] = None,
                          reference: Optional[dict] = None,
                          contact: Optional[str] = None) -> 'KernelIntegral':
        """``int L(s) C_{ab}(s) ds`` in closed form; see
        :func:`integrate_against`."""
        return self.modes(a, b, reference=reference).integrate_against(
            kernel, values=values, contact=contact)


def _resubstitute(cov: TreeCovariance, vals: dict) -> TreeCovariance:
    G = cov.G.apply_map(lambda e: SR(e).subs(
        {cov.bridge.sym_of[k]: SR(v) for k, v in vals.items()}))
    D = cov.D.apply_map(lambda e: SR(e).subs(
        {cov.bridge.sym_of[k]: SR(v) for k, v in vals.items()}))
    new = tree_covariance_from_matrices(
        G, D, cov.omega, phys_names=cov.phys_names,
        resp_names=cov.resp_names, model_name=cov.model_name,
        reference=cov.reference)
    new.mean_field_values = dict(cov.mean_field_values)
    new.mean_field_values.update(vals)
    new.values = dict(cov.values)
    new._model, new._ft, new._parameters = cov._model, cov._ft, cov._parameters
    new._num_params = cov._num_params
    new._mf_inputs = cov._mf_inputs
    return new


def _contact_of(bridge: _Bridge, f):
    """``(c_inf, deg_gap)`` for a field element ``f``: ``c_inf`` (SR) is the
    omega -> infinity limit when the degrees match, ``0`` when ``f`` decays;
    ``deg_gap = deg_den - deg_num`` (negative gap raises)."""
    num, den = f.numerator(), f.denominator()
    if num == 0:
        return SR(0), None
    dn, dd = bridge.deg_s(num), bridge.deg_s(den)
    gap = dd - dn
    if gap > 0:
        return SR(0), gap
    if gap < 0:
        raise UnsupportedModelError(
            f'C0 grows like omega^{-gap}: the covariance contains '
            f'derivatives of delta(tau); not representable as a smooth '
            f'function plus one contact term')
    cn = num.coefficient({bridge.s_gen: dn})
    cd = den.coefficient({bridge.s_gen: dd})
    c = bridge.F(cn) / bridge.F(cd)
    return _frac_params(bridge, c), 0


def _frac_params(bridge: _Bridge, c):
    """SR of a field element that does not depend on s."""
    num = bridge.poly_to_sr(c.numerator(), in_omega=False).expand()
    den = bridge.poly_to_sr(c.denominator(), in_omega=False).expand()
    return num / den if den != 1 else num


# ═════════════════════════════════════════════════════════════════════════════
# Residues
# ═════════════════════════════════════════════════════════════════════════════

def _sample_points(names: Sequence[str], reference: Optional[dict],
                   n_samples: int = 12, spread: float = 0.7, seed: int = 0):
    """Reference point followed by log-uniform perturbations of it.  A symbol
    without a reference value is assumed POSITIVE (base value 1)."""
    ref = {k: float(v) for k, v in (reference or {}).items()
           if k in names and np.isfinite(float(v))}
    base = {k: ref.get(k, 1.0) for k in names}
    rng = np.random.default_rng(seed)
    pts = [dict(base)]
    for _ in range(n_samples):
        pts.append({k: (v * float(np.exp(rng.uniform(-spread, spread)))
                        if v != 0 else 0.0)
                    for k, v in base.items()})
    return pts


def _eval_sr(expr, pt: dict, free_syms: dict) -> complex:
    sub = {free_syms[k]: SR(float(v)) for k, v in pt.items()
           if k in free_syms}
    return complex(SR(expr).subs(sub).n(digits=30))


def _sign_over_samples(expr, pts, free_syms, what: str):
    """+1/-1 if ``expr`` (real) has one sign at every sample point."""
    signs = set()
    for p in pts:
        v = _eval_sr(expr, p, free_syms)
        if abs(v.imag) > 1e-9 * max(1.0, abs(v)):
            raise ClosedFormUnavailable(f'{what} is not real at {p}')
        if abs(v.real) < 1e-12:
            raise ClosedFormUnavailable(
                f'{what} vanishes at a sample point {p}')
        signs.add(1 if v.real > 0 else -1)
    if len(signs) != 1:
        raise ClosedFormUnavailable(
            f'sign of {what} is not constant over the sampled parameter '
            f'neighbourhood; pass reference= values inside the region of '
            f'interest')
    return signs.pop()


@dataclass
class _Pole:
    omega_pole: Any            # SR expression
    mult: int
    s_root: Any                # SR expression, s = i*omega
    upper: bool                # Im(omega) > 0  <=>  Re(s) < 0


def _sqrt_exact(poly, bridge: _Bridge, pts, free_syms):
    """Closed form of sqrt(poly) for a REAL polynomial in the parameters,
    valid on the sampled region (factors raised to integer halves, sign
    chosen by the samples)."""
    poly = bridge.R(poly)
    if poly == 0:
        return SR(0)
    fac = poly.factor()
    unit = QQ(fac.unit())
    factors = []
    for p, e in fac:
        p = bridge.R(p)
        if p.is_constant():                 # numeric factor: fold into unit
            unit *= QQ(p.constant_coefficient()) ** int(e)
        else:
            factors.append((p, int(e)))
    out = SR(1)
    neg = unit < 0
    odd = SR(1)
    for p, e in factors:
        p_sr = bridge.poly_to_sr(p, in_omega=False)
        if e // 2:
            sg = _sign_over_samples(p_sr, pts, free_syms, f'factor {p}')
            out *= (sg * p_sr) ** (e // 2)
        if e % 2:
            odd *= p_sr
    if odd != 1:
        sg = _sign_over_samples(odd, pts, free_syms, f'radicand {odd}')
        # sqrt(odd) with the sign of (unit * odd) deciding real vs imaginary
        rad = (-1 if neg else 1) * odd
        sg_total = -sg if neg else sg
        out *= _sqrt(sg_total * rad) * (_I if sg_total < 0 else 1)
        return out * _sqrt(abs(unit))
    # pure number (times perfect squares)
    return out * (_I if neg else 1) * _sqrt(abs(unit))


def _poles_of_denominator(den, bridge: _Bridge, reference, free_syms):
    """Factor ``den`` (poly in s, params) and return
    ``(poles, lead)`` with ``den == lead_poly_const * prod (s - r_j)^{m_j}``
    expressed as SR: ``lead`` is the SR constant ``unit * prod a^m`` (the
    product of leading coefficients and parameter-only factors)."""
    names = sorted(bridge.sym_of)
    pts = _sample_points(names, reference)
    fac = den.factor()
    lead = SR(QQ(fac.unit()))
    poles: list[_Pole] = []
    for g, m in fac:
        d = bridge.deg_s(g)
        if d == 0:
            lead *= bridge.poly_to_sr(g, in_omega=False) ** int(m)
            continue
        if d > 2:
            raise ClosedFormUnavailable(
                f'irreducible factor of degree {d} in omega ({g}): its '
                f'roots have no explicit closed form; use tree_modes / '
                f'TreeCovariance.modes (RootOf modes, any degree), '
                f'pair_with_kernel(..., any_degree=True), or the numeric '
                f'fallback (numeric_inverse_transform / '
                f'numeric_pair_with_kernel) with parameter values')
        coeffs = [bridge.R(g.coefficient({bridge.s_gen: k}))
                  for k in range(d + 1)]
        csr = [bridge.poly_to_sr(c, in_omega=False) for c in coeffs]
        a = csr[d]
        lead *= a ** int(m)
        if d == 1:
            roots = [-csr[0] / a]
        else:
            disc_poly = coeffs[1] ** 2 - 4 * coeffs[2] * coeffs[0]
            root_disc = _sqrt_exact(disc_poly, bridge, pts, free_syms)
            roots = [(-csr[1] + root_disc) / (2 * a),
                     (-csr[1] - root_disc) / (2 * a)]
        for r in roots:
            res_re = {_eval_sr(r, p, free_syms).real for p in pts}
            vals = [_eval_sr(r, p, free_syms).real for p in pts]
            if min(abs(v) for v in vals) < _REAL_AXIS_TOL * max(
                    1.0, max(abs(v) for v in vals)):
                raise ClosedFormUnavailable(
                    f'a pole of the integrand lies on the real omega axis '
                    f'(s-root {r}); the integral diverges or needs a '
                    f'principal-value prescription')
            if len({v < 0 for v in vals}) != 1:
                raise ClosedFormUnavailable(
                    f'the half-plane of the pole s={r} changes over the '
                    f'sampled parameter neighbourhood; pass reference= '
                    f'values inside the region of interest')
            poles.append(_Pole(omega_pole=-_I * r, mult=int(m), s_root=r,
                               upper=vals[0] < 0))
    return poles, lead


def _residue_sum(f_field, bridge: _Bridge, weight, upper: bool, reference,
                 simplify: bool, prefactor=1):
    """``prefactor * sum Res[f(omega) * weight(omega)]`` over the poles of
    ``f`` in the upper (``upper=True``) or lower half-plane.  ``weight`` is an
    SR function of omega (``1`` or ``exp(-I*omega*tau)``).

    With ``simplify`` every pole's term is split as (rational coefficient) x
    (its exponential) so the coefficient is simplified without mangling the
    exponentials."""
    omega = bridge.omega
    num, den = f_field.numerator(), f_field.denominator()
    free_syms = dict(bridge.sym_of)
    poles, lead = _poles_of_denominator(den, bridge, reference, free_syms)
    num_sr = bridge.poly_to_sr(num)             # s -> I*omega
    # den = lead * prod (s - r_j)^m_j = lead * prod (i (omega - w_j))^m_j
    total_mult = sum(p.mult for p in poles)
    const = lead * _I ** total_mult
    total = SR(0)
    for k, pk in enumerate(poles):
        if pk.upper != upper:
            continue
        g = num_sr * weight / const
        for j, pj in enumerate(poles):
            if j != k:
                g = g / (omega - pj.omega_pole) ** pj.mult
        if pk.mult > 1:
            g = g.diff(omega, pk.mult - 1) / factorial(pk.mult - 1)
        term = prefactor * g.subs({omega: pk.omega_pole})
        if simplify:
            ex = SR(weight).subs({omega: pk.omega_pole})
            if ex != 1:
                term = (term / ex).simplify_full() * ex
            else:
                term = term.simplify_full()
        total += term
    return total


def _check_real_axis_free_decay(bridge, f, gap_needed: int, what: str):
    num = f.numerator()
    if num == 0:
        return
    gap = bridge.deg_s(f.denominator()) - bridge.deg_s(num)
    if gap < gap_needed:
        raise SymbolicOutError(
            f'{what}: the integrand decays only like omega^{-gap} (need '
            f'omega^-{gap_needed}); the integral diverges or is only '
            f'conditionally convergent.  A contact (constant) term in C0, '
            f'e.g. Poisson shot noise on dn, must be removed or paired with '
            f'a kernel that decays.')


@dataclass
class KernelPairing:
    """Result of :func:`pair_with_kernel`.

    ``expression``: exact closed form (Sage) or ``None``; ``value``: its
    numeric value at ``values`` / the numeric quadrature fallback;
    ``exact``: True iff ``expression`` is the residue sum; ``abs_error`` and
    ``tolerance``: quadrature error estimate and the requested tolerance
    (``None`` for exact); ``reason``: why the exact route was not used."""
    expression: Any = None
    value: Optional[complex] = None
    exact: bool = False
    abs_error: Optional[float] = None
    tolerance: Optional[tuple] = None
    reason: Optional[str] = None
    assumptions: list = field(default_factory=list)

    def __float__(self):
        return float(np.real(self.value))


def _conjugate_real(L_tilde, omega, bridge_hint=None):
    """conj(L(omega)) for real omega and real parameters: s -> -s."""
    L = SR(L_tilde)
    b = _Bridge([L], omega)
    f = b.to_field(L)
    return b.frac_to_sr(b.neg_s(f))


def pair_with_kernel(C0_entry, L_tilde, omega=None, *,
                     values: Optional[dict] = None,
                     reference: Optional[dict] = None,
                     simplify: bool = True, fallback: bool = True,
                     epsabs: float = 1e-12, epsrel: float = 1e-10,
                     any_degree: bool = True) -> KernelPairing:
    r"""``int d omega / (2 pi)  conj(L_tilde(omega)) * C0_entry(omega)``.

    This is ``int dt L(t) C(-t)`` for a real kernel ``L`` with
    ``L_tilde(omega) = int dt L(t) exp(-i omega t)``.

    Default (since item 40, C2 decision (a)): the stage-C2 engine,
    ``any_degree=True``: residues are taken in K[x]/(p) for every
    irreducible factor p of the denominator (any degree), summed by traces,
    and the half-plane split is stated as conditions on the parameters
    (``assumptions`` of the result) instead of being sampled; ``simplify``
    applies ``simplify_full`` to the closed form.

    ``any_degree=False`` (the pre-C2 route, kept for comparison): when
    ``conj(L_tilde) * C0`` is rational in omega and each irreducible factor
    of its denominator has degree <= 2, the integral is
    ``i * sum_{Im omega_k > 0} Res``.  Which poles lie in the upper half plane
    is decided numerically at ``reference`` and at 12 log-uniform perturbations
    of it (parameters without a reference value are assumed positive); the
    pairing is refused if that decision changes inside the sample.  The
    integrand must decay at least like omega^-2.

    Fallback (``fallback=True``, needs a numeric value for every symbol in
    ``values``): adaptive quadrature of the same integrand
    (:func:`numeric_pair_with_kernel`) with tolerances ``epsabs``/``epsrel``.

    Both routes: the integrand must decay at least like omega^-2 and have
    no pole on the real omega axis.
    """
    omega = SR('omega') if omega is None else SR(omega)
    if any_degree:
        try:
            out = _generic_retry(
                lambda ref: _pair_any_degree(C0_entry, L_tilde, omega,
                                             values=values, reference=ref),
                [C0_entry, L_tilde], omega, reference)
            if simplify and out.expression is not None:
                try:
                    out.expression = SR(out.expression).simplify_full()
                except Exception:               # pragma: no cover
                    pass
            return out
        except (NonRationalPropagatorError, ClosedFormUnavailable) as exc:
            if not fallback:
                raise
            q = numeric_pair_with_kernel(C0_entry, L_tilde, omega,
                                         values=values, epsabs=epsabs,
                                         epsrel=epsrel)
            return KernelPairing(expression=None, value=q.value, exact=False,
                                 abs_error=q.error,
                                 tolerance=(epsabs, epsrel), reason=str(exc))
    expr = SR(C0_entry)
    vals = _as_name_dict(values)
    sub = {}
    if vals:
        free = {str(v): v for v in expr.variables()}
        free.update({str(v): v for v in SR(L_tilde).variables()})
        sub = {free[k]: SR(v) for k, v in vals.items() if k in free}
    reason = None
    try:
        Lc = _conjugate_real(SR(L_tilde).subs(sub), omega)
        integrand = Lc * expr.subs(sub)
        bridge = _Bridge([integrand], omega)
        f = bridge.to_field(integrand)
        _check_real_axis_free_decay(bridge, f, 2, 'pair_with_kernel')
        ref = dict(reference or {})
        ref.update({k: v for k, v in vals.items()})
        exact_expr = _residue_sum(f, bridge, SR(1), True, ref, simplify,
                                  prefactor=_I)
        if simplify:
            exact_expr = exact_expr.simplify_full()
        out = KernelPairing(expression=exact_expr, exact=True)
        if vals:
            try:
                out.value = complex(exact_expr.n(digits=20))
            except Exception:
                pass
        return out
    except (NonRationalPropagatorError, ClosedFormUnavailable) as exc:
        reason = str(exc)
        if not fallback:
            raise
    except SymbolicOutError as exc:
        reason = str(exc)
        raise
    # numeric fallback
    q = numeric_pair_with_kernel(C0_entry, L_tilde, omega, values=values,
                                 epsabs=epsabs, epsrel=epsrel)
    return KernelPairing(expression=None, value=q.value, exact=False,
                         abs_error=q.error, tolerance=(epsabs, epsrel),
                         reason=reason)


def _private_symbols(exprs, keep=()):
    """Map every free symbol of ``exprs`` (except those in ``keep``) to a private, domain-free symbol.

    A Sage symbol is global: a ``positive``/``real`` domain declared for ``g`` by an earlier model builder or test stays attached to
    every later use of ``g``, and the residue/factorisation steps below returned a wrong closed form for an underdamped spectrum
    when ``g`` and ``w0`` carried one (full-suite ordering failure). The computation therefore runs on fresh symbols that nothing
    else has touched; ``fwd`` maps originals to fresh symbols, ``back`` the other way."""
    fwd, back = {}, {}
    keep_names = {str(k) for k in keep}
    for e in exprs:
        for v in SR(e).variables():
            nm = str(v)
            if nm in keep_names or v in fwd:
                continue
            fresh = SR.var(nm + '__c1p')
            fwd[v], back[fresh] = fresh, v
    return fwd, back


def _generic_retry(fn, exprs, omega, reference):
    """Call ``fn(reference)``.  When it fails with
    :class:`AssumptionViolation` because poles coincide at the reference
    point (the pre-C2 routes accepted such a point: they sampled a
    neighbourhood), retry at a nearby generic point: symbols without a
    reference value (taken as 1, i.e. assumed positive) at ``1 + (j + 1)
    sqrt(2) 1e-3``, and, if that is not enough, every reference value
    scaled by ``1 + (j + 1) sqrt(2) 1e-3``.  The half-plane split is then
    decided where the poles are distinct, and every sign condition it states
    must hold at the reference point itself (else ``ClosedFormUnavailable``:
    the move would have changed the split).  The stated assumptions keep
    the coincidence conditions: a value
    that violates them is refused at evaluation (substitute it with
    ``values=`` to make the repeated pole exact)."""
    try:
        return fn(reference)
    except AssumptionViolation as exc0:
        ref = dict(_as_name_dict(reference) or {})
        om = str(omega)
        names = sorted({str(v) for e in exprs for v in SR(e).variables()}
                       - {om})
        orig = dict(ref)
        orig.update({nm: 1.0 for nm in names if nm not in orig})

        def checked(out):
            # the split decided at the moved point must be the reference
            # point's own: every sign condition (not the distinctness ones)
            # must hold there too (review: a pole 1e-4 from coinciding
            # could cross under the move)
            ass = getattr(out, 'assumptions', None)
            if ass is None and isinstance(out, dict):
                ass = out.get('assumptions')
            for a in ass or []:
                if a.relation == '>' and a.kind != 'distinct':
                    try:
                        ok = a.holds(orig)
                    except Exception:
                        ok = False
                    if not ok:
                        raise ClosedFormUnavailable(
                            f'poles coincide at the reference point and no '
                            f'nearby generic point keeps its half-plane '
                            f'split ({a}); give a reference where the poles '
                            f'are distinct, or substitute the values with '
                            f'values=') from exc0
            return out
        missing = [nm for nm in names if nm not in ref]
        if missing:
            for j, nm in enumerate(missing):
                ref[nm] = 1.0 + (j + 1) * 2 ** 0.5 * 1e-3
            try:
                return checked(fn(ref))
            except AssumptionViolation:
                pass
        for j, nm in enumerate(sorted(ref)):
            try:
                ref[nm] = float(ref[nm]) * (1.0 + (j + 1) * 2 ** 0.5 * 1e-3)
            except (TypeError, ValueError):
                pass
        return checked(fn(ref))


def inverse_transform(C0_entry, omega=None, tau=None, *,
                      reference: Optional[dict] = None, simplify: bool = True,
                      legacy: bool = False):
    r"""Exact inverse transform ``C(tau) = (1/2 pi) int d omega C0(omega)
    exp(-i omega tau)`` (any denominator degree).

    Since item 40 this runs on the stage-C2 engine (:func:`tree_modes`): the
    residues are taken per irreducible factor in ``K[x]/(p)``, roots of
    factors of degree >= 3 are :class:`RootOf` symbols, and the half-plane
    split is stated as conditions on the parameters (``'assumptions'``)
    instead of being sampled around ``reference`` (C2 decision (a)).
    ``legacy=True`` runs the pre-C2 residue route (denominator factors of
    degree <= 2, sampled split; kept for comparison).

    Returns a dict with
      ``'positive'``  the expression for ``tau > 0``,
      ``'negative'``  the expression for ``tau < 0``,
      ``'contact'``   ``c_inf``, the weight of ``delta(tau)`` (0 if none),
      ``'even'``      the expression in ``abs(tau)`` if the two sides are
                      mirror images (``C(tau) = C(-tau)``), else ``None``,
      ``'tau'``       the SR lag symbol used (``tau``, or ``tau_lag`` when
                      ``tau`` is a parameter of ``C0_entry``),
      ``'modes'``     the :class:`ModeSum`, ``'roots'`` its :class:`RootOf`
                      symbols, ``'assumptions'`` its conditions.
    A numeric ``tau`` evaluates the matching branch and returns the dict with
    those values substituted."""
    if legacy:
        return _inverse_transform_legacy(C0_entry, omega, tau,
                                         reference=reference,
                                         simplify=simplify)
    om = SR('omega') if omega is None else SR(omega)
    ms = _generic_retry(lambda ref: tree_modes(C0_entry, om, reference=ref),
                        [C0_entry], om, reference)
    t_sym = ms.tau if tau is None or not hasattr(tau, 'is_symbol') \
        else SR(tau)
    pos = ms.expression('positive').subs({ms.tau: t_sym})
    neg = ms.expression('negative').subs({ms.tau: t_sym})
    if simplify:
        try:
            pos, neg = pos.simplify_full(), neg.simplify_full()
        except Exception:                       # pragma: no cover
            pass
    even = None
    if ms.roots:
        # RootOf symbols: decide the mirror symmetry on the modes at the
        # reference point (both sides are exact there)
        try:
            ts = np.array([0.37, 1.3, 4.1])
            a, b = ms.evaluate(ts), ms.evaluate(-ts)
            if np.allclose(a, b, rtol=1e-12, atol=1e-300):
                even = pos.subs({t_sym: _abs(t_sym)})
        except SymbolicOutError:
            even = None
    elif _is_zero(pos - neg.subs({t_sym: -t_sym}), reference):
        even = pos.subs({t_sym: _abs(t_sym)})
    out = {'positive': pos, 'negative': neg, 'contact': ms.contact,
           'even': even, 'tau': t_sym, 'modes': ms, 'roots': ms.roots,
           'assumptions': ms.assumptions}
    if tau is not None and not hasattr(tau, 'is_symbol'):
        t = float(tau)
        if t == 0:
            raise SymbolicOutError(
                'tau = 0 is the join of the two branches; evaluate the '
                'smooth part as the limit of either side (it is continuous '
                'for diagonal entries) or use the Ito convention explicitly')
        branch = (pos if t > 0 else neg).subs({t_sym: t})
        out['value'] = branch
        try:
            out['value_numeric'] = complex(branch.n(digits=20))
        except (TypeError, ValueError):
            pass            # free parameters remain
    return out


def _inverse_transform_legacy(C0_entry, omega=None, tau=None, *,
                              reference: Optional[dict] = None,
                              simplify: bool = True):
    """The pre-C2 route (``legacy=True``); see ``_inverse_transform_impl``.
    Runs on private, domain-free symbols."""
    om = SR('omega') if omega is None else SR(omega)
    fwd, back = _private_symbols([C0_entry], keep=[om])
    fresh_ref = None
    if reference is not None:
        by_name = {str(v): fresh for v, fresh in fwd.items()}
        fresh_ref = {(str(by_name[k]) if k in by_name else k): val for k, val in reference.items()}
    out = _inverse_transform_impl(SR(C0_entry).subs(fwd), om, tau, reference=fresh_ref, simplify=simplify)
    for key, val in list(out.items()):
        if hasattr(val, 'subs') and key != 'tau':
            out[key] = SR(val).subs(back)
    return out


def _inverse_transform_impl(C0_entry, omega=None, tau=None, *,
                            reference: Optional[dict] = None, simplify: bool = True):
    r"""Exact inverse transform  ``C(tau) = (1/2 pi) int d omega C0(omega)
    exp(-i omega tau)`` by residues.

    Returns a dict with
      ``'positive'``  the expression for ``tau > 0``,
      ``'negative'``  the expression for ``tau < 0``,
      ``'contact'``   ``c_inf``, the weight of ``delta(tau)`` (0 if none),
      ``'even'``      the expression in ``abs(tau)`` if the two sides are
                      mirror images (``C(tau) = C(-tau)``), else ``None``,
      ``'tau'``       the SR symbol ``tau`` used.
    A numeric ``tau`` evaluates the matching branch and returns the dict with
    those values substituted."""
    omega = SR('omega') if omega is None else SR(omega)
    expr = SR(C0_entry)
    tau_sym = SR.var('tau') if tau is None or not hasattr(tau, 'is_symbol') \
        else SR(tau)
    bridge = _Bridge([expr], omega)
    f = bridge.to_field(expr)
    c_inf, gap = _contact_of(bridge, f)
    fp = f - _field_const(bridge, f) if gap == 0 else f
    if fp.numerator() != 0:
        _check_real_axis_free_decay(bridge, fp, 1, 'inverse_transform')
    weight = _exp(-_I * omega * tau_sym)
    pos = _residue_sum(fp, bridge, weight, False, reference, simplify,
                       prefactor=-_I)
    neg = _residue_sum(fp, bridge, weight, True, reference, simplify,
                       prefactor=_I)
    even = None
    if _is_zero(pos - neg.subs({tau_sym: -tau_sym}), reference):
        even = pos.subs({tau_sym: _abs(tau_sym)})
    out = {'positive': pos, 'negative': neg, 'contact': c_inf, 'even': even,
           'tau': tau_sym}
    if tau is not None and not hasattr(tau, 'is_symbol'):
        t = float(tau)
        if t == 0:
            raise SymbolicOutError(
                'tau = 0 is the join of the two branches; evaluate the '
                'smooth part as the limit of either side (it is continuous '
                'for diagonal entries) or use the Ito convention explicitly')
        branch = (pos if t > 0 else neg).subs({tau_sym: t})
        out['value'] = branch
        try:
            out['value_numeric'] = complex(branch.n(digits=20))
        except (TypeError, ValueError):
            pass            # free parameters remain
    return out


def _is_zero(expr, reference) -> bool:
    """Symbolic zero test, with a numeric cross-check at sample points when
    the symbolic simplifier cannot decide."""
    expr = SR(expr)
    if bool(expr.simplify_full() == 0):
        return True
    free = {str(v): v for v in expr.variables()}
    pts = _sample_points([k for k in free if k != 'tau'], reference,
                         n_samples=4)
    rng = np.random.default_rng(1)
    for p in pts:
        sub = {free[k]: SR(float(v)) for k, v in p.items() if k in free}
        if 'tau' in free:
            sub[free['tau']] = SR(float(rng.uniform(0.2, 2.0)))
        try:
            val = complex(expr.subs(sub).n(digits=20))
        except Exception:
            return False
        if abs(val) > 1e-9:
            return False
    return True


def _field_const(bridge, f):
    """The omega -> infinity limit of ``f`` as a field constant."""
    num, den = f.numerator(), f.denominator()
    dn, dd = bridge.deg_s(num), bridge.deg_s(den)
    return bridge.F(num.coefficient({bridge.s_gen: dn})) / bridge.F(
        den.coefficient({bridge.s_gen: dd}))


# ═════════════════════════════════════════════════════════════════════════════
# Numeric fallbacks
# ═════════════════════════════════════════════════════════════════════════════

Quadrature = namedtuple(
    'Quadrature', 'value error epsabs epsrel contact subtracted converged')
Quadrature.__doc__ = (
    "Result of a numeric quadrature: ``value`` (complex), ``error`` (sum of "
    "quad's absolute error estimates), the tolerances requested, the "
    "``contact`` constant removed (``subtracted``) and ``converged`` (False "
    "if scipy flagged an IntegrationWarning: the value then only meets the "
    "looser tolerance implied by ``error``).")


def numeric_inverse_transform(f: Callable, tau: float, *,
                              contact: Optional[complex] = None,
                              epsabs: float = 1e-12, epsrel: float = 1e-10,
                              limit: int = 500) -> Quadrature:
    r"""``(1/2 pi) int_{-inf}^{inf} f(omega) exp(-i omega tau) d omega`` by
    adaptive quadrature (scipy ``quad`` with the QAWF cos/sin weights for
    ``tau != 0``) of the smooth part: the constant ``contact`` =
    ``f(inf)`` is subtracted first (estimated from ``f`` at very large omega
    if ``None``).  ``f(omega)`` is a scalar function ``omega -> complex``
    (e.g. a :class:`SymbolicCallable` with its parameters bound).

    Tolerances: ``epsabs`` / ``epsrel`` per quad call; the result carries the
    summed error estimate.  This is the SMOOTH part, i.e. the delta(tau)
    contribution of ``contact`` is not included (pipeline convention)."""
    from scipy.integrate import quad
    scalar = lambda w: complex(np.asarray(f(w)).reshape(-1)[0]) \
        if not np.isscalar(w) else complex(np.asarray(f(w)).reshape(-1)[0])
    if contact is None:
        contact = 0.5 * (scalar(1e9) + scalar(-1e9))
    h = lambda w: scalar(w) - contact
    err = 0.0

    converged = True

    def part(fun, **kw):
        nonlocal err, converged
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            v, e = quad(fun, 0, np.inf, epsabs=epsabs, epsrel=epsrel,
                        limit=limit, **kw)
        if caught:
            converged = False
        err += e
        return v

    tau = float(tau)
    if tau == 0.0:
        re = part(lambda u: (h(u) + h(-u)).real)
        im = part(lambda u: (h(u) + h(-u)).imag)
    else:
        # f(u) e^{-iut} + f(-u) e^{iut},  integrate over u in (0, inf)
        a = lambda u: h(u)
        c = lambda u: h(-u)
        w = abs(tau)
        sg = 1.0 if tau > 0 else -1.0       # e^{-iut} = cos(ut) - i sg sin(|t|u)
        re = (part(lambda u: (a(u) + c(u)).real, weight='cos', wvar=w)
              + sg * part(lambda u: (a(u) - c(u)).imag, weight='sin',
                          wvar=w))
        im = (part(lambda u: (a(u) + c(u)).imag, weight='cos', wvar=w)
              - sg * part(lambda u: (a(u) - c(u)).real, weight='sin',
                          wvar=w))
    return Quadrature(complex(re, im) / (2 * np.pi), err / (2 * np.pi),
                      epsabs, epsrel, contact, True, converged)


def numeric_pair_with_kernel(C0_entry, L_tilde, omega=None, *,
                             values: Optional[dict] = None,
                             epsabs: float = 1e-12, epsrel: float = 1e-10,
                             limit: int = 500) -> Quadrature:
    r"""Numeric ``int d omega/(2 pi) conj(L_tilde) C0`` over the real line
    (scipy ``quad`` on the real and imaginary parts), every parameter bound
    through ``values``.  ``L_tilde`` may be any Sage expression, including
    non-rational ones (delays, ...)."""
    from scipy.integrate import quad
    omega = SR('omega') if omega is None else SR(omega)
    vals = _as_name_dict(values)
    cC = to_callable(SR(C0_entry), omega)
    cL = to_callable(SR(L_tilde), omega)
    kC = {k: vals[k] for k in cC.params if k in vals}
    kL = {k: vals[k] for k in cL.params if k in vals}
    miss = [k for k in cC.params if k not in vals] + \
        [k for k in cL.params if k not in vals]
    if miss:
        raise SymbolicOutError(f'values= is missing {sorted(set(miss))}')

    def g(w):
        return complex(np.conj(cL(w, **kL)) * cC(w, **kC))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        re, e1 = quad(lambda w: g(w).real, -np.inf, np.inf, epsabs=epsabs,
                      epsrel=epsrel, limit=limit)
        im, e2 = quad(lambda w: g(w).imag, -np.inf, np.inf, epsabs=epsabs,
                      epsrel=epsrel, limit=limit)
    return Quadrature(complex(re, im) / (2 * np.pi), (e1 + e2) / (2 * np.pi),
                      epsabs, epsrel, 0.0, False, not caught)


# ═════════════════════════════════════════════════════════════════════════════
# Stage C2: time-domain mode sums for any denominator degree
# ═════════════════════════════════════════════════════════════════════════════
#
# The smooth part of C(tau) = (1/2 pi) int C0(omega) exp(-i omega tau) d omega
# is, with s = i*omega,
#
#     C(tau) = (1/2 pi i) int_{-i oo}^{i oo} C0(s) exp(-s tau) ds
#            = + sum_{Re rho < 0} Res_{s=rho} C0(s) exp(-s tau)      (tau < 0)
#            = - sum_{Re rho > 0} Res_{s=rho} C0(s) exp(-s tau)      (tau > 0)
#
# so every pole rho (a root of an irreducible factor p of the denominator, with
# multiplicity m) gives the terms  coeff * tau^j * exp(-rho tau), j < m, on the
# side its half-plane selects.  The residue is computed ONCE per factor in the
# field L = K[x]/(p(x)), K = QQ(parameters): it is a polynomial c(x) of degree
# < deg p with coefficients rational in the parameters, valid at every root of
# p (Galois conjugates share it).  Degree 1 and 2 roots are written out;
# degree >= 3 roots are ``RootOf`` symbols.  A sum over all roots of p that lie
# on one side is the trace Tr_{L/K} c, a rational function of the parameters:
# kernel integrals are exact rational functions whenever each factor is
# one-sided, which is always the case for C0 = G D G^dagger of a stable model.
#
# The half-plane split of each factor is decided by the Routh array (or a
# Sturm chain for even factors) at a reference point, and the signs it needs
# are returned as explicit conditions on the parameters (``Assumption``):
# within the region where they hold the split -- hence the mode list -- is
# the same (Routh-Hurwitz theorem).  Distinct-root conditions (discriminants,
# resultants) are stated too.  Parameter sampling remains only as a
# cross-check of the Routh count against numeric roots.

SIDES = ('positive', 'negative')
_CC = None


def _cc():
    global _CC
    if _CC is None:
        from sage.all import ComplexField
        _CC = ComplexField(256)
    return _CC


class AssumptionViolation(SymbolicOutError):
    """The parameter values lie outside the region in which a symbolic
    mode sum (or kernel integral) was derived."""


@dataclass(frozen=True)
class Mode:
    """One term ``coeff * tau**power * exp(rate * tau)`` of a covariance,
    valid for ``tau > 0`` (``side='positive'``) or ``tau < 0``
    (``'negative'``).  ``coeff`` and ``rate`` are Sage expressions (symbolic
    modes) or complex numbers (numeric modes)."""
    coeff: Any
    rate: Any
    power: int
    side: str


@dataclass(frozen=True)
class RootOf:
    """``symbol`` stands for the root number ``index`` (0-based, roots sorted
    by increasing real part, then imaginary part) of ``polynomial``, a Sage
    polynomial in ``symbol`` whose coefficients are the parameters.  The
    root is a pole location in ``s = i*omega``; a mode built on it has rate
    ``-symbol``."""
    symbol: Any
    polynomial: Any
    index: int
    side: str

    def __str__(self):
        return f'{self.symbol} = RootOf({self.polynomial}, {self.index})'


@dataclass(frozen=True)
class Assumption:
    """``expression relation 0`` with ``relation`` in ``'>'``, ``'!='``.

    ``kind``: ``'stability'`` (Routh/Sturm sign that fixes the half-plane
    split of a factor), ``'distinct'`` (discriminant or resultant: no
    coincident poles), ``'nonzero'`` (a parameter-only denominator factor),
    ``'kernel'`` (decay of a kernel term).  ``origin`` names the factor."""
    expression: Any
    relation: str
    kind: str
    origin: str
    _poly: Any = field(default=None, repr=False, compare=False)
    _ctx: Any = field(default=None, repr=False, compare=False)

    def __str__(self):
        return f'{self.expression} {self.relation} 0   [{self.kind}: {self.origin}]'

    def value(self, values: dict) -> complex:
        """The expression at ``values`` (names -> numbers)."""
        return complex(self._ctx.eval_P(self._poly, self._ctx.point(values)))

    def holds(self, values: dict) -> bool:
        v = self._ctx.eval_P(self._poly, self._ctx.point(values))
        if self.relation == '>':
            return bool(v.real() > 0)
        return bool(v != 0)


class _AlgCtx:
    """Coefficient field ``K = QQ(params)`` and ``U = K[x]`` built from a
    bridge (same symbol order; the s-generator becomes ``x``), plus exact
    conversions to SR and high-precision evaluation."""

    def __init__(self, bridge: '_Bridge', name_map: Optional[dict] = None,
                 back: Optional[dict] = None):
        self.bridge = bridge
        self.back = dict(back or {})        # private symbol -> user symbol
        self.pnames = list(bridge._names[1:])
        # explicit n: multivariate even with no parameters (only the pad)
        self.P = PolynomialRing(QQ, len(self.pnames), self.pnames)
        self.K = self.P.fraction_field()
        self.U = PolynomialRing(self.K, 'x__')
        self.x = self.U.gen()
        # private ring name -> user-facing name (values are given by these)
        self.name_map = dict(name_map or {})

    # ── conversions ─────────────────────────────────────────────────────
    def from_R(self, f):
        by_s: dict = {}
        for exps, c in f.dict().items():
            e = tuple(exps)
            by_s.setdefault(int(e[0]), {})[e[1:]] = c
        return self.U({k: self.K(self.P(d)) for k, d in by_s.items()})

    def F_to_K(self, f):
        """A bridge field element that does not depend on s, as a K element."""
        n, d = self.from_R(f.numerator()), self.from_R(f.denominator())
        if n.degree() > 0 or d.degree() > 0:
            raise SymbolicOutError('expected an omega-independent quantity')
        return n[0] / d[0]

    def P_to_sr(self, poly):
        poly = self.P(poly)
        if poly.is_constant():
            return SR(poly.constant_coefficient())
        if not hasattr(self, '_sr_gens'):
            self._sr_gens = [SR(0) if n == _PAD else self.bridge._gen_sr(n)
                             for n in self.pnames]
        return SR(poly(*self._sr_gens))

    def K_to_sr(self, c):
        c = self.K(c)
        n, d = self.P_to_sr(c.numerator()), self.P_to_sr(c.denominator())
        return n if d == 1 else n / d

    def L_to_sr(self, g, root_sr):
        """SR of the element g(x) of K[x]/(p) at the root ``root_sr``."""
        total = SR(0)
        for i, c in enumerate(g.list()):
            if c != 0:
                total += self.K_to_sr(c) * root_sr ** i
        return total

    # ── evaluation ──────────────────────────────────────────────────────
    def point(self, values: dict) -> list:
        """CC(256) value per ring generator from ``values`` keyed by the
        user-facing names."""
        CC = _cc()
        out = []
        missing = []
        for name in self.pnames:
            if name == _PAD:
                out.append(CC(0))
                continue
            if name in self.bridge.sym_of:
                user = self.name_map.get(name, name)
                if user not in values:
                    missing.append(user)
                    out.append(CC(0))
                    continue
                out.append(CC(SR(values[user])) if not isinstance(
                    values[user], (int, float, complex, np.number))
                    else CC(complex(values[user])))
            else:                                   # fractional power
                base, q = self.bridge.pow_of[name]
                user = self.name_map.get(str(base), str(base))
                if user not in values:
                    missing.append(user)
                    out.append(CC(0))
                    continue
                out.append(CC(complex(values[user])) ** CC(q))
        if missing:
            raise SymbolicOutError(
                f'values= is missing {sorted(set(missing))}')
        return out

    def eval_P(self, poly, pt):
        return self.P(poly)(*pt) if poly != 0 else _cc()(0)

    def eval_K(self, c, pt):
        c = self.K(c)
        return self.eval_P(c.numerator(), pt) / self.eval_P(c.denominator(),
                                                            pt)

    def eval_U(self, g, pt) -> list:
        """Coefficient list (CC) of g at the point."""
        return [self.eval_K(c, pt) for c in g.list()]

    def numeric_roots(self, p, pt) -> list:
        """Roots of p at the point, sorted by (real, imag)."""
        from sage.all import PolynomialRing as _PR
        CC = _cc()
        coeffs = self.eval_U(p, pt)
        poly = _PR(CC, 'z')(coeffs)
        rts = poly.roots(multiplicities=False)
        if len(rts) != p.degree():
            raise SymbolicOutError(
                f'numeric root finding returned {len(rts)} roots for a '
                f'degree-{p.degree()} factor (leading coefficient zero at '
                f'these values?)')
        return sorted(rts, key=lambda z: (float(z.real()), float(z.imag())))


def _hasse(g, k: int):
    """k-th Hasse derivative g^(k)/k! of a U polynomial."""
    if k == 0:
        return g
    return g.derivative(k) * (~g.parent().base_ring()(factorial(k)))


def _inv_mod(a, p):
    a = a % p
    if a == 0:
        raise ZeroDivisionError('not invertible modulo the factor')
    g, u, _ = a.xgcd(p)
    if g.degree() != 0:
        raise ZeroDivisionError('not invertible modulo the factor')
    return (u * (~g[0])) % p


def _series_mul(A, B, p, m):
    out = [p.parent()(0)] * m
    for i in range(m):
        if A[i] == 0:
            continue
        for j in range(m - i):
            if B[j] != 0:
                out[i + j] += A[i] * B[j]
    return [o % p for o in out]


def _laurent(N, Q, p, m: int) -> list:
    """Taylor coefficients h_0..h_{m-1} (elements of K[x]/(p)) of
    ``h(eps) = N(rho+eps) / (Q(rho+eps) * (p(rho+eps)/eps)^m)`` at a root rho
    of the irreducible factor p; ``N/(p^m Q)`` near rho is h(eps)/eps^m."""
    num = [_hasse(N, k) % p for k in range(m)]
    q = [_hasse(Q, k) % p for k in range(m)]
    p1 = [_hasse(p, k + 1) % p for k in range(m)]
    S = q
    for _ in range(m):
        S = _series_mul(S, p1, p, m)
    inv0 = _inv_mod(S[0], p)
    invs = [inv0]
    for k in range(1, m):
        acc = p.parent()(0)
        for i in range(1, k + 1):
            acc += S[i] * invs[k - i]
        invs.append((-inv0 * acc) % p)
    return _series_mul(num, invs, p, m)


def _trace(g, p):
    """Tr_{L/K} of g(x) mod p: sum of g over all roots of p (power sums by
    Newton's identities)."""
    K = p.parent().base_ring()
    d = p.degree()
    lc = p.leading_coefficient()
    b = [c / lc for c in p.list()]               # monic coefficients b_0..b_d
    ps = [K(d)]
    for k in range(1, d):
        acc = k * b[d - k]
        for i in range(1, k):
            acc += b[d - i] * ps[k - i]
        ps.append(-acc)
    g = g % p
    return sum((c * ps[i] for i, c in enumerate(g.list())), K(0))


def _sign_of(v) -> int:
    r = float(v.real())
    return 0 if r == 0 else (1 if r > 0 else -1)


@dataclass
class _Family:
    """An irreducible denominator factor and its residue data."""
    index: int
    p: Any                 # U polynomial
    p_R: Any               # the same factor in the bridge ring
    mult: int
    laurent: list          # h_0..h_{m-1} in K[x]/(p)
    n_neg: int = 0         # roots with Re < 0 (left half plane)
    method: str = ''

    @property
    def deg(self):
        return int(self.p.degree())

    def side_of(self, k: int) -> str:
        return 'negative' if k < self.n_neg else 'positive'


def _routh_first_column(p):
    """First column of the Routh array of p (K coefficients), or ``None`` if
    some entry is identically zero."""
    d = p.degree()
    a = p.list()
    K = p.parent().base_ring()
    row0 = [a[d - 2 * i] for i in range(d // 2 + 1)]
    row1 = [a[d - 1 - 2 * i] for i in range((d - 1) // 2 + 1)]
    rows = [row0, row1]
    first = [row0[0], row1[0]]
    if row1[0] == 0:
        return None
    for _ in range(d - 1):
        A, B = rows[-2], rows[-1]
        new = []
        for i in range(len(A) - 1):
            bn = B[i + 1] if i + 1 < len(B) else K(0)
            new.append((B[0] * A[i + 1] - A[0] * bn) / B[0])
        if not new or new[0] == 0:
            return None
        rows.append(new)
        first.append(new[0])
    return first


def _sturm_conditions_even(p):
    """For an even factor p(x) = r(x^2): the Sturm chain of r and the K
    elements whose signs fix the number of roots of r in (-oo, 0]."""
    K = p.parent().base_ring()
    U = p.parent()
    coeffs = p.list()
    r = U([coeffs[2 * i] for i in range(len(coeffs) // 2 + 1)
           if 2 * i < len(coeffs)])
    chain = [r, r.derivative()]
    while chain[-1].degree() > 0:
        rem = -(chain[-2] % chain[-1])
        if rem == 0:
            break
        chain.append(rem)
    return r, chain


def _condition_items(r, sigma: int, ctx: _AlgCtx, ref_pt, kind, origin):
    """``sigma * r > 0`` for a K element r, split into factor sign
    conditions that hold at the reference point (sufficient conditions;
    squared factors become ``!= 0``)."""
    r = ctx.K(r)
    poly = ctx.P(r.numerator() * r.denominator()) * sigma
    if poly == 0:
        raise SymbolicOutError('internal: sign condition on zero')
    out = []
    fac = poly.factor()
    for f, e in fac:
        f = ctx.P(f)
        if f.is_constant():
            continue
        if e % 2 == 0:
            out.append(('!=', f, kind, origin))
            continue
        sg = _sign_of(ctx.eval_P(f, ref_pt))
        if sg == 0:
            raise ClosedFormUnavailable(
                f'{ctx.P_to_sr(f).subs(ctx.back)} vanishes at the reference '
                f'point: it is on '
                f'the boundary of the region where the half-plane split of '
                f'{origin} is fixed')
        out.append(('>', sg * f, kind, origin))
    return out


def _nonzero_items(poly, ctx: _AlgCtx, kind, origin):
    poly = ctx.P(poly)
    if poly == 0:
        raise SymbolicOutError(f'internal: {origin} is identically zero')
    if poly.is_constant():
        return []
    return [('!=', ctx.P(f), kind, origin) for f, _ in poly.factor()
            if not ctx.P(f).is_constant()]


def _split_factor(fam: _Family, ctx: _AlgCtx, ref_pt, origin: str):
    """Decide the half-plane split of a family at the reference point and
    return the condition items that keep it."""
    p = fam.p
    d = fam.deg
    items = []
    first = _routh_first_column(p)
    if first is not None:
        signs = []
        for k, c in enumerate(first):
            sg = _sign_of(ctx.eval_K(c, ref_pt))
            if sg == 0:
                raise ClosedFormUnavailable(
                    f'Routh entry {k} of {origin} vanishes at the reference '
                    f'point: a pole on the real omega axis or a boundary of '
                    f'the stability region; no closed form there')
            signs.append(sg)
            if not ctx.K(c).numerator().is_constant() or \
                    not ctx.K(c).denominator().is_constant():
                items += _condition_items(c, sg, ctx, ref_pt, 'stability',
                                          f'{origin}, Routh entry {k}')
        n_pos = sum(1 for a, b in zip(signs, signs[1:]) if a != b)
        fam.n_neg = d - n_pos
        fam.method = 'routh'
        return items
    U = p.parent()
    pm = U([c * (-1) ** i for i, c in enumerate(p.list())])
    if pm == p or pm == -p:
        r, chain = _sturm_conditions_even(p)
        v_minf, v_zero = [], []
        for g in chain:
            lc = g.leading_coefficient()
            sg = _sign_of(ctx.eval_K(lc, ref_pt))
            if sg == 0:
                raise ClosedFormUnavailable(
                    f'Sturm chain of {origin} degenerates at the reference '
                    f'point')
            if not (ctx.K(lc).numerator().is_constant()
                    and ctx.K(lc).denominator().is_constant()):
                items += _condition_items(lc, sg, ctx, ref_pt, 'stability',
                                          f'{origin}, Sturm leading coeff')
            v_minf.append(sg * (-1) ** int(g.degree()))
            c0 = g[0]
            if c0 != 0:
                s0 = _sign_of(ctx.eval_K(c0, ref_pt))
                if s0 == 0:
                    raise ClosedFormUnavailable(
                        f'{origin} has a pole at omega = 0 at the reference '
                        f'point')
                if not (ctx.K(c0).numerator().is_constant()
                        and ctx.K(c0).denominator().is_constant()):
                    items += _condition_items(c0, s0, ctx, ref_pt,
                                              'stability',
                                              f'{origin}, Sturm value at 0')
                v_zero.append(s0)
        if r[0] == 0:
            raise ClosedFormUnavailable(f'{origin} has a pole at omega = 0')

        def changes(seq):
            return sum(1 for a, b in zip(seq, seq[1:]) if a != b)
        if changes(v_minf) - changes(v_zero) != 0:
            raise ClosedFormUnavailable(
                f'{origin} has poles on the real omega axis at the reference '
                f'point; the transform needs a principal-value prescription')
        fam.n_neg = d // 2
        fam.method = 'sturm-even'
        return items
    # Singular, not even: the Routh array of p(s)*prod(s + c_i), c_i > 0,
    # adds known left-half-plane roots and is regular for generic c_i.
    K = p.parent().base_ring()
    shifts = [(c,) for c in (1, 2, 3, QQ(1) / 2, 5, QQ(1) / 3)] + \
        [(1, 2), (1, 3), (2, 5), (QQ(1) / 2, 3), (1, 2, 3), (1, 3, 7),
         (QQ(1) / 2, 2, 5), (QQ(1) / 3, 1, 4)]
    for cs_ in shifts:
        q = p
        for c in cs_:
            q = q * (ctx.x + K(c))
        c = ', '.join(str(x) for x in cs_)
        first = _routh_first_column(q)
        if first is None:
            continue
        vals = [ctx.eval_K(e, ref_pt) for e in first]
        if any(_sign_of(v) == 0 for v in vals):
            continue
        signs = [_sign_of(v) for v in vals]
        for k, e in enumerate(first):
            if not (ctx.K(e).numerator().is_constant()
                    and ctx.K(e).denominator().is_constant()):
                items += _condition_items(
                    e, signs[k], ctx, ref_pt, 'stability',
                    f'{origin} times prod(s + c) for c in ({c}), Routh '
                    f'entry {k}')
        n_pos = sum(1 for a_, b_ in zip(signs, signs[1:]) if a_ != b_)
        fam.n_neg = (d + len(cs_) - n_pos) - len(cs_)
        fam.method = f'routh-shifted({c})'
        return items
    raise ClosedFormUnavailable(
        f'the Routh array of {origin} is singular at the reference point '
        f'(a pole on the real omega axis?); its half-plane split cannot be '
        f'stated as sign conditions here.  Use the numeric route '
        f'(numeric_inverse_transform) at parameter values')


class _ModeData:
    """Families, assumptions and root presentation of a rational function
    of s (a bridge field element) at a reference point."""

    def __init__(self, f, bridge: _Bridge, ctx: _AlgCtx, ref_pt,
                 laurent_order: Optional[int] = None):
        self.bridge, self.ctx = bridge, ctx
        num, den = f.numerator(), f.denominator()
        N, Dn = ctx.from_R(num), ctx.from_R(den)
        self.families: list = []
        items: list = []
        param_factors = []
        for g, m in den.factor():
            if bridge.deg_s(g) == 0:
                param_factors.append(g)
                continue
            p = ctx.from_R(g)
            Q, rem = Dn.quo_rem(p ** int(m))
            if rem != 0:
                raise SymbolicOutError('internal: factor does not divide')
            lau = _laurent(N, Q, p, int(m))
            self.families.append(_Family(len(self.families), p, g, int(m),
                                         lau))
        for g in param_factors:
            items += _nonzero_items(ctx.F_to_K(bridge.F(g)).numerator(), ctx,
                                    'nonzero', 'denominator factor')
        for fam in self.families:
            origin = f'factor {self._factor_label(fam)}'
            items += _split_factor(fam, ctx, ref_pt, origin)
            if fam.deg >= 2:
                disc = ctx.F_to_K(bridge.F(fam.p_R.discriminant(
                    bridge.s_gen)))
                items += _nonzero_items(disc.numerator(), ctx, 'distinct',
                                        f'discriminant of {origin}')
        for i, fa in enumerate(self.families):
            for fb in self.families[i + 1:]:
                res = ctx.F_to_K(bridge.F(fa.p_R.resultant(fb.p_R,
                                                           bridge.s_gen)))
                items += _nonzero_items(
                    res.numerator(), ctx, 'distinct',
                    f'resultant of factors {fa.index} and {fb.index}')
        self.assumptions = _dedupe_assumptions(items, ctx)
        for a in self.assumptions:
            if not _holds_pt(a, ref_pt):
                shown = (f'{SR(a.expression).subs(ctx.back)} {a.relation} 0'
                         f'   [{a.kind}: {a.origin}]')
                raise AssumptionViolation(
                    f'at the reference point {shown} fails: poles coincide or a '
                    f'denominator vanishes there.  Substitute those values '
                    f'with values= so the repeated pole is treated exactly, '
                    f'or choose another reference.')
        self._root_sr: dict = {}
        self.roots: list = []
        self._ref_pt = ref_pt
        self._present_roots()

    def _factor_label(self, fam):
        x = SR.var('s')
        return str(SR(sum(self.ctx.K_to_sr(c) * x ** i
                          for i, c in enumerate(fam.p.list()))).subs(
                              self.ctx.back))

    def _present_roots(self):
        """SR expression for every root (explicit for degree <= 2)."""
        ctx = self.ctx
        used = set(self.bridge.sym_of) | set(ctx.name_map.values()) | {
            str(v) for v in ctx.back.values()}
        for fam in self.families:
            d = fam.deg
            a = fam.p.list()
            if d == 1:
                self._root_sr[(fam.index, 0)] = ctx.K_to_sr(-a[0] / a[1])
                continue
            if d == 2:
                c, b, a2 = a
                disc = b * b - 4 * a2 * c
                dval = ctx.eval_K(disc, self._ref_pt)
                dsr = ctx.K_to_sr(disc)
                if float(dval.real()) < 0:
                    rd = _I * _sqrt(ctx.K_to_sr(-disc))
                else:
                    rd = _sqrt(dsr)
                plus = (ctx.K_to_sr(-b) + rd) / ctx.K_to_sr(2 * a2)
                minus = (ctx.K_to_sr(-b) - rd) / ctx.K_to_sr(2 * a2)
                if _sign_of(ctx.eval_K(a2, self._ref_pt)) > 0:
                    order = [minus, plus]
                else:
                    order = [plus, minus]
                for k in range(2):
                    self._root_sr[(fam.index, k)] = order[k]
                continue
            for k in range(d):
                base = f'rho{fam.index}_{k}'
                nm = base
                while nm in used:
                    nm += '_'
                used.add(nm)
                sym = SR.var(nm)
                poly = sum(ctx.K_to_sr(cf) * sym ** i
                           for i, cf in enumerate(a))
                self._root_sr[(fam.index, k)] = sym
                self.roots.append(RootOf(sym, poly, k, fam.side_of(k)))

    # ── per-root SR of an L element ──────────────────────────────────────
    def root_sr(self, fam, k):
        return self._root_sr[(fam.index, k)]

    def element_sr(self, fam, k, g):
        g = g % fam.p
        if fam.deg == 1:
            return self.ctx.K_to_sr(g[0])
        return self.ctx.L_to_sr(g, self.root_sr(fam, k))


def _holds_pt(a: Assumption, pt) -> bool:
    v = a._ctx.eval_P(a._poly, pt)
    if a.relation == '>':
        return bool(v.real() > 0)
    return bool(v != 0)


def _dedupe_assumptions(items, ctx: _AlgCtx) -> list:
    seen = {}
    for rel, poly, kind, origin in items:
        poly = ctx.P(poly)
        key_poly = poly
        if rel == '!=':
            lc = poly.lc()
            key_poly = poly * (~lc)
        key = (rel, key_poly)
        if key in seen:
            continue
        seen[key] = Assumption(ctx.P_to_sr(poly), rel, kind, origin,
                               _poly=poly, _ctx=ctx)
    # a '!=' condition implied by a '>' on the same polynomial is dropped
    gt = {k[1] for k in seen if k[0] == '>'}
    gt |= {(-p) for p in gt}
    out = []
    for (rel, kp), a in seen.items():
        if rel == '!=' and any(kp == g * (~g.lc()) for g in gt):
            continue
        out.append(a)
    return out


def _check_assumptions(assumptions, values: dict, what: str):
    bad = [a for a in assumptions if not a.holds(values)]
    if bad:
        lines = '\n  '.join(str(a) for a in bad)
        raise AssumptionViolation(
            f'{what}: the parameter values violate the stated assumptions\n'
            f'  {lines}\nThe symbolic result does not apply there (the '
            f'half-plane split of the poles, or their distinctness, '
            f'changes).  Rebuild it with reference= set to these values, or '
            f'substitute coinciding values with values=.')


def _name_maps(fwd: dict):
    """private name -> user name, from a _private_symbols forward map."""
    return {str(fresh): str(orig) for orig, fresh in fwd.items()}


def _reference_point(names: Sequence[str], reference: Optional[dict]) -> dict:
    """The reference values by user name; a symbol without one is 1.0
    (assumed positive), as in the sampling cross-check."""
    ref = {}
    for k in names:
        if reference and k in reference:
            v = reference[k]
            ref[k] = v if not isinstance(v, (np.floating,)) else float(v)
        else:
            ref[k] = 1.0
    return ref


def _cross_check(md: _ModeData, ctx: _AlgCtx, ref: dict, names,
                 n_samples: int = 12):
    """Sampling cross-check: at sample points inside the stated region the
    numeric root count per half plane must equal the Routh/Sturm count."""
    real_ref = {}
    for k in names:
        try:
            real_ref[k] = float(complex(ref[k]).real)
        except (TypeError, ValueError):
            real_ref[k] = 1.0
    for pt_d in _sample_points(list(names), real_ref, n_samples=n_samples):
        pt = ctx.point(pt_d)
        if not all(_holds_pt(a, pt) for a in md.assumptions):
            continue
        for fam in md.families:
            rts = ctx.numeric_roots(fam.p, pt)
            n_neg = sum(1 for z in rts if float(z.real()) < 0)
            if n_neg != fam.n_neg:
                raise SymbolicOutError(
                    f'internal error: the stated assumptions hold at {pt_d} '
                    f'but factor {fam.index} has {n_neg} left-half-plane '
                    f'roots there, not {fam.n_neg} (please report)')


def _side_sign(side: str) -> int:
    return -1 if side == 'positive' else 1


@dataclass
class ModeSum:
    """Time-domain covariance as a sum of exponential modes plus a contact
    term:

        C(tau) = sum_{terms on side(tau)} coeff * tau**power * exp(rate*tau)
                 + contact * delta(tau)

    ``terms``: list of :class:`Mode` (Sage expressions in the parameters and,
    for poles of irreducible factors of degree >= 3, the ``RootOf`` symbols
    listed in ``roots``); ``assumptions``: :class:`Assumption` list under
    which the list is valid; ``reference``: the parameter point at which the
    half-plane split was decided; ``tau``: the lag symbol used by
    :meth:`expression`.  Built by :func:`tree_modes` /
    :meth:`TreeCovariance.modes`."""
    terms: list
    contact: Any
    roots: list
    assumptions: list
    reference: dict
    tau: Any
    omega: Any
    _md: Any = field(default=None, repr=False)
    _back: dict = field(default_factory=dict, repr=False)
    _source: Any = field(default=None, repr=False)
    _contact_K: Any = field(default=None, repr=False)
    _explicit: frozenset = field(default=frozenset(), repr=False)
    _fixed: dict = field(default_factory=dict, repr=False)

    # ── symbolic views ──────────────────────────────────────────────────
    def expression(self, side: str = 'positive'):
        """Sum of the terms of one side as a Sage expression in ``tau``."""
        if side not in SIDES:
            raise SymbolicOutError(f'side must be one of {SIDES}')
        t = self.tau
        return sum((m.coeff * t ** m.power * _exp(m.rate * t)
                    for m in self.terms if m.side == side), SR(0))

    def to_dicts(self) -> dict:
        """JSON-ready dict (strings for the expressions)."""
        return {
            'format': FORMAT_TAG + '/modes',
            'terms': [{'coeff': str(m.coeff), 'rate': str(m.rate),
                       'power': int(m.power), 'side': m.side}
                      for m in self.terms],
            'contact': str(self.contact),
            'roots': [{'symbol': str(r.symbol), 'polynomial': str(r.polynomial),
                       'index': r.index, 'side': r.side} for r in self.roots],
            'assumptions': [{'expression': str(a.expression),
                             'relation': a.relation, 'kind': a.kind,
                             'origin': a.origin} for a in self.assumptions],
            'reference': {k: complex(v).real if np.isreal(complex(v)) else
                          str(v) for k, v in self.reference.items()},
        }

    # ── numeric views ───────────────────────────────────────────────────
    def check(self, values: dict) -> None:
        """Raise :class:`AssumptionViolation` if ``values`` lie outside the
        region of validity."""
        _check_assumptions(self.assumptions, self._full(values), 'ModeSum')

    def _full(self, values, extra_names=()):
        """``values`` over the explicitly given reference values (symbols
        that were only defaulted to 1 must be given).  A name that was
        substituted when the modes were built must carry the same value (it
        is not a parameter of the result any more); an unknown name is an
        error -- never silently ignored."""
        if values is None:
            return dict(self.reference)
        vals = _as_name_dict(values)
        fixed = dict(self._fixed)
        fixed.update((self._source[2] if self._source else {}) or {})
        out = {k: x for k, x in self.reference.items() if k in self._explicit}
        for k, x in vals.items():
            if k in self.reference or k in extra_names:
                out[k] = x
                continue
            if k in fixed:
                try:
                    same = complex(SR(x).n()) == complex(SR(fixed[k]).n())
                except (TypeError, ValueError):
                    same = False
                if not same:
                    raise SymbolicOutError(
                        f'{k!r} entered the construction of these modes '
                        f'({k} = {fixed[k]}: substituted, or used for the '
                        f'mean-field solve); rebuild them to use {k} = {x}')
            # any other name does not occur in C: it cannot change it
        return out

    def numeric_terms(self, values: Optional[dict] = None,
                      check: bool = True) -> list:
        """The modes at ``values`` (default: the reference point) as
        :class:`Mode` with complex ``coeff``/``rate``; roots are found
        numerically (256-bit) and assigned to sides by the stated split.

        Near-coincident poles give large mode coefficients that cancel; sum
        such modes with :meth:`evaluate` (256-bit), not in double
        precision."""
        return [Mode(complex(c), complex(r), j, sd)
                for c, r, j, sd in self._numeric_hp(values, check)]

    def _numeric_hp(self, values, check: bool = True) -> list:
        vals = self._full(values)
        if check:
            _check_assumptions(self.assumptions, vals, 'ModeSum')
        md = self._md
        ctx = md.ctx
        pt = ctx.point(vals)
        out = []
        for fam in md.families:
            rts = ctx.numeric_roots(fam.p, pt)
            m = fam.mult
            for k, z in enumerate(rts):
                side = fam.side_of(k)
                if (float(z.real()) < 0) != (side == 'negative'):
                    raise SymbolicOutError(
                        f'internal error: root {z} of factor {fam.index} is '
                        f'on the wrong side although the assumptions hold')
                sg = _side_sign(side)
                for j in range(m):
                    cj = _coeff_poly(fam, j)
                    if cj == 0:
                        continue
                    cv = sum((c * z ** i for i, c in
                              enumerate(ctx.eval_U(cj, pt))), _cc()(0))
                    out.append((sg * cv, -z, j, side))
        return out

    def contact_value(self, values: Optional[dict] = None) -> complex:
        vals = self._full(values)
        if self._contact_K is None:
            return 0j
        return complex(self._md.ctx.eval_K(self._contact_K,
                                           self._md.ctx.point(vals)))

    def __call__(self, tau, values: Optional[dict] = None,
                 at_zero: str = 'negative'):
        return self.evaluate(tau, values, at_zero=at_zero)

    def evaluate(self, tau, values: Optional[dict] = None,
                 at_zero: str = 'negative'):
        """Smooth part C(tau) at ``values`` (complex ndarray, shape of tau).

        ``at_zero``: which side's limit is used at tau = 0; the default
        'negative' is the pipeline's Ito left limit."""
        if at_zero not in SIDES:
            raise SymbolicOutError(f'at_zero must be one of {SIDES}')
        terms = self._numeric_hp(values)
        CC = _cc()
        t = np.asarray(tau, dtype=float)
        out = np.zeros(t.shape, dtype=complex)
        for idx in np.ndindex(t.shape):
            tv = float(t[idx])
            side = 'positive' if (tv > 0 or (tv == 0 and
                                             at_zero == 'positive')) \
                else 'negative'
            T = CC(tv)
            acc = CC(0)
            for c, r, j, sd in terms:
                if sd == side:
                    acc += c * T ** j * (r * T).exp()
            out[idx] = complex(acc)
        return out

    # ── kernel integrals ────────────────────────────────────────────────
    def integrate_against(self, kernel: 'Kernel', values: Optional[dict]
                          = None, contact: Optional[str] = None
                          ) -> 'KernelIntegral':
        """``int L(s) C(s) ds`` in closed form from the modes; see
        :func:`integrate_against`."""
        return _integrate_modes(self, kernel, values, contact)


def _coeff_poly(fam: _Family, j: int):
    """R_j(x): residue of C0 e^{-s tau} at a root, coefficient of tau^j
    (before the side sign)."""
    m = fam.mult
    h = fam.laurent[m - 1 - j]
    if h == 0:
        return h
    K = fam.p.parent().base_ring()
    return (h * K((-1) ** j) * (~K(factorial(j)))) % fam.p


def _exact_value(v):
    """A substituted value as an exact SR number: a float becomes the
    rational it represents exactly (not a nearby 'simplest' one), so that
    cancellations between terms survive the substitution."""
    if isinstance(v, (float, np.floating)):
        return SR(QQ(float(v)))
    return SR(v)


def _prepare(expr, omega, extra=(), values=None, reference=None):
    """Private-symbol bridge, context and reference point for ``expr``."""
    om = SR('omega') if omega is None else SR(omega)
    expr = SR(expr)
    vals = _as_name_dict(values)
    if vals:
        free = {str(v): v for v in expr.variables()}
        for e in extra:
            free.update({str(v): v for v in SR(e).variables()})
        unknown = sorted(k for k in vals if k not in free)
        if unknown:
            raise SymbolicOutError(
                f'values= names {unknown} do not occur; known symbols: '
                f'{sorted(k for k in free if k != str(om))}')
        sub = {free[k]: _exact_value(v) for k, v in vals.items()}
        expr = expr.subs(sub)
        extra = [SR(e).subs(sub) for e in extra]
    fwd, back = _private_symbols([expr] + list(extra), keep=[om])
    pexpr = expr.subs(fwd)
    pextra = [SR(e).subs(fwd) for e in extra]
    bridge = _Bridge([pexpr] + pextra, om)
    ctx = _AlgCtx(bridge, _name_maps(fwd), back)
    user_names = sorted(ctx.name_map.get(n, n) for n in bridge.sym_of)
    ref = _reference_point(user_names, reference)
    ref_pt = ctx.point(ref)
    return om, pexpr, pextra, bridge, ctx, ref, ref_pt, back


def _lag_symbol(names):
    return SR.var('tau') if 'tau' not in names else SR.var('tau_lag')


def tree_modes(C0_entry, omega=None, *, values: Optional[dict] = None,
               reference: Optional[dict] = None, cross_check: bool = True
               ) -> ModeSum:
    r"""Time-domain mode sum of ``C(tau) = (1/2 pi) int C0(omega)
    exp(-i omega tau) d omega`` for a rational ``C0`` of any denominator
    degree (stage C2).

    Every pole ``rho`` (in ``s = i*omega``) of multiplicity ``m`` gives the
    terms ``coeff * tau^j * exp(-rho tau)``, ``j < m``, on ``side``
    'positive' (``Re rho > 0``) or 'negative' (``Re rho < 0``); a constant
    tail is the ``contact`` weight of ``delta(tau)`` (not in the terms).
    Roots of irreducible factors of degree <= 2 are explicit, higher ones are
    :class:`RootOf` symbols and the coefficients are polynomials in them with
    coefficients rational in the parameters.

    ``values``: symbols substituted before factoring (so coinciding values
    make a repeated pole exact).  ``reference``: the point at which the
    half-plane split is decided (default: every remaining symbol = 1, i.e.
    assumed positive); the result's ``assumptions`` are the sign and
    distinctness conditions that keep it valid, checked again whenever the
    modes are evaluated.  ``cross_check`` compares the stated split with
    numeric roots at sampled points (an internal consistency check, not the
    decision)."""
    om, pexpr, _, bridge, ctx, ref, ref_pt, back = _prepare(
        C0_entry, omega, values=values, reference=reference)
    f = bridge.to_field(pexpr)
    c_inf, gap = _contact_of(bridge, f)
    contact_K = None
    if gap == 0:
        cf = _field_const(bridge, f)
        contact_K = ctx.F_to_K(cf)
        fp = f - cf
    else:
        fp = f
    if fp.numerator() != 0:
        _check_real_axis_free_decay(bridge, fp, 1, 'tree_modes')
        md = _ModeData(fp, bridge, ctx, ref_pt)
    else:
        md = _ModeData(bridge.F(1), bridge, ctx, ref_pt)
    if cross_check:
        _cross_check(md, ctx, ref, ref.keys())
    terms = []
    for fam in md.families:
        for k in range(fam.deg):
            side = fam.side_of(k)
            sg = _side_sign(side)
            rho = md.root_sr(fam, k)
            for j in range(fam.mult):
                cj = _coeff_poly(fam, j)
                if cj == 0:
                    continue
                coeff = sg * md.element_sr(fam, k, cj)
                terms.append(Mode(SR(coeff).subs(back), SR(-rho).subs(back),
                                  j, side))
    roots = [RootOf(r.symbol, SR(r.polynomial).subs(back), r.index, r.side)
             for r in md.roots]
    assumptions = [Assumption(SR(a.expression).subs(back), a.relation,
                              a.kind, a.origin, _poly=a._poly, _ctx=a._ctx)
                   for a in md.assumptions]
    return ModeSum(terms=terms, contact=SR(c_inf).subs(back), roots=roots,
                   assumptions=assumptions, reference=ref,
                   tau=_lag_symbol(set(ref)), omega=om, _md=md, _back=back,
                   _source=(SR(C0_entry), om, dict(_as_name_dict(values))),
                   _contact_K=contact_K,
                   _explicit=frozenset(k for k in ref
                                       if k in _as_name_dict(reference)))


# ── kernels and time-domain kernel integrals ────────────────────────────────

@dataclass(frozen=True)
class Kernel:
    """A kernel ``L(s) = sum coeff * s**power * exp(rate * s)`` over its
    terms, each on ``side`` 'positive' (s > 0) or 'negative' (s < 0).
    Terms are :class:`Mode` objects; coefficients and rates are real Sage
    expressions or numbers.  ``rate`` must make the term non-growing on its
    side (``rate < 0`` for s > 0, ``rate > 0`` for s < 0, or exactly 0)."""
    terms: tuple
    name: str = ''

    def __post_init__(self):
        for t in self.terms:
            if t.side not in SIDES or int(t.power) < 0:
                raise SymbolicOutError(f'bad kernel term {t}')

    @classmethod
    def exponential(cls, amplitude, rate, side: str = 'positive',
                    power: int = 0) -> 'Kernel':
        """``amplitude * s**power * exp(rate * s)`` on one side."""
        return cls((Mode(SR(amplitude), SR(rate), int(power), side),),
                   'exponential')

    @classmethod
    def two_sided_exponential(cls, a_pos, tau_pos, a_neg, tau_neg) -> 'Kernel':
        """``L(s) = a_pos e^{-s/tau_pos}`` (s > 0),
        ``a_neg e^{s/tau_neg}`` (s < 0)."""
        return cls((Mode(SR(a_pos), -1 / SR(tau_pos), 0, 'positive'),
                    Mode(SR(a_neg), 1 / SR(tau_neg), 0, 'negative')),
                   'two_sided_exponential')

    def __add__(self, other: 'Kernel') -> 'Kernel':
        return Kernel(tuple(self.terms) + tuple(other.terms),
                      self.name or other.name)

    def at_zero(self, side: str):
        """``L(0+)`` (side 'positive') or ``L(0-)``."""
        return sum((SR(t.coeff) for t in self.terms
                    if t.side == side and t.power == 0), SR(0))

    def fourier(self, omega):
        """``L~(omega) = int L(s) exp(-i omega s) ds`` (pipeline
        convention)."""
        w = SR(omega)
        out = SR(0)
        for t in self.terms:
            q = int(t.power)
            if t.side == 'positive':
                out += SR(t.coeff) * factorial(q) / (_I * w - t.rate) ** (q + 1)
            else:
                out += (SR(t.coeff) * (-1) ** q * factorial(q)
                        / (t.rate - _I * w) ** (q + 1))
        return out

    def expressions(self) -> list:
        return [SR(x) for t in self.terms for x in (t.coeff, t.rate)]


@dataclass
class KernelIntegral:
    """``int L(s) C(s) ds``: ``smooth`` (the modes), ``contact`` (the
    delta(tau) weight times the chosen L(0)), ``expression`` = their sum,
    ``value`` at the given values (or ``None``), ``assumptions`` (modes and
    kernel decay) and the ``roots`` any split factor leaves in the result."""
    expression: Any
    smooth: Any
    contact: Any
    value: Optional[complex]
    assumptions: list
    roots: list
    exact: bool = True

    def __float__(self):
        return float(np.real(self.value))


def _kernel_assumption_items(kterms, ctx, ref_pt):
    items = []
    for t, a_K, al_K in kterms:
        if al_K == 0:
            continue
        need = -1 if t.side == 'positive' else 1
        v = ctx.eval_K(al_K, ref_pt)
        if _sign_of(v) != need:
            raise SymbolicOutError(
                f'kernel term {t} grows on its side at the reference point '
                f'(rate {t.rate}): the integral diverges')
        items += _condition_items(al_K, need, ctx, ref_pt, 'kernel',
                                  f'decay of kernel term on the {t.side} '
                                  f'side')
    return items


def _integrate_modes(ms: ModeSum, kernel: Kernel, values, contact):
    if not isinstance(kernel, Kernel):
        raise SymbolicOutError('kernel must be a Kernel (Kernel.two_sided_exponential, '
                               'Kernel.exponential, ...)')
    if contact not in (None, 'exclude', 'positive', 'negative', 'mid'):
        raise SymbolicOutError(
            "contact must be None, 'exclude', 'positive', 'negative' or "
            "'mid'")
    src, om, src_vals = ms._source
    k_names = {str(v) for e in kernel.expressions() for v in SR(e).variables()}
    vals_in = _as_name_dict(values)
    k_vals = {k: v for k, v in vals_in.items() if k in k_names}
    # the C part of values must lie in the region of the ModeSum (same rule
    # as ModeSum.evaluate); the split is decided at the ModeSum's reference
    c_vals = None
    if values is not None:
        c_vals = ms._full({k: v for k, v in vals_in.items()
                           if k not in k_names}, extra_names=k_names)
        _check_assumptions(ms.assumptions, c_vals, 'integrate_against')
    ref_in = dict(ms.reference)
    ref_in.update(k_vals)
    _, pexpr, pextra, bridge, ctx, ref, ref_pt, back = _prepare(
        src, om, extra=kernel.expressions(), values=src_vals,
        reference=ref_in)
    f = bridge.to_field(pexpr)
    _, gap = _contact_of(bridge, f)
    c_K = ctx.K(0)
    if gap == 0:
        cf = _field_const(bridge, f)
        c_K = ctx.F_to_K(cf)
        f = f - cf
    md = _ModeData(f if f.numerator() != 0 else bridge.F(1), bridge, ctx,
                   ref_pt)
    kterms = []
    for t, (pc, pr) in zip(kernel.terms, zip(pextra[0::2], pextra[1::2])):
        kterms.append((t, ctx.F_to_K(bridge.to_field(pc)),
                       ctx.F_to_K(bridge.to_field(pr))))
    k_items = _kernel_assumption_items(kterms, ctx, ref_pt)
    assumptions = list(md.assumptions) + _dedupe_assumptions(k_items, ctx)
    K = ctx.K
    total_K = K(0)
    split_sr = SR(0)
    numeric_parts = []          # ('trace', fam, G) | ('roots', fam, ...)
    for fam in md.families:
        p, x = fam.p, ctx.x
        for t, a_K, al_K in kterms:
            idx = [k for k in range(fam.deg) if fam.side_of(k) == t.side]
            if not idx:
                continue
            q = int(t.power)
            if len(idx) < fam.deg:
                # split factor: per root, -R_j(rho) a (q+j)! / (rho-alpha)^n.
                # (An inverse of x - alpha in K[x]/(p) would put p(alpha) in
                # the denominator, which vanishes when alpha is a root on the
                # OTHER side.)
                terms_j = [(j, _coeff_poly(fam, j)) for j in range(fam.mult)]
                terms_j = [(j, c) for j, c in terms_j if c != 0]
                a_sr, al_sr = ctx.K_to_sr(a_K), ctx.K_to_sr(al_K)
                for k in idx:
                    rho = md.root_sr(fam, k)
                    for j, cj in terms_j:
                        split_sr += (-md.element_sr(fam, k, cj) * a_sr
                                     * factorial(q + j)
                                     / (rho - al_sr) ** (q + j + 1))
                numeric_parts.append(('roots', fam, terms_j, a_K, al_K, q,
                                      idx))
                continue
            lin = (x - al_K) % p
            try:
                inv1 = _inv_mod(lin, p)
            except ZeroDivisionError:
                raise SymbolicOutError(
                    f'a pole of C coincides with the kernel rate {t.rate}: '
                    f'the integral diverges') from None
            G = ctx.U(0)
            for j in range(fam.mult):
                cj = _coeff_poly(fam, j)
                if cj == 0:
                    continue
                n = q + j + 1
                w = inv1
                for _ in range(n - 1):
                    w = (w * inv1) % p
                G += -cj * a_K * K(factorial(q + j)) * w
            G = G % p
            if G == 0:
                continue
            total_K += _trace(G, p)
            numeric_parts.append(('trace', fam, G))
    # contact: c * L(0)
    l0 = {s: sum((a_K for t, a_K, _ in kterms
                  if t.side == s and t.power == 0), K(0)) for s in SIDES}
    if c_K == 0 or contact == 'exclude':
        contact_K = K(0)
    elif contact is None:
        if l0['positive'] != l0['negative']:
            raise SymbolicOutError(
                f'C has a contact term {ctx.K_to_sr(c_K).subs(back)} * '
                f'delta(tau) and the kernel jumps at s = 0 (L(0+) = '
                f'{ctx.K_to_sr(l0["positive"]).subs(back)}, L(0-) = '
                f'{ctx.K_to_sr(l0["negative"]).subs(back)}): choose '
                f"contact='exclude', 'positive', 'negative' or 'mid'")
        contact_K = c_K * l0['positive']
    elif contact == 'mid':
        contact_K = c_K * (l0['positive'] + l0['negative']) / 2
    else:
        contact_K = c_K * l0[contact]
    smooth = (ctx.K_to_sr(total_K) + split_sr).subs(back)
    contact_sr = ctx.K_to_sr(contact_K).subs(back)
    roots = [RootOf(r.symbol, SR(r.polynomial).subs(back), r.index, r.side)
             for r in md.roots if SR(split_sr).has(r.symbol)]
    value = None
    names_needed = {ctx.name_map.get(n, n) for n in bridge.sym_of}
    if values is not None:
        vals = dict(c_vals)
        vals.update(k_vals)
    else:
        vals = {k: v for k, v in ms.reference.items() if k in ms._explicit}
    if values is not None or names_needed <= set(vals):
        if not names_needed <= set(vals):
            raise SymbolicOutError(
                f'values= is missing {sorted(names_needed - set(vals))}')
        _check_assumptions(assumptions, vals, 'integrate_against')
        pt = ctx.point(vals)
        CC = _cc()
        tot = ctx.eval_K(contact_K, pt)
        for part in numeric_parts:
            if part[0] == 'trace':
                tot += ctx.eval_K(_trace(part[2], part[1].p), pt)
                continue
            _, fam, terms_j, a_K, al_K, q, idx = part
            rts = ctx.numeric_roots(fam.p, pt)
            a_v, al_v = ctx.eval_K(a_K, pt), ctx.eval_K(al_K, pt)
            for k in idx:
                z = rts[k]
                for j, cj in terms_j:
                    cv = sum((c * z ** i for i, c in
                              enumerate(ctx.eval_U(cj, pt))), CC(0))
                    tot += (-cv * a_v * CC(factorial(q + j))
                            / (z - al_v) ** (q + j + 1))
        value = complex(tot)
    return KernelIntegral(expression=smooth + contact_sr, smooth=smooth,
                          contact=contact_sr, value=value,
                          assumptions=[Assumption(SR(a.expression).subs(back),
                                                  a.relation, a.kind,
                                                  a.origin, _poly=a._poly,
                                                  _ctx=a._ctx)
                                       for a in assumptions],
                          roots=roots)


def integrate_against(C0_entry, kernel: Kernel, omega=None, *,
                      values: Optional[dict] = None,
                      reference: Optional[dict] = None,
                      contact: Optional[str] = None) -> KernelIntegral:
    r"""``int L(s) C(s) ds`` for the covariance ``C(tau)`` of ``C0_entry``
    (convention of :func:`tree_modes`) and a :class:`Kernel` ``L``, in
    closed form from the modes:

        int_0^oo  a s^q e^{alpha s} * c s^j e^{-rho s} ds
            = a c (q+j)! / (rho - alpha)^{q+j+1}

    and its mirror for s < 0.  Summed over all roots of an irreducible factor
    that lie on the kernel's side, the result is a trace in K[x]/(p): an exact
    rational function of the parameters (any degree).  ``contact`` decides
    the delta(tau) term: ``None`` includes ``c * L(0)`` when the kernel is
    continuous at 0 and refuses otherwise; ``'exclude'`` / ``'positive'`` /
    ``'negative'`` / ``'mid'`` use 0 / L(0+) / L(0-) / their mean.

    Equals ``pair_with_kernel(C0_smooth, L.fourier(omega)(-omega))`` (the
    Fourier route) for the smooth part."""
    ms = tree_modes(C0_entry, omega, values=values, reference=reference)
    return ms.integrate_against(kernel, contact=contact)


def _pair_any_degree(C0_entry, L_tilde, omega, *, values, reference):
    """Fourier-route pairing ``int d omega/(2 pi) conj(L~) C0`` by residues
    for any denominator degree: ``sum_{Re rho < 0} Res_s`` (closing to the
    left in s = i*omega), traces for one-sided factors."""
    om = SR('omega') if omega is None else SR(omega)
    vals = _as_name_dict(values)
    expr = SR(C0_entry)
    L = SR(L_tilde)
    if vals:
        free = {str(v): v for v in expr.variables()}
        free.update({str(v): v for v in L.variables()})
        sub = {free[k]: _exact_value(v) for k, v in vals.items()
               if k in free}
        expr, L = expr.subs(sub), L.subs(sub)
    integrand = _conjugate_real(L, om) * expr
    ref_in = dict(reference or {})
    ref_in.update(vals)
    _, pexpr, _, bridge, ctx, ref, ref_pt, back = _prepare(
        integrand, om, reference=ref_in)
    f = bridge.to_field(pexpr)
    _check_real_axis_free_decay(bridge, f, 2, 'pair_with_kernel')
    if f.numerator() == 0:
        return KernelPairing(expression=SR(0), value=0j, exact=True)
    md = _ModeData(f, bridge, ctx, ref_pt)
    _cross_check(md, ctx, ref, ref.keys())
    total_K = ctx.K(0)
    split_sr = SR(0)
    parts = []
    for fam in md.families:
        G = fam.laurent[fam.mult - 1]
        if G == 0 or fam.n_neg == 0:
            continue
        if fam.n_neg == fam.deg:
            total_K += _trace(G, fam.p)
            parts.append((fam, G, True))
        else:
            for k in range(fam.n_neg):
                split_sr += md.element_sr(fam, k, G)
            parts.append((fam, G, False))
    expr_out = (ctx.K_to_sr(total_K) + split_sr).subs(back)
    value = None
    names_needed = {ctx.name_map.get(n, n) for n in bridge.sym_of}
    if (values is not None or not names_needed) and names_needed <= set(
            ref_in):
        pt = ctx.point(ref_in)
        tot = _cc()(0)
        for fam, G, traced in parts:
            if traced:
                tot += ctx.eval_K(_trace(G, fam.p), pt)
            else:
                rts = ctx.numeric_roots(fam.p, pt)
                cs = ctx.eval_U(G, pt)
                for k in range(fam.n_neg):
                    tot += sum((c * rts[k] ** i for i, c in enumerate(cs)),
                               _cc()(0))
        value = complex(tot)
    out = KernelPairing(expression=expr_out, value=value, exact=True)
    out.assumptions = [Assumption(SR(a.expression).subs(back), a.relation,
                                  a.kind, a.origin, _poly=a._poly,
                                  _ctx=a._ctx) for a in md.assumptions]
    return out
