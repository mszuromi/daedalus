r"""
Symbolic tree-level covariance in the Fourier domain, with exports.

Stage C1 of ``docs/feature_plan_sde_and_delta.md``.  The pipeline builds the
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
* Closed-form residues need every irreducible factor of the denominator to
  have degree <= 2 in omega; higher degrees raise ``ClosedFormUnavailable``
  and the numeric quadrature fallback is used instead.
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
        cov = cov.fix_mean_field(
            {str(k): v for k, v in result['num_params'].items()})
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
        mf_vals = _mf_symbols_only(prop, D, npar)
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
    return cov


def _restrict(vals: dict, prop, D, omega) -> dict:
    """Keep the names that actually occur (MF solves return extra symbols)."""
    occurring = {str(v) for m in (prop['K_ft'], D) for e in m.list()
                 for v in SR(e).variables()}
    if prop.get('G_ft') is not None:
        occurring |= {str(v) for e in prop['G_ft'].list()
                      for v in SR(e).variables()}
    return {k: v for k, v in vals.items() if k in occurring}


def _mf_symbols_only(prop, D, num_params) -> dict:
    """Entries of the MF ``num_params`` that are saddle values: the symbols of
    K and D that are not model parameters.  The solver returns parameters,
    saddles and Taylor-coefficient symbols together; we keep the star values
    and let the caller's ``values=`` handle ordinary parameters."""
    out = {}
    for k, v in num_params.items():
        name = str(k)
        if 'star' in name:
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
        """New TreeCovariance with the saddle symbols replaced."""
        vals = {k: v for k, v in mf_values.items()
                if k in self.bridge.sym_of and 'star' in k}
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
                f'roots have no closed form here; use the numeric fallback '
                f'(numeric_inverse_transform / numeric_pair_with_kernel) '
                f'with parameter values')
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
                     epsabs: float = 1e-12, epsrel: float = 1e-10
                     ) -> KernelPairing:
    r"""``int d omega / (2 pi)  conj(L_tilde(omega)) * C0_entry(omega)``.

    This is ``int dt L(t) C(-t)`` for a real kernel ``L`` with
    ``L_tilde(omega) = int dt L(t) exp(-i omega t)``.

    Exact route: when ``conj(L_tilde) * C0`` is rational in omega and each
    irreducible factor of its denominator has degree <= 2, the integral is
    ``i * sum_{Im omega_k > 0} Res``.  Which poles lie in the upper half plane
    is decided numerically at ``reference`` and at 12 log-uniform perturbations
    of it (parameters without a reference value are assumed positive); the
    pairing is refused if that decision changes inside the sample.  The
    integrand must decay at least like omega^-2.

    Fallback (``fallback=True``, needs a numeric value for every symbol in
    ``values``): adaptive quadrature of the same integrand
    (:func:`numeric_pair_with_kernel`) with tolerances ``epsabs``/``epsrel``.
    """
    omega = SR('omega') if omega is None else SR(omega)
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


def inverse_transform(C0_entry, omega=None, tau=None, *,
                      reference: Optional[dict] = None, simplify: bool = True):
    """Exact inverse transform by residues; see ``_inverse_transform_impl``. Runs on private, domain-free symbols."""
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
