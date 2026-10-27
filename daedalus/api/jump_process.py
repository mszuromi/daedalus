"""api.jump_process -- reaction / transition builder for jump processes (``dd.JumpProcess``).

A chemical-kinetics or shot-noise model is a list of *transitions*: at state ``x`` the
density jumps ``x -> x + jump / Omega`` at rate ``r(x)`` (``Omega`` the system size, the
inverse jump size).  ``JumpProcess`` collects them and hands them to the two engines that
already exist, with no new physics::

    import daedalus as dd
    jp = (dd.JumpProcess('dimerisation')
            .species('A', 'B')
            .parameter('k1', 1.0).parameter('k2', 0.5).parameter('s', 2.0)
            .system_size(100)
            .reaction('-> A', rate='s')
            .reaction('A + A -> B', rate='k1')           # 2A -> B, mass action
            .reaction('B ->', rate='k2'))
    res = dd.run(jp, dd.Config(k=2, max_ell=1, ...))      # the SDE-front-end model, built
    jp.compare(ks=(1, 2), max_ell=1)                      # loops vs the exact master equation

* **Building** (``sde()``, ``build()``): the SDE-front-end model whose noise is one
  state-dependent :class:`api.noise.Poisson` source of rate ``r(x)`` and size ``1/Omega`` per
  transition, coupled to every species with the jump vector as coefficients, and zero drift.
  The text of the action is exactly that of the hand-written model.
* **Validation** (``to_master_equation()``, ``compare()``): the same transitions on the
  lattice ``x = n / Omega`` as an exact :class:`api.master_equation.MasterEquation`.

Conventions
-----------
``transition(rate, jump)``: ``rate`` is the event rate (events per unit time) as a function of
the densities, the parameters and ``Omega``; ``per_volume=True`` means ``rate`` is the rate per
unit system size and ``Omega * rate`` is used.  ``jump`` is ``{species: integer}`` (or one
integer per species, in declaration order); the species change by ``jump / Omega``.

``reaction('A + B -> C', rate='k')``: mass action.  The event rate is
``Omega * k * g(x)`` with ``g`` the product over the reactants (stoichiometry ``nu``) of

* ``combinatorics='stochastic'`` (default): the falling factorial of the *count*,
  ``x (x - 1/Omega) ... (x - (nu-1)/Omega) / nu!``, i.e. the chemical master equation (a pair
  of identical molecules reacts at ``n (n-1) / 2``);
* ``'macroscopic'``: ``x^nu / nu!`` (the mean-field rate law with the pair counted once);
* ``'rate_law'``: ``x^nu`` (``k`` is the coefficient of the rate law, no ``1/nu!``).

All three have the same mean-field limit up to the ``1/nu!`` of the last two; the first
differs from ``'macroscopic'`` at relative order ``1/Omega``, the order of the one-loop
correction, which is why the oracle comparison uses the default.  ``'A <-> B'`` takes
``rate=(forward, reverse)``.  An empty side is written ``''``, ``'0'`` or ``'∅'``.
"""
from __future__ import annotations

import itertools
import math
import re
from typing import Any, Optional

import numpy as np

from engine.errors import DaedalusError

__all__ = ['JumpProcess', 'JumpProcessError']

_IDENT = re.compile(r'^[A-Za-z_][A-Za-z_0-9]*$')
_TERM = re.compile(r'^(?:(\d+)\s*\*?\s*)?([A-Za-z_][A-Za-z_0-9]*)$')
_NULLS = {'', '0', '∅', 'Ø', 'null', 'None'}
_ARROWS = (('<->', True), ('<=>', True), ('⇌', True), ('↔', True),
           ('->', False), ('→', False), ('=>', False))
_COMBINATORICS = ('stochastic', 'macroscopic', 'rate_law')
_RESERVED = {'Dt', 'Dx', 'Conv', 'exp', 'log', 'sqrt', 'sin', 'cos', 'tan', 'sinh', 'cosh',
             'tanh', 'Max', 'Min', 'Abs', 'abs', 'pi', 'I', 'E'}


class JumpProcessError(DaedalusError, ValueError):
    """A jump-process declaration that is ill-formed or refused (negative rate, bad jump
    vector, undeclared name, missing parameter value)."""


def _wrap(text: str) -> str:
    t = text.strip()
    return t if (_IDENT.match(t) or re.match(r'^\d+(\.\d*)?([eE][-+]?\d+)?$', t)) else f'({t})'


def _num_or_str(value, where):
    if isinstance(value, bool):
        raise JumpProcessError(f'{where}: a bool is not a rate.')
    if isinstance(value, (int, float, np.integer, np.floating)):
        return repr(float(value)) if not float(value).is_integer() else str(int(value))
    if isinstance(value, str):
        if not value.strip():
            raise JumpProcessError(f'{where}: empty rate.')
        return value.strip()
    raise JumpProcessError(f'{where}: rate must be a string or a number, got {value!r}.')


class JumpProcess:
    """Collect species, parameters and transitions of a jump process; see the module
    docstring.  All declaration methods return ``self``.

    Parameters
    ----------
    name : model name.
    system_size : name of the system-size parameter (default ``'Omega'``); the jump size of
        every transition is ``1/Omega``.  Give its value with :meth:`system_size`.
    check_rates : probe each rate for negative values on the lattice of densities
        ``n/Omega`` at the default parameters when the transition is declared (a rate that
        is negative there is refused).  Rates that cannot be probed (a parameter without a
        default) are not checked.
    """

    def __init__(self, name: str = 'jump process', *, system_size: str = 'Omega',
                 check_rates: bool = True):
        if not _IDENT.match(system_size):
            raise JumpProcessError(f'system_size name {system_size!r} is not an identifier.')
        self.name = name
        self._omega = system_size
        self._omega_value: Optional[float] = None
        self._check_rates = bool(check_rates)
        self._species: list[str] = []
        self._params: dict[str, dict] = {}
        self._transitions: list[dict] = []

    # ── declarations ────────────────────────────────────────────────────────

    def species(self, *names: str):
        """Declare species (state densities ``x = n / Omega``), in order."""
        if not names:
            raise JumpProcessError('species(): give at least one name.')
        for n in names:
            if not isinstance(n, str) or not _IDENT.match(n):
                raise JumpProcessError(f'species {n!r} is not an identifier.')
            if n in self._species or n in self._params or n == self._omega:
                raise JumpProcessError(f'name {n!r} is already declared.')
            if n in _RESERVED:
                raise JumpProcessError(f'species name {n!r} is reserved.')
            self._species.append(n)
        return self

    def parameter(self, name: str, default=None, *, domain: Optional[str] = None):
        """Declare a parameter with an optional default (needed by the master-equation
        oracle).  ``domain=None`` declares a positive default as ``'positive'``."""
        if name == self._omega:
            return self.system_size(default)
        if not isinstance(name, str) or not _IDENT.match(name):
            raise JumpProcessError(f'parameter {name!r} is not an identifier.')
        if name in self._species or name in self._params or name in _RESERVED:
            raise JumpProcessError(f'name {name!r} is already declared or reserved.')
        if default is not None:
            default = float(default)
            if not math.isfinite(default):
                raise JumpProcessError(f'parameter {name!r}: default must be finite.')
        self._params[name] = dict(default=default, domain=domain)
        return self

    def system_size(self, value=None):
        """Set the default value of the system size ``Omega`` (events change the densities
        by ``1/Omega``)."""
        if value is not None:
            value = float(value)
            if not (value > 0 and math.isfinite(value)):
                raise JumpProcessError(f'the system size must be positive, got {value!r}.')
        self._omega_value = value
        return self

    # ── transitions ─────────────────────────────────────────────────────────

    def transition(self, rate, jump, *, name: Optional[str] = None, per_volume: bool = False):
        """One jump channel: the densities change by ``jump / Omega`` at event rate ``rate``
        (``Omega * rate`` with ``per_volume=True``).  ``jump`` is ``{species: integer}`` or a
        sequence with one integer per species."""
        text = _num_or_str(rate, 'transition')
        if per_volume:
            text = f'{self._omega}*{_wrap(text)}'
        return self._add(text, self._jump_vector(jump, 'transition'), name)

    def reaction(self, equation: str, rate, *, name: Optional[str] = None,
                 combinatorics: str = 'stochastic'):
        """Mass-action reaction(s) from ``'A + B -> C'`` (``'2A -> B'``, ``'-> A'``,
        ``'A ->'``, ``'A <-> B'`` with ``rate=(forward, reverse)``).  ``rate`` is the rate
        constant (a number or an expression in the parameters)."""
        if combinatorics not in _COMBINATORICS:
            raise JumpProcessError(
                f'combinatorics must be one of {_COMBINATORICS}, got {combinatorics!r}.')
        if not isinstance(equation, str):
            raise JumpProcessError(f'reaction: equation must be a string, got {equation!r}.')
        arrow = next(((a, rev) for a, rev in _ARROWS if a in equation), None)
        if arrow is None:
            raise JumpProcessError(
                f"reaction {equation!r}: no arrow; write 'A + B -> C' (or '<->' for a pair).")
        a, reversible = arrow
        if equation.count(a) != 1:
            raise JumpProcessError(f'reaction {equation!r}: more than one arrow.')
        left, right = (self._side(s, equation) for s in equation.split(a))
        if reversible:
            if not (isinstance(rate, (tuple, list)) and len(rate) == 2):
                raise JumpProcessError(
                    f"reaction {equation!r}: a reversible reaction needs rate=(forward, reverse).")
            pairs = [(left, right, rate[0], '_f'), (right, left, rate[1], '_r')]
        else:
            if isinstance(rate, (tuple, list)):
                raise JumpProcessError(
                    f"reaction {equation!r}: one rate constant for an irreversible reaction.")
            pairs = [(left, right, rate, '')]
        for lhs, rhs, k, suffix in pairs:
            kt = _num_or_str(k, f'reaction {equation!r}')
            jump = {s: rhs.get(s, 0) - lhs.get(s, 0) for s in set(lhs) | set(rhs)}
            factors = [self._mass_action(s, nu, combinatorics) for s, nu in lhs.items()]
            text = '*'.join([self._omega, _wrap(kt)] + factors)
            self._add(text, self._jump_vector(jump, f'reaction {equation!r}'),
                      None if name is None else name + suffix)
        return self

    def _side(self, text, equation):
        out: dict[str, int] = {}
        if text.strip() in _NULLS:
            return out
        for term in text.split('+'):
            m = _TERM.match(term.strip())
            if not m:
                raise JumpProcessError(
                    f'reaction {equation!r}: cannot read {term.strip()!r} as [n] species.')
            nu = int(m.group(1)) if m.group(1) else 1
            sp = m.group(2)
            if nu < 1:
                raise JumpProcessError(f'reaction {equation!r}: stoichiometry of {sp!r} must be >= 1.')
            if sp not in self._species:
                raise JumpProcessError(
                    f'reaction {equation!r}: undeclared species {sp!r}; species are {self._species}.')
            out[sp] = out.get(sp, 0) + nu
        return out

    def _mass_action(self, sp, nu, comb):
        if nu == 1:
            return sp
        fact = math.factorial(nu)
        if comb == 'stochastic':
            body = '*'.join([sp] + [f'({sp} - {m}/{self._omega})' for m in range(1, nu)])
        else:
            body = f'{sp}^{nu}'
        if comb == 'rate_law':
            return body
        return f'{body}/{fact}'

    def _jump_vector(self, jump, where):
        if not self._species:
            raise JumpProcessError(f'{where}: declare the species first.')
        d = len(self._species)
        if isinstance(jump, dict):
            for s in jump:
                if s not in self._species:
                    raise JumpProcessError(
                        f'{where}: jump names undeclared species {s!r}; species are {self._species}.')
            raw = [jump.get(s, 0) for s in self._species]
        else:
            try:
                raw = list(jump) if not np.isscalar(jump) else [jump]
            except TypeError:
                raise JumpProcessError(f'{where}: jump must be a dict or a sequence, got {jump!r}.')
            if len(raw) != d:
                raise JumpProcessError(
                    f'{where}: jump {jump!r} has {len(raw)} component(s) for {d} species.')
        out = []
        for v in raw:
            if isinstance(v, bool) or not isinstance(v, (int, float, np.integer, np.floating)) \
                    or not float(v).is_integer():
                raise JumpProcessError(f'{where}: jump components must be integers, got {v!r}.')
            out.append(int(v))
        if not any(out):
            raise JumpProcessError(f'{where}: the zero jump is not a transition.')
        return tuple(out)

    def _add(self, rate_text, jump, name):
        idx = len(self._transitions)
        name = name or f't{idx}'
        if not _IDENT.match(name) or any(t['name'] == name for t in self._transitions):
            raise JumpProcessError(f'transition name {name!r} is not a unique identifier.')
        self._check_names(rate_text, name)
        if self._check_rates:
            self._probe_rate(rate_text, name)
        self._transitions.append(dict(rate=rate_text, jump=jump, name=name))
        return self

    # ── rate validation ─────────────────────────────────────────────────────

    def _parse(self, text, where):
        import sympy
        from sympy.parsing.sympy_parser import (parse_expr, standard_transformations,
                                                convert_xor)
        names = self._species + list(self._params) + [self._omega]
        try:
            expr = parse_expr(text, local_dict={n: sympy.Symbol(n) for n in names},
                              transformations=standard_transformations + (convert_xor,))
        except Exception as e:
            raise JumpProcessError(f'{where}: cannot parse rate {text!r}: {e}') from e
        return expr

    def _check_names(self, text, where):
        expr = self._parse(text, f'transition {where!r}')
        known = set(self._species) | set(self._params) | {self._omega}
        unknown = {s.name for s in expr.free_symbols} - known
        if unknown:
            raise JumpProcessError(
                f'transition {where!r}: rate {text!r} uses undeclared name(s) {sorted(unknown)}; '
                f'species {self._species}, parameters {sorted(self._params)}, '
                f'system size {self._omega!r}.')

    def _probe_rate(self, text, where):
        import sympy
        expr = self._parse(text, f'transition {where!r}')
        syms = {s.name: s for s in expr.free_symbols}
        subs = {}
        for n, s in syms.items():
            if n in self._params:
                v = self._params[n]['default']
                if v is None:
                    return                        # cannot probe without the value
                subs[s] = v
        omega = self._omega_value
        if self._omega in syms:
            subs[syms[self._omega]] = omega if omega is not None else 100.0
        expr = expr.subs(subs)
        state = [s for n, s in syms.items() if n in self._species]
        Om = omega if omega is not None else 100.0
        counts = [0, 1, 2, 3, 4, 5, 8, 10, 20, 50, 100, 200, 400]
        axes = [np.array(counts, float) / Om] * len(state)
        if not state:
            val = float(expr)
            if val < 0:
                raise JumpProcessError(f'transition {where!r}: negative rate {text!r} = {val:g}.')
            return
        fn = sympy.lambdify(state, expr, modules='numpy')
        if len(state) <= 3:
            grids = np.meshgrid(*axes, indexing='ij')
            args = [g.ravel() for g in grids]
        else:
            rng = np.random.default_rng(0)
            args = [rng.choice(a, 3000) for a in axes]
        vals = np.asarray(fn(*args), dtype=float) * np.ones_like(args[0])
        if not np.all(np.isfinite(vals)):
            raise JumpProcessError(f'transition {where!r}: rate {text!r} is not finite on the lattice.')
        bad = vals < -1e-12 * max(1.0, float(np.abs(vals).max()))
        if bad.any():
            i = int(np.argmax(bad))
            at = ', '.join(f'{s.name}={a[i]:g}' for s, a in zip(state, args))
            raise JumpProcessError(
                f'transition {where!r}: negative rate {text!r} = {vals[i]:g} at {at} '
                f'(default parameters, Omega={Om:g}); rates must be non-negative.')

    # ── inspection ──────────────────────────────────────────────────────────

    @property
    def transitions(self) -> list[dict]:
        """The declared transitions: ``[{'name', 'rate', 'jump'}]`` (jump in species order)."""
        return [dict(t) for t in self._transitions]

    def __repr__(self):
        return (f'JumpProcess({self.name!r}, species={self._species}, '
                f'{len(self._transitions)} transition(s))')

    def summary(self) -> str:
        """The transitions, one per line."""
        lines = [f'JumpProcess {self.name!r}: species {self._species}, system size {self._omega!r}']
        for t in self._transitions:
            jv = ', '.join(f'{s}{j:+d}' for s, j in zip(self._species, t['jump']) if j)
            lines.append(f"  {t['name']}: rate {t['rate']}   jump {jv}  (x 1/{self._omega})")
        return '\n'.join(lines)

    # ── the SDE-front-end model ─────────────────────────────────────────────

    def _require_declared(self):
        if not self._species:
            raise JumpProcessError('declare species first.')
        if not self._transitions:
            raise JumpProcessError('declare at least one transition or reaction.')

    def sde(self):
        """The SDE-front-end model: zero drift and, per transition, one state-dependent
        ``Poisson(rate=r(x), size=1/Omega)`` source coupled to the species with the jump
        components as coefficients."""
        from api.sde import SDE
        from api.noise import Poisson
        self._require_declared()
        s = SDE(self.name)
        for n, p in self._params.items():
            dom = p['domain']
            if dom is None and p['default'] is not None and p['default'] > 0:
                dom = 'positive'
            s.parameter(n, default=p['default'], domain=dom)
        s.parameter(self._omega, default=self._omega_value, domain='positive')
        for f in self._species:
            s.field(f)
        for f in self._species:
            s.drift(f, '0')
        for t in self._transitions:
            s.noise(t['name'], Poisson(rate=t['rate'], size=f'1/{self._omega}'),
                    couples={f: j for f, j in zip(self._species, t['jump']) if j})
        return s

    def build(self) -> dict:
        """The built model dict for ``compute_cumulants`` / ``dd.run`` (also accepted
        directly: ``dd.run(jp, cfg)``)."""
        return self.sde().build()

    def check(self, *args, **kwargs):
        """``SDE.check`` of the built model (declarations, action terms, vertices)."""
        return self.sde().check(*args, **kwargs)

    def latex(self, *args, **kwargs):
        """``SDE.latex`` of the built model."""
        return self.sde().latex(*args, **kwargs)

    # ── the exact master equation ───────────────────────────────────────────

    def _values(self, params):
        vals = {n: p['default'] for n, p in self._params.items()}
        vals[self._omega] = self._omega_value
        for k, v in (params or {}).items():
            if k not in vals:
                raise JumpProcessError(
                    f'unknown parameter {k!r}; parameters are {sorted(vals)}.')
            vals[k] = v
        missing = [k for k, v in vals.items() if v is None]
        if missing:
            raise JumpProcessError(
                f'no value for {missing}; give a default in parameter()/system_size() or '
                f'pass params={{...}}.')
        return {k: float(v) for k, v in vals.items()}

    def to_master_equation(self, params=None, *, box='auto', **kwargs):
        """The same transitions as an exact :class:`api.master_equation.MasterEquation` on the
        lattice ``x = n / Omega`` (``h = 1/Omega``).  ``params`` overrides parameter values
        (including the system size); ``box`` is ``'auto'`` (linear-noise box, lower bound 0) or
        index ranges; further keywords go to ``master_equation``."""
        from api.master_equation import master_equation, Transition
        self._require_declared()
        vals = self._values(params)
        Om = vals[self._omega]
        trs = [Transition(t['rate'], t['jump'], t['name']) for t in self._transitions]
        if isinstance(box, str) and box == 'auto':
            kwargs.setdefault('lower', 0.0)
        return master_equation(trs, fields=tuple(self._species), params=vals, h=1.0 / Om,
                               box=box, **kwargs)

    def compare(self, params=None, *, box='auto', ks=(1, 2), max_ell=1, config=None,
                verbose=False, me_options=None):
        """Loop-order cumulants of ``compute_cumulants`` against the exact master equation,
        as a :class:`api.master_equation.ComparisonResult` (``.rows``, ``.table()``,
        ``.notes``).  The expansion is asymptotic in ``1/Omega``: it should converge to the
        exact values as ``Omega`` grows, not match at a fixed small ``Omega``."""
        me = self.to_master_equation(params, box=box, **(me_options or {}))
        vals = self._values(params)
        return me.compare_with_daedalus(self.build(), ks=ks, max_ell=max_ell,
                                        parameters=vals, config=config, verbose=verbose)

    def convergence(self, omegas, params=None, *, box='auto', ks=(1, 2), max_ell=1,
                    config=None, me_options=None):
        """:meth:`compare` over a list of system sizes.  ``box`` may be a function of
        ``Omega``.  Returns ``{Omega: ComparisonResult}``."""
        out = {}
        for Om in omegas:
            p = dict(params or {})
            p[self._omega] = Om
            out[Om] = self.compare(p, box=box(Om) if callable(box) else box, ks=ks,
                                   max_ell=max_ell, config=config, me_options=me_options)
        return out
