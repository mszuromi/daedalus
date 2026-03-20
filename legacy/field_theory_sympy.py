"""
field_theory.py
===============
Model-agnostic MSRJD field theory expansion framework.

Usage
-----
from field_theory import FieldTheory
from models.hawkes import HAWKES_MODEL

ft = FieldTheory(HAWKES_MODEL, taylor_order=4)
ft.expand()
ft.sanity_check()
ft.summary()
S_free = ft.free_action()
"""

from __future__ import annotations
from typing import Any
import sympy as sp
from sympy import symbols, Add, Mul, Rational, factorial, exp


# ---------------------------------------------------------------------------
# Display classes (usable with both SymPy and SageMath kernels)
# ---------------------------------------------------------------------------

class Conv(sp.Function):
    """Conv(kappa, f) — convolution operator (κ∗f)(t)."""
    @classmethod
    def eval(cls, kappa, f):
        return None

    def _latex(self, printer):
        k, f = self.args
        return r'\left(' + printer.doprint(k) + r' \ast ' + printer.doprint(f) + r'\right)'

    def _sympystr(self, printer):
        k, f = self.args
        return f'Conv({printer.doprint(k)}, {printer.doprint(f)})'


class IP(sp.Function):
    """IP(a, b) — inner product / time integral ∫ a(t) b(t) dt."""
    @classmethod
    def eval(cls, a, b):
        return None

    def _latex(self, printer):
        a, b = self.args
        return printer.doprint(a) + r'^{\top} ' + printer.doprint(b)

    def _sympystr(self, printer):
        a, b = self.args
        return f'IP({printer.doprint(a)}, {printer.doprint(b)})'


# ---------------------------------------------------------------------------
# Namespace object — what the model's action lambda receives
# ---------------------------------------------------------------------------

class _Namespace:
    """Holds all symbols accessible to the action callable."""
    pass


# ---------------------------------------------------------------------------
# Degree-counting helpers (pure SymPy, no polynomial ring needed)
# ---------------------------------------------------------------------------

def _sym_degree(term: sp.Expr, syms: list) -> int:
    return sum(int(sp.degree(term, x)) for x in syms)


def _collect_by_bigrade(expr: sp.Expr, tilde_syms: list, phys_syms: list) -> dict:
    """
    Split an expanded SymPy expression by (n_tilde, n_phys) bigrade.
    Keys are (int, int), values are SymPy expressions.
    """
    expr = sp.expand(expr)
    result: dict = {}
    for term in Add.make_args(expr):
        n_t = _sym_degree(term, tilde_syms)
        n_p = _sym_degree(term, phys_syms)
        key = (n_t, n_p)
        result[key] = result.get(key, sp.Integer(0)) + term
    return result


# ---------------------------------------------------------------------------
# Taylor expansion helpers
# ---------------------------------------------------------------------------

def _poly_exp_m1(x: sp.Symbol, order: int) -> sp.Expr:
    """Taylor polynomial for e^x - 1 up to x^order."""
    return sum(x**n / factorial(n) for n in range(1, order + 1))


def _poly_taylor(f_coeffs: list, x: sp.Symbol) -> sp.Expr:
    """
    Build a Taylor polynomial from a list of coefficients [f0, f1, f2, ...].
    Result: sum_n (f_coeffs[n] / n!) * x^n
    """
    return sum(c * x**n / factorial(n)
               for n, c in enumerate(f_coeffs))


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class FieldTheory:
    """
    Model-agnostic MSRJD field theory expander.

    Parameters
    ----------
    model : dict
        Model specification (see models/hawkes.py for the dict format).
    taylor_order : int
        Maximum order for Taylor expansions of nonlinear functions.
    """

    def __init__(self, model: dict, taylor_order: int = 4):
        self.model = model
        self.taylor_order = taylor_order

        self._ns: _Namespace | None = None          # built namespace
        self._S_raw: sp.Expr | None = None          # raw expanded action
        self._by_tp: dict | None = None             # bigraded sectors
        self._tilde_syms: list = []
        self._phys_syms: list = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def expand(self) -> None:
        """Build namespace, call model action, expand, classify by bigrade."""
        ns = self._build_namespace()
        self._ns = ns

        raw = self.model['action'](ns)
        S = sp.expand(raw)

        # Apply MF background conditions (e.g. phi0_i = nstar_i, vstar EOM).
        # These are supplied by the model as a callable returning a subs dict.
        if 'mf_bg_conditions' in self.model:
            subs = self.model['mf_bg_conditions'](ns)
            S = sp.expand(S.subs(subs))

        self._S_raw = S
        self._tilde_syms = ns._tilde_syms
        self._phys_syms  = ns._phys_syms
        self._by_tp = _collect_by_bigrade(self._S_raw,
                                          self._tilde_syms,
                                          self._phys_syms)

    def sanity_check(self) -> bool:
        """
        Check expected zero sectors:
          (0,0) — constant (background should cancel)
          (1,0) — linear in response fields only (tadpole / MF condition)
          (0,1) — linear in physical fluctuations only (EOM residual)
        Returns True if all pass.
        """
        self._require_expanded()
        checks = [
            ((0, 0), 'constant term'),
            ((1, 0), 'tadpole — must vanish at MF saddle'),
            ((0, 1), 'linear physical-only — must vanish at EOM'),
        ]
        all_pass = True
        print('=== Sanity checks ===')
        for key, label in checks:
            val = sp.expand(self._by_tp.get(key, sp.Integer(0)))
            ok  = (val == 0)
            status = 'PASS' if ok else 'FAIL'
            print(f'  [{status}]  (n_tilde={key[0]}, n_phys={key[1]})  {label}')
            if not ok:
                try:
                    from IPython.display import display
                    display(val)
                except ImportError:
                    print('    ', val)
            all_pass = all_pass and ok
        return all_pass

    def summary(self) -> None:
        """Print all non-zero bigrade sectors."""
        self._require_expanded()
        try:
            from IPython.display import display, Math
            _display = display
        except ImportError:
            _display = print

        print('=== Action sectors ===')
        for (n_t, n_p), expr in sorted(self._by_tp.items()):
            e = sp.expand(expr)
            if e == 0:
                continue
            label = self._sector_label(n_t, n_p)
            print(f'  (n_tilde={n_t}, n_phys={n_p})  [{label}]:')
            _display(e)

    def free_action(self) -> sp.Expr:
        """Return the (1,1) sector — bilinear free action."""
        self._require_expanded()
        return sp.expand(self._by_tp.get((1, 1), sp.Integer(0)))

    def noise_kernel(self) -> dict:
        """Return all (≥2, 0) sectors — noise/source terms."""
        self._require_expanded()
        return {(n_t, n_p): sp.expand(e)
                for (n_t, n_p), e in self._by_tp.items()
                if n_t >= 2 and n_p == 0}

    def vertices(self) -> dict:
        """Return all sectors with total degree ≥ 3 (interaction vertices)."""
        self._require_expanded()
        return {(n_t, n_p): sp.expand(e)
                for (n_t, n_p), e in self._by_tp.items()
                if n_t + n_p >= 3}

    def sectors(self) -> dict:
        """Return the full bigrade dict."""
        self._require_expanded()
        return dict(self._by_tp)

    # ------------------------------------------------------------------
    # Namespace builder
    # ------------------------------------------------------------------

    # SymPy assumption keys accepted by Symbol()
    _SYMPY_ASSUMPTIONS = frozenset({
        'commutative', 'complex', 'real', 'imaginary', 'positive', 'negative',
        'nonnegative', 'nonpositive', 'zero', 'nonzero', 'integer', 'rational',
        'algebraic', 'transcendental', 'irrational', 'finite', 'infinite',
        'prime', 'composite', 'odd', 'even', 'hermitian', 'antihermitian',
    })

    @classmethod
    def _sym_kwargs(cls, spec: dict) -> dict:
        """Extract only valid SymPy assumption kwargs from a spec dict."""
        return {k: v for k, v in spec.items() if k in cls._SYMPY_ASSUMPTIONS}

    def _build_namespace(self) -> _Namespace:
        """
        Construct the _Namespace from the model dict.
        All ring generators and SR scalar symbols are attached here.
        """
        m   = self.model
        ns  = _Namespace()
        idx = m['index_sets']           # e.g. {'pop': [0, 1]}

        # --- Index sets ---
        for name, lst in idx.items():
            setattr(ns, name, list(lst))

        # The first index set is assumed to be the primary population index
        primary_idx = list(list(idx.values())[0])

        # --- Physical fluctuation field symbols ---
        phys_fields = m['physical_fields']   # list of dicts
        phys_syms: list = []
        for fspec in phys_fields:
            name    = fspec['name']          # e.g. 'dn'
            indexed = fspec.get('indexed', True)
            if indexed:
                arr = [symbols(f"{name}{i+1}") for i in primary_idx]
                setattr(ns, name, arr)
                phys_syms.extend(arr)
            else:
                sym = symbols(name)
                setattr(ns, name, sym)
                phys_syms.append(sym)

        # --- Response field symbols ---
        resp_fields = m['response_fields']   # list of dicts
        tilde_syms: list = []
        for fspec in resp_fields:
            name    = fspec['name']
            indexed = fspec.get('indexed', True)
            if indexed:
                arr = [symbols(f"{name}{i+1}") for i in primary_idx]
                setattr(ns, name, arr)
                tilde_syms.extend(arr)
            else:
                sym = symbols(name)
                setattr(ns, name, sym)
                tilde_syms.append(sym)

        ns._tilde_syms = tilde_syms
        ns._phys_syms  = phys_syms

        # --- Scalar parameters (SR symbols, not ring generators) ---
        for pspec in m.get('parameters', []):
            pname   = pspec['name']
            kwargs  = self._sym_kwargs(pspec)   # only valid SymPy assumption keys
            indexed = pspec.get('indexed', False)
            if indexed:
                arr = [symbols(f"{pname}{i+1}", **kwargs) for i in primary_idx]
                setattr(ns, pname, arr)
            else:
                sym = symbols(pname, **kwargs)
                setattr(ns, pname, sym)

        # --- Kernels (treated as SR symbols / algebraic placeholders) ---
        for kspec in m.get('kernels', []):
            sym = symbols(kspec.get('latex', kspec['name']))
            setattr(ns, kspec['name'], sym)

        # --- Operators (SR algebraic symbols) ---
        for ospec in m.get('operators', []):
            sym = symbols(ospec.get('latex', ospec['name']))
            setattr(ns, ospec['name'], sym)

        # Special kernel symbols that are always available
        ns.delta_D  = symbols(r'\delta')    # identity convolution kernel δ(t)
        ns.delta_Dp = symbols(r"\delta'")   # derivative kernel δ'(t)

        # --- Nonlinear functions (Taylor polynomial builders) ---
        for fspec in m.get('functions', []):
            fname   = fspec['name']
            coeffs  = fspec['taylor_coeffs']   # list of lists (one per population index)
            indexed = fspec.get('indexed', True)
            if indexed:
                # coeffs[i] = [f0_i, f1_i, f2_i, ...]  (SymPy exprs)
                def _make_fn(c_list):
                    def _fn(x, order=None):
                        o = order if order is not None else self.taylor_order
                        cs = c_list[:o+1]
                        return _poly_taylor(cs, x)
                    return _fn
                fns = [_make_fn(coeffs[i]) for i in primary_idx]
                setattr(ns, fname, lambda i, x, _fns=fns, **kw: _fns[i](x, **kw))
            else:
                c_list = coeffs
                def _make_fn_scalar(c_list):
                    def _fn(x, order=None):
                        o = order if order is not None else self.taylor_order
                        cs = c_list[:o+1]
                        return _poly_taylor(cs, x)
                    return _fn
                setattr(ns, fname, _make_fn_scalar(c_list))

        # --- exp(x)-1 Taylor polynomial ---
        def _exp_m1(x, order=None):
            o = order if order is not None else self.taylor_order
            return _poly_exp_m1(x, o)
        ns.exp_m1 = _exp_m1

        # --- Background / MF substitutions ---
        for sub in m.get('mf_substitutions', []):
            setattr(ns, sub['name'], sub['value'](ns))

        return ns

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _require_expanded(self):
        if self._by_tp is None:
            raise RuntimeError("Call expand() first.")

    @staticmethod
    def _sector_label(n_t: int, n_p: int) -> str:
        total = n_t + n_p
        if (n_t, n_p) == (1, 1):
            return 'free action'
        if n_t >= 2 and n_p == 0:
            return 'noise kernel'
        if total == 1:
            return 'tadpole / background'
        if total >= 3:
            return f'vertex (order {total})'
        return f'sector ({n_t},{n_p})'
