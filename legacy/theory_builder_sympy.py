"""
MSRJD Theory Builder — backend module.

Usage in a notebook:
    from theory_builder import Theory, TheoryUI
    ui = TheoryUI()
    ui.show()
"""
import re
import ipywidgets as w
from IPython.display import display, HTML, clear_output
import sympy as sp
import yaml
from pathlib import Path

_LATEX_OK = False
try:
    from sympy.parsing.latex import parse_latex
    _LATEX_OK = True
except ImportError:
    print('⚠ sympy.parsing.latex not available.')
    print('  Install with:  pip install antlr4-python3-runtime==4.11.1')

# ===========================================================================
# Custom SymPy operators: Conv and IP
# ===========================================================================

class Conv(sp.Function):
    """
    Conv(kappa, f) — convolution (κ ∗ f)(t) = ∫ κ(t−s) f(s) ds.

    LaTeX display : (κ ∗ f)
    SymPy input   : Conv(kappa, f)
    LaTeX input   : \\kappa \\ast f
    """
    @classmethod
    def eval(cls, kappa, f):
        return None   # keep symbolic

    def _eval_derivative(self, s):
        kappa, f = self.args
        return Conv(kappa, sp.diff(f, s))   # linear in second argument

    def _latex(self, printer):
        k, f = self.args
        return r'\left(' + printer.doprint(k) + r' \ast ' + printer.doprint(f) + r'\right)'

    def _sympystr(self, printer):
        k, f = self.args
        return 'Conv(' + printer.doprint(k) + ', ' + printer.doprint(f) + ')'


class IP(sp.Function):
    """
    IP(a, b) — inner product / transpose product  aᵀb = ∫ a(t) b(t) dt.

    LaTeX display : a^{\\top} b
    SymPy input   : IP(a, b)
    LaTeX input   : a^{\\top} b   or   a^T b
    """
    @classmethod
    def eval(cls, a, b):
        return None   # keep symbolic

    def _eval_derivative(self, s):
        a, b = self.args
        return IP(sp.diff(a, s), b) + IP(a, sp.diff(b, s))   # bilinear

    def _latex(self, printer):
        a, b = self.args
        return printer.doprint(a) + r'^{\top} ' + printer.doprint(b)

    def _sympystr(self, printer):
        a, b = self.args
        return 'IP(' + printer.doprint(a) + ', ' + printer.doprint(b) + ')'


# ===========================================================================
# LaTeX ↔ SymPy utilities
# ===========================================================================

_SPECIAL_LATEX = [
    (r'\Theta', 'Heaviside'),
    (r'\theta', 'Heaviside'),
    (r'\delta',  'DiracDelta'),
]

LATEX_CHEATSHEET = [
    (r'\tilde{v}',         'v_tilde',           'response field for v'),
    (r'\delta v',          'dv',                'fluctuation field (from MF background)'),
    (r'v^*',               'v_star',             'mean-field value of v'),
    (r'\phi(v)',           '(evaluates φ at v)', 'function evaluation'),
    (r"\phi'(v^*)",        "φ'(v*)",             'first derivative at mean field'),
    (r"\phi''(v^*)",       "φ''(v*)",            'second derivative at mean field'),
    (r'\phi^{(n)}(v^*)',   'φ^(n)(v*)',           'nth derivative at mean field'),
    (r'\phi(v^*, w)',      '(multi-arg eval)',    'multi-argument function'),
    (r'\kappa \ast v',          'Conv(kappa, v)',       'convolution κ ∗ v'),
    (r'\tilde{v}^{\top} v',    'IP(v_tilde, v)',       'inner product ṽᵀv (plain symbols only)'),
    (r'\partial_t v',          'del_t*v',             'operator × field (define operator first)'),
    (r'IP(v_tilde, del_t * v)','IP(v_tilde, del_t*v)','operator in inner product (SymPy mode)'),
    (r'\omega',            'omega',              'frequency variable'),
    (r'\tau',              'tau',                'time constant'),
    (r'\theta(t)',         'Heaviside(t)',        'Heaviside step function θ(t)'),
    (r'\delta(t)',         'DiracDelta(t)',       'Dirac delta δ(t)'),
    (r'-i\omega',         '-I*omega',            'imaginary frequency'),
    (r'\frac{1}{a+b}',    '1/(a+b)',             'fraction'),
]


def _split_args(s):
    """Split a comma-separated string respecting brace nesting."""
    parts, depth, cur = [], 0, []
    for c in s:
        if   c == '{': depth += 1; cur.append(c)
        elif c == '}': depth -= 1; cur.append(c)
        elif c == ',' and depth == 0:
            parts.append(''.join(cur).strip()); cur = []
        else:
            cur.append(c)
    if cur: parts.append(''.join(cur).strip())
    return [p for p in parts if p]


def _preprocess_func_derivatives(s, func_defs):
    """
    Replace function-call and derivative notation with placeholder tokens.

    Handles (in order, longest patterns first):
      \\phi^{(n)}(arg, ...)   nth derivative w.r.t. first argument
      \\phi''(arg, ...)       prime notation
      \\phi(arg, ...)         plain evaluation

    Returns (s_with_tokens, token_dict) where token_dict maps token_str -> sympy value.
    Tokens are re-substituted in latex_to_sympy via sp.latex() before parse_latex.

    func_defs entries must have:
      'latex_name' : str           e.g. r'\\phi'
      'args'       : list[str]     dummy variable names
      'expr_sym'   : sympy expr    function body
    """
    subs    = {}
    counter = [0]

    def _tok():
        t = f'PLHD{counter[0]}'; counter[0] += 1; return t

    def _eval(body, arg_syms, args_latex_str, n_deriv=0):
        """Differentiate body n_deriv times w.r.t. first arg, then substitute all args."""
        arg_strs = _split_args(args_latex_str)
        subs_d   = {}
        for sym, astr in zip(arg_syms, arg_strs):
            try:
                subs_d[sym] = parse_latex(astr.strip())
            except Exception:
                subs_d[sym] = sp.Symbol(f'_unk_{sym}')
        diff_sym = arg_syms[0] if arg_syms else None
        result   = sp.diff(body, diff_sym, n_deriv) if (n_deriv and diff_sym) else body
        return result.subs(subs_d)

    for fdef in func_defs:
        lname    = fdef['latex_name']
        lre      = re.escape(lname)
        arg_syms = [sp.Symbol(a) for a in fdef['args']]
        body     = fdef['expr_sym']

        # 1. Superscript nth derivative: \phi^{(n)}(...)
        pat = lre + r'\^\{\(([0-9]+)\)\}\(([^()]*)\)'
        while (m := re.search(pat, s)):
            try:    val = _eval(body, arg_syms, m.group(2), int(m.group(1)))
            except: val = sp.Symbol(f'_unkND{m.group(1)}')
            tok = _tok(); subs[tok] = val
            s = s[:m.start()] + tok + s[m.end():]

        # 2. Prime notation: \phi''(...)
        pat = lre + r"('+)\(([^()]*)\)"
        while (m := re.search(pat, s)):
            try:    val = _eval(body, arg_syms, m.group(2), len(m.group(1)))
            except: val = sp.Symbol(f'_unkP{len(m.group(1))}')
            tok = _tok(); subs[tok] = val
            s = s[:m.start()] + tok + s[m.end():]

        # 3. Plain evaluation: \phi(...)
        pat = lre + r'\(([^()]*)\)'
        while (m := re.search(pat, s)):
            try:    val = _eval(body, arg_syms, m.group(1), 0)
            except: val = sp.Symbol('_unkE')
            tok = _tok(); subs[tok] = val
            s = s[:m.start()] + tok + s[m.end():]

    return s, subs


# Token regex: matches Q_{n} placeholder tokens, LaTeX commands, or plain words
_TOK = r'(?:Q_\{\d+\}|\\[a-zA-Z]+|\w+)'


def _preprocess_conv_ip(s, tilde_subs, alias_subs, param_aliases, start_idx):
    """
    After tilde/alias tokenisation, replace conv/IP LaTeX operators with Q tokens
    whose SymPy values are Conv(...) or IP(...) built directly from already-resolved
    symbol lookups.

    Returns (s_updated, conv_ip_subs_dict, next_free_idx).
    """
    subs = {}
    idx  = [start_idx]

    def _tok_to_sym(tok):
        """Map a single post-tokenisation LaTeX token to a SymPy symbol."""
        qm = re.match(r'^Q_\{(\d+)\}$', tok)
        if qm:
            q = sp.Symbol(f'Q_{{{qm.group(1)}}}')
            # Check conv_ip subs first (for nested conv/IP tokens created earlier)
            return subs.get(q, tilde_subs.get(q, alias_subs.get(q, q)))
        if param_aliases and tok in param_aliases:
            return sp.Symbol(param_aliases[tok])
        return sp.Symbol(tok.lstrip('\\'))

    def _sub_conv(m):
        sa, sb = _tok_to_sym(m.group(1)), _tok_to_sym(m.group(2))
        tok = f'Q_{{{idx[0]}}}'; subs[sp.Symbol(tok)] = Conv(sa, sb); idx[0] += 1
        return tok

    def _sub_ip(m):
        sa, sb = _tok_to_sym(m.group(1)), _tok_to_sym(m.group(2))
        tok = f'Q_{{{idx[0]}}}'; subs[sp.Symbol(tok)] = IP(sa, sb); idx[0] += 1
        return tok

    # A \ast B
    s = re.sub(rf'({_TOK})\s*\\ast\s*({_TOK})', _sub_conv, s)
    # A^{\top} B
    s = re.sub(rf'({_TOK})\^\{{\\top\}}\s*({_TOK})', _sub_ip, s)
    # A^\top B  (no braces)
    s = re.sub(rf'({_TOK})\^\\top\s*({_TOK})', _sub_ip, s)
    # A^T B  (plain capital-T shorthand, only before a symbol to avoid x^T_0 false matches)
    s = re.sub(rf'({_TOK})\^T\s+({_TOK})', _sub_ip, s)

    return s, subs, idx[0]


def latex_to_sympy(latex_str, fields, param_aliases=None,
                   func_defs=None, field_star_map=None, field_fluct_map=None):
    """
    Convert a LaTeX expression string to a SymPy expression.

    Pre-processing order
    --------------------
    1. \\delta v  → fluct symbol   (field_fluct_map)
    2. v^*        → star symbol    (field_star_map)
    3. \\phi(...)  → evaluate/diff  (func_defs), tokens stored
    4. \\tilde{v}  → v_tilde
    5. param aliases
    6. special functions (Heaviside, DiracDelta)
    7. replace placeholder tokens with sp.latex(val) so parse_latex sees valid LaTeX
    8. parse_latex
    """
    if not _LATEX_OK:
        raise ImportError('sympy.parsing.latex not available')

    s = latex_str.strip()

    # 1. \delta {field} or \delta field  →  fluct_name
    if field_fluct_map:
        for f, fluct in sorted(field_fluct_map.items(), key=lambda x: -len(x[0])):
            fe = re.escape(f)
            # Use lambda replacement to avoid re interpreting backslashes in fluct name
            s  = re.sub(rf'\\delta\s*\{{{fe}\}}', lambda _, r=fluct: r, s)
            s  = re.sub(rf'\\delta\s+{fe}(?![a-zA-Z_0-9])', lambda _, r=fluct: r, s)

    # 2. field^* / field^{*} / field^{\ast}  →  star_name
    if field_star_map:
        for f, star in sorted(field_star_map.items(), key=lambda x: -len(x[0])):
            fe = re.escape(f)
            # Use lambda replacement to avoid re interpreting backslashes in star name
            s  = re.sub(fe + r'\^\{\\ast\}', lambda _, r=star: r, s)
            s  = re.sub(fe + r'\^\{\*\}',    lambda _, r=star: r, s)
            s  = re.sub(fe + r'\^\*',         lambda _, r=star: r, s)

    # 3. Function evaluation / derivatives
    func_subs = {}
    if func_defs:
        s, func_subs = _preprocess_func_derivatives(s, func_defs)

    # 4. \tilde{field} → placeholder token Q_{idx}
    #    parse_latex cannot handle multi-letter subscripts like v_tilde correctly
    #    (it parses 'v_tilde' as v*t*i*l*d*e), so use Q_{0}, Q_{1} etc. as tokens
    #    and substitute back after parse_latex.
    tilde_subs = {}
    for idx, f in enumerate(sorted(fields, key=len, reverse=True)):
        tok_latex = f'Q_{{{idx}}}'           # e.g. Q_{0}  →  Symbol('Q_{0}')
        tilde_subs[sp.Symbol(f'Q_{{{idx}}}')] = sp.Symbol(f + '_tilde')
        s = s.replace(f'\\tilde{{{f}}}', tok_latex)

    # 5. Parameter aliases → placeholder tokens (indices continue after tilde tokens)
    #    Direct replacement of e.g. \alpha with 'alpha' breaks parse_latex (treats
    #    multi-letter words as products of single letters).  Use Q_{n} tokens instead.
    alias_subs = {}
    if param_aliases:
        for idx, (lk, pn) in enumerate(
                sorted(param_aliases.items(), key=lambda x: -len(x[0]))):
            tok_idx  = len(fields) + idx
            tok_latex = f'Q_{{{tok_idx}}}'
            alias_subs[sp.Symbol(f'Q_{{{tok_idx}}}')] = sp.Symbol(pn)
            s = s.replace(lk, tok_latex)

    # 6. Special functions
    for latex_frag, py_frag in _SPECIAL_LATEX:
        s = s.replace(latex_frag, py_frag)

    # 6.5. Conv / IP operators  (after tilde+alias tokenisation, so Q_{n} tokens are set)
    n_aliases = len(param_aliases) if param_aliases else 0
    s, conv_ip_subs, _ = _preprocess_conv_ip(
        s, tilde_subs, alias_subs, param_aliases,
        start_idx=len(fields) + n_aliases)

    # 7. Replace placeholder tokens with their LaTeX representations.
    #    This avoids parse_latex treating 'PLHD0' as P*L*H*D*0 = 0.
    #    sp.latex(val) converts the computed sympy value back to valid LaTeX,
    #    which parse_latex can then parse correctly.
    if func_subs:
        for tok, val in func_subs.items():
            s = s.replace(tok, f'({sp.latex(val)})')

    # 8. Parse
    result = parse_latex(s)

    # 9. Substitute Q_{n} placeholder tokens back (tilde fields, param aliases, conv/IP)
    all_subs = {**tilde_subs, **alias_subs, **conv_ip_subs}
    if all_subs:
        result = result.subs(all_subs)

    return result


def _cheatsheet_html():
    rows = ''.join(
        f"<tr><td style='font-family:monospace;padding:2px 10px'>{lt}</td>"
        f"<td style='font-family:monospace;padding:2px 10px;color:#1a6e1a'>{r}</td>"
        f"<td style='color:#555;padding:2px 10px'>{d}</td></tr>"
        for lt, r, d in LATEX_CHEATSHEET
    )
    return (
        "<details style='margin:6px 0'><summary style='cursor:pointer;color:#336;font-size:12px'>"
        "LaTeX cheat sheet (click to expand)</summary>"
        "<table style='font-size:12px;border-collapse:collapse;margin-top:4px'>"
        "<tr><th style='text-align:left;padding:2px 10px'>LaTeX</th>"
        "<th style='text-align:left;padding:2px 10px'>Result</th>"
        "<th style='text-align:left;padding:2px 10px'>Meaning</th></tr>"
        + rows + "</table></details>"
    )


# ===========================================================================
# Theory dataclass
# ===========================================================================
class Theory:
    RESPONSE_SUFFIX = '_tilde'

    def __init__(self):
        self.name        = ''
        self.description = ''
        self.stationary  = True
        self.fields      = []
        self.parameters  = []   # {'symbol', 'latex_alias', 'description'}
        # {'name', 'latex_name', 'args': list[str], 'expr'}
        self.func_defs   = []
        # {'name', 'latex_name', 'expr_time', 'expr_fourier'}
        self.filter_defs = []
        # {'name', 'latex_name', 'fourier', 'description'}
        self.operator_defs = []
        # MF background (always saved, used in both manual and auto modes)
        self.mf_background  = []   # [{'field','star','fluct'}]
        self.mf_equations   = ''
        # Propagators
        self.prop_exprs  = {}   # (i,j) -> str
        self.prop_domain = {}   # (i,j) -> 'frequency'|'time'
        # Vertices
        self.vertex_mode = 'manual'
        # Each vertex: {'factor','output','input','description','expr'}
        # expr = factor * output * input  (computed on save)
        self.vertices    = []
        # Auto-derive metadata
        self.full_action = ''
        self.expand_order = 4
        self.input_mode  = 'sympy'

    @property
    def response_fields(self):
        return [f + self.RESPONSE_SUFFIX for f in self.fields]

    @property
    def param_symbols(self):
        return [p['symbol'] for p in self.parameters]

    @property
    def all_symbols(self):
        extras    = ['omega', 't'] + ([] if self.stationary else ['t1'])
        bg_syms   = [bg['star']  for bg in self.mf_background] + \
                    [bg['fluct'] for bg in self.mf_background]
        func_args    = [a for fd in self.func_defs for a in fd.get('args', [])]
        filter_names   = [fd['name'] for fd in self.filter_defs   if fd.get('name')]
        operator_names = [fd['name'] for fd in self.operator_defs if fd.get('name')]
        return (self.fields + self.response_fields + self.param_symbols
                + bg_syms + func_args + filter_names + operator_names + extras)

    def _sympy_locals(self):
        d = {s: sp.Symbol(s) for s in self.all_symbols}
        d['I'] = sp.I; d['Heaviside'] = sp.Heaviside; d['DiracDelta'] = sp.DiracDelta
        d['Conv'] = Conv; d['IP'] = IP
        return d

    def validate_expr(self, expr_str):
        if not expr_str or expr_str.strip() == '0':
            return True, sp.Integer(0)
        try:
            return True, sp.sympify(expr_str, locals=self._sympy_locals())
        except Exception as e:
            return False, str(e)

    def get_propagator(self, i, j):
        ok, expr = self.validate_expr(self.prop_exprs.get((i, j), '0'))
        return expr if ok else sp.Integer(0)

    def get_poles(self, i, j):
        if self.prop_domain.get((i, j), 'frequency') != 'frequency': return []
        G = self.get_propagator(i, j)
        if G == 0: return []
        omega = sp.Symbol('omega')
        _, denom = sp.fraction(sp.together(G))
        return sp.solve(denom, omega)

    def get_interacting_action(self):
        terms = []
        for v in self.vertices:
            ok, expr = self.validate_expr(v.get('expr', ''))
            if ok and expr != 0: terms.append(expr)
        return sp.Add(*terms) if terms else sp.Integer(0)

    # ── Serialisation ─────────────────────────────────────────────────────
    def to_dict(self):
        n = len(self.fields)
        pm = [['0']         * n for _ in range(n)]
        pd = [['frequency'] * n for _ in range(n)]
        for (i,j), e in self.prop_exprs.items():  pm[i][j] = e
        for (i,j), d in self.prop_domain.items(): pd[i][j] = d
        return {
            'name':              self.name,
            'description':       self.description,
            'stationary':        self.stationary,
            'response_suffix':   self.RESPONSE_SUFFIX,
            'fields':            self.fields,
            'parameters':        self.parameters,
            'func_defs':         self.func_defs,
            'filter_defs':       self.filter_defs,
            'operator_defs':     self.operator_defs,
            'mf_background':     self.mf_background,
            'mf_equations':      self.mf_equations,
            'propagator_matrix': pm,
            'propagator_domain': pd,
            'vertex_mode':       self.vertex_mode,
            'vertices':          self.vertices,
            'full_action':       self.full_action,
            'expand_order':      self.expand_order,
            'input_mode':        self.input_mode,
        }

    def save(self, path):
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, 'w') as f:
            yaml.dump(self.to_dict(), f,
                      default_flow_style=False, sort_keys=False, allow_unicode=True)
        return str(p)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            data = yaml.safe_load(f)
        t = cls()
        t.name           = data.get('name', '')
        t.description    = data.get('description', '')
        t.stationary     = data.get('stationary', True)
        t.fields         = data.get('fields', [])
        t.parameters     = data.get('parameters', [])
        t.func_defs      = data.get('func_defs', [])
        t.filter_defs    = data.get('filter_defs', [])
        t.operator_defs  = data.get('operator_defs', [])
        t.mf_background  = data.get('mf_background', [])
        t.mf_equations   = data.get('mf_equations', '')
        for i,row in enumerate(data.get('propagator_matrix', [])):
            for j,e in enumerate(row): t.prop_exprs[(i,j)]  = e
        for i,row in enumerate(data.get('propagator_domain', [])):
            for j,d in enumerate(row): t.prop_domain[(i,j)] = d
        t.vertex_mode  = data.get('vertex_mode', 'manual')
        t.vertices     = data.get('vertices', [])
        t.full_action  = data.get('full_action', '')
        t.expand_order = data.get('expand_order', 4)
        t.input_mode   = data.get('input_mode', 'sympy')
        return t

    def summary(self):
        lines = [
            f'Theory     : {self.name}',
            f'Descr      : {self.description}',
            f'Stationary : {self.stationary}',
            f'Fields     : {self.fields}  →  {self.response_fields}',
            f'Params     : {self.param_symbols}',
        ]
        if self.func_defs:
            lines.append('Functions  :')
            for fd in self.func_defs:
                args = ', '.join(fd.get('args', ['x']))
                lines.append(f'  {fd["latex_name"]}({args}) = {fd["expr"]}')
        if self.operator_defs:
            lines.append('Operators  :')
            for od in self.operator_defs:
                lines.append(f'  {od["name"]} ({od["latex_name"]})'
                             + (f'  fourier: {od["fourier"]}' if od.get('fourier') else '')
                             + (f'  # {od["description"]}' if od.get('description') else ''))
        if self.filter_defs:
            lines.append('Filters    :')
            for fd in self.filter_defs:
                t_expr = fd.get('expr_time', '')
                f_expr = fd.get('expr_fourier', '')
                lines.append(f'  {fd["name"]} ({fd["latex_name"]})'
                             + (f'  time: {t_expr}' if t_expr else '')
                             + (f'  fourier: {f_expr}' if f_expr else ''))
        if self.mf_background:
            lines.append('MF background:')
            for bg in self.mf_background:
                lines.append(f'  {bg["field"]} → {bg["star"]} + δ·{bg["fluct"]}')
        lines += ['', 'Propagator matrix:']
        for i in range(len(self.fields)):
            for j in range(len(self.fields)):
                e   = self.prop_exprs.get((i,j), '0')
                dom = self.prop_domain.get((i,j), 'frequency')
                if e and e != '0':
                    poles = self.get_poles(i, j)
                    lines.append(f'  G[{self.fields[i]}][{self.response_fields[j]}]  [{dom}] = {e}')
                    if poles: lines.append(f'    poles: {poles}')
        lines += ['', f'Vertices  (mode: {self.vertex_mode}):']
        for k, v in enumerate(self.vertices):
            d = f"  # {v['description']}" if v.get('description') else ''
            lines.append(f'  [{k+1}] {v.get("expr", "?")}  '
                         f'[factor={v.get("factor","?")}, '
                         f'out={v.get("output","?")}, '
                         f'in={v.get("input","?")}]{d}')
        if not self.vertices: lines.append('  (none)')
        return '\n'.join(lines)


# ===========================================================================
# TheoryUI
# ===========================================================================
class TheoryUI:

    _SECTION = "font-family:monospace;font-size:14px;font-weight:bold;margin-top:10px;margin-bottom:4px;"
    _HINT    = "color:#666;font-size:11px;"
    # NOTE: Layout objects are created per-instance in _build() (not class-level)
    # to ensure they are registered with the Jupyter comm AFTER display() is ready.

    def __init__(self):
        self.theory        = Theory()
        self._field_rows   = []
        self._param_rows   = []
        self._func_rows    = []   # (name_w, latex_w, args_w, expr_w, row)
        self._filter_rows   = []   # (name_w, latex_w, time_w, fourier_w, row)
        self._operator_rows = []   # (name_w, latex_w, fourier_w, desc_w, row)
        self._prop_widgets = {}
        self._vertex_rows  = []   # (factor_w, output_w, input_w, desc_w, preview_w, box)
        self._bg_rows      = []   # (field_name, star_w, fluct_w, row)
        self._input_mode   = 'sympy'
        self._stationary   = True
        self._vertex_mode  = 'manual'
        self._build()

    # =========================================================================
    # Build
    # =========================================================================
    def _build(self):
        # Layout objects created here (not at class level) so they are
        # registered with the Jupyter/VS Code widget comm after display() is ready.
        self._W_FIELD = w.Layout(width='140px')
        self._W_SYM   = w.Layout(width='100px')
        self._W_ALIAS = w.Layout(width='100px')
        self._W_DESC  = w.Layout(width='210px')
        self._W_PROP  = w.Layout(width='260px')
        self._W_BTN   = w.Layout(width='36px')

        header = w.HTML(
            "<h2 style='font-family:monospace;margin-bottom:4px'>MSRJD Theory Builder</h2>"
            "<p style='color:#555;font-size:12px;margin-top:0'>"
            "Response fields auto-generated as <code>field_tilde</code>.</p>"
        )

        # Global toggles
        self._mode_toggle = w.ToggleButtons(
            options=[('SymPy', 'sympy'), ('LaTeX', 'latex')],
            value='sympy', description='Input mode:',
            style={'description_width': '90px'},
        )
        self._stat_toggle = w.ToggleButtons(
            options=[('Stationary', True), ('Non-stationary', False)],
            value=True, description='Theory type:',
            style={'description_width': '90px'},
        )
        self._cheatsheet_w = w.HTML(_cheatsheet_html() if _LATEX_OK else '')
        self._cheatsheet_w.layout.display = 'none'
        latex_warn = w.HTML(
            '' if _LATEX_OK else
            "<span style='color:red;font-size:11px'>"
            "⚠ LaTeX parsing unavailable — pip install antlr4-python3-runtime==4.11.1</span>"
        )
        self._mode_toggle.observe(self._on_mode_change, names='value')
        self._stat_toggle.observe(self._on_stat_change, names='value')

        toggles_sec = w.VBox([
            w.HTML(f"<div style='{self._SECTION}'>Global settings</div>"),
            self._mode_toggle, self._stat_toggle, latex_warn, self._cheatsheet_w,
        ])

        # Name / description
        self._name_w = w.Text(placeholder='e.g. Nonlinear Hawkes', layout=w.Layout(width='380px'))
        self._desc_w = w.Text(placeholder='optional description',   layout=w.Layout(width='380px'))
        meta = w.VBox([
            w.HTML(f"<div style='{self._SECTION}'>Theory info</div>"),
            w.HBox([w.Label('Name:',        layout=w.Layout(width='90px')), self._name_w]),
            w.HBox([w.Label('Description:', layout=w.Layout(width='90px')), self._desc_w]),
        ])

        # Fields
        self._fields_vbox = w.VBox([])
        add_field_btn = w.Button(description='+ field', button_style='info',
                                 layout=w.Layout(width='90px'))
        add_field_btn.on_click(self._add_field)
        fields_sec = w.VBox([
            w.HTML(f"<div style='{self._SECTION}'>Fields</div>"
                   f"<div style='{self._HINT}'>Physical fields only — "
                   "<code>field_tilde</code> auto-generated</div>"),
            self._fields_vbox, add_field_btn,
        ])

        # Parameters
        self._params_vbox = w.VBox([])
        add_param_btn = w.Button(description='+ parameter', button_style='info',
                                 layout=w.Layout(width='110px'))
        add_param_btn.on_click(self._add_param)
        params_sec = w.VBox([
            w.HTML(f"<div style='{self._SECTION}'>Parameters</div>"
                   f"<div style='{self._HINT}'>Symbol | LaTeX alias | Description</div>"),
            self._params_vbox, add_param_btn,
        ])

        # Functions
        func_sec = self._build_func_section()

        # Filters / Kernels
        filter_sec = self._build_filter_section()

        # Operators
        operator_sec = self._build_operator_section()

        # Mean-field background (always visible)
        mf_sec = self._build_mf_section()

        # Propagator
        self._prop_outer = w.VBox([
            w.HTML(f"<div style='{self._SECTION}'>Propagator matrix</div>"
                   f"<div style='{self._HINT}'><i>Add fields to populate.</i></div>")
        ])

        # Vertices
        vertex_sec = self._build_vertex_section()

        # Save
        self._path_w   = w.Text(value='theories/my_theory.yaml', layout=w.Layout(width='340px'))
        self._save_out = w.Output()
        save_btn = w.Button(description='Save YAML', button_style='success',
                            layout=w.Layout(width='110px'))
        save_btn.on_click(self._save)
        save_sec = w.VBox([
            w.HTML(f"<div style='{self._SECTION}'>Save</div>"),
            w.HBox([w.Label('Output path:', layout=w.Layout(width='90px')),
                    self._path_w, save_btn]),
            self._save_out,
        ])

        def _hr():
            return w.HTML("<hr style='border:none;border-top:1px solid #ddd;margin:10px 0'>")

        self._ui = w.VBox(
            [header, _hr(), toggles_sec, _hr(), meta, _hr(),
             fields_sec, _hr(), params_sec, _hr(),
             func_sec, _hr(), filter_sec, _hr(), operator_sec, _hr(),
             mf_sec, _hr(), self._prop_outer, _hr(), vertex_sec, _hr(), save_sec],
            layout=w.Layout(padding='18px', max_width='980px')
        )

    # =========================================================================
    # Function definitions section
    # =========================================================================
    def _build_func_section(self):
        self._funcs_vbox = w.VBox([])
        add_btn = w.Button(description='+ function', button_style='info',
                           layout=w.Layout(width='100px'))
        add_btn.on_click(self._add_func)
        return w.VBox([
            w.HTML(
                f"<div style='{self._SECTION}'>Field Functions</div>"
                f"<div style='{self._HINT}'>"
                "Nonlinear functions of <em>field variables</em> (e.g. φ(v), exp(v/v₀)). "
                "Multiple arguments separated by commas.<br>"
                "In LaTeX mode: <code>\\phi''(v^*)</code> differentiates w.r.t. the "
                "<em>first</em> argument and evaluates at all given arguments.<br>"
                "<b>Name</b> = Python id &nbsp;|&nbsp; "
                "<b>LaTeX</b> = symbol in expressions (e.g. <code>\\phi</code>) &nbsp;|&nbsp; "
                "<b>Args</b> = comma-separated dummy variables &nbsp;|&nbsp; "
                "<b>Expr</b> = body in terms of Args"
                "</div>"
            ),
            self._funcs_vbox, add_btn,
        ])

    def _add_func(self, _=None, name='', latex_name='', args='x', expr=''):
        name_w  = w.Text(value=name,       placeholder='phi',      layout=w.Layout(width='80px'))
        latex_w = w.Text(value=latex_name, placeholder=r'\phi',    layout=w.Layout(width='80px'))
        args_w  = w.Text(value=args,       placeholder='x  or  x, y', layout=w.Layout(width='100px'))
        expr_w  = w.Text(value=expr,       placeholder='x**2/b',   layout=w.Layout(width='300px'))
        preview_w = w.Output(layout=w.Layout(min_height='24px', display='none'))
        rbtn    = w.Button(description='×', button_style='danger', layout=self._W_BTN)

        row = w.VBox([
            w.HBox([
                w.Label('name:',  layout=w.Layout(width='44px')), name_w,
                w.Label('LaTeX:', layout=w.Layout(width='48px')), latex_w,
                w.Label('args:',  layout=w.Layout(width='38px')), args_w,
                w.Label('expr:',  layout=w.Layout(width='40px')), expr_w,
                rbtn,
            ]),
            preview_w,
        ], layout=w.Layout(border='1px solid #e0e4f0', padding='4px 6px', margin='3px 0'))

        def _on_change(change, _pw=preview_w):
            if self._input_mode == 'latex':
                ln  = latex_w.value.strip() or name_w.value.strip() or 'f'
                av  = args_w.value.strip()  or 'x'
                ex  = expr_w.value.strip()
                self._render_math(_pw, f'{ln}({av}) = {ex}' if ex else '')

        def _on_remove(_):
            self._func_rows = [r for r in self._func_rows if r[0] is not name_w]
            self._funcs_vbox.children = [r[-1] for r in self._func_rows]

        for ww in (name_w, latex_w, args_w, expr_w):
            ww.observe(_on_change, names='value')
        rbtn.on_click(_on_remove)
        self._func_rows.append((name_w, latex_w, args_w, expr_w, row))
        self._funcs_vbox.children = [r[-1] for r in self._func_rows]

    # =========================================================================
    # Filter / Kernel section
    # =========================================================================
    def _build_filter_section(self):
        self._filters_vbox = w.VBox([])
        add_btn = w.Button(description='+ filter', button_style='info',
                           layout=w.Layout(width='90px'))
        add_btn.on_click(self._add_filter)
        return w.VBox([
            w.HTML(
                f"<div style='{self._SECTION}'>Filters / Kernels</div>"
                f"<div style='{self._HINT}'>"
                "Define convolution kernels for use in expressions.<br>"
                "In LaTeX mode: <code>\\kappa \\ast v</code> → <code>Conv(kappa, v)</code> &nbsp;|&nbsp; "
                "<code>\\tilde{{v}}^{{\\top}} v</code> → <code>IP(v_tilde, v)</code><br>"
                "<b>Name</b> = Python id &nbsp;|&nbsp; "
                "<b>LaTeX</b> = symbol (e.g. <code>\\kappa</code>) &nbsp;|&nbsp; "
                "<b>Time expr</b> = κ(t) &nbsp;|&nbsp; "
                "<b>Fourier expr</b> = κ̂(ω)"
                "</div>"
            ),
            self._filters_vbox, add_btn,
        ])

    def _add_filter(self, _=None, name='', latex_name='', expr_time='', expr_fourier=''):
        name_w    = w.Text(value=name,         placeholder='kappa',      layout=w.Layout(width='80px'))
        latex_w   = w.Text(value=latex_name,   placeholder=r'\kappa',    layout=w.Layout(width='80px'))
        time_w    = w.Text(value=expr_time,    placeholder='exp(-t/tau)*Heaviside(t)',
                           layout=w.Layout(width='240px'))
        fourier_w = w.Text(value=expr_fourier, placeholder='1/(1 + I*omega*tau)',
                           layout=w.Layout(width='220px'))
        rbtn = w.Button(description='×', button_style='danger', layout=self._W_BTN)

        row = w.VBox([
            w.HBox([
                w.Label('name:',    layout=w.Layout(width='44px')), name_w,
                w.Label('LaTeX:',   layout=w.Layout(width='48px')), latex_w,
                w.Label('time:',    layout=w.Layout(width='40px')), time_w,
                w.Label('Fourier:', layout=w.Layout(width='52px')), fourier_w,
                rbtn,
            ]),
        ], layout=w.Layout(border='1px solid #e0e4f0', padding='4px 6px', margin='3px 0'))

        def _on_remove(_):
            self._filter_rows = [r for r in self._filter_rows if r[0] is not name_w]
            self._filters_vbox.children = [r[-1] for r in self._filter_rows]
            self._refresh_symbols()

        def _on_change(_):
            self._refresh_symbols()

        for ww in (name_w, latex_w):
            ww.observe(_on_change, names='value')
        rbtn.on_click(_on_remove)
        self._filter_rows.append((name_w, latex_w, time_w, fourier_w, row))
        self._filters_vbox.children = [r[-1] for r in self._filter_rows]
        self._refresh_symbols()

    # =========================================================================
    # Operator section
    # =========================================================================
    def _build_operator_section(self):
        self._operators_vbox = w.VBox([])
        add_btn = w.Button(description='+ operator', button_style='info',
                           layout=w.Layout(width='100px'))
        add_btn.on_click(self._add_operator)
        return w.VBox([
            w.HTML(
                f"<div style='{self._SECTION}'>Operators</div>"
                f"<div style='{self._HINT}'>"
                "Differential or other linear operators acting on fields "
                "(e.g. ∂<sub>t</sub>, ∇, ∇²).<br>"
                "In expressions: <code>del_t * v</code> (SymPy) or "
                "<code>\\partial_t v</code> (LaTeX) — operators multiply fields.<br>"
                "<b>Name</b> = Python symbol &nbsp;|&nbsp; "
                "<b>LaTeX</b> = e.g. <code>\\partial_t</code>, <code>\\nabla</code> "
                "&nbsp;|&nbsp; "
                "<b>Fourier</b> = frequency-space equivalent "
                "(e.g. <code>-I*omega</code>, <code>I*k</code>) &nbsp;|&nbsp; "
                "<b>Description</b> = optional"
                "</div>"
            ),
            self._operators_vbox, add_btn,
        ])

    def _add_operator(self, _=None, name='', latex_name='', fourier='', description=''):
        name_w    = w.Text(value=name,        placeholder='del_t',          layout=w.Layout(width='80px'))
        latex_w   = w.Text(value=latex_name,  placeholder=r'\partial_t',    layout=w.Layout(width='100px'))
        fourier_w = w.Text(value=fourier,     placeholder='-I*omega',       layout=w.Layout(width='160px'))
        desc_w    = w.Text(value=description, placeholder='time derivative', layout=w.Layout(width='200px'))
        rbtn      = w.Button(description='×', button_style='danger', layout=self._W_BTN)

        row = w.VBox([
            w.HBox([
                w.Label('name:',    layout=w.Layout(width='44px')), name_w,
                w.Label('LaTeX:',   layout=w.Layout(width='48px')), latex_w,
                w.Label('Fourier:', layout=w.Layout(width='52px')), fourier_w,
                w.Label('Desc:',    layout=w.Layout(width='38px')), desc_w,
                rbtn,
            ]),
        ], layout=w.Layout(border='1px solid #e0e4f0', padding='4px 6px', margin='3px 0'))

        def _on_remove(_):
            self._operator_rows = [r for r in self._operator_rows if r[0] is not name_w]
            self._operators_vbox.children = [r[-1] for r in self._operator_rows]
            self._refresh_symbols()

        def _on_change(_):
            self._refresh_symbols()

        for ww in (name_w, latex_w):
            ww.observe(_on_change, names='value')
        rbtn.on_click(_on_remove)
        self._operator_rows.append((name_w, latex_w, fourier_w, desc_w, row))
        self._operators_vbox.children = [r[-1] for r in self._operator_rows]
        self._refresh_symbols()

    # =========================================================================
    # Mean-field background section (always visible)
    # =========================================================================
    def _build_mf_section(self):
        self._bg_vbox     = w.VBox([])
        self._bg_render_w = w.Output(layout=w.Layout(min_height='28px', display='none'))
        self._mf_eqs_w = w.Textarea(
            placeholder=(
                'One mean-field equation per line (lhs = rhs).\n'
                'Example (LaTeX):  \\mu - v^*/\\tau = \\phi(v^*)\n'
                'Example (SymPy):  mu - v_star/tau = phi(v_star)'
            ),
            layout=w.Layout(width='500px', height='100px')
        )
        self._symbols_html = w.HTML('')
        return w.VBox([
            w.HTML(
                f"<div style='{self._SECTION}'>Mean-field background</div>"
                f"<div style='{self._HINT}'>"
                "Defines the mean-field background substitution used in all expression inputs.<br>"
                "In LaTeX mode: <code>v^*</code> → star symbol, "
                "<code>\\delta v</code> → fluctuation symbol."
                "</div>"
            ),
            self._bg_vbox,
            self._bg_render_w,
            self._symbols_html,
            w.HTML(f"<b style='font-size:12px;margin-top:8px'>Mean-field equations:</b>"),
            self._mf_eqs_w,
        ])

    def _rebuild_bg_table(self):
        fields = self._get_fields()
        old    = {r[0]: (r[1].value, r[2].value) for r in self._bg_rows if r[0] in fields}
        self._bg_rows = []
        rows = []
        il = (self._input_mode == 'latex')
        for f in fields:
            os, of = old.get(f, (f+'_star', 'd'+f))
            star_w  = w.Text(value=os, placeholder=(f+'^{*}' if il else f+'_star'), layout=w.Layout(width='110px'))
            fluct_w = w.Text(value=of, placeholder=(r'\delta '+f if il else 'd'+f), layout=w.Layout(width='110px'))
            star_w.observe( lambda _: (self._refresh_symbols(), self._update_bg_render()), names='value')
            fluct_w.observe(lambda _: (self._refresh_symbols(), self._update_bg_render()), names='value')
            row = w.HBox([
                w.HTML(f"<code style='min-width:70px;display:inline-block'>{f}</code>"),
                w.Label('→'), star_w, w.Label('+'), fluct_w,
            ], layout=w.Layout(margin='2px 0'))
            self._bg_rows.append((f, star_w, fluct_w, row))
            rows.append(row)
        self._bg_vbox.children = rows
        self._refresh_symbols()
        self._update_bg_render()

    def _update_bg_render(self):
        if not hasattr(self, '_bg_render_w'):
            return
        if self._input_mode != 'latex' or not self._bg_rows:
            self._render_math(self._bg_render_w, '')
            self._bg_render_w.layout.display = 'none'
            return
        parts = []
        for f, sw, fw_, _ in self._bg_rows:
            sn = sw.value.strip() or f + r'_{\star}'
            fn = fw_.value.strip() or r'\delta ' + f
            # Convert standard Python symbol names to LaTeX for display
            sn_tex = (f + '^{*}')       if sn == f + '_star' else sn.replace('_', r'\_')
            fn_tex = (r'\delta ' + f)   if fn == 'd' + f     else fn.replace('_', r'\_')
            parts.append(rf'{f} \to {sn_tex} + {fn_tex}')
        self._bg_render_w.layout.display = ''
        self._render_math(self._bg_render_w, r', \quad '.join(parts))

    # =========================================================================
    # Vertex section
    # =========================================================================
    def _build_vertex_section(self):
        self._vertex_mode_toggle = w.ToggleButtons(
            options=[('Manual', 'manual'), ('Auto-derive from action', 'auto')],
            value='manual', description='Vertex input:',
            style={'description_width': '100px'},
        )
        self._vertex_mode_toggle.observe(self._on_vertex_mode_change, names='value')

        # Manual panel
        self._manual_vbox = w.VBox([])
        add_vertex_btn = w.Button(description='+ vertex', button_style='info',
                                  layout=w.Layout(width='90px'))
        add_vertex_btn.on_click(self._add_vertex)
        self._manual_panel = w.VBox([
            w.HTML(
                f"<div style='{self._HINT}'>"
                "Each vertex = <b>Factor</b> × <b>Output fields</b> × <b>Input fields</b>.<br>"
                "<em>Factor</em>: coefficient (e.g. <code>\\phi''(v^*)/2</code>). &nbsp;"
                "<em>Output</em>: response field monomials (e.g. <code>\\tilde{{v}}</code>). &nbsp;"
                "<em>Input</em>: physical field monomials (e.g. <code>(\\delta v)^2</code>)."
                "</div>"
            ),
            self._manual_vbox,
            add_vertex_btn,
        ])

        # Auto panel
        self._full_action_w = w.Textarea(
            placeholder='Enter the full action S (all terms, including free part).',
            layout=w.Layout(width='640px', height='100px')
        )
        self._full_action_preview = w.Output(layout=w.Layout(min_height='36px', display='none'))
        self._full_action_w.observe(
            lambda c: self._update_preview(c, self._full_action_preview), names='value'
        )
        self._order_w = w.BoundedIntText(
            value=4, min=2, max=10, step=1,
            description='Max order:', layout=w.Layout(width='160px'),
            style={'description_width': '80px'},
        )
        expand_btn = w.Button(description='Expand → extract vertices',
                              button_style='warning', layout=w.Layout(width='220px'))
        expand_btn.on_click(self._expand_action)
        self._expand_out = w.Output()

        self._auto_panel = w.VBox([
            w.HTML(
                f"<div style='{self._HINT}'>"
                "Enter the <b>full</b> action S. The mean-field background above is used "
                "to perform the expansion. O(0)–O(2) terms are dropped; O(≥3) = S_int."
                "</div>"
            ),
            w.HTML("<b style='font-size:12px'>Full action S:</b>"),
            self._full_action_w, self._full_action_preview,
            w.HBox([self._order_w, expand_btn]),
            self._expand_out,
        ])
        self._auto_panel.layout.display = 'none'

        return w.VBox([
            w.HTML(f"<div style='{self._SECTION}'>Vertices / Interacting action</div>"),
            self._vertex_mode_toggle,
            self._manual_panel,
            self._auto_panel,
        ])

    # =========================================================================
    # Global toggle handlers
    # =========================================================================
    def _on_mode_change(self, change):
        self._input_mode = change['new']
        il = (self._input_mode == 'latex')
        self._cheatsheet_w.layout.display        = '' if il else 'none'
        self._full_action_preview.layout.display = '' if il else 'none'
        for (i,j), cell in self._prop_widgets.items():
            cell['expr'].placeholder       = self._prop_placeholder(i, j, cell['domain'].value)
            cell['preview'].layout.display = '' if il else 'none'
        for factor_w, output_w, input_w, _, pw, _ in self._vertex_rows:
            factor_w.placeholder = (r"\phi''(v^*)/2" if il else 'phi_pp/2')
            output_w.placeholder = (r'\tilde{v}'     if il else 'v_tilde')
            input_w.placeholder  = (r'(\delta v)^2' if il else 'dv**2')
            pw.layout.display    = '' if il else 'none'
        for _, _, _, expr_w, _ in self._func_rows:
            expr_w.placeholder = (r'\frac{x^2}{b}' if il else 'x**2/b')
        for f, star_w, fluct_w, _ in self._bg_rows:
            star_w.placeholder  = (f + '^{*}'       if il else f + '_star')
            fluct_w.placeholder = (r'\delta ' + f   if il else 'd' + f)
            if il:
                if star_w.value  == f + '_star': star_w.value  = f + '^{*}'
                if fluct_w.value == 'd' + f:     fluct_w.value = r'\delta ' + f
            else:
                if star_w.value  == f + '^{*}':         star_w.value  = f + '_star'
                if fluct_w.value == r'\delta ' + f:     fluct_w.value = 'd' + f
        self._update_prop_render()
        self._update_bg_render()

    def _on_stat_change(self, change):
        self._stationary = change['new']
        self._rebuild_prop_matrix()
        self._rebuild_bg_table()

    def _on_vertex_mode_change(self, change):
        self._vertex_mode = change['new']
        self._manual_panel.layout.display = '' if self._vertex_mode == 'manual' else 'none'
        self._auto_panel.layout.display   = '' if self._vertex_mode == 'auto'   else 'none'

    # =========================================================================
    # Field management
    # =========================================================================
    def _add_field(self, _=None, value=''):
        fw   = w.Text(value=value, placeholder=f'field_{len(self._field_rows)+1}',
                      layout=self._W_FIELD)
        resp = w.HTML('')
        rbtn = w.Button(description='×', button_style='danger', layout=self._W_BTN)
        row  = w.HBox([fw, w.Label('→'), resp, rbtn], layout=w.Layout(margin='2px 0'))

        def _on_name(_):
            n = fw.value.strip()
            resp.value = (f"<code style='color:#1a6e1a'>{n}_tilde</code>" if n else '')
            self._rebuild_prop_matrix(); self._rebuild_bg_table()

        def _on_remove(_):
            self._field_rows = [(f,r) for f,r in self._field_rows if f is not fw]
            self._fields_vbox.children = [r for _,r in self._field_rows]
            self._rebuild_prop_matrix(); self._rebuild_bg_table()

        fw.observe(_on_name, names='value')
        rbtn.on_click(_on_remove)
        self._field_rows.append((fw, row))
        self._fields_vbox.children = [r for _,r in self._field_rows]

    # =========================================================================
    # Parameter management
    # =========================================================================
    def _add_param(self, _=None, symbol='', latex_alias='', description=''):
        sw   = w.Text(value=symbol,      placeholder='sym',           layout=self._W_SYM)
        aw   = w.Text(value=latex_alias, placeholder=r'e.g. \lambda', layout=self._W_ALIAS)
        dw   = w.Text(value=description, placeholder='description',   layout=self._W_DESC)
        rbtn = w.Button(description='×', button_style='danger', layout=self._W_BTN)
        row  = w.HBox([sw, aw, dw, rbtn], layout=w.Layout(margin='2px 0'))

        def _on_remove(_):
            self._param_rows = [(s,a,d,r) for s,a,d,r in self._param_rows if s is not sw]
            self._params_vbox.children = [r for _,_,_,r in self._param_rows]
            self._refresh_symbols()

        sw.observe(lambda _: self._refresh_symbols(), names='value')
        rbtn.on_click(_on_remove)
        self._param_rows.append((sw, aw, dw, row))
        self._params_vbox.children = [r for _,_,_,r in self._param_rows]

    # =========================================================================
    # Vertex management (manual) — factor / output / input
    # =========================================================================
    def _add_vertex(self, _=None, factor='', output='', input_='', description=''):
        is_latex = (self._input_mode == 'latex')
        idx      = len(self._vertex_rows) + 1

        lbl      = w.HTML(
            f"<div style='font-family:monospace;color:#555;padding-right:6px;min-width:28px'>"
            f"V{idx}</div>"
        )
        factor_w = w.Text(value=factor,
                          placeholder=(r"\phi''(v^*)/2" if is_latex else 'phi_pp/2'),
                          layout=w.Layout(width='260px'))
        output_w = w.Text(value=output,
                          placeholder=(r'\tilde{v}'     if is_latex else 'v_tilde'),
                          layout=w.Layout(width='210px'))
        input_w  = w.Text(value=input_,
                          placeholder=(r'(\delta v)^2' if is_latex else 'dv**2'),
                          layout=w.Layout(width='210px'))
        desc_w   = w.Text(value=description, placeholder='description (optional)',
                          layout=w.Layout(width='160px'))
        preview_w = w.Output(layout=w.Layout(min_height='28px', display=('' if is_latex else 'none')))
        rbtn = w.Button(description='×', button_style='danger', layout=self._W_BTN)

        box = w.VBox([
            w.HBox([
                lbl,
                w.Label('Factor:',  layout=w.Layout(width='50px')), factor_w,
                w.Label('Output:', layout=w.Layout(width='52px')), output_w,
                w.Label('Input:',  layout=w.Layout(width='46px')), input_w,
                desc_w, rbtn,
            ]),
            preview_w,
        ], layout=w.Layout(border='1px solid #e0e0e0', padding='4px 6px', margin='3px 0'))

        def _on_change(change=None, _pw=preview_w):
            if self._input_mode == 'latex':
                fac = factor_w.value.strip()
                out = output_w.value.strip()
                inp = input_w.value.strip()
                parts = [p for p in [fac, out, inp] if p]
                self._render_math(_pw, r'\cdot'.join(f'\\left({p}\\right)' for p in parts) if parts else '')

        def _on_remove(_):
            self._vertex_rows = [
                t for t in self._vertex_rows if t[0] is not factor_w
            ]
            self._rebuild_vertex_vbox()

        for ww in (factor_w, output_w, input_w):
            ww.observe(_on_change, names='value')
        rbtn.on_click(_on_remove)
        self._vertex_rows.append((factor_w, output_w, input_w, desc_w, preview_w, box))
        self._rebuild_vertex_vbox()

    def _rebuild_vertex_vbox(self):
        for i, row in enumerate(self._vertex_rows):
            box = row[-1]
            box.children[0].children[0].value = (
                f"<div style='font-family:monospace;color:#555;"
                f"padding-right:6px;min-width:28px'>V{i+1}</div>"
            )
        self._manual_vbox.children = [row[-1] for row in self._vertex_rows]

    # =========================================================================
    # Auto-derive: expansion
    # =========================================================================
    def _expand_action(self, _=None):
        with self._expand_out:
            clear_output(wait=True)
            fields   = self._get_fields()
            is_latex = (self._input_mode == 'latex')

            action_str = self._full_action_w.value.strip()
            if not action_str: print('⚠  No action entered.'); return
            ok, action_sym = self._parse_expr(action_str, is_latex=is_latex)
            if not ok: print(f'✗  Action parse error: {action_sym}'); return

            epsilon    = sp.Symbol('_eps_')
            subs_eps   = {}
            fluct_syms = []
            for f, star_w, fluct_w, _ in self._bg_rows:
                sn = star_w.value.strip()  or (f+'_star')
                fn = fluct_w.value.strip() or ('d'+f)
                subs_eps[sp.Symbol(f)]          = sp.Symbol(sn) + epsilon * sp.Symbol(fn)
                subs_eps[sp.Symbol(f+'_tilde')] = epsilon * sp.Symbol(f+'_tilde')
                fluct_syms += [sp.Symbol(fn), sp.Symbol(f+'_tilde')]

            if not subs_eps: print('⚠  Add fields first.'); return

            order = self._order_w.value
            try:
                shifted = action_sym.subs(subs_eps)
                series  = sp.series(shifted, epsilon, 0, order+1)
            except Exception as e:
                print(f'✗  Series expansion failed: {e}'); return

            mf_subs = {}
            for line in self._mf_eqs_w.value.strip().split('\n'):
                line = line.strip()
                if '=' not in line: continue
                ls, rs = line.split('=', 1)
                ol, le = self._parse_expr(ls.strip())
                or_, re_ = self._parse_expr(rs.strip())
                if ol and or_: mf_subs[le] = re_

            terms = {}
            for k in range(order+1):
                t = series.coeff(epsilon, k)
                if mf_subs: t = t.subs(mf_subs)
                terms[k] = sp.expand(t)

            labels = {0:'(MF free energy)', 1:'(linear — should vanish at saddle point)',
                      2:'(bilinear — defines propagator)'}
            print('Expansion by ε-order:\n')
            for k, t in terms.items():
                label = labels.get(k, '← VERTEX')
                print(f'  O(ε^{k}): {"0" if t==0 else "non-zero"}  {label}')
                if t != 0 and k >= 3: print(f'           {t}\n')

            S_int = sp.expand(sum(terms.get(k,0) for k in range(3, order+1)))
            print(f'\nS_int = {S_int}\n')

            print('Extracted vertex monomials:')
            try:
                poly = sp.Poly(S_int, *fluct_syms)
                vertex_exprs = []
                for monom, coeff in zip(poly.monoms(), poly.coeffs()):
                    if sum(monom) < 3: continue
                    field_part = sp.Mul(*[s**p for s,p in zip(fluct_syms,monom)])
                    vertex_exprs.append((str(coeff), str(field_part)))
                    print(f'  coeff={coeff}  fields={field_part}')
                if vertex_exprs:
                    self._vertex_rows = []
                    for coeff_s, field_s in vertex_exprs:
                        self._add_vertex(factor=coeff_s, output='', input_=field_s,
                                         description='auto-extracted')
                    self._vertex_mode_toggle.value = 'manual'
                    print('\n→ Populated vertex list (review Output/Input split manually).')
            except Exception:
                self._vertex_rows = []
                self._add_vertex(factor=str(S_int), description='auto-derived S_int')
                self._vertex_mode_toggle.value = 'manual'

    # =========================================================================
    # Helpers
    # =========================================================================
    def _get_fields(self):
        return [fw.value.strip() for fw,_ in self._field_rows if fw.value.strip()]

    def _get_params(self):
        return [{'symbol': sw.value.strip(), 'latex_alias': aw.value.strip(),
                 'description': dw.value.strip()}
                for sw,aw,dw,_ in self._param_rows if sw.value.strip()]

    def _get_latex_aliases(self):
        aliases = {aw.value.strip(): sw.value.strip()
                   for sw,aw,dw,_ in self._param_rows
                   if sw.value.strip() and aw.value.strip()}
        # Filter latex names → Python names  (e.g. \kappa → kappa)
        for nw, lw, _, _, _ in self._filter_rows:
            if nw.value.strip() and lw.value.strip():
                aliases[lw.value.strip()] = nw.value.strip()
        # Operator latex names → Python names  (e.g. \partial_t → del_t)
        for nw, lw, _, _, _ in self._operator_rows:
            if nw.value.strip() and lw.value.strip():
                aliases[lw.value.strip()] = nw.value.strip()
        return aliases

    def _get_field_star_map(self):
        m = {f: f+'_star' for f in self._get_fields()}
        for f, sw, fw_, _ in self._bg_rows:
            if sw.value.strip(): m[f] = sw.value.strip()
        return m

    def _get_field_fluct_map(self):
        m = {f: 'd'+f for f in self._get_fields()}
        for f, sw, fw_, _ in self._bg_rows:
            if fw_.value.strip(): m[f] = fw_.value.strip()
        return m

    def _get_func_defs(self):
        result = []
        for name_w, latex_w, args_w, expr_w, _ in self._func_rows:
            name  = name_w.value.strip()
            lname = latex_w.value.strip()
            args  = [a.strip() for a in args_w.value.split(',') if a.strip()] or ['x']
            expr  = expr_w.value.strip()
            if not name or not lname or not expr: continue
            syms  = {s: sp.Symbol(s) for s in self._get_all_symbols() + args}
            syms['I'] = sp.I
            try:
                expr_sym = sp.sympify(expr, locals=syms)
            except Exception:
                try:
                    expr_sym = latex_to_sympy(expr, self._get_fields(),
                                              self._get_latex_aliases())
                    for a in args:
                        expr_sym = expr_sym.subs(sp.Symbol(a), sp.Symbol(a))
                except Exception:
                    continue
            result.append({'name': name, 'latex_name': lname,
                           'args': args, 'expr': expr, 'expr_sym': expr_sym})
        return result

    def _get_all_symbols(self):
        fields    = self._get_fields()
        resp      = [f+'_tilde' for f in fields]
        params    = [p['symbol'] for p in self._get_params()]
        bg_syms   = []
        for _, sw, fw_, _ in self._bg_rows:
            if sw.value.strip():  bg_syms.append(sw.value.strip())
            if fw_.value.strip(): bg_syms.append(fw_.value.strip())
        func_args    = [a for _,_,aw,_,_ in self._func_rows
                        for a in aw.value.split(',') if a.strip()]
        filter_names   = [nw.value.strip() for nw,_,_,_,_ in self._filter_rows
                          if nw.value.strip()]
        operator_names = [nw.value.strip() for nw,_,_,_,_ in self._operator_rows
                          if nw.value.strip()]
        extras         = ['omega','t'] + ([] if self._stationary else ['t1'])
        return (fields + resp + params + bg_syms + func_args
                + filter_names + operator_names + extras)

    def _refresh_symbols(self):
        syms = self._get_all_symbols()
        self._symbols_html.value = (
            f"<div style='{self._HINT}'>Available symbols: "
            + ', '.join(f'<code>{s}</code>' for s in syms) + '</div>'
        ) if syms else ''

    def _latex_preview_html(self, s):
        return r'\[' + s.strip() + r'\]' if s.strip() else ''

    def _render_math(self, output_w, latex_str):
        """Render LaTeX in an Output widget.
        Uses only text/latex (no text/plain fallback) so VS Code renders exactly one copy."""
        s = latex_str.strip() if latex_str else ''
        if s:
            output_w.outputs = ({'output_type': 'display_data',
                                  'data': {'text/latex': f'$${s}$$'},
                                  'metadata': {}},)
        else:
            output_w.outputs = ()

    def _update_preview(self, change, pw):
        if self._input_mode == 'latex':
            self._render_math(pw, change['new'])

    def _parse_expr(self, expr_str, is_latex=False):
        if not expr_str or expr_str.strip() == '0':
            return True, sp.Integer(0)
        if is_latex and _LATEX_OK:
            try:
                return True, latex_to_sympy(
                    expr_str, self._get_fields(),
                    param_aliases=self._get_latex_aliases(),
                    func_defs=self._get_func_defs(),
                    field_star_map=self._get_field_star_map(),
                    field_fluct_map=self._get_field_fluct_map(),
                )
            except Exception as e:
                return False, str(e)
        syms = {s: sp.Symbol(s) for s in self._get_all_symbols()}
        syms['I'] = sp.I; syms['Heaviside'] = sp.Heaviside; syms['DiracDelta'] = sp.DiracDelta
        try:
            return True, sp.sympify(expr_str, locals=syms)
        except Exception as e:
            return False, str(e)

    # =========================================================================
    # Propagator matrix
    # =========================================================================
    def _prop_placeholder(self, i, j, domain):
        il = (self._input_mode == 'latex'); d = (i == j)
        if domain == 'frequency':
            return (r'\frac{1}{-i\omega\tau - \lambda\tau}' if d else '0') if il \
                else ('1/(-I*omega*tau - lam*tau)' if d else '0')
        if self._stationary:
            return (r'\theta(t)\,e^{\lambda t}' if d else '0') if il \
                else ('Heaviside(t)*exp(lam*t)' if d else '0')
        return ("\\theta(t-t')\\,e^{\\lambda(t-t')}" if (il and d) else '0' if (il and not d)
                else ('Heaviside(t-t1)*exp(lam*(t-t1))' if d else '0'))

    def _rebuild_prop_matrix(self):
        fields = self._get_fields(); n = len(fields)
        if n == 0:
            self._prop_outer.children = [
                w.HTML(f"<div style='{self._SECTION}'>Propagator matrix</div>"
                       f"<div style='{self._HINT}'><i>Add fields to populate.</i></div>")
            ]; self._prop_widgets = {}; return

        resp     = [f+'_tilde' for f in fields]
        old_expr = {k: v['expr'].value   for k,v in self._prop_widgets.items()}
        old_dom  = {k: v['domain'].value for k,v in self._prop_widgets.items()}
        self._prop_widgets = {}

        col_hdrs = [w.HTML("<div style='width:160px'></div>")] + [
            w.HTML(f"<div style='width:305px;text-align:center;"
                   f"font-weight:bold;font-family:monospace'>{f}</div>") for f in fields
        ]
        matrix_rows = [w.HBox(col_hdrs)]

        for i, rf in enumerate(resp):
            row_lbl = w.HTML(
                f"<div style='width:160px;text-align:right;padding-right:8px;"
                f"font-family:monospace;font-weight:bold'>{rf}</div>"
            )
            cells = [row_lbl]
            for j in range(n):
                is_diag  = (i == j)
                dom_init = old_dom.get((i,j), 'time')
                _tlabel = "Time  G(t,t')" if not self._stationary else 'Time  G(t)'
                domain_dd = w.Dropdown(
                    options=[('Frequency  G(ω)', 'frequency'), (_tlabel, 'time')],
                    value=dom_init, layout=w.Layout(width='165px'),
                )
                expr_w = w.Text(
                    value=old_expr.get((i,j),''),
                    placeholder=self._prop_placeholder(i, j, dom_init),
                    layout=self._W_PROP,
                )
                preview_w = w.Output(
                    layout=w.Layout(min_height='28px',
                                    display=('' if self._input_mode=='latex' else 'none'))
                )
                self._prop_widgets[(i,j)] = {'expr': expr_w, 'domain': domain_dd, 'preview': preview_w}

                def _on_dom(change, _i=i, _j=j):
                    self._prop_widgets[(_i,_j)]['expr'].placeholder = \
                        self._prop_placeholder(_i, _j, change['new'])
                    self._update_prop_render()
                def _on_expr(change, _pw=preview_w):
                    if self._input_mode == 'latex':
                        self._render_math(_pw, change['new'])
                    self._update_prop_render()

                domain_dd.observe(_on_dom,  names='value')
                expr_w.observe(_on_expr, names='value')
                cells.append(w.VBox(
                    [domain_dd, expr_w, preview_w],
                    layout=w.Layout(border='1px solid #ccc', padding='6px', margin='2px',
                                    background='#f5f8ff' if is_diag else '#fff')
                ))
            matrix_rows.append(w.HBox(cells))

        if not hasattr(self, '_prop_render_w'):
            self._prop_render_w = w.Output(layout=w.Layout(min_height='36px', display='none'))
        self._prop_outer.children = [
            w.HTML(
                f"<div style='{self._SECTION}'>Propagator matrix</div>"
                f"<div style='{self._HINT}'>"
                "G[i][j] = retarded propagator <code>j_tilde → i</code>. "
                "Rows = physical fields, cols = response fields. "
                f"Theory is {'<b>stationary</b>' if self._stationary else '<b>non-stationary</b>'}."
                "</div>"
            ),
            w.VBox(matrix_rows),
            self._prop_render_w,
            w.HTML(f"<div style='{self._HINT};margin-top:4px'>"
                   "Highlighted cells = diagonal entries.</div>"),
        ]
        self._update_prop_render()

    def _update_prop_render(self):
        if not hasattr(self, '_prop_render_w') or not self._prop_widgets:
            return
        if self._input_mode != 'latex':
            self._prop_render_w.outputs = ()
            self._prop_render_w.layout.display = 'none'
            return
        fields = self._get_fields()
        n = len(fields)
        if n == 0:
            self._render_math(self._prop_render_w, '')
            self._prop_render_w.layout.display = 'none'
            return
        # Build labeled array: first col = row label (response field), then n expr cols
        col_spec   = r'r|' + 'c' * n
        col_header = r' & ' + ' & '.join(
            r'\scriptstyle\mathrm{' + f.replace('_', r'\_') + '}' for f in fields
        )
        data_rows  = []
        resp = [f + '_tilde' for f in fields]
        for i in range(n):
            rl = r'\scriptstyle\mathrm{' + resp[i].replace('_', r'\_') + '}'
            cells = []
            for j in range(n):
                cell = self._prop_widgets.get((i, j))
                val  = (cell['expr'].value.strip() if cell else '') or r'{\cdot}'
                cells.append(val)
            data_rows.append(rl + ' & ' + ' & '.join(cells))
        body = r' \\ '.join(data_rows)
        latex_body = (
            r'\mathbf{G} = \begin{array}{' + col_spec + r'}'
            + col_header + r' \\ \hline ' + body + r'\end{array}'
        )
        self._prop_render_w.layout.display = ''
        self._render_math(self._prop_render_w, latex_body)

    # =========================================================================
    # Save
    # =========================================================================
    def _save(self, _=None):
        with self._save_out:
            clear_output(wait=True)
            is_latex = (self._input_mode == 'latex')

            t = Theory()
            t.name        = self._name_w.value.strip()
            t.description = self._desc_w.value.strip()
            t.stationary  = self._stationary
            t.fields      = self._get_fields()
            t.parameters  = self._get_params()
            t.vertex_mode = self._vertex_mode
            t.input_mode  = self._input_mode

            if not t.name:   print('⚠  Enter a theory name.'); return
            if not t.fields: print('⚠  Add at least one field.'); return

            # Operator definitions
            for name_w, latex_w, fourier_w, desc_w, _ in self._operator_rows:
                name = name_w.value.strip()
                lname = latex_w.value.strip()
                if name and lname:
                    t.operator_defs.append({
                        'name':        name,
                        'latex_name':  lname,
                        'fourier':     fourier_w.value.strip(),
                        'description': desc_w.value.strip(),
                    })

            # Filter definitions
            for name_w, latex_w, time_w, fourier_w, _ in self._filter_rows:
                name = name_w.value.strip()
                lname = latex_w.value.strip()
                if name and lname:
                    t.filter_defs.append({
                        'name':         name,
                        'latex_name':   lname,
                        'expr_time':    time_w.value.strip(),
                        'expr_fourier': fourier_w.value.strip(),
                    })

            # Function definitions
            for name_w, latex_w, args_w, expr_w, _ in self._func_rows:
                name  = name_w.value.strip()
                lname = latex_w.value.strip()
                args  = [a.strip() for a in args_w.value.split(',') if a.strip()] or ['x']
                expr  = expr_w.value.strip()
                if name and lname and expr:
                    t.func_defs.append({'name': name, 'latex_name': lname,
                                        'args': args, 'expr': expr})

            # MF background
            t.mf_background = [
                {'field': f,
                 'star':  sw.value.strip() or f+'_star',
                 'fluct': fw_.value.strip() or 'd'+f}
                for f, sw, fw_, _ in self._bg_rows
            ]
            t.mf_equations = self._mf_eqs_w.value.strip()

            # Propagators
            for (i,j), cell in self._prop_widgets.items():
                raw = cell['expr'].value.strip() or '0'
                if is_latex and raw != '0' and _LATEX_OK:
                    ok, expr = self._parse_expr(raw, is_latex=True)
                    t.prop_exprs[(i,j)] = str(expr) if ok else raw
                else:
                    t.prop_exprs[(i,j)] = raw
                t.prop_domain[(i,j)] = cell['domain'].value

            # Vertices: factor * output * input → expr
            for factor_w, output_w, input_w, desc_w, _, _ in self._vertex_rows:
                fac_raw = factor_w.value.strip()
                out_raw = output_w.value.strip()
                inp_raw = input_w.value.strip()
                if not any([fac_raw, out_raw, inp_raw]): continue

                # Parse each part
                def _p(s):
                    if not s: return sp.Integer(1)
                    ok, e = self._parse_expr(s, is_latex=is_latex)
                    return e if ok else sp.Symbol('_parse_err')

                expr_sym = sp.expand(_p(fac_raw) * _p(out_raw) * _p(inp_raw))

                # Store raw (SymPy string) values
                def _store(s):
                    if not s: return ''
                    if is_latex and _LATEX_OK:
                        ok, e = self._parse_expr(s, is_latex=True)
                        return str(e) if ok else s
                    return s

                t.vertices.append({
                    'factor':      _store(fac_raw),
                    'output':      _store(out_raw),
                    'input':       _store(inp_raw),
                    'expr':        str(expr_sym),
                    'description': desc_w.value.strip(),
                })

            # Auto-mode metadata
            t.full_action  = self._full_action_w.value.strip()
            t.expand_order = self._order_w.value

            saved = t.save(self._path_w.value.strip())
            self.theory = t
            print(f'✓  Saved to  {saved}\n')
            print(t.summary())

    def show(self):
        display(self._ui)


if __name__ == '__main__':
    print('theory_builder.py loaded successfully.')
    print(f'  LaTeX input mode: {"enabled" if _LATEX_OK else "disabled (install antlr4-python3-runtime==4.11.1)"}')
    print(f'  Classes available: Theory, TheoryUI')
