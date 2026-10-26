"""Declarations of the SDE front end return Sage symbols (api/sde_symbols.py), and the renamed keywords
(``noise(into=, as_field=)``, formerly ``couples=`` / ``expose=``).

A model written with symbols and Sage expressions is compared with the string model whose text is the expression's
printed form: the action text, mean-field equations, notes and built model dict must be identical (``digest``).  The
old keywords must give byte-identical models (with a DeprecationWarning).
"""
import os
import warnings

import pytest
from sage.all import SR
from sage.symbolic.expression import Expression

from api.noise import Gaussian, Poisson
from api.operators import Laplacian, TimeDerivative
from api.sde import SDE, SDEError
from api.sde_symbols import Conv, Symbol
from tests import _sde_model_catalogue as cat
from tests._sde_model_catalogue import digest


def _ouq_strings(rhs='-eps*x^3 - mu*x', var='2*D'):
    return (SDE('OU quartic').parameter('mu', default=1.0, domain='positive')
            .parameter('eps', default=0.02, domain='positive').parameter('D', default=1.0, domain='positive')
            .field('x').drift('x', rhs).noise('xi', Gaussian(var=var), into={'x': 1}))


def test_symbols_give_the_string_model():
    m = SDE('OU quartic')
    mu, eps, D = (m.parameter(n, default=v, domain='positive') for n, v in (('mu', 1.0), ('eps', 0.02), ('D', 1.0)))
    x = m.field('x')
    assert m.drift(x, -mu*x - eps*x**3).noise('xi', Gaussian(var=2*D), into={x: 1}) is m
    assert digest(m) == digest(_ouq_strings())


def test_a_symbol_is_a_sage_symbol():
    m = SDE('s')
    x = m.field('x')
    assert isinstance(x, Symbol) and isinstance(x, Expression)
    assert (x - SR.var('x')).is_zero() and str(x) == 'x' and SR(x) == SR.var('x')
    assert type(x + 1) is Expression and str((x + 1)**2) == '(x + 1)^2'
    assert {x: 1}[SR.var('x')] == 1
    assert x.operator() is None                       # Sage's Expression.operator() without arguments
    i, j = SR.var('i j')
    w = m.parameter('w', default=[[1.0]], indexed='matrix')
    from api.noise import expr_to_text
    assert expr_to_text(w[i, j] * x[j]) == 'w[i,j]*x[j]'
    with pytest.raises(TypeError, match='not callable'):
        w(x)


def test_declarations_still_chain():
    """Every declaration returns a symbol that forwards the builder's methods: the chained style is unchanged."""
    s = (SDE('OU quartic').parameter('mu', default=1.0, domain='positive')
         .parameter('eps', default=0.02, domain='positive').parameter('D', default=1.0, domain='positive')
         .field('x'))
    assert isinstance(s, Symbol) and s.name == 'OU quartic' and s._sde._fields == {'x': s._fields['x']}
    out = s.drift('x', '-eps*x^3 - mu*x').noise('xi', Gaussian(var='2*D'), into={'x': 1})
    assert isinstance(out, SDE)
    assert digest(out) == digest(_ouq_strings())
    # the colliding names: operator(name, op) and show() are the builder's
    op = SDE('o').field('u').operator('D_t', TimeDerivative())
    assert isinstance(op, Symbol) and 'D_t' in op._operators


def test_operators_functions_kernels_and_exposed_sources_as_symbols():
    m = SDE('ops')
    D_t = m.operator('D_t', TimeDerivative())
    tau, E = m.parameter('tau', default=2.0), m.parameter('E', default=0.5)
    v = m.field('v')
    m.equation(v, (tau*D_t + 1)*v, E).noise('n', Gaussian(var=1), into={v: 1})
    s = (SDE('ops').operator('D_t', TimeDerivative()).parameter('tau', default=2.0).parameter('E', default=0.5)
         .field('v').equation('v', '(D_t*tau + 1)*v', 'E').noise('n', Gaussian(var=1), into={'v': 1}))
    assert digest(m) == digest(s)
    m2 = SDE('call')
    D_t = m2.operator('D_t', TimeDerivative())
    x = m2.field('x')
    m2.equation(x, D_t(x), -x)
    assert m2._eqs['x']['lhs_raw'] == 'D_t(x)'                       # a call, not Sage's substitution
    # a function, a kernel and an exposed source
    h = SDE('hawkes')
    a, w, taug = h.parameter('a', default=1.0), h.parameter('w', default=0.3), h.parameter('taug', default=1.0)
    phi = h.define_function('phi', ['v'], a*SR.var('v'))
    g = h.define_kernel('g', time_expr=SR('exp(-t/taug)/taug*heaviside(t)'))
    v = h.field('v')
    n = h.noise('spikes', Poisson(rate=phi(v)), as_field='n')
    assert isinstance(n, Symbol) and n._kind == 'field'
    h.drift(v, -v + w*Conv(g, n))
    hs = (SDE('hawkes').parameter('a', default=1.0).parameter('w', default=0.3).parameter('taug', default=1.0)
          .define_function('phi', ['v'], 'a*v').define_kernel('g', time_expr='exp(-t/taug)*heaviside(t)/taug')
          .field('v').noise('spikes', Poisson(rate='phi(v)'), as_field='n').drift('v', 'w*Conv(g, n) - v'))
    assert h.action_text() == hs.action_text() and h.equations() == hs.equations()


def test_point_process_and_spde_with_symbols():
    m = SDE('pp')
    lam, J = m.parameter('lam', default=0.5), m.parameter('J', default=0.2)
    v = m.field('v')
    n = m.point_process('n', lam, size=J)
    m.drift(v, -v + Conv(m.define_kernel('r', time_expr='exp(-t)*heaviside(t)'), n))
    s = (SDE('pp').parameter('lam', default=0.5).parameter('J', default=0.2).field('v')
         .point_process('n', 'lam', size='J').define_kernel('r', time_expr='exp(-t)*heaviside(t)')
         .drift('v', '-v + Conv(r, n)'))
    assert digest(m) == digest(s)
    a = SDE('ac', space=1)
    D_t, lap = a.operator('D_t', TimeDerivative()), a.operator('lap', Laplacian())
    D, mu, T = (a.parameter(p, default=1.0, domain='positive') for p in ('D', 'mu', 'T'))
    phi = a.field('phi')
    a.equation(phi, D_t(phi), D*lap(phi) - mu*phi).noise('eta', Gaussian(var=2*T, space='white'), into={phi: 1})
    b = (SDE('ac', space=1).operator('D_t', TimeDerivative()).operator('lap', Laplacian())
         .parameter('D', default=1.0, domain='positive').parameter('mu', default=1.0, domain='positive')
         .parameter('T', default=1.0, domain='positive').field('phi')
         .equation('phi', 'D_t(phi)', '-mu*phi + D*lap(phi)').noise('eta', Gaussian(var='2*T', space='white'),
                                                                into={'phi': 1}))
    assert digest(a) == digest(b)


# ── renamed keywords ─────────────────────────────────────────────────────────

def test_old_keywords_warn_and_give_the_same_model():
    with pytest.warns(DeprecationWarning, match=r'couples=.*use into='):
        old = (SDE('OU quartic').parameter('mu', default=1.0, domain='positive')
               .parameter('eps', default=0.02, domain='positive').parameter('D', default=1.0, domain='positive')
               .field('x').drift('x', '-eps*x^3 - mu*x').noise('xi', Gaussian(var='2*D'), couples={'x': 1}))
    assert digest(old) == digest(_ouq_strings())
    with pytest.warns(DeprecationWarning, match=r'expose=.*use as_field='):
        e_old = SDE('h').parameter('a', default=1.0).noise('s', Poisson(rate='a'), expose='n')
    e_new = SDE('h').parameter('a', default=1.0).noise('s', Poisson(rate='a'), as_field='n')
    assert repr(e_old._sources) == repr(e_new._sources) and e_old._fields == e_new._fields
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        _ouq_strings()                                          # the new names never warn
    with pytest.raises(SDEError, match='not both'):
        SDE('b').field('x').noise('n', Gaussian(var=1), couples={'x': 1}, into={'x': 1})
    with pytest.raises(SDEError, match='not both'):
        SDE('b').noise('n', Poisson(rate=1), expose='n', as_field='n')


def test_every_catalogue_model_is_byte_identical_under_the_new_keywords(monkeypatch):
    """The catalogue is written with couples= / expose=; rewritten to into= / as_field= it builds the same."""
    names = [n for n in dir(cat) if not n.startswith('_') and callable(getattr(cat, n))
             and n not in ('digest', 'SDE', 'Kernel') and not isinstance(getattr(cat, n), type)]
    old = {}
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', DeprecationWarning)
        for n in names:
            try:
                m = getattr(cat, n)()
            except TypeError:
                continue
            if isinstance(m, SDE):
                old[n] = digest(m)
    orig = SDE.noise

    def new_noise(self, name, source, couples=None, *, expose=None, **kw):
        if couples is not None:
            kw['into'] = couples
        if expose is not None:
            kw['as_field'] = expose
        return orig(self, name, source, **kw)

    monkeypatch.setattr(SDE, 'noise', new_noise)
    with warnings.catch_warnings():
        warnings.simplefilter('error', DeprecationWarning)
        for n, d in old.items():
            assert digest(getattr(cat, n)()) == d, n
    assert len(old) >= 15


def _cfg(tmp_path, noise_line):
    from api.config_run import load_config
    text = ('sde:\n  name: OU\n  parameters:\n    mu: {default: 1.0, domain: positive}\n'
            '    D: {default: 1.0, domain: positive}\n  fields: [x]\n  drift: {x: "-mu*x"}\n'
            f'  noise:\n    xi: {noise_line}\nquestion: {{k: 2, max_ell: 0}}\n')
    p = os.path.join(tmp_path, 'c.yaml')
    open(p, 'w').write(text)
    return load_config(p)


def test_config_noise_takes_into_and_a_per_source_interpretation(tmp_path):
    a, b = _cfg(tmp_path, '{type: Gaussian, var: "2*D", into: {x: 1}}'), \
        _cfg(tmp_path, '{type: Gaussian, var: "2*D", couples: {x: 1}}')
    assert a[0]['sde']['noise']['xi']['into'] == {'x': 1}
    assert cat._canon(a[1]) == cat._canon(b[1])
    s = _cfg(tmp_path, '{type: Gaussian, var: "D*x^2", into: {x: 1}, interpretation: stratonovich}')
    assert s[0]['sde']['noise']['xi']['interpretation'] == 'stratonovich'
    py = (SDE('OU').parameter('mu', default=1.0, domain='positive').parameter('D', default=1.0, domain='positive')
          .field('x').drift('x', '-mu*x')
          .noise('xi', Gaussian(var='D*x^2'), into={'x': 1}, interpretation='stratonovich'))
    assert cat._canon(s[1]) == cat._canon(py.build())
    with pytest.raises(Exception, match='interpretation must be ito, stratonovich or marcus'):
        _cfg(tmp_path, '{type: Gaussian, var: "D", into: {x: 1}, interpretation: strat}')
    with pytest.raises(Exception, match="give 'into' or 'couples'"):
        _cfg(tmp_path, '{type: Gaussian, var: "D", into: {x: 1}, couples: {x: 1}}')


def test_a_delayed_field_symbol_is_the_delay_kernel():
    m = SDE('delay')
    d, a = m.parameter('d', default=1.0, domain='positive'), m.parameter('a', default=0.5)
    x = m.field('x')
    t = SR.var('t')
    m.drift(x, -x - a*x(t - d)).noise('xi', Gaussian(var=1), into={x: 1})
    s = (SDE('delay').parameter('d', default=1.0, domain='positive').parameter('a', default=0.5).field('x')
         .drift('x', '-a*x(t - d) - x').noise('xi', Gaussian(var=1), into={'x': 1}))
    assert digest(m) == digest(s) and 'delay_d' in m._kernels
