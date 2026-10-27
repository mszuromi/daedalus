"""Paper-style equations for the SDE front end (``SDE.sde``, daedalus/api/sde_paper.py) and per-source interpretations
(``noise(..., interpretation=)``).

A paper-style line is read into the existing declarations, so each model here is compared with the explicit model
those calls make: the action text, the mean-field equations, the notes and the built model dict must be identical
(``digest`` of the SDE catalogue).  Mixed interpretations are also compared with the Ito model that carries the
Stratonovich drift correction or the Marcus noise term written out by hand.
"""
import pytest
from sage.all import SR

from daedalus.api.noise import CompoundPoisson, Exponential, Gaussian, GammaProcess, Poisson, parse_expr
from daedalus.api.sde import SDE, SDEError
from tests._sde_model_catalogue import digest
from tests import _sde_model_catalogue as cat


def _p(m, **extra):
    for n, d in dict(mu=1.0, eps=0.02, D=1.0, **extra).items():
        m.parameter(n, default=d)
    return m


def _same(a, b):
    assert digest(a) == digest(b)


def _action(m):
    return parse_expr(m._index(m.action_text().replace('\n', ' ')
                               .replace('sum(', '(').replace('for i in pop)', ')')))


# ── the paper-style line is the explicit model ──────────────────────────────

@pytest.mark.parametrize('text', [
    'dx = (-mu*x - eps*x^3) dt + sqrt(2*D) dW',
    'dx = -mu*x*dt - eps*x^3*dt + sqrt(2*D)*dW',
    '  dx = (-mu*x - eps*x^3)*dt\n       + sqrt(2*D) dW    # a comment',
    'dx = (-mu*x - eps*x^3) dt + sqrt(2*D) dW_{W_x}',
])
def test_ou_quartic_paper_line_is_the_explicit_model(text):
    a = _p(SDE('ouq')).sde(text)
    b = (_p(SDE('ouq')).field('x').drift('x', '-mu*x - eps*x^3')
         .noise('W_x', Gaussian(var=1), into={'x': 'sqrt(2*D)'}, interpretation='ito'))
    _same(a, b)
    c = _p(SDE('ouq')).field('x').drift('x', '-mu*x - eps*x^3').noise('W_x', Gaussian(var=1),
                                                                      into={'x': 'sqrt(2*D)'})
    _same(a, c)                     # interpretation='ito' is the default


def test_signs_brackets_and_a_left_factor():
    a = _p(SDE('m'), tau=2.0, E=0.5).sde('tau*dv = (E - v) dt - (mu + eps) dW_a')
    b = (_p(SDE('m'), tau=2.0, E=0.5).field('v').equation('v', lhs='tau*Dt*v', rhs='E - v')
         .noise('a', Gaussian(var=1), into={'v': '-(mu + eps)'}))
    _same(a, b)
    c = _p(SDE('m')).sde('dx = -(mu*x + eps*x^3) dt + dW')
    assert c._eqs['x']['rhs_raw'] == '-(mu*x + eps*x^3)'
    d = _p(SDE('m')).sde('dx = -mu*x*dt - (eps*x^3 - D)*dt + dW')
    assert d._eqs['x']['rhs_raw'] == '-mu*x - (eps*x^3 - D)'
    e = _p(SDE('m')).sde('dx = -mu*x dt + 1e-3*x dt + dW')
    assert e._eqs['x']['rhs_raw'] == '-mu*x + 1e-3*x'


def test_a_named_process_shared_by_two_equations_is_one_source():
    a = _p(SDE('two'), k=0.5).sde('''
        dx = (-mu*x + k*y) dt + sqrt(D) dW_c + dW
        dy = (-mu*y) dt - sqrt(D) dW_c
    ''')
    b = (_p(SDE('two'), k=0.5).field('x').field('y').drift('x', '-mu*x + k*y').drift('y', '-mu*y')
         .noise('c', Gaussian(var=1), into={'x': 'sqrt(D)', 'y': '-sqrt(D)'})
         .noise('W_x', Gaussian(var=1), into={'x': 1}))
    _same(a, b)
    assert list(a._sources) == ['c', 'W_x']


def test_jumps_and_given_sources():
    a = _p(SDE('j'), lam=0.5).sde('dx = (-mu*x) dt + 0.3 dN(lam*exp(-x^2)) + dL_g',
                                  noises={'g': GammaProcess(shape_rate=2, scale='D')})
    b = (_p(SDE('j'), lam=0.5).field('x').drift('x', '-mu*x')
         .noise('N_x', Poisson(rate='lam*exp(-x^2)'), into={'x': '0.3'})
         .noise('g', GammaProcess(shape_rate=2, scale='D'), into={'x': 1}))
    _same(a, b)
    c = _p(SDE('j')).sde('dx = (-mu*x) dt + dW_a', noises={'a': Gaussian(var='2*D')})
    _same(c, _p(SDE('j')).field('x').drift('x', '-mu*x').noise('a', Gaussian(var='2*D'), into={'x': 1}))


def test_existing_fields_and_declared_time_derivative_names_are_used():
    from daedalus.api.operators import TimeDerivative
    a = _p(SDE('o')).operator('dt_', TimeDerivative()).field('x', domain='positive') \
        .sde('dx = (mu - x) dt + sqrt(x) dW')
    assert a._eqs['x']['lhs_raw'] == 'dt_*x' and a._fields['x']['domain'] == 'positive'


# ── interpretations ─────────────────────────────────────────────────────────

@pytest.mark.parametrize('mark', ['∘ ', '∘', 'o ', '*o*', '◇ '])
def test_all_marked_gaussian_terms_are_the_stratonovich_model(mark):
    a = _p(SDE('s')).sde(f'dx = (-mu*x) dt + sqrt(D)*x {mark}dW')
    b = (_p(SDE('s')).field('x').drift('x', '-mu*x')
         .noise('W_x', Gaussian(var=1), into={'x': 'sqrt(D)*x'}).interpretation('stratonovich'))
    _same(a, b)
    assert a._interp_mode() == 'stratonovich'
    assert a.latex() == b.latex()
    ra, rb = a.check(verbose=False), b.check(verbose=False)
    assert ra['interpretation'] == rb['interpretation'] == 'stratonovich'
    assert ra['corrections'] == rb['corrections'] and ra['vertices'] == rb['vertices']


def test_all_marked_terms_with_a_jump_are_the_marcus_model():
    text = 'dx = (-mu*x) dt + sqrt(D)*x ∘ dW + (1 + b*x) ∘ dN_j(lam)'
    a = _p(SDE('mc'), lam=0.5, b=-0.2).sde(text)
    b = (_p(SDE('mc'), lam=0.5, b=-0.2).field('x').drift('x', '-mu*x')
         .noise('W_x', Gaussian(var=1), into={'x': 'sqrt(D)*x'})
         .noise('j', Poisson(rate='lam'), into={'x': '1 + b*x'}).interpretation('marcus'))
    _same(a, b)
    assert a._interp_mode() == 'marcus' and a.build()['interpretation'] == 'marcus'
    assert a.latex() == b.latex() and r'\diamond' in a.latex()['sde']
    assert a.check(verbose=False)['corrections'] == b.check(verbose=False)['corrections']


def test_mixed_gaussian_terms_are_the_ito_model_with_the_hand_correction():
    """dx = -mu x dt + s dW_1 + g x o dW_2: only the marked source gets the Stratonovich drift g^2 x / 2."""
    a = _p(SDE('mix'), s=0.7, g=0.3).sde('dx = (-mu*x) dt + s dW_a + g*x ∘ dW_b')
    explicit = (_p(SDE('mix'), s=0.7, g=0.3).field('x').drift('x', '-mu*x')
                .noise('a', Gaussian(var=1), into={'x': 's'}, interpretation='ito')
                .noise('b', Gaussian(var=1), into={'x': 'g*x'}, interpretation='stratonovich'))
    _same(a, explicit)
    hand = (_p(SDE('mix'), s=0.7, g=0.3).field('x').drift('x', '-mu*x + g^2*x/2')
            .noise('a', Gaussian(var=1), into={'x': 's'}).noise('b', Gaussian(var=1), into={'x': 'g*x'}))
    assert (_action(a) - _action(hand)).expand().is_zero()
    assert a._interp_mode() == 'mixed' and 'interpretation' not in a.build()
    rep = a.check(verbose=False)
    assert rep['interpretation'] == 'mixed'
    assert rep['interpretations'] == {'a': 'ito', 'b': 'stratonovich'}
    assert [c['source'] for c in rep['corrections']] == ['b']
    tex = a.latex()['sde']
    assert r's\,\mathrm{d}W_{a}' in tex and r'\circ\,\mathrm{d}W_{b}' in tex


def test_mixed_ito_gaussian_and_marcus_jump_is_the_marcus_jump_term_plus_the_ito_gaussian():
    a = _p(SDE('mj'), lam=0.5, b=-0.2).sde(
        'dx = (-mu*x) dt + sqrt(D)*x dW + (1 + b*x) ∘ dN(lam)')
    explicit = (_p(SDE('mj'), lam=0.5, b=-0.2).field('x').drift('x', '-mu*x')
                .noise('W_x', Gaussian(var=1), into={'x': 'sqrt(D)*x'}, interpretation='ito')
                .noise('N_x', Poisson(rate='lam'), into={'x': '1 + b*x'}, interpretation='marcus'))
    _same(a, explicit)
    # the Marcus term of the jump alone (a Marcus model without the Gaussian) plus the Ito Gaussian term
    jump_only = (_p(SDE('mj'), lam=0.5, b=-0.2).field('x').drift('x', '-mu*x')
                 .noise('N_x', Poisson(rate='lam'), into={'x': '1 + b*x'}).interpretation('marcus'))
    gauss_ito = (_p(SDE('mj'), lam=0.5, b=-0.2).field('x').drift('x', '-mu*x')
                 .noise('W_x', Gaussian(var=1), into={'x': 'sqrt(D)*x'}))
    x, xt = SR.var('x__i'), SR.var('xt__i')
    dyn = xt * (SR.var('Dt') * x + SR.var('mu') * x)
    assert (_action(a) - (_action(jump_only) + _action(gauss_ito) - dyn)).expand().is_zero()
    m = a.build()
    assert m['interpretation'] == 'marcus' and m['marcus_exact_order']['sources'] == ['N_x']
    assert a.check(verbose=False)['interpretations'] == {'W_x': 'ito', 'N_x': 'marcus'}


def test_mixed_model_simulates_with_the_per_source_rules():
    a = _p(SDE('sim'), s=0.5, g=0.4).sde('dx = (-mu*x) dt + s dW_a + g*x ∘ dW_b')
    a.action_text()
    assert list(a._ito_drift) == ['x'] and len(a._ito_drift['x']) == 1      # source b only
    rec = a.simulate(T=20.0, dt=0.01, n_paths=4, seed=1, burn_in=1.0)
    assert rec is not None


# ── refusals ────────────────────────────────────────────────────────────────

@pytest.mark.parametrize('text,word,char', [
    ('dx = (-mu*x) + sqrt(2*D) dW', 'no differential', ')'),
    ('dx = (-mu*x) ∘ dt + dW', 'not the drift', '∘'),
    ('dx = -mu*x dt + dN', 'needs its rate', None),
    ('dx = -mu*x dt + dZ_q', "no source 'q'", 'd'),
    ('dx = -mu*x dt + dW + 2*dW', 'appears twice', None),
    ('x = -mu*x dt + dW', 'not d<field>', 'x'),
    ('dx = -mu*x dt + sqrt(2 D) dW', 'missing operator', None),
    ('dx -mu*x dt', 'needs "="', None),
    ('dx = -mu*x dt = dW', 'second "="', '='),
])
def test_malformed_lines_point_at_the_problem(text, word, char):
    with pytest.raises(SDEError) as ei:
        _p(SDE('bad')).sde(text)
    msg = str(ei.value)
    assert word in msg and '^' in msg, msg
    if char is not None:
        line, caret = msg.splitlines()[1:3]
        assert line[caret.index('^')] == char, msg


def test_one_source_takes_one_interpretation_and_one_rate():
    with pytest.raises(SDEError, match='marked in one equation and not in another'):
        _p(SDE('b')).sde('dx = -x dt + ∘ dW_c\ndy = -y dt + dW_c')
    with pytest.raises(SDEError, match='one source has one rate'):
        _p(SDE('b')).sde('dx = -x dt + dN_c(mu)\ndy = -y dt + dN_c(D)')
    with pytest.raises(SDEError, match='two equations'):
        _p(SDE('b')).sde('dx = -x dt + dW\ndx = -x dt')


def test_model_wide_interpretation_does_not_mix_with_paper_terms():
    with pytest.raises(SDEError, match='carry their own'):
        _p(SDE('b')).interpretation('stratonovich').sde('dx = -x dt + x dW')
    m = _p(SDE('b')).sde('dx = -x dt + x dW')
    with pytest.raises(SDEError, match='carry their own interpretation'):
        m.interpretation('marcus')
    m.interpretation('ito')                       # (the default: no conflict)


def test_unrepresentable_combinations_are_refused():
    with pytest.raises(SDEError, match=r"noise\('j', interpretation='stratonovich'\).*Marcus rule"):
        (_p(SDE('b'), lam=0.5).field('x').drift('x', '-x')
         .noise('j', Poisson(rate='lam'), into={'x': 'x'}, interpretation='stratonovich').build())
    with pytest.raises(SDEError, match='exposed source'):
        (_p(SDE('b'), lam=0.5).field('x').drift('x', '-x + n')
         .noise('j', Poisson(rate='lam'), expose='n', interpretation='marcus').build())
    with pytest.raises(SDEError, match='joint'):
        (_p(SDE('b', n_populations=2), lam=0.5).field('x').drift('x', '-x')
         .noise('j', Poisson(rate='lam'), into={'x': 'x[j]'}, interpretation='marcus').build())
    with pytest.raises(SDEError, match="noise\\('n'\\) is stratonovich"):
        from daedalus.api.operators import Laplacian
        (SDE('s', space=1).operator('lap', Laplacian()).field('u').drift('u', 'lap(u) - u')
         .noise('n', Gaussian(var=1, space='white'), into={'u': 1}, interpretation='stratonovich')
         .build())
    with pytest.raises(SDEError, match="use 'ito'"):
        SDE('b').field('x').noise('n', Gaussian(var=1), into={'x': 1}, interpretation='strat')


def test_a_per_source_interpretation_overrides_the_model_wide_one():
    a = (_p(SDE('o')).field('x').drift('x', '-mu*x').interpretation('stratonovich')
         .noise('a', Gaussian(var=1), into={'x': 'x'})
         .noise('b', Gaussian(var=1), into={'x': 'x'}, interpretation='ito'))
    assert a._interp_mode() == 'mixed'
    assert [c['source'] for c in a.check(verbose=False)['corrections']] == ['a']


# ── round trip ──────────────────────────────────────────────────────────────

_MODELS = {
    'ou_quartic': lambda: _p(SDE('r')).sde('dx = (-mu*x - eps*x^3) dt + sqrt(2*D) dW'),
    'shared': lambda: _p(SDE('r'), k=0.5).sde('dx = (-mu*x + k*y) dt + sqrt(D) o dW_c\n'
                                             'dy = (-mu*y) dt - sqrt(D) ∘ dW_c + dW'),
    'mixed': lambda: _p(SDE('r'), lam=0.5, b=-0.2, tau=2.0).sde(
        'tau*dx = (-mu*x) dt + sqrt(D)*x dW + (1 + b*x) ∘ dN(lam)'),
    'given': lambda: _p(SDE('r'), lam=0.5).sde(
        'dx = (-mu*x) dt + dL_g', noises={'g': CompoundPoisson(rate='lam', jump=Exponential(mean='D'))}),
    'explicit_strat': lambda: (_p(SDE('r')).field('x').drift('x', '-mu*x')
                               .noise('W_x', Gaussian(var=1), into={'x': 'x'}).interpretation('stratonovich')),
}


@pytest.mark.parametrize('name', sorted(_MODELS))
def test_sde_text_round_trips(name):
    m = _MODELS[name]()
    text = m.sde_text()
    again = SDE(m.name)
    again._params, again._param_names = list(m._params), list(m._param_names)
    again.sde(text, noises=text.noises)
    _same(again, m)
    assert again.latex() == m.latex()
    assert again.check(verbose=False)['corrections'] == m.check(verbose=False)['corrections']


def test_sde_text_is_readable():
    assert _MODELS['ou_quartic']().sde_text() == 'dx = (-mu*x - eps*x^3) dt + sqrt(2*D) dW'
    assert _MODELS['mixed']().sde_text() == \
        'tau*dx = (-mu*x) dt + sqrt(D)*x dW + (1 + b*x) ∘ dN(lam)'


def test_sde_text_refuses_what_it_cannot_write():
    assert cat.ou_quartic(drift_form=False).sde_text().startswith('(Dt+mu)*x[i] = ')   # the operator form
    with pytest.raises(SDEError, match='point process or exposed'):
        cat.linear_hawkes().sde_text()


def test_catalogue_models_that_paper_text_writes_round_trip():
    done = 0
    for name in dir(cat):
        f = getattr(cat, name)
        if name.startswith('_') or not callable(f) or name in ('digest', 'SDE', 'Kernel') \
                or isinstance(f, type):
            continue
        try:
            m = f()
        except TypeError:
            continue
        if not isinstance(m, SDE) or m._functions or m._kernels or m._pops or m._extra:
            continue
        try:
            text = m.sde_text()
        except SDEError:
            continue
        again = SDE(m.name)
        again._params, again._param_names = list(m._params), list(m._param_names)
        for f_, info in m._fields.items():
            again.field(f_, **{k: info[k] for k in ('population', 'description', 'domain', 'latex')})
        again.series_order(m._series_order).marcus_route(m._marcus_route)
        again.sde(text, noises=text.noises)
        # (the text declares sources in order of first appearance, so a source shared by two
        # equations may move: the same action terms, possibly in another order)
        assert sorted(again.action_text().splitlines()) == sorted(m.action_text().splitlines()), name
        assert again.equations() == m.equations(), name
        done += 1
    assert done >= 3
