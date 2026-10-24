"""Declared operators (api/operators.py) in the SDE front end, and the SDE -> SPDE bridge (api/sde.py).

A. Temporal models are unchanged: every model of tests/_sde_model_catalogue.py gives the action text, mean-field
   equations, notes and built model dict of the snapshot taken before declared operators existed; renaming ``Dt``
   and the call style give the same bytes.
B. The bridge (``SDE(..., space=d)`` / a Laplacian) builds the model the tracked hand-written spatial models build:
   the same dict (fields, parameters, equations, operators, spatial block, ...), an action text equal as an
   expression, and the same expanded field theory; an end-to-end k=2 run (tree + one loop) is bit-identical.
C. Spatial refusals are unchanged through the bridge; temporal-only features with a Laplacian, and operator names
   that collide with the model's names, are refused.
"""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip('sage.all')
from sage.all import SR                                                       # noqa: E402

from api.model import SpatialModelBuilder                                     # noqa: E402
from api.noise import (Gaussian, NoiseError, Poisson, parse_expr)             # noqa: E402
from api.operators import Laplacian, Operator, TimeDerivative                 # noqa: E402
from api.sde import SDE, SDEError                                             # noqa: E402

from tests._sde_model_catalogue import CATALOGUE, digest                      # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
GOLDEN = json.loads((ROOT / 'tests' / 'fixtures' / 'declared_operators_sde_golden.json').read_text('utf-8'))


def _json(d):
    return json.loads(json.dumps(d, sort_keys=True))


# ── A: temporal identity ─────────────────────────────────────────────────────────────────────

def test_catalogue_matches_the_snapshot_list():
    assert sorted(CATALOGUE) == sorted(GOLDEN)


@pytest.mark.parametrize('name', sorted(CATALOGUE))
def test_temporal_model_is_byte_identical_to_the_snapshot(name):
    assert _json(digest(CATALOGUE[name]())) == GOLDEN[name]


def _ou(lhs=None, drift=None, rename=None, explicit_dt=False):
    s = SDE('OU Quartic (white noise)')
    if explicit_dt:
        s.operator('Dt', TimeDerivative())
    if rename:
        s.operator(rename, TimeDerivative())
    s.population('pop', size=1).field('x', population='pop', description='variable')
    for p, d in (('mu', 1.0), ('eps', 0.02), ('D', 1.0)):
        s.parameter(p, default=d, domain='positive')
    if drift is not None:
        s.drift('x', drift)
    else:
        s.equation('x', lhs=lhs, rhs='-eps*x^3')
    return s.noise('xi', Gaussian(var='2*D'), couples={'x': 1})


@pytest.mark.parametrize('variant', [
    dict(drift='-mu*x - eps*x^3', explicit_dt=True),                    # Dt redeclared explicitly
    dict(drift='-mu*x - eps*x^3', rename='dt'),                         # Dt renamed: drift() uses dt
    dict(drift='-mu*x - eps*x^3', rename='Δ'),                          # a Unicode identifier
])
def test_renamed_time_derivative_gives_the_same_bytes_drift_form(variant):
    assert _json(digest(_ou(**variant))) == GOLDEN['ou_quartic_drift']


@pytest.mark.parametrize('lhs,rename', [('(dt+mu)*x[i]', 'dt'), ('(d_t+mu)*x[i]', 'd_t'),
                                        ('(Dt+mu)*x[i]', None)])
def test_renamed_time_derivative_gives_the_same_bytes_equation_form(lhs, rename):
    assert _json(digest(_ou(lhs=lhs, rename=rename))) == GOLDEN['ou_quartic_equation']


def test_call_style_gives_the_same_bytes_as_product_style():
    """``dt(x)`` is ``Dt*x``: the call-style equation equals drift() (lhs 'Dt*x') byte for byte."""
    call = (SDE('OU Quartic (white noise)').operator('dt', TimeDerivative())
            .population('pop', size=1).field('x', population='pop', description='variable'))
    for p, d in (('mu', 1.0), ('eps', 0.02), ('D', 1.0)):
        call.parameter(p, default=d, domain='positive')
    call.equation('x', lhs='dt(x)', rhs='-mu*x - eps*x^3').noise('xi', Gaussian(var='2*D'), couples={'x': 1})
    assert _json(digest(call)) == GOLDEN['ou_quartic_drift']


def test_call_and_product_style_are_the_same_model():
    """``tau*dt(v) + v`` and ``(tau*dt + 1)*v``: the same action and equations as expressions, the same model."""
    def hawkes(lhs):
        return (SDE('exp Hawkes 1-pop (SDE)').operator('dt', TimeDerivative())
                .parameter('Em', default=-1.5).parameter('tau', default=2.0, domain='positive')
                .parameter('taug', default=1.0, domain='positive').parameter('a', default=1.0, domain='positive')
                .parameter('w', default=0.4)
                .define_kernel('g', time_expr='exp(-t/taug)/taug*heaviside(t)')
                .noise('spikes', Poisson(rate='a*exp(v)'), expose='n')
                .field('v').equation('v', lhs=lhs, rhs='Em + w*g*n').stability_analysis(True))
    prod, call = hawkes('(tau*dt + 1)*v'), hawkes('tau*dt(v) + v')
    assert _json(digest(prod)) == GOLDEN['exp_hawkes']
    assert _same_text(prod.action_text(), call.action_text())
    for a, b in zip(prod.equations(), call.equations()):
        assert _same_text(a['lhs'] + '-(' + a['rhs'] + ')', b['lhs'] + '-(' + b['rhs'] + ')')


def test_into_is_couples():
    a = _ou(drift='-mu*x - eps*x^3')
    b = (SDE('OU Quartic (white noise)').population('pop', size=1)
         .field('x', population='pop', description='variable'))
    for p, d in (('mu', 1.0), ('eps', 0.02), ('D', 1.0)):
        b.parameter(p, default=d, domain='positive')
    b.drift('x', '-mu*x - eps*x^3').noise('xi', Gaussian(var='2*D'), into={'x': 1})
    assert _json(digest(a)) == _json(digest(b)) == GOLDEN['ou_quartic_drift']
    with pytest.raises(SDEError, match='not both'):
        SDE('m').field('x').noise('xi', Gaussian(var=1), couples={'x': 1}, into={'x': 1})


def test_renamed_dt_is_no_longer_accepted():
    s = SDE('m').operator('dt', TimeDerivative()).parameter('mu').field('x')
    with pytest.raises(SDEError, match="Dt is not declared in this model; its time derivative is named dt"):
        s.equation('x', lhs='Dt*x', rhs='-mu*x')
    with pytest.raises(SDEError, match='must contain the time derivative dt'):
        s.equation('x', lhs='x', rhs='-mu*x')
    with pytest.raises(SDEError, match='dt found in rhs'):
        s.equation('x', lhs='dt(x)', rhs='-mu*dt(x)')


def test_time_derivative_of_an_expression_is_refused():
    s = SDE('m').operator('dt', TimeDerivative()).parameter('mu').field('x')
    with pytest.raises(SDEError, match=r'dt\(x\^2\) acts on a nonlinear expression'):
        s.equation('x', lhs='dt(x^2)', rhs='-mu*x')
    with pytest.raises(SDEError, match='takes one argument'):
        s.equation('x', lhs='dt(x, x)', rhs='-mu*x')


# ── collisions and reserved names ────────────────────────────────────────────────────────────

@pytest.mark.parametrize('declare,op_name,match', [
    (lambda s: s.field('u'), 'u', "already the field 'u'"),
    (lambda s: s.field('u'), 'ut', "response field of 'u'"),
    (lambda s: s.field('u'), 'du', "fluctuation of 'u'"),
    (lambda s: s.field('u'), 'ustar', "mean-field saddle of 'u'"),
    (lambda s: s.parameter('D'), 'D', "already the parameter 'D'"),
    (lambda s: s.define_function('f', ['v'], 'v^2'), 'f', "already the function 'f'"),
    (lambda s: s.define_kernel('g', freq_image='1/(1+I*omega)'), 'g', "already the kernel 'g'"),
    (lambda s: s.population('E', size=2), 'E', "already the population 'E'"),
])
def test_operator_name_colliding_with_a_declared_name_is_refused(declare, op_name, match):
    s = SDE('m')
    declare(s)
    with pytest.raises(SDEError, match=match):
        s.operator(op_name, Laplacian())


@pytest.mark.parametrize('declare,kind', [
    (lambda s: s.field('lap'), 'field'),
    (lambda s: s.parameter('lap'), 'parameter'),
    (lambda s: s.define_function('lap', ['v'], 'v^2'), 'function'),
    (lambda s: s.define_kernel('lap', freq_image='1/(1+I*omega)'), 'kernel'),
    (lambda s: s.population('lap'), 'population'),
])
def test_declaring_a_name_after_an_operator_of_that_name_is_refused(declare, kind):
    s = SDE('m', space=1).operator('lap', Laplacian())
    with pytest.raises(SDEError, match=f"{kind} name 'lap' is a declared operator"):
        declare(s)


def test_field_whose_derived_name_is_an_operator_is_refused():
    s = SDE('m').operator('dx', TimeDerivative())
    with pytest.raises(SDEError, match="derived name 'dx' is a declared operator"):
        s.field('x')


@pytest.mark.parametrize('name,op,match', [
    ('Dt', Laplacian(), 'reserved for the TimeDerivative'),
    ('Laplacian', TimeDerivative(), 'reserved for the Laplacian'),
    ('Lap', TimeDerivative(), 'reserved for the Laplacian'),
    ('exp', Laplacian(), 'reserved'),
    ('t', TimeDerivative(), 'reserved'),
    ('i', Laplacian(), 'reserved'),
    ('sum', Laplacian(), 'reserved'),
    ('gamma', Laplacian(), 'reserved'),
    ('for', Laplacian(), 'Python identifier'),
    ('∂t', TimeDerivative(), 'Python identifier'),
    ('a__b', Laplacian(), 'double underscore'),
])
def test_reserved_operator_names_are_refused(name, op, match):
    with pytest.raises(SDEError, match=match):
        SDE('m').operator(name, op)


def test_operator_declared_twice_and_non_operators_are_refused():
    s = SDE('m').operator('lap', Laplacian())
    with pytest.raises(SDEError, match='is declared'):
        s.operator('lap', Laplacian())
    with pytest.raises(SDEError, match='an operator from api.operators'):
        s.operator('L2', 'Laplacian')
    assert isinstance(SDE('m').operator('L2', Laplacian)._operators['L2'], Operator)   # a class is instantiated


def test_operator_shadowing_a_function_is_refused_in_either_order():
    """The adversarial case: an operator named like a declared function (a call ``f(u)`` would silently change
    meaning) is refused whichever is declared first."""
    a = SDE('m', space=1).define_function('f', ['v'], 'v^3')
    with pytest.raises(SDEError, match="already the function 'f'"):
        a.operator('f', Laplacian())
    b = SDE('m', space=1).operator('f', Laplacian())
    with pytest.raises(SDEError, match="function name 'f' is a declared operator"):
        b.define_function('f', ['v'], 'v^3')


# ── B: the SPDE bridge against the tracked spatial models ───────────────────────────────────

def _tracked(name):
    spec = importlib.util.spec_from_file_location(f'decl_ops_{name}', ROOT / 'models' / f'{name}.model.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _white(var):
    return Gaussian(var=var, space='white')


def sde_allen_cahn(boundary=None, style='operator'):
    name = ('1D stochastic Allen-Cahn (subcritical, periodic domain)' if boundary
            else '1D stochastic Allen-Cahn (subcritical, infinite domain)')
    s = (SDE(name, space=1).operator('dt', TimeDerivative()).operator('lap', Laplacian())
         .field('phi', description='order parameter')
         .parameter('mu', default=1.0, domain='positive')
         .parameter('D', default=1.0, domain='positive')
         .parameter('lam', default=0.1, domain='positive')
         .parameter('T', default=1.0, domain='positive'))
    if boundary:
        s.parameter('L', default=20.0, domain='positive').boundary('periodic', length='L')
    if style == 'operator':
        s.equation('phi', lhs='(dt + mu - D*lap)*phi', rhs='-lam*phi^3')
    else:                                               # the call style of a drift
        s.equation('phi', lhs='dt(phi)', rhs='D*lap(phi) - mu*phi - lam*phi^3')
    return s.noise('eta', _white('2*T'), into={'phi': 1}).stability_analysis(True)


def sde_rd_quadratic_1d():
    return (SDE('1D reaction-diffusion (quadratic, bubble test)', space=1).operator('lap', Laplacian())
            .field('phi', description='density field')
            .parameter('mu', default=1.0, domain='positive', description='linear relaxation / mass')
            .parameter('D', default=1.0, domain='positive', description='diffusion constant')
            .parameter('g', default=0.3, domain='real', description='quadratic reaction coupling')
            .parameter('T', default=1.0, domain='positive', description='noise temperature')
            .equation('phi', lhs='(Dt + mu - D*lap)*phi', rhs='-g*phi^2')
            .noise('eta', _white('2*T'), into={'phi': 1}))


def sde_rd_2d():
    return (SDE('reaction-diffusion (d=2)', space=2).operator('Δ', Laplacian())
            .field('phi')
            .parameter('mu', default=1.0, domain='positive')
            .parameter('D', default=1.0, domain='positive')
            .parameter('g', default=0.2, domain='real')
            .parameter('T', default=1.0, domain='positive')
            .equation('phi', lhs='(Dt + mu - D*Δ)*phi', rhs='-g*phi^2')
            .noise('eta', _white('2*T'), into={'phi': 1}))


def sde_coupled_rd():
    return (SDE('coupled-rd-2species-1d', space=1).operator('lap', Laplacian())
            .field('a').field('b')
            .parameter('mua', default=1.5, domain='positive')
            .parameter('mub', default=1.2, domain='positive')
            .parameter('Da', default=0.8, domain='positive')
            .parameter('Db', default=0.8, domain='positive')
            .parameter('g', default=0.4).parameter('h', default=0.3)
            .parameter('ga', default=0.3).parameter('gb', default=0.3)
            .parameter('Ta', default=1.0, domain='positive')
            .parameter('Tb', default=0.7, domain='positive')
            .equation('a', lhs='(Dt+mua-Da*lap)*a + g*b + ga*a^3', rhs='0')
            .equation('b', lhs='(Dt+mub-Db*lap)*b - h*a + gb*b^3', rhs='0')
            .noise('eta_a', _white('2*Ta'), into={'a': 1})
            .noise('eta_b', _white('2*Tb'), into={'b': 1}))


def sde_linear_diffusion():
    return (SDE('Linear Diffusion Test', space=1).operator('lap', Laplacian())
            .field('phi')
            .parameter('mu', default=1.0, domain='positive')
            .parameter('D', default=1.0, domain='positive')
            .parameter('T', default=1.0, domain='positive')
            .equation('phi', lhs='(Dt + mu - D*lap)*phi', rhs='0')
            .noise('eta', _white('2*T'), into={'phi': 1}))


def sde_conserved_rd():
    """``g*lap(phi^2)``: a Laplacian acting on an expression puts the action in the spatial operator IR."""
    return (SDE('1D conserved reaction-diffusion (derivative vertex)', space=1)
            .operator('lap', Laplacian())
            .field('phi', description='conserved density')
            .parameter('mu', default=1.0, domain='positive')
            .parameter('D', default=2.0, domain='positive')
            .parameter('g', default=0.3, domain='real')
            .parameter('T', default=1.0, domain='positive')
            .equation('phi', lhs='(Dt + mu - D*lap)*phi', rhs='g*lap(phi^2)')
            .noise('eta', _white('2*T'), into={'phi': 1}))


BRIDGE_CASES = {
    'allen_cahn_1d_subcritical_infinite': lambda: sde_allen_cahn(),
    'allen_cahn_1d_subcritical_pbc': lambda: sde_allen_cahn(boundary='periodic'),
    'reaction_diffusion_quadratic_1d': sde_rd_quadratic_1d,
    'reaction_diffusion_2d': sde_rd_2d,
    'coupled_rd_2species_1d': sde_coupled_rd,
    'linear_diffusion_test': sde_linear_diffusion,
    'reaction_diffusion_conserved_1d': sde_conserved_rd,
}


def _unsum(text):
    """The body of a one-population action ``sum(... for i in pop)``."""
    t = ' '.join(text.split())
    if t.startswith('sum( ') and t.endswith(' for i in pop)'):
        t = t[len('sum( '):-len(' for i in pop)')]
    return t


def _same_text(a, b):
    """Two action / residual texts are the same expression (operators as symbols or formal calls)."""
    return bool((parse_expr(_unsum(a)) - parse_expr(_unsum(b))).expand().is_zero())


def _canon(v):
    if callable(v):
        return '<callable>'
    if isinstance(v, dict):
        return {str(k): _canon(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_canon(x) for x in v]
    return v if v is None or isinstance(v, (bool, int, float, str)) else repr(v)


def _expanded(model):
    """The expanded action by (response, physical) grade, printed (the polynomial ring's canonical form)."""
    from engine.core.field_theory import FieldTheory
    ft = FieldTheory(model, taylor_order=4)
    ft.expand()
    return {tp: str(v) for tp, v in ft._by_tp.items()}


def assert_same_model(got, ref):
    """The bridge's dict is the hand-written one: equal keys and values, except the action text (equal as an
    expression; ``spec_signature`` hashes it) and the compiled hooks (the same expanded field theory)."""
    assert set(got) == set(ref)
    text_keys = {'action_text', 'spec_signature'}
    for k in sorted(set(ref) - text_keys):
        assert _canon(got[k]) == _canon(ref[k]), k
    assert _same_text(got['action_text'], ref['action_text'])
    assert _expanded(got) == _expanded(ref)


@pytest.mark.parametrize('name', sorted(BRIDGE_CASES))
def test_bridge_builds_the_tracked_spatial_model(name):
    sde = BRIDGE_CASES[name]()
    assert isinstance(sde.to_builder(), SpatialModelBuilder)
    assert_same_model(sde.build(), _tracked(name).build())


def test_bridge_equations_and_ir_flag():
    assert sde_rd_quadratic_1d().equations() == [
        {'lhs': '(Dt + mu - D*Laplacian)*phi', 'rhs': '-g*phi^2', 'population': None}]
    m = sde_conserved_rd().build()
    assert m['operator_ir'] is True
    assert m['equations'][0]['rhs_text'] == 'g*Laplacian*phi^2'
    assert 'Lap(phi^2)' in m['action_text'] and 'Dt(phi)' in m['action_text']
    assert sde_rd_quadratic_1d().build()['operator_ir'] is False


def test_call_style_drift_is_the_tracked_field_theory():
    """``dt(phi) = D*lap(phi) - mu*phi - lam*phi^3`` (the drift written in call style) expands to the same field
    theory as the tracked operator-form model; its mean-field equation is the residual of the same system."""
    got, ref = sde_allen_cahn(style='call').build(), _tracked('allen_cahn_1d_subcritical_infinite').build()
    assert _expanded(got) == _expanded(ref)
    (eq,), (eq_ref,) = got['equations'], ref['equations']
    zero = {SR.var('Dt'): 0, SR.var('Laplacian'): 0}
    res = parse_expr(eq['lhs_text']) - parse_expr(eq['rhs_text'])
    res_ref = parse_expr(eq_ref['lhs_text']) - parse_expr(eq_ref['rhs_text'])
    assert bool((res - res_ref).subs(zero).expand().is_zero())


def test_product_and_call_style_laplacian_are_the_same_model():
    a = sde_rd_quadratic_1d()
    b = (SDE('1D reaction-diffusion (quadratic, bubble test)', space=1).operator('lap', Laplacian())
         .field('phi', description='density field')
         .parameter('mu', default=1.0, domain='positive', description='linear relaxation / mass')
         .parameter('D', default=1.0, domain='positive', description='diffusion constant')
         .parameter('g', default=0.3, domain='real', description='quadratic reaction coupling')
         .parameter('T', default=1.0, domain='positive', description='noise temperature')
         .equation('phi', lhs='Dt(phi) + mu*phi - D*lap(phi)', rhs='-g*phi^2')
         .noise('eta', _white('2*T'), into={'phi': 1}))
    assert b.equations() == [{'lhs': 'Dt*phi + mu*phi - D*Laplacian*phi', 'rhs': '-g*phi^2', 'population': None}]
    ga, gb = a.build(), b.build()
    assert _same_text(ga['action_text'], gb['action_text'])
    assert _expanded(ga) == _expanded(gb)


def _run(model, max_ell, spatial_grid=None, **kw):
    from api import compute_cumulants
    return compute_cumulants(model=model, k=2, max_ell=max_ell,
                             fundamental={'mu': 1.0, 'D': 1.0, 'lam': 0.1, 'T': 1.0},
                             external_fields=[('phi', 1), ('phi', 2)], tau_max=1.0, tau_step=0.5,
                             spatial_grid=np.array([0.0, 1.0, 2.5]) if spatial_grid is None else spatial_grid,
                             parallel=False, verbose=False, use_cache=False, **kw)


def test_end_to_end_spde_k2_tree_and_one_loop_bit_identical():
    """Allen-Cahn written as an SDE in call style through the bridge vs the tracked hand-written spatial model:
    C(x, tau) at tree level and at one loop, bit for bit."""
    def same(a, b):
        for key in ('C_tau', 'C_tau_x', 'tau_grid', 'spatial_grid'):
            x, y = np.asarray(a[key]), np.asarray(b[key])
            assert x.shape == y.shape and np.array_equal(x, y), key
        for key in ('C_tau_x_by_ell', 'C_tau_x_by_order'):
            assert sorted(a[key]) == sorted(b[key]), key
            for o in b[key]:
                assert np.array_equal(np.asarray(a[key][o]), np.asarray(b[key][o])), (key, o)
        def plain(v):
            if isinstance(v, dict):
                return {k: plain(x) for k, x in v.items()}
            if isinstance(v, (list, tuple, np.ndarray)):
                return [plain(x) for x in v]
            return v
        for key in ('status', 'mf_values', 'route', 'spatial_info'):
            assert plain(a[key]) == plain(b[key]), key

    got = _run(sde_allen_cahn(style='call').build(), max_ell=1)
    ref = _run(_tracked('allen_cahn_1d_subcritical_infinite').build(), max_ell=1)
    assert ref['status'] == 'ok' and sorted(ref['C_tau_x_by_ell']) == [0, 1]
    same(got, ref)
    tree = _run(sde_allen_cahn(style='call').build(), max_ell=0)
    tree_ref = _run(_tracked('allen_cahn_1d_subcritical_infinite').build(), max_ell=0)
    same(tree, tree_ref)
    assert not np.array_equal(np.asarray(tree['C_tau_x']), np.asarray(got['C_tau_x']))   # the loop moved it


# ── C: refusals ─────────────────────────────────────────────────────────────────────────────

def _raises_same(make_got, make_ref, call):
    errs = []
    for make in (make_got, make_ref):
        with pytest.raises(Exception) as ei:
            call(make())
        errs.append(ei.value)
    assert type(errs[0]) is type(errs[1]) and str(errs[0]) == str(errs[1])
    return errs[0]


def test_spatial_refusals_are_unchanged_through_the_bridge():
    # one-loop on a periodic domain (the loop integral is the infinite-domain one)
    e = _raises_same(lambda: sde_allen_cahn(boundary='periodic').build(),
                     lambda: _tracked('allen_cahn_1d_subcritical_pbc').build(),
                     lambda m: _run(m, max_ell=1))
    assert 'periodic' in str(e)
    # switch-on transients are temporal-only
    _raises_same(lambda: sde_allen_cahn().build(),
                 lambda: _tracked('allen_cahn_1d_subcritical_infinite').build(),
                 lambda m: _run(m, max_ell=0, t_min=0.0, times=[[1.0, 1.0]]))


def test_uv_divergent_loop_refused_through_the_bridge():
    """phi^4 in d=2: the one-loop tadpole is UV-divergent; the bridge model is refused as the hand-written one."""
    from engine.integration.spatial.full_integrator import SpatialUVDivergenceError

    def hand():
        return (SpatialModelBuilder('AC d=2', n_populations=0).physical_field('phi', spatial_dim=2)
                .parameter('mu', default=1.0, domain='positive').parameter('D', default=1.0, domain='positive')
                .parameter('lam', default=0.1, domain='positive').parameter('T', default=1.0, domain='positive')
                .set_action_text('phit*((Dt + mu - D*Laplacian)*phi + lam*phi^3) - T*phit^2')
                .equation(lhs='(Dt + mu - D*Laplacian)*phi', rhs='-lam*phi^3')
                .boundary('infinite').initial('stationary').build())

    def bridge():
        return (SDE('AC d=2', space=2).operator('lap', Laplacian()).field('phi')
                .parameter('mu', default=1.0, domain='positive').parameter('D', default=1.0, domain='positive')
                .parameter('lam', default=0.1, domain='positive').parameter('T', default=1.0, domain='positive')
                .equation('phi', lhs='(Dt + mu - D*lap)*phi', rhs='-lam*phi^3')
                .noise('eta', _white('2*T'), into={'phi': 1}).build())

    assert_same_model(bridge(), hand())
    e = _raises_same(bridge, hand, lambda m: _run(m, max_ell=1, spatial_grid=np.array([0.5, 1.0])))
    assert isinstance(e, SpatialUVDivergenceError)


@pytest.mark.parametrize('kw,exc', [
    (dict(k=3, max_ell=0), NotImplementedError),                 # spatial k != 2 needs spatial_points
    (dict(k=2, max_ell=3), NotImplementedError),                 # spatial loops stop at max_ell = 2
    (dict(k=2, max_ell=0, spatial_grid=None), Exception),        # a spatial model needs spatial_grid
])
def test_spatial_k_and_loop_gates_unchanged_through_the_bridge(kw, exc):
    from api import compute_cumulants

    def call(m):
        args = dict(model=m, fundamental={'mu': 1.0, 'D': 1.0, 'lam': 0.1, 'T': 1.0},
                    external_fields=[('phi', 1)] * kw['k'], tau_max=1.0, tau_step=0.5,
                    spatial_grid=np.array([0.0, 1.0]), parallel=False, verbose=False, use_cache=False)
        args.update(kw)
        if args['spatial_grid'] is None:
            del args['spatial_grid']
        return compute_cumulants(**args)

    e = _raises_same(lambda: sde_allen_cahn(style='call').build(),
                     lambda: _tracked('allen_cahn_1d_subcritical_infinite').build(), call)
    assert isinstance(e, exc)


def test_derivative_vertex_loop_is_approximate_through_the_bridge():
    """``g*lap(phi^2)`` through the bridge: the one-loop derivative-vertex result is 'approximate' with the same
    accuracy record and the same numbers as the tracked operator-IR model, and strict policy raises for both."""
    from api import compute_cumulants
    from api.compute import SpatialAccuracyError
    kw = dict(k=2, max_ell=1, external_fields=[('phi', 1)] * 2, tau_grid=np.array([0.0, 1.0]),
              spatial_grid=np.array([0.5]), use_cache=False, parallel=False, spatial_parallel=False,
              verbose=False)
    got_m, ref_m = sde_conserved_rd().build(), _tracked('reaction_diffusion_conserved_1d').build()
    with pytest.warns(UserWarning, match='approximate'):
        got = compute_cumulants(got_m, **kw)
    with pytest.warns(UserWarning, match='approximate'):
        ref = compute_cumulants(ref_m, **kw)
    assert ref['status'] == got['status'] == 'approximate'
    assert [r['reason'] for r in ref['spatial_accuracy']] == ['derivative_vertex_loop']
    assert got['spatial_accuracy'] == ref['spatial_accuracy']
    assert np.array_equal(np.asarray(got['C_tau_x']), np.asarray(ref['C_tau_x']))
    _raises_same(lambda: got_m, lambda: ref_m,
                 lambda m: compute_cumulants(m, policy={'mode': 'strict'}, **kw))
    with pytest.raises(SpatialAccuracyError):
        compute_cumulants(got_m, policy={'mode': 'strict'}, **kw)


def test_spatial_reserved_names_refused_as_by_the_spatial_builder():
    """A spatial field named x (a spatial coordinate) is refused by the spatial builder, through the bridge too."""
    s = (SDE('m', space=1).operator('lap', Laplacian()).parameter('D', default=1.0).field('x')
         .equation('x', lhs='Dt*x', rhs='D*lap(x) - x').noise('eta', _white(1), into={'x': 1}))
    with pytest.raises(ValueError, match="physical field name 'x' is reserved"):
        s.build()


def _spde():
    return (SDE('m', space=1).operator('lap', Laplacian()).parameter('D', default=1.0)
            .parameter('mu', default=1.0).field('u'))


@pytest.mark.parametrize('make,match', [
    (lambda: _spde().equation('u', lhs='Dt*u', rhs='D*lap(u) - mu*u(t - 1)')
     .noise('eta', _white(1), into={'u': 1}), 'delays'),
    (lambda: _spde().define_kernel('K', freq_image='1/(1+I*omega)')
     .equation('u', lhs='Dt*u', rhs='D*lap(u) - mu*Conv(K, u)').noise('eta', _white(1), into={'u': 1}),
     'delays'),
    (lambda: _spde().point_process('n', 'mu').equation('u', lhs='Dt*u', rhs='D*lap(u) - mu*u + n')
     .noise('eta', _white(1), into={'u': 1}), 'point_process'),
    (lambda: _spde().equation('u', lhs='Dt*u', rhs='D*lap(u) - mu*u')
     .noise('eta', Poisson('mu'), into={'u': 1}).interpretation('marcus'), 'marcus'),
    (lambda: _spde().equation('u', lhs='Dt*u', rhs='D*lap(u) - mu*u')
     .noise('eta', Poisson('mu'), into={'u': 1}), 'space-time white Gaussian'),
    (lambda: _spde().equation('u', lhs='Dt*u', rhs='D*lap(u) - mu*u')
     .noise('eta', Gaussian(var=1), into={'u': 1}), "space='white'"),
    (lambda: _spde().equation('u', lhs='Dt*u', rhs='D*lap(u) - mu*u')
     .noise('eta', _white(1), into={'u': 1}).interpretation('stratonovich'), 'stratonovich'),
    (lambda: _spde().population('E', size=2).equation('u', lhs='Dt*u', rhs='D*lap(u) - mu*u')
     .noise('eta', _white(1), into={'u': 1}), 'populations'),
    (lambda: _spde().noise('eta', Poisson('mu'), expose='n').equation('u', lhs='Dt*u', rhs='D*lap(u) - n'),
     'exposed'),
])
def test_temporal_only_features_with_a_laplacian_are_refused(make, match):
    with pytest.raises(SDEError, match=match):
        make().action_text()


def test_laplacian_without_space_and_spatial_noise_without_laplacian_are_refused():
    s = (SDE('m').operator('lap', Laplacian()).parameter('D', default=1.0).field('u')
         .equation('u', lhs='Dt*u', rhs='D*lap(u) - u').noise('eta', _white(1), into={'u': 1}))
    with pytest.raises(SDEError, match=r'give its spatial dimension, SDE\(.m., space=d\)'):
        s.build()
    t = SDE('m').field('u').drift('u', '-u').noise('eta', _white(1), into={'u': 1})
    with pytest.raises(SDEError, match="is spatial"):
        t.build()
    with pytest.raises(SDEError, match='boundary'):
        SDE('m').field('u').drift('u', '-u').boundary('periodic', length=5.0).build()
    with pytest.raises(SDEError, match=r'spatial dimension must be an integer d >= 1'):
        SDE('m', space=0)
    with pytest.raises(SDEError, match='Laplacian is not a declared operator'):
        SDE('m').parameter('D').field('u').equation('u', lhs='Dt*u', rhs='D*Laplacian*u')


def test_conserved_noise_and_nonlinear_product_style_are_refused():
    with pytest.raises(NoiseError, match='conserved noise'):
        Gaussian(var=1, space='conserved')
    with pytest.raises(NoiseError, match="use space='white'"):
        Gaussian(var=1, space='colored')
    s = (_spde().equation('u', lhs='Dt*u', rhs='D*lap*u^2 - mu*u').noise('eta', _white(1), into={'u': 1}))
    with pytest.raises(SDEError, match=r'write an operator acting on an expression in call style'):
        s.action_text()
    s = (_spde().equation('u', lhs='Dt*u', rhs='D*lap*u^2 - mu*lap(u^2)').noise('eta', _white(1), into={'u': 1}))
    with pytest.raises(SDEError, match='must multiply exactly one field linearly'):
        s.action_text()


@pytest.mark.parametrize('make', [
    lambda: _spde().equation('u', lhs='Dt*u', rhs='D*lap(u) + lap(mu) - mu*u'),
    lambda: SDE('m').operator('dt', TimeDerivative()).parameter('mu').field('x')
    .equation('x', lhs='dt(x) + dt(mu)', rhs='-mu*x'),
])
def test_operator_acting_on_no_field_is_refused(make):
    with pytest.raises(SDEError, match=r'acts on no field'):
        make()


@pytest.mark.parametrize('make', [
    lambda: _spde().define_function('g', ['v'], 'lap(v)').equation('u', lhs='Dt*u', rhs='D*lap(u) + g(u) - mu*u')
    .noise('eta', _white(1), into={'u': 1}),
    lambda: SDE('m').operator('dt', TimeDerivative()).parameter('mu', default=1.0)
    .define_function('g', ['v'], 'dt*v').field('x').equation('x', lhs='dt(x)', rhs='g(x) - mu*x')
    .noise('xi', Gaussian(var=1), into={'x': 1}),
])
def test_operator_inside_a_function_body_is_refused(make):
    """A function body is passed to the engine as written: an operator name there would be an undeclared symbol
    (``dt*v`` was silently accepted), so it is refused."""
    with pytest.raises(SDEError, match=r"its expression uses the operator '(lap|dt)'"):
        make().build()


def test_spde_has_no_temporal_simulator():
    with pytest.raises(SDEError, match='simulator is temporal'):
        sde_allen_cahn().simulate(T=1.0, dt=0.1)
