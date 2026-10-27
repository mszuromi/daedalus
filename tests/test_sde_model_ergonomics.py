"""Model ergonomics of the SDE front end: delayed fields ``x(t - d)``, ``SDE.connectivity``, ``SDE.observable``
(api/observables.py) and the text form ``SDE.to_text`` / ``SDE.from_text`` (api/sde_text.py, the YAML ``sde:``
block of api/config_run.py).

Identities are checked with ``digest`` (action text, mean-field equations, notes and built model dict).  The
nonlinear observable is checked against the exact stationary density of a 1-d SDE: the one-loop mean must miss it
by O(D^2) (the tree by O(D)).
"""
import os
import textwrap
import warnings

import numpy as np
import pytest
from scipy.integrate import quad

from api import compute_cumulants
from api.config_run import ConfigError, run_config
from api.delay import Kernel
from api.noise import Gaussian, GaussianVector, Poisson
from api.observables import ObservableError
from api.operators import Laplacian, TimeDerivative
from api.sde import SDE, SDEError
from tests import _sde_model_catalogue as cat
from tests._sde_model_catalogue import digest

TAUS = [0.0, 0.5, 1.0]
#: two_field_cross away from its marginal point (all its defaults are 0.5: mu1 mu2 = J1 J2)
STABLE = {'mu1': 1.0, 'mu2': 1.5}


def _pop(**kw):
    return (SDE('pd').population('E', size=2)
            .parameter('w', default=[[0.1, 0.2], [0.3, 0.1]], indexed_by=['E', 'E'])
            .parameter('d', default=0.5, domain='positive').field('v', population='E'))


# ── delayed fields x(t - d) ──────────────────────────────────────────────────

def test_indexed_delayed_field_is_the_delay_kernel():
    a = _pop().equation('v', '(Dt + 1)*v', 'sum(w[i,j]*v[j](t - d) for j in E)') \
        .noise('xi', Gaussian(var=1), into={'v': 1})
    b = (_pop().define_kernel('delay_d', delay='d')
         .equation('v', '(Dt + 1)*v', 'sum(w[i,j]*Conv(delay_d, v[j]) for j in E)')
         .noise('xi', Gaussian(var=1), into={'v': 1}))
    assert digest(a) == digest(b)
    c = _pop().drift('v', '-v[i] + 0.3*v[ i ](t - (d))').noise('xi', Gaussian(var=1), into={'v': 1})
    assert c._eqs['v']['rhs_raw'] == '-v[i] + 0.3*Conv(delay_d, v[i])' and c._kernels['delay_d']['delay'] == 'd'


def test_point_process_rate_takes_a_delayed_field():
    def base():
        return (SDE('pp').parameter('d', default=0.5).parameter('a', default=1.0).field('x').drift('x', '-x')
                .noise('xi', Gaussian(var=1), into={'x': 1}))
    m = base()
    m.point_process('n', 'a*exp(x(t - d))')
    e = base().define_kernel('delay_d', delay='d')
    e.point_process('n', 'a*exp(Conv(delay_d, x))')
    assert digest(m) == digest(e)
    with pytest.raises(SDEError, match=r'point_process\(.n.\): cannot read the delay'):
        base().point_process('n', 'a*x(t - t)')


def test_indexed_delayed_field_switch_on_is_the_explicit_kernel_run():
    """t_min runs of a delayed model: v[j](t - d) takes the switch-on route of the explicit Conv form, bit for bit."""
    def run(m):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            return compute_cumulants(m, k=2, max_ell=0, external_fields=[('v', 1), ('v', 2)], t_min=0.0,
                                     times=np.array([[0.2, 0.2], [0.7, 0.7], [1.5, 1.0]]), parallel=False,
                                     verbose=False, use_cache=False, delay={'order': 8})
    a = run(_pop().equation('v', '(Dt + 1)*v', 'sum(w[i,j]*v[j](t - d) for j in E)')
            .noise('xi', Gaussian(var=1), into={'v': 1}))
    b = run(_pop().define_kernel('g', kernel=Kernel.delay('d'))
            .equation('v', '(Dt + 1)*v', 'sum(w[i,j]*Conv(g, v[j]) for j in E)')
            .noise('xi', Gaussian(var=1), into={'v': 1}))
    assert a['status'] == b['status'] == 'ok'
    np.testing.assert_array_equal(a['C_times'], b['C_times'])


def test_delayed_fields_write_back_as_written():
    m = SDE('paper').parameter('d', default=0.5).parameter('c', default=0.3).sde('dx = (-x + c*x(t - d)) dt + dW')
    text = m.sde_text()
    assert text == 'dx = (-x + c*x(t - d)) dt + dW'
    again = SDE('paper').parameter('d', default=0.5).parameter('c', default=0.3).sde(text)
    assert digest(again) == digest(m)


# ── connectivity ─────────────────────────────────────────────────────────────

def test_connectivity_is_the_matrix_parameter():
    W = np.array([[0.30, 0.25], [0.30, 0.35]])
    a = SDE('h', n_populations=2)
    w = a.connectivity('w', W, domain='positive')
    b = SDE('h', n_populations=2).parameter('w', indexed='matrix', default=W.tolist(), domain='positive')
    assert w._kind == 'parameter' and a._params == b._params
    p = SDE('p').population('E', size=2).population('I', size=1)
    p.connectivity('wEI', [[0.1], [0.2]], between=('E', 'I'))
    assert p._params[-1][1]['indexed_by'] == ['E', 'I'] and p._params[-1][1]['default'] == [[0.1], [0.2]]
    scipy_sparse = pytest.importorskip('scipy.sparse')
    q = SDE('q').population('E', size=2)
    q.connectivity('w', scipy_sparse.csr_matrix(W))
    assert q._params[-1][1] == dict(default=W.tolist(), indexed=False, domain=None, description='',
                                    indexed_by=['E', 'E'])


@pytest.mark.parametrize('make,match', [
    (lambda: SDE('h').connectivity('w', np.eye(2)), r'expected shape \(1, 1\).*n_populations=2'),
    (lambda: SDE('h', n_populations=2).connectivity('w', np.ones(2)), '2-d array'),
    (lambda: SDE('h', n_populations=2).connectivity('w', [[1, np.nan], [0, 1]]), 'finite'),
    (lambda: SDE('h').population('E', size=2).population('I').connectivity('w', np.eye(2)), 'between='),
    (lambda: SDE('h').population('E', size=2).connectivity('w', np.eye(2), between=('E', 'X')),
     "'X' is not a declared population"),
    (lambda: SDE('h', n_populations=2).connectivity('w', [['a', 'b'], ['c', 'd']]), 'array of numbers'),
])
def test_connectivity_refusals(make, match):
    with pytest.raises(SDEError, match=match):
        make()


# ── observables ──────────────────────────────────────────────────────────────

def _asym(D):
    return (SDE('asym').parameter('mu', default=1.0).parameter('b', default=0.5).parameter('eps', default=0.3)
            .parameter('D', default=D).field('x').drift('x', 'b - mu*x - eps*x^3')
            .noise('xi', Gaussian(var='2*D'), into={'x': 1})
            .observable('O', 'exp(x/2) + x^2'))


def _exact_mean(D, f):
    w = lambda x: np.exp((0.5 * x - 0.5 * x**2 - 0.3 * x**4 / 4) / D)
    return quad(lambda x: f(x) * w(x), -10, 10, limit=400)[0] / quad(w, -10, 10, limit=400)[0]


def test_nonlinear_observable_mean_is_exact_to_one_loop():
    """<exp(x/2) + x^2> of dx = (b - mu x - eps x^3) dt + sqrt(2D) dW: the one-loop mean misses the exact
    stationary mean by O(D^2) (halving D divides the error by 4), the saddle value by O(D)."""
    err, err_tree = [], []
    for D in (0.02, 0.01):
        r = compute_cumulants(_asym(D), k=1, max_ell=1, external_fields=['O'], tau_grid=[0.0], verbose=False)
        assert r['status'] == 'ok' and set(r['C_tau_by_ell']) == {0, 1}
        assert abs(r['C_tau_by_ell'][0][0]) == 0                      # the tree shift is 0, as for a field
        ex = _exact_mean(D, lambda x: np.exp(x / 2) + x**2)
        mean = np.real(r['observable']['mean'][0])
        assert mean == pytest.approx(r['observable']['saddle'][0] + np.real(r['C_tau'][0]), rel=1e-14)
        err.append(abs(mean - ex))
        err_tree.append(abs(r['observable']['saddle'][0] - ex))
    assert 3.5 < err[0] / err[1] < 4.5 and 1.8 < err_tree[0] / err_tree[1] < 2.2
    assert err[1] < 1e-4


def test_nonlinear_observable_tree_covariance_and_mixed_legs():
    m = _asym(0.02)
    r = compute_cumulants(m, k=2, max_ell=0, external_fields=['O', 'x'], tau_grid=TAUS, verbose=False)
    rx = compute_cumulants(m, k=2, max_ell=0, external_fields=[('x', 1), ('x', 1)], tau_grid=TAUS, verbose=False)
    g = r['observable']['linear'][0][('x', 1)]
    xs = r['mf_values']['xstar'][0]
    assert g == pytest.approx(0.5 * np.exp(xs / 2) + 2 * xs, rel=1e-12)
    np.testing.assert_allclose(r['C_tau'], g * rx['C_tau'], rtol=1e-13)
    assert r['total_C'](0.0, 0.5) == pytest.approx(g * rx['total_C'](0.0, 0.5), rel=1e-13)
    import daedalus as dd
    rd = dd.run(m, dd.Config(k=2, max_ell=0, external_fields=['O', 'x'], tau_grid=np.array(TAUS), verbose=False))
    np.testing.assert_array_equal(rd['C_tau'], r['C_tau'])


def test_linear_observable_is_exact_at_one_loop():
    m = cat.two_field_cross().observable('S', 'x + 2*y')
    kw = dict(k=2, max_ell=1, tau_grid=TAUS, parameters=STABLE, verbose=False)
    r = compute_cumulants(m, external_fields=['S', ('x', 1)], **kw)
    xx = compute_cumulants(m, external_fields=[('x', 1), ('x', 1)], **kw)
    yx = compute_cumulants(m, external_fields=[('y', 1), ('x', 1)], **kw)
    np.testing.assert_allclose(r['C_tau'], xx['C_tau'] + 2 * yx['C_tau'], rtol=1e-12, atol=1e-15)
    for e in (0, 1):
        np.testing.assert_allclose(r['C_tau_by_ell'][e], xx['C_tau_by_ell'][e] + 2 * yx['C_tau_by_ell'][e],
                                   rtol=1e-12, atol=1e-15)
    assert len(r['observable']['terms']) == 2


def test_population_observable_reads_one_unit():
    """rate = phi(v) = a[i]*v of the linear Hawkes model: at the saddle it is n* (unit by unit), and its loop
    shift is a_u times the shift of v_u (a linear observable)."""
    m = cat.linear_hawkes().observable('rate', 'phi(v)')
    for u in (1, 2):
        r = compute_cumulants(m, k=1, max_ell=1, external_fields=[('rate', u)], tau_grid=[0.0], verbose=False)
        rv = compute_cumulants(m, k=1, max_ell=1, external_fields=[('v', u)], tau_grid=[0.0], verbose=False)
        assert r['observable']['saddle'][0] == pytest.approx(r['mf_values']['nstar'][u - 1], rel=1e-12)
        np.testing.assert_allclose(r['C_tau'], 1.0 * rv['C_tau'], rtol=1e-12)
    with pytest.raises(ObservableError, match=r"read at one unit: give \('rate', u\)"):
        compute_cumulants(m, k=1, max_ell=0, external_fields=['rate'], tau_grid=[0.0], verbose=False)


def test_observable_refusals():
    m = _asym(0.02)
    with pytest.raises(ObservableError, match='nonlinear in the fields.*coincident-time'):
        compute_cumulants(m, k=2, max_ell=1, external_fields=['O', 'O'], tau_grid=[0.0], verbose=False)
    with pytest.raises(ObservableError, match='nonlinear'):
        compute_cumulants(m, k=3, max_ell=0, external_fields=['O', 'x', 'x'], verbose=False)
    with pytest.raises(ObservableError, match='spectra= is not supported'):
        compute_cumulants(m, k=2, max_ell=0, external_fields=['O', 'O'], spectra=True, verbose=False)
    with pytest.raises(SDEError, match='only linearly'):
        cat.linear_hawkes().observable('bad', 'n^2').build()
    with pytest.raises(SDEError, match='a population sum is not supported'):
        cat.linear_hawkes().observable('tot', 'sum(v[j] for j in pop)').build()
    with pytest.raises(SDEError, match="already the parameter 'mu'"):
        _asym(0.1).observable('mu', 'x')
    with pytest.raises(SDEError, match="undeclared symbol 'z'"):
        _asym(0.1).observable('Z', 'z^2').build()
    assert 'observables' not in cat.two_field_cross().build()            # unchanged without observables


# ── to_text / from_text ──────────────────────────────────────────────────────

@pytest.mark.parametrize('name', sorted(cat.CATALOGUE))
def test_every_catalogue_model_round_trips_through_the_text(name):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', DeprecationWarning)
        m = cat.CATALOGUE[name]()
        text = m.to_text()                                    # (verified: reading it back gives the same model)
    assert text.startswith('sde:\n') and digest(SDE.from_text(text)) == digest(m)


def _osc():
    m = SDE('osc')
    for p, v in (('mass', 1.0), ('gam', 0.5), ('k', 2.0), ('D', 0.3)):
        m.parameter(p, default=v, domain='positive')
    return m


ROUND_TRIP = {
    'second order, named velocity': lambda: _osc().field('x')
        .equation('x', '(mass*Dt^2 + gam*Dt + k)*x', '-0.1*x^3', velocity='v')
        .noise('xi', Gaussian(var='2*D'), into={'x': 1}),
    'noise before the equation': lambda: _osc().field('x').noise('xi', Gaussian(var='2*D'), into={'x': 1})
        .equation('x', '(mass*Dt^2 + gam*Dt + k)*x'),
    'coloured': lambda: SDE('c').parameter('a', default=1.0).parameter('tau', default=0.5).field('x')
        .drift('x', '-a*x').noise('eta', Gaussian(var=1, kernel=Kernel.exponential('tau')), into={'x': 1}),
    'coloured, named state, then a field': lambda: SDE('c').parameter('a', default=1.0).parameter('tau', default=0.5)
        .field('x').noise('eta', Gaussian(var=1, kernel=Kernel.alpha('tau')), into={'x': 'a'}, as_field='z')
        .field('y').drift('x', '-a*x').drift('y', '-y + x'),
    'vector': lambda: SDE('v').parameter('a', default=1.0).parameter('b', default=2.0).field('x').field('y')
        .drift('x', '-a*x').drift('y', '-b*y')
        .noise('eta', GaussianVector(cov=[[1.0, 0.3], [0.3, 'a']]), into={'x': [1, 0], 'y': [0, 1]}),
    'paper style, mixed interpretations': lambda: SDE('p').parameter('mu', default=1.0).parameter('s', default=0.3)
        .parameter('lam', default=0.5).sde('dx = (-mu*x) dt + s*x ∘ dW + 0.1 dN(lam)'),
    'spde': lambda: (SDE('Allen-Cahn', space=1).operator('D_t', TimeDerivative()).operator('lap', Laplacian())
                     .parameter('mu', default=1.0, domain='positive').parameter('D', default=1.0, domain='positive')
                     .parameter('lam', default=0.1, domain='positive').parameter('T', default=1.0, domain='positive')
                     .field('phi').equation('phi', lhs='D_t(phi)', rhs='D*lap(phi) - mu*phi - lam*phi^3')
                     .noise('eta', Gaussian(var='2*T', space='white'), into={'phi': 1})),
    'population delay, connectivity, observable': lambda: (
        SDE('pd').population('E', size=2).connectivity('w', np.array([[0.1, 0.2], [0.3, 0.1]]))
        .parameter('d', default=0.5, domain='positive').field('v', population='E', description='voltage')
        .equation('v', '(Dt + 1)*v', 'sum(w[i,j]*v[j](t - d) for j in E)')
        .noise('xi', Gaussian(var=1), into={'v': 1}).observable('sq', 'v^2', description='square')),
    'exposed source, functions, extra action': lambda: (
        SDE('h', n_populations=2).parameter('a', default=[1.0, 2.0], indexed=True)
        .connectivity('w', [[0.1, 0.2], [0.3, 0.1]]).define_function('phi', ['v'], 'a*v', population='pop_all')
        .noise('spikes', Poisson(rate='phi(v)'), as_field='n').field('v')
        .equation('v', '(Dt + 1)*v', 'sum(w[i,j]*n[j] for j in pop)').observable('rate', 'phi(v)')),
}


@pytest.mark.parametrize('label', list(ROUND_TRIP))
def test_round_trips(label):
    m = ROUND_TRIP[label]()
    text = m.to_text()
    assert digest(SDE.from_text(text)) == digest(m)
    assert digest(SDE.from_text(textwrap.dedent(text.split('\n', 1)[1]))) == digest(m)   # the block alone


def test_text_is_readable():
    text = ROUND_TRIP['population delay, connectivity, observable']().to_text()
    assert 'rhs: sum(w[i,j]*v[j](t - d) for j in E)' in text          # the delay as written
    assert 'delay_d' not in text and 'default: [[0.1, 0.2], [0.3, 0.1]]' in text
    assert 'var: 1' in text and 'observables:' in text


def test_an_order_the_block_does_not_record_is_refused():
    m = (_osc().field('x').equation('x', '(mass*Dt^2 + gam*Dt + k)*x').field('y').drift('y', '-y + x')
         .noise('xi', Gaussian(var=1), into={'x': 1, 'y': 1}))
    with pytest.raises(SDEError, match=r"order the sde: block does not record \(fields \['x', 'xdot', 'y'\]"):
        m.to_text()
    text = m.to_text(verify=False)                    # (unchecked: the same model up to that order)
    assert list(SDE.from_text(text)._fields) == ['x', 'y', 'xdot']


@pytest.mark.parametrize('sigma, tol', [('sigma', 1e-6), ('sigma/2 + 0.1', 1e-4)])
def test_gaussian_jitter_kernel_round_trips(sigma, tol):
    """Kernel.gaussian is written as {gaussian: {sigma, tol}} (fitted again on reading); the product kernel of
    Conv(kX*g, n) is derived again, not written."""
    m = (SDE('jit').parameter('tauX', default=0.5, domain='positive').parameter('lam', default=2.0)
         .parameter('sigma', default=0.3, domain='positive')
         .define_kernel('kX', kernel=Kernel.exponential('tauX'))
         .define_kernel('g', kernel=Kernel.gaussian(sigma, tol=tol))
         .point_process('n', 'lam').field('v').equation('v', lhs='(Dt + 1)*v', rhs='Conv(kX*g, n)'))
    text = m.to_text()
    import yaml
    kernels = yaml.safe_load(text)['sde']['kernels']
    assert kernels['g'] == {'gaussian': {'sigma': sigma, 'tol': tol}} and set(kernels) == {'kX', 'g'}
    back = SDE.from_text(text)
    assert digest(back) == digest(m) and set(back._jitter) == {'g', 'kX_g'}


@pytest.mark.parametrize('bad, msg', [('{gaussian: {tol: 1e-6}}', 'gaussian needs sigma'),
                                      ('{gaussian: s, delay: d}', 'gaussian is given alone'),
                                      ('{gaussian: {sigma: s, tl: 1}}', "unknown key 'tl'")])
def test_gaussian_kernel_key_errors(bad, msg):
    with pytest.raises(ConfigError, match=msg) as e:
        SDE.from_text(f'parameters: {{s: 0.3}}\nkernels: {{g: {bad}}}\nfields: [x]\ndrift: {{x: "-x"}}\n'
                      f'noise: {{xi: {{type: Gaussian, var: 1, into: {{x: 1}}}}}}\n')
    assert e.value.line == 2


def test_from_text_errors_name_the_line():
    with pytest.raises(ConfigError) as e:
        SDE.from_text('sde:\n  name: a\n  fields: [x]\n  drift: {x: "-x"}\n  noise:\n    xi: {type: Gausian}\n')
    assert e.value.line == 6 and "'type' must be one of" in str(e.value)
    with pytest.raises(ConfigError) as e:
        SDE.from_text('fields: [x]\ndrift: {z: "-z"}\nnoise:\n  xi: {type: Gaussian, var: 1, into: {x: 1}}\n')
    assert e.value.line == 2 and 'declare the field first' in str(e.value)


def test_the_text_is_a_daedalus_run_config(tmp_path):
    """to_text() + a question section is a config: daedalus run gives the numbers of the Python model."""
    m = cat.two_field_cross()
    path = os.path.join(tmp_path, 'c.yaml')
    with open(path, 'w') as fh:
        fh.write(m.to_text() + 'parameters: {mu1: 1.0, mu2: 1.5}\nquestion:\n  k: 2\n  max_ell: 0\n'
                 '  external_fields: [[y, 1], [x, 1]]\n  tau_grid: [0.0, 0.5, 1.0]\n'
                 'policy: {verbose: false, parallel: false}\n')
    result, resolved = run_config(path)
    direct = compute_cumulants(m, k=2, max_ell=0, external_fields=[('y', 1), ('x', 1)], tau_grid=TAUS,
                               parameters=STABLE, verbose=False, parallel=False)
    np.testing.assert_array_equal(result['C_tau'], direct['C_tau'])
    assert resolved['sde']['noise']['common'] == {'type': 'Gaussian', 'var': '2*r', 'into': {'x': 1, 'y': 1}}
