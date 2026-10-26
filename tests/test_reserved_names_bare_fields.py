"""Paper-syntax increment names are reserved (no operator may be called dt, dW, dN, dW_<name>, d<field>), ``∂t`` names
the time derivative, and external fields may be bare field names (``['x', 'x']``) for fields with one unit.

Bare names must give the same run as the ``(field, 1)`` pairs, byte for byte; a bare name for a field with several
units is refused with a message asking for the unit.
"""
import os

import numpy as np
import pytest

import daedalus as dd
from api import compute_cumulants
from api.covariance import covariance_matrix
from api.noise import Gaussian
from api.operators import Laplacian, TimeDerivative
from api.sde import SDE, SDEError
from tests._sde_model_catalogue import digest


# ── R1: reserved increment names ─────────────────────────────────────────────

@pytest.mark.parametrize('name', ['dt', 'dW', 'dN', 'dW_a', 'dN_b', 'dL_g'])
def test_paper_increments_cannot_name_an_operator(name):
    with pytest.raises(SDEError) as ei:
        SDE('m').operator(name, TimeDerivative())
    msg = str(ei.value)
    assert 'increment of the paper-style syntax' in msg and 'Dt' in msg and '∂t' in msg
    with pytest.raises(SDEError, match='increment of the paper-style syntax'):
        SDE('m').operator(name, Laplacian())


def test_a_fields_increment_is_reserved_in_both_orders():
    with pytest.raises(SDEError, match="'dx' is already the fluctuation of 'x'"):
        SDE('m').field('x').operator('dx', TimeDerivative())
    with pytest.raises(SDEError, match="derived name 'dx' is a declared operator"):
        SDE('m').operator('dx', Laplacian()).field('x')


def test_other_names_and_the_unicode_time_derivative():
    SDE('m').operator('D_t', TimeDerivative())                         # still a valid rename
    SDE('m').operator('dW1', Laplacian())                              # not an increment
    a = (SDE('ou').parameter('mu', default=1.0).parameter('D', default=1.0).field('x')
         .equation('x', '∂t(x)', '-mu*x').noise('xi', Gaussian(var='2*D'), into={'x': 1}))
    b = (SDE('ou').parameter('mu', default=1.0).parameter('D', default=1.0).field('x')
         .equation('x', 'Dt(x)', '-mu*x').noise('xi', Gaussian(var='2*D'), into={'x': 1}))
    assert digest(a) == digest(b)
    c = (SDE('ou').operator('∂t', TimeDerivative()).parameter('mu', default=1.0).parameter('tau', default=2.0)
         .parameter('D', default=1.0).field('x').equation('x', '(tau*∂_t + 1)*x', '0')
         .noise('xi', Gaussian(var='2*D'), into={'x': 1}))
    d = (SDE('ou').parameter('mu', default=1.0).parameter('tau', default=2.0).parameter('D', default=1.0)
         .field('x').equation('x', '(tau*Dt + 1)*x', '0').noise('xi', Gaussian(var='2*D'), into={'x': 1}))
    assert digest(c) == digest(d) and list(c._operators) == ['Dt']


# ── R2: bare external-field names ───────────────────────────────────────────

TAUS = np.array([0.0, 0.5, 1.0])


def _ouq():
    return dd.load_model('ou_quartic')[0]


def _same(a, b):
    assert np.array_equal(np.asarray(a['C_tau']), np.asarray(b['C_tau']))
    for ell in a['C_tau_by_ell']:
        assert np.array_equal(np.asarray(a['C_tau_by_ell'][ell]), np.asarray(b['C_tau_by_ell'][ell]))


def test_compute_cumulants_takes_bare_names():
    m = _ouq()
    kw = dict(k=2, max_ell=1, tau_grid=TAUS, verbose=False)
    pair = compute_cumulants(m, external_fields=[('x', 1), ('x', 1)], **kw)
    for ext in (['x', 'x'], ['dx', 'dx'], [('dx', 1), 'x']):
        _same(compute_cumulants(m, external_fields=ext, **kw), pair)
    one = compute_cumulants(m, k=1, max_ell=1, external_fields=['x'], tau_grid=TAUS, verbose=False)
    _same(one, compute_cumulants(m, k=1, max_ell=1, external_fields=[('dx', 1)], tau_grid=TAUS, verbose=False))
    with pytest.raises(ValueError, match='a list of k legs'):
        compute_cumulants(m, k=1, max_ell=0, external_fields='x', tau_grid=TAUS, verbose=False)


def test_dd_run_and_covariance_matrix_take_bare_names():
    m = _ouq()
    cfg = dict(k=2, max_ell=0, tau_grid=TAUS, verbose=False, parallel=False)
    _same(dd.run(m, dd.Config(external_fields=['x', 'x'], **cfg)),
          dd.run(m, dd.Config(external_fields=[('x', 1), ('x', 1)], **cfg)))
    a = covariance_matrix(m, ['x'], max_ell=0, tau_grid=TAUS)
    b = covariance_matrix(m, [('x', 1)], max_ell=0, tau_grid=TAUS)
    assert np.array_equal(a['Sigma'], b['Sigma'])


def test_a_bare_name_of_a_multi_unit_field_asks_for_the_unit():
    m = dd.load_model('linear_hawkes')[0]
    with pytest.raises(ValueError, match=r"population 'pop' of 2 units.*\('n', i\)"):
        compute_cumulants(m, k=2, max_ell=0, external_fields=['n', 'n'], tau_grid=TAUS, verbose=False)
    with pytest.raises(ValueError, match='2 units'):
        covariance_matrix(m, ['n'], max_ell=0, tau_grid=TAUS)
    with pytest.raises(ValueError, match='2 units'):
        dd.run(m, dd.Config(k=2, max_ell=0, external_fields=['n', 'n'], tau_grid=TAUS, verbose=False))


def test_k1_is_the_loop_shift_and_moment_is_the_mean():
    m = (SDE('shift').parameter('mu', default=1.0, domain='positive').parameter('h', default=0.5)
         .parameter('eps', default=0.3, domain='positive').parameter('D', default=0.5, domain='positive')
         .field('x').drift('x', 'h - mu*x - eps*x^3').noise('xi', Gaussian(var='2*D'), into={'x': 1}))
    r = compute_cumulants(m, k=1, max_ell=1, external_fields=['x'], tau_grid=[0.0], verbose=False)
    assert np.real(r['C_tau_by_ell'][0][0]) == 0.0                                  # tree: no shift
    shift = float(np.real(r['C_tau'][0]))
    mom = dd.run(m, dd.Config(k=1, max_ell=1, external_fields=['x'], output='moment', verbose=False,
                              parallel=False))
    assert float(np.real(np.ravel(mom['moment'])[0])) == pytest.approx(r['mf_values']['xstar'][0] + shift,
                                                                       rel=1e-12)


def _yaml(tmp_path, ext):
    from api.config_run import load_config
    p = os.path.join(tmp_path, 'c.yaml')
    open(p, 'w').write(f'model: {ext[0]}\nquestion:\n  k: 2\n  max_ell: 0\n  external_fields: {ext[1]}\n')
    return load_config(p)


def test_yaml_takes_bare_names(tmp_path):
    from api.config_run import ConfigError
    a = _yaml(tmp_path, ('ou_quartic', '[x, x]'))[0]
    b = _yaml(tmp_path, ('ou_quartic', '[[x, 1], [x, 1]]'))[0]
    assert a['question']['external_fields'] == b['question']['external_fields']
    with pytest.raises(ConfigError, match=r'has 2 units.*\[n, i\]'):
        _yaml(tmp_path, ('linear_hawkes', '[n, n]'))


def test_the_analyzer_takes_bare_names():
    from api.analyze import Policy, Question, analyze
    rep = analyze(_ouq(), Question(k=2, max_ell=0, external_fields=('x', 'x')), Policy())
    assert not [f for f in rep.findings if f.code == 'G0_EXTERNAL_FIELD']
