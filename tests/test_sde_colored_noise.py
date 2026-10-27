"""Coloured Gaussian noise (``Gaussian(var, kernel=...)``, Markov-embedded) and correlated vector noise
(``GaussianVector(cov=...)``) in the SDE front end.

Linear models are checked against the exact covariance of the embedded linear system (Lyapunov:
A S + S A^T + B = 0, C_ab(tau) = (expm(A tau) S)_ba); tree level is exact for them.
"""
import numpy as np
import pytest
from scipy.linalg import expm, solve_continuous_lyapunov
from sage.all import SR

from daedalus.api import compute_cumulants
from daedalus.api.delay import Kernel
from daedalus.api.noise import Gaussian, GaussianVector, NoiseError
from daedalus.api.sde import SDE, SDEError
from tests._sde_model_catalogue import digest

TAUS = [0.0, 0.4, 1.1]
MU, TAU, D, A_ = 1.0, 0.7, 0.4, 0.3


def _exact(A, B, i, j):
    A = np.asarray(A, float)
    S = solve_continuous_lyapunov(A, -np.asarray(B, float))
    return np.array([(expm(A * t) @ S)[j, i] for t in TAUS])


def _C(m, a, b, max_ell=0):
    r = compute_cumulants(m, k=2, max_ell=max_ell, external_fields=[(a, 1), (b, 1)], tau_grid=TAUS,
                          verbose=False)
    assert r['status'] == 'ok'
    return np.real(np.asarray(r['C_tau']))


def _model(name, kernel, coupling=1, **extra):
    m = SDE(name)
    for p, v in dict(mu=MU, tau=TAU, D=D, a=A_, **extra).items():
        m.parameter(p, default=v, domain='positive')
    m.field('x').drift('x', '-mu*x').noise('eta', Gaussian(var='2*D', kernel=kernel), into={'x': coupling})
    return m


def test_ou_driven_process_is_exact():
    m = _model('OU driven', Kernel.exponential('tau'))
    assert m._fields['eta']['colored_of'] == 'eta'
    A, B = [[-MU, 1.0], [0, -1 / TAU]], np.diag([0, 2 * D / TAU ** 2])
    np.testing.assert_allclose(_C(m, 'x', 'x'), _exact(A, B, 0, 0), rtol=1e-11, atol=1e-14)
    np.testing.assert_allclose(_C(m, 'x', 'eta'), _exact(A, B, 0, 1), rtol=1e-11, atol=1e-14)
    # the noise itself: <eta eta>(s) = var/(2 tau) exp(-|s|/tau), unit area
    np.testing.assert_allclose(_C(m, 'eta', 'eta'), 2 * D / (2 * TAU) * np.exp(-np.array(TAUS) / TAU),
                               rtol=1e-12)


def test_second_order_kernel_is_exact():
    m = _model('alpha driven', Kernel.alpha('tau'))
    assert m._embedded['eta']['fields'] == ['etadot']
    A = [[-MU, 1.0, 0], [0, 0, 1.0], [0, -1 / TAU ** 2, -2 / TAU]]
    B = np.diag([0, 0, 2 * D / TAU ** 4])
    np.testing.assert_allclose(_C(m, 'x', 'x'), _exact(A, B, 0, 0), rtol=1e-10, atol=1e-14)
    np.testing.assert_allclose(_C(m, 'x', 'eta'), _exact(A, B, 0, 1), rtol=1e-10, atol=1e-14)


def test_a_proper_kernel_adds_its_direct_white_part():
    """H = (1 + a s)/(1 + tau s) = a/tau + (1 - a/tau)/(1 + tau s): the same white noise drives x and the state."""
    m = _model('direct part', Kernel.image('(1 + I*omega*a)/(1 + I*omega*tau)'))
    assert m._sources['eta']['couples']['x'] not in (None, '0')
    b = np.array([A_ / TAU, 1 / TAU])
    A, B = [[-MU, 1 - A_ / TAU], [0, -1 / TAU]], 2 * D * np.outer(b, b)
    np.testing.assert_allclose(_C(m, 'x', 'x'), _exact(A, B, 0, 0), rtol=1e-11, atol=1e-14)


def test_coloured_mean_and_a_state_dependent_coupling():
    m = SDE('mean')
    for p, v in (('mu', MU), ('tau', TAU), ('D', D), ('m0', 0.5)):
        m.parameter(p, default=v, domain='positive')
    m.field('x').drift('x', '-mu*x').noise('eta', Gaussian(var='2*D', mean='m0',
                                                            kernel=Kernel.exponential('tau')), into={'x': 1})
    r = compute_cumulants(m, k=1, max_ell=0, external_fields=[('x', 1)], tau_grid=[0.0], verbose=False)
    assert r['mf_values']['xstar'][0] == pytest.approx(0.5, abs=1e-12)            # H(0) m0 / mu
    # c(x)*eta is the hand-written two-field model, to one loop
    col = _model('mult', Kernel.exponential('tau'), coupling='1 + x^2/10')
    hand = SDE('mult')
    for p, v in dict(mu=MU, tau=TAU, D=D, a=A_).items():
        hand.parameter(p, default=v, domain='positive')
    hand.field('x').field('eta').drift('x', '-mu*x + (1 + x^2/10)*eta') \
        .equation('eta', '(Dt*tau + 1)*eta', '0').noise('eta', Gaussian(var='2*D'), into={'eta': 1})
    np.testing.assert_allclose(_C(col, 'x', 'x', max_ell=1), _C(hand, 'x', 'x', max_ell=1), rtol=1e-12)


def test_the_coloured_field_can_be_named_and_used():
    m = SDE('named')
    for p, v in (('mu', MU), ('tau', TAU), ('D', D)):
        m.parameter(p, default=v, domain='positive')
    m.field('x').drift('x', '-mu*x')
    zeta = m.noise('eta', Gaussian(var='2*D', kernel=Kernel.exponential('tau')), into={'x': 1}, as_field='zeta')
    assert str(zeta) == 'zeta' and 'zeta' in m._fields and m._sources['eta']['couples'] == {'zeta': 1}
    tex = m.latex()['sde']
    assert r'\zeta\,\mathrm{d}t' in tex and r'\dot{W}_{\eta}' in tex


def test_correlated_vector_noise_is_exact():
    v = SDE('vector')
    for p, val in (('mu', 1.0), ('nu', 2.0), ('rho', 0.6)):
        v.parameter(p, default=val)
    v.field('x').field('y').drift('x', '-mu*x').drift('y', '-nu*y')
    v.noise('xi', GaussianVector(cov=[[1, 'rho'], ['rho', 2]]), into={'x': [1, 0], 'y': ['1/2', 1]})
    L = np.array([[1, 0], [0.5, 1]])
    A, B = np.diag([-1.0, -2.0]), L @ np.array([[1, 0.6], [0.6, 2]]) @ L.T
    for a, b, i, j in (('x', 'x', 0, 0), ('x', 'y', 0, 1), ('y', 'x', 1, 0), ('y', 'y', 1, 1)):
        np.testing.assert_allclose(_C(v, a, b), _exact(A, B, i, j), rtol=1e-11, atol=1e-14)
    rec = v.simulate(T=200, dt=0.01, n_paths=40, seed=2, burn_in=10)
    X, Y = (np.asarray(rec['fields'][f]).ravel() for f in 'xy')
    S = solve_continuous_lyapunov(A, -B)
    assert np.cov(X, Y)[0, 1] == pytest.approx(S[0, 1], rel=0.1)
    tex = v.latex()
    assert r'\mathrm{d}W_{\xi,2}' in tex['sde'] and r'\begin{pmatrix}' in tex['noise']


def test_vector_noise_stratonovich_correction():
    """dx = -mu x dt + x o dW_1, dy = ... + o dW_2 with <dW_1 dW_2> = rho dt: drift of x += cov_11 x / 2."""
    v = SDE('vs')
    for p, val in (('mu', 1.0), ('nu', 2.0), ('rho', 0.6)):
        v.parameter(p, default=val)
    v.field('x').field('y').drift('x', '-mu*x').drift('y', '-nu*y')
    v.noise('xi', GaussianVector(cov=[[1, 'rho'], ['rho', 2]]), into={'x': ['x', 0], 'y': ['y', 1]},
            interpretation='stratonovich')
    corr = {(c['field'], c['text']) for c in v.check(verbose=False)['corrections']}
    # x: 1/2 (cov_11 x); y: 1/2 sum_l cov_1l b_yl d_y b_y1 = 1/2 (cov_11 y + cov_12)
    assert corr == {('x', '(1/2)*x[i]'), ('y', '(1/2)*rho + (1/2)*y[i]')}


def test_white_gaussians_are_unchanged():
    m = SDE('w').parameter('mu', default=1.0).field('x').drift('x', '-mu*x') \
        .noise('xi', Gaussian(var=2), into={'x': 1})
    assert repr(m._sources['xi']['source']) == "Gaussian(var='2', mean='0')"
    assert 'colored' not in m._sources['xi'] and m.action_text() == (
        'sum(\n    xt[i]*(Dt*x[i] - (-mu*x[i]))\n    - (xt[i]^2)\n    for i in pop)')


@pytest.mark.parametrize('make,exc,match', [
    (lambda: Gaussian(var=1, kernel=Kernel.delay('d')), NoiseError, 'no Markov embedding'),
    (lambda: Gaussian(var=1, kernel='exp'), NoiseError, 'daedalus.api.delay.Kernel'),
    (lambda: _model('b', Kernel.image('exp(-I*omega*a)')), SDEError, 'not rational'),
    (lambda: _model('b', Kernel.image('1 + I*omega*a')), SDEError, 'improper'),
    (lambda: _model('b', Kernel.image('2')), SDEError, 'white noise; drop kernel='),
    (lambda: SDE('b').field('x').noise('n', Gaussian(var='x^2', kernel=Kernel.exponential(1)), into={'x': 1}),
     SDEError, 'var must be free of the fields'),
    (lambda: SDE('b').field('x').noise('n', GaussianVector(cov=[[1, 0], [0, 1]]), into={'x': [1]}),
     SDEError, 'one coefficient'),
    (lambda: SDE('b').field('x').noise('n', Gaussian(var=1), into={'x': [1, 0]}), SDEError, 'needs a GaussianVector'),
    (lambda: GaussianVector(cov=[[1, 2], [0, 1]]), NoiseError, 'symmetric'),
    (lambda: GaussianVector(cov=[[1, 2], [2, 1]]), NoiseError, 'positive semidefinite'),
    (lambda: SDE('b').field('x').drift('x', '-x').noise('n', GaussianVector(cov=[['x', 0], [0, 1]]),
                                                        into={'x': [1, 0]}).action_text(),
     SDEError, 'cov must be free of the fields'),
])
def test_refusals(make, exc, match):
    with pytest.raises(exc, match=match):
        make()


def test_sde_text_refuses_coloured_and_vector_sources():
    with pytest.raises(SDEError, match='coloured or a GaussianVector'):
        _model('t', Kernel.exponential('tau')).sde_text()


def test_config_takes_a_gaussian_vector(tmp_path):
    import os
    from daedalus.api.config_run import load_config
    from tests import _sde_model_catalogue as cat
    text = ('sde:\n  name: V\n  parameters: {mu: 1.0, nu: 2.0, rho: 0.6}\n  fields: [x, y]\n'
            '  drift: {x: "-mu*x", y: "-nu*y"}\n  noise:\n'
            '    xi: {type: GaussianVector, cov: [[1, rho], [rho, 2]], into: {x: [1, 0], y: [0, 1]}}\n'
            'question: {k: 2, max_ell: 0}\n')
    p = os.path.join(tmp_path, 'c.yaml')
    open(p, 'w').write(text)
    model = load_config(p)[1]
    py = (SDE('V').parameter('mu', default=1.0).parameter('nu', default=2.0).parameter('rho', default=0.6)
          .field('x').field('y').drift('x', '-mu*x').drift('y', '-nu*y')
          .noise('xi', GaussianVector(cov=[[1, 'rho'], ['rho', 2]]), into={'x': [1, 0], 'y': [0, 1]}))
    assert cat._canon(model) == cat._canon(py.build())
