"""Higher-order equations (automatic velocity embedding), pure integrators and the operator form of paper-style lines
in the SDE front end (daedalus/api/sde.py, daedalus/api/sde_paper.py).

Linear models are checked against the exact stationary covariance from the Lyapunov equation
A S + S A^T + B = 0 and C_ab(tau) = <a(0) b(tau)> = (expm(A tau) S)_ba: tree level is exact for them.
"""
import numpy as np
import pytest
from scipy.linalg import expm, solve_continuous_lyapunov

from daedalus.api import compute_cumulants
from daedalus.api.noise import Gaussian
from daedalus.api.operators import TimeDerivative
from daedalus.api.sde import SDE, SDEError
from tests import _sde_model_catalogue as cat
from tests._sde_model_catalogue import digest

TAUS = [0.0, 0.3, 0.7, 1.5]


def _exact(A, B, i, j):
    S = solve_continuous_lyapunov(np.asarray(A, float), -np.asarray(B, float))
    return np.array([(expm(np.asarray(A, float) * t) @ S)[j, i] for t in TAUS])


def _C(m, a, b, k=2, max_ell=0):
    r = compute_cumulants(m, k=k, max_ell=max_ell, external_fields=[(a, 1), (b, 1)], tau_grid=TAUS,
                          verbose=False)
    assert r['status'] == 'ok'
    return np.real(np.asarray(r['C_tau']))


def _osc(name='Langevin oscillator'):
    m = SDE(name)
    for p, v in (('mass', 1.0), ('gam', 0.5), ('k', 2.0), ('D', 0.3)):
        m.parameter(p, default=v, domain='positive')
    return m


def test_langevin_oscillator_matches_the_lyapunov_covariance():
    m = _osc().field('x').equation('x', '(mass*Dt^2 + gam*Dt + k)*x').noise('xi', Gaussian(var='2*D'),
                                                                          into={'x': 1})
    A, B = [[0, 1], [-2.0, -0.5]], np.diag([0, 0.6])
    for a, b, i, j in (('x', 'x', 0, 0), ('xdot', 'xdot', 1, 1), ('x', 'xdot', 0, 1), ('xdot', 'x', 1, 0)):
        np.testing.assert_allclose(_C(m, a, b), _exact(A, B, i, j), rtol=1e-11, atol=1e-13)
    assert _C(m, 'x', 'x')[0] == pytest.approx(0.3 / (0.5 * 2.0), rel=1e-12)        # D/(gamma k)


def test_rlc_circuit_equipartition():
    """L q'' + R q' + q/C = Johnson noise sqrt(2 R kT) xi: <q^2> = C kT, <i^2> = kT/L, <q i> = 0."""
    m = SDE('RLC')
    for p, v in (('L', 0.5), ('R', 0.8), ('C', 2.0), ('kT', 0.3)):
        m.parameter(p, default=v, domain='positive')
    m.sde('(L*Dt^2 + R*Dt + 1/C)*q = sqrt(2*R*kT) dW/dt', )
    assert m._embedded['q']['top'] == 'qdot'
    A, B = [[0, 1], [-1.0, -1.6]], np.diag([0, 2 * 0.8 * 0.3 / 0.25])
    np.testing.assert_allclose(_C(m, 'q', 'q'), _exact(A, B, 0, 0), rtol=1e-11, atol=1e-13)
    np.testing.assert_allclose(_C(m, 'q', 'qdot'), _exact(A, B, 0, 1), rtol=1e-11, atol=1e-13)
    assert _C(m, 'q', 'q')[0] == pytest.approx(0.6, rel=1e-12)
    assert _C(m, 'qdot', 'qdot')[0] == pytest.approx(0.6, rel=1e-12)


def test_third_order_equation_and_velocity_names():
    m = SDE('third')
    for p, v in (('a2', 3.0), ('a1', 3.0), ('a0', 1.0), ('D', 0.5)):
        m.parameter(p, default=v, domain='positive')
    m.field('y').equation('y', '(Dt^3 + a2*Dt^2 + a1*Dt + a0)*y', velocity=['u', 'w']) \
        .noise('xi', Gaussian(var='2*D'), into={'y': 1})
    assert list(m._fields) == ['y', 'u', 'w'] and m._fields['w']['embedded_of'] == 'y'
    A, B = [[0, 1, 0], [0, 0, 1], [-1.0, -3.0, -3.0]], np.diag([0, 0, 1.0])
    np.testing.assert_allclose(_C(m, 'y', 'y'), _exact(A, B, 0, 0), rtol=1e-10, atol=1e-13)
    np.testing.assert_allclose(_C(m, 'y', 'w'), _exact(A, B, 0, 2), rtol=1e-10, atol=1e-13)


def test_embedding_is_the_hand_written_first_order_system():
    a = _osc().field('x').equation('x', '(mass*Dt^2 + gam*Dt + k)*x', '-0.1*x^3') \
        .noise('xi', Gaussian(var='2*D'), into={'x': 1})
    b = (_osc().field('x').field('xdot', description='dx/dt (embedded velocity)').drift('x', 'xdot')
         .equation('xdot', 'mass*Dt*xdot + gam*xdot', '-0.1*x^3 - k*x')
         .noise('xi', Gaussian(var='2*D'), into={'xdot': 1}))
    da, db = digest(a), digest(b)
    assert da['action_text'] == db['action_text'] and da['equations'] == db['equations']
    assert da['model'] == db['model']
    assert a.notes[0].startswith("equation('x'): order-2 lhs")
    # a noise declared before the equation is moved to the top derivative too, and the operator spelling counts
    c = _osc().operator('D_t', TimeDerivative()).field('x').noise('xi', Gaussian(var='2*D'), into={'x': 1}) \
        .equation('x', 'mass*D_t^2*x + gam*D_t*x + k*x', '-0.1*x^3')
    assert c._sources['xi']['couples'] == {'xdot': 1}
    assert c.equations() == a.equations()
    # one loop on a nonlinear oscillator runs and matches the hand-written system
    np.testing.assert_allclose(_C(a, 'x', 'x', max_ell=1), _C(b, 'x', 'x', max_ell=1), rtol=1e-12)


def test_pure_integrator_mean_field_is_solved_with_the_system():
    """dx = v dt (no x on the rhs) with a forced damped v: x* = F/k, v* = 0."""
    m = (_osc('integrator').parameter('F', default=0.8).field('x').field('v').drift('x', 'v')
         .equation('v', '(mass*Dt + gam)*v', 'F - k*x').noise('xi', Gaussian(var='2*D'), into={'v': 1}))
    assert any('pure integrator' in n for n in (m.action_text() and m.notes))
    r = compute_cumulants(m, k=1, max_ell=0, external_fields=[('x', 1)], tau_grid=[0.0], verbose=False)
    assert r['mf_values']['xstar'][0] == pytest.approx(0.4, abs=1e-12)
    assert r['mf_values']['vstar'][0] == pytest.approx(0.0, abs=1e-12)
    # the paper-style line of a pure integrator
    p = (_osc('integrator').parameter('F', default=0.8)
         .sde('dx = v dt\nmass*dv = (F - k*x - gam*v) dt + sqrt(2*D) dW'))
    np.testing.assert_allclose(_C(p, 'x', 'x'), _exact([[0, 1], [-2.0, -0.5]], np.diag([0, 0.6]), 0, 0),
                               rtol=1e-11, atol=1e-13)


def test_refusals():
    with pytest.raises(SDEError, match='does not depend on x or any other field'):
        SDE('rw').field('x').drift('x', '0').noise('xi', Gaussian(var=1), into={'x': 1}).action_text()
    with pytest.raises(SDEError, match='no other equation holds x'):
        _osc().field('x').equation('x', '(mass*Dt^2 + gam*Dt)*x').noise('xi', Gaussian(var=1),
                                                                      into={'x': 1}).action_text()
    with pytest.raises(SDEError, match='polynomial in the time derivative acting on x alone'):
        _osc().field('x').field('y').equation('x', 'Dt^2*x + y')
    with pytest.raises(SDEError, match="velocity name 'k' is taken"):
        _osc().field('x').equation('x', '(Dt^2 + 1)*x', velocity='k')
    with pytest.raises(SDEError, match='needs 1 velocity'):
        _osc().field('x').equation('x', '(Dt^2 + 1)*x', velocity=['a', 'b'])
    with pytest.raises(SDEError, match='first order'):
        _osc().field('x').equation('x', '(Dt + 1)*x', velocity='v')
    with pytest.raises(SDEError, match='must act on one field'):
        _osc().field('y').sde('(Dt^2 + 1)*x + y = dW/dt')
    with pytest.raises(SDEError, match='per unit time'):
        _osc().sde('(Dt + k)*x = -x dt + dW/dt')
    with pytest.raises(SDEError, match='is not c dW/dt'):
        _osc().sde('(Dt + k)*x = sqrt(D) dt/dt')


# ── the operator form of paper-style lines ───────────────────────────────────

def test_operator_form_paper_lines():
    a = _osc().sde('(mass*Dt^2 + gam*Dt + k)*x = -0.1*x^3 + sqrt(2*D) dW/dt')
    b = (_osc().field('x').equation('x', '(mass*Dt^2 + gam*Dt + k)*x', '-0.1*x^3')
         .noise('W_x', Gaussian(var=1), into={'x': 'sqrt(2*D)'}, interpretation='ito'))
    assert digest(a) == digest(b)
    # a population sum on the rhs (the drift form cannot split it)
    p = (SDE('pop', n_populations=2).parameter('w', indexed='matrix', default=[[0.1, 0.2], [0.3, 0.1]])
         .parameter('tau', default=2.0).parameter('s', default=0.5)
         .sde('(tau*Dt + 1)*v = sum(w[i,j]*v[j] for j in pop_all) + s ∘ dW_a/dt'))
    q = (SDE('pop', n_populations=2).parameter('w', indexed='matrix', default=[[0.1, 0.2], [0.3, 0.1]])
         .parameter('tau', default=2.0).parameter('s', default=0.5).field('v')
         .equation('v', '(tau*Dt + 1)*v', 'sum(w[i,j]*v[j] for j in pop_all)')
         .noise('a', Gaussian(var=1), into={'v': 's'}, interpretation='stratonovich'))
    assert digest(p) == digest(q)


@pytest.mark.parametrize('make', [
    lambda: _osc().sde('(mass*Dt^2 + gam*Dt + k)*x = (-0.1*x^3) + sqrt(2*D) dW/dt'),
    lambda: _osc().field('x').equation('x', '(mass*Dt^2 + k)*x', '-gam*x^3')
    .noise('xi', Gaussian(var='2*D'), into={'x': 1}),
    lambda: (SDE('pop', n_populations=2).parameter('w', indexed='matrix', default=[[0.1, 0.2], [0.3, 0.1]])
             .parameter('tau', default=2.0).parameter('s', default=0.5)
             .sde('(tau*Dt + 1)*v = sum(w[i,j]*v[j] for j in pop_all) + s dW/dt')),
    lambda: cat.ou_quartic(drift_form=False),
])
def test_sde_text_round_trips_the_operator_form(make):
    m = make()
    text = m.sde_text()
    again = SDE(m.name, n_populations=m._n_pop_default if not m._pops else None)
    again._params, again._param_names = list(m._params), list(m._param_names)
    again._pops = list(m._pops)
    for f, info in m._fields.items():
        if not info.get('embedded_of'):
            again.field(f, **{k: info[k] for k in ('population', 'description', 'domain', 'latex')})
    again.sde(text, noises=text.noises)
    assert again.action_text() == m.action_text() and again.equations() == m.equations()


def test_sde_text_refuses_named_velocities():
    m = _osc().field('x').equation('x', '(mass*Dt^2 + k)*x', velocity='v')
    with pytest.raises(SDEError, match='named with velocity='):
        m.sde_text()


def test_the_simulator_runs_the_embedded_system():
    m = _osc('sim osc').sde('(mass*Dt^2 + gam*Dt + k)*x = sqrt(2*D) dW/dt')
    rec = m.simulate(T=200.0, dt=0.01, n_paths=50, seed=3, burn_in=20.0)
    assert np.asarray(rec['fields']['x']).var() == pytest.approx(0.3, rel=0.1)       # D/(gamma k)
    assert np.asarray(rec['fields']['xdot']).var() == pytest.approx(0.6, rel=0.1)    # D/gamma
