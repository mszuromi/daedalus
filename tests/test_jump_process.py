"""The reaction / transition builder (api/jump_process.py, ``dd.JumpProcess``).

  * the built model is the hand-written SDE-front-end model (identical action text, equations and
    mean-field equations): one state-dependent Poisson source of size 1/Omega per transition;
  * immigration-death: exact Poisson law (kappa_k = lambda/mu / Omega^(k-1)) from the oracle and
    from ``compute_cumulants``;
  * a dimerisation network (``2A -> B``): the one-loop cumulants converge to the exact master
    equation as Omega grows, tree error ~ 1/Omega and one-loop error ~ 1/Omega^2;
  * a two-species linear network against the Lyapunov equation (tree level is exact);
  * the mass-action conventions and the lattice rates the oracle sees;
  * refusals: negative rates, ill-formed jumps and reactions, missing values.
"""
import numpy as np
import pytest
from scipy.linalg import solve_continuous_lyapunov

import daedalus as dd
from api.jump_process import JumpProcess, JumpProcessError


# ── the built model is the hand-written model ───────────────────────────────────────────────

def _hand_written(rate_birth='Omega*lam', rate_death='Omega*mu*x'):
    from api.sde import SDE
    from api.noise import Poisson
    s = SDE('imm-death')
    s.parameter('lam', default=2.0, domain='positive')
    s.parameter('mu', default=1.0, domain='positive')
    s.parameter('Omega', default=50.0, domain='positive')
    s.field('x')
    s.drift('x', '0')
    s.noise('t0', Poisson(rate=rate_birth, size='1/Omega'), couples={'x': 1})
    s.noise('t1', Poisson(rate=rate_death, size='1/Omega'), couples={'x': -1})
    return s


def _imm_death(Om=50.0):
    return (JumpProcess('imm-death').species('x').parameter('lam', 2.0).parameter('mu', 1.0)
            .system_size(Om)
            .reaction('-> x', rate='lam').reaction('x ->', rate='mu'))


def _text(model):
    """Everything in the built model that is data (the action text, the equations, the
    declarations); the action and mean-field callables are compiled from these."""
    keys = ('action_text', 'equations', 'parameters', 'physical_fields', 'response_fields',
            'kernels', 'operators', 'functions', 'populations')
    return {k: model[k] for k in keys}


def test_reactions_build_the_hand_written_model_exactly():
    mine = _imm_death().build()
    ref = _hand_written().build()
    assert _text(mine) == _text(ref)
    # the same transitions declared directly, per unit volume or as the event rate
    t = (JumpProcess('imm-death').species('x').parameter('lam', 2.0).parameter('mu', 1.0)
         .system_size(50.0)
         .transition('lam', {'x': 1}, per_volume=True)
         .transition('Omega*mu*x', [-1]))
    assert _text(t.build()) == _text(ref)
    # an equivalent respelling of the rate text gives the same action
    assert _text(_hand_written('lam*Omega', 'Omega*x*mu').build()) == _text(ref)


def test_sde_structure_zero_drift_one_poisson_source_per_transition():
    jp = (JumpProcess('net').species('A', 'B', 'C').parameter('k', 1.0).system_size(10)
          .reaction('A + B -> C', rate='k'))
    t, = jp.transitions
    assert t['jump'] == (-1, -1, 1) and t['rate'] == 'Omega*k*A*B'
    s = jp.sde()
    assert s._sources['t0']['couples'] == {'A': -1, 'B': -1, 'C': 1}
    assert str(s._sources['t0']['source'].params['size']) == '1/Omega'
    assert dd.JumpProcess is JumpProcess and dd.JumpProcessError is JumpProcessError


def test_run_accepts_the_builder_directly():
    jp = _imm_death(40.0)
    res = dd.run(jp, dd.Config(k=2, max_ell=0, external_fields=[('dx', 1), ('dx', 1)],
                               tau_grid=np.array([0.0]), parallel=False, verbose=False))
    assert float(np.real(np.atleast_1d(res['C_tau_by_ell'][0])[0])) == pytest.approx(2.0 / 40.0, rel=1e-8)   # lam/(mu Omega)


# ── immigration-death: exact Poisson ─────────────────────────────────────────────────────

def test_immigration_death_is_poisson_in_the_oracle_and_in_the_loop_expansion():
    Om, lam, mu = 50.0, 2.0, 1.0
    jp = _imm_death(Om)
    me = jp.to_master_equation(box=(0, 250))
    assert me.h == pytest.approx(1 / Om)
    cm = me.cumulants(3)
    for k in (1, 2, 3):
        # counts are Poisson(Omega lam/mu): kappa_k(n) = Omega lam/mu, x = n/Omega
        expect = (lam / mu) * Om / Om ** k
        assert float(cm[k].ravel()[0]) == pytest.approx(expect, rel=1e-8)
    cmp = jp.compare(box=(0, 250), ks=(1, 2, 3), max_ell=1)
    assert [r['cumulant'] for r in cmp.rows] == ['kappa_1(x)', 'kappa_2(x,x)', 'kappa_3(x,x,x)']
    for r in cmp.rows:                        # linear rates: the tree level is exact
        assert r['rel_error'][0] < 1e-8 and r['rel_error'][1] < 1e-8, r
    assert 'kappa_2(x,x)' in cmp.table()


# ── dimerisation: convergence to the oracle as Omega grows ───────────────────────────────────

def _dimer():
    return (JumpProcess('dimerisation').species('A', 'B')
            .parameter('s', 2.0).parameter('k1', 1.0).parameter('k2', 0.5).system_size(25)
            .reaction('-> A', rate='s').reaction('2A -> B', rate='k1').reaction('B ->', rate='k2'))


def test_dimerisation_conventions_and_lattice_rates():
    jp = _dimer()
    assert [t['jump'] for t in jp.transitions] == [(1, 0), (-2, 1), (0, -1)]
    assert jp.transitions[1]['rate'] == 'Omega*k1*A*(A - 1/Omega)/2'
    me = jp.to_master_equation()
    X = np.array([[0.0, 0.0], [1 / 25, 0.0], [2 / 25, 0.0]])
    r = me.rates(X)
    assert r.shape == (3, 3)                               # (transition, state)
    assert r[1, 0] == 0.0 and r[1, 1] == 0.0               # no pair from 0 or 1 molecule
    assert r[2, 0] == 0.0                                  # no B to remove
    assert r[1, 2] == pytest.approx(25 * 1.0 * (2 / 25) * (1 / 25) / 2)    # n(n-1)/2 / Omega
    mac = (JumpProcess().species('A').parameter('k1', 1.0).system_size(25)
           .reaction('2A ->', rate='k1', combinatorics='macroscopic'))
    law = (JumpProcess().species('A').parameter('k1', 1.0).system_size(25)
           .reaction('2A ->', rate='k1', combinatorics='rate_law'))
    assert mac.transitions[0]['rate'] == 'Omega*k1*A^2/2'
    assert law.transitions[0]['rate'] == 'Omega*k1*A^2'
    assert (JumpProcess().species('A').system_size(5).parameter('k', 1.0)
            .reaction('3A ->', rate='k').transitions[0]['rate']
            == 'Omega*k*A*(A - 1/Omega)*(A - 2/Omega)/6')


def test_dimerisation_one_loop_converges_to_the_oracle_as_system_size_grows():
    jp = _dimer()
    errs = {}
    for Om in (25, 50, 100):
        cmp = jp.compare({'Omega': Om}, ks=(1, 2), max_ell=1)
        errs[Om] = {r['cumulant']: (r['rel_error'][0], r['rel_error'][1]) for r in cmp.rows}
    for key in ('kappa_1(A)', 'kappa_2(A,A)', 'kappa_2(A,B)', 'kappa_2(B,B)'):
        tree = [errs[Om][key][0] for Om in (25, 50, 100)]
        loop = [errs[Om][key][1] for Om in (25, 50, 100)]
        assert all(l < t / 10 for l, t in zip(loop, tree)), key
        assert tree[0] / tree[1] == pytest.approx(2.0, rel=0.15), key        # tree error ~ 1/Omega
        assert tree[1] / tree[2] == pytest.approx(2.0, rel=0.15), key
        assert loop[0] / loop[1] > 3.2 and loop[1] / loop[2] > 3.5, key      # one loop ~ 1/Omega^2
    # B is only produced and removed linearly: its mean is exact
    assert errs[25]['kappa_1(B)'][0] < 1e-9


def test_convergence_helper_and_catalyst_reaction():
    jp = _dimer()
    out = jp.convergence([25, 50], ks=(1,), max_ell=1)
    assert set(out) == {25, 50} and out[50].rows[0]['rel_error'][1] < out[25].rows[0]['rel_error'][1]
    cat = (JumpProcess().species('A', 'B', 'C').parameter('k', 1.0)
           .reaction('A + B -> A + C', rate='k'))
    assert cat.transitions[0]['jump'] == (0, -1, 1)
    assert cat.sde()._sources['t0']['couples'] == {'B': -1, 'C': 1}
    rev = (JumpProcess().species('A', 'B').parameter('kf', 1.0).parameter('kr', 2.0)
           .reaction('A <-> B', rate=('kf', 'kr'), name='iso'))
    assert [(t['name'], t['jump'], t['rate']) for t in rev.transitions] == [
        ('iso_f', (-1, 1), 'Omega*kf*A'), ('iso_r', (1, -1), 'Omega*kr*B')]


def test_schlogl_reactions_are_the_oracle_schlogl_transitions():
    Om = 30.0
    jp = (JumpProcess('schlogl').species('x').system_size(Om)
          .parameter('a', 1.0).parameter('b', 1.0).parameter('c', 1.0).parameter('d', 1.0)
          .reaction('2x -> 3x', rate='a', combinatorics='rate_law')
          .reaction('-> x', rate='c')
          .reaction('3x -> 2x', rate='b', combinatorics='rate_law')
          .reaction('x ->', rate='d'))
    ref = dd.master_equation(
        [('Om*(a*x^2)', +1), ('Om*c', +1), ('Om*(b*x^3)', -1), ('Om*d*x', -1)],
        fields=['x'], params=dict(Om=Om, a=1.0, b=1.0, c=1.0, d=1.0), h=1 / Om, box=(0, 150))
    Q = jp.to_master_equation(box=(0, 150)).generator()
    assert abs(Q - ref.generator()).max() < 1e-12


# ── two-species linear network against Lyapunov ──────────────────────────────────────────────

def test_two_species_linear_network_matches_lyapunov():
    Om = 80.0
    lam, lj, m1, k, m2 = 12.0, 3.0, 1.0, 0.7, 0.4
    jp = (JumpProcess('chain').species('x', 'y')
          .parameter('lam', lam).parameter('lj', lj).parameter('m1', m1).parameter('k', k).parameter('m2', m2)
          .system_size(Om)
          .reaction('-> x', rate='lam').reaction('-> x + y', rate='lj')
          .reaction('x ->', rate='m1').reaction('x -> y', rate='k').reaction('y ->', rate='m2'))
    A = np.array([[-(m1 + k), 0.0], [k, -m2]])
    mu = np.linalg.solve(-A, [lam + lj, lj])
    D = sum(r * np.outer(j, j) for r, j in [(lam, (1, 0)), (lj, (1, 1)), (m1 * mu[0], (-1, 0)),
                                            (k * mu[0], (-1, 1)), (m2 * mu[1], (0, -1))])
    S = solve_continuous_lyapunov(A, -D) / Om                    # covariance of the densities
    assert S[0, 1] > 0
    me = jp.to_master_equation()
    cm = me.cumulants(2)
    np.testing.assert_allclose(cm[1], mu, rtol=1e-7)
    np.testing.assert_allclose(cm[2], S, rtol=1e-5, atol=1e-9)
    cmp = jp.compare(ks=(1, 2), max_ell=1)
    assert len(cmp.rows) == 2 + 3
    for r in cmp.rows:                                           # linear: tree exact, loops vanish
        assert r['rel_error'][0] < 1e-5 and r['rel_error'][1] < 1e-5, r


# ── refusals ───────────────────────────────────────────────────────────────────────────────

def _one():
    return JumpProcess().species('x').parameter('k', 1.0).system_size(20)


def test_negative_rates_are_refused():
    with pytest.raises(JumpProcessError, match='negative rate'):
        _one().transition('x - 5', {'x': 1})
    with pytest.raises(JumpProcessError, match='negative rate'):
        _one().transition('-1', {'x': 1})
    with pytest.raises(JumpProcessError, match='negative rate'):
        JumpProcess().species('x').parameter('k', -1.0).reaction('-> x', rate='k')
    with pytest.raises(JumpProcessError, match='negative rate'):
        _one().reaction('-> x', rate=-2.0)
    # the stochastic factor x (x - 1/Omega) is zero (not negative) on the lattice: accepted
    _one().reaction('2x ->', rate='k')
    # opt-out and parameters without a default are not probed
    JumpProcess(check_rates=False).species('x').transition('x - 5', {'x': 1})
    JumpProcess().species('x').parameter('q').transition('q - x', {'x': 1})


def test_ill_formed_jumps_are_refused():
    with pytest.raises(JumpProcessError, match='integers'):
        _one().transition('k', {'x': 0.5})
    with pytest.raises(JumpProcessError, match='integers'):
        _one().transition('k', {'x': True})
    with pytest.raises(JumpProcessError, match='zero jump'):
        _one().transition('k', {'x': 0})
    with pytest.raises(JumpProcessError, match='undeclared species'):
        _one().transition('k', {'z': 1})
    with pytest.raises(JumpProcessError, match='component'):
        _one().transition('k', [1, 1])
    with pytest.raises(JumpProcessError, match='undeclared name'):
        _one().transition('q*x', {'x': 1})
    with pytest.raises(JumpProcessError, match='cannot parse'):
        _one().transition('k*(x', {'x': 1})
    with pytest.raises(JumpProcessError, match='species first'):
        JumpProcess().transition('1', {'x': 1})


def test_ill_formed_reactions_and_declarations_are_refused():
    with pytest.raises(JumpProcessError, match='no arrow'):
        _one().reaction('x', rate='k')
    with pytest.raises(JumpProcessError, match='undeclared species'):
        _one().reaction('x + y ->', rate='k')
    with pytest.raises(JumpProcessError, match='cannot read'):
        _one().reaction('x * 2 ->', rate='k')
    with pytest.raises(JumpProcessError, match='zero jump'):
        _one().reaction('x -> x', rate='k')                    # nothing changes
    with pytest.raises(JumpProcessError, match='forward, reverse'):
        _one().reaction('x <-> ', rate='k')
    with pytest.raises(JumpProcessError, match='one rate constant'):
        _one().reaction('x ->', rate=('k', 'k'))
    with pytest.raises(JumpProcessError, match='combinatorics'):
        _one().reaction('x ->', rate='k', combinatorics='exotic')
    with pytest.raises(JumpProcessError, match='already declared'):
        _one().species('x')
    with pytest.raises(JumpProcessError, match='system size'):
        _one().system_size(-3)
    with pytest.raises(JumpProcessError, match='unique'):
        _one().transition('k', {'x': 1}, name='a').transition('k', {'x': -1}, name='a')


def test_missing_values_are_reported_by_the_oracle_bridge():
    jp = JumpProcess().species('x').parameter('k').reaction('-> x', rate='k').reaction('x ->', rate=1)
    with pytest.raises(JumpProcessError, match="no value for \\['k', 'Omega'\\]"):
        jp.to_master_equation(box=(0, 10))
    with pytest.raises(JumpProcessError, match='unknown parameter'):
        jp.to_master_equation({'zz': 1.0}, box=(0, 10))
    with pytest.raises(JumpProcessError, match='at least one transition'):
        JumpProcess().species('x').sde()


def test_shot_noise_with_jump_three_matches_the_oracle_and_the_closed_form():
    Om, lam, mu = 60.0, 4.0, 1.0
    jp = (JumpProcess('shot').species('x').parameter('lam', lam).parameter('mu', mu).system_size(Om)
          .transition('lam', {'x': 3}, per_volume=True)
          .transition('mu*x', {'x': -1}, per_volume=True))
    cmp = jp.compare(box=(0, 1300), ks=(1, 2, 3), max_ell=1)
    mean = 3 * lam / mu
    k2 = (9 * lam + mu * mean) / (2 * mu) / Om                       # linear noise, exact here
    assert [r['exact'] for r in cmp.rows][:2] == [pytest.approx(mean, rel=1e-8),
                                                  pytest.approx(k2, rel=1e-8)]
    for r in cmp.rows:
        assert r['rel_error'][1] < 1e-8, r
