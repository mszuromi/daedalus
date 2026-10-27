"""
Point-process front end: ``SDE.point_process('n', rate=...)``.

Each line contributes its own action term: ``.point_process('n', rate=lam)``
gives ``nt*n - (exp(nt) - 1)*lam`` and the mean field ``n* = lam*`` (kernels at
omega = 0).  Rates may filter ANY field, the spike train itself included
(spike-only Hawkes), through ``Conv``, with population sums, declared
functions and products of kernels (convolutions).

A: the spike-only exponential Hawkes process written with ``.point_process``
is the hand-written builder model (``tests/test_conv_leg_operator_hawkes.py``)
bit for bit: action, saddle, tree and one loop.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(__file__))

from sage.all import SR                                             # noqa: E402

from daedalus.api import compute_cumulants                                   # noqa: E402
from daedalus.api.delay import Kernel                                        # noqa: E402
from daedalus.api.sde import SDE, SDEError                                   # noqa: E402
from daedalus.api.noise import Gaussian                                      # noqa: E402
from test_conv_leg_operator_hawkes import (ASYM, SYM, TAU,          # noqa: E402
                                           spike_only_form, voltage_form)

R_TIME = '(exp(-t/tau[i]) - exp(-t/taug[i,j]))/(tau[i] - taug[i,j])*heaviside(t)'
R_FREQ = '1/((1 + I*omega*tau[i])*(1 + I*omega*taug[i,j]))'
RATE = 'a[i]*exp(Em[i] + sum(w[i,j]*Conv(r[i,j], n[j]) for j in E))'


def _common(s, n_pop, p):
    return (s.population('E', size=n_pop)
            .parameter('Em', default=p['Em'][:n_pop], indexed_by=['E'])
            .parameter('tau', default=p['tau'][:n_pop], indexed_by=['E'], domain='positive')
            .parameter('taug', default=[r[:n_pop] for r in p['taug'][:n_pop]],
                       indexed_by=['E', 'E'], domain='positive')
            .parameter('a', default=p['a'][:n_pop], indexed_by=['E'], domain='positive')
            .parameter('w', default=[r[:n_pop] for r in p['w'][:n_pop]],
                       indexed_by=['E', 'E']))


def pp_spike_only(n_pop, p=ASYM, name=None):
    s = _common(SDE(name or f'Exp Hawkes spike-only test {n_pop} both'), n_pop, p)
    return (s.define_kernel('r', time_expr=R_TIME, freq_image=R_FREQ, latex_name='r',
                            indexed_by=['E', 'E'])
            .point_process('n', RATE, population='E'))


def pp_composed(n_pop, p=ASYM):
    """The same model with r written as the convolution h * g."""
    s = _common(SDE(f'Exp Hawkes composed {n_pop}'), n_pop, p)
    return (s.define_kernel('h', kernel=Kernel.exponential('tau[i]'), indexed_by=['E'])
            .define_kernel('g', kernel=Kernel.exponential('taug[i,j]'), indexed_by=['E', 'E'])
            .point_process('n', 'a*exp(Em + sum(w[i,j]*Conv(h[i]*g[i,j], n[j]) for j in E))',
                           population='E'))


def _run(m, k, max_ell, ext, tau, **kw):
    return compute_cumulants(m, k=k, max_ell=max_ell, external_fields=ext, tau_grid=tau,
                             parallel=False, verbose=False, use_cache=False, **kw)


def _identical(r1, r2, max_ell, status='ok'):
    assert r1['status'] == r2['status'] == status
    assert repr(r1['mf_values']) == repr(r2['mf_values'])
    for ell in range(max_ell + 1):
        a = np.asarray(r1['C_tau_by_ell'][ell])
        b = np.asarray(r2['C_tau_by_ell'][ell])
        assert np.array_equal(a, b), (ell, np.max(np.abs(a - b)))


# ── derivation ────────────────────────────────────────────────────────

def test_action_and_saddle_text():
    s = pp_spike_only(1)
    act = s.action_text()
    assert 'nt[i]*n[i] - (exp(nt[i]) - 1)*(a[i]*exp(Em[i] + sum(w[i,j]*Conv(r[i,j], n[j]) ' \
           'for j in E)))' in act
    (eq,) = s.equations()
    assert eq['lhs'] == 'n[i]' and 'Conv' not in eq['rhs']
    assert s._star_equations(s._derive()['equations'])[0][0] == 'nstar'


def test_spike_only_one_neuron_bit_identical_to_builder():
    mp, mb = pp_spike_only(1).build(), spike_only_form(1)
    assert bool((SR(mp['action']) - SR(mb['action'])).is_zero()) \
        if not callable(mp['action']) else True
    ext = [('n', 1), ('n', 1)]
    _run(mb, 2, 1, ext, TAU)                                    # warm-up
    _identical(_run(mp, 2, 1, ext, TAU), _run(mb, 2, 1, ext, TAU), 1)


def test_spike_only_one_neuron_matches_voltage_form():
    v = _run(voltage_form(1), 2, 1, [('n', 1), ('n', 1)], TAU)
    s = _run(pp_spike_only(1).build(), 2, 1, [('n', 1), ('n', 1)], TAU)
    for ell in (0, 1):
        np.testing.assert_allclose(np.asarray(s['C_tau_by_ell'][ell]),
                                   np.asarray(v['C_tau_by_ell'][ell]), rtol=1e-11, atol=0)


def test_kernel_product_is_the_convolution():
    """Conv(h[i]*g[i,j], n[j]) = Conv(r[i,j], n[j]) with r = h * g."""
    s = pp_composed(1)
    s._derive()
    k = s._kernels['h_g']
    assert k['indexed_by'] == ['E', 'E'] and k['delay'] is None
    assert 'tau[i]' in k['freq_image'] and 'taug[i,j]' in k['freq_image']
    a = _run(s.build(), 2, 1, [('n', 1), ('n', 1)], TAU)
    b = _run(pp_spike_only(1, name='Exp Hawkes spike-only composed ref').build(), 2, 1,
             [('n', 1), ('n', 1)], TAU)
    assert repr(a['mf_values']) == repr(b['mf_values'])
    for ell in (0, 1):
        np.testing.assert_allclose(np.asarray(a['C_tau_by_ell'][ell]),
                                   np.asarray(b['C_tau_by_ell'][ell]), rtol=1e-11, atol=1e-17)


def test_kernel_alpha_and_delay_composition():
    k = Kernel.delay('d') * Kernel.alpha('taug[i,j]')
    assert k.freq_image == '1/(1 + I*omega*taug[i,j])^2' and k.delay == 'd'
    assert Kernel.alpha('2*s').freq_image == '1/(1 + I*omega*(2*s))^2'
    s = (SDE('compose delays').population('E', size=2)
         .parameter('d', default=1.0, domain='positive')
         .parameter('tau', default=1.0, domain='positive')
         .define_kernel('h', kernel=Kernel.delay('d') * Kernel.exponential('tau'))
         .define_kernel('g', kernel=Kernel.delay('d') * Kernel.alpha('tau'))
         .point_process('n', 'exp(-1 + 0.2*sum(Conv(h*g, n[j]) for j in E))', population='E'))
    s._derive()
    assert s._kernels['h_g']['delay'] == '(d) + (d)'
    assert s._kernels['h_g']['indexed_by'] is None


def test_mixed_model_with_drift_and_gaussian_noise_builds():
    s = (SDE('pp mixed').population('E', size=2)
         .parameter('a', default=1.0, domain='positive')
         .parameter('Em', default=-1.5).parameter('tau', default=2.0, domain='positive')
         .parameter('D', default=0.1, domain='positive')
         .parameter('w', default=[[0.0, 0.3], [0.3, 0.0]], indexed_by=['E', 'E'])
         .define_kernel('g', kernel=Kernel.exponential('tau'))
         .define_function('phi', args=['v'], expression='a*exp(v)')
         .point_process('n', 'phi(v)', population='E')
         .field('v', population='E')
         .equation('v', lhs='(tau*Dt + 1)*v', rhs='Em + sum(w[i,j]*Conv(g, n[j]) for j in E)')
         .noise('xi', Gaussian(var='2*D'), couples={'v': 1}))
    act = s.action_text()
    assert 'nt[i]*n[i] - (exp(nt[i]) - 1)*(phi[i](v[i]))' in act and 'D*vt[i]^2' in act
    assert [e for e, _ in s._star_equations(s._derive()['equations'])] == ['nstar', 'vstar']
    r = _run(s.build(), 2, 0, [('n', 1), ('n', 2)], np.array([0.5]))
    assert r['status'] == 'ok'


def test_errors_are_named():
    base = lambda: (SDE('pp errors').parameter('a', default=1.0)
                    .define_kernel('g', kernel=Kernel.exponential('a')))
    with pytest.raises(SDEError, match='outside Conv'):
        base().point_process('n', 'a*exp(n)').action_text()
    with pytest.raises(SDEError, match='undeclared'):
        base().point_process('n', 'a*exp(Conv(q, n))').action_text()
    with pytest.raises(SDEError, match='time t'):
        base().point_process('n', 'a*t')
    with pytest.raises(SDEError, match='declared kernel'):
        base().point_process('n', 'a*exp(Conv(g*a, n))').action_text()
    with pytest.raises(SDEError, match='reserved'):
        base().point_process('e', 'a')
    with pytest.raises(SDEError, match='stratonovich|Stratonovich|non-Gaussian'):
        base().point_process('n', 'a').interpretation('stratonovich').action_text()


def test_marked_size():
    s = SDE('pp marks').parameter('a', default=0.5).parameter('c', default=2.0) \
        .point_process('n', 'a', jump_size='c')
    assert '(exp(c*nt[i]) - 1)*a' in s.action_text()
    assert s.equations()[0]['rhs'].replace(' ', '') in ('c*a', 'a*c')


# ── simulator ─────────────────────────────────────────────────────────

def test_simulator_poisson_rate_and_bias_free_constant_rate():
    from daedalus.api.pp_simulate import mean_rate, cross_covariance
    s = SDE('pp const').parameter('a', default=2.0).point_process('n', 'a')
    rec = s.simulate(T=200, dt=0.01, n_paths=200, burn_in=1.0, bin=0.1, seed=3)
    m, se = mean_rate(rec, 'n', 0)
    assert abs(m - 2.0) < 4 * se
    cov, cse = cross_covariance(rec, ('n', 0), ('n', 0), [1, 3, 10])
    assert np.all(np.abs(cov) < 4 * cse + 1e-12)


def _theory_1n(tau):
    s = pp_spike_only(1)
    r2 = _run(s.build(), 2, 1, [('n', 1), ('n', 1)], tau)
    r1 = _run(s.build(), 1, 1, [('n', 1)], np.array([0.0]))
    rate = r1['mf_values']['nstar'][0] + complex(r1['C_tau_by_ell'][1][0]).real
    C = np.real(np.asarray(r2['C_tau_by_ell'][0]) + np.asarray(r2['C_tau_by_ell'][1]))
    return rate, C


def test_simulator_one_neuron_hawkes_vs_theory_fast():
    """CI-fast: rate and autocovariance of the 1-neuron spike-only Hawkes vs
    tree + one loop, within 4 standard errors (+ the O(dt) bias budget)."""
    from daedalus.api.pp_simulate import mean_rate, cross_covariance
    lags = np.array([4, 8, 16])
    rate, C = _theory_1n(lags * 0.25)
    rec = pp_spike_only(1).simulate(T=400, dt=0.01, n_paths=400, burn_in=40, bin=0.25, seed=11)
    m, se = mean_rate(rec, 'n', 0)
    assert abs(m - rate) < 4 * se + 0.01 * rate
    cov, cse = cross_covariance(rec, ('n', 0), ('n', 0), lags)
    assert np.all(np.abs(cov - C) < 4 * cse + 0.02 * np.max(np.abs(C)))


@pytest.mark.slow
def test_simulator_two_neuron_hawkes_cross_covariance_vs_theory():
    from daedalus.api.pp_simulate import mean_rate, cross_covariance
    lags = np.array([-8, -2, 2, 8])
    s = pp_spike_only(2, p=SYM, name='Exp Hawkes spike-only sym 2')
    r = _run(s.build(), 2, 1, [('n', 1), ('n', 2)], lags * 0.25)
    C = np.real(np.asarray(r['C_tau_by_ell'][0]) + np.asarray(r['C_tau_by_ell'][1]))
    rec = s.simulate(T=2000, dt=0.005, n_paths=1000, burn_in=40, bin=0.25, seed=5)
    cov, cse = cross_covariance(rec, ('n', 0), ('n', 1), lags)
    assert np.all(np.abs(cov - C) < 4 * cse + 0.01 * np.max(np.abs(C)))


@pytest.mark.slow
def test_spike_only_two_neuron_bit_identical_to_builder():
    """Symmetric rates (the asymmetric spike-only one loop exceeds several GB
    at set-up; see test_conv_leg_operator_hawkes)."""
    ext = [('n', 1), ('n', 2)]
    mb = spike_only_form(2, p=SYM)
    mp = pp_spike_only(2, p=SYM).build()
    _run(mb, 2, 1, ext, TAU)                                    # warm-up
    _identical(_run(mp, 2, 1, ext, TAU), _run(mb, 2, 1, ext, TAU), 1)


# ── B: delayed alpha synapses ─────────────────────────────────────────

def _delay_builder():
    """Hand-written voltage form: n_i ~ PP(a exp(v_i)), tau v_i' = -(v_i - Em)
    + sum_j w_ij (g * n_j), g the alpha function delayed by d."""
    from daedalus.api.model import TemporalModelBuilder
    b = (TemporalModelBuilder('Hawkes delayed alpha')
         .population('E', size=2).physical_field('n', population='E')
         .physical_field('v', population='E')
         .parameter('Em', default=[-1.5, -1.5], indexed_by=['E'])
         .parameter('tau', default=[2.0, 2.0], indexed_by=['E'], domain='positive')
         .parameter('taug', default=[[1.0, 1.0], [1.0, 1.0]], indexed_by=['E', 'E'],
                    domain='positive')
         .parameter('a', default=[1.0, 1.0], indexed_by=['E'], domain='positive')
         .parameter('w', default=[[0.0, 0.5], [0.5, 0.0]], indexed_by=['E', 'E'])
         .parameter('d', default=1.5, domain='positive'))
    return (b.define_function('phi', args=['v'], expression='a[i]*exp(v)', population='E')
            .define_kernel('g', freq_image='1/(1 + I*omega*taug[i,j])^2', indexed_by=['E', 'E'],
                           delay='d')
            .set_action_text('sum( nt[i]*n[i] - (exp(nt[i])-1)*phi[i](v[i]) + vt[i]*((tau[i]*Dt + 1)'
                             '*v[i] - Em[i] - sum(w[i,j]*g[i,j]*n[j] for j in E)) for i in E)')
            .set_mf_equation('vstar', '(Em[i] + sum(w[i, j]*g[i,j]*nstar[j] for j in E))')
            .set_mf_equation('nstar', 'phi[i](vstar[i])').build())


def _delay_pp(form, n_pop=2, w=((0.0, 0.5), (0.5, 0.0))):
    s = (SDE('Hawkes delayed alpha' + ('' if form == 'voltage' else f' spike-only {n_pop}'))
         .population('E', size=n_pop)
         .parameter('Em', default=[-1.5] * n_pop, indexed_by=['E'])
         .parameter('tau', default=[2.0] * n_pop, indexed_by=['E'], domain='positive')
         .parameter('taug', default=[[1.0] * n_pop] * n_pop, indexed_by=['E', 'E'],
                    domain='positive')
         .parameter('a', default=[1.0] * n_pop, indexed_by=['E'], domain='positive')
         .parameter('w', default=[list(r[:n_pop]) for r in w[:n_pop]], indexed_by=['E', 'E'])
         .parameter('d', default=1.5, domain='positive')
         .define_kernel('g', kernel=Kernel.delay('d') * Kernel.alpha('taug[i,j]'),
                        indexed_by=['E', 'E']))
    if form == 'voltage':
        return (s.point_process('n', 'phi(v)', population='E').field('v', population='E')
                .define_function('phi', args=['v'], expression='a[i]*exp(v)', population='E')
                .equation('v', lhs='(tau*Dt + 1)*v',
                          rhs='Em + sum(w[i,j]*Conv(g[i,j], n[j]) for j in E)'))
    return (s.define_kernel('h', kernel=Kernel.exponential('tau[i]'), indexed_by=['E'])
            .point_process('n', 'a*exp(Em + sum(w[i,j]*Conv(h[i]*g[i,j], n[j]) for j in E))',
                           population='E'))


@pytest.mark.slow
def test_delayed_alpha_network_voltage_form_bit_identical_to_builder():
    """~45 min on a loaded machine: the 2-neuron delayed network's one loop (Pade loops), twice."""
    ext, tau = [('n', 1), ('n', 2)], np.array([-3.0, 0.5, 3.0])
    _run(_delay_builder(), 2, 0, ext, tau)                      # warm-up
    # the automatic Pade order stops at n_max = 16 with an estimated 1.4e-5 against the 1e-8 target, which
    # #50 marks status 'target_missed' (was 'ok'); both forms must agree on that too.  No wall-time cap: the
    # order-16 step is predicted near the default 900 s, so on a slow runner one form stopped at n = 8
    # and the other went on to 16 (different orders, not different arithmetic)
    nocap = dict(delay={'wall_max': float('inf')})
    r1 = _run(_delay_pp('voltage').build(), 2, 1, ext, tau, **nocap)
    r2 = _run(_delay_builder(), 2, 1, ext, tau, **nocap)
    _identical(r1, r2, 1, status='target_missed')
    assert r1['delay']['n'] == r2['delay']['n'] == 16
    assert r1['delay']['est_rel_error'] == r2['delay']['est_rel_error'] < 1e-4


@pytest.mark.slow
def test_delayed_spike_only_equals_voltage_form():
    """~2.5 min: a delayed alpha kernel inside the rate's exponential
    (spike-only form, h * g composed in Conv) equals the voltage form at the
    same Pade order of the loops (tree exact)."""
    ext, tau = [('n', 1), ('n', 1)], np.array([-3.0, 0.5, 3.0])
    out = []
    for form in ('voltage', 'spike'):
        r = compute_cumulants(_delay_pp(form, n_pop=1, w=((0.5,),)).build(), k=2, max_ell=1,
                              external_fields=ext, tau_grid=tau, parallel=False, verbose=False,
                              use_cache=False, delay={'order': 8})
        assert r['status'] == 'ok'
        out.append(r)
    for ell in (0, 1):
        a = np.asarray(out[0]['C_tau_by_ell'][ell])
        b = np.asarray(out[1]['C_tau_by_ell'][ell])
        np.testing.assert_allclose(b, a, rtol=1e-12, atol=0)


# ── review regressions ────────────────────────────────────────────────

def test_kernel_product_with_swapped_indices():
    """Conv(h[j]*g[i,j], n[j]): the composite keeps i first (h_g[i,j] with
    h's index on j); a transposed call of one kernel collapses transposed."""
    s = (SDE('pp swap').population('E', size=2)
         .parameter('A', default=[[1.0, 2.0], [3.0, 4.0]], indexed_by=['E', 'E'])
         .parameter('tau', default=[1.0, 3.0], indexed_by=['E'], domain='positive')
         .define_kernel('h', freq_image='1/(1+I*omega*tau[i])', indexed_by=['E'])
         .define_kernel('g', freq_image='A[i,j]/(1+I*omega*tau[j])', indexed_by=['E', 'E'])
         .point_process('n', 'exp(-2 + 0.1*sum(Conv(h[j]*g[i,j], n[j]) + Conv(g[j,i], n[j]) '
                             'for j in E))', population='E'))
    act = s.action_text()
    assert 'Conv(h_g[i,j], n[j])' in act
    assert s._kernels['h_g']['freq_image'] == '(1/(1+I*omega*tau[j]))*(A[i,j]/(1+I*omega*tau[j]))'
    rhs = s.equations()[0]['rhs'].replace(' ', '')
    assert '((A[i,j]))*(n[j])' in rhs and '((A[j,i]))*(n[j])' in rhs


def test_simulator_shared_gaussian_noise_and_refusals():
    from daedalus.api.pp_simulate import PPSimulationError
    s = (SDE('pp corr').parameter('D', default=0.5).field('x').field('y')
         .equation('x', lhs='Dt*x + x', rhs='0').equation('y', lhs='(2*Dt + 1)*y', rhs='0')
         .noise('xi', Gaussian(var='2*D'), couples={'x': 1, 'y': 1}))
    rec = s.simulate(T=100, dt=0.01, n_paths=200, burn_in=10, bin=0.1, seed=1)
    x, y = rec['fields']['x'][:, 0, :], rec['fields']['y'][:, 0, :]
    cov = np.mean(x * y) - x.mean() * y.mean()
    assert abs(cov - 1 / 3) < 0.03                              # exact 1/3 (one shared dW)
    bad = (SDE('pp conv expr').field('v').define_kernel('K', freq_image='1/(1+I*omega)')
           .equation('v', lhs='Dt*v + v', rhs='0').noise('xi', Gaussian(var='1'), couples={'v': 1})
           .point_process('n', 'exp(-1 + Conv(K, 2*v))'))
    with pytest.raises(PPSimulationError, match='field element'):
        bad.simulate(T=1, dt=0.01, n_paths=2)
    marks = SDE('pp marks sim').parameter('a', default=2.0).point_process('n', 'a', jump_size='0.5')
    rec = marks.simulate(T=50, dt=0.01, n_paths=50, burn_in=1, bin=0.1, seed=2)
    assert abs(rec['counts']['n'].sum() / (0.5 * 2.0 * rec['T'] * 50) - 1) < 0.05
