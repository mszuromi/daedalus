"""
Convolutional leg operators: the exponential Hawkes process written with the
voltages integrated out agrees with its voltage form at every loop order.

Voltage form:  n_i ~ PP(a_i exp(v_i)),  tau_i v_i' = -(v_i - Em_i) + sum_j w_ij
(g_ij * n_j),  g_ij(t) = e^{-t/taug_ij}/taug_ij.
Spike-only form:  n_i ~ PP(a_i exp(Em_i + sum_j w_ij (r_ij * n_j))),  r_ij =
H_i * g_ij,  r_ij(t) = (e^{-t/tau_i} - e^{-t/taug_ij}) / (tau_i - taug_ij):
every physical leg of every interaction vertex carries the two-exponential
kernel r as a leg operator (``ConvVertexType``), on duplicated legs too.

Phase J folds a kernel that is a finite exponential sum into the propagator
of the line it sits on, ``(r * G)``, again a pole sum (no extra time
variable); the propagator's rounding-level residues at the kernel's own rates
(poles of r-hat, which are zeros of G) are dropped as exact zeros.  Before,
the spike-only one loop did not finish (more than 12 minutes, 2 GB, still
running) and on a warm cache its diagrams were dropped.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from api import compute_cumulants                                   # noqa: E402
from api.model import TemporalModelBuilder                          # noqa: E402
import engine.integration.time_domain.final_integral as FI          # noqa: E402

TAU = np.array([-3.0, -0.25, 0.25, 1.0, 6.0])


#: asymmetric two-neuron parameters (6 distinct rates) and the symmetric set
ASYM = dict(Em=[-1.5, -1.4], tau=[2.0, 2.5], taug=[[1.0, 0.8], [0.9, 1.2]],
            a=[1.0, 0.9], w=[[0.4, 0.3], [0.35, 0.25]])
SYM = dict(Em=[-1.5, -1.5], tau=[2.0, 2.0], taug=[[1.0, 1.0], [1.0, 1.0]],
           a=[1.0, 1.0], w=[[0.4, 0.4], [0.4, 0.4]])


def _params(b, n_pop, p):
    b.population('E', size=n_pop)
    b.physical_field('n', population='E')
    b.parameter('Em', default=p['Em'][:n_pop], indexed_by=['E'])
    b.parameter('tau', default=p['tau'][:n_pop], indexed_by=['E'],
                domain='positive')
    b.parameter('taug', default=[row[:n_pop] for row in p['taug'][:n_pop]],
                indexed_by=['E', 'E'], domain='positive')
    b.parameter('a', default=p['a'][:n_pop], indexed_by=['E'],
                domain='positive')
    b.parameter('w', default=[row[:n_pop] for row in p['w'][:n_pop]],
                indexed_by=['E', 'E'])
    return b


def voltage_form(n_pop, p=ASYM):
    b = _params(TemporalModelBuilder(f'Exp Hawkes voltage test {n_pop}'),
                n_pop, p)
    b.physical_field('v', population='E')
    b.define_function('phi', args=['v'], expression='a[i]*exp(v)',
                      population='E')
    b.define_kernel('g', time_expr='(1/taug[i,j])*exp(-t/taug[i,j])'
                                   '*heaviside(t)',
                    latex_name='g', indexed_by=['E', 'E'])
    b.set_action_text('sum(nt[i]*n[i] - (exp(nt[i])-1)*phi[i](v[i]) '
                      '+ vt[i]*((tau[i]*Dt + 1)*v[i] - Em[i] '
                      '- sum(w[i,j]*g[i,j]*n[j] for j in E)) for i in E)')
    b.set_mf_equation('vstar', '(Em[i] + sum(w[i, j]*g[i,j]*nstar[j] '
                               'for j in E))')
    b.set_mf_equation('nstar', 'phi[i](vstar[i])')
    return b.build()


def spike_only_form(n_pop, p=ASYM):
    b = _params(TemporalModelBuilder(f'Exp Hawkes spike-only test {n_pop}'),
                n_pop, p)
    b.define_kernel('r', time_expr='(exp(-t/tau[i]) - exp(-t/taug[i,j]))'
                                   '/(tau[i] - taug[i,j])*heaviside(t)',
                    freq_image='1/((1 + I*omega*tau[i])'
                               '*(1 + I*omega*taug[i,j]))',
                    latex_name='r', indexed_by=['E', 'E'])
    b.set_action_text('sum(nt[i]*n[i] - (exp(nt[i])-1)*a[i]*exp(Em[i] '
                      '+ sum(w[i,j]*Conv(r[i,j], n[j]) for j in E)) '
                      'for i in E)')
    b.set_mf_equation('nstar', 'a[i]*exp(Em[i] + sum(w[i, j]*nstar[j] '
                               'for j in E))')
    return b.build()


def _both(n_pop, k, max_ell, ext, tau, p=ASYM):
    out = []
    for mk in (voltage_form, spike_only_form):
        FI._reset_runtime_counters()
        r = compute_cumulants(mk(n_pop, p), k=k, max_ell=max_ell,
                              external_fields=ext, tau_grid=tau,
                              parallel=False, verbose=False, use_cache=False)
        assert r['status'] == 'ok'
        out.append((r, dict(FI._RUNTIME_COUNTERS)))
    return out


def _agree(volt, spk, max_ell, rtol):
    for ell in range(max_ell + 1):
        a = np.asarray(volt['C_tau_by_ell'][ell])
        b = np.asarray(spk['C_tau_by_ell'][ell])
        if not np.any(a):
            assert not np.any(b)
            continue
        np.testing.assert_allclose(b, a, rtol=rtol, atol=0)


@pytest.mark.slow
def test_two_neuron_tree_agrees():
    # (slow: the cold symbolic propagator of either form with six distinct
    # rates takes minutes)
    # tree: no interaction vertex; the two propagators agree to the
    # precision of their poles (~1e-11 relative with six distinct rates)
    (v, _), (s, _) = _both(2, 2, 0, [('n', 1), ('n', 2)], TAU)
    _agree(v, s, 0, 1e-10)


def test_one_neuron_one_loop_agrees_and_folds_the_kernel():
    (v, _), (s, cs) = _both(1, 2, 1, [('n', 1), ('n', 1)], TAU)
    _agree(v, s, 1, 1e-11)
    # every kernel was folded into its line: no extra time variable, no
    # quadrature
    assert cs['conv_lines_dressed'] > 0 and cs['conv_kernel_leg_edges'] == 0
    assert cs['nquad_calls'] == 0


def test_one_neuron_rate_correction_agrees():
    (v, _), (s, _) = _both(1, 1, 1, [('n', 1)], np.array([0.0]))
    np.testing.assert_allclose(complex(s['C_tau_by_ell'][1][0]),
                               complex(v['C_tau_by_ell'][1][0]), rtol=1e-11)


@pytest.mark.slow
def test_two_neuron_one_loop_agrees():
    # symmetric rates: 1446 one-loop diagrams of the spike-only form in about
    # 35 s warm, 4 min cold (the voltage form has 146); with six distinct
    # rates the setup-time
    # pole-tuple plans of that many diagrams exceed several GB
    (v, _), (s, cs) = _both(2, 2, 1, [('n', 1), ('n', 2)], TAU, p=SYM)
    _agree(v, s, 1, 1e-11)
    assert cs['conv_kernel_leg_edges'] == 0 and cs['nquad_calls'] == 0
