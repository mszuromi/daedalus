"""
Delay kernels inside nonlinear terms (``api.delay``).

* A delayed strictly proper kernel at a vertex leg (a delayed leg operator
  inside a rate nonlinearity): the spike-only exponential Hawkes neuron with a
  delayed alpha x membrane leg operator inside ``exp`` equals its voltage form
  (the delay in the linear voltage equation) at tree and one loop, at the
  same Pade order.
* A bare hard delay inside a product, ``x * x(t - d)``: the all-pass (n, n)
  image ``(-1)^n + (strictly proper)`` runs as it is (item 56: Phase J folds
  the direct term into the dressed line, on both routes); it equals the same
  rational kernel declared without a delay and its Markov embedding (n
  auxiliary linear fields, no kernel), converges with n to the same limit as
  the strictly proper ``[n-1/n]`` approximant, and the tree stays exact.  A
  line Phase J cannot dress falls back on the ``[n-1/n]`` rerun.
* A bare hard delay inside the exponential of a spike field multiplies delta
  functions: refused by name (incomplete + warning, never silent).  Per line
  (item 56, D10): a bare delay of a continuous field inside ``exp`` in a
  spiking model runs (equal to the delay moved into the linear voltage
  equation), and so does a bare delay of the spike field in a linear term.
"""
from __future__ import annotations

import math
import os
import sys
import warnings

import numpy as np
import pytest

pytest.importorskip('sage.all')

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from api import compute_cumulants                                      # noqa: E402
from api.delay import (DelayOrderWarning, pade_coefficients,          # noqa: E402
                       pade_mn_coefficients)
from api.model import TemporalModelBuilder                             # noqa: E402
from api.noise import Gaussian                                         # noqa: E402
from api.sde import SDE                                                # noqa: E402

EXT = [('x', 1), ('x', 1)]
TAU = np.array([0.0, 0.5, 2.0])


# ── the [m/n] approximant ───────────────────────────────────────────────


def test_pade_mn_coefficients():
    for n in (1, 2, 4, 8):
        a, b = pade_mn_coefficients(n, n)
        assert a == b == pade_coefficients(n)
    for n in range(1, 13):
        a, b = pade_mn_coefficients(n - 1, n)
        # Hurwitz denominator (poles of the [n-1/n] image in the left half plane)
        roots = np.roots([float(c) for c in b[::-1]])
        assert np.all(roots.real < 0)
        # order of contact m + n at small z
        z = 1e-2
        val = (sum(float(c) * (-z) ** k for k, c in enumerate(a))
               / sum(float(c) * z ** k for k, c in enumerate(b)))
        assert abs(val - math.exp(-z)) < 10 * z ** (2 * n) + 1e-15
    with pytest.raises(ValueError):
        pade_mn_coefficients(3, 2)


# ── a bare hard delay inside a product ──────────────────────────────────


def _product_model(name, **kern):
    s = (SDE(name).parameter('a', default=1.0).parameter('b', default=1.0)
         .parameter('c', default=0.5).parameter('d', default=0.5, domain='positive')
         .parameter('D', default=0.1).field('x'))
    if kern:
        s = s.define_kernel('g', **kern).drift('x', '-a*x + b - c*x*Conv(g, x)')
    else:
        s = s.drift('x', '-a*x + b - c*x*x(t - d)')
    return s.noise('xi', Gaussian(var='2*D'), couples={'x': 1}).build()


def _rational_text(m, n):
    a, b = pade_mn_coefficients(m, n)
    z = '(I*omega*d)'
    num = ' + '.join(f'({float(c)!r})*(-{z})^{k}' for k, c in enumerate(a))
    den = ' + '.join(f'({float(c)!r})*({z})^{k}' for k, c in enumerate(b))
    return f'({num})/({den})'


def _markov_product_model(name, n):
    """x*x(t - d) with the Pade (n, n) kernel as n auxiliary linear fields
    (no kernel at all): the image c + beta(s)/alpha(s) (monic alpha, exact
    coefficients) in controllable form, z_i' = z_{i+1}, z_n' = -alpha . z + x,
    and x(t - d) -> c x + beta . z."""
    from fractions import Fraction
    a, b = pade_mn_coefficients(n, n)
    d = Fraction(1, 2)
    num = [Fraction(a[k]) * (-d) ** k for k in range(n + 1)]
    den = [Fraction(b[k]) * d ** k for k in range(n + 1)]
    num, den = [x / den[n] for x in num], [x / den[n] for x in den]
    c = num[n]
    beta = [num[k] - c * den[k] for k in range(n)]
    s = (SDE(name).parameter('a', default=1.0).parameter('b', default=1.0)
         .parameter('c', default=0.5).parameter('D', default=0.1).field('x'))
    for i in range(1, n + 1):
        s = s.field(f'z{i}')
    y = f'({float(c)!r})*x + ' + ' + '.join(f'({float(beta[k])!r})*z{k + 1}' for k in range(n))
    s = s.drift('x', f'-a*x + b - c*x*({y})')
    for i in range(1, n):
        # the restoring term on the lhs (the mean-field equation needs it)
        s = s.equation(f'z{i}', f'(Dt + 1)*z{i}', f'z{i + 1} + z{i}')
    s = s.drift(f'z{n}', '-' + ' - '.join(f'({float(den[k])!r})*z{k + 1}' for k in range(n))
                + ' + x')
    return s.noise('xi', Gaussian(var='2*D'), couples={'x': 1}).build()


def _counter(name):
    from engine.integration.time_domain import final_integral as FI
    return FI._RUNTIME_COUNTERS.get(name, 0)


def test_hard_delay_in_a_product_runs_the_all_pass_approximant():
    """x * x(t - d) at orders n = 4 and 8: the (n, n) approximant itself (its
    direct term dressed into the line), equal (Pade tree and one loop, <=
    1e-13) to the same rational kernel declared without a delay, which never
    enters the delay driver."""
    kw = dict(k=2, max_ell=1, external_fields=EXT, tau_grid=TAU, parallel=False,
              verbose=False, use_cache=False)
    for n in (4, 8):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            c0 = _counter('conv_lines_dressed_direct')
            rd = compute_cumulants(_product_model(f'dnl hard product {n}', freq_image='1',
                                                  delay='d'),
                                   delay={'order': n, 'tree': 'pade'}, **kw)
            assert _counter('conv_lines_dressed_direct') > c0
            rr = compute_cumulants(_product_model(f'dnl rational product {n}',
                                                  freq_image=_rational_text(n, n)), **kw)
        assert rd['status'] == rr['status'] == 'ok'
        assert rd['delay']['approximant'] == 'pade' and rd['delay']['incomplete'] is None
        for ell in (0, 1):
            x = np.real(np.asarray(rd['C_tau_by_ell'][ell]))
            y = np.real(np.asarray(rr['C_tau_by_ell'][ell]))
            assert np.max(np.abs(y)) > 0
            assert np.max(np.abs(x - y)) <= 1e-13 * np.max(np.abs(y))


@pytest.mark.slow
def test_hard_delay_in_a_product_equals_its_markov_embedding():
    """(slow, ~25 s) x * x(t - d) at n = 4 on the direct-term route equals the
    Markov embedding of the same (n, n) kernel (four auxiliary linear fields,
    no kernel code at all), tree and one loop (measured 1.9e-15)."""
    n = 4
    kw = dict(k=2, max_ell=1, external_fields=EXT, tau_grid=TAU, parallel=False,
              verbose=False, use_cache=False)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        rd = compute_cumulants(_product_model('dnl hard product mk', freq_image='1', delay='d'),
                               delay={'order': n, 'tree': 'pade'}, **kw)
        rm = compute_cumulants(_markov_product_model('dnl markov product', n), **kw)
    assert rd['status'] == rm['status'] == 'ok'
    for ell in (0, 1):
        x = np.real(np.asarray(rd['C_tau_by_ell'][ell]))
        z = np.real(np.asarray(rm['C_tau_by_ell'][ell]))
        assert np.max(np.abs(z)) > 0
        assert np.max(np.abs(x - z)) <= 1e-12 * np.max(np.abs(z))


def test_undressable_line_falls_back_on_the_strictly_proper_approximant(monkeypatch):
    """With line dressing off, the direct term has no route (a kernel edge
    holds only a strictly proper kernel): refused by name in Phase J, and the
    driver reruns with the [n-1/n] approximant, equal to that rational kernel
    declared without a delay (the item-47 route)."""
    from engine.integration.time_domain import final_integral as FI
    monkeypatch.setattr(FI, 'DRESS_CONV_KERNELS', False)
    n = 4
    kw = dict(k=2, max_ell=1, external_fields=EXT, tau_grid=TAU, parallel=False,
              verbose=False, use_cache=False)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        c0 = _counter('conv_kernel_direct_refused')
        rd = compute_cumulants(_product_model('dnl hard product leg', freq_image='1', delay='d'),
                               delay={'order': n, 'tree': 'pade'}, **kw)
        assert _counter('conv_kernel_direct_refused') > c0
        rr = compute_cumulants(_product_model('dnl rational product leg',
                                              freq_image=_rational_text(n - 1, n)), **kw)
    assert rd['status'] == 'ok' and rr['status'] == 'ok'
    assert rd['delay']['approximant'] == 'pade_proper' and rd['delay']['incomplete'] is None
    for ell in (0, 1):
        x = np.real(np.asarray(rd['C_tau_by_ell'][ell]))
        y = np.real(np.asarray(rr['C_tau_by_ell'][ell]))
        assert np.max(np.abs(y)) > 0
        np.testing.assert_allclose(x, y, rtol=1e-12, atol=0)


def test_hard_delay_in_a_product_drift_form_converges_with_an_exact_tree(monkeypatch):
    """x(t - d) written in the drift: complete with the (n, n) approximant,
    the exact tree, loops at n = 4 and 8 within 1e-4 of max|C|, and the same
    limit as the strictly proper [n-1/n] approximant (forced by switching
    line dressing off) at n = 8 (simulation agreement of the [n-1/n] route:
    internal (not distributed) acceptance run, within 0.6 SE)."""
    from engine.integration.time_domain import final_integral as FI
    m = _product_model('dnl hard product drift')
    kw = dict(k=2, max_ell=1, external_fields=EXT, tau_grid=TAU, parallel=False,
              verbose=False)
    out = {}
    for n in (4, 8):
        with warnings.catch_warnings(record=True) as ws:
            warnings.simplefilter('always')
            r = compute_cumulants(m, delay={'order': n}, **kw)
        assert r['status'] == 'ok'
        assert r['delay']['approximant'] == 'pade' and r['delay']['tree'] == 'exact'
        assert not any('could not evaluate' in str(w.message) for w in ws)
        rep = r['analysis']
        assert any('Pade (%d, %d)' % (n, n) in f.message for f in rep.findings
                   if f.code == 'G3_DELAY_APPROXIMATED')
        out[n] = [np.real(np.asarray(r['C_tau_by_ell'][ell])) for ell in (0, 1)]
    monkeypatch.setattr(FI, 'DRESS_CONV_KERNELS', False)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        rp = compute_cumulants(m, delay={'order': 8}, **kw)
    assert rp['delay']['approximant'] == 'pade_proper'
    proper = [np.real(np.asarray(rp['C_tau_by_ell'][ell])) for ell in (0, 1)]
    np.testing.assert_array_equal(out[4][0], out[8][0])          # exact tree
    np.testing.assert_array_equal(out[8][0], proper[0])
    scale = np.max(np.abs(out[8][0] + out[8][1]))
    assert np.max(np.abs(out[8][1])) > 1e-3 * scale             # the loop matters
    assert np.max(np.abs(out[8][1] - out[4][1])) < 1e-4 * scale
    assert np.max(np.abs(out[8][1] - proper[1])) < 1e-4 * scale


# ── delayed leg operators inside a rate nonlinearity ────────────────────


def _hawkes_base(name, p):
    return (TemporalModelBuilder(name).population('E', size=1)
            .physical_field('n', population='E')
            .parameter('Em', default=[-1.5], indexed_by=['E'])
            .parameter('tau', default=[2.0], indexed_by=['E'], domain='positive')
            .parameter('taug', default=[[1.0]], indexed_by=['E', 'E'], domain='positive')
            .parameter('a', default=[1.0], indexed_by=['E'], domain='positive')
            .parameter('w', default=[[p]], indexed_by=['E', 'E'])
            .parameter('d', default=1.5, domain='positive')
            .define_function('phi', args=['v'], expression='a[i]*exp(v)', population='E'))


def _hawkes_voltage(p=0.5):
    return (_hawkes_base('dnl Hawkes delayed alpha voltage', p).physical_field('v', population='E')
            .define_kernel('g', freq_image='1/(1 + I*omega*taug[i,j])^2',
                           indexed_by=['E', 'E'], delay='d')
            .set_action_text('sum( nt[i]*n[i] - (exp(nt[i])-1)*phi[i](v[i]) + vt[i]*((tau[i]*Dt + 1)*v[i]'
                             ' - Em[i] - sum(w[i,j]*g[i,j]*n[j] for j in E)) for i in E)')
            .set_mf_equation('vstar', '(Em[i] + sum(w[i, j]*g[i,j]*nstar[j] for j in E))')
            .set_mf_equation('nstar', 'phi[i](vstar[i])').build())


def _hawkes_spike_only(p=0.5, image='1/((1 + I*omega*tau[i])*(1 + I*omega*taug[i,j])^2)',
                       name='dnl Hawkes delayed alpha spike-only'):
    return (_hawkes_base(name, p)
            .define_kernel('r', freq_image=image, indexed_by=['E', 'E'], delay='d')
            .set_action_text('sum(nt[i]*n[i] - (exp(nt[i])-1)*phi[i](Em[i] + sum(w[i,j]*Conv(r[i,j], n[j])'
                             ' for j in E)) for i in E)')
            .set_mf_equation('nstar', 'phi[i](Em[i] + sum(w[i, j]*nstar[j] for j in E))').build())


@pytest.mark.slow
def test_delayed_leg_operator_inside_exp_equals_the_voltage_form():
    # (slow: the spike-only one loop dresses every leg, ~75 s)
    kw = dict(k=2, max_ell=1, external_fields=[('dn', 1), ('dn', 1)],
              tau_grid=np.array([-4.0, 0.0, 0.5, 1.5, 6.0]), parallel=False, verbose=False,
              delay={'order': 4})
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        v = compute_cumulants(_hawkes_voltage(), **kw)
        s = compute_cumulants(_hawkes_spike_only(), **kw)
    assert v['status'] == s['status'] == 'ok'
    assert s['delay']['approximant'] == 'pade' and s['delay']['tree'] == 'exact'
    for ell in (0, 1):
        a = np.real(np.asarray(v['C_tau_by_ell'][ell]))
        b = np.real(np.asarray(s['C_tau_by_ell'][ell]))
        assert np.max(np.abs(a)) > 0
        np.testing.assert_allclose(b, a, rtol=1e-12, atol=0)


def test_bare_delay_inside_exp_of_a_spike_field_is_refused_by_name():
    m = _hawkes_spike_only(p=0.3, image='1', name='dnl Hawkes bare delay in exp')
    with warnings.catch_warnings(record=True) as ws:
        warnings.simplefilter('always')
        r = compute_cumulants(m, k=2, max_ell=1, external_fields=[('dn', 1), ('dn', 1)],
                              tau_grid=TAU, parallel=False, verbose=False, delay={'order': 4})
    assert r['status'] == 'incomplete'
    assert r['delay']['approximant'] == 'pade' and r['delay']['target_met'] is False
    assert r['delay']['cap'].startswith('refused') and 'delta' in r['delay']['cap']
    assert 'z_r_1_1 (dn1)' in r['delay']['cap']           # per line (D10)
    assert any(issubclass(w.category, DelayOrderWarning) for w in ws)
    # the (n, n) pass's own warning is held back, then re-emitted (once)
    inc = [w for w in ws if type(w.message).__name__ == 'PhaseJIncompleteWarning']
    assert len(inc) == 1


# ── D10: the refusal is per line ────────────────────────────────────────


def _hawkes_delayed_voltage_leg(p=0.3, name='dnl Hawkes bare delay of v in exp'):
    """A spiking neuron whose rate reads its voltage through a bare delay,
    phi(v(t - d)): a bare delay of a CONTINUOUS field inside exp."""
    return (_hawkes_base(name, p).physical_field('v', population='E')
            .define_kernel('al', freq_image='1/(1 + I*omega*taug[i,j])^2', indexed_by=['E', 'E'])
            .define_kernel('g', freq_image='1', indexed_by=['E', 'E'], delay='d')
            .set_action_text('sum( nt[i]*n[i] - (exp(nt[i])-1)*phi[i](sum(Conv(g[i,j], v[j]) for j in E))'
                             ' + vt[i]*((tau[i]*Dt + 1)*v[i] - Em[i] - sum(w[i,j]*Conv(al[i,j], n[j])'
                             ' for j in E)) for i in E)')
            .set_mf_equation('vstar', '(Em[i] + sum(w[i, j]*nstar[j] for j in E))')
            .set_mf_equation('nstar', 'phi[i](vstar[i])').build())


def _hawkes_delayed_voltage_eq(p=0.3, name='dnl Hawkes delayed alpha in v'):
    """The same process with u(t) = v(t - d) as the field: the delay moves
    into the linear voltage equation (a delayed alpha filter)."""
    return (_hawkes_base(name, p).physical_field('v', population='E')
            .define_kernel('al', freq_image='1/(1 + I*omega*taug[i,j])^2', indexed_by=['E', 'E'],
                           delay='d')
            .set_action_text('sum( nt[i]*n[i] - (exp(nt[i])-1)*phi[i](v[i]) + vt[i]*((tau[i]*Dt + 1)*v[i]'
                             ' - Em[i] - sum(w[i,j]*Conv(al[i,j], n[j]) for j in E)) for i in E)')
            .set_mf_equation('vstar', '(Em[i] + sum(w[i, j]*nstar[j] for j in E))')
            .set_mf_equation('nstar', 'phi[i](vstar[i])').build())


def _linear_hawkes_bare_delay(name='dnl linear Hawkes bare delay'):
    """A linear Hawkes neuron with a bare delay of its own spike train: the
    delay of a field with a delta part, but only in a linear term."""
    return (_hawkes_base(name, 0.3)
            .define_kernel('g', freq_image='1', indexed_by=['E', 'E'], delay='d')
            .set_action_text('sum(nt[i]*n[i] - (exp(nt[i])-1)*(a[i] + sum(w[i,j]*Conv(g[i,j], n[j])'
                             ' for j in E)) for i in E)')
            .set_mf_equation('nstar', 'a[i] + sum(w[i, j]*nstar[j] for j in E)').build())


def test_bare_delays_on_delta_lines_is_per_kernel_and_field():
    from api import analyze as an
    from api.delay import bare_delays_on_delta_lines, bare_delay_kernels
    q = an.Question(k=2, max_ell=1, external_fields=(('dn', 1), ('dn', 1)))
    got = {}
    for m in (_hawkes_spike_only(p=0.3, image='1', name='dnl d10 spike'),
              _hawkes_delayed_voltage_leg(name='dnl d10 voltage leg'),
              _linear_hawkes_bare_delay(name='dnl d10 linear')):
        rep, ctx = an.analyze_with_context(m, q, None, use_cache=False)
        bare = bare_delay_kernels(m, ctx.ns)
        assert bare
        got[m['name']] = bare_delays_on_delta_lines(ctx, bare)
    assert got == {'dnl d10 spike': ['z_r_1_1 (dn1)'], 'dnl d10 voltage leg': [],
                   'dnl d10 linear': []}


def test_bare_delay_of_a_continuous_field_inside_exp_runs_in_a_spiking_model():
    """phi(v(t - d)) in a model with point-process lines (the old per-model
    rule refused it): it runs with the (n, n) approximant and equals the same
    process written with u = v(t - d) (the delay in the linear voltage
    equation as a delayed alpha filter), tree and one loop, same order."""
    kw = dict(k=2, max_ell=1, external_fields=[('dn', 1), ('dn', 1)],
              tau_grid=np.array([-4.0, 0.0, 0.5, 1.5, 6.0]), parallel=False, verbose=False,
              delay={'order': 4, 'tree': 'pade'})
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        a = compute_cumulants(_hawkes_delayed_voltage_leg(), **kw)
        b = compute_cumulants(_hawkes_delayed_voltage_eq(), **kw)
    assert a['status'] == b['status'] == 'ok'
    assert a['delay']['approximant'] == b['delay']['approximant'] == 'pade'
    for ell in (0, 1):
        x = np.real(np.asarray(a['C_tau_by_ell'][ell]))
        y = np.real(np.asarray(b['C_tau_by_ell'][ell]))
        assert np.max(np.abs(y)) > 0
        assert np.max(np.abs(x - y)) <= 1e-12 * np.max(np.abs(y))


def test_bare_delay_of_a_spike_field_in_a_linear_term_runs():
    """A bare delay of the spike train in the linear Hawkes rate: complete
    with the (n, n) approximant (its direct term dresses lines with a delta
    part at single-leg vertices), and the one loop of a linear Hawkes
    two-point function is exactly zero."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        r = compute_cumulants(_linear_hawkes_bare_delay(), k=2, max_ell=1,
                              external_fields=[('dn', 1), ('dn', 1)], tau_grid=TAU,
                              parallel=False, verbose=False, delay={'order': 4})
    assert r['status'] == 'ok' and r['delay']['approximant'] == 'pade'
    assert r['delay']['cap'] is None or not str(r['delay']['cap']).startswith('refused')
    assert np.max(np.abs(np.asarray(r['C_tau_by_ell'][0]))) > 0
    assert np.max(np.abs(np.asarray(r['C_tau_by_ell'][1]))) == 0


def test_delay_warnings_are_shown_under_sages_startup_filters(monkeypatch):
    """The Sage launcher sets PYTHONWARNINGS=ignore and Sage installs
    message-specific numpy ``ignore`` filters in front of it; the package's
    own warnings (DelayOrderWarning, PhaseJIncompleteWarning) must still be
    shown, while the user's category-wide ignore hides them (docket 51)."""
    import sys
    m = _hawkes_spike_only(p=0.3, image='1', name='dnl Hawkes bare delay in exp')
    kw = dict(k=2, max_ell=1, external_fields=[('dn', 1), ('dn', 1)],
              tau_grid=TAU, parallel=False, verbose=False, delay={'order': 4})
    monkeypatch.setattr(sys, 'warnoptions', ['ignore'])

    def sage_like():
        warnings.simplefilter('ignore')
        warnings.filterwarnings('ignore', message='numpy.ndarray size changed',
                                category=Warning)
    with warnings.catch_warnings(record=True) as ws:
        sage_like()
        r = compute_cumulants(m, **kw)
    assert r['status'] == 'incomplete'
    assert {type(w.message).__name__ for w in ws} >= {'DelayOrderWarning',
                                                      'PhaseJIncompleteWarning'}
    assert len([w for w in ws if type(w.message).__name__
                == 'PhaseJIncompleteWarning']) == 1
    with warnings.catch_warnings(record=True) as ws:
        sage_like()
        warnings.filterwarnings('ignore', category=DelayOrderWarning)
        compute_cumulants(m, **kw)
    assert not any(issubclass(w.category, DelayOrderWarning) for w in ws)
