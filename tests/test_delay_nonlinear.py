"""
Delay kernels inside nonlinear terms (``api.delay``).

* A delayed strictly proper kernel at a vertex leg (a delayed leg operator
  inside a rate nonlinearity): the spike-only exponential Hawkes neuron with a
  delayed alpha x membrane leg operator inside ``exp`` equals its voltage form
  (the delay in the linear voltage equation) at tree and one loop, at the
  same Pade order.
* A bare hard delay inside a product, ``x * x(t - d)``: the all-pass (n, n)
  image has a direct term the kernel routes do not take, so the driver reruns
  with the strictly proper ``[n-1/n]`` approximant; its loops equal the same
  rational kernel declared without a delay, converge with n, and the tree
  stays exact.
* A bare hard delay inside the exponential of a spike field multiplies delta
  functions: refused by name (incomplete + warning, never silent).
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


def test_hard_delay_in_a_product_equals_its_strictly_proper_rational_kernel():
    """x * x(t - d) at order n: the strictly proper [n-1/n] approximant, equal
    (tree with the Pade tree, and one loop) to the same rational kernel
    declared without a delay, which never enters the delay driver."""
    n = 4
    kw = dict(k=2, max_ell=1, external_fields=EXT, tau_grid=TAU, parallel=False,
              verbose=False, use_cache=False)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        rd = compute_cumulants(_product_model('dnl hard product', freq_image='1', delay='d'),
                               delay={'order': n, 'tree': 'pade'}, **kw)
        rr = compute_cumulants(_product_model('dnl rational product',
                                              freq_image=_rational_text(n - 1, n)), **kw)
    assert rd['status'] == 'ok' and rr['status'] == 'ok'
    assert rd['delay']['approximant'] == 'pade_proper' and rd['delay']['incomplete'] is None
    for ell in (0, 1):
        x = np.real(np.asarray(rd['C_tau_by_ell'][ell]))
        y = np.real(np.asarray(rr['C_tau_by_ell'][ell]))
        assert np.max(np.abs(y)) > 0
        np.testing.assert_allclose(x, y, rtol=1e-12, atol=0)


def test_hard_delay_in_a_product_drift_form_converges_with_an_exact_tree():
    """x(t - d) written in the drift: complete (strictly proper retry), the
    exact tree, loops at n = 4 and 8 within 1e-4 of max|C| (simulation
    agreement: internal (not distributed) acceptance run, within 0.6 SE)."""
    m = _product_model('dnl hard product drift')
    out = {}
    for n in (4, 8):
        with warnings.catch_warnings(record=True) as ws:
            warnings.simplefilter('always')
            r = compute_cumulants(m, k=2, max_ell=1, external_fields=EXT, tau_grid=TAU,
                                  parallel=False, verbose=False, delay={'order': n})
        assert r['status'] == 'ok'
        assert r['delay']['approximant'] == 'pade_proper' and r['delay']['tree'] == 'exact'
        # the first (n, n) attempt's incompleteness warning is not shown
        assert not any('could not evaluate' in str(w.message) for w in ws)
        rep = r['analysis']
        assert any('Pade (%d, %d)' % (n - 1, n) in f.message for f in rep.findings
                   if f.code == 'G3_DELAY_APPROXIMATED')
        out[n] = [np.real(np.asarray(r['C_tau_by_ell'][ell])) for ell in (0, 1)]
    np.testing.assert_array_equal(out[4][0], out[8][0])          # exact tree
    scale = np.max(np.abs(out[8][0] + out[8][1]))
    assert np.max(np.abs(out[8][1])) > 1e-3 * scale             # the loop matters
    assert np.max(np.abs(out[8][1] - out[4][1])) < 1e-4 * scale


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
    assert any(issubclass(w.category, DelayOrderWarning) for w in ws)
    # the (n, n) pass's own warning is held back, then re-emitted (once)
    inc = [w for w in ws if type(w.message).__name__ == 'PhaseJIncompleteWarning']
    assert len(inc) == 1


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
