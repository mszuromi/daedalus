"""
tests/test_hartree_kernels.py
=============================
Kernels and hard delays in the Gaussian closure and the screened series
(``api/hartree.py``, ``api/resum.py``).

A kernel acts on the fields linearly, so the closure embeds it before it
runs: a rational kernel exactly, as linear state fields realized by the
engine's own state-space code; a hard delay through its Pade approximant,
with the order chosen by a search like the loop path's.  Results are on the
model's own fields.

* an alpha-synapse ``Conv`` equals the model written with the two filter
  stages by hand: Hartree to rounding, and the screened series (k = 2 and
  k = 1) at one trial propagator;
* the kernel states add no dimension to the quadrature;
* a linear model with a hard delay: the closure is exact, and its
  spectrum equals the exact delay spectrum within the reported error;
* a nonlinear delayed model: the Pade order search (orders, estimates,
  cache replay), the refusal of a target that is not reached, the always
  present advisory, and the screened order 1 at the Hartree mass equals
  Hartree curve for curve (loops and closure see the same Pade tree);
* the refusals that stay: a kernel without a frequency image and non-local
  noise cumulant kernels (exogenous inputs: ``test_hartree_inputs.py``).
"""
import os
import sys
import warnings

import numpy as np
import pytest

pytest.importorskip('sage.all')

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))

import daedalus as dd  # noqa: E402
from api.delay import Kernel  # noqa: E402
from api.hartree import hartree, HartreeError, HartreeUnsupported  # noqa: E402
from api.model import TemporalModelBuilder  # noqa: E402
from api.noise import Gaussian  # noqa: E402
from api.sde import SDE  # noqa: E402

TAU = np.array([0.0, 0.5, 1.0])


# ── models ──────────────────────────────────────────────────────────────────

def _rate(hand):
    """A quadratic-rate model: v (cubic-stabilised) drives the rate u = v^2/a
    through a linear stage, and u feeds back on v through an alpha synapse
    ``Conv(alpha, u)`` (``hand``: the same synapse as two filter stages
    written out)."""
    m = SDE('alpha synapse rate' + (' by hand' if hand else ''))
    for p, d in dict(ts=0.5, D=0.3, mu=1.0, eps=0.4, a=2.0).items():
        m.parameter(p, default=d, domain='positive')
    m.field('v').field('u')
    if hand:
        m.field('s1').field('s2')
        m.drift('s1', '(-s1 + u)/ts').drift('s2', '(-s2 + s1)/ts')
        m.drift('v', '-mu*v - eps*v^3 + s2')
    else:
        m.define_kernel('alpha', kernel=Kernel.alpha('ts'))
        m.drift('v', '-mu*v - eps*v^3 + Conv(alpha, u)')
    m.drift('u', '-u + v^2/a')
    m.noise('xi', Gaussian(var='2*D'), into={'v': 1})
    return m.build()


def _delayed(eps=0.0, c=-0.6, d=0.8, a=1.0, D=0.5):
    """``dx = (-a x + c x(t - d) - eps x^3) dt + sqrt(2 D) dW``."""
    m = SDE('delayed' + ('' if eps == 0 else ' cubic'))
    for p, v in dict(a=a, c=c, d=d, D=D, eps=eps).items():
        m.parameter(p, default=v) if p == 'c' else m.parameter(
            p, default=v, domain='positive')
    m.field('x')
    m.drift('x', '-a*x + c*x(t - d)' + (' - eps*x^3' if eps else ''))
    m.noise('xi', Gaussian(var='2*D'), into={'x': 1})
    return m.build()


def _close(a, b, tol):
    return np.max(np.abs(np.asarray(a) - np.asarray(b))) <= tol * max(
        np.max(np.abs(b)), 1e-300)


# ── a rational kernel is exact ──────────────────────────────────────────────

def test_alpha_synapse_equals_the_hand_embedded_model():
    kern, hand = _rate(False), _rate(True)
    a, b = hartree(kern, warn=False), hartree(hand, warn=False)
    assert a.converged and a.fields == ['dv1', 'du1']
    assert _close(a.mean, b.mean[:2], 1e-12)
    assert _close(a.cov, b.cov[:2, :2], 1e-12)
    tau = np.array([-0.4, 0.0, 0.3, 1.0, 3.0])
    assert _close(a.C_matrix(tau), b.C_matrix(tau)[:, :2, :2], 1e-12)
    om = np.array([0.0, 0.7, 2.5])
    assert _close(a.spectrum(om, 0, 1), b.spectrum(om, 0, 1), 1e-12)
    # the kernel states are listed, outside the fields, and add no
    # quadrature dimension
    assert a.info['kernel_states']['states'] == 2
    assert len(a.full['aux']) == 2 and a.full['A'].shape == (4, 4)
    assert a.info['nonlinear_fields'] == ['dv1']
    assert a.info['quadrature_nodes'] == 3
    # the slowest rate of the whole state is a decay rate of the model
    assert np.all(np.real(a.rates) > 0) and a.rates.size == 4


@pytest.mark.parametrize('k', [2, 1])
def test_screened_series_with_a_kernel_equals_the_hand_embedded_one(k):
    """Same PMS functional (the variances of v and u), so the same trial
    propagator, and loops that run the kernel route against the state
    route."""
    ext = [('v', 1), ('v', 1)] if k == 2 else [('v', 1)]
    out = []
    for hand in (False, True):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            out.append(dd.run(_rate(hand), dd.Config(
                k=k, max_ell=1, tau_grid=TAU, external_fields=ext,
                resum={'method': 'screened', 'observable': ['v', 'u']},
                verbose=False)))
    a, b = out
    assert a['resummation']['kappa'] == pytest.approx(
        b['resummation']['kappa'], rel=1e-9)
    assert _close(a['C_tau'], b['C_tau'], 1e-8)
    if k == 1:
        assert a['mean'] == pytest.approx(b['mean'], rel=1e-8)


def test_hartree_run_with_a_kernel_through_config():
    r = dd.run(_rate(False), dd.Config(
        k=2, resum='hartree', tau_grid=TAU, external_fields=[('v', 1), ('v', 1)]))
    h = hartree(_rate(False), warn=False)
    assert np.allclose(r['C_tau'], h.C(TAU, 0, 0))


# ── hard delays: the Pade order search ──────────────────────────────────────

def test_linear_hard_delay_is_exact_within_the_reported_error():
    """The Gaussian closure of a linear model is exact; its spectrum is the
    delay spectrum 2 D / |i w + a - c exp(-i w d)|^2."""
    from scipy.integrate import quad
    a, c, d, D = 1.0, -0.6, 0.8, 0.5
    r = hartree(_delayed(c=c, d=d, a=a, D=D), warn=False)
    info = r.info['delay']
    assert info['order_source'] == 'search' and info['order'] in (4, 8, 16)
    assert info['estimate'] <= info['target'] == 1e-8
    assert info['orders'][-1] == info['order']
    S = lambda w: 2 * D / abs(1j * w + a - c * np.exp(-1j * w * d)) ** 2
    om = np.array([0.0, 0.7, 2.0])
    got = r.spectrum(om).real
    assert np.allclose(got, [S(w) for w in om], rtol=1e-8, atol=0)
    var = quad(S, -np.inf, np.inf, limit=400)[0] / (2 * np.pi)
    assert abs(r.cov[0, 0] - var) < 1e-7 * var
    # never silent: the advisory names the order and the error
    codes = [x['code'] for x in r.advisories]
    assert 'HARTREE_DELAY_PADE' in codes
    msg = next(x['message'] for x in r.advisories
               if x['code'] == 'HARTREE_DELAY_PADE')
    assert f"order {info['order']}" in msg and 'lag correlator' in msg
    assert r.info['model'] == 'delayed'          # not the Pade copy's name


def test_pade_order_converges_and_the_search_is_cached():
    m = _delayed(eps=0.8, c=-0.9, d=1.0)
    ref = hartree(m, warn=False, delay={'order': 16}).cov[0, 0]
    err = {n: abs(hartree(m, warn=False, delay={'order': n}).cov[0, 0] - ref)
           for n in (2, 4, 8)}
    assert err[2] > err[4] > err[8]
    assert err[8] < 1e-8 * ref
    r1 = hartree(m, warn=False)
    assert r1.info['delay']['order_source'] == 'search'
    assert len(r1.info['delay']['estimates']) >= 1
    r2 = hartree(m, warn=False)
    assert r2.info['delay']['order_source'] == 'replay'
    assert r2.info['delay']['order'] == r1.info['delay']['order']
    assert np.array_equal(r1.cov, r2.cov)
    r3 = hartree(m, warn=False, delay={'order_cache': False})
    assert r3.info['delay']['order_source'] == 'search'


def test_a_target_that_is_not_reached_is_refused():
    m = _delayed(eps=0.8, c=-0.9, d=1.0)
    with pytest.raises(HartreeError, match='did not reach the target') as ei:
        hartree(m, warn=False, delay={'tol': 1e-30, 'n_start': 2, 'n_max': 8})
    assert 'orders tried [2, 4, 8]' in str(ei.value)
    # an order given explicitly is accepted, flagged as having no estimate
    r = hartree(m, warn=False, delay={'order': 4})
    assert r.info['delay']['order_source'] == 'given'
    assert r.info['delay']['estimate'] is None
    assert any('no error estimate' in x['message'] for x in r.advisories)


def test_delay_advisory_is_warned():
    with pytest.warns(UserWarning, match='HARTREE_DELAY_PADE'):
        hartree(_delayed(), delay={'order': 4})


def test_screened_order_one_at_the_hartree_mass_is_hartree_with_delays():
    """The loops (with the Pade tree) and the closure see the same Pade
    approximation of the delay: order 1 at kappa = 1 is the Hartree curve."""
    m = _delayed(eps=0.8, c=-0.9, d=1.0)
    delay = {'order': 4, 'tree': 'pade'}
    tau = np.array([0.0, 1.0])
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        s = dd.run(m, dd.Config(
            k=2, max_ell=1, resum={'method': 'screened', 'mass': 'hartree'},
            tau_grid=tau, external_fields=[('x', 1), ('x', 1)], delay=delay,
            verbose=False))
        h = dd.run(m, dd.Config(
            k=2, resum='hartree', tau_grid=tau,
            external_fields=[('x', 1), ('x', 1)], delay=delay))
    assert np.allclose(s['C_tau'], h['C_tau'], rtol=1e-10, atol=1e-12)
    assert s['resummation']['hartree']['rates'].size == 5     # x + 4 Pade states


# ── what stays refused ──────────────────────────────────────────────────────

def test_a_kernel_without_a_frequency_image_is_refused():
    m = dict(_rate(False), kernel_ft_image=None)
    with pytest.raises(HartreeUnsupported, match='frequency image'):
        hartree(m, warn=False)


def test_non_local_noise_cumulant_kernels_are_refused_unless_embedded():
    b = (TemporalModelBuilder('colored, not embedded')
         .physical_field('x')
         .parameter('mu', default=0.6, domain='positive')
         .parameter('D', default=0.5, domain='positive')
         .parameter('tauc', default=0.8, domain='positive')
         .declare_cgf_term('CXX', response_legs=['xt', 'xt'], order=2,
                           coefficient='2*D/tauc', kernel='exp(-abs(tau)/tauc)')
         .markovianize(False)
         .set_action_text('xt*((Dt+mu)*x)').mf_residual(lhs='(Dt+mu)*x', rhs='0'))
    with pytest.raises(HartreeUnsupported, match='Markovian embedding'):
        hartree(b.build(), warn=False)
