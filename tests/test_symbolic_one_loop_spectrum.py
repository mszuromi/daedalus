"""The one-loop spectrum as self-energy diagrams (``daedalus.api.symbolic_spectrum``).

``S^{(1)}_ab(w) = L_{a alpha}(w) Sigma_{alpha beta}(w) L_{b beta}(-w)``, with
tadpoles, mean-shift insertions and bubbles read off the action by Wick's
theorem.  Every model reproduces the pipeline's exact one-loop spectrum
(``dd.spectra``, state-space route) at several frequencies.
"""
import os
import sys

import numpy as np
import pytest
import sympy
from sage.all import SR

import daedalus as dd
from daedalus.api.noise import Gaussian
from daedalus.api.sde import SDE
from daedalus.api.symbolic_out import SymbolicOutError, UnsupportedModelError
from daedalus.api.symbolic_spectrum import one_loop_spectrum

sys.path.insert(0, os.path.dirname(__file__))
from test_symbolic_mean import rate_network, W2  # noqa: E402

OMEGA = [0.0, 0.7, 2.3]


def _pipeline(model, a, b):
    """The pipeline's one-loop spectrum increment and the run's parameters."""
    r = dd.run(model, dd.Config(k=2, max_ell=1, tau_grid=[0.0],
                                external_fields=[a, b],
                                spectra={'omega': OMEGA}))
    npar = {str(k): float(v) for k, v in r['num_params'].items()
            if 'star' not in str(k)}
    return np.asarray(r['spectra']['S_by_ell'][1]), npar


def _close(got, want, rtol=1e-9):
    assert np.max(np.abs(got - want)) <= rtol * np.max(np.abs(want)), (got, want)


def _cross():
    return (SDE('cross').parameter('c', default=0.3).parameter('h', default=0.2)
            .parameter('tau', default=2.0).parameter('sig', default=0.4)
            .field('x').field('y').drift('x', '-x + c*x*y + h')
            .drift('y', '(x - y)/tau')
            .noise('xi', Gaussian(var='sig^2'), into={'x': 1})).build()


def _multiplicative(interpretation):
    return (SDE(f'multiplicative {interpretation}').parameter('s0', default=0.3)
            .parameter('s1', default=0.2).parameter('g', default=0.25)
            .parameter('h', default=0.1).field('x').drift('x', '-x + g*x^2 + h')
            .noise('xi', Gaussian(var=1), into={'x': 's0 + s1*x'},
                   interpretation=interpretation)).build()


def _cubic():
    return (SDE('cubic drift').parameter('g', default=0.3).parameter('h', default=0.2)
            .parameter('b', default=0.3).parameter('D', default=0.05)
            .field('x').drift('x', '-x + g*x^2 - h*x^3 + b')
            .noise('xi', Gaussian(var='2*D'), into={'x': 1})).build()


@pytest.fixture(scope='module')
def net2():
    model = rate_network(W2, [1.0, 2.0])
    return model, one_loop_spectrum(model, mean_field='solve')


def test_diagrams_of_the_network(net2):
    model, sp = net2
    counts = {k: len(sp.by_kind(k)) for k in ('tadpole', 'mean_shift', 'bubble')}
    assert counts == {'tadpole': 0, 'mean_shift': 8, 'bubble': 48}
    # causality: no physical-physical self-energy
    phys = set(sp.phys_names)
    assert not any(t.left in phys and t.right in phys for t in sp.terms)
    # the quadratic vertex -W_rj/a xt_r dx_j^2 on the mean shift: 2 W_rj/a
    a = SR.var('a')
    ms = {(t.left, t.right, t.shift): t.coefficient for t in sp.by_kind('mean_shift')}
    assert (ms[('vt1', 'dv2', 'dv2')] - 2 * SR.var('W12') / a).simplify_rational() == 0
    # the noise bubble of two such vertices: 2 W11^2/a^2 C0_11 C0_11
    noise = [t for t in sp.by_kind('bubble') if (t.left, t.right) == ('vt1', 'vt1')
             and all(ln.kind == 'C0' and (ln.row, ln.col) == ('dv1', 'dv1')
                     for ln in t.lines)]
    assert len(noise) == 1
    assert (noise[0].coefficient - 2 * SR.var('W11') ** 2 / a ** 2
            ).simplify_rational() == 0
    # named lines, kappa symbols and the held loop integral
    s = str(sp.spectrum(('v', 1), ('v', 2)))
    for piece in ('G_dv1_vt1(omega)', 'C0_dv1_dv1(nu_loop)', 'kappa_dv1__l1',
                  'integrate'):
        assert piece in s, piece
    ex = str(sp.sigma('vt1', 'dv1', form='explicit', c0='integral'))
    assert 'G_dv' not in ex and 'C0_dv' not in ex and 'integrate' in ex


@pytest.mark.parametrize('pair', [(('v', 1), ('v', 2)), (('v', 2), ('v', 2))])
def test_network_matches_the_pipeline(net2, pair):
    model, sp = net2
    want, npar = _pipeline(model, *pair)
    _close(sp.evaluate(*pair, OMEGA, npar), want)


def test_equal_time_forms_agree(net2):
    model, sp = net2
    _, npar = _pipeline(model, ('v', 1), ('v', 1))
    ref = sp.evaluate(('v', 1), ('v', 1), OMEGA, npar)
    for c0 in ('integral', 'residues'):
        _close(sp.evaluate(('v', 1), ('v', 1), OMEGA, npar, c0=c0), ref)


@pytest.mark.parametrize('build, pairs', [
    (_cross, [(('x', 1), ('y', 1)), (('y', 1), ('y', 1))]),
    (lambda: _multiplicative('ito'), [(('x', 1), ('x', 1))]),
    (lambda: _multiplicative('stratonovich'), [(('x', 1), ('x', 1))]),
    (_cubic, [(('x', 1), ('x', 1))]),
], ids=['cross', 'multiplicative-ito', 'multiplicative-stratonovich', 'cubic'])
def test_sdes_match_the_pipeline(build, pairs):
    """An off-diagonal loop (x y vertex), noise vertices (multiplicative
    noise: a tadpole of the (2, 2) sector and (2, 1) bubbles) and a quartic
    drift tadpole."""
    model = build()
    sp = one_loop_spectrum(model, mean_field='solve')
    for pair in pairs:
        want, npar = _pipeline(model, *pair)
        _close(sp.evaluate(*pair, OMEGA, npar), want)


def test_point_process_model_with_kernels():
    """Spike trains with exponential kernels: lines with an instantaneous
    part (G_nn) on the outside, third-cumulant noise vertices inside."""
    model, _ = dd.load_model('single_population_quad_exp_test')
    want, npar = _pipeline(model, ('n', 1), ('v', 2))
    sp = one_loop_spectrum(model, mean_field='solve', values=npar)
    assert sp.by_kind('tadpole') and sp.by_kind('bubble')
    _close(sp.evaluate(('n', 1), ('v', 2), OMEGA), want)


def test_loop_without_an_ito_value_is_refused():
    """A spike train's delta(tau) times a retarded line that jumps at the
    tie: the bubble converges only conditionally."""
    model, _ = dd.load_model('single_population_spike_reset_test')
    with pytest.raises(UnsupportedModelError, match='no Itô value'):
        one_loop_spectrum(model, mean_field='solve')


def test_refusals_and_exports(net2):
    model, sp = net2
    with pytest.raises(SymbolicOutError, match='form must be'):
        sp.sigma('vt1', 'dv1', form='closed')
    with pytest.raises(SymbolicOutError, match='c0 must be'):
        sp.sigma('vt1', 'dv1', c0='closed')
    with pytest.raises(SymbolicOutError, match='unknown physical field'):
        sp.spectrum(('v', 3), ('v', 1))
    with pytest.raises(SymbolicOutError, match='no value'):
        sp.evaluate(('v', 1), ('v', 1), 0.5)
    with pytest.raises(SymbolicOutError, match='kind must be'):
        sp.by_kind('ladder')
    assert isinstance(sp.sympy(('v', 1), ('v', 1)), sympy.Expr)
    assert r'\Sigma_{vt1, dv1}' in sp.latex()
    assert sp.sigma_matrix().nrows() == 4
    assert 'bubble' in str(sp)
