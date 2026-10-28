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


def _pipeline(model, a, b, parameters=None):
    """The pipeline's one-loop spectrum increment and the run's parameters."""
    r = dd.run(model, dd.Config(k=2, max_ell=1, tau_grid=[0.0],
                                external_fields=[a, b], parameters=parameters,
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


def test_evaluate_falls_back_on_the_model_defaults(net2):
    """Defaults fill the parameters not given; 'a' moves the saddle, which
    is re-solved: the pipeline run at a = 2.5."""
    model, sp = net2
    want, npar = _pipeline(model, ('v', 1), ('v', 2))
    _close(sp.evaluate(('v', 1), ('v', 2), OMEGA), want, rtol=1e-12)
    got = sp.evaluate(('v', 1), ('v', 2), OMEGA, {'a': 2.5})
    want_a, _ = _pipeline(model, ('v', 1), ('v', 2), {'a': 2.5})
    _close(got, want_a)
    assert np.max(np.abs(got - want)) > 1e-3 * np.max(np.abs(want))


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


def test_alpha_kernel_model():
    """Alpha kernels, real in ``s = i omega`` once the ``I`` of their image
    cancels; lines of order 2 and 3 in the bubbles."""
    model, _ = dd.load_model('quadratic_hawkes_alpha')
    want, npar = _pipeline(model, ('n', 1), ('v', 2))
    sp = one_loop_spectrum(model, mean_field='solve', values=npar)
    _close(sp.evaluate(('n', 1), ('v', 2), OMEGA), want)
    # values= that move the saddle (w21 0.2 -> 0.35) feed its solve
    want, npar = _pipeline(model, ('n', 1), ('v', 2),
                           parameters={'w': [[0.25, 0.25], [0.35, 0.3]]})
    sp = one_loop_spectrum(model, mean_field='solve', values=npar)
    for c0 in ('rational', 'residues'):
        _close(sp.evaluate(('n', 1), ('v', 2), OMEGA, c0=c0), want)


@pytest.mark.parametrize('name, pair', [
    ('single_population_quad_exp_test', (('n', 1), ('v', 2))),
    ('single_population_spike_reset_test', (('v', 1), ('v', 2)))])
def test_built_at_moved_values_evaluates_without_values(name, pair):
    """Built with ``values=`` off the defaults (every one moving the saddle),
    then evaluated with no values: the same as ``dd.run(parameters=...)``."""
    from test_symbolic_one_loop_mean import MOVED_BUILTINS
    model, _ = dd.load_model(name)
    want, npar = _pipeline(model, *pair, parameters=dict(MOVED_BUILTINS)[name])
    sp = one_loop_spectrum(model, mean_field='solve', values=npar)
    _close(sp.evaluate(*pair, OMEGA), want)
    with pytest.raises(SymbolicOutError, match='Em1 was fixed'):
        sp.evaluate(*pair, OMEGA, values={'Em1': 1.0})


@pytest.mark.parametrize('a, b', [(('n', 1), ('n', 1)), (('n', 1), ('v', 2))])
def test_spike_reset_bubble_takes_the_ito_value(a, b):
    """The reset ``v n`` meets a spike train's ``delta(tau)`` times the
    retarded ``G_vv``, which is zero at the tie (Itô); the bubble is the
    integral of the parts that decay."""
    model, _ = dd.load_model('single_population_spike_reset_test')
    want, npar = _pipeline(model, a, b)
    sp = one_loop_spectrum(model, mean_field='solve', values=npar)
    for c0 in ('rational', 'residues'):
        _close(sp.evaluate(a, b, OMEGA, c0=c0), want)
    assert any('_inf' in str(sp.sigma(l, r)) for l, r in sp.pairs())   # named delta weights


def test_instantaneous_partner_bubble_and_tadpole():
    """``y = h + c x + eta``: the contact term of ``C0_yy`` times ``G_xx``,
    and the tadpole ``C0_xy(0)`` that jumps at the tie."""
    from test_symbolic_one_loop_mean import _instantaneous_partner
    model = _instantaneous_partner('x', 'y')
    for a, b in ((('x', 1), ('x', 1)), (('x', 1), ('y', 1))):
        want, _ = _pipeline(model, a, b)
        sp = one_loop_spectrum(model, mean_field='solve')
        for c0 in ('rational', 'integral', 'residues'):
            _close(sp.evaluate(a, b, OMEGA, c0=c0), want)


def test_dendritic_spike_train_spectrum():
    model, _ = dd.load_model('dendritic_quad_soma_sigmoid')
    want, npar = _pipeline(model, ('nS', 1), ('vD', 1))
    sp = one_loop_spectrum(model, mean_field='solve', values=npar)
    _close(sp.evaluate(('nS', 1), ('vD', 1), OMEGA), want)


def test_delta_squared_bubble_is_refused():
    """Two lines with a ``delta(tau)`` part: ``C0_yy`` twice."""
    from daedalus.api.model import TemporalModelBuilder
    b = (TemporalModelBuilder('delta squared').population('E', size=1)
         .physical_field('x', population='E').physical_field('y', population='E'))
    for k, v in (('w', 0.3), ('lam', 0.2), ('h', 0.5), ('c', 0.4), ('g', 0.1),
                 ('D', 0.1)):
        b = b.parameter(k, default=[v], indexed_by=['E'], domain='real')
    model = (b.set_action_text('''
                sum( xt[i]*((Dt + 1)*x[i] - w[i]*y[i] - lam[i]*y[i]^2)
                   + yt[i]*(y[i] - h[i] - c[i]*x[i] - g[i]*x[i]*y[i])
                   - D[i]*yt[i]^2/2 for i in E)''')
             .set_mf_equation('xstar', 'w[i]*ystar[i] + lam[i]*ystar[i]^2')
             .set_mf_equation('ystar', '(h[i] + c[i]*xstar[i])/(1 - g[i]*xstar[i])')
             .build())
    with pytest.raises(UnsupportedModelError, match=r'delta\(tau\)\^2'):
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
        one_loop_spectrum(model).evaluate(('v', 1), ('v', 1), 0.5)   # no saddle
    with pytest.raises(SymbolicOutError, match='kind must be'):
        sp.by_kind('ladder')
    assert isinstance(sp.sympy(('v', 1), ('v', 1)), sympy.Expr)
    assert r'\Sigma_{vt1, dv1}' in sp.latex()
    assert sp.sigma_matrix().nrows() == 4
    assert 'bubble' in str(sp)


@pytest.mark.parametrize('a, b', [(('x', 1), ('x', 1)), (('x', 1), ('y', 1))])
def test_equal_time_by_diagram_sums_to_the_cumulant(a, b):
    """``C^{(1)}_ab(0)`` diagram by diagram: the sum is the pipeline's and
    the symbolic cumulant's ``k = 2``, ``ell = 1`` value at ``tau = 0``."""
    model = _cross()
    r = dd.run(model, dd.Config(k=2, max_ell=1, tau_grid=[0.0],
                                external_fields=[a, b]))
    want = complex(np.ravel(r['C_tau_by_ell'][1])[0]).real
    sp = one_loop_spectrum(model, mean_field='solve')
    terms = sp.equal_time(a, b)
    assert {t.kind for t, _ in terms} >= {'mean_shift', 'bubble'}
    total = sum(v for _, v in terms)
    assert abs(total - want) <= 1e-10 * abs(want)
    sc = dd.symbolic_cumulant(model, k=2, max_ell=1, external_fields=[a, b])
    assert abs(complex(sc.evaluate(ell=1)) - total) <= 1e-10 * abs(want)


def test_equal_time_refused_where_the_correlation_jumps():
    """``C_xy(tau)`` of the white-noise partner jumps at the tie at one loop
    too; ``C_xx`` does not."""
    from test_symbolic_one_loop_mean import _instantaneous_partner
    model = _instantaneous_partner('x', 'y')
    sp = one_loop_spectrum(model, mean_field='solve')
    with pytest.raises(UnsupportedModelError, match='jump at tau = 0'):
        sp.equal_time(('x', 1), ('y', 1))
    r = dd.run(model, dd.Config(k=2, max_ell=1, tau_grid=[0.0],
                                external_fields=[('x', 1), ('x', 1)]))
    want = complex(np.ravel(r['C_tau_by_ell'][1])[0]).real
    total = sum(v for _, v in sp.equal_time(('x', 1), ('x', 1)))
    assert abs(total - want) <= 1e-10 * abs(want)
