"""The one-loop mean as tadpole diagrams (``daedalus.api.symbolic_mean``).

Each tadpole is ``-G(0)_{a r} V^{(2)}_{r; pq} C0_pq(0)``.  The equal-time tree
covariance ``C0_pq(0)`` named, as its Fourier integral, exactly (rational) and
by residues: every form reproduces the pipeline's k = 1, ell = 1 mean.
"""
import os
import sys

import numpy as np
import pytest
from sage.all import SR

import daedalus as dd
from daedalus.api.noise import Gaussian
from daedalus.api.sde import SDE
from daedalus.api.symbolic_out import SymbolicOutError
from daedalus.api.symbolic_mean import one_loop_mean, mean_relation

sys.path.insert(0, os.path.dirname(__file__))
from test_symbolic_mean import rate_network, W2, W3, TAU0, _value  # noqa: E402

FORMS = ('rational', 'integral', 'residues')


def _one_loop(model, fields):
    """Pipeline k = 1 one-loop means and the run's parameters."""
    out, npar = {}, None
    for name, i in fields:
        r = dd.run(model, dd.Config(k=1, max_ell=1, tau_grid=TAU0,
                                    external_fields=[(name, i)]))
        out[f'd{name}{i}'] = _value(r, 1)
        npar = {str(k): float(v) for k, v in r['num_params'].items()}
    return out, npar


def _params(npar):
    return {k: v for k, v in npar.items() if 'star' not in k}


def _close(got, want, rtol=1e-10):
    for a, v in want.items():
        assert abs(got[a].real - v) <= rtol * abs(v), (a, got[a], v)
        assert abs(got[a].imag) <= 1e-12 * abs(v)


@pytest.fixture(scope='module')
def net2():
    model = rate_network(W2, [1.0, 2.0])
    return model, _one_loop(model, [('v', 1), ('v', 2)])


def test_tadpoles_of_the_network(net2):
    model, _ = net2
    lm = dd.one_loop_mean(model)
    a = SR.var('a')
    got = {(t.response, t.legs): t.coefficient for t in lm.tadpoles}
    want = {(f'vt{r}', (f'dv{j}', f'dv{j}')): -SR.var(f'W{r}{j}') / a
            for r in (1, 2) for j in (1, 2)}
    assert got.keys() == want.keys()
    assert all((got[k] - want[k]).simplify_rational() == 0 for k in want)
    assert lm.loops == [('dv1', 'dv1'), ('dv2', 'dv2')]
    # the named form is the relation's one-loop order, diagram by diagram
    ref = mean_relation(model).loop_order(1)
    for f, e in lm.mean_shift('named').items():
        assert (e - ref[f]).simplify_rational() == 0
    # named propagator: G0 symbols times vertex times kappa^(0)
    ms = lm.mean_shift('named', propagator='named')['dv1']
    assert {'G0_dv1_vt1', 'G0_dv1_vt2', 'kappa_dv1_dv1__l0'} <= {
        str(v) for v in ms.variables()}
    tex = lm.latex()
    assert r'\kappa^{(0)}' in tex and 'G_{' in tex and r'^{(1)}' in tex


@pytest.mark.parametrize('form', FORMS)
def test_every_form_matches_the_pipeline(net2, form):
    model, (want, npar) = net2
    lm = one_loop_mean(model, mean_field='solve')
    _close(lm.evaluate(form, _params(npar)), want)


def test_rational_form_stays_symbolic(net2):
    """Saddle and parameters symbolic; no cumulant symbol left."""
    model, (want, npar) = net2
    lm = one_loop_mean(model)
    c0 = lm.C0('dv1', 'dv1', form='rational')
    assert {str(v) for v in c0.variables()} >= {'tau1', 'tau2', 'sig', 'vstar1'}
    mean = lm.mean_shift('rational')
    assert not lm.relation.cumulant_symbols(mean['dv1'])
    _close(lm.evaluate('rational', npar), want)
    integ = lm.C0('dv1', 'dv2', form='integral')
    assert 'integrate' in str(integ) and 'omega' in str(integ)


def test_cross_vertex_closes_an_off_diagonal_loop():
    """dx = (-x + c x y + h) dt + noise, dy = (x - y)/tau dt: the loop is
    C0_xy(0), and y is driven only through x."""
    model = (SDE('cross').parameter('c', default=0.3).parameter('h', default=0.2)
             .parameter('tau', default=2.0).parameter('sig', default=0.4)
             .field('x').field('y').drift('x', '-x + c*x*y + h')
             .drift('y', '(x - y)/tau')
             .noise('xi', Gaussian(var='sig^2'), into={'x': 1})).build()
    want, npar = _one_loop(model, [('x', 1), ('y', 1)])
    lm = one_loop_mean(model, mean_field='solve')
    assert lm.loops == [('dx1', 'dy1')]
    for form in FORMS:
        _close(lm.evaluate(form, _params(npar)), want)


def test_three_units_by_residues_with_rootof():
    """Distinct tau, every parameter numeric: the C0 poles are roots of a
    cubic, so the residue form carries RootOf symbols."""
    model = rate_network(W3, [1.0, 2.0, 0.7])
    want, npar = _one_loop(model, [('v', 1), ('v', 2), ('v', 3)])
    lm = one_loop_mean(model, mean_field='solve', values=_params(npar))
    assert any(lm.modes(*l).roots for l in lm.loops)
    for form in FORMS:
        _close(lm.evaluate(form), want)


def test_point_process_model_with_kernels():
    """Spike trains with exponential synaptic kernels (K(0) carries the
    kernels): the loops are the smooth voltage entries C0_vv(0)."""
    model, _ = dd.load_model('single_population_quad_exp_test')
    want, npar = _one_loop(model, [('n', 1), ('v', 1)])
    lm = one_loop_mean(model, mean_field='solve', values=_params(npar))
    assert lm.loops == [('dv1', 'dv1'), ('dv2', 'dv2')]
    for form in FORMS:
        got = lm.evaluate(form)
        _close({a: got[a] for a in want}, want)


def test_non_polynomial_drift_is_exact_at_one_loop():
    model = (SDE('exponential drift').parameter('a', default=0.3)
             .parameter('sig', default=0.3).parameter('b', default=-1.0)
             .field('x').drift('x', '-x + b + a*exp(x)')
             .noise('xi', Gaussian(var='sig^2'), into={'x': 1})).build()
    want, npar = _one_loop(model, [('x', 1)])
    lm = one_loop_mean(model, mean_field='solve')
    assert not any('truncated' in n for n in lm.notes)
    _close(lm.evaluate('rational', _params(npar)), want)


def test_refusals_and_exports(net2):
    import sympy
    model, (want, npar) = net2
    lm = dd.one_loop_mean(model, mean_field='solve')
    with pytest.raises(SymbolicOutError, match='form must be'):
        lm.C0('dv1', 'dv1', form='closed')
    with pytest.raises(SymbolicOutError, match='no value'):
        lm.evaluate('rational')
    with pytest.raises(SymbolicOutError, match='no value'):
        lm.evaluate('residues', {'a': 2.0})
    with pytest.raises(SymbolicOutError, match="'explicit' or 'named'"):
        lm.G0('dv1', 'vt1', propagator='closed')
    assert all(isinstance(v, sympy.Expr) for v in lm.sympy().values())
    assert 'tadpoles' in str(lm)
