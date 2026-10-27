"""The exact stationary mean relation (``daedalus.api.symbolic_mean``).

Rate network ``tau_l dv_l = -(v_l - b_l) + sum_k W_lk phi(v_k) + noise``,
``phi = v^2 / a``: the relation is ``(1 - W diag phi'(v*)) dv = W diag(1/a)
(Var v + dv^2)`` with no ``1/tau``, and its loop orders reproduce the
pipeline's k = 1 means at one and two loops.  A cubic drift is checked
against its exact stationary density (all orders), a non-polynomial drift
against the pipeline with the truncation reported.
"""
import numpy as np
import pytest
from sage.all import SR, matrix, identity_matrix, diagonal_matrix

import daedalus as dd
from daedalus.api.noise import Gaussian
from daedalus.api.sde import SDE
from daedalus.api.symbolic_out import SymbolicOutError, UnsupportedModelError
from daedalus.api.symbolic_mean import mean_relation

TAU0 = np.array([0.0])
W2 = [[0.2, 0.15], [0.1, 0.25]]
W3 = [[0.2, 0.1, 0.05], [0.1, 0.25, 0.1], [0.05, 0.15, 0.2]]


def rate_network(W, tau):
    n = len(W)
    return (SDE('rate network', n_populations=n)
            .parameter('tau', indexed=True, default=list(tau), domain='positive')
            .parameter('b', indexed=True, default=[0.5, 0.4, 0.6][:n])
            .parameter('a', default=2.0, domain='positive')
            .parameter('sig', default=0.3, domain='positive')
            .connectivity('W', W)
            .field('v')
            .noise('xi', Gaussian(var='sig^2'), into={'v': 1})
            .equation('v', lhs='(tau[i]*Dt + 1)*v',
                      rhs='b[i] + sum(W[i, j]*v[j]^2/a for j in pop)')).build()


def _value(res, ell):
    return complex(np.ravel(res['C_tau_by_ell'][ell])[0]).real


def _pipeline(model, n, max_ell):
    """k = 1 means to ``max_ell`` loops and the equal-time variances to
    ``max_ell - 1`` loops, unit by unit."""
    mean, var, npar = {}, {}, None
    for i in range(1, n + 1):
        r1 = dd.run(model, dd.Config(k=1, max_ell=max_ell, tau_grid=TAU0,
                                     external_fields=[('v', i)]))
        r2 = dd.run(model, dd.Config(k=2, max_ell=max_ell - 1, tau_grid=TAU0,
                                     external_fields=[('v', i), ('v', i)]))
        mean[i] = {l: _value(r1, l) for l in range(1, max_ell + 1)}
        var[i] = {l: _value(r2, l) for l in range(max_ell)}
        npar = {str(k): float(v) for k, v in r1['num_params'].items()}
    return mean, var, npar


@pytest.mark.parametrize('W', [W2, W3])
def test_network_relation_is_the_stated_form(W):
    n = len(W)
    rel = mean_relation(rate_network(W, [1.0, 2.0, 1.5][:n]))
    assert rel.exact and 'polynomial of degree 2' in rel.truncation
    Ws = matrix(SR, n, n, lambda i, j: SR.var(f'W{i + 1}{j + 1}'))
    a = SR.var('a')
    vs = [SR.var(f'vstar{i + 1}') for i in range(n)]
    dv = [rel.kappa(f'dv{i + 1}') for i in range(n)]
    var = [rel.kappa(f'dv{i + 1}', f'dv{i + 1}') for i in range(n)]
    lhs = (identity_matrix(SR, n) - Ws * diagonal_matrix([2 * v / a for v in vs])) * matrix(dv).T
    rhs = Ws * diagonal_matrix([1 / a] * n) * matrix([var[i] + dv[i] ** 2 for i in range(n)]).T
    for r, eq in enumerate(rel.equations()):
        assert not any(str(v).startswith('tau') for v in eq.variables())
        assert ((eq.lhs() - eq.rhs()) - (lhs[r, 0] - rhs[r, 0])).expand() == 0
    # one loop: dv_1 = (1 - W diag phi')^-1 W diag(1/a) Var_tree v
    A = identity_matrix(SR, n) - Ws * diagonal_matrix([2 * v / a for v in vs])
    var0 = matrix([rel.kappa(f'dv{i + 1}', f'dv{i + 1}', ell=0) for i in range(n)]).T
    want = A.inverse() * Ws * diagonal_matrix([1 / a] * n) * var0
    got = rel.loop_order(1)
    for i in range(n):
        assert (got[f'dv{i + 1}'] - want[i, 0]).simplify_rational() == 0


def test_network_loop_orders_match_the_pipeline_two_units():
    """Distinct tau; at two loops the dv^2 term enters (kappa_1^(1)^2)."""
    model = rate_network(W2, [1.0, 2.0])
    mean, var, npar = _pipeline(model, 2, max_ell=2)
    rel = mean_relation(model)
    cum = {}
    for i in (1, 2):
        cum.update({f'kappa_dv{i}_dv{i}__l0': var[i][0], f'kappa_dv{i}_dv{i}__l1': var[i][1],
                    f'kappa_dv{i}__l1': mean[i][1]})
    two = rel.loop_order(2)
    assert 'kappa_dv1__l1' in rel.cumulant_symbols(two['dv1'])
    for ell in (1, 2):
        got = rel.evaluate(rel.loop_order(ell), npar, cum)
        for i in (1, 2):
            assert abs(mean[i][ell]) > 1e-6
            assert got[f'dv{i}'].real == pytest.approx(mean[i][ell], rel=1e-10)


def test_network_one_loop_matches_the_pipeline_three_units():
    model = rate_network(W3, [1.0, 2.0, 1.5])
    mean, var, npar = _pipeline(model, 3, max_ell=1)
    rel = mean_relation(model)
    cum = {f'kappa_dv{i}_dv{i}__l0': var[i][0] for i in (1, 2, 3)}
    got = rel.evaluate(rel.loop_order(1), npar, cum)
    for i in (1, 2, 3):
        assert got[f'dv{i}'].real == pytest.approx(mean[i][1], rel=1e-10)


def test_exact_relation_against_the_stationary_density():
    """dx = (-x + g x^2 - h x^3) dt + sqrt(2D) dW: the density is
    exp(-U/D), so the relation holds with the exact moments, all orders."""
    from scipy.integrate import quad
    g, h, D = 0.6, 0.5, 0.4
    model = (SDE('cubic drift').parameter('g', default=g).parameter('h', default=h)
             .parameter('D', default=D).field('x').drift('x', '-x + g*x^2 - h*x^3')
             .noise('xi', Gaussian(var='2*D'), into={'x': 1})).build()
    rel = mean_relation(model, mean_field='solve')
    assert rel.exact and rel.values == {'xstar1': 0}
    U = lambda x: (x ** 2 / 2 - g * x ** 3 / 3 + h * x ** 4 / 4) / D
    Z = quad(lambda x: np.exp(-U(x)), -10, 10)[0]
    mu = [quad(lambda x, k=k: x ** k * np.exp(-U(x)), -10, 10, epsabs=1e-14,
               epsrel=1e-13)[0] / Z for k in (1, 2, 3)]
    k1, k2 = mu[0], mu[1] - mu[0] ** 2
    k3 = mu[2] - 3 * mu[1] * mu[0] + 2 * mu[0] ** 3
    eq = rel.equations()[0]
    assert {'kappa_dx1', 'kappa_dx1_dx1', 'kappa_dx1_dx1_dx1'} <= set(rel.cumulant_symbols(eq.rhs()))
    res = rel.evaluate(eq.lhs() - eq.rhs(), {'g': g, 'h': h, 'D': D},
                       {'kappa_dx1': k1, 'kappa_dx1_dx1': k2, 'kappa_dx1_dx1_dx1': k3})
    assert abs(res) < 1e-12 * abs(k1)
    # a wrong third cumulant breaks it
    bad = rel.evaluate(eq.lhs() - eq.rhs(), {'g': g, 'h': h, 'D': D},
                       {'kappa_dx1': k1, 'kappa_dx1_dx1': k2, 'kappa_dx1_dx1_dx1': 0})
    assert abs(bad) > 1e-3 * abs(k1)


def test_non_polynomial_drift_is_truncated_and_reported():
    model = (SDE('exponential drift').parameter('a', default=0.3).parameter('sig', default=0.3)
             .parameter('b', default=-1.0).field('x').drift('x', '-x + b + a*exp(x)')
             .noise('xi', Gaussian(var='sig^2'), into={'x': 1})).build()
    with pytest.raises(SymbolicOutError, match='give order='):
        mean_relation(model)
    rel2 = mean_relation(model, order=2)
    assert not rel2.exact and 'Taylor-truncated at drift degree 2' in rel2.truncation
    assert rel2.truncation in rel2.notes
    with pytest.raises(SymbolicOutError, match='needs the drift to degree 4'):
        rel2.loop_order(2)
    rel4 = mean_relation(model, order=4)
    cfg = lambda k, ell: dd.Config(k=k, max_ell=ell, tau_grid=TAU0, external_fields=['x'] * k)
    r1, r2, r3 = dd.run(model, cfg(1, 2)), dd.run(model, cfg(2, 1)), dd.run(model, cfg(3, 0))
    npar = {str(k): float(v) for k, v in r1['num_params'].items()}
    cum = {'kappa_dx1_dx1__l0': _value(r2, 0), 'kappa_dx1_dx1__l1': _value(r2, 1),
           'kappa_dx1__l1': _value(r1, 1), 'kappa_dx1_dx1_dx1__l0': _value(r3, 0)}
    assert rel2.evaluate(rel2.loop_order(1), npar, cum)['dx1'].real == pytest.approx(_value(r1, 1), rel=1e-10)
    two = rel4.loop_order(2)
    assert 'kappa_dx1_dx1_dx1__l0' in rel4.cumulant_symbols(two['dx1'])
    assert rel4.evaluate(two, npar, cum)['dx1'].real == pytest.approx(_value(r1, 2), rel=1e-10)


def test_poisson_model_with_a_kernel_in_the_linear_sector():
    """K(omega = 0) carries the kernel's integral; the shot-noise sector does
    not enter."""
    model, _ = dd.load_model('quadratic_hawkes_alpha')
    rel = mean_relation(model)
    assert rel.exact
    eqs = {str(e.lhs() - e.rhs()) for e in rel.equations()}
    a1, v1 = SR.var('a1'), SR.var('vstar1')
    want = (rel.kappa('dn1') - 2 * a1 * v1 * rel.kappa('dv1')
            - a1 * (rel.kappa('dv1', 'dv1') + rel.kappa('dv1') ** 2))
    assert any((SR(e) - want).expand() == 0 for e in eqs)
    w11, w12 = SR.var('w11'), SR.var('w12')
    want_v = rel.kappa('dv1') - w11 * rel.kappa('dn1') - w12 * rel.kappa('dn2')
    assert any((SR(e) - want_v).expand() == 0 for e in eqs)


def test_refusals():
    m, _ = dd.load_model('ou_quartic')
    with pytest.raises(UnsupportedModelError, match='singular'):
        mean_relation(m, values={'mu': 0}, mean_field={'xstar1': 0})
    with pytest.raises(SymbolicOutError, match='xstar1, tau2'):
        mean_relation(m, mean_field={'xstar': 0})
    with pytest.raises(UnsupportedModelError, match='spatial'):
        mean_relation(dd.load_model('kpz_1d')[0])
    from daedalus.api.model import TemporalModelBuilder
    b = TemporalModelBuilder('kernel on a product')
    b.population('P', size=1)
    for f in ('x', 'y'):
        b.physical_field(f, population='P')
    for p, d in (('mu', 1.0), ('nu', 0.7), ('eps', 0.3), ('h', 0.5), ('hx', 0.4), ('D', 0.5), ('taug', 2.0)):
        b.parameter(p, default=d, domain='positive')
    b.set_mf_equation('ystar', 'h/nu')
    b.set_mf_equation('xstar', 'hx/(mu - eps*h/nu)')
    b.define_kernel('g', time_expr='(1/taug)*exp(-t/taug)*heaviside(t)', latex_name='g')
    b.set_action_text('sum(xt[i]*((Dt+mu)*x[i] - hx - eps*x[i]*Conv(g,y[i])) '
                      '+ yt[i]*((Dt+nu)*y[i] - h) - D*xt[i]^2 - D*yt[i]^2 for i in P)')
    with pytest.raises(UnsupportedModelError, match='kernel'):
        mean_relation(b.build())
    with pytest.raises(SymbolicOutError, match='ell >= 1'):
        mean_relation(m).loop_order(0)


def test_exports():
    import sympy
    rel = dd.mean_relation(rate_network(W2, [1.0, 2.0]))
    eqs = rel.sympy()
    assert len(eqs) == 2 and all(isinstance(e, sympy.Eq) for e in eqs)
    assert sympy.Symbol('kappa_dv1_dv1') in eqs[0].rhs.free_symbols
    tex = rel.latex()
    assert r'\kappa(\delta v_{1}, \delta v_{1})' in tex and r'\langle \delta v_{1} \rangle' in tex
    assert r'\kappa^{(0)}(\delta v_{1}, \delta v_{1})' in rel.latex(1)
    assert set(rel.sympy(1)) == {'dv1', 'dv2'}
    assert 'MeanRelation' in str(rel)
    with pytest.raises(SymbolicOutError, match='no value'):
        rel.evaluate(rel.loop_order(1), {}, {})
