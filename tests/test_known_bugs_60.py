"""Item 60: known-bugs sweep -- regression tests (public inline models only).

Bug 2 (derivative atoms in ``taylor()``) is pinned in ``tests/test_leftovers_39.py``
next to the tests it replaced.
"""
import os
import sys
import warnings

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from api.sde import SDE                                    # noqa: E402
from api.noise import Gaussian                             # noqa: E402
from api import compute_cumulants                          # noqa: E402

RUN = dict(parallel=False, verbose=False, use_cache=False)


# ── bug 1: a delta inside the heaviside product of a time-domain kernel ──────

def _kernel_model(**kernel):
    return (SDE('bug60 kernel delta').parameter('c', default=0.3)
            .parameter('tau', default=2.0, domain='positive')
            .parameter('D', default=0.2, domain='positive').field('x')
            .define_kernel('K', **kernel)
            .equation('x', lhs='Dt*x + 2*x', rhs='0.5 + 0.4*Conv(K, x) - 0.1*x^2')
            .noise('xi', Gaussian(var='2*D'), couples={'x': 1}))


def test_bug1_delta_inside_heaviside_end_to_end():
    """Three spellings of the same kernel (delta inside / outside the step,
    frequency image) give the same saddle, tree and one loop (fixed on main by
    #20; the DC gain is pinned in test_leftovers_39)."""
    forms = [dict(time_expr='(c*dirac_delta(t) + (1-c)*exp(-t/tau)/tau)*heaviside(t)'),
             dict(time_expr='c*dirac_delta(t) + (1-c)*exp(-t/tau)/tau*heaviside(t)'),
             dict(freq_image='c + (1-c)/(1 + I*omega*tau)')]
    out = []
    for kw in forms:
        r = compute_cumulants(_kernel_model(**kw).build(), k=2, max_ell=1,
                              external_fields=[('x', 1), ('x', 1)],
                              tau_grid=np.array([0.0, 1.0]), **RUN)
        out.append([np.asarray(r['C_tau_by_ell'][ell]) for ell in (0, 1)])
    for o in out[1:]:
        for ell in (0, 1):
            np.testing.assert_allclose(o[ell], out[0][ell], rtol=1e-13, atol=0)


# ── bug 4: placement-wrapped contributions in the symbolic loop ──────────────

def test_bug4_symbolic_loop_refuses_placement_wrappers_by_name():
    from api.symbolic_loop import _phase_j_parts_or_refuse, SymbolicLoopRefused

    def placed(*a):
        return 0.0
    placed.phase_j_placements = ((0.5, lambda *a: 0.0), (0.5, lambda *a: 0.0))
    with pytest.raises(SymbolicLoopRefused) as ei:
        _phase_j_parts_or_refuse(placed)
    assert 'kernel_edge' in str(ei.value) and 'phase_j_placements' in str(ei.value)

    def plain(*a):
        return 0.0
    with pytest.raises(SymbolicLoopRefused, match='no subset structure'):
        _phase_j_parts_or_refuse(plain)
    plain.phase_j_parts = {'subsets': []}
    assert _phase_j_parts_or_refuse(plain) == {'subsets': []}


# ── bug 5: G6 sees a broken rounded realization ──────────────────────────────

def _delayed_ou(name):
    return (SDE(name).parameter('a', default=1.0).parameter('c', default=0.5)
            .parameter('d', default=1.0, domain='positive')
            .parameter('D', default=0.1, domain='positive').field('x')
            .equation('x', lhs='Dt*x + a*x', rhs='c*x(t - d)')
            .noise('xi', Gaussian(var='2*D'), couples={'x': 1}))


def _g6_run(name, order):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return compute_cumulants(_delayed_ou(name).build(), k=2, max_ell=0,
                                 external_fields=[('x', 1), ('x', 1)],
                                 tau_grid=np.array([0.0, 1.0]),
                                 delay={'order': order, 'tree': 'pade'}, **RUN)


def test_bug5_graded_realization_measured_and_harmless():
    """A graded Pade-8 realization is checked a posteriori; its measured error
    is at rounding level and G6 stays quiet."""
    r = _g6_run('bug60 g6 graded', 8)
    rc = r['phase_j_engine']['realization_check']
    assert rc['checked'] and rc['graded']
    assert rc['err_rel'] is not None and rc['err_rel'] < 1e-12
    assert not r['analysis'].has('G6_ACCURACY_LOW')


def test_bug5_broken_rounded_realization_reaches_g6(monkeypatch):
    """Synthetic: the rounded Lyapunov solution off by 1e-3 passes the 1e-2
    refusal check but its measured error now enters G6's estimate."""
    import api._realization as R
    from engine.integration.time_domain import statespace as ss
    orig = R.lyapunov_refined
    monkeypatch.setattr(R, 'lyapunov_refined',
                        lambda A, Q, **k: orig(A, Q, **k) * (1 + 1e-3))
    monkeypatch.setattr(R, 'GRADED_BALANCE_RANGE', 1.0)    # check this small model
    ss._clear_caches()
    try:
        r = _g6_run('bug60 g6 broken', 4)
    finally:
        ss._clear_caches()
    rc = r['phase_j_engine']['realization_check']
    assert rc['checked'] and abs(rc['err_rel'] - 1e-3) < 1e-4
    assert r['analysis'].has('G6_ACCURACY_LOW')
    assert r['analysis'].diagnostics['accuracy_estimate'] >= 1e-3 * 0.9


# ── bug 7: standalone analyze() agrees with the in-run verdict ───────────────

def test_bug7_standalone_analyzer_makes_the_route_prediction(monkeypatch):
    from api.analyze import analyze, Question
    monkeypatch.setenv('DAEDALUS_POLE_CLUSTERS', '0')        # an unresolved cluster
    s = (SDE('bug60 g3 standalone').parameter('a', default=1.0)
         .parameter('D', default=0.1, domain='positive').field('x').field('y')
         .equation('x', lhs='Dt*x + a*x', rhs='0.5*y - 0.1*x^2')
         .equation('y', lhs='Dt*y + a*y', rhs='0')
         .noise('xi', Gaussian(var='2*D'), couples={'x': 1})
         .noise('eta', Gaussian(var='2*D'), couples={'y': 1}))
    m = s.build()
    ext = [('x', 1), ('x', 1)]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        r = compute_cumulants(m, k=2, max_ell=1, external_fields=ext,
                              tau_grid=np.array([0.5]), **RUN)
    inrun = {f.code: f.level for f in r['analysis'].findings if f.code.startswith('G3')}
    alone = {f.code: f.level for f in analyze(m, Question(k=2, max_ell=1,
                                                          external_fields=ext)).findings
             if f.code.startswith('G3')}
    assert inrun.get('G3_POLE_CLUSTER_UNRESOLVED') == 'info'
    assert alone == inrun


# ── bug 8: a population named I in the simulator ─────────────────────────────

def _pp_pop(pop):
    return (SDE('bug60 pop ' + pop).population(pop, size=2)
            .parameter('w', default=[[0.0, 0.3], [0.3, 0.0]], indexed_by=[pop, pop])
            .parameter('tau', default=1.0, domain='positive')
            .define_kernel('g', freq_image='1/(1 + I*omega*tau)')
            .define_function('amp', args=['u'], expression='abs(1 + exp(I*u))',
                             population=pop)
            .point_process('n', f'exp(-1.5 + 0.2*amp(0.7) + sum(w[i,j]*Conv(g, n[j]) '
                                f'for j in {pop}))', population=pop))


def test_bug8_population_named_I_keeps_the_imaginary_unit():
    kw = dict(T=50, dt=0.01, n_paths=5, burn_in=5, bin=0.1, seed=1)
    a = _pp_pop('E').simulate(**kw)['counts']['n']
    b = _pp_pop('I').simulate(**kw)['counts']['n']
    assert np.array_equal(a, b)
    from api.pp_simulate import _ImagPopulation
    z = _ImagPopulation(3)
    assert list(z) == [0, 1, 2] and z * 2.0 == 2j and abs(np.exp(1j * 0) - 1) == 0


# ── bug 10: delay in dd.Config / config files; dd.run takes an SDE ───────────

def test_bug10_config_delay_and_sde_model(tmp_path):
    import daedalus as dd
    from api.config_run import run_config, ConfigError
    s = (SDE('bug60 cfg delayed').parameter('a', default=1.0).parameter('c', default=0.5)
         .parameter('d', default=1.0, domain='positive')
         .parameter('D', default=0.1, domain='positive').field('x')
         .equation('x', lhs='Dt*x + a*x', rhs='0.3 + c*x(t - d) - 0.1*x^3')
         .noise('xi', Gaussian(var='2*D'), couples={'x': 1}))
    tau = np.array([0.0, 1.0])
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        direct = compute_cumulants(s.build(), k=2, max_ell=0, external_fields=[('x', 1), ('x', 1)],
                                   tau_grid=tau, delay={'order': 4}, **RUN)
        via = dd.run(s, dd.Config(k=2, max_ell=0, external_fields=[('x', 1), ('x', 1)],
                                  tau_grid=tau, delay={'order': 4}))
    assert via['delay']['n'] == 4
    np.testing.assert_array_equal(np.asarray(via['C_tau']), np.asarray(direct['C_tau']))
    cfg = tmp_path / 'c.yaml'
    body = '''sde:
  name: bug60 cfg delayed
  parameters: {a: 1.0, c: 0.5, d: {default: 1.0, domain: positive}, D: {default: 0.1, domain: positive}}
  fields: [x]
  equations: {x: {lhs: "Dt*x + a*x", rhs: "0.3 + c*x(t - d) - 0.1*x^3"}}
  noise: {xi: {type: Gaussian, var: "2*D", couples: {x: 1}}}
question: {k: 2, max_ell: 0, external_fields: [[x, 1], [x, 1]], tau_grid: [0.0, 1.0], delay: DELAY}
policy: {use_cache: false}
'''
    cfg.write_text(body.replace('DELAY', '{order: 4}'))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        res, resolved = run_config(str(cfg))
    assert resolved['question']['delay'] == {'order': 4}
    np.testing.assert_array_equal(np.asarray(res['C_tau']), np.asarray(direct['C_tau']))
    cfg.write_text(body.replace('DELAY', '{ordr: 4}'))
    with pytest.raises(ConfigError, match="did you mean 'order'"):
        run_config(str(cfg), dry_run=True, out=open(os.devnull, 'w'))


# ── bug 11: Sage-preparsed literals in the simulators ────────────────────────

def test_bug11_sage_literals_in_simulators():
    from sage.all import Integer, RealNumber
    from api import noise_samplers as NS
    from simulations import delay_sde_sim as DS
    s = SDE('bug60 sage lits').parameter('a', default=1.0).point_process('n', 'a')
    rec = s.simulate(T=Integer(20), dt=RealNumber('0.01'), n_paths=Integer(5),
                     burn_in=Integer(1), bin=RealNumber('0.1'), seed=Integer(3))
    ref = s.simulate(T=20, dt=0.01, n_paths=5, burn_in=1, bin=0.1, seed=3)
    assert np.array_equal(rec['counts']['n'], ref['counts']['n'])
    x = NS.simulate_ou_paths(Gaussian(var='0.2'), RealNumber('1.0'), T=Integer(5),
                             dt=RealNumber('0.01'), n_paths=Integer(4),
                             rng=np.random.default_rng(0))
    assert x.shape == (4, 500)
    r = DS.simulate(lambda x, xd: -x + 0.3 * xd[0], [RealNumber('0.5')], [[1.0]], [0.0],
                    dt=RealNumber('0.01'), t_max=Integer(2), n_paths=Integer(3),
                    burn_in=Integer(1), seed=Integer(4))
    assert r['x'].shape[2] == 3


# ── bug 13: conductance field x Conv(K, spike train) via the SDE front end ───

def test_bug13_conductance_term_with_a_point_process():
    """``(Es - v)*Conv(g, n)``: the saddle equation of v references v itself;
    it is solved as a residual equation (it was emitted as an explicit
    ``vstar = ...`` substitution and failed the engine's saddle check)."""
    from scipy.optimize import brentq
    from api.delay import Kernel
    s = (SDE('bug60 conductance pp').population('E', size=1)
         .parameter('Em', default=[-1.5], indexed_by=['E'])
         .parameter('tau', default=[2.0], indexed_by=['E']).parameter('Es', default=1.0)
         .parameter('w', default=[[0.4]], indexed_by=['E', 'E'])
         .parameter('a', default=[1.0], indexed_by=['E'])
         .parameter('taug', default=[[1.0]], indexed_by=['E', 'E'])
         .define_kernel('g', kernel=Kernel.exponential('taug[i,j]'), indexed_by=['E', 'E'])
         .point_process('n', 'phi(v)', population='E').field('v', population='E')
         .define_function('phi', args=['v'], expression='a[i]*exp(v)', population='E')
         .equation('v', lhs='(tau*Dt + 1)*v',
                   rhs='Em + (Es - v)*sum(w[i,j]*Conv(g[i,j], n[j]) for j in E)'))
    assert s._star_equations(s._derive()['equations']) is None
    r = compute_cumulants(s.build(), k=2, max_ell=0, external_fields=[('n', 1), ('n', 1)],
                          tau_grid=np.array([1.0]), **RUN)
    assert r['status'] == 'ok' and np.all(np.isfinite(r['C_tau']))
    vstar = brentq(lambda v: v - (-1.5 + (1.0 - v) * 0.4 * np.exp(v)), -5, 1)
    assert abs(r['mf_values']['vstar'][0] - vstar) < 1e-10
    assert abs(r['mf_values']['nstar'][0] - np.exp(vstar)) < 1e-11


# ── bug 3: overflow / NaN never accepted by the mean-field solvers ───────────

def test_bug3_dae_residual_penalizes_nan_and_overflow(monkeypatch):
    """``np.errstate(all='ignore')`` turns an overflow into inf / NaN instead
    of an exception; the DAE residual now penalizes it (NaN compares False
    against the acceptance tolerance and could pass as a root)."""
    import api._mean_field_dae as MD
    cap = {}
    orig = MD._build_residual

    def spy(*a, **k):
        R, ex = orig(*a, **k)
        cap['R'] = R
        return R, ex
    monkeypatch.setattr(MD, '_build_residual', spy)
    s = (SDE('bug60 dae nan').parameter('D', default=0.1, domain='positive').field('x')
         .equation('x', lhs='Dt*x + x', rhs='0.5 + 0*exp(800*x)')
         .noise('xi', Gaussian(var='2*D'), couples={'x': 1}))
    r = compute_cumulants(s.build(), k=1, max_ell=0, external_fields=[('x', 1)],
                          tau_grid=np.array([0.0]), **RUN)
    assert abs(r['mf_values']['xstar'][0] - 0.5) < 1e-12
    assert cap['R'](np.array([0.5]))[0] == 0.0
    assert cap['R'](np.array([2.0]))[0] == MD._PENALTY       # 0*inf = NaN before (penalty 1e10 then)


def test_bug3_saddle_residual_check():
    from api._mean_field import _check_saddle_residual
    _check_saddle_residual('t', [1e-12, -2e-12], [0.3, 1.5])
    for bad in ([1e-3, 0.0], [float('nan'), 0.0], [1e10, 1e10]):
        with pytest.raises(RuntimeError, match='did not converge'):
            _check_saddle_residual('t', bad, [0.3, 1.5])


# ── bug 9: an invalid DAEDALUS_PHASE_J_ENGINE fails at the API entry ─────────

def test_bug9_invalid_engine_fails_before_any_work(monkeypatch):
    import api.compute as C
    monkeypatch.setenv('DAEDALUS_PHASE_J_ENGINE', 'statespce')
    called = []
    monkeypatch.setattr(C, 'build_propagator',
                        lambda *a, **k: called.append(1) or (_ for _ in ()).throw(AssertionError))
    s = (SDE('bug60 engine env').parameter('D', default=0.1, domain='positive').field('x')
         .equation('x', lhs='Dt*x + x', rhs='0.5 - 0.1*x^2')
         .noise('xi', Gaussian(var='2*D'), couples={'x': 1}))
    with pytest.raises(ValueError, match='DAEDALUS_PHASE_J_ENGINE'):
        compute_cumulants(s.build(), k=2, max_ell=1, external_fields=[('x', 1), ('x', 1)],
                          tau_grid=np.array([0.5]), **RUN)
    assert not called


# ── bug 14: a conductance vertex mixing a plain and a kernel term ────────────

def _mixed(name, drift, kernels):
    s = (SDE(name).parameter('a', default=1.0).parameter('b', default=1.0)
         .parameter('c', default=0.5).parameter('c2', default=0.3)
         .parameter('ee', default=0.3).parameter('D', default=0.1).field('x'))
    for kname, img in kernels:
        s = s.define_kernel(kname, freq_image=img)
    return s


@pytest.mark.parametrize('case', ['one_kernel', 'two_kernels'])
def test_bug14_mixed_conductance_vertex_equals_markov_form(case):
    """``-ee*x*x - c*x*Conv(h, x)``: the vertex coefficient ``c*z_h + ee``
    is integrated term by term (the plain term was convolved too: one loop
    0.49 / 0.86 relative off the Markov form before)."""
    if case == 'one_kernel':
        conv = _mixed('bug60 mixed conv', None, [('h', '1/(1 + I*omega)')]) \
            .drift('x', '-a*x + b - ee*x*x - c*x*Conv(h, x)')
        markov = _mixed('bug60 mixed markov', None, []).field('y') \
            .drift('x', '-a*x + b - ee*x*x - c*x*y').drift('y', 'x - y')
    else:
        conv = _mixed('bug60 mixed2 conv', None, [('h', '1/(1 + I*omega)'),
                                                 ('h2', '1/(1 + 2*I*omega)')]) \
            .drift('x', '-a*x + b - ee*x*x - c*x*Conv(h, x) - c2*x*Conv(h2, x)')
        markov = _mixed('bug60 mixed2 markov', None, []).field('y').field('y2') \
            .drift('x', '-a*x + b - ee*x*x - c*x*y - c2*x*y2') \
            .drift('y', 'x - y').drift('y2', '(x - y2)/2')
    out = []
    for s in (conv, markov):
        s = s.noise('xi', Gaussian(var='2*D'), couples={'x': 1})
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            r = compute_cumulants(s.build(), k=2, max_ell=1, external_fields=[('x', 1), ('x', 1)],
                                  tau_grid=np.array([0.0, 0.5, 2.0]), **RUN)
        out.append([np.real(np.asarray(r['C_tau_by_ell'][ell])) for ell in (0, 1)])
    for ell in (0, 1):
        np.testing.assert_allclose(out[0][ell], out[1][ell], rtol=1e-11,
                                   atol=1e-12 * np.max(np.abs(out[1][ell])))


def test_bug14_split_helper_refuses_a_non_product_prefactor():
    from sage.all import SR, var
    from engine.integration.time_domain import final_integral as FI
    from engine.core.vertices import ConvVertexType
    z = var('z_h')
    vt = ConvVertexType.__new__(ConvVertexType)
    vt.coefficient = SR('c*z_h + ee')
    vt.kernel_attachments = [{'symbol': z, 'leg': ('dx', 1)}]
    va = {3: vt}
    split = FI._conv_monomial_split(va, SR('4*D*(c*z_h + ee)'))
    assert [str(p) for p, _m in split] == ['4*D*ee', '4*D*c*z_h']
    assert isinstance(FI._conv_monomial_split(va, SR('4*D*(c*z_h + 2*ee)')), str)
    vt.coefficient = SR('c*z_h')
    assert FI._conv_monomial_split(va, SR('4*D*c*z_h')) is None


def test_bug14_split_vertex_triples_equal_c_tau():
    """The term-by-term parts are exposed as ``phase_j_placements`` (weight
    1): the matrix-exponential triples equal C(tau) on a mixed vertex."""
    import daedalus as dd
    s = _mixed('bug60 mixed me', None, [('h', '1/(1 + I*omega)')]) \
        .drift('x', '-a*x + b - ee*x*x - c*x*Conv(h, x)') \
        .noise('xi', Gaussian(var='2*D'), couples={'x': 1})
    tt = np.array([-2.0, -0.5, 0.3, 1.0, 3.0])
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        r = compute_cumulants(s.build(), k=2, max_ell=1, external_fields=[('x', 1), ('x', 1)],
                              tau_grid=tt, **RUN)
        me = dd.cumulant(r).me
    assert me is not None and me.complete
    for ell in (0, 1):
        ref = np.asarray(r['C_tau_by_ell'][ell])
        assert np.max(np.abs(np.asarray(me(tt, ell=ell)) - ref)) <= 1e-13 * np.max(np.abs(ref))
