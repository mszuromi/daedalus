"""
tests/test_audit_w3b_front_end.py
=================================
Regression tests for the Oct 2026 audit's fix wave W3b (front end: model
builder, compiler, Markovian embedding, serializer, delay controller).
Inline public models only.

* ``declare_cgf_term(kernel=None)`` is the documented white (``prod delta``)
  cumulant, not a constant one;
* CGF rows on a multi-unit field see their leg indices (``i``, ``j``,
  ``kronecker_delta``, ``shared``): independent per-unit noise matches the
  independent-unit twin, a row that does not state its unit structure is
  refused;
* the Markovian embedding adds its coupling as a separate term: every
  action layout gives the Lyapunov covariance of the exact embedding;
  multi-unit fields are not embedded;
* a re-declared builder kernel replaces the earlier declaration (no stale
  delay);
* the model-file round trip keeps ``physical_field(domain=...)`` (the saddle
  selection) and the other field switches, and reports keywords it cannot
  carry;
* the delay order controller samples k >= 3 cumulants at the model's own
  time scale (the delay, capped by the local relaxation time: unit
  invariant) and marks a missed accuracy target in ``result['status']``.
"""
import os
import sys
import warnings

import numpy as np
import pytest

pytest.importorskip('sage.all')

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from scipy.linalg import expm, solve_continuous_lyapunov               # noqa: E402

from api import compute_cumulants                                      # noqa: E402
from api.model import TemporalModelBuilder                             # noqa: E402
from api.noise import Gaussian                                         # noqa: E402
from api.sde import SDE                                                # noqa: E402

TAUS = [0.0, 0.5, 1.3]
REPO = os.path.join(os.path.dirname(__file__), '..')


def _cc(model, ext, fund, k=2, **kw):
    kw.setdefault('tau_grid', TAUS)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return compute_cumulants(model, fundamental=fund, k=k, max_ell=0,
                                 external_fields=ext, use_cache=False,
                                 parallel=False, verbose=False, **kw)


def _real(r):
    return np.array([complex(v).real for v in r['C_tau']])


# ── ACC-04: kernel=None is the white cumulant ───────────────────────────


def _ou_cgf(kernel, mu=0.8, D=0.3):
    return (TemporalModelBuilder('OU cgf kernel')
            .physical_field('x')
            .parameter('mu', default=mu, domain='positive')
            .parameter('D', default=D, domain='positive')
            .markovianize(False)
            .declare_cgf_term('C', response_legs=['xt', 'xt'], order=2,
                              coefficient='2*D', kernel=kernel)
            .set_action_text('xt*((Dt+mu)*x)')
            .mf_residual(lhs='(Dt+mu)*x', rhs='0')
            .build())


def test_cgf_kernel_none_is_white():
    mu, D = 0.8, 0.3
    ref = D / mu * np.exp(-mu * np.array(TAUS))
    fund = {'mu': mu, 'D': D}
    for kernel in (None, '', 'dirac_delta(tau)'):
        got = _real(_cc(_ou_cgf(kernel), [('x', 1), ('x', 1)], fund))
        assert np.max(np.abs(got - ref)) < 1e-12, (kernel, got, ref)


# ── ACC-05: population leg indices ──────────────────────────────────────


def _pop(coeff, size=3, act_noise=''):
    b = (TemporalModelBuilder('pop cgf').population('P', size=size)
         .physical_field('x', population='P')
         .parameter('mu', default=0.7, domain='positive')
         .parameter('D', default=0.4, domain='positive')
         .parameter('rho', default=0.35, domain='real')
         .markovianize(False).set_mf_equation('xstar', '0'))
    if coeff is not None:
        b = b.declare_cgf_term('C', response_legs=['xt', 'xt'], order=2,
                               coefficient=coeff, kernel='dirac_delta(tau)')
    return b.set_action_text('sum(xt[i]*((Dt+mu)*x[i])%s for i in P)' % act_noise).build()


@pytest.mark.parametrize('coeff,cross_factor', [
    ('2*D*kronecker_delta(i, j)', 0.0),
    ('2*D*kronecker_delta(i1, i2)', 0.0),
    ('2*D*shared(i, j)', 1.0),
    ('2*D*(kronecker_delta(i, j) + rho*(1 - kronecker_delta(i, j)))', 0.35),
])
def test_cgf_rows_see_population_leg_indices(coeff, cross_factor):
    mu, D = 0.7, 0.4
    fund = {'mu': mu, 'D': D, 'rho': 0.35}
    ref = D / mu * np.exp(-mu * np.array(TAUS))
    m = _pop(coeff)
    auto = _real(_cc(m, [('x', 2), ('x', 2)], fund))
    cross = _real(_cc(m, [('x', 1), ('x', 3)], fund))
    assert np.max(np.abs(auto - ref)) < 1e-12
    assert np.max(np.abs(cross - cross_factor * ref)) < 1e-12


def test_cgf_independent_rows_match_the_independent_unit_twin():
    fund = {'mu': 0.7, 'D': 0.4, 'rho': 0.35}
    cgf, twin = _pop('2*D*kronecker_delta(i, j)'), _pop(None, act_noise=' - D*xt[i]^2')
    for ext in ([('x', 1), ('x', 1)], [('x', 1), ('x', 2)], [('x', 3), ('x', 2)]):
        assert np.max(np.abs(_real(_cc(cgf, ext, fund)) - _real(_cc(twin, ext, fund)))) < 1e-13


def test_cgf_row_without_unit_structure_is_refused():
    with pytest.raises(ValueError, match='kronecker_delta'):
        _cc(_pop('2*D'), [('x', 1), ('x', 2)], {'mu': 0.7, 'D': 0.4, 'rho': 0.35})
    # a single-unit field needs no index (scalar models are unchanged)
    _cc(_ou_cgf('dirac_delta(tau)'), [('x', 1), ('x', 1)], {'mu': 0.8, 'D': 0.3})


def _two_pop_cross(cxy, extra_param=None):
    b = (TemporalModelBuilder('cross pops').population('A', size=2).population('B', size=2)
         .physical_field('x', population='A').physical_field('y', population='B')
         .parameter('mu', default=0.7, domain='positive')
         .parameter('D', default=0.4, domain='positive')
         .parameter('rho', default=0.3).markovianize(False))
    if extra_param:
        b = b.parameter(extra_param, default=0.25)
    for name, legs, coeff in (('Cxx', ['xt', 'xt'], '2*D*kronecker_delta(i, j)'),
                              ('Cyy', ['yt', 'yt'], '2*D*kronecker_delta(i, j)'),
                              ('Cxy', ['xt', 'yt'], cxy)):
        b = b.declare_cgf_term(name, response_legs=legs, order=2, coefficient=coeff,
                               kernel='dirac_delta(tau)')
    return (b.set_action_text('sum(xt[i]*((Dt+mu)*x[i]) for i in pop_A) + '
                              'sum(yt[i]*((Dt+mu)*y[i]) for i in pop_B)')
            .set_mf_equation('xstar', '0').set_mf_equation('ystar', '0').build())


def test_cgf_cross_field_population_rows():
    mu, D, rho = 0.7, 0.4, 0.3
    fund = {'mu': mu, 'D': D, 'rho': rho}
    ref = D / mu * np.exp(-mu * np.array(TAUS))
    m = _two_pop_cross('2*D*rho*kronecker_delta(i, j)')
    for ext, f in (([('x', 2), ('y', 2)], rho), ([('x', 1), ('y', 2)], 0.0),
                   ([('x', 2), ('x', 2)], 1.0), ([('y', 1), ('x', 1)], rho)):
        assert np.max(np.abs(_real(_cc(m, ext, fund)) - f * ref)) < 1e-12, ext
    # an index expression that is not symmetric under exchanging the legs
    # would be averaged by the framework's reversed pass: refused
    with pytest.raises(ValueError, match='not symmetric'):
        _cc(_two_pop_cross('D*kronecker_delta(i, 0)*kronecker_delta(j, 1)'),
            [('x', 1), ('y', 2)], fund)


def test_cgf_index_name_shadowed_by_a_parameter_is_refused():
    with pytest.raises(ValueError, match='leg-index name'):
        _cc(_two_pop_cross('2*D*rho*kronecker_delta(i, j)', extra_param='j'),
            [('x', 1), ('y', 1)], {'mu': 0.7, 'D': 0.4, 'rho': 0.3, 'j': 0.25})


# ── ACC-06: Markovian embedding, layout independent ─────────────────────


def _colored(action, mu=0.6, D=0.5, tauc=0.8, s=2.5, markov=True):
    b = (TemporalModelBuilder('colored layout')
         .physical_field('x', description='v')
         .parameter('mu', default=mu, domain='positive')
         .parameter('D', default=D, domain='positive')
         .parameter('tauc', default=tauc, domain='positive')
         .parameter('s', default=s, domain='positive')
         .declare_cgf_term('CXX', response_legs=['xt', 'xt'], order=2,
                           coefficient='2*D/tauc', kernel='exp(-abs(tau)/tauc)')
         .set_action_text(action).mf_residual(lhs='(Dt+mu)*x', rhs='0'))
    if not markov:
        b = b.markovianize(False)
    return b.build(), b


def _lyapunov(mu, D, tauc, scale2):
    """x' = -mu x + xi, xi the OU process with <xi xi> = (2D/tauc) e^{-|t|/tauc}
    (noise variance times ``scale2``)."""
    c = 2 * D / tauc
    A = np.array([[-mu, 1.0], [0.0, -1 / tauc]])
    S = solve_continuous_lyapunov(A, -np.array([[0, 0], [0, 2 * c / tauc * scale2]]))
    return np.array([(expm(A * t) @ S)[0, 0] for t in TAUS])


@pytest.mark.parametrize('action,scale2', [
    ('xt*((Dt+mu)*x)', 1.0),
    ('((Dt+mu)*x)*xt', 1.0),
    ('xt*Dt*x + mu*x*xt', 1.0),
    ('xt*((Dt+mu)*x)/s', 2.5 ** 2),
])
def test_markovianize_any_action_layout(action, scale2):
    mu, D, tauc = 0.6, 0.5, 0.8
    m, b = _colored(action)
    assert b._markovianize_applied is not None
    got = _real(_cc(m, [('x', 1), ('x', 1)], {'mu': mu, 'D': D, 'tauc': tauc, 's': 2.5}))
    ref = _lyapunov(mu, D, tauc, scale2)
    assert np.max(np.abs(got - ref)) < 1e-12 * np.max(np.abs(ref)), (got, ref)


def test_markovianize_skips_multi_unit_fields():
    def build(explicit):
        return (TemporalModelBuilder('pop colored').population('P', size=2)
                .physical_field('x', population='P')
                .parameter('mu', default=0.6, domain='positive')
                .parameter('D', default=0.5, domain='positive')
                .parameter('tauc', default=0.8, domain='positive')
                .declare_cgf_term('C', response_legs=['xt', 'xt'], order=2,
                                  coefficient='2*D/tauc*kronecker_delta(i, j)',
                                  kernel='exp(-abs(tau)/tauc)', markovianize=explicit)
                .set_action_text('sum(xt[i]*((Dt+mu)*x[i]) for i in P)')
                .set_mf_equation('xstar', '0'))
    b = build(None)
    m = b.build()
    assert getattr(b, '_markovianize_applied', None) is None
    assert 'C' in m['correlated_noises']
    with pytest.raises(ValueError, match='single-unit'):
        build(True).build()


def test_markovianize_single_unit_populations_cross_correlated():
    """Two size-1 populations with a cross-correlated colored noise: the
    auxiliaries live in their source fields' populations, and the result is
    the Lyapunov covariance of the 4-d Markov embedding."""
    mu, D, rho, tc = 0.7, 0.4, 0.3, 0.9
    rows = [('Cxx', ['xt', 'xt'], 'D/tauc'), ('Cyy', ['yt', 'yt'], 'D/tauc'),
            ('Cxy', ['xt', 'yt'], 'rho*D/tauc')]
    b = (TemporalModelBuilder('two pops colored').population('A', size=1)
         .population('B', size=1)
         .physical_field('x', population='A').physical_field('y', population='B')
         .parameter('mu', default=mu, domain='positive')
         .parameter('D', default=D, domain='positive')
         .parameter('rho', default=rho).parameter('tauc', default=tc, domain='positive'))
    for name, legs, coeff in rows:
        b = b.declare_cgf_term(name, response_legs=legs, order=2, coefficient=coeff,
                               kernel='exp(-abs(tau)/tauc)')
    b = (b.set_action_text('sum(xt[i]*((Dt+mu)*x[i]) for i in pop_A) + '
                           'sum(yt[i]*((Dt+mu)*y[i]) for i in pop_B)')
         .mf_residual(lhs='(Dt+mu)*x', rhs='0', population='A')
         .mf_residual(lhs='(Dt+mu)*y', rhs='0', population='B'))
    m = b.build()
    assert b._markovianize_applied['matched_rows'] == ['Cxx', 'Cyy', 'Cxy']
    c = D / tc
    A = np.diag([-mu, -mu, -1 / tc, -1 / tc])
    A[0, 2] = A[1, 3] = 1.0
    Q = np.zeros((4, 4))
    Q[2:, 2:] = 2 * c / tc * np.array([[1, rho], [rho, 1]])
    S = solve_continuous_lyapunov(A, -Q)
    fund = {'mu': mu, 'D': D, 'rho': rho, 'tauc': tc}
    for ext, (a, bb) in (([('x', 1), ('x', 1)], (0, 0)), ([('y', 1), ('x', 1)], (1, 0))):
        got = _real(_cc(m, ext, fund))
        ref = np.array([(expm(A * t) @ S)[a, bb] for t in TAUS])
        assert np.max(np.abs(got - ref)) < 1e-12, (ext, got, ref)


# ── ACC-13: a re-declared kernel replaces the old one ───────────────────


def test_redeclared_kernel_replaces_the_old_spec():
    mu, J, D, tg = 1.0, 0.3, 1.0, 2.0

    def build(redeclare):
        b = (TemporalModelBuilder('kernel redeclared').physical_field('x')
             .parameter('mu', default=mu, domain='positive')
             .parameter('J', default=J).parameter('D', default=D, domain='positive')
             .parameter('tg', default=tg, domain='positive')
             .parameter('d', default=1.5, domain='positive'))
        if redeclare:
            b = b.define_kernel('g', delay='d')
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            b = b.define_kernel('g', freq_image='1/(1 + I*omega*tg)')
        assert any('re-declared' in str(x.message) for x in w) == redeclare
        return (b.set_action_text('xt*((Dt+mu)*x - J*Conv(g, x)) - D*xt^2')
                .mf_residual(lhs='(Dt+mu)*x', rhs='J*x').build()), b

    # the symbol keeps its place and a custom registration
    b2 = (TemporalModelBuilder('kernel symbols').kernel('h').kernel('g', sage_name='z_custom')
          .define_kernel('g', freq_image='1/(1 + I*omega)'))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        b2 = b2.define_kernel('g', freq_image='1/(1 + 2*I*omega)', indexed='vector')
    assert [(k.name, k.sage_name, k.indexed) for k in b2.kernels] == [
        ('h', 'z_h', False), ('g', 'z_custom', True)]
    m, b = build(True)
    assert [k['name'] for k in b._kernel_specs] == ['g']
    assert 'delay' not in b._kernel_specs[0] and len(b.kernels) == 1
    r = _cc(m, [('x', 1), ('x', 1)], {'mu': mu, 'J': J, 'D': D, 'tg': tg, 'd': 1.5},
            tau_grid=[0.0, 1.0])
    assert not r.get('delay')
    # Markov embedding: x' = -mu x + J y + noise, y' = (x - y)/tg
    A = np.array([[-mu, J], [1 / tg, -1 / tg]])
    S = solve_continuous_lyapunov(A, -np.diag([2 * D, 0.0]))
    ref = np.array([(expm(A * t) @ S)[0, 0] for t in (0.0, 1.0)])
    assert np.max(np.abs(_real(r) - ref)) < 1e-10


# ── ACC-11: model-file round trip ───────────────────────────────────────


def _load_file(path):
    import importlib.util
    spec = importlib.util.spec_from_file_location('w3b_rt_' + str(abs(hash(path))), path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.build()


def test_round_trip_keeps_physical_field_domain_and_saddle(tmp_path):
    from api.model_serialize import load_spec_from_file, render_model_file
    src = os.path.join(REPO, 'models', 'ou_quartic_double_well.model.py')
    spec = load_spec_from_file(src)
    assert [f.get('domain') for f in spec['physical_fields']] == ['real']
    out = tmp_path / 'rt.model.py'
    out.write_text(render_model_file(spec))
    assert "domain='real'" in out.read_text()
    assert load_spec_from_file(str(out))['physical_fields'] == spec['physical_fields']
    fund = {'mu': -1.0, 'eps': 1.0, 'D': 0.1}
    doms, stars = [], []
    for path in (src, str(out)):
        model = _load_file(path)
        doms.append({q.get('name'): q.get('domain') for q in model['parameters']
                     if q.get('mean_field')})
        r = _cc(model, [('x', 1)], fund, k=1, tau_grid=[0.0])
        stars.append(float(np.real(np.ravel(r['mf_values']['xstar'])[0])))
    assert doms[0] == doms[1] and set(doms[0].values()) == {'real'}
    # exact wells +-sqrt(-mu/eps) = +-1: the round trip selects the same root
    assert abs(abs(stars[0]) - 1.0) < 1e-9 and stars[1] == stars[0]


def test_round_trip_field_switches_and_kernel_expressions(tmp_path):
    from api.model_serialize import load_spec_from_file, render_model_file
    src = tmp_path / 'sw.model.py'
    src.write_text(
        'from api.model import TemporalModelBuilder\n'
        'from api.delay import Kernel\n\n\n'
        'def build():\n'
        '    return (\n'
        "        TemporalModelBuilder('switches', n_populations=1)\n"
        "        .response_field('yt')\n"
        "        .physical_field('y', domain='positive', auto_response=False)\n"
        "        .parameter('a', default=1.0)\n"
        "        .define_kernel('g', kernel=Kernel.delay('d') * Kernel.exponential('tau'))\n"
        "        .set_action_text('yt*((Dt+a)*y)')\n"
        '        .build()\n'
        '    )\n')
    spec = load_spec_from_file(str(src))
    f = spec['physical_fields'][0]
    assert f['domain'] == 'positive' and f['auto_response'] is False
    k = spec['kernels'][0]
    assert k['delay'] == 'd' and 'tau' in k['freq_image']
    text = render_model_file(spec)
    assert "domain='positive'" in text and 'auto_response=False' in text
    rt = tmp_path / 'sw2.model.py'
    rt.write_text(text)
    spec2 = load_spec_from_file(str(rt))
    f2 = spec2['physical_fields'][0]
    assert (f2['domain'], f2['auto_response']) == ('positive', False)
    assert spec2['kernels'][0]['delay'] == 'd'
    assert spec2['kernels'][0]['freq_image'] == k['freq_image']


def test_loader_reports_keywords_it_cannot_carry(tmp_path):
    from api.model_serialize import load_spec_from_file
    src = tmp_path / 'odd.model.py'
    src.write_text(
        'from api.model import TemporalModelBuilder\n\n\n'
        'def build():\n'
        '    return (\n'
        "        TemporalModelBuilder('odd', n_populations=1)\n"
        "        .physical_field('x', future_option=3)\n"
        "        .parameter('a', default=1/3)\n"
        "        .set_action_text('xt*((Dt+a)*x)')\n"
        '        .build()\n'
        '    )\n')
    with pytest.warns(UserWarning) as rec:
        load_spec_from_file(str(src))
    msgs = ' | '.join(str(w.message) for w in rec)
    assert 'future_option' in msgs and 'default (not a literal)' in msgs


def test_loader_never_executes_kernel_expressions(tmp_path):
    from api.model_serialize import load_spec_from_file
    marker = tmp_path / 'executed.txt'
    src = tmp_path / 'esc.model.py'
    payload = ("Kernel.image(Kernel.__init__.__globals__['__builtins__']['open']"
               f"({str(marker)!r}, 'w').write('x') or '1')")
    src.write_text(
        'from api.model import TemporalModelBuilder\n'
        'from api.delay import Kernel\n\n\n'
        'def build():\n'
        '    return (\n'
        "        TemporalModelBuilder('esc', n_populations=1)\n"
        "        .physical_field('x')\n"
        f"        .define_kernel('g', kernel={payload})\n"
        "        .set_action_text('xt*((Dt+1)*x)')\n"
        '        .build()\n'
        '    )\n')
    with pytest.raises(ValueError, match='literal arguments'):
        load_spec_from_file(str(src))
    assert not marker.exists()


def test_loader_warns_on_a_non_literal_cgf_kernel(tmp_path):
    from api.model_serialize import load_spec_from_file
    src = tmp_path / 'cgf.model.py'
    src.write_text(
        'from api.model import TemporalModelBuilder\n'
        "KSTR = 'exp(-abs(tau))'\n\n\n"
        'def build():\n'
        '    return (\n'
        "        TemporalModelBuilder('cgf', n_populations=1)\n"
        "        .physical_field('x')\n"
        "        .declare_cgf_term('C', 'xt', order=2, coefficient='1', kernel=KSTR)\n"
        "        .set_action_text('xt*((Dt+1)*x)')\n"
        '        .build()\n'
        '    )\n')
    with pytest.warns(UserWarning, match=r'kernel \(not a literal\)'):
        load_spec_from_file(str(src))


# ── ACC-14 / ACC-22: the delay order controller ─────────────────────────


def _delay_k3(s):
    return (SDE(f'delay k3 units s{s}').parameter('a', default=1.3 * s)
            .parameter('c', default=-1.4 * s).parameter('d', default=0.8 / s, domain='positive')
            .parameter('b', default=0.45 * s).parameter('D', default=0.05 * s).field('x')
            .drift('x', '-a*x + c*x(t - d) - b*x^2')
            .noise('xi', Gaussian(var='2*D'), couples={'x': 1}).build())


def test_delay_controller_k3_is_unit_invariant():
    recs = {}
    for s in (1.0, 0.25):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            r = compute_cumulants(_delay_k3(s), k=3, max_ell=0, external_fields=[('x', 1)] * 3,
                                  parallel=False, verbose=False, use_cache=False,
                                  delay={'n_start': 2, 'n_max': 8, 'tol': 1e-4})
        recs[s] = r['delay']
        # min(longest delay 0.8, slowest local relaxation 1/|-a + c| = 1/2.7), in units s
        assert r['delay']['sample_time_scale'] == pytest.approx(min(0.8, 1 / 2.7) / s, rel=1e-12)
    a, b = recs[1.0], recs[0.25]
    assert a['n'] == b['n'] and a['target_met'] == b['target_met']
    assert sorted(a['est_by_order']) == sorted(b['est_by_order'])
    for n in a['est_by_order']:
        assert b['est_by_order'][n] == pytest.approx(a['est_by_order'][n], rel=1e-9)


def test_delay_target_missed_is_marked(monkeypatch):
    import api.compute as _ac
    from engine.integration.time_domain.final_integral import PhaseJResourceError
    real = _ac.compute_cumulants

    def refuse_order_8(model, **kw):
        if (model.get('_delay_pade') or {}).get('n') == 8:
            raise PhaseJResourceError('the Phase J memory guard refused this order (test)')
        return real(model, **kw)

    monkeypatch.setattr(_ac, 'compute_cumulants', refuse_order_8)
    m = (SDE('delayed OU budget').parameter('a', default=1.0).parameter('c', default=-2.0)
         .parameter('d', default=0.5, domain='positive').parameter('eps', default=0.02)
         .parameter('D', default=0.1).field('x').drift('x', '-a*x + c*x(t - d) - eps*x^3')
         .noise('xi', Gaussian(var='2*D'), couples={'x': 1}).build())
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        r = compute_cumulants(m, k=2, max_ell=1, external_fields=[('x', 1)] * 2,
                              tau_grid=np.array([0.0, 1.0]), parallel=False, verbose=False,
                              use_cache=False, delay={'n_start': 2, 'tol': 1e-12})
    d = r['delay']
    assert r['status'] == 'target_missed'
    assert d['n'] == 4 and d['target_met'] is False and 8 in d['orders_failed']
    assert 'PHASE_J_MEM_BUDGET_GB' in d['cap']
    assert any('PHASE_J_MEM_BUDGET_GB' in str(x.message) for x in w)
