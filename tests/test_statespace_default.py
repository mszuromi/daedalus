"""
tests/test_statespace_default.py
================================
S3: the state-space Phase J route is the DEFAULT engine
(``engine/integration/time_domain/statespace.py``);
``DAEDALUS_PHASE_J_ENGINE=poles`` switches back to the pole engine.  Inline
public models only:

* the switch back: ``poles`` never consults the route, and its numbers do not
  depend on a state-space run before it (discarded warm-up);
* the default route end to end (OU with cubic drift: closed form), its
  census in ``result['phase_j_engine']`` / ``result['route']`` and the
  analyzer (G6 accuracy estimate, ``G6_PHASE_J_ENGINE``, the ledger);
* the call-time fallback: a deliberately ill-suited subset (every segment
  refused as stiff) is evaluated by the pole route, counted by reason and
  warned about; ``DAEDALUS_PHASE_J_SS_FALLBACK=0`` raises instead;
* the lifted refusal: an alpha-kernel conductance in KERNEL form at equal
  and near-equal rates (the pole engine refuses both) against its Markov
  embedding;
* item 35's pre-pass: a run the pole-tuple estimate refuses (more than 8x
  the budget) runs on the route, charged its own estimate;
* the pole-cluster deformation is bypassed when the route covers the run
  (an exact double pole), and runs after all when a subset declines;
* the τ batch inside ``total_C_batch``, informational G3 findings.

Runs compared bit for bit get a discarded warm-up.
"""
import os
import sys
import warnings

import numpy as np
import pytest

pytest.importorskip('sage.all')

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from api import compute_cumulants                                    # noqa: E402
from api.compute import PhaseJEngineFallbackWarning                   # noqa: E402
from api.model import TemporalModelBuilder                           # noqa: E402
import engine.integration.time_domain.final_integral as FI           # noqa: E402
import engine.integration.time_domain.statespace as SS               # noqa: E402

ENV = SS.ENGINE_ENV
TAUS = [0.0, 0.5, 2.0]


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    monkeypatch.delenv(ENV, raising=False)
    monkeypatch.delenv(SS.FALLBACK_ENV, raising=False)
    SS._clear_caches()
    yield
    SS._clear_caches()


def _cc(model, k, ell, ext, taus=TAUS, params=None, **kw):
    kw.setdefault('use_cache', False)
    return compute_cumulants(model, k=k, max_ell=ell, external_fields=ext,
                             tau_grid=np.asarray(taus, dtype=float),
                             parameters=params, parallel=False,
                             verbose=False, **kw)


def _arr(res, ell):
    return np.asarray(res['C_tau_by_ell'][ell], dtype=complex)


# ── models (inline, public) ─────────────────────────────────────────────

def ou_cubic():
    from api.sde import SDE
    from api.noise import Gaussian
    return (SDE('s3_ou_cubic').parameter('mu', default=1.0)
            .parameter('eps', default=0.02).parameter('D', default=1.0)
            .field('x').drift('x', '-mu*x - eps*x^3')
            .noise('xi', Gaussian(var='2*D'), couples={'x': 1}).build())


def ou_colour(tc):
    """x' = -mu x + e - eps x^3, e' = -e/tc + noise: an exact double pole
    at tc = 1/mu."""
    b = TemporalModelBuilder('s3_ou_colour')
    b.population('P', size=1)
    b.physical_field('x', population='P').physical_field('e', population='P')
    for p, d in (('mu', 1.0), ('eps', 0.02), ('D', 1.0), ('tc', tc)):
        b.parameter(p, default=d, domain='positive')
    b.set_mf_equation('estar', '0')
    b.set_mf_equation('xstar', '0')
    b.set_action_text('sum(xt[i]*((Dt+mu)*x[i] - e[i] + eps*x[i]^3) '
                      '+ et[i]*((Dt + 1/tc)*e[i]) - 2*D/tc^2*et[i]^2 '
                      'for i in P)')
    return b.build()


def ou_erlang(n):
    """x' = -a x + c (g * x) - eps x^3 + noise, g an Erlang-n kernel."""
    from api.sde import SDE
    from api.noise import Gaussian
    return (SDE(f's3_ou_erlang{n}').parameter('a', default=1.0)
            .parameter('c', default=-2.0).parameter('d', default=1.0,
                                                    domain='positive')
            .parameter('eps', default=0.1).parameter('D', default=0.1)
            .field('x')
            .define_kernel('g', freq_image=f'(1 + I*omega*d/{n})^(-{n})')
            .drift('x', '-a*x + c*Conv(g, x) - eps*x^3')
            .noise('xi', Gaussian(var='2*D'), couples={'x': 1}).build())


_P_ALPHA = dict(a=1.0, b=0.3, h=0.5, tau=0.7, D=0.15)


def _alpha_base(name, tau2):
    from api.theory import TemporalTheoryBuilder
    b = (TemporalTheoryBuilder(name).population('E', size=1)
         .physical_field('x', population='E'))
    for k, v in dict(_P_ALPHA, tau2=tau2).items():
        b = b.parameter(k, default=[v], indexed_by=['E'],
                        **({'domain': 'positive'}
                           if k in ('tau', 'tau2', 'D') else {}))
    return b


def alpha_kernel(tau2):
    """x' = -a x + h - b x (g * x) + noise, g = 1/((1 + i w tau)(1 + i w
    tau2)): the alpha kernel at tau2 = tau."""
    fi = '1/((1 + I*omega*tau[i])*(1 + I*omega*tau2[i]))'
    return (_alpha_base(f's3_alpha_kernel_{tau2}', tau2)
            .define_kernel('g', indexed_by=['E'], freq_image=fi)
            .set_action_text('sum(xt[i]*((Dt + a[i])*x[i] - h[i] '
                             '+ b[i]*x[i]*Conv(g[i], x[i])) - D[i]*xt[i]^2 '
                             'for i in E)')
            .set_mf_equation('xstar', '(h[i] - b[i]*xstar[i]^2)/a[i]')
            .build())


def alpha_embedding(tau2):
    """The same model with the kernel as two linear fields u, s."""
    b = (_alpha_base(f's3_alpha_embedding_{tau2}', tau2)
         .physical_field('s', population='E')
         .physical_field('u', population='E'))
    act = ('sum(xt[i]*((Dt + a[i])*x[i] - h[i] + b[i]*x[i]*s[i]) '
           '- D[i]*xt[i]^2 + ut[i]*((tau[i]*Dt + 1)*u[i] - x[i]) '
           '+ st[i]*((tau2[i]*Dt + 1)*s[i] - u[i]) for i in E)')
    return (b.set_action_text(act).set_mf_equation('ustar', 'xstar[i]')
            .set_mf_equation('xstar', '(h[i] - b[i]*xstar[i]^2)/a[i]')
            .set_mf_equation('sstar', 'xstar[i]').build())


X2 = [('x', 1), ('x', 1)]
DX2 = [('dx', 1), ('dx', 1)]


# ── the switch back ─────────────────────────────────────────────────────

def test_engine_values_and_the_default():
    assert SS.engine_name({}) == 'statespace'
    assert SS.engine_name({ENV: 'default'}) == 'statespace'
    assert SS.engine_name({ENV: 'statespace'}) == 'statespace'
    assert SS.engine_name({ENV: 'poles'}) == 'poles'
    with pytest.raises(ValueError, match='poles'):
        SS.engine_name({ENV: 'residues'})


def test_poles_never_consults_the_route_and_is_reproducible(monkeypatch):
    monkeypatch.setenv(ENV, 'poles')
    _cc(ou_cubic(), 2, 1, X2)                               # warm-up
    a_ = _cc(ou_cubic(), 2, 1, X2)
    monkeypatch.delenv(ENV)
    d_ = _cc(ou_cubic(), 2, 1, X2)                          # the default
    assert d_['phase_j_engine']['ss_subsets'] > 0
    monkeypatch.setenv(ENV, 'poles')

    def boom(**_kw):
        raise AssertionError('the state-space route was consulted')
    monkeypatch.setattr(SS, 'subset_maker', boom)
    FI._reset_runtime_counters()
    b_ = _cc(ou_cubic(), 2, 1, X2)
    assert b_['phase_j_engine']['engine'] == 'poles'
    assert b_['phase_j_engine']['ss_subsets'] == 0
    assert not b_['analysis'].has('G6_PHASE_J_ENGINE')
    for ell in (0, 1):
        assert np.array_equal(_arr(a_, ell), _arr(b_, ell))
        # the two engines agree to rounding
        assert np.max(np.abs(_arr(a_, ell) - _arr(d_, ell))) <= (
            1e-14 * np.max(np.abs(_arr(a_, ell))))


def test_default_route_closed_form_census_and_analyzer():
    mu, eps, D = 1.0, 0.02, 1.0
    res = _cc(ou_cubic(), 2, 1, X2)
    rep = res['phase_j_engine']
    assert rep['engine'] == 'statespace' and rep['covers_run']
    assert rep['ss_subsets'] > 0 and rep['ss_declined'] == 0
    assert rep['statespace_share'] == 1.0 and rep['fallback_evals'] == 0
    assert res['route']['phase_j_engine'] is rep
    for t, v0, v1 in zip([0.0, 0.5, 2.0], _arr(res, 0), _arr(res, 1)):
        a = abs(t)
        tree = D / mu * np.exp(-mu * a)
        loop = -3 * eps * D / mu * D / mu ** 2 * (1 + mu * a) * np.exp(-mu * a)
        assert abs(v0 - tree) <= 1e-15 * abs(tree)
        assert abs(v1 - loop) <= 1e-15 * abs(loop)
    an = res['analysis']
    assert an.has('G6_PHASE_J_ENGINE') and not an.has('G6_SS_FALLBACK')
    R = an.diagnostics['accuracy_estimate']
    assert R is not None and 0 < R < 1e-12          # the route's own estimate
    assert an.diagnostics['ss_estimated'] > 0
    assert any(e.name == 'phase_j_engine' and e.satisfied for e in an.ledger)


# ── the call-time fallback ──────────────────────────────────────────────

def _refuse_every_segment(monkeypatch):
    """Every segment is 'stiff beyond the triangular route': the route
    refuses each subset with an internal vertex at call time."""
    monkeypatch.setattr(SS, 'TRI_DENSE_MAX', -1)
    monkeypatch.setattr(SS, 'TAYLOR_S_HARD', 0)


def test_late_refusal_falls_back_on_the_pole_route(monkeypatch):
    monkeypatch.setenv(ENV, 'poles')
    ref = _cc(ou_cubic(), 2, 1, X2)
    monkeypatch.delenv(ENV)
    _refuse_every_segment(monkeypatch)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        res = _cc(ou_cubic(), 2, 1, X2)
    rep = res['phase_j_engine']
    assert rep['fallback_subsets'] > 0 and rep['fallback_evals'] > 0
    assert set(rep['fallback_by_reason']) == {'stiff_segment'}
    assert any(issubclass(x.category, PhaseJEngineFallbackWarning) for x in w)
    an = res['analysis']
    assert an.has('G6_SS_FALLBACK')
    assert any(e.name == 'phase_j_engine' and e.satisfied is False
               for e in an.ledger)
    for ell in (0, 1):            # the pole route's values at those subsets
        assert np.max(np.abs(_arr(res, ell) - _arr(ref, ell))) <= (
            1e-14 * np.max(np.abs(_arr(ref, ell))))


def test_fallback_hook_tag_and_strict_switch(monkeypatch):
    _refuse_every_segment(monkeypatch)
    seen = []
    monkeypatch.setattr(FI, '_SUBSET_HOOK', seen.append)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        _cc(ou_cubic(), 2, 0, X2)
    assert seen
    assert all(p['engine'] in ('statespace', 'poles') for p in seen)
    fb = [p for p in seen if p['ss_fallback']]
    assert fb and all(p['engine'] == 'poles' and p['ss_fallback'] ==
                      'stiff_segment' for p in fb)
    monkeypatch.setattr(FI, '_SUBSET_HOOK', None)
    monkeypatch.setenv(SS.FALLBACK_ENV, '0')
    with pytest.raises(SS.SSLateRefusal, match='stiff segment'):
        _cc(ou_cubic(), 2, 0, X2)


# ── the lifted conductance-kernel refusal ───────────────────────────────

@pytest.mark.parametrize('frac', [1.0, 0.999])
def test_alpha_kernel_form_at_equal_rates_vs_embedding(monkeypatch, frac):
    tau2 = 0.7 * frac
    kern = _cc(alpha_kernel(tau2), 2, 1, DX2)
    c = kern['phase_j_engine']
    assert kern['status'] == 'ok' and c['ss_declined'] == 0
    emb = _cc(alpha_embedding(tau2), 2, 1, DX2)
    for ell in (0, 1):
        e_ = _arr(emb, ell).real
        assert np.max(np.abs(_arr(kern, ell).real - e_)) <= (
            1e-13 * np.max(np.abs(e_)))
    monkeypatch.setenv(ENV, 'poles')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        pj = _cc(alpha_kernel(tau2), 2, 1, DX2)
    assert pj['status'] == 'incomplete'         # the pole engine refuses it


# ── item 35's pre-pass ──────────────────────────────────────────────────

def test_prepass_charges_the_route_estimate(monkeypatch):
    # OU + Erlang-4 kernel at two loops: the pole-tuple estimate (2.0e6
    # tuples, 2.15 GB) is more than 8x a 100 MB budget; the route's own
    # estimate (stationary states x the tau batch, ~12 MB, + one segment
    # system, <= 65 MB) fits, and it holds far less (~0.75 MB)
    monkeypatch.setenv('PHASE_J_MEM_BUDGET_GB', '0.1')
    monkeypatch.setenv(ENV, 'poles')
    with pytest.raises(FI.PhaseJResourceError, match='8x over budget'):
        _cc(ou_erlang(4), 2, 2, X2, taus=[0.0, 0.5])
    monkeypatch.delenv(ENV)
    res = _cc(ou_erlang(4), 2, 2, X2, taus=[0.0, 0.5])
    est = res['phase_j_memory']
    # each diagram is charged the cheaper engine (a tree diagram's few pole
    # tuples are below the route's bound)
    assert est['ss_diagrams'] > est['n_diagrams'] // 2
    assert 0 < est['ss_bytes'] and est['bytes'] <= 1e8
    assert 0 < est['built_bytes'] <= est['ss_bytes']
    assert res['phase_j_engine']['ss_declined'] == 0
    assert np.all(np.isfinite(_arr(res, 2)))


# ── the pole-cluster deformation ────────────────────────────────────────

def test_exact_double_pole_bypasses_the_deformation():
    res = _cc(ou_colour(1.0), 2, 1, X2)
    pc = res['pole_clusters']
    assert pc['status'] == 'bypassed' and pc['status_poles'] == 'resolved'
    assert res['phase_j_engine']['pole_deformation'] == 'bypassed'
    assert res['phase_j_engine']['ss_evals'] == 2 * 3 * 2   # one pass
    an = res['analysis']
    f = {x.code: x for x in an.findings}
    assert f['G3_REPEATED_POLE'].level == 'info'
    assert 'informational' in f['G3_REPEATED_POLE'].message
    R = an.diagnostics['accuracy_estimate']
    assert R is not None and 0 < R < 1e-12


def test_bypass_runs_the_deformation_when_a_subset_declines(monkeypatch):
    # every subset declines the route (too many downsets): the bypass does
    # not hold, the deformation runs and the pole engine's numbers come out
    monkeypatch.setattr(SS, 'MAX_DOWNSETS', 0)
    res = _cc(ou_colour(1.0), 2, 1, X2)
    rep = res['phase_j_engine']
    assert rep['pole_deformation'] == 'used' and rep['bypass_failed']
    assert res['pole_clusters']['status'] == 'resolved'
    monkeypatch.setattr(SS, 'MAX_DOWNSETS', 50000)
    ok = _cc(ou_colour(1.0), 2, 1, X2)
    for ell in (0, 1):
        assert np.max(np.abs(_arr(res, ell) - _arr(ok, ell))) <= (
            1e-11 * np.max(np.abs(_arr(ok, ell))))


def _unresolved_clusters(monkeypatch):
    """An unresolved pole cluster: the exact residue tier unavailable (it
    fails for non-rationalizable parameters or on its time budget), so the
    deformation cannot be built.  (Before audit W7 these tests switched the
    deformation off, ``DAEDALUS_POLE_CLUSTERS=0``.)"""
    from api import _propagator as PR
    monkeypatch.setattr(PR, '_compute_residues_via_polynomial_fracfield',
                        lambda *a, **kw: (None, None, None))


def test_unresolved_cluster_is_informational_on_the_route(monkeypatch):
    # an unresolved cluster: the pole engine loses digits (G3 warns,
    # tests/test_analyzer_routing.py); the route uses no poles
    _unresolved_clusters(monkeypatch)
    near = 1 / (1 + 1e-4)
    res = _cc(ou_colour(near), 2, 1, X2)
    an = res['analysis']
    f = {x.code: x for x in an.findings}
    assert f['G3_POLE_CLUSTER_UNRESOLVED'].level == 'info'
    assert not an.has('G6_ACCURACY_LOW')
    assert an.diagnostics['accuracy_estimate'] < 1e-12


# ── the τ batch ─────────────────────────────────────────────────────────

def test_total_c_batch_batches_the_grid():
    taus = np.linspace(-4.0, 4.0, 33)
    FI._reset_runtime_counters()
    res = _cc(ou_erlang(2), 2, 1, X2, taus=taus)
    assert SS._counters()['ss_prefetch_hits'] > 0
    fn = res['total_C_by_ell'][1]
    per_point = np.array([complex(fn(0.0, t)) for t in taus])
    b_ = _arr(res, 1)
    assert np.max(np.abs(per_point - b_)) <= 1e-14 * np.max(np.abs(b_))


# ── kernel realizations (S3 review) ─────────────────────────────────────

def test_kernel_refusals_are_named():
    from sage.all import SR, I, exp, heaviside, dirac_delta
    from api._realization import RealizationRefused, realize_kernel
    from engine.core.vertices import KernelTimeImage
    w, t = SR.var('s3om'), SR.var('s3tt')
    cases = [
        ({'kernel_ft_fn': KernelTimeImage((1 + I * w / 2) / (1 + I * w), w)},
         'kernel_not_strictly_proper'),
        ({'kernel_td_fn': KernelTimeImage(dirac_delta(t) + exp(-t), t)},
         'kernel_not_strictly_proper'),
        ({'kernel_ft_fn': KernelTimeImage(1 / (1 - I * w), w)},
         'kernel_not_hurwitz'),
        ({'kernel_td_fn': KernelTimeImage(heaviside(t - 2) * exp(-t), t)},
         'kernel_shifted_step'),
        ({}, 'no_kernel_image'),
    ]
    for att, reason in cases:
        with pytest.raises(RealizationRefused) as ei:
            realize_kernel(dict(att, symbol='g'), {})
        assert ei.value.reason == reason
    # the time image's variable leaves no global assumption behind
    from sage.all import assumptions
    assert not [a for a in assumptions() if 'tkernelreal' in str(a)]


def test_graded_dressed_line_impulse():
    # a fast line (rate 500) dressed by a slow alpha kernel (rate 1/0.7): a
    # graded cascade; the impulse takes the triangular route (S3 review:
    # the trace-shifted Taylor action lost ~1.8e-14 of max|g| here)
    import types
    import mpmath as mp
    from sage.all import SR, I
    from api import _realization as RZ
    from engine.core.vertices import KernelTimeImage
    w = SR.var('s3om2')
    att = {'symbol': 'g', 'kernel_ft_fn': KernelTimeImage(
        1 / (1 + I * w * SR(7) / 10) ** 2, w)}
    r = 500.0
    ent = RZ.EntryRealization(a=np.array([[-r]]), b=np.array([1.0]),
                              c=np.array([1.0]), d=0.0, n_states=1)
    model = SS._ModelOps(types.SimpleNamespace(A=np.zeros((1, 1)),
                                               entries={(0, 0): ent}), {})
    e = model.dressed_entry(0, 0, att)
    tau = mp.mpf(7) / 10
    errs, top = [], 0.0
    with mp.workdps(30):
        for t in (0.05, 0.2, 0.54, 1.29, 3.0, 8.0):
            ref = mp.quad(lambda x: (t - x) * mp.e ** (-(t - x) / tau) / tau ** 2
                          * mp.e ** (-r * x), [0, mp.mpf(1) / 100, t])
            errs.append(abs(e.impulse(t) - float(ref)))
            top = max(top, abs(float(ref)))
    assert max(errs) <= 2e-15 * top


# ── review round 2 ──────────────────────────────────────────────────────

def test_lifted_kernel():
    # a kernel only the route realizes (equal rates, frequency image): a
    # formal function stands for it in the symbolic factors.  (Until audit
    # W7 also run on the retired legacy setup path,
    # DAEDALUS_PHASE_J_LEGACY_SETUP=1, whose eager display build had to
    # cope with it; equal to 1e-14.)
    res = _cc(alpha_kernel(0.7), 2, 1, DX2)
    assert res['status'] == 'ok'
    for g in res['phase_j_by_ell'][1]['groups']:
        assert g['handled_by'] != 'skipped'
    for ell in (0, 1):
        assert np.all(np.isfinite(_arr(res, ell)))


def test_late_refusal_after_a_bypassed_deformation_raises(monkeypatch):
    res = _cc(ou_colour(1.0), 2, 1, X2)
    assert res['pole_clusters']['status'] == 'bypassed'
    _refuse_every_segment(monkeypatch)
    with pytest.raises(SS.SSNoPoleFallback, match='deformation'):
        res['total_C_by_ell'][1](0.0, 1.7)     # a later call, a new point


def test_fallback_outside_a_run_warns(monkeypatch):
    res = _cc(ou_cubic(), 2, 1, X2)
    _refuse_every_segment(monkeypatch)
    monkeypatch.setattr(SS, '_WARNED_OUTSIDE', set())
    with pytest.warns(PhaseJEngineFallbackWarning, match='returned callable'):
        v = res['total_C_by_ell'][1](0.0, 1.3)
    monkeypatch.setenv(ENV, 'poles')
    ref = _cc(ou_cubic(), 2, 1, X2, taus=[1.3])
    assert abs(v - _arr(ref, 1)[0]) <= 1e-14 * abs(v)


def test_partial_coverage_is_reported_and_strict_keeps_g3(monkeypatch):
    # every subset declines (too many downsets) on a run predicted covered
    monkeypatch.setattr(SS, 'MAX_DOWNSETS', 0)
    res = _cc(ou_cubic(), 2, 1, X2)
    an = res['analysis']
    assert an.has('G6_SS_PARTIAL')
    assert res['phase_j_engine']['ss_subsets'] == 0
    monkeypatch.setattr(SS, 'MAX_DOWNSETS', 50000)
    # strict policy: an unresolved cluster stops the run before Phase J, as
    # on the pole engine (the levels are not lowered under strict)
    _unresolved_clusters(monkeypatch)
    from api.analyze import AnalyzerError
    with pytest.raises(AnalyzerError):
        _cc(ou_colour(1 / (1 + 1e-4)), 2, 1, X2, policy={'mode': 'strict'})


def test_pure_delta_propagator_on_the_route():
    # decoupled spike model (a pole-free, pure-δ propagator): exact zeros.
    # (Before audit W7 with structural zeros off, the retired
    # STRUCTURAL_ZEROS = False.)
    import daedalus as dd
    model, _mod = dd.load_model('single_population_linear_delta_spikes_test')
    res = compute_cumulants(
        model, k=2, max_ell=0, external_fields=[('n', 1), ('n', 2)],
        parameters={'Em': [0.8, 0.78], 'tau': [10.0, 9.0],
                    'w': [[0.0, 0.0], [0.0, 0.0]]},
        tau_grid=np.array([0.0, 2.5, -2.5]), parallel=False, verbose=False)
    assert np.array_equal(_arr(res, 0), np.zeros(3, dtype=complex))
