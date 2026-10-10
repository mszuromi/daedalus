"""Model analyzer (api/analyze.py): golden traits of every tracked model (A), a gallery of unsupported models
that must each produce their specific finding without an exception escaping (B), and non-interference: under 10 s
per model and the model dict left unchanged (C).

The two models whose cold expansion takes longer than the budget (dendritic_quad_soma_sigmoid ~5 min,
multipopulation_test ~10 s) are timed only when the pipeline's expand cache is warm; analyze reads that cache and
never writes it."""
import copy
import importlib.util
import time
import types
from pathlib import Path

import os
import pytest

pytest.importorskip('sage.all')
from sage.all import SR, sin                                                  # noqa: E402

from api import _expand_cache                                                 # noqa: E402
from api.analyze import FINDING_CODES, Finding, ModelReport, Policy, analyze  # noqa: E402
from api.model import TemporalModelBuilder                                    # noqa: E402
from api.noise import CompoundPoisson, Dirac, Gaussian                        # noqa: E402
from api.sde import SDE                                                       # noqa: E402

MODELS = Path(__file__).resolve().parent.parent / 'models'
BUDGET_S = 10.0
HEAVY = {'dendritic_quad_soma_sigmoid', 'multipopulation_test'}   # cold expand >= budget


def _load(name):
    spec = importlib.util.spec_from_file_location(name, MODELS / f'{name}.model.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _tracked(name):
    mod = _load(name)
    fund = getattr(mod, 'DEFAULT_FUNDAMENTAL', None) or None
    return mod.build(), ({'parameters': fund} if fund else None)


# ── SDE-front-end references (built here; no model files added) ───────────────────────────────

def _sde_ou_quartic():
    return (SDE('OU quartic (SDE ref)')
            .parameter('mu', default=1.0, domain='positive')
            .parameter('eps', default=0.02, domain='positive')
            .parameter('D', default=1.0, domain='positive')
            .field('x').drift('x', '-mu*x - eps*x^3')
            .noise('xi', Gaussian(var='2*D'), couples={'x': 1})
            .stability_analysis(True).build())


def _sde_multiplicative():
    return (SDE('multiplicative linear variance (SDE ref)')
            .parameter('mu', default=1.0, domain='positive').parameter('m', default=2.0, domain='positive')
            .parameter('s0', default=0.5, domain='positive').parameter('s1', default=0.3, domain='positive')
            .field('x').drift('x', '-mu*(x - m)').noise('W', Gaussian(var='s0 + s1*x'), couples={'x': 1})
            .stability_analysis(True).build())


def _sde_poisson_ou():
    s = SDE('Poisson-driven OU (SDE ref)')
    for p, d in (('mu', 1.0), ('lam', 3.0), ('a', 0.7)):
        s.parameter(p, default=d, domain='positive')
    return (s.field('x').drift('x', '-mu*x').noise('eta', CompoundPoisson('lam', Dirac('a')), couples={'x': 1})
            .stability_analysis(True).build())


SDE_REFS = {'sde_ou_quartic': _sde_ou_quartic, 'sde_multiplicative': _sde_multiplicative,
            'sde_poisson_ou': _sde_poisson_ou}


# ── A: golden traits ──────────────────────────────────────────────────────────────────────────
# Columns: n_physical, polynomial, noise_class, noise_gaussian, propagator_class, n_poles, stable, stability_source,
# time_homogeneous, n_saddles, state_dependent_noise, and the exact set of warning/error codes under the default
# (permissive) policy and question.  n_poles is the degree of the common denominator Q(omega) of G = K^-1 (lcm of
# the entries' reduced denominators); None where G3 does not apply (spatial).

_W, _CM, _SP = 'white', 'colored-markov-embedded', 'spatial'
_HAWKES = frozenset({'G0_CONVENTION_MISSING', 'G1_TAYLOR_CONSENT_MISSING'})
_LIN = 'linearization'
GOLDEN = {
    # name: (n_phys, poly, noise, gauss, prop, n_poles, stable, source, homog, n_saddles, sdn, codes)
    'ou_quartic':              (1, True, _W, True, 'rational', 1, True, _LIN, True, 1, False, frozenset()),
    'ou_sextic':               (1, True, _W, True, 'rational', 1, True, _LIN, True, 1, False, frozenset()),
    'ou_quartic_double_well':  (1, True, _W, True, 'rational', 1, True, _LIN, True, 1, False, frozenset()),
    'toy_quartic_double_well': (1, True, _W, True, 'rational', 1, True, 'poles', True, 1, False, frozenset()),
    'ou_quartic_colored':      (2, True, _CM, True, 'rational', 2, True, _LIN, True, 1, False,
                                frozenset({'G4_MARKOV_CONSENT_MISSING'})),
    'ou_quartic_two_dim_color_corr': (4, True, _CM, True, 'rational', 3, True, _LIN, True, 1, False,
                                      frozenset({'G4_MARKOV_CONSENT_MISSING'})),
    'linear_hawkes':           (4, False, _W, False, 'rational', 4, True, 'poles', True, 1, True, _HAWKES),
    'single_population_quad_exp_test': (4, False, _W, False, 'rational', 4, True, 'poles', True, 1, True,
                                        _HAWKES | {'G0_DOMAIN_BOUNDARY'}),
    'single_population_linear_delta_spikes_test': (4, False, _W, False, 'rational', 2, True, 'poles', True, 1,
                                                   True, _HAWKES | {'G0_DOMAIN_BOUNDARY'}),
    'single_population_spike_reset_test': (4, False, _W, False, 'rational', 2, True, 'poles', True, 1, True,
                                           _HAWKES),
    'quadratic_hawkes_alpha':  (4, False, _W, False, 'rational', 8, True, 'poles', True, 1, True, _HAWKES),
    'multipopulation_test':    (8, False, _W, False, 'rational', 16, True, 'poles', True, 1, True,
                                _HAWKES | {'G3_CLOSE_POLES'}),
    'dendritic_quad_soma_sigmoid': (8, False, _W, False, 'rational', 4, True, 'poles', True, 1, True, _HAWKES),
    # spatial: G3 skipped; stability is the k = 0 (homogeneous) linearization
    'allen_cahn_1d_subcritical_infinite': (1, True, _W, True, _SP, None, True, _LIN, True, 1, False, frozenset()),
    'allen_cahn_1d_subcritical_pbc':      (1, True, _W, True, _SP, None, True, _LIN, True, 1, False, frozenset()),
    'burgers_1d':                         (1, True, _W, True, _SP, None, True, _LIN, True, 1, False, frozenset()),
    'combined_allencahn_modelb_kpz_1d':   (1, True, _W, True, _SP, None, True, _LIN, True, 1, False, frozenset()),
    'coupled_rd_2species_1d':             (2, True, _W, True, _SP, None, True, _LIN, True, 1, False, frozenset()),
    'edwards_wilkinson_1d':               (1, True, _W, True, _SP, None, True, _LIN, True, 1, False, frozenset()),
    'kpz_1d':                             (1, True, _W, True, _SP, None, True, _LIN, True, 1, False, frozenset()),
    'linear_diffusion_test':              (1, True, _W, True, _SP, None, True, _LIN, True, 1, False, frozenset()),
    'reaction_diffusion_2d':              (1, True, _W, True, _SP, None, True, _LIN, True, 1, False, frozenset()),
    'reaction_diffusion_conserved_1d':    (1, True, _W, True, _SP, None, True, _LIN, True, 1, False, frozenset()),
    'reaction_diffusion_quadratic_1d':    (1, True, _W, True, _SP, None, True, _LIN, True, 1, False, frozenset()),
    # SDE front-end references
    'sde_ou_quartic':     (1, True, _W, True, 'rational', 1, True, _LIN, True, 1, False, frozenset()),
    'sde_multiplicative': (1, True, _W, True, 'rational', 1, True, _LIN, True, 1, True,
                           frozenset({'G0_CONVENTION_MISSING'})),
    'sde_poisson_ou':     (1, False, _W, False, 'rational', 1, True, _LIN, True, 1, False,
                           frozenset({'G1_TAYLOR_CONSENT_MISSING'})),
}


def _tracked_model_names():
    """The model files under version control (local-only model files on a developer's disk are not part of the claim);
    falls back to every file when git is not available (a source tarball)."""
    import subprocess
    try:
        out = subprocess.run(['git', 'ls-files', '--', str(MODELS)], capture_output=True, text=True, check=True,
                             cwd=str(MODELS.parent)).stdout.split()
        names = {os.path.basename(f)[:-len('.model.py')] for f in out if f.endswith('.model.py')}
        if names:
            return names
    except Exception:
        pass
    return {p.name[:-len('.model.py')] for p in MODELS.glob('*.model.py')}


def test_golden_table_covers_every_tracked_model():
    tracked = _tracked_model_names()
    assert tracked <= set(GOLDEN), sorted(tracked - set(GOLDEN))


def _build(name):
    if name in SDE_REFS:
        return SDE_REFS[name](), None
    return _tracked(name)


def _warm(name, model, order=4):
    return _expand_cache.find_best_cached_order(model, order) is not None


def _check_golden(name, use_cache=True):
    (n_phys, poly, noise, gauss, prop, n_poles, stable, source, homog, n_sad, sdn, codes) = GOLDEN[name]
    model, q = _build(name)
    if use_cache and name == 'dendritic_quad_soma_sigmoid' and not _warm(name, model):
        pytest.skip('cold expand takes ~5 min: warm saved_models/ with one pipeline run, or run the slow '
                    'test_golden_traits_cold')
    r = analyze(model, question=q, use_cache=use_cache)
    t = r.traits
    got = (t.n_physical, t.polynomial, t.noise_class, t.noise_gaussian, t.propagator_class, t.n_poles,
           t.stable, t.stability_source, t.time_homogeneous, t.n_saddles, t.state_dependent_noise,
           frozenset(f.code for f in r.findings if f.level != 'info'))
    assert got == GOLDEN[name], (got, r.summary())
    assert t.n_response == t.n_physical
    if prop == 'rational':
        assert t.causal is True and t.repeated_poles is False and len(t.poles) == n_poles
        assert all(p.imag > 0 for p in t.poles)


@pytest.mark.parametrize('name', list(GOLDEN))
def test_golden_traits(name):
    _check_golden(name)


@pytest.mark.slow
def test_golden_traits_cold():
    """The heaviest model with the cache ignored (a fresh expansion, ~5 min)."""
    _check_golden('dendritic_quad_soma_sigmoid', use_cache=False)


def test_golden_detail_values():
    """Spot values: the OU pole is i*mu, the colored OU adds i/tauc, Hawkes has a delta part and 4th-order
    cumulants after the order-4 Taylor expansion, and the 2-D colored model's noise is cross-correlated."""
    r = analyze(_tracked('ou_quartic')[0])
    assert r.traits.poles == pytest.approx((1j,), abs=1e-14)
    assert r.traits.eigenvalues == pytest.approx((-1 + 0j,), abs=1e-14)
    assert r.traits.vertex_species == ((1, 2), (1, 3)) and r.traits.taylor_order == 4
    r = analyze(*_tracked('ou_quartic_colored'))
    assert sorted(p.imag for p in r.traits.poles) == pytest.approx([0.1, 1.0], rel=1e-12)
    assert r.traits.fast_slow_ratio == pytest.approx(10.0, rel=1e-12)
    r = analyze(*_tracked('linear_hawkes'))
    assert r.traits.delta_component is True and r.traits.max_noise_cumulant_order == 4
    assert r.traits.nonpolynomial_factors == ('e^nt1', 'e^nt2')   # phi = a*v is polynomial
    assert dict(r.traits.saddle)['nstar'] == pytest.approx([1.8671052631578948, 2.107894736842105], rel=1e-12)
    r = analyze(_tracked('ou_quartic_two_dim_color_corr')[0])
    assert r.traits.noise_cross_correlated and r.traits.noise_cross_pairs == (('xit1', 'yit1'),)


def test_policy_strict_consents_and_ledger():
    model, q = _tracked('linear_hawkes')
    r = analyze(model, question=dict(q, k=2, max_ell=1, external_fields=[('n', 1), ('n', 2)]))
    assert r.ok and set(f.code for f in r.warnings) == _HAWKES
    names = {e.name: e for e in r.ledger}
    assert {'taylor_truncation', 'interpretation', 'loop_truncation', 'stationary_state'} <= set(names)
    assert names['taylor_truncation'].declared is False
    strict = analyze(model, question=q, policy={'mode': 'strict'})
    assert not strict.ok and set(f.code for f in strict.errors) == _HAWKES
    ok = analyze(model, question=q, policy=Policy(mode='strict', allow_taylor_truncation=True,
                                                  interpretation='ito'))
    assert ok.ok and not ok.warnings
    assert {e.name: e.declared for e in ok.ledger}['taylor_truncation'] is True


def test_positivity_assertion_checked():
    model, q = _tracked('linear_hawkes')
    r = analyze(model, question=q, policy={'assertions': {'positive_at_saddle': ['nstar'], 'my_claim': 1}})
    led = {e.name: e for e in r.ledger}
    assert led['positive_at_saddle:nstar'].satisfied is True
    assert led['my_claim'].declared and led['my_claim'].satisfied is None


def test_bad_question_and_policy_arguments():
    with pytest.raises(TypeError, match='unknown key'):
        analyze(_tracked('ou_quartic')[0], question={'kk': 2})
    r = analyze(_tracked('ou_quartic')[0], question={'k': 3, 'external_fields': [('x', 1)]},
                policy={'mode': 'lenient'})
    assert {'G0_QUESTION_INVALID', 'G0_POLICY_INVALID'} <= set(r.codes)


# ── B: unsupported gallery ────────────────────────────────────────────────────────────────────

def _undriven():
    return (TemporalModelBuilder('Gallery undriven field')
            .physical_field('x').physical_field('y')
            .parameter('mu', default=1.0, domain='positive').parameter('D', default=1.0, domain='positive')
            .set_action_text('xt*((Dt+mu)*x) - D*xt^2 + yt*((Dt+mu)*y)')
            .equation(lhs='(Dt+mu)*x', rhs='0').equation(lhs='(Dt+mu)*y', rhs='0')
            .build()), None


def _unstable():
    return (TemporalModelBuilder('Gallery unstable saddle')
            .physical_field('x')
            .parameter('a', default=0.5, domain='positive').parameter('D', default=1.0, domain='positive')
            .set_action_text('xt*((Dt - a)*x) - D*xt^2')
            .equation(lhs='(Dt - a)*x', rhs='0')
            .build()), None


def _time_dependent():
    model = _tracked('ou_quartic')[0]
    act = model['action']
    model = dict(model)
    model['action'] = lambda ns: act(ns) + SR.var('A') * sin(SR.var('t')) * ns.xt[0]
    return model, None


def _kernel_model(name, image):
    return (TemporalModelBuilder(name)
            .population('A', size=1)
            .physical_field('x', population='A')
            .parameter('mu', default=[1.0], indexed_by=['A'], domain='positive')
            .parameter('J', default=[0.3], indexed_by=['A'], domain='positive')
            .parameter('D', default=[1.0], indexed_by=['A'], domain='positive')
            .parameter('tg', default=2.0, domain='positive').parameter('d', default=1.5, domain='positive')
            .define_kernel('g', freq_image=image, latex_name='g')
            .set_action_text('sum(xt[i]*((Dt + mu[i])*x[i] - J[i]*g*x[i]) - D[i]*xt[i]^2 for i in A)')
            .set_mf_equation('xstar', '0')
            .build()), None


def _delay_kernel():
    return _kernel_model('Gallery delay kernel', 'exp(-I*omega*d)/(1 + I*omega*tg)')


def _sqrt_kernel():
    return _kernel_model('Gallery sqrt kernel', '1/sqrt(1 + I*omega*tg)')


def _missing_external():
    model = _tracked('ou_quartic')[0]
    return model, {'k': 2, 'external_fields': [('y', 1), ('y', 1)]}


def _marginal():
    return (TemporalModelBuilder('Gallery marginal mode')
            .population('A', size=1)
            .physical_field('x', population='A')
            .parameter('g', default=[1.0], indexed_by=['A'], domain='positive')
            .parameter('D', default=[0.1], indexed_by=['A'], domain='positive')
            .set_action_text('sum(xt[i]*(Dt*x[i] + g[i]*x[i]^3) - D[i]*xt[i]^2 for i in A)')
            .set_mf_equation('xstar', '0')
            .build()), None


def _two_stable():
    model = _tracked('ou_quartic_double_well')[0]
    return model, {'parameters': {'mu': -1.0, 'eps': 0.1, 'D': 1.0}}


GALLERY = {
    'undriven field':        (_undriven, 'G1_UNDRIVEN_FIELD', 'dy1'),
    'unstable saddle':       (_unstable, 'G2_UNSTABLE_SADDLE', 'unstable'),
    'explicit time':         (_time_dependent, 'G2_TIME_DEPENDENT', 'sin(t)'),
    'delay kernel':          (_delay_kernel, 'G3_NONRATIONAL_KERNEL', 'e^(-I*d*omega)'),
    'sqrt kernel':           (_sqrt_kernel, 'G3_NONRATIONAL_KERNEL', 'sqrt'),
    'missing external field': (_missing_external, 'G0_EXTERNAL_FIELD', "('y', 1) not in phys_idx"),
    'marginal mode':         (_marginal, 'G2_MARGINAL_MODE', 'marginal'),
    'two stable saddles':    (_two_stable, 'G2_MULTISTABLE', '2 stable saddles'),
}


@pytest.mark.parametrize('case', list(GALLERY))
def test_unsupported_gallery(case):
    build, code, cause = GALLERY[case]
    model, q = build()
    r = analyze(model, question=q)                  # must not raise
    hits = [f for f in r.findings if f.code == code]
    assert hits, r.summary()
    assert cause in hits[0].message, hits[0].message
    assert hits[0].remedy
    assert 'INTERNAL_ERROR' not in r.codes, r.summary()
    if FINDING_CODES[code][1] == 'error':
        assert not r.ok


def test_two_stable_saddles_confirmed():
    model, q = _two_stable()
    r = analyze(model, question=dict(q, fixed_point_index=1), policy={'assertions': {'saddle_confirmed': True}})
    f = [f for f in r.findings if f.code == 'G2_MULTISTABLE'][0]
    assert f.level == 'info' and r.ok
    assert r.traits.n_saddles == 3 and r.traits.n_stable_saddles == 2 and r.traits.saddle_index == 1
    assert dict(r.traits.saddle)['xstar'][0] == pytest.approx(10 ** 0.5, rel=1e-8)
    r = analyze(model, question=dict(q, fixed_point_index=2))
    assert 'G2_FIXED_POINT_INDEX' in r.codes


def test_findings_are_typed():
    r = analyze(*_unstable())
    assert isinstance(r, ModelReport) and all(isinstance(f, Finding) for f in r.findings)
    for f in r.findings:
        assert f.level in ('error', 'warning', 'info') and f.code in FINDING_CODES
        assert f.gate == FINDING_CODES[f.code][0]
    assert r.route.startswith('unrouted')


# ── C: non-interference ───────────────────────────────────────────────────────────────────────

def _same(a, b, path='model'):
    """Structural equality of two model dicts; functions by identity, plain objects by their attributes, Sage
    objects by their string."""
    if isinstance(a, dict):
        assert isinstance(b, dict) and set(a) == set(b), path
        for k in a:
            _same(a[k], b[k], f'{path}[{k!r}]')
    elif isinstance(a, (list, tuple)):
        assert type(a) is type(b) and len(a) == len(b), path
        for i, (x, y) in enumerate(zip(a, b)):
            _same(x, y, f'{path}[{i}]')
    elif isinstance(a, types.FunctionType):
        assert a is b, path                         # deepcopy keeps functions (and lambdas) by identity
    elif hasattr(a, '__dict__') and not hasattr(a, 'parent'):
        assert type(a) is type(b), path             # plain objects (e.g. _CGFKernelCallable): their state
        _same(vars(a), vars(b), f'{path}.__dict__')
    elif isinstance(a, (str, int, float, bool, type(None))):
        assert a == b, path
    else:
        assert str(a) == str(b), path


@pytest.mark.parametrize('name', sorted(GOLDEN))
def test_fast_and_leaves_model_unchanged(name):
    model, q = _build(name)
    if name in HEAVY and not _warm(name, model):
        pytest.skip(f'{name}: cold expand exceeds the budget; warm saved_models/ first')
    before = copy.deepcopy(model)
    t0 = time.perf_counter()
    analyze(model, question=q)
    wall = time.perf_counter() - t0
    _same(model, before)
    assert wall < BUDGET_S, f'{name}: {wall:.2f}s'


def test_analyze_never_writes_the_cache(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)                     # saved_models/ resolves relative to the cwd
    analyze(_tracked('ou_quartic')[0])
    assert not (tmp_path / 'saved_models').exists()
