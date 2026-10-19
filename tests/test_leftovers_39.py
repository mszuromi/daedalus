"""Front-end and expansion leftovers (work item 39), each reproduced on main before its fix:

1. The mean-field-sector check of ``FieldTheory.expand`` (``_verify_and_zero_mf_sector``): a coefficient
   shown nonzero by interval arithmetic skips ``simplify_full`` (it cannot reach 0), and the saddle is
   solved once per check, not once per coefficient.  Same verdicts, same messages, same warnings.
2. Actions with ``D[0](f)(...)`` atoms: not reachable from the public front end, and the legacy
   ``taylor()`` leaves such an atom unexpanded too (pinned below), so nothing to extend.
3. ``SDE._kernel_dc`` of ``(c*dirac_delta(t) + h(t))*heaviside(t)``: the delta keeps its full weight.
4. A scalar model (no ``.population()``) with ``set_mf_equation('nstar', 'a*exp(...)')`` passes the
   mean-field self-check, as its 1-element population twin does.
5. SDE front-end engine bugs: a numpy overflow in the DAE residual no longer raises
   ``KeyError('__import__')``; a population named ``pop`` with size > 1 no longer truncates sums; a
   Stratonovich lhs whose Dt-coefficient is not constant is refused (it was read as ``a = 1``);
   scalar-mode ``linear_stability`` on ``exp(field)`` is not reproducible (pinned).  A source shared
   across a population is a feature, not a bug (documented in the report).

Every bit-for-bit comparison runs the first form once and discards it (cold and warm caches differ by
~1e-16)."""
import warnings

import numpy as np
import pytest

pytest.importorskip('sage.all')
from sage.all import SR, assumptions, var, sqrt, erf, exp, function, taylor   # noqa: E402

import api.compute as AC                                             # noqa: E402
import engine.core.field_theory as F                                 # noqa: E402
from api.model import TemporalModelBuilder                           # noqa: E402
from api.noise import Gaussian                                       # noqa: E402
from api.sde import SDE, SDEError                                    # noqa: E402

TAU = np.array([-1.0, 0.0, 0.7])


def _run(model, ext, ell=1):
    return AC.compute_cumulants(model, k=2, max_ell=ell, external_fields=ext, tau_grid=TAU,
                                use_cache=False, parallel=False, verbose=False)


def _same_run(a, b):
    """Same saddle, propagator and cumulants (every loop order), bit for bit."""
    assert a['mf_values'] == b['mf_values']
    for key in ('K_ft', 'G_ft'):
        if key in a['propagator']:
            assert str(a['propagator'][key]) == str(b['propagator'][key])
    assert set(a['C_tau_by_ell']) == set(b['C_tau_by_ell'])
    for ell in a['C_tau_by_ell']:
        assert np.array_equal(np.asarray(a['C_tau_by_ell'][ell]), np.asarray(b['C_tau_by_ell'][ell]))


def _expanded(model, order=2):
    ft = F.FieldTheory(model, taylor_order=order)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        ft.expand()
    return ({bg: str(p) for bg, p in ft._by_tp.items()}, str(ft._S_raw),
            sorted(str(x.message) for x in w))


def _compare(build_a, build_b, ext, ell=1):
    _run(build_a(), ext, ell)                                   # warm-up, discarded
    ra, rb = _run(build_a(), ext, ell), _run(build_b(), ext, ell)
    _same_run(ra, rb)
    assert _expanded(build_a(), 4)[:2] == _expanded(build_b(), 4)[:2]
    return ra


# ── a voltage-Hawkes model with a mean-field closure (tasks 1 and 4) ─────────────────────────

def _hawkes_mf(population, form='inline', mf_scale=1):
    """n ~ Poisson(a exp(Em + c v)), v = Em + w n; the closure is given by ``set_mf_equation``.
    ``mf_scale != 1`` makes the closure inconsistent with the action (the self-check must fail)."""
    b = TemporalModelBuilder('hawkes mf')
    kw, ix = {}, ''
    if population:
        b.population('E', size=1)
        kw, ix = dict(population='E'), '[i]'
    b = (b.physical_field('n', **kw).physical_field('v', **kw)
         .parameter('Em', default=-1.2).parameter('tau', default=2.0, domain='positive')
         .parameter('a', default=1.0, domain='positive').parameter('w', default=0.4)
         .parameter('c', default=0.5))
    if form == 'define_function':
        b = b.define_function('phi', args=['v'], expression='a*exp(Em + c*v)', **kw)
        rate, mf_rate = f'phi(v{ix})', 'phi(vstar[i])'
    else:
        rate, mf_rate = f'a*exp(Em + c*v{ix})', 'a*exp(Em + c*vstar[i])'
    body = (f'nt{ix}*n{ix} - (exp(nt{ix}) - 1)*{rate} + '
            f'vt{ix}*((tau*Dt + 1)*v{ix} - Em - w*n{ix})')
    b = b.set_action_text(f'sum({body} for i in E)' if population else body)
    mf_rate = f'{mf_scale}*{mf_rate}' if mf_scale != 1 else mf_rate
    return (b.set_mf_equation('vstar', 'Em + w*nstar[i]').set_mf_equation('nstar', mf_rate)).build()


# ── 1. the mean-field-sector check ───────────────────────────────────────────────────────────

def test_proved_nonzero_exact_expressions():
    x, y = var('x y')
    assert F._proved_nonzero(x + y + 1)
    assert F._proved_nonzero(exp(x) - 3)                  # nonzero as a function (zero only at x = log 3)
    # identically zero in disguise: never "proved nonzero"
    assert not F._proved_nonzero((x + 1) ** 2 - x ** 2 - 2 * x - 1)
    assert not F._proved_nonzero(SR(0))
    # a rigorous operator set only: erf goes to Sage's own test
    assert not F._proved_nonzero(erf(x) + 5)


def test_proved_nonzero_respects_assumptions():
    q = var('q_pos39')
    from sage.all import assume, forget
    assume(q > 1)                 # the probe value must satisfy the assumption: 0.2..0.8 does not
    try:
        assert not F._proved_nonzero(q - 100)
        assert F._sr_nonzero(q - 100)                       # Sage's own verdict still decides
    finally:
        forget(q > 1)


def test_proved_nonzero_float_literals_only_with_a_margin():
    x = var('x39')
    c = SR(0.3) * x + SR(0.1)
    assert not F._proved_nonzero(c)                         # floats are excluded by default
    assert F._proved_nonzero(c, floats_ok=True)             # a genuine nonzero function
    # a small but real float part, far above the 1e-8 margin
    assert F._proved_nonzero(SR(1e-3) * x + SR(0.5) * x ** 2, floats_ok=True)


def test_proved_nonzero_float_dust_is_not_nonzero():
    """Floats an ulp apart cancel for Sage and Maxima (``exp(-0.19999999999999998*x) - exp(-0.2*x)`` is
    zero for them); the interval proof on the exact rationals must not call that nonzero."""
    x = var('x39d')
    from sage.all import tanh
    dust = [exp(SR(-0.3) * x) * exp(SR(0.1) * x) - exp(SR(-0.2) * x),
            tanh(SR(0.30000000000000004) * x) - tanh(SR(0.3) * x)]
    for c in dust:
        assert not F._proved_nonzero(c, floats_ok=True), c
        assert c.simplify_full() == 0, c   # zero for the old path (the verdict the shortcut must not override)


def test_mf_check_float_dust_coefficient_passes_silently_as_on_main(monkeypatch):
    """Drive the check with a sector coefficient that is float dust after the mean-field substitution:
    the verdict is the old one (zero, no warning), with and without the interval shortcut."""
    from sage.all import PolynomialRing, tanh
    a, b = var('a39', 'b39')
    R = PolynomialRing(SR, 'u,v')

    class NS:
        pass

    def verdict():
        by_tp = {(1, 0): R({(1, 0): exp(SR(-0.3) * a) * exp(b) - exp(SR(-0.2) * a)})}
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            try:
                F._verify_and_zero_mf_sector(by_tp, {b: SR(0.1) * a}, {}, NS(), R, {'parameters': []})
                return 'pass', [str(x.message) for x in w]
            except AssertionError as exc:
                return 'fail', str(exc)[:40]
    new = verdict()
    monkeypatch.setattr(F, '_proved_nonzero', lambda *a, **k: False)
    assert new == verdict() == ('pass', [])


def test_sr_nonzero_is_unchanged_by_the_refactor():
    q = var('q39b')
    assert F._sr_nonzero(q + 1)
    assert not F._sr_nonzero(SR(0))
    assert not F._sr_nonzero(SR(0.1) * q - q / 10)           # Sage's approximate float zero
    assert F._sr_nonzero(exp(q) - 1)


def test_mf_check_skips_simplify_and_keeps_every_verdict(monkeypatch):
    """The check with the interval shortcut equals the check without it (``_proved_nonzero`` forced
    to "not proved"): same expansion, same warnings."""
    new = _expanded(_hawkes_mf(True), 2)
    monkeypatch.setattr(F, '_proved_nonzero', lambda *a, **k: False)
    old = _expanded(_hawkes_mf(True), 2)
    assert new == old
    assert new[2], 'expected the "vanishes only numerically" warning'


def test_mf_check_solves_the_saddle_once_per_check(monkeypatch):
    calls = []
    orig = F._mf_saddle_values
    monkeypatch.setattr(F, '_mf_saddle_values', lambda *a: calls.append(1) or orig(*a))
    _expanded(_hawkes_mf(True), 2)
    assert len(calls) == 1                                  # the check has several coefficients
    n_coeff = []
    orig_res = F.mf_sector_residual
    monkeypatch.setattr(F, 'mf_sector_residual', lambda *a, **k: n_coeff.append(1) or orig_res(*a, **k))
    _expanded(_hawkes_mf(True), 2)
    assert len(n_coeff) > 1


def test_mf_check_failure_message_is_the_simplified_one(monkeypatch):
    """A closure inconsistent with the action still fails, with the same text on both routes."""
    def message():
        with pytest.raises(AssertionError) as exc:
            _expanded(_hawkes_mf(True, mf_scale=2), 2)
        return str(exc.value)
    new = message()
    monkeypatch.setattr(F, '_proved_nonzero', lambda *a, **k: False)
    old = message()
    assert new == old
    assert 'symbolic residual:' in new and 'numerical residual:' in new


# ── 2. D[0](f)(...) atoms ────────────────────────────────────────────────────────────────────

def test_derivative_atoms_per_atom_taylor_is_exact_in_one_variable():
    """Item 60 (bug 2): Maxima's multivariate ``taylor()`` stops ``k`` orders early on a ``k``-th
    derivative atom (``taylor(x*D[0](f)(x), (x, 0), 3)`` has no ``x^3`` term); the per-atom route
    expands an atom of ONE field variable with the univariate form, which is exact."""
    f, g = function('f39'), function('g39')
    x, y = var('x39', 'y39')
    for e in (f(x).diff(x), x * f(x).diff(x), 3 * g(y).diff(y) + 1, x ** 2 * g(x).diff(x, 2)):
        got = F._truncated_taylor(e, [x, y], 3)
        v = x if e.has(x) else y
        ref = SR(taylor(e, v, 0, 3)).expand()
        sm = SR(0)
        for ex, c in got.items():
            sm += c * x ** ex[0] * y ** ex[1]
        assert (sm - ref).expand().is_trivial_zero()
    # the degree-3 term the multivariate form dropped
    got = F._truncated_taylor(x * f(x).diff(x), [x], 3)
    assert got[(3,)] != 0


def test_derivative_atom_of_several_field_variables_is_refused_by_name():
    """A derivative atom the univariate form cannot take (several field variables, or the
    whole-action ``taylor()`` route) raises ``DerivativeAtomTaylorError`` instead of a truncated expansion."""
    f, g = function('f39b'), function('g39b')
    x, y = var('x39b', 'y39b')
    for e in (y * f(x + y).diff(x), g(x, y).diff(x)):
        with pytest.raises(F.DerivativeAtomTaylorError):
            F._taylor_action(e, [x, y], 3)


# ── 3. a delta inside the heaviside product of a causal kernel ───────────────────────────────

def _sde_dirac(time_expr=None, freq_image=None):
    return (SDE('delta kernel')
            .parameter('tau', default=1.3, domain='positive')
            .parameter('c', default=0.3)
            .parameter('D', default=0.1, domain='positive')
            .define_kernel('g', **({'time_expr': time_expr} if time_expr else {'freq_image': freq_image}))
            .field('x').noise('xi', Gaussian(var='2*D'), couples={'x': 1})
            .equation('x', lhs='(Dt + 1)*x', rhs='-0.2*Conv(g, x)'))


DELTA_FORMS = {
    'inside': '(c*dirac_delta(t) + exp(-t/tau)/tau)*heaviside(t)',
    'outside': 'c*dirac_delta(t) + exp(-t/tau)/tau*heaviside(t)',
}


@pytest.mark.parametrize('name', sorted(DELTA_FORMS))
def test_kernel_dc_delta_keeps_its_full_weight(name):
    c, tau = var('c'), var('tau')
    before = list(assumptions())
    dc = _sde_dirac(time_expr=DELTA_FORMS[name])._kernel_dc('g')
    assert not dc.has(var('t')) and 'heaviside' not in str(dc)
    assert (dc - (c + 1)).simplify_full().is_zero()
    ref = _sde_dirac(freq_image='c + 1/(1 + I*omega*tau)')._kernel_dc('g')
    assert (dc - ref).simplify_full().is_zero()
    assert list(assumptions()) == before


@pytest.mark.parametrize('name', sorted(DELTA_FORMS))
def test_delta_kernel_equations_equal_the_freq_image_form(name):
    a = _sde_dirac(time_expr=DELTA_FORMS[name]).equations()
    b = _sde_dirac(freq_image='c + 1/(1 + I*omega*tau)').equations()
    assert a == b


def test_kernel_dc_delta_alone_and_delta_with_a_t_factor():
    c, tau = var('c'), var('tau')
    assert (_sde_dirac(time_expr='c*dirac_delta(t)*heaviside(t)')._kernel_dc('g') - c).is_zero()
    # f(t) dirac_delta(t) = f(0) dirac_delta(t): no t is left in the integral
    dc = _sde_dirac(time_expr='c*exp(-t/tau)*dirac_delta(t)*heaviside(t)')._kernel_dc('g')
    assert not dc.has(var('t')) and (dc - c).is_zero()


# ── 4. scalar model, set_mf_equation closure ─────────────────────────────────────────────────

@pytest.mark.parametrize('form', ['inline', 'define_function'])
def test_scalar_mf_closure_passes_the_self_check_and_equals_the_population_twin(form):
    ext = [('dn', 1), ('dn', 1)]
    ra = _compare(lambda: _hawkes_mf(False, form), lambda: _hawkes_mf(True, form), ext, ell=1)
    assert any(np.abs(np.asarray(ra['C_tau_by_ell'][1])) > 0)


def test_scalar_mf_closure_that_is_inconsistent_still_fails():
    with pytest.raises(AssertionError, match='mean-field saddle does not satisfy'):
        _expanded(_hawkes_mf(False, mf_scale=2), 2)


def test_scalar_mf_saddle_is_the_population_twin_saddle():
    a = _run(_hawkes_mf(False), [('dn', 1), ('dn', 1)], 0)['mf_values']
    b = _run(_hawkes_mf(True), [('dn', 1), ('dn', 1)], 0)['mf_values']
    assert a == b and a['nstar'][0] > 0


# ── 5. SDE front-end engine bugs ─────────────────────────────────────────────────────────────

def _saturating_model():
    return (TemporalModelBuilder('overflow').population('E', size=1)
            .physical_field('x', population='E')
            .parameter('k', default=300.0, domain='positive')
            .set_action_text('sum(xt[i]*((Dt + 1)*x[i] - 1/(1 + exp(k*x[i]^2))) for i in E)')
            .equation(lhs='(Dt+1)*x', rhs='1/(1 + exp(k*x[i]^2))', population='E').build())


@pytest.mark.parametrize('action', ['always', 'default', 'error'])
def test_dae_overflow_does_not_raise_keyerror(action):
    """``exp(300 x^2)`` overflows for the larger seeds; the warning used to be issued from a frame
    without ``__builtins__`` and raised ``KeyError('__import__')``.  The values are unchanged: a
    saturating expression is a valid number."""
    from api._mean_field_dae import solve_mean_field_dae
    model = _saturating_model()
    with warnings.catch_warnings():
        warnings.simplefilter(action)
        r = solve_mean_field_dae(model, {'k': 300.0}, verbose=False)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        ref = solve_mean_field_dae(model, {'k': 300.0}, verbose=False)
    assert r['mf_values'] == ref['mf_values']
    x = r['mf_values']['xstar'][0]
    assert abs(x - 1 / (1 + np.exp(300.0 * x * x))) < 1e-12


def _pop_model(pname, size=3):
    return (TemporalModelBuilder('p').population(pname, size=size)
            .physical_field('x', population=pname)
            .parameter('mu', default=[1.0, 1.5, 2.0][:size], indexed_by=[pname], domain='positive')
            .parameter('D', default=0.2, domain='positive')
            .set_action_text(f'sum(xt[i]*((Dt + mu[i])*x[i]) - D*xt[i]^2 for i in {pname})')
            .equation(lhs='(Dt + 1)*x[i]', rhs='(1 - mu[i])*x[i]', population=pname).build())


def test_population_named_pop_is_not_truncated():
    a, b = _expanded(_pop_model('E'), 2), _expanded(_pop_model('pop'), 2)
    assert a == b
    assert 'xt3' in a[1]                                    # the third member is in the action
    ra = _run(_pop_model('E'), [('dx', 3), ('dx', 3)], 0)
    rb = _run(_pop_model('pop'), [('dx', 3), ('dx', 3)], 0)
    assert any(np.abs(np.asarray(ra['C_tau_by_ell'][0])) > 0)
    _same_run(ra, rb)


def test_population_named_pop_of_size_one_is_unchanged():
    # size 1: the legacy binding and the population's own indices coincide
    assert _expanded(_pop_model('pop', 1), 2) == _expanded(_pop_model('E', 1), 2)


def _two_pops(pname):
    """``pname`` (size 2) beside two populations of size 1: n_pop = 3 exceeds the size of ``pname``."""
    b = (TemporalModelBuilder('p3').population(pname, size=2).population('F1', size=1).population('F2', size=1)
         .physical_field('x', population=pname).physical_field('y1', population='F1')
         .physical_field('y2', population='F2')
         .parameter('mu', default=[1.0, 1.5], indexed_by=[pname], domain='positive')
         .parameter('D', default=0.2, domain='positive'))
    txt = (f'sum(xt[i]*((Dt + mu[i])*x[i]) - D*xt[i]^2 for i in {pname}) + '
           'sum(y1t[i]*((Dt + 1)*y1[i]) - D*y1t[i]^2 for i in F1) + '
           'sum(y2t[i]*((Dt + 1)*y2[i]) - D*y2t[i]^2 for i in F2)')
    return (b.set_action_text(txt).equation(lhs='(Dt + 1)*x[i]', rhs='(1 - mu[i])*x[i]', population=pname)
            .equation(lhs='(Dt + 1)*y1[i]', rhs='0', population='F1')
            .equation(lhs='(Dt + 1)*y2[i]', rhs='0', population='F2').build())


def test_population_named_pop_beside_other_populations_covers_its_members_only():
    a, b = _expanded(_two_pops('E'), 2), _expanded(_two_pops('pop'), 2)
    assert a == b and 'xt2' in a[1] and 'xt3' not in a[1]


def test_sde_population_named_pop_accepts_size_above_one():
    def sde(pname):
        return (SDE('p').population(pname, size=2)
                .parameter('mu', default=[1.0, 1.5], indexed_by=[pname], domain='positive')
                .parameter('D', default=0.2, domain='positive')
                .field('x', population=pname)
                .noise('xi', Gaussian(var='2*D'), couples={'x': 1})
                .equation('x', lhs='(Dt + 1)*x', rhs='(1 - mu[i])*x[i]'))
    eq_pop, eq_e = sde('pop').equations(), sde('E').equations()
    assert eq_pop == eq_e or [e['rhs'] for e in eq_pop] == [e['rhs'] for e in eq_e]
    ra = _run(sde('E').build(), [('dx', 1), ('dx', 1)], 0)
    rb = _run(sde('pop').build(), [('dx', 1), ('dx', 1)], 0)
    _same_run(ra, rb)
    assert any(np.abs(np.asarray(ra['C_tau_by_ell'][0])) > 0)


@pytest.mark.parametrize('rhs', ['a*exp(-x)', 'a*exp(0.5*x)', 'exp(-x)', 'a*log(1 + x*x) + 0.1'])
def test_scalar_mode_linear_stability_on_exp_of_the_field(rhs):
    """Not reproducible on main (the scalar field wrapper is unwrapped by the Sage math names):
    ``linear_stability`` of a population-less model with ``exp(field)``."""
    from api._mean_field_dae import linear_stability, solve_mean_field_dae
    m = (TemporalModelBuilder('s').physical_field('x').parameter('a', default=0.5, domain='positive')
         .stability_analysis(True).set_action_text('xt*((Dt + 1)*x)')
         .equation(lhs='(Dt+1)*x', rhs=rhs).build())
    r = solve_mean_field_dae(m, {'a': 0.5}, verbose=False)
    ls = linear_stability(m, {'a': 0.5}, r['mf_values'], verbose=False)
    assert ls['stable'] and ls['eigenvalues_finite'].size == 1
    xs = r['mf_values']['xstar'][0]
    d = {'a*exp(-x)': -0.5 * np.exp(-xs), 'a*exp(0.5*x)': 0.25 * np.exp(0.5 * xs),
         'exp(-x)': -np.exp(-xs), 'a*log(1 + x*x) + 0.1': 0.5 * 2 * xs / (1 + xs * xs)}[rhs]
    assert abs(ls['eigenvalues_finite'][0].real - (-1 + d)) < 1e-9


def _strat(lhs):
    s = (SDE('s').parameter('D', default=0.1, domain='positive').field('x')
         .interpretation('stratonovich')
         .noise('xi', Gaussian(var='2*D'), couples={'x': '(1 + 0.5*x^2)'})
         .equation('x', lhs=lhs, rhs='-x'))
    if 'y' in lhs:                          # Dt of another field in the x equation: read off the x part, as on main
        s = (s.field('y').noise('eta', Gaussian(var='2*D'), couples={'y': 1})
             .equation('y', lhs='(Dt + 1)*y', rhs='0'))
    return s


@pytest.mark.parametrize('lhs', ['(1+x^2)*Dt*x', 'Dt*x + x*Dt*x', 'x*Dt*x'])
def test_stratonovich_refuses_a_nonconstant_dt_coefficient(lhs):
    """On main ``(1+x^2)*Dt*x`` was read as ``a = 1`` (the x-part of x + x^3) and got the correction
    of ``Dt*x``: silently wrong."""
    with pytest.raises(SDEError, match='constant Dt coefficient'):
        _strat(lhs).equations()


@pytest.mark.parametrize('lhs, scale', [('Dt*x', 1), ('2*Dt*x', 2), ('(Dt*x)', 1), ('(Dt + 1)*x', 1),
                                         ('Dt*x + Dt*y', 1)])
def test_stratonovich_constant_dt_coefficient_is_unchanged(lhs, scale):
    eq = _strat(lhs).equations()
    assert eq and 'x[i]^3' in eq[0]['rhs']


def test_scalar_mf_equation_model_without_equation_gets_the_saddle_values_in_the_result():
    """A scalar model with only ``set_mf_equation``: ``mf_values`` was empty on main (the saddle was not
    recorded), it is the twin's now; the cumulants are the same bit for bit."""
    def ou(population):
        b = TemporalModelBuilder('ou mf')
        kw, ix = {}, ''
        if population:
            b.population('E', size=1)
            kw, ix = dict(population='E'), '[i]'
        b = (b.physical_field('x', **kw).parameter('m', default=0.7)
             .parameter('D', default=0.2, domain='positive'))
        body = f'xt{ix}*((Dt + 1)*x{ix} - m) - D*xt{ix}^2'
        return b.set_action_text(f'sum({body} for i in E)' if population else body) \
            .set_mf_equation('xstar', 'm').build()
    ra, rb = _run(ou(False), [('dx', 1), ('dx', 1)], 1), _run(ou(True), [('dx', 1), ('dx', 1)], 1)
    for ell in ra['C_tau_by_ell']:
        assert np.array_equal(np.asarray(ra['C_tau_by_ell'][ell]), np.asarray(rb['C_tau_by_ell'][ell]))
    assert rb['mf_values'] == {'xstar': [0.7]}
    assert ra['mf_values'] == rb['mf_values']
