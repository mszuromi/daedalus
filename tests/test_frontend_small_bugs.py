"""Front-end small bugs (work item 37), each reproduced on main before its fix:

1. ``SDE._kernel_dc`` on a ``time_expr`` kernel that Maxima cannot integrate (a difference of
   exponentials: 'requested additional constraints'; a heaviside-free kernel: 'Integral is
   divergent') now falls back to the closed form of ``causal_exponential_ft`` at omega = 0 with the
   rates positive by declaration; no Maxima assumption leaks.
2. A scalar (no ``.population()``) model with ``Conv(r, n)`` on a bare field builds: ``Conv`` unwraps
   the size-1 field wrapper as the declared functions already do.
3. ``_is_single_function_call`` no longer reports ``f(a) + g(b)`` / ``f(a)*g(b)`` as one call.
4. ``SDE.notes`` is rebuilt on every derivation (already so on main: the guard test pins it).

Every bit-for-bit comparison runs the first form once and discards it (cold and warm caches differ
by ~1e-16)."""
import numpy as np
import pytest

pytest.importorskip('sage.all')
from sage.all import SR, assumptions, var                           # noqa: E402

import api.compute as AC                                            # noqa: E402
from api.model_compiler import _is_single_function_call             # noqa: E402
from api.noise import Gaussian                                      # noqa: E402
from api.sde import SDE, SDEError                                   # noqa: E402
from api.theory import TemporalTheoryBuilder                        # noqa: E402

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


def _action(model):
    from engine.core.field_theory import FieldTheory
    ft = FieldTheory(model, taylor_order=4)
    ft.expand()
    return {bg: str(p) for bg, p in ft._by_tp.items()}


def _compare(build_a, build_b, ext, ell=1):
    _run(build_a(), ext, ell)                                   # warm-up, discarded
    ra, rb = _run(build_a(), ext, ell), _run(build_b(), ext, ell)
    _same_run(ra, rb)
    assert _action(build_a()) == _action(build_b())
    return ra


# ── 1. SDE time_expr kernel integral ─────────────────────────────────────────────────────────

# name: (time_expr, freq_image).  The difference of exponentials is written in the form the compiled
# transform returns (factored closed form), so its propagator text and every cumulant are
# bit-identical.  A hand-written image of the other two cannot reproduce the algebraic form of the
# compiled one (alpha: Sage's own unfactored transform, which never needed the fallback; three
# exponentials: one common denominator), so they agree to rounding (rtol 1e-12) and exactly in
# the kernel integral and the mean-field text.  Maxima integrates alpha and three_exp itself (they
# are regression guards that pass on main); only FALLBACK_KERNELS reach the new code.
KERNELS = {
    'alpha': ('t*exp(-t/tau)/tau^2*heaviside(t)', '1/(1 + I*omega*tau)^2'),
    'diff_exp': ('(exp(-t/tau) - exp(-t/taug))/(tau - taug)*heaviside(t)',
                 '1/((1 + I*omega*tau)*(1 + I*omega*taug))'),
    # mixed signs, three rates 1/tau, 1/taug, 1/(tau + taug)
    'three_exp': ('(2*exp(-t/tau) - 3*exp(-t/taug) + 2*exp(-t/(tau + taug)))*heaviside(t)',
                  '2/(1/tau + I*omega) - 3/(1/taug + I*omega) + 2/(1/(tau + taug) + I*omega)'),
}
BIT_IDENTICAL = {'diff_exp'}


def _sde_kernel(**kernel):
    """Scalar quartic OU with a kernel-filtered linear feedback (one loop is non-trivial)."""
    return (SDE('kernel dc')
            .parameter('tau', default=1.3, domain='positive')
            .parameter('taug', default=0.7, domain='positive')
            .parameter('D', default=0.1, domain='positive')
            .parameter('eps', default=0.05, domain='positive')
            .define_kernel('g', **kernel)
            .field('x').noise('xi', Gaussian(var='2*D'), couples={'x': 1})
            .equation('x', lhs='Dt*x', rhs='-x - eps*x^3 - 0.2*Conv(g, x)'))


@pytest.mark.parametrize('name', sorted(KERNELS))
def test_sde_time_expr_kernel_dc_equals_freq_image_at_omega_0(name):
    te, fi = KERNELS[name]
    before = list(assumptions())
    dc = _sde_kernel(time_expr=te)._kernel_dc('g')
    ref = _sde_kernel(freq_image=fi)._kernel_dc('g')
    assert (dc - ref).simplify_full().is_zero()
    assert list(assumptions()) == before                       # nothing assumed, nothing leaked
    # the value itself, from the closed form n!/a^(n+1) summed (independent of the code under test)
    tau, taug = var('tau'), var('taug')
    expected = {'alpha': SR(1), 'diff_exp': SR(1),
                'three_exp': 2 * tau - 3 * taug + 2 * (tau + taug)}[name]
    assert (dc - expected).simplify_full().is_zero()


# kernels Maxima cannot integrate: expected integral
FALLBACK_KERNELS = {
    'diff_exp': ('(exp(-t/tau) - exp(-t/taug))/(tau - taug)*heaviside(t)', 1),
    # mixed signs, a double pole, the same rate twice
    'diff_minus_alpha': ('((exp(-t/tau) - exp(-t/taug))/(tau - taug) - 0.5*t*exp(-t/taug)/taug^2)'
                         '*heaviside(t)', 0.5),
    # a delta part beside the retarded part
    'delta_plus_diff': ('0.3*dirac_delta(t) + (exp(-t/tau) - exp(-t/taug))/(tau - taug)*heaviside(t)',
                        1.3),
}


@pytest.mark.parametrize('name', sorted(FALLBACK_KERNELS))
def test_sde_kernel_dc_fallback_is_taken_and_exact(name, monkeypatch):
    te, expected = FALLBACK_KERNELS[name]
    calls = []
    orig = SDE._kernel_dc_causal
    monkeypatch.setattr(SDE, '_kernel_dc_causal',
                        lambda self, *a: calls.append(1) or orig(self, *a))
    before = list(assumptions())
    dc = _sde_kernel(time_expr=te)._kernel_dc('g')
    assert calls                                               # Maxima could not do it
    assert (dc - SR(expected)).simplify_full().is_zero()
    assert list(assumptions()) == before


@pytest.mark.parametrize('name', sorted(KERNELS))
def test_sde_time_expr_kernel_cumulants_equal_freq_image_form(name):
    te, fi = KERNELS[name]
    ext = [('dx', 1), ('dx', 1)]

    def build_a():
        return _sde_kernel(time_expr=te).build()

    def build_b():
        return _sde_kernel(freq_image=fi).build()
    if name in BIT_IDENTICAL:
        ra = _compare(build_a, build_b, ext, ell=1)
    else:
        _run(build_a(), ext)                                    # warm-up, discarded
        ra, rb = _run(build_a(), ext), _run(build_b(), ext)
        for ell in ra['C_tau_by_ell']:
            assert np.allclose(ra['C_tau_by_ell'][ell], rb['C_tau_by_ell'][ell], rtol=1e-12, atol=1e-15)
    assert any(np.abs(np.asarray(ra['C_tau_by_ell'][1])) > 0)  # a non-trivial loop was compared
    assert _sde_kernel(time_expr=te).equations() == _sde_kernel(freq_image=fi).equations()


def test_sde_indexed_time_expr_kernel_dc_uses_the_declared_positivity_of_indexed_rates():
    """Adversarial: per-pair rates ``tau[i]``, ``taug[i,j]`` (symbols ``taug__i__j``) declared
    positive, a difference of exponentials: integral 1 for every pair."""
    s = (SDE('indexed kernel').population('E', size=2)
         .parameter('tau', default=[1.3, 1.1], indexed=True, domain='positive')
         .parameter('taug', default=[[0.7, 0.8], [0.6, 0.9]], indexed='matrix', domain='positive')
         .define_kernel('g', time_expr='(exp(-t/tau[i]) - exp(-t/taug[i,j]))/(tau[i] - taug[i,j])'
                                       '*heaviside(t)', indexed_by=['E', 'E'])
         .field('x', population='E').noise('xi', Gaussian(var='1'), couples={'x': 1})
         .equation('x', lhs='(Dt + 1)*x', rhs='-0.2*sum(Conv(g[i,j], x[j]) for j in E)'))
    assert s._kernel_dc('g') == 1
    assert s.equations()[0]['rhs'] == '-0.2*sum(((1))*(x[j]) for j in E)'


def test_sde_kernel_dc_without_declared_positive_rates_is_refused_not_guessed():
    """A difference of exponentials whose time constants are not declared positive has no
    decidable sign: a clear error, never a silent value."""
    s = (SDE('undeclared').parameter('tau', default=1.3).parameter('taug', default=0.7)
         .define_kernel('g', time_expr='(exp(-t/tau) - exp(-t/taug))/(tau - taug)*heaviside(t)')
         .field('x').noise('xi', Gaussian(var='1'), couples={'x': 1})
         .equation('x', lhs='Dt*x', rhs='-x - 0.2*Conv(g, x)'))
    with pytest.raises(SDEError, match='positive'):
        s._kernel_dc('g')


def test_sde_kernel_dc_non_causal_time_expr_is_a_clear_error():
    """No heaviside: the integral over the whole line diverges (as on main); the compiled kernel
    transform refuses it too.  The error names the cause."""
    s = _sde_kernel(time_expr='t*exp(-t/tau)/tau^2')
    with pytest.raises(SDEError, match='heaviside'):
        s._kernel_dc('g')


# ── 2. scalar model, Conv on a bare field ────────────────────────────────────────────────────

def _scalar_hawkes(population, conv_arg='n'):
    """Voltage Hawkes model: n ~ Poisson(a exp(v)), v filters n through the kernel g."""
    b = TemporalTheoryBuilder('scalar conv')
    kw = {}
    if population:
        b.population('E', size=1)
        kw = dict(population='E')
    ix = '[i]' if population else ''
    b = (b.physical_field('n', **kw).physical_field('v', **kw)
         .parameter('Em', default=-1.2).parameter('tau', default=2.0, domain='positive')
         .parameter('taug', default=1.5, domain='positive').parameter('a', default=1.0, domain='positive')
         .parameter('w', default=0.4)
         .define_kernel('g', freq_image='1/(1 + I*omega*taug)'))
    body = (f'nt{ix}*n{ix} - (exp(nt{ix}) - 1)*a*exp(v{ix}) + '
            f'vt{ix}*((tau*Dt + 1)*v{ix} - Em - w*Conv(g, {conv_arg.replace("n", "n" + ix)}))')
    # the kernel's integral is 1: the mean-field equations carry no Conv
    for lhs, rhs in ((f'n{ix}', f'a*exp(v{ix})'), (f'v{ix}', f'Em + w*({conv_arg.replace("n", "n" + ix)})')):
        b = b.equation(lhs=lhs, rhs=rhs, **kw)
    return b.set_action_text(f'sum({body} for i in E)' if population else body).build()


def test_scalar_model_conv_on_bare_field_equals_one_element_population_twin():
    ra = _compare(lambda: _scalar_hawkes(False), lambda: _scalar_hawkes(True), [('dn', 1), ('dn', 1)], ell=1)
    assert any(np.abs(np.asarray(ra['C_tau_by_ell'][1])) > 0)


def test_scalar_conv_of_an_expression_in_the_field_equals_the_population_twin():
    """Adversarial: a wrapped field inside arithmetic (``Conv(g, 2*n + 0.1*n*n)``), tree level."""
    arg = '2*n + 0.1*n*n'
    ra = _compare(lambda: _scalar_hawkes(False, arg), lambda: _scalar_hawkes(True, arg),
                  [('dn', 1), ('dn', 1)], ell=0)
    assert any(np.abs(np.asarray(ra['C_tau'])) > 0)


def test_scalar_conv_still_validates_arity():
    from engine.core.field_theory import FieldTheory
    m = TemporalTheoryBuilder('bad').physical_field('n').parameter('a', default=1.0) \
        .define_kernel('g', freq_image='1/(1 + I*omega)') \
        .set_action_text('nt*n - (exp(nt) - 1)*a*Conv(n)').equation(lhs='n', rhs='a').build()
    with pytest.raises(ValueError, match='Conv'):               # an arity error, not a coercion error
        FieldTheory(m, taylor_order=2).expand()


# ── 3. _is_single_function_call ──────────────────────────────────────────────────────────────

@pytest.mark.parametrize('text, expected', [
    ('phi(v)', True),
    ('phi[i](vstar[i])', True),
    ('phi[i, j](a*(b + c))', True),
    ('  phi ( x )  ', True),
    ('f(g(a))', True),
    ('f(g(a), h(b))', True),
    ('f(a + b) ', True),
    ('f(a) + g(b)', False),
    ('f(a)*g(b)', False),
    ('f(a) + 1', False),
    ('f(a)(b)', False),
    ('(f(a))', False),
    ('2*f(a)', False),
    ('f(a) / (1 + g(b))', False),
    ('f(a)+g(b)', False),
    ('f[i](a) + g[i](b)', False),
    ('f((a))', True),
    ('f(g(a)) + h(b)', False),
    ('f(a', False),
    ('f(a))', False),
    ('', False),
    ('f(")") + g(x)', False),                                   # a quoted parenthesis
    ('f("(")', True),
])
def test_is_single_function_call(text, expected):
    assert _is_single_function_call(text) is expected


# ── 4. SDE.notes ─────────────────────────────────────────────────────────────────────────────

def _noted_sde():
    """Both a Gaussian-mean note and a moved-linear-term note per derivation."""
    return (SDE('notes').parameter('mu', default=-1.0)
            .field('x').noise('xi', Gaussian(var='2', mean='0.5'), couples={'x': 1})
            .equation('x', lhs='Dt*x', rhs='mu*x - x^3'))


def test_sde_notes_are_not_duplicated_by_repeated_derivations():
    s = _noted_sde()
    s.equations()
    first = list(s.notes)
    assert len(first) == 2 and len(set(first)) == 2
    s.equations()
    s.build()
    s.action_text()
    s.to_builder()
    assert s.notes == first
    s.show()
    assert s.notes == first
