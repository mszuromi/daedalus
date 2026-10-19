"""Mean-field / saddle fixes (work item 52), each reproduced on main before its fix:

1. A kernel whose integral is not 1 (``freq_image='2/(1+I*omega)'``) inside ``Conv`` stopped the run
   with "mean-field saddle does not satisfy its own equations": the engine reduced
   ``Conv(K, constant)`` to the constant (normalised-kernel rule) while the mean-field equations carry
   the integral.  The engine now takes ``K(omega = 0)`` there (and for a kernel symbol left in a
   hand-written mean-field text); a model with a non-normalised kernel equals its twin with the
   normalised kernel and the integral moved into the weight.
2. (a) The numerical mean-field check ran only at the default ``fixed_point_index``; the selected
   root is now checked too.  (b) A field with no declared domain was seeded on ``[0, 5s]`` only, so
   ``fixed_point_index`` could not reach the negative root of ``x - x^3``.  (c) ``analyze`` merged
   mean-field defaults into the parameter point (``compute_cumulants`` does not).
3. A scalar model (no ``.population()``) whose ``set_mf_equation`` text names a saddle bare
   (``'a*exp(Em + c*vstar)'``, not ``vstar[i]``) failed to parse; it now equals the 1-element
   population form.

Every bit-for-bit comparison runs the first form once and discards it (cold and warm caches differ
by ~1e-16)."""
import warnings

import numpy as np
import pytest

pytest.importorskip('sage.all')

import api.compute as AC                                            # noqa: E402
from api.noise import Gaussian, Poisson                             # noqa: E402
from api.sde import SDE                                             # noqa: E402

TAU = np.array([-0.8, 0.0, 0.5, 1.3])
NN = [('n', 1), ('n', 1)]


def _run(model, ext, ell=1, **kw):
    kw.setdefault('tau_grid', TAU)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return AC.compute_cumulants(model, k=2, max_ell=ell, external_fields=ext,
                                    use_cache=False, parallel=False, verbose=False, **kw)


def _close(a, b, rtol=1e-12):
    for key in a['mf_values']:
        assert np.allclose(a['mf_values'][key], b['mf_values'][key], rtol=rtol, atol=0), key
    assert set(a['C_tau_by_ell']) == set(b['C_tau_by_ell'])
    for ell in a['C_tau_by_ell']:
        x, y = np.asarray(a['C_tau_by_ell'][ell]), np.asarray(b['C_tau_by_ell'][ell])
        assert np.allclose(x, y, rtol=rtol, atol=1e-15 * np.max(np.abs(y))), (ell, x, y)


# ── 1. kernels whose integral is not 1 ───────────────────────────────────────────────────────

def _pp(amp, weight, name='pp52', delay=None):
    s = SDE(name).parameter('d', default=0.4, domain='positive')
    kw = dict(freq_image=f'{amp}/(1+I*omega)')
    if delay:
        kw['delay'] = delay
    return s.define_kernel('K', **kw).point_process('n', f'exp(-1 + {weight}*Conv(K, n))').build()


def _voltage(amp, weight, name='v52'):
    return (SDE(name).define_kernel('K', freq_image=f'{amp}/(1+I*omega)').field('v')
            .noise('s', Poisson(rate='exp(v)'), expose='n')
            .equation('v', lhs='Dt*v + v', rhs=f'-1 + {weight}*Conv(K, n)').build())


def test_non_unit_kernel_point_process_saddle_is_the_fixed_point():
    mf = _run(_pp(2, 0.1), NN, ell=0)['mf_values']['nstar'][0]
    assert mf == pytest.approx(np.exp(-1 + 0.2 * mf), rel=1e-12)       # n = exp(-1 + 0.1*2*n)


@pytest.mark.parametrize('amp, weight', [(2, 0.1), (-1, -0.2), ('0.5', 0.4)])
def test_non_unit_kernel_equals_normalised_twin_point_process(amp, weight):
    """Integral 2, a negative integral, and 1/2: the integral moved into the weight gives the same
    saddle, tree and one loop."""
    twin = float(amp) * weight
    _close(_run(_pp(amp, weight), NN), _run(_pp(1, twin, 'pp52twin'), NN))


def test_non_unit_kernel_equals_normalised_twin_in_a_drift():
    _close(_run(_voltage(2, 0.1), NN), _run(_voltage(1, 0.2, 'v52twin'), NN))


def test_delayed_and_scaled_kernel_equals_normalised_twin():
    """A delayed kernel with integral 3: same saddle and tree as the delayed normalised kernel with
    the weight times 3 (exact delayed tree; one loop through the Pade copies)."""
    a = _run(_pp(3, 0.05, 'pp52d', delay='d'), NN, ell=0)
    b = _run(_pp(1, 0.15, 'pp52dtwin', delay='d'), NN, ell=0)
    _close(a, b)
    n = a['mf_values']['nstar'][0]
    assert n == pytest.approx(np.exp(-1 + 0.15 * n), rel=1e-12)


def _indexed(scaled):
    """Two neurons, a matrix kernel g[i,j] with per-entry integrals A[i,j] (or 1, with the weights
    times A[i,j])."""
    A = [[2.0, -0.5], [1.5, 3.0]]
    w = [[0.05, 0.1], [-0.08, 0.04]]
    s = (SDE('pp52ix' + ('s' if scaled else 'n')).population('E', size=2)
         .parameter('Em', default=[-1.0, -1.2], indexed_by=['E'])
         .parameter('tau', default=[[1.0, 0.7], [0.5, 1.3]], indexed_by=['E', 'E'], domain='positive'))
    if scaled:
        s = (s.parameter('A', default=A, indexed_by=['E', 'E'])
             .parameter('w', default=w, indexed_by=['E', 'E'])
             .define_kernel('g', freq_image='A[i,j]/(1+I*omega*tau[i,j])', indexed_by=['E', 'E']))
    else:
        s = (s.parameter('w', default=[[w[i][j] * A[i][j] for j in range(2)] for i in range(2)],
                         indexed_by=['E', 'E'])
             .define_kernel('g', freq_image='1/(1+I*omega*tau[i,j])', indexed_by=['E', 'E']))
    return s.point_process('n', 'exp(Em[i] + sum(w[i,j]*Conv(g[i,j], n[j]) for j in E))',
                           population='E').build()


def test_indexed_kernel_with_per_entry_integrals_equals_normalised_twin():
    ext = [('n', 1), ('n', 2)]
    a, b = _run(_indexed(True), ext, ell=0), _run(_indexed(False), ext, ell=0)
    _close(a, b)
    n = np.asarray(a['mf_values']['nstar'])
    A = np.array([[2.0, -0.5], [1.5, 3.0]])
    w = np.array([[0.05, 0.1], [-0.08, 0.04]])
    assert np.allclose(n, np.exp(np.array([-1.0, -1.2]) + (w * A) @ n), rtol=1e-12, atol=1e-12)


def test_normalised_kernel_expansion_is_unchanged():
    """``Conv(K, constant)`` with a normalised kernel: the reduction returns the constant itself
    (no ``1*`` factor), so the expanded action is the one built without the integrals."""
    from engine.core.convolution import reduce_conv_in_action, Conv, KernelDCGains
    from sage.all import SR, var
    g, ns_, dn = var('g52 nstar52 dn52')

    class NS:
        g52 = g
    model = {'kernels': [{'name': 'g52'}], 'kernel_ft_image': lambda ns, om: {g: 1 / (1 + om)}}
    expr = Conv(g, ns_ + dn) * (1 + dn)
    plain = reduce_conv_in_action(expr, {dn})
    with_gain = reduce_conv_in_action(expr, {dn}, dc_gains=KernelDCGains(NS(), model))
    assert str(plain) == str(with_gain)
    model2 = {'kernels': [{'name': 'g52'}], 'kernel_ft_image': lambda ns, om: {g: 2 / (1 + om)}}
    scaled = reduce_conv_in_action(expr, {dn}, dc_gains=KernelDCGains(NS(), model2))
    assert bool((scaled - plain - ns_ * (1 + dn)).expand() == 0)


def test_kernel_integral_through_a_removable_singularity():
    """A boxcar ``(1 - exp(-2 I omega))/(2 I omega)`` and ``2 sin(3 omega)/(3 omega)`` are 0/0 at
    omega = 0 but integrate to 1 and 2; a pole (``1/(I omega)``) is refused.  The hook runs only for
    an expression that holds a kernel symbol."""
    from engine.core.convolution import KernelDCGains
    from sage.all import SR, I, exp, sin, var
    g, h, k = var('g52b h52b k52b')

    class NS:
        pass
    ns = NS()
    ns.g, ns.h, ns.k = g, h, k
    calls = []

    def hook(ns_, om):
        calls.append(1)
        return {g: (1 - exp(-2 * I * om)) / (2 * I * om), h: 2 * sin(3 * om) / (3 * om),
                k: 1 / (I * om)}
    kd = KernelDCGains(ns, {'kernels': [{'name': 'g'}, {'name': 'h'}, {'name': 'k'}],
                            'kernel_ft_image': hook})
    assert kd.subs_dict(var('x52b') + 1) == {} and not calls
    assert kd.symbol_gain(g) == 1 and kd.symbol_gain(h) == 2
    with pytest.raises(ValueError, match='singular at omega = 0'):
        kd.symbol_gain(k)
    assert len(calls) == 1


def _builder_pp(freq, mf_rate):
    """The point process written by hand: ``n ~ Poisson(exp(-1 + 0.1*Conv(K, n)))``."""
    from api.model import TemporalModelBuilder
    return (TemporalModelBuilder('pp52b').physical_field('n')
            .define_kernel('K', freq_image=freq)
            .set_action_text('nt*n - (exp(nt) - 1)*exp(-1 + 0.1*Conv(K, n))')
            .set_mf_equation('nstar', mf_rate).build())


def test_hand_written_model_with_non_unit_kernel_builds():
    """The engine side alone (no SDE front end): the hand-written mean-field text carries the
    integral, the action carries ``Conv`` of the non-normalised kernel."""
    a = _run(_builder_pp('2/(1+I*omega)', 'exp(-1 + 0.2*nstar[i])'), NN)
    b = _run(_pp(2, 0.1), NN)
    _close(a, b)


def test_kernel_without_finite_integral_is_refused_by_name():
    with pytest.raises(ValueError, match='singular at omega = 0'):
        _run(_builder_pp('1/(I*omega)', 'exp(-1 + 0.1*nstar[i])'), NN, ell=0)


# ── 3. scalar model, bare saddle names in set_mf_equation ────────────────────────────────────

def _hawkes(population, mf_v, mf_n, fn=False):
    """n ~ Poisson(a exp(Em + c v)), (tau Dt + 1) v = w n; the saddle by ``set_mf_equation``."""
    from api.theory import TemporalTheoryBuilder
    b = TemporalTheoryBuilder('hawkes52')
    kw, ix = {}, ''
    if population:
        b.population('E', size=1)
        kw, ix = dict(population='E'), '[i]'
    b = (b.physical_field('n', **kw).physical_field('v', **kw)
         .parameter('Em', default=-1.0).parameter('c', default=0.5)
         .parameter('tau', default=2.0, domain='positive')
         .parameter('a', default=1.0, domain='positive').parameter('w', default=0.4))
    if fn:
        b = b.define_function('phi', args=['v'], expression='a*exp(Em + c*v)', **kw)
        rate = f'phi(v{ix})'
    else:
        rate = f'a*exp(Em + c*v{ix})'
    body = f'nt{ix}*n{ix} - (exp(nt{ix}) - 1)*{rate} + vt{ix}*((tau*Dt + 1)*v{ix} - w*n{ix})'
    b = b.set_action_text(f'sum({body} for i in E)' if population else body)
    return b.set_mf_equation('vstar', mf_v).set_mf_equation('nstar', mf_n).build()


def _same_run(a, b):
    assert a['mf_values'] == b['mf_values']
    assert set(a['C_tau_by_ell']) == set(b['C_tau_by_ell'])
    for ell in a['C_tau_by_ell']:
        assert np.array_equal(np.asarray(a['C_tau_by_ell'][ell]), np.asarray(b['C_tau_by_ell'][ell]))


DN = [('dn', 1), ('dn', 1)]


@pytest.mark.parametrize('mf_v, mf_n, fn', [
    ('w*nstar', 'a*exp(Em + c*vstar)', False),           # the form of the open item, both bare
    ('w*nstar[i]', 'a*exp(Em + c*vstar)', False),        # mixed
    ('w*nstar', 'phi(vstar)', True),                     # define_function, closure call
])
def test_scalar_bare_saddle_names_equal_the_population_twin(mf_v, mf_n, fn):
    twin_v = mf_v if '[i]' in mf_v else mf_v + '[i]'
    twin_n = mf_n.replace('vstar)', 'vstar[i])').replace('vstar)', 'vstar[i])')
    if fn:
        twin_n = 'phi[i](vstar[i])'
    _run(_hawkes(False, mf_v, mf_n, fn), DN)                  # discarded warm-up
    a = _run(_hawkes(False, mf_v, mf_n, fn), DN)
    b = _run(_hawkes(True, twin_v, twin_n, fn), DN)
    _same_run(a, b)
    assert a['mf_values']['nstar'][0] > 0 and np.max(np.abs(np.asarray(a['C_tau_by_ell'][1]))) > 0


def test_scalar_bare_saddle_inconsistent_closure_still_fails():
    with pytest.raises(AssertionError, match='mean-field saddle does not satisfy'):
        _run(_hawkes(False, 'w*nstar', '2*a*exp(Em + c*vstar)'), DN, ell=0)


# ── 2. fixed_point_index ─────────────────────────────────────────────────────────────────────

XX = [('x', 1), ('x', 1)]


def _bistable(domain=None, rhs=None):
    sde = (SDE('bist52').field('x', domain=domain).drift('x', 'x - x**3')
           .noise('xi', Gaussian(var='0.1'), couples={'x': 1}))
    if rhs is None:
        return sde.build()
    b = sde.to_builder()
    b._equations[0]['rhs_text'] = rhs
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return b.build()


def test_selected_root_is_checked_not_only_the_default_one():
    """The equation drift ``-(x + 1) x (x - 2)`` (rhs ``-x^3 + x^2 + x`` next to the lhs
    ``Dt x - x``) shares the stable root -1 (index 0: stable roots first,
    where the expansion's check runs) with the action's drift ``x - x^3`` but not the stable root 2
    (index 1): that run must stop, not compute at a saddle the action does not have."""
    m = _bistable(rhs='-x**3 + x**2 + x')
    with pytest.raises(AssertionError, match=r'at the selected root \(fixed_point_index=1'):
        _run(m, XX, ell=0, fixed_point_index=1)


def test_consistent_model_passes_the_selected_root_check():
    r = _run(_bistable(), XX, ell=1, fixed_point_index=1)
    assert r['mf_values']['xstar'][0] == pytest.approx(1.0, abs=1e-12)


def test_negative_root_of_an_undeclared_field_is_selectable():
    """Undeclared domain: the default seeds cover x >= 0 (the historical box) and the mirrored seeds
    x <= 0; every root they reach is selectable (audit W6: which negative roots were selectable
    depended on the field's units).  Stability analysis off: the stable roots first, ascending."""
    from api._mean_field_dae import solve_mean_field_dae
    with warnings.catch_warnings():
        warnings.simplefilter('error', UserWarning)
        r = solve_mean_field_dae(_bistable(), {}, fixed_point_index=0)
    assert [x['values']['xstar'][0] for x in r['mf_all_roots']] == [
        pytest.approx(-1.0, abs=1e-12), pytest.approx(1.0, abs=1e-12), 0.0]
    assert r['mf_values']['xstar'][0] == pytest.approx(-1.0, abs=1e-12)
    assert r['mf_unsearched_roots'] == []


def test_declared_real_domain_reaches_the_negative_root():
    with warnings.catch_warnings():
        warnings.simplefilter('error', UserWarning)
        from api._mean_field_dae import solve_mean_field_dae
        r = solve_mean_field_dae(_bistable('real'), {}, fixed_point_index=0)
    assert r['mf_values']['xstar'][0] == pytest.approx(-1.0, abs=1e-12)
    assert r['mf_unsearched_roots'] == []
    a = _run(_bistable('real'), XX, ell=1, fixed_point_index=0)
    b = _run(_bistable('real'), XX, ell=1, fixed_point_index=1)     # the other well
    assert a['mf_values']['xstar'][0] == pytest.approx(-1.0, abs=1e-12)
    # x -> -x symmetry of x - x^3 with additive noise: the two wells have the same C(tau)
    for ell in (0, 1):
        assert np.allclose(a['C_tau_by_ell'][ell], b['C_tau_by_ell'][ell], rtol=1e-10, atol=1e-12)


# ── 2c. the analyzer's parameter point ───────────────────────────────────────────────────────

def _linear_hawkes(pin):
    import importlib.util
    import os
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        'models', 'linear_hawkes.model.py')
    spec = importlib.util.spec_from_file_location('lh_mf52', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    m = mod.build()
    if pin:
        for p in m['parameters']:
            if p.get('mean_field') and p['name'] == 'nstar':
                p['default'] = [5.0, 5.0]
    return m, dict(mod.DEFAULT_FUNDAMENTAL)


def test_analyzer_does_not_merge_a_saddle_default_into_given_parameters():
    from api.analyze import analyze
    saddles = []
    for pin in (False, True):
        m, params = _linear_hawkes(pin)
        r = analyze(m, question={'parameters': params, 'k': 2, 'max_ell': 0,
                                 'external_fields': [('n', 1), ('n', 1)]})
        saddles.append(dict(r.traits.saddle))
    assert saddles[0] == saddles[1]
    assert saddles[0]['nstar'][0] != 5.0


def test_analyzer_partial_parameters_wording_matches_compute_cumulants():
    from api.analyze import analyze
    m = (SDE('nd52').parameter('a', default=1.0).parameter('D').field('x').drift('x', '-a*x + 0.1')
         .noise('xi', Gaussian(var='2*D'), couples={'x': 1}).build())
    r = analyze(m, question={'parameters': {'a': 2.0}})
    f = [f for f in r.findings if f.code == 'G0_PARAMETERS_MISSING'][0]
    assert 'the ``parameters=`` given do not include them' in f.message and 'D' in f.message
    with pytest.raises(ValueError, match='the ``parameters=`` given do not include them'):
        _run(m, XX, ell=0, parameters={'a': 2.0})
    r = analyze(m, question={'parameters': {'a': 2.0, 'D': 0.5}})
    assert 'G0_PARAMETERS_MISSING' not in r.codes
