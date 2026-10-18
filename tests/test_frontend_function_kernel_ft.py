"""Two front-end gaps.

1. A declared function applied to an expression (``phi[i](Em[i] + sum(w[i,j]*v[j] ...))``,
   ``g(a*x)``, or a function whose formal argument names no field) has no saddle-point Taylor
   rename; such calls are inlined, so the define_function form and the body written out by hand
   give the same action, saddle, propagator and cumulants, bit for bit.
2. A kernel given only by ``time_expr`` as a causal sum of exponentials with symbolic time
   constants (``(exp(-t/tau) - exp(-t/taug))/(tau - taug)*heaviside(t)``) is transformed in closed
   form; the image equals the hand-written ``freq_image`` and the cumulants are bit-identical.

Every bit-for-bit comparison runs the first form once and discards it (cold and warm caches differ
by ~1e-16)."""
import numpy as np
import pytest

pytest.importorskip('sage.all')
from sage.all import SR, assumptions, exp, heaviside, I, var        # noqa: E402

import api.compute as AC                                            # noqa: E402
from api.model_compiler import causal_exponential_ft                # noqa: E402
from api.noise import Gaussian                                      # noqa: E402
from api.sde import SDE                                             # noqa: E402
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


def _raw_action(model):
    """The action before expansion (formal calls still visible)."""
    from engine.core.field_theory import FieldTheory
    ft = FieldTheory(model, taylor_order=2)
    ft.expand()                                                 # fills the namespace
    return str(model['action'](ft._ns))


def _check_pair(build_a, build_b, ext, ell=1):
    _run(build_a(), ext, ell)                                   # warm-up, discarded
    ra, rb = _run(build_a(), ext, ell), _run(build_b(), ext, ell)
    _same_run(ra, rb)
    assert _action(build_a()) == _action(build_b())
    return ra


# ── Gap 1: declared functions applied to expressions ─────────────────────────────────────────

def _hawkes_1pop(fn):
    """One population of one neuron, voltage Hawkes, phi applied to an affine expression of the
    voltage.  (Without a declared population the legacy single-'nstar' closure cannot take a
    concrete saddle text that references vstar, with or without define_function; see the report.)"""
    b = (TemporalTheoryBuilder('fn affine 1pop')
         .population('E', size=1)
         .physical_field('n', population='E').physical_field('v', population='E')
         .parameter('Em', default=-1.0).parameter('c', default=0.5)
         .parameter('tau', default=2.0, domain='positive').parameter('a', default=1.0, domain='positive')
         .parameter('w', default=0.4))
    if fn:
        b.define_function('phi', args=['v'], expression='a*exp(v)', population='E')
        rate = 'phi[i](Em + c*{x}[i])'
    else:
        rate = 'a*exp(Em + c*{x}[i])'
    return (b.set_action_text('sum(nt[i]*n[i] - (exp(nt[i]) - 1)*' + rate.format(x='v')
                              + ' + vt[i]*((tau*Dt + 1)*v[i] - w*n[i]) for i in E)')
            .set_mf_equation('vstar', 'w*nstar[i]')
            .set_mf_equation('nstar', rate.format(x='vstar'))
            .build())


def _hawkes_2pop(fn, name='fn weighted sum 2pop'):
    """Two neurons; each rate is phi of a weighted sum of the other voltages; the voltages filter
    the spikes through a kernel in a LINEAR drift term."""
    b = (TemporalTheoryBuilder(name)
         .population('E', size=2)
         .physical_field('n', population='E').physical_field('v', population='E')
         .parameter('Em', default=[-1.2, -1.0], indexed_by=['E'])
         .parameter('tau', default=2.0, domain='positive')
         .parameter('taug', default=1.5, domain='positive')
         .parameter('a', default=1.0, domain='positive')
         .parameter('w', default=[[0.0, 0.4], [0.3, 0.0]], indexed_by=['E', 'E'])
         .define_kernel('g', freq_image='1/(1 + I*omega*taug)'))
    if fn:
        b.define_function('phi', args=['v'], expression='a*exp(v)', population='E')
        rate = 'phi[i](Em[i] + sum(w[i,j]*{x}[j] for j in E))'
    else:
        rate = 'a*exp(Em[i] + sum(w[i,j]*{x}[j] for j in E))'
    return (b.set_action_text(
                'sum(nt[i]*n[i] - (exp(nt[i]) - 1)*' + rate.format(x='v')
                + ' + vt[i]*((tau*Dt + 1)*v[i] - Conv(g, n[i])) for i in E)')
            .set_mf_equation('vstar', 'nstar[i]')
            .set_mf_equation('nstar', rate.format(x='vstar'))
            .build())


def test_function_of_affine_expression_1pop():
    """(a) nstar = phi[i](Em + c*vstar[i]) is a compound saddle and phi is inlined in the action."""
    r = _check_pair(lambda: _hawkes_1pop(True), lambda: _hawkes_1pop(False), [('dn', 1), ('dn', 1)])
    assert np.isfinite(np.asarray(r['C_tau_by_ell'][1])).all()


def _expanded(model):
    from engine.core.field_theory import FieldTheory
    ft = FieldTheory(model, taylor_order=4)
    ft.expand()
    bg = {str(k): str(v) for k, v in model['mf_bg_conditions'](ft._ns).items()}
    return {bg_: str(p) for bg_, p in ft._by_tp.items()}, bg, model.get('iteration_saddles')


def test_function_of_weighted_sum_2pop_action():
    """(b), fast: same expanded action, MF substitution and iteration saddles."""
    assert _expanded(_hawkes_2pop(True)) == _expanded(_hawkes_2pop(False))


@pytest.mark.slow
def test_function_of_weighted_sum_2pop():
    """(b) heterogeneous population: phi[i] applied to a weighted sum of the other voltages.
    (~2.5 min: three runs, each ~33 s of symbolic propagator.)"""
    r = _check_pair(lambda: _hawkes_2pop(True), lambda: _hawkes_2pop(False), [('dn', 1), ('dn', 2)])
    assert any(np.abs(np.asarray(r['C_tau_by_ell'][1])) > 0)


@pytest.mark.slow
def test_function_with_no_saddle_argument_spike_only_mean_field():
    """The reported failure: phi's formal argument 'v' is no field, and the MF equation applies it
    to a weighted sum of saddles.  The 2-pop spike-only form builds and agrees with the explicit
    form at tree level (its loop vertices are kernel-leg vertices, not exercised here)."""
    def build(fn):
        b = (TemporalTheoryBuilder('fn no saddle arg')
             .population('E', size=2).physical_field('n', population='E')
             .parameter('Em', default=[-1.5, -1.5], indexed_by=['E'])
             .parameter('tau', default=[2.0, 2.0], indexed_by=['E'], domain='positive')
             .parameter('taug', default=[[1.0, 1.0], [1.0, 1.0]], indexed_by=['E', 'E'], domain='positive')
             .parameter('a', default=[1.0, 1.0], indexed_by=['E'], domain='positive')
             .parameter('w', default=[[0.4, 0.4], [0.4, 0.4]], indexed_by=['E', 'E'])
             .define_kernel('r', freq_image='1/((1 + I*omega*tau[i])*(1 + I*omega*taug[i,j]))',
                            indexed_by=['E', 'E']))
        arg = 'Em[i] + sum(w[i,j]*{x} for j in E)'
        if fn:
            b.define_function('phi', ['v'], 'a[i]*exp(v)')
            rate = 'phi[i](' + arg + ')'
        else:
            rate = 'a[i]*exp(' + arg + ')'
        return (b.set_action_text('sum(nt[i]*n[i] - (exp(nt[i]) - 1)*'
                                  + rate.format(x='Conv(r[i,j], n[j])') + ' for i in E)')
                .set_mf_equation('nstar', rate.format(x='nstar[j]'))
                .build())
    _check_pair(lambda: build(True), lambda: build(False), [('dn', 1), ('dn', 2)], ell=0)


def _mixed(fn):
    """The same function both on its own field (formal) and on a shifted argument (inlined)."""
    b = (TemporalTheoryBuilder('fn mixed sites')
         .population('E', size=1)
         .physical_field('n', population='E').physical_field('v', population='E')
         .parameter('Em', default=-1.0).parameter('c', default=0.3)
         .parameter('a', default=0.5, domain='positive').parameter('w', default=0.4))
    if fn:
        b.define_function('phi', args=['v'], expression='a*exp(v)', population='E')
        rate = '(phi[i]({x}[i]) + phi[i]({x}[i] + c))'
    else:
        rate = '(a*exp({x}[i]) + a*exp({x}[i] + c))'
    return (b.set_action_text('sum(nt[i]*n[i] - (exp(nt[i]) - 1)*' + rate.format(x='v')
                              + ' + vt[i]*((Dt + 1)*v[i] - Em - w*n[i]) for i in E)')
            .set_mf_equation('vstar', 'Em + w*nstar[i]')
            .set_mf_equation('nstar', rate.format(x='vstar'))
            .build())


def test_formal_and_inlined_sites_of_one_function():
    """Adversarial: phi[i](v[i]) keeps the formal path while phi[i](v[i] + c) is inlined; the result
    agrees with the explicit form to rounding (the formal path is a different, equal expression)."""
    raw = _raw_action(_mixed(True))
    assert 'phi_1(dv1 + vstar1)' in raw and 'phi_1(c' not in raw and 'e^(c + dv1' in raw
    ext = [('dn', 1), ('dn', 1)]
    ra, rb = _run(_mixed(True), ext), _run(_mixed(False), ext)
    for k in ra['mf_values']:
        np.testing.assert_allclose(ra['mf_values'][k], rb['mf_values'][k], rtol=1e-12)
    for ell in rb['C_tau_by_ell']:
        np.testing.assert_allclose(np.asarray(ra['C_tau_by_ell'][ell]),
                                   np.asarray(rb['C_tau_by_ell'][ell]), rtol=1e-12, atol=1e-14)


def _sde(variant):
    s = SDE('fn sde ' + variant).population('pop', size=1).field('x', population='pop')
    for p, d in (('a', 1.5), ('b', 0.3), ('D', 0.5)):
        s.parameter(p, default=d, domain='positive')
    if variant == 'explicit':
        s.drift('x', 'b - x - 0.1*tanh(a*x)')
    elif variant == 'arg_v':                 # formal argument names no field
        s.define_function('g', ['v'], 'tanh(a*v)')
        s.drift('x', 'b - x - 0.1*g(x)')
    elif variant == 'affine':                # applied to an expression of the field
        s.define_function('g', ['x'], 'tanh(x)')
        s.drift('x', 'b - x - 0.1*g(a*x)')
    elif variant == 'field':                 # applied to the field itself: formal path, unchanged
        s.define_function('g', ['x'], 'tanh(a*x)')
        s.drift('x', 'b - x - 0.1*g(x)')
    s.noise('xi', Gaussian(var='2*D'), couples={'x': 1})
    return s.build()


@pytest.mark.parametrize('variant', ['arg_v', 'affine'])
def test_sde_defined_function_in_drift(variant):
    """(c) the SDE front end: g in the drift, nonzero saddle."""
    _check_pair(lambda: _sde(variant), lambda: _sde('explicit'), [('dx', 1), ('dx', 1)])


def test_sde_function_of_the_field_keeps_the_formal_path():
    """g(x) with g's formal argument the field x stays a formal call and agrees with the explicit
    drift to rounding."""
    assert 'g_1(' in _raw_action(_sde('field')) and 'g_1(' not in _raw_action(_sde('affine'))
    ra, rb = _run(_sde('field'), [('dx', 1), ('dx', 1)]), _run(_sde('explicit'), [('dx', 1), ('dx', 1)])
    for ell in ra['C_tau_by_ell']:
        np.testing.assert_allclose(np.asarray(ra['C_tau_by_ell'][ell]),
                                   np.asarray(rb['C_tau_by_ell'][ell]), rtol=1e-12, atol=1e-14)


def test_formal_call_site_rule():
    """Formal iff the arguments at zero fluctuation are all the saddles or all 0 (the two Taylor
    rename points); otherwise the body is inlined."""
    from api.model_compiler import (_IndexedFormalFunction, _IndexedSaddleRename,
                                    _make_function_callable)
    vs, dv, c = SR.var('vstarQ1'), SR.var('dvQ1'), SR.var('cQ')
    spec = {'name': 'phiQ', 'args': ['v'], 'expression': 'c*exp(v)'}
    fb = _make_function_callable(spec, {'c': c})
    f = _IndexedFormalFunction('phiQ', fallback=fb, saddles=[[vs]], zero_subs={dv: 0})
    assert str(f[0](vs + dv)) == 'phiQ_1(dvQ1 + vstarQ1)'
    assert str(f(vs + dv)) == 'phiQ_1(dvQ1 + vstarQ1)'
    assert str(f[0](dv)) == 'phiQ_1(dvQ1)'                      # all-zero point: formal, as before
    assert f[0](2*vs + dv) == c*exp(2*vs + dv)
    g = _IndexedFormalFunction('phiQ', fallback=fb, saddles=None, zero_subs={dv: 0})
    assert g[0](vs + dv) == c*exp(vs + dv)
    assert str(_IndexedFormalFunction('phiQ')[0](2*vs)) == 'phiQ_1(2*vstarQ1)'   # no fallback
    r = _IndexedSaddleRename('phiQ', fallback=fb, saddles=[[vs]])
    assert str(r([vs])) == 'phiQ0_1' and str(r[0](vs)) == 'phiQ0_1'   # bare saddle list: renamed
    assert r[0](vs + 1) == c*exp(vs + 1)
    # a bare call in a multi-element population is not silently element 0
    v2 = [SR.var('vstarQ1'), SR.var('vstarQ2')]
    spec_i = {'name': 'phiQ', 'args': ['v'], 'expression': 'a[i]*exp(v)'}
    h = _IndexedFormalFunction('phiQ', fallback=_make_function_callable(
        spec_i, {'a': [SR.var('aQ1'), SR.var('aQ2')]}), saddles=[v2], zero_subs={dv: 0})
    with pytest.raises(ValueError, match=r'phiQ\[i\]'):
        h(vs + 1)
    assert h[1](vs + 1) == SR.var('aQ2')*exp(vs + 1)


def test_mf_classification_of_inlined_calls():
    """A single call is an iteration (closure) saddle only when it has a rename target."""
    from api.model_compiler import _call_has_rename_target as ok
    fns = [{'name': 'phi', 'args': ['v']}, {'name': 'f', 'args': ['v', 'n']}]
    names = {'vstar', 'nstar'}
    assert ok('phi(vstar[i])', fns, names) and ok('phi[i]( (vstar[i]) )', fns, names)
    assert ok('f[i](vstar[i], nstar[i])', fns, names) and ok('exp(vstar[i])', fns, names)
    assert not ok('phi[i](Em[i] + sum(w[i,j]*vstar[j] for j in E))', fns, names)
    assert not ok('phi(2*vstar[i])', fns, names)
    assert not ok('f[i](nstar[i], vstar[i])', fns, names)
    assert not ok('phi(vstar[i])', [{'name': 'phi', 'args': ['u']}], names)
    assert not ok('phi[i](vstar[j])', fns, names) and not ok('phi[i](vstar[0])', fns, names)
    assert ok('phi(vstar)', fns, names) and ok('phi[k](vstar[ k ])', fns, names)
    assert ok('phi[i, j](vstar[j])', fns, names)          # tuple index: never inlined
    assert ok('phi(vstar[i])', None, names)


# ── Gap 2: time_expr kernels with symbolic time constants ────────────────────────────────────

def test_causal_exponential_ft_closed_forms():
    t, w = var('t omega')
    tau, taug = SR.var('tauX', domain='positive'), SR.var('taugX', domain='positive')
    before = sorted(map(str, assumptions()))
    img = causal_exponential_ft((exp(-t/tau) - exp(-t/taug))/(tau - taug)*heaviside(t), t, w)
    ref = 1/((1 + I*w*tau)*(1 + I*w*taug))
    assert (img - ref).is_trivial_zero() and str(img) == str(ref)
    alpha = causal_exponential_ft(t*exp(-t/tau)/tau**2*heaviside(t), t, w)
    assert (alpha - 1/(1 + I*w*tau)**2).simplify_rational().is_zero()
    assert sorted(map(str, assumptions())) == before
    x = SR.var('xX')
    with pytest.raises(ValueError, match='not known to be positive'):
        causal_exponential_ft(exp(-t/x)*heaviside(t), t, w)
    with pytest.raises(ValueError, match='heaviside'):
        causal_exponential_ft(exp(-t/tau), t, w)
    diff = (exp(-t/tau) - exp(-t/taug))/(tau - taug)*heaviside(t)
    assert str(causal_exponential_ft(diff, t, w, positive={'tauX', 'taugX'})) == str(ref)
    with pytest.raises(ValueError, match='not known to be positive'):     # by declaration only
        causal_exponential_ft(diff, t, w, positive={'tauX'})


def _kernel_model(kind, n, tau_g=1.0, te=None, fi=None, taug_domain='positive'):
    """Voltage Hawkes with a rise-and-decay kernel r[i] in the linear voltage drift.  One
    population: per-population (indexed) time constants; two: shared ones (four distinct indexed
    time constants make the symbolic 6x6 propagator alone take ~2 min, either kernel form)."""
    ix = '[i]' if n == 1 else ''
    te = te or f'(exp(-t/tau{ix}) - exp(-t/taug{ix}))/(tau{ix} - taug{ix})*heaviside(t)'
    fi = fi or f'1/((1 + I*omega*tau{ix})*(1 + I*omega*taug{ix}))'
    kw = dict(time_expr=te) if kind == 'time' else dict(freq_image=fi)
    tg = dict(domain=taug_domain) if taug_domain else {}
    per = dict(indexed_by=['E']) if n == 1 else {}
    return (TemporalTheoryBuilder(f'kernel ft {n}pop')
            .population('E', size=n)
            .physical_field('n', population='E').physical_field('v', population='E')
            .parameter('Em', default=[-1.2, -1.0][:n], indexed_by=['E'])
            .parameter('tau', default=[2.0] if n == 1 else 2.0, domain='positive', **per)
            .parameter('taug', default=[tau_g] if n == 1 else tau_g, **per, **tg)
            .parameter('a', default=[1.0, 0.8][:n], indexed_by=['E'], domain='positive')
            .parameter('w', default=0.3)
            .define_kernel('r', indexed_by=['E'], **kw)
            .set_action_text('sum(nt[i]*n[i] - (exp(nt[i]) - 1)*a[i]*exp(v[i]) + vt[i]*('
                             '(Dt + 1)*v[i] - Em[i] - sum(w*Conv(r[i], n[j]) for j in E)) for i in E)')
            .set_mf_equation('vstar', 'Em[i] + sum(w*nstar[j] for j in E)')
            .set_mf_equation('nstar', 'a[i]*exp(vstar[i])')
            .build())


def _images(model):
    from engine.core.field_theory import FieldTheory
    ft = FieldTheory(model, taylor_order=2)
    ft.expand()                                                 # fills the namespace
    out = {str(k): v for k, v in model['kernel_ft_image'](ft._ns, SR.var('omega')).items()}
    assert out
    return out


@pytest.mark.parametrize('n', [1, 2])
def test_time_expr_kernel_image_equals_freq_image(n):
    t_img, f_img = _images(_kernel_model('time', n)), _images(_kernel_model('freq', n))
    assert t_img.keys() == f_img.keys()
    for k in t_img:
        assert (t_img[k] - f_img[k]).is_trivial_zero(), (k, t_img[k], f_img[k])


@pytest.mark.parametrize('n', [1, pytest.param(2, marks=pytest.mark.slow)])
def test_time_expr_kernel_cumulants_bit_identical(n):
    """k = 2, tree + one loop; no assumption added by the time_expr run.  (n = 2: ~3 min, the
    symbolic propagator build.)"""
    ext = [('dn', 1), ('dn', n)]
    _run(_kernel_model('freq', n), ext)                         # warm-up
    held = sorted(map(str, assumptions()))
    rt = _run(_kernel_model('time', n), ext)
    assert sorted(map(str, assumptions())) == held              # no assumption leaks
    _same_run(_run(_kernel_model('freq', n), ext), rt)


def test_time_expr_kernel_removable_singularity():
    """tau = taug: the time_expr is 0/0 there, its factored image is regular."""
    ext = [('dn', 1), ('dn', 1)]
    _run(_kernel_model('freq', 1, tau_g=2.0), ext)
    _same_run(_run(_kernel_model('freq', 1, tau_g=2.0), ext), _run(_kernel_model('time', 1, tau_g=2.0), ext))


def test_time_expr_kernel_unsigned_parameter_refused_clearly():
    """No domain='positive' on a time constant: the error names it and points to freq_image."""
    _images(_kernel_model('time', 1))            # taug1 declared positive by this model ...
    with pytest.raises(ValueError, match=r"'taug'.*freq_image"):
        _images(_kernel_model('time', 1, taug_domain=None))     # ... does not carry over


def test_single_exponential_time_expr_unchanged():
    """A kernel Sage already transforms keeps Sage's image (the structural rule is a fallback)."""
    from engine.core.field_theory import fourier_transform
    img = _images(_kernel_model('time', 1, te='exp(-t/taug[i])/taug[i]*heaviside(t)'))
    t, w = SR.var('t'), SR.var('omega')
    taug = next(v for v in img['z_r_1'].variables() if str(v) == 'taug1')   # the model's symbol
    assert str(img['z_r_1']) == str(SR(fourier_transform(exp(-t/taug)/taug*heaviside(t), t, w)))
