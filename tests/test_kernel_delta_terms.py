"""Compiled kernels accept ``c*dirac_delta(t)`` terms.

A kernel written ``time_expr='(0.3*dirac_delta(t) + exp(-t/tau)/tau)*heaviside(t)'`` was refused by
the compiled transform ("factor dirac_delta(t) is not t^n or an exponential").  The delta of a
causal kernel keeps its full weight (the constant ``c`` in the image, as in ``SDE._kernel_dc``), so
the kernel equals the one given by ``freq_image='0.3 + 1/(1 + I*omega*tau)'``: the same image, and
the same cumulants bit for bit, through the builder and the SDE front end.  A delta at another time
is a delay and is refused with a pointer to ``Kernel.delay``.

Every bit-for-bit comparison runs the first form once and discards it (cold and warm caches differ
by ~1e-16)."""
import numpy as np
import pytest

pytest.importorskip('sage.all')
from sage.all import SR, assumptions, dirac_delta, exp, heaviside, I, log, var   # noqa: E402

import api.compute as AC                                            # noqa: E402
from api.model_compiler import causal_exponential_ft                # noqa: E402
from api.noise import Gaussian                                      # noqa: E402
from api.sde import SDE                                             # noqa: E402
from api.theory import TemporalTheoryBuilder                        # noqa: E402

TAU = np.array([-0.6, 0.0, 1.2])
TE = '(0.3*dirac_delta(t) + exp(-t/tau)/tau)*heaviside(t)'
FI = '0.3 + 1/(1 + I*omega*tau)'


# ── the transform itself ──────────────────────────────────────────────────────────────────────

def test_delta_terms_are_the_constant_in_the_image():
    t, w = var('t omega')
    tau = SR.var('tauD', domain='positive')
    c = SR.var('cD')
    ref = c + 1/(1 + I*w*tau)
    for f in ((c*dirac_delta(t) + exp(-t/tau)/tau)*heaviside(t),         # inside the step
              c*dirac_delta(t) + exp(-t/tau)/tau*heaviside(t),            # outside the step
              c*dirac_delta(t)*heaviside(t) + exp(-t/tau)/tau*heaviside(t)):
        img = causal_exponential_ft(f, t, w)
        assert (img - ref).simplify_rational().is_zero(), (f, img)
    assert (causal_exponential_ft(c*dirac_delta(t), t, w) - c).is_zero()
    assert (causal_exponential_ft(c*dirac_delta(t)*heaviside(t), t, w) - c).is_zero()


def test_delta_term_t_factors_are_evaluated_at_the_delta():
    t, w = var('t omega')
    tau, c = SR.var('tauD', domain='positive'), SR.var('cD')
    # f(t) delta(t) = f(0) delta(t), as SDE._kernel_dc: exp(-t/tau) -> 1, t^n -> 0
    assert (causal_exponential_ft(c*exp(-t/tau)*dirac_delta(t), t, w) - c).is_zero()
    assert causal_exponential_ft(c*t*dirac_delta(t), t, w).is_zero()


def test_delta_at_another_time_is_refused_pointing_to_the_delay_api():
    t, w = var('t omega')
    tau = SR.var('tauD', domain='positive')
    for f in (dirac_delta(t - 1)*heaviside(t), (dirac_delta(t - tau) + exp(-t/tau))*heaviside(t)):
        with pytest.raises(ValueError, match='Kernel.delay'):
            causal_exponential_ft(f, t, w)
    with pytest.raises(ValueError, match='more than one'):
        causal_exponential_ft(dirac_delta(t)*dirac_delta(t - 1), t, w)


def test_delta_term_edge_cases_are_refused_by_name():
    t, w = var('t omega')
    tau, c = SR.var('tauD', domain='positive'), SR.var('cD')
    assert (causal_exponential_ft(c*dirac_delta(t)*heaviside(t)**2, t, w) - c).is_zero()
    for f, msg in ((dirac_delta(t)*log(t), 'singular'), (dirac_delta(t)/t, 'singular'),
                   (dirac_delta(t)*heaviside(-t), 'heaviside'),
                   (dirac_delta(t)*heaviside(t - 1), 'heaviside'),
                   (dirac_delta(2*t), 'only dirac_delta'), (dirac_delta(-t), 'only dirac_delta'),
                   (dirac_delta(t)**2, 'plain factor'), (dirac_delta(t).diff(t), 'plain factor')):
        with pytest.raises(ValueError, match=msg):
            causal_exponential_ft(f, t, w)


def test_no_change_without_delta_terms_and_no_assumption_leaks():
    t, w = var('t omega')
    tau, taug = SR.var('tauD', domain='positive'), SR.var('taugD', domain='positive')
    before = sorted(map(str, assumptions()))
    img = causal_exponential_ft((exp(-t/tau) - exp(-t/taug))/(tau - taug)*heaviside(t), t, w)
    assert str(img) == str(1/((1 + I*w*tau)*(1 + I*w*taug)))
    causal_exponential_ft(0.3*dirac_delta(t) + exp(-t/tau)/tau*heaviside(t), t, w)
    assert sorted(map(str, assumptions())) == before
    with pytest.raises(ValueError, match='heaviside'):
        causal_exponential_ft(exp(-t/tau), t, w)


# ── through the front ends ────────────────────────────────────────────────────────────────────

def _run(model, ext, ell=1):
    return AC.compute_cumulants(model, k=2, max_ell=ell, external_fields=ext, tau_grid=TAU,
                                use_cache=False, parallel=False, verbose=False)


def _same_run(a, b):
    assert a['mf_values'] == b['mf_values']
    assert set(a['C_tau_by_ell']) == set(b['C_tau_by_ell'])
    for ell in a['C_tau_by_ell']:
        assert np.array_equal(np.asarray(a['C_tau_by_ell'][ell]), np.asarray(b['C_tau_by_ell'][ell]))


def _sde(**kw):
    return (SDE('dk').parameter('tau', default=0.8, domain='positive').parameter('c', default=-0.5)
            .parameter('eps', default=0.06).parameter('D', default=0.1).field('x')
            .define_kernel('g', **kw).drift('x', '-x + c*Conv(g, x) - eps*x^3')
            .noise('xi', Gaussian(var='2*D'), couples={'x': 1}).build())


@pytest.mark.parametrize('te', [TE, '0.3*dirac_delta(t) + exp(-t/tau)/tau*heaviside(t)'])
def test_sde_delta_kernel_bit_identical_to_freq_image(te):
    ext = [('x', 1), ('x', 1)]
    _run(_sde(freq_image=FI), ext)                                  # warm-up, discarded
    rf = _run(_sde(freq_image=FI), ext)
    rt = _run(_sde(time_expr=te), ext)
    _same_run(rt, rf)
    assert np.any(np.abs(np.asarray(rt['C_tau_by_ell'][1])) > 0)    # a real loop is compared


def test_sde_delayed_delta_in_time_expr_is_refused():
    with pytest.raises(Exception, match='Kernel.delay'):           # wrapped in SDEError
        _run(_sde(time_expr='(dirac_delta(t - 1) + exp(-t/tau)/tau)*heaviside(t)'),
             [('x', 1), ('x', 1)], ell=0)


def _builder(**kw):
    return (TemporalTheoryBuilder('delta kernel builder')
            .population('E', size=1)
            .physical_field('n', population='E').physical_field('v', population='E')
            .parameter('Em', default=[-1.0], indexed_by=['E'])
            .parameter('tau', default=[2.0], domain='positive', indexed_by=['E'])
            .parameter('a', default=[1.0], indexed_by=['E'], domain='positive')
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
    ft.expand()
    out = {str(k): v for k, v in model['kernel_ft_image'](ft._ns, SR.var('omega')).items()}
    assert out
    return out


_B_TE = '(0.3*dirac_delta(t) + exp(-t/tau[i])/tau[i])*heaviside(t)'
_B_FI = '0.3 + 1/(1 + I*omega*tau[i])'


def test_builder_delta_kernel_image_and_cumulants_equal_freq_image():
    t_img, f_img = _images(_builder(time_expr=_B_TE)), _images(_builder(freq_image=_B_FI))
    assert t_img.keys() == f_img.keys()
    for k in t_img:
        assert (t_img[k] - f_img[k]).is_trivial_zero(), (k, t_img[k], f_img[k])
    ext = [('dn', 1), ('dn', 1)]
    _run(_builder(freq_image=_B_FI), ext)                           # warm-up, discarded
    held = sorted(map(str, assumptions()))
    rt = _run(_builder(time_expr=_B_TE), ext)
    assert sorted(map(str, assumptions())) == held
    _same_run(_run(_builder(freq_image=_B_FI), ext), rt)
