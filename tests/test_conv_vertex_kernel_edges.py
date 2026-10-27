"""
Conductance-style vertices (a field multiplying ``Conv(g, y)``): the kernel is
an edge of the diagram, and the result does not depend on the cache state.

A vertex ``x·(g ∗ y)`` puts its ``y`` leg at its own time ``s`` and joins it to
the vertex time by the kernel ``g(t − s)``.  Three defects made such models
wrong or unusable:

* every edge at the vertex was shifted by the kernel delay (the leg was looked
  up by its population index only, at both ends of an edge), which turned the
  kernel into an instantaneous coupling;
* the delay was the integration variable, so every edge into the attached leg
  had a three-term constraint row; the exact box-free routes declined and the
  subsets went to nested quadrature (days per tau point);
* a warm expand cache did not restore the kernel attachments, so the vertex was
  built as a plain vertex and its diagrams failed ("unexpected free symbols")
  and were dropped; a cold run could not store the typed diagrams either.

The reference is the same model with the kernel written as an auxiliary linear
field (``z' = (y − z)/taug``, the Markov embedding of ``g``): its vertex
``x·z`` is local, so it runs through none of the code above.  Both forms must
agree to rounding at tree level and one loop.
"""
from __future__ import annotations

import os
import sys
import warnings

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from daedalus.api import compute_cumulants                                   # noqa: E402
from daedalus.api.model import TemporalModelBuilder                          # noqa: E402
from daedalus.engine.core.vertices import ConvVertexType                     # noqa: E402
import daedalus.engine.integration.time_domain.final_integral as FI          # noqa: E402
import daedalus.engine.integration.time_domain.dbm_integral as DBM           # noqa: E402

TAU = np.array([0.0, 0.5, 2.0, 5.0])
PARAMS = (('mu', 1.0), ('nu', 0.7), ('eps', 0.3), ('h', 0.5), ('hx', 0.4),
          ('D', 0.5), ('taug', 2.0))


def _base(name, fields):
    b = TemporalModelBuilder(name)
    b.population('P', size=1)
    for f in fields:
        b.physical_field(f, population='P')
    for p, d in PARAMS:
        b.parameter(p, default=d, domain='positive')
    b.set_mf_equation('ystar', 'h/nu')
    b.set_mf_equation('xstar', 'hx/(mu - eps*h/nu)')
    return b


def conv_model():
    """x' = -mu x + hx + eps x (g ∗ y) + noise,  y' = -nu y + h + noise."""
    b = _base('Delayed multiplicative coupling (kernel)', ('x', 'y'))
    b.define_kernel('g', time_expr='(1/taug)*exp(-t/taug)*heaviside(t)',
                    latex_name='g')
    b.set_action_text('sum(xt[i]*((Dt+mu)*x[i] - hx - eps*x[i]*Conv(g,y[i])) '
                      '+ yt[i]*((Dt+nu)*y[i] - h) - D*xt[i]^2 - D*yt[i]^2 '
                      'for i in P)')
    return b.build()


def embedded_model():
    """The same model with z = g ∗ y as a field: taug z' = y - z."""
    b = _base('Delayed multiplicative coupling (embedded)', ('x', 'y', 'z'))
    b.set_mf_equation('zstar', 'h/nu')
    b.set_action_text('sum(xt[i]*((Dt+mu)*x[i] - hx - eps*x[i]*z[i]) '
                      '+ yt[i]*((Dt+nu)*y[i] - h) '
                      '+ zt[i]*((taug*Dt+1)*z[i] - y[i]) '
                      '- D*xt[i]^2 - D*yt[i]^2 for i in P)')
    return b.build()


def _run(model, ext, **kw):
    kw.setdefault('use_cache', False)
    return compute_cumulants(model, k=2, max_ell=1, external_fields=ext,
                             tau_grid=TAU, parallel=False, verbose=False, **kw)


@pytest.fixture(scope='module')
def embedded():
    return {tuple(ext): _run(embedded_model(), ext)
            for ext in ([('x', 1), ('x', 1)], [('x', 1), ('y', 1)])}


@pytest.mark.parametrize('ext', [[('x', 1), ('x', 1)], [('x', 1), ('y', 1)]])
def test_kernel_vertex_equals_its_markov_embedding(ext, embedded):
    FI._reset_runtime_counters()
    res = _run(conv_model(), ext)
    ref = embedded[tuple(ext)]
    assert res['status'] == 'ok' and ref['status'] == 'ok'
    n_conv = sum(isinstance(vt, ConvVertexType)
                 for d in res['diagrams']
                 for vt in d['typed_diagram'].vertex_assignments.values())
    assert n_conv > 0
    for ell in (0, 1):
        a = np.asarray(res['C_tau_by_ell'][ell])
        b = np.asarray(ref['C_tau_by_ell'][ell])
        assert np.all(np.abs(b) > 0)
        np.testing.assert_allclose(a, b, rtol=1e-12, atol=0)
    # every subset on an exact route: no quadrature
    assert FI._RUNTIME_COUNTERS['nquad_calls'] == 0


def test_warm_cache_keeps_the_kernel_vertex(tmp_path, monkeypatch):
    """Cold run, then two warm runs from the expand and typed-diagram
    caches: the same vertex types, no failed diagram, the same numbers."""
    monkeypatch.chdir(tmp_path)
    ext = [('x', 1), ('x', 1)]
    runs = [_run(conv_model(), ext, use_cache=True) for _ in range(3)]
    for r in runs:
        assert r['status'] == 'ok'
        assert any(isinstance(vt, ConvVertexType)
                   for d in r['diagrams']
                   for vt in d['typed_diagram'].vertex_assignments.values())
    for r in runs[1:]:
        for ell in (0, 1):
            np.testing.assert_allclose(r['C_tau_by_ell'][ell],
                                       runs[0]['C_tau_by_ell'][ell],
                                       rtol=1e-13, atol=0)


def _mixed_leg_model(embedded):
    """x' = -mu x + hx - eps x (g ∗ x) + noise: the vertex has two copies
    of the leg x, one plain and one under the kernel.  ``embedded``: the
    same model with z = g ∗ x as a field (taug z' = x - z), vertex x·z."""
    b = TemporalModelBuilder('Mixed kernel leg' + (' (embedded)' if embedded
                                                   else ''))
    b.population('P', size=1)
    b.physical_field('x', population='P')
    if embedded:
        b.physical_field('z', population='P')
    for p, d in (('mu', 1.0), ('eps', 0.1), ('hx', 0.5), ('D', 0.5),
                 ('taug', 2.0)):
        b.parameter(p, default=d, domain='positive')
    xstar = '(sqrt(mu^2 + 4*eps*hx) - mu)/(2*eps)'
    b.set_mf_equation('xstar', xstar)
    if embedded:
        b.set_mf_equation('zstar', xstar)
        b.set_action_text('sum(xt[i]*((Dt+mu)*x[i] - hx + eps*x[i]*z[i]) '
                          '+ zt[i]*((taug*Dt+1)*z[i] - x[i]) - D*xt[i]^2 '
                          'for i in P)')
    else:
        b.define_kernel('g', time_expr='(1/taug)*exp(-t/taug)*heaviside(t)',
                        latex_name='g')
        b.set_action_text('sum(xt[i]*((Dt+mu)*x[i] - hx '
                          '+ eps*x[i]*Conv(g,x[i])) - D*xt[i]^2 for i in P)')
    return b.build()


def test_mixed_kernel_leg_equals_its_markov_embedding():
    """A vertex with a plain and a kernel copy of the same leg: the typed
    diagram's weight counts the copies as identical (the automorphism
    colouring does not mark kernel legs), so the integrand is the average
    over the distinct placements of the lines in the two slots.  Checked
    against the embedded model, whose legs are distinct fields."""
    ext = [('x', 1), ('x', 1)]
    res = _run(_mixed_leg_model(False), ext)
    ref = _run(_mixed_leg_model(True), ext)
    assert res['status'] == 'ok' and ref['status'] == 'ok'
    assert any(g.get('handled_by') != 'skipped'
               for g in res['phase_j_by_ell'][1]['groups'])
    for ell in (0, 1):
        np.testing.assert_allclose(res['C_tau_by_ell'][ell],
                                   ref['C_tau_by_ell'][ell],
                                   rtol=1e-12, atol=0)


def test_exp_mode_extraction_rejects_a_sum_of_exponentials():
    from sage.all import SR, exp
    t = SR.var('t')
    assert FI._extract_exp_mode(2 * exp(-t / 3), t) == pytest.approx(
        (2.0, -1.0 / 3.0), rel=1e-12)
    assert FI._extract_exp_mode(exp(-t) + exp(-2 * t), t) is None
    assert FI._extract_exp_mode(t * exp(-t), t) is None


def test_high_precision_dbm_answers_a_cancelling_closed_form():
    """∫_0^∞ (e^{a1 s} - e^{a2 s}) ds = 1/|a1| - 1/|a2| with a1 = -1e-3 and
    a2 = a1 (1 + 1e-6): the expansion's two terms cancel 6 digits.  The
    elimination at MP_DPS digits returns the exact integral of these
    (double) rates.  The double closed form's verdict is the same in every
    time unit (audit W4b, ACC-64: the floor is in the integral's units, so
    the rounding error, 1e-15 of the integral's scale |C|/|a|, passes here
    as it did at a1 = -1; before, only the slow-unit copy was declined)."""
    import mpmath as mp
    a1, a2 = -1e-3, -1e-3 * (1 + 1e-6)
    rows = [((1.0,), 0.0)]
    seeds = [(1.0, 0.0, (a1,)), (-1.0, 0.0, (a2,))]
    lo = DBM.integrate_exp_sum(rows, 1, seeds, float('inf'))
    twin = DBM.integrate_exp_sum(rows, 1, [(1.0, 0.0, (a1 * 1e3,)),
                                           (-1.0, 0.0, (a2 * 1e3,))],
                                 float('inf'))
    assert lo.status == twin.status
    assert abs(lo.error_ratio - twin.error_ratio) <= 1e-6 * twin.error_ratio
    hi = DBM.integrate_exp_sum_mp(rows, 1, seeds, float('inf'))
    assert hi.status == DBM.STATUS_OK
    with mp.workdps(50):
        exact = float(1 / -mp.mpf(a1) - 1 / -mp.mpf(a2))
    assert abs(hi.value - exact) <= 1e-15 * exact
    # the double path is untouched by the high-precision call
    again = DBM.integrate_exp_sum(rows, 1, seeds, float('inf'))
    assert again.status == lo.status and again.value == lo.value


def _chain_kernel_model(tau2, embedded, name='Two-stage kernel'):
    """x' = h - a x - b x (g * x) + noise with ĝ = 1/((1+iωτ)(1+iωτ2)) (a
    two-stage filter; τ2 = τ is the alpha kernel t e^{-t/τ}/τ²).
    ``embedded``: the same model with the filter as a chain of fields
    (τ u' = x - u, τ2 s' = u - s) and the local vertex b x s."""
    b = TemporalModelBuilder(name)
    b.population('P', size=1)
    b.physical_field('x', population='P')
    if embedded:
        b.physical_field('u', population='P')
        b.physical_field('s', population='P')
    for p, d in (('a', 1.0), ('bb', 0.3), ('h', 0.5), ('D', 0.15),
                 ('tau', 0.7), ('tau2', tau2)):
        b.parameter(p, default=d, domain='positive')
    xstar = '(sqrt(a^2 + 4*bb*h) - a)/(2*bb)'
    b.set_mf_equation('xstar', xstar)
    if embedded:
        b.set_mf_equation('ustar', xstar)
        b.set_mf_equation('sstar', xstar)
        b.set_action_text('sum(xt[i]*((Dt+a)*x[i] - h + bb*x[i]*s[i]) '
                          '+ ut[i]*((tau*Dt+1)*u[i] - x[i]) '
                          '+ st[i]*((tau2*Dt+1)*s[i] - u[i]) '
                          '- D*xt[i]^2 for i in P)')
    else:
        b.define_kernel('g', freq_image='1/((1 + I*omega*tau)'
                                        '*(1 + I*omega*tau2))')
        b.set_action_text('sum(xt[i]*((Dt+a)*x[i] - h '
                          '+ bb*x[i]*Conv(g,x[i])) - D*xt[i]^2 for i in P)')
    return b.build()


@pytest.mark.parametrize('engine', ['poles', 'statespace'])
@pytest.mark.parametrize('tau2, ok', [(0.4, True), (0.62, True),
                                      (0.69, False), (0.7, False)])
def test_kernel_rates_separated_or_refused(tau2, ok, engine, monkeypatch):
    """Distinct kernel rates: equal to the embedding to 1e-12.  On the pole
    engine a repeated rate (τ2 = τ: the alpha kernel, a t e^{-t/τ} term no
    mode sum holds) or a near-repeated one (the ±1/gap coefficients cancel,
    ~1e-15/gap²) is refused by name -- before, a numerically split double
    root gave a one loop 7x too large with status 'ok'.  S3: the
    state-space route (the default) realizes the kernel exactly (a Jordan
    chain at equal rates), so it takes every rate ratio."""
    monkeypatch.setenv('DAEDALUS_PHASE_J_ENGINE', engine)
    ok = ok or engine == 'statespace'
    ext = [('x', 1), ('x', 1)]
    with warnings.catch_warnings(record=True):
        warnings.simplefilter('always')
        res = _run(_chain_kernel_model(tau2, False), ext)
    ref = _run(_chain_kernel_model(tau2, True), ext)
    np.testing.assert_allclose(res['C_tau_by_ell'][0], ref['C_tau_by_ell'][0],
                               rtol=1e-12, atol=0)
    if ok:
        assert res['status'] == 'ok'
        np.testing.assert_allclose(res['C_tau_by_ell'][1],
                                   ref['C_tau_by_ell'][1], rtol=1e-12, atol=0)
    else:
        assert res['status'] == 'incomplete'
        assert all('separated rates' in r['reason']
                   for recs in res['phase_j_incomplete'].values()
                   for r in recs)


def test_same_model_name_other_kernel_is_not_served_from_cache(tmp_path,
                                                              monkeypatch):
    """The propagator and typed-diagram caches are keyed by the model NAME;
    a model that differs only in its kernel image must not get the other
    model's propagator (it did before: identical numbers)."""
    monkeypatch.chdir(tmp_path)
    ext = [('x', 1), ('x', 1)]
    a = _run(_chain_kernel_model(0.4, False, name='Same name'), ext,
             use_cache=True)
    b = _run(_chain_kernel_model(0.62, False, name='Same name'), ext,
             use_cache=True)
    fresh = _run(_chain_kernel_model(0.62, False, name='Same name'), ext)
    assert not np.allclose(a['C_tau_by_ell'][0], b['C_tau_by_ell'][0])
    for ell in (0, 1):
        np.testing.assert_allclose(b['C_tau_by_ell'][ell],
                                   fresh['C_tau_by_ell'][ell],
                                   rtol=1e-13, atol=0)
