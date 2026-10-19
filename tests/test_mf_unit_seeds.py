"""Mean-field root search in other units (final check, item b): the seed boxes
come from each variable's own equations (``_equation_scales``), not from the
largest parameter value, and the Newton refinement solves an equilibrated
system.  Exact twins of tracked models pick the same root to round-off.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

pytest.importorskip('sage.all')

import daedalus as dd                                                 # noqa: E402
from api import compute_cumulants                                     # noqa: E402


def _run(model, params, ext, k=2):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return compute_cumulants(model, k=k, max_ell=0, external_fields=ext, parameters=params,
                                 tau_grid=np.array([0.0, 1.0]), parallel=False, verbose=False,
                                 use_cache=False)


_SR = {'Em': [3.5, 3.5], 'tau': [10.0, 9.0], 'a': [2.5, 2.5], 'w': [[0.55, 0.65], [0.7, 0.8]]}


@pytest.mark.parametrize('s', (1e-3, 1e3, 1e6))
def test_spike_reset_voltage_units_keep_the_physical_root(s):
    """Tracked ``single_population_spike_reset_test``, the voltage in units s
    (Em, w times s; the gain over s): the physical root n* = (4.209, 4.545)
    in every unit.  At s = 1e3 the default box (scaled with the largest
    parameter, 3500) led the solver to the negative-rate root n1* = -3.345
    (G4_NEGATIVE_VARIANCE)."""
    model, _ = dd.load_model('single_population_spike_reset_test')
    ext = [('n', 1), ('n', 2)]
    base = _run(model, _SR, ext)
    p = dict(_SR, Em=[x * s for x in _SR['Em']], w=[[x * s for x in r] for r in _SR['w']],
             a=[x / s for x in _SR['a']])
    twin = _run(model, p, ext)
    assert twin['status'] == base['status'] == 'ok'
    assert len(twin['mf_all_roots']) == len(base['mf_all_roots'])
    assert np.asarray(twin['mf_values']['nstar']) == pytest.approx(
        np.asarray(base['mf_values']['nstar']), rel=1e-14)
    assert np.asarray(twin['mf_values']['vstar']) == pytest.approx(
        s * np.asarray(base['mf_values']['vstar']), rel=1e-14)
    assert np.asarray(twin['C_tau_by_ell'][0]) == pytest.approx(
        np.asarray(base['C_tau_by_ell'][0]), rel=1e-13)


_DQ = {'aS': [0.3, 0.3], 'ES': [0.5, 0.5], 'ED': [0.0, 0.0], 'tauS': [1.0, 1.0],
       'tauD': [1.0, 1.0], 'wSD': [[0.1, 0.03], [0.03, 0.1]], 'wDS': [[0.1, 0.03], [0.03, 0.1]]}


@pytest.mark.parametrize('a', (1e-9, 1e3, 1e9))
def test_dendritic_time_units_keep_the_stable_root_to_round_off(a):
    """Tracked ``dendritic_quad_soma_sigmoid``, time in units 1/a (spike
    rates times a): the same stable symmetric root, to round-off.  At a = 1e3
    the solver returned an unstable root (G2_UNSTABLE_SADDLE); at a = 1e+-9
    the root was right to ~1e-10 only (the two symmetric populations
    differed by 1e-12 .. 2e-9: the Newton correction was solved on an
    unequilibrated Jacobian)."""
    model, _ = dd.load_model('dendritic_quad_soma_sigmoid')
    ext = [('dnS', 1), ('dnS', 2)]
    base = _run(model, _DQ, ext)
    p = dict(_DQ, tauS=[x / a for x in _DQ['tauS']], tauD=[x / a for x in _DQ['tauD']],
             aS=[x * a for x in _DQ['aS']], wSD=[[x / a for x in r] for r in _DQ['wSD']],
             wDS=[[x / a for x in r] for r in _DQ['wDS']])
    twin = _run(model, p, ext)
    assert twin['status'] == base['status'] == 'ok'
    assert len(twin['mf_all_roots']) == len(base['mf_all_roots'])
    for f, fac in (('nSstar', a), ('nDstar', a), ('vSstar', 1.0), ('vDstar', 1.0)):
        assert np.asarray(twin['mf_values'][f]) == pytest.approx(
            fac * np.asarray(base['mf_values'][f]), rel=2e-15), f
    nS = np.asarray(twin['mf_values']['nSstar'])
    assert abs(nS[0] - nS[1]) <= 4e-16 * abs(nS[0])


def test_two_population_kernel_model_selects_the_stable_root():
    """The two-population exponential-rate model of the kernel front-end test
    (``tests/test_frontend_function_kernel_ft.py``, saddles declared with
    .set_mf_equation, so no linearization classifies its roots): the stable
    fixed point is selected, not the unstable one nearer to 0 (G2_UNSTABLE_
    SADDLE, a propagator pole at -0.30i, since the one-solver change).  The
    roots are ordered by the relaxation of their saddle equations."""
    from tests.test_frontend_function_kernel_ft import _kernel_model
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        r = compute_cumulants(_kernel_model('freq', 2), k=2, max_ell=0,
                              external_fields=[('dn', 1), ('dn', 2)],
                              tau_grid=np.array([0.0, 1.0]), parallel=False,
                              verbose=False, use_cache=False)
    assert r['status'] == 'ok'
    assert len(r['mf_all_roots']) >= 2
    codes = {f.code for f in r['analysis'].findings}
    assert 'G2_UNSTABLE_SADDLE' not in codes
