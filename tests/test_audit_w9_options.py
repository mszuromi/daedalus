"""Audit wave 9, option surface (CON-02, decision B4): compute_cumulants,
dd.Config and the YAML config read the same options, and read a 3-entry grid
the same way."""
from __future__ import annotations

import io
import contextlib
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import daedalus as dd                                   # noqa: E402
from api import compute_cumulants                       # noqa: E402
from api import config_run                              # noqa: E402
from api.config_run import ConfigError, load_config     # noqa: E402

_KW = dict(k=2, max_ell=0, external_fields=[('dx', 1), ('dx', 1)])


def _ou():
    return dd.load_model('ou_quartic')


def test_three_entry_tuple_is_a_range_everywhere():
    """B4: (lo, hi, n) is a range in compute_cumulants too (it used to be
    refused there while dd.Config read it as a range)."""
    model, mod = _ou()
    r1 = compute_cumulants(model, **_KW, tau_grid=(-2.0, 2.0, 9),
                           verbose=False, parallel=False)
    assert np.allclose(r1['tau_grid'], np.linspace(-2.0, 2.0, 9))
    r2 = dd.run(model, dd.Config(**_KW, tau_grid=(-2.0, 2.0, 9)), mod)
    np.testing.assert_array_equal(np.asarray(r1['C_tau']), np.asarray(r2['C_tau']))


@pytest.mark.parametrize('grid', [[-5, 5, 51], [0.0, 1.0, 4.0]])
def test_three_entry_list_slip_is_refused_by_every_entry_point(grid, tmp_path):
    model, _mod = _ou()
    with pytest.raises(ValueError, match='3-element LIST'):
        compute_cumulants(model, **_KW, tau_grid=grid, verbose=False,
                          parallel=False)
    with pytest.raises(ValueError, match='3-element LIST'):
        dd.Config(**_KW, tau_grid=grid)
    p = tmp_path / 'c.yaml'
    p.write_text(f"model: ou_quartic\nquestion:\n  tau_grid: {grid}\n")
    with pytest.raises(ConfigError, match='3-element LIST'):
        load_config(str(p))


def test_literal_array_of_three_points_is_fine_everywhere():
    model, mod = _ou()
    g = np.array([0.0, 1.0, 4.0])
    r1 = compute_cumulants(model, **_KW, tau_grid=g, verbose=False, parallel=False)
    r2 = dd.run(model, dd.Config(**_KW, tau_grid=g), mod)
    np.testing.assert_array_equal(r1['tau_grid'], g)
    np.testing.assert_array_equal(np.asarray(r1['C_tau']), np.asarray(r2['C_tau']))


def test_config_carries_the_numeric_options_to_compute_cumulants(monkeypatch):
    import api
    seen = {}
    real = api.compute_cumulants

    def spy(*a, **kw):
        seen.update(kw)
        return real(*a, **kw)
    monkeypatch.setattr(api, 'compute_cumulants', spy)
    model, mod = _ou()
    dd.run(model, dd.Config(**_KW, tau_grid=(0.0, 1.0, 3), taylor_order=4,
                            use_cache=False, accuracy=1e-8, discard='auto'), mod)
    assert seen['taylor_order'] == 4 and seen['use_cache'] is False
    assert seen['accuracy'] == 1e-8 and seen['discard'] == 'auto'
    seen.clear()
    dd.run(model, dd.Config(**_KW, tau_grid=(0.0, 1.0, 3)), mod)
    for opt in ('taylor_order', 'use_cache', 'accuracy', 'discard', 'backend'):
        assert opt not in seen            # unset options keep compute_cumulants' defaults


@pytest.mark.parametrize('bad, match', [
    (dict(taylor_order=0), 'taylor_order'),
    (dict(use_cache='yes'), 'use_cache'),
    (dict(accuracy=-1.0), 'accuracy'),
    (dict(accuracy=float('nan')), 'accuracy'),
    (dict(discard='maybe'), 'discard'),
])
def test_config_validates_the_new_options(bad, match):
    with pytest.raises(ValueError, match=match):
        dd.Config(**_KW, **bad)


def test_config_options_lists_delay_and_the_aliases():
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        dd.config_options()
    text = buf.getvalue()
    for word in ('delay', 'taylor_order', 'use_cache', 'accuracy', 'backend',
                 'discard', 'alias: fundamental', 'alias: spatial_grid'):
        assert word in text, word


def test_yaml_carries_accuracy_backend_and_discard(tmp_path, monkeypatch):
    p = tmp_path / 'c.yaml'
    p.write_text("model: ou_quartic\nquestion:\n  k: 2\n  max_ell: 0\n"
                 "  tau_grid: {start: 0, stop: 2, num: 3}\n  taylor_order: 4\n"
                 "  accuracy: 1e-8\npolicy:\n  use_cache: false\n  discard: auto\n")
    resolved = load_config(str(p))[0]
    assert resolved['question']['accuracy'] == 1e-8
    assert resolved['policy']['discard'] == 'auto'
    import api
    seen = {}
    real = api.compute_cumulants

    def spy(*a, **kw):
        seen.update(kw)
        return real(*a, **kw)
    monkeypatch.setattr(api, 'compute_cumulants', spy)
    with contextlib.redirect_stdout(io.StringIO()):
        config_run.run_config(str(p))
    assert seen['taylor_order'] == 4 and seen['accuracy'] == 1e-8
    assert seen['use_cache'] is False and seen['discard'] == 'auto'


def test_yaml_rejects_bad_accuracy_with_its_line(tmp_path):
    p = tmp_path / 'c.yaml'
    p.write_text("model: ou_quartic\nquestion:\n  accuracy: -1\n")
    with pytest.raises(ConfigError, match='line 3.*accuracy'):
        load_config(str(p))


# ---------------------------------------------------------------- CON-07 ----

def _sde():
    from api.noise import Gaussian
    from api.sde import SDE
    return (SDE('w9 ou')
            .parameter('mu', default=1.0, domain='positive')
            .parameter('D', default=0.5, domain='positive')
            .field('x')
            .drift('x', '-mu*x')
            .noise('xi', Gaussian(var='2*D'), couples={'x': 1}))


def test_every_entry_point_accepts_an_unbuilt_sde():
    """compute_cumulants, dd.describe_model and dd.symbolic_cumulant took only
    the built dict (AttributeError: 'SDE' object has no attribute 'get');
    dd.run built it itself."""
    taus = np.array([0.0, 0.5, 1.0])
    r_sde = compute_cumulants(_sde(), k=2, max_ell=0, tau_grid=taus,
                              external_fields=[('x', 1), ('x', 1)],
                              verbose=False, parallel=False)
    r_dict = compute_cumulants(_sde().build(), k=2, max_ell=0, tau_grid=taus,
                               external_fields=[('x', 1), ('x', 1)],
                               verbose=False, parallel=False)
    np.testing.assert_array_equal(r_sde['C_tau'], r_dict['C_tau'])
    # closed form: C(tau) = D/mu exp(-mu |tau|)
    np.testing.assert_allclose(np.real(r_sde['C_tau']),
                               0.5 * np.exp(-taus), rtol=1e-10)
    with contextlib.redirect_stdout(io.StringIO()):
        assert 'w9 ou' in dd.describe_model(_sde())
    from api.symbolic_loop import symbolic_cumulant
    sc = symbolic_cumulant(_sde(), k=2, max_ell=0,
                           external_fields=[('x', 1), ('x', 1)])
    assert sc is not None


def test_a_non_model_is_a_clear_type_error():
    with pytest.raises(TypeError, match='builder with a build'):
        compute_cumulants(42, k=2, max_ell=0, tau_grid=[0.0],
                          external_fields=[('x', 1)] * 2, verbose=False)


def test_spatial_k2_result_carries_status_and_ell_increments():
    """C_tau_x_by_ell are the per-order INCREMENTS (they sum to C_tau_x, like the
    temporal C_tau_by_ell and total_C_by_ell); C_tau_x_by_order stays cumulative."""
    model, mod = dd.load_model('allen_cahn_1d_subcritical_infinite')
    r = dd.run(model, dd.Config(k=2, max_ell=1, chi_grid=[0.0, 1.0],
                                tau_max=0.0, tau_step=1.0), mod)
    assert r['status'] == 'ok'
    by_ell, by_order = r['C_tau_x_by_ell'], r['C_tau_x_by_order']
    assert sorted(by_ell) == sorted(by_order) == [0, 1]
    total = np.asarray(r['C_tau_x'])
    np.testing.assert_allclose(sum(np.asarray(v) for v in by_ell.values()),
                               total, rtol=1e-12, atol=1e-15)
    np.testing.assert_array_equal(by_ell[0], by_order[0])
    np.testing.assert_allclose(by_ell[1], np.asarray(by_order[1]) - np.asarray(by_order[0]),
                               rtol=1e-12, atol=1e-15)
    np.testing.assert_allclose(np.asarray(by_order[1]), total, rtol=1e-12, atol=1e-15)
    # the callables of the same increments
    for e in by_ell:
        np.testing.assert_allclose(complex(r['total_C_by_ell'][e](0.0, 0.0)),
                                   complex(np.asarray(by_ell[e])[0, 0]), rtol=1e-12, atol=1e-15)


# ---------------------------------------------------------------- CON-09 ----

def _delayed_ou_pade():
    from api.delay import pade_model
    from api.noise import Gaussian
    from api.sde import SDE
    m = (SDE('w9 delayed ou').parameter('a', default=1.0)
         .parameter('c', default=-2.0)
         .parameter('d', default=0.5, domain='positive')
         .parameter('eps', default=0.02).parameter('D', default=0.1)
         .field('x').drift('x', '-a*x + c*x(t - d) - eps*x^3')
         .noise('xi', Gaussian(var='2*D'), couples={'x': 1}).build())
    return pade_model(m, 8, False)


def test_balanced_truncation_enters_the_g6_estimate_and_the_reference_guard():
    """CON-09: a truncated propagator entry's H-infinity bound is part of the
    accuracy estimate (it was recorded and read by nobody), and a supplied
    truncated result is refused as a symbolic reference.  Reference: the
    untruncated run (tolerance 0)."""
    import warnings
    from api import _realization as RZ
    from api.symbolic_loop import SymbolicLoopRefused, _refuse_truncated_reference
    import engine.integration.time_domain.statespace as SS
    taus = np.array([0.0, 0.5, 1.0, 2.0, 4.0])
    runs = {}
    for tol in (0.0, 1e-3):
        SS._clear_caches()
        with RZ.truncation_scope(tol), warnings.catch_warnings():
            warnings.simplefilter('ignore')
            r = compute_cumulants(_delayed_ou_pade(), k=2, max_ell=1,
                                  external_fields=[('x', 1)] * 2,
                                  tau_grid=taus, parallel=False, verbose=False,
                                  taylor_order=4, use_cache=False)
            runs[tol] = r
            if tol > 0:
                with pytest.raises(SymbolicLoopRefused, match='truncated'):
                    _refuse_truncated_reference(r)
    exact, cut = runs[0.0], runs[1e-3]
    shift = (np.max(np.abs(cut['C_tau'] - exact['C_tau']))
             / np.max(np.abs(exact['C_tau'])))
    d_exact = exact['analysis'].diagnostics
    d_cut = cut['analysis'].diagnostics
    assert d_exact['accuracy_estimate'] < 1e-12
    assert d_cut['truncated_entries'] >= 1
    # the estimate covers the measured effect of the truncation
    assert d_cut['accuracy_estimate'] >= shift > 1e-5
    assert any(f.code == 'G6_ACCURACY_LOW' for f in cut['analysis'].findings)
    rec = cut['phase_j_engine']['realization_check']['truncation']
    assert rec['n_truncated'] >= 1 and 0 < rec['max_bound_rel'] < 1e-3
