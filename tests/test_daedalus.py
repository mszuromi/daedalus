"""Fast logic tests for notebooks/daedalus.py (the shared demo
scaffolding).  Exercises load/introspection/Config/fundamental layering
and that the adaptable plotters run on synthetic result dicts — WITHOUT
calling compute_cumulants (the e2e runs are validated in the notebooks
themselves)."""
import os
import sys

import numpy as np
import pytest
import matplotlib
matplotlib.use('Agg')

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', 'notebooks')))

import daedalus as dd  # noqa: E402


def test_repo_root_and_theories():
    assert os.path.isdir(os.path.join(dd.REPO_ROOT, 'pipeline'))
    names = dd.list_theories()
    assert 'kpz_1d' in names and 'ou_quartic_double_well' in names


def test_load_theory_and_introspection():
    m, mod = dd.load_theory('kpz_1d')
    assert dd.is_spatial(m) and dd.spatial_dim(m) == 1
    assert not dd.is_multifield(m)
    assert m.get('operator_ir') is True            # KPZ uses operator-IR
    m2, _ = dd.load_theory('multipopulation_test')
    assert dd.is_multifield(m2)                     # 2 pops of size 2
    m3, _ = dd.load_theory('ou_quartic_double_well')
    assert not dd.is_spatial(m3) and not dd.is_multifield(m3)


def test_fundamental_layering():
    m, mod = dd.load_theory('reaction_diffusion_quadratic_1d')
    base = dd.fundamental_from_model(m)
    # param defaults, saddle (*star) skipped
    assert base['mu'] == 1.0 and base['g'] == 0.3 and base['T'] == 1.0
    assert not any(k.endswith('star') for k in base)


def test_k_external_fields_mismatch_raises():
    """An explicit k that disagrees with the explicit external_fields length is
    a contradiction — it must raise, not silently rebuild the legs.  (Raises
    early in run(), before compute_cumulants.)"""
    m, mod = dd.load_theory('ou_quartic_double_well')
    with pytest.raises(ValueError, match='k=4 but external_fields'):
        dd.run(m, dd.Config(k=4, external_fields=[('dx', 1), ('dx', 1)],
                            tau_max=2.0, tau_step=2.0), mod)


def test_k_inferred_from_external_fields():
    """Omit k and it is counted from external_fields (a k-point correlator has
    exactly k legs)."""
    m, mod = dd.load_theory('ou_quartic_double_well')
    r2 = dd.run(m, dd.Config(external_fields=[('dx', 1), ('dx', 1)],
                             max_ell=0, tau_max=2.0, tau_step=2.0), mod)
    assert r2['_resolved']['k'] == 2
    r3 = dd.run(m, dd.Config(external_fields=[('dx', 1)] * 3,
                             max_ell=0, tau_max=2.0, tau_step=2.0), mod)
    assert r3['_resolved']['k'] == 3


def test_kpoint_slices_synthesized():
    """k≥3 temporal: run() synthesizes the k−1 independent-difference slices
    (τ_j = t_j − t_0), and plot_cumulant draws one panel per slice."""
    import matplotlib.pyplot as plt
    m, mod = dd.load_theory('ou_quartic_double_well')
    r = dd.run(m, dd.Config(k=3, max_ell=0, tau_max=2.0, tau_step=1.0), mod)
    assert set(r['C_tau_slices']) == {1, 2}              # k-1 = 2 slices
    assert np.array_equal(r['C_tau'], r['C_tau_slices'][1])
    fig = dd.plot_cumulant(r, r['_cfg'], m)
    assert len(fig.axes) == 2                            # one panel per slice
    plt.close(fig)


def test_kpoint_base_lags_and_full_grid():
    """k≥3: kpoint_base_lags fixes the non-swept legs (validated against the
    full grid), and kpoint_full_grid returns the (k−1)-dim tensor."""
    import matplotlib.pyplot as plt
    m, mod = dd.load_theory('ou_quartic_double_well')
    T = dict(tau_max=2.0, tau_step=1.0)
    # wrong-length base → clear error
    with pytest.raises(ValueError, match='k.1 = 2 entries'):
        dd.run(m, dd.Config(k=3, max_ell=0, kpoint_base_lags=[1.0], **T), mod)
    # full 2-D grid for k=3 + heatmap dispatch
    rg = dd.run(m, dd.Config(k=3, max_ell=0, kpoint_full_grid=True, **T), mod)
    ax = np.asarray(rg['tau_axes'][0])
    assert np.asarray(rg['C_tau_grid']).shape == (ax.size, ax.size)
    fig = dd.plot_cumulant(rg, rg['_cfg'], m)
    assert len(fig.axes) == 2                       # pcolormesh + colorbar
    plt.close(fig)
    # base-shifted slice 2 (leg 1 fixed at 1.0) == the matching grid row
    ia = int(np.argmin(np.abs(ax - 1.0)))
    rb = dd.run(m, dd.Config(k=3, max_ell=0, kpoint_base_lags=[1.0, 0.0], **T), mod)
    assert np.allclose(np.real(rb['C_tau_slices'][2]),
                       np.real(rg['C_tau_grid'])[ia, :], atol=1e-9)


def test_config_grid_resolution():
    cfg = dd.Config(spatial_grid=(-6, 6, 13))
    g = cfg.resolved_grid()
    assert g.shape == (13,) and g[0] == -6.0 and g[-1] == 6.0
    assert dd.Config().resolved_grid() is None


def test_cumulative_curves():
    by_ell = {0: np.array([1.0, 2.0]), 1: np.array([0.1, 0.2]),
              2: np.array([0.01, 0.02])}
    cum = dd.cumulative_curves(by_ell)
    assert np.allclose(cum[0], [1.0, 2.0])
    assert np.allclose(cum[1], [1.1, 2.2])
    assert np.allclose(cum[2], [1.11, 2.22])


def test_plotters_run_on_synthetic():
    m, _ = dd.load_theory('ou_quartic_double_well')
    tau = np.linspace(-2, 2, 21)
    res_t = {'tau_grid': tau,
             'C_tau': np.exp(-np.abs(tau)),
             'C_tau_by_ell': {0: np.exp(-np.abs(tau)),
                              1: 0.1 * np.exp(-2 * np.abs(tau))},
             '_resolved': {'k': 2, 'max_ell': 1}}
    for so in ('cumulative', 'incremental', 'total'):
        fig = dd.plot_temporal(res_t, dd.Config(show_orders=so), m,
                               sim={'tau': tau, 'C': np.exp(-np.abs(tau)),
                                    'C_err': 0.01 * np.ones_like(tau)})
        assert fig is not None

    ms, _ = dd.load_theory('reaction_diffusion_quadratic_1d')
    xs = np.linspace(-5, 5, 31)
    res_s = {'spatial_grid': xs, 'tau_grid': np.array([0.0]),
             'C_tau_x': np.exp(-xs ** 2)[None, :],
             'spatial_info': {}, '_model': ms,
             '_resolved': {'k': 2, 'max_ell': 0}}
    assert dd.plot_spatial(res_s, dd.Config(), ms) is not None

    res_k = {'C_kpoint': np.array([-0.05, -0.03]),
             'C_kpoint_by_ell': {0: np.array([-0.04, -0.025]),
                                 1: np.array([-0.01, -0.005])},
             'k': 3, '_model': ms, '_resolved': {'k': 3}}
    assert dd.plot_kpoint(res_k, dd.Config(), ms) is not None
