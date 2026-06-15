"""Fast logic tests for notebooks/nb_support.py (the shared demo
scaffolding).  Exercises load/introspection/Config/fundamental layering
and that the adaptable plotters run on synthetic result dicts — WITHOUT
calling compute_cumulants (the e2e runs are validated in the notebooks
themselves)."""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', 'notebooks')))

import nb_support as nb  # noqa: E402


def test_repo_root_and_theories():
    assert os.path.isdir(os.path.join(nb.REPO_ROOT, 'pipeline'))
    names = nb.list_theories()
    assert 'kpz_1d' in names and 'ou_quartic_double_well' in names


def test_load_theory_and_introspection():
    m, mod = nb.load_theory('kpz_1d')
    assert nb.is_spatial(m) and nb.spatial_dim(m) == 1
    assert not nb.is_multifield(m)
    assert m.get('operator_ir') is True            # KPZ uses operator-IR
    m2, _ = nb.load_theory('multipopulation_test')
    assert nb.is_multifield(m2)                     # 2 pops of size 2
    m3, _ = nb.load_theory('ou_quartic_double_well')
    assert not nb.is_spatial(m3) and not nb.is_multifield(m3)


def test_fundamental_layering():
    m, mod = nb.load_theory('reaction_diffusion_quadratic_1d')
    base = nb.fundamental_from_model(m)
    # param defaults, saddle (*star) skipped
    assert base['mu'] == 1.0 and base['g'] == 0.3 and base['T'] == 1.0
    assert not any(k.endswith('star') for k in base)


def test_config_grid_resolution():
    cfg = nb.Config(spatial_grid=(-6, 6, 13))
    g = cfg.resolved_grid()
    assert g.shape == (13,) and g[0] == -6.0 and g[-1] == 6.0
    assert nb.Config().resolved_grid() is None


def test_cumulative_curves():
    by_ell = {0: np.array([1.0, 2.0]), 1: np.array([0.1, 0.2]),
              2: np.array([0.01, 0.02])}
    cum = nb.cumulative_curves(by_ell)
    assert np.allclose(cum[0], [1.0, 2.0])
    assert np.allclose(cum[1], [1.1, 2.2])
    assert np.allclose(cum[2], [1.11, 2.22])


def test_plotters_run_on_synthetic():
    m, _ = nb.load_theory('ou_quartic_double_well')
    tau = np.linspace(-2, 2, 21)
    res_t = {'tau_grid': tau,
             'C_tau': np.exp(-np.abs(tau)),
             'C_tau_by_ell': {0: np.exp(-np.abs(tau)),
                              1: 0.1 * np.exp(-2 * np.abs(tau))},
             '_resolved': {'k': 2, 'max_ell': 1}}
    for so in ('cumulative', 'incremental', 'total'):
        fig = nb.plot_temporal(res_t, nb.Config(show_orders=so), m,
                               sim={'tau': tau, 'C': np.exp(-np.abs(tau)),
                                    'C_err': 0.01 * np.ones_like(tau)})
        assert fig is not None

    ms, _ = nb.load_theory('reaction_diffusion_quadratic_1d')
    xs = np.linspace(-5, 5, 31)
    res_s = {'spatial_grid': xs, 'tau_grid': np.array([0.0]),
             'C_tau_x': np.exp(-xs ** 2)[None, :],
             'spatial_info': {}, '_model': ms,
             '_resolved': {'k': 2, 'max_ell': 0}}
    assert nb.plot_spatial(res_s, nb.Config(), ms) is not None

    res_k = {'C_kpoint': np.array([-0.05, -0.03]),
             'C_kpoint_by_ell': {0: np.array([-0.04, -0.025]),
                                 1: np.array([-0.01, -0.005])},
             'k': 3, '_model': ms, '_resolved': {'k': 3}}
    assert nb.plot_kpoint(res_k, nb.Config(), ms) is not None
