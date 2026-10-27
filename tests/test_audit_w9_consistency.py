"""Audit wave 9: consistency, loud edge cases and option surface.

Each test pins one finding of the wave (ids in the test docstrings) against an
independent reference or an inline public model.
"""
from __future__ import annotations

import os
import stat
import sys
import threading

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# --------------------------------------------------------------------------
# LOUD-03 / LOUD-04: prediagram cache writes
# --------------------------------------------------------------------------


def test_concurrent_cert_writers_do_not_clobber(tmp_path):
    """LOUD-03: two writers of one cell both succeed and leave a valid file
    and no stray temp files (the old fixed '<file>.tmp' name raced)."""
    import daedalus.engine.enumeration.prediagram_cache as P
    certs = {bytes([i, j]) for i in range(64) for j in range(64)}
    path = str(tmp_path / 'streaming_v2' / 'cell.pkl')
    errors = []

    def work():
        try:
            for _ in range(5):
                P.write_cert_file(path, 2, 1, certs)
        except BaseException as exc:                      # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=work) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
    got, stamp = P.read_cert_file(path, 2, 1)
    assert got == certs and stamp['count'] == len(certs)
    assert os.listdir(os.path.dirname(path)) == ['cell.pkl']


def test_failed_cert_write_leaves_no_temp_file(tmp_path, monkeypatch):
    import pickle
    import daedalus.engine.enumeration.prediagram_cache as P
    path = str(tmp_path / 'd' / 'cell.pkl')

    def boom(*a, **k):
        raise RuntimeError('disk full')
    monkeypatch.setattr(pickle, 'dump', boom)
    with pytest.raises(RuntimeError):
        P.write_cert_file(path, 2, 1, {b'ab'})
    assert os.listdir(os.path.dirname(path)) == []


@pytest.mark.skipif(os.geteuid() == 0, reason='root ignores directory modes')
def test_shipped_cell_loads_from_a_read_only_cache_root(tmp_path, caplog):
    """LOUD-04: a shipped-cell hit is served from memory when the local root
    cannot be written (it used to raise PermissionError)."""
    import daedalus.engine.enumeration.prediagram_cache as P
    ro = tmp_path / 'ro'
    ro.mkdir()
    os.chmod(ro, stat.S_IRUSR | stat.S_IXUSR)
    try:
        recs, src = P.load_prediagrams(str(ro / 'sub'), 2, 1)
        assert src == 'shipped' and len(recs) > 0
        ref, _ = P.load_prediagrams(str(tmp_path / 'rw'), 2, 1)
        assert len(ref) == len(recs)
    finally:
        os.chmod(ro, stat.S_IRWXU)


# --------------------------------------------------------------------------
# ACC-71 / GEN-13: TikZ labels use the real edge key and field roles
# --------------------------------------------------------------------------


def _twin_model(y):
    from daedalus.api.noise import Gaussian
    from daedalus.api.sde import SDE
    return (SDE('w9_twin_' + y)
            .parameter('mu', default=1.0, domain='positive')
            .parameter('nu', default=2.0, domain='positive')
            .parameter('c', default=0.5)
            .parameter('eps', default=0.1)
            .parameter('D', default=1.0, domain='positive')
            .field('x').field(y)
            .drift('x', f'-mu*x + c*{y} - eps*x*{y}')
            .drift(y, f'-nu*{y}')
            .noise('xi', Gaussian(var='2*D'), couples={'x': 1})
            .noise('zeta', Gaussian(var='2*D'), couples={y: 1})
            .build())


def _tikz_view(y):
    import re
    from daedalus.api import compute_cumulants
    from daedalus.engine.diagrams.tikz_export import (edge_style_map, to_tikz_feynman)
    res = compute_cumulants(_twin_model(y), k=2, max_ell=1,
                            external_fields=[('x', 1), (y, 1)],
                            tau_grid=[0.0, 0.5], parallel=False,
                            verbose=False, use_cache=False)
    tds = [d['typed_diagram'] for d in res['diagrams']]
    tex = '\n'.join(to_tikz_feynman(td, style_by_field=True,
                                      propagator_label='auto') for td in tds)
    return tds, tex, sorted(edge_style_map(tds)), \
        sorted(set(re.findall(r'\\delta [A-Za-z]+', tex)))


def test_tikz_field_names_ending_in_t_and_real_edge_keys():
    """GEN-13 + ACC-71: renaming the second field q -> input must only
    rename the symbols (it used to eat the last letter of 'input'), and the
    per-edge propagator labels must reach the output (the reconstructed
    (u, v, copy) key never matched a real typed diagram, so every label was
    the bare 'G')."""
    import re
    ref = _tikz_view('q')
    new = _tikz_view('input')
    sub = lambda s: s.replace('q', 'input')                  # noqa: E731
    assert sorted(sub(str(p)) for p in ref[2]) == sorted(str(p) for p in new[2])
    assert sorted(sub(e) for e in ref[3]) == sorted(new[3])
    # the matrix-element labels are present and name both fields
    labs = set(re.findall(r"edge label'=\\\((G[^)]*)\\\)", new[1]))
    assert labs and any('input' in l for l in labs), labs
    assert any(l == 'G_{x}' or 'x' in l for l in labs)
    # and no mangled symbol anywhere
    assert 'inpu' not in re.sub(r'input', '', new[1])


def test_tikz_field_symbol_roles():
    from daedalus.engine.diagrams.tikz_export import _field_symbol
    assert _field_symbol(('dinput', 1), 'physical') == 'input'
    assert _field_symbol(('inputt', 1), 'response') == 'input'
    assert _field_symbol(('inputt', 1), 'physical') == 'inputt'


# --------------------------------------------------------------------------
# LOUD-06: generate_report(result=...) is self-describing
# --------------------------------------------------------------------------


def test_generate_report_reuses_result_without_parameters(tmp_path, monkeypatch):
    """The cover page is labelled from the result's config; no parameters=
    needed (it used to raise AttributeError on fundamental.items())."""
    import numpy as np
    from daedalus.api import compute_cumulants, report
    seen = {}

    def fake_cover(pdf, model, k, max_ell, fundamental, external_fields, result):
        seen.update(k=k, max_ell=max_ell, fundamental=fundamental,
                    external_fields=external_fields, name=model.get('name'))
    monkeypatch.setattr(report, '_draw_cover_page', fake_cover)
    monkeypatch.setattr(report, '_draw_diagram_page', lambda *a, **k: None)
    res = compute_cumulants(_twin_model('q'), k=2, max_ell=1,
                            external_fields=[('x', 1), ('q', 1)],
                            tau_grid=np.array([0.0, 0.5]), parallel=False,
                            verbose=False, use_cache=False)
    report.generate_report({}, 2, output_pdf=str(tmp_path / 'r.pdf'),
                           result=res, verbose=False)        # call says max_ell=0
    assert seen['max_ell'] == 1 and seen['k'] == 2
    assert seen['fundamental'] and 'mu' in seen['fundamental']
    assert seen['external_fields'] and seen['name'] == 'w9_twin_q'


# --------------------------------------------------------------------------
# LOUD-12 / ACC-72: symbolic bridge exponents and default_values
# --------------------------------------------------------------------------


def test_bridge_accepts_exact_float_exponent_of_a_parameter():
    """D**0.5 is the rational power D**(1/2): scan and conversion agree
    (the old scan skipped floats, the conversion then blamed omega)."""
    from sage.all import I, SR, var
    from daedalus.api.symbolic_out import _Bridge
    w, D, a = var('w D a')
    b_f = _Bridge([D**0.5 / (a + I * w)], w)
    b_q = _Bridge([D**(1 / SR(2)) / (a + I * w)], w)
    assert [str(x) for x in b_f.pow_of] == [str(x) for x in b_q.pow_of]
    f = b_f.to_field(D**0.5 / (a + I * w))
    assert f == b_q.to_field(D**(1 / SR(2)) / (a + I * w))


def test_bridge_refuses_inexact_exponent_with_a_clear_message():
    from sage.all import I, pi, var
    from daedalus.api.symbolic_out import _Bridge, NonRationalPropagatorError
    w, D, a = var('w D a')
    expr = D**pi / (a + I * w)
    with pytest.raises(NonRationalPropagatorError, match='rational'):
        _Bridge([expr], w).to_field(expr)


def test_default_values_failure_is_not_silent():
    import warnings
    from daedalus.api.symbolic_out import TreeCovariance
    tc = TreeCovariance.__new__(TreeCovariance)
    tc._num_params = {}
    tc._cache = {}
    tc._model = object()          # no usable model: the solve must fail
    tc._ft = object()
    tc._parameters = None
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        assert tc.default_values() == {}
    assert any('default_values' in str(x.message) for x in w)


# --------------------------------------------------------------------------
# LOUD-10: noise library edge cases
# --------------------------------------------------------------------------


def test_finite_pmf_accepts_round_off_in_probabilities():
    """FinitePMF and Hyperexponential share one tolerance (1e-12)."""
    import numpy as np
    from daedalus.api.noise import FinitePMF, Hyperexponential, NoiseError
    p = (np.array([0.1, 0.2, 0.7]) * (1 + 2e-16)).tolist()
    assert sum(p) != 1.0
    FinitePMF([0, 1, 2], p)
    Hyperexponential(p, [1, 2, 3])
    with pytest.raises(NoiseError, match='sum'):
        FinitePMF([0, 1, 2], [0.1, 0.2, 0.71])


def test_cgf_non_analytic_names_the_problem():
    from daedalus.api.noise import CGF, NoiseError
    with pytest.raises(NoiseError, match='analytic'):
        CGF('1/theta')


# --------------------------------------------------------------------------
# LOUD-11: simulator edge cases (runaway rates, single path, mu <= 0)
# --------------------------------------------------------------------------


def _hawkes(w):
    from daedalus.api.delay import Kernel
    from daedalus.api.sde import SDE
    return (SDE('w9_hawkes').population('E', size=2)
            .parameter('a', default=[1.0, 1.0], indexed_by=['E'])
            .parameter('Em', default=[-1.5, -1.5], indexed_by=['E'])
            .parameter('tau', default=2.0, domain='positive')
            .parameter('w', default=w, indexed_by=['E', 'E'])
            .define_kernel('r', kernel=Kernel.exponential('tau'))
            .point_process('n', 'a*exp(Em + sum(w[i,j]*Conv(r, n[j]) for j in E))',
                           population='E'))


def test_runaway_exponential_hawkes_refuses_loudly():
    """A supercritical exp-Hawkes diverges; the old run died inside numpy
    ('lam value too large') after producing garbage."""
    from daedalus.api.pp_simulate import PPSimulationError
    m = _hawkes([[5.0, 5.0], [5.0, 5.0]])
    with pytest.raises(PPSimulationError, match='diverged'):
        m.simulate(T=50, dt=0.01, n_paths=4, bin=0.25, seed=0)


def test_subcritical_hawkes_still_simulates_and_standard_errors_need_two_paths():
    from daedalus.api import pp_simulate as pp
    m = _hawkes([[0.1, 0.1], [0.1, 0.1]])
    rec = m.simulate(T=40, dt=0.01, n_paths=4, bin=0.25, seed=1, burn_in=10)
    rate, se = pp.mean_rate(rec, 'n', 0)
    assert rate > 0 and se >= 0
    one = m.simulate(T=40, dt=0.01, n_paths=1, bin=0.25, seed=1, burn_in=10)
    with pytest.raises(ValueError, match='2 independent paths'):
        pp.mean_rate(one, 'n', 0)
    with pytest.raises(ValueError, match='2 independent paths'):
        pp.cross_covariance(one, ('n', 0), ('n', 1), [0, 1])


@pytest.mark.parametrize('mu', [0.0, -0.5])
def test_spatial_simulators_refuse_non_positive_mass(mu):
    from daedalus.simulations.spatial_field_1d_sim import simulate
    from daedalus.simulations.spatial_field_2d_sim import simulate_2d
    with pytest.raises(ValueError, match='positive mass'):
        simulate(L=4.0, N=16, mu=mu, n_steps=20, burn_in=5, record_every=5)
    with pytest.raises(ValueError, match='positive mass'):
        simulate_2d(L=4.0, N=8, mu=mu, n_steps=20, burn_in=5, record_every=5)
    simulate(L=4.0, N=16, mu=0.5, n_steps=20, burn_in=5, record_every=5)


def test_single_path_estimators_refuse_instead_of_returning_nan():
    import numpy as np
    from daedalus.simulations.delay_sde_sim import autocovariance
    from daedalus.simulations.spatial_field_1d_sim import third_cumulant_x
    with pytest.raises(ValueError, match='at least 2'):
        autocovariance(np.zeros((50, 1, 1)), 0, 0, [0, 1])
    with pytest.raises(ValueError, match='at least 2'):
        third_cumulant_x(np.zeros((1, 16)))


# --------------------------------------------------------------------------
# B3: spatial raw k=2 moments that need loop-corrected means refuse up front
# --------------------------------------------------------------------------


def _driven_rd(h):
    from daedalus.api.model import SpatialModelBuilder
    return (SpatialModelBuilder('w9 driven rd', n_populations=0)
            .physical_field('phi', spatial_dim=1, description='density')
            .parameter('mu', default=1.0, domain='positive')
            .parameter('D', default=1.0, domain='positive')
            .parameter('g', default=0.3, domain='real')
            .parameter('h', default=h, domain='real')
            .parameter('T', default=1.0, domain='positive')
            .set_action_text(
                'phit*((Dt + mu - D*Laplacian)*phi + g*phi^2 - h) - T*phit^2')
            .mf_residual(lhs='(Dt + mu - D*Laplacian)*phi', rhs='-g*phi^2 + h')
            .boundary('infinite').initial('stationary').build())


def _count_k2_calls(monkeypatch):
    import daedalus.api as api
    real = api.compute_cumulants
    calls = []

    def spy(*a, **kw):
        calls.append((kw.get('k'), kw.get('max_ell')))
        return real(*a, **kw)
    monkeypatch.setattr(api, 'compute_cumulants', spy)
    return calls


def test_spatial_raw_moment_two_loops_refuses_before_any_run(monkeypatch):
    import daedalus as dd
    calls = _count_k2_calls(monkeypatch)
    cfg = dd.Config(k=2, max_ell=2, chi_grid=[0.0, 1.0], tau_max=0.0,
                    tau_step=1.0, output='moment',
                    external_fields=[('phi', 1), ('phi', 1)])
    with pytest.raises(NotImplementedError, match='raw moment'):
        dd.run(_driven_rd(0.0), cfg)
    assert calls == []


def test_spatial_raw_moment_one_loop_nonzero_saddle_refuses_before_the_loop_run(
        monkeypatch):
    import daedalus as dd
    calls = _count_k2_calls(monkeypatch)
    cfg = dd.Config(k=2, max_ell=1, chi_grid=[0.0, 1.0], tau_max=0.0,
                    tau_step=1.0, output='moment',
                    external_fields=[('phi', 1), ('phi', 1)])
    with pytest.raises(NotImplementedError, match='raw moment'):
        dd.run(_driven_rd(0.5), cfg)
    assert (2, 1) not in calls and (1, 0) in calls     # only the tree-level saddle probe ran
