"""Feature B: the δ components of a cumulant (``api.cumulant``).

Oracles:
* linear Hawkes, tree: the autocorrelation spike weight is the mean rate
  λ̄ = (I − aW)⁻¹ aE exactly; the power spectrum is
  S(ω) = (I − A)⁻¹ diag(λ̄) (I − A)⁻†, A = aW / ((1 + iωτ)(1 + iωτ_g));
* a point process's autocorrelation spike weight is its mean rate at every
  loop order, so the ℓ = 1 δ weights sum to the k = 1, ℓ = 1 mean (spike
  reset); for k = 3 the coefficient of δ(t_1 − t_2) is the k = 2 smooth
  cumulant at t_3 − t_1, and the t_1 = t_2 = t_3 point has weight λ̄.
"""
import json
import os

import numpy as np
import pytest

pytest.importorskip('sage.all')

import daedalus as dd                                        # noqa: E402
from api.cumulant import (Cumulant, DeltaComponent,          # noqa: E402
                          DeltaRecordsIncompleteWarning, _rref,
                          _components_from_records, _bin_index)

# the zoo's spike-reset working point (rate ≈ 4.2, ℓ = 1 shift ≈ −0.9 %)
P_SPIKE = {'Em': [3.5, 3.5], 'tau': [10.0, 9.0], 'a': [2.5, 2.5],
           'w': [[0.55, 0.65], [0.7, 0.8]]}


def _run(name, k, ell, ext, fund=None, tau=(-1.0, -0.5, 0.0, 0.5, 1.0),
         **kw):
    model, mod = dd.load_model(name)
    f = dict(getattr(mod, 'DEFAULT_FUNDAMENTAL', {}) or {}) or \
        dd.fundamental_from_model(model)
    f.update(fund or {})
    return dd.compute_cumulants(model, k=k, max_ell=ell,
                                external_fields=list(ext), fundamental=f,
                                tau_grid=np.asarray(tau, float),
                                parallel=False, verbose=False, **kw)


def _lam_linear_hawkes(fund=None):
    _, mod = dd.load_model('linear_hawkes')
    F = dict(mod.DEFAULT_FUNDAMENTAL)
    F.update(fund or {})
    E, W = np.array(F['E'], float), np.array(F['w'], float)
    return F, np.linalg.solve(np.eye(2) - F['a'] * W, F['a'] * E)


# ── unit level (no pipeline) ─────────────────────────────────────────────

def test_rref_jacobian():
    # δ(2 t1 − 2 t2) = δ(t1 − t2) / 2
    M, c, f = _rref([[2.0, -2.0]], [0.0])
    assert np.allclose(M, [[1.0, -1.0]], rtol=1e-12, atol=1e-12) and f == pytest.approx(0.5, rel=1e-12)
    # δ(−t1 + t2 − 3) = δ(t1 − t2 + 3)
    M, c, f = _rref([[-1.0, 1.0]], [-3.0])
    assert np.allclose(M, [[1.0, -1.0]], rtol=1e-12, atol=1e-12) and np.allclose(c, [3.0], rtol=1e-12, atol=1e-12) and f == 1
    # two rows: δ²(t1 − t3, t2 − t3) in another basis, |det| = 2
    M, c, f = _rref([[1.0, 1.0, -2.0], [1.0, -1.0, 0.0]], [0.0, 0.0])
    assert np.allclose(M, [[1, 0, -1], [0, 1, -1]], rtol=1e-12, atol=1e-12) and f == pytest.approx(.5, rel=1e-12)
    assert _rref([[1.0, -1.0], [2.0, -2.0]], [0.0, 0.0]) is None


def test_synthetic_record_wick_sum_and_scaling():
    """A record δ(2x + 1) with coefficient 3 + x in the first mapping's free
    time x = t_2 − t_1, two Wick permutations, compensation 1: the identity
    fires at τ = −1/2 and the swap (x = t_1 − t_2 = −τ) at τ = +1/2, both
    with coefficient 3 + (−1/2) and the Jacobian 1/2."""
    rec = {'coeff_fc': lambda x: 3.0 + x, 'equality_a': [2.0],
           'equality_c': 1.0, 'retardation_data': [], 'free_ext_idx': [1],
           'wick_perms': [(0, 1), (1, 0)], 'wick_compensation': 1,
           'origin_leaf_idx': 0, 'n_ext': 2, 'diagram_index': 0}
    comps = _components_from_records([rec], 2, 0, 'test')
    got = sorted((d.location, d.weight.real) for d in comps)
    assert got == [(pytest.approx(-0.5, rel=1e-12), pytest.approx(1.25, rel=1e-12)),
                   (pytest.approx(0.5, rel=1e-12), pytest.approx(1.25, rel=1e-12))]
    # compensation 2: each permutation carries half
    rec['wick_compensation'] = 2
    got2 = sorted(d.weight.real for d in
                  _components_from_records([rec], 2, 0, 'test'))
    assert got2 == [pytest.approx(0.625, rel=1e-12), pytest.approx(0.625, rel=1e-12)]


def test_synthetic_k3_record_maps_each_permutation_to_its_plane():
    """k = 3, origin leg 0, free x = (t_2 − t_1, t_3 − t_1); a record
    δ(x_1) with an asymmetric coefficient 1 + x_2 and the half-space
    x_2 > 0.  Identity: plane t_1 = t_2, weight 1 + (t_3 − t_1) for t_3 >
    t_1, else 0.  Swapping legs 2 and 3: plane t_1 = t_3, weight
    1 + (t_2 − t_1) for t_2 > t_1."""
    rec = {'coeff_fc': lambda x1, x2: 1.0 + x2, 'equality_a': [1.0, 0.0],
           'equality_c': 0.0, 'retardation_data': [([0.0, 1.0], 0.0)],
           'free_ext_idx': [1, 2], 'wick_perms': [(0, 1, 2), (0, 2, 1)],
           'wick_compensation': 1, 'origin_leaf_idx': 0, 'n_ext': 3,
           'diagram_index': 0}
    comps = {d.equality: d for d in
             _components_from_records([rec], 3, 1, 'test')}
    p12, p13 = ((1.0, -1.0, 0.0), 0.0), ((1.0, 0.0, -1.0), 0.0)
    assert set(comps) == {p12, p13}
    assert comps[p12].weight_at((0.0, 0.0, 0.5)) == pytest.approx(1.5, rel=1e-12)
    assert comps[p12].weight_at((0.0, 0.0, -0.5)) == 0
    assert comps[p13].weight_at((0.2, 0.9, 0.2)) == pytest.approx(1.7, rel=1e-12)
    assert comps[p13].weight_at((0.2, 0.1, 0.2)) == 0
    assert comps[p12].location is None          # a hyperplane at k = 3


def test_bin_index_nonuniform_and_outside():
    tau = np.array([-1.0, 0.0, 0.5, 2.0])
    assert _bin_index(tau, 0.0) == (1, pytest.approx(0.75, rel=1e-12))
    assert _bin_index(tau, 0.3) == (2, pytest.approx(1.0, rel=1e-12))
    assert _bin_index(tau, 3.0) == (None, None)
    # unsorted grid: same bins
    assert _bin_index(tau[::-1], 0.3) == (1, pytest.approx(1.0, rel=1e-12))


def test_spike_on_a_bin_edge_is_split():
    from api.cumulant import _bin_deposits
    tau = np.linspace(-1.0, 1.0, 10)          # no τ = 0 point
    dep = sorted(_bin_deposits(tau, 0.0))
    w = tau[1] - tau[0]
    assert [i for i, _ in dep] == [4, 5]
    assert all(f == pytest.approx(0.5 / w, rel=1e-12) for _, f in dep)
    assert _bin_deposits(tau, 0.05) == [(5, pytest.approx(1 / w, rel=1e-12))]


# ── pipeline oracles ─────────────────────────────────────────────────────

def test_linear_hawkes_tree_weight_is_the_rate():
    F, lam = _lam_linear_hawkes()
    for i in (1, 2):
        r = _run('linear_hawkes', 2, 0, [('n', i), ('n', i)])
        c = dd.cumulant(r)
        assert c.complete and len(c.deltas) == 1
        d = c.deltas[0]
        assert isinstance(d, DeltaComponent) and d.location == 0.0
        assert d.loop_order == 0 and d.diagram == f'ell=0 #{d.diagram_index}'
        # the label indexes the result's diagram list of that order
        assert [x for x in r['diagrams'] if x['ell'] == 0][d.diagram_index]
        assert abs(d.weight - lam[i - 1]) <= 1e-13 * lam[i - 1]
        # the spike is NOT in C_tau
        assert np.array_equal(c.total_on_grid(mode='exclude'), r['C_tau'])
    cross = dd.cumulant(_run('linear_hawkes', 2, 0, [('n', 1), ('n', 2)]))
    assert cross.complete and cross.deltas == []


@pytest.fixture(scope='module')
def spike_reset_runs():
    # discarded warm-up: a cold on-disk cache and a warm one pick different
    # (isomorphic) diagram representatives, ~1e-16 apart, and the tests below
    # compare runs bit for bit
    _run('single_population_spike_reset_test', 2, 1, [('n', 1), ('n', 1)],
         P_SPIKE)
    out = {}
    out['k2'] = _run('single_population_spike_reset_test', 2, 1,
                     [('n', 1), ('n', 1)], P_SPIKE)
    out['k1'] = _run('single_population_spike_reset_test', 1, 1,
                     [('n', 1)], P_SPIKE, tau=(0.0,))
    return out


def test_one_loop_delta_weight_is_the_one_loop_mean(spike_reset_runs):
    r = spike_reset_runs['k2']
    c = dd.cumulant(r)
    assert c.complete, c.incomplete_reasons
    assert c.source == 'phase_j_per_diagram'
    w = c.delta_weights_by_ell()
    nstar = r['mf_values']['nstar'][0]
    assert abs(w[0][0.0] - nstar) <= 1e-10 * nstar
    mean1 = complex(spike_reset_runs['k1']['C_tau_by_ell'][1][0])
    assert abs(w[1][0.0] - mean1) <= 1e-12 * abs(mean1)
    assert all(d.provenance['integrated'] for d in c.deltas
               if d.loop_order == 1)
    assert all('theta0_rows_on_surface' in d.provenance for d in c.deltas
               if d.loop_order == 1)


# (``test_integrated_records_never_touch_C_tau``, the switch-off check of
# ``USE_SHOTNOISE_INTEGRATED``, whose off path dropped the integrated
# shot-noise subsets, went with that switch in audit W7.)


def test_records_survive_the_pole_cluster_deformation(monkeypatch):
    """τ = τ_g with weak diagonal coupling: near-degenerate poles resolved
    by the deformation (records combined over the nodes).  S3: the pole
    engine's deformation (the state-space route, the default, bypasses
    it)."""
    monkeypatch.setenv('DAEDALUS_PHASE_J_ENGINE', 'poles')
    fund = {'tau': 2.5, 'tau_g': 2.5, 'w': [[1e-4, 0.0], [0.0, 2e-4]]}
    F, lam = _lam_linear_hawkes(fund)
    r = _run('linear_hawkes', 2, 0, [('n', 2), ('n', 2)], fund)
    assert (r.get('pole_clusters') or {}).get('status') == 'resolved'
    c = dd.cumulant(r)
    assert c.complete and len(c.deltas) == 1
    d = c.deltas[0]
    assert d.provenance['pole_deformation_nodes'] >= 2
    assert abs(d.weight - lam[1]) <= 1e-12 * lam[1]


def test_k3_hyperplanes_and_point_spike():
    F, lam = _lam_linear_hawkes()
    r3 = _run('linear_hawkes', 3, 0, [('n', 1)] * 3, tau=(0.0,))
    c = dd.cumulant(r3)
    assert c.complete, c.incomplete_reasons
    pts = [d for d in c.deltas if d.codim == 2]
    assert len(pts) == 1 and pts[0].location == (0.0, 0.0)
    assert abs(pts[0].weight - lam[0]) <= 1e-13 * lam[0]
    with pytest.raises(ValueError):
        c.deltas[0].weight                      # a hyperplane: weight_at
    taus = [-2.0, -0.5, 0.5, 2.0]
    r2 = _run('linear_hawkes', 2, 0, [('n', 1), ('n', 1)], tau=taus)
    line = [d for d in c.deltas if d.codim == 1
            and d.equality == ((1.0, -1.0, 0.0), 0.0)]
    assert len(line) >= 2                      # several diagrams
    for t, c2 in zip(taus, r2['C_tau']):
        w = sum(d.weight_at((0.0, 0.0, t)) for d in line)
        assert abs(w - c2) <= 1e-11 * abs(c2)
    with pytest.raises(ValueError):
        line[0].weight_at((0.0, 0.3, 1.0))      # off the surface


# ── grids, spectrum, save / load, plots ──────────────────────────────────

def test_total_on_grid_bin_convention():
    F, lam = _lam_linear_hawkes()
    tau = np.arange(-2.0, 2.0001, 0.25)
    r = _run('linear_hawkes', 2, 0, [('n', 1), ('n', 1)], tau=tau)
    c = dd.cumulant(r)
    tot = c.total_on_grid(mode='bin')
    i0 = int(np.argmin(np.abs(tau)))
    diff = tot - r['C_tau']
    assert abs(diff[i0] - lam[0] / 0.25) <= 1e-12 * lam[0] / 0.25
    assert np.all(diff[np.arange(tau.size) != i0] == 0)
    frame = c.to_frame()
    assert np.array_equal(frame['C_total_bin'], tot)
    assert np.array_equal(frame['C_smooth'], r['C_tau'])
    # off-grid evaluation from the callables (τ = 0 at the Itô left limit)
    t2 = np.array([-0.5, 0.0, 0.5])
    assert np.allclose(c.total_on_grid(t2, mode='exclude'),
                       r['C_tau'][[6, 8, 10]], rtol=1e-12, atol=0)
    with pytest.raises(ValueError):
        c.total_on_grid(mode='both')
    with pytest.raises(ValueError):
        c.total_on_grid(mode='exclude', ell=3)          # no such order
    # a one-point grid: no bin columns, no crash
    r1 = _run('linear_hawkes', 2, 0, [('n', 1), ('n', 1)], tau=(0.0,))
    fr = dd.cumulant(r1).to_frame()
    assert 'C_total_bin' not in fr and fr['C_smooth'].shape == (1,)
    # a binned sim equal to the bin-convention theory has zero residual
    cmp_ = c.compare_to_sim({'tau': tau, 'C': tot,
                             'C_err': np.ones(tau.size)})
    assert np.array_equal(cmp_['theory'], tot) and np.all(cmp_['z'] == 0)


def test_spectrum_matches_exact_linear_hawkes():
    """Exact to <= 1e-6 (measured ~1e-14 auto, ~1e-12 cross) with the
    callables integrated by QAWF on (0, ∞) (no grid involved)."""
    F, lam = _lam_linear_hawkes()
    a, W = F['a'], np.array(F['w'], float)

    def S_exact(w):
        A = a * W / ((1 + 1j * w * F['tau']) * (1 + 1j * w * F['tau_g']))
        M = np.linalg.inv(np.eye(2) - A)
        return M @ np.diag(lam) @ M.conj().T
    om = np.array([-0.7, 0.0, 0.05, 0.4, 2.0])
    for ext, (i, j) in (([('n', 1), ('n', 1)], (0, 0)),
                        ([('n', 1), ('n', 2)], (1, 0))):
        c = dd.cumulant(_run('linear_hawkes', 2, 0, ext))
        S = c.spectrum(om)
        Se = np.array([S_exact(w)[i, j] for w in om])
        assert np.max(np.abs(S - Se) / np.abs(Se)) <= 1e-6


def test_save_load_round_trip(tmp_path):
    F, lam = _lam_linear_hawkes()
    npz, csv = str(tmp_path / 'r.npz'), str(tmp_path / 'r.csv')
    _run('linear_hawkes', 2, 0, [('n', 1), ('n', 1)], output_npz=npz,
         output_csv=csv)
    d = dd.load_npz(npz)
    assert d['delta_weight'].shape == (1,)
    assert abs(d['delta_weight'][0] - lam[0]) <= 1e-13 * lam[0]
    assert d['delta_location'][0] == 0.0 and d['delta_ell'][0] == 0
    assert str(d['delta_equality'][0]) == 't_1 - t_2 = 0'
    assert json.loads(str(d['delta_provenance'][0]))['source'] == \
        'phase_j_per_diagram'
    assert 'PHASE_J_T_MIN=' in str(d['phase_j_flags'][0])
    c = Cumulant.from_saved(d)
    assert c.complete and c.deltas[0].weight == d['delta_weight'][0]
    assert np.array_equal(c.total_on_grid(mode='exclude'), d['C_total'])
    lines = [ln for ln in open(csv) if ln.startswith('#   {')]

    def strict(x):
        raise ValueError(f'non-standard JSON constant {x}')
    assert len(lines) == 1
    assert json.loads(lines[0][4:], parse_constant=strict)['ell'] == 0
    # a file written before Feature B (no delta_* arrays) still loads
    old = str(tmp_path / 'old.npz')
    np.savez(old, **{k: v for k, v in d.items()
                     if not k.startswith('delta_')})
    c_old = Cumulant.from_saved(dd.load_npz(old))
    assert not c_old.complete and c_old.deltas == []


def test_plot_smoke(tmp_path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    r = _run('linear_hawkes', 2, 0, [('n', 1), ('n', 1)],
             tau=np.arange(-3.0, 3.001, 0.5))
    sim = {'tau': r['tau_grid'], 'C': dd.cumulant(r).total_on_grid().real,
           'C_err': np.full(r['tau_grid'].shape, 0.01)}
    for style in ('arrow', 'bin'):
        fig = dd.plot_cumulant(r, show_delta=True, delta_style=style, sim=sim)
        fig.savefig(str(tmp_path / f'{style}.png'))
        plt.close(fig)
    r3 = _run('linear_hawkes', 3, 0, [('n', 1)] * 3, tau=(0.0,))
    fig = dd.plot_cumulant(r3, show_delta=True)
    plt.close(fig)
    with pytest.raises(ValueError):
        dd.plot_cumulant(r, show_delta=True, delta_style='cone')
    with pytest.raises(ValueError):
        dd.plot_cumulant(r, show_delta=True, residual=True)


def test_rejects_wrong_k_and_non_results():
    r = _run('linear_hawkes', 2, 0, [('n', 1), ('n', 1)])
    with pytest.raises(ValueError):
        dd.cumulant(r, k=3)
    with pytest.raises(TypeError):
        dd.cumulant({'C_tau': None})


# ── slow: the nonlinear Hawkes ℓ = 1 oracle and a simulation bin ────────

@pytest.mark.slow
def test_nonlinear_hawkes_one_loop_delta_weight():
    """Quadratic-rate Hawkes with exponential synapses (~3 min)."""
    P_SP = {'Em': [0.8, 0.78], 'tau': [10.0, 9.0], 'a': [0.44, 0.44],
            'taug': [[2.0, 3.0], [1.0, 3.0]], 'w': [[0.25, 0.25],
                                                   [0.2, 0.3]]}
    name = 'single_population_quad_exp_test'
    c = dd.cumulant(_run(name, 2, 1, [('n', 1), ('n', 1)], P_SP,
                         tau=(-0.5, 0.0, 0.5)))
    mean1 = complex(_run(name, 1, 1, [('n', 1)], P_SP,
                         tau=(0.0,))['C_tau_by_ell'][1][0])
    assert c.complete
    assert abs(c.delta_weights_by_ell()[1][0.0] - mean1) <= 1e-10 * abs(mean1)


@pytest.mark.slow
def test_spike_reset_weight_against_a_simulation_bin():
    """Euler simulation of the model (Itô: rate and reset from the pre-step
    v): the mean rate (= the δ weight) and the τ = 0 bin Var(N_Δ)/Δ² against
    the theory through ℓ = 1, which a tree-level weight misses by > 10 SE.
    Reference run (T = 1e5, 10 seeds, dt 5e-4..4e-3): rate 4.166..4.170
    ± 0.0015 vs theory 4.1709 (tree 4.2089)."""
    numba = pytest.importorskip('numba')

    @numba.njit
    def sim(n_steps, dt, tau, a, Em, W, v0, seed):
        np.random.seed(seed)
        v = v0.copy()
        tot = 0.0
        sp = np.zeros(2)
        for _ in range(n_steps):
            for i in range(2):
                lam = max(a[i] * v[i], 0.0)
                sp[i] = np.random.poisson(lam * dt)
            tot += sp[0]
            for i in range(2):
                inp = W[i, 0] * sp[0] + W[i, 1] * sp[1]
                v[i] += (dt * (Em[i] - v[i]) + inp - v[i] * sp[i]) / tau[i]
        return tot

    r = _run('single_population_spike_reset_test', 2, 1,
             [('n', 1), ('n', 1)], P_SPIKE)
    w = dd.cumulant(r).delta_weights_by_ell()
    w_tree, w_1 = w[0][0.0].real, (w[0][0.0] + w[1][0.0]).real
    T, dt = 2.0e4, 0.002
    p = {k: np.asarray(v, float) for k, v in P_SPIKE.items()}
    rates = [sim(int(T / dt), dt, p['tau'], p['a'], p['Em'], p['w'],
                 np.array(r['mf_values']['vstar'], float), 100 + s) / T
             for s in range(8)]
    m, se = np.mean(rates), np.std(rates, ddof=1) / np.sqrt(len(rates))
    assert abs(m - w_1) <= 4 * se + 2e-3
    assert abs(m - w_tree) > abs(m - w_1)
