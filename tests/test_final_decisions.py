"""Decisions accepted after the FINAL verification (Oct 2026): W2 (a tracked
two-field OU quartic model), W5 (Cumulants above its declared order warns),
F2 (dd.run k >= 3 slices at the exact tie), F6 (the simulators read
PHASE_J_MEM_BUDGET_GB)."""
import os
import subprocess
import sys
import warnings

import mpmath as mp
import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


# ── W2: the two-field OU quartic is a tracked model ───────────────────────────

def test_two_field_ou_quartic_is_tracked_and_used_by_the_zoo():
    from tests.tools import phase_j_zoo_baseline as ZB
    out = subprocess.run(['git', 'ls-files', 'daedalus/models/ou_quartic_coupled.model.py'],
                         cwd=ROOT, capture_output=True, text=True)
    if out.returncode != 0:
        pytest.skip('not a git checkout')
    assert out.stdout.strip() == 'daedalus/models/ou_quartic_coupled.model.py'
    names = [e['name'] for e in ZB.all_entries(include_local=False)
             if e.get('model') == 'ou_quartic_coupled']
    assert names == ['ou_quartic_coupled-k2-l2-xx', 'ou_quartic_coupled-k2-l2-xy']
    for n in names:
        assert ZB.destination(ZB.entry_by_name(n)) == 'tracked'


def test_two_field_ou_quartic_equals_the_inline_model():
    """The tracked file is the textbook model of test_phase_j_setup_fastpath:
    same tree and one loop, bit for bit."""
    import daedalus as dd
    from daedalus.api import compute_cumulants
    sys.path.insert(0, HERE)
    from test_phase_j_setup_fastpath import _two_field_ou
    tracked, _ = dd.load_model('ou_quartic_coupled')
    kw = dict(k=2, max_ell=1, external_fields=[('dx', 1), ('dy', 1)],
              tau_grid=np.array([0.0, 0.7, 2.0]), parallel=False, verbose=False,
              use_cache=False)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        a = compute_cumulants(tracked, **kw)
        b = compute_cumulants(_two_field_ou(), **kw)
    for ell in (0, 1):
        assert np.array_equal(np.asarray(a['C_tau_by_ell'][ell]),
                              np.asarray(b['C_tau_by_ell'][ell]))


# ── W5: Cumulants above its declared order ────────────────────────────────────

def test_cumulants_source_warns_once_above_its_order():
    from daedalus.api.noise import Cumulants, CumulantTruncationWarning
    src = Cumulants([1.0, 0.5])                 # kappa2, kappa3: order 3
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        assert float(src.cumulant(3)) == 0.5    # declared: no warning
    with pytest.warns(CumulantTruncationWarning, match='up to order 3'):
        assert float(src.cumulant(4)) == 0.0
    with warnings.catch_warnings():
        warnings.simplefilter('error')          # once per source
        assert float(src.cumulant(5)) == 0.0
    with pytest.warns(CumulantTruncationWarning):
        Cumulants([1.0]).cumulant(3)            # a new source warns again


def _levy_ou(kappas):
    from daedalus.api.sde import SDE
    from daedalus.api.noise import Cumulants
    return (SDE('w5 cumulants ou').parameter('mu', default=1.0).field('x')
            .drift('x', '-mu*x').noise('xi', Cumulants(kappas), couples={'x': 1})
            .build())


def test_compute_cumulants_warns_when_a_vertex_order_is_above_the_declared_order():
    import daedalus.api.compute as C
    from daedalus.api import compute_cumulants
    from daedalus.api.noise import CumulantTruncationWarning
    C._CUMULANT_TRUNCATION_WARNED.clear()
    kw = dict(k=3, max_ell=0, external_fields=[('dx', 1)] * 3,
              tau_grid=np.array([0.0]), parallel=False, verbose=False, use_cache=False)
    m = _levy_ou([1.0])
    assert m['noise_declared_orders'] == {'xi': 2}
    with pytest.warns(CumulantTruncationWarning, match="'xi'.*up to order 2"):
        compute_cumulants(m, **kw)
    with warnings.catch_warnings(record=True) as rec:     # once per process
        warnings.simplefilter('always')
        compute_cumulants(m, **kw)
    assert not [w for w in rec if issubclass(w.category, CumulantTruncationWarning)]
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter('always')
        compute_cumulants(_levy_ou([1.0, 0.4]), **kw)   # order 3 declared
    assert not [w for w in rec if issubclass(w.category, CumulantTruncationWarning)]


# ── F2: dd.run k >= 3 slices at the exact tie ─────────────────────────────────

MU, G, D = 1.3, 0.4, 0.7


def _kappa3_exact(t):
    """Tree third cumulant of x' = -mu x - g x^2 + sqrt(2D) xi (one cubic
    vertex, exact in g): sum_i -g int_{-inf}^{t_i} e^{-mu(t_i - s)}
    2 C(s - t_j) C(s - t_k) ds with C(tau) = (D/mu) e^{-mu|tau|}."""
    C = lambda u: mp.mpf(D) / MU * mp.exp(-MU * abs(u))
    tot = mp.mpf(0)
    with mp.workdps(30):
        for i in range(3):
            j, l = [x for x in range(3) if x != i]
            f = lambda s: mp.exp(-MU * (t[i] - s)) * 2 * C(s - t[j]) * C(s - t[l])
            pts = sorted({mp.mpf(x) for x in t if x <= t[i]})
            tot += -G * mp.quad(f, [-mp.inf] + pts)
    return float(tot)


def _ou_quadratic():
    from daedalus.api.sde import SDE
    from daedalus.api.noise import Gaussian
    return (SDE('f2 ou quadratic').parameter('mu', default=MU).parameter('g', default=G)
            .parameter('D', default=D).field('x').drift('x', '-mu*x - g*x^2')
            .noise('xi', Gaussian(var='2*D'), couples={'x': 1}).build())


@pytest.mark.parametrize('base', [[0.0, 0.0], [0.0, 0.5]])
def test_k3_slices_are_exact_at_ties(base, monkeypatch):
    import daedalus as dd
    monkeypatch.delenv('DAEDALUS_TAU0', raising=False)
    taus = [-0.6, 0.0, 0.5, 1.2]
    cfg = dd.Config(k=3, max_ell=0, external_fields=[('x', 1)] * 3, tau_grid=taus,
                    kpoint_base_lags=base, parallel=False, verbose=False)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        r = dd.run(_ou_quadratic(), cfg)
    for j in (1, 2):
        for i, t in enumerate(taus):
            times = [0.0] + list(base)
            times[j] = t
            ref = _kappa3_exact(times)
            got = complex(r['C_tau_slices'][j][i])
            assert abs(got - ref) <= 1e-13 * abs(ref), (base, j, t, got, ref)
            # the times are the nominal ones: no 1e-6 offsets
            assert list(r['_kpoint_slice_times'][j][i]) == times
    # a tied point lists the swept leg last (the engine's tie order makes it
    # the earliest leg: the left limit)
    assert list(r['_kpoint_slice_orders'][1][1]) == [0, 2, 1]


def test_k3_full_grid_uses_the_nominal_times(monkeypatch):
    """The full grid evaluates every leg at its nominal time (0 is 0, not
    -1e-6): equal to the exact third cumulant at every grid point, ties
    included (continuous here, so any tie order gives the same value)."""
    import daedalus as dd
    monkeypatch.delenv('DAEDALUS_TAU0', raising=False)
    taus = [-0.5, 0.0, 0.5]
    cfg = dd.Config(k=3, max_ell=0, external_fields=[('x', 1)] * 3, tau_grid=taus,
                    kpoint_full_grid=True, parallel=False, verbose=False)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        r = dd.run(_ou_quadratic(), cfg)
    grid = np.asarray(r['C_tau_grid'])
    assert grid.shape == (3, 3)
    for a, t1 in enumerate(taus):
        for b, t2 in enumerate(taus):
            ref = _kappa3_exact([0.0, t1, t2])
            assert abs(complex(grid[a, b]) - ref) <= 1e-13 * abs(ref), (t1, t2)


def test_k3_nudge_mode_is_the_old_placement(monkeypatch):
    import daedalus as dd
    monkeypatch.setenv('DAEDALUS_TAU0', 'nudge')
    cfg = dd.Config(k=3, max_ell=0, external_fields=[('x', 1)] * 3,
                    tau_grid=[0.0, 0.5], parallel=False, verbose=False)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        r = dd.run(_ou_quadratic(), cfg)
    t = r['_kpoint_slice_times'][1]
    assert t[0][1] == pytest.approx(-2e-6, abs=1e-18) and t[1][2] == -1e-6
    ref = _kappa3_exact([0.0, 0.5, 0.0])
    # the 1e-6 offsets move the value by O(rate x 1e-6) (4e-7 here)
    assert 1e-8 < abs(complex(r['C_tau_slices'][1][1]) - ref) / abs(ref) < 1e-5


def test_k3_tie_is_the_left_limit_with_mixed_fields():
    """Legs of different fields: a tied point needs the run with the swept leg
    last (a second compute_cumulants), and is the left limit of its slice,
    not the right one (spike-reset has a jump at the tie)."""
    import daedalus as dd
    from tests.tools import phase_j_zoo_baseline as ZB
    e = ZB.entry_by_name('spike_reset-k2-l1')
    m, mod, _ = ZB._load_model(e)
    P = ZB.resolve_params(e, m, mod)
    cfg = dd.Config(k=3, max_ell=0, external_fields=[('n', 1), ('v', 1), ('n', 2)],
                    tau_grid=[0.0, 0.7], parameters=P, parallel=False, verbose=False)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        r = dd.run(m, cfg, mod)
    f = r['total_C']
    for j in (1, 2):
        tied = complex(r['C_tau_slices'][j][0])
        times = [0.0, 0.0, 0.0]
        times[j] = -1e-10
        left = complex(f(*times))
        times[j] = 1e-10
        right = complex(f(*times))
        assert abs(tied - left) <= 1e-8 * abs(tied)
        assert abs(tied - right) > 1e-3 * abs(tied)


# ── F6: the simulators read PHASE_J_MEM_BUDGET_GB ─────────────────────────────

def test_pp_simulate_reads_the_phase_j_budget(monkeypatch):
    sys.path.insert(0, HERE)
    from test_point_process_front_end import pp_spike_only
    s = pp_spike_only(1)
    monkeypatch.setenv('PHASE_J_MEM_BUDGET_GB', '1e-5')        # 10 kB
    with pytest.raises(MemoryError, match='PHASE_J_MEM_BUDGET_GB'):
        s.simulate(T=2.0, dt=1e-3, n_paths=3, burn_in=0.0)     # 48 kB
    monkeypatch.setenv('PHASE_J_MEM_BUDGET_GB', 'off')         # no limit
    out = s.simulate(T=2.0, dt=1e-3, n_paths=3, burn_in=0.0, seed=1)
    assert sum(v.nbytes for v in out['counts'].values()) == 2000 * 3 * 8
    monkeypatch.setenv('PHASE_J_MEM_BUDGET_GB', '1e-5')        # explicit wins
    s.simulate(T=2.0, dt=1e-3, n_paths=3, burn_in=0.0, seed=1, max_bytes=1e6)


def test_coupled_rd_simulator_reads_the_phase_j_budget(monkeypatch):
    from daedalus.simulations.coupled_rd_1d_sim import simulate_coupled_rd_1d
    monkeypatch.setenv('PHASE_J_MEM_BUDGET_GB', '0.01')
    with pytest.raises(MemoryError, match='PHASE_J_MEM_BUDGET_GB'):
        simulate_coupled_rd_1d(np.eye(2), [1.0, 1.0], np.eye(2), t_run=1e3)


@pytest.mark.parametrize('pair', ['xx', 'xy'])
def test_two_field_ou_quartic_zoo_entries_reproduce_their_values(pair):
    """The zoo entries ``ou_quartic_coupled-k2-l2-*`` have no frozen pre-M1
    baseline (their model was a local file then; ``_ADDED_AFTER_FREEZE`` in
    test_phase_j_legacy_baseline.py).  Their values (bit-identical to the
    local model's at main 3053a45, zoo A/B of the final decisions) are pinned
    here."""
    from tests.tools import phase_j_zoo_baseline as ZB
    from daedalus.api import compute_cumulants
    ref = np.load(os.path.join(HERE, 'fixtures', 'zoo_ou_quartic_coupled.npz'))
    e = ZB.entry_by_name(f'ou_quartic_coupled-k2-l2-{pair}')
    model, module, _ = ZB._load_model(e)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        r = compute_cumulants(model, k=2, max_ell=2, external_fields=[tuple(x) for x in e['ext']],
                              parameters=ZB.resolve_params(e, model, module),
                              tau_grid=np.asarray(e['tau_grid'], float), parallel=False,
                              verbose=False, use_cache=False)
    assert np.array_equal(np.asarray(r['tau_grid'], float), ref[f'{pair}::tau_grid'])
    for ell in (0, 1, 2):
        got = np.asarray(r['C_tau_by_ell'][ell], dtype=complex)
        want = ref[f'{pair}::ell{ell}']
        assert np.max(np.abs(got - want)) <= 1e-12 * np.max(np.abs(want)), ell
