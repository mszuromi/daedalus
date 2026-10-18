"""
Item 53: the tier-1 residue clip and the RAM-aware memory guards.

1. Tier-1 residues (``api._propagator``) were clipped below 1e-12 of the
   LARGEST entry at their pole.  With exact pole membership every member
   residue is exactly nonzero, so the clip could only drop a genuine residue
   -- the off-diagonal entry of two fields whose units differ by 1e14.  On
   ``x' = -a x + c z + noise, z' = -b z + noise`` with ``c = 1e-14`` the pole
   engine returned ``C_xz == 0`` (the state-space engine, which does not
   read the residues for the tree, was right).  Oracle: the Lyapunov
   covariance, ``C_xz(tau) = [S expm(A^T tau)]_xz``.

2. ``PHASE_J_MEM_BUDGET_GB`` / ``SPATIAL_MEM_BUDGET_GB`` unset: the budget is
   ``min(6 GB, 0.6 x RAM - this process)``, at least max(0.5 GB, 10 % of RAM)
   (``final_integral.default_mem_budget``); explicit values as before.  The
   refusals say how the budget was set and, from the pre-pass, the RAM.

3. ``periodic_inverse_ft`` in d=3 built the whole (2 n_cut + 1)^3 lattice
   (13.6 GB at L = 30, k_max = 60: the peak of ``test_spatial_correlator``,
   outside any guard); it now sums one n1 slab at a time.
"""
from __future__ import annotations

import math
import os
import sys
import tracemalloc
import warnings

import mpmath as mp
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from api import compute_cumulants                                   # noqa: E402
from api.noise import Gaussian                                      # noqa: E402
from api.sde import SDE                                             # noqa: E402
from engine.integration.time_domain import final_integral as FI     # noqa: E402

GB = 1e9
TAU = np.array([0.0, 0.5, 1.0, 2.0])


# ═══════════════════════════════════════════════════════════════════════
# 1. tier-1 residues of fields in very different units
# ═══════════════════════════════════════════════════════════════════════

def _units_model(a, b, c, D, Dz, name='units_split'):
    """x' = -a x + c z + sqrt(2D) xi,  z' = -b z + sqrt(2Dz) zeta."""
    return (SDE(name).parameter('a', default=a).parameter('b', default=b)
            .parameter('c', default=c).parameter('D', default=D)
            .parameter('Dz', default=Dz).field('x').field('z')
            .drift('x', '-a*x + c*z').drift('z', '-b*z')
            .noise('xi', Gaussian(var='2*D'), couples={'x': 1})
            .noise('zeta', Gaussian(var='2*Dz'), couples={'z': 1}).build())


def _oracle(a, b, c, D, Dz, i, j, taus):
    """<f_i(t) f_j(t + tau)> of the linear system, 50 digits."""
    with mp.workdps(50):
        A = mp.matrix([[-a, c], [0, -b]])
        Q = mp.matrix([[2 * D, 0], [0, 2 * Dz]])
        K = mp.zeros(4, 4)
        for r in range(2):
            for s in range(2):
                for u in range(2):
                    for v in range(2):
                        K[2 * r + s, 2 * u + v] = (A[r, u] * (s == v)
                                                   + (r == u) * A[s, v])
        sv = mp.lu_solve(K, mp.matrix([-Q[0, 0], -Q[0, 1], -Q[1, 0],
                                       -Q[1, 1]]))
        S = mp.matrix([[sv[0], sv[1]], [sv[2], sv[3]]])
        return np.array([float((S * mp.expm(A.T * mp.mpf(t)))[i, j])
                         for t in taus])


def _run(model, ext, engine, monkeypatch):
    monkeypatch.setenv('DAEDALUS_PHASE_J_ENGINE', engine)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return compute_cumulants(model, k=2, max_ell=0,
                                 external_fields=list(ext), parallel=False,
                                 verbose=False, use_cache=False,
                                 tau_grid=TAU)


@pytest.mark.parametrize('engine', ['poles', 'statespace'])
@pytest.mark.parametrize('c', [1e-14, 1e-20])
def test_small_cross_residue_is_kept(engine, c, monkeypatch):
    a, b, D, Dz = 1.0, 0.5, 1e-28, 1.0
    r = _run(_units_model(a, b, c, D, Dz), [('x', 1), ('z', 1)], engine,
             monkeypatch)
    got = np.array([complex(v).real for v in r['C_tau']])
    ref = _oracle(a, b, c, D, Dz, 0, 1, TAU)
    # tau = 0 is the Ito left limit (evaluated at -1e-6): compare tau > 0
    np.testing.assert_allclose(got[1:], ref[1:], rtol=1e-12, atol=0)
    # the residue of G_xz at both poles is ~c / (a - b): kept, not clipped
    p = r['propagator']
    assert (p.get('pole_analysis') or {}).get('residue_route') == 'double'
    for C in p['C_mats']:
        assert abs(complex(C[0, 1])) == pytest.approx(c / (a - b), rel=1e-12, abs=0)


def test_large_field_driven_by_tiny_one(monkeypatch):
    """The other way round: a field of unit size driven by one 1e-15 times
    smaller (the cross residue is then the LARGE entry at the driver's pole
    and the driver's own residue the small one)."""
    # x' = -a x + c z with z tiny: scale z by 1e-15 -> c = 1e15, Dz = 1e-30
    a, b, c, D, Dz = 1.0, 3.0, 1e15, 1e-4, 1e-30
    r = _run(_units_model(a, b, c, D, Dz), [('x', 1), ('x', 1)], 'poles',
             monkeypatch)
    got = np.array([complex(v).real for v in r['C_tau']])
    ref = _oracle(a, b, c, D, Dz, 0, 0, TAU)
    np.testing.assert_allclose(got[1:], ref[1:], rtol=1e-11, atol=0)
    r = _run(_units_model(a, b, c, D, Dz), [('z', 1), ('z', 1)], 'poles',
             monkeypatch)
    got = np.array([complex(v).real for v in r['C_tau']])
    ref = _oracle(a, b, c, D, Dz, 1, 1, TAU)
    np.testing.assert_allclose(got[1:], ref[1:], rtol=1e-11, atol=0)


# ═══════════════════════════════════════════════════════════════════════
# 2. the default memory budget
# ═══════════════════════════════════════════════════════════════════════

@pytest.fixture
def machine(monkeypatch):
    """Pretend the machine has ``ram`` bytes and this process ``fp``."""
    def set_(ram, fp):
        monkeypatch.setattr(FI, '_ram_bytes', lambda: ram)
        monkeypatch.setattr(FI, 'process_footprint_bytes', lambda: fp)
    monkeypatch.delenv('PHASE_J_MEM_BUDGET_GB', raising=False)
    monkeypatch.delenv('SPATIAL_MEM_BUDGET_GB', raising=False)
    return set_


@pytest.mark.parametrize('ram_gb, fp_gb, want_gb', [
    (8, 1.5, 0.6 * 8 - 1.5),        # a small machine: 3.3 GB, not 6
    (8, 4.6, 0.8),                  # already short: the floor, 10 % of RAM
    (4, 2.0, 0.5),                  # ... and never below 0.5 GB
    (16, 1.5, 6.0),                 # 9.6 - 1.5 > 6: the cap
    (16, 5.0, 0.6 * 16 - 5.0),
    (64, 1.5, 6.0),
    (64, 30.0, 6.0),
    (64, 40.0, 6.0),                # a big process on a big machine: 10 %
    (16, 12.0, 1.6),
])
def test_default_budget_follows_the_machine(machine, ram_gb, fp_gb, want_gb):
    machine(ram_gb * GB, fp_gb * GB)
    assert FI.phase_j_mem_budget_bytes({}) == pytest.approx(want_gb * GB, rel=1e-12)
    b, rule = FI.mem_budget('SPATIAL_MEM_BUDGET_GB', {})
    assert b == pytest.approx(want_gb * GB, rel=1e-12)
    assert rule.startswith('default min(6.00 GB, 0.6 x')
    assert f'{ram_gb * GB / 1e9:.2f} GB RAM' in rule
    assert rule.endswith(f'= {want_gb:.2f} GB')
    assert ('at least' in rule) == (0.6 * ram_gb - fp_gb < want_gb)


def test_default_budget_unknown_ram_or_footprint(machine):
    machine(None, 1 * GB)
    b, rule = FI.default_mem_budget()
    assert b == 6 * GB and 'unknown' in rule
    machine(8 * GB, None)
    b, rule = FI.default_mem_budget()
    assert b == pytest.approx(0.6 * 8 * GB, rel=1e-12) and 'taken as 0' in rule


@pytest.mark.parametrize('ram_gb', [8, 16, 64])
def test_explicit_budget_is_used_as_given(machine, ram_gb):
    machine(ram_gb * GB, 1.5 * GB)
    f = FI.phase_j_mem_budget_bytes
    assert f({'PHASE_J_MEM_BUDGET_GB': '24'}) == 24 * GB
    assert f({'PHASE_J_MEM_BUDGET_GB': '0.5'}) == 0.5 * GB
    for off in ('off', 'inf', 'none', 'unlimited'):
        assert f({'PHASE_J_MEM_BUDGET_GB': off}) is None
    # 0 / negative raise (they meant "no limit" here and "refuse everything"
    # for SPATIAL_MEM_BUDGET_GB; audit W7, RED-05)
    for bad in ('0', '-1'):
        with pytest.raises(ValueError, match='positive number of GB'):
            f({'PHASE_J_MEM_BUDGET_GB': bad})
    with pytest.raises(ValueError, match='PHASE_J_MEM_BUDGET_GB'):
        f({'PHASE_J_MEM_BUDGET_GB': 'lots'})


@pytest.mark.skipif(not (sys.platform == 'darwin'
                         or sys.platform.startswith('linux')),
                    reason='footprint measured on macOS and Linux only')
def test_process_footprint_is_measured():
    fp = FI.process_footprint_bytes()
    assert fp is not None and 50e6 < fp < 64 * GB
    # The growth is measured in a fresh interpreter: inside a long test run
    # the allocator can serve 200 MB from memory the process already holds
    # (glibc keeps freed arenas resident), so the footprint need not grow.
    import subprocess
    code = ('import numpy as np, engine.integration.time_domain.final_integral '
            'as FI\nfp = FI.process_footprint_bytes()\n'
            'big = np.ones(25_000_000)\n'               # 200 MB, touched
            'print(FI.process_footprint_bytes() - fp)\n')
    root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
    out = subprocess.run([sys.executable, '-c', code], cwd=root, check=True,
                         capture_output=True, text=True, timeout=600)
    assert int(out.stdout.strip().splitlines()[-1]) > 150e6


def test_running_guard_names_the_default_rule(machine):
    machine(8 * GB, 4.6 * GB)                            # budget: 0.8 GB floor
    with pytest.raises(FI.PhaseJResourceError) as ei:
        FI._plan_guard(10 ** 6, 10 ** 6, where='a test subset')
    msg = str(ei.value)
    assert 'over the budget of default min(6.00 GB, 0.6 x 8.00 GB RAM' in msg
    assert 'at least 0.80 GB = 0.80 GB' in msg
    assert ei.value.estimate['budget_bytes'] == pytest.approx(0.8 * GB, rel=1e-12)
    FI._plan_guard(10 ** 5, 10 ** 5)                    # 0.11 GB: fits


def test_scope_keeps_the_budget_it_started_with(machine):
    machine(16 * GB, 1 * GB)
    with FI.phase_j_memory_scope(what='scope test'):
        machine(8 * GB, 4.6 * GB)                        # memory got short
        with FI._PLAN_MEM_LOCK:
            scoped, budget, rule, _used = FI._budget_now()
        assert scoped and budget == 6 * GB and rule.endswith('= 6.00 GB')


def _oscillator():
    from tests.test_zero_mode_pruning import _oscillator as osc
    return osc()


def test_prepass_refusal_states_ram_and_estimate(machine, monkeypatch):
    machine(8 * GB, 1.5 * GB)
    monkeypatch.setenv('DAEDALUS_PHASE_J_ENGINE', 'poles')   # pole tuples
    monkeypatch.setenv('PHASE_J_MEM_BUDGET_GB', '0.0001')
    with pytest.raises(FI.PhaseJResourceError) as ei:
        compute_cumulants(_oscillator(), k=3, max_ell=1,
                          external_fields=[('dT', 1)] * 3, parallel=False,
                          verbose=False)
    msg = str(ei.value)
    est = ei.value.estimate
    assert 'This machine has 8.0 GB of RAM' in msg
    assert f"the estimate is {est['bytes'] / 1e9:.2f} GB" in msg
    assert 'PHASE_J_MEM_BUDGET_GB=0.0001' in msg
    assert est['budget_rule'] == 'PHASE_J_MEM_BUDGET_GB=0.0001'


def test_default_budget_reaches_the_prepass(machine):
    machine(8 * GB, 1.5 * GB)
    r = compute_cumulants(_oscillator(), k=2, max_ell=1,
                          external_fields=[('dT', 1)] * 2, parallel=False,
                          verbose=False, tau_grid=np.array([0.0, 0.7]))
    est = r['phase_j_memory']
    assert est['budget_bytes'] == pytest.approx(3.3 * GB, rel=1e-12)
    assert est['budget_rule'].startswith('default min(6.00 GB')


def test_run_reads_the_budget_once(monkeypatch):
    """The pre-pass and the running scope use one reading of the default
    budget, however the process grows in between (reviewer finding)."""
    monkeypatch.delenv('PHASE_J_MEM_BUDGET_GB', raising=False)
    monkeypatch.setattr(FI, '_ram_bytes', lambda: 8 * GB)
    calls = []

    # +10 MB per read (the run-growth guard of audit W8 reads the footprint
    # at every checkpoint: a fake +0.5 GB per read is a growth it refuses)
    def growing():
        calls.append(1)
        return (1.5 + 0.01 * (len(calls) - 1)) * GB
    monkeypatch.setattr(FI, 'process_footprint_bytes', growing)
    r = compute_cumulants(_oscillator(), k=2, max_ell=1,
                          external_fields=[('dT', 1)] * 2, parallel=False,
                          verbose=False, tau_grid=np.array([0.0, 0.7]))
    est = r['phase_j_memory']
    assert len(calls) > 3                                 # it did grow
    # the first reading (1.5 GB held) sets the budget of the whole run
    assert est['budget_bytes'] == pytest.approx((4.8 - 1.5) * GB, rel=1e-12)
    assert FI._PLAN_MEM['budget'] == est['budget_bytes']
    assert FI._PLAN_MEM['rule'] == est['budget_rule']


def _allen_cahn():
    import importlib.util
    path = os.path.join(os.path.dirname(__file__), '..', 'models',
                        'allen_cahn_1d_subcritical_infinite.model.py')
    spec = importlib.util.spec_from_file_location('ac53', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.build()


def _spatial_loop():
    return compute_cumulants(
        model=_allen_cahn(), k=2, max_ell=1,
        fundamental={'mu': 1.0, 'D': 1.0, 'lam': 0.1, 'T': 1.0},
        external_fields=[('phi', 1), ('phi', 2)], tau_max=1.0, tau_step=1.0,
        spatial_grid=np.array([0.0, 1.0]), parallel=False, verbose=False,
        use_cache=True)


def test_spatial_guard_uses_the_default_rule(machine, monkeypatch):
    from engine.integration.spatial.heat_kernel import SpatialPropagatorError
    machine(8 * GB, 1.5 * GB)
    monkeypatch.setenv('PHASE_J_MEM_BUDGET_GB', 'off')     # the spatial guard
    monkeypatch.setattr(FI, 'MEM_BUDGET_CAP_GB', 1e-12)
    monkeypatch.setattr(FI, 'MEM_BUDGET_FLOOR_GB', 0.0)
    with pytest.raises(SpatialPropagatorError) as ei:
        _spatial_loop()
    assert 'default min(0.00 GB, 0.6 x 8.00 GB RAM' in str(ei.value)
    monkeypatch.setenv('SPATIAL_MEM_BUDGET_GB', '1e-12')
    with pytest.raises(SpatialPropagatorError) as ei:
        _spatial_loop()
    assert 'SPATIAL_MEM_BUDGET_GB=1e-12' in str(ei.value)
    monkeypatch.setenv('SPATIAL_MEM_BUDGET_GB', '6')     # explicit: runs
    assert np.isfinite(np.asarray(_spatial_loop()['C_tau_x'])).all()


# ═══════════════════════════════════════════════════════════════════════
# 3. the periodic lattice sum, one slab at a time
# ═══════════════════════════════════════════════════════════════════════

def _full_lattice(q, Cq, xs, d, L):
    """The previous whole-lattice sum (reference; small lattices only)."""
    two_pi_L = 2 * math.pi / L
    k_max = float(q[-1])
    n = int(math.ceil(k_max / two_pi_L)) + 1
    ns = np.arange(-n, n + 1).astype(float)
    grids = np.meshgrid(*([ns] * d), indexing='ij')
    k = two_pi_L * np.sqrt(sum(g * g for g in grids))
    Sk = np.where(k <= k_max, np.interp(k, q, Cq.real)
                  + 1j * np.interp(k, q, Cq.imag), 0)
    return np.array([np.sum(np.cos(two_pi_L * grids[0] * x) * Sk)
                     for x in xs]) / L ** d


@pytest.mark.parametrize('d', [1, 2, 3])
@pytest.mark.parametrize('L', [4.0, 9.0])
def test_slab_sum_equals_the_whole_lattice(d, L):
    from engine.integration.spatial.spatial_correlator import (
        periodic_inverse_ft)
    q = np.linspace(20.0 / 8000, 20.0, 2000)
    Cq = (1.0 + 0.3j) / (1.0 + q * q)
    xs = np.array([0.0, 0.3, 1.5])
    ref = _full_lattice(q, Cq, xs, d, L)
    got = periodic_inverse_ft(q, Cq, xs, d, L=L)
    np.testing.assert_allclose(got, ref, rtol=1e-13, atol=1e-15)
    got_r = periodic_inverse_ft(q, Cq.real, xs, d, L=L)
    assert got_r.dtype == float
    np.testing.assert_allclose(got_r, ref.real, rtol=1e-13, atol=1e-15)


def test_slab_sum_memory_is_a_slab_not_the_lattice():
    """d=3, n_cut = 193: the whole lattice is 387^3 doubles = 464 MB per
    array (~9 arrays before); the slab sum holds a few 387^2 arrays."""
    from engine.integration.spatial.spatial_correlator import (
        periodic_inverse_ft)
    q = np.linspace(60.0 / 24000, 60.0, 6000)
    Cq = 1.0 / (1.0 + q * q)
    tracemalloc.start()
    try:
        v = periodic_inverse_ft(q, Cq, np.array([0.3, 1.5]), 3, L=20.0)
        _cur, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert np.isfinite(v).all()
    assert peak < 30e6
