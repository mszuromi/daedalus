"""Compute and cache every number plotted in Figure 1.

Every curve is now cheap.  Panel (a) is a sample path; panel (b) is a simulated
C(0) sweep plus tree / 1-loop / 2-loop / 3-loop theory, one real pipeline run
per plotted epsilon.

The 3-loop curve used to be opt-in behind ``--loop3``, farmed out to
``_fig1_loop3_worker.py`` subprocesses, and reconstructed by fitting a single
c3 coefficient, because one epsilon at ``max_ell=3`` was measured at >10.5 h
and 3.2 GB.  That cost is gone: the enumeration work merged in 95c79e6
(canonical-form dedup, the |V| <= 3k+3l-3 orientability cap, and the shared
(k,ell)-keyed prediagram cache) brings a full max_ell=3 point to ~0.5 s of
evaluation.  Stage 3 therefore runs ell=3 directly at every epsilon and the
curve is per-point exact rather than fitted.

Accuracy anchor: at eps=0.01 the pipeline reproduces the exact series
    C(0) = 1 - 3 eps + 24 eps^2 - 297 eps^3 + 4896 eps^4     (mu = D = 1)
through the 3-loop coefficient -297 to a relative error of 1.2e-13.

Note the series is ASYMPTOTIC -- its coefficients grow factorially.  At the
right edge of the panel (eps = 0.06) the third loop improves the error only
0.034 -> 0.030, and a fourth would make it worse (0.033).  That is the point
the figure is making, not a defect.

Usage::

    sage -python make_fig1_data.py
"""
import os, subprocess, sys, time

import numpy as np

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _root)
sys.path.insert(0, os.path.join(_root, 'notebooks'))
os.chdir(os.path.join(_root, 'notebooks'))

import daedalus as dd
from simulations.ou_langevin_sim_numba import sim_ou_quartic_numba

f = float
MU, D = f(1.0), f(1.0)
EPS_TRACE = f(0.06)                 # nonlinearity shown in the sample path
EPS_MAX   = f(0.06)                 # right edge of panel (b)
N_DENSE, N_SIM = int(61), int(7)

# 3-loop epsilons, most important first.  ONE suffices to fix c3; the rest are
# the cross-check that the O(eps^3) scaling holds.  The job is resumable, so an
# unfinished tail simply means fewer cross-checks.
EPS_L3 = [f(0.06), f(0.03), f(0.05), f(0.04)]

OUT = os.path.join('data', 'fig1_figdata.npz')
os.makedirs('data', exist_ok=True)
_state = {}


def checkpoint(**kw):
    _state.update(kw)
    np.savez(OUT, **_state)
    print('  [checkpoint] %s  (%d arrays)' % (OUT, len(_state)), flush=True)


# ── 1. sample path (panel a) ────────────────────────────────────────────────
print('[1/4] sample path', flush=True)
dt_tr, bs_tr, T_tr = f(0.0025), int(8), f(30.0)
n_tr = int(T_tr / dt_tr); nb_tr = int(n_tr // bs_tr)
_ = sim_ou_quartic_numba(int(1000), dt_tr, MU, f(0.05), D, f(0.0),
                         bs_tr, int(100), int(0))                 # JIT warmup
x_tr = np.asarray(sim_ou_quartic_numba(n_tr, dt_tr, MU, EPS_TRACE, D, f(0.0),
                                       bs_tr, nb_tr, int(11)), float).ravel()
t_tr = np.arange(nb_tr) * (dt_tr * bs_tr)
checkpoint(x_tr=x_tr, t_tr=t_tr, T_tr=T_tr, eps_trace=EPS_TRACE,
           mu=MU, D=D, eps_max=EPS_MAX)

# ── 2. simulated C(0) sweep (panel b points) ────────────────────────────────
print('[2/4] simulated C(0) sweep', flush=True)
dt_s, bs_s, T_s, N_RUNS = f(0.0025), int(4), f(3.0e5), int(6)
ns = int(T_s / dt_s); nbs = int(ns // bs_s)
eps_sim = np.linspace(f(0.0), EPS_MAX, N_SIM)
C_sim, C_err = np.zeros(N_SIM), np.zeros(N_SIM)
t0 = time.perf_counter()
for i, e in enumerate(eps_sim):
    v = np.array([f(np.var(sim_ou_quartic_numba(ns, dt_s, MU, f(e), D, f(0.0),
                                                bs_s, nbs, int(500 + 7 * i + r))))
                  for r in range(N_RUNS)])
    C_sim[i], C_err[i] = f(v.mean()), f(v.std(ddof=1) / np.sqrt(N_RUNS))
print('  %.1f s' % (time.perf_counter() - t0), flush=True)
checkpoint(eps_sim=eps_sim, C_sim=C_sim, C_err=C_err)

# ── 3. tree / 1-loop / 2-loop: a real pipeline run per plotted epsilon ──────
print('[3/3] dense max_ell=3 sweep (%d pipeline runs)' % N_DENSE, flush=True)
model, mod = dd.load_model('ou_quartic')
eps_dense = np.linspace(f(1.0e-4), EPS_MAX, N_DENSE)
C_dense = np.zeros((4, N_DENSE))          # rows: tree, +1loop, +2loop, +3loop
t0 = time.perf_counter()
for i, e in enumerate(eps_dense):
    cfg = dd.Config(k=2, max_ell=int(3), external_fields=[('dx', 1), ('dx', 1)],
                    parameters={'mu': MU, 'eps': f(e), 'D': D},
                    tau_grid=(f(-1.0), f(1.0), int(5)), parallel=False)
    r = dd.run(model, cfg, mod)
    tau = np.asarray(r['tau_grid'], float); i0 = int(np.argmin(np.abs(tau)))
    inc = np.array([f(np.asarray(r['C_tau_by_ell'][l], float).ravel()[i0])
                    for l in (int(0), int(1), int(2), int(3))])
    C_dense[:, i] = np.cumsum(inc)
print('  %.1f s' % (time.perf_counter() - t0), flush=True)
checkpoint(eps_dense=eps_dense, C_dense=C_dense, has_loop3=True)

# NOTE: 3-loop used to be opt-in behind ``--loop3``, farmed out to
# ``_fig1_loop3_worker.py`` subprocesses, and the curve was reconstructed by
# fitting a single c3 coefficient and extrapolating C2 + c3*eps^3.  That was a
# workaround for a cost that no longer exists: after the enumeration work on
# perf/canonical-form-dedup (canonical-form dedup, the |V| <= 3k+3l-3
# orientability cap, and the shared (k,ell)-keyed prediagram cache) a full
# max_ell=3 point evaluates in ~0.5 s, so stage 3 above now runs ell=3 directly
# at every plotted epsilon.  The curve is per-point exact rather than fitted.
print('\ndone.  3-loop computed directly at all %d epsilons.' % N_DENSE, flush=True)
