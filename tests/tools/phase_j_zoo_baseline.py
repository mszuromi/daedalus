"""
tests/tools/phase_j_zoo_baseline.py
===================================
Model-zoo baseline for Phase J (``docs/integration_speedup_plan.md`` §4.3,
milestone M0.3).  It records the pre-change reference data that the legacy
rollback umbrella test (``DAEDALUS_PHASE_J_LEGACY=1``, M1 onward) compares
against forever.

For every zoo entry (model × config) it records:

* values: the full per-loop-order arrays and their total.
  - k in {1, 2}: the pipeline's own τ-grid arrays (``C_tau_by_ell``).  The k=2
    τ=0 grid point is evaluated at t1 = -``api.compute._ITO_EPS`` (Itô left
    limit), exactly as ``compute_cumulants`` does.
  - k >= 3: the raw per-ell callables at explicit external-time points.  When
    a point has a non-anchor leg at 0, the API-nudged value (every such leg
    moved to -1e-6, the per-leg mapping ``daedalus._args`` applies on the full
    k-point grid) is recorded too.  (Since 0.2.0 the points of ``dd.run``'s
    k >= 3 slices are placed by ``daedalus._kpoint_slice_times`` instead: a
    swept leg that meets another leg goes one eps below it, so a recorded
    ``*_api_nudged`` value is an engine value at those times, not
    necessarily a slice's tau = 0 point.)
  - spatial: ``C_tau_x`` and the cumulative ``C_tau_x_by_order``.
* counters: the full ``final_integral._RUNTIME_COUNTERS`` snapshot, including
  ``nquad_calls`` (real scipy.nquad calls, m>=1), ``polytope_m0_direct``
  (m=0 δ-collapsed subsets, evaluated directly), ``nquad_fallback_by_reason``,
  ``zero_normal_rows_seen`` and ``scipy_nquad_called_m1/_m2/_mge3``.  M0.3
  records used an older ``nquad_calls`` that also counted m=0 entries; see
  ``COUNTER_SEMANTICS`` / ``migrate_counters``.  It is taken after the k>=3 point
  evaluation, which is where the k>=3 Phase J integrals run.  (With nquad
  stubbed, the per-m stub call counts are in ``stub_calls``: the stub
  replaces ``_integrate_polytope``, so its counters stay 0.)
* spatial only: ``certify_modes`` pass/fail and the residual at each
  q in {0, 0.7, 1.5}, computed on the arguments the production run passed.
* wall time: the ``compute_cumulants`` wall, the point-evaluation wall and the
  whole child-process wall.
* diagram sources: where each (k, ell) diagram list came from (typed-diagram
  cache, or ``load_prediagrams``' v2 / v1 / shipped / computed / eager) and
  the cwd -- nquad counts and path censuses depend on them.  Not recorded
  for the M0.3 baseline (added afterwards); those runs were made from the
  repo root with ``use_cache=True``, i.e. from the developer's local caches.
* provenance: git HEAD, the dirty tracked files, sha256 of ``final_integral.py``
  / ``grouped_integral.py`` / the model file, the numeric flags, library
  versions and ``PYTHONHASHSEED``, which is required to be ``0``.

Each entry runs in its OWN subprocess with a hard timeout.  The default is
10 min; the four models that timed out in the earlier scan and the heavy
notebook models get 15 min.  A timeout is recorded as ``status='TIMEOUT'``.
Entries that share a model run sequentially in one worker, so this tool never
has two writers on one ``saved_models/<model>/`` cache.  The exception is an
explicit ``lane``: e.g. ℓ=2 entries that only add higher-order cache files
may get a lane of their own.  ``parallel=False`` is forced: no fork anywhere,
and the counters see every call.

Routing (what may enter a tracked file)
---------------------------------------
Only entries whose model definition file is tracked by git, or is new and not
git-ignored, go into the tracked outputs
``tests/fixtures/phase_j_legacy_baseline.{npz,json}``.  Every other entry
(git-ignored or excluded model files, or any spec flagged ``local_only``) goes
to ``scratch/integration_plan/m0_baseline_local.{npz,json}`` (gitignored).

Extension point (local-only entries)
------------------------------------
Entries for models that must never appear in a tracked file have no spec
here.  ``DAEDALUS_ZOO_EXTRA`` names one or more Python files
(``os.pathsep``-separated), each defining ``ZOO_EXTRA`` (or ``LOCAL_ZOO``): a
list of keyword dicts for ``_E``.  Every entry loaded from such a file is
``local_only``, so its values ALWAYS go to the local baseline.  Unset, the
gitignored default ``DEFAULT_ZOO_EXTRA``
(``scratch/integration_plan/m0_zoo_local_specs.py``) is loaded when it
exists; set to the empty string, nothing is.  A fresh clone has none.

An extra file may reference a parameter set of its model module instead of
copying the numbers: ``params={'__from_module__': ['PARAMS_ATTR', 'key']}``
runs with ``getattr(model_module, 'PARAMS_ATTR')['key']``, resolved at run
time.

Safety
------
* Configs that would take hours in scipy.nquad (fast poles at max_ell=2) are
  never listed.  Slow ℓ=2 entries get a hard timeout, and their nquad census
  is recorded with nquad stubbed.  Stubbed values are flagged
  ``census_only`` and are not baselines.

Usage (repo root; PYTHONHASHSEED=0 is required)::

    PYTHONHASHSEED=0 sage -python -m tests.tools.phase_j_zoo_baseline --list
    PYTHONHASHSEED=0 sage -python -m tests.tools.phase_j_zoo_baseline \\
        --run all --workers 4          # children -> scratch/phase_j_zoo_baseline/runs
    PYTHONHASHSEED=0 sage -python -m tests.tools.phase_j_zoo_baseline \\
        --run all --workers 4 --out-dir scratch/phase_j_zoo_baseline/runs_b
    PYTHONHASHSEED=0 sage -python -m tests.tools.phase_j_zoo_baseline \\
        --assemble --compare-dir scratch/phase_j_zoo_baseline/runs_b --note '...'

The second run feeds the per-entry ``reproducibility`` record (array_equal,
max_rel, counters_equal).  At M0.3 most entries were bit-identical run to run.
spike_reset (both paths) and multipopulation_test showed ulp-level jitter
(max rel 2.8e-16 to 1.6e-15), with PYTHONHASHSEED=0 in both runs.  So in-repo
comparisons use rtol 1e-13 (plan §4.4), never ``==``.

The baseline is never refrozen: ``--assemble`` refuses to replace it unless
``--overwrite-baseline`` is given.  A number-moving milestone re-runs the zoo
into its own directory, assembles it elsewhere and prints the delta table
against the baseline (status, max abs/rel change, nquad and zero-normal
counters before/after, Θ(0) counters; an entry that moves although the
baseline saw no zero-normal row AND the new run's ``M1_VALUE_KEYS`` counters
are all 0 is flagged ``VIOLATION``, see ``delta_table``)::

    PYTHONHASHSEED=0 sage -python -m tests.tools.phase_j_zoo_baseline \\
        --run all --workers 4 --out-dir scratch/<milestone>/runs
    PYTHONHASHSEED=0 sage -python -m tests.tools.phase_j_zoo_baseline \\
        --assemble --out-dir scratch/<milestone>/runs \\
        --assemble-to scratch/<milestone>/zoo --delta scratch/<milestone>/zoo

Importable API: ``ZOO``, ``all_entries()``, ``entry_by_name()``,
``run_entry()`` (in-process, used by ``tests/test_phase_j_legacy_baseline.py``),
``load_baseline()``, ``compare_values()``, ``load_merged()``, ``delta_table()``.
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import datetime
import hashlib
import importlib.util
import json
import math
import os
import platform
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..'))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

SCHEMA_VERSION = 1
#: Meaning of the ``nquad_calls`` runtime counter in a run record / baseline.
#:   1 -- M0.3 runs: every ``_integrate_polytope`` entry, INCLUDING m=0
#:        (a δ-collapsed subset, evaluated directly, not quadrature);
#:   2 -- ``nquad_calls`` = m>=1 entries only (real scipy.nquad, equal to
#:        ``scipy_nquad_called_m1 + _m2 + _mge3``), m=0 entries counted in
#:        ``polytope_m0_direct``.
#: Semantics-1 counters are converted EXACTLY by ``migrate_counters`` (the
#: split is derived from the per-m scipy counters recorded in the same run).
COUNTER_SEMANTICS = 2
WORK_DIR = os.path.join(_REPO_ROOT, 'scratch', 'phase_j_zoo_baseline', 'runs')
TRACKED_NPZ = os.path.join(_REPO_ROOT, 'tests', 'fixtures',
                           'phase_j_legacy_baseline.npz')
TRACKED_JSON = os.path.join(_REPO_ROOT, 'tests', 'fixtures',
                            'phase_j_legacy_baseline.json')
LOCAL_NPZ = os.path.join(_REPO_ROOT, 'scratch', 'integration_plan',
                         'm0_baseline_local.npz')
LOCAL_JSON = os.path.join(_REPO_ROOT, 'scratch', 'integration_plan',
                          'm0_baseline_local.json')
#: Environment variable naming the extra (local-only) zoo files, see the
#: module docstring.  Excluded from the recorded env knobs (it is a path).
ZOO_EXTRA_ENV = 'DAEDALUS_ZOO_EXTRA'
#: Loaded when ``DAEDALUS_ZOO_EXTRA`` is unset (only if it exists).
DEFAULT_ZOO_EXTRA = (os.path.join(_REPO_ROOT, 'scratch', 'integration_plan',
                                  'm0_zoo_local_specs.py'),)

DEFAULT_TIMEOUT = 600.0
LONG_TIMEOUT = 900.0
ITO_NUDGE = -1e-6            # daedalus._args k>=3 full-grid nudge (== api.compute._ITO_EPS)
SPATIAL_CERTIFY_Q = (0.0, 0.7, 1.5)
SEP = '::'                   # npz key separator: '<entry>::<array>'

# ═══════════════════════════════════════════════════════════════════════
# Zoo (plan §4.3).  One dict per (model, config).
# ═══════════════════════════════════════════════════════════════════════
#
# ``m1``: the expected effect of milestone M1 (the Θ(0) constant-row fix) on
# this entry, from the M0.8 exec'd-copy scan (fix=0 vs fix=1) and the plan:
#   'moves'      measured to move;
#   'unchanged'  measured bit-identical (no c_eff == 0 zero-normal polygon rows);
#   'unknown'    not measured (timed out / not scanned).
# The legacy umbrella test must reproduce EVERY entry under
# DAEDALUS_PHASE_J_LEGACY=1; with the new defaults only 'unchanged' entries
# may be compared directly.


def _E(name, *, model=None, model_file=None, kind='temporal', k=2, max_ell=1,
       ext=None, params=None, tau_grid=None, points=None, chi_grid=None,
       grouped=False, timeout=DEFAULT_TIMEOUT, row='', m1='unknown',
       notes='', stub_nquad=False, local_only=False, spatial_n_q=None,
       lane=None):
    return dict(
        name=name, model=model, model_file=model_file, kind=kind, k=int(k),
        max_ell=int(max_ell),
        ext=[list(e) for e in (ext or [])],
        params=params,
        tau_grid=None if tau_grid is None else [float(t) for t in tau_grid],
        points=None if points is None else [[float(x) for x in p]
                                            for p in points],
        chi_grid=None if chi_grid is None else [float(x) for x in chi_grid],
        grouped=bool(grouped), timeout=float(timeout), row=row, m1=m1,
        notes=notes, stub_nquad=bool(stub_nquad), local_only=bool(local_only),
        spatial_n_q=spatial_n_q, lane=lane)


T_OU = [0.0, 0.5, 2.0]
T3 = [0.0, 2.5, 10.0]
T_SPIKE = [0.0, 1.0, 3.0, 5.0, 10.0]

P_OU = {'mu': 1.0, 'eps': 0.02, 'D': 1.0}
P_OU_COLORED = {'mu': 1.0, 'eps': 0.02, 'D': 1.0, 'tauc': 0.5}
P_OU_2D_CC = {'mu1': 1.0, 'mu2': 1.0, 'eps1': 0.02, 'eps2': 0.02, 'J1': 0.3,
              'J2': 0.3, 'D1': 1.0, 'D2': 1.0, 'rho': 0.5, 'tauc': 2.0}
P_SPIKE = {'Em': [3.5, 3.5], 'tau': [10.0, 9.0], 'a': [2.5, 2.5],
           'w': [[0.55, 0.65], [0.7, 0.8]]}
P_QSPIKE = {'Em': [1.0, 1.0], 'tau': [10.0, 9.0], 'a': [0.5, 0.5],
            'w': [[0.25, 0.25], [0.2, 0.3]]}
P_SP = {'Em': [0.8, 0.78], 'tau': [10.0, 9.0], 'a': [0.44, 0.44],
        'taug': [[2.0, 3.0], [1.0, 3.0]], 'w': [[0.25, 0.25], [0.2, 0.3]]}
P_LD = {'Em': [0.8, 0.78], 'tau': [10.0, 9.0], 'w': [[0.0, 0.25], [0.2, 0.0]]}
P_LINH = {'E': [0.78, 0.81], 'w': [[0.30, 0.25], [0.30, 0.35]], 'tau': 10.0,
          'a': 1.0, 'tau_g': 2.5}
P_QH = {'E': [0.78, 0.81], 'w': [[0.30, 0.25], [0.30, 0.35]], 'tau': 10.0,
        'a': 0.44, 'tau_g': 2.5}
P_MP = {'tauE': [10.0, 9.5], 'tauI': [8.0, 7.0], 'EmE': [0.7, 0.72],
        'EmI': [0.4, 0.42], 'aE': [0.37, 0.41], 'aI': [0.23, 0.28],
        'wEE': [[0.25, 0.22], [0.21, 0.19]], 'wEI': [[0.12, 0.15], [0.13, 0.10]],
        'wIE': [[0.19, 0.17], [0.15, 0.12]], 'wII': [[0.12, 0.13], [0.14, 0.15]],
        'taugEE': [[4.0, 4.0], [3.0, 3.0]], 'taugEI': [[2.0, 1.0], [1.0, 3.0]],
        'taugIE': [[5.0, 6.0], [2.0, 1.0]], 'taugII': [[1.5, 1.2], [1.1, 1.0]]}
P_MPSR = {'tauE': [10.0, 9.5], 'tauI': [8.0, 7.0], 'EmE': [1.8, 1.8],
          'EmI': [0.9, 1.1], 'aE': [1.5, 1.5], 'aI': [0.9, 0.95],
          'wEE': [[0.35, 0.32], [0.31, 0.39]], 'wEI': [[0.12, 0.15], [0.13, 0.10]],
          'wIE': [[0.15, 0.16], [0.14, 0.13]], 'wII': [[0.15, 0.15], [0.15, 0.15]],
          'taugEE': [[4.0, 4.0], [3.0, 3.0]], 'taugEI': [[2.0, 1.0], [1.0, 3.0]],
          'taugIE': [[5.0, 6.0], [2.0, 1.0]], 'taugII': [[1.5, 1.2], [1.1, 1.0]]}
P_COND = {'Em': [0.5], 'tau': [10.0], 'taug': [[5.0]], 'a': [0.3], 'w': [[0.4]]}
P_DNL = {'aS': [0.3, 0.3], 'ES': [0.5, 0.5], 'ED': [0.0, 0.0],
         'tauS': [1.0, 1.0], 'tauD': [1.0, 1.0],
         'wSD': [[0.1, 0.03], [0.03, 0.1]], 'wDS': [[0.1, 0.03], [0.03, 0.1]]}
P_QHA = {'Em': [0.8, 0.78], 'tau': [10.0, 9.0], 'taug': [[2.0, 3.0], [1.0, 3.0]],
         'a': [0.44, 0.44], 'w': [[0.25, 0.25], [0.2, 0.3]]}

ZOO = [
    # ── OU quartic family: single pole, 0 nquad today; must stay unchanged ──
    _E('ou_quartic-k2-l3', model='ou_quartic', k=2, max_ell=3,
       ext=[('dx', 1), ('dx', 1)], params=P_OU, tau_grid=T_OU,
       row='ou_quartic', m1='unchanged'),
    # k=3: identically 0 at the symmetric saddle, no Phase J integral runs
    # (poset/interval/polygon attempted = 0): NOT tie evidence (use k=4).
    _E('ou_quartic-k3-l2', model='ou_quartic', k=3, max_ell=2,
       ext=[('dx', 1), ('dx', 1), ('dx', 1)], params=P_OU,
       points=[(0.0, 0.4, 1.0), (0.0, 0.7, 0.7), (0.0, 0.0, 0.7)],
       row='ou_quartic', m1='unchanged',
       notes='symmetric saddle: odd cumulants vanish; tied points kept'),
    _E('ou_quartic-k4-l2', model='ou_quartic', k=4, max_ell=2,
       ext=[('dx', 1)] * 4, params=P_OU,
       points=[(0.0, 0.3, 0.6, 0.9), (0.0, 0.5, 0.5, 0.5)],
       row='ou_quartic', m1='unchanged',
       notes='one generic point + one with t1=t2=t3 tied'),
    _E('ou_quartic_two_dim-k2-l2-xx', model='ou_quartic_two_dim', k=2,
       max_ell=2, ext=[('dx', 1), ('dx', 1)], params=None, tau_grid=T_OU,
       row='ou_quartic_two_dim', m1='unchanged',
       notes="chain-simplex hot path (P5/P6 timing); ('dx','dx')"),
    _E('ou_quartic_two_dim-k2-l2-xy', model='ou_quartic_two_dim', k=2,
       max_ell=2, ext=[('dx', 1), ('dy', 1)], params=None, tau_grid=T_OU,
       row='ou_quartic_two_dim', m1='unchanged', notes="('dx','dy')"),
    _E('ou_quartic_colored-k2-l1', model='ou_quartic_colored', k=2, max_ell=1,
       ext=[('dx', 1), ('dx', 1)], params=P_OU_COLORED, tau_grid=[0.0, 1.0, 4.0],
       row='ou_quartic_colored', m1='unchanged',
       notes='NoiseSource τ rows, TAU_KERNEL_CAP box, SR branch'),
    _E('ou_quartic_two_dim_color_corr-k2-l1',
       model='ou_quartic_two_dim_color_corr', k=2, max_ell=1,
       ext=[('dx', 1), ('dy', 1)], params=P_OU_2D_CC, tau_grid=[0.0, 1.0, 4.0],
       row='ou_quartic_two_dim_color_corr', m1='unchanged'),

    # ── spike-reset family: live Θ(0) bug, P3 inheritance ──
    _E('spike_reset-k2-l1', model='single_population_spike_reset_test', k=2,
       max_ell=1, ext=[('n', 1), ('n', 2)], params=P_SPIKE, tau_grid=T_SPIKE,
       row='single_population_spike_reset_test', m1='moves'),
    _E('spike_reset-k2-l1-grouped', model='single_population_spike_reset_test',
       k=2, max_ell=1, ext=[('n', 1), ('n', 2)], params=P_SPIKE,
       tau_grid=T_SPIKE, grouped=True,
       row='single_population_spike_reset_test', m1='moves'),
    _E('spike_reset-k1-l1', model='single_population_spike_reset_test', k=1,
       max_ell=1, ext=[('n', 1)], params=P_SPIKE, tau_grid=[0.0],
       row='single_population_spike_reset_test', m1='unchanged'),
    _E('quad_spike_reset-k2-l1', model='single_population_quad_spike_reset_test',
       k=2, max_ell=1, ext=[('n', 1), ('n', 2)], params=P_QSPIKE, tau_grid=T3,
       row='single_population_quad_spike_reset_test', m1='moves'),

    # ── Θ(0) rows present but values expected exactly 0 ──
    _E('quad_exp-k2-l1', model='single_population_quad_exp_test', k=2,
       max_ell=1, ext=[('n', 1), ('n', 2)], params=P_SP, tau_grid=T3,
       timeout=LONG_TIMEOUT, row='single_population_quad_exp_test',
       m1='unchanged', notes='355 s in the M0.8 scan under load'),

    # ── no m=2 polygons; must be unchanged ──
    _E('linear_hawkes-k2-l1', model='linear_hawkes', k=2, max_ell=1,
       ext=[('n', 1), ('n', 2)], params=P_LINH, tau_grid=T3,
       row='linear_hawkes', m1='unknown'),
    _E('multipopulation_test-k2-l1', model='multipopulation_test', k=2,
       max_ell=1, ext=[('nE', 1), ('nE', 2)], params=P_MP, tau_grid=T3,
       row='multipopulation_test', m1='unchanged'),
    _E('linear_delta_spikes-k2-l1',
       model='single_population_linear_delta_spikes_test', k=2, max_ell=1,
       ext=[('n', 1), ('n', 2)], params=P_LD, tau_grid=T3,
       row='single_population_linear_delta_spikes_test', m1='unchanged'),

    # ── previously timed out at 45 s; 15-min budget each ──
    _E('multipopulation_spike_reset-k2-l1',
       model='multipopulation_spike_reset_test', k=2, max_ell=1,
       ext=[('nE', 1), ('nE', 2)], params=P_MPSR, tau_grid=T3,
       timeout=LONG_TIMEOUT, row='multipopulation_spike_reset_test'),
    _E('cubic_alpha-k2-l1', model='single_population_cubic_alpha_test', k=2,
       max_ell=1, ext=[('n', 1), ('n', 2)], params=P_SP, tau_grid=T3,
       timeout=LONG_TIMEOUT, row='single_population_cubic_alpha_test'),
    _E('conductance-k2-l1', model='single_population_conductance_test', k=2,
       max_ell=1, ext=[('n', 1), ('n', 1)], params=P_COND, tau_grid=T3,
       timeout=LONG_TIMEOUT, row='single_population_conductance_test'),

    # ── notebook δ-models (reduced τ grid) and ConvVertex models ──
    _E('dendritic_quad_soma_sigmoid-k2-l1', model='dendritic_quad_soma_sigmoid',
       k=2, max_ell=1, ext=[('dnS', 1), ('dnS', 2)], params=P_DNL,
       tau_grid=[0.0, 2.5], timeout=LONG_TIMEOUT,
       row='notebook delta-model', notes='timed out at 900 s (3 τ) in M0.8'),
    _E('quadratic_hawkes_alpha-k2-l1', model='quadratic_hawkes_alpha', k=2,
       max_ell=1, ext=[('dn', 1), ('dn', 2)], params=P_QHA, tau_grid=T3,
       timeout=LONG_TIMEOUT, row='notebook delta-model / ConvVertex',
       notes='timed out at 900 s in M0.8'),
    _E('quadratic_hawkes-k2-l1', model='quadratic_hawkes', k=2, max_ell=1,
       ext=[('n', 1), ('n', 2)], params=P_QH, tau_grid=T3,
       timeout=LONG_TIMEOUT, row='quadratic_hawkes (ConvVertex)',
       notes='earlier scan: MF assertion or setup > 500 s'),

    # ── spatial: certify_modes at q in {0, 0.7, 1.5} + totals ──
    _E('spatial-allen_cahn_1d-l2', model='allen_cahn_1d_subcritical_infinite',
       kind='spatial', k=2, max_ell=2, ext=[('dphi', 1), ('dphi', 1)],
       params={'mu': 1.0, 'D': 1.0, 'lam': 0.1, 'T': 1.0},
       tau_grid=[0.0, 0.5], chi_grid=[0.0, 1.0], row='spatial'),
    _E('spatial-reaction_diffusion_2d-l2', model='reaction_diffusion_2d',
       kind='spatial', k=2, max_ell=2, ext=[('dphi', 1), ('dphi', 1)],
       params={'mu': 1.0, 'D': 1.0, 'g': 0.2, 'T': 1.0},
       tau_grid=[0.0, 0.5], chi_grid=[0.4, 1.0], row='spatial'),
    _E('spatial-coupled_rd_2species_1d-l2', model='coupled_rd_2species_1d',
       kind='spatial', k=2, max_ell=2, ext=[('da', 1), ('da', 1)],
       params={'mua': 1.5, 'mub': 1.2, 'Da': 0.8, 'Db': 0.8, 'g': 0.4,
               'h': 0.3, 'ga': 0.3, 'gb': 0.3, 'Ta': 1.0, 'Tb': 0.7},
       tau_grid=[0.0, 0.5], chi_grid=[0.0, 1.0], row='spatial'),
    _E('spatial-reaction_diffusion_conserved_1d-l1',
       model='reaction_diffusion_conserved_1d', kind='spatial', k=2,
       max_ell=1, ext=[('dphi', 1), ('dphi', 1)],
       params={'mu': 1.0, 'D': 2.0, 'g': 0.3, 'T': 1.0},
       tau_grid=[0.0, 0.5], chi_grid=[0.0, 1.0], row='spatial',
       notes='conserved derivative vertex (Σ(q→0) ∝ q²)'),
    _E('spatial-edwards_wilkinson_1d-massless-l0', model='edwards_wilkinson_1d',
       kind='spatial', k=2, max_ell=0, ext=[('dphi', 1), ('dphi', 1)],
       params={'mu': 0.0, 'D': 2.0, 'T': 1.0},
       tau_grid=[0.0, 0.5], chi_grid=[0.0, 1.0], row='spatial (massless)',
       notes='mu=0: λ=0 at q=0 (massless probe).  EW is free: ℓ≥1 has no '
             'diagrams (SpatialPropagatorError), so tree + certify only'),
    _E('spatial-edwards_wilkinson_1d-l0', model='edwards_wilkinson_1d',
       kind='spatial', k=2, max_ell=0, ext=[('dphi', 1), ('dphi', 1)],
       params={'mu': 0.5, 'D': 2.0, 'T': 1.0},
       tau_grid=[0.0, 0.5], chi_grid=[0.0, 1.0], row='spatial',
       notes='massive reference for the massless probe (free model: tree)'),
    _E('spatial-reaction_diffusion_conserved_1d-massless-l1',
       model='reaction_diffusion_conserved_1d', kind='spatial', k=2,
       max_ell=1, ext=[('dphi', 1), ('dphi', 1)],
       params={'mu': 0.0, 'D': 2.0, 'g': 0.3, 'T': 1.0},
       tau_grid=[0.0, 0.5], chi_grid=[0.0, 1.0], row='spatial (massless)',
       notes='mu=0: interacting massless probe (λ=0 at q=0)'),
    # ℓ=1 companions of the spatial ℓ=2 entries (the ℓ=2 runs exceed 10 min)
    _E('spatial-reaction_diffusion_2d-l1', model='reaction_diffusion_2d',
       kind='spatial', k=2, max_ell=1, ext=[('dphi', 1), ('dphi', 1)],
       params={'mu': 1.0, 'D': 1.0, 'g': 0.2, 'T': 1.0},
       tau_grid=[0.0, 0.5], chi_grid=[0.4, 1.0], row='spatial'),
    _E('spatial-coupled_rd_2species_1d-l1', model='coupled_rd_2species_1d',
       kind='spatial', k=2, max_ell=1, ext=[('da', 1), ('da', 1)],
       params={'mua': 1.5, 'mub': 1.2, 'Da': 0.8, 'Db': 0.8, 'g': 0.4,
               'h': 0.3, 'ga': 0.3, 'gb': 0.3, 'Ta': 1.0, 'Tb': 0.7},
       tau_grid=[0.0, 0.5], chi_grid=[0.0, 1.0], row='spatial'),
]


def _census(name, timeout=DEFAULT_TIMEOUT):
    """nquad-stubbed census twin of a (slow) zoo entry: per-m nquad counts
    and fallback reasons.  Values are NOT baselines."""
    e = copy.deepcopy(next(x for x in ZOO if x['name'] == name))
    e.update(name=name + '-nquadstub', stub_nquad=True, timeout=float(timeout),
             notes='CENSUS ONLY (nquad stubbed -> 0) of ' + name)
    return e


# Censuses of the temporal entries that exceed their budget unstubbed.
ZOO += [_census(n) for n in (
    'multipopulation_spike_reset-k2-l1', 'cubic_alpha-k2-l1',
    'dendritic_quad_soma_sigmoid-k2-l1', 'quadratic_hawkes_alpha-k2-l1')]


def zoo_extra_files():
    """The extra (local-only) zoo files in effect: ``DAEDALUS_ZOO_EXTRA``
    (``os.pathsep``-separated; every listed file must exist) when set, else
    the ``DEFAULT_ZOO_EXTRA`` files that exist."""
    raw = os.environ.get(ZOO_EXTRA_ENV)
    if raw is None:
        return [p for p in DEFAULT_ZOO_EXTRA if os.path.exists(p)]
    paths = [os.path.abspath(os.path.expanduser(p))
             for p in raw.split(os.pathsep) if p.strip()]
    missing = [p for p in paths if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(f'{ZOO_EXTRA_ENV}: no such file(s): {missing}')
    return paths


def _load_extra_entries():
    """Entries of every extra zoo file (``ZOO_EXTRA`` or ``LOCAL_ZOO``), all
    flagged ``local_only``: they never reach a tracked output."""
    out = []
    for i, path in enumerate(zoo_extra_files()):
        spec = importlib.util.spec_from_file_location(
            f'_phase_j_zoo_extra_{i}', path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        specs = getattr(mod, 'ZOO_EXTRA', None)
        if specs is None:
            specs = getattr(mod, 'LOCAL_ZOO', [])
        for d in specs:
            e = _E(**d)
            e['local_only'] = True
            out.append(e)
    return out


def all_entries(include_local=True):
    """The zoo; with ``include_local`` also the extra (local-only) entries."""
    ents = list(ZOO) + (_load_extra_entries() if include_local else [])
    names = [e['name'] for e in ents]
    dup = {n for n in names if names.count(n) > 1}
    if dup:
        raise ValueError(f'duplicate zoo entry names: {sorted(dup)}')
    return ents


def entry_by_name(name, include_local=True):
    for e in all_entries(include_local):
        if e['name'] == name:
            return e
    raise KeyError(f'no zoo entry {name!r}')


# ═══════════════════════════════════════════════════════════════════════
# Provenance helpers
# ═══════════════════════════════════════════════════════════════════════

def _sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def _git(*args):
    try:
        r = subprocess.run(['git', *args], cwd=_REPO_ROOT, capture_output=True,
                           text=True, timeout=60)
        return r.returncode, r.stdout.rstrip('\n')   # keep porcelain columns
    except Exception as exc:                      # noqa: BLE001
        return -1, f'{type(exc).__name__}: {exc}'


def model_path(entry):
    """Absolute path of the file that defines the entry's model."""
    if entry.get('model_file'):
        return os.path.join(_REPO_ROOT, entry['model_file'].partition(':')[0])
    return os.path.join(_REPO_ROOT, 'models', f"{entry['model']}.model.py")


def git_route(path):
    """'tracked' if the file is in the index, 'new' if it exists and is not
    git-ignored, 'ignored' if git-ignored, 'missing' if absent."""
    if not os.path.exists(path):
        return 'missing'
    rel = os.path.relpath(path, _REPO_ROOT)
    rc, _ = _git('ls-files', '--error-unmatch', rel)
    if rc == 0:
        return 'tracked'
    rc, _ = _git('check-ignore', '-q', rel)
    return 'ignored' if rc == 0 else 'new'


def destination(entry):
    """'tracked' (tests/fixtures) or 'local' (scratch) for this entry."""
    if entry.get('local_only'):
        return 'local'
    return ('tracked' if git_route(model_path(entry)) in ('tracked', 'new')
            else 'local')


def code_provenance():
    rc, head = _git('rev-parse', 'HEAD')
    _, branch = _git('rev-parse', '--abbrev-ref', 'HEAD')
    _, dirty = _git('status', '--porcelain', '--untracked-files=no')
    fi = os.path.join(_REPO_ROOT, 'engine', 'integration', 'time_domain',
                      'final_integral.py')
    gi = os.path.join(_REPO_ROOT, 'engine', 'integration', 'time_domain',
                      'grouped_integral.py')
    return {
        'git_head': head.strip() if rc == 0 else None,
        'git_branch': branch.strip(),
        'git_dirty_tracked': [ln for ln in dirty.splitlines() if ln.strip()],
        'sha256_final_integral': _sha256(fi),
        'sha256_grouped_integral': _sha256(gi),
    }


def _library_versions():
    out = {'python': platform.python_version(), 'platform': platform.platform()}
    for mod in ('numpy', 'scipy', 'sympy', 'mpmath', 'numba'):
        try:
            out[mod] = __import__(mod).__version__
        except Exception:                          # noqa: BLE001
            out[mod] = None
    try:
        from sage.version import version as sv
        out['sage'] = sv
    except Exception:                              # noqa: BLE001
        out['sage'] = None
    return out


def _numeric_flags():
    import engine.integration.time_domain.final_integral as FI
    names = ('USE_POLYGON_M2_INTEGRATOR', 'POLYGON_BBOX_CAP',
             'POSET_PHYSICAL_MARGIN', 'USE_1D_INTEGRATOR',
             'USE_NUMBA_CHAIN_SIMPLEX', 'USE_CHAIN_SIMPLEX_PRECISION_FIX',
             'USE_POSET_INTEGRATOR', 'USE_POSET_CAP_MATCH_SCIPY',
             'USE_POSET_MPMATH_ACCUMULATION', 'TAU_KERNEL_CAP', 'QUAD_OPTS',
             '_HAVE_NUMBA',
             # M1 call-time flag ('<absent>' in pre-M1 records)
             'THETA0_CONST_ROW_MODE',
             # M2a call-time flag ('<absent>' in pre-M2a records)
             'STRUCTURAL_ZEROS',
             # M2b call-time flag and the hardened fallback's tolerances
             # ('<absent>' in pre-M2b records)
             'NQUAD_HARDENED', 'NQUAD_EPSABS_FACTOR', 'NQUAD_EPSREL',
             'NQUAD_LIMIT', 'NQUAD_TAIL_K', 'NQUAD_UNCERTIFIED_CAP',
             # M3 call-time flag ('<absent>' in pre-M3 records)
             'USE_DBM_FALLBACK')
    out = {}
    for n in names:
        v = getattr(FI, n, '<absent>')
        out[n] = v if isinstance(v, (bool, int, float, str, type(None))) else (
            {k: (vv if isinstance(vv, (bool, int, float, str)) else repr(vv))
             for k, vv in v.items()} if isinstance(v, dict) else repr(v))
    out['_SUBSET_HOOK_is_None'] = getattr(FI, '_SUBSET_HOOK', None) is None
    return out


def _env_knobs():
    # ZOO_EXTRA_ENV is a (local) file path, not a numeric knob: never recorded.
    return {k: v for k, v in sorted(os.environ.items())
            if (k.startswith(('DAEDALUS_', 'SPATIAL_', 'PHASE_J_', 'OMP_',
                              'OPENBLAS_', 'MKL_', 'NUMBA_'))
                or k == 'PYTHONHASHSEED') and k != ZOO_EXTRA_ENV}


# ═══════════════════════════════════════════════════════════════════════
# Encoding (exact float round-trip through JSON: repr is shortest-exact)
# ═══════════════════════════════════════════════════════════════════════

def _enc_c(arr):
    import numpy as np
    a = np.asarray(arr, dtype=complex).ravel()
    return {'shape': list(np.shape(arr)),
            're': [float(z.real) for z in a], 'im': [float(z.imag) for z in a]}


def _dec_c(d):
    import numpy as np
    re = np.asarray(d['re'], dtype=float)
    im = np.asarray(d['im'], dtype=float)
    return (re + 1j * im).reshape(d['shape'])


def _jsonable(x):
    import numpy as np
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if isinstance(x, np.ndarray):
        return _jsonable(x.tolist())
    if isinstance(x, (bool, int, float, str, type(None))):
        return x
    try:
        return float(x)
    except Exception:                              # noqa: BLE001
        return repr(x)


# ═══════════════════════════════════════════════════════════════════════
# In-process evaluation of one entry
# ═══════════════════════════════════════════════════════════════════════

def _load_model(entry):
    """(model dict, module or None, source description)."""
    if entry.get('model_file'):
        rel, _, fn = entry['model_file'].partition(':')
        path = os.path.join(_REPO_ROOT, rel)
        spec = importlib.util.spec_from_file_location(
            '_zoo_model_' + os.path.basename(path).replace('.', '_'), path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return getattr(mod, fn or 'build')(), mod, entry['model_file']
    import daedalus as dd
    model, mod = dd.load_model(entry['model'])
    return model, mod, f"models/{entry['model']}.model.py"


def resolve_params(entry, model, module):
    """The explicit parameter dict the entry runs with (recorded, so a later
    change of a model's defaults is detected rather than silently absorbed)."""
    p = entry.get('params')
    if isinstance(p, dict) and '__from_module__' in p:
        # A parameter set of the model's own module (extra zoo files), so
        # the spec never duplicates the numbers.
        attr, key = p['__from_module__']
        return copy.deepcopy(dict(getattr(module, attr)[key]))
    if p:
        return copy.deepcopy(p)
    dflt = getattr(module, 'DEFAULT_FUNDAMENTAL', None) if module else None
    if dflt:
        return copy.deepcopy(dflt)
    return {s['name']: s['default'] for s in (model.get('parameters') or [])
            if s.get('default') is not None}


class _NquadStub:
    """Counting stand-in for ``_integrate_polytope`` (m>=1 -> 0j; m=0 passes
    through).  Installed in final_integral AND grouped_integral."""

    def __init__(self):
        self.calls = {}
        self._orig = None

    def __call__(self, integrand_callable, s_constraints, free_ext_vals, m,
                 **kw):
        # ``**kw``: the M1 keywords (``raw_rows``, ``row_kinds``) are passed
        # through to the real m=0 evaluation.
        if m == 0:
            return self._orig(integrand_callable, s_constraints,
                              free_ext_vals, m, **kw)
        self.calls[m] = self.calls.get(m, 0) + 1
        return 0.0 + 0.0j

    @contextlib.contextmanager
    def installed(self):
        import engine.integration.time_domain.final_integral as FI
        import engine.integration.time_domain.grouped_integral as GI
        self._orig = FI._integrate_polytope
        saved = (FI._integrate_polytope, GI._integrate_polytope)
        FI._integrate_polytope = self
        GI._integrate_polytope = self
        try:
            yield self
        finally:
            FI._integrate_polytope, GI._integrate_polytope = saved


def migrate_counters(c, semantics):
    """Return a copy of the counter dict ``c`` in ``COUNTER_SEMANTICS`` form.

    ``semantics`` is the counter semantics ``c`` was recorded under (1 when a
    record predates the marker).  Exact: under semantics 1, ``nquad_calls``
    minus the per-m scipy counters is the number of m=0 entries.  Empty /
    ``None`` counters pass through.
    """
    if not c:
        return c
    c = copy.deepcopy(c)
    if (semantics or 1) >= COUNTER_SEMANTICS:
        return c
    n_mge1 = sum(int(c.get(k) or 0) for k in (
        'scipy_nquad_called_m1', 'scipy_nquad_called_m2',
        'scipy_nquad_called_mge3'))
    n_all = c.get('nquad_calls')
    if n_all is not None:
        m0 = int(n_all) - n_mge1
        if m0 < 0:
            raise ValueError(f'inconsistent semantics-1 counters: nquad_calls='
                             f'{n_all} < scipy m>=1 total {n_mge1}')
        c['nquad_calls'] = n_mge1
        c['polytope_m0_direct'] = m0
    return c


def _api_nudged(pt):
    """The per-leg mapping of ``daedalus._args`` (the full k-point grid): a
    non-anchor leg with |t| <= 1e-12 is moved to -1e-6.  (Not the ``dd.run``
    slice placement, ``daedalus._kpoint_slice_times``; see the module
    docstring.)"""
    return [pt[0]] + [t if abs(t) > 1e-12 else ITO_NUDGE for t in pt[1:]]


def run_entry(entry, *, with_provenance=True):
    """Run one zoo entry IN THIS PROCESS and return a JSON-able result dict.

    ``parallel=False`` always.  Never call this for a config that runs for
    hours in scipy.nquad (fast poles at ℓ=2).
    """
    import numpy as np
    import engine.integration.time_domain.final_integral as FI
    from api import compute_cumulants
    from api.compute import _ITO_EPS

    res_out = {
        'name': entry['name'], 'entry': copy.deepcopy(entry),
        'schema_version': SCHEMA_VERSION,
        'started': datetime.datetime.now().isoformat(timespec='seconds'),
        'pythonhashseed': os.environ.get('PYTHONHASHSEED'),
        'hash_randomization_flag': int(sys.flags.hash_randomization),
        'ito_eps': float(_ITO_EPS),
        'counter_semantics': COUNTER_SEMANTICS,
    }
    mpath = model_path(entry)
    res_out['model_path'] = os.path.relpath(mpath, _REPO_ROOT)
    if not os.path.exists(mpath):
        res_out['status'] = 'MISSING_MODEL'
        return res_out
    res_out['model_sha256'] = _sha256(mpath)
    if with_provenance:
        res_out['model_git'] = git_route(mpath)
        res_out['flags'] = _numeric_flags()
        res_out['versions'] = _library_versions()
        res_out['env'] = _env_knobs()

    try:
        model, module, src = _load_model(entry)
    except Exception as exc:                       # noqa: BLE001
        res_out['status'] = f'ERR load: {type(exc).__name__}: {exc}'[:500]
        return res_out
    params = resolve_params(entry, model, module)
    res_out['params_effective'] = _jsonable(params)
    # ModelBuilder text fingerprint (action, equations, kernels, fields, ...):
    # identifies the MODEL independently of edits elsewhere in its file.
    res_out['model_spec_signature'] = model.get('spec_signature')
    try:
        from api._expand_cache import cache_dir
        res_out['cache_dir_existed'] = os.path.isdir(
            os.path.join(_REPO_ROOT, cache_dir(model)))
    except Exception:                              # noqa: BLE001
        res_out['cache_dir_existed'] = None

    ext = [tuple(e) for e in entry['ext']]
    kw = dict(k=entry['k'], max_ell=entry['max_ell'], external_fields=ext,
              parameters=params, parallel=False, verbose=False,
              use_cache=True, use_grouped_phase_j=entry['grouped'])
    if entry['kind'] == 'spatial':
        kw['tau_grid'] = np.asarray(entry['tau_grid'], dtype=float)
        kw['chi_grid'] = np.asarray(entry['chi_grid'], dtype=float)
        kw['spatial_parallel'] = False
        if entry.get('spatial_n_q'):
            kw['spatial_n_q'] = int(entry['spatial_n_q'])
    elif entry['k'] in (1, 2):
        kw['tau_grid'] = np.asarray(entry['tau_grid'], dtype=float)
    else:
        kw['tau_grid'] = np.asarray([0.0], dtype=float)   # unused for k>=3

    # Spatial: capture the production certify_modes arguments.
    captured = []
    pb = None
    if entry['kind'] == 'spatial':
        import engine.integration.spatial.pipeline_bridge as pb
        orig_certify = pb.certify_modes

        def _cap(modes, prop, records, external_fields, base_np_sr,
                 q_samples, tau_samples, k=2):
            captured.append(dict(modes=modes, prop=prop, records=records,
                                 external_fields=external_fields,
                                 base_np_sr=base_np_sr,
                                 q_samples=tuple(q_samples),
                                 tau_samples=tuple(tau_samples), k=k))
            return orig_certify(modes, prop, records, external_fields,
                                base_np_sr, q_samples, tau_samples, k=k)
        pb.certify_modes = _cap

    stub = _NquadStub() if entry.get('stub_nquad') else None
    from tests.tools.phase_j_subset_diff import recorded_diagram_sources
    FI._reset_runtime_counters()
    t0 = time.perf_counter()
    try:
        with (stub.installed() if stub else contextlib.nullcontext()), \
                recorded_diagram_sources() as sources:
            res_out['diagram_sources'] = sources      # filled during the run
            res_out['cwd'] = _scrub(os.getcwd())
            res = compute_cumulants(model, **kw)
            wall = time.perf_counter() - t0
            counters_compute = copy.deepcopy(FI._RUNTIME_COUNTERS)
            t1 = time.perf_counter()
            values = _collect_values(entry, res, np)
            eval_wall = time.perf_counter() - t1
            # k>=3: the Phase J integrals run HERE (point evaluation), so the
            # recorded counters are taken after it.
            counters = copy.deepcopy(FI._RUNTIME_COUNTERS)
    except Exception as exc:                       # noqa: BLE001
        import traceback
        res_out['status'] = f'ERR: {type(exc).__name__}: {exc}'[:800]
        res_out['traceback_tail'] = traceback.format_exc()[-3000:]
        res_out['wall_compute'] = time.perf_counter() - t0
        res_out['counters'] = _jsonable(copy.deepcopy(FI._RUNTIME_COUNTERS))
        if pb is not None:
            # A failed spatial run still reports the per-q certification on
            # whatever certify_modes arguments were captured before the error.
            res_out['spatial'] = _spatial_certify({}, captured, orig_certify)
        return res_out
    finally:
        if pb is not None:
            pb.certify_modes = orig_certify

    res_out['status'] = 'census_only' if stub else 'ok'
    res_out['census_only'] = bool(stub)
    res_out['stub_calls'] = ({str(m): n for m, n in sorted(stub.calls.items())}
                             if stub else None)
    res_out['wall_compute'] = wall
    res_out['wall_eval_points'] = eval_wall
    res_out['counters'] = _jsonable(counters)
    res_out['counters_compute_only'] = _jsonable(counters_compute)
    res_out['values'] = values
    if entry['kind'] == 'spatial':
        res_out['spatial'] = _spatial_certify(res.get('spatial_info') or {},
                                              captured, orig_certify)
    return res_out


def _collect_values(entry, res, np):
    out = {}
    k = entry['k']
    if entry['kind'] == 'spatial':
        out['tau_grid'] = [float(t) for t in res['tau_grid']]
        out['chi_grid'] = [float(x) for x in res['spatial_grid']]
        out['total'] = _enc_c(np.asarray(res['C_tau_x'], dtype=complex))
        cum = res.get('C_tau_x_by_order') or {}
        out['cumulative_by_order'] = {
            str(int(o)): _enc_c(np.asarray(cum[o], dtype=complex))
            for o in sorted(cum)}
        return out
    if k in (1, 2):
        out['tau_grid'] = [float(t) for t in entry['tau_grid']]
        by = res.get('C_tau_by_ell') or {}
        out['by_ell'] = {str(int(e)): _enc_c(np.asarray(by[e], dtype=complex))
                         for e in sorted(by) if by[e] is not None}
        out['total'] = _enc_c(np.asarray(res['C_tau'], dtype=complex))
        return out
    # k >= 3: explicit points, raw per-ell callables (+ API-nudged variant)
    pts = entry['points']
    out['points'] = pts
    tcb = res.get('total_C_by_ell') or {}
    raw, nudged = {}, {}
    for e in sorted(tcb):
        fn = tcb[e]
        rv, nv = [], []
        for pt in pts:
            v = complex(fn(*pt))
            rv.append(v)
            npt = _api_nudged(pt)
            nv.append(v if npt == list(pt) else complex(fn(*npt)))
        raw[str(int(e))] = _enc_c(np.asarray(rv))
        nudged[str(int(e))] = _enc_c(np.asarray(nv))
    out['by_ell'] = raw
    out['by_ell_api_nudged'] = nudged
    out['points_api_nudged'] = [_api_nudged(p) for p in pts]
    tot = np.zeros(len(pts), dtype=complex)
    totn = np.zeros(len(pts), dtype=complex)
    for e in raw:
        tot += _dec_c(raw[e])
        totn += _dec_c(nudged[e])
    out['total'] = _enc_c(tot)
    out['total_api_nudged'] = _enc_c(totn)
    return out


def _spatial_certify(info, captured, certify_fn):
    """Production certification result (``info`` = the run's spatial_info)
    + per-q residuals, recomputed with ``certify_fn`` (the original
    ``pipeline_bridge.certify_modes``) on the captured arguments of the FIRST
    certify_modes call (the tree-mode certification)."""
    out = {
        'pipeline_certified': info.get('pipeline_certified'),
        'certify_max_rel': (None if info.get('certify_max_rel') is None
                            else float(info['certify_max_rel'])),
        'certify_tol': 1e-8,
        'n_certify_calls': len(captured),
        'coupled': info.get('coupled'),
        'per_q': {},
    }
    if not captured:
        out['per_q_note'] = 'certify_modes not called (coupled driver / n/a)'
        return out
    c = captured[0]
    out['q_samples'] = list(c['q_samples'])
    out['tau_samples'] = list(c['tau_samples'])
    for q in SPATIAL_CERTIFY_Q:
        try:
            r = certify_fn(c['modes'], c['prop'], c['records'],
                           c['external_fields'], c['base_np_sr'],
                           (q,), c['tau_samples'], k=c['k'])
            out['per_q'][repr(q)] = {'residual': float(r),
                                     'pass': bool(r <= 1e-8)}
        except Exception as exc:                   # noqa: BLE001
            out['per_q'][repr(q)] = {'error': f'{type(exc).__name__}: {exc}'[:300]}
    return out


# ═══════════════════════════════════════════════════════════════════════
# Subprocess driver
# ═══════════════════════════════════════════════════════════════════════

_LOCK = threading.Lock()


def _write_json_atomic(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + '.tmp'
    with open(tmp, 'w') as fh:
        json.dump(obj, fh, indent=1, allow_nan=True)
    os.replace(tmp, path)


def _run_child(entry, out_dir, log):
    out = os.path.join(out_dir, f"{entry['name']}.json")
    env = dict(os.environ, PYTHONHASHSEED='0')
    cmd = [sys.executable, '-m', 'tests.tools.phase_j_zoo_baseline',
           '--child', entry['name'], '--child-out', out]
    t0 = time.time()
    started = datetime.datetime.now().isoformat(timespec='seconds')
    try:
        r = subprocess.run(cmd, cwd=_REPO_ROOT, env=env, capture_output=True,
                           text=True, timeout=entry['timeout'])
        if os.path.exists(out):
            with open(out) as fh:
                rec = json.load(fh)
        else:
            rec = {'name': entry['name'], 'entry': entry,
                   'status': f'NO-RESULT rc={r.returncode}'}
        if r.returncode != 0 or not str(rec.get('status', '')).startswith(
                ('ok', 'census_only')):
            rec['stderr_tail'] = (r.stderr or '')[-3000:]
    except subprocess.TimeoutExpired:
        rec = {'name': entry['name'], 'entry': entry, 'status': 'TIMEOUT',
               'timeout': entry['timeout']}
    rec['wall_process'] = time.time() - t0
    rec['process_started'] = started
    try:
        rec['loadavg_at_start'] = list(os.getloadavg())
    except OSError:
        pass
    _write_json_atomic(out, rec)
    with _LOCK:
        msg = (f"[{time.strftime('%H:%M:%S')}] {entry['name']:45s} "
               f"{str(rec.get('status'))[:60]:60s} "
               f"wall={rec['wall_process']:.1f}s")
        print(msg, flush=True)
        if log:
            with open(log, 'a') as fh:
                fh.write(msg + '\n')
    return rec


def run_many(names, *, workers=3, out_dir=WORK_DIR, log=None):
    """Run the named entries in subprocesses.  Entries sharing a model run
    sequentially in one worker (one writer per saved_models/<model> cache)."""
    ents = [entry_by_name(n) for n in names]
    groups = {}
    for e in ents:
        key = e.get('lane') or e['model'] or e['model_file']
        groups.setdefault(key, []).append(e)
    order = sorted(groups.values(),
                   key=lambda g: -sum(e['timeout'] for e in g))

    def _job(group):
        return [_run_child(e, out_dir, log) for e in group]

    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as ex:
        results = [r for rs in ex.map(_job, order) for r in rs]
    return results


# ═══════════════════════════════════════════════════════════════════════
# Assembly: run JSONs -> npz + human-readable json
# ═══════════════════════════════════════════════════════════════════════

def _scrub(text):
    """Strip local absolute paths (repo root, home dir) from free text that
    goes into a tracked file."""
    if not isinstance(text, str):
        return text
    return (text.replace(_REPO_ROOT, '<repo>')
            .replace(os.path.expanduser('~'), '~'))


def _summarise(rec):
    import numpy as np
    s = {k: rec.get(k) for k in (
        'status', 'wall_compute', 'wall_eval_points', 'wall_process',
        'census_only', 'stub_calls', 'model_path', 'model_sha256',
        'model_spec_signature', 'model_git', 'params_effective',
        'cache_dir_existed', 'diagram_sources', 'cwd',
        'pythonhashseed', 'loadavg_at_start', 'process_started', 'timeout')
        if k in rec}
    e = rec.get('entry') or {}
    s['config'] = {k: e.get(k) for k in (
        'model', 'model_file', 'kind', 'k', 'max_ell', 'ext', 'tau_grid',
        'points', 'chi_grid', 'grouped', 'stub_nquad', 'timeout')}
    s['row'] = e.get('row')
    s['m1_expected'] = e.get('m1')
    s['notes'] = e.get('notes')
    c = migrate_counters(rec.get('counters') or {},
                         rec.get('counter_semantics', 1))
    if c:
        s['nquad'] = {
            'nquad_calls': c.get('nquad_calls'),
            'polytope_m0_direct': c.get('polytope_m0_direct'),
            'scipy_nquad_called_m1': c.get('scipy_nquad_called_m1'),
            'scipy_nquad_called_m2': c.get('scipy_nquad_called_m2'),
            'scipy_nquad_called_mge3': c.get('scipy_nquad_called_mge3'),
            'nquad_fallback_by_reason': c.get('nquad_fallback_by_reason'),
            'zero_normal_rows_seen': c.get('zero_normal_rows_seen'),
            'polygon_attempted': c.get('polygon_attempted'),
            'poset_attempted': c.get('poset_attempted'),
        }
        s['counters'] = c
    v = rec.get('values')
    if v:
        def fmt(d):
            a = _dec_c(d).ravel()
            return [f'{z.real:.15g}' + (f'{z.imag:+.3g}j' if z.imag else '')
                    for z in a]
        vs = {}
        if 'by_ell' in v:
            vs['by_ell'] = {e: fmt(d) for e, d in v['by_ell'].items()}
        if 'cumulative_by_order' in v:
            vs['cumulative_by_order'] = {o: fmt(d) for o, d in
                                         v['cumulative_by_order'].items()}
        vs['total'] = fmt(v['total'])
        if 'total_api_nudged' in v:
            vs['total_api_nudged'] = fmt(v['total_api_nudged'])
        for key in ('tau_grid', 'points', 'chi_grid'):
            if key in v:
                vs[key] = v[key]
        s['values'] = vs
        s['max_abs_total'] = float(np.max(np.abs(_dec_c(v['total']))))
    if rec.get('spatial'):
        s['spatial'] = rec['spatial']
    if rec.get('stderr_tail') and not str(rec.get('status', '')).startswith(
            ('ok', 'census_only')):
        s['stderr_tail'] = _scrub(rec['stderr_tail'])[-1200:]
    if rec.get('traceback_tail'):
        s['traceback_tail'] = _scrub(rec['traceback_tail'])[-1200:]
    s['status'] = _scrub(s.get('status'))
    return s


def reproducibility(rec_a, rec_b):
    """Run-to-run comparison of two results of the same entry."""
    import numpy as np
    out = {'status': [rec_a.get('status', '')[:40], rec_b.get('status', '')[:40]],
           'counters_equal': (
               migrate_counters(rec_a.get('counters'),
                                rec_a.get('counter_semantics', 1))
               == migrate_counters(rec_b.get('counters'),
                                   rec_b.get('counter_semantics', 1))),
           'params_equal': (rec_a.get('params_effective')
                            == rec_b.get('params_effective'))}
    if (rec_a.get('values') and rec_b.get('values')
            and not rec_a.get('census_only')):
        A, B = result_arrays(rec_a), result_arrays(rec_b)
        cmp = compare_values(A, B)
        out['array_equal'] = all(np.array_equal(A[k], B[k]) for k in A
                                 if k in B) and set(A) == set(B)
        out['max_rel'] = max(v[1] for v in cmp.values())
        out['within_rtol_1e-13'] = all(v[2] for v in cmp.values())
    return out


def assemble(run_dir=WORK_DIR, *, tracked_npz=TRACKED_NPZ,
             tracked_json=TRACKED_JSON, local_npz=LOCAL_NPZ,
             local_json=LOCAL_JSON, run_meta=None, compare_dir=None,
             plan_label='M0.3 (pre-M1 values)'):
    """Build the tracked and local npz/json files from the per-entry run JSONs
    in ``run_dir``.  Entries without a run JSON are recorded as NOT_RUN.
    With ``compare_dir`` (an independent run of the same entries), each entry
    also gets a ``reproducibility`` record (array_equal / max_rel / counters)."""
    import numpy as np
    prov = code_provenance()
    meta_common = {
        'schema_version': SCHEMA_VERSION,
        'generated': datetime.datetime.now().isoformat(timespec='seconds'),
        'generator': 'tests/tools/phase_j_zoo_baseline.py',
        'plan': f'docs/integration_speedup_plan.md §4.3 / {plan_label}',
        'tau_convention': ('k=2 grid τ=0 is evaluated at t1=-_ITO_EPS=-1e-6 '
                           '(Itô left limit); k>=3 points are raw, with the '
                           'API-nudged variant alongside'),
        'npz_key_format': ("'<entry>::<array>' complex128/float64 arrays; "
                           "'__meta__' = this JSON as utf-8 bytes (uint8)"),
        'code': prov,
        'run_meta': run_meta or {},
        'counter_semantics': COUNTER_SEMANTICS,
    }
    buckets = {'tracked': ({}, {}), 'local': ({}, {})}
    runtime = {}                   # flags / versions / env, once per run
    for e in all_entries():
        dest = destination(e)
        arrays, summ = buckets[dest]
        path = os.path.join(run_dir, f"{e['name']}.json")
        if not os.path.exists(path):
            summ[e['name']] = {'status': 'NOT_RUN',
                               'config': {k: e.get(k) for k in (
                                   'model', 'model_file', 'k', 'max_ell')}}
            continue
        with open(path) as fh:
            rec = json.load(fh)
        summ[e['name']] = _summarise(rec)
        if rec.get('code'):
            code_at_run = {k: rec['code'].get(k) for k in (
                'git_head', 'sha256_final_integral', 'sha256_grouped_integral')}
            rec = dict(rec, code_at_run=code_at_run)
        for key in ('flags', 'versions', 'env', 'code_at_run'):
            if key in rec:
                if key not in runtime:
                    runtime[key] = rec[key]
                elif rec[key] != runtime[key]:
                    runtime.setdefault('mismatch', []).append(
                        [e['name'], key])
        other = (os.path.join(compare_dir, f"{e['name']}.json")
                 if compare_dir else None)
        if other and os.path.exists(other):
            with open(other) as fh:
                rec_b = json.load(fh)
            summ[e['name']]['reproducibility'] = dict(
                reproducibility(rec, rec_b),
                against=os.path.relpath(other, _REPO_ROOT),
                other_started=rec_b.get('process_started'))
        v = rec.get('values')
        if not v or rec.get('census_only'):
            continue          # census-only (nquad stubbed) values are not baselines
        n = e['name']
        arrays[f'{n}{SEP}total'] = _dec_c(v['total'])
        for ell, d in (v.get('by_ell') or {}).items():
            arrays[f'{n}{SEP}ell{ell}'] = _dec_c(d)
        for ell, d in (v.get('by_ell_api_nudged') or {}).items():
            arrays[f'{n}{SEP}ell{ell}_api_nudged'] = _dec_c(d)
        if 'total_api_nudged' in v:
            arrays[f'{n}{SEP}total_api_nudged'] = _dec_c(v['total_api_nudged'])
        for o, d in (v.get('cumulative_by_order') or {}).items():
            arrays[f'{n}{SEP}order{o}_cumulative'] = _dec_c(d)
        if 'tau_grid' in v:
            arrays[f'{n}{SEP}tau_grid'] = np.asarray(v['tau_grid'], float)
        if 'chi_grid' in v:
            arrays[f'{n}{SEP}chi_grid'] = np.asarray(v['chi_grid'], float)
        if 'points' in v:
            arrays[f'{n}{SEP}points'] = np.asarray(v['points'], float)
    written = []
    for dest, (npz_path, json_path) in (('tracked', (tracked_npz, tracked_json)),
                                        ('local', (local_npz, local_json))):
        arrays, summ = buckets[dest]
        meta = dict(meta_common, destination=dest, runtime=runtime,
                    entries=summ)
        os.makedirs(os.path.dirname(npz_path), exist_ok=True)
        blob = json.dumps(meta, allow_nan=True).encode('utf-8')
        np.savez_compressed(npz_path,
                            __meta__=np.frombuffer(blob, dtype=np.uint8),
                            **arrays)
        _write_json_atomic(json_path, meta)
        written += [npz_path, json_path]
    return written


# ═══════════════════════════════════════════════════════════════════════
# Loading / comparison (used by tests/test_phase_j_legacy_baseline.py)
# ═══════════════════════════════════════════════════════════════════════

def load_baseline(path=TRACKED_NPZ):
    """(meta dict, {entry: {array_name: ndarray}}).  Counters recorded under
    an older ``counter_semantics`` are converted with ``migrate_counters``
    (in memory; the file is not touched)."""
    import numpy as np
    with np.load(path, allow_pickle=False) as z:
        meta = json.loads(z['__meta__'].tobytes().decode('utf-8'))
        arrays = {}
        for key in z.files:
            if key == '__meta__':
                continue
            entry, _, arr = key.partition(SEP)
            arrays.setdefault(entry, {})[arr] = z[key]
    sem = meta.get('counter_semantics', 1)
    if sem < COUNTER_SEMANTICS:
        for s in (meta.get('entries') or {}).values():
            if s.get('counters'):
                s['counters'] = migrate_counters(s['counters'], sem)
            if s.get('nquad'):
                s['nquad'] = migrate_counters(s['nquad'], sem)
        meta['counter_semantics'] = COUNTER_SEMANTICS
    return meta, arrays


def result_arrays(rec):
    """{array_name: ndarray} of a ``run_entry`` result, keyed like the npz."""
    import numpy as np
    v = rec['values']
    out = {'total': _dec_c(v['total'])}
    for ell, d in (v.get('by_ell') or {}).items():
        out[f'ell{ell}'] = _dec_c(d)
    for ell, d in (v.get('by_ell_api_nudged') or {}).items():
        out[f'ell{ell}_api_nudged'] = _dec_c(d)
    if 'total_api_nudged' in v:
        out['total_api_nudged'] = _dec_c(v['total_api_nudged'])
    for o, d in (v.get('cumulative_by_order') or {}).items():
        out[f'order{o}_cumulative'] = _dec_c(d)
    for key in ('tau_grid', 'chi_grid', 'points'):
        if key in v:
            out[key] = np.asarray(v[key], float)
    return out


def compare_values(ref, cur, *, rtol=1e-13, floor_rel=1e-15):
    """{array_name: (max_abs_diff, max_rel_diff, ok)}.  ``ok`` iff
    |cur-ref| <= rtol*|ref| + floor_rel*max|ref| elementwise (plan §4.4:
    rtol 1e-13 absorbs the 1-ulp hash-randomisation jitter; the scale floor
    absorbs exact-zero entries)."""
    import numpy as np
    out = {}
    for name in sorted(set(ref) | set(cur)):
        if name not in ref or name not in cur:
            out[name] = (math.inf, math.inf, False)
            continue
        a, b = np.asarray(ref[name]), np.asarray(cur[name])
        if a.shape != b.shape:
            out[name] = (math.inf, math.inf, False)
            continue
        d = np.abs(b - a)
        scale = float(np.max(np.abs(a))) if a.size else 0.0
        tol = rtol * np.abs(a) + floor_rel * scale
        rel = d / np.maximum(np.abs(a), 1e-300)
        out[name] = (float(d.max()) if d.size else 0.0,
                     float(rel.max()) if rel.size else 0.0,
                     bool(np.all(d <= tol)))
    return out


# ═══════════════════════════════════════════════════════════════════════
# Milestone deltas: a later zoo run vs the pre-change baseline
# ═══════════════════════════════════════════════════════════════════════

#: Counters shown in the delta table (after = the new run).  The ``theta0_*``
#: / ``polygon_zero_area`` / ``poset_empty_*`` counters exist from M1 on.
DELTA_THETA0_KEYS = ('theta0_const_empty', 'theta0_const_drop', 'theta0_tie',
                     'theta0_tie_ordered', 'theta0_subsets_pruned',
                     'polygon_zero_area', 'poset_empty_const',
                     'poset_empty_cycle')
#: The M1 bit-identity invariant.  'ito' can differ from the pre-M1 code only
#: on an evaluation where it decided a constant row with the Θ(0) helper (in
#: ANY path: ``zero_normal_rows_seen`` counts only the polygon and poset
#: paths, so e.g. m=1 / m=0 constant rows at k>=3 ties are missed by it),
#: skipped a τ-independent EMPTY subset, or made an exact structural
#: emptiness decision that needs no constant row (``polygon_zero_area``: an
#: m=2 polygon with an exactly contradictory opposite pair of rows or
#: exactly zero area; ``poset_empty_cycle``: an order cycle whose shifts sum
#: to <= 0; both may replace a rounding-level value, or an nquad value /
#: exception, by an exact 0).  With all of these counters of the NEW run at
#: 0 (and the baseline's ``zero_normal_rows_seen`` at 0) the values must be
#: unchanged.
M1_VALUE_KEYS = ('theta0_const_empty', 'theta0_const_drop',
                 'theta0_subsets_pruned', 'polygon_zero_area',
                 'poset_empty_cycle')
#: The subset of ``M1_VALUE_KEYS`` that can also change ROUTE counters
#: (skipped subsets and answered-empty regions that bailed to nquad before);
#: a constant row decided inside an analytic path does not.
M1_ROUTE_KEYS = ('theta0_subsets_pruned', 'polygon_zero_area',
                 'poset_empty_cycle')
#: The M2a structural zeros (``final_integral.STRUCTURAL_ZEROS``): an
#: ``_integrate_polytope`` region with an order cycle whose shifts sum to <= 0
#: answered 0 (``polytope_empty_cycle``), and δ-subsets never built because
#: they keep an identically-zero edge smooth (``forced_delta_pruned``).  Both
#: skip only exact zeros, so they may change a value only where the pre-M2a
#: code raised, returned NaN, or integrated a rounding-level sliver; they are
#: allowed to move a value in ``delta_table`` like ``M1_VALUE_KEYS``.
M2A_VALUE_KEYS = ('polytope_empty_cycle', 'forced_delta_pruned')
#: The M2a counters that change ROUTE counters: a pruned subset is never
#: evaluated, so its analytic / nquad attempts disappear (a cycle answered
#: at the entry of ``_integrate_polytope`` is still counted there).
M2A_ROUTE_KEYS = ('forced_delta_pruned',)
#: The M2b hardened quadrature fallback (``final_integral.NQUAD_HARDENED``):
#: every region served by it may move at the accuracy of the pre-M2b
#: default-tolerance scipy.nquad (measured up to 6.8e-5 relative per region),
#: so a value may move wherever this counter is nonzero; routes do not
#: change (the same regions reach the fallback).
M2B_VALUE_KEYS = ('nquad_hardened_calls',)
#: M3 (``final_integral.USE_DBM_FALLBACK``): an m>=3 region refused by the
#: poset lower-bound inheritance rule moves (it was integrated over too small
#: a region), and every region the exact DBM route answers may move at the
#: accuracy of the fallback that served it before (or by the ±200 box, see
#: the flag's comment); the answered regions no longer reach the fallback,
#: so routes change where ``dbm_answered`` is nonzero.
M3_VALUE_KEYS = ('poset_lower_not_inherited', 'dbm_answered')
M3_ROUTE_KEYS = ('dbm_answered',)


def load_merged(paths):
    """Merge assembled npz files (e.g. a tracked and a local one) into one
    (meta, arrays).  An entry recorded ``NOT_RUN`` in one file yields to a
    real record in another.  Missing paths are skipped."""
    merged_entries, merged_arrays, metas = {}, {}, []
    for p in paths:
        if not p or not os.path.exists(p):
            continue
        meta, arrays = load_baseline(p)
        metas.append(meta)
        for name, s in (meta.get('entries') or {}).items():
            old = merged_entries.get(name)
            if old is None or old.get('status') == 'NOT_RUN':
                merged_entries[name] = s
        for name, arrs in arrays.items():
            merged_arrays.setdefault(name, arrs)
    code = [m.get('code') for m in metas]
    return {'entries': merged_entries, 'code': code}, merged_arrays


def _ctr(s, key):
    c = (s or {}).get('counters') or {}
    return c.get(key)


def delta_table(base, new, *, rtol=1e-13):
    """One row per entry of ``base`` ∪ ``new`` (each ``(meta, arrays)`` from
    ``load_merged``): status before/after, max abs / rel change over every
    stored array (and of ``total`` alone), whether that is within ``rtol``
    (``compare_values``; cross-process jitter is ~1e-16, plan §4.4), nquad
    and zero-normal counters before/after, the new run's Θ(0) counters and a
    verdict:

    * ``unchanged`` -- every array within ``rtol``;
    * ``MOVES``     -- some array outside it;
    * ``VIOLATION`` -- moved although the baseline saw no zero-normal row
      (``zero_normal_rows_seen == 0``) and every ``M1_VALUE_KEYS``,
      ``M2A_VALUE_KEYS``, ``M2B_VALUE_KEYS`` and ``M3_VALUE_KEYS`` counter
      of the new run is 0 or absent: such an entry must not move at M1 /
      M2a / M2b / M3;
    * ``status``    -- the status changed (e.g. TIMEOUT -> ok);
    * ``census``    -- nquad-stubbed on both sides (counts only);
    * ``n/a``       -- no values on either side (TIMEOUT / ERR both times).
    """
    import numpy as np
    bmeta, barr = base
    nmeta, narr = new
    names = sorted(set(bmeta['entries']) | set(nmeta['entries']))
    rows = []
    for name in names:
        sb = bmeta['entries'].get(name) or {}
        sn = nmeta['entries'].get(name) or {}
        st_b = str(sb.get('status', 'absent'))[:40]
        st_n = str(sn.get('status', 'absent'))[:40]
        row = {'name': name, 'status_before': st_b, 'status_after': st_n,
               'm1_expected': sb.get('m1_expected', sn.get('m1_expected')),
               'wall_before': sb.get('wall_process'),
               'wall_after': sn.get('wall_process'),
               'nquad_before': _ctr(sb, 'nquad_calls'),
               'nquad_after': _ctr(sn, 'nquad_calls'),
               'zero_normal_before': _ctr(sb, 'zero_normal_rows_seen'),
               'zero_normal_after': _ctr(sn, 'zero_normal_rows_seen'),
               'stub_calls_before': sb.get('stub_calls'),
               'stub_calls_after': sn.get('stub_calls'),
               'theta0': {k: _ctr(sn, k) for k in DELTA_THETA0_KEYS},
               'm2a': {k: _ctr(sn, k) for k in M2A_VALUE_KEYS},
               'm2b': {k: _ctr(sn, k) for k in M2B_VALUE_KEYS},
               'm3': {k: _ctr(sn, k) for k in M3_VALUE_KEYS}}
        if name in barr and name in narr:
            cmp = compare_values(barr[name], narr[name], rtol=rtol)
            row['max_abs'] = max(v[0] for v in cmp.values())
            row['max_rel'] = max(v[1] for v in cmp.values())
            if 'total' in cmp:
                row['total_max_abs'], row['total_max_rel'] = cmp['total'][:2]
            row['within_rtol'] = all(v[2] for v in cmp.values())
            row['arrays_moved'] = sorted(k for k, v in cmp.items() if not v[2])
            if row['within_rtol']:
                row['verdict'] = 'unchanged'
            elif (sb.get('counters')
                  and _ctr(sb, 'zero_normal_rows_seen') == 0
                  and not any(_ctr(sn, k) for k in M1_VALUE_KEYS
                              + M2A_VALUE_KEYS + M2B_VALUE_KEYS
                              + M3_VALUE_KEYS)):
                row['verdict'] = 'VIOLATION'
            else:
                row['verdict'] = 'MOVES'
        elif st_b.startswith('census') and st_n.startswith('census'):
            row['verdict'] = 'census'
        elif st_b != st_n:
            row['verdict'] = 'status'
        else:
            row['verdict'] = 'n/a'
        rows.append(row)
    return rows


def format_delta_table(rows):
    def f(x, spec):
        return '-' if x is None else format(x, spec)

    def cnt(x):
        return '-' if x is None else str(x)
    hdr = (f"{'entry':48s} {'status (before -> after)':34s} {'verdict':9s} "
           f"{'max_abs':>9s} {'max_rel':>9s} {'nquad b->a':>12s} "
           f"{'zero-normal b->a':>17s}  theta0 (empty/drop/tie/ordered/"
           f"pruned/zero_area/poset_const/poset_cycle)  m2a (polytope_cycle/"
           f"forced_delta_pruned)  m2b (nquad_hardened_calls)  m3 "
           f"(poset_lower_not_inherited/dbm_answered)")
    out = [hdr]
    for r in rows:
        th = r['theta0']
        th_s = '/'.join(cnt(th[k]) for k in DELTA_THETA0_KEYS)
        m2a = r.get('m2a') or {}
        m2a_s = '/'.join(cnt(m2a.get(k)) for k in M2A_VALUE_KEYS)
        m2b = r.get('m2b') or {}
        m2a_s += '  ' + '/'.join(cnt(m2b.get(k)) for k in M2B_VALUE_KEYS)
        m3 = r.get('m3') or {}
        m2a_s += '  ' + '/'.join(cnt(m3.get(k)) for k in M3_VALUE_KEYS)
        st = f"{r['status_before'][:15]} -> {r['status_after'][:15]}"
        out.append(
            f"{r['name']:48s} {st:34s} {r['verdict']:9s} "
            f"{f(r.get('max_abs'), '9.2e')} {f(r.get('max_rel'), '9.2e')} "
            f"{cnt(r['nquad_before']) + '->' + cnt(r['nquad_after']):>12s} "
            f"{cnt(r['zero_normal_before']) + '->' + cnt(r['zero_normal_after']):>17s}"
            f"  {th_s}  {m2a_s}")
        if r['verdict'] == 'census' and (r['stub_calls_before']
                                         or r['stub_calls_after']):
            out.append(f"{'':48s}   stub calls by m: {r['stub_calls_before']}"
                       f" -> {r['stub_calls_after']}")
    return '\n'.join(out)


# ═══════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════

def _require_hashseed():
    if os.environ.get('PYTHONHASHSEED') != '0':
        sys.exit('phase_j_zoo_baseline: PYTHONHASHSEED=0 is required '
                 f"(got {os.environ.get('PYTHONHASHSEED')!r}).")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('--list', action='store_true')
    ap.add_argument('--run', help="comma-separated entry names, or 'all', "
                                  "'tracked', 'local'")
    ap.add_argument('--workers', type=int, default=3)
    ap.add_argument('--out-dir', default=WORK_DIR)
    ap.add_argument('--assemble', action='store_true')
    ap.add_argument('--compare-dir', default=None,
                    help='independent run dir for the reproducibility record')
    ap.add_argument('--note', default='', help='free-text run_meta note')
    ap.add_argument('--log', default=None)
    ap.add_argument('--assemble-to', default=None, metavar='PREFIX',
                    help='write the assembled run to PREFIX.{npz,json} '
                         '(tracked-routable entries) and PREFIX_local.'
                         '{npz,json} (local-only entries) instead of the '
                         'baseline files; use a gitignored path (scratch/)')
    ap.add_argument('--overwrite-baseline', action='store_true',
                    help='allow --assemble without --assemble-to to replace '
                         'the existing pre-change baseline (never after M0)')
    ap.add_argument('--delta', default=None, metavar='PREFIX',
                    help='print the delta table of PREFIX.npz (+ '
                         'PREFIX_local.npz) against the baseline and write '
                         'PREFIX_delta.json')
    ap.add_argument('--delta-base', default=None,
                    help='comma list of baseline npz files (default: the '
                         'tracked baseline + the local one when present)')
    ap.add_argument('--child', help=argparse.SUPPRESS)
    ap.add_argument('--child-out', help=argparse.SUPPRESS)
    a = ap.parse_args(argv)

    if a.child:
        _require_hashseed()
        import warnings
        warnings.simplefilter('ignore')
        t0 = time.time()
        rec = run_entry(entry_by_name(a.child))
        rec['wall_child_total'] = time.time() - t0
        rec['code'] = code_provenance()
        _write_json_atomic(a.child_out, _jsonable(rec))
        return 0

    if a.list:
        for e in all_entries():
            print(f"{e['name']:48s} {destination(e):8s} "
                  f"{git_route(model_path(e)):8s} timeout={e['timeout']:.0f}s "
                  f"m1={e['m1']}")
        return 0

    if a.delta and not (a.run or a.assemble):
        return _delta_main(a)
    _require_hashseed()
    if a.assemble and not a.assemble_to and not a.overwrite_baseline and (
            os.path.exists(TRACKED_NPZ) or os.path.exists(LOCAL_NPZ)):
        sys.exit('phase_j_zoo_baseline: refusing to overwrite the pre-change '
                 'baseline (it is never refrozen).  Use --assemble-to '
                 'scratch/<prefix> for a later run, or --overwrite-baseline.')
    if a.run:
        if a.run in ('all', 'tracked', 'local'):
            names = [e['name'] for e in all_entries()
                     if a.run == 'all' or destination(e) == a.run]
        else:
            names = [s.strip() for s in a.run.split(',') if s.strip()]
        log = a.log or os.path.join(a.out_dir, '..', 'run.log')
        os.makedirs(a.out_dir, exist_ok=True)
        run_many(names, workers=a.workers, out_dir=a.out_dir, log=log)
    if a.assemble:
        meta = {'assembled_by': 'phase_j_zoo_baseline --assemble',
                'run_dir': os.path.relpath(a.out_dir, _REPO_ROOT),
                'compare_dir': (os.path.relpath(a.compare_dir, _REPO_ROOT)
                                if a.compare_dir else None),
                'note': a.note}
        paths = {}
        if a.assemble_to:
            pre = os.path.abspath(a.assemble_to)
            paths = dict(tracked_npz=pre + '.npz', tracked_json=pre + '.json',
                         local_npz=pre + '_local.npz',
                         local_json=pre + '_local.json',
                         plan_label=a.note or 'post-change zoo run')
        for p in assemble(a.out_dir, run_meta=meta,
                          compare_dir=a.compare_dir, **paths):
            print('wrote', os.path.relpath(p, _REPO_ROOT))
    if a.delta:
        return _delta_main(a)
    return 0


def _delta_main(a):
    pre = os.path.abspath(a.delta)
    new = load_merged([pre + '.npz', pre + '_local.npz'])
    base_paths = (a.delta_base.split(',') if a.delta_base
                  else [TRACKED_NPZ, LOCAL_NPZ])
    base = load_merged(base_paths)
    rows = delta_table(base, new)
    print(format_delta_table(rows))
    counts = {}
    for r in rows:
        counts[r['verdict']] = counts.get(r['verdict'], 0) + 1
    print('verdicts:', counts)
    out = pre + '_delta.json'
    _write_json_atomic(out, _jsonable({
        'base': [os.path.relpath(p, _REPO_ROOT) for p in base_paths],
        'new': os.path.relpath(pre, _REPO_ROOT), 'rows': rows,
        'verdicts': counts}))
    print('wrote', os.path.relpath(out, _REPO_ROOT))
    return 1 if counts.get('VIOLATION') else 0


if __name__ == '__main__':
    sys.exit(main())
