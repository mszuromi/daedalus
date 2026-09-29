"""
tests/test_phase_j_legacy_baseline.py
=====================================
Pre-change (M0.3) Phase J baseline guard (``docs/integration_speedup_plan.md``
§3.1 "Rollback and flag hygiene", §4.3, §4.4).

The data files:

* ``tests/fixtures/phase_j_legacy_baseline.npz`` (and its readable ``.json``
  twin) -- the model-zoo baseline written by
  ``tests/tools/phase_j_zoo_baseline.py`` on the pre-M1 code (PYTHONHASHSEED=0);
* ``tests/phase_j_refactor_fixtures/legacy/*.npz`` -- byte-for-byte copies of
  the four frozen Phase J fixtures, taken before M1.

Neither is ever refrozen.  This file checks that:

1. the baseline loads, is self-consistent and contains no local-only model;
2. the legacy fixture copies are byte-identical to their M0 hashes, and
   their probes (including the raw k=2 exact tie (0, 0) and the negative-τ
   probes) reproduce under the legacy flags at the fixture tolerance, in
   either cache state (``test_legacy_fixture_values_reproduce``; the
   default-mode fixture test of the moved ``spike_reset_k2_ell1`` is a
   strict xfail until M9, so this is its rollback guard -- in the slow
   suite, while the default suite runs the same model, parameters and
   flags through the zoo entry 'spike_reset-k2-l1' (legacy));
3. zoo entries reproduce in-process to rtol 1e-13 (plan §4.4: hash
   randomisation causes 1-ulp jitter, so ``==`` is not used in-repo), and so
   do their nquad / zero-normal counters, in two modes:

   * ``legacy`` -- the flags ``DAEDALUS_PHASE_J_LEGACY=1`` selects (Θ(0)
     ``legacy_clip``; the bounding box is read at call time in every mode),
     set in-process on ``final_integral`` with ``monkeypatch`` (the flags
     are read at call time).  Checked for an M1 mover AND a non-mover: the
     umbrella must reproduce the pre-change numbers of every entry, and it
     must never reach the M1 Θ(0) code (its counters stay 0).
   * ``default`` -- the current defaults, only for entries whose numbers did
     not move at M1 (``m1_expected != 'moves'``).  The M1 zoo re-run measured
     every such tracked entry unchanged to rtol 1e-13 (plan §7, M1 delta
     table).

   The heavier entries of both modes are ``slow`` (``pytest -m slow``).
   (Both kernel backends at M9.)

Cache state (plan Appendix C.1).  The baseline was recorded from the
developer's local caches: the gitignored v1 prediagram cells in
``saved_prediagrams/`` (cwd-relative).  A checkout without them (a fresh
clone) enumerates from the shipped v2 cells and gets other isomorphism
representatives, and an entry that reaches scipy.nquad then takes other
routes: different nquad / path counts, and values that differ at the nquad
tolerance (measured on a fresh clone: grouped spike reset k=2 l=1 legacy 30
nquad calls vs 180, quad_exp k=2 l=1 ell1 2.0e-9 rel).  So for a ZOO entry
whose baseline made nquad calls, and only when the v1 cells of every
(k, ell) of the entry are absent, the values are compared at rtol 1e-8 and
the route counters are not compared (the Θ(0) counters still must stay 0
in legacy mode).  Entries without nquad calls are compared strictly in
both states, and so are the four legacy FIXTURE copies (item 2: they
reproduce to ~1e-14 in both states, well inside their rtol 1e-10).

Cost (default suite): about 35 s (three small models and three legacy
fixtures, warm ``saved_models/`` cache).
"""
from __future__ import annotations

import hashlib
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from tests.tools import phase_j_zoo_baseline as Z   # noqa: E402

_LEGACY_DIR = os.path.join(os.path.dirname(__file__),
                           'phase_j_refactor_fixtures', 'legacy')
_LEGACY_SHA256 = {
    'spike_reset_k1_ell1.npz':
        '7814492635b59703f4209771e8b0a555817dcf18f8ab65273b9fc2d809eaf079',
    'spike_reset_k2_ell0.npz':
        '50e50379c3f8e73778b01ab5f278acbb4839b905bbe2eca62d40b75bfa6f205d',
    'spike_reset_k2_ell1.npz':
        '2fafe8fad0107bceb25cbf40872667fc3e8757c56d834713af7fca3c106d6ec7',
    'quad_exp_k2_ell0.npz':
        'ac2d0c0ec008ac4428b6f5f539af79e1ede530751080d9d6f9b1295d0844993e',
}

# (entry, mode) pairs re-run in-process.  'legacy' = the umbrella's flags,
# 'default' = the current defaults (never for an M1 mover).  Keep the
# default-suite cost <= ~30 s; everything heavier is in _REPRODUCE_SLOW.
_REPRODUCE = [
    ('ou_quartic-k2-l3', 'default'),     # non-mover: 0 nquad, no zero-normal rows
    ('spike_reset-k1-l1', 'default'),    # non-mover (M1 zoo: bit-identical)
    ('spike_reset-k1-l1', 'legacy'),     # non-mover under the umbrella
    # M1 mover under the umbrella.  Also NOT bit-reproducible run to run
    # (1-2 ulp, measured at M0.3: max rel 7.7e-16), hence rtol 1e-13.
    ('spike_reset-k2-l1', 'legacy'),
]
_REPRODUCE_SLOW = [
    ('spike_reset-k2-l1-grouped', 'legacy'),     # mover, grouped dispatch
    # non-mover WITH zero-normal rows (all c_eff < 0: EMPTY either way)
    ('quad_exp-k2-l1', 'legacy'),
    ('quad_exp-k2-l1', 'default'),
    # the other tracked non-movers, current defaults.  NOTE: ou_quartic at
    # k=3 is identically 0 (odd cumulants vanish at the symmetric saddle): no
    # Phase J integral is evaluated (poset/interval/polygon attempted = 0), so
    # this entry is no evidence about ties.  The OU tie evidence is k=4
    # (below, and test_phase_j_theta0.py::
    # test_ou_quartic_near_ties_do_not_depend_on_the_mode).
    ('ou_quartic-k3-l2', 'default'),
    ('ou_quartic-k4-l2', 'default'),
    ('ou_quartic_colored-k2-l1', 'default'),
    ('ou_quartic_two_dim_color_corr-k2-l1', 'default'),
    ('linear_hawkes-k2-l1', 'default'),
    ('multipopulation_test-k2-l1', 'default'),
    ('linear_delta_spikes-k2-l1', 'default'),
    ('spatial-allen_cahn_1d-l2', 'default'),
    ('spatial-reaction_diffusion_conserved_1d-l1', 'default'),
    ('spatial-edwards_wilkinson_1d-l0', 'default'),
    ('spatial-reaction_diffusion_2d-l1', 'default'),
    ('spatial-coupled_rd_2species_1d-l1', 'default'),
]
_MODES = ('default', 'legacy')

_COUNTER_KEYS = ('nquad_calls', 'polytope_m0_direct', 'scipy_nquad_called_m1',
                 'scipy_nquad_called_m2', 'scipy_nquad_called_mge3',
                 'nquad_fallback_by_reason', 'zero_normal_rows_seen',
                 'polygon_attempted', 'poset_attempted')


@pytest.fixture(scope='module')
def baseline():
    if not os.path.exists(Z.TRACKED_NPZ):
        pytest.fail(f'missing baseline {Z.TRACKED_NPZ}: regenerate with '
                    '`PYTHONHASHSEED=0 sage -python -m '
                    'tests.tools.phase_j_zoo_baseline --run tracked '
                    '--assemble` on the PRE-M1 code only')
    return Z.load_baseline(Z.TRACKED_NPZ)


def test_legacy_fixture_copies_are_frozen():
    """The four pre-M1 fixture copies are byte-identical to their M0 hashes."""
    for fname, sha in _LEGACY_SHA256.items():
        path = os.path.join(_LEGACY_DIR, fname)
        assert os.path.exists(path), f'missing legacy fixture {fname}'
        with open(path, 'rb') as fh:
            got = hashlib.sha256(fh.read()).hexdigest()
        assert got == sha, (f'{fname} changed: legacy fixtures are never '
                            'refrozen')


# The four frozen fixtures under the legacy flags (plan §3.1: the umbrella
# test compares against legacy/*.npz).  spike_reset_k2_ell1 is the only
# expensive one (about 10 s; the others take under 1 s each) and is slow
# here: the default suite already runs the same model, parameters, k, ell
# and legacy flags through the zoo entry 'spike_reset-k2-l1' (legacy, in
# ``_REPRODUCE``, values AND route counters).  What only this fixture adds
# is its raw probes, among them the exact k=2 tie (0, 0).
_LEGACY_FIXTURES_SLOW = ('spike_reset_k2_ell1',)


def _legacy_fixture_params():
    from tests.phase_j_refactor_fixtures._configs import FIXTURES
    return [pytest.param(fx, id=fx.name,
                         marks=([pytest.mark.slow]
                                if fx.name in _LEGACY_FIXTURES_SLOW else []))
            for fx in FIXTURES]


@pytest.mark.parametrize('fx', _legacy_fixture_params())
def test_legacy_fixture_values_reproduce(fx):
    """Every probe of a frozen pre-change fixture copy (``legacy/``)
    reproduces under the legacy umbrella's flags, set in-process, at the
    fixture's own tolerance (``fx.rtol``, 1e-10; cross-process, so not
    ``==``) -- in BOTH cache states.  Unlike the zoo entries (module
    docstring) the fixtures need no relaxation without the local v1
    prediagram cells: measured on a fresh clone (no v1 cells, warm and
    fully cold caches, three hash seeds) every fixture matched its copy to
    <= 1.9e-14 relative, including spike_reset_k2_ell1, which makes 576
    scipy.nquad calls in either state (7.7e-16 with the local cells)."""
    from tests.phase_j_refactor_fixtures._runner import evaluate
    from tests.tools.phase_j_subset_diff import phase_j_flags
    ref = np.load(os.path.join(_LEGACY_DIR, f'{fx.name}.npz'))
    with phase_j_flags('legacy') as flags:
        assert flags['THETA0_CONST_ROW_MODE'] == 'legacy_clip'
        cur = evaluate(fx)
    np.testing.assert_array_equal(cur['tau_probes'], ref['tau_probes'])
    np.testing.assert_allclose(
        cur['C_values'], ref['C_values'], rtol=fx.rtol, atol=fx.atol,
        err_msg=f'{fx.name}: legacy flags vs legacy/{fx.name}.npz')


def test_baseline_loads_and_is_consistent(baseline):
    meta, arrays = baseline
    assert meta['schema_version'] == Z.SCHEMA_VERSION
    assert meta['destination'] == 'tracked'
    entries = meta['entries']
    tracked_zoo = [e['name'] for e in Z.all_entries(include_local=False)
                   if Z.destination(e) == 'tracked']
    # every tracked zoo entry is recorded (possibly as TIMEOUT/ERR)
    missing = sorted(set(tracked_zoo) - set(entries))
    assert not missing, f'zoo entries absent from the baseline: {missing}'
    # nothing local-only leaked into the tracked file: neither a zoo entry
    # routed to the local file nor an entry of a local extra-zoo file
    # (``DAEDALUS_ZOO_EXTRA``; none on a fresh clone)
    local = {e['name'] for e in Z.all_entries(include_local=True)
             if Z.destination(e) == 'local'}
    leaked = sorted((set(entries) | set(arrays)) & local)
    assert not leaked, f'local-only entries in the tracked baseline: {leaked}'
    for name, s in entries.items():
        # every model behind a tracked entry is tracked (or new, not ignored)
        cfg = s.get('config') or {}
        path = Z.model_path(cfg) if (cfg.get('model') or
                                     cfg.get('model_file')) else None
        if path is not None:
            assert Z.git_route(path) in ('tracked', 'new'), (name, path)
        if s['status'] == 'ok':
            assert s['pythonhashseed'] == '0', name
            assert s['model_git'] in ('tracked', 'new'), name
            assert name in arrays and 'total' in arrays[name], name
            assert np.all(np.isfinite(arrays[name]['total'])), name
        if s['status'] == 'census_only':
            assert name not in arrays, 'census-only values must not be stored'
    # the pre-M1 code state is recorded
    assert meta['code']['git_head']
    assert len(meta['code']['sha256_final_integral']) == 64


# Counters added at M1 (Θ(0) helper): the legacy umbrella never reaches that
# code, so they stay 0 in legacy mode.
_THETA0_COUNTER_KEYS = ('theta0_const_empty', 'theta0_const_drop',
                        'theta0_tie', 'theta0_tie_ordered',
                        'theta0_subsets_pruned', 'polygon_zero_area',
                        'poset_empty_const', 'poset_empty_cycle')

# Relaxed value tolerance for an nquad-served entry run from other diagram
# representatives than the baseline's (see the module docstring).
_OTHER_REPRESENTATIVES_RTOL = 1e-8


def _baseline_cache_state(entry):
    """True when this checkout has the local v1 prediagram cells the
    baseline was recorded from, for every (k, ell) of ``entry`` (cwd-relative,
    like every cache root).  Spatial entries do not use them."""
    if entry.get('kind') == 'spatial':
        return True
    from api._diagrams import PREDIAGRAM_CACHE_ROOT
    from engine.enumeration.prediagram_cache import v1_exists
    return all(v1_exists(PREDIAGRAM_CACHE_ROOT, entry['k'], ell)
               for ell in range(int(entry['max_ell']) + 1))


def test_reproduce_lists_respect_the_m1_movers(baseline):
    """An M1 mover is only ever compared under the legacy umbrella; both a
    mover and a non-mover are checked in legacy mode in the default suite;
    every listed entry has an ok baseline."""
    meta, _arrays = baseline
    for name, mode in _REPRODUCE + _REPRODUCE_SLOW:
        assert mode in _MODES, (name, mode)
        s = meta['entries'][name]
        assert s['status'] == 'ok', (name, s['status'])
        if s.get('m1_expected') == 'moves':
            assert mode == 'legacy', (
                f'{name} moved at M1: compare it under the legacy umbrella '
                'only')
    legacy = [n for n, mode in _REPRODUCE if mode == 'legacy']
    assert any(meta['entries'][n].get('m1_expected') == 'moves'
               for n in legacy)
    assert any(meta['entries'][n].get('m1_expected') != 'moves'
               for n in legacy)
    assert any(mode == 'default' for _n, mode in _REPRODUCE)


def _reproduce(baseline, name, mode, monkeypatch):
    import engine.integration.time_domain.final_integral as FI
    meta, arrays = baseline
    s = meta['entries'].get(name)
    if s is None or s['status'] != 'ok':
        pytest.fail(f'{name}: no ok baseline entry ({s and s["status"]})')
    if mode == 'legacy':
        flags = FI._initial_phase_j_flags({'DAEDALUS_PHASE_J_LEGACY': '1'})
        assert flags['THETA0_CONST_ROW_MODE'] == 'legacy_clip'
        for flag, value in flags.items():
            monkeypatch.setattr(FI, flag, value)
    else:
        assert s.get('m1_expected') != 'moves', name
        default = FI._initial_phase_j_flags({})
        for flag, value in default.items():
            monkeypatch.setattr(FI, flag, value)
    entry = Z.entry_by_name(name, include_local=False)
    # Representative dependence (module docstring): only an nquad-served
    # entry run without the baseline's local prediagram cells is relaxed.
    strict = (_baseline_cache_state(entry)
              or not s['counters'].get('nquad_calls'))
    rec = Z.run_entry(entry, with_provenance=False)
    assert rec['status'] == 'ok', rec.get('status')
    assert rec['params_effective'] == s['params_effective'], name
    # The MODEL must be the one baselined (its file may change elsewhere).
    assert rec['model_spec_signature'] == s['model_spec_signature'], (
        f'{name}: model definition changed since the baseline')
    cmp = Z.compare_values(arrays[name], Z.result_arrays(rec),
                           rtol=1e-13 if strict
                           else _OTHER_REPRESENTATIVES_RTOL)
    bad = {k: v for k, v in cmp.items() if not v[2]}
    how = 'strict' if strict else 'other representatives'
    assert not bad, (f'{name} [{mode}, {how}]: (max_abs, max_rel, ok) per '
                     f'array: {bad}')
    if mode == 'legacy':
        for key in _THETA0_COUNTER_KEYS:
            assert rec['counters'].get(key, 0) == 0, (name, key)
    if not strict:
        return          # routes depend on the representatives (docstring)
    # Control flow is deterministic: counters reproduce exactly.  In the
    # default mode Θ(0)=0 may skip EMPTY constant-row subsets before they
    # reach an integrator, and the exact emptiness tests may answer 0 where a
    # subset bailed before (``Z.M1_ROUTE_KEYS``), so the routes are compared
    # exactly only when the baseline saw no zero-normal row and none of those
    # counters of this run is nonzero.
    if mode == 'legacy' or not (
            s['counters'].get('zero_normal_rows_seen')
            or any(rec['counters'].get(k) for k in Z.M1_ROUTE_KEYS)):
        for key in _COUNTER_KEYS:
            assert rec['counters'][key] == s['counters'][key], (name, key)
    else:
        assert (rec['counters']['nquad_calls']
                <= s['counters']['nquad_calls']), name


@pytest.mark.parametrize('name,mode', _REPRODUCE,
                         ids=[f'{n}-{m}' for n, m in _REPRODUCE])
def test_cheap_entries_reproduce(baseline, name, mode, monkeypatch):
    """The pre-M1 baseline reproduces to rtol 1e-13: under the legacy
    umbrella's flags for a mover and a non-mover, and with the current
    defaults for the non-movers."""
    _reproduce(baseline, name, mode, monkeypatch)


@pytest.mark.slow
@pytest.mark.parametrize('name,mode', _REPRODUCE_SLOW,
                         ids=[f'{n}-{m}' for n, m in _REPRODUCE_SLOW])
def test_heavier_entries_reproduce(baseline, name, mode, monkeypatch):
    """As ``test_cheap_entries_reproduce``, for the rest of the tracked
    non-movers (current defaults) and the grouped mover / a zero-normal
    non-mover under the umbrella."""
    _reproduce(baseline, name, mode, monkeypatch)


# ═══════════════════════════════════════════════════════════════════════
# Milestone tooling (model-free): delta table, overwrite guard, flag context
# ═══════════════════════════════════════════════════════════════════════

def _meta(entries):
    return {'entries': entries, 'code': []}


def _ok(zero_normal, nquad=0, **extra):
    return {'status': 'ok', 'wall_process': 1.0,
            'counters': {'zero_normal_rows_seen': zero_normal,
                         'nquad_calls': nquad, **extra}}


def test_delta_table_verdicts():
    a = np.array([1.0 + 0j, -2.0 + 0j])
    base = _meta({'same': _ok(0), 'jitter': _ok(0), 'moves': _ok(5, 4),
                  'bad': _ok(0), 'pz': _ok(0), 'pc': _ok(0), 'tc': _ok(0),
                  'fixed': {'status': 'TIMEOUT'},
                  'both_to': {'status': 'TIMEOUT'},
                  'cen': {'status': 'census_only', 'stub_calls': {'3': 2}}})
    new = _meta({'same': _ok(0), 'jitter': _ok(0), 'moves': _ok(2, 0),
                 'bad': _ok(0, polygon_zero_area=0, poset_empty_cycle=0),
                 'pz': _ok(0, polygon_zero_area=3),
                 'pc': _ok(0, poset_empty_cycle=1),
                 'tc': _ok(0, theta0_const_drop=2),     # an m=1 tie, say
                 'fixed': _ok(0), 'both_to': {'status': 'TIMEOUT'},
                 'cen': {'status': 'census_only', 'stub_calls': {}}})
    barr = {n: {'total': a.copy()}
            for n in ('same', 'jitter', 'moves', 'bad', 'pz', 'pc', 'tc')}
    narr = {'same': {'total': a.copy()},
            'jitter': {'total': a * (1 + 1e-15)},     # cross-process ulps
            'moves': {'total': a * 1.5},
            'bad': {'total': a * 1.5},
            'pz': {'total': a * 1.5},
            'pc': {'total': a * 1.5},
            'tc': {'total': a * 1.5},
            'fixed': {'total': a.copy()}}
    rows = {r['name']: r for r in Z.delta_table((base, barr), (new, narr))}
    assert rows['same']['verdict'] == 'unchanged'
    assert rows['same']['max_abs'] == 0.0
    assert rows['jitter']['verdict'] == 'unchanged'
    assert rows['moves']['verdict'] == 'MOVES'
    assert rows['moves']['max_rel'] == pytest.approx(0.5)
    assert (rows['moves']['nquad_before'], rows['moves']['nquad_after']) == (4, 0)
    # moved although the baseline saw no zero-normal row and the new run
    # decided nothing the M1 rules can change (Z.M1_VALUE_KEYS all 0)
    assert rows['bad']['verdict'] == 'VIOLATION'
    # the M1 invariant is keyed on those counters too: a move where the new
    # run answered a region empty, or decided a constant row outside the
    # polygon / poset paths (zero_normal_rows_seen misses those), is a MOVE
    assert rows['pz']['verdict'] == 'MOVES'
    assert rows['pc']['verdict'] == 'MOVES'
    assert rows['tc']['verdict'] == 'MOVES'
    assert rows['fixed']['verdict'] == 'status'
    assert rows['both_to']['verdict'] == 'n/a'
    assert rows['cen']['verdict'] == 'census'
    text = Z.format_delta_table(list(rows.values()))
    assert 'VIOLATION' in text and 'stub calls by m' in text


def test_assemble_refuses_to_overwrite_the_baseline(monkeypatch):
    """``--assemble`` without ``--assemble-to`` never replaces an existing
    pre-change baseline (it is never refrozen)."""
    assert os.path.exists(Z.TRACKED_NPZ)

    def _must_not_run(*_a, **_k):
        raise AssertionError('assemble() reached: the baseline would have '
                             'been overwritten')
    monkeypatch.setattr(Z, 'assemble', _must_not_run)
    monkeypatch.setenv('PYTHONHASHSEED', '0')
    with pytest.raises(SystemExit) as exc:
        Z.main(['--assemble'])
    assert 'refusing to overwrite' in str(exc.value)


def test_fixture_report_flag_context_restores():
    import engine.integration.time_domain.final_integral as FI
    from tests.tools import phase_j_subset_diff as H
    before = (FI.THETA0_CONST_ROW_MODE, FI.POLYGON_BBOX_CAP)
    with pytest.raises(RuntimeError):
        with H.phase_j_flags('legacy') as flags:
            assert FI.THETA0_CONST_ROW_MODE == 'legacy_clip'
            assert set(flags) == {'THETA0_CONST_ROW_MODE'}
            raise RuntimeError('restored even on error')
    assert (FI.THETA0_CONST_ROW_MODE, FI.POLYGON_BBOX_CAP) == before
    with H.phase_j_flags('default'):
        assert (FI.THETA0_CONST_ROW_MODE, FI.POLYGON_BBOX_CAP) == before
    with pytest.raises(ValueError):
        with H.phase_j_flags('nope'):
            pass
