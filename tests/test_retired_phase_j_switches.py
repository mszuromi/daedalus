"""Audit W7: the retired Phase J rollback and setup switches.

Each retired environment variable, when set, warns ONCE per process
(``FutureWarning``, through ``engine._env.retired``) and is otherwise
ignored: a child process with every one of them set to its old non-default
value gives the numbers of this process, where none is set.  The retired
module attributes are gone.
"""
import json
import math
import os
import subprocess
import sys
import textwrap

import pytest

pytest.importorskip('sage.all')

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import engine.integration.time_domain.final_integral as FI   # noqa: E402
import engine.diagrams.symmetry as SYM                       # noqa: E402
import api._pole_clusters as PC                              # noqa: E402

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Every retired variable with the value that used to select the legacy path.
_RETIRED = {
    'DAEDALUS_PHASE_J_LEGACY': '1',
    'DAEDALUS_PHASE_J_THETA0_CONST_ROW': 'legacy_clip',
    'DAEDALUS_PHASE_J_STRUCTURAL_ZEROS': '0',
    'DAEDALUS_PHASE_J_NQUAD_HARDENED': '0',
    'DAEDALUS_PHASE_J_DBM': '0',
    'DAEDALUS_PHASE_J_DD_KERNELS': '0',
    'DAEDALUS_CHAIN_UPPERS_KERNEL': 'transfer',
    'DAEDALUS_CHAIN_UPPERS_MEMO': '0',
    'DAEDALUS_PHASE_J_SEMIINF_CHAIN': '0',
    'DAEDALUS_PHASE_J_UNBOUNDED_2D': 'box',
    'DAEDALUS_PRUNE_ZERO_MODES': '0',
    'DAEDALUS_POLE_CLUSTERS': '0',
    'DAEDALUS_PHASE_J_LEGACY_SETUP': '1',
    'DAEDALUS_SETUP_ZERO_EXIT': '0',
    'DAEDALUS_SETUP_PROP_TD': '0',
    'DAEDALUS_SETUP_LAZY_SR': '0',
    'DAEDALUS_SETUP_AUT_MEMO': '0',
    'DAEDALUS_SETUP_DELTA_SOLVE': '0',
    'DAEDALUS_SHOTNOISE_INTEGRATED': '0',
}

_RETIRED_ATTRS = {
    FI: ('THETA0_CONST_ROW_MODE', 'STRUCTURAL_ZEROS', 'NQUAD_HARDENED',
         'USE_DBM_FALLBACK', 'USE_PHASE_J_DD_KERNELS', 'CHAIN_UPPERS_KERNEL',
         'USE_CHAIN_UPPERS_MEMO', 'SEMIINF_CHAIN', 'UNBOUNDED_2D',
         'PRUNE_ZERO_MODES', 'USE_SETUP_ZERO_EXIT', 'USE_SETUP_PROP_TD',
         'USE_SETUP_LAZY_SR', 'USE_SETUP_DELTA_SOLVE',
         'USE_SHOTNOISE_INTEGRATED', 'POLYGON_BBOX_CAP', '_theta0_legacy',
         '_structural_zeros_on', '_nquad_hardened_on', '_dbm_route_on',
         '_phase_j_dd_kernels_on', '_chain_uppers_memo_on',
         '_semiinf_chain_on', '_unbounded_2d_exact', '_setup_lever_on',
         '_shotnoise_integrated_on', '_legacy_umbrella_flags',
         '_initial_phase_j_flags'),
    SYM: ('USE_AUT_MEMO', '_aut_memo_on'),
    PC: ('ENV_SWITCH', 'enabled'),
}

# The child (and this process) evaluates: model-free closed forms on the
# m = 1, 2, >= 3 and DBM routes, and ou_quartic k = 2, ell <= 1 (pole
# engine) with its shot-noise-free result; prints them as JSON.
_PROBE = textwrap.dedent(r'''
    import json, os, sys, warnings
    sys.path.insert(0, sys.argv[1])
    warnings.simplefilter('always')
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter('always')
        import engine.integration.time_domain.final_integral as FI
        import engine.diagrams.symmetry as SYM
        import api._pole_clusters as PC
    msgs = [str(w.message) for w in rec if issubclass(w.category,
                                                     FutureWarning)]

    def ems(*modes):
        return FI.EdgeModeSum(ri=0, pi=0, delta_coeff=0j, modes=tuple(modes))

    out = {}
    wedge = [((-1.0, 0.0), (1.0,), 0.0), ((1.0, -1.0), (0.0,), 0.0),
             ((0.0, -1.0), (1.0,), 0.3)]
    wm = [ems((1.0, -0.03)), ems((0.7, -1.0 + 0.5j)), ems((1.0, -0.03 + 0.2j))]
    out['m2'] = FI._integrate_2d_polygon_modesum(wm, 1.0, wedge, [0.7])
    iv = [((-1.0,), (1.0,), 0.0), ((1.0,), (-1.0,), 2.0)]
    out['m1'] = FI._integrate_1d_polytope_modesum(
        [ems((1.0, -1.0)), ems((1.0, -1.0 + 1e-12))], 0.7 - 0.1j, iv, [0.4])
    chain = [((-1.0, 1.0, 0.0), (0.0,), 0.0), ((0.0, -1.0, 1.0), (0.0,), 0.0),
             ((0.0, 0.0, -1.0), (1.0,), 0.0)]
    cm = [ems((1.0, -0.05)), ems((0.8, -1.5)), ems((1.0, -3.0))]
    out['m3'] = FI._integrate_nd_polytope_poset_modesum(cm, 1.0, chain,
                                                        [0.7], 3)
    dbm_rows = [((-1.0, 0.0, 0.0), (1.0,), 0.0), ((1.0, -1.0, 0.0), (0.0,), 0.0),
                ((0.0, -1.0, 1.0), (0.0,), 0.0), ((0.0, 0.0, -1.0), (1.0,), 0.0)]
    dm = [ems((1.0, -1.0)), ems((0.8, -0.01)), ems((1.0, -0.01 + 0.3j)),
          ems((0.6, -2.0))]
    out['dbm'] = FI._integrate_subset_dbm(dm, 1.0, dbm_rows, [0.7], 3)
    os.environ['DAEDALUS_PHASE_J_ENGINE'] = 'poles'
    import numpy as np
    import daedalus as dd
    from api import compute_cumulants
    os.chdir(sys.argv[2])
    model, _ = dd.load_model('ou_quartic')
    res = compute_cumulants(model, k=2, max_ell=1,
                            external_fields=[('dx', 1)] * 2,
                            parameters={'mu': 1.0, 'D': 1.0, 'eps': 0.1},
                            tau_grid=np.array([0.0, 0.5, 2.0]),
                            use_cache=False, parallel=False, verbose=False)
    for ell, a in res['C_tau_by_ell'].items():
        out[f'ou{ell}'] = [complex(x) for x in a]
    flags = res['config']['phase_j_flags']
    print('RESULT', json.dumps({
        'msgs': msgs, 'flags': flags,
        'vals': {k: ([[z.real, z.imag] for z in v] if isinstance(v, list)
                     else [complex(v).real, complex(v).imag])
                 for k, v in out.items()}}))
''')


def _probe(tmp_path, env_extra, tag):
    child = tmp_path / 'probe.py'
    child.write_text(_PROBE)
    cwd = tmp_path / f'cwd_{tag}'          # no shared cache between probes
    cwd.mkdir(exist_ok=True)
    env = {k: v for k, v in os.environ.items() if k not in _RETIRED}
    env.update(env_extra, PYTHONHASHSEED='0')
    r = subprocess.run([sys.executable, str(child), _REPO, str(cwd)],
                       capture_output=True, text=True, timeout=900, env=env)
    assert r.returncode == 0, r.stderr[-3000:]
    line = [ln for ln in r.stdout.splitlines() if ln.startswith('RESULT ')]
    assert line, r.stdout[-3000:]
    return json.loads(line[-1][len('RESULT '):])


def test_the_retired_attributes_are_gone():
    for mod, names in _RETIRED_ATTRS.items():
        for n in names:
            assert not hasattr(mod, n), (mod.__name__, n)


def test_the_retired_table_names_every_switch():
    names = [n for n, _note in FI._RETIRED_PHASE_J_ENV]
    assert len(names) == len(set(names))
    assert set(names) | {'DAEDALUS_SETUP_AUT_MEMO',
                         'DAEDALUS_POLE_CLUSTERS'} == set(_RETIRED)
    notes = dict(FI._RETIRED_PHASE_J_ENV)
    assert all(isinstance(note, str) and note for note in notes.values())
    umb = notes['DAEDALUS_PHASE_J_LEGACY']
    assert 'no longer reproducible' in umb
    assert 'phase_j_legacy_baseline' in umb


@pytest.mark.slow
def test_each_retired_switch_warns_once_and_changes_no_number(tmp_path):
    """One child with every retired variable set to its old legacy value,
    one with none: one FutureWarning per variable in the first, none in the
    second, and the same numbers (cross-process, so rtol 1e-13)."""
    got = _probe(tmp_path, _RETIRED, 'retired')
    ref = _probe(tmp_path, {}, 'none')
    assert ref['msgs'] == []
    for name in _RETIRED:
        hits = [m for m in got['msgs']
                if m.startswith(f'{name} is retired and ignored: ')]
        assert len(hits) == 1, (name, got['msgs'])
    assert len(got['msgs']) == len(_RETIRED)
    assert set(got['vals']) == set(ref['vals'])
    for key, v in ref['vals'].items():
        a = [complex(*p) for p in (v if isinstance(v[0], list) else [v])]
        b = [complex(*p) for p in (got['vals'][key]
                                   if isinstance(got['vals'][key][0], list)
                                   else [got['vals'][key]])]
        for x, y in zip(a, b):
            assert abs(x - y) <= 1e-13 * max(abs(x), 1e-300), (key, x, y)
            assert math.isfinite(abs(x))
    assert got['flags'] == ref['flags']


def test_a_retired_switch_set_after_import_warns_at_the_next_pass(monkeypatch):
    """The retired readers ran at call time, so a switch set after import
    (a notebook ``os.environ[...] = ...``) warns at the next Phase J pass;
    fast (no computation: an empty diagram list)."""
    from engine import _env
    from engine.integration.time_domain.pipeline import compute_correction_td
    names = [n for n, _ in FI._RETIRED_PHASE_J_ENV] + [
        'DAEDALUS_SETUP_AUT_MEMO', 'DAEDALUS_POLE_CLUSTERS']
    for n in names:
        monkeypatch.setenv(n, '0')
        _env._RETIRED_SEEN.discard(n)
    import warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        compute_correction_td(typed_diagrams=[], prefactors=[],
                              propagator_data={'pole_vals': [], 'C_mats': []},
                              k=2)
    got = sorted(str(x.message).split(' is retired')[0] for x in w
                 if issubclass(x.category, FutureWarning))
    assert got == sorted(names)
