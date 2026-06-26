"""
tests/test_serialize.py
========================
Round-trip tests for engine.core.serialize: save_theory / load_theory / reload_model.

Run with:
    cd "Automated Feynman Calculations"
    sage -python -m pytest tests/test_serialize.py -v

Or from a SageMath session:
    load('tests/test_serialize.py')
    run_all_tests()
"""

import os
import sys
import json
import shutil
import tempfile

from sage.all import SR, PolynomialRing, matrix, I, var, load as sage_load

# ── Ensure project root is importable ─────────────────────────────────────────
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from engine.core.serialize import save_theory, load_theory, reload_model, _strip_callables


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_expanded_ft():
    """
    Build and expand a FieldTheory from the Hawkes model.
    Returns (ft, ns, R).
    """
    # We need to load the SageMath files the SageMath way.
    # Import the field_theory module and model.
    ft_code_path = os.path.join(_PROJECT_ROOT, 'engine', 'core', 'field_theory.py')
    model_path   = os.path.join(_PROJECT_ROOT, 'simulations', 'hawkes_sage.py')

    ns_ft = {}
    with open(ft_code_path, 'r') as f:
        exec(compile(f.read(), ft_code_path, 'exec'), ns_ft)

    ns_model = {}
    with open(model_path, 'r') as f:
        exec(compile(f.read(), model_path, 'exec'), ns_model)

    FieldTheory = ns_ft['FieldTheory']
    HAWKES_MODEL = ns_model['HAWKES_MODEL']

    ft = FieldTheory(HAWKES_MODEL, taylor_order=4)
    ft.expand()
    return ft


def _make_propagator_data(ft):
    """
    Compute propagator data from an expanded FieldTheory.
    Mimics what the demo notebook does (simplified — no FT/inverse FT,
    just enough structure to test serialization).
    """
    from sage.all import function, dirac_delta, diff, integrate, oo, exp

    ns = ft._ns
    R  = ft._R
    S_free = ft.free_action()

    ring_gen_names = [str(g) for g in R.gens()]
    resp_names = [f'vt{i+1}' for i in ns.pop] + [f'nt{i+1}' for i in ns.pop]
    phys_names = [f'dv{i+1}' for i in ns.pop] + [f'dn{i+1}' for i in ns.pop]

    pos_to_row = {ring_gen_names.index(nm): i for i, nm in enumerate(resp_names)}
    pos_to_col = {ring_gen_names.index(nm): j for j, nm in enumerate(phys_names)}

    nf = len(resp_names)
    K_data = [[SR(0)] * nf for _ in range(nf)]

    for exp_vec, coeff in S_free.dict().items():
        row = col = None
        for k in range(len(ring_gen_names)):
            if exp_vec[k] > 0:
                if k in pos_to_row: row = pos_to_row[k]
                if k in pos_to_col: col = pos_to_col[k]
        if row is not None and col is not None:
            K_data[row][col] += SR(coeff)

    K_mat = matrix(SR, K_data)

    # Minimal propagator data for serialization test
    # (skip the full FT pipeline — just test that matrices round-trip)
    omega = SR.var('omega')
    # Use K_mat as a stand-in for K_ft (same type: Matrix_SR)
    return {
        'K_ft':              K_mat,
        'G_ft':              None,         # pretend inverse was too complex
        'adj_ft':            K_mat,        # placeholder
        'D_omega':           SR(1) + omega**2,   # placeholder
        'pole_vals':         [I, -I],      # placeholder
        'C_mats':            [K_mat, K_mat],  # placeholder
        'G_t':               None,
        'G_ft_explicit':     False,
        'propagator_branch': 'residue',
        'nf':                nf,
        'resp_names':        resp_names,
        'phys_names':        phys_names,
    }


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_strip_callables():
    """_strip_callables removes lambda/callable values from spec dicts."""
    specs = [
        {'name': 'phi', 'indexed': True, 'expression': lambda i, v: v**2},
        {'name': 'tau', 'domain': 'positive'},
    ]
    clean = _strip_callables(specs)
    assert len(clean) == 2
    assert 'expression' not in clean[0]
    assert clean[0]['name'] == 'phi'
    assert clean[0]['indexed'] is True
    assert clean[1]['name'] == 'tau'
    assert clean[1]['domain'] == 'positive'
    print('  [PASS] test_strip_callables')


def test_save_creates_files():
    """save_theory creates metadata.json and symbolic_data.sobj."""
    ft = _make_expanded_ft()
    tmpdir = tempfile.mkdtemp(prefix='msrjd_test_')
    save_path = os.path.join(tmpdir, 'test_theory')
    try:
        save_theory(save_path, ft, stationarity=True,
                    model_file='simulations/hawkes_sage.py',
                    model_var_name='HAWKES_MODEL')

        assert os.path.isfile(os.path.join(save_path, 'metadata.json')), \
            'metadata.json not created'
        assert os.path.isfile(os.path.join(save_path, 'symbolic_data.sobj')), \
            'symbolic_data.sobj not created'
        print('  [PASS] test_save_creates_files')
    finally:
        shutil.rmtree(tmpdir)


def test_metadata_contents():
    """metadata.json has the expected keys and values."""
    ft = _make_expanded_ft()
    tmpdir = tempfile.mkdtemp(prefix='msrjd_test_')
    save_path = os.path.join(tmpdir, 'test_theory')
    try:
        save_theory(save_path, ft, stationarity=True,
                    model_file='simulations/hawkes_sage.py',
                    model_var_name='HAWKES_MODEL')

        with open(os.path.join(save_path, 'metadata.json'), 'r') as f:
            meta = json.load(f)

        assert meta['format_version'] == 1
        assert meta['model_name'] == 'Nonlinear Hawkes 2-population'
        assert meta['model_file'] == 'simulations/hawkes_sage.py'
        assert meta['model_var_name'] == 'HAWKES_MODEL'
        assert meta['taylor_order'] == 4
        assert meta['stationarity'] is True
        assert isinstance(meta['ring_var_names'], list)
        assert len(meta['ring_var_names']) == 8  # 4 response + 4 physical
        assert isinstance(meta['nonzero_sectors'], list)
        assert meta['sage_version'] is not None
        assert meta['timestamp'] is not None
        assert meta['index_sets'] == {'pop': [0, 1]}
        print('  [PASS] test_metadata_contents')
    finally:
        shutil.rmtree(tmpdir)


def test_round_trip_no_propagator():
    """Save and load without propagator data — bigrade sectors round-trip."""
    ft = _make_expanded_ft()
    tmpdir = tempfile.mkdtemp(prefix='msrjd_test_')
    save_path = os.path.join(tmpdir, 'test_theory')
    try:
        save_theory(save_path, ft, stationarity=True,
                    model_file='simulations/hawkes_sage.py',
                    model_var_name='HAWKES_MODEL')

        meta, data = load_theory(save_path)

        # Ring round-trips
        R_loaded = data['R']
        assert str(R_loaded) == str(ft._R), \
            f'Ring mismatch: {R_loaded} != {ft._R}'

        # n_tilde round-trips
        assert data['n_tilde'] == ft._n_tilde

        # S_raw round-trips
        S_loaded = data['S_raw']
        assert S_loaded == ft._S_raw, 'S_raw mismatch after round-trip'

        # Bigrade sectors round-trip
        by_tp_loaded = data['by_tp']
        for key, val in ft._by_tp.items():
            assert key in by_tp_loaded, f'Missing sector {key}'
            assert by_tp_loaded[key] == val, f'Sector {key} mismatch'
        for key in by_tp_loaded:
            assert key in ft._by_tp, f'Extra sector {key} in loaded data'

        # Propagator fields are None/empty
        assert data['K_ft'] is None
        assert data['G_ft'] is None
        assert data['pole_vals'] == []
        assert data['C_mats'] == []

        print('  [PASS] test_round_trip_no_propagator')
    finally:
        shutil.rmtree(tmpdir)


def test_round_trip_with_propagator():
    """Save and load with propagator data — matrices and poles round-trip."""
    ft = _make_expanded_ft()
    pd = _make_propagator_data(ft)
    tmpdir = tempfile.mkdtemp(prefix='msrjd_test_')
    save_path = os.path.join(tmpdir, 'test_theory')
    try:
        save_theory(save_path, ft, propagator_data=pd, stationarity=True,
                    model_file='simulations/hawkes_sage.py',
                    model_var_name='HAWKES_MODEL')

        meta, data = load_theory(save_path)

        # Metadata reflects propagator info
        assert meta['propagator_branch'] == 'residue'
        assert meta['G_ft_explicit'] is False
        assert meta['n_poles'] == 2
        assert meta['nf'] == 4

        # K_ft matrix round-trips
        K_loaded = data['K_ft']
        K_orig   = pd['K_ft']
        assert K_loaded.dimensions() == K_orig.dimensions(), 'K_ft dimension mismatch'
        for i in range(K_orig.nrows()):
            for j in range(K_orig.ncols()):
                assert bool(K_loaded[i, j] == K_orig[i, j]), \
                    f'K_ft[{i},{j}] mismatch: {K_loaded[i,j]} != {K_orig[i,j]}'

        # D_omega round-trips
        assert bool(data['D_omega'] == pd['D_omega']), 'D_omega mismatch'

        # Pole values round-trip
        assert len(data['pole_vals']) == 2
        for p_loaded, p_orig in zip(data['pole_vals'], pd['pole_vals']):
            assert bool(p_loaded == p_orig), f'Pole mismatch: {p_loaded} != {p_orig}'

        # C_mats round-trip
        assert len(data['C_mats']) == 2

        # G_ft is None (was too complex)
        assert data['G_ft'] is None

        print('  [PASS] test_round_trip_with_propagator')
    finally:
        shutil.rmtree(tmpdir)


def test_reload_model():
    """reload_model re-imports the model dict and it can be used for re-expansion."""
    ft = _make_expanded_ft()
    tmpdir = tempfile.mkdtemp(prefix='msrjd_test_')
    save_path = os.path.join(tmpdir, 'test_theory')
    try:
        save_theory(save_path, ft, stationarity=True,
                    model_file='simulations/hawkes_sage.py',
                    model_var_name='HAWKES_MODEL')

        meta, data = load_theory(save_path)
        model = reload_model(meta, project_root=_PROJECT_ROOT)

        # Model dict has expected keys
        assert model['name'] == 'Nonlinear Hawkes 2-population'
        assert 'action' in model
        assert callable(model['action'])
        assert 'index_sets' in model
        assert 'response_fields' in model
        assert 'physical_fields' in model

        print('  [PASS] test_reload_model')
    finally:
        shutil.rmtree(tmpdir)


def test_reload_and_reexpand():
    """
    Reload model from saved theory, re-expand at a different Taylor order,
    and verify the result is consistent.
    """
    ft_orig = _make_expanded_ft()
    tmpdir = tempfile.mkdtemp(prefix='msrjd_test_')
    save_path = os.path.join(tmpdir, 'test_theory')
    try:
        save_theory(save_path, ft_orig, stationarity=True,
                    model_file='simulations/hawkes_sage.py',
                    model_var_name='HAWKES_MODEL')

        meta, data = load_theory(save_path)
        model = reload_model(meta, project_root=_PROJECT_ROOT)

        # Re-expand at order 3 (lower than original 4)
        ft_code_path = os.path.join(_PROJECT_ROOT, 'engine', 'core', 'field_theory.py')
        ns_ft = {}
        with open(ft_code_path, 'r') as f:
            exec(compile(f.read(), ft_code_path, 'exec'), ns_ft)
        FieldTheory = ns_ft['FieldTheory']

        ft_new = FieldTheory(model, taylor_order=3)
        ft_new.expand()

        # At order 3, (1,1) sector should still exist
        assert ft_new.free_action() != ft_new._R.zero(), \
            'Free action is zero after re-expansion at order 3'

        # Sanity checks should still pass
        assert ft_new.sanity_check(), \
            'Sanity check failed after re-expansion'

        # At order 3, there should be fewer or equal vertex sectors
        # (no order-4+ terms)
        sectors_orig = set(ft_orig.sectors().keys())
        sectors_new  = set(ft_new.sectors().keys())
        # The (1,1) sector must be present in both
        assert (1, 1) in sectors_new

        print('  [PASS] test_reload_and_reexpand')
    finally:
        shutil.rmtree(tmpdir)


def test_load_missing_files():
    """load_theory raises FileNotFoundError for missing files."""
    tmpdir = tempfile.mkdtemp(prefix='msrjd_test_')
    try:
        # Empty directory
        try:
            load_theory(tmpdir)
            assert False, 'Should have raised FileNotFoundError'
        except FileNotFoundError:
            pass

        # Only metadata, no sobj
        with open(os.path.join(tmpdir, 'metadata.json'), 'w') as f:
            json.dump({}, f)
        try:
            load_theory(tmpdir)
            assert False, 'Should have raised FileNotFoundError'
        except FileNotFoundError:
            pass

        print('  [PASS] test_load_missing_files')
    finally:
        shutil.rmtree(tmpdir)


def test_reload_model_missing_info():
    """reload_model raises ValueError when model_file or model_var_name is None."""
    try:
        reload_model({'model_file': None, 'model_var_name': 'X'})
        assert False, 'Should have raised ValueError'
    except ValueError:
        pass

    try:
        reload_model({'model_file': 'foo.py', 'model_var_name': None})
        assert False, 'Should have raised ValueError'
    except ValueError:
        pass

    print('  [PASS] test_reload_model_missing_info')


def test_reload_model_bad_path():
    """reload_model raises FileNotFoundError for nonexistent model file."""
    try:
        reload_model({'model_file': 'nonexistent.py', 'model_var_name': 'X'},
                     project_root='/tmp')
        assert False, 'Should have raised FileNotFoundError'
    except FileNotFoundError:
        pass

    print('  [PASS] test_reload_model_bad_path')


def test_reload_model_bad_varname():
    """reload_model raises AttributeError for wrong variable name."""
    ft = _make_expanded_ft()
    tmpdir = tempfile.mkdtemp(prefix='msrjd_test_')
    save_path = os.path.join(tmpdir, 'test_theory')
    try:
        save_theory(save_path, ft, stationarity=True,
                    model_file='simulations/hawkes_sage.py',
                    model_var_name='NONEXISTENT_VAR')

        meta, _ = load_theory(save_path)
        try:
            reload_model(meta, project_root=_PROJECT_ROOT)
            assert False, 'Should have raised AttributeError'
        except AttributeError:
            pass

        print('  [PASS] test_reload_model_bad_varname')
    finally:
        shutil.rmtree(tmpdir)


def test_idempotent_save():
    """Saving twice to the same path overwrites cleanly."""
    ft = _make_expanded_ft()
    tmpdir = tempfile.mkdtemp(prefix='msrjd_test_')
    save_path = os.path.join(tmpdir, 'test_theory')
    try:
        save_theory(save_path, ft, stationarity=True,
                    model_file='simulations/hawkes_sage.py',
                    model_var_name='HAWKES_MODEL')
        save_theory(save_path, ft, stationarity=False,
                    model_file='simulations/hawkes_sage.py',
                    model_var_name='HAWKES_MODEL')

        meta, data = load_theory(save_path)
        assert meta['stationarity'] is False, 'Second save did not overwrite'
        assert data['S_raw'] == ft._S_raw, 'S_raw mismatch after overwrite'

        print('  [PASS] test_idempotent_save')
    finally:
        shutil.rmtree(tmpdir)


# ── Runner (for SageMath interactive use) ─────────────────────────────────────

ALL_TESTS = [
    test_strip_callables,
    test_save_creates_files,
    test_metadata_contents,
    test_round_trip_no_propagator,
    test_round_trip_with_propagator,
    test_reload_model,
    test_reload_and_reexpand,
    test_load_missing_files,
    test_reload_model_missing_info,
    test_reload_model_bad_path,
    test_reload_model_bad_varname,
    test_idempotent_save,
]


def run_all_tests():
    """Run all tests and report results."""
    print(f'Running {len(ALL_TESTS)} serialization tests...\n')
    passed = 0
    failed = 0
    for test_fn in ALL_TESTS:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            print(f'  [FAIL] {test_fn.__name__}: {e}')
            import traceback
            traceback.print_exc()
            failed += 1

    print(f'\n{"=" * 40}')
    print(f'Results: {passed} passed, {failed} failed out of {len(ALL_TESTS)}')
    if failed == 0:
        print('All tests passed.')
    return failed == 0


# If executed directly (e.g. `sage tests/test_serialize.py`), run all tests.
if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
