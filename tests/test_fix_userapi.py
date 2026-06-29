"""Regression tests for the user-facing API fixes (KEY=userapi).

Covers:

  FIX A — ``TheoryUI.load()`` must not drop a LEGACY ``indexed=`` /
  ``'vector'`` / ``'matrix'`` annotation.  Before the fix, ``load()``
  only mapped the new ``indexed_by`` key onto the index dropdowns and
  ignored the legacy ``indexed`` key entirely, so ``_collect()``
  re-emitted the parameter / kernel as SCALAR — a matrix parameter
  silently became scalar on a UI load → save cycle.

  FIX B — ``daedalus.run`` auto-builds ``external_fields`` when the
  caller gave none and the METADATA recommendation is stale / missing.
  The old build collapsed every leg onto population 1 (``[(f0, 1)] * k``)
  → the wrong correlator for a MULTI-population model.  The fix spreads
  the legs over the first field's populations.  Also checks the corrected
  ``recommended_external_fields`` in the shipped spike-reset theory.

  MINORS — spatial ``k=1`` with ``Config(output='moment')`` returns the
  raw first moment (the mean) instead of raising a misleading
  "spatial k≥3 not implemented" error, and ``result['_resolved']``
  reports the ``max_ell`` actually computed (0) for spatial k=1.

Run:
    sage -python -m pytest tests/test_fix_userapi.py -q
"""
import os
import sys
import tempfile

import matplotlib
matplotlib.use('Agg')

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import pytest  # noqa: E402

import daedalus as dd  # noqa: E402
from api.ui.main import TheoryUI  # noqa: E402
from api.theory_serialize import (  # noqa: E402
    render_theory_file, load_spec_from_file,
)


# ── FIX A: legacy indexed survives a UI load → _collect round-trip ───────

def _collected(spec_or_path):
    """Load a spec into a fresh headless TheoryUI and return _collect()."""
    ui = TheoryUI()
    ui.load(spec_or_path)
    return ui._collect()


def test_legacy_indexed_survives_no_named_populations():
    """linear_hawkes.theory.py carries n_populations=2 (no NAMED
    populations) with legacy ``indexed=True`` (E) and ``indexed='matrix'``
    (w).  The index dropdowns have no population names to hold, so the
    legacy value must round-trip through _collect() as a re-emitted
    ``indexed`` key — NOT silently degrade to scalar."""
    out = _collected('theories/linear_hawkes.theory.py')
    by_name = {p['name']: p for p in out['parameters']}

    # Vector parameter E.
    assert by_name['E'].get('indexed') is True, \
        f"E lost its vector indexing: {by_name['E']}"
    # Matrix parameter w — the headline silent-corruption case.
    assert by_name['w'].get('indexed') == 'matrix', \
        f"w silently degraded to scalar: {by_name['w']}"
    # A genuinely scalar parameter must stay scalar.
    assert by_name['tau'].get('indexed') in (None, False)
    assert not by_name['tau'].get('indexed_by')


def test_legacy_indexed_full_disk_round_trip():
    """The whole load → _collect → render → reload chain (i.e. what an
    actual UI 'Save' does) preserves the matrix/vector shape on disk."""
    out = _collected('theories/linear_hawkes.theory.py')
    src = render_theory_file(out)
    with tempfile.NamedTemporaryFile('w', suffix='.theory.py',
                                     delete=False) as f:
        f.write(src)
        path = f.name
    try:
        reloaded = load_spec_from_file(path)
    finally:
        os.unlink(path)
    by_name = {p['name']: p for p in reloaded['parameters']}
    assert by_name['E'].get('indexed') is True
    assert by_name['w'].get('indexed') == 'matrix'


def test_legacy_indexed_maps_to_dropdowns_when_named_pops_exist():
    """When NAMED populations exist, legacy ``indexed=`` is upgraded to
    the modern ``indexed_by`` list (mapped onto the dropdowns) rather
    than re-emitted as the legacy key."""
    spec = {
        'name': 'Legacy+Pops',
        'populations': [{'name': 'A', 'size': 2}, {'name': 'B', 'size': 3}],
        'n_populations': 2,
        'physical_fields': [],
        'parameters': [
            {'name': 'vparam', 'indexed': True, 'default': [1, 1]},
            {'name': 'mparam', 'indexed': 'matrix', 'default': [[1, 2], [3, 4]]},
        ],
        'kernels': [{'name': 'gk', 'indexed': 'matrix', 'time_expr': 'exp(-t)'}],
    }
    out = _collected(spec)
    by_name = {p['name']: p for p in out['parameters']}
    assert by_name['vparam'].get('indexed_by') == ['A']
    assert by_name['mparam'].get('indexed_by') == ['A', 'B']
    kby = {k['name']: k for k in out['kernels']}
    assert kby['gk'].get('indexed_by') == ['A', 'B']


def test_modern_indexed_by_still_round_trips():
    """The pre-existing modern path (named pops + indexed_by) is
    unaffected by the fix."""
    out = _collected('theories/multipopulation_test.theory.py')
    by_name = {p['name']: p for p in out['parameters']}
    assert by_name['tauE'].get('indexed_by') == ['E']
    assert by_name['wEE'].get('indexed_by') == ['E', 'E']
    kby = {k['name']: k for k in out['kernels']}
    assert kby['gEE'].get('indexed_by') == ['E', 'E']


# ── FIX B: multi-pop external_fields auto-build keeps distinct legs ──────

def test_auto_external_fields_spreads_over_populations():
    """A multi-population field gets distinct (field, pop) legs; a
    single-population field keeps the old [(f0, 1)] * k behaviour."""
    m_multi, _ = dd.load_theory('linear_hawkes')          # field 'n', 2 pops
    assert dd._auto_external_fields(m_multi, 2) == [('n', 1), ('n', 2)]
    # k > npop cycles, but still uses BOTH populations (not all pop 1).
    assert dd._auto_external_fields(m_multi, 3) == [('n', 1), ('n', 2), ('n', 1)]

    m_single, _ = dd.load_theory('ou_quartic')            # scalar field 'x'
    assert dd._auto_external_fields(m_single, 2) == [('x', 1), ('x', 1)]


def test_stale_recommendation_is_rejected_then_rebuilt():
    """A METADATA recommendation naming a non-existent field (the exact
    bug fixed in single_population_spike_reset_test) is detected as
    invalid, so run() rebuilds distinct legs instead of feeding garbage
    through."""
    m, _ = dd.load_theory('single_population_spike_reset_test')
    # Stale recommendation 'nE' (the real field is 'n') is invalid …
    assert not dd._ext_is_valid(m, [('nE', 1), ('nE', 2)], 2)
    # … and the rebuild spreads over both populations.
    assert dd._auto_external_fields(m, 2) == [('n', 1), ('n', 2)]
    # The corrected, shipped recommendation IS valid.
    assert dd._ext_is_valid(m, [('n', 1), ('n', 2)], 2)


def test_spike_reset_metadata_recommendation_is_corrected():
    """The shipped theory's recommended_external_fields names the REAL
    field 'n' (was the stale 'nE')."""
    _, mod = dd.load_theory('single_population_spike_reset_test')
    rec = dd._meta(mod, 'recommended_external_fields')
    assert rec == [('n', 1), ('n', 2)], rec
    valid = set(dd.field_names(dd.load_theory(
        'single_population_spike_reset_test')[0]))
    assert all(name in valid for name, _ in rec)


# ── MINORS: spatial k=1 moment + max_ell lockstep ───────────────────────

def test_spatial_k1_moment_does_not_raise():
    """spatial k=1 with output='moment' returns the raw first moment
    (the mean φ*), not a misleading 'spatial k≥3 not implemented'
    error.  central_moment of a 1-point function is 0 by definition."""
    m, mod = dd.load_theory('linear_diffusion_test')
    assert dd.is_spatial(m)
    res = dd.run(m, dd.Config(k=1, output='moment', max_ell=0), mod)
    assert res.get('moment') is not None
    assert res.get('output_kind') == 'moment'
    res_c = dd.run(m, dd.Config(k=1, output='central_moment', max_ell=0), mod)
    assert res_c.get('moment') == 0.0


def test_spatial_k1_resolved_max_ell_matches_computed():
    """spatial k=1 cannot do loop tadpoles, so a requested max_ell>0 is
    dropped to 0 — and _resolved must report the 0 that was actually
    computed, not the request."""
    m, mod = dd.load_theory('linear_diffusion_test')
    res = dd.run(m, dd.Config(k=1, max_ell=2), mod)
    assert res['_resolved']['max_ell'] == 0


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-q']))
