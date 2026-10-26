"""Resident state after a run (audit MEM-08) and the saddle-name lookup of the
model editor (audit GEN-04)."""
import gc

import numpy as np
import pytest

pytest.importorskip('sage.all')

from api import compute_cumulants                        # noqa: E402
from api.noise import Gaussian                           # noqa: E402
from api.sde import SDE                                  # noqa: E402
from engine.integration.time_domain import statespace as SS   # noqa: E402


def _cubic():
    return (SDE('resident state cubic').parameter('mu', default=1.2)
            .parameter('g', default=0.5).parameter('D', default=0.4)
            .field('x').drift('x', '-mu*x - g*x^2')
            .noise('xi', Gaussian(var='2*D'), couples={'x': 1}).build())


PTS = [(0.0, 0.0, 0.0), (0.0, 0.3, 1.1), (0.0, -0.4, 0.0)]


def _solve_data_entries():
    with SS._REAL_LOCK:
        cached = [v[1] for v in SS._REAL_CACHE.values()]
    return sum(len(o._solve_data) for o in cached if o is not None)


def test_discard_run_releases_the_resolvent_data_and_stays_bit_identical():
    kw = dict(k=3, max_ell=1, external_fields=[('x', 1)] * 3, verbose=False,
              use_cache=False, parallel=False)
    SS._clear_caches()
    keep = compute_cumulants(_cubic(), tau_grid=np.array([0.0]), **kw)
    want = [complex(keep['total_C'](*p)) for p in PTS]
    assert _solve_data_entries() > 0                   # a kept run still holds it
    SS._clear_caches()
    gone = compute_cumulants(_cubic(), points=PTS, discard=True, **kw)
    assert _solve_data_entries() == 0
    assert [complex(gone['total_C'](*p)) for p in PTS] == want


def test_symbolic_cumulant_does_not_pin_the_pipeline_result():
    from api import symbolic_loop as SL
    r = SL.symbolic_cumulant(_cubic(), 2, 1, external_fields=[('x', 1)] * 2)
    gc.collect()
    assert 'result' not in r._ctx
    pinned = [o for o in gc.get_objects() if type(o) is dict
              and isinstance(o.get('total_C_by_ell'), dict) and 'phase_j_by_ell' in o]
    assert not pinned


def test_editor_names_a_legacy_saddle_equation_by_the_declared_natural_name():
    from api.ui.main import _legacy_mf_equation_rows
    spec = {'physical_fields': [{'name': 'x', 'natural_name': 'x', 'population': 'E'},
                                {'name': 'y', 'population': None}],
            'parameters': [{'name': 'rate_mf', 'natural_name': 'x'}],
            'mf_equations': [{'saddle': 'rate_mf', 'rhs': 'h - x'},
                             {'saddle': 'ystar', 'rhs': '1'}]}
    assert _legacy_mf_equation_rows(spec) == [
        {'lhs': 'x[i]', 'rhs': 'h - x', 'population': 'E'},
        {'lhs': 'y[i]', 'rhs': '1', 'population': None}]
