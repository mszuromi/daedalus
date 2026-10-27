"""k >= 3 results are evaluated lazily, after ``compute_cumulants`` returned
(``total_C_by_ell[ell](t1, t2, ...)`` runs the Phase J integrals then).
Those calls resume the run's memory scope: the same footprint baseline and
budget reading, and a running plan scope, so a point that grows the process
past the budget is refused (``PhaseJResourceError`` naming the point)
instead of growing without limit (a cubic+quartic k = 3 three-loop point
passed 9.5 GB under an 8 GB budget)."""
import numpy as np
import pytest

pytest.importorskip('sage.all')

from daedalus.api import compute_cumulants
from daedalus.api.noise import Gaussian
from daedalus.api.sde import SDE
from daedalus.engine.integration.time_domain import final_integral as FI


def _cubic():
    return (SDE('lazy point cubic').parameter('mu', default=1.2)
            .parameter('g', default=0.5).parameter('D', default=0.4)
            .field('x').drift('x', '-mu*x - g*x^2')
            .noise('xi', Gaussian(var='2*D'), couples={'x': 1}).build())


def _run(**kw):
    return compute_cumulants(_cubic(), k=3, max_ell=1,
                             external_fields=[('x', 1)] * 3,
                             tau_grid=np.array([0.0]), parallel=False,
                             verbose=False, use_cache=False, **kw)


def test_lazy_point_values_unchanged(monkeypatch):
    monkeypatch.setenv('PHASE_J_MEM_BUDGET_GB', '4')
    r = _run()
    fn = r['total_C_by_ell'][1]
    assert isinstance(fn, FI._ResumedRunCallable)
    pt = (0.0, 0.3, 1.1)
    assert complex(fn(*pt)) == complex(fn._fn(*pt))   # the wrapped callable
    assert complex(r['total_C'](*pt)) == complex(r['total_C']._fn(*pt))
    assert fn.__wrapped__ is fn._fn


def test_lazy_point_refused_over_budget(monkeypatch):
    monkeypatch.setenv('PHASE_J_MEM_BUDGET_GB', '1')
    r = _run()
    fp0, budget = r['total_C_by_ell'][1]._state[:2]
    assert budget == pytest.approx(1e9)
    # the process "has grown" by 2 GB since the run started
    monkeypatch.setattr(FI, 'process_footprint_bytes', lambda: fp0 + 2e9)
    with pytest.raises(FI.PhaseJResourceError) as ei:
        r['total_C_by_ell'][1](0.0, 0.3, 1.1)
    msg = str(ei.value)
    assert 'total_C_by_ell[1](0, 0.3, 1.1)' in msg
    assert 'lazy point cubic k=3 max_ell=1' in msg
    assert ei.value.estimate['growth_bytes'] == pytest.approx(2e9)


def test_lazy_point_without_a_budget_is_unwrapped(monkeypatch):
    monkeypatch.setenv('PHASE_J_MEM_BUDGET_GB', 'off')
    r = _run()
    assert not isinstance(r['total_C_by_ell'][1], FI._ResumedRunCallable)
