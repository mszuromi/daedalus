"""A propagator with poles and an entry growing at large omega is refused by name.

``Dt x = -a x + u`` (one pole), ``u = h + xi`` algebraic with white ``xi`` (``C_uu = 2D δ``), and
``y = Dt u``: ``G_yu(omega) = i omega`` grows (``deg P - deg Q = 1``), i.e. ``G_yu(t) = δ'(t)``.
Neither Phase J route represents that part (the poles hold the strictly proper part, ``D_delta`` the
constant one).  With the analyzer off (or ``G3_IMPROPER`` downgraded) both routes used to drop it:
the exact propagator tier's ``deg P <= deg Q`` assertion was caught by its broad fallback, the
numerical tier zeroed the entry, and only a ``PoleResidueWarning`` was left (the state-space route
declined the model, ``index_gt_1``, and handed it to the pole engine).  The tree-level
``C_yx(tau) = d/dtau C_ux(tau)`` is ``2 D a e^{-a/2} = 0.0607`` in size at one of ``tau = -0.5, 0.5``
(the same run gives ``C_ux(0.5) = 2D e^{-a/2}``); it came back as 0 at both.  Both routes now raise
``ImproperPropagatorError`` (``reason == 'improper_propagator'``) naming ``G[dy1, ut1]``.

Unchanged: with the analyzer on, ``AnalyzerError [G3_IMPROPER]`` stops the run first; a propagator with
no pole at all (``y = Dt x``, ``x`` algebraic) still raises ``PoleFreePropagatorError``."""
import re

import numpy as np
import pytest

pytest.importorskip('sage.all')

import daedalus as dd                                                     # noqa: E402
from daedalus.api import compute_cumulants                                # noqa: E402
from daedalus.api.theory import TemporalTheoryBuilder                     # noqa: E402

ENV = 'DAEDALUS_PHASE_J_ENGINE'
TAU = np.array([-0.5, 0.5])
XS = '(sqrt(1 + 4*g*h) - 1)/(2*g)'


def _build(name, act, fields, mf):
    b = TemporalTheoryBuilder(name).population('pop', size=1)
    for f in fields:
        b = b.physical_field(f, population='pop')
    for k, v in dict(h=0.1, g=0.1, D=0.05, a=1.0).items():
        b = b.parameter(k, default=v, domain='positive')
    txt = re.sub(r'\b([xyu])\b', r'\1[i]', act)
    txt = re.sub(r'\b([xyu])t\b', r'\1t[i]', txt)
    b = b.set_action_text(f'sum( {txt} for i in pop)')
    for f, v in mf.items():
        b = b.set_mf_equation(f + 'star', v)
    return b.build()


@pytest.fixture(scope='module')
def with_poles():
    return _build('improper with poles',
                  'xt*(Dt*x + a*x - u) + ut*(u - h) + yt*(y - Dt*u) - D*ut^2',
                  ['x', 'y', 'u'], dict(x='h/a', y='0*h', u='h'))


@pytest.fixture(scope='module')
def pole_free():
    return _build('improper pole free', 'xt*(x - h + g*x^2) + yt*(y - Dt*x) - D*xt^2',
                  ['x', 'y'], dict(x=XS, y='0*h'))


def _run(m, k, ext, max_ell=1, **kw):
    return compute_cumulants(m, k=k, max_ell=max_ell, external_fields=ext, tau_grid=TAU,
                             parallel=False, verbose=False, **kw)


CASES = [(1, [('y', 1)], 1), (2, [('y', 1), ('x', 1)], 0), (2, [('y', 1), ('x', 1)], 1)]
PREFLIGHT = [dict(preflight=False),
             dict(policy=dd.Policy(downgrade=('G3_IMPROPER',)))]


@pytest.mark.filterwarnings('ignore')
@pytest.mark.parametrize('engine', ['statespace', 'poles'])
@pytest.mark.parametrize('k, ext, max_ell', CASES)
@pytest.mark.parametrize('pf', PREFLIGHT, ids=['preflight_off', 'g3_downgraded'])
def test_improper_entry_with_poles_is_refused_by_name(with_poles, engine, k, ext, max_ell, pf,
                                                     monkeypatch):
    monkeypatch.setenv(ENV, engine)
    with pytest.raises(dd.ImproperPropagatorError) as ei:
        _run(with_poles, k, ext, max_ell=max_ell, **pf)
    err = ei.value
    assert err.reason == 'improper_propagator' and isinstance(err, ValueError)
    assert err.entries == ((1, 2),) and err.growth == {(1, 2): 1}
    assert 'improper_propagator' in str(err) and 'G[dy1, ut1]' in str(err)


def test_analyzer_on_still_stops_it_first(with_poles):
    with pytest.raises(dd.AnalyzerError, match='G3_IMPROPER'):
        _run(with_poles, 1, [('y', 1)])


@pytest.mark.parametrize('engine', ['statespace', 'poles'])
def test_pole_free_improper_keeps_its_own_error(pole_free, engine, monkeypatch):
    from daedalus.engine.integration.time_domain.final_integral import (
        PoleFreePropagatorError)
    monkeypatch.setenv(ENV, engine)
    with pytest.raises(PoleFreePropagatorError):
        _run(pole_free, 1, [('y', 1)], preflight=False)


def test_guard_needs_poles_and_an_improper_entry():
    from daedalus.engine.integration.time_domain.final_integral import (
        refuse_improper_propagator)
    refuse_improper_propagator({'improper_entries': ((1, 0, 1),), 'pole_vals': []})
    refuse_improper_propagator({'improper_entries': (), 'pole_vals': [1j]})
    with pytest.raises(dd.ImproperPropagatorError) as ei:
        refuse_improper_propagator({'improper_entries': ((1, 0, None),), 'pole_vals': [1j],
                                    'nf': 2, 'ring_gen_names': ['xt', 'yt', 'x', 'y']})
    assert 'G[y, xt]' in str(ei.value) and ei.value.growth == {(1, 0): None}


def test_numerical_growth_probe():
    from daedalus.api._propagator import _improper_entries_probe
    # K = [[i w + 1, 0], [-(i w), 1]]: G = [[1/(i w + 1), 0], [i w/(i w + 1), 1]], proper
    proper = lambda w: np.array([[1j * w + 1, 0], [-1j * w, 1]])
    # K = [[1, 0], [-(i w), 1]]: G_10 = i w grows
    growing = lambda w: np.array([[1, 0], [-1j * w, 1]], dtype=complex)
    assert _improper_entries_probe(proper, 2) == ()
    assert _improper_entries_probe(growing, 2) == ((1, 0, None),)
