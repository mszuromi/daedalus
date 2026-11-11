"""A δ-subset whose δ-carrying lines close a cycle carries δ(0) and is refused by name.

``Dt x = -a x + g u^2`` with ``u = h + xi`` algebraic and ``xi`` white (``C_uu = 2D δ``): the one-loop
mean of ``x`` is ``g <u^2> / a`` with ``<u^2> = h^2 + 2D δ(0)``, which is infinite.  The δ-subset that
takes both ``u`` lines of the bubble as their δ parts has two δ constraints on the same pair of times;
once one is solved the other is identically zero, i.e. reads δ(0).  Both Phase J routes used to drop
that constraint and keep the subset with its δ weights, so the k = 1 one-loop term came back as
``g 2D = 0.03`` and the k = 2 one as ``9e-4 e^{-tau}``, with no warning.  They now raise
``DeltaCycleError`` (``reason == 'delta_cycle'``) on both routes, analyzer on or off.

The control (``u`` an OU field, no δ part) is finite and unchanged: its one-loop mean is
``g D / (a * 1) = 0.015``."""
import numpy as np
import pytest

pytest.importorskip('sage.all')

import daedalus as dd                                                     # noqa: E402
from daedalus.api import compute_cumulants                                # noqa: E402
from daedalus.api.theory import TemporalTheoryBuilder                     # noqa: E402

ENV = 'DAEDALUS_PHASE_J_ENGINE'
PARS = dict(h=0.5, g=0.3, D=0.05, a=1.0)
TAU = np.array([0.0, 0.5])


def _model(white):
    b = (TemporalTheoryBuilder('delta cycle ' + ('white' if white else 'ou'))
         .population('pop', size=1)
         .physical_field('x', population='pop').physical_field('u', population='pop'))
    for k, v in PARS.items():
        b = b.parameter(k, default=v, domain='positive')
    u_eq = 'u[i] - h' if white else 'Dt*u[i] + u[i] - h'
    act = f'xt[i]*(Dt*x[i] + a*x[i] - g*u[i]^2) + ut[i]*({u_eq}) - D*ut[i]^2'
    return (b.set_action_text(f'sum( {act} for i in pop)')
            .set_mf_equation('xstar', 'g*h^2/a').set_mf_equation('ustar', 'h')
            .build())


@pytest.fixture(scope='module')
def white():
    return _model(True)


@pytest.fixture(scope='module')
def ou():
    return _model(False)


def _run(m, k, max_ell=1, preflight=True):
    ext = [('x', 1)] * k
    return compute_cumulants(m, k=k, max_ell=max_ell, external_fields=ext, tau_grid=TAU,
                             parallel=False, verbose=False, preflight=preflight)


@pytest.mark.parametrize('engine', ['statespace', 'poles'])
@pytest.mark.parametrize('preflight', [True, False])
def test_white_one_loop_mean_is_refused_by_name(white, engine, preflight, monkeypatch):
    monkeypatch.setenv(ENV, engine)
    with pytest.raises(dd.DeltaCycleError) as ei:
        _run(white, 1, preflight=preflight)
    err = ei.value
    assert err.reason == 'delta_cycle'
    assert 'delta_cycle' in str(err) and 'δ(0)' in str(err)
    # the bubble's two u lines, both taken as their δ parts, between the same two vertices
    assert len(err.delta_edges) == 2
    assert len({frozenset((u, v)) for (u, v, _p, _r) in err.delta_edges}) == 1
    assert isinstance(err, ValueError)


@pytest.mark.parametrize('engine', ['statespace', 'poles'])
def test_white_one_loop_covariance_is_refused_by_name(white, engine, monkeypatch):
    monkeypatch.setenv(ENV, engine)
    with pytest.raises(dd.DeltaCycleError):
        _run(white, 2, preflight=False)


def test_white_tree_level_is_finite_and_runs(white):
    # no loop, no δ cycle: C_xx(0) = g^2 (2h)^2 2D / (2a) = 0.0045
    r = _run(white, 2, max_ell=0)
    np.testing.assert_allclose(np.real(r['C_tau_by_ell'][0])[0], 0.0045, rtol=1e-12)


@pytest.mark.parametrize('engine', ['statespace', 'poles'])
def test_ou_control_is_finite(ou, engine, monkeypatch):
    monkeypatch.setenv(ENV, engine)
    r = _run(ou, 1)
    # <x> one loop = g <du^2> / a, with <du^2> = D / 1 for the unit-rate OU field
    np.testing.assert_allclose(np.real(r['C_tau_by_ell'][1]), 0.015, rtol=1e-10)

