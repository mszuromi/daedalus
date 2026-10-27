"""
A diagram Phase J cannot evaluate is never dropped silently.

Before this, a diagram whose ``integrate_diagram`` returned a failure status
was recorded in ``phase_j_by_ell[ell]['skipped_kernel_ids']`` and simply left
out of ``C_tau``; ``compute_cumulants`` raised nothing and said nothing.  Now
every route (per-diagram, the near-degenerate-pole deformation)
returns ``result['status'] == 'incomplete'`` with the diagrams and the reason
in ``result['phase_j_incomplete']``, warns with ``PhaseJIncompleteWarning``,
adds the analyzer finding ``G6_DIAGRAMS_SKIPPED``, and ``policy={'mode':
'strict'}`` raises ``PhaseJIncompleteError``.

The broken diagram is made deliberately: the per-diagram integrator is wrapped
so that it reports a failure for every loop diagram.
"""
from __future__ import annotations

import os
import sys
import warnings

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from daedalus.api import compute_cumulants                                   # noqa: E402
from daedalus.api.compute import PhaseJIncompleteError, PhaseJIncompleteWarning  # noqa: E402
from daedalus.api.cumulant import Cumulant                                   # noqa: E402
from daedalus.api.model import TemporalModelBuilder                          # noqa: E402
import daedalus.engine.integration.time_domain.pipeline as PL                # noqa: E402

TAU = np.array([0.0, 1.0])
REASON = 'deliberately broken diagram (test)'


def _ou_quartic():
    return (TemporalModelBuilder('Never silent OU quartic')
            .physical_field('x')
            .parameter('mu', default=1.0, domain='positive')
            .parameter('eps', default=0.02, domain='positive')
            .parameter('D', default=1.0, domain='positive')
            .set_action_text('xt*((Dt+mu)*x + eps*x^3) - D*xt^2')
            .mf_residual(lhs='(Dt+mu)*x', rhs='-eps*x^3')
            .build())


def _chain(gap):
    """x <- y <- z with rates mu, mu + gap, 1: a near-degenerate pole pair,
    so ``compute_cumulants`` takes the pole-cluster deformation path."""
    b = TemporalModelBuilder('Never silent chain')
    for f in ('x', 'y', 'z'):
        b.physical_field(f)
    for p, d in (('mu', 1.0), ('g1', 1.0 + gap), ('g2', 2.0), ('eps', 0.02),
                 ('D', 1.0)):
        b.parameter(p, default=d, domain='positive')
    b.set_action_text('xt*((Dt+mu)*x - y + eps*x^3) + yt*((Dt+g1)*y - z) '
                      '+ zt*((Dt+g2)*z) - D*zt^2')
    b.mf_residual(lhs='(Dt+mu)*x', rhs='y - eps*x^3')
    b.mf_residual(lhs='(Dt+g1)*y', rhs='z')
    b.mf_residual(lhs='(Dt+g2)*z', rhs='0')
    b.stability_analysis(True)
    return b.build()


@pytest.fixture
def broken_loops(monkeypatch):
    """Every loop diagram fails in ``integrate_diagram``; tree diagrams run."""
    real = PL.integrate_diagram

    def fake(*, typed_diagram, **kw):
        if PL._loop_number_from_graph(typed_diagram) > 0:
            return {'status': 'failed', 'contribution': None,
                    'reason': REASON}
        return real(typed_diagram=typed_diagram, **kw)

    monkeypatch.setattr(PL, 'integrate_diagram', fake)


def _run(model, field='x', **kw):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        res = compute_cumulants(model, k=2, max_ell=1,
                                external_fields=[(field, 1), (field, 1)],
                                tau_grid=TAU, parallel=False, verbose=False,
                                use_cache=False, **kw)
    return res, [w for w in caught
                 if issubclass(w.category, PhaseJIncompleteWarning)]


def _assert_reported(res, warns):
    assert res['status'] == 'incomplete'
    fail = res['phase_j_incomplete']
    assert list(fail) == [1]                       # the tree is complete
    assert fail[1] and all(REASON in r['reason'] for r in fail[1])
    assert len(warns) == 1 and REASON in str(warns[0].message)
    assert res['analysis'].has('G6_DIAGRAMS_SKIPPED')
    assert not Cumulant.from_result(res).complete


def test_complete_run_says_ok():
    res, warns = _run(_ou_quartic())
    assert res['status'] == 'ok' and res['phase_j_incomplete'] == {}
    assert not warns
    assert not res['analysis'].has('G6_DIAGRAMS_SKIPPED')


def test_per_diagram_route_reports_a_broken_diagram(broken_loops):
    _assert_reported(*_run(_ou_quartic()))


def test_pole_cluster_route_reports_a_broken_diagram(broken_loops,
                                                    monkeypatch):
    # the pole engine's deformation (S3: bypassed on the default route)
    monkeypatch.setenv('DAEDALUS_PHASE_J_ENGINE', 'poles')
    res, warns = _run(_chain(1e-6), policy={'allow_markov_embedding': True})
    assert (res.get('pole_clusters') or {}).get('status') == 'resolved'
    _assert_reported(res, warns)


def test_strict_policy_raises(broken_loops):
    with pytest.raises(PhaseJIncompleteError) as ei:
        _run(_ou_quartic(), policy={'mode': 'strict'})
    err = ei.value
    assert REASON in str(err)
    assert err.result['status'] == 'incomplete' and 1 in err.failures
