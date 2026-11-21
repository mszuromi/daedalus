"""
Cost honesty of the lazy evaluations behind ``spectra`` and the delay route.

* ``spectra`` integrates a loop order numerically when it has no
  matrix-exponential triple (a subset the state-space route declined, e.g. an
  ill-conditioned high Pade order, runs on mode sums, and every evaluation of
  its callable is a mode sum).  It times one evaluation, predicts a lower bound
  of the integration's wall time (``result['spectra']['quad_cost']``) and says
  so before it integrates when the prediction exceeds
  ``spectra.QUAD_ADVISORY_S``.
* The delay record lists the share of each order's subsets on the
  state-space route (``statespace_share_by_order``).
"""
from __future__ import annotations

import os
import sys
import warnings

import numpy as np
import pytest

pytest.importorskip('sage.all')

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import daedalus as dd                                                          # noqa: E402
from daedalus.api import compute_cumulants                                     # noqa: E402
from daedalus.api import spectra as SP                                         # noqa: E402
from daedalus.api.noise import Gaussian                                        # noqa: E402
from daedalus.api.sde import SDE                                               # noqa: E402


def _ou():
    return (SDE('cost honesty OU').parameter('mu', default=0.9)
            .parameter('D', default=0.4).field('x')
            .drift('x', '-mu*x')
            .noise('xi', Gaussian(var='2*D'), couples={'x': 1})).build()


@pytest.fixture(scope='module')
def ou_result():
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return compute_cumulants(_ou(), k=2, max_ell=0, external_fields=[('x', 1), ('x', 1)],
                                 tau_grid=np.array([0.0, 1.0]), verbose=False)


def test_quadrature_cost_is_predicted(ou_result):
    with warnings.catch_warnings(record=True) as W:
        warnings.simplefilter('always')
        q = dd.spectra(ou_result, np.array([0.0, 1.0]), np.array([2.0]), method='quad')
    qc = q['quad_cost']
    assert qc['orders'] == [0] and qc['eval_s_by_ell'][0] > 0
    # 1 nonzero frequency (4), S(0) (2), both infinite sides (3), 1 window (4)
    assert qc['predicted_s_min'] == pytest.approx(
        qc['eval_s_by_ell'][0] * 13 * SP._QUAD_EVALS_MIN, rel=1e-3, abs=0.06)
    assert 'advisory' not in qc
    assert not [w for w in W if issubclass(w.category, SP.SpectraWarning)]


def test_quadrature_advisory_fires_before_integrating(monkeypatch, ou_result):
    monkeypatch.setattr(SP, 'QUAD_ADVISORY_S', 0.0)
    with warnings.catch_warnings(record=True) as W:
        warnings.simplefilter('always')
        q = dd.spectra(ou_result, np.array([0.5]), None, method='quad')
    msg = [str(w.message) for w in W if issubclass(w.category, SP.SpectraWarning)]
    assert msg and 'predicted to take at least' in msg[0]
    assert q['quad_cost']['advisory'] == msg[0]


def test_exact_route_has_no_quadrature_cost(ou_result):
    q = dd.spectra(ou_result, np.array([0.5]), None)
    assert q['quad_cost'] is None


def test_delay_record_lists_the_state_space_share():
    m = (SDE('cost honesty delayed OU').parameter('a', default=1.1)
         .parameter('b', default=-0.4).parameter('d', default=0.7).parameter('s', default=0.5)
         .field('x').drift('x', '-a*x + b*x(t - d)')
         .noise('xi', Gaussian(var='s^2'), into={'x': 1})).build()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        r = compute_cumulants(m, k=2, max_ell=0, external_fields=[('x', 1), ('x', 1)],
                              tau_grid=np.array([0.0, 1.0]), verbose=False,
                              delay={'tree': 'pade', 'order': 4})
    share = r['delay']['statespace_share_by_order']
    assert set(share) >= {2, 4} and all(0.0 <= v <= 1.0 for v in share.values())
    assert share[4] == 1.0
