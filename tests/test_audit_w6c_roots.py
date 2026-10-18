"""
tests/test_audit_w6c_roots.py
=============================
Mean-field roots under an exact rescaling of the field (audit W6 follow-up).

The cubic drift ``-(x - 0.3 s)(x + 1.1 s)(x - 1.7 s) / s^2`` with noise variance
``0.01 s^2`` is the same model in field units ``s``: roots ``-1.1 s, 0.3 s,
1.7 s`` (the outer two stable, linearization rate 3.92 at both).  For every
``s`` in 1e-8 .. 1e8 the selectable roots, the reported unsearched roots, the
selection and ``C(0) / s^2`` must be the same.

* Every root is selectable in every unit, with or without a declared domain
  (a negative root of a field without a domain was selectable at s = 1 only,
  where a seed happened to overshoot to it; ``domain='real'`` lost ``1.7 s``
  at s >= 1e5: the second-pass seed boxes are now repeated for the root scales
  they discover).
* With stability analysis off, the linearly stable roots come first: index 0
  is ``-1.1 s`` and index 1 ``1.7 s`` at every scale, the unstable ``0.3 s``
  last.
* Selecting the unstable root is refused by name with the analyzer off too
  (it surfaced as ``PoleFreePropagatorError``).

Run:  PYTHONHASHSEED=0 sage -python -m pytest tests/test_audit_w6c_roots.py -q
"""
from __future__ import annotations

import os
import sys
import warnings

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from api import compute_cumulants                          # noqa: E402
from api.noise import Gaussian                             # noqa: E402
from api.sde import SDE                                    # noqa: E402
from api._mean_field_dae import (                          # noqa: E402
    UnstableSaddleError, solve_mean_field_dae)

RUN = dict(parallel=False, verbose=False, use_cache=False)
SCALES = (1e-8, 1e-5, 1.0, 1e5, 1e8)
LAM = (1.7 - 0.3) * (1.7 + 1.1)        # rate at both stable roots: 3.92


def _cubic(s, domain=None):
    sde = SDE(f'w6c cubic {s:g} {domain}')
    sde = sde.field('x', domain=domain) if domain else sde.field('x')
    return (sde.drift('x', '-(x - %r)*(x + %r)*(x - %r)/%r'
                      % (0.3 * s, 1.1 * s, 1.7 * s, s * s))
            .noise('xi', Gaussian(var=repr(0.01 * s * s)), couples={'x': 1})
            .build())


def _roots(model):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        r = solve_mean_field_dae(model, {})
    return ([rr['values']['xstar'][0] for rr in r['mf_all_roots']],
            [u['xstar'][0] for u in r['mf_unsearched_roots']])


@pytest.mark.parametrize('domain', [None, 'real'])
@pytest.mark.parametrize('s', SCALES)
def test_roots_and_their_order_scale_exactly(s, domain):
    sel, unsearched = _roots(_cubic(s, domain))
    assert np.allclose(np.array(sel) / s, [-1.1, 1.7, 0.3], rtol=1e-13, atol=0)
    assert unsearched == []


@pytest.mark.parametrize('s', (1e-5, 1.0, 1e5))
def test_selection_and_covariance_scale_exactly(s):
    """fixed_point_index 0 and 1 are the two stable roots at every scale and
    C(0) = 0.01 s^2 / (2 * 3.92) at both."""
    for idx, root in ((0, -1.1), (1, 1.7)):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            r = compute_cumulants(_cubic(s), k=2, max_ell=0, external_fields=[('x', 1)] * 2,
                                  tau_grid=np.array([0.0]), fixed_point_index=idx, **RUN)
        assert r['mf_values']['xstar'][0] == pytest.approx(root * s, rel=1e-14)
        assert complex(r['C_tau'][0]).real == pytest.approx(0.01 * s * s / (2 * LAM), rel=1e-12)


@pytest.mark.parametrize('s', (1e-5, 1.0))
def test_unstable_selection_is_refused_by_name(s):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        with pytest.raises(UnstableSaddleError, match=r'fixed_point_index=0 \(stable roots: \[0, 1\]\)'):
            compute_cumulants(_cubic(s), k=2, max_ell=0, external_fields=[('x', 1)] * 2,
                              tau_grid=np.array([0.0]), preflight=False,
                              fixed_point_index=2, **RUN)
        from api.analyze import AnalyzerError
        with pytest.raises(AnalyzerError, match='G2_UNSTABLE_SADDLE'):
            compute_cumulants(_cubic(s), k=2, max_ell=0, external_fields=[('x', 1)] * 2,
                              tau_grid=np.array([0.0]), fixed_point_index=2, **RUN)
