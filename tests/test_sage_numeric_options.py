"""Options typed under the Sage Jupyter kernel: every literal there is a Sage
``Integer`` / ``Rational`` / ``RealNumber``, none of which is an ``int`` or
``float``.  The grid rule (tuple = range, list = literal) and the numeric
options must read them like Python numbers."""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

sage_all = pytest.importorskip('sage.all')
from sage.all import Integer, QQ, RealNumber, RR           # noqa: E402

import daedalus as dd                                      # noqa: E402
from daedalus.api._grids import resolve_grid                        # noqa: E402
from daedalus.api.spectra import resolve_spectra_options            # noqa: E402


def test_tuple_range_with_sage_numbers():
    ref = np.linspace(-6.0, 6.0, 13)
    for g in ((-6.0, 6.0, Integer(13)),
              (RR(-6), RR(6), Integer(13)),
              (QQ(-6), Integer(6), RR(13)),
              (Integer(-6), Integer(6), 13)):
        assert np.allclose(resolve_grid(g, 'tau_grid'), ref)


def test_list_is_literal_and_ambiguous_list_raises_with_sage_numbers():
    assert np.allclose(resolve_grid([Integer(0), RR(0.5), QQ(1)]),
                       [0, 0.5, 1])
    with pytest.raises(ValueError, match='TUPLE'):
        resolve_grid([Integer(-5), Integer(5), Integer(51)], 'tau_grid')
    with pytest.raises(ValueError):
        resolve_grid((0, 1, RR(2.5)), 'tau_grid')


def test_config_and_options_accept_sage_numbers():
    cfg = dd.Config(k=Integer(2), max_ell=Integer(0), taylor_order=Integer(3),
                    accuracy=RR(1e-8), tau_grid=(-6.0, 6.0, Integer(13)),
                    n_workers=Integer(1))
    assert len(cfg.resolved_tau_grid()) == 13
    dd.Config(accuracy=QQ(1) / 1000)
    opts = resolve_spectra_options(
        {'epsabs': RR(1e-9), 'epsrel': QQ(1) / 10 ** 6, 'limit': Integer(50)},
        'spectra') if hasattr(resolve_spectra_options, '__call__') else None
    assert opts is None or opts['limit'] == 50
