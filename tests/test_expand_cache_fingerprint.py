"""
tests/test_expand_cache_fingerprint.py
======================================
The expand cache is addressed by model NAME + Taylor order, and the cached
expansion carries the mean-field substitution in its MF sector.  Editing the
action or the mean-field rows under the same name must therefore invalidate
the bundle — otherwise a model whose MF rows were once missing keeps failing
the tadpole check forever (the model-builder failure of Sept 2026).  Also
covers the two pieces that let that happen in the first place: MF rows with a
blank right-hand side were dropped silently, and a model without any
mean-field equation built without a word.

Run:  sage -python -m pytest tests/test_expand_cache_fingerprint.py -q
"""
from __future__ import annotations

import os
import sys
import tempfile
import warnings

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from api.model import TemporalModelBuilder
from api.model_serialize import normalize_equation_rows

ACTION = ('nt*n - (exp(nt)-1)*f + ut*u - (exp(ut)-1)*m*r - log(1 + p*(exp(ut)-1))*m*n'
          ' + mt*(Dt*m + u) - (exp(mt)-1)*(M-m)/tauM')
ROWS = [{'lhs': 'n-f', 'rhs': '0'}, {'lhs': 'u-m*r-m*n*p', 'rhs': '0'},
        {'lhs': 'Dt*m + u - (M-m)/tauM', 'rhs': '0'}]


def _model(rows, name='Cache Fingerprint Model'):
    b = (TemporalModelBuilder(name).physical_field('n').physical_field('u').physical_field('m')
         .parameter('f', default=0.5).parameter('r', default=0.3).parameter('p', default=0.2)
         .parameter('M', default=2.0).parameter('tauM', default=1.5)
         .set_action_text(ACTION))
    for r in rows:
        b = b.equation(lhs=r['lhs'], rhs=r['rhs'])
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return b.build()


def test_normalize_equation_rows_accepts_what_people_type():
    rows = [{'lhs': 'n-f', 'rhs': ''},                       # blank RHS → 0
            {'lhs': 'u - m*r = 0', 'rhs': None},             # 'lhs = rhs' in one box
            {'lhs': '(Dt+mu)*x', 'rhs': '-eps*x^3', 'population': 'pop'},
            {'lhs': '', 'rhs': '0'}]                         # empty LHS → skipped
    out = normalize_equation_rows(rows)
    assert out == [{'lhs': 'n-f', 'rhs': '0', 'population': None},
                   {'lhs': 'u - m*r', 'rhs': '0', 'population': None},
                   {'lhs': '(Dt+mu)*x', 'rhs': '-eps*x^3', 'population': 'pop'}]


def test_build_warns_when_no_mean_field_equations_reach_it():
    b = (TemporalModelBuilder('no mf').physical_field('x')
         .parameter('mu', default=1.0).parameter('D', default=1.0)
         .set_action_text('xt*((Dt+mu)*x + x^3) - D*xt^2'))
    with pytest.warns(UserWarning, match='no mean-field equations'):
        b.build()


def test_spec_signature_tracks_the_text_declarations():
    a = _model(ROWS)['spec_signature']
    b = _model(ROWS[:1])['spec_signature']
    c = _model(ROWS)['spec_signature']
    assert a == c and a != b and len(a) == 16


def test_stale_bundle_is_rejected_and_reexpansion_passes():
    """The reported sequence: first build with the MF rows lost (tadpole FAIL,
    bundle written), then the same model name with the rows present.  The
    second run must NOT be served the first run's expansion."""
    from api import precompute
    from api import _expand_cache as _ec
    with tempfile.TemporaryDirectory() as tmp:
        cwd = os.getcwd()
        os.chdir(tmp)                                         # cache root is relative
        try:
            bad = precompute(_model([]), verbose=False)
            assert bad['mf_check'] == 'FAIL' and bad['mf_values'] == {}
            assert os.path.isfile(_ec.expand_cache_path(_model([]), 2))
            good = precompute(_model(ROWS), verbose=False)
            assert good['mf_check'] == 'PASS', good['mf_check']
            assert abs(float(good['mf_values']['mstar'][0]) - 1.25) < 1e-9
            # and a plain load of the (now refreshed) bundle for the wrong model text is refused
            from api.compute import FieldTheory
            ft = FieldTheory(_model(ROWS[:1]), taylor_order=2)
            _ec.prepare_for_load(ft)
            assert _ec.load_expand(_model(ROWS[:1]), ft, target_order=2) is False
        finally:
            os.chdir(cwd)
