"""
tests/test_audit_wave1.py
=========================
Regression tests for the Oct 2026 audit's fix wave 1 (core accuracy).
Inline public models only.

* a non-white ``noise_structure`` is refused (the analyzer's
  G2_NOISE_STRUCTURE and a hard stop in compute), and the source amplitude
  is a scalar prefactor whatever the temporal type;
* the exact delayed tree is unit invariant (its boundary-value Jacobian was
  built by a finite difference that cancelled the O(N) constant);
* the delay approximant's estimated error enters G6_ACCURACY;
* a failing gentreeg raises instead of truncating the tree list;
* the k = 2 tau = 0 point is the exact Ito left limit C(0-) (default), the
  legacy -1e-6 sample under DAEDALUS_TAU0=nudge.
"""
import copy
import os
import stat
import sys
import warnings

import numpy as np
import pytest

pytest.importorskip('sage.all')

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from api import compute_cumulants                                      # noqa: E402
from api.analyze import AnalyzerError, Finding                         # noqa: E402
from api._delay_exact import DelayedStateSpace, ExactTree              # noqa: E402
from api.noise import Gaussian                                         # noqa: E402
from api.sde import SDE                                                # noqa: E402


def _ou(name, mu=1.0, D=3.0):
    return (SDE(name).parameter('mu', default=mu).parameter('D', default=D).field('x')
            .drift('x', '-mu*x').noise('xi', Gaussian(var='2*D'), couples={'x': 1})
            .build())


def _cc(model, taus, ell=0, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return compute_cumulants(model, k=2, max_ell=ell, external_fields=[('x', 1)] * 2,
                                 tau_grid=np.asarray(taus, dtype=float), parallel=False,
                                 verbose=False, use_cache=False, **kw)


# ── colored noise_structure (audit A3a-5 / B2-2) ─────────────────────────

def test_colored_noise_structure_refused_and_amplitude_kept():
    from engine.diagrams.symmetry import classify_coefficient_factors
    white = _ou('aw1 ou white')
    colored = copy.deepcopy(white)
    colored['noise_structure'] = {'temporal_type': 'colored', 'amplitude_params': []}
    with pytest.raises(AnalyzerError, match='G2_NOISE_STRUCTURE'):
        _cc(colored, [0.5])
    with pytest.raises(ValueError, match='noise_structure temporal_type'):
        _cc(colored, [0.5], preflight=False)
    r = _cc(white, [0.5])
    assert abs(np.real(r['C_tau'])[0] / (3.0 * np.exp(-0.5)) - 1) < 1e-13
    tds = [rec['typed_diagram'] for rec in r['diagrams']]
    assert tds
    for td in tds:
        pw = classify_coefficient_factors(td, [], white.get('noise_structure'))
        pc = classify_coefficient_factors(td, [], colored['noise_structure'])
        assert (pw['scalar_prefactor'] - pc['scalar_prefactor']).is_zero()
        for info in pc['source_time_info'].values():
            assert info['amplitude'] == 1          # nothing left to count twice


# ── exact delayed tree: unit invariance (audit 3B-02) ────────────────────

def _ss(S, N):
    """Two-field delayed system, fields rescaled u -> S u (S diagonal)."""
    A = np.array([[-1.3, 0.4], [-0.5, -0.9]])
    Ad = np.array([[0.3, 0.0], [0.6, -0.4]])
    B = np.eye(2)
    Si = np.diag(1.0 / np.diag(S))
    return DelayedStateSpace(S @ A @ Si, S @ Ad @ Si, S @ B, np.zeros((2, 2)),
                             np.eye(2), np.zeros((2, 2)), 0.7, N)


@pytest.mark.parametrize('scales', [(1e6, 1e6), (1e6, 1e-3), (1e-5, 3e4)])
def test_exact_delayed_tree_is_unit_invariant(scales):
    N = np.array([[0.4, 0.1], [0.1, 0.25]])
    taus = [0.0, 0.3, 1.1, 2.0]
    ref = {(a, b): ExactTree(_ss(np.eye(2), N), a, b).values(taus)
           for a in range(2) for b in range(2)}
    S = np.diag(scales)
    for (a, b), v in ref.items():
        got = ExactTree(_ss(S, N), a, b).values(taus) / (scales[a] * scales[b])
        assert np.max(np.abs(got - v)) <= 1e-11 * np.max(np.abs(v))   # 6e-13 measured


# ── the delay approximant's error in G6 (audit 3B-13) ────────────────────

def test_delay_estimate_folds_into_g6_accuracy():
    from api import analyze as an
    from api.delay import _fold_delay_into_g6
    g6 = Finding(code='G6_ACCURACY', level='info', gate='G6',
                 message='accuracy estimate R = 2.03e-14: summed error estimates 3.7e-17 of 4 '
                         'closed-form subset evaluation(s)')
    other = Finding(code='G3_X', level='info', gate='G3', message='x')
    pol = an._coerce(None, an.Policy, 'policy')
    out, d = _fold_delay_into_g6(an, [g6, other],
                                 {'accuracy_estimate': 2.03e-14, 'accuracy_threshold': 1e-10},
                                 {'est_rel_error': 1.3e-4, 'n': 16}, pol)
    assert d['accuracy_estimate'] == pytest.approx(1.3e-4 + 2.03e-14, rel=1e-12)
    assert d['accuracy_estimate_phase_j'] == 2.03e-14
    codes = [f.code for f in out]
    assert codes.count('G6_ACCURACY_LOW') == 1 and 'G3_X' in codes
    msg = next(f.message for f in out if f.code == 'G6_ACCURACY')
    assert 'delay approximant' in msg and '0.00013' in msg
    # no estimate (fixed order, or the exact tree made the approximant moot)
    for est in (None, 0.0):
        out2, d2 = _fold_delay_into_g6(an, [g6], {'accuracy_estimate': 2e-14},
                                       {'est_rel_error': est, 'n': 4}, pol)
        assert out2 == [g6] and d2['accuracy_estimate'] == 2e-14


# ── gentreeg exit status (audit B1) ──────────────────────────────────────

def test_failing_gentreeg_raises(tmp_path, monkeypatch):
    import engine.enumeration.loop_diagram_enumeration as L
    if not L._fe.available:
        pytest.skip('compiled tree stage not built')
    real = L._gentreeg_path()
    k, ell, n = 3, 1, 9
    spec = L._tree_bounds_by_j(k, ell)
    good = L._trees_of_order_fast(n, 0, 1, k, ell, spec)
    assert len(good) > 3
    for body in ('| head -3\nexit 1', '| head -3\nkill -9 $$'):
        w = tmp_path / 'gentreeg_fail'
        w.write_text(f'#!/bin/sh\n"{real}" "$@" {body}\n')
        w.chmod(w.stat().st_mode | stat.S_IEXEC)
        monkeypatch.setattr(L, '_gentreeg_path', lambda: str(w))
        with pytest.raises(RuntimeError, match='gentreeg.*exited with status'):
            L._trees_of_order_fast(n, 0, 1, k, ell, spec)


# ── tau = 0: the exact Ito left limit (audit A3a-1 / 3B-01) ──────────────

@pytest.mark.parametrize('mu', [0.5, 40.0, 1e3])
def test_tau0_is_the_exact_left_limit(mu, monkeypatch):
    D = 0.2
    m = _ou(f'aw1 ou tau0 {mu}', mu=mu, D=D)
    monkeypatch.delenv('DAEDALUS_TAU0', raising=False)
    r = _cc(m, [0.0, 1.0 / mu])
    ex = D / mu
    assert abs(np.real(r['C_tau'])[0] / ex - 1) < 1e-13
    assert abs(np.real(r['total_C'](0.0, 0.0)) / ex - 1) < 1e-13
    assert abs(np.real(r['C_tau'])[1] / (ex * np.exp(-1.0)) - 1) < 1e-13
    monkeypatch.setenv('DAEDALUS_TAU0', 'nudge')
    rn = _cc(m, [0.0])
    assert abs(np.real(rn['C_tau'])[0] / ex - 1) == pytest.approx(mu * 1e-6, rel=1e-3)
    monkeypatch.setenv('DAEDALUS_TAU0', 'bogus')
    with pytest.raises(ValueError, match='DAEDALUS_TAU0'):
        _cc(m, [0.0])
