"""Validation of the cumulants → moments cluster expansion in
daedalus.py (Config.output = 'moment' | 'central_moment').

The moment of a k-point set is the set-partition sum over cumulants
M(1…k) = Σ_π ∏_B κ(B); central = the same with singleton blocks → 0, raw =
singletons → the mean ⟨φ⟩.  These exercise the assembler against its exact
algebraic identities (Bell numbers; M₄ = κ₄ + 3κ₂²; the μ² mean term).

Run:  sage -python -m pytest tests/test_daedalus_moments.py -q
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', 'notebooks')))

import daedalus as dd  # noqa: E402

_TAU = dict(tau_max=2.0, tau_step=2.0)          # 3-pt grid {-2, 0, 2}


def test_set_partitions_are_bell_numbers():
    for k, bell in [(1, 1), (2, 2), (3, 5), (4, 15), (5, 52), (6, 203)]:
        n = sum(1 for _ in dd._set_partitions(list(range(k))))
        assert n == bell, (k, n, bell)


def test_central_moment_k4_is_kappa4_plus_3_kappa2_sq():
    """k=4 central equal-time moment must equal κ₄(0) + 3·κ₂(0)² — the full
    block + the 3 pair-partitions (singletons dropped)."""
    m, mod = dd.load_theory('ou_quartic_double_well')
    r4 = dd.run(m, dd.Config(k=4, max_ell=0, output='central_moment', **_TAU), mod)
    tau = np.asarray(r4['tau_grid'])
    zi = int(np.argmin(np.abs(tau)))
    k4 = float(np.real(np.asarray(r4['C_tau']))[zi])        # 4-pt cumulant @ τ=0
    r2 = dd.run(m, dd.Config(k=2, max_ell=0, **_TAU), mod)
    k2 = float(np.real(np.asarray(r2['C_tau']))[
        int(np.argmin(np.abs(r2['tau_grid'])))])
    M4 = float(np.real(r4['moment'])[zi])
    assert abs(M4 - (k4 + 3.0 * k2 * k2)) < 1e-9, (M4, k4, k2)
    assert r4['output_kind'] == 'central_moment'


def test_raw_moment_adds_mean_squared(monkeypatch):
    """k=2 raw moment = κ₂(τ) + ⟨φ⟩²; force a non-zero mean and check the
    constant μ² offset over the whole curve."""
    m, mod = dd.load_theory('ou_quartic_double_well')
    monkeypatch.setattr(dd, '_external_mean', lambda model, res: 2.0)
    rm = dd.run(m, dd.Config(k=2, max_ell=0, output='moment', **_TAU), mod)
    monkeypatch.undo()
    rc = dd.run(m, dd.Config(k=2, max_ell=0, **_TAU), mod)
    diff = np.real(rm['moment']) - np.real(rc['C_tau'])
    assert np.allclose(diff, 4.0, atol=1e-9)                # μ² = 2² = 4


def test_central_equals_cumulant_for_k_le_3():
    """Central moments coincide with cumulants for k ≤ 3 (μ₂=κ₂, μ₃=κ₃)."""
    m, mod = dd.load_theory('ou_quartic_double_well')
    for k in (2, 3):
        rc = dd.run(m, dd.Config(k=k, max_ell=0, **_TAU), mod)
        rk = dd.run(m, dd.Config(k=k, max_ell=0, output='central_moment',
                                 **_TAU), mod)
        assert np.allclose(np.real(rk['moment']),
                           np.real(rc['C_tau']), atol=1e-9), k


def _zi(res):
    return int(np.argmin(np.abs(np.asarray(res['tau_grid']))))


def test_one_loop_moment_shares_the_loop_budget():
    """The perturbatively-consistent convention: at L=1 the 4-pt central
    moment's pair-blocks sum over Σℓ_B ≤ 1 — i.e. κ₂⁽⁰⁾² + 2κ₂⁽⁰⁾κ₂⁽¹⁾ —
    NOT the dressed product (κ₂⁽⁰⁾+κ₂⁽¹⁾)², which would add a stray κ₂⁽¹⁾²
    (a partial, incomplete 2-loop term).  Pins the convention so a regression
    to the dressed product is caught."""
    m, mod = dd.load_theory('ou_quartic_double_well')
    r2 = dd.run(m, dd.Config(k=2, max_ell=1, **_TAU), mod)
    r4 = dd.run(m, dd.Config(k=4, max_ell=1, output='central_moment',
                             **_TAU), mod)
    z2, z4 = _zi(r2), _zi(r4)
    k2_0 = float(np.real(r2['C_tau_by_ell'][0])[z2])
    k2_1 = float(np.real(r2['C_tau_by_ell'][1])[z2])
    k4_0 = float(np.real(r4['C_tau_by_ell'][0])[z4])
    k4_1 = float(np.real(r4['C_tau_by_ell'][1])[z4])
    M4 = float(np.real(r4['moment'])[z4])

    consistent = (k4_0 + k4_1) + 3.0 * (k2_0 * k2_0 + 2.0 * k2_0 * k2_1)
    dressed = (k4_0 + k4_1) + 3.0 * (k2_0 + k2_1) ** 2
    assert abs(M4 - consistent) < 1e-7, (M4, consistent)
    # the two conventions genuinely differ here (κ₂⁽¹⁾ ≠ 0), so this is a
    # real discriminating test, not a tautology:
    assert abs(consistent - dressed) > 1e-4, (k2_1, consistent, dressed)
