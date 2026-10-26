"""
tests/test_noise_kernel_box_quadrature.py
=========================================
Non-local noise sources whose kernel has a slow (power-law) tail.

The τ box of such a source is read off the kernel (``TAU_KERNEL_CAP = None``):
for ``1/(1+τ^4)`` it reaches ~1e4, for ``1/(1+τ^2)`` ~1e8, while the kernel's
peak is O(1) wide.  Without breakpoints the hardened quadrature stepped over
the peak: the tree of ``x' = -x + η`` was 33 % off at τ = 0.7 with status
'ok', and at τ = 0 it was exactly 0.0 for ``1/(1+τ^2)``, ``(1+τ^2)^{-3/4}``,
``1/(1+(τ/40)^2)`` and ``(1+τ^2)^{-1/2}``: with equal external times the only
breakpoint of the τ variable is 0, so the peak sat at the end of two panels
~1e8 wide, whose Gauss-Kronrod nodes (the nearest 0.0022 panel widths from
the end) all saw an integrand that had underflowed to 0 -- an estimate 0 with
error 0, accepted.  The τ variable now carries a ratio-4 ladder of breakpoints from below
the kernel's core width out to the box (``_noise_kernel_ladder``), and the
fallback's QUADPACK error estimates reach G6.

Reference: ``C(τ) = ∫ e^{-μ|u|}/(2μ) 2D K(τ - u) du`` by mpmath.  Every
kernel here is bounded and the line decays, so C is finite, also for
``(1+τ^2)^{-1/2}`` whose integral (the noise's zero-frequency power)
diverges.

Run:  sage -python -m pytest tests/test_noise_kernel_box_quadrature.py -q
"""
from __future__ import annotations

import os
import sys
import warnings

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import mpmath as mp                                     # noqa: E402

import engine.integration.time_domain.final_integral as FI   # noqa: E402

KERNELS = {
    'pow4': ('1/(1+tau^4)', lambda t: 1 / (1 + t ** 4)),
    'cauchy': ('1/(1+tau^2)', lambda t: 1 / (1 + t ** 2)),
    'cauchy_wide': ('1/(1+(tau/40)^2)', lambda t: 1 / (1 + (t / 40) ** 2)),
    'pow15': ('(1+tau^2)^(-3/4)', lambda t: (1 + t ** 2) ** mp.mpf(-0.75)),
    'inv_abs_tail': ('(1+tau^2)^(-1/2)', lambda t: (1 + t ** 2) ** mp.mpf(-0.5)),
    'exp': ('exp(-abs(tau))', lambda t: mp.exp(-abs(t))),
    'gauss': ('exp(-tau^2)', lambda t: mp.exp(-t ** 2)),
}


def _model(expr, mu, name):
    from api.model import TemporalModelBuilder
    return (TemporalModelBuilder(name, n_populations=0)
            .physical_field('x', indexed=True, description='variable')
            .parameter('mu', default=mu, domain='positive')
            .parameter('D', default=0.5, domain='positive')
            .declare_cgf_term('CXX', response_legs=['xt', 'xt'], order=2,
                              coefficient='2*D', kernel=expr,
                              markovianize=False)
            .set_action_text('xt*((Dt+mu)*x)')
            .mf_residual(lhs='(Dt+mu)*x', rhs='0')).build()


def _exact(kf, mu, taus):
    mp.mp.dps = 30
    out = []
    for t in taus:
        f = lambda u: mp.exp(-mu * abs(u)) / (2 * mu) * 2 * mp.mpf(0.5) * kf(t - u)
        pts = sorted({-mp.inf, mp.mpf(0), mp.mpf(t), mp.inf})
        out.append(float(mp.quad(f, pts)))
    return np.array(out)


def _run(m, taus):
    from api import compute_cumulants
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return compute_cumulants(m, k=2, max_ell=0,
                                 external_fields=[('x', 1), ('x', 1)],
                                 tau_grid=np.asarray(taus, float),
                                 parallel=False, verbose=False, use_cache=False)


@pytest.mark.parametrize('kid', sorted(KERNELS))
def test_tree_with_a_slow_kernel_tail_against_mpmath(kid):
    expr, kf = KERNELS[kid]
    taus = [0.0, 0.7, 3.0, 10.0]
    r = _run(_model(expr, 1.0, f'kernel box {kid}'), taus)
    assert r['status'] == 'ok'
    got = np.real(np.asarray(r['C_tau_by_ell'][0]))
    ex = _exact(kf, 1.0, taus)
    assert got[0] != 0.0                     # the equal-time point
    assert np.max(np.abs(got - ex) / np.abs(ex)) < 1e-10, (got, ex)
    codes = {f.code for f in r['analysis'].findings}
    assert 'G6_ACCURACY_LOW' not in codes


def test_time_rescaled_twin_of_the_power_law_kernel():
    """x' = -x/S + η with K(τ) = 1/(1+(τ/S)^4): the same answer up to the
    exact rescaling, at S = 100."""
    S = 100.0
    taus = [0.0, 0.7 * S, 3.0 * S]
    r = _run(_model(f'1/(1+(tau/{S:g})^4)', 1.0 / S, 'kernel box twin'), taus)
    got = np.real(np.asarray(r['C_tau_by_ell'][0]))
    ex = _exact(lambda t: 1 / (1 + (t / S) ** 4), 1.0 / S, taus)
    assert np.max(np.abs(got - ex) / np.abs(ex)) < 1e-10


def test_quadrature_share_reaches_g6():
    r = _run(_model('1/(1+tau^4)', 1.0, 'kernel box g6'), [0.0, 0.7])
    g6 = [f for f in r['analysis'].findings if f.code == 'G6_ACCURACY']
    assert g6 and 'from the quadrature fallback' in g6[0].message
    assert FI._RUNTIME_COUNTERS.get('nquad_hardened_err_abs_sum', 0.0) >= 0.0


def test_ladder_reaches_the_box_and_resolves_the_core():
    from sage.all import SR, exp
    t = SR.var('tt')
    hw, core = FI._noise_kernel_scales([1 / (1 + t ** 4)], t)
    assert 1e3 < hw < 1e6 and 0.5 <= core <= 2.0
    lad = FI._noise_kernel_ladder(core, hw)
    pos = [x for x in lad if x > 0]
    assert min(pos) <= 16 * core and max(pos) > hw / 4
    assert all(b / a <= FI._TAU_LADDER_RATIO * (1 + 1e-12)
               for a, b in zip(pos, pos[1:]))
    # a kernel that decays fast keeps the breakpoints it had (none added)
    hw, core = FI._noise_kernel_scales([exp(-abs(t))], t)
    assert FI._noise_kernel_ladder(core, hw) == ()
    # a kernel that cannot be sampled: the box only (no ladder)
    assert FI._noise_kernel_ladder(None, 50.0) == ()


def test_quadrature_budget_refuses_by_name(monkeypatch):
    """A region past NQUAD_EVAL_BUDGET raises PhaseJQuadratureBudgetError
    (here with a tiny budget; the default refuses the one-loop colored-noise
    OU after ~3 minutes instead of running for hours)."""
    from engine.errors import DaedalusError
    monkeypatch.setattr(FI, 'NQUAD_EVAL_BUDGET', 1000)
    with pytest.raises(FI.PhaseJQuadratureBudgetError) as ei:
        _run(_model('1/(1+tau^4)', 1.0, 'kernel box budget'), [0.0, 0.7])
    assert isinstance(ei.value, DaedalusError)
    assert 'NQUAD_EVAL_BUDGET' in str(ei.value)
