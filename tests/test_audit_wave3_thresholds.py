"""
tests/test_audit_wave3_thresholds.py
====================================
Audit wave 3c, thresholds of the default route that were absolute in the
model's time unit.  Every check here is a pair of exact time-rescaled twins
(or a closed form): the answers must not depend on the unit.

* ACC-07: the G2-K argument principle read the large-``s`` law of
  ``det K(s)`` upwards from a fixed 1e3, so for rates above ~2.5e5 it took
  the constant regime for the leading term and called unstable systems
  'stable'.
* ACC-15: non-local noise sources were boxed at an absolute ±50 in their τ
  variable (wide kernels truncated); the box is now read off the kernel.
* ACC-16: dressed-line modes were dropped by residue size alone (a slow mode
  with a small residue is a real tail), uncounted at the input; undressed
  lines of a dressed diagram were clipped too.
* ACC-17: the hardened quadrature fallback without mode data cut open sides
  at an absolute 200 time units; it now cuts at K/κ beyond the farthest
  breakpoint, κ the model's slowest line / kernel rate.  Its warnings were
  issued once per model NAME; now once per set of diagrams and parameter
  point.

Run:  sage -python -m pytest tests/test_audit_wave3_thresholds.py -q
"""
from __future__ import annotations

import math
import os
import sys
import warnings

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import engine.integration.time_domain.final_integral as FI   # noqa: E402


# --- ACC-07 ---------------------------------------------------------------

@pytest.mark.parametrize('r', [1e-6, 1.0, 3e5, 1e8, 1e12])
def test_kernel_stability_finds_a_fast_unstable_root(r):
    from api._kernel_stability import count_rhp_roots
    ks = count_rhp_roots(lambda s: np.asarray(s, complex) - r)
    assert ks.status == 'unstable' and ks.n_rhp == 1
    assert abs(ks.rightmost - r) <= 1e-9 * r


def test_kernel_stability_of_time_rescaled_delayed_cycle():
    """The two-population delayed cycle of tests/test_delay_kernels.py,
    rescaled in time by T: the same verdict, roots and radius x 1/T."""
    from sage.all import SR, I, exp, matrix
    from api._kernel_stability import kernel_stability
    om, d = SR.var('omega'), SR.var('d')
    tau0, aa, W = [1.0, 1.5], [1.0, 1.0], [[0.3, -1.6], [1.4, 0.2]]
    ref = None
    for T in (1.0, 1e-6, 1e-9, 1e4):
        tau = [t * T for t in tau0]
        K = matrix(SR, 2, 2, lambda i, j: (tau[i] * I * om + 1 if i == j else 0)
                   - aa[i] * W[i][j] * exp(-I * om * d))
        ks = kernel_stability(K, om, {d: 1.0 * T})
        assert ks.status == 'unstable' and ks.n_rhp == 2, (T, ks)
        got = (complex(ks.rightmost) * T, ks.radius * T)
        if ref is None:
            ref = got
            # Pade-12 reference roots (tests/test_delay_kernels.py): 0.0735 +- 0.7206i
            assert abs(abs(got[0].imag) - 0.720581045742538) < 1e-9
            assert abs(got[0].real - 0.07346345713013953) < 1e-9
        else:
            assert abs(got[0].real - ref[0].real) < 1e-9
            assert abs(abs(got[0].imag) - abs(ref[0].imag)) < 1e-9
            assert got[1] == pytest.approx(ref[1], rel=1e-12)


# --- ACC-15 ---------------------------------------------------------------

def _colored_ou(S):
    from api.model import TemporalModelBuilder
    b = (TemporalModelBuilder(f'w3c colored OU S{S:g}', n_populations=0)
         .physical_field('x', indexed=True, description='variable')
         .parameter('mu', default=1.0 / S, domain='positive')
         .parameter('D', default=1.0, domain='positive')
         .parameter('tauc', default=2.0 * S, domain='positive')
         .declare_cgf_term('CXX', response_legs=['xt', 'xt'], order=2,
                           coefficient='2*D/tauc', kernel='exp(-abs(tau)/tauc)',
                           markovianize=False)
         .set_action_text('xt*((Dt+mu)*x)')
         .equation(lhs='(Dt+mu)*x', rhs='0'))
    return b.build()


def _colored_ou_closed(S, tau):
    mu, k, c = 1.0 / S, 1.0 / (2 * S), 2.0 / (2 * S)
    return c / (mu ** 2 - k ** 2) * (np.exp(-k * tau) - (k / mu) * np.exp(-mu * tau))


@pytest.mark.parametrize('S', [1.0, 100.0, 1e4])
def test_colored_noise_source_is_unit_invariant(S):
    from api import compute_cumulants
    tau = np.array([0.0, 0.5 * S, 3.0 * S])
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        r = compute_cumulants(_colored_ou(S), k=2, max_ell=0,
                              external_fields=[('x', 1), ('x', 1)], tau_grid=tau,
                              parallel=False, verbose=False, use_cache=False)
    assert r['status'] == 'ok'
    got = np.real(np.asarray(r['C_tau_by_ell'][0]))
    ex = _colored_ou_closed(S, tau)
    assert np.max(np.abs(got - ex) / np.abs(ex)) < 1e-8


def test_noise_kernel_box_moves_with_the_kernel():
    from sage.all import SR, exp
    t = SR.var('tt')
    w = [FI._noise_kernel_halfwidth([exp(-abs(t) / L) / L], t) for L in (1.0, 1e3)]
    assert w[1] / w[0] == pytest.approx(1e3, rel=0.8)
    # beyond the box the kernel is below TAU_KERNEL_RTOL of its peak
    for L, hw in zip((1.0, 1e3), w):
        assert math.exp(-hw / L) <= FI.TAU_KERNEL_RTOL


# --- ACC-16 ---------------------------------------------------------------

def _conv_exact(modes, kmodes, D):
    return sum(C * c * (np.exp(l * D) - np.exp(m * D)) / (l - m)
               for (C, l) in modes for (c, m) in kmodes)


@pytest.mark.parametrize('modes,kmodes,tol', [
    # a slow mode with a tiny residue (the line's real tail)
    (((1, -1), (5e-13, -1e-6)), [(3, -3)], 1e-14),
    # a kernel rate close to a pole: the coefficients' sum is inflated by
    # the cancelling pair C c / (lam - mu); the -3 mode is real
    (((1, -1), (1e-6, -3)), [(1, -1 - 2e-6)], 1e-9),
])
def test_dressing_keeps_modes_with_integral_weight(modes, kmodes, tol):
    from engine.integration.time_domain.final_integral import EdgeModeSum
    out = FI._dress_edge_modes(EdgeModeSum(0, 0, 0j, modes), kmodes)
    integ_exact = sum(C * c / (-l) / (-m) for (C, l) in modes for (c, m) in kmodes)
    integ_got = sum(complex(C) / -complex(l) for (C, l) in out.modes)
    assert abs(integ_got - integ_exact) <= tol * abs(integ_exact)
    for D in (0.5, 30.0, 3e5):
        ex = _conv_exact(modes, kmodes, D)
        got = sum(complex(C) * np.exp(complex(l) * D) for (C, l) in out.modes)
        assert abs(got - ex) <= 1e-6 * abs(ex) + 1e-300


def test_dressing_counts_a_dropped_rounding_mode():
    from engine.integration.time_domain.final_integral import EdgeModeSum
    FI._reset_runtime_counters()
    out = FI._dress_edge_modes(EdgeModeSum(0, 0, 0j, ((1, -1), (1e-20, -5))),
                               [(3, -3)])
    assert FI._RUNTIME_COUNTERS['conv_dress_zero_modes_dropped'] == 1
    assert all(abs(complex(l) + 5) > 1e-9 for (_C, l) in out.modes)


def _hand_built_conv_diagram(kernel):
    from sage.all import SR, DiGraph
    from engine.core.vertices import ConvVertexType, SourceType
    from engine.diagrams.type_assignment import TypedDiagram
    D = DiGraph(multiedges=True, loops=False)
    D.add_edge(2, 0, 'a')
    D.add_edge(3, 2, 'b')
    D.add_edge(3, 1, 'c')
    g = SR.var('g')
    conv = ConvVertexType(
        coefficient=g, response_legs=[('xt', 1)], physical_legs=[('x', 1)],
        bigrade=(1, 1),
        kernel_attachments=[{'symbol': g, 'leg': ('x', 1), 'leg_index': 0,
                             'kernel_td_fn': kernel}])
    src = SourceType(coefficient=SR(1), response_legs=[('xt', 1), ('xt', 1)],
                     bigrade=(2, 0))
    et = {(2, 0, 'a'): (('xt', 1), ('x', 1)),
          (3, 2, 'b'): (('xt', 1), ('x', 1)),
          (3, 1, 'c'): (('xt', 1), ('x', 1))}
    td = TypedDiagram((D, D, [0, 1], [2, 3]), {2: conv, 3: src}, et,
                      {0: ('x', 1), 1: ('x', 1)}, {e: (0, 0) for e in et})
    return td, g


def test_undressed_line_keeps_its_slow_tail(monkeypatch):
    """Line G = e^{-t} + 5e-13 e^{-1e-5 t} (a slow, weakly coupled tail),
    kernel 3 e^{-3t}: the dressed default agrees with the leg-time path at
    every lag, including where the tail dominates."""
    from sage.all import SR, I, matrix, exp
    prop = {'pole_vals': [SR(1.0) * I, SR(1e-5) * I],
            'C_mats': [matrix(SR, [[1]]), matrix(SR, [[5e-13]])],
            'D_delta': matrix(SR, [[0]]), 'nf': 1}
    pts = [(0.0, 0.5), (0.0, 30.0), (0.0, 300.0)]

    def run(dress):
        monkeypatch.setattr(FI, 'DRESS_CONV_KERNELS', dress)
        td, g = _hand_built_conv_diagram(lambda tau: 3 * exp(-3 * tau))
        res = FI.integrate_diagram(td, prop, SR(0.3) * g,
                                   [SR.var('t_1'), SR.var('t_2')],
                                   external_fields=[('x', 1), ('x', 1)])
        assert res['status'] == 'ok', res.get('reason')
        return np.array([complex(res['contribution'](*p)) for p in pts])

    vd, vl = run(True), run(False)
    assert np.max(np.abs(vd - vl) / np.abs(vl)) < 1e-12


# --- ACC-17 ---------------------------------------------------------------

_ROWS = [((-1.0, 1.0), 0.0), ((0.0, -1.0), 0.0)]     # s0 < s1 < 0


def _meta(serial, params=None, floor=None):
    return {'model': ('model', 'w3c m'), 'source': 'per_diagram',
            'loop_number': 1, 'diagram_serial': serial, 'params': params,
            'kappa_floor': floor}


@pytest.mark.parametrize('k', [1.0, 0.02, 1e-4])
@pytest.mark.parametrize('T', [1.0, 1e3])
def test_hardened_fallback_without_modes_cuts_at_the_decay_floor(k, T):
    kk = k / T
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        v = FI._integrate_polytope_hardened(
            lambda s0, s1: math.exp(kk * (s0 + s1)), _ROWS, [], 2,
            mode_info=None, diag_meta=_meta(1, floor=kk))
    ex = 1 / (2 * kk * kk)
    assert abs(v - ex) <= 1e-12 * ex


def test_hardened_fallback_warns_per_diagram_and_parameter_point():
    def warns(metas):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            for meta in metas:
                FI._nquad_note_region(meta, 2, flag_msgs=('roundoff',))
        return [x for x in w if x.category is FI.PhaseJNquadFallbackWarning]

    FI._NQUAD_WARNED.clear()
    assert len(warns([_meta(101, params=1)])) == 2           # fallback + flags
    assert len(warns([_meta(101, params=1)])) == 0           # the same again
    assert len(warns([_meta(102, params=1)])) == 2           # another diagram
    assert len(warns([_meta(101, params=2)])) == 2           # other parameters
