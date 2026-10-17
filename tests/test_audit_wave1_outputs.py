"""
tests/test_audit_wave1_outputs.py
=================================
Regression tests for the Oct 2026 audit's fix wave 1, outputs
(``api.symbolic_out``, ``api.symbolic_loop``, ``api.me_output``,
``api.save``).  Inline public models only; each case is the audit's
reproducer (it fails on the code before the fix).
"""
import math
import os
import sys
import warnings

import numpy as np
import pytest

pytest.importorskip('sage.all')

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from sage.all import SR, I, matrix                                     # noqa: E402

from api import symbolic_out as so                                     # noqa: E402
from api.symbolic_out import SymbolicOutError                          # noqa: E402

omega, mu, D, tau, a, b = SR.var('omega mu D tau a b')


def test_values_are_not_snapped_to_zero():
    """G1-symbolic 0: D = 1e-13 gave C0 = 0."""
    G = matrix(SR, [[1 / (mu + I * omega)]])
    Dm = matrix(SR, [[2 * D]])
    c13 = so.tree_covariance_from_matrices(G, Dm, omega, values={'D': 1e-13, 'mu': 1.0}).C0[0, 0]
    c1 = so.tree_covariance_from_matrices(G, Dm, omega, values={'D': 1.0, 'mu': 1.0}).C0[0, 0]
    r = complex((SR(c13) / SR(c1)).subs(omega=0.3).n())
    assert abs(r / 1e-13 - 1) < 1e-12


def test_saddle_snap_is_relative():
    """G1-symbolic 0 (solver side): a genuine 1e-14 saddle of a small-unit
    model survives, dust far below the run's scale is snapped."""
    m = {'parameters': [{'name': 'xstar', 'mean_field': True}]}
    out = so._mf_symbols_only(m, {SR('xstar'): 3e-14, SR('k'): 2e-9})
    assert out['xstar'] == 3e-14
    out = so._mf_symbols_only(m, {SR('xstar'): 3e-14, SR('k'): 2.0})
    assert out['xstar'] == 0


def test_legacy_inverse_transform_keeps_a_parameter_named_tau():
    """G1-symbolic 1: the lag and a parameter tau were one symbol."""
    out = so.inverse_transform(2 * D / (1 / tau ** 2 + omega ** 2), omega, legacy=True)
    lag = out['tau']
    assert str(lag) == 'tau_lag'
    want = D * tau * SR('e') ** (-lag / tau)
    assert bool((SR(out['positive']) - want).simplify_full() == 0)


def test_retry_does_not_leak_the_moved_reference():
    """G1-symbolic 2: the ModeSum carried the perturbed point."""
    out = so.inverse_transform(1 / ((a ** 2 + omega ** 2) * (b ** 2 + omega ** 2)), omega,
                               reference={'a': 1, 'b': 1})
    ms = out['modes']
    assert {k: float(v) for k, v in ms.reference.items()} == {'a': 1.0, 'b': 1.0}
    with pytest.raises(Exception):          # the coinciding point is refused, loudly
        ms.evaluate([0.5])
    # a distinct point evaluates
    v = ms.evaluate([0.5], {'a': 1.0, 'b': 2.0})
    assert np.isfinite(v).all()


def test_even_test_is_relative():
    """G1-symbolic 3: a 1e-12-amplitude asymmetric entry was 'even'."""
    for eps in (1, 1e-12):
        C0 = eps / ((1 + I * omega) * (2 - I * omega))
        assert so.inverse_transform(C0, omega)['even'] is None
    sym = so.inverse_transform(1e-12 * 2 / (1 + omega ** 2), omega)
    assert sym['even'] is not None


def test_rootof_even_test_at_fast_rates():
    """G1-symbolic 4: rates ~1e5 underflowed to 0 == 0 and were 'even'."""
    s = I * omega

    def q(x):
        return x ** 3 + x ** 2 + 2 * x + 1
    for L in (1, 10 ** 5):
        C0 = (1 + s / L) / (q(s / L) * q(-s / L))
        out = so.inverse_transform(C0, omega)
        assert out['roots'] and out['even'] is None


def test_modesum_refuses_unknown_names():
    """G1-symbolic 5: a typo evaluated at the reference."""
    ms = so.tree_modes(2 * D / (mu ** 2 + omega ** 2), omega, reference={'mu': 1, 'D': 1})
    with pytest.raises(SymbolicOutError, match="unknown name 'Mu'"):
        ms.evaluate([0.5], {'Mu': 3.0})
    assert abs(ms.evaluate([0.5], {'mu': 3.0})[0].real - math.exp(-1.5) / 3) < 1e-14


def test_tree_covariance_refuses_unknown_values():
    """G1-symbolic 6: a misspelled values= name was dropped."""
    from api.noise import Gaussian
    from api.sde import SDE
    m = (SDE('aw1 sym ou').parameter('mu', default=1.0).parameter('D', default=0.5).field('x')
         .drift('x', '-mu*x').noise('xi', Gaussian(var='2*D'), couples={'x': 1}).build())
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        with pytest.raises(SymbolicOutError, match='typo_name'):
            so.tree_covariance(m, values={'typo_name': 5.0, 'D': 1.0}, mean_field='solve')
        cov = so.tree_covariance(m, values={'D': 1.0}, mean_field='solve')
    assert cov.values.get('D') == 1.0


def test_symbolic_loop_saddle_floor_is_relative():
    """G1-symbolic 7: the floor max(1, |x|) made the saddle checks absolute."""
    from api.symbolic_loop import _value_floor
    assert _value_floor({'a': 1e-6, 'b': -2e-6}) == pytest.approx(2e-12)
    assert _value_floor({'a': 3e4}) == pytest.approx(3e-2)
    # the audit's case: a 1e-6 saddle moved by 1.4 % is a move
    ref, new = 1e-6, 1e-6 * 1.014
    assert abs(new - ref) > 1e-9 * max(abs(ref), _value_floor({'x': ref, 'k': 1e-3}))


def test_side_me_refuses_beyond_its_chamber():
    """G2-outputs 1: a finite chamber's side answered beyond u_max."""
    from api.me_output import SideME
    s = SideME('(0, 1)', [], 0, u_max=1.0)
    assert s(np.array([0.0, 1.0])).shape == (2,)
    with pytest.raises(ValueError, match='beyond'):
        s(np.array([1.5]))
    assert SideME('+', [], 0)(np.array([1e6])).shape == (1,)      # infinite side


def test_slice_me_refuses_a_nan_base_lag():
    """G2-outputs 3: a NaN base lag was pinned at the left limit."""
    from api.me_output import slice_me
    with pytest.raises(ValueError, match='finite'):
        slice_me({'config': {'k': 3}}, 1, base=[0.0, float('nan')])


def test_save_csv_round_trips(tmp_path):
    """G2-outputs 2: 10 significant digits."""
    from api import compute_cumulants
    from api.noise import Gaussian
    from api.save import save_csv
    from api.sde import SDE
    m = (SDE('aw1 csv ou').parameter('mu', default=1.3).parameter('D', default=0.7).field('x')
         .drift('x', '-mu*x').noise('xi', Gaussian(var='2*D'), couples={'x': 1}).build())
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        r = compute_cumulants(m, k=2, max_ell=0, external_fields=[('x', 1)] * 2,
                              tau_grid=np.array([0.1, 0.7, 1.9]), parallel=False,
                              verbose=False, use_cache=False)
    p = tmp_path / 'r.csv'
    save_csv(r, str(p))
    lines = [ln for ln in p.read_text().splitlines() if ln and not ln.startswith('#')]
    hdr = lines[0].split(',')
    col = hdr.index('C_total.real')
    got = np.array([float(ln.split(',')[col]) for ln in lines[1:]])
    assert np.array_equal(got, np.real(np.asarray(r['C_tau'])))
