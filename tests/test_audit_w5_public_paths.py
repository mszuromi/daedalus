"""Public paths that no test executed (audit W5, TST-08), each against an independent value.

  * ``dd.prediagram_mappings``: the label maps of the 1-field quartic OU at k = 2, ell <= 1;
  * ``RationalKernel`` at a delta contact (``kernel_at_zero`` / ``RationalKernel.at_zero``): the
    initial-value theorem L(0+) = lim p num(p)/den(p), and the contact weight it picks;
  * ``Cumulant.smooth``: the sum of the smooth parts over loop orders;
  * ``PhaseType`` / ``Hyperexponential`` built from floats and from a symbolic rate: the moments
    m! alpha (-T)^-m 1 in exact rational arithmetic;
  * the simulator's delayed-kernel ring buffer and ``record=``: the stationary variance of a
    delayed-kernel linear SDE against its exact tree.
"""
import math
from fractions import Fraction

import numpy as np
import pytest

import api.noise as N
import api.symbolic_out as so
from api import me_output as ME
from api.compute import compute_cumulants
from api.cumulant import Cumulant


# ----------------------------------------------------------------------------- prediagram_mappings

def test_prediagram_mappings_of_the_quartic_ou():
    import daedalus as dd
    res, text = dd.prediagram_mappings('ou_quartic', 2, 1, printout=False)
    assert sorted(res) == [0, 1]
    assert len(res[0]) == 1 and len(res[1]) == 4          # tree: one source; one loop: 4 prediagrams
    for ell, entries in res.items():
        for e in entries:
            assert {'idx', 'labels', 'sig', 'typings'} <= set(e)
            assert len(e['typings']) == 1                  # one field: one typing each
    assert 'source i    → K^(2) ⟨xt xt⟩' in text
    assert text.count('external legs → 1=dx, 2=dx') == 5
    # the propagator filter drops nothing in a one-field model
    res_p, _ = dd.prediagram_mappings('ou_quartic', 2, 1, printout=False, use_propagator=True)
    assert {l: len(v) for l, v in res_p.items()} == {0: 1, 1: 4}


# ----------------------------------------------------------------------------- RationalKernel at a contact

def test_rational_kernel_value_at_zero_is_the_initial_value_theorem():
    """L(s) = InvLaplace[(3p + 1)/((p + 1)(p + 2))] = -2 e^{-s} + 5 e^{-2s}: L(0) = 3."""
    k = ME.RationalKernel([3.0, 1.0], [1.0, 3.0, 2.0], side='+')
    assert abs(k.at_zero() - 3.0) < 1e-15
    assert abs(k(0.7) - (-2 * math.exp(-0.7) + 5 * math.exp(-1.4))) < 1e-14
    km = ME.RationalKernel([3.0, 1.0], [1.0, 3.0, 2.0], side='-')
    assert ME.kernel_at_zero(k) == (3.0, 0j) and ME.kernel_at_zero(km) == (0j, 3.0)
    # mixed with a symbolic kernel term: L(0+) adds up
    lp, lm = ME.kernel_at_zero([k, so.PairingKernel.exponential(0.5, -1.0, 'positive')])
    assert abs(lp - 3.5) < 1e-15 and lm == 0


def test_rational_kernel_contact_weight_on_a_delta_at_zero():
    """A one-population linear Hawkes C(tau) carries lambda delta(tau); a one-sided rational kernel
    jumps there (L(0+) = 3, L(0-) = 0), so the contact needs a rule: 'positive' takes 3 w0,
    'negative' 0, 'mid' 1.5 w0, and the smooth part is the same for every rule."""
    from tests.test_c0_output import _hawkes_1pop
    res = compute_cumulants(_hawkes_1pop(), k=2, max_ell=0, external_fields=[('n', 1)] * 2,
                            tau_grid=np.linspace(-5, 5, 21), parallel=False, verbose=False)
    c = Cumulant.from_result(res)
    w0 = sum(d.weight for d in c.deltas if d.location == 0.0)
    assert w0 != 0
    k = ME.RationalKernel([3.0, 1.0], [1.0, 3.0, 2.0], side='+')
    with pytest.raises(ValueError, match='contact'):
        c.integrate_against(k)
    got = {rule: c.integrate_against(k, contact=rule) for rule in ('positive', 'negative', 'mid')}
    for rule, l0 in (('positive', 3.0), ('negative', 0.0), ('mid', 1.5)):
        assert abs(got[rule].contact - l0 * w0) <= 1e-14 * abs(w0), rule
        assert abs(got[rule].smooth - got['positive'].smooth) <= 1e-14 * abs(got['positive'].smooth)
    q = c.integrate_against(k, contact='positive', method='quad')
    assert abs(q.smooth - got['positive'].smooth) <= 1e-10 * abs(got['positive'].smooth)


# ----------------------------------------------------------------------------- Cumulant.smooth

def test_cumulant_smooth_is_the_sum_over_loop_orders():
    import daedalus as dd
    res = compute_cumulants(dd.load_model('ou_quartic')[0], k=2, max_ell=1, external_fields=[('dx', 1)] * 2,
                            tau_grid=np.linspace(-3, 3, 13), parallel=False, verbose=False)
    c = Cumulant.from_result(res)
    parts = [np.asarray(v) for v in c.smooth_by_ell.values() if v is not None]
    assert len(parts) == 2
    np.testing.assert_allclose(c.smooth, parts[0] + parts[1], rtol=1e-15, atol=0)
    np.testing.assert_allclose(c.smooth, np.asarray(res['C_tau']), rtol=1e-14, atol=0)


# ----------------------------------------------------------------------------- phase-type jumps

def _ph_moment(alpha, T, m):
    """m! alpha (-T)^-m 1 in exact rational arithmetic (upper-triangular T)."""
    n = len(alpha)
    T = [[Fraction(x) for x in row] for row in T]
    # (-T)^-1 by back substitution on the upper-triangular -T
    U = [[-x for x in row] for row in T]
    inv = [[Fraction(0)] * n for _ in range(n)]
    for col in range(n):
        for i in range(n - 1, -1, -1):
            s = Fraction(1 if i == col else 0) - sum(U[i][j] * inv[j][col] for j in range(i + 1, n))
            inv[i][col] = s / U[i][i]
    v = [Fraction(1)] * n
    for _ in range(m):
        v = [sum(inv[i][j] * v[j] for j in range(n)) for i in range(n)]
    return math.factorial(m) * sum(Fraction(a) * x for a, x in zip(alpha, v))


def test_phase_type_from_floats_matches_the_exact_moments():
    from sage.all import SR
    alpha, T = [0.25, 0.75], [[-1.0, 0.5], [0.0, -3.0]]
    cp = N.CompoundPoisson(1, N.PhaseType(alpha, T))
    for m in (1, 2, 3, 4):
        ref = float(_ph_moment(alpha, T, m))
        assert abs(float(SR(cp.cumulant(m))) / ref - 1) < 1e-13, m
    # the same jump from rationals is exact
    cpq = N.CompoundPoisson(1, N.PhaseType(['1/4', '3/4'], [[-1, '1/2'], [0, -3]]))
    for m in (1, 2, 3):
        f = _ph_moment(alpha, T, m)
        assert SR(cpq.cumulant(m)) == SR(f.numerator) / f.denominator, m


def test_hyperexponential_from_floats_and_phase_type_with_a_symbolic_rate():
    from sage.all import SR, factorial
    cp = N.CompoundPoisson(1, N.Hyperexponential([0.25, 0.75], [1.0, 1 / 3]))
    for m in (1, 2, 3):
        ref = math.factorial(m) * (0.25 + 0.75 / 3 ** m)
        assert abs(float(SR(cp.cumulant(m))) / ref - 1) < 1e-14, m
    r = SR.var('r')
    cps = N.CompoundPoisson(1, N.PhaseType([1], [['-r']]))
    for m in (1, 2, 3):
        assert bool((SR(cps.cumulant(m)) - factorial(m) / r ** m).simplify_full() == 0), m


# ----------------------------------------------------------------------------- simulator: delayed kernel

def test_simulated_variance_of_a_delayed_kernel_sde_is_the_exact_tree():
    """x' = -a x + b (g * x) + c (g_d * x) + xi with g_d a delayed Erlang-2 kernel: the simulator's
    ring buffer delays the filter input by d.  The stationary variance of 4000 paths (dt = 0.01,
    Euler bias ~ a dt / 2) matches the exact tree to 2 %; without the delay it is 15 % lower."""
    from tests.test_audit_w5_mutation_kills import DK, _delayed_kernel_model, _spectral_reference
    from api.delay import Kernel
    from api.sde import SDE
    s = SDE('w5 delayed kernel sim')
    for k, v in DK.items():
        s = s.parameter(k, default=v, **({'domain': 'positive'} if k in ('tk', 'd', 'm', 'D') else {}))
    s = (s.define_kernel('g', kernel=Kernel.exponential('tk'))
         .define_kernel('gd', kernel=Kernel.delay('d') * Kernel.gamma(2, 'm'))
         .field('x').drift('x', '-a*x + b*Conv(g, x) + c*Conv(gd, x)')
         .noise('xi', N.Gaussian(var='2*D'), couples={'x': 1}))
    rec = s.simulate(T=60, dt=0.01, n_paths=4000, bin=1.0, burn_in=20, seed=3, record=['x'])
    assert set(rec['fields']) == {'x'} and rec['fields']['x'].shape == (60, 1, 4000)
    var = float(np.var(rec['fields']['x']))
    ref = float(_spectral_reference(0.0))
    assert abs(var / ref - 1) < 0.02, (var, ref)
    with pytest.raises(Exception, match='record'):
        s.simulate(T=1, dt=0.01, n_paths=2, record=['nope'])
