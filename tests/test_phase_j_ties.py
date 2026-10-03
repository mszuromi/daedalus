"""Tied external times at k=3 / k=4: exact ties, the Itô left limit, and the
equal-time Boltzmann oracle (M0.2b of docs/integration_speedup_plan.md, §4.2).

What is new here, relative to the existing suite
------------------------------------------------
* ``test_all_k_boltzmann.py`` (slow) already checks equal-time kappa_3
  (tree + 1-loop, drift a x^2 + b x^3) and kappa_4 (tree + 1-loop, drift
  eps x^3) against the Boltzmann series, by calling ``integrate_diagram``
  directly at all-zero times.  It never goes through ``compute_cumulants``,
  never through the grouped path, and never evaluates a PARTIAL tie.
* ``test_identical_externals_wick.py`` / ``test_fix_tau0enum.py`` cover
  k=2 only (tau=0 left limit, Wick permutations).
* ``test_daedalus_moments.py`` checks moment assembly identities, not
  cumulant values against an oracle.

This file adds:

1. k=4 (``ou_quartic``: dx = -(mu x + eps x^3) dt + sqrt(2D) dW) and k=3
   (asymmetric OU, drift a x^2 + b x^3, since the symmetric model has
   kappa_3 == 0) through ``compute_cumulants``, on the per-diagram AND the
   grouped Phase J path, with non-unit mu and D so the (D/mu)^{k/2} scaling
   is checked too:
   - fully tied times (t0=...=t_{k-1}) against the Boltzmann equal-time
     cumulant, per loop order, both at the exact tie and at the Itô nudge
     (0, -eps, ..., -eps) (the free legs tied at -eps, as the non-swept legs
     of a default ``dd.run`` k>=4 slice are);
   - partial ties (a leg tied with the anchor leg 0, two free legs tied,
     triples, two pairs) against the one-sided limit approaching the tie.
     The limit is a quadratic Richardson extrapolation from nudges
     h = 1e-4, 1e-5, 1e-6.  For white-noise OU the cumulants are continuous
     in the times (with kinks at ties), so the value at the exact tie must
     equal the Itô LEFT limit, and the right limit too.
2. ``dd.run`` k>=3 slices, so the tie policy is exercised through the API's
   k>=3 time nudge (``daedalus._kpoint_slice_times``): non-swept legs at a
   base lag of 0 go to -_ITO_EPS (all of them, still tied among themselves),
   ties among the non-swept legs reach the engine as EXACT ties, and the
   swept leg is placed one more _ITO_EPS below every leg it meets (so it
   never ties; the left-limit property of those points is checked on models
   with delta parts in ``test_kpoint_tau0_left_limit.py``).
3. kappa_3 == 0 for the symmetric ``ou_quartic`` (x -> -x parity), at every
   tie configuration.  (The plan's §4.2 wording "OU + eps x^3 has a nonzero
   stationary third cumulant at O(eps)" does not hold at the symmetric saddle
   x*=0; the nonzero-kappa_3 oracle therefore uses the a x^2 + b x^3 drift.)
4. (slow) the 2-loop equal-time kappa_4 and kappa_3 against Boltzmann.
   Neither is covered anywhere else.  kappa_4 runs at mu=1.1: at mu > 1.2 the
   k=4 ell=2 evaluation falls off the analytic path into an 8-D scipy.nquad
   for every m>=3 subset (plan problem P4); that configuration is pinned as a
   strict xfail with nquad stubbed, so it fails fast and flips when M7 lands.

OU quartic has no delta / instantaneous propagator parts, so the planned
Theta(0) constant-row fix (M1) must leave every number here unchanged, and
no evaluation here may reach scipy.nquad (asserted via
``_RUNTIME_COUNTERS``).

Oracle.  For rho(x) ∝ exp(-(mu x^2/2 + a x^3/3 + b x^4/4)/N) (N = D or T,
noise <xi xi> = 2N delta), put x = sigma y, sigma^2 = N/mu.  Then
kappa_k(x) = sigma^k kappa_k(y) with a_y = a sqrt(N)/mu^{3/2} and
b_y = b N/mu^2.  kappa_k(y) is expanded exactly (rational arithmetic) by
Gaussian moments in a bookkeeping parameter s (a -> s a, b -> s^2 b); the
ell-loop part of kappa_k is the coefficient of s^{k-2+2 ell}.

Tolerances (plan §4.4).  Oracle: 1e-9 relative, at the exact tie and at the
Itô nudge (measured <= 1.2e-15 at the exact tie, <= 2.4e-12 at the nudge).
Tie vs one-sided limit: 1e-11 relative (measured <= 6e-14; the limit is
extrapolation-limited, so 1e-11 leaves room without hiding an O(h) error:
the value at h=1e-6 itself is up to 8e-7 away from the tie).

Runtime (measured): default-suite part ~36 s; slow part ~45 s (k=4 ell=2),
~40 s (the stubbed xfail), ~20 s (k=3 ell=2).

Run:  sage -python -m pytest tests/test_phase_j_ties.py -q
      sage -python -m pytest tests/test_phase_j_ties.py -q -m slow
"""
import functools
import os
import sys
from fractions import Fraction
from math import factorial

import numpy as np
import pytest
import matplotlib
matplotlib.use('Agg')

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', 'notebooks')))

import api                                                  # noqa: E402
import daedalus as dd                                       # noqa: E402
from api import compute_cumulants                           # noqa: E402
from api.compute import _ITO_EPS                            # noqa: E402
from api.model import TemporalModelBuilder                  # noqa: E402
import engine.integration.time_domain.final_integral as FI  # noqa: E402

_RTOL_ORACLE = 1e-9      # plan §4.4, oracle comparisons
_RTOL_TIE = 1e-11        # exact tie vs Richardson one-sided limit
_H = (1e-4, 1e-5, 1e-6)  # nudges for the one-sided limit

# Non-unit mu and noise so the sigma^k scaling is exercised.
_P4 = {'mu': 1.25, 'D': 0.8, 'eps': 0.05}             # ou_quartic
_P3 = {'mu': 1.25, 'T': 0.8, 'a': 0.05, 'b': 0.05}    # asymmetric OU
# k=4 at 2 loops: mu must stay <= 1.2 (see the xfail test at the end).
_P4_L2 = {'mu': 1.1, 'D': 0.8, 'eps': 0.05}
_P4_L2_CLIFF = {'mu': 1.25, 'D': 0.8, 'eps': 0.05}

_PATHS = [False, True]
_PATH_IDS = ['perdiag', 'grouped']


# ---------------------------------------------------------------------------
# Boltzmann oracle
# ---------------------------------------------------------------------------

def _dfact(n):
    r = 1
    while n > 1:
        r *= n
        n -= 2
    return r


def _boltzmann_series(kmax, a, b, order):
    """kappa_n(y) = sum_j c[n][j] s^j, exact rationals, for
    rho(y) ∝ exp(-(y^2/2 + s a y^3/3 + s^2 b y^4/4)), n = 1..kmax,
    truncated at s^order."""
    a, b = Fraction(a), Fraction(b)
    w = {}      # (s power, y power) -> coefficient of exp(-s a y^3/3 - s^2 b y^4/4)
    for n in range(order + 1):
        for i in range(n + 1):              # i cubic factors, n - i quartic
            sp = i + 2 * (n - i)
            if sp > order:
                continue
            c = (Fraction(1, factorial(i) * factorial(n - i))
                 * (-a / 3) ** i * (-b / 4) ** (n - i))
            key = (sp, 3 * i + 4 * (n - i))
            w[key] = w.get(key, 0) + c

    def gauss(extra):                       # E_0[y^extra * weight], series in s
        out = [Fraction(0)] * (order + 1)
        for (sp, yp), c in w.items():
            if (yp + extra) % 2 == 0:
                out[sp] += c * _dfact(yp + extra - 1)
        return out

    def mul(p, q):
        r = [Fraction(0)] * (order + 1)
        for i, pi in enumerate(p):
            if pi:
                for j in range(order + 1 - i):
                    r[i + j] += pi * q[j]
        return r

    z = gauss(0)
    zinv = [Fraction(0)] * (order + 1)
    zinv[0] = 1 / z[0]
    for n in range(1, order + 1):
        zinv[n] = -sum(z[j] * zinv[n - j] for j in range(1, n + 1)) / z[0]
    mom = {n: mul(gauss(n), zinv) for n in range(1, kmax + 1)}
    kap = {}
    for n in range(1, kmax + 1):            # moments -> cumulants
        kn = list(mom[n])
        for j in range(1, n):
            c = factorial(n - 1) // (factorial(j - 1) * factorial(n - j))
            kn = [x - c * y for x, y in zip(kn, mul(kap[j], mom[n - j]))]
        kap[n] = kn
    return kap


def _kappa_oracle(k, ell, mu, noise, a, b):
    """ell-loop part of the equal-time Boltzmann kappa_k for the drift
    -(mu x + a x^2 + b x^3) with noise strength ``noise`` (<xi xi> = 2 noise)."""
    sigma = np.sqrt(noise / mu)
    kap = _boltzmann_series(k, a * np.sqrt(noise) / mu ** 1.5,
                            b * noise / mu ** 2, k - 2 + 2 * ell)
    return sigma ** k * float(kap[k][k - 2 + 2 * ell])


def _oracle4(ell, p=_P4):
    return _kappa_oracle(4, ell, p['mu'], p['D'], 0.0, p['eps'])


def _oracle3(ell):
    return _kappa_oracle(3, ell, _P3['mu'], _P3['T'], _P3['a'], _P3['b'])


def test_boltzmann_series_helper_reproduces_known_coefficients():
    """The oracle helper reproduces the series already pinned elsewhere in
    the suite (kappa_2 = 1 - 3b + 24b^2, test_identical_externals_wick;
    kappa_3 = -2a + 42ab - 32a^3 and kappa_4 = -6b + 126b^2 at a=0,
    test_all_k_boltzmann) plus the 2-loop terms derived independently with a
    sympy series expansion (M0.2b): kappa_4 ⊃ -2682 b^3 at a=0 and
    kappa_3 ⊃ -700a^5 + 1908a^3 b - 894 a b^2."""
    for a, b in [(Fraction(1, 20), Fraction(1, 20)), (Fraction(1, 7), Fraction(2, 9))]:
        kap = _boltzmann_series(4, a, b, 6)
        assert kap[2][:5] == [1, 0, 4 * a**2 - 3 * b, 0,
                              50 * a**4 - 109 * a**2 * b + 24 * b**2]
        assert kap[3][1] == -2 * a
        assert kap[3][3] == 42 * a * b - 32 * a**3
        assert kap[3][5] == -700 * a**5 + 1908 * a**3 * b - 894 * a * b**2
        assert kap[4][2] == 12 * a**2 - 6 * b
        assert kap[4][4] == 384 * a**4 - 708 * a**2 * b + 126 * b**2
        assert all(c == 0 for c in kap[3][0::2]) and kap[4][:2] == [0, 0]
    kap0 = _boltzmann_series(4, 0, Fraction(1, 20), 6)
    assert kap0[4][2] == -6 * Fraction(1, 20)
    assert kap0[4][6] == -2682 * Fraction(1, 20) ** 3
    assert all(c == 0 for c in kap0[3])


# ---------------------------------------------------------------------------
# Runs (cached per module; use_cache=False keeps the suite hermetic)
# ---------------------------------------------------------------------------

def _asym_model():
    return (TemporalModelBuilder('ou-asymmetric-ties')
            .physical_field('x')
            .parameter('mu', default=1.0, domain='positive')
            .parameter('T', default=1.0, domain='positive')
            .parameter('a', default=0.05)
            .parameter('b', default=0.05)
            .set_action_text('xt*((Dt+mu)*x + a*x^2 + b*x^3) - T*xt^2')
            .equation(lhs='(Dt+mu)*x + a*x^2 + b*x^3', rhs='0')
            .build())


def _compute(k, grouped, max_ell, params):
    model = dd.load_model('ou_quartic')[0] if k == 4 else _asym_model()
    return compute_cumulants(
        model, k=k, max_ell=max_ell, external_fields=[('dx', 1)] * k,
        parameters=dict(params), tau_grid=np.array([0.0]), verbose=False,
        use_cache=False, parallel=False, use_grouped_phase_j=grouped)


@functools.lru_cache(maxsize=None)
def _run(k, grouped, max_ell, params):
    return _compute(k, grouped, max_ell, params)


def _fns(k, grouped, max_ell=1, params=None):
    if params is None:
        params = _P4 if k == 4 else _P3
    fns = _run(k, grouped, max_ell, tuple(sorted(params.items())))[
        'total_C_by_ell']
    assert sorted(fns) == list(range(max_ell + 1))
    return fns


def _val(fn, args):
    v = complex(fn(*args))
    assert np.isfinite(v.real) and abs(v.imag) < 1e-12, (args, v)
    return v.real


def _one_sided_limit(fn, tie, direction, hs=_H):
    """Quadratic Richardson extrapolation to h=0 of fn(tie + h*direction)."""
    vals = [_val(fn, tuple(t + d * h for t, d in zip(tie, direction)))
            for h in hs]
    out = 0.0
    for i, hi in enumerate(hs):
        w = 1.0
        for j, hj in enumerate(hs):
            if j != i:
                w *= (0.0 - hj) / (hi - hj)
        out += w * vals[i]
    return out


def _split_left_direction(args):
    """Approach direction that breaks every exact tie from the LEFT: within a
    group of equal times the first-listed leg stays, the later legs move to
    -1, -2, ... (earlier times).  Leg 0 (the anchor) is always first."""
    d = [0] * len(args)
    seen = {}
    for i, t in enumerate(args):
        n = seen.get(t, 0)
        d[i] = -n
        seen[t] = n + 1
    return tuple(d)


def _reset_nquad():
    FI._reset_runtime_counters()


def _assert_no_nquad():
    c = FI._RUNTIME_COUNTERS
    n = (c['scipy_nquad_called_m1'] + c['scipy_nquad_called_m2']
         + c['scipy_nquad_called_mge3'])
    assert n == 0, {kk: v for kk, v in c.items() if v}


def _assert_rel(got, ref, rtol, what):
    err = abs(got - ref) / abs(ref)
    assert err <= rtol, f'{what}: got {got!r} ref {ref!r} rel err {err:.3e}'


# ---------------------------------------------------------------------------
# 1. Fully tied times vs the Boltzmann equal-time cumulant (ell <= 1)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('grouped', _PATHS, ids=_PATH_IDS)
@pytest.mark.parametrize('k', [3, 4])
def test_full_tie_matches_boltzmann(k, grouped):
    """kappa_k(t, ..., t) per loop order = Boltzmann coefficient, at the exact
    tie and at the API's Itô nudge (0, -eps, ..., -eps)."""
    fns = _fns(k, grouped)
    oracle = _oracle4 if k == 4 else _oracle3
    _reset_nquad()
    for ell in (0, 1):
        ref = oracle(ell)
        exact = _val(fns[ell], (0.0,) * k)
        nudged = _val(fns[ell], (0.0,) + (-_ITO_EPS,) * (k - 1))
        _assert_rel(exact, ref, _RTOL_ORACLE, f'k={k} ell={ell} exact tie')
        _assert_rel(nudged, ref, _RTOL_ORACLE, f'k={k} ell={ell} Itô nudge')
    _assert_no_nquad()


# ---------------------------------------------------------------------------
# 2. Partial ties: value at the exact tie == one-sided (Itô left) limit
# ---------------------------------------------------------------------------

# (tie point, approach directions).  Directions with only non-positive
# entries are left (Itô) limits; the positive ones check continuity.
_K4_TIES = {
    't0=t1':          ((0.0, 0.0, 0.3, 0.7), [(0, -1, 0, 0), (0, 1, 0, 0)]),
    't1=t2':          ((0.0, 0.3, 0.3, 0.7),
                       [(0, 0, -1, 0), (0, -1, 0, 0), (0, 0, 1, 0)]),
    't0=t1=t2':       ((0.0, 0.0, 0.0, 0.7),
                       [(0, -1, -1, 0), (0, -1, -2, 0), (0, 1, 2, 0)]),
    't1=t2=t3':       ((0.0, 0.3, 0.3, 0.3),
                       [(0, 0, -1, -2), (0, 0, -1, -1), (0, 2, 1, 0)]),
    't0=t1,t2=t3':    ((0.0, 0.0, 0.5, 0.5), [(0, -1, 0, -1), (0, 1, 0, 1)]),
    't0=t1=t2=t3':    ((0.0, 0.0, 0.0, 0.0),
                       [(0, -1, -1, -1), (0, -1, -2, -3), (0, 1, 2, 3)]),
}
_K3_TIES = {
    't0=t1':    ((0.0, 0.0, 0.5), [(0, -1, 0), (0, 1, 0)]),
    't1=t2':    ((0.0, 0.5, 0.5), [(0, 0, -1), (0, -1, 0), (0, 0, 1)]),
    't0=t1=t2': ((0.0, 0.0, 0.0), [(0, -1, -1), (0, -1, -2), (0, 1, 2)]),
}


def _check_tie_limits(fns, tie, dirs, label):
    for ell in sorted(fns):
        at_tie = _val(fns[ell], tie)
        for d in dirs:
            lim = _one_sided_limit(fns[ell], tie, d)
            _assert_rel(at_tie, lim, _RTOL_TIE,
                        f'{label} ell={ell} tie {tie} vs limit along {d}')


@pytest.mark.parametrize('grouped', _PATHS, ids=_PATH_IDS)
@pytest.mark.parametrize('case', list(_K4_TIES))
def test_k4_tie_equals_ito_left_limit(case, grouped):
    tie, dirs = _K4_TIES[case]
    _reset_nquad()
    _check_tie_limits(_fns(4, grouped), tie, dirs, f'k=4 {case}')
    _assert_no_nquad()


@pytest.mark.parametrize('grouped', _PATHS, ids=_PATH_IDS)
@pytest.mark.parametrize('case', list(_K3_TIES))
def test_k3_tie_equals_ito_left_limit(case, grouped):
    tie, dirs = _K3_TIES[case]
    _reset_nquad()
    _check_tie_limits(_fns(3, grouped), tie, dirs, f'k=3 {case}')
    _assert_no_nquad()


def test_symmetric_ou_quartic_kappa3_vanishes_at_ties():
    """x -> -x parity of OU + eps x^3 at x*=0: kappa_3 == 0 at every tie."""
    model, _ = dd.load_model('ou_quartic')
    res = compute_cumulants(
        model, k=3, max_ell=1, external_fields=[('dx', 1)] * 3,
        parameters=_P4, tau_grid=np.array([0.0]), verbose=False,
        use_cache=False, parallel=False)
    for ell, fn in res['total_C_by_ell'].items():
        for args in [(0.0, 0.0, 0.0), (0.0, -_ITO_EPS, -_ITO_EPS),
                     (0.0, 0.0, 0.5), (0.0, 0.5, 0.5), (0.0, 0.2, 0.9)]:
            assert abs(complex(fn(*args))) < 1e-15, (ell, args)


# ---------------------------------------------------------------------------
# 3. Through the API: dd.run k>=3 slices and the ``daedalus`` time nudge
# ---------------------------------------------------------------------------

def _dd_run(monkeypatch, model, module, cfg):
    # dd.run calls ``api.compute_cumulants`` with the default use_cache=True;
    # force use_cache=False so this test never writes the on-disk caches.
    orig = api.compute_cumulants
    monkeypatch.setattr(api, 'compute_cumulants',
                        lambda **kw: orig(**{**kw, 'use_cache': False}))
    return dd.run(model, cfg, module)


def _slice_args(k, base, j, tau):
    """What ``dd.run`` passes for slice j at tau (``daedalus.
    _kpoint_slice_times``): the non-swept legs at their base lag, or at
    -_ITO_EPS when it is within 1e-12 of 0; the swept leg at tau, unless tau
    is within 1e-12 of the anchor or of a non-swept leg's lag or time -- then
    one more _ITO_EPS below the earliest leg it meets (the left limit of the
    slice).  The bases used here never need the repeated step."""
    a = [0.0] * k
    pinned = [i for i in range(1, k) if i != j]
    for i in pinned:
        b = float(base[i - 1])
        a[i] = b if abs(b) > 1e-12 else -_ITO_EPS
    t = float(tau)
    met = ([0.0] if abs(t) <= 1e-12 else []) + [
        a[i] for i in pinned
        if abs(t - float(base[i - 1])) <= 1e-12 or abs(t - a[i]) <= 1e-12]
    if met:
        t = min(met) - _ITO_EPS
    a[j] = t
    assert all(abs(t - a[i]) > 1e-12 for i in range(k) if i != j)
    return tuple(a)


def _check_api_equal_time(res, k, oracle):
    tau = np.asarray(res['tau_grid'])
    i0 = int(np.argmin(np.abs(tau)))
    assert tau[i0] == 0.0
    for j in range(1, k):
        by_ell = res['C_tau_slices_by_ell'][j]
        for ell in (0, 1):
            _assert_rel(float(np.real(by_ell[ell][i0])), oracle(ell),
                        _RTOL_ORACLE, f'dd.run k={k} slice {j} ell={ell} tau=0')
        _assert_rel(float(np.real(res['C_tau_slices'][j][i0])),
                    oracle(0) + oracle(1), _RTOL_ORACLE,
                    f'dd.run k={k} slice {j} total tau=0')


def _check_api_ties(res, k, base, tie_count_min):
    """Every slice point is the callable at the ``_slice_args`` times (also
    recorded in ``res['_kpoint_slice_times']``); where those times contain
    an exact tie between legs (non-swept legs only) it must equal the Itô
    left limit of the callable."""
    fns = res['total_C_by_ell']
    tau = np.asarray(res['tau_grid'])
    n_ties = 0
    for j in range(1, k):
        for i, t in enumerate(tau):
            args = _slice_args(k, base, j, t)
            assert args == tuple(res['_kpoint_slice_times'][j][i]), (j, t)
            has_tie = len(set(args)) < len(args)
            n_ties += has_tie
            for ell in (0, 1):
                got = float(np.real(res['C_tau_slices_by_ell'][j][ell][i]))
                _assert_rel(got, _val(fns[ell], args), 1e-13,
                            f'dd.run slice {j} tau={t} ell={ell} vs callable')
                if has_tie:
                    lim = _one_sided_limit(fns[ell], args,
                                           _split_left_direction(args))
                    _assert_rel(got, lim, _RTOL_TIE,
                                f'dd.run slice {j} tie {args} ell={ell}')
    assert n_ties >= tie_count_min, n_ties


def test_api_k4_slices_at_ties(monkeypatch):
    model, module = dd.load_model('ou_quartic')
    _reset_nquad()
    # (a) default base: the two non-swept legs sit at -_ITO_EPS (tied), the
    #     swept leg at tau (at tau=0: -2 _ITO_EPS) -> at tau=0 the equal-time
    #     Boltzmann kappa_4 (the cumulant is continuous there).
    res = _dd_run(monkeypatch, model, module, dd.Config(
        k=4, max_ell=1, parameters=_P4, external_fields=[('dx', 1)] * 4,
        tau_grid=np.array([0.0, 0.3])))
    assert _slice_args(4, [0.0] * 3, 1, 0.0) == (
        0.0, -2 * _ITO_EPS, -_ITO_EPS, -_ITO_EPS)
    _check_api_equal_time(res, 4, _oracle4)
    _check_api_ties(res, 4, [0.0, 0.0, 0.0], tie_count_min=6)
    # (b) base lags with free-leg ties: ties among the non-swept legs reach
    #     the engine UN-nudged (e.g. (0, .3, .3, .7) on slice 3); a swept leg
    #     meeting a base lag goes one _ITO_EPS below it.
    base = [0.3, 0.3, 0.7]
    res = _dd_run(monkeypatch, model, module, dd.Config(
        k=4, max_ell=1, parameters=_P4, external_fields=[('dx', 1)] * 4,
        tau_grid=np.array([0.0, 0.3, 0.7]), kpoint_base_lags=base))
    assert _slice_args(4, base, 3, 0.7) == (0.0, 0.3, 0.3, 0.7)
    assert _slice_args(4, base, 3, 0.3) == (0.0, 0.3, 0.3, 0.3 - _ITO_EPS)
    assert _slice_args(4, base, 1, 0.7) == (0.0, 0.7 - _ITO_EPS, 0.3, 0.7)
    _check_api_ties(res, 4, base, tie_count_min=3)
    _assert_no_nquad()


def test_api_k3_slices_at_ties(monkeypatch):
    """k=3 has one non-swept leg, so no dd.run k=3 point is a tie any more
    (the swept leg goes below the leg it meets); the points must still be
    the callable at those times, and tau=0 the equal-time Boltzmann value."""
    model = _asym_model()
    _reset_nquad()
    res = _dd_run(monkeypatch, model, None, dd.Config(
        k=3, max_ell=1, parameters=_P3, external_fields=[('dx', 1)] * 3,
        tau_grid=np.array([0.0, 0.5])))
    _check_api_equal_time(res, 3, _oracle3)
    _check_api_ties(res, 3, [0.0, 0.0], tie_count_min=0)
    base = [0.5, 0.5]
    res = _dd_run(monkeypatch, model, None, dd.Config(
        k=3, max_ell=1, parameters=_P3, external_fields=[('dx', 1)] * 3,
        tau_grid=np.array([0.0, 0.5]), kpoint_base_lags=base))
    assert _slice_args(3, base, 1, 0.5) == (0.0, 0.5 - _ITO_EPS, 0.5)
    _check_api_ties(res, 3, base, tie_count_min=0)
    _assert_no_nquad()


# ---------------------------------------------------------------------------
# 4. Two loops (slow)
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_k4_two_loop_full_tie_matches_boltzmann():
    """kappa_4 at 2 loops (-2682 b^3, scaled) at the exact full tie and at
    the Itô nudge, mu=1.1 (below the P4 cliff, see the next test).  ~45 s,
    ~35 s of it enumeration/setup.  Partial ties are not repeated at 2 loops:
    a nudged k=4 ell=2 evaluation costs ~17 s (measured), so a Richardson
    triple would break the < 60 s budget."""
    fns = _fns(4, False, max_ell=2, params=_P4_L2)
    _reset_nquad()
    ref = _oracle4(2, _P4_L2)
    _assert_rel(_val(fns[2], (0.0,) * 4), ref, _RTOL_ORACLE, 'k=4 ell=2 tie')
    _assert_rel(_val(fns[2], (0.0,) + (-_ITO_EPS,) * 3), ref, _RTOL_ORACLE,
                'k=4 ell=2 Itô nudge')
    _assert_no_nquad()


@pytest.mark.slow
@pytest.mark.xfail(strict=True, reason=(
    'Plan P4 (margin-50 chain lower bound): for ou_quartic k=4 ell=2 at '
    'mu=1.25 EVERY one of the 936 m>=3 poset subsets bails with '
    'poset_chain_none -- _exp_over_chain_simplex_polynomial returns None '
    'because the chain alphas (+-2mu) reach a cumulative Re beta of 5*2.5 = '
    '12.5 and |Re beta * L| = 625 > EXP_REAL_LIMIT=600 at L = earliest_ext '
    '- POSET_PHYSICAL_MARGIN = -50 (exp(-625) is a harmless underflow).  '
    'Each subset then goes to an 8-D scipy.nquad (1872 calls per '
    'evaluation); one unstubbed evaluation did not finish in 15 min.  '
    'mu = 0.9, 1.0, 1.01, 1.1 have 0 fallbacks and match Boltzmann to '
    '<= 3e-15.  Expected to XPASS once the semi-infinite chain (L4/F6, M7) '
    'lands -- then drop this marker.'))
def test_k4_two_loop_no_nquad_fallback_above_mu_1p2(monkeypatch):
    """Same full-tie oracle at mu=1.25, with scipy.nquad stubbed so the
    fallback is COUNTED instead of run (an unstubbed run takes > 15 min)."""
    import scipy.integrate
    calls = []

    def _stub_nquad(func, ranges, *args, **kwargs):
        calls.append(len(ranges))
        return 0.0, 0.0

    monkeypatch.setattr(scipy.integrate, 'nquad', _stub_nquad)
    fns = _compute(4, False, 2, tuple(sorted(_P4_L2_CLIFF.items())))[
        'total_C_by_ell']
    got = _val(fns[2], (0.0,) * 4)
    assert not calls, (f'{len(calls)} scipy.nquad calls (integration dims '
                       f'{sorted(set(calls))}); stubbed value {got!r}')
    _assert_rel(got, _oracle4(2, _P4_L2_CLIFF), _RTOL_ORACLE,
                'k=4 ell=2 mu=1.25 tie')


@pytest.mark.slow
def test_k3_two_loop_full_tie_matches_boltzmann():
    """kappa_3 at 2 loops (-700a^5 + 1908a^3 b - 894 a b^2, scaled) at the
    exact tie and at the Itô nudge.  ~20 s.  (A nudged k=3 ell=2 evaluation
    off the full tie costs ~20 s, so partial ties stay at ell <= 1.)"""
    fns = _fns(3, False, max_ell=2)
    _reset_nquad()
    ref = _oracle3(2)
    _assert_rel(_val(fns[2], (0.0,) * 3), ref, _RTOL_ORACLE, 'k=3 ell=2 tie')
    _assert_rel(_val(fns[2], (0.0, -_ITO_EPS, -_ITO_EPS)), ref, _RTOL_ORACLE,
                'k=3 ell=2 Itô nudge')
    _assert_no_nquad()
