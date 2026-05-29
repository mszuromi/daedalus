"""
Phase 5 math spike  [Phase 0 de-risk artifact, 2026-05-28 — GO]
================================================================
Committed reference for the two load-bearing formulas of the
spatial v1 plan.  Run BEFORE any framework code as the Phase 0
go/no-go check; all five tests passed at machine precision,
confirming the Rescue A premise (Decision 4 in
docs/spatial_design_decisions_v1.md).

Preserved here (not left in /tmp) because the erf-split closed
form below is exactly what Phase 5's
`_integrate_1d_polytope_with_erfc` must implement — this file's
regime tables seed `tests/test_1d_polytope_erfc.py`, and the
semigroup-collapse check seeds Phase 2's heat_kernel.py tests.

RESULT (sage -python docs/spatial_spikes/phase5_erfsplit_semigroup_spike.py):
  1a real-α erf-split        PASS  (≤6e-15 across 7 regimes)
  1b complex-α erf-split     PASS  (~1e-25..1e-31; incl. near-pure-
                                    imaginary α=0.05+2j stress case)
  1c semi-infinite U→∞        PASS  (≤2e-16)
  2  semigroup collapse       PASS  (chain-2 + chain-3, ≤3e-16)
  3  end-to-end tree C(x,τ)   PASS  (3-way; τ=0 reproduces
                                    T/(2√(μD))·exp(-|x|/ξ) exactly)

CAVEAT for the Phase 5 implementer: this spike uses mpmath at 30
dps throughout, so it confirms the *math* is correct and stable
in extended precision.  The float64 production path will need the
`USE_ERFC_CHAIN_SIMPLEX_PRECISION_FIX` mpmath gate (already in the
plan) for the close-erf-argument regime — erf(α+β/√U) − erf(α+β/√L)
cancels catastrophically in float64 when those arguments are near
each other, the exact analog of the close-paired-pole concern in
docs/m_ge3_precision_bug_audit.md.

----------------------------------------------------------------
Original spike intent:

De-risk the two load-bearing claims of the spatial v1 plan BEFORE
writing any framework code.

Claim 1 (Rescue A premise): the residual temporal integral after
spatial collapse,

    I = ∫_L^U  s^{-1/2}  exp[ -β/s - α s ]  ds,   (T(s)=s case)

has a clean erf-split closed form.  Verify:
  1a. erf-split antiderivative correctness (real α), many regimes
  1b. complex α (the "does Rescue A generalize past free Allen-Cahn"
      stress case — multi-pole chains give complex α)
  1c. semi-infinite limit U→∞ (this is the τ-correlator integral)

Claim 2 (Phase 5.3): the spatial Gaussian-convolution semigroup
collapse,

    ∫ dx_v  G(t1, x-x_v) G(t2, x_v-x')  =  G(t1+t2, x-x'),

and the chain-of-3 generalization.

End-to-end (Claim 1 ⊗ Claim 2): the tree-level free two-time
correlator C(x, τ) built from these primitives must reproduce the
known closed form, and at τ=0 reduce to T/(2√(μD)) exp(-|x|/ξ).

All reference integrals use scipy.quad (real) or mpmath.quad
(complex); the closed forms use mpmath.erf (handles complex).

Run:  sage -python docs/spatial_spikes/phase5_erfsplit_semigroup_spike.py
"""
from __future__ import annotations

import cmath
import math

import mpmath as mp
import numpy as np
from scipy import integrate

mp.mp.dps = 30  # 30-digit reference precision

PASS = '\033[92mPASS\033[0m'
FAIL = '\033[91mFAIL\033[0m'


def _verdict(name, rel, tol):
    tag = PASS if rel <= tol else FAIL
    print(f'  [{tag}] {name}: rel={rel:.3e} (tol {tol:.0e})')
    return rel <= tol


# ─────────────────────────────────────────────────────────────────
# Closed forms
# ─────────────────────────────────────────────────────────────────
def F_erf_split(w, alpha, beta):
    """Antiderivative of exp(-alpha w^2 - beta/w^2) via the erf split.
    alpha, beta may be complex; uses mpmath for complex erf."""
    a = mp.sqrt(alpha)
    b = mp.sqrt(beta)
    pref = mp.sqrt(mp.pi) / (4 * a)
    e_plus = mp.e ** (2 * a * b)
    e_minus = mp.e ** (-2 * a * b)
    erf_plus = mp.erf(a * w + b / w)
    erf_minus = mp.erf(a * w - b / w)
    return pref * (e_plus * erf_plus + e_minus * erf_minus)


def F_at_zero(alpha, beta):
    """w→0⁺ limit of F_erf_split.  For β>0: erf(b/w→+∞)=1,
    erf(-b/w→-∞)=-1 ⇒ F = pref·(e^{2ab} - e^{-2ab}).  For β=0:
    erf(0)=0 both terms ⇒ F = 0."""
    if beta == 0:
        return mp.mpf(0)
    a = mp.sqrt(alpha)
    b = mp.sqrt(beta)
    pref = mp.sqrt(mp.pi) / (4 * a)
    return pref * (mp.e ** (2 * a * b) - mp.e ** (-2 * a * b))


def _F(w, alpha, beta):
    """F_erf_split with the w→0 limit handled."""
    if w == 0:
        return F_at_zero(alpha, beta)
    return F_erf_split(w, alpha, beta)


def I_closed(L, U, alpha, beta):
    """Closed form for ∫_L^U s^{-1/2} exp(-beta/s - alpha s) ds.
    U may be mp.inf for the semi-infinite case (real positive alpha).
    L may be 0 (handled via the w→0⁺ limit)."""
    wL = mp.sqrt(L)
    if U == mp.inf:
        # erf(±∞-ish) → 1 for Re(sqrt(alpha)) > 0
        a = mp.sqrt(alpha)
        b = mp.sqrt(beta)
        pref = mp.sqrt(mp.pi) / (4 * a)
        F_inf = pref * (mp.e ** (2 * a * b) + mp.e ** (-2 * a * b))
        return 2 * (F_inf - _F(wL, alpha, beta))
    wU = mp.sqrt(U)
    return 2 * (_F(wU, alpha, beta) - _F(wL, alpha, beta))


# ─────────────────────────────────────────────────────────────────
# Reference integrators
# ─────────────────────────────────────────────────────────────────
def I_ref_real(L, U, alpha, beta):
    """scipy.quad reference for real alpha, beta."""
    f = lambda s: s ** -0.5 * math.exp(-beta / s - alpha * s)
    val, err = integrate.quad(f, L, U, limit=400, epsabs=1e-13, epsrel=1e-13)
    return val


def I_ref_complex(L, U, alpha, beta):
    """mpmath.quad reference for complex alpha (beta real)."""
    f = lambda s: s ** mp.mpf('-0.5') * mp.e ** (-beta / s - alpha * s)
    return mp.quad(f, [L, U])


# ─────────────────────────────────────────────────────────────────
# Heat kernel
# ─────────────────────────────────────────────────────────────────
def G_heat(t, x, D, mu):
    """1D retarded heat-kernel propagator G_R(t, x)."""
    if t <= 0:
        return 0.0
    return 1.0 / math.sqrt(4 * math.pi * D * t) * math.exp(
        -x * x / (4 * D * t) - mu * t)


# ═════════════════════════════════════════════════════════════════
# TEST 1a — erf-split antiderivative, real alpha, many regimes
# ═════════════════════════════════════════════════════════════════
def test_1a():
    print('\n=== TEST 1a: erf-split closed form vs scipy.quad (real α) ===')
    # (alpha, beta, L, U)
    regimes = [
        ('small μ, small X, mid interval', 0.1, 0.01, 0.5, 5.0),
        ('large μ, small X, mid interval', 10.0, 0.01, 0.5, 5.0),
        ('small μ, large X, mid interval', 0.1, 10.0, 0.5, 5.0),
        ('large μ, large X, mid interval', 10.0, 10.0, 0.5, 5.0),
        ('mid μ, mid X, wide interval',    1.0, 1.0, 0.01, 100.0),
        ('mid μ, tiny X (→pure decay)',    1.0, 1e-6, 0.1, 50.0),
        ('mid μ, mid X, near-origin L',    1.0, 1.0, 1e-3, 20.0),
    ]
    ok = True
    for name, alpha, beta, L, U in regimes:
        clf = complex(I_closed(L, U, alpha, beta))
        ref = I_ref_real(L, U, alpha, beta)
        rel = abs(clf.real - ref) / max(abs(ref), 1e-300)
        ok &= _verdict(name, rel, 1e-10)
    return ok


# ═════════════════════════════════════════════════════════════════
# TEST 1b — complex alpha (multi-pole-chain generality stress)
# ═════════════════════════════════════════════════════════════════
def test_1b():
    print('\n=== TEST 1b: erf-split vs mpmath.quad (complex α) ===')
    # Complex alpha = mu + i*omega: damped-oscillatory pole pairs.
    regimes = [
        ('damped osc, weak imag',  mp.mpc(1.0, 0.5), mp.mpf(1.0), 0.5, 5.0),
        ('damped osc, strong imag', mp.mpc(1.0, 3.0), mp.mpf(1.0), 0.5, 5.0),
        ('damped osc, large X',    mp.mpc(1.0, 2.0), mp.mpf(10.0), 0.5, 5.0),
        ('weak damp, mid imag',    mp.mpc(0.3, 1.5), mp.mpf(1.0), 0.5, 5.0),
        ('near-pure-imag (stress)', mp.mpc(0.05, 2.0), mp.mpf(1.0), 0.5, 5.0),
    ]
    ok = True
    for name, alpha, beta, L, U in regimes:
        clf = I_closed(L, U, alpha, beta)
        ref = I_ref_complex(L, U, alpha, beta)
        rel = abs(clf - ref) / max(abs(ref), mp.mpf('1e-300'))
        ok &= _verdict(name, float(rel), 1e-10)
    return ok


# ═════════════════════════════════════════════════════════════════
# TEST 1c — semi-infinite limit U→∞ (the τ-correlator integral)
# ═════════════════════════════════════════════════════════════════
def test_1c():
    print('\n=== TEST 1c: semi-infinite U→∞ closed form vs scipy.quad ===')
    regimes = [
        ('μ=1, X²/4D=1, L=0.5', 1.0, 1.0, 0.5),
        ('μ=0.5, β=2, L=0.1',   0.5, 2.0, 0.1),
        ('μ=2, β=0.5, L=1.0',   2.0, 0.5, 1.0),
        ('μ=0.1, β=0.01, L=0.5 (slow decay)', 0.1, 0.01, 0.5),
    ]
    ok = True
    for name, alpha, beta, L in regimes:
        clf = complex(I_closed(L, mp.inf, alpha, beta))
        # scipy on [L, large-cutoff] — pick cutoff where e^{-αs} is tiny
        cutoff = L + 50.0 / alpha
        ref = I_ref_real(L, cutoff, alpha, beta)
        rel = abs(clf.real - ref) / max(abs(ref), 1e-300)
        ok &= _verdict(name, rel, 1e-9)
    return ok


# ═════════════════════════════════════════════════════════════════
# TEST 2 — spatial Gaussian-convolution semigroup collapse
# ═════════════════════════════════════════════════════════════════
def test_2():
    print('\n=== TEST 2: heat-kernel semigroup collapse ===')
    D, mu = 1.3, 0.7
    ok = True

    # Chain of 2: ∫ dx_v G(t1, x-x_v) G(t2, x_v-x') = G(t1+t2, x-x')
    cases2 = [
        (1.0, 2.0, 3.0, -1.0),
        (0.5, 0.5, 0.0, 0.0),
        (2.0, 0.3, 5.0, 2.0),
    ]
    for t1, t2, x, xp in cases2:
        f = lambda xv: G_heat(t1, x - xv, D, mu) * G_heat(t2, xv - xp, D, mu)
        conv, _ = integrate.quad(f, -50, 50, limit=400, epsabs=1e-13)
        exact = G_heat(t1 + t2, x - xp, D, mu)
        rel = abs(conv - exact) / max(abs(exact), 1e-300)
        ok &= _verdict(f'chain-2 (t1={t1},t2={t2},x={x},xp={xp})', rel, 1e-8)

    # Chain of 3: ∫∫ dxv dxw G(t1,x-xv)G(t2,xv-xw)G(t3,xw-xp)
    #            = G(t1+t2+t3, x-xp)
    t1, t2, t3, x, xp = 1.0, 1.5, 0.7, 2.0, -1.0
    def inner(xw):
        f = lambda xv: G_heat(t1, x - xv, D, mu) * G_heat(t2, xv - xw, D, mu)
        v, _ = integrate.quad(f, -40, 40, limit=200, epsabs=1e-12)
        return v * G_heat(t3, xw - xp, D, mu)
    conv3, _ = integrate.quad(inner, -40, 40, limit=200, epsabs=1e-11)
    exact3 = G_heat(t1 + t2 + t3, x - xp, D, mu)
    rel3 = abs(conv3 - exact3) / max(abs(exact3), 1e-300)
    ok &= _verdict('chain-3 (D_eff/mu_eff bookkeeping)', rel3, 1e-6)
    return ok


# ═════════════════════════════════════════════════════════════════
# TEST 3 — end-to-end tree-level free two-time correlator
# ═════════════════════════════════════════════════════════════════
def test_3():
    print('\n=== TEST 3: tree-level free correlator C(x,τ), 3-way check ===')
    D, mu, T = 1.0, 1.0, 1.0
    xi = math.sqrt(D / mu)   # correlation length
    ok = True

    def C_closed(x, tau):
        """C(x,τ) = (T/√(4πD)) ∫_τ^∞ s^{-1/2} exp(-x²/(4Ds) - μ s) ds
        via the erf-split semi-infinite closed form."""
        beta = x * x / (4 * D)
        I = I_closed(max(tau, 0.0), mp.inf, mu, beta)
        return float((T / mp.sqrt(4 * mp.pi * D)) * I)

    def C_direct(x, tau):
        """Direct 2D (t_v, x_v) quadrature of the MSR tree diagram:
        C = 2T ∫_{-∞}^{t2} dt_v ∫ dx_v G_R(t1-t_v, x-x_v) G_R(t2-t_v, -x_v)
        with t1 = tau, t2 = 0 (τ>0).  Substitute to s-integral-free form
        by integrating both vertex coords numerically."""
        t1, t2 = tau, 0.0
        def inner(tv):
            f = lambda xv: (G_heat(t1 - tv, x - xv, D, mu) *
                            G_heat(t2 - tv, 0.0 - xv, D, mu))
            v, _ = integrate.quad(f, -60, 60, limit=300, epsabs=1e-12)
            return v
        val, _ = integrate.quad(inner, -60, t2, limit=300, epsabs=1e-12)
        return 2 * T * val

    # τ=0 analytic target
    print('  -- equal-time (τ=0) vs analytic T/(2√(μD))·exp(-|x|/ξ) --')
    for x in [0.0, 0.5, 1.0, 2.0, 3.0]:
        analytic = T / (2 * math.sqrt(mu * D)) * math.exp(-abs(x) / xi)
        clf = C_closed(x, 0.0)
        rel = abs(clf - analytic) / max(abs(analytic), 1e-300)
        ok &= _verdict(f'C({x},0) closed vs analytic', rel, 1e-10)

    # τ>0: closed form vs direct 2D quadrature
    print('  -- two-time (τ>0): erf-split closed vs direct 2D quad --')
    for x, tau in [(0.0, 0.5), (1.0, 0.5), (1.0, 2.0), (2.0, 1.0), (0.5, 3.0)]:
        clf = C_closed(x, tau)
        ref = C_direct(x, tau)
        rel = abs(clf - ref) / max(abs(ref), 1e-300)
        ok &= _verdict(f'C({x},{tau}) closed vs 2D-quad', rel, 1e-6)
    return ok


if __name__ == '__main__':
    print('=' * 68)
    print('Phase 5 math spike — heat-kernel erf-split + semigroup collapse')
    print('=' * 68)
    results = {
        '1a real-α erf-split':   test_1a(),
        '1b complex-α erf-split': test_1b(),
        '1c semi-infinite limit': test_1c(),
        '2  semigroup collapse':  test_2(),
        '3  end-to-end tree C':   test_3(),
    }
    print('\n' + '=' * 68)
    print('SUMMARY')
    print('=' * 68)
    all_ok = True
    for name, passed in results.items():
        tag = PASS if passed else FAIL
        print(f'  [{tag}] {name}')
        all_ok &= passed
    print('\n' + ('GO — Rescue A premise + semigroup collapse confirmed.'
                  if all_ok else
                  'NO-GO — at least one claim failed; revisit before coding.'))
