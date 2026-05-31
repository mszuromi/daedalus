"""
tests/test_loop_dyson.py
========================
The 1-loop Dyson assembly for the spatial bubble
(``msrjd.integration.spatial.loop_dyson``).

Pins that the closed-form equal-time ``δC(q,0)`` matches an independent
frequency-route Dyson computation (representation independence → the assembly
is coded correctly), and the basic physical shape (positive, q-decaying,
finite ``δ⟨φ²⟩``).

Run:  sage -python -m pytest tests/test_loop_dyson.py -q
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np
import pytest
from scipy import integrate

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from msrjd.integration.spatial.loop_dyson import (
    bubble_delta_S, bubble_delta_phi2, sigma_R_time, sigma_K_time,
    _dyson_terms, C_R, C_K, bubble_delta_C_q_tau,
)

MU = D = T = 1.0


def _mq(q):
    return MU + D * q * q


def _dS_frequency(q):
    """Independent frequency-route δC(q,0): build Σ(q,ω) by FT of the time
    kernels, assemble the Dyson product, integrate over ω."""
    tg = np.linspace(1e-3, 30.0, 500)
    SRt = np.array([sigma_R_time(q, t, MU, D, T) for t in tg])
    SKt = np.array([sigma_K_time(q, t, MU, D, T) for t in tg])
    wg = np.linspace(-50.0, 50.0, 1500)
    SRw = np.array([np.trapz(SRt * np.exp(1j * w * tg), tg) for w in wg])
    SKw = np.array([np.trapz(2 * SKt * np.cos(w * tg), tg) for w in wg])
    m = _mq(q)
    GR = 1.0 / (m - 1j * wg)
    GA = 1.0 / (m + 1j * wg)
    C = 2 * T / (m * m + wg * wg)
    integ = GR * SRw * C + GR * SKw * GA + C * np.conj(SRw) * GA
    return float(np.trapz(integ.real, wg) / (2 * math.pi))


@pytest.mark.parametrize('q', [0.0, 0.6, 1.2, 2.0])
def test_equal_time_dyson_freq_vs_time(q):
    """Closed-form time-route Dyson sum T1+T2 == frequency-route (≈2% num.).
    Validates the Dyson STRUCTURE (the physical weights C_R,C_K are separate)."""
    t1, t2 = _dyson_terms(q, MU, D, T)
    a = t1 + t2
    b = _dS_frequency(q)
    assert abs(a - b) <= 4e-2 * max(abs(a), 1e-12)


def test_delta_S_positive_and_decaying():
    vals = [bubble_delta_S(q, MU, D, T, g=1.0) for q in (0.0, 0.5, 1.0, 2.0)]
    assert all(v > 0 for v in vals)
    assert vals[0] > vals[1] > vals[2] > vals[3]    # monotone decay in q


def test_physical_weights_pinned():
    """bubble_delta_S applies the framework-pinned weights g²(C_R·T1 + C_K·T2)
    with C_R=4, C_K=2 (from the M(Γ)=16,8 uniform-momentum diagram values)."""
    assert (C_R, C_K) == (4.0, 2.0)
    q, g = 0.7, 1.3
    t1, t2 = _dyson_terms(q, MU, D, T)
    assert abs(bubble_delta_S(q, MU, D, T, g) - g * g * (4 * t1 + 2 * t2)) <= 1e-12


def test_delta_phi2_finite_positive():
    d = bubble_delta_phi2(MU, D, T, g=0.3)
    assert math.isfinite(d) and d > 0


@pytest.mark.parametrize('q', [0.0, 0.6, 1.2, 2.0])
def test_tau_dependent_reduces_to_closed_form(q):
    """The full time-displaced bubble δC(q,τ) (time route) reduces to the
    closed-form structure factor ``bubble_delta_S(q)`` at τ=0 (<1%), and decays
    monotonically in τ to ~0."""
    g = 0.3
    taus = np.array([0.0, 0.5, 1.0, 2.0, 4.0])
    dC = bubble_delta_C_q_tau(q, taus, MU, D, T, g)
    closed = bubble_delta_S(q, MU, D, T, g)
    assert abs(dC[0] - closed) <= 1e-2 * closed          # τ=0 matches exact
    assert all(dC[i] >= dC[i + 1] - 1e-12 for i in range(len(dC) - 1))
    assert dC[-1] >= -1e-9 and dC[-1] < 0.2 * dC[0]      # decays toward 0


def test_sigma_kernels_match_direct():
    """Sanity: the module's self-energy kernels equal a fresh direct ∫dℓ."""
    for q in (0.0, 0.9):
        for t in (0.3, 1.0):
            fR = lambda l: math.exp(-_mq(l) * t) / 1.0 * (T / _mq(q - l)) * math.exp(-_mq(q - l) * t)
            rR, _ = integrate.quad(fR, -np.inf, np.inf, limit=120)
            assert abs(sigma_R_time(q, t, MU, D, T) - rR / (2 * math.pi)) <= 1e-9


def test_sigma_formfactor(q=0.9, t=0.7):
    """Phase 4: a vertex form factor F(ℓ) multiplies the loop integrand.
    F=1 reproduces the plain bubble (regression); a derivative form factor
    F(ℓ)=−ℓ² (a ∇² on the loop leg) is applied and matches an independent
    direct ∫dℓ with that factor."""
    base = sigma_R_time(q, t, MU, D, T)
    assert abs(sigma_R_time(q, t, MU, D, T, formfactor=lambda l: 1.0) - base) <= 1e-12 * abs(base) + 1e-15
    ff = lambda l: -l ** 2
    ref = integrate.quad(
        lambda l: ff(l) * math.exp(-_mq(l) * t) * (T / _mq(q - l)) * math.exp(-_mq(q - l) * t),
        -np.inf, np.inf, limit=120)[0] / (2 * math.pi)
    got = sigma_R_time(q, t, MU, D, T, formfactor=ff)
    assert abs(got - ref) <= 1e-9 and abs(got - base) > 1e-6     # applied, and changes the result


def test_bubble_delta_S_formfactor(q=0.7):
    """The form factor threads through the full equal-time bubble assembly:
    F=1 reproduces the plain bubble (regression); a derivative F(ℓ)=−ℓ² changes
    it (and stays finite)."""
    base = bubble_delta_S(q, MU, D, T, g=1.0)
    same = bubble_delta_S(q, MU, D, T, g=1.0, formfactor=lambda l: 1.0)
    assert abs(same - base) <= 1e-9 * abs(base) + 1e-15
    diff = bubble_delta_S(q, MU, D, T, g=1.0, formfactor=lambda l: -l ** 2)
    assert math.isfinite(diff) and abs(diff - base) > 1e-6


def test_bubble_delta_C_q_tau_formfactor(q=0.7):
    """The (vectorized) τ-dependent bubble path also threads the form factor:
    F=1 reproduces the plain time-displaced correlator; F(ℓ)=−ℓ² changes it and
    stays finite over τ."""
    taus = np.array([0.0, 1.0, 2.0])
    base = bubble_delta_C_q_tau(q, taus, MU, D, T, g=1.0)
    same = bubble_delta_C_q_tau(q, taus, MU, D, T, g=1.0,
                                formfactor=lambda l: np.ones_like(l))
    assert np.max(np.abs(same - base)) <= 1e-9 * (np.max(np.abs(base)) + 1e-30)
    diff = bubble_delta_C_q_tau(q, taus, MU, D, T, g=1.0,
                                formfactor=lambda l: -l ** 2)
    assert np.all(np.isfinite(diff)) and abs(diff[0] - base[0]) > 1e-6


@pytest.mark.parametrize('q', [0.3, 0.7, 1.2, 2.0, 3.0])
def test_derivative_vertex_formfactor_matches_direct_quad(q):
    """The conserved ∇²(φ²) derivative-vertex bubble: the tabulated τ-dependent
    assembly with the route-extracted SEPARATE form factors (Σ_R: F_R=q²ℓ²,
    Σ_K: F_K=q⁴) reproduces an INDEPENDENT direct ``scipy.quad`` of σ_R/σ_K (the
    sim-validated spike's ``dC_deriv``) at τ=0.

    Guards the integrable-singularity handling: under F_R=q²ℓ² the retarded
    self-energy is UV-sensitive, σ_R(a)~A·a^{-1/2} as a→0⁺.  A naive uniform-grid
    trapezoid with a σ(1e-7) ``a=0`` prepend OVER-counts that first sliver ~100×;
    the power-law sliver + 1/m-adaptive grid fix it to ≲2%.  (σ_K with F_K=q⁴ is
    ℓ-flat ⇒ finite at 0, so only Σ_R needed the fix.)"""
    m = _mq(q)
    fR = lambda l: q * q * l * l
    fK = lambda l: q ** 4
    t1 = integrate.quad(lambda u: sigma_R_time(q, u, MU, D, T, formfactor=fR)
                        * math.exp(-m * u), 0, np.inf, limit=200)[0] * T / (m * m)
    t2 = integrate.quad(lambda u: sigma_K_time(q, u, MU, D, T, formfactor=fK)
                        * math.exp(-m * u), 0, np.inf, limit=200)[0] / m
    ref = C_R * t1 + C_K * t2                       # g=1 ⇒ the spike's dC_deriv
    got = bubble_delta_C_q_tau(
        q, [0.0], MU, D, T, g=1.0,
        formfactor=lambda l: q * q * l * l * np.ones_like(l),
        formfactor_K=lambda l: q ** 4 * np.ones_like(l))[0]
    assert abs(got - ref) <= 3e-2 * abs(ref)        # ≲2% (mid-band ≲0.5%); the
    #                                                 bug it guards was ~10–37×
