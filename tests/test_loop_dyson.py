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
    """Closed-form time-route δC(q,0) == frequency-route (≈2% numerics)."""
    a = bubble_delta_S(q, MU, D, T)
    b = _dS_frequency(q)
    assert abs(a - b) <= 4e-2 * max(abs(a), 1e-12)


def test_delta_S_positive_and_decaying():
    vals = [bubble_delta_S(q, MU, D, T) for q in (0.0, 0.5, 1.0, 2.0)]
    assert all(v > 0 for v in vals)
    assert vals[0] > vals[1] > vals[2] > vals[3]    # monotone decay in q


def test_delta_phi2_finite_positive():
    d = bubble_delta_phi2(MU, D, T)
    assert math.isfinite(d) and d > 0


def test_sigma_kernels_match_direct():
    """Sanity: the module's self-energy kernels equal a fresh direct ∫dℓ."""
    for q in (0.0, 0.9):
        for t in (0.3, 1.0):
            fR = lambda l: math.exp(-_mq(l) * t) / 1.0 * (T / _mq(q - l)) * math.exp(-_mq(q - l) * t)
            rR, _ = integrate.quad(fR, -np.inf, np.inf, limit=120)
            assert abs(sigma_R_time(q, t, MU, D, T) - rR / (2 * math.pi)) <= 1e-9
