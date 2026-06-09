"""
tests/test_phi_n.py
===================
Dyson step D-1: validate the ``Φ_n`` divided-difference evaluator
(``msrjd.integration.spatial.spectral_propagator.phi_n`` / ``phi_n_batch``),

    Φ_n(t; ν_1,…,ν_n) = ∫_{σ_n} tⁿ e^{−t Σ uᵢ νᵢ} d𝐮     (σ_n = {uᵢ ≥ 0, Σ uᵢ ≤ 1}),

evaluated via Hermite–Genocchi + the Opitz ``expm``-of-bidiagonal form.

  * Φ_0 = 1 and the Φ_1 closed form (1 − e^{−tν})/ν (real AND complex ν);
  * brute-force iterated-quad simplex oracle at n = 1, 2, 3, real and complex
    nodes separately (complex ⇒ real+imag parts integrated separately);
  * confluent limit (all nodes equal) vs the 1-d radial oracle
    tⁿ ∫_0^1 s^{n−1}/(n−1)! e^{−tνs} ds, plus NEAR-confluent stability
    (ν_i = ν + i·1e−13 must not 0/0-blow-up — the point of the Opitz form);
  * permutation symmetry of the nodes; Φ_n(0; ν) = 0 for n ≥ 1 (the tⁿ factor);
  * decay bound |Φ_n(t; ν)| ≤ tⁿ/n! for Re νᵢ ≥ 0 (|e^{−t·u·ν}| ≤ 1 on σ_n);
  * ``phi_n_batch`` matches ``phi_n`` elementwise and preserves shape.

Run:  sage -python -m pytest tests/test_phi_n.py -q
"""
from __future__ import annotations

import math
import os
import sys
from itertools import permutations

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from scipy.integrate import quad                               # noqa: E402

from msrjd.integration.spatial.spectral_propagator import (    # noqa: E402
    phi_n, phi_n_batch,
)

_QUAD_KW = dict(epsabs=1e-11, epsrel=1e-11, limit=200)


def _quad_c(f, a, b):
    """``scipy.integrate.quad`` for a complex-valued integrand (real + imag
    parts integrated separately)."""
    re = quad(lambda x: complex(f(x)).real, a, b, **_QUAD_KW)[0]
    im = quad(lambda x: complex(f(x)).imag, a, b, **_QUAD_KW)[0]
    return re + 1j * im


def _simplex_oracle(t, nus):
    """Brute-force iterated-quad evaluation of the DEFINING simplex integral
    ``∫_{σ_n} tⁿ e^{−t Σ uᵢ νᵢ} d𝐮`` (u_i ≥ 0, Σ u_i ≤ 1).  Slow but
    definitionally faithful; complex nodes ⇒ complex quadrature."""
    nus = [complex(nu) for nu in nus]
    n = len(nus)
    is_complex = any(nu.imag != 0.0 for nu in nus)

    def level(i, acc, budget):
        if i == n:
            return t ** n * np.exp(-t * acc)
        f = lambda u: level(i + 1, acc + u * nus[i], budget - u)  # noqa: E731
        if is_complex:
            return _quad_c(f, 0.0, budget)
        return quad(lambda u: f(u).real, 0.0, budget, **_QUAD_KW)[0]

    return level(0, 0.0 + 0.0j, 1.0)


def test_phi_0_is_one():
    """Empty node list ⇒ Φ_0 = 1 (scalar and batch)."""
    for t in (0.0, 0.3, 2.7):
        assert phi_n(t, []) == pytest.approx(1.0)
    out = phi_n_batch(np.array([0.0, 0.5, 1.0]), [])
    assert out.shape == (3,) and np.allclose(out, 1.0)


@pytest.mark.parametrize('t,nu', [
    (0.7, 1.3), (2.0, 0.5), (0.9, 4.2),                  # real ν
    (1.1, 2.0 + 1.5j), (0.4, 0.3 - 0.8j),                # complex ν
])
def test_phi_1_closed_form(t, nu):
    """Φ_1(t; ν) = (1 − e^{−tν})/ν = ∫_0^1 t e^{−tuν} du (the worked Opitz
    n=1 check in the phi_n docstring)."""
    expected = (1.0 - np.exp(-t * nu)) / nu
    assert phi_n(t, [nu]) == pytest.approx(expected, rel=1e-12)


@pytest.mark.parametrize('t,nus', [
    (0.8, [1.7]),
    (1.1, [0.6, 2.3]),
    (0.9, [0.4, 1.2, 2.7]),
])
def test_simplex_oracle_real(t, nus):
    """Brute-force iterated-quad oracle of the defining simplex integral,
    real nodes, n = 1, 2, 3."""
    assert phi_n(t, nus) == pytest.approx(_simplex_oracle(t, nus), rel=1e-8)


@pytest.mark.parametrize('t,nus', [
    (0.7, [1.0 + 2.0j]),
    (1.0, [0.8 + 1.5j, 0.8 - 1.5j]),                     # conjugate pair
    (0.6, [0.5 + 1.0j, 0.5 - 1.0j, 1.5]),                # pair + real node
])
def test_simplex_oracle_complex(t, nus):
    """Same oracle with COMPLEX nodes (real + imag parts integrated
    separately) — eigenvalue differences of a real M with complex spectrum
    come in conjugate pairs, so this path is load-bearing."""
    got = phi_n(t, nus)
    want = _simplex_oracle(t, nus)
    assert got.real == pytest.approx(want.real, rel=1e-8, abs=1e-10)
    assert got.imag == pytest.approx(want.imag, rel=1e-8, abs=1e-10)


@pytest.mark.parametrize('t,nu,n', [
    (0.8, 1.4, 2), (1.2, 0.7, 3), (0.9, 0.6 + 1.1j, 3),
])
def test_confluent_limit(t, nu, n):
    """All nodes equal ⇒ radial reduction
    Φ_n(t; ν,…,ν) = tⁿ ∫_0^1 s^{n−1}/(n−1)! e^{−tνs} ds (1-d quad oracle)."""
    radial = _quad_c(
        lambda s: s ** (n - 1) / math.factorial(n - 1) * np.exp(-t * nu * s),
        0.0, 1.0)
    want = t ** n * radial
    got = phi_n(t, [nu] * n)
    assert got.real == pytest.approx(want.real, rel=1e-10, abs=1e-12)
    assert got.imag == pytest.approx(want.imag, rel=1e-10, abs=1e-12)
    # ν = 0 (confluent at zero): Z nilpotent ⇒ Φ_n = tⁿ/n! exactly.
    assert phi_n(t, [0.0] * n) == pytest.approx(t ** n / math.factorial(n))


@pytest.mark.parametrize('nu', [1.3, 0.5 + 0.9j])
def test_near_confluent_stability(nu):
    """ν_i = ν + i·1e−13: a naive divided-difference table 0/0-blows-up on
    such close-paired nodes; the Opitz expm form must agree with the exactly
    confluent value to 1e-9."""
    t, n = 0.9, 3
    exact = phi_n(t, [nu] * n)
    perturbed = phi_n(t, [nu + i * 1e-13 for i in range(n)])
    assert abs(perturbed - exact) < 1e-9
    assert np.isfinite([perturbed.real, perturbed.imag]).all()


def test_permutation_symmetry():
    """Φ_n is symmetric in the nodes (the simplex integral is — divided
    differences are permutation-invariant)."""
    t, nus = 0.8, [0.4, 1.2 + 0.7j, 1.2 - 0.7j]
    vals = [phi_n(t, p) for p in permutations(nus)]
    assert all(abs(v - vals[0]) < 1e-12 for v in vals[1:])


def test_t_zero_vanishes_for_n_ge_1():
    """Φ_n(0; ν) = 0ⁿ · Vol(σ_n) = 0 for n ≥ 1 (the tⁿ factor); Φ_0(0) = 1."""
    assert phi_n(0.0, []) == pytest.approx(1.0)
    for nus in ([1.3], [0.6, 2.3], [0.4 + 1.0j, 0.4 - 1.0j, 1.5]):
        assert abs(phi_n(0.0, nus)) < 1e-15


@pytest.mark.parametrize('t', [0.3, 1.0, 4.0])
@pytest.mark.parametrize('nus', [
    [0.7], [0.0, 2.1], [1.5 + 2.0j, 1.5 - 2.0j, 0.3],
])
def test_decay_bound(t, nus):
    """|Φ_n(t; ν)| ≤ tⁿ/n! for Re νᵢ ≥ 0 (|e^{−t Σ uᵢνᵢ}| ≤ 1 on σ_n, and
    Vol(σ_n) = 1/n!)."""
    n = len(nus)
    assert abs(phi_n(t, nus)) <= t ** n / math.factorial(n) + 1e-12


def test_batch_matches_scalar():
    """phi_n_batch == elementwise phi_n; shape preserved (incl. 2-d ts)."""
    nus = [0.5 + 1.0j, 0.5 - 1.0j]
    ts = np.array([[0.0, 0.4], [1.1, 3.0]])
    out = phi_n_batch(ts, nus)
    assert out.shape == ts.shape and out.dtype == complex
    for idx in np.ndindex(ts.shape):
        assert abs(out[idx] - phi_n(ts[idx], nus)) < 1e-14
