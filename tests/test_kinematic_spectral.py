"""
tests/test_kinematic_spectral.py
================================
Dyson 3c-1: the per-segment-mass spectral kinematic
(``full_integrator.diagram_kinematic_spectral`` + ``spectral_rows``).

  * TREE C edge closed form (exact oracle):
    I(q,τ) = e^{−(m_v+Dq²)τ} / (m_u+m_v+2Dq²)  — incl. COMPLEX masses;
  * uniform-mass limit: I_orig = 2^{n_C}·I_spec on a hand-built bubble AND
    tadpole, q-path and xs-path (the two-segment C representation halves each
    σ integral; the single-field path folds that 2 into its 2^{−n_C});
  * xs-path tree vs direct σ-quadrature oracle;
  * mass-table batching: many assignments in one call == one-at-a-time.

Run:  sage -python -m pytest tests/test_kinematic_spectral.py -q
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from scipy.integrate import quad                                  # noqa: E402

from msrjd.integration.spatial.diagram_descriptor import (        # noqa: E402
    CEdge, CStackDiagram,
)
from msrjd.integration.spatial.full_integrator import (           # noqa: E402
    diagram_kinematic, diagram_kinematic_spectral, spectral_rows,
)


def _tree():
    """Single C edge between the two external legs (t=0 at leg 0, t=τ at leg 1)."""
    return CStackDiagram(
        internal_vertices=(),
        external_legs=(0, 1),
        edges=(CEdge(a=(), b=(1.0,), kind='C', u=0, v=1, external=True),),
        n_loops=0)


def _bubble():
    """Hand-built 1-loop bubble: legs 0 (t=0), 1 (t=τ); internal 2, 3.
    Momentum: q in at leg 0 → loop ℓ on the C edge, q−ℓ on the internal R."""
    return CStackDiagram(
        internal_vertices=(2, 3),
        external_legs=(0, 1),
        edges=(
            CEdge(a=(0.0,), b=(1.0,), kind='R', u=0, v=2, external=True),
            CEdge(a=(0.0,), b=(1.0,), kind='R', u=3, v=1, external=True),
            CEdge(a=(1.0,), b=(0.0,), kind='C', u=2, v=3, external=False),
            CEdge(a=(-1.0,), b=(1.0,), kind='R', u=2, v=3, external=False),
        ),
        n_loops=1)


def _tadpole():
    """C self-loop on one internal vertex between the legs."""
    return CStackDiagram(
        internal_vertices=(2,),
        external_legs=(0, 1),
        edges=(
            CEdge(a=(0.0,), b=(1.0,), kind='R', u=0, v=2, external=True),
            CEdge(a=(0.0,), b=(1.0,), kind='R', u=2, v=1, external=True),
            CEdge(a=(1.0,), b=(0.0,), kind='C', u=2, v=2, external=False),
        ),
        n_loops=1)


def test_spectral_rows_expansion():
    rows = spectral_rows(_bubble())
    assert [h for _, _, h in rows] == ['R', 'R', 'Cu', 'Cv', 'R']
    assert len(spectral_rows(_tree())) == 2                       # Cu + Cv


@pytest.mark.parametrize('m_u,m_v', [
    (1.0, 1.0), (0.8, 2.3),
    (1.0 + 0.7j, 1.0 - 0.7j),                                     # conjugate pair
])
def test_tree_closed_form_q(m_u, m_v):
    """Tree C edge: I(q,τ) = e^{−(m_v+Dq²)τ}/(m_u+m_v+2Dq²) exactly."""
    D, q, tau = 0.6, 0.9, 0.7
    et = {0: 0.0, 1: tau}
    table = np.array([m_u, m_v], dtype=complex)
    val = diagram_kinematic_spectral(_tree(), [q], et, table, D, n_s=40)[0]
    mq = D * q * q
    expected = np.exp(-(m_v + mq) * tau) / (m_u + m_v + 2 * mq)
    assert abs(val - expected) < 1e-7 * abs(expected)


def test_tree_xs_vs_quadrature():
    """Tree xs-path vs direct σ-quadrature of the two-segment representation."""
    D, tau = 0.6, 0.5
    m_u, m_v = 0.9, 1.7
    xs = np.array([0.0, 0.8, 1.6])
    et = {0: 0.0, 1: tau}
    val = diagram_kinematic_spectral(_tree(), [0.0], et,
                                     np.array([m_u, m_v]), D, n_s=40, xs=xs)[0]

    def oracle(x):
        def f(s):
            B = D * (tau + 2 * s)
            hk = (4 * np.pi * B) ** -0.5 * np.exp(-x * x / (4 * B))
            return np.exp(-m_u * s - m_v * (tau + s)) * hk
        return quad(f, 0, 60.0, limit=400)[0]

    for i, x in enumerate(xs):
        assert abs(val[i].real - oracle(x)) < 2e-7
        assert abs(val[i].imag) < 1e-12


@pytest.mark.parametrize('mk', ['bubble', 'tadpole'])
@pytest.mark.parametrize('path', ['q', 'xs'])
def test_uniform_limit_matches_single_field(mk, path):
    """Uniform masses: diagram_kinematic == 2^{n_C} · diagram_kinematic_spectral
    (same descriptor, same grids; only the C-edge σ representation differs)."""
    descr = _bubble() if mk == 'bubble' else _tadpole()
    mu, D, tau = 1.3, 0.7, 0.6
    et = {0: 0.0, 1: tau}
    n_C = sum(1 for e in descr.edges if e.kind == 'C')
    table = np.full(len(spectral_rows(descr)), mu, dtype=complex)
    kw = dict(n_t=26, n_s=40)
    if path == 'q':
        ref = diagram_kinematic(descr, [0.8], et, mu, D, **kw)
        new = diagram_kinematic_spectral(descr, [0.8], et, table, D, **kw)[0]
        assert new.imag == pytest.approx(0.0, abs=1e-14)
        assert (2.0 ** n_C) * new.real == pytest.approx(ref, rel=2e-5)
    else:
        xs = np.array([0.0, 1.0])
        ref = diagram_kinematic(descr, [0.0], et, mu, D, xs=xs, **kw)
        new = diagram_kinematic_spectral(descr, [0.0], et, table, D, xs=xs, **kw)[0]
        assert np.allclose((2.0 ** n_C) * new.real, ref, rtol=2e-5)
        assert np.allclose(new.imag, 0.0, atol=1e-14)


def test_mass_table_batching():
    """A (n_rows, n_assign) batch returns exactly the per-column results."""
    descr = _bubble()
    et = {0: 0.0, 1: 0.5}
    rows = spectral_rows(descr)
    rng = np.random.default_rng(7)
    cols = [1.0 + 0.5 * rng.random(len(rows)) + 0.2j * rng.standard_normal(len(rows))
            for _ in range(4)]
    table = np.stack(cols, axis=1)
    ms = float(np.min(table.real))                     # pin one shared grid scale
    batch = diagram_kinematic_spectral(descr, [0.7], et, table, 0.6, mu_scale=ms)
    for j, c in enumerate(cols):
        single = diagram_kinematic_spectral(descr, [0.7], et, c, 0.6,
                                            mu_scale=ms)[0]
        assert abs(batch[j] - single) < 1e-12 * max(1.0, abs(single))


def test_rejects_nonpositive_mass():
    with pytest.raises(ValueError, match='Re m > 0'):
        diagram_kinematic_spectral(_tree(), [0.0], {0: 0.0, 1: 0.5},
                                   np.array([1.0, -0.2]), 0.5)
