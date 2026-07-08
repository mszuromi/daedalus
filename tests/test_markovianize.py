"""
Unit tests for the colored-noise → Markovian-embedding preprocessor
(``api/colored_to_markovian.py``).

Coverage:
  * ``detect_lorentzian`` matches the v1 single-Lorentzian template
    and rejects everything else.
  * Building ``ou_quartic_colored.model.py`` (the worked example)
    through the default-ON preprocessor produces the expected
    auxiliary-field + white-noise spec.
  * Tree-level free-theory ``C(τ=0)`` matches the analytic OU result
    ``2D / (μ (μτc + 1))``.  We use parameters that avoid the
    ``μ = 1/τc`` exact-degeneracy edge case (see v2 follow-up).
  * Builder-level ``.markovianize(False)`` opt-out keeps the legacy
    colored row in the spec.
  * The ``markovianize=`` round-trip through ``model_serialize``
    (save → load) preserves both per-row and per-builder overrides.

Run with:
    sage -python -m pytest tests/test_markovianize.py -v
"""
from __future__ import annotations

import os
import shutil
import tempfile
import time

import pytest


# Where ``ou_quartic_colored.model.py`` lives.
HERE          = os.path.dirname(os.path.abspath(__file__))
COLORED_FILE  = os.path.join(HERE, '..', 'models',
                             'ou_quartic_colored.model.py')


def _build_colored() -> dict:
    """Build the user's worked-example colored-noise model using
    its on-disk file (so we exercise the load → builder → markovianize
    chain end-to-end)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location('th', COLORED_FILE)
    th = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(th)
    return th.build()


@pytest.fixture
def isolated_cache(monkeypatch, tmp_path):
    """Run with an isolated cache root so the markovianized model's
    rebuilt cache doesn't interfere with other tests."""
    monkeypatch.chdir(tmp_path)
    yield str(tmp_path)


# ── (1) Pattern matcher ───────────────────────────────────────────────


def test_lorentzian_kernel_detection():
    """``detect_lorentzian`` must match the canonical
    ``c·exp(-|tau|/tauc)`` shape (including parameter name variants)
    and reject confusing non-matches.  Returns the τc expression and
    aux-drive coefficient on a match."""
    from api.colored_to_markovian import detect_lorentzian

    # Standard form, as in the user's worked example.
    info = detect_lorentzian('exp(-abs(tau)/tauc)', '2*D/tauc',
                             declared_params=['D', 'tauc', 'mu', 'eps'])
    assert info is not None
    assert info['tauc_text']      == 'tauc'
    # Aux drive = 2 c / τc = 2 · (2D/τc) / τc = 4D/τc²
    assert info['aux_drive_text'] == '4*D/tauc^2'

    # Underscored parameter name should also work.
    info_u = detect_lorentzian('exp(-abs(tau)/tau_c)', '2*D/tau_c',
                               declared_params=['D', 'tau_c'])
    assert info_u is not None
    assert info_u['tauc_text']    == 'tau_c'

    # Cross-correlation amplitude — the SAME kernel shape, different
    # coefficient.
    info_x = detect_lorentzian('exp(-abs(tau)/tauc)',
                               'rho*sqrt(D1*D2)/tauc',
                               declared_params=['D1', 'D2', 'rho', 'tauc'])
    assert info_x is not None
    assert info_x['tauc_text']    == 'tauc'

    # Gaussian: must NOT match.
    assert detect_lorentzian('exp(-tau^2)', '2*D',
                             declared_params=['D']) is None

    # Empty kernel (whitenoise placeholder): must NOT match.
    assert detect_lorentzian('', '1.0',
                             declared_params=['D']) is None

    # Bare exponential without absolute value: must NOT match (it
    # would correspond to a non-physical causal-only kernel,
    # asymmetric in τ).
    assert detect_lorentzian('exp(-tau/tauc)', '2*D/tauc',
                             declared_params=['D', 'tauc']) is None


# ── (2) Builder round-trip ────────────────────────────────────────────


def test_ou_quartic_colored_roundtrip(isolated_cache):
    """Building the user's worked example through the default-ON
    markovianize path must produce the expected augmented spec:
    one extra physical field, one extra equation, white-noise CGF
    row, and an action text containing both ``xt`` and ``xit``."""
    model = _build_colored()

    # Two physical fluctuation fields: x (user's) and xi (auxiliary).
    phys_names = sorted(f['name'] for f in model['physical_fields'])
    assert phys_names == ['dx', 'dxi']

    # Two response fields, matching the autoresponse rule.
    resp_names = sorted(f['name'] for f in model['response_fields'])
    assert resp_names == ['xit', 'xt']

    # Two DAE equations: x's, plus xi's drift.
    eq_count = len(model.get('equations', []))
    assert eq_count == 2

    # Exactly one correlated-noise entry, now white.
    cn = model.get('correlated_noises', {})
    assert len(cn) == 1
    [(name, spec)] = list(cn.items())
    assert name.endswith('_markov_aux')
    # The single κ² entry's response legs are on the auxiliary field.
    assert spec['response_legs'][2] == ['xit', 'xit']


def test_markovianize_opt_out_keeps_colored_row():
    """Builder-level ``.markovianize(False)`` must leave the original
    colored CGF row untouched.  Required for users whose kernels don't
    match the v1 template, or who've hand-coded their own embedding."""
    from api.model import ModelBuilder

    builder = (
        ModelBuilder('Test colored opt-out', n_populations=0)
        .physical_field('x', indexed=True)
        .parameter('mu',   default=0.1, domain='real')
        .parameter('D',    default=1.0, domain='positive')
        .parameter('tauc', default=1.0, domain='positive')
        .markovianize(False)
        .declare_cgf_term('C', response_legs=['xt', 'xt'], order=2,
                          coefficient='2*D/tauc',
                          kernel='exp(-abs(tau)/tauc)')
        .set_action_text('xt*((Dt+mu)*x)')
        .equation(lhs='(Dt+mu)*x', rhs='0')
    )
    model = builder.build()
    cn = model.get('correlated_noises', {})
    assert list(cn.keys()) == ['C']
    # Only one physical field — no auxiliary was injected.
    phys_names = [f['name'] for f in model['physical_fields']]
    assert phys_names == ['dx']


# ── (3) Tree-level analytic agreement ─────────────────────────────────


# Parameters chosen to AVOID the ``μ = 1/τc`` exact-degeneracy edge
# case (which produces a multiplicity-2 pole the current single-pole
# residue infrastructure can't handle — see the v2 follow-up).  At
# ``μ = 0.1, τc = 1, D = 1`` the analytic prediction is
#   C(0) = 2D / (μ (μτc + 1)) = 2 / (0.1 · 1.1) ≈ 18.182
def test_colored_ou_tree_level_matches_analytic(isolated_cache):
    """``compute_cumulants`` on the Markovianized OU at k=2, max_ell=0
    must agree with the analytic free-theory variance
    ``C(0) = 2D / (μ (μτc + 1))``."""
    from api import compute_cumulants

    model = _build_colored()
    mu, D, tauc = 0.1, 1.0, 1.0
    res = compute_cumulants(
        model,
        fundamental={'mu': mu, 'eps': 0.0, 'D': D, 'tauc': tauc},
        k=2, max_ell=0,
        external_fields=[('x', 1), ('x', 1)],
        taylor_order=2,
        tau_max=2.0,
        tau_step=1.0,
        verbose=False,
    )
    c_tau = res['C_tau']
    c0 = c_tau[len(c_tau) // 2]
    analytic = 2 * D / (mu * (mu * tauc + 1))
    assert abs(complex(c0).real - analytic) < 1e-6, (
        f'C(0)={complex(c0).real} vs analytic={analytic}')
    # Symmetric: C(τ) = C(-τ).
    assert abs(c_tau[0] - c_tau[-1]) < 1e-9


# ── (4) Serialization round-trip of the markovianize keyword ──────────


def test_markovianize_serialize_round_trip():
    """``model_serialize`` must preserve the explicit per-row
    ``markovianize=`` keyword AND the builder-level
    ``.markovianize(False)`` opt-out across save → load."""
    from api.model_serialize import (
        render_model_file, load_spec_from_file, save_model_to_file,
    )

    spec = {
        'name': 'roundtrip_markov',
        'description': '',
        'populations': [],
        'n_populations': 1,
        'response_fields': [],
        'physical_fields': [
            {'name': 'x', 'indexed': True, 'natural_name': 'x'}],
        'parameters': [
            {'name': 'tauc', 'default': 1.0, 'domain': 'positive'},
            {'name': 'D',    'default': 1.0, 'domain': 'positive'},
            {'name': 'mu',   'default': 0.1, 'domain': 'real'},
        ],
        'functions': [],
        'kernels': [],
        'cgf_terms': [
            {'name':         'CXX',
             'response_legs': ['xt', 'xt'],
             'order':         2,
             'coefficient':   '2*D/tauc',
             'kernel':        'exp(-abs(tau)/tauc)',
             'markovianize':  False},
        ],
        'action_text': 'xt*((Dt+mu)*x)',
        'equations':   [{'lhs': '(Dt+mu)*x', 'rhs': '0'}],
        'mf_equations': [],
        'stability_analysis': False,
        'markovianize_default': False,
        'default_fundamental': {},
        'metadata': {},
    }
    src = render_model_file(spec)
    # Builder-level opt-out emits a ``.markovianize(False)`` call.
    assert '.markovianize(False)' in src
    # Per-row keyword emits ``markovianize=False`` inside the
    # ``.declare_cgf_term(...)`` call.
    assert 'markovianize=False' in src

    # Round-trip on disk.
    with tempfile.TemporaryDirectory() as tmp:
        path = save_model_to_file(spec, tmp)
        loaded = load_spec_from_file(path)
        assert loaded['markovianize_default'] is False
        assert loaded['cgf_terms'][0]['markovianize'] is False
