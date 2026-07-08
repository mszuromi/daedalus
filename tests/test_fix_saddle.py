"""Regression tests for the env-safe mean-field saddle fix.

History: a first attempt widened the seed box to the symmetric [-3s, 3s] by
DEFAULT for any undeclared-domain field (gated on stability_analysis).  That
changed the sampling of linear_hawkes's voltage field and made its multi-start
saddle solve env-sensitive — test_linear_hawkes_runs passed locally but FAILED
in CI on a spurious root.  It was reverted.

The env-safe design (this fix):
- the seed box is driven by the mean-field SADDLE PARAMETER's domain;
- 'positive' OR undeclared -> POSITIVE box [0, 5s] (preserves EVERY existing
  model's historical sampling, byte for byte);
- a field opts a double-well into the symmetric box by declaring
  physical_field(domain='real') on the field (flows to the auto-saddle).
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np                                          # noqa: E402
import daedalus as dd                                       # noqa: E402


def _saddle_domains(model):
    return {p.get('natural_name'): p.get('domain')
            for p in model.get('parameters', []) if p.get('mean_field')}


def test_double_well_field_opts_into_real_domain():
    """physical_field(domain='real') flows to the auto-generated saddle param."""
    model, _ = dd.load_model('ou_quartic_double_well')
    assert _saddle_domains(model).get('x') == 'real'


def test_double_well_finds_negative_well_at_mu_negative():
    """With the symmetric box, the solver reaches the stable -sqrt(-mu/eps) well
    (the unstable x=0 is rejected by stability_analysis), instead of being
    trapped on the positive half-line."""
    model, mod = dd.load_model('ou_quartic_double_well')
    cfg = dd.Config(k=2, max_ell=0,
                    parameters={'mu': -1.0, 'eps': 1.0, 'D': 0.1},
                    tau_grid=(-4.0, 4.0, 9))
    res = dd.run(model, cfg, mod)
    xstar = np.atleast_1d(res['mf_values']['xstar']).astype(float)
    assert np.all(np.abs(xstar) > 0.5), f'expected a non-zero well, got {xstar}'


def test_double_well_default_mu_positive_unchanged():
    """At the default mu>0 the only real root is x=0; the symmetric box still
    finds it, so the default behaviour is unchanged."""
    model, mod = dd.load_model('ou_quartic_double_well')
    cfg = dd.Config(k=2, max_ell=0, tau_grid=(-4.0, 4.0, 9))   # default mu=0.1>0
    res = dd.run(model, cfg, mod)
    xstar = np.atleast_1d(res['mf_values']['xstar']).astype(float)
    assert np.allclose(xstar, 0.0, atol=1e-6), f'expected x*=0, got {xstar}'


def test_rate_and_voltage_fields_keep_positive_box():
    """Env-safety guard: linear_hawkes's rate field 'n' is 'positive' and its
    voltage 'v' is undeclared — NEITHER may be 'real'/'symmetric', so both keep
    the positive seed box (the reverted attempt sampled 'v' negative -> the
    CI-only spurious-root regression)."""
    model, _ = dd.load_model('linear_hawkes')
    doms = _saddle_domains(model)
    assert doms.get('n') == 'positive'
    assert doms.get('v') not in ('real', 'symmetric')
