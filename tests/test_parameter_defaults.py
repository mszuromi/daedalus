"""``compute_cumulants`` parameter resolution.

This path accreted three fixes found by accident rather than by test --
a bare NameError when ``parameters=`` was omitted, no naming of which
parameter was missing, and an empty dict slipping past the fallback -- so it
gets its own coverage.
"""

import importlib.util
import numpy as np
import pytest

from api.compute import compute_cumulants


def _load(name):
    spec = importlib.util.spec_from_file_location('m', f'models/{name}.model.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _tree_C0(model, **kw):
    res = compute_cumulants(model, k=2, max_ell=0,
                            external_fields=[('dx', 1)] * 2,
                            tau_grid=np.array([0.0]), use_cache=True,
                            verbose=False, **kw)
    return complex(res['total_C_by_ell'][0](0.0, 0.0)).real


def test_omitted_parameters_use_declared_defaults():
    """mu=D=1 by declaration, so the tree cumulant is D/mu = 1."""
    model = _load('ou_quartic').build()
    assert _tree_C0(model) == pytest.approx(1.0, rel=1e-5)


def test_empty_parameters_are_treated_as_omitted():
    """Several models ship DEFAULT_FUNDAMENTAL = {}; forwarding it verbatim
    must not sail past the fallback and fail inside the mean-field solve."""
    model = _load('ou_quartic').build()
    assert _tree_C0(model, parameters={}) == pytest.approx(1.0, rel=1e-5)


def test_explicit_parameters_still_win():
    model = _load('ou_quartic').build()
    got = _tree_C0(model, parameters={'mu': 2.0, 'eps': 0.02, 'D': 1.0})
    assert got == pytest.approx(0.5, rel=1e-5), 'D/mu with mu=2'


def _model_without_defaults():
    """A minimal model whose parameters declare no defaults.

    Built inline rather than loaded from ``models/``: the only shipped model
    with no defaults at all is gitignored (kept local), so loading it made
    this test pass in a working tree and fail in a fresh clone.
    """
    from api.model import TemporalModelBuilder
    return (
        TemporalModelBuilder('no-defaults probe')
        .population('pop', size=1)
        .physical_field('x', population='pop', description='variable')
        .parameter('E', domain='positive')
        .parameter('w', domain='positive')
        .set_action_text('sum(xt[i]*((Dt+E)*x[i]) - w*xt[i]^2 for i in pop)')
        .equation(lhs='(Dt+E)*x[i]', rhs='0', population='pop')
        .build()
    )


def test_missing_required_parameters_are_named():
    """A model with no defaults must say WHICH parameters it needs, not raise
    a bare NameError from deep inside the Sage substitution."""
    model = _model_without_defaults()
    with pytest.raises(ValueError) as exc:
        compute_cumulants(model, k=2, max_ell=0,
                          external_fields=[('dx', 1)] * 2,
                          tau_grid=np.array([0.0]), use_cache=False,
                          verbose=False)
    msg = str(exc.value)
    assert 'E' in msg and 'w' in msg, f'missing names not reported: {msg}'
    assert 'parameters=' in msg, 'the error should say how to fix it'


def test_mean_field_parameters_are_not_required():
    """xstar/nstar/vstar are SOLVED, not supplied, so they must never appear
    in the missing-parameter list."""
    model = _load('ou_quartic').build()
    names = [p['name'] for p in model.get('parameters', [])
             if p.get('mean_field')]
    assert names, 'expected at least one mean-field parameter to exist'
    _tree_C0(model)          # must not raise despite those having no default
