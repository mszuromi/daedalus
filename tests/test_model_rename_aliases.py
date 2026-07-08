"""Regression: the July-2026 theory -> model rename keeps deprecated aliases.

Old user code (``dd.load_theory``, ``dd.TheoryBuilder``, ``from api.theory
import ...``) must keep working while the new Model* names are canonical.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import daedalus as dd                                      # noqa: E402


def test_load_model_and_alias():
    m1, mod1 = dd.load_model('ou_quartic')
    m2, mod2 = dd.load_theory('ou_quartic')               # deprecated alias
    assert dd.load_theory is dd.load_model
    assert m1['name'] == m2['name']


def test_list_models_alias_and_registry():
    names = dd.list_models()
    assert 'ou_quartic' in names and 'linear_hawkes' in names
    assert dd.list_theories is dd.list_models


def test_builder_aliases():
    from api.model import ModelBuilder, TheoryBuilder
    assert TheoryBuilder is ModelBuilder
    assert dd.TheoryBuilder is dd.ModelBuilder


def test_api_theory_shim_module():
    import api.theory as shim
    from api.model import ModelBuilder
    assert shim.ModelBuilder is ModelBuilder
    assert shim.TheoryBuilder is ModelBuilder


def test_ui_alias():
    from api.ui import ModelUI, TheoryUI
    assert TheoryUI is ModelUI


def test_model_files_layout():
    root = os.path.join(os.path.dirname(__file__), '..')
    assert os.path.isdir(os.path.join(root, 'models'))
    assert os.path.exists(os.path.join(root, 'models', 'ou_quartic.model.py'))
    assert not os.path.isdir(os.path.join(root, 'theories'))
