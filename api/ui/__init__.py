"""
pipeline.ui — ipywidgets-based model input UI.

This package powers ``notebooks/model_builder.ipynb``.  Run that
notebook, fill out the form, click Save, and a ``.model.py`` file
appears in ``models/`` ready to be loaded by
``notebooks/model_runner.ipynb``.
"""
from api.ui.main import ModelUI

TheoryUI = ModelUI       # deprecated pre-July-2026 alias
__all__ = ['ModelUI', 'TheoryUI']
