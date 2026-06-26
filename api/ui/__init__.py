"""
pipeline.ui — ipywidgets-based theory input UI.

This package powers ``notebooks/theory_builder.ipynb``.  Run that
notebook, fill out the form, click Save, and a ``.theory.py`` file
appears in ``theories/`` ready to be loaded by
``notebooks/theory_runner.ipynb``.
"""
from api.ui.main import TheoryUI

__all__ = ['TheoryUI']
