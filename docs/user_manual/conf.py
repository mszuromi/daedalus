"""Sphinx configuration of the Daedalus user manual.

Build it with SageMath's Python (autodoc imports the package and the doctests run it):

    sage -python -m sphinx -W --keep-going -b html docs/user_manual docs/user_manual/_build/html
    sage -python -m sphinx -W --keep-going -b doctest docs/user_manual docs/user_manual/_build/doctest

Every extension used here ships with Sphinx or with SageMath 10.9 itself (furo, sphinx-copybutton), so a Sage
install builds the manual without installing anything.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, ROOT)

import daedalus  # noqa: E402  (the version below is the package's own)

project = 'Daedalus'
author = 'Matthew Szuromi and Gabriel K. Ocker'
copyright = '2026, ' + author
release = daedalus.__version__
version = '.'.join(release.split('.')[:2])

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.doctest',
    'sphinx.ext.extlinks',
    'sphinx.ext.mathjax',
    'sphinx.ext.todo',
    'sphinx.ext.viewcode',
    'sphinx_copybutton',
]

exclude_patterns = ['_build', 'README.md', 'requirements.txt']

# ``.. todo::`` marks a part that waits for unmerged work; the status page collects them.
todo_include_todos = True

# Links into the repository (tutorials, example notebooks, files) and to pull requests.
_GH = 'https://github.com/mszuromi/daedalus'
extlinks = {
    'pr': (_GH + '/pull/%s', '#%s'),
    'src': (_GH + '/blob/main/%s', '%s'),
    'tutorial': (_GH + '/blob/main/notebooks/tutorials/%s.ipynb', 'tutorial %s'),
    'example': (_GH + '/blob/main/notebooks/examples/%s.ipynb', '%s'),
}

autodoc_member_order = 'bysource'
autodoc_default_options = {'members': True, 'undoc-members': False}
autosummary_generate = False

# Only the manual's own ``testcode`` / ``doctest`` directives run; the ``>>>`` examples inside docstrings that
# autodoc pulls in are illustrations, not tests. Each document runs in one fresh namespace.
doctest_test_doctest_blocks = ''
doctest_global_setup = '''
import numpy as np
'''

html_theme = 'furo'
html_title = f'Daedalus {release}'
html_theme_options = {
    'source_repository': _GH,
    'source_branch': 'main',
    'source_directory': 'docs/user_manual/',
}

# xelatex: the text uses Unicode (Greek letters, ℓ, ∘, ≥) that pdflatex's inputenc cannot set.
latex_engine = 'xelatex'
latex_documents = [('index', 'daedalus_manual.tex', 'The Daedalus Manual', author, 'manual')]
latex_elements = {'papersize': 'a4paper', 'pointsize': '11pt'}
