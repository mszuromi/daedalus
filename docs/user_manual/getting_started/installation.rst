.. _getting_started-installation:

Installation
============

.. admonition:: Depends on unmerged work
   :class: warning

   Parts of this chapter wait for: the ``daedalus-msrjd`` release; the namespace move.

.. todo::

   Installation: write the parts that wait for the ``daedalus-msrjd`` release; the namespace move once they merge.

.. admonition:: Chapter plan (replace with the text)
   :class: note

   - Purpose: a working environment in under 15 minutes on macOS, Linux or WSL2.
   - Audience: new users; people who have never used SageMath.
   - Contents: what SageMath is and why Daedalus needs it (one paragraph, no more); route A conda/mamba with
     ``environment.yml``; route B native Sage 10.9 + ``sage -pip install numba pydot``; editable install
     ``sage -pip install -e .``; the ``python`` vs ``sage -python`` rule inside the conda env; Jupyter kernels
     (``sagemath``, ``sagemath-10.9``); verify the install (the README smoke test, doctested); optional extras (graphviz,
     pyperclip); tested versions table; platform notes (macOS never forks, Windows via WSL2); upgrading between
     releases (caches, ``load_npz`` stamps).
   - Sources: README (Quick start, Requirements, Installation, Verify, Dependencies), ``environment.yml``,
     ``pyproject.toml``, CONTRIBUTING.
   - Examples: the smoke test as a doctest.
   - Auto: the tested-versions table could be generated from ``environment.yml`` (small script; later).
   - Status: now; gets a PyPI section **after the ``daedalus-msrjd`` release** and an import-path note **after the namespace
     move**. Effort: S.
