.. hand-written: the plan is kept below; the module pages render from the docstrings.

.. _reference-api:

API reference
=============

Generated from the docstrings. Modules whose docstrings do not yet render as reStructuredText without warnings
are listed in the plan below and join this page after the docstring pass.

.. admonition:: Depends on unmerged work
   :class: warning

   Module paths change with the move to a ``daedalus/`` package (``api.sde`` becomes ``daedalus.api.sde``); the
   directives on this page are renamed once, then. ``api.hartree`` and ``api.resum`` (:pr:`106`),
   ``api.jump_process`` (:pr:`102`) and ``api.sde_symbols`` (:pr:`107`) join when they merge.

.. todo::

   API reference: add ``daedalus`` (front module), ``api.compute``, ``api.delay``, ``api.me_output``,
   ``api.covariance``, ``api.symbolic_out``, ``api.symbolic_loop``, ``api.analyze`` and ``api.master_equation``
   after the docstring pass (72 reST warnings measured on Oct 26 2026: 34 in ``api.compute``, 10 in
   ``api.symbolic_loop``, 7 in ``api.master_equation``, 4 each in ``api.symbolic_out`` and ``api.delay``, 3 in
   ``daedalus``, the rest 1 or 2 per module), and the modules of the unmerged features.

.. admonition:: Chapter plan (replace with the text)
   :class: note

   - Pages: the ``daedalus`` front module (``run``, ``Config``, ``load_model``, plotting, ``summary``,
     ``export_tikz``, caches); ``api.sde``, ``api.sde_paper``, ``api.operators``, ``api.noise``, ``api.delay``;
     ``api.compute_cumulants``; outputs (``api.cumulant``, ``api.me_output``, ``api.spectra``, ``api.covariance``,
     ``api.symbolic_out``, ``api.symbolic_loop``); ``api.analyze``; ``api.master_equation``; builders
     (``api.model``); persistence (``api.save``, ``api.report``).
   - One page per group, with a short introduction that links the chapter explaining the feature.
   - Later gate: build with ``-n`` (nitpicky) so every cross-reference to the API must resolve.
   - Effort: M (docstring pass) + S per release.

Model input
-----------

.. automodule:: api.sde

.. automodule:: api.sde_paper

.. automodule:: api.operators

.. automodule:: api.noise

Building models by hand
-----------------------

.. automodule:: api.model

Outputs
-------

.. automodule:: api.cumulant

.. automodule:: api.spectra

Saving and reports
------------------

.. automodule:: api.save

.. automodule:: api.report
