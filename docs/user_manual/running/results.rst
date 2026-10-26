.. _running-results:

The result and how to save it
=============================

.. admonition:: Chapter plan (replace with the text)
   :class: note

   - Contents: the result dict schema (temporal, switch-on, spatial k = 2, spatial k ≥ 3); ``status``
     (``ok`` / ``incomplete`` / ``approximate``) and what to do with each; ``C_tau``, ``C_tau_by_ell``, ``total_C``; ``mf``, ``params``;
     ``route``, ``phase_j_engine``, ``analysis``, ``config`` echo; ``dd.summary``; plotting (``dd.plot_cumulant`` and its
     dispatch); ``save_npz`` / ``save_csv`` / ``load_npz`` / ``load_provenance`` and the version stamp; ``generate_report`` (PDF);
     ``export_tikz`` for diagrams.
   - Sources: ``api/compute.py`` result docstring, ``api/save.py``, ``api/report.py``, ``daedalus.export_tikz``.
   - Auto: partly: the schema table could be generated if the result docstring is turned into a structured table
     (follow-up: one source of truth for the schema).
   - Status: now. Effort: M.
