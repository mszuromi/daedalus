.. _concepts-saddle:

The saddle point (mean field)
=============================

.. admonition:: Chapter plan (replace with the text)
   :class: note

   - Purpose: what the expansion point is, how it is found, and what happens with several.
   - Contents: saddle equations; all roots, linear stability, ``fixed_point_index``, multi-start search
     (``mf_dae_n_starts``, ``mf_dae_seed_box``); marginal modes; non-polynomial drifts (special functions supported in the
     residual); the k = 1 convention (tree mean = saddle, loops shift it); bistable models and why the expansion is
     about one basin.
   - Sources: ``api/_mean_field*.py``, analyzer G2, CHANGELOG 0.3.x mean-field entries, ``models/ou_quartic_double_well``.
   - Examples: double well, two stable roots, choosing one; figure of the two basins.
   - Status: now. Effort: M.
