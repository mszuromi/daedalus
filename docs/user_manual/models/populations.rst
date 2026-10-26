.. _models-populations:

Populations and indexed parameters
==================================

.. admonition:: Depends on unmerged work
   :class: warning

   Parts of this chapter wait for: the array-connectivity feature.

.. todo::

   Populations and indexed parameters: write the parts that wait for the array-connectivity feature once they merge.

.. admonition:: Chapter plan (replace with the text)
   :class: note

   - Contents: ``population('E', size=n)``; indexed parameters (``indexed_by=``), sums over populations in rates and
     drifts; external fields ``('n', 2)``; accessors ``res['params']['w', 1, 2]``; **connectivity from numpy arrays**
     (backlog); size limits and cost.
   - Sources: ``api/sde.py`` (population, walk, shape), README Hawkes snippet, ``models/multipopulation_test``.
   - Examples: two-population Hawkes network cross-covariance.
   - Status: now; array-connectivity section **after the array-connectivity feature** (backlog). Effort: S.
