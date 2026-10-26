.. _running-cumulants_moments:

Cumulants, moments, delta components and k >= 3
===============================================

.. admonition:: Chapter plan (replace with the text)
   :class: note

   - Contents: connected cumulant vs moment vs central moment (``output=``), the extra runs it costs; k ≥ 3 axis slices,
     base lags, full grid, tie handling at slice points; ``dd.cumulant(res)``: smooth part per order, ``DeltaComponent``,
     ``ModeTerm``, matrix-exponential triples (``.me``, ``.modes``, ``.integrate_against``, ``.spectrum``); ``discard=`` for memory.
   - Sources: ``daedalus`` k-point helpers, ``api/cumulant.py``, ``api/me_output.py``, CHANGELOG (evaluate-and-discard).
   - Examples: skewness of the OU quartic (k = 3) and its slices; δ spike of a Hawkes autocovariance.
   - Status: now. Effort: M.
