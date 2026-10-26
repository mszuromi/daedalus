.. _concepts-diagrams:

Diagrams and what loop order means
==================================

.. admonition:: Chapter plan (replace with the text)
   :class: note

   - Purpose: how to choose ``k``, ``max_ell``, ``taylor_order``, and what the numbers per order are.
   - Contents: vertices from the Taylor expansion about the saddle (vertex table of ``check()``); cumulants as sums of
     connected diagrams; loop order :math:`\ell` as the power of the small parameter (noise amplitude or :math:`1/N`); the default
     ``taylor_order = max(k + 2 max_ell, 2)``; ``g_eff``, the loop-parameter estimate; cumulants vs moments (set-partition
     assembly); k ≥ 3 cumulants as functions of :math:`k-1` lags; how many diagrams per :math:`(k,\ell)` (pointer to ch. 39).
   - Sources: ``engine/core/vertices.py``, analyzer G5 (``_g_eff``), ``daedalus._assemble_moment_temporal``,
     ``dd.plot_prediagrams``.
   - Examples: prediagram figure for :math:`(k,\ell)=(2,1)` (``dd.plot_prediagrams``, needs graphviz: build-time optional);
     OU quartic orders vs the exact series.
   - Status: now. Effort: M.
