.. _performance-memory:

Memory
======

.. admonition:: Chapter plan (replace with the text)
   :class: note

   - Contents: the budgets ``PHASE_J_MEM_BUDGET_GB`` and ``SPATIAL_MEM_BUDGET_GB`` (one semantics), ``PhaseJResourceError``
     and how to read its numbers; ``discard=True/'auto'``; per-worker caps; on macOS, memory pressure vs RSS (compressed
     memory) and how to watch it; typical footprints by :math:`(k,\ell)`.
   - Sources: ``engine/_env.py`` (``budget_gb``), ``api/compute.py``, CHANGELOG (memory guards), ``docs/parallel.md``.
   - Status: now. Effort: S.
