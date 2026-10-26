.. _reference-environment:

Environment variables
=====================

.. admonition:: Chapter plan (replace with the text)
   :class: note

   user-facing switches (``DAEDALUS_PHASE_J_ENGINE``, ``PHASE_J_MEM_BUDGET_GB``,
   ``SPATIAL_MEM_BUDGET_GB``, ``DAEDALUS_PARALLEL``, ``DAEDALUS_PREFLIGHT``, ``DAEDALUS_TAU0``, ``DAEDALUS_CACHE_URL``, the
   ``SPATIAL_*`` knobs) separated from developer and rollback switches. Today about 55 names are read across the code with
   no registry; a registry in ``engine/_env.py`` would let this table be generated and tested (follow-up). Now, S.
