.. _performance-caches:

Caches
======

.. admonition:: Chapter plan (replace with the text)
   :class: note

   - Contents: what is cached (prediagram skeletons, typed diagrams, the expanded action, propagators) and where
     (``saved_prediagrams/``, ``saved_models/``); the 21 shipped ``(k, ℓ)`` cells; ``dd.build_cache`` (report, estimate,
     ``dry_run``) and ``dd.fetch_cache``; invalidation and staleness; ``use_cache=False``; disk size; cold and warm results
     are identical; delay-order cache (:pr:`105`).
   - Sources: README prediagram cache, ``engine/enumeration/prediagram_cache.py``, ``api/_expand_cache.py``.
   - Status: now. Effort: S.
