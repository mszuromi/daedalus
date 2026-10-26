.. _trust-invariance:

Unit invariance and determinism
===============================

.. admonition:: Chapter plan (replace with the text)
   :class: note

   - Contents: rescaling time or field units gives the rescaled result (what is tested, where thresholds are relative);
     byte-identical results across processes, worker counts, cold and warm caches; what is not guaranteed (cross-machine
     round-off, BLAS); how to record provenance (``run_info.json``, npz stamps).
   - Sources: CHANGELOG 0.3.0 (unit invariance, determinism), 0.3.2 (cold/warm caches), ``docs/parallel.md``.
   - Status: now. Effort: S.
