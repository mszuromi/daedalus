.. _concepts-conventions:

Conventions
===========

.. admonition:: Chapter plan (replace with the text)
   :class: note

   - Purpose: every sign, limit and index convention in one place, each with a test that pins it.
   - Contents: Itô default and per-source interpretation; equal-time :math:`\tau=0` is the left limit (k = 2 and k ≥ 3 slice
     ties), ``DAEDALUS_TAU0``; lag direction; Fourier signs (``spectra`` vs ``Cumulant.spectrum``); shot-noise spikes and
     ``delta0``; per-order increments vs cumulative dicts (temporal and spatial); 1-based population indices and accessors
     (``res['mf']['v', 1]``); internal field names (``dx``) vs natural names; units (results are unit-covariant; ch. 33).
   - Sources: README spectra conventions, ``api/compute.py`` result docstring, CHANGELOG ("τ = 0 is the exact Itô left
     limit"), ``api/access.py``.
   - Auto: no; each convention gets a doctest that demonstrates it.
   - Status: now. Effort: M.
