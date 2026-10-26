.. _running-resummation:

Resummation
===========

.. admonition:: Depends on unmerged work
   :class: warning

   Parts of this chapter wait for: :pr:`106`.

.. todo::

   Resummation: write the parts that wait for :pr:`106` once they merge.

.. admonition:: Chapter plan (replace with the text)
   :class: note

   - Contents: ``dd.hartree`` and ``Config(resum='hartree')``: multi-start, several solutions, advisories; spectra and
     covariance from Hartree; ``resum='screened'``: trial propagator, minimal sensitivity, output keys
     (``C_tau_by_order``, ``C_tau_spread``, ``C_tau_loops``), cost; refusals (``HartreeUnsupported``); what is out of scope.
   - Sources: ``api/hartree.py``, ``api/resum.py``, ``docs/resummation.md`` (all in :pr:`106`).
   - Status: **after :pr:`106`**. Effort: M.
