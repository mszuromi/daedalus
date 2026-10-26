.. _concepts-asymptotics:

Asymptotics, validity and resummation
=====================================

.. admonition:: Depends on unmerged work
   :class: warning

   Parts of this chapter wait for: :pr:`106`.

.. todo::

   Asymptotics, validity and resummation: write the parts that wait for :pr:`106` once they merge.

.. admonition:: Chapter plan (replace with the text)
   :class: note

   - Purpose: the honest caveat: the loop series is asymptotic, and what to do when it fails.
   - Contents: why the series is asymptotic (zero radius of convergence in the coupling); signs that it is failing
     (growing increments, ``g_eff`` warning, Taylor-check warning); optimal truncation; what resummation does
     (Hartree self-consistent Gaussian; screened loop series with minimal-sensitivity :math:`\kappa`); when to trust which.
   - Sources: analyzer G5 advisories; ``docs/resummation.md`` and ``api/hartree.py``, ``api/resum.py`` (:pr:`106`).
   - Examples: OU quartic at :math:`g=0.1,1,10,1000`: loops vs Hartree vs screened vs exact (numbers from :pr:`106`'s validation).
   - Status: concept part now; resummation part **after :pr:`106`**. Effort: M.
