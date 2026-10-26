.. _running-covariance:

Covariance matrices and parameter derivatives
=============================================

.. admonition:: Chapter plan (replace with the text)
   :class: note

   - Contents: ``dd.covariance_matrix`` / ``Config(covariance=)``; shared mean field and propagator; lower triangle and the
     :math:`\tau=0` rule for :math:`(j,i)` entries, ``pairs='all'``; derivatives by fourth-order differences, error estimates,
     ``CovarianceDerivativeError``; refusals.
   - Sources: ``api/covariance.py``, README section.
   - Examples: coupled OU covariance and :math:`\partial\Sigma/\partial J`.
   - Status: now. Effort: S.
