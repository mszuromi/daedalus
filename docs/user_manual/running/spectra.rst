.. _running-spectra:

Spectra and integrated correlators
==================================

.. admonition:: Chapter plan (replace with the text)
   :class: note

   - Contents: ``spectra=`` / ``dd.spectra``: :math:`S(\omega)`, :math:`S(0)`, :math:`\int_0^T C`, :math:`\int_0^\infty C`, :math:`\tau_c` per order;
     conventions (cross spectra, spikes, equal times, truncated correlation time); exact vs quadrature fallback
     (``MEFallbackWarning``, ``method='exact'``); refusals.
   - Sources: ``api/spectra.py``, README spectra section, CHANGELOG 0.3.2.
   - Examples: OU quartic spectrum by order; windowed count variance of a Hawkes process.
   - Status: now. Effort: S.
