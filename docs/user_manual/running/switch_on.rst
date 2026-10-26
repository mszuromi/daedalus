.. _running-switch_on:

Switch-on transients
====================

.. admonition:: Chapter plan (replace with the text)
   :class: note

   - Contents: ``t_min``, ``times`` ((n, k) array or 1-D/tuple for equal times), ``initial_covariance`` (k = 2 tree);
     ``C_times``; which routes support it (state-space only), ``SwitchOnUnsupported``; what is and is not non-stationary.
   - Sources: ``api/compute.py`` (switch-on), CHANGELOG, tutorial 05.
   - Examples: variance relaxing to its stationary value, tree vs one loop.
   - Status: now. Effort: S.
