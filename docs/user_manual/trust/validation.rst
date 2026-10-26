.. _trust-validation:

Validation oracles
==================

.. admonition:: Chapter plan (replace with the text)
   :class: note

   - Contents: exact cases (linear models, linear Hawkes, OU series); the master-equation oracle (``dd.master_equation``:
     stationary law, cumulants to 4, lag correlations, truncation flux, Kramers-Moyal extrapolation,
     ``compare_with_daedalus``); simulators (``SDE.simulate``, ``simulations/`` integrators, ``cumulant_estimator``); comparing
     with simulation honestly (standard errors, z-scores, burn-in, binning of spike trains); what the loop expansion
     should and should not match.
   - Sources: ``api/master_equation.py``, ``docs/master_equation.md``, ``api/pp_simulate.py``, ``simulations/``, example
     notebooks' simulation sections.
   - Examples: immigration-death to fourth order; OU quartic vs Euler-Maruyama with error bars.
   - Status: now. Effort: M.
