.. _models-point_processes:

Point processes and Hawkes models
=================================

.. admonition:: Chapter plan (replace with the text)
   :class: note

   - Contents: ``point_process(name, rate, population=)``; rates that filter fields or spike trains (``Conv``); linear
     Hawkes as an exact check; nonlinear rates (exponential, quadratic, sigmoidal); shot-noise δ components of the
     correlator and how to read them (``dd.cumulant(res).delta``); spike-reset style models; ``SDE.simulate`` for point
     processes (``pp_simulate``).
   - Sources: ``api/sde.py`` (point_process), ``api/pp_simulate.py``, ``api/cumulant.py``, tutorial 03, example
     ``Hawkes_quadratic_alpha``, ``models/linear_hawkes``.
   - Examples: linear Hawkes tree = closed form (one loop exactly 0); quadratic alpha Hawkes vs simulation.
   - Status: now. Effort: M.
