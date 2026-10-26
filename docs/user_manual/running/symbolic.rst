.. _running-symbolic:

Symbolic outputs
================

.. admonition:: Chapter plan (replace with the text)
   :class: note

   - Contents: tree-level covariance in the Fourier domain (``tree_covariance``, exports to sympy, LaTeX, numpy, JSON);
     time-domain mode sums (``tree_modes``) with Routh assumptions (``AssumptionViolation``); kernel pairings
     (``pair_with_kernel``, ``integrate_against``); loop-level symbolic cumulants and triples (``symbolic_cumulant``,
     ``symbolic_triple``) and when they are refused (``SymbolicLoopRefused``, ``SaddleWarning``).
   - Sources: ``api/symbolic_out.py``, ``api/symbolic_loop.py``, CHANGELOG (symbolic outputs), tutorial 05.
   - Examples: OU covariance :math:`D/\mu\,e^{-\mu|\tau|}` as LaTeX; the one-loop equal-time variance as a rational function.
   - Status: now. Effort: M.
