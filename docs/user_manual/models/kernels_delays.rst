.. _models-kernels_delays:

Kernels, convolutions and delays
================================

.. admonition:: Depends on unmerged work
   :class: warning

   Parts of this chapter wait for: :pr:`105`.

.. todo::

   Kernels, convolutions and delays: write the parts that wait for :pr:`105` once they merge.

.. admonition:: Chapter plan (replace with the text)
   :class: note

   - Contents: ``Kernel`` constructors (exponential, alpha, …); ``define_kernel`` and ``Conv(K, x)``; rational kernels and
     the state-space realization; heavy-tailed or non-rational kernels (refused or boxed, which); discrete delays
     ``x(t - d)``; the Padé route (automatic order, ``delay={'order':…}`` or ``{'tol', 'n_max', 'wall_max', 'tree'}``),
     exact tree with delays; ``DelayOrderWarning``; the **automatic order cache** (replay per parameter point,
     ``order_from_cache``, ``clear_order_cache``).
   - Sources: ``api/delay.py``, ``api/_delay_exact.py``, CHANGELOG (delay kernels), tutorial 04, delay example notebooks
     (PR :pr:`35`).
   - Examples: delayed OU vs its exact solution; delayed-alpha Hawkes.
   - Status: now; order-cache section **after :pr:`105`**. Effort: M.
