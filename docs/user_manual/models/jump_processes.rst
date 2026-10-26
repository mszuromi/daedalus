.. _models-jump_processes:

Jump processes and reaction networks
====================================

.. admonition:: Depends on unmerged work
   :class: warning

   Parts of this chapter wait for: :pr:`102`.

.. todo::

   Jump processes and reaction networks: write the parts that wait for :pr:`102` once they merge.

.. admonition:: Chapter plan (replace with the text)
   :class: note

   - Contents: ``dd.JumpProcess``: species, system size :math:`\Omega`, ``reaction('A + B -> C', rate=)`` with stochastic /
     macroscopic / rate-law combinatorics, ``<->``; ``transition(rate, jump)``; what model it builds (zero drift plus one
     Poisson source per transition); loop order as :math:`1/\Omega`; ``to_master_equation``, ``compare``, ``convergence``;
     refusals (``JumpProcessError``).
   - Sources: ``api/jump_process.py`` (:pr:`102`), ``api/master_equation.py``, ``docs/master_equation.md``.
   - Examples: immigration-death (exact Poisson); dimerisation, tree error :math:`\sim 1/\Omega`, one loop :math:`\sim 1/\Omega^2`;
     Schlögl.
   - Status: **after :pr:`102`**. Effort: M.
