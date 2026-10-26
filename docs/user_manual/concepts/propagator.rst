.. _concepts-propagator:

The propagator and its state-space form
=======================================

.. admonition:: Chapter plan (replace with the text)
   :class: note

   - Purpose: the linear response that every diagram is built from; why "poles" appear in messages.
   - Contents: :math:`G(\omega)=K(\omega)^{-1}`, causality and the retarded structure; instantaneous (δ) parts; the
     state-space realization :math:`G(t)=Ce^{At}B` (default engine) vs the pole engine; close and repeated rates; DAE
     models; undriven fields (singular propagator) and the analyzer finding for them.
   - Sources: ``api/_realization.py``, ``api/_propagator.py``, ARCHITECTURE (Phase J engine), ``docs/phase_j_unbounded_exact.md``.
   - Examples: OU, two coupled fields with equal rates (state-space handles it), a DAE example (``Dendritic_nonlinear``).
   - Status: now. Effort: M.
