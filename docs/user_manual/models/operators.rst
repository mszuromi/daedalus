.. _models-operators:

Operators and equation forms
============================

.. admonition:: Depends on unmerged work
   :class: warning

   Parts of this chapter wait for: the second-order-equation feature.

.. todo::

   Operators and equation forms: write the parts that wait for the second-order-equation feature once they merge.

.. admonition:: Chapter plan (replace with the text)
   :class: note

   - Contents: declared operators (``TimeDerivative``, ``Laplacian``), product and call style, renaming ``Dt``, collisions
     refused; ``equation(lhs=, rhs=)``; linear operator left-hand sides ``(tau*dt + 1)*v``; **second-order equations**
     :math:`(m\,\partial_t^2+\gamma\,\partial_t+k)x`; constraints and DAEs; ``boundary``; which engines accept which operator
     (temporal-only features refused with a Laplacian).
   - Sources: ``api/operators.py``, ``api/sde.py``, CHANGELOG 0.3.2 (declared operators).
   - Examples: a filtered rate equation; a damped oscillator (second order).
   - Status: operators now; second-order section **after the second-order-equation feature** (backlog, not started).
     Effort: S now + S later.
