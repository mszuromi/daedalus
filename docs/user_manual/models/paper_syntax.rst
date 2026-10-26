.. _models-paper_syntax:

Paper-style equations and symbols
=================================

.. admonition:: Depends on unmerged work
   :class: warning

   Parts of this chapter wait for: :pr:`107`.

.. todo::

   Paper-style equations and symbols: write the parts that wait for :pr:`107` once they merge.

.. admonition:: Chapter plan (replace with the text)
   :class: note

   - Contents: ``m.sde('dx = f dt + g dW + c dN(rate)')``; subscripted shared sources; ``d<X>_<name>`` with ``noises=``;
     ``∘`` / ``o`` for Stratonovich (Gaussian) or Marcus (jump) per term, mixing interpretations; ``sde_text()`` round trip;
     equivalence with the builder calls (byte-identical models); limits (operator left-hand sides);
     **symbols**: declarations return Sage symbols (``x = m.field('x')``), expressions as Sage objects, calls ``phi(v)``,
     indexed ``w[i, j]``, delays ``x(t - d)``.
   - Sources: ``api/sde_paper.py``, ``api/sde_symbols.py`` (:pr:`107`), README "Mixed noise".
   - Examples: the mixed-noise model; the same model three ways (paper text, builder strings, symbols).
   - Status: paper syntax now (:pr:`103` merged); symbols **after :pr:`107`**. Effort: S + S.
