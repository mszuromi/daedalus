.. _models-sde_front_end:

The SDE front end
=================

.. admonition:: Depends on unmerged work
   :class: warning

   Parts of this chapter wait for: :pr:`107`.

.. todo::

   The SDE front end: write the parts that wait for :pr:`107` once they merge.

.. admonition:: Chapter plan (replace with the text)
   :class: note

   - Purpose: the main way to write a model.
   - Audience: all users.
   - Contents: ``SDE(name)``; ``field``, ``parameter`` (defaults, ``domain=``, indexed parameters), ``drift`` / ``equation``,
     ``noise(name, source, into=)``, ``define_function``, ``define_kernel``, ``stability_analysis``, ``series_order``;
     ``build()`` and what the model dict is; ``check()`` report sections; ``latex()``; ``show()``; parse errors with carets
     and hints; ``to_builder()``; ``simulate()`` for a matching simulator; refusals at declaration time.
   - Sources: ``api/sde.py``, README SDE sections, tutorial 01, CHANGELOG 0.3.2 (``check``, ``latex``, caret errors).
   - Examples: OU quartic; a two-field model; a deliberate typo and its error message (doctest of the message).
   - Auto: method list via autosummary (in ch. 41); the text is written.
   - Status: now; keyword names change with **:pr:`107`** (``couples=`` → ``into=``, ``expose=`` → ``as_field=``): write with the new
     names and merge after :pr:`107`. Effort: M.
