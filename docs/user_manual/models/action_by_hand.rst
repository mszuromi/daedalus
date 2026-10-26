.. _models-action_by_hand:

Writing the action by hand
==========================

.. admonition:: Chapter plan (replace with the text)
   :class: note

   - Purpose: models the front end cannot express, and reading old model files.
   - Contents: ``ModelBuilder`` / ``TemporalModelBuilder`` / ``SpatialModelBuilder``; action text grammar (fields,
     response fields, ``Conv``, ``Dt``, ``Lap``); ``set_action_text``, ``extra_action``; noise templates (``GTaSNoise``);
     equivalence check against the SDE front end; the deprecated ``Theory*`` aliases.
   - Sources: ``api/model.py``, ``api/model_compiler.py``, ``api/model_templates.py``, tutorial 06.
   - Status: now. Effort: M.
