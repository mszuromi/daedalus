.. _models-model_files:

Model files, YAML configs and the graphical builder
===================================================

.. admonition:: Depends on unmerged work
   :class: warning

   Parts of this chapter wait for: the observables feature.

.. todo::

   Model files, YAML configs and the graphical builder: write the parts that wait for the observables feature once they merge.

.. admonition:: Chapter plan (replace with the text)
   :class: note

   - Contents: ``models/*.model.py`` (structure, ``METADATA``, ``DEFAULT_FUNDAMENTAL``, docstring), ``dd.load_model``,
     ``dd.list_models``, ``dd.describe_model``, writing a model file (``model_serialize``); YAML: ``daedalus run config.yaml``,
     ``--dry-run``, sections (``model`` or inline ``sde:``, ``parameters``, ``question``, ``policy``, ``outputs``), errors with line
     numbers, ``config_resolved.yaml`` and ``run_info.json``; ``ModelUI`` (point-and-click builder, what it emits);
     **observables** (``m.observable('rate', 'phi(v)')``, backlog); a round-trip text format (backlog).
   - Sources: ``daedalus.load_model``, ``api/config_run.py``, ``api/model_serialize.py``, ``api/ui/``, README config section.
   - Auto: the YAML schema table (ch. 43) from ``config_run``'s validator.
   - Status: now; observables **after the observables feature** (backlog). Effort: M.
