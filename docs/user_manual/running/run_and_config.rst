.. _running-run_and_config:

dd.run and dd.Config
====================

.. admonition:: Chapter plan (replace with the text)
   :class: note

   - Contents: ``dd.run(model, cfg, module)`` vs ``compute_cumulants(...)`` (same engine, which to use); the question
     (``k``, ``external_fields``, ``max_ell``, ``parameters``, ``output``); grids (``tau_grid`` as array or ``(lo, hi, n)`` tuple, the
     tuple-vs-list rule, ``tau_max/tau_step``); numerics (``taylor_order``, ``accuracy``, ``use_cache``, ``backend``, ``discard``);
     model defaults from ``METADATA``; the analyzer options (``policy``, ``preflight``); plotting options; ``dd.config_options()``.
   - Sources: ``daedalus.Config``, ``daedalus.config_options``, ``api/compute.compute_cumulants`` signature.
   - Auto: **yes**: the option table is generated from ``dd.Config`` fields and the ``config_options`` descriptions (ch. 42).
   - Status: now. Effort: M.
