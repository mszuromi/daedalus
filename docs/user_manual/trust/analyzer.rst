.. _trust-analyzer:

The model analyzer
==================

.. admonition:: Chapter plan (replace with the text)
   :class: note

   - Contents: what it is for (refuse or warn before computing); gates G0 declarations, G1 structure, G2 saddle,
     G3 propagator, G4 noise, G5 regime (``g_eff``, Taylor check), G6 run diagnostics (error accounting, accuracy
     estimate); findings (``code``, ``level``, ``message``, ``remedy``); the assumptions ledger; ``Policy`` (strict / permissive,
     consents, ``downgrade``), ``preflight``; standalone ``analyze(model, question, policy)``, ``report.table()``,
     ``--dry-run``; reading the table line by line.
   - Sources: ``api/analyze.py``, ``api/routing.py``, README dry-run table.
   - Auto: **yes**: the findings code table (ch. 46) from the analyzer's code registry (needs a small registry if
     codes are only string literals today; follow-up).
   - Status: now. Effort: M.
