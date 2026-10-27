#### Audit cleanups: builder messages without version jargon, stale docstrings, one dead table

- **Builder refusals.** `boundary()`, `initial()` and `dyson()` named an internal "v1 / v1.5 / v2" roadmap and pointed at
  design notes that are not distributed. They now say what is supported (`'infinite'` and `'periodic'`; `'stationary'`;
  `'off'` and `'fixed'`) and what is not.
- **Docstrings.** `phase_j_pipeline` no longer says loop diagrams are skipped (every diagram goes through
  `integrate_diagram`); the `initial()` and `dyson(tol=)` docs lose the version labels.
- **Dead code.** `api.noise._RESERVED` (never read) is removed.
- **Test.** `slice_me` at the nominal lag grid equals `dd.run`'s slice to 1e-13 (the old 3e-7 mismatch came from nudged
  pinned legs and went away with the exact tie).
- No numbers move.
