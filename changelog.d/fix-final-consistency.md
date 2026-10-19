#### Spatial `C_tau_x_by_ell` is per-order increments; stage and audit labels out of library text (final consistency pass)

- **`result['C_tau_x_by_ell']` of a spatial k = 2 run is now the per-loop-order INCREMENTS** (ell = 0 the tree, ell >= 1
  the ell-loop contribution), like the temporal `C_tau_by_ell` and the spatial `total_C_by_ell`, so
  `sum(C_tau_x_by_ell.values()) == C_tau_x`. #70 had made it an alias of the CUMULATIVE `C_tau_x_by_order`, so summing
  it double-counted the tree (the one-loop Allen-Cahn 1-d term read 0.4625 instead of the exact -0.0375).
  `C_tau_x_by_order` is unchanged (cumulative: tree, tree + 1-loop, ...). Both are documented in the
  `compute_cumulants` return docs.
- Library comments and messages no longer carry internal stage or audit labels (`audit W7`, `W8p S4`, `ACC-61`,
  `S3:`, `stage S4`, ...); the sentences keep their content, and the retired-switch notes no longer end in
  "(audit W7)". `tests/test_audit_w9_docs.py` now also fails on these patterns and on placeholder tokens in
  `changelog.d/` fragments (a leftover `a leftover placeholder` in the W7 environment-switch fragment is fixed).
