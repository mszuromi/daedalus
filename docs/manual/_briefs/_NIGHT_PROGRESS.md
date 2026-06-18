# Overnight job — progress tracker

Task: (1) document all changes + docs freshness pass, (2) code/folder cleanup to
`legacy/`/`docs/archive/`, (3) book-length LaTeX master manual under
`docs/manual/`, with per-section code audits + a big final audit; log
inconsistencies separately for tomorrow.

## Status

- [x] **Side task from earlier:** `theories/ou_quartic.theory.py` (white-noise
      single-field quartic OU) created + validated (2-loop 0.2s; theory C_xx(0)
      0.9496 vs sim 0.9464 ± 0.0009). Example notebook
      `notebooks/examples/temporal_ou_quartic_white.ipynb` generated + executed
      clean (binning fixed to dt_bin=0.02 so the variance matches).
- [x] **Manual scaffold:** `docs/manual/daedalus_manual.tex` (preamble, TikZ,
      callout boxes, listings) — COMPILES clean (39 pp w/ stubs). 24 section
      stubs created. `docs/manual/README.md` written.
- [x] **Front matter authored by main loop:** `sections/00-how-to-read.tex`,
      `sections/01-overview.tex` (architecture + pipeline figure).
- [x] **Workflow 1 (understand):** 18 subsystem briefs in `_briefs/*.md`
      (17.5k lines). Inventory agents (cleanup + docs manifests) finishing.
- [x] **Workflow 2 (manual-draft) attempt 1:** DIED at the 2:30am session
      limit — only `02-quickstart` drafted (666 ln). Lost the run + /tmp script.
- [x] **Cleanup phase 1 (safe, untracked only):** deleted all untracked scratch
      probes/logs/wave-json; moved `borinsky_*.{pdf,txt}` →
      `Literature/tropical_monte_carlo/`. 28 TRACKED scratch tooling files left
      in place (manifest marks them "investigate" — decide post-draft).
- [~] **Workflow 2 attempt 2 (manual-draft):** RELAUNCHED morning of 06-18,
      run **wf_fc1b9a84-ced**. Pipeline over the 19 remaining chapters (03-21)
      + appendices A-C: draft (brief+source → `sections/NN-slug.tex`) → adversarial
      audit vs code (→ `_briefs/_audit_<slug>.md`). Appendix D + AUDIT_FINDINGS
      assembled in main loop after. Each draft writes to disk immediately →
      robust to another interruption (re-run only the remaining stubs).
      Progress so far: 05-colored-markovian (1095 ln) landed.

## Remaining (gated on Workflow 2 attempt 2)

- [ ] **Cleanup phase 2 (tracked, post-draft):** archive 6 completed plan-docs
      → `docs/archive/` (BUILD_PHASE_OUTLINES, PIPELINE_PLAN, phase_j_refactor_notes,
      m_ge3_chain_simplex_fix_proposal, spatial_analytic_ift_plan,
      theory_builder_split_plan); add `scratch/` + manual build-artifacts to
      `.gitignore` and `git rm --cached` the 28 tracked scratch files; handle the
      2 dead stubs (`msrjd/core/propagator.py`, `msrjd/integration/numerical.py`)
      — delete after confirming 0 imports + tests green. Decide the "investigate"
      tracked scratch tooling. RUN TESTS before committing any source move.
- [ ] Assemble + compile full manual; iterate pdflatex to zero errors.
- [ ] Apply CRITICAL audit findings; write `sections/D-known-issues.tex` +
      `docs/manual/AUDIT_FINDINGS.md` from all audit findings + brief
      open-questions.
- [ ] Workflow 3: big final holistic audit of the assembled manual vs code +
      completeness critic.
- [ ] Update `docs/CHANGELOG.md` (this session's engine changes + white OU +
      manual + cleanup); docs freshness pass per docs manifest.
- [ ] Commit + push (per standing instruction: always push; this IS the main
      working dir; branch `spatial-extension`). Copy is implicit (working dir =
      main dir).

## Known issues already surfaced by briefs (seed for Appendix D)
- `msrjd/core/propagator.py` is a stale stub (10-line docstring; real code is
  `pipeline/_propagator.py`) — flagged in propagator brief. Cleanup candidate.
- (more to come from the audit stages / cleanup manifest)
