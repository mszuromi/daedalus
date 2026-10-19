#### One environment-switch parser, rollback switches retired, every state-space decline reason named (audit wave W7, part 2)

- **Retired switches (RED-02; FutureWarning when set, value ignored).** Each of them restored code that is now
  deleted (~2.6k lines in `final_integral.py`), most of it reproducing numbers known to be wrong:
  `DAEDALUS_PHASE_J_LEGACY` (the pre-0.2.0 numbers are no longer reproducible; the frozen baselines in
  `tests/fixtures/phase_j_legacy_baseline.{npz,json}` keep them), `DAEDALUS_PHASE_J_THETA0_CONST_ROW=legacy_clip`
  (the Theta(0) = 1 polygon clip), `DAEDALUS_PHASE_J_STRUCTURAL_ZEROS=0`, `DAEDALUS_PHASE_J_NQUAD_HARDENED=0` (the
  default-tolerance quadrature routines stay where the hardened fallback uses them), `DAEDALUS_PHASE_J_DBM=0`,
  `DAEDALUS_PHASE_J_DD_KERNELS=0` (the boxed closed forms with +-600 overflow guards and the chain simplex),
  `DAEDALUS_CHAIN_UPPERS_KERNEL`, `DAEDALUS_CHAIN_UPPERS_MEMO`, `DAEDALUS_PHASE_J_SEMIINF_CHAIN=0`,
  `DAEDALUS_PHASE_J_UNBOUNDED_2D=box` (with `POLYGON_BBOX_CAP`), `DAEDALUS_PRUNE_ZERO_MODES`,
  `DAEDALUS_POLE_CLUSTERS=0` (the deformation is always on), `DAEDALUS_PHASE_J_LEGACY_SETUP`,
  `DAEDALUS_SETUP_ZERO_EXIT`, `DAEDALUS_SETUP_PROP_TD`, `DAEDALUS_SETUP_LAZY_SR`, `DAEDALUS_SETUP_AUT_MEMO`,
  `DAEDALUS_SETUP_DELTA_SOLVE` (the speed levers: same numbers by construction), `DAEDALUS_SHOTNOISE_INTEGRATED=0`
  (it silently dropped shot-noise subsets), `DAEDALUS_EXPAND_LEGACY` (its single `taylor()` dropped degree-3 terms),
  `DAEDALUS_EDGE_GENERATOR`, `DAEDALUS_ORIENTATION_CHECKER`, `DAEDALUS_DELTA_ORBIT_DEDUP` (the alternative
  enumeration generators stay as test references, selectable by module attribute only). The module attributes the
  retired switches set (`THETA0_CONST_ROW_MODE`, `STRUCTURAL_ZEROS`, `NQUAD_HARDENED`, `USE_DBM_FALLBACK`,
  `USE_PHASE_J_DD_KERNELS`, `SEMIINF_CHAIN`, `UNBOUNDED_2D`, `PRUNE_ZERO_MODES`, `USE_CHAIN_UPPERS_MEMO`,
  `CHAIN_UPPERS_KERNEL`, `USE_SETUP_*`, `USE_SHOTNOISE_INTEGRATED`, `symmetry.USE_AUT_MEMO`) are gone, and so are
  their entries in the saved `phase_j_flags` stamp. Old result files stamped `theta0_legacy_clip` still load (with
  their warning). Older CHANGELOG entries that say a retired switch "reproduces" or "restores" something describe
  0.2.1 and earlier. A retired switch set after import warns at the next Phase J pass. The pole engine (`DAEDALUS_PHASE_J_ENGINE=poles`) stays fully supported; its default routes
  are unchanged: the zoo is bit-identical on both engines (`DAEDALUS_PHASE_J_ENGINE` default and `poles`).
- **One parser for every switch (RED-05).** New `engine._env` (on/off words `1 true yes on` / `0 false no off`,
  choices, integers, numbers, memory budgets); garbage raises `ValueError` naming the variable instead of meaning
  "on" at some sites and "off" at others. `DAEDALUS_PHASE_J_ENGINE` has one reader (four of five accepted a typo as
  the default). The two memory budgets share one meaning: a positive number of GB, or `off` / `none` / `inf` /
  `unlimited` for no limit; `0` and negative values now raise (they meant "no limit" for `PHASE_J_MEM_BUDGET_GB`
  and "refuse every loop" for `SPATIAL_MEM_BUDGET_GB`, where `off` raised), and an empty value is the default (it
  meant "no limit" for the Phase J budget); `SPATIAL_MEM_BUDGET_GB=0` now raises `ValueError` (it raised
  `SpatialPropagatorError` for every loop) and `off` works. Validated now (an unknown word raises instead of meaning
  "on" or "off"): `DAEDALUS_PHASE_J_SS_FALLBACK` (checked at the start of `compute_cumulants`, not at the first late
  refusal), `DAEDALUS_PREFLIGHT`, `DAEDALUS_DELAY_EXACT_TREE` and `DAEDALUS_FASTENUM` (only `0` switched them off;
  now `off` / `false` / `no` do too), `DAEDALUS_CERT_BACKEND` (`fast` / `sage`, any case; `nauty` silently meant
  `sage`), `DAEDALUS_PREDIAGRAM_PROCS` (an integer >= 1: `0` and negatives were clamped to 1, now raise; empty = 1),
  `DAEDALUS_PREDIAGRAM_FORMAT` (empty = `auto`), `DAEDALUS_PHASE_J_DRESS_KERNELS`, `DAEDALUS_PHASE_J_DBM_MP`,
  `DAEDALUS_PHASE_J_VALIDATE_DELTA`, `DAEDALUS_PHASE_J_ON_DIVERGENCE`, `DAEDALUS_TAU0`. The spatial resolution knobs
  (`SPATIAL_GRID_NT`/`NS`, `SPATIAL_Q_CUT`, `SPATIAL_N_Q`, `SPATIAL_MC_N`, `SPATIAL_INTEGRATOR`,
  `SPATIAL_FORCE_NUMERICAL_FT`, `SPATIAL_EQTIME`, `SPATIAL_SEGMENT_NODES`) keep their own readers (unchanged here).
- **Every state-space decline reason is named (RED-01, stage 1).** `statespace.DECLINE_REASONS` /
  `DECLINE_PREFIXES` list every reason the route gives for leaving a model, diagram or subset to the pole engine, or
  for refusing an evaluation at call time; `tests/test_ss_decline_reasons.py` fails when the code produces a reason
  that is not named (or names one it no longer produces).
- **Repeated-rate conductance kernels on the pole engine say where to go (LOUD-05).** With
  `DAEDALUS_PHASE_J_ENGINE=poles` an alpha (Erlang-2) conductance kernel at one loop stays `status='incomplete'`
  (the pole engine needs separated rates); the failure reason now names the default state-space engine, which
  computes it exactly (equal to the Markov-embedded twin to 1e-10, new test).
