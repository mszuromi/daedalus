#### Loud edge cases, TikZ labels, caches and simulators (audit wave W9, part 1)

No computed number moves: every change below is a refusal, a message, a label, or a file-handling fix. The zoo is
bit-identical.

- **Prediagram cache writes are race-free and tolerate a read-only root (LOUD-03, LOUD-04).** Cache files are written
  through a writer-private temp name (pid + random suffix) and renamed at the end, and the temp file is removed on
  any failure, so two cold-cache runs of one cell no longer crash with `FileNotFoundError`. A shipped-cell hit that
  cannot be copied into the local cache root (read-only mount, no permission) is now served from memory with a
  warning instead of raising `PermissionError`; `build_cache` still raises, since writing is its job.
- **TikZ export names edges and fields correctly (ACC-71, GEN-13).** Per-edge propagator labels and the
  `style_by_field` line styles were looked up under a reconstructed `(u, v, copy)` key that never matched a real
  typed diagram, so every edge read the bare `G`; they now use the diagram's own edge key (the old spelling is the
  fallback for hand-built stand-ins). Field symbols are decoded by role (a response leg loses a trailing `t`, a
  physical leg a leading `d`): a field called `input` was drawn as `inpu`/`dinpu`.
- **`generate_report(result=...)` describes the result (LOUD-06).** The cover page reads `k`, `max_ell`, the
  parameters, the external fields and the model name from `result['config']`; `parameters=` is no longer needed with
  a reused result (it raised `AttributeError`), and a stale `max_ell=0` default no longer mislabels the cover.
- **Symbolic output (ACC-72, LOUD-12).** `TreeCovariance.default_values()` no longer swallows a failed mean-field
  solve into `{}`: it warns (`RuntimeWarning`, with the cause). A parameter raised to an exactly representable
  float power (`D**0.5`) is the rational power `D**(1/2)` in the bridge (the scan and the conversion disagreed and
  refused it with a message about omega); an inexact exponent gets a message about the exponent.
- **Noise library (LOUD-10).** `FinitePMF` accepts float probabilities that sum to 1 up to 1e-12, like
  `Hyperexponential`; `CGF('1/theta')` raises `NoiseError` ("not analytic at theta = 0") instead of a raw Sage
  `ValueError`. (`PhaseType` with float entries at n >= 5 was checked and does not overflow; no change.)
- **Simulators refuse what has no answer (LOUD-11).** The point-process simulator raises `PPSimulationError` naming
  the field, the step and the runaway rate when a rate feedback diverges (the README's exponential-Hawkes example at
  seed 0 died inside numpy with "lam value too large"); `mean_rate`, `cross_covariance`, the delay-SDE
  `autocovariance`, `third_cumulant_x` and the coupled reaction-diffusion simulator need >= 2 independent
  paths/snapshots/replicas for an error bar (they returned NaN); the 1-D and 2-D spectral simulators refuse
  `mu <= 0` (NaN fields, or a negative default `dt`).
- **Spatial raw moments refuse before the run (decision B3).** `dd.run(..., output='moment')` for a spatial k = 2
  run that needs the means' loop corrections (`max_ell >= 2`, or `max_ell = 1` with a nonzero saddle) now raises
  before the expensive loop run (a one-event tree-level k = 1 run reads the saddle for `max_ell = 1`).
- The two-field OU test of the Phase J setup fast path uses an inline model instead of a git-ignored model file.
