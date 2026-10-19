#### One option surface, one grid rule, one exception base, result-schema docs, truncation in the accuracy estimate (audit wave W9, part 2)

Zoo numbers are bit-identical. What moves: the G6 accuracy estimate `R` of a run whose propagator was balanced-
truncated (by at most the truncation bound, <= 1e-12 at the default tolerance), and the grid rule of
`compute_cumulants` (below).

- **`dd.Config` carries the numeric options of `compute_cumulants` (CON-02).** New fields `taylor_order`, `use_cache`,
  `accuracy`, `backend`, `discard` (None = `compute_cumulants`' default; validated at construction) are forwarded by
  `dd.run`. The YAML config reads them through `dd.Config` instead of a monkey-patched `compute_cumulants`
  (`question.accuracy`, `policy.backend`, `policy.discard` are new keys; `taylor_order` and `use_cache` already
  existed). `dd.config_options()` now lists `delay`, the new options and the alias spellings (`fundamental`,
  `spatial_grid`). (`dd.run` still takes its default tau grid from the model's `METADATA`, tau_max 10, while
  `compute_cumulants` defaults to tau_max 50: documented, not changed.)
- **One grid rule everywhere (decision B4).** `api._grids.resolve_grid` is shared by `compute_cumulants`, `dd.Config` and
  the YAML config: an `(lo, hi, n)` TUPLE is a `linspace` range, a list or array is literal points, and a 3-element
  LIST whose last entry is an integer count >= 2 (`[-5, 5, 51]`, also `[0.0, 1.0, 4.0]`) is refused with a message that
  names both spellings. `compute_cumulants` used to refuse the tuple and accept `[0.0, 0.5, 2.0]` as three literal
  points; pass `np.array([0.0, 0.5, 2.0])` for those.
- **Unbuilt models are accepted everywhere (CON-07).** `compute_cumulants`, `dd.describe_model` and
  `dd.symbolic_cumulant` build an `api.sde.SDE` / builder themselves, as `dd.run` does (`AttributeError: 'SDE' object
  has no attribute 'get'` before). The spatial k = 2 result gains `status` and `C_tau_x_by_ell` (the temporal schema's
  names; `C_tau_x_by_order` stays), and the `compute_cumulants` return docs list the spatial k = 2 keys and say that
  the k = 1 `C_tau` is the mean shift, not the mean.
- **Balanced truncation reaches the accuracy estimate and the reference guard (CON-09).** The relative H-infinity bound
  of a truncated propagator entry or dressed line (`result['phase_j_engine']['realization_check']['truncation']`) is
  added to `R` (G6) and named in `G6_ACCURACY`; a delayed OU at tolerance 1e-3 now reports R = 4.6e-4 for a measured
  shift of 1.9e-4 (it reported 1e-14). `symbolic_cumulant(result=...)` refuses a result whose own record shows
  truncated entries or dressed lines (`truncated_reference`); before, the guard re-realized the model and missed the
  dressed lines.
- **`DaedalusError`, the common base of the package's own exceptions (CON-04).** `engine.errors.DaedalusError`
  (also `api.DaedalusError`, `dd.DaedalusError`) is now a base of every exception class `api/` and `engine/` define
  (31 classes: `SDEError`, `NoiseError`, `ConfigError`, `SymbolicOutError`, `PhaseJResourceError`,
  `SpatialUVDivergenceError`, ...). Each keeps the builtin it always was (`ValueError`, `RuntimeError`,
  `NotImplementedError`, `MemoryError`, ...), so every existing `except` clause still catches it; a test fails when
  a new public exception class forgets the base. Builtins raised directly for a bad argument are unchanged. Two
  vague messages are fixed: a population sum in a drift no longer reads "not supported inside a noise parameter",
  and "Call expand() first." names what has to run.
- **Two classes called `Kernel` (CON-06).** `api.symbolic_out.Kernel` (the pairing kernel `L(s)` of the C2 output) is
  now `api.symbolic_out.PairingKernel` (`dd.PairingKernel`); `Kernel` stays as an alias of the same class, so
  `dd.Kernel`, `so.Kernel.exponential(...)` and `isinstance` checks keep working (`api.delay.Kernel` is the delay
  kernel builder and is untouched).
- **`combinatorial_factor` refuses a fractional symmetry factor (CON-10).** A Wick numerator not divisible by the
  automorphism order was floored silently; it now raises `RuntimeError` like `external_wick_compensation` (the case
  never occurs on the zoo; the check was dead defensive code).
- **The Taylor norm estimate is reproducible (found by the discard bit-identity test on Linux CI).** Since the cheaper
  norm bounds (audit SPD-03), `d_2 = ||A^2||_1^(1/2)` of a generator with more than 300 states came from scipy
  `onenormest`, which draws its start vectors from numpy's UNSEEDED global generator: two identical runs picked
  different Taylor parameters (estimates differing by ~10 %) and differed in the last bit, so
  `test_discard_is_bit_identical_and_releases` failed on about half of the Linux processes (current main included).
  `d_2` is now exact up to 2000 states (`statespace._EXACT_NORM_N`; the dense generator is already held, n^3 flops) and
  both remaining `onenormest` calls (beyond that size, and the matrix-free operator) run with a fixed seed and restore
  the caller's random state. Values of models with 301-2000 states move in the last digits against 3776472 (the
  estimate was random there).
