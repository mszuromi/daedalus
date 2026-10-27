#### Conserved noise `div(eta)`: refused with the reason, in every form; two DeprecationWarnings removed

- **Conserved noise** (the divergence of a white vector noise, noise spectrum `var |q|^2`, the noise of Model B) is
  refused, and each refusal now says why: the spatial integrator takes momentum-independent noise only (its tree heat
  kernels and its loop integral carry one constant noise vertex per correlation line), and the Model B dynamics
  conserved noise belongs to also need the `k^4` propagator of `lap(lap(u))` and the conserved vertex `lap(u^3)`,
  which the spatial pipeline refuses as well. The reason (`engine.integration.spatial.CONSERVED_NOISE_WHY`) is given by
  `Gaussian(space='conserved')` (before: "not supported yet"; also on its line of a YAML `sde:` block), an operator in
  a noise coefficient of an SPDE (before: only "operators act in the equations"), `div(eta)` / `grad` / `nabla` in an
  SPDE (before: "declare the function"), an operator on a response field in `extra_action` (before: "acts on no
  field"), `lap(dW)` in a paper-style equation (before: "has no differential"), and a hand-written action whose
  response-field quadratic term holds a Laplacian, at every spatial driver (before: "momentum-dependent noise is not
  supported by the spatial white-noise extractors"). Documented in `api/sde.py`, `api/noise.py`, the README and
  `docs/spatial_pipeline.md`. No model or number changes; temporal models keep their messages.
- **mpmath >= 1.4**: the pole root finder (`api._pole_clusters`) and the residue certificate (`api._propagator`) passed
  descending coefficients to `mp.polyroots` / `mp.polyval` without `asc=`, which warns (`DeprecationWarning`) on every
  pole-route and delay run. They now pass ascending coefficients with `asc=True` where mpmath takes it (older releases:
  the descending call as before); the roots and values are the same numbers.
- `tests/_sde_model_catalogue.py` uses `into=` / `as_field=` (was the deprecated `couples=` / `expose=`); the models
  are unchanged.
- Tests (`tests/test_sde_conserved_noise.py`): each form above refuses with the reason; temporal models keep the old
  hints; the noise extractor keeps its old message for a missing value and returns white noise unchanged; the
  hand-written conserved action is refused at the k = 2 tree, the k = 2 one-loop and the k = 3 drivers. The mpmath
  helpers give the roots and values of the descending call with every warning an error.
