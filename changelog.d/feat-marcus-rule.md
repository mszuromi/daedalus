#### The Marcus rule for jump and non-Gaussian sources: `SDE(...).interpretation('marcus')`

A multiplicative jump `g(x) <> dL` now has a Marcus interpretation. A jump `z` moves the state along the flow of the
coupling, `x -> exp(z G) x` with `G = sum_i g_i(x) d/dx_i`. A conductance-type jump `-w (V - Es) <> dN`, for example,
takes `V - Es` to `e^{-w} (V - Es)`, so it never crosses `Es`. The Ito jump `(1 - w)(V - Es)` does cross it when `w > 1`.

The action stays Ito-ordered, because it is built from the generator. A source with cumulant rates `kappa_k` contributes
`sum_k kappa_k/k! D_k`, with `D_k = e^{-xt.x} G^k e^{xt.x}`, in place of `K(sum_i g_i xt_i)`. Enumeration and Phase J
are therefore unchanged.

- **Gaussian sources and Gaussian parts of a sum** take the existing Stratonovich conversion. A Gaussian-only model gives
  exactly the `'stratonovich'` action, equations and notes.
- **Jump sources with a constant coupling** are unchanged, since the flow of a constant field is a translation. A model
  with only additive jumps gives exactly the Ito action.
- **Jump sources with a state-dependent coupling** carry the series to `SDE.series_order`. Its theta-linear part (the
  mean drift) enters the action once and the mean-field equations once, as for any non-Gaussian source.
  `compute_cumulants` sizes the first omitted term at the saddle, per response degree up to `k + 2*max_ell`, and
  stores the result in `result['marcus_truncation']`. Above `1e-10` it warns (`MarcusTruncationWarning`) and adds the
  G6 finding `G6_MARCUS_TRUNCATION_LOW`; otherwise it adds `G6_MARCUS_TRUNCATION`.
- **Rates.** A state-dependent rate is the pre-jump rate and stays outside the flow.
- **Refused under Marcus, with messages:**
  - exposed sources;
  - a point process whose spikes enter an equation with a state-dependent coefficient or size;
  - couplings that read other units of a population (joint jumps);
  - state-dependent `Cumulants` and `CGF` sources, whose Gaussian part cannot be told from their jumps;
  - moment-series jump laws below the series order;
  - spatial models (`Policy(interpretation='marcus')`).
- **`'stratonovich'` with a jump source** still refuses, and its message now points to `'marcus'`.

The simulator (`SDE.simulate`) now handles jump sources coupled to fields:

- **Ito:** the pre-jump coupling times the increment.
- **Marcus:** the exact flow of the coupling over the step's increment (the flows of one source compose). It uses the
  closed form `x + psi_z(A) b(x)` for an affine coupling, and DOP853 at `rtol 1e-11` otherwise.
- **Gaussian sources under `'stratonovich'`/`'marcus'`** take the Ito drift correction.

`'marcus'` is accepted by `config_run` (`sde.interpretation`, `policy.interpretation`, and a new `sde.series_order`) and by
the analyzer's convention ledger. Models without `'marcus'` are unchanged (no numbers move).
