#### Exact spectra, integrated correlators and correlation times (k = 2)

A k = 2 run can now return its spectral and integrated outputs directly, with no lag grid:
`compute_cumulants(..., spectra={'omega': ..., 'T': ...})`, `dd.Config(spectra=...)`, or the YAML key
`question.spectra`. The values are in `result['spectra']`; for a live result, `dd.spectra(result, omega, T)` gives the
same values afterwards. Each output is given per loop order and in total:

- the spectrum `S(omega)` on a frequency grid, and `S(0)`;
- the windows `int_0^T C` and `int_{-T}^0 C` for each `T`;
- the one-sided integrals to infinity;
- `C(0)`;
- the correlation time `tau_c = int_0^inf C / C(0)`.

`tau_grid=np.array([0.0])` is enough for the run, so a spectrum no longer needs a dense lag grid. With a warm cache,
a one-loop OU quartic run with spectra takes about 0.5 s.

- **Exact route.** The outputs come from the state-space matrix-exponential triples of `api/me_output.py`:
  - `S(omega)` and the integrals to infinity: one triangular resolvent solve per block.
  - `int_0^T C`: the augmented exponential `exp([[G, x0], [0, 0]] T)`. It has no cancellation at small `T` and needs
    no inverse of `G`.

  No pole, residue or quadrature is involved. Against a 384-point Gauss-Legendre quadrature of the package's own
  `C_tau`, OU quartic agrees to about 1e-15 at one and two loops. With colour noise 1e-9 from a double pole, and at the
  exact double pole, the tree agrees with 50-digit references to 1e-13.
- **Conventions.**
  - `S(omega) = int C(tau) exp(+i omega tau) dtau`, with no `1/2pi`. For a cross pair this is
    `<x1^(omega) x2^(omega)*>`, the `(1, 2)` entry of `H Q H^H`, and `S_21 = conj(S_12)`.
    `Cumulant.spectrum` keeps its opposite sign: `S(omega) = Cumulant.spectrum(-omega)`.
  - A shot-noise spike `w delta(tau)` at `tau = 0` is the flat floor `w` in `S`. It is reported as `delta0` and is in
    neither window, so `S(0) = integral_inf + integral_inf_neg + delta0`. Spikes off `tau = 0` are counted inside
    their window.
  - `C0` is the Ito left limit `C(0-)`.
  - `tau_c_by_ell[l]` is the correlation time of the cumulant truncated at order `l`.
- **Fallbacks.**
  - A loop order without triples (the pole engine, or the pole-cluster deformation) is computed by adaptive QUADPACK
    quadrature of its Phase J callables. Every value then has an error estimate (`error_by_ell`, `abs_error`), and an
    `MEFallbackWarning` is given. `method='exact'` refuses such orders instead, and `method='quad'` forces quadrature.
  - The exact delayed tree of a delay model gives `S` and the infinite integrals exactly, from its Laplace transform.
    Its finite windows are computed by quadrature, with an error estimate.
  - Padé loops are exact for the approximant. A note gives the approximant's estimated error, `result['delay']`.
- **Refused before the run.** k != 2, spatial models, `t_min`, `discard` other than False, and `output` other than
  `'cumulant'`.
- **After the run.** A `C` that does not decay gives NaN for `S` and the integrals to infinity, with the reason in
  `notes`; the finite windows are still given. A refusal at this stage, such as `method='exact'` without triples,
  keeps the result, records `{'status': 'refused', 'reason': ...}` and gives a `SpectraWarning`.
- **Saved files.** `save_npz` writes `spectra_*` arrays.
