#### Paper-style SDE equations and per-source interpretations

- **`m.sde(text, noises=None)`** reads equations written as in a paper, one per line (or `;`), into the existing
  declarations (`api/sde_paper.py`): `dx = (-mu*x - eps*x^3) dt + sqrt(2*D) dW`, `tau*dv = (E - v) dt + s ∘ dW_a +
  J dN(r*exp(v))`. Each term ends in `dt` (drift), `dW` (standard Wiener, `Gaussian(var=1)`), `dN(rate)`
  (`Poisson(rate)`) or `d<X>_<name>` for a source passed in `noises=`; the factor in front is the coupling (`into=`).
  A subscript names a process, and one name in several equations is one source; an unnamed `dW` / `dN` is a source
  of its own equation (`W_x`, `N_x`). Fields not yet declared are declared. Malformed lines raise `SDEError` with a
  caret (no differential, a mark on the drift, `dN` without a rate, an unknown `d<X>_<name>`, the same process twice
  in one equation, a second `=`, ...).
- **Per-term interpretation.** A circle before the differential (`∘ dW`, ASCII `o dW`; `◇` / `⋄` read the same) makes
  a Gaussian term Stratonovich and a jump term Marcus; a plain term is Ito. One source takes one interpretation (a
  named process marked in one equation and not in another is refused). The model-wide `interpretation(...)` is
  refused together with paper-style terms, which carry their own.
- **`noise(..., interpretation='ito'|'stratonovich'|'marcus')`**: the same per source on the explicit API; it
  overrides the model-wide mode for that source. Mixed models build as the explicit model with the same per-source
  interpretations: only the Stratonovich sources get the drift correction, only the Marcus jump sources the Marcus
  noise term, and the model dict is stamped `interpretation: marcus` when any source takes the Marcus rule. When the
  per-source choices amount to a model-wide mode (all Gaussian terms marked: Stratonovich; all terms marked with a
  jump: Marcus), the model, `check()` and `latex()` are byte-identical to that mode's model. `check()` reports
  `interpretation: 'mixed'` plus `interpretations` per source otherwise; `latex()` marks each term by its own rule.
  Refused as before: Stratonovich on a jump source, Stratonovich/Marcus on exposed sources and SPDEs, Marcus on a
  coupling that reads other units (joint jumps). The simulator applies the rules per source.
- **`m.sde_text()`** writes the equations back in this form (its `noises` attribute holds the sources it names);
  `.sde(text, noises=text.noises)` on a model with the same parameters rebuilds the same model.
- Existing models are unchanged: the byte-identical snapshot of the SDE catalogue passes. Latex: a source named
  `W_x` for its own letter prints `W_{x}` (was `W_{W_{x}}`; no catalogue model has such a name).
- Tests: `tests/test_sde_paper.py` (paper lines vs explicit models by digest, Stratonovich/Marcus/mixed
  interpretations vs hand-corrected Ito actions, refusals with carets, `sde_text` round trips including catalogue
  models), and a README example in `tests/test_readme_snippets.py`.
