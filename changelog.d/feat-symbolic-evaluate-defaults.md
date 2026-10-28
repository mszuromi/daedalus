#### Symbolic output: `evaluate` takes the model's parameter defaults and keeps the saddle consistent

`MeanRelation.evaluate`, `OneLoopMean.evaluate` and `OneLoopSpectrum.evaluate` fall back on the model's declared
parameter defaults (`parameters=` of the builder call over them, available as `MeanRelation.defaults`). `values=`
overrides only the parameters it names, either as a whole list (`{'tau': [1, 3]}`) or as a single entry (`{'tau2':
3}`). A parameter with no default and no value still raises `SymbolicOutError('no value for [...]')`, and so does a
saddle when the relation was built without `mean_field=`.

A relation built with `mean_field='solve'` carries its saddle as a number, solved at the build parameters.
Previously, evaluating at other values of a parameter the saddle depends on silently mixed the new parameters with
the old saddle. Two examples, for `dx = (-x - g x^2 + h) dt + sqrt(2 D) dW` built at `g = 0.3`, `h = 0.8`:
- `values={'h': 1.5}` gave the default's `-0.0076531`; the correct value is `-0.0053571`;
- `values={'g': 0.5}` gave `-0.0090000`; the correct value is `-0.0096154`.

`MeanRelation.saddle_at(values)` now re-solves the saddle there, with the build's root choice. When the saddle moves,
`OneLoopMean` and `OneLoopSpectrum` evaluate the diagrams rebuilt at the new parameters and saddle (`.at(...)`,
cached). `MeanRelation.evaluate` takes arbitrary expressions and so cannot rebuild them; it refuses with a named
error. Further refusals:
- a change in the number of stable roots, which makes the root's identity ambiguous;
- a value that differs from one fixed at the build (`values=` or the saddle).

A parameter the saddle does not depend on, such as a noise strength, is evaluated as before. Calls that used to
raise for a missing parameter with a declared default now return its value.
