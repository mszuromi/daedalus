#### Grid and numeric options accept Sage numbers

Under the Sage Jupyter kernel every literal is a Sage `Integer`, `Rational` or `RealNumber`, and the one-grid-rule
change (#70) rejected them: `tau_grid=(-6.0, 6.0, 13)` raised, which failed the example notebooks. The grid
resolver, `Config.accuracy` and the spectra options (`epsabs`, `epsrel`, `limit`) now read Sage numbers like
Python and numpy ones. The rule is unchanged: a tuple is a range, a list is literal, and an ambiguous 3-entry
integer list is refused.
