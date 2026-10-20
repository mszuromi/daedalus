#### Faster mean-field residual: rows compiled once, one error state per call

A mean-field solve evaluates its residual thousands of times: the multi-start hybr runs, the seed-scale search of
fix-final-mf-seeds, and the Newton refinement. Each call used to re-parse every row's equation text with
`eval(str)` (twice per row, once per side), enter a numpy error-state context per row, and unwrap each value through
`asarray(...).reshape(...)`.

Each row is now compiled once as `(lhs) - (rhs)`, together with its term-size form. The error state is entered once
per call, a scalar row is unwrapped directly, and `define_function` expressions are compiled once. The arithmetic is
the same (lhs, then rhs, then the difference), and a check on `ou_quartic_two_dim_color_corr` and
`dendritic_quad_soma_sigmoid` (k=2, one loop) gives bit-identical saddles and cumulants.

Load-independent measure: Python calls of one `analyze` of `ou_quartic_two_dim_color_corr` (cProfile, warm expand
cache) were 1,315,523 before fix-final-mf-seeds and 1,595,299 with it (+21 %). They are now 944,560: 28 % below
the earlier figure and 41 % below main. The residual call count is unchanged (21,778), since the scale search's
2,976 extra evaluations stay. Each call is now much cheaper: under load, that model's mean-field solves went from
10.8 s to 2.9 s.
