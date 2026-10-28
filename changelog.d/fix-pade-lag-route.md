#### State-space route: stiff-looking segments of high-order Padé delays stay on the route at nonzero lags

A two-stage delayed negative feedback (`x -> y`, with `y(t - d)` repressing `x`, `d = 0.5`) at Padé order 16 failed
with `PhaseJResourceError` at any nonzero lag. `tau = 0` alone and order 8 both ran. The automatic order controller
hit the same refusal at order 16 and kept order 8.

The cause was the dressed lines: a propagator entry in series with the delay kernel. They are built from both
factors' realizations as given. Since the unit-free realization of v0.3.0, those come out with `b` and `c` graded in
opposite directions: `|b|` is about `1e16` on a propagator entry, and `|c|` about `5e7` on the Padé-16 kernel. The
cascade's own balancing does not undo this. The grading does no harm to the stationary states, but it made a moving
segment look stiff:
- its firing maps reached about `1e23`;
- the segment needed about `1e13` Taylor steps and was refused;
- the pole route that took over needed 1.9 million pole tuples, over the 6 GB guard.

A dressed line now also records its factors' own balancing, as a power-of-two scaling of its states. A segment that
would be refused as stiff is evaluated in that scaling instead. This is an exact similarity, `D^-1 G D` with `v / D`
in and `D y` out. The example's segments that took this path need at most about 180 Taylor steps, with no
fallback. Every segment that ran before takes the same path as before, so results that did not hit the refusal are
unchanged.

The one-loop term at `tau = 1` matches the same Padé-16 delay written as 16 auxiliary linear fields (no kernel, no
dressed line) to 2e-19 absolute (4e-14 relative). The route's counters gain `ss_segments_scaled`.
