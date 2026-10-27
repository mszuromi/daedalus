#### Symbolic output: the one-loop spectrum, diagram by diagram

`dd.one_loop_spectrum(model)` (`daedalus.api.symbolic_spectrum`) writes the one-loop (`k = 2`, `ell = 1`) increment of
the spectrum `S_ab(omega)` (the convention of `dd.spectra`) as `S^(1)_ab = sum L_{a alpha}(omega) Sigma_{alpha
beta}(omega) L_{b beta}(-omega)`, with `L = G` for a response leg and `C0` for a physical leg. The self-energy
`Sigma` is read off the cubic and quartic vertices of the expanded action by Wick's theorem. It has three kinds of
diagrams (`SelfEnergyTerm`): tadpoles (a quartic vertex closed by `C0_pq(0)`), mean-shift insertions (a cubic vertex
on the one-loop mean `kappa^(1)_p` of `dd.one_loop_mean`) and bubbles (two cubic vertices, `(1/2 pi) int L1(nu)
L2(omega - nu) d nu`).

The noise block `Sigma_{xt xt}` and the response blocks come out separately, and the physical-physical block
vanishes by causality. The tree lines are kept as named functions (`G_dv1_vt1(omega)`, `C0_dv1_dv2(nu)`) or written
out from the symbolic propagator, each bubble as its held integral. The equal-time loops and mean shifts take the
four forms of `dd.one_loop_mean` (named, integral, rational, residues). Output is available as Sage expressions,
sympy and LaTeX, as the matrix `Sigma` and term by term. Nothing is simplified towards a closed form. `evaluate`
gives numbers: equal-time parts exactly, bubbles by quadrature in `nu`, and every line as a numerically normalized
ratio of polynomials, since the exact rationals of float inputs overflow double precision.

It reproduces the pipeline's exact one-loop spectrum (`dd.spectra`, state-space route) to about 1e-15 at several
frequencies for the following models:
- a 2-unit and a 3-unit quadratic rate network;
- a cross vertex `x y`, whose loop is off-diagonal;
- multiplicative noise `(s0 + s1 x) dW`, under both Itô and Stratonovich: a `(2, 2)` tadpole and `(2, 1)` bubbles;
- a cubic drift, giving a quartic tadpole;
- an exponential drift;
- spike trains with exponential kernels, with third-cumulant noise vertices and instantaneous lines.

Refused: spatial models, and a time derivative or kernel on a vertex. Also refused: a loop with no Itô value,
which is either a tadpole `C0_pq(0)` with a contact term, or a bubble decaying slower than `nu^-2`, where a spike
train's `delta(tau)` meets a line that jumps at the tie. The spike-reset example model hits the second case.
The tests of `dd.one_loop_mean` now import their shared helpers standalone. No existing number changes.
