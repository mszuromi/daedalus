#### Symbolic output: the one-loop spectrum takes the Itô equal-time rule for spike trains

`dd.one_loop_spectrum` now covers lines with an instantaneous part, such as a spike train's `G` or contact term:
`L = L(inf) + L_r`, where `L(inf)` is the `omega -> infinity` limit and `L_r` decays.
- **Bubbles.** The `delta(tau)` of `L1(inf)` puts both vertices at one time. There the other line takes its Itô value:
  zero for a retarded `G` (`G(0) = 0`), and `C0_pq(0)` for `C0`. The bubble is
  `(1/2 pi) int L1_r(nu) L2(w - nu) d nu + L1(inf) L2(tau = 0)`, and the integral converges absolutely. In the
  `'named'` form the weights appear as the symbols `G_a_r_inf` / `C0_a_b_inf`.
- **Tadpoles.** A tadpole whose `C_pq(tau)` jumps at the tie takes its Itô value (`OneLoopMean.C0`).
- **Still refused:** a tadpole with a contact term (`delta(0)`), and a bubble of two lines that both have a
  `delta(tau)` part (`delta(tau)^2`).

Checked against the pipeline's one-loop spectrum to <= 1e-14 at three frequencies:
- `single_population_spike_reset_test`: the reset `v n` times a spike train;
- `dendritic_quad_soma_sigmoid`;
- a model whose second field has a white-noise part.

All three were refused before. No existing number changes.
