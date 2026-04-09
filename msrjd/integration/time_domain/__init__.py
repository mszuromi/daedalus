"""
msrjd.integration.time_domain
=============================
Phase J — hybrid loop-kernel reduction pipeline.

Frequency space is used for unique loop kernel identification and
deduplication (reusing Phase I's `group_diagrams_by_kernel` /
`loop_only_signature` machinery). Actual integration is performed in the
time domain, where causal exponential propagators reduce to polyhedral
exponential integrals that admit closed-form solutions.

MVP scope (tree-level only)
---------------------------
The initial build validates the Phase J *evaluation layer* only — it
handles typed diagrams with `loop_number == 0` by:

  1. reading the time-domain retarded propagator matrix G_R(t) from
     `propagator_data` via pole-residue reconstruction;
  2. assigning a symbolic time variable to every vertex, pinning one
     external leaf's time as the origin, and integrating over the
     remaining vertex times;
  3. summing the result across all tree-level kernel groups with their
     combined prefactors.

Loop kernel reduction, kernel caching, and parent-diagram contraction
(Phases 3-5 of the full hybrid pipeline) are intentionally deferred to
Extension 1 of the build plan; tree-level diagrams bypass those phases
entirely because they have no loop kernels to reduce.

Phase I (the frequency-domain residue backend in
`msrjd.integration.symbolic`) is untouched and remains the default /
fallback backend.

Fourier convention (fixed pipeline-wide)
----------------------------------------
    G(t) = (1 / 2π) ∫ dω  exp(i ω t)  Ĝ(ω)

Under this convention poles with Im(ω) > 0 yield decaying exponentials
for t > 0 and growing exponentials for t < 0. The pipeline's causality
filter (`msrjd.diagrams.causality`) guarantees that the kernel matrix's
pole values all have Im > 0 — the retarded propagator then follows by
multiplying the analytic pole-residue sum by `heaviside(t)`.

Public API
----------
- `propagator_td.build_G_t_matrix` — dict {'smooth', 'delta', 't_var'}
  with the pole-residue sum for the smooth part and the ω→∞ limits
  for the δ(t) coefficients.
- `propagator_td.G_t_entry`         — retarded edge propagator lookup
  (smooth part only; accepts either the new dict or a bare matrix)
- `propagator_td.G_t_delta_coeff`   — δ(t) coefficient lookup for
  an instantaneous response entry
- `subgraph.identify_loop_subgraphs` — loop subgraph identification (stub at MVP)
- `final_integral.integrate_tree_diagram` — vertex-time integration (tree only)
- `pipeline.compute_correction_td`  — Phase J entry point / orchestrator
"""

from msrjd.integration.time_domain.propagator_td import (
    build_G_t_matrix,
    G_t_entry,
    G_t_delta_coeff,
)
from msrjd.integration.time_domain.subgraph import (
    LoopSubgraph,
    identify_loop_subgraphs,
)
from msrjd.integration.time_domain.final_integral import (
    integrate_tree_diagram,
    format_td_integral_latex,
    eval_delta_contributions_on_tau_grid,
)
from msrjd.integration.time_domain.pipeline import (
    compute_correction_td,
)

__all__ = [
    'build_G_t_matrix',
    'G_t_entry',
    'G_t_delta_coeff',
    'LoopSubgraph',
    'identify_loop_subgraphs',
    'integrate_tree_diagram',
    'format_td_integral_latex',
    'eval_delta_contributions_on_tau_grid',
    'compute_correction_td',
]
