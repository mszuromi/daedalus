"""
msrjd.integration.time_domain
=============================
Time-domain tree-level correlator evaluation.

This module evaluates Feynman diagrams directly in the time domain via
explicit numerical quadrature of vertex-time integrals.  The pipeline
works from:

  1. A list of typed diagrams (from the enumeration pipeline in
     ``msrjd.diagrams``).
  2. Their scalar prefactors (from ``classify_coefficient_factors``
     in ``msrjd.diagrams.symmetry``).
  3. The retarded propagator in pole-residue form: ``pole_vals``,
     ``C_mats``, ``D_delta`` (computed once from the kernel matrix
     ``K(ω)`` via eigenvalue decomposition).

No frequency-domain integral construction or loop-kernel grouping is
used.  The frequency-domain code in ``msrjd.integration.symbolic`` is
preserved for future extensions but is NOT called by this module.

Convention
----------
``total_C(t_1, t_2, ..., t_k)`` returns the tree-level correlator.
Position i is ALWAYS the time of ``external_fields[i]``:

  - ``external_fields[0]`` → ``t_1`` (base time, pinned to 0 for
    stationary systems)
  - ``external_fields[n]`` → ``t_{n+1}``,  ``τ_n = t_{n+1} - t_1``

This is enforced by the canonical time remapping in
``integrate_tree_diagram``.

Propagator decomposition
------------------------
Each retarded propagator entry is decomposed as:

    G_R[p, r](t) = c_δ · δ(t) + Θ(t) · G_smooth[p, r](t)

where ``G_smooth = Σ_k C_k[p, r] · exp(i ω_k t)`` is the pole-residue
sum.  The Ito convention ``Θ(0) = 1`` is used throughout.

For tree diagrams with ``|E|`` edges, the product of propagators is
expanded into ``2^|E|`` subsets (each edge either δ or smooth).  Deltas
with integration-variable arguments are integrated symbolically; any
deltas that survive as functions of external times only (e.g., δ(τ₁))
are reported but NOT added to the smooth callable ``total_C``.

Fourier convention (fixed pipeline-wide)
----------------------------------------
    G(t) = (1 / 2π) ∫ dω  exp(i ω t)  Ĝ(ω)

Poles with Im(ω) > 0 yield decaying exponentials for t > 0.

Public API
----------
- ``propagator_td.build_G_t_matrix``       — pole-residue + delta matrix
- ``propagator_td.G_t_entry``              — single propagator entry lookup
- ``propagator_td.G_t_delta_coeff``        — delta coefficient lookup
- ``final_integral.integrate_tree_diagram`` — tree-level vertex-time integrator
- ``pipeline.compute_correction_td``       — entry point / orchestrator
"""

from msrjd.integration.time_domain.propagator_td import (
    build_G_t_matrix,
    G_t_entry,
    G_t_delta_coeff,
)
from msrjd.integration.time_domain.final_integral import (
    integrate_tree_diagram,
    _loop_number_from_graph,
    format_td_integral_latex,
    eval_delta_contributions_on_tau_grid,
    eval_delta_contributions_on_2d_grid,
)
from msrjd.integration.time_domain.pipeline import (
    compute_correction_td,
)

__all__ = [
    'build_G_t_matrix',
    'G_t_entry',
    'G_t_delta_coeff',
    'integrate_tree_diagram',
    '_loop_number_from_graph',
    'format_td_integral_latex',
    'eval_delta_contributions_on_tau_grid',
    'eval_delta_contributions_on_2d_grid',
    'compute_correction_td',
]
