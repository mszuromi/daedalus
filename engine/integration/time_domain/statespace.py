r"""
State-space subset integrator for Phase J (S2; the default since S3)
====================================================================

Evaluates one Phase J subset integral

    I(f) = P · ∫_R  Π_{e ∈ E} g_e(Δt_e)  ds,      g_e(t) = c_e e^{a_e t} b_e  (t > 0),

without poles, residues or pole tuples.  Each smooth edge ``e`` carries the
minimal SISO realization ``(a_e, b_e, c_e)`` of ITS propagator entry
``G[pi_e, ri_e]`` (``api._realization``, S1: ``Realization.entries``), so its
state axis has ``n_e = entries[(pi, ri)].n_states`` components, never the
global state dimension.  ``R`` is the subset's region: every row is
``Δt_e = a_int·s + a_ext·f + c0 > 0`` with ``a_int`` one head (+1) and/or one
tail (-1) internal vertex; heads/tails without a vertex are the external
times ("constants", affine in the free external times ``f``).  Rows without
an internal vertex are scalar factors ``g_e(c_eff)`` decided by Phase J's own
Θ(0) / tie rule (``final_integral._const_row_verdict``).  ``P`` is the
subset prefactor.

The engine is chosen by ``DAEDALUS_PHASE_J_ENGINE`` (read for each subset
when a diagram is integrated): unset, ``default`` or ``statespace`` take this
route for every eligible subset (the default since S3); ``poles`` restores
the pole engine everywhere (every number as before S3, bit for bit: this
module is then not even imported by the dispatch); any other value raises
(at the first subset).  The route applies to every ``integrate_diagram``
call: the per-diagram path and grouped Phase J's per-diagram groups
(``api._grouped_phase_j``: a group the route covers is integrated per
diagram); the spatial certification's per-momentum calls are declined
(``realization_spatial_per_q``).  A subset is taken here when every row is
an edge with a realization (``subset_maker``): a propagator entry of the
model's realization, a conductance kernel (``api._realization.
realize_kernel``: its exact chain, equal rates included), or a dressed line
(the kernel cascaded with its propagator entry); no noise-kernel boxes, a
numeric prefactor, a Hurwitz realization, rows of the shape above, every
vertex followed by an external time.  Everything else stays on the pole
routes (counted by reason in ``ss_declined_by_reason``).

**Call-time fallback** (S3).  A routed subset whose evaluation is refused
late -- a stiff segment beyond the triangular route's size
(``TRI_DENSE_MAX`` states, ``TAYLOR_S_HARD`` steps), a segment system over
the memory budget or too large to build, a non-finite value -- is evaluated
by the pole route instead, for that call: the pole-route closure (and its
modesum plan) is built the first time it is needed (``fallback``, a lazy
builder passed by the dispatch), counted (``ss_fallback_evals``,
``ss_fallback_subsets``, ``ss_fallback_by_reason``) and reported
(``compute_cumulants``: ``result['phase_j_engine']``, analyzer G6).
``DAEDALUS_PHASE_J_SS_FALLBACK=0`` raises the refusal instead (S2).  A
subset with no pole route (a kernel only this route realizes) raises.

The downset sweep
-----------------
Time runs upward.  The state at time ``t`` is indexed by the set ``I`` of
internal vertices already placed ("fired") -- a downset of the subset's
causal poset -- and is a tensor with one axis per ACTIVE edge (tail passed,
head not):

    X_I(t) = ∫_{R_I(t)} Π_{e done} g_e(Δt_e) ⊗_{e active} e^{a_e (t - t_e^tail)} b_e  ds_I,

``R_I(t)``: the times of the vertices of ``I``, all ``< t``, consistent with
the rows among ``I`` and the external times already passed.

1. **Generator.**  Between external times ``X`` solves the linear system
   ``X_I' = M_I X_I + Σ_{v ∈ max I, v placeable} F_v X_{I \ v}`` with
   ``M_I = ⊕_{e active} a_e`` (a Kronecker sum, applied axis by axis, never
   formed) and the firing map ``F_v``: contract the axes of the edges with
   head ``v`` by ``c_e``, open axes for the edges with tail ``v`` by ``b_e``.
2. **Stationary start** (``t_min = -inf``): before the first external time
   the problem is translation invariant, so ``X_I`` is constant there and
   ``X_∅ = 1``, ``X_J = (-M_J)^{-1} Σ_v F_v X_{J \ v}`` (resolvents of
   Kronecker sums; with two active edges this is the Lyapunov equation).
   These states do not depend on the external times: computed once per
   subset and cached.  A finite ``final_integral.PHASE_J_T_MIN`` is a finite
   first segment from ``X = {∅: 1}`` instead (switch-on transients, below).
3. **External time** ``c``: contract the edges whose head is ``c``, open the
   edges whose tail is ``c``, drop the downsets that miss a vertex required
   before ``c``.  Equal external times form one event.
4. **Segment** ``(c_j, c_{j+1})``: ``X ← exp(G L) X``, ``L = c_{j+1} - c_j``,
   ``G`` block lower triangular over the downsets reachable in the segment
   (diagonal ``M_I``, sub-diagonal ``F_v``).
5. The answer is ``P · Π(scalar factors) · X_V`` (all vertices placed, no
   active edge) after the last external time.

**Proof sketch (the poset generalisation of Van Loan / Carbonell-Jiménez-
Pedroso).**  (a) Differentiating ``X_I(t)`` in its upper limit ``t`` gives
two terms: the open factors ``e^{a_e (t - t_tail)}`` give ``M_I X_I``; the
moving face ``max_{v∈I} s_v = t`` of ``R_I(t)`` splits (up to a null set)
by WHICH vertex is the latest, necessarily a maximal element ``v`` of ``I``;
on that face the edges into ``v`` complete with ``Δt = t - t_tail``, i.e.
``c_e^T`` applied to the open axis of ``X_{I∖v}``, and the edges out of ``v``
open with ``e^{a_e·0} b_e = b_e``: that is ``F_v X_{I∖v}(t)``, present only
while ``v``'s window (its external-time rows) admits ``t``.  So the stacked
state solves the linear ODE ``X' = G X`` on each segment, and the solution
is ``exp(G L) X``.  (b) ``G`` is block triangular with nilpotent
off-diagonal part (each firing adds a vertex), so the Dyson series of
``exp(G L)`` terminates; its block ``(J, I)`` is the sum over all saturated
chains ``I = I_0 ⊂ I_1 ⊂ ... ⊂ I_k = J`` of the nested integrals
``∫_{0<u_1<...<u_k<L} e^{M_J (L-u_k)} F ... F e^{M_I u_1}``.  A chain of the
downset lattice from ``∅`` to ``V`` is exactly a linear extension of the
poset, so every vertex order compatible with the rows is integrated once and
none is enumerated.  For a chain poset the lattice is a chain and this is
Van Loan's (1978) block-triangular exponential / Carbonell et al.'s (2008)
multiple integrals; with scalar edges it is Opitz's divided-difference
formula that Phase J's chain kernels evaluate per pole tuple (Phase J =
diagonalise every edge, then run this sweep once per pole tuple).  (c)
Before the first external time the region and integrand are invariant under
a common shift of all times, so ``X_I(t)`` is constant; the ODE then gives
``0 = M_I X_I + Σ F_v X_{I∖v}``; ``M_I`` is invertible because every
active edge is Hurwitz and a nonempty downset has at least one active edge
(its maximal vertex has an out-edge: ``subset_maker`` requires every vertex
to be followed by an external time).  Absolute convergence: every open
direction is damped by an active Hurwitz edge.  (d) At an external time the
rows with that head/tail change the integrand only by completing/opening
factors, and a vertex whose window ends there must already be placed.  ∎

Switch-on transients (item 44)
------------------------------
A finite ``t_min`` starts the fluctuations about the saddle at zero at
``t_min``: every internal vertex time is ``> t_min``, so the sweep starts at
``t_min`` from the empty state ``X = {∅: 1}`` (no vertex placed yet) and its
first segment, ``t_min`` to the first external time, is a finite
``exp(G L)`` over the downsets of the vertices that need no external time
first (``free0``) instead of the resolvents of the stationary start.  The
proof above holds unchanged on ``(t_min, ∞)``: (a), (b) and (d) are local in
time, and (c) is simply not used.  As ``t_min -> -inf`` the first segment's
state tends to the stationary one (every block is damped by a Hurwitz
Kronecker sum), so the stationary result is the limit.
``compute_cumulants(t_min=...)`` runs the whole Phase J stage, and every
later call of the result's callables, inside :func:`switch_on_scope`: a
thread-local bound that this route reads in place of
``final_integral.PHASE_J_T_MIN`` (left untouched), and this route as the
only one: a subset it declines, and a call-time refusal that would fall
back to the pole route, raise :class:`SwitchOnUnsupported` instead (the
pole engine's own finite-lower-bound paths are not used for these runs).
The stationary states are neither built nor charged against the memory
budget for these runs.

Numerics
--------
* **Resolvents** ``(-M)^{-1} r`` in the per-edge Schur basis (exact
  identity / reversal for triangular ``a_e``, e.g. Erlang chains; complex
  Schur otherwise): the Kronecker sum of upper-triangular factors is upper
  triangular, solved by a vectorised back substitution (Jacobi sweeps on the
  nilpotent strictly-upper part: exact after ``1 + Σ(n_e - 1)`` sweeps and
  operation-for-operation the back substitution), then mapped back and,
  when a unitary (complex Schur) factor was used, refined on the residual in
  the original basis, up to ``REFINE_STEPS`` times, keeping the best (S1: a
  plain Schur-basis solve loses up to ~3e-13 on graded realizations).  Divisions are by sums
  of eigenvalues of Hurwitz factors only: no eigenvalue gap enters.
* **Segments and impulses** ``exp(G L) x`` / ``c e^{a t} b`` by the
  Al-Mohy-Higham (2011) truncated-Taylor action with the trace shift: no
  squaring (measured here: scipy's ``expm`` of a 2 x 2 propagator block is
  2e-14 / 9e-14 off relative to its norm at ``t = 3 / 30``; the action
  4e-16 / 1.5e-15).  The generator is dense up to ``OP_DENSE_MAX`` and
  matrix-free beyond (per-block axis products and einsum firings; the
  ``||A^p||^{1/p}`` of the step selection from ``onenormest`` with the
  adjoint map).  The action's error grows like (steps) x eps, so a STIFF
  segment (fast rates over a long gap) is exponentiated instead in the
  per-edge triangular basis with the blocks in reverse order, where the
  generator is upper triangular, by the reset-diagonal triangular Padé
  (``api._realization.expm_upper_triangular``: exact ``e^{lambda L}`` on the
  diagonal); with an exact basis (scalar / triangular entries, 2 x 2 Schur
  forms) from ``TAYLOR_S_MAX`` steps, with a Schur basis of a larger
  (graded) entry only from ``TAYLOR_S_MAX_SCHUR``, since such a basis costs
  accuracy itself (S1).  Beyond ``TRI_DENSE_MAX`` states a segment needing
  more than ``TAYLOR_S_HARD`` steps raises.
* **Per-entry balancing** (``_balance_siso``): of the entry as given, ``a``
  balanced alone, and ``[[a, b], [c, 0]]`` balanced, each followed by an
  exact ``b``/``c`` equalisation, the one with the smallest
  ``max(|a|_1, |b|_1 |c|_inf)``.
* **τ batches** (``SubsetSS.evaluate_many``, ``tau_grid_prefetch``): points
  that share the order of the external times reuse the stationary state and
  the first event; the last segment is stepped from one length to the next.

C0 output (S4).  On a line of the external times (k = 2: the lag on one
side of 0; a k >= 3 slice whose swept leg is first or last) the order of the
external times is fixed and exactly one segment moves, so a subset is one
matrix exponential ``c e^{G s} x0`` there: ``SubsetSS.me_line`` returns the
moving segment, the state entering it and the functional of everything after
it.  ``api.me_output`` sums these over subsets, Wick permutations and
diagrams with the closures' own weights (``_contrib.ss_prefactor``; the
contribution's ``phase_j_parts``) into ``Cumulant.me``; nothing here changes
a value.  On a finite chamber where several segments move (a k >= 3 slice
whose swept leg lies between two legs) ``SubsetSS.me_chain`` returns the
moving segments and the dense maps between them (item 54).

Error estimate (analyzer G6; S3).  Every evaluation adds to the counter
``ss_err_abs_sum`` the estimate ``|P s| M (1e-15 (1 + n_steps) + rho)``:
``M`` = the final contraction taken in absolute values (``|c| . |X|``: the
magnitude the last sum cancels from), ``n_steps`` = the Taylor scaling steps
of the evaluation (the action's tolerance is ``2^-53`` per step, norm-wise),
``rho`` = the largest relative residual of the subset's stationary
resolvent solves after refinement, ``s`` = the constant rows' factors, ``P``
the prefactor; ``ss_estimated`` counts the evaluations.  Like the closed-form
routes' ``1e-15 * sum|term|`` it is an estimate, not a bound.

Memory: before a subset's stationary states are allocated their size (the
product of the per-edge dimensions over each downset's active edges, summed)
is charged against ``PHASE_J_MEM_BUDGET_GB`` through item 35's running total
(``final_integral.phase_j_memory_scope``); over budget raises
``PhaseJResourceError`` before anything is allocated.  A segment system is
checked the same way (its working set alone) when it is first built, at the
first evaluation that needs it.  Not charged: the module's caches that
outlive a run (dense segment generators and the small resolvent blocks'
dense operators, one LRU capped at ``DENSE_CACHE_BYTES``; the last
``_REAL_CACHE_MAX`` realizations).
"""
from __future__ import annotations

import itertools
import math
import os
import threading
from collections import OrderedDict

import numpy as np
import scipy.linalg as sl

__all__ = [
    'ENGINE_ENV', 'FALLBACK_ENV', 'engine_requested', 'engine_name',
    'subset_maker', 'SubsetSS', 'realization_for', 'tau_grid_prefetch',
    'SSIneligible', 'SSLateRefusal', 'SSNoPoleFallback', 'model_route',
    'diagram_route',
    'census', 'switch_on_scope', 'switch_on_active', 'SwitchOnUnsupported',
]

ENGINE_ENV = 'DAEDALUS_PHASE_J_ENGINE'
FALLBACK_ENV = 'DAEDALUS_PHASE_J_SS_FALLBACK'

#: dense generator (Taylor action with dense mat-vecs) up to this dimension;
#: matrix-free beyond
OP_DENSE_MAX = 2000
#: Taylor scaling steps above which a segment (or an impulse) counts as
#: stiff: the action's error grows like (steps) x eps (measured: a 1e4 rate
#: over tau = 50, 4.9e-12); stiff segments use the triangular route
TAYLOR_S_MAX = 50
#: ... when a unitary (complex Schur) basis of an entry with n >= 3 states is
#: involved: S1 measured that a Schur basis of a graded realization (Schwarz
#: / companion blocks of rational kernels) costs accuracy (Pade-24: 3e-13
#: through this route vs 5e-14 by the Taylor action at a few hundred steps),
#: so the triangular route is taken only when the Taylor action would need
#: far more steps (a 2 x 2 Schur form is one rotation of a balanced block)
TAYLOR_S_MAX_SCHUR = 1000
#: an impulse ``c e^{a t} b`` of an entry with an exact triangular basis
#: (triangular ``a``, a 2 x 2 Schur form; n >= 2) takes the triangular route
#: beyond this many Taylor steps: a graded entry (a fast line cascaded with
#: a slow kernel: a dressed line) loses up to ~1e-14 in the trace-shifted
#: action well before ``TAYLOR_S_MAX`` (S3 review: 4e-15 already at 2
#: steps), the triangular Pade ~3e-16, at a negligible cost for these small
#: entries
TAYLOR_S_MAX_IMPULSE = 0
#: a segment with exact triangular bases whose slowest diagonal rate is below
#: ``GRADED_RATIO`` of the mean (graded: the trace shift leaves the slow modes
#: growing fast) takes the triangular route beyond ``TAYLOR_S_MAX_GRADED``
#: steps (S3 review)
GRADED_RATIO = 0.1
TAYLOR_S_MAX_GRADED = 2
#: the triangular route (dense upper-triangular generator in the per-edge
#: Schur basis, reset-diagonal Pade) up to this many states
TRI_DENSE_MAX = 600
#: beyond the triangular route's size, a segment needing more Taylor steps
#: than this is refused with an error rather than computed inaccurately
TAYLOR_S_HARD = 2000
#: refinement steps of a resolvent solve done in a unitary Schur basis
REFINE_STEPS = 2
#: a resolvent block up to this many entries is solved by one dense
#: back substitution (``scipy.linalg.solve_triangular`` on the cached
#: upper-triangular Kronecker sum in the per-edge triangular basis: the
#: same back substitution the Jacobi sweeps perform, in one call); larger
#: blocks keep the matrix-free sweeps (S3, walls)
DENSE_SOLVE_MAX = 256
#: a subset with more stationary downsets than this is declined
MAX_DOWNSETS = 50000
#: bytes of dense segment generators kept across calls (LRU)
DENSE_CACHE_BYTES = 256 * 2 ** 20
#: at most this many smooth edges (einsum labels)
MAX_EDGES = 50


#: Every reason the state-space route gives for not taking a model, a diagram
#: or a subset (``ss_declined_by_reason`` / ``run_route``'s ``reason``), or
#: for refusing an evaluation at call time (``ss_fallback_by_reason``), with
#: what it means.  The pole engine evaluates whatever is declined.  Names
#: starting with a ``DECLINE_PREFIXES`` key are families: ``realization_`` +
#: an ``api._realization.RealizationRefused`` reason (or ``error``),
#: ``kernel_`` + a kernel's refusal reason, ``numerical_`` + an exception
#: type.  ``tests/test_ss_decline_reasons.py`` checks that every reason the
#: code can produce is named here (audit W7, RED-01 stage 1).
DECLINE_REASONS = {
    # the run / the model (``model_route``, ``realization_for``)
    'engine_poles': "DAEDALUS_PHASE_J_ENGINE=poles",
    'realization_spatial_per_q': "the spatial certification's per-momentum "
                                 "temporal calls (the spatial reference "
                                 "stays on the pole routes)",
    'not_hurwitz': 'the realization is not certified Hurwitz',
    # a diagram / a vertex type (``diagram_route``, ``run_route``)
    'noise_kernel_box': 'a noise source with a cumulant kernel (a '
                        "'noise_box' row)",
    # a subset (``subset_maker``)
    'non_propagator_rows': 'a smooth row that is neither a propagator entry, '
                           'a conductance kernel nor a dressed line',
    'symbolic_prefactor': 'the prefactor is not a number',
    'non_edge_rows': "a row kind other than 'edge' / 'conv_pseudo' (noise "
                     "boxes, convolution tau rows)",
    'row_count': 'the constraint rows and the smooth edges do not match',
    'entry_missing': 'the realization has no entry for a propagator row',
    'kernel_direct_term': 'a conductance kernel with a direct (delta) term '
                          'on a kernel edge (dressed lines take it)',
    'dressed_line_without_kernel': 'a dressed-line row without its kernel',
    'stationary_over_budget': 'the stationary states do not fit the memory '
                              'budget (and the pole plan does)',
    # a subset's shape (``SSIneligible``)
    'too_many_edges': f'more than MAX_EDGES ({MAX_EDGES}) smooth edges',
    'row_shape': 'a row that is not a difference of two times',
    'offset_row': 'a row with an external-time offset the route does not '
                  'take',
    'vertex_after_last_time': 'an internal vertex after the last external '
                              'time',
    'no_external_time': 'a subset without an external time',
    'too_many_downsets': f'more than MAX_DOWNSETS ({MAX_DOWNSETS}) '
                         'stationary downsets',
    'finite_t_min': 'a finite PHASE_J_T_MIN where only the stationary '
                    'start is implemented',
    'row_crosses_zero': 'a constant row that changes sign on the evaluated '
                        'line',
    'several_lags': 'a C0 matrix-exponential output needs one moving '
                    'segment',
    # call time (``SSLateRefusal``; the dispatch falls back, counted)
    'stiff_segment': 'a stiff segment beyond the triangular route',
    'segment_too_large': 'a segment program too large to build',
    'segment_memory': 'a segment over the memory budget '
                      '(PhaseJResourceError)',
    'non_finite': 'a non-finite value',
}
DECLINE_PREFIXES = {
    'realization_': 'the propagator has no supported realization '
                    '(api._realization.RealizationRefused.reason, or error)',
    'kernel_': "a conductance kernel the route cannot realize (its "
               "RealizationRefused reason, 'kernel_'-prefixed)",
    'numerical_': 'a numerical exception at call time (its type name)',
}


def decline_reason_known(reason):
    """True when ``reason`` is in ``DECLINE_REASONS`` or a
    ``DECLINE_PREFIXES`` family."""
    return reason in DECLINE_REASONS or any(
        str(reason).startswith(p) for p in DECLINE_PREFIXES)


class SSIneligible(Exception):
    """A subset (or model) the state-space route does not take; ``reason``
    is the counter key."""

    def __init__(self, reason, detail=''):
        self.reason = reason
        super().__init__(f'{reason}: {detail}' if detail else reason)


class SSLateRefusal(RuntimeError):
    """A routed subset's evaluation refused at call time (``reason``: the
    ``ss_fallback_by_reason`` key); the dispatch falls back to the pole
    route for that call (``FALLBACK_ENV``)."""

    def __init__(self, reason, detail=''):
        self.reason = reason
        super().__init__(f'state-space route: {detail or reason}; set '
                         f'{ENGINE_ENV}=poles to use the pole routes')


class SSNoPoleFallback(RuntimeError):
    """A routed subset was refused at call time and the pole route cannot
    take it: the subset has none (a kernel only this route realizes), or
    ``compute_cumulants`` skipped the pole-cluster deformation for the run
    (``ss_no_pole_fallback``).  Never silent: it raises."""


class SwitchOnUnsupported(ValueError):
    """A switch-on run (``compute_cumulants(t_min=...)``,
    :func:`switch_on_scope`) met a subset this route does not take, or one it
    refused at call time: such runs are computed on the state-space route
    only (item 44), so it raises instead of using the pole route."""

    def __init__(self, reason, detail=''):
        self.reason = reason
        self.detail = detail
        super().__init__(
            f'switch-on transients (a finite t_min) are computed on the '
            f'state-space route only, and it cannot take a subset of this '
            f'run ({reason}{": " + str(detail) if detail else ""}); the pole '
            f'engine is not used for them.  Models with noise-kernel sources, '
            f'non-rational kernels or index > 1 descriptors are not covered '
            f'yet; drop t_min for the stationary result.')


# switch-on scope (item 44): the bound of the current thread (None: none)
_SO_TL = threading.local()


def switch_on_active():
    """True inside :func:`switch_on_scope` (in this thread)."""
    return getattr(_SO_TL, 't_min', None) is not None


def _switch_on_t_min():
    """The current thread's switch-on bound, or ``None``."""
    return getattr(_SO_TL, 't_min', None)


class switch_on_scope:
    """``with switch_on_scope(t_min): ...`` -- every Phase J evaluation in
    the block, in this thread, starts the fluctuations at zero at the
    absolute time ``t_min`` on this route only (:class:`SwitchOnUnsupported`
    otherwise).  The bound is thread-local and read by this route's
    ``_t_min_frame``; ``final_integral.PHASE_J_T_MIN`` itself is not touched
    (the pole routes never see the bound, and a stationary evaluation in
    another thread is unaffected).  Nests with the same ``t_min``; a
    different ``t_min`` inside an active scope raises ``ValueError``."""

    def __init__(self, t_min):
        t = float(t_min)
        if not math.isfinite(t):
            raise ValueError(f'switch_on_scope: t_min must be finite, got '
                             f'{t_min!r}')
        if not engine_requested():
            raise SwitchOnUnsupported(
                'engine_poles', f'{ENGINE_ENV}=poles selects the pole engine')
        self.t_min = t
        self._prev = None

    def __enter__(self):
        prev = _switch_on_t_min()
        if prev is not None and prev != self.t_min:
            raise ValueError(
                f'switch_on_scope({self.t_min!r}) inside an active switch-on '
                f'scope at t_min = {prev!r}: one switch-on time per thread at '
                f'a time')
        self._prev = prev
        _SO_TL.t_min = self.t_min
        return self

    def __exit__(self, *exc):
        _SO_TL.t_min = self._prev
        return False


class _Refused:
    """A prefetched grid point the route refused (``reason``, ``detail``)."""
    __slots__ = ('reason', 'detail')

    def __init__(self, reason, detail=''):
        self.reason = reason
        self.detail = detail


def engine_requested(environ=None):
    """True when ``DAEDALUS_PHASE_J_ENGINE`` selects the state-space route:
    unset, ``default`` or ``statespace`` (the default since S3); ``poles``:
    False (the pole engine everywhere).  Any other value raises (the one
    parser: ``engine._env.phase_j_engine``)."""
    from engine import _env
    return _env.phase_j_engine(environ) == 'statespace'


def engine_name(environ=None):
    """'statespace' or 'poles' (``engine_requested``)."""
    return 'statespace' if engine_requested(environ) else 'poles'


def fallback_enabled(environ=None):
    """``DAEDALUS_PHASE_J_SS_FALLBACK``: on (default) a late refusal falls
    back to the pole route; off (``0`` / ``off`` / ``false`` / ``no``): it
    raises (S2's behaviour).  Any other value raises."""
    from engine import _env
    return _env.flag(FALLBACK_ENV, True, environ)


# ─── counters (in final_integral._RUNTIME_COUNTERS, created on import) ────

_FI_MOD = []


def _FI():
    if not _FI_MOD:
        from engine.integration.time_domain import final_integral as FI
        _FI_MOD.append(FI)
    return _FI_MOD[0]


_COUNTER_KEYS = (
    'ss_subsets_built', 'ss_subsets_declined', 'ss_evals',
    'ss_stationary_builds', 'ss_resolvent_solves', 'ss_refinements',
    'ss_segments_taylor', 'ss_taylor_matvecs', 'ss_segments_triangular',
    'ss_batch_points', 'ss_prefetch_hits',
    'ss_realizations_built', 'ss_realization_refused',
    'ss_fallback_evals', 'ss_fallback_subsets', 'ss_kernel_entries',
    'ss_dressed_entries', 'ss_dressed_reduced', 'ss_estimated',
    'ss_err_abs_sum',
)
_COUNTER_DICTS = ('ss_declined_by_reason', 'ss_fallback_by_reason')


def _counters():
    C = _FI()._RUNTIME_COUNTERS
    if 'ss_fallback_evals' not in C:
        for k_ in _COUNTER_KEYS:
            C.setdefault(k_, 0)
        for k_ in _COUNTER_DICTS:
            C.setdefault(k_, {})
    return C


def census(counters=None):
    """The engine census of the counters (``final_integral.
    _RUNTIME_COUNTERS`` or a run's delta): subsets built on the route,
    declined (by reason), fallbacks (subsets / evaluations, by reason)."""
    C = _counters() if counters is None else counters
    return {
        'engine': engine_name(),
        'ss_subsets': int(C.get('ss_subsets_built', 0)),
        'ss_declined': int(C.get('ss_subsets_declined', 0)),
        'ss_declined_by_reason': dict(C.get('ss_declined_by_reason') or {}),
        'ss_evals': int(C.get('ss_evals', 0)),
        'fallback_subsets': int(C.get('ss_fallback_subsets', 0)),
        'fallback_evals': int(C.get('ss_fallback_evals', 0)),
        'fallback_by_reason': dict(C.get('ss_fallback_by_reason') or {}),
    }


def _count(key, n=1):
    # (``_reset_runtime_counters`` clears the dict in place: its identity
    # is stable)
    C = _FI()._RUNTIME_COUNTERS
    try:
        C[key] += n
    except KeyError:
        C = _counters()
        C[key] = C.get(key, 0) + n


def _decline(reason):
    if switch_on_active():
        raise SwitchOnUnsupported(reason, 'declined by the state-space route')
    C = _counters()
    C['ss_subsets_declined'] = C.get('ss_subsets_declined', 0) + 1
    d = C.setdefault('ss_declined_by_reason', {})
    d[reason] = d.get(reason, 0) + 1
    return None


# ─── realization cache ────────────────────────────────────────────────────

_REAL_LOCK = threading.RLock()
_REAL_CACHE = OrderedDict()
_REAL_LAST = [None]
_REAL_CACHE_MAX = 16


def _exact_key(v):
    """A key that separates any two different numeric values (the exact
    binary value; Sage ``RealNumber`` reprs print only ~15 digits)."""
    try:
        z = complex(v)
        return (z.real.hex(), z.imag.hex())
    except (TypeError, ValueError, OverflowError):
        return ('repr', repr(v))


def _np_key(num_params):
    if not num_params:
        return ()
    try:
        return tuple(sorted((str(k), _exact_key(v))
                            for k, v in num_params.items()))
    except Exception:
        return ('id', id(num_params))


def realization_for(propagator_data, num_params):
    """``(_ModelOps, None)`` or ``(None, reason)`` for this propagator at
    ``num_params``.  Cached by the identity of ``K_ft`` (shared by the
    pole-cluster deformation copies) and the parameter values; the cache
    holds strong references, so an id cannot be recycled while cached."""
    K = propagator_data.get('K_ft') if hasattr(propagator_data, 'get') else None
    om = propagator_data.get('omega') if K is not None else None
    # identity fast path (S3, walls: one lookup per subset): the same K_ft,
    # omega, num_params dict and value objects as the last call (strong
    # references are held, so no id can be recycled while cached)
    vals = tuple(num_params.values()) if isinstance(num_params, dict) else None
    last = _REAL_LAST[0]
    if (last is not None and vals is not None and last[0] is K
            and last[1] is om and last[2] is num_params
            and len(last[3]) == len(vals)
            and all(a is b for a, b in zip(last[3], vals))):
        return last[4], last[5]
    npk = _np_key(num_params)
    key = (id(K), id(om) if K is not None else None, npk)
    # content key (S3, walls): every compute_cumulants call builds a new
    # K_ft; the same propagator text, omega and exact parameter values are
    # the same realization (S1 is a function of exactly these)
    ckey = None
    with _REAL_LOCK:
        hit = _REAL_CACHE.get(key)
        if hit is not None and hit[0] is K:
            _REAL_CACHE.move_to_end(key)
            if vals is not None:
                _REAL_LAST[0] = (K, om, num_params, vals, hit[1], hit[2])
            return hit[1], hit[2]
    ck = _content_key(K, om)
    if ck is not None:
        ckey = ('content', ck, npk)
        with _REAL_LOCK:
            hit = _REAL_CACHE.get(ckey)
            if hit is not None:
                _REAL_CACHE.move_to_end(ckey)
                _REAL_CACHE[key] = (K, hit[1], hit[2])
                if vals is not None:
                    _REAL_LAST[0] = (K, om, num_params, vals, hit[1], hit[2])
                return hit[1], hit[2]
    from api._realization import try_realize
    if any(str(k_).startswith('Laplacian') for k_ in (num_params or {})):
        # the spatial certification's per-momentum temporal calls
        # (``pipeline_bridge.pipeline_C_q_tau``): the spatial engine's
        # reference stays on the pole routes
        ops, why = None, 'realization_spatial_per_q'
        _count('ss_realization_refused')
    else:
        try:
            real, exc = try_realize(propagator_data, num_params or {})
        except Exception as e:                  # never let S1 break a run
            real, exc = None, e
        if real is None:
            reason = getattr(exc, 'reason', None) or 'error'
            ops, why = None, f'realization_{reason}'
            _count('ss_realization_refused')
        elif not (real.hurwitz is True or (real.hurwitz is None and
                                           (real.lyapunov_margin or 0) > 0)):
            ops, why = None, 'not_hurwitz'
            _count('ss_realization_refused')
        else:
            ops, why = _ModelOps(real, num_params), None
            _count('ss_realizations_built')
    with _REAL_LOCK:
        _REAL_CACHE[key] = (K, ops, why)
        _REAL_CACHE.move_to_end(key)
        if ckey is not None:
            _REAL_CACHE[ckey] = (None, ops, why)
        if vals is not None:
            _REAL_LAST[0] = (K, om, num_params, vals, ops, why)
        while len(_REAL_CACHE) > _REAL_CACHE_MAX:
            _REAL_CACHE.popitem(last=False)
    return ops, why


_CONTENT = OrderedDict()


def _content_key(K, om):
    """The text of ``K_ft`` (every entry) and of ``omega``, memoized by the
    identity of ``K`` (strong reference held); ``None`` when it cannot be
    formed."""
    if K is None:
        return None
    with _REAL_LOCK:
        hit = _CONTENT.get(id(K))
        if hit is not None and hit[0] is K:
            return hit[1]
    try:
        # the exact values of float literals (a RealNumber prints ~15
        # digits): S1's exactification, then the text
        from api._realization import _exactify
        from sage.all import SR as _SR
        ck = (str(om), int(K.nrows()), int(K.ncols()),
              tuple(str(_exactify(_SR(x))) for x in K.list()))
    except Exception:
        ck = None
    with _REAL_LOCK:
        _CONTENT[id(K)] = (K, ck)
        while len(_CONTENT) > 2 * _REAL_CACHE_MAX:
            _CONTENT.popitem(last=False)
    return ck


def _clear_caches():
    """Drop the realization and dense-generator caches (tests)."""
    with _REAL_LOCK:
        _REAL_CACHE.clear()
        _CONTENT.clear()
        _REAL_LAST[0] = None
    with _DENSE_LOCK:
        _DENSE_CACHE.clear()
        _DENSE_BYTES[0] = 0


# ─── per-entry operators ──────────────────────────────────────────────────

def _mode(x, M, ax):
    """``M`` applied along axis ``ax`` of tensor ``x``."""
    return np.moveaxis(np.tensordot(M, x, axes=(1, ax)), 0, ax)


def _balance_siso(a, b, c):
    """Exact power-of-two similarity balancing of the augmented matrix
    ``[[a, b], [c, 0]]`` (``scipy.linalg.matrix_balance``, no permutation):
    ``a -> S^{-1} a S``, ``b -> S^{-1} b s``, ``c -> c S / s``, so
    ``c e^{a t} b`` is unchanged bit for bit in exact arithmetic and the
    rounding of ``a`` is untouched (powers of two).  S1 balances ``a`` alone;
    rational kernels then leave ``b`` and ``c`` graded in opposite directions
    (|b| ~ 1e20, |c| ~ 1e-20 on a Pade-16 entry), which inflates the norms of
    the firing maps and with them the number of Taylor steps.  Candidates:
    the entry as given, ``a`` balanced alone, and the augmented balancing,
    each followed by a power-of-two ``b``/``c`` equalisation; the one with the
    smallest ``max(|a|_1, |b|_1 |c|_inf)`` is kept.  A diagonal similarity
    keeps a triangular ``a`` triangular."""
    n = a.shape[0]
    aug = np.zeros((n + 1, n + 1), dtype=a.dtype)
    aug[:n, :n] = a
    aug[:n, n] = b
    aug[n, :n] = c
    if not np.all(np.isfinite(aug)):
        return a, b, c
    cands = [_equalize_bc(a, b, c)]
    _ba, (sa, _pa) = sl.matrix_balance(a, permute=False, separate=True)
    sa = np.asarray(sa, dtype=float)
    if not np.all(sa == 1.0):                  # a alone (S1's choice)
        cands.append(_equalize_bc(a * (sa[None, :] / sa[:, None]), b / sa,
                                  c * sa))
    _bal, (s, _p) = sl.matrix_balance(aug, permute=False, separate=True)
    s = np.asarray(s, dtype=float)
    if not np.all(s == 1.0):
        S, sn = s[:n], s[n]
        cands.append(_equalize_bc(a * (S[None, :] / S[:, None]),
                                  b * (sn / S), c * (S / sn)))

    # the augmented balancing can inflate ||a|| on a graded entry (3.9e3 vs
    # 4.9 measured): keep the candidate whose generator / firing norms are
    # smaller (both enter ||G|| and the Taylor step count)
    def cost(abc):
        a_, b_, c_ = abc
        return max(float(np.abs(a_).sum(axis=0).max()),
                   float(np.abs(b_).sum()) * float(np.abs(c_).max()))
    return min(cands, key=cost)


def _equalize_bc(a, b, c):
    """``b -> b σ``, ``c -> c / σ`` with ``σ`` the power of two closest to
    ``sqrt(|c|_inf / |b|_1)`` (exact, ``c e^{at} b`` unchanged)."""
    nb, nc = float(np.abs(b).sum()), float(np.abs(c).max())
    if not (nb > 0 and nc > 0):
        return a, b, c
    sig = 2.0 ** round(0.5 * math.log2(nc / nb))
    return a, b * sig, c / sig


class _EntryOps:
    """One propagator entry ``g(t) = c e^{a t} b``: the realization and its
    triangular (Schur) basis.  ``kind``: 'zero' (no smooth part), 'scalar'
    (n = 1), 'upper' / 'lower' (triangular ``a``: the basis change is the
    identity / the index reversal, exact), 'schur' (complex Schur, unitary).
    """
    __slots__ = ('n', 'a', 'b', 'c', 'kind', 'T', 'U', 'Uh', 'dT', 'Nup',
                 'trace', 'norm1', 'cnorm', 'bnorm', 'lam', '_tay', 'bt',
                 'ct')

    def __init__(self, ent, dtype):
        self.n = n = int(ent.n_states)
        self.a = np.asarray(ent.a, dtype=dtype)
        self.b = np.asarray(ent.b, dtype=dtype).reshape(n)
        self.c = np.asarray(ent.c, dtype=dtype).reshape(n)
        if n:
            self.a, self.b, self.c = _balance_siso(self.a, self.b, self.c)
        self.trace = complex(np.trace(self.a)) if n else 0j
        self.norm1 = float(np.abs(self.a).sum(axis=0).max()) if n else 0.0
        self.cnorm = float(np.abs(self.c).max()) if n else 0.0
        self.bnorm = float(np.abs(self.b).sum()) if n else 0.0
        self.U = self.Uh = None
        self.lam = None
        self._tay = None
        if n == 0:
            self.kind = 'zero'
            self.T = self.dT = self.Nup = self.bt = self.ct = None
            return
        if n == 1:
            self.kind = 'scalar'
            self.lam = self.a[0, 0]
            self.T = self.a
            self.dT = self.a.diagonal().copy()
            self.Nup = None
            self.bt, self.ct = self.b, self.c
            return
        a = self.a
        if not np.any(np.tril(a, -1)):
            self.kind, T = 'upper', a
        elif not np.any(np.triu(a, 1)):
            self.kind, T = 'lower', a[::-1, ::-1]
        else:
            self.kind = 'schur'
            T, U = sl.schur(np.asarray(a, dtype=complex), output='complex')
            self.U, self.Uh = U, U.conj().T
        self.T = T
        self.dT = T.diagonal().copy()
        self.Nup = np.triu(T, 1)
        # b, c in the triangular basis (opening / contracting there)
        self.bt = self.to_basis(self.b, 0)
        self.ct = (self.c @ self.U if self.kind == 'schur'
                   else self.c[::-1] if self.kind == 'lower' else self.c)

    def to_basis(self, x, ax):
        if self.kind == 'lower':
            return np.flip(x, axis=ax)
        if self.kind == 'schur':
            return _mode(x, self.Uh, ax)
        return x

    def from_basis(self, x, ax):
        if self.kind == 'lower':
            return np.flip(x, axis=ax)
        if self.kind == 'schur':
            return _mode(x, self.U, ax)
        return x

    def impulse(self, t):
        """``c e^{a t} b`` (``t >= 0``) by the trace-shifted Taylor action
        (scipy's scaling-and-squaring ``expm`` loses up to ~1e-13 relative
        on decaying exponentials; the action does not square)."""
        if self.n == 0 or t < 0:
            return 0.0
        if self._tay is None:
            mu = self.trace / self.n
            if not np.iscomplexobj(self.a):
                mu = float(np.real(mu))
            A = self.a - mu * np.eye(self.n, dtype=self.a.dtype)
            self._tay = (A, mu, _onenorm_powers(A))
        A, mu, dn = self._tay
        t = float(t)
        m_, s_ = _taylor_params(dn, t)
        _est_steps(s_)
        if s_ > (TAYLOR_S_MAX_SCHUR if (self.kind == 'schur' and self.n > 2)
                 else TAYLOR_S_MAX_IMPULSE):
            # stiff (|a| t large): the reset-diagonal triangular Pade in the
            # triangular basis (exact e^{lambda t} on the diagonal)
            from api._realization import expm_upper_triangular
            v = self.ct @ (expm_upper_triangular(self.T, t) @ self.bt)
            return v.real if not np.iscomplexobj(self.a) else v
        return (self.c @ _taylor_action(lambda y: A @ y, mu, dn, self.b,
                                        t)).item()


def _att_key(att):
    """A key of a conductance kernel attachment's image (its symbol and its
    time / frequency expression)."""
    out = [str(att.get('symbol'))]
    for nm in ('kernel_td_fn', 'kernel_ft_fn'):
        f = att.get(nm)
        if f is not None:
            out.append((nm, str(getattr(f, 'expr', f)),
                        str(getattr(f, 'placeholder', ''))))
    return tuple(out)


class _ModelOps:
    """The realization of one model at one parameter point, as per-entry
    operators, plus the resolvent data per active-edge pattern (shared by
    every subset of the model); also the conductance kernels' realizations
    and the dressed lines' cascades (S3), built on first use."""

    def __init__(self, real, num_params=None):
        self.real = real
        self.num_params = num_params
        self.is_real = not np.iscomplexobj(real.A)
        self.dtype = float if self.is_real else complex
        self.entries = {}
        for key, ent in real.entries.items():
            self.entries[key] = _EntryOps(ent, self.dtype)
        self._solve_data = {}
        self._extra = {}
        self._lock = threading.RLock()

    def entry(self, pi, ri):
        return self.entries.get((int(pi), int(ri)))

    def _kernel_real(self, att):
        """``(EntryRealization, None)`` or ``(None, reason)``."""
        key = ('kernel_real', _att_key(att))
        with self._lock:
            hit = self._extra.get(key)
        if hit is not None:
            return hit
        from api._realization import realize_kernel, RealizationRefused
        try:
            # item 56: a direct term (g = d delta + smooth) is realized too;
            # only a dressed line takes it (kernel_entry refuses it)
            ent, fname = realize_kernel(att, self.num_params, direct=True)
            out = ((None, 'kernel_complex') if fname != 'QQ' and self.is_real
                   else (ent, None))
        except RealizationRefused as exc:
            out = (None, exc.reason if exc.reason.startswith('kernel_')
                   else f'kernel_{exc.reason}')
        except Exception as exc:                # never let a kernel break a run
            out = (None, f'kernel_error_{type(exc).__name__}')
        with self._lock:
            self._extra[key] = out
        return out

    def kernel_entry(self, att):
        """The ``_EntryOps`` of a conductance kernel ``g`` (a kernel edge
        ``s -> anchor``), or a decline reason (str)."""
        key = ('kernel', _att_key(att))
        with self._lock:
            hit = self._extra.get(key)
        if hit is not None:
            return hit
        ent, why = self._kernel_real(att)
        if ent is not None and ent.d != 0:
            # a kernel edge s -> anchor is a smooth row: a direct term
            # (c delta: s = anchor) has no place in it (item 56: dressed
            # lines take such kernels)
            ent, why = None, 'kernel_direct_term'
        out = why if ent is None else _EntryOps(ent, self.dtype)
        if ent is not None:
            _count('ss_kernel_entries')
        with self._lock:
            self._extra[key] = out
        return out

    def kernel_route(self, att):
        """For the run-level predictions (``run_route``, ``diagram_route``,
        ``diagram_estimate``): the kernel's ``EntryRealization`` when the
        route takes it -- a strictly proper kernel as a kernel edge or a
        dressed line, one with a direct term (item 56) as a dressed line
        only (Phase J refuses it on a kernel edge by name) -- or a decline
        reason (str).  A direct term counts only where Phase J dresses it: a
        frequency image whose strictly proper part has separated modes
        (``final_integral._kernel_direct_split``); a local ``c delta`` kernel
        or a time image ``c delta + smooth`` stays declined, as before."""
        key = ('kernel_route', _att_key(att))
        with self._lock:
            hit = self._extra.get(key)
        if hit is not None:
            return hit
        ent, why = self._kernel_real(att)
        out = why if ent is None else ent
        if ent is not None and ent.d != 0 and _FI()._kernel_direct_split(
                att, self.num_params) is None:
            out = 'kernel_direct_term'
        with self._lock:
            self._extra[key] = out
        return out

    def dressed_entry(self, pi, ri, att):
        """The ``_EntryOps`` of a dressed line ``(g * G)(t) = int_0^t
        g(t - x) G(x) dx``: the series connection of the propagator entry
        ``G[pi, ri]`` (``a1, b1, c1``, delta part ``d1``) and the kernel
        (``a2, b2, c2``, direct term ``d2``; item 56), ``A = [[a1, 0],
        [b2 c1, a2]]``, ``B = [b1; b2 d1]``, ``C = [d2 c1, c2]`` (smooth
        impulse response ``d2 G_smooth(t) + d1 g_smooth(t) + (g_smooth *
        G_smooth)(t)``; the delta part ``d1 d2`` is the dispatch's, as for
        every edge).  Or a reason.

        Item 61: the cascade is reduced by balanced truncation at the
        truncation tolerance in effect (``api._realization.truncation_tol``;
        ``0``, the exact setting: not reduced).  When the propagator cancels
        the kernel's poles (a closed loop through the vertex: one
        population, or a network whose ``G = (I - Lambda W K)^-1`` vanishes
        at them) the kernel block is uncontrollable and the minimal line
        has the entry's states only (n_G + n_K -> n_G); for a rank-1
        multi-population ``W`` nothing can be discarded and the cascade is
        returned as built.  The truncated line is stable (Hurwitz, checked)
        with the H-infinity bound ``2 sum(discarded HSV) <= tol sigma_1``,
        recorded per line in the realization's truncation record
        (``Realization.truncation['dressed']``)."""
        from api._realization import truncation_tol
        tol = truncation_tol()
        key = ('dress', int(pi), int(ri), _att_key(att), tol)
        with self._lock:
            hit = self._extra.get(key)
        if hit is not None:
            return hit
        ent_g = self.real.entries.get((int(pi), int(ri)))
        kre, why = self._kernel_real(att)
        if ent_g is None:
            out = 'entry_missing'
        elif kre is None:
            out = why
        else:
            dt = self.dtype
            a1 = np.asarray(ent_g.a, dtype=dt)
            b1 = np.asarray(ent_g.b, dtype=dt).reshape(-1)
            c1 = np.asarray(ent_g.c, dtype=dt).reshape(-1)
            d1 = complex(ent_g.d) if not self.is_real else float(
                np.real(ent_g.d))
            a2 = np.asarray(kre.a, dtype=dt)
            b2 = np.asarray(kre.b, dtype=dt).reshape(-1)
            c2 = np.asarray(kre.c, dtype=dt).reshape(-1)
            d2 = complex(kre.d) if not self.is_real else float(
                np.real(kre.d))
            n1, n2 = a1.shape[0], a2.shape[0]
            if (n2 == 0 and (d2 == 0 or n1 == 0)) or (n1 == 0 and d1 == 0):
                A = np.zeros((0, 0), dtype=dt)
                B = C = np.zeros(0, dtype=dt)
            else:
                A = np.zeros((n1 + n2, n1 + n2), dtype=dt)
                A[:n1, :n1] = a1
                A[n1:, :n1] = np.outer(b2, c1)
                A[n1:, n1:] = a2
                B = np.concatenate([b1, b2 * d1])
                C = np.concatenate([d2 * c1 if d2 != 0 else
                                    np.zeros(n1, dtype=dt), c2])
                if tol > 0 and A.shape[0] >= 2:
                    A, B, C = self._reduce_dressed(A, B, C, tol, pi, ri, att)
            import types as _types
            out = _EntryOps(_types.SimpleNamespace(
                a=A, b=B, c=C, n_states=A.shape[0]), dt)
            _count('ss_dressed_entries')
        with self._lock:
            self._extra[key] = out
        return out

    def _reduce_dressed(self, A, B, C, tol, pi, ri, att):
        """The dressed cascade ``(A, B, C)`` balanced-truncated at ``tol``
        (``dressed_entry``), its record kept in the realization's
        truncation record; the same arrays when nothing can be
        discarded."""
        from api._realization import balanced_truncation
        ar, br, cr, info = balanced_truncation(A, B, C, tol)
        rec = {'n': int(info['n']), 'r': int(info['r']),
               'bound_rel': float(info['bound_rel']),
               'err_rel': float(info['err_rel']), 'kept': info['kept']}
        tr = getattr(self.real, 'truncation', None)
        if isinstance(tr, dict):
            with self._lock:
                d = tr.setdefault('dressed', {})
                d[(int(pi), int(ri), str(att.get('symbol')))] = rec
                tr['dressed_states'] = (
                    sum(v['n'] for v in d.values()),
                    sum(v['r'] for v in d.values()))
                tr['dressed_max_bound_rel'] = max(
                    v['bound_rel'] for v in d.values())
        if info['r'] >= A.shape[0]:
            return A, B, C
        _count('ss_dressed_reduced')
        return (np.asarray(ar, dtype=self.dtype),
                np.asarray(br, dtype=self.dtype).reshape(-1),
                np.asarray(cr, dtype=self.dtype).reshape(-1))

    def solve_data(self, ents):
        """Resolvent data for a tensor whose axes carry the entries ``ents``
        (a tuple of ``_EntryOps``)."""
        key = tuple(id(e) for e in ents)
        with self._lock:
            sd = self._solve_data.get(key)
            if sd is None:
                sd = _SolveData(ents, self.is_real)
                self._solve_data[key] = sd
            return sd


_SD_SERIAL = itertools.count()


class _SolveData:
    """``(⊕ a_e) y = -r`` for one active-edge pattern."""

    def __init__(self, ents, is_real):
        self.ents = ents
        self.shape = tuple(e.n for e in ents)
        self.is_real = is_real
        lam = np.zeros(self.shape, dtype=complex)
        for ax, e in enumerate(ents):
            sh = [1] * len(ents)
            sh[ax] = e.n
            lam = lam + e.dT.reshape(sh)
        if is_real and not any(e.kind == 'schur' for e in ents):
            lam = lam.real
        self.lam = lam
        self.nontriv = [(ax, e) for ax, e in enumerate(ents) if e.n > 1]
        self.sweeps = 1 + sum(e.n - 1 for e in ents)
        self.unitary = any(e.kind == 'schur' for e in ents)
        self.scalar = sum(e.lam for e in ents if e.n == 1)
        self.size = int(np.prod(self.shape, dtype=np.int64)) if ents else 1
        self.serial = next(_SD_SERIAL)

    def kron_sum(self, dtype, attr='a'):
        """The dense Kronecker sum ``⊕ a_e`` (``attr='a'``) or ``⊕ T_e``
        (``'T'``: the per-edge triangular basis, upper triangular) of this
        pattern, cached (shared by every subset and segment of the model)."""
        key = ('kron', self.serial, attr, np.dtype(dtype).str)
        hit = _dense_get(key)
        if hit is not None:
            return hit[0]
        M = _kron_sum_dense(self.ents, dtype, attr=attr)
        M.setflags(write=False)
        _dense_put(key, M, M.nbytes)
        return M

    def _dense_ops(self):
        """For a block of at most ``DENSE_SOLVE_MAX`` entries: ``(U, U^H,
        T, M)``, the Kronecker product of the per-edge basis changes
        (identity, the index reversal, the unitary Schur factor), the
        upper-triangular ``⊕ T_e`` and the original ``⊕ a_e``, all dense and
        cached: a solve is then two matvecs and one back substitution."""
        hit = _dense_get(('sdops', self.serial))
        if hit is not None:
            return hit[0]
        cplx = np.iscomplexobj(self.lam) or self.unitary
        dt = complex if cplx else float
        U = np.ones((1, 1), dtype=dt)
        for e in self.ents:
            if e.kind == 'schur':
                f = np.asarray(e.U, dtype=dt)
            elif e.kind == 'lower':
                f = np.eye(e.n, dtype=dt)[::-1]
            else:
                f = np.eye(e.n, dtype=dt)
            U = np.kron(U, f)
        ops = (U, U.conj().T.copy(), self.kron_sum(dt, 'T'),
               self.kron_sum(self.ents[0].a.dtype if self.ents else float))
        _dense_put(('sdops', self.serial), ops,
                   sum(int(x.nbytes) for x in ops))
        return ops

    def apply_M(self, y):
        """``(⊕ a_e) y`` in the original basis."""
        if self.nontriv and self.size <= DENSE_SOLVE_MAX:
            M = self._dense_ops()[3]
            return (M @ np.ravel(y)).reshape(self.shape)
        out = self.scalar * y if self.scalar != 0 else np.zeros_like(y)
        for ax, e in self.nontriv:
            out = out + _mode(y, e.a, ax)
        return out

    def _solve_once(self, r):
        if self.nontriv and self.size <= DENSE_SOLVE_MAX:
            U, Uh, T, _M = self._dense_ops()
            rt = Uh @ np.ravel(r)
            y = U @ sl.solve_triangular(T, -rt, lower=False,
                                        check_finite=False)
            y = y.reshape(self.shape)
            if self.is_real and np.iscomplexobj(y):
                y = y.real
            return y
        rt = r
        for ax, e in self.nontriv:
            rt = e.to_basis(rt, ax)
        y = -rt / self.lam
        for _ in range(self.sweeps - 1):
            acc = -rt
            for ax, e in self.nontriv:
                acc = acc - _mode(y, e.Nup, ax)
            y = acc / self.lam
        for ax, e in self.nontriv:
            y = e.from_basis(y, ax)
        if self.is_real and np.iscomplexobj(y):
            y = y.real
        return y

    def solve_shifted(self, r, z):
        """``y`` with ``(⊕ a_e − z I) y = −r`` for a complex shift ``z``
        (C0 output: resolvents of blocks beyond the dense forms).  The same
        two paths as :meth:`solve_neg` with the diagonal shifted: the dense
        back substitution with ``T − z I``, or the per-edge triangular sweeps
        (exact: the off-diagonal part is nilpotent) with ``λ − z``; Schur
        patterns refined as there.  No counters (the run's census is not
        touched)."""
        z = complex(z)
        r = np.asarray(r, dtype=complex)
        y = self._solve_shifted_once(r, z)
        if self.unitary:
            for _ in range(REFINE_STEPS):
                res = -r - (self.apply_M(y) - z * y)
                if not np.any(res):
                    break
                y = y + self._solve_shifted_once(-res, z)
        return y

    def _solve_shifted_once(self, r, z):
        if self.nontriv and self.size <= DENSE_SOLVE_MAX:
            U, Uh, T, _M = self._dense_ops()
            y = U @ sl.solve_triangular(T - z * np.eye(T.shape[0]),
                                        -(Uh @ np.ravel(r)), lower=False,
                                        check_finite=False)
            return y.reshape(self.shape)
        rt = r
        for ax, e in self.nontriv:
            rt = e.to_basis(rt, ax)
        lam = self.lam - z
        if np.any(lam == 0):
            raise ZeroDivisionError(f'the shift z = {z} is an eigenvalue of '
                                    f'the pattern (singular resolvent)')
        y = -rt / lam
        for _ in range(self.sweeps - 1):
            acc = -rt
            for ax, e in self.nontriv:
                acc = acc - _mode(y, e.Nup, ax)
            y = acc / lam
        for ax, e in self.nontriv:
            y = e.from_basis(y, ax)
        return y

    def solve_neg(self, r, want_residual=False):
        """``y`` with ``(⊕ a_e) y = -r``, i.e. ``y = (-M)^{-1} r``.  With
        ``want_residual``: ``(y, rho)``, ``rho`` the relative residual
        ``|r + M y|_inf / |r|_inf`` of the returned ``y`` (G6's estimate)."""
        _count('ss_resolvent_solves')
        y = self._solve_once(r)
        if not self.unitary or REFINE_STEPS <= 0:
            if not want_residual:
                return y
            rn = float(np.abs(-r - self.apply_M(y)).max()) if y.size else 0.0
            return y, _rel_res(rn, r)
        res = -r - self.apply_M(y)
        best, rbest = y, float(np.abs(res).max())
        for _ in range(REFINE_STEPS):
            if rbest == 0.0:
                break
            y = best + self._solve_once(-res)
            res = -r - self.apply_M(y)
            rn = float(np.abs(res).max())
            _count('ss_refinements')
            if not rn < rbest:
                break
            best, rbest = y, rn
        return (best, _rel_res(rbest, r)) if want_residual else best


def _rel_res(rn, r):
    rm = float(np.abs(r).max()) if np.size(r) else 0.0
    return rn / rm if rm > 0 else (0.0 if rn == 0 else math.inf)


# ─── the error estimate's per-evaluation accumulators (G6) ────────────────

_EST = threading.local()
#: the per-step unit of the error estimate (the closed-form routes'
#: ``COND_ERR_FACTOR`` convention)
EST_UNIT = 1e-15


def _est_steps(n):
    st = getattr(_EST, 'steps', None)
    if st is not None:
        _EST.steps = st + n


# ─── dense segment generators (LRU, byte capped) ──────────────────────────

_DENSE_LOCK = threading.RLock()
_DENSE_CACHE = OrderedDict()
_DENSE_BYTES = [0]


def _dense_get(key):
    with _DENSE_LOCK:
        hit = _DENSE_CACHE.get(key)
        if hit is not None:
            _DENSE_CACHE.move_to_end(key)
        return hit


def _dense_put(key, val, nbytes):
    with _DENSE_LOCK:
        if nbytes > DENSE_CACHE_BYTES or key in _DENSE_CACHE:
            return
        _DENSE_CACHE[key] = (val, nbytes)
        _DENSE_BYTES[0] += nbytes
        while _DENSE_BYTES[0] > DENSE_CACHE_BYTES and _DENSE_CACHE:
            _k, (_v, nb) = _DENSE_CACHE.popitem(last=False)
            _DENSE_BYTES[0] -= nb


# ─── Taylor action (Al-Mohy & Higham 2011) ────────────────────────────────

_THETA = {1: 2.29e-16, 2: 2.58e-08, 3: 1.39e-05, 4: 3.4e-04, 5: 2.4e-03,
          6: 9.07e-03, 7: 2.38e-02, 8: 5.0e-02, 9: 8.96e-02, 10: 1.44e-01,
          11: 2.14e-01, 12: 3.0e-01, 13: 4.0e-01, 14: 5.14e-01, 15: 6.41e-01,
          16: 7.81e-01, 17: 9.31e-01, 18: 1.09, 19: 1.26, 20: 1.44,
          21: 1.62, 22: 1.82, 23: 2.01, 24: 2.22, 25: 2.43, 26: 2.64,
          27: 2.86, 28: 3.08, 29: 3.31, 30: 3.54, 35: 4.7, 40: 6.0,
          45: 7.2, 50: 8.5, 55: 9.9}
_M_MAX = 55
_P_MAX = 8                      # largest p with p(p-1) <= m_max + 1
_TOL = 2.0 ** -53


def _taylor_params(dnorms, t):
    """(m*, s) of Al-Mohy & Higham's fragment (3.1) for ``t A``; ``dnorms[p]``
    = ``||A^p||_1^{1/p}`` (an upper bound is safe), p = 1 .. P_MAX + 1."""
    a1 = t * dnorms[1]
    if a1 == 0.0:
        return 0, 1
    ell = 2
    if a1 <= 2 * ell * _P_MAX * (_P_MAX + 3) * _THETA[_M_MAX] / _M_MAX:
        best = None
        for m, th in _THETA.items():
            s = int(math.ceil(a1 / th))
            if best is None or m * s < best[0] * best[1]:
                best = (m, s)
        return best
    best = None
    for p in range(2, _P_MAX + 1):
        alpha = t * max(dnorms[p], dnorms[p + 1])
        for m in range(p * (p - 1) - 1, _M_MAX + 1):
            th = _THETA.get(m)
            if th is None:
                continue
            s = max(1, int(math.ceil(alpha / th)))
            if best is None or m * s < best[0] * best[1]:
                best = (m, s)
    return best


def _taylor_action(apply_A, mu, dnorms, x, t, min_terms=0):
    """``exp(t (A + mu I)) x`` by the truncated Taylor series of ``t A`` in
    ``s`` steps (``A`` already trace shifted).  ``min_terms``: never stop a
    step's series before this many terms (the nilpotent firing part: a block
    reached after k firings starts at order t^k, and the norm-wise stopping
    test alone would cut a deep, tiny block to 0)."""
    m, s = _taylor_params(dnorms, t)
    if m == 0:
        return x * np.exp(t * mu) if mu != 0 else x.copy()
    if s > 1:                      # only a short segment can hide a block
        min_terms = 0
    m = max(m, min_terms)
    _count('ss_segments_taylor')
    _est_steps(s)
    F = x
    b = x
    eta = np.exp(t * mu / s)
    nmv = 0
    for _ in range(s):
        c1 = float(np.abs(b).max())
        for j in range(1, m + 1):
            b = apply_A(b) * (t / (s * j))
            nmv += 1
            c2 = float(np.abs(b).max())
            F = F + b
            if j >= min_terms and c1 + c2 <= _TOL * float(np.abs(F).max()):
                break
            c1 = c2
        F = eta * F
        b = F
    _count('ss_taylor_matvecs', nmv)
    return F


def _onenorm_powers(A, pmax=_P_MAX + 1):
    """``d_p = ||A^p||_1^{1/p}``, p = 1..pmax: exact for a small dense ``A``,
    1-norm estimates (scipy ``onenormest``) for larger ones."""
    d = {1: float(np.abs(A).sum(axis=0).max()) if A.size else 0.0}
    n = A.shape[0]
    if n <= 300:
        P = A
        for p in range(2, pmax + 1):
            P = P @ A
            d[p] = float(np.abs(P).sum(axis=0).max()) ** (1.0 / p)
        return d
    from scipy.sparse.linalg import LinearOperator, onenormest
    AT = A.conj().T
    for p in range(2, pmax + 1):
        def mv(v, p=p):
            for _ in range(p):
                v = A @ v
            return v

        def rmv(v, p=p):
            for _ in range(p):
                v = AT @ v
            return v
        op = LinearOperator((n, n), matvec=mv, rmatvec=rmv,
                            matmat=mv, rmatmat=rmv, dtype=A.dtype)
        d[p] = float(onenormest(op)) ** (1.0 / p)
    return d


# ─── geometry ─────────────────────────────────────────────────────────────

class _Segment:
    """The block system of one finite segment: blocks = downsets reachable
    in it, each with its active edges; firings between them."""

    def __init__(self, sub, blocks, acts, trans, key, depth=0):
        self.sub = sub
        self.blocks = blocks
        self.index = {I: k for k, I in enumerate(blocks)}
        self.acts = acts
        self.ents = [tuple(sub.edge_ops[e] for e in act) for act in acts]
        self.shapes = [tuple(e.n for e in en) for en in self.ents]
        self.sizes = [int(np.prod(sh, dtype=np.int64)) for sh in self.shapes]
        offs, o = [], 0
        for sz in self.sizes:
            offs.append(o)
            o += sz
        self.offs = offs
        self.D = o
        self.trans = trans          # (kI, kJ, sublist operands builder)
        self.key = key
        self._sd = [sub.model.solve_data(en) for en in self.ents]
        self._dnorms_free = None
        self._adj = None
        self._graded = None
        # ``depth``: firings to the deepest block plus the orders at which the
        # impulses of the edges born and completed in the segment start
        self.min_terms = min(_M_MAX, depth)

    # stacking
    def stack(self, xd, dtype):
        v = np.zeros(self.D, dtype=dtype)
        for I, x in xd.items():
            k = self.index[I]
            v[self.offs[k]:self.offs[k] + self.sizes[k]] = np.ravel(x)
        return v

    def unstack(self, v):
        return {I: v[self.offs[k]:self.offs[k] + self.sizes[k]].reshape(
                    self.shapes[k]) for k, I in enumerate(self.blocks)}

    # matrix-free apply of G
    def apply(self, v):
        out = np.zeros_like(v)
        for k in range(len(self.blocks)):
            o, sz = self.offs[k], self.sizes[k]
            x = v[o:o + sz].reshape(self.shapes[k])
            out[o:o + sz] += np.ravel(self._sd[k].apply_M(x))
        for (kI, kJ, fire) in self.trans:
            o, sz = self.offs[kI], self.sizes[kI]
            x = v[o:o + sz].reshape(self.shapes[kI])
            oJ, szJ = self.offs[kJ], self.sizes[kJ]
            out[oJ:oJ + szJ] += np.ravel(fire(x))
        return out

    def resolve(self, v, z):
        """``(z I − G)^{-1} v`` (complex ``z``) by forward substitution over
        the blocks (``G`` is block lower triangular: a firing goes from a
        downset to one vertex larger), one shifted Kronecker-sum solve per
        block (:meth:`_SolveData.solve_shifted`).  Matrix free: no dense
        form of the segment (C0 output for blocks beyond ``MAX_BLOCK``)."""
        v = np.asarray(v, dtype=complex)
        y = np.zeros(self.D, dtype=complex)
        into = {}
        for (kI, kJ, fire) in self.trans:
            into.setdefault(kJ, []).append((kI, fire))
        order = sorted(range(len(self.blocks)),
                       key=lambda k: len(self.blocks[k]))
        done = set()
        for k in order:
            o, sz = self.offs[k], self.sizes[k]
            r = v[o:o + sz].reshape(self.shapes[k]).copy()
            for kI, fire in into.get(k, ()):
                if kI not in done:
                    raise RuntimeError('segment firings are not ordered by '
                                       'downset size')
                oI, szI = self.offs[kI], self.sizes[kI]
                r = r + fire(y[oI:oI + szI].reshape(self.shapes[kI]))
            # (z − M_k) y_k = r  <=>  (M_k − z) y_k = −r
            y[o:o + sz] = np.ravel(self._sd[k].solve_shifted(r, z))
            done.add(k)
        return y

    def graded(self):
        """Exact triangular bases and the slowest diagonal rate below
        ``GRADED_RATIO`` of the mean one (``action``)."""
        g = self._graded
        if g is None:
            if any(e.kind == 'schur' and e.n > 2
                   for en in self.ents for e in en) or not self.D:
                g = False
            else:
                mu = (self.trace() / self.D).real
                # blocks with an active edge (a block without one is
                # constant, not a slow mode)
                slow = max([float(np.max(np.real(sd.lam)))
                            for sd in self._sd if sd.ents and np.size(sd.lam)]
                           or [-math.inf])
                g = bool(mu < 0 and slow > -math.inf
                         and slow > GRADED_RATIO * mu)
            self._graded = g
        return g

    def trace(self):
        tr = 0j
        for k, en in enumerate(self.ents):
            for e in en:
                tr += e.trace * (self.sizes[k] / e.n)
        return tr

    def dense(self, dtype):
        key = ('G', self.key)
        hit = _dense_get(key)
        if hit is not None:
            return hit[0]
        D = self.D
        G = np.zeros((D, D), dtype=dtype)
        for k in range(len(self.blocks)):
            o, sz = self.offs[k], self.sizes[k]
            G[o:o + sz, o:o + sz] = self._sd[k].kron_sum(dtype)
        for (kI, kJ, fire) in self.trans:
            o, sz = self.offs[kI], self.sizes[kI]
            oJ, szJ = self.offs[kJ], self.sizes[kJ]
            basis = np.eye(sz, dtype=dtype).reshape(self.shapes[kI] + (sz,))
            G[oJ:oJ + szJ, o:o + sz] = fire(basis, batch=True).reshape(szJ, sz)
        mu = np.trace(G) / D if D else 0.0
        if self.sub.model.is_real:
            mu = float(np.real(mu))
        A = G - mu * np.eye(D, dtype=dtype)
        dn = _onenorm_powers(A)
        val = (G, A, mu, dn)
        _dense_put(key, val, 2 * G.nbytes)
        return val

    def apply_adj(self, v):
        """Matrix-free ``G^H v``."""
        out = np.zeros_like(v)
        for k in range(len(self.blocks)):
            o, sz = self.offs[k], self.sizes[k]
            x = v[o:o + sz].reshape(self.shapes[k])
            y = np.zeros_like(x)
            for ax, e in enumerate(self.ents[k]):
                y = y + _mode(x, e.a.conj().T, ax)
            out[o:o + sz] += np.ravel(y)
        if self._adj is None:
            self._adj = [(kI, kJ, f.variant('adj'))
                         for (kI, kJ, f) in self.trans]
        for (kI, kJ, fire) in self._adj:
            oJ, szJ = self.offs[kJ], self.sizes[kJ]
            x = v[oJ:oJ + szJ].reshape(self.shapes[kJ])
            o, sz = self.offs[kI], self.sizes[kI]
            out[o:o + sz] += np.ravel(fire(x))
        return out

    def free_norms(self):
        """``mu`` and ``||(G - mu I)^p||_1^{1/p}`` for the matrix-free
        operator: the exact 1-norm bound for p = 1, scipy ``onenormest`` on
        the operator powers (with the adjoint) for p >= 2, as scipy's
        ``expm_multiply`` does; a crude bound for every p would cost ~60x
        more steps on a graded realization."""
        if self._dnorms_free is None:
            from scipy.sparse.linalg import LinearOperator, onenormest
            mu = self.trace() / self.D if self.D else 0.0
            if self.sub.model.is_real:
                mu = float(np.real(mu))
            col = [0.0] * len(self.blocks)
            for k, en in enumerate(self.ents):
                col[k] = sum(e.norm1 for e in en) + abs(mu)
            for (kI, kJ, fire) in self.trans:
                col[kI] += fire.norm1
            nrm = max(col) if col else 0.0
            dn = {1: nrm}
            n = self.D
            dt = complex if not self.sub.model.is_real else float
            A = lambda y: self.apply(y) - mu * y
            AH = lambda y: self.apply_adj(y) - np.conj(mu) * y
            for p in range(2, _P_MAX + 2):
                def mv(v, p=p):
                    v = np.asarray(v).reshape(n, -1)
                    out = np.empty_like(v, dtype=np.result_type(v, dt))
                    for j in range(v.shape[1]):
                        y = v[:, j]
                        for _ in range(p):
                            y = A(y)
                        out[:, j] = y
                    return out

                def rmv(v, p=p):
                    v = np.asarray(v).reshape(n, -1)
                    out = np.empty_like(v, dtype=np.result_type(v, dt))
                    for j in range(v.shape[1]):
                        y = v[:, j]
                        for _ in range(p):
                            y = AH(y)
                        out[:, j] = y
                    return out
                op = LinearOperator((n, n), matvec=mv, rmatvec=rmv,
                                    matmat=mv, rmatmat=rmv, dtype=dt)
                try:
                    dn[p] = min(nrm, float(onenormest(op)) ** (1.0 / p))
                except Exception:
                    dn[p] = nrm
            self._dnorms_free = (mu, dn)
        return self._dnorms_free

    def tri(self):
        """The generator in the per-edge triangular basis with the blocks in
        reverse order (larger downsets first): upper triangular (diagonal
        blocks Kronecker sums of the triangular ``T_e``, firings above the
        diagonal).  Cached (LRU)."""
        key = ('T', self.key)
        hit = _dense_get(key)
        if hit is not None:
            return hit[0]
        cplx = (not self.sub.model.is_real) or any(
            e.kind == 'schur' for en in self.ents for e in en)
        dt = complex if cplx else float
        nb = len(self.blocks)
        roffs, o = [0] * nb, 0
        for k in reversed(range(nb)):
            roffs[k] = o
            o += self.sizes[k]
        G = np.zeros((self.D, self.D), dtype=dt)
        for k in range(nb):
            r, sz = roffs[k], self.sizes[k]
            G[r:r + sz, r:r + sz] = self._sd[k].kron_sum(dt, 'T')
        for (kI, kJ, fire) in self.trans:
            rI, szI = roffs[kI], self.sizes[kI]
            rJ, szJ = roffs[kJ], self.sizes[kJ]
            basis = np.eye(szI, dtype=dt).reshape(self.shapes[kI] + (szI,))
            G[rJ:rJ + szJ, rI:rI + szI] = fire.variant('tri')(
                basis, batch=True).reshape(szJ, szI)
        val = (G, roffs)
        _dense_put(key, val, G.nbytes)
        return val

    def _action_tri(self, v, L, dtype):
        """``exp(G L) v`` through the upper-triangular form (``tri``) and
        the reset-diagonal triangular Pade (``api._realization.
        expm_upper_triangular``): exact ``e^{lambda L}`` on the diagonal, so
        a stiff decay costs no accuracy."""
        from api._realization import expm_upper_triangular
        G, roffs = self.tri()
        _count('ss_segments_triangular')
        _est_steps(1)
        xt = np.zeros(self.D, dtype=G.dtype)
        for k in range(len(self.blocks)):
            x = v[self.offs[k]:self.offs[k] + self.sizes[k]].reshape(
                self.shapes[k])
            for ax, e in enumerate(self.ents[k]):
                x = e.to_basis(x, ax)
            xt[roffs[k]:roffs[k] + self.sizes[k]] = np.ravel(x)
        yt = expm_upper_triangular(G, L) @ xt
        out = np.zeros(self.D, dtype=np.result_type(v, G.dtype))
        for k in range(len(self.blocks)):
            y = yt[roffs[k]:roffs[k] + self.sizes[k]].reshape(self.shapes[k])
            for ax, e in enumerate(self.ents[k]):
                y = e.from_basis(y, ax)
            out[self.offs[k]:self.offs[k] + self.sizes[k]] = np.ravel(y)
        if np.iscomplexobj(out) and not np.iscomplexobj(v) and (
                self.sub.model.is_real):
            out = out.real
        return out

    def propagate(self, xd, L, dtype):
        """``exp(G L)`` applied to the state dict ``xd`` (missing blocks: 0)."""
        v = self.stack(xd, dtype)
        if L == 0.0 or self.D == 0:
            return self.unstack(v)
        out = self.action(v, L, dtype)
        return self.unstack(out)

    def action(self, v, L, dtype):
        """``exp(G L) v``: the trace-shifted truncated-Taylor action (dense
        ``G`` up to ``OP_DENSE_MAX``, matrix-free beyond; no squaring: scipy's
        ``expm`` loses up to ~1e-13 relative on decaying blocks).  A stiff
        segment (more than ``TAYLOR_S_MAX`` scaling steps with exact
        triangular bases, ``TAYLOR_S_MAX_SCHUR`` with a unitary Schur basis)
        takes the triangular route up to ``TRI_DENSE_MAX`` states; beyond it,
        more than ``TAYLOR_S_HARD`` steps raise."""
        if self.D <= OP_DENSE_MAX:
            G, A, mu, dn = self.dense(dtype)
            apply_A = (lambda y: A @ y)
        else:
            mu, dn = self.free_norms()
            apply_A = ((lambda y: self.apply(y) - mu * y) if mu != 0
                       else self.apply)
        _m, s_ = _taylor_params(dn, L)
        unitary = any(e.kind == 'schur' and e.n > 2
                      for en in self.ents for e in en)
        lim = (TAYLOR_S_MAX_SCHUR if unitary else
               TAYLOR_S_MAX_GRADED if self.graded() else TAYLOR_S_MAX)
        if s_ > lim and self.D <= TRI_DENSE_MAX:
            return self._action_tri(v, L, dtype)
        if s_ > TAYLOR_S_HARD:
            raise SSLateRefusal(
                'stiff_segment', f'a stiff segment ({self.D:,} states, '
                f'{s_:,} Taylor steps for length {L:g}) is beyond its accurate '
                f'range')
        return _taylor_action(apply_A, mu, dn, v, L, self.min_terms)


def _kron_sum_dense(ents, dtype, attr='a'):
    shape = tuple(e.n for e in ents)
    D = int(np.prod(shape, dtype=np.int64))
    M = np.zeros((D, D), dtype=dtype)
    for ax, e in enumerate(ents):
        left = int(np.prod(shape[:ax], dtype=np.int64))
        right = int(np.prod(shape[ax + 1:], dtype=np.int64))
        M = M + np.kron(np.kron(np.eye(left, dtype=dtype), getattr(e, attr)),
                        np.eye(right, dtype=dtype))
    return M


class _Fire:
    """Contract the edges ``ends`` (by ``c_e``) and open the edges
    ``starts`` (by ``b_e``) of a tensor whose axes are ``act_in``; the result
    has axes ``act_out`` (all lists of edge ids, increasing)."""
    __slots__ = ('args', 'out', 'act_in', 'norm1', 'sub', 'ends', 'starts')

    def __init__(self, sub, act_in, ends, starts, act_out, basis='orig'):
        """``basis``: 'orig' (``c_e``, ``b_e``), 'tri' (the triangular-basis
        ``ct``, ``bt``) or 'adj' (the adjoint map: ``act_in``/``act_out`` are
        then the adjoint's, the edges ``ends`` are contracted by conj(b) and
        ``starts`` opened by conj(c))."""
        args = []
        ops = sub.edge_ops
        for e in ends:
            v = (ops[e].c if basis == 'orig' else ops[e].ct if basis == 'tri'
                 else ops[e].b.conj())
            args += [v, [e]]
        for e in starts:
            v = (ops[e].b if basis == 'orig' else ops[e].bt if basis == 'tri'
                 else ops[e].c.conj())
            args += [v, [e]]
        self.args = args
        self.sub, self.ends, self.starts = sub, list(ends), list(starts)
        self.act_in = list(act_in)
        self.out = list(act_out)
        n1 = 1.0
        for e in ends:
            n1 *= sub.edge_ops[e].cnorm
        for e in starts:
            n1 *= sub.edge_ops[e].bnorm
        self.norm1 = n1

    def variant(self, basis):
        """The same firing with the triangular-basis vectors ('tri') or its
        adjoint ('adj', from ``out`` back to ``act_in``)."""
        if basis == 'adj':
            return _Fire(self.sub, self.out, self.starts, self.ends,
                         self.act_in, basis='adj')
        return _Fire(self.sub, self.act_in, self.ends, self.starts, self.out,
                     basis=basis)

    def __call__(self, x, batch=False):
        if batch:
            z = MAX_EDGES + 1
            return np.einsum(x, self.act_in + [z], *self.args,
                             self.out + [z])
        return np.einsum(x, self.act_in, *self.args, self.out)

    def absval(self, x):
        """The firing in absolute values, ``|c| . |x|`` (a scalar for the
        last contraction): the magnitude of the error estimate."""
        aa = [np.abs(a_) if i % 2 == 0 else a_
              for i, a_ in enumerate(self.args)]
        return float(np.abs(np.einsum(np.abs(x), self.act_in, *aa,
                                      self.out)).max(initial=0.0))


_SUBSET_SERIAL = itertools.count()


class SubsetSS:
    """One subset's geometry and its state-space evaluation (no prefactor).

    ``ents``: the ``_EntryOps`` of the smooth rows (``None`` entries are
    not allowed); ``cdata``: the rows ``(a_int, a_ext, c0)``; ``m``: number
    of internal times; ``n_free``: number of free external times.  Raises
    ``SSIneligible`` for a shape it does not take.
    """

    def __init__(self, model, ents, cdata, m, n_free):
        self.model = model
        self.m = int(m)
        self.n_free = int(n_free)
        if len(cdata) > MAX_EDGES:
            raise SSIneligible('too_many_edges', str(len(cdata)))
        self.const_rows = []            # (row idx, a_int, a_ext, c0, ent)
        edges = []                      # (tail, head, ent, row idx)
        slots = {}
        slot_list = []

        def slot(a_ext, c0):
            key = tuple(0.0 if x == 0 else float(x) for x in a_ext) + (
                0.0 if c0 == 0 else float(c0),)
            k_ = slots.get(key)
            if k_ is None:
                k_ = slots[key] = len(slot_list)
                slot_list.append(key)
            return k_

        atol = _FI()._ROW_COEF_ATOL
        for idx, ((a_int, a_ext, c0), ent) in enumerate(zip(cdata, ents)):
            a_int = [0.0 if abs(float(x)) <= atol else float(x) for x in a_int]
            a_ext = [float(x) for x in a_ext]
            c0 = float(c0)
            if len(a_int) != self.m or len(a_ext) != self.n_free:
                raise SSIneligible('row_shape', 'row length')
            nz = [(i, a) for i, a in enumerate(a_int) if a != 0.0]
            if not nz:
                self.const_rows.append((idx, a_int, a_ext, c0, ent))
                continue
            heads = [i for i, a in nz if a == 1.0]
            tails = [i for i, a in nz if a == -1.0]
            if (len(heads) + len(tails) != len(nz) or len(heads) > 1
                    or len(tails) > 1):
                raise SSIneligible('row_shape', f'a_int={a_int}')
            if heads and tails:
                if any(x != 0.0 for x in a_ext) or c0 != 0.0:
                    raise SSIneligible('offset_row', f'{a_ext}, {c0}')
                edges.append((('v', tails[0]), ('v', heads[0]), ent, idx))
            elif heads:                 # Δt = s_v + (a·f + c0): tail -(a·f+c0)
                edges.append((('c', slot([-x for x in a_ext], -c0)),
                              ('v', heads[0]), ent, idx))
            else:                       # Δt = -s_u + (a·f + c0): head a·f+c0
                edges.append((('v', tails[0]),
                              ('c', slot(a_ext, c0)), ent, idx))
        self.edges = edges
        self.edge_ops = [e[2] for e in edges]
        self.slots = slot_list
        m_ = self.m
        self.in_edges = [[] for _ in range(m_)]
        self.out_edges = [[] for _ in range(m_)]
        self.preds = [set() for _ in range(m_)]
        self.succs = [set() for _ in range(m_)]
        self.lo_slots = [set() for _ in range(m_)]
        self.hi_slots = [set() for _ in range(m_)]
        self.slot_heads = [[] for _ in slot_list]   # edges ending at a slot
        self.slot_tails = [[] for _ in slot_list]   # edges starting at a slot
        for k_, (t, h, _ent, _i) in enumerate(edges):
            if t[0] == 'v':
                self.out_edges[t[1]].append(k_)
            else:
                self.slot_tails[t[1]].append(k_)
            if h[0] == 'v':
                self.in_edges[h[1]].append(k_)
            else:
                self.slot_heads[h[1]].append(k_)
            if t[0] == 'v' and h[0] == 'v':
                if t[1] == h[1]:
                    raise SSIneligible('row_shape', 'self loop')
                self.preds[h[1]].add(t[1])
                self.succs[t[1]].add(h[1])
            elif t[0] == 'c':
                self.lo_slots[h[1]].add(t[1])
            else:
                self.hi_slots[t[1]].add(h[1])
        self.full = frozenset(range(m_))
        self.serial = next(_SUBSET_SERIAL)
        self.rho = 0.0                  # stationary solves' residual (G6)
        self._stat = None
        self._patterns = {}
        self._lock = threading.RLock()
        self._prefetch = {}
        # a directed cycle of internal rows (s_u < s_v < ... < s_u, every
        # internal row has zero offset): the region is empty, the value 0
        order = self._topo_order()
        self.empty = order is None
        if self.empty:
            self.free0 = frozenset()
            self._stat_downsets = [frozenset()]
            return
        # every vertex followed by an external time
        reach = [False] * m_
        for v in reversed(order):
            reach[v] = bool(self.hi_slots[v]) or any(reach[w]
                                                     for w in self.succs[v])
        if m_ and not all(reach):
            raise SSIneligible('vertex_after_last_time')
        if m_ and not slot_list:
            raise SSIneligible('no_external_time')
        # F0: vertices with no external-time tail in their ancestry
        free0 = [False] * m_
        for v in order:
            free0[v] = (not self.lo_slots[v]) and all(free0[u]
                                                      for u in self.preds[v])
        self.free0 = frozenset(v for v in range(m_) if free0[v])
        self._stat_downsets = self._downsets(self.free0)

    # ── structure helpers ─────────────────────────────────────────────
    def _topo_order(self):
        indeg = [len(p) for p in self.preds]
        stack = [v for v in range(self.m) if indeg[v] == 0]
        out = []
        while stack:
            v = stack.pop()
            out.append(v)
            for w in sorted(self.succs[v]):
                indeg[w] -= 1
                if indeg[w] == 0:
                    stack.append(w)
        if len(out) != self.m:
            return None
        return out

    def _closure(self, starts, placeable):
        """Downsets reachable from ``starts`` by placing vertices of
        ``placeable`` (whose predecessors are placed); sorted."""
        seen = set(starts)
        frontier = list(starts)
        while frontier:
            nxt = []
            for I in frontier:
                for v in placeable:
                    if v in I or not self.preds[v] <= I:
                        continue
                    J = I | {v}
                    if J not in seen:
                        seen.add(J)
                        nxt.append(J)
                        if len(seen) > MAX_DOWNSETS:
                            raise SSIneligible('too_many_downsets')
            frontier = nxt
        return sorted(seen, key=lambda I: (len(I), sorted(I)))

    def _downsets(self, verts):
        return self._closure([frozenset()], sorted(verts))

    def _act(self, I, passed):
        """Active edges (increasing ids) of downset ``I`` once the slots in
        ``passed`` are behind (``None``: the stationary segment)."""
        out = []
        for k_, (t, h, _e, _i) in enumerate(self.edges):
            if t[0] == 'v':
                t_done = t[1] in I
            else:
                t_done = passed is not None and t[1] in passed
            if not t_done:
                continue
            if h[0] == 'v':
                h_done = h[1] in I
            else:
                h_done = passed is not None and h[1] in passed
            if not h_done:
                out.append(k_)
        return tuple(out)

    def _maximal(self, J):
        return [v for v in sorted(J) if not (self.succs[v] & J)]

    def state_bytes(self):
        """Bytes of the cached stationary states (the memory estimate)."""
        item = 8 if self.model.is_real else 16
        tot = 0
        for J in self._stat_downsets:
            sz = 1
            for e in self._act(J, None):
                sz *= self.edge_ops[e].n
            tot += sz
        return tot * item

    def max_width(self):
        """(largest number of active edges, largest block size) over the
        stationary downsets."""
        w = b = 0
        for J in self._stat_downsets:
            act = self._act(J, None)
            sz = 1
            for e in act:
                sz *= self.edge_ops[e].n
            w, b = max(w, len(act)), max(b, sz)
        return w, b

    # ── stationary states (cached) ────────────────────────────────────
    def stationary(self):
        with self._lock:
            if self._stat is not None:
                return self._stat
            _count('ss_stationary_builds')
            dt = self.model.dtype
            x = {frozenset(): np.ones((), dtype=dt)}
            acts = {frozenset(): ()}
            for J in self._stat_downsets:
                if not J:
                    continue
                actJ = self._act(J, None)
                acts[J] = actJ
                r = None
                for v in self._maximal(J):
                    I = J - {v}
                    xI = x.get(I)
                    if xI is None:
                        continue
                    f = _Fire(self, acts[I], self.in_edges[v],
                              self.out_edges[v], actJ)
                    term = f(xI)
                    r = term if r is None else r + term
                if r is None:
                    continue
                sd = self.model.solve_data(tuple(self.edge_ops[e]
                                                 for e in actJ))
                x[J], rho = sd.solve_neg(r, want_residual=True)
                if rho > self.rho:
                    self.rho = rho
            self._stat = x
            return x

    # ── per-order programs ────────────────────────────────────────────
    def _program(self, pattern, finite_start):
        key = (pattern, finite_start)
        with self._lock:
            prog = self._patterns.get(key)
            if prog is not None:
                if isinstance(prog, BaseException):
                    raise prog          # a refusal: not rebuilt per call
                return prog
        try:
            prog = self._build_program(pattern, finite_start)
        except SSIneligible as exc:
            with self._lock:
                self._patterns[key] = exc
            raise
        with self._lock:
            self._patterns[key] = prog
        return prog

    def _build_program(self, pattern, finite_start):
        steps = []
        if finite_start:
            blocks = self._downsets(self.free0)
            seg = self._segment(blocks, None, sorted(self.free0),
                                ('tmin', self.serial, pattern))
            steps.append(('seg', seg))
            states = list(blocks)
        else:
            states = list(self._stat_downsets)
        passed = set()
        for j, group in enumerate(pattern):
            before = frozenset(passed)
            passed |= set(group)
            after = frozenset(passed)
            req = frozenset(v for v in range(self.m)
                            if self.hi_slots[v] & after)
            ends = [k_ for s in group for k_ in self.slot_heads[s]]
            starts = [k_ for s in group for k_ in self.slot_tails[s]]
            ev = []
            kept = []
            for I in states:
                if not req <= I:
                    continue
                act_in = self._act(I, before)
                act_out = self._act(I, after)
                e_in = sorted(e for e in ends if e in act_in)
                f = _Fire(self, act_in, e_in, sorted(starts), act_out)
                ev.append((I, f))
                kept.append(I)
            steps.append(('event', ev))
            if j + 1 == len(pattern):
                states = kept
                break
            placeable = [v for v in range(self.m)
                         if self.lo_slots[v] <= after
                         and not (self.hi_slots[v] & after)]
            blocks = self._closure(kept, placeable)
            seg = self._segment(blocks, after, placeable,
                                (self.serial, pattern, finite_start, j))
            steps.append(('seg', seg))
            states = blocks
        if self.full not in states:
            steps.append(('empty', None))
        return steps

    def _segment(self, blocks, passed, placeable, key):
        acts = [self._act(I, passed) for I in blocks]
        _check_transient(self, acts)
        index = {I: k for k, I in enumerate(blocks)}
        trans = []
        pl = set(placeable)
        for kI, I in enumerate(blocks):
            for v in sorted(pl - I):
                if not self.preds[v] <= I:
                    continue
                J = I | {v}
                kJ = index.get(J)
                if kJ is None:
                    continue
                f = _Fire(self, acts[kI], sorted(self.in_edges[v]),
                          sorted(self.out_edges[v]), acts[kJ])
                trans.append((kI, kJ, f))
        depth = len(pl) + sum(
            self.edge_ops[k_].n - 1
            for k_, (t, h, _e, _i) in enumerate(self.edges)
            if (t[0] == 'v' and t[1] in pl) or (h[0] == 'v' and h[1] in pl))
        return _Segment(self, blocks, acts, trans, key, depth)

    # ── evaluation ────────────────────────────────────────────────────
    def _slot_values(self, free_vals):
        vals = []
        for key in self.slots:
            v = key[-1]
            for a, f in zip(key[:-1], free_vals):
                if a != 0.0:
                    v = v + a * float(f)
            vals.append(v)
        return vals

    def _pattern(self, vals):
        order = sorted(range(len(vals)), key=lambda i: vals[i])
        groups, gv = [], []
        for i in order:
            if gv and vals[i] == gv[-1]:
                groups[-1].append(i)
            else:
                groups.append([i])
                gv.append(vals[i])
        return tuple(tuple(sorted(g)) for g in groups), gv

    def scalar_factor(self, free_vals, tie_ctx):
        """Product of the constant rows' factors; 0 when one is EMPTY."""
        FI = _FI()
        scal = 1.0
        for (_idx, a_int, a_ext, c0, ent) in self.const_rows:
            verdict = FI._const_row_verdict(a_int, a_ext, c0, free_vals,
                                            'edge', tie_ctx)
            if verdict == 'EMPTY':
                return 0.0
            c_eff = float(c0) + sum(float(a_ext[j]) * float(free_vals[j])
                                    for j in range(len(a_ext)))
            scal = scal * ent.impulse(c_eff)
            if scal == 0:
                return 0.0
        return scal

    def evaluate(self, free_vals, tie_ctx=None, t_min=float('-inf')):
        """The subset integral (WITHOUT prefactor) at ``free_vals``.  Sets
        ``_EST.err`` (thread-local) to this evaluation's error estimate
        (module docstring, without the prefactor)."""
        _EST.err = 0.0
        if self.empty or any(e.kind == 'zero' for e in self.edge_ops) or any(
                r[4].kind == 'zero' for r in self.const_rows):
            return 0.0
        _EST.steps = 0              # the constant rows' impulses count too
        try:
            scal = self.scalar_factor(free_vals, tie_ctx)
            imp = len(self.const_rows) + _EST.steps
        finally:
            _EST.steps = None
        if scal == 0:
            return 0.0
        if self.m == 0:
            _EST.err = EST_UNIT * abs(scal) * (1 + imp)
            return scal
        vals = self._slot_values(free_vals)
        if not all(math.isfinite(v) for v in vals):
            raise ValueError('state-space route: external times must be '
                             f'finite, got {list(free_vals)}')
        pattern, gv = self._pattern(vals)
        finite = t_min != float('-inf')
        prog = self._checked_program(pattern, finite)
        _EST.steps, _EST.mag = imp, 0.0
        try:
            x = self._run(prog, gv, finite, t_min)
            _EST.err = abs(scal) * _EST.mag * (
                EST_UNIT * (1 + _EST.steps) + (0.0 if finite else self.rho))
        finally:
            _EST.steps = None
        return scal * x

    def _checked_program(self, pattern, finite):
        try:
            return self._program(pattern, finite)
        except SSIneligible as exc:
            raise SSLateRefusal(
                'segment_too_large',
                f'a segment of this subset is too large ({exc})') from exc

    def _run(self, prog, gv, finite, t_min, upto=None):
        """Run a program; ``gv``: the event times.  With ``upto``: return
        the state dict before step ``upto``."""
        dt = self.model.dtype
        x = ({frozenset(): np.ones((), dtype=dt)} if finite
             else self.stationary())
        j = 0
        prev = t_min if finite else None
        for i, (kind, data) in enumerate(prog):
            if upto is not None and i == upto:
                return x
            if kind == 'seg':
                L = gv[j] - prev
                if L < 0:
                    raise ValueError('state-space route: external times '
                                     'before PHASE_J_T_MIN')
                x = data.propagate(x, L, dt)
            elif kind == 'event':
                nx = {}
                last = i == len(prog) - 1
                for I, f in data:
                    xI = x.get(I)
                    if xI is not None:
                        nx[I] = f(xI)
                        if last and I == self.full:
                            _EST.mag = f.absval(xI)
                x = nx
                prev = gv[j]
                j += 1
            else:
                return 0.0
        v = x.get(self.full)
        return 0.0 if v is None else v.item()

    def evaluate_many(self, points, t_min=float('-inf'), errors=None):
        """Values (without prefactor) at many ``(free_vals, tie_ctx)``
        points.  Points sharing the order of the external times share the
        stationary state and every step before the last segment; along the
        last segment they are stepped from one length to the next (sorted).
        A point the route refuses late (``SSLateRefusal``, a segment over
        the memory budget) gets a ``_Refused`` value (the caller falls back
        for it).  ``errors``: a list filled with each point's error
        estimate (``evaluate``)."""
        try:
            return self._evaluate_many(points, t_min, errors)
        finally:
            _EST.steps = None

    def _evaluate_many(self, points, t_min, errors):
        late = _late_errors()
        out = [None] * len(points)
        err = [0.0] * len(points)
        if errors is not None:
            errors[:] = err
        groups = {}
        finite = t_min != float('-inf')
        for i, (fv, tc) in enumerate(points):
            if self.empty or any(e.kind == 'zero' for e in self.edge_ops) or any(
                    r[4].kind == 'zero' for r in self.const_rows):
                out[i] = 0.0
                continue
            vals = self._slot_values(fv)
            if not all(math.isfinite(v) for v in vals):
                raise ValueError('state-space route: external times must be '
                                 f'finite, got {list(fv)}')
            _EST.steps = 0
            scal = self.scalar_factor(fv, tc)
            imp = len(self.const_rows) + _EST.steps
            _EST.steps = None
            if scal == 0 or self.m == 0:
                out[i] = scal if self.m == 0 else 0.0
                err[i] = EST_UNIT * abs(scal) * (1 + imp) if self.m == 0 \
                    else 0.0
                continue
            pattern, gv = self._pattern(vals)
            groups.setdefault(pattern, []).append((i, gv, scal, imp))
        dt = self.model.dtype
        rho = 0.0 if finite else None
        for pattern, items in groups.items():
            try:
                prog = self._checked_program(pattern, finite)
            except late as exc:
                for (i, _gv, _scal, _imp) in items:
                    out[i] = _Refused(_late_reason(exc), _late_detail(exc))
                continue
            _count('ss_batch_points', len(items))
            segs = [i for i, (k_, _d) in enumerate(prog) if k_ == 'seg']
            if (finite or not segs or prog[-1][0] != 'event'
                    or segs[-1] != len(prog) - 2 or len(items) < 2):
                for (i, gv, scal, imp) in items:
                    _EST.steps, _EST.mag = imp, 0.0
                    try:
                        out[i] = scal * self._run(prog, gv, finite, t_min)
                    except late as exc:
                        out[i] = _Refused(_late_reason(exc), _late_detail(exc))
                        continue
                    err[i] = abs(scal) * _EST.mag * (
                        EST_UNIT * (1 + _EST.steps)
                        + (self.rho if rho is None else rho))
                continue
            last = segs[-1]
            # the state before the last segment depends only on the lengths
            # of the earlier segments (translation invariance): group by them
            pre = {}
            for (i, gv, scal, imp) in items:
                lens = tuple(gv[q + 1] - gv[q] for q in range(len(gv) - 1))
                pre.setdefault(lens[:-1], []).append((i, gv, scal, lens[-1],
                                                      imp))
            seg = prog[last][1]
            ev_last = [(I, f) for I, f in prog[last + 1][1]
                       if I == self.full]
            for _k, its in pre.items():
                _EST.steps, _EST.mag = 0, 0.0
                try:
                    x0 = self._run(prog, its[0][1], False, t_min, upto=last)
                except late as exc:
                    for it in its:
                        out[it[0]] = _Refused(_late_reason(exc), _late_detail(exc))
                    continue
                its = sorted(its, key=lambda it: it[3])
                v = seg.stack(x0, dt)
                cur = 0.0
                v0 = v
                refused = None
                for (i, gv, scal, L_last, imp) in its:
                    if refused is not None:
                        out[i] = refused
                        continue
                    if L_last < cur:            # never expected (sorted)
                        v, cur = v0, 0.0
                    if L_last > cur:
                        try:
                            v = seg.action(v, L_last - cur, dt)
                        except late as exc:
                            # longer segments are refused too (sorted)
                            refused = out[i] = _Refused(_late_reason(exc), _late_detail(exc))
                            continue
                        cur = L_last
                    nx = 0.0
                    mag = 0.0
                    if ev_last:
                        I, f = ev_last[0]
                        k = seg.index.get(I)
                        if k is not None:
                            xI = v[seg.offs[k]:seg.offs[k] + seg.sizes[k]]
                            xI = xI.reshape(seg.shapes[k])
                            nx = f(xI).item()
                            mag = f.absval(xI)
                    out[i] = scal * nx
                    err[i] = abs(scal) * mag * (
                        EST_UNIT * (1 + _EST.steps + imp) + self.rho)
        if errors is not None:
            errors[:] = err
        return out

    # ── C0 output (S4): the subset on a line as one matrix exponential ───
    def me_line(self, free_base, free_dir, t_anchor, t_ref, tie_ctx=None):
        """The subset integral (WITHOUT prefactor) on the line ``free_vals =
        free_base + t * free_dir`` inside the chamber that contains
        ``t_ref`` (an interval of ``t`` on which the order of the external
        times does not change), as a matrix-exponential triple in ``s =
        sigma (t - t_anchor) >= 0``, ``sigma = sign(t_ref - t_anchor)``
        (S4, item C0)::

            I(s) = const * prod_r g_r(rho0_r + rho1_r s) * c . exp(G beta s) . x0

        ``None`` when the integral vanishes on the whole chamber (an empty
        region, a zero entry, an EMPTY or negative constant row); otherwise a
        dict with ``const`` (the constant rows that do not move, and the
        value when nothing moves), ``rows`` (``[(rho0, rho1, entry)]``: the
        constant rows ``g_r(t) = c_r e^{a_r t} b_r`` that move), ``seg``
        (the moving segment's ``_Segment``, or ``None``), ``beta`` (its
        length per unit ``s``), ``x0`` (the stacked state that enters it at
        the anchor) and ``c`` (the stacked functional of everything after
        it: the remaining events and fixed segments; for a last segment the
        contraction of the last external time).  ``G`` is
        ``seg.dense(dtype)[0]``.

        Every slot value is affine in ``t``, the order of the external times
        is fixed in the chamber, so every segment length is affine in
        ``t``; exactly one may move (a k = 2 lag; a k >= 3 slice whose swept
        leg is first or last).  Raises ``SSIneligible`` (``several_lags``:
        more than one segment moves; ``finite_t_min``) otherwise; the
        route's late refusals (``SSLateRefusal``) propagate."""
        FI = _FI()
        try:
            t_min = float(FI.PHASE_J_T_MIN)
        except (TypeError, ValueError):
            t_min = None
        if t_min != float('-inf') or switch_on_active():
            raise SSIneligible('finite_t_min', repr(
                _switch_on_t_min() if switch_on_active()
                else FI.PHASE_J_T_MIN))
        if self.empty or any(e.kind == 'zero' for e in self.edge_ops) or any(
                r[4].kind == 'zero' for r in self.const_rows):
            return None
        fb = [float(x) for x in free_base]
        fd = [float(x) for x in free_dir]
        t_anchor, t_ref = float(t_anchor), float(t_ref)
        sigma = 1.0 if t_ref >= t_anchor else -1.0
        f_ref = [b + t_ref * d for b, d in zip(fb, fd)]
        const = 1.0
        rows = []
        for (_idx, a_int, a_ext, c0, ent) in self.const_rows:
            if FI._const_row_verdict(a_int, a_ext, c0, f_ref, 'edge',
                                     tie_ctx) == 'EMPTY':
                return None
            r0 = float(c0) + sum(float(a) * b for a, b in zip(a_ext, fb))
            r1 = sum(float(a) * d for a, d in zip(a_ext, fd))
            if r1 == 0.0:
                const = const * ent.impulse(r0)
                if const == 0:
                    return None
                continue
            rho0, rho1 = r0 + r1 * t_anchor, r1 * sigma
            if rho1 < 0 and rho0 <= 0:
                return None             # retarded: 0 on the whole chamber
            if rho0 < 0 or rho1 < 0:
                # the row changes sign inside the chamber (only with a
                # constant offset or between two other legs): not one
                # matrix exponential
                raise SSIneligible('row_crosses_zero',
                                   f'rho0={rho0:g}, rho1={rho1:g}')
            rows.append((rho0, rho1, ent))
        out = {'const': const, 'rows': rows, 'seg': None, 'beta': 0.0,
               'x0': None, 'c': None}
        if self.m == 0:
            return out

        def slot_vals(f):
            return self._slot_values(f)
        v_ref = slot_vals(f_ref)
        v_base = slot_vals(fb)
        # slopes from the slot keys (exact for the integer directions of
        # the callers; a difference of two slot values would carry rounding)
        v_dir = []
        for key in self.slots:
            v = 0.0
            for a_, d_ in zip(key[:-1], fd):
                if a_ != 0.0 and d_ != 0.0:
                    v = v + a_ * d_
            v_dir.append(v)
        pattern, gv = self._pattern(v_ref)
        if any(v_dir[s_] != v_dir[g[0]] for g in pattern for s_ in g):
            raise SSIneligible('several_lags', 't_ref is on a chamber '
                               'boundary (tied slots move apart)')
        prog = self._checked_program(pattern, False)
        if prog and prog[-1][0] == 'empty':
            return None
        segs = [i for i, (k_, _d) in enumerate(prog) if k_ == 'seg']
        # slope of each segment's length (the groups' common slot values)
        slope = []
        for j in range(len(pattern) - 1):
            slope.append(v_dir[pattern[j + 1][0]] - v_dir[pattern[j][0]])
        moving = [q for q, sl_ in enumerate(slope) if sl_ != 0.0]
        if not moving:
            out['const'] = const * self._run(prog, gv, False, float('-inf'))
            return out
        if len(moving) > 1:
            raise SSIneligible('several_lags', f'{len(moving)} segments move')
        q = moving[0]
        i = segs[q]
        dt = self.model.dtype
        seg = prog[i][1]
        # the moving segment's length at the anchor and its slope in s
        L_anchor = (v_base[pattern[q + 1][0]] - v_base[pattern[q][0]]
                    + slope[q] * t_anchor)
        beta = slope[q] * sigma
        if L_anchor < -1e-12 * max(1.0, abs(t_anchor)):
            raise SSIneligible('several_lags', 'the chamber does not start '
                               'at the anchor')
        L_anchor = max(L_anchor, 0.0)
        x_before = self._run(prog, gv, False, float('-inf'), upto=i)
        x0 = seg.stack(x_before, dt)
        if L_anchor > 0:
            x0 = seg.action(x0, L_anchor, dt)
        if i == len(prog) - 2 and prog[-1][0] == 'event':
            k = seg.index.get(self.full)
            last = [f for I, f in prog[-1][1] if I == self.full]
            if k is None or not last:
                return None
            f = last[0]
            if f.starts or f.out:
                raise SSIneligible('several_lags', 'edges open at the last '
                                   'time')
            cten = (np.einsum(*f.args, list(f.act_in)) if f.args
                    else np.ones((), dtype=dt))
            c = np.zeros(seg.D, dtype=np.result_type(cten, dt))
            c[seg.offs[k]:seg.offs[k] + seg.sizes[k]] = np.ravel(cten)
        else:
            c = self._downstream(prog, gv, i, seg, dt)
        out.update(seg=seg, beta=float(beta), x0=x0, c=c)
        return out

    def me_chain(self, free_base, free_dir, t_anchor, t_ref, t_end,
                 tie_ctx=None):
        """:meth:`me_line` on a FINITE chamber ``t`` between ``t_anchor`` and
        ``t_end`` (the one containing ``t_ref``) where several segments may
        move (item 54: a k >= 3 slice whose swept leg lies between two other
        legs: the segment before it grows while the one after it shrinks)::

            I(s) = const * prod_r g_r(rho0_r + rho1_r s)
                   * c . e^{G_M L_M(s)} M_{M-1} ... M_1 e^{G_1 L_1(s)} . x0,

            L_j(s) = L0_j + beta_j s,   0 <= s <= s_max = |t_end - t_anchor|.

        ``None`` when the integral vanishes on the chamber; otherwise a dict
        with ``const`` and ``rows`` as :meth:`me_line` (a moving constant row
        may shrink here, ``rho1 < 0``, as long as it stays >= 0 on the
        chamber) and ``chain`` (``None`` when no segment moves) =
        ``{'segs': [_Segment], 'betas', 'L0s', 'x0'`` (the stacked state
        entering the first moving segment), ``'maps'`` (dense ``M_j``: the
        events and fixed segments between two moving segments, from one
        stacked space to the next), ``'c'`` (the functional of everything
        after the last moving segment)}.  Nothing here changes a value; the
        sweep is untouched."""
        FI = _FI()
        try:
            t_min = float(FI.PHASE_J_T_MIN)
        except (TypeError, ValueError):
            t_min = None
        if t_min != float('-inf') or switch_on_active():
            raise SSIneligible('finite_t_min', repr(
                _switch_on_t_min() if switch_on_active()
                else FI.PHASE_J_T_MIN))
        if self.empty or any(e.kind == 'zero' for e in self.edge_ops) or any(
                r[4].kind == 'zero' for r in self.const_rows):
            return None
        fb = [float(x) for x in free_base]
        fd = [float(x) for x in free_dir]
        t_anchor, t_ref, t_end = float(t_anchor), float(t_ref), float(t_end)
        s_max = abs(t_end - t_anchor)
        if not math.isfinite(s_max):
            raise SSIneligible('several_lags', 'me_chain needs a finite '
                               'chamber')
        tol = 1e-12 * max(1.0, abs(t_anchor), abs(t_end))
        sigma = 1.0 if t_ref >= t_anchor else -1.0
        f_ref = [b + t_ref * d for b, d in zip(fb, fd)]
        const = 1.0
        rows = []
        for (_idx, a_int, a_ext, c0, ent) in self.const_rows:
            if FI._const_row_verdict(a_int, a_ext, c0, f_ref, 'edge',
                                     tie_ctx) == 'EMPTY':
                return None
            r0 = float(c0) + sum(float(a) * b for a, b in zip(a_ext, fb))
            r1 = sum(float(a) * d for a, d in zip(a_ext, fd))
            if r1 == 0.0:
                const = const * ent.impulse(r0)
                if const == 0:
                    return None
                continue
            rho0, rho1 = r0 + r1 * t_anchor, r1 * sigma
            rho_end = rho0 + rho1 * s_max
            if rho0 <= tol and rho_end <= tol:
                return None             # retarded on the whole chamber
            if rho0 < -tol or rho_end < -tol:
                raise SSIneligible('row_crosses_zero',
                                   f'rho0={rho0:g}, rho1={rho1:g}, '
                                   f's_max={s_max:g}')
            rows.append((max(rho0, 0.0), rho1, ent))
        out = {'const': const, 'rows': rows, 'chain': None}
        if self.m == 0:
            return out
        v_ref = self._slot_values(f_ref)
        v_base = self._slot_values(fb)
        v_dir = []
        for key in self.slots:
            v = 0.0
            for a_, d_ in zip(key[:-1], fd):
                if a_ != 0.0 and d_ != 0.0:
                    v = v + a_ * d_
            v_dir.append(v)
        pattern, gv = self._pattern(v_ref)
        if any(v_dir[s_] != v_dir[g[0]] for g in pattern for s_ in g):
            raise SSIneligible('several_lags', 't_ref is on a chamber '
                               'boundary (tied slots move apart)')
        prog = self._checked_program(pattern, False)
        if prog and prog[-1][0] == 'empty':
            return None
        segs = [i for i, (k_, _d) in enumerate(prog) if k_ == 'seg']
        slope = [v_dir[pattern[j + 1][0]] - v_dir[pattern[j][0]]
                 for j in range(len(pattern) - 1)]
        moving = [q for q, sl_ in enumerate(slope) if sl_ != 0.0]
        if not moving:
            out['const'] = const * self._run(prog, gv, False, float('-inf'))
            return out
        dt = self.model.dtype
        idx = [segs[q] for q in moving]
        msegs = [prog[i][1] for i in idx]
        L0s, betas = [], []
        for q in moving:
            L0 = (v_base[pattern[q + 1][0]] - v_base[pattern[q][0]]
                  + slope[q] * t_anchor)
            beta = slope[q] * sigma
            if L0 < -tol or L0 + beta * s_max < -tol:
                raise SSIneligible('several_lags', 'a moving segment is '
                                   'negative on the chamber')
            L0s.append(max(L0, 0.0))
            betas.append(float(beta))
        x_before = self._run(prog, gv, False, float('-inf'), upto=idx[0])
        x0 = msegs[0].stack(x_before, dt)
        maps = [self._between(prog, gv, idx[j], idx[j + 1], msegs[j],
                              msegs[j + 1], dt)
                for j in range(len(idx) - 1)]
        i, seg = idx[-1], msegs[-1]
        if i == len(prog) - 2 and prog[-1][0] == 'event':
            k = seg.index.get(self.full)
            last = [f for I, f in prog[-1][1] if I == self.full]
            if k is None or not last:
                return None
            f = last[0]
            if f.starts or f.out:
                raise SSIneligible('several_lags', 'edges open at the last '
                                   'time')
            cten = (np.einsum(*f.args, list(f.act_in)) if f.args
                    else np.ones((), dtype=dt))
            c = np.zeros(seg.D, dtype=np.result_type(cten, dt))
            c[seg.offs[k]:seg.offs[k] + seg.sizes[k]] = np.ravel(cten)
        else:
            c = self._downstream(prog, gv, i, seg, dt)
        out['chain'] = {'segs': msegs, 'betas': betas, 'L0s': L0s,
                        'x0': x0, 'maps': maps, 'c': c}
        return out

    def _between(self, prog, gv, i_from, i_to, seg_from, seg_to, dt):
        """The dense linear map (``seg_to.D x seg_from.D``) of the program
        steps strictly between steps ``i_from`` and ``i_to`` (events and
        fixed segments), from ``seg_from``'s stacked space to ``seg_to``'s:
        one run per basis vector (``me_chain``)."""
        n_ev = sum(1 for k_, _d in prog[:i_from + 1] if k_ == 'event')
        _check_dense_map(seg_to.D, seg_from.D)
        M = np.zeros((seg_to.D, seg_from.D), dtype=complex)
        for qb in range(seg_from.D):
            e = np.zeros(seg_from.D, dtype=dt)
            e[qb] = 1.0
            x = seg_from.unstack(e)
            j = n_ev
            prev = gv[j - 1] if j else None
            for kind, data in prog[i_from + 1:i_to]:
                if kind == 'seg':
                    x = data.propagate(x, gv[j] - prev, dt)
                elif kind == 'event':
                    nx = {}
                    for I, f in data:
                        xI = x.get(I)
                        if xI is not None:
                            nx[I] = f(xI)
                    x = nx
                    prev = gv[j]
                    j += 1
                else:
                    x = {}
                    break
            if x:
                M[:, qb] = seg_to.stack(x, complex)
        if np.isrealobj(np.zeros(0, dtype=dt)) and not np.any(M.imag):
            M = M.real
        return M

    def _downstream(self, prog, gv, i, seg, dt):
        """The linear functional ``state after step i -> value`` of the rest
        of a program (events and fixed segments), on ``seg``'s stacked
        space: one run per basis vector."""
        n_ev = sum(1 for k_, _d in prog[:i + 1] if k_ == 'event')
        c = np.zeros(seg.D, dtype=complex)
        for qb in range(seg.D):
            e = np.zeros(seg.D, dtype=dt)
            e[qb] = 1.0
            x = seg.unstack(e)
            j = n_ev
            prev = gv[j - 1] if j else None
            # the event closing the moving segment is at gv[j]
            for kind, data in prog[i + 1:]:
                if kind == 'seg':
                    x = data.propagate(x, gv[j] - prev, dt)
                elif kind == 'event':
                    nx = {}
                    for I, f in data:
                        xI = x.get(I)
                        if xI is not None:
                            nx[I] = f(xI)
                    x = nx
                    prev = gv[j]
                    j += 1
                else:
                    x = {}
                    break
            v = x.get(self.full)
            c[qb] = 0.0 if v is None else v.item()
        if np.isrealobj(np.zeros(0, dtype=dt)) and not np.any(c.imag):
            c = c.real
        return c


# ─── τ-grid prefetch (batching through the per-point API) ─────────────────

_PREFETCH = threading.local()


_PREFETCH_TOKENS = itertools.count(1)


class tau_grid_prefetch:
    """Within this context, the first call of a state-space subset with a
    given Wick permutation (its tie context's leg mapping) evaluates the
    whole registered grid of raw external-time tuples at once
    (``SubsetSS.evaluate_many``) and later calls of the grid are served
    from that memo.  ``points``: the k-tuples of raw external times exactly
    as they will be passed to ``contribution(*times)``.  The values differ
    from per-point evaluation only by rounding (the last segment is
    stepped from one length to the next).  Per thread; nests.  The memos
    (two arrays per subset and leg mapping: 24 B per point) are released
    when the context exits (S3)."""

    def __init__(self, points):
        self.points = [tuple(float(x) for x in p) for p in points]
        self.index = {}
        for i, p in enumerate(self.points):
            self.index.setdefault(p, i)
        self.subsets = []

    def __enter__(self):
        self._prev = getattr(_PREFETCH, 'grid', None)
        self.token = next(_PREFETCH_TOKENS)
        _PREFETCH.grid = self
        return self

    def __exit__(self, *exc):
        _PREFETCH.grid = self._prev
        for sub in self.subsets:
            with sub._lock:
                sub._prefetch.pop(self.token, None)
        self.subsets = []
        return False


def _prefetched(sub, free_vals, tie_ctx, t_min):
    """``(value, error estimate)`` of a registered grid point (the value a
    ``_Refused`` for a point the route refused), or ``None``."""
    grid = getattr(_PREFETCH, 'grid', None)
    if grid is None or tie_ctx is None or t_min != float('-inf'):
        return None
    try:
        times = tuple(float(x) for x in tie_ctx.times)
    except (TypeError, ValueError):
        return None
    i = grid.index.get(times)
    if i is None:
        return None
    mapping = (tuple(tie_ctx.free_legs), tie_ctx.origin_leg)
    with sub._lock:
        memos = sub._prefetch.get(grid.token)
        if memos is None:
            memos = sub._prefetch[grid.token] = {}
            grid.subsets.append(sub)
        memo = memos.get(mapping)
    if memo is None:
        points = grid.points
        pts = []
        for T in points:
            if len(T) != len(times):
                return None
            o = 0.0 if tie_ctx.origin_leg is None else T[tie_ctx.origin_leg]
            fv = [T[leg] - o for leg in tie_ctx.free_legs]
            pts.append((fv, type(tie_ctx)(T, tie_ctx.free_legs,
                                          tie_ctx.origin_leg)))
        errs = []
        vals = sub.evaluate_many(pts, errors=errs)
        refused = {j: v for j, v in enumerate(vals) if isinstance(v, _Refused)}
        arr = np.array([0.0 if isinstance(v, _Refused) else v for v in vals],
                       dtype=complex)
        memo = (arr, np.asarray(errs, dtype=float), refused)
        with sub._lock:
            memos[mapping] = memo
    arr, errs, refused = memo
    _count('ss_prefetch_hits')
    r = refused.get(i)
    if r is not None:
        return r, 0.0
    v = arr[i]
    return (v.real if v.imag == 0 else complex(v)), float(errs[i])


def _late_errors():
    """The exceptions a routed evaluation may fall back on: the route's own
    late refusals, a segment over the memory budget, a numerical failure.
    (A ``ValueError`` -- non-finite times, ``PHASE_J_T_MIN`` after an
    external time -- is the caller's and is raised.)"""
    return (SSLateRefusal, _FI().PhaseJResourceError, np.linalg.LinAlgError,
            ArithmeticError)


def _late_reason(exc):
    if isinstance(exc, SSLateRefusal):
        return exc.reason
    if isinstance(exc, _FI().PhaseJResourceError):
        return 'segment_memory'
    return 'numerical_' + type(exc).__name__


_WARNED_OUTSIDE = set()


def _warn_outside_run(reason):
    """A fallback outside a ``compute_cumulants`` run (a returned callable
    evaluated later, e.g. ``dd.run``'s k >= 3 slices) is not in a run's
    census: warn once per process and reason (never silent)."""
    FI = _FI()
    with FI._PLAN_MEM_LOCK:
        inside = FI._PLAN_MEM['active'] > 0
    if inside or reason in _WARNED_OUTSIDE:
        return
    _WARNED_OUTSIDE.add(reason)
    import warnings
    from api.compute import PhaseJEngineFallbackWarning
    from api.save import _ensure_warning_shown
    _msg = (f'Phase J: the state-space route fell back on the pole route '
            f'({reason}) while a returned callable was evaluated; the '
            f'counters ss_fallback_* have the details (warned once per '
            f'reason)')
    _ensure_warning_shown(PhaseJEngineFallbackWarning, 'always',
                          message=_msg, stacklevel=4)
    warnings.warn(_msg, PhaseJEngineFallbackWarning, stacklevel=4)


def _late_detail(exc):
    d = str(exc)
    pre = 'state-space route: '
    if d.startswith(pre):
        d = d[len(pre):]
    return d.split(f'; set {ENGINE_ENV}=poles')[0]


def _count_fallback(reason, first):
    C = _counters()
    C['ss_fallback_evals'] = C.get('ss_fallback_evals', 0) + 1
    d = C.setdefault('ss_fallback_by_reason', {})
    d[reason] = d.get(reason, 0) + 1
    if first:
        C['ss_fallback_subsets'] = C.get('ss_fallback_subsets', 0) + 1


# ─── the dispatch point's maker ───────────────────────────────────────────

def _edge_entry(model, spec, pi_ri):
    """The ``_EntryOps`` of one smooth row (``spec``: ``None`` = the plain
    propagator entry ``pi_ri``; ``('kernel', att)``; ``('dress', pi, ri,
    att)``; ``('unsupported', reason)``), or a decline reason (str)."""
    if spec is None:
        e = model.entry(*pi_ri)
        return 'entry_missing' if e is None else e
    kind = spec[0]
    if kind == 'kernel':
        return model.kernel_entry(spec[1])
    if kind == 'dress':
        return model.dressed_entry(spec[1], spec[2], spec[3])
    return spec[1] if kind == 'unsupported' else 'non_propagator_rows'


def subset_maker(*, propagator_data, num_params, pi_ri, plain_edges,
                 prefactor, cdata, m, row_kinds, n_free, modes=None,
                 edge_specs=None):
    """Called by ``integrate_diagram`` for each subset unless
    ``DAEDALUS_PHASE_J_ENGINE=poles``.  Returns ``None`` when the route is
    off or the subset is not eligible (counted by reason in
    ``ss_declined_by_reason``); otherwise a function with the signature of
    ``integrate_diagram``'s ``_make_subset_contrib`` (plus ``fallback``, the
    lazy builder of the pole-route closure) that builds the subset closure
    ``_contrib(free_vals, _hook_ctx=None, _tie_ctx=None)``.  ``edge_specs``
    (parallel to ``pi_ri``; ``None``: all plain propagator rows): the
    conductance kernel rows and dressed lines (``_edge_entry``)."""
    if not engine_requested():
        if switch_on_active():
            raise SwitchOnUnsupported('engine_poles',
                                      f'{ENGINE_ENV}=poles')
        return None
    FI = _FI()
    if not plain_edges:
        return _decline('non_propagator_rows')
    if prefactor is None:
        return _decline('symbolic_prefactor')
    if row_kinds is not None and any(k_ not in ('edge', 'conv_pseudo')
                                     for k_ in row_kinds):
        return _decline('non_edge_rows')
    if len(cdata) != len(pi_ri):
        return _decline('row_count')
    if edge_specs is not None and len(edge_specs) != len(pi_ri):
        return _decline('row_count')
    model, why = realization_for(propagator_data, num_params)
    if model is None:
        return _decline(why)
    ents = []
    for i_, pr in enumerate(pi_ri):
        e = _edge_entry(model, None if edge_specs is None else edge_specs[i_],
                        pr)
        if isinstance(e, str):
            return _decline(e)
        ents.append(e)
    try:
        sub = SubsetSS(model, ents, cdata, m, n_free)
    except SSIneligible as exc:
        return _decline(exc.reason)
    try:
        if not switch_on_active():  # a switch-on run never builds them
            _charge(sub)
    except FI.PhaseJResourceError:
        # the stationary states do not fit: the pole route takes the subset
        # when ITS plan would (the running guard then charges it), else the
        # state-space message is the one to read
        if modes is None or not _pole_plan_fits(modes):
            raise
        return _decline('stationary_over_budget')
    _count('ss_subsets_built')
    pref = complex(prefactor)
    edge_modes = modes            # the dispatch passes its own (None) modes
    # ``compute_cumulants`` skipped the pole-cluster deformation for this
    # run (the route covers it): the undeformed poles cannot serve a
    # fallback, which then raises with this text
    no_fallback = (propagator_data.get('ss_no_pole_fallback')
                   if hasattr(propagator_data, 'get') else None)

    def make(fc, cdata_, m_val, modes=None, pref_c=None, pole_tuples=None,
             plan=None, hook_meta=None, row_kinds=None, theta0_empty=False,
             fallback=None):
        pole = []
        pole_lock = threading.Lock()

        def _pole_route(exc):
            reason = _late_reason(exc)
            if switch_on_active():
                raise SwitchOnUnsupported(
                    reason, f'refused at call time: {exc}') from exc
            with pole_lock:
                first = not pole
                if first:
                    if not fallback_enabled():
                        raise exc
                    if no_fallback:
                        raise SSNoPoleFallback(
                            f'state-space route refused this subset '
                            f'({reason}: {exc}) and {no_fallback}') from exc
                    built = None if fallback is None else fallback(
                        dict(hook_meta or {}, ss_fallback=reason))
                    if built is None:
                        raise SSNoPoleFallback(
                            f'state-space route refused this subset '
                            f'({reason}: {exc}) and it has no pole route '
                            f'(a kernel only the state-space route '
                            f'realizes); nothing to fall back on') from exc
                    pole.append(built)
                    _warn_outside_run(reason)
                _count_fallback(reason, first)
                return pole[0]

        def _contrib(free_vals, _hook_ctx=None, _tie_ctx=None):
            if theta0_empty:
                FI._RUNTIME_COUNTERS['theta0_subsets_pruned'] += 1
                return 0.0 + 0.0j
            t_min = _t_min_frame(free_vals, _tie_ctx)
            try:
                hit = _prefetched(sub, free_vals, _tie_ctx, t_min)
                if hit is None:
                    val = sub.evaluate(free_vals, _tie_ctx, t_min)
                    err = _EST.err
                else:
                    val, err = hit
                    if isinstance(val, _Refused):
                        raise SSLateRefusal(val.reason, val.detail)
                val = complex(pref * val)
                if not (math.isfinite(val.real) and math.isfinite(val.imag)):
                    raise SSLateRefusal('non_finite', 'a non-finite value')
            except _late_errors() as exc:
                return _pole_route(exc)(free_vals, _hook_ctx, _tie_ctx)
            C = _counters()
            C['ss_evals'] = C.get('ss_evals', 0) + 1
            C['ss_estimated'] = C.get('ss_estimated', 0) + 1
            C['ss_err_abs_sum'] = C.get('ss_err_abs_sum', 0) + abs(pref) * err
            hook = FI._SUBSET_HOOK
            if hook is not None:
                # the documented payload (``plan`` / ``pole_tuples``: none;
                # ``integrand``: the closure the pole routes would hand to
                # quadrature, for references) plus ``ss_subset``
                FI._emit_subset_hook(
                    lambda p: hook(dict(p, ss_subset=sub)), hook_meta,
                    _hook_ctx, free_vals, m_val, cdata_,
                    path='ss', evaluator='statespace.SubsetSS', branch='ss',
                    value=val, bail_reason=None, attempted='ss',
                    modes=edge_modes, prefactor=pref, plan=None,
                    pole_tuples=None, integrand=fc,
                    row_kinds=row_kinds, tie_ctx=_tie_ctx)
            return val
        _contrib.ss_subset = sub
        # C0 output (S4, ``api.me_output``): what the closure multiplies the
        # subset integral by, and whether it is pruned to 0
        _contrib.ss_prefactor = pref
        _contrib.ss_theta0_empty = bool(theta0_empty)
        return _contrib
    return make


def _pole_plan_fits(modes):
    """Whether the pole route's modesum plan for a subset with these edge
    mode sums would fit the running budget (``_plan_guard``'s test)."""
    FI = _FI()
    try:
        n = 1
        for ms in modes:
            n *= max(1, len(ms.modes))
    except Exception:
        return False
    nbytes = n * FI.PLAN_BYTES_PER_TUPLE
    with FI._PLAN_MEM_LOCK:
        _scoped, budget, _rule, used = FI._budget_now()
    return budget is None or used + nbytes <= budget


def _t_min_frame(free_vals, tie_ctx):
    """``final_integral.PHASE_J_T_MIN`` -- or, inside
    :func:`switch_on_scope`, the thread's switch-on bound -- in the frame of
    one evaluation (as ``final_integral._t_min_frame``, without its
    box-free-route requirement, which concerns the pole routes only)."""
    FI = _FI()
    raw = _switch_on_t_min()
    if raw is None:
        raw = FI.PHASE_J_T_MIN
    try:
        t = float(raw)
    except (TypeError, ValueError):
        raise ValueError(f'final_integral.PHASE_J_T_MIN={raw!r}: expected '
                         f'a float') from None
    if t == float('-inf'):
        return t
    if not math.isfinite(t):
        raise ValueError(f'final_integral.PHASE_J_T_MIN={t!r}: expected '
                         f'-inf or a finite float')
    t0 = 0.0
    if tie_ctx is not None and tie_ctx.origin_leg is not None:
        t0 = float(tie_ctx.times[tie_ctx.origin_leg])
    t_rel = t - t0
    if t_rel > min([0.0] + [float(x) for x in free_vals]):
        raise ValueError(
            f'final_integral.PHASE_J_T_MIN={t!r} is after an external time '
            f'(the cumulant is asked for before the switch-on)')
    return t_rel


def _check_transient(sub, acts):
    """Refuse (``PhaseJResourceError``) a segment system whose working set
    (the Taylor action's vectors, and the dense generator when it is formed)
    alone would exceed ``PHASE_J_MEM_BUDGET_GB``, before it is built."""
    FI = _FI()
    with FI._PLAN_MEM_LOCK:
        _scoped, budget, rule, _used = FI._budget_now()
    if budget is None:
        return
    item = 8 if sub.model.is_real else 16
    D = 0
    for act in acts:
        sz = 1
        for e in act:
            sz *= sub.edge_ops[e].n
        D += sz
    nbytes = 6 * D * item + (2 * D * D * item if D <= OP_DENSE_MAX else 0)
    if nbytes > budget:
        raise FI.PhaseJResourceError(
            f'Phase J memory budget exceeded: one state-space segment system '
            f'has {D:,} states ({len(acts)} downsets), a working set of '
            f'{FI._fmt_gb(nbytes)} over the budget of '
            f'{FI._fmt_budget(budget, rule)}.  '
            f'Nothing was allocated for it.  Options: {FI._BUDGET_HOW}.',
            {'bytes': nbytes, 'states': D, 'budget_bytes': budget})


def _check_dense_map(d_to, d_from):
    """Refuse (``PhaseJResourceError``) a dense ``d_to x d_from`` complex map
    of a C0 chain output (``_between``) larger than the budget,
    before it is allocated (uncharged above ``OP_DENSE_MAX`` states: 6.4 GB
    at D = 2e4; audit MEM-07)."""
    FI = _FI()
    with FI._PLAN_MEM_LOCK:
        _scoped, budget, rule, _used = FI._budget_now()
    nbytes = 16 * int(d_to) * int(d_from)
    if budget is None or nbytes <= budget:
        return
    raise FI.PhaseJResourceError(
        f'Phase J memory budget exceeded: the dense map between two moving '
        f'segments of a chain output ({int(d_to):,} x {int(d_from):,} '
        f'states) needs {FI._fmt_gb(nbytes)}, over the budget of '
        f'{FI._fmt_budget(budget, rule)}.  Nothing was allocated for it.  '
        f'Options: {FI._BUDGET_HOW}.',
        {'bytes': nbytes, 'states': (int(d_to), int(d_from)),
         'budget_bytes': budget})


def _charge(sub):
    """Charge the subset's stationary-state bytes against item 35's running
    budget (``PHASE_J_MEM_BUDGET_GB``) before anything is allocated."""
    FI = _FI()
    nbytes = sub.state_bytes()
    with FI._PLAN_MEM_LOCK:
        scoped, budget, rule, used = FI._budget_now()
        if budget is not None and used + nbytes > budget:
            w, b = sub.max_width()
            run = f' of {FI._PLAN_MEM["what"]}' if (
                scoped and FI._PLAN_MEM['what']) else ''
            raise FI.PhaseJResourceError(
                f'Phase J memory budget exceeded{run}: the state-space '
                f'route would hold {FI._fmt_gb(nbytes)} of stationary states '
                f'for one subset ({len(sub._stat_downsets)} downsets, '
                f'widest cut {w} edges, largest block {b:,} entries) on top '
                f'of {FI._fmt_gb(used)}, over the budget of '
                f'{FI._fmt_budget(budget, rule)}.  Nothing was allocated '
                f'for it.  Options: {FI._BUDGET_HOW}.',
                {'bytes': nbytes, 'bytes_so_far': used,
                 'budget_bytes': budget, 'width': w, 'block': b})
        if scoped:
            FI._PLAN_MEM['used'] += nbytes


# ─── route prediction and the memory pre-pass (S3) ────────────────────────

def model_route(propagator_data, num_params):
    """``(True, None)`` when the state-space route can take this model's
    subsets (the engine is on, the Itô Θ(0) rule, a realization accepted
    and Hurwitz), else ``(False, reason)``.  Builds (and caches) the
    realization; a diagram can still decline (``diagram_route``) and a
    subset (its row shapes, ``subset_maker``)."""
    if not engine_requested():
        return False, 'engine_poles'
    # (the realization never reads the poles and residues, so a repeated
    # pole, ``jordan_incomplete``, does not matter here)
    ops, why = realization_for(propagator_data, num_params)
    return (True, None) if ops is not None else (False, why)


def diagram_route(typed_diagram, propagator_data, num_params):
    """``(True, None)`` when every vertex of the diagram is one the route
    takes (no noise-kernel source, every conductance kernel realizable),
    else ``(False, reason)``; ``model_route`` must hold too."""
    from engine.core.vertices import NoiseSourceType, ConvVertexType
    model, why = realization_for(propagator_data, num_params)
    if model is None:
        return False, why
    va = getattr(typed_diagram, 'vertex_assignments', None) or {}
    for _v, vt in va.items():
        if isinstance(vt, NoiseSourceType) and vt.cumulant_specs:
            return False, 'noise_kernel_box'
        if isinstance(vt, ConvVertexType):
            for att in vt.kernel_attachments or ():
                if _FI()._conv_kernel_local_weight(att, num_params) is not None:
                    continue
                e = model.kernel_route(att)
                if isinstance(e, str):
                    return False, e
    return True, None


def run_route(propagator_data, num_params, vertex_types=()):
    """The run-level prediction of ``compute_cumulants`` (S3): whether the
    route takes every subset of a run whose diagrams are built from
    ``vertex_types`` (the model's vertex and source types): the model is
    routable (``model_route``) and no vertex type is one the route declines
    (a noise-kernel source; a conductance kernel it cannot realize).  A
    subset can still decline on its rows (offset rows, a vertex after the
    last external time, a symbolic prefactor): the caller checks the run's
    census.  Returns ``{'engine', 'covers_run', 'reason'}``."""
    from engine.core.vertices import NoiseSourceType, ConvVertexType
    eng = engine_name()
    ok, why = model_route(propagator_data, num_params)
    if ok:
        model, _w = realization_for(propagator_data, num_params)
        for vt in vertex_types or ():
            if isinstance(vt, NoiseSourceType) and vt.cumulant_specs:
                ok, why = False, 'noise_kernel_box'
                break
            if isinstance(vt, ConvVertexType):
                for att in vt.kernel_attachments or ():
                    if _FI()._conv_kernel_local_weight(
                            att, num_params) is not None:
                        continue
                    e = model.kernel_route(att)
                    if isinstance(e, str):
                        ok, why = False, e
                        break
                if not ok:
                    break
    return {'engine': eng, 'covers_run': bool(ok), 'reason': why}


#: downsets enumerated per diagram by the pre-pass estimate before it falls
#: back to the 2^|V| bound
PREPASS_MAX_DOWNSETS = 20000
#: the pre-pass takes the closed bound ``2^(internal vertices) x prod n_e``
#: of a diagram's stationary states when it is at most this many bytes,
#: else it enumerates the downsets
PREPASS_CHEAP_BYTES = 1 << 16
#: bytes one τ-batch memo holds per subset, grid point and Wick leg mapping
#: (a complex value and a float estimate; released after the batch)
PREFETCH_BYTES_PER_POINT = 24


def diagram_estimate(typed_diagram, propagator_data, num_params, has_delta,
                     n_tau=1, n_perm=1):
    """Upper bound (bytes) of what the route holds for one diagram (item
    35's pre-pass, S3; item 61): ``(stationary, transient)``.

    The diagram DAG (prediagram edges ``u -> v``: ``Δt = t_v - t_u``; the
    leaves, the external times, are its sinks) is rebuilt as Phase J builds
    it (``_line_layouts``): a line into a conductance vertex's kernel slot
    is either DRESSED (one edge carrying the cascade's states,
    ``_ModelOps.dressed_entry``, no δ part, no extra vertex: the kernel has
    separated modes and dressing is on, ``final_integral`` section 2c) or on
    the LEG path (one more vertex ``s``: the propagator edge ``u -> s``
    with its δ part, then the kernel edge ``s -> v``).  Each distinct
    placement of the kernel slots is built (and held) separately, so the
    placements add.  Per δ-subset (every subset of the edges with a δ
    part), the δ edges are contracted (a downset of the contracted DAG is
    one of the original containing both ends or neither; cycles merged)
    and a state block indexed by a downset ``I`` has ``b(I)`` = the product
    of the per-edge state counts ``n_e`` over the smooth edges crossing it;
    the stationary states of the subset are ``sum b(I)`` over the downsets
    without a leaf, held for the run, with the τ batch's ``n_tau x n_perm``
    values.  A subset that keeps an edge without states smooth is never
    built.  Before enumerating, the closed bound ``2^(edges with δ) x
    2^(vertices) x prod n_e`` is taken when small (``PREPASS_CHEAP_BYTES``);
    beyond ``PREPASS_MAX_DELTA_EDGES`` δ edges the subsets are bounded by
    the uncontracted DAG.  ``transient``: one segment system at a time, at
    most ``D = sum b(I)`` over all downsets of a subset (``6 D`` vectors
    plus the dense ``2 D^2`` generator, ``D`` capped at ``OP_DENSE_MAX``).
    ``has_delta(pi, ri)``: whether the entry has a δ part (the pre-pass's
    own test).  ``None`` when the diagram is not covered."""
    model, _why = realization_for(propagator_data, num_params)
    if model is None:
        return None
    layouts = _line_layouts(typed_diagram, model, propagator_data,
                            num_params, has_delta)
    if layouts is None:
        return None
    item = 8 if model.is_real else 16
    stat = trans = 0
    for (nv, leaf_mask, edges, mult) in layouts:
        s_, t_ = _layout_estimate(nv, leaf_mask, edges, item, n_tau, n_perm)
        stat += mult * s_
        trans = max(trans, t_)
    return stat, trans


#: placements of a diagram's kernel slots enumerated by the pre-pass before
#: it charges every slot-capable line as a kernel line in one layout
PREPASS_MAX_PLACEMENTS = 256
#: δ edges per layout whose subsets the pre-pass contracts one by one
PREPASS_MAX_DELTA_EDGES = 12


def _line_layouts(typed_diagram, model, propagator_data, num_params,
                  has_delta):
    """The DAGs Phase J builds for one typed diagram, one per placement of
    its kernel slots: ``[(nv, leaf_mask, edges, multiplicity)]`` with
    ``edges`` = ``[(u, v, n_e, delta)]``, or ``None`` when a kernel is not
    realizable.
    Mirrors ``final_integral.integrate_diagram`` section 2c (slots per leg
    from the coefficient's degree in the kernel symbol, placements as
    permutations, a local-weight kernel leaves the line as it is)."""
    from engine.core.vertices import ConvVertexType
    from sage.all import SR
    FI = _FI()
    D_ = typed_diagram.prediagram[0]
    verts = list(D_.vertices())
    index = {v: i for i, v in enumerate(verts)}
    leaf_mask = 0
    for v in set(typed_diagram.prediagram[2]):
        if v in index:
            leaf_mask |= 1 << index[v]
    can_dress = (propagator_data.get('C_mats') is not None
                 and FI._dress_kernels_on())
    def kernel_info(att):
        # cached on the model (one parameter point): the mode extraction is
        # symbolic
        key = ('est_kernel_info', _att_key(att), can_dress)
        with model._lock:
            hit = model._extra.get(key)
        if hit is None:
            if FI._conv_kernel_local_weight(att, num_params) is not None:
                hit = ('local',)
            else:
                # (item 56: a kernel with a direct term c delta is taken
                # dressed only; its dressed line keeps c x the line's delta)
                e = model.kernel_route(att)
                if isinstance(e, str):
                    hit = ('refused', e)
                else:
                    n_k = int(e.n_states)
                    km = (FI._kernel_modes(att, num_params) if can_dress
                          else None)
                    direct = 0
                    if km is None and can_dress:
                        split = FI._kernel_direct_split(att, num_params)
                        if split is not None:
                            direct, km = split
                    hit = (('dress', n_k, tuple(complex(mu) for (_c, mu) in km),
                            direct != 0)
                           if km is not None else ('leg', n_k))
            with model._lock:
                model._extra[key] = hit
        return hit

    edge_types = typed_diagram.edge_types
    va = getattr(typed_diagram, 'vertex_assignments', None) or {}
    conv = [(v, vt) for v, vt in va.items()
            if isinstance(vt, ConvVertexType) and vt.kernel_attachments]
    for _v, vt in conv:
        for att in vt.kernel_attachments:
            if kernel_info(att)[0] == 'refused':
                return None
    # item 60: a coefficient that sums terms with different powers of the
    # kernel symbols is integrated once per choice of one term (grouped by
    # those powers) per vertex, each with exactly its power's slots (a plain
    # term none), and all of them are held (``_conv_monomial_split``)
    powers = {}
    for v, vt in conv:
        syms = []
        for att in vt.kernel_attachments:
            if att['symbol'] not in syms:
                syms.append(att['symbol'])
        e = SR(vt.coefficient).expand()
        op = e.operator()
        terms = (e.operands() if op is not None
                 and getattr(op, '__name__', '') == 'add_vararg' else [e])
        try:
            keys = sorted({tuple(int(t.degree(SR(z))) for z in syms)
                           for t in terms})
        except Exception:
            keys = [None]
        powers[v] = (syms, keys)
    split = any(len(k) > 1 for _s, k in powers.values())
    combos = (list(itertools.product(*[powers[v][1] for v, _vt in conv]))
              if split else [None])
    n_combos = len(combos)
    if n_combos > PREPASS_MAX_PLACEMENTS:
        combos = [None]               # bounded below: the whole coefficient

    def slot_choices(combo):
        """Per (vertex, leg): the placements of its kernel slots."""
        choices = []
        for ci, (v, vt) in enumerate(conv):
            phys_legs = [tuple(lg) for lg in vt.physical_legs]
            key = None if combo is None else combo[ci]
            syms = powers[v][0]
            coeff = SR(vt.coefficient)
            by_leg = {}
            for idx, att in enumerate(vt.kernel_attachments):
                if key is not None:
                    nq = max(key[syms.index(att['symbol'])], 0)
                else:
                    try:
                        q = int(coeff.degree(SR(att['symbol'])))
                    except Exception:
                        q = 1
                    nq = max(q, 1)
                by_leg.setdefault(tuple(att['leg']), []).extend([idx] * nq)
            for leg, slots in by_leg.items():
                n_copies = phys_legs.count(leg)
                hits = sorted(ek for ek in D_.edges() if ek[1] == v
                              and tuple(edge_types[ek][1]) == leg)
                if len(slots) > n_copies or len(hits) != n_copies:
                    return None         # Phase J refuses the diagram
                slots = tuple(slots) + (None,) * (n_copies - len(slots))
                pls = sorted(set(itertools.permutations(slots)),
                             key=lambda pl: tuple(-1 if x is None else x
                                                  for x in pl))
                choices.append([{ek: vt.kernel_attachments[i]
                                 for ek, i in zip(hits, pl) if i is not None}
                                for pl in pls])
        return choices

    plan = []                         # (placements, multiplicity, force_leg)
    for combo in combos:
        choices = slot_choices(combo)
        if choices is None:
            return None
        n_pl = 1
        for c in choices:
            n_pl *= len(c)
        mult0 = n_combos if (combo is None and split) else 1
        if n_pl > PREPASS_MAX_PLACEMENTS or (combo is None and split):
            # every slot-capable line a kernel line (its largest kernel) on
            # the leg path (the largest of a plain, a dressed and a leg
            # line), in one layout counted n_pl (x the term choices) times
            merged = {}
            for c in choices:
                for pl in c:
                    for ek, att in pl.items():
                        old = merged.get(ek)
                        n_new = (kernel_info(att) + (0, 0))[1]
                        if old is None or n_new > (kernel_info(old)
                                                   + (0, 0))[1]:
                            merged[ek] = att
            plan.append(([merged], n_pl * mult0, True))
        else:
            placements = [{}]
            for c in choices:
                placements = [{**p, **q} for p in placements for q in c]
            plan.append((placements, 1, False))
    pidx = typed_diagram.propagator_indices
    base = []
    for ek in D_.edges():
        hit = pidx.get(ek)
        ri, pi = hit if hit is not None else FI._lookup_prop_indices(
            typed_diagram, ek)
        e = model.entry(pi, ri)
        base.append((ek, pi, ri, e.n if e is not None else 0,
                     bool(has_delta(pi, ri))))
    def coincides(pi, ri, mus):
        # Phase J dresses a line only when no kernel rate equals one of its
        # poles (``_dress_edge_modes``; else it reruns the diagram with that
        # line on the leg path): the poles of a minimal entry realization
        # all carry a residue
        key = ('est_entry_poles', int(pi), int(ri))
        with model._lock:
            lam = model._extra.get(key)
        if lam is None:
            e = model.entry(pi, ri)
            lam = (tuple(complex(x) for x in np.linalg.eigvals(e.a))
                   if e is not None and e.n else ())
            with model._lock:
                model._extra[key] = lam
        tol = FI.DRESS_RATE_RTOL
        return any(abs(la - mu) <= tol * max(abs(la), abs(mu), 1e-300)
                   for la in lam for mu in mus)

    out = []
    for (placements, mult, force_leg) in plan:
        for pl in placements:
            nv = len(verts)
            edges = []
            for (ek, pi, ri, n, dl) in base:
                u, v = index[ek[0]], index[ek[1]]
                att = pl.get(ek)
                info = kernel_info(att) if att is not None else ('none',)
                if info[0] == 'dress' and (force_leg or coincides(pi, ri, info[2])):
                    info = ('leg', info[1])
                if info[0] == 'dress':
                    de = model.dressed_entry(pi, ri, att)
                    nd = de.n if not isinstance(de, str) else n + info[1]
                    # a direct term keeps c x the line's delta part (item 56)
                    edges.append((u, v, nd, dl and info[3]))
                elif info[0] == 'leg':
                    s_ = nv
                    nv += 1
                    edges.append((u, s_, n, dl))
                    edges.append((s_, v, info[1], False))
                else:
                    edges.append((u, v, n, dl))
            out.append((nv, leaf_mask, tuple(edges), mult))
    return out


def _layout_estimate(nv, leaf_mask, edges, item, n_tau, n_perm):
    """``(stationary, transient)`` bytes of one layout over its δ-subsets
    (``diagram_estimate``)."""
    ckey = (nv, leaf_mask, edges, item, n_tau, n_perm)
    hit = _EST_CACHE.get(ckey)
    if hit is not None:
        return hit
    prefetch = PREFETCH_BYTES_PER_POINT * max(1, n_tau) * max(1, n_perm)
    didx = [i for i, (_u, _v, _n, dl) in enumerate(edges) if dl]
    n_int = nv - bin(leaf_mask).count('1')
    prod_n = 1
    for (_u, _v, n, _dl) in edges:
        prod_n *= max(n, 1)
    n_sub = 2 ** len(didx)
    # a stateless delta line (n == 0) is never kept smooth (``_contract``:
    # that subset is never built), so only the delta lines with states
    # double the subsets built (audit MEM-05: 2^k over-estimate since #38)
    n_sub_built = 2 ** sum(1 for i in didx if edges[i][2] > 0)
    cheap_stat = (2 ** n_int) * prod_n
    if (cheap_stat * item <= PREPASS_CHEAP_BYTES
            or len(didx) > PREPASS_MAX_DELTA_EDGES):
        if cheap_stat * item <= PREPASS_CHEAP_BYTES:
            stat_sum, full_sum = cheap_stat, (2 ** nv) * prod_n
        else:
            stat_sum, full_sum = _downset_sums(
                nv, leaf_mask, [(u, v, max(n, 1)) for (u, v, n, _d) in edges])
        D = full_sum
        Dd = min(D, OP_DENSE_MAX)
        out = (n_sub_built * (stat_sum * item + prefetch),
               6 * D * item + 2 * Dd * Dd * item)
    else:
        stat = 0
        Dmax = 0
        for bits in range(n_sub):
            delta = {didx[j] for j in range(len(didx)) if bits >> j & 1}
            red = _contract(nv, leaf_mask, edges, delta)
            if red is None:
                continue                # a stateless line kept smooth
            nv2, lm2, e2 = red
            s_sum, f_sum = _downset_sums(nv2, lm2, e2)
            stat += s_sum * item + prefetch
            Dmax = max(Dmax, f_sum)
        Dd = min(Dmax, OP_DENSE_MAX)
        out = (stat, 6 * Dmax * item + 2 * Dd * Dd * item)
    if len(_EST_CACHE) > 4096:
        _EST_CACHE.clear()
    _EST_CACHE[ckey] = out
    return out


def _contract(nv, leaf_mask, edges, delta):
    """The DAG of one δ-subset: the edges in ``delta`` contracted, cycles
    merged (their vertices share a time), the smooth edges kept:
    ``(nv, leaf_mask, [(u, v, n)])``, or ``None`` when a smooth edge has no
    states (the subset is never built)."""
    parent = list(range(nv))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for i in delta:
        u, v = edges[i][0], edges[i][1]
        parent[find(u)] = find(v)
    smooth = []
    for i, (u, v, n, _dl) in enumerate(edges):
        if i in delta:
            continue
        if n == 0:
            return None
        smooth.append((find(u), find(v), n))
    # merge the strongly connected components (a cycle forces equal times)
    while True:
        roots = sorted({find(x) for x in range(nv)})
        succ = {r: set() for r in roots}
        for (u, v, _n) in smooth:
            if find(u) != find(v):
                succ[find(u)].add(find(v))
        cyc = _find_cycle(roots, succ)
        if not cyc:
            break
        for r in cyc[1:]:
            parent[find(r)] = find(cyc[0])
    roots = sorted({find(x) for x in range(nv)})
    ren = {r: i for i, r in enumerate(roots)}
    lm = 0
    for x in range(nv):
        if leaf_mask >> x & 1:
            lm |= 1 << ren[find(x)]
    e2 = tuple((ren[find(u)], ren[find(v)], n) for (u, v, n) in smooth
               if find(u) != find(v))
    return len(roots), lm, e2


def _find_cycle(nodes, succ):
    """One directed cycle (a list of nodes) of the graph, or ``None``."""
    color = {x: 0 for x in nodes}
    stack_path = []

    def dfs(x):
        color[x] = 1
        stack_path.append(x)
        for y in succ[x]:
            if color[y] == 1:
                return stack_path[stack_path.index(y):]
            if color[y] == 0:
                c = dfs(y)
                if c:
                    return c
        stack_path.pop()
        color[x] = 2
        return None
    for x in nodes:
        if color[x] == 0:
            c = dfs(x)
            if c:
                return c
    return None


def _downset_sums(nv, leaf_mask, edges):
    """``(sum b(I) over the downsets without a leaf, sum b(I) over all
    downsets)`` of the DAG (``edges``: ``(u, v, n_e)``), enumerated up to
    ``PREPASS_MAX_DOWNSETS`` (beyond: ``2^nv x prod n_e``)."""
    preds = [0] * nv
    for (u, v, _n) in edges:
        preds[v] |= 1 << u
    seen = {0}
    frontier = [0]
    while frontier:
        nxt = []
        for I in frontier:
            for v in range(nv):
                b = 1 << v
                if I & b or (preds[v] & ~I):
                    continue
                J = I | b
                if J not in seen:
                    seen.add(J)
                    nxt.append(J)
                    if len(seen) > PREPASS_MAX_DOWNSETS:
                        blk = 1
                        for (_u, _v, n) in edges:
                            blk *= n
                        return (2 ** nv) * blk, (2 ** nv) * blk
        frontier = nxt
    stat = full = 0
    for I in seen:
        sz = 1
        for (u, v, n) in edges:
            if (I >> u) & 1 and not (I >> v) & 1:
                sz *= n
        full += sz
        if not I & leaf_mask:
            stat += sz
    return stat, full


_EST_CACHE = {}


