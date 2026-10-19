"""
engine.integration.time_domain._light
====================================
The Sage-free pieces of Phase J that the state-space route evaluates with:
the runtime counters (``_RUNTIME_COUNTERS`` and the thread / worker overlay,
:class:`counter_delta`), the Θ(0) rule for constant constraint rows
(``_const_row_verdict`` and its helpers, ``_TieContext``) and the Wick
permutation sum of one diagram (:func:`diagram_sum`).

``final_integral`` re-exports every name (one object, one source of truth);
a spawned Phase J worker (``engine._spawn_worker``) imports this module,
``statespace``, numpy and scipy only -- never Sage.  So this module may
import nothing but the standard library.
"""
from collections import namedtuple

from engine.errors import DaedalusError


class PhaseJResourceError(DaedalusError, MemoryError):
    r"""Phase J would hold more modesum-plan memory than the budget
    ``PHASE_J_MEM_BUDGET_GB`` (default: ``default_mem_budget``) allows: raised
    BEFORE the plan that crosses it is allocated (and, from
    ``compute_cumulants``, before Phase J starts when its estimate already
    crosses it), under every analyzer policy.  ``estimate``: the numbers
    (tuples, bytes, budget, the offending diagram / subset).  (Defined here,
    re-exported by ``final_integral``, so a spawned worker raises the same
    class without importing Sage.)"""

    def __init__(self, message, estimate=None):
        super().__init__(message)
        self.estimate = estimate

import collections as _collections_ctr
import threading as _threading_ctr

_CTR_TLS = _threading_ctr.local()
_COUNTER_LOCK = _threading_ctr.Lock()


class _CounterDict(dict):
    """``_RUNTIME_COUNTERS``: a plain dict, except in a thread that opened
    :class:`counter_delta` (a worker of a parallel Phase J batch).  There
    every write goes to that thread's overlay (a read of a key the overlay
    lacks gives 0, or an empty ``Counter`` for a nested dict), so concurrent
    workers never race on a read-modify-write, and the parent merges the
    overlays in a fixed order (:func:`merge_counter_delta`).  Keys ending in
    ``_max`` are maxima, merged with ``max``."""

    __slots__ = ()

    def __getitem__(self, k):
        ov = getattr(_CTR_TLS, 'overlay', None)
        if ov is None:
            return dict.__getitem__(self, k)
        if k in ov:
            return ov[k]
        base = dict.__getitem__(self, k)            # KeyError as without
        if isinstance(base, dict):
            ov[k] = _collections_ctr.Counter()
            return ov[k]
        return 0.0 if isinstance(base, float) else 0

    def __setitem__(self, k, v):
        ov = getattr(_CTR_TLS, 'overlay', None)
        if ov is None:
            dict.__setitem__(self, k, v)
        else:
            ov[k] = v

    def __contains__(self, k):
        ov = getattr(_CTR_TLS, 'overlay', None)
        return dict.__contains__(self, k) or (ov is not None and k in ov)

    def get(self, k, default=None):
        ov = getattr(_CTR_TLS, 'overlay', None)
        if ov is None:
            return dict.get(self, k, default)
        if k in ov or dict.__contains__(self, k):
            return self[k]
        return default

    def setdefault(self, k, default=None):
        ov = getattr(_CTR_TLS, 'overlay', None)
        if ov is None:
            return dict.setdefault(self, k, default)
        if k in ov or dict.__contains__(self, k):
            return self[k]
        ov[k] = default
        return default


class counter_delta:
    """Context manager for a worker of a parallel batch: inside it this
    thread's ``_RUNTIME_COUNTERS`` writes collect in ``self.delta`` (see
    :class:`_CounterDict`); the parent hands ``self.delta`` to
    :func:`merge_counter_delta`.  Not reentrant."""

    def __enter__(self):
        self.delta = {}
        self._prev = getattr(_CTR_TLS, 'overlay', None)
        _CTR_TLS.overlay = self.delta
        return self

    def __exit__(self, *exc):
        _CTR_TLS.overlay = self._prev
        return False


def merge_counter_delta(delta):
    """Add a worker's counter ``delta`` (``counter_delta.delta``, or the
    plain dict a worker process returned) to ``_RUNTIME_COUNTERS``: numbers
    are summed, keys ending in ``_max`` take the maximum, nested dicts are
    merged key by key.  Call in a fixed order for reproducible float sums."""
    c = _RUNTIME_COUNTERS
    with _COUNTER_LOCK:
        for k, v in delta.items():
            if isinstance(v, dict):
                cur = dict.get(c, k)
                if not isinstance(cur, dict):
                    cur = {}
                    dict.__setitem__(c, k, cur)
                for r, n in v.items():
                    cur[r] = cur.get(r, 0) + n
            elif k.endswith('_max'):
                dict.__setitem__(c, k, max(dict.get(c, k, 0), v))
            else:
                dict.__setitem__(c, k, dict.get(c, k, 0) + v)



_RUNTIME_COUNTERS = _CounterDict({
    # m=2 polygon path
    'polygon_attempted': 0,
    'polygon_returned_none': 0,
    # m≥3 poset path
    'poset_attempted': 0,
    'poset_extract_returned_none': 0,
    'poset_consistent_lower_failed': 0,
    'poset_maximality_failed': 0,
    'chain_simplex_fast_returned_none': 0,
    'chain_simplex_polynomial_called': 0,
    'chain_simplex_polynomial_returned_none': 0,
    # memoisation (see the wrappers below)
    'chain_simplex_memo_hits': 0,
    'chain_simplex_memo_misses': 0,
    # M4: the chain memo (``_chain_dd_log``); ``evictions`` counts the
    # table clears at the size cap:
    'chain_uppers_memo_hits': 0,
    'chain_uppers_memo_misses': 0,
    'chain_uppers_memo_evictions': 0,
    'poset_returned_none_total': 0,
    # m=1 interval path
    'interval_attempted': 0,
    'interval_returned_none': 0,
    # scipy.nquad fallback (counted at _integrate_polytope entry for m≥1)
    'scipy_nquad_called_m1': 0,
    'scipy_nquad_called_m2': 0,
    'scipy_nquad_called_mge3': 0,
    # ── M0.1 observational counters (the internal Phase J plan, not
    # distributed).  Purely observational: nothing reads them to
    # decide control flow.
    #
    # Entries into the scipy.nquad fallback: ``_integrate_polytope`` calls
    # with m >= 1, so ``nquad_calls == scipy_nquad_called_m1 + _m2 + _mge3``
    # always.  (An entry may still return before invoking
    # scipy.integrate.nquad, and an invoked quadrature integrates the real
    # and imaginary parts separately: invocations <= 2 * nquad_calls.)
    # The plan's "0 nquad calls on the model zoo" gate (M9) is
    # keyed on this counter.
    'nquad_calls': 0,
    # ``_integrate_polytope`` entries with m = 0: a direct constraint check
    # + evaluation of a δ-collapsed subset, NOT quadrature.
    'polytope_m0_direct': 0,
    # Why an m≥1 subset call reached ``_integrate_polytope`` (counted by the
    # per-diagram and grouped dispatch closures, not by direct callers).
    # Keys are exactly ``_NQUAD_FALLBACK_REASONS``; the fine-grained reason
    # (e.g. 'polygon_gamma_overflow') is passed to ``_SUBSET_HOOK`` instead.
    'nquad_fallback_by_reason': {
        'polygon_guard': 0, 'polygon_other': 0,
        'poset_extract_none': 0, 'poset_no_extension': 0,
        'poset_chain_none': 0, 'other': 0,
        'divergent': 0, 'dd_overflow': 0, 'dd_unsupported': 0,
    },
    # Constraint rows with every |a_int| <= 1e-12 (a smooth edge whose Δt is
    # constant after δ-elimination) processed by ``_polygon_from_2d_
    # constraints`` or ``_extract_causal_poset``.
    'zero_normal_rows_seen': 0,
    # ── M1 Θ(0) counters (M1).  One count per
    # ``_const_row_verdict`` call (i.e. per zero-normal row per
    # evaluation):
    #   theta0_const_empty -- verdict EMPTY (c_eff < 0, or a tie decided
    #                         empty);
    #   theta0_const_drop  -- verdict DROP (c_eff > 0, or a tie decided by
    #                         the leg order to hold);
    #   theta0_tie         -- c_eff == 0.0 exactly, decided EMPTY by
    #                         Θ(0) = 0 itself (no external-time dependence);
    #   theta0_tie_ordered -- c_eff == 0.0 exactly on a difference of two
    #                         legs' times, decided by their raw times / the
    #                         leg order (``_tie_order_sign``; DROP or EMPTY).
    'theta0_const_empty': 0,
    'theta0_const_drop': 0,
    'theta0_tie': 0,
    'theta0_tie_ordered': 0,
    # δ-subset evaluations that returned 0 without integration because a
    # smooth-edge row is constant, τ-independent and EMPTY (a_int ≡ 0,
    # a_ext ≡ 0, c0 <= 0).  The subset is flagged when ``integrate_diagram``
    # builds it.
    'theta0_subsets_pruned': 0,
    # m=2 regions that are empty or degenerate by an exact test, answered 0
    # (``_polygon_region_dd``): two rows with exactly opposite normals whose
    # constants sum to <= 0 (e.g. the 2-cycle rows s1 > s0, s0 > s1), or a
    # region of measure zero.
    'polygon_zero_area': 0,
    # m≥3 posets answered 0 without integration: a constant EMPTY row, or a
    # directed cycle of order rows whose shifts sum to <= 0 (strict
    # inequalities around such a cycle cannot all hold: empty).
    'poset_empty_const': 0,
    'poset_empty_cycle': 0,
    # ── M2a structural-zero counters ──
    # ``_integrate_polytope`` entries (m >= 2, counted in ``nquad_calls`` /
    # ``scipy_nquad_called_*`` as before) answered 0j without quadrature:
    # a directed cycle of unit-coefficient pair rows whose shifts sum to
    # <= 0 (``_region_structurally_empty``).
    'polytope_empty_cycle': 0,
    # δ-subsets never built (per-diagram and grouped builds) because they
    # keep smooth an edge whose smooth part is provably zero (every stored
    # residue of its entry is an exact zero and its ``G_ft`` entry contains
    # no ω; for a pole-free propagator: the ``G_ft`` test alone).  Counted
    # once per subset per build, not per call.
    'forced_delta_pruned': 0,
    # M5 (P6): the per-diagram setup levers (see the section before
    # ``integrate_diagram``).  ``setup_zero_exit``: diagrams whose
    # numerically-zero prefactor returned before any setup (L1).
    'setup_zero_exit': 0,
    # L2: ``integrate_diagram`` calls served from the
    # per-call ``PropagatorTD``; calls handed a ``PropagatorTD`` that no
    # longer matched their propagator data / ``num_params`` (built locally);
    # lazy ``build_G_t_matrix`` builds and per-entry residue-table builds.
    'setup_prop_td_used': 0,
    'setup_prop_td_stale': 0,
    'setup_prop_td_g_t_builds': 0,
    'setup_prop_td_entry_builds': 0,
    # L3: edge ``smooth_factor`` SR objects whose
    # build was deferred / built on demand, and the same for the display-only
    # ``stripped_integrand``.
    'setup_lazy_sr_deferred': 0,
    'setup_lazy_sr_built': 0,
    'setup_lazy_display_deferred': 0,
    'setup_lazy_display_built': 0,
    # L5 (``engine.diagrams.symmetry``): automorphism-order memo
    # hits / misses / table clears, and the ``external_wick_compensation``
    # calls skipped because the diagram has a single Wick mapping.
    'setup_aut_memo_hits': 0,
    'setup_aut_memo_misses': 0,
    'setup_aut_memo_evictions': 0,
    'setup_aut_wick_skipped': 0,
    # M10 (L4a): δ-edge eliminations solved by
    # exact rational linear algebra / handed to ``sage_solve`` because the
    # equation is not an exactly-rational linear form in the eliminated
    # variable; and, under ``VALIDATE_DELTA``, eliminations checked against
    # ``sage_solve`` (substitution and constraint rows exactly equal).
    'setup_delta_solve_fast': 0,
    'setup_delta_solve_fallback': 0,
    'setup_delta_validated': 0,
    # ── M2b hardened quadrature counters ──
    # ``_integrate_polytope`` entries (m >= 1) handed to
    # ``_integrate_polytope_hardened`` (still counted in ``nquad_calls``):
    'nquad_hardened_calls': 0,
    # ... of which answered 0 without quadrature because the closure of
    # their difference rows is infeasible (an exact test; a constant row
    # <= 0 counts here too when the entry did not decide it):
    'nquad_hardened_empty': 0,
    # nested-quadrature intervals found empty (lower >= upper) and answered
    # 0 WITHOUT evaluating the integrand (one count per such interval):
    'nquad_hardened_empty_intervals': 0,
    # entries in which an open direction was truncated at NQUAD_TAIL_K / κ
    # beyond its farthest breakpoint (decay-certified truncation; κ the
    # certified decay rate along it):
    'nquad_hardened_capped': 0,
    # entries in which an open direction had no decay certificate (no mode
    # data, a mode with Re λ >= 0, or, for rows that are not unit
    # difference rows that are edges, a failed linear-programming
    # certificate) and the legacy ±200 cap was used:
    'nquad_hardened_uncertified': 0,
    'nquad_hardened_floor_cut': 0,
    # entries without mode data (epsabs from a pre-sample of the integrand,
    # open sides at the legacy cap, no geometric breakpoints):
    'nquad_hardened_no_modes': 0,
    # entries with a row that is not a difference row (exact level bounds
    # by Fourier-Motzkin elimination, breakpoints at vertex projections):
    'nquad_hardened_general_rows': 0,
    # ... of which the elimination exceeded _NQUAD_FM_MAX_ROWS rows (level
    # bounds: the difference-row closure and the rows filed by innermost
    # variable, a superset of the region's projection):
    'nquad_hardened_fm_capped': 0,
    # levels at which the vertex projections were not enumerated (more
    # than _NQUAD_VERTEX_MAX_COMBOS subsets of rows): kinks incomplete:
    'nquad_hardened_kinks_incomplete': 0,
    # scipy.integrate.quad calls of a region's final pass that still
    # returned a nonzero ``ier`` (roundoff detected, ..., or the limit
    # NQUAD_LIMIT_MAX reached) after the retries, and whose error estimate,
    # times the widths of the enclosing levels' intervals, exceeds
    # NQUAD_EPSREL × the region's scale (|result| in a second pass) and the
    # requested tolerance (or the limit reached); the value is still used
    # and a PhaseJNquadFallbackWarning reports them (once per model, source
    # and loop order):
    'nquad_hardened_quad_flags': 0,
    # quad calls that reached their subinterval limit and were repeated
    # with a larger one (every pass):
    'nquad_hardened_quad_retries': 0,
    # entries integrated a second time with epsabs from the first result
    # (its magnitude times NQUAD_EPSABS_FACTOR), the first pass's epsabs
    # (from the bound / sample of |integrand|) having been larger:
    'nquad_hardened_reruns': 0,
    # entries whose outermost open side was cut farther out (the added strip
    # integrated and added): the bound of the tail dropped by the first cut
    # (K = NQUAD_TAIL_K) was above _NQUAD_TAIL_REL × |result|; or whose
    # inner levels' cuts moved out (``region_tail``; the region integrated
    # again):
    'nquad_hardened_tail_widened': 0,
    # entries whose inner levels were cut but whose cuts could not be
    # checked (a vertex enumeration incomplete, no bound from linear
    # programming, or ``_NQUAD_TAIL_ANCHOR`` off):
    'nquad_hardened_inner_unchecked': 0,
    # entries whose second-pass absolute tolerance was set by the rounding
    # floor (1e-9 × sampled max) rather than by the result,
    # with an implied relative tolerance above 1e-8: reported in the
    # aggregated warning, never silently accepted
    'nquad_hardened_floor_limited': 0,
    # closed-form innermost integrals (``_NquadModes.integrate_innermost``)
    # whose exponentials overflowed and were integrated by quad instead:
    'nquad_hardened_innermost_overflow': 0,
    # entries whose closed-form innermost expansion would exceed
    # _NQUAD_INNERMOST_MAX_TERMS terms (s_0 integrated by quad):
    'nquad_hardened_innermost_capped': 0,
    # largest truncation distance K / κ used (0.0: none):
    'nquad_hardened_cap_span_max': 0.0,
    # ── M3 counters ──
    # m≥3 posets refused by the inheritance rule (P3): a variable without a
    # scalar lower and without a lower-bounded predecessor in the transitive
    # closure of the strict order; the poset path bails
    # ('poset_lower_not_inherited'):
    'poset_lower_not_inherited': 0,
    # ``_integrate_subset_dbm`` calls (m≥3 subsets whose poset path bailed):
    'dbm_attempted': 0,
    # ... answered by the DBM route (a value, counted once):
    'dbm_answered': 0,
    # ... of which the region is empty or of measure zero (a constant EMPTY
    # row, or a closed-DBM cycle of weight <= 0, exact): exactly 0:
    'dbm_empty': 0,
    # ... of which the poset path had bailed on the inheritance rule (P3):
    'dbm_answered_p3': 0,
    # ... of which the first closed form failed the conditioning test and the
    # thin-interval retry passed it (``dbm_integral.integrate_exp_sum``):
    'dbm_answered_thin_retry': 0,
    # calls declined (the subset goes on to ``_integrate_polytope``): a row
    # that is not a difference row (e.g. a ConvVertex 3-term row):
    'dbm_declined_rows': 0,
    # ill-conditioned closed forms re-evaluated at MP_DPS digits
    'dbm_mp_attempted': 0,
    'dbm_mp_answered': 0,
    # conductance kernels folded into lines / exact-zero modes
    'conv_lines_dressed': 0,
    'conv_dress_zero_modes_dropped': 0,
    'conv_kernel_leg_edges': 0,
    # lines dressed with a kernel that has a direct term (c δ + a
    # pole sum), and such kernels refused on a kernel edge
    'conv_lines_dressed_direct': 0,
    'conv_kernel_direct_refused': 0,
    # exact-zero residue modes left out of the modesum plans
    # (modes, and the pole tuples of the full product that took one), the
    # plans built pruned, and pruned plans whose group order could not be
    # replayed from the full product (beyond ``PRUNE_ORDER_MAX_TUPLES``)
    'zero_modes_pruned': 0,
    'zero_mode_tuples_pruned': 0,
    'zero_mode_plans_pruned': 0,
    'zero_mode_order_unreplayed': 0,
    'zero_mode_order_mismatch': 0,
    # F2: pole-tuple groups whose folded coefficient is 0, skipped
    # by the batched poset kernels before any chain is computed
    'dd_batch_zero_groups_skipped': 0,
    # ... a final term beyond the log-overflow limit:
    'dbm_declined_overflow': 0,
    # ... a closed form whose estimated rounding error (1e-15 × the summed
    # magnitudes of its expansion's terms) exceeds max(1e-10 |value|,
    # 1e-14 × the integrand's scale), also after the thin-interval retry:
    # close poles or a thin region (``dbm_integral.STATUS_ILL_CONDITIONED``):
    'dbm_declined_ill_conditioned': 0,
    # ... anything else (rows and modes that do not match, a non-finite
    # shift or exponent, more than dbm_integral.MAX_CASES elimination
    # cases):
    'dbm_declined_other': 0,
    # largest (estimated rounding error) / (accepted error) of an answered
    # DBM value (<= 1 by construction; 0.0: none):
    'dbm_error_ratio_max': 0.0,
    # ── M7 counters (box-free routing; all 0 with the M7 flags off) ──
    # subset evaluations answered on the DD kernels, per path:
    'dd_polygon': 0, 'dd_interval': 0, 'dd_poset': 0, 'dd_grouped_m0': 0,
    # ... of which the region was unbounded and integrated exactly (m=2
    # ideal-vertex fan / m≥3 chain with L = -inf or U = +inf):
    'polygon_exact_unbounded': 0,
    'poset_semiinf_chains': 0,
    # DD-path bails: the integral itself overflows (log|value| > 700, the
    # one-sided final-exponent test); a shape the box-free kernels do not
    # take (an open-top chain with intermediate uppers, a chain state that
    # underflowed); an open direction that does not decay (the divergence
    # policy is M8's):
    'dd_overflow': 0,
    'dd_unsupported': 0,
    'divergent_subsets': 0,
    # M8: pole-tuple groups that fail the convergence certificate but whose
    # merged coefficient is exactly 0 (dropped, not divergent):
    'divergent_cancelled': 0,
    # Error accounting of the box-free subset sums (``_dd_accuracy``): the
    # worst estimated-error / accepted-error ratio, and the subsets above 1.
    'dd_err_ratio_max': 0.0,
    'dd_ill_conditioned': 0,
    # D2 (G6 of ``api.analyze``): the summed absolute rounding-error
    # estimates (``COND_ERR_FACTOR`` x the expansion's magnitude) of the
    # box-free subset sums and of the answered DBM closed forms, for the
    # total-relative accuracy bound at the ``compute_cumulants`` level.
    # Bookkeeping only.
    'dd_err_abs_sum': 0.0,
    'dbm_err_abs_sum': 0.0,
    # DBM calls without a box, with the slow-mode box ±max(200, 40/κ_min)
    # wider than ±200 (the pre-0.2.0 DD route, retired: always
    # 0), and declined as divergent (box-free):
    'dbm_box_free': 0,
    'dbm_slow_mode_box': 0,
    'dbm_declined_divergent': 0,
    # legacy ``_chain_with_intermediate_uppers`` evaluations in which
    # at least one cut case returned None while others did not, and those
    # returning a finite value beyond e^700 (that evaluator was retired in
    # always 0):
    'chain_uppers_legacy_partial': 0,
    'chain_uppers_legacy_huge': 0,
})


# A row has a zero normal when every |a_int_j| <= _ROW_COEF_ATOL (see
# final_integral, THETA_AT_ZERO_CONST_ROW).
_ROW_COEF_ATOL = 1e-12

# Row provenance kinds (see final_integral, ``_row_kind``).
_EDGE_ROW_KINDS = ('edge', 'conv_pseudo')
_BOX_ROW_KINDS = ('noise_box', 'conv_tau')
ROW_KINDS = _EDGE_ROW_KINDS + _BOX_ROW_KINDS


class _TieContext(namedtuple('_TieContext',
                             ('times', 'free_legs', 'origin_leg'))):
    """Which external legs the free external variables of ONE Wick
    permutation evaluation stand for (built by ``contribution()``).

    * ``times``      -- the raw external times, one per external leg, in the
                        caller's leg order (``contribution(*times)``),
                        exactly as passed: NOT converted to float, so a
                        time argument that is never compared (every k=1
                        call, any evaluation without an exact tie) may be
                        of any type the pre-0.2.0 code accepted;
    * ``free_legs``  -- the leg whose time each entry of ``free_ext_vals``
                        carries (same order as ``free_ext_vals``);
    * ``origin_leg`` -- the leg pinned at 0 (every free value is
                        ``times[leg] − times[origin_leg]``), or ``None``.

    ``_tie_order_sign`` uses it to decide a constant row that is exactly 0.0
    by the raw order of the two legs it compares (converting only those two
    times).  Purely descriptive: it never changes a time value.
    """
    __slots__ = ()


def _raw_time_order(t_p, t_q):
    """``t_p − t_q`` reduced to its order: a float whose sign is that of the
    difference (0.0 for equal times), NaN when the times have no order, or
    ``None`` when neither the times nor their difference is a real number.

    Float-convertible times (every API call) are compared as floats, as
    before.  Otherwise (e.g. translation-invariant symbolic times such as
    ``(t, t)``, whose free values are numeric although the times are not)
    the difference is converted instead."""
    try:
        a, b = float(t_p), float(t_q)
    except (TypeError, ValueError):
        try:
            return float(t_p - t_q)
        except (TypeError, ValueError):
            return None
    if a > b:
        return 1.0
    if a < b:
        return -1.0
    if a == b:
        return 0.0
    return float('nan')


def _tie_order_sign(a_ext, c0, tie_ctx):
    r"""Decide ``Θ(Δt)`` for a constant row whose ``c_eff`` is exactly 0.0.

    Returns ``+1`` (the row holds: DROP) or ``-1`` (it fails: EMPTY) when
    ``Δt = c0 + Σ_j a_ext_j t_j`` is the difference ``t_p − t_q`` of two
    external legs' times (``c0 == 0``; mapped to legs through ``tie_ctx``,
    including the origin leg the free values are measured from), and ``0``
    otherwise (no context, or another row shape: Θ(0) = 0 applies).

    The two legs are compared on their RAW times, so a difference that
    rounded to 0.0 only after the translation to the origin leg still gets
    its true sign, the same in every Wick permutation.  Exactly equal raw
    times are ordered by leg index: a leg with a LARGER index counts as
    infinitesimally EARLIER (the k=2 Itô left limit τ = t_1 − t_0 → 0⁻,
    extended to every tie).  Exactly one of ``Θ(t_p − t_q)`` and
    ``Θ(t_q − t_p)`` therefore holds at a tie.  (The leg index is the
    position in the caller's argument list, i.e. in ``external_fields``:
    at coincident times of legs of DIFFERENT fields the value depends on
    the order in which they are listed, exactly as the k=2 τ = 0 grid
    value does -- C_AB(τ → 0⁻) and C_BA(τ → 0⁻) are the two one-sided
    limits of the same physical function.)

    Only the two compared times are converted, and only here
    (``_raw_time_order``); times without a real order give ``0``.
    """
    if tie_ctx is None or float(c0) != 0.0:
        return 0
    free_legs = tie_ctx.free_legs
    if len(a_ext) > len(free_legs):
        return 0
    coef = {}
    total = 0.0
    for j, a in enumerate(a_ext):
        a = float(a)
        if a == 0.0:
            continue
        leg = free_legs[j]
        coef[leg] = coef.get(leg, 0.0) + a
        total += a
    if tie_ctx.origin_leg is not None and total != 0.0:
        leg = tie_ctx.origin_leg
        coef[leg] = coef.get(leg, 0.0) - total
    legs = [(leg, c) for leg, c in coef.items() if c != 0.0]
    if len(legs) != 2:
        return 0
    (leg_a, c_a), (leg_b, c_b) = legs
    if not (c_a == -c_b and abs(c_a) == 1.0):
        return 0
    p, q = (leg_a, leg_b) if c_a > 0 else (leg_b, leg_a)   # Δt = t_p − t_q
    d = _raw_time_order(tie_ctx.times[p], tie_ctx.times[q])
    if d is None or d != d:                 # no real order (NaN, complex)
        return 0
    if d > 0.0:
        return 1
    if d < 0.0:
        return -1
    return 1 if p < q else -1


def _const_row_decision(a_int, a_ext, c0, free_ext_vals, kind='edge',
                        tie_ctx=None):
    r"""The Θ(0) = 0 rule for one row, without side effects:
    ``(verdict, basis)``.

    ``verdict`` is ``None`` for a row with a nonzero normal (a genuine
    half-space), else ``'DROP'`` (the constant condition holds everywhere:
    drop it from the GEOMETRY only, the edge still contributes
    ``exp(λ_e·c_eff)`` to γ) or ``'EMPTY'`` (the region is empty).
    ``basis`` says how: ``'sign'`` (c_eff ≠ 0), ``'theta0'`` (c_eff == 0.0,
    Θ(0) = 0) or ``'tie_order'`` (c_eff == 0.0 on a two-leg time
    difference, ``_tie_order_sign``).  ``c_eff = c0 + Σ_j a_ext_j t_j`` is
    the same arithmetic as every call site.  A constant 'noise_box' /
    'conv_tau' row, or an unknown ``kind``, raises.
    """
    for a in a_int:
        if abs(float(a)) > _ROW_COEF_ATOL:
            return None, None
    if kind not in _EDGE_ROW_KINDS:
        if kind in _BOX_ROW_KINDS:
            raise ValueError(
                f'constant {kind!r} constraint row (a_int={list(a_int)!r}, '
                f'a_ext={list(a_ext)!r}, c0={c0!r}): a kernel/box '
                f'integration variable was eliminated by a δ edge, which no '
                f'model is expected to produce (plan §7.6); refusing to '
                f'guess its Θ(0) convention')
        raise ValueError(f'unknown constraint-row kind {kind!r}; expected '
                         f'one of {ROW_KINDS}')
    c_eff = float(c0) + sum(float(a_ext[j]) * float(free_ext_vals[j])
                            for j in range(len(a_ext)))
    if c_eff > 0.0:
        return 'DROP', 'sign'
    if c_eff != 0.0:                        # < 0 (or NaN): empty
        return 'EMPTY', 'sign'
    s = _tie_order_sign(a_ext, c0, tie_ctx)
    if s > 0:
        return 'DROP', 'tie_order'
    if s < 0:
        return 'EMPTY', 'tie_order'
    return 'EMPTY', 'theta0'                # THETA_AT_ZERO_CONST_ROW = 0


def _const_row_verdict(a_int, a_ext, c0, free_ext_vals, kind='edge',
                       tie_ctx=None):
    r"""Θ(0) = 0 verdict for one constraint row ``a_int·s + c_eff > 0``.

    Returns ``None`` if the row has a nonzero normal (some
    ``|a_int_j| > _ROW_COEF_ATOL``): it is a genuine half-space, not this
    helper's business.  Otherwise the row is the constant condition
    ``c_eff > 0`` with ``c_eff = c0 + Σ_j a_ext_j t_j``, and the verdict is

    * ``'DROP'``  if ``c_eff > 0``: the condition holds everywhere.  Drop it
      from the GEOMETRY only -- the row stays in the per-edge data, so the
      edge still contributes ``exp(λ_e·c_eff)`` to γ;
    * ``'EMPTY'`` if ``c_eff < 0``: the region is empty;
    * a tie, ``c_eff == 0.0`` exactly: ``'EMPTY'`` by Θ(0) = 0
      (``THETA_AT_ZERO_CONST_ROW``, the Itô convention), except a difference
      of two external legs' times, which ``tie_ctx`` orders
      (``_tie_order_sign``: exactly one orientation holds).

    There is no tie tolerance (see the THETA_AT_ZERO_CONST_ROW comment).
    ``kind`` is the row's provenance (see ``ROW_KINDS``): edge kinds get the
    rule above; a constant 'noise_box' / 'conv_tau' row raises.  Counts
    ``theta0_*`` in ``_RUNTIME_COUNTERS`` (``_const_row_decision`` is the
    side-effect-free form).

    """
    verdict, basis = _const_row_decision(a_int, a_ext, c0, free_ext_vals,
                                         kind, tie_ctx)
    if verdict is None:
        return None
    if verdict == 'DROP':
        _RUNTIME_COUNTERS['theta0_const_drop'] += 1
    else:
        _RUNTIME_COUNTERS['theta0_const_empty'] += 1
    if basis == 'theta0':
        _RUNTIME_COUNTERS['theta0_tie'] += 1
    elif basis == 'tie_order':
        _RUNTIME_COUNTERS['theta0_tie_ordered'] += 1
    return verdict


def diagram_sum(perms, k, origin_leaf_idx, free_ext_idx, comp,
                ext_time_values, n_subsets, subset_value):
    """One diagram's value at ``ext_time_values``: the sum over the Wick
    permutations ``perms`` and the subsets ``0 .. n_subsets-1`` of
    ``subset_value(j, free_vals, tie_ctx)``, divided by ``comp`` -- the
    arithmetic of ``final_integral.integrate_diagram``'s ``contribution``
    (which calls this without a subset hook) and of a spawned worker."""
    raw_times = tuple(ext_time_values)
    total = 0.0 + 0.0j
    for perm in perms:
        permuted = [ext_time_values[perm[j]] for j in range(k)]
        if origin_leaf_idx is not None:
            t_origin = permuted[origin_leaf_idx]
            permuted = [pt - t_origin for pt in permuted]
        free_vals = [float(permuted[j]) for j in free_ext_idx]
        tie_ctx = _TieContext(
            raw_times, tuple(perm[j] for j in free_ext_idx),
            None if origin_leaf_idx is None else perm[origin_leaf_idx])
        for j in range(n_subsets):
            total = total + complex(subset_value(j, free_vals, tie_ctx))
    return total / comp
