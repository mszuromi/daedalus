"""
pipeline.compute — single-call entry point for the full MSR-JD pipeline.

Wraps the notebook flow (cells 1–28 of hawkes_td_*.ipynb) into a
function that takes a model dict + parameter dict + run config, runs
the entire FT expansion → diagram enumeration → Phase J integration
chain, and returns a result dict (with optional .npz save).

Status (prototype): k=2 / k=3 tree + 1-loop validated end-to-end.
Higher-k slice machinery deferred to future iteration.
"""
from __future__ import annotations

import os
import time
from typing import Any

import numpy as np
from sage.all import SR

# msrjd internals
from msrjd.core.field_theory import FieldTheory
from msrjd.core.vertices import (
    extract_vertex_types, extract_source_types, NoiseSourceType,
)
from msrjd.diagrams.type_assignment import build_field_index_map
from msrjd.diagrams.symmetry import classify_coefficient_factors
from msrjd.integration.time_domain.pipeline import (
    compute_correction_td,
)

# Pipeline-package helpers
from pipeline._propagator import build_propagator, compute_poles_and_residues
from pipeline._mean_field import solve_mean_field
from pipeline._diagrams  import enumerate_unique_diagrams
from pipeline.access     import (
    MeanField, Parameters, normalize_external_fields,
)


def compute_cumulants(
    model: dict,
    k: int,
    max_ell: int = 0,
    fundamental: dict = None,
    external_fields: list[tuple[str, int]] = None,
    *,
    tau_max: float = 50.0,
    tau_step: float = 0.5,
    taylor_order: int = None,
    origin_leaf_idx: int = 0,
    output_npz: str = None,
    output_csv: str = None,
    use_cache: bool = True,
    parallel: bool = True,
    n_workers: int = None,
    use_grouped_phase_j: bool = False,
    verbose: bool = True,
) -> dict[str, Any]:
    """
    Compute the k-point cumulant slice for ``model`` at the given
    parameter point, all the way through Phase J time-domain integration.

    Parameters
    ----------
    model : dict
        The model dict (e.g. ``HAWKES_MODEL`` from
        ``models/hawkes_linear_expg_gtas.py``).  Must declare:
        ``response_fields``, ``physical_fields``, ``parameters``,
        ``kernels``, ``operators``, ``functions``, ``mf_substitutions``,
        ``mf_bg_conditions``, ``specializations``, ``kernel_ft_image``,
        ``phi_concrete``, ``mf_equations``, ``action``.
        Optional: ``correlated_noises``.
    k : int
        Number of external legs (k-point cumulant).
    max_ell : int, default 0
        Maximum loop order (0 = tree only).
    fundamental : dict
        Numerical parameter values keyed by param name.  E.g.::

            {'a': 1.0, 'tau': 10.0, 'tau_g': 2.5,
             'E': [1.1, 1.05], 'w': [[0.35, 0.4], [0.3, 0.5]],
             'w_X': 0.3, 'lambda_X': 2.5, 'p_part': 0.6,
             'mu_shift_diff': 0.0, 'sigma_shift_diff_sq': 1.0}

    external_fields : list of (str, int)
        Length-k list of leaf field tuples, e.g. ``[('dn', 1), ('dn', 2)]``.
    tau_max, tau_step : float
        τ-grid extent and spacing for the Phase J slice evaluation.
    taylor_order : int or None
        Truncation order for FieldTheory's nonlinear-function Taylor.
        ``None`` (default) auto-picks ``max(k + 2 * max_ell, 4)``,
        which is the smallest order that captures every vertex needed
        for a connected diagram with ``k`` external legs at ``ell``
        loops (max vertex order ≤ ``k + 2·ell``), with a 4-floor so
        common 1-loop 2-cumulant calculations share the same
        ``saved_theories/<...>_taylor4`` cache directory.  Pass an
        explicit integer to override (e.g. for non-standard
        diagrammatic content or when probing higher-order vertices
        for cache invalidation testing).
    origin_leaf_idx : int
        Which canonical position to pin to t=0 for the slice (default 0).
    output_npz : str or None
        If given, save the result to this ``.npz`` path via
        ``pipeline.save.save_npz``.  See that module for the schema —
        per-loop-order curves (``C_tree``, ``C_1_loop``, ...,
        ``C_total``), adaptive mean-field arrays, parameter keys.
    output_csv : str or None
        If given, save a wide-format CSV companion to ``output_npz``
        via ``pipeline.save.save_csv`` (parameter / mean-field
        metadata as comment lines, then a τ-grid table).
    use_cache : bool
        Whether to reuse cached symbolic propagator (per ``(model, taylor)``)
        and unique typed diagrams (per ``(model, taylor, k, ell, ext_fields)``).
        Both caches live under ``saved_theories/<model-tag>_taylor<N>/``.
    parallel : bool, default True
        Enable fork-based multiprocessing for the two heavy stages
        that support it: per-prediagram type assignment in step [5]
        and per-τ Phase J evaluation in step [7].  Both stages share
        the same flag.
    n_workers : int or None, default None
        Worker process count when ``parallel=True``.  ``None`` lets
        each stage pick its own default (``min(os.cpu_count(),
        n_tasks)``).
    verbose : bool
        Print progress messages.

    Returns
    -------
    dict with keys:
        'total_C'         : callable(*tau) — full sum across ells
        'total_C_by_ell'  : {ell: callable(*tau)} — per-loop-order
        'C_tau'           : ndarray of total_C on the τ-grid (k=1, 2;
                             None for k≥3 — caller uses total_C / per-
                             ell callables instead)
        'C_tau_by_ell'    : {ell: ndarray or None} — per-loop-order
                             grid eval; ``C_tau == sum(values())``
        'tau_grid'        : ndarray of τ values
        'mf_values'       : raw {internal_name: [v_pop1, ...]}, adaptive
        'mf'              : :class:`pipeline.access.MeanField` accessor —
                             ``mf['v', 1]`` returns ``v*_1``,
                             ``mf['n']`` returns the whole nstar vector
        'params'          : :class:`pipeline.access.Parameters` accessor —
                             ``params['E', 1]``, ``params['w', 1, 2]``,
                             ``params['tau']`` (1-based indexing)
        'num_params'      : {SR symbol: float}
        'propagator'      : the propagator data dict
        'diagrams'        : list of dicts {'typed_diagram', 'classify',
                             'combined_prefactor', 'ell'}
        'kernel_groups'   : list of dicts as fed to compute_correction_td
        'phase_j_by_ell'  : {ell: td_result dict}
        'config'          : input args echoed back; in addition to the
                             obvious keys, ``external_fields_in`` echoes
                             the user-passed form (e.g. ``[('n', 1)]``)
                             while ``external_fields`` stores the
                             internal form (e.g. ``[('dn', 1)]``).
    """
    if fundamental is None:
        fundamental = {}
    if external_fields is None:
        raise ValueError('external_fields is required')
    if len(external_fields) != k:
        raise ValueError(
            f'external_fields has {len(external_fields)} entries but k={k}'
        )

    # Auto-pick a Taylor budget that covers every vertex the
    # prediagram enumerator could ask for at the chosen (k, max_ell).
    # Connected diagrams with k external legs at ell loops have at
    # most ``k + 2·ell``-leg vertices, so that's the tight upper
    # bound — clamp at 4 below so the saved-theories cache directory
    # stays at ``_taylor4`` for the common 1-loop 2-cumulant case.
    if taylor_order is None:
        taylor_order = max(k + 2 * max_ell, 4)

    # Accept user-facing natural names and translate to the internal
    # fluctuation names the action / propagator-typing code expect.
    # The mapping comes from ``model['naming_convention']`` (declared
    # by TheoryBuilder); pipeline falls back to a classic n/v/m map
    # when the model doesn't declare its own.
    naming_convention   = model.get('naming_convention')
    external_fields_user = list(external_fields)
    external_fields      = normalize_external_fields(
        external_fields, naming_convention=naming_convention)

    # ── Per-phase wall-time tracker (verbose only) ─────────────────
    # Recorded into a dict so callers / notebooks can inspect the
    # breakdown via ``th['phase_walls']`` after the run, and so the
    # final `Done.` line can echo a summary.  Stays None when
    # verbose=False so the bookkeeping is opt-in.
    phase_walls: dict[str, float] | None = {} if verbose else None

    def _phase_time(label: str, t0: float) -> None:
        if phase_walls is None:
            return
        dt = time.perf_counter() - t0
        phase_walls[label] = dt
        print(f'      [{label}] done in {dt:.2f}s')

    # ── 1. FieldTheory expansion ──────────────────────────────────
    if verbose:
        print(f'[1/7] FieldTheory.expand (taylor_order={taylor_order})...')
    _t_phase = time.perf_counter()
    ft = FieldTheory(model, taylor_order=taylor_order)
    ft.expand()
    sanity_ok = ft.sanity_check()
    if not sanity_ok:
        raise RuntimeError(
            'FieldTheory.sanity_check() failed — see printout above.'
        )

    vtypes = extract_vertex_types(ft)
    stypes = extract_source_types(ft)
    n_noise = sum(1 for s in stypes if isinstance(s, NoiseSourceType))
    if verbose:
        print(f'      vtypes: {len(vtypes)}, sources: {len(stypes)} '
              f'(NoiseSourceType: {n_noise})')
    _phase_time('expand', _t_phase)

    # ── 2. Propagator (symbolic, cached) ──────────────────────────
    if verbose:
        print('[2/7] Build propagator (K_ker → K_ft → G_ft → D_delta)...')
    _t_phase = time.perf_counter()
    prop = build_propagator(ft, model, use_cache=use_cache, verbose=verbose)
    _phase_time('propagator', _t_phase)

    # ── 3. Mean-field solve + num_params assembly ─────────────────
    if verbose:
        print('[3/7] Solve MF self-consistency...')
    _t_phase = time.perf_counter()
    mf = solve_mean_field(ft, model, fundamental, verbose=verbose)
    num_params = mf['num_params']
    _phase_time('mean_field', _t_phase)

    # ── 4. Numerical poles + residues (fills prop in place) ───────
    if verbose:
        print('[4/7] Compute numerical poles + residue matrices...')
    _t_phase = time.perf_counter()
    compute_poles_and_residues(prop, num_params, verbose=verbose)
    _phase_time('poles', _t_phase)

    # ── 5. Field-index maps + prediagram + typed diagram pipeline ─
    if verbose:
        print(f'[5/7] Enumerate prediagrams (k={k}, max_ell={max_ell})...')
    _t_phase = time.perf_counter()
    ring_var_names = list(ft._ns._ring_var_names)
    n_tilde = ft._n_tilde
    resp_idx, phys_idx = build_field_index_map(ring_var_names, n_tilde)
    for f in external_fields:
        if f not in phys_idx:
            raise ValueError(
                f'external field {f} not in phys_idx '
                f'{sorted(phys_idx.keys())}'
            )

    unique_by_ell, all_unique = enumerate_unique_diagrams(
        ft, model,
        k               = k,
        max_ell         = max_ell,
        external_fields = external_fields,
        G_ft            = prop['G_ft'],
        resp_idx        = resp_idx,
        phys_idx        = phys_idx,
        vtypes          = vtypes,
        stypes          = stypes,
        use_cache       = use_cache,
        parallel        = parallel,
        n_workers       = n_workers,
        verbose         = verbose,
    )
    _phase_time('diagrams', _t_phase)

    # ── 6. Diagram-level prefactor classification + kernel groups ─
    if verbose:
        print('[6/7] Classify coefficient factors per diagram...')
    _t_phase = time.perf_counter()
    time_dep_params = model.get('time_dependent_parameters', []) or []
    noise_structure = model.get(
        'noise_structure', {'temporal_type': 'white', 'amplitude_params': []}
    )

    diagram_records = []
    kernel_groups = []
    # Walk by ell so each record carries the ell tag — needed for the
    # per-loop-order Phase J decomposition in step [7].
    for ell in sorted(unique_by_ell.keys()):
        for td in unique_by_ell[ell]:
            info = classify_coefficient_factors(
                td, time_dep_params, noise_structure
            )
            combined_prefactor = SR(info['scalar_prefactor'])
            kernel_groups.append({
                'diagrams':           [td],
                'combined_prefactor': combined_prefactor,
            })
            diagram_records.append({
                'typed_diagram':      td,
                'classify':           info,
                'combined_prefactor': combined_prefactor,
                'ell':                ell,
            })
    _phase_time('classify', _t_phase)

    # ── 7. Phase J time-domain integration (per ell) ──────────────
    if verbose:
        print(f'[7/7] Phase J: compute_correction_td per ell '
              f'(0..{max_ell})...')
    _t_phase = time.perf_counter()
    tau_grid = np.arange(-tau_max, tau_max + tau_step * 0.5, tau_step)

    propagator_data = {
        'K_ker':   prop['K_ker'],
        'K_ft':    prop['K_ft'],
        'G_ft':    prop['G_ft'],
        'adj_ft':  prop['adj_ft'],
        'D_omega': prop['D_omega'],
        'D_delta': prop['D_delta'],
        't_var':   prop['t_var'],
        'omega':   prop['omega'],
        'nf':      prop['nf'],
        'pole_vals': prop['pole_vals'],
        'C_mats':    prop['C_mats'],
    }

    # Pick the τ-grid evaluation pattern by k (None = no grid eval)
    if k == 1:
        tau_points = [(float(t),) for t in tau_grid]
    elif k == 2:
        # Single-axis slice: vary leaf 1 over tau_grid, leaf 0 pinned.
        tau_points = [(0.0, float(t)) for t in tau_grid]
    else:
        tau_points = None   # k≥3: caller handles grid evaluation

    # Run Phase J once per ell so we get the per-loop-order
    # decomposition the saver / notebook plots need.  Diagrams within
    # an ell are summed in that ell's total_C; the master total_C is
    # the sum across ells.
    phase_j_by_ell  = {}
    total_C_by_ell  = {}
    C_tau_by_ell    = {}
    for ell in sorted(unique_by_ell.keys()):
        records_ell = [r for r in diagram_records if r['ell'] == ell]
        if not records_ell:
            # No diagrams at this ell (e.g. tree-level for some
            # subset configs).  Contribute zero.
            total_C_by_ell[ell] = (lambda *t: 0.0 + 0.0j)
            phase_j_by_ell[ell] = None
            if tau_points is not None:
                C_tau_by_ell[ell] = np.zeros(len(tau_grid), dtype=complex)
            else:
                C_tau_by_ell[ell] = None
            continue

        if use_grouped_phase_j:
            # Prototype: group typed diagrams by parent prediagram and
            # sum integrands before fast_callable + quadrature.  See
            # ``pipeline/_grouped_phase_j.py`` for the math + caveats.
            from pipeline._grouped_phase_j import (
                compute_correction_td_grouped,
            )
            td_result_ell = compute_correction_td_grouped(
                typed_diagrams   = [r['typed_diagram']
                                    for r in records_ell],
                prefactors       = [r['combined_prefactor']
                                    for r in records_ell],
                k                = k,
                propagator_data  = propagator_data,
                external_fields  = external_fields,
                num_params       = num_params,
                origin_leaf_idx  = origin_leaf_idx,
            )
        else:
            td_result_ell = compute_correction_td(
                typed_diagrams   = [r['typed_diagram']
                                    for r in records_ell],
                prefactors       = [r['combined_prefactor']
                                    for r in records_ell],
                k                = k,
                propagator_data  = propagator_data,
                external_fields  = external_fields,
                num_params       = num_params,
                origin_leaf_idx  = origin_leaf_idx,
            )
        total_C_by_ell[ell] = td_result_ell['total_C']
        phase_j_by_ell[ell] = td_result_ell

        if tau_points is not None:
            C_tau_by_ell[ell] = np.array(
                td_result_ell['total_C_batch'](
                    tau_points, parallel=parallel, n_workers=n_workers),
                dtype=complex,
            )
        else:
            C_tau_by_ell[ell] = None

    _phase_time('phase_j', _t_phase)

    # Master total_C: sum across ell (for caller convenience)
    def total_C(*ext_time_values):
        return sum(complex(fn(*ext_time_values))
                   for fn in total_C_by_ell.values())

    # Aggregate τ-grid C(τ) = Σ_ell C_ell(τ)
    if tau_points is not None:
        C_tau = sum(C_tau_by_ell[ell] for ell in C_tau_by_ell)
    else:
        C_tau = None

    # ── Adaptive mean-field dict ──────────────────────────────────
    # Discover MF saddle quantities from the model itself.  Two paths:
    #   1. ``model['naming_convention']['mf_parameters']`` (preferred,
    #      written by TheoryBuilder when ``mean_field=True`` is set)
    #   2. ``parameters[i]['mean_field']`` flag (the same data)
    #   3. Classic n/v/m fallback for hand-written model files that
    #      pre-date the declarative API.
    npop = len(ft._ns.pop)

    mf_param_names: list[str] = []
    if naming_convention and naming_convention.get('mf_parameters'):
        mf_param_names = list(naming_convention['mf_parameters'])
    else:
        # Walk model['parameters'] looking for the flag
        for pspec in model.get('parameters', []) or []:
            if pspec.get('mean_field'):
                mf_param_names.append(pspec['name'])
        # Last-resort fallback for legacy models
        if not mf_param_names:
            for legacy in ('nstar', 'vstar', 'mstar'):
                if hasattr(ft._ns, legacy):
                    mf_param_names.append(legacy)

    # Map each MF param name to the range of local indices it owns.
    # Heterogeneous theories declare ``indexed_by=['<pop>']`` on each
    # saddle; the param's SR array is sized to ``len(pop_<pop>)``.
    # Legacy single-pop theories use the flat ``ns.pop`` length.
    param_specs_by_name = {
        pspec['name']: pspec
        for pspec in (model.get('parameters', []) or [])
    }
    pop_size_map = getattr(ft._ns, '_pop_size', {}) or {}

    def _saddle_indices(pname):
        pspec = param_specs_by_name.get(pname, {})
        ib = pspec.get('indexed_by') or []
        if ib:
            # Heterogeneous: iterate over the saddle's own population
            # (single-population saddles only — physical_fields are
            # currently restricted to one population per field).
            n = pop_size_map.get(ib[0], 0)
            return list(range(n))
        # Legacy: flat pop index
        return list(ft._ns.pop)

    mf_values: dict[str, list] = {}
    for pname in mf_param_names:
        if not hasattr(ft._ns, pname):
            continue
        ns_syms = getattr(ft._ns, pname)
        # ns_syms may be a single SR var (un-indexed param) or a list
        if not isinstance(ns_syms, (list, tuple)):
            ns_syms = [ns_syms]
        vals: list[float] = []
        for i in _saddle_indices(pname):
            if i >= len(ns_syms):
                vals.append(float('nan'))
                continue
            sym = ns_syms[i]
            if sym in num_params:
                vals.append(float(num_params[sym]))
            else:
                vals.append(float('nan'))
        # Drop entries that are entirely NaN (model declared the
        # symbol but the MF solver never substituted — e.g. mstar
        # in a non-GTaS model that still ships an mstar declaration).
        if not all(v != v for v in vals):
            mf_values[pname] = vals

    result = {
        'total_C':         total_C,
        'total_C_by_ell':  total_C_by_ell,
        'C_tau':           C_tau,
        'C_tau_by_ell':    C_tau_by_ell,
        'tau_grid':        tau_grid,
        'mf_values':       mf_values,
        'mf':              MeanField(mf_values,
                                     naming_convention=naming_convention),
        'params':          Parameters(fundamental),
        'num_params':      num_params,
        'propagator':      prop,
        'diagrams':        diagram_records,
        'kernel_groups':   kernel_groups,
        'phase_j_by_ell':  phase_j_by_ell,
        'phase_walls':     phase_walls,  # dict[label → seconds], None if not verbose
        'config': {
            'k':                  k,
            'max_ell':            max_ell,
            'fundamental':        fundamental,
            'external_fields':    external_fields,        # internal names
            'external_fields_in': external_fields_user,   # as user passed
            'tau_max':            tau_max,
            'tau_step':           tau_step,
            'taylor_order':       taylor_order,
            'model_name':         model.get('name', '<unnamed>'),
        },
    }

    # ── Optional NPZ / CSV save (pipeline.save handles the schema) ─
    if output_npz or output_csv:
        from pipeline.save import save_npz, save_csv
        if output_npz:
            save_npz(result, output_npz)
            if verbose:
                print(f'\nSaved NPZ: {output_npz}')
        if output_csv:
            save_csv(result, output_csv)
            if verbose:
                print(f'\nSaved CSV: {output_csv}')

    if verbose:
        print(f'\nDone.  k={k}, max_ell={max_ell}, '
              f'{len(all_unique)} unique diagrams, '
              f'{len(tau_grid)} τ points.')
        if phase_walls:
            total = sum(phase_walls.values())
            print(f'\n  Phase wall summary (Σ = {total:.1f}s):')
            for label, dt in phase_walls.items():
                pct = 100.0 * dt / total if total > 0 else 0.0
                print(f'    {label:12s}  {dt:6.2f}s  ({pct:4.1f}%)')

    return result
