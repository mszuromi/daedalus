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
from pipeline._mean_field_dae import (
    solve_mean_field_dae_compat,
    linear_stability,
)
from pipeline._diagrams  import enumerate_unique_diagrams
from pipeline.access     import (
    MeanField, Parameters, normalize_external_fields,
)


def _trunc(s, maxlen=200):
    s = str(s)
    return s if len(s) <= maxlen else s[:maxlen - 3] + '...'


def _print_action_sectors(ft):
    """Print MF / Free / Interaction action sectors after expand()."""
    ring_gen_names = [str(g) for g in ft.ring().gens()]

    def _fmt_poly(poly):
        lines = []
        for exp_vec, coeff in poly.dict().items():
            mono = '·'.join(
                f'{ring_gen_names[i]}^{exp_vec[i]}' if exp_vec[i] > 1
                else ring_gen_names[i]
                for i in range(len(exp_vec)) if exp_vec[i] > 0
            )
            mono = mono if mono else '1'
            lines.append(f'        {mono}  *  ({_trunc(coeff, 220)})')
        return lines

    print()
    print('      ── MF action (bigrade ≤1 in each index; vanishes at saddle) ──')
    mf_raw = getattr(ft, '_mf_sector_raw', None) or {}
    any_mf = False
    for key in [(0, 0), (1, 0), (0, 1)]:
        poly = mf_raw.get(key)
        if poly is None or str(poly) == '0':
            continue
        any_mf = True
        print(f'      bigrade {key}:')
        for line in _fmt_poly(poly):
            print(line)
    if not any_mf:
        print('        (empty)')

    print()
    print('      ── Free action (1,1) bilinear sector ──')
    S_free = ft.free_action()
    if str(S_free) == '0':
        print('        (empty)')
    else:
        for line in _fmt_poly(S_free):
            print(line)

    print()
    print('      ── Interaction action (total degree ≥ 2, excluding (1,1)) ──')
    sectors = ft.sectors()
    any_int = False
    for (n_t, n_p), poly in sorted(sectors.items()):
        if (n_t, n_p) == (1, 1) or n_t + n_p < 2:
            continue
        if str(poly) == '0':
            continue
        any_int = True
        print(f'      bigrade ({n_t},{n_p}):')
        for line in _fmt_poly(poly):
            print(line)
    if not any_int:
        print('        (empty)')
    print()


def compute_cumulants(
    model: dict,
    k: int,
    max_ell: int = 0,
    fundamental: dict = None,
    external_fields: list[tuple[str, int]] = None,
    *,
    tau_max: float = 50.0,
    tau_step: float = 0.5,
    spatial_grid=None,
    taylor_order: int = None,
    origin_leaf_idx: int = 0,
    output_npz: str = None,
    output_csv: str = None,
    use_cache: bool = True,
    parallel: bool = True,
    n_workers: int = None,
    use_grouped_phase_j: bool = False,
    fixed_point_index: int = 0,
    mf_dae_n_starts: int = 64,
    mf_dae_seed_box: dict | None = None,
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
        ``None`` (default) auto-picks ``max(k + 2 * max_ell, 2)``,
        which is the smallest order that captures every vertex needed
        for a connected diagram with ``k`` external legs at ``ell``
        loops (max vertex order ≤ ``k + 2·ell``).  The floor of 2 is
        the structural minimum — order 2 still has the (1,1) bilinear
        propagator kernel plus the (0,0)/(1,0)/(0,1) MF saddle sectors,
        which is everything the cache machinery needs even for the
        degenerate ``k=2, max_ell=0`` case (tree-level pair correlator
        = bare propagator, no interaction vertices).  Pass an explicit
        integer to override (e.g. for non-standard diagrammatic
        content or when probing higher-order vertices for cache
        invalidation testing).

        **Previously this floor was 4**, which forced theories at
        ``k=2, max_ell=0`` to pay an order-4 expansion they didn't
        mathematically need — see ``docs/CHANGELOG.md``
        (theory-precompute-cache branch) for the cost it imposed on
        the dendritic-linear theory.
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
                             'combined_prefactor', 'multiplicity', 'ell'}
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
    # bound.  The floor of 2 is structural (anything less wouldn't
    # capture the (1,1) bilinear propagator + (0,0)/(1,0)/(0,1) MF
    # sectors that downstream code unconditionally reads).
    #
    # Historically this floor was 4 — chosen to keep all 1-loop
    # 2-cumulant runs in one cache directory back when the layout
    # was ``saved_theories/<theory>_taylor<N>/``.  The new layout
    # (``saved_theories/<theory>/expand_taylor<N>.sobj``) sibling-
    # files different orders cleanly, so the floor is no longer
    # needed for cache-directory cohesion.  Dropping it from 4 → 2
    # saves ~90 min on heavy Bernoulli theories at ``k=2, max_ell=0``
    # (the only case where the old floor exceeded the math minimum).
    if taylor_order is None:
        taylor_order = max(k + 2 * max_ell, 2)

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

    # ── 1. FieldTheory expansion (cache-aware) ────────────────────
    # Try the on-disk expand cache before paying the full
    # multivariate-taylor cost.  The cache stores ft._by_tp (the
    # bigrade-classified action dict) at each previously-computed
    # taylor_order; we accept any cached order >= the requested one
    # and downgrade-filter to total degree <= target.
    if verbose:
        print(f'[1/7] FieldTheory.expand (taylor_order={taylor_order})...')
    _t_phase = time.perf_counter()
    ft = FieldTheory(model, taylor_order=taylor_order)

    from pipeline import _expand_cache as _ec
    cache_hit = False
    if use_cache:
        cached_order = _ec.find_best_cached_order(model, taylor_order)
        if cached_order is not None:
            _ec.prepare_for_load(ft)
            cache_hit = _ec.load_expand(
                model, ft,
                target_order=taylor_order,
                cached_order=cached_order,
                verbose=verbose,
            )
    if not cache_hit:
        ft.expand()
        if use_cache:
            try:
                _ec.save_expand(model, ft, verbose=verbose)
            except Exception as e:
                if verbose:
                    print(f'      [expand-cache] save failed ({e!r}); '
                          f'continuing without persistence.')

    sanity_ok = ft.sanity_check(verbose=verbose)
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
        _print_action_sectors(ft)
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
    if model.get('equations'):
        # Route to the DAE-based multi-root solver when the theory
        # declares ``.equation(lhs=..., rhs=..., population=...)``.
        # Returns the same legacy-shape keys plus DAE-specific extras
        # (mf_all_roots, mf_index_used, ...) that we surface on the
        # final th dict below.
        mf = solve_mean_field_dae_compat(
            ft, model, fundamental,
            fixed_point_index=fixed_point_index,
            n_starts=mf_dae_n_starts,
            seed_box=mf_dae_seed_box,
            verbose=verbose,
        )
    else:
        mf = solve_mean_field(ft, model, fundamental, verbose=verbose)
    num_params = mf['num_params']
    _phase_time('mean_field', _t_phase)

    # ── 3.5 Spatial short-circuit ─────────────────────────────────
    # A spatial model's propagator carries the inert ``Laplacian``
    # symbol, which the (ω) pole-finder and time-domain Phase J can't
    # consume.  Spatial models route to the full-diagram momentum
    # integrator instead (Symanzik ∫dᵈℓ → causal chambers → q→x FT),
    # which returns the real-space correlator C(x, τ).  See
    # docs/spatial_pipeline.md.  Requires spatial_grid (the x points).
    if spatial_grid is not None and not model.get('spatial'):
        import warnings as _warnings
        _warnings.warn(
            'spatial_grid was provided but the model is not spatial (no field '
            'declares spatial_dim>=1); it is ignored and the temporal C(τ) is '
            'returned.', stacklevel=2)
    if model.get('spatial') and spatial_grid is not None:
        import numpy as _np
        from msrjd.integration.spatial.pipeline_bridge import (
            compute_spatial_correlator_via_pipeline,
        )
        from msrjd.integration.spatial.spatial_correlator import (
            free_two_point,
        )
        if k != 2:
            raise NotImplementedError(
                f'spatial correlators are implemented for k=2 (two-point) '
                f'in v1; got k={k}.')
        if max_ell > 2:
            raise NotImplementedError(
                f'spatial v1 implements tree (max_ell=0), 1-loop (max_ell=1) and '
                f'2-loop (max_ell=2) via the full-diagram integrator; got '
                f'max_ell={max_ell}.  Higher loops work by construction but are '
                f'increasingly expensive (more diagrams, higher-dim integrals).')
        # Initial-condition compatibility (Phase 4): v1 supports the
        # stationary IC, for which two-time correlators are well-posed.
        ic_mode = (model.get('initial') or {}).get('mode', 'stationary')
        if ic_mode != 'stationary':
            raise NotImplementedError(
                f"spatial v1 supports the 'stationary' initial condition "
                f"only; got {ic_mode!r}.")

        tau_grid = _np.arange(-tau_max, tau_max + tau_step * 0.5, tau_step)
        spatial_grid_arr = _np.asarray(spatial_grid, dtype=float)
        if verbose:
            print('[4/7] (spatial) Momentum stays symbolic (Laplacian) — skip the '
                  'ω pole-finder + time-domain Phase J; route to the full-diagram '
                  'integrator.')
            print(f'[spatial] {"loop" if max_ell >= 1 else "tree-level"} '
                  f'C(x, τ): max_ell={max_ell}, {len(tau_grid)} τ × '
                  f'{len(spatial_grid_arr)} x points...')
        # Route through the SHARED pipeline (Stage B/C): the bridge runs the
        # real diagram pipeline at sample momenta to CERTIFY the per-mode
        # (A,B,N) structure, then does the q→x FT.  At max_ell≥1 the FULL-DIAGRAM
        # integrator sums EVERY enumerated diagram up to max_ell loops through one
        # genuine integral (Symanzik ∫dᵈℓ → causal-chamber time integral →
        # ret+adv), weighted by the enumeration M(Γ) — no shortcut, no diagram
        # dropped (docs/spatial_generic_pipeline_plan.md).
        if max_ell >= 1:
            from msrjd.integration.spatial.pipeline_bridge import (
                compute_spatial_correlator_generic,
            )
            C_tau_x, sp_info = compute_spatial_correlator_generic(
                ft, model, prop, num_params, external_fields,
                tau_grid, spatial_grid_arr, verbose=verbose, max_ell=max_ell,
            )
        else:
            C_tau_x, sp_info = compute_spatial_correlator_via_pipeline(
                ft, model, prop, num_params, external_fields,
                tau_grid, spatial_grid_arr, verbose=verbose, stage_headers=True,
            )
        if verbose:
            print(f'[spatial] done — C(x,τ) ready; tree-mode certified='
                  f'{sp_info.get("pipeline_certified")} '
                  f'(max rel {sp_info.get("certify_max_rel"):.1e})')
        # x=0 slice as the conventional C_tau (matches the time-only
        # API's C_tau shape).
        x0_idx = int(_np.argmin(_np.abs(spatial_grid_arr)))
        C_tau = C_tau_x[:, x0_idx].copy()

        sp_d = int(sp_info.get('spatial_dim', 1))

        def total_C(*tau_then_x):
            """C(τ) at x=0 (1 arg) or C(x, τ) (2 args: τ, x)."""
            if len(tau_then_x) == 1:
                tau = float(tau_then_x[0])
                xq = 0.0
            else:
                tau, xq = float(tau_then_x[0]), float(tau_then_x[1])
            if sp_d == 1:
                val = 0j
                for (A, B, N) in sp_info['modes']:
                    val += free_two_point(A, B, N, xq, tau,
                                          bc_mode=sp_info['bc_mode'],
                                          L=sp_info['L'])
                # 1-loop tadpole mass-shift correction δC = Σ·∂C₀/∂A.
                if sp_info.get('Sigma'):
                    A0, B0, N0 = sp_info['modes'][0]
                    A0 = float(_np.real(A0))
                    h = 1e-4 * max(1.0, abs(A0))
                    fp = free_two_point(A0 + h, B0, N0, xq, tau,
                                        bc_mode=sp_info['bc_mode'], L=sp_info['L'])
                    fm = free_two_point(A0 - h, B0, N0, xq, tau,
                                        bc_mode=sp_info['bc_mode'], L=sp_info['L'])
                    val += sp_info['Sigma'] * (fp - fm) / (2.0 * h)
                return val
            # d≥2: radial q→x transform of the momentum-space correlator
            # (+ optional tadpole mass-shift via a finite difference in A).
            from msrjd.integration.spatial.spatial_correlator import (
                radial_inverse_ft,
            )
            qg = _np.linspace(40.0 / 8000.0, 40.0, 2000)
            at = abs(tau)
            Cq = _np.zeros_like(qg)
            for (A, B, N) in sp_info['modes']:
                A = float(_np.real(A)); B = float(_np.real(B)); N = float(_np.real(N))
                m = A + B * qg * qg
                Cq += (N / m) * _np.exp(-m * at)
            if sp_info.get('Sigma'):
                A0, B0, N0 = sp_info['modes'][0]
                A0 = float(_np.real(A0)); B0 = float(_np.real(B0)); N0 = float(_np.real(N0))
                h = 1e-4 * max(1.0, abs(A0))
                mp = (A0 + h) + B0 * qg * qg
                mm = (A0 - h) + B0 * qg * qg
                Cq += sp_info['Sigma'] * (
                    (N0 / mp) * _np.exp(-mp * at) - (N0 / mm) * _np.exp(-mm * at)
                ) / (2.0 * h)
            return complex(radial_inverse_ft(qg, Cq, abs(xq), sp_d))

        # MF values (for the result dict) — reuse the saddle solve.
        mf_values_sp = {}
        for pname, vals in (mf.get('mf_values') or {}).items():
            mf_values_sp[pname] = vals

        return {
            'total_C':        total_C,
            'total_C_by_ell': {0: total_C},
            'C_tau':          C_tau,
            'C_tau_x':        C_tau_x,
            'tau_grid':       tau_grid,
            'spatial_grid':   spatial_grid_arr,
            'spatial_info':   sp_info,
            'mf_values':      mf_values_sp,
            'mf':             MeanField(mf_values_sp,
                                        naming_convention=naming_convention),
            'params':         Parameters(fundamental),
            'num_params':     num_params,
            'propagator':     prop,
            'config': {
                'k':               k,
                'max_ell':         max_ell,
                'fundamental':     fundamental,
                'external_fields': external_fields,
                'tau_max':         tau_max,
                'tau_step':        tau_step,
                'spatial':         True,
                'model_name':      model.get('name', '<unnamed>'),
            },
        }

    # A spatial model reaching this point means spatial_grid was NOT provided
    # (the spatial branch above returns whenever model.spatial AND spatial_grid).
    # The temporal ω-pole-finder / Phase-J path below cannot consume the inert
    # Laplacian symbol carried in G_ft — it would die with a cryptic Sage
    # TypeError deep in compute_poles_and_residues.  Fail clearly instead.
    if model.get('spatial'):
        raise ValueError(
            f"model {model.get('name', '<unnamed>')!r} is spatial (a field "
            f"declares spatial_dim>=1): compute_cumulants requires spatial_grid="
            f"... to route to the spatial integrator.  The temporal pole-finder/"
            f"Phase-J path cannot consume the Laplacian operator.  Pass "
            f"spatial_grid (e.g. the theory's METADATA['spatial_grid'], or a "
            f"numpy array like np.linspace(0.0, 6.0, 25)).")

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

    unique_by_ell, multiplicity_by_ell, all_unique = enumerate_unique_diagrams(
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
        mults = multiplicity_by_ell.get(ell, [1] * len(unique_by_ell[ell]))
        for td, mult in zip(unique_by_ell[ell], mults):
            info = classify_coefficient_factors(
                td, time_dep_params, noise_structure
            )
            # Path A: ``combinatorial_factor`` in symmetry.py now
            # computes M(Γ) = ∏ n_leg! / |Aut_fixed_ext(Γ)| using the
            # full coloured incidence-graph automorphism order.  That
            # already accounts for every Feynman-rule symmetry —
            # same-type vertex swaps, parallel-edge swaps, self-loop
            # leg swaps — so multiplying by ``mult`` (the dedup
            # equivalence-class size) here would double-count and is
            # disabled.
            combined_prefactor = SR(info['scalar_prefactor'])
            kernel_groups.append({
                'diagrams':           [td],
                'combined_prefactor': combined_prefactor,
            })
            diagram_records.append({
                'typed_diagram':      td,
                'classify':           info,
                'combined_prefactor': combined_prefactor,
                'multiplicity':       mult,
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

    # ── DAE-MF extras (when the new equation-based solver ran) ────
    # ``mf_all_roots`` carries every distinct root with a per-root
    # stability annotation (``stable: bool``, ``eigenvalues_finite``).
    # ``fixed_point_index`` indexes over the STABLE subset only —
    # ``mf_stable_roots`` is that subset in the same sort order.
    # Unstable roots are surfaced via ``mf_unstable_roots`` for
    # inspection but are not selectable as expansion points.
    if 'mf_all_roots' in mf:
        result['mf_all_roots']        = mf['mf_all_roots']
        result['mf_stable_roots']     = mf.get('mf_stable_roots', [])
        result['mf_unstable_roots']   = mf.get('mf_unstable_roots', [])
        result['mf_index_used']       = mf['mf_index_used']
        result['mf_state_var_order']  = mf['state_var_order']
        # The selected root is guaranteed stable; its stability dict
        # (with full eigenvalue spectrum + A/B matrices) is the
        # canonical record for downstream consumers.
        try:
            result['mf_stability'] = linear_stability(
                model, fundamental, mf['mf_values'], verbose=False)
        except Exception as e:
            if verbose:
                print(f'  [stability] skipped: {type(e).__name__}: {e}')
            result['mf_stability'] = None

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
