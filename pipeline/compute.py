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
from pipeline._diagrams import enumerate_unique_diagrams


def compute_cumulants(
    model: dict,
    k: int,
    max_ell: int = 0,
    fundamental: dict = None,
    external_fields: list[tuple[str, int]] = None,
    *,
    tau_max: float = 50.0,
    tau_step: float = 0.5,
    taylor_order: int = 4,
    origin_leaf_idx: int = 0,
    output_npz: str = None,
    use_cache: bool = True,
    parallel: bool = True,
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
    taylor_order : int
        Truncation order for FieldTheory's nonlinear-function Taylor.
    origin_leaf_idx : int
        Which canonical position to pin to t=0 for the slice (default 0).
    output_npz : str or None
        If given, save the result dict to this .npz path.
    use_cache : bool
        Whether to reuse cached symbolic propagator (per ``(model, taylor)``)
        and unique typed diagrams (per ``(model, taylor, k, ell, ext_fields)``).
        Both caches live under ``saved_theories/<model-tag>_taylor<N>/``.
    parallel : bool
        Pass-through to compute_correction_td for multi-process eval.
    verbose : bool
        Print progress messages.

    Returns
    -------
    dict with keys:
        'total_C'        : callable(*tau)
        'C_tau'          : ndarray of total_C on the τ-grid (k=2 only;
                            for k≥3 a dict of slices)
        'tau_grid'       : ndarray of τ values
        'mf_values'      : {'nstar': [..], 'vstar': [..], 'mstar': [..]}
        'num_params'     : {SR symbol: float}
        'propagator'     : the propagator data dict
        'diagrams'       : list of TypedDiagram with classify info
        'kernel_groups'  : list of dicts as fed to compute_correction_td
        'config'         : input args echoed back
    """
    if fundamental is None:
        fundamental = {}
    if external_fields is None:
        raise ValueError('external_fields is required')
    if len(external_fields) != k:
        raise ValueError(
            f'external_fields has {len(external_fields)} entries but k={k}'
        )

    # ── 1. FieldTheory expansion ──────────────────────────────────
    if verbose:
        print(f'[1/7] FieldTheory.expand (taylor_order={taylor_order})...')
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

    # ── 2. Propagator (symbolic, cached) ──────────────────────────
    if verbose:
        print('[2/7] Build propagator (K_ker → K_ft → G_ft → D_delta)...')
    prop = build_propagator(ft, model, use_cache=use_cache, verbose=verbose)

    # ── 3. Mean-field solve + num_params assembly ─────────────────
    if verbose:
        print('[3/7] Solve MF self-consistency...')
    mf = solve_mean_field(ft, model, fundamental, verbose=verbose)
    num_params = mf['num_params']

    # ── 4. Numerical poles + residues (fills prop in place) ───────
    if verbose:
        print('[4/7] Compute numerical poles + residue matrices...')
    compute_poles_and_residues(prop, num_params, verbose=verbose)

    # ── 5. Field-index maps + prediagram + typed diagram pipeline ─
    if verbose:
        print(f'[5/7] Enumerate prediagrams (k={k}, max_ell={max_ell})...')
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
        verbose         = verbose,
    )

    # ── 6. Diagram-level prefactor classification + kernel groups ─
    if verbose:
        print('[6/7] Classify coefficient factors per diagram...')
    time_dep_params = model.get('time_dependent_parameters', []) or []
    noise_structure = model.get(
        'noise_structure', {'temporal_type': 'white', 'amplitude_params': []}
    )

    diagram_records = []
    kernel_groups = []
    for td in all_unique:
        info = classify_coefficient_factors(
            td, time_dep_params, noise_structure
        )
        # combined_prefactor for compute_correction_td is the scalar
        # prefactor returned by classify_coefficient_factors
        combined_prefactor = SR(info['scalar_prefactor'])
        kernel_groups.append({
            'diagrams':           [td],
            'combined_prefactor': combined_prefactor,
        })
        diagram_records.append({
            'typed_diagram':      td,
            'classify':           info,
            'combined_prefactor': combined_prefactor,
        })

    # ── 7. Phase J time-domain integration ────────────────────────
    if verbose:
        print('[7/7] Phase J: compute_correction_td on τ-grid...')
    tau_grid = np.arange(-tau_max, tau_max + tau_step * 0.5, tau_step)

    # Build typed_diagrams + prefactors lists (legacy unpack interface)
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

    # Build typed_diagrams + prefactors lists (the preferred direct
    # interface — kernel_groups is the legacy unpack format).
    typed_diagrams_list = [td_record['typed_diagram']
                           for td_record in diagram_records]
    prefactors_list     = [td_record['combined_prefactor']
                           for td_record in diagram_records]

    td_result = compute_correction_td(
        typed_diagrams   = typed_diagrams_list,
        prefactors       = prefactors_list,
        k                = k,
        propagator_data  = propagator_data,
        external_fields  = external_fields,
        num_params       = num_params,
        origin_leaf_idx  = origin_leaf_idx,
    )

    # ── Build a τ-grid evaluation of total_C ──────────────────────
    total_C = td_result['total_C']
    if k == 2:
        # Single-axis slice: vary leaf 1 over tau_grid, leaf 0 pinned
        C_tau = np.array([
            complex(total_C(0.0, float(t))) for t in tau_grid
        ], dtype=complex)
    else:
        # k>=3: leave evaluation up to caller — total_C is a callable
        C_tau = None

    result = {
        'total_C':        total_C,
        'C_tau':          C_tau,
        'tau_grid':       tau_grid,
        'mf_values': {
            'nstar': mf['nstar_vals'],
            'vstar': mf['vstar_vals'],
            'mstar': [
                fundamental['lambda_X'] * fundamental['p_part']
                for _ in ft._ns.pop
            ] if 'lambda_X' in fundamental else None,
        },
        'num_params':     num_params,
        'propagator':     prop,
        'diagrams':       diagram_records,
        'kernel_groups':  kernel_groups,
        'phase_j_result': td_result,
        'config': {
            'k':               k,
            'max_ell':         max_ell,
            'fundamental':     fundamental,
            'external_fields': external_fields,
            'tau_max':         tau_max,
            'tau_step':        tau_step,
            'taylor_order':    taylor_order,
            'model_name':      model.get('name', '<unnamed>'),
        },
    }

    # ── Optional .npz save ────────────────────────────────────────
    if output_npz:
        save_payload = {
            'tau_grid':       tau_grid,
            'C_tau_real':     (C_tau.real if C_tau is not None
                               else np.array([])),
            'C_tau_imag':     (C_tau.imag if C_tau is not None
                               else np.array([])),
            'nstar':          np.array(mf['nstar_vals'], dtype=float),
            'vstar':          np.array(mf['vstar_vals'], dtype=float),
            'k':              np.array([k], dtype=int),
            'max_ell':        np.array([max_ell], dtype=int),
            'model_name':     np.array([model.get('name', '')]),
        }
        if 'lambda_X' in fundamental and 'p_part' in fundamental:
            save_payload['mstar'] = np.array(
                [fundamental['lambda_X'] * fundamental['p_part']
                 for _ in ft._ns.pop], dtype=float)
        os.makedirs(os.path.dirname(os.path.abspath(output_npz)) or '.',
                    exist_ok=True)
        np.savez(output_npz, **save_payload)
        if verbose:
            print(f'\nSaved: {output_npz}')

    if verbose:
        print(f'\nDone.  k={k}, max_ell={max_ell}, '
              f'{len(all_unique)} unique diagrams, '
              f'{len(tau_grid)} τ points.')

    return result
