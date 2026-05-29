"""
msrjd.integration.spatial.pipeline_bridge
==========================================
The "symbolic-in-q bridge" (spatial Phase 5, Stage A production) — route a
Tier-1 spatial theory through the SHARED diagram pipeline in the mixed
``(t, k)`` representation, then do the external ``q → x`` Fourier transform
ANALYTICALLY (heat-kernel / erf closed form: exact at ``τ = 0``, no ringing).

Why this exists
---------------
The bespoke :func:`compute_spatial_correlator_tree` builds ``C(x, τ)``
directly from the propagator's heat-kernel block.  It is correct but
bypasses the shared diagram machinery, so it does not generalize to loops.
This bridge instead reproduces the same answer THROUGH the shared pipeline:

  1. run the SAME pipeline a time-only theory uses
     (``compute_poles_and_residues`` → ``enumerate_unique_diagrams`` →
     ``classify_coefficient_factors`` → ``compute_correction_td``) with
     ``Laplacian → -q²`` substituted into ``num_params``, so the pipeline
     sees a time-only rational propagator at effective mass ``m(q)=A+Bq²``
     and returns the mixed correlator ``C(q, τ)``;

  2. CERTIFY that the pipeline's ``C(q, τ)`` equals the per-mode heat-kernel
     structure ``Σ_α N_α/(A_α+B_α q²)·e^{-(A_α+B_α q²)|τ|}`` read from the
     propagator (``ac_mass``, ``ac_diffusion``) and the noise sector — the
     bridge between "the diagrams are right" and "the modes are right";

  3. do the external ``q → x`` FT analytically: ``C(x, τ) = Σ_α
     free_two_point(A_α, B_α, N_α; x, τ)`` — each mode's q-FT IS the
     validated :func:`heat_kernel`-family closed form
     (:func:`spatial_correlator.free_two_point`).

For v1's scope (tree level + the constant-mass-shift Allen-Cahn tadpole)
the dressed propagator stays SINGLE-mode, so the certification is exact to
machine precision and ``C(x, τ)`` matches the bespoke oracle.  Multi-mode
correlators (momentum-dependent self-energy, 2-loop+) are future work; the
mode list is the natural place that generalization plugs in.

This module is ADDITIVE — it does NOT modify ``compute.py`` /
``integrate_diagram``.  It is validated against the bespoke oracle and the
spatial test suite (see ``tests/test_spatial_pipeline_bridge.py``) ahead of
retiring the bespoke short-circuit (Stage B).
"""
from __future__ import annotations

import math

import numpy as np
from sage.all import SR

from msrjd.integration.spatial.heat_kernel import SpatialPropagatorError
from msrjd.integration.spatial.spatial_correlator import (
    extract_noise_coefficients, free_two_point,
)


# ── helpers ───────────────────────────────────────────────────────
def _norm_sr(num_params):
    """Normalize ``num_params`` (str- or SR-keyed) to an SR-keyed dict."""
    out = {}
    for kk, vv in num_params.items():
        out[SR.var(str(kk))] = vv
    return out


def _field_index(prop, leg_name):
    """Map an external-leg field name to the diagonal field index in the
    propagator's physical columns (mirrors ``spatial_correlator``)."""
    ring_names = prop['ring_gen_names']
    n_tilde = prop['nf']
    phys_names = ring_names[n_tilde:]
    base = str(leg_name)
    for idx, pn in enumerate(phys_names):
        if base == pn or base.rstrip('0123456789') == pn.rstrip('0123456789'):
            return idx
    return 0


def _legs_to_phys_idx(external_fields, phys_idx):
    """Map external-leg specs to VALID phys_idx keys for diagram enumeration.

    ``compute_cumulants`` normalizes external fields to internal fluctuation
    names but KEEPS the per-leg label, e.g. a 2-point auto-correlation arrives
    as ``[('dphi',1),('dphi',2)]``.  A single-population field has only the
    phys_idx key ``('dphi',1)``, so the ``('dphi',2)`` leg is invalid and
    ``enumerate_unique_diagrams`` would type 0 diagrams (the Stage A unblock:
    both auto-correlation legs must sit on the same valid key).  Map each leg's
    field NAME to its actual phys_idx ``(name, population)`` key.
    """
    keys = list(phys_idx.keys())
    out = []
    for leg in external_fields:
        if isinstance(leg, (tuple, list)):
            if tuple(leg) in phys_idx:
                out.append(tuple(leg))
                continue
            name = str(leg[0])
        else:
            name = str(leg)
        base = name.rstrip('0123456789')
        match = None
        for key in keys:
            kn = str(key[0])
            if kn == name or kn.rstrip('0123456789') == base:
                match = key
                break
        out.append(match if match is not None else (keys[0] if keys else None))
    return out


def _bc_from_prop(prop, num_params_sr):
    """Resolve (bc_mode, L) the way the bespoke correlator does."""
    bc_mode = prop.get('bc_mode', 'infinite')
    bc_params = prop.get('bc_params', {}) or {}
    L = None
    if bc_mode == 'periodic':
        lname = bc_params.get('length')
        if isinstance(lname, str):
            key = SR.var(lname)
            if key not in num_params_sr:
                raise SpatialPropagatorError(
                    f'periodic length parameter {lname!r} not in num_params.')
            L = float(num_params_sr[key])
        else:
            L = float(lname)
    return bc_mode, L


# ── 1. per-mode (A, B, N) structure ───────────────────────────────
def diagonal_modes_from_propagator(prop, ft, num_params, field_index):
    """Return ``[(A, B, N)]`` — the per-mode heat-kernel structure for the
    diagonal field ``field_index``.

    v1 emits a SINGLE mode: ``A = ac_mass`` (relaxation), ``B = ac_diffusion``
    (Laplacian coefficient), ``N`` = white-noise spectral weight from the
    ``(2, 0)`` action sector.  The list shape is what generalizes to a
    multi-mode dressed propagator.  ``num_params`` may be str- or SR-keyed.
    """
    nps_sr = _norm_sr(num_params)
    nps_str = {str(kk): vv for kk, vv in num_params.items()}

    A_expr = prop['ac_mass'][field_index]
    B_expr = prop['ac_diffusion'][field_index]
    A = complex(SR(A_expr).subs(nps_sr))
    Bsub = SR(B_expr).subs(nps_sr)
    B = float(Bsub.real() if hasattr(Bsub, 'real') else Bsub)
    if B <= 0.0:
        if B < 0.0:
            raise SpatialPropagatorError(
                f'field index {field_index} has NEGATIVE diffusion B={B}: the '
                f'spatial operator is anti-diffusive / ill-posed.  Check the '
                f'sign of the Laplacian term (should be "- D*Laplacian", D>0).')
        raise SpatialPropagatorError(
            f'field index {field_index} has zero diffusion (B=0): it is a '
            f'time-only (spatial_dim=0) field with no heat-kernel mode.  A '
            f'spatial correlator is only defined for dim >= 1.')
    noise = extract_noise_coefficients(ft, nps_str)
    N = noise.get(field_index)
    if N is None:
        raise SpatialPropagatorError(
            f'no white-noise coefficient for field index {field_index} '
            f'(noise sector empty?). Available: {noise}')
    return [(A, B, float(N))]


def _modes_C_q_tau(modes, qval, taus):
    """Reference ``C(q, τ) = Σ_α N_α/(A_α+B_α q²)·e^{-(A_α+B_α q²)|τ|}``."""
    taus = np.asarray(taus, dtype=float)
    out = np.zeros(taus.shape, dtype=np.complex128)
    for (A, B, N) in modes:
        m = A + B * qval * qval
        out += (N / m) * np.exp(-m * np.abs(taus))
    return out


# ── 2. run the SHARED pipeline at Laplacian = -q² ─────────────────
def build_pipeline_records(ft, model, prop, external_fields, max_ell=0, k=2):
    """Enumerate + classify the (q-independent) diagram topology ONCE.

    Returns ``{ell: [(typed_diagram, scalar_prefactor), ...]}`` for
    ``compute_correction_td``.  Uses the exact entry points
    ``pipeline/compute.py`` uses, so this is the real shared path
    (lazy-imported to avoid any import cycle).
    """
    from msrjd.core.vertices import extract_vertex_types, extract_source_types
    from msrjd.diagrams.type_assignment import build_field_index_map
    from msrjd.diagrams.symmetry import classify_coefficient_factors
    from pipeline._diagrams import enumerate_unique_diagrams

    vtypes = extract_vertex_types(ft)
    stypes = extract_source_types(ft)
    ring_var_names = list(ft._ns._ring_var_names)
    n_tilde = ft._n_tilde
    resp_idx, phys_idx = build_field_index_map(ring_var_names, n_tilde)

    unique_by_ell, _, _ = enumerate_unique_diagrams(
        ft, model, k=k, max_ell=max_ell, external_fields=external_fields,
        G_ft=prop['G_ft'], resp_idx=resp_idx, phys_idx=phys_idx,
        vtypes=vtypes, stypes=stypes, use_cache=False, verbose=False)
    by_ell = {}
    for ell in unique_by_ell:
        recs = []
        for td in unique_by_ell[ell]:
            info = classify_coefficient_factors(
                td, [], {'temporal_type': 'white', 'amplitude_params': []})
            recs.append((td, SR(info['scalar_prefactor'])))
        by_ell[ell] = recs
    return by_ell


def pipeline_C_q_tau(prop, records, external_fields, base_np_sr, qval, taus,
                     k=2):
    """Run the SHARED pipeline at ``Laplacian = -q²`` → ``C(q, τ)``.

    ``base_np_sr`` is the SR-keyed parameter map WITHOUT ``Laplacian``.
    Mutates ``prop``'s pole/residue cache (re-solved per q), exactly as the
    spike and ``compute.py`` do.
    """
    from pipeline._propagator import compute_poles_and_residues
    from msrjd.integration.time_domain.pipeline import compute_correction_td

    Lap = SR.var('Laplacian')
    nps = dict(base_np_sr)
    nps[Lap] = -(qval ** 2)
    compute_poles_and_residues(prop, nps, verbose=False)
    pdata = {key: prop[key] for key in (
        'K_ker', 'K_ft', 'G_ft', 'adj_ft', 'D_omega', 'D_delta',
        't_var', 'omega', 'nf', 'pole_vals', 'C_mats')}
    res = compute_correction_td(
        typed_diagrams=[r[0] for r in records],
        prefactors=[r[1] for r in records],
        k=k, propagator_data=pdata, external_fields=external_fields,
        num_params=nps, origin_leaf_idx=0)
    tC = res['total_C']
    return np.array([complex(tC(0.0, float(t))) for t in taus])


def certify_modes(modes, prop, records, external_fields, base_np_sr,
                  q_samples, tau_samples, k=2):
    """Max relative error between the pipeline's ``C(q, τ)`` and the per-mode
    reference, over the ``(q, τ)`` sample grid.  Small ⇒ the diagrams the
    pipeline produced ARE the heat-kernel modes the q-FT will transform.
    """
    taus = np.asarray(tau_samples, dtype=float)
    worst = 0.0
    for qv in q_samples:
        pipe = pipeline_C_q_tau(prop, records, external_fields, base_np_sr,
                                qv, taus, k=k).real
        ref = _modes_C_q_tau(modes, qv, taus).real
        denom = np.maximum(np.abs(ref), 1e-30)
        worst = max(worst, float(np.max(np.abs(pipe - ref) / denom)))
    return worst


# ── 3. top-level: pipeline-certified, analytic q-FT correlator ────
def compute_spatial_correlator_via_pipeline(
        ft, model, prop, num_params, external_fields, tau_grid, spatial_grid,
        verbose=False, certify=True, q_samples=(0.0, 0.7, 1.5),
        tau_samples=(0.5, 1.0), certify_tol=1e-8):
    """Drop-in alternative to :func:`compute_spatial_correlator_tree` that
    ROUTES THROUGH THE SHARED PIPELINE.

    Steps: read the per-mode ``(A, B, N)`` from the propagator; (optionally)
    CERTIFY them against the pipeline's diagram-based ``C(q, τ)`` at a few
    sample momenta; then build ``C(x, τ)`` by the analytic ``q → x`` FT
    ``Σ_α free_two_point(A_α, B_α, N_α; x, τ)`` (exact at ``τ = 0``).

    Returns ``(C_tau_x, info)`` mirroring the bespoke API; ``info`` adds
    ``pipeline_certified`` and ``certify_max_rel``.
    """
    if not prop.get('spatial_dim'):
        raise SpatialPropagatorError('propagator has no spatial block.')
    if prop.get('G_tx_sym') is None:
        why = prop.get('spatial_tier1_error', 'no closed-form spatial block')
        raise NotImplementedError(
            'pipeline bridge supports only the Tier-1 diagonal heat-kernel '
            f'propagator (reason: {why}).')

    nps_sr = _norm_sr(num_params)
    base_np_sr = {kk: vv for kk, vv in nps_sr.items()
                  if str(kk) != 'Laplacian'}

    # Translate external legs to VALID phys_idx keys for the pipeline
    # enumeration (both auto-correlation legs land on the single field key).
    from msrjd.diagrams.type_assignment import build_field_index_map
    ring_var_names = list(ft._ns._ring_var_names)
    _, phys_idx = build_field_index_map(ring_var_names, ft._n_tilde)
    ext_int = _legs_to_phys_idx(external_fields, phys_idx)

    leg_names = [f[0] if isinstance(f, (tuple, list)) else f
                 for f in ext_int]
    fi = _field_index(prop, leg_names[0])
    modes = diagonal_modes_from_propagator(prop, ft, num_params, fi)
    bc_mode, L = _bc_from_prop(prop, nps_sr)

    if verbose:
        A, B, N = modes[0]
        print(f'      bridge: field#{fi} modes={[(complex(a), b, n) for a, b, n in modes]} '
              f'bc={bc_mode}' + (f' L={L}' if L else ''))

    certify_max_rel = None
    certified = False
    if certify:
        records = build_pipeline_records(ft, model, prop, ext_int).get(0, [])
        certify_max_rel = certify_modes(
            modes, prop, records, ext_int, base_np_sr,
            q_samples, tau_samples)
        certified = certify_max_rel <= certify_tol
        if verbose:
            print(f'      bridge: pipeline certification max rel = '
                  f'{certify_max_rel:.2e} (tol {certify_tol:.0e}) '
                  f'-> {"PASS" if certified else "FAIL"}')
        if not certified:
            raise SpatialPropagatorError(
                f'pipeline certification failed: the shared-pipeline C(q,τ) '
                f'disagrees with the propagator modes by {certify_max_rel:.2e} '
                f'(> tol {certify_tol:.0e}).  The (A,B,N) extraction or the '
                f'diagram routing is wrong for this theory.')

    C = np.zeros((len(tau_grid), len(spatial_grid)), dtype=np.complex128)
    for it, tau in enumerate(tau_grid):
        for ix, x in enumerate(spatial_grid):
            val = 0j
            for (A, B, N) in modes:
                val += free_two_point(A, B, N, float(x), float(tau),
                                      bc_mode=bc_mode, L=L)
            C[it, ix] = val

    info = {'field_index': fi, 'modes': modes, 'bc_mode': bc_mode, 'L': L,
            'pipeline_certified': certified, 'certify_max_rel': certify_max_rel}
    return C, info


# ── 4. 1-loop tadpole (constant mass-shift self-energy) ───────────
def compute_spatial_correlator_one_loop(
        ft, model, prop, num_params, external_fields, tau_grid, spatial_grid,
        verbose=False, q_samples=(0.0, 0.8, 1.5), g_qindep_rtol=1e-4):
    """Spatial 1-loop correlator ``C(x,τ) = C₀ + δC`` for a TADPOLE
    (momentum-independent mass-shift) self-energy, routed through the SHARED
    pipeline.

    Mechanism (validated, ``docs/spatial_spikes/stageC_tadpole_spike.py``):
    the pipeline's ``ell=1`` correction at external momentum ``q`` is
    ``ell1(q,τ) = Σ_pipe(q)·∂C₀(q,τ)/∂A`` with ``Σ_pipe(q) = g·C₀(q,0)`` — the
    pipeline uses the loop edge at momentum ``q`` (un-integrated) and supplies
    the combinatorial coefficient ``g = M(Γ)·coupling`` (NOT hardcoded; for
    Allen-Cahn ``g = 3λ``).  ``g`` is q-INDEPENDENT iff the self-energy is a
    pure mass shift (a tadpole).  The CORRECT self-energy replaces the loop
    value by the momentum integral ``⟨φ²⟩₀ = ∫dℓ/2π C₀(ℓ,0) =
    free_two_point(A,B,N,0,0)`` (the §4c′ residue closed form), giving
    ``Σ = g·⟨φ²⟩₀`` and the strict-1-loop ``δC(x,τ) = Σ·∂C₀(x,τ)/∂A`` (the
    external q-FT is automatic because ``C₀(x,τ)`` is already the q-FT'd tree).

    A momentum-DEPENDENT self-energy (bubble) makes ``g`` q-dependent → raises
    NotImplementedError pointing at Stage C.5 (the per-edge ``∫dℓ`` integrator).
    Returns ``(C1_tau_x, info)``; ``info`` adds ``Sigma``, ``self_energy_coeff_g``,
    ``phi2_0``, ``A_eff_hartree``.
    """
    C0, tree_info = compute_spatial_correlator_via_pipeline(
        ft, model, prop, num_params, external_fields, tau_grid, spatial_grid,
        verbose=verbose, certify=True)
    modes = tree_info['modes']
    if len(modes) != 1:
        raise NotImplementedError(
            'spatial 1-loop v1 supports a single-mode (single-field) tree only.')
    A0, B0, N0 = modes[0]
    A0 = float(np.real(A0))
    bc_mode, L = tree_info['bc_mode'], tree_info['L']

    from msrjd.diagrams.type_assignment import build_field_index_map
    ring_var_names = list(ft._ns._ring_var_names)
    _, phys_idx = build_field_index_map(ring_var_names, ft._n_tilde)
    ext_int = _legs_to_phys_idx(external_fields, phys_idx)

    nps_sr = _norm_sr(num_params)
    base_np_sr = {kk: vv for kk, vv in nps_sr.items()
                  if str(kk) != 'Laplacian'}

    by_ell = build_pipeline_records(ft, model, prop, ext_int, max_ell=1)
    ell1 = by_ell.get(1, [])
    if not ell1:
        raise SpatialPropagatorError('no 1-loop diagrams were enumerated.')

    # Extract the self-energy coefficient g from the pipeline (q-independent
    # for a tadpole): ell1(q,0) = Σ_pipe·∂C₀/∂A, Σ_pipe = g·C₀_mom(q).
    def _C0_mom(q):
        return N0 / (A0 + B0 * q * q)

    def _dC0dA_mom(q):
        return -N0 / (A0 + B0 * q * q) ** 2

    gs = []
    for q in q_samples:
        e1 = pipeline_C_q_tau(prop, ell1, ext_int, base_np_sr, q,
                              [0.0])[0].real
        gs.append((e1 / _dC0dA_mom(q)) / _C0_mom(q))
    gs = np.array(gs)
    gmean = float(np.mean(gs))
    spread = float(np.max(np.abs(gs - gmean)) / (abs(gmean) + 1e-30))
    if spread > g_qindep_rtol:
        raise NotImplementedError(
            'spatial 1-loop v1 supports only the TADPOLE (momentum-independent '
            'mass-shift) self-energy, but the pipeline-extracted coefficient is '
            f'q-DEPENDENT (g={gs}, rel spread {spread:.2e} > {g_qindep_rtol:.0e}). '
            'A momentum-dependent self-energy (bubble) needs the per-edge ∫dℓ '
            'loop integrator (Stage C.5) — see '
            'docs/spatial_phase5_rearchitecture_plan.md.')

    # Loop integral ⟨φ²⟩₀ = ∫dℓ/2π C₀(ℓ,0) (residue closed form) and Σ.
    phi2_0 = free_two_point(A0, B0, N0, 0.0, 0.0,
                            bc_mode=bc_mode, L=L).real
    Sigma = gmean * phi2_0

    # Strict-1-loop δC(x,τ) = Σ·∂C₀(x,τ)/∂A (finite difference in the mass A).
    h = 1e-4 * max(1.0, abs(A0))
    C1 = np.array(C0, dtype=np.complex128)
    for it, tau in enumerate(tau_grid):
        for ix, x in enumerate(spatial_grid):
            fp = free_two_point(A0 + h, B0, N0, float(x), float(tau),
                                bc_mode=bc_mode, L=L)
            fm = free_two_point(A0 - h, B0, N0, float(x), float(tau),
                                bc_mode=bc_mode, L=L)
            C1[it, ix] += Sigma * (fp - fm) / (2.0 * h)

    # Self-consistent Hartree mass (resummed) for reference.
    A_eff = None
    try:
        import scipy.optimize as opt

        def _f(Ae):
            return Ae - (A0 + gmean * free_two_point(
                Ae, B0, N0, 0.0, 0.0, bc_mode=bc_mode, L=L).real)
        A_eff = float(opt.brentq(_f, 0.05 * abs(A0) + 1e-6,
                                 20.0 * abs(A0) + 10.0))
    except Exception:
        A_eff = None

    info = dict(tree_info)
    info.update({'one_loop': True, 'self_energy_coeff_g': gmean,
                 'g_q_spread': spread, 'phi2_0': phi2_0, 'Sigma': Sigma,
                 'A_tree': A0, 'A_eff_hartree': A_eff})
    if verbose:
        print(f'      1-loop tadpole: g={gmean:.6f} (q-spread {spread:.1e}) '
              f'⟨φ²⟩₀={phi2_0:.6f} Σ={Sigma:.6f} '
              f'A_eff(Hartree)={A_eff}')
    return C1, info
