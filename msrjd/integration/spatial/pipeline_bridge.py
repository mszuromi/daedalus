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
     structure ``Σ_α κ_α/(μ_α+D_α q²)·e^{-(μ_α+D_α q²)|τ|}`` read from the
     propagator (``ac_mass``, ``ac_diffusion``) and the noise sector — the
     bridge between "the diagrams are right" and "the modes are right";

  3. do the external ``q → x`` FT analytically: ``C(x, τ) = Σ_α
     free_two_point(μ_α, D_α, κ_α; x, τ)`` — each mode's q-FT IS the
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
import os

import numpy as np
from sage.all import SR

from msrjd.integration.spatial.heat_kernel import SpatialPropagatorError
from msrjd.integration.spatial.spatial_correlator import (
    extract_noise_coefficients, free_two_point,
)


# ── Notation (code ↔ paper App. B) ───────────────────────────────────
#   Rcal     𝓡(k_e)  per-edge derivative-vertex form-factor polynomial
#   Bcal     𝓑(w)    external quadratic form (= D·Q_eff)  [wick-moment scope]
#   Lam      Λ       loop / first-Symanzik matrix (in the form-factor moment math)
#   mu,D,kap  μ,D_i,κ  per-mode mass/diffusion/noise (was A,B,N — Tier 4a)
#   Scal     𝒮(Γ)    symmetry factor — prose 𝒮(Γ); local var Scal; dict key 'M' kept
# ─────────────────────────────────────────────────────────────────────


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
        # try the natural name AND the 'd'-prefixed fluctuation name (a user
        # passes ('b',1); the ring key is ('db',1))
        cands = (name, 'd' + name)
        bases = (base, 'd' + base)
        match = None
        for key in keys:
            kn = str(key[0])
            if kn in cands or kn.rstrip('0123456789') in bases:
                match = key
                break
        if match is None:
            # NEVER fall back silently (a wrong leg would quietly compute a
            # DIFFERENT correlator — e.g. C_aa instead of C_ab)
            raise SpatialPropagatorError(
                f'external leg {leg!r} does not name any physical field '
                f'(available: {keys}).')
        out.append(match)
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


# ── 1. per-mode (mu, D, kap) structure ───────────────────────────────
def diagonal_modes_from_propagator(prop, ft, num_params, field_index):
    """Return ``[(mu, D, kap)]`` — the per-mode heat-kernel structure for the
    diagonal field ``field_index``.

    v1 emits a SINGLE mode: ``mu = ac_mass`` (relaxation), ``D = ac_diffusion``
    (Laplacian coefficient), ``kap`` = white-noise spectral weight from the
    ``(2, 0)`` action sector.  The list shape is what generalizes to a
    multi-mode dressed propagator.  ``num_params`` may be str- or SR-keyed.
    """
    nps_sr = _norm_sr(num_params)
    nps_str = {str(kk): vv for kk, vv in num_params.items()}

    mu_expr = prop['ac_mass'][field_index]
    D_expr = prop['ac_diffusion'][field_index]
    mu = complex(SR(mu_expr).subs(nps_sr))
    Dsub = SR(D_expr).subs(nps_sr)
    D = float(Dsub.real() if hasattr(Dsub, 'real') else Dsub)
    if D <= 0.0:
        if D < 0.0:
            raise SpatialPropagatorError(
                f'field index {field_index} has NEGATIVE diffusion D={D}: the '
                f'spatial operator is anti-diffusive / ill-posed.  Check the '
                f'sign of the Laplacian term (should be "- D*Laplacian", D>0).')
        raise SpatialPropagatorError(
            f'field index {field_index} has zero diffusion (D=0): it is a '
            f'time-only (spatial_dim=0) field with no heat-kernel mode.  A '
            f'spatial correlator is only defined for dim >= 1.')
    noise = extract_noise_coefficients(ft, nps_str)
    kap = noise.get(field_index)
    if kap is None:
        raise SpatialPropagatorError(
            f'no white-noise coefficient for field index {field_index} '
            f'(noise sector empty?). Available: {noise}')
    return [(mu, D, float(kap))]


def _modes_C_q_tau(modes, qval, taus):
    """Reference ``C(q, τ) = Σ_α κ_α/(μ_α+D_α q²)·e^{-(μ_α+D_α q²)|τ|}``."""
    taus = np.asarray(taus, dtype=float)
    out = np.zeros(taus.shape, dtype=np.complex128)
    for (mu, D, kap) in modes:
        m = mu + D * qval * qval
        out += (kap / m) * np.exp(-m * np.abs(taus))
    return out


# ── 2. run the SHARED pipeline at Laplacian = -q² ─────────────────
def build_pipeline_records(ft, model, prop, external_fields, max_ell=0, k=2,
                           verbose=False, header='[spatial pipeline]'):
    """Enumerate + classify the (q-independent) diagram topology ONCE.

    Returns ``{ell: [(typed_diagram, scalar_prefactor), ...]}`` for
    ``compute_correction_td``.  Uses the exact entry points
    ``pipeline/compute.py`` uses (the SAME ``enumerate_unique_diagrams`` /
    ``classify_coefficient_factors`` the time-only path runs), so this is the
    real shared diagram machinery — lazy-imported to avoid any import cycle.

    ``header`` is the top verbose line's prefix (``None`` suppresses it so a
    caller can print its own staged ``[N/7]`` header); the per-``ell`` detail
    lines always print when ``verbose``.
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

    if verbose and header is not None:
        print(f'{header} enumerate prediagrams + typed diagrams '
              f'(k={k}, max_ell={max_ell}) — the SAME enumerate_unique_diagrams '
              f'the temporal path runs [taylor_order='
              f'{getattr(ft, "taylor_order", "?")}, vertices={len(vtypes)}, '
              f'sources={len(stypes)}]...')
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
    if verbose:
        for ell in sorted(by_ell):
            prefs = [str(p) for _, p in by_ell[ell]]
            print(f'        ell={ell}: {len(by_ell[ell])} typed diagram(s); '
                  f'𝒮(Γ)·prefactor(s) = {prefs}')
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
        tau_samples=(0.5, 1.0), certify_tol=1e-8, q_cut=40.0, n_q=2000,
        enum_verbose=None, stage_headers=False):
    """Drop-in alternative to :func:`compute_spatial_correlator_tree` that
    ROUTES THROUGH THE SHARED PIPELINE.

    Steps: read the per-mode ``(mu, D, kap)`` from the propagator; (optionally)
    CERTIFY them against the pipeline's diagram-based ``C(q, τ)`` at a few
    sample momenta; then build ``C(x, τ)`` by the analytic ``q → x`` FT
    ``Σ_α free_two_point(μ_α, D_α, κ_α; x, τ)`` (exact at ``τ = 0``).

    Returns ``(C_tau_x, info)`` mirroring the bespoke API; ``info`` adds
    ``pipeline_certified`` and ``certify_max_rel``.
    """
    if not prop.get('spatial_dim'):
        raise SpatialPropagatorError('propagator has no spatial block.')
    if prop.get('G_tx_sym') is None:
        # Diagonal Tier-1 heat-kernel block unavailable (e.g. off-diagonal
        # coupling).  If the inverse propagator is COUPLED but scalar-diffusion,
        # route to the spectral-Lyapunov coupled tree-level driver (Dyson 3b);
        # otherwise re-raise with the original reason.
        why = prop.get('spatial_tier1_error', 'no closed-form spatial block')
        try:
            return compute_coupled_tree_correlator(
                ft, model, prop, num_params, external_fields, tau_grid,
                spatial_grid, verbose=verbose)
        except (NotImplementedError, SpatialPropagatorError) as e:
            raise NotImplementedError(
                'pipeline bridge: the diagonal Tier-1 heat-kernel propagator is '
                f'unavailable ({why}) and the coupled scalar-diffusion path does '
                f'not apply ({type(e).__name__}: {e}).')

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

    # The diagonal per-mode form below is the AUTO-correlator ⟨φ_i φ_i⟩ of one
    # field.  A CROSS-correlator ⟨φ_i φ_j⟩ (external legs on two different
    # fields) is the only thing it cannot represent — it would wrongly
    # reconstruct ⟨φ_i φ_i⟩ — so route those to the genuine coupled
    # scalar-diffusion correlator (the full N×N spectral-Lyapunov 2-point,
    # which carries the off-diagonal drift AND noise).  Auto-correlators are
    # left on the fast diagonal path even when the noise is off-diagonal:
    # for diagonal drift the cross noise does not enter ⟨φ_i φ_i⟩ at tree
    # (C_ii = N_ii/2μ_i), and the diagonal path works at any d while the
    # coupled driver is d=1 only.  (Off-diagonal DRIFT already routes earlier
    # via ``prop['G_tx_sym'] is None``.)
    _cross_legs = (len(leg_names) >= 2
                   and _field_index(prop, leg_names[1]) != fi)
    if _cross_legs:
        return compute_coupled_tree_correlator(
            ft, model, prop, num_params, external_fields, tau_grid,
            spatial_grid, verbose=verbose)

    modes = diagonal_modes_from_propagator(prop, ft, num_params, fi)
    bc_mode, L = _bc_from_prop(prop, nps_sr)

    if verbose and stage_headers:
        print('[5/7] (spatial) Read per-mode (mu,D,kap) from the propagator '
              '+ certify vs the shared-pipeline C(q,τ)...')
    if verbose:
        mu, D, kap = modes[0]
        print(f'      modes: field#{fi}={[(complex(a), b, n) for a, b, n in modes]} '
              f'bc={bc_mode}' + (f' L={L}' if L else ''))

    certify_max_rel = None
    certified = False
    if certify:
        ev = verbose if enum_verbose is None else enum_verbose
        records = build_pipeline_records(
            ft, model, prop, ext_int, verbose=ev).get(0, [])
        if verbose:
            print(f'      certify Phase J (compute_correction_td) at '
                  f'q={list(q_samples)} → tree modes vs diagram C(q,τ)...')
        certify_max_rel = certify_modes(
            modes, prop, records, ext_int, base_np_sr,
            q_samples, tau_samples)
        certified = certify_max_rel <= certify_tol
        if verbose:
            print(f'      certify: max rel = '
                  f'{certify_max_rel:.2e} (tol {certify_tol:.0e}) '
                  f'-> {"PASS" if certified else "FAIL"}')
        if not certified:
            raise SpatialPropagatorError(
                f'pipeline certification failed: the shared-pipeline C(q,τ) '
                f'disagrees with the propagator modes by {certify_max_rel:.2e} '
                f'(> tol {certify_tol:.0e}).  The (mu,D,kap) extraction or the '
                f'diagram routing is wrong for this theory.')

    if verbose and stage_headers:
        print('[6/7] (spatial) Tree level — no loop diagrams to enumerate.')
        print('[7/7] (spatial) Analytic q→x FT: Σ_modes free_two_point(mu,D,kap; x,τ) '
              f'on {len(tau_grid)} τ × {len(spatial_grid)} x points...')
    d = int(prop.get('spatial_dim', 1))
    C = np.zeros((len(tau_grid), len(spatial_grid)), dtype=np.complex128)
    if d == 1:
        # d=1: the analytic erf/heat-kernel q→x FT (exact at τ=0, no ringing).
        for it, tau in enumerate(tau_grid):
            for ix, x in enumerate(spatial_grid):
                val = 0j
                for (mu, D, kap) in modes:
                    val += free_two_point(mu, D, kap, float(x), float(tau),
                                          bc_mode=bc_mode, L=L)
                C[it, ix] = val
    else:
        # d≥2: the radial/Hankel q→x transform of the momentum-space correlator
        # Σ_modes kap/(mu+Dq²) e^{−(mu+Dq²)|τ|}, truncated at q_cut (Regime 1 — a
        # physical cutoff; the continuum limit is q_cut→∞ with fine n_q).
        from msrjd.integration.spatial.spatial_correlator import radial_inverse_ft
        qg = np.linspace(q_cut / (4 * n_q), q_cut, n_q)
        xs = np.array([float(x) for x in spatial_grid])
        m_modes = [(float(np.real(mu)), float(np.real(D)), float(np.real(kap)))
                   for (mu, D, kap) in modes]
        for it, tau in enumerate(tau_grid):
            at = abs(float(tau))
            Cq = np.zeros_like(qg)
            for (mu, D, kap) in m_modes:
                m = mu + D * qg * qg
                Cq += (kap / m) * np.exp(-m * at)
            C[it, :] = radial_inverse_ft(qg, Cq, xs, d)

    info = {'field_index': fi, 'modes': modes, 'bc_mode': bc_mode, 'L': L,
            'spatial_dim': d,
            'pipeline_certified': certified, 'certify_max_rel': certify_max_rel}
    return C, info


def compute_coupled_tree_correlator(
        ft, model, prop, num_params, external_fields, tau_grid, spatial_grid,
        *, q_cut=60.0, n_q=6000, n_modes=600, verbose=False):
    """Coupled-field tree-level ``C_ij(x, τ)`` via the spectral-Lyapunov 2-point
    (Dyson step 3a) + a ``q → x`` FT — for theories whose inverse propagator has
    OFF-DIAGONAL coupling (so the diagonal heat-kernel block is absent) but whose
    diffusion is SCALAR (``𝒟̂ = 0``, exact at ``n=0``, no Dyson series).

    Reads ``prop['K_ft']`` (always built by ``build_propagator`` even when the
    diagonal Tier-1 block is rejected), extracts the reaction matrix ``M``,
    diffusion ``𝒟`` and drift ``V`` (``heat_kernel.reaction_diffusion_matrices``),
    requires scalar ``𝒟`` and ``V=0``, extracts the noise matrix ``N``
    (``spatial_correlator.extract_noise_matrix``), then for the ``(i, j)`` external
    legs FTs ``C_ij(q,τ) = coupled_two_point(ref, N, q², τ)`` to real space.

    Returns ``(C, info)`` mirroring :func:`compute_spatial_correlator_via_pipeline`
    (``C`` is ``[len(tau_grid), len(spatial_grid)]`` for the external pair).
    """
    from msrjd.integration.spatial.heat_kernel import (
        reaction_diffusion_matrices, SpatialPropagatorError)
    from msrjd.integration.spatial.spectral_propagator import (
        build_reference, coupled_two_point)
    from msrjd.integration.spatial.spatial_correlator import extract_noise_matrix

    K_ft = prop.get('K_ft')
    if K_ft is None:
        raise SpatialPropagatorError('coupled tree correlator: prop has no K_ft.')
    d = int(prop.get('spatial_dim', 1))
    if d not in (1, 2, 3):
        raise NotImplementedError(
            f'coupled tree correlator: d must be 1, 2 or 3 (got {d}).')

    ns = ft._ns
    omega = prop['omega']
    k_var = SR.var('k')
    lap_sym = ns.Laplacian
    grad_sym = getattr(ns, 'GradX', None)
    nps_sr = _norm_sr(num_params)
    nps_str = {str(kk): vv for kk, vv in num_params.items()}

    M_sr, D_sr, V_sr = reaction_diffusion_matrices(K_ft, omega, k_var, lap_sym, grad_sym)
    n = int(M_sr.nrows())

    def _num(sm):
        return np.array([[complex(SR(sm[a, b]).subs(nps_sr)).real for b in range(n)]
                         for a in range(n)], dtype=float)

    M, Dm, Vm = _num(M_sr), _num(D_sr), _num(V_sr)
    if not np.allclose(Vm, 0.0, atol=1e-9):
        raise NotImplementedError(
            'coupled tree correlator: drift (V≠0) not supported (scalar-diffusion '
            'coupled v1).')
    # Reference diffusion D₀: user override (.reference_diffusion(D0), stored
    # in the model) else trace/N (build_reference default).
    _ref_D0 = (model.get('spatial') or {}).get('reference_diffusion')
    ref = build_reference(M, Dm, D0=_ref_D0)
    dyson_order = None
    if not ref.is_scalar_diffusion:
        # UNEQUAL diffusion (𝒟̂≠0): consume the Dyson policy (builder D-4 —
        # model['spatial']['dyson'], or the SPATIAL_DYSON_ORDER env override).
        _env = os.environ.get('SPATIAL_DYSON_ORDER')
        pol = ({'mode': 'fixed', 'order': int(_env)} if _env else
               (model.get('spatial') or {}).get('dyson') or {'mode': 'off'})
        if pol.get('mode') != 'fixed':
            raise SpatialPropagatorError(
                'coupled tree correlator needs scalar diffusion (𝒟̂=0); unequal '
                'diffusion requires the Dyson–Duhamel series — set a truncation '
                'order with SpatialTheoryBuilder.dyson_order(N) (or the '
                'SPATIAL_DYSON_ORDER env) to enable the dressed propagator.')
        dyson_order = int(pol['order'])
        # series convergence: the insertion 𝒟̂|k|² vs the reference D₀|k|²
        rho = float(np.max(np.abs(np.linalg.eigvals(ref.Dhat)))) / ref.D0
        if rho >= 1.0:
            raise SpatialPropagatorError(
                f'Dyson series divergent: ‖𝒟̂‖/D₀ = {rho:.3f} >= 1 (the '
                f'large-|k| insertion outgrows the reference).  Choose a '
                f'better D₀ via .reference_diffusion() or reformulate.')
        if verbose and rho > 0.6:
            print(f'[coupled-tree] WARNING: ‖𝒟̂‖/D₀ = {rho:.2f} — slow Dyson '
                  f'convergence; consider a larger truncation order.')
    N = extract_noise_matrix(ft, nps_str)

    # External legs → physical field indices (i, j).  Robust to the
    # 'd'-prefixed fluctuation names compute_cumulants passes (e.g. 'da'→'a').
    phys_names = list(prop['ring_gen_names'][prop['nf']:])

    def _leg_idx(spec):
        nm = str(spec[0] if isinstance(spec, (tuple, list)) else spec)
        for cand in (nm, nm[1:] if nm.startswith('d') else 'd' + nm):
            base = cand.rstrip('0123456789')
            for idx, pn in enumerate(phys_names):
                if cand == pn or base == pn.rstrip('0123456789'):
                    return idx
        # NEVER default silently — a wrong leg index computes a different C_ij
        raise SpatialPropagatorError(
            f'external leg {spec!r} does not name any physical field '
            f'(available: {phys_names}).')

    i = _leg_idx(external_fields[0])
    j = _leg_idx(external_fields[-1])

    # Boundary from the MODEL (the coupled prop skips build_spatial_propagator,
    # so it carries no bc_mode/bc_params).
    bnd = model.get('boundary') or {}
    bc_mode = bnd.get('mode', 'infinite')
    L = None
    if bc_mode == 'periodic':
        lname = bnd.get('length')
        if isinstance(lname, str):
            _lkey = SR.var(lname)
            if _lkey in nps_sr:
                L = float(nps_sr[_lkey])
            else:
                # auto-created PBC length (inline-number shortcut backs it
                # with a hidden parameter carrying the default)
                _pdef = next((p.get('default')
                              for p in (model.get('parameters') or [])
                              if p.get('name') == lname), None)
                if _pdef is None:
                    raise SpatialPropagatorError(
                        f'periodic length parameter {lname!r} has no value in '
                        f'num_params and no model default.')
                L = float(_pdef)
        elif lname is not None:
            L = float(lname)
    taus = [float(t) for t in tau_grid]
    xs = np.array([float(x) for x in spatial_grid])
    C = np.zeros((len(taus), len(xs)), dtype=np.complex128)

    if dyson_order is not None:
        # dressed (unequal-𝒟) q-space 2-point, truncated at the policy order
        from msrjd.integration.spatial.dyson_dressing import dressed_tree_C

        def _Cq(q, tau):
            return dressed_tree_C(q, tau, M, ref.Dhat, ref.D0, N,
                                  dyson_order)[i, j]
    else:
        def _Cq(q, tau):
            return coupled_two_point(ref, N, q * q, tau)[i, j]

    if d >= 2:
        # d≥2: radial/Hankel q→x transform of the ISOTROPIC coupled C_ij(q,τ)
        # — the same ``radial_inverse_ft`` the diagonal d≥2 path uses.  ``_Cq``
        # is already dimension-agnostic (it depends on |k|²=q²), so this covers
        # the scalar-𝒟 case, the Dyson-dressed case, and any external (i,j).
        # Periodic BC at d≥2 (image lattice) is a separate task — infinite only.
        if bc_mode == 'periodic':
            raise NotImplementedError(
                'coupled tree correlator: periodic BC at d≥2 is not supported '
                'yet (infinite-domain only).')
        from msrjd.integration.spatial.spatial_correlator import (
            radial_inverse_ft)
        qg = np.linspace(q_cut / (4 * n_q), q_cut, n_q)
        for it, tau in enumerate(taus):
            Cq = np.array([_Cq(q, tau) for q in qg], dtype=complex)
            C[it, :] = radial_inverse_ft(qg, Cq, xs, d)
    elif bc_mode == 'periodic' and L:
        Lf = float(L)
        qs = 2.0 * np.pi * np.arange(-n_modes, n_modes + 1) / Lf      # discrete modes
        for it, tau in enumerate(taus):
            Cq = np.array([_Cq(q, tau) for q in qs])
            for ix, x in enumerate(xs):
                C[it, ix] = (1.0 / Lf) * np.sum(np.exp(1j * qs * x) * Cq)
    elif dyson_order is not None or os.environ.get('SPATIAL_FORCE_NUMERICAL_FT'):
        # dressed (unequal-𝒟) path, or the cross-check escape hatch: brute
        # cosine-FT on a finite q-grid (q_cut truncation ~ N_ij/(π·D₀·q_cut))
        qg = np.linspace(q_cut / (4 * n_q), q_cut, n_q)               # cosine FT
        dq = qg[1] - qg[0]
        for it, tau in enumerate(taus):
            Cq = np.array([_Cq(q, tau) for q in qg])
            for ix, x in enumerate(xs):
                C[it, ix] = (1.0 / np.pi) * np.sum(np.cos(qg * x) * Cq) * dq
    else:
        # ANALYTIC spectral IFT (exact — no q-grid, no q_cut truncation).
        # C(q,τ≥0) = Σ_{αβ} (P_α N P_βᵀ)/(m_α+m_β+2D₀q²)·e^{−(m_α+D₀q²)τ}; each
        # (α,β) term is a single-mode correlator with denominator mass
        # μ_d=(m_α+m_β)/2 and an extra factor e^{−(m_α−μ_d)τ}:
        #   IFT = (P_αNP_βᵀ)_{ij}·e^{−(m_α−μ_d)τ}·free_two_point(μ_d, D₀, ½; x, τ).
        # τ<0 uses C_ij(−τ)=C_ji(τ).  free_two_point handles complex μ_d (the
        # erf/mpmath path); conjugate (α,β)↔(β,α)* pairs make the sum real.
        from msrjd.integration.spatial.spectral_propagator import (
            spectral_projectors as _spectral_projectors)
        eig, proj = _spectral_projectors(ref.M)
        nf = len(eig)
        PNP = {(a_, b_): proj[a_] @ np.asarray(N, float) @ proj[b_].T
               for a_ in range(nf) for b_ in range(nf)}
        _f2p_cache = {}

        def _f2p(mu_d, x, at):
            key = (complex(mu_d), float(x), float(at))
            if key not in _f2p_cache:
                _f2p_cache[key] = free_two_point(mu_d, ref.D0, 0.5,
                                                 float(x), float(at))
            return _f2p_cache[key]

        for it, tau in enumerate(taus):
            at = abs(float(tau))
            ii, jj = (i, j) if tau >= 0 else (j, i)
            for ix, x in enumerate(xs):
                val = 0.0 + 0.0j
                for a_ in range(nf):
                    for b_ in range(nf):
                        w = PNP[(a_, b_)][ii, jj]
                        if abs(w) < 1e-300:
                            continue
                        mu_d = 0.5 * (eig[a_] + eig[b_])
                        val += (w * np.exp(-(eig[a_] - mu_d) * at)
                                * _f2p(mu_d, x, at))
                C[it, ix] = val

    info = {'coupled': True, 'M': M, 'D0': ref.D0, 'Dhat': ref.Dhat, 'N': N,
            'legs': (i, j), 'bc_mode': bc_mode, 'L': L, 'spatial_dim': d}
    if dyson_order is not None:
        info['dyson_order'] = dyson_order
    return C, info


def _diagram_is_bubble(td):
    """True iff the 1-loop diagram is a momentum-DEPENDENT **bubble**: some edge
    carries a momentum MIXING the external ``q`` and the loop ``ℓ`` — a cross
    term ``q·ℓ`` (e.g. ``(q−ℓ)²``), detected as a nonzero mixed second partial
    of the edge ``k²``.  A **tadpole** (decoupled ⟨φ²⟩ loop) has every edge at
    pure ``q²``, pure ``ℓ²`` or ``0`` (no cross term) → its self-energy is
    q-independent.  Topology-agnostic: catches BOTH the φ̃φ² 2-vertex tadpole
    (with a ``k=0`` connecting line) and the φ³ 1-vertex self-loop tadpole.
    """
    from msrjd.integration.spatial.momentum_routing import route_momenta
    import sympy as _sp
    for v in route_momenta(td).edge_k2().values():
        e = _sp.expand(v)
        syms = sorted(e.free_symbols, key=str)
        for ii in range(len(syms)):
            for jj in range(ii + 1, len(syms)):
                if _sp.expand(e.diff(syms[ii]).diff(syms[jj])) != 0:
                    return True
    return False


def diagram_form_factor(td, vertex_terms, mode=None, d=1):
    """Assemble the momentum-space **form factor** ``Rcal(q,ℓ)`` a derivative-vertex
    theory puts on an ARBITRARY diagram ``td`` — **any loop order ``ell``, any
    ``k``, and any MIX of derivative-vertex types** (NOT bubble-specific and NOT
    single-type: it is a product over the diagram's interaction vertices, and
    each vertex looks up its OWN factor, so vertices "wire together" by
    construction).

    A derivative is a LOCAL per-vertex feature, so

        Rcal(q,ℓ) = ∏_{interaction vertices v}  𝔉(v),

    and the per-vertex factor sums over the derivative-vertex *types* whose
    physical-leg count matches this node (``v``):

        𝔉(v) = Σ_{type t : n_phys(t)=deg(v)}  w_t · 𝔣_t(v),

    with ``w_t`` the coupling weight (``c_t / Σ c``, from the operator-IR table
    ``ns._operator_ir_vertex_terms``; ``Σ w_t = 1`` so the prefactor's merged
    coupling reconstructs every cross term) and ``𝔣_t`` the type's kernel:

      * ``mode='composite'`` — ``f_chain`` at the **response-leg** momentum
        (``out_mom[v][0]`` — the φ² composite momentum; Model B ∇²(φ²),
        Burgers ∂ₓ(φ²)),
      * ``mode='perleg'``   — ``∏`` ``f_chain`` over the **physical-leg**
        momenta (``in_mom[v]``; KPZ (∂ₓφ)²),

    where ``f_chain`` is the Fourier factor (``Lap → −p²``, ``Dx → i p`` in 1-D).
    A node whose physical-leg count matches NO derivative type contributes ``1``
    (a plain vertex, e.g. Allen-Cahn's φ³).  An empty table → ``1`` (plain
    diagram).  Returns a sympy expr in the routing symbols ``ℓ₀…q₀…``.

    ``vertex_terms`` is the (numeric-weight) table
    ``[{'weight','n_phys','chain','mode'}, …]``.  **Backward-compatible**: a bare
    ``op_chain`` tuple + ``mode=`` kwarg is accepted as a single term with
    weight 1 applied to every interaction node (the old single-type call) —
    validated φ̃φ² ``Σ_R`` bubble → ``q₀²(q₀−ℓ₀)²``, L=2 vs brute ``∫dℓ₀dℓ₁``
    1e-14."""
    import sympy as _sp
    from msrjd.integration.spatial.momentum_routing import route_momenta
    if isinstance(vertex_terms, tuple):           # backward-compat single chain
        vertex_terms = [{'weight': 1, 'n_phys': None, 'chain': vertex_terms,
                         'mode': mode or 'composite'}]
    rr = route_momenta(td)
    leaves = set(td.prediagram[2])
    out_mom = {}        # vertex -> [outgoing-edge momenta]  (response leg)
    in_mom = {}         # vertex -> [incoming-edge momenta]  (physical legs)
    for (u, v, _l), mom in rr.edge_momenta.items():
        out_mom.setdefault(u, []).append(mom)
        in_mom.setdefault(v, []).append(mom)
    iverts = [n for n in out_mom if n not in leaves and len(out_mom[n]) == 1]

    def _comp(p, alpha):
        # component α of a scalar routed momentum p (linear in lᵢ,qⱼ): the SAME
        # combo per spatial axis, lᵢ→lᵢ_α, qⱼ→qⱼ_α (d-independent routing).
        return p.subs({s: _sp.Symbol(f'{s}_{alpha}') for s in p.free_symbols})

    def _f_chain(chain, p):
        # d=1: scalar momentum p (lₐₚ→−p², ∂ₓ→ip).  d≥2: vector — Lap→−|p|²=
        # −Σ_α p_α², Dx_i→i·p_i (the i-th component), built from per-axis symbols.
        f = _sp.Integer(1)
        for entry in chain:
            if entry[0] == 'Lap':
                f *= (-p ** 2 if d == 1
                      else -sum(_comp(p, a) ** 2 for a in range(d)))
            elif entry[0] == 'Dx':
                ax = int(entry[1]) if len(entry) > 1 else 0
                f *= (_sp.I * p if d == 1 else _sp.I * _comp(p, ax))
            else:
                raise NotImplementedError(
                    f"diagram_form_factor: operator {entry[0]!r} not "
                    f"supported (only Lap, Dx).")
        return f

    def _term_factor(term, n):
        if term['mode'] == 'composite':
            return _f_chain(term['chain'], out_mom[n][0])     # response-leg momentum
        f = _sp.Integer(1)                                    # perleg
        for p in in_mom.get(n, []):
            f *= _f_chain(term['chain'], p)
        return f

    Rcal = _sp.Integer(1)
    for n in iverts:
        n_phys = len(in_mom.get(n, []))
        matched = [t for t in vertex_terms
                   if t.get('n_phys') is None or t['n_phys'] == n_phys]
        if not matched:
            continue                              # plain (non-derivative) vertex → 1
        node = _sp.Integer(0)
        for t in matched:
            node += _sp.sympify(t['weight']) * _term_factor(t, n)
        Rcal *= node
    return _sp.expand(Rcal)


# Backward-compatible alias (the function used to be bubble-specific in name).
bubble_loop_form_factor = diagram_form_factor


def _min_gh_order(Rcal, loop_syms):
    """Minimal Gauss–Hermite order that integrates the POLYNOMIAL form factor ``Rcal``
    EXACTLY over the loop Gaussian.  GH(n) is exact for degree ≤ 2n−1; after the
    Cholesky map ``ℓ = ℓ̄ + Ch·Z`` a monomial of TOTAL loop-degree ``D`` can place
    degree ``D`` on a single ``Z`` (e.g. ``ℓ₀ℓ₁ → Z₀²``), so ``n = ⌈(D+1)/2⌉`` with
    ``D`` the total degree of ``Rcal`` in the loop momenta.  Returns a safe high
    fallback (6) if ``Rcal`` is not a polynomial in the loop symbols."""
    import sympy as _sp
    if not loop_syms:
        return 1                                   # constant in ℓ → 1 node is exact
    try:
        D = int(_sp.Poly(_sp.expand(Rcal), *loop_syms).total_degree())
    except Exception:
        return 6                                   # non-polynomial → caller's default
    return max(1, (D + 2) // 2)                     # ⌈(D+1)/2⌉


def _isserlis(idx, Sget):
    """Symbolic Isserlis/Wick moment ``E[∏_a ξ_{idx[a]}]`` for a zero-mean
    Gaussian with covariance ``Sget(i,j)=Σ_ij``: sum over all perfect matchings
    of the index multiset ``idx`` of ``∏_{pairs} Σ``.  Odd length → ``0``
    (caller guarantees even).  ``∏`` over the ``(len-1)!!`` pairings."""
    import sympy as _sp
    if not idx:
        return _sp.Integer(1)
    first, rest = idx[0], idx[1:]
    return sum((Sget(first, rest[j]) * _isserlis(rest[:j] + rest[j + 1:], Sget)
                for j in range(len(rest))), _sp.Integer(0))


def _build_wick_moment(Rcal, ls, qs):
    """Analytic spatial IFT of a derivative-vertex form factor by the **joint
    `(ℓ,q)`-Gaussian moment** (Case C of docs/spatial_analytic_ift_plan.md) —
    the principled one-pass route that replaces the `q_deg+1`-node polynomial
    fit.  Returns a numpy callable ``moment_x(a, S, Bcal, xs) → (P, n_x)`` complex.

    Per chamber the IFT factorizes as ``δC(x)|_w = A·K(Bcal,x)·M_F`` with the
    heat kernel ``K`` applied by the caller and the **form-factor moment**

        M_F = E_{Q~N(c,s)} E_{ξ~N(0,Σ)}[ Rcal(a·Q+ξ, Q) ],  c=ix/2Bcal, s=1/2Bcal,

    where the loop momentum is split ``ℓ = ℓ̄(q)+ξ = a·q + ξ`` (``a=−Lam⁻¹N``,
    ``Σ=(2D·Lam)⁻¹``) and the FT source turns the external ``q`` into the complex
    Gaussian ``Q``.  Both expectations are closed-form Gaussian moments of a
    polynomial: ``E[Q^m]`` (non-central, in ``c,s``) and ``E[∏ξ^k]`` (Isserlis
    in ``Σ``).  The symbolic moment is built **once per diagram** and lambdified
    in the per-chamber numerics ``(a, Σ, Bcal, x)`` — no q-grid, no GH grid, exact.

    ``k=2`` only (single external ``q`` symbol); ``d=1``.  ``a``/``S`` are the
    FULL ``(P,L)``/``(P,L,L)`` Gaussians — only the loop indices appearing in
    ``Rcal`` are used (the marginal sub-block; ``Rcal`` is independent of the rest)."""
    import sympy as _sp
    from math import comb

    # loop indices actually present in Rcal (e.g. ['l0','l2'] → [0,2]); Rcal is
    # independent of the absent loops, so the Gaussian marginal sub-block of
    # (a, Σ) over exactly these indices is all that enters.
    lidx = [int(str(s)[1:]) for s in ls]
    p = len(ls)
    if len(qs) > 1:
        raise NotImplementedError('Wick-moment IFT is k=2 (one external q); '
                                  f'got {len(qs)} external momentum symbols.')

    Qv = _sp.Symbol('_Q')                                  # external momentum r.v.
    Xi = list(_sp.symbols('_Xi0:%d' % p)) if p else []
    asym = list(_sp.symbols('_a0:%d' % p)) if p else []
    subs = {ls[k]: asym[k] * Qv + Xi[k] for k in range(p)}
    for qq in qs:
        subs[qq] = Qv                                      # k=2: single external → Q
    G = _sp.expand(_sp.sympify(Rcal).subs(subs))

    csym, ssym = _sp.Symbol('_c'), _sp.Symbol('_s')        # mean/var of Q

    def Qmom(m):                                            # E[Q^m], Q~N(c,s)
        return sum((_sp.binomial(m, 2 * j) * _sp.factorial2(2 * j - 1)
                    * ssym ** j * csym ** (m - 2 * j)
                    for j in range(m // 2 + 1)), _sp.Integer(0))

    Ssym = {(i, j): _sp.Symbol('_S%d_%d' % (i, j))
            for i in range(p) for j in range(i, p)}

    def Sget(i, j):
        return Ssym[(i, j)] if i <= j else Ssym[(j, i)]

    xi_cache = {}

    def xi_moment(kvec):                                   # E[∏ ξ_i^{k_i}]
        if kvec in xi_cache:
            return xi_cache[kvec]
        if sum(kvec) % 2:
            val = _sp.Integer(0)
        else:
            ids = []
            for i, k in enumerate(kvec):
                ids += [i] * k
            val = _sp.expand(_isserlis(ids, Sget))
        xi_cache[kvec] = val
        return val

    Gp = _sp.Poly(G, Qv)
    EF = _sp.Integer(0)
    for (m,), coeff in Gp.terms():
        coeff = _sp.expand(coeff)
        if p and any(x in coeff.free_symbols for x in Xi):
            cp = _sp.Poly(coeff, *Xi)
            acc = sum((ccoef * xi_moment(tuple(kvec))
                       for kvec, ccoef in cp.terms()), _sp.Integer(0))
        else:
            acc = coeff                                    # no ξ → ⟨1⟩=1
        EF += Qmom(m) * acc

    Xs, Bcals = _sp.Symbol('_X'), _sp.Symbol('_Bcal')
    EF = _sp.expand(EF.subs({csym: _sp.I * Xs / (2 * Bcals), ssym: 1 / (2 * Bcals)}))

    # KEY perf factorization: EF is a polynomial in X (degree ≤ deg_q Rcal) whose
    # coefficients gₖ(a,Σ,Bcal) are x-INDEPENDENT.  Evaluate the expensive per-sample
    # gₖ ONCE (not once per x-point), then contract with the cheap X-powers — a
    # ~n_x speedup (n_x≈25 on a real grid), where the form-factor moment eval is
    # otherwise the 2-loop bottleneck (chamber quad is cheap).  ``cse=True`` folds
    # the shared 1/Bcalᵏ / aⁱ subexpressions.
    Sord = [(i, j) for i in range(p) for j in range(i, p)]
    EFp = _sp.Poly(EF, Xs) if EF != 0 else None
    Kdeg = EFp.degree() if EFp is not None else 0
    gcoeffs = ([EFp.coeff_monomial(Xs ** k) for k in range(Kdeg + 1)]
               if EFp is not None else [_sp.Integer(0)])
    gargs = asym + [Ssym[ij] for ij in Sord] + [Bcals]
    gfn = _sp.lambdify(tuple(gargs), gcoeffs, 'numpy', cse=True)

    def moment_x(a, S, Bcal, xs):
        """``a:(P,L)``, ``S:(P,L,L)``, ``Bcal:(P,)``, ``xs:(n_x,)`` → ``(P,n_x)``."""
        P = a.shape[0]
        vals = ([a[:, lidx[k]] for k in range(p)]
                + [S[:, lidx[i], lidx[j]] for (i, j) in Sord] + [Bcal])   # each (P,)
        g = gfn(*vals)                                      # list of (P,) or scalars
        gmat = np.stack([np.broadcast_to(np.asarray(gk, dtype=complex), (P,))
                         for gk in g], axis=0)              # (K+1, P)
        X = np.asarray(xs, dtype=float)
        Xpow = (X[None, :] ** np.arange(Kdeg + 1)[:, None]).astype(complex)  # (K+1,n_x)
        return np.einsum('kp,kx->px', gmat, Xpow)          # (P, n_x)

    # ── λ-grading for the Bessel-K backend ──────────────────────────────────
    # Under the radial scaling w→λw: Σ→Σ/λ and Bcal→λBcal, so each EF monomial scales as
    # λ^{−m}, m = (Σ-degree) + (1/Bcal-degree).  Grade EF by m → M_F(λ)=Σ_m λ^{−m}·EF_m,
    # so the radial integral is Σ_m EF_m·K(P−m).  Extract m via Σ→tg·Σ, Bcal→Bcal/tg.
    _tg = _sp.Symbol('_tg')
    _gsub = {Ssym[ij]: _tg * Ssym[ij] for ij in Sord}
    _gsub[Bcals] = Bcals / _tg
    EFg = _sp.expand(EF.subs(_gsub))
    EFgp = _sp.Poly(EFg, _tg) if EFg != 0 else None
    bpowers = ([int(m) for (m,), _c in EFgp.terms()] if EFgp is not None else [0])
    bcoeffs = ([EFgp.coeff_monomial(_tg ** m) for m in bpowers]
               if EFgp is not None else [_sp.Integer(0)])
    bfn = _sp.lambdify(tuple(gargs + [Xs]), bcoeffs, 'numpy', cse=True)
    _bpow = np.array(bpowers, dtype=float)

    def moment_bessel(a, S, Bcal, xs):
        """λ-graded moment for the Bessel backend.  Returns
        ``(powers:(n_m,), g:(n_m,P,n_x))`` with ``M_F(λ)=Σ_m g[m]·λ^{−powers[m]}``."""
        P = a.shape[0]
        X = np.asarray(xs, dtype=float)
        aS = [a[:, lidx[k]] for k in range(p)]
        SS = [S[:, lidx[i], lidx[j]] for (i, j) in Sord]
        outg = np.empty((_bpow.size, P, X.size), dtype=complex)
        for ix in range(X.size):
            gx = bfn(*(aS + SS + [Bcal, np.full(P, X[ix])]))
            for im in range(_bpow.size):
                outg[im, :, ix] = np.broadcast_to(
                    np.asarray(gx[im], dtype=complex), (P,))
        return _bpow, outg

    return moment_x, moment_bessel


def _formfactor_callable(td, vertex_terms, mode=None, d=1):
    """Numpy ``Rcal(ell, q)`` for the full-diagram integrator from the symbolic
    diagram form factor (:func:`diagram_form_factor`).  Possibly **complex**
    (``∂_x → ik``) — NOT forced real here (the imaginary part is resolved at the
    real-space output).  Generic in ``L`` (any ``ell``), ``n_ext`` (any ``k``),
    the MIX of derivative-vertex types, AND the spatial dimension ``d``.

    The table's weights MUST already be numeric (couplings substituted).
    ``d=1``: ``ell`` is ``(...,L)``, ``q`` is ``(n_ext,)`` — symbols ``lᵢ→ell[...,i]``,
    ``qⱼ→q[j]``.  ``d≥2``: ``ell`` is ``(...,L,d)``, ``q`` is ``(n_ext,d)`` — the
    per-axis symbols ``lᵢ_α → ell[...,i,α]``, ``qⱼ_α → q[j,α]``.  ``Rcal=0`` → zeros."""
    import sympy as _sp
    Rcal = _sp.expand(diagram_form_factor(td, vertex_terms, mode=mode, d=d))

    def _zero_ff_with_moments(squeeze_axes):
        """A form factor that is identically zero — the diagram vanishes by
        the conservation law (e.g. the ∇²(φ²) / (∂φ)² tadpole with a
        forced-zero leg momentum).  Every evaluation path must see 0, NOT
        a missing-moment error: attach zero-returning moment callables so
        the analytic IFT (scalar AND multivariate) contributes 0 just like
        the q-path GH average does."""
        def ff(ell, q):
            return np.zeros(ell.shape[:squeeze_axes], dtype=complex)
        ff.gh_order_needed = 1
        ff.q_poly_deg = 0
        ff.moment_x = lambda a, S, Bcal, xs: np.zeros(
            (a.shape[0], np.asarray(xs).shape[0]), dtype=complex)
        ff.moment_x_multi = lambda a, S, Binv, X: np.zeros(
            (Binv.shape[0], np.asarray(X).shape[0]), dtype=complex)
        ff.moment_bessel = lambda a, S, Bcal, xs: (
            np.array([0.0]),
            np.zeros((1, a.shape[0], np.asarray(xs).shape[0]), dtype=complex))
        return ff

    if d == 1:
        if Rcal == 0:
            return _zero_ff_with_moments(-1)
        ls = sorted([s for s in Rcal.free_symbols if str(s)[:1] == 'l'], key=str)
        qs = sorted([s for s in Rcal.free_symbols if str(s)[:1] == 'q'], key=str)
        fn = _sp.lambdify(tuple(ls) + tuple(qs), Rcal, 'numpy')
        nl, nq = len(ls), len(qs)

        def ff(ell, q):
            qvec = np.atleast_1d(np.asarray(q, dtype=float))
            args = ([ell[..., i] for i in range(nl)]
                    + [float(qvec[j]) for j in range(nq)])
            return fn(*args) * np.ones(ell.shape[:-1])   # complex if Rcal has i (∂_x)
        # Minimal EXACT Gauss–Hermite order: GH(n) integrates degree ≤ 2n−1
        # exactly.  Use the TOTAL degree of Rcal in the loop momenta (NOT max per-
        # variable): the Cholesky map ℓ=ℓ̄+Ch·Z mixes loops, so ℓ₀ℓ₁ → a Z₀² term
        # whose per-Z degree reaches the total degree.  This is the cheap, exact
        # speedup (e.g. 6 → 2-3 ⇒ the GH grid shrinks (n/6)^L).
        ff.gh_order_needed = _min_gh_order(Rcal, ls)
        # q-degree of ⟨Rcal⟩_ℓ ≤ total degree of Rcal (the ℓ-average turns ℓ̄∝q into q):
        # the number of q-nodes for the analytic-IFT polynomial fit (Phase 2).
        try:
            ff.q_poly_deg = int(_sp.Poly(Rcal, *(ls + qs)).total_degree()) if (ls or qs) else 0
        except Exception:
            ff.q_poly_deg = 8
        # Principled analytic IFT: the joint-(ℓ,q)-Gaussian moment (one pass per
        # diagram, no q-node loop / no GH grid).  Used by _formfactor_average_x;
        # falls back to the polynomial fit if construction fails.
        try:
            ff.moment_x, ff.moment_bessel = _build_wick_moment(Rcal, ls, qs)
        except Exception:
            ff.moment_x = ff.moment_bessel = None
        # n_ext>=2 (k>=3): the multivariate joint-(ℓ,Q)-Gaussian moment
        # for the analytic IFT (vector complex Gaussian
        # Q ~ N((i/2)·Binv·X, Binv/2)).
        ff.moment_x_multi = None
        if len(qs) >= 2:
            try:
                ff.moment_x_multi = _build_wick_moment_multi(Rcal, ls, qs)
            except Exception:
                ff.moment_x_multi = None
        return ff

    # ── d ≥ 2: symbols are lᵢ_α / qⱼ_α (loop/external index _ spatial axis) ──
    if Rcal == 0:
        return _zero_ff_with_moments(-2)
    parsed = []                       # (symbol, kind 'l'/'q', index, axis)
    for s in sorted(Rcal.free_symbols, key=str):
        nm = str(s)
        idx, ax = nm[1:].split('_')
        parsed.append((s, nm[0], int(idx), int(ax)))
    fn = _sp.lambdify(tuple(s for s, *_ in parsed), Rcal, 'numpy')

    def ff(ell, q):
        qarr = np.asarray(q, dtype=float)                # (n_ext, d)
        args = [(ell[..., idx, ax] if kind == 'l' else float(qarr[idx, ax]))
                for (_s, kind, idx, ax) in parsed]
        return fn(*args) * np.ones(ell.shape[:-2])

    # Minimal EXACT Gauss–Hermite order from the TOTAL loop-degree (see d=1
    # note): the d≥2 grid is gh_order^{L·d}, so this is a large saving.
    loopsyms = [s for (s, kind, _i, _a) in parsed if kind == 'l']
    ff.gh_order_needed = _min_gh_order(Rcal, loopsyms)
    return ff


def _prefactor_is_live(pre, num_params, tol=1e-12):
    """True if the diagram's scalar prefactor is nonzero at the saddle/params.
    A topological bubble whose prefactor ``∝ φ*²`` (e.g. the cubic-from-quartic
    vertex of a φ⁴ theory expanded around φ*=0) is DEAD at φ*=0 and must NOT
    trigger the bubble route — only LIVE bubbles do.  Substitutes ``num_params``
    (which carries the saddle ``phistar*``) into the SR prefactor; if free
    symbols remain (a param is missing) it conservatively returns True."""
    nps = _norm_sr(num_params)
    try:
        val = SR(pre).subs(nps)
        if val.free_variables():
            return True
        return abs(complex(val)) > tol
    except Exception:
        return True

def _live_bubbles(records, num_params):
    """The records that are LIVE momentum-dependent bubbles."""
    return [r for r in records
            if _diagram_is_bubble(r[0]) and _prefactor_is_live(r[1], num_params)]


# ── 4. the GENERIC 1-loop correlator — sum ALL enumerated diagrams ────
#    (the ONE path; the bespoke per-self-energy routines it replaced — the
#     constant-mass-shift tadpole and the Stage-C.5 bubble — have been removed.)
# NOTE: fork-based spatial parallelism was REMOVED — forking a Jupyter kernel on
# macOS (after matplotlib/Cocoa/BLAS init) crashes the kernel and the OS even with
# a single worker.  The spatial loop integral runs SERIALLY (see
# compute_spatial_correlator_generic).  A safe speedup path (thread-based, or
# batching the q-grid into the numpy ops) is future work — must NOT use fork.


def compute_coupled_loop_correlator(
        ft, model, prop, num_params, external_fields, tau_grid, spatial_grid,
        C0, tree_info, max_ell=1, verbose=False):
    """Loop corrections for a COUPLED scalar-diffusion theory via **spectral
    assignments** (Dyson 3c).

    With scalar diffusion every edge's momentum factor is the same heat kernel
    ``e^{−D₀k²w}`` — the Symanzik machinery is untouched.  The coupling lives in
    the time/matrix factor: each retarded segment carries ``Σ_α P_α e^{−m_α w}``
    (``m_α, P_α`` = eigenvalues / spectral projectors of the reaction matrix
    ``M``).  Expanding every segment (R edge = 1; C edge = 2 glued halves,
    :func:`full_integrator.spectral_rows`) turns one coupled diagram into a
    weighted sum of SCALAR diagrams::

        value(Γ) = pv · Σ_{{α_r}} ∏_r [P_{α_r}]_{p_r r_r} · I_spec({m_{α_r}})

    where ``(r_r, p_r)`` are the segment's (response, physical) matrix indices
    threaded through ``CEdge.fpairs``, ``pv`` is the enumeration
    ``𝒮(Γ)·prefactor`` (noise + couplings, UNCHANGED — the two-segment C
    representation natively produces the Lyapunov ``1/(m_α+m_β+2Dk²)`` so the
    single-field ``2^{−n_C}`` conversion does NOT apply), and ``I_spec`` is
    :func:`full_integrator.diagram_kinematic_spectral` (one shared quadrature/
    Symanzik pass per diagram, all assignments batched).

    Tree-level anchor: the SAME machinery applied to the tree diagram equals the
    spectral form ``Σ_{αβ} P_α N P_βᵀ/(m_α+m_β+2Dq²)`` — the Lyapunov solution
    (pinned in tests/test_coupled_loop.py).

    Scope (v1): scalar diffusion (``𝒟̂=0``; unequal diffusion → Dyson dressing),
    plain (non-derivative) vertices, infinite boundary, k=2.
    Returns ``(C1_tau_x_real, info)``.
    """
    import itertools

    from msrjd.diagrams.type_assignment import build_field_index_map
    from msrjd.integration.spatial.diagram_descriptor import diagram_to_cstack
    from msrjd.integration.spatial.full_integrator import (
        diagram_kinematic_spectral, spectral_rows, external_times_2pt,
        _is_retarded_type)
    from msrjd.integration.spatial.spectral_propagator import spectral_projectors

    M = np.asarray(tree_info['M'], dtype=float)
    Dhat = np.asarray(tree_info['Dhat'], dtype=float)
    D0 = float(tree_info['D0'])
    dress_order = 0
    if not np.allclose(Dhat, 0.0):
        # UNEQUAL diffusion: per-edge Dyson dressing (D-3 loop), policy-gated.
        _env = os.environ.get('SPATIAL_DYSON_ORDER')
        pol = ({'mode': 'fixed', 'order': int(_env)} if _env else
               (model.get('spatial') or {}).get('dyson') or {'mode': 'off'})
        if pol.get('mode') != 'fixed':
            raise NotImplementedError(
                'coupled loop corrections with unequal diffusion (𝒟̂≠0) need '
                'the Dyson–Duhamel dressing — set a truncation order with '
                'SpatialTheoryBuilder.dyson_order(N) (any N≥0; loop '
                'insertions are exact at every order, cost grows '
                'combinatorially).')
        dress_order = int(pol['order'])
        # No order cap: insertions are exact at every order via the
        # ln-derivative partition expansion (kinematic) and the
        # generalized-partial-fraction 𝓗_n labels (driver).  Cost grows
        # combinatorially with the order — that is the honest price of
        # the Dyson series, not a correctness limit.
    if tree_info.get('bc_mode') == 'periodic':
        raise NotImplementedError(
            'coupled loop corrections v1: infinite boundary only.')
    vterms_sym = getattr(ft._ns, '_operator_ir_vertex_terms', None) or []
    if vterms_sym:
        raise NotImplementedError(
            'coupled loop corrections v1: plain (non-derivative) interaction '
            'vertices only; derivative-vertex form factors are not yet combined '
            'with the spectral-assignment sum.')
    eig, proj = spectral_projectors(M)
    nf = len(eig)
    # 𝓗_n string matrices P(𝒟̂P)^n are built on demand inside
    # ``_hn_labels`` (tiny nf×nf products) — no precomputed tables.
    mu_scale = float(np.min(eig.real))
    if not mu_scale > 0.0:
        raise SpatialPropagatorError(
            f'coupled loop: reaction matrix M has an eigenvalue with '
            f'Re m = {mu_scale} <= 0 (unstable/critical theory).')

    ring_var_names = list(ft._ns._ring_var_names)
    _, phys_idx = build_field_index_map(ring_var_names, ft._n_tilde)
    ext_int = _legs_to_phys_idx(external_fields, phys_idx)
    nps_sr = _norm_sr(num_params)
    base_np_sr = {kk: vv for kk, vv in nps_sr.items() if str(kk) != 'Laplacian'}

    if verbose:
        print(f'[coupled-loop] spectral assignments over {nf} modes '
              f'(eig Re: {np.round(eig.real, 4)}), max_ell={max_ell}...')
    by_ell = build_pipeline_records(ft, model, prop, ext_int, max_ell=max_ell,
                                    verbose=verbose, header=None)
    d = int(prop.get('spatial_dim', 1))
    taus = np.asarray(tau_grid, dtype=float)
    xg = np.asarray(spatial_grid, dtype=float)
    dCx_by_ell = {el: np.zeros((len(taus), len(xg)), dtype=complex)
                  for el in range(1, max_ell + 1)}
    n_diag = n_live = 0
    for el in range(1, max_ell + 1):
        for td, pre in by_ell.get(el, []):
            n_diag += 1
            try:
                pv = float(SR(pre).subs(base_np_sr))
            except (TypeError, ValueError):
                continue                              # q-dependent prefactor
            if abs(pv) < 1e-14:
                continue
            dd = diagram_to_cstack(td)
            rows = spectral_rows(dd)
            n_rows = len(rows)
            # per-segment (resp, phys) matrix indices from CEdge.fpairs
            elems = np.empty((n_rows, nf), dtype=complex)
            fp_rows = []
            for r, (ei, e, half) in enumerate(rows):
                if not e.fpairs:
                    raise SpatialPropagatorError(
                        'coupled loop: a C-stack edge carries no propagator '
                        'field indices (fpairs) — descriptor built from a '
                        'typed diagram without propagator_indices?')
                ri_, pi_ = e.fpairs[0] if half in ('R', 'Cu') else e.fpairs[1]
                fp_rows.append((ri_, pi_))
                for a_i in range(nf):
                    elems[r, a_i] = proj[a_i][pi_, ri_]
            assign = np.array(list(itertools.product(range(nf), repeat=n_rows)),
                              dtype=int).T              # (n_rows, n_assign)
            Wgt = np.prod(elems[np.arange(n_rows)[:, None], assign], axis=0)
            keep = np.abs(Wgt) > 1e-14 * max(np.max(np.abs(Wgt)), 1e-300)
            if not np.any(keep):
                continue
            n_live += 1
            Wk = Wgt[keep]
            mass_table = eig[assign[:, keep]]           # (n_rows, n_kept)

            # ── Dyson dressing patterns (𝒟̂≠0, total order ≤ 2) ────────────
            # Each Dyson order n on a segment carries (−|k_r|²)ⁿ·𝒟̂ⁿ-strings
            # times the n-fold Duhamel convolution of e^{−mw} — the divided
            # difference f[m₀,…,m_n] of f(m)=e^{−mw} (all-equal limit
            # wⁿe^{−mw}/n!).  𝓗₁ (string P_α𝒟̂P_β): poles (m_α, +mel/Δm),
            # (m_β, −mel/Δm), confluent (μ, κ=1, mel).  𝓗₂ (string
            # P_α𝒟̂P_β𝒟̂P_γ): triple divided difference — distinct poles
            # e^{−m_iw}/∏_{j≠i}(m_i−m_j); two-equal (a,a,b):
            # (b,0,1/(b−a)²),(a,0,−1/(b−a)²),(a,1,1/(b−a)); all-equal
            # (μ,2,1/2).  Total order 2 = one row at n=2 OR two distinct
            # rows at n=1 each; the momentum insertions ride
            # diagram_kinematic_spectral(insert_rows=...).
            eig_scale = float(np.max(np.abs(eig))) or 1.0
            _tol = 1e-8 * eig_scale

            def _pf_labels(masses, mel):
                """Generalized partial fractions of the n-fold Duhamel
                convolution: time factor = L⁻¹[∏ᵢ 1/(s+mᵢ)](w) as labels
                (μ_j, κ, coeff)·mel.  Masses are grouped by closeness
                (``_tol``); a group of multiplicity m_j contributes
                A_{jp}·w^{p−1}e^{−μ_j w}/(p−1)! for p = 1..m_j, with
                A_{jp} = g_j^{(m_j−p)}(−μ_j)/(m_j−p)! and g_j(s) =
                ∏_{l≠j}(s+μ_l)^{−m_l}; derivatives of g_j via the
                log-derivative Bell expansion (g^{(p)}/g over set
                partitions of h^{(q)} = −Σ m_l(−1)^{q−1}(q−1)!/(s+μ_l)^q)."""
                from math import factorial as _fct
                from msrjd.integration.spatial.full_integrator import (
                    _set_partitions)
                groups = []                       # [mean, mult]
                for m_ in masses:
                    for gr in groups:
                        if abs(m_ - gr[0]) <= _tol:
                            gr[0] = (gr[0] * gr[1] + m_) / (gr[1] + 1)
                            gr[1] += 1
                            break
                    else:
                        groups.append([complex(m_), 1])
                out = []
                for j, (mu_j, mj) in enumerate(groups):
                    s0 = -mu_j
                    others = [(mu_l, ml) for l, (mu_l, ml)
                              in enumerate(groups) if l != j]
                    g0 = 1.0 + 0.0j
                    for mu_l, ml in others:
                        g0 = g0 / (s0 + mu_l) ** ml

                    def _hq(q):
                        return -sum(ml * ((-1.0) ** (q - 1)) * _fct(q - 1)
                                    / (s0 + mu_l) ** q for mu_l, ml in others)

                    def _gp(p):
                        if p == 0:
                            return g0
                        acc = 0.0 + 0.0j
                        for part in _set_partitions(list(range(p))):
                            t_ = 1.0 + 0.0j
                            for blk in part:
                                t_ = t_ * _hq(len(blk))
                            acc = acc + t_
                        return g0 * acc

                    for p in range(1, mj + 1):
                        A = _gp(mj - p) / _fct(mj - p)
                        out.append((mu_j, float(p - 1),
                                    mel * A / _fct(p - 1)))
                return out

            def _hn_labels(r, n_r):
                """𝓗_{n_r} labels for row r: mode strings (α₀..α_{n_r})
                with mel = [P(𝒟̂P)^{n_r}]_{p,r}, expanded by ``_pf_labels``."""
                import itertools as _it2
                ri_r, pi_r = fp_rows[r]
                out = []
                for alphas in _it2.product(range(nf), repeat=n_r + 1):
                    Mstr = proj[alphas[0]]
                    for al in alphas[1:]:
                        Mstr = Mstr @ Dhat @ proj[al]
                    mel = Mstr[pi_r, ri_r]
                    if abs(mel) < 1e-300:
                        continue
                    out.extend(_pf_labels([eig[al] for al in alphas], mel))
                return out

            def _undressed(s):
                return [(eig[a2], 0.0, elems[s, a2]) for a2 in range(nf)]

            def _mk_pattern(dressed, irows):
                """dressed: {row: labels}; others undressed.  Returns the
                (irows, Wp, mt, pt) pattern or None if all weights vanish."""
                per_row = [dressed.get(s, _undressed(s))
                           for s in range(n_rows)]
                combos = list(itertools.product(*per_row))
                Wp = np.array([np.prod([c[s][2] for s in range(n_rows)])
                               for c in combos], dtype=complex)
                kp = np.abs(Wp) > 1e-14 * max(np.max(np.abs(Wp)), 1e-300)
                if not np.any(kp):
                    return None
                mt = np.array([[c[s][0] for c in combos]
                               for s in range(n_rows)], dtype=complex)
                pt = np.array([[c[s][1] for c in combos]
                               for s in range(n_rows)], dtype=float)
                pt_k = pt[:, kp]
                return (irows, Wp[kp], mt[:, kp],
                        pt_k if np.any(pt_k) else None)

            patterns = [(None, Wk, mass_table, None)]
            if dress_order >= 1:
                # Dyson order n: distribute n insertions over the rows
                # (multisets); a row carrying n_r insertions gets the
                # 𝓗_{n_r} label set.  Cost grows combinatorially —
                # C(n_rows+n−1, n) multisets × nf^{n_r+1} mode strings per
                # dressed row — which is the honest price of the series.
                _label_cache = {}

                def _labels(r, n_r):
                    if (r, n_r) not in _label_cache:
                        _label_cache[(r, n_r)] = _hn_labels(r, n_r)
                    return _label_cache[(r, n_r)]

                for order_n in range(1, dress_order + 1):
                    for multiset in itertools.combinations_with_replacement(
                            range(n_rows), order_n):
                        counts = {}
                        for r in multiset:
                            counts[r] = counts.get(r, 0) + 1
                        dressed = {}
                        ok_ms = True
                        for r, n_r in counts.items():
                            lab = _labels(r, n_r)
                            if not lab:
                                ok_ms = False
                                break
                            dressed[r] = lab
                        if not ok_ms:
                            continue
                        p = _mk_pattern(dressed, tuple(multiset))
                        if p is not None:
                            patterns.append(p)
            n_C = sum(1 for e_ in dd.edges if e_.kind == 'C')
            nt, ns = (22, 24) if n_C <= 2 else (16, 14)
            # same accuracy overrides as the single-field generic path (_grid)
            nt = int(os.environ.get('SPATIAL_GRID_NT', nt))
            ns = int(os.environ.get('SPATIAL_GRID_NS', ns))

            # ── External-time ORIENTATION (the i≠j fix) ────────────────────
            # Enumeration sums over leg-to-leaf permutations.  For i==j the
            # mirror pair DEDUPES to one record and the retarded Γ(τ)+Γ(−τ)
            # completion restores it.  For i≠j BOTH mirror records survive, so
            # each record must be evaluated ONCE, at the times matching ITS OWN
            # leaf field order (C_ij(τ) = ⟨φ_i(t+τ)φ_j(t)⟩, the tree-driver
            # convention: leg i later for τ>0) — applying the ±τ completion
            # there would double-count and symmetrize away the genuine
            # τ-asymmetry of the cross-correlator.
            ext_i = int(phys_idx[ext_int[0]])
            ext_j = int(phys_idx[ext_int[-1]])
            leaves = list(dd.external_legs)
            if ext_i == ext_j:
                retd = _is_retarded_type(dd)

                def _ets(tau):
                    return [external_times_2pt(dd, float(tau))] + \
                        ([external_times_2pt(dd, -float(tau))]
                         if retd and tau != 0.0 else []), \
                        (2.0 if retd and tau == 0.0 else 1.0)
            else:
                lf = {leaf: int(phys_idx[fld])
                      for leaf, fld in td.external_legs.items()}
                f0, f1 = lf[leaves[0]], lf[leaves[1]]
                if (f0, f1) == (ext_i, ext_j):
                    li, lj = leaves[0], leaves[1]
                elif (f0, f1) == (ext_j, ext_i):
                    li, lj = leaves[1], leaves[0]
                else:                                  # not this correlator
                    continue

                def _ets(tau):
                    return [{li: float(tau), lj: 0.0}], 1.0

            for it, tau in enumerate(taus):
                et_list, fac = _ets(tau)
                val = 0.0
                for et in et_list:
                    for irow, Wp, mt, pt in patterns:
                        I = diagram_kinematic_spectral(
                            dd, [0.0], et, mt, D0, spatial_dim=d, xs=xg,
                            n_t=nt, n_s=ns, mu_scale=mu_scale,
                            power_table=pt, insert_rows=irow)  # (n_kept, n_x)
                        val = val + pv * (Wp @ I)
                dCx_by_ell[el][it, :] += fac * val

    C_by_order = {0: np.array(C0, dtype=np.complex128)}
    running = np.array(C0, dtype=np.complex128)
    for el in range(1, max_ell + 1):
        running = running + dCx_by_ell[el]
        C_by_order[el] = running.copy()
    C1 = running
    max_abs_imag = float(np.max(np.abs(np.imag(C1))))
    ref = float(np.max(np.abs(np.real(C1)))) or 1.0
    info = dict(tree_info)
    info.update({'one_loop': max_ell >= 1, 'generic': True,
                 'full_integrator': True, 'coupled_loop': True,
                 'max_ell': max_ell, 'n_modes': nf,
                 'eigvals': eig, 'max_abs_imag': max_abs_imag,
                 'imag_frac': max_abs_imag / ref,
                 'n_diagrams': n_diag, 'n_live_diagrams': n_live,
                 'C_by_order': {el: np.real(C).astype(float)
                                for el, C in C_by_order.items()}})
    return np.real(C1).astype(float), info


def compute_spatial_correlator_generic(
        ft, model, prop, num_params, external_fields, tau_grid, spatial_grid,
        verbose=False, q_cut=30.0, n_q=64, max_ell=1,
        parallel=True, n_workers=None):      # THREAD-based (no fork — safe in Jupyter/macOS)
    """Spatial correlator ``C(x,τ) = C₀ + δC`` to loop order ``max_ell`` via the
    **full-diagram integrator** — the ONE genuine path.

    EVERY enumerated diagram up to ``max_ell`` loops (bubble, tadpole, sunset, …)
    is mapped to the C-stack (:func:`diagram_descriptor.diagram_to_cstack`) and
    evaluated by the SAME full integral (``full_integrator.diagram_correlator``:
    Symanzik ``∫dᵈℓ`` → causal-chamber time integral → retarded+advanced sum),
    weighted by the enumeration ``𝒮(Γ)·prefactor`` (× the universal ``2^{−n_C}``).
    No Dyson convolution, no mass-shift, no diagram dropped — the loop correction
    is the honest ``Σ_Γ Γ(q,τ)`` summed over every live diagram at every
    ``1 ≤ ell ≤ max_ell``.

    Scope: **simple (non-derivative) interaction vertices**, single field.  The
    momentum integral is general in ``d`` and ``ell``; ``ell=2`` works but is
    heavier (many diagrams, higher-dim time integral — a coarser quadrature grid
    is used automatically for the bigger diagrams).  For ``d≥2`` a tadpole's
    ``⟨φ²⟩₀`` is UV-sensitive (the finite Schwinger cutoff sets the scale).
    Returns ``(C1_tau_x, info)``.
    """
    from msrjd.integration.spatial.diagram_descriptor import diagram_to_cstack
    from msrjd.integration.spatial.full_integrator import (
        diagram_correlator, diagram_correlator_x)

    if verbose:
        print('[5/7] (spatial) Certify tree modes (mu,D,kap) vs the shared-pipeline '
              'C(q,τ) at sample momenta (mode-structure check)...')
    C0, tree_info = compute_spatial_correlator_via_pipeline(
        ft, model, prop, num_params, external_fields, tau_grid, spatial_grid,
        verbose=verbose, certify=True, enum_verbose=False, stage_headers=False)
    if tree_info.get('coupled'):
        # Coupled (matrix-M) theory: the tree routed to the spectral-Lyapunov
        # driver; loops go through the spectral-assignment path (Dyson 3c).
        return compute_coupled_loop_correlator(
            ft, model, prop, num_params, external_fields, tau_grid,
            spatial_grid, C0, tree_info, max_ell=max_ell, verbose=verbose)
    modes = tree_info['modes']
    if len(modes) != 1:
        raise NotImplementedError(
            'generic spatial 1-loop v1 supports a single-mode (single-field) '
            'tree only.')
    mu0, D0, kap0 = modes[0]
    mu0 = float(np.real(mu0)); D0 = float(np.real(D0)); kap0 = float(np.real(kap0))

    # Derivative/∇ interaction vertices (operator-IR theories, e.g. Model-B
    # conserved ∇²(φ²)) deposit a momentum-space FORM FACTOR Rcal(ℓ,q) on the loop.
    # The full-diagram integrator averages it over the loop-momentum Gaussian by
    # Gauss–Hermite (exact for the polynomial Rcal).  Extracted per diagram below;
    # bubble-specific ⇒ 1-loop only (higher-loop form factors are future work).
    # The operator-IR lowering stashes a per-vertex-type TABLE: each
    # derivative-vertex type carries its coupling weight, physical-leg count,
    # operator chain, and mode ('composite' — operator on the φⁿ composite, the
    # response-leg momentum: Model B ∇²(φ²), Burgers ∂_x(φ²); 'perleg' — operator
    # on EACH physical leg: KPZ (∂_xφ)², ∏ i·p_leg).  diagram_form_factor sums the
    # matching types PER NODE (coupling-weighted), so a theory mixing distinct
    # derivative vertices (even of the same φ̃φ² signature, e.g. Model B + KPZ)
    # reconstructs every cross term — the couplings (substituted below) are real.
    vterms_sym = getattr(ft._ns, '_operator_ir_vertex_terms', None) or []
    # NOTE: the form factor is extracted + integrated GENERICALLY per diagram
    # (a product over interaction vertices; the L-dim Gauss–Hermite loop average),
    # so ANY ell works — the L=2 momentum integral matches a brute ∫dℓ₀dℓ₁ to
    # 1e-14.  The remaining real limits are gated elsewhere: d≥2
    # (full_integrator.diagram_kinematic), and field-degree≥3 composite vertices
    # (pipeline.theory_compiler).
    if vterms_sym and max_ell >= 2:
        import warnings as _warnings
        _warnings.warn(
            'derivative-vertex (form-factor) theory at max_ell>=2: correct '
            '(the L-loop form-factor average is validated), but EXPENSIVE — the '
            'GH loop-momentum grid multiplies the already-heavy ell>=2 chamber '
            'quadrature. Expect long runtimes; consider a coarser q-grid (n_q).',
            stacklevel=2)

    from msrjd.diagrams.type_assignment import build_field_index_map
    ring_var_names = list(ft._ns._ring_var_names)
    _, phys_idx = build_field_index_map(ring_var_names, ft._n_tilde)
    ext_int = _legs_to_phys_idx(external_fields, phys_idx)
    nps_sr = _norm_sr(num_params)
    base_np_sr = {kk: vv for kk, vv in nps_sr.items() if str(kk) != 'Laplacian'}

    # Substitute the numeric couplings into the symbolic weights c_t/Σc → a
    # numeric form-factor table (the lambdified Rcal then carries only momentum
    # symbols).  For a single derivative vertex the weight is 1 (unchanged).  A
    # 0/0 weight means ALL couplings of that signature are 0 — the diagram's
    # prefactor is then 0 too (dropped by the live filter), so weight 0 is safe.
    def _num_weight(w):
        try:
            return float(np.real(complex(SR(w).subs(base_np_sr))))
        except (ValueError, ZeroDivisionError, TypeError):
            return 0.0
    vterms = [{'weight': _num_weight(t['weight']),
               'n_phys': t['n_phys'], 'chain': t['chain'], 'mode': t['mode']}
              for t in vterms_sym]

    # ── Drift guard ───────────────────────────────────────────────────
    # The bilinear-Dx cross-term of a gradient nonlinearity (Burgers
    # ∂_x(φ²) → 2φ*∂_x(δφ)) has a coefficient ∝ φ*, so the propagator DRIFT
    # V vanishes at the homogeneous saddle φ*=0 and the integrator's
    # m_k = μ + D·k² is exact.  A *genuine* drift (a constant advection
    # v·∂_xφ, with V≠0 at the saddle) would need the drifting propagator
    # wired into the Symanzik momentum reduction — validated at the
    # heat-kernel (oracle) level but NOT yet in the integrator.  Refuse it
    # cleanly rather than silently dropping the drift.
    _ac_drift = prop.get('ac_drift', {}) or {}
    for _fi, _Vexpr in _ac_drift.items():
        if SR(_Vexpr).is_zero():
            continue
        try:
            _V0 = complex(SR(_Vexpr).subs(base_np_sr))
        except (TypeError, ValueError):
            continue                       # still symbolic ⇒ A,B (mass,diff) used
        if abs(_V0) > 1e-9:
            raise SpatialPropagatorError(
                f'field {_fi}: propagator DRIFT V={_V0:.4g} ≠ 0 at the saddle '
                f'(a genuine advection v·∂_xφ).  The drifting propagator is '
                f'validated at the heat-kernel (oracle) level but is not yet '
                f'wired into the momentum integrator (m_k=μ+D·k²); only φ*=0 '
                f'gradient theories (Burgers/KPZ, where V→0) run end-to-end.')

    if verbose:
        print(f'[6/7] (spatial) Enumerate prediagrams + typed diagrams → classify '
              f'coefficient factors → map to C-stack descriptors (max_ell={max_ell})...')
    by_ell = build_pipeline_records(ft, model, prop, ext_int, max_ell=max_ell,
                                    verbose=verbose, header=None)
    # map every enumerated diagram (all loop orders 1..max_ell) → (descriptor,
    # 𝒮(Γ)·prefactor value at saddle).  No filter, no shortcut.
    _d = int(prop.get('spatial_dim', 1))         # form factors are d-aware (vector legs)
    descrs = []
    for ell in range(1, max_ell + 1):
        for td, pre in by_ell.get(ell, []):
            try:
                pv = float(SR(pre).subs(base_np_sr))
            except (TypeError, ValueError):
                continue                             # q-dependent prefactor (skip)
            ff = _formfactor_callable(td, vterms, d=_d) if vterms else None
            descrs.append((diagram_to_cstack(td), pv, ff, ell))   # tag the loop order
    if not descrs:
        raise SpatialPropagatorError('no loop diagrams were enumerated.')
    live = [(dd, pv, ff, el) for dd, pv, ff, el in descrs if abs(pv) > 1e-14]
    if not live:
        raise SpatialPropagatorError('no live loop diagrams at the saddle.')
    if verbose:
        print(f'        {len(descrs)} typed diagram(s) → {len(live)} live at the '
              f'saddle ({len(descrs) - len(live)} zero-prefactor dropped)')
    # adaptive quadrature grid: coarser for the bigger (higher-n_C) diagrams so
    # ell=2 stays tractable (validated: n_t=16,n_s=14 is <0.1% on the sunset).
    # SPATIAL_GRID_NT / SPATIAL_GRID_NS override the loop grid (coarsen 2-loop to
    # make it memory-feasible — accuracy tradeoff; see the memory guard below).
    import os as _osg
    _nt_ov, _ns_ov = _osg.environ.get('SPATIAL_GRID_NT'), _osg.environ.get('SPATIAL_GRID_NS')

    def _grid(dd):
        nC = sum(1 for e in dd.edges if e.kind == 'C')
        nt, ns = (22, 24) if nC <= 2 else (16, 14)
        return (int(_nt_ov) if _nt_ov else nt, int(_ns_ov) if _ns_ov else ns)
    if verbose:
        print(f'[7/7] (spatial) Full-diagram integration: Σ_Γ 2^(-n_C)·𝒮(Γ) '
              f'∫dᵈℓ(Symanzik) ∫dt(causal chambers) → ret+adv → q→x FT '
              f'[{len(live)} live diagram(s), q-grid n_q={n_q}, '
              f'(mu,D,kap)=({mu0:.4f},{D0:.4f},{kap0:.4f})]...')

    d = int(prop.get('spatial_dim', 1))
    taus = np.asarray(tau_grid, dtype=float)
    # SPATIAL_Q_CUT / SPATIAL_N_Q env overrides — for the numerical-FT cross-check
    # only (derivative-vertex form factors with a q⁴ tail need a large q_cut for
    # the truncated trapz to converge; the analytic path has no such limit).
    import os as _os2
    q_cut = float(_os2.environ.get('SPATIAL_Q_CUT', q_cut))
    n_q = int(_os2.environ.get('SPATIAL_N_Q', n_q))
    qg = (np.linspace(0.0, q_cut, n_q) if d == 1
          else np.linspace(q_cut / (4 * n_q), q_cut, n_q))

    xg = np.asarray(spatial_grid, dtype=float)
    ells = sorted({el for _dd, _pv, _ff, el in live})
    # δC(q,τ) accumulated PER loop order — ONE integration pass over all diagrams
    # (the ℓ=L run already contains every ℓ<L diagram, so a single call yields the
    # whole cumulative progression; no need to re-run for each order).
    live_g = [(dd, pv, ff, el) + _grid(dd) for dd, pv, ff, el in live]

    # Integrator backend (switchable): 'grid' (deterministic causal-chamber product
    # quadrature, default, validated) or 'mc' (importance-sampled Monte-Carlo —
    # bounded memory, O(1/√N); the feasible ℓ≥2 path for PLAIN φⁿ theories where
    # the product grid OOMs).  See docs/spatial_loop_integral_analytic_mc.md.
    import os as _osi
    _integrator = _osi.environ.get('SPATIAL_INTEGRATOR', 'grid').strip().lower()
    _mc_n = int(float(_osi.environ.get('SPATIAL_MC_N', '1000000')))
    if _integrator == 'mc' and verbose:
        _msg = ('plain vertices' if all(rec[2] is None for rec in live_g)
                else 'WARNING — DERIVATIVE vertices are BIASED under MC (det M→0 '
                      'singularity → infinite variance); use SPATIAL_INTEGRATOR=bessel')
        print(f'        [MC] Monte-Carlo integrator, N={_mc_n:.0e} ({_msg})')
    if _integrator == 'bessel' and verbose:
        print(f'        [BESSEL] radial-Bessel-K × angular-MC integrator, N={_mc_n:.0e} '
              '(memory-safe; regularizes the det M→0 singularity → handles DERIVATIVE '
              'vertices at ℓ≥2; x=0 equal-point is UV-sensitive)')

    # ── MEMORY GUARD ──────────────────────────────────────────────────────────
    # A chamber's causal-time × Schwinger quadrature is P = n_t^{n_V}·n_s^{n_C}
    # samples (n_V internal vertices, n_C correlation edges → an (n_V+n_C)-D grid).
    # At ℓ=2 this hits the curse of dimensionality: a KPZ 2-loop diagram has
    # n_V=4, n_C=3 ⇒ P≈1.8e8/chamber at (n_t=16,n_s=14), and the (P, n_x) heat-
    # kernel array alone is tens of GB → an OOM that crashes the kernel AND the OS.
    # Refuse up-front (vs silently thrashing/crashing) with the numbers + the knobs.
    _nx = max(1, len(np.asarray(spatial_grid, dtype=float)))
    _budget_gb = float(_osg.environ.get('SPATIAL_MEM_BUDGET_GB', '6'))
    _peak_gb, _worst = 0.0, None
    for _dd, _pv, _ff, _el, _nt, _ns in live_g:
        _nV = len(_dd.internal_vertices)
        _nC = sum(1 for e in _dd.edges if e.kind == 'C')
        _P = (_nt ** _nV) * (_ns ** _nC)
        _gb = _P * _nx * 16.0 / 1e9                        # one (P, n_x) complex array
        if _gb > _peak_gb:
            _peak_gb, _worst = _gb, (_el, _nV, _nC, _nt, _ns, _P)
    if _integrator not in ('mc', 'bessel') and _peak_gb > _budget_gb:
        _el, _nV, _nC, _nt, _ns, _P = _worst
        raise SpatialPropagatorError(
            f'spatial ℓ={max(ells)} loop integration would allocate ~{_peak_gb:.0f} GB '
            f'for a single chamber and almost certainly OOM-crash the machine.  The '
            f'worst diagram (ℓ={_el}) has n_V={_nV} internal vertices + n_C={_nC} '
            f'correlation edges ⇒ an {_nV + _nC}-D causal-chamber/Schwinger '
            f'quadrature of P=n_t^{_nV}·n_s^{_nC}={_P:.1e} points at grid '
            f'(n_t={_nt}, n_s={_ns}), × {_nx} output points.  This is the curse of '
            f'dimensionality in the time/σ quadrature (NOT the form factor).  '
            f'Options: (1) use a lower max_ell — max_ell=1 is fast + validated; '
            f'(2) SPATIAL_INTEGRATOR=mc — the Monte-Carlo backend (bounded memory, '
            f'O(1/√N); validated <0.1% for PLAIN φⁿ vertices, BIASED for derivative '
            f'vertices); (3) coarsen the loop grid via SPATIAL_GRID_NT / SPATIAL_GRID_NS '
            f'(accuracy tradeoff — validate vs the simulator); (4) raise the cap '
            f'via SPATIAL_MEM_BUDGET_GB if you truly have the RAM + time.')
    # ──────────────────────────────────────────────────────────────────────────
    _all_plain = all(rec[2] is None for rec in live_g)    # rec=(dd,pv,ff,el,nt,ns)
    # ANALYTIC heat-kernel IFT covers: plain vertices (Phase 1, Case A) AND — at
    # d=1 — derivative-vertex form factors (Phase 2, Cases B/C: joint (ℓ,q)
    # Gaussian → polynomial-fit + closed-form heat-kernel q-moments).  The d≥2
    # transverse handling for derivative vertices is Phase 3 → numerical FT.
    # SPATIAL_FORCE_NUMERICAL_FT=1 keeps the numerical-FT path reachable as the
    # validated cross-check reference (it is exact only in the q_cut→∞ / n_q→∞
    # limit; the analytic path has no such truncation).
    import os as _os
    _force_num = _os.environ.get('SPATIAL_FORCE_NUMERICAL_FT', '') == '1'
    _use_analytic = (_all_plain or d == 1) and not _force_num
    if _integrator in ('mc', 'bessel') and not _use_analytic and verbose:
        print(f'        [{_integrator.upper()}] NOTE: requested backend does NOT apply here '
              '(d≥2 derivative vertices route to the numerical FT) — falling back to the '
              'grid q-loop.  Analytic mc/bessel cover plain (any d) + derivative (d=1).')
    dCx_by_ell = {el: np.zeros((len(taus), len(xg))) for el in ells}   # real-space δC(τ,x)

    if _use_analytic:
        # ── ANALYTIC heat-kernel IFT ──
        # δC(x,τ) directly: each Schwinger/chamber sample's q-Gaussian becomes a
        # heat kernel (4πB)^{−d/2}e^{−|x|²/4B} (× the form-factor q-moments for a
        # derivative vertex), summed over the (single) chamber quadrature.  NO
        # q-grid, NO numerical FT — exact, no ringing, no n_q/q_cut.
        if verbose:
            _kind = ('plain vertices' if _all_plain
                     else 'plain + d=1 derivative vertices')
            print(f'        analytic heat-kernel IFT ({_kind}) — '
                  'no q-grid / no FT (exact)')
        for _di, (dd, pv, ff, el, nt, ns) in enumerate(live_g):
            for it, tau in enumerate(taus):
                dCx_by_ell[el][it, :] += diagram_correlator_x(
                    dd, pv, xg, float(tau), mu0, D0, spatial_dim=d,
                    n_t=nt, n_s=ns, formfactor=ff,
                    method=_integrator, mc_n=_mc_n, mc_seed=1234 + _di)
    else:
        # ── NUMERICAL q→x FT (derivative-vertex form factors; Phase 2 will do
        #    these analytically via the joint (ℓ,q) Gaussian) ──
        dC_by_ell = {el: np.zeros((len(qg), len(taus)), dtype=complex) for el in ells}
        import os
        _cores = os.cpu_count() or 4
        _nw = (int(n_workers) if n_workers is not None else min(8, max(1, _cores)))
        _ntasks = len(qg) * len(live)
        # SMART thread gate: threading pays only at L≥2 (big-array numpy, GIL
        # released — ~2.5×); L=1 is dispatch-bound (0.7×, slower).  Threads only
        # (no fork — safe in Jupyter/macOS); main-thread accumulate → bit-identical.
        _heavy = any(rec[3] >= 2 for rec in live_g)
        if parallel and _heavy and _nw > 1 and _ntasks >= max(8, 2 * _nw):
            from concurrent.futures import ThreadPoolExecutor

            def _one(task):                           # one diagram's column at one q
                iq, q, dd, pv, ff, el, nt, ns = task
                col = np.array(
                    [diagram_correlator(dd, pv, q, float(tau), mu0, D0, spatial_dim=d,
                                        n_t=nt, n_s=ns, formfactor=ff) for tau in taus],
                    dtype=complex)
                return iq, el, col

            tasks = [(iq, float(q)) + rec
                     for iq, q in enumerate(qg) for rec in live_g]
            if verbose:
                print(f'        parallel: {_nw} THREAD(s) over {len(qg)} q-points × '
                      f'{len(live)} diagram(s) — no fork (GIL released in numpy)')
            with ThreadPoolExecutor(max_workers=_nw) as ex:
                for iq, el, col in ex.map(_one, tasks):
                    dC_by_ell[el][iq, :] += col
        else:
            for iq, q in enumerate(qg):
                for it, tau in enumerate(taus):
                    for dd, pv, ff, el, nt, ns in live_g:
                        dC_by_ell[el][iq, it] += diagram_correlator(
                            dd, pv, float(q), float(tau), mu0, D0, spatial_dim=d,
                            n_t=nt, n_s=ns, formfactor=ff)

        def _ft_to_x(dC_qt):                          # (n_q,n_τ) → (n_τ,n_x) real-space δC
            add = np.zeros((len(taus), len(xg)), dtype=complex)
            if d == 1:                                # δC(x,τ)=(1/π)∫₀^∞cos(qx)δC dq
                for it in range(len(taus)):
                    col = dC_qt[:, it]
                    for ix, x in enumerate(xg):
                        add[it, ix] = np.trapz(np.cos(qg * float(x)) * col, qg) / math.pi
            else:
                from msrjd.integration.spatial.spatial_correlator import radial_inverse_ft
                for it in range(len(taus)):
                    add[it, :] = radial_inverse_ft(qg, dC_qt[:, it], xg, d)
            return add
        for el in ells:
            dCx_by_ell[el] = _ft_to_x(dC_by_ell[el])

    # cumulative correlator at each order: {0: tree, 1: tree+1-loop, …, L: total}
    C_by_order = {0: np.array(C0, dtype=np.complex128)}
    running = np.array(C0, dtype=np.complex128)
    for el in ells:
        running = running + dCx_by_ell[el]
        C_by_order[el] = running.copy()
    C1 = running                                      # total = highest order

    # The physical correlator C(x,τ) is REAL.  A complex form factor (∂_x→ik,
    # Burgers/KPZ) can leave a residual imaginary part: odd-in-q diagrams
    # contribute only to the ANTISYMMETRIC correlator and are projected out by
    # the cos/radial (even) transform.  Record |Im| as a diagnostic, then take
    # the real part (no-op for the non-derivative / real-form-factor cases).
    max_abs_imag = float(np.max(np.abs(np.imag(C1))))
    ref = float(np.max(np.abs(np.real(C1)))) or 1.0
    C1 = np.real(C1).astype(float)
    C_by_order = {el: np.real(C).astype(float) for el, C in C_by_order.items()}

    info = dict(tree_info)
    info.update({'one_loop': max_ell >= 1, 'generic': True,
                 'full_integrator': True, 'max_ell': max_ell,
                 'A_tree': mu0, 'mu': mu0, 'D': D0, 'T': kap0,
                 'vertex_mode': ('+'.join(sorted({t['mode'] for t in vterms_sym}))
                                 if vterms_sym else None),
                 'max_abs_imag': max_abs_imag, 'imag_frac': max_abs_imag / ref,
                 'n_diagrams': len(descrs), 'n_live_diagrams': len(live),
                 'n_ell1_diagrams': len(by_ell.get(1, [])),
                 # cumulative C(x,τ) at each loop order {0: tree, 1: +1-loop, …}
                 # — the whole progression from ONE call (no per-ℓ re-runs).
                 'C_by_order': C_by_order})
    return C1, info


def compute_spatial_kpoint(ft, model, prop, num_params, external_fields,
                           points, max_ell=1, verbose=False,
                           n_t_loop=10, n_s_loop=12):
    """k-point spatial cumulant at external EVENTS (general-k public path).

    ``points`` : array-like ``(n_pts, k-1, 2)`` — for each evaluation
    point, the ``(x_j, tau_j)`` offsets of external slots ``1..k-1``
    relative to slot 0 (pinned at the origin; cumulants are
    translation-invariant in space and time under the stationary IC).

    Sums every enumerated diagram (ell = 0..max_ell) through the
    k-generic mapping-sum driver ``diagram_correlator_pts`` (the
    orbit-stabilizer external-Wick architecture — see
    ``tests/test_spatial_general_k.py`` for the trust battery and the
    k=3 sim anchor).  Loop descriptors with ``n_V >= 3`` use the
    reduced chamber quadrature ``(n_t_loop, n_s_loop)`` — the default
    ``22^{n_V}·24^{n_C}`` grid is prohibitive there; convergence is
    certified by the route-equivalence test.  Points sharing a
    tau-configuration are batched through one chamber integral.

    Returns ``(values (n_pts,), info)`` with per-ell breakdown in
    ``info['per_ell']``.
    """
    import numpy as _np
    from sage.all import SR as _SR
    from msrjd.diagrams.symmetry import external_wick_compensation
    from msrjd.integration.spatial.diagram_descriptor import diagram_to_cstack
    from msrjd.integration.spatial.full_integrator import (
        diagram_correlator_pts, field_respecting_mappings)

    k = len(external_fields)
    pts = _np.asarray(points, dtype=float)
    if pts.ndim == 2 and k == 2:
        pts = pts[:, None, :]
    if pts.ndim != 3 or pts.shape[1] != k - 1 or pts.shape[2] != 2:
        raise ValueError(
            f'compute_spatial_kpoint: points must be (n_pts, {k-1}, 2) '
            f'[(x_j, tau_j) offsets per non-anchor slot]; got {pts.shape}.')
    n_pts = pts.shape[0]

    # (mu, D) from the certified tree modes of the SAME field's 2-point
    # function — the same extraction the generic k=2 driver uses.
    pair = [external_fields[0], external_fields[0]]
    _xg = _np.array([0.0, 1.0])
    _tg = _np.array([0.0])
    _C0, tree_info = compute_spatial_correlator_via_pipeline(
        ft, model, prop, num_params, pair, _tg, _xg,
        verbose=False, certify=True, enum_verbose=False,
        stage_headers=False)
    modes = tree_info.get('modes')
    if not modes or len(modes) != 1:
        raise NotImplementedError(
            'compute_spatial_kpoint: single-field (one tree mode) only; '
            f'got {len(modes) if modes else 0} modes.  Coupled-field '
            'k>=3 routes through compute_coupled_kpoint.')
    mu0, D0, _kap0 = modes[0]
    mu0 = float(_np.real(mu0))
    D0 = float(_np.real(D0))
    spatial_dim = int(tree_info.get('spatial_dim', 1))

    be = build_pipeline_records(ft, model, prop, external_fields,
                                max_ell=max_ell, k=k, verbose=verbose)
    slot_fields = list(external_fields)
    per_ell = {}
    total = _np.zeros(n_pts)
    # group evaluation points by tau-configuration (slot 0 at tau=0)
    tau_groups = {}
    for ip in range(n_pts):
        key = tuple(float(t) for t in pts[ip, :, 1])
        tau_groups.setdefault(key, []).append(ip)

    for ell in sorted(be):
        acc = _np.zeros(n_pts)
        for td, p in be[ell]:
            pv = float(_SR(p).subs(num_params))
            if abs(pv) < 1e-14:
                continue
            d = diagram_to_cstack(td)
            leaf_fields = slot_fields  # enumeration emits matching multiset
            maps = field_respecting_mappings(slot_fields, leaf_fields)
            comp = external_wick_compensation(td)
            n_V = len(d.internal_vertices)
            kw = {} if n_V < 3 else {'n_t': int(n_t_loop),
                                     'n_s': int(n_s_loop)}
            for tkey, idxs in tau_groups.items():
                x_pts = _np.zeros((len(idxs), k))
                for col, j in enumerate(range(1, k)):
                    x_pts[:, j] = pts[idxs, j - 1, 0]
                t_pts = [0.0] + list(tkey)
                v = diagram_correlator_pts(
                    d, pv, x_pts, t_pts, mu0, D0,
                    spatial_dim=spatial_dim, mappings=maps, comp=comp,
                    **kw)
                acc[idxs] += v
        per_ell[ell] = acc
        total = total + acc
        if verbose:
            print(f'[spatial k={k}] ell={ell}: '
                  f'{len(be[ell])} records summed')

    info = {'per_ell': per_ell, 'mu': mu0, 'D': D0,
            'spatial_dim': spatial_dim, 'k': k, 'max_ell': max_ell}
    return total, info


def _build_wick_moment_multi(Rcal, ls, qs):
    """Multivariate (k>=3) analytic-IFT form-factor moment — the n_ext>=2
    generalization of :func:`_build_wick_moment`.

    Under the multivariate IFT the external momenta become a complex
    Gaussian VECTOR ``Q ~ N(c, S_Q)`` with

        c = (i/2) * Binv @ X,      S_Q = Binv / 2,    Binv = (D*Qeff)^{-1},

    (k=2 scalar reduction: c = iX/2Bcal, s = 1/2Bcal), and the loop split
    is ``l_k = sum_j a_kj Q_j + Xi_k`` with ``a = -Lam^{-1} N`` now (L, n).
    Writing ``Q_j = c_j + zeta_j`` with ``zeta ~ N(0, S_Q)`` independent of
    ``Xi ~ N(0, Sigma)``, the moment is ONE joint Isserlis pass over the
    block-diagonal zero-mean vector ``(Xi, zeta)`` with the means ``c_j``
    entering as symbols.

    Returns ``moment_x_multi(a, S, Binv, X) -> (P, n_x)`` complex, with
    ``a:(P,L,n)``, ``S:(P,L,L)`` (= (2D Lam)^{-1}), ``Binv:(P,n,n)``,
    ``X:(n_x,n)`` the IFT conjugates.  Vectorized over samples P, loop
    over the (short, event-list) n_x.
    """
    import sympy as _sp

    lidx = [int(str(s)[1:]) for s in ls]
    p = len(ls)
    n = len(qs)
    if n < 2:
        raise ValueError('use _build_wick_moment for n_ext == 1')

    Xi = list(_sp.symbols('_Xi0:%d' % p)) if p else []
    Ze = list(_sp.symbols('_Ze0:%d' % n))
    Cm = list(_sp.symbols('_c0:%d' % n))
    Am = {(k, j): _sp.Symbol('_a%d_%d' % (k, j))
          for k in range(p) for j in range(n)}
    subs = {}
    for k in range(p):
        subs[ls[k]] = sum(Am[(k, j)] * (Cm[j] + Ze[j])
                          for j in range(n)) + Xi[k]
    for j, qq in enumerate(qs):
        subs[qq] = Cm[j] + Ze[j]
    G = _sp.expand(_sp.sympify(Rcal).subs(subs))

    # joint covariance symbols: Xi-Xi -> Sigma, zeta-zeta -> S_Q, cross -> 0
    Ssym = {(i, j): _sp.Symbol('_S%d_%d' % (i, j))
            for i in range(p) for j in range(i, p)}
    SQsym = {(i, j): _sp.Symbol('_SQ%d_%d' % (i, j))
             for i in range(n) for j in range(i, n)}

    def Sget(i, j):
        i, j = min(i, j), max(i, j)
        if i < p and j < p:
            return Ssym[(i, j)]
        if i >= p and j >= p:
            return SQsym[(i - p, j - p)]
        return _sp.Integer(0)

    mom_cache = {}

    def joint_moment(kvec):
        if kvec in mom_cache:
            return mom_cache[kvec]
        if sum(kvec) % 2:
            val = _sp.Integer(0)
        else:
            ids = []
            for i, k in enumerate(kvec):
                ids += [i] * k
            val = _sp.expand(_isserlis(ids, Sget))
        mom_cache[kvec] = val
        return val

    gauss = Xi + Ze                               # joint zero-mean vector
    Gp = _sp.Poly(G, *gauss) if gauss else None
    EF = _sp.Integer(0)
    if Gp is None:
        EF = G
    else:
        for kvec, coeff in Gp.terms():
            EF += coeff * joint_moment(tuple(kvec))
    EF = _sp.expand(EF)

    Sord = [(i, j) for i in range(p) for j in range(i, p)]
    SQord = [(i, j) for i in range(n) for j in range(i, n)]
    gargs = ([Am[(k, j)] for k in range(p) for j in range(n)]
             + [Ssym[ij] for ij in Sord]
             + [SQsym[ij] for ij in SQord] + Cm)
    gfn = _sp.lambdify(tuple(gargs), EF, 'numpy', cse=True)

    def moment_x_multi(a, S, Binv, X):
        P = Binv.shape[0]
        X = np.asarray(X, dtype=float)
        n_x = X.shape[0]
        SQ = 0.5 * Binv                                   # (P, n, n)
        cfull = 0.5j * np.einsum('pjk,xk->pxj', Binv, X)  # (P, n_x, n)
        base = ([a[:, lidx[k], j] for k in range(p) for j in range(n)]
                + [S[:, lidx[i], lidx[j]] for (i, j) in Sord]
                + [SQ[:, i, j] for (i, j) in SQord])
        out = np.empty((P, n_x), dtype=complex)
        for ix in range(n_x):
            cv = [cfull[:, ix, j] for j in range(n)]
            out[:, ix] = np.broadcast_to(
                np.asarray(gfn(*(base + cv)), dtype=complex), (P,))
        return out

    return moment_x_multi


def compute_coupled_kpoint(ft, model, prop, num_params, external_fields,
                           points, tree_info, max_ell=1, verbose=False,
                           n_t_loop=10, n_s_loop=12):
    """k-point cumulant for a COUPLED scalar-diffusion theory at external
    EVENTS (H2 driver) — the general-k companion of
    :func:`compute_coupled_loop_correlator`, built on the same spectral-
    assignment sum but with the orbit-stabilizer external-Wick mapping
    sum replacing the k=2 orientation/±τ machinery (the two coincide at
    k=2: mixed-field leaves give singleton mappings with comp=1 — the
    per-record orientation; same-field leaves give the 2-mapping sum ÷
    comp — the ±τ completion).

    ``points`` : (n_pts, k-1, 2) of (x_j, tau_j) offsets per non-anchor
    canonical slot (slot 0 at the origin), as in
    :func:`compute_spatial_kpoint`.

    Every record (ell = 0..max_ell) flows through ONE path:
    ``value = pv * sum_assign W * I_spec`` with
    :func:`full_integrator.diagram_kinematic_spectral` (matrix-B branches
    for n_ext >= 2; NO 2^{-n_C} — the glued-segment C convention natively
    carries the Lyapunov denominator).

    Scope (v1): scalar diffusion only (Dhat = 0 — coupled unequal-D loop
    dressing is k=2: the (-|k_r|^2) insertion is gated in the kinematic),
    plain vertices, infinite boundary, stationary IC.

    Returns ``(values (n_pts,), info)`` with per-ell breakdown.
    """
    import itertools

    from msrjd.diagrams.symmetry import external_wick_compensation
    from msrjd.diagrams.type_assignment import build_field_index_map
    from msrjd.integration.spatial.diagram_descriptor import diagram_to_cstack
    from msrjd.integration.spatial.full_integrator import (
        diagram_kinematic_spectral, spectral_rows, field_respecting_mappings)
    from msrjd.integration.spatial.spectral_propagator import (
        spectral_projectors)

    M = np.asarray(tree_info['M'], dtype=float)
    Dhat = np.asarray(tree_info['Dhat'], dtype=float)
    D0 = float(tree_info['D0'])
    if not np.allclose(Dhat, 0.0):
        raise NotImplementedError(
            'compute_coupled_kpoint v1: scalar diffusion only (Dhat = 0).  '
            'Coupled unequal-D at k>=3 needs the Dyson insertion, which is '
            'k=2-gated in diagram_kinematic_spectral.')
    if tree_info.get('bc_mode') == 'periodic':
        raise NotImplementedError(
            'compute_coupled_kpoint v1: infinite boundary only.')
    if getattr(ft._ns, '_operator_ir_vertex_terms', None):
        raise NotImplementedError(
            'compute_coupled_kpoint v1: plain (non-derivative) vertices only.')

    eig, proj = spectral_projectors(M)
    nf = len(eig)
    mu_scale = float(np.min(eig.real))
    if not mu_scale > 0.0:
        raise SpatialPropagatorError(
            f'coupled k-point: reaction matrix M has Re m = {mu_scale} <= 0.')

    k = len(external_fields)
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 3 or pts.shape[1] != k - 1 or pts.shape[2] != 2:
        raise ValueError(
            f'points must be (n_pts, {k-1}, 2); got {pts.shape}.')
    n_pts = pts.shape[0]

    ring_var_names = list(ft._ns._ring_var_names)
    _, phys_idx = build_field_index_map(ring_var_names, ft._n_tilde)
    ext_int = _legs_to_phys_idx(external_fields, phys_idx)
    nps_sr = _norm_sr(num_params)
    base_np_sr = {kk: vv for kk, vv in nps_sr.items()
                  if str(kk) != 'Laplacian'}
    d = int(prop.get('spatial_dim', 1))

    be = build_pipeline_records(ft, model, prop, ext_int, max_ell=max_ell,
                                k=k, verbose=verbose, header=None)
    tau_groups = {}
    for ip in range(n_pts):
        key = tuple(float(t) for t in pts[ip, :, 1])
        tau_groups.setdefault(key, []).append(ip)

    per_ell = {}
    total = np.zeros(n_pts, dtype=complex)
    slot_fields = [tuple(f) for f in external_fields]
    for el in sorted(be):
        acc = np.zeros(n_pts, dtype=complex)
        for td, pre in be.get(el, []):
            try:
                pv = float(SR(pre).subs(base_np_sr))
            except (TypeError, ValueError):
                continue
            if abs(pv) < 1e-14:
                continue
            dd = diagram_to_cstack(td)
            rows = spectral_rows(dd)
            n_rows = len(rows)
            elems = np.empty((n_rows, nf), dtype=complex)
            for r, (ei, e, half) in enumerate(rows):
                if not e.fpairs:
                    raise SpatialPropagatorError(
                        'coupled k-point: C-stack edge without fpairs.')
                ri_, pi_ = e.fpairs[0] if half in ('R', 'Cu') else e.fpairs[1]
                elems[r, :] = [proj[a_i][pi_, ri_] for a_i in range(nf)]
            assign = np.array(
                list(itertools.product(range(nf), repeat=n_rows)),
                dtype=int).T
            Wgt = np.prod(elems[np.arange(n_rows)[:, None], assign], axis=0)
            keep = np.abs(Wgt) > 1e-14 * max(np.max(np.abs(Wgt)), 1e-300)
            if not np.any(keep):
                continue
            Wk = Wgt[keep]
            mass_table = eig[assign[:, keep]]

            leaves = list(dd.external_legs)
            leaf_fields = [tuple(td.external_legs[lf]) for lf in leaves]
            maps = field_respecting_mappings(slot_fields, leaf_fields)
            comp = external_wick_compensation(td)
            n_V = len(dd.internal_vertices)
            n_C = sum(1 for e_ in dd.edges if e_.kind == 'C')
            if n_V >= 3:
                nt, ns = int(n_t_loop), int(n_s_loop)
            else:
                nt, ns = (22, 24) if n_C <= 2 else (16, 14)
            nt = int(os.environ.get('SPATIAL_GRID_NT', nt))
            ns = int(os.environ.get('SPATIAL_GRID_NS', ns))

            for tkey, idxs in tau_groups.items():
                x_full = np.zeros((len(idxs), k))
                for j in range(1, k):
                    x_full[:, j] = pts[idxs, j - 1, 0]
                t_full = np.array([0.0] + list(tkey))
                # dedup identical (times, X) mapping configurations
                cfgs = {}
                for m in maps:
                    et = {leaves[j]: float(t_full[m[j]]) for j in range(k)}
                    X = np.stack([x_full[:, m[j]] - x_full[:, m[k - 1]]
                                  for j in range(k - 1)], axis=1)
                    ck = (tuple(sorted(et.items())), X.tobytes())
                    if ck in cfgs:
                        cfgs[ck][2] += 1
                    else:
                        cfgs[ck] = [et, X, 1]
                for et, X, mult in cfgs.values():
                    I = diagram_kinematic_spectral(
                        dd, [0.0] * (k - 1), et, mass_table, D0,
                        spatial_dim=d, xs=X, n_t=nt, n_s=ns,
                        mu_scale=mu_scale)              # (n_kept, n_x)
                    acc[idxs] += (mult * pv / comp) * (Wk @ I)
        per_ell[el] = np.real(acc).astype(float)
        total = total + acc
        if verbose:
            print(f'[coupled k={k}] ell={el}: summed')

    info = {'per_ell': per_ell, 'eig': eig, 'D0': D0, 'k': k,
            'max_ell': max_ell, 'n_modes': nf,
            'max_abs_imag': float(np.max(np.abs(np.imag(total))))}
    return np.real(total).astype(float), info
