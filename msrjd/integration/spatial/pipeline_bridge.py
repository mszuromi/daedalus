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
def build_pipeline_records(ft, model, prop, external_fields, max_ell=0, k=2,
                           verbose=False):
    """Enumerate + classify the (q-independent) diagram topology ONCE.

    Returns ``{ell: [(typed_diagram, scalar_prefactor), ...]}`` for
    ``compute_correction_td``.  Uses the exact entry points
    ``pipeline/compute.py`` uses (the SAME ``enumerate_unique_diagrams`` /
    ``classify_coefficient_factors`` the time-only path runs), so this is the
    real shared diagram machinery — lazy-imported to avoid any import cycle.
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

    if verbose:
        print(f'[spatial pipeline] FieldTheory taylor_order='
              f'{getattr(ft, "taylor_order", "?")}; vertices={len(vtypes)}, '
              f'sources={len(stypes)} — enumerating diagrams '
              f'(k={k}, max_ell={max_ell}) via enumerate_unique_diagrams...')
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
            print(f'[spatial pipeline]   ell={ell}: {len(by_ell[ell])} typed '
                  f'diagram(s); M(Γ)·prefactors = {prefs}')
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
        tau_samples=(0.5, 1.0), certify_tol=1e-8, q_cut=40.0, n_q=2000):
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
        records = build_pipeline_records(
            ft, model, prop, ext_int, verbose=verbose).get(0, [])
        if verbose:
            print(f'[spatial pipeline] Phase J (compute_correction_td) at '
                  f'q={list(q_samples)} → certifying tree modes vs the '
                  f'diagram C(q,τ)...')
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

    d = int(prop.get('spatial_dim', 1))
    C = np.zeros((len(tau_grid), len(spatial_grid)), dtype=np.complex128)
    if d == 1:
        # d=1: the analytic erf/heat-kernel q→x FT (exact at τ=0, no ringing).
        for it, tau in enumerate(tau_grid):
            for ix, x in enumerate(spatial_grid):
                val = 0j
                for (A, B, N) in modes:
                    val += free_two_point(A, B, N, float(x), float(tau),
                                          bc_mode=bc_mode, L=L)
                C[it, ix] = val
    else:
        # d≥2: the radial/Hankel q→x transform of the momentum-space correlator
        # Σ_modes N/(A+Bq²) e^{−(A+Bq²)|τ|}, truncated at q_cut (Regime 1 — a
        # physical cutoff; the continuum limit is q_cut→∞ with fine n_q).
        from msrjd.integration.spatial.spatial_correlator import radial_inverse_ft
        qg = np.linspace(q_cut / (4 * n_q), q_cut, n_q)
        xs = np.array([float(x) for x in spatial_grid])
        m_modes = [(float(np.real(A)), float(np.real(B)), float(np.real(N)))
                   for (A, B, N) in modes]
        for it, tau in enumerate(tau_grid):
            at = abs(float(tau))
            Cq = np.zeros_like(qg)
            for (A, B, N) in m_modes:
                m = A + B * qg * qg
                Cq += (N / m) * np.exp(-m * at)
            C[it, :] = radial_inverse_ft(qg, Cq, xs, d)

    info = {'field_index': fi, 'modes': modes, 'bc_mode': bc_mode, 'L': L,
            'spatial_dim': d,
            'pipeline_certified': certified, 'certify_max_rel': certify_max_rel}
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


def bubble_loop_form_factor(td, op_chain):
    """Phase 4c-2: assemble the loop-momentum **form factor** ``F(q,ℓ)`` a
    derivative-vertex theory puts on a 1-loop bubble diagram ``td``.

    Each interaction vertex carries a derivative operator (``op_chain``, e.g.
    ``(('Lap',),)`` for ``∇²(φ²)``) acting on its φ² composite; by momentum
    conservation the composite momentum equals the vertex's **response (φ̃) leg**
    momentum — which in the all-``G_R`` representation is the vertex's UNIQUE
    OUTGOING edge.  ``route_momenta`` supplies that momentum per edge, so

        F(q,ℓ) = ∏_{interaction vertices v}  f_chain( p_v ),

    with ``p_v`` the outgoing-edge momentum and ``f_chain`` the per-leg Fourier
    factor (``Lap → −p²``, ``Dx → i p`` in 1-D).  Returns a sympy expression in
    the routing symbols ``q₀`` (external) and ``ℓ₀`` (loop); an empty chain
    gives ``1`` (the plain bubble).  Validated: the φ̃φ² ``Σ_R`` bubble with a
    ``Lap`` chain returns ``q₀²(q₀−ℓ₀)²`` (matches the hand derivation).
    """
    import sympy as _sp
    from msrjd.integration.spatial.momentum_routing import route_momenta
    rr = route_momenta(td)
    leaves = set(td.prediagram[2])
    out_mom = {}
    for (u, _v, _l), mom in rr.edge_momenta.items():
        out_mom.setdefault(u, []).append(mom)
    iverts = [n for n in out_mom if n not in leaves and len(out_mom[n]) == 1]

    def _f_chain(p):
        f = _sp.Integer(1)
        for entry in op_chain:
            if entry[0] == 'Lap':
                f *= -p ** 2
            elif entry[0] == 'Dx':
                f *= _sp.I * p
            else:
                raise NotImplementedError(
                    f"bubble_loop_form_factor: operator {entry[0]!r} not "
                    f"supported (only Lap, Dx).")
        return f

    F = _sp.Integer(1)
    for n in iverts:
        F *= _f_chain(out_mom[n][0])
    return _sp.expand(F)


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

    by_ell = build_pipeline_records(ft, model, prop, ext_int, max_ell=1,
                                    verbose=verbose)
    ell1 = by_ell.get(1, [])
    if not ell1:
        raise SpatialPropagatorError('no 1-loop diagrams were enumerated.')

    # Classify BEFORE the (slow) Phase J g-extraction: a momentum-DEPENDENT
    # bubble breaks the tadpole mass-shift assumption AND would hang here —
    # evaluating a bubble's time-polytope at q=0 (exact-degenerate edges) hits
    # the close-pair slow-path.  Raise immediately so the caller routes to the
    # Stage C.5 momentum-first bubble integrator.
    if _live_bubbles(ell1, num_params):
        raise NotImplementedError(
            'spatial 1-loop has a LIVE momentum-DEPENDENT bubble self-energy (an '
            'edge carries q±ℓ with a nonzero prefactor at the saddle); the '
            'constant-mass-shift tadpole path does not apply.  Route to the '
            'Stage C.5 momentum-first bubble integrator '
            '(compute_spatial_correlator_bubble).')
    if verbose:
        print(f'[spatial pipeline] Phase J (compute_correction_td) on the '
              f'{len(ell1)} ell=1 diagram(s) at q={list(q_samples)} → '
              f'extracting the tadpole self-energy coefficient g...')

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


# ── 5. 1-loop BUBBLE (momentum-dependent self-energy) — Stage C.5 ──
def compute_spatial_correlator_bubble(
        ft, model, prop, num_params, external_fields, tau_grid, spatial_grid,
        verbose=False, q0_samples=None, q_cut=30.0, n_q=160, n_t=2000,
        g2_qindep_rtol=1e-2, formfactor=None):
    """Spatial 1-loop correlator ``C(x,τ) = C₀ + δC_bubble`` from a
    momentum-DEPENDENT **bubble** self-energy (Stage C.5), routed through the
    close-pair-free momentum-first integrator (``loop_dyson``).

    Mechanism.  The φ̃φ² 1-loop self-energy is a bubble: a retarded part
    ``Σ_R = ∫dℓ/2π G_R(q−ℓ)C(ℓ)`` and a Keldysh part ``Σ_K = ∫dℓ/2π C(ℓ)C(q−ℓ)``.
    These ``∫dℓ`` are pole-free (momentum integrals of products of
    exponentials/Lorentzians) — the ``m≥3`` close-pair bug lived only in the
    time-polytope, which this momentum-first route bypasses, so it is fast and
    robust at every q.  ``loop_dyson`` assembles the MSR Dyson equation
    ``δC(q,τ) = G_R⁰Σ_R C⁰ + G_R⁰Σ_K G_A⁰ + C⁰Σ_A G_A⁰`` (validated vs direct
    ∫dℓ to 1e-12 and vs simulation, B≈1), and this routine q-FTs it to ``(x,τ)``.

    Normalization is taken from the framework's OWN uniform-momentum bubble
    value — NO hardcoded factor.  At ``Laplacian=−q²`` the bubble diagrams sum to
    ``V_bub = 2g²N0²/m⁴`` (= ``4g²T1^unif + 2g²T2^unif`` with the pinned
    ``c_R=4, c_K=2``), so the coupling is ``g² = V_bub·m⁴/(2N0²)`` — robust,
    q-independent, and self-checked over ``q0_samples``.

    NOTE (scope): returns ONLY the momentum-dependent bubble.  A φ²-tadpole (the
    decoupled ⟨φ²⟩₀ loop on a routed ``k=0`` line → a saddle/mass shift) is a
    separate contribution handled by the tadpole machinery and is NOT added here
    — for the φ̃φ² test theory the bubble is the novel piece validated vs
    simulation.  Returns ``(C1_tau_x, info)``.
    """
    from msrjd.integration.spatial.loop_dyson import bubble_delta_C_q_tau
    from msrjd.diagrams.type_assignment import build_field_index_map

    C0, tree_info = compute_spatial_correlator_via_pipeline(
        ft, model, prop, num_params, external_fields, tau_grid, spatial_grid,
        verbose=verbose, certify=True)
    modes = tree_info['modes']
    if len(modes) != 1:
        raise NotImplementedError(
            'spatial bubble v1 supports a single-mode (single-field) tree only.')
    A0, B0, N0 = modes[0]
    A0 = float(np.real(A0)); B0 = float(np.real(B0)); N0 = float(np.real(N0))

    # NOTE: ``q0_samples`` / ``g2_qindep_rtol`` are retained for signature
    # back-compat but are now INERT — the coupling is read analytically from the
    # diagram prefactor below (W9), not sampled numerically over q0.

    ring_var_names = list(ft._ns._ring_var_names)
    _, phys_idx = build_field_index_map(ring_var_names, ft._n_tilde)
    ext_int = _legs_to_phys_idx(external_fields, phys_idx)
    nps_sr = _norm_sr(num_params)
    base_np_sr = {kk: vv for kk, vv in nps_sr.items() if str(kk) != 'Laplacian'}

    # The operator-IR unfold of a derivative vertex (e.g. −g∇²(φ²)) leaves a
    # spurious φ*·operator bilinear in the kernel (−2g·φ*·∇²); at the HOMOGENEOUS
    # saddle (the only spatial-bubble case) φ*=0, but the symbol survives into the
    # SR propagator entries and slows/hangs Phase J's compute_correction_td.
    # Substitute the saddle (φ*→its value) into the symbolic propagator so the
    # bubble path's Phase J runs on the clean propagator.
    _sad = {kk: vv for kk, vv in nps_sr.items() if 'star' in str(kk)}
    prop_b = dict(prop)
    if _sad:
        for _key in ('K_ker', 'K_ft', 'G_ft', 'adj_ft', 'D_omega', 'D_delta'):
            _M = prop.get(_key)
            if _M is None:
                continue
            try:
                prop_b[_key] = (_M.apply_map(lambda e: SR(e).subs(_sad))
                                if hasattr(_M, 'apply_map')
                                else SR(_M).subs(_sad))
            except Exception:
                pass
    prop = prop_b

    by_ell = build_pipeline_records(ft, model, prop, ext_int, max_ell=1,
                                    verbose=verbose)
    ell1 = by_ell.get(1, [])
    if not ell1:
        raise SpatialPropagatorError('no 1-loop diagrams were enumerated.')

    # classify (topology-agnostic): LIVE bubble = an edge mixes q and ℓ (q±ℓ)
    # AND the prefactor is nonzero at the saddle; everything else (tadpoles,
    # and dead φ*²-bubbles at φ*=0) is handled by the tadpole machinery.
    bubbles = _live_bubbles(ell1, num_params)
    tadpoles = [r for r in ell1 if r not in bubbles]
    if not bubbles:
        raise NotImplementedError(
            'no bubble diagrams found (all ell=1 are tadpoles) — use '
            'compute_spatial_correlator_one_loop.')

    # Coupling g, read ANALYTICALLY from the diagram M(Γ)·prefactor — NO
    # compute_correction_td (the close-pair-prone temporal Phase-J path the old
    # numerical V_bub extraction used, which hung at m≳4; W9).  The single-field
    # φ̃φ² spatial bubble (the only topology this path supports — it raises
    # otherwise) has, summed over the live bubble diagrams,
    #     Σ M(Γ)·prefactor = 24 · N0² · g²        (N0 = T, the noise amplitude),
    # so g² = (Σ prefactor) / (24·N0²).  The "24" is the φ̃φ² bubble combinatorial
    # constant — verified to recover g EXACTLY (to machine precision) against the
    # old numerical V_bub for BOTH reaction-diffusion AND the conserved ∇²(φ²)
    # derivative theory, over varying g, T, D (a pinned topology constant, exactly
    # like loop_dyson's c_R=4 / c_K=2).  The prefactor is a pure coupling monomial
    # (q-independent by construction); were it q-dependent (a Laplacian in it) the
    # ``.subs`` below would leave a free symbol and float() would raise.
    _BUBBLE_MGAMMA = 24.0
    pref_sum = sum((p for _, p in bubbles), SR(0))
    try:
        pref_val = float(SR(pref_sum).subs(base_np_sr))
    except (TypeError, ValueError) as _e:
        raise SpatialPropagatorError(
            f'bubble M(Γ)·prefactor {pref_sum} is not a pure coupling constant '
            f'at the saddle (q-dependent / unsupported topology?): {_e}')
    g2 = pref_val / (_BUBBLE_MGAMMA * N0 ** 2)
    g_spread = 0.0                         # analytic read → no q-sampling spread
    g = math.sqrt(abs(g2))
    if verbose:
        print(f'[spatial pipeline] Stage C.5 bubble: {len(bubbles)} bubble + '
              f'{len(tadpoles)} tadpole diagram(s); coupling g={g:.6f} read '
              f'analytically from M(Γ)·prefactor={pref_sum} (no compute_correction_td); '
              f'(μ,D,T)=({A0:.4f},{B0:.4f},{N0:.4f}); momentum-first ∫dℓ → Dyson → '
              f'q-FT over {n_q} q × {len(tau_grid)} τ...')

    # Phase 4c-2/4d: derivative-vertex form factors.  When the theory was
    # authored with the operator IR and carries a derivative vertex, the unfolded
    # bubbles enumerate the φ̃φ² topology and ``bubble_loop_form_factor`` reads the
    # per-vertex form factor off ``route_momenta``: Σ_R (one external + one loop
    # vertex) is ℓ-dependent, Σ_K (both external) is ℓ-independent.  Map to
    # loop_dyson's loop variable (routing ℓ₀ → q−ℓ) and inject F_R into Σ_R, F_K
    # into Σ_K.  ``None`` chain ⇒ the plain φ̃φ² bubble (validated B=0.99).
    _ffR_q = _ffK_q = None
    _chain = getattr(ft._ns, '_operator_ir_vertex_chain', None)
    if _chain:
        import sympy as _sp
        from msrjd.integration.spatial.momentum_routing import route_momenta as _rm
        rr0 = _rm(bubbles[0][0])
        q0 = rr0.q_syms[0]
        l0 = rr0.loop_syms[0]
        ld = _sp.Symbol('_ld')
        FR = FK = None
        for td, _pre in bubbles:
            F = bubble_loop_form_factor(td, _chain)
            if l0 in F.free_symbols:
                FR = _sp.expand(F.subs({l0: q0 - ld}))   # → loop_dyson ℓ
            else:
                FK = _sp.expand(F)

        def _mk_ff(expr):
            if expr is None:
                return None
            def per_q(qval):
                fn = _sp.lambdify(ld, expr.subs({q0: qval}), 'numpy')
                return lambda l: fn(l) * np.ones_like(l)
            return per_q
        _ffR_q, _ffK_q = _mk_ff(FR), _mk_ff(FK)
        if verbose:
            print(f'      derivative-vertex form factors: F_R(q,ℓ)={FR}, '
                  f'F_K(q)={FK} (chain {_chain})')

    if _chain:
        _ffR_of = (lambda qq: _ffR_q(qq)) if _ffR_q else (lambda qq: None)
        _ffK_of = (lambda qq: _ffK_q(qq)) if _ffK_q else (lambda qq: None)
    else:
        _ffR_of = ((lambda qq: formfactor(qq)) if formfactor is not None
                   else (lambda qq: None))
        _ffK_of = lambda qq: None

    # dimension: d=1 uses the analytic ∫dℓ + cosine q→x; d≥2 the direct ∫dᵈℓ
    # self-energy (loop_dyson._sigma_grids_dD, validated vs the C-stack) + the
    # radial/Hankel q→x.  Derivative-vertex form factors in d>1 are deferred.
    d = int(prop.get('spatial_dim', 1))
    if d >= 2 and _chain:
        raise NotImplementedError(
            'd>1 DERIVATIVE-vertex bubbles (form factors) are not yet wired; '
            'd>1 currently supports the plain φ̃φ² bubble.')

    # bubble δC(q,τ) on the q×τ grid (even in q), then q-FT to x.
    qg = np.linspace(0.0, q_cut, n_q) if d == 1 else np.linspace(q_cut / (4 * n_q), q_cut, n_q)
    taus = np.asarray(tau_grid, dtype=float)
    if d == 1:
        dC_q_tau = np.array([
            bubble_delta_C_q_tau(
                float(q), taus, A0, B0, N0, g, n_t=n_t,
                formfactor=_ffR_of(float(q)), formfactor_K=_ffK_of(float(q)))
            for q in qg])                               # (n_q, n_tau)
    else:
        dC_q_tau = np.array([
            bubble_delta_C_q_tau(float(q), taus, A0, B0, N0, g, n_t=n_t,
                                 spatial_dim=d, L_cut=q_cut)
            for q in qg])
    xg = np.asarray(spatial_grid, dtype=float)
    C1 = np.array(C0, dtype=np.complex128)
    if d == 1:
        for it in range(len(taus)):
            col = dC_q_tau[:, it]
            for ix, x in enumerate(xg):       # δC(x,τ)=(1/π)∫₀^∞ cos(qx)δC dq
                C1[it, ix] += np.trapz(np.cos(qg * float(x)) * col, qg) / math.pi
    else:
        from msrjd.integration.spatial.spatial_correlator import radial_inverse_ft
        for it in range(len(taus)):            # radial/Hankel q→x at d≥2
            C1[it, :] += radial_inverse_ft(qg, dC_q_tau[:, it], xg, d)

    info = dict(tree_info)
    info.update({'one_loop': True, 'bubble': True,
                 'self_energy_coupling_g': g, 'g2_q_spread': g_spread,
                 'A_tree': A0, 'mu': A0, 'D': B0, 'T': N0,
                 'n_bubble_diagrams': len(bubbles),
                 'n_tadpole_diagrams': len(tadpoles)})
    return C1, info


# ── 6. the GENERIC 1-loop correlator — sum ALL enumerated diagrams ────
def compute_spatial_correlator_generic(
        ft, model, prop, num_params, external_fields, tau_grid, spatial_grid,
        verbose=False, q_cut=30.0, n_q=160):
    """Spatial 1-loop correlator ``C(x,τ) = C₀ + δC`` the GENERIC way — the ONE
    path that replaces the bespoke bubble/tadpole routines.

    Every enumerated ``ell=1`` diagram (bubble, tadpole, …) is mapped to the
    C-stack (:func:`diagram_descriptor.diagram_to_cstack`) and evaluated by the
    SAME momentum-first evaluator (``generic_evaluator.diagram_delta_C``: Symanzik
    ``∫dᵈℓ`` → causal-chamber/Dyson time integral), weighted by the enumeration
    ``M(Γ)·prefactor`` (× the universal ``2^{−n_C}``).  The bubble-vs-tadpole
    distinction is automatic (a property of the Symanzik ``F`` / a self-loop edge),
    not a code branch, and NO diagram is dropped — so the complete 1-loop
    correction is ``Σ_Γ δC_Γ`` (``generic_evaluator.delta_C_one_loop``).

    Returns ``(C1_tau_x, info)``.  Single-field, single tree mode (the v1 scope).
    """
    from msrjd.integration.spatial.diagram_descriptor import diagram_to_cstack
    from msrjd.integration.spatial.generic_evaluator import delta_C_one_loop

    C0, tree_info = compute_spatial_correlator_via_pipeline(
        ft, model, prop, num_params, external_fields, tau_grid, spatial_grid,
        verbose=verbose, certify=True)
    modes = tree_info['modes']
    if len(modes) != 1:
        raise NotImplementedError(
            'generic spatial 1-loop v1 supports a single-mode (single-field) '
            'tree only.')
    A0, B0, N0 = modes[0]
    A0 = float(np.real(A0)); B0 = float(np.real(B0)); N0 = float(np.real(N0))

    from msrjd.diagrams.type_assignment import build_field_index_map
    ring_var_names = list(ft._ns._ring_var_names)
    _, phys_idx = build_field_index_map(ring_var_names, ft._n_tilde)
    ext_int = _legs_to_phys_idx(external_fields, phys_idx)
    nps_sr = _norm_sr(num_params)
    base_np_sr = {kk: vv for kk, vv in nps_sr.items() if str(kk) != 'Laplacian'}

    # substitute the saddle into the symbolic propagator (the operator-IR unfold
    # of a derivative vertex leaves a spurious φ*·operator bilinear that slows
    # the enumeration's Phase-J certify; at the homogeneous saddle φ*=its value).
    _sad = {kk: vv for kk, vv in nps_sr.items() if 'star' in str(kk)}
    prop_g = dict(prop)
    if _sad:
        for _key in ('K_ker', 'K_ft', 'G_ft', 'adj_ft', 'D_omega', 'D_delta'):
            _M = prop.get(_key)
            if _M is None:
                continue
            try:
                prop_g[_key] = (_M.apply_map(lambda e: SR(e).subs(_sad))
                                if hasattr(_M, 'apply_map') else SR(_M).subs(_sad))
            except Exception:
                pass
    prop = prop_g

    by_ell = build_pipeline_records(ft, model, prop, ext_int, max_ell=1,
                                    verbose=verbose)
    ell1 = by_ell.get(1, [])
    if not ell1:
        raise SpatialPropagatorError('no 1-loop diagrams were enumerated.')

    # map each enumerated diagram → (descriptor, M(Γ)·prefactor value at saddle)
    descrs = []
    for td, pre in ell1:
        try:
            pv = float(SR(pre).subs(base_np_sr))
        except (TypeError, ValueError):
            continue                                 # q-dependent prefactor (skip)
        descrs.append((diagram_to_cstack(td), pv))
    n_live = sum(1 for _, pv in descrs if abs(pv) > 1e-14)
    if verbose:
        print(f'[spatial pipeline] GENERIC 1-loop: {len(descrs)} ell=1 diagram(s), '
              f'{n_live} live at the saddle; summing δC_Γ over the q-grid '
              f'(A,B,N)=({A0:.4f},{B0:.4f},{N0:.4f})...')

    d = int(prop.get('spatial_dim', 1))
    taus = np.asarray(tau_grid, dtype=float)
    qg = (np.linspace(0.0, q_cut, n_q) if d == 1
          else np.linspace(q_cut / (4 * n_q), q_cut, n_q))
    dC_q_tau = np.array([
        delta_C_one_loop(descrs, float(q), taus, A0, B0, A0, B0, spatial_dim=d,
                         L_cut=q_cut)
        for q in qg])                                # (n_q, n_tau)

    xg = np.asarray(spatial_grid, dtype=float)
    C1 = np.array(C0, dtype=np.complex128)
    if d == 1:
        for it in range(len(taus)):
            col = dC_q_tau[:, it]
            for ix, x in enumerate(xg):              # δC(x,τ)=(1/π)∫₀^∞cos(qx)δC dq
                C1[it, ix] += np.trapz(np.cos(qg * float(x)) * col, qg) / math.pi
    else:
        from msrjd.integration.spatial.spatial_correlator import radial_inverse_ft
        for it in range(len(taus)):
            C1[it, :] += radial_inverse_ft(qg, dC_q_tau[:, it], xg, d)

    info = dict(tree_info)
    info.update({'one_loop': True, 'generic': True,
                 'A_tree': A0, 'mu': A0, 'D': B0, 'T': N0,
                 'n_ell1_diagrams': len(descrs), 'n_live_diagrams': n_live})
    return C1, info
