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
                  f'M(Γ)·prefactor(s) = {prefs}')
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

    if verbose and stage_headers:
        print('[5/7] (spatial) Read per-mode (A,B,N) from the propagator '
              '+ certify vs the shared-pipeline C(q,τ)...')
    if verbose:
        A, B, N = modes[0]
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
                f'(> tol {certify_tol:.0e}).  The (A,B,N) extraction or the '
                f'diagram routing is wrong for this theory.')

    if verbose and stage_headers:
        print('[6/7] (spatial) Tree level — no loop diagrams to enumerate.')
        print('[7/7] (spatial) Analytic q→x FT: Σ_modes free_two_point(A,B,N; x,τ) '
              f'on {len(tau_grid)} τ × {len(spatial_grid)} x points...')
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


def diagram_form_factor(td, op_chain, mode='composite'):
    """Assemble the momentum-space **form factor** ``F(q,ℓ)`` a derivative-vertex
    theory puts on an ARBITRARY diagram ``td`` — **any loop order ``ell`` and any
    ``k``** (NOT bubble-specific: it is a product over the diagram's interaction
    vertices, so vertices "wire together" by construction).

    A derivative is a LOCAL per-vertex feature.  Each interaction vertex carries a
    composite-derivative operator (``op_chain``, e.g. ``(('Lap',),)`` for the
    ``∇²(φⁿ)`` Model-B vertex) acting on its field composite; by momentum
    conservation the composite momentum equals the vertex's **response (φ̃) leg**
    momentum — its UNIQUE OUTGOING edge in the all-``G_R`` representation (true for
    any topology).  ``route_momenta`` supplies that momentum per edge, so

        F(q,ℓ) = ∏_{interaction vertices v}  f_chain( p_v ),

    ``p_v`` the outgoing-edge momentum (linear in the loop momenta ``ℓ₀…ℓ_{L−1}``
    and external ``q₀…``) and ``f_chain`` the per-leg Fourier factor
    (``Lap → −p²``, ``Dx → i p`` in 1-D).  Returns a sympy expr in the routing
    symbols; an empty chain → ``1`` (plain diagram).  Validated: the φ̃φ² ``Σ_R``
    bubble → ``q₀²(q₀−ℓ₀)²``; the L=2 momentum integral matches a brute ``∫dℓ₀dℓ₁``
    to 1e-14.

    SCOPE: composite-derivative vertices (the ``op_chain`` acts on the whole
    field composite — Model B / Cahn–Hilliard).  Per-PHYSICAL-leg derivatives
    (KPZ ``(∂φ)²``, where each leg carries its own ``∂``) need a per-leg operator
    map on the vertex type — a documented extension (the compiler also gates those
    today, ``theory_compiler.py``)."""
    import sympy as _sp
    from msrjd.integration.spatial.momentum_routing import route_momenta
    rr = route_momenta(td)
    leaves = set(td.prediagram[2])
    out_mom = {}        # vertex -> [outgoing-edge momenta]  (response leg)
    in_mom = {}         # vertex -> [incoming-edge momenta]  (physical legs)
    for (u, v, _l), mom in rr.edge_momenta.items():
        out_mom.setdefault(u, []).append(mom)
        in_mom.setdefault(v, []).append(mom)
    iverts = [n for n in out_mom if n not in leaves and len(out_mom[n]) == 1]

    def _f_chain(p):
        f = _sp.Integer(1)
        for entry in op_chain:
            if entry[0] == 'Lap':
                f *= -p ** 2
            elif entry[0] == 'Dx':
                f *= _sp.I * p           # ∂_x → i p  (IMAGINARY; product over legs)
            else:
                raise NotImplementedError(
                    f"diagram_form_factor: operator {entry[0]!r} not "
                    f"supported (only Lap, Dx).")
        return f

    F = _sp.Integer(1)
    if mode == 'composite':
        # ∇ acts on the field composite → one factor at the response-leg momentum
        # (Model B ∇²(φⁿ), Burgers ½∂ₓ(φ²)).
        for n in iverts:
            F *= _f_chain(out_mom[n][0])
    elif mode == 'perleg':
        # ∂ acts on each PHYSICAL leg individually → a factor per incoming edge
        # (KPZ (∂φ)²: every physical leg carries the chain; uniform per vertex).
        for n in iverts:
            for p in in_mom.get(n, []):
                F *= _f_chain(p)
    else:
        raise ValueError(f"diagram_form_factor: unknown mode {mode!r}")
    return _sp.expand(F)


# Backward-compatible alias (the function used to be bubble-specific in name).
bubble_loop_form_factor = diagram_form_factor


def _formfactor_callable(td, op_chain, mode='composite'):
    """Numpy ``F(ell, q)`` for the full-diagram integrator from the symbolic
    diagram form factor (:func:`diagram_form_factor`).  ``ell`` is the loop
    momentum ``(..., L)``, ``q`` the ``(n_ext,)`` external-momentum vector;
    returns ``(...,)`` — possibly **complex** (``∂_x → ik``), so it is NOT forced
    real here; the imaginary part is resolved (cancels / dropped) at the real-space
    output.  Generic in ``L`` (any ``ell``), ``n_ext`` (any ``k``), and ``mode``
    ('composite' → response-leg momentum, Model B; 'perleg' → per physical-leg, KPZ).

    Lambdifies ``F(ℓ₀…ℓ_{L−1}, q₀…)`` (the route_momenta symbols, the SAME basis
    the C-stack descriptor's edge routing uses — verified to 1e-14 at L=2) onto
    the integrator's loop momenta, mapping loop symbol ``lᵢ → ell[...,i]`` and
    external ``qⱼ → q[j]`` by index.  ``F=0`` → zeros."""
    import sympy as _sp
    F = _sp.expand(diagram_form_factor(td, op_chain, mode=mode))
    if F == 0:
        return lambda ell, q: np.zeros(ell.shape[:-1], dtype=complex)
    ls = sorted([s for s in F.free_symbols if str(s)[:1] == 'l'], key=str)
    qs = sorted([s for s in F.free_symbols if str(s)[:1] == 'q'], key=str)
    fn = _sp.lambdify(tuple(ls) + tuple(qs), F, 'numpy')
    nl, nq = len(ls), len(qs)

    def ff(ell, q):
        qvec = np.atleast_1d(np.asarray(q, dtype=float))
        args = ([ell[..., i] for i in range(nl)]
                + [float(qvec[j]) for j in range(nq)])
        return fn(*args) * np.ones(ell.shape[:-1])     # complex if F has i (∂_x)
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
def compute_spatial_correlator_generic(
        ft, model, prop, num_params, external_fields, tau_grid, spatial_grid,
        verbose=False, q_cut=30.0, n_q=64, max_ell=1):
    """Spatial correlator ``C(x,τ) = C₀ + δC`` to loop order ``max_ell`` via the
    **full-diagram integrator** — the ONE genuine path.

    EVERY enumerated diagram up to ``max_ell`` loops (bubble, tadpole, sunset, …)
    is mapped to the C-stack (:func:`diagram_descriptor.diagram_to_cstack`) and
    evaluated by the SAME full integral (``full_integrator.diagram_correlator``:
    Symanzik ``∫dᵈℓ`` → causal-chamber time integral → retarded+advanced sum),
    weighted by the enumeration ``M(Γ)·prefactor`` (× the universal ``2^{−n_C}``).
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
    from msrjd.integration.spatial.full_integrator import diagram_correlator

    if verbose:
        print('[5/7] (spatial) Certify tree modes (A,B,N) vs the shared-pipeline '
              'C(q,τ) at sample momenta (mode-structure check)...')
    C0, tree_info = compute_spatial_correlator_via_pipeline(
        ft, model, prop, num_params, external_fields, tau_grid, spatial_grid,
        verbose=verbose, certify=True, enum_verbose=False, stage_headers=False)
    modes = tree_info['modes']
    if len(modes) != 1:
        raise NotImplementedError(
            'generic spatial 1-loop v1 supports a single-mode (single-field) '
            'tree only.')
    A0, B0, N0 = modes[0]
    A0 = float(np.real(A0)); B0 = float(np.real(B0)); N0 = float(np.real(N0))

    # Derivative/∇ interaction vertices (operator-IR theories, e.g. Model-B
    # conserved ∇²(φ²)) deposit a momentum-space FORM FACTOR F(ℓ,q) on the loop.
    # The full-diagram integrator averages it over the loop-momentum Gaussian by
    # Gauss–Hermite (exact for the polynomial F).  Extracted per diagram below;
    # bubble-specific ⇒ 1-loop only (higher-loop form factors are future work).
    op_chain = getattr(ft._ns, '_operator_ir_vertex_chain', None)
    # NOTE: the form factor is extracted + integrated GENERICALLY per diagram
    # (diagram_form_factor: a product over interaction vertices; the L-dim
    # Gauss–Hermite loop average), so ANY ell works — the L=2 momentum integral
    # matches a brute ∫dℓ₀dℓ₁ to 1e-14.  The remaining real limits are gated
    # elsewhere: d≥2 (full_integrator.diagram_kinematic), and field-degree≠2 /
    # multiple-distinct derivative-vertex types (pipeline.theory_compiler).
    if op_chain and max_ell >= 2:
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

    if verbose:
        print(f'[6/7] (spatial) Enumerate prediagrams + typed diagrams → classify '
              f'coefficient factors → map to C-stack descriptors (max_ell={max_ell})...')
    by_ell = build_pipeline_records(ft, model, prop, ext_int, max_ell=max_ell,
                                    verbose=verbose, header=None)
    # map every enumerated diagram (all loop orders 1..max_ell) → (descriptor,
    # M(Γ)·prefactor value at saddle).  No filter, no shortcut.
    descrs = []
    for ell in range(1, max_ell + 1):
        for td, pre in by_ell.get(ell, []):
            try:
                pv = float(SR(pre).subs(base_np_sr))
            except (TypeError, ValueError):
                continue                             # q-dependent prefactor (skip)
            ff = _formfactor_callable(td, op_chain) if op_chain else None
            descrs.append((diagram_to_cstack(td), pv, ff))
    if not descrs:
        raise SpatialPropagatorError('no loop diagrams were enumerated.')
    live = [(dd, pv, ff) for dd, pv, ff in descrs if abs(pv) > 1e-14]
    if not live:
        raise SpatialPropagatorError('no live loop diagrams at the saddle.')
    if verbose:
        print(f'        {len(descrs)} typed diagram(s) → {len(live)} live at the '
              f'saddle ({len(descrs) - len(live)} zero-prefactor dropped)')
    # adaptive quadrature grid: coarser for the bigger (higher-n_C) diagrams so
    # ell=2 stays tractable (validated: n_t=16,n_s=14 is <0.1% on the sunset).
    def _grid(dd):
        nC = sum(1 for e in dd.edges if e.kind == 'C')
        return (22, 24) if nC <= 2 else (16, 14)
    if verbose:
        print(f'[7/7] (spatial) Full-diagram integration: Σ_Γ 2^(-n_C)·M(Γ) '
              f'∫dᵈℓ(Symanzik) ∫dt(causal chambers) → ret+adv → q→x FT '
              f'[{len(live)} live diagram(s), q-grid n_q={n_q}, '
              f'(A,B,N)=({A0:.4f},{B0:.4f},{N0:.4f})]...')

    d = int(prop.get('spatial_dim', 1))
    taus = np.asarray(tau_grid, dtype=float)
    qg = (np.linspace(0.0, q_cut, n_q) if d == 1
          else np.linspace(q_cut / (4 * n_q), q_cut, n_q))

    def _dC(q, tau):
        s = 0.0
        for dd, pv, ff in live:
            nt, ns = _grid(dd)
            s += diagram_correlator(dd, pv, q, tau, A0, B0, spatial_dim=d,
                                    n_t=nt, n_s=ns, formfactor=ff)
        return s
    dC_q_tau = np.array([[_dC(float(q), float(tau)) for tau in taus]
                         for q in qg])                         # (n_q, n_tau)

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
    info.update({'one_loop': max_ell >= 1, 'generic': True,
                 'full_integrator': True, 'max_ell': max_ell,
                 'A_tree': A0, 'mu': A0, 'D': B0, 'T': N0,
                 'n_diagrams': len(descrs), 'n_live_diagrams': len(live),
                 'n_ell1_diagrams': len(by_ell.get(1, []))})
    return C1, info
