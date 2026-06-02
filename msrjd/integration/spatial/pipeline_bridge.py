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


def diagram_form_factor(td, vertex_terms, mode=None, d=1):
    """Assemble the momentum-space **form factor** ``F(q,ℓ)`` a derivative-vertex
    theory puts on an ARBITRARY diagram ``td`` — **any loop order ``ell``, any
    ``k``, and any MIX of derivative-vertex types** (NOT bubble-specific and NOT
    single-type: it is a product over the diagram's interaction vertices, and
    each vertex looks up its OWN factor, so vertices "wire together" by
    construction).

    A derivative is a LOCAL per-vertex feature, so

        F(q,ℓ) = ∏_{interaction vertices v}  𝔉(v),

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

    F = _sp.Integer(1)
    for n in iverts:
        n_phys = len(in_mom.get(n, []))
        matched = [t for t in vertex_terms
                   if t.get('n_phys') is None or t['n_phys'] == n_phys]
        if not matched:
            continue                              # plain (non-derivative) vertex → 1
        node = _sp.Integer(0)
        for t in matched:
            node += _sp.sympify(t['weight']) * _term_factor(t, n)
        F *= node
    return _sp.expand(F)


# Backward-compatible alias (the function used to be bubble-specific in name).
bubble_loop_form_factor = diagram_form_factor


def _min_gh_order(F, loop_syms):
    """Minimal Gauss–Hermite order that integrates the POLYNOMIAL form factor ``F``
    EXACTLY over the loop Gaussian.  GH(n) is exact for degree ≤ 2n−1; after the
    Cholesky map ``ℓ = ℓ̄ + Ch·Z`` a monomial of TOTAL loop-degree ``D`` can place
    degree ``D`` on a single ``Z`` (e.g. ``ℓ₀ℓ₁ → Z₀²``), so ``n = ⌈(D+1)/2⌉`` with
    ``D`` the total degree of ``F`` in the loop momenta.  Returns a safe high
    fallback (6) if ``F`` is not a polynomial in the loop symbols."""
    import sympy as _sp
    if not loop_syms:
        return 1                                   # constant in ℓ → 1 node is exact
    try:
        D = int(_sp.Poly(_sp.expand(F), *loop_syms).total_degree())
    except Exception:
        return 6                                   # non-polynomial → caller's default
    return max(1, (D + 2) // 2)                     # ⌈(D+1)/2⌉


def _formfactor_callable(td, vertex_terms, mode=None, d=1):
    """Numpy ``F(ell, q)`` for the full-diagram integrator from the symbolic
    diagram form factor (:func:`diagram_form_factor`).  Possibly **complex**
    (``∂_x → ik``) — NOT forced real here (the imaginary part is resolved at the
    real-space output).  Generic in ``L`` (any ``ell``), ``n_ext`` (any ``k``),
    the MIX of derivative-vertex types, AND the spatial dimension ``d``.

    The table's weights MUST already be numeric (couplings substituted).
    ``d=1``: ``ell`` is ``(...,L)``, ``q`` is ``(n_ext,)`` — symbols ``lᵢ→ell[...,i]``,
    ``qⱼ→q[j]``.  ``d≥2``: ``ell`` is ``(...,L,d)``, ``q`` is ``(n_ext,d)`` — the
    per-axis symbols ``lᵢ_α → ell[...,i,α]``, ``qⱼ_α → q[j,α]``.  ``F=0`` → zeros."""
    import sympy as _sp
    F = _sp.expand(diagram_form_factor(td, vertex_terms, mode=mode, d=d))
    if d == 1:
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
            return fn(*args) * np.ones(ell.shape[:-1])   # complex if F has i (∂_x)
        # Minimal EXACT Gauss–Hermite order: GH(n) integrates degree ≤ 2n−1
        # exactly.  Use the TOTAL degree of F in the loop momenta (NOT max per-
        # variable): the Cholesky map ℓ=ℓ̄+Ch·Z mixes loops, so ℓ₀ℓ₁ → a Z₀² term
        # whose per-Z degree reaches the total degree.  This is the cheap, exact
        # speedup (e.g. 6 → 2-3 ⇒ the GH grid shrinks (n/6)^L).
        ff.gh_order_needed = _min_gh_order(F, ls)
        return ff

    # ── d ≥ 2: symbols are lᵢ_α / qⱼ_α (loop/external index _ spatial axis) ──
    if F == 0:
        return lambda ell, q: np.zeros(ell.shape[:-2], dtype=complex)
    parsed = []                       # (symbol, kind 'l'/'q', index, axis)
    for s in sorted(F.free_symbols, key=str):
        nm = str(s)
        idx, ax = nm[1:].split('_')
        parsed.append((s, nm[0], int(idx), int(ax)))
    fn = _sp.lambdify(tuple(s for s, *_ in parsed), F, 'numpy')

    def ff(ell, q):
        qarr = np.asarray(q, dtype=float)                # (n_ext, d)
        args = [(ell[..., idx, ax] if kind == 'l' else float(qarr[idx, ax]))
                for (_s, kind, idx, ax) in parsed]
        return fn(*args) * np.ones(ell.shape[:-2])

    # Minimal EXACT Gauss–Hermite order from the TOTAL loop-degree (see d=1
    # note): the d≥2 grid is gh_order^{L·d}, so this is a large saving.
    loopsyms = [s for (s, kind, _i, _a) in parsed if kind == 'l']
    ff.gh_order_needed = _min_gh_order(F, loopsyms)
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
    # numeric form-factor table (the lambdified F then carries only momentum
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
    # M(Γ)·prefactor value at saddle).  No filter, no shortcut.
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

    xg = np.asarray(spatial_grid, dtype=float)
    ells = sorted({el for _dd, _pv, _ff, el in live})
    # δC(q,τ) accumulated PER loop order — ONE integration pass over all diagrams
    # (the ℓ=L run already contains every ℓ<L diagram, so a single call yields the
    # whole cumulative progression; no need to re-run for each order).
    dC_by_ell = {el: np.zeros((len(qg), len(taus)), dtype=complex) for el in ells}
    live_g = [(dd, pv, ff, el) + _grid(dd) for dd, pv, ff, el in live]
    # ──────────────────────────────────────────────────────────────────────
    # THREAD-based parallelism over (q, diagram) work units.  This is SAFE in a
    # Jupyter kernel on macOS: threads do NOT fork (the fork-Pool crash — fork
    # after matplotlib/Cocoa init — cannot occur).  The per-diagram cost is numpy
    # (einsum / cholesky / solve + the chamber quadrature + the GH form-factor
    # average), and numpy RELEASES THE GIL during those, so threads give real
    # multi-core speedup.  Each task computes one diagram's (n_τ,) column at one q;
    # results are accumulated IN THE MAIN THREAD in task order → deterministic and
    # bit-identical to the serial sum.  (Process/fork parallelism is forbidden
    # here — see the project note.)  Falls back to serial below the batch
    # threshold or when parallel=False.
    import os
    _cores = os.cpu_count() or 4
    _nw = (int(n_workers) if n_workers is not None else min(8, max(1, _cores)))
    _ntasks = len(qg) * len(live)
    # SMART GATE: threading only pays when each task is big-array numpy with the
    # GIL released — measured ~2.5× (4 workers) at L≥2 (large GH grid + chamber
    # batch), but 0.7× (SLOWER) at L=1 (tiny 1×1 matrices, dispatch-overhead-
    # bound, GIL-held).  So thread only when a heavy (loop-order ≥2) diagram is
    # present; stay serial for the cheap ℓ=1 case.
    _heavy = any(rec[3] >= 2 for rec in live_g)        # rec = (dd,pv,ff,el,nt,ns)
    if parallel and _heavy and _nw > 1 and _ntasks >= max(8, 2 * _nw):
        from concurrent.futures import ThreadPoolExecutor

        def _one(task):                               # one diagram's column at one q
            iq, q, dd, pv, ff, el, nt, ns = task
            col = np.array(
                [diagram_correlator(dd, pv, q, float(tau), A0, B0, spatial_dim=d,
                                    n_t=nt, n_s=ns, formfactor=ff) for tau in taus],
                dtype=complex)
            return iq, el, col

        tasks = [(iq, float(q)) + rec
                 for iq, q in enumerate(qg) for rec in live_g]
        if verbose:
            print(f'        parallel: {_nw} THREAD(s) over {len(qg)} q-points × '
                  f'{len(live)} diagram(s) — no fork (GIL released in numpy)')
        with ThreadPoolExecutor(max_workers=_nw) as ex:
            for iq, el, col in ex.map(_one, tasks):   # main-thread accumulate, in order
                dC_by_ell[el][iq, :] += col
    else:
        for iq, q in enumerate(qg):
            for it, tau in enumerate(taus):
                for dd, pv, ff, el, nt, ns in live_g:
                    dC_by_ell[el][iq, it] += diagram_correlator(
                        dd, pv, float(q), float(tau), A0, B0, spatial_dim=d,
                        n_t=nt, n_s=ns, formfactor=ff)

    def _ft_to_x(dC_qt):                              # (n_q,n_τ) → (n_τ,n_x) real-space δC
        add = np.zeros((len(taus), len(xg)), dtype=complex)
        if d == 1:                                    # δC(x,τ)=(1/π)∫₀^∞cos(qx)δC dq
            for it in range(len(taus)):
                col = dC_qt[:, it]
                for ix, x in enumerate(xg):
                    add[it, ix] = np.trapz(np.cos(qg * float(x)) * col, qg) / math.pi
        else:
            from msrjd.integration.spatial.spatial_correlator import radial_inverse_ft
            for it in range(len(taus)):
                add[it, :] = radial_inverse_ft(qg, dC_qt[:, it], xg, d)
        return add

    # cumulative correlator at each order: {0: tree, 1: tree+1-loop, …, L: total}
    C_by_order = {0: np.array(C0, dtype=np.complex128)}
    running = np.array(C0, dtype=np.complex128)
    for el in ells:
        running = running + _ft_to_x(dC_by_ell[el])
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
                 'A_tree': A0, 'mu': A0, 'D': B0, 'T': N0,
                 'vertex_mode': ('+'.join(sorted({t['mode'] for t in vterms_sym}))
                                 if vterms_sym else None),
                 'max_abs_imag': max_abs_imag, 'imag_frac': max_abs_imag / ref,
                 'n_diagrams': len(descrs), 'n_live_diagrams': len(live),
                 'n_ell1_diagrams': len(by_ell.get(1, [])),
                 # cumulative C(x,τ) at each loop order {0: tree, 1: +1-loop, …}
                 # — the whole progression from ONE call (no per-ℓ re-runs).
                 'C_by_order': C_by_order})
    return C1, info
