"""
Propagator construction (extracted from notebook cell 8).

Given an expanded ``FieldTheory`` and the model dict, build:
  K_ker  — kernel-form bilinear matrix in time domain (with δ, δ′)
  K_ft   — Fourier image of K_ker (with kernel symbols replaced via
           model['kernel_ft_image'])
  G_ft   — propagator = K_ft^{-1} (rows = physical, cols = response by
           Sage's matrix.inverse() convention; the type-assignment and
           Phase J machinery use G_ft[pi, ri] consistently)
  adj_ft — K_ft.adjugate()
  D_omega — det(K_ft)
  D_delta — coefficient matrix of δ(t) in the time-domain propagator
            (= lim_{ω→∞} G_ft(ω) entrywise)

Cached on disk under ``saved_theories/<model-tag>_taylor<N>/propagator.sobj``
so kernel restarts skip the expensive 6×6 inverse + factor() pass.
"""
from __future__ import annotations

import re

from sage.all import (
    SR, matrix, dirac_delta, diff, oo, limit as _sage_limit,
)

from msrjd.core.cache import PipelineCache
from msrjd.core.field_theory import fourier_transform


def _safe_factor(e):
    """Per-entry factor() that tolerates Maxima aborts on complex 6×6
    inverse entries (the GTaS model triggers these regularly).  factor()
    is purely cosmetic for display — the integrator does not require
    factored form."""
    try:
        return e.factor()
    except (RuntimeError, ValueError, TypeError, ArithmeticError):
        return e


def _to_kernel(c, Dt, delta_D, delta_Dp):
    """Convert an SR free-action entry (which can contain Dt and ns.g)
    into kernel form: c → c0·δ + c1·δ′  (so Fourier transforms cleanly).

    Constants without δ_D or Dt get wrapped in δ_D so FT yields the
    constant back (instead of a 2π·δ(ω) distribution).  Kernel symbols
    (e.g. ns.g) survive untransformed; their frequency image is applied
    after FT via the model's ``kernel_ft_image`` hook.
    """
    c = SR(c)
    if c.has(delta_D):
        return c
    p0   = c.subs({Dt: 0})
    rest = (c - p0).subs({Dt: delta_Dp})
    return p0 * delta_D + rest


def build_propagator(ft, model, cache_dir_root='saved_theories',
                     use_cache=True, verbose=True):
    """
    Build the symbolic propagator data dict for the given expanded
    ``FieldTheory``.

    Returns a dict with keys:
      'K_ker', 'K_ft', 'G_ft', 'adj_ft', 'D_omega', 'D_delta',
      't_var', 'omega', 'nf', 'ring_gen_names'

    Cached by ``model['name'] + taylor_order``.  Cache auto-invalidates
    if ``ft._n_tilde`` differs from the cached ``nf`` (catches model
    field-list edits without renaming).
    """
    R  = ft.ring()
    ns = ft._ns

    # ── Cache lookup ──────────────────────────────────────────────
    prop_tag = re.sub(r'[^A-Za-z0-9]+', '_', model['name']).strip('_').lower()
    cache_dir = f"{cache_dir_root}/{prop_tag}_taylor{ft.taylor_order}"
    cache = PipelineCache(cache_dir)

    if use_cache and cache.exists('propagator'):
        try:
            prop = cache.load('propagator')
            cached_nf = prop.get('nf', None)
            if cached_nf is not None and cached_nf != ft._n_tilde:
                if verbose:
                    print(f'[propagator] Cached nf={cached_nf} but model '
                          f'has n_tilde={ft._n_tilde}; rebuilding.')
            else:
                if verbose:
                    print(f'[propagator] Loaded from cache: '
                          f'{cache_dir}/propagator.sobj')
                return prop
        except Exception as e:
            if verbose:
                print(f'[propagator] Cache load failed ({e!r}); rebuilding.')

    # ── Build K_ker from the (1,1) free action ────────────────────
    S_free = ft.free_action()
    ring_gen_names = [str(g) for g in R.gens()]

    resp_names = ring_gen_names[:ft._n_tilde]
    phys_names = ring_gen_names[ft._n_tilde:]
    pos_to_row = {ring_gen_names.index(nm): i for i, nm in enumerate(resp_names)}
    pos_to_col = {ring_gen_names.index(nm): j for j, nm in enumerate(phys_names)}

    nf = len(resp_names)
    K_data = [[SR(0)] * nf for _ in range(nf)]
    for exp_vec, coeff in S_free.dict().items():
        row = col = None
        for idx in range(len(ring_gen_names)):
            if exp_vec[idx] > 0:
                if idx in pos_to_row:
                    row = pos_to_row[idx]
                if idx in pos_to_col:
                    col = pos_to_col[idx]
        if row is not None and col is not None:
            K_data[row][col] += SR(coeff)
    K_mat = matrix(SR, K_data)

    Dt       = ns.Dt
    delta_D  = ns.delta_D
    delta_Dp = ns.delta_Dp

    K_ker = matrix(
        SR,
        [[_to_kernel(K_mat[i, j], Dt, delta_D, delta_Dp)
          for j in range(nf)] for i in range(nf)],
    )

    # ── Fourier transform K_ker → K_ft ────────────────────────────
    t_var = SR.var('t')
    omega = SR.var('omega', latex_name=r'\omega')
    time_subs = {
        delta_D:  dirac_delta(t_var),
        delta_Dp: diff(dirac_delta(t_var), t_var),
    }
    K_ft_data = [[SR(0)] * nf for _ in range(nf)]
    for i in range(nf):
        for j in range(nf):
            c = K_ker[i, j]
            if not c.is_zero():
                K_ft_data[i][j] = fourier_transform(
                    SR(c).subs(time_subs), t_var, omega
                )
    K_ft = matrix(SR, K_ft_data)

    # Apply model's kernel frequency-image hook (g → ĝ(ω)).
    kft_hook = model.get('kernel_ft_image')
    if kft_hook is not None:
        kft_subs = kft_hook(ns, omega)
        K_ft = K_ft.apply_map(lambda e: SR(e).subs(kft_subs))

    # ── Propagator inverse ────────────────────────────────────────
    G_ft = K_ft.inverse().apply_map(_safe_factor)

    # ── Adjugate, det, δ-coefficient matrix ───────────────────────
    adj_ft  = K_ft.adjugate()
    D_omega = K_ft.det()

    D_delta_data = [[SR(0)] * nf for _ in range(nf)]
    for i in range(nf):
        for j in range(nf):
            entry = SR(G_ft[i, j])
            if entry.is_zero():
                continue
            try:
                lim_val = _sage_limit(entry, **{str(omega): oo})
                if not lim_val.is_zero():
                    D_delta_data[i][j] = lim_val
            except Exception:
                pass
    D_delta = matrix(SR, D_delta_data)

    prop = {
        'K_ker':         K_ker,
        'K_ft':          K_ft,
        'G_ft':          G_ft,
        'adj_ft':        adj_ft,
        'D_omega':       D_omega,
        'D_delta':       D_delta,
        't_var':         t_var,
        'omega':         omega,
        'nf':            nf,
        'ring_gen_names': ring_gen_names,
        # Filled in later by compute_poles_and_residues():
        'pole_vals':     None,
        'C_mats':        None,
    }

    if use_cache:
        try:
            cache.save('propagator', prop)
            if verbose:
                print(f'[propagator] Cached to: '
                      f'{cache_dir}/propagator.sobj')
        except Exception as e:
            if verbose:
                print(f'[propagator] Cache save failed ({e!r}).')

    return prop


def compute_poles_and_residues(prop, num_params, verbose=True):
    """
    Given a propagator dict (from build_propagator) and a num_params
    substitution dict (SR.var → float), find the retarded poles of
    G_ft and the residue matrices.  Mutates ``prop`` in place to fill
    ``prop['pole_vals']`` and ``prop['C_mats']``.

    Implementation follows the notebook cell 23 deferred pole/residue
    computation: characteristic polynomial extracted as the highest-
    degree denominator across G_ft entries, residue at each pole
    computed as ``i · adj(ω_k) / det'(ω_k)``.
    """
    from sage.all import CDF, PolynomialRing

    K_ft   = prop['K_ft']
    adj_ft = prop['adj_ft']
    G_ft   = prop['G_ft']
    nf     = prop['nf']
    omega  = prop['omega']

    K_ft_num   = K_ft.apply_map(lambda e: SR(e).subs(num_params))
    adj_ft_num = adj_ft.apply_map(lambda e: SR(e).subs(num_params))
    G_ft_num   = G_ft.apply_map(lambda e: SR(e).subs(num_params))

    PR = PolynomialRing(CDF, 'omega')
    FR = PR.fraction_field()

    char_poly = PR(1)
    for i in range(nf):
        for j in range(nf):
            entry = SR(G_ft_num[i, j])
            if entry.is_zero():
                continue
            try:
                den_p = PR(entry.denominator())
            except Exception:
                try:
                    rat = FR(entry)
                    den_p = rat.denominator()
                except Exception:
                    continue
            if den_p.degree() > char_poly.degree():
                char_poly = den_p

    # Retarded convention: Im(ω) > 0 in this codebase's FT
    # (fourier_transform uses e^{-iωt} → poles in upper half plane
    # → causal closure of the inverse-FT contour).  Deduplicate.
    roots_all = [complex(r) for r, _ in char_poly.roots(CDF)]
    pruned = []
    for r in roots_all:
        if r.imag <= 1e-9:
            continue
        if any(abs(r - q) < 1e-7 for q in pruned):
            continue
        pruned.append(r)
    pole_vals = sorted(pruned, key=lambda r: (r.imag, r.real))

    # Residue at each pole:  C_k = i · adj(ω_k) / det'(ω_k)
    # Build Sage CDF matrices so build_G_t_matrix's apply_map works.
    from sage.all import matrix as _matrix
    K_det_sr = SR(K_ft_num.det())
    K_det_prime_sr = K_det_sr.derivative(omega)
    C_mats = []
    for pk in pole_vals:
        denom = complex(K_det_prime_sr.subs({omega: pk}))
        C_entries = [[1j * complex(SR(adj_ft_num[i, j]).subs({omega: pk}))
                      / denom
                      for j in range(nf)] for i in range(nf)]
        C_mats.append(_matrix(CDF, C_entries))

    prop['pole_vals'] = pole_vals
    prop['C_mats']    = C_mats
    if verbose:
        print(f'[propagator] {len(pole_vals)} retarded poles (Im(ω) > 0):')
        for k, p in enumerate(pole_vals):
            print(f'  ω_{k+1} = {p.real:+.6f} + ({p.imag:+.6f}) i')
    return prop
