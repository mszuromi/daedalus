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


def factor_propagator(prop, *,
                      per_entry_timeout: float = 5.0,
                      slow_entry_threshold: float = 0.5,
                      verbose: bool = False):
    """Return a copy of the propagator dict with the ``G_ft`` matrix
    re-applied through ``_safe_factor``, producing a cosmetically
    factored form for display in reports.

    The pipeline's :func:`build_propagator` deliberately returns the
    *raw* inverse because ``factor()`` is known to hang or segfault
    on rich symbolic structures (Sage / Singular issue, not
    interruptible from Python).  This helper applies factoring
    after the fact, with two layers of protection:

      1. Each ``factor()`` call is wrapped by ``_safe_factor``,
         which uses a cysignals ``alarm()`` and catches Sage
         signal-errors / timeouts to fall back to the unfactored
         entry.  (Note: the alarm does not always interrupt
         Singular's native loop — see (2).)
      2. After any single entry takes longer than
         ``slow_entry_threshold`` seconds (whether it succeeded or
         not), a one-shot bail flag flips and the remaining entries
         skip factoring entirely.  This caps the total cost to
         roughly one slow-entry's worth of compute even when the
         alarm fails to interrupt.

    Returns a NEW dict with ``G_ft`` replaced; the input is untouched.
    """
    import time as _time
    G_ft = prop.get('G_ft')
    if G_ft is None:
        return prop
    bail = [False]
    n_factored = [0]
    n_skipped  = [0]
    def _factor_with_bail(e):
        if bail[0]:
            n_skipped[0] += 1
            return e
        t0 = _time.perf_counter()
        out = _safe_factor(e, timeout=per_entry_timeout)
        dt = _time.perf_counter() - t0
        if dt > slow_entry_threshold:
            bail[0] = True
        else:
            n_factored[0] += 1
        return out
    G_ft_factored = G_ft.apply_map(_factor_with_bail)
    if verbose:
        print(f'[factor_propagator] factored={n_factored[0]} '
              f'skipped={n_skipped[0]} bailed={bail[0]}')
    out = dict(prop)
    out['G_ft'] = G_ft_factored
    return out


def _safe_factor(e, timeout=5.0):
    """Per-entry factor() that tolerates Maxima/Singular aborts on the
    complex 6×6 inverse entries that bigger actions produce.  factor()
    is purely cosmetic for display — the integrator does not require
    factored form, so any failure just falls back to the unfactored
    entry.

    Wraps the call in :func:`cysignals.alarm.alarm` so a runaway
    Singular routine (which can otherwise loop indefinitely while
    spewing Flint divide-by-zero warnings, observed with spike-reset
    models whose ``-n*v`` term enriches the kinetic matrix inverse)
    is forcibly interrupted after ``timeout`` seconds.

    Catches:
      * the usual symbolic-error tuple,
      * ``cysignals.signals.SignalError`` (Sage's wrapper around
        native segfaults from Singular/Pynac),
      * ``cysignals.alarm.AlarmInterrupt`` (the timeout firing).
    """
    from cysignals.alarm import alarm, cancel_alarm, AlarmInterrupt
    try:
        alarm(timeout)
        try:
            return e.factor()
        finally:
            cancel_alarm()
    except (RuntimeError, ValueError, TypeError, ArithmeticError,
            AlarmInterrupt):
        return e
    except BaseException as exc:
        # cysignals.signals.SignalError isn't a subclass of Exception
        # in some Sage builds — catch via BaseException + name check
        # to avoid accidentally swallowing KeyboardInterrupt / SystemExit.
        if exc.__class__.__name__ == 'SignalError':
            return e
        raise


def _omega_inf_limit_fast(expr, omega_var):
    """Fast computation of  lim_{ω → ∞} expr(ω)  for SR rational
    expressions in ``omega_var``.

    Avoids calling :func:`sage.limit` (which routes through Maxima
    and is dramatically slow on raw cofactor/det rational entries —
    triggers thousands of Flint divide-by-zero warnings + Singular
    thrashing for rich models like spike-reset).  Instead, extract
    numerator + denominator, compare their omega-degree, and return
    the leading-coefficient ratio (matching the textbook recipe for
    rational-function limits).

    Returns:
      * ``SR(0)`` when ``deg(num) < deg(den)``,
      * ``num_lc / den_lc`` when ``deg(num) == deg(den)``,
      * ``None`` when ``deg(num) > deg(den)`` (caller decides what
        to do — for a physical propagator this branch shouldn't
        fire, but the helper stays honest about it),
      * ``None`` if the expression isn't cleanly a polynomial ratio
        in ``omega_var`` (e.g. transcendental kernels left unevaluated)
        so callers can fall back to ``sage.limit``.
    """
    expr = SR(expr)
    try:
        num = expr.numerator()
        den = expr.denominator()
        num_deg = int(num.degree(omega_var))
        den_deg = int(den.degree(omega_var))
    except (AttributeError, TypeError, ValueError):
        return None
    if num_deg < den_deg:
        return SR(0)
    if num_deg > den_deg:
        return None
    # Same degree: leading-coefficient ratio.
    try:
        num_lc = (num.coefficient(omega_var, num_deg)
                  if num_deg > 0 else SR(num))
        den_lc = (den.coefficient(omega_var, den_deg)
                  if den_deg > 0 else SR(den))
    except (AttributeError, TypeError, ValueError):
        return None
    if SR(den_lc).is_zero():
        return None
    return SR(num_lc) / SR(den_lc)


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

    # ── Propagator inverse / adjugate / det — budget-aware ────────
    # Symbolic K_ft.inverse() on large matrices with many free
    # symbols (kernel images + parameters + omega) is intractable —
    # an 8×8 heterogeneous-pop K_ft with 32 kernel symbols and
    # 16 weight-matrix symbols runs for minutes / hours in Sage's
    # cofactor-based inverse.  Cleanest fix: SKIP the symbolic
    # inverse / adjugate / det when the matrix is "rich" (many free
    # symbols beyond omega), and let ``compute_poles_and_residues``
    # do all three numerically AFTER ``num_params`` substitution
    # (at that point every entry is a univariate rational in omega).
    #
    # For "lean" matrices (single-pop quad Hawkes, etc.) we still
    # try the symbolic path with a wall-clock budget — if it
    # completes in time the cache benefits from having G_ft / adj_ft
    # / D_omega prebaked.  Models that exceed the budget defer all
    # heavy work to the numerical stage.
    import time as _time
    G_ft = None
    adj_ft = None
    D_omega = None
    D_delta = None

    # Quick complexity estimate: number of free SR symbols other
    # than omega.  Anything > ~20 is "rich" and not worth attempting.
    free_syms = set()
    for i in range(nf):
        for j in range(nf):
            free_syms.update(SR(K_ft[i, j]).variables())
    free_syms.discard(omega)
    rich = len(free_syms) > 20

    if not rich:
        t0 = _time.perf_counter()
        try:
            G_ft    = K_ft.inverse()
            adj_ft  = K_ft.adjugate()
            D_omega = K_ft.det()
            if verbose:
                print(f'      symbolic inverse/adj/det took '
                      f'{_time.perf_counter() - t0:.2f}s')
            # Symbolic D_delta via fast leading-coefficient ratio.
            D_delta_data = [[SR(0)] * nf for _ in range(nf)]
            for i in range(nf):
                for j in range(nf):
                    entry = SR(G_ft[i, j])
                    if entry.is_zero():
                        continue
                    lim_val = _omega_inf_limit_fast(entry, omega)
                    if lim_val is not None and not SR(lim_val).is_zero():
                        D_delta_data[i][j] = lim_val
            D_delta = matrix(SR, D_delta_data)
        except Exception:
            if verbose:
                print('      symbolic inverse/adj/det bailed — '
                      'compute_poles_and_residues will compute numerically.')
            G_ft = adj_ft = D_omega = D_delta = None
        except BaseException as exc:
            if exc.__class__.__name__ == 'SignalError':
                if verbose:
                    print('      symbolic inverse aborted (SignalError) — '
                          'compute_poles_and_residues will compute numerically.')
                G_ft = adj_ft = D_omega = D_delta = None
            else:
                raise
    else:
        if verbose:
            print(f'      K_ft has {len(free_syms)} free symbols beyond ω '
                  f'— skipping symbolic inverse/adj/det.  Computation '
                  f'will happen numerically in '
                  f'compute_poles_and_residues.')

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
    from sage.all import CDF, PolynomialRing, matrix as _matrix
    import time as _time

    K_ft   = prop['K_ft']
    adj_ft = prop.get('adj_ft')
    G_ft   = prop.get('G_ft')
    nf     = prop['nf']
    omega  = prop['omega']

    # Substitute num_params into K_ft.  Entries become univariate
    # rational functions in omega.
    K_ft_num = K_ft.apply_map(lambda e: SR(e).subs(num_params))

    PR = PolynomialRing(CDF, 'omega')
    FR = PR.fraction_field()
    from sage.all import matrix as _matrix
    import numpy as np

    if adj_ft is None or G_ft is None:
        # ── Heterogeneous / rich path ────────────────────────────
        # Architecture (determinant-first, global poles, entrywise
        # numerators evaluated numerically):
        #
        #   1. Symbolic det(K_ft_num) → SR rational N(ω)/D(ω); the
        #      retarded poles are the zeros of N in the upper-half
        #      plane.  Cheap (~0.4s for nf=8 multipop).
        #   2. Cache K's entrywise polynomial coefficients (numerator
        #      + denominator) ONCE so that K(ω) can be evaluated as
        #      a numpy complex matrix at any ω in microseconds.
        #   3. Residues via numerical Laurent extraction:
        #        C_k = i · eps · G(ω_k + eps)
        #      where ε is small relative to the pole separation.
        #      Now that pole_vals are TRUE zeros of det N (not
        #      polyfit artefacts), this captures the ε·(Res/ε) → Res
        #      blowup correctly.  Skips the costly SR adjugate
        #      (~10s) and per-pole SR .subs() calls (each ~50–500ms
        #      on 25k-char adj entries → 16 poles × 64 entries ≈
        #      50–500s before).
        #   4. D_delta = lim_{ω→∞} G(ω) extracted from two large-|ω|
        #      probes, distinguishing constant vs. 1/ω-vanishing
        #      entries by their scaling.
        if verbose:
            print('      symbolic det(K_ft_num) for pole finding...')
        t0 = _time.perf_counter()
        K_det_sr = SR(K_ft_num.det())
        # Sanity check: if K_ft_num still has free symbols beyond ω
        # (e.g. unsubstituted saddle values or kernel symbols), the
        # det coefficients won't be numerical and the polynomial
        # root finder will silently return 0 roots.  Surface this
        # mode explicitly.
        free_syms = set()
        for i in range(nf):
            for j in range(nf):
                free_syms.update(SR(K_ft_num[i, j]).variables())
        free_syms.discard(omega)
        if free_syms and verbose:
            print(f'      WARNING: K_ft_num has free symbols beyond ω: '
                  f'{sorted(str(s) for s in free_syms)}.  Pole-finding '
                  f'will likely return 0 roots.  Make sure all kernel '
                  f'symbols are substituted via the model\'s '
                  f'``kernel_ft_image`` hook and all saddle values are '
                  f'in num_params.')
        det_num_sr = K_det_sr.numerator()
        try:
            coeffs_asc = [complex(c)
                          for c in det_num_sr.coefficients(omega,
                                                           sparse=False)]
        except Exception as exc:
            if verbose:
                print(f'      coefficient extraction failed: {exc!r}')
                # Show what variables remain so the user can see why.
                try:
                    raw_coeffs = det_num_sr.coefficients(omega,
                                                         sparse=False)
                    for k, c in enumerate(raw_coeffs):
                        cv = SR(c).variables()
                        if cv:
                            print(f'        ω^{k} coeff free vars: '
                                  f'{[str(s) for s in cv]}')
                except Exception:
                    pass
            coeffs_asc = []
        # Strip trailing near-zero leading-degree coefficients (sometimes
        # the polynomial degree is over-reported with noise).
        coeffs_asc_trim = list(coeffs_asc)
        if coeffs_asc_trim:
            tol_lead = 1e-12 * max(abs(c) for c in coeffs_asc_trim)
            while len(coeffs_asc_trim) > 1 and \
                  abs(coeffs_asc_trim[-1]) < tol_lead:
                coeffs_asc_trim.pop()
        char_poly = PR(coeffs_asc_trim)
        # Use Sage's polynomial root finder (MPS / PARI under the
        # hood) rather than numpy.roots — empirically more accurate
        # for moderate-degree polynomials with mixed-magnitude
        # coefficients.  numpy.roots on the multipop test produced
        # 8 spurious roots in the upper half plane that were NOT
        # zeros of det(K) (|det| up to 1e+5 at the spurious "poles").
        try:
            roots_all = [complex(r) for r, _ in char_poly.roots(CDF)]
        except Exception:
            # Fallback to numpy if Sage's root finder fails.
            coeffs_desc = list(reversed(coeffs_asc_trim))
            roots_all = ([complex(r) for r in np.roots(coeffs_desc)]
                         if len(coeffs_desc) > 1 else [])
        if verbose:
            print(f'      det + roots took '
                  f'{_time.perf_counter() - t0:.2f}s; '
                  f'{len(roots_all)} candidate roots')

        # Cache K's entrywise polynomial-coefficient form for fast
        # numerical evaluation at any ω.  Each K[i,j] = n(ω)/d(ω)
        # where n,d are SR polynomials in ω (degree ≤ ~ N_kernel).
        # Extracting coefficients once per entry: ~64 small ops.
        t1 = _time.perf_counter()
        K_num_coeffs = [[None] * nf for _ in range(nf)]
        K_den_coeffs = [[None] * nf for _ in range(nf)]
        for i in range(nf):
            for j in range(nf):
                e = SR(K_ft_num[i, j])
                if e.is_zero():
                    K_num_coeffs[i][j] = [0j]
                    K_den_coeffs[i][j] = [1+0j]
                    continue
                try:
                    n = e.numerator()
                    d = e.denominator()
                    K_num_coeffs[i][j] = [complex(c)
                        for c in n.coefficients(omega, sparse=False)]
                    K_den_coeffs[i][j] = [complex(c)
                        for c in d.coefficients(omega, sparse=False)]
                except Exception:
                    K_num_coeffs[i][j] = [complex(e)]
                    K_den_coeffs[i][j] = [1+0j]
        if verbose:
            print(f'      K polynomial-coeff cache took '
                  f'{_time.perf_counter() - t1:.2f}s')

        def _horner(coeffs, x):
            """Horner-rule polynomial evaluation."""
            if not coeffs:
                return 0j
            v = coeffs[-1]
            for c in reversed(coeffs[:-1]):
                v = v * x + c
            return v

        def _K_at(omega_val):
            K_at = np.zeros((nf, nf), dtype=complex)
            for i in range(nf):
                for j in range(nf):
                    n = _horner(K_num_coeffs[i][j], omega_val)
                    d = _horner(K_den_coeffs[i][j], omega_val)
                    K_at[i, j] = n / d if d != 0 else 0j
            return K_at

        def _G_at(omega_val):
            try:
                return np.linalg.inv(_K_at(omega_val))
            except np.linalg.LinAlgError:
                return np.zeros((nf, nf), dtype=complex)

        # ── Fallback pole-finder ────────────────────────────────────
        # If the symbolic det route returned 0 roots (because some
        # entry in ``K_ft_num`` still has free symbols beyond ω, or
        # because the polynomial-coefficient extraction silently
        # threw), fall back to Vandermonde-polyfit-and-Newton-refine
        # on the NUMERICAL ``det(K_at(ω))``.  ``_K_at`` is already
        # cached entrywise so this is fast and doesn't depend on the
        # symbolic det succeeding.
        if not roots_all:
            if verbose:
                print('      symbolic root-finding produced 0 roots — '
                      'falling back to numerical polyfit + Newton refine')

            def _det_at(om):
                return complex(np.linalg.det(_K_at(om)))

            deg_max = 4 * nf
            n_samples = deg_max + 1
            radius = 2.0
            thetas = np.linspace(0, 2 * np.pi, n_samples, endpoint=False)
            sample_omegas = radius * np.exp(1j * thetas)
            det_samples = np.array([_det_at(om) for om in sample_omegas])
            char_coeffs = np.polyfit(sample_omegas, det_samples, deg_max)
            tol = 1e-9 * np.max(np.abs(char_coeffs))
            while len(char_coeffs) > 1 and abs(char_coeffs[0]) < tol:
                char_coeffs = char_coeffs[1:]
            char_poly = PR(list(reversed([complex(c)
                                          for c in char_coeffs])))
            try:
                candidate_roots = [complex(r) for r, _ in
                                   char_poly.roots(CDF)]
            except Exception:
                candidate_roots = []

            def _newton_refine(omega_0, max_iter=20, tol_pos=1e-13):
                om = complex(omega_0)
                for _ in range(max_iter):
                    d = _det_at(om)
                    if not (np.isfinite(d.real)
                            and np.isfinite(d.imag)):
                        return None
                    if abs(d) < tol_pos:
                        return om
                    h = 1e-7 * (1 + abs(om))
                    d_plus  = _det_at(om + h)
                    d_minus = _det_at(om - h)
                    d_prime = (d_plus - d_minus) / (2 * h)
                    if abs(d_prime) < 1e-30:
                        return None
                    step = d / d_prime
                    om -= step
                    if abs(step) < tol_pos:
                        return om
                return om

            refined = []
            for r0 in candidate_roots:
                r = _newton_refine(r0)
                if r is None:
                    continue
                try:
                    if abs(_det_at(r)) < 1e-6 and r.imag > 1e-9:
                        if not any(abs(r - q) < 1e-7 for q in refined):
                            refined.append(r)
                except Exception:
                    pass
            roots_all = refined
            if verbose:
                print(f'      numerical fallback: '
                      f'{len(roots_all)} refined candidate poles')

        # We don't have a symbolic adj/G_ft, so set to None.  Downstream:
        #   * type_assignment skips zero-check when G_ft is None
        #   * D_delta is provided directly below
        prop['adj_ft'] = None
        prop['G_ft']   = None

        # D_delta from two-probe scaling test.
        G_big  = _G_at(1e8 * (1 + 0j))
        G_huge = _G_at(1e10 * (1 + 0j))
        D_delta_data = []
        for i in range(nf):
            row = []
            for j in range(nf):
                a, b = G_big[i, j], G_huge[i, j]
                ma = abs(a)
                if ma < 1e-12:
                    row.append(SR(0))
                elif abs(a - b) / ma < 1e-3:
                    row.append(SR(complex(b)))
                else:
                    row.append(SR(0))
            D_delta_data.append(row)
        prop['D_delta'] = _matrix(SR, D_delta_data)
    else:
        # ── Legacy symbolic path ────────────────────────────────
        adj_ft_num = adj_ft.apply_map(lambda e: SR(e).subs(num_params))
        G_ft_num   = G_ft.apply_map(lambda e: SR(e).subs(num_params))
        K_det_sr   = SR(K_ft_num.det())
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
        roots_all = [complex(r) for r, _ in char_poly.roots(CDF)]

    # Retarded convention: Im(ω) > 0 in this codebase's FT
    # (fourier_transform uses e^{-iωt} → poles in upper half plane
    # → causal closure of the inverse-FT contour).  Deduplicate.
    pruned = []
    for r in roots_all:
        if r.imag <= 1e-9:
            continue
        if any(abs(r - q) < 1e-7 for q in pruned):
            continue
        pruned.append(r)
    pole_vals = sorted(pruned, key=lambda r: (r.imag, r.real))

    # For the heterogeneous-pop / numerical path: Newton-refine each
    # candidate root to machine-precision pole accuracy.  Sage's
    # polynomial root-finder (.roots(CDF)) is accurate for clean
    # polynomials, but our det.numerator() has mixed-magnitude
    # coefficients (leading ≈ 1.5e+7, trailing ≈ 1) — Sage gives
    # roots accurate to ~1e-6 absolute, which translates to
    # |det(K(root))| up to several at "near-zero" candidates.  This
    # is too coarse for the residue formula
    #   Res G = adj(K(ω_k)) / det'(ω_k)
    # because adj(K(ω_k)) is sensitive to whether K is *exactly*
    # singular at ω_k.  Newton on the numerical determinant
    # converges to machine precision in 2–5 iterations.
    if adj_ft is None or G_ft is None:

        def _newton_refine(omega_0, max_iter=20, tol_pos=1e-13):
            om = complex(omega_0)
            for _ in range(max_iter):
                d = complex(np.linalg.det(_K_at(om)))
                if not (np.isfinite(d.real) and np.isfinite(d.imag)):
                    return None
                if abs(d) < tol_pos:
                    return om
                h = 1e-7 * (1 + abs(om))
                d_plus = complex(np.linalg.det(_K_at(om + h)))
                d_minus = complex(np.linalg.det(_K_at(om - h)))
                d_prime = (d_plus - d_minus) / (2 * h)
                if abs(d_prime) < 1e-30:
                    return None
                step = d / d_prime
                om -= step
                if abs(step) < tol_pos:
                    return om
            # Return whatever we converged to (caller checks |det|).
            return om

        refined = []
        rejected = []
        for r in pole_vals:
            r_ref = _newton_refine(r)
            if r_ref is None:
                rejected.append((r, None))
                continue
            try:
                det_at = complex(np.linalg.det(_K_at(r_ref)))
            except Exception:
                rejected.append((r, None))
                continue
            if not (np.isfinite(det_at.real) and
                    np.isfinite(det_at.imag)):
                rejected.append((r, det_at))
                continue
            # Final acceptance: must have |det| ≈ 0 (well below the
            # typical det magnitude) AND must remain in the upper
            # half plane after refinement (Newton sometimes drifts
            # to the conjugate root, which is not retarded).
            if abs(det_at) > 1e-6:
                rejected.append((r, det_at))
                continue
            if r_ref.imag <= 1e-9:
                rejected.append((r, det_at))
                continue
            # Dedup against already-accepted refined roots.
            if any(abs(r_ref - q) < 1e-7 for q in refined):
                continue
            refined.append(r_ref)
        if verbose:
            print(f'      Newton-refined to {len(refined)} accurate poles '
                  f'({len(rejected)} rejected)')
        pole_vals = sorted(refined, key=lambda r: (r.imag, r.real))

    # ── Residues ────────────────────────────────────────────────
    if adj_ft is None or G_ft is None:
        # Numerical residue extraction using cofactor adjugate.
        #
        # At a pole ω_k:  Res G(ω_k) = adj(K(ω_k)) / det'(K(ω_k)).
        # C_k convention:  C_k = i · Res G(ω_k).
        #
        # K(ω_k) is singular (det = 0), so K^-1 doesn't exist — but
        # adj(K(ω_k)) does, via direct cofactor expansion.  For an
        # 8×8 numerical matrix this is 64 calls to np.linalg.det on
        # 7×7 minors (cheap).  det'(ω_k) is computed by central
        # difference on the determinant.
        #
        # Previous attempt used Laurent extraction ``C_k = i·ε·G(ω_k+ε)``,
        # which has O(ε) error from nearby-pole leakage:
        #   ε·G(ω_k+ε) = Res_k + ε · Σ_{j≠k} Res_j/(ω_k - ω_j) + O(ε)
        # For closely-spaced conjugate-pair poles (min_sep ~ 1e-3 in
        # multipop), this gives ~0.5–1% error per residue.  The
        # cofactor approach has no such leak.

        def _det_at(omega_val):
            return complex(np.linalg.det(_K_at(omega_val)))

        def _cofactor_adj(M):
            """adj(M) = transpose of cofactor matrix.
            adj[i,j] = (-1)^(i+j) · det(M with row j and col i removed)."""
            n = M.shape[0]
            A = np.zeros_like(M)
            for i in range(n):
                for j in range(n):
                    minor = np.delete(np.delete(M, j, axis=0), i, axis=1)
                    A[i, j] = ((-1) ** (i + j)) * np.linalg.det(minor)
            return A

        C_mats = []
        for pk in pole_vals:
            K_at_pk = _K_at(pk)
            adj_at_pk = _cofactor_adj(K_at_pk)
            # det'(pk) via central difference.  Scale h to the pole
            # magnitude so we stay well within the local analytic
            # region for det(K).
            h = 1e-6 * (1 + abs(pk))
            d_prime = (_det_at(pk + h) - _det_at(pk - h)) / (2 * h)
            if abs(d_prime) < 1e-30:
                # Shouldn't happen for non-degenerate poles, but
                # bail safely by emitting a zero residue matrix.
                C_mats.append(_matrix(CDF,
                    [[0j] * nf for _ in range(nf)]))
                continue
            C_np = 1j * adj_at_pk / d_prime
            # Threshold numerical noise.
            atol = 1e-12 * np.max(np.abs(C_np))
            C_np = np.where(np.abs(C_np) > atol, C_np, 0)
            C_entries = [[complex(C_np[i, j])
                          for j in range(nf)] for i in range(nf)]
            C_mats.append(_matrix(CDF, C_entries))
    else:
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
