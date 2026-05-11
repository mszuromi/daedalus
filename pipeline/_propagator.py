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

    # ── Propagator inverse ────────────────────────────────────────
    # Use the raw inverse — no ``factor()`` in the pipeline path.
    # Previously ``factor()`` was needed to make the D_delta limit
    # tractable (Maxima's general ``sage.limit`` is slow on raw
    # cofactor/det rational expressions); now that the D_delta loop
    # below uses ``_omega_inf_limit_fast`` (leading-coefficient
    # ratio in ω), it works just as fast on unfactored entries —
    # and we no longer depend on Singular's ``factor()``, which
    # hangs uninterruptibly on rich actions (e.g. spike-reset's
    # ``-n*v`` bilinear vertex).  Cosmetic factored form is still
    # available on demand via :func:`factor_propagator`, with its
    # own bail-out for report generation.
    G_ft = K_ft.inverse()

    # ── Adjugate, det, δ-coefficient matrix ───────────────────────
    adj_ft  = K_ft.adjugate()
    D_omega = K_ft.det()

    # ── D_delta = lim_{ω → ∞} G_ft  (the instantaneous part) ──────
    # Try the fast leading-coefficient ratio symbolically — works for
    # simple models, completes in ~ms.  If any entry can't be handled
    # symbolically (``.numerator()`` / ``.denominator()`` hang on rich
    # rational structures, e.g. spike-reset's bilinear-coupled
    # inverse), abort the symbolic pass and set ``D_delta = None``.
    # The integrator (``msrjd.integration.time_domain.propagator_td``)
    # then computes D_delta lazily AFTER numerical-parameter
    # substitution, where every G_ft entry is a clean rational
    # function in ω alone and the limit is trivial.
    #
    # The symbolic pass uses a wall-clock budget per entry — if any
    # one entry exceeds the budget, we bail out.  We CAN'T rely on a
    # cysignals alarm to interrupt mid-call (Singular's native loop
    # doesn't yield to Python signals).
    import time as _time
    D_delta = None
    try:
        D_delta_data = [[SR(0)] * nf for _ in range(nf)]
        per_entry_budget = 1.0
        for i in range(nf):
            for j in range(nf):
                entry = SR(G_ft[i, j])
                if entry.is_zero():
                    continue
                t_e = _time.perf_counter()
                lim_val = _omega_inf_limit_fast(entry, omega)
                if _time.perf_counter() - t_e > per_entry_budget:
                    # If the fast path itself was slow on a single entry,
                    # subsequent entries will likely be slow too — bail.
                    raise TimeoutError(f'_omega_inf_limit_fast on '
                                       f'entry ({i},{j}) exceeded '
                                       f'{per_entry_budget}s budget')
                if lim_val is None:
                    # Skip — integrator will compute this entry post-
                    # num_params substitution.
                    continue
                if not SR(lim_val).is_zero():
                    D_delta_data[i][j] = lim_val
        D_delta = matrix(SR, D_delta_data)
    except (TimeoutError, Exception) as exc:
        if verbose:
            print(f'      D_delta symbolic pass bailed ({type(exc).__name__}: '
                  f'{exc}) — integrator will compute lazily.')
        D_delta = None
    except BaseException as exc:
        if exc.__class__.__name__ == 'SignalError':
            if verbose:
                print(f'      D_delta symbolic pass aborted '
                      f'(SignalError) — integrator will compute lazily.')
            D_delta = None
        else:
            raise

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
