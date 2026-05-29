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
    QQ, CDF, PolynomialRing, CyclotomicField,
)

from msrjd.core.cache import PipelineCache
from msrjd.core.field_theory import fourier_transform


# Maximum wall-time (seconds) we'll spend in the polynomial fraction-field
# path before falling back to the numpy-cofactor path.  Quad-φ completes in
# <1s; spike-reset's matrix structure (vstar coupling in row 2/3 col 0/1 +
# nstar coupling on the diagonal) drives Sage's CF[ω] matrix inverse into a
# very expensive simplification cascade that has been observed to exceed
# 10 minutes without converging.  15s is generous for the fast cases and a
# clean tripwire for the pathological ones.
#
# Set to ``None`` to disable the timeout entirely (the path will run to
# completion no matter how long it takes — useful for cross-checking
# against the no-timeout result).
_POLYNOMIAL_PATH_BUDGET_SEC = 120

try:
    import signal as _signal
    _HAS_SIGALRM = hasattr(_signal, 'SIGALRM')
except ImportError:
    _signal = None
    _HAS_SIGALRM = False


class _PolynomialPathTimeout(Exception):
    """Raised when the polynomial fraction-field inverse exceeds budget."""


def _run_with_timeout(fn, timeout_sec, *args, **kwargs):
    """Run ``fn(*args, **kwargs)`` with a SIGALRM-based time budget.

    Returns ``fn``'s result on completion within budget, or raises
    :class:`_PolynomialPathTimeout` on expiry.  No-op timeout (just calls
    ``fn`` directly) on platforms without SIGALRM, or when
    ``timeout_sec`` is ``None`` (explicit disable).
    """
    if not _HAS_SIGALRM or timeout_sec is None:
        return fn(*args, **kwargs)

    def _handler(signum, frame):
        raise _PolynomialPathTimeout(
            f'polynomial fraction-field path exceeded {timeout_sec}s budget')

    old_handler = _signal.signal(_signal.SIGALRM, _handler)
    _signal.alarm(int(timeout_sec))
    try:
        return fn(*args, **kwargs)
    finally:
        _signal.alarm(0)
        _signal.signal(_signal.SIGALRM, old_handler)


def _compute_residues_via_polynomial_fracfield(K_ft, omega, num_params, nf,
                                                verbose=False):
    """Compute (pole_vals, C_mats) via EXACT polynomial fraction-field
    arithmetic in CyclotomicField(4)[ω] = QQ[i][ω].

    Why this is the right path:

      * K_ft entries (after num_params substitution) are rational
        functions in ω with rational (or rationalizable) coefficients.
      * CyclotomicField(4) is QQ extended by i = ζ₄; polynomials over
        this field have reliable GCD, so when we invert the matrix in
        Frac(CF[ω]) Sage returns each G_ft[i,j] in **canonical
        irreducible form** P_ij(ω)/Q(ω).
      * Every entry shares the same denominator Q(ω) (the polynomial
        numerator of det(K_ft)), and roots of Q are exactly the system
        poles — no spurious cancellable factors.
      * Per-pole residue is then a clean polynomial evaluation:
        C_k[i,j] = i · P_ij(ω_k) / Q'(ω_k).

    Compared to the previous symbolic ``adj_ft.subs(omega=pk)`` path,
    this has full machine precision (matched numpy.linalg.inv to ~3e-15
    in testing) and no catastrophic cancellation near kernel poles.

    Compared to a CDF[ω] (floating-point coefficient) path, this avoids
    the unreliable numerical GCD that left degree-18 polynomials with
    ~13 spurious roots; exact CF[ω] GCD reduces to the true degree-5
    denominator directly.

    Returns ``(pole_vals, C_mats)`` on success, ``(None, None)`` on
    failure (caller falls back to numpy cofactor).
    """
    try:
        import numpy as np

        CF = CyclotomicField(4, 'I_')
        i_CF = CF.gen()
        PR_exact = PolynomialRing(CF, 'om_exact')
        F_exact  = PR_exact.fraction_field()

        K_ft_num = K_ft.apply_map(lambda e: SR(e).subs(num_params))

        def _sr_complex_to_CF(c):
            c = SR(c)
            re = QQ(c.real_part())
            im = QQ(c.imag_part())
            return CF(re) + CF(im) * i_CF

        def _sr_to_F_exact(e):
            e = SR(e)
            if e.is_zero():
                return F_exact(0)
            num = SR(e.numerator())
            den = SR(e.denominator())
            n_coeffs = [_sr_complex_to_CF(c)
                        for c in num.coefficients(omega, sparse=False)]
            d_coeffs = [_sr_complex_to_CF(c)
                        for c in den.coefficients(omega, sparse=False)]
            return F_exact(PR_exact(n_coeffs)) / F_exact(PR_exact(d_coeffs))

        K_ft_F_data = [[_sr_to_F_exact(K_ft_num[i, j])
                        for j in range(nf)] for i in range(nf)]
        K_ft_F = matrix(F_exact, K_ft_F_data)
        # Wrap the inverse in a SIGALRM watchdog — spike-reset's K_ft
        # structure has been observed to drive Sage's CF[ω] inverse into
        # a multi-minute simplification cascade.  On budget expiry we
        # let the caller fall through to the numpy-cofactor path.
        G_ft_F = _run_with_timeout(
            K_ft_F.inverse, _POLYNOMIAL_PATH_BUDGET_SEC)

        # Convert CF[om] polynomials to CDF[om] for numerical root-finding.
        PR_cdf = PolynomialRing(CDF, 'om_cdf')

        def _cf_poly_to_cdf(p_cf):
            return PR_cdf([complex(c)
                           for c in p_cf.coefficients(sparse=False)])

        # Shared denominator Q(ω) — for diagonally-dominant K_ft most
        # entries share the same Frac(CF[ω])-canonical denominator
        # (= the polynomial numerator of det(K_ft)).  Upper-triangular
        # K_ft from the Markovian-embedding preprocessor produces
        # entries with DIFFERENT canonical denominators (e.g. G[0,0]
        # carries only one pole factor, G[0,1] carries the product of
        # two), so picking the first nonzero entry's denominator
        # under-counts the system poles.
        #
        # Robust path: take ``Q_poly = LCM(per-entry denominators)``.
        # This is the universal denominator whose roots include every
        # system pole.  Per-entry residue at a pole that does NOT
        # divide the entry's own denominator evaluates to 0 (handled
        # below via the ``Q_per_entry[i][j] == Q_poly`` branch / its
        # else arm).
        Q_poly = None
        for i in range(nf):
            for j in range(nf):
                if G_ft_F[i, j] == 0:
                    continue
                d_ij = G_ft_F[i, j].denominator()
                if Q_poly is None:
                    Q_poly = d_ij
                else:
                    Q_poly = Q_poly.lcm(d_ij)
        if Q_poly is None:
            return None, None, None

        # Higher-multiplicity (Jordan-block) poles need the
        # ``(τ·exp(-pt))`` derivative term, which the single-pole
        # residue formula below can't represent.  When ``Q_poly`` is
        # not squarefree (e.g. mu = 1/tauc in the Markovianized OU
        # produces ``Q = (iω + mu)^2``), fail out of the polynomial
        # path so the caller can fall back to the numpy-cofactor
        # tier (which has the same limitation but is more robust to
        # the numerical near-degeneracy CDF root-finders introduce).
        # A v2 of the polynomial path should compute the m-th
        # derivative residue and the downstream mode-sum integrator
        # should learn how to absorb ``τ^k · exp(-pt)`` modes.
        try:
            if not Q_poly.is_squarefree():
                if verbose:
                    print(
                        '[propagator] polynomial-fracfield: Q(ω) is '
                        'non-squarefree (multi-pole) — single-pole '
                        'residue formula does not apply.  Falling '
                        'back to next tier.'
                    )
                return None, None, None
        except (AttributeError, TypeError):
            # ``is_squarefree`` may not be implemented on the field;
            # silently continue and let downstream catch issues.
            pass

        Q_cdf = _cf_poly_to_cdf(Q_poly)
        Q_prime_cdf = Q_cdf.derivative()

        roots = Q_cdf.roots(CDF)
        pole_vals = [complex(r) for r, _ in roots
                     if complex(r).imag > 1e-9]
        pole_vals.sort(key=lambda p: (p.imag, p.real))

        # Cache numerator polynomials per entry (CDF[om]) for evaluation.
        P_cdf = [[PR_cdf(0)] * nf for _ in range(nf)]
        Q_per_entry = [[None] * nf for _ in range(nf)]
        for i in range(nf):
            for j in range(nf):
                e = G_ft_F[i, j]
                if e == 0:
                    continue
                P_cdf[i][j] = _cf_poly_to_cdf(e.numerator())
                Q_per_entry[i][j] = e.denominator()

        # Cache per-entry denominator CDF polynomial and its derivative
        # so the residue loop doesn't re-convert per pole.
        Q_entry_cdf = [[None] * nf for _ in range(nf)]
        Q_entry_prime_cdf = [[None] * nf for _ in range(nf)]
        for i in range(nf):
            for j in range(nf):
                if Q_per_entry[i][j] is None:
                    continue
                if Q_per_entry[i][j] == Q_poly:
                    Q_entry_cdf[i][j] = Q_cdf
                    Q_entry_prime_cdf[i][j] = Q_prime_cdf
                else:
                    q_cdf = _cf_poly_to_cdf(Q_per_entry[i][j])
                    Q_entry_cdf[i][j] = q_cdf
                    Q_entry_prime_cdf[i][j] = q_cdf.derivative()

        C_mats = []
        for pk in pole_vals:
            C_entries = [[0j] * nf for _ in range(nf)]
            for i in range(nf):
                for j in range(nf):
                    if Q_per_entry[i][j] is None:
                        continue
                    # Skip entries where pk isn't a pole of THIS
                    # entry's canonical denominator — the residue
                    # vanishes there.  We detect this via the
                    # magnitude of the entry's Q evaluated at pk.
                    qij_at = complex(Q_entry_cdf[i][j](pk))
                    if abs(qij_at) > 1e-9:
                        continue
                    qijp = complex(Q_entry_prime_cdf[i][j](pk))
                    if abs(qijp) < 1e-30:
                        # Higher-multiplicity pole (Q_per_entry has pk
                        # with order ≥ 2).  The single-pole residue
                        # formula doesn't apply; leave 0 and rely on
                        # the polynomial fall-back to flag the issue.
                        continue
                    P_at = complex(P_cdf[i][j](pk))
                    C_entries[i][j] = 1j * P_at / qijp
            C_np = np.array(C_entries)
            mx = float(np.max(np.abs(C_np))) if C_np.size else 0.0
            if mx > 0:
                atol = 1e-12 * mx
                C_np = np.where(np.abs(C_np) > atol, C_np, 0)
            C_mats.append(matrix(CDF, C_np.tolist()))

        # D_delta = lim_{ω→∞} G_ft(ω), computed entrywise from leading
        # coefficients of P_ij / Q.  For a proper rational entry:
        #   deg(P_ij) <  deg(Q)   →  D_delta[i,j] = 0
        #   deg(P_ij) == deg(Q)  →  D_delta[i,j] = lc(P_ij) / lc(Q)
        # Stored back into prop so downstream consumers always see a
        # value (the symbolic build_propagator path may have left it None
        # if its inverse exceeded budget).
        Q_lead = complex(Q_cdf.leading_coefficient())
        Q_deg  = Q_cdf.degree()
        D_delta_data = [[0j] * nf for _ in range(nf)]
        for i in range(nf):
            for j in range(nf):
                if Q_per_entry[i][j] is None:
                    continue
                P_pij = P_cdf[i][j]
                if P_pij.degree() == Q_deg:
                    D_delta_data[i][j] = (
                        complex(P_pij.leading_coefficient()) / Q_lead)
                # else: D_delta[i,j] stays 0 (strictly proper)
        D_delta = matrix(SR, D_delta_data)

        if verbose:
            print(f'[propagator] polynomial-fracfield: '
                  f'Q(ω) degree {Q_cdf.degree()}, '
                  f'{len(pole_vals)} retarded pole(s) found.')
        return pole_vals, C_mats, D_delta

    except _PolynomialPathTimeout:
        if verbose:
            print(f'[propagator] polynomial-fracfield exceeded '
                  f'{_POLYNOMIAL_PATH_BUDGET_SEC}s budget — '
                  f'falling back to numpy cofactor.')
        return None, None, None
    except Exception as exc:
        if verbose:
            print(f'[propagator] polynomial-fracfield failed '
                  f'({type(exc).__name__}: {exc!s:.80}); falling back.')
        return None, None, None


def _trunc_str(s, maxlen=400):
    s = str(s)
    return s if len(s) <= maxlen else s[:maxlen - 3] + '...'


def _print_matrix(label, M, resp_names, phys_names):
    """Print non-zero entries of an SR matrix with row/col labels."""
    print()
    print(f'      ── {label} (shape {M.nrows()} × {M.ncols()}) ──')
    print(f'        rows (response): {resp_names}')
    print(f'        cols (physical): {phys_names}')
    for i in range(M.nrows()):
        for j in range(M.ncols()):
            s = str(M[i, j])
            if s == '0':
                continue
            print(f'        [{i},{j}] = {_trunc_str(s, 400)}')


def _print_propagator_symbolic_stages(prop, resp_names, phys_names):
    """Print D_omega, G_ft, adj_ft, D_delta (whichever are non-None)."""
    if prop.get('D_omega') is not None:
        print()
        print('      ── D(ω) = det(K_ft) ──')
        print(f'        {_trunc_str(prop["D_omega"], 600)}')
    if prop.get('G_ft') is not None:
        _print_matrix('G_ft = K_ft⁻¹', prop['G_ft'], resp_names, phys_names)
    if prop.get('adj_ft') is not None:
        _print_matrix('adj_ft = G_ft · D(ω)', prop['adj_ft'],
                      resp_names, phys_names)
    if prop.get('D_delta') is not None:
        _print_matrix('D_delta = lim_{ω→∞} G_ft (instantaneous)',
                      prop['D_delta'], resp_names, phys_names)


def _print_propagator_stages(prop):
    """Print everything available in a propagator dict (cache-loaded path).
    resp/phys names are derived from ring_gen_names + nf.
    """
    ring_gen_names = prop.get('ring_gen_names') or []
    nf = prop.get('nf') or len(ring_gen_names) // 2
    resp_names = ring_gen_names[:nf]
    phys_names = ring_gen_names[nf:]
    if prop.get('K_ker') is not None:
        _print_matrix('K_ker', prop['K_ker'], resp_names, phys_names)
    if prop.get('K_ft') is not None:
        _print_matrix('K_ft', prop['K_ft'], resp_names, phys_names)
    _print_propagator_symbolic_stages(prop, resp_names, phys_names)


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
                     use_cache=True, verbose=True, force=False):
    """
    Build the symbolic propagator data dict for the given expanded
    ``FieldTheory``.

    Returns a dict with keys:
      'K_ker', 'K_ft', 'G_ft', 'adj_ft', 'D_omega', 'D_delta',
      't_var', 'omega', 'nf', 'ring_gen_names'

    Cached under ``saved_theories/<theory_slug>/propagator.sobj``.
    The propagator is taylor-order-independent: it depends only on
    the bilinear (1,1) sector of the action, which is fully captured
    at ``taylor_order >= 2``.  A precompute call at order 2 fills this
    cache once; subsequent runs at any higher order skip the build.

    Cache auto-invalidates if ``ft._n_tilde`` differs from the cached
    ``nf`` (catches model field-list edits without renaming).
    """
    R  = ft.ring()
    ns = ft._ns

    # ── Cache lookup ──────────────────────────────────────────────
    prop_tag = re.sub(r'[^A-Za-z0-9]+', '_', model['name']).strip('_').lower()
    cache_dir = f"{cache_dir_root}/{prop_tag}"
    cache = PipelineCache(cache_dir)

    if use_cache and not force and cache.exists('propagator'):
        try:
            prop = cache.load('propagator')
            cached_nf = prop.get('nf', None)
            # Stale-cache guard: a spatial model whose cached propagator
            # predates the spatial block (no ``G_tx_sym``) must rebuild
            # — otherwise the heat-kernel propagator is silently absent.
            stale_spatial = (bool(model.get('spatial'))
                             and prop.get('G_tx_sym') is None)
            if cached_nf is not None and cached_nf != ft._n_tilde:
                if verbose:
                    print(f'[propagator] Cached nf={cached_nf} but model '
                          f'has n_tilde={ft._n_tilde}; rebuilding.')
            elif stale_spatial:
                if verbose:
                    print('[propagator] Cached propagator predates the '
                          'spatial block (no G_tx_sym); rebuilding.')
            else:
                if verbose:
                    print(f'[propagator] Loaded from cache: '
                          f'{cache_dir}/propagator.sobj')
                    _print_propagator_stages(prop)
                # Rebuild the (unpicklable) spatial G_tx callables from
                # the cached symbolic block.
                if prop.get('G_tx_sym') is not None:
                    from msrjd.integration.spatial.heat_kernel import (
                        make_g_tx_callables,
                    )
                    prop['G_tx'] = make_g_tx_callables(prop)
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

    if verbose:
        _print_matrix('K_mat', K_mat, resp_names, phys_names)

    Dt       = ns.Dt
    delta_D  = ns.delta_D
    delta_Dp = ns.delta_Dp

    K_ker = matrix(
        SR,
        [[_to_kernel(K_mat[i, j], Dt, delta_D, delta_Dp)
          for j in range(nf)] for i in range(nf)],
    )

    if verbose:
        _print_matrix('K_ker', K_ker, resp_names, phys_names)

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

    if verbose:
        _print_matrix('K_ft', K_ft, resp_names, phys_names)

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

    # Complexity gate.  Sage's ``K_ft.inverse()`` is cofactor-based —
    # O(n!) in the matrix dimension ``nf``.  For nf=8 that's 40 320
    # minor determinants × the cost-per-minor of simplifying the
    # rational SR entries; in practice it runs for HOURS regardless
    # of how lean the symbol set is.  Spike-reset (single-pop with
    # ``population_size=2`` so nf=8 and only ~18 free symbols) was the
    # case that exposed the symbol-count gate as the wrong metric.
    # Switch to a matrix-size gate; keep the symbol-count threshold
    # as a secondary trigger for the rare lean-but-symbol-rich case.
    free_syms = set()
    for i in range(nf):
        for j in range(nf):
            free_syms.update(SR(K_ft[i, j]).variables())
    free_syms.discard(omega)
    rich = nf >= 6 or len(free_syms) > 20

    if not rich:
        # Wall-time budget for the entire symbolic SR matrix-inverse +
        # adj + det + D_delta block.  Quad-φ finishes in ~0.3s; spike-
        # reset (with both vstar and nstar coupling in K_mat) gets
        # stuck on the nested-fraction G_ft entries — the inverse
        # itself is fast (<1s) but ``_omega_inf_limit_fast`` calling
        # ``.numerator()`` / ``.denominator()`` on each entry triggers
        # an expensive fraction-combination cascade.  Cap the whole
        # block at 60s — if it busts, fall through to the rich/numerical
        # path (compute_poles_and_residues uses the polynomial fracfield
        # + numpy cofactor tiers, both of which give correct residues).
        #
        # Set to ``None`` to disable the timeout entirely (the symbolic
        # block runs to completion no matter how long it takes — useful
        # for cross-checking poles between the symbolic and polynomial
        # fracfield paths).
        _SYMBOLIC_INVERSE_BUDGET_SEC = 120

        def _do_symbolic_inverse_block():
            t0 = _time.perf_counter()
            G    = K_ft.inverse()
            D_om = K_ft.det()
            # Adjugate via the identity ``adj(K) = K^(-1) · det(K)``.
            # ``K_ft.adjugate()`` would re-expand all n² cofactor
            # sub-determinants WITHOUT reusing the inverse's work —
            # measured at 734 s on the 4×4 spike-reset K_ft vs ~11 s
            # for the inverse itself.
            adj  = G * D_om
            if verbose:
                print(f'      symbolic inverse/adj/det took '
                      f'{_time.perf_counter() - t0:.2f}s')
            # Symbolic D_delta via fast leading-coefficient ratio.
            Dd_data = [[SR(0)] * nf for _ in range(nf)]
            for i in range(nf):
                for j in range(nf):
                    entry = SR(G[i, j])
                    if entry.is_zero():
                        continue
                    lim_val = _omega_inf_limit_fast(entry, omega)
                    if lim_val is not None and not SR(lim_val).is_zero():
                        Dd_data[i][j] = lim_val
            return G, D_om, adj, matrix(SR, Dd_data)

        try:
            G_ft, D_omega, adj_ft, D_delta = _run_with_timeout(
                _do_symbolic_inverse_block, _SYMBOLIC_INVERSE_BUDGET_SEC)
        except _PolynomialPathTimeout:
            if verbose:
                print(f'      symbolic inverse/D_delta block exceeded '
                      f'{_SYMBOLIC_INVERSE_BUDGET_SEC}s budget — '
                      f'compute_poles_and_residues will compute residues '
                      f'numerically.')
            G_ft = adj_ft = D_omega = D_delta = None
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
            reason = (
                f'nf={nf} ≥ 6 (cofactor inverse is O(nf!))'
                if nf >= 6
                else f'{len(free_syms)} free symbols > 20 threshold'
            )
            print(f'      K_ft skipped symbolic inverse/adj/det '
                  f'({reason}).  Computation deferred to '
                  f'compute_poles_and_residues (numerical, LU-based).')

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

    # ── Spatial stage (v1): build the (t, x) heat-kernel propagator ─
    # When the model declares a spatial field, substitute the inert
    # ``Laplacian`` operator with ``-k²`` and invert both transforms
    # to real space.  Tier 1 (diagonal Allen-Cahn-like) is closed
    # form; the Tier-2 numerical-inverse-FT fallback is deferred to
    # v2, so a non-Tier-1 spatial model logs the reason and leaves the
    # propagator time-domain-only (precompute still succeeds — the
    # symbolic G_ft carries Laplacian for downstream use).
    if model.get('spatial'):
        try:
            from msrjd.integration.spatial.heat_kernel import (
                build_spatial_propagator,
            )
            spatial_block = build_spatial_propagator(
                K_ft, omega, ns, model, resp_names, phys_names,
                verbose=verbose,
            )
            prop.update(spatial_block)
        except Exception as e:
            if verbose:
                print(f'      spatial propagator (heat kernel) not built '
                      f'(Tier-1 closed form inapplicable): '
                      f'{type(e).__name__}: {e}')
            prop['spatial_dim'] = int((model.get('spatial') or {}).get('dim', 0))
            prop['G_tx'] = None
            # Record that the Tier-1 spatial block is unavailable so the
            # downstream spatial correlator can raise a CLEAR error
            # (rather than a cryptic KeyError on 'G_tx_sym').  This fires
            # for e.g. off-diagonal multi-field coupling (a Tier-2 / v2
            # case — see SpatialPropagatorError).
            prop['G_tx_sym'] = None
            prop['spatial_tier1_error'] = f'{type(e).__name__}: {e}'

    if verbose:
        _print_propagator_symbolic_stages(prop, resp_names, phys_names)

    if use_cache:
        try:
            # ``prop`` is picklable here — the spatial block stores only
            # SR exprs + plain data; the G_tx callables are attached
            # AFTER the save (closures don't pickle).
            cache.save('propagator', prop)
            if verbose:
                print(f'[propagator] Cached to: '
                      f'{cache_dir}/propagator.sobj')
        except Exception as e:
            if verbose:
                print(f'[propagator] Cache save failed ({e!r}).')

    # Attach the runtime spatial G_tx callables (post-cache-save so the
    # cached artefact stays picklable).
    if prop.get('G_tx_sym') is not None:
        from msrjd.integration.spatial.heat_kernel import (
            make_g_tx_callables,
        )
        prop['G_tx'] = make_g_tx_callables(prop)

    return prop


def compute_poles_and_residues(prop, num_params, verbose=True):
    """
    Given a propagator dict (from build_propagator) and a num_params
    substitution dict (SR.var → float), find the retarded poles of
    G_ft and the residue matrices.  Mutates ``prop`` in place to fill
    ``prop['pole_vals']`` and ``prop['C_mats']``.

    Three-tier strategy:

      1. **Polynomial fraction-field (CyclotomicField(4)[ω])** — the
         primary path.  Exact QQ[i] arithmetic, canonical irreducible
         P_ij(ω)/Q(ω) form per entry, machine-precision residues.
         Works for any model whose num_params can be QQ-rationalized
         (most cases — params are typically floats convertible to QQ).
      2. **Numpy cofactor on numerical K(pk)** — fallback when the
         polynomial path fails (e.g. non-rationalizable parameters).
         Same numerical cofactor algorithm the rich path already uses.
      3. **Legacy symbolic adj_ft.subs** — final fallback, kept for
         historical reasons.  Has the catastrophic-cancellation
         precision bug fixed by my earlier patch (now uses numpy
         cofactor in the lean path too), so tier-2 and tier-3 are
         numerically equivalent in current code.

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

    # ── Tier 1: exact polynomial fraction-field path ──────────────
    pole_vals_tier1, C_mats_tier1, D_delta_tier1 = (
        _compute_residues_via_polynomial_fracfield(
            K_ft, omega, num_params, nf, verbose=verbose))
    if pole_vals_tier1 is not None:
        prop['pole_vals'] = pole_vals_tier1
        prop['C_mats']    = C_mats_tier1
        # Fill D_delta if build_propagator left it None (its symbolic
        # inverse may have exceeded its own budget for complex models).
        if prop.get('D_delta') is None and D_delta_tier1 is not None:
            prop['D_delta'] = D_delta_tier1
        if verbose:
            print(f'[propagator] {len(pole_vals_tier1)} retarded poles '
                  f'(Im(ω) > 0) — exact polynomial path:')
            for k, p in enumerate(pole_vals_tier1):
                print(f'  ω_{k+1} = {p.real:+.6f} + ({p.imag:+.6f}) i')
            print()
            print(f'      ── C_mats (residue matrix at each pole) ──')
            ring_gen_names = prop.get('ring_gen_names') or []
            resp_names = ring_gen_names[:nf]
            phys_names = ring_gen_names[nf:]
            print(f'        rows (response): {resp_names}')
            print(f'        cols (physical): {phys_names}')
            for k_p, C in enumerate(C_mats_tier1):
                print(f'      C_mats[{k_p}]  (residue at ω_{k_p+1} = '
                      f'{pole_vals_tier1[k_p].real:+.4f}'
                      f'{pole_vals_tier1[k_p].imag:+.4f}i):')
                try:
                    for ii in range(C.nrows()):
                        for jj in range(C.ncols()):
                            v = complex(C[ii, jj])
                            if abs(v) > 1e-15:
                                print(f'        [{ii},{jj}] = '
                                      f'{v.real:+.6e}{v.imag:+.6e}j')
                except Exception as e:
                    print(f'        (error displaying: {e})')
        return prop

    if verbose:
        print('[propagator] polynomial fraction-field unavailable; '
              'falling back to numpy cofactor.')

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
        # Determinant-first: poles are zeros of num(det(K(ω))).  We do
        # NOT use the largest entry-wise denominator of G_ft here —
        # that polynomial can be a strict divisor of num(det(K)) when
        # mode-i,j decouples from adj(K)[i,j] for all (i,j) (the
        # entry-wise simplifier cancels those factors out), or it can
        # be polluted by lingering filter-denominator artefacts.
        # Either way the resulting roots disagree with the residue
        # formula (which uses K_det_sr.derivative), giving missing or
        # bogus poles and a magnitude error in C(τ).
        adj_ft_num = adj_ft.apply_map(lambda e: SR(e).subs(num_params))
        G_ft_num   = G_ft.apply_map(lambda e: SR(e).subs(num_params))
        K_det_sr   = SR(K_ft_num.det())
        det_num_sr = SR(K_det_sr.numerator())
        det_den_sr = SR(K_det_sr.denominator())
        try:
            coeffs_asc = [complex(c) for c in
                          det_num_sr.coefficients(omega, sparse=False)]
        except Exception as exc:
            if verbose:
                print(f'      det-num coeff extraction failed: {exc!r}; '
                      f'falling back to entrywise-denominator method')
            coeffs_asc = []
        # Trim trailing near-zero leading coefficients (polynomial-degree
        # noise from symbolic simplification).
        if coeffs_asc:
            tol_lead = 1e-12 * max(abs(c) for c in coeffs_asc)
            while len(coeffs_asc) > 1 and \
                  abs(coeffs_asc[-1]) < tol_lead:
                coeffs_asc.pop()
        if coeffs_asc and len(coeffs_asc) > 1:
            char_poly = PR(coeffs_asc)
        else:
            # Fallback to old entrywise-denominator approach if the
            # determinant coefficient extraction failed (e.g.
            # K_det_sr still has free symbols beyond ω).
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
        if verbose:
            try:
                deg_dnum = PR(det_num_sr).degree()
            except Exception:
                deg_dnum = -1
            print(f'      char_poly degree = {char_poly.degree()}, '
                  f'num(det K) degree = {deg_dnum}')
        roots_all = [complex(r) for r, _ in char_poly.roots(CDF)]

        # Cache K's entrywise polynomial-coefficient form for fast
        # numerical evaluation at any ω — needed for the spurious-root
        # filter below.  Same construction as the rich path.
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

        def _horner(coeffs, x):
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

    # Retarded convention: Im(ω) > 0 in this codebase's FT
    # (fourier_transform uses e^{-iωt} → poles in upper half plane
    # → causal closure of the inverse-FT contour).  Pre-dedup at the
    # polynomial-root-finder accuracy scale (~1e-5).  Newton refinement
    # below tightens each kept candidate to machine precision and
    # provides the authoritative dedup pass.
    pruned = []
    for r in roots_all:
        if r.imag <= 1e-9:
            continue
        if any(abs(r - q) < 1e-5 for q in pruned):
            continue
        pruned.append(r)
    pole_vals = sorted(pruned, key=lambda r: (r.imag, r.real))

    # Newton-refine each candidate on the numerical determinant.
    # Sage's char-poly root finder is accurate to ~1e-5 absolute for
    # this problem class.  A true zero of det(K) has |det| at machine
    # precision after Newton; a "spurious" root either diverges or
    # converges to a non-zero residual.  This separates the two
    # cleanly without needing a parameter-dependent threshold.
    try:
        # Reference scale for the |det| acceptance check.
        max_pole_imag = max((p.imag for p in pole_vals), default=1.0)
        probe_omegas = [10.0 * max(1.0, max_pole_imag) * (1 + 0.1j),
                        20.0 * max(1.0, max_pole_imag) * (1 + 0.1j),
                        -15.0 * max(1.0, max_pole_imag) * (1 + 0.1j)]
        probe_dets = []
        for om in probe_omegas:
            try:
                probe_dets.append(abs(complex(
                    np.linalg.det(_K_at(om)))))
            except Exception:
                pass
        det_scale = max(probe_dets) if probe_dets else 1.0

        def _newton_refine(omega_0, max_iter=25, tol_pos=1e-13):
            om = complex(omega_0)
            for _ in range(max_iter):
                try:
                    d = complex(np.linalg.det(_K_at(om)))
                except Exception:
                    return None
                if not (np.isfinite(d.real) and np.isfinite(d.imag)):
                    return None
                if abs(d) < tol_pos:
                    return om
                h = 1e-7 * (1 + abs(om))
                try:
                    d_plus  = complex(np.linalg.det(_K_at(om + h)))
                    d_minus = complex(np.linalg.det(_K_at(om - h)))
                except Exception:
                    return None
                d_prime = (d_plus - d_minus) / (2 * h)
                if abs(d_prime) < 1e-30:
                    return None
                step = d / d_prime
                om -= step
                if abs(step) < tol_pos:
                    return om
            return om

        # Acceptance: |det(K(ω_refined))| / det_scale < 1e-9.  Newton
        # on a true pole converges to |det| at machine precision
        # (relative ratio 1e-15 to 1e-19); roots that converge to
        # |det|/scale ~ 1e-7 or worse are spurious.  Clean 6+ order
        # of magnitude gap between true and spurious.
        tol_rel = 1e-9
        refined = []
        rejected = []
        for pk in pole_vals:
            r_ref = _newton_refine(pk)
            if r_ref is None:
                rejected.append((pk, None, 'no-converge'))
                continue
            try:
                det_at = abs(complex(np.linalg.det(_K_at(r_ref))))
            except Exception:
                rejected.append((pk, None, 'det-eval-fail'))
                continue
            if not np.isfinite(det_at):
                rejected.append((pk, det_at, 'det-not-finite'))
                continue
            # Newton can drift to the conjugate root in the lower half
            # plane — reject those (not retarded).
            if r_ref.imag <= 1e-9:
                rejected.append((pk, det_at, 'drifted-to-LHP'))
                continue
            if det_at > tol_rel * det_scale:
                rejected.append((pk, det_at, '|det| too large'))
                continue
            # Dedup after refinement: two candidates that Newton
            # converges to the same root within ~1e-7 are duplicates
            # of one physical pole.
            if any(abs(r_ref - q) < 1e-7 for q in refined):
                continue
            refined.append(r_ref)
        if verbose:
            if rejected:
                print(f'      Newton-refined: {len(refined)} kept, '
                      f'{len(rejected)} rejected '
                      f'(scale = {det_scale:.2e}):')
                for pk, dv, reason in rejected:
                    pk_str = f'{pk.real:+.6f}{pk.imag:+.6f}i'
                    dv_str = (f'{dv:.2e}' if dv is not None
                              else 'n/a')
                    print(f'        ω = {pk_str}: |det| = {dv_str}'
                          f'  [{reason}]')
            else:
                print(f'      Newton-refined: {len(refined)} kept '
                      f'(scale = {det_scale:.2e})')
        pole_vals = sorted(refined, key=lambda r: (r.imag, r.real))
    except Exception as exc:
        if verbose:
            print(f'      Newton refinement skipped ({exc!r})')

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
        # ── Lean symbolic path — numerical residue evaluation ──────
        # Even though we have a symbolic adj_ft / G_ft from K_ft.inverse(),
        # we DO NOT use them for residue computation.  Sage stores
        # adj_ft entries as sum-of-rationals — each summand has a
        # denominator like ``1/(1+iω·taug_ij)`` that diverges by 10³–10⁴
        # when pk lies near a kernel pole ω = i/taug_ij.  The terms
        # then catastrophically cancel against each other at pk,
        # losing 3–5 digits of floating-point precision per residue
        # entry (measured: 11–42% relative error on quad's residues
        # depending on which pole).
        #
        # Numpy LU on the numerical K_ft(pk) matrix builds the
        # 3×3 minors directly and computes their determinants without
        # any sum-of-rationals cancellation.  Same algorithm as the
        # ``rich`` path above.
        K_ft_num_coeffs_num = [[None] * nf for _ in range(nf)]
        K_ft_num_coeffs_den = [[None] * nf for _ in range(nf)]
        for i in range(nf):
            for j in range(nf):
                e = SR(K_ft_num[i, j])
                if e.is_zero():
                    K_ft_num_coeffs_num[i][j] = [0j]
                    K_ft_num_coeffs_den[i][j] = [1+0j]
                    continue
                try:
                    n_p = e.numerator()
                    d_p = e.denominator()
                    K_ft_num_coeffs_num[i][j] = [complex(c)
                        for c in n_p.coefficients(omega, sparse=False)]
                    K_ft_num_coeffs_den[i][j] = [complex(c)
                        for c in d_p.coefficients(omega, sparse=False)]
                except Exception:
                    K_ft_num_coeffs_num[i][j] = [complex(e)]
                    K_ft_num_coeffs_den[i][j] = [1+0j]

        def _horner(coeffs, x):
            v = 0j
            for c in reversed(coeffs):
                v = v * x + c
            return v

        def _K_at_lean(omega_val):
            K_at = np.zeros((nf, nf), dtype=complex)
            for i in range(nf):
                for j in range(nf):
                    n = _horner(K_ft_num_coeffs_num[i][j], omega_val)
                    d = _horner(K_ft_num_coeffs_den[i][j], omega_val)
                    K_at[i, j] = n / d if d != 0 else 0j
            return K_at

        def _det_at_lean(omega_val):
            return complex(np.linalg.det(_K_at_lean(omega_val)))

        def _cofactor_adj_lean(M):
            n = M.shape[0]
            A = np.zeros_like(M)
            for i in range(n):
                for j in range(n):
                    minor = np.delete(np.delete(M, j, axis=0), i, axis=1)
                    A[i, j] = ((-1) ** (i + j)) * np.linalg.det(minor)
            return A

        C_mats = []
        for pk in pole_vals:
            K_at_pk = _K_at_lean(pk)
            adj_at_pk = _cofactor_adj_lean(K_at_pk)
            # det'(pk) via central difference, scaled to pk magnitude
            h = 1e-6 * (1 + abs(pk))
            d_prime = (_det_at_lean(pk + h) - _det_at_lean(pk - h)) / (2 * h)
            if abs(d_prime) < 1e-30:
                C_mats.append(_matrix(CDF,
                    [[0j] * nf for _ in range(nf)]))
                continue
            C_np = 1j * adj_at_pk / d_prime
            atol = 1e-12 * np.max(np.abs(C_np))
            C_np = np.where(np.abs(C_np) > atol, C_np, 0)
            C_entries = [[complex(C_np[i, j])
                          for j in range(nf)] for i in range(nf)]
            C_mats.append(_matrix(CDF, C_entries))

    prop['pole_vals'] = pole_vals
    prop['C_mats']    = C_mats
    if verbose:
        print(f'[propagator] {len(pole_vals)} retarded poles (Im(ω) > 0):')
        for k, p in enumerate(pole_vals):
            print(f'  ω_{k+1} = {p.real:+.6f} + ({p.imag:+.6f}) i')
        print()
        print(f'      ── C_mats (residue matrix at each pole) ──')
        ring_gen_names = prop.get('ring_gen_names') or []
        resp_names = ring_gen_names[:nf]
        phys_names = ring_gen_names[nf:]
        print(f'        rows (response): {resp_names}')
        print(f'        cols (physical): {phys_names}')
        for k, C in enumerate(C_mats):
            print(f'      C_mats[{k}]  (residue at ω_{k+1} = '
                  f'{pole_vals[k].real:+.4f}{pole_vals[k].imag:+.4f}i):')
            try:
                nr, nc = C.nrows(), C.ncols()
                for i in range(nr):
                    for j in range(nc):
                        v = complex(C[i, j])
                        if abs(v) > 1e-15:
                            print(f'        [{i},{j}] = '
                                  f'{v.real:+.6e}{v.imag:+.6e}j')
            except Exception as e:
                print(f'        (error displaying: {e})')
    return prop
