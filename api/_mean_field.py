"""
Mean-field solve and num_params assembly (extracted from notebook cell 23).

Given a model dict and a ``fundamental`` parameter dict, solve the MF
self-consistency equations and assemble the full ``num_params``
substitution dict that the propagator-pole / Phase J machinery needs.
"""
from __future__ import annotations

from sage.all import SR, diff
from scipy.optimize import fsolve


def solve_mean_field(ft, model, fundamental, verbose=True):
    """
    Solve the MF self-consistency equations and return:

      {
        'nstar_vals':       [float, ...],    # n*_i for each population
        'vstar_vals':       [float, ...],    # v*_i
        'phi_deriv_vals':   {(dk, i): float},  # d^k φ / dv^k at v*_i
        'num_params':       {SR.var(...): float, ...},
        'param_subs':       {SR.var(...): float, ...},   # raw user params
      }

    The v* expression is read SYMBOLICALLY from
    ``model['mf_bg_conditions'](ns)[ns.vstar[i]]``, so any model whose
    saddle includes extra terms (GTaS feedforward, additional couplings,
    etc.) is picked up automatically without hardcoding.

    The returned ``num_params`` is suitable for direct use in
    ``compute_poles_and_residues`` and ``compute_correction_td``.

    Heterogeneous-population models take the
    :func:`_solve_mean_field_hetero` branch — multiple iteration
    saddles (one per pop), separate phi-functions per pop, per-pop
    field arrays.  Legacy single-pop models keep the original
    flat-pop code path below.
    """
    if model.get('populations'):
        return _solve_mean_field_hetero(ft, model, fundamental, verbose)

    ns = ft._ns
    taylor_order = ft.taylor_order

    # ── Build basic param substitution dict ──────────────────────
    # For each parameter, resolve its axis sizes from the spec:
    #   * ``indexed_by=['A', 'B']``  → use pop sizes of A, B.
    #   * legacy ``indexed=True/'matrix'``  → use len(ns.pop) for
    #     every axis (legacy single-population path).
    # The user's numerical value's actual shape determines whether we
    # iterate as scalar / vector / matrix; the spec just tells us
    # where the SR vars came from.
    pop_size_map = getattr(ns, '_pop_size', {}) or {}
    param_subs = {}
    for pspec in model.get('parameters', []):
        pname = pspec['name']
        if pname not in fundamental:
            continue
        val = fundamental[pname]
        ib = pspec.get('indexed_by')
        if ib:
            # Heterogeneous-pop path.
            if len(ib) == 2:
                n_rows = pop_size_map.get(ib[0], len(ns.pop))
                n_cols = pop_size_map.get(ib[1], len(ns.pop))
                for i in range(n_rows):
                    for j in range(n_cols):
                        param_subs[SR.var(f'{pname}{i+1}{j+1}')] = val[i][j]
            elif len(ib) == 1:
                n = pop_size_map.get(ib[0], len(ns.pop))
                for i in range(n):
                    param_subs[SR.var(f'{pname}{i+1}')] = val[i]
            else:
                # Scalar via empty indexed_by — treat like un-indexed.
                param_subs[SR.var(pname)] = val
        elif pspec.get('indexed', False):
            # Legacy single-pop path.
            if isinstance(val, list) and val and isinstance(val[0], list):
                for i in ns.pop:
                    for j in ns.pop:
                        param_subs[SR.var(f'{pname}{i+1}{j+1}')] = val[i][j]
            else:
                for i in ns.pop:
                    param_subs[SR.var(f'{pname}{i+1}')] = val[i]
        else:
            param_subs[SR.var(pname)] = val

    # ── phi derivatives, symbolic ────────────────────────────────
    v_sym = SR.var('_v_mf_')
    phi_derivs = {}
    for i in ns.pop:
        phi_expr = model['phi_concrete'](ns, i, v_sym)
        phi_derivs[i] = {}
        for dk in range(taylor_order + 1):
            if dk == 0:
                phi_derivs[i][dk] = phi_expr
            else:
                phi_derivs[i][dk] = diff(phi_expr, v_sym, dk)

    def phi_num(i, v_val):
        return float(phi_derivs[i][0].subs(param_subs).subs({v_sym: v_val}))

    # ── Build symbolic v*_i from mf_bg_conditions, bake params ──
    vstar_subs_dict = model.get('mf_bg_conditions', lambda ns: {})(ns)
    vstar_sym = {
        i: SR(vstar_subs_dict.get(ns.vstar[i], ns.E[i]))
        for i in ns.pop
    }
    g_to_one = {ns.g: SR(1)}
    vstar_baked = {
        i: SR(vstar_sym[i]).subs(g_to_one).subs(param_subs)
        for i in ns.pop
    }

    def vstar_num(i, nstar_vec):
        sub = {ns.nstar[j]: float(nstar_vec[j]) for j in ns.pop}
        return float(vstar_baked[i].subs(sub))

    # ── Build the iteration target for nstar ────────────────────
    # If the user's mf_eq for nstar is in the dict (compound or
    # otherwise), use it as the iteration target.  Otherwise fall
    # back to the legacy ``n* = phi(v*)`` hardcode.
    nstar_rhs_baked = None
    if hasattr(ns, 'nstar') and ns.nstar[0] in vstar_subs_dict:
        nstar_rhs_baked = {
            i: SR(vstar_subs_dict[ns.nstar[i]])
                 .subs(g_to_one)
                 .subs(param_subs)
            for i in ns.pop
        }

    def nstar_target(i, nstar_vec):
        """Evaluate the user-declared RHS of nstar's mf_eq at the
        current iteration point."""
        if nstar_rhs_baked is None:
            # Legacy: assume n* = phi(v*).
            return phi_num(i, vstar_num(i, nstar_vec))
        v_vals = {ns.vstar[j]: vstar_num(j, nstar_vec) for j in ns.pop}
        n_vals = {ns.nstar[j]: float(nstar_vec[j]) for j in ns.pop}
        return float(nstar_rhs_baked[i].subs(v_vals).subs(n_vals))

    # ── fsolve the self-consistency  n*_i = <user RHS>(n*_j) ────
    # Wrap mf_residual so symbolic singularities (e.g. division by
    # zero when an mf_eq RHS like ``1/(1 - nstar)`` is evaluated at
    # the wrong point) become large finite residuals instead of
    # raising — that lets fsolve back away from the singularity and
    # keep searching rather than aborting the whole solve.
    def mf_residual(nstar_vec):
        try:
            return [nstar_vec[i] - nstar_target(i, nstar_vec)
                    for i in ns.pop]
        except (ValueError, ZeroDivisionError):
            return [1e10] * len(ns.pop)

    npop = len(ns.pop)

    # Try a sequence of initial guesses, starting with small values
    # safe for mf_eqs that have singularities at large nstar (e.g.
    # spike-reset with vstar = ... / (1 - nstar)), and falling back
    # to the legacy [1.0] guess for models whose saddle lives
    # there.  ``fsolve`` returns ier=1 on clean convergence; we
    # keep the first ier=1 result and otherwise fall through with
    # the last attempt.
    initial_guesses = [
        [0.1]  * npop,
        [0.5]  * npop,
        [1.0]  * npop,    # legacy default
        [0.01] * npop,
    ]
    sol = None
    for x0 in initial_guesses:
        try:
            attempt = fsolve(mf_residual, x0, full_output=True)
        except (ValueError, ZeroDivisionError):
            continue
        sol = attempt
        if attempt[2] == 1:    # ier == 1  ⇒  converged
            break

    if sol is None:
        raise RuntimeError(
            'solve_mean_field: fsolve failed on every initial guess.  '
            'Check the mf_eq for symbolic singularities or supply a '
            'custom starting point.')
    nstar_vals = [float(x) for x in sol[0]]

    # ── Sanity-check the recovered saddle ─────────────────────────
    # Non-finite or wildly large values mean fsolve diverged into a
    # symbolic singularity (e.g. spike-reset's ``1/(1 - n*)`` blows
    # up as n* → 1).  Catch that explicitly rather than handing
    # garbage downstream to the propagator / Phase J machinery.
    if not all(__import__('math').isfinite(x) for x in nstar_vals):
        raise RuntimeError(
            f'solve_mean_field: fsolve returned non-finite n* = '
            f'{nstar_vals!r}.  The mf_eq probably has a singularity '
            f'within the iteration basin; check parameter values.')
    # Heuristic warning: if ier != 1 we may have a non-converged point.
    sol_ier = sol[2]
    if sol_ier != 1:
        msg = sol[3] if len(sol) > 3 else 'fsolve reported non-convergence'
        if verbose:
            print(f'  ⚠ solve_mean_field: fsolve ier={sol_ier} '
                  f'({msg!r}) — saddle may be unreliable.')

    # ── Evaluate v* and phi derivatives at the fixed point ──────
    vstar_vals = []
    phi_deriv_vals = {}
    for i in ns.pop:
        vi = vstar_num(i, nstar_vals)
        if not __import__('math').isfinite(vi):
            raise RuntimeError(
                f'solve_mean_field: vstar[{i}] = {vi!r} is non-finite. '
                f'Likely cause: nstar = {nstar_vals[i]:.6g} is near a '
                f'pole of the v-saddle equation (e.g. spike-reset '
                f'1/(1 - n*) ).  Reduce excitation or supply a custom '
                f'initial guess.')
        vstar_vals.append(vi)
        for dk in range(taylor_order + 1):
            phi_deriv_vals[(dk, i)] = float(
                phi_derivs[i][dk].subs(param_subs).subs({v_sym: vi})
            )

    if verbose:
        print(f'\nMean-field solution:')
        for i in ns.pop:
            phi0 = phi_deriv_vals[(0, i)]
            ok = abs(phi0 - nstar_vals[i]) < 1e-10
            print(f'  pop {i+1}:  v* = {vstar_vals[i]:.6f},  '
                  f'n* = {nstar_vals[i]:.6f},  '
                  f'phi(v*) = {phi0:.6f}  '
                  f'{"OK" if ok else "MISMATCH!"}')

    # ── Assemble full num_params (param_subs + nstar/vstar/mstar/phi*) ──
    num_params = dict(param_subs)
    for i in ns.pop:
        num_params[ns.nstar[i]] = float(nstar_vals[i])
        num_params[ns.vstar[i]] = float(vstar_vals[i])
    # mstar (if declared in the model — gracefully skip if not).
    # m*_i = b_X = lambda_X · p_part for GTaS models.  (The earlier
    # ``SR(model['mf_equations'](ns))`` probe here raised on the
    # equation *list* and, being inside the try, silently swallowed the
    # assignment below — so mstar never made it into num_params and the
    # GTaS rate b_X showed as missing.)
    if hasattr(ns, 'mstar') and 'lambda_X' in fundamental \
            and 'p_part' in fundamental:
        for i in ns.pop:
            try:
                num_params[ns.mstar[i]] = float(
                    fundamental['lambda_X'] * fundamental['p_part']
                )
            except (TypeError, ValueError):
                pass
    for i in ns.pop:
        for dk in range(1, taylor_order + 1):
            sym = SR.var(f'phi{dk}_{i+1}')
            if (dk, i) in phi_deriv_vals:
                num_params[sym] = phi_deriv_vals[(dk, i)]

    # ── phi0_i (φ at the saddle) from mf_bg_conditions ───────────
    # Template-built models declare φ as a FORMAL function, so the
    # (2,0) Poisson-noise sector carries the formal symbol ``phi0_i``
    # (= the mean firing rate ``nstar_i`` at the saddle).  The
    # per-derivative loop above only resolves ``phi1_i`` upward, so
    # without this ``phi0_i`` stays symbolic, the noise-source
    # coefficient ``-1/2*phi0_i`` never becomes numeric, and the whole
    # correlator collapses to 0.  ``vstar_subs_dict`` (the model's
    # ``mf_bg_conditions``) maps ``phi0_i -> nstar_i``; evaluate each
    # non-saddle entry at the solved saddle.  (Text-built models
    # inline a concrete φ and declare no ``phi0_i`` key, so this is a
    # no-op for them.)
    for _lhs, _rhs in vstar_subs_dict.items():
        if _lhs in num_params:
            continue          # saddle vars already resolved above
        try:
            num_params[_lhs] = float(
                SR(_rhs).subs(g_to_one).subs(num_params))
        except (TypeError, ValueError):
            pass              # still symbolic — leave it out

    return {
        'nstar_vals':     nstar_vals,
        'vstar_vals':     vstar_vals,
        'phi_deriv_vals': phi_deriv_vals,
        'num_params':     num_params,
        'param_subs':     param_subs,
    }


# ── Heterogeneous-population MF solver ────────────────────────────
def _solve_mean_field_hetero(ft, model, fundamental, verbose=True):
    """MF solver for models that declare ``model['populations']``.

    Generalises the legacy single-pop solver:

      * Iteration saddles are detected from mf_bg_conditions (each
        saddle whose mf_eq RHS is a single function call ⇒ closure
        saddle).  Typical pattern: ``nEstar = phiE(vEstar)``,
        ``nIstar = phiI(vIstar)``.
      * Compound saddles (``vEstar``, ``vIstar``) are evaluated
        concretely from their mf_eq RHS, with iteration-saddle SR
        vars left raw so they can be substituted at each iteration.
      * fsolve iterates over a flat vector concatenating all
        iteration-saddle elements (per-pop sized).  Each element's
        target is its closure RHS evaluated at the current point.
    """
    ns = ft._ns
    taylor_order = ft.taylor_order
    populations = model['populations']
    pop_size = {p['name']: int(p['size']) for p in populations}

    # ── param_subs (vector + matrix per indexed_by) ──────────────
    param_subs = {}
    for pspec in model.get('parameters', []):
        pname = pspec['name']
        if pname not in fundamental:
            continue
        val = fundamental[pname]
        ib = pspec.get('indexed_by')
        if ib:
            if len(ib) == 2:
                n_rows = pop_size.get(ib[0], 0)
                n_cols = pop_size.get(ib[1], 0)
                for i in range(n_rows):
                    for j in range(n_cols):
                        param_subs[SR.var(f'{pname}{i+1}{j+1}')] = val[i][j]
            elif len(ib) == 1:
                n = pop_size.get(ib[0], 0)
                for i in range(n):
                    param_subs[SR.var(f'{pname}{i+1}')] = val[i]
        else:
            param_subs[SR.var(pname)] = val

    # ── Identify saddles and the population each one belongs to ──
    # Saddle params have mean_field=True; their indexed_by tells us
    # which population's local indices they range over.
    saddle_info = {}    # name → {'pop': X, 'size': N, 'sr_array': [...]}
    for pspec in model.get('parameters', []):
        if not pspec.get('mean_field'):
            continue
        sname = pspec['name']
        ib    = pspec.get('indexed_by') or []
        pop   = ib[0] if ib else None
        size  = pop_size.get(pop, 0)
        if size == 0 or not hasattr(ns, sname):
            continue
        saddle_info[sname] = {
            'pop':       pop,
            'size':      size,
            'sr_array':  getattr(ns, sname),
        }

    # ── mf_bg dict (raw concrete with iteration saddles unbaked) ─
    mf_bg = model.get('mf_bg_conditions', lambda ns: {})(ns)
    # The mf_bg dict maps each saddle's SR var → its mf_eq RHS expr
    # (post param_subs / closure baking).  Compound saddles depend
    # on iteration-saddle SR vars; iteration saddles depend on
    # compound-saddle SR vars.  We classify by the rhs structure:
    # if the saddle's rhs is a single function call (e.g. phi(vstar)),
    # it's an iteration saddle.

    # Iteration vs compound classification.  Prefer the model's
    # ``iteration_saddles`` list (computed at ModelBuilder build()
    # time by the same _classify_mf_eqs the compiler uses) — that's
    # the authoritative source.  Fall back to a structural heuristic
    # only if it's missing.
    declared_iter = list(model.get('iteration_saddles') or [])
    if declared_iter:
        iteration_saddles = [s for s in declared_iter if s in saddle_info]
        compound_saddles  = [s for s in saddle_info if s not in iteration_saddles]
    else:
        # Legacy / fallback: every saddle is iteration (caller will
        # pick whatever fsolve converges to).
        iteration_saddles = list(saddle_info.keys())
        compound_saddles  = []
    if verbose:
        print(f'  iteration saddles: {iteration_saddles}')
        print(f'  compound saddles:  {compound_saddles}')

    # ── Kernel-symbol → 1 substitution dict ──────────────────────
    # At the saddle, every kernel integrates to 1 (kernels are
    # normalized).  Build a substitution dict mapping every kernel
    # SR var on ns to 1, so the mf_bg RHS reduces to plain
    # parameter / saddle algebra.  Handles scalar (``ns.g``), vector
    # (``ns.g = [g1, g2]``), and matrix (``ns.g = [[g11, ...]]``)
    # kernels uniformly.
    kernel_to_one = {}
    for kspec in model.get('kernels', []):
        kname = kspec['name']
        if not hasattr(ns, kname):
            continue
        kval = getattr(ns, kname)
        if isinstance(kval, list):
            for row in kval:
                if isinstance(row, list):
                    for sym in row:
                        kernel_to_one[sym] = SR(1)
                else:
                    kernel_to_one[row] = SR(1)
        else:
            kernel_to_one[kval] = SR(1)

    # ── Pre-bake compound-saddle RHS with params + kernels → 1 ───
    compound_rhs = {}
    for sname in compound_saddles:
        arr = saddle_info[sname]['sr_array']
        compound_rhs[sname] = [
            SR(mf_bg.get(sym, sym))
              .subs(kernel_to_one)
              .subs(param_subs)
            for sym in arr
        ]

    # ── Iteration target: closure RHS evaluated at current point ─
    iter_rhs = {}
    for sname in iteration_saddles:
        arr = saddle_info[sname]['sr_array']
        iter_rhs[sname] = [
            SR(mf_bg.get(sym, sym))
              .subs(kernel_to_one)
              .subs(param_subs)
            for sym in arr
        ]

    # Total iteration-vector size:  sum of all iteration saddle sizes.
    sizes = [saddle_info[s]['size'] for s in iteration_saddles]
    total_n = sum(sizes)
    # Index in flat fsolve vector ↔ (saddle name, local index).
    flat_index = []
    for sname in iteration_saddles:
        for i in range(saddle_info[sname]['size']):
            flat_index.append((sname, i))

    def _unflatten(flat_vec):
        """Convert a flat fsolve vector to a dict of per-saddle arrays."""
        out: dict = {}
        offset = 0
        for sname in iteration_saddles:
            n = saddle_info[sname]['size']
            out[sname] = [float(flat_vec[offset + k]) for k in range(n)]
            offset += n
        return out

    def _eval_compound(iter_vals):
        """Given current iteration values, compute compound saddle
        values by substituting iteration-saddle SR vars."""
        # Build the substitution dict from iteration values.
        sub = {}
        for sname, vals in iter_vals.items():
            for i, v in enumerate(vals):
                sub[saddle_info[sname]['sr_array'][i]] = v
        # Compound saddles also reference each other?  For Hawkes
        # models, compounds depend only on iteration saddles.
        # Evaluate each compound RHS once with the iteration subs.
        out = {}
        for sname in compound_saddles:
            arr = compound_rhs[sname]
            out[sname] = [float(SR(expr).subs(sub)) for expr in arr]
        return out

    def _residual(flat_vec):
        try:
            iter_vals = _unflatten(flat_vec)
            comp_vals = _eval_compound(iter_vals)
            # Build the full substitution dict for closure RHS eval.
            sub = {}
            for sname, vals in iter_vals.items():
                for i, v in enumerate(vals):
                    sub[saddle_info[sname]['sr_array'][i]] = v
            for sname, vals in comp_vals.items():
                for i, v in enumerate(vals):
                    sub[saddle_info[sname]['sr_array'][i]] = v
            # Closure target for each iteration-saddle element.
            res = []
            for sname in iteration_saddles:
                arr = iter_rhs[sname]
                for i, target_expr in enumerate(arr):
                    target = float(SR(target_expr).subs(sub))
                    res.append(float(iter_vals[sname][i]) - target)
            return res
        except (ValueError, ZeroDivisionError, TypeError):
            return [1e10] * total_n

    # ── fsolve ──────────────────────────────────────────────────
    initial_guesses = [
        [0.1] * total_n, [0.5] * total_n,
        [1.0] * total_n, [0.01] * total_n,
    ]
    sol = None
    for x0 in initial_guesses:
        try:
            attempt = fsolve(_residual, x0, full_output=True)
        except (ValueError, ZeroDivisionError):
            continue
        sol = attempt
        if attempt[2] == 1:
            break
    if sol is None:
        raise RuntimeError(
            'solve_mean_field (heterogeneous): fsolve failed on every '
            'initial guess.  Check the mf_eqs for symbolic singularities.')

    # ── Unpack and verify ────────────────────────────────────────
    flat_vals = list(sol[0])
    import math
    if not all(math.isfinite(x) for x in flat_vals):
        raise RuntimeError(
            f'solve_mean_field (heterogeneous): fsolve returned non-finite '
            f'iteration-saddle values {flat_vals!r}.')

    iter_vals = _unflatten(flat_vals)
    comp_vals = _eval_compound(iter_vals)

    # ── Assemble num_params: every saddle SR var → its solved value ─
    num_params = dict(param_subs)
    for sname, vals in iter_vals.items():
        arr = saddle_info[sname]['sr_array']
        for i, v in enumerate(vals):
            num_params[arr[i]] = float(v)
    for sname, vals in comp_vals.items():
        arr = saddle_info[sname]['sr_array']
        for i, v in enumerate(vals):
            num_params[arr[i]] = float(v)

    if verbose:
        print('\nMean-field solution:')
        for sname in iteration_saddles + compound_saddles:
            label = ('iter ' if sname in iteration_saddles else 'comp ')
            pop   = saddle_info[sname]['pop']
            vals  = (iter_vals.get(sname) or comp_vals.get(sname))
            print(f'  {label}{sname}[pop {pop}]: {vals}')

    # Legacy keys for downstream compatibility — pick the first
    # n-rate-like iteration saddle (size = pop_E for spike-train rate).
    # Callers reading 'nstar_vals' / 'vstar_vals' get a concatenated
    # vector across all populations (in declaration order).
    nstar_concat = []
    vstar_concat = []
    for sname in iteration_saddles:
        nstar_concat.extend(iter_vals[sname])
    for sname in compound_saddles:
        vstar_concat.extend(comp_vals[sname])

    return {
        'nstar_vals':     nstar_concat,
        'vstar_vals':     vstar_concat,
        'phi_deriv_vals': {},     # populated below if needed by callers
        'num_params':     num_params,
        'param_subs':     param_subs,
        'saddle_values':  {**iter_vals, **comp_vals},
        'saddle_info':    saddle_info,
    }
