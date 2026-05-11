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
    """
    ns = ft._ns
    taylor_order = ft.taylor_order

    # ── Build basic param substitution dict ──────────────────────
    param_subs = {}
    for pspec in model.get('parameters', []):
        pname = pspec['name']
        if pname not in fundamental:
            continue
        val = fundamental[pname]
        if pspec.get('indexed', False):
            if isinstance(val, list) and val and isinstance(val[0], list):
                # 2D matrix-valued (e.g. w[i][j])
                for i in ns.pop:
                    for j in ns.pop:
                        param_subs[SR.var(f'{pname}{i+1}{j+1}')] = val[i][j]
            else:
                # 1D vector-valued (e.g. E[i])
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
    # to the legacy [1.0] guess for theories whose saddle lives
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
    # mstar (if declared in the model — gracefully skip if not)
    if hasattr(ns, 'mstar'):
        for i in ns.pop:
            try:
                # m*_i = b_X = lambda_X · p_part for GTaS models
                m_expr = SR(model.get('mf_equations', lambda ns: [])(ns))
                # Simpler: just compute it from fundamental params
                if 'lambda_X' in fundamental and 'p_part' in fundamental:
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

    return {
        'nstar_vals':     nstar_vals,
        'vstar_vals':     vstar_vals,
        'phi_deriv_vals': phi_deriv_vals,
        'num_params':     num_params,
        'param_subs':     param_subs,
    }
