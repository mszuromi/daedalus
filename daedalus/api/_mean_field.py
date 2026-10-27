"""
Mean-field front door: parameter completion, the kernel values at the
saddle, and :func:`solve_mean_field`, which hands every model to the one
multi-root solver in :mod:`daedalus.api._mean_field_dae` (the
former flat-population branch read the names ``nstar``/``vstar``/``E``/``g``
and the single-root iteration started from four fixed unit-scale guesses).
"""
from __future__ import annotations

from sage.all import SR


def _kernel_saddle_subs(ns, model, kernel_syms, exprs):
    """``{kernel symbol: value at the saddle}``: the kernel's integral
    ``ĝ(0)`` (read off ``kernel_ft_image`` at omega = 0) when the symbol
    occurs in ``exprs``, else (and for a kernel without an image, or with
    integral 1) the exact ``SR(1)`` of the normalised rule."""
    from daedalus.engine.core.field_theory import kernel_dc_gains_of
    present = set()
    for e in exprs:
        try:
            present.update(SR(e).variables())
        except (TypeError, ValueError):
            continue
    gains = kernel_dc_gains_of(ns, model)
    out = {}
    for sym in kernel_syms:
        g = gains.symbol_gain(sym) if sym in present else None
        out[sym] = SR(1) if g is None else g
    return out


def complete_parameters(model, fundamental, who='the mean-field solve',
                        strict=True):
    """``fundamental`` with every omitted parameter filled from the model's
    declared ``default=``; a parameter that is omitted AND has no default
    raises a :class:`ValueError` naming it (a partial ``parameters=`` dict
    otherwise surfaced as a bare ``NameError`` from inside the solve).
    ``strict=False`` only fills (the solvers use it: a no-default parameter
    the saddle equations never read, e.g. a noise strength, must not stop a
    direct solve; ``compute_cumulants`` is the strict front door).
    Mean-field parameters (``mean_field=True``) are solved, never supplied.
    The caller's dict is not modified; a complete dict comes back equal."""
    out = dict(fundamental or {})
    missing = []
    for pspec in model.get('parameters', []) or []:
        name = pspec['name']
        if name in out or pspec.get('mean_field'):
            continue
        if pspec.get('default') is not None:
            out[name] = pspec['default']
        else:
            missing.append(name)
    if missing and strict:
        raise ValueError(
            f'{who}: these model parameters are not in the parameters '
            f'given and declare no default: {", ".join(sorted(missing))}. '
            f'Pass them in parameters={{...}} or declare defaults via '
            f'ModelBuilder.parameter(name, default=...).')
    return out


def solve_mean_field(ft, model, fundamental, verbose=True, *,
                     fixed_point_index=0, n_starts=64, seed_box=None):
    """Solve the model's mean-field equations (``.equation`` rows or
    ``.set_mf_equation`` saddle equations) and return the solver dict of
    :func:`daedalus.api._mean_field_dae.solve_mean_field_dae_compat`: ``num_params``
    (every parameter and saddle SR symbol on ``ft._ns`` -> value, ready for
    the propagator and Phase J), ``saddle_values`` / ``mf_values`` (``{saddle
    name: [values]}``), ``param_subs`` and the root census.  Multiple roots
    are all found; ``fixed_point_index`` picks one (sorted by the first
    physical field's saddle)."""
    if not model.get('populations'):
        raise ValueError(
            'solve_mean_field: the model declares no populations; build it '
            'with ModelBuilder (which declares one for every model: the '
            'scalar population, the n_populations=N population or the '
            '.population(...) declarations).')
    from daedalus.api._mean_field_dae import solve_mean_field_dae_compat
    return solve_mean_field_dae_compat(
        ft, model, fundamental, fixed_point_index=fixed_point_index,
        n_starts=n_starts, seed_box=seed_box, verbose=verbose)
