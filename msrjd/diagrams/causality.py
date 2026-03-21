"""
msrjd.diagrams.causality
=========================
Retarded propagator consistency and pole-structure compatibility
checks for typed diagrams.

The MSRJD formalism uses retarded Green's functions: G_{ij}(t) is
nonzero only for t > 0.  In Fourier space, this requires all poles
of det(K_hat) to lie in the upper half of the complex omega plane
(Im(omega_k) > 0), ensuring the contour closure gives the retarded
boundary condition.

Build Phase F.
"""

from sage.all import SR, imag_part


def check_pole_structure(pole_vals, omega=None):
    """
    Check whether all poles lie in the upper half-plane (retarded condition).

    Parameters
    ----------
    pole_vals : list
        Pole locations (SR expressions or numbers).
    omega : SR variable or None
        The frequency variable (unused, kept for interface consistency).

    Returns
    -------
    passed : bool
        True if all poles are verified retarded, False if any fails.
        If symbolic evaluation is inconclusive, returns True with conditions.
    details : str
        Human-readable explanation.
    conditions : list of SR expressions
        Symbolic conditions that must hold (e.g., Im(pole) > 0) when
        the check is inconclusive.
    """
    if not pole_vals:
        return True, 'No poles — trivially causal.', []

    failed = []
    conditional = []
    passed_poles = []

    for pole in pole_vals:
        pole_sr = SR(pole)
        im = imag_part(pole_sr)

        try:
            im_simplified = im.simplify_full()
        except Exception:
            im_simplified = im

        # Try to determine sign
        try:
            if bool(im_simplified > 0):
                passed_poles.append(pole_sr)
                continue
        except (TypeError, ValueError):
            pass

        try:
            if bool(im_simplified <= 0):
                failed.append(pole_sr)
                continue
        except (TypeError, ValueError):
            pass

        try:
            if bool(im_simplified == 0):
                failed.append(pole_sr)
                continue
        except (TypeError, ValueError):
            pass

        # Inconclusive — record as conditional
        conditional.append(pole_sr)

    if failed:
        details = (f'FAILED: {len(failed)} pole(s) have Im <= 0: '
                   f'{[str(p) for p in failed]}')
        return False, details, []

    if conditional:
        cond_exprs = [imag_part(p) for p in conditional]
        details = (f'CONDITIONAL: {len(conditional)} pole(s) require '
                   f'Im(pole) > 0 symbolically: {[str(p) for p in conditional]}')
        return True, details, cond_exprs

    details = f'All {len(passed_poles)} pole(s) have Im > 0 — retarded.'
    return True, details, []


def check_causality(typed_diagram, pole_vals=None):
    """
    Check causal consistency of a typed diagram.

    Parameters
    ----------
    typed_diagram : TypedDiagram
    pole_vals : list or None
        Pole locations from propagator computation.

    Returns
    -------
    passed : bool
    details : str
    conditions : list
    """
    D = typed_diagram.prediagram[0]

    # Structural check: the prediagram must be a DAG
    if not D.is_directed_acyclic():
        return False, 'Prediagram contains a directed cycle — acausal.', []

    # Pole check (if available)
    if pole_vals is not None and len(pole_vals) > 0:
        return check_pole_structure(pole_vals)

    return True, 'Structural DAG check passed. No poles to verify.', []


def filter_causal(typed_diagrams, pole_vals=None):
    """
    Keep only causally consistent typed diagrams.

    Parameters
    ----------
    typed_diagrams : list of TypedDiagram
    pole_vals : list or None

    Returns
    -------
    kept : list of TypedDiagram
    n_discarded : int
    details : list of str
        One detail string per discarded diagram.
    """
    kept = []
    discarded_details = []

    for td in typed_diagrams:
        passed, detail, _ = check_causality(td, pole_vals)
        if passed:
            kept.append(td)
        else:
            discarded_details.append(detail)

    return kept, len(typed_diagrams) - len(kept), discarded_details
