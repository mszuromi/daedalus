"""The mean-field sector check at a parameter point where the saddle's closed form has a pole.

``MasterEquation.to_sde()`` of a birth-death process with death rate ``mu x + c x (x - 1)`` and birth ``a + b x`` writes
the drift's linear part on the left (restoring coefficient ``Om h (mu - b - c)``).  At a default point with
``mu = b + c`` that coefficient is exactly 0: the symbolic saddle substitution divides by it, the substituted
coefficient could not be evaluated, and the check reported "the MF action sector is non-zero" although the saddle
(``x* = sqrt(a / c)``) is fine.  The check now falls back on the coefficient before the substitution at the numerical
saddle.
"""
import warnings

import numpy as np
import pytest

pytest.importorskip('sage.all')

import daedalus as dd                                                  # noqa: E402

PARAMS = dict(a=1.0, b=0.2, mu=0.5, c=0.3)            # mu - b - c = 0 exactly (in floats too)


def _me(Om, **over):
    p = dict(PARAMS, Om=Om, **over)
    return dd.master_equation([('Om*(a + b*x)', +1), ('Om*(mu*x + c*x*(x-1))', -1)], fields=['x'],
                              params=p, h=1 / Om, box='auto', lower=0, guess=[1.8])


def test_degenerate_restoring_coefficient_builds_and_matches_the_exact_cumulants():
    """One loop against the exact master equation at system sizes 100 and 200: the error falls by the factor 4
    that O(h^2) predicts (tree error O(h))."""
    err = {}
    for Om in (100.0, 200.0):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            rows = _me(Om).compare_with_daedalus(ks=(1, 2), max_ell=1).rows
        err[Om] = {r['cumulant']: (r['rel_error'][0], r['rel_error'][1]) for r in rows}
    for name in err[100.0]:
        (t100, l100), (t200, l200) = err[100.0][name], err[200.0][name]
        assert 1.8 < t100 / t200 < 2.2, (name, t100, t200)            # tree: O(h)
        assert l200 < 2e-6 and 3.0 < l100 / l200 < 5.5, (name, l100, l200)   # one loop: O(h^2)


def test_generic_point_is_unchanged():
    """Away from the degenerate point the model builds as before (the saddle sector check passed there)."""
    me = _me(100.0, b=0.1)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        rows = me.compare_with_daedalus(ks=(1,), max_ell=1).rows
    assert rows


# (a wrong mean-field equation is still refused: tests/test_leftovers_39.py, tests/test_mf_saddle_fixes.py)
