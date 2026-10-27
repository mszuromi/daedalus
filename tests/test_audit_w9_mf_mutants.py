"""Direct tests of the mean-field root-acceptance and linear-stability tolerances.

The mutation check of the final audit found `_ROOT_RTOL` 1e-7 -> 1e-4 surviving every
mean-field test file (and 1e-7 -> 1e-2 killed by one indirect test only): no test fed
`_refine_root` a point that is NOT a root with a small relative residual.  These do,
with closed-form references, and pin the tolerance itself.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

pytest.importorskip('sage.all')

from daedalus.api import _mean_field_dae as MF          # noqa: E402


def _residual(rows):
    """``R(x, m=None)`` of residual rows ``rows(x) -> [(lhs, rhs), ...]``: the residual is
    lhs - rhs, and ``m`` (when given) is filled with |lhs| + |rhs| (the round-off scale)."""
    def R(x, m=None):
        pairs = rows(np.asarray(x, dtype=float))
        f = np.array([a - b for a, b in pairs])
        if m is not None:
            m[:] = [abs(a) + abs(b) for a, b in pairs]
        return f
    return R


def _refine(rows, x0):
    c = np.maximum(np.abs(np.asarray(x0, dtype=float)), 1e-8)
    return MF._refine_root(_residual(rows), np.asarray(x0, dtype=float), c)


def test_the_tolerance_values_themselves():
    assert MF._ROOT_RTOL == 1e-7
    assert MF._EIG_RTOL == 1e-12


def test_a_consistent_overdetermined_system_is_accepted():
    # x = 1 solves both rows exactly
    out = _refine(lambda x: [(x[0], 1.0), (2 * x[0], 2.0)], [1.0 + 1e-3])
    assert out is not None
    assert out[0][0] == pytest.approx(1.0, rel=1e-12)


@pytest.mark.parametrize('gap', [3e-5, 3e-3])
def test_a_point_with_a_small_relative_residual_is_not_a_root(gap):
    """Two rows that disagree by a relative `gap`: the least-squares point has residuals of
    gap/2 against terms of size ~1, so it is no root.  1e-7 rejects it; the mutants 1e-4
    (gap 3e-5) and 1e-2 (both gaps) accept it."""
    out = _refine(lambda x: [(x[0], 1.0), (x[0], 1.0 + gap)], [1.0])
    assert out is None


def test_the_acceptance_is_relative_not_absolute():
    """The same system in huge units (rows of size 1e9) with the same relative gap 3e-5 is
    still rejected, and a consistent one in tiny units (1e-9) is still accepted: the test
    is relative to the row's own terms (the absolute 1e-7 rule it replaced got both wrong)."""
    big = 1e9
    assert _refine(lambda x: [(big * x[0], big), (big * x[0], big * (1 + 3e-5))], [1.0]) is None
    tiny = 1e-9
    out = _refine(lambda x: [(tiny * x[0], tiny), (2 * tiny * x[0], 2 * tiny)], [1.0 + 1e-4])
    assert out is not None and out[0][0] == pytest.approx(1.0, rel=1e-9)


def test_a_residual_below_the_tolerance_is_still_accepted():
    """The other side of the boundary: rows disagreeing by a relative 1e-8 (residual 5e-9
    per row against terms ~1) are within the 1e-7 acceptance; a mutant tightening it to
    1e-9 would reject genuine roots of equations written with ~1e-8 rounding."""
    out = _refine(lambda x: [(x[0], 1.0), (x[0], 1.0 + 1e-8)], [1.0])
    assert out is not None and out[0][0] == pytest.approx(1.0, rel=1e-7)
