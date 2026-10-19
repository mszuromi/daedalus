"""Speed changes of audit wave W8 (SPD-03, SPD-06).

* ``_onenorm_powers``: d_1, d_2 exact; d_p (p > 2) the submultiplicative
  bound, an UPPER bound of the true ``||A^p||_1^{1/p}`` (what the Taylor
  fragment needs) and never above ``||A||_1``;
* ``from api import compute_cumulants`` imports neither matplotlib.pyplot,
  numba nor IPython (they are imported on use).
"""
import os
import subprocess
import sys

import numpy as np
import pytest

pytest.importorskip('sage.all')

from engine.integration.time_domain import statespace as SS


@pytest.mark.parametrize('n', [1, 5, 40])
def test_onenorm_powers_are_upper_bounds(n):
    rng = np.random.default_rng(n)
    for A in (rng.standard_normal((n, n)),
              np.triu(rng.standard_normal((n, n))) - 3 * np.eye(n),
              rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))):
        d = SS._onenorm_powers(A)
        assert sorted(d) == list(range(1, SS._P_MAX + 2))
        P = np.eye(n, dtype=A.dtype)
        for p in range(1, SS._P_MAX + 2):
            P = P @ A
            true = float(np.abs(P).sum(axis=0).max()) ** (1.0 / p)
            assert d[p] >= true * (1 - 1e-12), (p, d[p], true)
            assert d[p] <= d[1] * (1 + 1e-12)
        assert d[1] == pytest.approx(float(np.abs(A).sum(axis=0).max()),
                                     rel=1e-15)


def test_api_import_is_lazy():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    code = ('import sys; from api import compute_cumulants; '
            "print(sorted(m for m in ('matplotlib.pyplot', 'numba', 'IPython')"
            ' if m in sys.modules))')
    out = subprocess.run([sys.executable, '-c', code], cwd=root,
                         capture_output=True, text=True, timeout=600)
    assert out.returncode == 0, out.stderr[-2000:]
    assert out.stdout.strip().splitlines()[-1] == '[]', out.stdout
