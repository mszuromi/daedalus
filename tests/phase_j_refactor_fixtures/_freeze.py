"""
Phase J refactor — freeze the regression fixtures.

Usage (from repo root)::

    sage -python tests/phase_j_refactor_fixtures/_freeze.py

Runs every config in ``_configs.FIXTURES``, evaluates ``total_C(*τ)`` at
each probe, and writes one ``.npz`` per fixture (overwriting if
present).  Print a summary of wall times + first probe values so a
human can sanity-check before committing.

Only run this on the CURRENT, KNOWN-GOOD code state.  Once committed
to the branch, ``test_phase_j_refactor_regression.py`` compares
post-refactor evaluations to these frozen values.
"""
from __future__ import annotations

import os
import sys

# Ensure the repo root is importable.
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, '..', '..'))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from tests.phase_j_refactor_fixtures._configs import FIXTURES
from tests.phase_j_refactor_fixtures._runner import evaluate, save_frozen


def main() -> int:
    print(f'Freezing {len(FIXTURES)} fixtures to '
          f'{os.path.relpath(_HERE, _REPO_ROOT)}/ ...')
    print()
    for fx in FIXTURES:
        print(f'  [{fx.name}]')
        print(f'    model   = {fx.model_file}')
        print(f'    k={fx.k}, max_ell={fx.max_ell}, '
              f'n_probes={len(fx.tau_probes)}')
        result = evaluate(fx)
        path = save_frozen(result)
        rel = os.path.relpath(path, _REPO_ROOT)
        print(f'    wall     = {result["wall_time"]:.2f}s')
        print(f'    sample   = C({fx.tau_probes[0]}) = '
              f'{result["C_values"][0]}')
        print(f'    written  = {rel}')
        print()
    print('Done.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
