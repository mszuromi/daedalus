"""Regression test for the Wick-permutation bug on identical external
fields.

Before the fix in commit (TBD), ``integrate_diagram``'s ``contribution``
closure enumerated 2! Wick contractions for ⟨x(t1) x(t2)⟩ when both
externals were the same field (e.g. ``[('x', 1), ('x', 1)]``) and re-fed
``ext_time_values`` through each permutation.  With ``origin_leaf_idx``
pinning one leaf at t=0, the swap permutation routed the pinned 0 into
the free-integration slot and the user's free time into the pinned slot,
producing ``(C(τ) + C(0)) / 2`` instead of ``C(τ)``.

For the toy scalar Langevin ``dx/dt = μ x + √(2D)·η`` at xstar=0 the
exact tree-level cumulant is the OU formula
``C(τ) = (D/|μ|)·exp(−|μ|·|τ|)``.  This test runs the framework's
tree-level k=2 evaluation against that closed form and asserts
agreement at machine precision.

Hawkes-style theories with distinct external fields (e.g.
``[('n', 1), ('n', 2)]``) never triggered the bug because each field
group had exactly one canonical position → no permutations enumerated.
"""
import importlib.util
import os
import sys
import numpy as np

THEORY_PATH = os.path.join(
    os.path.dirname(__file__), '..', 'theories',
    'toy_quartic_double_well.theory.py',
)


def _load_theory(path):
    spec = importlib.util.spec_from_file_location('theory', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.build()


def test_identical_externals_match_OU():
    """Tree-level ⟨x(0) x(τ)⟩ for the toy quartic at xstar=0 must
    equal the OU formula (D/|μ|)·exp(−|μ|·|τ|).

    Before the Wick-permutation fix this test failed by a factor that
    grew with |τ| (up to ~10× at τ=3).  After the fix the ratio is
    machine-precision 1.0."""
    sys.path.insert(0, os.path.abspath(
        os.path.join(os.path.dirname(__file__), '..')))
    from pipeline import compute_cumulants
    m = _load_theory(THEORY_PATH)
    mu, D = -1.0, 0.1
    fundamental = {'mu': [mu], 'g': [1.0], 'D': [D]}
    res = compute_cumulants(
        m, k=2, max_ell=0,
        external_fields=[('x', 1), ('x', 1)],
        fundamental=fundamental,
        tau_max=3.0, tau_step=0.5,
        use_grouped_phase_j=False,
        parallel=False, use_cache=False, verbose=False,
    )
    tau = res['tau_grid']
    C_framework = res['C_tau'].real
    C_OU = (D / abs(mu)) * np.exp(-abs(mu) * np.abs(tau))
    # Imaginary part should be at numerical-zero level.
    assert np.max(np.abs(res['C_tau'].imag)) < 1e-12

    rel_err = np.max(np.abs(C_framework - C_OU) / np.maximum(np.abs(C_OU), 1e-30))
    assert rel_err < 1e-8, (
        f'Tree-level C(τ) does not match OU formula '
        f'(max rel err = {rel_err:.2e})\n'
        f'  tau  = {tau}\n'
        f'  C    = {C_framework}\n'
        f'  C_OU = {C_OU}'
    )


if __name__ == '__main__':
    test_identical_externals_match_OU()
    print('OK')
