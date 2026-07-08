"""Regression test for the Wick-permutation bug on identical external
fields, in BOTH the per-diagram and grouped Phase J paths.

Before the fix, both ``integrate_diagram``'s ``contribution`` closure
(``engine/integration/time_domain/final_integral.py``) and
``integrate_grouped_diagram``'s ``contribution`` closure
(``engine/integration/time_domain/grouped_integral.py``) enumerated 2!
Wick contractions for ⟨x(t1) x(t2)⟩ when both externals were the same
field (e.g. ``[('x', 1), ('x', 1)]``) and re-fed ``ext_time_values``
through each permutation.  With ``origin_leaf_idx`` pinning one leaf at
t=0, the swap permutation routed the pinned 0 into the free-integration
slot and the user's free time into the pinned slot, producing
``(C(τ) + C(0)) / 2`` instead of ``C(τ)``.

For the toy scalar Langevin ``dx/dt = μ x + √(2D)·η`` at xstar=0 the
exact tree-level cumulant is the OU formula
``C(τ) = (D/|μ|)·exp(−|μ|·|τ|)``.  This test runs the framework's
tree-level k=2 evaluation against that closed form and asserts
agreement at machine precision.

Hawkes-style models with distinct external fields (e.g.
``[('n', 1), ('n', 2)]``) never triggered the bug because each field
group had exactly one canonical position → no permutations enumerated.
"""
import importlib.util
import os
import sys
import numpy as np

MODEL_PATH = os.path.join(
    os.path.dirname(__file__), '..', 'models',
    'toy_quartic_double_well.model.py',
)


def _load_model(path):
    spec = importlib.util.spec_from_file_location('model', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.build()


def _run_tree(use_grouped):
    sys.path.insert(0, os.path.abspath(
        os.path.join(os.path.dirname(__file__), '..')))
    from api import compute_cumulants
    m = _load_model(MODEL_PATH)
    mu, D = -1.0, 0.1
    fundamental = {'mu': [mu], 'g': [1.0], 'D': [D]}
    return compute_cumulants(
        m, k=2, max_ell=0,
        external_fields=[('x', 1), ('x', 1)],
        fundamental=fundamental,
        tau_max=3.0, tau_step=0.5,
        use_grouped_phase_j=use_grouped,
        parallel=False, use_cache=False, verbose=False,
    )


def _assert_matches_OU(res, label):
    mu, D = -1.0, 0.1
    tau = res['tau_grid']
    C_framework = res['C_tau'].real
    C_OU = (D / abs(mu)) * np.exp(-abs(mu) * np.abs(tau))
    assert np.max(np.abs(res['C_tau'].imag)) < 1e-12, (
        f'[{label}] imaginary part above numerical zero'
    )
    rel_err = np.max(np.abs(C_framework - C_OU) / np.maximum(np.abs(C_OU), 1e-30))
    # Tolerance accommodates the ~1e-6 offset from the Itô τ=0 left-limit
    # convention (C(0) is evaluated at τ=−ITO_EPS≈−1e-6, not the symmetric 0);
    # away from τ=0 the match is ~machine precision.
    assert rel_err < 1e-5, (
        f'[{label}] Tree-level C(τ) does not match OU formula '
        f'(max rel err = {rel_err:.2e})\n'
        f'  tau  = {tau}\n'
        f'  C    = {C_framework}\n'
        f'  C_OU = {C_OU}'
    )


def test_identical_externals_match_OU_perdiag():
    """Per-diagram path: tree-level ⟨x(0) x(τ)⟩ for the toy quartic at
    xstar=0 must equal the OU formula (D/|μ|)·exp(−|μ|·|τ|).

    Before the Wick-permutation fix in ``final_integral.py`` this test
    failed by a factor that grew with |τ| (up to ~10× at τ=3)."""
    _assert_matches_OU(_run_tree(use_grouped=False), 'per-diag')


def test_identical_externals_match_OU_grouped():
    """Grouped Phase J path: same OU agreement as per-diagram.

    The grouped path had the same Wick-permutation bug in its own
    ``contribution`` closure (``engine/integration/time_domain/
    grouped_integral.py``); this regression locks in the parallel fix."""
    _assert_matches_OU(_run_tree(use_grouped=True), 'grouped')


def _run_1loop(use_grouped):
    sys.path.insert(0, os.path.abspath(
        os.path.join(os.path.dirname(__file__), '..')))
    from api import compute_cumulants
    m = _load_model(MODEL_PATH)
    fundamental = {'mu': [-1.0], 'g': [1.0], 'D': [0.1]}
    return compute_cumulants(
        m, k=2, max_ell=1,
        external_fields=[('x', 1), ('x', 1)],
        fundamental=fundamental,
        tau_max=3.0, tau_step=0.5,
        use_grouped_phase_j=use_grouped,
        parallel=False, use_cache=False, verbose=False,
    )


def _assert_1loop_symmetric(res, label):
    """Equilibrium 2-point function must satisfy C(τ) = C(−τ).

    Before the time-shift fix in the ``contribution`` closure, the
    1-loop kernel of the cubic-vertex tadpole was asymmetric in τ
    (only Case B 'xt at leaf 1' was evaluated; Case A was lost
    because the swap permutation fed a wrong free_val).  After the
    fix, both cases enter via permutation summation and the result
    is exactly symmetric."""
    tau = res['tau_grid']
    C1 = res['C_tau_by_ell'][1].real
    n = len(tau)
    max_asym = 0.0
    for i in range(n):
        j = n - 1 - i   # mirror index
        if abs(tau[i] + tau[j]) < 1e-12:
            max_asym = max(max_asym, abs(C1[i] - C1[j]))
    assert max_asym < 1e-10, (
        f'[{label}] 1-loop C(τ) asymmetry: max|C(τ)−C(−τ)| = '
        f'{max_asym:.2e}\n  tau = {tau}\n  C1  = {C1}'
    )


def test_1loop_symmetric_in_tau_perdiag():
    """Per-diagram path: 1-loop ⟨x(0)x(τ)⟩ symmetric in τ."""
    _assert_1loop_symmetric(_run_1loop(use_grouped=False), 'per-diag')


def test_1loop_symmetric_in_tau_grouped():
    """Grouped path: 1-loop ⟨x(0)x(τ)⟩ symmetric in τ."""
    _assert_1loop_symmetric(_run_1loop(use_grouped=True), 'grouped')


def _run_ou_quartic_2loop(eps, use_grouped=False):
    """Helper: run OU+εx³ to 2 loops at the symmetric saddle xstar=0.

    Uses the production model ``models/ou_quartic_double_well``
    (with literal sign convention ``dx = (−μx − εx³)dt + √(2D)dW``)
    so the Boltzmann perturbative coefficients match exactly.
    """
    sys.path.insert(0, os.path.abspath(
        os.path.join(os.path.dirname(__file__), '..')))
    from api import compute_cumulants
    model_path = os.path.join(
        os.path.dirname(__file__), '..', 'models',
        'ou_quartic_double_well.model.py',
    )
    m = _load_model(model_path)
    return compute_cumulants(
        m, k=2, max_ell=2,
        external_fields=[('x', 1), ('x', 1)],
        fundamental={'mu': 1.0, 'eps': eps, 'D': 1.0},
        tau_max=0.0, tau_step=1.0,
        use_grouped_phase_j=use_grouped,
        parallel=False, use_cache=False, verbose=False,
    )


def test_ou_quartic_2loop_matches_boltzmann_perdiag():
    """OU+εx³ at xstar=0 must satisfy the Boltzmann perturbative series:

        ⟨x²⟩ = 1 − 3ε + 24ε² + O(ε³)

    The 24·ε² coefficient is the 2-loop contribution.  Until the
    May-2026 ``vertex_role_signature`` fix to Phase J's
    ``_compensation``, the framework gave 30·ε² at 2-loop (25% over,
    from double-counting external-leaf permutations whose swap is
    already a graph automorphism — e.g. the 2-loop watermelon, where
    the two cubic vertices are interchangeable under the noise S_3).
    """
    for eps in (0.001, 0.01, 0.1):
        res = _run_ou_quartic_2loop(eps, use_grouped=False)
        tree = res['C_tau_by_ell'][0].real[0]
        l1 = res['C_tau_by_ell'][1].real[0]
        l2 = res['C_tau_by_ell'][2].real[0]
        total = tree + l1 + l2
        boltz = 1 - 3*eps + 24*eps**2
        # 1e-5 absorbs the ~1e-6 Itô τ=0 left-limit offset (the equal-time ⟨x²⟩
        # is the left-limit value); the ε-coefficients still match exactly.
        assert abs(total - boltz) < 1e-5, (
            f'OU+εx³ at ε={eps}: framework total {total} vs '
            f'Boltzmann perturbative {boltz} differ by {total-boltz}'
        )


def test_ou_quartic_2loop_matches_boltzmann_grouped():
    """Same as the per-diagram test, on the grouped Phase J path.

    Locks in the parallel ``_compensation`` fix in
    ``engine/integration/time_domain/grouped_integral.py``.
    """
    for eps in (0.001, 0.01, 0.1):
        res = _run_ou_quartic_2loop(eps, use_grouped=True)
        tree = res['C_tau_by_ell'][0].real[0]
        l1 = res['C_tau_by_ell'][1].real[0]
        l2 = res['C_tau_by_ell'][2].real[0]
        total = tree + l1 + l2
        boltz = 1 - 3*eps + 24*eps**2
        # 1e-5 absorbs the ~1e-6 Itô τ=0 left-limit offset (see the per-diag test).
        assert abs(total - boltz) < 1e-5, (
            f'OU+εx³ at ε={eps} (grouped): framework total {total} vs '
            f'Boltzmann perturbative {boltz} differ by {total-boltz}'
        )


if __name__ == '__main__':
    test_identical_externals_match_OU_perdiag()
    test_identical_externals_match_OU_grouped()
    test_1loop_symmetric_in_tau_perdiag()
    test_1loop_symmetric_in_tau_grouped()
    test_ou_quartic_2loop_matches_boltzmann_perdiag()
    test_ou_quartic_2loop_matches_boltzmann_grouped()
    print('OK')
