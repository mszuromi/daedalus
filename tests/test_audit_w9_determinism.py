"""The Taylor norm bounds of the state-space route are reproducible: the
1-norm estimate used beyond the exact range drew unseeded random start
vectors, so two identical runs could pick different Taylor parameters and
differ in the last bits (seen on Linux CI in
test_delay_loops_memory_61::test_discard_is_bit_identical_and_releases)."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from daedalus.engine.integration.time_domain import statespace as SS  # noqa: E402


def _matrix(n, seed=3):
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((n, n)) / np.sqrt(n)
    return A - 2.0 * np.eye(n)


def test_norms_are_reproducible_in_the_estimated_range(monkeypatch):
    monkeypatch.setattr(SS, '_EXACT_NORM_N', 50)        # force the estimator
    A = _matrix(120)
    first = SS._onenorm_powers(A)
    np.random.seed(1)
    for _ in range(5):
        np.random.rand(7)                                # disturb the global stream
        assert SS._onenorm_powers(A) == first
    # the caller's random stream is untouched by the call
    np.random.seed(5)
    a = np.random.rand(3)
    np.random.seed(5)
    SS._onenorm_powers(A)
    assert np.array_equal(np.random.rand(3), a)


def test_exact_range_is_the_true_norm_and_p_bounds_are_upper_bounds():
    A = _matrix(350)                                     # n > 300: exact since the fix
    d = SS._onenorm_powers(A)
    assert d[2] == float(np.abs(A @ A).sum(axis=0).max()) ** 0.5
    P = A.copy()
    for p in range(2, 6):
        P = P @ A
        assert d[p] >= float(np.abs(P).sum(axis=0).max()) ** (1.0 / p) * (1 - 1e-12)


# ── Pynac's operand order: numeric substitution is the same in every process ──
#
# ``expr.subs({a: 1.1, b: 2.3, c: 3.7})`` on ``a*b*c`` multiplies the floats in an order
# derived from per-process symbol hashes: the last bit of the Phase J prefactors (and so of
# every cumulant) differed from one process to the next (1-4 ulp), whatever PYTHONHASHSEED.
# daedalus.engine.core.detnum.numeric_subs evaluates in a canonical order instead.

import subprocess                                   # noqa: E402
import textwrap                                     # noqa: E402


def _run_py(code, timeout=600, seed='0'):
    env = dict(os.environ, PYTHONHASHSEED=str(seed))
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    code = f"import sys; sys.path.insert(0, {root!r})\n" + textwrap.dedent(code)
    return subprocess.Popen([sys.executable, '-c', code], stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, text=True, env=env, cwd=root)


def _collect(procs):
    outs = []
    for p in procs:
        out, err = p.communicate(timeout=900)
        assert p.returncode == 0, err[-2000:]
        outs.append(out.strip().splitlines()[-1])
    return outs


def test_numeric_subs_unit_values():
    import math
    from sage.all import SR, var, exp
    from daedalus.engine.core.detnum import numeric_subs
    a, b, c, d = var('a b c d')
    vals = {a: 1.1, b: 2.3, c: 3.7, d: 0.31}
    exact = {k: SR(str(v)) for k, v in vals.items()}       # the decimal values, exactly
    for expr in (-a * b * c, a * b * c * d / (a + b), (a + b) * c**2 - d,
                 a * b + c * d - a * d):
        got = complex(numeric_subs(expr, vals))
        ref = complex(expr.subs(exact).n(prec=200))
        assert abs(got - ref) <= 8e-16 * abs(ref), (expr, got, ref)
    # a function of the parameters, a complex value, and a symbol it does not cover
    assert complex(numeric_subs(exp(-a * b), vals)) == pytest.approx(math.exp(-1.1 * 2.3), rel=1e-14)
    assert complex(numeric_subs(a * SR('I') * b, vals)) == pytest.approx(1.1 * 2.3j, rel=1e-14)
    zz = var('zz')
    assert str(numeric_subs(a * zz, {a: 2.0})) == str((a * zz).subs({a: 2.0}))


def test_numeric_subs_is_identical_across_processes():
    code = """
        from sage.all import var
        from daedalus.engine.core.detnum import numeric_subs
        a1, a2, v, w, x = var('a1 a2 v w x')
        vals = {a1: 2.5, a2: 3.7, v: 3.0733, w: 0.1129, x: 1.7e-3}
        out = [numeric_subs(-a1 * a2 * v, vals), numeric_subs(a1 * a2 * v * w * x, vals),
               numeric_subs(a1 * w + a2 * v * x - a1 * a2 / (v + w), vals)]
        print([float(z).hex() for z in out])
    """
    outs = _collect([_run_py(code) for _ in range(8)])
    assert len(set(outs)) == 1, outs


def test_a_loop_cumulant_is_byte_identical_across_processes():
    """The model of the zoo's spike_reset entry (several float factors in every prefactor)
    gave 2-3 byte variants of its one-loop cumulant across processes before."""
    code = """
        import warnings; warnings.simplefilter('ignore')
        import numpy as np
        import daedalus as dd
        from daedalus.api import compute_cumulants
        m, mod = dd.load_model('single_population_spike_reset_test')
        r = compute_cumulants(m, k=2, max_ell=1, external_fields=[('n', 1), ('n', 2)],
            parameters={'Em': [3.5, 3.5], 'tau': [10.0, 9.0], 'a': [2.5, 2.5],
                        'w': [[0.55, 0.65], [0.7, 0.8]]},
            tau_grid=np.array([0.0, 1.0, 3.0]), parallel=False, verbose=False, use_cache=False)
        print([[float(np.real(z)).hex() for z in np.atleast_1d(r['C_tau_by_ell'][l])]
               for l in sorted(r['C_tau_by_ell'])])
    """
    outs = _collect([_run_py(code) for _ in range(6)])
    assert len(set(outs)) == 1, outs


def test_the_dedup_representative_does_not_depend_on_pythonhashseed():
    """Two fields give several external-leg permutations; the deduplication keeps the first
    diagram of each class, and the permutations came from a set of tuples of strings, so the
    representative (and the last bits of the cumulant) changed with PYTHONHASHSEED."""
    code = """
        import warnings; warnings.simplefilter('ignore')
        import numpy as np
        from daedalus.api import compute_cumulants
        from daedalus.api.noise import Gaussian
        from daedalus.api.sde import SDE
        m = (SDE('seedtwin').parameter('mu', default=1.0, domain='positive')
             .parameter('nu', default=2.0, domain='positive').parameter('c', default=0.5)
             .parameter('eps', default=0.1).parameter('D', default=1.0, domain='positive')
             .field('x').field('q').drift('x', '-mu*x + c*q - eps*x*q')
             .drift('q', '-nu*q + eps*x^2')
             .noise('xi', Gaussian(var='2*D'), couples={'x': 1})
             .noise('zeta', Gaussian(var='2*D'), couples={'q': 1}).build())
        out = []
        for ext in ([('x', 1), ('q', 1)],):
            r = compute_cumulants(m, k=2, max_ell=2, external_fields=ext,
                                  tau_grid=np.array([0.0, 0.5, 1.5]), parallel=False,
                                  verbose=False, use_cache=False)
            out.append([[float(np.real(z)).hex() for z in np.atleast_1d(r['C_tau_by_ell'][l])]
                        for l in sorted(r['C_tau_by_ell'])])
        print(out)
    """
    outs = _collect([_run_py(code, seed=sd) for sd in (0, 1, 2, 3, 4, 5)])
    assert len(set(outs)) == 1, outs
