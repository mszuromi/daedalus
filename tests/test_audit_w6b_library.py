"""
tests/test_audit_w6b_library.py
===============================
Audit wave W6 (library group): noise library, symbolic covariance,
realization test, simulator arrays.

* LOUD-01: ``NormalInverseGaussian`` with parameter-named ``alpha``/``beta``
  runs and equals its literal twin and the closed forms (mean
  ``delta beta / sqrt(alpha^2 - beta^2)``, ``C(0) = kappa_2 / (2 mu)``).
* GEN-08: the symbolic C2 ``tree_covariance`` of a fully coupled 3-field
  linear SDE with float coefficients equals its rational twin and the
  Lyapunov covariance; a supplied ``G`` in the wrong convention is still
  refused.
* GEN-15: the exact Hurwitz test also decides complex (``QQ[I]``)
  realizations (agrees with the numeric eigenvalues), so their a-posteriori
  check can run.
* GEN-16: ``build_sim_arrays`` takes the coupling signs from the model's
  equations, not from population names: an A/B-named copy of an E/I model
  gives the same signed ``W``; a coupled pair without a time constant is
  NaN, not 1.0.

Run:  PYTHONHASHSEED=0 sage -python -m pytest tests/test_audit_w6b_library.py -q
"""
from __future__ import annotations

import math
import os
import random
import sys
import warnings
from fractions import Fraction

import numpy as np
import pytest
import scipy.linalg as sla

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from api import compute_cumulants                          # noqa: E402
from api.model import ModelBuilder                         # noqa: E402
from api.noise import Gaussian, NormalInverseGaussian      # noqa: E402
from api.sde import SDE                                    # noqa: E402

RUN = dict(parallel=False, verbose=False, use_cache=False)


# ── LOUD-01 ───────────────────────────────────────────────────────────
def _nig(src, pv, mu=1.0, tag=''):
    s = SDE('w6b nig ' + tag).parameter('mu', default=mu)
    for p, v in pv.items():
        s = s.parameter(p, default=v)
    m = s.field('x').drift('x', '-mu*x').noise('xi', src, couples={'x': 1}).build()
    r = compute_cumulants(m, k=2, max_ell=0, external_fields=[('x', 1), ('x', 1)],
                          tau_grid=np.array([0.0, 0.7]), **RUN)
    return r['mf_values']['xstar'][0], np.array([complex(c).real for c in r['C_tau']])


def test_loud01_nig_symbolic_parameters_match_closed_form():
    a, b, d, mu = 3.0, 0.5, 0.4, 1.0
    x_sym, c_sym = _nig(NormalInverseGaussian('na', 'nb', 'nd'),
                        dict(na=a, nb=b, nd=d), mu, 'sym')
    x_lit, c_lit = _nig(NormalInverseGaussian(a, b, d), {}, mu, 'lit')
    mean = d * b / math.sqrt(a * a - b * b)
    k2 = d * a * a / (a * a - b * b) ** 1.5
    assert x_sym == pytest.approx(mean / mu, rel=4e-16)
    assert x_lit == pytest.approx(mean / mu, rel=4e-16)
    assert c_sym[0] == pytest.approx(k2 / (2 * mu), rel=1e-15)
    assert c_sym[1] == pytest.approx(k2 / (2 * mu) * math.exp(-0.7 * mu), rel=1e-14)
    assert np.max(np.abs(c_sym - c_lit)) <= 4e-16 * np.max(np.abs(c_lit))


def test_loud01_nig_cgf_vanishes_at_zero_for_symbols():
    from sage.all import SR
    from api.noise import THETA
    K = NormalInverseGaussian('na', 'nb', 'nd').action_cgf()
    assert SR(K.subs({THETA: 0})).is_trivial_zero()


# ── GEN-08 ────────────────────────────────────────────────────────────
_A = np.array([[-1.0, 0.6, 0.0], [-0.4, -0.8, 0.3], [0.2, 0.0, -1.5]])
_BS = [([1, 0, 0], 0.6), ([0, 1, 0.5], 0.2)]


def _ou3(fmt, tag):
    names = ['x', 'y', 'z']
    s = SDE('w6b symc2 ' + tag)
    for n in names:
        s = s.field(n)
    for i, n in enumerate(names):
        s = s.drift(n, ' + '.join(f'({fmt(float(_A[i][j]))})*{names[j]}'
                                  for j in range(3) if _A[i][j] != 0))
    for q, (bv, v) in enumerate(_BS):
        s = s.noise(f'xi{q}', Gaussian(var=fmt(v)),
                    couples={names[i]: bv[i] for i in range(3) if bv[i] != 0})
    return s.build()


def test_gen08_float_three_field_symbolic_c2_matches_lyapunov():
    from api.symbolic_out import tree_covariance
    Q = sum(v * np.outer(bv, bv) for bv, v in _BS)
    Sig = sla.solve_continuous_lyapunov(_A, -Q)
    taus = np.array([0.0, 0.5, 1.3])
    vals = {}
    for tag, fmt in (('float', repr), ('rational', lambda x: str(Fraction(repr(x))))):
        tc = tree_covariance(_ou3(fmt, tag))
        out = np.zeros((3, 3, taus.size))
        for i, a in enumerate('xyz'):
            for j, b in enumerate('xyz'):
                out[i, j] = np.real(np.asarray(
                    tc.modes(f'd{a}1', f'd{b}1').evaluate(taus), dtype=complex))
                ref = [(Sig @ sla.expm(_A.T * t))[i, j] for t in taus]
                assert np.max(np.abs(out[i, j] - ref)) < 5e-15 * np.max(np.abs(Sig))
        vals[tag] = out
    assert np.max(np.abs(vals['float'] - vals['rational'])) < 1e-15


def test_gen08_supplied_inverse_in_wrong_convention_is_refused():
    from sage.all import SR, matrix
    from api.symbolic_out import tree_covariance_from_matrices, SymbolicOutError
    w = SR.var('omega')
    K = matrix(SR, [[SR.I() * w + 1, SR(2)], [SR(0), SR.I() * w + 3]])
    G = K.inverse()
    D = matrix(SR, [[1, 0], [0, 1]])
    tree_covariance_from_matrices(G, D, w, K=K)            # consistent: fine
    with pytest.raises(SymbolicOutError, match='K \\* G != identity'):
        tree_covariance_from_matrices(G.transpose(), D, w, K=K)


# ── GEN-15 ────────────────────────────────────────────────────────────
def test_gen15_complex_hurwitz_test_matches_eigenvalues():
    from sage.all import matrix, QQ
    from api._realization import _routh_hurwitz, _gauss_field
    F = _gauss_field()
    I = F.gen()
    rng = random.Random(11)
    n_hur = n_not = 0
    for _ in range(200):
        n = rng.choice([1, 2, 3, 4])

        def q():
            return QQ(rng.randint(-9, 9)) / QQ(rng.randint(1, 5))
        M = matrix(F, n, n, [q() + q() * I for _ in range(n * n)])
        M = M - QQ(rng.randint(0, 12)) * matrix.identity(F, n)
        ev = np.linalg.eigvals(np.array(
            [[complex(float(x.list()[0]), float(x.list()[1])) for x in row]
             for row in M]))
        if np.min(np.abs(ev.real)) < 1e-9:
            continue
        want = bool(np.all(ev.real < 0))
        assert _routh_hurwitz(M, F) is want
        n_hur += want
        n_not += not want
    assert n_hur > 20 and n_not > 20


# ── GEN-16 ────────────────────────────────────────────────────────────
def _ei(P, Q, with_taug=True):
    b = ModelBuilder(f'w6b ei {P}{Q} {with_taug}').population(P, size=2).population(Q, size=2)
    for X in (P, Q):
        b = (b.physical_field(f'n{X}', population=X).physical_field(f'v{X}', population=X)
             .parameter(f'tau{X}', default=[10.0, 9.0], indexed_by=[X], domain='positive')
             .parameter(f'a{X}', default=[0.3, 0.35], indexed_by=[X], domain='positive')
             .parameter(f'Em{X}', default=[0.7, 0.6], indexed_by=[X])
             .define_function(f'phi{X}', args=[f'v{X}'], expression=f'a{X}[i] * v{X}',
                              population=X))
    for X in (P, Q):
        for Y in (P, Q):
            tk = f'taug{X}{Y}' if with_taug else f'tk{X}{Y}'
            b = (b.parameter(f'w{X}{Y}', default=[[0.2, 0.1], [0.15, 0.25]], indexed_by=[X, Y])
                 .parameter(tk, default=[[2.0, 3.0], [1.5, 2.5]], indexed_by=[X, Y],
                            domain='positive')
                 .define_kernel(f'g{X}{Y}', indexed_by=[X, Y],
                                time_expr=f'(1/{tk}[i,j])*exp(-t/{tk}[i,j])*heaviside(t)'))

    def drive(X, st=''):
        return (f'Em{X}[i] + sum(w{X}{P}[i,j]*g{X}{P}[i,j]*n{P}{st}[j] for j in {P})'
                f' - sum(w{X}{Q}[i,j]*g{X}{Q}[i,j]*n{Q}{st}[j] for j in {Q})')
    b = b.set_action_text(' + '.join(
        f'sum(n{X}t[i]*n{X}[i] - (exp(n{X}t[i])-1)*phi{X}[i](v{X}[i])'
        f' + v{X}t[i]*((tau{X}[i]*Dt + 1)*v{X}[i] - ({drive(X)})) for i in {X})'
        for X in (P, Q)))
    for X in (P, Q):
        b = (b.set_mf_equation(f'v{X}star', drive(X, 'star'))
             .set_mf_equation(f'n{X}star', f'phi{X}[i](v{X}star[i])'))
    return b.build()


def _arrays(m, P):
    from simulations.hawkes_sim_multipop_numba import build_sim_arrays
    r = compute_cumulants(m, k=1, max_ell=0, external_fields=[(f'n{P}', 1)],
                          tau_grid=np.array([0.0]), **RUN)
    f = {p['name']: p['default'] for p in m['parameters'] if p.get('default') is not None}
    return build_sim_arrays(m, f, r['mf_values'])


def test_gen16_coupling_signs_come_from_the_model():
    w = np.array([[0.2, 0.1], [0.15, 0.25]])
    ref = np.block([[w, -w], [w, -w]])               # + excitatory, - inhibitory
    a_ei = _arrays(_ei('E', 'I'), 'E')
    a_ab = _arrays(_ei('A', 'B'), 'A')
    assert np.array_equal(a_ei['W'], ref)
    assert np.array_equal(a_ab['W'], ref)
    assert np.array_equal(a_ab['tau_g'], a_ei['tau_g'])


def test_gen16_missing_time_constant_of_a_filter_raises():
    with pytest.raises(KeyError, match='time constant'):
        _arrays(_ei('E', 'I', with_taug=False), 'E')


def test_gen16_delta_kernel_needs_no_time_constant():
    import daedalus as dd
    from simulations.hawkes_sim_multipop_numba import build_sim_arrays
    m, mod = dd.load_model('single_population_spike_reset_test')
    f = {**dd.fundamental_from_model(m), **(getattr(mod, 'DEFAULT_FUNDAMENTAL', {}) or {})}
    r = compute_cumulants(m, k=1, max_ell=0, external_fields=[('n', 1)],
                          tau_grid=np.array([0.0]), parameters=f, **RUN)
    a = build_sim_arrays(m, f, r['mf_values'])
    assert np.array_equal(a['W'], np.asarray(f['w'], dtype=float))
    assert np.all(a['tau_g'] == 1.0)
