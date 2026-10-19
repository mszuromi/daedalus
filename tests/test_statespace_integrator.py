"""
tests/test_statespace_integrator.py
===================================
State-space subset integrator for Phase J (S2; the default since S3,
``DAEDALUS_PHASE_J_ENGINE=poles`` switches it off): the downset sweep of
``engine/integration/time_domain/statespace.py`` on S1's realization
(``api/_realization.py``).  Inline public models only:

* OU + cubic drift, k = 2, tree + one loop: closed form (Lyapunov + the exact
  first-order Hartree shift) and Phase J;
* OU driven by an Ornstein-Uhlenbeck colour field (auxiliary field) at an
  exact double pole: 50-digit Lyapunov + Hartree oracle;
* OU with an Erlang-8 kernel in its linear drift (rational kernel, class C
  realization, per-entry minimal state counts): 50-digit oracle;
* conductance coupling with an alpha kernel (two rates) against its Markov
  embedding (the kernel form's conductance edges are declined end to end),
  and the embedding at equal kernel rates, where the kernel form is refused;
* a damped oscillator driven by red noise through a multiplicative coupling,
  k = 3 one loop, against Phase J at a generic and at a tied point;
* unit tests of the sweep on hand-made subsets (chain, antichain, constant
  rows, a cycle), the τ-grid prefetch, a finite ``PHASE_J_T_MIN``, the memory
  guard, the hook tag and the flag values.

Runs that are compared bit for bit get a discarded warm-up.
"""
import os
import sys
import types

import numpy as np
import pytest

pytest.importorskip('sage.all')
import mpmath as mp                                                   # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from api import compute_cumulants                                    # noqa: E402
from api import _realization as RZ                                   # noqa: E402
from api.model import TemporalModelBuilder                           # noqa: E402
import engine.integration.time_domain.final_integral as FI           # noqa: E402
import engine.integration.time_domain.statespace as SS               # noqa: E402

ENV = SS.ENGINE_ENV


@pytest.fixture(autouse=True)
def _clean_engine(monkeypatch):
    monkeypatch.delenv(ENV, raising=False)
    SS._clear_caches()
    yield
    SS._clear_caches()


def _cc(model, k, ell, ext, taus, params=None, **kw):
    kw.setdefault('use_cache', False)
    return compute_cumulants(model, k=k, max_ell=ell, external_fields=ext,
                             tau_grid=np.asarray(taus, dtype=float),
                             parameters=params, parallel=False,
                             verbose=False, **kw)


def _ss_counters():
    return {k: v for k, v in FI._RUNTIME_COUNTERS.items()
            if str(k).startswith('ss_')}


# ── models ────────────────────────────────────────────────────────────────

def ou_cubic():
    """x' = -mu x - eps x^3 + noise (variance 2 D)."""
    from api.sde import SDE
    from api.noise import Gaussian
    return (SDE('ss_ou_cubic').parameter('mu', default=1.0)
            .parameter('eps', default=0.02).parameter('D', default=1.0)
            .field('x').drift('x', '-mu*x - eps*x^3')
            .noise('xi', Gaussian(var='2*D'), couples={'x': 1}).build())


def ou_colour(tc):
    """x' = -mu x + eta - eps x^3, eta' = -eta/tc + noise (variance 4D/tc^2)."""
    b = TemporalModelBuilder('ss_ou_colour')
    b.population('P', size=1)
    b.physical_field('x', population='P').physical_field('e', population='P')
    for p, d in (('mu', 1.0), ('eps', 0.02), ('D', 1.0), ('tc', tc)):
        b.parameter(p, default=d, domain='positive')
    b.set_mf_equation('estar', '0')
    b.set_mf_equation('xstar', '0')
    b.set_action_text('sum(xt[i]*((Dt+mu)*x[i] - e[i] + eps*x[i]^3) '
                      '+ et[i]*((Dt + 1/tc)*e[i]) - 2*D/tc^2*et[i]^2 '
                      'for i in P)')
    return b.build()


def ou_erlang(n):
    """x' = -a x + c (g * x) - eps x^3 + noise, g an Erlang-n kernel."""
    from api.sde import SDE
    from api.noise import Gaussian
    return (SDE(f'ss_ou_erlang{n}').parameter('a', default=1.0)
            .parameter('c', default=-2.0).parameter('d', default=1.0,
                                                    domain='positive')
            .parameter('eps', default=0.1).parameter('D', default=0.1)
            .field('x').define_kernel('g', freq_image=f'(1 + I*omega*d/{n})^(-{n})')
            .drift('x', '-a*x + c*Conv(g, x) - eps*x^3')
            .noise('xi', Gaussian(var='2*D'), couples={'x': 1}).build())


_P_ALPHA = dict(a=1.0, b=0.3, h=0.5, tau=0.7, D=0.15)


def _alpha_base(name, tau2):
    from api.theory import TemporalTheoryBuilder
    b = (TemporalTheoryBuilder(name).population('E', size=1)
         .physical_field('x', population='E'))
    for k, v in dict(_P_ALPHA, tau2=tau2).items():
        b = b.parameter(k, default=[v], indexed_by=['E'],
                        **({'domain': 'positive'}
                           if k in ('tau', 'tau2', 'D') else {}))
    return b


def alpha_kernel_model(tau2):
    """x' = -a x + h - b x (g * x) + noise, g the alpha kernel (rates
    1/tau, 1/tau2)."""
    fi = '1/((1 + I*omega*tau[i])*(1 + I*omega*tau2[i]))'
    return (_alpha_base(f'ss_alpha_kernel_{tau2}', tau2)
            .define_kernel('g', indexed_by=['E'], freq_image=fi)
            .set_action_text('sum(xt[i]*((Dt + a[i])*x[i] - h[i] '
                             '+ b[i]*x[i]*Conv(g[i], x[i])) - D[i]*xt[i]^2 '
                             'for i in E)')
            .set_mf_equation('xstar', '(h[i] - b[i]*xstar[i]^2)/a[i]')
            .build())


def alpha_markov_model(tau2):
    """The same model with the kernel as two linear fields u, s."""
    b = (_alpha_base(f'ss_alpha_markov_{tau2}', tau2)
         .physical_field('s', population='E')
         .physical_field('u', population='E'))
    act = ('sum(xt[i]*((Dt + a[i])*x[i] - h[i] + b[i]*x[i]*s[i]) '
           '- D[i]*xt[i]^2 + ut[i]*((tau[i]*Dt + 1)*u[i] - x[i]) '
           '+ st[i]*((tau2[i]*Dt + 1)*s[i] - u[i]) for i in E)')
    return (b.set_action_text(act).set_mf_equation('ustar', 'xstar[i]')
            .set_mf_equation('xstar', '(h[i] - b[i]*xstar[i]^2)/a[i]')
            .set_mf_equation('sstar', 'xstar[i]').build())


def oscillator_red_noise():
    """p' = -lam p + w q + s z (1 + B p), q' = -w p, z' = -r z + noise:
    a damped oscillator driven by red noise with a multiplicative coupling."""
    b = TemporalModelBuilder('ss_oscillator_red_noise')
    b.population('P', size=1)
    for f in ('p', 'q', 'z'):
        b.physical_field(f, population='P')
    for k, v in dict(lam=0.2, w=0.15, s=0.3, r=0.6, B=0.5).items():
        b.parameter(k, default=v, domain='positive')
    b.set_action_text('sum(pt[i]*(Dt*p[i] + lam*p[i] - w*q[i] - s*z[i] '
                      '- s*B*z[i]*p[i]) + qt[i]*(Dt*q[i] + w*p[i]) '
                      '+ zt[i]*(Dt*z[i] + r*z[i]) - r*zt[i]^2 for i in P)')
    b.equation(lhs='(Dt + lam)*p[i]', rhs='w*q[i] + s*z[i] + s*B*z[i]*p[i]',
               population='P')
    b.equation(lhs='Dt*q[i] + w*p[i]', rhs='0', population='P')
    b.equation(lhs='(Dt + r)*z[i]', rhs='0', population='P')
    return b.build()


# ── oracles ───────────────────────────────────────────────────────────────

def _mp_lyap(A, Q):
    n = A.rows
    M = mp.matrix(n * n, n * n)
    rhs = mp.matrix(n * n, 1)
    for i in range(n):
        for j in range(n):
            r = i * n + j
            rhs[r] = -Q[i, j]
            for k in range(n):
                M[r, k * n + j] += A[i, k]
                M[r, i * n + k] += A[j, k]
    s = mp.lu_solve(M, rhs)
    return mp.matrix([[s[i * n + j] for j in range(n)] for i in range(n)])


def _tree_and_hartree(A, Q, eps, taus, i=0):
    """C_ii(τ) of the linear system dX = A X dt + noise (Q) and its exact
    first-order correction for a drift -eps x_i^3 (A_ii -> A_ii - 3 eps <x_i^2>,
    derivative of e^{Aτ} by the Van Loan block)."""
    n = A.rows
    S = _mp_lyap(A, Q)
    dA = mp.matrix(n, n)
    dA[i, i] = -1
    dS = _mp_lyap(A, dA * S + S * dA.T)
    out = []
    for t in taus:
        t = abs(mp.mpf(t))
        Z = mp.matrix(2 * n, 2 * n)
        for a in range(n):
            for b in range(n):
                Z[a, b] = A[a, b] * t
                Z[n + a, n + b] = A[a, b] * t
                Z[a, n + b] = dA[a, b] * t
        EZ = mp.expm(Z)
        E = mp.matrix([[EZ[a, b] for b in range(n)] for a in range(n)])
        dE = mp.matrix([[EZ[a, n + b] for b in range(n)] for a in range(n)])
        out.append(((E * S)[i, i], 3 * eps * S[i, i] * (dE * S + E * dS)[i, i]))
    return out


def _rel(v, ref):
    return float(abs(mp.mpf(complex(v).real) - ref) / abs(ref))


TAUS = [0.0, 0.5, 2.0]
_TAU_PTS = [0.0, 0.5, 2.0]          # compute_cumulants' τ = 0: the exact Itô left limit


# ── end to end ────────────────────────────────────────────────────────────

def test_engine_flag_values():
    # S3: the state-space route is the default; ``poles`` switches it off
    assert SS.engine_requested({}) is True
    assert SS.engine_requested({ENV: 'default'}) is True
    assert SS.engine_requested({ENV: 'poles'}) is False
    assert SS.engine_requested({ENV: ' StateSpace '}) is True
    with pytest.raises(ValueError, match='statespace'):
        SS.engine_requested({ENV: 'fast'})


def test_ou_cubic_closed_form_and_route(monkeypatch):
    mu, eps, D = 1.0, 0.02, 1.0
    monkeypatch.setenv(ENV, 'statespace')
    seen = []
    monkeypatch.setattr(FI, '_SUBSET_HOOK', seen.append)
    FI._reset_runtime_counters()
    res = _cc(ou_cubic(), 2, 1, [('x', 1), ('x', 1)], TAUS)
    c = _ss_counters()
    assert c['ss_subsets_built'] > 0 and c['ss_subsets_declined'] == 0
    assert seen and all(p['path'] == 'ss' and p['branch'] == 'ss'
                        for p in seen)
    for t, v0, v1 in zip(_TAU_PTS, res['C_tau_by_ell'][0],
                         res['C_tau_by_ell'][1]):
        a = abs(t)
        tree = D / mu * np.exp(-mu * a)
        loop = -3 * eps * D / mu * D / mu ** 2 * (1 + mu * a) * np.exp(-mu * a)
        assert abs(v0 - tree) <= 1e-15 * abs(tree)
        assert abs(v1 - loop) <= 1e-15 * abs(loop)


@pytest.mark.parametrize('gap', [0.1, 0.0])
def test_colour_noise_double_pole_vs_oracle(monkeypatch, gap):
    tc = 1.0 / (1.0 + gap)
    monkeypatch.setenv(ENV, 'statespace')
    FI._reset_runtime_counters()
    res = _cc(ou_colour(tc), 2, 1, [('x', 1), ('x', 1)], TAUS)
    c = _ss_counters()
    assert c['ss_evals'] > 0 and c['ss_subsets_declined'] == 0
    if gap == 0.0:
        # a repeated pole: the pole engine would average 7 deformed
        # propagators; the route covers the run, so the deformation is
        # bypassed (S3)
        assert (res.get('pole_clusters') or {}).get('status') == 'bypassed'
        assert res['phase_j_engine']['pole_deformation'] == 'bypassed'
    with mp.workdps(50):
        mu, eps, D, t = (mp.mpf(1), mp.mpf('0.02'), mp.mpf(1), mp.mpf(tc))
        A = mp.matrix([[-mu, 1], [0, -1 / t]])
        Q = mp.matrix([[0, 0], [0, 4 * D / t ** 2]])
        ref = _tree_and_hartree(A, Q, eps, [mp.mpf(x) for x in _TAU_PTS])
    for i, (r0, r1) in enumerate(ref):
        assert _rel(res['C_tau_by_ell'][0][i], r0) <= 2e-15
        assert _rel(res['C_tau_by_ell'][1][i], r1) <= 2e-15


def test_erlang8_kernel_vs_oracle(monkeypatch):
    n, a, c, D, eps = 8, 1.0, -2.0, 0.1, 0.1
    monkeypatch.setenv(ENV, 'statespace')
    FI._reset_runtime_counters()
    res = _cc(ou_erlang(n), 2, 1, [('x', 1), ('x', 1)], TAUS)
    assert _ss_counters()['ss_subsets_declined'] == 0
    with mp.workdps(50):
        N = n + 1
        A = mp.matrix(N, N)                       # (x, chain states), exact
        A[0, 0] = -a
        A[0, n] = c
        for k in range(1, N):
            A[k, k] = -mp.mpf(n)
            A[k, k - 1] = mp.mpf(n)
        Q = mp.matrix(N, N)
        Q[0, 0] = 2 * mp.mpf(D)
        ref = _tree_and_hartree(A, Q, mp.mpf(eps), [mp.mpf(x) for x in _TAU_PTS])
    for i, (r0, r1) in enumerate(ref):
        assert _rel(res['C_tau_by_ell'][0][i], r0) <= 1e-14
        assert _rel(res['C_tau_by_ell'][1][i], r1) <= 1e-14


def test_alpha_conductance_vs_markov_embedding(monkeypatch):
    ext = [('dx', 1), ('dx', 1)]
    # kernel form (rates 1/0.7, 1/0.4): since S3 the conductance kernel
    # edges and dressed lines are routed too (the kernel's own realization,
    # cascaded with G for a dressed line); the pole engine is the reference
    monkeypatch.setenv(ENV, 'poles')
    kern = _cc(alpha_kernel_model(0.4), 2, 1, ext, TAUS)
    monkeypatch.setenv(ENV, 'statespace')
    FI._reset_runtime_counters()
    kern_on = _cc(alpha_kernel_model(0.4), 2, 1, ext, TAUS)
    c = _ss_counters()
    assert c['ss_subsets_declined'] == 0 and c['ss_subsets_built'] > 0
    assert c['ss_kernel_entries'] + c['ss_dressed_entries'] > 0
    for ell in (0, 1):
        # real parts: the pole routes carry ~2e-17 of imaginary rounding on
        # values ~1e-3 here, the state-space route returns real values
        k_, o_ = np.asarray(kern['C_tau_by_ell'][ell]).real, np.asarray(
            kern_on['C_tau_by_ell'][ell]).real
        assert np.max(np.abs(k_ - o_)) <= 1e-12 * np.max(np.abs(k_))
    FI._reset_runtime_counters()
    emb = _cc(alpha_markov_model(0.4), 2, 1, ext, TAUS)
    c = _ss_counters()
    assert c['ss_subsets_built'] > 0 and c['ss_subsets_declined'] == 0
    for ell in (0, 1):
        k_, e_ = np.asarray(kern['C_tau_by_ell'][ell]), np.asarray(
            emb['C_tau_by_ell'][ell])
        assert np.max(np.abs(k_ - e_)) <= 1e-12 * np.max(np.abs(e_))
    # equal kernel rates: main refuses the kernel form (a repeated kernel
    # rate); the embedding (whose poles the feedback through x* splits) runs
    # on the state-space route and is continuous through tau2 = tau
    eq = _cc(alpha_markov_model(0.7), 2, 1, ext, TAUS)
    near = _cc(alpha_markov_model(0.7 * (1 + 1e-9)), 2, 1, ext, TAUS)
    monkeypatch.setenv(ENV, 'poles')
    pj = _cc(alpha_markov_model(0.7), 2, 1, ext, TAUS)   # pole routes
    for ell in (0, 1):
        a_ = np.asarray(eq['C_tau_by_ell'][ell])
        assert np.max(np.abs(a_ - np.asarray(near['C_tau_by_ell'][ell]))) <= (
            1e-8 * np.max(np.abs(a_)))
        assert np.max(np.abs(a_ - np.asarray(pj['C_tau_by_ell'][ell]))) <= (
            1e-11 * np.max(np.abs(a_)))


def test_oscillator_red_noise_k3_one_loop_vs_phase_j(monkeypatch):
    ext = [('dp', 1)] * 3
    pts = [(0.0, 0.4, 1.1), (0.0, -2e-6, -1e-6), (0.0, 0.5, 0.5)]

    def values():
        res = _cc(oscillator_red_noise(), 3, 1, ext, [0.0])
        return {ell: [complex(pj['total_C'](*pt)) for pt in pts]
                for ell, pj in res['phase_j_by_ell'].items()}
    ref = values()
    monkeypatch.setenv(ENV, 'statespace')
    FI._reset_runtime_counters()
    seen, payloads = [], []

    def hook(p):
        payloads.append((p['path'], p['plan'], p['integrand']))
        seen.append(p['ss_subset'])
    monkeypatch.setattr(FI, '_SUBSET_HOOK', hook)
    got = values()
    assert all(pa == 'ss' and pl is None and it is not None
               for pa, pl, it in payloads)
    c = _ss_counters()
    assert c['ss_subsets_built'] > 0 and c['ss_subsets_declined'] == 0
    # per-entry minimal realizations: each edge axis has its own entry's
    # state count (the red-noise entry 1, the oscillator entries 3)
    dims = {e.n for sub in seen for e in sub.edge_ops}
    assert seen and min(dims) == 1 and max(dims) == 3
    assert all(sub.model.real.n_states == 3 for sub in seen)
    for ell in (0, 1):
        for v, r in zip(got[ell], ref[ell]):
            assert abs(v - r) <= 1e-13 * abs(r) + 1e-17


def test_tau_grid_prefetch_matches_per_point(monkeypatch):
    monkeypatch.setenv(ENV, 'statespace')
    taus = np.linspace(0.0, 4.0, 21)
    per_point = _cc(ou_colour(0.5), 2, 1, [('x', 1), ('x', 1)], taus)
    pts = [(0.0, float(t)) for t in taus]
    FI._reset_runtime_counters()
    with SS.tau_grid_prefetch(pts):
        batched = _cc(ou_colour(0.5), 2, 1, [('x', 1), ('x', 1)], taus)
    assert _ss_counters()['ss_prefetch_hits'] > 0
    for ell in (0, 1):
        a_ = np.asarray(per_point['C_tau_by_ell'][ell])
        b_ = np.asarray(batched['C_tau_by_ell'][ell])
        assert np.max(np.abs(a_ - b_)) <= 1e-14 * np.max(np.abs(a_))


def test_flag_off_is_the_pole_engine_bit_for_bit(monkeypatch):
    # S3: ``poles`` is the pole engine; a state-space run in between does
    # not change its numbers
    monkeypatch.setenv(ENV, 'poles')
    _cc(ou_cubic(), 2, 1, [('x', 1), ('x', 1)], TAUS)        # warm-up
    a_ = _cc(ou_cubic(), 2, 1, [('x', 1), ('x', 1)], TAUS)
    monkeypatch.delenv(ENV)
    _cc(ou_cubic(), 2, 1, [('x', 1), ('x', 1)], TAUS)        # the default
    monkeypatch.setenv(ENV, 'poles')
    FI._reset_runtime_counters()
    b_ = _cc(ou_cubic(), 2, 1, [('x', 1), ('x', 1)], TAUS)
    assert not _ss_counters().get('ss_subsets_built')
    for ell in (0, 1):
        assert np.array_equal(np.asarray(a_['C_tau_by_ell'][ell]),
                              np.asarray(b_['C_tau_by_ell'][ell]))


def test_finite_t_min_tree_closed_form_and_phase_j(monkeypatch):
    t_min = -1.5
    monkeypatch.setattr(FI, 'PHASE_J_T_MIN', t_min)

    def run():
        res = _cc(ou_cubic(), 2, 1, [('x', 1), ('x', 1)], [0.0])
        pts = [(0.0, 0.3), (0.0, -0.8), (0.0, 1.0)]
        return pts, {ell: [complex(pj['total_C'](*pt)) for pt in pts]
                     for ell, pj in res['phase_j_by_ell'].items()}
    pts, ref = run()
    monkeypatch.setenv(ENV, 'statespace')
    FI._reset_runtime_counters()
    _pts, got = run()
    assert _ss_counters()['ss_evals'] > 0
    mu, D = 1.0, 1.0
    for (t1, t2), v in zip(pts, got[0]):
        exact = D / mu * (np.exp(-mu * abs(t1 - t2))
                          - np.exp(-mu * (t1 + t2 - 2 * t_min)))
        assert abs(v - exact) <= 1e-15 * abs(exact) + 1e-16
    for v, r in zip(got[1], ref[1]):
        assert abs(v - r) <= 1e-12 * abs(r) + 1e-17


# ── unit tests on hand-made subsets ───────────────────────────────────────

def _fake_model(rates):
    """A model whose entry (i, i) is the scalar e^{-rate_i t}."""
    ents = {(i, i): RZ.EntryRealization(
        a=np.array([[-float(r)]]), b=np.array([1.0]), c=np.array([1.0]),
        d=0.0, n_states=1) for i, r in enumerate(rates)}
    real = types.SimpleNamespace(A=np.zeros((1, 1)), entries=ents)
    return SS._ModelOps(real)


def _quad2(f, a, b, c, d):
    with mp.workdps(30):
        return complex(mp.quad(lambda x: mp.quad(lambda y: f(x, y), [c(x), d(x)]),
                               [a, b]))


def test_two_vertex_poset_vs_quadrature():
    # u -> v -> external 0, u -> external tau:  rows (a_int, a_ext, c0)
    # e0: s_v - s_u > 0 (rate 1), e1: -s_v + 0 > 0 (rate 2),
    # e2: -s_u + tau > 0 (rate 3)
    model = _fake_model([1.0, 2.0, 3.0])
    ents = [model.entry(0, 0), model.entry(1, 1), model.entry(2, 2)]
    rows = [([-1.0, 1.0], [0.0], 0.0), ([0.0, -1.0], [0.0], 0.0),
            ([-1.0, 0.0], [1.0], 0.0)]
    sub = SS.SubsetSS(model, ents, rows, 2, 1)
    for tau in (0.7, -0.4, 0.0):
        val = sub.evaluate([tau])
        lo = -mp.inf
        up_u = min(0.0, tau)
        ref = _quad2(lambda su, sv: mp.exp(-(sv - su)) * mp.exp(2 * sv)
                     * mp.exp(-3 * (tau - su)), lo, up_u,
                     lambda su: su, lambda su: 0)
        assert abs(val - ref) <= 1e-15 * abs(ref)


def test_antichain_and_constant_rows():
    # two independent vertices before external 0, one constant row
    # tau - 0 > 0 (rate 4): factor e^{-4 tau} when tau > 0, empty otherwise
    model = _fake_model([1.0, 2.0, 4.0])
    ents = [model.entry(0, 0), model.entry(1, 1), model.entry(2, 2)]
    rows = [([-1.0, 0.0], [0.0], 0.0), ([0.0, -1.0], [1.0], 0.0),
            ([0.0, 0.0], [1.0], 0.0)]
    sub = SS.SubsetSS(model, ents, rows, 2, 1)
    tau = 0.6
    ref = (1.0 / 1.0) * (1.0 / 2.0) * np.exp(-4 * tau)
    assert abs(sub.evaluate([tau]) - ref) <= 1e-15 * ref
    assert sub.evaluate([-0.3]) == 0.0
    # tau == 0 exactly without a tie context: Θ(0) = 0
    assert sub.evaluate([0.0]) == 0.0


def test_cycle_is_empty_and_vertices_need_a_later_time():
    model = _fake_model([1.0, 1.0, 1.0])
    e = model.entry(0, 0)
    cyc = [([-1.0, 1.0], [0.0], 0.0), ([1.0, -1.0], [0.0], 0.0),
           ([0.0, -1.0], [0.0], 0.0)]
    assert SS.SubsetSS(model, [e, e, e], cyc, 2, 1).evaluate([0.5]) == 0.0
    with pytest.raises(SS.SSIneligible, match='vertex_after_last_time'):
        SS.SubsetSS(model, [e], [([1.0], [0.0], 0.0)], 1, 1)


def test_memory_guard_raises_before_allocating():
    model = _fake_model([1.0, 2.0, 3.0])
    ents = [model.entry(0, 0), model.entry(1, 1), model.entry(2, 2)]
    rows = [([-1.0, 1.0], [0.0], 0.0), ([0.0, -1.0], [0.0], 0.0),
            ([-1.0, 0.0], [1.0], 0.0)]
    sub = SS.SubsetSS(model, ents, rows, 2, 1)
    with FI.phase_j_memory_scope(budget=1.0, what='test'):
        with pytest.raises(FI.PhaseJResourceError, match='state-space'):
            SS._charge(sub)
    assert sub._stat is None
    with FI.phase_j_memory_scope(budget=1e6, what='test') as scope:
        SS._charge(sub)
    assert scope.used == sub.state_bytes() > 0


def test_realization_cache_key_and_per_momentum_calls():
    from sage.all import RR
    a = RR(0.3)
    b = a + RR(2) ** -54 * 3                    # same 15-digit repr
    assert repr(a) == repr(b)
    assert SS._np_key({'x': a}) != SS._np_key({'x': b})
    # the spatial certification's per-momentum temporal propagators (the
    # Laplacian bound to -q^2) stay on the pole routes
    ops, why = SS.realization_for({'K_ft': object(), 'omega': None},
                                  {'Laplacian': -0.49})
    assert ops is None and why == 'realization_spatial_per_q'


def test_stiff_segment_takes_the_triangular_route():
    # one vertex v in (0, tau): 0 -> v through a stiff 2-state entry (rates
    # near -1e4 and -1e-2), v -> tau through e^{-t}; the Taylor action would
    # need ~1e4 steps (error ~ steps x eps), the triangular route is exact on
    # the diagonal
    a = np.array([[-1e4, 5.0], [0.7, -1e-2]])
    b, c = np.array([1.0, 1.0]), np.array([1.0, 2.0])
    ents = {(0, 0): RZ.EntryRealization(a=a, b=b, c=c, d=0.0, n_states=2),
            (1, 1): RZ.EntryRealization(a=np.array([[-1.0]]),
                                        b=np.array([1.0]), c=np.array([1.0]),
                                        d=0.0, n_states=1)}
    model = SS._ModelOps(types.SimpleNamespace(A=np.zeros((1, 1)),
                                               entries=ents))
    rows = [([1.0], [0.0], 0.0), ([-1.0], [1.0], 0.0)]
    sub = SS.SubsetSS(model, [model.entry(0, 0), model.entry(1, 1)], rows, 1, 1)
    FI._reset_runtime_counters()
    for tau in (1.0, 50.0):
        val = sub.evaluate([tau])
        with mp.workdps(50):
            A = mp.matrix(a.tolist())
            E, ER = mp.eig(A)
            EL = mp.inverse(ER)
            ref = mp.mpf(0)
            for k_ in range(2):
                r = (mp.matrix([c.tolist()]) * ER[:, k_])[0] * (
                    EL[k_, :] * mp.matrix(b.tolist()))[0]
                lam = E[k_]
                ref += r * mp.exp(-tau) * (mp.exp((lam + 1) * tau) - 1) / (lam + 1)
        assert abs(val - complex(ref).real) <= 1e-14 * abs(complex(ref))
    assert SS._counters()['ss_segments_triangular'] >= 2


def test_batch_rejects_non_finite_times():
    model = _fake_model([1.0, 2.0, 3.0])
    ents = [model.entry(0, 0), model.entry(1, 1), model.entry(2, 2)]
    rows = [([-1.0, 1.0], [0.0], 0.0), ([0.0, -1.0], [0.0], 0.0),
            ([-1.0, 0.0], [1.0], 0.0)]
    sub = SS.SubsetSS(model, ents, rows, 2, 1)
    with pytest.raises(ValueError, match='finite'):
        sub.evaluate_many([([3.0], None), ([float('nan')], None),
                           ([1.0], None)])
    with pytest.raises(ValueError, match='finite'):
        sub.evaluate([float('inf')])
