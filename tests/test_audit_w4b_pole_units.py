"""Audit W4b: unit invariance of the pole engine (``DAEDALUS_PHASE_J_ENGINE
=poles``, the supported backup of the state-space default).

Every threshold of the pole route that changed an answer under an exact
change of units is now relative to the problem's own scale.  Each test runs a
time-rescaled TWIN (all rates x S, times / S) or a field-rescaled one and
asks for the same poles, residues and rescaled values -- and, end to end, the
state-space engine's numbers or a closed form (Lyapunov, mpmath).

ACC-60  ``_extract_exp_mode(s)``: fixed probe times underflowed for a fast
        kernel, which was fitted as ONE real mode (one loop 39 % off); a
        cos-written kernel is now split into its complex modes.
ACC-61  numerical residue tiers: the clip at 1e-12 of the LARGEST entry
        zeroed a genuine small residue (``propagator['C_mats']`` on every
        engine), and some exact-tier failures entered them silently.
ACC-62  numerical pole tiers: absolute retarded floor, dedup radii, Newton
        exits and probe points lost a pole in slow units (RED-08: one Newton).
ACC-63  poset shared lower bound: an absolute 1e-9 merged distinct lowers.
ACC-64  DBM / DD conditioning floor: units [integrand] instead of
        [integrand][time]^m, and the caller's scale lacked e^{Re E}.
ACC-65  retarded-pole tests (exact tier, cluster analysis): absolute 1e-9
        decay floor dropped a stable slow mode.
"""
from __future__ import annotations

import math
import os
import sys
import warnings

import mpmath as mp
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from sage.all import SR, I, cos, exp, matrix, var                  # noqa: E402

import engine.integration.time_domain.final_integral as FI         # noqa: E402
from engine.integration.time_domain import dbm_integral as DBM      # noqa: E402
from engine.integration.time_domain.final_integral import EdgeModeSum  # noqa: E402
import engine.integration.time_domain.statespace as SS              # noqa: E402
from api import compute_cumulants                                   # noqa: E402
from api import _propagator as PR                                   # noqa: E402
from api.noise import Gaussian                                      # noqa: E402
from api.sde import SDE                                             # noqa: E402

SCALES = [1e-6, 1e-3, 1.0, 1e3, 1e6]
RUN = dict(parallel=False, verbose=False, use_cache=False)


@pytest.fixture
def engine(monkeypatch):
    """``engine('poles')`` / ``engine(None)`` (the state-space default)."""
    def set_engine(name):
        if name is None:
            monkeypatch.delenv(SS.ENGINE_ENV, raising=False)
        else:
            monkeypatch.setenv(SS.ENGINE_ENV, name)
        SS._clear_caches()
    yield set_engine
    SS._clear_caches()


def _quiet_run(model, **kw):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        r = compute_cumulants(model, **RUN, **kw)
    r['_warnings'] = caught
    return r


def _lyapunov_C(A, Q, i, j, taus):
    """Stationary C_ij(tau) = (S e^{A^T tau})_ij of dx = A x dt + noise Q (mpmath)."""
    with mp.workdps(50):
        A = mp.matrix(A)
        Q = mp.matrix(Q)
        n = A.rows
        K = mp.zeros(n * n, n * n)
        for r in range(n):
            for s in range(n):
                for u in range(n):
                    for v in range(n):
                        K[n * r + s, n * u + v] = (A[r, u] * (s == v)
                                                   + (r == u) * A[s, v])
        sv = mp.lu_solve(K, mp.matrix([-Q[r, s] for r in range(n)
                                       for s in range(n)]))
        S = mp.matrix([[sv[n * r + s] for s in range(n)] for r in range(n)])
        return np.array([float((S * mp.expm(A.T * mp.mpf(float(t))))[i, j])
                         for t in taus])


# ── ACC-60: _extract_exp_mode(s) on the kernel's own time scale ──────────────

@pytest.mark.parametrize('S', SCALES)
def test_exp_modes_are_unit_invariant(S):
    t = SR.var('t')
    r = 2.0 * S
    got = FI._extract_exp_modes(r * exp(-r * t) * cos(r * t), t)
    assert got is not None and len(got) == 2     # never ONE real mode
    for C, lam in got:
        assert abs(C - S) <= 1e-13 * S
        assert abs(abs(lam.imag) - r) <= 1e-13 * r and abs(lam.real + r) <= 1e-13 * r
    two = sorted(FI._extract_exp_modes(r * exp(-r * t) + 3 * r * exp(-5 * r * t), t),
                 key=lambda m: -m[1].real)
    assert [abs(C) for C, _ in two] == pytest.approx([r, 3 * r], rel=1e-14)
    assert [lam.real for _, lam in two] == pytest.approx([-r, -5 * r], rel=1e-14)
    # not exponential sums, in every unit
    assert FI._extract_exp_modes(r * t * exp(-r * t), t) is None
    assert FI._extract_exp_modes(exp(-(r * t) ** 2), t) is None


def test_single_mode_fit_probes_its_own_time_scale():
    """The one-term fit: probes at 0.37, 1.9 and 4.3 underflowed for a rate
    2000 kernel (nothing compared, the fit accepted)."""
    t = SR.var('t')
    for S in SCALES:
        assert FI._extract_exp_mode(2000 * S * exp(-2000 * S * t) * cos(2000 * S * t), t) is None
        assert FI._extract_exp_mode(cos(S * t), t) is None
        C, lam = FI._extract_exp_mode(3 * exp(-7 * S * t), t)
        assert C == pytest.approx(3) and lam == pytest.approx(-7 * S, rel=1e-14)


def _cos_kernel_model(S, form):
    te = ('r*exp(-r*t)*cos(r*t)*heaviside(t)' if form == 'cos' else
          '(r/2)*(exp((-r + I*r)*t) + exp((-r - I*r)*t))*heaviside(t)')
    return (SDE(f'w4b damped kernel {form} {S:g}').parameter('a', default=1.0 * S)
            .parameter('b', default=1.0 * S).parameter('c', default=0.5 * S)
            .parameter('ee', default=0.3 * S).parameter('r', default=2.0 * S, domain='positive')
            .parameter('D', default=0.1 * S).field('x')
            .define_kernel('h', time_expr=te)
            .drift('x', '-a*x + b - ee*x*x - c*x*Conv(h, x)')
            .noise('xi', Gaussian(var='2*D'), couples={'x': 1}).build())


def _one_loop(model, S):
    r = _quiet_run(model, k=2, max_ell=1, external_fields=[('x', 1), ('x', 1)],
                   tau_grid=np.array([0.0, 0.5, 2.0]) / S)
    return np.real(np.asarray(r['C_tau_by_ell'][1])), r


def test_damped_cos_kernel_one_loop_twins_agree_with_state_space(engine):
    """End to end (k = 2, one loop, a Conv vertex with r e^{-rt} cos(rt)):
    the pole route in fast units (rates x 1000) gave 39 % off, silently; in
    unit time it refused the kernel.  Now both units give the state-space
    value, and so does the same kernel written as exponentials."""
    engine(None)
    ref, _ = _one_loop(_cos_kernel_model(1.0, 'cos'), 1.0)
    assert np.all(np.abs(ref) > 1e-4)
    engine('poles')
    for S in (1.0, 1e3):
        for form in ('cos', 'exp'):
            got, r = _one_loop(_cos_kernel_model(S, form), S)
            assert not any('could not evaluate' in str(w.message) for w in r['_warnings'])
            assert np.allclose(got, ref, rtol=1e-11, atol=0), (S, form, got, ref)


# ── ACC-61 / ACC-62: the numerical tiers ─────────────────────────────────────

def _two_field(a, b, c, D, Dz, name='w4b two field'):
    return (SDE(name).parameter('a', default=a).parameter('b', default=b)
            .parameter('c', default=c).parameter('D', default=D)
            .parameter('Dz', default=Dz).field('x').field('z')
            .drift('x', '-a*x + c*z').drift('z', '-b*z')
            .noise('xi', Gaussian(var='2*D'), couples={'x': 1})
            .noise('zeta', Gaussian(var='2*Dz'), couples={'z': 1}).build())


@pytest.fixture
def no_exact_tier(monkeypatch):
    monkeypatch.setattr(PR, '_compute_residues_via_polynomial_fracfield',
                        lambda *a, **k: (None, None, None))


TAU1 = np.array([0.0, 0.5, 1.0, 2.0])


def test_numerical_tier_keeps_a_small_coupling_residue(engine, monkeypatch):
    """x' = -x + c z, z' = -z/2 with c = 1e-14 (x in units 1e14 times
    smaller than z): C_xz ~ 1e-14 was clipped to 0 at 1e-12 of the largest
    residue entry, on the pole engine's values and on every engine's
    ``propagator['C_mats']``.  The numerical tier's residues now match the
    exact tier's entry by entry, and the field-rescaled twin c = 1 agrees."""
    engine('poles')
    out = {}
    for c in (1e-14, 1.0):
        def run():
            return _quiet_run(_two_field(1.0, 0.5, c, 1e-28, 1.0), k=2, max_ell=0,
                              external_fields=[('x', 1), ('z', 1)], tau_grid=TAU1)
        exact = run()['propagator']
        with monkeypatch.context() as mpc:
            mpc.setattr(PR, '_compute_residues_via_polynomial_fracfield',
                        lambda *a, **k: (None, None, None))
            r = run()
        ref = _lyapunov_C([[-1.0, c], [0.0, -0.5]], [[2e-28, 0], [0, 2.0]], 0, 1, TAU1)
        got = np.array([complex(v).real for v in r['C_tau']])
        assert np.allclose(got, ref, rtol=1e-12, atol=0), (c, got, ref)
        P, E = r['propagator'], exact
        assert np.allclose([complex(p) for p in P['pole_vals']],
                           [complex(p) for p in E['pole_vals']], rtol=1e-13, atol=0)
        for Ck, Ek in zip(P['C_mats'], E['C_mats']):
            for i in range(Ek.nrows()):
                for j in range(Ek.ncols()):
                    e, v = complex(Ek[i, j]), complex(Ck[i, j])
                    assert (v == 0) if e == 0 else abs(v - e) <= 1e-12 * abs(e), (c, i, j, v, e)
        out[c] = got / c
    assert np.allclose(out[1e-14], out[1.0], rtol=1e-12, atol=0)


@pytest.mark.parametrize('s', [1e-6, 1e-4, 1.0, 1e4, 1e6])
def test_numerical_tier_time_twins(engine, no_exact_tier, s):
    """x' = -a x + c z, z' = -b z with a = s, b = 1.05 s: at s = 1e-4 the
    absolute 1e-5 pre-Newton dedup merged the two poles (C_xx 236x off,
    silently).  Every s keeps both poles at i s and 1.05 i s, and the
    rescaled C_xx(tau / s) is the s = 1 one (and the closed form)."""
    engine('poles')
    tau = TAU1 / s
    r = _quiet_run(_two_field(s, 1.05 * s, s, s, s), k=2, max_ell=0,
                   external_fields=[('x', 1), ('x', 1)], tau_grid=tau)
    pv = sorted(complex(p).imag / s for p in r['propagator']['pole_vals'])
    assert pv == pytest.approx([1.0, 1.05], rel=1e-12)
    got = np.array([complex(v).real for v in r['C_tau']])
    ref = _lyapunov_C([[-1.0, 1.0], [0.0, -1.05]], [[2.0, 0], [0, 2.0]], 0, 0, TAU1)
    assert np.allclose(got, ref, rtol=1e-10, atol=0), (s, got, ref)


def test_silent_exact_tier_failure_warns(monkeypatch, engine):
    """An exact-tier failure inside its inverse (an exception, the time
    budget) used to enter the numerical tier silently: now a
    PoleResidueWarning names it, and the numbers are still right."""
    def boom(*a, **k):
        raise RuntimeError('forced inverse failure')
    monkeypatch.setattr(PR, '_run_with_timeout', boom)
    engine('poles')
    r = _quiet_run(_two_field(1.0, 0.7, 1.0, 0.1, 0.1), k=2, max_ell=0,
                   external_fields=[('x', 1), ('x', 1)], tau_grid=np.array([0.5]))
    msgs = [str(w.message) for w in r['_warnings']
            if issubclass(w.category, PR.PoleResidueWarning)]
    assert any('returned no residues' in m and 'forced inverse failure' in m for m in msgs)
    ref = _lyapunov_C([[-1.0, 1.0], [0.0, -0.7]], [[0.2, 0], [0, 0.2]], 0, 0, [0.5])
    assert abs(complex(r['C_tau'][0]).real - ref[0]) <= 1e-10 * abs(ref[0])


@pytest.mark.parametrize('scale', [1e-8, 1.0, 1e8])
def test_newton_and_singularity_tests_are_scale_free(scale):
    roots = [scale * (0.3 + 1j), scale * (-2 + 0.5j), scale * (0.3 + 1.0001j)]
    det = lambda w: (w - roots[0]) * (w - roots[1]) * (w - roots[2]) / scale ** 3
    for r0 in roots:
        r = PR._newton_det(det, r0 * (1 + 1e-6), scale=scale)
        assert abs(r - r0) <= 1e-13 * abs(r0)
        assert PR._singular_at(det, r, scale)
    assert not PR._singular_at(det, scale * (1 + 1j), scale)
    kept = PR._refine_retarded_poles([r * (1 + 1e-7) for r in roots] + [roots[0]],
                                     det, PR._omega_scale(roots))
    assert len(kept) == 3
    assert sorted(abs(k - q) / abs(q) < 1e-12 for k in kept for q in roots).count(True) == 3


def test_structural_zeros_are_exact():
    # K upper triangular in (x, z): G_zx is a structural zero, G_xz is not
    nz = [[True, True], [False, True]]
    Z = PR._structural_zero_entries(nz)
    assert Z == [[False, False], [True, False]]
    assert PR._structural_zero_entries([[True, False], [False, True]]) == [
        [False, True], [True, False]]


# ── ACC-63: the poset's shared lower bound ───────────────────────────────────

_ROWS63 = [((1, 0, 0), (0, 0), 0), ((0, 1, 0), (-1, 0), 0), ((-1, 0, 1), (0, 0), 0),
           ((0, -1, 1), (0, 0), 0), ((0, 0, -1), (0, 1), 0)]


def _ref63(r, t1, tau):
    r, t1, tau = mp.mpf(r), mp.mpf(t1), mp.mpf(tau)
    with mp.workdps(30):
        return mp.quad(lambda s2: s2 * (s2 - t1) * mp.e ** (-r * (s2 + tau - t1)), [t1, tau])


@pytest.mark.parametrize('S', [1.0, 1e3, 1e9])
def test_poset_lower_bounds_are_compared_relatively(S):
    """The m = 3 region 0 < s0, t1 < s1, s0 < s2, s1 < s2 < tau with five
    e^{-r .} edges: the lowers 0 and t1 differ, so the single-L chain
    simplex must not take it.  At S = 1e9 they were within the absolute
    1e-9 and merged (36.6 % low).  Every unit now bails, and the subset's
    exact route gives the reference (x S^3)."""
    r, t1, tau = 1.0 * S, 0.5 / S, 2.0 / S
    ems = [EdgeModeSum(ri=0, pi=0, delta_coeff=0j, modes=((1 + 0j, complex(-r)),))
           for _ in range(5)]
    FI._reset_runtime_counters()
    assert FI._integrate_nd_polytope_poset_modesum(ems, 1.0 + 0j, _ROWS63, [t1, tau], 3) is None
    assert FI._RUNTIME_COUNTERS['poset_consistent_lower_failed'] == 1
    v = FI._integrate_subset_dbm(ems, 1.0 + 0j, _ROWS63, [t1, tau], 3)
    ref = complex(_ref63(1.0, 0.5, 2.0))
    assert v is not None and abs(v * S ** 3 - ref) <= 1e-10 * abs(ref)
    # an exact tie (t1 = 0) is still one lower, in every unit
    p = FI._integrate_nd_polytope_poset_modesum(ems, 1.0 + 0j, _ROWS63, [0.0, tau], 3)
    ref0 = complex(_ref63(1.0, 0.0, 2.0))
    assert p is not None and abs(p * S ** 3 - ref0) <= 1e-12 * abs(ref0)


# ── ACC-64: the DBM / DD conditioning floor ──────────────────────────────────

def _chain3(T):
    return [((1., 0., 0.), 0.), ((-1., 1., 0.), 0.), ((0., -1., 1.), 0.), ((0., 0., -1.), T)]


@pytest.mark.parametrize('eps', [1e-8, 1e-10])
def test_dbm_verdict_is_time_unit_invariant(eps):
    """0 < s0 < s1 < s2 < 2/lam with beta = lam (-1, 1 + eps, -1): the same
    integral (x lam^-3) in every unit.  The floor COND_ATOL * sum|C| (units
    of the integrand) turned 'ill_conditioned' into 'ok' at lam = 1e3, 1e5."""
    out = []
    for lam in (1e-3, 1.0, 1e3, 1e5):
        seeds = [(1.0 + 0j, 0j, (-lam, lam * (1 + eps), -lam))]
        res = DBM.integrate_exp_sum(_chain3(2.0 / lam), 3, seeds, 200.0 / lam)
        out.append((res.status, res.error_ratio, res.value * lam ** 3))
    assert {o[0] for o in out} == {DBM.STATUS_ILL_CONDITIONED}
    # the same ratio (to the rounding of the gap eps itself)
    assert all(abs(o[1] - out[0][1]) <= 1e-4 * out[0][1] for o in out)
    assert DBM.integral_scale([(2.0, -1.0, (-lam, 3 * lam))], 2) == pytest.approx(
        2 * math.exp(-1.0) / (3 * lam) ** 2)


def test_dbm_tail_keeps_its_verdict_and_mp_retry():
    """The same ill-conditioned chain times e^-30 (a correlator tail): with
    the caller's Σ|C| it passed as 'ok' (rel err 1.2e-6) and skipped the mp
    retry; the integral's own scale keeps 'ill_conditioned'."""
    beta = (-1.0, 1.0 + 1e-10, -1.0)
    for E in (0.0, -30.0):
        seeds = [(1.0 + 0j, complex(E), beta)]
        res = DBM.integrate_exp_sum(_chain3(2.0), 3, seeds, 200.0,
                                    scale=DBM.integral_scale(seeds, 3))
        assert res.status == DBM.STATUS_ILL_CONDITIONED
        mp_res = DBM.integrate_exp_sum_mp(_chain3(2.0), 3, seeds, 200.0)
        assert mp_res.status == DBM.STATUS_OK


def test_dd_accounting_is_time_unit_invariant(monkeypatch):
    """The DD chain route's error accounting (counters read by G6), half
    line s < t: two slow modes with nearly opposite residues cancel next to
    a fast one.  Flagged with the same ratio in every time unit (the floor
    had units of the integrand, so the ratio scaled with the unit)."""
    monkeypatch.setattr(FI, 'USE_PHASE_J_DD_KERNELS', True)
    monkeypatch.setattr(FI, 'SEMIINF_CHAIN', True)
    monkeypatch.setattr(FI, 'UNBOUNDED_2D', 'exact')
    monkeypatch.setattr(FI, 'PHASE_J_T_MIN', -math.inf)
    half_line = [((-1.0,), (1.0,), 0.0)]
    ratios = []
    for S in (1e-3, 1.0, 1e3):
        modes = ((1.0 + 0j, complex(-1e-7 * S)),
                 (-1.0 + 0j, complex(-1e-7 * S * (1 + 1e-9))),
                 (1.0 + 0j, complex(-100.0 * S)))
        ems = [EdgeModeSum(ri=0, pi=0, delta_coeff=0j, modes=modes)]
        FI._reset_runtime_counters()
        v = FI._integrate_1d_polytope_modesum(ems, 1.0, half_line, [0.4 / S])
        assert v is not None
        assert FI._RUNTIME_COUNTERS['dd_ill_conditioned'] == 1
        ratios.append(FI._RUNTIME_COUNTERS['dd_err_ratio_max'])
    assert ratios[0] > 1e3
    assert max(ratios) <= (1 + 1e-6) * min(ratios)


# ── ACC-65: a stable slow mode is a retarded pole in every unit ──────────────

def _two_ou(a, b, Dx, Dy):
    return (SDE('w4b two ou').parameter('a', default=a).parameter('b', default=b)
            .parameter('Dx', default=Dx).parameter('Dy', default=Dy)
            .field('x').field('y').drift('x', '-a*x').drift('y', '-b*y')
            .noise('xi', Gaussian(var='2*Dx'), couples={'x': 1})
            .noise('eta', Gaussian(var='2*Dy'), couples={'y': 1}).build())


@pytest.mark.parametrize('S', [1.0, 1e3])
def test_slow_mode_kept_on_the_pole_engine(engine, S):
    """y' = -b y with b = 5e-10 S next to x' = -S x: Var y = 0.2.  The G2
    gate refuses the marginal-looking mode by default; downgraded, the pole
    engine dropped it (C_yy = 0) below the absolute 1e-9 floor, while the
    state-space engine and the closed form give 0.2."""
    vals = {}
    for eng in ('poles', None):
        engine(eng)
        r = _quiet_run(_two_ou(1.0 * S, 5e-10 * S, 1.0 * S, 1e-10 * S), k=2, max_ell=0,
                       external_fields=[('y', 1), ('y', 1)], tau_grid=np.array([0.0, 1.0]) / S,
                       policy={'downgrade': ['G2_MARGINAL_MODE']})
        vals[eng] = np.real(np.asarray(r['C_tau_by_ell'][0], complex))
    ref = 0.2 * np.exp(-5e-10 * np.array([0.0, 1.0]))
    assert np.allclose(vals['poles'], ref, rtol=1e-12, atol=0)
    assert np.allclose(vals['poles'], vals[None], rtol=1e-12, atol=0)


@pytest.mark.parametrize('S', [1e-3, 1.0, 1e6])
def test_cluster_analysis_keeps_a_slow_retarded_root(S):
    """The exact cluster analysis (a near-degenerate pair forces it) used
    Re lambda < -1e-9: the slow root 5e-10 S was not retarded at S = 1 and
    was at S = 1e6.  Now it is a pole in every unit."""
    om = var('omega')
    rates = [S * SR(1), S * (1 + SR(1) / 10 ** 9), S * SR(5) / 10 ** 10]
    K = matrix(SR, [[I * om + rates[0], 0, 0], [0, I * om + rates[1], 0],
                    [0, 0, I * om + rates[2]]])
    cl = {}
    pv, Cm, _ = PR._compute_residues_via_polynomial_fracfield(K, om, {}, 3, cluster_out=cl)
    assert 'analysis' in cl
    got = sorted(complex(p).imag / S for p in pv)
    assert got == pytest.approx([5e-10, 1.0, 1.0 + 1e-9], rel=1e-12)
