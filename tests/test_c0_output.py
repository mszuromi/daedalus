"""
tests/test_c0_output.py
=======================
C0 output (state-space stage S4, ``api/me_output.py``): the smooth part of a
computed cumulant as matrix-exponential triples ``C(τ) = c e^{G|τ|} x0`` per
loop order and side, the mode list derived from them, kernel pairings and
the spectrum by resolvents, and the k >= 3 slices.

* tree level: the triples and their modes against item 23's symbolic modes
  (``tree_modes``) evaluated at numbers: OU, underdamped OU, an Erlang-8
  kernel, the cubic-denominator feedback chain;
* loop level: the triples against ``C_tau_by_ell`` on a 201-point grid (OU
  quartic to two loops, colour-noise OU at the exact double pole, a damped
  oscillator driven by red noise);
* ``integrate_against`` against quadrature of the pipeline's ``C(τ)``
  (exponential, polynomial × exponential, rational kernels; a kernel with a
  jump at 0 and an explicit ``contact``), the spectrum against Feature B's
  quadrature and the exact linear-Hawkes spectrum;
* k = 3 slices against ``dd.run``; never-silent fallbacks; ``dd.`` exports.

Inline public models or tracked ones only.
"""
import importlib.util
import os
import sys
import warnings

import numpy as np
import pytest

pytest.importorskip('sage.all')

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, '..'))

import daedalus as dd                                                # noqa: E402
from api import compute_cumulants                                    # noqa: E402
from api import me_output as ME                                      # noqa: E402
from api import symbolic_out as so                                   # noqa: E402
from api.cumulant import Cumulant                                    # noqa: E402
from api.noise import Gaussian, Poisson                              # noqa: E402
from api.sde import SDE                                              # noqa: E402
import engine.integration.time_domain.statespace as SS               # noqa: E402

TAUS = np.linspace(-5.0, 5.0, 201)
TT = TAUS.copy()     # C_tau's τ = 0 point: the exact Itô left limit (the triple's at_zero='-')


def _ss_models():
    spec = importlib.util.spec_from_file_location(
        'ss_models_c0', os.path.join(HERE, 'test_statespace_integrator.py'))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


SSM = _ss_models()


@pytest.fixture(autouse=True)
def _default_engine(monkeypatch):
    monkeypatch.delenv(SS.ENGINE_ENV, raising=False)


def _cc(model, ext, ell, params=None, taus=TAUS, **kw):
    return compute_cumulants(model, k=2, max_ell=ell, external_fields=ext,
                             parameters=params, tau_grid=taus,
                             parallel=False, verbose=False, use_cache=False,
                             **kw)


def _rel(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return float(np.max(np.abs(a - b)) / max(np.max(np.abs(b)), 1e-300))


# ── models (C2's tree oracles) ──────────────────────────────────────────────

def _rotation():
    return (SDE('c0 underdamped OU').parameter('g', default=0.2)
            .parameter('w0', default=1.0).parameter('D', default=1.0)
            .field('x').field('y')
            .drift('x', '-g*x - w0*y').drift('y', 'w0*x - g*y')
            .noise('xi', Gaussian(var='2*D'), couples={'x': 1})).build()


def _feedback_chain():
    return (SDE('c0 feedback filter chain').parameter('a', default=1.0)
            .parameter('k', default=2.0).parameter('D', default=1.0)
            .field('x').field('y').field('z')
            .drift('x', '-a*x - k*z').drift('y', '-a*y + x')
            .drift('z', '-a*z + y')
            .noise('xi', Gaussian(var='2*D'), couples={'x': 1})).build()


def _hawkes_1pop():
    return (SDE('c0 linear Hawkes').parameter('Em', default=0.5)
            .parameter('tau', default=2.0).parameter('a', default=1.0)
            .parameter('w', default=0.4)
            .define_function('phi', args=['v'], expression='a*v')
            .noise('spikes', Poisson(rate='phi(v)'), expose='n')
            .field('v').equation('v', lhs='(tau*Dt + 1)*v', rhs='Em + w*n')
            ).build()


def _modes_match(me_terms, c2_terms, tol=1e-12):
    """Mode lists equal as multisets (rate, power, side) with coefficients
    within ``tol`` of the largest coefficient."""
    side = {'positive': '+', 'negative': '-'}
    ref = [(complex(t.coeff), complex(t.rate), int(t.power), side[t.side])
           for t in c2_terms if complex(t.coeff) != 0]
    got = [(complex(t.coeff), complex(t.rate), int(t.power), t.side)
           for t in me_terms]
    scale = max(abs(c) for c, *_ in ref)
    got = [g for g in got if abs(g[0]) > 1e-13 * scale]
    assert len(got) == len(ref), (got, ref)
    for c, r, p, s in ref:
        cand = [g for g in got if g[2] == p and g[3] == s
                and abs(g[1] - r) <= 1e-10 * max(1.0, abs(r))]
        assert len(cand) == 1, (c, r, p, s, got)
        assert abs(cand[0][0] - c) <= tol * scale, (cand[0], c)


# ═════════════════════════════════════════════════════════════════════════════
# A. tree level vs item 23's symbolic modes
# ═════════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize('case', ['ou', 'rotation_xx', 'rotation_xy',
                                  'chain_zz', 'chain_xz', 'chain_yx',
                                  'erlang8'])
def test_tree_triples_equal_symbolic_modes(case):
    if case == 'ou':
        m, mod = dd.load_model('ou_quartic')
        par = dict(mod.DEFAULT_FUNDAMENTAL)
        ext, mf, ab = [('dx', 1)] * 2, {'xstar1': 0}, ('dx1', 'dx1')
    elif case.startswith('rotation'):
        m, par = _rotation(), {'g': 0.2, 'w0': 1.0, 'D': 1.0}
        a, b = case[-2], case[-1]
        ext, mf = [(a, 1), (b, 1)], {'xstar1': 0, 'ystar1': 0}
        ab = (f'd{a}1', f'd{b}1')
    elif case.startswith('chain'):
        m, par = _feedback_chain(), {'a': 1.0, 'k': 2.0, 'D': 1.0}
        a, b = case[-2], case[-1]
        ext = [(a, 1), (b, 1)]
        mf = {'xstar1': 0, 'ystar1': 0, 'zstar1': 0}
        ab = (f'd{a}1', f'd{b}1')
    else:
        m, par = SSM.ou_erlang(8), None
        ext, mf, ab = [('x', 1)] * 2, None, ('dx1', 'dx1')
    res = _cc(m, ext, 0, par)
    me = ME.from_result(res)
    assert me.complete
    npar = {str(k): float(v) for k, v in res['num_params'].items()}
    # the triple is C_tau (same subsets, same arithmetic)
    assert _rel(me(TT, ell=0), res['C_tau_by_ell'][0]) <= 1e-13
    if case == 'erlang8':
        # degree-18 denominator: the parameters are substituted (exactly)
        # before factoring, so the oracle is C2's modes at these numbers
        cov = so.tree_covariance_from_result(res, model=m)
        vals = {k: v for k, v in npar.items()
                if k in {str(x) for x in cov.entry(*ab).variables()}}
        ms = cov.modes(*ab, values=vals)
    else:
        cov = so.tree_covariance(m, use_cache=False, mean_field=mf)
        ms = cov.modes(*ab, reference=npar)
    # values: triple == symbolic modes at numbers (256-bit summation there)
    t = TAUS[TAUS != 0]
    assert _rel(me(t, ell=0), ms.evaluate(t, npar)) <= 1e-13
    assert abs(me(np.array([0.0]), ell=0)[0]
               - ms.evaluate(np.array([0.0]), npar)[0]) <= 1e-13 * np.max(
                   np.abs(ms.evaluate(t, npar)))
    # the derived mode list == the symbolic one
    with warnings.catch_warnings():
        warnings.simplefilter('error', ME.MEConditioningWarning)
        _modes_match(me.modes(ell=0)[0], ms.numeric_terms(npar))


# ═════════════════════════════════════════════════════════════════════════════
# A. loop level vs the pipeline's C_tau_by_ell
# ═════════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize('case', ['ou_quartic_l2', 'colour_double_pole',
                                  'oscillator_red_noise'])
def test_loop_triples_equal_c_tau(case):
    if case == 'ou_quartic_l2':
        res = _cc(dd.load_model('ou_quartic')[0], [('dx', 1)] * 2, 2)
    elif case == 'colour_double_pole':
        res = _cc(SSM.ou_colour(1.0), [('x', 1)] * 2, 1)   # rates 1 and 1
    else:
        res = _cc(SSM.oscillator_red_noise(), [('p', 1)] * 2, 1)
    c = Cumulant.from_result(res)
    assert c.me is not None and c.me.complete and c.me_reasons == []
    for ell in c.me.orders:
        ref = res['C_tau_by_ell'][ell]
        assert _rel(c.me(TT, ell=ell), ref) <= 1e-13, ell
        # the modes reproduce the triple (well-conditioned here)
        with warnings.catch_warnings():
            warnings.simplefilter('error', ME.MEConditioningWarning)
            md = c.modes[ell]
        v = np.zeros(TT.shape, dtype=complex)
        for mt in md:
            on = (TT > 0) if mt.side == '+' else (TT <= 0)
            v[on] += mt.coeff * TT[on] ** mt.power * np.exp(mt.rate * TT[on])
        assert _rel(v, ref) <= 1e-13, ell
    if case == 'colour_double_pole':
        # the exact double pole gives tau^n e^{-tau} terms (Jordan chains)
        assert max(mt.power for mt in c.modes[0]) == 1
    # the dense triple is the same function
    G, x0, cv = c.me.sides[1]['+'].triple()
    from scipy.linalg import expm
    assert abs(cv @ expm(G * 0.7) @ x0 - c.me(np.array([0.7]), ell=1)[0]) \
        <= 1e-13 * np.max(np.abs(res['C_tau_by_ell'][1]))


def test_ou_quartic_one_loop_modes_closed_form():
    """OU + eps x^3, one loop: C_1(τ) = -3 eps D^2/mu^3 (1 + mu|τ|)
    e^{-mu|τ|} (first-order Hartree shift of the rate)."""
    m, mod = dd.load_model('ou_quartic')
    par = dict(mod.DEFAULT_FUNDAMENTAL)
    res = _cc(m, [('dx', 1)] * 2, 1, par)
    np_ = {str(k): float(v) for k, v in res['num_params'].items()}
    mu, D, eps = np_['mu'], np_['D'], np_['eps']
    md = Cumulant.from_result(res).modes[1]
    a = -3 * eps * D ** 2 / mu ** 3
    want = {('+', 0): a, ('+', 1): a * mu, ('-', 0): a, ('-', 1): -a * mu}
    assert len(md) == 4
    for mt in md:
        assert abs(mt.rate - (-mu if mt.side == '+' else mu)) < 1e-14
        assert abs(mt.coeff - want[(mt.side, mt.power)]) < 1e-15


# ═════════════════════════════════════════════════════════════════════════════
# B. pairings and the spectrum
# ═════════════════════════════════════════════════════════════════════════════

@pytest.fixture(scope='module')
def hawkes():
    m = _hawkes_1pop()
    res = compute_cumulants(m, k=2, max_ell=1, external_fields=[('n', 1)] * 2,
                            tau_grid=np.linspace(-5, 5, 21), parallel=False,
                            verbose=False, use_cache=False)
    return res, Cumulant.from_result(res)


@pytest.fixture(scope='module')
def ouq():
    res = compute_cumulants(dd.load_model('ou_quartic')[0], k=2, max_ell=1,
                            external_fields=[('dx', 1)] * 2,
                            tau_grid=np.linspace(-5, 5, 21), parallel=False,
                            verbose=False, use_cache=False)
    return res, Cumulant.from_result(res)


def _kernels():
    K = so.PairingKernel
    return {
        'two_sided': K.two_sided_exponential(1.0, 0.7, -0.5, 1.3),
        'poly_exp': (K.exponential(1.0, -0.8, 'positive', power=2)
                     + K.exponential(0.3, 1.1, 'negative', power=1)),
        'rational': ME.RationalKernel([1.0, 0.5], [1.0, 3.0, 2.0], side='+'),
        'rational_neg': [ME.RationalKernel([2.0], [1.0, 1.5], side='-'),
                         K.exponential(0.2, -2.0, 'positive')],
        # complex poles (an oscillating kernel), both sides (review: the
        # Sylvester solve must not mix a real Schur form into ztrsyl)
        'oscillatory': [ME.RationalKernel([1.0], [1.0, 0.4, 4.04], '+'),
                        ME.RationalKernel([0.5, 1.0], [1.0, 0.4, 4.04], '-')],
    }


@pytest.mark.parametrize('kname', ['two_sided', 'poly_exp', 'rational',
                                   'rational_neg', 'oscillatory'])
def test_integrate_against_vs_quadrature(ouq, kname):
    _res, c = ouq
    kern = _kernels()[kname]
    ex = c.integrate_against(kern, method='exact')
    q = c.integrate_against(kern, method='quad')
    assert ex.method == 'exact' and q.method == 'quad'
    assert abs(ex.value - q.value) <= 1e-12
    assert ex.contact == 0                       # no δ component here
    # 'auto' takes the exact route
    assert c.integrate_against(kern).method == 'exact'


def test_rational_kernel_equals_its_partial_fractions(ouq):
    """L(s) = (s + 0.5)/((s + 1)(s + 2)) = -0.5 e^{-s} + 1.5 e^{-2s}."""
    _res, c = ouq
    r = c.integrate_against(ME.RationalKernel([1.0, 0.5], [1.0, 3.0, 2.0]))
    K = so.PairingKernel
    pf = (K.exponential(-0.5, -1.0, 'positive')
          + K.exponential(1.5, -2.0, 'positive'))
    assert abs(r.value - c.integrate_against(pf).value) <= 1e-14


def test_contact_rule_with_a_jump_at_zero(hawkes):
    res, c = hawkes
    assert c.deltas and c.me is not None and c.me.complete
    w0 = sum(d.weight for d in c.deltas if d.location == 0.0)
    L = so.PairingKernel.two_sided_exponential(1.0, 0.7, -0.5, 1.3)  # L(0±) = 1, -0.5
    with pytest.raises(ValueError, match='contact'):
        c.integrate_against(L)
    ex = c.integrate_against(L, contact='mid')
    q = c.integrate_against(L, contact='mid', method='quad')
    assert abs(ex.smooth - q.smooth) <= 1e-12
    assert abs(ex.contact - w0 * 0.25) <= 1e-14
    for rule, l0 in (('positive', 1.0), ('negative', -0.5), ('exclude', 0.0)):
        p = c.integrate_against(L, contact=rule)
        assert abs(p.value - (ex.smooth + w0 * l0)) <= 1e-14
    # a continuous kernel needs no rule
    Lc = so.PairingKernel.two_sided_exponential(1.0, 0.7, 1.0, 1.3)
    p = c.integrate_against(Lc)
    assert abs(p.contact - w0) <= 1e-14


def test_spectrum_exact_vs_quadrature_and_linear_hawkes_closed_form(hawkes):
    res, c = hawkes
    om = np.array([0.0, 0.3, 1.0, 4.0])
    ex = c.spectrum(om)                         # 'auto' -> exact
    q = c.spectrum(om, method='quad')
    assert np.max(np.abs(ex - q)) <= 1e-8 * np.max(np.abs(ex))
    # linear Hawkes, exponential kernel: S(ω) = λ (1 + ω²τ²) / ((1 - b)² + ω²τ²)
    p = {str(k): float(v) for k, v in res['num_params'].items()}
    b = p['a'] * p['w']
    lam = p['a'] * p['Em'] / (1 - b)
    tau = p['tau']
    S = lam * (1 + om ** 2 * tau ** 2) / ((1 - b) ** 2 + om ** 2 * tau ** 2)
    assert np.max(np.abs(ex - S)) <= 1e-12 * np.max(S)


def test_spectrum_exact_vs_quadrature_loop_level(ouq):
    _res, c = ouq
    om = np.array([0.0, 0.5, 2.0])
    for ell in (0, 1):
        ex = c.spectrum(om, ell=ell, method='exact')
        q = c.spectrum(om, ell=ell, method='quad')
        assert np.max(np.abs(ex - q)) <= 1e-8 * np.max(np.abs(ex))
    # OU quartic: S_0 = 2D/(ω² + μ²) exactly
    p = {str(k): float(v) for k, v in _res['num_params'].items()}
    S0 = 2 * p['D'] / (om ** 2 + p['mu'] ** 2)
    assert np.max(np.abs(c.spectrum(om, ell=0) - S0)) <= 1e-14


# ═════════════════════════════════════════════════════════════════════════════
# conditioning, k >= 3, fallbacks, exports
# ═════════════════════════════════════════════════════════════════════════════

def test_nearly_coincident_poles_warn_but_the_triple_is_exact():
    """Colour noise 1e-7 away from the double pole: the mode coefficients are
    ~1e7 and cancel (warned); the triple and the resolvents are exact."""
    gap = 1e-7
    res = _cc(SSM.ou_colour(1.0 / (1.0 + gap)), [('x', 1)] * 2, 0)
    c = Cumulant.from_result(res)
    assert _rel(c.me(TT, ell=0), res['C_tau_by_ell'][0]) <= 1e-13
    with pytest.warns(ME.MEConditioningWarning):
        c.me.modes(0)
    om = np.array([0.0, 1.0])
    assert np.max(np.abs(c.spectrum(om, method='exact')
                         - c.spectrum(om, method='quad'))) <= 1e-8


@pytest.mark.parametrize('ext', [[('p', 1), ('q', 1), ('z', 1)],
                                 [('p', 1)] * 3])
def test_k3_slices_equal_dd_run(ext):
    """Distinct and identical legs (identical: Wick permutations whose
    origin is the swept leg, the case where slopes must be exact)."""
    model = SSM.oscillator_red_noise()
    for base in (None, [0.3, -0.4]):
        cfg = dd.Config(k=3, max_ell=1, tau_grid=(-3, 3, 21),
                        external_fields=ext, kpoint_base_lags=base)
        res = dd.run(model, cfg)
        for j in (1, 2):
            sm = ME.slice_me(res, j)
            st = res['_kpoint_slice_times'][j][:, j]
            inner = [ch for ch in sm.chambers
                     if np.isfinite(ch[0]) and np.isfinite(ch[1])]
            for ell, ref in res['C_tau_slices_by_ell'][j].items():
                ok = np.array([not any(lo < t <= hi for lo, hi, *_ in inner)
                               for t in st])
                got = sm(st[ok], ell=ell)
                assert _rel(got, ref[ok]) <= 1e-13
                if base is not None:
                    # a swept leg between two legs: two segments move
                    # (item 54: a Kronecker chain triple)
                    assert _rel(sm(st[~ok], ell=ell), ref[~ok]) <= 1e-13
            md = sm.modes()
            assert all(md[(lo, hi)] for lo, hi, *_ in sm.chambers
                       if (lo, hi) in md)


def test_pole_engine_is_never_silent(monkeypatch):
    monkeypatch.setenv(SS.ENGINE_ENV, 'poles')
    res = _cc(dd.load_model('ou_quartic')[0], [('dx', 1)] * 2, 1,
              taus=np.linspace(-3, 3, 7))
    c = Cumulant.from_result(res)
    assert c.me is not None and not c.me.complete
    assert any('pole route' in r or 'poles' in r for r in c.me_reasons)
    with pytest.raises(ME.MEUnavailable):
        c.spectrum([0.5], method='exact')
    with pytest.warns(ME.MEFallbackWarning):
        s = c.spectrum([0.5])
    assert np.isfinite(s).all()
    with pytest.warns(ME.MEFallbackWarning):
        assert c.modes is None


def test_loaded_file_has_no_triples(tmp_path):
    from api import save_npz, load_npz
    res = _cc(dd.load_model('ou_quartic')[0], [('dx', 1)] * 2, 0,
              taus=np.linspace(-3, 3, 7))
    p = save_npz(res, str(tmp_path / 'r.npz'))
    c = Cumulant.from_saved(load_npz(p if isinstance(p, str)
                                     else str(tmp_path / 'r.npz')))
    assert c.me is None and c.me_reasons and c.modes is None


def test_dd_exports():
    for name in ('tree_modes', 'Kernel', 'integrate_against',
                 'tree_covariance', 'tree_covariance_from_result',
                 'TreeCovariance', 'ModeSum', 'KernelIntegral',
                 'inverse_transform', 'pair_with_kernel', 'ModeTerm',
                 'CumulantME', 'RationalKernel', 'MEUnavailable',
                 'MEFallbackWarning', 'MEConditioningWarning', 'slice_me'):
        assert getattr(dd, name) is not None, name
    assert dd.PairingKernel is so.PairingKernel and dd.tree_modes is so.tree_modes


def test_generic_reference_retry_never_flips_the_split():
    """Poles coincide at the reference (a = b = 1); a pole 1e-4 from the
    imaginary axis would cross under the move: refused by name."""
    from sage.all import var, I
    w, a, b = var('omega'), var('a'), var('b')
    C0 = 1 / ((I * w + a - b + SRf(1e-4)) * (w ** 2 + a ** 2) * (w ** 2 + b ** 2))
    with pytest.raises(so.ClosedFormUnavailable, match='half-plane'):
        so.pair_with_kernel(C0, 1, w, fallback=False,
                            reference={'a': 1.0, 'b': 1.0})
    # a coincidence that the move cannot flip is accepted (item 23's C1 test)
    mu, T = var('mu'), var('T')
    p = so.pair_with_kernel(1 / (mu + I * w), 1 / (1 + I * w * T), w,
                            reference={'mu': 1.0, 'T': 1.0})
    assert bool((p.expression - 1 / (1 + mu * T)).simplify_full() == 0)


def SRf(x):
    from sage.all import QQ, SR
    return SR(QQ(x))
