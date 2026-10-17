"""
tests/test_me_completeness.py
=============================
Item 54: matrix-exponential output completeness (``api/me_output.py``).

* k >= 3 middle chambers: a swept leg between two other legs moves two
  segments (one grows, one shrinks); ``SubsetSS.me_chain`` returns the
  chain and :class:`api.me_output.MEChainBlock` holds it as one Kronecker
  triple.  Every chamber of every slice has triples; they equal the Phase J
  callables (``total_C_by_ell``) and ``dd.run``'s slices, and the mode list
  of a middle chamber reproduces the values.
* the exact delayed tree (``api.delay``, k = 2): values, Laplace transforms
  (pairings) and spectrum without Pade or quadrature, checked against the
  closed-form spectrum of the scalar delayed OU process and an independent
  spectral quadrature of its Laplace transforms.

Inline public models or tracked ones only.
"""
import importlib.util
import math
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
from api.noise import Gaussian                                       # noqa: E402
from api.sde import SDE                                              # noqa: E402
from api.symbolic_out import Kernel                                  # noqa: E402
import engine.integration.time_domain.statespace as SS               # noqa: E402


def _ss_models():
    spec = importlib.util.spec_from_file_location(
        'ss_models_me54', os.path.join(HERE, 'test_statespace_integrator.py'))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


SSM = _ss_models()


@pytest.fixture(autouse=True)
def _default_engine(monkeypatch):
    monkeypatch.delenv(SS.ENGINE_ENV, raising=False)


def _rel(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return float(np.max(np.abs(a - b)) / max(np.max(np.abs(b)), 1e-300))


def _run3(model, ext, base, max_ell=1, params=None):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        cfg = dd.Config(k=3, max_ell=max_ell, tau_grid=(-2, 2, 9),
                        external_fields=ext, kpoint_base_lags=base,
                        parameters=params)
        return dd.run(model, cfg)


def _check_every_chamber(res, tol=1e-13):
    """Every chamber of both slices against the Phase J callables at the
    same external times; returns the number of chain blocks seen."""
    fns = res['total_C_by_ell']
    n_chain = 0
    for j in (1, 2):
        sm = ME.slice_me(res, j)
        assert not any(any(v.values()) for v in sm.reasons.values()), \
            sm.reasons
        for lo, hi, _a, _sg, sides in sm.chambers:
            lo_ = lo if np.isfinite(lo) else hi - 2.5
            hi_ = hi if np.isfinite(hi) else lo + 2.5
            tt = np.linspace(lo_, hi_, 7)[1:-1]
            for ell in sorted(fns):
                n_chain += sum(isinstance(b, ME.MEChainBlock)
                               for b in sides[ell].blocks)
                ref = []
                for t in tt:
                    T = [0.0, 0.0, 0.0]
                    for leg, v in sm.pinned.items():
                        T[leg] = v
                    T[j] = t
                    ref.append(complex(fns[ell](*T)))
                assert _rel(sm(tt, ell=ell), ref) <= tol, (j, lo, hi, ell)
    return n_chain


@pytest.mark.parametrize('ext', [[('p', 1), ('q', 1), ('z', 1)],
                                 [('p', 1)] * 3])
def test_middle_chambers_oscillator(ext):
    res = _run3(SSM.oscillator_red_noise(), ext, [0.3, -0.4])
    assert _check_every_chamber(res) > 0
    # dd.run's own slices, middle chambers included
    for j in (1, 2):
        sm = ME.slice_me(res, j)
        st = res['_kpoint_slice_times'][j][:, j]
        for ell, ref in res['C_tau_slices_by_ell'][j].items():
            assert _rel(sm(st, ell=ell), ref) <= 1e-13


def test_middle_chamber_modes_reproduce_the_values():
    res = _run3(SSM.oscillator_red_noise(), [('p', 1), ('q', 1), ('z', 1)],
                [0.3, -0.4])
    for j in (1, 2):
        sm = ME.slice_me(res, j)
        md = sm.modes()
        for lo, hi, *_ in sm.chambers:
            if not (np.isfinite(lo) and np.isfinite(hi)):
                continue
            tt = np.linspace(lo, hi, 7)[1:]
            for ell, terms in md[(lo, hi)].items():
                mv = sum(t.coeff * tt ** t.power * np.exp(t.rate * tt)
                         for t in terms)
                assert _rel(mv, sm(tt, ell=ell)) <= 1e-12, (j, ell)


def test_chain_block_kronecker_forms_equal_the_chain():
    """The dense Kronecker triple and its triangular form of each chain
    block equal the segment-by-segment values (blocks that are not zero to
    rounding)."""
    import scipy.linalg as sl
    res = _run3(SSM.oscillator_red_noise(), [('p', 1), ('q', 1), ('z', 1)],
                [0.3, -0.4])
    sm = ME.slice_me(res, 2)
    lo, hi, _a, _sg, sides = sm.chambers[1]
    u = np.linspace(0.0, hi - lo, 5)
    side_scale = np.max(np.abs(sides[1](u)))
    seen = 0
    for b in sides[1].blocks:
        if not isinstance(b, ME.MEChainBlock):
            continue
        v = b.values(u)
        if np.max(np.abs(v)) < 1e-8 * side_scale:
            continue
        G, x, c = b.triple()
        T, xt, ct = b.tri()
        assert np.allclose(np.tril(T, -1), 0)
        dv = np.array([c @ sl.expm(G * s) @ x for s in u])
        tv = np.array([ct @ sl.expm(T * s) @ xt for s in u])
        assert _rel(dv, v) <= 1e-11 and _rel(tv, v) <= 1e-11
        with pytest.raises(ME.MEUnavailable, match='finite interval'):
            b.resolvent(1.0)
        seen += 1
    assert seen


def test_middle_chambers_double_well():
    """An asymmetric saddle (mu < 0: a nonzero third cumulant)."""
    res = _run3(dd.load_model('ou_quartic_double_well')[0], [('dx', 1)] * 3,
                [0.3, -0.4], params={'mu': -1.0, 'eps': 0.1, 'D': 0.1})
    assert max(np.max(np.abs(v)) for v in
               res['C_tau_slices_by_ell'][1].values()) > 1e-6
    assert _check_every_chamber(res) > 0


def test_cumulant_me_slice():
    res = _run3(SSM.oscillator_red_noise(), [('p', 1)] * 3, [0.3, -0.4],
                max_ell=0)
    c = dd.cumulant(res)
    assert c.me is None
    sm = c.me_slice(2)
    st = res['_kpoint_slice_times'][2][:, 2]
    assert _rel(sm(st, ell=0), res['C_tau_slices_by_ell'][2][0]) <= 1e-13


# ── the exact delayed tree ─────────────────────────────────────────────────

A_, C_, D_, NS = 1.2, -0.7, 0.9, 0.15


@pytest.fixture(scope='module')
def delayed_ou():
    m = (SDE('me54_delay_ou').parameter('a', default=A_)
         .parameter('c', default=C_)
         .parameter('d', default=D_, domain='positive')
         .parameter('D', default=NS).field('x')
         .drift('x', '-a*x + c*x(t - d)')
         .noise('xi', Gaussian(var='2*D'), couples={'x': 1}).build())
    T = np.linspace(-4.0, 4.0, 41)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        r = compute_cumulants(m, k=2, max_ell=0,
                              external_fields=[('x', 1), ('x', 1)],
                              tau_grid=T, parallel=False, verbose=False)
    assert r['delay']['tree'] == 'exact'
    return r, T


def _S(w):
    """The closed-form spectrum of x' = -a x + c x(t - d) + sqrt(2D) xi."""
    return 2 * NS / abs(1j * w + A_ - C_ * np.exp(-1j * w * D_)) ** 2


def test_exact_tree_spectrum_closed_form(delayed_ou):
    r, T = delayed_ou
    c = dd.cumulant(r)
    om = np.array([0.0, 0.4, 1.3, 3.0, 9.0])
    ref = np.array([_S(w) for w in om])
    with warnings.catch_warnings():
        warnings.simplefilter('error', ME.MEFallbackWarning)
        S = c.spectrum(om)
    assert _rel(S, ref) <= 1e-13
    me = c.me
    Sl = np.array([me.sides[0]['+'].resolvent(1j * w)
                   + me.sides[0]['-'].resolvent(-1j * w) for w in om])
    assert _rel(Sl, ref) <= 1e-13
    TT = T          # C_tau's τ = 0 point: the exact Itô left limit (at_zero='-')
    assert _rel(me(TT, ell=0), r['C_tau_by_ell'][0]) <= 1e-15


@pytest.mark.parametrize('z', [0.3, 1.0, 2.5])
@pytest.mark.parametrize('q', [0, 1, 2])
def test_exact_tree_laplace_vs_spectral_quadrature(delayed_ou, z, q):
    """``∫_0^∞ u^q e^{-zu} C(u) du = (1/2π) ∫ S(ω) q! / (z − iω)^{q+1}
    dω`` (C is even here: both sides)."""
    from scipy.integrate import quad
    r, _T = delayed_ou
    me = dd.cumulant(r).me

    def f(w):
        return (_S(w) * math.factorial(q) / (z - 1j * w) ** (q + 1)).real
    ref = quad(f, 0, np.inf, epsabs=1e-15, epsrel=1e-13, limit=2000)[0] / np.pi
    for s in ('+', '-'):
        got = me.sides[0][s].laplace(z, q)
        assert abs(got - ref) <= 1e-12 * abs(ref), (s, got, ref)


def test_exact_tree_pairings_and_refusals(delayed_ou):
    r, _T = delayed_ou
    c = dd.cumulant(r)
    me = c.me
    assert me.reasons == {0: []} and me.mode_reasons[0]
    L2 = Kernel.two_sided_exponential(1.0, 0.7, -0.6, 1.3)
    with warnings.catch_warnings():
        warnings.simplefilter('error', ME.MEFallbackWarning)
        p = c.integrate_against(L2, contact='exclude')
        # (s + 0.5) / ((s + 1)(s + 2)) = -0.5 / (s + 1) + 1.5 / (s + 2)
        rk = ME.RationalKernel([1.0, 0.5], [1.0, 3.0, 2.0], side='-')
        p2 = c.integrate_against(rk, contact='exclude')
    assert p.method == 'exact' and c.me_fallbacks == {}
    ref = (me.sides[0]['+'].laplace(1 / 0.7)
           - 0.6 * me.sides[0]['-'].laplace(1 / 1.3))
    assert abs(complex(p) - ref) <= 1e-14 * abs(ref)
    ref2 = (-0.5 * me.sides[0]['-'].laplace(1.0)
            + 1.5 * me.sides[0]['-'].laplace(2.0))
    assert abs(complex(p2) - ref2) <= 1e-13 * abs(ref2)
    # explicit quadrature to infinite lag stays refused (the method-of-steps
    # stack is capped); the exact tree has no triple and no mode list
    with pytest.raises(ME.MEUnavailable, match='infinite lag'):
        c.integrate_against(L2, contact='exclude', method='quad')
    with pytest.raises(ME.MEUnavailable, match='no finite triple'):
        me.sides[0]['+'].triple()
    with pytest.raises(ME.MEUnavailable, match='no finite triple'):
        me.modes(0)
    with pytest.warns(ME.MEFallbackWarning, match='no finite mode list'):
        assert c.modes is None
