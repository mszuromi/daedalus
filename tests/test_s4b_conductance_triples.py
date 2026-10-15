"""
tests/test_s4b_conductance_triples.py
=====================================
S4b: matrix-exponential triples (``api/me_output.py``) for models with
field x Conv (conductance) vertices and delay kernels.  A conductance diagram
whose kernel slots can be filled in several ways is integrated as the
average of its placements (``final_integral._average_conv_placements``); its
contribution now exposes them (``phase_j_placements``), so the dressed
lines and kernel edges of every placement give triples, and pairings and
spectra are exact.  Inline public models only:

* a two-rate conductance kernel (tree + one loop): triples == ``C_tau_by_ell``
  on a 201-point grid, ``integrate_against`` 'exact' == quadrature of the
  same cumulant, spectrum exact == quadrature;
* the alpha kernel at equal rates (exact double pole) against its Markov
  embedding (triples and spectra);
* a delayed exponential conductance kernel (Pade-4 copy): triples of the Pade
  tree and loop == ``C_tau_by_ell``; with the exact delayed tree (default)
  the tree has no triple (named), the loop's pairing stays exact, and the
  tree's quadrature fallback is warned, counted and refused by name where
  the exact tree cannot be evaluated.
"""
import os
import sys
import warnings

import numpy as np
import pytest

pytest.importorskip('sage.all')

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import daedalus as dd                                                # noqa: E402
from api import compute_cumulants                                    # noqa: E402
from api import me_output as ME                                      # noqa: E402
from api.compute import _ITO_EPS                                     # noqa: E402
from api.noise import Gaussian                                       # noqa: E402
from api.sde import SDE                                              # noqa: E402
from api.symbolic_out import Kernel                                  # noqa: E402
import engine.integration.time_domain.statespace as SS               # noqa: E402

TAUS = np.linspace(-5.0, 5.0, 201)
TT = np.where(TAUS == 0, -_ITO_EPS, TAUS)     # C_tau's τ = 0 sample
DX2 = [('dx', 1), ('dx', 1)]
X2 = [('x', 1), ('x', 1)]
L2 = Kernel.two_sided_exponential(1.0, 0.7, -0.6, 1.3)


@pytest.fixture(autouse=True)
def _default_engine(monkeypatch):
    monkeypatch.delenv(SS.ENGINE_ENV, raising=False)


def _dense_laplace(me, ell, z):
    """``Σ_sides c (z_s I − G_s)^{-1} x0`` from the dense block-diagonal
    triples (``scipy.linalg.solve``; independent of the triangular resolvent
    solves behind ``integrate_against`` / ``spectrum``); ``z = (z_+, z_-)``
    and the '-' side weighted by ``w_-``: ``z = ((z_+, w_+), (z_-, w_-))``."""
    import scipy.linalg as sl
    tot = 0j
    for e in ([ell] if ell is not None else me.orders):
        for side, (zz, w) in zip(('+', '-'), z):
            G, x, c = me.sides[e][side].triple()
            if G.size:
                tot += w * (c @ sl.solve(zz * np.eye(len(G)) - G, x))
    return tot


L2_DENSE = ((1 / 0.7, 1.0), (1 / 1.3, -0.6))     # L2 term by term


def _rel(a, b):
    b = np.asarray(b)
    return float(np.max(np.abs(np.asarray(a) - b)) / np.max(np.abs(b)))


def _cc(model, ext, ell=1, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return compute_cumulants(model, k=2, max_ell=ell, external_fields=ext,
                                 tau_grid=TAUS, parallel=False, verbose=False,
                                 use_cache=False, **kw)


def _conductance(name, tau2):
    """x' = h - a x - b x (g * x) + noise, g = 1/((1 + iωτ)(1 + iωτ2)) (the
    alpha kernel at τ2 = τ)."""
    from api.theory import TemporalTheoryBuilder
    P = dict(a=1.0, b=0.3, h=0.5, tau=0.7, tau2=tau2, D=0.15)
    b = (TemporalTheoryBuilder(name).population('E', size=1)
         .physical_field('x', population='E'))
    for k, v in P.items():
        b = b.parameter(k, default=[v], indexed_by=['E'],
                        **({'domain': 'positive'}
                           if k in ('tau', 'tau2', 'D') else {}))
    return (b.define_kernel('g', indexed_by=['E'], freq_image=(
                '1/((1 + I*omega*tau[i])*(1 + I*omega*tau2[i]))'))
            .set_action_text('sum(xt[i]*((Dt + a[i])*x[i] - h[i] + b[i]*x[i]'
                             '*Conv(g[i], x[i])) - D[i]*xt[i]^2 for i in E)')
            .set_mf_equation('xstar', '(h[i] - b[i]*xstar[i]^2)/a[i]')
            .build())


def _alpha_embedding(name):
    """The alpha-kernel model with the kernel as two linear fields."""
    from api.theory import TemporalTheoryBuilder
    P = dict(a=1.0, b=0.3, h=0.5, tau=0.7, tau2=0.7, D=0.15)
    b = (TemporalTheoryBuilder(name).population('E', size=1)
         .physical_field('x', population='E')
         .physical_field('s', population='E')
         .physical_field('u', population='E'))
    for k, v in P.items():
        b = b.parameter(k, default=[v], indexed_by=['E'],
                        **({'domain': 'positive'}
                           if k in ('tau', 'tau2', 'D') else {}))
    act = ('sum(xt[i]*((Dt + a[i])*x[i] - h[i] + b[i]*x[i]*s[i]) '
           '- D[i]*xt[i]^2 + ut[i]*((tau[i]*Dt + 1)*u[i] - x[i]) '
           '+ st[i]*((tau2[i]*Dt + 1)*s[i] - u[i]) for i in E)')
    return (b.set_action_text(act).set_mf_equation('ustar', 'xstar[i]')
            .set_mf_equation('xstar', '(h[i] - b[i]*xstar[i]^2)/a[i]')
            .set_mf_equation('sstar', 'xstar[i]').build())


def _delayed_conductance(name):
    """x' = b - a x - c x (g * x) + noise, g a delayed exponential."""
    return (SDE(name).parameter('a', default=1.0).parameter('b', default=1.0)
            .parameter('c', default=0.3)
            .parameter('d', default=0.5, domain='positive')
            .parameter('tau', default=0.4, domain='positive')
            .parameter('D', default=0.05).field('x')
            .define_kernel('g', delay='d', freq_image='1/(1 + I*omega*tau)')
            .drift('x', '-a*x + b - c*x*Conv(g, x)')
            .noise('xi', Gaussian(var='2*D'), couples={'x': 1}).build())


@pytest.fixture(scope='module')
def cond():
    return _cc(_conductance('s4b_conductance', 0.4), DX2)


def test_placements_are_exposed(cond):
    pj = cond['phase_j_by_ell'][1]
    fns = [g['contribution'] for g in pj['groups']
           if isinstance(g, dict) and g.get('contribution') is not None]
    lists = [ME._parts_list(f) for f in fns]
    assert all(pl is not None for pl in lists)
    placed = [pl for f, pl in zip(fns, lists)
              if getattr(f, 'phase_j_placements', None) is not None]
    assert placed                       # the model has averaged placements
    for pl in placed:
        assert len(pl) > 1 and abs(sum(w for w, _p in pl) - 1.0) <= 1e-15


def test_conductance_triples_equal_c_tau(cond):
    c = dd.cumulant(cond)
    me = c.me
    assert me is not None and me.complete, c.me_reasons
    for ell in (0, 1):
        ref = np.asarray(cond['C_tau_by_ell'][ell])
        assert np.max(np.abs(ref)) > 0
        assert _rel(me(TT, ell=ell), ref) <= 1e-13, ell


def test_conductance_pairing_and_spectrum_exact(cond):
    c = dd.cumulant(cond)
    with warnings.catch_warnings():
        warnings.simplefilter('error', ME.MEFallbackWarning)
        p = c.integrate_against(L2, contact='exclude')
        S = c.spectrum([0.0, 0.7, 2.5])
    assert p.method == 'exact' and not c.me_fallbacks
    q = c.integrate_against(L2, contact='exclude', method='quad')
    assert abs(complex(p) - complex(q)) <= 1e-8 * abs(complex(q))
    Sq = c.spectrum([0.7], method='quad')
    assert abs(S[1] - Sq[0]) <= 1e-8 * abs(Sq[0])


def test_structured_resolvent_equals_the_dense_one(cond, monkeypatch):
    """Blocks beyond MAX_BLOCK are paired by forward block substitution on
    the segment (matrix free): forced on every block here, it equals the
    dense triangular route (pairing, spectrum, a double power)."""
    c = dd.cumulant(cond)
    me = c.me
    om = [0.0, 0.7, 2.5]
    p_d = me.integrate_smooth(L2)
    S_d = me.spectrum_smooth(om)
    side = me.sides[1]['+']
    r2_d = side.resolvent(1.3 + 0.4j, 2)
    blocks = side.blocks
    monkeypatch.setattr(ME, 'MAX_BLOCK', 0)
    for b in blocks:
        b._tri = None
    p_s = me.integrate_smooth(L2)
    S_s = me.spectrum_smooth(om)
    r2_s = side.resolvent(1.3 + 0.4j, 2)
    assert abs(p_s - p_d) <= 1e-13 * abs(p_d)
    assert np.max(np.abs(S_s - S_d)) <= 1e-13 * np.max(np.abs(S_d))
    assert abs(r2_s - r2_d) <= 1e-13 * abs(r2_d)


def test_fallback_without_triples_keeps_the_requested_order(cond):
    # review: with no triples at all, 'auto' must still pair only ``ell``
    c = dd.cumulant(cond)
    c._me, c._me_reasons = None, ['forced: no triples']
    with pytest.warns(ME.MEFallbackWarning, match=r'loop order\(s\) \[1\]'):
        p = c.integrate_against(L2, ell=1, contact='exclude')
    q = c.integrate_against(L2, ell=1, contact='exclude', method='quad')
    assert p.method == 'quad' and complex(p) == complex(q)
    assert c.me_fallbacks == {('integrate_against', 1): 1}
    with pytest.warns(ME.MEFallbackWarning):
        S = c.spectrum([0.7], ell=1)
    assert S[0] == c.spectrum([0.7], ell=1, method='quad')[0]


def test_alpha_kernel_at_equal_rates_vs_embedding():
    kern = _cc(_conductance('s4b_alpha_kernel', 0.7), DX2)
    emb = _cc(_alpha_embedding('s4b_alpha_embedding'), DX2)
    ck, ce = dd.cumulant(kern), dd.cumulant(emb)
    assert ck.me.complete and ce.me.complete, (ck.me_reasons, ce.me_reasons)
    for ell in (0, 1):
        ref = np.asarray(emb['C_tau_by_ell'][ell])
        assert _rel(ck.me(TT, ell=ell), ref) <= 1e-13, ell
    om = [0.0, 0.3, 1.1, 4.0]
    assert _rel(ck.spectrum(om), ce.spectrum(om)) <= 1e-12


def test_delayed_conductance_pade_tree_triples():
    r = _cc(_delayed_conductance('s4b_delay_pade'), X2,
            delay={'order': 4, 'tree': 'pade'})
    c = dd.cumulant(r)
    assert c.me.complete, c.me_reasons
    for ell in (0, 1):
        ref = np.asarray(r['C_tau_by_ell'][ell])
        assert _rel(c.me(TT, ell=ell), ref) <= 1e-13, ell
    p = c.integrate_against(L2, contact='exclude')
    assert p.method == 'exact'
    q = _dense_laplace(c.me, None, L2_DENSE)
    assert abs(complex(p) - q) <= 1e-12 * abs(q)


def test_exact_delayed_tree_is_named_and_split():
    r = _cc(_delayed_conductance('s4b_delay_exact'), X2, delay={'order': 4})
    assert r['delay']['tree'] == 'exact'
    c = dd.cumulant(r)
    assert c.me.reasons[1] == []
    assert any('exact delayed covariance' in s for s in c.me.reasons[0])
    # the loop's triples are the Pade copy's loop == C_tau's loop
    assert _rel(c.me(TT, ell=1), r['C_tau_by_ell'][1]) <= 1e-13
    with pytest.raises(ME.MEUnavailable, match='exact delayed covariance'):
        c.integrate_against(L2, contact='exclude', method='exact')
    # the tree's fallback would be quadrature to infinite lag, which the
    # exact tree (a capped method-of-steps stack) cannot finish: refused up
    # front by name, for 'auto' and 'quad' alike, and nothing is counted
    for meth in ('auto', 'quad'):
        with pytest.raises(ME.MEUnavailable, match='infinite lag'):
            c.integrate_against(L2, contact='exclude', method=meth)
        with pytest.raises(ME.MEUnavailable, match='infinite lag'):
            c.spectrum([0.0, 1.0], method=meth)
    assert c.me_fallbacks == {}
    # the loop alone stays exact, silently, and equals its quadrature
    with warnings.catch_warnings():
        warnings.simplefilter('error', ME.MEFallbackWarning)
        p = c.integrate_against(L2, ell=1, contact='exclude')
        S = c.spectrum([0.0, 1.0], ell=1)
    assert p.method == 'exact'
    q = _dense_laplace(c.me, 1, L2_DENSE)
    assert abs(complex(p) - q) <= 1e-12 * abs(q)
    # the spectrum at ω = 1: c (iω − G_+)^{-1} x0 + c (−iω − G_−)^{-1} x0
    S1 = _dense_laplace(c.me, 1, ((1j, 1.0), (-1j, 1.0)))
    assert abs(S[1] - S1) <= 1e-12 * abs(S1)
