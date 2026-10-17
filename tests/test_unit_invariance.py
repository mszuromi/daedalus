"""Unit invariance (audit W4a): an exact change of units must not change any
verdict and must rescale every number exactly.

Two exact symmetries of every model:

* **field units** ``x -> s x``: each coefficient picks up the powers of ``s``
  that keep the equations the same; the saddle scales by ``s``, a k-point
  cumulant by the product of its legs' factors;
* **time units** ``t -> t / a`` (``y(t) = x(a t)``): every rate is multiplied
  by ``a``, every time constant and delay divided by it, a Gaussian
  variance multiplied by ``a``; ``C'(tau) = C(a tau)``; a spike train picks
  up a factor ``a`` per leg.

The twins below change the units by up to 1e+-12 and assert identical status,
refusal codes, root sets and selected roots, and the rescaled values.  They
cover the thresholds fixed in W4a (the mean-field DAE search and its checks,
the analyzer's marginality floors, the Lambert-W branch of G2-K, the
delta-contact rule, PhaseType validity, ``numeric_inverse_transform``), on the
default (state-space) engine.  Tracked models: ``ou_quartic``,
``ou_quartic_colored``; the rest are inline.
"""
from __future__ import annotations

import math
import warnings

import numpy as np
import pytest

pytest.importorskip('sage.all')

import daedalus as dd                                                 # noqa: E402
from api import compute_cumulants                                     # noqa: E402
from api._mean_field_dae import solve_mean_field_dae                  # noqa: E402
from api.noise import (CompoundPoisson, GammaProcess, Gaussian,       # noqa: E402
                       Hyperexponential, PhaseType, Poisson)
from api.sde import SDE                                               # noqa: E402

FIELD = (1e-9, 1e-6, 1e-3, 1e3, 1e6)
TIME = (1e-6, 1e-3, 1e3, 1e6)


def _quiet(fn, *a, **k):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return fn(*a, **k)


def _cc(model, params=None, k=2, ell=1, ext=(('x', 1), ('x', 1)), taus=(0.0, 0.4, 1.3)):
    return _quiet(compute_cumulants, model, k=k, max_ell=ell, external_fields=list(ext),
                  parameters=params, tau_grid=np.asarray(taus, dtype=float),
                  parallel=False, verbose=False, use_cache=False)


def _codes(r):
    an = r.get('analysis')
    return sorted(f.code for f in an.findings) if an is not None else None


def _assert_twin(base, twin, factor, rtol):
    """Every loop order of ``twin`` equals ``factor`` times ``base``."""
    assert set(base['C_tau_by_ell']) == set(twin['C_tau_by_ell'])
    for ell, v in base['C_tau_by_ell'].items():
        a = factor * np.asarray(v, dtype=complex)
        b = np.asarray(twin['C_tau_by_ell'][ell], dtype=complex)
        scale = np.max(np.abs(a))
        assert np.max(np.abs(a - b)) <= rtol * scale, (ell, a, b)


# ── 1. the mean-field DAE root search (UNIT-02, ACC-21) ──────────────────────────────────────

def _cubic(K, s, D=0.01):
    """x' = -K (x - s)(x - 2s)(x - 3s): roots s, 2s, 3s; K = 1/s^2 keeps the
    rates fixed, so ``s`` is a pure change of field units."""
    return (SDE('w4a_cubic').parameter('K', default=K).parameter('s', default=s)
            .parameter('D', default=D).field('x')
            .drift('x', '-K*(x - s)*(x - 2*s)*(x - 3*s)')
            .noise('xi', Gaussian(var='2*D'), couples={'x': 1}).build())


@pytest.mark.parametrize('s', (1.0,) + FIELD)
def test_cubic_root_set_is_field_unit_invariant(s):
    """The three roots are found and kept distinct in any field unit (a 1e-10
    absolute dedup floor merged {1,2,3}e-9; the default seed box scaled with
    the largest parameter K = 1/s^2 and missed s and 2s), and the selected
    root is the same one."""
    for idx in (0, 2):
        r = _quiet(solve_mean_field_dae, _cubic(1 / s**2, s), {'K': 1 / s**2, 's': s, 'D': 0.01},
                   fixed_point_index=idx)
        roots = sorted(x['values']['xstar'][0] / s for x in r['mf_all_roots'])
        assert roots == pytest.approx([1.0, 2.0, 3.0], rel=1e-13, abs=0)
        assert r['mf_values']['xstar'][0] / s == pytest.approx(1.0 + idx, rel=1e-13)


@pytest.mark.parametrize('a', (1e-9,) + TIME + (1e9,))
def test_cubic_roots_are_time_unit_invariant(a):
    """Time units multiply every residual row by ``a``: the roots (and their
    digits: the saddle is refined to round-off, ACC-21) do not move."""
    r0 = _quiet(solve_mean_field_dae, _cubic(1.0, 1.0), {'K': 1.0, 's': 1.0, 'D': 0.01})
    r1 = _quiet(solve_mean_field_dae, _cubic(a, 1.0), {'K': a, 's': 1.0, 'D': 0.01})
    v0 = [x['values']['xstar'][0] for x in r0['mf_all_roots']]
    v1 = [x['values']['xstar'][0] for x in r1['mf_all_roots']]
    assert v1 == pytest.approx(v0, rel=4e-15, abs=0)


def _transcendental(K, drift='K*(exp(-x) - x)'):
    return (SDE('w4a_transc').parameter('K', default=K).parameter('D', default=0.01)
            .field('x').drift('x', drift)
            .noise('xi', Gaussian(var='2*D'), couples={'x': 1}).build())


@pytest.mark.parametrize('K', (1.0, 1e4, 1e8, 1e12))
def test_mf_sector_check_is_relative(K):
    """A saddle solved by iteration (``exp(-x) = x``) in fast time units: the
    expand-time check used an absolute 1e-9 residual and refused K = 1e8
    (residual 7e-9 = 7e-17 of the terms).  The saddle is W(1) in every unit."""
    for drift, root in (('K*(exp(-x) - x)', 0.5671432904097838),
                        ('K*(2 - x - x^3/3 - exp(-x))', 1.1630524254621382)):
        r = _cc(_transcendental(K, drift), k=1, ell=0, ext=(('x', 1),), taus=(0.0,))
        assert r['mf_values']['xstar'][0] == pytest.approx(root, rel=1e-15, abs=0)


def test_root_acceptance_rejects_a_point_that_is_not_a_root():
    """A hybr 'success' far from any root (reached from a seed many decades
    away, with a garbage Broyden Jacobian) is not accepted: the acceptance
    uses a fresh finite-difference Jacobian, relative to each row."""
    m = (SDE('w4a_lin2').parameter('qa', default=-1.85).parameter('qb', default=0.26)
         .parameter('qd', default=0.17).parameter('qf', default=-1.62)
         .parameter('qg', default=-0.087).parameter('qh', default=-0.032)
         .field('xa').field('xb')
         .drift('xa', 'qa*xa + qb*xb + qd*xb*xa').drift('xb', 'qf*xb + qg + qh*xb^3')
         .noise('xi', Gaussian(var='0.3'), couples={'xa': 1, 'xb': 1}).build())
    par = dict(qa=-1.85, qb=0.26, qd=0.17, qf=-1.62, qg=-0.087, qh=-0.032)
    r = _quiet(solve_mean_field_dae, m, par)
    xa = [x['values']['xastar'][0] for x in r['mf_all_roots']]
    xb = [x['values']['xbstar'][0] for x in r['mf_all_roots']]
    assert len(xa) == 1                               # xa is linear given xb: one root
    assert par['qa'] * xa[0] + par['qb'] * xb[0] + par['qd'] * xb[0] * xa[0] == \
        pytest.approx(0.0, abs=1e-16)


# ── 2. end-to-end twins on the default engine ────────────────────────────────────────────────

_OUQ = {'mu': 1.0, 'eps': 0.02, 'D': 1.0}
_OUQ_EXT = (('dx', 1), ('dx', 1))
_TAUS = np.array([0.0, 0.4, 1.3])


@pytest.fixture(scope='module')
def ouq():
    model, _ = dd.load_model('ou_quartic')
    return model, _cc(model, _OUQ, ell=1, ext=_OUQ_EXT, taus=_TAUS)


@pytest.mark.parametrize('a', TIME)
def test_ou_quartic_time_twin(ouq, a):
    """Tracked ``ou_quartic`` with every rate times ``a`` (``mu``, ``eps``,
    ``D``): C'(tau / a) = C(tau), tree and one loop; the same analyzer codes."""
    model, base = ouq
    par = {k: v * a for k, v in _OUQ.items()}
    twin = _cc(model, par, ell=1, ext=_OUQ_EXT, taus=_TAUS / a)
    assert _codes(twin) == _codes(base)
    _assert_twin(base, twin, 1.0, 1e-12)


@pytest.mark.parametrize('s', (1e-6, 1e-3, 1e3, 1e6))
def test_ou_quartic_field_twin(ouq, s):
    """x -> s x: ``eps / s^2``, ``D s^2``; C' = s^2 C."""
    model, base = ouq
    par = {'mu': 1.0, 'eps': 0.02 / s**2, 'D': s**2}
    twin = _cc(model, par, ell=1, ext=_OUQ_EXT, taus=_TAUS)
    assert _codes(twin) == _codes(base)
    _assert_twin(base, twin, s**2, 1e-12)


@pytest.mark.slow
@pytest.mark.parametrize('a', (1e-3, 1e3))
def test_ou_quartic_colored_time_twin(a):
    """Tracked ``ou_quartic_colored`` (colored noise, Markov embedded):
    ``mu, eps, D`` times ``a``, ``tauc / a``."""
    model, _ = dd.load_model('ou_quartic_colored')
    p0 = {'mu': 1.0, 'eps': 0.1, 'D': 1.0, 'tauc': 1.0}
    base = _cc(model, p0, ell=1, ext=(('x', 1), ('x', 1)), taus=_TAUS)
    par = {'mu': a, 'eps': 0.1 * a, 'D': a, 'tauc': 1.0 / a}
    twin = _cc(model, par, ell=1, ext=(('x', 1), ('x', 1)), taus=_TAUS / a)
    assert _codes(twin) == _codes(base)
    _assert_twin(base, twin, 1.0, 1e-11)


def _spiking(s):
    """A field driving an exponential-rate spike train through a kernel, the
    field in units ``s`` (the saddle of the field is negative: reached from
    the default non-negative seeds).  Spike train n; x -> s x."""
    return (SDE(f'w4a_spk').parameter('qa', default=-1.59).parameter('qf', default=-0.487 * 1.0)
            .parameter('qo', default=-0.170 * s).parameter('qk', default=0.898)
            .parameter('ql', default=0.131 / s).parameter('qe', default=0.9)
            .parameter('qn', default=1.35).parameter('qj', default=-0.578 * s)
            .field('xa')
            .define_kernel('g0', freq_image='1/(1 + I*omega*qe)')
            .define_kernel('g1', freq_image='1/(1 + I*omega*qn)')
            .point_process('sp', 'qk*exp(ql*xa)')
            .drift('xa', 'qa*xa + qf*Conv(g0, xa) + qo*Conv(g1, sp)')
            .noise('nz', Gaussian(var='0.3'), couples={'xa': 'qj'}).build())


@pytest.mark.parametrize('s', (1e-6, 1e6))
def test_spiking_field_twin(s):
    """Field units 1e-6 found no saddle (G2_NO_SADDLE: the seed box scaled with
    the largest parameter, 1/s, and the root sits at -0.07 s)."""
    ext = (('xa', 1), ('sp', 1))
    base = _cc(_spiking(1.0), ell=1, ext=ext)
    twin = _cc(_spiking(s), ell=1, ext=ext)
    assert twin['mf_values']['xastar'][0] == pytest.approx(s * base['mf_values']['xastar'][0],
                                                           rel=1e-14)
    assert _codes(twin) == _codes(base)
    _assert_twin(base, twin, s, 1e-12)


def _gamma_two_field(s):
    """Two coupled fields with a Gamma-process (log CGF) and a Gaussian noise,
    both fields in units ``s``."""
    return (SDE('w4a_gam').parameter('qa', default=-1.85).parameter('qb', default=0.261)
            .parameter('qd', default=0.170 / s).parameter('qf', default=-1.62)
            .parameter('qg', default=-0.0874 * s).parameter('qh', default=-0.0323 / s**2)
            .parameter('ql', default=0.206).parameter('qm', default=-0.587 * s)
            .parameter('qk', default=1.05).parameter('qq', default=0.435 * s)
            .parameter('qr', default=0.559 * s)
            .field('xa').field('xb')
            .drift('xa', 'qa*xa + qb*xb + qd*xb*xa').drift('xb', 'qf*xb + qg + qh*xb^3')
            .noise('gp', GammaProcess(shape_rate='qk', scale='ql'), couples={'xb': 'qm'})
            .noise('gs', Gaussian(var='0.3'), couples={'xa': 'qq', 'xb': 'qr'}).build())


@pytest.mark.parametrize('s', (1e-6, 1e-3, 1e3, 1e6))
def test_gamma_noise_field_twin(s):
    """Field units 1e+-3 and 1e6 were refused by the absolute 1e-9 mean-field
    sector check (UNIT-01), 1e-6 accepted a saddle wrong at 3e-5 (the 1e-7
    absolute root acceptance)."""
    ext = (('xb', 1), ('xa', 1))
    base = _cc(_gamma_two_field(1.0), ell=1, ext=ext)
    twin = _cc(_gamma_two_field(s), ell=1, ext=ext)
    for v in ('xastar', 'xbstar'):
        assert twin['mf_values'][v][0] == pytest.approx(s * base['mf_values'][v][0], rel=1e-14)
    assert _codes(twin) == _codes(base)
    _assert_twin(base, twin, s * s, 1e-12)


# ── 3. analyzer marginality (UNIT-03) and G2-K Lambert W (UNIT-04) ───────────────────────────

def _ou_slow(a):
    return (SDE('w4a_ouslow').parameter('mu', default=a).parameter('eps', default=0.02 * a)
            .parameter('D', default=a).field('x').drift('x', '-mu*x - eps*x^3')
            .noise('xi', Gaussian(var='2*D'), couples={'x': 1}).build())


@pytest.mark.parametrize('a', (1e-12, 1e-9, 1e-6, 1e6, 1e9))
def test_analyzer_verdict_is_time_unit_invariant(a):
    """Every rate times ``a``: the analyzer's codes are the codes at a = 1 (a
    floor of 1.0 on the eigenvalue scale called a = 1e-9 marginal)."""
    from api.analyze import analyze
    codes = lambda m: sorted(f.code for f in _quiet(analyze, m, use_cache=False).findings
                             if f.level in ('error', 'warning'))
    assert codes(_ou_slow(a)) == codes(_ou_slow(1.0))


@pytest.mark.parametrize('a', (1e-12, 1e-9, 1.0, 1e6, 1e12))
def test_classify_eigs_is_scale_free(a):
    from api.analyze import _classify_eigs
    for eigs in ([-1.0], [-1.0, -3.0 + 2j, -3.0 - 2j], [0.0, -1.0], [2.0, -1.0],
                 [1e-12j - 1e-11, -1.0]):
        assert _classify_eigs(np.asarray(eigs) * a) == _classify_eigs(np.asarray(eigs))


def _delay_K(alpha, beta, gamma, X):
    from sage.all import SR, I, exp, matrix
    om = SR.var('omega')
    return matrix(SR, 1, 1, [alpha * I * om + beta + gamma * exp(-I * om * X)]), om


@pytest.mark.parametrize('a', (1e-9, 1e-6, 1.0, 1e3, 1e9))
def test_lambert_w_branch_is_time_unit_invariant(a):
    """``K(s) = s + b + g e^{-s d}`` in time units ``a`` (``b, g`` times ``a``,
    ``d / a``): same status, same number of right-half-plane roots, roots
    times ``a``, and a branch count that does not grow with ``a`` (it was
    8e7 at a 1e9 rescaling)."""
    from api._kernel_stability import kernel_stability, lambert_roots
    for b, g, d in ((1.0, -2.0, 0.5), (0.5, 0.3, 1.0), (-1.0, 0.5, 2.0), (1e-10, 1.5e-10, 1.0)):
        K0, om = _delay_K(1.0, b, g, d)
        K1, _ = _delay_K(1.0, b * a, g * a, d / a)
        s0, s1 = kernel_stability(K0, om, {}), kernel_stability(K1, om, {})
        assert (s1.status, s1.n_rhp) == (s0.status, s0.n_rhp)
        if s0.rightmost is not None:
            assert s1.rightmost == pytest.approx(a * s0.rightmost, rel=1e-9, abs=1e-12 * a)
        assert len(lambert_roots(1.0, b * a, g * a, d / a)) == len(lambert_roots(1.0, b, g, d))


def test_lambert_w_unstable_outranks_marginal():
    """An exact axis root at +-7i next to real roots at 8.03 and 0.32: the
    characteristic function is unstable, as the argument principle says (it
    was reported 'marginal')."""
    from api._kernel_stability import kernel_stability
    b, g = -7 / math.tan(7), 7 / math.sin(7)
    K, om = _delay_K(1.0, b, g, 1.0)
    st = kernel_stability(K, om, {})
    assert st.status == 'unstable' and st.n_rhp == 2
    assert sorted(z.real for z in st.roots) == pytest.approx([0.3236141308572, 8.029136338906],
                                                            rel=1e-10)


# ── 4. outputs: the delta-contact rule (GEN-14), numeric_inverse_transform (UNIT-06) ────────

@pytest.fixture(scope='module')
def hawkes_cumulant():
    from api.cumulant import Cumulant
    m = (SDE('w4a_c0').parameter('Em', default=0.5).parameter('tau', default=2.0)
         .parameter('a', default=1.0).parameter('w', default=0.4)
         .define_function('phi', args=['v'], expression='a*v')
         .noise('spikes', Poisson(rate='phi(v)'), expose='n')
         .field('v').equation('v', lhs='(tau*Dt + 1)*v', rhs='Em + w*n')).build()
    res = _cc(m, ell=1, ext=(('n', 1), ('n', 1)), taus=np.linspace(-5, 5, 11))
    return Cumulant.from_result(res)


@pytest.mark.parametrize('scale', (1.0, 1e-6, 1e-15, 1e12))
def test_delta_contact_jump_is_judged_relative_to_the_kernel(hawkes_cumulant, scale):
    """A kernel that jumps at 0 is refused (contact=None) at any amplitude
    (below 1e-14 it was taken for round-off and L(0+) used silently), and
    the explicit 'mid' rule scales with it."""
    from api import symbolic_out as so
    L = so.Kernel.two_sided_exponential(scale, 0.7, -0.5 * scale, 1.3)
    with pytest.raises(ValueError, match='jumps at 0'):
        hawkes_cumulant.integrate_against(L)
    ref = hawkes_cumulant.integrate_against(
        so.Kernel.two_sided_exponential(1.0, 0.7, -0.5, 1.3), contact='mid')
    p = hawkes_cumulant.integrate_against(L, contact='mid')
    assert complex(p) == pytest.approx(scale * complex(ref), rel=1e-12)


@pytest.mark.parametrize('g', (1e-9, 1e-6, 1e-3, 1.0, 1e3, 1e6, 1e9, 1e12))
def test_numeric_inverse_transform_time_twin(g):
    """S(w) = 2D/(w^2 + g^2) + c with D = g, c = 0.5/g: the time-rescaled twin
    of g = 1, C(tau) = exp(-g |tau|).  The contact was estimated at an
    absolute w = 1e9 and the quadrature ran in absolute w: tau = 0 was off by
    100 % for g >= 1e6 and g <= 1e-6, tau = 0.5/g returned 3e307 at 1e-9."""
    from api.symbolic_out import numeric_inverse_transform
    f = lambda w: 2 * g / (w * w + g * g) + 0.5 / g
    for x in (0.0, 0.5, 2.0):
        q = _quiet(numeric_inverse_transform, f, x / g)
        assert q.converged
        assert q.contact == pytest.approx(0.5 / g, rel=1e-12)
        assert q.value.real == pytest.approx(math.exp(-x), rel=1e-11)
        assert abs(q.value.imag) <= 1e-11


# ── 5. noise sources: PhaseType validity (UNIT-05) ───────────────────────────────────────────

@pytest.mark.parametrize('c', (1e-7, 1e-5, 1e-3, 1.0, 1e3, 1e6))
def test_phase_type_rates_in_any_unit(c):
    """An Erlang-3 / Coxian sub-generator at rates times ``c``: accepted (the
    singularity floor refused rates < 1e-4, the row-sum check rates ~1e6, and
    Maxima overflowed at 1e6), with the m-th cumulant of the jump times
    ``c^-m``; a compound Poisson source with a symbolic rate is accepted."""
    alpha = [0.3, 0.7, 0.0]
    T = [[-1.3, 0.3, 0.7], [0.1, -0.7, 0.2], [0.0, 0.4, -1.1]]
    ref = CompoundPoisson(2.0, PhaseType(alpha, T))
    src = CompoundPoisson(2.0, PhaseType(alpha, [[c * x for x in row] for row in T]))
    for m in (1, 2, 3):
        assert float(src.cumulant(m)) == pytest.approx(float(ref.cumulant(m)) / c**m, rel=1e-12)
    CompoundPoisson('lam', PhaseType([1 / 3, 1 / 3, 1 / 3],
                                     [[c * x for x in row] for row in T]))


def test_phase_type_row_sum_round_off():
    a, b = 134364.2441124012, 847433.7369372327          # -(a + b) + a + b = +1.2e-10 in floats
    assert (-(a + b)) + a + b > 1e-12
    PhaseType([1, 0, 0], [[-(a + b), a, b], [0, -1e6, 0], [0, 0, -1e6]])
    with pytest.raises(Exception, match='sums to'):
        PhaseType([1, 0], [[-1, 2], [0, -1]])


def test_compound_poisson_symbolic_rate_with_float_jump_law():
    """K(0) = rate (M(0) - 1) with M(0) = 1 - 1.1e-16 in floats: round-off of
    the dimensionless M(0), not a refusal (it was 'K(0) != 0')."""
    CompoundPoisson('lam', Hyperexponential([1 / 3, 1 / 3, 1 / 3], [1, 2, 3]))
    with pytest.raises(Exception, match='sum to'):
        CompoundPoisson('lam', Hyperexponential([0.3, 0.3, 0.3], [1, 2, 3]))
