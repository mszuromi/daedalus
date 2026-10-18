"""
Near-degenerate and repeated poles (``api._pole_clusters``).

Phase J sums pole-residue terms; residues ~ 1/gap of a close pole cluster
cancel in those sums (about E log10(1/gap) digits lost, E cluster-carrying
lines) and a repeated pole (gap 0) cannot be written that way at all.
``compute_cumulants`` now averages Phase J over a deformation that splits each
ill-conditioned cluster on a circle (Cauchy mean value, trapezoid rule).

References are exact: the colored OU (``ou_quartic_colored``: x' = -mu x
- eps x^3 + eta, <eta eta> = (2 D / tauc) e^{-|tau|/tauc}) has the tree
``C0 = A/(mu^2 - g^2) (e^{-g tau} - (g/mu) e^{-mu tau})`` (A = 2 D / tauc,
g = 1/tauc) and, its only nonlinearity being eps x^3, the one-loop term
``C1 = 3 eps C0(0) dC0/dmu`` (the Hartree tadpole); the three-pole chain
x <- y <- z likewise (spectral density 2 D / prod (omega^2 + rate^2)).
The tau = 0 point is compared at tau = 0, the exact Ito left limit the pipeline
evaluates (C is even and continuous here).  Before the fix the colored OU's one-loop term was noise of order
1 at gap 1e-4 and the tree was 1.40625 at gap 1e-7 and -2e15 at gap 0.
"""
from __future__ import annotations

import importlib.util
import os
import sys
import warnings
from pathlib import Path

import mpmath as mp
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from sage.all import QQ, SR, matrix                                 # noqa: E402

from api import compute_cumulants                                   # noqa: E402
from api import _pole_clusters as PC                                # noqa: E402
from api import _propagator as PR                                   # noqa: E402
from api.model import TemporalModelBuilder                          # noqa: E402

MODELS = Path(__file__).resolve().parent.parent / 'models'
TAU = np.array([0.0, 0.5, 2.0])
TAU_REF = ['0', '0.5', '2']
GAPS = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-10, 0.0]
RTOL = 1e-9


def _load(name):
    spec = importlib.util.spec_from_file_location(name, MODELS / f'{name}.model.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.build()


_MARKOV = {'allow_markov_embedding': True}


def _run(model, params, tau=TAU, **kw):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        r = compute_cumulants(model, k=2, max_ell=1, external_fields=[('x', 1), ('x', 1)], parallel=False,
                              verbose=False, tau_grid=np.asarray(tau, dtype=float), parameters=params, **kw)
    r['_pole_warnings'] = [w for w in caught if issubclass(w.category, PC.PoleClusterWarning)]
    return r


def _real(arr):
    return np.array([complex(v).real for v in arr])


def _hartree(C0, mu, eps, taus):
    """Tree and one-loop (3 eps C0(0) dC0/dmu) from ``C0(t, mu)`` at 90 digits."""
    with mp.workdps(90):
        c00 = C0(mp.mpf(0), mp.mpf(mu))
        tree = [C0(mp.mpf(t), mp.mpf(mu)) for t in taus]
        loop = [3 * mp.mpf(eps) * c00 * mp.diff(lambda m: C0(mp.mpf(t), m), mp.mpf(mu), h=mp.mpf(10) ** -15)
                for t in taus]
    return np.array([float(x) for x in tree]), np.array([float(x) for x in loop])


def _colored_ref(tauc, mu=1.0, eps=0.02, D=1.0, taus=TAU_REF):
    def C0(t, m):
        g = 1 / mp.mpf(tauc)
        A = 2 * mp.mpf(D) / mp.mpf(tauc)
        if m == g:                     # the removable singularity: a hair off it (90 digits)
            m = g + mp.mpf(10) ** -40
        return A / (m ** 2 - g ** 2) * (mp.e ** (-g * t) - (g / m) * mp.e ** (-m * t))
    return _hartree(C0, mu, eps, taus)


def _chain_model():
    b = TemporalModelBuilder('Three pole chain test')
    for f in ('x', 'y', 'z'):
        b.physical_field(f)
    for p, d in (('mu', 1.0), ('g1', 1.0), ('g2', 1.0), ('eps', 0.02), ('D', 1.0)):
        b.parameter(p, default=d, domain='positive')
    b.set_action_text('xt*((Dt+mu)*x - y + eps*x^3) + yt*((Dt+g1)*y - z) + zt*((Dt+g2)*z) - D*zt^2')
    b.equation(lhs='(Dt+mu)*x', rhs='y - eps*x^3')
    b.equation(lhs='(Dt+g1)*y', rhs='z')
    b.equation(lhs='(Dt+g2)*z', rhs='0')
    b.stability_analysis(True)
    return b.build()


def _chain_ref(mu, g1, g2, eps=0.02, D=1.0, taus=TAU_REF):
    def C0(t, m):
        rates = [m, mp.mpf(g1), mp.mpf(g2)]
        for i in range(3):                       # split exact repeats by 1e-40 (90 digits)
            for j in range(i):
                if rates[i] == rates[j]:
                    rates[i] += mp.mpf(10) ** -40 * (i + 1)
        tot = 0
        for a in rates:
            p = 2 * a
            for b in rates:
                if b is not a:
                    p *= (b ** 2 - a ** 2)
            tot += mp.e ** (-a * t) / p
        return 2 * mp.mpf(D) * tot
    return _hartree(C0, mu, eps, taus)


def _params(gap):
    return {'mu': 1.0, 'eps': 0.02, 'D': 1.0, 'tauc': 1 / (1 + gap)}


@pytest.fixture(scope='module')
def colored():
    model = _load('ou_quartic_colored')
    return {gap: _run(model, _params(gap), policy=_MARKOV) for gap in GAPS}


# ── A: the colored OU against its exact reference ──────────────────────────────────────────────

@pytest.mark.parametrize('gap', GAPS)
def test_colored_ou_matches_the_exact_reference(colored, gap):
    r = colored[gap]
    tree_ref, loop_ref = _colored_ref(1 / (1 + gap))
    np.testing.assert_allclose(_real(r['C_tau_by_ell'][0]), tree_ref, rtol=1e-10, atol=0)
    np.testing.assert_allclose(_real(r['C_tau_by_ell'][1]), loop_ref, rtol=RTOL, atol=0)
    pc = r['pole_clusters']
    if gap <= 5e-2:                    # the pair is a cluster (5 % relative) with kappa > 10
        assert pc['status'] == 'resolved' and pc['kappa_max'] > PC.KAPPA_MAX, pc
        assert pc['accuracy'] < RTOL and pc['node_runs'] == pc['n_nodes'] // 2 + 1, pc
    else:
        assert pc is None
    assert not r['_pole_warnings']
    assert not r['analysis'].has('G6_ACCURACY_LOW') and not r['analysis'].has('G3_POLE_CLUSTER_UNRESOLVED')


def test_colored_ou_is_continuous_in_the_gap(colored):
    """Between successive gaps the pipeline moves exactly as the reference does: no jump at the cluster
    threshold, and gap -> 0 reaches the exact double-pole value."""
    prev = None
    for gap in GAPS:
        cur = np.concatenate([_real(colored[gap]['C_tau_by_ell'][0]), _real(colored[gap]['C_tau_by_ell'][1])])
        ref = np.concatenate(_colored_ref(1 / (1 + gap)))
        if prev is not None:
            np.testing.assert_allclose(cur - prev[0], ref - prev[1], rtol=0, atol=RTOL * np.max(np.abs(ref)))
        prev = (cur, ref)


def test_exact_repeated_pole_is_the_limit(colored):
    """tauc = 1/mu: Q(omega) = (i omega + 1)^2.  Finite, equal to the gap -> 0 limit; the nominal residues of
    a Jordan block never reach Phase J directly."""
    r = colored[0.0]
    assert r['pole_clusters']['repeated_pole'] and r['pole_clusters']['status'] == 'resolved'
    assert r['analysis'].has('G3_REPEATED_POLE') and not r['analysis'].warnings
    np.testing.assert_allclose(_real(r['C_tau_by_ell'][0]), _real(colored[1e-10]['C_tau_by_ell'][0]),
                               rtol=1e-9)
    assert abs(complex(r['C_tau_by_ell'][0][0]) - 1.0) < 1e-12        # Var x = 1/mu at tauc = 1/mu
    assert abs(complex(r['C_tau_by_ell'][1][0]) + 0.09) < 1e-12       # 3 eps C0(0) dC0/dmu = -0.09
    from engine.integration.time_domain import compute_correction_td
    with pytest.raises(ValueError, match='repeated pole'):
        compute_correction_td(typed_diagrams=[r['diagrams'][0]['typed_diagram']], prefactors=[SR(1)], k=2,
                              propagator_data=r['propagator'])


def test_long_tau_grid_keeps_its_accuracy():
    """The deformation radius shrinks with the grid's extent (h <= 3/tau_max): the values stay at the
    reference to 1e-9 relative up to tau = 50 (C(50) ~ 1e-21)."""
    taus = [0.0, 2.0, 10.0, 30.0, 50.0]
    r = _run(_load('ou_quartic_colored'), _params(1e-6), tau=taus, policy=_MARKOV)
    tree_ref, loop_ref = _colored_ref(1 / (1 + 1e-6), taus=['0'] + [str(t) for t in taus[1:]])
    np.testing.assert_allclose(_real(r['C_tau_by_ell'][0]), tree_ref, rtol=1e-10, atol=0)
    np.testing.assert_allclose(_real(r['C_tau_by_ell'][1]), loop_ref, rtol=RTOL, atol=0)
    pc = r['pole_clusters']
    assert pc['status'] == 'resolved' and pc['clusters'][0]['h'] <= 3 / 50 + 1e-15
    assert pc['accuracy'] < 1e-12 and max(pc['accuracy_by_tau']) < 1e-8     # pointwise: C(30) ~ 3e-12


# ── three clustered poles (a Jordan block of size 3 at d = 0) ──────────────────────────────────

@pytest.mark.parametrize('d', [1e-3, 1e-8, 1e-11, 1e-13, 0.0])
def test_three_pole_cluster(d):
    """Includes gaps where double-precision root membership fails (d = 1e-11: the entries' own roots were
    missed, their residues dropped): membership is exact (a coprime base in QQ[i][lambda])."""
    par = {'mu': 1.0, 'g1': 1.0 + d, 'g2': 1.0 + 2 * d, 'eps': 0.02, 'D': 1.0}
    r = _run(_chain_model(), par)
    tree_ref, loop_ref = _chain_ref(1.0, 1.0 + d, 1.0 + 2 * d)
    np.testing.assert_allclose(_real(r['C_tau_by_ell'][0]), tree_ref, rtol=RTOL, atol=0)
    np.testing.assert_allclose(_real(r['C_tau_by_ell'][1]), loop_ref, rtol=RTOL, atol=0)
    pc = r['pole_clusters']
    assert pc['status'] == 'resolved' and pc['repeated_pole'] == (d == 0.0)
    assert [len(c['poles']) for c in pc['clusters'] if c['deformed']] == [1 if d == 0.0 else 3]
    assert not r['_pole_warnings']


def test_decoupled_near_equal_rates_are_exact():
    """Two uncoupled OU processes, rates 1 and 1 + g: a harmless cluster (no cancellation) that the old tier-1
    residue loop mishandled -- it attributes a pole to an entry by |Q_ij(p)| <= 1e-9, so with double-precision
    roots C_xx came out 0 at g <= 1e-8.  The 50-digit residues with exact membership give C_xx = (D/mu1)
    e^{-mu1 tau}."""
    b = TemporalModelBuilder('Two decoupled OU test')
    for f in ('x', 'y'):
        b.physical_field(f)
    for p, d in (('mu1', 1.0), ('mu2', 1.0), ('D', 1.0)):
        b.parameter(p, default=d, domain='positive')
    b.set_action_text('xt*((Dt+mu1)*x) + yt*((Dt+mu2)*y) - D*xt^2 - D*yt^2')
    b.equation(lhs='(Dt+mu1)*x', rhs='0')
    b.equation(lhs='(Dt+mu2)*y', rhs='0')
    b.stability_analysis(True)
    model = b.build()
    for g in (1e-6, 1e-8, 1e-10):
        r = compute_cumulants(model, k=2, max_ell=0, external_fields=[('x', 1), ('x', 1)], parallel=False,
                              verbose=False, tau_grid=np.array([0.0, 1.0]),
                              parameters={'mu1': 1.0, 'mu2': 1.0 + g, 'D': 1.0})
        np.testing.assert_allclose(_real(r['C_tau_by_ell'][0]), [1.0, np.exp(-1.0)], rtol=1e-13)
        assert r['pole_clusters']['status'] == 'harmless'


# ── B: never silent ─────────────────────────────────────────────────────────────────────────────

def _assert_flagged(r, status, min_digits):
    pc = r['pole_clusters']
    assert pc['status'] == status and pc['lost_digits'] >= min_digits, pc
    assert r['_pole_warnings'] and 'digits lost' in str(r['_pole_warnings'][0].message)
    an = r['analysis']
    assert an.has('G3_POLE_CLUSTER_UNRESOLVED') and an.has('G6_ACCURACY_LOW'), an.summary()


def test_cluster_without_the_exact_tier_is_flagged(monkeypatch):
    """Three clustered poles (d = 1e-4) when the exact tier is unavailable (it fails for non-rationalizable
    parameters or on its time budget): the numerical tier's residues cannot be deformed, and the run says
    so with its lost-digit estimate instead of returning silent noise."""
    monkeypatch.setattr(PR, '_compute_residues_via_polynomial_fracfield', lambda *a, **kw: (None, None, None))
    r = _run(_chain_model(), {'mu': 1.0, 'g1': 1.0001, 'g2': 1.0002, 'eps': 0.02, 'D': 1.0})
    _assert_flagged(r, 'unresolved', 10)
    assert 'exact tier failed' in r['pole_clusters']['reason']


def test_switched_off_is_flagged(monkeypatch):
    """DAEDALUS_POLE_CLUSTERS=0 restores the old numbers (the one-loop term is rounding noise at gap 1e-4)
    and says so."""
    monkeypatch.setenv(PC.ENV_SWITCH, '0')
    r = _run(_load('ou_quartic_colored'), _params(1e-4), policy=_MARKOV)
    _assert_flagged(r, 'disabled', 10)
    _, loop_ref = _colored_ref(1 / (1 + 1e-4))
    assert abs(complex(r['C_tau_by_ell'][1][0]).real - loop_ref[0]) > 1e-3


def test_strict_policy_refuses_only_unresolved_clusters(monkeypatch):
    model = _load('ou_quartic_colored')
    strict = {'mode': 'strict', 'allow_markov_embedding': True}
    r = _run(model, _params(1e-4), policy=strict)                    # resolved: runs
    assert r['pole_clusters']['status'] == 'resolved'
    monkeypatch.setenv(PC.ENV_SWITCH, '0')
    from api.analyze import AnalyzerError
    with pytest.raises(AnalyzerError, match='G3_POLE_CLUSTER_UNRESOLVED'):
        _run(model, _params(1e-4), policy=strict)


# ── the analyzer on its own ─────────────────────────────────────────────────────────────────────

def test_analyze_predicts_the_resolution():
    from api.analyze import analyze
    model = _load('ou_quartic_colored')
    rep = analyze(model, question={'parameters': _params(1e-3)})
    assert rep.traits.pole_cluster_status == 'resolved' and rep.traits.pole_cluster_kappa > 100
    f = [f for f in rep.findings if f.code == 'G3_CLOSE_POLES'][0]
    assert f.level == 'info' and 'deformed propagators' in f.message
    assert not rep.has('G3_POLE_CLUSTER_UNRESOLVED')
    rep = analyze(model, question={'parameters': _params(0.3)})      # 23 % apart: no cluster
    assert rep.traits.pole_cluster_status is None and not rep.has('G3_CLOSE_POLES')


# ── Tier 1: a float QQ refuses through SR ───────────────────────────────────────────────────────

def test_tier1_converts_complex_double_backed_floats():
    """``(-2.0 I) * (-1.0 I)``-type products leave SR numbers backed by a ComplexDoubleElement (2.0 + 0i);
    ``QQ`` refused them and the exact tier silently fell back to the numerical one."""
    omega = SR.var('omega')
    K = matrix(SR, [[SR.I() * omega - SR(0.3) / (SR(-2.0) * SR.I() * omega + 1) + SR(1.0)]])
    coeffs = [SR(c).real_part() for c in SR(K[0, 0].numerator()).coefficients(omega, sparse=False)]
    assert [PR._qq(c) for c in coeffs] == [QQ(float(complex(c).real)) for c in coeffs]
    pv, cm, dd = PR._compute_residues_via_polynomial_fracfield(K, omega, {}, 1)
    assert pv is not None and len(pv) == 1          # the exact tier answers (one retarded pole; the kernel's
    p, C = pv[0], complex(cm[0][0, 0])              # own pole 1/(1 - 2 i omega) is anti-causal)
    assert abs(complex(K[0, 0].subs(omega=p))) < 1e-12
    assert abs(C - 1j / complex(K[0, 0].diff(omega).subs(omega=p))) < 1e-12 * abs(C)   # C = i Res 1/K


# ── the combination and its error estimate ──────────────────────────────────────────────────────

def test_trapezoid_combination_and_tail_estimate():
    """F(w) = 1/(1 - w/4) (analytic in |w| < 4): the symmetric and the full node sets give F(0) = 1 with the
    a-priori error 4^-N, and the Fourier-tail estimate bounds it."""
    N = 12
    dummy = PC.PoleDeformation(nf=1, roots=[], retarded=[], entries=[], clusters=[], n_nodes=N,
                               real_system=True, a_priori_error=0.0)
    for sym in (True, False):
        nodes = dummy.nodes(sym)
        vals = [np.array([1 / (1 - w / 4)]) for w, _, _ in nodes]
        F0 = PC.combine_values(vals, nodes)[0]
        err = PC.tail_estimate(vals, nodes, N)[0]
        assert abs(F0 - 1) == pytest.approx(4.0 ** -N / (1 - 4.0 ** -N), rel=1e-12, abs=0)
        assert err >= abs(F0 - 1)
    assert len(dummy.nodes(True)) == N // 2 + 1 and len(dummy.nodes(False)) == N


# S3: the state-space route is the default; these tests pin the pole
# engine's own routes / counters / plans, so the module runs with
# ``DAEDALUS_PHASE_J_ENGINE=poles`` (module scope: the module's fixtures that
# compute results run on the pole engine too).
@pytest.fixture(autouse=True, scope='module')
def _pole_engine_s3():
    with pytest.MonkeyPatch.context() as mp_:
        mp_.setenv('DAEDALUS_PHASE_J_ENGINE', 'poles')
        yield
