"""M8: the convergence certificate's verdicts and the divergence policy.

* A pole-tuple group that fails the certificate (an open direction that does
  not decay: a zero-rate or growing mode) but whose merged coefficient is
  exactly zero is dropped (counter ``divergent_cancelled``), on the interval,
  polygon and chain routes; the result equals that of the modes without it.
* A non-cancelling one is a divergence (``divergent_subsets``, bail
  ``<route>_divergent``, fallback category 'divergent'), and the caller's
  policy decides: 'fallback' warns once and takes the boxed fallback, 'warn'
  names every subset, 'raise' raises ``PhaseJDivergenceError``.
* The spatial certification passes 'fallback' whatever the temporal default.
"""
import math
import warnings

import pytest

from tests.test_phase_j_box_free import (_CHAIN_ROWS, _WEDGE_ROWS,
                                         _ems, _set_flags)


@pytest.fixture
def FI():
    pytest.importorskip('sage.all')
    import engine.integration.time_domain.final_integral as fi
    return fi


def _counters(fi):
    c = fi._RUNTIME_COUNTERS
    return {k: c[k] for k in ('divergent_subsets', 'divergent_cancelled')}


def _call(fi, fn, *args):
    fi._reset_runtime_counters()
    fi._pop_bail_reason()
    v = fn(fi, *args)
    return v, fi._pop_bail_reason(), _counters(fi)


# ═══════════════════════════════════════════════════════════════════════
# Exactly cancelling groups are dropped; the others are divergences
# ═══════════════════════════════════════════════════════════════════════

# m = 1: s < t only (L = -inf on the box-free route); λ = 0 is a zero-rate
# mode, which never decays toward the past.
_HALF_LINE = [((-1.0,), (1.0,), 0.0)]


def _interval(fi, modes, plan):
    ems = [_ems(fi, md) for md in modes]
    p = fi._build_modesum_plan(ems, _HALF_LINE, 1, 1) if plan else None
    return fi._integrate_1d_polytope_modesum(
        ems, 0.7 - 0.1j, _HALF_LINE, [0.4],
        pole_tuples=None if p is None else p['pole_tuples'], plan=p)


@pytest.mark.parametrize('plan', [False, True])
def test_interval_zero_rate_modes(FI, monkeypatch, plan):
    _set_flags(monkeypatch, FI)
    base = [[(1.0, -0.8 + 0.3j)]]
    v0, r0, c0 = _call(FI, _interval, base, plan)
    assert r0 is None and c0 == {'divergent_subsets': 0,
                                 'divergent_cancelled': 0}
    # a cancelling pair at λ = 0: dropped, the value is unchanged
    v, r, c = _call(FI, _interval,
                    [base[0] + [(0.5, 0.0), (-0.5, 0.0)]], plan)
    assert r is None and c == {'divergent_subsets': 0,
                               'divergent_cancelled': 1}
    assert v == v0
    # not cancelling: a divergence
    v, r, c = _call(FI, _interval,
                    [base[0] + [(0.5, 0.0), (-0.3, 0.0)]], plan)
    assert v is None and r == 'interval_divergent'
    assert c == {'divergent_subsets': 1, 'divergent_cancelled': 0}
    assert FI._bail_category(r) == 'divergent'


def _polygon(fi, modes, plan):
    ems = [_ems(fi, md) for md in modes]
    if plan:
        p = fi._build_modesum_plan(ems, _WEDGE_ROWS, 2, 1)
        return fi._integrate_2d_polygon_modesum(
            ems, 1.0, _WEDGE_ROWS, [0.7], pole_tuples=p['pole_tuples'],
            plan=p)
    return fi._integrate_2d_polygon_modesum(ems, 1.0, _WEDGE_ROWS, [0.7])


# The wedge s1 < s0 < t has the ray (0, -1), on which α·r = λ1 + λ2: with
# λ2 = -0.6 + 1j, λ1 = 0.6 - 1j makes it exactly 0 (a zero-rate direction).
_W0, _W2 = [(1.0, -1.3 + 0.4j)], [(1.0, -0.6 + 1j)]


@pytest.mark.parametrize('plan', [False, True])
def test_polygon_zero_rate_direction(FI, monkeypatch, plan):
    _set_flags(monkeypatch, FI)
    base = [(0.8 - 0.2j, -0.9)]
    v0, r0, c0 = _call(FI, _polygon, [_W0, base, _W2], plan)
    assert r0 is None and c0['divergent_subsets'] == 0
    v, r, c = _call(FI, _polygon,
                    [_W0, base + [(0.5, 0.6 - 1j), (-0.5, 0.6 - 1j)], _W2],
                    plan)
    assert r is None and c == {'divergent_subsets': 0,
                               'divergent_cancelled': 1}
    assert abs(v - v0) <= 1e-15 * abs(v0)
    v, r, c = _call(FI, _polygon,
                    [_W0, base + [(0.5, 0.6 - 1j), (-0.25, 0.6 - 1j)], _W2],
                    plan)
    assert v is None and r == 'polygon_divergent'
    assert c == {'divergent_subsets': 1, 'divergent_cancelled': 0}


def _poset(fi, modes, plan):
    ems = [_ems(fi, md) for md in modes]
    if plan:
        p = fi._build_modesum_plan(ems, _CHAIN_ROWS, 3, 1)
        return fi._integrate_nd_polytope_poset_modesum(
            ems, 1.0, _CHAIN_ROWS, [0.7], 3, pole_tuples=p['pole_tuples'],
            plan=p)
    return fi._integrate_nd_polytope_poset_modesum(ems, 1.0, _CHAIN_ROWS,
                                                   [0.7], 3)


@pytest.mark.parametrize('plan', [False, True])
def test_chain_zero_rate_modes(FI, monkeypatch, plan):
    """The chain fold merges each α-group before evaluating its chain, so
    an exactly cancelling group never reaches the certificate."""
    _set_flags(monkeypatch, FI)
    base = [[(1.0, -2.0 + 1j)], [(0.8, -1.5)], [(1.0, -3.0)]]
    v0, r0, _c = _call(FI, _poset, base, plan)
    assert r0 is None
    # λ = 0 on the first edge (s1 - s0): prefix sum P_1 = -λ_0 = 0
    zero = [base[0] + [(0.5, 0.0), (-0.5, 0.0)]] + base[1:]
    v, r, c = _call(FI, _poset, zero, plan)
    assert r is None and c['divergent_subsets'] == 0
    assert abs(v - v0) <= 1e-15 * abs(v0)
    zero = [base[0] + [(0.5, 0.0), (-0.25, 0.0)]] + base[1:]
    v, r, c = _call(FI, _poset, zero, plan)
    assert v is None and r == 'poset_divergent'
    assert c['divergent_subsets'] == 1


def test_bail_categories(FI):
    cat = FI._bail_category
    for route in ('polygon', 'interval', 'poset', 'grouped_m0',
                  'grouped_m1'):
        assert cat(f'{route}_divergent') == 'divergent'
        assert cat(f'{route}_dd_overflow') == 'dd_overflow'
        assert cat(f'{route}_dd_unsupported') == 'dd_unsupported'
    assert cat('dbm_divergent') == 'divergent'
    assert cat('polygon_gamma_overflow') == 'polygon_other'
    assert cat('polygon_triangle_guard') == 'polygon_guard'
    assert cat('poset_gamma_overflow') == 'other'
    assert cat(None) == 'other'
    assert {'divergent', 'dd_overflow', 'dd_unsupported'} <= set(
        FI._NQUAD_FALLBACK_REASONS)
    assert FI._DIVERGENT_BAIL_REASONS == {
        f'{r}_divergent' for r in ('polygon', 'interval', 'poset',
                                   'grouped_m0', 'grouped_m1', 'dbm')}


# ═══════════════════════════════════════════════════════════════════════
# The policy
# ═══════════════════════════════════════════════════════════════════════

def test_policy_flag_and_keyword_validation(FI, monkeypatch):
    assert FI._initial_on_divergence({}) == 'fallback'
    for v in ('fallback', 'warn', 'raise', ' RAISE '):
        assert FI._initial_on_divergence(
            {'DAEDALUS_PHASE_J_ON_DIVERGENCE': v}) == v.strip().lower()
    with pytest.raises(ValueError):
        FI._initial_on_divergence({'DAEDALUS_PHASE_J_ON_DIVERGENCE': 'drop'})
    assert FI.PHASE_J_ON_DIVERGENCE == 'fallback'          # the default
    monkeypatch.setattr(FI, 'PHASE_J_ON_DIVERGENCE', 'warn')
    assert FI._divergence_policy(None) == 'warn'
    assert FI._divergence_policy('raise') == 'raise'
    monkeypatch.setattr(FI, 'PHASE_J_ON_DIVERGENCE', 'Raise')
    with pytest.raises(ValueError):
        FI._divergence_policy(None)
    with pytest.raises(ValueError):
        FI.integrate_diagram(None, None, None, None, on_divergence='drop')


def _verdict(fi, policy):
    meta = {'diagram': 'D', 'source': 'per_diagram', 'loop_number': 1,
            'subset_id': (0, 1)}
    fi._divergence_verdict(policy, 'polygon_divergent', 2, meta, (0.0, 1.5))


def test_policy_verdicts(FI, monkeypatch):
    monkeypatch.setattr(FI, '_DIVERGENCE_FALLBACK_WARNED', [False])
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('default')
        _verdict(FI, 'fallback')
        _verdict(FI, 'fallback')                  # once per process
        assert len(w) == 1
        assert issubclass(w[0].category, FI.PhaseJDivergenceWarning)
        _verdict(FI, 'warn')
        assert len(w) == 2 and 'subset (0, 1)' in str(w[1].message)
        assert 'polygon_divergent' in str(w[1].message)
    with pytest.raises(FI.PhaseJDivergenceError,
                       match=r'polygon_divergent.*subset \(0, 1\).*1\.5'):
        _verdict(FI, 'raise')


def _diverge_every_interval(FI, monkeypatch):
    """Every m = 1 subset on the DD route reports
    a divergence: the code under test is the dispatch, not the
    certificate (``ou_quartic`` has m = 1 and m >= 3 regions only)."""
    bf = FI._bf()
    monkeypatch.setattr(
        FI, '_interval_dd_total',
        lambda terms, pref, L, U, prefix, *a, **k: FI._dd_status_bail(
            bf.DIVERGENT, prefix))


def _ou_k2(FI, **kw):
    import tests.tools.phase_j_zoo_baseline as Z
    import daedalus as dd
    from api import compute_cumulants
    import numpy as np
    model, _mod = dd.load_model('ou_quartic')
    return compute_cumulants(
        model, k=2, max_ell=1, external_fields=[('dx', 1), ('dx', 1)],
        parameters=dict(Z.P_OU), tau_grid=np.array([0.0, 0.5]),
        parallel=False, verbose=False, **kw)


def test_policy_end_to_end(FI, monkeypatch):
    """The dispatch consults the policy before the boxed fallback, on the
    per-diagram path: 'raise' raises, 'fallback' gives the
    fallback's value with the counter and the category."""
    _set_flags(monkeypatch, FI)
    _diverge_every_interval(FI, monkeypatch)
    monkeypatch.setattr(FI, 'PHASE_J_ON_DIVERGENCE', 'raise')
    with pytest.raises(FI.PhaseJDivergenceError, match='interval_divergent'):
        _ou_k2(FI)
    monkeypatch.setattr(FI, 'PHASE_J_ON_DIVERGENCE', 'fallback')
    monkeypatch.setattr(FI, '_DIVERGENCE_FALLBACK_WARNED', [False])
    FI._reset_runtime_counters()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('default')
        res = _ou_k2(FI)
    assert sum(issubclass(x.category, FI.PhaseJDivergenceWarning)
               for x in w) == 1
    c = FI._RUNTIME_COUNTERS
    n = c['nquad_fallback_by_reason']['divergent']
    assert n > 0 and c['divergent_subsets'] == n
    assert all(math.isfinite(abs(complex(x))) for x in res['C_tau'])


def test_spatial_certification_keeps_fallback(FI, monkeypatch):
    """``pipeline_bridge`` passes on_divergence='fallback' explicitly, so a
    temporal default of 'raise' does not reach the spatial certification."""
    import engine.integration.spatial.pipeline_bridge as pb
    import engine.integration.time_domain.pipeline as pl
    from api import _propagator
    seen = []

    def spy(*a, **k):
        seen.append(k.get('on_divergence'))
        raise RuntimeError('stop')
    monkeypatch.setattr(pl, 'compute_correction_td', spy)
    monkeypatch.setattr(_propagator, 'compute_poles_and_residues',
                        lambda *a, **k: None)
    monkeypatch.setattr(FI, 'PHASE_J_ON_DIVERGENCE', 'raise')
    prop = {key: None for key in (
        'K_ker', 'K_ft', 'G_ft', 'adj_ft', 'D_omega', 'D_delta', 't_var',
        'omega', 'nf', 'pole_vals', 'C_mats')}
    with pytest.raises(RuntimeError, match='stop'):
        pb.pipeline_C_q_tau(prop, [], [], {}, 0.0, [0.0])
    assert seen == ['fallback']


@pytest.mark.parametrize('plan', [False, True])
def test_duplicated_alphas_that_do_not_cancel_are_kept(FI, monkeypatch, plan):
    """Adversarial for the cancellation rule: repeated poles (the same α in
    several tuples) whose merged coefficient is NOT zero are integrated, on
    every route; two such groups interleaved in tuple order with a
    cancelling one."""
    _set_flags(monkeypatch, FI)
    split = [(0.5, -1.1), (0.3, -1.1), (0.25, 0.0), (-0.25, 0.0)]
    merged = [(0.8, -1.1)]
    v1, r1, c1 = _call(FI, _poset, [[(1.0, -2.0 + 1j), (0.4, -1.7)] + [],
                                    split, [(1.0, -3.0)]], plan)
    v2, r2, _c = _call(FI, _poset, [[(1.0, -2.0 + 1j), (0.4, -1.7)],
                                    merged, [(1.0, -3.0)]], plan)
    assert r1 is None and r2 is None and c1['divergent_subsets'] == 0
    assert abs(v1 - v2) <= 1e-14 * abs(v2)
    v1, r1, c1 = _call(FI, _interval, [[(1.0, -0.8)] + split], plan)
    v2, r2, _c = _call(FI, _interval, [[(1.0, -0.8)] + merged], plan)
    assert r1 is None and c1 == {'divergent_subsets': 0,
                                 'divergent_cancelled': 1}
    assert abs(v1 - v2) <= 1e-14 * abs(v2)


def test_a_divergence_found_by_the_dbm_reaches_the_policy(FI, monkeypatch):
    """The chain route can bail for another reason (here: rows it cannot
    order) and leave the divergence verdict to the exact DBM route; that
    verdict goes through the policy too."""
    _set_flags(monkeypatch, FI)
    monkeypatch.setattr(FI, '_integrate_nd_polytope_poset_modesum',
                        lambda *a, **k: FI._bail('poset_extract_none'))
    monkeypatch.setattr(FI, '_integrate_subset_dbm',
                        lambda *a, **k: FI._bail('dbm_divergent'))
    monkeypatch.setattr(FI, 'PHASE_J_ON_DIVERGENCE', 'raise')
    with pytest.raises(FI.PhaseJDivergenceError, match='dbm_divergent'):
        _ou_k2(FI)


def test_the_divergence_verdicts(FI, monkeypatch):
    """The interval route reports a non-decaying open side as
    'interval_divergent', a divergence verdict for the policy.  (The retired
    boxed routes had their own reasons, 'interval_divergent_flat' / ...;
    they went with those routes in audit W7.)"""
    _set_flags(monkeypatch, FI)
    v, r, _c = _call(FI, _interval, [[(1.0, 0.0)]], False)
    assert v is None and r == 'interval_divergent'
    assert FI._is_divergence_verdict(r)
    for reason in ('polygon_divergent', 'poset_divergent', 'dbm_divergent'):
        assert FI._is_divergence_verdict(reason)
    for reason in ('polygon_triangle_guard', 'interval_divergent_flat',
                   'poset_extract_none'):
        assert not FI._is_divergence_verdict(reason)


# |s0 - s1| < 1: a strip, i.e. a region containing a line
_STRIP_ROWS = [((1.0, -1.0), (0.0,), 1.0), ((-1.0, 1.0), (0.0,), 1.0)]


def _strip(fi, modes):
    ems = [_ems(fi, md) for md in modes]
    return fi._integrate_2d_polygon_modesum(ems, 1.0, _STRIP_ROWS, [0.0])


def test_a_region_with_a_line_cancels_or_diverges(FI, monkeypatch):
    """Every exponent diverges along a line: the integral is 0 when every
    pole-tuple group cancels exactly, otherwise a divergence (review)."""
    _set_flags(monkeypatch, FI)
    v, r, c = _call(FI, _strip, [[(0.5, -1.0), (-0.5, -1.0)],
                                 [(1.0, -2.0)]])
    assert v == 0 and r is None
    assert c == {'divergent_subsets': 0, 'divergent_cancelled': 1}
    v, r, c = _call(FI, _strip, [[(0.5, -1.0), (-0.25, -1.0)],
                                 [(1.0, -2.0)]])
    assert v is None and r == 'polygon_divergent'
    assert c['divergent_subsets'] == 1


def test_the_policy_is_validated_at_every_entry_point(FI):
    from engine.integration.time_domain.pipeline import compute_correction_td
    with pytest.raises(ValueError, match='on_divergence'):
        compute_correction_td(on_divergence='drop')


# S3: the state-space route is the default; these tests pin the pole
# engine's own routes / counters / plans, so the module runs with
# ``DAEDALUS_PHASE_J_ENGINE=poles`` (module scope: the module's fixtures that
# compute results run on the pole engine too).
@pytest.fixture(autouse=True, scope='module')
def _pole_engine_s3():
    with pytest.MonkeyPatch.context() as mp_:
        mp_.setenv('DAEDALUS_PHASE_J_ENGINE', 'poles')
        yield
