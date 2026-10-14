"""Item 35: exact-zero residue modes left out of the Phase J modesum plans
(F1), zero-coefficient groups skipped by the batched poset kernels (F2), the
k >= 3 slices of ``dd.run`` formed from the per-ell slices (F3), the Phase J
memory budget ``PHASE_J_MEM_BUDGET_GB`` and pure-integrator mean-field
equations.

The model is a recharge oscillator (sea-surface temperature T, thermocline
depth h) driven by a red-noise filter x with a state-dependent amplitude
(Jin 1997; Levine & Jin 2010): the filter is driven but does not feed back,
so the propagator is block triangular and several of its entries miss a
pole (exact-zero residues).  Every bit-for-bit comparison runs a discarded
warm-up first (the first run of a process can differ in the last bits).
"""
import math

import numpy as np
import pytest

pytest.importorskip('sage.all')

import engine.integration.time_domain.final_integral as FI  # noqa: E402
from api.compute import compute_cumulants  # noqa: E402
from api.model import TemporalModelBuilder  # noqa: E402

PARS = dict(lam=1 / 6, om=2 * math.pi / 48, sig=1 / math.sqrt(24),
            r=1 / 1.5, B=0.5)
ACTION = ('sum(Tt[i]*(Dt*T[i] + lam*T[i] - om*h[i] - sig*x[i] '
          '- sig*B*x[i]*T[i]) + ht[i]*(Dt*h[i] + om*T[i]) '
          '+ xt[i]*(Dt*x[i] + r*x[i]) - r*xt[i]^2 for i in pop)')
H_WORKAROUND = ('Dt*h[i] + om*T[i]', '0')
H_PURE = ('Dt*h[i]', '-om*T[i]')


def _oscillator(h_eq=H_WORKAROUND):
    mb = (TemporalModelBuilder('recharge oscillator, red multiplicative noise')
          .population('pop', size=1)
          .physical_field('T', population='pop')
          .physical_field('h', population='pop')
          .physical_field('x', population='pop'))
    for k, v in PARS.items():
        mb = mb.parameter(k, default=float(v), domain='positive')
    return (mb.set_action_text(ACTION)
            .equation(lhs='(Dt + lam)*T[i]',
                      rhs='om*h[i] + sig*x[i] + sig*B*x[i]*T[i]',
                      population='pop')
            .equation(lhs=h_eq[0], rhs=h_eq[1], population='pop')
            .equation(lhs='(Dt + r)*x[i]', rhs='0', population='pop')
            .build())


@pytest.fixture(scope='module')
def model():
    return _oscillator()


def _cumulants(model, k=2, ell=1, field='dT', grouped=False):
    kw = dict(k=k, max_ell=ell, external_fields=[(field, 1)] * k,
              parallel=False, verbose=False)
    if grouped:
        kw['use_grouped_phase_j'] = True
    if k <= 2:
        r = compute_cumulants(model, tau_grid=np.array([-0.4, 0.0, 0.7]),
                              **kw)
        return r, {e: [repr(complex(v)) for v in np.atleast_1d(c)]
                   for e, c in r['C_tau_by_ell'].items()}
    r = compute_cumulants(model, **kw)
    pt = (0.0, -2e-6, -1e-6)
    return r, {e: repr(complex(f(*pt))) for e, f in r['total_C_by_ell'].items()}


def _ab(monkeypatch, model, variants, **kw):
    """Values for each ``{attr: value}`` variant of FI, after a warm-up."""
    _cumulants(model, **kw)
    out = []
    for flags in variants:
        for a, v in flags.items():
            monkeypatch.setattr(FI, a, v)
        FI._reset_runtime_counters()
        r, vals = _cumulants(model, **kw)
        out.append((vals, dict(FI._RUNTIME_COUNTERS), r))
    return out


# ═══════════════════════════════════════════════════════════════════════
# F1 / F2 on the pipeline: pruned == unpruned, bit for bit
# ═══════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize('k, field', [(2, 'dT'), (2, 'dh'), (1, 'dh')])
def test_pruning_and_zero_group_skip_change_no_bit(monkeypatch, model, k,
                                                   field):
    variants = [dict(PRUNE_ZERO_MODES=p, DD_BATCH_SKIP_ZERO_GROUPS=f)
                for p in (True, False) for f in (True, False)]
    runs = _ab(monkeypatch, model, variants, k=k, field=field)
    ref = runs[0][0]
    for vals, _c, _r in runs[1:]:
        assert vals == ref
    pruned = runs[0][1]
    assert pruned['zero_modes_pruned'] > 0
    assert pruned['zero_mode_tuples_pruned'] > 0
    assert pruned['zero_mode_order_mismatch'] == 0
    assert runs[2][1]['zero_modes_pruned'] == 0           # pruning off


def test_k3_one_loop_pruned_equals_unpruned(monkeypatch, model):
    runs = _ab(monkeypatch, model, [dict(PRUNE_ZERO_MODES=True),
                                    dict(PRUNE_ZERO_MODES=False)], k=3)
    assert runs[0][0] == runs[1][0]
    c = runs[0][1]
    # 21 plans: 4,374 nonzero tuples of 393,903
    assert c['zero_mode_plans_pruned'] == 21
    assert c['zero_mode_tuples_pruned'] == 393_903 - 4_374
    tree = complex(runs[0][0][0])
    loop = complex(runs[0][0][1])
    assert abs(tree.real - 0.12476252951567) < 1e-12
    assert abs(loop.real - 0.0419736529578) < 1e-11


def test_grouped_route_pruned_equals_unpruned(monkeypatch, model):
    runs = _ab(monkeypatch, model, [dict(PRUNE_ZERO_MODES=True),
                                    dict(PRUNE_ZERO_MODES=False)],
               grouped=True)
    assert runs[0][0] == runs[1][0]
    assert runs[0][1]['zero_modes_pruned'] > 0
    assert runs[1][1]['zero_modes_pruned'] == 0


# ═══════════════════════════════════════════════════════════════════════
# F1 on one plan: the unpruned group order is replayed
# ═══════════════════════════════════════════════════════════════════════

# Edges 0 and 1 are parallel (same row), so tuples that swap their poles
# share an alpha.  Edge 1's first mode is an exact zero: the group of
# (lam2, lam1) is first met at a ZERO tuple, before the first nonzero tuple
# of the group (lam1, lam3) -- the pruned enumeration alone would order the
# groups differently from the unpruned plan.
_L = (-1.3 + 0.4j, -0.7, -2.1 - 0.3j, -0.9 + 1.1j)
_ROWS = [((-1.0, 1.0, 0.0), (0.0,), 0.0),
         ((-1.0, 1.0, 0.0), (0.0,), 0.0),
         ((0.0, -1.0, 1.0), (0.0,), 0.0),
         ((0.0, 0.0, -1.0), (1.0,), 0.0)]


def _ems(modes):
    return FI.EdgeModeSum(ri=0, pi=0, delta_coeff=0j, modes=tuple(modes))


def _modes(n):
    return [[(1.0 - 0.1j * j, _L[j]) for j in range(n)],
            [(0j if j == 0 else 0.5 + 0.2j * j, _L[j]) for j in range(n)],
            [(0.8, -1.1), (0.0, -0.4)],
            [(1.0, -0.5)]]


@pytest.mark.parametrize('n', [3, 4])
def test_pruned_plan_keeps_the_unpruned_group_order(n):
    ems = [_ems(md) for md in _modes(n)]
    full = FI._build_modesum_plan(ems, _ROWS, 3, 1, prune=False)
    pruned = FI._build_modesum_plan(ems, _ROWS, 3, 1, prune=True)
    assert pruned['n_tuples_full'] == len(full['pole_tuples'])
    assert pruned['n_groups_full'] == len(full['tuple_groups'])
    pt = full['pole_tuples']
    live = [(a, tuple(t for t in idx if pt[t][0] != 0))
            for a, idx in full['tuple_groups']]
    live = [(a, idx) for a, idx in live if idx]
    assert [a for a, _ in pruned['tuple_groups']] == [a for a, _ in live]
    qt = pruned['pole_tuples']
    for (_a, idx_p), (_b, idx_f) in zip(pruned['tuple_groups'], live):
        assert [qt[t] for t in idx_p] == [pt[t] for t in idx_f]
    # the case is adversarial: the pruned tuples' own first-occurrence
    # order differs
    seen = []
    for a in pruned['alphas_per_tuple']:
        if a not in seen:
            seen.append(a)
    assert seen != [a for a, _ in live]
    # decay margin from ALL modes
    assert pruned['_m7_decay_tol'] == FI._decay_tol(ems, None, None, _ROWS)


@pytest.mark.parametrize('n', [3, 4])
def test_poset_value_is_bit_identical_pruned_or_not(monkeypatch, n):
    monkeypatch.setattr(FI, 'USE_PHASE_J_DD_KERNELS', True)
    monkeypatch.setattr(FI, 'SEMIINF_CHAIN', True)
    ems = [_ems(md) for md in _modes(n)]
    vals = []
    for prune in (False, True, False):
        FI._chain_simplex_memo_clear()
        p = FI._build_modesum_plan(ems, _ROWS, 3, 1, prune=prune)
        vals.append(FI._integrate_nd_polytope_poset_modesum(
            ems, 0.9 - 0.1j, _ROWS, [0.6], 3, pole_tuples=p['pole_tuples'],
            plan=p))
    assert vals[0] is not None and vals[0] != 0
    assert repr(vals[1]) == repr(vals[0]) == repr(vals[2])


def test_order_replay_over_its_cap_keeps_the_pruned_order(monkeypatch):
    monkeypatch.setattr(FI, 'PRUNE_ORDER_MAX_TUPLES', 4)
    FI._reset_runtime_counters()
    ems = [_ems(md) for md in _modes(3)]
    p = FI._build_modesum_plan(ems, _ROWS, 3, 1, prune=True)
    assert FI._RUNTIME_COUNTERS['zero_mode_order_unreplayed'] == 1
    assert all(c != 0 for c, _l in p['pole_tuples'])
    assert len(p['pole_tuples']) == 3 * 2 * 1 * 1


def test_edge_without_a_nonzero_mode_keeps_the_unpruned_plan():
    modes = _modes(3)
    modes[2] = [(0j, -1.1), (0j, -0.4)]
    ems = [_ems(md) for md in modes]
    p = FI._build_modesum_plan(ems, _ROWS, 3, 1, prune=True)
    q = FI._build_modesum_plan(ems, _ROWS, 3, 1, prune=False)
    assert p['pole_tuples'] == q['pole_tuples']
    assert p['tuple_groups'] == q['tuple_groups']


def test_only_exact_zeros_are_pruned():
    assert FI._mode_is_zero(0j) and FI._mode_is_zero(-0.0)
    assert not FI._mode_is_zero(1e-300)
    assert not FI._mode_is_zero(complex('nan'))
    modes = _modes(3)
    modes[1][0] = (5e-324 + 0j, _L[0])                 # tiny, kept
    modes[2][1] = (0.3, -0.4)
    ems = [_ems(md) for md in modes]
    assert FI._build_pruned_modesum_plan(ems, _ROWS, 3, 1) is None


# ═══════════════════════════════════════════════════════════════════════
# F3: the k >= 3 slices of dd.run
# ═══════════════════════════════════════════════════════════════════════

def test_k3_slices_are_the_sum_of_the_per_ell_slices(model):
    import daedalus as dd
    cfg = dd.Config(k=3, max_ell=1, external_fields=[('dT', 1)] * 3,
                    tau_grid=np.array([0.0, 0.5]), parallel=False,
                    verbose=False)
    dd.run(model, cfg)                                  # warm-up
    FI._reset_runtime_counters()
    r = dd.run(model, cfg)
    by = r['C_tau_slices_by_ell']
    for j, sl in r['C_tau_slices'].items():
        for i in range(len(sl)):
            assert complex(sl[i]) == sum(complex(by[j][e][i]) for e in by[j])
    # and equal to total_C at the slice point (warm: same bits)
    total = r['total_C']
    pt = r['_kpoint_slice_times'][1][1]
    assert complex(total(*pt)) == complex(r['C_tau_slices'][1][1])


# ═══════════════════════════════════════════════════════════════════════
# The memory budget
# ═══════════════════════════════════════════════════════════════════════

def test_budget_parsing():
    f = FI.phase_j_mem_budget_bytes
    assert f({}) == 6e9
    assert f({'PHASE_J_MEM_BUDGET_GB': '0.5'}) == 0.5e9
    for off in ('off', 'inf', '0', '-1', 'none'):
        assert f({'PHASE_J_MEM_BUDGET_GB': off}) is None
    with pytest.raises(ValueError, match='PHASE_J_MEM_BUDGET_GB'):
        f({'PHASE_J_MEM_BUDGET_GB': 'lots'})


def test_estimate_is_recorded_and_default_budget_passes(model):
    r, _v = _cumulants(model, k=3)
    est = r['phase_j_memory']
    assert est['tuples'] == 4_374 and est['tuples_unpruned'] == 393_903
    assert est['bytes'] == 4_374 * FI.PLAN_BYTES_PER_TUPLE
    assert est['budget_bytes'] == 6e9
    assert 0 < est['built_tuples'] <= est['tuples']
    an = r.get('analysis')
    if an is not None:
        assert an.diagnostics['phase_j_memory']['tuples'] == 4_374
        assert any(e.name == 'phase_j_memory_budget' and e.satisfied
                   for e in an.ledger)


def test_over_budget_refuses_before_phase_j(monkeypatch, model):
    monkeypatch.setenv('PHASE_J_MEM_BUDGET_GB', '0.001')
    built = []
    orig = FI._plan_from_tuples
    monkeypatch.setattr(FI, '_plan_from_tuples',
                        lambda *a, **k: built.append(1) or orig(*a, **k))
    with pytest.raises(FI.PhaseJResourceError) as ei:
        _cumulants(model, k=3)
    msg = str(ei.value)
    assert 'PHASE_J_MEM_BUDGET_GB' in msg and '4,374' in msg
    assert 'Largest diagram' in msg
    assert ei.value.estimate['tuples'] == 4_374
    assert built == []                         # nothing was allocated
    import daedalus as dd
    assert dd.PhaseJResourceError is FI.PhaseJResourceError


def test_strict_policy_refuses_the_same_way(monkeypatch, model):
    monkeypatch.setenv('PHASE_J_MEM_BUDGET_GB', '0.001')
    with pytest.raises(FI.PhaseJResourceError):
        compute_cumulants(model, k=3, max_ell=1,
                          external_fields=[('dT', 1)] * 3, parallel=False,
                          verbose=False, policy={'mode': 'strict'})


def test_running_guard_refuses_before_allocating(monkeypatch):
    ems = [_ems(md) for md in _modes(4)]
    calls = []
    orig = FI._enumerate_pole_tuples

    def spy(e):
        calls.append(1)
        return orig(e)
    monkeypatch.setattr(FI, '_enumerate_pole_tuples', spy)
    n_kept = 4 * 3 * 1 * 1
    budget = (n_kept + 0.5) * FI.PLAN_BYTES_PER_TUPLE
    with FI.phase_j_memory_scope(budget=budget, what='test run') as sc:
        FI._build_modesum_plan(ems, _ROWS, 3, 1, prune=True, where='first')
        assert sc is not None and calls == [1]
        with pytest.raises(FI.PhaseJResourceError) as ei:
            FI._build_modesum_plan(ems, _ROWS, 3, 1, prune=True,
                                   where='second plan')
    assert calls == [1]                        # the second was never built
    assert 'second plan' in str(ei.value) and 'test run' in str(ei.value)
    assert ei.value.estimate['bytes_so_far'] == n_kept * FI.PLAN_BYTES_PER_TUPLE
    # outside a scope each plan is checked on its own
    monkeypatch.setenv('PHASE_J_MEM_BUDGET_GB', str(budget / 1e9))
    FI._build_modesum_plan(ems, _ROWS, 3, 1, prune=True)
    FI._build_modesum_plan(ems, _ROWS, 3, 1, prune=True)
    with pytest.raises(FI.PhaseJResourceError):
        FI._build_modesum_plan(ems, _ROWS, 3, 1, prune=False)  # 16 tuples


# ═══════════════════════════════════════════════════════════════════════
# Pure-integrator mean-field equations
# ═══════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize('k, field', [(2, 'dT'), (2, 'dh'), (1, 'dh')])
def test_pure_integrator_equals_the_moved_form(model, k, field):
    pure = _oscillator(H_PURE)
    _cumulants(pure, k=k, field=field)                 # warm-up
    _r, a = _cumulants(model, k=k, field=field)
    _r, b = _cumulants(pure, k=k, field=field)
    assert a == b
    if k == 1:            # <h> = -sig^2 B / (om (r + lam + om^2/r))
        P = PARS
        exact = -P['sig'] ** 2 * P['B'] / (
            P['om'] * (P['r'] + P['lam'] + P['om'] ** 2 / P['r']))
        assert abs(complex(b[1][0]).real - exact) < 1e-12


def _two_field(h_eq):
    return (TemporalModelBuilder('decoupled integrator')
            .population('pop', size=1)
            .physical_field('T', population='pop')
            .physical_field('h', population='pop')
            .parameter('lam', default=0.5, domain='positive')
            .parameter('om', default=0.3, domain='positive')
            .parameter('a', default=0.2, domain='real')
            .set_action_text('sum(Tt[i]*(Dt*T[i] + lam*T[i]) '
                             '+ ht[i]*(Dt*h[i] + om*T[i]) - Tt[i]^2 '
                             'for i in pop)')
            .equation(lhs='(Dt + lam)*T[i]', rhs='0', population='pop')
            .equation(lhs=h_eq[0], rhs=h_eq[1], population='pop')
            .build())


def test_pure_integrator_without_a_field_is_refused_at_build():
    with pytest.raises(ValueError, match='pure integrator.*nothing pins'):
        _two_field(('Dt*h[i]', 'a'))


def test_unpinned_pure_integrator_is_refused_by_the_saddle_solve():
    m = _two_field(H_PURE)                 # h integrates T; nothing pins h
    with pytest.raises(Exception, match='do not determine the saddle'):
        compute_cumulants(m, k=2, max_ell=0, external_fields=[('dT', 1)] * 2,
                          tau_grid=np.array([0.0]), parallel=False,
                          verbose=False)
