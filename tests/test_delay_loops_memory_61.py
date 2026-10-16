"""
Item 61: delay / dressed-line loops that fit.

D. The memory pre-pass is honest:
   (i) the exact-zero-prefactor test also drops diagrams charged the
   state-space (SS) estimate; (ii) a conductance-kernel line is charged as
   it is built: dressed (one edge, the cascade's states, no δ part) or on
   the leg path (an extra vertex, the propagator edge with its δ part and
   the kernel edge in series); (iii) each δ-subset is counted on its
   contracted DAG; (iv) a forked τ batch uses no more workers than the
   budget allows.  Reference: an exact census of the subsets Phase J builds
   (every subset constructed, no stationary state allocated).
C. ``compute_cumulants(..., discard=True / 'auto')``: each diagram is
   evaluated on the τ grid right after it is built and released; the values
   are bit-identical, the callables answer the grid only.
B. Dressed lines are reduced to their minimal part (balanced truncation at
   the truncation tolerance in effect).

Models (inline, public): the delayed alpha Hawkes of
``test_point_process_front_end`` (spike-only and voltage forms), an
exponential-kernel spike-only Hawkes neuron, and the textbook delayed
negative feedback ``x' = c / (1 + (eps x(t - d))^2) - g x`` as a chemical
master equation.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(__file__))

from api import compute_cumulants                                   # noqa: E402
from api.delay import Kernel                                        # noqa: E402
from api.sde import SDE                                             # noqa: E402
from engine.integration.time_domain import final_integral as FI     # noqa: E402
from engine.integration.time_domain import pipeline as PL           # noqa: E402
from engine.integration.time_domain import statespace as SS         # noqa: E402
from test_point_process_front_end import _delay_pp                  # noqa: E402

TAU = np.array([-2.0, 0.0, 0.5, 3.0])


def _spike_exp(P=1):
    """Spike-only Hawkes with an exponential synaptic kernel (dressed lines:
    the kernels have separated modes)."""
    return (SDE(f'exp-kernel spike Hawkes P{P}').population('E', size=P)
            .parameter('Em', default=[-1.5] * P, indexed_by=['E'])
            .parameter('tau', default=[2.0] * P, indexed_by=['E'],
                       domain='positive')
            .parameter('taug', default=[[1.0] * P] * P, indexed_by=['E', 'E'],
                       domain='positive')
            .parameter('a', default=[1.0] * P, indexed_by=['E'],
                       domain='positive')
            .parameter('w', default=[[0.4 / P] * P] * P, indexed_by=['E', 'E'])
            .define_kernel('g', kernel=Kernel.exponential('taug[i,j]'),
                           indexed_by=['E', 'E'])
            .define_kernel('h', kernel=Kernel.exponential('tau[i]'),
                           indexed_by=['E'])
            .point_process(
                'n', 'a*exp(Em + sum(w[i,j]*Conv(h[i]*g[i,j], n[j]) for j in E))',
                population='E')).build()


def _negative_feedback(kind='erlang', K=8):
    """The textbook delayed negative feedback as a chemical master equation
    (x = n / Om: creation Om c / (1 + (eps x(t - d))^2), degradation
    Om g x)."""
    from api.theory import TemporalTheoryBuilder
    b = (TemporalTheoryBuilder(f'delayed negative feedback {kind}')
         .population('pop', size=1).physical_field('x', population='pop'))
    for k, v in dict(c=1.0, eps=0.4, g=1.0, d=5.0, Om=100.0).items():
        b = b.parameter(k, default=v, domain='positive')
    b = (b.define_kernel('gd', delay='d') if kind == 'hard'
         else b.define_kernel('gd', kernel=Kernel.gamma(K, f'd/{K}')))
    return (b.define_function('f', args=['y'], expression='c/(1 + (eps*y)^2)',
                              population='pop')
            .set_action_text('''sum( xt[i]*Dt*x[i]
                - Om*f[i](Conv(gd, x[i]))*(exp(xt[i]/Om) - 1)
                - Om*g*x[i]*(exp(-xt[i]/Om) - 1) for i in pop)''')
            .set_mf_equation('xstar', 'f[i](xstar[i])/g')
            .build())


# ═══════════════════════════════════════════════════════════════════════
# D. the pre-pass against an exact census
# ═══════════════════════════════════════════════════════════════════════

@pytest.fixture
def census(monkeypatch):
    """Construct every subset Phase J builds, allocate no stationary state:
    returns the list of per-subset stationary bytes."""
    rec = []
    orig = SS.SubsetSS.__init__

    def init(self, *a, **k):
        orig(self, *a, **k)
        rec.append(self.state_bytes())

    def ev(self, free_vals, tie_ctx=None, t_min=float('-inf')):
        SS._EST.err = 0.0
        return 0.0
    monkeypatch.setattr(SS.SubsetSS, '__init__', init)
    monkeypatch.setattr(SS.SubsetSS, 'evaluate', ev)
    monkeypatch.setattr(SS, '_prefetched', lambda *a, **k: None)
    monkeypatch.setenv('PHASE_J_MEM_BUDGET_GB', 'off')
    return rec


@pytest.mark.parametrize('form, n_pop', [('spike', 1), ('voltage', 1),
                                         ('voltage', 2)])
def test_estimate_bounds_the_census_tightly(census, form, n_pop):
    """Leg-path lines (the delayed alpha kernel has a repeated pole) were
    charged as dressed lines with every kernel attachment doubling the
    downsets: 0.185 GB for the 1-neuron spike form, 12.9 MB built."""
    w = ((0.5,),) if n_pop == 1 else ((0.0, 0.5), (0.5, 0.0))
    ext = [('n', 1), ('n', 1)] if n_pop == 1 else [('n', 1), ('n', 2)]
    r = compute_cumulants(_delay_pp(form, n_pop=n_pop, w=w).build(), k=2,
                          max_ell=1, external_fields=ext,
                          tau_grid=np.array([0.5]), parallel=False,
                          verbose=False, use_cache=False,
                          delay={'order': 4})
    built, est = sum(census), r['phase_j_memory']['ss_bytes']
    assert built > 0
    assert built <= est <= 1.05 * built
    assert r['phase_j_memory']['max_diagram_bytes'] <= est


def _prepass_only(monkeypatch, model, **kw):
    """The pre-pass's estimate; Phase J never starts."""
    import api.compute as C

    class _Stop(Exception):
        pass
    got = {}
    orig = getattr(C._phase_j_memory_preflight, '_orig',
                   C._phase_j_memory_preflight)

    def pre(*a, **k):
        got['est'] = orig(*a, **k)
        raise _Stop()
    pre._orig = orig
    monkeypatch.setattr(C, '_phase_j_memory_preflight', pre)
    with pytest.raises(_Stop):
        C.compute_cumulants(model, parallel=False, verbose=False, **kw)
    return got['est']


def test_zero_prefactor_diagrams_on_the_ss_branch(monkeypatch):
    """2-neuron delayed-alpha spike-only one loop, w11 = w22 = 0: 1,300 of
    the 1,446 diagrams are exactly zero.  Refused at ~122 GB before; the
    census builds 0.39 GB."""
    kw = dict(k=2, max_ell=1, external_fields=[('n', 1), ('n', 2)],
              tau_grid=np.array([-3.0, 0.5, 3.0]), delay={'order': 4})
    model = _delay_pp('spike', n_pop=2).build()
    est = _prepass_only(monkeypatch, model, **kw)       # default budget
    assert not est['above_budget'] and est['bytes'] < 6e9
    monkeypatch.setenv('PHASE_J_MEM_BUDGET_GB', '1')    # the zero test runs
    est = _prepass_only(monkeypatch, model, **kw)
    assert est['prefactors_tested']
    assert est['ss_diagrams'] == 146
    assert 0.38e9 < est['bytes'] < 0.40e9


def test_contracted_dag():
    # u -> v (δ) and u -> w -> v (smooth): contracting u = v makes a cycle,
    # merged into one time
    edges = ((0, 1, 2, True), (0, 2, 3, False), (2, 1, 5, False))
    nv, lm, e2 = SS._contract(3, 0b010, edges, {0})
    assert nv == 1 and lm == 1 and e2 == ()
    # a smooth line without states: the subset is never built
    assert SS._contract(2, 0b10, ((0, 1, 0, True),), set()) is None
    # nothing contracted: the smooth edges as they are
    assert SS._contract(2, 0b10, ((0, 1, 4, True),), set()) == (
        2, 0b10, ((0, 1, 4),))


def test_parallel_workers_within_budget():
    assert PL._workers_within_budget(8) == 8            # no scope
    with FI.phase_j_memory_scope(budget=10e9, rule='test', use_env=False):
        assert PL._workers_within_budget(8) == 8        # nothing charged
        with FI._PLAN_MEM_LOCK:
            FI._PLAN_MEM['used'] = 2e9
        assert PL._workers_within_budget(8) == 4        # (1 + 4) x 2 <= 10
        with FI._PLAN_MEM_LOCK:
            FI._PLAN_MEM['used'] = 6e9
        assert PL._workers_within_budget(8) == 1        # serial
    with FI.phase_j_memory_scope(budget=None, use_env=False):
        with FI._PLAN_MEM_LOCK:
            FI._PLAN_MEM['used'] = 6e9
        assert PL._workers_within_budget(8) == 8        # no budget


# ═══════════════════════════════════════════════════════════════════════
# C. evaluate-and-discard
# ═══════════════════════════════════════════════════════════════════════

def _run(model, discard, ext=(('n', 1), ('n', 1)), ell=1, **kw):
    return compute_cumulants(model, k=2, max_ell=ell,
                             external_fields=list(ext), tau_grid=TAU,
                             parallel=False, verbose=False, discard=discard,
                             **kw)


def test_discard_is_bit_identical_and_releases():
    model = _delay_pp('voltage', n_pop=1, w=((0.5,),)).build()
    _run(model, False, delay={'order': 4})                       # warm
    a = _run(model, False, delay={'order': 4})
    b = _run(model, True, delay={'order': 4})
    for e in a['C_tau_by_ell']:
        assert np.array_equal(np.asarray(a['C_tau_by_ell'][e]),
                              np.asarray(b['C_tau_by_ell'][e]))
    assert np.array_equal(a['C_tau'], b['C_tau'])
    pm = b['phase_j_memory']
    assert pm['discard'] and pm['built_bytes'] == a['phase_j_memory'][
        'built_bytes']
    assert pm['kept_bytes'] < pm['built_bytes']
    assert pm['discard_refunded'] > 0
    # the callables answer the grid (the same sum) and nothing else
    assert complex(b['total_C'](0.0, 0.5)) == complex(a['total_C'](0.0, 0.5))
    assert complex(b['total_C'](0.0, 0.0)) == complex(a['total_C'](0.0, 0.0))
    import daedalus as dd
    with pytest.raises(dd.DiscardedDiagramsError, match='discard=False'):
        b['total_C'](0.0, 0.7)
    g = b['phase_j_by_ell'][1]['groups'][0]['contribution']
    with pytest.raises(dd.DiscardedDiagramsError, match='released'):
        g(0.0, 0.5)


def test_discard_auto_fires_only_over_budget(monkeypatch):
    """'auto' is decided by the pre-pass: on exactly when the summed
    estimate is over the budget and the largest diagram (+ one segment
    working set, left out here: its dense-generator bound is ~30 MB on this
    model) fits."""
    orig = SS.diagram_estimate
    monkeypatch.setattr(SS, 'diagram_estimate',
                        lambda *a, **k: (lambda e: e and (e[0], 0))(
                            orig(*a, **k)))
    model = _delay_pp('spike', n_pop=1, w=((0.5,),)).build()
    kw = dict(k=2, max_ell=1, external_fields=[('n', 1), ('n', 1)],
              tau_grid=TAU, delay={'order': 4}, discard='auto')
    est = _prepass_only(monkeypatch, model, **kw)
    assert est['discard'] is False and est['held_bytes'] == est['bytes']
    # a budget below the sum (12.9 MB) but above the largest diagram (6.7 MB)
    b = (est['bytes'] + est['max_diagram_bytes']) / 2
    assert est['max_diagram_bytes'] < b < est['bytes']
    monkeypatch.setenv('PHASE_J_MEM_BUDGET_GB', repr(b / 1e9))
    est = _prepass_only(monkeypatch, model, **kw)
    assert est['discard'] is True and not est['above_budget']
    assert est['held_bytes'] == est['max_diagram_bytes']
    # below the largest diagram: off, and the refusal suggests nothing
    monkeypatch.setenv('PHASE_J_MEM_BUDGET_GB',
                       repr(0.5 * est['max_diagram_bytes'] / 1e9))
    est = _prepass_only(monkeypatch, model, **kw)
    assert est['discard'] is False


def test_discard_needs_fixed_points():
    model = _negative_feedback('erlang')
    with pytest.raises(ValueError, match='k >= 3'):
        compute_cumulants(model, k=3, max_ell=0,
                          external_fields=[('x', 1)] * 3, parallel=False,
                          verbose=False, discard=True)
    with pytest.raises(ValueError, match="discard='yes'"):
        _run(model, 'yes', ext=[('x', 1), ('x', 1)], ell=0)


# ═══════════════════════════════════════════════════════════════════════
# B. minimal dressed lines
# ═══════════════════════════════════════════════════════════════════════

def test_dressed_lines_are_minimal(monkeypatch):
    """One population: the propagator cancels the kernel's poles, so the
    dressed cascade (n_G + n_K states) is reduced to n_G; values within the
    truncation tolerance of the unreduced cascade."""
    model = _spike_exp(1)

    def run(reduce):
        if not reduce:
            monkeypatch.setattr(SS._ModelOps, '_reduce_dressed',
                                lambda self, A, B, C, *a: (A, B, C))
        else:
            monkeypatch.undo()
        SS._REAL_CACHE.clear()
        return _run(model, False, use_cache=False)
    full, red = run(False), run(True)
    np.testing.assert_allclose(red['C_tau'], full['C_tau'], rtol=1e-12,
                               atol=0)
    # each dressed line 2 -> 1 states: the stationary states shrink
    assert (red['phase_j_memory']['built_bytes']
            < full['phase_j_memory']['built_bytes'])


def test_dressed_line_record_and_rank_one_network():
    """A rank-1 two-population network: nothing cancels, the cascade is
    returned as built (the same arrays), and its record says so."""
    from types import SimpleNamespace
    a1 = np.array([[-1.0, 0.0], [0.0, -2.0]])
    ent = SimpleNamespace(a=a1, b=np.array([1.0, 1.0]), c=np.array([1.0, 0.5]),
                          d=0.0, n_states=2)
    real = SimpleNamespace(entries={(0, 0): ent}, A=a1, truncation={})
    ops = SS._ModelOps.__new__(SS._ModelOps)
    ops.real, ops.num_params, ops.is_real, ops.dtype = real, None, True, float
    ops.entries, ops._solve_data, ops._extra = {}, {}, {}
    import threading
    ops._lock = threading.RLock()
    A = np.array([[-1.0, 0.0, 0.0], [0.0, -2.0, 0.0], [1.0, 0.5, -3.0]])
    B = np.array([1.0, 1.0, 0.0])
    Cc = np.array([0.0, 0.0, 1.0])
    out = ops._reduce_dressed(A, B, Cc, 1e-12, 0, 0, {'symbol': 'g'})
    assert out[0] is A                                  # minimal: unchanged
    rec = real.truncation['dressed'][(0, 0, 'g')]
    assert rec['n'] == rec['r'] == 3
    # an uncontrollable kernel block (b2 c1 = 0 and d1 = 0): reduced
    A2 = np.array([[-1.0, 0.0, 0.0], [0.0, -2.0, 0.0], [0.0, 0.0, -3.0]])
    out = ops._reduce_dressed(A2, B, np.array([1.0, 1.0, 1.0]), 1e-12, 0, 1,
                              {'symbol': 'g'})
    assert out[0].shape == (2, 2)
    assert np.all(np.linalg.eigvals(out[0]).real < 0)
    assert real.truncation['dressed'][(0, 1, 'g')]['r'] == 2
