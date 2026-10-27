"""
Every reason the state-space Phase J route gives for declining a model, a
diagram or a subset, or for refusing an evaluation at call time, is named in
``statespace.DECLINE_REASONS`` (audit W7, RED-01 stage 1: the pole engine
stays the supported backup for everything declined, and no decline is
anonymous).  The source scan fails when a new reason is added without a
name, or a named one no longer exists; a few reasons are also triggered.
"""
import os
import re
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from daedalus.engine.integration.time_domain import statespace as SS     # noqa: E402
import daedalus.api._realization as R                                     # noqa: E402

_ROOT = os.path.join(os.path.dirname(__file__), '..')
_SS = os.path.join(_ROOT, 'daedalus', 'engine', 'integration', 'time_domain',
                   'statespace.py')
_FI = os.path.join(_ROOT, 'daedalus', 'engine', 'integration', 'time_domain',
                   'final_integral.py')

# literal reasons handed to the decline / refusal constructors
_PATTERNS = (r"_decline\('([a-z_0-9]+)'\)",
             r"SSIneligible\(\s*'([a-z_0-9]+)'",
             r"SSLateRefusal\(\s*'([a-z_0-9]+)'",
             r"ops, why = None, '([a-z_0-9]+)'",
             r"ok, why = False, '([a-z_0-9]+)'",
             r"return False, '([a-z_0-9]+)'",
             r"ent, why = None, '([a-z_0-9]+)'",
             r"out = '([a-z_0-9]+)'",
             r"return '([a-z_0-9]+)' if e is None",
             r"return '(segment_memory)'")


def _source_reasons():
    src = open(_SS).read()
    found = set()
    for pat in _PATTERNS:
        found |= set(re.findall(pat, src))
    # the ('unsupported', reason) edge specs final_integral hands the route
    found |= set(re.findall(r"\('unsupported', '([a-z_0-9]+)'\)",
                            open(_FI).read()))
    return found


def test_every_produced_reason_is_named():
    found = _source_reasons()
    assert len(found) > 20, found          # the scan itself works
    unnamed = sorted(r for r in found if not SS.decline_reason_known(r))
    assert not unnamed, f'unnamed state-space decline reasons: {unnamed}'


def test_no_named_reason_is_stale():
    src = open(_SS).read() + open(_FI).read()
    stale = [r for r in SS.DECLINE_REASONS if f"'{r}'" not in src.replace(
        "    '" + r + "':", '')]
    assert not stale, f'named but never produced: {stale}'


def test_realization_and_kernel_families_cover_every_refusal():
    """``realization_<reason>`` / ``kernel_<reason>`` cover every
    ``RealizationRefused`` reason, and the generic 'error' fallback."""
    for r in R._REASONS + ('error',):
        assert SS.decline_reason_known(f'realization_{r}')
        assert SS.decline_reason_known(
            r if r.startswith('kernel_') else f'kernel_{r}')
    assert SS.decline_reason_known('numerical_LinAlgError')
    assert not SS.decline_reason_known('no_such_reason')


def test_every_reason_has_a_description():
    for name, text in list(SS.DECLINE_REASONS.items()) + list(
            SS.DECLINE_PREFIXES.items()):
        assert isinstance(text, str) and len(text) > 10, name


def test_engine_poles_is_the_run_reason(monkeypatch):
    monkeypatch.setenv('DAEDALUS_PHASE_J_ENGINE', 'poles')
    pred = SS.run_route({}, {}, ())
    assert pred == {'engine': 'poles', 'covers_run': False,
                    'reason': 'engine_poles'}
    assert SS.decline_reason_known(pred['reason'])


def test_unrealizable_propagator_reason_is_named():
    """A propagator without K_ft (nothing to realize) is declined by a named
    ``realization_*`` reason."""
    ok, why = SS.model_route({'pole_vals': [], 'C_mats': []}, {})
    assert ok is False and SS.decline_reason_known(why), why
    assert why.startswith('realization_')


# ── LOUD-05: a repeated-rate conductance kernel on the pole engine ─────────

_P05 = dict(h=0.9, a=1.2, q=0.35, c=0.25, m=0.6, D=0.12)


def _alpha_conductance(embedded):
    """x' = h - a x - q x^2 - c x (g * x) + noise with the Erlang-2 (alpha)
    kernel g (a repeated rate); ``embedded``: its exact Markov twin (two
    auxiliary linear fields)."""
    import daedalus.api.noise as N
    from daedalus.api.sde import SDE
    from daedalus.api.delay import Kernel
    s = SDE('w7 alpha conductance' + (' embedded' if embedded else ''))
    for p, v in _P05.items():
        s = s.parameter(p, default=v,
                        **({'domain': 'positive'} if p in ('m', 'D') else {}))
    if embedded:
        s = (s.field('x').field('y1').field('y2')
             .drift('x', 'h - a*x - q*x^2 - c*x*y2')
             .drift('y1', '(2/m)*(x - y1)').drift('y2', '(2/m)*(y1 - y2)'))
    else:
        s = (s.define_kernel('g', kernel=Kernel.gamma(2, 'm')).field('x')
             .drift('x', 'h - a*x - q*x^2 - c*x*Conv(g, x)'))
    return s.noise('xi', N.Gaussian(var='2*D'), couples={'x': 1}).build()


def _run05(model):
    import numpy as np
    from daedalus.api.compute import compute_cumulants
    return compute_cumulants(
        model, k=2, max_ell=1, external_fields=[('x', 1), ('x', 1)],
        tau_grid=np.array([0.0, 0.5, 2.0]), parallel=False, verbose=False,
        use_cache=False)


def test_alpha_kernel_loop_state_space_matches_markov_twin(monkeypatch):
    import numpy as np
    monkeypatch.delenv('DAEDALUS_PHASE_J_ENGINE', raising=False)
    r = _run05(_alpha_conductance(False))
    ref = _run05(_alpha_conductance(True))
    assert r['status'] == 'ok' and not r['phase_j_incomplete']
    for ell in (0, 1):
        a, b = np.asarray(r['C_tau_by_ell'][ell]), np.asarray(
            ref['C_tau_by_ell'][ell])
        assert np.max(np.abs(b)) > 1e-4
        assert np.allclose(a, b, rtol=1e-10, atol=0), (ell, a, b)


def test_alpha_kernel_loop_pole_engine_refuses_loudly(monkeypatch):
    from daedalus.api.compute import PhaseJIncompleteWarning
    monkeypatch.setenv('DAEDALUS_PHASE_J_ENGINE', 'poles')
    with pytest.warns(PhaseJIncompleteWarning):
        r = _run05(_alpha_conductance(False))
    assert r['status'] == 'incomplete'
    reasons = [f['reason'] for f in r['phase_j_incomplete'][1]]
    assert reasons and all('state-space engine' in x for x in reasons)
