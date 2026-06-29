"""
tests/test_fix_colored.py
=========================
Regression test for the colored-noise routing bug in the GROUPED
Phase-J path (``engine/integration/time_domain/grouped_integral.py``).

Bug
---
The grouped path builds each noise source's per-leg time map keyed on
the leg ORDINAL / ``edge_pop_idx`` (grouped_integral.py ~491-505 build,
~533-545 lookup).  For a homogeneous auto-cumulant ``Cxx`` whose two
response legs share the same ``(field, pop_idx)`` (legs ``(0, 0)``),
both legs collapse onto a single dict key and the relative-time
coupling ``τ`` that the kernel ``κ(τ)`` rides on is DROPPED.  A colored
noise source then silently degrades to its WHITE-noise limit, so the
grouped and per-diagram results DISAGREE (the per-diagram path in
``final_integral.py`` routes by EDGE-KEY and is correct).

Fix (SAFE guard)
----------------
``integrate_grouped_diagram`` now DETECTS a colored noise-source kernel
(a ``NoiseSourceType`` with non-empty ``cumulant_specs``) — or a
``ConvVertexType`` conductance kernel, which the grouped path doesn't
scaffold at all — and returns ``status='failed'`` with a clear reason.
The dispatcher (``api/_grouped_phase_j.compute_correction_td_grouped``)
already falls back to the correct per-diagram path on a group-level
failure, so colored noise is NEVER silently wrong.  White-noise groups
(no ``cumulant_specs``) are untouched and keep using the fast grouped
path.

What this file asserts
----------------------
1. ``test_grouped_guard_trips_on_colored_noise_source`` — a group
   containing a colored ``NoiseSourceType`` td makes
   ``integrate_grouped_diagram`` return ``status='failed'`` (UNIT
   level; FAILS without the guard because the old code proceeds and
   returns ``status='ok'``).
2. ``test_grouped_guard_trips_on_conv_vertex`` — same for a
   ``ConvVertexType`` conductance kernel.
3. ``test_grouped_guard_does_not_trip_on_white_source`` — a plain
   white ``SourceType`` does NOT trip the guard (control; the function
   proceeds past the guard).
4. ``test_white_noise_grouped_still_matches_perdiag`` — END-TO-END:
   a white-noise theory through the grouped path still agrees with the
   per-diagram path (proves the guard left white noise unaffected).

Run with::

    sage -python -m pytest tests/test_fix_colored.py -q
"""
from __future__ import annotations

import importlib.util
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))

from sage.all import SR, DiGraph

from engine.core.vertices import (
    NoiseSourceType, ConvVertexType, SourceType,
)
from engine.integration.time_domain.grouped_integral import (
    integrate_grouped_diagram,
)


# ── Minimal TypedDiagram stand-ins ───────────────────────────────────
# The colored-kernel guard runs BEFORE any graph processing — it only
# reads ``td.vertex_assignments`` and requires every td in the group to
# share the same ``prediagram`` object (by identity).  So a tiny stub
# carrying just those two attributes is enough to exercise the guard
# without building a full enumeration.
class _StubTD:
    def __init__(self, prediagram, vertex_assignments):
        self.prediagram = prediagram
        self.vertex_assignments = vertex_assignments


def _real_prediagram():
    """A minimal but well-formed prediagram tuple ``(D, G, leaves,
    internal)`` whose ``D`` is a real ``DiGraph``.

    Used by every test here so that — with the guard disabled — the td
    flows PAST the colored guard into the downstream graph-based
    bookkeeping (``_loop_number_from_graph`` etc.) rather than crashing
    on a stub, making the without-fix failure a genuine routing
    difference rather than an unrelated ``AttributeError``.
    """
    D = DiGraph(multiedges=True, loops=False)
    D.add_edge(0, 1, 'a')   # one internal vertex (1) feeding one leaf (0)
    leaves = [0]
    internal = [1]
    return (D, D, leaves, internal)

_EXT_TIME_VARS = [SR.var('t_1', latex_name=r't_{1}')]


def _colored_noise_source():
    """A ``NoiseSourceType`` carrying a non-trivial cumulant kernel."""
    return NoiseSourceType(
        coefficient=SR.var('z_kappa_X_2_0_0'),
        response_legs=[('xt', 1), ('xt', 1)],
        bigrade=(2, 0),
        cumulant_specs=[{
            'symbol':     SR.var('z_kappa_X_2_0_0'),
            'kernel_fn':  lambda i, j, tau: SR(1),
            'legs':       (0, 0),          # homogeneous: both legs pop_idx 0
            'leg_fields': ('xt', 'xt'),
            'tau_var':    SR.var('tau'),
            'sign':       SR(-1) / 2,
            'noise':      'X',
            'order':      2,
        }],
    )


def _white_source():
    """A plain white ``SourceType`` (no cumulant kernel)."""
    return SourceType(
        coefficient=SR.var('D'),
        response_legs=[('xt', 1), ('xt', 1)],
        bigrade=(2, 0),
    )


def _conv_vertex():
    """A ``ConvVertexType`` carrying a conductance kernel attachment."""
    return ConvVertexType(
        coefficient=SR.var('w'),
        response_legs=[('vt', 1)],
        physical_legs=[('n', 1)],
        bigrade=(1, 1),
        kernel_attachments=[{
            'kernel':        SR.var('g'),
            'leg':           ('n', 1),
            'kernel_td_fn':  lambda tau: SR(1),
        }],
    )


# ── (1) Guard trips on a colored noise source ────────────────────────
def test_grouped_guard_trips_on_colored_noise_source():
    """A group with a colored ``NoiseSourceType`` must make the grouped
    evaluator bail with ``status='failed'`` (so the caller falls back to
    the correct per-diagram path).

    WITHOUT the fix the guard is absent, so the colored td flows into
    the buggy per-leg routing instead of bailing — the
    ``status == 'failed'`` assertion below then does not hold.
    """
    td = _StubTD(_real_prediagram(),
                 {1: _colored_noise_source()})
    result = integrate_grouped_diagram(
        typed_diagrams=[td],
        combined_prefactors=[SR(1)],
        propagator_data={},
        ext_time_vars=_EXT_TIME_VARS,
    )
    assert result['status'] == 'failed', (
        'grouped path must refuse a colored noise-source group; got '
        f"status={result['status']!r}"
    )
    assert 'colored' in result['reason'].lower() or \
           'cumulant_specs' in result['reason'], (
        f"reason should name the colored kernel; got {result['reason']!r}"
    )


# ── (2) Guard trips on a ConvVertexType conductance kernel ───────────
def test_grouped_guard_trips_on_conv_vertex():
    """A group with a ``ConvVertexType`` conductance kernel must also
    bail — the grouped path has no per-leg kernel scaffolding for it."""
    td = _StubTD(_real_prediagram(),
                 {1: _conv_vertex()})
    result = integrate_grouped_diagram(
        typed_diagrams=[td],
        combined_prefactors=[SR(1)],
        propagator_data={},
        ext_time_vars=_EXT_TIME_VARS,
    )
    assert result['status'] == 'failed', (
        'grouped path must refuse a ConvVertexType group; got '
        f"status={result['status']!r}"
    )
    assert 'convolution' in result['reason'].lower() or \
           'conv' in result['reason'].lower(), (
        f"reason should name the conv kernel; got {result['reason']!r}"
    )


# ── (3) Control: white source does NOT trip the guard ────────────────
def test_grouped_guard_does_not_trip_on_white_source():
    """A plain white ``SourceType`` carries no cumulant kernel, so the
    colored guard must NOT fire — the function proceeds PAST the guard
    to the next validation step.

    To prove pass-through cleanly (without depending on real propagator
    data / graph), we feed an ``ext_time_vars`` of the wrong length: the
    function then returns the DOWNSTREAM ``'ext_time_vars has length …'``
    failure, which is reachable only if the colored guard let the td
    through.  This pins down that the guard keys specifically on the
    colored kernel, not on sources in general.
    """
    td = _StubTD(_real_prediagram(),
                 {1: _white_source()})
    result = integrate_grouped_diagram(
        typed_diagrams=[td],
        combined_prefactors=[SR(1)],
        propagator_data={},
        ext_time_vars=[],                 # wrong length -> downstream fail
    )
    # The white source slips past the colored guard.  Whatever happens
    # next, the failure reason must NOT be the colored-kernel one.
    reason = (result.get('reason') or '')
    assert 'colored' not in reason.lower(), (
        'white SourceType wrongly tripped the colored-kernel guard; '
        f'reason={reason!r}'
    )
    assert 'convolution' not in reason.lower(), (
        'white SourceType wrongly tripped the conv-kernel guard; '
        f'reason={reason!r}'
    )
    # Reached the downstream length check -> colored guard passed it on.
    assert 'ext_time_vars' in reason, (
        'white SourceType should reach the downstream length check, '
        f'proving it passed the colored guard; got reason={reason!r}'
    )


# ── (4) End-to-end: white noise grouped == per-diagram (unaffected) ──
def _load_theory(name):
    path = os.path.join(
        os.path.dirname(__file__), '..', 'theories', name)
    spec = importlib.util.spec_from_file_location('theory', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.build()


def test_white_noise_grouped_still_matches_perdiag():
    """The guard must leave WHITE-noise theories on the fast grouped
    path and unchanged vs the per-diagram path.

    Uses the toy scalar quartic Langevin (white ``-D·xt²`` noise, no
    colored cgf term) at k=2 / 1-loop — the same theory the grouped
    Wick-permutation regression already exercises.
    """
    from api import compute_cumulants

    m = _load_theory('toy_quartic_double_well.theory.py')
    fundamental = {'mu': [-1.0], 'g': [1.0], 'D': [0.1]}

    def run(use_grouped):
        return compute_cumulants(
            m, k=2, max_ell=1,
            external_fields=[('x', 1), ('x', 1)],
            fundamental=fundamental,
            tau_max=3.0, tau_step=1.0,
            use_grouped_phase_j=use_grouped,
            parallel=False, use_cache=False, verbose=False,
        )

    r_grouped = run(True)
    r_perdiag = run(False)

    np.testing.assert_allclose(
        r_grouped['C_tau'], r_perdiag['C_tau'],
        rtol=1e-6, atol=1e-9,
        err_msg='white-noise grouped vs per-diagram disagree — the '
                'colored guard must not have changed the white-noise '
                'grouped path.',
    )
