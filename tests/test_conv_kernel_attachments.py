"""
tests/test_conv_kernel_attachments.py
=====================================
Phase 1 of the conductance-vertex kernel-handling work
(``docs/conductance_vertex_kernels_design.md``).

Verifies that ``reduce_conv_in_action`` records each rule-4
``Conv(g, fluct) → g · fluct`` emission as a ``kernel → {attached
fluct vars}`` entry in the ``attachments_out`` dict, and that
``kernel_attachments_in_coefficient`` correctly identifies kernel
symbols sitting in an interaction-vertex coefficient.

These attachments are what subsequent phases (vertex extraction,
integrator) will use to substitute the proper ``ĝ(ω_leg)`` factor at
each interaction vertex.
"""
import sys

# Allow ``pytest`` from the project root without installing the package.
sys.path.insert(0, '/Users/matthewszuromi/Documents/Education/BU PhD/'
                   'Ocker Lab/Automated Feynman Calculations')

from sage.all import SR
from engine.core.convolution import (
    Conv, reduce_conv_in_action, kernel_attachments_in_coefficient,
)


# ─── Fixtures ────────────────────────────────────────────────────────
def _symbols():
    g, g1, g2 = SR.var('g g1 g2')
    v, n, vt, n1, n2 = SR.var('v n vt n1 n2')
    w, vstar, nstar = SR.var('w vstar nstar')
    return locals()


# ─── Single-Conv attachment ──────────────────────────────────────────
def test_single_conv_attaches_kernel_to_field():
    """``Conv(g, n)`` should record ``g → {n}``."""
    s = _symbols()
    attachments = {}
    reduced = reduce_conv_in_action(
        Conv(s['g'], s['n']),
        fluct_vars={s['n']},
        attachments_out=attachments,
    )
    assert reduced.expand() == s['g'] * s['n']
    assert attachments == {s['g']: {s['n']}}


def test_conductance_term_collects_attachment():
    """``v * Conv(g, n)`` is the conductance-synapse case.  After
    saddle-shift v→vstar+v, n→nstar+n, the (1,2) ``vt·v·n`` vertex
    carries a ``g`` factor — the attachment record must say that ``g``
    is attached to ``n`` (not v), so the integrator substitutes
    ``ĝ(ω_n)`` not ``ĝ(ω_v)``."""
    s = _symbols()
    attachments = {}
    expr = s['v'] * Conv(s['g'], s['n'])
    reduced = reduce_conv_in_action(
        expr, fluct_vars={s['v'], s['n']},
        attachments_out=attachments,
    )
    # After Conv reduction the kernel factor lives next to ``n``.
    assert reduced.expand() == s['g'] * s['n'] * s['v']
    # Attachment recorded exclusively to n, not v.
    assert attachments == {s['g']: {s['n']}}


def test_taylor_passthrough_does_not_break_attachments():
    """When ``taylor_order`` is set, rule-0 expands nonlinear args
    before applying rules 1–4.  The attachment record must still
    capture every leg the kernel ends up paired with."""
    s = _symbols()
    a = SR.var('a')
    attachments = {}
    # phi(n) = a*n^3 — nonlinear in fluct.
    expr = Conv(s['g'], a * s['n']**3)
    reduced = reduce_conv_in_action(
        expr, fluct_vars={s['n']},
        taylor_order=4,
        attachments_out=attachments,
    )
    # Rule 4 fires on the (single) cubic term, so g→{n}.
    assert reduced.expand() == a * s['g'] * s['n']**3
    assert attachments == {s['g']: {s['n']}}


# ─── Multi-Conv attachment ───────────────────────────────────────────
def test_distinct_kernels_distinct_attachments():
    """Two independent Conv's: ``Conv(g1, n1) + Conv(g2, n2)``."""
    s = _symbols()
    attachments = {}
    expr = Conv(s['g1'], s['n1']) + Conv(s['g2'], s['n2'])
    reduced = reduce_conv_in_action(
        expr, fluct_vars={s['n1'], s['n2']},
        attachments_out=attachments,
    )
    assert reduced.expand() == s['g1'] * s['n1'] + s['g2'] * s['n2']
    assert attachments == {
        s['g1']: {s['n1']},
        s['g2']: {s['n2']},
    }


def test_same_kernel_multiple_legs_accumulates():
    """A theory can reuse a single kernel symbol ``g`` for several
    legs (e.g. one shared synapse kernel coupling to two distinct
    presynaptic populations).  The attachment dict accumulates both
    fields, and the caller is expected to disambiguate per-vertex
    by index matching."""
    s = _symbols()
    attachments = {}
    expr = Conv(s['g'], s['n1']) + Conv(s['g'], s['n2'])
    reduced = reduce_conv_in_action(
        expr, fluct_vars={s['n1'], s['n2']},
        attachments_out=attachments,
    )
    assert reduced.expand() == s['g'] * s['n1'] + s['g'] * s['n2']
    assert attachments == {s['g']: {s['n1'], s['n2']}}


# ─── Detection in vertex coefficients ────────────────────────────────
def test_kernel_attachments_in_coefficient_basic():
    """After reduce + saddle expansion, the (1,2) vertex coefficient
    is ``-w*g``.  The helper should find ``g`` and return its
    attached leg ``n``."""
    s = _symbols()
    attachments = {}
    expr = -s['w'] * s['v'] * Conv(s['g'], s['n'])
    reduce_conv_in_action(
        expr, fluct_vars={s['v'], s['n']},
        attachments_out=attachments,
    )
    # Imagine the (1,2) vertex's coefficient — the SR factor sitting
    # in front of vt*v*n in the action polynomial.
    vertex_coeff = -s['w'] * s['g']
    detected = kernel_attachments_in_coefficient(vertex_coeff, attachments)
    assert detected == {s['g']: s['n']}


def test_kernel_attachments_in_coefficient_no_kernel():
    """A vertex whose coefficient has no kernel symbol returns
    empty dict — the (1,2) sector of a purely-local theory like the
    quad-exp ``-(exp(nt)-1)*phi(v)`` interaction shouldn't trigger
    any Conv-vertex special handling."""
    s = _symbols()
    attachments = {s['g']: {s['n']}}
    vertex_coeff = -SR.var('a') * s['vstar']  # no g present
    detected = kernel_attachments_in_coefficient(vertex_coeff, attachments)
    assert detected == {}


def test_kernel_attachments_in_coefficient_multi_leg_returns_frozenset():
    """When a kernel attaches to multiple legs, the helper returns
    a frozenset — the caller resolves the ambiguity via index
    matching against the vertex's leg list."""
    s = _symbols()
    attachments = {s['g']: {s['n1'], s['n2']}}
    vertex_coeff = -s['w'] * s['g']
    detected = kernel_attachments_in_coefficient(vertex_coeff, attachments)
    # Single key, but its value is the ambiguous-set
    assert set(detected.keys()) == {s['g']}
    val = detected[s['g']]
    assert isinstance(val, frozenset)
    assert val == frozenset({s['n1'], s['n2']})


def test_kernel_attachments_in_coefficient_filter_by_kernel_list():
    """Caller can restrict the scan to a specific kernel-symbol list
    (e.g. model['kernels']) to avoid false positives if a kernel
    name accidentally collides with an unrelated symbol the user
    wrote directly in the action."""
    s = _symbols()
    attachments = {s['g']: {s['n']}, s['g1']: {s['n1']}}
    vertex_coeff = -s['w'] * s['g'] * s['g1']
    # Only ask about g, not g1.
    detected = kernel_attachments_in_coefficient(
        vertex_coeff, attachments, kernel_symbols=[s['g']]
    )
    assert detected == {s['g']: s['n']}


# ─── Backward compatibility ──────────────────────────────────────────
def test_attachments_out_is_optional():
    """Existing callers that don't supply ``attachments_out`` get
    the legacy behaviour — no kernel records, no errors."""
    s = _symbols()
    reduced = reduce_conv_in_action(
        Conv(s['g'], s['n']),
        fluct_vars={s['n']},
    )
    assert reduced.expand() == s['g'] * s['n']


if __name__ == '__main__':
    import pytest
    pytest.main([__file__, '-v'])
