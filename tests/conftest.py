"""Shared fixtures: diagram-list order (see ``tests/_diagram_order.py``).

``shuffle_diagram_order`` (opt-in, function scope) serves every diagram list
of the test in a fixed-seed shuffled order: the prediagram records and, per
loop order, the typed diagrams with their multiplicities, so diagrams of one
prediagram are reordered among themselves too.  Code that picks a diagram by
its list position then gets another diagram than in a plain run, unless the
permutation happens to leave that position in place: a probe, not a proof.

``DAEDALUS_TEST_DIAGRAM_ORDER=shuffle`` (or ``reverse``) applies the same
reordering to a whole run, module-scoped fixtures included; the seed is
``DAEDALUS_TEST_DIAGRAM_SEED`` (default 0).  ``reverse`` reverses each loop
order's typed-diagram list.  Unset, nothing is patched.  A module that pins
the exact order (``test_prediagram_cache``) undoes the wrappers for itself
with ``unwrapped_order_patches``.  ``shuffle_diagram_order`` and
``apply_diagram_order`` REPLACE the whole-run order for their test instead
of composing with it, so the order they give is the same in every run; a
module-scoped result computed before them still carries the whole-run order.
"""
import os

import pytest

from tests._diagram_order import (OLD_ENVS, ORDER_ENV, SEED_ENV,
                                  apply_diagram_order, order_patches)


def _order_mode():
    mode = os.environ.get(ORDER_ENV, '').strip().lower()
    return None if mode in ('', 'none', 'off') else mode


def pytest_configure(config):
    old = [name for name in OLD_ENVS if os.environ.get(name, '').strip()]
    if old:
        raise pytest.UsageError(
            f'{", ".join(old)} was renamed: use {ORDER_ENV} / {SEED_ENV}')
    if _order_mode() not in (None, 'shuffle', 'reverse'):
        raise pytest.UsageError(
            f'{ORDER_ENV} must be shuffle, reverse or unset, '
            f'not {os.environ[ORDER_ENV]!r}')


@pytest.fixture(scope='session', autouse=True)
def _diagram_order_from_env():
    mode = _order_mode()
    if mode is None:
        yield
        return
    patches = order_patches(mode, seed=int(os.environ.get(SEED_ENV, '0')))
    saved = [(mod, name, getattr(mod, name)) for mod, name, _ in patches]
    for mod, name, value in patches:
        setattr(mod, name, value)
    try:
        yield
    finally:
        for mod, name, value in saved:
            setattr(mod, name, value)


@pytest.fixture
def shuffle_diagram_order(monkeypatch):
    """Serve every diagram list in a fixed-seed shuffled order (seed 1)."""
    apply_diagram_order(monkeypatch, 'shuffle', seed=1)


@pytest.fixture(autouse=True)
def _fresh_delay_order_cache():
    """A delayed model's automatic Pade order is cached for the process
    (``api.delay``); a test that asserts the order search starts from none."""
    import sys
    mod = sys.modules.get('api.delay')
    if mod is not None:
        mod.clear_order_cache()
    yield


# ── Sage assumption guard ────────────────────────────────────────────────
# Sage symbols and their assumptions are GLOBAL: a test that removes an
# assumption others hold, or leaves a symbol with conflicting domains, makes
# later tests order-dependent (two such failures in Oct 2026).  After every
# test (only once Sage is loaded) the assumptions are compared with those
# before it:
#   * an assumption that was held before is gone        -> the test FAILS;
#   * a NEW symbol is declared complex AND real/signed    -> the test FAILS;
#   * a symbol that had assumptions gains or loses one   -> reported in the
#     terminal summary only: model builders declare parameter domains
#     globally, so two models sharing a parameter name with different
#     domains accumulate both (engine design, not a test bug).
# First-time parameter domains of a model build are expected and ignored.
# DAEDALUS_TEST_SAGE_GUARD=off disables the guard.
import re as _re
import sys as _sys

_SAGE_GUARD_REPORT = []


def _assumptions_by_symbol(strs):
    out = {}
    for a in strs:
        m = _re.match(r'^([A-Za-z_][A-Za-z0-9_]*) (?:is |[<>]=? )', a)
        out.setdefault(m.group(1) if m else a, set()).add(a)
    return out


def _domain_kinds(strs):
    kinds = set()
    for a in strs:
        if ' is ' in a:
            kinds.add(a.split(' is ', 1)[1])
        elif _re.search(r' [<>]=? 0$', a):
            kinds.add('signed')
    return kinds


@pytest.fixture(autouse=True)
def _sage_assumption_guard(request):
    if (os.environ.get('DAEDALUS_TEST_SAGE_GUARD', '').strip().lower()
            == 'off' or 'sage.all' not in _sys.modules):
        yield
        return
    from sage.all import assumptions
    before = {str(a) for a in assumptions()}
    yield
    after = {str(a) for a in assumptions()}
    if after == before:
        return
    b, a = _assumptions_by_symbol(before), _assumptions_by_symbol(after)
    gone = sorted(before - after)
    # 'is complex' next to a real or signed declaration contradicts it;
    # '> 0' with 'is real' (or 'is integer') is compatible
    conflict = sorted(s for s in a if s not in b
                      and 'complex' in _domain_kinds(a[s])
                      and len(_domain_kinds(a[s])) > 1)
    changed = sorted(s for s in b if s in a and a[s] != b[s])
    if changed:
        _SAGE_GUARD_REPORT.append(
            (request.node.nodeid,
             [(s, sorted(b[s]), sorted(a[s])) for s in changed]))
    if gone or conflict:
        pytest.fail(
            f'Sage assumption leak: removed {gone} (held before this '
            f'test); conflicting domains left on new symbols '
            f'{[(s, sorted(a[s])) for s in conflict]}.  Use private symbol '
            f'names and forget() what the test assumed.', pytrace=False)


def pytest_terminal_summary(terminalreporter):
    if _SAGE_GUARD_REPORT:
        terminalreporter.section('Sage assumptions accumulated on shared '
                                 'symbols (report only)')
        for nodeid, items in _SAGE_GUARD_REPORT:
            terminalreporter.write_line(f'{nodeid}: {items}')
