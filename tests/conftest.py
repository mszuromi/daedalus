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
