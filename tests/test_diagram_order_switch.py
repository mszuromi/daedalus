"""The diagram-order switch of ``tests/_diagram_order.py`` itself.

Reordering the prediagram records only permutes the per-prediagram blocks of
the typed-diagram list; the switch must also reorder the typed diagrams of
one prediagram among themselves (most of the list for a multi-field model),
keep each multiplicity with its diagram, and patch every binding of
``enumerate_unique_diagrams``.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))

from tests._diagram_order import (OLD_ENVS, ORDER_ENV, _MARK,      # noqa: E402
                                  apply_diagram_order,
                                  shuffled_enumerate_unique_diagrams,
                                  shuffled_load_prediagrams, signature_key,
                                  unwrapped_order_patches)


def _unwrap(fn):
    while getattr(fn, _MARK, False):
        fn = fn.__wrapped__
    return fn


# ── The typed-level wrapper on a hand-built enumeration ──────────────────────

def _fake_enumeration(flat_ok=True, mult_ok=True):
    tds = [object() for _ in range(7)]
    unique = {0: tds[:1], 1: tds[1:]}
    mults = {0: [1], 1: [1, 2, 3, 4, 5, 6]}
    if not mult_ok:
        mults[1] = mults[1][:-1]
    flat = list(tds) if flat_ok else tds[::-1]

    def enumerate_unique_diagrams(*args, k, max_ell, **kw):
        return ({e: list(v) for e, v in unique.items()},
                {e: list(v) for e, v in mults.items()}, list(flat))
    return enumerate_unique_diagrams, unique, mults


def _pairs(unique, mults, ell):
    return sorted((id(td), m) for td, m in zip(unique[ell], mults[ell]))


@pytest.mark.parametrize('mode', ['shuffle', 'reverse'])
def test_typed_wrapper_reorders_each_loop_order_with_its_multiplicities(mode):
    orig, unique, mults = _fake_enumeration()
    wrapped = shuffled_enumerate_unique_diagrams(orig, mode, seed=1)
    u, m, flat = wrapped(k=2, max_ell=1)
    assert list(u) == [0, 1]
    for ell in u:
        assert _pairs(u, m, ell) == _pairs(unique, mults, ell)
    assert [id(t) for t in flat] == [id(t) for e in u for t in u[e]]
    ids = [id(t) for t in u[1]]
    if mode == 'reverse':
        assert ids == [id(t) for t in unique[1][::-1]]
    else:
        assert ids != [id(t) for t in unique[1]]
    again = wrapped(k=2, max_ell=1)                      # fixed seed
    assert [id(t) for t in again[0][1]] == ids


@pytest.mark.parametrize('flat_ok, mult_ok', [(False, True), (True, False)])
def test_typed_wrapper_refuses_lists_it_cannot_keep_aligned(flat_ok, mult_ok):
    orig, _, _ = _fake_enumeration(flat_ok=flat_ok, mult_ok=mult_ok)
    with pytest.raises(AssertionError):
        shuffled_enumerate_unique_diagrams(orig)(k=2, max_ell=1)


# ── Bindings ─────────────────────────────────────────────────────────────────

def test_switch_patches_every_binding_and_undoes_cleanly(monkeypatch):
    import api._diagrams as dg
    import api.compute as compute
    from engine.enumeration import prediagram_cache as pdc

    before = (pdc.load_prediagrams, dg.enumerate_unique_diagrams,
              compute.enumerate_unique_diagrams)
    real_load, real_enum = _unwrap(before[0]), _unwrap(before[1])
    with monkeypatch.context() as mp:
        apply_diagram_order(mp, 'shuffle', 1)
        assert compute.enumerate_unique_diagrams is dg.enumerate_unique_diagrams
        # on the real functions, whatever a whole-run switch installed
        assert dg.enumerate_unique_diagrams.__wrapped__ is real_enum
        assert pdc.load_prediagrams.__wrapped__ is real_load
        shuffled_load = pdc.load_prediagrams
        shuffled_enum = dg.enumerate_unique_diagrams
        with monkeypatch.context() as mp2:
            apply_diagram_order(mp2, 'reverse')
            # replaces the outer shuffle instead of composing with it
            assert pdc.load_prediagrams is real_load
            assert compute.enumerate_unique_diagrams is \
                dg.enumerate_unique_diagrams
            assert dg.enumerate_unique_diagrams.__wrapped__ is real_enum
            undo = {(mod.__name__, name): value
                    for mod, name, value in unwrapped_order_patches()}
            assert undo == {
                ('api._diagrams', 'enumerate_unique_diagrams'): real_enum,
                ('api.compute', 'enumerate_unique_diagrams'): real_enum}
        assert (pdc.load_prediagrams, dg.enumerate_unique_diagrams) == \
            (shuffled_load, shuffled_enum)
    assert (pdc.load_prediagrams, dg.enumerate_unique_diagrams,
            compute.enumerate_unique_diagrams) == before


@pytest.mark.parametrize('env, value', [(OLD_ENVS[0], 'shuffle'),
                                        (OLD_ENVS[1], '3'),
                                        (ORDER_ENV, 'sideways')])
def test_session_switch_refuses_old_names_and_unknown_modes(env, value,
                                                            monkeypatch):
    import tests.conftest as cf
    for name in OLD_ENVS + (ORDER_ENV,):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv(env, value)
    with pytest.raises(pytest.UsageError):
        cf.pytest_configure(None)


# ── A real multi-field enumeration ───────────────────────────────────────────

class _Stop(Exception):
    pass


def _typed_lists(monkeypatch):
    """``coupled_rd_2species_1d`` k=2, ell<=1 as ``{ell: [(signature,
    multiplicity, block), ...]}`` in list order, where ``block`` numbers the
    prediagram the diagram came from.  ``compute_cumulants`` is stopped right
    after the enumeration, through whatever bindings are installed."""
    import daedalus as dd
    import api._diagrams as dg
    import api.compute as compute
    from api.compute import compute_cumulants

    cur = compute.enumerate_unique_diagrams
    seen = []

    def spy(*args, **kw):
        seen.append(cur(*args, **kw))
        raise _Stop

    with monkeypatch.context() as mp:
        mp.setattr(compute, 'enumerate_unique_diagrams', spy)
        mp.setattr(dg, 'enumerate_unique_diagrams', spy)
        with pytest.raises(_Stop):
            compute_cumulants(dd.load_model('coupled_rd_2species_1d')[0],
                              k=2, max_ell=1,
                              external_fields=[('a', 1), ('a', 1)],
                              tau_grid=np.array([0.0]),
                              chi_grid=np.array([1.0]), use_cache=False,
                              verbose=False, spatial_parallel=False)
    unique, mults, _ = seen[0]
    out = {}
    for ell in unique:
        blocks = {}
        out[ell] = [(signature_key(td), m,
                     blocks.setdefault(id(td.prediagram), len(blocks)))
                    for td, m in zip(unique[ell], mults[ell])]
    return out


def _block_orders(lists, ref):
    """The signature order inside each prediagram block of ``ref`` (keyed by
    the block's signature set), as it appears in ``lists``."""
    out = {}
    for ell, rows in ref.items():
        blocks = {}
        for sig, _, b in rows:
            blocks.setdefault(b, set()).add(sig)
        for sigs in blocks.values():
            out[(ell, frozenset(sigs))] = [s for s, _, _ in lists[ell]
                                           if s in sigs]
    return out


@pytest.fixture
def unswitched(monkeypatch):
    """Undo a whole-run ``DAEDALUS_TEST_DIAGRAM_ORDER`` for one test, so its
    "plain" lists are the enumerator's own."""
    for mod, name, value in unwrapped_order_patches():
        monkeypatch.setattr(mod, name, value)


def test_switch_reorders_typed_diagrams_within_a_prediagram(unswitched,
                                                            monkeypatch):
    """Measured: 49 one-loop diagrams in 4 prediagram blocks of 8 to 16,
    multiplicities 1, 2 and 4.  The prediagram shuffle alone keeps every
    block's internal order; the switch changes it, keeps each multiplicity
    with its diagram, and ``reverse`` reverses each loop order exactly."""
    from engine.enumeration import prediagram_cache as pdc

    plain = _typed_lists(monkeypatch)
    assert len(plain[1]) > len({b for _, _, b in plain[1]}) > 1
    assert len({m for _, m, _ in plain[1]}) > 1

    with monkeypatch.context() as mp:
        mp.setattr(pdc, 'load_prediagrams',
                   shuffled_load_prediagrams(pdc.load_prediagrams,
                                             'shuffle', 1))
        records_only = _typed_lists(monkeypatch)
    with monkeypatch.context() as mp:
        apply_diagram_order(mp, 'shuffle', 1)
        shuffled = _typed_lists(monkeypatch)
    with monkeypatch.context() as mp:
        apply_diagram_order(mp, 'reverse')
        reversed_ = _typed_lists(monkeypatch)

    want = _block_orders(plain, plain)
    assert _block_orders(records_only, plain) == want
    got = _block_orders(shuffled, plain)
    assert set(got) == set(want)
    assert any(got[key] != want[key] for key in want if len(want[key]) > 1)
    for ell in plain:
        assert sorted(r[:2] for r in shuffled[ell]) == \
            sorted(r[:2] for r in plain[ell])
        assert [r[:2] for r in reversed_[ell]] == \
            [r[:2] for r in plain[ell]][::-1]
