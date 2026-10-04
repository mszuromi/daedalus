"""Order-independent diagram selection and record shuffling for tests.

The order of a diagram list depends on where its prediagrams came from: the
eager enumerator's generation order, or the sorted packed certificates of the
``streamed`` / ``v2`` / ``shipped`` sources (whose bytes depend on the
canonical-labelling backend).  A test that takes "the first live diagram"
therefore certifies whichever class happens to come first.  Tests select by
STRUCTURE instead (:func:`_pick_live`) and break any remaining tie by
``diagram_signature``, a complete isomorphism invariant, never by position.

:func:`order_patches` reorders the lists that carry the order downstream,
the prediagram records and each loop order's typed diagrams (with their
multiplicities); ``tests/conftest.py`` exposes it as the opt-in
``shuffle_diagram_order`` fixture and, for whole-file runs, through
``DAEDALUS_TEST_DIAGRAM_ORDER``.
"""
from __future__ import annotations

import random


def signature_key(td):
    """Sort key of a typed diagram: its ``diagram_signature`` as a string."""
    from engine.diagrams.symmetry import diagram_signature
    return str(diagram_signature(td))


def prefactor_value(prefactor, params):
    """``float`` of a scalar prefactor at ``params`` (an SR-keyed dict)."""
    from sage.all import SR
    return float(SR(prefactor).subs(params))


def live_matches(records, pred=None, params=None, tol=1e-14):
    """``[(signature_key, td, prefactor), ...]`` sorted by signature.

    ``records`` is a list of ``(typed_diagram, prefactor)`` pairs (one ``ell``
    entry of ``build_pipeline_records``).  A record is kept when its prefactor
    is live at ``params`` (``|value| > tol``; skipped when ``params`` is None)
    and ``pred(td)`` holds (skipped when ``pred`` is None).
    """
    out = []
    for td, pre in records:
        if params is not None and abs(prefactor_value(pre, params)) <= tol:
            continue
        if pred is not None and not pred(td):
            continue
        out.append((signature_key(td), td, pre))
    out.sort(key=lambda r: r[0])
    return out


def _pick_live(records, pred=None, params=None, *, unique=True):
    """The live record matching ``pred``, chosen without using list position.

    Returns ``(typed_diagram, prefactor)``.  With ``unique=True`` (default)
    the predicate must single out exactly one isomorphism class, so a test
    states which class it certifies; with ``unique=False`` the class with the
    smallest ``diagram_signature`` wins.

    Raises :class:`LookupError` when nothing matches and :class:`ValueError`
    when ``unique=True`` and several classes match.  Neither is an
    ``AssertionError``, so a broken pick inside an ``xfail(raises=...)`` test
    is never mistaken for the expected failure.
    """
    hits = live_matches(records, pred, params)
    if not hits:
        raise LookupError('no live record matches the structural predicate')
    n_cls = len({h[0] for h in hits})
    if unique and n_cls != 1:
        raise ValueError(
            f'the structural predicate matches {n_cls} classes, '
            f'expected exactly one')
    return hits[0][1], hits[0][2]


# ── Structure of a C-stack descriptor ──────────────────────────────────────

def cstack_structure(td):
    """Relabelling-invariant summary of a typed diagram's C-stack descriptor.

    Returns ``(bundles, self_loops, external)``: ``bundles`` is the sorted
    tuple of edge-kind bundles between pairs of distinct vertices joined by
    internal edges (e.g. ``(('C', 'R'),)`` for an R-C bubble), ``self_loops``
    the sorted kinds of the self-loop edges (tadpoles), and ``external`` the
    sorted kinds of the external-leg edges.
    """
    from engine.integration.spatial.diagram_descriptor import diagram_to_cstack
    d = diagram_to_cstack(td)
    pairs = {}
    for e in d.edges:
        if e.external or e.u == e.v:
            continue
        pairs.setdefault(frozenset((e.u, e.v)), []).append(e.kind)
    bundles = tuple(sorted(tuple(sorted(v)) for v in pairs.values()))
    self_loops = tuple(sorted(e.kind for e in d.edges
                              if not e.external and e.u == e.v))
    external = tuple(sorted(e.kind for e in d.edges if e.external))
    return bundles, self_loops, external


def has_structure(bundles, self_loops=(), external=None):
    """Predicate: ``cstack_structure(td)`` equals the given parts
    (``external=None`` leaves the external legs unconstrained)."""
    want_b = tuple(sorted(tuple(sorted(b)) for b in bundles))
    want_s = tuple(sorted(self_loops))
    want_e = None if external is None else tuple(sorted(external))

    def pred(td):
        b, s, e = cstack_structure(td)
        return b == want_b and s == want_s and (want_e is None or e == want_e)
    return pred


#: One-loop R-C bubble with a correlation line on one external leg.
RC_BUBBLE_C_LEG = has_structure([('C', 'R')], (), ('C', 'R'))
#: One-loop bubble of two correlation lines; both external legs retarded.
CC_BUBBLE = has_structure([('C', 'C')], (), ('R', 'R'))
#: Two-loop chain: two R-C bubbles joined by a correlation line, external
#: legs retarded.
RC_BUBBLE_CHAIN = has_structure([('C', 'R'), ('C', 'R'), ('C',)], (),
                                ('R', 'R'))


# ── Diagram-order switch ───────────────────────────────────────────────────
#
# Two lists carry the order to every consumer: the prediagram records
# (``prediagram_cache.load_prediagrams``) and, per loop order, the unique typed
# diagrams with their multiplicities (``api._diagrams.enumerate_unique_diagrams``,
# whose list holds one block of typed diagrams per prediagram).  Reordering
# the prediagram records only permutes those blocks; the typed level is what
# reorders the diagrams of one prediagram among themselves, which is most of
# the list for a multi-field model.  The switch reorders both.

#: Environment switch read by ``tests/conftest.py`` for a whole run:
#: ``shuffle`` or ``reverse`` (see :func:`order_patches`); unset / ``none`` /
#: ``off`` leaves every list in its own order.
ORDER_ENV = 'DAEDALUS_TEST_DIAGRAM_ORDER'
#: Seed for ``shuffle`` (default 0).
SEED_ENV = 'DAEDALUS_TEST_DIAGRAM_SEED'
#: Earlier names of the two switches; ``tests/conftest.py`` refuses them
#: rather than silently running unshuffled.
OLD_ENVS = ('DAEDALUS_TEST_PREDIAGRAM_ORDER', 'DAEDALUS_TEST_PREDIAGRAM_SEED')

_MODES = ('shuffle', 'reverse')
_MARK = '_diagram_order_wrapper'


def reorder(records, mode, seed, k, ell, level='prediagram'):
    """A reordered copy of ``records`` (``mode`` 'shuffle' or 'reverse').

    The shuffle is seeded per ``(seed, k, ell)`` and per ``level``
    ('prediagram' or 'typed'), so repeated loads of one cell in a process see
    the same order (bit-identity comparisons between two runs of the same
    test stay valid)."""
    out = list(records)
    if mode == 'reverse':
        out.reverse()
    elif mode == 'shuffle':
        key = f'{seed}:{int(k)}:{int(ell)}'
        if level != 'prediagram':
            key += f':{level}'
        random.Random(key).shuffle(out)
    else:
        raise ValueError(f'unknown diagram order mode {mode!r}')
    return out


def shuffled_load_prediagrams(orig, mode='shuffle', seed=0):
    """Wrap ``load_prediagrams`` so its records come back reordered."""
    def load_prediagrams(root, k, ell, **kw):
        records, source = orig(root, k, ell, **kw)
        return reorder(records, mode, seed, k, ell), source
    load_prediagrams.__wrapped__ = orig
    setattr(load_prediagrams, _MARK, True)
    return load_prediagrams


def shuffled_enumerate_unique_diagrams(orig, mode='shuffle', seed=0):
    """Wrap ``api._diagrams.enumerate_unique_diagrams`` so each loop order's
    typed diagrams come back reordered, their multiplicities with them.

    Applies to fresh enumerations and typed-diagram cache hits alike, and
    leaves what the wrapped function writes to its cache untouched.  The
    flat third list is rebuilt from the reordered per-``ell`` lists (it is
    their concatenation in ``ell`` order; checked)."""
    def enumerate_unique_diagrams(*args, **kw):
        unique_by_ell, mult_by_ell, all_unique = orig(*args, **kw)
        flat = [td for ell in unique_by_ell for td in unique_by_ell[ell]]
        if [id(td) for td in all_unique] != [id(td) for td in flat]:
            raise AssertionError(
                'enumerate_unique_diagrams: the flat list is no longer the '
                'per-ell lists concatenated; update the order switch')
        new_u, new_m, new_all = {}, {}, []
        for ell, unique in unique_by_ell.items():
            mults = mult_by_ell[ell]
            if len(mults) != len(unique):
                raise AssertionError(
                    f'ell={ell}: {len(unique)} diagrams but '
                    f'{len(mults)} multiplicities')
            idx = reorder(range(len(unique)), mode, seed, kw['k'], ell,
                          level='typed')
            new_u[ell] = [unique[i] for i in idx]
            new_m[ell] = [mults[i] for i in idx]
            new_all.extend(new_u[ell])
        return new_u, new_m, new_all
    enumerate_unique_diagrams.__wrapped__ = orig
    setattr(enumerate_unique_diagrams, _MARK, True)
    return enumerate_unique_diagrams


def _bindings(owner, name):
    """``[(module, name), ...]``: ``owner`` and every other loaded module that
    binds the object ``owner.<name>`` under the same name."""
    import sys
    value = getattr(owner, name)
    out = [(owner, name)]
    for mod in list(sys.modules.values()):
        d = getattr(mod, '__dict__', None)
        if mod is not owner and isinstance(d, dict) and d.get(name) is value:
            out.append((mod, name))
    return out


def _targets():
    """``[(module, name), ...]`` that carry the two lists' order: the loader
    and every binding of ``enumerate_unique_diagrams`` (``api.compute``
    imports it by name; it is imported here first, so its binding is
    patched now rather than captured from a patched module later)."""
    import api._diagrams as dg
    import api.compute  # noqa: F401
    from engine.enumeration import prediagram_cache as pdc
    return [(pdc, 'load_prediagrams')] + _bindings(
        dg, 'enumerate_unique_diagrams')


def _unwrapped(fn):
    """``fn`` with every wrapper of this switch peeled off."""
    while getattr(fn, _MARK, False):
        fn = fn.__wrapped__
    return fn


def order_patches(mode='shuffle', seed=0):
    """``[(module, name, wrapped), ...]``: set each attribute to serve every
    diagram list in ``mode`` order.

    ``shuffle``: the prediagram records and, independently, each loop
    order's typed diagrams (with their multiplicities) in a fixed-seed
    random order.  ``reverse``: each loop order's typed-diagram list exactly
    reversed; the prediagram records keep the loader's own order (reversing
    them too would put each prediagram's block of typed diagrams back in
    place).

    The wrappers go on the real functions, so they REPLACE any wrapper of
    this switch already installed (a whole-run ``DAEDALUS_TEST_DIAGRAM_ORDER``
    or an outer patch) rather than composing with it: the order a test sees
    under ``(mode, seed)`` is the same in every run.

    One permutation per seed is a probe, not a proof: a positional pick that
    the permutation happens to leave in place goes unnoticed."""
    if mode not in _MODES:
        raise ValueError(f'unknown diagram order mode {mode!r}')
    targets = _targets()
    pdc, name = targets[0]
    real_load = _unwrapped(getattr(pdc, name))
    out = [(pdc, name, shuffled_load_prediagrams(real_load, mode, seed)
            if mode == 'shuffle' else real_load)]
    dg, name = targets[1]
    wrapped = shuffled_enumerate_unique_diagrams(
        _unwrapped(getattr(dg, name)), mode, seed)
    out += [(mod, name, wrapped) for mod, name in targets[1:]]
    return out


def apply_diagram_order(mp, mode='shuffle', seed=0):
    """Install :func:`order_patches` through ``mp`` (a pytest ``monkeypatch``
    or one of its ``context()`` objects), which undoes them.  Replaces any
    order already installed (see :func:`order_patches`)."""
    for mod, name, value in order_patches(mode, seed):
        mp.setattr(mod, name, value)


def unwrapped_order_patches():
    """``[(module, name, original), ...]`` that undo the switch's wrappers on
    every target (for a module that pins the exact order; empty when
    nothing is wrapped)."""
    out = []
    for mod, name in _targets():
        cur = getattr(mod, name)
        orig = _unwrapped(cur)
        if orig is not cur:
            out.append((mod, name, orig))
    return out
