"""
Diagram-stage enumeration with disk caching.

Wraps the four-stage diagram pipeline:
    enumerate_prediagrams_all → enumerate_all_typed
                              → filter_causal
                              → deduplicate_typed_diagrams

into a single ``enumerate_unique_diagrams(...)`` call that caches the
final ``unique`` list per ``(model_tag, taylor_order, k, ell, ext_fields)``
as a sibling ``.sobj`` file under ``saved_theories/<theory>/``, alongside
the Taylor-expansion artefact ``saved_theories/<theory>/expand_taylor<N>.sobj``.

The cache slot lives in the same directory as the propagator cache
written by ``api._propagator``, so a single ``saved_theories/<theory>``
directory holds all symbolic+combinatorial state for a theory.

Cache invalidation
------------------
- Filename embeds ``k``, ``ell`` and a stable tag of ``external_fields``.
- Model field-list edits should bump ``model['name']`` (or remove the
  whole cache dir).  External-field permutations get distinct cache
  files automatically via the tag.
"""
from __future__ import annotations

import re

from engine.core.cache import PipelineCache
from engine.diagrams.causality import filter_causal
from engine.diagrams.symmetry import deduplicate_with_multiplicities
from engine.diagrams.type_assignment import enumerate_all as enumerate_all_typed
from engine.enumeration.loop_diagram_enumeration import (
    enumerate_all as enumerate_prediagrams_all,
)


def _ext_fields_tag(external_fields):
    """Stable filename-safe tag for a list of (str, int) external-field tuples."""
    return '_'.join(f'{name}{idx}' for name, idx in external_fields)


def _model_cache_dir(model, taylor_order, cache_dir_root):
    """Cache directory for a model's cross-call artefacts.

    Per-theory (NOT per-taylor-order): the propagator and any other
    taylor-order-independent assets live here.  Artefacts that DO
    depend on taylor_order (the diagram set, the expand result) carry
    that dependence in their filenames instead (``_taylor<N>`` suffix).
    """
    prop_tag = re.sub(r'[^A-Za-z0-9]+', '_', model['name']).strip('_').lower()
    return f'{cache_dir_root}/{prop_tag}'


def enumerate_unique_diagrams(
    ft,
    model: dict,
    *,
    k: int,
    max_ell: int,
    external_fields,
    G_ft,
    resp_idx,
    phys_idx,
    vtypes,
    stypes,
    cache_dir_root: str = 'saved_theories',
    use_cache: bool = True,
    parallel: bool = False,
    n_workers: int | None = None,
    verbose: bool = True,
):
    """
    Enumerate deduplicated typed causal diagrams for ``ell = 0 ... max_ell``.

    Parameters
    ----------
    ft : FieldTheory
        Already-expanded field theory (must satisfy ``ft.taylor_order``).
    model : dict
        Model spec, must contain ``model['name']`` for the cache key.
    k, max_ell : int
        External-leg count and max loop order.
    external_fields : list of (str, int)
        Length-k list of leaf field tuples.  Encoded in the cache key.
    G_ft : Sage matrix
        Symbolic propagator (rows=phys, cols=resp).  Used by typing
        only — its parameter values don't matter to the diagram set,
        only its zero/nonzero pattern.
    resp_idx, phys_idx : dict
        Field-index maps from ``build_field_index_map``.
    vtypes, stypes : lists
        Vertex/source type lists from ``extract_*_types``.
    cache_dir_root : str
    use_cache : bool
        If False, always recompute and never write.
    parallel : bool, default False
        If True, fan the per-prediagram type-assignment stage across a
        fork-based ``multiprocessing.Pool`` (see
        ``engine.diagrams.type_assignment.enumerate_all``).  Skipped on
        cache hits.
    n_workers : int or None, default None
        Worker count when ``parallel=True``.  ``None`` lets the
        underlying enumerator pick
        ``min(os.cpu_count(), len(prediagrams))``.
    verbose : bool

    Returns
    -------
    unique_by_ell : dict[int, list[TypedDiagram]]
    multiplicity_by_ell : dict[int, list[int]]
        Parallel to ``unique_by_ell``: ``multiplicity_by_ell[ell][i]``
        is the dedup-equivalence-class size for ``unique_by_ell[ell][i]``,
        needed to recover the correct 𝒮(Γ) when the per-vertex
        combinatorial formula misses physical-leg permutations at
        sink vertices (see ``deduplicate_with_multiplicities``).
    all_unique    : list[TypedDiagram]
        Concatenation of ``unique_by_ell.values()`` in ell order.
    """
    cache_dir = _model_cache_dir(model, ft.taylor_order, cache_dir_root)
    cache = PipelineCache(cache_dir)

    ext_tag = _ext_fields_tag(external_fields)
    # Bumped from ``unique_typed_*`` to invalidate caches written before
    # the multiplicity-aware dedup landed; old caches lack the
    # multiplicity field and would silently zero the bug-fix.
    #
    # The taylor_order suffix is now in the stage name (rather than the
    # directory) because the diagram set DOES depend on which vertices
    # got included — higher taylor_order brings in higher-degree
    # interaction vertices, which can produce additional valid
    # diagrams at the same (k, ell).  Sibling files for different
    # taylor orders coexist in the same theory's cache dir.
    # Cache version bumped 2026-05-26 (v1 → v2): the
    # ``extract_source_types`` upgrade that promotes cross-cumulant /
    # auto-cumulant source monomials to :class:`NoiseSourceType` (so
    # the Phase J non-local kernel substitution actually fires) was
    # not in scope when v1 caches were written.  Old v1 caches encode
    # plain :class:`SourceType` for what should be ``NoiseSourceType``,
    # which silently drops the noise-kernel substitution and
    # collapses every colored-noise diagram to the white-limit
    # answer.  Bumping the version sidesteps the per-leg integrity
    # check we'd otherwise need.
    # Cache version bumped 2026-06-10 (v2 → v3): ``diagram_signature``
    # became a complete isomorphism invariant (canonical form of the
    # coloured incidence digraph).  v2 caches were deduplicated with
    # the old incomplete signature, which COLLIDED non-isomorphic
    # diagram classes at k>=3 (e.g. 4 of the 11 a^3-sector 1-loop
    # classes at k=3 for OU+ax^2+bx^3), silently dropping their
    # integrals.  Loading a v2 cache would resurrect that bug.
    stage_name = f'unique_typed_mult_v3_{ext_tag}_taylor{ft.taylor_order}'

    unique_by_ell: dict[int, list] = {}
    multiplicity_by_ell: dict[int, list] = {}
    all_unique: list = []

    for ell in range(max_ell + 1):
        # ── Cache lookup ──────────────────────────────────────────
        if use_cache and cache.exists(stage_name, k=k, loop_order=ell):
            try:
                cached = cache.load(stage_name, k=k, loop_order=ell)
                unique = cached['unique']
                multiplicities = cached['multiplicities']
                if verbose:
                    print(f'      ell={ell}: loaded {len(unique)} unique '
                          f'diagrams from cache')
                unique_by_ell[ell] = unique
                multiplicity_by_ell[ell] = multiplicities
                all_unique.extend(unique)
                continue
            except Exception as e:
                if verbose:
                    print(f'      ell={ell}: cache load failed ({e!r}); '
                          f'rebuilding.')

        # ── Build the four stages ────────────────────────────────
        _, _, prediagrams, _ = enumerate_prediagrams_all(
            k=k, ell=ell, verbose=False,
        )
        typed = enumerate_all_typed(
            prediagrams, external_fields, vtypes, stypes,
            G_ft=G_ft,
            resp_index=resp_idx, phys_index=phys_idx,
            parallel=parallel, n_workers=n_workers,
        )
        causal, n_disc, _ = filter_causal(typed)
        unique, multiplicities = deduplicate_with_multiplicities(causal)

        if verbose:
            print(f'      ell={ell}: {len(prediagrams)} prediag → '
                  f'{len(typed)} typed → {len(causal)} causal → '
                  f'{len(unique)} unique')

        # ── Cache write ──────────────────────────────────────────
        if use_cache:
            try:
                cache.save(stage_name,
                           {'unique': unique,
                            'multiplicities': multiplicities},
                           k=k, loop_order=ell)
                if verbose:
                    print(f'             cached to '
                          f'{cache_dir}/{stage_name}_k{k}_l{ell}.sobj')
            except Exception as e:
                if verbose:
                    print(f'             cache save failed ({e!r}).')

        unique_by_ell[ell] = unique
        multiplicity_by_ell[ell] = multiplicities
        all_unique.extend(unique)

    return unique_by_ell, multiplicity_by_ell, all_unique
