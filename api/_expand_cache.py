"""
Disk cache for ``FieldTheory.expand()`` results.

Why this exists
---------------
`FieldTheory.expand()` is the dominant cost for non-trivial models
— for a 4-field × 2-population compartmental Bernoulli model at
``taylor_order=4`` it runs ~90 minutes single-threaded inside Sage's
multivariate ``taylor()``.  The output is the bigrade-classified
action dict ``ft._by_tp`` (key = ``(n_tilde, n_phys)``, value =
polynomial ring element), plus rebuilt ``_S_raw`` and a snapshot of
the pre-zero MF sector.

Crucially, ``_by_tp`` at ``taylor_order=N`` is a STRICT SUPERSET of
``_by_tp`` at any ``taylor_order=M`` with ``M <= N``: the bigrade
entries at total degree ``<= M`` are byte-identical (the Taylor
truncation only adds higher-order entries, never modifies lower
ones).  So:

  * **Downgrade is free.**  Asking for order 4 when order 6 is
    cached: load the order-6 cache and filter to total degree ``<= 4``.
  * **Upgrade re-runs taylor() fully** (current MVP).  The new
    higher-order file is saved alongside the existing one, so future
    downgrades back to the lower order are still free.
  * **Order-2 cache is privileged.**  At taylor_order=2 the cache
    already contains the (0,0)/(1,0)/(0,1) saddle sectors and the
    (1,1) bilinear propagator kernel.  That's everything the MF check
    + propagator construction need, regardless of how high the
    downstream cumulant calculation reaches.  ``api.precompute``
    populates this cache as a one-time, ~few-second pass.

Cache layout
------------
::

    saved_models/<model_slug>/
        propagator.sobj                       # taylor-order-independent
        expand_taylor2.sobj                   # base layer (MF + propagator)
        expand_taylor4.sobj                   # optional
        expand_taylor6.sobj                   # optional
        unique_typed_mult_v1_<ext>_k<k>_l<l>_taylor<N>.sobj
        manifest.json                         # PipelineCache bookkeeping

``<model_slug>`` is ``re.sub(r'[^A-Za-z0-9]+', '_', name).lower()`` —
the same slug `_propagator.py` and `_diagrams.py` use.

What gets pickled
-----------------
Each ``expand_taylor<N>.sobj`` holds a dict with:

  * ``'by_tp'`` — ``{(n_tilde, n_phys): dict_form}``, where
    ``dict_form`` is ``poly.dict()`` (exponent-tuple → SR coefficient).
    Storing the dict form, not the polynomial object, sidesteps
    polynomial-ring identity issues across pickle/unpickle.
  * ``'S_raw_dict'`` — same dict form for the rebuilt full polynomial.
  * ``'mf_sector_raw'`` — same dict form for the (0,0)/(1,0)/(0,1)
    pre-zero sectors (diagnostic only).
  * ``'ring_var_names'`` — list of generator names; used as a sanity
    check on load (the freshly-rebuilt ring must have matching
    generators in matching order).
  * ``'taylor_order'`` — the order this was computed at (may exceed
    the request the caller is asking for).
  * ``'n_tilde'`` — number of response variables (for MF-sector
    filtering checks).
  * ``'vertex_signature'`` — fingerprint of the operator-IR
    derivative-vertex form-factor table (``None`` for plain/temporal
    models).  The on-disk slug is only ``model['name']`` + taylor
    order, which does NOT capture the per-vertex form-factor / mode
    state (it lives on the namespace as an *action-eval side effect*,
    not in ``_by_tp``).  ``load_expand`` reconstructs the live table and
    rejects a bundle whose signature differs — otherwise a Model-B⊕KPZ
    1-loop value silently loads as the bare φ̃φ² number.  See
    :func:`vertex_form_factor_signature`.
"""
from __future__ import annotations

import hashlib
import json
import os
import re

# Sage's pickle helpers — these handle complex SR / polynomial-ring
# state better than plain ``pickle.dump`` for the small subset we
# round-trip here.
from sage.all import save as sage_save, load as sage_load


__all__ = [
    'cache_dir',
    'expand_cache_path',
    'list_cached_orders',
    'find_best_cached_order',
    'prepare_for_load',
    'save_expand',
    'load_expand',
    'downgrade_by_tp_dict',
    'vertex_form_factor_signature',
]


# ── Path helpers ──────────────────────────────────────────────────────


def _slug(model: dict) -> str:
    """Filesystem-safe slug for the cache directory."""
    return re.sub(r'[^A-Za-z0-9]+', '_', model['name']).strip('_').lower()


def cache_dir(model: dict, cache_dir_root: str = 'saved_models') -> str:
    """Return ``<cache_dir_root>/<model_slug>/`` for a given model.

    Migration: pre-July-2026 the cache root was named ``saved_theories``; if
    the old directory exists and the new one does not, rename it in place so
    warm caches survive the theory -> model rename."""
    legacy = cache_dir_root.replace('saved_models', 'saved_theories')
    if (legacy != cache_dir_root and not os.path.isdir(cache_dir_root)
            and os.path.isdir(legacy)):
        try:
            os.rename(legacy, cache_dir_root)
        except OSError:
            pass
    return os.path.join(cache_dir_root, _slug(model))


def expand_cache_path(model: dict, taylor_order: int,
                      cache_dir_root: str = 'saved_models') -> str:
    """Path to the expand-cache file at this order."""
    return os.path.join(cache_dir(model, cache_dir_root),
                        f'expand_taylor{int(taylor_order)}.sobj')


def list_cached_orders(model: dict,
                       cache_dir_root: str = 'saved_models') -> list[int]:
    """Return all taylor orders that have a cached expand result, sorted."""
    d = cache_dir(model, cache_dir_root)
    if not os.path.isdir(d):
        return []
    out: list[int] = []
    for fname in os.listdir(d):
        m = re.fullmatch(r'expand_taylor(\d+)\.sobj', fname)
        if m:
            out.append(int(m.group(1)))
    out.sort()
    return out


def find_best_cached_order(model: dict, target_order: int,
                           cache_dir_root: str = 'saved_models') -> int | None:
    """Return the smallest cached order ``>= target_order``, or ``None``.

    Choosing the smallest minimizes the post-load filter work — every
    bigrade we keep is one we have to coerce back into the freshly-
    built polynomial ring.
    """
    cached = list_cached_orders(model, cache_dir_root)
    candidates = [o for o in cached if o >= target_order]
    return min(candidates) if candidates else None


# ── _by_tp ↔ dict-form conversions ────────────────────────────────────


def _poly_to_dict_form(poly) -> dict:
    """Pickle-stable representation of a polynomial ring element.

    Returns ``{exp_tuple: SR_coeff}``.  ``poly.dict()`` is what Sage
    polynomial rings expose for this purpose; the SR coefficients
    survive pickle/unpickle cleanly.
    """
    return dict(poly.dict())


def _dict_form_to_poly(R, exp_to_coeff: dict):
    """Inverse of :func:`_poly_to_dict_form`: build a polynomial in
    ring ``R`` from its exponent-tuple → coefficient dict."""
    return R(exp_to_coeff) if exp_to_coeff else R.zero()


def _by_tp_to_dict_form(by_tp: dict) -> dict:
    """Convert ``{(a, b): poly}`` to ``{(a, b): poly.dict()}``."""
    return {key: _poly_to_dict_form(p) for key, p in by_tp.items()}


def _by_tp_from_dict_form(R, by_tp_dict: dict) -> dict:
    """Inverse of :func:`_by_tp_to_dict_form`."""
    return {key: _dict_form_to_poly(R, d) for key, d in by_tp_dict.items()}


# ── Downgrade filter ──────────────────────────────────────────────────


def downgrade_by_tp_dict(by_tp_dict: dict, target_order: int) -> dict:
    """Keep only bigrades whose total degree (a + b) is ``<= target_order``.

    Operates on the dict-form representation, so callable both before
    and after polynomial-ring rehydration.  Higher-degree entries are
    dropped (they're the surplus from a higher-order expansion).
    """
    return {key: poly_dict
            for key, poly_dict in by_tp_dict.items()
            if sum(key) <= target_order}


# ── Operator-IR form-factor signature ─────────────────────────────────


def _canon_chain(chain) -> list | None:
    """Canonical JSON-able form of an operator chain tuple.

    ``(('Lap',),)`` → ``[['Lap']]``;  ``(('Dx', 0),)`` → ``[['Dx', '0']]``.
    Every entry is stringified so the form round-trips identically
    through pickle and across Sage sessions.
    """
    if chain is None:
        return None
    return [[str(x) for x in entry] for entry in chain]


def vertex_form_factor_signature(ns) -> str | None:
    """Stable fingerprint of a model's operator-IR derivative-vertex
    form-factor table (``ns._operator_ir_vertex_terms``).

    Returns ``None`` for a model that does NOT use the operator IR
    (``ns._operator_ir`` falsey, or ``ns is None``) — so the expand
    cache of every pre-existing temporal / plain-vertex model keeps its
    ``None``-vs-``None`` match and continues to load unchanged.  For an
    operator-IR model it returns ``'operator_ir:' + <sha256[:16]>``
    built from each vertex term's ``mode``, physical-leg count, operator
    ``chain``, and (coupling-mixing) ``weight`` — the four pieces that
    decide which momentum form factor every interaction node carries
    (see
    :func:`engine.integration.spatial.pipeline_bridge.diagram_form_factor`).

    Adding the KPZ ``(∂ₓφ)²`` vertex to a Model-B-only model, dropping
    it, or re-weighting the mix all change the signature, so a stale
    on-disk expand bundle whose signature does not match the
    freshly-built model is rejected by :func:`load_expand` (which then
    falls back to a fresh ``expand()``).

    Stability note: the weight is serialised via ``str()`` of the
    already-``simplify_full``'d SR expression.  A purely cosmetic
    re-ordering across Sage versions would, at worst, force one extra
    fresh ``expand()`` (then a re-save with the new signature) — never a
    wrong load.
    """
    if ns is None or not getattr(ns, '_operator_ir', False):
        return None
    terms = getattr(ns, '_operator_ir_vertex_terms', None)
    canon: list = []
    for t in (terms or []):
        n_phys = t.get('n_phys')
        canon.append([
            str(t.get('mode')),
            int(n_phys) if n_phys is not None else None,
            _canon_chain(t.get('chain')),
            str(t.get('weight')),
        ])
    # Order-independent: the table-build order is not contractual.
    canon.sort(key=lambda e: json.dumps(e, sort_keys=True))
    blob = json.dumps(canon, sort_keys=True)
    return 'operator_ir:' + hashlib.sha256(blob.encode('utf-8')).hexdigest()[:16]


def _reconstruct_operator_ir_table(ft) -> None:
    """Re-populate ``ft._ns._operator_ir_vertex_terms`` for an
    operator-IR model that is being loaded from cache.

    The derivative-vertex form-factor table is a *side effect of
    evaluating the action lambda* — ``_lower_operator_ir_action`` runs
    inside ``model['action'](ns)`` — NOT of ``_build_namespace``.  A
    cache load deliberately skips ``FieldTheory.expand()`` (where that
    evaluation lives), so the loaded namespace would otherwise carry NO
    table: every derivative vertex then silently collapses to its plain
    value (form factor ``1``), e.g. a Model-B⊕KPZ 1-loop variance loads
    as the bare φ̃φ² number instead of the correct composite+perleg one.

    Re-running the action lambda is cheap (it builds the symbolic action
    and lowers the IR — no Taylor expansion) and reproduces the table
    bit-identically.  ``hasattr``-guarded so it is a no-op when the
    table is already present (a fresh ``expand()`` or a prior
    reconstruct) and skipped entirely for non-operator-IR models.
    """
    ns = getattr(ft, '_ns', None)
    if ns is None or not getattr(ns, '_operator_ir', False):
        return
    if hasattr(ns, '_operator_ir_vertex_terms'):
        return
    try:
        ft.model['action'](ns)
    except Exception:
        # Leave the table absent; the signature check below (or a
        # downstream fresh expand) catches the resulting inconsistency.
        pass


# ── Save / load (live FieldTheory) ────────────────────────────────────


def _get_ring_var_names(ft) -> list[str]:
    """Pull the polynomial ring variable names out of a FieldTheory.

    Canonical location is ``ft._ns._ring_var_names`` (set during
    ``_build_namespace``); fall back to enumerating ring generators
    if that attribute isn't there yet.
    """
    if ft._ns is not None and hasattr(ft._ns, '_ring_var_names'):
        return list(ft._ns._ring_var_names)
    return [str(g) for g in ft._R.gens()]


def prepare_for_load(ft) -> None:
    """Populate ``ft._ns``, ``ft._R``, ``ft._n_tilde`` from
    ``_build_namespace()`` without running the taylor expansion.

    Used by callers that intend to load ``_by_tp`` from disk: the
    namespace + ring must exist before the load can coerce the
    cached polynomials back into a live ring.

    For operator-IR models this also rebuilds the derivative-vertex
    form-factor table (``_reconstruct_operator_ir_table``), which a bare
    ``_build_namespace`` does not populate — see that helper's docstring.
    """
    if ft._ns is None or ft._R is None:
        ns, R, n_tilde = ft._build_namespace()
        ft._ns      = ns
        ft._R       = R
        ft._n_tilde = n_tilde
    # No-op for non-operator-IR models and when the table is already
    # present (e.g. the caller already ran expand()).
    _reconstruct_operator_ir_table(ft)


def save_expand(model: dict, ft, cache_dir_root: str = 'saved_models',
                verbose: bool = False) -> str:
    """Pickle ``ft._by_tp`` + ``_S_raw`` + ``_mf_sector_raw`` for this
    model at ``ft.taylor_order``.

    Returns the file path written.
    """
    target = ft.taylor_order
    path = expand_cache_path(model, target, cache_dir_root)
    os.makedirs(os.path.dirname(path), exist_ok=True)

    bundle = {
        'by_tp':           _by_tp_to_dict_form(ft._by_tp),
        'S_raw_dict':      (_poly_to_dict_form(ft._S_raw)
                            if ft._S_raw is not None else {}),
        'mf_sector_raw':   _by_tp_to_dict_form(
                              getattr(ft, '_mf_sector_raw', {}) or {}),
        'ring_var_names':  _get_ring_var_names(ft),
        'taylor_order':    int(target),
        'n_tilde':         int(ft._n_tilde),
        'vertex_signature': vertex_form_factor_signature(ft._ns),
        'cache_version':   2,
    }
    sage_save(bundle, path.removesuffix('.sobj'))
    if verbose:
        print(f'[expand-cache] saved order={target} → {path}')
    return path


def load_expand(model: dict, ft, target_order: int,
                cached_order: int | None = None,
                cache_dir_root: str = 'saved_models',
                verbose: bool = False) -> bool:
    """Populate ``ft._by_tp`` / ``ft._S_raw`` / ``ft._mf_sector_raw``
    from disk for the requested ``target_order``.

    ``cached_order`` lets the caller pick which on-disk file to read
    (caller usually picked it via :func:`find_best_cached_order`); if
    ``None`` we re-discover it ourselves.

    The caller must have already populated ``ft._ns``, ``ft._R``, and
    ``ft._n_tilde`` — typically by running ``FieldTheory._build_namespace``
    before this call.

    Returns ``True`` on success (and ``ft`` is now equivalent to a
    fresh ``expand()`` call at ``target_order``).  Returns ``False``
    if no usable cache was found or load failed; ``ft`` is left
    unchanged in that case.
    """
    if cached_order is None:
        cached_order = find_best_cached_order(model, target_order,
                                              cache_dir_root)
    if cached_order is None:
        return False
    path = expand_cache_path(model, cached_order, cache_dir_root)
    if not os.path.isfile(path):
        return False
    try:
        bundle = sage_load(path)
    except Exception as e:
        if verbose:
            print(f'[expand-cache] load failed for {path}: {e!r}')
        return False

    # Structural sanity: the freshly-rebuilt ring must agree on
    # variable names + order (otherwise a model edit invalidated this).
    expect_names = _get_ring_var_names(ft)
    cached_names = list(bundle.get('ring_var_names', []))
    if expect_names != cached_names:
        if verbose:
            print(f'[expand-cache] ring mismatch — stale cache.  '
                  f'expect {expect_names!r}, cached {cached_names!r}')
        return False

    if int(bundle.get('n_tilde', -1)) != int(ft._n_tilde):
        if verbose:
            print(f'[expand-cache] n_tilde mismatch — stale cache.')
        return False

    # Operator-IR form-factor signature.  The on-disk slug is only the
    # model name + taylor order; it does NOT capture the per-vertex
    # form-factor / mode table, which lives on the namespace as an
    # action-eval side effect (NOT in ``_by_tp``).  Reconstruct the live
    # table (cheap; no-op if ``prepare_for_load`` already did it) and
    # reject the bundle if its recorded signature differs — this catches
    # both a genuinely changed model that re-used the same slug AND any
    # pre-signature bundle for a derivative-vertex model (cached_sig is
    # ``None`` there, but the live operator-IR model signs non-``None``),
    # which would otherwise load with the form factors silently dropped.
    _reconstruct_operator_ir_table(ft)
    expect_sig = vertex_form_factor_signature(ft._ns)
    cached_sig = bundle.get('vertex_signature')
    if expect_sig != cached_sig:
        if verbose:
            print(f'[expand-cache] form-factor signature mismatch — stale '
                  f'cache.  expect {expect_sig!r}, cached {cached_sig!r}')
        return False

    # Filter to target_order if we loaded a higher one.
    by_tp_dict = bundle['by_tp']
    if cached_order > target_order:
        by_tp_dict = downgrade_by_tp_dict(by_tp_dict, target_order)

    ft._by_tp = _by_tp_from_dict_form(ft._R, by_tp_dict)
    # Rebuild S_raw from the (potentially-filtered) by_tp so it stays
    # consistent with the requested target_order rather than carrying
    # the loaded higher-order surplus.
    S_raw = ft._R.zero()
    for poly in ft._by_tp.values():
        S_raw = S_raw + poly
    ft._S_raw = S_raw

    ft._mf_sector_raw = _by_tp_from_dict_form(
        ft._R, bundle.get('mf_sector_raw', {}))

    # ── Restore _cumulant_kernels side effect ────────────────────────
    # A fresh ``FieldTheory.expand()`` calls ``_build_cumulant_action``
    # which has two effects: (a) it appends the symbolic noise-cumulant
    # action term to ``S_sr``, and (b) it populates
    # ``ns._cumulant_kernels[(noise, order, leg_key)]`` with kernel-fn
    # metadata that the downstream ``extract_source_types`` reads to
    # promote a plain ``SourceType`` monomial into a ``NoiseSourceType``
    # (so the Phase J integrator can substitute the user's
    # ``z_kappa_<...>`` placeholder with the actual K(τ_v) kernel).
    #
    # The cached ``_by_tp`` already captures effect (a) — the cumulant
    # action term is baked into the on-disk polynomial.  But effect (b)
    # is a *runtime side effect on the namespace*: ``ns._cumulant_kernels``
    # is NOT in the bundle (the kernel callables can be captured-closure
    # objects that don't round-trip cleanly through pickle), so a
    # cache-load leaves the dict empty.  Then ``extract_source_types``
    # finds no matching cumulant specs, never promotes the (2,0)
    # noise-source monomials to ``NoiseSourceType``, and the integrator
    # has nothing to substitute the ``z_kappa`` symbols with — every
    # colored-noise diagram silently collapses to 0.
    #
    # Cure: re-execute ``_build_cumulant_action`` for its side effect.
    # We discard the SR return value because the action term it
    # produces is already in ``_by_tp`` from the cache; we only need
    # ``ns._cumulant_kernels`` populated.  This is a no-op for models
    # without a ``correlated_noises`` block (the function returns SR(0)
    # immediately).  See bug write-up in commit-message body.
    try:
        from engine.core.field_theory import _build_cumulant_action
        _build_cumulant_action(ft._ns, model)
    except Exception as e:
        if verbose:
            print(f'[expand-cache] _cumulant_kernels rebuild failed '
                  f'({e!r}); noise-source models may silently produce 0. '
                  f'Falling back to a fresh expand() is the safe path.')
        return False

    if verbose:
        if cached_order == target_order:
            print(f'[expand-cache] hit at order={target_order} '
                  f'(exact, no filter)')
        else:
            print(f'[expand-cache] hit at order={cached_order}, '
                  f'filtered down to {target_order}')

    return True
