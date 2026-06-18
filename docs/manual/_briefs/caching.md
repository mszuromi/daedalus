# Caching: Enumeration cache & Expand cache

> Subsystem slug: `caching`
> Primary source files:
> - `pipeline/_diagrams.py` — `enumerate_unique_diagrams(...)`, the diagram-enumeration cache wrapper.
> - `pipeline/_expand_cache.py` — disk cache for `FieldTheory.expand()` results.
> - `pipeline/_precompute.py` — `precompute(model)`, the one-time structural pass.
> - `msrjd/core/cache.py` — `PipelineCache`, the low-level stage-keyed `.sobj` store both caches build on.

---

## Overview

Daedalus turns a *model spec* (a Python dict describing an MSR-JD field theory) into
numbers — the connected cumulants of the fields, computed perturbatively from Feynman
diagrams. Two of the steps on that road are *brutally expensive* and *highly reusable*:

1. **Expanding the action.** Sage takes the symbolic MSR-JD action `S[φ̃, φ]` and runs a
   multivariate Taylor expansion around the mean-field saddle to a chosen `taylor_order`.
   For a 4-field × 2-population compartmental Bernoulli theory at `taylor_order=4` this
   single call runs *roughly 90 minutes single-threaded* inside Sage's `taylor()`. The
   output is the bigrade-classified action dictionary `ft._by_tp` (described below).

2. **Enumerating diagrams.** For each loop order `ℓ = 0 … max_ell` the pipeline
   enumerates every topologically-distinct, type-consistent, causal, *non-isomorphic*
   Feynman diagram with `k` external legs. This is a four-stage combinatorial pipeline
   (`prediagrams → typed → causal → unique`) whose cost grows fast with `k` and `ℓ`.

Neither result depends on the *numerical* parameter values a user later plugs in — they
depend only on the *structure* of the theory (its field list, its interaction vertices,
the requested `k`, `ℓ`, and external-field labels). So both are cached on disk, keyed by
structure, and reloaded across notebook restarts and across runs that vary only the
numerical inputs.

This subsystem is the caching layer for those two artefacts. It sits between the model
definition and the numerical integration:

```
TheoryBuilder.build()  →  model dict
        │
        ▼
   precompute(model)  ── one-time structural pass ──┐
        │   (expand@order2 + propagator + MF check) │  writes
        ▼                                            ▼
  compute_cumulants(...)                  saved_theories/<slug>/
        │                                    ├── propagator.sobj
        ├─[1] FieldTheory.expand            ├── expand_taylor2.sobj
        │     ← EXPAND CACHE                ├── expand_taylor4.sobj
        │       (pipeline/_expand_cache.py) ├── expand_taylor6.sobj
        │                                    ├── unique_typed_mult_v3_<ext>_k<k>_l<l>_taylor<N>.sobj
        ├─[5] enumerate_unique_diagrams      └── manifest.json
        │     ← ENUMERATION CACHE
        │       (pipeline/_diagrams.py)
        ▼
   diagrams → integration → cumulants
```

- **What feeds the expand cache:** a freshly-built `FieldTheory` after `expand()` ran
  (`ft._by_tp`, `ft._S_raw`, `ft._mf_sector_raw`). **What consumes it:** the propagator
  builder, the sanity check, the vertex/source extractors, and the diagram pipeline — all
  of which read `ft._by_tp`.
- **What feeds the enumeration cache:** the four-stage diagram pipeline's final `unique`
  list and per-class `multiplicities`. **What consumes it:** the coefficient-classification
  + kernel-grouping + integration stages in `pipeline/compute.py` (and the spatial bridge),
  which walk `unique_by_ell` and integrate each diagram.

Both caches share one physical store, `PipelineCache` (`msrjd/core/cache.py`), which
writes Sage `.sobj` files into a single per-theory directory and maintains a human-readable
`manifest.json` index. The *expand* cache bypasses `PipelineCache` and calls Sage's
`save`/`load` directly (because its files are keyed by taylor order, not by `(k, ℓ)`), but
it writes into the *same directory*.

The central design principle, stated three times across the docstrings, is **decoupling**:

- The **expand cache is keyed by `taylor_order` only** (one file per order), *not* by
  `k`, `ℓ`, or external fields. A higher-order expansion is a strict superset of a lower
  one, so a high-order cache serves any lower-order request for free (a "downgrade").
- The **enumeration cache is keyed by `(model, taylor_order, k, ℓ, external-fields)`** — a
  finer key — because the diagram *set* does depend on which interaction vertices are in
  scope, which depends on `taylor_order`.

---

## The math

### MSR-JD action and its Taylor expansion (what the *expand* cache stores)

Daedalus works in the Martin–Siggia–Rose–Janssen–De Dominicis (MSR-JD) field-theory
representation of a stochastic dynamical system. A Langevin equation
`∂_t φ = F[φ] + noise` becomes a path integral with an action

```
    S[φ̃, φ] = ∫ dt { φ̃ (∂_t φ − F[φ]) − ½ φ̃ B φ̃ + … }
```

over a *physical* field `φ` and a *response* field `φ̃` (the "tilde" field). Cumulants are
read off from a perturbative expansion of `exp(−S)` around the mean-field (saddle) point.

To do perturbation theory you need the action expanded in powers of the *fluctuations*
about the saddle. Write `φ = φ* + δφ`, `φ̃ = φ̃* + δφ̃`. Daedalus expands `S` as a
multivariate Taylor series in `(δφ̃, δφ)` and **classifies each monomial by its bigrade**

```
    (n_tilde, n_phys)  =  (# response-field factors, # physical-field factors).
```

The dictionary `ft._by_tp` is exactly this classification:

```
    _by_tp[(n_tilde, n_phys)]  =  Σ (all action monomials of that bigrade)
```

Each value is a *polynomial-ring element*. The total degree of a bigrade is
`n_tilde + n_phys`. The Taylor truncation at order `N` keeps exactly the bigrades with

```
    n_tilde + n_phys  ≤  N.
```

The bigrades carry physical meaning:

- `(0,0)`, `(1,0)`, `(0,1)` — the **mean-field (MF) / saddle sectors**. The saddle
  condition is precisely that these vanish at `φ*`, `φ̃*` (this is what `sanity_check`
  verifies). Order 0 in the fluctuations, this is the classical equation of motion.
- `(1,1)` — the **bilinear / propagator kernel**. The free (Gaussian) propagator `G` is
  the inverse of the matrix of `(1,1)` coefficients. The propagator builder reads exactly
  this sector.
- `(n_tilde, n_phys)` with total degree `≥ 3` — the **interaction vertices**. A monomial
  of bigrade `(a, b)` with `a + b = d` is a degree-`d` vertex with `a` response legs and
  `b` physical legs. These are the Feynman rules consumed by the diagram pipeline.

**The superset/downgrade theorem** (the mathematical justification for the expand cache,
stated in `_expand_cache.py:14–18`): the Taylor expansion at order `N` is a strict
superset of the expansion at any order `M ≤ N`. The bigrade entries of total degree `≤ M`
are *byte-identical* between the two — Taylor truncation only adds higher-order terms,
never modifies the lower ones. Concretely:

```
    by_tp@N  ⊇  by_tp@M    (for M ≤ N)
    by_tp@N restricted to {(a,b): a+b ≤ M}  ==  by_tp@M
```

This is why `downgrade_by_tp_dict` (keep only `sum(key) ≤ target_order`) is *exact*, not
an approximation, and why a cached order-6 file fully satisfies an order-2 request.

### `taylor_order` from `(k, max_ell)` (the link between the two caches)

The default rule (`pipeline/compute.py:278–279`) is

```
    taylor_order  =  max( k + 2·max_ell , 2 ).
```

The intuition: a connected diagram with `k` external legs and `ℓ` loops needs interaction
vertices whose total leg-count covers the external legs (`k`) plus the internal
propagator legs. Each loop adds (heuristically) two leg-degrees of internal structure,
giving the `2·max_ell` term; the floor of `2` guarantees the propagator sector `(1,1)` is
always present even at `k=2, max_ell=0`. This is the *taylor order needed so that every
diagram at `(k, ℓ)` has its vertices in scope*. A historical floor of `4` was dropped to
`2` once the cache layout siblings different orders cleanly (see the comment at
`compute.py:270–279`) — saving ~90 min on heavy Bernoulli theories at `k=2, max_ell=0`.

This formula is the bridge between the two caches: it tells you which `taylor_order` the
expand cache must serve to satisfy a given diagram request, and it tells you which
`taylor<N>` suffix appears in the enumeration-cache filename.

### Diagram enumeration and the equivalence relation (what the *enumeration* cache stores)

The diagram pipeline produces, per loop order `ℓ`, the list of **non-isomorphic typed
causal diagrams** plus a per-diagram **multiplicity**. The four stages:

1. **Prediagrams** (`enumerate_prediagrams_all(k, ℓ)`): all undirected topologies — trees
   → topologies → prediagrams — with `k` external legs and `ℓ` loops, *before* any field
   assignment. The number of vertices is bounded; the proven enumeration bound is
   `k + 3ℓ − j − 1` (see the `enumeration_bound_fix` memory note).
2. **Typing** (`enumerate_all_typed`): assign each vertex an interaction type from the
   theory's vertex list (`vtypes`) or a source type (`stypes`), and assign each internal
   edge a propagator type, consistent with the propagator's zero/nonzero pattern `G_ft`.
   Many prediagrams admit several typings; many admit none.
3. **Causality** (`filter_causal`): drop diagrams that violate the retarded
   (Itô/causal) structure of the MSR-JD propagator — e.g. those containing an acausal
   response-to-response loop. Returns `(causal, n_discarded, _)`.
4. **Deduplication** (`deduplicate_with_multiplicities`): merge diagrams that are
   *isomorphic as coloured graphs*. The equivalence used is `diagram_signature`, the
   **canonical form of the coloured incidence digraph** (leaves coloured by field only,
   `fix_external=False`). Two typed diagrams get the same signature **iff** they are
   isomorphic — same vertex types, same edge propagator types, same topology, with
   same-field external leaves allowed to permute. The integration layer re-expands the
   permuted leaves via its `_all_mappings` sum divided by `external_wick_compensation`.

The **symmetry factor** `𝒮(Γ)` of a diagram (the Feynman-rule combinatorial weight) is
carried entirely by `combinatorial_factor` — the orbit–stabiliser count
`∏ n_leg! / |Aut_fixed_ext(Γ)|` of Wick pairings on the *representative* (Path A). The
dedup **multiplicity** (the size of each equivalence class) is therefore **diagnostic
only**: multiplying by it would *double-count*, because the merged copies are isomorphic
and their pairings are already inside `𝒮(Γ)`. The multiplicity is retained in the cache
purely for diagnostics and historical-compatibility reasons (it once compensated for an
*incomplete* signature — see Gotchas).

---

## External tools used

This subsystem touches three external systems: **SageMath** (heavily), the Python standard
library (`re`, `os`, `json`, `hashlib`, `time`, `traceback`, `datetime`), and — only
transitively, through the four-stage pipeline it wraps — **nauty/sympy/networkx**. Below,
each is explained from scratch with the exact lines that use it.

### SageMath (`sage`)

**What it is.** SageMath is a large open-source mathematics system built on top of Python.
It bundles symbolic algebra, exact and arbitrary-precision arithmetic, polynomial rings,
graph theory (via an embedded copy of **nauty**), and number theory under one Python API.
When you `import` from `sage.all`, you get Sage's global namespace — its symbolic ring
`SR`, its `matrix` constructor, its graph classes, and crucially its **object serialization
helpers** `save` and `load`.

In Daedalus, the symbolic action, the polynomial ring elements stored in `_by_tp`, the
propagator matrix `G_ft`, and the `TypedDiagram` objects (which wrap Sage `DiGraph`s) are
all *Sage objects*. Plain Python `pickle` does not round-trip them reliably — Sage objects
carry references to parent structures (rings, fields) whose identity matters. So Daedalus
uses Sage's own pickle wrappers everywhere it touches disk.

**`.sobj` files.** A `.sobj` ("Sage object") file is Sage's serialization format: it is a
gzip-compressed pickle produced by `sage.all.save(obj, path)` and read back by
`sage.all.load(path)`. The convention is that `save` *appends* the `.sobj` extension if you
don't supply it — which is why every call here strips it first.

**Exact uses in this subsystem:**

- `msrjd/core/cache.py:34`
  ```python
  from sage.all import save as sage_save, load as sage_load
  ```
  - `cache.py:108` — `sage_save(data, path.removesuffix('.sobj'))` writes a stage result.
    The `.removesuffix('.sobj')` strips the extension so that Sage's auto-append produces
    exactly one `.sobj` and not `....sobj.sobj`.
  - `cache.py:123` — `return sage_load(path)` reads it back. Here the full `.sobj` path is
    passed (Sage's `load` accepts either form).

- `pipeline/_expand_cache.py:85`
  ```python
  # Sage's pickle helpers — these handle complex SR / polynomial-ring
  # state better than plain ``pickle.dump`` for the small subset we
  # round-trip here.
  from sage.all import save as sage_save, load as sage_load
  ```
  - `_expand_cache.py:345` — `sage_save(bundle, path.removesuffix('.sobj'))` writes the
    expand bundle dict. Note the bundle's *values* are already plain Python dicts/lists
    (the dict-form representation, see below) so even though Sage is the serializer, most
    of the bundle is pickle-trivial — except the `SR` coefficients inside the dict form,
    which is exactly where Sage's serializer earns its keep.
  - `_expand_cache.py:380` — `bundle = sage_load(path)` reads it back, wrapped in a
    `try/except` that returns `False` (cache miss) on any failure.

**The polynomial-ring `.dict()` API.** Sage polynomial-ring elements expose a method
`poly.dict()` that returns `{exponent_tuple: coefficient}`. The expand cache stores this
*dict form* rather than the polynomial object itself, because polynomial-ring objects have
parent-identity issues across pickle/unpickle (the unpickled ring is a *different* Python
object than the freshly-built one, even with identical generators), whereas the dict form
— integer exponent tuples plus `SR` coefficients — survives cleanly. Reconstruction calls
`R(exp_to_coeff)`, the ring's call operator, which builds a polynomial from such a dict.
See `_poly_to_dict_form` / `_dict_form_to_poly`.

**Graph canonical labelling (nauty, transitive).** `diagram_signature` (in
`msrjd/diagrams/symmetry.py`, called by `deduplicate_with_multiplicities`, called here)
uses Sage's `DiGraph.canonical_label(partition=…, certificate=True)`. Sage delegates the
canonical-form computation to **nauty** (B. McKay's graph-automorphism library, vendored
inside Sage). This is what makes the dedup signature a *complete* isomorphism invariant.
This subsystem never imports nauty directly — it inherits the dependency through the
`deduplicate_with_multiplicities` call at `_diagrams.py:186`.

### nauty (transitive, via Sage)

**What it is.** *nauty* ("No AUTomorphisms, Yes?") is a C library for computing graph
automorphism groups and canonical labellings. Given a coloured graph it returns a
canonical form such that two graphs have the *same* canonical form **iff** they are
isomorphic. Sage embeds nauty and exposes it through `Graph`/`DiGraph` methods like
`canonical_label`. The diagram-dedup stage uses it to detect isomorphic Feynman diagrams.
The enumeration cache *stores the output* of this computation (the surviving `unique`
representatives) so the nauty work is done once per `(k, ℓ, …)`.

### sympy / numba / scipy / networkx

These libraries are **not** used directly by any of the four caching files. The four-stage
diagram pipeline that `enumerate_unique_diagrams` wraps uses Sage's graph machinery
(nauty), not networkx. The symbolic work is Sage's `SR`, not sympy. The numerics (numba,
scipy) live downstream in the integration layer, past where the caches hand off. They are
listed here only to record that this subsystem touches *none* of them — a reader of the
manual should not expect to find them in the caching chapter.

### Python standard library

- `re` (`_diagrams.py:26`, `_expand_cache.py:80`, transitively `_propagator.py`): builds
  filesystem-safe theory slugs via `re.sub(r'[^A-Za-z0-9]+', '_', name)` and parses
  `expand_taylor(\d+)\.sobj` filenames with `re.fullmatch`.
- `os` (`_expand_cache.py:79`, `_precompute.py:48`, `cache.py:31`): path joining,
  `os.makedirs(..., exist_ok=True)`, `os.path.isfile`, `os.listdir`, `os.remove`.
- `json` (`_expand_cache.py:78`, `cache.py:30`): the manifest is JSON; the operator-IR
  signature canonicalises its table via `json.dumps(..., sort_keys=True)`.
- `hashlib` (`_expand_cache.py:77`): `hashlib.sha256(blob.encode()).hexdigest()[:16]`
  builds the operator-IR form-factor signature.
- `time` / `traceback` (`_precompute.py:49–50`): wall-time measurement and formatted
  exception capture.
- `datetime` (`cache.py:32`): ISO timestamps in the manifest.

---

## Components

### `msrjd/core/cache.py` — `PipelineCache`

The low-level, stage-keyed disk store. Both higher-level caches write into directories it
manages; the enumeration cache uses it directly.

#### `class PipelineCache` — `cache.py:37`

A stage-based disk cache for diagram-pipeline results. One directory per model; files keyed
by `(stage_name, k, loop_order)`.

- **`STAGES`** (`cache.py:48`) — a tuple of well-known stage names
  (`'prediagrams'`, `'filtered'`, `'typed'`, `'unique_typed'`, `'kernel_groups'`,
  `'integrand_results'`). Documented as *informational, not enforced* — you can save any
  stage name.

#### `PipelineCache.__init__(self, root)` — `cache.py:57`
- **Takes:** `root` (str) — cache directory path.
- **Does:** stores `self.root = os.path.expanduser(root)` (expands `~`). Does not create
  the directory yet — that happens lazily on first `save`.

#### `PipelineCache._to_int(val)` (staticmethod) — `cache.py:62`
- **Takes:** `val` (possibly a SageMath `Integer`, numpy int, `None`, or plain int).
- **Returns:** `None` if `val is None`, else `int(val)`.
- **Why it exists:** `k` and `loop_order` often arrive as Sage `Integer`s; coercing to
  plain `int` keeps filenames and manifest JSON clean (`k2` not `kInteger(2)`).

#### `PipelineCache._stage_key(cls, stage, k=None, loop_order=None)` (classmethod) — `cache.py:69`
- **Takes:** stage name + optional `k`, `loop_order`.
- **Returns:** a filename stem string, joining with `_`: `stage`, then `k{k}` if `k` is
  not None, then `l{loop_order}` if not None. E.g. `('unique_typed', 2, 1)` →
  `'unique_typed_k2_l1'`.
- **Step-by-step:** coerce both ints via `_to_int`; build `parts=[stage]`; append `k…`
  and `l…` conditionally; `'_'.join(parts)`.

#### `PipelineCache._sobj_path(self, stage, k=None, loop_order=None)` — `cache.py:81`
- **Returns:** `os.path.join(self.root, key + '.sobj')` where `key` is the stage key.

#### `PipelineCache.exists(self, stage, k=None, loop_order=None)` — `cache.py:87`
- **Returns:** `os.path.isfile(self._sobj_path(...))` — whether the cached file is present.

#### `PipelineCache.save(self, stage, data, k=None, loop_order=None)` — `cache.py:91`
- **Takes:** `stage` (str), `data` (any Sage-serializable object), optional `k`,
  `loop_order`.
- **Does:** `os.makedirs(self.root, exist_ok=True)`; compute path; `sage_save(data,
  path.removesuffix('.sobj'))`; then `self._update_manifest(stage, k, loop_order)`.
- **Returns:** nothing.

#### `PipelineCache.load(self, stage, k=None, loop_order=None)` — `cache.py:111`
- **Returns:** the unpickled object via `sage_load(path)`.
- **Raises:** `FileNotFoundError` with a descriptive message if the file is absent. (Note:
  the *enumeration cache* calls `exists` first and wraps `load` in a `try/except` — it
  never relies on this exception, but it is the contract.)

#### `PipelineCache.get_or_compute(self, stage, compute_fn, k=None, loop_order=None)` — `cache.py:125`
- **Takes:** `compute_fn` — a zero-arg callable.
- **Does:** if cached, `return self.load(...)`; else call `compute_fn()`, `save` the
  result, return it. The classic memoise pattern. (The enumeration cache does *not* use
  this — it does manual `exists`/`load`/`save` so it can print per-stage progress and
  handle load failures gracefully.)

#### `PipelineCache._manifest_path(self)` — `cache.py:150`
- **Returns:** `os.path.join(self.root, 'manifest.json')`.

#### `PipelineCache._load_manifest(self)` — `cache.py:153`
- **Returns:** the parsed manifest dict, or a fresh `{'entries': [], 'created': <iso>}` if
  the file is missing **or corrupt** (`json.JSONDecodeError`/`ValueError` are swallowed and
  a fresh structure returned). Robust to partial writes.

#### `PipelineCache._update_manifest(self, stage, k, loop_order)` — `cache.py:168`
- **Does:** load manifest; remove any existing entry with the same `key`; append a new
  entry `{key, stage, k, loop_order, saved_at:<iso>}`; set `manifest['updated']`; write
  back with `json.dump(..., indent=2)`. The manifest is a *human-readable index*, not a
  correctness-critical structure — the `.sobj` files are the source of truth.

#### `PipelineCache.list_cached(self)` — `cache.py:190`
- **Returns:** `self._load_manifest().get('entries', [])` — the list of entry dicts.

#### `PipelineCache.clear(self, stage=None, k=None, loop_order=None)` — `cache.py:194`
- **Does:** if all three args are `None`, `shutil.rmtree(self.root)` — nuke the whole
  cache dir. Otherwise remove just the one `.sobj` file and prune its manifest entry.

#### `PipelineCache.__repr__(self)` — `cache.py:221`
- **Returns:** `PipelineCache('<root>', <N> entries)`.

---

### `pipeline/_diagrams.py` — the enumeration cache

#### `_ext_fields_tag(external_fields)` — `_diagrams.py:37`
- **Takes:** `external_fields` — a list of `(name: str, idx: int)` tuples (length `k`).
- **Returns:** a filename-safe string `'_'.join(f'{name}{idx}' for name, idx in …)`. E.g.
  `[('v', 1), ('v', 2)]` → `'v1_v2'`. This tag is what makes *different external-field
  permutations* land in distinct cache files automatically (cache invalidation note,
  `_diagrams.py:19–22`).

#### `_model_cache_dir(model, taylor_order, cache_dir_root)` — `_diagrams.py:42`
- **Takes:** the model dict, a `taylor_order` (**accepted but deliberately unused** — see
  below), and the cache root.
- **Returns:** `f'{cache_dir_root}/{prop_tag}'` where
  `prop_tag = re.sub(r'[^A-Za-z0-9]+', '_', model['name']).strip('_').lower()`.
- **Key point (and a subtle signature wart):** the directory is **per-theory, NOT
  per-taylor-order**. The `taylor_order` parameter is in the signature but never read —
  the function deliberately keeps everything for a theory in one directory and lets the
  *filenames* carry the `taylor_order` dependence (`_taylor<N>` suffix). This is the same
  slug `_expand_cache._slug` and `_propagator.py` produce, so all three write into one
  directory. (Recorded as an open question: the unused parameter is harmless but a likely
  vestige of the old per-taylor-order directory layout.)

#### `enumerate_unique_diagrams(...)` — `_diagrams.py:54`
The headline function. Signature:
```python
def enumerate_unique_diagrams(
    ft, model, *, k, max_ell, external_fields, G_ft, resp_idx, phys_idx,
    vtypes, stypes, cache_dir_root='saved_theories', use_cache=True,
    parallel=False, n_workers=None, verbose=True):
```
- **Takes:**
  - `ft` — an already-expanded `FieldTheory`; only `ft.taylor_order` is read here (used in
    the cache key).
  - `model` — needs `model['name']` for the slug.
  - `k`, `max_ell` — external-leg count and max loop order.
  - `external_fields` — length-`k` list of `(str, int)` leaf-field tuples; encoded in the
    cache key.
  - `G_ft` — the symbolic propagator matrix. Used by *typing only*, and only its
    zero/nonzero pattern matters (not its parameter values) — so its presence in the key
    is unnecessary (only the model name + taylor order key the file).
  - `resp_idx`, `phys_idx` — field→matrix-index maps from `build_field_index_map`.
  - `vtypes`, `stypes` — vertex/source type lists from `extract_vertex_types` /
    `extract_source_types`.
  - `use_cache` — if `False`, always recompute and never write (the spatial bridge calls
    with `use_cache=False`).
  - `parallel`, `n_workers` — fork-pool fan-out for the *type-assignment* stage only.
    Skipped on cache hits. **See the fork-safety caveat below.**
- **Returns:** a 3-tuple
  - `unique_by_ell: dict[int, list[TypedDiagram]]` — keyed by `ℓ`.
  - `multiplicity_by_ell: dict[int, list[int]]` — parallel to `unique_by_ell`;
    `multiplicity_by_ell[ℓ][i]` is the equivalence-class size of `unique_by_ell[ℓ][i]`.
  - `all_unique: list[TypedDiagram]` — flat concatenation in `ℓ` order.
- **Step-by-step:**
  1. `cache_dir = _model_cache_dir(model, ft.taylor_order, cache_dir_root)`;
     `cache = PipelineCache(cache_dir)` (`_diagrams.py:119–120`).
  2. `ext_tag = _ext_fields_tag(external_fields)` (`:122`).
  3. Build the **stage name** (`:150`):
     ```python
     stage_name = f'unique_typed_mult_v3_{ext_tag}_taylor{ft.taylor_order}'
     ```
     The `_v3_` is a **cache-version tag** bumped twice (see the long comment block
     `:123–149`); the `_mult_` records that the file carries multiplicities; the
     `_taylor<N>` suffix records the taylor-order dependence (so sibling files for
     different orders coexist).
  4. Loop `for ell in range(max_ell + 1)`:
     - **Cache lookup** (`:158`): if `use_cache and cache.exists(stage_name, k=k,
       loop_order=ell)`, try `cached = cache.load(...)`, extract `cached['unique']` and
       `cached['multiplicities']`, populate the three return structures, and `continue`.
       Any exception during load prints a `cache load failed … rebuilding` message and
       falls through to recompute.
     - **Build the four stages** (`:176–186`):
       ```python
       _, _, prediagrams, _ = enumerate_prediagrams_all(k=k, ell=ell, verbose=False)
       typed = enumerate_all_typed(prediagrams, external_fields, vtypes, stypes,
                                   G_ft=G_ft, resp_index=resp_idx, phys_index=phys_idx,
                                   parallel=parallel, n_workers=n_workers)
       causal, n_disc, _ = filter_causal(typed)
       unique, multiplicities = deduplicate_with_multiplicities(causal)
       ```
     - **Cache write** (`:194–205`): if `use_cache`, `cache.save(stage_name, {'unique':
       unique, 'multiplicities': multiplicities}, k=k, loop_order=ell)`, wrapped in
       `try/except` so a save failure only prints a warning.
     - Populate the three return structures and continue.
  5. Return `(unique_by_ell, multiplicity_by_ell, all_unique)`.

The wrapped four-stage entry points (imported at `_diagrams.py:28–34`):
- `enumerate_prediagrams_all` = `msrjd.enumeration.loop_diagram_enumeration.enumerate_all`,
  signature `enumerate_all(k, ell, n_threads=1, max_vertices_search=50, verbose=True)`,
  returns `(trees, topos, prediagrams, counts)` — only the prediagrams are kept.
- `enumerate_all_typed` = `msrjd.diagrams.type_assignment.enumerate_all`.
- `filter_causal` = `msrjd.diagrams.causality.filter_causal`, returns `(kept,
  discarded_count, _)`.
- `deduplicate_with_multiplicities` = `msrjd.diagrams.symmetry.deduplicate_with_multiplicities`.

---

### `pipeline/_expand_cache.py` — the expand cache

`__all__` (`_expand_cache.py:88`): `cache_dir`, `expand_cache_path`, `list_cached_orders`,
`find_best_cached_order`, `prepare_for_load`, `save_expand`, `load_expand`,
`downgrade_by_tp_dict`, `vertex_form_factor_signature`.

#### `_slug(model)` — `_expand_cache.py:104`
- **Returns:** `re.sub(r'[^A-Za-z0-9]+', '_', model['name']).strip('_').lower()` — the
  filesystem-safe theory slug. Identical to the slug `_diagrams.py` and `_propagator.py`
  use, so all caches share one directory.

#### `cache_dir(model, cache_dir_root='saved_theories')` — `_expand_cache.py:109`
- **Returns:** `os.path.join(cache_dir_root, _slug(model))`.

#### `expand_cache_path(model, taylor_order, cache_dir_root='saved_theories')` — `_expand_cache.py:114`
- **Returns:** `…/<slug>/expand_taylor{int(taylor_order)}.sobj` — the on-disk path for one
  order's bundle.

#### `list_cached_orders(model, cache_dir_root='saved_theories')` — `_expand_cache.py:121`
- **Returns:** a sorted `list[int]` of all taylor orders that have a cached bundle. Lists
  the directory and `re.fullmatch(r'expand_taylor(\d+)\.sobj', fname)` on each entry.
  Returns `[]` if the directory does not exist.

#### `find_best_cached_order(model, target_order, cache_dir_root='saved_theories')` — `_expand_cache.py:136`
- **Returns:** the **smallest cached order `≥ target_order`**, or `None`. The "smallest
  that suffices" choice minimises post-load downgrade-filter work — every bigrade kept must
  be coerced back into the freshly-built ring, so loading the *minimal* sufficient file is
  cheapest. Built from `[o for o in cached if o >= target_order]` then `min(…)`.

#### `_poly_to_dict_form(poly)` — `_expand_cache.py:152`
- **Returns:** `dict(poly.dict())` — the pickle-stable `{exp_tuple: SR_coeff}`
  representation of a polynomial-ring element.

#### `_dict_form_to_poly(R, exp_to_coeff)` — `_expand_cache.py:162`
- **Returns:** `R(exp_to_coeff)` (or `R.zero()` if empty) — rebuilds a polynomial in ring
  `R` from the dict form. `R` is the *freshly-built* ring, so this is where the
  parent-identity problem is sidestepped.

#### `_by_tp_to_dict_form(by_tp)` / `_by_tp_from_dict_form(R, by_tp_dict)` — `_expand_cache.py:168, 173`
- Map the whole `{(a,b): poly}` dict to/from `{(a,b): poly.dict()}`. The save path uses the
  first; the load path uses the second with the rehydrated ring `R`.

#### `downgrade_by_tp_dict(by_tp_dict, target_order)` — `_expand_cache.py:181`
- **Takes:** the dict-form `by_tp` and a target order.
- **Returns:** `{key: poly_dict for key, poly_dict in … if sum(key) <= target_order}` —
  keep only bigrades of total degree `≤ target_order`. Operates on dict form so it works
  both before and after ring rehydration. This implements the *downgrade* half of the
  superset theorem.

#### `_canon_chain(chain)` — `_expand_cache.py:196`
- **Takes:** an operator-chain tuple like `(('Lap',),)` or `(('Dx', 0),)`.
- **Returns:** a JSON-able stringified list, e.g. `[['Lap']]` / `[['Dx', '0']]`, or `None`
  for `None`. Every entry is `str()`-ed so it round-trips identically through pickle and
  across Sage sessions.

#### `vertex_form_factor_signature(ns)` — `_expand_cache.py:208`
- **Takes:** a `FieldTheory` namespace object `ns` (or `None`).
- **Returns:**
  - `None` if `ns is None` or `ns._operator_ir` is falsey — so every plain/temporal theory
    keeps a `None`-vs-`None` match and loads unchanged.
  - else `'operator_ir:' + sha256(blob)[:16]`, where `blob` is a sorted JSON of each
    operator-IR vertex term's `(mode, n_phys, chain, weight)` — the four pieces that decide
    which momentum form factor each interaction node carries.
- **Step-by-step:** pull `ns._operator_ir_vertex_terms`; for each term build a 4-list
  `[str(mode), int(n_phys) or None, _canon_chain(chain), str(weight)]`; sort the list of
  4-lists order-independently (`json.dumps(e, sort_keys=True)` key); hash the sorted JSON.
- **Why it matters:** the on-disk slug is only `model['name']` + taylor order; it does
  **not** capture the per-vertex form-factor / mode table, which lives on the namespace as
  an *action-eval side effect*, not in `_by_tp`. Without this signature a Model-B⊕KPZ
  1-loop value would silently load as the bare φ̃φ² number. `load_expand` rejects a bundle
  whose signature differs.
- **Stability note (from the docstring):** the `weight` is serialised via `str()` of the
  already-`simplify_full`'d SR expression; a purely cosmetic Sage-version re-ordering
  would at worst force one extra fresh `expand()` (then a re-save), never a *wrong* load.

#### `_reconstruct_operator_ir_table(ft)` — `_expand_cache.py:253`
- **Takes:** a `FieldTheory` being loaded from cache.
- **Does:** for an operator-IR theory whose namespace lacks `_operator_ir_vertex_terms`,
  re-run `ft.model['action'](ns)` (cheap — builds the symbolic action and lowers the IR; no
  Taylor expansion) to repopulate the derivative-vertex form-factor table. `hasattr`-guarded
  (no-op if already present) and skipped entirely for non-operator-IR theories. Exceptions
  are swallowed (`pass`) — the table is left absent and the signature check / a downstream
  fresh expand catches the inconsistency.
- **Why:** the table is a side effect of `_lower_operator_ir_action` inside the action
  lambda, run during `FieldTheory.expand()`. A cache load deliberately *skips* `expand()`,
  so without this re-run every derivative vertex would collapse to form factor `1`.

#### `_get_ring_var_names(ft)` — `_expand_cache.py:288`
- **Returns:** the polynomial-ring generator names. Canonical source is
  `ft._ns._ring_var_names`; falls back to `[str(g) for g in ft._R.gens()]`. Used as the
  structural-integrity check on load.

#### `prepare_for_load(ft)` — `_expand_cache.py:300`
- **Does:** if `ft._ns`/`ft._R` are unset, run `ft._build_namespace()` and assign
  `ft._ns`, `ft._R`, `ft._n_tilde` — *without* running the Taylor expansion (the whole
  point is to skip `expand()`). Then call `_reconstruct_operator_ir_table(ft)`.
- **Why:** the namespace + ring must exist before `load_expand` can coerce the cached dict
  form back into a live ring. Callers (`precompute`, `compute_cumulants`) call this before
  `load_expand`.

#### `save_expand(model, ft, cache_dir_root='saved_theories', verbose=False)` — `_expand_cache.py:322`
- **Takes:** the model dict and a `FieldTheory` after `expand()`.
- **Returns:** the file path written.
- **Step-by-step:**
  1. `target = ft.taylor_order`; `path = expand_cache_path(model, target, …)`;
     `os.makedirs(dirname, exist_ok=True)`.
  2. Build the **bundle dict** (`:333–344`):
     ```python
     {
       'by_tp':            _by_tp_to_dict_form(ft._by_tp),
       'S_raw_dict':       _poly_to_dict_form(ft._S_raw) if ft._S_raw else {},
       'mf_sector_raw':    _by_tp_to_dict_form(getattr(ft, '_mf_sector_raw', {}) or {}),
       'ring_var_names':   _get_ring_var_names(ft),
       'taylor_order':     int(target),
       'n_tilde':          int(ft._n_tilde),
       'vertex_signature': vertex_form_factor_signature(ft._ns),
       'cache_version':    2,
     }
     ```
  3. `sage_save(bundle, path.removesuffix('.sobj'))`; return `path`.

#### `load_expand(model, ft, target_order, cached_order=None, cache_dir_root='saved_theories', verbose=False)` — `_expand_cache.py:351`
- **Takes:** model, a `FieldTheory` whose `_ns`/`_R`/`_n_tilde` are already populated
  (via `prepare_for_load`), the requested `target_order`, and optionally which on-disk
  `cached_order` file to read.
- **Returns:** `True` on success (then `ft` is equivalent to a fresh `expand()` at
  `target_order`); `False` if no usable cache / load failed / a sanity check rejected the
  bundle (`ft` is left unchanged on `False`).
- **Step-by-step:**
  1. If `cached_order is None`, re-discover via `find_best_cached_order`. If still `None`,
     return `False`.
  2. `path = expand_cache_path(model, cached_order, …)`; if not a file, return `False`.
  3. `bundle = sage_load(path)` inside `try/except` → `False` on failure.
  4. **Ring-name check** (`:388–394`): `_get_ring_var_names(ft)` must equal
     `bundle['ring_var_names']` (exact order included) — else a model edit invalidated the
     cache; return `False`.
  5. **`n_tilde` check** (`:396–399`): `bundle['n_tilde']` must match `ft._n_tilde`.
  6. **Form-factor signature check** (`:411–418`): `_reconstruct_operator_ir_table(ft)`,
     then `vertex_form_factor_signature(ft._ns)` must equal `bundle['vertex_signature']`.
     Catches both a changed model that re-used the slug *and* any pre-signature bundle for a
     derivative-vertex theory (which signs `None` on disk but non-`None` live).
  7. **Downgrade filter** (`:421–423`): if `cached_order > target_order`,
     `by_tp_dict = downgrade_by_tp_dict(bundle['by_tp'], target_order)`.
  8. **Rehydrate** (`:425–435`): `ft._by_tp = _by_tp_from_dict_form(ft._R, by_tp_dict)`;
     rebuild `ft._S_raw` as the *sum of the (filtered) by_tp polynomials* (so it matches the
     requested order, not the loaded surplus); `ft._mf_sector_raw =
     _by_tp_from_dict_form(ft._R, bundle.get('mf_sector_raw', {}))`.

     > **Note (open question):** the bundle stores `'S_raw_dict'` but `load_expand` *does
     > not read it* — it reconstructs `_S_raw` by summing `by_tp`. The stored
     > `S_raw_dict` appears to be dead-on-load (write-only). This is correct (the sum is
     > the right value) but the stored field is then redundant.

  9. **Restore `_cumulant_kernels` side effect** (`:464–472`): re-execute
     `_build_cumulant_action(ft._ns, model)` from `msrjd.core.field_theory` for its side
     effect only (the action term is already baked into the cached `by_tp`; what is needed
     is `ns._cumulant_kernels`, which carries closure callables that don't round-trip
     through pickle). A failure here returns `False` (forcing a fresh expand) because a
     missing `_cumulant_kernels` would silently collapse every colored-noise diagram to 0.
  10. Print a hit message (exact vs filtered) and return `True`.

---

### `pipeline/_precompute.py` — the one-time structural pass

#### `precompute(model, *, force=False, verbose=True)` — `_precompute.py:55`
- **Takes:** a model dict from `TheoryBuilder.build()`; `force` to ignore caches and
  rebuild; `verbose`.
- **Returns:** a status dict (see Data structures) with keys `mf_check`, `sanity_ok`,
  `mf_values`, `taylor_order` (always `2`), `cache_dir`, `propagator_built`,
  `wall_seconds`, `log`.
- **Step-by-step:**
  - Set up `log` + nested `_log(msg)` helper that appends to `log` and prints if verbose.
  - Initialise `out` with default/placeholder values.
  - **Lazy imports** (`:91–100`) of `FieldTheory`, `build_propagator`, and `_expand_cache`
    inside the function (keeps module import cheap; on `ImportError` returns early with
    `mf_check='IMPORT_FAILED: …'`).
  - `out['cache_dir'] = _ec.cache_dir(model)`.
  - **Stage 1 — expand at order 2** (`:106–139`): build
    `ft = FieldTheory(model, taylor_order=2)`. If `not force`, try the expand cache:
    `cached_order = _ec.find_best_cached_order(model, 2)`; if found,
    `_ec.prepare_for_load(ft)` then `_ec.load_expand(model, ft, target_order=2,
    cached_order=cached_order)`. On a miss, `ft.expand()` fresh (catching exceptions →
    `mf_check='EXPAND_RAISED: …'`) and `_ec.save_expand(model, ft)` (a save failure is
    non-fatal).
  - **Stage 2 — sanity check** (`:142–152`): `ft.sanity_check()`; sets
    `mf_check='PASS'`/`'FAIL'`. Returns early if it fails.
  - **Stage 3 — solve MF** (`:155–164`): `_build_fundamental_defaults(model)` then
    `_solve_mf_at_saddle(model, fundamental, ft)`. Non-fatal on failure (the propagator can
    still be built).
  - **Stage 4 — propagator** (`:167–176`): `build_propagator(ft, model, use_cache=True,
    force=force)` → writes `…/propagator.sobj`; sets `propagator_built`.
  - Record `wall_seconds`; return `out`.
- **Why order 2 is privileged** (`_precompute.py:8–22`, `_expand_cache.py:26–30`): order 2
  already contains the `(0,0)/(1,0)/(0,1)` MF sectors *and* the `(1,1)` propagator kernel —
  everything the structural validation + propagator construction need, regardless of how
  high the downstream cumulant calculation reaches. So this is a cheap (~seconds) one-time
  pass that makes the propagator + MF check free for all later runs.

#### `_build_fundamental_defaults(model)` — `_precompute.py:186`
- **Returns:** a `fundamental` dict `{param_name: default_value}` built from each model
  parameter's `default=` declaration, **skipping** parameters marked `mean_field` (those are
  saddles, solved not configured) and any with `default` absent or `None`.

#### `_solve_mf_at_saddle(model, fundamental, ft)` — `_precompute.py:200`
- **Returns:** the saddle-values dict. If `model['equations']` is present, lazy-imports and
  runs `solve_mean_field_dae(model, fundamental)` and returns `result['mf_values']`.
  Otherwise runs the legacy `solve_mean_field(ft, model, fundamental)` and returns
  `result['num_saddles']` (or `{}`).

---

## Data structures

### The expand bundle (on disk in `expand_taylor<N>.sobj`)
Produced by `save_expand`, consumed by `load_expand`. A plain Python dict:

| key | type | meaning |
|---|---|---|
| `'by_tp'` | `{(n_tilde, n_phys): {exp_tuple: SR_coeff}}` | the bigrade-classified action, in pickle-stable dict form |
| `'S_raw_dict'` | `{exp_tuple: SR_coeff}` | dict form of the rebuilt full polynomial (written but **not read on load**) |
| `'mf_sector_raw'` | `{(a,b): {exp_tuple: SR_coeff}}` | the pre-zero `(0,0)/(1,0)/(0,1)` sectors (diagnostic) |
| `'ring_var_names'` | `list[str]` | generator names, e.g. `['vt1','vt2','nt1','nt2','dv1','dv2','dn1','dn2']`; structural sanity check |
| `'taylor_order'` | `int` | the order this was computed at (may exceed the request) |
| `'n_tilde'` | `int` | number of response variables; MF-sector filter check |
| `'vertex_signature'` | `str` or `None` | operator-IR form-factor fingerprint (`'operator_ir:<sha256[:16]>'` or `None`) |
| `'cache_version'` | `int` (=2) | bundle-format version |

### `ft._by_tp` (in memory)
`{(n_tilde, n_phys): poly}` where `poly` is a live Sage polynomial-ring element in `ft._R`.
After a cache load it is `_by_tp_from_dict_form(ft._R, by_tp_dict)`.

### The enumeration-cache payload (on disk in `unique_typed_mult_v3_<ext>_k<k>_l<l>_taylor<N>.sobj`)
A dict `{'unique': list[TypedDiagram], 'multiplicities': list[int]}`.

### `TypedDiagram` (`msrjd/diagrams/type_assignment.py:29`)
A `__slots__` class (with explicit `__getstate__`/`__setstate__` for pickle) holding:
- `prediagram` — `(D, G, leaves, internal)`; `D` is a Sage `DiGraph`.
- `vertex_assignments` — `{vertex_id: VertexType | SourceType}`.
- `edge_types` — `{(u, v, label): (resp_leg, phys_leg)}`, each leg `(field_base, pop_idx)`.
- `external_legs` — `{leaf_vertex: (field_base, pop_idx)}`.
- `propagator_indices` — `{(u, v, label): (resp_row, phys_col)}` into `G_ft`.

### `enumerate_unique_diagrams` return triple
- `unique_by_ell: dict[int, list[TypedDiagram]]`
- `multiplicity_by_ell: dict[int, list[int]]` (parallel)
- `all_unique: list[TypedDiagram]` (flat, ℓ-ordered)

### `precompute` status dict
Keys: `mf_check` (`'PASS'`/`'FAIL'`/`'NOT_RUN'`/error string), `sanity_ok` (bool),
`mf_values` (dict `{<var>star: …}`), `taylor_order` (always `2`), `cache_dir` (str),
`propagator_built` (bool), `wall_seconds` (float), `log` (list[str]).

### `PipelineCache` manifest (`manifest.json`)
`{'entries': [{key, stage, k, loop_order, saved_at}], 'created': iso, 'updated': iso}`.

### On-disk directory layout (`saved_theories/<slug>/`)
Observed live (`saved_theories/1d_kpz_per_leg_gradient_vertex/`):
```
expand_taylor2.sobj        propagator.sobj
expand_taylor4.sobj        manifest.json
unique_typed_mult_v3_<ext>_k<k>_l<l>_taylor<N>.sobj   (one per (k, ℓ, ext, order))
```
`<slug>` = `re.sub(r'[^A-Za-z0-9]+', '_', name).strip('_').lower()`.

---

## Data flow

A full `compute_cumulants(model, k=2, max_ell=1, …)` run, end to end through the caches:

1. **Taylor order chosen** (`compute.py:278`): `taylor_order = max(2 + 2·1, 2) = 4`.
2. **Expand cache** (`compute.py:312–340`):
   - `ft = FieldTheory(model, taylor_order=4)`.
   - `cached_order = find_best_cached_order(model, 4)` → e.g. `4` (or `6` if only a higher
     order is cached, then downgrade-filtered; or `None` → miss).
   - On hit: `prepare_for_load(ft)` then `load_expand(model, ft, target_order=4,
     cached_order=…)` → `ft._by_tp` populated.
   - On miss: `ft.expand()` then `save_expand(model, ft)` → writes
     `saved_theories/<slug>/expand_taylor4.sobj`.
3. Downstream of expand: `extract_vertex_types(ft)` → `vtypes`,
   `extract_source_types(ft)` → `stypes`, `build_propagator(ft, model)` → `prop['G_ft']`,
   `build_field_index_map(ring_var_names, n_tilde)` → `resp_idx, phys_idx`.
4. **Enumeration cache** (`compute.py:649`): `enumerate_unique_diagrams(ft, model, k=2,
   max_ell=1, external_fields=[('v',1),('v',2)], G_ft=prop['G_ft'], resp_idx, phys_idx,
   vtypes, stypes, use_cache=True)`.
   - `stage_name = 'unique_typed_mult_v3_v1_v2_taylor4'`.
   - For `ℓ=0`: file `unique_typed_mult_v3_v1_v2_taylor4_k2_l0.sobj` — hit → load
     `{'unique', 'multiplicities'}`; miss → build four stages, save.
   - For `ℓ=1`: file `…_k2_l1.sobj` — same.
   - Returns `unique_by_ell={0:[…], 1:[…]}`, `multiplicity_by_ell`, `all_unique`.
5. **Consumers** (`compute.py:680+`): walk `unique_by_ell` per `ℓ`,
   `classify_coefficient_factors(td, …)` each diagram, group by kernel, integrate.

The **precompute** flow is the same but truncated: `precompute(model)` exercises only the
expand cache at order 2 plus the propagator cache, writing `expand_taylor2.sobj` and
`propagator.sobj` so that a later `compute_cumulants` at order 2 hits both immediately
(and at order 4 still pays the order-4 `expand()` once).

The **spatial bridge** (`pipeline_bridge.py:239`) calls `enumerate_unique_diagrams` with
`use_cache=False` — it re-enumerates every time (the diagram topology is cheap relative to
the spatial integral and the bridge does not want to manage cache keys).

---

## Gotchas & caveats

- **Fork-based `multiprocessing` is dangerous on macOS notebooks.** `enumerate_unique_diagrams`
  exposes `parallel=True` which fans the type-assignment stage across a *fork-based* pool
  (`enumerate_all_typed(..., start_method='fork')`). Per the project memory, forking a
  Jupyter kernel after matplotlib/Cocoa/BLAS init *crashes the kernel and the OS*; this
  exact path was guarded in commit `a141fdd` for the temporal pipeline and the spatial path
  was made serial-only. The default here is `parallel=False`. **Do not flip `parallel=True`
  from a notebook on macOS** — the cache wrapper itself does not re-apply the
  `_fork_unsafe_in_notebook()` guard; it trusts the caller. (Cache hits skip the parallel
  path entirely.)
- **Cache-version tags are load-bearing.** The enumeration stage name embeds `_v3_`. It was
  bumped `v1→v2` (2026-05-26, `extract_source_types` `NoiseSourceType` promotion) and
  `v2→v3` (2026-06-10, `diagram_signature` became a *complete* isomorphism invariant). A
  `v2` cache loaded today would *resurrect a known bug*: it deduplicated with the old
  incomplete signature, which collided non-isomorphic diagram classes at `k≥3` and silently
  dropped their integrals (e.g. `κ₃` came out `-68/3·a³` instead of `-32·a³`). The version
  bump is the *only* invalidation — old files are simply never read because their filename
  stem differs. **If you change the dedup/typing semantics, you must bump the version
  string**, or stale caches will load and silently produce wrong numbers.
- **The expand cache's on-disk slug under-determines the bundle.** The filename is only
  `model['name']` + taylor order. It does **not** capture: (a) the operator-IR per-vertex
  form-factor table (a namespace side effect of evaluating the action lambda, *not* in
  `_by_tp`), nor (b) the `_cumulant_kernels` closure metadata. Both are reconstructed on
  load (`_reconstruct_operator_ir_table`, `_build_cumulant_action`) and the form-factor one
  is *validated* via `vertex_signature`. If either reconstruction silently no-ops, a
  derivative-vertex or colored-noise theory loads a *wrong* (collapsed) result. The
  signature check protects (a); a failed (b) rebuild forces a fresh expand by returning
  `False`. **This is the most fragile corner of the subsystem** — three docstring blocks
  (`_expand_cache.py:60–73`, `:253–283`, `:437–472`) are devoted to it.
- **A model edit that does NOT change the name will silently re-use the cache.** The slug is
  derived from `model['name']` only. Editing the field list or a vertex without bumping the
  name leaves a stale `expand_taylor<N>.sobj` and stale enumeration files. The *partial*
  defences are the `ring_var_names` and `n_tilde` structural checks in `load_expand` (which
  catch field-list changes that alter the ring) and the form-factor signature (which catches
  operator-IR vertex changes) — but a change that alters, say, a coupling *value* baked into
  a coefficient while keeping the same ring would not be caught. The docstring advice
  (`_diagrams.py:18–22`): bump `model['name']` or delete the cache dir on model edits.
- **`S_raw_dict` is written but never read on load.** `save_expand` stores it; `load_expand`
  reconstructs `_S_raw` by summing `by_tp` instead. Harmless (the sum is correct) but the
  field is dead-on-load — flagged in open questions.
- **`_model_cache_dir(model, taylor_order, …)` ignores its `taylor_order` argument.** The
  directory is per-theory; the parameter is vestigial from the old
  `saved_theories/<theory>_taylor<N>/` layout. Harmless but confusing — flagged.
- **`load_expand` requires `prepare_for_load` first.** The caller *must* have populated
  `ft._ns`, `ft._R`, `ft._n_tilde` before calling `load_expand` — otherwise the dict-form
  rehydration has no ring to coerce into. `precompute` and `compute_cumulants` both honor
  this; a new caller that forgets will get cryptic failures.
- **`downgrade` is exact only because of the superset theorem.** It is *not* a re-expansion
  — it merely drops high-degree bigrades. This relies on Taylor truncation never modifying
  lower-order entries (`_expand_cache.py:14–18`). If a future change made the expansion
  *non-truncating* (e.g. resummation), downgrade would silently be wrong.
- **Manifest is best-effort, not authoritative.** A corrupt `manifest.json` is silently
  reset to empty (`cache.py:164–166`); the `.sobj` files remain the source of truth.
  `enumerate_unique_diagrams` checks `cache.exists` (filesystem), not the manifest, so a
  missing manifest entry never causes a false miss.
- **`enumerate_unique_diagrams` is robust to load failures** but the expand cache returns
  `False` (clean miss → fresh expand) on *any* check failure, while the enumeration cache
  *rebuilds the single `ℓ` slot* on a load exception (`_diagrams.py:170–173`). Both degrade
  gracefully, never raise out of a cache miss.
- **`use_cache=False` means never-write-too.** Both for the enumeration cache
  (`_diagrams.py:194`) and the expand path — a `use_cache=False`/`force=True` run does not
  pollute the cache, but also gets no persistence. The spatial bridge always passes
  `use_cache=False`.

---

## Glossary

- **MSR-JD action** — Martin–Siggia–Rose–Janssen–De Dominicis: the path-integral action
  `S[φ̃, φ]` representing a stochastic dynamical system; `φ` physical field, `φ̃` response
  ("tilde") field.
- **`taylor_order` (N)** — the truncation order of the multivariate Taylor expansion of the
  action in the fluctuation fields; equals `max(k + 2·max_ell, 2)` by default.
- **bigrade `(n_tilde, n_phys)`** — the `(# response-factors, # physical-factors)` grading
  of an action monomial; total degree `= n_tilde + n_phys`. The keys of `_by_tp`.
- **`_by_tp`** — the FieldTheory attribute mapping each bigrade to the sum of its action
  monomials (a polynomial-ring element). The central thing the expand cache stores.
- **MF / saddle sector** — the bigrades `(0,0)/(1,0)/(0,1)`; the mean-field equations,
  required to vanish at the saddle (`sanity_check`).
- **propagator kernel** — the `(1,1)` bilinear sector; the free propagator `G` is its
  matrix inverse.
- **interaction vertex** — an action monomial of total degree `≥ 3`; a Feynman-rule vertex
  with `n_tilde` response legs and `n_phys` physical legs.
- **downgrade (filter)** — keeping only bigrades of total degree `≤ target_order` from a
  higher-order cached expansion; exact by the superset theorem.
- **superset theorem** — `by_tp@N ⊇ by_tp@M` for `M ≤ N`, byte-identical on the shared
  degrees; the justification for the expand cache.
- **prediagram** — an untyped diagram topology (tree → topology → prediagram), before field
  assignment; `(D, G, leaves, internal)`.
- **typed diagram** — a prediagram with vertex types, edge propagator types, and external
  legs assigned (`TypedDiagram`).
- **causal filtering** — dropping diagrams that violate the retarded MSR-JD propagator
  structure (`filter_causal`).
- **diagram_signature** — the canonical form of a typed diagram's coloured incidence
  digraph; a *complete* isomorphism invariant. Two diagrams are duplicates iff their
  signatures match.
- **dedup multiplicity** — the size of a diagram's isomorphism-equivalence class; **diagnostic
  only** under Path A (do not multiply it back into the weight).
- **symmetry factor `𝒮(Γ)`** — the Feynman-rule combinatorial weight
  `∏ n_leg! / |Aut_fixed_ext(Γ)|`; carried by `combinatorial_factor`, not the multiplicity.
- **operator IR** — the intermediate representation for derivative interaction vertices
  (Lap/Dt/Dx binding nodes), used by spatial theories (Model B, KPZ, Burgers).
- **form-factor signature** — `vertex_form_factor_signature`: a sha256 fingerprint of the
  operator-IR vertex table; guards the expand cache against the slug under-determining the
  per-vertex momentum form factors.
- **`_cumulant_kernels`** — namespace dict of noise-cumulant kernel callables; a runtime
  side effect of `_build_cumulant_action`, *not* pickled, rebuilt on cache load.
- **SageMath / `sage.all`** — the host mathematics system; provides `SR` (symbolic ring),
  polynomial rings, `DiGraph`, and `save`/`load`.
- **`.sobj`** — Sage Object: a gzip-compressed pickle written by `sage.all.save` and read
  by `sage.all.load`; the on-disk format for every cached artefact here.
- **`poly.dict()`** — Sage polynomial-ring method returning `{exp_tuple: coeff}`; the
  pickle-stable "dict form" the expand cache stores instead of polynomial objects.
- **nauty** — the C graph-isomorphism/canonical-form library vendored inside Sage; powers
  `DiGraph.canonical_label`, hence `diagram_signature`.
- **PipelineCache** — the low-level `(stage, k, ℓ)`-keyed `.sobj` store with a JSON manifest;
  `msrjd/core/cache.py`.
- **manifest.json** — the human-readable index of cached `PipelineCache` entries; best-effort,
  not authoritative.
- **slug** — `re.sub(r'[^A-Za-z0-9]+','_', name).strip('_').lower()`; the per-theory cache
  directory name, shared by all three caches.
- **precompute** — the cheap one-time structural pass: expand@2 + sanity + MF solve +
  propagator, populating the durable caches.

---

## Proposed manual subsections

1. **Why cache at all** — the 90-minute expand, the combinatorial diagram blow-up, and the
   structure-vs-numbers separation.
2. **The on-disk layout** — `saved_theories/<slug>/`, the slug rule, what each filename
   means, and `.sobj` / Sage serialization explained from scratch.
3. **`PipelineCache`: the low-level store** — stage keys, `save`/`load`/`exists`/`clear`,
   the manifest, int coercion.
4. **The expand cache I: the bigrade dictionary and the superset theorem** — `_by_tp`,
   bigrades, the math of Taylor truncation, why downgrade is exact.
5. **The expand cache II: dict-form serialization** — why `poly.dict()` and not the
   polynomial object; ring-identity across pickle.
6. **The expand cache III: hidden state and its reconstruction** — the operator-IR
   form-factor table, the form-factor signature, `_cumulant_kernels`, and the
   structural-integrity checks in `load_expand`.
7. **`precompute`: the one-time pass** — order-2 privilege, the four stages, the status dict.
8. **The enumeration cache** — the four-stage pipeline it wraps, the cache key
   `(model, taylor, k, ℓ, ext)`, the version tags and what each bump fixed.
9. **`taylor_order = max(k + 2·max_ell, 2)`** — the bridge between the two caches.
10. **Data flow walk-through** — one full `compute_cumulants` run, both caches in sequence.
11. **Gotchas** — fork safety, version bumps, slug under-determination, model-edit
    invalidation, `use_cache=False`.
