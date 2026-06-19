# Dendritic theory — performance investigation (read-only)

Theory: `theories/dendritic_quad_soma_sigmoid.theory.py`
Slug: `dendritic_quadratic_soma_sigmoid_probability_dendrite`
Scope: static code reading only. No `.py` edited; no pipeline run.

---

## 1. Executive summary

**What dominates.** For this theory the wall time is split between two
symbolic stages, and the split *flips* as you go from 1 → 2 neurons:

- **The Taylor expand** (`FieldTheory.expand()`,
  `msrjd/core/field_theory.py:894-898`) — one `taylor()` call over
  *every* field fluctuation variable simultaneously. The 1-neuron model
  has **8** field vars; the 2-neuron model has **16** (4 physical + 4
  response, per neuron, ×2). The action contains two nested
  transcendentals — `exp(nSt)·vS²` and
  `log(1 + (exp(nDt)-1)·σ(vD))` (theory line 55-56) — so `taylor()` must
  series-expand a genuinely non-polynomial expression in many variables
  at once. This is super-linear in the variable count; the owner's
  "90+ minute" memory is this call at high order on a comparable model.

- **The propagator** (`pipeline/_propagator.py`). For 1 neuron the
  reported 11.9 s is **not** the symbolic matrix inverse — at `nf=8` the
  inverse is already *skipped* (the `rich = nf >= 6` gate at
  `_propagator.py:686`). The 11.9 s is the **K_ker → K_ft Fourier
  transform + `_to_kernel` build** (`_propagator.py:585-648`), which runs
  per non-zero entry through Sage's symbolic `fourier_transform`. At 2
  neurons `nf=16`: the symbolic inverse stays skipped, but the
  **numerical pole/residue stage** (`compute_poles_and_residues`,
  phase `[4]`) must now exact-invert a **16×16** matrix in
  `CyclotomicField(4)[ω]` (`_compute_residues_via_polynomial_fracfield`,
  `_propagator.py:81-313`) or fall back to repeated **16×16 numpy
  cofactor adjugates** (`_propagator.py:1470-1502`, an O(nf · nf³) =
  O(nf⁴) per pole construction). Both blow up with field count.

**Single highest-leverage fix (NO code change):** run
`pipeline._precompute.precompute(model)` **once to completion** for this
theory (and let it finish — every prior run was killed by the timeout
*before* the cache write at `_propagator.py:822` and
`compute.py:332`). That single pass writes
`saved_theories/<slug>/expand_taylor2.sobj` **and** `propagator.sobj`.
After that, every notebook `compute_cumulants(... taylor_order=2)` run
loads both from disk in seconds (`compute.py:318-327`,
`_propagator.py:551-580`) and pays **zero** expand + propagator cost.
The cache for `dendritic_1_neuron_test` already exists and proves the
mechanism works (2.2 KB `expand_taylor2.sobj` + 51 KB `propagator.sobj`);
the 2-neuron slug directory does **not** exist yet — confirming no run
ever completed.

The crucial subtlety: **you must let the *first* full run survive the
overnight job's CPU contention.** The cache is written only on a
completed `expand()`/`build_propagator()`; a SIGALRM/timeout kill leaves
nothing on disk and the next run starts from scratch.

---

## 2. Prioritized suggestions

| # | Idea | Why it helps | Rough speedup | Risk | Effort | Tag |
|---|------|--------------|---------------|------|--------|-----|
| 1 | **Run `precompute(model)` once, to completion, then rely on the disk cache** for all notebook runs. | Writes `expand_taylor2.sobj` + `propagator.sobj`; subsequent runs skip both expand and propagator entirely. The only reason no cache exists is that every run was killed before the write. | 2nd run: **24.5 s → ~1-3 s** (1-neuron); 2-neuron: minutes → seconds once the *one* slow build finishes. | None | None | NO-CODE-CHANGE |
| 2 | **Keep `taylor_order=2`** for the tree/`k=2,ℓ=0` slice (it is already the default, `compute.py:278-279`). Do **not** pass a higher order unless you actually need ≥cubic vertices. | Expand cost grows steeply with order; order 2 already has the (1,1) propagator + MF sectors — everything tree needs. The old floor of 4 cost "~90 min" (see `compute.py:271-277`). | Avoids a 10×-100× expand blowup vs order 4-6. | None | None | NO-CODE-CHANGE |
| 3 | **Precompute on a quiet machine / when the overnight job is idle**, or `nice`/pin it to spare cores, so the *one* expensive build completes and persists. | The bottleneck is a single uninterruptible symbolic build; it only needs to succeed once. | Converts an infinite "always times out" loop into a one-time cost. | None | None | NO-CODE-CHANGE |
| 4 | **Reduce `mf_dae_n_starts`** from the default 64 (`compute.py:131`) for this theory once a good fixed point is known (e.g. 8-16). | Phase `[3]` MF solve runs the DAE multistart `n_starts` times (`_mean_field_dae.py:481`). For a 16-state system each solve is non-trivial. | Phase [3] roughly linear in n_starts: 64→16 ≈ **4×** on that phase. | Low (may miss a root if the system is multistable; keep enough starts to find your target). | None (call-site arg) | NO-CODE-CHANGE |
| 5 | **Pre-seed `mf_dae_seed_box`** (`compute.py:132`) near the known saddle so the solver converges in fewer iterations and fewer starts. | Tightens the multistart search. | Compounds with #4. | Low | None (call-site arg) | NO-CODE-CHANGE |
| 6 | **Memoize the Fourier transform of K_ker entries** in `build_propagator` (cache `fourier_transform(c, t, ω)` by the string of `c`). Many `K_ker[i,j]` entries are structurally identical across the 2-neuron blocks (same `tauS·Dt+1`, same δ′ pattern). | `_propagator.py:631-639` FTs every non-zero entry independently; the 2-neuron K has a repeated per-neuron block structure, so identical entries are transformed twice. | Cuts the phase-[2] FT work for block-structured K; ~2× on the 11.9 s when the two neurons are symmetric. | Low (pure memo of a deterministic transform). | Small (local dict in `build_propagator`). | SMALL-CHANGE |
| 7 | **Lower the polynomial-fracfield budget and let it fall straight to numpy** for `nf≥12`, OR raise it only if needed. Currently `_POLYNOMIAL_PATH_BUDGET_SEC = 120` (`_propagator.py:43`). | At `nf=16` the exact `CyclotomicField(4)[ω]` 16×16 inverse (`_propagator.py:147-148`) can burn the full 120 s before failing, *then* the numpy path runs anyway. Failing fast saves that wasted budget. | Saves up to ~120 s of dead time per pole-stage run on the 2-neuron model. | Low (numpy cofactor path is already the validated fallback and is exact-enough; precision note at `_propagator.py:1503-1518`). | Small (one constant, or a `nf`-gated value). | SMALL-CHANGE |
| 8 | **Exploit the block / sparse structure of K_ft.** The 2-neuron coupling is weak off-diagonal (`wSD`, `wDS` only); K_ft is two 8×8 per-neuron diagonal blocks + sparse cross terms. A block-LU / Schur-complement pole solve is far cheaper than a dense 16×16 exact inverse. | The exact and cofactor inverses are dense-O(nf⁴); a 2-block structure makes it ~2·O(8⁴)+coupling ≪ O(16⁴) ≈ 8× fewer ops. | ~5-8× on the pole/residue stage at 2 neurons; grows with neuron count. | Medium (must verify the block decomposition matches the field ordering and that cross-blocks are handled exactly). | Big (new code path in `compute_poles_and_residues`). | BIG-CHANGE |
| 9 | **Parallelize the expand by bigrade / per-population term.** The action is `sum(... for i in E)` (theory line 51-57); each neuron's self-terms are independent and only the `wSD/wDS` cross-terms couple. One could Taylor-expand per-neuron sub-actions and combine. | Splits the one giant `taylor()` into independent smaller ones. | Potentially large, but bounded by the cross-coupling terms that still need joint expansion. | **High** — and **must be thread/term-split, NOT fork**: per repo MEMORY, fork-based multiprocessing from a Jupyter kernel on this macOS box has crashed the OS twice. Any parallel expand must avoid `fork`. | Big | BIG-CHANGE |

---

## 3a. Propagator (`pipeline/_propagator.py`)

**What actually runs for this theory.** `nf` = 2·(#physical fields) =
8 (1 neuron) / 16 (2 neurons). The complexity gate
`rich = nf >= 6 or len(free_syms) > 20` (`_propagator.py:686`) is
**True in both cases**, so the symbolic `K_ft.inverse()`/`adjugate()`/
`det()` block (`_propagator.py:706-733`) is **skipped entirely**. That
means:

- The 1-neuron phase-`[2]` cost (11.9 s) is **not** the inverse. It is
  the K_mat assembly + `_to_kernel` (`_propagator.py:595-622`) and the
  per-entry symbolic **`fourier_transform`** (`_propagator.py:631-639`),
  plus the `kernel_ft_image` substitution (`_propagator.py:642-645`).
  These are the only heavy symbolic steps left in phase [2]. → suggestion #6.

- The real per-field-count blowup lives in **phase `[4]`**,
  `compute_poles_and_residues` (`_propagator.py:851`), via:
  - **Tier 1**, the exact `CyclotomicField(4)[ω]` inverse of the full
    `nf×nf` matrix (`_compute_residues_via_polynomial_fracfield`,
    `_propagator.py:140-148`). Exact polynomial-fraction inversion of a
    16×16 matrix is the expensive object, guarded only by a 120 s
    SIGALRM (`_propagator.py:43`, `:147-148`). → suggestion #7.
  - **Tier 2 fallback**, per-pole **numpy cofactor adjugate**
    (`_cofactor_adj`, `_propagator.py:1470-1479` and `:1551-1558`):
    `nf²` minors, each an `(nf-1)×(nf-1)` determinant → **O(nf⁴) per
    pole**, times the number of poles. At nf=16 this is ~16× the nf=8
    cost per pole *and* there are more poles. → suggestions #7, #8.

**Numerical-vs-exact note.** The pipeline already does pole-finding
**numerically** (Sage/PARI roots on the determinant polynomial,
`_propagator.py:1019` / Newton refine `_propagator.py:1297-1372`) and
residues **numerically** (cofactor + central-difference `det'`,
`_propagator.py:1481-1502`). So "use numerical pole-finding" is already
the implemented fallback — the lever is making it the *primary* path for
large `nf` (skip the exact CF[ω] inverse that times out) and giving it
**block structure** (suggestion #8), not switching exact→numerical.

**Caching.** `propagator.sobj` is written at `_propagator.py:822` and is
**taylor-order-independent** (depends only on the (1,1) bilinear sector,
docstring `_propagator.py:535-541`). It auto-invalidates if `nf` changes
(`_propagator.py:560`). So one completed build serves every order/run.
**But note:** the cached `prop` stores `G_ft=None`/`pole_vals=None` for
this `rich` theory — the cache saves the FT'd `K_ft` (the expensive
part) but the **pole/residue stage [4] still reruns every time** because
`pole_vals`/`C_mats` are filled in `compute_poles_and_residues` *after*
the cache load and are **not persisted** (`compute.py:632`,
`_propagator.py:776-779`). → see suggestion in §3c.

## 3b. Taylor expand (`msrjd/core/field_theory.py`)

**The cost is one call:**
```
S_sr = taylor(S_sr, *[(v, 0) for v in ns._all_field_sr_vars], self.taylor_order)
```
(`field_theory.py:894-898`). `ns._all_field_sr_vars` has **16** entries
for the 2-neuron model. Multivariate `taylor()` to total degree N over
m variables is the dominant cost and scales badly in **m** (field count)
— doubling neurons doubles m. The integrand is genuinely transcendental:
`exp(nSt)`, and `log(1 + (exp(nDt)-1)·(1/(1+exp(-vD))))`
(theory `:55-56`) — the `log` of a `1 + (exp-1)·sigmoid` raises the
sigmoid to all orders, exactly the "nonlinearity makes it longer" the
owner observed.

Secondary expand costs, all in `expand()`:
- `reduce_conv_in_action` with a nested Taylor (`field_theory.py:865-868`)
  — here a near-no-op (no `Conv` atoms in this theory).
- `S_sr.expand()` then `_sr_to_ring` (`field_theory.py:921`,
  `_sr_to_ring` at `:241-281`) — walks every summand and calls
  `.degree()`/`.coefficient()` per ring variable (32 vars). For a large
  post-Taylor polynomial this is non-trivial but secondary to `taylor()`.
- `_verify_and_zero_mf_sector` (`field_theory.py:585`) calls
  `simplify_full()` per MF-sector monomial (`:621`) and, on symbolic
  failure, builds a **second** `FieldTheory`-proxy and runs the **MF
  solver** inside `_mf_numerical_residual` (`field_theory.py:688-701`).
  For a closed-form saddle this is cheap; if `simplify_full` can't close
  it, the numerical fallback adds an MF solve to *expand* time. Worth
  knowing if expand is slower than the pure `taylor()` would predict.

**Why order 2 is the win:** the `_by_tp` at order N is a strict superset
of order M≤N (`_expand_cache.py:14-30`), and order 2 already carries the
(0,0)/(1,0)/(0,1) MF sectors + the (1,1) propagator kernel — everything
tree + propagator + MF-check consume. Higher orders only add interaction
vertices for loops. Keep `taylor_order=2` unless computing loops
(suggestion #2). The default already does this (`compute.py:278-279`).

**Memoization/parallel by bigrade:** the code does **not** memoize the
Taylor across runs except via the disk cache, and does **not** split the
expand by bigrade or population. The action's `sum(... for i in E)`
structure (theory `:51-57`) is *almost* separable per neuron (only
`wSD`/`wDS` couple), so a per-population expand + recombine is feasible
(suggestion #9) — but **must not use fork** (repo MEMORY: fork from a
Jupyter kernel crashed this macOS box twice; spatial path is serial-only
for that reason).

## 3c. Caching / precompute (`pipeline/_expand_cache.py`, `_precompute.py`)

**What persists (confirmed from `dendritic_1_neuron_test/`):**
```
expand_taylor2.sobj   2.2 KB   ← _by_tp + S_raw + mf_sector (cache_version 2)
expand_taylor3.sobj   2.9 KB
propagator.sobj      51   KB   ← K_ker, K_ft, nf, ring names (G_ft=None for rich)
unique_typed_mult_*.sobj       ← per-(k,ℓ) diagram dedup
manifest.json
```
The 2-neuron slug dir does **not** exist → no run ever completed.

**What makes the 2nd run fast:**
- Expand: `load_expand` (`_expand_cache.py:351`) reads `_by_tp` from the
  best cached order ≥ target and downgrade-filters
  (`_expand_cache.py:421-435`). Order-2 cache serves every `taylor_order≥2`
  request via downgrade. Invoked at `compute.py:318-327`.
- Propagator: `build_propagator` returns the cached `prop`
  (`_propagator.py:551-580`) — skips the K_ker build + FT.

**The gap (worth a SMALL change later):** for a `rich` theory the cached
`propagator.sobj` has `pole_vals=None`/`C_mats=None`
(`_propagator.py:776-779`), so the **phase-[4] pole/residue solve reruns
on every call** even with a warm cache — and for nf=16 that's the
expensive 16×16 exact/cofactor work (§3a). `compute_poles_and_residues`
depends on `num_params` (the saddle), so it can't be cached blindly, but
it **could be memoized per `(slug, num_params-hash)`** so repeated runs
at the *same* parameter point reuse poles/residues. Today nothing
persists them.

**Pre-build/persist recipe (NO code change):** call
```python
from pipeline._precompute import precompute
precompute(model, verbose=True)   # writes expand_taylor2.sobj + propagator.sobj
```
once, to completion, on an idle machine. `precompute` always expands at
order 2 (`_precompute.py:107-108`, docstring `:39`), runs the MF check +
saddle solve, and builds+caches the propagator (`_precompute.py:166-176`).
It is documented as "~seconds for any theory" — true *only if the build
is allowed to finish*. For this heavy theory the first build is the
expensive event; everything after is a cache hit. `precompute` is **not**
exposed in `pipeline/__init__` and has no CLI — import it directly as
above (e.g. from a one-off `sage -python` cell *after* the overnight job
frees the CPU).

Note: `precompute` only ever writes the **order-2** expand cache. If you
later need loop vertices (`taylor_order ≥ 3`), the first `compute_cumulants`
at that order pays the higher Taylor once and then sibling-caches
`expand_taylor<N>.sobj` (`compute.py:330-336`), so it too becomes a
one-time cost per order.

---

## 4. File:line index of what was read

- `theories/dendritic_quad_soma_sigmoid.theory.py:51-63` — the action
  text (nested exp/log/sigmoid) + MF equations; 4 fields × 2-pop.
- `msrjd/core/field_theory.py:814-964` — `expand()`; the one-shot
  `taylor()` at **:894-898** is the dominant cost.
- `msrjd/core/field_theory.py:241-281` — `_sr_to_ring` (per-summand
  degree/coefficient walk over ring vars).
- `msrjd/core/field_theory.py:585-728` — `_verify_and_zero_mf_sector` +
  `_mf_numerical_residual` (a second MF solve can land in expand time at
  **:688-701**).
- `msrjd/core/field_theory.py:1041-1364` — `_build_namespace` (field-var
  / ring construction, derivative-rename combinatorics
  `_iter_multi_indices` at **:202-222**, grows with `n_args`×order).
- `pipeline/_propagator.py:43` — `_POLYNOMIAL_PATH_BUDGET_SEC = 120`.
- `pipeline/_propagator.py:81-313` — exact `CyclotomicField(4)[ω]`
  inverse (the nf×nf exact-inverse cost; SIGALRM-guarded at **:147-148**).
- `pipeline/_propagator.py:524-838` — `build_propagator`; FT block
  **:631-645**, `rich` gate **:686**, symbolic-inverse skip **:754-763**,
  cache save **:822**, `G_ft=None` for rich **:776-779**.
- `pipeline/_propagator.py:851-1605` — `compute_poles_and_residues`;
  numerical pole finder **:1019/:1297-1372**, cofactor adjugate
  O(nf⁴) **:1470-1479 / :1551-1558**, residues **:1481-1502 / :1560-1576**.
- `pipeline/_expand_cache.py:14-30, 322-482` — superset/downgrade logic,
  `save_expand`/`load_expand`.
- `pipeline/_precompute.py:55-180` — `precompute()` (always order 2,
  builds+caches propagator).
- `pipeline/compute.py:278-279` — `taylor_order` auto floor of 2;
  `:305-357` — cache-aware expand + propagator; `:628-633` — phase [4]
  pole solve (not persisted).
- `saved_theories/dendritic_1_neuron_test/` — proves the cache mechanism
  (expand_taylor2.sobj 2.2 KB + propagator.sobj 51 KB); 2-neuron slug
  `dendritic_quadratic_soma_sigmoid_probability_dendrite/` absent.
