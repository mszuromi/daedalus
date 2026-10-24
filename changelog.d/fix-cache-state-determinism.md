#### Cold and warm caches give byte-identical cumulants

Before this fix, a run that read its typed diagrams from the on-disk cache (warm) could differ in the last bits
from the run that wrote them (cold). On a k=3 one-loop sum of 83,068 diagrams the difference was about 3e-14
relative, with or without `discard`. The diagrams were the same; the order in which they were summed was not.

- **Cause.** A Sage graph does not survive a pickle round trip unchanged. The unpickler re-adds the edges in
  iteration order, and the sparse backend lists parallel edges newest first, so every round trip reverses each
  group of parallel edges in `D.edges()`. Above 10 vertices it can also move vertex slots, which changes the
  order of `G.edges()`. The typed-diagram cache stored the prediagram records as plain Sage graphs, so a warm run
  saw every multi-edge loop in reverse order. The Linux fork pool of the type-assignment stage
  (`parallel=True`) returns its typed diagrams pickled, so it had the same problem.
- **Fix.** Prediagram records are now `PrediagramRecord`s (`engine.enumeration.prediagram_cache`). Each one is
  still a `(D, G, leaves, internal)` tuple. It pickles the inputs it was built from (vertex insertion order, arcs
  in label order, leaves, internal; a small `bytes` blob) and is rebuilt by the same code that built it, so the
  restored graphs are identical to the originals, iteration orders included. Records from certificates
  (shipped, v2, computed, streamed) are built this way directly, so their numbers do not change. Eager-enumerator
  records (`use_cache=False` on the temporal path) and legacy v1 `.sobj` cells are rebuilt once on load by
  `canonical_record`. They keep the same labelled graphs and leaves, but their edge iteration order is fixed
  anew, so their totals can move once, by rounding.
- **Cache version.** The typed-diagram cache stage is now `unique_typed_mult_v4_*`. Old `v3` files were written
  with the reversing pickles, so they are no longer read: the first run of each model recomputes its typed
  diagrams once.
- **Memory.** A record costs about 0.4 kB more for its recipe. The prediagram memory estimate is now 11.6 kB per
  class (was 11.2 kB).
- **Shipped cells.** The cells in `engine/enumeration/shipped_prediagrams` are unchanged and load to the same
  records as before.
- **Test.** `tests/test_cache_state_determinism.py` covers this. It runs cold and then warm in a temporary
  directory and requires byte-identical totals and per-loop-order values, and the same for a cold run through the
  fork pool. It also checks that the records survive a pickle round trip exactly at (3,2) and (5,1), and for the
  eager records.
