#### Audit cleanups: state released after a discard run, editor saddle names, README

- **Resident state (MEM-08).** A run made with `discard=True` evaluated its diagrams and released them, but the cached
  realizations kept the resolvent data (`lam` tensors of every active-edge pattern) alive for the life of the
  process. It is now released at the end of such a run (rebuilt on first use, bit-identically: a test compares a kept run
  and a discard run). `SymbolicCumulant._ctx` held the pipeline result of its reference run (every diagram's callables)
  as long as the symbolic result lived; it now keeps the propagator and substitutions only.
- **Model editor (GEN-04).** Converting a legacy `set_mf_equation(saddle, rhs)` entry took the field name by cutting a
  `star` suffix off the saddle; it now uses the saddle parameter's declared `natural_name` and falls back on the
  `<field>star` spelling only for an undeclared saddle.
- **README.** The "Changes between versions" paragraph said `DAEDALUS_PHASE_J_DD_KERNELS=0` and
  `DAEDALUS_PHASE_J_LEGACY=1` restore the pre-0.2.0 behaviour; both are retired (set, they warn once and are ignored).
- **Test.** `test_leaf_swap_reproduces_the_v2_total_bit_for_bit` no longer skips when v1 and v2 agree: the swap claim
  holds either way.
- No numbers move.
