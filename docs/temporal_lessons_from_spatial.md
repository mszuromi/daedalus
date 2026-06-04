# Lessons from the spatial extension → the temporal pipeline

**Branch `spatial-extension`, June 2026.** The spatial loop-integral work
(analytic heat-kernel IFT, Symanzik reduction, causal-chamber quadrature, MC
and Bessel-K backends, memory guard, Wick-moment factorization) produced
several techniques and hard-won safety lessons. This note audits **which of
them transfer back to the temporal (Phase J) pipeline**, separating:

- ✅ **realized** — already true in temporal (sometimes because the idea
  *originated* in temporal and only *looks* like a spatial import);
- 🔧 **done this session** — a genuine reverse-transfer landed now;
- 🟡 **opportunity** — a real, scoped improvement worth doing later;
- ⛔ **does not transfer** — spatial-specific, no temporal analog.

## TL;DR

| # | Lesson | Transfers? | Status |
|---|--------|-----------|--------|
| 1 | **Don't fork in a macOS notebook** | yes — safety | 🔧 **done** (commit a141fdd) |
| 2 | Push the *last* 1-D integral to closed form (no `scipy.quad`) | yes | ✅ already realized (m=1 mode-sum, Stage 4a-perdiag) |
| 3 | Switchable integrator backends via **env var** | yes — ergonomic | 🟡 opportunity |
| 4 | A **sampling backend** when exact enumeration is too big | yes — but the wall differs | 🟡 opportunity (see efficiency research) |
| 5 | **Coefficient/moment factorization** (eval τ-invariant parts once) | yes | ✅ already realized (grouped merged-residue tensor + plan cache) |
| 6 | **Resource guard** before a blow-up instead of crashing | partially | 🟡 opportunity (complexity-budget, not bytes) |
| 7 | Symanzik / heat-kernel-IFT / Bessel-K radial reductions | no | ⛔ spatial-specific |

## ⚠️ Directionality caveat (read first)

Several techniques that feel like "spatial innovations to port back" actually
**originated in the temporal pipeline and were *reused* by spatial** — the
arrow already points the *other* way:

- **Causal-chamber decomposition** = the temporal **causal poset / linear-
  extension** machinery (`_extract_causal_poset`,
  `_enumerate_linear_extensions`, `_exp_over_chain_simplex` in
  `final_integral.py`). Spatial's `causal_chambers.py` explicitly *reuses* the
  temporal poset + smooth quadrature (task C2-full).
- **Closed-form mode-sum over an integration simplex** — temporal has had the
  analytic polygon (m=2) and poset chain-simplex (m≥3) integrators since the
  Stage 3 Phase-J refactor; spatial's analytic σ/time handling descends from
  the same idea.

So the genuine *reverse*-transfer set is **smaller than it first looks**. The
items below are the ones that are really new information for temporal.

---

## 1. 🔧 Don't fork in a macOS notebook — DONE this session

**The single most valuable transfer, and it is a safety issue, not a speedup.**

`msrjd/integration/time_domain/pipeline.py` returned batch evaluators
(`total_C_batch`, `eval_per_diagram_batch`) that defaulted to
`parallel=True, start_method='fork'` **and** set
`OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES`. This is *exactly* the
fork-after-Cocoa/BLAS-init pattern that hard-crashed the user's machine twice
(June 2026) and was removed from the spatial integrator (now thread-based /
serial-gated, commit fd3af81). The `OBJC_DISABLE_…` flag does not make fork
*safe* — it suppresses the runtime's own warning and makes the crash *more*
likely.

The bit-identity regression tests
(`test_phase_J_total_C_batch_parallel_matches_serial`, `…_nested_…`) pass
only because **pytest runs in a plain interpreter, never a ZMQ kernel**, where
Cocoa was never initialised. They certify *numerical determinism*, not
*notebook crash-safety* — a real gap.

**Fix (commit a141fdd):** `_fork_unsafe_in_notebook(start_method)` — a narrow
guard that trips **only** on `darwin` + a live Jupyter/ZMQ kernel + `fork`. In
that one context the batch helpers `warn` once and degrade to the
always-correct serial path. Linux, terminal IPython, plain `sage -python`
scripts, and the test suite are unaffected and keep the fast fork path. Pinned
by `test_phase_J_fork_in_notebook_guard_degrades_to_serial` (patches
`sys.platform` + `IPython.get_ipython`, deterministic on any host OS).

**Latent follow-up:** the spatial path replaced fork with a *smart-gated
ThreadPoolExecutor*. Temporal could do the same to recover notebook
parallelism — but threads help only where the inner work releases the GIL.
The temporal inner work (`scipy.quad` callbacks, the pure-Python analytic
mode-sums) is largely GIL-bound, so a thread pool would give little speedup.
The serial fallback is therefore the right default for now; if notebook
parallelism is wanted, the real lever is **picklable closures + `spawn`**
(already sketched in the pipeline.py Windows-support note), not threads.

## 2. ✅ Push the last integral to closed form — already realized

The spatial arc ended by replacing the last numerical integral (the radial
Schwinger integral) with a closed-form **Bessel-K**. The temporal analog —
"don't leave a 1-D `scipy.quad` in the hot path when the integrand is a sum of
exponentials" — **was already independently realized.** The Phase-J
dispatcher (`final_integral.py` ~L3624/3671) routes:

- `m=0` → analytic exponential sum;
- `m=1` → `_integrate_1d_polytope_modesum` (**analytic, plan-cached**,
  Stage 4a-perdiag);
- `m=2` → `_integrate_2d_polygon_modesum` (analytic);
- `m≥3` → `_integrate_nd_polytope_poset_modesum` (analytic chain-simplex).

`scipy.quad`/`nquad` are now **fallbacks only** — m=1 falls back solely for an
unbounded divergent endpoint the closed form returns `None` for. **Memory
correction:** the note "per-diagram m=1 still uses scipy.quad" in
`project_grouped_phase_j_precision.md` is **stale** and is being updated.

There is no headline win left here; the lesson is already lived. (Minor:
`fast_callable` does not pass `cse=True`; the analytic paths bypass it, so the
benefit would be small and is not pursued.)

## 3. 🟡 Switchable backends via env var — ergonomic opportunity

Spatial exposes `SPATIAL_INTEGRATOR=grid|mc|bessel`, `SPATIAL_MC_N`,
`SPATIAL_FORCE_NUMERICAL_FT`, `SPATIAL_MEM_BUDGET_GB` as **env knobs**, so a
notebook can switch strategy without editing code. Temporal's analog switches
(`USE_1D_INTEGRATOR`, `USE_POLYGON_M2_INTEGRATOR`, `USE_POSET_INTEGRATOR`,
`USE_POSET_MPMATH_ACCUMULATION`, `USE_GROUPED_ANALYTIC_MODESUM`) are
**module-level booleans** — flipping them means editing `final_integral.py` or
poking module globals from a cell. **Recommendation:** read these from
`os.environ` with the current boolean as default (one-line each), mirroring the
spatial convention, so backend selection and precision/accumulation mode are
notebook-controllable. Low risk, purely additive.

## 4. 🟡 A sampling backend for the enumeration blow-up — real, but a *different* wall

Spatial's ℓ≥2 wall is a **dense-grid memory wall** (`(P, n_x)` array,
`P=n_t^{n_V}·n_s^{n_C}` → tens of GB → OOM), cured by **Monte-Carlo /
tropical importance sampling** over the Schwinger parameters.

The temporal pipeline has **no dense product grid** — it is polytope-sparse,
so it will not OOM the same way. Its blow-up at high `(k, ℓ)` is instead
**combinatorial compute**:

- **pole-tuple enumeration** `n_poles^{|E|}` per subset (the merged-residue
  tensor `B_α = Σ_td cp_td · Π_e C_{α_e,e}` is the analytic mode-sum's inner
  sum), and
- **linear-extension enumeration** (number of topological sorts of the causal
  poset, worst-case `m!`) in the m≥3 path.

The transferable *idea* is the same — **when exact enumeration is too big,
importance-sample it** — but the target is the pole-tuple / linear-extension
sum, not a quadrature grid. This is a genuine future backend (e.g. sample the
dominant pole-tuples by `|B_α|`, or the dominant linear extensions), and it is
exactly what the in-flight **efficiency deep-research** is scoping (tropical
sampling, QMC, variance reduction). **Deferred to that synthesis** — see
`docs/spatial_loop_integral_analytic_mc.md` and the forthcoming efficiency
report; do not hand-roll before reading it.

## 5. ✅ Coefficient / moment factorization — already realized

Spatial's Wick-moment win was: the moment is `M_F = Σ_k g_k(a,Σ,B)·X^k` with
**x-independent coefficients `g_k` evaluated once per sample, then contracted
against the cheap `X`-powers** (47 s → 0.97 s). The temporal analog —
"factor the τ-invariant part out of the per-τ inner loop" — **already exists**:

- the **grouped merged-residue tensor** (`grouped_integral.py`) builds `B_α`
  **once per prediagram**, not per typed diagram (~1000× fewer pole iterations
  at k=2 ℓ=1); and
- the **Stage 4a plan cache** (`_build_modesum_plan`, 2026-05-15) precomputes
  the per-pole-tuple `(α_s, γ_const, γ_slope)` decomposition **once at subset
  setup** and threads it through every `_contrib(free_vals)` call, so a τ-dense
  sweep pays the edge-loop once instead of `N_τ` times — the direct temporal
  twin of "g_k once, contract per output point."

Parity is good here; no action.

## 6. 🟡 Guard before the blow-up — complexity budget, not bytes

Spatial added a **memory guard**: estimate the worst chamber's `(P,n_x)`
allocation up front and `raise` with the offending numbers instead of letting
the kernel OOM-`Killed:9`. Temporal cannot OOM the same way (no dense grid),
so a byte-budget guard is the wrong analog. The right analog is a
**complexity-budget guard**: before the m≥3 poset path enumerates linear
extensions, or before a subset expands `n_poles^{|E|}` pole-tuples, estimate
the count and, if it exceeds a budget, either warn + fall back to the sampling
backend (Lesson 4) or raise a clear, actionable error (as spatial does) rather
than silently spending minutes. Low priority until a real case hits it, but
the *principle* — "predict the blow-up and fail loud / degrade gracefully,
never thrash" — is the most broadly valuable thing spatial taught.

## 7. ⛔ What does NOT transfer

These are intrinsic to having a **spatial momentum / position** axis and have
no pure-temporal analog:

- **Symanzik U/F graph-polynomial momentum reduction** — there is no spatial
  loop momentum to integrate in the pure-temporal theory.
- **Analytic heat-kernel inverse Fourier transform** `q → x` — no `x`.
- **Bessel-K radial / Dirichlet angular-simplex** reduction — these
  parametrize the spatial overall-scale and direction; temporal's "radial"
  direction is already handled by the elementary exponential closed forms of
  Lesson 2.

---

## Net actions from this audit

1. **Done:** fork-in-notebook safety guard (Lesson 1, commit a141fdd) — the
   one genuine safety transfer.
2. **Memory correction:** m=1 is analytic by default (Lesson 2); the stale
   "still uses scipy.quad" note is being updated.
3. **Cheap, additive, queued:** env-var backend switches (Lesson 3).
4. **Deferred to the efficiency research synthesis:** a sampling backend for
   the temporal pole-tuple / linear-extension enumeration (Lesson 4) and a
   complexity-budget guard (Lesson 6).
