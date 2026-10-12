# Correlated noise: capabilities and shortcomings

This is the reference for what the MSR-JD pipeline's correlated-noise
machinery (the CGF tab in the theory builder, equivalently
`TheoryBuilder.declare_cgf_term` and the lower-level
`TheoryBuilder.correlated_noise`) can and can't express, with rough
estimates for what it'd take to lift the limitations.

If your noise model doesn't fit, the doc lists the workarounds.  If
you want to extend the framework, the doc estimates the cost.

## Scope: what kind of noise this is about

We're talking about additive stochastic forces driving any of your
declared physical fields.  In MSR-JD form, this is the contribution

```
S_noise = − Σ_n (1/n!) ∫ dt₁ … dtₙ
                κ⁽ⁿ⁾_{i₁,…,iₙ}(t₁−t₁′, …, tₙ−t₁′) m̃_{i₁}(t₁) … m̃_{iₙ}(tₙ)
```

added to the action, where `m̃` is a response field, `κ⁽ⁿ⁾` is the
n-th cumulant of the noise process, and time-translation invariance
reduces the cumulant to a function of `n−1` relative times.

The framework decomposes each kernel as

```
K(τ₁, …, τ_{n−1}) = c_local · ∏_k δ(τ_k)  +  K_smooth(τ₁, …, τ_{n−1})
```

The δ-collapsed part becomes a direct `(n, 0)`-bigrade term in the
action (one Sage SR variable per response leg).  The smooth part
becomes a frequency-domain noise vertex that the Phase J integrator
multiplies into the diagram.

## What works today

### 1. Gaussian noise (κ² with any kernel shape)

Either via an inline `(2, 0)`-bigrade term in the Action tab (e.g.,
`-D*xt^2`) or via a CGF-tab row with `order=2`.  Supported kernel
shapes:

| Name | `kernel` field | Notes |
|---|---|---|
| White | `dirac_delta(tau)` | δ-collapse → local action term |
| OU / Lorentzian | `exp(-abs(tau)/tauc) / (2*tauc)` | smooth residual → frequency-domain vertex |
| Double-exponential | `c1*exp(-abs(tau)/t1) + c2*exp(-abs(tau)/t2)` | smooth residual |
| Lagged δ | `dirac_delta(tau - Delta)` | implicitly symmetrized to `½[δ(τ−Δ) + δ(τ+Δ)]` |
| Any analytic Sage expression | as above | as long as Sage can extract the δ part vs. smooth part |

### 1.5 Markovian embedding (auto, v1)

For the most common smooth-kernel case — a single Lorentzian
`c · exp(-|τ|/τc)` driving a Gaussian noise on one or two response
fields — the framework now applies an **automatic Markovian
embedding** at `TheoryBuilder.build()` time.  Instead of leaving the
smooth kernel in place (which forces the time-domain integrator
through the slow `scipy.nquad` fallback at `max_ell ≥ 1`), the
preprocessor rewrites your colored noise as a white-noise-driven
auxiliary OU field:

```
dx/dt  = f(x) + ξ
dξ/dt  = -ξ/τc + sqrt(2c/τc) · η,   ⟨η η⟩(τ) = δ(τ)
```

The auxiliary's stationary autocovariance is
`⟨ξ ξ⟩(τ) = c · exp(-|τ|/τc)`, exactly recovering your colored kernel.
The new theory has the same observables for `x` and an extra hidden
`ξ`.  Your `external_fields=[('x', 1), …]` request still picks `x`
unchanged — no notebook edit needed.

**What's auto-Markovianized in v1.**  A CGF row matches the
preprocessor when ALL of:

1. `order == 2`
2. `response_legs` is either `[r, r]` (auto-cumulant) OR `[r1, r2]`
   (cross-cumulant — the 2D case)
3. `kernel` parses as `c · exp(-|tau|/p)` where `p` is a declared
   positive parameter (or a positive symbolic combination thereof)
4. `coefficient` is a Sage expression that the preprocessor can use
   to derive the embedding's amplitude

For each matched row the preprocessor:

* injects one new `physical_field` per affected source (`x → xi`,
  `y → yi`, etc., with auto-response `xit`, `yit`);
* adds `(Dt + 1/τc) · xi = 0` to the DAE equations;
* replaces the colored CGF row with a white-noise row on the
  auxiliaries: `coefficient = 2c/τc`, `kernel = dirac_delta(tau)`;
* appends `- xi` to the original `xt*(...)` block in your action text
  (encoding `dx/dt = ... + ξ`) and adds the new kinetic term
  `xit*(Dt + 1/τc)*xi`.

The cross-correlated 2D case (3 CGF rows: Cxx, Cyy, Cxy) becomes
3 white-noise rows on the auxiliaries (auxXX, auxYY, auxXY), all with
`dirac_delta(τ)` kernels.

**Opting out.**  Per-builder: `.markovianize(False)` keeps your
colored rows intact and runs the legacy scipy fallback (with the
existing warning).  Per-row: `.declare_cgf_term(..., markovianize=False)`
preserves the original kernel on JUST that row.  Use this when your
colored kernel has a shape the v1 preprocessor doesn't recognize and
you've already hand-coded the embedding yourself.

**Opting in with a strict check.**  `.declare_cgf_term(...,
markovianize=True)` is a strict opt-in: if the kernel doesn't match
the v1 template, the build raises a `ValueError` instead of silently
falling through to scipy.  Use this when you've hand-tuned a
Lorentzian-shaped coefficient and want to ensure it round-trips
through the preprocessor.

**v1 limitations.**

| Kernel shape | v1 status | v2 plan |
|---|---|---|
| `c · exp(-|τ|/τc)` (Lorentzian) | **handled** | n/a |
| `c · exp(-|τ|/τc) · cos(Ω τ)` (underdamped oscillatory) | not handled | embed as a 2-state OU with imaginary eigenvalue |
| `c1·exp(-|τ|/t1) + c2·exp(-|τ|/t2)` (double-exp) | not handled | embed as a sum of two independent OU processes |
| polynomially-modulated `τ^k · exp(-|τ|/τc)` | not handled | embed as a chain of k+1 OU processes |
| arbitrary smooth kernel (no rational FT) | not handled | n/a — falls back to scipy.nquad |

**Edge case: degenerate poles.**  The Markovian embedding adds the
auxiliary's eigenvalue `1/τc` to the system's pole spectrum.  If your
deterministic system already has an eigenvalue at `1/τc` (e.g. a
linear field with rate `μ = 1/τc`), the joint propagator has a
multiplicity-2 pole.  The current single-pole residue + mode-sum
infrastructure can't represent the `τ · exp(-pt)` modes a
multiplicity-2 pole requires, so the numerical computation becomes
unstable in this exactly-degenerate case.  Workaround: perturb `τc`
or `μ` slightly so the poles are distinct.

### 2. Cross-field cumulants (κ², any kernel)

Via `response_legs=['xt', 'yt']` on the CGF tab — the new
multi-leg path.  The framework auto-symmetrizes by iterating the
canonical permutation and its reverse, so the antisymmetric part of
an asymmetric kernel is correctly dropped (matching the cumulant's
index-permutation symmetry).

Practical use: 2D OU with cross-correlated Gaussian noise (white or
colored), multi-population Hawkes with cross-cell GTaS noise, etc.

### 3. Higher cumulants with fully-local kernels (κⁿ, n ≥ 3, ∏δ form)

Anything where `K(τ₁, …, τ_{n−1}) = c · ∏_k δ(τ_k)` works at any n.
This covers:

- **Poisson shot noise / compound Poisson.**  Campbell's theorem
  gives `κ⁽ⁿ⁾ ∝ rate · ⟨Jⁿ⟩ · ∏δ` for any n.
- **GTaS Bernoulli-marked Poisson.**  Auto-cumulants (`i = j = k = …`)
  are all `∏δ` (Poisson statistics for the marginal); cross-
  cumulants vanish at n ≥ 3 in the standard GTaS construction.
- **Cluster point processes with same-time clustering** (events
  arrive simultaneously, marks are correlated).

The framework iterates the n-tuple of leg indices via
`itertools.product`, evaluates the kernel, peels off the δ
coefficient via repeated `K.coefficient(dirac_delta(τ_k))`, and
injects `−(1/n!) c_local · m̃_{i₁} … m̃_{iₙ}` directly into the
action as an `(n, 0)`-bigrade source vertex.  The diagram engine
treats this identically to any other multi-leg interaction vertex.

### 4. Mixed-cumulant declarations

Multiple CGF rows with the same `name` but different `order` sum
into one noise process (κ², κ³, κ⁴, … all on the same field set).
Multiple rows with the same `(name, order)` AND `response_legs` sum
into the same κ entry.  Rows with the same `name` but different
`response_legs` stay as distinct contributions.

This lets you build, e.g., a marked-Poisson process with arbitrary
mark moments by listing one row per cumulant order.

---

## What doesn't work

### 1. Smooth κⁿ for n ≥ 3 (integrator-engineering gap)

If your κ⁽ⁿ⁾ kernel at n ≥ 3 has a NON-δ smooth part, the framework
emits this warning at `expand()` time and silently drops the smooth
contribution:

```
correlated_noises[<noise_name>] cumulant order <n> legs <idx>:
kernel has a non-local smooth residual that requires an n-leg
time map in the integrator (currently only n=2 implemented).
Skipping.
```

The δ-local part (if any) still gets injected correctly.

**Examples that hit this limit:**

- Cluster point processes (e.g. Hawkes-of-Hawkes) where event times
  are correlated → κ³ has temporal structure.
- Non-Markovian shot noise where marks are correlated across events
  with finite-time autocorrelation.
- Multi-time-scale compound noise where high cumulants don't fully
  factor into delta products.

**Workaround:** none structural.  You can validate the theory's
action / vertex extraction (the action term gets correctly registered
via `_cumulant_kernels` even when Phase J skips it — useful for
checking the symbolic side), but the actual diagrammatic computation
of cumulants involving these vertices doesn't run.

**Extension cost:** ~2-4 weeks of focused work.  Three phases:

1. *Phase A — bookkeeping (1-2 days).*  Replace the warning with
   registration of the smooth symbol on `ns._cumulant_kernels` (the
   data structure already supports arbitrary leg-tuples; just turn
   the n ≥ 2 branch into the general one).  Audit `vertices.py`
   `NoiseSourceType` extraction to confirm n > 2 sources flow into
   the diagram enumerator the same way 2-leg sources do.
2. *Phase B — integrator (1-2 weeks).*  Extend
   `msrjd/integration/time_domain/grouped_integral.py` +
   `final_integral.py` to thread n − 1 relative-time variables per
   noise vertex through the diagram's time-integration plan.  Derive
   and implement 2D analytic mode-sum templates for the κ³ smooth
   case (paralleling the existing 1D `polygon_m2`, `chain_simplex`).
   Add `scipy.dblquad` fallback for kernels without clean closed
   forms.
3. *Phase C — higher orders (~few days per order).*  Iterate the
   pattern to 3D for κ⁴, etc.

The math is well-defined throughout; the cost is in the integrator
engineering and analytic prefactor derivations, plus preserving the
n = 2 fast-path performance (a lot of effort went into the
`phase-j-refactor` Stage 4a optimizations; mustn't break them).

### 2. Non-stationary noise (fundamental, not just engineering)

The framework assumes time-translation invariance: `κ⁽ⁿ⁾(t₁, …, tₙ)`
collapses to a function of `n − 1` relative times.  Non-stationary
noise like a stimulus-triggered transient, a non-equilibrium
temperature ramp, or a windowed perturbation isn't expressible as a
single cumulant kernel in the κ-series form.

**Workaround:** split your model into

- a deterministic time-varying drive (encoded as a parameter or
  function in the deterministic action), and
- a stationary residual that fits the κ-series form.

The deterministic part needs custom handling — not a CGF-tab thing.

**Extension cost:** large, possibly framework-redesigning.  The
diagram engine's time-integration machinery assumes stationarity
deeply (Phase J's analytic mode-sums, the Fourier-domain
propagator, the convolution-kernel handling).  Adding non-stationary
external sources would require a parallel "non-equilibrium" path
through compute_cumulants.  Estimated 1-3 months for a first pass.

### 3. Multiplicative noise on fields (fundamental, doesn't fit κ-series form)

E.g., `dx/dt = −μx + ξ(t)·x` where the noise multiplies the field.
This DOESN'T have the form `S_noise = −W[m̃]` that the κ-series
machinery expects.  Multiplicative noise corresponds to a cubic
(or higher) coupling in the action (`xt · ξ · x → xt·ξ·(xstar + dx)`)
which is structurally an interaction vertex, not a noise vertex.

**Workaround:** declare `ξ` as a SEPARATE external field with its
own κ⁽²⁾ via the CGF tab (white or colored Gaussian, etc.) and put
the multiplicative coupling `xt · g · ξ · x` directly in the Action
tab.  The diagram engine then sees this as a (1, 2) interaction
vertex (one tilde + one fluctuation + one external) plus a noise
source on the `ξ` field — which it can handle.

This works.  It just doesn't go through the CGF tab as a single
unit; the multiplicative structure lives in the action text.

### 4. Lévy / heavy-tailed noise without finite cumulants

The κ-series form requires all cumulants you reference to exist
(finite).  For α-stable Lévy noise (1 < α < 2), the second moment
diverges, and you can't write `κ⁽²⁾ < ∞` at all.  The framework
machinery doesn't apply.

**Workaround:** if your application can tolerate it, replace the
Lévy noise with a "tempered" or "truncated" version where high-
moment marks are cut off — restoring finite cumulants.  Often the
tail truncation is physically justified (e.g., real biological
event sizes have finite max).

**Extension cost:** would require a completely different formalism
(e.g., functional integrals over Lévy paths instead of cumulant
series).  Not on any roadmap.

### 5. Order-2 asymmetric kernels: antisymmetric part silently dropped

If you declare a κ² with an asymmetric kernel like
`kernel='dirac_delta(tau - Delta)'` (lagged cross-correlation), the
framework iterates both leg-field permutations and effectively
symmetrizes the kernel under `τ → −τ`.  This corresponds to the
cumulant's index symmetry `κ²_{xy}(τ) = κ²_{yx}(−τ)` and is the
physically correct treatment for a Gaussian (κ²-only) cumulant —
the antisymmetric part of `K_{xy}(τ)` doesn't contribute to the
action term anyway.

But if you EXPECTED the antisymmetric part to appear (perhaps from
mis-deriving your model), you won't see it.  Heads-up rather than
a true bug.

---

## Decision tree: which path for which noise

A quick reference for choosing the right tool:

```
Is your noise stationary (time-translation invariant)?
├─ No: not directly supported; see §2 workaround.
└─ Yes:
   │
   Does it have a finite κ²?
   ├─ No (Lévy-like): not supported; see §4 workaround.
   └─ Yes:
      │
      Is it Gaussian (κ³ = κ⁴ = … = 0)?
      ├─ Yes:
      │  │
      │  Single field (auto only)?
      │  ├─ Yes:  inline `-D*xt^2` in Action  OR  CGF row at order 2.
      │  └─ No (cross-field):
      │     │
      │     Is the kernel symmetric in τ?
      │     ├─ Yes: CGF tab with `response_legs=['xt','yt']`.
      │     └─ No:  Same as Yes — framework symmetrizes; cf. §5.
      │
      └─ No (κⁿ ≠ 0 for some n ≥ 3):
         │
         Are ALL κ⁽ⁿ⁾ kernels fully-δ (Campbell-form)?
         ├─ Yes: CGF tab with one row per order, all kernels = `∏δ`.
         │        Works at any n.
         └─ No (some κⁿ has smooth structure):
            │
            Is the smooth structure only at n = 2?
            ├─ Yes: CGF tab with `kernel=` non-δ at order 2;
            │        and δ-only rows at higher orders.  Works.
            └─ No: not supported (§1).  Use the workaround note
                   above and consider commissioning Phase B.
```

---

## Where this lives in the code

| Component | File | Lines (approx) |
|---|---|---|
| `make_correlated_noises_block` (text-row → callable) | `pipeline/theory_compiler.py` | `_build_cgf_kernel_callable` ~1500 |
| `_build_cumulant_action` (action injection) | `msrjd/core/field_theory.py` | ~330–560 |
| `extract_source_types` (diagram extraction) | `msrjd/core/vertices.py` | `NoiseSourceType` ~595 |
| Phase J integrator hooks for κ² smooth | `msrjd/integration/time_domain/grouped_integral.py`, `final_integral.py` | search for `kernel_fn` |
| `declare_cgf_term` (TheoryBuilder API) | `pipeline/theory.py` | ~654 |
| CGF tab (UI) | `pipeline/ui/main.py` | ~404 |
| Spec serialize / parse | `pipeline/theory_serialize.py` | `_emit_cgf_term` ~227, load ~623 |
| **Markovian embedding preprocessor (v1)** | `pipeline/colored_to_markovian.py` | whole file |
| Markovianize wiring into `build()` | `pipeline/theory.py` | search for `markovianize_spec` |

The warning that fires when you hit §1 (smooth n ≥ 3) is in
`_build_cumulant_action`'s smooth-residual `if order == 2` branch's
`else` arm.
