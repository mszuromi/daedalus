# Subsystem brief: Colored Noise → Markovian Embedding (`colored-markovian`)

Primary source file: `pipeline/colored_to_markovian.py` (521 lines, read in full).
Imports followed: `pipeline/theory.py` (the `TheoryBuilder` API this code mutates),
`pipeline/theory_compiler.py` (the CGF compiler that consumes the mutated spec),
and the worked theory files `theories/ou_quartic_colored.theory.py` and
`theories/ou_quartic_two_dim_color_corr.theory.py`, plus the test
`tests/test_markovianize.py`.

---

## Overview

### What this subsystem does, in one sentence

It is a **builder-level preprocessor** that rewrites a user's *colored-noise*
(finite-correlation-time) cumulant term into an equivalent *white-noise* problem
by introducing one extra dynamical field per noisy field — an auxiliary
Ornstein–Uhlenbeck (OU) process — so that the downstream analytic integrators can
handle it. It trades a hard-to-integrate smooth time kernel for a larger but
purely Markovian (white-driven) field theory.

### Why it exists — the concrete pain it removes

Daedalus' temporal integrator (`msrjd.integration.time_domain.final_integral`)
has fast, *analytic* mode-sum integrators for **white** noise (where the noise
two-point function is a Dirac delta in time, `δ(τ)`). When a user instead
declares a **colored** noise — a smooth kernel like `exp(-|τ|/τc)` with a finite
correlation time `τc` — those analytic integrators cannot absorb the smooth
factor. The pipeline then falls back to a generic numerical multi-dimensional
quadrature (`scipy.integrate.nquad`). The module docstring states the problem
exactly (`pipeline/colored_to_markovian.py:6-10`):

> `msrjd.integration.time_domain.final_integral` cannot absorb a smooth
> `exp(-|tau|/tauc)` factor coming from a CGF-tab `NoiseSourceType` vertex into
> its analytic mode-sum integrators. At `max_ell >= 1` the scipy `nquad`
> fallback path either hangs or runs for orders of magnitude longer than the
> equivalent white-noise theory.

So at loop order `ℓ ≥ 1` a colored theory is effectively unusable. (`max_ell` is
the loop-order cutoff; `ℓ = 0` is tree level, `ℓ ≥ 1` adds loop corrections.)

### The fix, in plain language

There is a classic trick in stochastic dynamics: a colored noise whose
autocovariance is a single decaying exponential is *exactly* the output of a
first-order linear filter (an OU process) driven by white noise. So instead of
feeding colored noise `ξ(t)` directly into the user's equation, you:

1. drive `ξ` itself with white noise through an OU relaxation equation, and
2. feed the **OU output** `ξ` into the user's field as an ordinary coupling.

The user's field now sees a *deterministic linear coupling* to an auxiliary
field, and the only stochastic forcing left in the whole system is white. The
white-noise analytic integrators apply again. The price is one extra field (and
its MSR-JD response partner) per noisy field, i.e. a bigger but cheaper-to-
integrate theory.

### Where it sits in the end-to-end pipeline

```
  user authors a TheoryBuilder        (pipeline/theory.py)
        .physical_field('x')
        .declare_cgf_term(..., kernel='exp(-abs(tau)/tauc)')   ← colored CGF row
        .set_action_text('xt*((Dt+mu)*x + ...)')
        .build()
            │
            │  inside build(), BEFORE text→lambda compilation:
            ▼
  ┌──────────────────────────────────────────────────────────┐
  │  markovianize_spec(builder)        ← THIS SUBSYSTEM        │
  │   • detect_lorentzian() on each CGF row                   │
  │   • inject aux physical field(s) xi (+ auto response xit) │
  │   • inject OU drift equation (Dt + 1/tauc)*xi = 0         │
  │   • replace colored CGF rows with white aux CGF rows      │
  │   • rewrite action_text: add  -xi coupling + xit kinetic  │
  └──────────────────────────────────────────────────────────┘
            │  mutated builder (now a pure white-noise theory)
            ▼
  _compile_text_declarations()         (pipeline/theory.py)
  make_correlated_noises_block(...)    (pipeline/theory_compiler.py)
            │  model['correlated_noises'] with white δ(τ) kernels only
            ▼
  compute_cumulants → diagram engine → time_domain analytic integrators
```

- **Feeds it:** the `TheoryBuilder` accumulator — specifically its `_cgf_terms`
  list (declared via `declare_cgf_term`), its `_action_text`, its
  `physical_fields` / `parameters` lists, and the gating flags
  `_markovianize_default` and per-row `markovianize`.
- **It calls back into the builder:** it does NOT return a value. It *mutates the
  builder in place* by calling the same public authoring methods a user would —
  `builder.physical_field(...)`, `builder.equation(...)` — and by overwriting
  `builder._cgf_terms` and `builder._action_text`.
- **Consumes its output:** `_compile_text_declarations()` (same `build()` call,
  immediately after) and then `make_correlated_noises_block()` in
  `theory_compiler.py`, which turns the now-white CGF rows into
  `model['correlated_noises']` callables that the diagram engine attaches to
  noise vertices.

The single integration point is in `pipeline/theory.py:1735-1737`:

```python
if self._cgf_terms:
    from pipeline.colored_to_markovian import markovianize_spec
    markovianize_spec(self)
```

placed deliberately *before* `self._compile_text_declarations()` (line 1742) so
the downstream compiler only ever sees the augmented, white-noise spec.

---

## The math

This section builds the embedding from the ground up. A reader who knows MSR-JD
field theory but not this particular trick should be able to reconstruct every
text string the code emits.

### 1. What "colored noise" means here

A noisy field `x(t)` is driven by a random force `ξ(t)`. The force is fully
characterized (for a Gaussian noise) by its mean (zero) and its two-time
autocovariance — the second cumulant:

```
    <ξ(t) ξ(t')> = κ(t - t') = κ(τ),     τ = t - t'.
```

- **White noise:** `κ(τ) = 2D · δ(τ)`. The force is uncorrelated across any two
  distinct instants. This is what the fast analytic integrators handle.
- **Colored noise (single Lorentzian):** `κ(τ) = c · exp(-|τ|/τc)`. The force
  "remembers" itself for a correlation time `τc`; as `τc → 0` with `c/τc` held
  fixed it limits back to white. The name *Lorentzian* is because the Fourier
  transform of `exp(-|τ|/τc)` is a Lorentzian `∝ 1/(ω² + 1/τc²)`.

In Daedalus a colored second cumulant is declared as a **CGF row** (cumulant-
generating-functional term) with `order=2`, a `coefficient` text (the amplitude
`c`), and a `kernel` text (the τ-shape). Concretely, from
`theories/ou_quartic_colored.theory.py:18`:

```python
.declare_cgf_term('CXX', response_legs=['xt', 'xt'], order=2,
                  coefficient='2*D/tauc', kernel='exp(-abs(tau)/tauc)')
```

so here `c = 2D/τc` and the kernel is `exp(-|τ|/τc)`, i.e. the full second
cumulant is `(2D/τc) · exp(-|τ|/τc)`.

### 2. The Markovian embedding (the OU filter)

The module docstring states the embedding (`pipeline/colored_to_markovian.py:13-23`):

```
    dx/dt   = f(x) + ξ
    dξ/dt   = -ξ/τc + sqrt(2c/τc) · η(t),     <η η>(τ) = δ(τ)
```

Read this as a two-line stochastic system:

- **Top line** is the *user's* equation, except the colored force `ξ` is no
  longer "random by fiat" — it is now a genuine dynamical field.
- **Bottom line** is the *auxiliary* OU equation: `ξ` relaxes toward zero with
  rate `1/τc`, and is itself kicked by **white** noise `η` of unit strength,
  scaled by amplitude `sqrt(2c/τc)`.

**Claim (the whole reason this works):** the *stationary* autocovariance of the
OU output `ξ` is exactly the desired Lorentzian:

```
    <ξ(t) ξ(t')> = c · exp(-|t - t'|/τc).
```

**Derivation sketch.** The OU SDE `dξ/dt = -ξ/τc + σ η`, with white `η`
(`<η(t)η(t')> = δ(t-t')`) and `σ = sqrt(2c/τc)`, is linear, so its stationary
two-point function is computable in closed form. The stationary variance is
`σ²·τc/2` (balance of injection `σ²` against dissipation `2/τc`), and the decay
in lag is the relaxation rate `1/τc`:

```
    <ξ ξ>(τ) = (σ² τc / 2) · exp(-|τ|/τc)
             = ( (2c/τc) · τc / 2 ) · exp(-|τ|/τc)
             = c · exp(-|τ|/τc).            ✓
```

That fixes the white drive amplitude: in **variance** terms the white CGF
coefficient (the strength `2D_aux` of `<η η> = 2D_aux δ(τ)`) must be

```
    aux white-noise coefficient = 2c/τc.
```

This `2c/τc` is exactly what `detect_lorentzian` computes and stores as
`aux_drive_text` (`pipeline/colored_to_markovian.py:183`):

```python
aux_drive_expr = (SR(2) * c_expr / tauc_expr).simplify_full()
```

For the worked example `c = 2D/τc, τc = τc`, this gives `2·(2D/τc)/τc =
4D/τc²`, which the test pins down (`tests/test_markovianize.py:73`):
`assert info['aux_drive_text'] == '4*D/tauc^2'`.

### 3. From SDE to MSR-JD action (what the action edits encode)

Daedalus works in the MSR-JD (Martin–Siggia–Rose / Janssen–De Dominicis) path-
integral. Each physical field `x` gets a conjugate **response field** `x̃`
(written `xt` in code). A linear equation of motion `(Dt + μ)x = (forcing)`
appears in the action as a term `x̃ · [(Dt + μ)x - (forcing)]`. The code's action
edits are precisely the MSR-JD encoding of the two SDE lines above.

**Edit (1) — couple the user's field to the aux output.** The original user
action is `xt*((Dt+mu)*x + eps*x^3)`. The colored force `ξ` enters the SDE as
`dx/dt = f(x) + ξ`, i.e. `(Dt+mu)x + eps·x³ - ξ = 0`. So the code inserts `- xi`
inside the user's parenthesized block (`_inject_aux_coupling`,
`pipeline/colored_to_markovian.py:472`):

```
    xt*((Dt+mu)*x + eps*x^3)   →   xt*((Dt+mu)*x + eps*x^3 - xi)
```

**Edit (2) — add the auxiliary OU kinetic term.** The OU line
`dξ/dt = -ξ/τc + (white)` corresponds to the deterministic operator
`(Dt + 1/τc)ξ`, contracted with the aux response field `xit`. The code appends
(`pipeline/colored_to_markovian.py:431-435`):

```
    + xit*((Dt + 1/tauc)*xi)
```

The white drive `sqrt(2c/τc) η` does **not** appear in the action text — it is
represented by the **new white CGF row** on the aux response legs (the `2c/τc`
coefficient with kernel `δ(τ)`), which is how Daedalus injects Gaussian white
noise generally. That is `new_cgf_rows` at `pipeline/colored_to_markovian.py:397-405`.

So the three structural pieces — aux field, aux OU drift equation, aux white CGF
row — together reproduce `dξ/dt = -ξ/τc + sqrt(2c/τc)η`, and the `-xi` coupling
plus the `+ xit*(...)` kinetic term wire `ξ` into the user's dynamics.

### 4. The cross-correlated 2D variant

For two fields `x, y` with a cross-correlated colored force,

```
    <ξx ξx>(τ) = c_xx e^{-|τ|/τc},
    <ξy ξy>(τ) = c_yy e^{-|τ|/τc},
    <ξx ξy>(τ) = c_xy e^{-|τ|/τc},
```

the embedding gives `x` its own aux OU `ξx` (code: `xi`) and `y` its own `ξy`
(code: `yi`), each driven by white noise, plus a *cross* white CGF row between
the two aux response legs. From `theories/ou_quartic_two_dim_color_corr.theory.py:25-27`
the three colored rows `Cxx`, `Cyy`, `Cxy` become three white aux rows: two auto
rows `[xit,xit]` and `[yit,yit]`, plus one cross row `[xit,yit]`. The aux white
covariance matrix between `(η_x, η_y)` then reproduces the desired colored cross-
covariance (because all three share the same OU relaxation rate `1/τc`, the
matrix factorizes cleanly). The docstring lays this out at
`pipeline/colored_to_markovian.py:33-39`.

### 5. The `μ = 1/τc` double-pole limitation

The free (Gaussian, `ℓ = 0`) two-point function of the embedded *user* field is
the convolution of the user-field propagator `1/(iω + μ)` with the OU output
spectrum `∝ 1/(ω² + 1/τc²)`. In the time/frequency-domain residue calculus this
product has poles at `ω = iμ` and `ω = i/τc`. The analytic mode-sum integrator
computes residues assuming these poles are **simple (multiplicity 1)**.

When `μ = 1/τc` exactly, the two poles **collide** into a single **double pole
(multiplicity 2)**. The closed-form tree variance the test checks,

```
    C(0) = 2D / ( μ (μ τc + 1) ),
```

(`tests/test_markovianize.py:169,190`) is the *simple-pole* result; at
`μ = 1/τc` it is the limit of a `0/0` and the residue infrastructure that
Daedalus uses cannot form the double-pole residue. The test deliberately picks
`μ = 0.1, τc = 1` to avoid the degeneracy, and the inline comment is explicit
(`tests/test_markovianize.py:165-168`):

> Parameters chosen to AVOID the `μ = 1/τc` exact-degeneracy edge case (which
> produces a multiplicity-2 pole the current single-pole residue infrastructure
> can't handle — see the v2 follow-up).

This is a *downstream* limitation (it lives in the residue calculus, not in this
module), but it constrains how a user may parameterize a markovianized theory:
**keep `μ ≠ 1/τc`.** The embedding code itself does not guard against it.

---

## External tools used

This module's *only* third-party dependency is **SageMath** (the `sage.all`
import). It does not touch nauty, numba, scipy, or networkx directly (those are
used elsewhere in the pipeline). One helper uses the **Python standard-library
`re`** module. Both are explained from scratch below.

### SageMath (`sage`)

**What it is.** SageMath is a large open-source mathematics system built on top
of Python. It bundles many math libraries behind a single Python-friendly API.
The piece this module uses is Sage's **Symbolic Ring**, abbreviated `SR`. Think
of `SR` as a calculator that works with *algebraic expressions containing named
variables* (like `tauc`, `D`, `tau`) instead of just numbers. An object of type
"element of `SR`" is a symbolic expression — e.g. `exp(-abs(tau)/tauc)` — that
you can differentiate, simplify, substitute into, and pretty-print, all
symbolically. This is the same idea as SymPy (Daedalus uses SymPy elsewhere), but
Sage's `SR` is a separate, more powerful engine; the two are NOT interchangeable
objects.

**The import line** (`pipeline/colored_to_markovian.py:70-73`):

```python
from sage.all import (
    SR, sage_eval, exp, abs_symbolic, simplify,
    sqrt as sage_sqrt,
)
```

Each imported name and exactly how this code uses it:

- **`SR`** — the Symbolic Ring object/factory.
  - `SR.var('name')` creates (and globally registers) a *symbolic variable*. The
    code calls `SR.var('_lorentz_tau')` (line 128) to make a fresh `tau` symbol
    and `SR.var(pname)` (line 135) to make one symbol per declared parameter.
  - `SR(x)` *coerces* an arbitrary object `x` into a symbolic expression. Used
    pervasively: `SR(K)` (line 141), `SR(operands[0])` (line 148), `SR(-1)`
    (line 165), `SR(2)` (line 183), `SR(expr)` in `_sr_to_text` (line 520).
    Coercion is how the code guarantees it is always operating on a Sage symbolic
    object and can call symbolic methods on it.

- **`sage_eval`** — Sage's *safe string-to-expression evaluator*. You give it a
  string of Sage syntax plus a dictionary `eval_ns` ("namespace") mapping names
  to Sage objects, and it parses and evaluates the string *in that namespace
  only*. It is like Python's built-in `eval()` but tuned to Sage syntax (e.g.
  `^` means power) and scoped to the dict you pass. Used twice:
  - `K = sage_eval(kernel_text, eval_ns)` (line 138) — turn the kernel string
    `'exp(-abs(tau)/tauc)'` into a symbolic expression.
  - `c_expr = sage_eval(coefficient_text or '0', eval_ns)` (line 177) — turn the
    coefficient string `'2*D/tauc'` into a symbolic expression. The `or '0'`
    guards against an empty coefficient string.

- **`exp`** — Sage's symbolic exponential function. Two roles:
  - it is *put into the evaluation namespace* (`'exp': exp`, line 132) so the
    kernel string's `exp(...)` resolves to the symbolic exponential;
  - it is *compared against* to recognize the kernel's top-level operation:
    `if K.operator() != exp` (line 143). `K.operator()` returns the outermost
    function/operation of the expression tree; for a Lorentzian kernel that must
    be `exp`.

- **`abs_symbolic`** — Sage's symbolic absolute-value function (`|·|`). The plain
  Python `abs` would try to evaluate numerically; `abs_symbolic` keeps `|tau|`
  as an unevaluated symbolic object so the matcher can reason about it. Used:
  - in the eval namespace as `'abs': abs_symbolic` (line 132) so the kernel
    string's `abs(tau)` becomes a symbolic `|tau|`;
  - to build `abs_tau = abs_symbolic(tau_sym)` (line 149), the object the matcher
    divides by and tests for.

- **`simplify`** — imported but **never used** in this file (see open questions).
  The code instead calls the *method* form `.simplify_full()` on expressions
  (lines 155, 165, 178, 183), which is a more aggressive simplifier than the
  free function `simplify`. `.simplify_full()` tries trigonometric, rational,
  factorial, log, and radical simplifications in sequence — the goal is to reduce
  e.g. `arg / |tau|` to a clean `-1/tauc`.

- **`sqrt as sage_sqrt`** — Sage's symbolic square root, imported under an alias.
  Also **not used directly** in this file (see open questions); the `sqrt` that
  appears in coefficient strings like `'rho*sqrt(D1*D2)/tauc'`
  (`ou_quartic_two_dim_color_corr.theory.py:27`) is resolved by `sage_eval`
  against Sage's *default* namespace, not via this import.

**Key symbolic methods this code calls on `SR` expressions** (so the reader knows
what each does):

- `.operator()` (line 143) → the outermost operation node (e.g. the function
  `exp`, or `+`, or `*`).
- `.operands()` (line 145) → the list of arguments of the outermost operation;
  `exp(z)` has exactly one operand, `z`.
- `.simplify_full()` (lines 155, 165, 178, 183) → return a maximally-simplified
  copy.
- `.has(sub)` (lines 158, 159, 170) → boolean: does this expression contain the
  given sub-expression anywhere in its tree? Used to assert the decomposition is
  "abs-free" and "tau-free".
- `str(SR(expr))` (line 520) → render back to a Sage-syntax string (`^` for
  powers etc.) the text pipeline can re-parse.

### Python standard library: `re` (regular expressions)

**What it is.** `re` is Python's built-in regular-expression module: it matches
text patterns. A *regular expression* is a small pattern language; e.g. `\b`
means "word boundary," `\s*` means "zero or more whitespace characters,"
`re.escape(s)` quotes any special characters in `s` so they are matched
literally.

**Where used.** Only inside `_inject_aux_coupling` (imported locally,
`pipeline/colored_to_markovian.py:486`). It builds a pattern to find the token
`<resp_field>*(` in the action text (`pipeline/colored_to_markovian.py:488-489`):

```python
pattern = re.compile(
    rf'\b{re.escape(resp_field)}\s*\*\s*\(', flags=0)
m = pattern.search(action_text)
```

`re.compile(...)` pre-compiles the pattern; `.search(...)` finds its first
occurrence and returns a match object `m` (or `None`). `m.end()` gives the index
just past the match, which the code uses to locate the opening parenthesis and
then walks the string by hand to find the matching close. Regex is used only to
*locate* the insertion point; the actual brace-matching is a manual depth
counter, because regex cannot reliably match balanced parentheses.

---

## Components

Every function in `pipeline/colored_to_markovian.py`, exhaustively.

### `detect_lorentzian` — the kernel matcher

- **Location:** `pipeline/colored_to_markovian.py:79`
- **Signature:**
  `detect_lorentzian(kernel_text: str, coefficient_text: str, declared_params: list[str]) -> Optional[dict]`
- **Takes:**
  - `kernel_text` — the CGF row's `kernel` string, e.g. `'exp(-abs(tau)/tauc)'`.
  - `coefficient_text` — the CGF row's `coefficient` string, e.g. `'2*D/tauc'`.
  - `declared_params` — list of declared parameter names (strings) so the matcher
    can build a namespace where `tauc`, `D`, etc. are known symbols.
- **Returns:** a dict describing the embedding on a match (keys below), or `None`
  if the kernel does not cleanly match the single-Lorentzian template. (Note the
  docstring at lines 96-106 lists four keys but the code returns six — see open
  questions.)
- **Step-by-step:**
  1. **Empty guard** (line 110): if `kernel_text` is falsy, return `None`.
  2. **Build the eval namespace** (lines 128-135): create a *fresh local* symbol
     `tau_sym = SR.var('_lorentz_tau')`, but bind it to the *name* `'tau'` in the
     namespace so the kernel string's `tau` resolves to it. Add `exp` and
     `abs → abs_symbolic`, then add every declared parameter as a generic
     `SR.var(pname)` via `setdefault` (so a parameter the user already declared
     wins, and unknown names are still defined).
     - *Why the fresh `_lorentz_tau`:* a long comment (lines 116-127) warns that
       other parts of the pipeline may register a global `tau` symbol with a
       `domain='positive'` assumption (e.g. an exponential-synapse time
       constant). If `tau > 0` is assumed globally, `abs(tau)` collapses to `tau`
       and the matcher would wrongly accept an *asymmetric* kernel like
       `exp(-tau/tauc)`, which is NOT a physical autocorrelation (autocovariances
       are symmetric in τ). Using a name the caller never declares positive keeps
       the matcher robust against test-ordering pollution.
  3. **Parse the kernel** (lines 137-141): `K = sage_eval(kernel_text, eval_ns)`
     inside a `try/except` that returns `None` on any parse error, then
     `K = SR(K)`.
  4. **Top-level must be exp** (lines 143-148): `if K.operator() != exp: return
     None`. Then `operands = K.operands()`; require exactly one operand; take
     `arg = SR(operands[0])` — the exponent. Build `abs_tau =
     abs_symbolic(tau_sym)`.
  5. **Decompose `arg = -|tau|/τc`** (lines 154-161): compute
     `r = (arg / abs_tau).simplify_full()`. For a true Lorentzian, `r` should be
     `-1/τc` — i.e. **abs-free** and **tau-free**. The code rejects (`return
     None`) if `r.has(abs_tau)` or `r.has(tau_sym)`. This is the test that
     distinguishes `exp(-|tau|/τc)` (good: `r = -1/τc`) from `exp(-tau/τc)`
     (bad: `r = -tau/(τc·|tau|)`, still has tau) and from a Gaussian
     `exp(-tau²/...)` (bad: `r` still has tau).
  6. **Compute and validate τc** (lines 163-170): `tauc_expr = (SR(-1) /
     r).simplify_full()`. Reject if `tauc_expr` still references `tau_sym` or
     `abs_tau`. (Positivity of `τc` is asserted in the docstring as a
     requirement but is *not* enforced numerically here — see open questions.)
  7. **Parse the amplitude** (lines 174-180): `c_expr = sage_eval(coefficient_text
     or '0', eval_ns)`, coerced and `.simplify_full()`-ed; `return None` on
     failure. This is the `c` in `c·exp(-|τ|/τc)`.
  8. **Compute the aux white drive** (line 183):
     `aux_drive_expr = (SR(2) * c_expr / tauc_expr).simplify_full()` — the
     `2c/τc` from the math section.
  9. **Return the embedding dict** (lines 185-192) with both SR expressions and
     their string renderings (`_sr_to_text`): `tauc_expr`, `tauc_text`,
     `amplitude_expr`, `amplitude_text`, `aux_drive_expr`, `aux_drive_text`.

### `markovianize_spec` — the builder mutation entry point

- **Location:** `pipeline/colored_to_markovian.py:198`
- **Signature:** `markovianize_spec(builder) -> None`
- **Takes:** a `TheoryBuilder` (actually a `TemporalTheoryBuilder`) instance that
  has at least `_cgf_terms`. It reads `builder._cgf_terms`,
  `builder._markovianize_default`, `builder.parameters`,
  `builder.physical_fields`, `builder._action_text`.
- **Returns:** `None`. All effects are *in-place mutations* of the builder.
- **Idempotent:** re-running on an already-markovianized builder is a no-op
  because no remaining row matches (the aux rows carry `markovianize=False`, and
  the colored rows are gone). Docstring lines 215-216.
- **Step-by-step:**
  1. **Snapshot & empty guard** (lines 223-225): `cgf_terms = list(...)`; return
     immediately if there are none.
  2. **Builder-level gating** (lines 227-230): read
     `builder_default = _markovianize_default` (default `True`). Compute
     `any_explicit_on = any(t.get('markovianize') is True ...)`. If the default
     is OFF *and* no row explicitly opts in, return (no-op). This is the
     `.markovianize(False)` opt-out path.
  3. **Collect declared parameter names** (line 232) for the matcher.
  4. **Classify each row** (lines 240-276) into `matched_rows` / `passthrough_rows`:
     - `markovianize is False` → passthrough (line 245-247).
     - `order != 2` → passthrough (colored embedding is a second-cumulant story;
       lines 250-252).
     - run `detect_lorentzian(...)` (lines 254-258).
     - on **no match**: if the row had `markovianize is True` (explicit), **raise
       `ValueError`** telling the user their kernel doesn't match the template
       (lines 264-272); otherwise passthrough silently (line 273).
     - on **match**: append `{'term': term, 'info': info}` to `matched_rows`
       (line 276).
  5. **Early out** (lines 278-279): if nothing matched, return.
  6. **Gather the response-leg fields** referenced by matched rows (lines 292-298):
     read each row's `response_legs`; if absent, broadcast its `response_field`
     to a 2-list; collect all leg names into `source_field_names`.
  7. **Invert the response→natural map** (lines 303-317): for each response field
     name `rname` (e.g. `'xt'`), find the physical field whose
     auto-response is `rname` by checking `f'{nat}t' == rname` (recall
     `physical_field('x')` auto-creates response `xt`); store `response_to_natural
     [rname] = nat` (e.g. `'x'`). If none found, store `None` (this row will fall
     back to scipy because it can't be embedded).
  8. **Single-τc enforcement** (lines 323-334): collect the set of distinct
     `tauc_text` strings across matched rows. If more than one distinct τc,
     **raise `NotImplementedError`** — v1 supports a single shared τc only. Else
     take `tauc_text = matched_rows[0]['info']['tauc_text']`.
  9. **Assign aux field names** (lines 339-352): for each source natural name (in
     sorted order, for determinism), call `_pick_aux_field_name(...)` to get a
     collision-safe aux name (`x → xi`), record it in `aux_natural_by_source`,
     and *reserve* it against later collisions. If the result map is empty,
     return.
  10. **Inject aux physical fields** (lines 355-360): for each `(source_nat,
      aux_nat)`, call `builder.physical_field(aux_nat, indexed=True,
      description=...)`. This is the standard authoring method — it
      auto-creates the conjugate response `<aux_nat>t` and the saddle parameter
      `<aux_nat>star` (per `physical_field` in `theory.py:806-911`).
  11. **Inject aux OU drift equations** (lines 363-367): for each aux field,
      `builder.equation(lhs=f'(Dt + 1/{tauc_text}) * {aux_nat}', rhs='0')` — the
      deterministic part of the OU process, `(Dt + 1/τc)ξ = 0`.
  12. **Build replacement white CGF rows** (lines 370-405): for each matched row,
      recover its two response legs, map them to their natural names `nat_a,
      nat_b`, derive the aux response names `aux_resp_a = f'{aux...[nat_a]}t'` and
      `aux_resp_b`, and emit one white row:
      ```python
      {
          'name':           f'{term["name"]}_markov_aux',
          'response_field': None,
          'response_legs':  [aux_resp_a, aux_resp_b],
          'order':          order,
          'coefficient':    info['aux_drive_text'],   # 2c/τc
          'kernel':         'dirac_delta(tau)',        # white
          'markovianize':   False,                     # don't re-process
      }
      ```
      For an auto-cumulant `[xt, xt]` both aux legs are `xit` (one row). For a
      cross-cumulant `[xt, yt]` the legs are `[xit, yit]` (the cross row); the
      auto rows `Cxx`, `Cyy` produce `[xit,xit]` and `[yit,yit]` separately.
  13. **Swap the CGF list** (line 408): `builder._cgf_terms = passthrough_rows +
      new_cgf_rows` — the colored rows are gone, replaced by white aux rows;
      untouched rows are preserved.
  14. **Augment the action text** (lines 424-436):
      - For each source field, call `_inject_aux_coupling(action_text, resp_field,
        aux_nat)` with `resp_field = f'{source_nat}t'` — inserts `- xi` inside the
        user's `xt*(...)` block.
      - Then for each aux field, append
        `+ {aux_resp}*((Dt + 1/{tauc_text})*{aux_nat})` — the OU kinetic term.
      - Write back to `builder._action_text`.
  15. **Record diagnostics** (lines 439-443): set
      `builder._markovianize_applied = {'aux_fields': [...], 'matched_rows':
      [...names...], 'tauc_text': tauc_text}`. (This is informational; tests read
      it indirectly via the resulting field lists.)

### `_pick_aux_field_name` — collision-safe aux naming

- **Location:** `pipeline/colored_to_markovian.py:449`
- **Signature:** `_pick_aux_field_name(source_nat: str, taken: set, additional_taken=()) -> str`
- **Takes:** the source natural name (`'x'`), a set of already-taken natural
  names, and an extra iterable of names taken in the current pass.
- **Returns:** a collision-free aux name string.
- **Step-by-step:** union `taken | additional_taken`. Try candidate
  `f'{source_nat}i'` (so `x → xi`, `y → yi`); if taken, try `f'xi_{source_nat}'`;
  if still taken, loop `f'xi_{source_nat}_{n}'` for `n = 2, 3, …`. Always
  terminates because the integer suffix grows without bound.

### `_inject_aux_coupling` — action-text surgery

- **Location:** `pipeline/colored_to_markovian.py:472`
- **Signature:** `_inject_aux_coupling(action_text: str, resp_field: str, aux_natural: str) -> str`
- **Takes:** the current action text, the response field token (`'xt'`), and the
  aux natural name to insert (`'xi'`).
- **Returns:** the action text with `- xi` inserted inside the parenthesized block
  following `xt*(`, OR the text unchanged if that pattern isn't found.
- **Step-by-step:**
  1. Compile a regex `\b{resp_field}\s*\*\s*\(` (word boundary, the field token,
     optional whitespace, `*`, optional whitespace, `(`). `re.escape` quotes the
     field name (lines 488-489).
  2. `.search(...)` for the first match; if none, return the text unchanged
     (graceful no-op for "baroque" action layouts; lines 491-492).
  3. From the match end, `start = m.end() - 1` is the index of the `(`. Walk
     forward with a `depth` counter, `+1` on `(` and `-1` on `)`, until `depth`
     returns to 0 — that is the *matching* close paren (lines 494-507).
  4. Insert `f' - {aux_natural}'` immediately before that close paren and return.
     If no matching close is found (malformed text), fall through and return
     unchanged (line 508).
  - **Note:** it edits **only the first** `<resp_field>*(` occurrence per call.
    `markovianize_spec` calls it once per source field.

### `_sr_to_text` — symbolic → re-parseable string

- **Location:** `pipeline/colored_to_markovian.py:511`
- **Signature:** `_sr_to_text(expr) -> str`
- **Takes:** any Sage symbolic expression (or coercible object).
- **Returns:** `str(SR(expr))` — the default Sage string form, which uses `^` for
  power and function names like `sqrt`. The docstring (lines 517-519) notes this
  is exactly the syntax `pipeline.theory_compiler._CGFKernelCallable` re-evaluates
  later, so round-tripping is lossless.

---

## Data structures

### The CGF-term dict (input rows, `builder._cgf_terms` entries)

Created by `declare_cgf_term` (`theory.py:546-554`). Fields:

| key | type | meaning |
|---|---|---|
| `name` | str | CGF identifier; rows with same name+order sum into one cumulant |
| `response_field` | str \| None | legacy single response field (broadcast to all legs) |
| `response_legs` | list[str] \| None | per-leg response field names, length = order |
| `order` | int | cumulant order (this subsystem only handles `order == 2`) |
| `coefficient` | str | Sage-syntax amplitude text (the `c`) |
| `kernel` | str \| None | Sage-syntax τ-kernel text; colored = `'exp(-abs(tau)/tauc)'`, white = `'dirac_delta(tau)'` |
| `markovianize` | bool \| None | per-row gate: `True` (must match, else raise), `False` (skip), `None`/`'auto'` (match if possible) |

### The detection-info dict (output of `detect_lorentzian`)

(`pipeline/colored_to_markovian.py:185-192`)

| key | type | meaning |
|---|---|---|
| `tauc_expr` | SR expr | symbolic τc |
| `tauc_text` | str | τc rendered as Sage syntax (e.g. `'tauc'`) |
| `amplitude_expr` | SR expr | symbolic amplitude `c` |
| `amplitude_text` | str | `c` as text |
| `aux_drive_expr` | SR expr | symbolic `2c/τc` |
| `aux_drive_text` | str | `2c/τc` as text (e.g. `'4*D/tauc^2'`) — used as the white aux coefficient |

### The `matched_rows` entry (internal working list)

(`pipeline/colored_to_markovian.py:276`) — `{'term': <the original CGF dict>,
'info': <the detection-info dict>}`.

### The diagnostics record (`builder._markovianize_applied`)

(`pipeline/colored_to_markovian.py:439-443`)

| key | type | meaning |
|---|---|---|
| `aux_fields` | list[str] | aux natural names injected (e.g. `['xi']` or `['xi','yi']`) |
| `matched_rows` | list[str] | names of the colored rows that were transformed |
| `tauc_text` | str | the single shared τc |

### Builder fields read/written

- **Read:** `builder._cgf_terms`, `builder._markovianize_default`,
  `builder.parameters` (list of `ParameterSpec`, `.name`),
  `builder.physical_fields` (list of `FieldSpec`, `.name`, `.natural_name`),
  `builder._action_text`.
- **Written:** `builder._cgf_terms` (overwritten), `builder._action_text`
  (overwritten), `builder._markovianize_applied` (new), plus the side-effect
  appends to `builder.physical_fields` / `builder.response_fields` /
  `builder.parameters` / the equations list via the `physical_field` and
  `equation` methods.

`FieldSpec` (`theory.py:71-86`): `name`, `indexed`, `latex`, `description`,
`natural_name`, `population`, `spatial_dim`. The matcher uses `name` and
`natural_name`. `ParameterSpec` (`theory.py:89-101`): `name`, …; only `name` is
read here.

---

## Data flow

### Concrete example: `ou_quartic_colored` (auto-cumulant)

**Input** (from `theories/ou_quartic_colored.theory.py:13-20`), builder state
just before `markovianize_spec`:

- `physical_fields`: `x` (internal `dx`, response `xt`).
- `_cgf_terms`: one row
  ```python
  {'name': 'CXX', 'response_field': None, 'response_legs': ['xt','xt'],
   'order': 2, 'coefficient': '2*D/tauc', 'kernel': 'exp(-abs(tau)/tauc)',
   'markovianize': None}
  ```
- `_action_text`: `'xt*((Dt+mu)*x + eps*x^3)'`.
- `_markovianize_default`: `True`.

**`detect_lorentzian('exp(-abs(tau)/tauc)', '2*D/tauc', ['x'-derived params...])`**
→ `{'tauc_text': 'tauc', 'amplitude_text': '2*D/tauc', 'aux_drive_text':
'4*D/tauc^2', ...}` (the test pins `tauc_text == 'tauc'` and `aux_drive_text ==
'4*D/tauc^2'`, `tests/test_markovianize.py:71-73`).

**Output** (builder after the transform):

- `physical_fields`: now `['dx', 'dxi']` — the aux `xi` (internal `dxi`,
  auto-response `xit`, auto-saddle `xistar`) was added. Test asserts `phys_names
  == ['dx', 'dxi']` (`tests/test_markovianize.py:116`).
- equations: the user's `(Dt+mu)*x = -eps*x^3` plus the new aux drift `(Dt +
  1/tauc)*xi = 0` (two DAE equations; test comment `tests/test_markovianize.py:122`).
- `_cgf_terms`: the single colored `CXX` row replaced by
  ```python
  {'name': 'CXX_markov_aux', 'response_field': None,
   'response_legs': ['xit','xit'], 'order': 2,
   'coefficient': '4*D/tauc^2', 'kernel': 'dirac_delta(tau)',
   'markovianize': False}
  ```
  Test: the row name ends with `_markov_aux` and the κ² entry's resolved legs are
  `['xit','xit']` (`tests/test_markovianize.py:130,132`).
- `_action_text`: `'xt*((Dt+mu)*x + eps*x^3 - xi) + xit*((Dt + 1/tauc)*xi)'`.

**Downstream:** `make_correlated_noises_block` (`theory_compiler.py:1833`) turns
the white aux row into a `model['correlated_noises']` entry whose order-2 kernel
callable is `_CGFKernelCallable('4*D/tauc^2', 'dirac_delta(tau)', 2)`. The
diagram engine attaches it to the `xit`–`xit` noise vertex. Because the kernel is
now `δ(τ)`, the fast white-noise analytic integrators run.

### Concrete example: cross-correlated 2D

Input three colored rows `Cxx [xt,xt]`, `Cyy [yt,yt]`, `Cxy [xt,yt]` (all kernel
`exp(-abs(tau)/tauc)`; coefficients `D1/tauc`, `D2/tauc`, `rho*sqrt(D1*D2)/tauc`;
`ou_quartic_two_dim_color_corr.theory.py:25-27`). After the transform: aux fields
`xi`, `yi`; aux OU equations for each; three white rows
`Cxx_markov_aux [xit,xit]`, `Cyy_markov_aux [yit,yit]`, `Cxy_markov_aux
[xit,yit]`; action gains `- xi` inside the `xt*(...)` block, `- yi` inside the
`yt*(...)` block, and two appended OU kinetic terms.

### Tree-level sanity check (the consumer's answer)

With `eps=0` (free theory), `compute_cumulants(model, k=2, max_ell=0, ...)` on the
markovianized `ou_quartic_colored` returns `C(0) = 2D/(μ(μτc+1))`
(`tests/test_markovianize.py:170-192`) — the analytic colored-OU variance,
confirming the embedding reproduces the original colored statistics.

---

## Gotchas & caveats

1. **Single-Lorentzian only (v1).** Only `c · exp(-|τ|/τc)` matches. Gaussian,
   bare-exponential `exp(-tau/tauc)` (asymmetric), double-exponential,
   oscillatory, and polynomially-modulated kernels do NOT match and pass through
   to the scipy fallback. Future kernels are enumerated in the docstring
   (`pipeline/colored_to_markovian.py:51-64`) and `correlated_noise_capabilities.md`
   §1.5.

2. **`μ = 1/τc` double-pole.** When the user-field relaxation rate equals the OU
   rate, the two propagator poles collide and the downstream simple-pole residue
   calculus cannot form the multiplicity-2 residue. This module does **not** guard
   against it; the user must keep `μ ≠ 1/τc`. See math §5 and
   `tests/test_markovianize.py:165-168`.

3. **`tau` symbol pollution.** The matcher deliberately uses a fresh
   `SR.var('_lorentz_tau')` instead of the global `tau`, because some theories
   declare a positive-domain `tau` parameter that would make `abs(tau)` collapse
   to `tau` and cause the matcher to wrongly accept asymmetric kernels. Long
   comment at `pipeline/colored_to_markovian.py:116-127`. This is subtle and
   order-dependent: a regression here would only show up if a positive `tau` was
   registered earlier in the same Sage session.

4. **`order != 2` rows are passed through silently.** Only second cumulants are
   embedded. A colored *third* cumulant would not be markovianized and would hit
   the (slow) scipy path. (`pipeline/colored_to_markovian.py:250-252`.)

5. **Action-text surgery is fragile by design.** `_inject_aux_coupling` only
   handles the canonical `<rt>*(...)` layout and edits the **first** match. If the
   user's action is "more baroque," the insertion silently no-ops and the
   coupling is missing — producing a *silently wrong* theory rather than an error.
   The docstring tells the user to `.markovianize(False)` and hand-roll in that
   case (`pipeline/colored_to_markovian.py:418-423, 481-485`). This is the most
   surprising failure mode: there is no validation that the `- xi` actually landed.

6. **Positivity of τc is documented but not enforced.** The docstring says τc
   "must be POSITIVE for a physical Lorentzian" (lines 163-164) and that the
   matcher requires the decay rate to "evaluate to a strictly positive
   expression" (lines 91-93), but the code only checks that τc is **tau-/abs-
   free**, never that it is numerically positive. A symbolic τc that happens to be
   negative for some parameter values would still be accepted. (Open question.)

7. **Multi-τc raises `NotImplementedError`.** If matched rows have *different* τc
   expressions, v1 bails loudly (`pipeline/colored_to_markovian.py:323-333`). All
   worked examples use a single shared τc.

8. **Explicit `markovianize=True` on a non-matching kernel raises `ValueError`.**
   This is intentional — it lets a user assert "this IS a Lorentzian, fail if you
   disagree" rather than silently falling back (`pipeline/colored_to_markovian.py:264-272`).

9. **Idempotency relies on the aux rows carrying `markovianize=False`.** If that
   flag were dropped, a second `build()` pass could try to re-embed the white aux
   rows. The aux row dict sets it explicitly (line 404).

10. **`response_to_natural[rname] = None` fallback.** If no physical field's
    auto-response matches a leg name, that row is left as the original colored row
    (falls back to scipy) rather than erroring. So a typo in `response_legs` won't
    crash — it silently de-optimizes (`pipeline/colored_to_markovian.py:313-316,
    385-390`).

11. **Saddle parameter side effect.** Injecting an aux `physical_field` also
    auto-creates a saddle parameter `xistar` (via `theory.py:899-910`). For a
    centered OU noise the saddle is zero; nothing in this module sets or checks it.

---

## Glossary

- **MSR-JD field theory** — Martin–Siggia–Rose / Janssen–De Dominicis path-
  integral representation of a stochastic differential equation. Each physical
  field `x` gets a conjugate **response field** `x̃` (written `xt`). The action's
  bilinear structure encodes both the deterministic dynamics and the noise.
- **CGF / CGF term / CGF row** — Cumulant-Generating-Functional term. A declared
  contribution to the noise's cumulants, given as `(coefficient, kernel, order,
  response_legs)`. Order 2 = the noise covariance.
- **Cumulant** — a connected correlation function. The 2nd cumulant of the noise
  is its covariance `<ξ ξ>`.
- **Colored noise** — noise with a finite correlation time; its covariance is a
  smooth function of the time lag τ (here `c·exp(-|τ|/τc)`), not a delta.
- **White noise** — noise with zero correlation time; covariance `∝ δ(τ)`.
- **Lorentzian (kernel)** — `exp(-|τ|/τc)`, named for its Lorentzian Fourier
  transform `∝ 1/(ω²+1/τc²)`.
- **OU (Ornstein–Uhlenbeck) process** — the simplest mean-reverting linear SDE,
  `dξ/dt = -ξ/τc + (white)`. Its stationary covariance is a single Lorentzian.
- **Markovian embedding** — enlarging the state space (adding auxiliary fields) so
  that a non-Markovian (memory-carrying, colored) process becomes Markovian
  (memoryless, white-driven). The output field is unchanged; only the
  representation grows.
- **τc (tauc)** — the noise correlation time / OU relaxation time.
- **`max_ell` / ℓ** — loop-order cutoff in the perturbative expansion. `ℓ = 0`
  tree level; `ℓ ≥ 1` includes loop corrections (where the scipy fallback hangs).
- **Auxiliary field (aux field)** — the injected OU field (`xi`/`ξ`) and its MSR-JD
  response partner (`xit`/`ξ̃`).
- **Saddle / mean-field parameter** — the `<field>star` parameter holding a field's
  steady-state value; auto-created with each `physical_field`.
- **`Dt`** — the time-derivative operator symbol usable in action/equation text.
- **`dirac_delta(tau)`** — Sage's symbolic Dirac delta; the white-noise kernel.
- **`nquad`** — SciPy's nested numerical multi-dimensional quadrature; the slow
  fallback this subsystem exists to avoid.
- **SageMath `SR`** — the Symbolic Ring: Sage's engine for symbolic algebra
  (variables, simplification, parsing).
- **`sage_eval`** — Sage's namespaced string→symbolic-expression evaluator.
- **`.operator()` / `.operands()`** — Sage methods returning an expression's
  outermost operation and its arguments.
- **`.simplify_full()`** — Sage's most aggressive symbolic simplifier.
- **`abs_symbolic`** — Sage's symbolic absolute value (keeps `|τ|` unevaluated).
- **`TheoryBuilder` / `TemporalTheoryBuilder`** — the fluent accumulator users
  call to author a theory; `.build()` emits the model dict.
- **Idempotent** — running the operation twice has the same effect as once.

---

## Proposed manual subsections

1. **Motivation: why colored noise breaks the fast integrators** — the
   `exp(-|τ|/τc)`-vs-`δ(τ)` distinction and the `ℓ ≥ 1` `nquad` blow-up.
2. **The single-Lorentzian autocovariance** — definition, symmetry, white-noise
   limit, Fourier (Lorentzian) view.
3. **The OU filter and why its stationary covariance is exactly Lorentzian** —
   full derivation of `<ξξ>(τ) = c·e^{-|τ|/τc}` and the `2c/τc` white drive.
4. **From SDE to MSR-JD action: the three structural pieces** — aux field, aux OU
   drift equation, aux white CGF row; the `-xi` coupling and `xit·(Dt+1/τc)·xi`
   kinetic term.
5. **The cross-correlated multi-field case** — per-field aux processes and the
   single cross CGF row; how the shared τc factorizes the covariance matrix.
6. **The kernel matcher (`detect_lorentzian`)** — the `arg/|τ|` decomposition, the
   abs-/tau-free tests, and the `_lorentz_tau` pollution guard.
7. **The builder transform (`markovianize_spec`)** — gating, classification,
   name resolution, injection, action surgery, idempotency.
8. **Worked example: `ou_quartic_colored` before/after** — the exact spec diff and
   the tree-level analytic cross-check `C(0)=2D/(μ(μτc+1))`.
9. **Limitations and how to live with them** — the `μ=1/τc` double pole; single
   τc; order-2 only; action-layout fragility and the `.markovianize(False)`
   escape hatch.
10. **Roadmap (v2)** — underdamped, double-exponential, polynomial kernels as
    larger auxiliary state spaces.
