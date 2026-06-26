"""Generate notebooks/build_your_own_theory.ipynb — a guide to authoring
TEMPORAL theories for the Daedalus MSR-JD pipeline.  Covers every component
of a temporal theory, both the graphical builder and the Python builder, with
a Langevin and a Hawkes worked example.  No simulation.  Run with python3."""
import json, os

CELLS = []
def md(t):   CELLS.append(('markdown', t))
def code(t): CELLS.append(('code', t))

# ───────────────────────────────────────────────────────────────────────
md(r"""# Building a temporal theory

Declare a temporal theory for the MSR-JD pipeline and compute its correlators. Sections: the
action (§1), the theory components (§2), the graphical builder (§3), the Python builder (§4),
worked examples (§5 Langevin SDE, §6 Hawkes process), your own theory (§7), syntax reference (§8).

Temporal theories only (no spatial extent). No simulation — mean field, cumulants, and loop
corrections are computed analytically.""")

# ───────────────────────────────────────────────────────────────────────
md("""## 0. Setup

This notebook requires the **SageMath** kernel (the pipeline is built on Sage). If a cell
errors on `import`, check the kernel selector — it should read *SageMath*, not a plain
Python 3. The cell below puts the repository on the path and imports the builder and the
`daedalus` front-end; it is the same setup used by `theory_runner.ipynb`.""")

code("""%matplotlib inline
import os, sys
import numpy as np
import matplotlib.pyplot as plt

# Locate the repo root (walk up until the 'pipeline' package appears) and add it to the path.
_root = os.path.abspath('')
while _root != os.path.dirname(_root) and not os.path.isdir(os.path.join(_root, 'pipeline')):
    _root = os.path.dirname(_root)
sys.path.insert(0, _root)
sys.path.insert(0, os.path.join(_root, 'notebooks'))

import daedalus as dd                            # run / summary / plot front-end
from pipeline.theory import TemporalTheoryBuilder
print('daedalus →', dd.REPO_ROOT)""")

# ───────────────────────────────────────────────────────────────────────
md(r"""## 1. The action

A theory is one expression: the MSR-JD action $S$, a functional of the physical fields and an
equal number of response fields. The pipeline derives the propagators, vertices, mean field,
and cumulants from it. Form:

$$S \;=\; \sum_{\text{fields}}(\text{response field})\times(\text{equation of motion}) \;+\; (\text{source terms}).$$

Build $S$ from the pieces below; the right column is what goes in `set_action_text` (§2.8).
Field `x` has response field `xt`.

| piece | equation | action term |
|---|---|---|
| drift / relaxation | $\dot x = -\mu x - \varepsilon x^3$ | `xt*((Dt+mu)*x + eps*x^3)` |
| white Gaussian noise | $\langle\xi\xi\rangle = 2D\delta$ | `- D*xt^2` |
| point-process spikes | $n \sim \mathrm{Poisson}[\varphi(v)]$ | `nt*n - (exp(nt)-1)*phi(v)` |
| memory / synaptic kernel | $\tau\dot v = -(v-E) + w\,(g*n)$ | `vt*((tau*Dt+1)*v - E - w*Conv(g,n))` |
| cross-correlated noise | $\langle\xi_a\xi_b\rangle = \rho\sqrt{D_aD_b}\,\delta$ | `- 2*rho*sqrt(Da*Db)*at*bt` |

Multi-population models replicate fields over an index $i$ and sum the action `for i in pop`.
§5 assembles the first two rows; §6 assembles rows 1–4.""")

# ───────────────────────────────────────────────────────────────────────
md(r"""## 2. The components of a temporal theory

The ten ingredients below are the tabs of the graphical builder (§3) and the methods of the
Python builder (§4). *Model*, *Fields*, *Parameters*, *Action*, and *Mean-field* are required;
the rest are used as needed.

### 2.1 Model
A name (and optional description). On save it becomes the `.theory.py` filename.
Python: the constructor argument, `TemporalTheoryBuilder('My model')`.

### 2.2 Populations *(optional)*
A **population** is a set of identical units sharing the same dynamics — $N$ neurons, $N$
spins. Declared with a name and size; thereafter fields and parameters carry an index `[i]`
and the action sums `for i in pop`. Omit it for a single scalar variable.
Python: `.population('E', size=4)`.

### 2.3 Fields
The physical variables — what you would write on the left of an SDE. Declaring a field `x`
**automatically** creates two companions:

| symbol | role | where it appears |
|---|---|---|
| `x` | the field | the equation of motion, the mean-field equations |
| `xt` | its MSR **response** field | the response×EOM term and the source terms |
| `xstar` | the **saddle** (steady state) | solved for by the mean-field equations |

Python: `.physical_field('x', population='E')` (drop `population` for a scalar). Internally the
*fluctuation* about the saddle is named `dx` — that prefixed name is what you put on a
correlator leg (§2.10).

### 2.4 Parameters
Numerical constants — rates, couplings, time constants, noise amplitudes. Shape is set by how
many population indices they carry:

| `indexed_by` | shape | written as | `default` |
|---|---|---|---|
| `None` | scalar | `mu` | `1.0` |
| `['E']` | vector $N_E$ | `mu[i]` | `[1.0, 2.0]` |
| `['E','E']` | matrix $N_E\times N_E$ | `w[i,j]` | `[[1,0.5],[0.5,1]]` |

`domain='positive'` / `'real'` guides the mean-field Newton solver. Python:
`.parameter('mu', default=1.0, domain='positive', indexed_by=['E'])`.

### 2.5 Transfer functions *(optional)*
Non-polynomial transformations of a field — a firing-rate curve $\varphi(v)$, `tanh`, a
sigmoid. (Plain polynomials like `x^3` go straight in the action; you do not need this for
them.) The pipeline Taylor-expands the function about the saddle for you. Python:
`.define_function('phi', args=['v'], expression='a[i]*v^2', population='E')`, then call it in
the action as `phi[i](v[i])`.

### 2.6 Kernels *(optional)*
Temporal **convolutions** — synaptic filters, memory, colored noise. A kernel $g(t)$ couples a
field to a time-filtered version of another. Give it as a time expression (use `heaviside(t)`
for causality) and/or its Fourier image. Python:
`.define_kernel('g', time_expr='(1/taug)*exp(-t/taug)*heaviside(t)', freq_image='1/(1+I*omega*taug)', indexed_by=['E','E'])`,
then in the action `Conv(g[i], n[j])` (or `g[i,j]*n[j]`).

### 2.7 Noise / source terms
The stochastic drive. It enters the action as one of:

| noise | action term |
|---|---|
| white Gaussian, $\langle\xi\xi\rangle = 2D\delta$ | `- D*xt^2` |
| point process / spikes, rate $\varphi$ | `- (exp(nt)-1)*phi(v)` |
| cross-correlated between fields $a,b$ | `- 2*rho*sqrt(Da*Db)*at*bt` |
| colored (finite correlation time) | a non-delta kernel on the noise (see the Noise tab) |

White Gaussian and point-process sources are written directly in the action (below); colored
and higher-cumulant noise can instead be declared on the builder's Noise tab.

### 2.8 The action
The MSR–JD action itself — the assembly of §2.1–2.7 into one expression of the form
*response × equation-of-motion + sources*. Syntax:

- physical field `x[i]`, response `xt[i]`; index `[i]` ranges over the population
- `Dt` is $\partial_t$: write `(Dt + mu)*x[i]`, `(tau[i]*Dt + 1)*v[i]`
- `^` is a power: `x[i]^3`
- `sum( expr for i in E )` sums over a population; inner sums `sum(w[i,j]*n[j] for j in E)`
- transfer functions `phi[i](v[i])`; convolutions `Conv(g[i], n[j])`
- write the **physical** field — the pipeline does the saddle + fluctuation split itself; do
  not write `xstar + dx` by hand

### 2.9 Mean-field equations
The deterministic equations whose steady state gives the saddle `xstar` the diagrams expand
around. One per field. `Dt` may appear; the solver sets `Dt → 0` and runs multi-start Newton.
Python: `.equation(lhs='(Dt+mu)*x[i]', rhs='-eps*x[i]^3', population='E')`, or for an algebraic
self-consistency, `.set_mf_equation('nstar', 'phi[i](vstar[i])')`. For multiple roots
(bistability) the solver sorts them and `fixed_point_index` picks one.

### 2.10 Run settings
Not part of the theory, but how you query it: the correlator order `k` (2 = two-point), the
loop order `max_ell` (0 = tree, 1 = +1-loop, …), the time-lag grid, and the **external legs** —
which fluctuation fields sit on the correlator. A leg names the `d`-prefixed field, e.g.
`[('dx', 1), ('dx', 1)]` for $\langle x\,x\rangle$; leave it `None` to default to the
auto-correlator of the first field.""")

# ───────────────────────────────────────────────────────────────────────
md(r"""## 3. Graphical builder

[`theory_builder.ipynb`](theory_builder.ipynb) is a form whose tabs are the §2 components
(*Model · Populations · Fields · Parameters · Functions · Kernels · Noise · Action · MF ·
Defaults*); a sidebar flags undeclared names and syntax errors, and **Save** writes
`theories/<name>.theory.py`. Launch it here:""")

code("""from pipeline.ui import TheoryUI
ui = TheoryUI()
ui.show()      # fill the tabs, then 'Save theory file' → theories/<name>.theory.py""")

md("""Load a saved theory by name (in `theory_runner.ipynb`, or here):

```python
model, mod = dd.load_theory('<name>')        # filename minus '.theory.py'
res = dd.run(model, dd.Config(k=2, max_ell=0), mod)
```""")

# ───────────────────────────────────────────────────────────────────────
md(r"""## 4. Python builder

Each §2 component is one chained method; `.build()` returns the `model`:

```python
model = (
    TemporalTheoryBuilder('name')      # 2.1 Model
    .population(...)                   # 2.2 Populations   (optional)
    .physical_field(...)              # 2.3 Fields
    .parameter(...)                   # 2.4 Parameters
    .define_function(...)             # 2.5 Transfer functions (optional)
    .define_kernel(...)               # 2.6 Kernels        (optional)
    .set_action_text('...')           # 2.7-2.8 Sources + action
    .equation(...) / .set_mf_equation(...)   # 2.9 Mean field
    .build()
)
```""")

# ───────────────────────────────────────────────────────────────────────
md(r"""## 5. Example — Langevin SDE

Model: $\dot x = -\mu x - \varepsilon x^{3} + \xi$, $\langle\xi\xi\rangle = 2D\delta$.
Action: `xt*((Dt+mu)*x + eps*x^3) - D*xt^2`.""")

code("""ou = (
    TemporalTheoryBuilder('Quartic OU process')
    .population('pop', size=1)                              # 2.2  one scalar unit
    .physical_field('x', population='pop',                  # 2.3  field x (+ response xt, saddle xstar)
                    description='the state variable')
    .parameter('mu',  default=1.0,  domain='positive')     # 2.4  relaxation rate
    .parameter('eps', default=0.05, domain='positive')     # 2.4  cubic nonlinearity
    .parameter('D',   default=1.0,  domain='positive')     # 2.4  noise strength  <xi xi> = 2 D delta
    .set_action_text(                                       # 2.8  response x EOM  +  Gaussian source
        'sum( xt[i]*((Dt+mu)*x[i] + eps*x[i]^3) - D*xt[i]^2  for i in pop )')
    .equation(lhs='(Dt+mu)*x[i]', rhs='-eps*x[i]^3',        # 2.9  deterministic EOM
              population='pop')
    .build()
)
dd.describe_model(ou);""")

md(r"""Run and plot $C(\tau)$:""")

code("""ou_cfg = dd.Config(
    k=2,                                      # two-point correlator <x x>
    max_ell=0,                                # tree level; raise to 1 for the 1-loop correction
    external_fields=[('dx', 1), ('dx', 1)],   # both legs on the x-fluctuation (or leave None)
    tau_max=8.0, tau_step=0.5,
)
ou_res = dd.run(ou, ou_cfg, None)             # mod=None: an inline-built model needs no theory file
print(dd.summary(ou_res))

fig = dd.plot_cumulant(ou_res, ou_cfg, ou)
plt.show()""")

# ───────────────────────────────────────────────────────────────────────
md(r"""## 6. Example — Hawkes process

Two units; spike train $n_i$ at rate $\varphi(v_i)=a_i v_i^2$; voltage $v_i$ integrating past
spikes through an alpha kernel $g_{ij}(t)=(t/\tau_g^2)e^{-t/\tau_g}\Theta(t)$:

$$\tau_i\,\dot v_i = -(v_i - E_i) + \textstyle\sum_j w_{ij}\,(g_{ij} * n_j), \qquad n_i \sim \mathrm{Poisson}[\varphi(v_i)].$$

Uses a population (2.2), two fields (2.3), vector + matrix parameters (2.4), a transfer
function (2.5), a kernel (2.6), a point-process source (2.7), and algebraic mean-field
equations (2.9).""")

code("""hawkes = (
    TemporalTheoryBuilder('Quadratic Hawkes (alpha-kernel)')
    .population('E', size=2, description='excitatory units')                    # 2.2
    .physical_field('n', population='E', description='spike train')             # 2.3
    .physical_field('v', population='E', description='synaptic voltage')        # 2.3
    .parameter('Em',   default=[0.8, 0.78],        indexed_by=['E'],      domain='positive')   # 2.4 vector
    .parameter('tau',  default=[10, 9],            indexed_by=['E'],      domain='positive')
    .parameter('a',    default=[0.44, 0.44],       indexed_by=['E'],      domain='positive')
    .parameter('taug', default=[[2, 3], [1, 3]],   indexed_by=['E', 'E'], domain='positive')   # 2.4 matrix
    .parameter('w',    default=[[0.25, 0.25], [0.2, 0.3]], indexed_by=['E', 'E'], domain='positive')
    .define_function('phi', args=['v'], expression='a[i]*v^2', population='E')  # 2.5 firing-rate curve
    .define_kernel('g', latex_name='g', indexed_by=['E', 'E'],                  # 2.6 alpha synaptic filter
                   time_expr='(t/taug[i,j]^2)*exp(-t/taug[i,j])*heaviside(t)')
    # 2.8 action: spike source  nt*n - (exp(nt)-1)*phi(v)  +  voltage EOM with synaptic convolution
    .set_action_text('''
        sum( nt[i]*n[i] - (exp(nt[i])-1)*phi[i](v[i])
        + vt[i]*((tau[i]*Dt + 1)*v[i] - Em[i]
        - sum(w[i,j]*g[i,j]*n[j] for j in E))
        for i in E)''')
    .set_mf_equation('vstar', '(Em[i] + sum(w[i,j]*g[i,j]*nstar[j] for j in E))')   # 2.9 algebraic MF
    .set_mf_equation('nstar', 'phi[i](vstar[i])')
    .build()
)
dd.describe_model(hawkes);""")

md(r"""Spike-train auto-correlator $C_{nn}(\tau)$ (legs on the `n`-fluctuation, `dn`):""")

code("""hawkes_cfg = dd.Config(
    k=2, max_ell=0,
    external_fields=[('dn', 1), ('dn', 2)],   # spike-train (n) auto-correlator
    tau_max=20.0, tau_step=2.5,
)
hawkes_res = dd.run(hawkes, hawkes_cfg, None)
print(dd.summary(hawkes_res))

fig = dd.plot_cumulant(hawkes_res, hawkes_cfg, hawkes)
plt.show()""")

# ───────────────────────────────────────────────────────────────────────
md(r"""## 7. Your own theory

Edit the cell below (starts as the §5 OU); for a point-process or multi-population model, copy
the §6 chain instead.

- a field's action drift term is minus its `equation` / `set_mf_equation` right-hand side —
  keep them consistent;
- `max_ell=0` is tree level; raise for loop corrections;
- `external_fields=None` gives the first field's auto-correlator; name the legs for a
  cross-correlator.""")

code("""my_model = (
    TemporalTheoryBuilder('My Theory')                     # EDIT: name it
    .population('pop', size=1)                              # EDIT: a population, or keep size=1 for a scalar
    .physical_field('x', population='pop',                  # EDIT: your field(s)
                    description='the state variable')
    .parameter('mu',  default=1.0,  domain='positive')     # EDIT: your parameters
    .parameter('eps', default=0.05, domain='positive')
    .parameter('D',   default=1.0,  domain='positive')
    .set_action_text(                                       # EDIT: your action (response x EOM + source)
        'sum( xt[i]*((Dt+mu)*x[i] + eps*x[i]^3) - D*xt[i]^2  for i in pop )')
    .equation(lhs='(Dt+mu)*x[i]', rhs='-eps*x[i]^3',        # EDIT: keep consistent with the action
              population='pop')
    .build()
)
dd.describe_model(my_model);""")

md("""Configure and run your theory:""")

code("""my_cfg = dd.Config(
    k=2, max_ell=0,
    external_fields=None,        # None → auto-correlator of the first field; or e.g. [('dx',1),('dx',1)]
    tau_max=8.0, tau_step=0.5,
)
my_res = dd.run(my_model, my_cfg, None)
print(dd.summary(my_res))

fig = dd.plot_cumulant(my_res, my_cfg, my_model)
plt.show()""")

# ───────────────────────────────────────────────────────────────────────
md(r"""## 8. Reference

**Action syntax**

| token | meaning |
|---|---|
| `Dt` | time derivative $\partial_t$ |
| `^` | power, e.g. `x[i]^3` |
| `x[i]`, `xt[i]` | a field and its response partner at index `i` |
| `sum(expr for i in E)` | sum over a population (omit for a scalar) |
| `phi[i](v[i])` | a declared transfer function |
| `Conv(g[i], n[j])`, `g[i,j]*n[j]` | a kernel convolution |

**Source-term forms**

| noise | term in the action |
|---|---|
| white Gaussian $\langle\xi\xi\rangle=2D\delta$ | `- D*xt^2` |
| point process at rate $\varphi$ | `- (exp(nt)-1)*phi(v)` |
| cross-correlated $a,b$ | `- 2*rho*sqrt(Da*Db)*at*bt` |

**Gotchas**

- No `Conv(...)` in mean-field equations — the stationary saddle has already collapsed
  convolutions of constants; put kernels only in the action.
- `max_ell >= 1` with colored noise is slow (an extra $\tau$-integral per diagram).
- For a bistable theory, enable the stability filter (MF tab / `stability_analysis(True)`) so
  the expansion sits at a linearly stable saddle.
- A purely Gaussian theory needs only the order-2 noise term; non-Gaussian noise needs its
  higher cumulants declared explicitly (Noise tab).

**Where to go next**

- [`theory_builder.ipynb`](theory_builder.ipynb) — the graphical form (§3), with a live
  error-checking sidebar and a pre-compute button.
- [`theory_runner.ipynb`](theory_runner.ipynb) — load any saved `theories/*.theory.py` by name
  and run it with one `dd.Config`.
- [`examples/`](examples/) — one notebook per pipeline capability (these *do* include
  simulation overlays for validation).

**Saving an inline theory as a reusable file.** Wrap the builder chain in a `build()` function
in `theories/<name>.theory.py`, and add module-level `DEFAULT_FUNDAMENTAL = {...}` (numeric
defaults) and `METADATA = {...}` (`k_default`, `tau_max`, `recommended_external_fields`, …).
Copy the structure of any file in `theories/` — `ou_quartic.theory.py` is §5, and
`quadratic_hawkes_alpha.theory.py` is §6. It then loads by name in the runner.""")

# ───────────────────────────────────────────────────────────────────────
nb = {
    'cells': [
        {'cell_type': t, 'metadata': {},
         **({'source': s.splitlines(keepends=True)} if t == 'markdown'
            else {'source': s.splitlines(keepends=True), 'outputs': [], 'execution_count': None})}
        for (t, s) in CELLS
    ],
    'metadata': {
        'kernelspec': {'display_name': 'SageMath 10.8', 'language': 'sage', 'name': 'sagemath-10.8'},
        'language_info': {'name': 'python'},
    },
    'nbformat': 4, 'nbformat_minor': 5,
}
out = os.path.join('notebooks', 'build_your_own_theory.ipynb')
with open(out, 'w') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
    f.write('\n')
print('wrote', out, 'with', len(CELLS), 'cells')
