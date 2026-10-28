#### Symbolic output: one-loop tadpoles take the Itô equal-time value where `C_pq(tau)` jumps at the tie

**Bug fixed (wrong numbers, no warning).** `dd.one_loop_mean` (since its introduction; not in v0.4.0) returned a wrong
one-loop mean when a tadpole loop `C0_pq(0)` pairs the instantaneous part of one leg with the retarded response of the
other. That happens with an algebraic field carrying white noise, or with a line that has a `delta` part, such as a spike
train. There `C_pq(tau)` jumps at `tau = 0`:
- the `'rational'` and `'residues'` forms took the one-sided limit `C_pq(0+)` picked by the order of the loop's legs;
- the `'integral'` form took the midpoint.

The Itô value (`G(0) = 0`) counts no response at the tie.

Hand model: `x' = -x + w y + lam x y`, `y = h + c x + eta`, with `eta` white of variance `D`, at `w = 0.3`, `lam = 0.2`,
`h = 0.5`, `c = 0.4`, `D = 0.1`. Itô: `x(t)` does not see `eta(t)`, so `<dx dy> = c Var x` and
`<dx>^(1) = lam c b^2 D / (2 kappa^2)`, with `b = w + lam x*` and `kappa = 1 - lam y* - c b`.

| `<dx>^(1)`, old → new | old meaning | either order of the loop's legs |
|---|---|---|
| `'rational'`, `'residues'`: `9.8851e-3` → `8.2150e-4` | `C_xy(0+)`, which adds `b D` | yes |
| `'integral'`: `5.3533e-3` → `8.2150e-4` | the midpoint | yes |

`<dy>^(1)` scales by `c`: `3.9541e-3` and `2.1413e-3` → `3.2860e-4`. The new value is the hand value and `dd.run`'s.

`dendritic_quad_soma_sigmoid` at its defaults, `'integral'` form, old → new (the pipeline's value):
- `dnS1`: `1.0453e-4` → `6.6096e-5`;
- `dvD1`: `1.3589e-5` → `8.5925e-6`;
- `dvS1`: `1.3330e-4` → `6.4602e-6`;
- `dnD1`: `1.0254e-3` → `4.9694e-5`.

Its `'rational'` and `'residues'` forms were already right, because the order of the loop's legs happened to pick the
Itô side.

**Not affected:** `dd.run` and `dd.symbolic_cumulant`. Their `k = 1`, `ell = 1` mean is the Itô value for the hand
model in both namings (`8.2150e-4`, `3.2860e-4`), before and after. Their `k = 2` equal-time output at a jump is the
external tie convention (the second leg earlier, `C_ab(0+)`): `C_xy(0)` is `3.7000e-2` and `C_yx(0)` is `3.0749e-3`.
That is a statement about an observable, not a loop, and it does not change.

**The fix.**
- `TreeCovariance.ito_entry(a, b)` is `C0_ab(omega)` without the terms that pair the instantaneous part
  `G_inf = lim G` of one line with the retarded part of the other: `[G_r D G_r^dagger]_ab`, with `G_r = G - G_inf`. It
  decays like `omega^-2`, and its integral is the Itô equal-time value.
- `TreeCovariance.tie_term(a, b)` is the term `C_ab(0+)` adds.
- The `'integral'` and `'residues'` forms integrate `ito_entry`. The `'rational'` form takes the state-space tie value
  minus `tie_term`.
- A loop whose `C0_pq` has a contact term (`delta(tau)`) is refused in every form. Before, the `'rational'` form
  returned the tie value.
