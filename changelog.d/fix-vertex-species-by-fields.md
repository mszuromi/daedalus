#### Spatial operator IR: derivative vertices of different fields at one leg count stay separate

The form-factor table of the operator IR (`ns._operator_ir_vertex_terms`) matched a diagram node to its entries by the
node's physical-leg count alone. In a model with several fields, vertices on different fields with the same leg count
(`-c1*Lap(a^2)` and `-c2*Dx(b, 0)^2`, both `phit*phi^2`) were therefore treated as one group: on the single-term
path each entry took the weight `c_t / (c1 + c2)` and every node of either field summed both form factors, and on
the multi-term path the model was refused ("two vertex groups with the same physical-leg count but different fields").
The one-loop correlator of one field then depended on the other field's vertex, or was refused as UV-divergent.

- **Fix.** Each table entry carries `fields`, the sorted propagator-matrix columns of its physical legs, and a node
  only takes the entries of its own field content (`diagram_form_factor`). The single-term table groups by field
  content, and the multi-term table no longer refuses a second field's vertex.
- **Unchanged.** A model with one field, or with distinct leg counts per field, gives the same table (plus the new
  `fields` key), the same lowered action and bit-identical numbers: checked by `array_equal` on the table, the action
  and the k = 2 tree and one-loop values of every tracked spatial model. Two fields with the SAME vertex type (both
  `Lap(.^2)`) already came out right, because the weights summed to 1 over equal form factors.
- **Cache.** The operator-IR expand-cache signature now includes `fields`: an operator-IR model's expand cache is
  rebuilt once.
- **Moved numbers.** Decoupled `a`, `b` (`ma = 1, mb = 1.6, DD = T = 1`, `c1 = 0.3, c2 = 0.2`), k = 2 one loop at
  `x = 0, 0.5, 1`, tau = 0: with `-c1*Dx(a, 0)^2` and `-c2*Lap(b^2)` before: both fields refused as UV-divergent;
  after: `a` = 0.0037887, 0.00353117, 0.00293334 and `b` = 0.0082327, 0.00015896, -0.0009592, equal to the one-field
  models. A coupled (non-diagonal) multi-field model with a derivative vertex is still refused at loops and k >= 3.
- **Checks** (`tests/test_operator_ir_vertex_fields.py`): the table of two fields at one leg count; a multi-term group
  next to another field's vertex; decoupled-field loops and k = 3 trees against the one-field models.
