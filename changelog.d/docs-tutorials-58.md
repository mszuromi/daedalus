#### Added: tutorial series and a new front page (item 58)

Documentation only; no code or number changes.

- **`notebooks/tutorials/`**: eight executed notebooks that build every model with the SDE / point-process front end
  (00 quick start, 01 the SDE front end, 02 the noise library with a live table of every source, jump law and
  composite, 03 point processes, 04 delays, 05 outputs, 06 writing the action by hand, 07 spatial models).
- **Delay examples** in `notebooks/examples/SDE/`: `Delayed_OU` (hard delay in the drift), `Delayed_product`
  (`x*x(t - d)`) and `Hawkes_delayed_alpha` (point processes coupled through delayed alpha kernels), each with a
  from-scratch simulation overlay.
- `notebooks/model_builder_tutorial.ipynb` moved to `notebooks/legacy/`; `README.md` (front page: why, quick start,
  tutorial map, noise library summary, capabilities and limits, citation), `notebooks/README.md` and
  `notebooks/examples/README.md` rewritten around the tutorials. The weekly notebook workflow also executes
  `notebooks/tutorials/` (timeout overrides are now keyed by the path under `notebooks/`).

