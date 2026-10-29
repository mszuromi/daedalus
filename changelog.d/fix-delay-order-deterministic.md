#### Delay route: the automatic Padé order is chosen by the error estimate alone, and a pinned order is estimated

**The automatic order no longer depends on the machine.** The search doubled the order (4, 8, 16) only while the
predicted wall time of the next order stayed under `wall_max` (900 s). That prediction is a measured wall time, so
the same script stopped at different orders on a busy and a quiet machine. Example: a delayed two-neuron Hawkes
two-loop rate stopped at n = 8 in about 195 s on a loaded machine, and reached n = 16 in about 650 s on a quiet one.
The results differed in their last digits and the cost differed threefold.
- `wall_max` now defaults to `None` (no cap). The search stops when the estimate meets `tol`, at `n_max`, or when a
  higher order cannot run (the memory budget, a diverging difference); each of these stops is reported.
- An explicit `delay={'wall_max': s}` is still honoured. Its stop says it depends on the machine's speed and load,
  in `result['delay']['cap']` / `['stop_reason']`, in the `DelayOrderWarning`, and in `status = 'target_missed'`.
- Searches that never reached the old cap run as before, bit for bit.
- Searches that hit it now go on to a higher order.

**A pinned order gets an error estimate.** `delay={'order': n}` used to report no estimate (`est_by_order == {}`)
and no warning. It now also runs the same pipeline at `n // 2` and reports the search's own estimate of order `n`:
- `est_rel_error` and `est_by_order[n]`, with `estimate_order = n // 2`;
- `target_met`, and when `tol` is missed, a `DelayOrderWarning` and `status = 'target_missed'`.

The run at `n` is unchanged and is still the one returned. Example: the one-stage delayed negative feedback at
`d = 5`, whose one-loop term moves by about 10 % from n = 8 to n = 16. A pinned n = 8 now reports 5.7e-5 relative
to max |C| and a missed target, where it used to report nothing. To run the order alone, as before, use
`delay={'order': n, 'estimate': False}` (no estimate, no warning).

**The chosen order and its stop are always on record.** The `G3_DELAY_APPROXIMATED` finding names the stop
(`order: target 1e-08 met at n = 8 ...`, `n_max = 16 reached`, `fixed order n = 8`), and a verbose run prints it.
The Hartree closure's delay search no longer has a wall cap by default either.
