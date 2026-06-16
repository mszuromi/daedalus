export const meta = {
  name: 'migrate-sim-compare-to-nb-support',
  description: 'Migrate each pipeline *_sim_compare notebook onto the nb_support core (load_theory + Config + run), preserving its simulator/plots/diagnostics; extract a theory file if the model is built inline; fix stale-API breakage; verify by execution',
  phases: [
    { title: 'Migrate', detail: 'one agent per notebook: swap theory-load+compute to nb_support, keep sim/plots, execute-verify' },
  ],
}

const VERDICT = {
  type: 'object',
  additionalProperties: false,
  required: ['path','status','theory_file','exec_ok','exec_secs','changed','c_check','notes'],
  properties: {
    path: { type: 'string' },
    status: { type: 'string', enum: ['migrated','migrated_and_fixed','left_as_is','failed'],
      description: 'migrated = clean swap; migrated_and_fixed = swap + repaired a stale-API break; left_as_is = judged too risky/no value (explain in notes); failed = could not get it running' },
    theory_file: { type: ['string','null'], description: 'theories/<name>.theory.py created (path) or reused (name), or null' },
    exec_ok: { type: 'boolean', description: 'did the final notebook execute end-to-end with 0 errors' },
    exec_secs: { type: ['number','null'] },
    changed: { type: 'string', description: 'one-line summary of cells changed (e.g. "cells 1,5 -> nb_support load+run+shim; cell 11 sim dict -> plot_cumulant")' },
    c_check: { type: 'string', description: 'the representative cumulant value(s) after migration and whether they match the pre-migration physics (same config -> same numerics)' },
    notes: { type: 'string', description: 'anything important: what was preserved, what stale key was fixed, residual concerns' },
  },
}

phase('Migrate')
const items = typeof args === 'string' ? JSON.parse(args) : args

const PREAMBLE = `You are migrating ONE Jupyter notebook onto the shared engine notebooks/nb_support.py, CONSERVATIVELY. The notebook compares diagrammatic theory (pipeline.compute_cumulants) against a matched simulator and has validated physics + diagnostics you MUST NOT break or strip.

The shared engine (read notebooks/nb_support.py first) provides:
  import nb_support as nb
  model, mod = nb.load_theory('<name>')            # loads theories/<name>.theory.py -> (model dict, module)
  cfg = nb.Config(k=, max_ell=, external_fields=, fundamental=, tau_max=, tau_step=, spatial_grid=, spatial_points=, dyson_order=, show_orders=, logy=, figsize=)
  res = nb.run(model, cfg, mod)                     # calls compute_cumulants with those inputs; returns the SAME result dict compute_cumulants returns, plus res['_cfg'], res['_model'], res['_resolved'] (k, max_ell, external_fields, fundamental)
  nb.plot_cumulant(res, cfg, model, sim={'tau'|'x':..., 'C':..., 'C_err':...})   # adaptable overlay
nb.run returns exactly what compute_cumulants returns (res['C_tau'], res['C_tau_by_ell'], res['tau_grid'], res['C_tau_x'], res['spatial_grid'], res['mf'], res['params'], res['spatial_info'], res['diagrams'], ...), so downstream cells keep working if you alias the names they use.

THE MIGRATION (minimal, localized):
1. Read the whole notebook (Read shows cells) + notebooks/nb_support.py. Identify: the cell(s) that ACQUIRE the model (inline *TheoryBuilder build, OR importlib/exec of a theory file, OR a package import), the cell that calls compute_cumulants (note its EXACT args: k, max_ell, external_fields, fundamental, tau_max/tau_step or spatial_grid/spatial_points, dyson, parallel, use_cache), the simulator cell(s), the diagnostic cell(s), and the final theory-vs-sim plot cell.
2. If the model is built INLINE (or imported from a package) and NO theories/<name>.theory.py covers it: CREATE theories/<name>.theory.py with a build() that reproduces the inline build VERBATIM (same builder calls/params/action/equations/boundary/initial), plus DEFAULT_FUNDAMENTAL (the numeric params the notebook ran at) and METADATA (k_default, ell_default, recommended_external_fields using the model's ACTUAL field names, and tau/spatial grid defaults). Verify with: sage -python -c "import sys;sys.path.insert(0,'.');sys.path.insert(0,'notebooks');import nb_support as nb;m,_=nb.load_theory('<name>');print(nb.field_names(m), m['name'])". If a theory file ALREADY exists for this model (the audit may name it), just use it.
3. Replace the model-acquisition cell + the compute_cumulants cell with:
     import nb_support as nb            # (in the setup cell if not already imported; ensure notebooks/ is on sys.path)
     model, mod = nb.load_theory('<name>')
     cfg = nb.Config(<the SAME k, max_ell, external_fields, fundamental, grids, dyson as the original call>)
     <result_var> = nb.run(model, cfg, mod)
   Then add a SHIM re-exporting the exact legacy variable names the downstream sim/diagnostic/plot cells reference (e.g. result=..., th=..., C_by_ell=res['C_tau_by_ell'], tau_grid_th=res['tau_grid'], C_theory_total=res['C_tau'], fundamental=res['_resolved']['fundamental'], xstar=res['mf'][<field>], model=model, k=, max_ell=, external_fields=res['_resolved']['external_fields']). Grep the notebook for which names are used; alias exactly those. PRESERVE the SAME config values so the numerics are identical.
4. IMPORTANT — preserve the field-name convention: theory files use d-prefixed fluctuation names (e.g. 'dh','dx','dphi'). If the original used external_fields like [('h',1),('h',1)] but the loaded model's fields are ['dh'], either pass [('dh',1),('dh',1)] or omit external_fields (nb.run auto-builds k copies of the first field). Verify the cumulant value is unchanged.
5. Keep the SIMULATOR cell(s) and DIAGNOSTIC cell(s) UNCHANGED, except: if a downstream cell errors on a stale result/spatial_info key that the current pipeline no longer produces (e.g. spatial_info['self_energy_coeff_g'], ['bubble']) or a Sage list->Symbolic coercion, make the MINIMAL fix (guard the key with .get(...), drop the dead diagnostic line, or coerce the type). Note every such fix.
6. The final theory-vs-sim plot: you MAY switch the primary C-overlay to nb.plot_cumulant(res, cfg, model, sim={...}) built from the existing sim arrays, BUT keep any additional validated panels (rate bar charts, residuals, v_infinity checks, log-axis, Hartree comparison) — either as extra cells or by leaving the original plot cell and ADDING nothing if it already works. Do not lose information. If unsure, KEEP the original plot cell as-is (it will still work via the shim) and do NOT switch to plot_cumulant.
7. EXECUTE the migrated notebook to verify: \`CELL_TIMEOUT=600 sage -python scratch/exec_nb.py <path>\` (this writes the executed notebook back on success). It MUST run with 0 errors and produce its figures. Confirm the representative cumulant value (e.g. C(tau=0) or C(0,0) or the rate) matches the pre-migration physics (same config => same value). If a cell HANGS past the timeout (some configs have a known Phase J hang, e.g. cross-correlated 2-D OU), do NOT keep retrying — restore the notebook with \`git checkout -- <path>\` and set status 'failed' with the hang noted.

GUARDRAILS:
- Be conservative. If a notebook is a pure mean-field demo with NO simulator and NO theory-vs-sim plot, or its run hinges on bespoke kwargs not in Config (fixed_point_index, seed_box, stability), do the MINIMAL change (swap only the model-load line to nb.load_theory if a file exists) or set status 'left_as_is' with a clear reason — do NOT force-fit it.
- NEVER weaken the physics: same k, max_ell, external_fields, fundamental, grids. Same simulator settings.
- If you cannot get the notebook to execute cleanly after a reasonable effort, restore it (\`git checkout -- <path>\`) and set status 'failed' with the error.
- All Python runs via \`sage -python\`. Work from the repo root.

Return the structured verdict ONLY.`

const results = await parallel(items.map((it) => () =>
  agent(
    `${PREAMBLE}

TARGET NOTEBOOK: ${it.path}
Audit hints — group=${it.group}; theory_name(if a file exists)=${it.theory_name}; simulator=${it.sim}; compute_cumulants cell idx=${it.cc}; plot cell idx=${it.plot}; sim_kind=${it.simkind}.
Extra notes from the audit: ${it.notes}`,
    { label: `migrate:${it.path.split('/').pop().replace('pipeline_','').replace('_sim_compare.ipynb','').replace('.ipynb','')}`,
      phase: 'Migrate', schema: VERDICT, agentType: 'general-purpose' }
  ).then(v => v ? { ...v, path: v.path || it.path } : { path: it.path, status: 'failed', exec_ok: false, exec_secs: null, theory_file: null, changed: '', c_check: '', notes: 'agent returned null' })
))

const ok = results.filter(Boolean)
return {
  total: items.length,
  by_status: ok.reduce((m, r) => { m[r.status] = (m[r.status]||0)+1; return m }, {}),
  exec_ok: ok.filter(r => r.exec_ok).length,
  failed: ok.filter(r => r.status === 'failed' || !r.exec_ok).map(r => ({ path: r.path, notes: r.notes })),
  results: ok,
}
