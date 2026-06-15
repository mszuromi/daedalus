"""Merge the audit descriptors + the execution-sweep results into the
migration workflow's per-notebook args. Emits scratch/migrate_args.json
(list) and prints a tiering summary.  Run with `sage -python`."""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
descs = json.load(open(os.path.join(ROOT, 'scratch/audit_descriptors.json')))

# parse sweep log -> {basename: (ok, error)}
sweep = {}
log = os.path.join(ROOT, 'scratch/exec_sweep.log')
if os.path.isfile(log):
    for line in open(log):
        if line.startswith('RESULT '):
            r = json.loads(line[7:])
            sweep[os.path.basename(r['path'])] = (r['ok'], r['error'])

items = []
for d in descs:
    base = os.path.basename(d['path'])
    ok, err = sweep.get(base, (None, None))
    inline = d['theory_name'] is None
    # wave 1 = needs care: currently broken OR built inline (theory extraction)
    wave = 1 if (ok is False or inline) else 2
    items.append({
        'path': d['path'],
        'theory_name': d['theory_name'],
        'group': d['group'],
        'sim': f"{d['simulator']['module']} :: {d['simulator']['funcs']}",
        'simkind': d['simulator']['sim_dict_kind'],
        'cc': d['cell_roles']['theory_compute_cell_idx'],
        'plot': d['cell_roles']['plot_cell_idx'],
        'currently_ok': ok,
        'currently_error': err,
        'inline_build': inline,
        'migratability': d['migratability'],
        'notes': (d['notes'] or '')[:900],
        'wave': wave,
    })

json.dump(items, open(os.path.join(ROOT, 'scratch/migrate_args.json'), 'w'), indent=1)

w1 = [i for i in items if i['wave'] == 1]
w2 = [i for i in items if i['wave'] == 2]
print(f'wrote {len(items)} items -> scratch/migrate_args.json')
print(f'\nWAVE 1 ({len(w1)}) — broken or inline (need theory extraction / fixes):')
for i in w1:
    flag = 'BROKEN' if i['currently_ok'] is False else 'inline'
    print(f"  [{flag:6}] {os.path.basename(i['path']):46s} theory={i['theory_name']}")
print(f'\nWAVE 2 ({len(w2)}) — working, theory-file-backed (localized swap):')
for i in w2:
    print(f"  {os.path.basename(i['path']):46s} theory={i['theory_name']}  ok={i['currently_ok']}")
