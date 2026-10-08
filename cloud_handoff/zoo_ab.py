"""Template for the lever A/B on one zoo entry: levers OFF then ON in one process, np.array_equal on every array.
usage (repo root): PYTHONHASHSEED=0 python cloud_handoff/zoo_ab.py ENTRY FLAG[,FLAG...]
  FLAG = a module-level bool in engine.integration.time_domain.final_integral (or another module given as mod:FLAG),
  e.g. USE_SETUP_ZERO_EXIT.  Edit CLEAR() to clear whatever per-run state your levers keep.  Warm the disk cache first
  (run the entry once and discard it), or the cold/warm difference (~1e-20) looks like a failure."""
import sys, os, json, time, warnings, importlib
warnings.simplefilter('ignore')
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT); os.chdir(ROOT)
import numpy as np
import tests.tools.phase_j_zoo_baseline as Z
FI = 'engine.integration.time_domain.final_integral'
flags = []
for f in sys.argv[2].split(','):
    mod, _, name = f.rpartition(':')
    flags.append((importlib.import_module(mod or FI), name))
def CLEAR():
    pass        # e.g. clear module memo tables your levers add
name = sys.argv[1]; entry = Z.entry_by_name(name); res = {}
for mode in (False, True):
    for m, n in flags: setattr(m, n, mode)
    CLEAR(); t0 = time.perf_counter()
    rec = Z.run_entry(entry, with_provenance=False)
    res[mode] = dict(status=rec['status'], wall=time.perf_counter() - t0, counters=rec['counters'],
                     arrays=Z.result_arrays(rec) if rec['status'] == 'ok' else {})
off, on = res[False], res[True]
ok = off['status'] == on['status'] == 'ok'
eq = ok and set(off['arrays']) == set(on['arrays']) and all(np.array_equal(off['arrays'][k], on['arrays'][k]) for k in off['arrays'])
line = dict(entry=name, flags=sys.argv[2], status=(off['status'], on['status']), bit_identical=bool(eq),
            wall_off=round(off['wall'], 2), wall_on=round(on['wall'], 2), counters_on=on['counters'])
print(json.dumps(line), flush=True)
os.makedirs(os.path.join(ROOT, 'cloud_handoff', 'out'), exist_ok=True)
open(os.path.join(ROOT, 'cloud_handoff', 'out', 'zoo_identity_m5.jsonl'), 'a').write(json.dumps(line) + '\n')
