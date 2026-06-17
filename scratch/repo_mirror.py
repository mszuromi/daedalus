import json, re, glob, os
def fix(s):
    s = s.replace('import nb_support as nb', 'import daedalus as dd')
    s = s.replace('nb_support', 'daedalus')
    s = re.sub(r'\bnb\.', 'dd.', s)
    return s
# 1) notebooks (JSON): sweep every cell's source
nbs = (glob.glob('notebooks/**/*.ipynb', recursive=True))
nch = 0
for p in nbs:
    o = json.load(open(p)); changed = False
    for c in o.get('cells', []):
        src = ''.join(c.get('source', []))
        if 'nb_support' in src or re.search(r'\bnb\.', src):
            new = fix(src)
            if new != src:
                c['source'] = new.splitlines(keepends=True); changed = True
    if changed:
        json.dump(o, open(p, 'w'), indent=1); nch += 1
print(f'notebooks updated: {nch}/{len(nbs)}')
# 2) text files: module, test, generators, README
texts = ['notebooks/daedalus.py', 'tests/test_daedalus.py', 'notebooks/README.md',
         'scratch/gen_simcompare.py', 'scratch/gen_simcompare_spatial.py',
         'scratch/gen_templates.py', 'scratch/gen_runner.py']
for p in texts:
    if not os.path.exists(p): continue
    s = open(p).read(); new = fix(s)
    if new != s:
        open(p, 'w').write(new); print('text updated:', p)
print('done')
