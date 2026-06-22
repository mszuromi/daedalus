"""Programmatic geometry audit over ALL contributing prediagrams: flag node-node
overlaps (the layout's main remaining failure mode).  Topology is field-agnostic,
so ou_quartic + ou_sextic over k+ℓ≤4 cover the whole topology space.
    sage -python scratch/_audit_geom.py
"""
import sys, os
sys.path.insert(0, os.path.abspath('.')); sys.path.insert(0, os.path.abspath('notebooks'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import daedalus as dd
from msrjd.diagrams.prediagram_plot import contributing_prediagrams, _generic_labels, topo_signature, sig_label

CASES = [('ou_quartic', 2, 2), ('ou_quartic', 3, 1), ('ou_quartic', 4, 0),
         ('ou_sextic', 2, 2), ('ou_sextic', 3, 1)]
OVERLAP = 0.40   # node "radius" ~0.16; centres closer than this visually touch

worst = []
n_total = 0
for theory, k, ell in CASES:
    model, _ = dd.load_theory(theory)
    pre = contributing_prediagrams(model, k, ell)
    for e in sorted(pre):
        for idx, pd in enumerate(pre[e]):
            labels = _generic_labels(pd[0], pd[2])
            pos = labels[0]
            vs = list(pos)
            mind = np.inf
            for a in range(len(vs)):
                for b in range(a + 1, len(vs)):
                    d = ((pos[vs[a]][0] - pos[vs[b]][0]) ** 2 + (pos[vs[a]][1] - pos[vs[b]][1]) ** 2) ** 0.5
                    mind = min(mind, d)
            n_total += 1
            tag = '%s k%d ℓ%d %s #%d' % (theory, k, e, sig_label(topo_signature(pd[0], pd[2])), idx + 1)
            worst.append((mind, tag))

worst.sort()
print('scanned %d diagrams across %d cases' % (n_total, len(CASES)))
print('min node-node separation (smallest first):')
for mind, tag in worst[:12]:
    flag = '  <-- OVERLAP' if mind < OVERLAP else ''
    print('   %.3f  %s%s' % (mind, tag, flag))
n_bad = sum(1 for m, _ in worst if m < OVERLAP)
print('diagrams with node overlap (< %.2f): %d' % (OVERLAP, n_bad))
