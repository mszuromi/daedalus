"""Render every contributing prediagram for a (theory, k, ell) as large panels,
paginated so each PNG is small enough to inspect.  Usage:
    sage -python scratch/_audit_render.py THEORY K ELL [per_page]
"""
import sys, os
sys.path.insert(0, os.path.abspath('.')); sys.path.insert(0, os.path.abspath('notebooks'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import daedalus as dd
from msrjd.diagrams.prediagram_plot import (contributing_prediagrams, draw_prediagram,
                                            topo_signature, sig_label)


def render(theory, k, ell_max, ncol=3, per_page=12, dpi=110):
    model, _ = dd.load_theory(theory)
    pre = contributing_prediagrams(model, k, ell_max)
    items = [(ell, i, pd) for ell in sorted(pre) for i, pd in enumerate(pre[ell])]
    outs = []
    for p, p0 in enumerate(range(0, len(items), per_page)):
        chunk = items[p0:p0 + per_page]
        nrow = int(np.ceil(len(chunk) / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(4.3 * ncol, 3.4 * nrow), squeeze=False)
        for ax in axes.flat:
            ax.axis('off')
        for j, (ell, idx, pd) in enumerate(chunk):
            ax = axes[j // ncol][j % ncol]
            draw_prediagram(pd[0], pd[2], ax,
                            title='ℓ%d·%s·#%d' % (ell, sig_label(topo_signature(pd[0], pd[2])), idx + 1))
        out = '/tmp/audit_%s_k%d_l%d_p%d.png' % (theory, k, ell_max, p)
        fig.suptitle('%s  k=%d ℓ≤%d  (page %d, %d of %d diagrams)'
                     % (theory, k, ell_max, p, len(chunk), len(items)), fontsize=11)
        fig.tight_layout(rect=[0, 0, 1, 0.98])
        fig.savefig(out, dpi=dpi, bbox_inches='tight')
        plt.close(fig)
        outs.append(out)
        print('wrote', out, '(%d panels)' % len(chunk))
    return outs


if __name__ == '__main__':
    a = sys.argv[1:]
    if a:
        render(a[0], int(a[1]), int(a[2]), per_page=int(a[3]) if len(a) > 3 else 12)
    else:
        render('ou_quartic', 2, 1)
