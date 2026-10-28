"""Shared figure style for the worked examples (figures/examples/*).

Line colours follow the paper: tree black, one loop blue, two loops orange, simulation dark grey markers.
Panel labels (a), (b), ... sit in the upper-left corner of each panel."""
import os, sys
import matplotlib
import matplotlib.pyplot as plt

C_TREE, C_L1, C_L2, C_SIM, C_MF = '#000000', '#1f4fd8', '#E08A00', '#222222', '#888888'
ORDER_STYLE = {0: (C_TREE, '-', 'tree'), 1: (C_L1, '--', 'one loop'), 2: (C_L2, '-.', 'two loops')}


def apply():
    """Matplotlib defaults used by every example figure."""
    plt.rcParams.update({'font.size': 13.0, 'axes.labelsize': 15.0, 'axes.linewidth': 1.0,
                         'mathtext.fontset': 'dejavusans', 'axes.spines.top': False, 'axes.spines.right': False,
                         'savefig.bbox': 'tight'})


def panel_labels(axes, x=-0.28, y=1.04):
    for ax, lab in zip(axes, 'abcdefgh'):
        ax.text(float(x), float(y), f'({lab})', transform=ax.transAxes, fontsize=15.0, fontweight='bold', va='bottom')


def repo_root(start=None):
    """Walk up from `start` (default: cwd) to the directory that holds the daedalus package; put it on sys.path."""
    d = os.path.abspath(start or os.getcwd())
    while d != os.path.dirname(d) and not os.path.isfile(os.path.join(d, 'daedalus', '__init__.py')):
        d = os.path.dirname(d)
    if d not in sys.path:
        sys.path.insert(0, d)
    return d
