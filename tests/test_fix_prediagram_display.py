"""Regression: dd.plot_prediagrams must not double-render in a Jupyter notebook.

A bare `dd.plot_prediagrams(...)` call returns a Matplotlib Figure, which Jupyter
auto-displays as the cell's return value.  If that figure is ALSO left open in
pyplot's registry, the inline backend auto-shows it at cell end too -> the plot
renders TWICE.  The fix closes the figure (drops it from the registry) before
returning, so only the returned fig renders, once.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import matplotlib                                          # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt                            # noqa: E402
from matplotlib.figure import Figure                      # noqa: E402
import daedalus as dd                                      # noqa: E402


def test_plot_prediagrams_returns_closed_figure():
    model, _ = dd.load_model('ou_quartic')
    n_before = len(plt.get_fignums())
    fig = dd.plot_prediagrams(model, k=2, max_ell=1)
    assert isinstance(fig, Figure)
    # The returned figure must NOT remain open in pyplot's registry — otherwise the
    # inline backend auto-shows it in addition to Jupyter rendering the return value.
    assert not plt.fignum_exists(fig.number), 'figure left open -> would double-render'
    assert len(plt.get_fignums()) == n_before, 'plot_prediagrams leaked an open figure'
