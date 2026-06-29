"""plot_temporal(..., residual=True) adds a companion residual panel.

The residual panel plots C(τ) − C_tree(τ): the cumulative loop contributions
(1-loop, 1+2-loop, …) against the simulated residual sim − tree, so a loop that
is a tiny fraction of the tree becomes the whole signal.  It must appear only
when a sim, a tree (ℓ=0) and at least one loop order are all present.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import matplotlib                                          # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt                            # noqa: E402
import numpy as np                                         # noqa: E402
import daedalus as dd                                      # noqa: E402


def _result(max_ell=2):
    """A minimal temporal k=2 C(τ) result (tree + 1-loop [+ 2-loop]) + a sim."""
    tau = np.linspace(-3.0, 3.0, 25)
    env = np.exp(-np.abs(tau))
    by_ell = {0: 0.05 * env, 1: 0.002 * env}
    if max_ell >= 2:
        by_ell[2] = 0.0003 * env
    total = sum(by_ell.values())
    res = {'tau_grid': tau, 'C_tau': total, 'C_tau_by_ell': by_ell}
    sim = {'tau': tau, 'C': total, 'C_err': np.full_like(tau, 1e-4)}
    return res, sim


def test_residual_adds_companion_panel():
    res, sim = _result(max_ell=2)
    fig = dd.plot_temporal(res, dd.Config(), {'name': 't'}, sim=sim, residual=True)
    assert len(fig.axes) == 2, 'residual=True should add a companion panel'
    # the residual panel should carry a 1-loop AND a 1+2-loop theory curve + sim−tree
    labels = [t.get_text() for t in fig.axes[1].get_legend().get_texts()]
    assert any('1-loop' in s for s in labels)
    assert any('1+2-loop' in s for s in labels)
    assert any('sim' in s for s in labels)
    plt.close(fig)


def test_residual_one_loop_only():
    res, sim = _result(max_ell=1)
    fig = dd.plot_temporal(res, dd.Config(), {'name': 't'}, sim=sim, residual=True)
    labels = [t.get_text() for t in fig.axes[1].get_legend().get_texts()]
    assert any('1-loop' in s for s in labels)
    assert not any('1+2-loop' in s for s in labels), 'no 2-loop curve when max_ell=1'
    plt.close(fig)


def test_default_is_single_panel():
    res, sim = _result(max_ell=2)
    fig = dd.plot_temporal(res, dd.Config(), {'name': 't'}, sim=sim)
    assert len(fig.axes) == 1, 'default (residual=False) must stay single-panel'
    plt.close(fig)


def test_residual_noop_without_sim():
    res, _ = _result(max_ell=2)
    fig = dd.plot_temporal(res, dd.Config(), {'name': 't'}, sim=None, residual=True)
    assert len(fig.axes) == 1, 'no sim → no residual panel'
    plt.close(fig)
