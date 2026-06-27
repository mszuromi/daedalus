"""Regression: the shipped ``linear_hawkes`` theory must LOAD and RUN at its own
defaults.

It used to load but fail on run with ``MF sector does not vanish at saddle`` —
the MF saddle-sector check could not verify numerically because ``E``/``w`` had
no per-parameter ``default=`` (only ``DEFAULT_FUNDAMENTAL`` did), so the verifier
fell back to all-ones (no real saddle) and the numerical residual came back N/A.
Adding the defaults (theories/linear_hawkes.theory.py) fixes it.  This guards
against re-dropping them.
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', 'notebooks')))

import daedalus as dd  # noqa: E402


def test_linear_hawkes_runs_at_defaults():
    m, mod = dd.load_theory('linear_hawkes')
    res = dd.run(m, dd.Config(k=2, max_ell=0, tau_grid=(-20.0, 20.0, 21)), mod)

    C = np.asarray(res['C_tau']).real
    assert np.all(np.isfinite(C)), 'linear_hawkes C(tau) must be finite'

    # The saddle must solve  vstar = E + w·nstar  (a=1, phi(v)=a*v ⇒ nstar=vstar).
    mf = res['mf_values']
    vstar = np.asarray(mf['vstar'], dtype=float)
    nstar = np.asarray(mf['nstar'], dtype=float)
    assert vstar.shape == (2,) and np.all(np.isfinite(vstar))
    E = np.array([0.78, 0.81])
    w = np.array([[0.30, 0.25], [0.30, 0.35]])
    assert np.allclose(vstar, E + w @ nstar, atol=1e-6)
