"""
tests/test_dt_first_order.py
============================
Pins the first-order-in-time guard (``engine/core/dt_order.py``).

The framework models Itô SDEs/SPDEs/DAEs, which must be FIRST ORDER in
the time-derivative symbol ``Dt``.  A residual of ``Dt``-degree ≥ 2 used
to be **silently dropped**: the linearization pencil takes ``B`` at
``Dt = 0`` and ``A = ∂²F/∂Dt∂x`` also at ``Dt = 0``, so a ``Dt²`` term
survives neither — it vanished with no warning and the run returned a
confidently wrong answer.  (In the propagator the same term became
``δ′(t)²``, whose Fourier image Sage renders with ``dirac_delta(0)``.)

What is pinned here:
  * a model nonlinear in ``Dt`` raises ``NonlinearDtError``, from BOTH
    the DAE-equation path and the action path, with the offending
    equation / field / monomial named in the message;
  * a plain first-order model is untouched (no false positive) and
    still produces the right eigenvalue;
  * a first-order DAE with a genuine ALGEBRAIC constraint row (zero
    ``Dt``-degree) is NOT rejected — that is legitimate, supported
    input, and is the false positive that would hurt most.

Run:  sage -python -m pytest tests/test_dt_first_order.py -q
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from api.model import TemporalModelBuilder            # noqa: E402
from api._mean_field_dae import (                     # noqa: E402
    linear_stability, solve_mean_field_dae,
)
from engine.core.dt_order import (                    # noqa: E402
    NonlinearDtError, dt_degree_exceeds_one,
)
from engine.core.field_theory import FieldTheory      # noqa: E402


# ── Model fixtures ───────────────────────────────────────────────────

def _damped_oscillator():
    """``m·ẍ + ẋ + μ·x = 0`` — NOT first order in Dt (the bug's shape)."""
    return (TemporalModelBuilder('dt2 damped oscillator')
            .physical_field('x')
            .parameter('m',  default=1.0, domain='positive')
            .parameter('mu', default=1.0, domain='positive')
            .parameter('T',  default=1.0, domain='positive')
            .set_action_text('xt*((m*Dt^2 + Dt + mu)*x) - T*xt^2')
            .equation(lhs='(m*Dt^2 + Dt + mu)*x', rhs='0')
            .build())


def _ou_first_order():
    """Plain first-order OU + cubic — must be completely unaffected."""
    return (TemporalModelBuilder('dt1 ou quartic')
            .physical_field('x')
            .parameter('mu',  default=1.0, domain='positive')
            .parameter('eps', default=0.02, domain='positive')
            .parameter('T',   default=1.0, domain='positive')
            .set_action_text('xt*((Dt+mu)*x + eps*x^3) - T*xt^2')
            .equation(lhs='(Dt+mu)*x', rhs='-eps*x^3')
            .build())


def _dae_with_algebraic_row():
    """A first-order DAE: one differential row PLUS one purely algebraic
    row (``n = a·v``, no ``Dt`` at all).  Zero ``Dt``-degree on that row
    is legitimate — the pencil simply gets a zero row in ``A`` and an
    infinite eigenvalue, which ``linear_stability`` filters out."""
    return (TemporalModelBuilder('first-order dae with constraint')
            .physical_field('v')
            .physical_field('n')
            .parameter('tau', default=10.0, domain='positive')
            .parameter('Em',  default=0.7,  domain='real')
            .parameter('w',   default=0.25, domain='positive')
            .parameter('a',   default=0.37, domain='positive')
            .parameter('T',   default=1.0,  domain='positive')
            .set_action_text('vt*((tau*Dt + 1)*v - Em - w*n) '
                             '+ nt*(n - a*v) - T*vt^2')
            .equation(lhs='(tau*Dt + 1)*v', rhs='Em + w*n')   # differential
            .equation(lhs='n',              rhs='a*v')        # ALGEBRAIC
            .build())


_OSC_FUND = {'m': 1.0, 'mu': 1.0, 'T': 1.0}
_OU_FUND  = {'mu': 1.0, 'eps': 0.02, 'T': 1.0}
_DAE_FUND = {'tau': 10.0, 'Em': 0.7, 'w': 0.25, 'a': 0.37, 'T': 1.0}


# ── The predicate itself ─────────────────────────────────────────────

def test_dt_degree_predicate():
    """``dt_degree_exceeds_one`` accepts every linear-in-Dt shape the
    repo's models actually use, and rejects genuine ``Dt``-nonlinearity
    (including non-polynomial dependence, which ``Poly(...).degree()``
    could not even be asked about)."""
    from sage.all import SR, exp
    Dt = SR.var('Dt')
    x, tau, mu, lap = SR.var('x'), SR.var('tau'), SR.var('mu'), SR.var('lap')

    for ok in [SR(0), SR(1), mu * x, Dt, (tau * Dt + 1) * x,
               (Dt + mu - lap) * x, Dt * lap * x, Dt * x - Dt * x]:
        assert not dt_degree_exceeds_one(ok, Dt), f'false positive on {ok}'

    for bad in [Dt ** 2, (Dt * x) ** 2, Dt ** 2 * x + Dt * x + mu * x,
                Dt * (Dt * x + x), exp(Dt) * x, Dt ** 3 * x]:
        assert dt_degree_exceeds_one(bad, Dt), f'missed {bad}'


# ── Rejection: nonlinear in Dt ───────────────────────────────────────

def test_dt_squared_equation_is_rejected_by_linearization():
    """The named bug: the pencil silently dropped ``m·Dt²·x`` (it fell
    out of both A and B) and reported a single stable eigenvalue for a
    second-order system.  Now it raises, naming equation and field."""
    model = _damped_oscillator()
    root = {'xstar': [0.0]}
    with pytest.raises(NonlinearDtError) as excinfo:
        linear_stability(model, _OSC_FUND, root)
    msg = str(excinfo.value)
    # Names the offending equation and the field it acts on ...
    assert "(m*Dt^2 + Dt + mu)*x" in msg
    assert 'equation #1' in msg
    assert 'field(s) x' in msg
    # ... and states the remedy, in first-order form.
    assert 'FIRST ORDER' in msg
    assert 'Dt*x' in msg and "rhs='v'" in msg
    assert 'costs no generality' in msg


def test_dt_squared_action_is_rejected_at_expand():
    """The action path (``set_action_text``) is guarded too — otherwise
    the propagator would map ``Dt²`` to ``δ′(t)²`` and hand back a
    kernel full of ``dirac_delta(0)``."""
    ft = FieldTheory(_damped_oscillator(), taylor_order=4)
    with pytest.raises(NonlinearDtError) as excinfo:
        ft.expand()
    msg = str(excinfo.value)
    assert 'action sector' in msg
    assert 'set_action_text' in msg
    assert 'FIRST ORDER' in msg


def test_dt_squared_not_swallowed_by_stability_classification():
    """``solve_mean_field_dae`` wraps ``linear_stability`` in a broad
    ``except Exception`` that demotes a failed root to "unstable".  The
    Dt guard must escape that net, or the user gets a misleading "no
    stable root" instead of the actionable message."""
    model = _damped_oscillator()
    model['stability_analysis'] = True
    with pytest.raises(NonlinearDtError):
        solve_mean_field_dae(model, _OSC_FUND, n_starts=4)


# ── No false positives ───────────────────────────────────────────────

def test_first_order_model_is_unaffected():
    """A plain first-order model still solves, still linearizes, and
    still gives σ = -mu at the x*=0 saddle."""
    model = _ou_first_order()
    res = solve_mean_field_dae(model, _OU_FUND, n_starts=16)
    root = res['mf_values']
    assert np.allclose(root['xstar'], [0.0], atol=1e-7)

    stab = linear_stability(model, _OU_FUND, root)
    assert np.allclose(stab['A'], [[1.0]])
    assert np.allclose(stab['B'], [[_OU_FUND['mu']]])
    assert stab['stable'] is True
    assert np.allclose(stab['eigenvalues_finite'], [-_OU_FUND['mu']])

    # And the action path is untouched.
    ft = FieldTheory(model, taylor_order=4)
    ft.expand()
    assert ft.free_action() != ft.ring().zero()


def test_algebraic_constraint_row_is_not_rejected():
    """THE false positive that would hurt most: a first-order DAE with a
    row carrying NO ``Dt`` at all.  Zero Dt-degree is legitimate — it is
    an algebraic constraint, gives a zero row in ``A``, and contributes
    an infinite (filtered) eigenvalue."""
    model = _dae_with_algebraic_row()
    res = solve_mean_field_dae(model, _DAE_FUND, n_starts=32)
    root = res['mf_values']

    # v = Em / (1 - w·a),  n = a·v
    v_expected = _DAE_FUND['Em'] / (1.0 - _DAE_FUND['w'] * _DAE_FUND['a'])
    assert np.allclose(root['vstar'], [v_expected], atol=1e-8)
    assert np.allclose(root['nstar'], [_DAE_FUND['a'] * v_expected],
                       atol=1e-8)

    stab = linear_stability(model, _DAE_FUND, root)
    # Differential row keeps tau in A; the algebraic row's A row is zero.
    A = np.asarray(stab['A'])
    assert np.count_nonzero(A) == 1
    assert np.isclose(A.max(), _DAE_FUND['tau'])
    zero_rows = [k for k in range(A.shape[0]) if not np.any(A[k])]
    assert len(zero_rows) == 1, 'the algebraic row must give a zero A row'

    # One finite mode survives: sigma = -(1 - w·a)/tau.
    finite = np.asarray(stab['eigenvalues_finite'])
    assert finite.size == 1
    sigma_expected = -(1.0 - _DAE_FUND['w'] * _DAE_FUND['a']) / _DAE_FUND['tau']
    assert np.isclose(complex(finite[0]).real, sigma_expected, atol=1e-9)
    assert stab['stable'] is True
