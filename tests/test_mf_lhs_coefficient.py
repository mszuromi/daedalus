"""
tests/test_mf_lhs_coefficient.py
================================
``ModelBuilder.equation(lhs, rhs)`` derives the legacy saddle text from
``LHS|_{Dt=0} = RHS``.  The derivation must honour the coefficient of the
field on the LHS and the algebraic remainder, and must refuse a pure time
derivative (whose saddle condition ``0 = RHS`` defines no ``xstar``) with a
message that says what to do — instead of deriving ``x* = RHS`` and letting
the tadpole sanity check fail downstream with no hint.  The three-field
models below are a plain immigration–death process written with explicit
flux fields: an input event train n (rate f), a decay flux u (rate r*m) and
a direct loss d*m, so the exact saddle is m* = f/(r+d), u* = r*m*.

Run:  sage -python -m pytest tests/test_mf_lhs_coefficient.py -q
"""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from api.model import TemporalModelBuilder


def _derived(b):
    b._auto_populate_mf_eqs_from_equations()
    return dict(b._mf_eqs_text)


def _ou(lhs, rhs):
    return (TemporalModelBuilder('ou').population('pop', size=1)
            .physical_field('x', population='pop')
            .parameter('mu', default=2.0).parameter('eps', default=0.02)
            .parameter('D', default=1.0)
            .set_action_text('sum(xt[i]*((Dt+mu)*x[i] + eps*x[i]^3) - D*xt[i]^2 for i in pop)')
            .equation(lhs=lhs, rhs=rhs, population='pop'))


def test_unit_coefficient_is_unchanged():
    d = _derived(_ou('(Dt+1)*x[i]', '-eps*x[i]^3'))
    assert d['xstar'] == '-eps*xstar[i]^3'


def test_coefficient_is_divided_out():
    d = _derived(_ou('(Dt+mu)*x[i]', '-eps*x[i]^3'))
    assert d['xstar'] == '((-eps*xstar[i]^3) - (0))/(mu)'


_FLUX_ACTION = ('nt*n - (exp(nt)-1)*f + ut*u - (exp(ut)-1)*r*m'
                ' + mt*(Dt*m + u - n) - (exp(-mt)-1)*d*m')


def test_algebraic_remainder_moves_to_rhs():
    b = (TemporalModelBuilder('flux-pop').population('pop', size=1)
         .physical_field('n', population='pop').physical_field('u', population='pop')
         .physical_field('m', population='pop')
         .parameter('f', default=0.5).parameter('r', default=0.3).parameter('d', default=0.1)
         .set_action_text('''sum( nt[i]*n[i] - (exp(nt[i])-1)*f
             + ut[i]*u[i] - (exp(ut[i])-1)*r*m[i]
             + mt[i]*(Dt*m[i] + u[i] - n[i]) - (exp(-mt[i])-1)*d*m[i] for i in pop)''')
         .equation(lhs='n[i]', rhs='f', population='pop')
         .equation(lhs='u[i]', rhs='r*m[i]', population='pop')
         .equation(lhs='Dt*m[i] + u[i]', rhs='n[i] - d*m[i]', population='pop'))
    d = _derived(b)
    assert d['nstar'] == 'f'
    # the third equation is linear in u with unit coefficient and no remainder
    assert d['ustar'] == 'nstar[i] - d*mstar[i]'
    # and the whole model builds and reproduces the analytic saddle
    import numpy as np
    import daedalus as dd
    model = b.build()
    cfg = dd.Config(k=2, max_ell=0, external_fields=[('dn', 1), ('dn', 1)],
                    tau_grid=(0.0, 2.0, 3), parallel=False)
    res = dd.run(model, cfg, None)
    mf = res.get('mf_values') or res.get('mf')
    f, r, d_ = 0.5, 0.3, 0.1
    m_exact = f / (r + d_)
    assert np.isclose(np.ravel(mf['nstar'])[0], f)
    assert np.isclose(np.ravel(mf['mstar'])[0], m_exact)
    assert np.isclose(np.ravel(mf['ustar'])[0], r * m_exact)


def test_pure_time_derivative_is_refused_with_guidance():
    b = (TemporalModelBuilder('pure').population('pop', size=1)
         .physical_field('u', population='pop').physical_field('m', population='pop')
         .parameter('a', default=0.5).parameter('r', default=0.4)
         .set_action_text('sum(ut[i]*u[i] - (exp(ut[i])-1)*r*m[i] + mt[i]*(Dt*m[i] + u[i]) - (exp(mt[i])-1)*a for i in pop)')
         .equation(lhs='u[i]', rhs='r*m[i]', population='pop')
         .equation(lhs='Dt*m[i]', rhs='a - u[i]', population='pop'))
    with pytest.raises(ValueError) as ei:
        b.build()
    msg = str(ei.value)
    assert 'vanishes at Dt=0' in msg and 'set_mf_equation' in msg and "lhs='Dt*m + u'" in msg


def test_scalar_form_without_population():
    b = (TemporalModelBuilder('scalar').physical_field('x')
         .parameter('mu', default=3.0).parameter('eps', default=0.1).parameter('D', default=1.0)
         .set_action_text('xt*((Dt+mu)*x + eps*x^3) - D*xt^2')
         .equation(lhs='(Dt+mu)*x', rhs='-eps*x^3'))
    d = _derived(b)
    assert d['xstar'] == '((-eps*xstar[i]^3) - (0))/(mu)'


def test_residual_form_as_entered_in_the_ui():
    """The UI's MF tab is 'LHS − RHS = 0'; entering the residuals verbatim with
    rhs='0' must define every saddle correctly."""
    b = (TemporalModelBuilder('flux-residual')
         .physical_field('n').physical_field('u').physical_field('m')
         .parameter('f', default=0.5).parameter('r', default=0.3).parameter('d', default=0.1)
         .set_action_text(_FLUX_ACTION)
         .equation(lhs='n-f', rhs='0')
         .equation(lhs='u-r*m', rhs='0')
         .equation(lhs='Dt*m + u + d*m - n', rhs='0'))
    d = _derived(b)
    assert set(d) == {'nstar', 'ustar', 'mstar'}
    assert d['nstar'] == '((0) - (-f))/(1)'
    assert d['ustar'] == '((0) - (-mstar[i]*r))/(1)'
    assert d['mstar'] == '((0) - (-nstar[i] + ustar[i]))/(d)'
    import numpy as np
    import daedalus as dd
    res = dd.run(b.build(), dd.Config(k=2, max_ell=0, external_fields=[('dm', 1), ('dm', 1)],
                                      tau_grid=(0.0, 2.0, 3), parallel=False), None)
    mf = res['mf_values']
    assert np.isclose(np.ravel(mf['mstar'])[0], 1.25) and np.isclose(np.ravel(mf['ustar'])[0], 0.375)
    # immigration–death is Poisson: Var = m*, C(tau) = m* exp(-(r+d)|tau|); the
    # tau=0 grid point is the Ito left limit tau = -_ITO_EPS
    from api.compute import _ITO_EPS
    assert np.isclose(float(np.real(res['C_tau'][0])), 1.25 * np.exp(-0.4 * _ITO_EPS), rtol=1e-9)
    assert np.isclose(float(np.real(res['C_tau'][1])), 1.25 * np.exp(-0.4), rtol=1e-9)


def test_unparseable_lhs_warns_instead_of_silently_guessing():
    """An LHS the symbolic split cannot read falls back to 'x* = RHS' — with a
    warning naming the equation, not silently."""
    import warnings
    b = (TemporalModelBuilder('conv').population('E', size=1)
         .physical_field('v', population='E')
         .parameter('tau', default=[1.0], indexed_by=['E']).parameter('Em', default=[1.0], indexed_by=['E'])
         .parameter('w', default=[[0.5]], indexed_by=['E', 'E'])
         .define_kernel('g', time_expr='exp(-t)*heaviside(t)', latex_name='g', indexed_by=['E', 'E'])
         .set_action_text('sum(vt[i]*((tau[i]*Dt + 1)*v[i] - Em[i] - sum(w[i,j]*Conv(g[i,j], v[j]) for j in E)) for i in E)')
         .equation(lhs='(tau[i]*Dt + 1)*v[i] - sum(w[i,j]*v[j] for j in E)', rhs='Em[i]', population='E'))
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter('always')
        d = _derived(b)
    assert d['vstar'] == 'Em[i]'
    assert any('could not read the coefficient' in str(w.message) for w in rec)
