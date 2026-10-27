"""
tests/test_audit_wave3_kernel_slots.py
======================================
Audit wave 3c, conductance-kernel bookkeeping.

* ACC-03: one kernel symbol convolving two different fields (``Conv(g, x)``
  in y's equation and ``Conv(g, y)`` in x's) used to be attached to the
  first matching leg of the vertex, so the answer depended on the field
  declaration order (32 % off at one loop).  The Conv reducer now writes the
  interaction sectors a second time with one alias per (kernel, field)
  emission, and vertex extraction reads the legs from those.
* GEN-09: a kernel shared across a population (``w[i,j]*Conv(r, n[j])``,
  the README's Hawkes example) left 924 one-loop diagrams unevaluated
  ('3 kernel slots on the leg'); it now runs and equals the same model with
  an indexed kernel ``r[i,j]``.
* LOUD-08: a vertex coefficient with a plain and a kernel term whose kernel
  rate equals a pole of the line was refused (the dress-failure rerun split
  the term's own prefactor again).

References: Markov-embedded twins (the kernel conductance written as an
auxiliary linear field ``u = g * x``, ``(taug Dt + 1) u = x``) and the
indexed-kernel twin.

Run:  sage -python -m pytest tests/test_audit_wave3_kernel_slots.py -q
"""
from __future__ import annotations

import functools
import os
import sys
import warnings

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from daedalus.api import compute_cumulants                       # noqa: E402
from daedalus.api.model import TemporalModelBuilder              # noqa: E402
from daedalus.engine.core.vertices import ConvVertexType         # noqa: E402

TAU = np.array([0.0, 0.7, 2.5])

# --- ACC-03 ---------------------------------------------------------------
# yt*((Dt+nu) y - h - e1 y (g*x) - e2 x (g*y)) + xt*((Dt+mu) x - hx - c (g*y)):
# the monomial dy*dx gets BOTH orientations of the kernel.
_P3 = (('mu', 1.3), ('nu', 0.9), ('e1', 0.25), ('e2', 0.15), ('h', 0.5),
       ('hx', 0.4), ('c', 0.2), ('D', 0.4), ('taug', 1.7))


def _saddle3():
    p = dict(_P3)
    # x = (hx + c y)/mu ; nu y = h + (e1+e2) y x
    a = (p['e1'] + p['e2']) * p['c'] / p['mu']
    b = (p['e1'] + p['e2']) * p['hx'] / p['mu'] - p['nu']
    ys = (-b - np.sqrt(b * b - 4 * a * p['h'])) / (2 * a)
    return float((p["hx"] + p["c"] * ys) / p["mu"]), float(ys)


def _model3(order, embedded):
    xs, ys = _saddle3()
    b = TemporalModelBuilder('w3c two-field kernel %s %s'
                             % (''.join(order), 'emb' if embedded else 'conv'))
    b.population('P', size=1)
    for f in list(order) + (['u', 'w'] if embedded else []):
        b.physical_field(f, population='P')
    for p, d in _P3:
        b.parameter(p, default=d, domain='positive')
    b.set_mf_equation('xstar', repr(xs))
    b.set_mf_equation('ystar', repr(ys))
    if embedded:
        b.set_mf_equation('ustar', repr(xs))
        b.set_mf_equation('wstar', repr(ys))
        b.set_action_text(
            'sum(yt[i]*((Dt+nu)*y[i] - h - e1*y[i]*u[i] - e2*x[i]*w[i]) '
            '+ xt[i]*((Dt+mu)*x[i] - hx - c*w[i]) '
            '+ ut[i]*((taug*Dt+1)*u[i] - x[i]) + wt[i]*((taug*Dt+1)*w[i] - y[i]) '
            '- D*xt[i]^2 - D*yt[i]^2 for i in P)')
    else:
        b.define_kernel('g', time_expr='(1/taug)*exp(-t/taug)*heaviside(t)',
                        latex_name='g')
        b.set_action_text(
            'sum(yt[i]*((Dt+nu)*y[i] - h - e1*y[i]*Conv(g,x[i]) '
            '- e2*x[i]*Conv(g,y[i])) + xt[i]*((Dt+mu)*x[i] - hx - c*Conv(g,y[i])) '
            '- D*xt[i]^2 - D*yt[i]^2 for i in P)')
    return b.build()


def _run(m, ext, k=2, max_ell=1):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return compute_cumulants(m, k=k, max_ell=max_ell, external_fields=ext,
                                 tau_grid=TAU, parallel=False, verbose=False,
                                 use_cache=False)


@functools.lru_cache(maxsize=None)
def _twin3(ext):
    return _run(_model3('yx', True), list(ext))


@pytest.mark.parametrize('order', ['yx', 'xy'])
def test_kernel_on_two_fields_matches_the_markov_twin(order):
    ext = [('y', 1), ('y', 1)]
    ref = _twin3(tuple(ext))
    r = _run(_model3(order, False), ext)
    assert r['status'] == 'ok' and ref['status'] == 'ok'
    a = np.asarray(r['C_tau_by_ell'][1])
    b = np.asarray(ref['C_tau_by_ell'][1])
    assert np.max(np.abs(a - b) / np.abs(b)) < 1e-12
    # every kernel factor of the y-equation vertex sits on its own field
    legs = set()
    for d in r['diagrams']:
        for vt in d['typed_diagram'].vertex_assignments.values():
            if isinstance(vt, ConvVertexType):
                for att in vt.kernel_attachments:
                    legs.add((str(att['symbol']).split('__at__')[-1],
                              tuple(att['leg'])))
    assert legs and all(sym == f'{leg[0]}{leg[1]}' for sym, leg in legs)


def test_single_field_kernel_keeps_one_symbol():
    """A kernel that convolves one field only is not aliased (its sectors
    and numbers are those of before)."""
    from daedalus.engine.core.field_theory import FieldTheory
    b = TemporalModelBuilder('w3c one-field kernel')
    b.population('P', size=1)
    b.physical_field('x', population='P')
    for p, d in (('mu', 1.0), ('h', 0.45), ('c', 0.3), ('e', 0.2), ('D', 0.5),
                 ('taug', 2.0)):
        b.parameter(p, default=d, domain='positive')
    b.set_mf_equation('xstar', '0.5')
    b.define_kernel('g', time_expr='(1/taug)*exp(-t/taug)*heaviside(t)',
                    latex_name='g')
    b.set_action_text('sum(xt[i]*((Dt+mu)*x[i] - h - e*x[i]*Conv(g,x[i])) '
                      '- D*xt[i]^2 for i in P)')
    ft = FieldTheory(b.build(), taylor_order=3)
    ft.expand()
    assert ft._vertex_alias_by_tp is None
    assert not getattr(ft._ns, '_kernel_aliases', None)


def test_shared_kernel_through_the_caches(tmp_path, monkeypatch):
    """The alias sectors survive the expand cache (warm run, typed-diagram
    files removed so the vertices are rebuilt from the bundle), and a bundle
    written before them is stale."""
    import glob
    from sage.all import load as sage_load
    from daedalus.api import _expand_cache as EC
    from daedalus.engine.core.field_theory import FieldTheory
    monkeypatch.chdir(tmp_path)
    ext = [('y', 1), ('y', 1)]

    def cached():
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            return compute_cumulants(_model3('yx', False), k=2, max_ell=1,
                                     external_fields=ext, tau_grid=TAU,
                                     parallel=False, verbose=False,
                                     use_cache=True)

    cold = np.asarray(cached()['C_tau_by_ell'][1])
    for f in glob.glob('saved_models/*/unique_typed_*'):
        os.remove(f)
    warm = np.asarray(cached()['C_tau_by_ell'][1])
    ref = np.asarray(_twin3(tuple(ext))['C_tau_by_ell'][1])
    assert np.max(np.abs(warm - ref) / np.abs(ref)) < 1e-12
    assert np.array_equal(warm, cold)
    m = _model3('yx', False)
    path = glob.glob('saved_models/*/expand_taylor*.sobj')[0]
    bundle = sage_load(path)
    assert bundle.get('kernel_aliases')
    ft = FieldTheory(m, taylor_order=int(bundle['taylor_order']))
    EC.prepare_for_load(ft)
    assert EC.bundle_stale_reason(m, ft, bundle) is None
    del bundle['kernel_aliases']
    assert 'per-field kernel' in EC.bundle_stale_reason(m, ft, bundle)


# --- GEN-09 (the README's Hawkes example) -----------------------------------

def _hawkes(shared, name):
    from daedalus.api.sde import SDE
    from daedalus.api.delay import Kernel
    m = (SDE(name).population('E', size=2)
         .parameter('a', default=[1.0, 1.0], indexed_by=['E'])
         .parameter('Em', default=[-1.5, -1.5], indexed_by=['E'])
         .parameter('tau', default=2.0, domain='positive')
         .parameter('w', default=[[0.4, 0.4], [0.4, 0.4]], indexed_by=['E', 'E']))
    if shared:
        m = (m.define_kernel('r', kernel=Kernel.exponential('tau'))
             .point_process('n', 'a*exp(Em + sum(w[i,j]*Conv(r, n[j]) for j in E))',
                            population='E'))
    else:
        m = (m.define_kernel('r', kernel=Kernel.exponential('tau'), indexed_by=['E', 'E'])
             .point_process('n', 'a*exp(Em + sum(w[i,j]*Conv(r[i,j], n[j]) for j in E))',
                            population='E'))
    return m.build()


def test_readme_hawkes_example_is_complete_and_matches_the_indexed_kernel():
    ext = [('n', 1), ('n', 2)]
    r = _run(_hawkes(True, 'Hawkes'), ext)
    assert r['status'] == 'ok'
    assert not r.get('phase_j_incomplete')
    ref = _run(_hawkes(False, 'w3c Hawkes indexed kernel'), ext)
    a, b = np.asarray(r['C_tau']), np.asarray(ref['C_tau'])
    assert np.max(np.abs(a - b) / np.abs(b)) < 1e-12


# --- LOUD-08 --------------------------------------------------------------
# xt*((Dt+mu) x - h + ee x y + c x (g*y)) + yt*((Dt+nu) y - hy): the vertex
# xt dx dy has the coefficient ee + c g, and the kernel rate 1/taug equals
# the y line's pole nu.
_NU = 1.5
_P8 = (('mu', 1.0), ('nu', _NU), ('ee', 0.2), ('c', 0.3), ('h', 0.6),
       ('hy', 0.9), ('D', 0.4), ('taug', 1.0 / _NU))


def _model8(embedded):
    ys = 0.9 / _NU
    xs = 0.6 / (1.0 + (0.2 + 0.3) * ys)
    b = TemporalModelBuilder('w3c plain+kernel %s' % ('emb' if embedded else 'conv'))
    b.population('P', size=1)
    for f in ['x', 'y'] + (['u'] if embedded else []):
        b.physical_field(f, population='P')
    for p, d in _P8:
        b.parameter(p, default=d, domain='positive')
    b.set_mf_equation('xstar', repr(xs))
    b.set_mf_equation('ystar', repr(ys))
    if embedded:
        b.set_mf_equation('ustar', repr(ys))
        b.set_action_text(
            'sum(xt[i]*((Dt+mu)*x[i] - h + ee*x[i]*y[i] + c*x[i]*u[i]) '
            '+ yt[i]*((Dt+nu)*y[i] - hy) + ut[i]*((taug*Dt+1)*u[i] - y[i]) '
            '- D*xt[i]^2 - D*yt[i]^2 for i in P)')
    else:
        b.define_kernel('g', time_expr='(1/taug)*exp(-t/taug)*heaviside(t)',
                        latex_name='g')
        b.set_action_text(
            'sum(xt[i]*((Dt+mu)*x[i] - h + ee*x[i]*y[i] + c*x[i]*Conv(g,y[i])) '
            '+ yt[i]*((Dt+nu)*y[i] - hy) - D*xt[i]^2 - D*yt[i]^2 for i in P)')
    return b.build()


def test_plain_plus_kernel_coefficient_at_a_kernel_rate_equal_to_a_pole():
    ext = [('x', 1), ('x', 1)]
    r = _run(_model8(False), ext)
    ref = _run(_model8(True), ext)
    assert r['status'] == 'ok', r.get('phase_j_incomplete')
    a = np.asarray(r['C_tau_by_ell'][1])
    b = np.asarray(ref['C_tau_by_ell'][1])
    assert np.max(np.abs(a - b) / np.abs(b)) < 1e-12
