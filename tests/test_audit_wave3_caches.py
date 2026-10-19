"""
tests/test_audit_wave3_caches.py
================================
Audit wave 3c, cache staleness.

* ACC-08: the propagator and typed-diagram caches are addressed by the model
  NAME, and the model-text fingerprint does not see population sizes: a
  population resized under the same name was served the old size's
  diagrams.  Both caches now carry ``model_cache_signature`` (the text
  fingerprint and every index set's members).
* ACC-09: a file without a signature (written before it, or by another
  writer) was accepted; it is stale now.
* ACC-18: a local prediagram cell whose class count disagrees with the
  package manifest (stale, truncated) was served forever; it now falls back
  to the shipped data (or a compute) with a warning.

References: the cold (use_cache=False) run of the same model, the exact
Lyapunov covariance of the Markov embedding, the shipped manifest.

Run:  sage -python -m pytest tests/test_audit_wave3_caches.py -q
"""
from __future__ import annotations

import glob
import os
import sys
import warnings

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from api import compute_cumulants                       # noqa: E402
from api._propagator import (cache_signature_stale,     # noqa: E402
                             model_cache_signature)
from api.model import TemporalModelBuilder              # noqa: E402

TAU = np.array([0.0, 1.0])


def _pop_model(n):
    b = TemporalModelBuilder('w3c resized pop')       # the same name for every size
    b.population('E', size=n)
    b.physical_field('x', population='E')
    for p, d in (('mu', 1.0), ('a', 0.3), ('D', 0.5)):
        b.parameter(p, default=d, domain='positive')
    b.set_mf_equation('xstar', '0')
    b.set_action_text('sum(xt[i]*((Dt+mu)*x[i] + a*x[i]*sum(x[j] for j in E)) '
                      '- D*xt[i]^2 for i in E)')
    return b.build()


def _run(m, use_cache, max_ell=1, ext=(('x', 1), ('x', 1)), tau=TAU):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return compute_cumulants(m, k=2, max_ell=max_ell, external_fields=list(ext),
                                 tau_grid=tau, parallel=False, verbose=False,
                                 use_cache=use_cache)


def test_signature_sees_population_sizes():
    m1, m2 = _pop_model(1), _pop_model(2)
    assert m1['spec_signature'] == m2['spec_signature']      # the text is the same
    assert model_cache_signature(m1) != model_cache_signature(m2)
    assert cache_signature_stale(m2, model_cache_signature(m1)) is not None
    assert cache_signature_stale(m2, model_cache_signature(m2)) is None
    assert 'without a model signature' in cache_signature_stale(m2, None)


def test_resized_population_is_not_served_the_old_diagrams(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _run(_pop_model(1), True)                                   # cold, cached
    warm = _run(_pop_model(2), True)                            # same name, other size
    cold = _run(_pop_model(2), False)
    for ell in (0, 1):
        a = np.asarray(warm['C_tau_by_ell'][ell])
        b = np.asarray(cold['C_tau_by_ell'][ell])
        assert np.array_equal(a, b)
    assert len(warm['diagrams']) == len(cold['diagrams'])


def _kernel_model(T):
    b = TemporalModelBuilder('w3c kernel edit')
    b.population('P', size=1)
    b.physical_field('x', population='P')
    for p, d in (('mu', 1.0), ('c', 0.5), ('h', 0.5), ('D', 0.5)):
        b.parameter(p, default=d, domain='positive')
    b.set_mf_equation('xstar', 'h/(mu - c)')
    b.define_kernel('g', time_expr='exp(-t/%s)/%s*heaviside(t)' % (T, T),
                    latex_name='g')
    b.set_action_text('sum(xt[i]*((Dt+mu)*x[i] - h - c*Conv(g,x[i])) '
                      '- D*xt[i]^2 for i in P)')
    return b.build()


def _lyapunov(T, tau):
    # x' = -mu x + c u + noise(2D), u' = (x - u)/T
    from scipy.linalg import expm, solve_continuous_lyapunov
    A = np.array([[-1.0, 0.5], [1 / T, -1 / T]])
    S = solve_continuous_lyapunov(A, -np.diag([1.0, 0.0]))
    return np.array([(expm(A * t) @ S)[0, 0] for t in tau])


def test_files_without_a_signature_are_stale(tmp_path, monkeypatch):
    from engine.core.cache import PipelineCache
    monkeypatch.chdir(tmp_path)
    tau = np.array([0.0, 1.0, 3.0])
    _run(_kernel_model(1), True, max_ell=0, tau=tau)
    # what a file written before the signature looks like
    cdir = 'saved_models/w3c_kernel_edit'
    pc = PipelineCache(cdir)
    prop = pc.load('propagator')
    prop['spec_signature'] = None
    prop.pop('cache_signature', None)
    pc.save('propagator', prop)
    files = glob.glob(cdir + '/unique_typed_*_k2_l0.sobj')
    assert files
    for f in files:
        stage = os.path.basename(f)[:-len('_k2_l0.sobj')]
        d = pc.load(stage, k=2, loop_order=0)
        d['spec_signature'] = None
        d.pop('cache_signature', None)
        pc.save(stage, d, k=2, loop_order=0)
    # the kernel edited under the same name: the stale propagator must not
    # be served
    warm = np.real(np.asarray(_run(_kernel_model(2), True, max_ell=0,
                                   tau=tau)['C_tau_by_ell'][0]))
    ex = _lyapunov(2.0, tau)
    assert np.max(np.abs(warm - ex) / np.abs(ex)) < 1e-12


# --- ACC-18 ---------------------------------------------------------------

def test_short_local_prediagram_cell_is_not_served(tmp_path):
    import json
    import engine.enumeration.prediagram_cache as P
    man = json.load(open(os.path.join(P.SHIPPED_DIR, 'MANIFEST.json')))
    want = {(e['k'], e['ell']): e['count'] for e in man['files']}[(2, 1)]
    root = str(tmp_path / 'pd')
    recs, src = P.load_prediagrams(root, 2, 1)
    assert src == 'shipped' and len(recs) == want
    certs = sorted(P.load_v2_certs(root, 2, 1))
    P.save_v2_certs(root, 2, 1, certs[:-3])                 # a truncated local cell
    with pytest.warns(RuntimeWarning, match='not served'):
        recs2, src2 = P.load_prediagrams(root, 2, 1)
    assert src2 == 'shipped' and len(recs2) == want
    # ... and the local file was rewritten: the next load is a clean v2 hit
    with warnings.catch_warnings():
        warnings.simplefilter('error', RuntimeWarning)
        recs3, src3 = P.load_prediagrams(root, 2, 1)
    assert src3 == 'v2' and len(recs3) == want


def test_short_local_v1_cell_is_not_served(tmp_path, monkeypatch):
    """A new case: a v1 file (another format, another cell)."""
    import json
    from sage.all import save as sage_save
    import engine.enumeration.prediagram_cache as P
    man = json.load(open(os.path.join(P.SHIPPED_DIR, 'MANIFEST.json')))
    want = {(e['k'], e['ell']): e['count'] for e in man['files']}[(3, 1)]
    root = str(tmp_path / 'pd')
    recs, _src = P.load_prediagrams(root, 3, 1)
    os.remove(P.v2_path(root, 3, 1))
    sage_save(list(recs[:2]), P.v1_path(root, 3, 1).removesuffix('.sobj'))
    assert P.v1_exists(root, 3, 1)
    with pytest.warns(RuntimeWarning, match='v1 file'):
        recs2, src2 = P.load_prediagrams(root, 3, 1)
    assert len(recs2) == want and src2 == 'shipped'
