"""A cold and a warm cache give byte-identical cumulants.

A Sage graph does not survive a pickle round trip unchanged: the sparse backend
iterates parallel edges newest first and the unpickler re-adds them in iteration
order, so every round trip reverses each group of parallel edges in ``D.edges()``
(and above 10 vertices can move vertex slots, hence ``G``'s edge order).  The
typed-diagram cache pickled the prediagram records as plain Sage graphs, so a
warm run summed the same diagrams in another order than the cold run that wrote
the file: ~3e-14 relative on a k=3 one-loop sum of 83k diagrams.  Prediagram
records are now :class:`PrediagramRecord`s, which pickle their construction
inputs and are rebuilt by the same code that built them.
"""
import os
import pickle
import subprocess
import sys
import textwrap

import pytest

pytest.importorskip("sage.all")
from daedalus.engine.enumeration import prediagram_cache as pdc            # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _state(rec):
    """Everything about a record whose ORDER reaches the integrator."""
    D, G, leaves, internal = rec
    out = [list(leaves), list(internal)]
    for H in (D, G):
        vs = list(H.vertex_iterator())
        out += [vs, H.edges(sort=False), [H.edges_incident(v, sort=False) for v in vs],
                [list(H.neighbor_iterator(v)) for v in vs]]
        if H.is_directed():
            out += [[list(H.neighbor_out_iterator(v)) for v in vs],
                    [list(H.neighbor_in_iterator(v)) for v in vs]]
    return out


def _roundtrips(recs):
    from sage.all import dumps, loads
    yield loads(dumps(recs))                                     # what PipelineCache.save does
    yield pickle.loads(pickle.dumps(recs, protocol=pickle.HIGHEST_PROTOCOL))


# (3,2): parallel edges everywhere; (5,1): graphs with more than 10 vertices
@pytest.mark.parametrize('k,ell', [(3, 2), (5, 1)])
def test_records_pickle_bit_for_bit(k, ell):
    recs = pdc.records_from_certs(pdc.load_shipped_certs(k, ell))
    assert all(isinstance(r, pdc.PrediagramRecord) for r in recs)
    want = [_state(r) for r in recs]
    for back in _roundtrips(recs):
        assert all(isinstance(r, pdc.PrediagramRecord) for r in back)
        assert [_state(r) for r in back] == want
    # a plain Sage pickle does NOT preserve the order (why the record exists)
    from sage.all import dumps, loads
    plain = loads(dumps([tuple(r) for r in recs]))
    assert [_state(r) for r in plain] != want


def test_eager_records_are_rebuilt_as_the_same_labelled_graphs():
    from daedalus.engine.enumeration.loop_diagram_enumeration import enumerate_all
    eager = enumerate_all(2, 2, verbose=False)[2]
    recs, source = pdc.load_prediagrams('unused', 2, 2, use_cache=False, stream=False)
    assert source == 'eager' and len(recs) == len(eager)
    for r, e in zip(recs, eager):
        assert isinstance(r, pdc.PrediagramRecord)
        assert sorted(r[0].edges(sort=False)) == sorted(e[0].edges(sort=False))
        assert sorted(r[1].edges(sort=False)) == sorted(e[1].edges(sort=False))
        assert list(r[2]) == list(e[2]) and list(r[3]) == list(e[3])
        assert pdc.canonical_record(r) is r
    want = [_state(r) for r in recs]
    for back in _roundtrips(recs):
        assert [_state(r) for r in back] == want


_RUN = """
    import os, sys, warnings
    warnings.simplefilter('ignore')
    sys.path.insert(0, {root!r})
    import numpy as np
    from daedalus.api import compute_cumulants
    from daedalus.api.noise import Poisson
    from daedalus.api.sde import SDE
    p = dict(al1=0.132, be1=0.132, ga1=3.9, al2=10.19, be2=0.0054, ga2=12.7, s=0.021,
             b1=1.04, b2=0.21)
    m = SDE('cache state twin')
    for k, v in p.items():
        m = m.parameter(k, default=v, domain='positive')
    m = m.field('u').field('w').drift('u', '0').drift('w', '0')
    cp = lambda i: f'((ga{{i}}+al{{i}}-be{{i}})/2)'
    cn = lambda i: f'((ga{{i}}-al{{i}}+be{{i}})/2)'
    m = (m.noise('up1', Poisson(rate=f'({{cp(1)}} + be1*exp(-u))/s', size='s'), couples={{'u': 'b1', 'w': 'b2'}})
          .noise('dn1', Poisson(rate=f'(al1*exp(u) + {{cn(1)}})/s', size='-s'), couples={{'u': 'b1', 'w': 'b2'}})
          .noise('up2', Poisson(rate=f'({{cp(2)}} + be2*exp(-w))/s', size='s'), couples={{'u': 'b2', 'w': 'b1'}})
          .noise('dn2', Poisson(rate=f'(al2*exp(w) + {{cn(2)}})/s', size='-s'), couples={{'u': 'b2', 'w': 'b1'}})
         ).build()
    r = compute_cumulants(m, k=2, max_ell=1, external_fields=[('du', 1)] * 2,
                          tau_grid=np.array([0.0, 0.7]), verbose=False, use_cache=True,
                          parallel={parallel}, n_workers={n_workers})
    by = r['C_tau_by_ell']
    print(len(r['diagrams']),
          [complex(z).real.hex() for z in np.atleast_1d(r['C_tau'])],
          [[complex(z).real.hex() for z in np.atleast_1d(by[l])] for l in sorted(by)])
"""


def _run(cwd, parallel=False, n_workers=None):
    code = textwrap.dedent(_RUN.format(root=ROOT, parallel=parallel, n_workers=n_workers))
    p = subprocess.run([sys.executable, '-c', code], cwd=cwd, capture_output=True,
                       text=True, timeout=900, env=dict(os.environ, PYTHONHASHSEED='0'))
    assert p.returncode == 0, p.stderr[-3000:]
    return p.stdout.strip().splitlines()[-1]


def _typed_cache_files(cwd):
    out = {}
    for dirpath, _, files in os.walk(os.path.join(cwd, 'saved_models')):
        for f in files:
            if f.startswith('unique_typed_mult_'):
                path = os.path.join(dirpath, f)
                out[path] = os.stat(path).st_mtime_ns
    return out


def test_cold_and_warm_cache_runs_are_byte_identical(tmp_path):
    """Two fields and parallel loop edges (925 diagrams, k=2 one loop): before the
    fix the warm one-loop value differed from the cold one in the last two bits."""
    cwd = str(tmp_path)
    cold = _run(cwd)
    files = _typed_cache_files(cwd)
    assert len(files) == 2, files                              # ell = 0 and 1 written
    warm = _run(cwd)
    assert _typed_cache_files(cwd) == files                    # served, not rewritten
    assert warm == cold
    # a cold run whose type-assignment fork pool returns its typed diagrams
    # pickled (Linux; serial on macOS, which never forks)
    par = tmp_path / 'parallel'
    par.mkdir()
    assert _run(str(par), parallel=True, n_workers=2) == cold
