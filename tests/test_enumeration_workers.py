"""One worker knob (W8p S3): ``compute_cumulants(n_workers=...)`` reaches the
streamed prediagram enumeration; ``DAEDALUS_PREDIAGRAM_PROCS`` stays an
override; macOS never forks (the enumeration is serial there).

Run:  PYTHONHASHSEED=0 sage -python -m pytest tests/test_enumeration_workers.py -q
"""
import sys

import pytest

pytest.importorskip('sage.all')

import daedalus as dd                                             # noqa: E402
from daedalus.api import _diagrams as D                                    # noqa: E402
from daedalus.api.compute import compute_cumulants                         # noqa: E402
from daedalus.engine.enumeration import prediagram_cache as pdc            # noqa: E402
import daedalus.engine.enumeration.loop_diagram_enumeration as L           # noqa: E402


def test_enumeration_procs_from_the_knob():
    assert D.enumeration_procs(False, 8) == 1
    assert D.enumeration_procs(True, None) == 1       # never started unasked
    assert D.enumeration_procs(True, 3) == 3


def test_env_override_then_n_procs_then_one(monkeypatch):
    monkeypatch.delenv(pdc._PROCS_ENV, raising=False)
    assert pdc._enum_procs() == 1
    assert pdc._enum_procs(4) == 4
    with pytest.raises(ValueError):
        pdc._enum_procs(0)
    monkeypatch.setenv(pdc._PROCS_ENV, '2')
    assert pdc._enum_procs(4) == 2
    monkeypatch.setenv(pdc._PROCS_ENV, '')
    assert pdc._enum_procs(4) == 4


def test_compute_cumulants_passes_n_workers_to_the_enumeration(monkeypatch):
    seen = []
    real = pdc.load_prediagrams

    def spy(*a, **k):
        seen.append(k.get('n_procs'))
        return real(*a, **k)
    monkeypatch.setattr(pdc, 'load_prediagrams', spy)
    monkeypatch.setenv('DAEDALUS_PARALLEL', '0')        # nothing forks here
    model, _ = dd.load_model('ou_quartic')
    compute_cumulants(model, k=2, max_ell=1, external_fields=[('dx', 1)] * 2,
                      parameters={'mu': 1.0, 'eps': 0.02, 'D': 1.0},
                      tau_grid=[0.0, 0.5], parallel=True, n_workers=3,
                      use_cache=False, verbose=False)
    assert seen and set(seen) == {3}


@pytest.mark.skipif(sys.platform == 'darwin',
                    reason='macOS never forks; Linux CI covers the pool')
def test_streamed_records_do_not_depend_on_the_worker_count(tmp_path,
                                                             monkeypatch):
    monkeypatch.delenv(pdc._PROCS_ENV, raising=False)
    monkeypatch.delenv('DAEDALUS_PARALLEL', raising=False)
    monkeypatch.setattr(pdc, '_STREAMED_MEMO', {})
    one = L.stream_prediagram_certs(3, 1, n_procs=1)
    monkeypatch.setattr(pdc, '_STREAMED_MEMO', {})
    recs, src = pdc.load_prediagrams(str(tmp_path), 3, 1, use_cache=False,
                                     stream=True, n_procs=2)
    assert src == 'streamed'
    assert len(recs) == len(one)
