"""``api.report`` matches each diagram page to its own Phase J callable.

The per-diagram panel used to index a callable list by page number, which
breaks as soon as the diagram list changes order (the prediagram source
decides the order) and could not line up across loop orders anyway.  The
pages now look their callable up by diagram identity; these tests pin that
(i) every diagram gets one, (ii) the per-diagram curves add up to the total
slice, (iii) the result does not depend on the record order, and (iv) the
pages themselves (order, "Diagram i / N", vertex numbers, graph, curves) do
not depend on which labelled representative of each class the prediagram
source provided.
"""
import importlib.util
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))

pytest.importorskip('sage.all')

from tests._diagram_order import signature_key                  # noqa: E402

_TAUS = np.array([0.0, 0.5, 1.0, 3.0])


def _ou_quartic_result(max_ell=1):
    from api.compute import compute_cumulants
    spec = importlib.util.spec_from_file_location(
        'ou_quartic_model', 'models/ou_quartic.model.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return compute_cumulants(mod.build(), k=2, max_ell=max_ell,
                             external_fields=[('dx', 1)] * 2,
                             tau_grid=_TAUS, use_cache=False,
                             parallel=False, verbose=False)


def _curves_by_signature(result):
    from api.report import (_diagram_contribution_curve,
                            _per_diagram_contributions)
    contribs = _per_diagram_contributions(result)
    out = {}
    for rec in result['diagrams']:
        fn = contribs[id(rec['typed_diagram'])]
        key = (rec['ell'], signature_key(rec['typed_diagram']))
        assert key not in out
        out[key] = _diagram_contribution_curve(fn, result['tau_grid'], 2)
    return out


@pytest.fixture(scope='module')
def result():
    return _ou_quartic_result()


def test_every_diagram_has_its_callable_and_they_sum_to_the_slice(result):
    from api.report import (_diagram_contribution_curve,
                            _per_diagram_contributions)
    contribs = _per_diagram_contributions(result)
    assert len(contribs) == len(result['diagrams']) == 5   # 1 tree + 4 loops
    for ell, c_ell in result['C_tau_by_ell'].items():
        recs = [r for r in result['diagrams'] if r['ell'] == ell]
        assert recs
        total = sum(_diagram_contribution_curve(
            contribs[id(r['typed_diagram'])], result['tau_grid'], 2)
            for r in recs)
        np.testing.assert_allclose(total, c_ell, rtol=1e-13, atol=1e-15)


def test_per_diagram_curves_do_not_depend_on_record_order(result,
                                                         monkeypatch):
    """Same curve for each diagram class whatever order the records arrive
    in: the loader's own order against a seed-1 shuffle.  Under a whole-run
    ``DAEDALUS_TEST_DIAGRAM_ORDER`` the module ``result`` is already
    reordered, so the test computes its own unreordered reference; the
    shuffle replaces the whole-run one, so both orders are the same in
    every run."""
    from tests._diagram_order import (apply_diagram_order,
                                      unwrapped_order_patches)
    undo = unwrapped_order_patches()
    plain = result
    if undo:
        with monkeypatch.context() as mp:
            for mod, name, value in undo:
                mp.setattr(mod, name, value)
            plain = _ou_quartic_result()
    with monkeypatch.context() as mp:
        apply_diagram_order(mp, 'shuffle', 1)
        shuffled = _ou_quartic_result()
    order = [[(r['ell'], signature_key(r['typed_diagram']))
              for r in res['diagrams']] for res in (plain, shuffled)]
    assert order[0] != order[1], 'the shuffle must actually reorder'
    a, b = _curves_by_signature(plain), _curves_by_signature(shuffled)
    assert sorted(a) == sorted(b)
    for key in a:
        np.testing.assert_allclose(a[key], b[key], rtol=1e-13, atol=1e-15)


def test_generate_report_hands_each_page_its_callable(result, tmp_path,
                                                      monkeypatch):
    import api.report as report
    seen = []
    orig = report._draw_diagram_page

    def spy(pdf, idx, total, td_record, res, k, contrib=None,
            display_id=None):
        seen.append(contrib is not None)
        return orig(pdf, idx, total, td_record, res, k, contrib, display_id)

    monkeypatch.setattr(report, '_draw_diagram_page', spy)
    out = tmp_path / 'report.pdf'
    cfg = result['config']
    report.generate_report({'name': 'ou_quartic'}, k=2, max_ell=1,
                           fundamental=dict(cfg.get('fundamental') or {}),
                           external_fields=cfg['external_fields'],
                           output_pdf=str(out), result=result, verbose=False)
    assert out.is_file() and out.stat().st_size > 0
    assert seen == [True] * len(result['diagrams'])


def test_canonical_view_key_is_the_diagram_signature(result):
    from api.report import _canonical_view
    from engine.diagrams.symmetry import diagram_signature
    for rec in result['diagrams']:
        td = rec['typed_diagram']
        key, display_id = _canonical_view(td)
        assert key == str(diagram_signature(td))
        n = len(list(td.prediagram[0].vertices()))
        assert sorted(display_id.values()) == list(range(n))
        assert sorted(display_id[v] for v in td.external_legs) == \
            list(range(len(td.external_legs)))          # leaves first


def _pages(result):
    """Everything a diagram page shows, in page order, as plain data."""
    from api.report import (_assignment_lines, _diagram_contribution_curve,
                            _pages_in_canonical_order,
                            _per_diagram_contributions, _prediagram_layout)
    contribs = _per_diagram_contributions(result)
    n = len(result['diagrams'])
    out = []
    for idx, (key, did, rec) in enumerate(
            _pages_in_canonical_order(result['diagrams']), 1):
        td = rec['typed_diagram']
        out.append(dict(
            key=key,
            title=(f'Diagram {idx} / {n}', rec['classify'].get('Scal'),
                   str(rec['combined_prefactor'])),
            assign=_assignment_lines(td, did),
            layout=_prediagram_layout(td, did),
            raw_layout=_prediagram_layout(td),      # the source's own labels
            curve=_diagram_contribution_curve(contribs[id(td)],
                                              result['tau_grid'], 2)))
    return out


def test_report_pages_do_not_depend_on_the_prediagram_source(
        result, monkeypatch):
    """The default records (``result``), the eager and the streamed records
    (both forced with ``DAEDALUS_PREDIAGRAM_EAGER``) and a shuffled list give
    the same pages: same order, same text, same drawing; the curves agree to
    rounding.  The eager and streamed records are different labelled
    representatives (checked), so this exercises the canonical numbering,
    not just the sort."""
    from engine.enumeration import prediagram_cache as pdc
    from tests._diagram_order import apply_diagram_order

    runs = []
    for env in ('1', '0'):
        with monkeypatch.context() as mp:
            mp.setenv(pdc._EAGER_ENV, env)
            runs.append(_ou_quartic_result())
    with monkeypatch.context() as mp:
        apply_diagram_order(mp, 'shuffle', 3)
        runs.append(_ou_quartic_result())
    ref = _pages(result)
    assert len(ref) == 5
    raw_differs = False
    for other in map(_pages, runs):
        assert len(other) == len(ref)
        for a, b in zip(ref, other):
            for field in ('key', 'title', 'assign', 'layout'):
                assert a[field] == b[field], field
            np.testing.assert_allclose(b['curve'], a['curve'],
                                       rtol=1e-13, atol=1e-15)
            raw_differs |= a['raw_layout'] != b['raw_layout']
    assert raw_differs, 'expected other labelled representatives'
