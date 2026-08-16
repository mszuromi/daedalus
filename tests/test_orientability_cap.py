"""The |V| <= 3k+3l-3 orientability cap must not drop any prediagram.

The cap is what makes ell=3 tractable (it removes ~33x of the candidate edge
multisets there), and enumeration COMPLETENESS is the framework's strongest
claim -- so the justification must not rest on a code comment recording an
offline check.  Here we lift the cap and assert the prediagram set is
unchanged: every topology above the cap admits zero valid orientations.
"""
import pytest
import engine.enumeration.loop_diagram_enumeration as L


def _prediagram_certs(k, ell):
    """Isomorphism certificates of every prediagram at (k, ell)."""
    return {L._iso_cert(D) for D, _G, _lv, _in in
            L.enumerate_all(k, ell, verbose=False)[2]}


@pytest.mark.parametrize('k,ell', [(2, 1), (3, 1), (4, 1), (2, 2), (3, 2)])
def test_cap_drops_no_prediagram(monkeypatch, k, ell):
    capped = _prediagram_certs(k, ell)

    # Lift the cap: fall back to the proven per-tree bounds alone.
    monkeypatch.setattr(L, 'v_max_orientable_bound',
                        lambda _k, _ell: 10 ** 6)
    uncapped = _prediagram_certs(k, ell)

    assert uncapped == capped, (
        f'(k={k}, ell={ell}): lifting the orientability cap changed the '
        f'prediagram set by {len(uncapped ^ capped)} class(es) -- the cap '
        f'is NOT safe, so |V| <= 3k+3l-3 drops orientable topologies.')


def test_bound_matches_its_formula():
    for k in range(2, 7):
        for ell in range(0, 4):
            assert L.v_max_orientable_bound(k, ell) == 3 * k + 3 * ell - 3
