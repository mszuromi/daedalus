"""M7: the grouped path's warning for dropped δ(τ) records and the Phase J
flags stamp of saved results (``api/_grouped_phase_j.py``, ``api/save.py``,
``api/compute.py``).  No number changes: the grouped path does not carry the
records (Feature B will), it only says so; the stamp is provenance only."""
import math

import numpy as np
import pytest


@pytest.fixture
def FI():
    pytest.importorskip('sage.all')
    import engine.integration.time_domain.final_integral as fi
    return fi


def _set_flags(monkeypatch, fi, dd=False, semiinf=False, u2='box',
               t_min=-math.inf):
    monkeypatch.setattr(fi, 'USE_PHASE_J_DD_KERNELS', dd)
    monkeypatch.setattr(fi, 'SEMIINF_CHAIN', semiinf)
    monkeypatch.setattr(fi, 'UNBOUNDED_2D', u2)
    monkeypatch.setattr(fi, 'PHASE_J_T_MIN', t_min)


_ALL_ON = dict(dd=True, semiinf=True, u2='exact')


def test_grouped_path_carries_delta_records_without_warning():
    """Linear Hawkes autocorrelation at tree level: the per-diagram path
    returns one δ(τ) record; since Feature B the grouped path carries the
    same record (its per-diagram fallback builds it), does not warn, and
    C_tau is identical."""
    pytest.importorskip('sage.all')
    import warnings
    import daedalus as dd
    import api.compute as AC
    from api._grouped_phase_j import GroupedDeltaRecordsDroppedWarning
    model, _mod = dd.load_model('linear_hawkes')
    out = {}
    for grouped in (False, True):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            res = AC.compute_cumulants(
                model, k=2, max_ell=0, external_fields=[('n', 1), ('n', 1)],
                tau_grid=np.array([0.0, 1.0]), parallel=False, verbose=False,
                use_grouped_phase_j=grouped)
        dropped = [x for x in w
                   if issubclass(x.category, GroupedDeltaRecordsDroppedWarning)]
        out[grouped] = (res, dropped)
    res_pd, w_pd = out[False]
    res_gr, w_gr = out[True]
    rec_pd = res_pd['phase_j_by_ell'][0]['delta_contributions']
    rec_gr = res_gr['phase_j_by_ell'][0]['delta_contributions']
    assert len(rec_pd) == len(rec_gr) == 1
    assert complex(rec_pd[0]['coeff_fc'](0.0)) == complex(
        rec_gr[0]['coeff_fc'](0.0))
    assert w_pd == [] and w_gr == []
    assert res_gr['phase_j_by_ell'][0]['n_delta_records_dropped'] == 0
    assert np.array_equal(res_pd['C_tau'], res_gr['C_tau'])


def test_save_stamp_records_the_phase_j_flags(FI, monkeypatch, tmp_path):
    from api import save as S
    _set_flags(monkeypatch, FI, **_ALL_ON)
    flags = S._phase_j_flags()
    assert 'USE_PHASE_J_DD_KERNELS=1' in flags
    assert 'SEMIINF_CHAIN=1' in flags and 'UNBOUNDED_2D=exact' in flags
    assert 'USE_DBM_FALLBACK=1' in flags
    # a result records the flags it was computed with; the stamp copies them
    res = {'config': {'k': 2, 'max_ell': 0, 'phase_j_convention':
                      S.PHASE_J_CONVENTION, 'phase_j_flags': 'X=1'},
           'tau_grid': np.array([0.0]), 'C_tau': np.array([1.0 + 0j]),
           'C_tau_by_ell': {0: np.array([1.0 + 0j])}}
    assert S._result_stamp(res)['phase_j_flags'] == 'X=1'
    del res['config']['phase_j_flags']
    _set_flags(monkeypatch, FI)
    assert 'USE_PHASE_J_DD_KERNELS=0' in S._result_stamp(res)['phase_j_flags']


def test_npz_and_csv_carry_the_phase_j_flags(FI, monkeypatch, tmp_path):
    from api import save as S
    res = {'config': {'k': 2, 'max_ell': 0, 'phase_j_convention':
                      S.PHASE_J_CONVENTION, 'phase_j_flags': 'X=1;Y=box',
                      'external_fields': [('dx', 1), ('dx', 1)],
                      'fundamental': {}},
           'tau_grid': np.array([0.0, 1.0]),
           'C_tau': np.array([1.0 + 0j, 0.5 + 0j]),
           'C_tau_by_ell': {0: np.array([1.0 + 0j, 0.5 + 0j])},
           'mf_values': {}}
    S.save_npz(res, str(tmp_path / 'r.npz'))
    with np.load(str(tmp_path / 'r.npz')) as z:
        assert z['phase_j_flags'].tolist() == ['X=1;Y=box']
    S.save_csv(res, str(tmp_path / 'r.csv'))
    assert '# phase_j_flags: X=1;Y=box' in open(tmp_path / 'r.csv').read()


def test_compute_cumulants_records_the_flags_it_ran_with(FI, monkeypatch):
    import api.compute as AC
    import daedalus as dd
    model, _mod = dd.load_model('ou_quartic')
    _set_flags(monkeypatch, FI, **_ALL_ON)
    res = AC.compute_cumulants(model, k=2, max_ell=0,
                               external_fields=[('dx', 1), ('dx', 1)],
                               tau_grid=np.array([0.0]), parallel=False,
                               verbose=False, use_grouped_phase_j=True)
    flags = res['config']['phase_j_flags']
    assert 'USE_PHASE_J_DD_KERNELS=1' in flags
    assert flags.endswith('use_grouped_phase_j=1')
    assert res['config']['use_grouped_phase_j'] is True


def test_dropped_record_warning_counts_every_ell_and_beats_ignore(
        monkeypatch):
    pytest.importorskip('sage.all')
    import sys
    import warnings
    from api._grouped_phase_j import (GroupedDeltaRecordsDroppedWarning,
                                      warn_dropped_delta_records)
    pj = {0: {'n_delta_records_dropped': 2}, 1: None,
          2: {'n_delta_records_dropped': 3}}

    def issued(w):
        return [x for x in w if issubclass(
            x.category, GroupedDeltaRecordsDroppedWarning)]

    # the environment ignores warnings (the Sage launcher's
    # PYTHONWARNINGS=ignore): shown anyway, once per call
    monkeypatch.setattr(sys, 'warnoptions', ['ignore'])
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('ignore')
        assert warn_dropped_delta_records(pj) == 5
        assert warn_dropped_delta_records(pj) == 5
        assert warn_dropped_delta_records({0: {}}) == 0
    got = issued(w)
    assert len(got) == 2 and 'dropped 5' in str(got[0].message)
    # a filter set for the category itself is respected
    with warnings.catch_warnings(record=True) as w:
        warnings.filterwarnings('ignore',
                                category=GroupedDeltaRecordsDroppedWarning)
        warn_dropped_delta_records(pj)
    assert not issued(w)
    # filters set in code (no environment ignore) are respected, -W error too
    monkeypatch.setattr(sys, 'warnoptions', [])
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('ignore')
        warn_dropped_delta_records(pj)
    assert not issued(w)
    monkeypatch.setattr(sys, 'warnoptions', ['error'])
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        with pytest.raises(GroupedDeltaRecordsDroppedWarning):
            warn_dropped_delta_records(pj)


@pytest.mark.parametrize('which', ['stale', 'nquad'])
def test_stale_and_nquad_warnings_beat_the_environment_ignore(monkeypatch,
                                                             which):
    """The Sage launcher exports PYTHONWARNINGS=ignore; StaleResultWarning
    and PhaseJNquadFallbackWarning are shown anyway (as the grouped δ-record
    warning), while a filter for the category itself is respected."""
    import sys
    import warnings
    if which == 'stale':
        import api.save as S
        category = S.StaleResultWarning
        monkeypatch.setattr(S, '_STALE_WARNED', set())

        def emit():
            S._STALE_WARNED.clear()
            S._warn_if_stale('0.1.0', None, 'x.npz')
    else:
        pytest.importorskip('sage.all')
        import engine.integration.time_domain.final_integral as FI
        category = FI.PhaseJNquadFallbackWarning
        import inspect
        src = inspect.getsource(FI)
        assert ('_ensure_warning_shown(PhaseJNquadFallbackWarning, message=msg,\n'
                '                              stacklevel=2)\n'
                '        warnings.warn(msg, PhaseJNquadFallbackWarning') in src

        def emit():
            FI._ensure_warning_shown(category)
            warnings.warn('probe', category)
    monkeypatch.setattr(sys, 'warnoptions', ['ignore'])
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('ignore')
        emit()
    assert any(issubclass(x.category, category) for x in w)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('ignore')
        warnings.filterwarnings('ignore', category=category)
        emit()
    assert not any(issubclass(x.category, category) for x in w)
    monkeypatch.setattr(sys, 'warnoptions', [])
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('ignore')
        emit()
    assert not any(issubclass(x.category, category) for x in w)


@pytest.mark.parametrize('which', ['stale', 'nquad'])
def test_the_visibility_helpers_respect_filters_set_in_code(monkeypatch,
                                                           which):
    """Under the environment's ignore (the LAST unconditional ignore filter)
    a filter set later in code -- 'error', or a targeted 'ignore' -- decides,
    not the helper (review finding)."""
    import sys
    import warnings
    if which == 'stale':
        import api.save as S
        helper, category = S._ensure_warning_shown, S.StaleResultWarning
    else:
        pytest.importorskip('sage.all')
        import engine.integration.time_domain.final_integral as FI
        helper = FI._ensure_warning_shown
        category = FI.PhaseJNquadFallbackWarning
    monkeypatch.setattr(sys, 'warnoptions', ['ignore'])

    def emit():
        helper(category)
        warnings.warn('probe', category)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')          # the environment's
        warnings.simplefilter('error')           # set in code: wins
        with pytest.raises(category):
            emit()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('ignore')          # the environment's
        warnings.filterwarnings('ignore', category=UserWarning)   # in code
        emit()
    assert not any(issubclass(x.category, category) for x in w)
    # (A plain simplefilter('ignore') set in code is the SAME filter as the
    # environment's; Python merges the two, so it cannot be told apart and
    # the warning is shown -- a documented limit of the helpers.)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('ignore')          # the environment's only
        emit()
    assert any(issubclass(x.category, category) for x in w)


def _visibility_case(which):
    """(helper, category, real_emit) for ``api.save`` / the engine copy."""
    import warnings
    if which == 'stale':
        import api.save as S
        helper, category = S._ensure_warning_shown, S.StaleResultWarning
    else:
        pytest.importorskip('sage.all')
        import engine.integration.time_domain.final_integral as FI
        helper, category = FI._ensure_warning_shown, FI.PhaseJNquadFallbackWarning

    def emit(msg='probe text'):
        helper(category, message=msg, stacklevel=1)
        warnings.warn(msg, category, stacklevel=1)
    return helper, category, emit


def _shown(w, category):
    return sum(issubclass(x.category, category) for x in w)


@pytest.mark.parametrize('which', ['stale', 'nquad'])
def test_the_visibility_helpers_skip_filters_aimed_at_other_warnings(
        monkeypatch, which):
    """Sage installs message-specific ``ignore`` filters for numpy at
    startup (``numpy.ndarray size changed``, ...).  They are not a user's
    decision about the package warning: it is shown, exactly once.  A
    category-wide ignore, or a message / module filter that does match,
    still hides it (docket item 51)."""
    import sys
    import warnings
    helper, category, emit = _visibility_case(which)
    monkeypatch.setattr(sys, 'warnoptions', ['ignore'])

    def sage_like():
        warnings.simplefilter('ignore')                    # env ignore-all
        warnings.filterwarnings('ignore', message='numpy.ndarray size changed',
                                category=Warning)          # Sage's, in front

    with warnings.catch_warnings(record=True) as w:
        sage_like()
        emit()
    assert _shown(w, category) == 1
    # a user ignore aimed at another message does not hide this one either
    with warnings.catch_warnings(record=True) as w:
        sage_like()
        warnings.filterwarnings('ignore', message='other text', category=category)
        emit()
        assert _shown(w, category) == 1
        n = len(warnings.filters)
        helper(category, message='probe text', stacklevel=1)   # idempotent
        assert len(warnings.filters) == n

    # the user's own filters still decide
    cases = {
        'category-wide ignore': dict(category=category),
        'matching message': dict(message='probe', category=category),
        'matching message, any category': dict(message='probe text'),
        'this module (regex)': dict(module=__name__.replace('.', r'\.'),
                                    category=category),
    }
    for label, kw in cases.items():
        with warnings.catch_warnings(record=True) as w:
            sage_like()
            warnings.filterwarnings('ignore', **kw)
            emit()
        assert _shown(w, category) == 0, label
    # a module filter aimed at another module does not hide it
    with warnings.catch_warnings(record=True) as w:
        sage_like()
        warnings.filterwarnings('ignore', module=r'some\.other\.module',
                                category=category)
        emit()
    assert _shown(w, category) == 1
    # a filter set in code, or 'error', is respected
    with warnings.catch_warnings():
        sage_like()
        warnings.simplefilter('error', category)
        with pytest.raises(category):
            emit()
    # a user ignore aimed at ANOTHER message of the category keeps working
    # after this one was let through
    with warnings.catch_warnings(record=True) as w:
        sage_like()
        warnings.filterwarnings('ignore', message='second text', category=category)
        emit('probe text')
        emit('second text')
    assert [str(x.message) for x in w if issubclass(x.category, category)] \
        == ['probe text']
    # message unknown (a caller that does not pass it): message-specific
    # filters are respected, a numpy one included -- the conservative reading
    with warnings.catch_warnings(record=True) as w:
        sage_like()
        helper(category)
        warnings.warn('probe text', category)
    assert _shown(w, category) == 0
    # and with no environment ignore the helper does nothing at all
    monkeypatch.setattr(sys, 'warnoptions', [])
    with warnings.catch_warnings(record=True) as w:
        sage_like()
        emit()
    assert _shown(w, category) == 0
