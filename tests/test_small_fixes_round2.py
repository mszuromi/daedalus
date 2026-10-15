"""Round 2 small fixes (work item 49), each reproduced on main before its fix:

1. A coupled linear drift written with float literals (a model with no declared parameters) failed
   the mean-field self-check at ``expand()`` ("mean-field saddle does not satisfy its own
   equations"); the same model with named parameters built.  Cause: ``_default_fundamental_point``
   returned ``None`` for a parameter-free model, which switched the numerical fallback off, and the
   symbolic check substitutes the coupled implicit saddle simultaneously, so it cannot close the
   loop (``x* = 0.8 y*``, ``y* = -0.5 x*``) on its own.
2. A PARTIAL ``parameters=`` dict reached the mean-field DAE with the omitted parameters unbound
   (bare ``NameError``).  Omitted parameters now come from the model's declared defaults; one with
   no default raises a ``ValueError`` naming it.
3. A NaN in ``tau_grid`` was evaluated as tau = 0 (the Ito equal-time nudge made the comparison
   pass silently).  NaN / inf in ``tau_grid`` / ``times`` / ``tau_max`` now raise up front.

Every bit-for-bit comparison runs the first form once and discards it (cold and warm caches differ
by ~1e-16)."""
import numpy as np
import pytest

pytest.importorskip('sage.all')

import api.compute as AC                                            # noqa: E402
from api.noise import Gaussian                                      # noqa: E402
from api.sde import SDE                                             # noqa: E402

TAU = np.array([-1.0, 0.0, 0.7, 1.5])
XY = [('x', 1), ('y', 1)]


def _run(model, ext, ell=1, **kw):
    kw.setdefault('tau_grid', TAU)
    return AC.compute_cumulants(model, k=2, max_ell=ell, external_fields=ext,
                                use_cache=False, parallel=False, verbose=False, **kw)


def _coupled(named, const=False, cubic=True, ints=False):
    """Non-symmetric two-field drift, with float literals or the same numbers as parameters."""
    s = SDE('coupled2')
    h = '+ 0.5' if const else ''
    if named:
        for n, v in (('a', 1.0), ('b', 0.8), ('c', 0.3), ('d', 0.6), ('h', 0.5)):
            s = s.parameter(n, default=v)
        dx = '-a*x + b*y' + ('+ h' if const else '')
        dy = '-c*x - d*y' + (' - 0.1*y**3' if cubic else '')
    elif ints:
        dx, dy = '-2*x + y' + h, '-x - 3*y' + (' - y**3' if cubic else '')
    else:
        dx = '-1.0*x + 0.8*y' + h
        dy = '-0.3*x - 0.6*y' + (' - 0.1*y**3' if cubic else '')
    return (s.field('x').field('y').drift('x', dx).drift('y', dy)
            .noise('xi', Gaussian(var='0.4'), couples={'x': 1})
            .noise('eta', Gaussian(var='0.1'), couples={'y': 1}).build())


def _same_run(a, b):
    assert a['mf_values'] == b['mf_values']
    assert set(a['C_tau_by_ell']) == set(b['C_tau_by_ell'])
    for ell in a['C_tau_by_ell']:
        assert np.array_equal(np.asarray(a['C_tau_by_ell'][ell]), np.asarray(b['C_tau_by_ell'][ell]))


# ── 1. float literals in a cross-coupled drift ───────────────────────────────────────────────

@pytest.mark.parametrize('const', [False, True])
def test_float_literal_coupled_drift_equals_the_named_form(const):
    float_model, named_model = _coupled(False, const), _coupled(True, const)
    _run(float_model, XY)                                     # discarded warm-up
    _run(named_model, XY)
    a, b = _run(float_model, XY), _run(named_model, XY)
    _same_run(a, b)
    # the cubic term gives a genuine one-loop part (the comparison is not 0 == 0)
    assert np.max(np.abs(np.asarray(a['C_tau_by_ell'][1]))) > 0


def test_float_literal_coupled_drift_saddle_is_the_linear_solve():
    mf = _run(_coupled(False, const=True, cubic=False), XY, ell=0)['mf_values']
    # -x + 0.8 y + 0.5 = 0 and -0.3 x - 0.6 y = 0  =>  x = 0.5/1.4, y = -x/2
    assert mf['xstar'][0] == pytest.approx(0.5 / 1.4, rel=1e-9)
    assert mf['ystar'][0] == pytest.approx(-0.25 / 1.4, rel=1e-9)


def test_integer_literal_coupled_drift_builds_and_has_the_right_saddle():
    mf = _run(_coupled(False, const=True, cubic=False, ints=True), XY, ell=0)['mf_values']
    # -2 x + y + 0.5 = 0 and -x - 3 y = 0  =>  x = 1.5/7, y = -x/3
    assert mf['xstar'][0] == pytest.approx(1.5 / 7, rel=1e-9)
    assert mf['ystar'][0] == pytest.approx(-0.5 / 7, rel=1e-9)


def test_parameter_free_model_with_an_inconsistent_mf_equation_still_fails():
    """The numerical fallback now runs for a parameter-free model; it must still reject a wrong
    mean-field equation (rhs shifted by 0.4 against the action's drift)."""
    sde = (SDE('bad_mf').field('x').field('y')
           .drift('x', '-1.0*x + 0.8*y + 0.5').drift('y', '-0.3*x - 0.6*y')
           .noise('xi', Gaussian(var='0.4'), couples={'x': 1})
           .noise('eta', Gaussian(var='0.1'), couples={'y': 1}))
    b = sde.to_builder()
    b._equations[0]['rhs_text'] += ' + 0.4'
    with pytest.raises(AssertionError, match='mean-field saddle does not satisfy'):
        _run(b.build(), XY, ell=0)


# ── 2. partial parameters dict ───────────────────────────────────────────────────────────────

def _ou_quad(default_b=True):
    s = SDE('ou_quad').parameter('a', default=1.0)
    s = s.parameter('b', default=0.5) if default_b else s.parameter('b')
    return (s.field('x').drift('x', '-a*x + b*x**2 + 0.1')
            .noise('xi', Gaussian(var='0.4'), couples={'x': 1}).build())


XX = [('x', 1), ('x', 1)]


@pytest.mark.parametrize('given', [{'a': 2.0}, {'b': 0.7}])
def test_partial_parameters_are_filled_from_the_defaults(given):
    m = _ou_quad()
    full = {'a': 1.0, 'b': 0.5, **given}
    _run(m, XX, ell=1, parameters=full)                       # discarded warm-up
    a = _run(m, XX, ell=1, parameters=given)
    b = _run(m, XX, ell=1, parameters=full)
    _same_run(a, b)
    c = _run(m, XX, ell=1)                                    # defaults: a different answer
    assert not np.array_equal(np.asarray(a['C_tau_by_ell'][0]), np.asarray(c['C_tau_by_ell'][0]))


def test_partial_parameters_do_not_mutate_the_callers_dict():
    given = {'a': 2.0}
    _run(_ou_quad(), XX, ell=0, parameters=given)
    assert given == {'a': 2.0}


def test_partial_parameters_missing_one_without_a_default_names_it():
    with pytest.raises(ValueError, match=r'declare no default: b\b'):
        _run(_ou_quad(default_b=False), XX, ell=0, parameters={'a': 2.0})
    # the empty-dict wording is unchanged
    with pytest.raises(ValueError, match='no ``parameters=`` given'):
        _run(_ou_quad(default_b=False), XX, ell=0, parameters={})


def test_complete_parameters_helper():
    from api._mean_field import complete_parameters
    m = _ou_quad()
    assert complete_parameters(m, {'a': 3.0}) == {'a': 3.0, 'b': 0.5}
    assert complete_parameters(m, None) == {'a': 1.0, 'b': 0.5}
    full = {'a': 3.0, 'b': 4.0}
    assert complete_parameters(m, full) == full
    with pytest.raises(ValueError, match=r'declare no default: b\b'):
        complete_parameters(_ou_quad(default_b=False), {'a': 3.0})


def test_partial_parameters_reach_the_mean_field_dae_directly():
    from api._mean_field_dae import solve_mean_field_dae
    m = _ou_quad()
    part = solve_mean_field_dae(m, {'a': 2.0})['mf_values']
    full = solve_mean_field_dae(m, {'a': 2.0, 'b': 0.5})['mf_values']
    assert part == full
    with pytest.raises(ValueError, match=r'declare no default: b\b'):
        solve_mean_field_dae(_ou_quad(default_b=False), {'a': 2.0})


# ── 3. NaN / inf lags ────────────────────────────────────────────────────────────────────────

def _ou():
    return (SDE('ou_nan').parameter('a', default=1.0).field('x').drift('x', '-a*x')
            .noise('xi', Gaussian(var='0.4'), couples={'x': 1}).build())


@pytest.fixture(params=['statespace', 'poles'])
def engine(request, monkeypatch):
    if request.param == 'poles':
        monkeypatch.setenv('DAEDALUS_PHASE_J_ENGINE', 'poles')
    else:
        monkeypatch.delenv('DAEDALUS_PHASE_J_ENGINE', raising=False)
    return request.param


@pytest.mark.parametrize('bad', [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize('pos', [0, 1, 2])
def test_non_finite_tau_grid_raises_up_front(engine, bad, pos):
    grid = np.array([0.0, 0.5, 1.0])
    grid[pos] = bad
    with pytest.raises(ValueError, match='tau_grid must be finite'):
        _run(_ou(), XX, ell=0, tau_grid=grid)


def test_non_finite_tau_grid_list_and_2d_raise(engine):
    with pytest.raises(ValueError, match='tau_grid must be finite'):
        _run(_ou(), XX, ell=0, tau_grid=[0.0, float('nan')])
    with pytest.raises(ValueError, match='tau_grid must be finite'):
        _run(_ou(), XX, ell=0, tau_grid=np.array([[0.0, 1.0], [np.nan, 2.0]]))


@pytest.mark.parametrize('kw', [dict(tau_max=float('nan')), dict(tau_max=float('inf')),
                                dict(tau_step=float('nan')), dict(tau_step=float('inf'))])
def test_non_finite_tau_extent_raises(kw):
    with pytest.raises(ValueError, match='finite tau_step'):
        AC.compute_cumulants(_ou(), k=2, max_ell=0, external_fields=XX, tau_grid=None,
                             parallel=False, verbose=False, **{'tau_step': 0.5, 'tau_max': 1.0, **kw})


@pytest.mark.parametrize('bad', [np.nan, np.inf])
def test_non_finite_switch_on_times_raise(bad):
    with pytest.raises(ValueError, match='finite'):
        _run(_ou(), XX, ell=0, tau_grid=None, t_min=0.0, times=np.array([[1.0, bad]]))


def test_a_finite_tau_grid_is_untouched():
    m = _ou()
    _run(m, XX, ell=0)
    r = _run(m, XX, ell=0, tau_grid=np.array([0.0, 0.5, 1.0]))
    # OU: C(tau) = D/a exp(-a|tau|) with var 0.4 -> 0.2 exp(-|tau|) (Ito-left at tau = 0)
    assert np.real(np.asarray(r['C_tau']))[1] == pytest.approx(0.2 * np.exp(-0.5), rel=1e-6)


def test_config_rejects_non_finite_grids():
    import daedalus as dd
    with pytest.raises(ValueError, match='tau_grid must be finite'):
        dd.Config(tau_grid=np.array([0.0, np.nan]))
    with pytest.raises(ValueError, match='tau_grid must be finite'):
        dd.Config(tau_grid=(0.0, np.inf, 5))
    with pytest.raises(ValueError, match='times must be finite'):
        dd.Config(t_min=0.0, times=[1.0, np.nan])
    with pytest.raises(ValueError, match='chi_grid must be finite'):
        dd.Config(chi_grid=np.array([np.nan]))
    assert dd.Config(tau_grid=(0.0, 1.0, 5)).resolved_tau_grid().size == 5
