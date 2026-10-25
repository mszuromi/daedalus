"""Audit wave 9: one exception base (CON-04) with every existing class kept."""
from __future__ import annotations

import importlib
import inspect
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import daedalus as dd                               # noqa: E402
from engine.errors import DaedalusError             # noqa: E402

# class -> the builtin it has always been (existing except-clauses keep working)
_KEPT = {
    'api._realization.RealizationRefused': ValueError,
    'api.symbolic_out.SymbolicOutError': ValueError,
    'api.symbolic_out.NonRationalPropagatorError': ValueError,
    'api.symbolic_out.ClosedFormUnavailable': ValueError,
    'api.symbolic_out.AssumptionViolation': ValueError,
    'api.config_run.ConfigError': ValueError,
    'api._delay_exact.ExactTreeUnavailable': ValueError,
    'api.sde.SDEError': ValueError,
    'api.pp_simulate.PPSimulationError': ValueError,
    'api.noise.NoiseError': ValueError,
    'api.me_output.MEUnavailable': ValueError,
    'api.symbolic_loop.SymbolicLoopRefused': ValueError,
    'api.routing.NoBackendError': Exception,
    'api.analyze.AnalyzerError': ValueError,
    'api._mean_field_dae.UnstableSaddleError': ValueError,
    'api.compute.SpatialAccuracyError': RuntimeError,
    'api.covariance.CovarianceDerivativeError': RuntimeError,
    'api.compute.PhaseJIncompleteError': RuntimeError,
    'engine.integration.time_domain.pipeline.DiscardedDiagramsError': RuntimeError,
    'engine.core.dt_order.NonlinearDtError': ValueError,
    'engine.core.field_theory.DerivativeAtomTaylorError': Exception,
    'engine.integration.time_domain.expdd.DivergentIntegralError': ValueError,
    'engine.integration.spatial.heat_kernel.SpatialPropagatorError': Exception,
    'engine.integration.time_domain.statespace.SSIneligible': Exception,
    'engine.integration.time_domain.statespace.SSLateRefusal': RuntimeError,
    'engine.integration.time_domain.statespace.SSNoPoleFallback': RuntimeError,
    'engine.integration.time_domain.statespace.SwitchOnUnsupported': ValueError,
    'engine.integration.spatial.full_integrator.SpatialMemoryBudgetError': MemoryError,
    'engine.integration.spatial.full_integrator.SpatialUVDivergenceError': NotImplementedError,
    'engine.integration.time_domain.final_integral.PoleFreePropagatorError': ValueError,
    'engine.integration.time_domain.final_integral.PhaseJResourceError': MemoryError,
    'engine.integration.time_domain.final_integral.PhaseJDivergenceError': ArithmeticError,
    'engine.integration.time_domain.final_integral.DeltaSolveValidationError': AssertionError,
}


@pytest.mark.parametrize('path, builtin', sorted(_KEPT.items()))
def test_every_error_class_is_a_daedalus_error_and_keeps_its_builtin(path, builtin):
    mod, name = path.rsplit('.', 1)
    cls = getattr(importlib.import_module(mod), name)
    assert issubclass(cls, DaedalusError), path
    assert issubclass(cls, builtin), path


def test_no_public_exception_class_was_left_out():
    """Any exception class defined in api/ or engine/ (a public name) is
    covered: a new one must join the base, not forget it."""
    import pkgutil
    seen = set()
    for pkg in ('api', 'engine'):
        root = importlib.import_module(pkg)
        for info in pkgutil.walk_packages(root.__path__, pkg + '.'):
            if info.name.startswith(('api.examples', 'api.tests', 'api.ui',
                                     'engine.enumeration._fast')):
                continue
            try:
                mod = importlib.import_module(info.name)
            except Exception:
                continue
            for n, obj in vars(mod).items():
                if (inspect.isclass(obj) and obj.__module__ == info.name
                        and issubclass(obj, Exception)
                        and not issubclass(obj, Warning)
                        and not n.startswith('_')
                        and obj is not DaedalusError):
                    seen.add((info.name, n, issubclass(obj, DaedalusError)))
    missing = sorted(f'{m}.{n}' for m, n, ok in seen if not ok)
    assert not missing, missing


def test_catching_the_base_catches_a_real_refusal_and_not_a_bug():
    from api.noise import Gaussian, NoiseError
    with pytest.raises(DaedalusError):
        Gaussian(var='-1')
    assert dd.DaedalusError is DaedalusError
    import api
    assert api.DaedalusError is DaedalusError
    try:
        {}['missing']
    except DaedalusError:                       # pragma: no cover
        pytest.fail('a KeyError must not be a DaedalusError')
    except KeyError:
        pass
    with pytest.raises(ValueError):             # the old clause still catches it
        raise NoiseError('x')


def test_population_sum_message_no_longer_blames_a_noise_parameter():
    from api.noise import parse_expr, NoiseError
    with pytest.raises(NoiseError) as ei:
        parse_expr('-mu*x[i] + sum(w[i,j]*x[j] for j in E)', what="rhs of x")
    msg = str(ei.value)
    assert 'population sums' in msg and 'rhs of x' in msg
    assert 'inside a noise parameter' not in msg


def test_pairing_kernel_rename_keeps_the_old_name():
    import api.delay as D
    import api.symbolic_out as so
    assert so.Kernel is so.PairingKernel and dd.Kernel is dd.PairingKernel
    assert so.PairingKernel is not D.Kernel
    L = so.PairingKernel.two_sided_exponential(1.0, 0.7, -0.5, 1.3)
    assert isinstance(L, so.Kernel) and type(L).__name__ == 'PairingKernel'
    assert isinstance(L + L, so.PairingKernel)
