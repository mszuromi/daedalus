"""Audit wave 9: one exception base (CON-04) with every existing class kept."""
from __future__ import annotations

import importlib
import inspect
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import daedalus as dd                               # noqa: E402
from daedalus.engine.errors import DaedalusError             # noqa: E402

# class -> the builtin it has always been (existing except-clauses keep working)
_KEPT = {
    'daedalus.api._realization.RealizationRefused': ValueError,
    'daedalus.api.symbolic_out.SymbolicOutError': ValueError,
    'daedalus.api.symbolic_out.NonRationalPropagatorError': ValueError,
    'daedalus.api.symbolic_out.ClosedFormUnavailable': ValueError,
    'daedalus.api.symbolic_out.AssumptionViolation': ValueError,
    'daedalus.api.config_run.ConfigError': ValueError,
    'daedalus.api._delay_exact.ExactTreeUnavailable': ValueError,
    'daedalus.api.sde.SDEError': ValueError,
    'daedalus.api.pp_simulate.PPSimulationError': ValueError,
    'daedalus.api.noise.NoiseError': ValueError,
    'daedalus.api.observables.ObservableError': ValueError,
    'daedalus.api.me_output.MEUnavailable': ValueError,
    'daedalus.api.symbolic_loop.SymbolicLoopRefused': ValueError,
    'daedalus.api.routing.NoBackendError': Exception,
    'daedalus.api.analyze.AnalyzerError': ValueError,
    'daedalus.api._mean_field_dae.UnstableSaddleError': ValueError,
    'daedalus.api.compute.SpatialAccuracyError': RuntimeError,
    'daedalus.api.covariance.CovarianceDerivativeError': RuntimeError,
    'daedalus.api.compute.PhaseJIncompleteError': RuntimeError,
    'daedalus.engine.integration.time_domain.pipeline.DiscardedDiagramsError': RuntimeError,
    'daedalus.engine.core.dt_order.NonlinearDtError': ValueError,
    'daedalus.engine.core.field_theory.DerivativeAtomTaylorError': Exception,
    'daedalus.engine.integration.time_domain.expdd.DivergentIntegralError': ValueError,
    'daedalus.engine.integration.spatial.heat_kernel.SpatialPropagatorError': Exception,
    'daedalus.engine.integration.time_domain.statespace.SSIneligible': Exception,
    'daedalus.engine.integration.time_domain.statespace.SSLateRefusal': RuntimeError,
    'daedalus.engine.integration.time_domain.statespace.SSNoPoleFallback': RuntimeError,
    'daedalus.engine.integration.time_domain.statespace.SwitchOnUnsupported': ValueError,
    'daedalus.engine.integration.spatial.full_integrator.SpatialMemoryBudgetError': MemoryError,
    'daedalus.engine.integration.spatial.full_integrator.SpatialUVDivergenceError': NotImplementedError,
    'daedalus.engine.integration.time_domain.final_integral.PoleFreePropagatorError': ValueError,
    'daedalus.engine.integration.time_domain.final_integral.PhaseJResourceError': MemoryError,
    'daedalus.engine.integration.time_domain.final_integral.PhaseJDivergenceError': ArithmeticError,
    'daedalus.engine.integration.time_domain.final_integral.DeltaSolveValidationError': AssertionError,
}


@pytest.mark.parametrize('path, builtin', sorted(_KEPT.items()))
def test_every_error_class_is_a_daedalus_error_and_keeps_its_builtin(path, builtin):
    mod, name = path.rsplit('.', 1)
    cls = getattr(importlib.import_module(mod), name)
    assert issubclass(cls, DaedalusError), path
    assert issubclass(cls, builtin), path


def test_no_public_exception_class_was_left_out():
    """Any exception class defined in daedalus/api/ or daedalus/engine/ (a public name) is
    covered: a new one must join the base, not forget it."""
    import pkgutil
    seen = set()
    for pkg in ('daedalus.api', 'daedalus.engine'):
        root = importlib.import_module(pkg)
        for info in pkgutil.walk_packages(root.__path__, pkg + '.'):
            if info.name.startswith(('daedalus.api.examples', 'daedalus.api.tests', 'daedalus.api.ui',
                                     'daedalus.engine.enumeration._fast')):
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
    assert len(seen) > 20, f'only {len(seen)} exception classes found: has the package moved?'
    missing = sorted(f'{m}.{n}' for m, n, ok in seen if not ok)
    assert not missing, missing


def test_catching_the_base_catches_a_real_refusal_and_not_a_bug():
    from daedalus.api.noise import Gaussian, NoiseError
    with pytest.raises(DaedalusError):
        Gaussian(var='-1')
    assert dd.DaedalusError is DaedalusError
    import daedalus.api as api
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
    from daedalus.api.noise import parse_expr, NoiseError
    with pytest.raises(NoiseError) as ei:
        parse_expr('-mu*x[i] + sum(w[i,j]*x[j] for j in E)', what="rhs of x")
    msg = str(ei.value)
    assert 'population sums' in msg and 'rhs of x' in msg
    assert 'inside a noise parameter' not in msg


def test_kernel_is_the_shape_builder_and_pairing_kernel_has_its_own_name():
    import daedalus.api.delay as D
    import daedalus.api.symbolic_out as so
    assert dd.Kernel is D.Kernel and dd.PairingKernel is so.PairingKernel
    assert dd.Kernel is not so.PairingKernel and not hasattr(so, 'Kernel')   # the old alias is gone
    L = so.PairingKernel.two_sided_exponential(1.0, 0.7, -0.5, 1.3)
    assert isinstance(L, so.PairingKernel) and type(L).__name__ == 'PairingKernel'
    assert isinstance(L + L, so.PairingKernel)
