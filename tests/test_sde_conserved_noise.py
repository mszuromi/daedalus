"""Conserved noise (``div`` of a white vector noise) is refused, with the reason, in every form it can be written.

The spatial integrator takes momentum-independent noise only (one constant noise vertex per correlation line), and the
Model B dynamics conserved noise belongs to also need the ``k^4`` propagator and the ``lap(u^3)`` vertex, which the
spatial pipeline refuses.  Each entry point names that reason (``daedalus.engine.integration.spatial.CONSERVED_NOISE_WHY``):
``Gaussian(space='conserved')`` (also from a YAML ``sde:`` block), an operator in a noise coefficient, ``div(eta)``, an
operator on a response field in ``extra_action``, ``lap(dW)`` in a paper-style equation, and a hand-written action whose
response-field quadratic term holds a Laplacian (the engine's noise extractor, every spatial driver).
"""
import os
import warnings

import pytest

pytest.importorskip('sage.all')

from sage.all import SR                                                 # noqa: E402

from daedalus.api import compute_cumulants                                      # noqa: E402
from daedalus.api.config_run import ConfigError                                 # noqa: E402
from daedalus.api.noise import Gaussian, NoiseError                             # noqa: E402
from daedalus.api.operators import Laplacian                                    # noqa: E402
from daedalus.api.sde import SDE, SDEError                                      # noqa: E402
from daedalus.engine.errors import DaedalusError                                # noqa: E402
from daedalus.engine.integration.spatial import CONSERVED_NOISE_WHY            # noqa: E402
from daedalus.engine.integration.spatial.heat_kernel import SpatialPropagatorError   # noqa: E402
from daedalus.engine.integration.spatial.spatial_correlator import _noise_value      # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RHS = '-mu*phi + D*lap(phi)'
WHY = 'conserved noise (the divergence of a white vector noise'


def _base():
    return (SDE('conserved', space=1).operator('lap', Laplacian())
            .parameter('mu', default=1.0, domain='positive').parameter('D', default=1.0, domain='positive')
            .parameter('T', default=1.0, domain='positive').field('phi'))


def _white():
    return Gaussian(var='2*T', space='white')


def test_the_reason_names_the_three_missing_pieces():
    assert 'momentum-independent (white) noise only' in CONSERVED_NOISE_WHY
    assert 'k^4 propagator' in CONSERVED_NOISE_WHY and 'lap(u^3)' in CONSERVED_NOISE_WHY


def test_gaussian_space_conserved_is_refused_with_the_reason():
    with pytest.raises(NoiseError) as e:
        Gaussian(var='2*T', space='conserved')
    assert CONSERVED_NOISE_WHY in str(e.value) and "space='white'" in str(e.value)
    assert isinstance(e.value, DaedalusError)
    with pytest.raises(NoiseError, match="use space='white'"):          # any other space: unchanged
        Gaussian(var=1, space='colored')


@pytest.mark.parametrize('form, build', [
    ('operator in a coefficient',
     lambda: _base().equation('phi', 'Dt(phi)', RHS).noise('eta', _white(), into={'phi': 'lap'}).build()),
    ('operator applied in a coefficient',
     lambda: _base().equation('phi', 'Dt(phi)', RHS).noise('eta', _white(), into={'phi': 'lap(1)'}).build()),
    ('div(eta)',
     lambda: _base().equation('phi', 'Dt(phi)', RHS + ' + div(eta)').noise('eta', _white(), into={'phi': 1})
     .build()),
    ('operator on a response field',
     lambda: _base().equation('phi', 'Dt(phi)', RHS).noise('eta', _white(), into={'phi': 1})
     .extra_action('T*phit*lap(phit)').build()),
    ('lap(dW) in a paper-style equation',
     lambda: _base().sde('dphi = (-mu*phi + D*lap(phi)) dt + sqrt(2*T) lap(dW)')),
])
def test_every_front_end_form_is_refused_with_the_reason(form, build):
    with pytest.raises(SDEError) as e:
        build()
    assert WHY in str(e.value), form


def test_div_in_a_temporal_model_keeps_the_plain_hint():
    """``div`` is an ordinary undeclared name without space (no conserved-noise hint)."""
    with pytest.raises(SDEError) as e:
        (SDE('t').parameter('mu', default=1.0).field('x').drift('x', '-mu*x + div(x)')
         .noise('xi', Gaussian(var=1), into={'x': 1}).build())
    assert WHY not in str(e.value) and 'define_function' in str(e.value)


def test_operator_in_a_temporal_noise_coefficient_keeps_the_plain_hint():
    with pytest.raises(SDEError) as e:
        (SDE('t').operator('D_t', __import__('daedalus.api.operators', fromlist=['TimeDerivative']).TimeDerivative())
         .field('x').equation('x', 'D_t(x)', '-x').noise('xi', Gaussian(var=1), into={'x': 'D_t'}).build())
    assert WHY not in str(e.value) and "move the operator to the equation's lhs/rhs" in str(e.value)


def test_yaml_block_refuses_on_the_noise_line():
    text = ('sde:\n  name: c\n  space: 1\n  operators: {lap: Laplacian}\n  parameters: {T: 1.0}\n'
            '  fields: [phi]\n  equations: {phi: {lhs: "Dt(phi)", rhs: "-phi + lap(phi)"}}\n  noise:\n'
            '    eta: {type: Gaussian, var: 2*T, space: conserved, into: {phi: 1}}\n')
    with pytest.raises(ConfigError) as e:
        SDE.from_text(text)
    assert e.value.line == 9 and CONSERVED_NOISE_WHY in str(e.value)


def test_noise_extractor_names_a_momentum_dependent_noise():
    with pytest.raises(SpatialPropagatorError) as e:
        _noise_value(SR('T*Laplacian'), {SR.var('T'): 1}, (2, 0))
    assert 'depends on the momentum' in str(e.value) and CONSERVED_NOISE_WHY in str(e.value)
    with pytest.raises(SpatialPropagatorError) as e:                   # a missing value: the old message
        _noise_value(SR('T*c'), {SR.var('T'): 1}, (2, 0))
    assert 'Supply every parameter' in str(e.value) and WHY not in str(e.value)
    assert _noise_value(SR('-T'), {SR.var('T'): 2}, (2, 0)) == -2      # white noise: unchanged


def _hand_written_conserved():
    """The tracked Allen-Cahn model with its noise term ``-T phit^2`` replaced by ``T phit Laplacian phit``."""
    src = open(os.path.join(ROOT, 'daedalus', 'models', 'allen_cahn_1d_subcritical_infinite.model.py')).read()
    new = src.replace("lam*phi^3) - T*phit^2'", "lam*phi^3) + T*phit*Laplacian*phit'")
    assert new != src
    ns = {'__name__': 'conserved_hand_written'}
    exec(compile(new, 'conserved_hand_written', 'exec'), ns)
    return ns['build']()


@pytest.mark.parametrize('k, ell, kw', [
    (2, 0, dict(tau_grid=[0.0], spatial_grid=[0.0])),
    (2, 1, dict(tau_grid=[0.0], spatial_grid=[0.0])),
    (3, 0, dict(spatial_points=[[(0.3, 0.0), (0.0, 0.2)]])),
])
def test_hand_written_conserved_noise_is_refused_by_every_spatial_driver(k, ell, kw):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        with pytest.raises(SpatialPropagatorError) as e:
            compute_cumulants(_hand_written_conserved(), k=k, max_ell=ell, external_fields=['phi'] * k,
                              verbose=False, **kw)
    assert CONSERVED_NOISE_WHY in str(e.value)
