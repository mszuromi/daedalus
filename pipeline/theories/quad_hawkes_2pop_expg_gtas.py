"""
Quadratic Hawkes, 2 populations, exp synaptic kernel, with GTaS
external correlated noise.

  phi(v) = a · v²
  g(t) = (1/tau_g) · exp(-t/tau_g) · Theta(t)
  GTaS external as in linear_hawkes_2pop_expg_gtas.

Equivalent to ``models/hawkes_quad_expg_gtas.py``.
"""
from pipeline.theory import TheoryBuilder
from pipeline.theory_templates import (
    HawkesAction, ExpSynapticKernel, GTaSNoise,
)


_t = (
    TheoryBuilder(
        'Quadratic Hawkes 2-pop + GTaS external (phi = a·v², exp filter τ_g, '
        'Bernoulli + Gaussian shifts)',
        n_populations=2,
    )
    .response_field('nt', indexed=True, latex=r'\tilde{n}')
    .response_field('vt', indexed=True, latex=r'\tilde{v}')
    .physical_field('dn', natural_name='n', indexed=True, latex=r'\delta\dot{n}')
    .physical_field('dv', natural_name='v', indexed=True, latex=r'\delta v')
    .parameter('nstar', mean_field=True, natural_name='n', indexed=True, domain='positive',
               description='background firing rate  n*_i = a · (v*_i)²')
    .parameter('vstar', mean_field=True, natural_name='v', indexed=True, description='background voltage')
    .parameter('E',     indexed=True)
    .parameter('tau',   default=10.0, domain='positive')
    .parameter('a',     default=0.44,
               description='quadratic gain: phi(v) = a · v²')
    .parameter('tau_g', default=2.5, domain='positive')
    .parameter('w',     indexed='matrix')
    .kernel('g', sage_name='z_g', latex_name='g')
    .use_action_template(HawkesAction(phi='quadratic'))
    .use_synaptic_kernel(ExpSynapticKernel(timescale_param='tau_g'))
    .add_gtas_noise(GTaSNoise())
)

HAWKES_MODEL = _t.build()
