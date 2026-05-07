"""
Linear Hawkes, 2 populations, exponential synaptic kernel.

  phi(v) = a · v
  g(t) = (1/tau_g) · exp(-t/tau_g) · Theta(t)
  no external GTaS noise

Equivalent to the hand-written ``models/hawkes_linear_expg.py``.
"""
from pipeline.theory import TheoryBuilder
from pipeline.theory_templates import HawkesAction, ExpSynapticKernel


_t = (
    TheoryBuilder('Linear Hawkes 2-population (phi = a·v, exp filter τ_g)',
                  n_populations=2)
    .response_field('nt', indexed=True, latex=r'\tilde{n}',
                    description='response field conjugate to spike train')
    .response_field('vt', indexed=True, latex=r'\tilde{v}',
                    description='response field conjugate to voltage')
    .physical_field('dn', indexed=True, latex=r'\delta\dot{n}',
                    description='spike-train fluctuation')
    .physical_field('dv', indexed=True, latex=r'\delta v',
                    description='voltage fluctuation')
    .parameter('nstar', indexed=True, domain='positive',
               description='background firing rate  n*_i = a · v*_i')
    .parameter('vstar', indexed=True,
               description='background voltage')
    .parameter('E',     indexed=True,
               description='external constant drive')
    .parameter('tau',   default=10.0, domain='positive',
               description='membrane time constant')
    .parameter('a',     default=1.0,
               description='linear transfer-function gain: phi(v) = a·v')
    .parameter('tau_g', default=2.5, domain='positive',
               description='synaptic exponential filter timescale')
    .parameter('w',     indexed='matrix',
               description='synaptic weight matrix w_{ij}')
    .kernel('g', sage_name='z_g', latex_name='g',
            description='exponential synaptic filter, unit integral')
    .use_action_template(HawkesAction(phi='linear'))
    .use_synaptic_kernel(ExpSynapticKernel(timescale_param='tau_g'))
)

HAWKES_MODEL = _t.build()
